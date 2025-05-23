from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import csv
import io

from app.core.auth import get_current_user_flexible
from app.db.database import get_db
from app.db.models import Call, Transcript, Recording, User, Assistant
from app.services.call_service import CallService
from app.api.schemas import CallResponse, TranscriptResponse, RecordingResponse, APIResponse

router = APIRouter(prefix="/api/v1/calls", tags=["calls"])


@router.get("/", response_model=List[CallResponse])
async def get_calls(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of items to return"),
    status: Optional[str] = Query(None, description="Filter by call status"),
    assistant_id: Optional[int] = Query(None, description="Filter by assistant ID"),
    customer_phone: Optional[str] = Query(None, description="Filter by customer phone number"),
    date_from: Optional[datetime] = Query(None, description="Filter calls from this date (ISO format)"),
    date_to: Optional[datetime] = Query(None, description="Filter calls to this date (ISO format)"),
    min_duration: Optional[int] = Query(None, description="Minimum call duration in seconds"),
    max_duration: Optional[int] = Query(None, description="Maximum call duration in seconds"),
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db),
):
    """
    Get a list of calls for your organization.
    
    Returns all calls that belong to assistants in your organization,
    with comprehensive filtering options.
    """
    # Build query with organization filter
    query = (
        db.query(Call)
        .join(Assistant)
        .filter(Assistant.organization_id == current_user.organization_id)
    )

    # Apply filters
    if status:
        query = query.filter(Call.status == status)

    if assistant_id:
        # Verify the assistant belongs to the user's organization
        assistant = db.query(Assistant).filter(
            Assistant.id == assistant_id,
            Assistant.organization_id == current_user.organization_id
        ).first()
        if not assistant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Assistant with ID {assistant_id} not found in your organization",
            )
        query = query.filter(Call.assistant_id == assistant_id)

    if customer_phone:
        query = query.filter(Call.customer_phone_number.like(f"%{customer_phone}%"))

    if date_from:
        query = query.filter(Call.started_at >= date_from)

    if date_to:
        query = query.filter(Call.started_at <= date_to)

    if min_duration is not None:
        query = query.filter(Call.duration >= min_duration)

    if max_duration is not None:
        query = query.filter(Call.duration <= max_duration)

    # Order by start time, newest first
    query = query.order_by(Call.started_at.desc())

    calls = query.offset(skip).limit(limit).all()
    return calls


@router.get("/export", response_class=Response)
async def export_calls(
    format: str = Query("csv", regex="^(csv|json)$", description="Export format: csv or json"),
    status: Optional[str] = Query(None, description="Filter by call status"),
    assistant_id: Optional[int] = Query(None, description="Filter by assistant ID"),
    date_from: Optional[datetime] = Query(None, description="Filter calls from this date"),
    date_to: Optional[datetime] = Query(None, description="Filter calls to this date"),
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db),
):
    """
    Export calls data in CSV or JSON format.
    
    Returns call data with the specified filters applied.
    """
    # Build query with same filters as get_calls
    query = (
        db.query(Call, Assistant.name.label('assistant_name'))
        .join(Assistant)
        .filter(Assistant.organization_id == current_user.organization_id)
    )

    # Apply filters
    if status:
        query = query.filter(Call.status == status)
    if assistant_id:
        query = query.filter(Call.assistant_id == assistant_id)
    if date_from:
        query = query.filter(Call.started_at >= date_from)
    if date_to:
        query = query.filter(Call.started_at <= date_to)

    # Get results
    results = query.order_by(Call.started_at.desc()).all()

    if format == "csv":
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "Call ID", "Call SID", "Assistant Name", "Customer Phone", 
            "Status", "Duration (seconds)", "Started At", "Ended At"
        ])
        
        # Write data
        for call, assistant_name in results:
            writer.writerow([
                call.id,
                call.call_sid,
                assistant_name,
                call.customer_phone_number,
                call.status,
                call.duration or 0,
                call.started_at.isoformat() if call.started_at else "",
                call.ended_at.isoformat() if call.ended_at else ""
            ])
        
        csv_content = output.getvalue()
        output.close()
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=calls_export.csv"}
        )
    
    else:  # JSON format
        data = []
        for call, assistant_name in results:
            data.append({
                "id": call.id,
                "call_sid": call.call_sid,
                "assistant_name": assistant_name,
                "customer_phone_number": call.customer_phone_number,
                "status": call.status,
                "duration": call.duration,
                "started_at": call.started_at.isoformat() if call.started_at else None,
                "ended_at": call.ended_at.isoformat() if call.ended_at else None,
            })
        
        import json
        json_content = json.dumps(data, indent=2)
        
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=calls_export.json"}
        )


@router.get("/analytics", response_model=dict)
async def get_call_analytics(
    period: str = Query("7d", regex="^(1d|7d|30d|90d)$", description="Analysis period"),
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db)
):
    """
    Get detailed call analytics for your organization.
    
    Returns comprehensive metrics and trends for the specified period.
    """
    # Calculate date range
    period_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}
    days = period_days[period]
    date_from = datetime.utcnow() - timedelta(days=days)
    
    # Base query for organization's calls in the period
    base_query = (
        db.query(Call)
        .join(Assistant)
        .filter(
            Assistant.organization_id == current_user.organization_id,
            Call.started_at >= date_from
        )
    )
    
    # Get all calls in period
    calls = base_query.all()
    
    # Calculate metrics
    total_calls = len(calls)
    completed_calls = [c for c in calls if c.status == "completed"]
    failed_calls = [c for c in calls if c.status == "failed"]
    ongoing_calls = [c for c in calls if c.status == "ongoing"]
    
    # Duration statistics
    durations = [c.duration for c in completed_calls if c.duration]
    total_duration = sum(durations) if durations else 0
    avg_duration = total_duration / len(durations) if durations else 0
    
    # Success rate
    success_rate = (len(completed_calls) / total_calls * 100) if total_calls > 0 else 0
    
    # Call volume by day
    daily_stats = {}
    for call in calls:
        day = call.started_at.date().isoformat()
        if day not in daily_stats:
            daily_stats[day] = {"total": 0, "completed": 0, "failed": 0}
        daily_stats[day]["total"] += 1
        daily_stats[day][call.status] = daily_stats[day].get(call.status, 0) + 1
    
    # Top assistants by call volume
    assistant_stats = {}
    for call in calls:
        assistant_id = call.assistant_id
        if assistant_id not in assistant_stats:
            assistant_stats[assistant_id] = {
                "calls": 0,
                "duration": 0,
                "assistant_name": call.assistant.name if call.assistant else "Unknown"
            }
        assistant_stats[assistant_id]["calls"] += 1
        if call.duration:
            assistant_stats[assistant_id]["duration"] += call.duration
    
    # Convert to sorted list
    top_assistants = sorted(
        [{"assistant_id": k, **v} for k, v in assistant_stats.items()],
        key=lambda x: x["calls"],
        reverse=True
    )[:10]
    
    return {
        "period": period,
        "date_range": {
            "from": date_from.isoformat(),
            "to": datetime.utcnow().isoformat()
        },
        "summary": {
            "total_calls": total_calls,
            "completed_calls": len(completed_calls),
            "failed_calls": len(failed_calls),
            "ongoing_calls": len(ongoing_calls),
            "success_rate": round(success_rate, 2),
            "total_duration_seconds": total_duration,
            "average_duration_seconds": round(avg_duration, 2)
        },
        "daily_statistics": daily_stats,
        "top_assistants": top_assistants
    }


@router.get("/count", response_model=dict)
async def get_calls_count(
    status: Optional[str] = Query(None, description="Filter by call status"),
    assistant_id: Optional[int] = Query(None, description="Filter by assistant ID"),
    date_from: Optional[datetime] = Query(None, description="Filter calls from this date"),
    date_to: Optional[datetime] = Query(None, description="Filter calls to this date"),
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db),
):
    """
    Get the count of calls in your organization with filtering options.
    """
    # Build query with organization filter
    query = (
        db.query(Call)
        .join(Assistant)
        .filter(Assistant.organization_id == current_user.organization_id)
    )

    # Apply filters
    if status:
        query = query.filter(Call.status == status)

    if assistant_id:
        # Verify the assistant belongs to the user's organization
        assistant = db.query(Assistant).filter(
            Assistant.id == assistant_id,
            Assistant.organization_id == current_user.organization_id
        ).first()
        if not assistant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Assistant with ID {assistant_id} not found in your organization",
            )
        query = query.filter(Call.assistant_id == assistant_id)

    if date_from:
        query = query.filter(Call.started_at >= date_from)

    if date_to:
        query = query.filter(Call.started_at <= date_to)

    count = query.count()
    
    # Also get breakdown by status
    status_breakdown = {}
    for call_status in ["ongoing", "completed", "failed"]:
        status_query = query.filter(Call.status == call_status) if not status else query
        status_breakdown[call_status] = status_query.filter(Call.status == call_status).count() if not status else (count if status == call_status else 0)

    return {
        "total_count": count,
        "status_breakdown": status_breakdown
    }


@router.get("/{call_id}", response_model=CallResponse)
async def get_call(
    call_id: int, 
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db)
):
    """
    Get a specific call by ID.
    
    Returns the call details if it belongs to an assistant in your organization.
    """
    call = (
        db.query(Call)
        .join(Assistant)
        .filter(
            Call.id == call_id,
            Assistant.organization_id == current_user.organization_id
        )
        .first()
    )
    
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with ID {call_id} not found in your organization",
        )
    return call


@router.get("/sid/{call_sid}", response_model=CallResponse)
async def get_call_by_sid(
    call_sid: str, 
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db)
):
    """
    Get a specific call by Twilio Call SID.
    
    Returns the call details if it belongs to an assistant in your organization.
    """
    call = (
        db.query(Call)
        .join(Assistant)
        .filter(
            Call.call_sid == call_sid,
            Assistant.organization_id == current_user.organization_id
        )
        .first()
    )
    
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with SID {call_sid} not found in your organization",
        )
    return call


@router.patch("/{call_id}/metadata", response_model=CallResponse)
async def update_call_metadata(
    call_id: int,
    metadata: dict,
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db)
):
    """
    Update call metadata.
    
    Allows updating custom metadata for a call.
    """
    call = (
        db.query(Call)
        .join(Assistant)
        .filter(
            Call.id == call_id,
            Assistant.organization_id == current_user.organization_id
        )
        .first()
    )
    
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with ID {call_id} not found in your organization",
        )
    
    # Update metadata
    if call.call_meta is None:
        call.call_meta = {}
    
    call.call_meta.update(metadata)
    db.commit()
    db.refresh(call)
    
    return call


@router.get("/{call_id}/transcripts", response_model=List[TranscriptResponse])
async def get_call_transcripts(
    call_id: int, 
    speaker: Optional[str] = Query(None, description="Filter by speaker (user/assistant)"),
    include_interim: bool = Query(False, description="Include interim (non-final) transcripts"),
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db)
):
    """
    Get all transcripts for a call.
    
    Returns transcripts if the call belongs to an assistant in your organization.
    """
    # Check if call exists and belongs to the organization
    call = (
        db.query(Call)
        .join(Assistant)
        .filter(
            Call.id == call_id,
            Assistant.organization_id == current_user.organization_id
        )
        .first()
    )
    
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with ID {call_id} not found in your organization",
        )

    # Get transcripts for the call
    query = db.query(Transcript).filter(Transcript.call_id == call_id)

    if speaker:
        query = query.filter(Transcript.speaker == speaker)

    if not include_interim:
        query = query.filter(Transcript.is_final == True)

    # Order by creation time
    query = query.order_by(Transcript.created_at)

    transcripts = query.all()
    return transcripts


@router.get("/{call_id}/transcripts/export", response_class=Response)
async def export_call_transcripts(
    call_id: int,
    format: str = Query("txt", regex="^(txt|json|csv)$", description="Export format"),
    speaker: Optional[str] = Query(None, description="Filter by speaker"),
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db)
):
    """
    Export call transcripts in various formats.
    """
    # Verify call exists and belongs to organization
    call = (
        db.query(Call)
        .join(Assistant)
        .filter(
            Call.id == call_id,
            Assistant.organization_id == current_user.organization_id
        )
        .first()
    )
    
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with ID {call_id} not found in your organization",
        )

    # Get transcripts
    query = db.query(Transcript).filter(
        Transcript.call_id == call_id,
        Transcript.is_final == True
    )
    
    if speaker:
        query = query.filter(Transcript.speaker == speaker)
    
    transcripts = query.order_by(Transcript.created_at).all()

    if format == "txt":
        content = f"Call Transcript - Call ID: {call_id}\n"
        content += f"Call SID: {call.call_sid}\n"
        content += f"Started: {call.started_at}\n\n"
        
        for transcript in transcripts:
            speaker_label = transcript.speaker or "Unknown"
            timestamp = transcript.created_at.strftime("%H:%M:%S") if transcript.created_at else ""
            content += f"[{timestamp}] {speaker_label}: {transcript.content}\n"
        
        return Response(
            content=content,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=call_{call_id}_transcript.txt"}
        )
    
    elif format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Timestamp", "Speaker", "Content", "Confidence"])
        
        for transcript in transcripts:
            writer.writerow([
                transcript.created_at.isoformat() if transcript.created_at else "",
                transcript.speaker or "",
                transcript.content,
                transcript.confidence or ""
            ])
        
        csv_content = output.getvalue()
        output.close()
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=call_{call_id}_transcript.csv"}
        )
    
    else:  # JSON
        data = {
            "call_id": call_id,
            "call_sid": call.call_sid,
            "started_at": call.started_at.isoformat() if call.started_at else None,
            "transcripts": [
                {
                    "timestamp": t.created_at.isoformat() if t.created_at else None,
                    "speaker": t.speaker,
                    "content": t.content,
                    "confidence": t.confidence,
                    "segment_start": t.segment_start,
                    "segment_end": t.segment_end
                }
                for t in transcripts
            ]
        }
        
        import json
        json_content = json.dumps(data, indent=2)
        
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=call_{call_id}_transcript.json"}
        )


@router.get("/sid/{call_sid}/transcripts", response_model=List[TranscriptResponse])
async def get_call_sid_transcripts(
    call_sid: str, 
    speaker: Optional[str] = Query(None, description="Filter by speaker (user/assistant)"),
    include_interim: bool = Query(False, description="Include interim (non-final) transcripts"),
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db)
):
    """
    Get all transcripts for a call by Twilio Call SID.
    
    Returns transcripts if the call belongs to an assistant in your organization.
    """
    # Check if call exists and belongs to the organization
    call = (
        db.query(Call)
        .join(Assistant)
        .filter(
            Call.call_sid == call_sid,
            Assistant.organization_id == current_user.organization_id
        )
        .first()
    )
    
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with SID {call_sid} not found in your organization",
        )

    # Get transcripts for the call
    query = db.query(Transcript).filter(Transcript.call_id == call.id)

    if speaker:
        query = query.filter(Transcript.speaker == speaker)

    if not include_interim:
        query = query.filter(Transcript.is_final == True)

    # Order by creation time
    query = query.order_by(Transcript.created_at)

    transcripts = query.all()
    return transcripts


@router.get("/{call_id}/recordings", response_model=List[RecordingResponse])
async def get_call_recordings(
    call_id: int, 
    recording_type: Optional[str] = Query(None, description="Filter by recording type"),
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db)
):
    """
    Get all recordings for a call.
    
    Returns recordings if the call belongs to an assistant in your organization.
    """
    # Check if call exists and belongs to the organization
    call = (
        db.query(Call)
        .join(Assistant)
        .filter(
            Call.id == call_id,
            Assistant.organization_id == current_user.organization_id
        )
        .first()
    )
    
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with ID {call_id} not found in your organization",
        )

    # Get recordings for the call
    query = db.query(Recording).filter(Recording.call_id == call_id)

    if recording_type:
        query = query.filter(Recording.recording_type == recording_type)

    # Order by creation time
    query = query.order_by(Recording.created_at)

    recordings = query.all()
    return recordings


@router.get("/sid/{call_sid}/recordings", response_model=List[RecordingResponse])
async def get_call_sid_recordings(
    call_sid: str, 
    recording_type: Optional[str] = Query(None, description="Filter by recording type"),
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db)
):
    """
    Get all recordings for a call by Twilio Call SID.
    
    Returns recordings if the call belongs to an assistant in your organization.
    """
    # Check if call exists and belongs to the organization
    call = (
        db.query(Call)
        .join(Assistant)
        .filter(
            Call.call_sid == call_sid,
            Assistant.organization_id == current_user.organization_id
        )
        .first()
    )
    
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with SID {call_sid} not found in your organization",
        )

    # Get recordings for the call
    query = db.query(Recording).filter(Recording.call_id == call.id)

    if recording_type:
        query = query.filter(Recording.recording_type == recording_type)

    # Order by creation time
    query = query.order_by(Recording.created_at)

    recordings = query.all()
    return recordings


@router.get("/stats", response_model=dict)
async def get_call_stats(
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db)
):
    """
    Get call statistics for your organization.
    
    Returns various metrics about calls in your organization.
    """
    # Base query for organization's calls
    base_query = (
        db.query(Call)
        .join(Assistant)
        .filter(Assistant.organization_id == current_user.organization_id)
    )
    
    # Get various statistics
    total_calls = base_query.count()
    
    ongoing_calls = base_query.filter(Call.status == "ongoing").count()
    completed_calls = base_query.filter(Call.status == "completed").count()
    failed_calls = base_query.filter(Call.status == "failed").count()
    
    # Get total duration (only for completed calls)
    completed_calls_with_duration = base_query.filter(
        Call.status == "completed",
        Call.duration.isnot(None)
    ).all()
    
    total_duration = sum(call.duration for call in completed_calls_with_duration if call.duration)
    average_duration = total_duration / len(completed_calls_with_duration) if completed_calls_with_duration else 0
    
    # Recent activity (last 24 hours)
    recent_cutoff = datetime.utcnow() - timedelta(hours=24)
    recent_calls = base_query.filter(Call.started_at >= recent_cutoff).count()
    
    return {
        "total_calls": total_calls,
        "ongoing_calls": ongoing_calls,
        "completed_calls": completed_calls,
        "failed_calls": failed_calls,
        "total_duration_seconds": total_duration,
        "average_duration_seconds": round(average_duration, 2),
        "success_rate": round((completed_calls / total_calls * 100) if total_calls > 0 else 0, 2),
        "recent_calls_24h": recent_calls
    }


@router.get("/search", response_model=List[CallResponse])
async def search_calls(
    q: str = Query(..., min_length=3, description="Search query (minimum 3 characters)"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db)
):
    """
    Search calls by various criteria.
    
    Searches through call SIDs, customer phone numbers, and assistant names.
    """
    # Search query
    search_term = f"%{q}%"
    
    calls = (
        db.query(Call)
        .join(Assistant)
        .filter(
            Assistant.organization_id == current_user.organization_id,
            or_(
                Call.call_sid.like(search_term),
                Call.customer_phone_number.like(search_term),
                Assistant.name.like(search_term)
            )
        )
        .order_by(Call.started_at.desc())
        .limit(limit)
        .all()
    )
    
    return calls
