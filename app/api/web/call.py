"""
Call routes
"""

# pylint: disable=logging-fstring-interpolation, broad-exception-caught
import logging
import json
import os
import datetime
from io import StringIO
from fastapi import APIRouter, Depends, Request, Form, HTTPException, Response
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    FileResponse,
)
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc

from app.db.database import get_db
from app.db.models import (
    Call,
    Recording,
    Transcript,
    Assistant,
)
from app.services.assistant_service import AssistantService
from app.services.auth_service import AuthService

# Create router without a prefix - web routes will be at the root level
router = APIRouter(tags=["web"])

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Security
security = HTTPBearer(auto_error=False)
auth_service = AuthService()


def get_template_context(request: Request, **extra_context) -> dict:
    """Get template context with session data and any extra context."""
    context = {
        "request": request,
        "session": {
            "user_id": request.session.get("user_id"),
            "organization_id": request.session.get("organization_id"),
            "user_email": request.session.get("user_email", ""),
            "user_first_name": request.session.get("user_first_name", ""),
            "user_last_name": request.session.get("user_last_name", ""),
            "organization_name": request.session.get("organization_name", ""),
            "organization_slug": request.session.get("organization_slug", ""),
            "api_key_count": request.session.get("api_key_count", 0),
        },
    }
    context.update(extra_context)
    return context


@router.get("/calls/{call_id}/recording/{recording_id}")
async def download_recording(
    request: Request, call_id: int, recording_id: int, db: Session = Depends(get_db)
):
    """Download or serve recording."""
    recording = (
        db.query(Recording)
        .filter(Recording.id == recording_id, Recording.call_id == call_id)
        .first()
    )

    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    # Log debug info
    logger = logging.getLogger(__name__)
    logger.info(
        f"Recording {recording_id}: file_path={recording.file_path}, recording_source={recording.recording_source}, status={recording.status}"
    )

    # Prefer local file if available
    if recording.file_path and os.path.exists(recording.file_path):
        logger.info(f"Serving local file: {recording.file_path}")
        
        # Determine media type based on file extension
        media_type = "audio/wav"
        filename_ext = "wav"
        if recording.file_path.lower().endswith('.mp3'):
            media_type = "audio/mpeg"
            filename_ext = "mp3"
        elif recording.file_path.lower().endswith('.wav'):
            media_type = "audio/wav"
            filename_ext = "wav"
        
        return FileResponse(
            path=recording.file_path,
            media_type=media_type,
            filename=f"recording_{recording.call.call_sid}_{recording.recording_type}.{filename_ext}",
        )
    else:
        # No local file available
        logger.error(
            f"Recording file not available: file_path={recording.file_path}, recording_source={recording.recording_source}"
        )
        raise HTTPException(status_code=404, detail="Recording file not available")


@router.get("/calls/{call_id}/recording/{recording_id}/play")
async def play_recording(
    request: Request, call_id: int, recording_id: int, db: Session = Depends(get_db)
):
    """Serve recording for in-browser audio player."""
    recording = (
        db.query(Recording)
        .filter(Recording.id == recording_id, Recording.call_id == call_id)
        .first()
    )

    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    # Only serve local files for audio player
    if recording.file_path and os.path.exists(recording.file_path):
        # Determine media type based on file extension
        media_type = "audio/wav"
        if recording.file_path.lower().endswith('.mp3'):
            media_type = "audio/mpeg"
        elif recording.file_path.lower().endswith('.wav'):
            media_type = "audio/wav"
        
        return FileResponse(
            path=recording.file_path,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )
    else:
        raise HTTPException(
            status_code=404, detail="Local recording file not available"
        )


@router.get("/calls/{call_id}/transcripts/export")
async def export_transcripts(
    request: Request, call_id: int, format: str = "txt", db: Session = Depends(get_db)
):
    """Export call transcripts in various formats."""
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    transcripts = (
        db.query(Transcript)
        .filter(Transcript.call_id == call_id)
        .order_by(Transcript.created_at)
        .all()
    )

    if not transcripts:
        raise HTTPException(status_code=404, detail="No transcripts found")

    if format == "txt":
        # Create plain text format
        content = f"Call Transcript - {call.call_sid}\n"
        content += f"Started: {call.started_at}\n"
        content += f"From: {call.customer_phone_number}\n"
        content += f"To: {call.to_phone_number}\n"
        content += "=" * 50 + "\n\n"

        for transcript in transcripts:
            speaker = transcript.speaker.upper() if transcript.speaker else "UNKNOWN"
            timestamp = transcript.created_at.strftime("%H:%M:%S")
            content += f"[{timestamp}] {speaker}: {transcript.content}\n"

        return Response(
            content=content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=call-{call.call_sid}-transcript.txt"
            },
        )

    elif format == "json":
        # Create JSON format
        transcript_data = {
            "call_sid": call.call_sid,
            "started_at": call.started_at.isoformat() if call.started_at else None,
            "customer_phone_number": call.customer_phone_number,
            "to_phone_number": call.to_phone_number,
            "transcripts": [
                {
                    "speaker": t.speaker,
                    "content": t.content,
                    "timestamp": t.created_at.isoformat() if t.created_at else None,
                    "confidence": t.confidence,
                    "is_final": t.is_final,
                }
                for t in transcripts
            ],
        }

        return Response(
            content=json.dumps(transcript_data, indent=2),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=call-{call.call_sid}-transcript.json"
            },
        )

    else:
        raise HTTPException(
            status_code=400, detail="Unsupported format. Use 'txt' or 'json'."
        )


# ========== Calls Routes ==========
@router.get("/calls", response_class=HTMLResponse)
async def list_calls(
    request: Request,
    page: int = 1,
    per_page: int = 10,
    search: str = None,
    status: str = None,
    assistant_id: int = None,
    date_range: str = None,
    sort_by: str = "started_at",
    sort_order: str = "desc",
    db: Session = Depends(get_db),
):
    """List calls with pagination, filtering, and sorting."""
    # Base query
    query = db.query(Call)

    # Apply search filter
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (Call.call_sid.ilike(search_term))
            | (Call.customer_phone_number.ilike(search_term))
            | (Call.to_phone_number.ilike(search_term))
        )

    # Apply status filter
    if status:
        if status == "active":
            query = query.filter(Call.status == "ongoing")
        elif status == "completed":
            query = query.filter(Call.status == "completed")
        elif status == "failed":
            query = query.filter(
                Call.status.in_(["failed", "no-answer", "busy", "canceled"])
            )
        else:
            query = query.filter(Call.status == status)

    # Apply assistant filter
    if assistant_id:
        query = query.filter(Call.assistant_id == assistant_id)

    # Apply date range filter
    if date_range:
        today = datetime.datetime.now().date()

        if date_range == "today":
            query = query.filter(func.date(Call.started_at) == today)
        elif date_range == "yesterday":
            yesterday = today - datetime.timedelta(days=1)
            query = query.filter(func.date(Call.started_at) == yesterday)
        elif date_range == "week":
            week_ago = today - datetime.timedelta(days=7)
            query = query.filter(Call.started_at >= week_ago)
        elif date_range == "month":
            month_ago = today - datetime.timedelta(days=30)
            query = query.filter(Call.started_at >= month_ago)

    # Get total count before pagination
    total_count = query.count()

    # Apply sorting
    if sort_by == "started_at":
        order_col = Call.started_at
    elif sort_by == "duration":
        order_col = Call.duration
    elif sort_by == "customer_phone":
        order_col = Call.customer_phone_number
    elif sort_by == "status":
        order_col = Call.status
    elif sort_by == "assistant":
        order_col = Assistant.name
        query = query.join(Assistant)
    else:
        order_col = Call.started_at

    if sort_order == "asc":
        query = query.order_by(asc(order_col))
    else:
        query = query.order_by(desc(order_col))

    # Apply pagination
    offset = (page - 1) * per_page
    calls = query.offset(offset).limit(per_page).all()

    # Calculate pagination info
    total_pages = (total_count + per_page - 1) // per_page
    has_prev = page > 1
    has_next = page < total_pages

    # Calculate page range for pagination display
    page_range_start = max(1, page - 2)
    page_range_end = min(total_pages + 1, page + 3)
    page_numbers = list(range(page_range_start, page_range_end))

    # Add additional data for each call
    calls_data = []
    for call in calls:
        recording_count = (
            db.query(Recording).filter(Recording.call_id == call.id).count()
        )
        transcript_count = (
            db.query(Transcript).filter(Transcript.call_id == call.id).count()
        )

        # Calculate call quality from transcripts
        avg_confidence = (
            db.query(func.avg(Transcript.confidence))
            .filter(Transcript.call_id == call.id, Transcript.confidence.isnot(None))
            .scalar()
        )
        quality = int(avg_confidence * 100) if avg_confidence else None

        calls_data.append(
            {
                "call": call,
                "recording_count": recording_count,
                "transcript_count": transcript_count,
                "has_recording": recording_count > 0,
                "has_transcripts": transcript_count > 0,
                "quality": quality,
            }
        )

    # Calculate overall statistics
    total_calls = db.query(Call).count()
    active_calls = db.query(Call).filter(Call.status == "ongoing").count()
    completed_calls = db.query(Call).filter(Call.status == "completed").count()
    failed_calls = (
        db.query(Call)
        .filter(Call.status.in_(["failed", "no-answer", "busy", "canceled"]))
        .count()
    )

    # Calculate average duration for completed calls
    avg_duration_result = (
        db.query(func.avg(Call.duration))
        .filter(Call.status == "completed", Call.duration.isnot(None))
        .scalar()
    )
    avg_duration = int(avg_duration_result) if avg_duration_result else 0

    # Calculate success rate
    success_rate = (completed_calls / total_calls * 100) if total_calls > 0 else 0

    # Get available assistants for filter dropdown
    organization_id = request.session.get("organization_id")
    if organization_id:
        assistants = await AssistantService.get_assistants(organization_id=organization_id, active_only=False)
    else:
        assistants = []

    return templates.TemplateResponse(
        "calls/index.html",
        get_template_context(
            request,
            calls_data=calls_data,
            pagination={
                "page": page,
                "per_page": per_page,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_prev": has_prev,
                "has_next": has_next,
                "prev_page": page - 1 if has_prev else None,
                "next_page": page + 1 if has_next else None,
                "page_range_start": page_range_start,
                "page_range_end": page_range_end,
                "page_numbers": page_numbers,
            },
            filters={
                "search": search or "",
                "status": status or "",
                "assistant_id": assistant_id or "",
                "date_range": date_range or "",
                "sort_by": sort_by,
                "sort_order": sort_order,
            },
            stats={
                "total_calls": total_calls,
                "active_calls": active_calls,
                "completed_calls": completed_calls,
                "failed_calls": failed_calls,
                "success_rate": round(success_rate, 1),
                "avg_duration": avg_duration,
            },
            assistants=assistants,
        ),
    )


@router.get("/calls/{call_id}", response_class=HTMLResponse)
async def view_call(request: Request, call_id: int, db: Session = Depends(get_db)):
    """View a call with recordings and transcripts."""
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    # Get recordings for this call
    recordings = db.query(Recording).filter(Recording.call_id == call_id).all()

    # Get transcripts for this call, ordered by creation time
    transcripts = (
        db.query(Transcript)
        .filter(Transcript.call_id == call_id)
        .order_by(Transcript.created_at)
        .all()
    )

    # Calculate dynamic metrics from actual data
    metrics = calculate_call_metrics(call, transcripts)

    # Group transcripts by speaker for conversation view
    conversation = []
    current_speaker = None
    current_messages = []

    for transcript in transcripts:
        if transcript.speaker != current_speaker:
            if current_messages:
                conversation.append(
                    {"speaker": current_speaker, "messages": current_messages}
                )
            current_speaker = transcript.speaker
            current_messages = [transcript]
        else:
            current_messages.append(transcript)

    # Add the last group
    if current_messages:
        conversation.append({"speaker": current_speaker, "messages": current_messages})

    return templates.TemplateResponse(
        "calls/view.html",
        get_template_context(
            request,
            call=call,
            recordings=recordings,
            transcripts=transcripts,
            conversation=conversation,
            metrics=metrics,
        ),
    )


def calculate_call_metrics(call, transcripts):
    """Calculate dynamic metrics from call and transcript data."""
    logger = logging.getLogger(__name__)
    logger.info(f"Calculating metrics for call {call.call_sid if call else 'None'} with {len(transcripts)} transcripts")
    
    if not transcripts:
        logger.info("No transcripts found, returning default metrics")
        return {
            "avg_response_time": 0,
            "fastest_response_time": 0,
            "response_consistency": 0,
            "conversation_flow_score": 0,
            "engagement_score": 0,
            "quality_score": 0,
            "user_percentage": 50,
            "ai_percentage": 50,
            "user_turns": 0,
            "ai_turns": 0,
        }

    # Separate transcripts by speaker
    user_transcripts = [t for t in transcripts if t.speaker == 'user']
    ai_transcripts = [t for t in transcripts if t.speaker == 'assistant']
    
    logger.info(f"Found {len(user_transcripts)} user transcripts and {len(ai_transcripts)} AI transcripts")
    
    # Sort all transcripts by timing for proper chronological order
    all_transcripts_with_timing = [
        t for t in transcripts 
        if t.segment_start is not None and t.segment_end is not None
    ]
    all_transcripts_with_timing.sort(key=lambda x: x.segment_start)
    
    logger.info(f"Found {len(all_transcripts_with_timing)} transcripts with timing data")
    
    # Log some sample timing data
    for i, t in enumerate(all_transcripts_with_timing[:3]):  # Log first 3
        logger.info(f"Sample transcript {i}: speaker={t.speaker}, start={t.segment_start}, end={t.segment_end}, content='{t.content[:50]}...'")
    
    # Calculate response times using chronological matching
    response_times = []
    
    # Method 1: Use timing data if available
    if all_transcripts_with_timing:
        for i, transcript in enumerate(all_transcripts_with_timing):
            if transcript.speaker == 'user':
                # Find the next assistant response after this user message
                for j in range(i + 1, len(all_transcripts_with_timing)):
                    next_transcript = all_transcripts_with_timing[j]
                    if next_transcript.speaker == 'assistant':
                        # Calculate response time
                        response_time = next_transcript.segment_start - transcript.segment_end
                        logger.info(f"Response time calculation: {next_transcript.segment_start} - {transcript.segment_end} = {response_time}")
                        if response_time > 0 and response_time <= 30:  # Reasonable response time
                            response_times.append(response_time)
                            logger.info(f"Added response time: {response_time}")
                        else:
                            logger.info(f"Rejected response time: {response_time} (out of bounds)")
                        break  # Only match with the first assistant response
    
    logger.info(f"Method 1 found {len(response_times)} response times: {response_times}")
    
    # Method 2: Fallback to timestamp differences if no timing data or no response times found
    if not response_times and user_transcripts and ai_transcripts:
        logger.info("Falling back to timestamp-based calculation")
        # Sort transcripts by creation time
        all_transcripts_by_time = sorted(transcripts, key=lambda x: x.created_at or datetime.datetime.min)
        
        for i, transcript in enumerate(all_transcripts_by_time):
            if transcript.speaker == 'user':
                # Find the next assistant response after this user message
                for j in range(i + 1, len(all_transcripts_by_time)):
                    next_transcript = all_transcripts_by_time[j]
                    if next_transcript.speaker == 'assistant':
                        if transcript.created_at and next_transcript.created_at:
                            response_time = (next_transcript.created_at - transcript.created_at).total_seconds()
                            logger.info(f"Timestamp response time: {response_time}")
                            # Only consider reasonable response times (0.1 to 30 seconds)
                            if 0.1 <= response_time <= 30:
                                response_times.append(response_time)
                                logger.info(f"Added timestamp response time: {response_time}")
                        break  # Only match with the first assistant response
    
    logger.info(f"Final response times: {response_times}")
    
    # Calculate average and fastest response times
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    fastest_response_time = min(response_times) if response_times else 0
    
    # Calculate response consistency (percentage of responses within 2s of average)
    if response_times:
        consistent_responses = sum(1 for rt in response_times if abs(rt - avg_response_time) <= 2.0)
        response_consistency = (consistent_responses / len(response_times)) * 100
    else:
        response_consistency = 0
    
    # Calculate speaking time distribution
    user_time = len(user_transcripts) * 3  # Approximate 3 seconds per segment
    ai_time = len(ai_transcripts) * 3
    total_time = user_time + ai_time
    
    if total_time > 0:
        user_percentage = round((user_time / total_time) * 100)
        ai_percentage = round((ai_time / total_time) * 100)
    else:
        user_percentage = 50
        ai_percentage = 50
    
    # Calculate quality score from confidence levels
    confidences = [t.confidence for t in transcripts if t.confidence is not None]
    quality_score = round(sum(confidences) / len(confidences) * 100) if confidences else 0
    
    # Calculate conversation flow score (based on turn-taking pattern)
    conversation_flow_score = min(95, max(50, 100 - abs(len(user_transcripts) - len(ai_transcripts)) * 5))
    
    # Calculate engagement score (based on transcript length and frequency)
    avg_transcript_length = sum(len(t.content) for t in transcripts) / len(transcripts) if transcripts else 0
    engagement_score = min(100, max(30, (avg_transcript_length / 50) * 100))
    
    calculated_metrics = {
        "avg_response_time": round(avg_response_time, 2),
        "fastest_response_time": round(fastest_response_time, 2),
        "response_consistency": round(response_consistency),
        "conversation_flow_score": round(conversation_flow_score),
        "engagement_score": round(engagement_score),
        "quality_score": quality_score,
        "user_percentage": user_percentage,
        "ai_percentage": ai_percentage,
        "user_turns": len(user_transcripts),
        "ai_turns": len(ai_transcripts),
    }
    
    logger.info(f"Calculated metrics: {calculated_metrics}")
    return calculated_metrics


@router.get("/calls/export", response_class=Response)
async def export_calls(
    request: Request,
    format: str = "csv",
    search: str = None,
    status: str = None,
    assistant_id: int = None,
    date_range: str = None,
    db: Session = Depends(get_db),
):
    """Export calls data in CSV or JSON format."""
    import csv

    # Get calls with same filtering as list view
    query = db.query(Call)

    # Apply search filter
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (Call.call_sid.ilike(search_term))
            | (Call.customer_phone_number.ilike(search_term))
            | (Call.to_phone_number.ilike(search_term))
        )

    # Apply status filter
    if status:
        if status == "active":
            query = query.filter(Call.status == "ongoing")
        elif status == "completed":
            query = query.filter(Call.status == "completed")
        elif status == "failed":
            query = query.filter(
                Call.status.in_(["failed", "no-answer", "busy", "canceled"])
            )
        else:
            query = query.filter(Call.status == status)

    # Apply assistant filter
    if assistant_id:
        query = query.filter(Call.assistant_id == assistant_id)

    # Apply date range filter
    if date_range:
        today = datetime.datetime.now().date()

        if date_range == "today":
            query = query.filter(func.date(Call.started_at) == today)
        elif date_range == "yesterday":
            yesterday = today - datetime.timedelta(days=1)
            query = query.filter(func.date(Call.started_at) == yesterday)
        elif date_range == "week":
            week_ago = today - datetime.timedelta(days=7)
            query = query.filter(Call.started_at >= week_ago)
        elif date_range == "month":
            month_ago = today - datetime.timedelta(days=30)
            query = query.filter(Call.started_at >= month_ago)

    calls = query.order_by(desc(Call.started_at)).all()

    # Prepare export data
    export_data = []
    for call in calls:
        recording_count = (
            db.query(Recording).filter(Recording.call_id == call.id).count()
        )
        transcript_count = (
            db.query(Transcript).filter(Transcript.call_id == call.id).count()
        )

        # Calculate call quality from transcripts
        avg_confidence = (
            db.query(func.avg(Transcript.confidence))
            .filter(Transcript.call_id == call.id, Transcript.confidence.isnot(None))
            .scalar()
        )
        quality = int(avg_confidence * 100) if avg_confidence else None

        export_data.append(
            {
                "call_sid": call.call_sid,
                "assistant_name": call.assistant.name if call.assistant else "Unknown",
                "customer_phone": call.customer_phone_number,
                "to_phone": call.to_phone_number,
                "status": call.status.capitalize(),
                "duration": f"{call.duration}s" if call.duration else "N/A",
                "recording_count": recording_count,
                "transcript_count": transcript_count,
                "quality": f"{quality}%" if quality is not None else "N/A",
                "started_at": (
                    call.started_at.strftime("%Y-%m-%d %H:%M:%S")
                    if call.started_at
                    else ""
                ),
                "ended_at": (
                    call.ended_at.strftime("%Y-%m-%d %H:%M:%S") if call.ended_at else ""
                ),
            }
        )

    if format.lower() == "json":
        # Export as JSON
        json_content = json.dumps(export_data, indent=2)
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=calls.json"},
        )
    else:
        # Export as CSV (default)
        output = StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "call_sid",
                "assistant_name",
                "customer_phone",
                "to_phone",
                "status",
                "duration",
                "recording_count",
                "transcript_count",
                "quality",
                "started_at",
                "ended_at",
            ],
        )
        writer.writeheader()
        writer.writerows(export_data)

        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=calls.csv"},
        )


@router.post("/calls/bulk-action")
async def bulk_action_calls(
    request: Request,
    action: str = Form(...),
    call_ids: str = Form(...),
    db: Session = Depends(get_db),
):
    """Perform bulk actions on calls."""
    try:
        # Parse call IDs
        ids = [int(id.strip()) for id in call_ids.split(",") if id.strip()]

        if not ids:
            return {"success": False, "message": "No calls selected"}

        # Get calls
        calls = db.query(Call).filter(Call.id.in_(ids)).all()

        if action == "delete":
            # Delete related records first
            for call in calls:
                # Delete recordings
                recordings = (
                    db.query(Recording).filter(Recording.call_id == call.id).all()
                )
                for recording in recordings:
                    # Delete physical file if exists
                    if recording.file_path and os.path.exists(recording.file_path):
                        try:
                            os.remove(recording.file_path)
                        except Exception as e:
                            logging.warning(
                                f"Could not delete recording file {recording.file_path}: {e}"
                            )
                    db.delete(recording)

                # Delete transcripts
                transcripts = (
                    db.query(Transcript).filter(Transcript.call_id == call.id).all()
                )
                for transcript in transcripts:
                    db.delete(transcript)

                # Delete call
                db.delete(call)

            message = (
                f"Deleted {len(calls)} calls with their recordings and transcripts"
            )

        elif action == "download_recordings":
            # Create a zip file with all recordings
            import zipfile
            import tempfile

            # Create temporary zip file
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")

            with zipfile.ZipFile(temp_zip.name, "w") as zip_file:
                for call in calls:
                    recordings = (
                        db.query(Recording).filter(Recording.call_id == call.id).all()
                    )
                    for recording in recordings:
                        if recording.file_path and os.path.exists(recording.file_path):
                            # Determine file extension
                            file_ext = "wav"
                            if recording.file_path.lower().endswith('.mp3'):
                                file_ext = "mp3"
                            elif recording.file_path.lower().endswith('.wav'):
                                file_ext = "wav"
                            
                            # Add file to zip with call-specific name
                            zip_filename = f"{call.call_sid}_{recording.recording_type}_{recording.id}.{file_ext}"
                            zip_file.write(recording.file_path, zip_filename)

            return {
                "success": True,
                "message": f"Created download for {len(calls)} calls",
                "download_url": f"/download/temp/{os.path.basename(temp_zip.name)}",
            }

        else:
            return {"success": False, "message": "Invalid action"}

        db.commit()
        return {"success": True, "message": message}

    except Exception as e:
        db.rollback()
        return {"success": False, "message": f"Error: {str(e)}"}
