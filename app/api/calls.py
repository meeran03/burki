from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Call, Transcript, Recording
from app.services.call_service import CallService
from app.api.schemas import CallResponse, TranscriptResponse, RecordingResponse

router = APIRouter(prefix="/calls", tags=["calls"])

@router.get("/", response_model=List[CallResponse])
async def get_calls(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    assistant_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Get a list of calls with optional filtering.
    """
    query = db.query(Call)
    
    if status:
        query = query.filter(Call.status == status)
    
    if assistant_id:
        query = query.filter(Call.assistant_id == assistant_id)
    
    # Order by start time, newest first
    query = query.order_by(Call.started_at.desc())
    
    calls = query.offset(skip).limit(limit).all()
    return calls

@router.get("/{call_id}", response_model=CallResponse)
async def get_call(
    call_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific call by ID.
    """
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with ID {call_id} not found"
        )
    return call

@router.get("/sid/{call_sid}", response_model=CallResponse)
async def get_call_by_sid(
    call_sid: str,
    db: Session = Depends(get_db)
):
    """
    Get a specific call by Twilio Call SID.
    """
    call = await CallService.get_call_by_sid(db, call_sid)
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with SID {call_sid} not found"
        )
    return call

@router.get("/{call_id}/transcripts", response_model=List[TranscriptResponse])
async def get_call_transcripts(
    call_id: int,
    speaker: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get all transcripts for a call.
    """
    # Check if call exists
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with ID {call_id} not found"
        )
    
    # Get transcripts for the call
    query = db.query(Transcript).filter(Transcript.call_id == call_id)
    
    if speaker:
        query = query.filter(Transcript.speaker == speaker)
    
    # Order by creation time
    query = query.order_by(Transcript.created_at)
    
    transcripts = query.all()
    return transcripts

@router.get("/sid/{call_sid}/transcripts", response_model=List[TranscriptResponse])
async def get_call_sid_transcripts(
    call_sid: str,
    speaker: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get all transcripts for a call by Twilio Call SID.
    """
    # Get call by SID
    call = await CallService.get_call_by_sid(db, call_sid)
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with SID {call_sid} not found"
        )
    
    # Get transcripts for the call
    transcripts = await CallService.get_call_transcripts(db, call_sid, speaker)
    return transcripts

@router.get("/{call_id}/recordings", response_model=List[RecordingResponse])
async def get_call_recordings(
    call_id: int,
    recording_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get all recordings for a call.
    """
    # Check if call exists
    call = db.query(Call).filter(Call.id == call_id).first()
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with ID {call_id} not found"
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
    recording_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get all recordings for a call by Twilio Call SID.
    """
    # Get call by SID
    call = await CallService.get_call_by_sid(db, call_sid)
    if not call:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Call with SID {call_sid} not found"
        )
    
    # Get recordings for the call
    recordings = await CallService.get_call_recordings(db, call_sid, recording_type)
    return recordings 