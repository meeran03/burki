import os
import logging
import time
import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import Call, Recording, Transcript, Assistant

logger = logging.getLogger(__name__)

class CallService:
    """
    Service class for handling Call, Recording, and Transcript operations.
    """
    
    RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "recordings")
    
    @staticmethod
    async def create_call(
        db: Session, 
        call_sid: str, 
        assistant_id: int, 
        to_phone_number: str, 
        customer_phone_number: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Call:
        """
        Create a new call record.
        
        Args:
            db: Database session
            call_sid: Twilio Call SID
            assistant_id: Assistant ID
            to_phone_number: Assistant's phone number
            customer_phone_number: Caller's phone number
            metadata: Additional metadata
            
        Returns:
            Call: Created call record
        """
        try:
            call_data = {
                "call_sid": call_sid,
                "assistant_id": assistant_id,
                "to_phone_number": to_phone_number,
                "customer_phone_number": customer_phone_number,
                "call_meta": metadata or {},
                "status": "ongoing"
            }
            
            call = Call(**call_data)
            db.add(call)
            db.commit()
            db.refresh(call)
            logger.info(f"Created call record with ID: {call.id}, SID: {call_sid}")

            return call
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating call record: {e}")
            raise
    
    @staticmethod
    async def get_call_by_sid(db: Session, call_sid: str) -> Optional[Call]:
        """
        Get call by SID.
        
        Args:
            db: Database session
            call_sid: Call SID
            
        Returns:
            Optional[Call]: Found call or None
        """
        return db.query(Call).filter(Call.call_sid == call_sid).first()
    
    @staticmethod
    async def update_call_status(db: Session, call_sid: str, status: str, duration: Optional[int] = None) -> Optional[Call]:
        """
        Update call status.
        
        Args:
            db: Database session
            call_sid: Call SID
            status: New status (ongoing, completed, failed)
            duration: Call duration in seconds
            
        Returns:
            Optional[Call]: Updated call or None
        """
        try:
            call = await CallService.get_call_by_sid(db, call_sid)
            if not call:
                return None
            
            # Store the previous status for webhook notification
            previous_status = call.status
            
            # Update status and duration
            call.status = status
            
            if status == "completed" or status == "failed":
                call.ended_at = datetime.datetime.utcnow()
                
                # Calculate duration if not provided
                if duration is None and call.started_at:
                    call.duration = int((datetime.datetime.utcnow() - call.started_at).total_seconds())
                else:
                    call.duration = duration
            
            db.commit()
            db.refresh(call)
            logger.info(f"Updated call status to {status} for call SID: {call_sid}")
            
            return call
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error updating call status: {e}")
            raise
    
    @staticmethod
    async def create_recording(
        db: Session, 
        call_sid: str, 
        audio_data: bytes, 
        format: str = "wav",
        recording_type: str = "full"
    ) -> Optional[Tuple[Recording, str]]:
        """
        Create a recording for a call.
        
        Args:
            db: Database session
            call_sid: Call SID
            audio_data: Audio data as bytes
            format: Audio format
            recording_type: Recording type (full, segment)
            
        Returns:
            Optional[Tuple[Recording, str]]: Recording object and file path or None
        """
        try:
            call = await CallService.get_call_by_sid(db, call_sid)
            if not call:
                logger.error(f"Call not found for SID: {call_sid}")
                return None
            
            # Create recordings directory if it doesn't exist
            os.makedirs(CallService.RECORDINGS_DIR, exist_ok=True)
            
            # Create a directory for this call
            call_dir = os.path.join(CallService.RECORDINGS_DIR, call_sid)
            os.makedirs(call_dir, exist_ok=True)
            
            # Generate file name and path
            timestamp = int(time.time())
            file_name = f"{recording_type}_{timestamp}.{format}"
            file_path = os.path.join(call_dir, file_name)
            
            # Save audio data to file
            with open(file_path, "wb") as f:
                f.write(audio_data)
            
            # Create recording record in database
            recording_data = {
                "call_id": call.id,
                "file_path": file_path,
                "format": format,
                "recording_type": recording_type
            }
            
            recording = Recording(**recording_data)
            db.add(recording)
            db.commit()
            db.refresh(recording)
            logger.info(f"Created recording with ID: {recording.id} for call SID: {call_sid}")
            
            return recording, file_path
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating recording: {e}")
            raise
        except Exception as e:
            logger.error(f"Error saving recording file: {e}")
            raise
    
    @staticmethod
    async def create_transcript(
        db: Session,
        call_sid: str,
        content: str,
        is_final: bool = True,
        speaker: Optional[str] = None,
        segment_start: Optional[float] = None,
        segment_end: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> Optional[Transcript]:
        """
        Create a transcript for a call.
        
        Args:
            db: Database session
            call_sid: Call SID
            content: Transcript content
            is_final: Whether this is a final transcript
            speaker: Speaker identifier (user, assistant)
            segment_start: Start time in seconds
            segment_end: End time in seconds
            confidence: Confidence score
            
        Returns:
            Optional[Transcript]: Created transcript or None
        """
        try:
            call = await CallService.get_call_by_sid(db, call_sid)
            if not call:
                logger.error(f"Call not found for SID: {call_sid}")
                return None
            
            # Create transcript record
            transcript_data = {
                "call_id": call.id,
                "content": content,
                "is_final": is_final,
                "speaker": speaker,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "confidence": confidence
            }
            
            transcript = Transcript(**transcript_data)
            db.add(transcript)
            db.commit()
            db.refresh(transcript)
            logger.info(f"Created transcript with ID: {transcript.id} for call SID: {call_sid}")

            return transcript
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating transcript: {e}")
            raise
    
    @staticmethod
    async def get_call_transcripts(db: Session, call_sid: str, speaker: Optional[str] = None) -> List[Transcript]:
        """
        Get all transcripts for a call.
        
        Args:
            db: Database session
            call_sid: Call SID
            speaker: Filter by speaker (user, assistant)
            
        Returns:
            List[Transcript]: List of transcripts
        """
        try:
            call = await CallService.get_call_by_sid(db, call_sid)
            if not call:
                return []
            
            query = db.query(Transcript).filter(Transcript.call_id == call.id)
            
            if speaker:
                query = query.filter(Transcript.speaker == speaker)
            
            # Order by creation time
            query = query.order_by(Transcript.created_at)
            
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting call transcripts: {e}")
            return []
    
    @staticmethod
    async def get_call_recordings(db: Session, call_sid: str, recording_type: Optional[str] = None) -> List[Recording]:
        """
        Get all recordings for a call.
        
        Args:
            db: Database session
            call_sid: Call SID
            recording_type: Filter by recording type (full, segment)
            
        Returns:
            List[Recording]: List of recordings
        """
        try:
            call = await CallService.get_call_by_sid(db, call_sid)
            if not call:
                return []
            
            query = db.query(Recording).filter(Recording.call_id == call.id)
            
            if recording_type:
                query = query.filter(Recording.recording_type == recording_type)
            
            # Order by creation time
            query = query.order_by(Recording.created_at)
            
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting call recordings: {e}")
            return [] 