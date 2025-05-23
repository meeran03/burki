# pylint: disable=logging-fstring-interpolation, broad-exception-caught
import os
import logging
import time
import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import Call, Recording, Transcript
from app.db.database import get_async_db_session

logger = logging.getLogger(__name__)


class CallService:
    """
    Service class for handling Call, Recording, and Transcript operations.
    """

    RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "recordings")

    @staticmethod
    async def create_call(
        assistant_id: int,
        call_sid: str,
        to_phone_number: str,
        customer_phone_number: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Call:
        """
        Create a new call record.

        Args:
            assistant_id: Assistant ID
            call_sid: Twilio Call SID
            to_phone_number: Assistant's phone number
            customer_phone_number: Caller's phone number
            metadata: Additional metadata

        Returns:
            Call: Created call record
        """
        try:
            async with await get_async_db_session() as db:
                call_data = {
                    "call_sid": call_sid,
                    "assistant_id": assistant_id,
                    "to_phone_number": to_phone_number,
                    "customer_phone_number": customer_phone_number,
                    "call_meta": metadata or {},
                    "status": "ongoing",
                }

                call = Call(**call_data)
                db.add(call)
                await db.commit()
                await db.refresh(call)
                logger.info(f"Created call record with ID: {call.id}, SID: {call_sid}")

                return call
        except SQLAlchemyError as e:
            logger.error(f"Error creating call record: {e}")
            raise

    @staticmethod
    async def get_call_by_sid(call_sid: str) -> Optional[Call]:
        """
        Get call by SID.

        Args:
            call_sid: Call SID

        Returns:
            Optional[Call]: Found call or None
        """
        async with await get_async_db_session() as db:
            query = select(Call).where(Call.call_sid == call_sid)
            result = await db.execute(query)
            return result.scalar_one_or_none()

    @staticmethod
    async def update_call_status(
        call_sid: str, status: str, duration: Optional[int] = None
    ) -> Optional[Call]:
        """
        Update call status.

        Args:
            call_sid: Call SID
            status: New status (ongoing, completed, failed)
            duration: Call duration in seconds

        Returns:
            Optional[Call]: Updated call or None
        """
        try:
            call = await CallService.get_call_by_sid(call_sid)
            if not call:
                return None

            async with await get_async_db_session() as db:
                # Store the previous status for webhook notification
                previous_status = call.status

                # Update status and duration
                call.status = status

                if status == "completed" or status == "failed":
                    call.ended_at = datetime.datetime.utcnow()

                    # Calculate duration if not provided
                    if duration is None and call.started_at:
                        call.duration = int(
                            (
                                datetime.datetime.utcnow() - call.started_at
                            ).total_seconds()
                        )
                    else:
                        call.duration = duration

                db.add(call)
                await db.commit()
                await db.refresh(call)
                logger.info(f"Updated call status to {status} for call SID: {call_sid}")

                return call
        except SQLAlchemyError as e:
            logger.error(f"Error updating call status: {e}")
            raise

    @staticmethod
    async def create_recording(
        call_sid: str,
        audio_data: bytes = None,
        format: str = "wav",
        recording_type: str = "full",
        recording_source: str = "local",
        recording_sid: str = None,
        recording_url: str = None,
        status: str = "recording",
    ) -> Optional[Tuple[Recording, str]]:
        """
        Create a recording for a call.

        Args:
            call_sid: Call SID
            audio_data: Audio data as bytes (for local recordings)
            format: Audio format
            recording_type: Recording type (full, segment)
            recording_source: Recording source (twilio, local)
            recording_sid: Twilio Recording SID (for Twilio recordings)
            recording_url: Twilio Recording URL (for Twilio recordings)
            status: Recording status (recording, completed, failed)

        Returns:
            Optional[Tuple[Recording, str]]: Recording object and file path or None
        """
        try:
            call = await CallService.get_call_by_sid(call_sid)
            if not call:
                logger.error(f"Call not found for SID: {call_sid}")
                return None

            file_path = None

            # Handle local recordings (save audio data to file)
            if recording_source == "local" and audio_data:
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

            async with await get_async_db_session() as db:
                # Create recording record in database
                recording_data = {
                    "call_id": call.id,
                    "recording_sid": recording_sid,
                    "file_path": file_path,
                    "recording_url": recording_url,
                    "format": format,
                    "recording_type": recording_type,
                    "recording_source": recording_source,
                    "status": status,
                }

                recording = Recording(**recording_data)
                db.add(recording)
                await db.commit()
                await db.refresh(recording)
                logger.info(
                    f"Created {recording_source} recording with ID: {recording.id} for call SID: {call_sid}"
                )

                return recording, file_path or recording_url
        except SQLAlchemyError as e:
            logger.error(f"Error creating recording: {e}")
            raise
        except Exception as e:
            logger.error(f"Error saving recording file: {e}")
            raise

    @staticmethod
    async def create_twilio_recording(
        call_sid: str,
        recording_sid: str,
        status: str = "recording",
    ) -> Optional[Recording]:
        """
        Create a Twilio recording record.

        Args:
            call_sid: Call SID
            recording_sid: Twilio Recording SID
            status: Recording status

        Returns:
            Optional[Recording]: Created recording record or None
        """
        try:
            result = await CallService.create_recording(
                call_sid=call_sid,
                format="mp3",  # Twilio default format
                recording_type="full",
                recording_source="twilio",
                recording_sid=recording_sid,
                status=status,
            )
            
            # Check if result is None (call not found)
            if result is None:
                logger.error(f"Failed to create recording: call {call_sid} not found in database")
                return None
            
            # Unpack the tuple safely
            recording, _ = result
            return recording
            
        except Exception as e:
            logger.error(f"Error creating Twilio recording record: {e}")
            return None

    @staticmethod
    async def update_recording_status(
        recording_sid: str,
        status: str,
        recording_url: str = None,
        duration: float = None,
        local_file_path: str = None,
    ) -> Optional[Recording]:
        """
        Update recording status and metadata.

        Args:
            recording_sid: Twilio Recording SID
            status: New status (completed, failed)
            recording_url: Recording URL from Twilio
            duration: Recording duration in seconds
            local_file_path: Local file path if recording was downloaded

        Returns:
            Optional[Recording]: Updated recording or None
        """
        try:
            async with await get_async_db_session() as db:
                query = select(Recording).where(Recording.recording_sid == recording_sid)
                result = await db.execute(query)
                recording = result.scalar_one_or_none()

                if not recording:
                    logger.error(f"Recording not found for SID: {recording_sid}")
                    return None

                # Update recording
                recording.status = status
                if recording_url:
                    recording.recording_url = recording_url
                if duration:
                    recording.duration = duration
                if local_file_path:
                    recording.file_path = local_file_path

                await db.commit()
                await db.refresh(recording)
                logger.info(f"Updated recording {recording_sid} status to {status}")

                return recording
        except SQLAlchemyError as e:
            logger.error(f"Error updating recording status: {e}")
            return None

    @staticmethod
    async def create_transcript(
        call_sid: str,
        content: str,
        is_final: bool = True,
        speaker: Optional[str] = None,
        segment_start: Optional[float] = None,
        segment_end: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> Optional[Transcript]:
        """
        Create a transcript for a call.

        Args:
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
            call = await CallService.get_call_by_sid(call_sid)
            if not call:
                logger.error(f"Call not found for SID: {call_sid}")
                return None

            async with await get_async_db_session() as db:
                # Create transcript record
                transcript_data = {
                    "call_id": call.id,
                    "content": content,
                    "is_final": is_final,
                    "speaker": speaker,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "confidence": confidence,
                }

                transcript = Transcript(**transcript_data)
                db.add(transcript)
                await db.commit()
                await db.refresh(transcript)
                logger.info(
                    f"Created transcript with ID: {transcript.id} for call SID: {call_sid}"
                )

                return transcript
        except SQLAlchemyError as e:
            logger.error(f"Error creating transcript: {e}")
            raise

    @staticmethod
    async def get_call_transcripts(
        call_sid: str, speaker: Optional[str] = None
    ) -> List[Transcript]:
        """
        Get all transcripts for a call.

        Args:
            call_sid: Call SID
            speaker: Filter by speaker (user, assistant)

        Returns:
            List[Transcript]: List of transcripts
        """
        try:
            call = await CallService.get_call_by_sid(call_sid)
            if not call:
                return []

            async with await get_async_db_session() as db:
                query = select(Transcript).where(Transcript.call_id == call.id)

                if speaker:
                    query = query.filter(Transcript.speaker == speaker)

                # Order by creation time
                query = query.order_by(Transcript.created_at)

                result = await db.execute(query)
                return list(result.scalars().all())
        except SQLAlchemyError as e:
            logger.error(f"Error getting call transcripts: {e}")
            return []

    @staticmethod
    async def get_call_recordings(
        call_sid: str, recording_type: Optional[str] = None
    ) -> List[Recording]:
        """
        Get all recordings for a call.

        Args:
            call_sid: Call SID
            recording_type: Filter by recording type (full, segment)

        Returns:
            List[Recording]: List of recordings
        """
        try:
            call = await CallService.get_call_by_sid(call_sid)
            if not call:
                return []

            async with await get_async_db_session() as db:
                query = select(Recording).where(Recording.call_id == call.id)

                if recording_type:
                    query = query.filter(Recording.recording_type == recording_type)

                # Order by creation time
                query = query.order_by(Recording.created_at)

                result = await db.execute(query)
                return list(result.scalars().all())
        except SQLAlchemyError as e:
            logger.error(f"Error getting call recordings: {e}")
            return []
