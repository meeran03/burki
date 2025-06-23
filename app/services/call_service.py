# pylint: disable=logging-fstring-interpolation, broad-exception-caught
import os
import logging
import time
import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import selectinload

from app.db.models import Call, Recording, Transcript, ChatMessage
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
        Get call by SID with assistant relationship eagerly loaded.

        Args:
            call_sid: Call SID

        Returns:
            Optional[Call]: Found call or None
        """
        async with await get_async_db_session() as db:
            query = select(Call).options(selectinload(Call.assistant)).where(Call.call_sid == call_sid)
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
    async def create_s3_recording(
        call_sid: str,
        s3_key: str,
        s3_url: str,
        duration: float,
        file_size: int,
        format: str = "mp3",
        sample_rate: int = 22050,
        channels: int = 1,
        recording_type: str = "mixed",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Recording]:
        """
        Create an S3-based recording for a call.

        Args:
            call_sid: Call SID
            s3_key: S3 object key
            s3_url: S3 public URL
            duration: Recording duration in seconds
            file_size: File size in bytes
            format: Audio format
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            recording_type: Recording type (user, assistant, mixed)
            metadata: Additional metadata

        Returns:
            Optional[Recording]: Created recording record or None
        """
        try:
            call = await CallService.get_call_by_sid(call_sid)
            if not call:
                logger.error(f"Call not found for SID: {call_sid}")
                return None

            async with await get_async_db_session() as db:
                # Create recording record in database
                recording_data = {
                    "call_id": call.id,
                    "s3_key": s3_key,
                    "s3_url": s3_url,
                    "s3_bucket": os.getenv("AWS_S3_BUCKET_NAME"),
                    "duration": duration,
                    "file_size": file_size,
                    "format": format,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "recording_type": recording_type,
                    "recording_source": "s3",
                    "status": "completed",
                    "uploaded_at": datetime.datetime.utcnow(),
                    "recording_metadata": metadata or {},
                }

                recording = Recording(**recording_data)
                db.add(recording)
                await db.commit()
                await db.refresh(recording)
                logger.info(
                    f"Created S3 recording with ID: {recording.id} for call SID: {call_sid}, S3 key: {s3_key}"
                )

                return recording
        except SQLAlchemyError as e:
            logger.error(f"Error creating S3 recording: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating S3 recording: {e}")
            raise

    @staticmethod
    async def create_recording(
        call_sid: str,
        audio_data: bytes = None,
        format: str = "mp3",
        recording_type: str = "mixed",
        recording_source: str = "s3",
        recording_sid: str = None,
        status: str = "recording",
        file_path: str = None,
        s3_key: str = None,
        s3_url: str = None,
        duration: float = None,
        file_size: int = None,
        sample_rate: int = None,
        channels: int = None,
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
            status: Recording status (recording, completed, failed)
            file_path: Direct file path (for existing files)
            s3_key: S3 object key
            s3_url: S3 public URL
            duration: Recording duration in seconds
            file_size: File size in bytes
            sample_rate: Sample rate in Hz
            channels: Number of audio channels

        Returns:
            Optional[Tuple[Recording, str]]: Recording object and file path or None
        """
        try:
            call = await CallService.get_call_by_sid(call_sid)
            if not call:
                logger.error(f"Call not found for SID: {call_sid}")
                return None

            # Handle local recordings (save audio data to file)
            if recording_source == "local" and audio_data and not file_path:
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
                    "format": format,
                    "recording_type": recording_type,
                    "recording_source": recording_source,
                    "status": status,
                    "s3_key": s3_key,
                    "s3_url": s3_url,
                    "duration": duration,
                    "file_size": file_size,
                    "sample_rate": sample_rate,
                    "channels": channels,
                }

                recording = Recording(**recording_data)
                db.add(recording)
                await db.commit()
                await db.refresh(recording)
                logger.info(
                    f"Created {recording_source} recording with ID: {recording.id} for call SID: {call_sid}"
                )

                return recording, file_path or s3_url
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
        s3_url: str = None,
        duration: float = None,
        local_file_path: str = None,
    ) -> Optional[Recording]:
        """
        Update recording status and metadata.

        Args:
            recording_sid: Twilio Recording SID
            status: New status (completed, failed)
            s3_url: S3 URL for the recording
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
                if s3_url:
                    recording.s3_url = s3_url
                if duration:
                    recording.duration = duration
                if local_file_path:
                    # Note: Recording model doesn't have file_path field, using s3_key instead for local paths
                    recording.s3_key = local_file_path

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

    @staticmethod
    async def create_chat_message(
        call_sid: str,
        role: str,
        content: str,
        message_index: int,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ChatMessage]:
        """
        Create a chat message for a call.

        Args:
            call_sid: Call SID
            role: Message role (system, user, assistant)
            content: Message content
            message_index: Index in the conversation
            llm_provider: LLM provider name (for assistant messages)
            llm_model: LLM model used (for assistant messages)
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            total_tokens: Total tokens used
            metadata: Additional metadata

        Returns:
            Optional[ChatMessage]: Created chat message or None
        """
        try:
            call = await CallService.get_call_by_sid(call_sid)
            if not call:
                logger.error(f"Call not found for SID: {call_sid}")
                return None

            async with await get_async_db_session() as db:
                # Create chat message record
                message_data = {
                    "call_id": call.id,
                    "role": role,
                    "content": content,
                    "message_index": message_index,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "message_metadata": metadata or {},
                }

                chat_message = ChatMessage(**message_data)
                db.add(chat_message)
                await db.commit()
                await db.refresh(chat_message)
                logger.debug(
                    f"Created chat message with ID: {chat_message.id} for call SID: {call_sid}, role: {role}"
                )

                return chat_message
        except SQLAlchemyError as e:
            logger.error(f"Error creating chat message: {e}")
            raise

    @staticmethod
    async def get_call_chat_messages(
        call_sid: str, role: Optional[str] = None
    ) -> List[ChatMessage]:
        """
        Get all chat messages for a call.

        Args:
            call_sid: Call SID
            role: Filter by role (system, user, assistant)

        Returns:
            List[ChatMessage]: List of chat messages ordered by message_index
        """
        try:
            call = await CallService.get_call_by_sid(call_sid)
            if not call:
                return []

            async with await get_async_db_session() as db:
                query = select(ChatMessage).where(ChatMessage.call_id == call.id)

                if role:
                    query = query.filter(ChatMessage.role == role)

                # Order by message index to maintain conversation order
                query = query.order_by(ChatMessage.message_index)

                result = await db.execute(query)
                return list(result.scalars().all())
        except SQLAlchemyError as e:
            logger.error(f"Error getting call chat messages: {e}")
            return []

    @staticmethod
    async def store_conversation_history(
        call_sid: str,
        conversation_history: List[Dict[str, str]],
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> bool:
        """
        Store entire conversation history to database.
        This method will check for existing messages and only store new ones.

        Args:
            call_sid: Call SID
            conversation_history: List of conversation messages
            llm_provider: LLM provider name
            llm_model: LLM model used

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get existing messages to avoid duplicates
            existing_messages = await CallService.get_call_chat_messages(call_sid)
            existing_count = len(existing_messages)

            # Only store new messages beyond what we already have
            new_messages = conversation_history[existing_count:]
            
            for i, message in enumerate(new_messages):
                message_index = existing_count + i
                role = message.get("role", "")
                content = message.get("content", "")
                
                if not role or not content:
                    continue
                
                await CallService.create_chat_message(
                    call_sid=call_sid,
                    role=role,
                    content=content,
                    message_index=message_index,
                    llm_provider=llm_provider if role == "assistant" else None,
                    llm_model=llm_model if role == "assistant" else None,
                )
            
            if new_messages:
                logger.info(f"Stored {len(new_messages)} new chat messages for call {call_sid}")
            
            return True
        except Exception as e:
            logger.error(f"Error storing conversation history for call {call_sid}: {e}")
            return False
