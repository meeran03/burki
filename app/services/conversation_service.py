# pylint: disable=logging-fstring-interpolation, broad-exception-caught
import os
import logging
import time
import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import selectinload

from app.db.models import Conversation, Recording, Transcript, ChatMessage
from app.db.database import get_async_db_session

logger = logging.getLogger(__name__)


class ConversationService:
    """
    Service class for handling Conversation, Recording, and Transcript operations.
    """

    RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "recordings")

    @staticmethod
    async def get_conversation_by_sid(channel_sid: str) -> Optional[Conversation]:
        """
        Get conversation by SID with assistant relationship eagerly loaded.

        Args:
            channel_sid: Call SID

        Returns:
            Optional[Conversation]: Found call or None
        """
        async with await get_async_db_session() as db:
            query = (
                select(Conversation)
                .options(selectinload(Conversation.assistant))
                .where(Conversation.channel_sid == channel_sid)
            )
            result = await db.execute(query)
            return result.scalar_one_or_none()

    @staticmethod
    async def create_s3_recording(
        channel_sid: str,
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
        Create an S3-based recording for a conversation.

        Args:
            channel_sid: Call SID
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
            conversation = await ConversationService.get_conversation_by_sid(channel_sid)
            if not conversation:
                logger.error(f"Call not found for SID: {channel_sid}")
                return None

            async with await get_async_db_session() as db:
                # Create recording record in database
                recording_data = {
                    "conversation_id": conversation.id,
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
                    f"Created S3 recording with ID: {recording.id} for conversation SID: {channel_sid}, S3 key: {s3_key}"
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
        channel_sid: str,
        audio_data: bytes = None,
        format: str = "mp3",
        recording_type: str = "mixed",
        recording_source: str = "s3",
        recording_sid: str = None,
        recording_url: str = None,
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
            channel_sid: Call SID
            audio_data: Audio data as bytes (for local recordings)
            format: Audio format
            recording_type: Recording type (full, segment)
            recording_source: Recording source (twilio, local)
            recording_sid: Twilio Recording SID (for Twilio recordings)
            recording_url: Twilio Recording URL (for Twilio recordings)
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
            call = await ConversationService.get_conversation_by_sid(channel_sid)
            if not call:
                logger.error(f"Call not found for SID: {channel_sid}")
                return None

            # Handle local recordings (save audio data to file)
            if recording_source == "local" and audio_data and not file_path:
                # Create recordings directory if it doesn't exist
                os.makedirs(ConversationService.RECORDINGS_DIR, exist_ok=True)

                # Create a directory for this call
                call_dir = os.path.join(ConversationService.RECORDINGS_DIR, channel_sid)
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
                    "conversation_id": call.id,
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
                    f"Created {recording_source} recording with ID: {recording.id} for call SID: {channel_sid}"
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
        channel_sid: str,
        recording_sid: str,
        status: str = "recording",
    ) -> Optional[Recording]:
        """
        Create a Twilio recording record.

        Args:
            channel_sid: Call SID
            recording_sid: Twilio Recording SID
            status: Recording status

        Returns:
            Optional[Recording]: Created recording record or None
        """
        try:
            result = await ConversationService.create_recording(
                channel_sid=channel_sid,
                format="mp3",  # Twilio default format
                recording_type="full",
                recording_source="twilio",
                recording_sid=recording_sid,
                status=status,
            )

            # Check if result is None (call not found)
            if result is None:
                logger.error(
                    f"Failed to create recording: call {channel_sid} not found in database"
                )
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
                query = select(Recording).where(
                    Recording.recording_sid == recording_sid
                )
                result = await db.execute(query)
                recording = result.scalar_one_or_none()

                if not recording:
                    logger.error(f"Recording not found for SID: {recording_sid}")
                    return None

                # Update recording
                recording.status = status
                if recording_url:
                    recording.s3_url = recording_url
                if duration:
                    recording.duration = duration

                await db.commit()
                await db.refresh(recording)
                logger.info(f"Updated recording {recording_sid} status to {status}")

                return recording
        except SQLAlchemyError as e:
            logger.error(f"Error updating recording status: {e}")
            return None

    @staticmethod
    async def update_recording(
        recording_id: int,
        recording_sid: Optional[str] = None,
        s3_key: Optional[str] = None,
        s3_url: Optional[str] = None,
        duration: Optional[float] = None,
        file_size: Optional[int] = None,
        status: Optional[str] = None,
        format: Optional[str] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        recording_type: Optional[str] = None,
        recording_source: Optional[str] = None,
        recording_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Recording]:
        """
        Update a recording record with new information.

        Args:
            recording_id: Recording ID to update
            recording_sid: Twilio Recording SID
            s3_key: S3 object key
            s3_url: S3 public URL
            duration: Recording duration in seconds
            file_size: File size in bytes
            status: Recording status
            format: Audio format
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            recording_type: Recording type
            recording_source: Recording source
            recording_metadata: Additional metadata

        Returns:
            Optional[Recording]: Updated recording or None
        """
        try:
            async with await get_async_db_session() as db:
                query = select(Recording).where(Recording.id == recording_id)
                result = await db.execute(query)
                recording = result.scalar_one_or_none()

                if not recording:
                    logger.error(f"Recording not found for ID: {recording_id}")
                    return None

                # Update fields only if provided
                if recording_sid is not None:
                    recording.recording_sid = recording_sid
                if s3_key is not None:
                    recording.s3_key = s3_key
                if s3_url is not None:
                    recording.s3_url = s3_url
                if duration is not None:
                    recording.duration = duration
                if file_size is not None:
                    recording.file_size = file_size
                if status is not None:
                    recording.status = status
                if format is not None:
                    recording.format = format
                if sample_rate is not None:
                    recording.sample_rate = sample_rate
                if channels is not None:
                    recording.channels = channels
                if recording_type is not None:
                    recording.recording_type = recording_type
                if recording_source is not None:
                    recording.recording_source = recording_source
                if recording_metadata is not None:
                    recording.recording_metadata = recording_metadata
                
                # Set uploaded timestamp if status is completed
                if status == "completed":
                    recording.uploaded_at = datetime.datetime.utcnow()

                await db.commit()
                await db.refresh(recording)
                logger.info(f"Updated recording {recording_id} with new information")

                return recording
        except SQLAlchemyError as e:
            logger.error(f"Error updating recording: {e}")
            return None

    @staticmethod
    async def create_transcript(
        channel_sid: str,
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
            channel_sid: Call SID
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
            call = await ConversationService.get_conversation_by_sid(channel_sid)
            if not call:
                logger.error(f"Call not found for SID: {channel_sid}")
                return None

            async with await get_async_db_session() as db:
                # Create transcript record
                transcript_data = {
                    "conversation_id": call.id,
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
                    f"Created transcript with ID: {transcript.id} for call SID: {channel_sid}"
                )

                return transcript
        except SQLAlchemyError as e:
            logger.error(f"Error creating transcript: {e}")
            raise

    @staticmethod
    async def get_call_transcripts(
        channel_sid: str, speaker: Optional[str] = None
    ) -> List[Transcript]:
        """
        Get all transcripts for a call.

        Args:
            channel_sid: Call SID
            speaker: Filter by speaker (user, assistant)

        Returns:
            List[Transcript]: List of transcripts
        """
        try:
            call = await ConversationService.get_conversation_by_sid(channel_sid)
            if not call:
                return []

            async with await get_async_db_session() as db:
                query = select(Transcript).where(Transcript.conversation_id == call.id)

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
        channel_sid: str, recording_type: Optional[str] = None
    ) -> List[Recording]:
        """
        Get all recordings for a call.

        Args:
            channel_sid: Call SID
            recording_type: Filter by recording type (full, segment)

        Returns:
            List[Recording]: List of recordings
        """
        try:
            call = await ConversationService.get_conversation_by_sid(channel_sid)
            if not call:
                logger.warning(f"No conversation found for channel_sid: {channel_sid}")
                return []

            logger.info(f"Found conversation ID {call.id} for channel_sid {channel_sid}")

            async with await get_async_db_session() as db:
                query = select(Recording).where(Recording.conversation_id == call.id)

                if recording_type:
                    query = query.filter(Recording.recording_type == recording_type)
                    logger.info(f"Filtering recordings by type: {recording_type}")

                # Order by creation time
                query = query.order_by(Recording.created_at)

                result = await db.execute(query)
                recordings = list(result.scalars().all())
                
                logger.info(f"Found {len(recordings)} total recordings for conversation {call.id}")
                for recording in recordings:
                    logger.info(f"Recording: ID={recording.id}, type={recording.recording_type}, s3_key={recording.s3_key}, status={recording.status}")
                
                return recordings
        except SQLAlchemyError as e:
            logger.error(f"Error getting call recordings: {e}", exc_info=True)
            return []

    @staticmethod
    async def create_chat_message(
        channel_sid: str = None,
        conversation_id: int = None,
        role: str = None,
        content: str = None,
        message_index: int = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ChatMessage]:
        """
        Create a chat message for a call/conversation.
        Supports both channel_sid (backward compatibility) and conversation_id.

        Args:
            channel_sid: Call SID (for backward compatibility)
            conversation_id: Conversation ID (preferred)
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
            # Handle backward compatibility - if channel_sid is provided, look up the conversation
            if channel_sid and not conversation_id:
                call = await ConversationService.get_conversation_by_sid(channel_sid)
                if not call:
                    logger.error(f"Call not found for SID: {channel_sid}")
                    return None
                conversation_id = call.id

            if not conversation_id:
                logger.error("Neither channel_sid nor conversation_id provided")
                return None

            async with await get_async_db_session() as db:
                # Create chat message record
                message_data = {
                    "conversation_id": conversation_id,  # Use conversation_id
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
                    f"Created chat message with ID: {chat_message.id} for conversation ID: {conversation_id}, role: {role}"
                )

                return chat_message
        except SQLAlchemyError as e:
            logger.error(f"Error creating chat message: {e}")
            raise

    @staticmethod
    async def get_call_chat_messages(
        channel_sid: str, role: Optional[str] = None
    ) -> List[ChatMessage]:
        """
        Get all chat messages for a call.

        Args:
            channel_sid: Call SID
            role: Filter by role (system, user, assistant)

        Returns:
            List[ChatMessage]: List of chat messages ordered by message_index
        """
        try:
            call = await ConversationService.get_conversation_by_sid(channel_sid)
            if not call:
                return []

            async with await get_async_db_session() as db:
                query = select(ChatMessage).where(
                    ChatMessage.conversation_id == call.id
                )

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
        channel_sid: str,
        conversation_history: List[Dict[str, str]],
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> bool:
        """
        Store entire conversation history to database.
        This method will check for existing messages and only store new ones.

        Args:
            channel_sid: Call SID
            conversation_history: List of conversation messages
            llm_provider: LLM provider name
            llm_model: LLM model used

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get existing messages to avoid duplicates
            existing_messages = await ConversationService.get_call_chat_messages(
                channel_sid
            )
            existing_count = len(existing_messages)

            # Only store new messages beyond what we already have
            new_messages = conversation_history[existing_count:]

            for i, message in enumerate(new_messages):
                message_index = existing_count + i
                role = message.get("role", "")
                content = message.get("content", "")

                if not role or not content:
                    continue

                await ConversationService.create_chat_message(
                    channel_sid=channel_sid,
                    role=role,
                    content=content,
                    message_index=message_index,
                    llm_provider=llm_provider if role == "assistant" else None,
                    llm_model=llm_model if role == "assistant" else None,
                )

            if new_messages:
                logger.info(
                    f"Stored {len(new_messages)} new chat messages for call {channel_sid}"
                )

            return True
        except Exception as e:
            logger.error(
                f"Error storing conversation history for call {channel_sid}: {e}"
            )
            return False

    # ========== New Conversation Methods for SMS Support ==========

    @staticmethod
    async def create_conversation(
        assistant_id: int,
        channel_sid: str,
        conversation_type: str,
        to_phone_number: str,
        customer_phone_number: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Conversation:
        """
        Create a new conversation record (for calls, SMS, etc).

        Args:
            assistant_id: Assistant ID
            channel_sid: Channel-specific SID (channel_sid, message_sid, etc)
            conversation_type: Type of conversation (call, sms, whatsapp, telegram)
            to_phone_number: Assistant's phone number
            customer_phone_number: Customer's phone number
            metadata: Additional metadata

        Returns:
            Conversation: Created conversation record
        """
        try:
            async with await get_async_db_session() as db:
                # Check if conversation already exists with this channel_sid
                existing_query = select(Conversation).where(
                    Conversation.channel_sid == channel_sid
                )
                existing_result = await db.execute(existing_query)
                existing_conversation = existing_result.scalar_one_or_none()
                
                if existing_conversation:
                    logger.info(
                        f"Conversation already exists for channel_sid {channel_sid}, returning existing record"
                    )
                    return existing_conversation

                conversation_data = {
                    "channel_sid": channel_sid,
                    "conversation_type": conversation_type,
                    "assistant_id": assistant_id,
                    "to_phone_number": to_phone_number,
                    "customer_phone_number": customer_phone_number,
                    "conversation_metadata": metadata or {},
                    "status": "ongoing" if conversation_type == "call" else "active",
                }

                conversation = Conversation(**conversation_data)
                db.add(conversation)
                await db.commit()
                await db.refresh(conversation)
                logger.info(
                    f"Created {conversation_type} conversation with ID: {conversation.id}, SID: {channel_sid}"
                )

                return conversation
        except SQLAlchemyError as e:
            logger.error(f"Error creating conversation record: {e}")
            raise

    @staticmethod
    async def get_conversation_by_id(conversation_id: int) -> Optional[Conversation]:
        """
        Get conversation by ID with assistant relationship eagerly loaded.

        Args:
            conversation_id: Conversation ID

        Returns:
            Optional[Conversation]: Found conversation or None
        """
        async with await get_async_db_session() as db:
            query = (
                select(Conversation)
                .options(selectinload(Conversation.assistant))
                .where(Conversation.id == conversation_id)
            )
            result = await db.execute(query)
            return result.scalar_one_or_none()

    @staticmethod
    async def get_conversation_by_phone_numbers(
        customer_phone_number: str,
        assistant_phone_number: str,
        conversation_type: str = "sms",
        status_not_in: Optional[List[str]] = None,
    ) -> Optional[Conversation]:
        """
        Get conversation by phone numbers.

        Args:
            customer_phone_number: Customer's phone number
            assistant_phone_number: Assistant's phone number
            conversation_type: Type of conversation to filter
            status_not_in: List of statuses to exclude

        Returns:
            Optional[Conversation]: Found conversation or None (most recent if multiple exist)
        """
        async with await get_async_db_session() as db:
            query = select(Conversation).where(
                Conversation.customer_phone_number == customer_phone_number,
                Conversation.to_phone_number == assistant_phone_number,
                Conversation.conversation_type == conversation_type,
            )

            if status_not_in:
                query = query.filter(~Conversation.status.in_(status_not_in))

            # Order by most recent first
            query = query.order_by(Conversation.started_at.desc())

            result = await db.execute(query)
            return result.scalars().first()  # Get the first (most recent) result

    @staticmethod
    async def update_conversation_status(
        channel_sid: str, status: str, duration: Optional[int] = None
    ) -> Optional[Conversation]:
        """
        Update conversation status by channel SID.

        Args:
            channel_sid: Channel SID (channel_sid, message_sid, etc)
            status: New status
            duration: Duration in seconds (for calls)

        Returns:
            Optional[Conversation]: Updated conversation or None
        """
        try:
            async with await get_async_db_session() as db:
                query = select(Conversation).where(
                    Conversation.channel_sid == channel_sid
                )
                result = await db.execute(query)
                conversation = result.scalar_one_or_none()

                if not conversation:
                    logger.error(
                        f"Conversation not found for channel SID: {channel_sid}"
                    )
                    return None

                # Update status
                conversation.status = status

                if (
                    status in ["completed", "failed"]
                    and conversation.conversation_type == "call"
                ):
                    conversation.ended_at = datetime.datetime.utcnow()

                    # Calculate duration if not provided
                    if duration is None and conversation.started_at:
                        conversation.duration = int(
                            (
                                datetime.datetime.utcnow() - conversation.started_at
                            ).total_seconds()
                        )
                    else:
                        conversation.duration = duration

                await db.commit()
                await db.refresh(conversation)
                logger.info(
                    f"Updated conversation status to {status} for channel SID: {channel_sid}"
                )

                return conversation
        except SQLAlchemyError as e:
            logger.error(f"Error updating conversation status: {e}")
            raise

    @staticmethod
    async def get_chat_messages_for_conversation(
        conversation_id: int, role: Optional[str] = None
    ) -> List[ChatMessage]:
        """
        Get all chat messages for a conversation.

        Args:
            conversation_id: Conversation ID
            role: Filter by role (user, assistant, system)

        Returns:
            List[ChatMessage]: List of chat messages
        """
        try:
            async with await get_async_db_session() as db:
                query = select(ChatMessage).where(
                    ChatMessage.conversation_id == conversation_id
                )

                if role:
                    query = query.filter(ChatMessage.role == role)

                # Order by message index
                query = query.order_by(ChatMessage.message_index)

                result = await db.execute(query)
                return list(result.scalars().all())
        except SQLAlchemyError as e:
            logger.error(f"Error getting conversation chat messages: {e}")
            return []
