"""
Recording service for handling S3-based audio recording during calls.
"""

import logging
import asyncio
import audioop
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from io import BytesIO

# Try to import pydub for better audio handling
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

from app.services.s3_service import S3Service

logger = logging.getLogger(__name__)


class RecordingService:
    """
    Service for recording audio to S3 during calls.
    Supports various audio formats and configurable recording settings.
    """

    def __init__(
        self,
        call_sid: str,
        enabled: bool = False,
        format: str = "mp3",
        sample_rate: int = 8000,
        channels: int = 1,
        record_user: bool = True,
        record_assistant: bool = True,
        record_mixed: bool = True,
        auto_save: bool = True,
        s3_service: Optional[S3Service] = None,
    ):
        """
        Initialize the recording service.

        Args:
            call_sid: The call SID for this recording session
            enabled: Whether recording is enabled
            format: Audio format ("wav", "mp3")
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            record_user: Whether to record user audio
            record_assistant: Whether to record assistant audio
            record_mixed: Whether to record mixed audio (both user and assistant)
            auto_save: Whether to automatically save recordings when call ends
            s3_service: S3Service instance (will create default if not provided)
        """
        self.call_sid = call_sid
        self.enabled = enabled
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.should_record_user_audio = record_user
        self.should_record_assistant_audio = record_assistant
        self.should_record_mixed_audio = record_mixed
        self.auto_save = auto_save

        # Validate format
        if self.format not in ["wav", "mp3"]:
            logger.warning(f"Unsupported format '{self.format}', defaulting to 'mp3'")
            self.format = "mp3"
        
        # Check if MP3 is requested but pydub is not available
        if self.format == "mp3" and not PYDUB_AVAILABLE:
            logger.warning("MP3 format requested but pydub is not available, falling back to WAV")
            self.format = "wav"

        # Initialize S3 service
        try:
            self.s3_service = s3_service or S3Service.create_default_instance()
        except Exception as e:
            logger.error(f"Failed to initialize S3 service: {e}")
            self.enabled = False
            self.s3_service = None

        # Recording state
        self.is_recording = False
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Audio buffers for different streams
        self.user_audio_segments = []
        self.assistant_audio_segments = []
        self.mixed_audio_segments = []

        # Callbacks
        self.recording_started_callback: Optional[Callable] = None
        self.recording_stopped_callback: Optional[Callable] = None
        self.recording_saved_callback: Optional[Callable] = None

        logger.info(f"Initialized S3 RecordingService for call {call_sid} - enabled: {enabled}, format: {self.format}, s3_available: {self.s3_service is not None}")

    def _convert_mulaw_to_audiosegment(self, mulaw_data: bytes) -> Optional[AudioSegment]:
        """
        Convert μ-law encoded audio data to AudioSegment.
        
        Args:
            mulaw_data: Raw μ-law encoded audio bytes
            
        Returns:
            Optional[AudioSegment]: Converted audio segment or None if conversion fails
        """
        try:
            if not mulaw_data:
                return None
                
            # Convert μ-law to 16-bit PCM using Python's built-in audioop
            pcm_data = audioop.ulaw2lin(mulaw_data, 2)  # 2 bytes per sample (16-bit)
            
            # Create AudioSegment from PCM data
            audio_segment = AudioSegment(
                data=pcm_data,
                sample_width=2,  # 16-bit = 2 bytes
                frame_rate=self.sample_rate,
                channels=self.channels
            )
            
            # For MP3 format, upsample to 22.05kHz for better quality
            if self.format == "mp3" and audio_segment.frame_rate < 22050:
                audio_segment = audio_segment.set_frame_rate(22050)
            
            return audio_segment
            
        except Exception as e:
            logger.error(f"Error converting μ-law to AudioSegment: {e}")
            return None

    async def start_recording(self) -> bool:
        """
        Start recording audio streams.

        Returns:
            bool: True if recording started successfully, False otherwise
        """
        if not self.enabled:
            logger.info(f"Recording disabled for call {self.call_sid}")
            return False

        if not self.s3_service:
            logger.error(f"S3 service not available for call {self.call_sid}")
            return False

        if self.is_recording:
            logger.warning(f"Recording already started for call {self.call_sid}")
            return True

        try:
            # Initialize audio buffers
            self.user_audio_segments = []
            self.assistant_audio_segments = []
            self.mixed_audio_segments = []

            # Set recording state
            self.is_recording = True
            self.start_time = datetime.utcnow()

            logger.info(f"Started S3 recording for call {self.call_sid}")

            # Call callback if set
            if self.recording_started_callback:
                try:
                    await self.recording_started_callback(self.call_sid)
                except Exception as e:
                    logger.error(f"Error in recording started callback: {e}")

            return True

        except Exception as e:
            logger.error(f"Error starting recording for call {self.call_sid}: {e}")
            return False

    async def stop_recording(self) -> bool:
        """
        Stop recording and optionally save to S3.

        Returns:
            bool: True if recording stopped successfully, False otherwise
        """
        if not self.is_recording:
            logger.warning(f"Recording not active for call {self.call_sid}")
            return True

        try:
            self.is_recording = False
            self.end_time = datetime.utcnow()

            logger.info(f"Stopped recording for call {self.call_sid}")

            # Call callback if set
            if self.recording_stopped_callback:
                try:
                    await self.recording_stopped_callback(self.call_sid)
                except Exception as e:
                    logger.error(f"Error in recording stopped callback: {e}")

            # Auto-save if enabled
            if self.auto_save:
                await self.save_recordings()

            return True

        except Exception as e:
            logger.error(f"Error stopping recording for call {self.call_sid}: {e}")
            return False

    async def record_user_audio(self, audio_data: bytes) -> bool:
        """
        Record user audio data.

        Args:
            audio_data: Raw audio data (μ-law encoded)

        Returns:
            bool: True if audio was recorded successfully, False otherwise
        """
        if not self.is_recording or not self.should_record_user_audio:
            return False

        try:
            # Convert μ-law to AudioSegment
            audio_segment = self._convert_mulaw_to_audiosegment(audio_data)
            if audio_segment:
                self.user_audio_segments.append(audio_segment)
                
                # Also add to mixed audio if enabled
                if self.should_record_mixed_audio:
                    self.mixed_audio_segments.append(audio_segment)

            return True

        except Exception as e:
            logger.error(f"Error recording user audio for call {self.call_sid}: {e}")
            return False

    async def record_assistant_audio(self, audio_data: bytes) -> bool:
        """
        Record assistant audio data.

        Args:
            audio_data: Raw audio data (μ-law encoded)

        Returns:
            bool: True if audio was recorded successfully, False otherwise
        """
        if not self.is_recording or not self.should_record_assistant_audio:
            return False

        try:
            # Convert μ-law to AudioSegment
            audio_segment = self._convert_mulaw_to_audiosegment(audio_data)
            if audio_segment:
                self.assistant_audio_segments.append(audio_segment)
                
                # Also add to mixed audio if enabled
                if self.should_record_mixed_audio:
                    self.mixed_audio_segments.append(audio_segment)

            return True

        except Exception as e:
            logger.error(f"Error recording assistant audio for call {self.call_sid}: {e}")
            return False

    async def save_recordings(self) -> Dict[str, Dict[str, Any]]:
        """
        Save all recordings to S3 and return file information.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping recording type to file info
        """
        if not self.s3_service:
            logger.error(f"S3 service not available for saving recordings for call {self.call_sid}")
            return {}

        saved_files = {}

        try:
            # Save user audio
            if self.user_audio_segments and self.should_record_user_audio:
                file_info = await self._save_audio_segments(
                    self.user_audio_segments, "user"
                )
                if file_info:
                    saved_files["user"] = file_info

            # Save assistant audio
            if self.assistant_audio_segments and self.should_record_assistant_audio:
                file_info = await self._save_audio_segments(
                    self.assistant_audio_segments, "assistant"
                )
                if file_info:
                    saved_files["assistant"] = file_info

            # Save mixed audio
            if self.mixed_audio_segments and self.should_record_mixed_audio:
                file_info = await self._save_audio_segments(
                    self.mixed_audio_segments, "mixed"
                )
                if file_info:
                    saved_files["mixed"] = file_info

            logger.info(f"Saved {len(saved_files)} recordings to S3 for call {self.call_sid}")

            # Call callback if set
            if self.recording_saved_callback and saved_files:
                try:
                    await self.recording_saved_callback(self.call_sid, saved_files)
                except Exception as e:
                    logger.error(f"Error in recording saved callback: {e}")

            return saved_files

        except Exception as e:
            logger.error(f"Error saving recordings for call {self.call_sid}: {e}")
            return {}

    async def _save_audio_segments(
        self, segments: list, recording_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Save audio segments to S3.

        Args:
            segments: List of AudioSegment objects
            recording_type: Type of recording (user, assistant, mixed)

        Returns:
            Optional[Dict[str, Any]]: File information or None if failed
        """
        try:
            if not segments:
                return None

            # Combine all segments
            combined_audio = AudioSegment.empty()
            for segment in segments:
                combined_audio += segment

            # Export to bytes
            buffer = BytesIO()
            export_params = {"format": self.format}
            
            if self.format == "mp3":
                export_params.update({
                    "bitrate": "320k",  # Highest quality bitrate
                    "parameters": [
                        "-q:a", "0",  # Highest quality
                        "-ar", str(combined_audio.frame_rate),
                        "-ac", str(combined_audio.channels),
                    ]
                })

            combined_audio.export(buffer, **export_params)
            audio_data = buffer.getvalue()
            buffer.close()

            # Calculate duration and file size
            duration = len(combined_audio) / 1000.0  # Convert ms to seconds
            file_size = len(audio_data)

            # Prepare metadata
            metadata = {
                "duration": str(duration),
                "file_size": str(file_size),
                "sample_rate": str(combined_audio.frame_rate),
                "channels": str(combined_audio.channels),
                "segments_count": str(len(segments)),
            }

            # Upload to S3
            s3_key, s3_url = await self.s3_service.upload_audio_file(
                audio_data=audio_data,
                call_sid=self.call_sid,
                recording_type=recording_type,
                format=self.format,
                metadata=metadata,
            )

            file_info = {
                "s3_key": s3_key,
                "s3_url": s3_url,
                "duration": duration,
                "file_size": file_size,
                "sample_rate": combined_audio.frame_rate,
                "channels": combined_audio.channels,
                "format": self.format,
                "recording_type": recording_type,
                "uploaded_at": datetime.utcnow().isoformat(),
            }

            logger.info(f"Saved {recording_type} recording to S3: {s3_key}")
            return file_info

        except Exception as e:
            logger.error(f"Error saving {recording_type} recording to S3: {e}")
            return None

    async def cleanup(self) -> None:
        """
        Clean up resources and clear audio buffers.
        """
        try:
            # Clear audio buffers
            self.user_audio_segments.clear()
            self.assistant_audio_segments.clear()
            self.mixed_audio_segments.clear()

            # Reset state
            self.is_recording = False
            self.start_time = None
            self.end_time = None

            logger.info(f"Cleaned up recording service for call {self.call_sid}")

        except Exception as e:
            logger.error(f"Error during cleanup for call {self.call_sid}: {e}")

    def get_recording_info(self) -> Dict[str, Any]:
        """
        Get current recording information.

        Returns:
            Dict[str, Any]: Recording status and statistics
        """
        return {
            "call_sid": self.call_sid,
            "enabled": self.enabled,
            "is_recording": self.is_recording,
            "format": self.format,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "user_segments": len(self.user_audio_segments),
            "assistant_segments": len(self.assistant_audio_segments),
            "mixed_segments": len(self.mixed_audio_segments),
            "s3_available": self.s3_service is not None,
            "recording_types": {
                "user": self.should_record_user_audio,
                "assistant": self.should_record_assistant_audio,
                "mixed": self.should_record_mixed_audio,
            },
        }

    def set_callbacks(
        self,
        recording_started_callback: Optional[Callable] = None,
        recording_stopped_callback: Optional[Callable] = None,
        recording_saved_callback: Optional[Callable] = None,
    ) -> None:
        """
        Set callback functions for recording events.

        Args:
            recording_started_callback: Called when recording starts
            recording_stopped_callback: Called when recording stops
            recording_saved_callback: Called when recordings are saved to S3
        """
        self.recording_started_callback = recording_started_callback
        self.recording_stopped_callback = recording_stopped_callback
        self.recording_saved_callback = recording_saved_callback 