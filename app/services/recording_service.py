"""
Recording service for handling local audio recording during calls.
"""

import os
import logging
import asyncio
import wave
import time
import numpy as np
import audioop
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from io import BytesIO

# Try to import pydub for better audio handling
try:
    from pydub import AudioSegment
    from pydub.utils import make_chunks
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

logger = logging.getLogger(__name__)


class RecordingService:
    """
    Service for recording audio locally during calls.
    Supports various audio formats and configurable recording settings.
    """

    def __init__(
        self,
        call_sid: str,
        enabled: bool = False,
        format: str = "wav",
        sample_rate: int = 8000,
        channels: int = 1,
        record_user: bool = True,
        record_assistant: bool = True,
        record_mixed: bool = True,
        auto_save: bool = True,
        recordings_dir: str = "recordings",
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
            recordings_dir: Directory to save recordings
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
        self.recordings_dir = recordings_dir

        # Validate format
        if self.format not in ["wav", "mp3"]:
            logger.warning(f"Unsupported format '{self.format}', defaulting to 'wav'")
            self.format = "wav"
        
        # Check if MP3 is requested but pydub is not available
        if self.format == "mp3" and not PYDUB_AVAILABLE:
            logger.warning("MP3 format requested but pydub is not available, falling back to WAV")
            self.format = "wav"

        # Recording state
        self.is_recording = False
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Audio buffers for different streams (using pydub AudioSegments for better quality)
        self.user_audio_segments = []
        self.assistant_audio_segments = []
        self.mixed_audio_segments = []

        # Legacy wave file writers (for WAV format only)
        self.user_wave_writer: Optional[wave.Wave_write] = None
        self.assistant_wave_writer: Optional[wave.Wave_write] = None
        self.mixed_wave_writer: Optional[wave.Wave_write] = None

        # File paths
        self.user_audio_path: Optional[str] = None
        self.assistant_audio_path: Optional[str] = None
        self.mixed_audio_path: Optional[str] = None

        # Callbacks
        self.recording_started_callback: Optional[Callable] = None
        self.recording_stopped_callback: Optional[Callable] = None
        self.recording_saved_callback: Optional[Callable] = None

        logger.info(f"Initialized RecordingService for call {call_sid} - enabled: {enabled}, format: {self.format}, pydub_available: {PYDUB_AVAILABLE}")

    def _convert_mulaw_to_audiosegment(self, mulaw_data: bytes) -> Optional[AudioSegment]:
        """
        Convert μ-law encoded audio to AudioSegment using proper μ-law decoding for better quality.
        
        Args:
            mulaw_data: μ-law encoded audio bytes from Twilio/ElevenLabs
            
        Returns:
            AudioSegment: High-quality audio segment, or None if conversion fails
        """
        if not PYDUB_AVAILABLE:
            return None
            
        try:
            # Use Python's built-in audioop for proper μ-law to linear PCM conversion
            # This gives much better quality than manual bit manipulation
            linear_pcm = audioop.ulaw2lin(mulaw_data, 2)  # Convert to 16-bit linear PCM
            
            # Create AudioSegment from the properly decoded PCM data
            audio_segment = AudioSegment(
                data=linear_pcm,
                sample_width=2,  # 16-bit (2 bytes per sample)
                frame_rate=self.sample_rate,
                channels=self.channels
            )
            
            # Optionally upsample to higher quality for better MP3 encoding
            if self.format == "mp3":
                # Upsample to 22.05 kHz for better MP3 quality (matches Twilio recordings)
                audio_segment = audio_segment.set_frame_rate(22050)
            
            return audio_segment
            
        except Exception as e:
            logger.error(f"Error converting μ-law to AudioSegment: {e}")
            return None

    def _convert_mulaw_to_pcm16(self, mulaw_data: bytes) -> bytes:
        """
        Convert μ-law encoded audio to 16-bit PCM using proper μ-law decoding.
        
        Args:
            mulaw_data: μ-law encoded audio bytes from Twilio/ElevenLabs
            
        Returns:
            bytes: 16-bit PCM audio bytes suitable for WAV files
        """
        try:
            # Use Python's built-in audioop for proper μ-law to linear PCM conversion
            # This gives much better quality than manual bit manipulation
            linear_pcm = audioop.ulaw2lin(mulaw_data, 2)  # Convert to 16-bit linear PCM
            return linear_pcm
            
        except Exception as e:
            logger.error(f"Error converting μ-law to PCM16: {e}")
            return b''

    async def start_recording(self) -> bool:
        """
        Start recording audio.

        Returns:
            bool: True if recording started successfully, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Recording not enabled for call {self.call_sid}")
            return False

        if self.is_recording:
            logger.warning(f"Recording already started for call {self.call_sid}")
            return True

        try:
            # Create recordings directory
            call_recordings_dir = os.path.join(self.recordings_dir, self.call_sid)
            os.makedirs(call_recordings_dir, exist_ok=True)

            # Generate file paths
            timestamp = int(time.time())
            if self.should_record_user_audio:
                self.user_audio_path = os.path.join(
                    call_recordings_dir, f"user_{timestamp}.{self.format}"
                )
            if self.should_record_assistant_audio:
                self.assistant_audio_path = os.path.join(
                    call_recordings_dir, f"assistant_{timestamp}.{self.format}"
                )
            if self.should_record_mixed_audio:
                self.mixed_audio_path = os.path.join(
                    call_recordings_dir, f"mixed_{timestamp}.{self.format}"
                )

            # Initialize wave writers for WAV format only
            # For MP3, we'll use pydub AudioSegments and export at the end
            if self.format == "wav":
                if self.should_record_user_audio and self.user_audio_path:
                    self.user_wave_writer = wave.open(self.user_audio_path, "wb")
                    self.user_wave_writer.setnchannels(self.channels)
                    self.user_wave_writer.setsampwidth(2)  # 16-bit audio
                    self.user_wave_writer.setframerate(self.sample_rate)

                if self.should_record_assistant_audio and self.assistant_audio_path:
                    self.assistant_wave_writer = wave.open(self.assistant_audio_path, "wb")
                    self.assistant_wave_writer.setnchannels(self.channels)
                    self.assistant_wave_writer.setsampwidth(2)  # 16-bit audio
                    self.assistant_wave_writer.setframerate(self.sample_rate)

                if self.should_record_mixed_audio and self.mixed_audio_path:
                    self.mixed_wave_writer = wave.open(self.mixed_audio_path, "wb")
                    self.mixed_wave_writer.setnchannels(self.channels)
                    self.mixed_wave_writer.setsampwidth(2)  # 16-bit audio
                    self.mixed_wave_writer.setframerate(self.sample_rate)
            else:
                # For MP3 and other formats, we'll collect AudioSegments and export later
                self.user_audio_segments = []
                self.assistant_audio_segments = []
                self.mixed_audio_segments = []

            # Set recording state
            self.is_recording = True
            self.start_time = datetime.now()

            logger.info(f"Started recording for call {self.call_sid}")

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
        Stop recording audio.

        Returns:
            bool: True if recording stopped successfully, False otherwise
        """
        if not self.is_recording:
            logger.debug(f"Recording not active for call {self.call_sid}")
            return True

        try:
            # Set recording state
            self.is_recording = False
            self.end_time = datetime.now()

            # Close wave writers
            if self.user_wave_writer:
                self.user_wave_writer.close()
                self.user_wave_writer = None

            if self.assistant_wave_writer:
                self.assistant_wave_writer.close()
                self.assistant_wave_writer = None

            if self.mixed_wave_writer:
                self.mixed_wave_writer.close()
                self.mixed_wave_writer = None

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
            audio_data: Raw μ-law encoded audio bytes from Twilio

        Returns:
            bool: True if audio was recorded successfully, False otherwise
        """
        if not self.is_recording or not self.should_record_user_audio:
            return False

        try:
            if self.format == "wav":
                # For WAV format, convert μ-law to PCM and write directly
                pcm_data = self._convert_mulaw_to_pcm16(audio_data)
                if not pcm_data:
                    return False

                # Write to wave file if available
                if self.user_wave_writer:
                    self.user_wave_writer.writeframes(pcm_data)

                # Also write to mixed audio if enabled
                if self.should_record_mixed_audio and self.mixed_wave_writer:
                    self.mixed_wave_writer.writeframes(pcm_data)
            else:
                # For MP3 and other formats, use pydub AudioSegments for better quality
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
            audio_data: Raw μ-law encoded audio bytes from ElevenLabs TTS

        Returns:
            bool: True if audio was recorded successfully, False otherwise
        """
        if not self.is_recording or not self.should_record_assistant_audio:
            return False

        try:
            if self.format == "wav":
                # For WAV format, convert μ-law to PCM and write directly
                pcm_data = self._convert_mulaw_to_pcm16(audio_data)
                if not pcm_data:
                    return False

                # Write to wave file if available
                if self.assistant_wave_writer:
                    self.assistant_wave_writer.writeframes(pcm_data)

                # Also write to mixed audio if enabled (this would need proper mixing logic)
                # For now, we'll just append assistant audio to mixed stream
                # In a real implementation, you'd want to properly mix the audio streams
                if self.should_record_mixed_audio and self.mixed_wave_writer:
                    # Note: This is a simple append, not proper audio mixing
                    # For proper mixing, you'd need to combine the audio samples
                    self.mixed_wave_writer.writeframes(pcm_data)
            else:
                # For MP3 and other formats, use pydub AudioSegments for better quality
                audio_segment = self._convert_mulaw_to_audiosegment(audio_data)
                if audio_segment:
                    self.assistant_audio_segments.append(audio_segment)
                    
                    # Also add to mixed audio if enabled
                    # Note: This is sequential append, not proper mixing
                    # For proper mixing, you'd overlay the audio segments
                    if self.should_record_mixed_audio:
                        self.mixed_audio_segments.append(audio_segment)

            return True

        except Exception as e:
            logger.error(f"Error recording assistant audio for call {self.call_sid}: {e}")
            return False

    async def save_recordings(self) -> Dict[str, str]:
        """
        Save all recordings and return file paths.

        Returns:
            Dict[str, str]: Dictionary mapping recording type to file path
        """
        saved_files = {}

        try:
            # For WAV format, files are already saved via wave writers
            if self.format == "wav":
                if self.user_audio_path and os.path.exists(self.user_audio_path):
                    saved_files["user"] = self.user_audio_path
                    logger.info(f"Saved user recording: {self.user_audio_path}")

                if self.assistant_audio_path and os.path.exists(self.assistant_audio_path):
                    saved_files["assistant"] = self.assistant_audio_path
                    logger.info(f"Saved assistant recording: {self.assistant_audio_path}")

                if self.mixed_audio_path and os.path.exists(self.mixed_audio_path):
                    saved_files["mixed"] = self.mixed_audio_path
                    logger.info(f"Saved mixed recording: {self.mixed_audio_path}")
            else:
                # For MP3 and other formats, export AudioSegments using pydub
                if PYDUB_AVAILABLE:
                    # Export user audio
                    if self.user_audio_segments and self.user_audio_path:
                        combined_user = AudioSegment.empty()
                        for segment in self.user_audio_segments:
                            combined_user += segment
                        
                        # Export with high quality settings for MP3
                        export_params = {"format": self.format}
                        if self.format == "mp3":
                            export_params.update({
                                "bitrate": "320k",  # Highest quality bitrate
                                "parameters": [
                                    "-q:a", "0",  # Highest quality
                                    "-ar", "22050",  # Sample rate to match Twilio
                                    "-ac", "1",  # Mono
                                    "-compression_level", "0"  # No compression
                                ]
                            })
                        
                        combined_user.export(self.user_audio_path, **export_params)
                        saved_files["user"] = self.user_audio_path
                        logger.info(f"Exported user recording: {self.user_audio_path}")

                    # Export assistant audio
                    if self.assistant_audio_segments and self.assistant_audio_path:
                        combined_assistant = AudioSegment.empty()
                        for segment in self.assistant_audio_segments:
                            combined_assistant += segment
                        
                        export_params = {"format": self.format}
                        if self.format == "mp3":
                            export_params.update({
                                "bitrate": "320k",
                                "parameters": [
                                    "-q:a", "0",
                                    "-ar", "22050",
                                    "-ac", "1",
                                    "-compression_level", "0"
                                ]
                            })
                        
                        combined_assistant.export(self.assistant_audio_path, **export_params)
                        saved_files["assistant"] = self.assistant_audio_path
                        logger.info(f"Exported assistant recording: {self.assistant_audio_path}")

                    # Export mixed audio
                    if self.mixed_audio_segments and self.mixed_audio_path:
                        combined_mixed = AudioSegment.empty()
                        for segment in self.mixed_audio_segments:
                            combined_mixed += segment
                        
                        export_params = {"format": self.format}
                        if self.format == "mp3":
                            export_params.update({
                                "bitrate": "320k",
                                "parameters": [
                                    "-q:a", "0",
                                    "-ar", "22050",
                                    "-ac", "1",
                                    "-compression_level", "0"
                                ]
                            })
                        
                        combined_mixed.export(self.mixed_audio_path, **export_params)
                        saved_files["mixed"] = self.mixed_audio_path
                        logger.info(f"Exported mixed recording: {self.mixed_audio_path}")

            # Call callback if set
            if self.recording_saved_callback:
                try:
                    await self.recording_saved_callback(self.call_sid, saved_files)
                except Exception as e:
                    logger.error(f"Error in recording saved callback: {e}")

            return saved_files

        except Exception as e:
            logger.error(f"Error saving recordings for call {self.call_sid}: {e}")
            return {}

    async def cleanup(self) -> None:
        """
        Clean up recording resources.
        """
        try:
            # Stop recording if still active
            if self.is_recording:
                await self.stop_recording()

            # Close any remaining wave writers
            if self.user_wave_writer:
                self.user_wave_writer.close()
                self.user_wave_writer = None

            if self.assistant_wave_writer:
                self.assistant_wave_writer.close()
                self.assistant_wave_writer = None

            if self.mixed_wave_writer:
                self.mixed_wave_writer.close()
                self.mixed_wave_writer = None

            logger.info(f"Cleaned up recording service for call {self.call_sid}")

        except Exception as e:
            logger.error(f"Error cleaning up recording service for call {self.call_sid}: {e}")

    def get_recording_info(self) -> Dict[str, Any]:
        """
        Get information about the current recording session.

        Returns:
            Dict[str, Any]: Recording information
        """
        duration = None
        if self.start_time:
            end_time = self.end_time or datetime.now()
            duration = (end_time - self.start_time).total_seconds()

        return {
            "call_sid": self.call_sid,
            "enabled": self.enabled,
            "is_recording": self.is_recording,
            "format": self.format,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "record_user_audio": self.should_record_user_audio,
            "record_assistant_audio": self.should_record_assistant_audio,
            "record_mixed_audio": self.should_record_mixed_audio,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": duration,
            "user_audio_path": self.user_audio_path,
            "assistant_audio_path": self.assistant_audio_path,
            "mixed_audio_path": self.mixed_audio_path,
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
            recording_saved_callback: Called when recordings are saved
        """
        self.recording_started_callback = recording_started_callback
        self.recording_stopped_callback = recording_stopped_callback
        self.recording_saved_callback = recording_saved_callback 