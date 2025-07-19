"""
Resemble AI TTS service implementation.
"""

# pylint: disable=too-many-ancestors,logging-fstring-interpolation,broad-exception-caught
import os
import json
import logging
import asyncio
import base64
from typing import Optional, Dict, Any, Callable
import websockets

from .tts_base import BaseTTSService, TTSOptions, TTSProvider, VoiceInfo, ModelInfo

# Configure logging
logger = logging.getLogger(__name__)


class ResembleTTSService(BaseTTSService):
    """
    Resemble AI TTS service implementation using WebSocket streaming.
    Manages WebSocket connection and text buffering for natural speech synthesis.
    """

    # Available voices mapping - empty by default since users must provide their own voice UUIDs
    # from their Resemble account
    _available_voices = {}

    # Available models - Resemble doesn't have multiple models like ElevenLabs, but we maintain the interface
    _available_models = {
        "default": ModelInfo(
            id="default",
            name="default",
            description="Resemble AI default synthesis model",
        ),
    }

    def __init__(
        self,
        call_sid: Optional[str] = None,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        project_uuid: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Resemble AI TTS service.

        Args:
            call_sid: The unique identifier for this call
            api_key: Optional custom API key for this call
            voice_id: The voice UUID to use
            model_id: The model ID to use (maintained for interface compatibility)
            project_uuid: The project UUID for Resemble AI (required for WebSocket)
            **kwargs: Additional parameters for compatibility
        """
        # Get API key from parameter or environment variable
        api_key = api_key or os.getenv("RESEMBLE_API_KEY")
        if not api_key:
            raise ValueError(
                "RESEMBLE_API_KEY environment variable is not set and no API key provided"
            )

        super().__init__(call_sid=call_sid, api_key=api_key)

        # Resemble WebSocket URL (correct endpoint from documentation)
        self.ws_url = "wss://websocket.cluster.resemble.ai/stream"

        # Sentence-ending punctuation
        self.sentence_endings = {".", "!", "?", ":", ";"}

        # WebSocket connection
        self.websocket = None
        self.buffer = ""

        # Store TTS settings - Fixed for Twilio compatibility
        self.voice_id = voice_id or self.get_voice_id("default_voice")
        self.model_id = model_id or self.get_model_id("default")
        self.project_uuid = project_uuid or os.getenv("RESEMBLE_PROJECT_UUID")

        # Fixed audio format for Twilio compatibility
        self.sample_rate = 8000  # Required by Twilio
        self.precision = "MULAW"  # G.711 μ-law encoding for Twilio
        self.output_format = "wav"  # Use provided format or default to wav
        self.binary_response = True  # We want binary audio data for streaming

        # Request tracking
        self.request_id_counter = 0

        if not self.project_uuid:
            logger.warning(
                "No project_uuid provided. WebSocket streaming requires a valid project UUID."
            )

        logger.info(
            f"ResembleTTSService initialized for call {call_sid} with Twilio-compatible format (MULAW, 8kHz)"
        )

    @property
    def provider(self) -> TTSProvider:
        """Get the TTS provider type."""
        return TTSProvider.RESEMBLE

    async def start_session(
        self,
        options: Optional[TTSOptions] = None,
        audio_callback: Optional[Callable[[bytes, bool, Dict[str, Any]], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Start a new TTS session using Resemble AI WebSocket streaming.

        Args:
            options: Configuration options for TTS (optional, will use instance defaults if not provided)
            audio_callback: Callback function for handling audio data
            metadata: Additional metadata for the session

        Returns:
            bool: Whether the session was started successfully
        """
        if not self.call_sid:
            logger.error("Cannot start TTS session without call_sid")
            return False

        if not self.project_uuid:
            logger.error("Cannot start WebSocket session without project_uuid")
            return False

        try:
            # Use options from parameters or instance settings
            session_options = options or TTSOptions(
                voice_id=self.voice_id,
                model_id=self.model_id,
                sample_rate=self.sample_rate,
                output_format=self.output_format,
            )

            # Store options in metadata for reconnection
            session_metadata = metadata or {}
            session_metadata.update(
                {
                    "voice_id": session_options.voice_id,
                    "model_id": session_options.model_id,
                    "project_uuid": self.project_uuid,
                    "sample_rate": session_options.sample_rate,
                    "precision": self.precision,
                    "output_format": session_options.output_format,
                    "binary_response": self.binary_response,
                }
            )

            # Store callback and metadata
            self.audio_callback = audio_callback
            self.metadata = session_metadata

            # Connect to Resemble WebSocket
            # Note: WebSocket API requires Business plan or higher
            headers = {"Authorization": f"Bearer {self.api_key}"}

            logger.info(f"Connecting to Resemble WebSocket: {self.ws_url}")

            # Create connection
            self.websocket = await websockets.connect(
                self.ws_url, additional_headers=headers
            )

            # Update session state
            self.is_connected = True

            # Start a background task to receive audio chunks
            asyncio.create_task(self._receive_audio_chunks())

            logger.info(f"Started TTS session for call {self.call_sid}")
            return True

        except Exception as e:
            logger.error(
                f"Error starting TTS session for call {self.call_sid}: {e}",
                exc_info=True,
            )
            return False

    async def _receive_audio_chunks(self) -> None:
        """
        Continuously receive audio chunks from the WebSocket.
        """
        if not self.websocket or not self.is_connected:
            return

        try:
            async for message in self.websocket:
                try:
                    if isinstance(message, bytes):
                        # Binary response format
                        if self.audio_callback:
                            await self.audio_callback(
                                message,
                                False,  # Not final
                                {"call_sid": self.call_sid, "format": "binary"},
                            )
                    else:
                        # JSON response format
                        data = json.loads(message)

                        if data.get("type") == "audio":
                            # Extract audio data
                            audio_content = data.get("audio_content")
                            if audio_content and self.audio_callback:
                                try:
                                    # Decode the base64-encoded audio data
                                    audio_data = base64.b64decode(audio_content)

                                    # Only process if we actually have audio data
                                    if audio_data:
                                        await self.audio_callback(
                                            audio_data,
                                            False,  # Not final
                                            {
                                                "call_sid": self.call_sid,
                                                "request_id": data.get("request_id"),
                                                "sample_rate": data.get("sample_rate"),
                                                "audio_timestamps": data.get(
                                                    "audio_timestamps"
                                                ),
                                            },
                                        )
                                except (ValueError, TypeError) as decode_error:
                                    logger.warning(
                                        f"Failed to decode audio data for call {self.call_sid}: {decode_error}"
                                    )
                                    continue

                        elif data.get("type") == "audio_end":
                            # Handle the end of audio for a specific request
                            request_id = data.get("request_id")
                            logger.info(f"Audio stream ended for request {request_id}")

                            # Signal final
                            if self.audio_callback:
                                await self.audio_callback(
                                    b"",  # Empty data for final signal
                                    True,  # Final
                                    {
                                        "call_sid": self.call_sid,
                                        "request_id": request_id,
                                    },
                                )

                        elif data.get("type") == "error":
                            error_msg = data.get("message", "Unknown error")
                            error_name = data.get("error_name", "Unknown")
                            status_code = data.get("status_code")
                            logger.error(
                                f"Resemble WebSocket error for call {self.call_sid}: {error_name} - {error_msg} (Status: {status_code})"
                            )

                except json.JSONDecodeError:
                    logger.warning(
                        f"Received non-JSON message from Resemble: {message[:100] if isinstance(message, str) else 'binary data'}"
                    )
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}", exc_info=True)

        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"WebSocket connection closed for call {self.call_sid}")
        except Exception as e:
            logger.error(f"Error receiving audio chunks: {e}", exc_info=True)

    async def process_text(self, text: str, force_flush: bool = False) -> bool:
        """
        Process text and convert to speech when appropriate.

        Args:
            text: The text to process
            force_flush: Whether to force immediate speech conversion

        Returns:
            bool: Whether the text was processed successfully
        """
        if not self.is_connected or not self.websocket:
            logger.warning(f"TTS session not connected for call {self.call_sid}")
            return False

        try:
            # Check for flush tag
            if "<flush/>" in text:
                # Split text at flush tag
                parts = text.split("<flush/>")
                # Add all parts except the last one to buffer
                self.buffer += "".join(parts[:-1])
                # Convert buffered text to speech
                await self._convert_to_speech()
                # Add the last part to buffer
                self.buffer += parts[-1]
                # Check if we need to convert the last part too
                if force_flush or self._should_convert(self.buffer):
                    await self._convert_to_speech()
            else:
                # Add text to buffer
                self.buffer += text

                # Check if we should convert to speech
                if force_flush or self._should_convert(self.buffer):
                    await self._convert_to_speech()

            return True

        except Exception as e:
            logger.error(
                f"Error processing text for call {self.call_sid}: {e}", exc_info=True
            )
            return False

    def _should_convert(self, text: str) -> bool:
        """
        Determine if the buffered text should be converted to speech.

        Args:
            text: The buffered text to check

        Returns:
            bool: Whether the text should be converted
        """
        # Convert if the text ends with sentence-ending punctuation
        return any(text.rstrip().endswith(p) for p in self.sentence_endings)

    async def _convert_to_speech(self) -> None:
        """
        Convert buffered text to speech and send to Resemble WebSocket.
        """
        if not self.buffer.strip():
            return

        try:
            # Get TTS settings from metadata
            metadata = self.metadata

            # Increment request ID
            self.request_id_counter += 1
            request_id = self.request_id_counter

            # Prepare the message for Resemble WebSocket API
            message = {
                "voice_uuid": metadata.get("voice_id", self.voice_id),
                "project_uuid": metadata.get("project_uuid", self.project_uuid),
                "data": self.buffer,
                "request_id": request_id,
                "binary_response": metadata.get(
                    "binary_response", self.binary_response
                ),
                # "output_format": metadata.get("output_format", self.output_format),
                "sample_rate": metadata.get("sample_rate", self.sample_rate),
                "precision": metadata.get("precision", self.precision),
                "no_audio_header": True,
            }

            # Send the text to Resemble
            await self.websocket.send(json.dumps(message))
            logger.info(
                f"Sent TTS request {request_id} with {len(self.buffer)} characters"
            )

            # Clear the buffer
            self.buffer = ""

        except Exception as e:
            logger.error(
                f"Error converting text to speech for call {self.call_sid}: {e}",
                exc_info=True,
            )
            # Attempt to reconnect
            await self._reconnect_session()

    async def _reconnect_session(self) -> bool:
        """
        Attempt to reconnect a TTS session.

        Returns:
            bool: Whether the reconnection was successful
        """
        try:
            # Close existing connection if any
            if self.websocket:
                await self.websocket.close()

            # Reconnect with same settings
            headers = {"Authorization": f"Bearer {self.api_key}"}

            websocket = await websockets.connect(
                self.ws_url, additional_headers=headers
            )

            # Update session state
            self.websocket = websocket
            self.is_connected = True

            # Start a new background task to receive audio chunks
            asyncio.create_task(self._receive_audio_chunks())

            logger.info(f"Reconnected TTS session for call {self.call_sid}")
            return True

        except Exception as e:
            logger.error(
                f"Error reconnecting TTS session for call {self.call_sid}: {e}",
                exc_info=True,
            )
            self.is_connected = False
            return False

    async def stop_synthesis(self) -> None:
        """
        Stop ongoing TTS synthesis.
        This is used when an interruption is detected to immediately stop the AI from speaking.
        """
        try:
            # Clear any buffered text
            self.buffer = ""

            # For Resemble WebSocket, we can close and reconnect to ensure clean state
            if self.websocket:
                try:
                    await self.websocket.close()
                except Exception:
                    pass  # Ignore errors when closing

            # Reconnect to ensure clean state for future synthesis
            await self._reconnect_session()

            logger.info(f"Stopped TTS synthesis for call {self.call_sid}")

        except Exception as e:
            logger.error(
                f"Error stopping TTS synthesis for call {self.call_sid}: {e}",
                exc_info=True,
            )
            # Attempt to reconnect in case of errors
            await self._reconnect_session()

    async def end_session(self) -> None:
        """
        End a TTS session and clean up resources.
        """
        try:
            # Convert any remaining buffered text
            if self.buffer.strip():
                await self._convert_to_speech()

            # Close WebSocket connection
            if self.websocket:
                await self.websocket.close()
                self.websocket = None

            # Reset state
            self.is_connected = False
            self.buffer = ""

            logger.info(f"Ended TTS session for call {self.call_sid}")

        except Exception as e:
            logger.error(
                f"Error ending TTS session for call {self.call_sid}: {e}", exc_info=True
            )

    @classmethod
    def get_available_voices(cls) -> Dict[str, VoiceInfo]:
        """
        Get available voices for Resemble AI.

        Returns:
            Dict[str, VoiceInfo]: Mapping of voice names to voice information
        """
        return cls._available_voices.copy()

    @classmethod
    def get_available_models(cls) -> Dict[str, ModelInfo]:
        """
        Get available models for Resemble AI.

        Returns:
            Dict[str, ModelInfo]: Mapping of model names to model information
        """
        return cls._available_models.copy()

    @classmethod
    def get_voice_id(cls, voice_name: str) -> str:
        """
        Get a voice ID by name.

        Args:
            voice_name: The voice UUID provided by the user

        Returns:
            str: The voice UUID (always returns the input as-is for Resemble)
        """
        # For Resemble, users always provide their own voice UUIDs
        # so we return the voice_name as-is
        return voice_name

    @classmethod
    def get_model_id(cls, model_name: str) -> str:
        """
        Get a model ID by name.

        Args:
            model_name: The name of the model to use

        Returns:
            str: The model ID
        """
        model_name = model_name.lower()
        if model_name in cls._available_models:
            return cls._available_models[model_name].id
        else:
            return cls._available_models["default"].id
