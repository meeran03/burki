"""
Deepgram TTS service implementation using WebSocket API.
"""

# pylint: disable=too-many-ancestors,logging-fstring-interpolation,broad-exception-caught
import os
import json
import logging
import asyncio
from typing import Optional, Dict, Any, Callable
import websockets

from .tts_base import BaseTTSService, TTSOptions, TTSProvider, VoiceInfo, ModelInfo

# Configure logging
logger = logging.getLogger(__name__)


class DeepgramTTSService(BaseTTSService):
    """
    Deepgram TTS service implementation using WebSocket API.
    Manages WebSocket connection and text buffering for natural speech synthesis.
    """

    # Available voices mapping (name -> full model id) based on Deepgram API
    _available_voices = {
        "asteria": VoiceInfo(
            id="aura-asteria-en",
            name="asteria",
            gender="female",
            language="en-US",
            description="Warm and expressive female voice",
        ),
        "thalia": VoiceInfo(
            id="aura-2-thalia-en",
            name="thalia",
            gender="female",
            language="en-US",
            description="Clear and professional female voice",
        ),
        "orion": VoiceInfo(
            id="aura-orion-en",
            name="orion",
            gender="male",
            language="en-US",
            description="Deep and authoritative male voice",
        ),
        "helios": VoiceInfo(
            id="aura-helios-en",
            name="helios",
            gender="male",
            language="en-US",
            description="Bright and energetic male voice",
        ),
        "luna": VoiceInfo(
            id="aura-luna-en",
            name="luna",
            gender="female",
            language="en-US",
            description="Soft and melodic female voice",
        ),
        "stella": VoiceInfo(
            id="aura-stella-en",
            name="stella",
            gender="female",
            language="en-US",
            description="Bright and clear female voice",
        ),
        "athena": VoiceInfo(
            id="aura-athena-en",
            name="athena",
            gender="female",
            language="en-US",
            description="Intelligent and clear female voice",
        ),
        "hera": VoiceInfo(
            id="aura-hera-en",
            name="hera",
            gender="female",
            language="en-US",
            description="Regal and authoritative female voice",
        ),
        "perseus": VoiceInfo(
            id="aura-2-perseus-en",
            name="perseus",
            gender="male",
            language="en-US",
            description="Strong and heroic male voice",
        ),
        "apollo": VoiceInfo(
            id="aura-2-apollo-en",
            name="apollo",
            gender="male",
            language="en-US",
            description="Musical and expressive male voice",
        ),
    }

    # Available models - for Deepgram, these are really voice families
    _available_models = {
        "aura": ModelInfo(
            id="aura", name="aura", description="Deepgram's Aura TTS model family"
        ),
        "aura-2": ModelInfo(
            id="aura-2",
            name="aura-2",
            description="Deepgram's advanced Aura 2 TTS model family",
        ),
    }

    def __init__(
        self,
        call_sid: Optional[str] = None,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        encoding: str = "mulaw",
        sample_rate: int = 8000,
        **kwargs,
    ):
        """
        Initialize the Deepgram TTS service.

        Args:
            call_sid: The unique identifier for this call
            api_key: Optional custom API key for this call
            voice_id: The voice ID to use (default is asteria)
            model_id: The model ID to use (for compatibility, but voice_id takes precedence)
            encoding: Audio encoding format (default is mulaw for Twilio)
            sample_rate: Sample rate in Hz (default is 8000 for Twilio)
            **kwargs: Additional configuration parameters
        """
        # Get API key from parameter or environment variable
        api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPGRAM_API_KEY environment variable is not set and no API key provided"
            )

        super().__init__(call_sid=call_sid, api_key=api_key)

        # Deepgram WebSocket base URL
        self.ws_base_url = "wss://api.deepgram.com/v1/speak"

        # Sentence-ending punctuation
        self.sentence_endings = {".", "!", "?", ":", ";"}

        # WebSocket connection
        self.websocket = None
        self.buffer = ""

        # Store TTS settings - for Deepgram, the voice_id IS the model
        self.voice_id = voice_id or self.get_voice_id("asteria")
        # For Deepgram, the model parameter is actually the full voice ID
        self.model_id = self.voice_id  # Use voice_id as model_id for Deepgram API
        self.encoding = encoding
        self.sample_rate = sample_rate

        logger.info(
            f"DeepgramTTSService initialized for call {call_sid} with model {self.model_id}"
        )

    @property
    def provider(self) -> TTSProvider:
        """Get the TTS provider type."""
        return TTSProvider.DEEPGRAM

    async def start_session(
        self,
        options: Optional[TTSOptions] = None,
        audio_callback: Optional[Callable[[bytes, bool, Dict[str, Any]], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Start a new TTS session.

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

        try:
            # Use options from parameters or instance settings
            session_options = options or TTSOptions(
                voice_id=self.voice_id,
                model_id=self.model_id,
                sample_rate=self.sample_rate,
            )

            # Validate voice ID before using it
            if not self._validate_voice_id(session_options.voice_id):
                logger.warning(
                    f"Invalid voice ID '{session_options.voice_id}', falling back to asteria"
                )
                session_options.voice_id = self.get_voice_id("asteria")

            # For Deepgram, the model parameter should be the full voice ID
            model_to_use = session_options.voice_id

            # Store options in metadata for reconnection
            session_metadata = metadata or {}
            session_metadata.update(
                {
                    "voice_id": session_options.voice_id,
                    "model_id": model_to_use,  # Store the full voice ID as model_id
                    "encoding": self.encoding,
                    "sample_rate": session_options.sample_rate,
                    "output_format": session_options.output_format,
                }
            )

            # Store callback and metadata
            self.audio_callback = audio_callback
            self.metadata = session_metadata

            # Connect to Deepgram WebSocket
            headers = {"Authorization": f"Token {self.api_key}"}

            # Build query parameters - use the full voice ID as the model parameter
            query_params = [
                f"model={model_to_use}",
                f"encoding={self.encoding}",
                f"sample_rate={session_options.sample_rate}",
            ]

            # Construct the final WebSocket URL
            ws_url = f"{self.ws_base_url}?{'&'.join(query_params)}"

            logger.info(f"Connecting to Deepgram with URL: {ws_url}")

            # Create connection with proper header passing
            self.websocket = await websockets.connect(
                ws_url, additional_headers=headers
            )

            # Update session state
            self.is_connected = True

            # Start a background task to receive audio chunks
            asyncio.create_task(self._receive_audio_chunks())

            logger.info(f"Started Deepgram TTS session for call {self.call_sid}")
            return True

        except Exception as e:
            logger.error(
                f"Error starting Deepgram TTS session for call {self.call_sid}: {e}",
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
                        # Binary audio data
                        if message and self.audio_callback:
                            await self.audio_callback(
                                message,
                                False,  # Not final
                                {"call_sid": self.call_sid},
                            )
                    else:
                        # JSON control message
                        try:
                            data = json.loads(message)

                            # Handle different message types
                            if data.get("type") == "Flushed":
                                logger.debug(
                                    f"Received Flushed confirmation for call {self.call_sid}"
                                )
                            elif data.get("type") == "Cleared":
                                logger.debug(
                                    f"Received Cleared confirmation for call {self.call_sid}"
                                )
                            elif data.get("type") == "Error":
                                logger.error(
                                    f"Deepgram error for call {self.call_sid}: {data}"
                                )
                            else:
                                logger.debug(f"Received control message: {data}")

                        except json.JSONDecodeError:
                            logger.warning(
                                f"Received non-JSON text message from Deepgram: {message[:100]}"
                            )

                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)

        except websockets.exceptions.ConnectionClosed:
            logger.warning(
                f"Deepgram WebSocket connection closed for call {self.call_sid}"
            )
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
                
                # Process each part except the last one immediately
                for i, part in enumerate(parts[:-1]):
                    if part.strip():  # Only process non-empty parts
                        self.buffer += part
                        await self._convert_to_speech()
                
                # Add the last part to buffer
                last_part = parts[-1]
                if last_part.strip():  # Only add if not empty
                    self.buffer += last_part
                
                # Always convert the last part if we have anything in buffer
                # This ensures text without punctuation gets processed
                if self.buffer.strip() and (force_flush or self._should_convert(self.buffer) or len(parts) > 1):
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

    def _validate_voice_id(self, voice_id: str) -> bool:
        """
        Validate if a voice ID is potentially valid for Deepgram.

        Args:
            voice_id: The voice ID to validate

        Returns:
            bool: Whether the voice ID appears to be valid
        """
        if not voice_id:
            return False

        # Check if it's a known voice name
        if voice_id.lower() in self._available_voices:
            return True

        # Check if it looks like a valid Deepgram Aura voice ID
        if voice_id.startswith("aura-") and voice_id.endswith("-en"):
            return True

        # Check if it's already a mapped voice ID
        for voice_info in self._available_voices.values():
            if voice_info.id == voice_id:
                return True

        return False

    async def _convert_to_speech(self) -> None:
        """
        Convert buffered text to speech and send to Deepgram.
        """
        if not self.buffer.strip():
            return

        try:
            # Prepare the Speak message for Deepgram
            speak_message = {"type": "Speak", "text": self.buffer.strip()}

            # Send the text to Deepgram
            await self.websocket.send(json.dumps(speak_message))

            # Send flush message to get the audio
            flush_message = {"type": "Flush"}
            await self.websocket.send(json.dumps(flush_message))

            # Clear the buffer
            self.buffer = ""

            logger.debug(f"Sent text to Deepgram for call {self.call_sid}")

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
                try:
                    await self.websocket.close()
                except Exception:
                    pass

            # Get stored session settings
            metadata = self.metadata

            # Build query parameters
            query_params = [
                f"model={metadata.get('model_id', 'aura-asteria-en')}",
                f"encoding={metadata.get('encoding', 'mulaw')}",
                f"sample_rate={metadata.get('sample_rate', 8000)}",
            ]

            # Construct the final WebSocket URL
            ws_url = f"{self.ws_base_url}?{'&'.join(query_params)}"

            # Reconnect with same settings
            headers = {"Authorization": f"Token {self.api_key}"}

            websocket = await websockets.connect(ws_url, additional_headers=headers)

            # Update session state
            self.websocket = websocket
            self.is_connected = True

            # Start a new background task to receive audio chunks
            asyncio.create_task(self._receive_audio_chunks())

            logger.info(f"Reconnected Deepgram TTS session for call {self.call_sid}")
            return True

        except Exception as e:
            logger.error(
                f"Error reconnecting Deepgram TTS session for call {self.call_sid}: {e}",
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

            # Send clear message to stop current synthesis
            if self.websocket and self.is_connected:
                try:
                    clear_message = {"type": "Clear"}
                    await self.websocket.send(json.dumps(clear_message))
                    logger.debug(f"Sent Clear message for call {self.call_sid}")
                except Exception as e:
                    logger.warning(f"Error sending clear message: {e}")

            logger.info(f"Stopped Deepgram TTS synthesis for call {self.call_sid}")

        except Exception as e:
            logger.error(
                f"Error stopping Deepgram TTS synthesis for call {self.call_sid}: {e}",
                exc_info=True,
            )

    async def end_session(self) -> None:
        """
        End a TTS session and clean up resources.
        """
        try:
            # Convert any remaining buffered text
            if self.buffer.strip():
                await self._convert_to_speech()

            # Send close message to properly close the connection
            if self.websocket and self.is_connected:
                try:
                    close_message = {"type": "Close"}
                    await self.websocket.send(json.dumps(close_message))

                    # Wait a bit for the close message to be processed
                    await asyncio.sleep(0.1)

                    # Signal final audio chunk
                    if self.audio_callback:
                        await self.audio_callback(
                            b"",  # Empty data for final signal
                            True,  # Final
                            {"call_sid": self.call_sid},
                        )

                except Exception as e:
                    logger.warning(f"Error sending close message: {e}")

            # Close WebSocket connection
            if self.websocket:
                try:
                    await self.websocket.close()
                except Exception:
                    pass

            # Reset state
            self.is_connected = False
            self.websocket = None
            self.buffer = ""

            logger.info(f"Ended Deepgram TTS session for call {self.call_sid}")

        except Exception as e:
            logger.error(
                f"Error ending Deepgram TTS session for call {self.call_sid}: {e}",
                exc_info=True,
            )

    @classmethod
    def get_available_voices(cls) -> Dict[str, VoiceInfo]:
        """
        Get available voices for Deepgram TTS.

        Returns:
            Dict[str, VoiceInfo]: Mapping of voice names to voice information
        """
        return cls._available_voices.copy()

    @classmethod
    def get_available_models(cls) -> Dict[str, ModelInfo]:
        """
        Get available models for Deepgram TTS.

        Returns:
            Dict[str, ModelInfo]: Mapping of model names to model information
        """
        return cls._available_models.copy()

    @classmethod
    def get_voice_id(cls, voice_name: str) -> str:
        """
        Get a voice ID by name.

        Args:
            voice_name: The name of the voice to use

        Returns:
            str: The voice ID
        """
        voice_name = voice_name.lower()
        if voice_name in cls._available_voices:
            return cls._available_voices[voice_name].id
        else:
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

        # For Deepgram, if they specify a model family, return a default voice from that family
        if model_name == "aura":
            return "aura-asteria-en"  # Default Aura voice
        elif model_name == "aura-2":
            return "aura-2-thalia-en"  # Default Aura-2 voice
        elif model_name in cls._available_models:
            return cls._available_models[model_name].id
        else:
            # If they provide a full voice ID like "aura-asteria-en", return it as-is
            if model_name.startswith("aura-") and model_name.endswith("-en"):
                return model_name
            # Default to asteria if unknown model
            return "aura-asteria-en"
