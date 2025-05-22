"""
This file contains the TTS service for handling ElevenLabs text-to-speech streaming.
"""

# pylint: disable=too-many-ancestors,logging-fstring-interpolation,broad-exception-caught
import os
import json
import logging
import asyncio
import base64
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TTSOptions:
    """Configuration options for TTS."""

    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Default voice ID (Rachel)
    model_id: str = "eleven_turbo_v2"  # Default model (Turbo)
    stability: float = 0.5  # Voice stability (0.0-1.0)
    similarity_boost: float = 0.75  # Voice similarity boost (0.0-1.0)
    style: float = 0.0  # Speaking style (0.0-1.0)
    use_speaker_boost: bool = True  # Enhance speech clarity and fidelity
    latency: int = 1  # 1-4, where 1 is lowest latency
    conditioning_phrase: Optional[str] = None  # Reference phrase for voice matching
    language: Optional[str] = None  # Optional language specification
    output_format: str = "ulaw_8000"  # Format compatible with Twilio's 8kHz mulaw


class TTSService:
    """
    Service for handling ElevenLabs text-to-speech streaming for a specific call.
    Manages WebSocket connection and text buffering for natural speech synthesis.
    """

    # Available voices mapping (name -> id)
    available_voices = {
        "rachel": "21m00Tcm4TlvDq8ikWAM",
        "domi": "AZnzlk1XvdvUeBnXmlld",
        "bella": "EXAVITQu4vr4xnSDxMaL",
        "antoni": "ErXwobaYiN019PkySvjV",
        "elli": "MF3mGyEYCl7XYWbV9V6O",
        "josh": "TxGEqnHWrfWFTfGW9XjX",
        "arnold": "VR6AewLTigWG4xSOukaG",
        "adam": "pNInz6obpgDQGcFmaJgB",
        "sam": "yoZ06aMxZJJ28mfd3POQ",
    }

    # Available models
    available_models = {
        "turbo": "eleven_turbo_v2",
        "enhanced": "eleven_enhanced_v2",
        "multilingual": "eleven_multilingual_v2",
    }

    def __init__(
        self,
        call_sid: Optional[str] = None,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model_id: Optional[str] = None,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True,
        latency: int = 1,
    ):
        """
        Initialize the TTS service for a specific call.

        Args:
            call_sid: The unique identifier for this call
            api_key: Optional custom API key for this call
            voice_id: The voice ID to use (default is Rachel)
            model_id: The model ID to use (default is Turbo)
            stability: Voice stability (0.0-1.0)
            similarity_boost: Voice similarity boost (0.0-1.0)
            style: Speaking style (0.0-1.0)
            use_speaker_boost: Enhance speech clarity and fidelity
            latency: 1-4, where 1 is lowest latency
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ELEVENLABS_API_KEY environment variable is not set and no API key provided"
            )

        # ElevenLabs WebSocket base URL
        self.ws_base_url = "wss://api.elevenlabs.io/v1/text-to-speech"

        # Sentence-ending punctuation
        self.sentence_endings = {".", "!", "?", ":", ";"}

        # Call-specific state
        self.call_sid = call_sid
        self.websocket = None
        self.buffer = ""
        self.is_connected = False
        self.audio_callback = None
        self.metadata = {}

        # Store TTS settings
        self.voice_id = voice_id or self.get_voice_id("rachel")
        self.model_id = model_id or self.get_model_id("turbo")
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.style = style
        self.use_speaker_boost = use_speaker_boost
        self.latency = latency

        logger.info(f"TTSService initialized for call {call_sid}")

    async def start_session(
        self,
        options: Optional[TTSOptions] = None,
        audio_callback: Optional[Callable[[bytes, bool, Dict[str, Any]], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Start a new TTS session for this call.

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
                stability=self.stability,
                similarity_boost=self.similarity_boost,
                style=self.style,
                use_speaker_boost=self.use_speaker_boost,
                latency=self.latency,
            )

            # Store options in metadata for reconnection
            session_metadata = metadata or {}
            session_metadata.update(
                {
                    "voice_id": session_options.voice_id,
                    "model_id": session_options.model_id,
                    "stability": session_options.stability,
                    "similarity_boost": session_options.similarity_boost,
                    "style": session_options.style,
                    "use_speaker_boost": session_options.use_speaker_boost,
                    "latency": session_options.latency,
                    "conditioning_phrase": session_options.conditioning_phrase,
                    "language": session_options.language,
                    "output_format": session_options.output_format,
                }
            )

            # Store callback and metadata
            self.audio_callback = audio_callback
            self.metadata = session_metadata

            # Connect to ElevenLabs WebSocket
            headers = {"xi-api-key": self.api_key}

            # Build the WebSocket URL with the voice_id in the path
            voice_id = session_options.voice_id

            # Build query parameters
            query_params = [
                f"model_id={session_options.model_id}",
                f"optimize_streaming_latency={session_options.latency}",
                f"output_format={session_options.output_format}",
                "sample_rate=8000",  # Explicitly set sample rate to 8000Hz for Twilio
            ]

            # Add optional parameters
            if session_options.language:
                query_params.append(f"language_code={session_options.language}")

            # Construct the final WebSocket URL with the stream-input endpoint
            ws_url = (
                f"{self.ws_base_url}/{voice_id}/stream-input?{'&'.join(query_params)}"
            )

            logger.info(f"Connecting to ElevenLabs with URL: {ws_url}")

            # Create connection with proper header passing
            self.websocket = await websockets.connect(
                ws_url, additional_headers=headers
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
                    # Parse the JSON message
                    data = json.loads(message)

                    # Extract audio data
                    if "audio" in data:
                        # Decode the base64-encoded audio data
                        audio_data = base64.b64decode(data["audio"])

                        # Call the audio callback if available
                        if self.audio_callback:
                            await self.audio_callback(
                                audio_data,
                                False,
                                {"call_sid": self.call_sid},  # Not final
                            )

                    # Handle the end of the stream
                    if data.get("isFinal", False) and self.audio_callback:
                        await self.audio_callback(
                            b"",  # Empty data for final signal
                            True,  # Final
                            {"call_sid": self.call_sid},
                        )

                except json.JSONDecodeError:
                    logger.warning(
                        f"Received non-JSON message from ElevenLabs: {message[:100]}"
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
        Convert buffered text to speech and send to ElevenLabs.
        """
        if not self.buffer.strip():
            return

        try:
            # Get TTS settings from metadata
            metadata = self.metadata

            # Prepare the message for ElevenLabs
            message = {
                "text": self.buffer,
                "voice_settings": {
                    "stability": metadata.get("stability", 0.5),
                    "similarity_boost": metadata.get("similarity_boost", 0.75),
                    "style": metadata.get("style", 0.0),
                    "use_speaker_boost": metadata.get("use_speaker_boost", True),
                },
                "try_trigger_generation": True,  # Force generation immediately
                "flush": True,  # Ensure the buffer is flushed immediately
            }

            # Add optional conditioning phrase if provided
            if metadata.get("conditioning_phrase"):
                message["voice_settings"]["conditioning_phrase"] = metadata[
                    "conditioning_phrase"
                ]

            # Send the text to ElevenLabs
            await self.websocket.send(json.dumps(message))

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

            # Get stored session settings
            metadata = self.metadata
            voice_id = metadata.get("voice_id", "21m00Tcm4TlvDq8ikWAM")

            # Build query parameters
            query_params = [
                f"model_id={metadata.get('model_id', 'eleven_turbo_v2')}",
                f"optimize_streaming_latency={metadata.get('latency', 1)}",
                f"output_format={metadata.get('output_format', 'ulaw_8000')}",
            ]

            # Add optional parameters
            if metadata.get("language"):
                query_params.append(f"language_code={metadata['language']}")

            # Construct the final WebSocket URL with the stream-input endpoint
            ws_url = (
                f"{self.ws_base_url}/{voice_id}/stream-input?{'&'.join(query_params)}"
            )

            # Reconnect with same settings
            headers = {"xi-api-key": self.api_key}

            # Use proper header passing
            websocket = await websockets.connect(ws_url, additional_headers=headers)

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

    async def end_session(self) -> None:
        """
        End a TTS session and clean up resources.
        """
        try:
            # Convert any remaining buffered text
            if self.buffer.strip():
                await self._convert_to_speech()

            # Send an empty string to close the connection properly
            if self.websocket and self.is_connected:
                try:
                    await self.websocket.send(json.dumps({"text": ""}))
                except Exception:
                    pass  # Ignore errors when trying to send close message

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
    def get_voice_id(cls, voice_name: str) -> str:
        """
        Get a voice ID by name.

        Args:
            voice_name: The name of the voice to use

        Returns:
            str: The voice ID
        """
        voice_name = voice_name.lower()
        return cls.available_voices.get(voice_name, cls.available_voices["rachel"])

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
        return cls.available_models.get(model_name, cls.available_models["turbo"])

    async def stop_synthesis(self) -> None:
        """
        Stop ongoing TTS synthesis for a call.
        This is used when an interruption is detected to immediately stop the AI from speaking.
        """
        try:
            # Clear any buffered text
            self.buffer = ""

            # Send an empty string to stop current synthesis
            if self.websocket and self.is_connected:
                try:
                    # Send a message to stop current synthesis
                    stop_message = {
                        "text": "",
                        "flush": True,  # Force flush any buffered audio
                        "stop": True,  # Signal to stop synthesis
                    }
                    await self.websocket.send(json.dumps(stop_message))
                except Exception as e:
                    logger.warning(f"Error sending stop message: {e}")

            # Close and reconnect the WebSocket to ensure clean state
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
