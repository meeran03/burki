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
from dataclasses import dataclass, field
import websockets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TTSState:
    """Represents the state of a TTS session."""

    call_sid: str
    websocket: Any = None
    buffer: str = ""
    is_connected: bool = False
    audio_callback: Optional[Callable[[bytes, bool, Dict[str, Any]], None]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    Service for handling ElevenLabs text-to-speech streaming.
    Manages WebSocket connections and text buffering for natural speech synthesis.
    """

    def __init__(self):
        """Initialize the TTS service."""
        # Get API key from environment variable
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable is not set")

        # ElevenLabs WebSocket base URL
        self.ws_base_url = "wss://api.elevenlabs.io/v1/text-to-speech"

        # Track active TTS sessions
        self.active_sessions: Dict[str, TTSState] = {}

        # Sentence-ending punctuation
        self.sentence_endings = {".", "!", "?", ":", ";"}

        # Available voices mapping (name -> id)
        self.available_voices = {
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
        self.available_models = {
            "turbo": "eleven_turbo_v2",
            "enhanced": "eleven_enhanced_v2",
            "multilingual": "eleven_multilingual_v2",
        }

        logger.info("TTSService initialized")

    async def start_session(
        self,
        call_sid: str,
        options: Optional[TTSOptions] = None,
        audio_callback: Optional[Callable[[bytes, bool, Dict[str, Any]], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Start a new TTS session for a call.

        Args:
            call_sid: The Twilio call SID
            options: Configuration options for TTS
            audio_callback: Callback function for handling audio data
            metadata: Additional metadata for the session

        Returns:
            bool: Whether the session was started successfully
        """
        try:
            # Use default options if none provided
            if options is None:
                options = TTSOptions()

            # Store options in metadata for reconnection
            session_metadata = metadata or {}
            session_metadata.update(
                {
                    "voice_id": options.voice_id,
                    "model_id": options.model_id,
                    "stability": options.stability,
                    "similarity_boost": options.similarity_boost,
                    "style": options.style,
                    "use_speaker_boost": options.use_speaker_boost,
                    "latency": options.latency,
                    "conditioning_phrase": options.conditioning_phrase,
                    "language": options.language,
                    "output_format": options.output_format,
                }
            )

            # Initialize session state
            self.active_sessions[call_sid] = TTSState(
                call_sid=call_sid,
                audio_callback=audio_callback,
                metadata=session_metadata,
            )

            # Connect to ElevenLabs WebSocket
            headers = {"xi-api-key": self.api_key}

            # Build the WebSocket URL with the voice_id in the path
            voice_id = options.voice_id

            # Build query parameters
            query_params = [
                f"model_id={options.model_id}",
                f"optimize_streaming_latency={options.latency}",
                f"output_format={options.output_format}",
                "sample_rate=8000",  # Explicitly set sample rate to 8000Hz for Twilio
            ]

            # Add optional parameters
            if options.language:
                query_params.append(f"language_code={options.language}")

            # Construct the final WebSocket URL with the stream-input endpoint
            ws_url = f"{self.ws_base_url}/{voice_id}/stream-input?{'&'.join(query_params)}"

            logger.info(f"Connecting to ElevenLabs with URL: {ws_url}")

            # Create connection with proper header passing
            websocket = await websockets.connect(
                ws_url,
                additional_headers=headers
            )

            # Update session state
            self.active_sessions[call_sid].websocket = websocket
            self.active_sessions[call_sid].is_connected = True

            # Start a background task to receive audio chunks
            asyncio.create_task(self._receive_audio_chunks(call_sid))

            logger.info(f"Started TTS session for call {call_sid}")
            return True

        except Exception as e:
            logger.error(
                f"Error starting TTS session for call {call_sid}: {e}", exc_info=True
            )
            if call_sid in self.active_sessions:
                del self.active_sessions[call_sid]
            return False

    async def _receive_audio_chunks(self, call_sid: str) -> None:
        """
        Continuously receive audio chunks from the WebSocket.

        Args:
            call_sid: The Twilio call SID
        """
        if call_sid not in self.active_sessions:
            return

        session = self.active_sessions[call_sid]
        if not session.websocket or not session.is_connected:
            return

        try:
            async for message in session.websocket:
                try:
                    # Parse the JSON message
                    data = json.loads(message)

                    # Extract audio data
                    if "audio" in data:
                        # Decode the base64-encoded audio data
                        audio_data = base64.b64decode(data["audio"])

                        # Call the audio callback if available
                        if session.audio_callback:
                            await session.audio_callback(
                                audio_data, False, {"call_sid": call_sid}  # Not final
                            )

                    # Handle the end of the stream
                    if data.get("isFinal", False) and session.audio_callback:
                        await session.audio_callback(
                            b"",  # Empty data for final signal
                            True,  # Final
                            {"call_sid": call_sid},
                        )

                except json.JSONDecodeError:
                    logger.warning(
                        f"Received non-JSON message from ElevenLabs: {message[:100]}"
                    )
                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}", exc_info=True)

        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"WebSocket connection closed for call {call_sid}")
        except Exception as e:
            logger.error(f"Error receiving audio chunks: {e}", exc_info=True)

    async def process_text(
        self, call_sid: str, text: str, force_flush: bool = False
    ) -> bool:
        """
        Process text and convert to speech when appropriate.

        Args:
            call_sid: The Twilio call SID
            text: The text to process
            force_flush: Whether to force immediate speech conversion

        Returns:
            bool: Whether the text was processed successfully
        """
        if call_sid not in self.active_sessions:
            logger.warning(f"No active TTS session for call {call_sid}")
            return False

        session = self.active_sessions[call_sid]
        if not session.is_connected or not session.websocket:
            logger.warning(f"TTS session not connected for call {call_sid}")
            return False

        try:
            # Check for flush tag
            if "<flush/>" in text:
                # Split text at flush tag
                parts = text.split("<flush/>")
                # Add all parts except the last one to buffer
                session.buffer += "".join(parts[:-1])
                # Convert buffered text to speech
                await self._convert_to_speech(call_sid)
                # Add the last part to buffer
                session.buffer += parts[-1]
                # Check if we need to convert the last part too
                if force_flush or self._should_convert(session.buffer):
                    await self._convert_to_speech(call_sid)
            else:
                # Add text to buffer
                session.buffer += text

                # Check if we should convert to speech
                if force_flush or self._should_convert(session.buffer):
                    await self._convert_to_speech(call_sid)

            return True

        except Exception as e:
            logger.error(
                f"Error processing text for call {call_sid}: {e}", exc_info=True
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

    async def _convert_to_speech(self, call_sid: str) -> None:
        """
        Convert buffered text to speech and send to ElevenLabs.

        Args:
            call_sid: The Twilio call SID
        """
        session = self.active_sessions[call_sid]
        if not session.buffer.strip():
            return

        try:
            # Get TTS settings from metadata
            metadata = session.metadata

            # Prepare the message for ElevenLabs
            message = {
                "text": session.buffer,
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
            await session.websocket.send(json.dumps(message))

            # Clear the buffer
            session.buffer = ""

        except Exception as e:
            logger.error(
                f"Error converting text to speech for call {call_sid}: {e}",
                exc_info=True,
            )
            # Attempt to reconnect
            await self._reconnect_session(call_sid)

    async def _reconnect_session(self, call_sid: str) -> bool:
        """
        Attempt to reconnect a TTS session.

        Args:
            call_sid: The Twilio call SID

        Returns:
            bool: Whether the reconnection was successful
        """
        if call_sid not in self.active_sessions:
            return False

        session = self.active_sessions[call_sid]
        try:
            # Close existing connection if any
            if session.websocket:
                await session.websocket.close()

            # Get stored session settings
            metadata = session.metadata
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
            ws_url = f"{self.ws_base_url}/{voice_id}/stream-input?{'&'.join(query_params)}"

            # Reconnect with same settings
            headers = {"xi-api-key": self.api_key}

            # Use proper header passing
            websocket = await websockets.connect(
                ws_url,
                additional_headers=headers
            )

            # Update session state
            session.websocket = websocket
            session.is_connected = True

            # Start a new background task to receive audio chunks
            asyncio.create_task(self._receive_audio_chunks(call_sid))

            logger.info(f"Reconnected TTS session for call {call_sid}")
            return True

        except Exception as e:
            logger.error(
                f"Error reconnecting TTS session for call {call_sid}: {e}",
                exc_info=True,
            )
            session.is_connected = False
            return False

    async def end_session(self, call_sid: str) -> None:
        """
        End a TTS session and clean up resources.

        Args:
            call_sid: The Twilio call SID
        """
        if call_sid not in self.active_sessions:
            return

        session = self.active_sessions[call_sid]
        try:
            # Convert any remaining buffered text
            if session.buffer.strip():
                await self._convert_to_speech(call_sid)

            # Send an empty string to close the connection properly
            if session.websocket and session.is_connected:
                try:
                    await session.websocket.send(json.dumps({"text": ""}))
                except Exception:
                    pass  # Ignore errors when trying to send close message

            # Close WebSocket connection
            if session.websocket:
                await session.websocket.close()

            # Clean up session
            del self.active_sessions[call_sid]

            logger.info(f"Ended TTS session for call {call_sid}")

        except Exception as e:
            logger.error(
                f"Error ending TTS session for call {call_sid}: {e}", exc_info=True
            )

    def get_voice_id(self, voice_name: str) -> str:
        """
        Get a voice ID by name.

        Args:
            voice_name: The name of the voice to use

        Returns:
            str: The voice ID
        """
        voice_name = voice_name.lower()
        return self.available_voices.get(voice_name, self.available_voices["rachel"])

    def get_model_id(self, model_name: str) -> str:
        """
        Get a model ID by name.

        Args:
            model_name: The name of the model to use

        Returns:
            str: The model ID
        """
        model_name = model_name.lower()
        return self.available_models.get(model_name, self.available_models["turbo"])

    def get_session_state(self, call_sid: str) -> Optional[TTSState]:
        """
        Get the current state of a TTS session.

        Args:
            call_sid: The Twilio call SID

        Returns:
            Optional[TTSState]: The session state if it exists
        """
        return self.active_sessions.get(call_sid)

    async def stop_synthesis(self, call_sid: str) -> None:
        """
        Stop ongoing TTS synthesis for a call.
        This is used when an interruption is detected to immediately stop the AI from speaking.

        Args:
            call_sid: The Twilio call SID
        """
        if call_sid not in self.active_sessions:
            return

        session = self.active_sessions[call_sid]
        try:
            # Clear any buffered text
            session.buffer = ""

            # Send an empty string to stop current synthesis
            if session.websocket and session.is_connected:
                try:
                    # Send a message to stop current synthesis
                    stop_message = {
                        "text": "",
                        "flush": True,  # Force flush any buffered audio
                        "stop": True,   # Signal to stop synthesis
                    }
                    await session.websocket.send(json.dumps(stop_message))
                except Exception as e:
                    logger.warning(f"Error sending stop message: {e}")

            # Close and reconnect the WebSocket to ensure clean state
            if session.websocket:
                try:
                    await session.websocket.close()
                except Exception:
                    pass  # Ignore errors when closing

            # Reconnect to ensure clean state for future synthesis
            await self._reconnect_session(call_sid)

            logger.info(f"Stopped TTS synthesis for call {call_sid}")

        except Exception as e:
            logger.error(f"Error stopping TTS synthesis for call {call_sid}: {e}", exc_info=True)
            # Attempt to reconnect in case of errors
            await self._reconnect_session(call_sid)
