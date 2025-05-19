"""
This file contains the DeepgramService for real-time speech-to-text transcription.
"""

# pylint: disable=logging-format-interpolation,logging-fstring-interpolation,broad-exception-caught
import os
import logging
import asyncio
import traceback
from typing import Callable, Dict, Any

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DeepgramService:
    """
    Service for handling real-time speech-to-text transcription using Deepgram.
    """

    # Constants
    MAX_RECONNECTS = 3

    def __init__(self):
        """Initialize the Deepgram client."""
        # Get API key from environment variable
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        # Allow override with auth token if specified
        self.auth_token = os.getenv("DEEPGRAM_AUTH_TOKEN")

        # Create client options with keepalive for better connection stability
        client_options = DeepgramClientOptions(
            options={
                "keepalive": "true",
            }
        )

        self.deepgram = None
        self.active_connections = {}  # Keyed by call_sid
        self.is_connected = {}  # Connection status keyed by call_sid
        self.reconnect_attempts = {}  # Reconnection attempts keyed by call_sid

        try:
            # Initialize with either API key or auth token
            if self.auth_token:
                # Use auth token if available
                self.deepgram = DeepgramClient(self.auth_token, client_options)
                logger.info("Deepgram client initialized with auth token")
            elif self.api_key:
                # Fall back to API key
                self.deepgram = DeepgramClient(self.api_key, client_options)
                logger.info("Deepgram client initialized with API key")
            else:
                logger.warning(
                    "Neither Deepgram API key nor auth token found. STT features will be limited."
                )
        except Exception as e:
            logger.error(f"Error initializing Deepgram client: {e}", exc_info=True)

    async def start_transcription(
        self,
        call_sid: str,
        transcript_callback: Callable[[str, bool, Dict], None],
        sample_rate: int = 8000,
        channels: int = 1,
    ):
        """
        Start a live transcription session for a call.

        Args:
            call_sid: Twilio Call SID
            transcript_callback: Callback function to handle transcripts
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
        """
        if not self.deepgram:
            logger.error("Deepgram client not initialized")
            return False

        # Initialize connection tracking
        self.is_connected[call_sid] = False
        self.reconnect_attempts[call_sid] = 0

        # Store callback
        self.active_connections[call_sid] = {
            "connection": None,
            "callback": transcript_callback,
        }

        # Establish connection
        return await self._ensure_connection(call_sid, sample_rate, channels)

    async def _ensure_connection(
        self, call_sid: str, sample_rate: int = 8000, channels: int = 1
    ) -> bool:
        """
        Ensure there is an active connection to Deepgram for the given call_sid.

        Args:
            call_sid: Twilio Call SID
            sample_rate: Audio sample rate
            channels: Number of audio channels

        Returns:
            bool: Whether a connection was successfully established
        """
        if (
            call_sid in self.is_connected
            and self.is_connected[call_sid]
            and call_sid in self.active_connections
            and self.active_connections[call_sid]["connection"]
        ):
            return True

        if (
            call_sid in self.reconnect_attempts
            and self.reconnect_attempts[call_sid] >= self.MAX_RECONNECTS
        ):
            logger.error(f"Max reconnection attempts reached for call {call_sid}")
            return False

        try:
            # Create a connection to Deepgram using listen API v1
            connection = self.deepgram.listen.asyncwebsocket.v("1")

            # Set up event handlers with proper coroutines
            connection.on(
                LiveTranscriptionEvents.Transcript,
                lambda ws, result: self._on_message_async(ws, result, call_sid),
            )
            connection.on(
                LiveTranscriptionEvents.Error,
                lambda ws, error: self._on_error_async(ws, error, call_sid),
            )
            connection.on(
                LiveTranscriptionEvents.Open,
                lambda ws, open: self._on_open_async(ws, open, call_sid),
            )
            connection.on(
                LiveTranscriptionEvents.Close,
                lambda ws, close: self._on_close_async(ws, close, call_sid),
            )

            # Store connection
            if call_sid in self.active_connections:
                self.active_connections[call_sid]["connection"] = connection
            else:
                self.active_connections[call_sid] = {
                    "connection": connection,
                    "callback": None,
                }

            # Configure options for streaming audio
            options = LiveOptions(
                model="nova-3",  # Use Nova-3 as requested
                punctuate=True,
                language="en-US",
                encoding="mulaw",  # Twilio uses mulaw encoding
                channels=channels,
                sample_rate=sample_rate,
                endpointing=100,  # More responsive stopping of conversation
                utterance_end_ms=1000,  # Word-based silence detector (1 second)
                smart_format=True,
                interim_results=True,  # Enable for real-time responses
            )

            # Start the connection
            try:
                await connection.start(options)
                logger.info(
                    f"Started Deepgram Nova-3 transcription for call {call_sid}"
                )

                # Connection start was successful if we get here
                self.is_connected[call_sid] = True
                self.reconnect_attempts[call_sid] = 0
                return True
            except Exception as e:
                logger.error(f"Failed to start with Nova-3 model, trying fallback: {e}")
                try:
                    # Try with nova-2 model as fallback
                    options = LiveOptions(
                        model="nova-2",  # Fallback to Nova-2
                        punctuate=True,
                        language="en-US",
                        encoding="mulaw",  # Twilio uses mulaw encoding
                        channels=channels,
                        sample_rate=sample_rate,
                        endpointing=100,
                        utterance_end_ms=1000,
                        smart_format=True,
                        interim_results=True,
                    )
                    await connection.start(options)
                    logger.info(
                        f"Started Deepgram Nova-2 transcription (fallback) for call {call_sid}"
                    )

                    # Connection start was successful if we get here
                    self.is_connected[call_sid] = True
                    self.reconnect_attempts[call_sid] = 0
                    return True
                except Exception as e2:
                    logger.error(f"Failed with Nova-2 model: {e2}")
                    if call_sid in self.reconnect_attempts:
                        self.reconnect_attempts[call_sid] += 1
                    return False

        except Exception as e:
            logger.error(f"Error establishing Deepgram connection: {e}", exc_info=True)
            if call_sid in self.reconnect_attempts:
                self.reconnect_attempts[call_sid] += 1
            return False

    async def _on_open_async(self, ws, open_event, call_sid: str):
        """Handle WebSocket open event."""
        logger.info(f"Deepgram connection opened for call {call_sid}")
        self.is_connected[call_sid] = True

    async def _on_message_async(self, ws, result, call_sid: str):
        """
        Handle transcript messages from Deepgram.

        Args:
            ws: WebSocket connection
            result: Transcript result
            call_sid: Twilio Call SID
        """
        try:
            if call_sid not in self.active_connections:
                logger.warning(f"Received transcript for unknown call {call_sid}")
                return

            callback = self.active_connections[call_sid].get("callback")

            # Extract the transcript from the result
            try:
                channel = result.channel
                alternatives = channel.alternatives

                if not alternatives:
                    return

                sentence = alternatives[0].transcript
                # Skip empty transcripts
                if not sentence or len(sentence.strip()) == 0:
                    return

                # Get transcript metadata
                is_final = result.is_final
                speech_final = result.speech_final
                # Log the transcript
                if speech_final:
                    logger.info(f"Speech Final - Transcript for {call_sid}: {sentence}")
                elif is_final:
                    logger.info(f"Final Result - Transcript for {call_sid}: {sentence}")
                else:
                    logger.info(
                        f"Interim Result - Transcript for {call_sid}: {sentence}"
                    )

                # Pass to callback if available
                if callback:
                    try:
                        # Create a task for the callback to avoid blocking the event handler
                        asyncio.create_task(
                            callback(
                                sentence,
                                is_final,
                                {"is_final": is_final, "speech_final": speech_final},
                            )
                        )
                    except Exception as cb_error:
                        logger.error(f"Error calling transcript callback: {cb_error}")

            except AttributeError as ae:
                logger.error(f"Error accessing result attributes: {ae}")
                logger.error(f"Result structure: {dir(result)}")
                return

        except Exception as e:
            logger.error(f"Error in transcript handler: {e}")
            logger.error(traceback.format_exc())

    async def _on_error_async(self, ws, error, call_sid: str):
        """Handle WebSocket error event."""
        logger.error(f"Deepgram error for call {call_sid}: {error}")

        # Mark connection as closed
        self.is_connected[call_sid] = False

        # Try to reconnect
        if (
            call_sid in self.reconnect_attempts
            and self.reconnect_attempts[call_sid] < self.MAX_RECONNECTS
        ):
            logger.info(
                f"Attempting to reconnect for call {call_sid}, attempt {self.reconnect_attempts[call_sid] + 1}/{self.MAX_RECONNECTS}"
            )
            # Get the sample rate and channels from the current connection if available
            connection_info = self.active_connections.get(call_sid, {})
            sample_rate = connection_info.get("sample_rate", 8000)
            channels = connection_info.get("channels", 1)

            # Try to reconnect
            await self._ensure_connection(call_sid, sample_rate, channels)

    async def _on_close_async(self, ws, close_event, call_sid: str):
        """Handle WebSocket close event."""
        logger.info(f"Deepgram connection closed for call {call_sid}")

        # Mark connection as closed
        self.is_connected[call_sid] = False

        # Clean up resources
        if call_sid in self.active_connections:
            self.active_connections[call_sid]["connection"] = None

        # Remove the call from our tracking
        if call_sid in self.reconnect_attempts:
            del self.reconnect_attempts[call_sid]

    async def send_audio(self, call_sid: str, audio_data: bytes) -> bool:
        """
        Send audio data to an active Deepgram transcription session.

        Args:
            call_sid: The Twilio call SID identifying the transcription session
            audio_data: Raw audio bytes to send to Deepgram

        Returns:
            bool: True if audio was sent successfully, False otherwise
        """
        # First ensure there's a connection
        if call_sid not in self.is_connected or not self.is_connected[call_sid]:
            logger.warning(
                f"No active connection for call: {call_sid}, attempting to reconnect"
            )
            if not await self._ensure_connection(call_sid):
                return False

        # Get the connection
        connection_info = self.active_connections.get(call_sid, {})
        connection = connection_info.get("connection")

        if not connection:
            logger.warning(f"No connection object for call: {call_sid}")
            return False

        try:
            # Log audio data details for debugging
            logger.debug(f"Sending audio chunk of size: {len(audio_data)} bytes")

            # Send the audio data
            await connection.send(audio_data)
            return True
        except Exception as e:
            logger.error(f"Error sending audio to Deepgram: {e}", exc_info=True)
            # Mark connection as failed
            self.is_connected[call_sid] = False

            # Try to reconnect on the next audio packet
            return False

    async def stop_transcription(self, call_sid: str) -> None:
        """
        Stop an active transcription session.

        Args:
            call_sid: The Twilio call SID identifying the transcription session
        """
        connection_info = self.active_connections.get(call_sid, {})
        connection = connection_info.get("connection")

        if not connection:
            logger.warning(f"No active transcription for call: {call_sid}")
            return

        try:
            # Finish the connection
            await connection.finish()
            logger.info(f"Stopped Deepgram transcription for call: {call_sid}")
        except Exception as e:
            logger.error(f"Error stopping Deepgram transcription: {e}", exc_info=True)
        finally:
            # Clean up resources
            if call_sid in self.active_connections:
                del self.active_connections[call_sid]
            if call_sid in self.is_connected:
                del self.is_connected[call_sid]
            if call_sid in self.reconnect_attempts:
                del self.reconnect_attempts[call_sid]
