"""
This file contains the DeepgramService for real-time speech-to-text transcription.
"""

# pylint: disable=logging-format-interpolation,logging-fstring-interpolation,broad-exception-caught
import os
import logging
import asyncio
import traceback
from typing import Callable, Dict, Optional

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

    # Class-level client (shared across instances)
    _client = None

    @classmethod
    def get_client(cls, api_key: Optional[str] = None):
        """
        Get or create the Deepgram client.

        Args:
            api_key: Optional API key to use instead of environment variable

        Returns:
            DeepgramClient: The Deepgram client
        """
        # If we're using a custom API key for this instance, create a new client
        if api_key:
            # Create client options with keepalive for better connection stability
            client_options = DeepgramClientOptions(
                options={
                    "keepalive": "true",
                }
            )
            try:
                client = DeepgramClient(api_key, client_options)
                logger.info("Deepgram client initialized with custom API key")
                return client
            except Exception as e:
                logger.error(
                    f"Error initializing Deepgram client with custom API key: {e}",
                    exc_info=True,
                )
                # Fall back to shared client

        # Otherwise use the shared client
        if cls._client is None:
            # Get API key from environment variable
            api_key = os.getenv("DEEPGRAM_API_KEY")
            # Allow override with auth token if specified
            auth_token = os.getenv("DEEPGRAM_AUTH_TOKEN")

            # Create client options with keepalive for better connection stability
            client_options = DeepgramClientOptions(
                options={
                    "keepalive": "true",
                }
            )

            try:
                # Initialize with either API key or auth token
                if auth_token:
                    # Use auth token if available
                    cls._client = DeepgramClient(auth_token, client_options)
                    logger.info("Deepgram client initialized with auth token")
                elif api_key:
                    # Fall back to API key
                    cls._client = DeepgramClient(api_key, client_options)
                    logger.info("Deepgram client initialized with API key")
                else:
                    logger.warning(
                        "Neither Deepgram API key nor auth token found. STT features will be limited."
                    )
            except Exception as e:
                logger.error(f"Error initializing Deepgram client: {e}", exc_info=True)

        return cls._client

    def __init__(
        self,
        call_sid: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "nova-3",
        language: str = "en-US",
        punctuate: bool = True,
        interim_results: bool = True,
        endpointing: int = 100,
        utterance_end_ms: int = 1000,
        smart_format: bool = True,
        **options,
    ):
        """
        Initialize a Deepgram service instance for a specific call.

        Args:
            call_sid: The unique identifier for this call
            api_key: Optional custom API key for this call
            model: Deepgram model to use (default: "nova-3")
            language: Language code (default: "en-US")
            punctuate: Whether to add punctuation (default: True)
            interim_results: Whether to provide interim results (default: True)
            endpointing: Time in ms for silence detection (default: 100)
            utterance_end_ms: Time in ms to wait before ending utterance (default: 1000)
            smart_format: Whether to apply smart formatting (default: True)
            **options: Additional configuration options
        """
        self.call_sid = call_sid
        self.api_key = api_key  # Store instance-specific API key
        self.deepgram = self.get_client(
            api_key
        )  # Use instance-specific API key if provided
        self.connection = None
        self.transcript_callback = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.sample_rate = 8000
        self.channels = 1

        # Store configuration options
        self.model = model
        self.language = language
        self.punctuate = punctuate
        self.interim_results = interim_results
        self.endpointing = endpointing
        self.utterance_end_ms = utterance_end_ms
        self.smart_format = smart_format
        self.options = options

        if call_sid:
            logger.info(f"DeepgramService initialized for call {call_sid}")

    async def start_transcription(
        self,
        transcript_callback: Callable[[str, bool, Dict], None],
        sample_rate: int = 8000,
        channels: int = 1,
    ) -> bool:
        """
        Start a live transcription session for this call.

        Args:
            transcript_callback: Callback function to handle transcripts
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels

        Returns:
            bool: Whether transcription was successfully started
        """
        if not self.deepgram:
            logger.error("Deepgram client not initialized")
            return False

        if not self.call_sid:
            logger.error("Cannot start transcription without call_sid")
            return False

        # Store callback and audio settings
        self.transcript_callback = transcript_callback
        self.sample_rate = sample_rate
        self.channels = channels

        # Establish connection
        return await self._ensure_connection()

    async def _ensure_connection(self) -> bool:
        """
        Ensure there is an active connection to Deepgram for this call.

        Returns:
            bool: Whether a connection was successfully established
        """
        if self.is_connected and self.connection:
            return True

        if self.reconnect_attempts >= self.MAX_RECONNECTS:
            logger.error(f"Max reconnection attempts reached for call {self.call_sid}")
            return False

        try:
            # Create a connection to Deepgram using listen API v1
            connection = self.deepgram.listen.asyncwebsocket.v("1")

            # Set up event handlers with proper coroutines
            connection.on(
                LiveTranscriptionEvents.Transcript,
                lambda ws, result: self._on_message_async(ws, result),
            )
            connection.on(
                LiveTranscriptionEvents.Error,
                lambda ws, error: self._on_error_async(ws, error),
            )
            connection.on(
                LiveTranscriptionEvents.Open,
                lambda ws, open: self._on_open_async(ws, open),
            )
            connection.on(
                LiveTranscriptionEvents.Close,
                lambda ws, close: self._on_close_async(ws, close),
            )

            # Store connection
            self.connection = connection

            # Configure options for streaming audio using instance settings
            live_options = LiveOptions(
                model=self.model,
                punctuate=self.punctuate,
                language=self.language,
                encoding="mulaw",  # Twilio uses mulaw encoding
                channels=self.channels,
                sample_rate=self.sample_rate,
                endpointing=self.endpointing,
                utterance_end_ms=self.utterance_end_ms,
                smart_format=self.smart_format,
                interim_results=self.interim_results,
            )

            # Start the connection
            try:
                await connection.start(live_options)
                logger.info(
                    f"Started Deepgram transcription with model {self.model} for call {self.call_sid}"
                )

                # Connection start was successful if we get here
                self.is_connected = True
                self.reconnect_attempts = 0
                return True
            except Exception as e:
                logger.error(
                    f"Failed to start with model {self.model}, trying fallback: {e}"
                )
                try:
                    # Try with nova-2 model as fallback
                    fallback_options = LiveOptions(
                        model="nova-2",  # Fallback to Nova-2
                        punctuate=self.punctuate,
                        language=self.language,
                        encoding="mulaw",  # Twilio uses mulaw encoding
                        channels=self.channels,
                        sample_rate=self.sample_rate,
                        endpointing=self.endpointing,
                        utterance_end_ms=self.utterance_end_ms,
                        smart_format=self.smart_format,
                        interim_results=self.interim_results,
                    )
                    await connection.start(fallback_options)
                    logger.info(
                        f"Started Deepgram Nova-2 transcription (fallback) for call {self.call_sid}"
                    )

                    # Connection start was successful if we get here
                    self.is_connected = True
                    self.reconnect_attempts = 0
                    return True
                except Exception as e2:
                    logger.error(f"Failed with Nova-2 model: {e2}")
                    self.reconnect_attempts += 1
                    return False

        except Exception as e:
            logger.error(f"Error establishing Deepgram connection: {e}", exc_info=True)
            self.reconnect_attempts += 1
            return False

    async def _on_open_async(self, ws, open_event):
        """Handle WebSocket open event."""
        logger.info(f"Deepgram connection opened for call {self.call_sid}")
        self.is_connected = True

    async def _on_message_async(self, ws, result):
        """
        Handle transcript messages from Deepgram.

        Args:
            ws: WebSocket connection
            result: Transcript result
        """
        try:
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
                    logger.info(
                        f"Speech Final - Transcript for {self.call_sid}: {sentence}"
                    )
                elif is_final:
                    logger.info(
                        f"Final Result - Transcript for {self.call_sid}: {sentence}"
                    )
                else:
                    logger.info(
                        f"Interim Result - Transcript for {self.call_sid}: {sentence}"
                    )

                # Pass to callback if available
                if self.transcript_callback:
                    try:
                        # Create a task for the callback to avoid blocking the event handler
                        asyncio.create_task(
                            self.transcript_callback(
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

    async def _on_error_async(self, ws, error):
        """Handle WebSocket error event."""
        logger.error(f"Deepgram error for call {self.call_sid}: {error}")

        # Mark connection as closed
        self.is_connected = False

        # Try to reconnect
        if self.reconnect_attempts < self.MAX_RECONNECTS:
            logger.info(
                f"Attempting to reconnect for call {self.call_sid}, attempt {self.reconnect_attempts + 1}/{self.MAX_RECONNECTS}"
            )
            # Try to reconnect
            await self._ensure_connection()

    async def _on_close_async(self, ws, close_event):
        """Handle WebSocket close event."""
        logger.info(f"Deepgram connection closed for call {self.call_sid}")

        # Mark connection as closed
        self.is_connected = False
        self.connection = None

    async def send_audio(self, audio_data: bytes) -> bool:
        """
        Send audio data to the Deepgram transcription session.

        Args:
            audio_data: Raw audio bytes to send to Deepgram

        Returns:
            bool: True if audio was sent successfully, False otherwise
        """
        # First ensure there's a connection
        if not self.is_connected:
            logger.warning(
                f"No active connection for call: {self.call_sid}, attempting to reconnect"
            )
            if not await self._ensure_connection():
                return False

        if not self.connection:
            logger.warning(f"No connection object for call: {self.call_sid}")
            return False

        try:
            # Log audio data details for debugging
            logger.debug(f"Sending audio chunk of size: {len(audio_data)} bytes")

            # Send the audio data
            await self.connection.send(audio_data)
            return True
        except Exception as e:
            logger.error(f"Error sending audio to Deepgram: {e}", exc_info=True)
            # Mark connection as failed
            self.is_connected = False

            # Try to reconnect on the next audio packet
            return False

    async def stop_transcription(self) -> None:
        """
        Stop the active transcription session.
        """
        if not self.connection:
            logger.warning(f"No active transcription for call: {self.call_sid}")
            return

        try:
            # Finish the connection
            await self.connection.finish()
            logger.info(f"Stopped Deepgram transcription for call: {self.call_sid}")
        except asyncio.CancelledError:
            # This is expected during shutdown - suppress the error
            logger.info(
                f"Deepgram connection tasks cancelled during shutdown for call: {self.call_sid}"
            )
        except Exception as e:
            logger.error(f"Error stopping Deepgram transcription: {e}", exc_info=True)
        finally:
            # Clean up resources
            self.connection = None
            self.is_connected = False
            self.transcript_callback = None
