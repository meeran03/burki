"""
This file contains the CallHandler class for managing call state and conversation flow.
"""

# pylint: disable=too-many-ancestors,logging-fstring-interpolation,broad-exception-caught
import logging
import base64
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from fastapi import WebSocket

from app.services.deepgram_service import DeepgramService
from app.services.llm_service import LLMService
from app.services.tts_service import TTSService, TTSOptions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class CallState:
    """Represents the state of an ongoing call."""

    call_sid: str
    websocket: WebSocket
    start_time: datetime = field(default_factory=datetime.now)
    to_number: Optional[str] = None
    from_number: Optional[str] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_ai_speaking: bool = False  # Track when AI is speaking
    ai_speaking_start_time: Optional[datetime] = None  # Track when AI started speaking
    interruption_threshold: int = 3  # Number of words to trigger interruption
    min_speaking_time: float = (
        0.5  # Minimum time AI must be speaking before interruption is valid (in seconds)
    )
    last_interruption_time: Optional[datetime] = None  # Track last interruption
    interruption_cooldown: float = (
        2.0  # Cooldown period in seconds between interruptions
    )
    transcript_buffer: str = ""  # Buffer for accumulating transcripts during cooldown
    is_in_cooldown: bool = False  # Track if we're in cooldown period


class CallHandler:
    """
    Handles the lifecycle of a call, managing transcription, LLM processing,
    and conversation state.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        custom_llm_url: Optional[str] = None,
    ):
        """
        Initialize the call handler.

        Args:
            system_prompt: Optional custom system prompt for the LLM
            custom_llm_url: Optional URL for custom LLM endpoint
        """
        # Initialize services internally
        self.deepgram_service = DeepgramService()
        self.llm_service = LLMService(custom_llm_url=custom_llm_url)
        self.tts_service = TTSService()
        self.system_prompt = system_prompt

        # Track active calls
        self.active_calls: Dict[str, CallState] = {}

    async def start_call(
        self,
        call_sid: str,
        websocket: WebSocket,
        to_number: Optional[str] = None,
        from_number: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Start handling a new call.

        Args:
            call_sid: The Twilio call SID
            websocket: The WebSocket connection for this call
            to_number: The destination phone number
            from_number: The caller's phone number
            metadata: Additional call metadata
        """
        # Initialize call state
        self.active_calls[call_sid] = CallState(
            call_sid=call_sid,
            websocket=websocket,
            to_number=to_number,
            from_number=from_number,
            metadata=metadata or {},
        )

        # Start LLM conversation
        await self.llm_service.start_conversation(
            call_sid=call_sid, system_prompt=self.system_prompt
        )

        # Start TTS session
        tts_options = TTSOptions(
            voice_id=self.tts_service.get_voice_id("rachel"),  # Default voice
            model_id=self.tts_service.get_model_id(
                "turbo"
            ),  # Turbo model for low latency
            latency=1,  # Lowest latency
        )

        # Define audio callback to handle TTS audio
        async def audio_callback(
            audio_data: bytes, is_final: bool, audio_metadata: Dict[str, Any]
        ) -> None:
            await self._handle_tts_audio(call_sid, audio_data, is_final, audio_metadata)

        await self.tts_service.start_session(
            call_sid=call_sid,
            options=tts_options,
            audio_callback=audio_callback,
            metadata={"call_sid": call_sid},
        )

        logger.info(f"Started handling call {call_sid}")

    async def _handle_tts_audio(
        self, call_sid: str, audio_data: bytes, is_final: bool, metadata: Dict[str, Any]
    ) -> None:
        """
        Handle TTS audio data and send it to Twilio.

        Args:
            call_sid: The Twilio call SID
            audio_data: The audio data bytes
            is_final: Whether this is the final audio chunk
            metadata: Additional metadata about the audio
        """
        if call_sid not in self.active_calls:
            return

        try:
            if not audio_data and is_final:
                logger.info(f"TTS audio stream complete for call {call_sid}")
                self.active_calls[call_sid].is_ai_speaking = False
                self.active_calls[call_sid].ai_speaking_start_time = None
                return

            # Set speaking state when we start sending audio
            if audio_data and not self.active_calls[call_sid].is_ai_speaking:
                self.active_calls[call_sid].is_ai_speaking = True
                self.active_calls[call_sid].ai_speaking_start_time = datetime.now()

            # Get the call's WebSocket connection
            websocket = self.active_calls[call_sid].websocket

            # Only send non-empty audio data
            if audio_data:
                # Create a message in Twilio Media Streams format
                message = {
                    "event": "media",
                    "streamSid": self.active_calls[call_sid].metadata.get("stream_sid"),
                    "media": {
                        "payload": base64.b64encode(audio_data).decode(
                            "utf-8"
                        ),  # Base64 encode audio data
                        "track": "outbound",  # This is audio going to the client
                    },
                }

                # Log audio chunk size for debugging
                logger.debug(
                    f"Sending audio chunk of size {len(audio_data)} bytes to Twilio"
                )

                # Send the audio data through the WebSocket
                await websocket.send_json(message)

        except Exception as e:
            logger.error(
                f"Error handling TTS audio for call {call_sid}: {e}", exc_info=True
            )
            self.active_calls[call_sid].is_ai_speaking = False
            self.active_calls[call_sid].ai_speaking_start_time = None

    async def _handle_interruption(
        self, call_sid: str, interrupting_transcript: Optional[str] = None
    ) -> None:
        """
        Handle user interruption by stopping AI speech and processing.

        Args:
            call_sid: The Twilio call SID
            interrupting_transcript: The transcript that caused the interruption
        """
        if call_sid not in self.active_calls:
            return

        try:
            # Send clear message to Twilio to stop buffered audio
            clear_message = {
                "event": "clear",
                "streamSid": self.active_calls[call_sid].metadata.get("stream_sid"),
            }
            await self.active_calls[call_sid].websocket.send_json(clear_message)

            # Stop TTS synthesis
            await self.tts_service.stop_synthesis(call_sid)

            # Reset speaking state
            self.active_calls[call_sid].is_ai_speaking = False
            self.active_calls[call_sid].ai_speaking_start_time = None

            logger.info(f"Handled interruption for call {call_sid}")

            # Process the interrupting speech if provided
            if interrupting_transcript and self.active_calls[call_sid].is_active:
                logger.info(
                    f"Processing interrupting speech: {interrupting_transcript}"
                )
                await self.llm_service.process_transcript(
                    call_sid=call_sid,
                    transcript=interrupting_transcript,
                    is_final=True,  # Always treat interrupting speech as final
                    metadata={
                        "interrupted": True,  # Mark as interrupted speech
                        "to_number": self.active_calls[call_sid].to_number,
                        "from_number": self.active_calls[call_sid].from_number,
                    },
                    response_callback=lambda content, is_final, response_metadata: self._handle_llm_response(
                        call_sid=call_sid,
                        content=content,
                        is_final=is_final,
                        metadata=response_metadata,
                    ),
                )

        except Exception as e:
            logger.error(
                f"Error handling interruption for call {call_sid}: {e}", exc_info=True
            )

    async def _handle_llm_response(
        self, call_sid: str, content: str, is_final: bool, metadata: Dict[str, Any]
    ) -> None:
        """
        Handle LLM response and send it for text-to-speech processing.

        Args:
            call_sid: The Twilio call SID
            content: The response content
            is_final: Whether this is the final response
            metadata: Additional metadata about the response
        """
        try:
            if call_sid not in self.active_calls:
                return

            # Check for special actions
            if is_final and metadata.get("action"):
                if metadata["action"] == "end_call":
                    logger.info(f"Ending call {call_sid} due to LLM endCall tool")
                    await self.end_call(call_sid, with_twilio=True)
                    return
                elif metadata["action"] == "transfer_call":
                    destination = metadata.get("destination")
                    if destination:
                        logger.info(f"Transferring call {call_sid} to {destination}")
                        
                        # Import TwilioService for call transfer
                        from app.twilio.twilio_service import TwilioService
                        
                        # Attempt to transfer the call using TwilioService
                        transfer_success = TwilioService.transfer_call(
                            call_sid=call_sid,
                            destination=destination
                        )
                        
                        if transfer_success:
                            logger.info(f"Successfully initiated transfer for call {call_sid}")
                        else:
                            logger.error(f"Failed to transfer call {call_sid}")
                        
                        # End the call in our system after transfer
                        return

            # Process text through TTS service
            if content:
                await self.tts_service.process_text(
                    call_sid=call_sid,
                    text=content,
                    force_flush=is_final,  # Force flush when it's the final response
                )

            # Log the response
            if is_final:
                response = metadata.get("full_response", "")
                logger.info(f"Final LLM response for {call_sid}: {response}")
            else:
                logger.debug(f"LLM response chunk for {call_sid}: {content}")

        except Exception as e:
            logger.error(
                f"Error handling LLM response for call {call_sid}: {e}", exc_info=True
            )

    async def handle_transcript(
        self, call_sid: str, transcript: str, is_final: bool, metadata: Dict[str, Any]
    ) -> None:
        """
        Handle incoming transcript and process with LLM.

        Args:
            call_sid: The Twilio call SID
            transcript: The transcribed text
            is_final: Whether this is a final transcript
            metadata: Additional metadata about the transcript
        """
        if call_sid not in self.active_calls:
            logger.warning(f"Received transcript for unknown call {call_sid}")
            return

        # Don't process if call is no longer active
        if not self.active_calls[call_sid].is_active:
            logger.debug(f"Ignoring transcript for inactive call {call_sid}")
            return

        if not transcript.strip():
            return

        try:
            # Check if we're in cooldown period
            last_interruption = self.active_calls[call_sid].last_interruption_time
            if last_interruption:
                cooldown_elapsed = (datetime.now() - last_interruption).total_seconds()
                if cooldown_elapsed < self.active_calls[call_sid].interruption_cooldown:
                    # We're in cooldown, buffer the transcript
                    if not self.active_calls[call_sid].is_in_cooldown:
                        self.active_calls[call_sid].is_in_cooldown = True
                        self.active_calls[call_sid].transcript_buffer = ""
                        logger.debug(f"Entering cooldown period for call {call_sid}")

                    # Add to buffer if it's not empty
                    if transcript.strip():
                        self.active_calls[call_sid].transcript_buffer += (
                            " " + transcript.strip()
                        )
                        logger.debug(
                            f"Buffered transcript during cooldown: {self.active_calls[call_sid].transcript_buffer}"
                        )

                    # If this is final, process the buffered transcript
                    if is_final and self.active_calls[call_sid].transcript_buffer:
                        buffered_transcript = self.active_calls[
                            call_sid
                        ].transcript_buffer.strip()
                        logger.info(
                            f"Processing buffered transcript after cooldown: {buffered_transcript}"
                        )
                        await self.llm_service.process_transcript(
                            call_sid=call_sid,
                            transcript=buffered_transcript,
                            is_final=True,
                            metadata={
                                **metadata,
                                "buffered": True,
                                "to_number": self.active_calls[call_sid].to_number,
                                "from_number": self.active_calls[call_sid].from_number,
                            },
                            response_callback=lambda content, is_final, response_metadata: self._handle_llm_response(
                                call_sid=call_sid,
                                content=content,
                                is_final=is_final,
                                metadata=response_metadata,
                            ),
                        )
                        # Reset buffer and cooldown state
                        self.active_calls[call_sid].transcript_buffer = ""
                        self.active_calls[call_sid].is_in_cooldown = False
                    return

            # Only check for interruption on final transcripts to avoid multiple triggers
            if is_final and self.active_calls[call_sid].is_ai_speaking:
                # Only consider it an interruption if AI has been speaking for minimum time
                speaking_start = self.active_calls[call_sid].ai_speaking_start_time
                if speaking_start:
                    speaking_duration = (
                        datetime.now() - speaking_start
                    ).total_seconds()
                    if (
                        speaking_duration
                        >= self.active_calls[call_sid].min_speaking_time
                    ):
                        word_count = len(transcript.strip().split())
                        if (
                            word_count
                            >= self.active_calls[call_sid].interruption_threshold
                        ):
                            logger.info(
                                f"Detected interruption in call {call_sid} with {word_count} words "
                                f"after {speaking_duration:.2f} seconds of AI speaking"
                            )
                            # Update last interruption time
                            self.active_calls[call_sid].last_interruption_time = (
                                datetime.now()
                            )
                            # Pass the interrupting transcript to handle_interruption
                            await self._handle_interruption(call_sid, transcript)
                            return
                    else:
                        logger.debug(
                            f"Ignoring potential interruption after {speaking_duration:.2f} seconds "
                            f"(minimum {self.active_calls[call_sid].min_speaking_time} seconds required)"
                        )

            # Process with LLM if it's a final transcript and call is still active
            # and not already handled as an interruption
            if is_final and self.active_calls[call_sid].is_active:
                logger.info(f"Processing final transcript for {call_sid}: {transcript}")

                # Process transcript with LLM
                await self.llm_service.process_transcript(
                    call_sid=call_sid,
                    transcript=transcript,
                    is_final=is_final,
                    metadata={
                        **metadata,
                        "to_number": self.active_calls[call_sid].to_number,
                        "from_number": self.active_calls[call_sid].from_number,
                    },
                    response_callback=lambda content, is_final, response_metadata: self._handle_llm_response(
                        call_sid=call_sid,
                        content=content,
                        is_final=is_final,
                        metadata=response_metadata,
                    ),
                )

        except Exception as e:
            logger.error(
                f"Error handling transcript for call {call_sid}: {e}", exc_info=True
            )
            # Only send error notification if call is still active
            if call_sid in self.active_calls and self.active_calls[call_sid].is_active:
                await self._handle_llm_response(
                    call_sid=call_sid,
                    content="I apologize, but I'm having trouble processing that right now.",
                    is_final=True,
                    metadata={"error": str(e)},
                )

    async def handle_audio(
        self,
        call_sid: str,
        audio_data: bytes,
        sample_rate: int = 8000,
        channels: int = 1,
    ) -> bool:
        """
        Handle incoming audio data.

        Args:
            call_sid: The Twilio call SID
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate
            channels: Number of audio channels

        Returns:
            bool: Whether the audio was processed successfully
        """
        if call_sid not in self.active_calls:
            logger.warning(f"Received audio for unknown call {call_sid}")
            return False

        try:
            # Send audio to Deepgram
            return await self.deepgram_service.send_audio(
                call_sid=call_sid, audio_data=audio_data
            )
        except Exception as e:
            logger.error(
                f"Error handling audio for call {call_sid}: {e}", exc_info=True
            )
            return False

    async def end_call(self, call_sid: str, with_twilio=False) -> None:
        """
        End call handling and clean up resources.

        Args:
            call_sid: The Twilio call SID
            with_twilio: Whether to end the call via Twilio API
        """
        if call_sid not in self.active_calls:
            logger.debug(f"Call {call_sid} is not active, skipping end_call")
            return

        try:
            # Mark call as inactive first to prevent new processing
            self.active_calls[call_sid].is_active = False

            # If we need to end the call with Twilio, use TwilioService
            if with_twilio:
                from app.twilio.twilio_service import TwilioService
                
                # Attempt to end the call using TwilioService
                end_success = TwilioService.end_call(call_sid=call_sid)
                
                if end_success:
                    logger.info(f"Successfully ended call {call_sid} via Twilio API")
                else:
                    logger.error(f"Failed to end call {call_sid} via Twilio API")

                return

            # Stop transcription first
            await self.deepgram_service.stop_transcription(call_sid)

            # End TTS session
            await self.tts_service.end_session(call_sid)

            # End LLM conversation
            await self.llm_service.end_conversation(call_sid)

            # Clean up call state - save a reference for logging before deletion
            call_state = self.active_calls.get(call_sid)
            
            # Only try to delete if the key still exists (may have been removed by another concurrent process)
            if call_sid in self.active_calls:
                del self.active_calls[call_sid]
                logger.info(f"Ended handling call {call_sid}")
            else:
                logger.warning(f"Call {call_sid} was already removed from active_calls")

        except Exception as e:
            logger.error(f"Error ending call {call_sid}: {e}", exc_info=True)
            # Ensure call state is cleaned up even if there's an error
            if call_sid in self.active_calls:
                del self.active_calls[call_sid]

    def get_call_state(self, call_sid: str) -> Optional[CallState]:
        """
        Get the current state of a call.

        Args:
            call_sid: The Twilio call SID

        Returns:
            Optional[CallState]: The call state if the call exists
        """
        return self.active_calls.get(call_sid)

    def get_conversation_history(self, call_sid: str) -> list:
        """
        Get the conversation history for a call.

        Args:
            call_sid: The Twilio call SID

        Returns:
            list: The conversation history
        """
        return self.llm_service.get_conversation_history(call_sid)
