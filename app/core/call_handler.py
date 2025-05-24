"""
This file contains the CallHandler class for managing call state and conversation flow.
"""

# pylint: disable=too-many-ancestors,logging-fstring-interpolation,broad-exception-caught
import logging
import base64
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from fastapi import WebSocket

from app.services.deepgram_service import DeepgramService
from app.services.llm_service import LLMService
from app.services.tts_service import TTSService
from app.services.call_service import CallService
from app.services.webhook_service import WebhookService
from app.twilio.twilio_service import TwilioService

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
    llm_service: Optional[Any] = None  # LLM service for this call
    deepgram_service: Optional[Any] = None  # Deepgram service for this call
    tts_service: Optional[Any] = None  # TTS service for this call
    assistant: Optional[Any] = None  # Assistant to use for this call
    
    # Idle timeout tracking
    last_activity_time: datetime = field(default_factory=datetime.now)  # Track last activity
    idle_message_count: int = 0  # Count of idle messages sent
    idle_timeout_task: Optional[Any] = None  # Background task for idle timeout
    idle_timeout_seconds: Optional[int] = None  # Timeout in seconds
    max_idle_messages: Optional[int] = None  # Max idle messages before ending call
    idle_message: Optional[str] = None  # Message to send when idle


class CallHandler:
    """
    Handles the lifecycle of a call, managing transcription, LLM processing,
    and conversation state.
    """

    def __init__(self):
        """
        Initialize the call handler.
        Configuration is now handled per-call through assistant objects.
        """
        # Track active calls
        self.active_calls: Dict[str, CallState] = {}

    async def start_call(
        self,
        call_sid: str,
        websocket: WebSocket,
        to_number: Optional[str] = None,
        from_number: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        assistant: Optional[Any] = None,
    ) -> None:
        """
        Start handling a new call.

        Args:
            call_sid: The Twilio call SID
            websocket: The WebSocket connection for this call
            to_number: The destination phone number
            from_number: The caller's phone number
            metadata: Additional call metadata
            assistant: The assistant to use for this call
        """
        # Create call record in database
        if assistant:
            try:
                await CallService.create_call(
                    assistant_id=assistant.id,
                    call_sid=call_sid,
                    to_phone_number=to_number or "",
                    customer_phone_number=from_number or "",
                    metadata=metadata or {},
                )
                logger.info(f"Created call record in database for call {call_sid}")
            except Exception as e:
                logger.error(f"Error creating call record for {call_sid}: {e}", exc_info=True)

        # Initialize call state
        self.active_calls[call_sid] = CallState(
            call_sid=call_sid,
            websocket=websocket,
            to_number=to_number,
            from_number=from_number,
            metadata=metadata or {},
            assistant=assistant,
        )

        # Set interruption settings from assistant if available
        if assistant and assistant.interruption_settings:
            settings = assistant.interruption_settings
            self.active_calls[call_sid].interruption_threshold = settings.get(
                "interruption_threshold", 3
            )
            self.active_calls[call_sid].min_speaking_time = settings.get(
                "min_speaking_time", 0.5
            )
            self.active_calls[call_sid].interruption_cooldown = settings.get(
                "interruption_cooldown", 2.0
            )

        # Set idle timeout settings from assistant if available
        if assistant:
            self.active_calls[call_sid].idle_timeout_seconds = assistant.idle_timeout
            self.active_calls[call_sid].max_idle_messages = assistant.max_idle_messages
            self.active_calls[call_sid].idle_message = (
                assistant.idle_message or 
                "Are you still there? I'm here to help if you need anything."
            )

        # Create a dedicated LLM service instance for this call
        # The new multi-provider LLMService handles configuration through the assistant object
        self.active_calls[call_sid].llm_service = LLMService(
            call_sid=call_sid,
            to_number=to_number,
            from_number=from_number,
            assistant=assistant,
        )

        # Get Deepgram configuration from assistant
        deepgram_api_key = None
        stt_settings = {}

        if assistant:
            deepgram_api_key = assistant.deepgram_api_key
            if assistant.stt_settings:
                stt_settings = assistant.stt_settings

        # Create Deepgram service with assistant settings
        self.active_calls[call_sid].deepgram_service = DeepgramService(
            call_sid=call_sid,
            api_key=deepgram_api_key,
            model=stt_settings.get("model", "nova-3"),
            language=stt_settings.get("language", "en-US"),
            punctuate=stt_settings.get("punctuate", True),
            interim_results=stt_settings.get("interim_results", True),
            endpointing=stt_settings.get("endpointing", {}).get("silence_threshold", 100),
            utterance_end_ms=stt_settings.get("utterance_end_ms", 1000),
            smart_format=stt_settings.get("smart_format", True),
            keywords=stt_settings.get("keywords", []),
            keyterms=stt_settings.get("keyterms", []),
        )

        # Get TTS configuration from assistant
        elevenlabs_api_key = None
        voice_id = None
        model_id = None
        stability = 0.5
        similarity_boost = 0.75
        style = 0.0
        use_speaker_boost = True
        latency = 1

        if assistant and assistant.tts_settings:
            # Get ElevenLabs API key
            elevenlabs_api_key = assistant.elevenlabs_api_key

            # Extract TTS settings
            settings = assistant.tts_settings
            voice_id = TTSService.get_voice_id(settings.get("voice_id", "rachel"))
            model_id = TTSService.get_model_id(settings.get("model_id", "turbo"))
            stability = settings.get("stability", 0.5)
            similarity_boost = settings.get("similarity_boost", 0.75)
            style = settings.get("style", 0.0)
            use_speaker_boost = settings.get("use_speaker_boost", True)
            latency = settings.get("latency", 1)

        # Create TTS service with assistant settings
        self.active_calls[call_sid].tts_service = TTSService(
            call_sid=call_sid,
            api_key=elevenlabs_api_key,
            voice_id=voice_id,
            model_id=model_id,
            stability=stability,
            similarity_boost=similarity_boost,
            style=style,
            use_speaker_boost=use_speaker_boost,
            latency=latency,
        )

        # Define audio callback to handle TTS audio
        async def audio_callback(
            audio_data: bytes, is_final: bool, audio_metadata: Dict[str, Any]
        ) -> None:
            await self._handle_tts_audio(call_sid, audio_data, is_final, audio_metadata)

        # Start TTS session with configured options
        await self.active_calls[call_sid].tts_service.start_session(
            audio_callback=audio_callback,
            metadata={"call_sid": call_sid},
        )

        # Start Deepgram transcription
        sample_rate = int(metadata.get("media_format", {}).get("rate", 8000))
        channels = int(metadata.get("media_format", {}).get("channels", 1))

        # Define transcription callback
        async def transcription_callback(
            transcript: str, is_final: bool, metadata: dict
        ) -> None:
            await self.handle_transcript(
                call_sid=call_sid,
                transcript=transcript,
                is_final=is_final,
                metadata=metadata,
            )

        # Start transcription with sample rate and channels from media format
        success = await self.active_calls[
            call_sid
        ].deepgram_service.start_transcription(
            transcript_callback=transcription_callback,
            sample_rate=sample_rate,
            channels=channels,
        )

        if success:
            logger.info(f"Started transcription for call: {call_sid}")

            # Start recording asynchronously (don't wait for it)
            asyncio.create_task(self._start_call_recording_async(call_sid, assistant, websocket))

            # Get welcome message from assistant or use default
            welcome_message = (
                "Hello! I'm your AI assistant. How can I help you today?<flush/>"
            )
            if (
                assistant
                and assistant.llm_settings
                and "welcome_message" in assistant.llm_settings
            ):
                welcome_message = (
                    assistant.llm_settings.get("welcome_message") + "<flush/>"
                )

            # Send a welcome message via TTS
            await self.active_calls[call_sid].tts_service.process_text(
                text=welcome_message,
                force_flush=True
            )

            # Start idle timeout monitoring if configured
            if (self.active_calls[call_sid].idle_timeout_seconds and 
                self.active_calls[call_sid].idle_timeout_seconds > 0):
                self.active_calls[call_sid].idle_timeout_task = asyncio.create_task(
                    self._monitor_idle_timeout(call_sid)
                )
                logger.info(f"Started idle timeout monitoring for call {call_sid} "
                           f"with {self.active_calls[call_sid].idle_timeout_seconds}s timeout, "
                           f"max {self.active_calls[call_sid].max_idle_messages} messages, "
                           f"message: '{self.active_calls[call_sid].idle_message}'")
            else:
                logger.info(f"No idle timeout configured for call {call_sid} "
                           f"(timeout: {self.active_calls[call_sid].idle_timeout_seconds})")

            # Send initial webhook status update after call starts
            if assistant:
                try:
                    call = await CallService.get_call_by_sid(call_sid)
                    if call:
                        asyncio.create_task(
                            WebhookService.send_status_update_webhook(
                                assistant=assistant,
                                call=call,
                                status="in-progress",
                                messages=[]
                            )
                        )
                        logger.info(f"Sent initial webhook status update for call {call_sid}")
                except Exception as e:
                    logger.error(f"Error sending initial webhook for call {call_sid}: {e}")

        else:
            logger.error(f"Failed to start transcription for call: {call_sid}")

        logger.info(f"Started handling call {call_sid}")

    async def _start_call_recording_async(self, call_sid: str, assistant: Any, websocket: WebSocket) -> None:
        """
        Start Twilio call recording asynchronously.

        Args:
            call_sid: The Twilio call SID
            assistant: The assistant instance
            websocket: The WebSocket connection to get host info
        """
        try:
            # Get Twilio credentials from assistant or environment
            account_sid = assistant.twilio_account_sid if assistant else None
            auth_token = assistant.twilio_auth_token if assistant else None

            # Build recording status callback URL
            # Get host from WebSocket headers
            host = websocket.headers.get("host", "localhost:8000")
            
            # Determine protocol based on host
            protocol = "https" if "ngrok" in host or "herokuapp" in host else "http"
            if "https" in host:
                protocol = "https"
            else:
                protocol = "http"
            recording_callback_url = f"{protocol}://{host}/recording-status"

            # Start recording via Twilio API
            recording_sid = TwilioService.start_call_recording(
                call_sid=call_sid,
                recording_channels="dual",
                recording_status_callback=recording_callback_url,
                account_sid=account_sid,
                auth_token=auth_token,
            )

            if recording_sid:
                # Create recording record in database
                await CallService.create_twilio_recording(
                    call_sid=call_sid,
                    recording_sid=recording_sid,
                    status="recording",
                )
                logger.info(f"Started Twilio recording {recording_sid} for call {call_sid}")
            else:
                logger.error(f"Failed to start Twilio recording for call {call_sid}")

        except Exception as e:
            logger.error(f"Error starting call recording: {e}", exc_info=True)

    async def _monitor_idle_timeout(self, call_sid: str) -> None:
        """
        Monitor idle timeout for a call and send idle messages when appropriate.
        
        Args:
            call_sid: The Twilio call SID
        """
        if call_sid not in self.active_calls:
            return
            
        try:
            while self.active_calls[call_sid].is_active:
                # Sleep for 1 second intervals to check timeout
                await asyncio.sleep(1.0)
                
                if call_sid not in self.active_calls or not self.active_calls[call_sid].is_active:
                    break
                
                call_state = self.active_calls[call_sid]
                
                # Skip if no timeout configured
                if not call_state.idle_timeout_seconds:
                    continue
                
                # Calculate time since last activity
                time_since_activity = (datetime.now() - call_state.last_activity_time).total_seconds()
                
                # Log debug info every 10 seconds for troubleshooting
                if int(time_since_activity) % 10 == 0 and time_since_activity > 0:
                    logger.debug(f"Idle monitoring for call {call_sid}: {time_since_activity:.1f}s since last activity (threshold: {call_state.idle_timeout_seconds}s)")
                
                # Check if we've exceeded the idle timeout
                if time_since_activity >= call_state.idle_timeout_seconds:
                    # Check if we've reached max idle messages
                    if (call_state.max_idle_messages and 
                        call_state.idle_message_count >= call_state.max_idle_messages):
                        logger.info(f"Max idle messages ({call_state.max_idle_messages}) reached for call {call_sid}, ending call")
                        await self.end_call(call_sid, with_twilio=True)
                        break
                    
                    # Send idle message
                    if call_state.idle_message:
                        logger.info(f"Sending idle message to call {call_sid} after {time_since_activity:.1f}s of inactivity")
                        await call_state.tts_service.process_text(
                            text=call_state.idle_message + "<flush/>",
                            force_flush=True
                        )
                        
                        # Increment idle message count and reset activity timer
                        call_state.idle_message_count += 1
                        call_state.last_activity_time = datetime.now()
                        
                        logger.info(f"Sent idle message #{call_state.idle_message_count} to call {call_sid}")
                        
        except asyncio.CancelledError:
            logger.info(f"Idle timeout monitoring cancelled for call {call_sid}")
        except Exception as e:
            logger.error(f"Error in idle timeout monitoring for call {call_sid}: {e}", exc_info=True)

    def _reset_idle_timer(self, call_sid: str) -> None:
        """
        Reset the idle timer for a call to indicate activity.
        
        Args:
            call_sid: The Twilio call SID
        """
        if call_sid in self.active_calls:
            self.active_calls[call_sid].last_activity_time = datetime.now()
            logger.debug(f"Reset idle timer for call {call_sid} due to activity")

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
                # Reset idle timer when AI starts speaking meaningful content
                # Only reset once at the start of speaking, not on every audio chunk
                self._reset_idle_timer(call_sid)

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
            await self.active_calls[call_sid].tts_service.stop_synthesis()

            # Reset speaking state
            self.active_calls[call_sid].is_ai_speaking = False
            self.active_calls[call_sid].ai_speaking_start_time = None

            logger.info(f"Handled interruption for call {call_sid}")

            # Process the interrupting speech if provided
            if interrupting_transcript and self.active_calls[call_sid].is_active:
                logger.info(
                    f"Processing interrupting speech: {interrupting_transcript}"
                )
                
                # Note: Transcript is already stored in handle_transcript method
                # Just process with LLM for interruption response
                await self.active_calls[call_sid].llm_service.process_transcript(
                    transcript=interrupting_transcript,
                    is_final=True,  # Always treat interrupting speech as final
                    metadata={"interrupted": True},  # Mark as interrupted speech
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

            # Check if this is a message that should be spoken before an action
            if metadata.get("speak_before_action"):
                if content:
                    logger.info(f"Speaking message before action for {call_sid}: {content}")
                    await self.active_calls[call_sid].tts_service.process_text(
                        text=content,
                        force_flush=True,  # Force flush to ensure message is spoken
                    )
                    # Wait a bit for the message to be spoken before continuing
                    await asyncio.sleep(1.0)
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

                        # Attempt to transfer the call using TwilioService
                        transfer_success = TwilioService.transfer_call(
                            call_sid=call_sid, destination=destination
                        )

                        if transfer_success:
                            logger.info(
                                f"Successfully initiated transfer for call {call_sid}"
                            )
                        else:
                            logger.error(f"Failed to transfer call {call_sid}")

                        # End the call in our system after transfer
                        return

            # Process text through TTS service
            if content:
                await self.active_calls[call_sid].tts_service.process_text(
                    text=content,
                    force_flush=is_final,  # Force flush when it's the final response
                )
                # Only reset idle timer when AI responds with actual content
                if content.strip():
                    self._reset_idle_timer(call_sid)

            # Log the response and store final responses as transcripts
            if is_final:
                response = metadata.get("full_response", "")
                logger.info(f"Final LLM response for {call_sid}: {response}")
                
                # Store assistant response in database asynchronously
                if response:
                    # Clean up flush tags and other formatting before storing
                    cleaned_response = response.replace("<flush/>", "").replace("<flush>", "").strip()
                    if cleaned_response:  # Only store if there's actual content
                        asyncio.create_task(
                            CallService.create_transcript(
                                call_sid=call_sid,
                                content=cleaned_response,
                                is_final=True,
                                speaker="assistant",
                            )
                        )
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
            # Only reset idle timer on final transcripts with meaningful content
            # Don't reset on interim results to avoid preventing idle detection
            if is_final and transcript.strip():
                self._reset_idle_timer(call_sid)
            
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
                        
                        # Store buffered transcript in database
                        if buffered_transcript:
                            asyncio.create_task(
                                CallService.create_transcript(
                                    call_sid=call_sid,
                                    content=buffered_transcript,
                                    is_final=True,
                                    speaker="user",
                                )
                            )
                        
                        # Process with LLM
                        await self.active_calls[
                            call_sid
                        ].llm_service.process_transcript(
                            transcript=buffered_transcript,
                            is_final=True,
                            metadata={**metadata, "buffered": True},
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

            # Process with LLM if it's a final transcript and call is still active
            # and not already handled as an interruption
            if is_final and self.active_calls[call_sid].is_active:
                logger.info(f"Processing final transcript for {call_sid}: {transcript}")

                # Clean up transcript content before storing
                cleaned_transcript = transcript.strip()
                if cleaned_transcript:  # Only process and store if there's actual content
                    
                    # Always store user transcript first (before any interruption processing)
                    asyncio.create_task(
                        CallService.create_transcript(
                            call_sid=call_sid,
                            content=cleaned_transcript,
                            is_final=is_final,
                            speaker="user",
                            confidence=metadata.get("confidence"),
                        )
                    )
                    
                    # Check if this should be treated as an interruption
                    is_interruption = False
                    if self.active_calls[call_sid].is_ai_speaking:
                        speaking_start = self.active_calls[call_sid].ai_speaking_start_time
                        if speaking_start:
                            speaking_duration = (datetime.now() - speaking_start).total_seconds()
                            if speaking_duration >= self.active_calls[call_sid].min_speaking_time:
                                word_count = len(cleaned_transcript.split())
                                if word_count >= self.active_calls[call_sid].interruption_threshold:
                                    logger.info(
                                        f"Detected interruption in call {call_sid} with {word_count} words "
                                        f"after {speaking_duration:.2f} seconds of AI speaking"
                                    )
                                    self.active_calls[call_sid].last_interruption_time = datetime.now()
                                    await self._handle_interruption(call_sid, cleaned_transcript)
                                    is_interruption = True

                    # Only process with LLM if it's not an interruption (interruptions are handled separately)
                    if not is_interruption:
                        # Process transcript with LLM
                        await self.active_calls[call_sid].llm_service.process_transcript(
                            transcript=cleaned_transcript,
                            is_final=is_final,
                            metadata=metadata,
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
            # Note: Don't reset idle timer here as audio packets flow constantly
            # Only reset on meaningful user activity (speech transcripts)
            
            # Send audio to Deepgram
            return await self.active_calls[call_sid].deepgram_service.send_audio(
                audio_data
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

            # Cancel idle timeout task if it exists
            if (call_sid in self.active_calls and 
                self.active_calls[call_sid].idle_timeout_task and 
                not self.active_calls[call_sid].idle_timeout_task.done()):
                self.active_calls[call_sid].idle_timeout_task.cancel()
                logger.info(f"Cancelled idle timeout task for call {call_sid}")

            # If we need to end the call with Twilio, use TwilioService
            if with_twilio:
                end_success = TwilioService.end_call(call_sid=call_sid)

                if end_success:
                    logger.info(f"Successfully ended call {call_sid} via Twilio API")
                else:
                    logger.error(f"Failed to end call {call_sid} via Twilio API")

                return

            # Stop transcription first
            await self.active_calls[call_sid].deepgram_service.stop_transcription()

            # End TTS session
            await self.active_calls[call_sid].tts_service.end_session()

            # Update call status in database
            try:
                start_time = self.active_calls[call_sid].start_time
                duration = int((datetime.now() - start_time).total_seconds())
                await CallService.update_call_status(
                    call_sid=call_sid,
                    status="completed",
                    duration=duration,
                )
                logger.info(f"Updated call {call_sid} status to completed in database")
            except Exception as e:
                logger.error(f"Error updating call status for {call_sid}: {e}", exc_info=True)

            # Send end-of-call webhook if assistant and webhook URL are configured
            assistant = self.active_calls[call_sid].assistant
            if assistant:
                try:
                    # Send delayed webhook that waits for recordings to complete
                    asyncio.create_task(
                        WebhookService.send_end_of_call_webhook_when_ready(
                            call_sid=call_sid,
                            max_wait_seconds=60  # Wait up to 60 seconds for recordings
                        )
                    )
                    logger.info(f"Scheduled delayed end-of-call webhook for call {call_sid}")
                except Exception as e:
                    logger.error(f"Error scheduling end-of-call webhook for call {call_sid}: {e}")

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
        if call_sid in self.active_calls and self.active_calls[call_sid].llm_service:
            return self.active_calls[call_sid].llm_service.get_conversation_history()
        return []
