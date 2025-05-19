"""
This file contains the API endpoints for the Twilio call.
"""

# pylint: disable=logging-format-interpolation,logging-fstring-interpolation,broad-exception-caught
import logging
import json
import base64
from fastapi import Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from fastapi import APIRouter
from twilio.twiml.voice_response import VoiceResponse, Connect

from app.core.call_handler import CallHandler
from app.services.deepgram_service import DeepgramService
from app.services.llm_service import LLMService
from app.services.tts_service import TTSService

router = APIRouter()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize services
deepgram_service = DeepgramService()
llm_service = LLMService()
tts_service = TTSService()

# Initialize call handler with default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant for a customer service call center. 
Your role is to assist customers professionally and efficiently. 
Keep responses concise, clear, and focused on resolving customer needs."""

call_handler = CallHandler(
    deepgram_service=deepgram_service,
    llm_service=llm_service,
    tts_service=tts_service,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
)


@router.post("/twiml")
async def get_twiml(request: Request):
    """
    Generate TwiML for incoming Twilio calls.
    This endpoint is called by Twilio when a call comes in.
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    to_phone_number = form_data.get("To")
    customer_phone_number = form_data.get("From")

    logger.info(
        f"Incoming call with SID: {call_sid} from {customer_phone_number} to {to_phone_number}"
    )
    logger.info("Request headers: %s", request.headers)

    # Create the TwiML response
    response = VoiceResponse()

    # Create a <Connect> verb with the WebSocket stream
    connect = Connect()

    # Setup a bi-directional WebSocket stream
    # Check for ngrok forwarding
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")

    # Determine protocol (ws:// or wss://)
    # When using ngrok, we should use wss:// for the forwarded connection
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_proto == "https":
        protocol = "wss"
    elif request.url.scheme == "https":
        protocol = "wss"
    else:
        protocol = "ws"

    # Build the stream URL
    stream_url = f"{protocol}://{host}/streams"

    # Add phone numbers as query params if available
    if to_phone_number and customer_phone_number:
        stream_url += f"?to={to_phone_number}&from={customer_phone_number}"

    logger.info("Setting up WebSocket stream at URL: %s", stream_url)
    connect.stream(url=stream_url)

    # Add the <Connect> verb to the response
    response.append(connect)

    # Log the full TwiML response
    twiml_response = str(response)
    logger.info(f"Returning TwiML response: {twiml_response}")

    return Response(content=twiml_response, media_type="application/xml")


@router.websocket("/streams")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    Handles real-time audio streaming with Twilio.
    """
    logger.info("WebSocket connection attempt from %s", websocket.client.host)
    logger.info("WebSocket request headers: %s", websocket.headers)
    logger.info("WebSocket query parameters: %s", websocket.query_params)

    stream_sid = None
    call_sid = None

    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")

        query_params = websocket.query_params
        to_phone_number = query_params.get("to")
        customer_phone_number = query_params.get("from")

        try:
            # Process incoming WebSocket messages
            async for message in websocket.iter_json():
                event_type = message.get("event")

                if event_type == "start":
                    logger.info(f"Media stream 'start' event received: {message}")
                    # Extract streamSid and callSid from the start message
                    stream_sid = message.get("streamSid")
                    call_sid = message.get("start", {}).get("callSid")

                    if not stream_sid or not call_sid:
                        logger.error("Missing streamSid or callSid in start message")
                        continue

                    tracks = message.get("start", {}).get("tracks", [])
                    media_format = message.get("start", {}).get("mediaFormat", {})

                    # Start call handling
                    try:
                        # Start the call handler with WebSocket
                        await call_handler.start_call(
                            call_sid=call_sid,
                            websocket=websocket,
                            to_number=to_phone_number,
                            from_number=customer_phone_number,
                            metadata={
                                "stream_sid": stream_sid,
                                "media_format": media_format,
                                "tracks": tracks,
                            },
                        )
                        
                        # Start Deepgram transcription
                        sample_rate = int(media_format.get("rate", 8000))
                        channels = int(media_format.get("channels", 1))
                        
                        # Create a callback function that includes all required parameters
                        async def transcription_callback(transcript: str, is_final: bool, metadata: dict) -> None:
                            await call_handler.handle_transcript(
                                call_sid=call_sid,
                                transcript=transcript,
                                is_final=is_final,
                                metadata=metadata,
                            )
                        
                        success = await deepgram_service.start_transcription(
                            call_sid,
                            transcription_callback,  # Use the properly defined callback
                            sample_rate=sample_rate,
                            channels=channels,
                        )
                        
                        if success:
                            logger.info(f"Started transcription for call: {call_sid}")
                            
                            # Send a welcome message via TTS
                            await tts_service.process_text(
                                call_sid=call_sid,
                                text="Hello! I'm your AI assistant. How can I help you today?<flush/>",
                                force_flush=True
                            )
                        else:
                            logger.error(
                                f"Failed to start transcription for call: {call_sid}"
                            )
                            
                    except Exception as e:
                        logger.error(
                            f"Error starting call handling: {e}", exc_info=True
                        )

                    # Send a mark event to acknowledge the start message
                    await websocket.send_json(
                        {
                            "event": "mark",
                            "streamSid": stream_sid,
                            "mark": {"name": "start_acknowledged"},
                        }
                    )

                elif event_type == "media":
                    # Process incoming audio
                    audio_data = message.get("media", {}).get("payload", "")
                    track = message.get("media", {}).get("track", "inbound")

                    current_stream_sid = message.get("streamSid")
                    if not current_stream_sid:
                        logger.error("Missing streamSid in media message")
                        continue

                    if track == "inbound" and audio_data and call_sid:
                        try:
                            # Decode the base64 audio data to bytes
                            decoded_audio = base64.b64decode(audio_data)

                            # Handle audio through call handler
                            await call_handler.handle_audio(call_sid, decoded_audio)

                        except Exception as audio_error:
                            logger.error(
                                f"Error processing audio data: {audio_error}",
                                exc_info=True,
                            )

                elif event_type == "stop":
                    # Get the current stream_sid from the message
                    current_stream_sid = message.get("streamSid")
                    if not current_stream_sid:
                        logger.error("Missing streamSid in stop message")
                        continue

                    # End call handling if we have a call_sid
                    if call_sid:
                        try:
                            await call_handler.end_call(call_sid)
                            logger.info(f"Ended call handling for call: {call_sid}")
                        except Exception as stop_error:
                            logger.error(
                                f"Error ending call handling: {stop_error}",
                                exc_info=True,
                            )
                    else:
                        logger.error(
                            f"Received stop event but call_sid is None for stream {current_stream_sid}"
                        )
                else:
                    logger.debug(f"Received message with event type: {event_type}")

        except WebSocketDisconnect:
            logger.warning(
                f"WebSocket disconnected during message exchange for: {stream_sid}"
            )
            # End call handling if connection is lost
            if call_sid:
                try:
                    await call_handler.end_call(call_sid)
                    logger.info(
                        f"Ended call handling after WebSocket disconnect for call: {call_sid}"
                    )
                except Exception as e:
                    logger.error(f"Error ending call handling: {e}", exc_info=True)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in WebSocket: {e}")

        except Exception as e:
            logger.error(f"Error processing WebSocket messages: {e}", exc_info=True)

    except WebSocketDisconnect:
        logger.warning(f"WebSocket disconnected before acceptance")

    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}", exc_info=True)
