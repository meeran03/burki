"""
This file contains the API endpoints for the Twilio call.
"""

# pylint: disable=logging-format-interpolation,logging-fstring-interpolation,broad-exception-caught
import logging
import json
import base64
import os
import asyncio
from typing import Optional
from datetime import datetime
from fastapi import Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response
from fastapi import APIRouter
from twilio.twiml.voice_response import VoiceResponse, Connect

from app.core.call_handler import CallHandler
from app.core.assistant_manager import AssistantManager
from app.services.call_service import CallService
from app.twilio.twilio_service import TwilioService
from app.services.billing_service import BillingService

router = APIRouter()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize call handler with default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant for a customer service call center. 
Your role is to assist customers professionally and efficiently. 
Keep responses concise, clear, and focused on resolving customer needs."""

# Get custom LLM URL from environment variable if available
CUSTOM_LLM_URL = os.getenv("CUSTOM_LLM_URL", "http://localhost:8001/ai/chat/completions")

call_handler = CallHandler(
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    custom_llm_url=CUSTOM_LLM_URL,
)
assistant_manager = AssistantManager()


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

    # Get assistant to check billing limits
    if to_phone_number:
        assistant = await assistant_manager.get_assistant_by_phone(to_phone_number)
        if assistant:
            # Check billing limits for the organization
            usage_check = await BillingService.check_usage_limits(assistant.organization_id)
            
            if not usage_check.get("allowed", False):
                # Create a TwiML response to reject the call or play a message
                response = VoiceResponse()
                
                if usage_check.get("needs_upgrade", False):
                    response.say(
                        "Your call cannot be completed at this time. "
                        "Your organization has exceeded its monthly usage limit. "
                        "Please contact your administrator to upgrade your plan or add top-up credits.",
                        voice="alice"
                    )
                else:
                    response.say(
                        "Your call cannot be completed at this time. "
                        "Please try again later.",
                        voice="alice"
                    )
                
                response.hangup()
                
                logger.warning(
                    f"Call rejected due to billing limits for organization {assistant.organization_id}: {usage_check}"
                )
                return Response(content=str(response), media_type="application/xml")

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

    # Create stream with custom parameters for phone numbers
    stream = connect.stream(url=stream_url)
    if to_phone_number:
        stream.parameter(name="To", value=to_phone_number)
    if customer_phone_number:
        stream.parameter(name="From", value=customer_phone_number)

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

    stream_sid = None
    call_sid = None
    to_number = None
    from_number = None

    try:
        await websocket.accept()
        logger.info("WebSocket connection accepted")

        try:
            # Process incoming WebSocket messages
            async for message in websocket.iter_json():
                event_type = message.get("event")

                if event_type == "connected":
                    logger.info(f"Media stream 'connected' event received: {message}")

                elif event_type == "start":
                    logger.info(f"Media stream 'start' event received: {message}")
                    # Extract streamSid and callSid from the start message
                    stream_sid = message.get("streamSid")
                    start_data = message.get("start", {})
                    call_sid = start_data.get("callSid")
                    
                    # Get phone numbers from customParameters if available
                    custom_params = start_data.get("customParameters", {})
                    to_number = custom_params.get("To")
                    from_number = custom_params.get("From")

                    if not stream_sid or not call_sid:
                        logger.error("Missing streamSid or callSid in start message")
                        continue

                    tracks = start_data.get("tracks", [])
                    media_format = start_data.get("mediaFormat", {})

                    # Start call handling
                    try:
                        # Get assistant based on the to_number
                        if to_number:
                            assistant = await assistant_manager.get_assistant_by_phone(to_number)
                            if not assistant:
                                logger.error(f"Assistant not found for phone number: {to_number}")
                                raise HTTPException(status_code=404, detail="Assistant not found")

                        # Start the call handler with WebSocket
                        await call_handler.start_call(
                            call_sid=call_sid,
                            websocket=websocket,
                            to_number=to_number,
                            from_number=from_number,
                            metadata={
                                "stream_sid": stream_sid,
                                "media_format": media_format,
                                "tracks": tracks,
                            },
                            assistant=assistant,
                        )

                        # The welcome message is now handled in the call_handler.start_call method
                        # since DeepgramService is now initialized per-call
                        # Removed redundant deepgram initialization as it's now handled in call_handler.start_call

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


@router.post("/recording-status")
async def recording_status_callback(request: Request):
    """
    Webhook endpoint for Twilio recording status callbacks.
    Called when a recording is completed or fails.
    """
    form_data = await request.form()
    recording_sid = form_data.get("RecordingSid")
    recording_status = form_data.get("RecordingStatus")
    call_sid = form_data.get("CallSid")
    recording_url = form_data.get("RecordingUrl")
    recording_duration = form_data.get("RecordingDuration")

    logger.info(
        f"Recording status callback: SID={recording_sid}, Status={recording_status}, "
        f"CallSID={call_sid}, URL={recording_url}, Duration={recording_duration}"
    )

    if not recording_sid or not call_sid:
        logger.error("Missing recording SID or call SID in callback")
        return Response(content="Missing required parameters", status_code=400)

    try:
        # Convert duration to float if available
        duration = float(recording_duration) if recording_duration else None

        # If recording is completed, download it locally
        local_file_path = None
        if recording_status.lower() == "completed":
            try:
                # Download the recording from Twilio
                download_result = TwilioService.download_recording_content(recording_sid)
                
                if download_result:
                    filename, content = download_result
                    
                    # Save to local recordings directory
                    recordings_dir = os.getenv("RECORDINGS_DIR", "recordings")
                    os.makedirs(recordings_dir, exist_ok=True)
                    
                    # Create call-specific directory
                    call_dir = os.path.join(recordings_dir, call_sid)
                    os.makedirs(call_dir, exist_ok=True)
                    
                    # Save the file
                    local_file_path = os.path.join(call_dir, filename)
                    with open(local_file_path, "wb") as f:
                        f.write(content)
                    
                    logger.info(f"Downloaded and saved recording {recording_sid} to {local_file_path}")
                else:
                    logger.error(f"Failed to download recording {recording_sid}")
            except Exception as e:
                logger.error(f"Error downloading recording {recording_sid}: {e}", exc_info=True)

        # Update recording status in database
        await CallService.update_recording_status(
            recording_sid=recording_sid,
            status=recording_status.lower(),
            recording_url=recording_url,
            duration=duration,
            local_file_path=local_file_path,  # Add the local file path
        )

        logger.info(f"Updated recording {recording_sid} status to {recording_status}")
        
        # If recording is completed, check if we should trigger end-of-call webhook
        if recording_status.lower() == "completed":
            try:
                # Get the call to find assistant info
                call = await CallService.get_call_by_sid(call_sid)
                if call and call.assistant and call.assistant.webhook_url and call.status == "completed":
                    # Check if all recordings for this call are now completed
                    recordings = await CallService.get_call_recordings(call_sid)
                    all_completed = all(r.status in ["completed", "failed"] for r in recordings)
                    
                    if all_completed:
                        # Record billing usage for the completed call
                        try:
                            await BillingService.record_call_usage(call.id)
                            logger.info(f"Recorded billing usage for call {call.id}")
                        except Exception as billing_error:
                            logger.error(f"Error recording billing usage for call {call.id}: {billing_error}")
                        
                        # Send webhook immediately since all recordings are ready
                        from app.services.webhook_service import WebhookService
                        
                        # Get the best recording URL using proper URL construction
                        best_recording = next((r for r in recordings if r.status == "completed"), None)
                        recording_url_for_webhook = None
                        if best_recording:
                            recording_url_for_webhook = WebhookService._construct_recording_url(best_recording)
                        
                        # Send webhook asynchronously - the locking mechanism will prevent duplicates
                        asyncio.create_task(
                            WebhookService.send_end_of_call_webhook(
                                assistant=call.assistant,
                                call=call,
                                ended_reason="customer-ended-call",
                                recording_url=recording_url_for_webhook
                            )
                        )
                        logger.info(f"Triggered immediate end-of-call webhook for completed recording of call {call_sid}")
            except Exception as e:
                logger.error(f"Error triggering webhook for recording completion: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error processing recording status callback: {e}", exc_info=True)
        return Response(content="Error processing callback", status_code=500)

    return Response(content="OK", status_code=200)
