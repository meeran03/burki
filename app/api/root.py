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
from fastapi.responses import Response, FileResponse
from fastapi import APIRouter
from twilio.twiml.voice_response import VoiceResponse, Connect

from app.core.call_handler import CallHandler
from app.core.assistant_manager import AssistantManager
from app.services.call_service import CallService
from app.services.webhook_service import WebhookService
from app.twilio.twilio_service import TwilioService
from app.services.billing_service import BillingService
from app.utils.url_utils import get_twiml_webhook_url

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

# Initialize call handler - configuration is now handled per-call through assistant objects
call_handler = CallHandler()
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

    # Check if this is an outbound call by looking for metadata in URL parameters
    # For outbound calls, the metadata is passed as URL parameters
    query_params = request.query_params
    is_outbound = query_params.get("outbound") == "true"
    outbound_assistant_id = query_params.get("assistant_id")
    outbound_welcome_message = query_params.get("welcome_message")
    outbound_agenda = query_params.get("agenda")
    outbound_to_phone = query_params.get("to_phone_number")

    if is_outbound:
        logger.info(
            f"Outbound call with SID: {call_sid} from {customer_phone_number} to {to_phone_number} "
            f"with assistant {outbound_assistant_id} and agenda: {outbound_agenda[:100] if outbound_agenda else 'None'}..."
        )
    else:
        logger.info(
            f"Incoming call with SID: {call_sid} from {customer_phone_number} to {to_phone_number}"
        )
    
    logger.info("Request headers: %s", request.headers)

    # Get assistant to check billing limits
    assistant = None
    
    if is_outbound and outbound_assistant_id:
        # For outbound calls, use the assistant ID from metadata
        try:
            assistant = await assistant_manager.get_assistant_by_id(int(outbound_assistant_id))
        except (ValueError, TypeError):
            logger.error(f"Invalid assistant_id in outbound call metadata: {outbound_assistant_id}")
    elif to_phone_number:
        # For inbound calls, lookup assistant by phone number
        assistant = await assistant_manager.get_assistant_by_phone(to_phone_number)
    
    if assistant:
        # Check billing limits for the organization - make this faster
        try:
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
        except Exception as billing_error:
            # Don't block calls if billing check fails
            logger.error(f"Billing check failed for {assistant.organization_id}, allowing call: {billing_error}")

    # Create call record in database and send initial webhook as soon as call comes in
    if assistant and call_sid:
        try:
            # For outbound calls, the customer is the "to" number, not the "from"
            if is_outbound:
                actual_customer_phone = outbound_to_phone or to_phone_number
                actual_to_phone = customer_phone_number  # The assistant's number
            else:
                actual_customer_phone = customer_phone_number
                actual_to_phone = to_phone_number
            
            # Create call record in database immediately
            # Store outbound call metadata in the call metadata
            call_metadata = {}
            if is_outbound:
                call_metadata = {
                    "outbound": True,
                    "agenda": outbound_agenda,
                    "custom_welcome_message": outbound_welcome_message
                }
            
            call = await CallService.create_call(
                assistant_id=assistant.id,
                call_sid=call_sid,
                to_phone_number=actual_to_phone or "",
                customer_phone_number=actual_customer_phone or "",
                metadata=call_metadata,
            )
            logger.info(f"Created call record in database for call {call_sid}")
            
            # Send initial webhook status update immediately
            if assistant.webhook_url:
                asyncio.create_task(
                    WebhookService.send_status_update_webhook(
                        assistant=assistant,
                        call=call,
                        status="in-progress",
                        messages=[]
                    )
                )
                logger.info(f"Sent immediate webhook status update for {'outbound' if is_outbound else 'incoming'} call {call_sid}")
        except Exception as e:
            logger.error(f"Error creating call record or sending webhook for {call_sid}: {e}", exc_info=True)

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

    # Create stream with custom parameters for phone numbers and outbound call metadata
    stream = connect.stream(url=stream_url)
    if to_phone_number:
        stream.parameter(name="To", value=to_phone_number)
    if customer_phone_number:
        stream.parameter(name="From", value=customer_phone_number)
    
    # Add outbound call metadata as stream parameters
    if is_outbound:
        stream.parameter(name="outbound", value="true")
        if outbound_assistant_id:
            stream.parameter(name="assistant_id", value=outbound_assistant_id)
        if outbound_welcome_message:
            stream.parameter(name="welcome_message", value=outbound_welcome_message)
        if outbound_agenda:
            stream.parameter(name="agenda", value=outbound_agenda)
        if outbound_to_phone:
            stream.parameter(name="to_phone_number", value=outbound_to_phone)

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

                    # Extract outbound call metadata from customParameters
                    is_outbound = custom_params.get("outbound") == "true"
                    outbound_assistant_id = custom_params.get("assistant_id")
                    outbound_welcome_message = custom_params.get("welcome_message")
                    outbound_agenda = custom_params.get("agenda")
                    outbound_to_phone = custom_params.get("to_phone_number")

                    if not stream_sid or not call_sid:
                        logger.error("Missing streamSid or callSid in start message")
                        continue

                    tracks = start_data.get("tracks", [])
                    media_format = start_data.get("mediaFormat", {})

                    # Start call handling
                    try:
                        # Get assistant based on the context
                        assistant = None
                        
                        if is_outbound and outbound_assistant_id:
                            # For outbound calls, use the assistant ID from metadata
                            try:
                                assistant = await assistant_manager.get_assistant_by_id(int(outbound_assistant_id))
                            except (ValueError, TypeError):
                                logger.error(f"Invalid assistant_id in outbound stream metadata: {outbound_assistant_id}")
                        elif to_number:
                            # For inbound calls, lookup assistant by phone number
                            assistant = await assistant_manager.get_assistant_by_phone(to_number)
                        
                        if not assistant:
                            error_msg = f"Assistant not found for {'outbound' if is_outbound else 'inbound'} call"
                            logger.error(error_msg)
                            raise HTTPException(status_code=404, detail=error_msg)

                        # Prepare call metadata including outbound call information
                        call_metadata = {
                            "stream_sid": stream_sid,
                            "media_format": media_format,
                            "tracks": tracks,
                        }
                        
                        # Add outbound call specific metadata
                        if is_outbound:
                            call_metadata.update({
                                "outbound": True,
                                "agenda": outbound_agenda,
                                "custom_welcome_message": outbound_welcome_message,
                                "outbound_to_phone": outbound_to_phone
                            })

                        # Start the call handler with WebSocket
                        await call_handler.start_call(
                            call_sid=call_sid,
                            websocket=websocket,
                            to_number=to_number,
                            from_number=from_number,
                            metadata=call_metadata,
                            assistant=assistant,
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
                            
                            # Log audio reception for debugging
                            logger.debug(f"Received audio for call {call_sid}: {len(decoded_audio)} bytes")

                            # Handle audio through call handler
                            result = await call_handler.handle_audio(call_sid, decoded_audio)
                            logger.debug(f"Audio processing result for call {call_sid}: {result}")

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


@router.post("/calls/initiate")
async def initiate_outbound_call(request: Request):
    """
    Initiate an outbound call through the API.
    
    Expected JSON body:
    {
        "assistant_id": 123,
        "to_phone_number": "+1234567890",
        "welcome_message": "Hello, this is your AI assistant calling...",
        "agenda": "I'm calling to discuss your recent order and confirm delivery details."
    }
    """
    try:
        # Parse JSON body
        body = await request.json()
        
        assistant_id = body.get("assistant_id")
        to_phone_number = body.get("to_phone_number")
        welcome_message = body.get("welcome_message")
        agenda = body.get("agenda")
        
        # Validate required fields
        if not assistant_id:
            raise HTTPException(status_code=400, detail="assistant_id is required")
        if not to_phone_number:
            raise HTTPException(status_code=400, detail="to_phone_number is required")
        if not agenda:
            raise HTTPException(status_code=400, detail="agenda is required")
        
        # Validate phone number format
        if not TwilioService.validate_phone_number(to_phone_number):
            raise HTTPException(status_code=400, detail="Invalid phone number format. Use E.164 format (e.g., +1234567890)")
        
        # Get the assistant
        assistant = await assistant_manager.get_assistant_by_id(assistant_id)
        if not assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")
        
        # Check billing limits for the organization
        try:
            usage_check = await BillingService.check_usage_limits(assistant.organization_id)
            
            if not usage_check.get("allowed", False):
                error_detail = "Usage limit exceeded"
                if usage_check.get("needs_upgrade", False):
                    error_detail = "Monthly usage limit exceeded. Please upgrade your plan or add top-up credits."
                
                raise HTTPException(status_code=429, detail=error_detail)
                
        except HTTPException:
            raise
        except Exception as billing_error:
            # Don't block calls if billing check fails
            logger.error(f"Billing check failed for organization {assistant.organization_id}, allowing call: {billing_error}")
        
        # Use the get_twiml_webhook_url function to determine the webhook URL
        webhook_url = get_twiml_webhook_url()
        
        # Prepare call metadata to pass through the webhook
        call_metadata = {
            "outbound": "true",
            "assistant_id": str(assistant_id),
            "welcome_message": welcome_message,
            "agenda": agenda,
            "to_phone_number": to_phone_number
        }
        
        # Get Twilio credentials from assistant or environment
        twilio_account_sid = assistant.twilio_account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        twilio_auth_token = assistant.twilio_auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        
        if not twilio_account_sid or not twilio_auth_token:
            raise HTTPException(status_code=500, detail="Twilio credentials not configured")
        
        # Use the assistant's phone number as the from number
        from_phone_number = assistant.phone_number
        
        # Initiate the outbound call through Twilio
        call_sid = TwilioService.initiate_outbound_call(
            to_phone_number=to_phone_number,
            from_phone_number=from_phone_number,
            webhook_url=webhook_url,
            call_metadata=call_metadata,
            account_sid=twilio_account_sid,
            auth_token=twilio_auth_token
        )
        
        if not call_sid:
            raise HTTPException(status_code=500, detail="Failed to initiate outbound call")
        
        logger.info(
            f"Initiated outbound call {call_sid} from assistant {assistant_id} "
            f"to {to_phone_number} with agenda: {agenda[:100]}..."
        )
        
        # Return success response
        return {
            "success": True,
            "call_sid": call_sid,
            "message": "Outbound call initiated successfully",
            "assistant_id": assistant_id,
            "to_phone_number": to_phone_number,
            "from_phone_number": from_phone_number
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating outbound call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
