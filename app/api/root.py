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
from app.core.sms_handler import SMSHandler
from app.services.conversation_service import ConversationService
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
CUSTOM_LLM_URL = os.getenv(
    "CUSTOM_LLM_URL", "http://localhost:8001/ai/chat/completions"
)

# Initialize call handler - configuration is now handled per-call through assistant objects
call_handler = CallHandler()
sms_handler = SMSHandler()
assistant_manager = AssistantManager()


# Lifecycle events for background tasks
@router.on_event("startup")
async def startup_event():
    """Start background tasks on application startup."""
    await sms_handler.start()
    logger.info("Started background tasks")


@router.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    await sms_handler.stop()
    logger.info("Stopped background tasks")


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
            assistant = await assistant_manager.get_assistant_by_id(
                int(outbound_assistant_id)
            )
        except (ValueError, TypeError):
            logger.error(
                f"Invalid assistant_id in outbound call metadata: {outbound_assistant_id}"
            )
    elif to_phone_number:
        # For inbound calls, lookup assistant by phone number
        assistant = await assistant_manager.get_assistant_by_phone(to_phone_number)

    if assistant:
        # Check billing limits for the organization - make this faster
        try:
            usage_check = await BillingService.check_usage_limits(
                assistant.organization_id
            )

            if not usage_check.get("allowed", False):
                # Create a TwiML response to reject the call or play a message
                response = VoiceResponse()

                if usage_check.get("needs_upgrade", False):
                    response.say(
                        "Your call cannot be completed at this time. "
                        "Your organization has exceeded its monthly usage limit. "
                        "Please contact your administrator to upgrade your plan or add top-up credits.",
                        voice="alice",
                    )
                else:
                    response.say(
                        "Your call cannot be completed at this time. "
                        "Please try again later.",
                        voice="alice",
                    )

                response.hangup()

                logger.warning(
                    f"Call rejected due to billing limits for organization {assistant.organization_id}: {usage_check}"
                )
                return Response(content=str(response), media_type="application/xml")
        except Exception as billing_error:
            # Don't block calls if billing check fails
            logger.error(
                f"Billing check failed for {assistant.organization_id}, allowing call: {billing_error}"
            )

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
                    "custom_welcome_message": outbound_welcome_message,
                }

            call = await ConversationService.create_conversation(
                assistant_id=assistant.id,
                channel_sid=call_sid,
                to_phone_number=actual_to_phone or "",
                customer_phone_number=actual_customer_phone or "",
                metadata=call_metadata,
                conversation_type="call",
            )
            logger.info(f"Created call record in database for call {call_sid}")

            # Send initial webhook status update immediately
            if assistant.webhook_url:
                asyncio.create_task(
                    WebhookService.send_status_update_webhook(
                        assistant=assistant,
                        call=call,
                        status="in-progress",
                        messages=[],
                    )
                )
                logger.info(
                    f"Sent immediate webhook status update for {'outbound' if is_outbound else 'incoming'} call {call_sid}"
                )
        except Exception as e:
            logger.error(
                f"Error creating call record or sending webhook for {call_sid}: {e}",
                exc_info=True,
            )

    # Create the TwiML response
    response = VoiceResponse()

    # Check for ngrok forwarding
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")

    # Determine protocol (ws:// or wss://)
    # When using ngrok, we should use wss:// for the forwarded connection
    forwarded_proto = request.headers.get("x-forwarded-proto")
    web_protocol = "http"
    if forwarded_proto == "https":
        protocol = "wss"
        web_protocol = "https"
    elif request.url.scheme == "https":
        protocol = "wss"
        web_protocol = "https"
    else:
        protocol = "ws"
        web_protocol = "http"

    # Build the recording status callback URL
    recording_status_callback_url = f"{web_protocol}://{host}/recording/status"

    # Start Twilio recording with callback
    response.record(
        recording_channels="dual",  # Record both sides of conversation
        recording_status_callback=recording_status_callback_url,
        recording_status_callback_event=["completed", "failed"],
        max_length=3600,  # Max 1 hour recording
        play_beep=False,  # Don't play beep when recording starts
    )

    # Create a <Connect> verb with the WebSocket stream
    connect = Connect()

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
                                assistant = await assistant_manager.get_assistant_by_id(
                                    int(outbound_assistant_id)
                                )
                            except (ValueError, TypeError):
                                logger.error(
                                    f"Invalid assistant_id in outbound stream metadata: {outbound_assistant_id}"
                                )
                        elif to_number:
                            # For inbound calls, lookup assistant by phone number
                            assistant = await assistant_manager.get_assistant_by_phone(
                                to_number
                            )

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
                            call_metadata.update(
                                {
                                    "outbound": True,
                                    "agenda": outbound_agenda,
                                    "custom_welcome_message": outbound_welcome_message,
                                    "outbound_to_phone": outbound_to_phone,
                                }
                            )

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
                            logger.debug(
                                f"Received audio for call {call_sid}: {len(decoded_audio)} bytes"
                            )

                            # Handle audio through call handler
                            result = await call_handler.handle_audio(
                                call_sid, decoded_audio
                            )
                            logger.debug(
                                f"Audio processing result for call {call_sid}: {result}"
                            )

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
            raise HTTPException(
                status_code=400,
                detail="Invalid phone number format. Use E.164 format (e.g., +1234567890)",
            )

        # Get the assistant
        assistant = await assistant_manager.get_assistant_by_id(assistant_id)
        if not assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")

        # Check billing limits for the organization
        try:
            usage_check = await BillingService.check_usage_limits(
                assistant.organization_id
            )

            if not usage_check.get("allowed", False):
                error_detail = "Usage limit exceeded"
                if usage_check.get("needs_upgrade", False):
                    error_detail = "Monthly usage limit exceeded. Please upgrade your plan or add top-up credits."

                raise HTTPException(status_code=429, detail=error_detail)

        except HTTPException:
            raise
        except Exception as billing_error:
            # Don't block calls if billing check fails
            logger.error(
                f"Billing check failed for organization {assistant.organization_id}, allowing call: {billing_error}"
            )

        # Use the get_twiml_webhook_url function to determine the webhook URL
        webhook_url = get_twiml_webhook_url()

        # Prepare call metadata to pass through the webhook
        call_metadata = {
            "outbound": "true",
            "assistant_id": str(assistant_id),
            "welcome_message": welcome_message,
            "agenda": agenda,
            "to_phone_number": to_phone_number,
        }

        # Get Twilio credentials from assistant or environment
        twilio_account_sid = assistant.twilio_account_sid or os.getenv(
            "TWILIO_ACCOUNT_SID"
        )
        twilio_auth_token = assistant.twilio_auth_token or os.getenv(
            "TWILIO_AUTH_TOKEN"
        )

        if not twilio_account_sid or not twilio_auth_token:
            raise HTTPException(
                status_code=500, detail="Twilio credentials not configured"
            )

        # Use the assistant's phone number as the from number
        from_phone_number = assistant.phone_number

        # Initiate the outbound call through Twilio
        call_sid = TwilioService.initiate_outbound_call(
            to_phone_number=to_phone_number,
            from_phone_number=from_phone_number,
            webhook_url=webhook_url,
            call_metadata=call_metadata,
            account_sid=twilio_account_sid,
            auth_token=twilio_auth_token,
        )

        if not call_sid:
            raise HTTPException(
                status_code=500, detail="Failed to initiate outbound call"
            )

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
            "from_phone_number": from_phone_number,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating outbound call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/sms/webhook")
async def handle_incoming_sms(request: Request):
    """
    Handle incoming SMS messages from Twilio.
    This endpoint is called by Twilio when an SMS is received.
    """
    try:
        # Parse form data from Twilio
        form_data = await request.form()

        # Extract SMS details
        message_sid = form_data.get("MessageSid")
        from_number = form_data.get("From")
        to_number = form_data.get("To")
        message_body = form_data.get("Body", "").strip()

        logger.info(
            f"Received SMS {message_sid} from {from_number} to {to_number}: {message_body[:100]}..."
        )

        # Validate required fields
        if not all([message_sid, from_number, to_number, message_body]):
            logger.error(
                f"Missing required SMS fields: message_sid={message_sid}, from={from_number}, to={to_number}"
            )
            return Response(content="Missing required fields", status_code=400)

        # Get assistant by phone number
        assistant = await assistant_manager.get_assistant_by_phone(to_number)
        if not assistant:
            logger.error(f"No assistant found for phone number {to_number}")
            return Response(
                content="No assistant configured for this number", status_code=404
            )

        # Check billing limits
        try:
            usage_check = await BillingService.check_usage_limits(
                assistant.organization_id
            )

            if not usage_check.get("allowed", False):
                logger.warning(
                    f"SMS rejected due to billing limits for organization {assistant.organization_id}: {usage_check}"
                )

                # Send error message back if limits exceeded
                error_message = "Your message cannot be processed due to usage limits. Please contact your administrator."
                if usage_check.get("needs_upgrade", False):
                    error_message = "Monthly usage limit exceeded. Please upgrade your plan or add credits to continue."

                # Send error SMS back to user
                TwilioService.send_sms(
                    to_phone_number=from_number,
                    from_phone_number=to_number,
                    message_body=error_message,
                    account_sid=assistant.twilio_account_sid,
                    auth_token=assistant.twilio_auth_token,
                )

                return Response(content="", status_code=200)  # Return 200 to Twilio

        except Exception as billing_error:
            logger.error(
                f"Billing check failed for {assistant.organization_id}, allowing SMS: {billing_error}"
            )

        # Extract additional metadata from Twilio
        metadata = {
            "account_sid": form_data.get("AccountSid"),
            "messaging_service_sid": form_data.get("MessagingServiceSid"),
            "num_media": form_data.get("NumMedia", "0"),
            "num_segments": form_data.get("NumSegments", "1"),
            "sms_status": form_data.get("SmsStatus"),
            "api_version": form_data.get("ApiVersion"),
        }

        # Process the SMS
        response_text = await sms_handler.process_incoming_sms(
            message_sid=message_sid,
            from_number=from_number,
            to_number=to_number,
            message_body=message_body,
            assistant=assistant,
            metadata=metadata,
        )

        # Send response if we have one
        if response_text:
            # Get Twilio credentials
            twilio_account_sid = assistant.twilio_account_sid or os.getenv(
                "TWILIO_ACCOUNT_SID"
            )
            twilio_auth_token = assistant.twilio_auth_token or os.getenv(
                "TWILIO_AUTH_TOKEN"
            )

            if twilio_account_sid and twilio_auth_token:
                response_sid = TwilioService.send_sms(
                    to_phone_number=from_number,
                    from_phone_number=to_number,
                    message_body=response_text,
                    account_sid=twilio_account_sid,
                    auth_token=twilio_auth_token,
                )

                if response_sid:
                    logger.info(f"Sent SMS response {response_sid} to {from_number}")
                else:
                    logger.error(f"Failed to send SMS response to {from_number}")
            else:
                logger.error("Missing Twilio credentials for sending SMS response")

        # Return empty 200 response to Twilio
        return Response(content="", status_code=200)

    except Exception as e:
        logger.error(f"Error handling incoming SMS: {e}", exc_info=True)
        # Return 200 to Twilio to avoid retries on our errors
        return Response(content="", status_code=200)


@router.post("/sms/send")
async def send_sms(request: Request):
    """
    Send an SMS message through the API.

    Expected JSON body:
    {
        "assistant_id": 123,
        "to_phone_number": "+1234567890",
        "message_body": "Hello, this is your AI assistant...",
        "agenda": "Optional: Follow up about their recent order and confirm delivery address"
    }
    """
    try:
        # Parse JSON body
        body = await request.json()

        assistant_id = body.get("assistant_id")
        to_phone_number = body.get("to_phone_number")
        message_body = body.get("message_body")
        agenda = body.get("agenda")  # Optional agenda for conversation context

        # Validate required fields
        if not assistant_id:
            raise HTTPException(status_code=400, detail="assistant_id is required")
        if not to_phone_number:
            raise HTTPException(status_code=400, detail="to_phone_number is required")
        if not message_body:
            raise HTTPException(status_code=400, detail="message_body is required")

        # Validate phone number format
        if not TwilioService.validate_phone_number(to_phone_number):
            raise HTTPException(
                status_code=400,
                detail="Invalid phone number format. Use E.164 format (e.g., +1234567890)",
            )

        # Get the assistant
        assistant = await assistant_manager.get_assistant_by_id(assistant_id)
        if not assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")

        # Check billing limits
        try:
            usage_check = await BillingService.check_usage_limits(
                assistant.organization_id
            )

            if not usage_check.get("allowed", False):
                error_detail = "Usage limit exceeded"
                if usage_check.get("needs_upgrade", False):
                    error_detail = "Monthly usage limit exceeded. Please upgrade your plan or add credits."

                raise HTTPException(status_code=429, detail=error_detail)

        except HTTPException:
            raise
        except Exception as billing_error:
            logger.error(
                f"Billing check failed for organization {assistant.organization_id}, allowing SMS: {billing_error}"
            )

        # Get Twilio credentials
        twilio_account_sid = assistant.twilio_account_sid or os.getenv(
            "TWILIO_ACCOUNT_SID"
        )
        twilio_auth_token = assistant.twilio_auth_token or os.getenv(
            "TWILIO_AUTH_TOKEN"
        )

        if not twilio_account_sid or not twilio_auth_token:
            raise HTTPException(
                status_code=500, detail="Twilio credentials not configured"
            )

        # Send the SMS
        message_sid = TwilioService.send_sms(
            to_phone_number=to_phone_number,
            from_phone_number=assistant.phone_number,
            message_body=message_body,
            account_sid=twilio_account_sid,
            auth_token=twilio_auth_token,
        )

        if not message_sid:
            raise HTTPException(status_code=500, detail="Failed to send SMS")

        logger.info(
            f"Sent SMS {message_sid} from assistant {assistant_id} to {to_phone_number}"
        )

        # Create conversation record for outbound SMS
        try:
            # Include agenda in metadata if provided
            conversation_metadata = {"outbound": True}
            if agenda:
                conversation_metadata["agenda"] = agenda
                logger.info(f"Outbound SMS with agenda: {agenda[:100]}...")

            conversation = await ConversationService.create_conversation(
                assistant_id=assistant.id,
                channel_sid=message_sid,
                conversation_type="sms",
                to_phone_number=to_phone_number,
                customer_phone_number=to_phone_number,  # For outbound, customer is the recipient
                metadata=conversation_metadata,
            )

            if conversation:
                # Store the message as a chat message
                await ConversationService.create_chat_message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=message_body,
                )

                # If there's an agenda, also store it as a system message
                if agenda:
                    await ConversationService.create_chat_message(
                        conversation_id=conversation.id,
                        role="system",
                        content=f"SMS CONVERSATION AGENDA: {agenda}",
                    )

                # Record SMS usage for billing
                await BillingService.record_sms_usage(conversation.id)

                # Send webhook if configured
                if assistant.webhook_url:
                    await WebhookService.send_sms_webhook(
                        assistant=assistant,
                        conversation_id=conversation.id,
                        webhook_type="sms-sent",
                        from_number=assistant.phone_number,
                        to_number=to_phone_number,
                        message_body=message_body,
                        direction="outbound",
                        metadata={"agenda": agenda} if agenda else None,
                    )

        except Exception as e:
            logger.error(f"Error creating conversation record for outbound SMS: {e}")

        # Return success response
        response_data = {
            "success": True,
            "message_sid": message_sid,
            "message": "SMS sent successfully",
            "assistant_id": assistant_id,
            "to_phone_number": to_phone_number,
            "from_phone_number": assistant.phone_number,
        }

        if agenda:
            response_data["agenda"] = agenda

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending SMS: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/recording/status")
async def handle_recording_status(request: Request):
    """
    Handle Twilio recording status callbacks.
    This endpoint is called by Twilio when a recording is completed or failed.
    """
    try:
        # Parse form data from Twilio
        form_data = await request.form()

        # Extract recording details
        recording_sid = form_data.get("RecordingSid")
        call_sid = form_data.get("CallSid")
        recording_status = form_data.get("RecordingStatus")
        recording_duration = form_data.get("RecordingDuration")
        recording_channels = form_data.get("RecordingChannels")

        logger.info(
            f"Received recording status callback: recording_sid={recording_sid}, "
            f"call_sid={call_sid}, status={recording_status}, duration={recording_duration}"
        )

        # Validate required fields
        if not all([recording_sid, call_sid, recording_status]):
            logger.error(
                f"Missing required recording fields: recording_sid={recording_sid}, call_sid={call_sid}, status={recording_status}"
            )
            return Response(content="Missing required fields", status_code=400)

        # Only process completed recordings
        if recording_status != "completed":
            logger.info(
                f"Recording {recording_sid} status is '{recording_status}', not processing"
            )
            return Response(content="", status_code=200)

        # Get the call/conversation from database
        call = await ConversationService.get_conversation_by_sid(call_sid)
        if not call:
            logger.error(f"No call found for call_sid {call_sid}")
            return Response(content="Call not found", status_code=404)

        # Get the assistant
        assistant = call.assistant
        if not assistant:
            logger.error(f"No assistant found for call {call_sid}")
            return Response(content="Assistant not found", status_code=404)

        # Download the recording from Twilio
        twilio_account_sid = assistant.twilio_account_sid or os.getenv(
            "TWILIO_ACCOUNT_SID"
        )
        twilio_auth_token = assistant.twilio_auth_token or os.getenv(
            "TWILIO_AUTH_TOKEN"
        )

        if not twilio_account_sid or not twilio_auth_token:
            logger.error(
                f"Missing Twilio credentials for downloading recording {recording_sid}"
            )
            return Response(
                content="Twilio credentials not configured", status_code=500
            )

        # Download recording content from Twilio
        download_result = TwilioService.download_recording_content(
            recording_sid=recording_sid,
            account_sid=twilio_account_sid,
            auth_token=twilio_auth_token,
        )

        if not download_result:
            logger.error(f"Failed to download recording {recording_sid} from Twilio")
            return Response(content="Failed to download recording", status_code=500)

        filename, recording_content = download_result

        # Upload recording to S3
        try:
            from app.services.s3_service import S3Service

            s3_service = S3Service.create_default_instance()

            if s3_service:
                # Prepare metadata
                metadata = {
                    "duration": str(recording_duration) if recording_duration else "0",
                    "file_size": str(len(recording_content)),
                    "channels": str(recording_channels) if recording_channels else "1",
                    "source": "twilio",
                    "recording_sid": recording_sid,
                }

                # Upload to S3 with "twilio" as recording type
                s3_key, s3_url = await s3_service.upload_audio_file(
                    audio_data=recording_content,
                    call_sid=call_sid,
                    recording_type="twilio",
                    format="mp3",
                    metadata=metadata,
                )

                if s3_key and s3_url:
                    logger.info(
                        f"Successfully uploaded Twilio recording {recording_sid} to S3: {s3_key}"
                    )

                    # Check if there's already a recording record for this call
                    existing_recordings = await ConversationService.get_call_recordings(
                        call_sid
                    )
                    
                    # Look for existing "mixed" recording to update with Twilio data
                    # Twilio's dual-channel recording is equivalent to our "mixed" recording
                    mixed_recording = None
                    for recording in existing_recordings:
                        if recording.recording_type == "mixed":
                            mixed_recording = recording
                            break
                    
                    if mixed_recording:
                        # Update the existing "mixed" recording with superior Twilio data
                        # This gives us the best of both worlds: real-time S3 recording during the call,
                        # then high-quality Twilio recording when completed
                        updated_recording = await ConversationService.update_recording(
                            recording_id=mixed_recording.id,
                            recording_sid=recording_sid,
                            s3_key=s3_key,
                            s3_url=s3_url,
                            duration=(
                                float(recording_duration)
                                if recording_duration
                                else None
                            ),
                            file_size=len(recording_content),
                            status="completed",
                            channels=(
                                int(recording_channels) if recording_channels else 2
                            ),  # Twilio dual-channel
                            recording_metadata={
                                **metadata,
                                "source": "twilio_override",  # Mark that this was updated with Twilio data
                                "original_source": "s3"  # Remember it started as S3
                            },
                        )
                        if updated_recording:
                            logger.info(
                                f"Updated existing 'mixed' recording {mixed_recording.id} with superior Twilio recording data"
                            )
                        else:
                            logger.error(
                                f"Failed to update mixed recording {mixed_recording.id} with Twilio data"
                            )
                    else:
                        # No existing mixed recording found, create new "twilio" recording
                        # This handles cases where S3 recording failed or is disabled
                        result = await ConversationService.create_recording(
                            channel_sid=call_sid,
                            recording_sid=recording_sid,
                            s3_key=s3_key,
                            s3_url=s3_url,
                            duration=(
                                float(recording_duration)
                                if recording_duration
                                else None
                            ),
                            file_size=len(recording_content),
                            format="mp3",
                            sample_rate=8000,  # Twilio default
                            channels=(
                                int(recording_channels) if recording_channels else 2
                            ),
                            recording_type="mixed",  # Use "mixed" instead of "twilio"
                            recording_source="s3",
                            status="completed",
                        )
                        if result:
                            logger.info(
                                f"Created new 'mixed' recording record for Twilio recording {recording_sid} (S3 recording was not available)"
                            )
                        else:
                            logger.error(
                                f"Failed to create recording record for Twilio recording {recording_sid}"
                            )

                    # Log summary of other S3 recordings for reference
                    other_recordings = [
                        r
                        for r in existing_recordings
                        if r.recording_source == "s3" and r.recording_type != "mixed"
                    ]
                    if other_recordings:
                        logger.info(
                            f"Call {call_sid} also has {len(other_recordings)} other S3 recordings: {[r.recording_type for r in other_recordings]}"
                        )

                else:
                    logger.error(f"Failed to upload recording {recording_sid} to S3")
            else:
                logger.error("S3 service not available for storing Twilio recording")

        except Exception as s3_error:
            logger.error(f"Error uploading Twilio recording to S3: {s3_error}")

        # Return success response to Twilio
        return Response(content="", status_code=200)

    except Exception as e:
        logger.error(f"Error handling recording status callback: {e}", exc_info=True)
        # Return 200 to Twilio to avoid retries on our errors
        return Response(content="", status_code=200)
