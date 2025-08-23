"""
This file contains the API endpoints for the Twilio call.
"""

# pylint: disable=logging-format-interpolation,logging-fstring-interpolation,broad-exception-caught
import logging
import json
import base64
import os
import asyncio
from typing import Optional, Dict, Set, Any
from datetime import datetime
from sqlalchemy import select
from fastapi import Request, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import Response
from fastapi import APIRouter
from twilio.twiml.voice_response import VoiceResponse, Connect

from app.core.call_handler import CallHandler
from app.core.assistant_manager import AssistantManager
from app.core.auth import get_current_user_flexible
from app.services.call_service import CallService
from app.services.webhook_service import WebhookService
from app.services.sms_webhook_service import SMSWebhookService
from app.db.database import get_async_db_session
from app.db.models import User

from app.core.telephony_provider import UnifiedTelephonyService
from app.telnyx.telnyx_webhook_handler import TelnyxWebhookHandler, validate_telnyx_webhook
from app.utils.url_utils import get_twiml_webhook_url
from app.utils.webhook_security import (
    require_twilio_webhook_auth, 
    get_webhook_security_headers
)

# Global mapping to store stream_id -> call_control_id relationships for Telnyx
telnyx_stream_mappings = {}

# Simple webhook deduplication cache (call_control_id -> last_processed_time)
telnyx_webhook_cache = {}
from app.api.schemas import (
    InitiateCallRequest, InitiateCallResponse, SendSMSRequest, SendSMSResponse,
)


router = APIRouter()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

call_handler = CallHandler()
assistant_manager = AssistantManager()

# Global transcript broadcaster
class TranscriptBroadcaster:
    """
    Manages WebSocket connections for live transcript streaming.
    """
    
    def __init__(self):
        # Dict mapping call_sid to set of websocket connections
        self.connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, call_sid: str, websocket: WebSocket):
        """Add a WebSocket connection for a specific call."""
        if call_sid not in self.connections:
            self.connections[call_sid] = set()
        self.connections[call_sid].add(websocket)
        logger.info(f"Client connected to transcript stream for call {call_sid}. Total connections: {len(self.connections[call_sid])}")
    
    async def disconnect(self, call_sid: str, websocket: WebSocket):
        """Remove a WebSocket connection for a specific call."""
        if call_sid in self.connections:
            self.connections[call_sid].discard(websocket)
            if not self.connections[call_sid]:
                # Remove empty set
                del self.connections[call_sid]
            logger.info(f"Client disconnected from transcript stream for call {call_sid}")
    
    async def broadcast_transcript(self, call_sid: str, transcript_data: dict):
        """Broadcast transcript data to all connected clients for a specific call."""
        logger.info(f"Attempting to broadcast transcript for call {call_sid}, speaker: {transcript_data.get('speaker')}")
        
        if call_sid not in self.connections:
            logger.info(f"No WebSocket connections found for call {call_sid}")
            return
        
        # Create a copy of connections to avoid modification during iteration
        connections = self.connections[call_sid].copy()
        if not connections:
            logger.info(f"No active connections for call {call_sid}")
            return
        
        logger.info(f"Broadcasting transcript to {len(connections)} connected clients for call {call_sid}")
        
        message = {
            "type": "transcript",
            "call_sid": call_sid,
            "timestamp": datetime.utcnow().isoformat(),
            "data": transcript_data
        }
        
        # Send to all connected clients
        disconnected = set()
        for websocket in connections:
            try:
                await websocket.send_json(message)
                logger.debug(f"Successfully sent transcript to client for call {call_sid}")
            except Exception as e:
                logger.warning(f"Failed to send transcript to client for call {call_sid}: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected:
            await self.disconnect(call_sid, websocket)
        
        logger.info(f"Successfully broadcasted transcript for call {call_sid} to {len(connections) - len(disconnected)} clients")
    
    async def broadcast_call_status(self, call_sid: str, status: str, metadata: Optional[dict] = None):
        """Broadcast call status updates to connected clients."""
        if call_sid not in self.connections:
            return
        
        # Create a copy of connections to avoid modification during iteration
        connections = self.connections[call_sid].copy()
        if not connections:
            return
        
        message = {
            "type": "call_status",
            "call_sid": call_sid,
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "metadata": metadata or {}
        }
        
        # Send to all connected clients
        disconnected = set()
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send call status to client for call {call_sid}: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected:
            await self.disconnect(call_sid, websocket)

# Global instance
transcript_broadcaster = TranscriptBroadcaster()

async def authenticate_websocket_connection(websocket: WebSocket, token: str = None):
    """
    Authenticate WebSocket connection using various methods.
    
    Supports authentication via:
    1. Query parameter 'token' with API key
    2. Authorization header with Bearer token  
    3. Subprotocol with token
    
    Returns:
        User object if authenticated, None otherwise
    """
    from app.services.auth_service import APIKeyService
    
    api_key = None
    
    # Method 1: Query parameter
    if token:
        api_key = token
        logger.debug("Using token from query parameter")
    
    # Method 2: Authorization header
    if not api_key:
        auth_header = None
        for name, value in websocket.headers:
            if name.lower() == b'authorization':
                auth_header = value.decode()
                break
        
        if auth_header and auth_header.startswith('Bearer '):
            api_key = auth_header[7:]  # Remove 'Bearer ' prefix
            logger.debug("Using token from Authorization header")
    
    # Method 3: Subprotocol (token passed as subprotocol)
    if not api_key:
        try:
            protocol_header = websocket.headers.get('sec-websocket-protocol')
            if protocol_header:
                protocols = protocol_header.decode().split(',')
                for protocol in protocols:
                    protocol = protocol.strip()
                    if protocol.startswith('burki-token-'):
                        api_key = protocol[12:]  # Remove 'burki-token-' prefix
                        logger.debug("Using token from WebSocket subprotocol")
                        break
        except Exception as e:
            logger.debug(f"Error reading WebSocket subprotocol: {e}")
    
    if not api_key:
        logger.warning("No authentication token provided for WebSocket connection")
        return None
    
    # Verify the API key
    try:
        result = await APIKeyService.verify_api_key(api_key)
        if result:
            api_key_obj, user = result
            logger.info(f"WebSocket authenticated for user: {user.email}")
            return user
        else:
            logger.warning("Invalid API key provided for WebSocket connection")
            return None
    except Exception as e:
        logger.error(f"Error verifying API key for WebSocket: {e}")
        return None

async def verify_call_access(user, call_sid: str) -> bool:
    """
    Verify that the authenticated user has access to the specified call.
    
    Args:
        user: Authenticated user object
        call_sid: Call SID to check access for
        
    Returns:
        bool: True if user has access, False otherwise
    """
    try:
        # Get the call and verify it belongs to the user's organization
        call = await CallService.get_call_by_sid(call_sid)
        if not call:
            logger.warning(f"Call {call_sid} not found")
            return False
        
        # Check if the call's assistant belongs to the user's organization
        if call.assistant and call.assistant.organization_id == user.organization_id:
            logger.debug(f"User {user.email} has access to call {call_sid}")
            return True
        else:
            logger.warning(f"User {user.email} does not have access to call {call_sid}")
            return False
            
    except Exception as e:
        logger.error(f"Error verifying call access: {e}")
        return False

@router.websocket("/live-transcript/{call_sid}")
async def live_transcript_endpoint(websocket: WebSocket, call_sid: str, token: str = None):
    """
    WebSocket endpoint for streaming live transcripts of a specific call.
    
    Clients can connect to this endpoint to receive real-time transcripts
    as they are generated during the call.
    
    Message format:
    {
        "type": "transcript",
        "call_sid": "CA123...",
        "timestamp": "2024-01-01T12:00:00.000Z",
        "data": {
            "content": "Hello, how can I help you?",
            "speaker": "assistant",
            "is_final": true,
            "confidence": 0.95,
            "segment_start": 5.2,
            "segment_end": 7.1
        }
    }
    
    Or for call status updates:
    {
        "type": "call_status", 
        "call_sid": "CA123...",
        "timestamp": "2024-01-01T12:00:00.000Z",
        "status": "in-progress|completed|failed",
        "metadata": {}
    }
    """
    logger.info(f"Live transcript WebSocket connection attempt for call {call_sid}")
    
    try:
        # Accept the WebSocket connection first
        await websocket.accept()
        logger.info(f"Live transcript WebSocket connection accepted for call {call_sid}")
        
        # Then authenticate the connection
        authenticated_user = await authenticate_websocket_connection(websocket, token)
        if not authenticated_user:
            await websocket.send_json({
                "type": "error",
                "call_sid": call_sid,
                "timestamp": datetime.utcnow().isoformat(),
                "error": "authentication_failed",
                "message": "Authentication failed"
            })
            await websocket.close(code=1008, reason="Authentication failed")
            return
        
        logger.info(f"User {authenticated_user.email} authenticated for call {call_sid}")
        
        # Verify user has access to this call
        if not await verify_call_access(authenticated_user, call_sid):
            await websocket.send_json({
                "type": "error",
                "call_sid": call_sid,
                "timestamp": datetime.utcnow().isoformat(),
                "error": "access_denied",
                "message": "Access denied to this call"
            })
            await websocket.close(code=1008, reason="Access denied to this call")
            return
        
        # Add this connection to the broadcaster
        await transcript_broadcaster.connect(call_sid, websocket)
        
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "call_sid": call_sid,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Connected to live transcript stream"
        })
        
        # Check if call exists and send current status
        try:
            call = await CallService.get_call_by_sid(call_sid)
            if call:
                await websocket.send_json({
                    "type": "call_status",
                    "call_sid": call_sid,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": call.status,
                    "metadata": {
                        "assistant_id": call.assistant_id,
                        "started_at": call.started_at.isoformat() if call.started_at else None,
                        "to_phone_number": call.to_phone_number,
                        "customer_phone_number": call.customer_phone_number
                    }
                })
                
                # Send any existing transcripts for this call
                try:
                    transcripts = await CallService.get_call_transcripts(call_sid)
                    for transcript in transcripts:
                        transcript_data = {
                            "content": transcript.content,
                            "speaker": transcript.speaker,
                            "is_final": transcript.is_final,
                            "confidence": transcript.confidence,
                            "segment_start": transcript.segment_start,
                            "segment_end": transcript.segment_end,
                            "created_at": transcript.created_at.isoformat() if transcript.created_at else None
                        }
                        await websocket.send_json({
                            "type": "transcript",
                            "call_sid": call_sid,
                            "timestamp": transcript.created_at.isoformat() if transcript.created_at else datetime.utcnow().isoformat(),
                            "data": transcript_data
                        })
                except Exception as transcript_error:
                    logger.error(f"Error sending existing transcripts for call {call_sid}: {transcript_error}")
                    
            else:
                await websocket.send_json({
                    "type": "error",
                    "call_sid": call_sid,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": f"Call {call_sid} not found"
                })
        except Exception as e:
            logger.error(f"Error fetching call info for live transcript stream {call_sid}: {e}")
            await websocket.send_json({
                "type": "error",
                "call_sid": call_sid,
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Error fetching call information"
            })
        
        # Keep connection alive and handle incoming messages (if any)
        try:
            while True:
                # Wait for incoming messages (clients might send heartbeats or requests)
                message = await websocket.receive_json()
                
                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong", 
                        "timestamp": datetime.utcnow().isoformat()
                    })
                elif message.get("type") == "request_status":
                    # Client requesting current call status
                    call = await CallService.get_call_by_sid(call_sid)
                    if call:
                        await websocket.send_json({
                            "type": "call_status",
                            "call_sid": call_sid,
                            "timestamp": datetime.utcnow().isoformat(),
                            "status": call.status,
                            "metadata": {
                                "assistant_id": call.assistant_id,
                                "started_at": call.started_at.isoformat() if call.started_at else None,
                                "duration": call.duration
                            }
                        })
                        
        except WebSocketDisconnect:
            logger.info(f"Live transcript WebSocket disconnected for call {call_sid}")
        except Exception as e:
            logger.error(f"Error in live transcript WebSocket for call {call_sid}: {e}", exc_info=True)
            try:
                await websocket.send_json({
                    "type": "error",
                    "call_sid": call_sid,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "Internal server error"
                })
            except:
                pass  # Connection might be closed already
        
    except Exception as e:
        logger.error(f"Error accepting live transcript WebSocket for call {call_sid}: {e}", exc_info=True)
        try:
            if websocket.client_state.name != "DISCONNECTED":
                await websocket.send_json({
                    "type": "error",
                    "call_sid": call_sid,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "Connection initialization failed"
                })
                await websocket.close(code=1011, reason="Internal server error")
        except:
            pass  # Connection might be closed already
    finally:
        # Clean up connection
        try:
            await transcript_broadcaster.disconnect(call_sid, websocket)
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up WebSocket connection for call {call_sid}: {cleanup_error}")


@router.post("/twiml")
async def get_twiml(request: Request):
    """
    Generate TwiML for incoming Twilio calls.
    This endpoint is called by Twilio when a call comes in.
    """
    # Validate Twilio webhook signature for security
    body = await request.body()
    require_twilio_webhook_auth(request, body)
    
    # Parse form data (recreate request since body was consumed)
    if body:
        form_data = {}
        body_str = body.decode('utf-8')
        for param in body_str.split('&'):
            if '=' in param:
                key, value = param.split('=', 1)
                import urllib.parse
                key = urllib.parse.unquote_plus(key)
                value = urllib.parse.unquote_plus(value)
                form_data[key] = value
    else:
        form_data = {}
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

    assistant = None
    phone_number_obj = None
    
    if is_outbound and outbound_assistant_id:
        # For outbound calls, use the assistant ID from metadata
        try:
            assistant = await assistant_manager.get_assistant_by_id(int(outbound_assistant_id))
        except (ValueError, TypeError):
            logger.error(f"Invalid assistant_id in outbound call metadata: {outbound_assistant_id}")
    elif to_phone_number:
        # For inbound calls, lookup assistant by phone number
        assistant = await assistant_manager.get_assistant_by_phone(to_phone_number)
        
        # Also get the phone number object to check for Google Voice forwarding
        try:
            from app.services.phone_number_service import PhoneNumberService
            phone_numbers = await PhoneNumberService.get_organization_phone_numbers(
                assistant.organization_id if assistant else None
            )
            for pn in phone_numbers:
                if pn.phone_number == to_phone_number:
                    phone_number_obj = pn
                    break
        except Exception as e:
            logger.error(f"Error fetching phone number object: {e}")

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

    # Check if Google Voice forwarding is enabled for this phone number
    should_send_dtmf = False
    if phone_number_obj and phone_number_obj.phone_metadata:
        google_voice_forwarding = phone_number_obj.phone_metadata.get("is_google_voice_forwarding", False)
        if google_voice_forwarding:
            should_send_dtmf = True
            logger.info(f"Google Voice forwarding enabled for {to_phone_number}, will send DTMF '1' after connection")

    # If Google Voice forwarding is enabled, send DTMF "1" after a pause
    if should_send_dtmf:
        # Add a pause and then send DTMF "1"
        # Using 'wwww' for 2 second pause to ensure the call is fully connected
        response.play(digits="wwww1")
        logger.info(f"Added DTMF '1' to TwiML response for Google Voice forwarding")

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

    # Add security headers to response
    security_headers = get_webhook_security_headers()
    return Response(
        content=twiml_response, 
        media_type="application/xml",
        headers=security_headers
    )


async def process_telnyx_recording_saved(
    call_control_id: str,
    recording_id: str,
    call_data: Dict[str, Any]
) -> None:
    """
    Process Telnyx recording_saved webhook by downloading the recording
    from Telnyx and uploading it to S3, similar to Twilio recording processing.
    """
    try:
        # Get the call from database using call_control_id as call_sid
        call = await CallService.get_call_by_sid(call_control_id)
        if not call:
            logger.error(f"No call found for Telnyx call_control_id {call_control_id}")
            return

        # Get the assistant
        assistant = call.assistant
        if not assistant:
            logger.error(f"No assistant found for call {call_control_id}")
            return

        # Get recording information from Telnyx
        from app.telnyx.telnyx_service import TelnyxService  # pylint: disable=import-outside-toplevel
        recording_info = TelnyxService.get_recording_info(
            recording_id=recording_id,
            api_key=os.getenv("TELNYX_API_KEY")
        )
        
        if not recording_info:
            logger.error(f"Could not fetch recording info for Telnyx recording {recording_id}")
            return

        # Extract recording details from Telnyx response
        download_url = recording_info.get("uri")  # Telnyx provides download URL
        duration = recording_info.get("duration", 0)
        channels_raw = recording_info.get("channels", 2)
        
        # Convert channels to integer if it's a string
        if isinstance(channels_raw, str):
            # Handle Telnyx channel formats: "dual" -> 2, "single" -> 1
            if channels_raw.lower() == "dual":
                channels = 2
            elif channels_raw.lower() == "single":
                channels = 1
            else:
                # Try to convert to int, fallback to 2
                try:
                    channels = int(channels_raw)
                except (ValueError, TypeError):
                    channels = 2
        else:
            channels = int(channels_raw) if channels_raw else 2
        
        if not download_url:
            logger.error(f"No download URL provided for Telnyx recording {recording_id}")
            return

        # Download the recording content from Telnyx
        recording_content = None
        filename = None
        
        try:
            recording_content_result = TelnyxService.download_recording_content(
                recording_id=recording_id,
                api_key=os.getenv("TELNYX_API_KEY")
            )
            
            if recording_content_result:
                filename, recording_content = recording_content_result
            else:
                # Fallback: download directly from URL if TelnyxService doesn't work
                import requests
                response = requests.get(download_url, timeout=30)
                response.raise_for_status()
                recording_content = response.content
                filename = f"telnyx_recording_{recording_id}.mp3"
                
        except Exception as e:
            logger.error(f"Error downloading Telnyx recording {recording_id}: {e}")
            return

        if not recording_content:
            logger.error(f"No recording content downloaded for Telnyx recording {recording_id}")
            return

        # Upload to S3
        try:
            from app.services.s3_service import S3Service  # pylint: disable=import-outside-toplevel
            s3_service = S3Service.create_default_instance()
            
            # Upload to S3 using the correct method signature
            s3_key, s3_url = await s3_service.upload_audio_file(
                audio_data=recording_content,
                call_sid=call.call_sid,
                recording_type="mixed",  # Telnyx provides mixed recordings
                format="mp3",
                metadata={
                    "provider": "telnyx",
                    "recording_id": recording_id,
                    "download_url": download_url
                }
            )
            
            if not s3_url:
                logger.error(f"Failed to upload Telnyx recording {recording_id} to S3")
                return
            
            logger.info(f"Successfully uploaded Telnyx recording {recording_id} to S3: {s3_key}")
            
        except Exception as e:
            logger.error(f"Error uploading Telnyx recording to S3: {e}")
            return

        # Update the existing recording record in database
        # Look for existing recording with "processing" status for this call
        async with await get_async_db_session() as db:
            from app.db.models import Recording  # pylint: disable=import-outside-toplevel
            
            # Find the processing recording for this call
            query = select(Recording).where(
                Recording.call_id == call.id,
                Recording.status == "processing"
            )
            result = await db.execute(query)
            recording = result.scalar_one_or_none()
            
            if recording:
                # Update the existing recording with Telnyx data
                recording.recording_sid = recording_id  # Store Telnyx recording ID
                recording.s3_key = s3_key
                recording.s3_url = s3_url
                recording.duration = duration
                recording.file_size = len(recording_content)
                recording.status = "completed"
                recording.channels = channels
                recording.recording_metadata = {
                    "provider": "telnyx",
                    "recording_id": recording_id,
                    "download_url": download_url,
                    **call_data  # Include all webhook data
                }
                recording.uploaded_at = datetime.utcnow()
                
                db.add(recording)
                await db.commit()
                await db.refresh(recording)
                
                logger.info(
                    f"Updated recording {recording.id} with Telnyx recording data for call {call_control_id}"
                )
            else:
                # Create new recording if none exists (fallback)
                logger.warning(f"No processing recording found for call {call_control_id}, creating new one")
                
                new_recording = await CallService.create_s3_recording(
                    call_sid=call_control_id,
                    s3_key=s3_key,
                    s3_url=s3_url,
                    duration=duration,
                    file_size=len(recording_content),
                    format="mp3",
                    sample_rate=22050,  # Default for Telnyx
                    channels=channels,
                    recording_type="mixed",
                    metadata={
                        "provider": "telnyx",
                        "recording_id": recording_id,
                        "download_url": download_url,
                        **call_data
                    }
                )
                
                if new_recording:
                    logger.info(f"Created new S3 recording {new_recording.id} for Telnyx call {call_control_id}")

    except Exception as e:
        logger.error(f"Error processing Telnyx recording_saved webhook: {e}", exc_info=True)


@router.post("/telnyx-webhook")
async def telnyx_webhook(request: Request):
    """
    Handle Telnyx Call Control webhooks.
    This endpoint is called by Telnyx for various call events.
    """
    try:
        # Get raw body for signature verification
        body = await request.body()
        
        # Validate webhook signature with body
        if not validate_telnyx_webhook(request, body):
            logger.warning("Invalid Telnyx webhook signature")
            return Response(content="Invalid signature", status_code=401)
        
        # Parse JSON data from Telnyx
        webhook_data = json.loads(body.decode('utf-8'))
        
        # Process the webhook
        call_data = TelnyxWebhookHandler.process_webhook(webhook_data)
        
        if not call_data:
            logger.debug("Webhook processed but no action required")
            return Response(
            content="", 
            status_code=200,
            headers=get_webhook_security_headers()
        )
        
        event_type = call_data.get('event_type')
        call_control_id = call_data.get('call_control_id')
        
        logger.info(f"Processing Telnyx webhook: {event_type} for call {call_control_id}")
        
        # Validate required fields
        if not call_control_id:
            logger.error(f"Missing call_control_id in Telnyx webhook data: {call_data}")
            return Response(content="Missing call_control_id", status_code=400)
        
        if event_type == 'call_initiated':
            # Simple deduplication for call_initiated events
            import time
            current_time = time.time()
            last_processed = telnyx_webhook_cache.get(call_control_id)
            
            if last_processed and (current_time - last_processed) < 30:  # 30 seconds dedup window
                logger.info(f"Ignoring duplicate call_initiated webhook for {call_control_id} (processed {current_time - last_processed:.1f}s ago)")
                return Response(
            content="", 
            status_code=200,
            headers=get_webhook_security_headers()
        )
            
            telnyx_webhook_cache[call_control_id] = current_time
            
            # Clean up old entries from cache (older than 5 minutes)
            cutoff_time = current_time - 300
            expired_keys = [k for k, v in telnyx_webhook_cache.items() if v <= cutoff_time]
            for key in expired_keys:
                del telnyx_webhook_cache[key]
            
            # Handle incoming call - similar to Twilio TwiML
            to_phone = call_data.get('to')
            from_phone = call_data.get('from')
            
            # Get assistant by phone number
            assistant = await assistant_manager.get_assistant_by_phone(to_phone)
            if not assistant:
                logger.error(f"No assistant found for Telnyx call to {to_phone}")
                # Hang up the call
                TelnyxWebhookHandler.send_call_control_command(
                    call_control_id=call_control_id,
                    command="hangup",
                    api_key=None  # Will use environment variable
                )
                return Response(
            content="", 
            status_code=200,
            headers=get_webhook_security_headers()
        )
            
            # Answer the call with streaming parameters
            # Construct stream URL 
            host = request.headers.get("x-forwarded-host") or request.headers.get("host")
            forwarded_proto = request.headers.get("x-forwarded-proto")
            
            if forwarded_proto == "https":
                protocol = "wss"
            elif request.url.scheme == "https":
                protocol = "wss"
            else:
                protocol = "ws"
            
            # For testing, use your ngrok URL
            # stream_url = "wss://echo.websocket.org"
            stream_url = f"{protocol}://{host}/streams"  # Use this for your ngrok URL
            
            logger.info(f"Answering Telnyx call with streaming to: {stream_url}")
            
            success = TelnyxWebhookHandler.send_call_control_command(
                call_control_id=call_control_id,
                command="answer",
                params={
                    "stream_url": stream_url,
                    "stream_track": "both_tracks",
                    "stream_bidirectional_mode": "rtp",
                    "stream_bidirectional_codec": "PCMU"
                },
                api_key=os.getenv("TELNYX_API_KEY")
            )
            
            if success:
                logger.info(f"Answered Telnyx call {call_control_id}")
                
                # Just create the call record on answer - streaming will be handled on call_answered event
                try:
                    call = await CallService.create_call(
                        assistant_id=assistant.id,
                        call_sid=call_control_id,  # Use call_control_id as call_sid
                        to_phone_number=to_phone,
                        customer_phone_number=from_phone,
                        metadata={
                            "provider": "telnyx", 
                            "call_session_id": call_data.get('call_session_id'),
                            "streaming_enabled": False  # Will be updated when streaming starts
                        },
                    )
                    logger.info(f"Created call record for Telnyx call {call_control_id}")
                    
                except Exception as e:
                    logger.error(f"Error creating call record for Telnyx call {call_control_id}: {e}")
                    
        elif event_type == 'call_answered':
            # Handle call answered - streaming was set up during answer command
            to_phone = call_data.get('to')
            from_phone = call_data.get('from')
            
            logger.info(f"Telnyx call {call_control_id} answered - streaming should already be active")
            
            # Get assistant for webhook sending
            assistant = await assistant_manager.get_assistant_by_phone(to_phone)
            if assistant:
                # Send initial webhook status update now that call is answered
                try:
                    call = await CallService.get_call_by_sid(call_control_id)
                    if call and assistant.webhook_url:
                        asyncio.create_task(
                            WebhookService.send_status_update_webhook(
                                assistant=assistant,
                                call=call,
                                status="in-progress",
                                messages=[]
                            )
                        )
                        logger.info(f"Sent webhook status update for Telnyx call {call_control_id}")
                except Exception as e:
                    logger.error(f"Error sending webhook for answered call {call_control_id}: {e}")
            else:
                logger.error(f"No assistant found for answered Telnyx call to {to_phone}")
            
        elif event_type == 'streaming_started':
            # Handle streaming started event
            logger.info(f"Telnyx streaming started for call {call_control_id}")
            stream_url = call_data.get('stream_url')
            stream_id = call_data.get('stream_id')
            
            # Store the mapping for WebSocket use
            if stream_id:
                telnyx_stream_mappings[stream_id] = call_control_id
                logger.info(f"Stored Telnyx stream mapping: {stream_id} -> {call_control_id}")
            
            logger.info(f"Streaming URL confirmed: {stream_url}")
            
        elif event_type == 'streaming_stopped':
            # Handle streaming stopped event  
            logger.info(f"Telnyx streaming stopped for call {call_control_id}")
            stream_id = call_data.get('stream_id')
            
            # Clean up stream mapping
            if stream_id and stream_id in telnyx_stream_mappings:
                del telnyx_stream_mappings[stream_id]
                logger.info(f"Cleaned up Telnyx stream mapping for stopped stream: {stream_id}")
            
        elif event_type == 'call_hangup':
            # Handle call end
            logger.info(f"Telnyx call {call_control_id} ended")
            
        elif event_type == 'recording_saved':
            # Handle recording completion - similar to Twilio recording callback
            recording_id = call_data.get('recording_id')
            logger.info(f"Telnyx recording {recording_id} saved for call {call_control_id}")
            
            # Process recording similar to Twilio recording callback
            await process_telnyx_recording_saved(
                call_control_id=call_control_id,
                recording_id=recording_id,
                call_data=call_data
            )
            
        elif event_type == 'message_received':
            # Handle incoming SMS message
            message_id = call_data.get('message_id')
            to_phone_number = call_data.get('to_number')
            from_phone_number = call_data.get('from_number')
            message_body = call_data.get('body')
            
            logger.info(
                f"Received Telnyx SMS: from={from_phone_number}, to={to_phone_number}, "
                f"message_id={message_id}, body_length={len(message_body) if message_body else 0}"
            )
            
            # Find the assistant by the phone number that received the SMS
            assistant = await assistant_manager.get_assistant_by_phone(to_phone_number)
            if not assistant:
                logger.warning(f"No assistant found for Telnyx SMS to phone number {to_phone_number}")
            elif not assistant.sms_webhook_url:
                logger.info(f"No SMS webhook URL configured for assistant {assistant.id}")
            else:
                # Create standardized SMS data from the already processed call_data
                standardized_sms_data = {
                    "message_id": message_id,
                    "from": from_phone_number,
                    "to": to_phone_number,
                    "body": message_body or "",
                    "media_urls": []  # Extract from call_data.get('media', []) if needed
                }
                
                # Forward the SMS webhook asynchronously
                asyncio.create_task(
                    SMSWebhookService.process_sms_webhook_async(
                        assistant_sms_webhook_url=assistant.sms_webhook_url,
                        sms_data=standardized_sms_data,
                        provider="telnyx"
                    )
                )
                
                logger.info(f"Forwarded Telnyx SMS webhook for assistant {assistant.id} to {assistant.sms_webhook_url}")
        
        elif event_type in ['message_sent', 'message_finalized']:
            # Handle outgoing SMS status updates (optional - could be useful for logging)
            message_id = call_data.get('message_id')
            logger.info(f"Telnyx SMS {event_type}: message_id={message_id}")
        
        return Response(
            content="", 
            status_code=200,
            headers=get_webhook_security_headers()
        )
        
    except Exception as e:
        logger.error(f"Error handling Telnyx webhook: {e}", exc_info=True)
        return Response(
            content="", 
            status_code=200,
            headers=get_webhook_security_headers()
        )  # Return 200 to avoid retries


@router.websocket("/streams")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Media Streams.
    Handles real-time audio streaming with both Twilio and Telnyx.
    
    Supports:
    - Twilio Media Streams format
    - Telnyx Media Streaming API format
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

                # Detect provider based on message format
                # Telnyx uses different event names and structure
                # Twilio has streamSid/callSid in start object, Telnyx has stream_id/call_control_id
                is_telnyx = (
                    "telnyx" in str(message).lower() or 
                    event_type in ["streaming_started", "streaming_stopped"] or
                    "stream_id" in message or
                    "call_control_id" in message
                )
                
                # Additional check: if we have a start object with streamSid/callSid, it's definitely Twilio
                start_data = message.get("start", {})
                if start_data and ("streamSid" in start_data or "callSid" in start_data):
                    is_telnyx = False

                if event_type == "connected":
                    logger.info(f"Media stream 'connected' event received: {message}")

                elif event_type == "start" or event_type == "streaming_started":
                    logger.info(f"Media stream 'start' event received (is_telnyx={is_telnyx}): {message}")
                    
                    if is_telnyx or event_type == "streaming_started":
                        # Telnyx format - check if data is nested in 'start' object
                        stream_sid = message.get("stream_id") or message.get("id")
                        
                        # For Telnyx, call_control_id and other data might be in the 'start' object
                        start_data = message.get("start", {})
                        if start_data:
                            call_sid = start_data.get("call_control_id")
                            to_number = start_data.get("to")
                            from_number = start_data.get("from")
                        else:
                            # Fallback to direct message fields
                            call_sid = message.get("call_control_id") or message.get("call_leg_id")
                            to_number = message.get("to")
                            from_number = message.get("from")
                        
                        # Telnyx doesn't have custom parameters in the same way
                        is_outbound = False  # Will be determined from call metadata
                        outbound_assistant_id = None
                        outbound_welcome_message = None
                        outbound_agenda = None
                        outbound_to_phone = None
                        
                        tracks = ["inbound", "outbound"]  # Both tracks for Telnyx
                        media_format = {"encoding": "rtp", "sampleRate": 8000, "channels": 1}
                        
                    else:
                        # Twilio format  
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
                        
                        tracks = start_data.get("tracks", [])
                        media_format = start_data.get("mediaFormat", {})
                    
                    # For Telnyx, store the stream mapping immediately after extracting the IDs
                    if is_telnyx and stream_sid and call_sid:
                        telnyx_stream_mappings[stream_sid] = call_sid
                        logger.info(f"Stored Telnyx stream mapping from start event: {stream_sid} -> {call_sid}")

                    if not stream_sid or not call_sid:
                        logger.error("Missing streamSid or callSid in start message")
                        continue

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
                            "provider": "telnyx" if is_telnyx else "twilio",
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
                            to_number=to_number if not is_outbound else from_number,
                            from_number=from_number if not is_outbound else to_number,
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
                    # Process incoming audio - support both Twilio and Telnyx formats
                    if is_telnyx:
                        # Telnyx format: audio is directly in the message
                        audio_data = message.get('media', {}).get("payload", "")
                        track = message.get('media', {}).get("track", "inbound")
                        current_stream_sid = message.get("stream_id") or stream_sid
                        
                        # For Telnyx, try to get call_sid from stored mapping if not already set
                        if not call_sid and current_stream_sid in telnyx_stream_mappings:
                            call_sid = telnyx_stream_mappings[current_stream_sid]
                            logger.info(f"Retrieved call_control_id from stream mapping: {call_sid}")
                    else:
                        # Twilio format: audio is in media object
                        audio_data = message.get("media", {}).get("payload", "")
                        track = message.get("media", {}).get("track", "inbound")
                        current_stream_sid = message.get("streamSid")

                    if not current_stream_sid:
                        logger.error("Missing streamSid/stream_id in media message")
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

                elif event_type == "stop" or event_type == "streaming_stopped":
                    # Get the current stream_sid from the message - support both formats
                    if is_telnyx or event_type == "streaming_stopped":
                        current_stream_sid = message.get("stream_id") or stream_sid
                        # For Telnyx stop events, try multiple ways to get call_control_id
                        if not call_sid:
                            # Try from the stop object first
                            if "stop" in message:
                                call_sid = message["stop"].get("call_control_id")
                                logger.info(f"Extracted call_control_id from stop event: {call_sid}")
                            # If not found, try from stored mapping
                            elif current_stream_sid in telnyx_stream_mappings:
                                call_sid = telnyx_stream_mappings[current_stream_sid]
                                logger.info(f"Retrieved call_control_id from stream mapping for stop event: {call_sid}")
                    else:
                        current_stream_sid = message.get("streamSid")
                        
                    if not current_stream_sid:
                        logger.error("Missing streamSid/stream_id in stop message")
                        continue

                    # End call handling if we have a call_sid
                    if call_sid:
                        try:
                            await call_handler.end_call(call_sid)
                            logger.info(f"Ended call handling for call: {call_sid}")
                            
                            # Clean up Telnyx stream mapping
                            if current_stream_sid in telnyx_stream_mappings:
                                del telnyx_stream_mappings[current_stream_sid]
                                logger.info(f"Cleaned up Telnyx stream mapping for {current_stream_sid}")
                                
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
        
        except RuntimeError as e:
            if "WebSocket is not connected" in str(e):
                logger.info("WebSocket connection closed during message processing")
            else:
                logger.error(f"Runtime error in WebSocket: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error processing WebSocket messages: {e}", exc_info=True)

    except WebSocketDisconnect:
        logger.warning(f"WebSocket disconnected before acceptance")

    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}", exc_info=True)


@router.post("/calls/initiate", response_model=InitiateCallResponse)
async def initiate_outbound_call(
    call_data: InitiateCallRequest,
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Initiates an outbound call from an assistant to a specified phone number.
    This endpoint is protected and requires authentication.
    """
    try:
        from_phone_number = call_data.from_phone_number
        to_phone_number = call_data.to_phone_number
        welcome_message = call_data.welcome_message
        agenda = call_data.agenda
        
        # Validate required fields
        if not from_phone_number:
            raise HTTPException(status_code=400, detail="from_phone_number is required")
        if not to_phone_number:
            raise HTTPException(status_code=400, detail="to_phone_number is required")

        # Get the assistant first to determine provider
        assistant = await assistant_manager.get_assistant_by_phone(from_phone_number)
        if not assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")
        
        # Verify user owns this assistant/phone number
        if assistant.organization_id != current_user.organization_id:
            raise HTTPException(status_code=403, detail="Unauthorized: phone number belongs to different organization")

        # Create unified telephony service for this assistant
        # Use phone number to determine provider rather than assistant to avoid SQLAlchemy session issues
        telephony_service = await UnifiedTelephonyService.create_from_phone_number(from_phone_number)
        
        # Validate phone number format using the provider's validation
        if not telephony_service.validate_phone_number(to_phone_number):
            raise HTTPException(status_code=400, detail="Invalid phone number format. Use E.164 format (e.g., +1234567890)")

        # Use the get_twiml_webhook_url function to determine the webhook URL
        webhook_url = get_twiml_webhook_url()
        
        # Prepare call metadata to pass through the webhook
        call_metadata = {
            "outbound": "true",
            "assistant_id": str(assistant.id),
            "welcome_message": welcome_message,
            "agenda": agenda,
            "to_phone_number": to_phone_number
        }
        
        # Check provider type for logging
        provider_type = telephony_service.get_provider_type()
        logger.info(f"Initiating outbound call using {provider_type} provider for assistant {assistant.id}")
        
        # For Telnyx calls, add streaming URL to metadata
        if provider_type == "telnyx":
            # Construct stream URL (same as webhook URL host but WebSocket protocol)
            try:
                from urllib.parse import urlparse
                parsed_webhook = urlparse(webhook_url)
                
                # Determine protocol
                if parsed_webhook.scheme == "https":
                    protocol = "wss"
                else:
                    protocol = "ws"
                
                stream_url = f"{protocol}://{parsed_webhook.netloc}/streams"
                call_metadata["stream_url"] = stream_url
                
                logger.info(f"Added stream URL for Telnyx outbound call: {stream_url}")
            except Exception as e:
                logger.warning(f"Could not construct stream URL for Telnyx call: {e}")
        
        # Initiate the outbound call through the unified service
        call_sid = telephony_service.initiate_outbound_call(
            to_phone_number=to_phone_number,
            from_phone_number=from_phone_number,
            webhook_url=webhook_url,
            call_metadata=call_metadata
        )
        
        if not call_sid:
            raise HTTPException(status_code=500, detail="Failed to initiate outbound call")
        
        logger.info(
            f"Initiated outbound call {call_sid} from assistant {assistant.id} "
            f"to {to_phone_number} with agenda: {agenda[:100] if agenda else 'None'}..."
        )
        
        # Return success response
        return {
            "success": True,
            "call_sid": call_sid,
            "message": "Outbound call initiated successfully",
            "assistant_id": assistant.id,
            "to_phone_number": to_phone_number,
            "from_phone_number": from_phone_number
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating outbound call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/sms/send", response_model=SendSMSResponse)
async def send_sms(
    sms_data: SendSMSRequest,
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Send an SMS message through an assistant.
    
    This endpoint allows sending SMS messages using the telephony provider
    associated with the assistant that owns the from_phone_number.
    
    The system will automatically:
    1. Find the assistant associated with the from_phone_number
    2. Use the assistant's configured telephony provider (Twilio or Telnyx)
    3. Send the SMS through the appropriate provider
    
    Args:
        sms_data: SMS request containing from/to numbers and message
        
    Returns:
        SendSMSResponse: Success status, message ID, and provider info
    """
    try:
        from_phone_number = sms_data.from_phone_number
        to_phone_number = sms_data.to_phone_number
        message = sms_data.message
        media_urls = sms_data.media_urls
        
        # Validate required fields
        if not from_phone_number:
            raise HTTPException(status_code=400, detail="from_phone_number is required")
        if not to_phone_number:
            raise HTTPException(status_code=400, detail="to_phone_number is required")
        if not message or len(message.strip()) == 0:
            raise HTTPException(status_code=400, detail="message is required")

        # Get the assistant first to determine provider
        assistant = await assistant_manager.get_assistant_by_phone(from_phone_number)
        if not assistant:
            raise HTTPException(
                status_code=404, 
                detail=f"No assistant found for phone number {from_phone_number}"
            )
        
        # Verify user owns this assistant/phone number
        if assistant.organization_id != current_user.organization_id:
            raise HTTPException(status_code=403, detail="Unauthorized: phone number belongs to different organization")

        # Create unified telephony service for this assistant
        # Use phone number to determine provider rather than assistant to avoid SQLAlchemy session issues
        telephony_service = await UnifiedTelephonyService.create_from_phone_number(from_phone_number)
        
        # Validate phone number format using the provider's validation
        if not telephony_service.validate_phone_number(to_phone_number):
            raise HTTPException(
                status_code=400, 
                detail="Invalid to_phone_number format. Use E.164 format (e.g., +1234567890)"
            )
        
        if not telephony_service.validate_phone_number(from_phone_number):
            raise HTTPException(
                status_code=400, 
                detail="Invalid from_phone_number format. Use E.164 format (e.g., +1234567890)"
            )

        # Get provider type for logging and response
        provider_type = telephony_service.get_provider_type()
        logger.info(f"Sending SMS using {provider_type} from {from_phone_number} to {to_phone_number}")
        
        # Send the SMS using the unified service
        message_id = telephony_service.send_sms(
            to_phone_number=to_phone_number,
            from_phone_number=from_phone_number,
            message=message,
            media_urls=media_urls
        )
        
        if message_id:
            logger.info(f"Successfully sent SMS via {provider_type}, message ID: {message_id}")
            return SendSMSResponse(
                success=True,
                message_id=message_id,
                message=f"SMS sent successfully via {provider_type}",
                provider=provider_type
            )
        else:
            logger.error(f"Failed to send SMS via {provider_type}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to send SMS via {provider_type}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending SMS: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/recording-status")
async def recording_status_callback(request: Request):
    """
    Handle Twilio recording status callbacks.
    This endpoint is called by Twilio when a recording is completed or failed.
    """
    try:
        # Validate Twilio webhook signature for security
        body = await request.body()
        require_twilio_webhook_auth(request, body)
        
        # Parse form data manually since body was consumed
        form_data = {}
        if body:
            body_str = body.decode('utf-8')
            for param in body_str.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    import urllib.parse
                    key = urllib.parse.unquote_plus(key)
                    value = urllib.parse.unquote_plus(value)
                    form_data[key] = value

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
            return Response(
            content="", 
            status_code=200,
            headers=get_webhook_security_headers()
        )

        # Get the call/conversation from database with relationships loaded
        from app.db.database import get_async_db_session
        async with await get_async_db_session() as db:
            from app.db.models import Call, Assistant
            from sqlalchemy.orm import selectinload
            
            # Query call with assistant and organization loaded
            result = await db.execute(
                select(Call)
                .options(
                    selectinload(Call.assistant).selectinload(Assistant.organization)
                )
                .where(Call.call_sid == call_sid)
            )
            call = result.scalar_one_or_none()
            
            if not call:
                logger.error(f"No call found for call_sid {call_sid}")
                return Response(content="Call not found", status_code=404)

            # Get the assistant
            assistant = call.assistant
            if not assistant:
                logger.error(f"No assistant found for call {call_sid}")
                return Response(content="Assistant not found", status_code=404)

            # Download the recording from Twilio
            # Get Twilio credentials from assistant's organization
            twilio_account_sid = assistant.organization.twilio_account_sid or os.getenv(
                "TWILIO_ACCOUNT_SID"
            )
            twilio_auth_token = assistant.organization.twilio_auth_token or os.getenv(
                "TWILIO_AUTH_TOKEN"
            )
            
            # Extract values we need outside the session
            call_id = call.id
            to_phone_number = call.to_phone_number

        if not twilio_account_sid or not twilio_auth_token:
            logger.error(
                f"Missing Twilio credentials for downloading recording {recording_sid}"
            )
            return Response(
                content="Twilio credentials not configured", status_code=500
            )

        # Create unified telephony service for this assistant
        # Use phone number from call to determine provider rather than assistant to avoid SQLAlchemy session issues
        telephony_service = await UnifiedTelephonyService.create_from_phone_number(to_phone_number)
        provider_type = telephony_service.get_provider_type()
        
        logger.info(f"Downloading recording {recording_sid} using {provider_type} provider")
        
        # Download recording content using the unified service
        download_result = telephony_service.download_recording_content(recording_id=recording_sid)

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

                    # Update existing recording record (created when call started)
                    existing_recordings = await CallService.get_call_recordings(call_sid)
                    
                    if existing_recordings:
                        # Update the existing processing recording
                        recording = existing_recordings[0]  # Should be only one
                        
                        try:
                            from app.db.database import get_async_db_session
                            
                            async with await get_async_db_session() as db:
                                # Update the recording with Twilio data
                                recording.recording_sid = recording_sid
                                recording.s3_key = s3_key
                                recording.s3_url = s3_url
                                recording.duration = (
                                    float(recording_duration)
                                    if recording_duration
                                    else None
                                )
                                recording.file_size = len(recording_content)
                                recording.status = "completed"
                                recording.channels = (
                                    int(recording_channels) if recording_channels else 2
                                )
                                recording.recording_metadata = metadata
                                recording.uploaded_at = datetime.utcnow()
                                
                                db.add(recording)
                                await db.commit()
                                await db.refresh(recording)
                                
                                logger.info(
                                    f"Updated recording {recording.id} with Twilio recording data for call {call_sid}"
                                )
                        except Exception as update_error:
                            logger.error(
                                f"Failed to update recording {recording.id} with Twilio data: {update_error}"
                            )
                    else:
                        # Fallback: create new recording if none exists
                        logger.warning(f"No existing recording found for call {call_sid}, creating new one")
                        result = await CallService.create_recording(
                            call_sid=call_sid,
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
                            recording_type="mixed",  # Twilio dual-channel is equivalent to mixed
                            recording_source="s3",
                            status="completed",
                        )
                        if result:
                            logger.info(
                                f"Created fallback recording record for Twilio recording {recording_sid}"
                            )

                else:
                    logger.error(f"Failed to upload recording {recording_sid} to S3")
            else:
                logger.error("S3 service not available for storing Twilio recording")

        except Exception as s3_error:
            logger.error(f"Error uploading Twilio recording to S3: {s3_error}")

        # Return success response to Twilio
        return Response(
            content="", 
            status_code=200,
            headers=get_webhook_security_headers()
        )

    except Exception as e:
        logger.error(f"Error handling recording status callback: {e}", exc_info=True)
        # Return 200 to Twilio to avoid retries on our errors
        return Response(
            content="", 
            status_code=200,
            headers=get_webhook_security_headers()
        )


@router.post("/twilio-sms-webhook")
async def twilio_sms_webhook(request: Request):
    """
    Handle Twilio SMS webhooks.
    This endpoint is called by Twilio when an SMS is received.
    Forwards the SMS data to the assistant's configured sms_webhook_url.
    """
    try:
        # Validate Twilio webhook signature for security
        body = await request.body()
        require_twilio_webhook_auth(request, body)
        
        # Parse form data manually since body was consumed
        form_data = {}
        if body:
            body_str = body.decode('utf-8')
            for param in body_str.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    import urllib.parse
                    key = urllib.parse.unquote_plus(key)
                    value = urllib.parse.unquote_plus(value)
                    form_data[key] = value
        
        # Extract SMS details
        to_phone_number = form_data.get("To")
        from_phone_number = form_data.get("From")
        message_body = form_data.get("Body")
        message_sid = form_data.get("MessageSid")
        
        logger.info(
            f"Received Twilio SMS webhook: from={from_phone_number}, to={to_phone_number}, "
            f"message_sid={message_sid}, body_length={len(message_body) if message_body else 0}"
        )
        
        # Validate required fields
        if not all([to_phone_number, from_phone_number, message_sid]):
            logger.error(
                f"Missing required SMS fields: to={to_phone_number}, from={from_phone_number}, "
                f"message_sid={message_sid}"
            )
            return Response(content="Missing required fields", status_code=400)
        
        # Find the assistant by the phone number that received the SMS
        assistant = await assistant_manager.get_assistant_by_phone(to_phone_number)
        if not assistant:
            logger.warning(f"No assistant found for Twilio SMS to phone number {to_phone_number}")
            return Response(content="Assistant not found", status_code=404)
        
        # Check if assistant has an SMS webhook URL configured
        if not assistant.sms_webhook_url:
            logger.info(f"No SMS webhook URL configured for assistant {assistant.id}")
            return Response(
            content="", 
            status_code=200,
            headers=get_webhook_security_headers()
        )
        
        # Normalize the SMS data
        normalized_sms_data = SMSWebhookService.normalize_twilio_sms_data(dict(form_data))
        
        # Forward the SMS webhook asynchronously
        asyncio.create_task(
            SMSWebhookService.process_sms_webhook_async(
                assistant_sms_webhook_url=assistant.sms_webhook_url,
                sms_data=normalized_sms_data,
                provider="twilio"
            )
        )
        
        logger.info(f"Forwarded Twilio SMS webhook for assistant {assistant.id} to {assistant.sms_webhook_url}")
        
        # Return success response to Twilio
        return Response(
            content="", 
            status_code=200,
            headers=get_webhook_security_headers()
        )
        
    except Exception as e:
        logger.error(f"Error handling Twilio SMS webhook: {e}", exc_info=True)
        # Return 200 to Twilio to avoid retries on our errors
        return Response(
            content="", 
            status_code=200,
            headers=get_webhook_security_headers()
        )