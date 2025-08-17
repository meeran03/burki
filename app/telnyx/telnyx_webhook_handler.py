"""
Telnyx webhook handler for processing incoming Call Control events.
Provides TwiML-like XML responses for compatibility.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import Request
from fastapi.responses import Response
import json

logger = logging.getLogger(__name__)


class TelnyxWebhookHandler:
    """
    Handler for Telnyx Call Control webhooks.
    Converts Telnyx events to be compatible with existing Twilio-based flows.
    """

    @staticmethod
    def process_webhook(request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming Telnyx webhook and extract call information.

        Args:
            request_data: The webhook payload from Telnyx

        Returns:
            Optional[Dict[str, Any]]: Standardized call data or None if not a call event
        """
        try:
            # Extract event data
            event_type = request_data.get("data", {}).get("event_type")
            payload = request_data.get("data", {}).get("payload", {})

            if event_type == "call.initiated":
                return TelnyxWebhookHandler._handle_call_initiated(payload)
            elif event_type == "call.answered":
                return TelnyxWebhookHandler._handle_call_answered(payload)
            elif event_type == "call.hangup":
                return TelnyxWebhookHandler._handle_call_hangup(payload)
            elif event_type == "call.recording.saved":
                return TelnyxWebhookHandler._handle_recording_saved(payload)
            else:
                logger.debug(f"Unhandled Telnyx event type: {event_type}")
                return None

        except Exception as e:
            logger.error(f"Error processing Telnyx webhook: {e}")
            return None

    @staticmethod
    def _handle_call_initiated(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle call.initiated event - equivalent to Twilio incoming call.

        Args:
            payload: The call payload from Telnyx

        Returns:
            Dict[str, Any]: Standardized call data
        """
        return {
            "event_type": "call_initiated",
            "call_control_id": payload.get("call_control_id"),
            "call_session_id": payload.get("call_session_id"),
            "from": payload.get("from"),
            "to": payload.get("to"),
            "direction": payload.get("direction"),
            "connection_id": payload.get("connection_id"),
            "state": payload.get("state"),
            "client_state": payload.get("client_state"),
            "provider": "telnyx",
        }

    @staticmethod
    def _handle_call_answered(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle call.answered event.

        Args:
            payload: The call payload from Telnyx

        Returns:
            Dict[str, Any]: Standardized call data
        """
        return {
            "event_type": "call_answered",
            "call_control_id": payload.get("call_control_id"),
            "call_session_id": payload.get("call_session_id"),
            "from": payload.get("from"),
            "to": payload.get("to"),
            "direction": payload.get("direction"),
            "state": payload.get("state"),
            "provider": "telnyx",
        }

    @staticmethod
    def _handle_call_hangup(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle call.hangup event.

        Args:
            payload: The call payload from Telnyx

        Returns:
            Dict[str, Any]: Standardized call data
        """
        return {
            "event_type": "call_hangup",
            "call_control_id": payload.get("call_control_id"),
            "call_session_id": payload.get("call_session_id"),
            "hangup_cause": payload.get("hangup_cause"),
            "hangup_source": payload.get("hangup_source"),
            "sip_hangup_cause": payload.get("sip_hangup_cause"),
            "provider": "telnyx",
        }

    @staticmethod
    def _handle_recording_saved(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle call.recording.saved event.

        Args:
            payload: The recording payload from Telnyx

        Returns:
            Dict[str, Any]: Standardized recording data
        """
        return {
            "event_type": "recording_saved",
            "recording_id": payload.get("recording_id"),
            "call_control_id": payload.get("call_control_id"),
            "call_session_id": payload.get("call_session_id"),
            "channels": payload.get("channels"),
            "duration_millis": payload.get("duration_millis"),
            "recording_urls": payload.get("recording_urls", {}),
            "status": payload.get("status"),
            "provider": "telnyx",
        }

    @staticmethod
    def generate_twiml_response(
        assistant_id: Optional[str] = None,
        is_outbound: bool = False,
        agenda: Optional[str] = None,
        custom_welcome_message: Optional[str] = None,
    ) -> str:
        """
        Generate TwiML-compatible response for Telnyx webhook.
        Since Telnyx uses Call Control API instead of TwiML, this is for compatibility.

        Args:
            assistant_id: The assistant ID to use for the call
            is_outbound: Whether this is an outbound call
            agenda: Optional agenda for outbound calls
            custom_welcome_message: Optional custom welcome message

        Returns:
            str: TwiML response (for compatibility, though Telnyx uses JSON)
        """
        # For Telnyx, we'll return JSON instructions instead of TwiML
        # This maintains compatibility with existing call flow

        response_data = {
            "commands": [
                {"command": "answer"},
                {
                    "command": "start_stream",
                    "params": {
                        "stream_name": (
                            f"stream_{assistant_id}"
                            if assistant_id
                            else "default_stream"
                        ),
                        "webhook_url": "/telnyx-stream",  # Custom endpoint for Telnyx streams
                        "assistant_id": assistant_id,
                        "is_outbound": is_outbound,
                        "agenda": agenda,
                        "custom_welcome_message": custom_welcome_message,
                    },
                },
            ]
        }

        # For compatibility, we still return TwiML format
        # The actual Telnyx Call Control commands would be sent via API calls
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Say>Please wait while we connect you to your AI assistant.</Say>
            <Connect>
                <Stream url="wss://{{your-domain}}/telnyx-websocket">
                    <Parameter name="assistant_id" value="{assistant_id or ''}" />
                    <Parameter name="is_outbound" value="{str(is_outbound).lower()}" />
                    <Parameter name="agenda" value="{agenda or ''}" />
                    <Parameter name="custom_welcome_message" value="{custom_welcome_message or ''}" />
                    <Parameter name="provider" value="telnyx" />
                </Stream>
            </Connect>
        </Response>"""

        return twiml

    @staticmethod
    def send_call_control_command(
        call_control_id: str,
        command: str,
        params: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> bool:
        """
        Send a Call Control command to Telnyx.

        Args:
            call_control_id: The call control ID
            command: The command to send (answer, hangup, etc.)
            params: Optional parameters for the command
            api_key: Optional Telnyx API key

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import telnyx
            from app.telnyx.telnyx_service import TelnyxService

            if not TelnyxService.get_telnyx_client(api_key):
                logger.error(
                    "Could not initialize Telnyx client for Call Control command"
                )
                return False

            if command == "answer":
                # Include streaming parameters if provided
                answer_params = {}
                if params:
                    if params.get("stream_url"):
                        answer_params.update({
                            "stream_url": params.get("stream_url"),
                            "stream_track": params.get("stream_track", "both_tracks"),
                            "stream_bidirectional_mode": params.get("stream_bidirectional_mode", "rtp"),
                        })
                        # Add codec if specified
                        if params.get("stream_bidirectional_codec"):
                            answer_params["stream_bidirectional_codec"] = params.get("stream_bidirectional_codec")
                
                result = telnyx.Call.create_answer(call_control_id, **answer_params)

            elif command == "hangup":
                result = telnyx.Call.create_hangup(call_control_id)

            elif command == "start_recording":
                recording_params = params or {}
                result = telnyx.Call.create_record_start(
                    call_control_id,
                    channels=recording_params.get("channels", "dual"),
                    format=recording_params.get("format", "mp3"),
                    play_beep=recording_params.get("play_beep", False),
                )

            elif command == "stop_recording":
                result = telnyx.Call.create_record_stop(call_control_id)

            elif command == "start_stream":
                stream_params = params or {}
                # Note: Telnyx streaming would be different from Twilio
                # This is a placeholder for future implementation
                logger.info(
                    f"Stream start requested for {call_control_id} - not yet implemented"
                )
                return True

            elif command == "transfer":
                transfer_params = params or {}
                transfer_to = transfer_params.get("to")
                if transfer_to:
                    result = telnyx.Call.create_transfer(
                        call_control_id, to=transfer_to
                    )
                else:
                    logger.error("Transfer command requires 'to' parameter")
                    return False

            elif command == "bridge":
                bridge_params = params or {}
                other_call_id = bridge_params.get("other_call_control_id")
                if other_call_id:
                    result = telnyx.Call.create_bridge(
                        call_control_id, other_call_control_id=other_call_id
                    )
                else:
                    logger.error(
                        "Bridge command requires 'other_call_control_id' parameter"
                    )
                    return False

            elif command == "speak":
                speak_params = params or {}
                text = speak_params.get("text", "")
                voice = speak_params.get("voice", "female")
                language = speak_params.get("language", "en-US")

                result = telnyx.Call.create_speak(
                    call_control_id, payload=text, voice=voice, language=language
                )

            elif command == "gather":
                gather_params = params or {}
                # Telnyx gather using DTMF
                result = telnyx.Call.create_gather_using_audio(
                    call_control_id,
                    audio_url=gather_params.get("audio_url", ""),
                    valid_digits=gather_params.get("valid_digits", "0123456789#*"),
                    max=gather_params.get("max_digits", 1),
                    timeout_millis=gather_params.get("timeout", 5000),
                )

            else:
                logger.error(f"Unknown Call Control command: {command}")
                return False

            logger.info(
                f"Successfully sent Call Control command {command} to {call_control_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Error sending Call Control command {command}: {e}")
            return False


def validate_telnyx_webhook(request: Request, body: bytes = None) -> bool:
    """
    Validate Telnyx webhook signature using Ed25519 verification.

    According to Telnyx documentation, webhooks should be verified using
    Ed25519 signature verification for security.

    Args:
        request: The FastAPI request object
        body: The raw request body bytes

    Returns:
        bool: True if valid or validation is disabled, False otherwise
    """
    try:
        import os
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
        import base64

        # Get webhook signature from headers
        signature = request.headers.get("telnyx-signature-ed25519")
        timestamp = request.headers.get("telnyx-timestamp")

        if not signature or not timestamp:
            logger.warning("Missing Telnyx webhook signature headers")
            # For development, allow unsigned webhooks if TELNYX_WEBHOOK_VERIFICATION is disabled
            if os.getenv("TELNYX_WEBHOOK_VERIFICATION", "true").lower() == "false":
                return True
            return False

        # Get Telnyx public key from environment or use default
        public_key_b64 = os.getenv("TELNYX_PUBLIC_KEY")
        if not public_key_b64:
            # Telnyx's default public key (this should be retrieved from Mission Control Portal)
            logger.warning("TELNYX_PUBLIC_KEY not set, using development mode")
            if os.getenv("TELNYX_WEBHOOK_VERIFICATION", "true").lower() == "false":
                return True
            return False

        try:
            # Decode the public key
            public_key_bytes = base64.b64decode(public_key_b64)
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)

            # Prepare the signed payload: timestamp + body
            if body is None:
                logger.error("Request body is required for signature verification")
                return False

            signed_payload = timestamp.encode("utf-8") + body

            # Decode the signature
            signature_bytes = base64.b64decode(signature)

            # Verify the signature
            public_key.verify(signature_bytes, signed_payload)

            logger.debug("Telnyx webhook signature verified successfully")
            return True

        except InvalidSignature:
            logger.error("Invalid Telnyx webhook signature")
            return False
        except Exception as e:
            logger.error(f"Error verifying Telnyx webhook signature: {e}")
            return False

    except ImportError:
        logger.warning(
            "cryptography library not available, skipping signature verification"
        )
        # Allow webhooks if cryptography is not installed (development mode)
        return True
    except Exception as e:
        logger.error(f"Error validating Telnyx webhook: {e}")
        return False
