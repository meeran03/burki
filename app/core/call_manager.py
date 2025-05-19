import logging
import os
import base64
import json
from typing import Dict, Optional, Any
from fastapi import WebSocket
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

logger = logging.getLogger(__name__)

class CallManager:
    """
    Manages Twilio call operations like ending calls, forwarding calls,
    and playing messages.
    """
    
    def __init__(self):
        """Initialize the call manager with Twilio credentials."""
        # Initialize Twilio client
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_client = None
        self.end_call_message = os.getenv("END_CALL_MESSAGE", "Thank you for calling. Goodbye!")
        
        # Active calls tracking
        self.active_calls = {}
        
        # Initialize Twilio client if credentials are available
        if self.account_sid and self.auth_token:
            self.twilio_client = Client(self.account_sid, self.auth_token)
        else:
            logger.warning("Twilio credentials not found. Call control features will be limited.")
    
    async def register_call(self, call_sid: str, stream_sid: str, websocket: WebSocket):
        """
        Register a new call with the call manager.
        
        Args:
            call_sid: Twilio Call SID
            stream_sid: Twilio Stream SID
            websocket: WebSocket connection for the call
        """
        self.active_calls[call_sid] = {
            "stream_sid": stream_sid,
            "websocket": websocket,
            "status": "active"
        }
        logger.info(f"Registered call {call_sid} with stream {stream_sid}")
    
    async def end_call(self, call_sid: str, custom_message: Optional[str] = None):
        """
        End a call with an optional goodbye message.
        
        Args:
            call_sid: Twilio Call SID
            custom_message: Optional custom goodbye message
        """
        if call_sid not in self.active_calls:
            logger.warning(f"Attempted to end unknown call: {call_sid}")
            return False
        
        # Send goodbye message if provided
        message = custom_message or self.end_call_message
        if message:
            await self.send_message(call_sid, message)
        
        # Mark call as ending
        self.active_calls[call_sid]["status"] = "ending"
        
        # Use Twilio API to end the call
        if self.twilio_client:
            try:
                self.twilio_client.calls(call_sid).update(status="completed")
                logger.info(f"Successfully ended call {call_sid}")
                return True
            except Exception as e:
                logger.error(f"Error ending call {call_sid}: {e}")
                return False
        else:
            logger.warning(f"No Twilio client available to end call {call_sid}")
            return False
    
    async def forward_call(self, call_sid: str, destination: str, message: Optional[str] = None):
        """
        Forward a call to another number.
        
        Args:
            call_sid: Twilio Call SID
            destination: Destination phone number
            message: Optional message to play before forwarding
        """
        if call_sid not in self.active_calls:
            logger.warning(f"Attempted to forward unknown call: {call_sid}")
            return False
        
        # Send transfer message if provided
        if message:
            await self.send_message(call_sid, message)
        
        # Use Twilio API to redirect the call
        if self.twilio_client:
            try:
                # Create a TwiML response to redirect the call
                response = VoiceResponse()
                response.redirect(destination)
                
                # Update the call with the new TwiML
                self.twilio_client.calls(call_sid).update(twiml=str(response))
                logger.info(f"Forwarding call {call_sid} to {destination}")
                return True
            except Exception as e:
                logger.error(f"Error forwarding call {call_sid}: {e}")
                return False
        else:
            logger.warning(f"No Twilio client available to forward call {call_sid}")
            return False
    
    async def send_message(self, call_sid: str, message: str):
        """
        Send a text message to be spoken to the caller.
        In a real implementation, this would use TTS.
        
        Args:
            call_sid: Twilio Call SID
            message: Message to be spoken to the caller
        """
        if call_sid not in self.active_calls:
            logger.warning(f"Attempted to send message to unknown call: {call_sid}")
            return False
        
        # For now, just log the message
        # In a real implementation, this would convert the text to speech
        # and send it to the caller through the websocket
        logger.info(f"Would send message to {call_sid}: {message}")
        
        # Here we'd normally send audio through the websocket
        # This is just a placeholder
        websocket = self.active_calls[call_sid]["websocket"]
        stream_sid = self.active_calls[call_sid]["stream_sid"]
        
        # In a real implementation, this would be real audio data
        # For now, we're just sending a placeholder message event
        try:
            await websocket.send_json({
                "event": "message",
                "streamSid": stream_sid,
                "message": message
            })
            return True
        except Exception as e:
            logger.error(f"Error sending message to {call_sid}: {e}")
            return False
    
    def unregister_call(self, call_sid: str):
        """
        Unregister a call when it ends.
        
        Args:
            call_sid: Twilio Call SID
        """
        if call_sid in self.active_calls:
            logger.info(f"Unregistering call {call_sid}")
            del self.active_calls[call_sid] 