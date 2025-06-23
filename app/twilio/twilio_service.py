import os
import logging
import requests
from typing import Optional, List, Dict, Any, Tuple
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

logger = logging.getLogger(__name__)

class TwilioService:
    """
    Service for interacting with the Twilio API to manage phone numbers and webhooks.
    """
    
    @staticmethod
    def get_twilio_client(account_sid: Optional[str] = None, auth_token: Optional[str] = None) -> Optional[Client]:
        """
        Get a Twilio client instance.
        
        Args:
            account_sid: Optional Twilio Account SID (uses env var if not provided)
            auth_token: Optional Twilio Auth Token (uses env var if not provided)
            
        Returns:
            Optional[Client]: Twilio client instance or None if credentials are missing
        """
        # Use provided credentials or fall back to environment variables
        account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        
        if not account_sid or not auth_token:
            logger.warning("Missing Twilio credentials. Cannot create Twilio client.")
            return None
        
        return Client(account_sid, auth_token)
    
    @staticmethod
    def get_available_phone_numbers(
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all available phone numbers from the Twilio account.
        
        Args:
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            List[Dict[str, Any]]: List of phone numbers with their details
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        if not client:
            return []
        
        try:
            # Get all incoming phone numbers in the account
            phone_numbers = client.incoming_phone_numbers.list()
            
            # Format the response
            formatted_numbers = []
            for phone in phone_numbers:
                formatted_numbers.append({
                    "sid": phone.sid,
                    "phone_number": phone.phone_number,
                    "friendly_name": phone.friendly_name,
                    "capabilities": phone.capabilities,
                })
            
            return formatted_numbers
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error fetching phone numbers: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching phone numbers: {e}")
            return []
    
    @staticmethod
    def update_phone_webhook(
        phone_number: str,
        webhook_url: str,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> bool:
        """
        Update the 'A Call Comes In' webhook URL for a phone number.
        
        Args:
            phone_number: The phone number to update (E.164 format)
            webhook_url: The webhook URL to set
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            bool: True if successful, False otherwise
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        if not client:
            return False
        
        try:
            # List incoming phone numbers to find the one we want to update
            numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
            
            if not numbers:
                logger.error(f"No phone number found matching {phone_number}")
                return False
            
            # Update the first matching phone number
            incoming_phone_number = numbers[0]
            incoming_phone_number.update(
                voice_url=webhook_url,
                voice_method='POST'
            )
            
            logger.info(f"Updated webhook for {phone_number} to {webhook_url}")
            return True
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error updating phone webhook: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating phone webhook: {e}")
            return False
    
    @staticmethod
    def get_phone_number_info(
        phone_number: str,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Optional[dict]:
        """
        Get information about a phone number.
        
        Args:
            phone_number: The phone number to query (E.164 format)
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            Optional[dict]: Phone number information or None if not found
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        if not client:
            return None
        
        try:
            numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
            
            if not numbers:
                logger.warning(f"No phone number found matching {phone_number}")
                return None
            
            # Return info about the first matching phone number
            phone = numbers[0]
            return {
                "sid": phone.sid,
                "phone_number": phone.phone_number,
                "friendly_name": phone.friendly_name,
                "voice_url": phone.voice_url,
                "voice_method": phone.voice_method,
                "capabilities": phone.capabilities,
            }
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error fetching phone info: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching phone info: {e}")
            return None 

    @staticmethod
    def end_call(
        call_sid: str,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> bool:
        """
        End a Twilio call via the REST API.
        
        Args:
            call_sid: The Twilio Call SID to end
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            bool: True if successful, False otherwise
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        if not client:
            logger.error("Could not get Twilio client to end call")
            return False
        
        try:
            # Update call status to completed to end the call
            client.calls(call_sid).update(status="completed")
            logger.info(f"Successfully ended call {call_sid} via Twilio API")
            return True
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error ending call: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error ending call: {e}")
            return False
    
    @staticmethod
    def transfer_call(
        call_sid: str,
        destination: str,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> bool:
        """
        Transfer a Twilio call to another number using TwiML.
        
        Args:
            call_sid: The Twilio Call SID to transfer
            destination: The phone number to transfer to (E.164 format)
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            bool: True if successful, False otherwise
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        if not client:
            logger.error("Could not get Twilio client to transfer call")
            return False
        
        try:
            # Create TwiML to transfer the call
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
            <Response>
                <Dial>{destination}</Dial>
            </Response>
            """
            
            # Update the call with the transfer TwiML
            client.calls(call_sid).update(twiml=twiml)
            logger.info(f"Successfully initiated transfer of call {call_sid} to {destination}")
            return True
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error transferring call: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error transferring call: {e}")
            return False

    @staticmethod
    def start_call_recording(
        call_sid: str,
        recording_channels: str = "dual",
        recording_status_callback: Optional[str] = None,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Optional[str]:
        """
        Start recording a Twilio call.
        
        Args:
            call_sid: The Twilio Call SID to record
            recording_channels: Recording channels (mono, dual)
            recording_status_callback: URL for recording status callbacks
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            Optional[str]: Recording SID if successful, None otherwise
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        if not client:
            logger.error("Could not get Twilio client to start recording")
            return None
        
        try:
            # Start recording with best quality settings
            recording = client.calls(call_sid).recordings.create(
                recording_channels=recording_channels,
                recording_status_callback=recording_status_callback,
                recording_status_callback_event=["completed", "failed"],
            )
            
            logger.info(f"Started recording for call {call_sid}, recording SID: {recording.sid}")
            return recording.sid
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error starting recording: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error starting recording: {e}")
            return None

    @staticmethod
    def get_recording_info(
        recording_sid: str,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a recording.
        
        Args:
            recording_sid: The Recording SID
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            Optional[Dict[str, Any]]: Recording information or None if not found
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        if not client:
            logger.error("Could not get Twilio client to fetch recording info")
            return None
        
        try:
            recording = client.recordings(recording_sid).fetch()
            
            return {
                "sid": recording.sid,
                "call_sid": recording.call_sid,
                "duration": recording.duration,
                "date_created": recording.date_created,
                "status": recording.status,
                "uri": recording.uri,
                "channels": recording.channels,
                "source": recording.source,
            }
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error fetching recording info: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching recording info: {e}")
            return None

    @staticmethod
    def download_recording_content(
        recording_sid: str,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Optional[Tuple[str, bytes]]:
        """
        Download the content of a recording from Twilio.
        
        Args:
            recording_sid: The Recording SID
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            Optional[Tuple[str, bytes]]: Recording filename and content if successful, None otherwise
        """
        # Use provided credentials or fall back to environment variables
        account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        
        if not account_sid or not auth_token:
            logger.error("Missing Twilio credentials for downloading recording")
            return None
        
        try:
            # Construct the recording download URL
            recording_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Recordings/{recording_sid}.mp3"
            
            # Download the recording with authentication
            response = requests.get(
                recording_url,
                auth=(account_sid, auth_token),
                timeout=30
            )
            response.raise_for_status()
            
            # Generate filename
            filename = f"recording_{recording_sid}.mp3"
            
            logger.info(f"Successfully downloaded recording {recording_sid}")
            return filename, response.content
            
        except requests.RequestException as e:
            logger.error(f"Error downloading recording content: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading recording content: {e}")
            return None

    @staticmethod
    def initiate_outbound_call(
        to_phone_number: str,
        from_phone_number: str,
        webhook_url: str,
        call_metadata: Optional[Dict[str, Any]] = None,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Optional[str]:
        """
        Initiate an outbound call through Twilio.
        
        Args:
            to_phone_number: The phone number to call (E.164 format)
            from_phone_number: The phone number to call from (must be a Twilio number)
            webhook_url: The webhook URL for handling the call
            call_metadata: Optional metadata to include in the webhook URL
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            Optional[str]: Call SID if successful, None otherwise
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        if not client:
            logger.error("Could not get Twilio client to initiate outbound call")
            return None
        
        try:
            # Build webhook URL with metadata if provided
            full_webhook_url = webhook_url
            if call_metadata:
                # Add metadata as URL parameters
                import urllib.parse
                
                # Convert metadata to URL parameters
                params = {}
                for key, value in call_metadata.items():
                    if value is not None:
                        params[key] = str(value)
                
                if params:
                    query_string = urllib.parse.urlencode(params)
                    separator = "&" if "?" in webhook_url else "?"
                    full_webhook_url = f"{webhook_url}{separator}{query_string}"
            
            # Initiate the outbound call
            call = client.calls.create(
                to=to_phone_number,
                from_=from_phone_number,
                url=full_webhook_url,
                method='POST'
            )
            
            logger.info(f"Successfully initiated outbound call to {to_phone_number}, call SID: {call.sid}")
            return call.sid
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error initiating outbound call: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error initiating outbound call: {e}")
            return None

    @staticmethod
    def validate_phone_number(phone_number: str) -> bool:
        """
        Validate if a phone number is in E.164 format.
        
        Args:
            phone_number: The phone number to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        import re
        
        # E.164 format: +[country code][subscriber number]
        # Length should be between 7 and 15 digits (excluding the +)
        pattern = r'^\+[1-9]\d{1,14}$'
        
        if not phone_number:
            return False
            
        return bool(re.match(pattern, phone_number)) 