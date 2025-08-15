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
    def update_phone_webhooks(
        phone_number: str,
        voice_webhook_url: Optional[str] = None,
        sms_webhook_url: Optional[str] = None,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Update voice and/or SMS webhook URLs for a phone number.
        
        Args:
            phone_number: The phone number to update (E.164 format)
            voice_webhook_url: The voice webhook URL to set (optional)
            sms_webhook_url: The SMS webhook URL to set (optional)
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            Dict[str, bool]: Success status for each webhook type updated
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        results = {}
        
        if not client:
            return {"voice": False, "sms": False}
        
        try:
            # List incoming phone numbers to find the one we want to update
            numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
            
            if not numbers:
                logger.error(f"No phone number found matching {phone_number}")
                return {"voice": False, "sms": False}
            
            # Prepare update parameters
            update_params = {}
            
            if voice_webhook_url is not None:
                update_params.update({
                    'voice_url': voice_webhook_url,
                    'voice_method': 'POST'
                })
                
            if sms_webhook_url is not None:
                update_params.update({
                    'sms_url': sms_webhook_url,
                    'sms_method': 'POST'
                })
            
            if not update_params:
                logger.warning("No webhook URLs provided for update")
                return {"voice": False, "sms": False}
            
            # Update the first matching phone number
            incoming_phone_number = numbers[0]
            incoming_phone_number.update(**update_params)
            
            # Track what was updated
            results["voice"] = voice_webhook_url is not None
            results["sms"] = sms_webhook_url is not None
            
            logger.info(f"Updated webhooks for {phone_number}: {update_params}")
            return results
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error updating phone webhooks: {e}")
            return {"voice": False, "sms": False}
        except Exception as e:
            logger.error(f"Unexpected error updating phone webhooks: {e}")
            return {"voice": False, "sms": False}

    @staticmethod
    def get_phone_webhooks(
        phone_number: str,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get current webhook configuration for a phone number.
        
        Args:
            phone_number: The phone number to query (E.164 format)
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            Optional[Dict[str, Any]]: Webhook configuration if successful, None otherwise
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        
        if not client:
            return None
        
        try:
            # List incoming phone numbers to find the one we want
            numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
            
            if not numbers:
                logger.error(f"No phone number found matching {phone_number}")
                return None
            
            number_obj = numbers[0]
            
            return {
                "voice_webhook_url": number_obj.voice_url,
                "voice_method": number_obj.voice_method,
                "sms_webhook_url": number_obj.sms_url,
                "sms_method": number_obj.sms_method,
                "status_callback_url": number_obj.status_callback,
                "status_callback_method": number_obj.status_callback_method,
                "phone_number_sid": number_obj.sid,
                "friendly_name": number_obj.friendly_name
            }
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error getting phone webhooks: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting phone webhooks: {e}")
            return None
    
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
    def send_sms(
        to_phone_number: str,
        from_phone_number: str,
        message: str,
        media_urls: Optional[List[str]] = None,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Optional[str]:
        """
        Send an SMS message using Twilio.
        
        Args:
            to_phone_number: Recipient phone number in E.164 format
            from_phone_number: Sender phone number (must be a Twilio number)
            message: Text message content
            media_urls: Optional list of media URLs for MMS
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            Optional[str]: Message SID if successful, None otherwise
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        
        if not client:
            logger.error("Could not get Twilio client to send SMS")
            return None
        
        try:
            # Prepare message parameters
            message_params = {
                'to': to_phone_number,
                'from_': from_phone_number,
                'body': message,
            }
            
            # Add media URLs if provided (for MMS)
            if media_urls:
                message_params['media_url'] = media_urls
            
            # Send the message
            message_obj = client.messages.create(**message_params)
            
            logger.info(f"Successfully sent SMS to {to_phone_number}, message SID: {message_obj.sid}")
            return message_obj.sid
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error sending SMS: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error sending SMS: {e}")
            return None

    @staticmethod
    def search_available_phone_numbers(
        country_code: str = "US",
        area_code: Optional[str] = None,
        contains: Optional[str] = None,
        locality: Optional[str] = None,
        region: Optional[str] = None,
        limit: int = 10,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for available phone numbers for purchase.
        
        Args:
            country_code: Country code (e.g., "US", "GB")
            area_code: Area code to search in
            contains: Pattern the number should contain
            locality: City/locality to search in
            region: State/region to search in
            limit: Maximum number of results
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            List[Dict[str, Any]]: List of available phone numbers
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        
        if not client:
            logger.error("Could not get Twilio client to search phone numbers")
            return []
        
        try:
            # Build search parameters
            search_params = {"limit": limit}
            
            if area_code:
                search_params["area_code"] = area_code
            if contains:
                search_params["contains"] = contains
            if locality:
                search_params["in_locality"] = locality
            if region:
                search_params["in_region"] = region
            
            # Search for available phone numbers
            available_numbers = client.available_phone_numbers(country_code).local.list(**search_params)
            
            # Format the response
            formatted_numbers = []
            for number in available_numbers:
                formatted_numbers.append({
                    "phone_number": number.phone_number,
                    "friendly_name": number.friendly_name,
                    "locality": getattr(number, 'locality', None),
                    "region": getattr(number, 'region', None),
                    "country_code": country_code,
                    "capabilities": {
                        "voice": getattr(number.capabilities, 'voice', False),
                        "sms": getattr(number.capabilities, 'sms', False),
                        "mms": getattr(number.capabilities, 'mms', False),
                        "fax": getattr(number.capabilities, 'fax', False)
                    },
                    "provider": "twilio"
                })
            
            logger.info(f"Found {len(formatted_numbers)} available phone numbers")
            return formatted_numbers
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error searching phone numbers: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching phone numbers: {e}")
            return []

    @staticmethod
    def purchase_phone_number(
        phone_number: str,
        voice_url: Optional[str] = None,
        sms_url: Optional[str] = None,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Purchase a phone number from Twilio.
        
        Args:
            phone_number: Phone number to purchase (e.g., "+1234567890")
            voice_url: URL for voice webhooks
            sms_url: URL for SMS webhooks
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            Optional[Dict[str, Any]]: Purchase details if successful, None otherwise
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        
        if not client:
            logger.error("Could not get Twilio client to purchase phone number")
            return None
        
        try:
            # Build purchase parameters
            purchase_params = {"phone_number": phone_number}
            
            if voice_url:
                purchase_params["voice_url"] = voice_url
            if sms_url:
                purchase_params["sms_url"] = sms_url
            
            # Purchase the phone number
            purchased_number = client.incoming_phone_numbers.create(**purchase_params)
            
            result = {
                "sid": purchased_number.sid,
                "phone_number": purchased_number.phone_number,
                "friendly_name": purchased_number.friendly_name,
                "voice_url": purchased_number.voice_url,
                "sms_url": purchased_number.sms_url,
                "capabilities": purchased_number.capabilities,
                "date_created": purchased_number.date_created.isoformat() if purchased_number.date_created else None,
                "provider": "twilio"
            }
            
            logger.info(f"Successfully purchased phone number {phone_number}, SID: {purchased_number.sid}")
            return result
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error purchasing phone number: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error purchasing phone number: {e}")
            return None

    @staticmethod
    def release_phone_number(
        phone_number: str,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> bool:
        """
        Release/delete a phone number from Twilio account.
        
        Args:
            phone_number: Phone number to release
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            bool: True if successful, False otherwise
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        
        if not client:
            logger.error("Could not get Twilio client to release phone number")
            return False
        
        try:
            # Find the phone number SID
            incoming_numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
            
            if not incoming_numbers:
                logger.error(f"Phone number {phone_number} not found in account")
                return False
            
            phone_number_sid = incoming_numbers[0].sid
            
            # Delete the phone number
            client.incoming_phone_numbers(phone_number_sid).delete()
            
            logger.info(f"Successfully released phone number {phone_number}")
            return True
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error releasing phone number: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error releasing phone number: {e}")
            return False

    @staticmethod
    def list_country_codes(
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> List[str]:
        """
        List available country codes for phone number search.
        
        Args:
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            List[str]: List of country codes
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        
        if not client:
            logger.error("Could not get Twilio client to list country codes")
            return []
        
        try:
            countries = client.available_phone_numbers.list()
            country_codes = [country.country_code for country in countries]
            
            logger.info(f"Found {len(country_codes)} available countries")
            return country_codes
            
        except TwilioRestException as e:
            logger.error(f"Twilio API error listing country codes: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing country codes: {e}")
            return []

    @staticmethod
    def get_phone_number_sid(phone_number: str, account_sid: Optional[str] = None, auth_token: Optional[str] = None) -> str:
        """
        Get the phone number SID from Twilio.
        
        Args:
            phone_number: The phone number to look up
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            str: The phone number SID
            
        Raises:
            ValueError: If phone number not found
        """
        client = TwilioService.get_twilio_client(account_sid, auth_token)
        if not client:
            raise ValueError("Could not get Twilio client")
        
        try:
            incoming_numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
            if not incoming_numbers:
                raise ValueError("Phone number not found")
            return incoming_numbers[0].sid
        except TwilioRestException as e:
            logger.error(f"Twilio API error getting phone number SID: {e}")
            raise ValueError("Phone number not found")

    @staticmethod
    async def enable_messaging_feature(
        phone_number: str,
        messaging_service_sid: str,
        sms_webhook_url: str,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enable messaging feature for a phone number.
        
        Args:
            phone_number: The phone number to enable messaging for
            messaging_service_sid: Twilio Messaging Service SID
            sms_webhook_url: SMS webhook URL
            account_sid: Optional Twilio Account SID  
            auth_token: Optional Twilio Auth Token
            
        Returns:
            Dict[str, Any]: Success status and details
        """
        try:
            phone_number_sid = TwilioService.get_phone_number_sid(phone_number, account_sid, auth_token)
            
            # Use provided credentials or fall back to environment variables
            account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
            
            url = f"https://messaging.twilio.com/v1/Services/{messaging_service_sid}/PhoneNumbers"
            auth = (account_sid, auth_token)
            data = {"PhoneNumberSid": phone_number_sid}
            
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=data, auth=auth)
                
                if response.status_code in [200, 201, 409]:
                    # Update SMS webhook URL
                    TwilioService.update_phone_webhooks(
                        phone_number=phone_number,
                        sms_webhook_url=sms_webhook_url,
                        account_sid=account_sid,
                        auth_token=auth_token
                    )
                    
                if response.status_code == 201:
                    return {
                        "detail": "Messaging feature enabled successfully",
                        "status": "pending",
                    }
                elif response.status_code == 200:
                    return {
                        "detail": "Messaging feature already enabled",
                        "status": "registered",
                    }
                elif response.status_code == 409:
                    return {
                        "detail": "Messaging feature already enabled",
                        "status": "registered",
                    }
                else:
                    return {
                        "detail": f"Failed to enable messaging: {response.text}",
                        "status": "error",
                        "status_code": response.status_code
                    }
                    
        except Exception as e:
            logger.error(f"Error enabling messaging feature: {e}")
            return {
                "detail": f"Error enabling messaging: {str(e)}",
                "status": "error",
                "status_code": 500
            }

    @staticmethod
    async def disable_messaging_feature(
        phone_number: str,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Disable messaging feature for a phone number.
        
        Args:
            phone_number: The phone number to disable messaging for
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
            
        Returns:
            Dict[str, Any]: Success status and details
        """
        try:
            # Set SMS URL to Twilio's demo URL to disable messaging
            sms_url = "https://demo.twilio.com/welcome/sms/reply"
            results = TwilioService.update_phone_webhooks(
                phone_number=phone_number,
                sms_webhook_url=sms_url,
                account_sid=account_sid,
                auth_token=auth_token
            )
            
            if results.get("sms", False):
                return {"detail": "Messaging feature disabled successfully"}
            else:
                return {
                    "detail": "Failed to disable messaging feature",
                    "status_code": 500
                }
                
        except Exception as e:
            logger.error(f"Error disabling messaging feature: {e}")
            return {
                "detail": f"Error disabling messaging: {str(e)}",
                "status_code": 500
            }

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