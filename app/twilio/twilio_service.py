import os
import logging
from typing import Optional, List, Dict, Any
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