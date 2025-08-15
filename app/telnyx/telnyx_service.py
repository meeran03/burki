# pylint: disable=all
import os
import logging
import requests
from typing import Optional, List, Dict, Any, Tuple
import telnyx
from telnyx.error import TelnyxError

logger = logging.getLogger(__name__)

class TelnyxService:
    """
    Service for interacting with the Telnyx API to manage phone numbers, calls, and webhooks.
    Compatible with Twilio workflows for easy migration.
    """
    
    @staticmethod
    def get_telnyx_client(api_key: Optional[str] = None) -> bool:
        """
        Initialize Telnyx client with API key.
        
        Args:
            api_key: Optional Telnyx API key (uses env var if not provided)
            
        Returns:
            bool: True if client was initialized successfully, False otherwise
        """
        # Use provided API key or fall back to environment variable
        api_key = api_key or os.getenv("TELNYX_API_KEY")
        
        if not api_key:
            logger.warning("Missing Telnyx API key. Cannot initialize Telnyx client.")
            return False
        
        # Set the global API key for Telnyx
        telnyx.api_key = api_key
        return True
    
    @staticmethod
    def get_available_phone_numbers(
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all available phone numbers from the Telnyx account.
        
        Args:
            api_key: Optional Telnyx API key
            
        Returns:
            List[Dict[str, Any]]: List of phone numbers with their details
        """
        if not TelnyxService.get_telnyx_client(api_key):
            return []
        
        try:
            # Get all phone numbers in the account
            phone_numbers = telnyx.PhoneNumber.list()
            
            # Format the response to match Twilio structure
            formatted_numbers = []
            for phone in phone_numbers.get('data', []):
                formatted_numbers.append({
                    "sid": phone.get("id"),  # Telnyx uses 'id' instead of 'sid'
                    "phone_number": phone.get("phone_number"),
                    "friendly_name": phone.get("phone_number"),  # Telnyx doesn't have friendly_name
                    "capabilities": {
                        "voice": True,  # Assume voice capability
                        "sms": True,   # Assume SMS capability
                        "mms": True    # Assume MMS capability
                    },
                })
            
            return formatted_numbers
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error fetching phone numbers: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching phone numbers: {e}")
            return []
    
    @staticmethod
    def update_phone_webhook(
        phone_number: str,
        webhook_url: str,
        api_key: Optional[str] = None
    ) -> bool:
        """
        Update the voice webhook URL for a phone number.
        This is a convenience method that calls update_phone_webhooks for voice only.
        
        Args:
            phone_number: The phone number to update (E.164 format)
            webhook_url: The voice webhook URL to set
            api_key: Optional Telnyx API key
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Delegate to the full webhook update method for voice only
        results = TelnyxService.update_phone_webhooks(
            phone_number=phone_number,
            voice_webhook_url=webhook_url,
            sms_webhook_url=None,
            api_key=api_key,
            fallback_connection_id=None  # No fallback for this method
        )
        
        return results.get("voice", False)

    @staticmethod
    def update_phone_webhooks(
        phone_number: str,
        voice_webhook_url: Optional[str] = None,
        sms_webhook_url: Optional[str] = None,
        api_key: Optional[str] = None,
        fallback_connection_id: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Update voice and/or SMS webhook URLs for a phone number.
        Note: In Telnyx, webhooks are configured at the Connection/Application level.
        
        Args:
            phone_number: The phone number to update (E.164 format)
            voice_webhook_url: The voice webhook URL to set (optional)
            sms_webhook_url: The SMS webhook URL to set (optional)
            api_key: Optional Telnyx API key
            fallback_connection_id: Optional connection ID to use if phone number has none
            
        Returns:
            Dict[str, bool]: Success status for each webhook type updated
        """
        if not TelnyxService.get_telnyx_client(api_key):
            return {"voice": False, "sms": False}
        
        try:
            # Get phone number details
            phone_numbers = telnyx.PhoneNumber.list(filter={'phone_number': phone_number})
            
            if not phone_numbers.get('data'):
                logger.error(f"No phone number found matching {phone_number}")
                return {"voice": False, "sms": False}
            
            phone_number_obj = phone_numbers['data'][0]
            connection_id = phone_number_obj.get('connection_id')
            
            results = {"voice": False, "sms": False}
            
            if not connection_id:
                logger.warning(f"No connection ID found for {phone_number}. This phone number may not be properly configured for Call Control.")
                logger.info(f"Attempting to create a Call Control Application for {phone_number}")
                
                # Try to create a Call Control Application if none exists
                # First, check if we have a default connection ID from environment or organization
                default_connection_id = os.getenv('TELNYX_CONNECTION_ID')
                
                # If we have a fallback connection ID parameter, use that
                # This would come from the organization's telnyx_connection_id
                if not default_connection_id and fallback_connection_id:
                    default_connection_id = fallback_connection_id
                
                if default_connection_id:
                    logger.info(f"Using default connection ID {default_connection_id} for {phone_number}")
                    connection_id = default_connection_id
                    # Note: We'll assign the phone number to the Call Control Application later
                    # since Call Control Applications ARE the connections in Telnyx
                else:
                    logger.error(f"No connection ID available for {phone_number}. Please configure a Telnyx Connection ID.")
                    return results
            
            # Update Call Control Application for voice webhooks
            if voice_webhook_url is not None:
                try:
                    # Get the Call Control Application
                    applications = telnyx.CallControlApplication.list()
                    
                    # Find the application specifically for this phone number
                    target_app = None
                    expected_app_name = f"Diwaar-{phone_number.replace('+', '')}"
                    
                    for app in applications.get('data', []):
                        # Look for application named for this specific phone number
                        if app.get('application_name') == expected_app_name:
                            target_app = app
                            break
                    
                    if target_app:
                        # Update the application's webhook URL
                        telnyx.CallControlApplication.modify(
                            target_app['id'],
                            webhook_event_url=voice_webhook_url,
                            webhook_event_failover_url=voice_webhook_url  # Use same URL as failover
                        )
                        
                        # Ensure the phone number is properly assigned to this Call Control Application
                        phone_number_id = phone_number_obj.get('id')
                        app_id = target_app['id']
                        
                        if phone_number_id:
                            try:
                                messaging_api_key = api_key or os.getenv('TELNYX_API_KEY')
                                
                                # Assign phone number to use this Call Control Application as its connection
                                # In Telnyx, Call Control Applications ARE the connections for phone numbers
                                connection_url = f"https://api.telnyx.com/v2/phone_numbers/{phone_number_id}"
                                connection_headers = {
                                    'Authorization': f'Bearer {messaging_api_key}',
                                    'Content-Type': 'application/json'
                                }
                                connection_payload = {
                                    'connection_id': app_id  # Use Call Control Application ID as connection ID
                                }
                                
                                conn_response = requests.patch(connection_url, json=connection_payload, headers=connection_headers)
                                if conn_response.status_code in [200, 201, 202]:
                                    logger.info(f"Successfully assigned {phone_number} to Call Control Application {app_id}")
                                    results["voice"] = True
                                    logger.info(f"Updated voice webhook and assigned {phone_number} to Call Control Application")
                                else:
                                    logger.error(f"Failed to assign {phone_number} to Call Control Application: {conn_response.status_code} - {conn_response.text}")
                                    results["voice"] = False
                                    
                            except Exception as assign_error:
                                logger.error(f"Error configuring phone number voice settings: {assign_error}")
                                results["voice"] = False
                        else:
                            logger.error(f"Cannot configure phone number - missing phone_number_id")
                            results["voice"] = False
                    else:
                        # Create a new Call Control Application for this specific phone number
                        app_name = f"Diwaar-{phone_number.replace('+', '')}"
                        
                        try:
                            # Create Call Control Application linked to the connection
                            # This allows the connection to route calls to this specific application
                            new_app = telnyx.CallControlApplication.create(
                                application_name=app_name,
                                webhook_event_url=voice_webhook_url,
                                webhook_event_failover_url=voice_webhook_url
                            )
                            
                            # Handle different response structures from Telnyx API
                            if hasattr(new_app, 'data') and hasattr(new_app.data, 'id'):
                                # Response object with .data attribute
                                app_id = new_app.data.id
                            elif isinstance(new_app, dict) and 'data' in new_app:
                                # Dictionary response with 'data' key
                                app_id = new_app['data']['id']
                            elif isinstance(new_app, dict) and 'id' in new_app:
                                # Direct dictionary response
                                app_id = new_app['id']
                            else:
                                # Fallback - log the structure and continue
                                logger.warning(f"Unexpected Call Control Application response structure: {type(new_app)} - {new_app}")
                                app_id = "unknown"
                            
                            # Now assign the phone number to use this specific Call Control Application
                            # This is the key step that makes the phone number route to this application
                            
                            phone_number_id = phone_number_obj.get('id')
                            if phone_number_id and app_id != "unknown":
                                try:
                                    # Assign phone number to use this Call Control Application as its connection
                                    # In Telnyx, Call Control Applications ARE the connections for phone numbers
                                    messaging_api_key = api_key or os.getenv('TELNYX_API_KEY')
                                    
                                    connection_url = f"https://api.telnyx.com/v2/phone_numbers/{phone_number_id}"
                                    connection_headers = {
                                        'Authorization': f'Bearer {messaging_api_key}',
                                        'Content-Type': 'application/json'
                                    }
                                    connection_payload = {
                                        'connection_id': app_id  # Use Call Control Application ID as connection ID
                                    }
                                    
                                    conn_response = requests.patch(connection_url, json=connection_payload, headers=connection_headers)
                                    if conn_response.status_code in [200, 201, 202]:
                                        logger.info(f"Successfully assigned {phone_number} to Call Control Application {app_id}")
                                        results["voice"] = True
                                        logger.info(f"Created and assigned Call Control Application '{app_name}' for {phone_number}")
                                    else:
                                        logger.error(f"Failed to assign {phone_number} to Call Control Application: {conn_response.status_code} - {conn_response.text}")
                                        results["voice"] = False
                                        
                                except Exception as assign_error:
                                    logger.error(f"Error configuring phone number voice settings: {assign_error}")
                                    results["voice"] = False
                            else:
                                logger.error(f"Cannot configure phone number - missing phone_number_id ({phone_number_id}) or app_id ({app_id})")
                                results["voice"] = False
                            
                        except TelnyxError as create_error:
                            if "already in use" in str(create_error):
                                # This shouldn't happen since we checked above, but handle gracefully
                                logger.warning(f"Application '{app_name}' already exists but wasn't found in list. Searching again...")
                                # Refresh the applications list and try to find it
                                applications = telnyx.CallControlApplication.list()
                                for app in applications.get('data', []):
                                    if app.get('application_name') == app_name:
                                        # Found it, update instead
                                        telnyx.CallControlApplication.modify(
                                            app['id'],
                                            webhook_event_url=voice_webhook_url,
                                            webhook_event_failover_url=voice_webhook_url
                                        )
                                        results["voice"] = True
                                        logger.info(f"Updated existing Call Control Application '{app_name}' for {phone_number}")
                                        break
                                else:
                                    logger.error(f"Application '{app_name}' exists but couldn't be found for update")
                            else:
                                # Different error, re-raise
                                raise create_error
                        
                except TelnyxError as e:
                    logger.error(f"Error updating voice webhook for {phone_number}: {e}")
            
            # Update Messaging Profile for SMS webhooks
            if sms_webhook_url is not None:
                try:
                    # Get messaging profiles
                    profiles = telnyx.MessagingProfile.list()
                    
                    # Find the messaging profile specifically for this phone number
                    target_profile = None
                    expected_profile_name = f"Diwaar-SMS-{phone_number.replace('+', '')}"
                    
                    for profile in profiles.get('data', []):
                        if profile.get('name') == expected_profile_name:
                            target_profile = profile
                            break
                    
                    if target_profile:
                        # Update existing messaging profile
                        telnyx.MessagingProfile.modify(
                            target_profile['id'],
                            webhook_url=sms_webhook_url,
                            webhook_failover_url=sms_webhook_url
                        )
                        results["sms"] = True
                        logger.info(f"Updated SMS webhook for {phone_number} to {sms_webhook_url}")
                    else:
                        # Create new messaging profile for this specific phone number
                        profile_name = f"Diwaar-SMS-{phone_number.replace('+', '')}"
                        
                        # Get whitelisted destinations from environment or use defaults
                        default_destinations = ["US", "CA", "GB", "AU", "DE", "FR", "IT", "ES", "NL", "BR", "MX"]
                        whitelisted_destinations = os.getenv('TELNYX_SMS_DESTINATIONS', ','.join(default_destinations)).split(',')
                        whitelisted_destinations = [dest.strip() for dest in whitelisted_destinations if dest.strip()]
                        
                        try:
                            new_profile = telnyx.MessagingProfile.create(
                                name=profile_name,
                                webhook_url=sms_webhook_url,
                                webhook_failover_url=sms_webhook_url,
                                whitelisted_destinations=whitelisted_destinations
                            )
                            logger.info(f"Created new messaging profile '{profile_name}' for {phone_number}")
                            
                        except TelnyxError as create_error:
                            if "already in use" in str(create_error) or "already exists" in str(create_error):
                                # This shouldn't happen since we checked above, but handle gracefully
                                logger.warning(f"Messaging profile '{profile_name}' already exists but wasn't found in list. Searching again...")
                                # Refresh the profiles list and try to find it
                                profiles = telnyx.MessagingProfile.list()
                                for profile in profiles.get('data', []):
                                    if profile.get('name') == profile_name:
                                        # Found it, update instead
                                        telnyx.MessagingProfile.modify(
                                            profile['id'],
                                            webhook_url=sms_webhook_url,
                                            webhook_failover_url=sms_webhook_url
                                        )
                                        new_profile = profile
                                        logger.info(f"Updated existing messaging profile '{profile_name}' for {phone_number}")
                                        break
                                else:
                                    logger.error(f"Messaging profile '{profile_name}' exists but couldn't be found for update")
                                    new_profile = None
                            else:
                                # Different error, re-raise
                                raise create_error
                        
                        if new_profile:
                            # Assign phone number to the new messaging profile
                            phone_number_id = phone_number_obj.get('id')
                            
                            # Handle different response structures from Telnyx API
                            if hasattr(new_profile, 'data') and hasattr(new_profile.data, 'id'):
                                # Response object with .data attribute
                                profile_id = new_profile.data.id
                            elif isinstance(new_profile, dict) and 'data' in new_profile:
                                # Dictionary response with 'data' key
                                profile_id = new_profile['data']['id']
                            elif isinstance(new_profile, dict) and 'id' in new_profile:
                                # Direct dictionary response
                                profile_id = new_profile['id']
                            else:
                                # Fallback - log the structure and raise error
                                logger.error(f"Unexpected messaging profile response structure: {type(new_profile)} - {new_profile}")
                                raise ValueError(f"Unable to extract profile ID from response: {new_profile}")
                            
                            # Use the correct Telnyx API endpoint for messaging settings
                            # According to the error, we need to use the messaging settings endpoint
                            messaging_api_key = api_key or os.getenv('TELNYX_API_KEY')
                            
                            messaging_url = f"https://api.telnyx.com/v2/phone_numbers/{phone_number_id}/messaging"
                            headers = {
                                'Authorization': f'Bearer {messaging_api_key}',
                                'Content-Type': 'application/json'
                            }
                            payload = {
                                'messaging_profile_id': profile_id
                            }
                            
                            response = requests.patch(messaging_url, json=payload, headers=headers)
                            if response.status_code not in [200, 201, 202]:
                                logger.error(f"Failed to assign messaging profile: {response.status_code} - {response.text}")
                                raise TelnyxError(f"Failed to assign messaging profile: {response.text}")
                            
                            logger.info(f"Successfully assigned phone number {phone_number} to messaging profile {profile_id}")
                            results["sms"] = True
                        else:
                            logger.error(f"Failed to create messaging profile for {phone_number} - all name attempts failed")
                        
                except TelnyxError as e:
                    logger.error(f"Error updating SMS webhook for {phone_number}: {e}")
            
            return results
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error updating phone webhooks: {e}")
            return {"voice": False, "sms": False}
        except Exception as e:
            logger.error(f"Unexpected error updating phone webhooks: {e}")
            return {"voice": False, "sms": False}

    @staticmethod
    def get_phone_webhooks(
        phone_number: str,
        api_key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get current webhook configuration for a phone number.
        
        Args:
            phone_number: The phone number to query (E.164 format)
            api_key: Optional Telnyx API key
            
        Returns:
            Optional[Dict[str, Any]]: Webhook configuration if successful, None otherwise
        """
        if not TelnyxService.get_telnyx_client(api_key):
            return None
        
        try:
            # Get phone number details
            phone_numbers = telnyx.PhoneNumber.list(filter={'phone_number': phone_number})
            
            if not phone_numbers.get('data'):
                logger.error(f"No phone number found matching {phone_number}")
                return None
            
            phone_number_obj = phone_numbers['data'][0]
            connection_id = phone_number_obj.get('connection_id')
            messaging_profile_id = phone_number_obj.get('messaging_profile_id')
            
            webhook_config = {
                "voice_webhook_url": None,
                "sms_webhook_url": None,
                "connection_id": connection_id,
                "messaging_profile_id": messaging_profile_id,
                "phone_number_id": phone_number_obj.get('id'),
                "status": phone_number_obj.get('status'),
                "features": phone_number_obj.get('features', [])
            }
            
            # Get voice webhook from Call Control Application
            if connection_id:
                try:
                    applications = telnyx.CallControlApplication.list()
                    for app in applications.get('data', []):
                        if app.get('connection_id') == connection_id:
                            webhook_config["voice_webhook_url"] = app.get('webhook_event_url')
                            webhook_config["voice_webhook_failover_url"] = app.get('webhook_event_failover_url')
                            break
                except TelnyxError as e:
                    logger.warning(f"Could not retrieve Call Control Application details: {e}")
            
            # Get SMS webhook from Messaging Profile
            if messaging_profile_id:
                try:
                    profile = telnyx.MessagingProfile.retrieve(messaging_profile_id)
                    if profile.get('data'):
                        profile_data = profile['data']
                        webhook_config["sms_webhook_url"] = profile_data.get('webhook_url')
                        webhook_config["sms_webhook_failover_url"] = profile_data.get('webhook_failover_url')
                except TelnyxError as e:
                    logger.warning(f"Could not retrieve Messaging Profile details: {e}")
            
            return webhook_config
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error getting phone webhooks: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting phone webhooks: {e}")
            return None
    
    @staticmethod
    def diagnose_phone_number_connection(
        phone_number: str,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Diagnose the connection status of a phone number in Telnyx.
        
        Args:
            phone_number: The phone number to diagnose (E.164 format)
            api_key: Optional Telnyx API key
            
        Returns:
            Dict[str, Any]: Diagnostic information about the phone number
        """
        if not TelnyxService.get_telnyx_client(api_key):
            return {"error": "Could not initialize Telnyx client"}
        
        try:
            # Get phone number details
            phone_numbers = telnyx.PhoneNumber.list(filter={'phone_number': phone_number})
            
            if not phone_numbers.get('data'):
                return {
                    "phone_number": phone_number,
                    "found": False,
                    "error": "Phone number not found in Telnyx account"
                }
            
            phone_data = phone_numbers['data'][0]
            connection_id = phone_data.get('connection_id')
            
            # Get connection details if connection_id exists
            connection_info = None
            if connection_id:
                try:
                    connections = telnyx.Connection.list()
                    for conn in connections.get('data', []):
                        if conn.get('id') == connection_id:
                            connection_info = {
                                "id": conn.get('id'),
                                "name": conn.get('connection_name'),
                                "type": conn.get('connection_type'),
                                "active": conn.get('active'),
                                "webhook_event_url": conn.get('webhook_event_url')
                            }
                            break
                except Exception as e:
                    logger.warning(f"Could not fetch connection details: {e}")
            
            # Get Call Control Applications
            call_control_apps = []
            try:
                applications = telnyx.CallControlApplication.list()
                for app in applications.get('data', []):
                    if app.get('connection_id') == connection_id:
                        call_control_apps.append({
                            "id": app.get('id'),
                            "name": app.get('application_name'),
                            "webhook_event_url": app.get('webhook_event_url'),
                            "connection_id": app.get('connection_id')
                        })
            except Exception as e:
                logger.warning(f"Could not fetch Call Control Applications: {e}")
            
            # Get the phone number's voice configuration
            voice_config = None
            try:
                phone_number_id = phone_data.get('id')
                if phone_number_id:
                    voice_response = requests.get(
                        f"https://api.telnyx.com/v2/phone_numbers/{phone_number_id}/voice",
                        headers={'Authorization': f'Bearer {api_key or os.getenv("TELNYX_API_KEY")}'}
                    )
                    if voice_response.status_code == 200:
                        voice_config = voice_response.json().get('data', {})
            except Exception as e:
                logger.warning(f"Could not fetch voice configuration: {e}")

            return {
                "phone_number": phone_number,
                "found": True,
                "phone_number_id": phone_data.get('id'),
                "connection_id": connection_id,
                "status": phone_data.get('status'),
                "features": phone_data.get('features', []),
                "messaging_profile_id": phone_data.get('messaging_profile_id'),
                "connection_info": connection_info,
                "call_control_applications": call_control_apps,
                "voice_configuration": voice_config,
                "assigned_call_control_app_id": voice_config.get('call_control_application_id') if voice_config else None,
                "raw_phone_data": phone_data
            }
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error diagnosing phone number: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error diagnosing phone number: {e}")
            return {"error": str(e)}

    @staticmethod
    def get_phone_number_info(
        phone_number: str,
        api_key: Optional[str] = None
    ) -> Optional[dict]:
        """
        Get information about a phone number.
        
        Args:
            phone_number: The phone number to query (E.164 format)
            api_key: Optional Telnyx API key
            
        Returns:
            Optional[dict]: Phone number information or None if not found
        """
        if not TelnyxService.get_telnyx_client(api_key):
            return None
        
        try:
            phone_numbers = telnyx.PhoneNumber.list(filter={'phone_number': phone_number})
            
            if not phone_numbers.get('data'):
                logger.warning(f"No phone number found matching {phone_number}")
                return None
            
            # Return info about the first matching phone number
            phone = phone_numbers['data'][0]
            return {
                "sid": phone.get("id"),
                "phone_number": phone.get("phone_number"),
                "friendly_name": phone.get("phone_number"),
                "voice_url": None,  # Telnyx uses Call Control Applications
                "voice_method": "POST",
                "capabilities": {
                    "voice": True,
                    "sms": True,
                    "mms": True
                },
                "connection_id": phone.get("connection_id"),
            }
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error fetching phone info: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching phone info: {e}")
            return None 

    @staticmethod
    def end_call(
        call_control_id: str,
        api_key: Optional[str] = None
    ) -> bool:
        """
        End a Telnyx call via the Call Control API.
        
        Args:
            call_control_id: The Telnyx Call Control ID (equivalent to Twilio Call SID)
            api_key: Optional Telnyx API key
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not TelnyxService.get_telnyx_client(api_key):
            logger.error("Could not initialize Telnyx client to end call")
            return False
        
        try:
            # End the call using Telnyx Call Control API
            # The Telnyx API expects the hangup method to be called on a Call object
            # Let's use the correct method for hanging up a call
            response = telnyx.Call.create_hangup(call_control_id)
            logger.info(f"Successfully ended call {call_control_id} via Telnyx API: {response}")
            return True
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error ending call: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error ending call: {e}")
            return False
    
    @staticmethod
    def transfer_call(
        call_control_id: str,
        destination: str,
        api_key: Optional[str] = None
    ) -> bool:
        """
        Transfer a Telnyx call to another number.
        
        Args:
            call_control_id: The Telnyx Call Control ID to transfer
            destination: The phone number to transfer to (E.164 format)
            api_key: Optional Telnyx API key
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not TelnyxService.get_telnyx_client(api_key):
            logger.error("Could not initialize Telnyx client to transfer call")
            return False
        
        try:
            # Transfer the call using Telnyx Call Control API
            telnyx.Call.transfer(
                call_control_id,
                to=destination
            )
            logger.info(f"Successfully initiated transfer of call {call_control_id} to {destination}")
            return True
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error transferring call: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error transferring call: {e}")
            return False

    @staticmethod
    def start_call_recording(
        call_control_id: str,
        channels: str = "dual",
        recording_format: str = "mp3",
        api_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Start recording a Telnyx call.
        
                    Args:
            call_control_id: The Telnyx Call Control ID to record
            channels: Recording channels (single, dual)
            recording_format: Recording format (mp3, wav)
            api_key: Optional Telnyx API key
            
        Returns:
            Optional[str]: Recording ID if successful, None otherwise
        """
        if not TelnyxService.get_telnyx_client(api_key):
            logger.error("Could not initialize Telnyx client to start recording")
            return None
        
        try:
            # Start recording using Telnyx Call Control API
            recording = telnyx.Call.create_record_start(
                call_control_id,
                channels=channels,
                format=recording_format
            )
            
            recording_id = recording.get('recording_id')
            logger.info(f"Started recording for call {call_control_id}, recording ID: {recording_id}")
            return recording_id
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error starting recording: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error starting recording: {e}")
            return None

    @staticmethod
    def get_recording_info(
        recording_id: str,
        api_key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a recording.
        
        Args:
            recording_id: The Recording ID
            api_key: Optional Telnyx API key
            
        Returns:
            Optional[Dict[str, Any]]: Recording information or None if not found
        """
        if not TelnyxService.get_telnyx_client(api_key):
            logger.error("Could not initialize Telnyx client to fetch recording info")
            return None
        
        try:
            # Get recording information
            recording = telnyx.Recording.retrieve(recording_id)
            
            return {
                "sid": recording.get("id"),
                "call_sid": recording.get("call_leg_id"),  # Telnyx equivalent
                "duration": recording.get("duration_millis", 0) // 1000,  # Convert to seconds
                "date_created": recording.get("created_at"),
                "status": recording.get("status"),
                "uri": recording.get("download_urls", {}).get("mp3"),  # Get download URL
                "channels": recording.get("channels"),
                "source": "telnyx",
            }
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error fetching recording info: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching recording info: {e}")
            return None

    @staticmethod
    def download_recording_content(
        recording_id: str,
        api_key: Optional[str] = None
    ) -> Optional[Tuple[str, bytes]]:
        """
        Download the content of a recording from Telnyx.
        
        Args:
            recording_id: The Recording ID
            api_key: Optional Telnyx API key
            
        Returns:
            Optional[Tuple[str, bytes]]: Recording filename and content if successful, None otherwise
        """
        if not TelnyxService.get_telnyx_client(api_key):
            logger.error("Missing Telnyx API key for downloading recording")
            return None
        
        try:
            # Get recording information first to get download URL
            recording = telnyx.Recording.retrieve(recording_id)
            download_url = recording.get("download_urls", {}).get("mp3")
            
            if not download_url:
                logger.error(f"No download URL available for recording {recording_id}")
                return None
            
            # Download the recording
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            
            # Generate filename
            filename = f"recording_{recording_id}.mp3"
            
            logger.info(f"Successfully downloaded recording {recording_id}")
            return filename, response.content
            
        except requests.RequestException as e:
            logger.error(f"Error downloading recording content: {e}")
            return None
        except TelnyxError as e:
            logger.error(f"Telnyx API error downloading recording: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading recording content: {e}")
            return None

    @staticmethod
    def initiate_outbound_call(
        to_phone_number: str,
        from_phone_number: str,
        connection_id: str,
        webhook_url: Optional[str] = None,
        call_metadata: Optional[Dict[str, Any]] = None,
        stream_url: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Initiate an outbound call through Telnyx.
        
        Args:
            to_phone_number: The phone number to call (E.164 format)
            from_phone_number: The phone number to call from (must be a Telnyx number)
            connection_id: Telnyx Connection ID (Call Control Application)
            webhook_url: Optional webhook URL for call events
            call_metadata: Optional metadata to include with the call
            stream_url: Optional WebSocket URL for real-time media streaming
            api_key: Optional Telnyx API key
            
        Returns:
            Optional[str]: Call Control ID if successful, None otherwise
        """
        if not TelnyxService.get_telnyx_client(api_key):
            logger.error("Could not initialize Telnyx client to initiate outbound call")
            return None
        
        try:
            # Prepare call parameters
            call_params = {
                'to': to_phone_number,
                'from_': from_phone_number,
                'connection_id': connection_id,
            }
            
            # Add webhook URL if provided
            if webhook_url:
                call_params['webhook_url'] = webhook_url
            
            # Add streaming parameters if provided
            if stream_url:
                call_params.update({
                    'stream_url': stream_url,
                    'stream_track': 'both_tracks',
                    'stream_bidirectional_mode': 'rtp'
                })
                logger.info(f"Adding streaming parameters to outbound call: {stream_url}")
            
            # Add custom headers for metadata if provided
            if call_metadata:
                custom_headers = []
                for key, value in call_metadata.items():
                    if value is not None:
                        custom_headers.append({
                            'name': f'X-Custom-{key}',
                            'value': str(value)
                        })
                if custom_headers:
                    call_params['custom_headers'] = custom_headers
            
            # Initiate the outbound call
            call = telnyx.Call.create(**call_params)
            
            call_control_id = call.get('data', {}).get('call_control_id')
            logger.info(f"Successfully initiated outbound call to {to_phone_number}, call control ID: {call_control_id}")
            return call_control_id
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error initiating outbound call: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error initiating outbound call: {e}")
            return None

    @staticmethod
    def search_available_phone_numbers(
        country_code: str = "US",
        area_code: Optional[str] = None,
        contains: Optional[str] = None,
        locality: Optional[str] = None,
        region: Optional[str] = None,
        limit: int = 10,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for available phone numbers for purchase from Telnyx.
        
        Args:
            country_code: Country code (e.g., "US", "GB")
            area_code: Area code to search in
            contains: Pattern the number should contain
            locality: City/locality to search in
            region: State/region to search in
            limit: Maximum number of results
            api_key: Optional Telnyx API key
            
        Returns:
            List[Dict[str, Any]]: List of available phone numbers
        """
        if not TelnyxService.get_telnyx_client(api_key):
            logger.error("Could not initialize Telnyx client to search phone numbers")
            return []
        
        try:
            # Build search filters according to Telnyx API
            search_filters = {
                "country_code": country_code,
                "limit": limit
            }
            
            if area_code:
                search_filters["national_destination_code"] = area_code
            if contains:
                search_filters["phone_number[contains]"] = contains
            if locality:
                search_filters["locality"] = locality
            if region:
                search_filters["administrative_area"] = region
            
            # Add features filter if supported
            search_filters["features"] = ["voice", "sms"]
            
            # Search for available phone numbers
            response = telnyx.AvailablePhoneNumber.list(filter=search_filters)
            
            # Format the response to match Twilio structure
            formatted_numbers = []
            for number in response.get('data', []):
                formatted_numbers.append({
                    "phone_number": number.get("phone_number"),
                    "friendly_name": number.get("phone_number"),  # Telnyx doesn't have friendly names
                    "locality": number.get("locality"),
                    "region": number.get("administrative_area"),
                    "country_code": country_code,
                    "capabilities": {
                        "voice": "voice" in number.get("features", []),
                        "sms": "sms" in number.get("features", []),
                        "mms": "mms" in number.get("features", []),
                        "fax": "fax" in number.get("features", [])
                    },
                    "cost": number.get("cost_information", {}).get("monthly_cost"),
                    "setup_cost": number.get("cost_information", {}).get("setup_cost"),
                    "provider": "telnyx"
                })
            
            logger.info(f"Found {len(formatted_numbers)} available Telnyx phone numbers")
            return formatted_numbers
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error searching phone numbers: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching phone numbers: {e}")
            return []

    @staticmethod
    def purchase_phone_number(
        phone_number: str,
        connection_id: Optional[str] = None,
        messaging_profile_id: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Purchase a phone number from Telnyx.
        
        Args:
            phone_number: Phone number to purchase (e.g., "+1234567890")
            connection_id: Telnyx connection ID for call control
            messaging_profile_id: Messaging profile ID for SMS/MMS
            api_key: Optional Telnyx API key
            
        Returns:
            Optional[Dict[str, Any]]: Purchase details if successful, None otherwise
        """
        if not TelnyxService.get_telnyx_client(api_key):
            logger.error("Could not initialize Telnyx client to purchase phone number")
            return None
        
        try:
            # Create number order
            order_params = {
                "phone_numbers": [{"phone_number": phone_number}]
            }
            
            if connection_id:
                order_params["connection_id"] = connection_id
            if messaging_profile_id:
                order_params["messaging_profile_id"] = messaging_profile_id
            
            # Purchase the phone number
            order = telnyx.NumberOrder.create(**order_params)
            
            # Debug: Log the full order response to understand structure
            logger.info(f"Telnyx order response: {order}")
            
            # Handle different response structures
            if hasattr(order, 'data'):
                order_data = order.data if hasattr(order.data, '__dict__') else order.data
            elif isinstance(order, dict):
                order_data = order.get('data', order)
            else:
                order_data = order
            
            order_id = order_data.get('id') if isinstance(order_data, dict) else getattr(order_data, 'id', None)
            order_status = order_data.get('status') if isinstance(order_data, dict) else getattr(order_data, 'status', None)
            
            logger.info(f"Parsed order data - ID: {order_id}, Status: {order_status}")
            
            # Check if order was successful or pending
            if order_status in ['success', 'pending', 'completed']:
                # The number was purchased successfully or is being processed
                result = {
                    "id": order_id,
                    "phone_number": phone_number,
                    "status": order_status,
                    "connection_id": connection_id,
                    "messaging_profile_id": messaging_profile_id,
                    "date_created": order_data.get('created_at') if isinstance(order_data, dict) else getattr(order_data, 'created_at', None),
                    "provider": "telnyx",
                    "phone_numbers_count": order_data.get('phone_numbers_count', 1) if isinstance(order_data, dict) else getattr(order_data, 'phone_numbers_count', 1),
                    "customer_reference": order_data.get('customer_reference') if isinstance(order_data, dict) else getattr(order_data, 'customer_reference', None)
                }
                
                if order_status == 'success':
                    logger.info(f"Successfully purchased Telnyx phone number {phone_number}, order ID: {order_id}")
                else:
                    logger.info(f"Telnyx phone number order {order_id} is {order_status} for {phone_number}")
                
                return result
            else:
                # Order failed or has unknown status
                logger.warning(f"Telnyx order {order_id} status: {order_status}")
                
                # Still return the order details for debugging, but mark it appropriately
                return {
                    "id": order_id,
                    "phone_number": phone_number,
                    "status": order_status or "unknown",
                    "connection_id": connection_id,
                    "messaging_profile_id": messaging_profile_id,
                    "date_created": order_data.get('created_at') if isinstance(order_data, dict) else getattr(order_data, 'created_at', None),
                    "provider": "telnyx",
                    "message": f"Order created with status: {order_status or 'unknown'}"
                }
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error purchasing phone number: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error purchasing phone number: {e}")
            return None

    @staticmethod
    def release_phone_number(
        phone_number: str,
        api_key: Optional[str] = None
    ) -> bool:
        """
        Release/delete a phone number from Telnyx account.
        
        Args:
            phone_number: Phone number to release
            api_key: Optional Telnyx API key
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not TelnyxService.get_telnyx_client(api_key):
            logger.error("Could not initialize Telnyx client to release phone number")
            return False
        
        try:
            # Find the phone number in the account
            phone_numbers = telnyx.PhoneNumber.list()
            
            phone_number_id = None
            for pn in phone_numbers.get('data', []):
                if pn.get('phone_number') == phone_number:
                    phone_number_id = pn.get('id')
                    break
            
            if not phone_number_id:
                logger.error(f"Phone number {phone_number} not found in Telnyx account")
                return False
            
            # Delete the phone number
            telnyx.PhoneNumber.delete(phone_number_id)
            
            logger.info(f"Successfully released Telnyx phone number {phone_number}")
            return True
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error releasing phone number: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error releasing phone number: {e}")
            return False

    @staticmethod
    def list_country_codes(
        api_key: Optional[str] = None
    ) -> List[str]:
        """
        List available country codes for phone number search.
        
        Args:
            api_key: Optional Telnyx API key
            
        Returns:
            List[str]: List of country codes
        """
        if not TelnyxService.get_telnyx_client(api_key):
            logger.error("Could not initialize Telnyx client to list country codes")
            return []
        
        try:
            # Get available countries from Telnyx
            # Note: This might need to be adjusted based on Telnyx's actual API
            countries_response = telnyx.AvailablePhoneNumber.list(filter={"limit": 1})
            
            # For now, return common country codes (this could be enhanced)
            common_countries = ["US", "CA", "GB", "AU", "DE", "FR", "IT", "ES", "NL", "BE"]
            
            logger.info(f"Returning {len(common_countries)} country codes for Telnyx")
            return common_countries
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error listing country codes: {e}")
            return ["US", "CA", "GB"]  # Fallback to basic countries
        except Exception as e:
            logger.error(f"Unexpected error listing country codes: {e}")
            return ["US", "CA", "GB"]  # Fallback to basic countries

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

    @staticmethod
    def send_sms(
        to_phone_number: str,
        from_phone_number: str,
        message: str,
        media_urls: Optional[List[str]] = None,
        api_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Send an SMS message through Telnyx.
        
        Args:
            to_phone_number: The phone number to send to (E.164 format)
            from_phone_number: The phone number to send from (must be a Telnyx number)
            message: The message content
            media_urls: Optional list of media URLs for MMS
            api_key: Optional Telnyx API key
            
        Returns:
            Optional[str]: Message ID if successful, None otherwise
        """
        if not TelnyxService.get_telnyx_client(api_key):
            logger.error("Could not initialize Telnyx client to send SMS")
            return None
        
        try:
            # Prepare message parameters
            message_params = {
                'to': to_phone_number,
                'from_': from_phone_number,
                'text': message,
            }
            
            # Add media URLs if provided (for MMS)
            if media_urls:
                message_params['media_urls'] = media_urls
            
            # Send the message
            message_obj = telnyx.Message.create(**message_params)
            
            message_id = message_obj.get('id')
            logger.info(f"Successfully sent SMS to {to_phone_number}, message ID: {message_id}")
            return message_id
            
        except TelnyxError as e:
            error_msg = str(e)
            logger.error(f"Telnyx API error sending SMS: {e}")
            
            # Check for 10DLC registration error
            if "40010" in error_msg or "10DLC" in error_msg:
                logger.error(
                    f"10DLC registration required for number {from_phone_number}. "
                    "Please register your brand and campaign in Telnyx Mission Control Portal. "
                    "See: https://developers.telnyx.com/docs/messaging/10dlc"
                )
            
            return None
        except Exception as e:
            logger.error(f"Unexpected error sending SMS: {e}")
            return None

    @staticmethod
    def assign_number_to_campaign(
        phone_number: str,
        messaging_profile_id: str,
        api_key: Optional[str] = None
    ) -> bool:
        """
        Assign a phone number to an existing 10DLC messaging profile/campaign.
        
        Args:
            phone_number: Phone number to assign (E.164 format)
            messaging_profile_id: ID of the 10DLC messaging profile
            api_key: Optional Telnyx API key
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not TelnyxService.get_telnyx_client(api_key):
            logger.error("Could not initialize Telnyx client")
            return False
        
        try:
            # Get phone number details
            phone_numbers = telnyx.PhoneNumber.list(filter={'phone_number': phone_number})
            
            if not phone_numbers.get('data'):
                logger.error(f"Phone number {phone_number} not found")
                return False
            
            phone_number_obj = phone_numbers['data'][0]
            phone_number_id = phone_number_obj.get('id')
            
            # Use the messaging settings endpoint to assign the profile
            messaging_api_key = api_key or os.getenv('TELNYX_API_KEY')
            
            messaging_url = f"https://api.telnyx.com/v2/phone_numbers/{phone_number_id}/messaging"
            headers = {
                'Authorization': f'Bearer {messaging_api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'messaging_profile_id': messaging_profile_id
            }
            
            response = requests.patch(messaging_url, json=payload, headers=headers)
            if response.status_code in [200, 201, 202]:
                logger.info(f"Successfully assigned {phone_number} to messaging profile {messaging_profile_id}")
                return True
            else:
                logger.error(f"Failed to assign messaging profile: {response.status_code} - {response.text}")
                return False
                
        except TelnyxError as e:
            logger.error(f"Telnyx API error assigning messaging profile: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error assigning messaging profile: {e}")
            return False

    @staticmethod
    def get_messaging_profiles(
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all messaging profiles (including 10DLC campaigns).
        
        Args:
            api_key: Optional Telnyx API key
            
        Returns:
            List[Dict[str, Any]]: List of messaging profiles
        """
        if not TelnyxService.get_telnyx_client(api_key):
            return []
        
        try:
            profiles = telnyx.MessagingProfile.list()
            formatted_profiles = []
            
            for profile in profiles.get('data', []):
                profile_info = {
                    "id": profile.get('id'),
                    "name": profile.get('name'),
                    "enabled": profile.get('enabled', False),
                    "webhook_url": profile.get('webhook_url'),
                    "campaign_id": profile.get('campaign_id'),
                    "brand_id": profile.get('brand_id'),
                    "is_10dlc": bool(profile.get('campaign_id')),
                    "created_at": profile.get('created_at'),
                    "updated_at": profile.get('updated_at')
                }
                formatted_profiles.append(profile_info)
            
            return formatted_profiles
            
        except TelnyxError as e:
            logger.error(f"Error getting messaging profiles: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting messaging profiles: {e}")
            return []

    @staticmethod
    def check_10dlc_status(
        phone_number: str,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check 10DLC registration status for a phone number.
        
        Args:
            phone_number: Phone number to check
            api_key: Optional Telnyx API key
            
        Returns:
            Dict[str, Any]: 10DLC status information
        """
        if not TelnyxService.get_telnyx_client(api_key):
            return {"error": "Could not initialize Telnyx client"}
        
        try:
            # Get phone number details
            phone_numbers = telnyx.PhoneNumber.list(filter={'phone_number': phone_number})
            
            if not phone_numbers.get('data'):
                return {"error": f"Phone number {phone_number} not found"}
            
            phone_data = phone_numbers['data'][0]
            messaging_profile_id = phone_data.get('messaging_profile_id')
            
            if not messaging_profile_id:
                return {
                    "phone_number": phone_number,
                    "10dlc_registered": False,
                    "messaging_enabled": False,
                    "message": "No messaging profile assigned. 10DLC registration required."
                }
            
            # Get messaging profile details
            try:
                profile = telnyx.MessagingProfile.retrieve(messaging_profile_id)
                profile_data = profile.get('data', {})
                
                # Check if this is a 10DLC profile
                profile_name = profile_data.get('name', '')
                is_10dlc = '10dlc' in profile_name.lower() or profile_data.get('campaign_id')
                
                return {
                    "phone_number": phone_number,
                    "messaging_profile_id": messaging_profile_id,
                    "profile_name": profile_name,
                    "10dlc_registered": is_10dlc,
                    "messaging_enabled": True,
                    "campaign_id": profile_data.get('campaign_id'),
                    "brand_id": profile_data.get('brand_id')
                }
                
            except TelnyxError as profile_error:
                return {
                    "phone_number": phone_number,
                    "messaging_profile_id": messaging_profile_id,
                    "error": f"Could not retrieve messaging profile: {profile_error}"
                }
            
        except TelnyxError as e:
            logger.error(f"Error checking 10DLC status: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error checking 10DLC status: {e}")
            return {"error": str(e)}

    @staticmethod
    def create_call_control_application(
        application_name: str,
        webhook_event_url: str,
        webhook_event_failover_url: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a Call Control Application for handling webhooks.
        
        Args:
            application_name: Name for the application
            webhook_event_url: Primary webhook URL for call events
            webhook_event_failover_url: Optional failover webhook URL
            api_key: Optional Telnyx API key
            
        Returns:
            Optional[str]: Application ID if successful, None otherwise
        """
        if not TelnyxService.get_telnyx_client(api_key):
            logger.error("Could not initialize Telnyx client to create Call Control Application")
            return None
        
        try:
            # Create Call Control Application
            app_params = {
                'application_name': application_name,
                'webhook_event_url': webhook_event_url,
            }
            
            if webhook_event_failover_url:
                app_params['webhook_event_failover_url'] = webhook_event_failover_url
            
            application = telnyx.CallControlApplication.create(**app_params)
            
            # Handle different response structures from Telnyx API
            if hasattr(application, 'data') and hasattr(application.data, 'id'):
                # Response object with .data attribute
                app_id = application.data.id
            elif isinstance(application, dict) and 'data' in application:
                # Dictionary response with 'data' key
                app_id = application['data']['id']
            elif isinstance(application, dict) and 'id' in application:
                # Direct dictionary response
                app_id = application['id']
            else:
                # Fallback - log the structure and return None
                logger.warning(f"Unexpected Call Control Application response structure: {type(application)} - {application}")
                return None
            
            logger.info(f"Successfully created Call Control Application: {app_id}")
            return app_id
            
        except TelnyxError as e:
            logger.error(f"Telnyx API error creating Call Control Application: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating Call Control Application: {e}")
            return None
