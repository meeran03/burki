"""
Abstraction layer for telephony providers (Twilio, Telnyx).
Provides a unified interface for managing calls, phone numbers, and recordings.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class TelephonyProvider(Enum):
    """Supported telephony providers."""
    TWILIO = "twilio"
    TELNYX = "telnyx"


class BaseTelephonyService(ABC):
    """
    Abstract base class for telephony service providers.
    Defines the interface that all providers must implement.
    """
    
    @abstractmethod
    def get_available_phone_numbers(self) -> List[Dict[str, Any]]:
        """Get all available phone numbers from the provider account."""
        ...
    
    @abstractmethod
    def update_phone_webhook(self, phone_number: str, webhook_url: str) -> bool:
        """Update the webhook URL for a phone number."""
        ...
    
    @abstractmethod
    def update_phone_webhooks(self, phone_number: str, voice_webhook_url: Optional[str] = None, sms_webhook_url: Optional[str] = None) -> Dict[str, bool]:
        """Update both voice and SMS webhook URLs for a phone number."""
        ...
    
    @abstractmethod
    def get_phone_number_info(self, phone_number: str) -> Optional[dict]:
        """Get information about a phone number."""
        ...
    
    @abstractmethod
    def end_call(self, call_id: str) -> bool:
        """End a call."""
        ...
    
    @abstractmethod
    def transfer_call(self, call_id: str, destination: str) -> bool:
        """Transfer a call to another number."""
        ...
    
    @abstractmethod
    def start_call_recording(self, call_id: str, **kwargs) -> Optional[str]:
        """Start recording a call."""
        ...
    
    @abstractmethod
    def get_recording_info(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a recording."""
        ...
    
    @abstractmethod
    def download_recording_content(self, recording_id: str) -> Optional[Tuple[str, bytes]]:
        """Download the content of a recording."""
        ...
    
    @abstractmethod
    def initiate_outbound_call(
        self, 
        to_phone_number: str, 
        from_phone_number: str, 
        webhook_url: str,
        call_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Initiate an outbound call."""
        ...
    
    @abstractmethod
    def validate_phone_number(self, phone_number: str) -> bool:
        """Validate if a phone number is in correct format."""
        ...
    
    @abstractmethod
    def send_sms(
        self, 
        to_phone_number: str, 
        from_phone_number: str, 
        message: str, 
        media_urls: Optional[List[str]] = None
    ) -> Optional[str]:
        """Send an SMS message."""
        ...


class TwilioTelephonyService(BaseTelephonyService):
    """Twilio implementation of the telephony service."""
    
    def __init__(self, account_sid: Optional[str] = None, auth_token: Optional[str] = None):
        """
        Initialize Twilio telephony service.
        
        Args:
            account_sid: Optional Twilio Account SID
            auth_token: Optional Twilio Auth Token
        """
        self.account_sid = account_sid
        self.auth_token = auth_token
    
    def get_available_phone_numbers(self) -> List[Dict[str, Any]]:
        """Get all available phone numbers from Twilio."""
        from app.twilio.twilio_service import TwilioService
        return TwilioService.get_available_phone_numbers(
            account_sid=self.account_sid,
            auth_token=self.auth_token
        )
    
    def update_phone_webhook(self, phone_number: str, webhook_url: str) -> bool:
        """Update the webhook URL for a Twilio phone number."""
        from app.twilio.twilio_service import TwilioService
        return TwilioService.update_phone_webhook(
            phone_number=phone_number,
            webhook_url=webhook_url,
            account_sid=self.account_sid,
            auth_token=self.auth_token
        )
    
    def update_phone_webhooks(self, phone_number: str, voice_webhook_url: Optional[str] = None, sms_webhook_url: Optional[str] = None) -> Dict[str, bool]:
        """Update both voice and SMS webhook URLs for a Twilio phone number."""
        from app.twilio.twilio_service import TwilioService
        return TwilioService.update_phone_webhooks(
            phone_number=phone_number,
            voice_webhook_url=voice_webhook_url,
            sms_webhook_url=sms_webhook_url,
            account_sid=self.account_sid,
            auth_token=self.auth_token
        )
    
    def get_phone_number_info(self, phone_number: str) -> Optional[dict]:
        """Get information about a Twilio phone number."""
        from app.twilio.twilio_service import TwilioService
        return TwilioService.get_phone_number_info(
            phone_number=phone_number,
            account_sid=self.account_sid,
            auth_token=self.auth_token
        )
    
    def end_call(self, call_id: str) -> bool:
        """End a Twilio call (call_id is Call SID)."""
        from app.twilio.twilio_service import TwilioService
        return TwilioService.end_call(
            call_sid=call_id,
            account_sid=self.account_sid,
            auth_token=self.auth_token
        )
    
    def transfer_call(self, call_id: str, destination: str) -> bool:
        """Transfer a Twilio call to another number."""
        from app.twilio.twilio_service import TwilioService
        return TwilioService.transfer_call(
            call_sid=call_id,
            destination=destination,
            account_sid=self.account_sid,
            auth_token=self.auth_token
        )
    
    def start_call_recording(self, call_id: str, **kwargs) -> Optional[str]:
        """Start recording a Twilio call."""
        from app.twilio.twilio_service import TwilioService
        return TwilioService.start_call_recording(
            call_sid=call_id,
            recording_channels=kwargs.get('recording_channels', 'dual'),
            recording_status_callback=kwargs.get('recording_status_callback'),
            account_sid=self.account_sid,
            auth_token=self.auth_token
        )
    
    def get_recording_info(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a Twilio recording."""
        from app.twilio.twilio_service import TwilioService
        return TwilioService.get_recording_info(
            recording_sid=recording_id,
            account_sid=self.account_sid,
            auth_token=self.auth_token
        )
    
    def download_recording_content(self, recording_id: str) -> Optional[Tuple[str, bytes]]:
        """Download the content of a Twilio recording."""
        from app.twilio.twilio_service import TwilioService
        return TwilioService.download_recording_content(
            recording_sid=recording_id,
            account_sid=self.account_sid,
            auth_token=self.auth_token
        )
    
    def initiate_outbound_call(
        self, 
        to_phone_number: str, 
        from_phone_number: str, 
        webhook_url: str,
        call_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Initiate an outbound call through Twilio."""
        from app.twilio.twilio_service import TwilioService
        return TwilioService.initiate_outbound_call(
            to_phone_number=to_phone_number,
            from_phone_number=from_phone_number,
            webhook_url=webhook_url,
            call_metadata=call_metadata,
            account_sid=self.account_sid,
            auth_token=self.auth_token
        )
    
    def validate_phone_number(self, phone_number: str) -> bool:
        """Validate if a phone number is in E.164 format."""
        from app.twilio.twilio_service import TwilioService
        return TwilioService.validate_phone_number(phone_number)
    
    def send_sms(
        self, 
        to_phone_number: str, 
        from_phone_number: str, 
        message: str, 
        media_urls: Optional[List[str]] = None
    ) -> Optional[str]:
        """Send an SMS message using Twilio."""
        from app.twilio.twilio_service import TwilioService
        return TwilioService.send_sms(
            to_phone_number=to_phone_number,
            from_phone_number=from_phone_number,
            message=message,
            media_urls=media_urls,
            account_sid=self.account_sid,
            auth_token=self.auth_token
        )


class TelnyxTelephonyService(BaseTelephonyService):
    """Telnyx implementation of the telephony service."""
    
    def __init__(self, api_key: Optional[str] = None, connection_id: Optional[str] = None):
        """
        Initialize Telnyx telephony service.
        
        Args:
            api_key: Optional Telnyx API key
            connection_id: Optional Telnyx Connection ID (for Call Control Application)
        """
        self.api_key = api_key
        self.connection_id = connection_id
    
    def get_available_phone_numbers(self) -> List[Dict[str, Any]]:
        """Get all available phone numbers from Telnyx."""
        from app.telnyx.telnyx_service import TelnyxService
        return TelnyxService.get_available_phone_numbers(api_key=self.api_key)
    
    def update_phone_webhook(self, phone_number: str, webhook_url: str) -> bool:
        """Update the webhook URL for a Telnyx phone number."""
        from app.telnyx.telnyx_service import TelnyxService
        return TelnyxService.update_phone_webhooks(
            phone_number=phone_number,
            voice_webhook_url=webhook_url,
            sms_webhook_url=None,
            api_key=self.api_key,
            fallback_connection_id=self.connection_id
        ).get("voice", False)
    
    def update_phone_webhooks(self, phone_number: str, voice_webhook_url: Optional[str] = None, sms_webhook_url: Optional[str] = None) -> Dict[str, bool]:
        """Update both voice and SMS webhook URLs for a Telnyx phone number."""
        from app.telnyx.telnyx_service import TelnyxService
        return TelnyxService.update_phone_webhooks(
            phone_number=phone_number,
            voice_webhook_url=voice_webhook_url,
            sms_webhook_url=sms_webhook_url,
            api_key=self.api_key,
            fallback_connection_id=self.connection_id
        )
    
    def get_phone_number_info(self, phone_number: str) -> Optional[dict]:
        """Get information about a Telnyx phone number."""
        from app.telnyx.telnyx_service import TelnyxService
        return TelnyxService.get_phone_number_info(
            phone_number=phone_number,
            api_key=self.api_key
        )
    
    def end_call(self, call_id: str) -> bool:
        """End a Telnyx call (call_id is Call Control ID)."""
        from app.telnyx.telnyx_service import TelnyxService
        return TelnyxService.end_call(
            call_control_id=call_id,
            api_key=self.api_key
        )
    
    def transfer_call(self, call_id: str, destination: str) -> bool:
        """Transfer a Telnyx call to another number."""
        from app.telnyx.telnyx_service import TelnyxService
        return TelnyxService.transfer_call(
            call_control_id=call_id,
            destination=destination,
            api_key=self.api_key
        )
    
    def start_call_recording(self, call_id: str, **kwargs) -> Optional[str]:
        """Start recording a Telnyx call."""
        from app.telnyx.telnyx_service import TelnyxService
        return TelnyxService.start_call_recording(
            call_control_id=call_id,
            channels=kwargs.get('channels', 'dual'),
            recording_format=kwargs.get('recording_format', kwargs.get('format', 'mp3')),
            api_key=self.api_key
        )
    
    def get_recording_info(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a Telnyx recording."""
        from app.telnyx.telnyx_service import TelnyxService
        return TelnyxService.get_recording_info(
            recording_id=recording_id,
            api_key=self.api_key
        )
    
    def download_recording_content(self, recording_id: str) -> Optional[Tuple[str, bytes]]:
        """Download the content of a Telnyx recording."""
        from app.telnyx.telnyx_service import TelnyxService
        return TelnyxService.download_recording_content(
            recording_id=recording_id,
            api_key=self.api_key
        )
    
    def initiate_outbound_call(
        self, 
        to_phone_number: str, 
        from_phone_number: str, 
        webhook_url: str,
        call_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Initiate an outbound call through Telnyx."""
        from app.telnyx.telnyx_service import TelnyxService
        
        # For Telnyx, we need to determine the stream URL for real-time audio
        # This should be passed from the calling code, but we can construct it here as fallback
        stream_url = None
        if call_metadata and call_metadata.get('stream_url'):
            stream_url = call_metadata['stream_url']
        
        return TelnyxService.initiate_outbound_call(
            to_phone_number=to_phone_number,
            from_phone_number=from_phone_number,
            connection_id=self.connection_id,
            webhook_url=webhook_url,
            call_metadata=call_metadata,
            stream_url=stream_url,
            api_key=self.api_key
        )
    
    def validate_phone_number(self, phone_number: str) -> bool:
        """Validate if a phone number is in E.164 format."""
        from app.telnyx.telnyx_service import TelnyxService
        return TelnyxService.validate_phone_number(phone_number)
    
    def send_sms(
        self, 
        to_phone_number: str, 
        from_phone_number: str, 
        message: str, 
        media_urls: Optional[List[str]] = None
    ) -> Optional[str]:
        """Send an SMS message using Telnyx."""
        from app.telnyx.telnyx_service import TelnyxService
        return TelnyxService.send_sms(
            to_phone_number=to_phone_number,
            from_phone_number=from_phone_number,
            message=message,
            media_urls=media_urls,
            api_key=self.api_key
        )


class TelephonyProviderFactory:
    """Factory for creating telephony service instances."""
    
    @staticmethod
    def create_provider(
        provider: TelephonyProvider,
        **kwargs
    ) -> BaseTelephonyService:
        """
        Create a telephony service instance.
        
        Args:
            provider: The telephony provider to create
            **kwargs: Provider-specific configuration
            
        Returns:
            BaseTelephonyService: The telephony service instance
        """
        if provider == TelephonyProvider.TWILIO:
            return TwilioTelephonyService(
                account_sid=kwargs.get('account_sid'),
                auth_token=kwargs.get('auth_token')
            )
        elif provider == TelephonyProvider.TELNYX:
            return TelnyxTelephonyService(
                api_key=kwargs.get('api_key'),
                connection_id=kwargs.get('connection_id')
            )
        else:
            raise ValueError(f"Unsupported telephony provider: {provider}")
    
    @staticmethod
    async def create_from_phone_number(phone_number) -> BaseTelephonyService:
        """
        Create a telephony service instance from phone number's provider and organization credentials.
        
        Args:
            phone_number: Either a PhoneNumber model object or a phone number string
            
        Returns:
            BaseTelephonyService: The telephony service instance
        """
        # If phone_number is a string, look up the PhoneNumber model
        if isinstance(phone_number, str):
            from app.services.phone_number_service import PhoneNumberService
            phone_number_obj = await PhoneNumberService.get_phone_number_by_number(phone_number)
            
            if not phone_number_obj:
                # If phone number not found in database, default to Twilio
                logger.warning(f"Phone number {phone_number} not found in database, defaulting to Twilio")
                return TwilioTelephonyService()
            
            phone_number = phone_number_obj
        
        organization = phone_number.organization
        
        if phone_number.provider == "telnyx":
            return TelnyxTelephonyService(
                api_key=organization.telnyx_api_key,
                connection_id=organization.telnyx_connection_id
            )
        else:  # Default to Twilio
            return TwilioTelephonyService(
                account_sid=organization.twilio_account_sid,
                auth_token=organization.twilio_auth_token
            )
    
    @staticmethod
    def create_from_assistant(assistant: Any) -> BaseTelephonyService:
        """
        DEPRECATED: Create a telephony service instance from assistant configuration.
        Use create_from_phone_number instead for proper provider detection.
        
        This method is kept for backward compatibility but will be removed.
        """
        # Fall back to organization-level configuration
        organization = assistant.organization
        
        # Default to Twilio with organization or environment credentials
        return TwilioTelephonyService(
            account_sid=getattr(organization, 'twilio_account_sid', None),
            auth_token=getattr(organization, 'twilio_auth_token', None)
        )


class UnifiedTelephonyService:
    """
    Unified telephony service that automatically detects and uses the appropriate provider.
    Provides backward compatibility with existing code.
    """
    
    def __init__(self, provider_service: Optional[BaseTelephonyService] = None, assistant: Optional[Any] = None):
        """
        Initialize unified telephony service.
        
        Args:
            provider_service: Pre-initialized provider service
            assistant: Optional assistant object for backward compatibility (deprecated)
        """
        if provider_service:
            self.provider_service = provider_service
        elif assistant:
            # Backward compatibility - use deprecated method
            self.provider_service = TelephonyProviderFactory.create_from_assistant(assistant)
        else:
            # Default to Twilio for backward compatibility
            self.provider_service = TwilioTelephonyService()
    
    @classmethod
    async def create_from_phone_number(cls, phone_number: str) -> 'UnifiedTelephonyService':
        """
        Create a UnifiedTelephonyService instance from a phone number string.
        
        Args:
            phone_number: Phone number string to determine provider from
            
        Returns:
            UnifiedTelephonyService: Configured service instance
        """
        provider_service = await TelephonyProviderFactory.create_from_phone_number(phone_number)
        return cls(provider_service=provider_service)
    
    def get_available_phone_numbers(self) -> List[Dict[str, Any]]:
        """Get all available phone numbers."""
        return self.provider_service.get_available_phone_numbers()
    
    def update_phone_webhook(self, phone_number: str, webhook_url: str) -> bool:
        """Update the webhook URL for a phone number."""
        return self.provider_service.update_phone_webhook(phone_number, webhook_url)
    
    def update_phone_webhooks(self, phone_number: str, voice_webhook_url: Optional[str] = None, sms_webhook_url: Optional[str] = None) -> Dict[str, bool]:
        """Update both voice and SMS webhook URLs for a phone number."""
        return self.provider_service.update_phone_webhooks(phone_number, voice_webhook_url, sms_webhook_url)
    
    def get_phone_number_info(self, phone_number: str) -> Optional[dict]:
        """Get information about a phone number."""
        return self.provider_service.get_phone_number_info(phone_number)
    
    def end_call(self, call_id: str) -> bool:
        """End a call."""
        return self.provider_service.end_call(call_id)
    
    def transfer_call(self, call_id: str, destination: str) -> bool:
        """Transfer a call to another number."""
        return self.provider_service.transfer_call(call_id, destination)
    
    def start_call_recording(self, call_id: str, **kwargs) -> Optional[str]:
        """Start recording a call."""
        return self.provider_service.start_call_recording(call_id, **kwargs)
    
    def get_recording_info(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a recording."""
        return self.provider_service.get_recording_info(recording_id)
    
    def download_recording_content(self, recording_id: str) -> Optional[Tuple[str, bytes]]:
        """Download the content of a recording."""
        return self.provider_service.download_recording_content(recording_id)
    
    def initiate_outbound_call(
        self, 
        to_phone_number: str, 
        from_phone_number: str, 
        webhook_url: str,
        call_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Initiate an outbound call."""
        return self.provider_service.initiate_outbound_call(
            to_phone_number, from_phone_number, webhook_url, call_metadata
        )
    
    def validate_phone_number(self, phone_number: str) -> bool:
        """Validate if a phone number is in correct format."""
        return self.provider_service.validate_phone_number(phone_number)
    
    def send_sms(
        self, 
        to_phone_number: str, 
        from_phone_number: str, 
        message: str, 
        media_urls: Optional[List[str]] = None
    ) -> Optional[str]:
        """Send an SMS message using the configured provider."""
        if not self.provider_service:
            logger.error("No provider service configured for SMS")
            return None
        
        return self.provider_service.send_sms(
            to_phone_number=to_phone_number,
            from_phone_number=from_phone_number,
            message=message,
            media_urls=media_urls
        )
    
    def get_provider_type(self) -> str:
        """Get the type of provider being used."""
        if isinstance(self.provider_service, TwilioTelephonyService):
            return "twilio"
        elif isinstance(self.provider_service, TelnyxTelephonyService):
            return "telnyx"
        else:
            return "unknown"
