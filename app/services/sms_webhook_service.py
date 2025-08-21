"""
SMS Webhook Service for forwarding SMS events to assistant webhook URLs.

This service normalizes SMS data from both Twilio and Telnyx providers and sends
a standardized webhook payload to the assistant's sms_webhook_url.

Example webhook payload sent to assistant endpoints:
{
    "type": "sms_received",
    "timestamp": "2024-01-15T10:30:00.123456",
    "data": {
        "message_id": "SM123abc...",
        "from": "+1234567890",
        "to": "+1987654321", 
        "body": "Hello, this is a test message",
        "media_urls": ["https://example.com/image.jpg"],
        "provider": "twilio"
    }
}
"""

import logging
import httpx
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SMSWebhookService:
    """
    Service for handling SMS webhook forwarding to assistant endpoints.
    """

    @staticmethod
    async def send_sms_webhook(
        assistant_sms_webhook_url: str,
        sms_data: Dict[str, Any],
        provider: str = "twilio"
    ) -> bool:
        """
        Send SMS webhook data to the assistant's SMS webhook URL.
        
        Args:
            assistant_sms_webhook_url: The assistant's SMS webhook URL
            sms_data: The standardized SMS data to forward
            provider: The telephony provider (twilio/telnyx)
            
        Returns:
            bool: True if webhook was sent successfully, False otherwise
        """
        try:
            logger.info(f"Sending SMS webhook to {assistant_sms_webhook_url}")
            
            # Prepare the webhook payload with standardized format
            webhook_payload = {
                "type": "sms_received",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "message_id": sms_data.get("message_id"),
                    "from": sms_data.get("from"),
                    "to": sms_data.get("to"), 
                    "body": sms_data.get("body"),
                    "media_urls": sms_data.get("media_urls", []),
                    "provider": provider
                }
            }
            
            # Send the webhook with timeout
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    assistant_sms_webhook_url,
                    json=webhook_payload,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "Burki-SMS-Webhook/1.0"
                    }
                )
                
                if response.status_code >= 200 and response.status_code < 300:
                    logger.info(f"Successfully sent SMS webhook to {assistant_sms_webhook_url}, status: {response.status_code}")
                    return True
                else:
                    logger.warning(f"SMS webhook returned non-success status {response.status_code}: {response.text}")
                    return False
                    
        except httpx.TimeoutException:
            logger.error(f"Timeout sending SMS webhook to {assistant_sms_webhook_url}")
            return False
        except httpx.RequestError as e:
            logger.error(f"Request error sending SMS webhook to {assistant_sms_webhook_url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending SMS webhook to {assistant_sms_webhook_url}: {e}")
            return False

    @staticmethod
    def normalize_twilio_sms_data(form_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Twilio SMS webhook data to a standard format.
        
        Args:
            form_data: Raw form data from Twilio webhook
            
        Returns:
            Dict[str, Any]: Normalized SMS data with essential fields only
        """
        # Extract media URLs if any
        num_media = int(form_data.get("NumMedia", 0))
        media_urls = []
        if num_media > 0:
            media_urls = [
                form_data.get(f"MediaUrl{i}")
                for i in range(num_media)
                if form_data.get(f"MediaUrl{i}")
            ]
        
        return {
            "message_id": form_data.get("MessageSid"),
            "from": form_data.get("From"),
            "to": form_data.get("To"),
            "body": form_data.get("Body", ""),
            "media_urls": media_urls
        }

    @staticmethod
    def normalize_telnyx_sms_data(webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Telnyx SMS webhook data to a standard format.
        
        Args:
            webhook_data: Raw webhook data from Telnyx
            
        Returns:
            Dict[str, Any]: Normalized SMS data with essential fields only
        """
        payload = webhook_data.get("data", {}).get("payload", {})
        
        # Extract media URLs from Telnyx media objects
        media_urls = []
        media_list = payload.get("media", [])
        if media_list:
            media_urls = [media.get("url") for media in media_list if media.get("url")]
        
        return {
            "message_id": payload.get("id"),
            "from": payload.get("from", {}).get("phone_number"),
            "to": payload.get("to", [{}])[0].get("phone_number") if payload.get("to") else None,
            "body": payload.get("text", ""),
            "media_urls": media_urls
        }

    @staticmethod
    async def process_sms_webhook_async(
        assistant_sms_webhook_url: str,
        sms_data: Dict[str, Any],
        provider: str
    ) -> None:
        """
        Process SMS webhook asynchronously (fire-and-forget).
        
        Args:
            assistant_sms_webhook_url: The assistant's SMS webhook URL
            sms_data: The standardized SMS data to forward
            provider: The telephony provider
        """
        try:
            await SMSWebhookService.send_sms_webhook(
                assistant_sms_webhook_url=assistant_sms_webhook_url,
                sms_data=sms_data,
                provider=provider
            )
        except Exception as e:
            logger.error(f"Error in async SMS webhook processing: {e}")
