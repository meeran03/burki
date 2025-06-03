import os
import logging
import socket
from typing import Optional

logger = logging.getLogger(__name__)

def get_server_base_url() -> str:
    """
    Get the server's base URL.
    
    If WEBHOOK_BASE_URL is set in the environment, use that.
    Otherwise, try to detect the URL based on the server settings.
    
    Returns:
        str: Base URL for webhooks (without trailing slash)
    """
    # First check if a base URL is explicitly configured
    base_url = os.getenv("WEBHOOK_BASE_URL")
    if base_url:
        # Remove trailing slash if present
        return base_url.rstrip('/')
    
    # Try to construct a URL from host and port
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8000")
    
    # If host is 0.0.0.0, try to get the actual IP address
    if host == "0.0.0.0":
        try:
            # Get the local machine's hostname
            hostname = socket.gethostname()
            # Get the IP address
            host = socket.gethostbyname(hostname)
        except Exception as e:
            logger.warning(f"Error getting local IP address: {e}")
            host = "localhost"  # Fallback
    
    protocol = "https" if os.getenv("USE_HTTPS", "").lower() == "true" else "http"
    
    # Construct the base URL
    if (protocol == "http" and port == "80") or (protocol == "https" and port == "443"):
        return f"{protocol}://{host}"
    else:
        return f"{protocol}://{host}:{port}"
    
def get_twiml_webhook_url() -> str:
    """
    Get the full TwiML webhook URL.
    
    Returns:
        str: Full URL for the TwiML webhook endpoint
    """
    base_url = get_server_base_url()
    return f"{base_url}/twiml"

def get_sms_webhook_url() -> str:
    """
    Get the full SMS webhook URL.
    
    Returns:
        str: Full URL for the SMS webhook endpoint
    """
    base_url = get_server_base_url()
    return f"{base_url}/sms/webhook" 