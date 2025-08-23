"""
Webhook security utilities for validating webhook signatures.
"""

import os
import logging
from typing import Dict
from fastapi import Request, HTTPException

logger = logging.getLogger(__name__)


def validate_twilio_webhook(request: Request, body: bytes) -> bool:
    """
    Validate Twilio webhook signature to ensure the request is authentic.

    Args:
        request: FastAPI Request object
        body: Raw request body as bytes

    Returns:
        bool: True if signature is valid, False otherwise
    """
    try:
        from twilio.request_validator import RequestValidator

        # Get Twilio Auth Token from environment
        twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        if not twilio_auth_token:
            logger.error("TWILIO_AUTH_TOKEN not found in environment variables")
            return False

        # Get the Twilio signature from headers
        twilio_signature = request.headers.get("X-Twilio-Signature", "")
        if not twilio_signature:
            logger.warning("No X-Twilio-Signature header found in request")
            return False

        # Get the full URL (important: must include query parameters if any)
        url = str(request.url)

        # For form data, we need to parse the POST parameters
        # Twilio sends form-encoded data, not JSON
        if body:
            try:
                # Decode the body to string for form parsing
                body_str = body.decode("utf-8")

                # Parse form data manually to maintain order and format
                post_vars = {}
                if body_str:
                    for param in body_str.split("&"):
                        if "=" in param:
                            key, value = param.split("=", 1)
                            # URL decode the key and value
                            import urllib.parse

                            key = urllib.parse.unquote_plus(key)
                            value = urllib.parse.unquote_plus(value)
                            post_vars[key] = value

            except Exception as e:
                logger.error(f"Error parsing form data for Twilio validation: {e}")
                return False
        else:
            post_vars = {}

        # Create RequestValidator and validate
        validator = RequestValidator(twilio_auth_token)
        is_valid = validator.validate(url, post_vars, twilio_signature)

        if not is_valid:
            logger.warning(f"Invalid Twilio signature for URL: {url}")
            logger.debug(
                f"Expected signature validation failed. URL: {url}, Params: {post_vars}"
            )
        else:
            logger.debug(f"Valid Twilio signature for URL: {url}")

        return is_valid

    except ImportError:
        logger.error("Twilio RequestValidator not available. Install twilio library.")
        return False
    except Exception as e:
        logger.error(f"Error validating Twilio webhook signature: {e}")
        return False


def require_twilio_webhook_auth(request: Request, body: bytes) -> None:
    """
    Middleware function that validates Twilio webhook signature and raises HTTPException if invalid.

    Args:
        request: FastAPI Request object
        body: Raw request body as bytes

    Raises:
        HTTPException: If signature validation fails
    """
    # Skip validation in development mode
    if os.getenv("ENVIRONMENT", "production").lower() in [
        "development",
        "dev",
        "local",
    ]:
        logger.debug("Skipping Twilio webhook validation in development mode")
        return

    if not validate_twilio_webhook(request, body):
        logger.error("Twilio webhook signature validation failed")
        raise HTTPException(status_code=403, detail="Invalid webhook signature")


def get_webhook_security_headers() -> Dict[str, str]:
    """
    Get recommended security headers for webhook responses.

    Returns:
        Dict[str, str]: Security headers
    """
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
