"""
This file contains the API endpoints for the Twilio call.
"""

# pylint: disable=logging-format-interpolation,logging-fstring-interpolation,broad-exception-caught
import logging
from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from fastapi import HTTPException, Depends
from fastapi import APIRouter

from app.db.database import get_async_db_session
from app.db.models import User
from app.core.auth import get_current_user_flexible

from app.utils.url_utils import get_twiml_webhook_url, get_telnyx_webhook_url

from app.api.schemas import (
    SearchPhoneNumbersRequest, SearchPhoneNumbersResponse, PurchasePhoneNumberRequest, 
    PurchasePhoneNumberResponse, ReleasePhoneNumberRequest, ReleasePhoneNumberResponse,
    CountryCodesResponse, AvailablePhoneNumber, UpdateWebhookRequest, UpdateWebhookResponse,
    GetWebhookResponse
)


router = APIRouter(prefix="/phone-numbers")
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@router.post("/search", response_model=SearchPhoneNumbersResponse)
async def search_phone_numbers(search_data: SearchPhoneNumbersRequest):
    """
    Search for available phone numbers for purchase.
    
    This endpoint allows you to search for available phone numbers from Twilio or Telnyx
    based on various criteria like area code, location, or number patterns.
    
    Args:
        search_data: Search criteria including country, area code, provider, etc.
        
    Returns:
        SearchPhoneNumbersResponse: List of available numbers with pricing and capabilities
    """
    try:
        provider = search_data.provider.lower()
        
        # Validate provider
        if provider not in ["twilio", "telnyx"]:
            raise HTTPException(status_code=400, detail="Provider must be 'twilio' or 'telnyx'")
        
        # Import the appropriate service
        if provider == "twilio":
            from app.twilio.twilio_service import TwilioService
            available_numbers = TwilioService.search_available_phone_numbers(
                country_code=search_data.country_code,
                area_code=search_data.area_code,
                contains=search_data.contains,
                locality=search_data.locality,
                region=search_data.region,
                limit=search_data.limit
            )
        else:  # telnyx
            from app.telnyx.telnyx_service import TelnyxService
            available_numbers = TelnyxService.search_available_phone_numbers(
                country_code=search_data.country_code,
                area_code=search_data.area_code,
                contains=search_data.contains,
                locality=search_data.locality,
                region=search_data.region,
                limit=search_data.limit
            )
        
        # Convert to response format
        phone_numbers = [AvailablePhoneNumber(**num) for num in available_numbers]
        
        logger.info(f"Found {len(phone_numbers)} available numbers from {provider}")
        
        return SearchPhoneNumbersResponse(
            success=True,
            numbers=phone_numbers,
            total_found=len(phone_numbers),
            provider=provider
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching phone numbers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/purchase", response_model=PurchasePhoneNumberResponse)
async def purchase_phone_number(
    purchase_data: PurchasePhoneNumberRequest,
    current_user: User = Depends(get_current_user_flexible)
):
    """
    Purchase a phone number from Twilio or Telnyx.
    
    This endpoint purchases a phone number and optionally assigns it to an assistant.
    The purchased number will be automatically configured with appropriate webhooks.
    
    Args:
        purchase_data: Purchase details including phone number, provider, and assistant
        
    Returns:
        PurchasePhoneNumberResponse: Purchase confirmation with details
    """
    try:
        provider = purchase_data.provider.lower()
        phone_number = purchase_data.phone_number
        
        # Validate provider
        if provider not in ["twilio", "telnyx"]:
            raise HTTPException(status_code=400, detail="Provider must be 'twilio' or 'telnyx'")
        
        # Validate phone number format
        if not phone_number.startswith('+'):
            raise HTTPException(
                status_code=400, 
                detail="Phone number must be in E.164 format (e.g., +1234567890)"
            )
        
        # Get webhook URLs for the purchase
        voice_webhook_url = get_twiml_webhook_url() if provider == "twilio" else None
        
        # Purchase the phone number
        if provider == "twilio":
            from app.twilio.twilio_service import TwilioService
            
            purchase_result = TwilioService.purchase_phone_number(
                phone_number=phone_number,
                voice_url=voice_webhook_url
            )
        else:  # telnyx
            from app.telnyx.telnyx_service import TelnyxService
            
            purchase_result = TelnyxService.purchase_phone_number(
                phone_number=phone_number
            )
        
        if not purchase_result:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to purchase phone number from {provider}"
            )
        
        # Add the number to our database
        from app.services.phone_number_service import PhoneNumberService
        from app.db.models import PhoneNumber, Organization
        from sqlalchemy import select
        
        async with await get_async_db_session() as db:
            # For now, we'll need an organization context - this could be improved
            # with proper authentication/organization detection
            
            # Check if phone number already exists
            existing_query = select(PhoneNumber).where(
                PhoneNumber.phone_number == phone_number
            )
            existing_result = await db.execute(existing_query)
            existing_phone_number = existing_result.scalar_one_or_none()
            
            if existing_phone_number:
                # Phone number already exists, update it instead of creating new
                logger.info(f"Phone number {phone_number} already exists in database, updating record")
                
                existing_phone_number.provider = provider
                existing_phone_number.provider_phone_id = purchase_result.get("sid") or purchase_result.get("id")
                existing_phone_number.phone_metadata = purchase_result
                existing_phone_number.is_active = True
                
                # Update friendly name if provided
                if purchase_data.friendly_name:
                    existing_phone_number.friendly_name = purchase_data.friendly_name
                
                # Update assistant assignment if provided
                if purchase_data.assistant_id:
                    existing_phone_number.assistant_id = purchase_data.assistant_id
                
                await db.commit()
                await db.refresh(existing_phone_number)
                new_phone_number = existing_phone_number
            else:
                # Create new phone number record
                new_phone_number = PhoneNumber(
                    organization_id=current_user.organization_id,
                    phone_number=phone_number,
                    friendly_name=purchase_data.friendly_name or phone_number,
                    provider=provider,
                    provider_phone_id=purchase_result.get("sid") or purchase_result.get("id"),
                    assistant_id=purchase_data.assistant_id,
                    capabilities={
                        "voice": True,
                        "sms": True,
                        "mms": True
                    },
                    phone_metadata=purchase_result,
                    is_active=True
                )
                
                db.add(new_phone_number)
                await db.commit()
                await db.refresh(new_phone_number)
        
        logger.info(f"Successfully purchased and registered phone number {phone_number} from {provider}")
        
        return PurchasePhoneNumberResponse(
            success=True,
            phone_number=phone_number,
            provider=provider,
            purchase_details=purchase_result,
            message=f"Successfully purchased phone number {phone_number} from {provider}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error purchasing phone number: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/{phone_number}/diagnose")
async def diagnose_phone_number_connection(phone_number: str):
    """
    Diagnose the connection status of a phone number in Telnyx.
    
    This endpoint provides detailed information about how a phone number
    is configured in Telnyx, including connection assignments and Call Control Applications.
    
    Args:
        phone_number: Phone number to diagnose (E.164 format, e.g., +1234567890)
        
    Returns:
        Dict: Diagnostic information about the phone number
    """
    try:
        from app.telnyx.telnyx_service import TelnyxService
        
        # Remove URL encoding if present
        phone_number = phone_number.replace("%2B", "+")
        
        # Validate phone number format
        if not phone_number.startswith('+'):
            raise HTTPException(
                status_code=400, 
                detail="Phone number must be in E.164 format (e.g., +1234567890)"
            )
        
        # Get diagnostic information
        diagnosis = TelnyxService.diagnose_phone_number_connection(phone_number)
        
        return {
            "success": True,
            "phone_number": phone_number,
            "diagnosis": diagnosis
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error diagnosing phone number: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/release", response_model=ReleasePhoneNumberResponse)
async def release_phone_number(release_data: ReleasePhoneNumberRequest):
    """
    Release a phone number from Twilio or Telnyx account.
    
    This endpoint releases a phone number, removing it from your account and 
    making it available for others to purchase.
    
    Args:
        release_data: Release details including phone number and provider
        
    Returns:
        ReleasePhoneNumberResponse: Release confirmation
    """
    try:
        phone_number = release_data.phone_number
        provider = release_data.provider
        
        # Auto-detect provider if not specified
        if not provider:
            from app.services.phone_number_service import PhoneNumberService
            from app.db.models import PhoneNumber
            
            async with await get_async_db_session() as db:
                phone_record = await db.execute(
                    select(PhoneNumber).where(PhoneNumber.phone_number == phone_number)
                )
                phone_obj = phone_record.scalar_one_or_none()
                
                if phone_obj:
                    provider = phone_obj.provider
                else:
                    raise HTTPException(
                        status_code=404, 
                        detail=f"Phone number {phone_number} not found in database"
                    )
        
        provider = provider.lower()
        
        # Validate provider
        if provider not in ["twilio", "telnyx"]:
            raise HTTPException(status_code=400, detail="Provider must be 'twilio' or 'telnyx'")
        
        # Release the phone number
        if provider == "twilio":
            from app.twilio.twilio_service import TwilioService
            success = TwilioService.release_phone_number(phone_number)
        else:  # telnyx
            from app.telnyx.telnyx_service import TelnyxService
            success = TelnyxService.release_phone_number(phone_number)
        
        if not success:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to release phone number from {provider}"
            )
        
        # Remove from our database
        async with await get_async_db_session() as db:
            phone_record = await db.execute(
                select(PhoneNumber).where(PhoneNumber.phone_number == phone_number)
            )
            phone_obj = phone_record.scalar_one_or_none()
            
            if phone_obj:
                await db.delete(phone_obj)
                await db.commit()
        
        logger.info(f"Successfully released phone number {phone_number} from {provider}")
        
        return ReleasePhoneNumberResponse(
            success=True,
            phone_number=phone_number,
            provider=provider,
            message=f"Successfully released phone number {phone_number} from {provider}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error releasing phone number: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/countries", response_model=CountryCodesResponse)
async def list_country_codes(provider: str = "telnyx"):
    """
    List available country codes for phone number search.
    
    This endpoint returns a list of country codes where phone numbers 
    are available for purchase from the specified provider.
    
    Args:
        provider: Provider to get country codes from ('twilio' or 'telnyx')
        
    Returns:
        CountryCodesResponse: List of available country codes
    """
    try:
        provider = provider.lower()
        
        # Validate provider
        if provider not in ["twilio", "telnyx"]:
            raise HTTPException(status_code=400, detail="Provider must be 'twilio' or 'telnyx'")
        
        # Get country codes
        if provider == "twilio":
            from app.twilio.twilio_service import TwilioService
            country_codes = TwilioService.list_country_codes()
        else:  # telnyx
            from app.telnyx.telnyx_service import TelnyxService
            country_codes = TelnyxService.list_country_codes()
        
        logger.info(f"Retrieved {len(country_codes)} country codes from {provider}")
        
        return CountryCodesResponse(
            success=True,
            country_codes=country_codes,
            provider=provider
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing country codes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.put("/webhooks", response_model=UpdateWebhookResponse)
async def update_phone_webhooks(webhook_data: UpdateWebhookRequest):
    """
    Update voice webhook URL and/or disable SMS webhooks for a phone number.
    
    This endpoint allows you to update the voice webhook URL and optionally disable
    SMS webhooks. SMS webhook URLs are managed automatically through assistant assignment.
    
    Args:
        webhook_data: Webhook update details including phone number and voice URL
        
    Returns:
        UpdateWebhookResponse: Update confirmation with details
    """
    try:
        phone_number = webhook_data.phone_number
        voice_webhook_url = webhook_data.voice_webhook_url
        disable_sms = webhook_data.disable_sms
        enable_sms = webhook_data.enable_sms
        provider = webhook_data.provider
        
        # Validate that at least one action is requested
        if not voice_webhook_url and not disable_sms and not enable_sms:
            raise HTTPException(
                status_code=400,
                detail="At least one action required: voice_webhook_url, disable_sms=true, or enable_sms=true"
            )
        
        # Validate that disable_sms and enable_sms are not both true
        if disable_sms and enable_sms:
            raise HTTPException(
                status_code=400,
                detail="Cannot set both disable_sms and enable_sms to true"
            )
        
        # Validate phone number format
        if not phone_number.startswith('+'):
            raise HTTPException(
                status_code=400,
                detail="Phone number must be in E.164 format (e.g., +1234567890)"
            )
        
        # Auto-detect provider if not specified
        if not provider:
            from app.db.models import PhoneNumber
            
            async with await get_async_db_session() as db:
                phone_record = await db.execute(
                    select(PhoneNumber).where(PhoneNumber.phone_number == phone_number)
                )
                phone_obj = phone_record.scalar_one_or_none()
                
                if phone_obj:
                    provider = phone_obj.provider
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Phone number {phone_number} not found in database"
                    )
        
        provider = provider.lower()
        
        # Validate provider
        if provider not in ["twilio", "telnyx"]:
            raise HTTPException(status_code=400, detail="Provider must be 'twilio' or 'telnyx'")
        
        # Update webhooks using the appropriate service
        if provider == "twilio":
            from app.twilio.twilio_service import TwilioService
            
            # Determine final webhook URLs
            final_voice_webhook_url = voice_webhook_url if voice_webhook_url else get_twiml_webhook_url()
            if "twiml" in final_voice_webhook_url:
                final_voice_webhook_url = get_twiml_webhook_url()
            
            # Handle SMS enable/disable
            sms_webhook_url = None
            if disable_sms:
                # Disable SMS by setting to demo URL
                sms_webhook_url = "https://demo.twilio.com/welcome/sms/reply"
                logger.info(f"Disabling SMS for {phone_number} by setting to demo URL")
                
                # Disable messaging feature
                disable_result = await TwilioService.disable_messaging_feature(phone_number)
                if disable_result.get("status_code") and disable_result.get("status_code") != 200:
                    logger.warning(f"Failed to disable messaging feature: {disable_result}")
            elif enable_sms:
                # Enable SMS based on assigned assistant's configuration
                from app.db.models import PhoneNumber, Assistant
                from app.utils.url_utils import get_server_base_url
                import os
                
                async with await get_async_db_session() as db:
                    # Find phone number and its assigned assistant
                    phone_record = await db.execute(
                        select(PhoneNumber).options(selectinload(PhoneNumber.assistant))
                        .where(PhoneNumber.phone_number == phone_number)
                    )
                    phone_obj = phone_record.scalar_one_or_none()
                    
                    if not phone_obj:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Phone number {phone_number} not found"
                        )
                    
                    if not phone_obj.assistant_id:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Phone number {phone_number} is not assigned to any assistant. Assign it to an assistant with sms_webhook_url configured first."
                        )
                    
                    assistant = phone_obj.assistant
                    if not assistant.sms_webhook_url:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Assistant '{assistant.name}' does not have sms_webhook_url configured. Configure it first."
                        )
                
                # Enable SMS with assistant's configuration
                sms_webhook_url = f"{get_server_base_url()}/twilio-sms-webhook"
                logger.info(f"Enabling SMS for {phone_number} with webhook URL {sms_webhook_url}")
                
                # Determine messaging service SID
                messaging_service_sid = assistant.messaging_service_sid or os.getenv("TWILIO_MESSAGING_SERVICE_SID")
                
                if messaging_service_sid:
                    # Enable messaging feature with messaging service
                    logger.info(f"Enabling messaging feature for {phone_number} with messaging service {messaging_service_sid}")
                    enable_result = await TwilioService.enable_messaging_feature(
                        phone_number=phone_number,
                        messaging_service_sid=messaging_service_sid,
                        sms_webhook_url=sms_webhook_url
                    )
                    if enable_result.get("status") == "error":
                        logger.warning(f"Failed to enable messaging feature: {enable_result}")
                else:
                    logger.info(f"No messaging service SID available, enabling SMS webhook only")
            
            # Update webhooks
            results = TwilioService.update_phone_webhooks(
                phone_number=phone_number,
                voice_webhook_url=final_voice_webhook_url if voice_webhook_url else None,
                sms_webhook_url=sms_webhook_url
            )
        else:  # telnyx
            from app.telnyx.telnyx_service import TelnyxService
            
            # Determine final webhook URLs
            final_voice_webhook_url = voice_webhook_url if voice_webhook_url else get_telnyx_webhook_url()
            
            # Handle SMS enable/disable for Telnyx
            sms_webhook_url = None
            if disable_sms:
                # Disable SMS by setting to empty/null URL (Telnyx disables when webhook is null)
                sms_webhook_url = ""  # Empty string disables SMS webhooks for Telnyx
                logger.info(f"Disabling SMS for {phone_number} by setting empty webhook URL")
            elif enable_sms:
                # Enable SMS based on assigned assistant's configuration
                from app.db.models import PhoneNumber, Assistant
                from app.utils.url_utils import get_server_base_url
                
                async with await get_async_db_session() as db:
                    # Find phone number and its assigned assistant
                    phone_record = await db.execute(
                        select(PhoneNumber).options(selectinload(PhoneNumber.assistant))
                        .where(PhoneNumber.phone_number == phone_number)
                    )
                    phone_obj = phone_record.scalar_one_or_none()
                    
                    if not phone_obj:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Phone number {phone_number} not found"
                        )
                    
                    if not phone_obj.assistant_id:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Phone number {phone_number} is not assigned to any assistant. Assign it to an assistant with sms_webhook_url configured first."
                        )
                    
                    assistant = phone_obj.assistant
                    if not assistant.sms_webhook_url:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Assistant '{assistant.name}' does not have sms_webhook_url configured. Configure it first."
                        )
                
                # Enable SMS with Telnyx webhook URL (same endpoint for both voice and SMS)
                sms_webhook_url = f"{get_server_base_url()}/telnyx-webhook"
                logger.info(f"Enabling SMS for {phone_number} with webhook URL {sms_webhook_url}")
            
            # Get organization's Telnyx credentials for webhook configuration
            api_key = None
            fallback_connection_id = None
            
            # Always fetch organization credentials for Telnyx numbers
            async with await get_async_db_session() as db:
                from app.db.models import PhoneNumber
                
                phone_record = await db.execute(
                    select(PhoneNumber).options(selectinload(PhoneNumber.organization))
                    .where(PhoneNumber.phone_number == phone_number)
                )
                phone_obj = phone_record.scalar_one_or_none()
                
                if phone_obj and phone_obj.organization:
                    api_key = phone_obj.organization.telnyx_api_key
                    fallback_connection_id = phone_obj.organization.telnyx_connection_id
            
            results = TelnyxService.update_phone_webhooks(
                phone_number=phone_number,
                voice_webhook_url=final_voice_webhook_url if voice_webhook_url else None,
                sms_webhook_url=sms_webhook_url,
                api_key=api_key,
                fallback_connection_id=fallback_connection_id
            )
        
        # Check if any updates were successful
        if not any(results.values()):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update webhooks for {phone_number} via {provider}"
            )
        
        # Build response showing what was updated
        updated_webhooks = {}
        if results.get("voice"):
            updated_webhooks["voice"] = final_voice_webhook_url
        if results.get("sms"):
            if disable_sms:
                if provider == "twilio":
                    updated_webhooks["sms"] = "disabled (demo URL)"
                else:  # telnyx
                    updated_webhooks["sms"] = "disabled (empty URL)"
            elif enable_sms:
                updated_webhooks["sms"] = f"enabled ({sms_webhook_url})"
            elif sms_webhook_url:
                updated_webhooks["sms"] = sms_webhook_url
        
        success_count = sum(results.values())
        message = f"Successfully updated {success_count} webhook(s) for {phone_number} via {provider}"
        
        logger.info(f"Updated webhooks for {phone_number}: {updated_webhooks}")
        
        return UpdateWebhookResponse(
            success=True,
            phone_number=phone_number,
            provider=provider,
            updated_webhooks=updated_webhooks,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating phone webhooks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/{phone_number}/webhooks", response_model=GetWebhookResponse)
async def get_phone_webhooks(phone_number: str, provider: Optional[str] = None):
    """
    Get current webhook configuration for a phone number.
    
    This endpoint returns the current voice and SMS webhook URLs configured
    for a specific phone number.
    
    Args:
        phone_number: Phone number to get webhooks for (E.164 format)
        provider: Provider ('twilio' or 'telnyx'). Auto-detected if not provided
        
    Returns:
        GetWebhookResponse: Current webhook configuration
    """
    try:
        # Validate phone number format
        if not phone_number.startswith('+'):
            raise HTTPException(
                status_code=400,
                detail="Phone number must be in E.164 format (e.g., +1234567890)"
            )
        
        # Auto-detect provider if not specified
        if not provider:
            from app.db.models import PhoneNumber
            
            async with await get_async_db_session() as db:
                phone_record = await db.execute(
                    select(PhoneNumber).where(PhoneNumber.phone_number == phone_number)
                )
                phone_obj = phone_record.scalar_one_or_none()
                
                if phone_obj:
                    provider = phone_obj.provider
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Phone number {phone_number} not found in database"
                    )
        
        provider = provider.lower()
        
        # Validate provider
        if provider not in ["twilio", "telnyx"]:
            raise HTTPException(status_code=400, detail="Provider must be 'twilio' or 'telnyx'")
        
        # Get webhook configuration using the appropriate service
        if provider == "twilio":
            from app.twilio.twilio_service import TwilioService
            config = TwilioService.get_phone_webhooks(phone_number)
        else:  # telnyx
            from app.telnyx.telnyx_service import TelnyxService
            config = TelnyxService.get_phone_webhooks(phone_number)
        
        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"Could not retrieve webhook configuration for {phone_number}"
            )
        
        logger.info(f"Retrieved webhook configuration for {phone_number}")
        
        return GetWebhookResponse(
            success=True,
            phone_number=phone_number,
            provider=provider,
            voice_webhook_url=config.get("voice_webhook_url"),
            sms_webhook_url=config.get("sms_webhook_url"),
            configuration=config
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting phone webhooks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

