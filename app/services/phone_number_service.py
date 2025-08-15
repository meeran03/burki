"""Phone Number Service for managing organization phone numbers."""
# pylint: disable=logging-fstring-interpolation,bare-except,broad-exception-caught
import logging
from typing import List, Optional, Dict, Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.db.models import PhoneNumber, Organization, Assistant
from app.core.telephony_provider import UnifiedTelephonyService
from app.db.database import get_async_db_session

logger = logging.getLogger(__name__)


class PhoneNumberService:
    """Service for managing phone numbers and telephony provider integration."""

    @staticmethod
    async def get_phone_number_by_number(phone_number_str: str) -> Optional[PhoneNumber]:
        """
        Get a PhoneNumber model instance by its phone number string.
        
        Args:
            phone_number_str: The phone number string (e.g., "+1234567890")
            
        Returns:
            PhoneNumber model instance if found, None otherwise
        """
        try:
            async with await get_async_db_session() as db:
                query = (
                    select(PhoneNumber)
                    .options(selectinload(PhoneNumber.organization))
                    .where(
                        PhoneNumber.phone_number == phone_number_str,
                        PhoneNumber.is_active == True,
                    )
                )
                
                result = await db.execute(query)
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error looking up phone number {phone_number_str}: {e}")
            return None

    @staticmethod
    async def get_organization_phone_numbers(organization_id: int) -> List[PhoneNumber]:
        """Get all phone numbers for an organization."""
        async with await get_async_db_session() as db:

            query = (
                select(PhoneNumber)
                .options(selectinload(PhoneNumber.assistant))
                .where(
                    PhoneNumber.organization_id == organization_id,
                    PhoneNumber.is_active == True,
                )
                .order_by(PhoneNumber.phone_number)
            )

            result = await db.execute(query)
            return list(result.scalars().all())

    @staticmethod
    async def sync_organization_phone_numbers(organization_id: int, provider: str = "twilio") -> Dict[str, Any]:
        """
        Sync phone numbers from telephony provider account to our database.
        This will fetch available phone numbers from the organization's telephony provider account.
        
        Args:
            organization_id: Organization ID
            provider: Provider to sync from ("auto", "twilio", "telnyx")
        """
        try:
            async with await get_async_db_session() as db:
                # Get organization with Twilio credentials
                org_query = select(Organization).where(
                    Organization.id == organization_id
                )
                org_result = await db.execute(org_query)
                organization = org_result.scalar_one_or_none()

                if not organization:
                    return {"success": False, "error": "Organization not found"}

                # Create unified telephony service for this organization
                # This will auto-detect the provider based on configured credentials
                telephony_service = UnifiedTelephonyService()
                
                # Check if organization has credentials for requested provider
                has_twilio = organization.twilio_account_sid and organization.twilio_auth_token
                has_telnyx = getattr(organization, 'telnyx_api_key', None) and getattr(organization, 'telnyx_connection_id', None)
                
                # Validate provider and credentials
                if provider == "telnyx":
                    if not has_telnyx:
                        return {
                            "success": False,
                            "error": "Telnyx credentials not configured for this organization",
                        }
                elif provider == "twilio":
                    if not has_twilio:
                        return {
                            "success": False,
                            "error": "Twilio credentials not configured for this organization",
                        }
                else:
                    return {
                        "success": False,
                        "error": f"Invalid provider '{provider}'. Must be 'twilio' or 'telnyx'",
                    }

                # Set organization credentials on the service based on requested provider
                if provider == "telnyx":
                    # Create Telnyx service with org credentials
                    from app.core.telephony_provider import TelnyxTelephonyService
                    telephony_service.provider_service = TelnyxTelephonyService(
                        api_key=organization.telnyx_api_key,
                        connection_id=organization.telnyx_connection_id
                    )
                elif provider == "twilio":
                    # Create Twilio service with org credentials
                    from app.core.telephony_provider import TwilioTelephonyService
                    telephony_service.provider_service = TwilioTelephonyService(
                        account_sid=organization.twilio_account_sid,
                        auth_token=organization.twilio_auth_token
                    )

                # Fetch phone numbers from telephony provider
                provider_numbers = telephony_service.get_available_phone_numbers()

                if not provider_numbers:
                    provider_type = telephony_service.get_provider_type()
                    return {
                        "success": False,
                        "error": f"No phone numbers found in {provider_type} account",
                    }

                # Process and sync phone numbers
                synced_count = 0
                errors = []

                for provider_number in provider_numbers:
                    try:
                        # Check if phone number already exists
                        existing_query = select(PhoneNumber).where(
                            PhoneNumber.phone_number == provider_number["phone_number"]
                        )
                        existing_result = await db.execute(existing_query)
                        existing_number = existing_result.scalar_one_or_none()

                        # Determine provider type
                        provider_type = telephony_service.get_provider_type().lower()
                        
                        if existing_number:
                            # Update existing phone number
                            existing_number.friendly_name = provider_number.get(
                                "friendly_name"
                            )
                            existing_number.provider = provider_type
                            existing_number.provider_phone_id = provider_number.get("sid") or provider_number.get("id")
                            existing_number.capabilities = provider_number.get(
                                "capabilities", {}
                            )
                            existing_number.phone_metadata = provider_number
                            existing_number.is_active = True
                        else:
                            # Create new phone number
                            new_number = PhoneNumber(
                                organization_id=organization_id,
                                phone_number=provider_number["phone_number"],
                                friendly_name=provider_number.get("friendly_name"),
                                provider=provider_type,
                                provider_phone_id=provider_number.get("sid") or provider_number.get("id"),
                                capabilities=provider_number.get("capabilities", {}),
                                phone_metadata=provider_number,
                                is_active=True,
                            )
                            db.add(new_number)

                        synced_count += 1

                    except Exception as e:
                        logger.error(
                            f"Error syncing phone number {provider_number.get('phone_number')}: {e}"
                        )
                        errors.append(
                            f"Failed to sync {provider_number.get('phone_number')}: {str(e)}"
                        )

                await db.commit()
                return {"success": True, "synced_count": synced_count}

        except Exception as e:
            logger.error(
                f"Error syncing telephony provider phone numbers for organization {organization_id}: {e}"
            )
            return {"success": False, "error": str(e)}

    @staticmethod
    async def assign_phone_to_assistant(
        phone_number_id: int, assistant_id: Optional[int], organization_id: int, friendly_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Assign or unassign a phone number to/from an assistant."""
        try:
            async with await get_async_db_session() as db:
                # Get phone number within organization
                phone_query = select(PhoneNumber).where(
                    PhoneNumber.id == phone_number_id,
                    PhoneNumber.organization_id == organization_id,
                )
                phone_result = await db.execute(phone_query)
                phone_number = phone_result.scalar_one_or_none()

                if not phone_number:
                    return {"success": False, "error": "Phone number not found"}

                # If assigning to an assistant, verify assistant exists and belongs to organization
                if assistant_id:
                    assistant_query = select(Assistant).where(
                        Assistant.id == assistant_id,
                        Assistant.organization_id == organization_id,
                    )
                    assistant_result = await db.execute(assistant_query)
                    assistant = assistant_result.scalar_one_or_none()

                    if not assistant:
                        return {"success": False, "error": "Assistant not found"}

                # Update assignment
                phone_number.assistant_id = assistant_id
                
                # Update friendly name if provided
                if friendly_name is not None:
                    phone_number.friendly_name = friendly_name.strip() if friendly_name else None
                
                await db.commit()

                # Configure telephony provider webhook for the phone number if assigning to an assistant
                if assistant_id and assistant:
                    try:
                        from app.utils.url_utils import get_twiml_webhook_url

                        # Get organization's credentials
                        org_query = select(Organization).where(
                            Organization.id == organization_id
                        )
                        org_result = await db.execute(org_query)
                        organization = org_result.scalar_one_or_none()

                        if organization:
                            # Create unified telephony service based on phone number's provider
                            if phone_number.provider == "telnyx":
                                from app.core.telephony_provider import TelnyxTelephonyService
                                provider_service = TelnyxTelephonyService(
                                    api_key=organization.telnyx_api_key,
                                    connection_id=organization.telnyx_connection_id
                                )
                            else:  # Default to Twilio
                                from app.core.telephony_provider import TwilioTelephonyService
                                provider_service = TwilioTelephonyService(
                                    account_sid=organization.twilio_account_sid,
                                    auth_token=organization.twilio_auth_token
                                )
                            
                            telephony_service = UnifiedTelephonyService(provider_service=provider_service)
                            
                            # Get appropriate webhook URL based on phone number's provider
                            if phone_number.provider == "telnyx":
                                from app.utils.url_utils import get_server_base_url
                                webhook_url = f"{get_server_base_url()}/telnyx-webhook"
                            else:  # Twilio
                                webhook_url = get_twiml_webhook_url()

                            # Configure webhook for this phone number
                            try:
                                webhook_success = telephony_service.update_phone_webhook(
                                    phone_number=phone_number.phone_number,
                                    webhook_url=webhook_url
                                )
                            except Exception as webhook_error:
                                logger.error(f"Failed to update webhook for {phone_number.phone_number}: {webhook_error}")
                                webhook_success = False

                            if webhook_success:
                                logger.info(
                                    f"Configured {phone_number.provider} webhook for {phone_number.phone_number} assigned to assistant {assistant_id}"
                                )
                            else:
                                logger.warning(
                                    f"Failed to configure {phone_number.provider} webhook for {phone_number.phone_number}"
                                )
                        else:
                            logger.warning(
                                f"Organization {organization_id} not found for webhook configuration"
                            )

                    except Exception as webhook_error:
                        logger.error(
                            f"Error configuring webhook for phone number {phone_number.phone_number}: {webhook_error}"
                        )
                        # Don't fail the assignment if webhook fails

                return {"success": True}

        except Exception as e:
            logger.error(
                f"Error assigning phone number {phone_number_id} to assistant {assistant_id}: {e}"
            )
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_available_phone_numbers(organization_id: int) -> List[PhoneNumber]:
        """Get phone numbers that are not assigned to any assistant."""
        async with await get_async_db_session() as db:
            from sqlalchemy.orm import selectinload

            query = (
                select(PhoneNumber)
                .options(selectinload(PhoneNumber.assistant))
                .where(
                    PhoneNumber.organization_id == organization_id,
                    PhoneNumber.is_active == True,
                    PhoneNumber.assistant_id.is_(None),
                )
                .order_by(PhoneNumber.phone_number)
            )

            result = await db.execute(query)
            return list(result.scalars().all())

    @staticmethod
    async def delete_phone_number(
        phone_number_id: int, organization_id: int
    ) -> Dict[str, Any]:
        """Delete/deactivate a phone number."""
        try:
            async with await get_async_db_session() as db:
                # Get phone number within organization
                phone_query = select(PhoneNumber).where(
                    PhoneNumber.id == phone_number_id,
                    PhoneNumber.organization_id == organization_id,
                )
                phone_result = await db.execute(phone_query)
                phone_number = phone_result.scalar_one_or_none()

                if not phone_number:
                    return {"success": False, "error": "Phone number not found"}

                # Check if phone number is assigned to an assistant
                if phone_number.assistant_id:
                    return {
                        "success": False,
                        "error": "Cannot delete phone number that is assigned to an assistant",
                    }

                # Deactivate instead of deleting to preserve call history
                phone_number.is_active = False
                await db.commit()

                return {"success": True}

        except Exception as e:
            logger.error(f"Error deleting phone number {phone_number_id}: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    async def update_organization_telephony_credentials(
        organization_id: int, 
        provider: str, 
        credentials: Dict[str, str]
    ) -> Dict[str, Any]:
        """Update organization's telephony provider credentials."""
        try:
            async with await get_async_db_session() as db:
                # Get organization
                org_query = select(Organization).where(
                    Organization.id == organization_id
                )
                org_result = await db.execute(org_query)
                organization = org_result.scalar_one_or_none()

                if not organization:
                    return {"success": False, "error": "Organization not found"}

                # Test credentials first by trying to fetch phone numbers
                try:
                    if provider == "twilio":
                        from app.core.telephony_provider import TwilioTelephonyService
                        test_service = TwilioTelephonyService(
                            account_sid=credentials.get("account_sid"),
                            auth_token=credentials.get("auth_token")
                        )
                    elif provider == "telnyx":
                        from app.core.telephony_provider import TelnyxTelephonyService
                        test_service = TelnyxTelephonyService(
                            api_key=credentials.get("api_key"),
                            connection_id=credentials.get("connection_id")
                        )
                    else:
                        return {"success": False, "error": f"Unsupported provider: {provider}"}
                    
                    # Test by attempting to get phone numbers
                    test_numbers = test_service.get_available_phone_numbers()
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Invalid {provider} credentials: {str(e)}",
                    }

                # Update credentials based on provider
                if provider == "twilio":
                    organization.twilio_account_sid = credentials.get("account_sid")
                    organization.twilio_auth_token = credentials.get("auth_token")
                elif provider == "telnyx":
                    organization.telnyx_api_key = credentials.get("api_key")
                    organization.telnyx_connection_id = credentials.get("connection_id")
                
                await db.commit()
                return {"success": True}

        except Exception as e:
            logger.error(
                f"Error updating {provider} credentials for organization {organization_id}: {e}"
            )
            return {"success": False, "error": str(e)}

    @staticmethod
    async def update_phone_number_metadata(
        phone_number_id: int, organization_id: int, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update phone number metadata."""
        try:
            async with await get_async_db_session() as db:
                # Get phone number within organization
                phone_query = select(PhoneNumber).where(
                    PhoneNumber.id == phone_number_id,
                    PhoneNumber.organization_id == organization_id,
                )
                phone_result = await db.execute(phone_query)
                phone_number = phone_result.scalar_one_or_none()

                if not phone_number:
                    return {"success": False, "error": "Phone number not found"}

                # Update metadata
                phone_number.phone_metadata = metadata
                await db.commit()

                return {"success": True}

        except Exception as e:
            logger.error(f"Error updating phone number metadata {phone_number_id}: {e}")
            return {"success": False, "error": str(e)}
