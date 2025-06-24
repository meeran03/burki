"""Phone Number Service for managing organization phone numbers."""

import logging
from typing import List, Optional, Dict, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import PhoneNumber, Organization, Assistant
from app.twilio.twilio_service import TwilioService
from app.db.database import get_async_db_session

logger = logging.getLogger(__name__)


class PhoneNumberService:
    """Service for managing phone numbers and Twilio integration."""

    @staticmethod
    async def get_organization_phone_numbers(organization_id: int) -> List[PhoneNumber]:
        """Get all phone numbers for an organization."""
        async with await get_async_db_session() as db:
            from sqlalchemy.orm import selectinload
            
            query = select(PhoneNumber).options(
                selectinload(PhoneNumber.assistant)
            ).where(
                PhoneNumber.organization_id == organization_id,
                PhoneNumber.is_active == True
            ).order_by(PhoneNumber.phone_number)
            
            result = await db.execute(query)
            return list(result.scalars().all())

    @staticmethod
    async def sync_twilio_phone_numbers(organization_id: int) -> Dict[str, Any]:
        """
        Sync phone numbers from Twilio account to our database.
        This will fetch available phone numbers from the organization's Twilio account.
        """
        try:
            async with await get_async_db_session() as db:
                # Get organization with Twilio credentials
                org_query = select(Organization).where(Organization.id == organization_id)
                org_result = await db.execute(org_query)
                organization = org_result.scalar_one_or_none()
                
                if not organization:
                    return {"success": False, "error": "Organization not found"}
                
                if not organization.twilio_account_sid or not organization.twilio_auth_token:
                    return {"success": False, "error": "Twilio credentials not configured"}
                
                # Fetch phone numbers from Twilio
                twilio_numbers = TwilioService.get_available_phone_numbers(
                    account_sid=organization.twilio_account_sid,
                    auth_token=organization.twilio_auth_token
                )
                
                if not twilio_numbers:
                    return {"success": False, "error": "No phone numbers found in Twilio account"}
                
                # Process and sync phone numbers
                synced_count = 0
                errors = []
                
                for twilio_number in twilio_numbers:
                    try:
                        # Check if phone number already exists
                        existing_query = select(PhoneNumber).where(
                            PhoneNumber.phone_number == twilio_number['phone_number']
                        )
                        existing_result = await db.execute(existing_query)
                        existing_number = existing_result.scalar_one_or_none()
                        
                        if existing_number:
                            # Update existing phone number
                            existing_number.friendly_name = twilio_number.get('friendly_name')
                            existing_number.twilio_sid = twilio_number.get('sid')
                            existing_number.capabilities = twilio_number.get('capabilities', {})
                            existing_number.phone_metadata = twilio_number
                            existing_number.is_active = True
                        else:
                            # Create new phone number
                            new_number = PhoneNumber(
                                organization_id=organization_id,
                                phone_number=twilio_number['phone_number'],
                                friendly_name=twilio_number.get('friendly_name'),
                                twilio_sid=twilio_number.get('sid'),
                                capabilities=twilio_number.get('capabilities', {}),
                                phone_metadata=twilio_number,
                                is_active=True
                            )
                            db.add(new_number)
                        
                        synced_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error syncing phone number {twilio_number.get('phone_number')}: {e}")
                        errors.append(f"Failed to sync {twilio_number.get('phone_number')}: {str(e)}")
                
                await db.commit()
                
                # Configure webhooks for all synced phone numbers
                webhook_configured_count = 0
                webhook_errors = []
                
                if organization.twilio_account_sid and organization.twilio_auth_token:
                    from app.utils.url_utils import get_twiml_webhook_url
                    webhook_url = get_twiml_webhook_url()
                    
                    for twilio_number in twilio_numbers:
                        try:
                            webhook_success = TwilioService.update_phone_webhook(
                                phone_number=twilio_number['phone_number'],
                                webhook_url=webhook_url,
                                account_sid=organization.twilio_account_sid,
                                auth_token=organization.twilio_auth_token
                            )
                            
                            if webhook_success:
                                webhook_configured_count += 1
                                logger.info(f"Configured webhook for {twilio_number['phone_number']}")
                            else:
                                webhook_errors.append(f"Failed to configure webhook for {twilio_number['phone_number']}")
                                
                        except Exception as webhook_error:
                            logger.error(f"Error configuring webhook for {twilio_number['phone_number']}: {webhook_error}")
                            webhook_errors.append(f"Webhook error for {twilio_number['phone_number']}: {str(webhook_error)}")
                
                return {
                    "success": True,
                    "synced_count": synced_count,
                    "webhook_configured_count": webhook_configured_count,
                    "errors": errors + webhook_errors if (errors or webhook_errors) else None
                }
                
        except Exception as e:
            logger.error(f"Error syncing Twilio phone numbers for organization {organization_id}: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    async def assign_phone_to_assistant(
        phone_number_id: int, 
        assistant_id: Optional[int], 
        organization_id: int
    ) -> Dict[str, Any]:
        """Assign or unassign a phone number to/from an assistant."""
        try:
            async with await get_async_db_session() as db:
                # Get phone number within organization
                phone_query = select(PhoneNumber).where(
                    PhoneNumber.id == phone_number_id,
                    PhoneNumber.organization_id == organization_id
                )
                phone_result = await db.execute(phone_query)
                phone_number = phone_result.scalar_one_or_none()
                
                if not phone_number:
                    return {"success": False, "error": "Phone number not found"}
                
                # If assigning to an assistant, verify assistant exists and belongs to organization
                if assistant_id:
                    assistant_query = select(Assistant).where(
                        Assistant.id == assistant_id,
                        Assistant.organization_id == organization_id
                    )
                    assistant_result = await db.execute(assistant_query)
                    assistant = assistant_result.scalar_one_or_none()
                    
                    if not assistant:
                        return {"success": False, "error": "Assistant not found"}
                
                # Update assignment
                phone_number.assistant_id = assistant_id
                await db.commit()
                
                # Configure Twilio webhook for the phone number if assigning to an assistant
                if assistant_id and assistant:
                    try:
                        from app.utils.url_utils import get_twiml_webhook_url
                        from app.twilio.twilio_service import TwilioService
                        
                        # Get organization's Twilio credentials
                        org_query = select(Organization).where(Organization.id == organization_id)
                        org_result = await db.execute(org_query)
                        organization = org_result.scalar_one_or_none()
                        
                        if organization and organization.twilio_account_sid and organization.twilio_auth_token:
                            webhook_url = get_twiml_webhook_url()
                            
                            # Configure webhook for this phone number
                            webhook_success = TwilioService.update_phone_webhook(
                                phone_number=phone_number.phone_number,
                                webhook_url=webhook_url,
                                account_sid=organization.twilio_account_sid,
                                auth_token=organization.twilio_auth_token
                            )
                            
                            if webhook_success:
                                logger.info(f"Configured webhook for {phone_number.phone_number} assigned to assistant {assistant_id}")
                            else:
                                logger.warning(f"Failed to configure webhook for {phone_number.phone_number}")
                        else:
                            logger.warning(f"No Twilio credentials found for organization {organization_id}")
                            
                    except Exception as webhook_error:
                        logger.error(f"Error configuring webhook for phone number {phone_number.phone_number}: {webhook_error}")
                        # Don't fail the assignment if webhook fails
                
                return {"success": True}
                
        except Exception as e:
            logger.error(f"Error assigning phone number {phone_number_id} to assistant {assistant_id}: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    async def get_available_phone_numbers(organization_id: int) -> List[PhoneNumber]:
        """Get phone numbers that are not assigned to any assistant."""
        async with await get_async_db_session() as db:
            from sqlalchemy.orm import selectinload
            
            query = select(PhoneNumber).options(
                selectinload(PhoneNumber.assistant)
            ).where(
                PhoneNumber.organization_id == organization_id,
                PhoneNumber.is_active == True,
                PhoneNumber.assistant_id.is_(None)
            ).order_by(PhoneNumber.phone_number)
            
            result = await db.execute(query)
            return list(result.scalars().all())

    @staticmethod
    async def delete_phone_number(phone_number_id: int, organization_id: int) -> Dict[str, Any]:
        """Delete/deactivate a phone number."""
        try:
            async with await get_async_db_session() as db:
                # Get phone number within organization
                phone_query = select(PhoneNumber).where(
                    PhoneNumber.id == phone_number_id,
                    PhoneNumber.organization_id == organization_id
                )
                phone_result = await db.execute(phone_query)
                phone_number = phone_result.scalar_one_or_none()
                
                if not phone_number:
                    return {"success": False, "error": "Phone number not found"}
                
                # Check if phone number is assigned to an assistant
                if phone_number.assistant_id:
                    return {"success": False, "error": "Cannot delete phone number that is assigned to an assistant"}
                
                # Deactivate instead of deleting to preserve call history
                phone_number.is_active = False
                await db.commit()
                
                return {"success": True}
                
        except Exception as e:
            logger.error(f"Error deleting phone number {phone_number_id}: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    async def update_organization_twilio_credentials(
        organization_id: int,
        account_sid: str,
        auth_token: str
    ) -> Dict[str, Any]:
        """Update organization's Twilio credentials."""
        try:
            async with await get_async_db_session() as db:
                # Get organization
                org_query = select(Organization).where(Organization.id == organization_id)
                org_result = await db.execute(org_query)
                organization = org_result.scalar_one_or_none()
                
                if not organization:
                    return {"success": False, "error": "Organization not found"}
                
                # Test credentials first
                try:
                    test_numbers = TwilioService.get_available_phone_numbers(
                        account_sid=account_sid,
                        auth_token=auth_token
                    )
                except Exception as e:
                    return {"success": False, "error": f"Invalid Twilio credentials: {str(e)}"}
                
                # Update credentials
                organization.twilio_account_sid = account_sid
                organization.twilio_auth_token = auth_token
                await db.commit()
                
                return {"success": True}
                
        except Exception as e:
            logger.error(f"Error updating Twilio credentials for organization {organization_id}: {e}")
            return {"success": False, "error": str(e)} 