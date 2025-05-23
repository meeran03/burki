# pylint: disable=logging-fstring-interpolation, broad-exception-caught
import logging
import os
import stripe
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from decimal import Decimal
from sqlalchemy import select, and_, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import joinedload

from app.db.models import (
    Organization, 
    BillingPlan, 
    BillingAccount, 
    UsageRecord, 
    BillingTransaction,
    Call,
    Assistant
)
from app.db.database import get_async_db_session, get_db

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")


class BillingService:
    """
    Service class for handling billing operations with Stripe integration.
    """

    @staticmethod
    async def initialize_default_plans():
        """
        Initialize default billing plans if they don't exist.
        """
        try:
            async with await get_async_db_session() as db:
                # Check if plans already exist
                query = select(BillingPlan)
                result = await db.execute(query)
                existing_plans = result.scalars().all()
                
                if len(existing_plans) > 0:
                    logger.info("Billing plans already exist, skipping initialization")
                    return

                # Create default plans
                starter_plan = BillingPlan(
                    name="Starter",
                    description="500 free minutes per month",
                    monthly_minutes=500,
                    price_cents=0,
                    features={
                        "unlimited_assistants": False,
                        "webhook_support": True,
                        "api_access": True,
                        "priority_support": False,
                        "custom_integrations": False,
                    },
                    sort_order=1
                )

                pro_plan = BillingPlan(
                    name="Pro",
                    description="Unlimited minutes for $20/month",
                    monthly_minutes=None,  # Unlimited
                    price_cents=2000,  # $20
                    stripe_price_id=os.getenv("STRIPE_PRO_PLAN_PRICE_ID"),
                    features={
                        "unlimited_assistants": True,
                        "webhook_support": True,
                        "api_access": True,
                        "priority_support": True,
                        "custom_integrations": True,
                    },
                    sort_order=2
                )

                db.add(starter_plan)
                db.add(pro_plan)
                await db.commit()
                logger.info("Default billing plans created successfully")

        except SQLAlchemyError as e:
            logger.error(f"Error initializing default billing plans: {e}")
            raise

    @staticmethod
    async def create_billing_account_for_organization(organization_id: int) -> BillingAccount:
        """
        Create a billing account for an organization with the starter plan.
        """
        try:
            async with await get_async_db_session() as db:
                # Get the starter plan
                query = select(BillingPlan).where(BillingPlan.name == "Starter")
                result = await db.execute(query)
                starter_plan = result.scalar_one_or_none()
                
                if not starter_plan:
                    raise ValueError("Starter plan not found")

                # Check if billing account already exists
                existing_query = select(BillingAccount).where(
                    BillingAccount.organization_id == organization_id
                )
                existing_result = await db.execute(existing_query)
                existing_account = existing_result.scalar_one_or_none()
                
                if existing_account:
                    return existing_account

                # Create billing account
                now = datetime.utcnow()
                billing_account = BillingAccount(
                    organization_id=organization_id,
                    plan_id=starter_plan.id,
                    current_period_start=now,
                    current_period_end=now.replace(day=1) + timedelta(days=32),  # Next month
                    current_period_minutes_used=0,
                )

                db.add(billing_account)
                await db.commit()
                await db.refresh(billing_account)
                
                # Load the plan relationship
                query = select(BillingAccount).options(
                    joinedload(BillingAccount.plan),
                    joinedload(BillingAccount.organization)
                ).where(BillingAccount.id == billing_account.id)
                result = await db.execute(query)
                billing_account = result.scalar_one()
                
                logger.info(f"Created billing account for organization {organization_id}")
                return billing_account

        except SQLAlchemyError as e:
            logger.error(f"Error creating billing account: {e}")
            raise

    @staticmethod
    async def get_billing_account(organization_id: int) -> Optional[BillingAccount]:
        """
        Get billing account for an organization.
        """
        async with await get_async_db_session() as db:
            query = select(BillingAccount).options(
                joinedload(BillingAccount.plan),
                joinedload(BillingAccount.organization)
            ).where(BillingAccount.organization_id == organization_id)
            result = await db.execute(query)
            return result.scalar_one_or_none()

    @staticmethod
    async def get_all_plans() -> List[BillingPlan]:
        """
        Get all active billing plans.
        """
        async with await get_async_db_session() as db:
            query = select(BillingPlan).where(
                BillingPlan.is_active == True
            ).order_by(BillingPlan.sort_order)
            result = await db.execute(query)
            return result.scalars().all()

    @staticmethod
    async def create_stripe_customer(organization: Organization, email: str) -> str:
        """
        Create a Stripe customer for an organization.
        """
        try:
            customer = stripe.Customer.create(
                email=email,
                name=organization.name,
                metadata={
                    "organization_id": str(organization.id),
                    "organization_slug": organization.slug,
                }
            )
            logger.info(f"Created Stripe customer {customer.id} for organization {organization.id}")
            return customer.id
        except stripe.error.StripeError as e:
            logger.error(f"Error creating Stripe customer: {e}")
            raise

    @staticmethod
    async def upgrade_to_pro_plan(organization_id: int, payment_method_id: str) -> Dict[str, Any]:
        """
        Upgrade an organization to the Pro plan.
        """
        try:
            async with await get_async_db_session() as db:
                # Get billing account
                billing_account = await BillingService.get_billing_account(organization_id)
                if not billing_account:
                    raise ValueError("Billing account not found")

                # Get Pro plan
                query = select(BillingPlan).where(BillingPlan.name == "Pro")
                result = await db.execute(query)
                pro_plan = result.scalar_one_or_none()
                
                if not pro_plan:
                    raise ValueError("Pro plan not found")

                # Create Stripe customer if needed
                if not billing_account.stripe_customer_id:
                    customer_id = await BillingService.create_stripe_customer(
                        billing_account.organization, 
                        billing_account.organization.users[0].email if billing_account.organization.users else "unknown@example.com"
                    )
                    billing_account.stripe_customer_id = customer_id

                # Attach payment method to customer
                stripe.PaymentMethod.attach(
                    payment_method_id,
                    customer=billing_account.stripe_customer_id,
                )

                # Set as default payment method
                stripe.Customer.modify(
                    billing_account.stripe_customer_id,
                    invoice_settings={
                        "default_payment_method": payment_method_id,
                    },
                )

                # Create subscription
                subscription = stripe.Subscription.create(
                    customer=billing_account.stripe_customer_id,
                    items=[{
                        "price": pro_plan.stripe_price_id,
                    }],
                    metadata={
                        "organization_id": str(organization_id),
                        "billing_account_id": str(billing_account.id),
                    }
                )

                # Update billing account
                billing_account.plan_id = pro_plan.id
                billing_account.stripe_subscription_id = subscription.id
                billing_account.is_payment_method_attached = True
                billing_account.status = "active"

                await db.commit()

                # Create transaction record
                await BillingService.create_transaction(
                    billing_account_id=billing_account.id,
                    transaction_type="subscription",
                    amount_cents=pro_plan.price_cents,
                    description=f"Subscription to {pro_plan.name} plan",
                    stripe_subscription_id=subscription.id,
                    status="succeeded"
                )

                logger.info(f"Successfully upgraded organization {organization_id} to Pro plan")
                return {
                    "success": True,
                    "subscription_id": subscription.id,
                    "customer_id": billing_account.stripe_customer_id
                }

        except stripe.error.StripeError as e:
            logger.error(f"Stripe error during plan upgrade: {e}")
            raise
        except Exception as e:
            logger.error(f"Error upgrading to Pro plan: {e}")
            raise

    @staticmethod
    async def setup_auto_topup(
        organization_id: int, 
        payment_method_id: str,
        threshold_minutes: int = 10,
        topup_amount_minutes: int = 100,
        topup_price_cents: int = 500
    ) -> bool:
        """
        Setup auto-topup for an organization.
        """
        try:
            async with await get_async_db_session() as db:
                billing_account = await BillingService.get_billing_account(organization_id)
                if not billing_account:
                    raise ValueError("Billing account not found")

                # Create Stripe customer if needed
                if not billing_account.stripe_customer_id:
                    customer_id = await BillingService.create_stripe_customer(
                        billing_account.organization,
                        billing_account.organization.users[0].email if billing_account.organization.users else "unknown@example.com"
                    )
                    billing_account.stripe_customer_id = customer_id

                # Attach payment method
                stripe.PaymentMethod.attach(
                    payment_method_id,
                    customer=billing_account.stripe_customer_id,
                )

                # Set as default payment method
                stripe.Customer.modify(
                    billing_account.stripe_customer_id,
                    invoice_settings={
                        "default_payment_method": payment_method_id,
                    },
                )

                # Update billing account
                billing_account.auto_topup_enabled = True
                billing_account.topup_threshold_minutes = threshold_minutes
                billing_account.topup_amount_minutes = topup_amount_minutes
                billing_account.topup_price_cents = topup_price_cents
                billing_account.is_payment_method_attached = True

                await db.commit()
                logger.info(f"Auto-topup setup for organization {organization_id}")
                return True

        except Exception as e:
            logger.error(f"Error setting up auto-topup: {e}")
            raise

    @staticmethod
    async def process_topup(billing_account_id: int) -> bool:
        """
        Process an automatic top-up when threshold is reached.
        """
        try:
            async with await get_async_db_session() as db:
                query = select(BillingAccount).options(
                    joinedload(BillingAccount.plan)
                ).where(BillingAccount.id == billing_account_id)
                result = await db.execute(query)
                billing_account = result.scalar_one_or_none()
                
                if not billing_account or not billing_account.auto_topup_enabled:
                    return False

                # Check if topup is needed
                remaining_minutes = billing_account.get_remaining_minutes()
                if remaining_minutes > billing_account.topup_threshold_minutes:
                    return False

                # Create payment intent
                payment_intent = stripe.PaymentIntent.create(
                    amount=billing_account.topup_price_cents,
                    currency="usd",
                    customer=billing_account.stripe_customer_id,
                    description=f"Auto top-up: {billing_account.topup_amount_minutes} minutes",
                    metadata={
                        "billing_account_id": str(billing_account_id),
                        "topup_minutes": str(billing_account.topup_amount_minutes),
                    },
                    confirm=True,
                    off_session=True,  # This indicates it's an automated payment
                )

                if payment_intent.status == "succeeded":
                    # Add minutes to account (this will be handled by adding a negative usage record)
                    await BillingService.add_usage_credit(
                        billing_account_id=billing_account_id,
                        minutes=billing_account.topup_amount_minutes,
                        description=f"Auto top-up: {billing_account.topup_amount_minutes} minutes"
                    )

                    # Create transaction record
                    await BillingService.create_transaction(
                        billing_account_id=billing_account_id,
                        transaction_type="topup",
                        amount_cents=billing_account.topup_price_cents,
                        description=f"Auto top-up: {billing_account.topup_amount_minutes} minutes",
                        stripe_payment_intent_id=payment_intent.id,
                        minutes_credited=billing_account.topup_amount_minutes,
                        status="succeeded"
                    )

                    logger.info(f"Processed auto top-up for billing account {billing_account_id}")
                    return True

        except stripe.error.CardError as e:
            logger.error(f"Card error during auto top-up: {e}")
            # Disable auto-topup if card fails
            await BillingService.disable_auto_topup(billing_account_id)
            return False
        except Exception as e:
            logger.error(f"Error processing auto top-up: {e}")
            return False

    @staticmethod
    async def disable_auto_topup(billing_account_id: int) -> bool:
        """
        Disable auto-topup for a billing account.
        """
        try:
            async with await get_async_db_session() as db:
                query = select(BillingAccount).where(BillingAccount.id == billing_account_id)
                result = await db.execute(query)
                billing_account = result.scalar_one_or_none()
                
                if billing_account:
                    billing_account.auto_topup_enabled = False
                    await db.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Error disabling auto-topup: {e}")
            return False

    @staticmethod
    async def record_call_usage(call_id: int) -> bool:
        """
        Record usage for a completed call.
        """
        try:
            async with await get_async_db_session() as db:
                # Get call with assistant and organization info
                query = select(Call).options(
                    joinedload(Call.assistant).joinedload(Assistant.organization).joinedload(Organization.billing_account)
                ).where(Call.id == call_id)
                result = await db.execute(query)
                call = result.scalar_one_or_none()
                
                if not call or not call.duration:
                    return False

                billing_account = call.assistant.organization.billing_account
                if not billing_account:
                    logger.warning(f"No billing account found for call {call_id}")
                    return False

                # Calculate minutes used (round up to next minute)
                minutes_used = max(1, (call.duration + 59) // 60)  # Round up to next minute

                # Create usage record
                usage_record = UsageRecord(
                    billing_account_id=billing_account.id,
                    call_id=call_id,
                    minutes_used=minutes_used,
                    usage_type="call",
                    description=f"Call {call.call_sid}",
                    billing_period_start=billing_account.current_period_start,
                    billing_period_end=billing_account.current_period_end,
                    record_metadata={
                        "call_sid": call.call_sid,
                        "assistant_id": call.assistant_id,
                        "duration_seconds": call.duration,
                    }
                )

                db.add(usage_record)

                # Update billing account usage
                billing_account.current_period_minutes_used += minutes_used
                
                await db.commit()

                # Check if auto-topup is needed
                if billing_account.auto_topup_enabled:
                    remaining_minutes = billing_account.get_remaining_minutes()
                    if remaining_minutes <= billing_account.topup_threshold_minutes:
                        # Process topup asynchronously
                        import asyncio
                        asyncio.create_task(BillingService.process_topup(billing_account.id))

                logger.info(f"Recorded {minutes_used} minutes usage for call {call_id}")
                return True

        except Exception as e:
            logger.error(f"Error recording call usage: {e}")
            return False

    @staticmethod
    async def add_usage_credit(billing_account_id: int, minutes: int, description: str) -> bool:
        """
        Add usage credit (negative usage) to a billing account.
        """
        try:
            async with await get_async_db_session() as db:
                query = select(BillingAccount).where(BillingAccount.id == billing_account_id)
                result = await db.execute(query)
                billing_account = result.scalar_one_or_none()
                
                if not billing_account:
                    return False

                # Create negative usage record (credit)
                usage_record = UsageRecord(
                    billing_account_id=billing_account_id,
                    minutes_used=-minutes,  # Negative for credit
                    usage_type="topup_credit",
                    description=description,
                    billing_period_start=billing_account.current_period_start,
                    billing_period_end=billing_account.current_period_end,
                )

                db.add(usage_record)

                # Update billing account usage (subtract the minutes)
                billing_account.current_period_minutes_used = max(0, billing_account.current_period_minutes_used - minutes)
                
                await db.commit()
                return True

        except Exception as e:
            logger.error(f"Error adding usage credit: {e}")
            return False

    @staticmethod
    async def create_transaction(
        billing_account_id: int,
        transaction_type: str,
        amount_cents: int,
        description: str,
        stripe_payment_intent_id: str = None,
        stripe_invoice_id: str = None,
        stripe_subscription_id: str = None,
        minutes_credited: int = None,
        status: str = "pending"
    ) -> BillingTransaction:
        """
        Create a billing transaction record.
        """
        try:
            async with await get_async_db_session() as db:
                transaction = BillingTransaction(
                    billing_account_id=billing_account_id,
                    transaction_type=transaction_type,
                    amount_cents=amount_cents,
                    description=description,
                    stripe_payment_intent_id=stripe_payment_intent_id,
                    stripe_invoice_id=stripe_invoice_id,
                    minutes_credited=minutes_credited,
                    status=status,
                    transaction_metadata={
                        "stripe_subscription_id": stripe_subscription_id
                    }
                )

                db.add(transaction)
                await db.commit()
                await db.refresh(transaction)
                return transaction

        except Exception as e:
            logger.error(f"Error creating transaction: {e}")
            raise

    @staticmethod
    async def check_usage_limits(organization_id: int, additional_minutes: int = 1) -> Dict[str, Any]:
        """
        Check if organization can make a call based on usage limits.
        """
        billing_account = await BillingService.get_billing_account(organization_id)
        if not billing_account:
            return {"allowed": False, "reason": "No billing account found"}

        # Check if within limits
        if billing_account.is_within_limits(additional_minutes):
            return {"allowed": True}

        # Check if payment method is attached for overages or Pro plan
        if billing_account.needs_payment_method() and billing_account.is_payment_method_attached:
            # If Pro plan (unlimited), always allow
            if billing_account.plan.monthly_minutes is None:
                return {"allowed": True}
            
            # If auto-topup is enabled, allow and trigger topup
            if billing_account.auto_topup_enabled:
                return {"allowed": True, "will_trigger_topup": True}

        return {
            "allowed": False,
            "reason": "Usage limit exceeded",
            "remaining_minutes": billing_account.get_remaining_minutes(),
            "needs_upgrade": True
        }

    @staticmethod
    async def get_usage_summary(organization_id: int) -> Dict[str, Any]:
        """
        Get usage summary for an organization.
        """
        billing_account = await BillingService.get_billing_account(organization_id)
        if not billing_account:
            return {}

        async with await get_async_db_session() as db:
            # Get current period usage
            usage_query = select(func.sum(UsageRecord.minutes_used)).where(
                and_(
                    UsageRecord.billing_account_id == billing_account.id,
                    UsageRecord.billing_period_start == billing_account.current_period_start
                )
            )
            usage_result = await db.execute(usage_query)
            total_usage = usage_result.scalar() or 0

            # Get recent transactions
            transactions_query = select(BillingTransaction).where(
                BillingTransaction.billing_account_id == billing_account.id
            ).order_by(BillingTransaction.created_at.desc()).limit(10)
            transactions_result = await db.execute(transactions_query)
            recent_transactions = transactions_result.scalars().all()

            return {
                "plan_name": billing_account.plan.name,
                "monthly_limit": billing_account.plan.monthly_minutes,
                "current_usage": int(total_usage),
                "remaining_minutes": billing_account.get_remaining_minutes(),
                "period_start": billing_account.current_period_start.isoformat() if billing_account.current_period_start else None,
                "period_end": billing_account.current_period_end.isoformat() if billing_account.current_period_end else None,
                "auto_topup_enabled": billing_account.auto_topup_enabled,
                "topup_threshold": billing_account.topup_threshold_minutes,
                "topup_amount": billing_account.topup_amount_minutes,
                "recent_transactions": [
                    {
                        "id": t.id,
                        "type": t.transaction_type,
                        "amount": t.amount_cents / 100,
                        "description": t.description,
                        "status": t.status,
                        "created_at": t.created_at.isoformat() if t.created_at else None,
                    }
                    for t in recent_transactions
                ]
            } 