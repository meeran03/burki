# pylint: disable=logging-fstring-interpolation, broad-exception-caught
import logging
import json
import os
import stripe
from typing import Optional
from fastapi import APIRouter, Depends, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User
from app.services.billing_service import BillingService
from app.services.auth_service import AuthService

# Create router
router = APIRouter(tags=["billing"])

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

logger = logging.getLogger(__name__)

# Stripe configuration
STRIPE_PUBLIC_KEY = os.getenv("STRIPE_PUBLIC_KEY")


# Helper functions from auth.py
def get_template_context(request: Request, **extra_context) -> dict:
    """Get template context with session data and any extra context."""
    context = {
        "request": request,
        "session": {
            "user_id": request.session.get("user_id"),
            "organization_id": request.session.get("organization_id"),
            "user_email": request.session.get("user_email", ""),
            "user_first_name": request.session.get("user_first_name", ""),
            "user_last_name": request.session.get("user_last_name", ""),
            "organization_name": request.session.get("organization_name", ""),
            "organization_slug": request.session.get("organization_slug", ""),
            "api_key_count": request.session.get("api_key_count", 0),
        },
        "stripe_public_key": STRIPE_PUBLIC_KEY,
    }
    context.update(extra_context)
    return context


async def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    """Get the current authenticated user from session."""
    user_id = request.session.get("user_id")
    if not user_id:
        return None

    user = db.query(User).filter(User.id == user_id).first()
    if user:
        user.split_full_name_if_needed()
    return user


async def require_auth(request: Request, db: Session = Depends(get_db)) -> User:
    """Require authentication and return the current user."""
    user = await get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    user.split_full_name_if_needed()
    return user


# ========== Billing Routes ==========

@router.get("/billing", response_class=HTMLResponse)
async def billing_dashboard(request: Request, current_user: User = Depends(require_auth)):
    """Display the billing dashboard."""
    try:
        # Extract success/error messages from query parameters
        success_message = request.query_params.get("success")
        error_message = request.query_params.get("error")
        
        # Get billing account and usage summary
        billing_account = await BillingService.get_billing_account(current_user.organization_id)
        if not billing_account:
            # Create billing account if it doesn't exist
            billing_account = await BillingService.create_billing_account_for_organization(
                current_user.organization_id
            )

        # Get usage summary
        usage_summary = await BillingService.get_usage_summary(current_user.organization_id)
        
        # Get all available plans
        plans = await BillingService.get_all_plans()
        
        # Ensure current_plan is available
        current_plan = billing_account.plan if billing_account.plan else (plans[0] if plans else None)

        return templates.TemplateResponse(
            "billing/dashboard.html",
            get_template_context(
                request,
                billing_account=billing_account,
                usage_summary=usage_summary,
                plans=plans,
                current_plan=current_plan,
                success=success_message,
                error=error_message,
            ),
        )

    except Exception as e:
        logger.error(f"Error loading billing dashboard: {e}")
        return templates.TemplateResponse(
            "billing/dashboard.html",
            get_template_context(
                request,
                error="Error loading billing information",
                billing_account=None,
                usage_summary={},
                plans=[],
                current_plan=None,
            ),
            status_code=500,
        )


@router.get("/billing/plans", response_class=HTMLResponse)
async def billing_plans(request: Request, current_user: User = Depends(require_auth)):
    """Display available billing plans."""
    try:
        # Extract success/error messages from query parameters
        success_message = request.query_params.get("success")
        error_message = request.query_params.get("error")
        
        # Get current billing account
        billing_account = await BillingService.get_billing_account(current_user.organization_id)
        if not billing_account:
            billing_account = await BillingService.create_billing_account_for_organization(
                current_user.organization_id
            )

        # Get all available plans
        plans = await BillingService.get_all_plans()
        
        # Ensure current_plan is available
        current_plan = billing_account.plan if billing_account.plan else (plans[0] if plans else None)

        return templates.TemplateResponse(
            "billing/plans.html",
            get_template_context(
                request,
                billing_account=billing_account,
                plans=plans,
                current_plan=current_plan,
                success=success_message,
                error=error_message,
            ),
        )

    except Exception as e:
        logger.error(f"Error loading billing plans: {e}")
        return templates.TemplateResponse(
            "billing/plans.html",
            get_template_context(
                request,
                error="Error loading billing plans",
                billing_account=None,
                plans=[],
                current_plan=None,
            ),
            status_code=500,
        )


@router.post("/billing/upgrade-to-pro")
async def upgrade_to_pro(
    request: Request,
    payment_method_id: str = Form(...),
    current_user: User = Depends(require_auth)
):
    """Handle Pro plan upgrade."""
    try:
        result = await BillingService.upgrade_to_pro_plan(
            organization_id=current_user.organization_id,
            payment_method_id=payment_method_id
        )

        if result.get("success"):
            return RedirectResponse(
                url="/billing?success=Upgraded to Pro plan successfully!",
                status_code=302
            )
        else:
            return RedirectResponse(
                url="/billing?error=Failed to upgrade to Pro plan",
                status_code=302
            )

    except stripe.error.CardError as e:
        logger.error(f"Card error during upgrade: {e}")
        return RedirectResponse(
            url=f"/billing?error=Payment failed: {e.user_message}",
            status_code=302
        )
    except Exception as e:
        logger.error(f"Error upgrading to Pro plan: {e}")
        return RedirectResponse(
            url="/billing?error=An error occurred during upgrade",
            status_code=302
        )


@router.get("/billing/topup", response_class=HTMLResponse)
async def topup_setup_page(request: Request, current_user: User = Depends(require_auth)):
    """Display the top-up setup page."""
    try:
        # Extract success/error messages from query parameters
        success_message = request.query_params.get("success")
        error_message = request.query_params.get("error")
        
        billing_account = await BillingService.get_billing_account(current_user.organization_id)
        if not billing_account:
            billing_account = await BillingService.create_billing_account_for_organization(
                current_user.organization_id
            )

        return templates.TemplateResponse(
            "billing/topup.html",
            get_template_context(
                request,
                billing_account=billing_account,
                success=success_message,
                error=error_message,
            ),
        )

    except Exception as e:
        logger.error(f"Error loading top-up page: {e}")
        return templates.TemplateResponse(
            "billing/topup.html",
            get_template_context(
                request,
                error="Error loading top-up settings",
                billing_account=None,
            ),
            status_code=500,
        )


@router.post("/billing/setup-topup")
async def setup_topup(
    request: Request,
    payment_method_id: str = Form(...),
    threshold_minutes: int = Form(10),
    topup_amount_minutes: int = Form(100),
    topup_price_cents: int = Form(500),
    current_user: User = Depends(require_auth)
):
    """Setup auto top-up."""
    try:
        success = await BillingService.setup_auto_topup(
            organization_id=current_user.organization_id,
            payment_method_id=payment_method_id,
            threshold_minutes=threshold_minutes,
            topup_amount_minutes=topup_amount_minutes,
            topup_price_cents=topup_price_cents
        )

        if success:
            return RedirectResponse(
                url="/billing?success=Auto top-up configured successfully!",
                status_code=302
            )
        else:
            return RedirectResponse(
                url="/billing?error=Failed to setup auto top-up",
                status_code=302
            )

    except stripe.error.CardError as e:
        logger.error(f"Card error during top-up setup: {e}")
        return RedirectResponse(
            url=f"/billing?error=Payment setup failed: {e.user_message}",
            status_code=302
        )
    except Exception as e:
        logger.error(f"Error setting up top-up: {e}")
        return RedirectResponse(
            url="/billing?error=An error occurred during top-up setup",
            status_code=302
        )


@router.post("/billing/disable-topup")
async def disable_topup(
    request: Request,
    current_user: User = Depends(require_auth)
):
    """Disable auto top-up."""
    try:
        billing_account = await BillingService.get_billing_account(current_user.organization_id)
        if billing_account:
            success = await BillingService.disable_auto_topup(billing_account.id)
            if success:
                return RedirectResponse(
                    url="/billing?success=Auto top-up disabled successfully!",
                    status_code=302
                )

        return RedirectResponse(
            url="/billing?error=Failed to disable auto top-up",
            status_code=302
        )

    except Exception as e:
        logger.error(f"Error disabling top-up: {e}")
        return RedirectResponse(
            url="/billing?error=An error occurred while disabling top-up",
            status_code=302
        )


@router.post("/billing/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        logger.error(f"Invalid payload in Stripe webhook: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature in Stripe webhook: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle the event
    try:
        if event["type"] == "invoice.payment_succeeded":
            # Handle successful subscription payment
            invoice = event["data"]["object"]
            logger.info(f"Payment succeeded for invoice: {invoice['id']}")
            
        elif event["type"] == "invoice.payment_failed":
            # Handle failed subscription payment
            invoice = event["data"]["object"]
            logger.error(f"Payment failed for invoice: {invoice['id']}")
            
        elif event["type"] == "customer.subscription.deleted":
            # Handle subscription cancellation
            subscription = event["data"]["object"]
            logger.info(f"Subscription cancelled: {subscription['id']}")
            
        elif event["type"] == "payment_intent.succeeded":
            # Handle successful one-time payment (like top-ups)
            payment_intent = event["data"]["object"]
            logger.info(f"Payment intent succeeded: {payment_intent['id']}")
            
        else:
            logger.info(f"Unhandled event type: {event['type']}")

    except Exception as e:
        logger.error(f"Error processing Stripe webhook: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")

    return JSONResponse(content={"status": "success"})


@router.get("/billing/usage", response_class=JSONResponse)
async def get_usage_api(request: Request, current_user: User = Depends(require_auth)):
    """API endpoint to get current usage information."""
    try:
        usage_summary = await BillingService.get_usage_summary(current_user.organization_id)
        return JSONResponse(content=usage_summary)
    except Exception as e:
        logger.error(f"Error getting usage summary: {e}")
        raise HTTPException(status_code=500, detail="Error fetching usage data") 