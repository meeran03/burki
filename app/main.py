"""
Entry Point for Application
"""

# pylint: disable=logging-format-interpolation,logging-fstring-interpolation,broad-exception-caught
import logging
import os
import subprocess
import sys

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from fastapi.openapi.utils import get_openapi

from app.core.call_manager import CallManager
from app.core.assistant_manager import assistant_manager
from app.api.assistants import router as assistants_router
from app.api.calls import router as calls_router
from app.api.web.index import router as web_router
from app.api.web.auth import router as web_auth_router
from app.api.web.assistant import router as web_assistant_router
from app.api.web.call import router as web_call_router
from app.api.web.billing import router as web_billing_router
from app.api.web.docs import router as web_docs_router
from app.api.root import router as root_router
from app.services.billing_service import BillingService

# Optional prettier reference docs
from app.api.documents import router as documents_router
# Import new reference router that we'll create
try:
    from app.api.docs_reference import router as reference_router
except ImportError:
    reference_router = None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_migrations():
    """Run database migrations using Alembic."""
    try:
        logger.info("Running database migrations...")
        result = subprocess.run([
            sys.executable, "-m", "alembic", "upgrade", "head"
        ], capture_output=True, text=True, check=True)
        logger.info("Database migrations completed successfully")
        if result.stdout:
            logger.info("Migration output: %s", result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error("Migration failed with return code %s", e.returncode)
        logger.error("Error output: %s", e.stderr)
        raise
    except Exception as e:
        logger.error("Error running migrations: %s", e)
        raise


app = FastAPI(
    title="Buraaq Voice-AI API",
    description="REST API for managing assistants, calls, documents and billing in the Buraaq Voice-AI platform.\n\nAuthenticate with an API key by setting the `Authorization` header to `Bearer &lt;your_key&gt;`. The **Reference** tab in the navigation provides an interactive explorer.",
    version="0.1.0",
    swagger_ui_parameters={"persistAuthorization": True, "defaultModelsExpandDepth": -1},
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add session middleware
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Health check endpoint for production monitoring
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "healthy", "service": "buraaq-voice-ai"}

# Include routers
app.include_router(root_router)
app.include_router(web_router, include_in_schema=False)
app.include_router(web_auth_router, include_in_schema=False)
app.include_router(web_assistant_router, include_in_schema=False)
app.include_router(web_call_router, include_in_schema=False)
app.include_router(web_billing_router, include_in_schema=False)
app.include_router(web_docs_router, include_in_schema=False)

# Include static/docs routers
app.include_router(web_docs_router, include_in_schema=False)
if reference_router:
    app.include_router(reference_router, include_in_schema=False)

# New API sub-routers
app.include_router(documents_router)

# API routers - no additional prefix needed since they include full path
app.include_router(assistants_router)
app.include_router(calls_router)

# Initialize state and handlers
call_manager = CallManager()


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    # Load active assistants
    try:
        await assistant_manager.load_assistants()
    except Exception as e:  # pylint: disable=
        logger.error("Error loading assistants: %s", e, exc_info=True)

    # Initialize billing service
    try:
        await BillingService.initialize_default_plans()
        logger.info("Billing service initialized successfully")
    except Exception as e:
        logger.error("Error initializing billing service: %s", e, exc_info=True)

    logger.info("Application startup complete")

# --- Custom OpenAPI with ApiKeyAuth security -----------------

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add ApiKeyAuth if not present
    components = openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})
    if "ApiKeyAuth" not in components:
        components["ApiKeyAuth"] = {
            "type": "apiKey",
            "in": "header",
            "name": "Authorization",
        }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
