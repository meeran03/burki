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

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

from app.core.call_manager import CallManager
from app.core.assistant_manager import assistant_manager
from app.api.assistants import router as assistants_router
from app.api.calls import router as calls_router
from app.api.web.index import router as web_router
from app.api.web.auth import router as web_auth_router
from app.api.web.assistant import router as web_assistant_router
from app.api.web.call import router as web_call_router
from app.api.web.docs import router as web_docs_router
from app.api.web.phone_numbers import router as web_phone_numbers_router
from app.api.root import router as root_router

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
    title="Burki",
    description="A system that uses AI to answer customer Calls.",
    version="0.1.0",
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
app.include_router(web_router)
app.include_router(web_auth_router)
app.include_router(web_assistant_router)
app.include_router(web_call_router)
app.include_router(web_docs_router)
app.include_router(web_phone_numbers_router)

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

    logger.info("Application startup complete")
