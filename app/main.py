"""
Entry Point for Application
"""

# pylint: disable=logging-format-interpolation,logging-fstring-interpolation,broad-exception-caught
import logging
import os
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
from app.api.root import router as root_router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Diwaar",
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

app.include_router(root_router)
app.include_router(web_router)
app.include_router(web_auth_router)
app.include_router(web_assistant_router)
app.include_router(web_call_router)
app.include_router(assistants_router, prefix="/api")
app.include_router(calls_router, prefix="/api")

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
