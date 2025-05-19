"""
Entry Point for Application
"""

# pylint: disable=logging-format-interpolation,logging-fstring-interpolation,broad-exception-caught
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from app.core.call_manager import CallManager
from app.core.assistant_manager import assistant_manager
from app.db.database import get_db
from app.api.assistants import router as assistants_router
from app.api.calls import router as calls_router
from app.api.web import router as web_router
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

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


app.include_router(root_router)
app.include_router(web_router)
app.include_router(assistants_router, prefix="/api")
app.include_router(calls_router, prefix="/api")

# Initialize state and handlers
call_manager = CallManager()


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    # Load active assistants
    db = next(get_db())
    try:
        await assistant_manager.load_assistants(db)
    except Exception as e:  # pylint: disable=
        logger.error("Error loading assistants: %s", e, exc_info=True)
    finally:
        db.close()

    logger.info("Application startup complete")
