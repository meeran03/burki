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
from app.api.web.billing import router as web_billing_router
from app.api.web.docs import router as web_docs_router
from app.api.root import router as root_router
from app.services.billing_service import BillingService

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
app.include_router(web_billing_router)
app.include_router(web_docs_router)

# API routers - no additional prefix needed since they include full path
app.include_router(assistants_router)
app.include_router(calls_router)

# Initialize state and handlers
call_manager = CallManager()


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    # Run database migrations first
    try:
        run_migrations()
    except Exception as e:
        logger.error("Failed to run database migrations: %s", e)
        # You might want to decide if you want to continue startup without migrations
        # For now, we'll continue but log the error
    
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


# Server configuration from environment variables
workers = int(os.getenv("WORKERS", "2"))
worker_class = "uvicorn.workers.UvicornWorker"
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8000')}"
timeout = int(os.getenv("TIMEOUT", "300"))
keepalive = int(os.getenv("KEEP_ALIVE", "2"))
worker_connections = int(os.getenv("WORKER_CONNECTIONS", "1000"))
max_requests = int(os.getenv("MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", "100"))
log_level = os.getenv("LOG_LEVEL", "info")


def run_with_gunicorn():
    """Run the application using Gunicorn."""
    import gunicorn.app.base

    class StandaloneApplication(gunicorn.app.base.BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    options = {
        "bind": bind,
        "workers": workers,
        "worker_class": worker_class,
        "timeout": timeout,
        "keepalive": keepalive,
        "worker_connections": worker_connections,
        "max_requests": max_requests,
        "max_requests_jitter": max_requests_jitter,
        "loglevel": log_level,
    }

    StandaloneApplication(app, options).run()


if __name__ == "__main__":
    server_type = os.getenv("SERVER_TYPE", "uvicorn").lower()
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting server with {server_type}")
    logger.info(f"Server will run on {host}:{port}")
    
    if server_type == "gunicorn":
        logger.info("Starting application with Gunicorn...")
        run_with_gunicorn()
    else:
        # Run with uvicorn (following the web examples pattern)
        logger.info("Starting application with Uvicorn...")
        import uvicorn
        uvicorn.run(
            app,  # Pass the app directly instead of string reference
            host=host,
            port=port,
            reload=debug,
            log_level=log_level,
            proxy_headers=True  # Following the blog example
        )
