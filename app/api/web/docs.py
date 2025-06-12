from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Templates setup
templates = Jinja2Templates(directory="app/templates")

# Router for docs (public, no authentication required)
router = APIRouter(tags=["docs"])
