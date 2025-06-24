from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Templates setup
templates = Jinja2Templates(directory="app/templates")

# Router for docs (public, no authentication required)
router = APIRouter(tags=["docs"])


@router.get("/docs", response_class=HTMLResponse)
async def docs_page(request: Request):
    """
    API Documentation page - publicly accessible.
    
    Comprehensive documentation for the Burki Voice AI API,
    including authentication, endpoints, examples, and code samples.
    """
    return templates.TemplateResponse(
        "docs.html",
        {
            "request": request,
            "title": "API Documentation - Burki Voice AI"
        }
    )


@router.get("/api-reference", response_class=HTMLResponse)
async def api_reference_redirect(request: Request):
    """
    Redirect /api-reference to /docs for consistency.
    """
    return templates.TemplateResponse(
        "docs.html",
        {
            "request": request,
            "title": "API Reference - Burki Voice AI"
        }
    )


@router.get("/documentation", response_class=HTMLResponse)
async def documentation_redirect(request: Request):
    """
    Redirect /documentation to /docs for consistency.
    """
    return templates.TemplateResponse(
        "docs.html",
        {
            "request": request,
            "title": "Documentation - Burki Voice AI"
        }
    ) 