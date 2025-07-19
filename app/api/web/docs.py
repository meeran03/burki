from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Templates setup
templates = Jinja2Templates(directory="app/templates")

# Router for docs (public, no authentication required)
router = APIRouter(tags=["docs"])


@router.get("/docs")
async def docs_page(request: Request):
    """
    Redirect to external documentation at docs.burki.dev
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="https://docs.burki.dev", status_code=301)


@router.get("/api-reference")
async def api_reference_redirect(request: Request):
    """
    Redirect /api-reference to external docs.
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="https://docs.burki.dev", status_code=301)


@router.get("/documentation")
async def documentation_redirect(request: Request):
    """
    Redirect /documentation to external docs.
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="https://docs.burki.dev", status_code=301) 