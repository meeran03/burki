from fastapi import APIRouter, HTMLResponse

router = APIRouter(tags=["reference"])

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"UTF-8\" />
    <title>Buraaq Voice-AI API Reference</title>
    <script type=\"module\" src=\"https://unpkg.com/rapidoc/dist/rapidoc-min.js\"></script>
    <style>
      body { margin: 0; }
      rapi-doc::part(section) { font-family: 'Inter', sans-serif; }
    </style>
  </head>
  <body>
    <rapi-doc
      spec-url=\"/openapi.json\"
      show-header=\"true\"
      render-style=\"read\"
      theme=\"light\"
      primary-color=\"#2563eb\"
      show-method-in-nav-bar=\"as-colored-block\"
      allow-authentication=\"true\"
      allow-spec-url-load=\"false\"
      persist-auth=\"true\"
    >
      <div slot=\"nav-logo\">
        <img src=\"/static/logo.svg\" alt=\"logo\" height=\"32\" />
      </div>
    </rapi-doc>
  </body>
</html>
"""

@router.get("/reference", response_class=HTMLResponse)
async def api_reference_page():
    """Serve interactive API reference powered by RapiDoc."""
    return HTML_TEMPLATE 