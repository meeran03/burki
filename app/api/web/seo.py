from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, Response
from datetime import datetime

router = APIRouter()

@router.get("/robots.txt", response_class=PlainTextResponse)
async def robots_txt():
    """Serve robots.txt for SEO crawling instructions"""
    robots_content = """User-agent: *
Allow: /

# Important pages for search engines
Allow: /docs
Allow: /static/
Allow: /auth/register
Allow: /auth/login

# Block admin and API endpoints from indexing
Disallow: /admin/
Disallow: /api/
Disallow: /auth/logout
Disallow: /webhooks/

# Allow all static assets
Allow: /static/css/
Allow: /static/logo/
Allow: /static/js/

# Sitemap location
Sitemap: https://burki.dev/sitemap.xml

# Crawl delay (optional - helps with server load)
Crawl-delay: 1"""
    
    return robots_content

@router.get("/sitemap.xml")
async def sitemap_xml(request: Request):
    """Generate XML sitemap for SEO"""
    base_url = str(request.base_url).rstrip('/')
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Define your important pages
    pages = [
        {"url": "/", "priority": "1.0", "changefreq": "weekly"},
        {"url": "/docs", "priority": "0.9", "changefreq": "monthly"},
        {"url": "/auth/register", "priority": "0.8", "changefreq": "monthly"},
        {"url": "/auth/login", "priority": "0.7", "changefreq": "monthly"},
        {"url": "/assistants", "priority": "0.8", "changefreq": "weekly"},
        {"url": "/calls", "priority": "0.8", "changefreq": "weekly"},
        {"url": "/phone_numbers", "priority": "0.7", "changefreq": "monthly"},
    ]
    
    sitemap_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">"""
    
    for page in pages:
        sitemap_xml += f"""
    <url>
        <loc>{base_url}{page['url']}</loc>
        <lastmod>{current_date}</lastmod>
        <changefreq>{page['changefreq']}</changefreq>
        <priority>{page['priority']}</priority>
    </url>"""
    
    sitemap_xml += """
</urlset>"""
    
    return Response(
        content=sitemap_xml,
        media_type="application/xml",
        headers={"Content-Type": "application/xml; charset=utf-8"}
    ) 