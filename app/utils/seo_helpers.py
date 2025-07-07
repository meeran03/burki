"""
SEO helper functions for consistent optimization across all pages
"""

from typing import List, Dict, Optional
from urllib.parse import urljoin

class SEOHelper:
    """Helper class for consistent SEO implementation"""
    
    @staticmethod
    def generate_title(page_title: str, section: str = None) -> str:
        """Generate SEO-optimized page titles"""
        base_title = "Burki Voice AI"
        
        if section:
            return f"{page_title} - {section} | {base_title} - vapi.ai Alternative"
        else:
            return f"{page_title} | {base_title} - Open Source vapi.ai Alternative"
    
    @staticmethod
    def generate_meta_description(page_content: str, max_length: int = 155) -> str:
        """Generate SEO-optimized meta descriptions"""
        suffix = " | Burki Voice AI - 5x faster than vapi.ai"
        
        if len(page_content) + len(suffix) <= max_length:
            return page_content + suffix
        else:
            truncated = page_content[:max_length - len(suffix) - 3] + "..."
            return truncated + suffix
    
    @staticmethod
    def generate_breadcrumbs(current_page: str, parent_pages: List[Dict] = None) -> List[Dict]:
        """Generate breadcrumb navigation with proper schema"""
        breadcrumbs = [{"name": "Home", "url": "/"}]
        
        if parent_pages:
            breadcrumbs.extend(parent_pages)
        
        breadcrumbs.append({"name": current_page, "url": "#"})
        return breadcrumbs
    
    @staticmethod
    def validate_heading_structure(headings: List[Dict]) -> List[str]:
        """Validate proper H1-H6 hierarchy for SEO"""
        issues = []
        current_level = 0
        h1_count = 0
        
        for heading in headings:
            level = int(heading['tag'][1])  # Extract number from H1, H2, etc.
            
            if heading['tag'] == 'h1':
                h1_count += 1
                if h1_count > 1:
                    issues.append("Multiple H1 tags found - should have only one per page")
            
            if level > current_level + 1:
                issues.append(f"Heading hierarchy skipped from H{current_level} to H{level}")
            
            current_level = level
        
        if h1_count == 0:
            issues.append("No H1 tag found - required for SEO")
        
        return issues
    
    @staticmethod
    def generate_schema_breadcrumbs(breadcrumbs: List[Dict], base_url: str) -> Dict:
        """Generate schema.org breadcrumb markup"""
        schema_items = []
        
        for idx, item in enumerate(breadcrumbs, 1):
            schema_items.append({
                "@type": "ListItem",
                "position": idx,
                "name": item["name"],
                "item": urljoin(base_url, item["url"])
            })
        
        return {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": schema_items
        }
    
    @staticmethod
    def get_competitive_keywords() -> List[str]:
        """Return list of competitive keywords to target"""
        return [
            "vapi.ai alternative",
            "open source voice AI",
            "voice AI platform", 
            "conversational AI infrastructure",
            "self-hosted voice AI",
            "low latency voice AI",
            "voice AI with RNNoise",
            "multi-tenant voice AI",
            "voice assistant platform",
            "AI phone system",
            "voice bot platform",
            "conversational AI solution"
        ]
    
    @staticmethod
    def generate_comparison_schema(competitor: str, advantages: List[str]) -> Dict:
        """Generate schema for competitor comparison pages"""
        return {
            "@context": "https://schema.org",
            "@type": "Comparison",
            "name": f"Burki vs {competitor}",
            "description": f"Detailed comparison between Burki Voice AI and {competitor}",
            "comparedItem": [
                {
                    "@type": "Product",
                    "name": "Burki Voice AI",
                    "description": "Open-source voice AI platform",
                    "offers": {
                        "@type": "Offer",
                        "price": "0",
                        "priceCurrency": "USD"
                    }
                },
                {
                    "@type": "Product", 
                    "name": competitor,
                    "description": f"Proprietary voice AI platform"
                }
            ],
            "advantages": advantages
        }

def get_page_seo_data(page_name: str, custom_title: str = None, custom_description: str = None) -> Dict:
    """Get complete SEO data for a page"""
    seo = SEOHelper()
    
    # Define page-specific SEO data
    page_configs = {
        "landing": {
            "title": "Open Source Alternative to vapi.ai | 5x Faster Voice AI",
            "description": "The vapi.ai alternative that actually works. 0.8-1.2s latency vs 4-5s, transparent pricing, and a UI that works. Build voice AI assistants without the frustrations.",
            "keywords": ["vapi.ai alternative", "open source voice AI", "voice AI platform"],
            "schema_type": "WebSite"
        },
        "assistants": {
            "title": "Voice AI Assistants Dashboard",
            "description": "Create and manage AI voice assistants with Burki's intuitive dashboard. Deploy faster than vapi.ai with better performance.",
            "keywords": ["voice assistants", "AI assistant management", "voice bot dashboard"],
            "schema_type": "WebApplication"
        },
        "calls": {
            "title": "Call Analytics & Management",
            "description": "Monitor voice AI calls with real-time analytics, call recordings, and performance metrics. Better insights than vapi.ai.",
            "keywords": ["call analytics", "voice AI monitoring", "call management"],
            "schema_type": "WebApplication"
        },
        "docs": {
            "title": "Documentation - Getting Started Guide",
            "description": "Complete Burki Voice AI documentation. Deploy in 5 minutes vs weeks with vapi.ai. Comprehensive guides and API reference.",
            "keywords": ["voice AI documentation", "Burki guide", "API reference"],
            "schema_type": "TechArticle"
        }
    }
    
    config = page_configs.get(page_name, {})
    
    return {
        "title": custom_title or config.get("title", seo.generate_title(page_name.title())),
        "description": custom_description or config.get("description", seo.generate_meta_description(f"Burki Voice AI {page_name}")),
        "keywords": config.get("keywords", seo.get_competitive_keywords()[:5]),
        "schema_type": config.get("schema_type", "WebPage"),
        "breadcrumbs": seo.generate_breadcrumbs(page_name.title())
    } 