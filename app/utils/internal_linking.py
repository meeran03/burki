"""
Internal linking automation for SEO optimization
Automatically adds relevant internal links to improve keyword distribution and user experience
"""

import re
from typing import Dict, List, Tuple

class InternalLinkManager:
    """Manages automatic internal linking for SEO optimization"""
    
    def __init__(self):
        # Define link patterns with priority (higher = more important)
        self.link_patterns = {
            # Primary competitive keywords
            "vapi.ai": {
                "url": "/vs/vapi",
                "title": "Burki vs vapi.ai Comparison",
                "priority": 10,
                "variations": ["vapi", "vapi.ai", "Vapi"]
            },
            "voice AI": {
                "url": "/docs/voice-ai-guide", 
                "title": "Voice AI Platform Guide",
                "priority": 9,
                "variations": ["voice AI", "voice artificial intelligence", "AI voice"]
            },
            "open source": {
                "url": "/docs/open-source",
                "title": "Open Source Voice AI Benefits", 
                "priority": 8,
                "variations": ["open source", "open-source", "FOSS"]
            },
            
            # Platform features
            "assistants": {
                "url": "/assistants",
                "title": "Voice AI Assistants Dashboard",
                "priority": 7,
                "variations": ["assistants", "voice assistants", "AI assistants"]
            },
            "call analytics": {
                "url": "/calls",
                "title": "Call Analytics & Management",
                "priority": 7,
                "variations": ["call analytics", "call monitoring", "call management"]
            },
            "deployment": {
                "url": "/docs/quickstart",
                "title": "5-Minute Deployment Guide",
                "priority": 6,
                "variations": ["deployment", "deploy", "installation", "setup"]
            },
            
            # Technical features
            "RNNoise": {
                "url": "/docs/rnnoise",
                "title": "RNNoise Audio Denoising",
                "priority": 6,
                "variations": ["RNNoise", "audio denoising", "noise reduction"]
            },
            "multi-tenant": {
                "url": "/docs/architecture",
                "title": "Multi-Tenant Architecture",
                "priority": 5,
                "variations": ["multi-tenant", "multi tenant", "multitenancy"]
            },
            "latency": {
                "url": "/docs/performance",
                "title": "Ultra-Low Latency Performance",
                "priority": 8,
                "variations": ["latency", "response time", "speed"]
            },
            
            # API and docs
            "API": {
                "url": "/docs/api",
                "title": "Burki API Reference",
                "priority": 5,
                "variations": ["API", "REST API", "API reference"]
            },
            "webhooks": {
                "url": "/docs/webhooks",
                "title": "Webhook Integration Guide",
                "priority": 4,
                "variations": ["webhooks", "webhook", "callbacks"]
            }
        }
    
    def add_contextual_links(self, content: str, current_page: str = "", max_links: int = 5) -> str:
        """
        Add contextual internal links to content
        
        Args:
            content: The text content to process
            current_page: Current page URL to avoid self-linking
            max_links: Maximum number of links to add
            
        Returns:
            Content with added internal links
        """
        if not content:
            return content
        
        # Track added links to avoid duplicates
        added_links = set()
        link_count = 0
        
        # Sort patterns by priority (higher first)
        sorted_patterns = sorted(
            self.link_patterns.items(), 
            key=lambda x: x[1]['priority'], 
            reverse=True
        )
        
        for keyword, link_data in sorted_patterns:
            if link_count >= max_links:
                break
            
            # Skip if linking to current page
            if current_page and link_data['url'] in current_page:
                continue
            
            # Check all variations of the keyword
            for variation in link_data['variations']:
                if link_count >= max_links:
                    break
                
                # Create pattern that doesn't match already linked text
                pattern = rf'\b({re.escape(variation)})\b(?![^<]*>)'
                
                # Find first occurrence that's not already linked
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                
                if matches and variation.lower() not in added_links:
                    # Get first match
                    match = matches[0]
                    matched_text = match.group(1)
                    
                    # Create the link
                    link_html = f'<a href="{link_data["url"]}" title="{link_data["title"]}" class="text-blue-400 hover:text-blue-300 underline transition-colors">{matched_text}</a>'
                    
                    # Replace only the first occurrence
                    content = content[:match.start()] + link_html + content[match.end():]
                    
                    added_links.add(variation.lower())
                    link_count += 1
                    break
        
        return content
    
    def get_related_links(self, page_type: str, keyword: str = "") -> List[Dict]:
        """
        Get related internal links for a specific page
        
        Args:
            page_type: Type of page (landing, docs, assistants, etc.)
            keyword: Optional keyword for more targeted suggestions
            
        Returns:
            List of related link suggestions
        """
        related_links = []
        
        if page_type == "landing":
            related_links = [
                {"text": "Get Started in 5 Minutes", "url": "/docs/quickstart", "priority": 10},
                {"text": "Compare with vapi.ai", "url": "/vs/vapi", "priority": 9},
                {"text": "View Live Demo", "url": "https://burki.dev", "priority": 8},
                {"text": "Explore API Documentation", "url": "/docs/api", "priority": 7}
            ]
        elif page_type == "docs":
            related_links = [
                {"text": "API Reference", "url": "/docs/api", "priority": 9},
                {"text": "Deployment Guide", "url": "/docs/quickstart", "priority": 8},
                {"text": "Architecture Overview", "url": "/docs/architecture", "priority": 7},
                {"text": "Performance Guide", "url": "/docs/performance", "priority": 6}
            ]
        elif page_type == "assistants":
            related_links = [
                {"text": "Assistant Configuration", "url": "/docs/assistants", "priority": 9},
                {"text": "Voice AI Guide", "url": "/docs/voice-ai-guide", "priority": 8},
                {"text": "Call Management", "url": "/calls", "priority": 7}
            ]
        
        # Filter by keyword if provided
        if keyword:
            related_links = [
                link for link in related_links 
                if keyword.lower() in link["text"].lower()
            ]
        
        return sorted(related_links, key=lambda x: x["priority"], reverse=True)
    
    def generate_seo_footer_links(self) -> Dict[str, List[Dict]]:
        """Generate SEO-optimized footer links for better site structure"""
        return {
            "Platform": [
                {"text": "Voice AI Assistants", "url": "/assistants"},
                {"text": "Call Analytics", "url": "/calls"},
                {"text": "API Documentation", "url": "/docs/api"},
                {"text": "Live Demo", "url": "https://burki.dev"}
            ],
            "Comparisons": [
                {"text": "Burki vs vapi.ai", "url": "/vs/vapi"},
                {"text": "Open Source Benefits", "url": "/docs/open-source"},
                {"text": "Performance Comparison", "url": "/docs/performance"},
                {"text": "Feature Comparison", "url": "/docs/features"}
            ],
            "Resources": [
                {"text": "Quick Start Guide", "url": "/docs/quickstart"},
                {"text": "Architecture Guide", "url": "/docs/architecture"},
                {"text": "RNNoise Setup", "url": "/docs/rnnoise"},
                {"text": "Webhook Integration", "url": "/docs/webhooks"}
            ],
            "Community": [
                {"text": "GitHub Repository", "url": "https://github.com/meeran03/burki"},
                {"text": "Documentation", "url": "https://docs.burki.dev"},
                {"text": "Issue Tracker", "url": "https://github.com/meeran03/burki/issues"},
                {"text": "Contributing Guide", "url": "/docs/contributing"}
            ]
        }
    
    def validate_links(self, content: str) -> List[str]:
        """Validate internal links in content and return any issues"""
        issues = []
        
        # Find all internal links
        link_pattern = r'<a[^>]+href="(/[^"]*)"[^>]*>([^<]+)</a>'
        links = re.findall(link_pattern, content)
        
        for url, text in links:
            # Check for common issues
            if not text.strip():
                issues.append(f"Empty link text for URL: {url}")
            
            if len(text) > 100:
                issues.append(f"Link text too long: {text[:50]}...")
            
            if url.count('/') > 4:
                issues.append(f"Deep URL structure: {url}")
        
        return issues

# Helper function for templates
def add_internal_links(content: str, current_page: str = "") -> str:
    """Template helper function to add internal links"""
    link_manager = InternalLinkManager()
    return link_manager.add_contextual_links(content, current_page)

# Helper function for related links
def get_related_links(page_type: str, keyword: str = "") -> List[Dict]:
    """Template helper function to get related links"""
    link_manager = InternalLinkManager()
    return link_manager.get_related_links(page_type, keyword) 