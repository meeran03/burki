#!/usr/bin/env python3
"""
SEO Validation Test Suite for Burki Voice AI
Tests all SEO implementations for compliance and performance
"""

import requests
import json
import sys
from urllib.parse import urljoin
from bs4 import BeautifulSoup

class SEOTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = {"passed": 0, "failed": 0, "warnings": 0}
    
    def test_robots_txt(self):
        """Test robots.txt accessibility and content"""
        print("🤖 Testing robots.txt...")
        try:
            response = self.session.get(f"{self.base_url}/robots.txt")
            if response.status_code == 200:
                if "Sitemap:" in response.text:
                    print("✅ robots.txt valid with sitemap")
                    self.results["passed"] += 1
                else:
                    print("⚠️  robots.txt missing sitemap reference")
                    self.results["warnings"] += 1
            else:
                print("❌ robots.txt not accessible")
                self.results["failed"] += 1
        except Exception as e:
            print(f"❌ robots.txt test failed: {e}")
            self.results["failed"] += 1
    
    def test_sitemap_xml(self):
        """Test XML sitemap accessibility and structure"""
        print("🗺️  Testing sitemap.xml...")
        try:
            response = self.session.get(f"{self.base_url}/sitemap.xml")
            if response.status_code == 200:
                if "<?xml" in response.text and "<urlset" in response.text:
                    print("✅ sitemap.xml valid XML structure")
                    self.results["passed"] += 1
                else:
                    print("❌ sitemap.xml invalid format")
                    self.results["failed"] += 1
            else:
                print("❌ sitemap.xml not accessible")
                self.results["failed"] += 1
        except Exception as e:
            print(f"❌ sitemap.xml test failed: {e}")
            self.results["failed"] += 1
    
    def test_manifest_json(self):
        """Test PWA manifest"""
        print("📱 Testing manifest.json...")
        try:
            response = self.session.get(f"{self.base_url}/static/manifest.json")
            if response.status_code == 200:
                manifest = json.loads(response.text)
                required_fields = ["name", "short_name", "start_url", "display", "theme_color"]
                missing_fields = [field for field in required_fields if field not in manifest]
                
                if not missing_fields:
                    print("✅ manifest.json complete")
                    self.results["passed"] += 1
                else:
                    print(f"⚠️  manifest.json missing: {missing_fields}")
                    self.results["warnings"] += 1
            else:
                print("❌ manifest.json not accessible")
                self.results["failed"] += 1
        except Exception as e:
            print(f"❌ manifest.json test failed: {e}")
            self.results["failed"] += 1
    
    def test_home_page_seo(self):
        """Test homepage SEO elements"""
        print("🏠 Testing homepage SEO...")
        try:
            response = self.session.get(self.base_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Test critical SEO elements
                title = soup.find('title')
                description = soup.find('meta', attrs={'name': 'description'})
                h1 = soup.find('h1')
                canonical = soup.find('link', attrs={'rel': 'canonical'})
                schema = soup.find_all('script', attrs={'type': 'application/ld+json'})
                
                checks = [
                    (title and "vapi.ai" in title.text.lower(), "Title contains competitive keywords"),
                    (description and len(description.get('content', '')) > 120, "Meta description adequate length"),
                    (h1 is not None, "H1 tag present"),
                    (canonical is not None, "Canonical URL set"),
                    (len(schema) > 0, "Schema markup present"),
                    (soup.find('meta', attrs={'property': 'og:title'}), "Open Graph tags present"),
                ]
                
                passed_checks = sum(1 for check, _ in checks if check)
                
                print(f"✅ Homepage SEO: {passed_checks}/{len(checks)} checks passed")
                
                if passed_checks == len(checks):
                    self.results["passed"] += 1
                elif passed_checks >= len(checks) * 0.8:
                    self.results["warnings"] += 1
                else:
                    self.results["failed"] += 1
                    
                # Print failed checks
                for check, description in checks:
                    if not check:
                        print(f"   ❌ {description}")
                        
            else:
                print("❌ Homepage not accessible")
                self.results["failed"] += 1
        except Exception as e:
            print(f"❌ Homepage SEO test failed: {e}")
            self.results["failed"] += 1
    
    def run_all_tests(self):
        """Run all SEO tests"""
        print("🚀 Starting SEO Test Suite for Burki Voice AI\n")
        
        self.test_robots_txt()
        self.test_sitemap_xml() 
        self.test_manifest_json()
        self.test_home_page_seo()
        
        print(f"\n📊 SEO Test Results:")
        print(f"✅ Passed: {self.results['passed']}")
        print(f"⚠️  Warnings: {self.results['warnings']}")
        print(f"❌ Failed: {self.results['failed']}")
        
        if self.results['failed'] == 0:
            print("\n🎉 All critical SEO tests passed!")
            return True
        else:
            print(f"\n❌ {self.results['failed']} critical issues found")
            return False

if __name__ == "__main__":
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    tester = SEOTester(base_url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)
