#!/bin/bash

# Burki Voice AI - Analytics Setup Script
# Helps you configure Google Analytics and Search Console

echo "🚀 Setting up Analytics for Burki Voice AI"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.example .env
fi

echo -e "\n${BLUE}📊 GOOGLE ANALYTICS 4 SETUP${NC}"
echo "=============================================="
echo "1. Visit: https://analytics.google.com"
echo "2. Create account: 'Burki Voice AI'"
echo "3. Create property: 'Burki Voice AI Platform'"
echo "4. Choose 'Web' platform"
echo "5. Add your website URL"
echo "6. Copy your Measurement ID (starts with G-)"
echo ""

read -p "Enter your Google Analytics Measurement ID (G-XXXXXXXXXX): " GA_ID

if [[ $GA_ID =~ ^G-[A-Z0-9]+$ ]]; then
    # Update .env file
    if grep -q "GOOGLE_ANALYTICS_ID" .env; then
        sed -i.bak "s/GOOGLE_ANALYTICS_ID=.*/GOOGLE_ANALYTICS_ID=$GA_ID/" .env
    else
        echo "GOOGLE_ANALYTICS_ID=$GA_ID" >> .env
    fi
    echo -e "${GREEN}✅ Google Analytics ID saved!${NC}"
else
    echo -e "${YELLOW}⚠️  Invalid format. Should be G-XXXXXXXXXX${NC}"
fi

echo -e "\n${BLUE}🔍 GOOGLE SEARCH CONSOLE SETUP${NC}"
echo "=============================================="
echo "1. Visit: https://search.google.com/search-console"
echo "2. Click 'Add Property'"
echo ""
echo "Choose your verification method:"
echo "A) Domain (covers all subdomains) - DNS verification"
echo "B) URL prefix (single URL) - HTML tag verification"
echo ""
read -p "Which method did you choose? (A/B): " VERIFICATION_METHOD

if [[ $VERIFICATION_METHOD =~ ^[Aa]$ ]]; then
    echo ""
    echo "📋 DOMAIN VERIFICATION STEPS:"
    echo "1. Copy the TXT record value from Google Search Console"
    echo "2. Go to your DNS provider (GoDaddy, Cloudflare, etc.)"
    echo "3. Add TXT record:"
    echo "   - Name: @ (or leave blank)"
    echo "   - Value: google-site-verification=YOUR_CODE"
    echo "4. Wait 15-60 minutes for DNS propagation"
    echo "5. Return to Search Console and click 'Verify'"
    echo ""
    read -p "Enter your DNS verification code (google-site-verification=YOUR_CODE): " SC_CODE
    
    # Extract just the verification code part if they pasted the full string
    SC_CODE=$(echo "$SC_CODE" | sed 's/google-site-verification=//')
    
else
    echo ""
    echo "📋 URL PREFIX VERIFICATION STEPS:"
    echo "1. Choose 'HTML tag' verification method"
    echo "2. Copy only the content value from the meta tag"
    echo "   Example: <meta name=\"google-site-verification\" content=\"ABC123\">"
    echo "   You want: ABC123"
    echo ""
    read -p "Enter your HTML verification code: " SC_CODE
fi

if [[ ! -z "$SC_CODE" ]]; then
    # Update .env file
    if grep -q "GOOGLE_SITE_VERIFICATION" .env; then
        sed -i.bak "s/GOOGLE_SITE_VERIFICATION=.*/GOOGLE_SITE_VERIFICATION=$SC_CODE/" .env
    else
        echo "GOOGLE_SITE_VERIFICATION=$SC_CODE" >> .env
    fi
    echo -e "${GREEN}✅ Search Console verification saved!${NC}"
else
    echo -e "${YELLOW}⚠️  No verification code entered${NC}"
fi

echo -e "\n${GREEN}🎉 ANALYTICS SETUP COMPLETE!${NC}"
echo "=============================================="
echo "Your .env file now contains:"

if [[ ! -z "$GA_ID" ]]; then
    echo "✅ Google Analytics: $GA_ID"
fi

if [[ ! -z "$SC_CODE" ]]; then
    echo "✅ Search Console: $SC_CODE"
fi

echo ""
echo -e "${BLUE}🚀 NEXT STEPS:${NC}"
echo "1. Start your server: uvicorn app.main:app --reload"
echo "2. Visit Google Search Console and click 'Verify'"
echo "3. Check Google Analytics Real-Time reports"
echo ""
echo -e "${YELLOW}💡 PRO TIPS:${NC}"
echo "• Analytics data appears within 24-48 hours"
echo "• Search Console data appears within 1-3 days"
echo "• Use our performance monitor to track Core Web Vitals"
echo "• Monitor 'vapi.ai alternative' keyword rankings"

# Create analytics verification script
cat > "scripts/verify_analytics.py" << 'EOF'
#!/usr/bin/env python3
"""
Analytics Verification Script
Checks if Google Analytics and Search Console are working
"""

import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

def check_analytics_setup():
    load_dotenv()
    
    ga_id = os.getenv('GOOGLE_ANALYTICS_ID')
    sc_code = os.getenv('GOOGLE_SITE_VERIFICATION')
    
    print("🔍 Checking Analytics Configuration...")
    print("=====================================")
    
    # Check environment variables
    if ga_id:
        print(f"✅ Google Analytics ID: {ga_id}")
    else:
        print("❌ Google Analytics ID not found in .env")
    
    if sc_code:
        print(f"✅ Search Console Code: {sc_code[:10]}...")
    else:
        print("❌ Search Console verification not found in .env")
    
    # Test local server
    try:
        response = requests.get('http://localhost:8000', timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check for GA4 script
            ga_script = soup.find('script', src=lambda x: x and 'googletagmanager.com/gtag/js' in x)
            if ga_script:
                print("✅ Google Analytics script found on page")
            else:
                print("❌ Google Analytics script not found")
            
            # Check for Search Console verification
            sc_meta = soup.find('meta', attrs={'name': 'google-site-verification'})
            if sc_meta:
                print("✅ Search Console verification meta tag found")
            else:
                print("❌ Search Console verification meta tag not found")
            
            print("\n🎉 Analytics setup verification complete!")
            
        else:
            print(f"❌ Server not responding (status: {response.status_code})")
            
    except requests.exceptions.RequestException:
        print("❌ Server not running. Start with: uvicorn app.main:app --reload")

if __name__ == "__main__":
    check_analytics_setup()
EOF

chmod +x scripts/verify_analytics.py

echo -e "\n${BLUE}📊 VERIFICATION SCRIPT CREATED${NC}"
echo "Run 'python scripts/verify_analytics.py' to test your setup!"

# Clean up backup files
rm -f .env.bak

echo -e "\n${GREEN}🏆 Ready to dominate search results!${NC}" 