#!/usr/bin/env python3
"""
DNS Verification Checker for Google Search Console
Checks if your TXT record is properly configured
"""

import subprocess
import sys
import re
from urllib.parse import urlparse

def check_dns_txt_record(domain, verification_code):
    """Check if Google verification TXT record exists for domain"""
    print(f"🔍 Checking DNS TXT records for {domain}...")
    
    try:
        # Use dig or nslookup to check TXT records
        try:
            # Try dig first (more detailed)
            result = subprocess.run(['dig', '+short', 'TXT', domain], 
                                 capture_output=True, text=True, timeout=10)
            output = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback to nslookup
            result = subprocess.run(['nslookup', '-type=TXT', domain], 
                                 capture_output=True, text=True, timeout=10)
            output = result.stdout
        
        if not output:
            print("❌ No TXT records found")
            return False
        
        print("📋 Found TXT records:")
        
        # Look for Google verification records
        verification_found = False
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'google-site-verification' in line.lower():
                print(f"   ✅ {line}")
                if verification_code and verification_code in line:
                    verification_found = True
                    print(f"   🎉 Your verification code FOUND!")
            elif line and '"' in line:
                print(f"   📝 {line}")
        
        if verification_code:
            if verification_found:
                print(f"\n✅ SUCCESS: Verification code {verification_code[:10]}... is properly configured!")
                return True
            else:
                print(f"\n❌ ISSUE: Verification code {verification_code[:10]}... not found in DNS")
                print("💡 Make sure you:")
                print("   1. Added the TXT record correctly")
                print("   2. Used @ or blank for the record name") 
                print("   3. Waited 15-60 minutes for DNS propagation")
                return False
        else:
            return len([l for l in lines if 'google-site-verification' in l.lower()]) > 0
            
    except subprocess.TimeoutExpired:
        print("❌ DNS lookup timed out")
        return False
    except Exception as e:
        print(f"❌ Error checking DNS: {e}")
        return False

def extract_domain_from_url(url):
    """Extract domain from URL"""
    if not url.startswith(('http://', 'https://')):
        return url
    
    parsed = urlparse(url)
    return parsed.netloc

def main():
    print("🌐 Google Search Console DNS Verification Checker")
    print("=" * 50)
    
    # Get domain from user
    if len(sys.argv) > 1:
        domain_input = sys.argv[1]
    else:
        domain_input = input("Enter your domain (e.g., burki.dev or https://burki.dev): ").strip()
    
    domain = extract_domain_from_url(domain_input)
    
    # Get verification code if provided
    if len(sys.argv) > 2:
        verification_code = sys.argv[2]
    else:
        verification_code = input("Enter your verification code (optional): ").strip()
        # Clean up if they pasted the full string
        verification_code = verification_code.replace('google-site-verification=', '')
    
    print(f"\n🔍 Checking domain: {domain}")
    if verification_code:
        print(f"🔑 Looking for code: {verification_code[:10]}...")
    
    success = check_dns_txt_record(domain, verification_code)
    
    if success:
        print(f"\n🎉 READY TO VERIFY!")
        print("Go back to Google Search Console and click 'Verify'")
    else:
        print(f"\n⏳ NOT READY YET")
        print("Please check your DNS settings and try again in 15-30 minutes")
        
    print(f"\n💡 You can also check manually at:")
    print(f"   https://toolbox.googleapps.com/apps/dig/#TXT/{domain}")

if __name__ == "__main__":
    main() 