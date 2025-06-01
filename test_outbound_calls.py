#!/usr/bin/env python3
"""
Test script for outbound calls functionality.

This script demonstrates how to use the outbound calls API endpoint.
"""

import json
import asyncio
import httpx

async def test_outbound_call():
    """Test the outbound calls API endpoint."""
    
    # Configuration
    base_url = "http://localhost:8000"  # Adjust to your server URL
    endpoint = f"{base_url}/calls/initiate"
    
    # Test data
    test_payload = {
        "assistant_id": 1,  # Replace with a valid assistant ID
        "to_phone_number": "+1234567890",  # Replace with a valid phone number
        "welcome_message": "Hello! This is your AI assistant calling from Diwaar. I hope you're having a great day!",
        "agenda": "I'm calling to follow up on your recent inquiry about our AI voice assistant services. I'd like to discuss how we can help improve your customer service with automated phone support."
    }
    
    print("🚀 Testing Outbound Calls API")
    print("=" * 50)
    print(f"Endpoint: {endpoint}")
    print(f"Payload: {json.dumps(test_payload, indent=2)}")
    print()
    
    try:
        async with httpx.AsyncClient() as client:
            print("📞 Initiating outbound call...")
            response = await client.post(
                endpoint,
                json=test_payload,
                headers={"Content-Type": "application/json"},
                timeout=30.0
            )
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Success!")
                print(f"Response: {json.dumps(result, indent=2)}")
                print(f"Call SID: {result.get('call_sid', 'N/A')}")
            else:
                print("❌ Error!")
                print(f"Error Response: {response.text}")
                
    except httpx.RequestError as e:
        print(f"❌ Request Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")


def test_phone_validation():
    """Test phone number validation."""
    from app.twilio.twilio_service import TwilioService
    
    print("\n📱 Testing Phone Number Validation")
    print("=" * 40)
    
    test_numbers = [
        "+1234567890",      # Valid US number
        "+447123456789",    # Valid UK number  
        "+8612345678901",   # Valid China number
        "1234567890",       # Invalid (no +)
        "+123",             # Invalid (too short)
        "+123456789012345678",  # Invalid (too long)
        "",                 # Invalid (empty)
        None,               # Invalid (None)
    ]
    
    for number in test_numbers:
        is_valid = TwilioService.validate_phone_number(str(number) if number else "")
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"{number:<20} -> {status}")


def test_webhook_url():
    """Test webhook URL generation."""
    try:
        from app.utils.url_utils import get_twiml_webhook_url, get_server_base_url
        
        print("\n🔗 Testing Webhook URL Generation")
        print("=" * 40)
        
        base_url = get_server_base_url()
        print(f"Base URL: {base_url}")
        
        webhook_url = get_twiml_webhook_url()
        print(f"TwiML Webhook URL: {webhook_url}")
        
        # Validate URL format
        if webhook_url.startswith(('http://', 'https://')) and webhook_url.endswith('/twiml'):
            print("✅ Webhook URL format is valid")
        else:
            print("❌ Webhook URL format is invalid")
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you're running this from the project root directory")
    except Exception as e:
        print(f"❌ Error testing webhook URL: {e}")


if __name__ == "__main__":
    print("🔧 Outbound Calls Test Suite")
    print("=" * 60)
    
    # Test webhook URL generation
    test_webhook_url()
    
    # Test phone validation
    test_phone_validation()
    
    # Test API call
    print("\n" + "=" * 60)
    asyncio.run(test_outbound_call())
    
    print("\n" + "=" * 60)
    print("📝 Test Instructions:")
    print("1. Make sure your server is running on localhost:8000")
    print("2. Update the assistant_id and to_phone_number in the test payload")
    print("3. Ensure you have valid Twilio credentials configured")
    print("4. The assistant should have a valid phone number and Twilio credentials")
    print("5. Check your server logs to see the outbound call processing")
    print("6. You can also test from the frontend by visiting /assistants/{id} and using the 'Test Outbound Call' form")
    print("\n💡 Frontend Testing:")
    print("- Navigate to an assistant's view page")
    print("- Scroll to the 'Test Outbound Call' section")
    print("- Fill in phone number and agenda")
    print("- Click 'Initiate Test Call' to test from the UI") 