#!/usr/bin/env python3
"""
Example script demonstrating the Diwaar Voice AI API usage.

This script shows how to:
1. Create an assistant
2. List assistants
3. Update an assistant
4. Get call statistics
5. Handle authentication properly

Prerequisites:
- pip install requests
- Get an API key from the web interface
"""

import requests
import json
from typing import Dict, Any, Optional

class DiwaarAPI:
    """Simple client for Diwaar Voice AI API."""
    
    def __init__(self, base_url: str, api_key: str):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API (e.g., "https://yourdomain.com")
            api_key: Your API key from the web interface
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an API request."""
        url = f"{self.base_url}/api/v1{endpoint}"
        
        try:
            response = requests.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            if response.content:
                try:
                    error_detail = response.json()
                    print(f"Error details: {error_detail}")
                except:
                    print(f"Response content: {response.content}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            raise
    
    # Assistant endpoints
    def create_assistant(self, assistant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new assistant."""
        return self._request("POST", "/assistants/", json=assistant_data)
    
    def list_assistants(self, active_only: bool = False, my_assistants_only: bool = False, 
                       limit: int = 100) -> list:
        """List assistants."""
        params = {
            "active_only": active_only,
            "my_assistants_only": my_assistants_only,
            "limit": limit
        }
        return self._request("GET", "/assistants/", params=params)
    
    def get_assistant(self, assistant_id: int) -> Dict[str, Any]:
        """Get an assistant by ID."""
        return self._request("GET", f"/assistants/{assistant_id}")
    
    def update_assistant(self, assistant_id: int, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an assistant."""
        return self._request("PUT", f"/assistants/{assistant_id}", json=update_data)
    
    def update_assistant_status(self, assistant_id: int, is_active: bool) -> Dict[str, Any]:
        """Update assistant status."""
        params = {"is_active": is_active}
        return self._request("PATCH", f"/assistants/{assistant_id}/status", params=params)
    
    def delete_assistant(self, assistant_id: int) -> Dict[str, Any]:
        """Delete an assistant."""
        return self._request("DELETE", f"/assistants/{assistant_id}")
    
    def get_assistants_count(self, active_only: bool = False) -> int:
        """Get count of assistants."""
        params = {"active_only": active_only}
        result = self._request("GET", "/assistants/count", params=params)
        return result["count"]
    
    def get_llm_providers(self) -> Dict[str, Any]:
        """Get supported LLM providers."""
        return self._request("GET", "/assistants/providers")
    
    # Call endpoints
    def list_calls(self, status: Optional[str] = None, assistant_id: Optional[int] = None,
                   limit: int = 100) -> list:
        """List calls."""
        params = {"limit": limit}
        if status:
            params["status"] = status
        if assistant_id:
            params["assistant_id"] = assistant_id
        return self._request("GET", "/calls/", params=params)
    
    def get_call(self, call_id: int) -> Dict[str, Any]:
        """Get a call by ID."""
        return self._request("GET", f"/calls/{call_id}")
    
    def get_call_transcripts(self, call_id: int, speaker: Optional[str] = None) -> list:
        """Get call transcripts."""
        params = {}
        if speaker:
            params["speaker"] = speaker
        return self._request("GET", f"/calls/{call_id}/transcripts", params=params)
    
    def get_call_stats(self) -> Dict[str, Any]:
        """Get call statistics."""
        return self._request("GET", "/calls/stats")
    
    def get_organization_info(self) -> Dict[str, Any]:
        """Get organization information."""
        return self._request("GET", "/assistants/me/organization")


def main():
    """Example usage of the Diwaar API."""
    
    # Configuration - replace with your actual values
    BASE_URL = "https://yourdomain.com"  # Replace with your domain
    API_KEY = "diwaar_your_api_key_here"  # Replace with your API key
    
    # Initialize the API client
    api = DiwaarAPI(BASE_URL, API_KEY)
    
    try:
        # Get organization info
        print("=== Organization Info ===")
        org_info = api.get_organization_info()
        print(json.dumps(org_info, indent=2))
        
        # Get supported LLM providers
        print("\n=== Supported LLM Providers ===")
        providers = api.get_llm_providers()
        print(json.dumps(providers, indent=2))
        
        # Create a new assistant
        print("\n=== Creating Assistant ===")
        assistant_data = {
            "name": "Example Customer Service Bot",
            "phone_number": "+1234567890",  # Replace with actual phone number
            "description": "Example assistant created via API",
            "llm_provider": "openai",
            "llm_provider_config": {
                "api_key": "your_openai_key_here",  # Replace with your OpenAI key
                "model": "gpt-4o-mini"
            },
            "llm_settings": {
                "temperature": 0.7,
                "max_tokens": 1000,
                "system_prompt": "You are a helpful customer service representative. Be polite and professional."
            },
            "tts_settings": {
                "voice_id": "rachel",
                "model_id": "turbo",
                "stability": 0.5,
                "similarity_boost": 0.75
            },
            "stt_settings": {
                "model": "nova-2",
                "language": "en-US",
                "punctuate": True
            },
            "is_active": True
        }
        
        new_assistant = api.create_assistant(assistant_data)
        print(f"Created assistant: {new_assistant['name']} (ID: {new_assistant['id']})")
        
        # List assistants
        print("\n=== Listing Assistants ===")
        assistants = api.list_assistants(active_only=True)
        print(f"Found {len(assistants)} active assistants:")
        for assistant in assistants:
            print(f"  - {assistant['name']} (ID: {assistant['id']}, Phone: {assistant['phone_number']})")
        
        # Get assistants count
        count = api.get_assistants_count(active_only=True)
        print(f"\nTotal active assistants: {count}")
        
        # Update the assistant we just created
        print(f"\n=== Updating Assistant {new_assistant['id']} ===")
        update_data = {
            "description": "Updated description via API",
            "llm_settings": {
                "temperature": 0.8,
                "system_prompt": "You are an updated helpful assistant."
            }
        }
        updated_assistant = api.update_assistant(new_assistant['id'], update_data)
        print("Assistant updated successfully")
        
        # Deactivate the assistant
        print(f"\n=== Deactivating Assistant {new_assistant['id']} ===")
        api.update_assistant_status(new_assistant['id'], False)
        print("Assistant deactivated")
        
        # Get call statistics
        print("\n=== Call Statistics ===")
        stats = api.get_call_stats()
        print(json.dumps(stats, indent=2))
        
        # List recent calls
        print("\n=== Recent Calls ===")
        calls = api.list_calls(limit=5)
        print(f"Found {len(calls)} recent calls:")
        for call in calls:
            print(f"  - Call {call['id']}: {call['status']} (Assistant: {call['assistant_id']})")
        
        # Clean up - delete the test assistant
        print(f"\n=== Deleting Assistant {new_assistant['id']} ===")
        result = api.delete_assistant(new_assistant['id'])
        print(f"Delete result: {result['message']}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    print("\n=== API Example Completed Successfully ===")
    return 0


if __name__ == "__main__":
    exit(main()) 