"""
Endpoint tool executor for calling HTTP APIs.
"""

import json
import requests
from typing import Dict, Any, Optional
from .base import BaseToolExecutor


class EndpointToolExecutor(BaseToolExecutor):
    """Executor for HTTP endpoint tools."""

    def execute(
        self,
        parameters: Dict[str, Any],
        configuration: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute an HTTP endpoint tool.
        
        Args:
            parameters: Tool parameters from LLM
            configuration: Tool configuration containing:
                - url: Endpoint URL
                - method: HTTP method
                - headers: Request headers (optional)
                - body_template: Request body template (optional)
            context: Execution context
            
        Returns:
            Dict: Execution result
        """
        # Validate configuration
        if 'url' not in configuration:
            raise ValueError("Missing required configuration: url")
        
        if 'method' not in configuration:
            raise ValueError("Missing required configuration: method")

        method = configuration['method'].upper()
        url = self.substitute_variables(configuration['url'], parameters, context)
        
        # Prepare headers
        headers = {'Content-Type': 'application/json'}
        if 'headers' in configuration and configuration['headers']:
            if isinstance(configuration['headers'], str):
                # Parse JSON string
                try:
                    config_headers = json.loads(configuration['headers'])
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid headers JSON: {e}")
            else:
                config_headers = configuration['headers']
            
            # Substitute variables in headers
            for key, value in config_headers.items():
                substituted_value = self.substitute_variables(str(value), parameters, context)
                headers[key] = substituted_value

        # Prepare request body
        data = None
        if method in ['POST', 'PUT', 'PATCH'] and 'body_template' in configuration:
            body_template = configuration['body_template']
            if isinstance(body_template, str):
                # Substitute variables in body template
                body_str = self.substitute_variables(body_template, parameters, context)
                try:
                    data = json.loads(body_str)
                except json.JSONDecodeError:
                    # If not valid JSON, send as string
                    data = body_str
                    headers['Content-Type'] = 'text/plain'
            else:
                # Already a dict/object
                data = body_template
                # Substitute variables in values
                data = self._substitute_in_dict(data, parameters, context)

        # Make HTTP request
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data if headers.get('Content-Type') == 'application/json' else None,
                data=data if headers.get('Content-Type') != 'application/json' else None,
                timeout=self.timeout_seconds
            )
            
            # Parse response
            response_data = self._parse_response(response)
            
            # Check if request was successful
            if response.status_code >= 400:
                return self.format_result(
                    success=False,
                    error=f"HTTP {response.status_code}: {response_data.get('error', response.reason)}",
                    data={
                        'status_code': response.status_code,
                        'response': response_data,
                        'url': url,
                        'method': method
                    }
                )
            
            return self.format_result(
                success=True,
                message=f"Successfully called {method} {url}",
                data={
                    'status_code': response.status_code,
                    'response': response_data,
                    'url': url,
                    'method': method
                }
            )

        except requests.exceptions.Timeout:
            return self.format_result(
                success=False,
                error=f"Request timeout after {self.timeout_seconds} seconds",
                data={'url': url, 'method': method}
            )
        
        except requests.exceptions.ConnectionError as e:
            return self.format_result(
                success=False,
                error=f"Connection error: {str(e)}",
                data={'url': url, 'method': method}
            )
        
        except requests.exceptions.RequestException as e:
            return self.format_result(
                success=False,
                error=f"Request error: {str(e)}",
                data={'url': url, 'method': method}
            )

    def _substitute_in_dict(
        self,
        data: Dict[str, Any],
        parameters: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Recursively substitute variables in a dictionary.
        
        Args:
            data: Dictionary to process
            parameters: Tool parameters
            context: Execution context
            
        Returns:
            Dict: Dictionary with substituted values
        """
        result = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.substitute_variables(value, parameters, context)
            elif isinstance(value, dict):
                result[key] = self._substitute_in_dict(value, parameters, context)
            elif isinstance(value, list):
                result[key] = [
                    self.substitute_variables(item, parameters, context) if isinstance(item, str)
                    else self._substitute_in_dict(item, parameters, context) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result

    def _parse_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Parse HTTP response to extract data.
        
        Args:
            response: HTTP response object
            
        Returns:
            Dict: Parsed response data
        """
        content_type = response.headers.get('Content-Type', '').lower()
        
        try:
            if 'application/json' in content_type:
                return response.json()
            elif 'text/' in content_type or 'application/xml' in content_type:
                return {'text': response.text}
            else:
                # For other content types, return basic info
                return {
                    'content_type': content_type,
                    'content_length': len(response.content),
                    'text': response.text[:1000] if len(response.text) <= 1000 else response.text[:1000] + '...'
                }
        except Exception as e:
            # Fallback if parsing fails
            return {
                'error': f"Failed to parse response: {str(e)}",
                'raw_content': response.text[:500] if hasattr(response, 'text') else 'No content',
                'content_type': content_type
            }
