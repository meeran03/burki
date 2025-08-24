"""
AWS Lambda tool executor for invoking Lambda functions.
"""

import json
import time
import boto3
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError
from .base import BaseToolExecutor


class LambdaToolExecutor(BaseToolExecutor):
    """Executor for AWS Lambda function tools."""

    def execute(
        self,
        parameters: Dict[str, Any],
        configuration: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute an AWS Lambda function tool.
        
        Args:
            parameters: Tool parameters from LLM
            configuration: Tool configuration containing:
                - function_name: Lambda function name
                - region: AWS region
                - payload_template: Payload template (optional)
                - access_key: AWS access key (optional)
                - secret_key: AWS secret key (optional)
            context: Execution context
            
        Returns:
            Dict: Execution result
        """
        # Validate configuration
        required_fields = ['function_name', 'region']
        for field in required_fields:
            if field not in configuration:
                raise ValueError(f"Missing required configuration: {field}")

        function_name = configuration['function_name']
        region = configuration['region']
        
        # Prepare payload
        payload = self._prepare_payload(parameters, configuration, context)
        
        try:
            # Create Lambda client
            lambda_client = self._create_lambda_client(configuration)
            
            # Invoke Lambda function
            response = lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',  # Synchronous execution
                LogType='Tail',  # Include logs in response
                Payload=json.dumps(payload)
            )
            
            # Parse response
            result = self._parse_lambda_response(response)
            
            return self.format_result(
                success=True,
                message=f"Successfully invoked Lambda function: {function_name}",
                data=result
            )

        except NoCredentialsError:
            return self.format_result(
                success=False,
                error="AWS credentials not found. Please configure access_key and secret_key."
            )
        
        except EndpointConnectionError as e:
            return self.format_result(
                success=False,
                error=f"Cannot connect to AWS endpoint in region {region}: {str(e)}"
            )
        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            
            if error_code == 'ResourceNotFoundException':
                return self.format_result(
                    success=False,
                    error=f"Lambda function '{function_name}' not found in region {region}"
                )
            elif error_code == 'AccessDeniedException':
                return self.format_result(
                    success=False,
                    error="Access denied. Please check your AWS credentials and permissions."
                )
            elif error_code == 'InvalidParameterValueException':
                return self.format_result(
                    success=False,
                    error=f"Invalid parameter: {error_message}"
                )
            else:
                return self.format_result(
                    success=False,
                    error=f"AWS error ({error_code}): {error_message}"
                )
        
        except Exception as e:
            return self.format_result(
                success=False,
                error=f"Lambda execution failed: {str(e)}"
            )

    def _create_lambda_client(self, configuration: Dict[str, Any]):
        """
        Create AWS Lambda client with credentials.
        
        Args:
            configuration: Tool configuration
            
        Returns:
            boto3.client: Lambda client
        """
        region = configuration['region']
        
        # Use provided credentials if available
        if 'access_key' in configuration and 'secret_key' in configuration:
            return boto3.client(
                'lambda',
                region_name=region,
                aws_access_key_id=configuration['access_key'],
                aws_secret_access_key=configuration['secret_key']
            )
        else:
            # Use default credentials (environment, IAM role, etc.)
            return boto3.client('lambda', region_name=region)

    def _prepare_payload(
        self,
        parameters: Dict[str, Any],
        configuration: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Prepare the payload for Lambda function.
        
        Args:
            parameters: Tool parameters
            configuration: Tool configuration
            context: Execution context
            
        Returns:
            Dict: Lambda payload
        """
        # Default payload structure
        payload = {
            'parameters': parameters,
            'context': context or {},
            'timestamp': time.time(),
        }
        
        # Use custom payload template if provided
        if 'payload_template' in configuration and configuration['payload_template']:
            payload_template = configuration['payload_template']
            
            if isinstance(payload_template, str):
                # Substitute variables in template string
                payload_str = self.substitute_variables(payload_template, parameters, context)
                try:
                    payload = json.loads(payload_str)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid payload template JSON: {e}")
            else:
                # Use template as-is and substitute variables in values
                payload = self._substitute_in_payload(payload_template, parameters, context)
        
        return payload

    def _substitute_in_payload(
        self,
        payload: Dict[str, Any],
        parameters: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Recursively substitute variables in payload.
        
        Args:
            payload: Payload dictionary
            parameters: Tool parameters
            context: Execution context
            
        Returns:
            Dict: Payload with substituted values
        """
        result = {}
        
        for key, value in payload.items():
            if isinstance(value, str):
                result[key] = self.substitute_variables(value, parameters, context)
            elif isinstance(value, dict):
                result[key] = self._substitute_in_payload(value, parameters, context)
            elif isinstance(value, list):
                result[key] = [
                    self.substitute_variables(item, parameters, context) if isinstance(item, str)
                    else self._substitute_in_payload(item, parameters, context) if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result

    def _parse_lambda_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Lambda function response.
        
        Args:
            response: Raw Lambda response
            
        Returns:
            Dict: Parsed response data
        """
        result = {
            'status_code': response['StatusCode'],
            'function_error': response.get('FunctionError'),
            'log_result': response.get('LogResult'),
        }
        
        # Parse payload
        if 'Payload' in response:
            payload_bytes = response['Payload'].read()
            payload_str = payload_bytes.decode('utf-8')
            
            try:
                result['payload'] = json.loads(payload_str)
            except json.JSONDecodeError:
                result['payload'] = payload_str
        
        # Decode logs if present
        if result['log_result']:
            import base64
            try:
                logs = base64.b64decode(result['log_result']).decode('utf-8')
                result['logs'] = logs
            except Exception:
                # Keep encoded logs if decoding fails
                pass
        
        # Check for function errors
        if result['function_error']:
            if result['function_error'] == 'Unhandled':
                error_msg = "Lambda function threw an unhandled exception"
                if 'payload' in result and isinstance(result['payload'], dict):
                    error_msg += f": {result['payload'].get('errorMessage', 'Unknown error')}"
                raise Exception(error_msg)
            elif result['function_error'] == 'Handled':
                # Function handled the error, include in result
                result['handled_error'] = True
        
        return result

    def validate_configuration(self, configuration: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Lambda tool configuration.
        
        Args:
            configuration: Configuration to validate
            
        Returns:
            Dict: Validation result
        """
        errors = []
        
        # Check required fields
        required_fields = ['function_name', 'region']
        for field in required_fields:
            if field not in configuration:
                errors.append(f"Missing required field: {field}")
        
        # Validate region
        if 'region' in configuration:
            valid_regions = [
                'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
                'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1',
                'ap-southeast-1', 'ap-southeast-2', 'ap-northeast-1',
                'ap-northeast-2', 'ap-south-1', 'ca-central-1',
                'sa-east-1'
            ]
            if configuration['region'] not in valid_regions:
                errors.append(f"Invalid region: {configuration['region']}")
        
        # Validate function name
        if 'function_name' in configuration:
            function_name = configuration['function_name']
            if not function_name or len(function_name) > 64:
                errors.append("Function name must be 1-64 characters")
        
        # Validate payload template if provided
        if 'payload_template' in configuration and configuration['payload_template']:
            payload_template = configuration['payload_template']
            if isinstance(payload_template, str):
                try:
                    json.loads(payload_template)
                except json.JSONDecodeError as e:
                    errors.append(f"Invalid payload template JSON: {e}")
        
        if errors:
            return {
                'valid': False,
                'errors': errors
            }
        
        return {
            'valid': True,
            'message': 'Configuration validation passed'
        }

    def test_connection(self, configuration: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test connection to AWS Lambda service.
        
        Args:
            configuration: Tool configuration
            
        Returns:
            Dict: Connection test result
        """
        try:
            lambda_client = self._create_lambda_client(configuration)
            
            # Try to get function configuration (without invoking)
            function_name = configuration['function_name']
            response = lambda_client.get_function_configuration(
                FunctionName=function_name
            )
            
            return {
                'success': True,
                'message': f"Successfully connected to Lambda function: {function_name}",
                'function_info': {
                    'runtime': response.get('Runtime'),
                    'timeout': response.get('Timeout'),
                    'memory_size': response.get('MemorySize'),
                    'last_modified': response.get('LastModified'),
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Connection test failed: {str(e)}"
            }

    def list_lambda_functions(self, configuration: Dict[str, Any]) -> Dict[str, Any]:
        """
        List available Lambda functions in the user's AWS account.
        
        Args:
            configuration: Tool configuration containing credentials and region
            
        Returns:
            Dict: List of available Lambda functions
        """
        try:
            lambda_client = self._create_lambda_client(configuration)
            
            # List all Lambda functions
            response = lambda_client.list_functions()
            functions = response.get('Functions', [])
            
            # Format function information
            function_list = []
            for func in functions:
                function_list.append({
                    'function_name': func.get('FunctionName'),
                    'description': func.get('Description', ''),
                    'runtime': func.get('Runtime'),
                    'timeout': func.get('Timeout'),
                    'memory_size': func.get('MemorySize'),
                    'last_modified': func.get('LastModified'),
                    'code_size': func.get('CodeSize'),
                    'handler': func.get('Handler'),
                    'version': func.get('Version'),
                })
            
            return {
                'success': True,
                'message': f"Found {len(function_list)} Lambda functions",
                'functions': function_list
            }
            
        except NoCredentialsError:
            return {
                'success': False,
                'error': "AWS credentials not found. Please configure access_key and secret_key."
            }
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            return {
                'success': False,
                'error': f"AWS error ({error_code}): {error_message}"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to list Lambda functions: {str(e)}"
            }
