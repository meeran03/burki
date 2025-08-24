"""
Base tool executor class.
All tool executors inherit from this class.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseToolExecutor(ABC):
    """Base class for all tool executors."""

    def __init__(self, timeout_seconds: int = 30, retry_attempts: int = 3):
        """
        Initialize the tool executor.
        
        Args:
            timeout_seconds: Maximum execution time
            retry_attempts: Number of retry attempts on failure
        """
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts

    @abstractmethod
    def execute(
        self,
        parameters: Dict[str, Any],
        configuration: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.
        
        Args:
            parameters: Tool parameters from LLM
            configuration: Tool configuration
            context: Execution context (call info, etc.)
            
        Returns:
            Dict: Execution result
            
        Raises:
            Exception: If execution fails
        """
        pass

    def execute_with_retry(
        self,
        parameters: Dict[str, Any],
        configuration: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute the tool with retry logic.
        
        Args:
            parameters: Tool parameters from LLM
            configuration: Tool configuration
            context: Execution context
            
        Returns:
            Dict: Execution result
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.retry_attempts + 1):
            try:
                return self.execute(parameters, configuration, context)
            except Exception as e:
                last_exception = e
                if attempt < self.retry_attempts:
                    # Wait before retry (exponential backoff)
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    # Last attempt failed
                    break
        
        # All attempts failed
        raise last_exception

    def validate_parameters(
        self,
        parameters: Dict[str, Any],
        required_params: list = None,
        optional_params: list = None
    ) -> None:
        """
        Validate tool parameters.
        
        Args:
            parameters: Parameters to validate
            required_params: List of required parameter names
            optional_params: List of optional parameter names
            
        Raises:
            ValueError: If validation fails
        """
        if required_params:
            for param in required_params:
                if param not in parameters:
                    raise ValueError(f"Missing required parameter: {param}")
        
        # Check for unexpected parameters
        if required_params or optional_params:
            allowed_params = set(required_params or []) | set(optional_params or [])
            unexpected_params = set(parameters.keys()) - allowed_params
            if unexpected_params:
                raise ValueError(f"Unexpected parameters: {', '.join(unexpected_params)}")

    def substitute_variables(
        self,
        template: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> str:
        """
        Substitute variables in a template string.
        
        Supports:
        - ${parameters.name} for parameter values
        - ${context.field} for context values
        - ${env.VAR} for environment variables
        
        Args:
            template: Template string
            parameters: Tool parameters
            context: Execution context
            
        Returns:
            str: String with variables substituted
        """
        import os
        import re
        
        if not template:
            return template
            
        result = template
        
        # Substitute parameters
        if parameters:
            for key, value in parameters.items():
                pattern = rf'\$\{{parameters\.{re.escape(key)}\}}'
                result = re.sub(pattern, str(value), result)
        
        # Substitute context
        if context:
            for key, value in context.items():
                pattern = rf'\$\{{context\.{re.escape(key)}\}}'
                result = re.sub(pattern, str(value), result)
        
        # Substitute environment variables
        env_pattern = r'\$\{env\.([A-Za-z_][A-Za-z0-9_]*)\}'
        def env_replacer(match):
            env_var = match.group(1)
            return os.getenv(env_var, '')
        
        result = re.sub(env_pattern, env_replacer, result)
        
        return result

    def format_result(
        self,
        success: bool,
        message: str = None,
        data: Any = None,
        error: str = None
    ) -> Dict[str, Any]:
        """
        Format a standardized tool execution result.
        
        Args:
            success: Whether execution was successful
            message: Human-readable message
            data: Result data
            error: Error message if failed
            
        Returns:
            Dict: Formatted result
        """
        result = {
            'success': success,
            'timestamp': time.time(),
        }
        
        if message:
            result['message'] = message
        
        if data is not None:
            result['data'] = data
        
        if error:
            result['error'] = error
        
        return result
