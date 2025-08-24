"""
Python function executor for running custom Python code in a secure sandbox.
"""

import sys
import io
import json
import time
import signal
import traceback
from typing import Dict, Any, Optional
from contextlib import contextmanager
from .base import BaseToolExecutor


class PythonToolExecutor(BaseToolExecutor):
    """Executor for Python function tools."""

    # Default allowed imports for security
    DEFAULT_ALLOWED_IMPORTS = [
        'datetime', 'time', 'json', 'math', 'random', 're', 'urllib.parse',
        'base64', 'hashlib', 'hmac', 'uuid', 'decimal', 'fractions',
        'collections', 'itertools', 'functools', 'operator', 'string'
    ]

    def execute(
        self,
        parameters: Dict[str, Any],
        configuration: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a Python function tool.
        
        Args:
            parameters: Tool parameters from LLM
            configuration: Tool configuration containing:
                - code: Python code to execute
                - allowed_imports: List of allowed import modules (optional)
            context: Execution context
            
        Returns:
            Dict: Execution result
        """
        # Validate configuration
        if 'code' not in configuration:
            raise ValueError("Missing required configuration: code")

        code = configuration['code']
        allowed_imports = configuration.get('allowed_imports', self.DEFAULT_ALLOWED_IMPORTS)
        
        # Ensure allowed_imports is a list
        if isinstance(allowed_imports, str):
            allowed_imports = [imp.strip() for imp in allowed_imports.split(',') if imp.strip()]

        try:
            # Execute code in sandbox
            result = self._execute_in_sandbox(code, parameters, context or {}, allowed_imports)
            
            if result is None:
                return self.format_result(
                    success=False,
                    error="Function did not return a result. Please ensure your function returns a dictionary with 'success' and 'message' fields."
                )
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                return self.format_result(
                    success=False,
                    error=f"Function must return a dictionary, got {type(result).__name__}"
                )
            
            # Validate result format
            if 'success' not in result:
                result['success'] = True  # Default to success if not specified
            
            if 'message' not in result:
                result['message'] = 'Function executed successfully'
            
            return self.format_result(
                success=result['success'],
                message=result['message'],
                data=result.get('data')
            )

        except TimeoutError:
            return self.format_result(
                success=False,
                error=f"Function execution timed out after {self.timeout_seconds} seconds"
            )
        
        except Exception as e:
            return self.format_result(
                success=False,
                error=f"Function execution failed: {str(e)}"
            )

    def _execute_in_sandbox(
        self,
        code: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
        allowed_imports: list
    ) -> Dict[str, Any]:
        """
        Execute Python code in a sandboxed environment.
        
        Args:
            code: Python code to execute
            parameters: Tool parameters
            context: Execution context
            allowed_imports: List of allowed import modules
            
        Returns:
            Dict: Function result
        """
        # Create a restricted globals environment
        restricted_globals = self._create_restricted_globals(allowed_imports)
        
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Redirect output
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Set up timeout
            def timeout_handler(signum, frame):
                raise TimeoutError("Function execution timed out")
            
            # Set alarm for timeout (Unix systems only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout_seconds)
            
            try:
                # Execute the code
                exec(code, restricted_globals)
                
                # Check if execute_tool function is defined
                if 'execute_tool' not in restricted_globals:
                    raise ValueError("Code must define an 'execute_tool' function")
                
                execute_tool_func = restricted_globals['execute_tool']
                
                # Call the function
                result = execute_tool_func(parameters, context)
                
                return result
                
            finally:
                # Clear the alarm
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
        
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Get captured output
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            # Log output if needed (for debugging)
            if stdout_output:
                print(f"Python tool stdout: {stdout_output}")
            if stderr_output:
                print(f"Python tool stderr: {stderr_output}")

    def _create_restricted_globals(self, allowed_imports: list) -> dict:
        """
        Create a restricted globals environment for code execution.
        
        Args:
            allowed_imports: List of allowed import modules
            
        Returns:
            dict: Restricted globals dictionary
        """
        # Start with basic builtins
        restricted_globals = {
            '__builtins__': {
                # Safe built-in functions
                'abs': abs,
                'bool': bool,
                'dict': dict,
                'float': float,
                'int': int,
                'len': len,
                'list': list,
                'max': max,
                'min': min,
                'range': range,
                'round': round,
                'set': set,
                'sorted': sorted,
                'str': str,
                'sum': sum,
                'tuple': tuple,
                'type': type,
                'zip': zip,
                'enumerate': enumerate,
                'filter': filter,
                'map': map,
                'any': any,
                'all': all,
                'isinstance': isinstance,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'print': print,
                
                # Safe exceptions
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'IndexError': IndexError,
                'AttributeError': AttributeError,
            }
        }
        
        # Add allowed imports
        import importlib
        for module_name in allowed_imports:
            try:
                # Validate module name (security check)
                if not self._is_safe_module(module_name):
                    continue
                    
                module = importlib.import_module(module_name)
                restricted_globals[module_name] = module
            except ImportError:
                # Skip modules that can't be imported
                continue
        
        return restricted_globals

    def _is_safe_module(self, module_name: str) -> bool:
        """
        Check if a module is safe to import.
        
        Args:
            module_name: Name of the module
            
        Returns:
            bool: True if safe, False otherwise
        """
        # Blacklist of dangerous modules
        dangerous_modules = [
            'os', 'sys', 'subprocess', 'importlib', 'builtins',
            'eval', 'exec', 'compile', '__import__',
            'open', 'file', 'input', 'raw_input',
            'reload', 'globals', 'locals', 'vars',
            'dir', 'help', 'copyright', 'credits', 'license',
            'quit', 'exit', 'socket', 'urllib2', 'httplib',
            'threading', 'multiprocessing', 'ctypes',
            'platform', 'tempfile', 'shutil', 'pickle',
            'cPickle', 'marshal', 'shelve', 'dbm',
            'signal', 'atexit', 'weakref', 'gc',
            'inspect', 'types', 'code', 'codeop'
        ]
        
        # Check if module is in blacklist
        if module_name in dangerous_modules:
            return False
        
        # Check for dangerous submodules
        for dangerous in dangerous_modules:
            if module_name.startswith(dangerous + '.'):
                return False
        
        return True

    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate Python code for syntax and basic security.
        
        Args:
            code: Python code to validate
            
        Returns:
            Dict: Validation result
        """
        try:
            # Check syntax
            compile(code, '<string>', 'exec')
            
            # Basic security checks
            dangerous_keywords = [
                'import os', 'import sys', 'import subprocess',
                'import socket', 'import urllib', 'import requests',
                '__import__', 'eval(', 'exec(', 'compile(',
                'open(', 'file(', 'input(', 'raw_input(',
                'globals(', 'locals(', 'vars(', 'dir(',
                'getattr', 'setattr', 'delattr', 'hasattr'
            ]
            
            code_lower = code.lower()
            found_dangerous = []
            
            for keyword in dangerous_keywords:
                if keyword in code_lower:
                    found_dangerous.append(keyword)
            
            if found_dangerous:
                return {
                    'valid': False,
                    'error': f"Code contains potentially dangerous operations: {', '.join(found_dangerous)}"
                }
            
            # Check for execute_tool function
            if 'def execute_tool(' not in code:
                return {
                    'valid': False,
                    'error': "Code must define an 'execute_tool(parameters, context)' function"
                }
            
            return {
                'valid': True,
                'message': 'Code validation passed'
            }
            
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f"Syntax error: {str(e)}"
            }
        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation error: {str(e)}"
            }
