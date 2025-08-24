"""
API endpoints for tool management.
Handles CRUD operations for custom tools and tool assignments.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.auth import get_current_user_flexible
from app.db.models import User, AssistantTool
from app.services.tool_service import ToolService

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["tools"])


# Pydantic models for request/response
class ToolCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Tool function name (snake_case)")
    display_name: str = Field(..., min_length=1, max_length=200, description="Human-readable name")
    description: str = Field(..., min_length=1, description="Tool description for LLM")
    tool_type: str = Field(..., pattern="^(endpoint|python_function|lambda)$", description="Tool type")
    configuration: Dict[str, Any] = Field(..., description="Tool-specific configuration")
    function_definition: Dict[str, Any] = Field(..., description="LLM function definition")
    timeout_seconds: int = Field(30, ge=1, le=300, description="Execution timeout")
    retry_attempts: int = Field(3, ge=0, le=10, description="Retry attempts")
    is_active: bool = Field(True, description="Whether tool is active")
    is_public: bool = Field(False, description="Whether tool is public")


class ToolUpdateRequest(BaseModel):
    display_name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=1)
    configuration: Optional[Dict[str, Any]] = None
    function_definition: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[int] = Field(None, ge=1, le=300)
    retry_attempts: Optional[int] = Field(None, ge=0, le=10)
    is_active: Optional[bool] = None
    is_public: Optional[bool] = None


class ToolTestRequest(BaseModel):
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Test parameters")


class ToolAssignRequest(BaseModel):
    tool_ids: List[int] = Field(..., description="List of tool IDs to assign")
    enabled: bool = Field(True, description="Whether tools are enabled")
    custom_configuration: Optional[Dict[str, Any]] = Field(None, description="Custom configuration")


class ToolConfigureRequest(BaseModel):
    enabled: bool = Field(True, description="Whether tool is enabled")
    custom_configuration: Optional[Dict[str, Any]] = Field(None, description="Custom configuration")


# Organization Tools Endpoints

@router.get("/organization/tools")
async def list_organization_tools(
    tool_type: Optional[str] = Query(None, pattern="^(endpoint|python_function|lambda)$"),
    is_active: Optional[bool] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    sort_by: str = Query("name", pattern="^(name|display_name|created_at|updated_at|execution_count)$"),
    sort_order: str = Query("asc", pattern="^(asc|desc)$"),
    current_user: User = Depends(get_current_user_flexible)
):
    """List tools for the organization with filtering and pagination."""
    try:
        tool_service = ToolService()
        result = await tool_service.list_tools(
            organization_id=current_user.organization_id,
            tool_type=tool_type,
            is_active=is_active,
            search=search,
            page=page,
            per_page=per_page,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Convert tools to dict format
        tools_data = []
        for tool in result['tools']:
            tools_data.append({
                'id': tool.id,
                'name': tool.name,
                'display_name': tool.display_name,
                'description': tool.description,
                'tool_type': tool.tool_type,
                'is_active': tool.is_active,
                'is_public': tool.is_public,
                'execution_count': tool.execution_count,
                'success_rate': tool.get_success_rate(),
                'last_executed_at': tool.last_executed_at.isoformat() if tool.last_executed_at else None,
                'created_at': tool.created_at.isoformat() if tool.created_at else None,
                'updated_at': tool.updated_at.isoformat() if tool.updated_at else None,
            })
        
        return {
            'success': True,
            'tools': tools_data,
            'pagination': result['pagination']
        }
        
    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")


@router.post("/organization/tools")
async def create_organization_tool(
    request: ToolCreateRequest,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Create a new tool for the organization."""
    try:
        tool_service = ToolService()
        tool = await tool_service.create_tool(
            organization_id=current_user.organization_id,
            user_id=current_user.id,
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            tool_type=request.tool_type,
            configuration=request.configuration,
            function_definition=request.function_definition,
            timeout_seconds=request.timeout_seconds,
            retry_attempts=request.retry_attempts,
            is_active=request.is_active,
            is_public=request.is_public,
        )
        
        return {
            'success': True,
            'message': 'Tool created successfully',
            'tool': {
                'id': tool.id,
                'name': tool.name,
                'display_name': tool.display_name,
                'tool_type': tool.tool_type,
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create tool: {str(e)}")


@router.get("/organization/tools/{tool_id}")
async def get_organization_tool(
    tool_id: int,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Get a specific tool by ID."""
    try:
        tool_service = ToolService()
        tool = await tool_service.get_tool(tool_id, current_user.organization_id)
        
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        # Get additional statistics
        assignment_count = db.query(AssistantTool).filter(AssistantTool.tool_id == tool_id).count()
        recent_logs = await tool_service.get_tool_execution_logs(tool_id=tool_id, limit=5)
        
        return {
            'success': True,
            'tool': {
                'id': tool.id,
                'name': tool.name,
                'display_name': tool.display_name,
                'description': tool.description,
                'tool_type': tool.tool_type,
                'configuration': tool.configuration,
                'function_definition': tool.function_definition,
                'timeout_seconds': tool.timeout_seconds,
                'retry_attempts': tool.retry_attempts,
                'is_active': tool.is_active,
                'is_public': tool.is_public,
                'execution_count': tool.execution_count,
                'success_count': tool.success_count,
                'failure_count': tool.failure_count,
                'success_rate': tool.get_success_rate(),
                'last_executed_at': tool.last_executed_at.isoformat() if tool.last_executed_at else None,
                'created_at': tool.created_at.isoformat() if tool.created_at else None,
                'updated_at': tool.updated_at.isoformat() if tool.updated_at else None,
                'assignment_count': assignment_count,
                'recent_executions': [log.get_summary() for log in recent_logs]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tool {tool_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get tool: {str(e)}")


@router.put("/organization/tools/{tool_id}")
async def update_organization_tool(
    tool_id: int,
    request: ToolUpdateRequest,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Update an existing tool."""
    try:
        tool_service = ToolService()
        
        # Only include non-None fields in update
        update_data = {k: v for k, v in request.dict().items() if v is not None}
        
        tool = await tool_service.update_tool(
            tool_id=tool_id,
            organization_id=current_user.organization_id,
            **update_data
        )
        
        return {
            'success': True,
            'message': 'Tool updated successfully',
            'tool': {
                'id': tool.id,
                'name': tool.name,
                'display_name': tool.display_name,
                'updated_at': tool.updated_at.isoformat() if tool.updated_at else None,
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating tool {tool_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update tool: {str(e)}")


@router.delete("/organization/tools/{tool_id}")
async def delete_organization_tool(
    tool_id: int,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Delete a tool."""
    try:
        tool_service = ToolService()
        deleted = await tool_service.delete_tool(tool_id, current_user.organization_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        return {
            'success': True,
            'message': 'Tool deleted successfully'
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting tool {tool_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete tool: {str(e)}")


@router.post("/organization/tools/{tool_id}/duplicate")
async def duplicate_organization_tool(
    tool_id: int,
    new_name: Optional[str] = None,
    new_display_name: Optional[str] = None,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Duplicate an existing tool."""
    try:
        tool_service = ToolService()
        duplicated_tool = await tool_service.duplicate_tool(
            tool_id=tool_id,
            organization_id=current_user.organization_id,
            new_name=new_name,
            new_display_name=new_display_name
        )
        
        return {
            'success': True,
            'message': 'Tool duplicated successfully',
            'tool_id': duplicated_tool.id,
            'tool': {
                'id': duplicated_tool.id,
                'name': duplicated_tool.name,
                'display_name': duplicated_tool.display_name,
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error duplicating tool {tool_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to duplicate tool: {str(e)}")


@router.post("/organization/tools/{tool_id}/test")
async def test_organization_tool(
    tool_id: int,
    request: ToolTestRequest,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Test a tool with given parameters."""
    try:
        tool_service = ToolService()
        result = await tool_service.test_tool(
            tool_id=tool_id,
            parameters=request.parameters,
            organization_id=current_user.organization_id
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error testing tool {tool_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to test tool: {str(e)}")


@router.get("/organization/tools/{tool_id}/export")
async def export_organization_tool(
    tool_id: int,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Export a tool configuration as JSON."""
    try:
        tool_service = ToolService()
        export_data = await tool_service.export_tool(tool_id, current_user.organization_id)
        
        # Return as downloadable file
        filename = f"tool-{export_data['name']}-export.json"
        content = json.dumps(export_data, indent=2)
        
        return JSONResponse(
            content=export_data,
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Type': 'application/json'
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting tool {tool_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to export tool: {str(e)}")


@router.post("/organization/tools/import")
async def import_organization_tool(
    file: UploadFile = File(...),
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Import a tool from JSON file."""
    try:
        # Read and parse the uploaded file
        content = await file.read()
        try:
            import_data = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
        
        tool_service = ToolService()
        tool = await tool_service.import_tool(
            organization_id=current_user.organization_id,
            user_id=current_user.id,
            import_data=import_data
        )
        
        return {
            'success': True,
            'message': 'Tool imported successfully',
            'tool': {
                'id': tool.id,
                'name': tool.name,
                'display_name': tool.display_name,
                'tool_type': tool.tool_type,
            }
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error importing tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to import tool: {str(e)}")


@router.get("/organization/tools/{tool_id}/logs")
async def get_tool_execution_logs(
    tool_id: int,
    status: Optional[str] = Query(None, pattern="^(success|error|timeout)$"),
    limit: int = Query(50, ge=1, le=500),
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Get execution logs for a tool."""
    try:
        # Verify tool belongs to organization
        tool_service = ToolService()
        tool = await tool_service.get_tool(tool_id, current_user.organization_id)
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        logs = await tool_service.get_tool_execution_logs(
            tool_id=tool_id,
            status=status,
            limit=limit
        )
        
        return {
            'success': True,
            'logs': [log.get_summary() for log in logs]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tool logs {tool_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get tool logs: {str(e)}")


# Assistant Tools Endpoints

@router.get("/assistants/{assistant_id}/tools")
async def get_assistant_tools(
    assistant_id: int,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Get all tools assigned to an assistant."""
    try:
        tool_service = ToolService()
        assignments = await tool_service.get_assistant_tools(assistant_id)
        
        tools_data = []
        for assignment in assignments:
            tool = assignment.tool
            tools_data.append({
                'tool': {
                    'id': tool.id,
                    'name': tool.name,
                    'display_name': tool.display_name,
                    'description': tool.description,
                    'tool_type': tool.tool_type,
                    'execution_count': tool.execution_count,
                },
                'assignment': {
                    'enabled': assignment.enabled,
                    'custom_configuration': assignment.custom_configuration,
                    'assigned_at': assignment.assigned_at.isoformat() if assignment.assigned_at else None,
                    'execution_count': assignment.execution_count,
                    'last_executed_at': assignment.last_executed_at.isoformat() if assignment.last_executed_at else None,
                }
            })
        
        return {
            'success': True,
            'tools': tools_data
        }
        
    except Exception as e:
        logger.error(f"Error getting assistant tools {assistant_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get assistant tools: {str(e)}")


@router.post("/lambda/list-functions")
async def list_lambda_functions(
    access_key: str = Query(..., description="AWS Access Key ID"),
    secret_key: str = Query(..., description="AWS Secret Access Key"),
    region: str = Query("us-east-1", description="AWS Region"),
    current_user: User = Depends(get_current_user_flexible)
):
    """List available Lambda functions in user's AWS account."""
    try:
        from app.services.tool_executors.lambda_executor import LambdaToolExecutor
        
        executor = LambdaToolExecutor()
        configuration = {
            'access_key': access_key,
            'secret_key': secret_key,
            'region': region,
            'function_name': 'dummy'  # Not needed for listing
        }
        
        result = executor.list_lambda_functions(configuration)
        
        if result['success']:
            return {
                'success': True,
                'functions': result['functions'],
                'message': result['message']
            }
        else:
            raise HTTPException(status_code=400, detail=result['error'])
            
    except Exception as e:
        logger.error(f"Error listing Lambda functions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list Lambda functions: {str(e)}")


@router.post("/assistants/{assistant_id}/tools/assign")
async def assign_tools_to_assistant(
    assistant_id: int,
    request: ToolAssignRequest,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Assign tools to an assistant."""
    try:
        tool_service = ToolService()
        assignments = []
        
        for tool_id in request.tool_ids:
            assignment = await tool_service.assign_tool_to_assistant(
                assistant_id=assistant_id,
                tool_id=tool_id,
                user_id=current_user.id,
                enabled=request.enabled,
                custom_configuration=request.custom_configuration
            )
            assignments.append(assignment)
        
        return {
            'success': True,
            'message': f'Successfully assigned {len(assignments)} tools',
            'assignments': len(assignments)
        }
        
    except Exception as e:
        logger.error(f"Error assigning tools to assistant {assistant_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to assign tools: {str(e)}")


@router.put("/assistants/{assistant_id}/tools/{tool_id}/configure")
async def configure_assistant_tool(
    assistant_id: int,
    tool_id: int,
    request: ToolConfigureRequest,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Configure a tool for a specific assistant."""
    try:
        tool_service = ToolService()
        assignment = await tool_service.assign_tool_to_assistant(
            assistant_id=assistant_id,
            tool_id=tool_id,
            user_id=current_user.id,
            enabled=request.enabled,
            custom_configuration=request.custom_configuration
        )
        
        return {
            'success': True,
            'message': 'Tool configuration updated successfully'
        }
        
    except Exception as e:
        logger.error(f"Error configuring tool {tool_id} for assistant {assistant_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to configure tool: {str(e)}")


@router.delete("/assistants/{assistant_id}/tools/{tool_id}/unassign")
async def unassign_tool_from_assistant(
    assistant_id: int,
    tool_id: int,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Unassign a tool from an assistant."""
    try:
        tool_service = ToolService()
        unassigned = await tool_service.unassign_tool_from_assistant(assistant_id, tool_id)
        
        if not unassigned:
            raise HTTPException(status_code=404, detail="Tool assignment not found")
        
        return {
            'success': True,
            'message': 'Tool unassigned successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unassigning tool {tool_id} from assistant {assistant_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unassign tool: {str(e)}")


@router.get("/assistants/{assistant_id}/tool-executions")
async def get_assistant_tool_executions(
    assistant_id: int,
    tool_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None, pattern="^(success|error|timeout)$"),
    limit: int = Query(50, ge=1, le=500),
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Get tool execution logs for an assistant."""
    try:
        tool_service = ToolService()
        logs = await tool_service.get_tool_execution_logs(
            tool_id=tool_id,
            assistant_id=assistant_id,
            status=status,
            limit=limit
        )
        
        return {
            'success': True,
            'logs': [log.get_summary() for log in logs]
        }
        
    except Exception as e:
        logger.error(f"Error getting assistant tool executions {assistant_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get tool executions: {str(e)}")


# Call Tools Endpoints

@router.get("/calls/{call_id}/tool-executions")
async def get_call_tool_executions(
    call_id: int,
    tool_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None, pattern="^(success|error|timeout)$"),
    limit: int = Query(50, ge=1, le=500),
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Get tool execution logs for a specific call."""
    try:
        tool_service = ToolService()
        logs = await tool_service.get_tool_execution_logs(
            tool_id=tool_id,
            call_id=call_id,
            status=status,
            limit=limit
        )
        
        return {
            'success': True,
            'logs': [log.get_summary() for log in logs]
        }
        
    except Exception as e:
        logger.error(f"Error getting call tool executions {call_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get tool executions: {str(e)}")


# General Tools Endpoints

@router.post("/tools/test")
async def test_tool_configuration(
    request: Dict[str, Any],
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Test a tool configuration without saving it."""
    try:
        # This endpoint is used by the frontend tool builder for testing
        from app.services.tool_executors.endpoint_executor import EndpointToolExecutor
        from app.services.tool_executors.python_executor import PythonToolExecutor
        from app.services.tool_executors.lambda_executor import LambdaToolExecutor
        
        tool_type = request.get('tool_type')
        configuration = request.get('configuration', {})
        test_parameters = request.get('test_parameters', {})
        
        if not tool_type:
            raise ValueError("Missing tool_type in request")
        
        # Get the appropriate executor
        if tool_type == 'endpoint':
            executor = EndpointToolExecutor(timeout_seconds=30, retry_attempts=1)
        elif tool_type == 'python_function':
            executor = PythonToolExecutor(timeout_seconds=30, retry_attempts=1)
        elif tool_type == 'lambda':
            executor = LambdaToolExecutor(timeout_seconds=30, retry_attempts=1)
        else:
            raise ValueError(f"Unknown tool type: {tool_type}")
        
        # Execute the test
        result = executor.execute(test_parameters, configuration, {})
        
        return {
            'success': True,
            'result': result,
            'message': 'Tool test completed successfully'
        }
        
    except Exception as e:
        logger.error(f"Error testing tool configuration: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'message': f'Tool test failed: {str(e)}'
        }
