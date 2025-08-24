"""
Web routes for tool management pages.
Serves the HTML templates for the tool management interface.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.auth import get_current_user_flexible
from app.db.models import User, AssistantTool
from app.services.tool_service import ToolService

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/tools", response_class=HTMLResponse)
async def tools_index(
    request: Request,
    search: Optional[str] = None,
    tool_type: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    per_page: int = 12,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Tools library main page."""
    try:
        tool_service = ToolService()
        
        # Get tools with filtering
        is_active = None
        if status == 'active':
            is_active = True
        elif status == 'inactive':
            is_active = False
        
        result = await tool_service.list_tools(
            organization_id=current_user.organization_id,
            tool_type=tool_type,
            is_active=is_active,
            search=search,
            page=page,
            per_page=per_page,
            sort_by='name',
            sort_order='asc'
        )
        
        # Calculate statistics (simplified for now)
        total_tools = 0
        active_tools = 0
        total_executions = 0
        total_success = 0
        
        success_rate = round((total_success / total_executions * 100) if total_executions > 0 else 0, 1)
        
        return templates.TemplateResponse("tools/index.html", {
            "request": request,
            "current_user": current_user,
            "session": {
                "user_id": request.session.get("user_id"),
                "organization_id": request.session.get("organization_id"),
                "user_first_name": current_user.first_name,
                "user_last_name": current_user.last_name,
            },
            "tools": result['tools'],
            "pagination": result['pagination'],
            "tools_count": total_tools,
            "active_tools_count": active_tools,
            "total_executions": total_executions,
            "success_rate": success_rate,
        })
        
    except Exception as e:
        logger.error(f"Error rendering tools index: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load tools page")


@router.get("/tools/new", response_class=HTMLResponse)
async def tools_new(
    request: Request,
    current_user: User = Depends(get_current_user_flexible)
):
    """Create new tool page."""
    return templates.TemplateResponse("tools/form.html", {
        "request": request,
        "current_user": current_user,
        "session": {
            "user_id": request.session.get("user_id"),
            "organization_id": request.session.get("organization_id"),
            "user_first_name": current_user.first_name,
            "user_last_name": current_user.last_name,
        },
        "tool": None,
    })


@router.get("/tools/{tool_id}", response_class=HTMLResponse)
async def tools_view(
    request: Request,
    tool_id: int,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Tool detail view page."""
    try:
        tool_service = ToolService()
        tool = await tool_service.get_tool(tool_id, current_user.organization_id)
        
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        # Get additional data for the view
        from app.db.database import get_async_db_session
        from sqlalchemy import select, func
        
        async with await get_async_db_session() as db:
            # Get assignment count
            count_result = await db.execute(
                select(func.count(AssistantTool.id)).where(AssistantTool.tool_id == tool_id)
            )
            assignment_count = count_result.scalar()
            
            # Get assigned assistants
            assistants_result = await db.execute(
                select(AssistantTool).where(AssistantTool.tool_id == tool_id)
            )
            assigned_assistants = assistants_result.scalars().all()
        
        # Get recent executions
        recent_executions = await tool_service.get_tool_execution_logs(tool_id=tool_id, limit=10)
        
        # Generate default test parameters based on function definition
        default_test_params = {}
        if tool.function_definition and 'function' in tool.function_definition:
            func_def = tool.function_definition['function']
            if 'parameters' in func_def and 'properties' in func_def['parameters']:
                properties = func_def['parameters']['properties']
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'string')
                    if param_type == 'string':
                        default_test_params[param_name] = "test_value"
                    elif param_type == 'number':
                        default_test_params[param_name] = 42
                    elif param_type == 'boolean':
                        default_test_params[param_name] = True
                    elif param_type == 'array':
                        default_test_params[param_name] = ["item1", "item2"]
                    elif param_type == 'object':
                        default_test_params[param_name] = {"key": "value"}
        
        return templates.TemplateResponse("tools/view.html", {
            "request": request,
            "current_user": current_user,
            "session": {
                "user_id": request.session.get("user_id"),
                "organization_id": request.session.get("organization_id"),
                "user_first_name": current_user.first_name,
                "user_last_name": current_user.last_name,
            },
            "tool": tool,
            "assistant_count": assignment_count,
            "assigned_assistants": assigned_assistants,
            "recent_executions": recent_executions,
            "success_rate": tool.get_success_rate(),
            "default_test_params": default_test_params,
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering tool view {tool_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load tool page")


@router.get("/tools/{tool_id}/edit", response_class=HTMLResponse)
async def tools_edit(
    request: Request,
    tool_id: int,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Edit tool page."""
    try:
        tool_service = ToolService()
        tool = await tool_service.get_tool(tool_id, current_user.organization_id)
        
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        return templates.TemplateResponse("tools/form.html", {
            "request": request,
            "current_user": current_user,
            "session": {
                "user_id": request.session.get("user_id"),
                "organization_id": request.session.get("organization_id"),
                "user_first_name": current_user.first_name,
                "user_last_name": current_user.last_name,
            },
            "tool": tool,
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering tool edit {tool_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load tool edit page")


@router.post("/tools", response_class=HTMLResponse)
async def tools_create(
    request: Request,
    name: str = Form(...),
    display_name: str = Form(...),
    description: str = Form(...),
    tool_type: str = Form(...),
    timeout_seconds: int = Form(30),
    retry_attempts: int = Form(3),
    is_active: bool = Form(False),
    is_public: bool = Form(False),
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Handle tool creation form submission."""
    try:
        # Get form data and parse JSON fields
        form_data = await request.form()
        
        # Parse configuration based on tool type
        configuration = {}
        if tool_type == 'endpoint':
            configuration = {
                'method': form_data.get('endpoint_method', 'GET'),
                'url': form_data.get('endpoint_url', ''),
                'headers': form_data.get('endpoint_headers', '{}'),
                'body_template': form_data.get('endpoint_body', '{}'),
            }
        elif tool_type == 'python_function':
            configuration = {
                'code': form_data.get('python_code', ''),
                'allowed_imports': form_data.get('python_imports', '').split(','),
            }
        elif tool_type == 'lambda':
            configuration = {
                'function_name': form_data.get('lambda_function_name', ''),
                'region': form_data.get('lambda_region', 'us-east-1'),
                'payload_template': form_data.get('lambda_payload', '{}'),
                'access_key': form_data.get('lambda_access_key', ''),
                'secret_key': form_data.get('lambda_secret_key', ''),
            }
        
        # Build function definition from parameters
        # This would need to be parsed from the form data
        # For now, using a basic structure
        function_definition = {
            'type': 'function',
            'function': {
                'name': name,
                'description': description,
                'parameters': {
                    'type': 'object',
                    'properties': {},
                    'required': []
                }
            }
        }
        
        tool_service = ToolService()
        tool = await tool_service.create_tool(
            organization_id=current_user.organization_id,
            user_id=current_user.id,
            name=name,
            display_name=display_name,
            description=description,
            tool_type=tool_type,
            configuration=configuration,
            function_definition=function_definition,
            timeout_seconds=timeout_seconds,
            retry_attempts=retry_attempts,
            is_active=is_active,
            is_public=is_public,
        )
        
        # Redirect to tool view page
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/tools/{tool.id}", status_code=303)
        
    except Exception as e:
        logger.error(f"Error creating tool: {str(e)}")
        # Return to form with error
        return templates.TemplateResponse("tools/form.html", {
            "request": request,
            "current_user": current_user,
            "session": {
                "user_id": request.session.get("user_id"),
                "organization_id": request.session.get("organization_id"),
                "user_first_name": current_user.first_name,
                "user_last_name": current_user.last_name,
            },
            "tool": None,
            "error": str(e),
        })


@router.post("/tools/{tool_id}/edit", response_class=HTMLResponse)
async def tools_update(
    request: Request,
    tool_id: int,
    
    current_user: User = Depends(get_current_user_flexible)
):
    """Handle tool update form submission."""
    try:
        tool_service = ToolService()
        tool = await tool_service.get_tool(tool_id, current_user.organization_id)
        
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        # Get form data
        form_data = await request.form()
        
        # Parse update data similar to create
        update_data = {}
        
        if 'display_name' in form_data:
            update_data['display_name'] = form_data['display_name']
        if 'description' in form_data:
            update_data['description'] = form_data['description']
        if 'timeout_seconds' in form_data:
            update_data['timeout_seconds'] = int(form_data['timeout_seconds'])
        if 'retry_attempts' in form_data:
            update_data['retry_attempts'] = int(form_data['retry_attempts'])
        if 'is_active' in form_data:
            update_data['is_active'] = form_data.get('is_active') == 'on'
        if 'is_public' in form_data:
            update_data['is_public'] = form_data.get('is_public') == 'on'
        
        # Update configuration based on tool type
        configuration = tool.configuration.copy()
        if tool.tool_type == 'endpoint':
            if 'endpoint_method' in form_data:
                configuration['method'] = form_data['endpoint_method']
            if 'endpoint_url' in form_data:
                configuration['url'] = form_data['endpoint_url']
            if 'endpoint_headers' in form_data:
                configuration['headers'] = form_data['endpoint_headers']
            if 'endpoint_body' in form_data:
                configuration['body_template'] = form_data['endpoint_body']
        elif tool.tool_type == 'python_function':
            if 'python_code' in form_data:
                configuration['code'] = form_data['python_code']
            if 'python_imports' in form_data:
                configuration['allowed_imports'] = form_data['python_imports'].split(',')
        elif tool.tool_type == 'lambda':
            if 'lambda_function_name' in form_data:
                configuration['function_name'] = form_data['lambda_function_name']
            if 'lambda_region' in form_data:
                configuration['region'] = form_data['lambda_region']
            if 'lambda_payload' in form_data:
                configuration['payload_template'] = form_data['lambda_payload']
            if 'lambda_access_key' in form_data:
                configuration['access_key'] = form_data['lambda_access_key']
            if 'lambda_secret_key' in form_data:
                configuration['secret_key'] = form_data['lambda_secret_key']
        
        update_data['configuration'] = configuration
        
        # Update the tool
        updated_tool = await tool_service.update_tool(tool_id, current_user.organization_id, **update_data)
        
        # Redirect to tool view page
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/tools/{updated_tool.id}", status_code=303)
        
    except Exception as e:
        logger.error(f"Error updating tool {tool_id}: {str(e)}")
        # Return to form with error
        return templates.TemplateResponse("tools/form.html", {
            "request": request,
            "current_user": current_user,
            "session": {
                "user_id": request.session.get("user_id"),
                "organization_id": request.session.get("organization_id"),
                "user_first_name": current_user.first_name,
                "user_last_name": current_user.last_name,
            },
            "tool": tool,
            "error": str(e),
        })
