"""
Tool Service module for managing custom tools.
Handles tool creation, configuration, and execution.
"""

import time
import datetime
from typing import Dict, Any, List, Optional
from sqlalchemy import and_, or_, desc, func, select

from app.db.models import Tool, AssistantTool, ToolExecutionLog
from app.db.database import get_async_db_session
from app.services.tool_executors.base import BaseToolExecutor
from app.services.tool_executors.endpoint_executor import EndpointToolExecutor
from app.services.tool_executors.python_executor import PythonToolExecutor
from app.services.tool_executors.lambda_executor import LambdaToolExecutor


class ToolService:
    """Service class for managing tools."""

    def __init__(self):
        pass

    @classmethod
    async def get_assistant_tools_static(cls, assistant_id: int) -> List[AssistantTool]:
        """
        Get all tools assigned to an assistant (static method).

        Args:
            assistant_id: Assistant ID

        Returns:
            List[AssistantTool]: List of tool assignments
        """
        async with await get_async_db_session() as db:
            result = await db.execute(
                select(AssistantTool).where(AssistantTool.assistant_id == assistant_id)
            )
            assignments = result.scalars().all()

            # Eagerly load the tool relationships to avoid lazy loading issues
            for assignment in assignments:
                # This will trigger loading of the tool if not already loaded
                _ = assignment.tool

            return assignments

    async def create_tool(
        self,
        organization_id: int,
        user_id: int,
        name: str,
        display_name: str,
        description: str,
        tool_type: str,
        configuration: Dict[str, Any],
        function_definition: Dict[str, Any],
        timeout_seconds: int = 30,
        retry_attempts: int = 3,
        is_active: bool = True,
        is_public: bool = False,
    ) -> Tool:
        """
        Create a new tool.

        Args:
            organization_id: Organization ID
            user_id: Creator user ID
            name: Tool function name (snake_case)
            display_name: Human-readable name
            description: Tool description for LLM
            tool_type: Type of tool ('endpoint', 'python_function', 'lambda')
            configuration: Tool-specific configuration
            function_definition: LLM function definition
            timeout_seconds: Execution timeout
            retry_attempts: Number of retry attempts
            is_active: Whether tool is active
            is_public: Whether tool is public

        Returns:
            Tool: Created tool instance

        Raises:
            ValueError: If tool name already exists in organization
        """
        async with await get_async_db_session() as db:
            # Check if tool name already exists in organization
            existing_tool_result = await db.execute(
                select(Tool).where(
                    and_(Tool.organization_id == organization_id, Tool.name == name)
                )
            )
            existing_tool = existing_tool_result.scalar_one_or_none()

            if existing_tool:
                raise ValueError(
                    f"Tool with name '{name}' already exists in this organization"
                )

            # Validate tool type
            if tool_type not in ["endpoint", "python_function", "lambda"]:
                raise ValueError(f"Invalid tool type: {tool_type}")

            # Validate configuration based on tool type
            self._validate_tool_configuration(tool_type, configuration)

            # Create tool
            tool = Tool(
                organization_id=organization_id,
                user_id=user_id,
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

            db.add(tool)
            await db.commit()
            await db.refresh(tool)

            return tool

    async def update_tool(self, tool_id: int, organization_id: int, **kwargs) -> Tool:
        """
        Update an existing tool.

        Args:
            tool_id: Tool ID
            organization_id: Organization ID
            **kwargs: Fields to update

        Returns:
            Tool: Updated tool instance

        Raises:
            ValueError: If tool not found or validation fails
        """
        async with await get_async_db_session() as db:
            tool = await self.get_tool(tool_id, organization_id)
            if not tool:
                raise ValueError(f"Tool with ID {tool_id} not found")

            # Update fields
            updatable_fields = [
                "display_name",
                "description",
                "configuration",
                "function_definition",
                "timeout_seconds",
                "retry_attempts",
                "is_active",
                "is_public",
            ]

            for field in updatable_fields:
                if field in kwargs:
                    setattr(tool, field, kwargs[field])

            # Validate configuration if updated
            if "configuration" in kwargs:
                self._validate_tool_configuration(
                    tool.tool_type, kwargs["configuration"]
                )

            tool.updated_at = datetime.datetime.utcnow()
            await db.commit()
            await db.refresh(tool)

            return tool

    async def get_tool(
        self, tool_id: int, organization_id: int = None
    ) -> Optional[Tool]:
        """
        Get a tool by ID.

        Args:
            tool_id: Tool ID
            organization_id: Organization ID (optional for filtering)

        Returns:
            Tool: Tool instance or None if not found
        """
        async with await get_async_db_session() as db:
            query = select(Tool).where(Tool.id == tool_id)

            if organization_id:
                query = query.where(Tool.organization_id == organization_id)

            result = await db.execute(query)
            return result.scalar_one_or_none()

    async def list_tools(
        self,
        organization_id: int,
        tool_type: str = None,
        is_active: bool = None,
        search: str = None,
        page: int = 1,
        per_page: int = 20,
        sort_by: str = "name",
        sort_order: str = "asc",
    ) -> Dict[str, Any]:
        """
        List tools with filtering and pagination.

        Args:
            organization_id: Organization ID
            tool_type: Filter by tool type
            is_active: Filter by active status
            search: Search term for name/description
            page: Page number
            per_page: Items per page
            sort_by: Sort field
            sort_order: Sort order ('asc' or 'desc')

        Returns:
            Dict containing tools list and pagination info
        """
        async with await get_async_db_session() as db:
            query = select(Tool).where(Tool.organization_id == organization_id)

            # Apply filters
            if tool_type:
                query = query.where(Tool.tool_type == tool_type)

            if is_active is not None:
                query = query.where(Tool.is_active == is_active)

            if search:
                search_term = f"%{search.lower()}%"
                query = query.where(
                    or_(
                        func.lower(Tool.name).like(search_term),
                        func.lower(Tool.display_name).like(search_term),
                        func.lower(Tool.description).like(search_term),
                    )
                )

            # Get total count
            count_query = select(func.count(Tool.id)).where(
                Tool.organization_id == organization_id
            )
            if tool_type:
                count_query = count_query.where(Tool.tool_type == tool_type)
            if is_active is not None:
                count_query = count_query.where(Tool.is_active == is_active)
            if search:
                search_term = f"%{search.lower()}%"
                count_query = count_query.where(
                    or_(
                        func.lower(Tool.name).like(search_term),
                        func.lower(Tool.display_name).like(search_term),
                        func.lower(Tool.description).like(search_term),
                    )
                )
            count_result = await db.execute(count_query)
            total_count = count_result.scalar()

            # Apply sorting
            sort_field = getattr(Tool, sort_by, Tool.name)
            if sort_order == "desc":
                query = query.order_by(desc(sort_field))
            else:
                query = query.order_by(sort_field)

            # Apply pagination
            offset = (page - 1) * per_page
            query = query.offset(offset).limit(per_page)

            result = await db.execute(query)
            tools = result.scalars().all()

            # Calculate pagination info
            total_pages = (total_count + per_page - 1) // per_page
            has_prev = page > 1
            has_next = page < total_pages

            return {
                "tools": tools,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_count": total_count,
                    "total_pages": total_pages,
                    "has_prev": has_prev,
                    "has_next": has_next,
                    "prev_page": page - 1 if has_prev else None,
                    "next_page": page + 1 if has_next else None,
                },
            }

    async def delete_tool(self, tool_id: int, organization_id: int) -> bool:
        """
        Delete a tool.

        Args:
            tool_id: Tool ID
            organization_id: Organization ID

        Returns:
            bool: True if deleted, False if not found
        """
        tool = await self.get_tool(tool_id, organization_id)
        if not tool:
            return False

        async with await get_async_db_session() as db:
            # Check if tool is assigned to any assistants
            assignment_count_result = await db.execute(
                select(func.count(AssistantTool.id)).where(
                    AssistantTool.tool_id == tool_id
                )
            )
            assignment_count = assignment_count_result.scalar()

            if assignment_count > 0:
                raise ValueError(
                    f"Cannot delete tool: it is assigned to {assignment_count} assistant(s)"
                )

            await db.delete(tool)
            await db.commit()
            return True

    async def duplicate_tool(
        self,
        tool_id: int,
        organization_id: int,
        new_name: str = None,
        new_display_name: str = None,
    ) -> Tool:
        """
        Duplicate an existing tool.

        Args:
            tool_id: Original tool ID
            organization_id: Organization ID
            new_name: New tool name (optional)
            new_display_name: New display name (optional)

        Returns:
            Tool: Duplicated tool instance
        """
        original_tool = await self.get_tool(tool_id, organization_id)
        if not original_tool:
            raise ValueError(f"Tool with ID {tool_id} not found")

        # Generate new names if not provided
        if not new_name:
            base_name = original_tool.name
            counter = 1

            async with await get_async_db_session() as db:
                while True:
                    new_name = f"{base_name}_copy_{counter}"
                    existing_result = await db.execute(
                        select(Tool).where(
                            and_(
                                Tool.organization_id == organization_id,
                                Tool.name == new_name,
                            )
                        )
                    )
                    existing = existing_result.scalar_one_or_none()
                    if not existing:
                        break
                    counter += 1

        if not new_display_name:
            new_display_name = f"{original_tool.display_name} (Copy)"

        # Create duplicate
        async with await get_async_db_session() as db:
            duplicate_tool = Tool(
                organization_id=organization_id,
                user_id=original_tool.user_id,
                name=new_name,
                display_name=new_display_name,
                description=original_tool.description,
                tool_type=original_tool.tool_type,
                configuration=original_tool.configuration.copy(),
                function_definition=original_tool.function_definition.copy(),
                timeout_seconds=original_tool.timeout_seconds,
                retry_attempts=original_tool.retry_attempts,
                is_active=original_tool.is_active,
                is_public=original_tool.is_public,
                parent_tool_id=original_tool.id,
            )

            db.add(duplicate_tool)
            await db.commit()
            await db.refresh(duplicate_tool)

            return duplicate_tool

    async def assign_tool_to_assistant(
        self,
        assistant_id: int,
        tool_id: int,
        user_id: int,
        enabled: bool = True,
        custom_configuration: Dict[str, Any] = None,
    ) -> AssistantTool:
        """
        Assign a tool to an assistant.

        Args:
            assistant_id: Assistant ID
            tool_id: Tool ID
            user_id: User ID making the assignment
            enabled: Whether the tool is enabled
            custom_configuration: Assistant-specific configuration overrides

        Returns:
            AssistantTool: Assignment instance
        """
        async with await get_async_db_session() as db:
            # Check if assignment already exists
            existing_result = await db.execute(
                select(AssistantTool).where(
                    and_(
                        AssistantTool.assistant_id == assistant_id,
                        AssistantTool.tool_id == tool_id,
                    )
                )
            )
            existing = existing_result.scalar_one_or_none()

            if existing:
                # Update existing assignment
                existing.enabled = enabled
                existing.custom_configuration = custom_configuration
                existing.assigned_at = datetime.datetime.utcnow()
                await db.commit()
                return existing

            # Create new assignment
            assignment = AssistantTool(
                assistant_id=assistant_id,
                tool_id=tool_id,
                assigned_by_user_id=user_id,
                enabled=enabled,
                custom_configuration=custom_configuration,
            )

            db.add(assignment)
            await db.commit()
            await db.refresh(assignment)

            return assignment

    async def unassign_tool_from_assistant(
        self, assistant_id: int, tool_id: int
    ) -> bool:
        """
        Unassign a tool from an assistant.

        Args:
            assistant_id: Assistant ID
            tool_id: Tool ID

        Returns:
            bool: True if unassigned, False if not found
        """
        async with await get_async_db_session() as db:
            assignment_result = await db.execute(
                select(AssistantTool).where(
                    and_(
                        AssistantTool.assistant_id == assistant_id,
                        AssistantTool.tool_id == tool_id,
                    )
                )
            )
            assignment = assignment_result.scalar_one_or_none()

            if not assignment:
                return False

            await db.delete(assignment)
            await db.commit()
            return True

    async def get_assistant_tools(self, assistant_id: int) -> List[AssistantTool]:
        """
        Get all tools assigned to an assistant.

        Args:
            assistant_id: Assistant ID

        Returns:
            List[AssistantTool]: List of tool assignments
        """
        async with await get_async_db_session() as db:
            result = await db.execute(
                select(AssistantTool).where(AssistantTool.assistant_id == assistant_id)
            )
            return result.scalars().all()

    async def execute_tool(
        self,
        tool_id: int,
        parameters: Dict[str, Any],
        assistant_id: int = None,
        call_id: int = None,
        execution_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool with given parameters.

        Args:
            tool_id: Tool ID
            parameters: Tool parameters
            assistant_id: Assistant ID (optional)
            call_id: Call ID (optional)
            execution_context: Additional context (optional)

        Returns:
            Dict containing execution result
        """
        async with await get_async_db_session() as db:
            tool_result = await db.execute(select(Tool).where(Tool.id == tool_id))
            tool = tool_result.scalar_one_or_none()
            if not tool:
                raise ValueError(f"Tool with ID {tool_id} not found")

            if not tool.is_active:
                raise ValueError(f"Tool '{tool.name}' is not active")

            # Create execution log
            log = ToolExecutionLog(
                tool_id=tool_id,
                assistant_id=assistant_id,
                call_id=call_id,
                parameters=parameters,
                execution_context=execution_context,
                status="running",
            )
            db.add(log)
            await db.commit()

            start_time = time.time()

            try:
                # Get tool executor
                executor = self._get_tool_executor(tool)

                # Merge custom configuration if from assistant
                config = tool.configuration.copy()
                if assistant_id:
                    assignment_result = await db.execute(
                        select(AssistantTool).where(
                            and_(
                                AssistantTool.assistant_id == assistant_id,
                                AssistantTool.tool_id == tool_id,
                            )
                        )
                    )
                    assignment = assignment_result.scalar_one_or_none()

                    if assignment and assignment.custom_configuration:
                        config.update(assignment.custom_configuration)

                # Execute tool
                result = executor.execute(parameters, config, execution_context)

                # Calculate duration
                end_time = time.time()
                duration_ms = int((end_time - start_time) * 1000)

                # Update execution log
                log.status = "success"
                log.result = result
                log.completed_at = datetime.datetime.utcnow()
                log.duration_ms = duration_ms

                # Update tool statistics
                tool.execution_count += 1
                tool.success_count += 1
                tool.last_executed_at = datetime.datetime.utcnow()

                # Update assignment statistics if applicable
                if assistant_id:
                    assignment_result = await db.execute(
                        select(AssistantTool).where(
                            and_(
                                AssistantTool.assistant_id == assistant_id,
                                AssistantTool.tool_id == tool_id,
                            )
                        )
                    )
                    assignment = assignment_result.scalar_one_or_none()

                    if assignment:
                        assignment.execution_count += 1
                        assignment.last_executed_at = datetime.datetime.utcnow()

                await db.commit()

                return {
                    "success": True,
                    "result": result,
                    "duration_ms": duration_ms,
                    "execution_log_id": log.id,
                }

            except Exception as e:
                # Calculate duration
                end_time = time.time()
                duration_ms = int((end_time - start_time) * 1000)

                # Update execution log
                log.status = "error"
                log.error_message = str(e)
                log.completed_at = datetime.datetime.utcnow()
                log.duration_ms = duration_ms

                # Update tool statistics
                tool.execution_count += 1
                tool.failure_count += 1
                tool.last_executed_at = datetime.datetime.utcnow()

                await db.commit()

                return {
                    "success": False,
                    "error": str(e),
                    "duration_ms": duration_ms,
                    "execution_log_id": log.id,
                }

    async def test_tool(
        self, tool_id: int, parameters: Dict[str, Any], organization_id: int = None
    ) -> Dict[str, Any]:
        """
        Test a tool with given parameters (without logging to database).

        Args:
            tool_id: Tool ID
            parameters: Tool parameters
            organization_id: Organization ID (optional for validation)

        Returns:
            Dict containing test result
        """
        tool = await self.get_tool(tool_id, organization_id)
        if not tool:
            raise ValueError(f"Tool with ID {tool_id} not found")

        start_time = time.time()

        try:
            # Get tool executor
            executor = self._get_tool_executor(tool)

            # Execute tool
            result = executor.execute(parameters, tool.configuration, {})

            # Calculate duration
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)

            return {
                "success": True,
                "result": result,
                "duration_ms": duration_ms,
                "message": "Tool executed successfully",
            }

        except Exception as e:
            # Calculate duration
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)

            return {
                "success": False,
                "error": str(e),
                "duration_ms": duration_ms,
                "message": f"Tool execution failed: {str(e)}",
            }

    async def get_tool_execution_logs(
        self,
        tool_id: int = None,
        assistant_id: int = None,
        call_id: int = None,
        status: str = None,
        limit: int = 100,
    ) -> List[ToolExecutionLog]:
        """
        Get tool execution logs with filtering.

        Args:
            tool_id: Tool ID filter
            assistant_id: Assistant ID filter
            call_id: Call ID filter
            status: Status filter
            limit: Maximum number of logs to return

        Returns:
            List[ToolExecutionLog]: List of execution logs
        """
        async with await get_async_db_session() as db:
            query = select(ToolExecutionLog)

            if tool_id:
                query = query.where(ToolExecutionLog.tool_id == tool_id)
            if assistant_id:
                query = query.where(ToolExecutionLog.assistant_id == assistant_id)
            if call_id:
                query = query.where(ToolExecutionLog.call_id == call_id)
            if status:
                query = query.where(ToolExecutionLog.status == status)

            query = query.order_by(desc(ToolExecutionLog.started_at)).limit(limit)
            result = await db.execute(query)
            return result.scalars().all()

    async def export_tool(self, tool_id: int, organization_id: int) -> Dict[str, Any]:
        """
        Export a tool configuration.

        Args:
            tool_id: Tool ID
            organization_id: Organization ID

        Returns:
            Dict: Tool export data
        """
        tool = await self.get_tool(tool_id, organization_id)
        if not tool:
            raise ValueError(f"Tool with ID {tool_id} not found")

        return {
            "name": tool.name,
            "display_name": tool.display_name,
            "description": tool.description,
            "tool_type": tool.tool_type,
            "configuration": tool.configuration,
            "function_definition": tool.function_definition,
            "timeout_seconds": tool.timeout_seconds,
            "retry_attempts": tool.retry_attempts,
            "version": tool.version,
            "exported_at": datetime.datetime.utcnow().isoformat(),
            "export_version": "1.0",
        }

    async def import_tool(
        self, organization_id: int, user_id: int, import_data: Dict[str, Any]
    ) -> Tool:
        """
        Import a tool from export data.

        Args:
            organization_id: Organization ID
            user_id: User ID
            import_data: Tool import data

        Returns:
            Tool: Imported tool instance
        """
        required_fields = [
            "name",
            "display_name",
            "description",
            "tool_type",
            "configuration",
            "function_definition",
        ]

        for field in required_fields:
            if field not in import_data:
                raise ValueError(f"Missing required field: {field}")

        # Generate unique name if conflicts
        base_name = import_data["name"]
        name = base_name
        counter = 1

        async with await get_async_db_session() as db:
            while True:
                existing_result = await db.execute(
                    select(Tool).where(
                        and_(Tool.organization_id == organization_id, Tool.name == name)
                    )
                )
                existing = existing_result.scalar_one_or_none()

                if not existing:
                    break

                name = f"{base_name}_imported_{counter}"
                counter += 1

        return await self.create_tool(
            organization_id=organization_id,
            user_id=user_id,
            name=name,
            display_name=import_data["display_name"],
            description=import_data["description"],
            tool_type=import_data["tool_type"],
            configuration=import_data["configuration"],
            function_definition=import_data["function_definition"],
            timeout_seconds=import_data.get("timeout_seconds", 30),
            retry_attempts=import_data.get("retry_attempts", 3),
            is_active=True,
            is_public=False,
        )

    def _validate_tool_configuration(
        self, tool_type: str, configuration: Dict[str, Any]
    ) -> None:
        """Validate tool configuration based on tool type."""
        if tool_type == "endpoint":
            required_fields = ["url", "method"]
            for field in required_fields:
                if field not in configuration:
                    raise ValueError(
                        f"Missing required field for endpoint tool: {field}"
                    )

        elif tool_type == "python_function":
            required_fields = ["code"]
            for field in required_fields:
                if field not in configuration:
                    raise ValueError(
                        f"Missing required field for Python function tool: {field}"
                    )

        elif tool_type == "lambda":
            required_fields = ["function_name", "region"]
            for field in required_fields:
                if field not in configuration:
                    raise ValueError(f"Missing required field for Lambda tool: {field}")

    def _get_tool_executor(self, tool: Tool) -> BaseToolExecutor:
        """Get the appropriate tool executor for a tool type."""
        if tool.tool_type == "endpoint":
            return EndpointToolExecutor(
                timeout_seconds=tool.timeout_seconds, retry_attempts=tool.retry_attempts
            )
        elif tool.tool_type == "python_function":
            return PythonToolExecutor(
                timeout_seconds=tool.timeout_seconds, retry_attempts=tool.retry_attempts
            )
        elif tool.tool_type == "lambda":
            return LambdaToolExecutor(
                timeout_seconds=tool.timeout_seconds, retry_attempts=tool.retry_attempts
            )
        else:
            raise ValueError(f"Unknown tool type: {tool.tool_type}")
