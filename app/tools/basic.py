"""
Basic tools for the assistant.
"""
# pylint: disable=missing-function-docstring

from app.db.models import Assistant


def get_end_call_tool(assistant: Assistant) -> dict:
    """Get the end call tool configuration."""
    tools_settings = getattr(assistant, 'tools_settings', {}) or {}
    end_call_config = tools_settings.get('end_call', {})
    
    if not end_call_config.get('enabled', False):
        return None

    scenarios = end_call_config.get('scenarios', [])
    
    # Build description with scenarios
    description = "Use this function to end the call."
    if scenarios:
        description += f" The scenarios are: {', '.join(scenarios)}"
    
    END_CALL_TOOL = {
        "type": "function",
        "function": {
            "name": "endCall",
            "description": description,
        },
    }

    return END_CALL_TOOL


def get_transfer_call_tool(assistant: Assistant) -> dict:
    """Get the transfer call tool configuration."""
    tools_settings = getattr(assistant, 'tools_settings', {}) or {}
    transfer_call_config = tools_settings.get('transfer_call', {})
    
    if not transfer_call_config.get('enabled', False):
        return None

    transfer_numbers = transfer_call_config.get('transfer_numbers', [])
    scenarios = transfer_call_config.get('scenarios', [])

    if not transfer_numbers:
        return None  # Can't transfer without phone numbers

    # Build description with scenarios
    description = "Use this function to transfer the call to a human representative."
    if scenarios:
        description += f" The scenarios are: {', '.join(scenarios)}"
    
    return {
        "type": "function",
        "function": {
            "name": "transferCall",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "enum": transfer_numbers,
                        "description": "The destination to transfer the call to.",
                    }
                },
                "required": ["destination"],
            },
        },
    }


def get_all_tools(assistant: Assistant) -> list:
    """Get all enabled tools for the assistant."""
    tools = []
    
    # Add end call tool if enabled
    end_call_tool = get_end_call_tool(assistant)
    if end_call_tool:
        tools.append(end_call_tool)
    
    # Add transfer call tool if enabled
    transfer_call_tool = get_transfer_call_tool(assistant)
    if transfer_call_tool:
        tools.append(transfer_call_tool)
    
    # Add custom tools if any
    tools_settings = getattr(assistant, 'tools_settings', {}) or {}
    custom_tools = tools_settings.get('custom_tools', [])
    for tool in custom_tools:
        if tool.get('enabled', False):
            tools.append(tool.get('definition', {}))
    
    return tools
