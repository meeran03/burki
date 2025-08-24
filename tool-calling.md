# Tool Calling Implementation Plan

## Overview

This document outlines the implementation plan for adding comprehensive tool calling support to our voice assistant system. Based on research into Vapi's architecture and analysis of our existing codebase, we will implement three types of tool calling capabilities:

1. **Endpoint-triggered tools** - Tools that call HTTP endpoints when triggered
2. **Dashboard-defined Python functions** - Tools that execute user-defined Python code
3. **AWS Lambda function tools** - Tools that invoke AWS Lambda functions

## Current State Analysis

### Existing Tool Infrastructure

Our system already has a basic tool framework:
- `tools_settings` JSON column in the Assistant model for tool configurations
- Basic tools implemented: `endCall` and `transferCall`
- Tool execution handled in `BaseLLMProvider._handle_tool_call()` method
- Tool definitions generated in `app/tools/basic.py`
- LangChain integration for tool binding and execution

### Current Tool Configuration Structure

```json
{
  "enabled_tools": ["endCall", "transferCall"],
  "end_call": {
    "enabled": false,
    "scenarios": [],
    "custom_message": null
  },
  "transfer_call": {
    "enabled": false,
    "scenarios": [],
    "transfer_numbers": [],
    "custom_message": null
  },
  "custom_tools": []
}
```

## Proposed Architecture

### Key Architectural Decision: Tool Independence

**Tools as Independent Entities**: Tools are now independent entities that can be shared across multiple assistants within an organization. This design provides several key benefits:

1. **Reusability**: Create a tool once, use it across multiple assistants
2. **Maintainability**: Update a tool's logic in one place, affects all assistants using it
3. **Organization-wide Library**: Build a library of tools that teams can share
4. **Better Testing**: Test tools independently from assistant configurations
5. **Template System**: Create template tools that can be cloned and customized
6. **Public Marketplace**: Share commonly used tools across organizations
7. **Version Control**: Track tool versions and updates independently
8. **Resource Optimization**: Reduce duplication and improve database efficiency

### Tool-Assistant Relationship Model

- **Many-to-Many**: One tool can be assigned to multiple assistants
- **Configuration Overrides**: Each assistant can have custom configuration for the same tool
- **Selective Enablement**: Tools can be enabled/disabled per assistant
- **Execution Order**: Define tool priority/order per assistant

### 1. Database Schema Extensions

#### New Table: `tools` (Independent Tool Definitions)
```sql
CREATE TABLE tools (
    id SERIAL PRIMARY KEY,
    organization_id INTEGER NOT NULL REFERENCES organizations(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    
    -- Tool identification
    name VARCHAR(100) NOT NULL,
    display_name VARCHAR(200) NOT NULL,
    description TEXT,
    tool_type VARCHAR(50) NOT NULL, -- 'endpoint', 'python_function', 'lambda'
    
    -- Tool configuration (JSON)
    configuration JSON NOT NULL DEFAULT '{}',
    
    -- Tool definition (for LLM)
    function_definition JSON NOT NULL DEFAULT '{}',
    
    -- Security and access
    is_active BOOLEAN DEFAULT true,
    require_confirmation BOOLEAN DEFAULT false,
    allowed_contexts JSON DEFAULT '[]', -- ['call', 'sms', 'webhook']
    
    -- Execution settings
    timeout_seconds INTEGER DEFAULT 30,
    retry_attempts INTEGER DEFAULT 3,
    
    -- Monitoring
    execution_count INTEGER DEFAULT 0,
    last_executed_at TIMESTAMP,
    
    -- Versioning and sharing
    version VARCHAR(20) DEFAULT '1.0.0',
    is_public BOOLEAN DEFAULT false, -- Can be shared across organization
    is_template BOOLEAN DEFAULT false, -- Is this a template tool
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(organization_id, name),
    INDEX(organization_id),
    INDEX(user_id),
    INDEX(tool_type),
    INDEX(is_public),
    INDEX(is_template)
);

#### New Table: `assistant_tools` (Many-to-Many Relationship)
```sql
CREATE TABLE assistant_tools (
    id SERIAL PRIMARY KEY,
    assistant_id INTEGER NOT NULL REFERENCES assistants(id) ON DELETE CASCADE,
    tool_id INTEGER NOT NULL REFERENCES tools(id) ON DELETE CASCADE,
    
    -- Tool-specific configuration for this assistant
    enabled BOOLEAN DEFAULT true,
    custom_configuration JSON DEFAULT '{}', -- Override tool defaults
    execution_order INTEGER DEFAULT 0, -- For ordered execution
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(assistant_id, tool_id),
    INDEX(assistant_id),
    INDEX(tool_id)
);
```

#### New Table: `tool_execution_logs`
```sql
CREATE TABLE tool_execution_logs (
    id SERIAL PRIMARY KEY,
    tool_id INTEGER NOT NULL REFERENCES tools(id) ON DELETE CASCADE,
    assistant_id INTEGER REFERENCES assistants(id),
    call_id INTEGER REFERENCES calls(id),
    
    -- Execution details
    execution_id VARCHAR(100) NOT NULL, -- UUID for tracking
    input_parameters JSON,
    output_result JSON,
    
    -- Status and timing
    status VARCHAR(50) NOT NULL, -- 'success', 'error', 'timeout', 'cancelled'
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_ms INTEGER,
    
    -- Error handling
    error_message TEXT,
    error_code VARCHAR(50),
    retry_count INTEGER DEFAULT 0,
    
    -- Context
    trigger_context VARCHAR(50), -- 'call', 'sms', 'webhook', 'test'
    user_id INTEGER REFERENCES users(id),
    
    INDEX(tool_id),
    INDEX(assistant_id),
    INDEX(call_id),
    INDEX(execution_id),
    INDEX(status),
    INDEX(started_at)
);
```

### 2. Tool Configuration Schema

#### Endpoint-Triggered Tools
```json
{
  "type": "endpoint",
  "method": "POST", // GET, POST, PUT, DELETE
  "url": "https://api.example.com/webhook",
  "headers": {
    "Authorization": "Bearer ${API_KEY}",
    "Content-Type": "application/json"
  },
  "authentication": {
    "type": "bearer", // "none", "bearer", "api_key", "basic"
    "token_field": "api_key", // field name in assistant config
    "header_name": "Authorization" // or "X-API-Key"
  },
  "request_body_template": {
    "call_sid": "${call_sid}",
    "phone_number": "${phone_number}",
    "parameters": "${parameters}"
  },
  "response_handling": {
    "success_field": "success",
    "message_field": "message",
    "data_field": "data"
  }
}
```

#### Python Function Tools
```json
{
  "type": "python_function",
  "code": "def execute_tool(parameters, context):\n    # User defined code\n    result = parameters.get('query', 'default')\n    return {'message': f'Processed: {result}', 'success': True}",
  "allowed_imports": ["datetime", "json", "requests", "math"],
  "execution_timeout": 10,
  "memory_limit_mb": 128,
  "environment_variables": ["DATABASE_URL", "API_KEY"]
}
```

#### AWS Lambda Tools
```json
{
  "type": "lambda",
  "function_name": "my-lambda-function",
  "aws_region": "us-east-1",
  "invocation_type": "RequestResponse", // or "Event"
  "payload_template": {
    "call_context": {
      "call_sid": "${call_sid}",
      "phone_number": "${phone_number}"
    },
    "parameters": "${parameters}"
  },
  "credentials": {
    "access_key_field": "aws_access_key",
    "secret_key_field": "aws_secret_key"
  }
}
```

### 3. Tool Definition Schema (for LLM)

Standard OpenAI function calling format:
```json
{
  "type": "function",
  "function": {
    "name": "search_knowledge_base",
    "description": "Search the company knowledge base for information",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query"
        },
        "category": {
          "type": "string",
          "enum": ["products", "support", "billing"],
          "description": "Category to search in"
        }
      },
      "required": ["query"]
    }
  }
}
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

#### 1.1 Database Migration
- Create migration for `tools`, `assistant_tools`, and `tool_execution_logs` tables
- Add indexes for performance
- Update Assistant model to include many-to-many relationship to tools

#### 1.2 Core Tool Engine
- Create `ToolExecutionEngine` class in `app/services/tool_execution_service.py`
- Implement base tool executor interface
- Add tool validation and security checks
- Implement execution logging

#### 1.3 Tool Type Implementations

**Endpoint Tool Executor:**
```python
class EndpointToolExecutor:
    async def execute(self, tool_config: dict, parameters: dict, context: dict) -> dict:
        # Implement HTTP request with timeout, retries, error handling
        pass
```

**Python Function Tool Executor:**
```python
class PythonFunctionExecutor:
    async def execute(self, tool_config: dict, parameters: dict, context: dict) -> dict:
        # Implement sandboxed Python execution using RestrictedPython
        pass
```

**Lambda Function Tool Executor:**
```python
class LambdaToolExecutor:
    async def execute(self, tool_config: dict, parameters: dict, context: dict) -> dict:
        # Implement AWS Lambda invocation
        pass
```

### Phase 2: LLM Integration (Week 2-3)

#### 2.1 Update Tool Loading
- Extend `get_all_tools()` in `app/tools/basic.py` to include tools from database
- Load tools associated with the assistant through `assistant_tools` relationship
- Generate proper function definitions for LLM
- Handle tool-specific configuration overrides per assistant

#### 2.2 Update Tool Execution
- Extend `BaseLLMProvider._handle_tool_call()` to handle tools from database
- Load tool configuration and merge with assistant-specific overrides
- Add proper error handling and fallbacks
- Implement tool execution result processing

#### 2.3 Context Management
- Pass call context (call_sid, phone_numbers, assistant info) to tools
- Add user and organization context
- Implement parameter interpolation for templates

### Phase 3: Dashboard UI (Week 3-4)

#### 3.1 Tool Management Interface
- Create organization-wide tools listing page
- Add tool creation/editing forms
- Implement tool testing interface
- Add execution logs viewer
- Tool library/marketplace for shared tools
- Assistant-specific tool assignment interface

#### 3.2 Tool Builder Components
- Visual tool configuration builder
- Code editor for Python functions
- Endpoint configuration form
- Lambda function selector

#### 3.3 Security and Validation
- Input validation for all tool configurations
- API key management for endpoint tools
- Sandbox restrictions for Python tools
- AWS credentials management

### Phase 4: Advanced Features (Week 4-5)

#### 4.1 Tool Marketplace
- Template library for common tools
- Import/export tool configurations
- Shareable tool templates

#### 4.2 Monitoring and Analytics
- Tool execution metrics dashboard
- Performance monitoring
- Error rate tracking
- Usage analytics

#### 4.3 Advanced Security
- Tool approval workflow
- Rate limiting per tool
- Audit logging
- Permission-based access control

## Security Considerations

### 1. Python Function Execution
- Use `RestrictedPython` for safe code execution
- Whitelist allowed imports and built-ins
- Implement resource limits (memory, CPU, execution time)
- Sandbox execution environment
- Validate user code before saving

### 2. Endpoint Tools
- Validate URLs and prevent SSRF attacks
- Implement request timeouts
- Rate limiting on external API calls
- Secure credential storage
- SSL/TLS verification

### 3. AWS Lambda Tools
- Secure AWS credential management
- IAM role-based access control
- Function execution monitoring
- VPC configuration for security
- Cost monitoring and limits

### 4. General Security
- Input validation and sanitization
- Output sanitization
- Audit logging of all tool executions
- Tool approval workflow for sensitive operations
- User permission validation

## API Endpoints

### Tool Management API
```python
# Organization-wide tool management
# GET /api/organization/tools
# POST /api/organization/tools
# GET /api/organization/tools/{tool_id}
# PUT /api/organization/tools/{tool_id}
# DELETE /api/organization/tools/{tool_id}
# POST /api/organization/tools/{tool_id}/test

# Assistant-specific tool assignment
# GET /api/assistants/{assistant_id}/tools
# POST /api/assistants/{assistant_id}/tools/{tool_id}/assign
# PUT /api/assistants/{assistant_id}/tools/{tool_id}/configure
# DELETE /api/assistants/{assistant_id}/tools/{tool_id}/unassign

# Tool library/marketplace
# GET /api/tools/public
# GET /api/tools/templates
# POST /api/organization/tools/from-template/{template_id}
```

### Tool Templates API
```python
# GET /api/tool-templates
# GET /api/tool-templates/{template_id}
# POST /api/assistants/{assistant_id}/tools/from-template/{template_id}
```

### Execution Logs API
```python
# GET /api/organization/tools/{tool_id}/logs
# GET /api/assistants/{assistant_id}/tool-executions
# GET /api/calls/{call_id}/tool-executions
```

## Testing Strategy

### 1. Unit Tests
- Tool executor implementations
- Configuration validation
- Security restrictions
- Error handling

### 2. Integration Tests
- End-to-end tool execution
- LLM integration
- Database operations
- External API calls

### 3. Security Tests
- Code injection attempts
- SSRF prevention
- Rate limiting
- Permission validation

### 4. Performance Tests
- Tool execution performance
- Concurrent execution handling
- Resource usage monitoring
- Timeout handling

## Monitoring and Observability

### 1. Metrics
- Tool execution success/failure rates
- Average execution time per tool type
- API call latencies for endpoint tools
- Resource usage for Python functions

### 2. Logging
- Structured logging for all tool executions
- Error logging with context
- Security event logging
- Performance metrics logging

### 3. Alerting
- Tool execution failures
- High error rates
- Unusual execution patterns
- Security violations

## Migration Strategy

### 1. Backward Compatibility
- Existing tools (`endCall`, `transferCall`) continue to work
- Gradual migration of basic tools to new system
- Support for both old and new configuration formats during transition

### 2. Data Migration
- Migrate existing `tools_settings` to new structure
- Preserve existing tool configurations
- Add new fields with sensible defaults

### 3. Feature Rollout
- Feature flags for new tool types
- Gradual rollout to user groups
- Monitoring and rollback capabilities

## Documentation Requirements

### 1. User Documentation
- Tool creation guide
- Best practices for each tool type
- Security guidelines
- Troubleshooting guide

### 2. Developer Documentation
- API documentation
- Tool executor interface
- Extension guidelines
- Security requirements

### 3. Examples and Templates
- Common tool patterns
- Code examples for Python functions
- Endpoint configuration examples
- Lambda function templates

## Success Metrics

### 1. Adoption Metrics
- Number of custom tools created
- Tool execution frequency
- User engagement with tool builder

### 2. Performance Metrics
- Tool execution success rate (target: >95%)
- Average execution time (target: <2 seconds)
- System reliability during tool execution

### 3. Security Metrics
- Zero security incidents
- Successful prevention of malicious code execution
- Audit compliance scores

## Future Enhancements

### 1. Advanced Tool Types
- Database query tools
- File system operations
- Email/SMS sending tools
- Calendar integration tools

### 2. Tool Orchestration
- Multi-step tool workflows
- Conditional tool execution
- Tool result chaining
- Parallel tool execution

### 3. AI-Powered Features
- Tool recommendation engine
- Automatic tool parameter detection
- Intelligent error recovery
- Performance optimization suggestions

## Frontend Implementation Plan

### Overview
The frontend will follow your existing design patterns using the electric slate theme with dark backgrounds, gradient accents, and consistent styling. The tool management interface will be integrated into the current navigation structure.

### 1. Navigation Structure Enhancement

#### Main Navigation Addition
Add new navigation items to the main navigation:
```html
<!-- Add to main navigation in base.html -->
<a href="/tools" class="nav-link">
    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
    </svg>
    Tools Library
</a>
```

### 2. Tool Library Interface (`/tools`)

#### Main Tools Page Layout
Following your existing patterns from assistants/index.html:

```html
<!-- Header Section with Electric Slate Theme -->
<div class="relative overflow-hidden rounded-2xl p-6" style="background: linear-gradient(135deg, #10B981, #A3FFAE);">
    <div class="absolute inset-0 bg-black/20"></div>
    <div class="relative z-10">
        <div class="flex flex-col lg:flex-row lg:items-center lg:justify-between">
            <div>
                <h1 class="font-satoshi text-3xl font-black text-white mb-2">
                    🛠️ Tools Library
                </h1>
                <p class="text-white/80 text-base max-w-2xl font-inter">
                    Create, manage, and share tools across your organization. Build once, use everywhere.
                </p>
            </div>
            <div class="mt-4 lg:mt-0 flex items-center space-x-3">
                <button id="import-tools-btn" class="px-4 py-2 text-sm font-medium text-white bg-white/20 hover:bg-white/30 rounded-lg transition-colors border border-white/20">
                    <svg class="w-4 h-4 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
                    </svg>
                    Import Tools
                </button>
                <a href="/tools/new" class="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-white/20 hover:bg-white/30 rounded-lg transition-colors border border-white/20">
                    <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                    </svg>
                    New Tool
                </a>
            </div>
        </div>
    </div>
</div>

<!-- Stats Overview -->
<div class="grid grid-cols-1 md:grid-cols-4 gap-6">
    <div class="auth-card p-6 rounded-xl">
        <div class="flex items-center">
            <div class="flex-shrink-0">
                <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-[#10B981] to-[#A3FFAE] flex items-center justify-center">
                    <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
                    </svg>
                </div>
            </div>
            <div class="ml-4">
                <div class="text-2xl font-bold text-[#E6EDF3]">{{ tools_count }}</div>
                <div class="text-sm text-[#7D8590]">Total Tools</div>
            </div>
        </div>
    </div>
    <!-- More stats cards for Active Tools, Executions, Success Rate -->
</div>
```

#### Tool Cards Layout
```html
<!-- Tool Cards Grid -->
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    {% for tool in tools %}
    <div class="auth-card rounded-xl p-6 hover:bg-[#161B22] transition-colors">
        <div class="flex items-start justify-between mb-4">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-[#10B981] to-[#A3FFAE] flex items-center justify-center">
                    <!-- Tool type icon -->
                    {% if tool.tool_type == 'endpoint' %}
                        <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                        </svg>
                    {% elif tool.tool_type == 'python_function' %}
                        <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                        </svg>
                    {% elif tool.tool_type == 'lambda' %}
                        <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                        </svg>
                    {% endif %}
                </div>
                <div>
                    <h3 class="text-sm font-semibold text-[#E6EDF3]">{{ tool.display_name }}</h3>
                    <p class="text-xs text-[#7D8590]">{{ tool.tool_type|title }} Tool</p>
                </div>
            </div>
            <div class="flex items-center space-x-2">
                {% if tool.is_public %}
                <span class="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-[#10B981]/20 text-[#10B981]">
                    🌍 Public
                </span>
                {% endif %}
                <div class="relative">
                    <button class="tool-menu-btn p-1 text-[#7D8590] hover:text-white rounded" data-tool-id="{{ tool.id }}">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
                        </svg>
                    </button>
                </div>
            </div>
        </div>
        
        <p class="text-sm text-[#7D8590] mb-4 line-clamp-2">{{ tool.description }}</p>
        
        <div class="flex items-center justify-between text-xs text-[#7D8590] mb-4">
            <span>{{ tool.execution_count }} executions</span>
            <span>Updated {{ tool.updated_at|timeago }}</span>
        </div>
        
        <div class="flex items-center justify-between">
            <div class="flex items-center space-x-2">
                <button onclick="testTool({{ tool.id }})" class="px-3 py-1 text-xs font-medium text-blue-400 bg-blue-500/20 hover:bg-blue-500/30 rounded transition-colors">
                    🧪 Test
                </button>
                <a href="/tools/{{ tool.id }}" class="px-3 py-1 text-xs font-medium text-[#7D8590] hover:text-white bg-white/10 hover:bg-white/20 rounded transition-colors">
                    View
                </a>
            </div>
            <div class="text-xs text-[#7D8590]">
                Used by {{ tool.assistant_count }} assistants
            </div>
        </div>
    </div>
    {% endfor %}
</div>
```

### 3. Tool Builder Interface (`/tools/new`)

#### Visual Tool Builder
```html
<!-- Multi-step Tool Builder -->
<div class="max-w-4xl mx-auto space-y-8">
    <!-- Progress Steps -->
    <div class="auth-card p-6 rounded-xl">
        <div class="flex items-center justify-between mb-4">
            <h2 class="font-satoshi text-lg font-semibold text-[#E6EDF3]">Tool Creation Progress</h2>
        </div>
        <div class="flex items-center space-x-4">
            <div class="step active" data-step="1">
                <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-[#10B981] to-[#A3FFAE] flex items-center justify-center">
                    <span class="text-white text-sm font-bold">1</span>
                </div>
                <span class="text-white/90 text-sm font-inter ml-2">Basic Info</span>
            </div>
            <div class="step" data-step="2">
                <div class="w-8 h-8 rounded-lg bg-[#30363D] flex items-center justify-center">
                    <span class="text-white/60 text-sm font-bold">2</span>
                </div>
                <span class="text-white/60 text-sm font-inter ml-2">Configuration</span>
            </div>
            <div class="step" data-step="3">
                <div class="w-8 h-8 rounded-lg bg-[#30363D] flex items-center justify-center">
                    <span class="text-white/60 text-sm font-bold">3</span>
                </div>
                <span class="text-white/60 text-sm font-inter ml-2">Testing</span>
            </div>
        </div>
    </div>

    <!-- Step 1: Basic Information -->
    <div id="step-1" class="auth-card p-6 rounded-xl">
        <h3 class="text-lg font-semibold text-[#E6EDF3] mb-6">Basic Information</h3>
        
        <!-- Tool Type Selection -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div class="tool-type-card" data-type="endpoint">
                <div class="p-6 border-2 border-[#30363D] rounded-lg hover:border-[#10B981] transition-colors cursor-pointer">
                    <div class="flex items-center space-x-3 mb-3">
                        <div class="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                            <svg class="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                            </svg>
                        </div>
                        <h4 class="text-sm font-semibold text-[#E6EDF3]">API Endpoint</h4>
                    </div>
                    <p class="text-xs text-[#7D8590]">Call external APIs and webhooks</p>
                </div>
            </div>
            
            <div class="tool-type-card" data-type="python_function">
                <div class="p-6 border-2 border-[#30363D] rounded-lg hover:border-[#10B981] transition-colors cursor-pointer">
                    <div class="flex items-center space-x-3 mb-3">
                        <div class="w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center">
                            <svg class="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                            </svg>
                        </div>
                        <h4 class="text-sm font-semibold text-[#E6EDF3]">Python Function</h4>
                    </div>
                    <p class="text-xs text-[#7D8590]">Execute custom Python code</p>
                </div>
            </div>
            
            <div class="tool-type-card" data-type="lambda">
                <div class="p-6 border-2 border-[#30363D] rounded-lg hover:border-[#10B981] transition-colors cursor-pointer">
                    <div class="flex items-center space-x-3 mb-3">
                        <div class="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                            <svg class="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                            </svg>
                        </div>
                        <h4 class="text-sm font-semibold text-[#E6EDF3]">AWS Lambda</h4>
                    </div>
                    <p class="text-xs text-[#7D8590]">Invoke serverless functions</p>
                </div>
            </div>
        </div>
        
        <!-- Basic Fields -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
                <label class="block text-sm font-medium text-[#E6EDF3] mb-2">Tool Name</label>
                <input type="text" name="name" class="block w-full px-4 py-3 rounded-lg bg-[#0D1117] border border-[#30363D] text-[#E6EDF3] focus:ring-2 focus:ring-[#10B981] focus:border-[#10B981]" placeholder="search_knowledge_base">
            </div>
            <div>
                <label class="block text-sm font-medium text-[#E6EDF3] mb-2">Display Name</label>
                <input type="text" name="display_name" class="block w-full px-4 py-3 rounded-lg bg-[#0D1117] border border-[#30363D] text-[#E6EDF3] focus:ring-2 focus:ring-[#10B981] focus:border-[#10B981]" placeholder="Search Knowledge Base">
            </div>
        </div>
        
        <div class="mt-6">
            <label class="block text-sm font-medium text-[#E6EDF3] mb-2">Description</label>
            <textarea name="description" rows="3" class="block w-full px-4 py-3 rounded-lg bg-[#0D1117] border border-[#30363D] text-[#E6EDF3] focus:ring-2 focus:ring-[#10B981] focus:border-[#10B981]" placeholder="Search the company knowledge base for relevant information"></textarea>
        </div>
    </div>
</div>
```

#### Code Editor Component (for Python functions)
```javascript
// Monaco Editor integration for Python functions
const editorContainer = document.getElementById('python-code-editor');
const editor = monaco.editor.create(editorContainer, {
    value: `def execute_tool(parameters, context):
    """
    Execute the tool with given parameters.
    
    Args:
        parameters: Dict containing the parameters passed by the LLM
        context: Dict containing call context (call_sid, phone_numbers, etc.)
    
    Returns:
        Dict with 'success', 'message', and optional 'data' fields
    """
    query = parameters.get('query', '')
    
    # Your custom logic here
    result = f"Processed query: {query}"
    
    return {
        'success': True,
        'message': result,
        'data': {'processed_query': query}
    }`,
    language: 'python',
    theme: 'vs-dark',
    minimap: { enabled: false },
    fontSize: 14,
    wordWrap: 'on'
});
```

### 4. Assistant Tool Assignment Interface

#### Enhanced Tools Section in Assistant Form
```html
<!-- Enhanced Tools Section -->
<div class="auth-card p-6 rounded-xl">
    <div class="flex items-center justify-between mb-6">
        <div>
            <h3 class="text-lg font-semibold text-[#E6EDF3]">Tools & Actions</h3>
            <p class="text-sm text-[#7D8590] mt-1">Configure tools that this assistant can use during calls</p>
        </div>
        <button onclick="openToolLibrary()" class="px-4 py-2 text-sm font-medium text-white bg-[#10B981] hover:bg-[#059669] rounded-lg transition-colors">
            <svg class="w-4 h-4 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
            Add Tool
        </button>
    </div>
    
    <!-- Basic Tools (existing) -->
    <div class="space-y-4 mb-6">
        <h4 class="text-sm font-semibold text-[#E6EDF3]">Built-in Tools</h4>
        <!-- Include existing end call and transfer call tools -->
        {% include "assistants/tools_section.html" %}
    </div>
    
    <!-- Custom Tools -->
    <div class="space-y-4">
        <div class="flex items-center justify-between">
            <h4 class="text-sm font-semibold text-[#E6EDF3]">Custom Tools</h4>
            <span class="text-xs text-[#7D8590]">{{ assistant_tools|length }} tools assigned</span>
        </div>
        
        <div id="assigned-tools" class="space-y-3">
            {% for tool_assignment in assistant_tools %}
            <div class="tool-assignment-card bg-[#161B22] border border-[#30363D] rounded-lg p-4">
                <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-3">
                        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-[#10B981] to-[#A3FFAE] flex items-center justify-center">
                            <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <!-- Tool type icon -->
                            </svg>
                        </div>
                        <div>
                            <h5 class="text-sm font-medium text-[#E6EDF3]">{{ tool_assignment.tool.display_name }}</h5>
                            <p class="text-xs text-[#7D8590]">{{ tool_assignment.tool.tool_type|title }} • {{ tool_assignment.tool.execution_count }} executions</p>
                        </div>
                    </div>
                    <div class="flex items-center space-x-2">
                        <label class="flex items-center">
                            <input type="checkbox" name="tool_{{ tool_assignment.tool.id }}_enabled" {% if tool_assignment.enabled %}checked{% endif %} class="w-4 h-4 rounded border-[#30363D] bg-[#0D1117] text-[#10B981]">
                            <span class="ml-2 text-xs text-[#7D8590]">Enabled</span>
                        </label>
                        <button onclick="configureToolForAssistant({{ tool_assignment.tool.id }})" class="p-1 text-[#7D8590] hover:text-white rounded" title="Configure">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
                            </svg>
                        </button>
                        <button onclick="removeToolFromAssistant({{ tool_assignment.tool.id }})" class="p-1 text-red-400 hover:text-red-300 rounded" title="Remove">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                        </button>
                    </div>
                </div>
                
                <!-- Tool Configuration Override -->
                <div class="tool-config-override mt-3 hidden" id="tool-config-{{ tool_assignment.tool.id }}">
                    <div class="border-t border-[#30363D] pt-3">
                        <h6 class="text-xs font-medium text-[#E6EDF3] mb-2">Assistant-Specific Configuration</h6>
                        <textarea name="tool_{{ tool_assignment.tool.id }}_config" class="w-full px-3 py-2 text-xs bg-[#0D1117] border border-[#30363D] rounded text-[#E6EDF3]" rows="3" placeholder='{"timeout": 30, "custom_param": "value"}'></textarea>
                        <p class="text-xs text-[#7D8590] mt-1">Override default tool configuration for this assistant (JSON format)</p>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <!-- Empty State -->
        {% if not assistant_tools %}
        <div class="text-center py-8 border-2 border-dashed border-[#30363D] rounded-lg">
            <div class="w-12 h-12 rounded-lg bg-gradient-to-br from-[#10B981] to-[#A3FFAE] flex items-center justify-center mx-auto mb-3">
                <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
                </svg>
            </div>
            <h5 class="text-sm font-medium text-[#E6EDF3] mb-2">No Custom Tools Assigned</h5>
            <p class="text-xs text-[#7D8590] mb-4">Add tools from your organization library to extend this assistant's capabilities</p>
            <button onclick="openToolLibrary()" class="px-4 py-2 text-sm font-medium text-white bg-[#10B981] hover:bg-[#059669] rounded-lg transition-colors">
                Browse Tools
            </button>
        </div>
        {% endif %}
    </div>
</div>
```

### 5. Tool Library Modal/Overlay
```html
<!-- Tool Library Modal -->
<div id="tool-library-modal" class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 hidden">
    <div class="auth-card rounded-xl p-6 max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div class="flex items-center justify-between mb-6">
            <h3 class="text-lg font-semibold text-[#E6EDF3]">Add Tools to Assistant</h3>
            <button onclick="closeToolLibrary()" class="p-2 text-[#7D8590] hover:text-white rounded">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>
        
        <!-- Search and Filter -->
        <div class="flex items-center space-x-4 mb-6">
            <div class="flex-1">
                <input type="text" id="tool-search" placeholder="Search tools..." class="w-full px-4 py-2 rounded-lg bg-[#0D1117] border border-[#30363D] text-[#E6EDF3] focus:ring-2 focus:ring-[#10B981]">
            </div>
            <select id="tool-type-filter" class="px-4 py-2 rounded-lg bg-[#0D1117] border border-[#30363D] text-[#E6EDF3]">
                <option value="">All Types</option>
                <option value="endpoint">API Endpoint</option>
                <option value="python_function">Python Function</option>
                <option value="lambda">AWS Lambda</option>
            </select>
        </div>
        
        <!-- Available Tools Grid -->
        <div id="available-tools-grid" class="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
            <!-- Tool cards with "Add" buttons -->
        </div>
        
        <div class="flex items-center justify-end space-x-3 mt-6">
            <button onclick="closeToolLibrary()" class="px-4 py-2 text-sm font-medium text-[#7D8590] hover:text-white bg-white/10 hover:bg-white/20 rounded-lg transition-colors">
                Cancel
            </button>
            <button onclick="addSelectedTools()" class="px-4 py-2 text-sm font-medium text-white bg-[#10B981] hover:bg-[#059669] rounded-lg transition-colors">
                Add Selected Tools
            </button>
        </div>
    </div>
</div>
```

### 6. Tool Execution Monitoring Dashboard
```html
<!-- Execution Logs Page -->
<div class="space-y-6">
    <!-- Real-time Execution Monitor -->
    <div class="auth-card p-6 rounded-xl">
        <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-[#E6EDF3]">Live Tool Executions</h3>
            <div class="flex items-center space-x-2">
                <div class="w-2 h-2 rounded-full bg-[#10B981] animate-pulse"></div>
                <span class="text-xs text-[#7D8590]">Live</span>
            </div>
        </div>
        
        <!-- Execution Timeline -->
        <div id="execution-timeline" class="space-y-3 max-h-64 overflow-y-auto">
            <!-- Real-time execution entries -->
        </div>
    </div>
    
    <!-- Execution Statistics -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
        <!-- Success Rate, Average Duration, etc. -->
    </div>
</div>
```

### 7. JavaScript Functionality

#### Tool Management Functions
```javascript
// Tool Library Management
function openToolLibrary() {
    const modal = document.getElementById('tool-library-modal');
    modal.classList.remove('hidden');
    loadAvailableTools();
}

function loadAvailableTools() {
    fetch('/api/organization/tools')
        .then(response => response.json())
        .then(tools => {
            renderToolsGrid(tools);
        });
}

// Real-time Updates
const executionSocket = new WebSocket(`ws://${location.host}/ws/tool-executions`);
executionSocket.onmessage = function(event) {
    const execution = JSON.parse(event.data);
    updateExecutionTimeline(execution);
};

// Tool Testing
async function testTool(toolId) {
    const testParams = prompt('Enter test parameters (JSON):');
    try {
        const params = JSON.parse(testParams || '{}');
        const response = await fetch(`/api/organization/tools/${toolId}/test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ parameters: params })
        });
        const result = await response.json();
        showTestResult(result);
    } catch (error) {
        showNotification('Test failed: ' + error.message, 'error');
    }
}
```

### 8. Mobile Responsiveness

Following your existing responsive patterns:
- Tools grid becomes single column on mobile
- Tool builder steps stack vertically
- Tool assignment cards optimize for touch interfaces
- Modal overlays adjust for mobile viewports

## Conclusion

This implementation plan provides a comprehensive approach to adding tool calling capabilities to our voice assistant system. By building on our existing tool infrastructure and following security best practices, we can create a powerful and flexible tool calling system that enables users to extend their assistants' capabilities while maintaining security and reliability.

The phased approach allows for iterative development and testing, ensuring that each component is thoroughly validated before moving to the next phase. The focus on security, monitoring, and user experience will help ensure successful adoption of the new features.
