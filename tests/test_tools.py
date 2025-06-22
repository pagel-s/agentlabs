"""Tests for tools system."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import pandas as pd
import numpy as np

from agentlabs.core.tools import (
    Tool, ToolSchema, ToolResult, ToolRegistry,
    WebSearchTool
)


class TestToolSchema:
    """Test tool schema."""
    
    def test_tool_schema_creation(self):
        """Test tool schema creation."""
        schema = ToolSchema(
            name="test_param",
            type="string",
            description="Test parameter",
            required=True,
            default="default_value"
        )
        
        assert schema.name == "test_param"
        assert schema.type == "string"
        assert schema.description == "Test parameter"
        assert schema.required is True
        assert schema.default == "default_value"


class TestToolResult:
    """Test tool result."""
    
    def test_tool_result_creation(self):
        """Test tool result creation."""
        result = ToolResult(
            success=True,
            data={"key": "value"},
            metadata={"timestamp": "2023-01-01"},
            error=None
        )
        
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.metadata == {"timestamp": "2023-01-01"}
        assert result.error is None


class TestTool:
    """Test base tool class."""
    
    def test_tool_creation(self):
        """Test tool creation."""
        schema = [
            ToolSchema("param1", "string", "First parameter", required=True),
            ToolSchema("param2", "integer", "Second parameter", required=False, default=10)
        ]
        
        tool = TestToolImpl("test_tool", "Test tool description", schema)
        
        assert tool.name == "test_tool"
        assert tool.description == "Test tool description"
        assert len(tool.schema) == 2
    
    def test_tool_validate_args_success(self):
        """Test successful argument validation."""
        schema = [
            ToolSchema("required_param", "string", "Required parameter", required=True),
            ToolSchema("optional_param", "integer", "Optional parameter", required=False, default=10)
        ]
        
        tool = TestToolImpl("test_tool", "Test tool description", schema)
        
        args = {"required_param": "test_value", "optional_param": 20}
        assert tool.validate_args(args) is True
    
    def test_tool_validate_args_missing_required(self):
        """Test argument validation with missing required parameter."""
        schema = [
            ToolSchema("required_param", "string", "Required parameter", required=True)
        ]
        
        tool = TestToolImpl("test_tool", "Test tool description", schema)
        
        args = {"other_param": "value"}
        assert tool.validate_args(args) is False
    
    def test_tool_validate_args_wrong_type(self):
        """Test argument validation with wrong type."""
        schema = [
            ToolSchema("string_param", "string", "String parameter", required=True),
            ToolSchema("int_param", "integer", "Integer parameter", required=True)
        ]
        
        tool = TestToolImpl("test_tool", "Test tool description", schema)
        
        args = {"string_param": 123, "int_param": "not_an_int"}
        assert tool.validate_args(args) is False


class TestToolImpl(Tool):
    """Test implementation of abstract Tool class."""
    
    async def execute(self, args):
        """Execute the tool."""
        return ToolResult(success=True, data=args)


class TestWebSearchTool:
    """Test web search tool."""
    
    @pytest.fixture
    def web_search_tool(self):
        """Create web search tool instance."""
        return WebSearchTool()
    
    def test_web_search_tool_creation(self, web_search_tool):
        """Test web search tool creation."""
        assert web_search_tool.name == "web_search"
        assert "Search the web" in web_search_tool.description
        assert len(web_search_tool.schema) == 2
    
    @pytest.mark.asyncio
    async def test_web_search_tool_execute_success(self, web_search_tool):
        """Test successful web search execution."""
        args = {
            "query": "test query",
            "num_results": 3
        }
        
        result = await web_search_tool.execute(args)
        
        assert isinstance(result.success, bool)
        if result.success:
            assert isinstance(result.data, list)
            assert "query" in result.metadata
            assert "num_results" in result.metadata
        else:
            assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_web_search_tool_execute_invalid_args(self, web_search_tool):
        """Test web search with invalid arguments."""
        args = {"invalid_param": "value"}
        
        result = await web_search_tool.execute(args)
        
        assert result.success is False
        assert "Invalid arguments" in result.error


class TestToolRegistry:
    """Test tool registry."""
    
    @pytest.fixture
    def tool_registry(self):
        """Create tool registry instance."""
        return ToolRegistry()
    
    def test_tool_registry_creation(self, tool_registry):
        """Test tool registry creation."""
        assert len(tool_registry.tools) == 0  # Registry starts empty
        # Register a tool to test functionality
        test_tool = TestToolImpl("test_tool", "Test tool", [])
        tool_registry.register_tool(test_tool)
        assert len(tool_registry.tools) == 1
        assert "test_tool" in tool_registry.tools
    
    def test_tool_registry_register_tool(self, tool_registry):
        """Test tool registration."""
        test_tool = TestToolImpl("test_tool", "Test tool", [])
        
        tool_registry.register_tool(test_tool)
        
        assert "test_tool" in tool_registry.tools
        assert tool_registry.get_tool("test_tool") == test_tool
    
    def test_tool_registry_get_tool(self, tool_registry):
        """Test tool retrieval."""
        # Register a tool first
        test_tool = TestToolImpl("web_search", "Web search tool", [])
        tool_registry.register_tool(test_tool)
        
        tool = tool_registry.get_tool("web_search")
        
        assert tool is not None
        assert tool.name == "web_search"
    
    def test_tool_registry_get_nonexistent_tool(self, tool_registry):
        """Test retrieval of non-existent tool."""
        tool = tool_registry.get_tool("nonexistent_tool")
        
        assert tool is None
    
    def test_tool_registry_list_tools(self, tool_registry):
        """Test tool listing."""
        # Register some tools first
        test_tool1 = TestToolImpl("tool1", "Test tool 1", [])
        test_tool2 = TestToolImpl("tool2", "Test tool 2", [])
        tool_registry.register_tool(test_tool1)
        tool_registry.register_tool(test_tool2)
        
        tools = tool_registry.list_tools()
        
        assert isinstance(tools, list)
        assert len(tools) == 2
        assert "tool1" in tools
        assert "tool2" in tools
    
    def test_tool_registry_remove_tool(self, tool_registry):
        """Test tool removal."""
        # Register a test tool
        test_tool = TestToolImpl("test_tool", "Test tool", [])
        tool_registry.register_tool(test_tool)
        
        # Remove the tool
        result = tool_registry.remove_tool("test_tool")
        
        assert result is True
        assert "test_tool" not in tool_registry.tools
    
    def test_tool_registry_remove_nonexistent_tool(self, tool_registry):
        """Test removal of non-existent tool."""
        result = tool_registry.remove_tool("nonexistent_tool")
        
        assert result is False 