"""Tests for tools system."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import pandas as pd
import numpy as np

from agentlabs.core.tools import (
    Tool, ToolSchema, ToolResult, ToolRegistry,
    WebSearchTool, DocumentReaderTool, DataAnalyzerTool
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
            default="default_value",
            enum=["value1", "value2"]
        )
        
        assert schema.name == "test_param"
        assert schema.type == "string"
        assert schema.description == "Test parameter"
        assert schema.required is True
        assert schema.default == "default_value"
        assert schema.enum == ["value1", "value2"]


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
    
    def test_tool_validate_args_enum(self):
        """Test argument validation with enum values."""
        schema = [
            ToolSchema("enum_param", "string", "Enum parameter", required=True, 
                      enum=["value1", "value2", "value3"])
        ]
        
        tool = TestToolImpl("test_tool", "Test tool description", schema)
        
        # Valid enum value
        args = {"enum_param": "value1"}
        assert tool.validate_args(args) is True
        
        # Invalid enum value
        args = {"enum_param": "invalid_value"}
        assert tool.validate_args(args) is False
    
    def test_tool_get_schema_dict(self):
        """Test tool schema dictionary conversion."""
        schema = [
            ToolSchema("param1", "string", "First parameter", required=True),
            ToolSchema("param2", "integer", "Second parameter", required=False, default=10)
        ]
        
        tool = TestToolImpl("test_tool", "Test tool description", schema)
        schema_dict = tool.get_schema_dict()
        
        assert schema_dict["name"] == "test_tool"
        assert schema_dict["description"] == "Test tool description"
        assert len(schema_dict["parameters"]) == 2
        assert schema_dict["parameters"][0]["name"] == "param1"
        assert schema_dict["parameters"][1]["name"] == "param2"


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
        assert len(web_search_tool.schema) == 3
    
    @pytest.mark.asyncio
    async def test_web_search_tool_execute_success(self, web_search_tool):
        """Test successful web search execution."""
        args = {
            "query": "test query",
            "num_results": 3,
            "search_engine": "google"
        }
        
        result = await web_search_tool.execute(args)
        
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) == 3
        assert "query" in result.metadata
        assert "num_results" in result.metadata
        assert "search_engine" in result.metadata
    
    @pytest.mark.asyncio
    async def test_web_search_tool_execute_invalid_args(self, web_search_tool):
        """Test web search with invalid arguments."""
        args = {"invalid_param": "value"}
        
        result = await web_search_tool.execute(args)
        
        assert result.success is False
        assert "Invalid arguments" in result.error
    
    @pytest.mark.asyncio
    async def test_web_search_tool_simulate_search(self, web_search_tool):
        """Test search simulation."""
        query = "test query"
        num_results = 2
        search_engine = "bing"
        
        results = await web_search_tool._simulate_search(query, num_results, search_engine)
        
        assert len(results) == 2
        assert all("title" in result for result in results)
        assert all("url" in result for result in results)
        assert all("snippet" in result for result in results)
        assert all(query in result["title"] for result in results)


class TestDocumentReaderTool:
    """Test document reader tool."""
    
    @pytest.fixture
    def document_reader_tool(self):
        """Create document reader tool instance."""
        return DocumentReaderTool()
    
    def test_document_reader_tool_creation(self, document_reader_tool):
        """Test document reader tool creation."""
        assert document_reader_tool.name == "document_reader"
        assert "Read and extract" in document_reader_tool.description
        assert len(document_reader_tool.schema) == 4
    
    @pytest.mark.asyncio
    async def test_document_reader_tool_execute_success(self, document_reader_tool):
        """Test successful document reading."""
        args = {
            "url": "https://example.com",
            "extract_text": True,
            "extract_links": False,
            "max_length": 1000
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="<html><title>Test</title><body>Test content</body></html>")
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await document_reader_tool.execute(args)
            
            assert result.success is True
            assert "url" in result.data
            assert "title" in result.data
            assert "text" in result.data
            assert "links" in result.data
    
    @pytest.mark.asyncio
    async def test_document_reader_tool_execute_http_error(self, document_reader_tool):
        """Test document reading with HTTP error."""
        args = {"url": "https://example.com"}
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.reason = "Not Found"
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await document_reader_tool.execute(args)
            
            assert result.success is False
            assert "HTTP 404" in result.error
    
    @pytest.mark.asyncio
    async def test_document_reader_tool_read_document(self, document_reader_tool):
        """Test document reading functionality."""
        url = "https://example.com"
        html_content = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Test Content</h1>
                <p>This is test content.</p>
                <a href="https://link1.com">Link 1</a>
                <a href="https://link2.com">Link 2</a>
            </body>
        </html>
        """
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=html_content)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await document_reader_tool._read_document(url, True, True, 1000)
            
            assert result["url"] == url
            assert result["title"] == "Test Page"
            assert "Test Content" in result["text"]
            assert len(result["links"]) == 2
            assert result["links"][0]["url"] == "https://link1.com"


class TestDataAnalyzerTool:
    """Test data analyzer tool."""
    
    @pytest.fixture
    def data_analyzer_tool(self):
        """Create data analyzer tool instance."""
        return DataAnalyzerTool()
    
    def test_data_analyzer_tool_creation(self, data_analyzer_tool):
        """Test data analyzer tool creation."""
        assert data_analyzer_tool.name == "data_analyzer"
        assert "Analyze data" in data_analyzer_tool.description
        assert len(data_analyzer_tool.schema) == 4
    
    @pytest.mark.asyncio
    async def test_data_analyzer_tool_execute_success(self, data_analyzer_tool):
        """Test successful data analysis."""
        csv_data = "name,age,city\nJohn,25,NYC\nJane,30,LA\nBob,35,Chicago"
        args = {
            "data": csv_data,
            "analysis_type": "summary",
            "columns": "",
            "output_format": "text"
        }
        
        result = await data_analyzer_tool.execute(args)
        
        assert result.success is True
        assert "analysis_type" in result.data
        assert "data_shape" in result.data
        assert "summary" in result.data
    
    @pytest.mark.asyncio
    async def test_data_analyzer_tool_execute_invalid_args(self, data_analyzer_tool):
        """Test data analysis with invalid arguments."""
        args = {"invalid_param": "value"}
        
        result = await data_analyzer_tool.execute(args)
        
        assert result.success is False
        assert "Invalid arguments" in result.error
    
    @pytest.mark.asyncio
    async def test_data_analyzer_tool_parse_data_csv(self, data_analyzer_tool):
        """Test CSV data parsing."""
        csv_data = "name,age,city\nJohn,25,NYC\nJane,30,LA"
        
        df = await data_analyzer_tool._parse_data(csv_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["name", "age", "city"]
    
    @pytest.mark.asyncio
    async def test_data_analyzer_tool_parse_data_json(self, data_analyzer_tool):
        """Test JSON data parsing."""
        json_data = '[{"name": "John", "age": 25}, {"name": "Jane", "age": 30}]'
        
        df = await data_analyzer_tool._parse_data(json_data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["name", "age"]
    
    @pytest.mark.asyncio
    async def test_data_analyzer_tool_analyze_data_summary(self, data_analyzer_tool):
        """Test summary analysis."""
        df = pd.DataFrame({
            "numeric": [1, 2, 3, 4, 5],
            "categorical": ["A", "B", "A", "B", "A"],
            "missing": [1, 2, None, 4, 5]
        })
        
        result = await data_analyzer_tool._analyze_data(df, "summary", "", "text")
        
        assert result["analysis_type"] == "summary"
        assert "numeric_columns" in result["summary"]
        assert "categorical_columns" in result["summary"]
        assert "missing_values" in result["summary"]
        assert "basic_stats" in result["summary"]
    
    @pytest.mark.asyncio
    async def test_data_analyzer_tool_analyze_data_correlation(self, data_analyzer_tool):
        """Test correlation analysis."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10]
        })
        
        result = await data_analyzer_tool._analyze_data(df, "correlation", "", "text")
        
        assert result["analysis_type"] == "correlation"
        assert "correlation_matrix" in result
    
    @pytest.mark.asyncio
    async def test_data_analyzer_tool_analyze_data_trend(self, data_analyzer_tool):
        """Test trend analysis."""
        df = pd.DataFrame({
            "values": [1, 2, 3, 4, 5]
        })
        
        result = await data_analyzer_tool._analyze_data(df, "trend", "", "text")
        
        assert result["analysis_type"] == "trend"
        assert "trends" in result
        assert "values" in result["trends"]
    
    @pytest.mark.asyncio
    async def test_data_analyzer_tool_analyze_data_outliers(self, data_analyzer_tool):
        """Test outlier analysis."""
        df = pd.DataFrame({
            "values": [1, 2, 3, 4, 100]  # 100 is an outlier
        })
        
        result = await data_analyzer_tool._analyze_data(df, "outliers", "", "text")
        
        assert result["analysis_type"] == "outliers"
        assert "outliers" in result
        assert "values" in result["outliers"]
    
    @pytest.mark.asyncio
    async def test_data_analyzer_tool_analyze_data_distribution(self, data_analyzer_tool):
        """Test distribution analysis."""
        df = pd.DataFrame({
            "values": [1, 2, 2, 3, 3, 3, 4, 4, 5]
        })
        
        result = await data_analyzer_tool._analyze_data(df, "distribution", "", "text")
        
        assert result["analysis_type"] == "distribution"
        assert "distributions" in result
        assert "values" in result["distributions"]


class TestToolRegistry:
    """Test tool registry."""
    
    @pytest.fixture
    def tool_registry(self):
        """Create tool registry instance."""
        return ToolRegistry()
    
    def test_tool_registry_creation(self, tool_registry):
        """Test tool registry creation."""
        assert len(tool_registry.tools) > 0
        assert "web_search" in tool_registry.tools
        assert "document_reader" in tool_registry.tools
        assert "data_analyzer" in tool_registry.tools
    
    def test_tool_registry_register_tool(self, tool_registry):
        """Test tool registration."""
        test_tool = TestToolImpl("test_tool", "Test tool")
        
        tool_registry.register_tool(test_tool)
        
        assert "test_tool" in tool_registry.tools
        assert tool_registry.get_tool("test_tool") == test_tool
    
    def test_tool_registry_get_tool(self, tool_registry):
        """Test tool retrieval."""
        tool = tool_registry.get_tool("web_search")
        
        assert tool is not None
        assert tool.name == "web_search"
    
    def test_tool_registry_get_nonexistent_tool(self, tool_registry):
        """Test retrieval of non-existent tool."""
        tool = tool_registry.get_tool("nonexistent_tool")
        
        assert tool is None
    
    def test_tool_registry_list_tools(self, tool_registry):
        """Test tool listing."""
        tools = tool_registry.list_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert all("name" in tool for tool in tools)
        assert all("description" in tool for tool in tools)
        assert all("parameters" in tool for tool in tools)
    
    def test_tool_registry_remove_tool(self, tool_registry):
        """Test tool removal."""
        # Register a test tool
        test_tool = TestToolImpl("test_tool", "Test tool")
        tool_registry.register_tool(test_tool)
        
        # Remove the tool
        result = tool_registry.remove_tool("test_tool")
        
        assert result is True
        assert "test_tool" not in tool_registry.tools
    
    def test_tool_registry_remove_nonexistent_tool(self, tool_registry):
        """Test removal of non-existent tool."""
        result = tool_registry.remove_tool("nonexistent_tool")
        
        assert result is False 