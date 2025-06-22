"""
Tools system for AgentLabs framework.
"""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.graph_objects import Figure
import plotly.express as px

from ..utils.logging import LoggedClass, log_async_function_call, log_async_execution_time


@dataclass
class ToolSchema:
    """Schema definition for tool parameters."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Tool(ABC, LoggedClass):
    """Abstract base class for tools."""
    
    def __init__(self, name: str, description: str, schema: List[ToolSchema]):
        super().__init__()
        self.name = name
        self.description = description
        self.schema = schema
        self.created_at = datetime.utcnow()
    
    @abstractmethod
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given arguments."""
        pass
    
    def validate_args(self, args: Dict[str, Any]) -> bool:
        """Validate tool arguments against schema."""
        for param in self.schema:
            if param.required and param.name not in args:
                self.logger.error(f"Required parameter '{param.name}' missing")
                return False
            
            if param.name in args:
                value = args[param.name]
                if not self._validate_type(value, param.type):
                    self.logger.error(f"Parameter '{param.name}' has invalid type. Expected {param.type}")
                    return False
        
        return True
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "float": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected = type_mapping.get(expected_type.lower())
        if expected is None:
            return True  # Unknown type, skip validation
        
        return isinstance(value, expected)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema as dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": param.name,
                    "type": param.type,
                    "description": param.description,
                    "required": param.required,
                    "default": param.default
                }
                for param in self.schema
            ]
        }


class WebSearchTool(Tool):
    """Tool for web search functionality."""
    
    def __init__(self, api_key: Optional[str] = None):
        schema = [
            ToolSchema("query", "string", "Search query", required=True),
            ToolSchema("num_results", "integer", "Number of results to return", required=False, default=5)
        ]
        super().__init__("web_search", "Search the web for information", schema)
        self.api_key = api_key
    
    @log_async_function_call
    @log_async_execution_time
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute web search."""
        if not self.validate_args(args):
            return ToolResult(False, None, "Invalid arguments")
        
        query = args["query"]
        num_results = args.get("num_results", 5)
        
        try:
            # Use DuckDuckGo for search (no API key required)
            search_url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        results = []
                        # Extract relevant information
                        if data.get("Abstract"):
                            results.append({
                                "title": "Abstract",
                                "snippet": data["Abstract"],
                                "url": data.get("AbstractURL", "")
                            })
                        
                        if data.get("RelatedTopics"):
                            for topic in data["RelatedTopics"][:num_results]:
                                if isinstance(topic, dict) and "Text" in topic:
                                    results.append({
                                        "title": topic.get("Text", "").split(" - ")[0],
                                        "snippet": topic["Text"],
                                        "url": topic.get("FirstURL", "")
                                    })
                        
                        return ToolResult(
                            success=True,
                            data=results,
                            metadata={"query": query, "num_results": len(results)}
                        )
                    else:
                        return ToolResult(False, None, f"Search failed with status {response.status}")
        
        except Exception as e:
            self.logger.error(f"Web search error: {str(e)}")
            return ToolResult(False, None, str(e))


class WebScrapingTool(Tool):
    """Tool for web scraping functionality."""
    
    def __init__(self, headless: bool = True):
        schema = [
            ToolSchema("url", "string", "URL to scrape", required=True),
            ToolSchema("selector", "string", "CSS selector for content extraction", required=False),
            ToolSchema("timeout", "integer", "Timeout in seconds", required=False, default=30)
        ]
        super().__init__("web_scraping", "Extract content from web pages", schema)
        self.headless = headless
    
    @log_async_function_call
    @log_async_execution_time
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute web scraping."""
        if not self.validate_args(args):
            return ToolResult(False, None, "Invalid arguments")
        
        url = args["url"]
        selector = args.get("selector")
        timeout = args.get("timeout", 30)
        
        try:
            # Use Selenium for JavaScript-heavy pages
            options = Options()
            if self.headless:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            driver = webdriver.Chrome(
                service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
                options=options
            )
            
            try:
                driver.set_page_load_timeout(timeout)
                driver.get(url)
                
                # Wait for page to load
                WebDriverWait(driver, timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                if selector:
                    # Extract content using CSS selector
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    content = "\n".join([elem.text for elem in elements if elem.text])
                else:
                    # Extract main content
                    content = driver.find_element(By.TAG_NAME, "body").text
                
                # Clean up content
                content = self._clean_content(content)
                
                return ToolResult(
                    success=True,
                    data={
                        "url": url,
                        "content": content,
                        "title": driver.title,
                        "length": len(content)
                    },
                    metadata={"selector": selector, "timeout": timeout}
                )
            
            finally:
                driver.quit()
        
        except Exception as e:
            self.logger.error(f"Web scraping error: {str(e)}")
            return ToolResult(False, None, str(e))
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content."""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        # Remove empty lines
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        return '\n'.join(lines)


class DataAnalysisTool(Tool):
    """Tool for data analysis functionality."""
    
    def __init__(self):
        schema = [
            ToolSchema("data", "string", "Data to analyze (CSV, JSON, or text)", required=True),
            ToolSchema("analysis_type", "string", "Type of analysis (summary, correlation, visualization)", required=True),
            ToolSchema("columns", "array", "Columns to analyze", required=False)
        ]
        super().__init__("data_analysis", "Analyze structured and unstructured data", schema)
    
    @log_async_function_call
    @log_async_execution_time
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute data analysis."""
        if not self.validate_args(args):
            return ToolResult(False, None, "Invalid arguments")
        
        data_input = args["data"]
        analysis_type = args["analysis_type"]
        columns = args.get("columns")
        
        try:
            # Parse data
            df = self._parse_data(data_input)
            if df is None:
                return ToolResult(False, None, "Failed to parse data")
            
            # Perform analysis
            if analysis_type == "summary":
                result = self._generate_summary(df, columns)
            elif analysis_type == "correlation":
                result = self._analyze_correlation(df, columns)
            elif analysis_type == "visualization":
                result = await self._create_visualization(df, columns)
            else:
                return ToolResult(False, None, f"Unknown analysis type: {analysis_type}")
            
            return ToolResult(
                success=True,
                data=result,
                metadata={"analysis_type": analysis_type, "data_shape": df.shape}
            )
        
        except Exception as e:
            self.logger.error(f"Data analysis error: {str(e)}")
            return ToolResult(False, None, str(e))
    
    def _parse_data(self, data_input: str) -> Optional[pd.DataFrame]:
        """Parse data input into DataFrame."""
        try:
            # Try parsing as CSV
            if data_input.strip().startswith(('name,', 'id,', 'date,')) or '\n' in data_input:
                return pd.read_csv(pd.StringIO(data_input))
            
            # Try parsing as JSON
            try:
                data = json.loads(data_input)
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    return pd.DataFrame([data])
            except json.JSONDecodeError:
                pass
            
            # Try parsing as space/tab separated
            return pd.read_csv(pd.StringIO(data_input), sep=None, engine='python')
        
        except Exception as e:
            self.logger.error(f"Data parsing error: {str(e)}")
            return None
    
    def _generate_summary(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate data summary."""
        if columns:
            df = df[columns]
        
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_summary": {}
        }
        
        # Numeric summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        return summary
    
    def _analyze_correlation(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze correlations between numeric columns."""
        if columns:
            df = df[columns]
        
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {"error": "No numeric columns found"}
        
        correlation_matrix = numeric_df.corr()
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "high_correlations": self._find_high_correlations(correlation_matrix)
        }
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find high correlations in the matrix."""
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr.append({
                        "column1": corr_matrix.columns[i],
                        "column2": corr_matrix.columns[j],
                        "correlation": corr_value
                    })
        return high_corr
    
    async def _create_visualization(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create data visualizations."""
        if columns:
            df = df[columns]
        
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {"error": "No numeric columns found for visualization"}
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title("Correlation Heatmap")
        
        # Save plot
        plot_path = f"correlation_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path)
        plt.close()
        
        return {
            "plot_path": plot_path,
            "plot_type": "correlation_heatmap",
            "columns_visualized": list(numeric_df.columns)
        }


class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.logger = LoggedClass().logger
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self.tools.keys())
    
    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool schema by name."""
        tool = self.get_tool(name)
        return tool.get_schema() if tool else None
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from the registry."""
        if name in self.tools:
            del self.tools[name]
            self.logger.info(f"Removed tool: {name}")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all tools from the registry."""
        self.tools.clear()
        self.logger.info("Cleared all tools from registry")


# Default tool registry with built-in tools
def create_default_tool_registry() -> ToolRegistry:
    """Create default tool registry with built-in tools."""
    registry = ToolRegistry()
    
    # Register built-in tools
    registry.register_tool(WebSearchTool())
    registry.register_tool(WebScrapingTool())
    registry.register_tool(DataAnalysisTool())
    
    return registry 