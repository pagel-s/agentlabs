"""
AgentLabs - LLM Agent Research Framework

A comprehensive framework for building, testing, and deploying LLM agents
for research applications including data collection, analysis, and automation.
"""

__version__ = "0.1.0"
__author__ = "Sebastian Pagel"

# Core imports
from .config import Config, LLMConfig, DatabaseConfig, RedisConfig, VectorStoreConfig
from .core.agent import Agent, AgentConfig, AgentRole, AgentContext, AgentFactory
from .core.tools import Tool, ToolRegistry, ToolResult, ToolSchema
from .core.memory import Memory, InMemoryMemory, FileMemory, ContextAwareMemory
from .core.providers import LLMProvider, OpenAIProvider, AnthropicProvider, LocalProvider
from .projects import Project, ProjectManager, ResearchTask, TaskPriority, TaskStatus
from .workflows import Workflow, WorkflowEngine, WorkflowStep, WorkflowStepType
from .utils.logging import setup_logging, get_logger

# CLI
from .cli import main

__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    
    # Configuration
    "Config",
    "LLMConfig", 
    "DatabaseConfig",
    "RedisConfig",
    "VectorStoreConfig",
    
    # Core components
    "Agent",
    "AgentConfig",
    "AgentRole",
    "AgentContext", 
    "AgentFactory",
    
    # Tools
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ToolSchema",
    
    # Memory
    "Memory",
    "InMemoryMemory",
    "FileMemory",
    "ContextAwareMemory",
    
    # Providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "LocalProvider",
    
    # Projects
    "Project",
    "ProjectManager",
    "ResearchTask",
    "TaskPriority",
    "TaskStatus",
    
    # Workflows
    "Workflow",
    "WorkflowEngine",
    "WorkflowStep",
    "WorkflowStepType",
    
    # Utilities
    "setup_logging",
    "get_logger",
    
    # CLI
    "main",
]

def main() -> None:
    print("Hello from agentlabs!")
