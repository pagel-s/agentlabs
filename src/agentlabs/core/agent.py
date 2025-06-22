"""Core agent system for AgentLabs framework."""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path

from pydantic import BaseModel, Field
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.config import LLMConfig
from ..utils.logging import LoggerMixin, LoggedClass, log_async_function_call, log_async_execution_time
from .llm import LLMProvider, LLMFactory, LLMResponse, LLMMessage
from .memory import Memory, MemoryStore
from .tools import Tool, ToolRegistry, ToolResult


class AgentState(str, Enum):
    """Agent execution states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class AgentRole(str, Enum):
    """Agent roles for different research tasks."""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    VALIDATOR = "validator"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    GENERAL = "general"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    
    name: str
    role: AgentRole
    description: str
    system_prompt: str
    llm_config: LLMConfig
    tools: List[str] = field(default_factory=list)
    memory_config: Optional[Dict[str, Any]] = None
    max_iterations: int = 10
    timeout: int = 300
    temperature: float = 0.7
    max_tokens: int = 4096
    verbose: bool = False
    memory_enabled: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.name.strip():
            raise ValueError("Agent name cannot be empty")
        if not self.system_prompt.strip():
            raise ValueError("System prompt cannot be empty")
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")


@dataclass
class AgentContext:
    """Context for agent execution."""
    
    project_id: str
    task_id: str
    session_id: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentResult:
    """Result of agent execution."""
    
    success: bool
    output: str
    iterations: int
    duration: float
    metadata: Dict[str, Any]
    error: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExecutionState:
    """State of agent execution."""
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[LLMMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent(LoggedClass):
    """Base agent class for research tasks."""
    
    def __init__(self, config: AgentConfig, context: Optional[AgentContext] = None) -> None:
        """Initialize agent."""
        super().__init__()
        self.config = config
        self.context = context or AgentContext(
            project_id=str(uuid.uuid4()),
            task_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4())
        )
        
        # Initialize components
        self.llm_provider = LLMFactory.create(config.llm_config)
        self.memory = MemoryStore.create(config.memory_config or {})
        self.tool_registry = ToolRegistry()
        self.state = AgentState.IDLE
        
        # Load tools
        self._load_tools()
        
        # Execution state
        self.current_iteration = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.messages: List[BaseMessage] = []
        self.tool_calls: List[Dict[str, Any]] = []
        
        self.logger.info(f"Initialized agent '{config.name}' with role '{config.role.value}'")
    
    def _load_tools(self) -> None:
        """Load tools for the agent."""
        for tool_name in self.config.tools:
            try:
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    self.logger.debug(f"Loaded tool: {tool_name}")
                else:
                    self.logger.warning(f"Tool not found: {tool_name}")
            except Exception as e:
                self.logger.error(f"Error loading tool {tool_name}: {e}")
    
    @log_async_function_call
    @log_async_execution_time
    async def execute(self, input_data: Union[str, Dict[str, Any]], **kwargs: Any) -> AgentResult:
        """Execute the agent with given input."""
        self.start_time = datetime.utcnow()
        self.state = AgentState.RUNNING
        self.current_iteration = 0
        
        try:
            self.logger.info(f"Starting agent execution with input: {input_data}")
            
            # Prepare initial messages
            self.messages = [
                SystemMessage(content=self.config.system_prompt),
                HumanMessage(content=str(input_data))
            ]
            
            # Main execution loop
            while self.current_iteration < self.config.max_iterations:
                self.current_iteration += 1
                self.logger.debug(f"Starting iteration {self.current_iteration}")
                
                # Check timeout
                if self._is_timeout():
                    raise TimeoutError(f"Agent execution timed out after {self.config.timeout} seconds")
                
                # Generate response
                response = await self._generate_response(**kwargs)
                
                # Process response
                should_continue = await self._process_response(response)
                
                if not should_continue:
                    break
            
            # Prepare result
            result = AgentResult(
                success=True,
                output=self._get_final_output(),
                iterations=self.current_iteration,
                duration=(datetime.utcnow() - self.start_time).total_seconds(),
                metadata=self._get_execution_metadata(),
                tool_calls=self.tool_calls
            )
            
            self.state = AgentState.COMPLETED
            self.logger.info(f"Agent execution completed successfully in {result.duration:.2f}s")
            
            return result
        
        except Exception as e:
            self.state = AgentState.FAILED
            self.logger.error(f"Agent execution failed: {e}")
            
            return AgentResult(
                success=False,
                output="",
                iterations=self.current_iteration,
                duration=(datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
                metadata=self._get_execution_metadata(),
                error=str(e),
                tool_calls=self.tool_calls
            )
        
        finally:
            self.end_time = datetime.utcnow()
    
    async def _generate_response(self, **kwargs: Any) -> LLMResponse:
        """Generate response from LLM."""
        try:
            # Add memory context if available
            memory_context = await self.memory.get_context(self.context.session_id)
            if memory_context:
                self.messages.insert(1, SystemMessage(content=f"Memory context: {memory_context}"))
            
            # Add available tools information
            if self.tool_registry.tools:
                tools_info = self._format_tools_info()
                self.messages.insert(1, SystemMessage(content=f"Available tools: {tools_info}"))
            
            response = await self.llm_provider.generate(
                self.messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]}
            )
            
            # Add response to messages
            self.messages.append(AIMessage(content=response.content))
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise
    
    async def _process_response(self, response: LLMResponse) -> bool:
        """Process LLM response and determine if execution should continue."""
        try:
            # Check for tool calls
            tool_calls = self._extract_tool_calls(response.content)
            
            if tool_calls:
                for tool_call in tool_calls:
                    result = await self._execute_tool_call(tool_call)
                    self.tool_calls.append({
                        "tool": tool_call["tool"],
                        "args": tool_call["args"],
                        "result": result,
                        "iteration": self.current_iteration
                    })
                    
                    # Add tool result to messages
                    self.messages.append(HumanMessage(content=f"Tool result: {result}"))
                
                return True  # Continue execution after tool calls
            
            # Check for completion indicators
            completion_indicators = [
                "task completed",
                "research complete",
                "analysis finished",
                "final answer",
                "conclusion"
            ]
            
            if any(indicator in response.content.lower() for indicator in completion_indicators):
                return False  # Stop execution
            
            return True  # Continue execution
        
        except Exception as e:
            self.logger.error(f"Error processing response: {e}")
            return False
    
    def _extract_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """Extract tool calls from response content."""
        tool_calls = []
        
        # Simple JSON extraction (can be enhanced with more sophisticated parsing)
        try:
            # Look for JSON blocks that might contain tool calls
            import re
            json_pattern = r'\{[^{}]*"tool"[^{}]*\}'
            matches = re.findall(json_pattern, content)
            
            for match in matches:
                try:
                    tool_call = json.loads(match)
                    if "tool" in tool_call and "args" in tool_call:
                        tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            self.logger.debug(f"Error extracting tool calls: {e}")
        
        return tool_calls
    
    async def _execute_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Execute a tool call."""
        try:
            tool_name = tool_call["tool"]
            args = tool_call.get("args", {})
            
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return f"Error: Tool '{tool_name}' not found"
            
            self.log_debug(f"Executing tool '{tool_name}' with args: {args}")
            result = await tool.execute(args)
            
            return str(result)
        
        except Exception as e:
            self.log_error(f"Error executing tool call: {e}")
            return f"Error executing tool: {str(e)}"
    
    def _format_tools_info(self) -> str:
        """Format available tools information for the LLM."""
        tools_info = []
        for tool in self.tool_registry.tools.values():
            tools_info.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tools_info)
    
    def _is_timeout(self) -> bool:
        """Check if execution has timed out."""
        if not self.start_time:
            return False
        return (datetime.utcnow() - self.start_time).total_seconds() > self.config.timeout
    
    def _get_final_output(self) -> str:
        """Get the final output from the agent."""
        if not self.messages:
            return ""
        
        # Get the last AI message
        for message in reversed(self.messages):
            if isinstance(message, AIMessage):
                return message.content
        
        return ""
    
    def _get_execution_metadata(self) -> Dict[str, Any]:
        """Get execution metadata."""
        return {
            "agent_name": self.config.name,
            "agent_role": self.config.role.value,
            "session_id": self.context.session_id,
            "project_id": self.context.project_id,
            "task_id": self.context.task_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "state": self.state.value,
            "messages_count": len(self.messages),
            "tool_calls_count": len(self.tool_calls)
        }
    
    async def stop(self) -> None:
        """Stop agent execution."""
        self.state = AgentState.STOPPED
        self.log_info("Agent execution stopped by user")
    
    async def pause(self) -> None:
        """Pause agent execution."""
        self.state = AgentState.PAUSED
        self.log_info("Agent execution paused")
    
    async def resume(self) -> None:
        """Resume agent execution."""
        self.state = AgentState.RUNNING
        self.log_info("Agent execution resumed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "state": self.state.value,
            "iteration": self.current_iteration,
            "max_iterations": self.config.max_iterations,
            "messages_count": len(self.messages),
            "tool_calls_count": len(self.tool_calls),
            "duration": (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        }


class AgentFactory:
    """Factory for creating agents."""
    
    _agent_templates = {
        AgentRole.RESEARCHER: {
            "system_prompt": """You are a research agent specialized in gathering and analyzing information. 
            Your goal is to conduct thorough research on given topics and provide comprehensive findings.
            Always cite your sources and provide evidence for your conclusions.""",
            "tools": ["web_search", "document_reader", "data_analyzer"],
            "max_iterations": 15,
            "temperature": 0.3
        },
        AgentRole.ANALYST: {
            "system_prompt": """You are an analytical agent focused on data analysis and pattern recognition.
            Your role is to analyze data, identify trends, and provide insights based on evidence.
            Always provide quantitative analysis when possible.""",
            "tools": ["data_analyzer", "statistical_tools", "visualization_tools"],
            "max_iterations": 10,
            "temperature": 0.2
        },
        AgentRole.WRITER: {
            "system_prompt": """You are a writing agent specialized in creating clear, well-structured content.
            Your role is to write reports, summaries, and documentation based on research findings.
            Always maintain a professional tone and ensure clarity.""",
            "tools": ["text_processor", "document_writer", "citation_manager"],
            "max_iterations": 8,
            "temperature": 0.7
        },
        AgentRole.VALIDATOR: {
            "system_prompt": """You are a validation agent responsible for verifying information and checking quality.
            Your role is to validate findings, check for accuracy, and ensure quality standards are met.
            Always be thorough and objective in your validation.""",
            "tools": ["fact_checker", "quality_checker", "consistency_checker"],
            "max_iterations": 5,
            "temperature": 0.1
        }
    }
    
    @classmethod
    def create_agent(
        cls,
        name: str,
        role: AgentRole,
        description: str,
        llm_config: LLMConfig,
        **kwargs: Any
    ) -> Agent:
        """Create an agent with the specified role."""
        template = cls._agent_templates.get(role, {})
        
        config = AgentConfig(
            name=name,
            role=role,
            description=description,
            system_prompt=kwargs.get("system_prompt", template.get("system_prompt", "")),
            llm_config=llm_config,
            tools=kwargs.get("tools", template.get("tools", [])),
            max_iterations=kwargs.get("max_iterations", template.get("max_iterations", 10)),
            temperature=kwargs.get("temperature", template.get("temperature", 0.7)),
            **{k: v for k, v in kwargs.items() if k not in ["system_prompt", "tools", "max_iterations", "temperature"]}
        )
        
        return Agent(config)
    
    @classmethod
    def register_template(cls, role: AgentRole, template: Dict[str, Any]) -> None:
        """Register a new agent template."""
        cls._agent_templates[role] = template 