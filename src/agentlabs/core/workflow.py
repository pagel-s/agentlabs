"""Workflow system for AgentLabs framework."""

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

from ..utils.logging import LoggerMixin, log_async_function_call, log_async_execution_time
from .agent import Agent, AgentConfig, AgentContext, AgentResult, AgentRole, AgentFactory
from .llm import LLMConfig, LLMProvider, LLMFactory
from .research import ResearchProject, ResearchTask, TaskStatus, TaskPriority, ProjectManager
from .tools import ToolRegistry, create_default_tool_registry
from .memory import Memory, create_memory


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStepType(str, Enum):
    """Types of workflow steps."""
    AGENT = "agent"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    WAIT = "wait"
    CUSTOM = "custom"


@dataclass
class WorkflowStep:
    """A step in a workflow."""
    
    id: str
    name: str
    step_type: WorkflowStepType
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Workflow execution context."""
    id: str
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.IDLE
    current_step: Optional[str] = None
    step_results: Dict[str, Any] = field(default_factory=dict)
    step_errors: Dict[str, str] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """A workflow for orchestrating research processes."""
    
    id: str
    name: str
    description: str
    steps: List[WorkflowStep] = field(default_factory=list)
    status: WorkflowStatus = WorkflowStatus.IDLE
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validate workflow after initialization."""
        if not self.name.strip():
            raise ValueError("Workflow name cannot be empty")
        if not self.description.strip():
            raise ValueError("Workflow description cannot be empty")


class StepExecutor(ABC, LoggerMixin):
    """Abstract base class for step executors."""
    
    def __init__(self, llm_provider: LLMProvider, tool_registry: ToolRegistry, memory: Memory):
        """Initialize step executor."""
        super().__init__()
        self.llm_provider = llm_provider
        self.tool_registry = tool_registry
        self.memory = memory
    
    @abstractmethod
    async def execute(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute a workflow step."""
        pass


class AgentStepExecutor(StepExecutor):
    """Executor for agent-based workflow steps."""
    
    async def execute(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute agent step."""
        try:
            # Get agent configuration from step config
            agent_config = step.config.get("agent", {})
            role = AgentRole(agent_config.get("role", AgentRole.GENERAL))
            input_data = agent_config.get("input", "")
            
            # Create agent configuration
            config = AgentConfig(
                name=f"{step.name}_agent",
                role=role,
                description=f"Agent for workflow step: {step.name}",
                tools=[tool.name for tool in self.tool_registry.tools.values()]
            )
            
            # Create agent
            agent = AgentFactory.create_agent(
                config=config,
                llm_provider=self.llm_provider,
                tool_registry=self.tool_registry,
                memory=self.memory
            )
            
            # Create context
            context = AgentContext(
                session_id=f"{execution.id}_{step.id}",
                project_id=execution.workflow_id,
                task_id=step.id
            )
            
            # Execute agent
            result = await agent.execute(input_data, context)
            
            return {
                "success": True,
                "result": result,
                "agent_id": agent.id,
                "execution_time": result.get("execution_time", 0)
            }
        
        except Exception as e:
            self.log_error(f"Agent step execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


class ConditionStepExecutor(StepExecutor):
    """Executor for conditional workflow steps."""
    
    async def execute(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute condition step."""
        try:
            condition_config = step.config.get("condition", {})
            condition_type = condition_config.get("type", "expression")
            
            if condition_type == "expression":
                # Evaluate Python expression
                expression = condition_config.get("expression", "")
                variables = execution.variables.copy()
                
                # Add step results to variables
                for step_id, result in execution.step_results.items():
                    variables[f"step_{step_id}"] = result
                
                # Evaluate expression
                result = eval(expression, {"__builtins__": {}}, variables)
                
                return {
                    "success": True,
                    "result": result,
                    "condition_met": bool(result)
                }
            
            elif condition_type == "llm":
                # Use LLM to evaluate condition
                prompt = condition_config.get("prompt", "")
                context = condition_config.get("context", "")
                
                # Create messages
                messages = [
                    {
                        "role": "system",
                        "content": "You are a condition evaluator. Respond with 'true' or 'false' only."
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\n\nPrompt: {prompt}\n\nRespond with 'true' or 'false':"
                    }
                ]
                
                response = await self.llm_provider.generate(messages)
                result = response.content.lower().strip() == "true"
                
                return {
                    "success": True,
                    "result": result,
                    "condition_met": result,
                    "llm_response": response.content
                }
            
            else:
                raise ValueError(f"Unknown condition type: {condition_type}")
        
        except Exception as e:
            self.log_error(f"Condition step execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


class LoopStepExecutor(StepExecutor):
    """Executor for loop steps."""
    
    async def execute(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute a loop step."""
        try:
            loop_config = step.config.get("loop", {})
            loop_type = loop_config.get("type", "for")
            max_iterations = loop_config.get("max_iterations", 10)
            
            if loop_type == "for":
                # For loop with range
                start = loop_config.get("start", 0)
                end = loop_config.get("end", 10)
                step_size = loop_config.get("step", 1)
                
                results = []
                for i in range(start, end, step_size):
                    if len(results) >= max_iterations:
                        break
                    
                    # Execute loop body
                    body_config = loop_config.get("body", {})
                    body_result = await self._execute_loop_body(body_config, execution, i)
                    results.append(body_result)
                
                return {
                    "success": True,
                    "result": results,
                    "iterations": len(results)
                }
            
            elif loop_type == "while":
                # While loop with condition
                condition = loop_config.get("condition", "")
                results = []
                iteration = 0
                
                while iteration < max_iterations:
                    # Check condition
                    condition_result = await self._evaluate_condition(condition, execution, iteration)
                    if not condition_result:
                        break
                    
                    # Execute loop body
                    body_config = loop_config.get("body", {})
                    body_result = await self._execute_loop_body(body_config, execution, iteration)
                    results.append(body_result)
                    
                    iteration += 1
                
                return {
                    "success": True,
                    "result": results,
                    "iterations": len(results)
                }
            
            else:
                raise ValueError(f"Unknown loop type: {loop_type}")
        
        except Exception as e:
            self.log_error(f"Loop step execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_loop_body(self, body_config: Dict[str, Any], execution: WorkflowExecution, iteration: int) -> Dict[str, Any]:
        """Execute loop body."""
        # This is a simplified implementation
        # In a real system, you might want to execute actual workflow steps
        return {
            "iteration": iteration,
            "timestamp": datetime.utcnow().isoformat(),
            "config": body_config
        }
    
    async def _evaluate_condition(self, condition: str, execution: WorkflowExecution, iteration: int) -> bool:
        """Evaluate loop condition."""
        try:
            variables = execution.variables.copy()
            variables["iteration"] = iteration
            return bool(eval(condition, {"__builtins__": {}}, variables))
        except Exception:
            return False


class ParallelStepExecutor(StepExecutor):
    """Executor for parallel steps."""
    
    async def execute(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute parallel steps."""
        try:
            parallel_config = step.config.get("parallel", {})
            steps_config = parallel_config.get("steps", [])
            max_concurrent = parallel_config.get("max_concurrent", 5)
            
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def execute_step_with_semaphore(step_config: Dict[str, Any]) -> Dict[str, Any]:
                async with semaphore:
                    # This is a simplified implementation
                    # In a real system, you would execute actual workflow steps
                    await asyncio.sleep(0.1)  # Simulate work
                    return {
                        "step_config": step_config,
                        "timestamp": datetime.utcnow().isoformat()
                    }
            
            # Execute steps in parallel
            tasks = [execute_step_with_semaphore(step_config) for step_config in steps_config]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_results = []
            errors = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(f"Step {i}: {str(result)}")
                else:
                    successful_results.append(result)
            
            return {
                "success": len(errors) == 0,
                "results": successful_results,
                "errors": errors,
                "total_steps": len(steps_config),
                "successful_steps": len(successful_results)
            }
        
        except Exception as e:
            self.log_error(f"Parallel step execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


class WaitStepExecutor(StepExecutor):
    """Executor for wait steps."""
    
    async def execute(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute a wait step."""
        try:
            wait_config = step.config.get("wait", {})
            duration = wait_config.get("duration", 1)  # seconds
            
            await asyncio.sleep(duration)
            
            return {
                "success": True,
                "result": f"Waited for {duration} seconds",
                "duration": duration
            }
        
        except Exception as e:
            self.log_error(f"Wait step execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


class WorkflowEngine(LoggerMixin):
    """Engine for executing workflows."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_registry: Optional[ToolRegistry] = None,
        memory: Optional[Memory] = None
    ):
        """Initialize workflow engine."""
        super().__init__()
        self.llm_provider = llm_provider
        self.tool_registry = tool_registry or create_default_tool_registry()
        self.memory = memory or create_memory("in_memory")
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        
        # Initialize step executors
        self.executors = {
            WorkflowStepType.AGENT: AgentStepExecutor(llm_provider, self.tool_registry, self.memory),
            WorkflowStepType.CONDITION: ConditionStepExecutor(llm_provider, self.tool_registry, self.memory),
            WorkflowStepType.LOOP: LoopStepExecutor(llm_provider, self.tool_registry, self.memory),
            WorkflowStepType.PARALLEL: ParallelStepExecutor(llm_provider, self.tool_registry, self.memory),
            WorkflowStepType.WAIT: WaitStepExecutor(llm_provider, self.tool_registry, self.memory)
        }
    
    @log_async_function_call
    def create_workflow(self, name: str, description: str = "") -> Workflow:
        """Create a new workflow."""
        workflow_id = str(uuid.uuid4())
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description
        )
        
        self.workflows[workflow_id] = workflow
        self.log_info(f"Created workflow: {name} (ID: {workflow_id})")
        
        return workflow
    
    @log_async_function_call
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self.workflows.get(workflow_id)
    
    @log_async_function_call
    def list_workflows(self) -> List[Workflow]:
        """List all workflows."""
        return list(self.workflows.values())
    
    @log_async_function_call
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            self.log_info(f"Deleted workflow: {workflow_id}")
            return True
        return False
    
    @log_async_function_call
    def add_step(
        self,
        workflow_id: str,
        name: str,
        step_type: WorkflowStepType,
        config: Dict[str, Any],
        dependencies: Optional[List[str]] = None
    ) -> Optional[WorkflowStep]:
        """Add a step to a workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            self.log_error(f"Workflow not found: {workflow_id}")
            return None
        
        step_id = str(uuid.uuid4())
        step = WorkflowStep(
            id=step_id,
            name=name,
            step_type=step_type,
            config=config,
            dependencies=dependencies or []
        )
        
        workflow.steps.append(step)
        workflow.updated_at = datetime.utcnow()
        
        self.log_info(f"Added step: {name} to workflow {workflow.name}")
        return step
    
    @log_async_function_call
    def get_step(self, workflow_id: str, step_id: str) -> Optional[WorkflowStep]:
        """Get a step by ID."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return None
        
        for step in workflow.steps:
            if step.id == step_id:
                return step
        return None
    
    @log_async_function_call
    @log_async_execution_time
    async def execute_workflow(self, workflow_id: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            return {"success": False, "error": "Workflow not found"}
        
        # Create execution context
        execution_id = str(uuid.uuid4())
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=workflow_id,
            variables=variables or {}
        )
        
        self.executions[execution_id] = execution
        execution.status = WorkflowStatus.RUNNING
        execution.started_at = datetime.utcnow()
        
        self.log_info(f"Starting workflow execution: {workflow.name}")
        
        try:
            # Sort steps by dependencies
            sorted_steps = self._sort_steps(workflow.steps)
            
            # Execute steps
            for step in sorted_steps:
                execution.current_step = step.id
                
                # Check dependencies
                if not self._check_step_dependencies(step, execution):
                    execution.status = WorkflowStatus.FAILED
                    return {
                        "success": False,
                        "error": f"Step dependencies not met: {step.name}"
                    }
                
                # Execute step
                result = await self._execute_step(step, execution)
                
                if result["success"]:
                    execution.step_results[step.id] = result
                else:
                    execution.step_errors[step.id] = result.get("error", "Unknown error")
                    
                    # Check if we should retry
                    if step.retry_count < step.max_retries:
                        step.retry_count += 1
                        self.log_info(f"Retrying step {step.name} (attempt {step.retry_count})")
                        result = await self._execute_step(step, execution)
                        if result["success"]:
                            execution.step_results[step.id] = result
                        else:
                            execution.status = WorkflowStatus.FAILED
                            return {
                                "success": False,
                                "error": f"Step failed after retries: {step.name}"
                            }
                    else:
                        execution.status = WorkflowStatus.FAILED
                        return {
                            "success": False,
                            "error": f"Step failed: {step.name}"
                        }
            
            # Workflow completed successfully
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            execution.current_step = None
            
            return {
                "success": True,
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "workflow_name": workflow.name,
                "step_results": execution.step_results,
                "execution_time": (execution.completed_at - execution.started_at).total_seconds()
            }
        
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.completed_at = datetime.utcnow()
            
            self.log_error(f"Workflow execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _sort_steps(self, steps: List[WorkflowStep]) -> List[WorkflowStep]:
        """Sort steps by dependencies using topological sort."""
        step_map = {step.id: step for step in steps}
        dependency_graph = {}
        
        for step in steps:
            dependency_graph[step.id] = []
            for dep_id in step.dependencies:
                if dep_id in step_map:
                    dependency_graph[step.id].append(dep_id)
        
        sorted_steps = []
        visited = set()
        temp_visited = set()
        
        def visit(step_id: str):
            if step_id in temp_visited:
                raise ValueError("Circular dependency detected")
            if step_id in visited:
                return
            
            temp_visited.add(step_id)
            
            for dep_id in dependency_graph[step_id]:
                visit(dep_id)
            
            temp_visited.remove(step_id)
            visited.add(step_id)
            sorted_steps.append(step_map[step_id])
        
        for step in steps:
            if step.id not in visited:
                visit(step.id)
        
        return sorted_steps
    
    def _check_step_dependencies(self, step: WorkflowStep, execution: WorkflowExecution) -> bool:
        """Check if step dependencies are met."""
        for dep_id in step.dependencies:
            if dep_id not in execution.step_results:
                return False
        return True
    
    async def _execute_step(self, step: WorkflowStep, execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute a single workflow step."""
        executor = self.executors.get(step.step_type)
        if not executor:
            return {
                "success": False,
                "error": f"No executor found for step type: {step.step_type}"
            }
        
        try:
            # Check timeout
            if step.timeout:
                result = await asyncio.wait_for(
                    executor.execute(step, execution),
                    timeout=step.timeout
                )
            else:
                result = await executor.execute(step, execution)
            
            return result
        
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Step execution timed out: {step.name}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @log_async_function_call
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID."""
        return self.executions.get(execution_id)
    
    @log_async_function_call
    def list_executions(self, workflow_id: Optional[str] = None) -> List[WorkflowExecution]:
        """List workflow executions."""
        executions = list(self.executions.values())
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        return executions
    
    @log_async_function_call
    def save_workflow(self, workflow_id: str, file_path: str) -> bool:
        """Save workflow to file."""
        try:
            workflow = self.get_workflow(workflow_id)
            if not workflow:
                return False
            
            # Convert to serializable format
            workflow_data = {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "created_at": workflow.created_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat(),
                "variables": workflow.variables,
                "metadata": workflow.metadata,
                "steps": [
                    {
                        "id": step.id,
                        "name": step.name,
                        "step_type": step.step_type.value,
                        "config": step.config,
                        "dependencies": step.dependencies,
                        "timeout": step.timeout,
                        "max_retries": step.max_retries,
                        "metadata": step.metadata
                    }
                    for step in workflow.steps
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(workflow_data, f, indent=2)
            
            self.log_info(f"Saved workflow to: {file_path}")
            return True
        
        except Exception as e:
            self.log_error(f"Error saving workflow: {str(e)}")
            return False
    
    @log_async_function_call
    def load_workflow(self, file_path: str) -> Optional[Workflow]:
        """Load workflow from file."""
        try:
            with open(file_path, 'r') as f:
                workflow_data = json.load(f)
            
            # Recreate workflow
            workflow = Workflow(
                id=workflow_data["id"],
                name=workflow_data["name"],
                description=workflow_data["description"],
                created_at=datetime.fromisoformat(workflow_data["created_at"]),
                updated_at=datetime.fromisoformat(workflow_data["updated_at"]),
                variables=workflow_data["variables"],
                metadata=workflow_data["metadata"]
            )
            
            # Recreate steps
            for step_data in workflow_data["steps"]:
                step = WorkflowStep(
                    id=step_data["id"],
                    name=step_data["name"],
                    step_type=WorkflowStepType(step_data["step_type"]),
                    config=step_data["config"],
                    dependencies=step_data["dependencies"],
                    timeout=step_data["timeout"],
                    max_retries=step_data["max_retries"],
                    metadata=step_data["metadata"]
                )
                workflow.steps.append(step)
            
            self.workflows[workflow.id] = workflow
            self.log_info(f"Loaded workflow: {workflow.name}")
            return workflow
        
        except Exception as e:
            self.log_error(f"Error loading workflow: {str(e)}")
            return None 