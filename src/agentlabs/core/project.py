"""
Project management system for AgentLabs framework.
"""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from .agent import Agent, AgentConfig, AgentRole, AgentContext
from .llm import LLMProvider, LLMFactory
from .tools import ToolRegistry, create_default_tool_registry
from .memory import Memory, create_memory
from ..utils.logging import LoggedClass, log_async_function_call, log_async_execution_time


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ResearchTask:
    """A research task within a project."""
    id: str
    name: str
    description: str
    agent_role: AgentRole
    input_data: str
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Project:
    """A research project containing multiple tasks."""
    id: str
    name: str
    description: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tasks: List[ResearchTask] = field(default_factory=list)
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProjectManager(LoggedClass):
    """Manager for research projects."""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_registry: Optional[ToolRegistry] = None,
        memory: Optional[Memory] = None,
        max_concurrent_tasks: int = 3
    ):
        super().__init__()
        self.llm_provider = llm_provider
        self.tool_registry = tool_registry or create_default_tool_registry()
        self.memory = memory or create_memory("in_memory")
        self.max_concurrent_tasks = max_concurrent_tasks
        self.projects: Dict[str, Project] = {}
        self.agents: Dict[str, Agent] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    @log_async_function_call
    def create_project(self, name: str, description: str = "") -> Project:
        """Create a new research project."""
        project_id = str(uuid.uuid4())
        project = Project(
            id=project_id,
            name=name,
            description=description
        )
        
        self.projects[project_id] = project
        self.logger.info(f"Created project: {name} (ID: {project_id})")
        
        return project
    
    @log_async_function_call
    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        return self.projects.get(project_id)
    
    @log_async_function_call
    def list_projects(self) -> List[Project]:
        """List all projects."""
        return list(self.projects.values())
    
    @log_async_function_call
    def delete_project(self, project_id: str) -> bool:
        """Delete a project."""
        if project_id in self.projects:
            del self.projects[project_id]
            self.logger.info(f"Deleted project: {project_id}")
            return True
        return False
    
    @log_async_function_call
    def create_task(
        self,
        project_id: str,
        name: str,
        description: str,
        agent_role: AgentRole,
        input_data: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        dependencies: Optional[List[str]] = None
    ) -> Optional[ResearchTask]:
        """Create a new task in a project."""
        project = self.get_project(project_id)
        if not project:
            self.logger.error(f"Project not found: {project_id}")
            return None
        
        task_id = str(uuid.uuid4())
        task = ResearchTask(
            id=task_id,
            name=name,
            description=description,
            agent_role=agent_role,
            input_data=input_data,
            priority=priority,
            dependencies=dependencies or []
        )
        
        project.tasks.append(task)
        project.updated_at = datetime.utcnow()
        
        self.logger.info(f"Created task: {name} in project {project.name}")
        return task
    
    @log_async_function_call
    def get_task(self, project_id: str, task_id: str) -> Optional[ResearchTask]:
        """Get a task by ID."""
        project = self.get_project(project_id)
        if not project:
            return None
        
        for task in project.tasks:
            if task.id == task_id:
                return task
        return None
    
    @log_async_function_call
    def update_task_status(self, project_id: str, task_id: str, status: TaskStatus) -> bool:
        """Update task status."""
        task = self.get_task(project_id, task_id)
        if not task:
            return False
        
        task.status = status
        if status == TaskStatus.RUNNING and not task.started_at:
            task.started_at = datetime.utcnow()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            task.completed_at = datetime.utcnow()
        
        project = self.get_project(project_id)
        if project:
            project.updated_at = datetime.utcnow()
        
        return True
    
    @log_async_function_call
    def get_or_create_agent(self, role: AgentRole) -> Agent:
        """Get or create an agent for a specific role."""
        agent_key = f"{role.value}_agent"
        
        if agent_key not in self.agents:
            # Create agent configuration
            config = AgentConfig(
                name=f"{role.value}_agent",
                role=role,
                description=f"Agent for {role.value} tasks",
                tools=[tool.name for tool in self.tool_registry.tools.values()]
            )
            
            # Create agent
            agent = Agent(
                config=config,
                llm_provider=self.llm_provider,
                tool_registry=self.tool_registry,
                memory=self.memory
            )
            
            self.agents[agent_key] = agent
            self.logger.info(f"Created agent for role: {role.value}")
        
        return self.agents[agent_key]
    
    @log_async_function_call
    @log_async_execution_time
    async def execute_task(self, project_id: str, task_id: str) -> Dict[str, Any]:
        """Execute a single task."""
        async with self._semaphore:
            task = self.get_task(project_id, task_id)
            if not task:
                return {"success": False, "error": "Task not found"}
            
            # Check dependencies
            if not await self._check_dependencies(project_id, task):
                return {"success": False, "error": "Task dependencies not met"}
            
            # Update status
            self.update_task_status(project_id, task_id, TaskStatus.RUNNING)
            
            try:
                # Get or create agent
                agent = self.get_or_create_agent(task.agent_role)
                
                # Create context
                context = AgentContext(
                    session_id=f"{project_id}_{task_id}",
                    project_id=project_id,
                    task_id=task_id
                )
                
                # Execute task
                result = await agent.execute(task.input_data, context)
                
                # Update task with result
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                
                self.logger.info(f"Task completed: {task.name}")
                return {"success": True, "result": result}
            
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.utcnow()
                
                self.logger.error(f"Task failed: {task.name} - {str(e)}")
                return {"success": False, "error": str(e)}
    
    async def _check_dependencies(self, project_id: str, task: ResearchTask) -> bool:
        """Check if task dependencies are met."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            dep_task = self.get_task(project_id, dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    @log_async_function_call
    @log_async_execution_time
    async def execute_project(self, project_id: str) -> Dict[str, Any]:
        """Execute all tasks in a project."""
        project = self.get_project(project_id)
        if not project:
            return {"success": False, "error": "Project not found"}
        
        self.logger.info(f"Starting project execution: {project.name}")
        
        # Sort tasks by priority and dependencies
        sorted_tasks = self._sort_tasks(project.tasks)
        
        # Execute tasks
        results = []
        for task in sorted_tasks:
            result = await self.execute_task(project_id, task.id)
            results.append({
                "task_id": task.id,
                "task_name": task.name,
                "result": result
            })
        
        # Update project status
        all_completed = all(
            task.status == TaskStatus.COMPLETED
            for task in project.tasks
        )
        
        if all_completed:
            project.status = "completed"
        else:
            project.status = "partial"
        
        project.updated_at = datetime.utcnow()
        
        return {
            "success": True,
            "project_id": project_id,
            "project_name": project.name,
            "results": results,
            "status": project.status
        }
    
    def _sort_tasks(self, tasks: List[ResearchTask]) -> List[ResearchTask]:
        """Sort tasks by priority and dependencies."""
        # Create dependency graph
        task_map = {task.id: task for task in tasks}
        dependency_graph = {}
        
        for task in tasks:
            dependency_graph[task.id] = []
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    dependency_graph[task.id].append(dep_id)
        
        # Topological sort with priority
        sorted_tasks = []
        visited = set()
        temp_visited = set()
        
        def visit(task_id: str):
            if task_id in temp_visited:
                raise ValueError("Circular dependency detected")
            if task_id in visited:
                return
            
            temp_visited.add(task_id)
            
            for dep_id in dependency_graph[task_id]:
                visit(dep_id)
            
            temp_visited.remove(task_id)
            visited.add(task_id)
            sorted_tasks.append(task_map[task_id])
        
        # Sort by priority first, then by dependencies
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3
        }
        
        sorted_by_priority = sorted(
            tasks,
            key=lambda t: priority_order[t.priority]
        )
        
        for task in sorted_by_priority:
            if task.id not in visited:
                visit(task.id)
        
        return sorted_tasks
    
    @log_async_function_call
    def save_project(self, project_id: str, file_path: str) -> bool:
        """Save project to file."""
        try:
            project = self.get_project(project_id)
            if not project:
                return False
            
            # Convert to serializable format
            project_data = {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "created_at": project.created_at.isoformat(),
                "updated_at": project.updated_at.isoformat(),
                "status": project.status,
                "metadata": project.metadata,
                "tasks": [
                    {
                        "id": task.id,
                        "name": task.name,
                        "description": task.description,
                        "agent_role": task.agent_role.value,
                        "input_data": task.input_data,
                        "priority": task.priority.value,
                        "dependencies": task.dependencies,
                        "status": task.status.value,
                        "created_at": task.created_at.isoformat(),
                        "started_at": task.started_at.isoformat() if task.started_at else None,
                        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                        "result": task.result,
                        "error": task.error,
                        "metadata": task.metadata
                    }
                    for task in project.tasks
                ]
            }
            
            import json
            with open(file_path, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            self.logger.info(f"Saved project to: {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving project: {str(e)}")
            return False
    
    @log_async_function_call
    def load_project(self, file_path: str) -> Optional[Project]:
        """Load project from file."""
        try:
            import json
            with open(file_path, 'r') as f:
                project_data = json.load(f)
            
            # Recreate project
            project = Project(
                id=project_data["id"],
                name=project_data["name"],
                description=project_data["description"],
                created_at=datetime.fromisoformat(project_data["created_at"]),
                updated_at=datetime.fromisoformat(project_data["updated_at"]),
                status=project_data["status"],
                metadata=project_data["metadata"]
            )
            
            # Recreate tasks
            for task_data in project_data["tasks"]:
                task = ResearchTask(
                    id=task_data["id"],
                    name=task_data["name"],
                    description=task_data["description"],
                    agent_role=AgentRole(task_data["agent_role"]),
                    input_data=task_data["input_data"],
                    priority=TaskPriority(task_data["priority"]),
                    dependencies=task_data["dependencies"],
                    status=TaskStatus(task_data["status"]),
                    created_at=datetime.fromisoformat(task_data["created_at"]),
                    started_at=datetime.fromisoformat(task_data["started_at"]) if task_data["started_at"] else None,
                    completed_at=datetime.fromisoformat(task_data["completed_at"]) if task_data["completed_at"] else None,
                    result=task_data["result"],
                    error=task_data["error"],
                    metadata=task_data["metadata"]
                )
                project.tasks.append(task)
            
            self.projects[project.id] = project
            self.logger.info(f"Loaded project: {project.name}")
            return project
        
        except Exception as e:
            self.logger.error(f"Error loading project: {str(e)}")
            return None
    
    def get_project_summary(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get project summary with task statistics."""
        project = self.get_project(project_id)
        if not project:
            return None
        
        total_tasks = len(project.tasks)
        completed_tasks = len([t for t in project.tasks if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in project.tasks if t.status == TaskStatus.FAILED])
        pending_tasks = len([t for t in project.tasks if t.status == TaskStatus.PENDING])
        
        return {
            "project_id": project.id,
            "project_name": project.name,
            "status": project.status,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "pending_tasks": pending_tasks,
            "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat()
        } 