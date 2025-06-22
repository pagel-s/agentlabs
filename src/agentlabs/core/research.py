"""Research project and task management for AgentLabs framework."""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field

from ..utils.logging import LoggerMixin
from .agent import Agent, AgentConfig, AgentContext, AgentResult, AgentRole, AgentFactory
from .llm import LLMConfig


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


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
    input_data: Union[str, Dict[str, Any]]
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[AgentResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    max_retries: int = 3
    retry_count: int = 0
    timeout: int = 3600  # 1 hour default
    
    def __post_init__(self) -> None:
        """Validate task after initialization."""
        if not self.name.strip():
            raise ValueError("Task name cannot be empty")
        if not self.description.strip():
            raise ValueError("Task description cannot be empty")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")


@dataclass
class ResearchProject:
    """A research project containing multiple tasks."""
    
    id: str
    name: str
    description: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "active"
    tasks: List[ResearchTask] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate project after initialization."""
        if not self.name.strip():
            raise ValueError("Project name cannot be empty")
        if not self.description.strip():
            raise ValueError("Project description cannot be empty")


class TaskExecutor(LoggerMixin):
    """Executor for research tasks."""
    
    def __init__(self, llm_config: LLMConfig) -> None:
        """Initialize task executor."""
        super().__init__()
        self.llm_config = llm_config
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    async def execute_task(self, task: ResearchTask, project_id: str) -> AgentResult:
        """Execute a research task."""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            
            self.log_info(f"Starting task '{task.name}' (ID: {task.id})")
            
            # Create agent context
            context = AgentContext(
                project_id=project_id,
                task_id=task.id,
                session_id=str(uuid.uuid4()),
                metadata=task.metadata
            )
            
            # Create agent
            agent = AgentFactory.create_agent(
                name=f"{task.name}_agent",
                role=task.agent_role,
                description=task.description,
                llm_config=self.llm_config,
                **task.metadata.get("agent_config", {})
            )
            
            # Execute agent
            result = await agent.execute(task.input_data)
            
            # Update task
            task.result = result
            task.completed_at = datetime.utcnow()
            task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
            
            self.log_info(f"Task '{task.name}' completed with status: {task.status}")
            
            return result
        
        except Exception as e:
            self.log_error(f"Error executing task '{task.name}': {e}")
            
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.result = AgentResult(
                success=False,
                output="",
                iterations=0,
                duration=0,
                metadata={},
                error=str(e)
            )
            
            return task.result
    
    async def execute_task_with_retry(self, task: ResearchTask, project_id: str) -> AgentResult:
        """Execute a task with retry logic."""
        last_result = None
        
        for attempt in range(task.max_retries + 1):
            try:
                task.retry_count = attempt
                self.log_info(f"Executing task '{task.name}' (attempt {attempt + 1}/{task.max_retries + 1})")
                
                result = await self.execute_task(task, project_id)
                last_result = result
                
                if result.success:
                    return result
                
                if attempt < task.max_retries:
                    self.log_warning(f"Task '{task.name}' failed, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                self.log_error(f"Error in attempt {attempt + 1} for task '{task.name}': {e}")
                if attempt < task.max_retries:
                    await asyncio.sleep(2 ** attempt)
        
        return last_result or AgentResult(
            success=False,
            output="",
            iterations=0,
            duration=0,
            metadata={},
            error="All retry attempts failed"
        )


class ProjectManager(LoggerMixin):
    """Manager for research projects."""
    
    def __init__(self, llm_config: LLMConfig) -> None:
        """Initialize project manager."""
        super().__init__()
        self.llm_config = llm_config
        self.executor = TaskExecutor(llm_config)
        self.projects: Dict[str, ResearchProject] = {}
    
    def create_project(self, name: str, description: str, **kwargs: Any) -> ResearchProject:
        """Create a new research project."""
        project_id = str(uuid.uuid4())
        project = ResearchProject(
            id=project_id,
            name=name,
            description=description,
            **kwargs
        )
        
        self.projects[project_id] = project
        self.log_info(f"Created project '{name}' (ID: {project_id})")
        
        return project
    
    def get_project(self, project_id: str) -> Optional[ResearchProject]:
        """Get a project by ID."""
        return self.projects.get(project_id)
    
    def list_projects(self) -> List[ResearchProject]:
        """List all projects."""
        return list(self.projects.values())
    
    def add_task(self, project_id: str, task: ResearchTask) -> bool:
        """Add a task to a project."""
        project = self.get_project(project_id)
        if not project:
            self.log_error(f"Project {project_id} not found")
            return False
        
        project.tasks.append(task)
        project.updated_at = datetime.utcnow()
        
        self.log_info(f"Added task '{task.name}' to project '{project.name}'")
        return True
    
    def create_task(
        self,
        project_id: str,
        name: str,
        description: str,
        agent_role: AgentRole,
        input_data: Union[str, Dict[str, Any]],
        **kwargs: Any
    ) -> Optional[ResearchTask]:
        """Create and add a task to a project."""
        task = ResearchTask(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            agent_role=agent_role,
            input_data=input_data,
            **kwargs
        )
        
        if self.add_task(project_id, task):
            return task
        return None
    
    async def execute_project(self, project_id: str, max_concurrent: int = 3) -> Dict[str, Any]:
        """Execute all tasks in a project."""
        project = self.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        self.log_info(f"Starting execution of project '{project.name}'")
        
        # Sort tasks by priority and dependencies
        sorted_tasks = self._sort_tasks(project.tasks)
        
        # Execute tasks with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}
        
        async def execute_task_with_semaphore(task: ResearchTask) -> None:
            async with semaphore:
                result = await self.executor.execute_task_with_retry(task, project_id)
                results[task.id] = result
        
        # Create tasks for execution
        tasks = []
        for task in sorted_tasks:
            if task.status == TaskStatus.PENDING:
                tasks.append(execute_task_with_semaphore(task))
        
        # Execute all tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update project status
        project.updated_at = datetime.utcnow()
        project.status = "completed"
        
        self.log_info(f"Project '{project.name}' execution completed")
        
        return {
            "project_id": project_id,
            "total_tasks": len(project.tasks),
            "completed_tasks": len([t for t in project.tasks if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in project.tasks if t.status == TaskStatus.FAILED]),
            "results": results
        }
    
    def _sort_tasks(self, tasks: List[ResearchTask]) -> List[ResearchTask]:
        """Sort tasks by priority and dependencies."""
        # Priority order: CRITICAL > HIGH > MEDIUM > LOW
        priority_order = {
            TaskPriority.CRITICAL: 4,
            TaskPriority.HIGH: 3,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 1
        }
        
        def task_key(task: ResearchTask) -> tuple:
            # Sort by priority first, then by creation time
            return (-priority_order[task.priority], task.created_at)
        
        return sorted(tasks, key=task_key)
    
    def get_project_summary(self, project_id: str) -> Dict[str, Any]:
        """Get a summary of project execution."""
        project = self.get_project(project_id)
        if not project:
            return {}
        
        task_statuses = {}
        for task in project.tasks:
            task_statuses[task.status.value] = task_statuses.get(task.status.value, 0) + 1
        
        total_duration = 0
        successful_tasks = 0
        
        for task in project.tasks:
            if task.result and task.result.success:
                successful_tasks += 1
                total_duration += task.result.duration
        
        return {
            "project_id": project_id,
            "name": project.name,
            "description": project.description,
            "status": project.status,
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat(),
            "total_tasks": len(project.tasks),
            "task_statuses": task_statuses,
            "successful_tasks": successful_tasks,
            "total_duration": total_duration,
            "average_duration": total_duration / successful_tasks if successful_tasks > 0 else 0
        }
    
    def save_project(self, project_id: str, file_path: str) -> bool:
        """Save project to file."""
        try:
            project = self.get_project(project_id)
            if not project:
                return False
            
            # Convert project to serializable format
            project_data = {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "created_at": project.created_at.isoformat(),
                "updated_at": project.updated_at.isoformat(),
                "status": project.status,
                "metadata": project.metadata,
                "settings": project.settings,
                "tasks": [
                    {
                        "id": task.id,
                        "name": task.name,
                        "description": task.description,
                        "agent_role": task.agent_role.value,
                        "input_data": task.input_data,
                        "status": task.status.value,
                        "priority": task.priority.value,
                        "created_at": task.created_at.isoformat(),
                        "started_at": task.started_at.isoformat() if task.started_at else None,
                        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                        "metadata": task.metadata,
                        "dependencies": task.dependencies,
                        "max_retries": task.max_retries,
                        "retry_count": task.retry_count,
                        "timeout": task.timeout
                    }
                    for task in project.tasks
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            self.log_info(f"Saved project '{project.name}' to {file_path}")
            return True
        
        except Exception as e:
            self.log_error(f"Error saving project: {e}")
            return False
    
    def load_project(self, file_path: str) -> Optional[ResearchProject]:
        """Load project from file."""
        try:
            with open(file_path, 'r') as f:
                project_data = json.load(f)
            
            # Reconstruct project
            project = ResearchProject(
                id=project_data["id"],
                name=project_data["name"],
                description=project_data["description"],
                created_at=datetime.fromisoformat(project_data["created_at"]),
                updated_at=datetime.fromisoformat(project_data["updated_at"]),
                status=project_data["status"],
                metadata=project_data["metadata"],
                settings=project_data["settings"]
            )
            
            # Reconstruct tasks
            for task_data in project_data["tasks"]:
                task = ResearchTask(
                    id=task_data["id"],
                    name=task_data["name"],
                    description=task_data["description"],
                    agent_role=AgentRole(task_data["agent_role"]),
                    input_data=task_data["input_data"],
                    status=TaskStatus(task_data["status"]),
                    priority=TaskPriority(task_data["priority"]),
                    created_at=datetime.fromisoformat(task_data["created_at"]),
                    started_at=datetime.fromisoformat(task_data["started_at"]) if task_data["started_at"] else None,
                    completed_at=datetime.fromisoformat(task_data["completed_at"]) if task_data["completed_at"] else None,
                    metadata=task_data["metadata"],
                    dependencies=task_data["dependencies"],
                    max_retries=task_data["max_retries"],
                    retry_count=task_data["retry_count"],
                    timeout=task_data["timeout"]
                )
                project.tasks.append(task)
            
            self.projects[project.id] = project
            self.log_info(f"Loaded project '{project.name}' from {file_path}")
            
            return project
        
        except Exception as e:
            self.log_error(f"Error loading project: {e}")
            return None 