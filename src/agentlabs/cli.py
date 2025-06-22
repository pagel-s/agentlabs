"""
Command-line interface for AgentLabs framework.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.tree import Tree

from .core.config import Config, Settings
from .core.llm import LLMFactory, LLMConfig
from .core.tools import ToolRegistry, create_default_tool_registry
from .core.memory import create_memory
from .core.agent import Agent, AgentConfig, AgentRole, AgentContext
from .core.project import ProjectManager, Project, ResearchTask, TaskStatus, TaskPriority
from .core.workflow import WorkflowEngine, Workflow, WorkflowStep, WorkflowStepType
from .utils.logging import setup_logging, get_logger

app = typer.Typer(help="AgentLabs - LLM Agent Research Framework")
console = Console()


@app.command()
def init(
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    interactive: bool = typer.Option(True, "--interactive", "-i", help="Interactive configuration setup")
):
    """Initialize AgentLabs configuration."""
    console.print(Panel.fit("🔧 AgentLabs Configuration Setup", style="bold blue"))
    
    if config_file and Path(config_file).exists():
        config = Config(config_file)
        console.print(f"✅ Loaded existing configuration from: {config_file}")
    else:
        config = Config()
        console.print("📝 Creating new configuration...")
    
    if interactive:
        _setup_interactive_config(config)
    
    # Save configuration
    save_path = config_file or "agentlabs_config.json"
    config.save_config(save_path)
    console.print(f"✅ Configuration saved to: {save_path}")
    
    # Setup logging
    setup_logging(config.get_logging_config())
    console.print("✅ Logging configured")


def _setup_interactive_config(config: Config):
    """Interactive configuration setup."""
    console.print("\n[bold]LLM Configuration[/bold]")
    
    # LLM Provider
    providers = LLMFactory.list_providers()
    provider = Prompt.ask(
        "Select LLM provider",
        choices=providers,
        default=config.settings.llm.provider
    )
    config.settings.llm.provider = provider
    
    # Model
    model = Prompt.ask(
        "Model name",
        default=config.settings.llm.model
    )
    config.settings.llm.model = model
    
    # API Key
    api_key = Prompt.ask(
        "API Key (leave empty to use environment variable)",
        password=True,
        default=""
    )
    if api_key:
        config.settings.llm.api_key = api_key
    
    # Temperature
    temperature = Prompt.ask(
        "Temperature",
        default=str(config.settings.llm.temperature),
        type=float
    )
    config.settings.llm.temperature = float(temperature)
    
    console.print("\n[bold]Tool Configuration[/bold]")
    
    # Web search
    web_search = Confirm.ask(
        "Enable web search tool?",
        default=config.settings.tools.web_search_enabled
    )
    config.settings.tools.web_search_enabled = web_search
    
    # Data analysis
    data_analysis = Confirm.ask(
        "Enable data analysis tools?",
        default=config.settings.tools.data_analysis_enabled
    )
    config.settings.tools.data_analysis_enabled = data_analysis


@app.command()
def list_tools():
    """List available tools."""
    console.print(Panel.fit("🔧 Available Tools", style="bold blue"))
    
    registry = create_default_tool_registry()
    tools = registry.list_tools()
    
    if not tools:
        console.print("No tools available.")
        return
    
    table = Table(title="Available Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Parameters", style="yellow")
    
    for tool_name in tools:
        tool = registry.get_tool(tool_name)
        if tool:
            params = ", ".join([p.name for p in tool.schema])
            table.add_row(tool_name, tool.description, params)
    
    console.print(table)


@app.command()
def create_agent(
    name: str = typer.Argument(..., help="Agent name"),
    role: str = typer.Option("general", "--role", "-r", help="Agent role"),
    description: str = typer.Option("", "--description", "-d", help="Agent description"),
    config_file: str = typer.Option("agentlabs_config.json", "--config", "-c", help="Configuration file")
):
    """Create a new agent."""
    console.print(Panel.fit(f"🤖 Creating Agent: {name}", style="bold blue"))
    
    # Load configuration
    config = Config(config_file)
    setup_logging(config.get_logging_config())
    
    # Create LLM provider
    llm_provider = LLMFactory.create_provider(config.get_llm_config())
    
    # Create tool registry
    tool_registry = create_default_tool_registry()
    
    # Create memory
    memory = create_memory("in_memory")
    
    # Create agent configuration
    agent_config = AgentConfig(
        name=name,
        role=AgentRole(role),
        description=description
    )
    
    # Create agent
    agent = Agent(
        config=agent_config,
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        memory=memory
    )
    
    console.print(f"✅ Agent '{name}' created successfully!")
    console.print(f"   Role: {role}")
    console.print(f"   Description: {description}")
    console.print(f"   ID: {agent.id}")


@app.command()
def run_agent(
    name: str = typer.Argument(..., help="Agent name"),
    task: str = typer.Argument(..., help="Task to execute"),
    config_file: str = typer.Option("agentlabs_config.json", "--config", "-c", help="Configuration file")
):
    """Run an agent with a specific task."""
    console.print(Panel.fit(f"🚀 Running Agent: {name}", style="bold blue"))
    
    # Load configuration
    config = Config(config_file)
    setup_logging(config.get_logging_config())
    
    # Create LLM provider
    llm_provider = LLMFactory.create_provider(config.get_llm_config())
    
    # Create tool registry
    tool_registry = create_default_tool_registry()
    
    # Create memory
    memory = create_memory("in_memory")
    
    # Create agent configuration
    agent_config = AgentConfig(
        name=name,
        role=AgentRole.GENERAL,
        description=f"Agent for task: {task}"
    )
    
    # Create agent
    agent = Agent(
        config=agent_config,
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        memory=memory
    )
    
    # Create context
    context = AgentContext(session_id=f"cli_{name}")
    
    # Execute agent
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task_id = progress.add_task("Executing agent...", total=None)
        
        async def execute():
            return await agent.execute(task, context)
        
        result = asyncio.run(execute())
    
    # Display results
    console.print("\n[bold green]✅ Task completed![/bold green]")
    console.print(f"Execution time: {result.get('execution_time', 0):.2f} seconds")
    
    if "result" in result:
        console.print("\n[bold]Result:[/bold]")
        console.print(result["result"])
    
    if "error" in result:
        console.print(f"\n[bold red]Error:[/bold red] {result['error']}")


@app.command()
def create_project(
    name: str = typer.Argument(..., help="Project name"),
    description: str = typer.Option("", "--description", "-d", help="Project description"),
    config_file: str = typer.Option("agentlabs_config.json", "--config", "-c", help="Configuration file")
):
    """Create a new research project."""
    console.print(Panel.fit(f"📁 Creating Project: {name}", style="bold blue"))
    
    # Load configuration
    config = Config(config_file)
    setup_logging(config.get_logging_config())
    
    # Create LLM provider
    llm_provider = LLMFactory.create_provider(config.get_llm_config())
    
    # Create project manager
    project_manager = ProjectManager(llm_provider)
    
    # Create project
    project = project_manager.create_project(name, description)
    
    console.print(f"✅ Project '{name}' created successfully!")
    console.print(f"   ID: {project.id}")
    console.print(f"   Description: {description}")
    console.print(f"   Created: {project.created_at}")


@app.command()
def add_task(
    project_id: str = typer.Argument(..., help="Project ID"),
    name: str = typer.Argument(..., help="Task name"),
    description: str = typer.Argument(..., help="Task description"),
    role: str = typer.Option("researcher", "--role", "-r", help="Agent role"),
    input_data: str = typer.Option("", "--input", "-i", help="Input data"),
    priority: str = typer.Option("medium", "--priority", "-p", help="Task priority"),
    config_file: str = typer.Option("agentlabs_config.json", "--config", "-c", help="Configuration file")
):
    """Add a task to a project."""
    console.print(Panel.fit(f"➕ Adding Task: {name}", style="bold blue"))
    
    # Load configuration
    config = Config(config_file)
    setup_logging(config.get_logging_config())
    
    # Create LLM provider
    llm_provider = LLMFactory.create_provider(config.get_llm_config())
    
    # Create project manager
    project_manager = ProjectManager(llm_provider)
    
    # Get project
    project = project_manager.get_project(project_id)
    if not project:
        console.print(f"[red]❌ Project not found: {project_id}[/red]")
        return
    
    # Create task
    task = project_manager.create_task(
        project_id=project_id,
        name=name,
        description=description,
        agent_role=AgentRole(role),
        input_data=input_data,
        priority=TaskPriority(priority)
    )
    
    if task:
        console.print(f"✅ Task '{name}' added successfully!")
        console.print(f"   ID: {task.id}")
        console.print(f"   Role: {role}")
        console.print(f"   Priority: {priority}")
    else:
        console.print("[red]❌ Failed to create task[/red]")


@app.command()
def run_project(
    project_id: str = typer.Argument(..., help="Project ID"),
    config_file: str = typer.Option("config.json", "--config", "-c", help="Configuration file path"),
    max_concurrent: int = typer.Option(3, "--max-concurrent", "-m", help="Maximum concurrent tasks")
):
    """Run a research project."""
    try:
        settings = setup_environment()
        
        # Load LLM config
        llm_config = LLMConfig(
            provider=settings.llm.provider,
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens
        )
        
        # Create project manager
        project_manager = ProjectManager(llm_config)
        
        # Get project
        project = project_manager.get_project(project_id)
        if not project:
            console.print(f"Project {project_id} not found.", style="red")
            sys.exit(1)
        
        console.print(f"Running project: {project.name}", style="blue")
        console.print(f"Total tasks: {len(project.tasks)}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Executing project...", total=None)
            
            # Run project
            result = asyncio.run(project_manager.execute_project(project_id, max_concurrent))
            
            progress.update(task, completed=True)
        
        # Display results
        console.print("✅ Project execution completed!", style="green")
        console.print(f"Total tasks: {result['total_tasks']}")
        console.print(f"Completed: {result['completed_tasks']}")
        console.print(f"Failed: {result['failed_tasks']}")
        
        # Show project summary
        summary = project_manager.get_project_summary(project_id)
        if summary:
            console.print(f"Total duration: {summary['total_duration']:.2f}s")
            console.print(f"Average duration: {summary['average_duration']:.2f}s")
    
    except Exception as e:
        console.print(f"Error running project: {e}", style="red")
        sys.exit(1)


@app.command()
def list_projects(
    config_file: str = typer.Option("config.json", "--config", "-c", help="Configuration file path")
):
    """List all projects."""
    try:
        settings = setup_environment()
        
        # Load LLM config
        llm_config = LLMConfig(
            provider=settings.llm.provider,
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens
        )
        
        # Create project manager
        project_manager = ProjectManager(llm_config)
        
        # List projects
        projects = project_manager.list_projects()
        
        if not projects:
            console.print("No projects found.", style="yellow")
            return
        
        table = Table(title="Research Projects")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Description", style="green")
        table.add_column("Status", style="blue")
        table.add_column("Tasks", style="yellow")
        table.add_column("Created", style="magenta")
        
        for project in projects:
            completed_tasks = len([t for t in project.tasks if t.status.value == "completed"])
            table.add_row(
                project.id[:8] + "...",
                project.name,
                project.description[:50] + "..." if len(project.description) > 50 else project.description,
                project.status,
                f"{completed_tasks}/{len(project.tasks)}",
                project.created_at.strftime("%Y-%m-%d %H:%M")
            )
        
        console.print(table)
    
    except Exception as e:
        console.print(f"Error listing projects: {e}", style="red")
        sys.exit(1)


@app.command()
def create_workflow(
    name: str = typer.Argument(..., help="Workflow name"),
    description: str = typer.Option("", "--description", "-d", help="Workflow description"),
    config_file: str = typer.Option("config.json", "--config", "-c", help="Configuration file path")
):
    """Create a new workflow."""
    try:
        settings = setup_environment()
        
        # Load LLM config
        llm_config = LLMConfig(
            provider=settings.llm.provider,
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens
        )
        
        # Create workflow engine
        workflow_engine = WorkflowEngine(llm_config)
        
        # Create workflow
        workflow = workflow_engine.create_workflow(name, description)
        
        console.print(f"Workflow '{name}' created successfully!", style="green")
        console.print(f"Workflow ID: {workflow.id}")
        console.print(f"Description: {workflow.description}")
        console.print(f"Created: {workflow.created_at}")
    
    except Exception as e:
        console.print(f"Error creating workflow: {e}", style="red")
        sys.exit(1)


@app.command()
def add_workflow_step(
    workflow_id: str = typer.Argument(..., help="Workflow ID"),
    name: str = typer.Argument(..., help="Step name"),
    step_type: str = typer.Argument(..., help="Step type (agent, condition)"),
    config_file: str = typer.Option("config.json", "--config", "-c", help="Configuration file path")
):
    """Add a step to a workflow."""
    try:
        settings = setup_environment()
        
        # Load LLM config
        llm_config = LLMConfig(
            provider=settings.llm.provider,
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens
        )
        
        # Create workflow engine
        workflow_engine = WorkflowEngine(llm_config)
        
        # Create step
        step = workflow_engine.create_step(
            workflow_id=workflow_id,
            name=name,
            step_type=WorkflowStepType(step_type),
            config={}
        )
        
        if step:
            console.print(f"Step '{name}' added to workflow successfully!", style="green")
            console.print(f"Step ID: {step.id}")
            console.print(f"Type: {step.step_type.value}")
        else:
            console.print("Failed to add step to workflow.", style="red")
    
    except Exception as e:
        console.print(f"Error adding workflow step: {e}", style="red")
        sys.exit(1)


@app.command()
def run_workflow(
    workflow_id: str = typer.Argument(..., help="Workflow ID"),
    config_file: str = typer.Option("config.json", "--config", "-c", help="Configuration file path")
):
    """Run a workflow."""
    try:
        settings = setup_environment()
        
        # Load LLM config
        llm_config = LLMConfig(
            provider=settings.llm.provider,
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens
        )
        
        # Create workflow engine
        workflow_engine = WorkflowEngine(llm_config)
        
        # Get workflow
        workflow = workflow_engine.get_workflow(workflow_id)
        if not workflow:
            console.print(f"Workflow {workflow_id} not found.", style="red")
            sys.exit(1)
        
        console.print(f"Running workflow: {workflow.name}", style="blue")
        console.print(f"Total steps: {len(workflow.steps)}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Executing workflow...", total=None)
            
            # Run workflow
            result = asyncio.run(workflow_engine.execute_workflow(workflow_id))
            
            progress.update(task, completed=True)
        
        # Display results
        console.print("✅ Workflow execution completed!", style="green")
        console.print(f"Total steps: {result['total_steps']}")
        console.print(f"Executed steps: {result['executed_steps']}")
        console.print(f"Status: {result['status']}")
    
    except Exception as e:
        console.print(f"Error running workflow: {e}", style="red")
        sys.exit(1)


@app.command()
def list_workflows(
    config_file: str = typer.Option("config.json", "--config", "-c", help="Configuration file path")
):
    """List all workflows."""
    try:
        settings = setup_environment()
        
        # Load LLM config
        llm_config = LLMConfig(
            provider=settings.llm.provider,
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens
        )
        
        # Create workflow engine
        workflow_engine = WorkflowEngine(llm_config)
        
        # List workflows
        workflows = workflow_engine.list_workflows()
        
        if not workflows:
            console.print("No workflows found.", style="yellow")
            return
        
        table = Table(title="Workflows")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Description", style="green")
        table.add_column("Status", style="blue")
        table.add_column("Steps", style="yellow")
        table.add_column("Created", style="magenta")
        
        for workflow in workflows:
            table.add_row(
                workflow.id[:8] + "...",
                workflow.name,
                workflow.description[:50] + "..." if len(workflow.description) > 50 else workflow.description,
                workflow.status.value,
                str(len(workflow.steps)),
                workflow.created_at.strftime("%Y-%m-%d %H:%M")
            )
        
        console.print(table)
    
    except Exception as e:
        console.print(f"Error listing workflows: {e}", style="red")
        sys.exit(1)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main() 