# AgentLabs - LLM Agent Research Framework

A comprehensive framework for building, testing, and deploying LLM agents for research applications including data collection, analysis, and automation.

## Features

- **Multi-Provider LLM Support**: OpenAI, Anthropic, and local models
- **Agent System**: Configurable agents with different roles (researcher, analyst, writer, validator)
- **Tool System**: Extensible tool registry with built-in tools for web search, document reading, and data analysis
- **Memory System**: Context-aware memory with multiple storage backends
- **Project Management**: Research project orchestration with task dependencies
- **Workflow Engine**: Complex workflow automation with conditional logic
- **CLI Interface**: Rich command-line interface for easy interaction
- **Comprehensive Testing**: Full test coverage with pytest
- **Configuration Management**: Flexible configuration with environment variables and files

## Installation

### Prerequisites

- Python 3.12+
- pip or uv

### Install from source

```bash
# Clone the repository
git clone https://github.com/pagel-s/agentlabs.git
cd agentlabs

# Install dependencies
pip install -e .

# Or using uv
uv sync
```

### Install dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Initialize Configuration

```bash
# Create default configuration
agentlabs init

# Or with custom config file
agentlabs init --config my_config.json
```

### 2. Set up API Keys

Edit the configuration file and add your API keys:

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "your-openai-api-key",
    "temperature": 0.7,
    "max_tokens": 4096
  }
}
```

### 3. Create and Run an Agent

```bash
# Create a research agent
agentlabs create-agent "my_researcher" --role researcher --description "Agent for research tasks"

# Run the agent
agentlabs run-agent "my_researcher" "Research the latest developments in AI"
```

### 4. Create a Research Project

```bash
# Create a project
agentlabs create-project "AI Research" --description "Research project on AI developments"

# Add tasks to the project
agentlabs add-task "project_id" "Literature Review" "Review recent AI papers" --role researcher --input "Find recent papers on transformer models"

# Run the project
agentlabs run-project "project_id"
```

## Core Concepts

### Agents

Agents are the core components that perform research tasks. Each agent has:

- **Role**: Defines the agent's specialization (researcher, analyst, writer, validator)
- **Tools**: Set of tools the agent can use
- **Memory**: Context awareness across sessions
- **Configuration**: LLM settings, timeouts, retry logic

```python
from agentlabs import AgentFactory, AgentRole, LLMConfig

# Create LLM configuration
llm_config = LLMConfig(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key"
)

# Create a research agent
agent = AgentFactory.create_agent(
    name="research_agent",
    role=AgentRole.RESEARCHER,
    description="Agent for conducting research",
    llm_config=llm_config
)

# Execute the agent
result = await agent.execute("Research quantum computing applications")
```

### Tools

Tools provide agents with capabilities to interact with external systems:

- **Web Search**: Search the web for information
- **Document Reader**: Extract content from web pages
- **Data Analyzer**: Analyze structured and unstructured data
- **Custom Tools**: Extend with your own tools

```python
from agentlabs.core.tools import Tool, ToolSchema, ToolResult

class CustomTool(Tool):
    def __init__(self):
        schema = [
            ToolSchema("input", "string", "Input data", required=True)
        ]
        super().__init__("custom_tool", "My custom tool", schema)
    
    async def execute(self, args):
        # Tool implementation
        return ToolResult(success=True, data="processed data")
```

### Projects

Projects organize multiple research tasks with dependencies:

```python
from agentlabs import ProjectManager, ResearchTask, AgentRole, TaskPriority

# Create project manager
project_manager = ProjectManager(llm_config)

# Create project
project = project_manager.create_project(
    name="Market Research",
    description="Research market trends"
)

# Add tasks
task = project_manager.create_task(
    project_id=project.id,
    name="Data Collection",
    description="Collect market data",
    agent_role=AgentRole.RESEARCHER,
    input_data="Find market data for tech industry",
    priority=TaskPriority.HIGH
)

# Execute project
results = await project_manager.execute_project(project.id)
```

### Workflows

Workflows orchestrate complex research processes:

```python
from agentlabs import WorkflowEngine, WorkflowStepType

# Create workflow engine
workflow_engine = WorkflowEngine(llm_config)

# Create workflow
workflow = workflow_engine.create_workflow(
    name="Research Workflow",
    description="Automated research process"
)

# Add workflow steps
step = workflow_engine.create_step(
    workflow_id=workflow.id,
    name="Data Collection",
    step_type=WorkflowStepType.AGENT,
    config={
        "agent": {
            "role": "researcher",
            "input_template": "Collect data about {topic}"
        }
    }
)

# Execute workflow
result = await workflow_engine.execute_workflow(workflow.id)
```

## Configuration

### Environment Variables

```bash
# Core settings
export APP_NAME="MyAgentLabs"
export DEBUG=true

# LLM settings
export LLM_PROVIDER="openai"
export LLM_MODEL="gpt-4"
export LLM_API_KEY="your-api-key"
export LLM_TEMPERATURE="0.7"

# Database settings
export DB_URL="postgresql://user:pass@localhost/agentlabs"

# Redis settings
export REDIS_URL="redis://localhost:6379"

# Logging settings
export LOG_LEVEL="INFO"
export LOG_FILE="logs/agentlabs.log"
```

### Configuration File

```json
{
  "app_name": "AgentLabs",
  "debug": false,
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "your-api-key",
    "temperature": 0.7,
    "max_tokens": 4096
  },
  "database": {
    "url": "sqlite:///agentlabs.db",
    "echo": false
  },
  "redis": {
    "url": "redis://localhost:6379",
    "db": 0
  },
  "vectorstore": {
    "provider": "chroma",
    "path": "./vectorstore",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "logging": {
    "level": "INFO",
    "file": "logs/agentlabs.log"
  }
}
```

## CLI Commands

### Agent Management

```bash
# List available tools
agentlabs list-tools

# Create an agent
agentlabs create-agent "my_agent" --role researcher --description "Research agent"

# Run an agent
agentlabs run-agent "my_agent" "Research topic" --output results.json
```

### Project Management

```bash
# Create a project
agentlabs create-project "Research Project" --description "My research project"

# List projects
agentlabs list-projects

# Add task to project
agentlabs add-task "project_id" "Task Name" "Task description" --role researcher

# Run project
agentlabs run-project "project_id" --max-concurrent 3
```

### Workflow Management

```bash
# Create workflow
agentlabs create-workflow "My Workflow" --description "Automated workflow"

# List workflows
agentlabs list-workflows

# Add workflow step
agentlabs add-workflow-step "workflow_id" "Step Name" "agent"

# Run workflow
agentlabs run-workflow "workflow_id"
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentlabs --cov-report=html

# Run specific test categories
pytest -m "unit"
pytest -m "integration"
pytest -m "not slow"
```

## Development

### Project Structure

```
agentlabs/
├── src/agentlabs/
│   ├── __init__.py
│   ├── cli.py                 # Command-line interface
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py          # Agent system
│   │   ├── llm.py            # LLM providers
│   │   ├── tools.py          # Tool system
│   │   ├── memory.py         # Memory system
│   │   ├── research.py       # Project management
│   │   └── workflow.py       # Workflow engine
│   └── utils/
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       └── logging.py        # Logging setup
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Test configuration
│   ├── test_config.py        # Configuration tests
│   ├── test_tools.py         # Tool tests
│   └── ...
├── pyproject.toml            # Project configuration
├── README.md                 # This file
└── requirements.txt          # Dependencies
```

### Adding New Tools

1. Create a new tool class inheriting from `Tool`
2. Implement the `execute` method
3. Register the tool in the registry

```python
from agentlabs.core.tools import Tool, ToolSchema, ToolResult

class MyCustomTool(Tool):
    def __init__(self):
        schema = [
            ToolSchema("input", "string", "Input data", required=True)
        ]
        super().__init__("my_tool", "My custom tool", schema)
    
    async def execute(self, args):
        # Your tool implementation
        input_data = args["input"]
        # Process input_data
        return ToolResult(success=True, data="processed result")

# Register the tool
from agentlabs.core.tools import tool_registry
tool_registry.register_tool(MyCustomTool())
```

### Adding New LLM Providers

1. Create a new provider class inheriting from `LLMProvider`
2. Implement the `generate` and `generate_stream` methods
3. Register the provider in the factory

```python
from agentlabs.core.llm import LLMProvider, LLMFactory

class MyLLMProvider(LLMProvider):
    async def generate(self, messages, **kwargs):
        # Your LLM implementation
        pass
    
    async def generate_stream(self, messages, **kwargs):
        # Your streaming implementation
        pass

# Register the provider
LLMFactory.register_provider("my_llm", MyLLMProvider)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:

- Create an issue on GitHub
- Check the documentation
- Join our community discussions

## Roadmap

- [ ] Database integration for persistent storage
- [ ] Vector database support for embeddings
- [ ] Web UI for project management
- [ ] Real-time collaboration features
- [ ] Advanced workflow orchestration
- [ ] Plugin system for custom integrations
- [ ] Performance monitoring and analytics
- [ ] Multi-agent coordination
- [ ] Advanced memory systems
- [ ] Export/import functionality
