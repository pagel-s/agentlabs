"""
Pytest configuration and fixtures for AgentLabs tests.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from agentlabs.core.config import Config, LLMConfig
from agentlabs.core.llm import LLMProvider, LLMMessage, LLMResponse
from agentlabs.core.tools import ToolRegistry, create_default_tool_registry
from agentlabs.core.memory import create_memory
from agentlabs.core.agent import Agent, AgentConfig, AgentRole, AgentContext


@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()


@pytest.fixture
def llm_config():
    """Create a test LLM configuration."""
    return LLMConfig(
        provider="openai",
        model="gpt-4",
        api_key="test-key",
        temperature=0.7,
        max_tokens=1000
    )


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = Mock(spec=LLMProvider)
    
    # Mock generate method
    async def mock_generate(messages, **kwargs):
        return LLMResponse(
            content="This is a test response",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="stop"
        )
    
    provider.generate = AsyncMock(side_effect=mock_generate)
    provider.generate_stream = AsyncMock()
    
    return provider


@pytest.fixture
def tool_registry():
    """Create a test tool registry."""
    return create_default_tool_registry()


@pytest.fixture
def memory():
    """Create a test memory instance."""
    return create_memory("in_memory")


@pytest.fixture
def agent_config():
    """Create a test agent configuration."""
    return AgentConfig(
        name="test_agent",
        role=AgentRole.GENERAL,
        description="Test agent for unit tests",
        system_prompt="You are a test agent.",
        tools=["web_search", "data_analysis"],
        memory_enabled=True,
        max_iterations=5,
        timeout=60
    )


@pytest.fixture
def agent_context():
    """Create a test agent context."""
    return AgentContext(
        session_id="test_session",
        user_id="test_user",
        project_id="test_project",
        task_id="test_task"
    )


@pytest.fixture
def agent(mock_llm_provider, tool_registry, memory, agent_config):
    """Create a test agent."""
    return Agent(
        config=agent_config,
        llm_provider=mock_llm_provider,
        tool_registry=tool_registry,
        memory=memory
    )


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "csv_data": "name,age,city\nJohn,30,NYC\nJane,25,LA\nBob,35,Chicago",
        "json_data": '[{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]',
        "text_data": "This is sample text data for testing purposes."
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Pytest markers
pytest_plugins = ["pytest_asyncio"]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests in integration/ directory as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        # Mark tests in unit/ directory as unit tests
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Mark tests with "slow" in the name as slow tests
        elif "slow" in item.name:
            item.add_marker(pytest.mark.slow) 