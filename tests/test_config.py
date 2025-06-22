"""
Tests for configuration system.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch

from agentlabs.core.config import (
    Config, Settings, LLMConfig, DatabaseConfig, RedisConfig,
    VectorStoreConfig, LoggingConfig, ToolConfig
)


class TestLLMConfig:
    """Test LLM configuration."""
    
    def test_llm_config_defaults(self):
        """Test LLM config default values."""
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.timeout == 60
        assert config.retry_attempts == 3
    
    def test_llm_config_custom_values(self):
        """Test LLM config with custom values."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet",
            api_key="test-key",
            temperature=0.5,
            max_tokens=2000,
            timeout=120,
            retry_attempts=5
        )
        assert config.provider == "anthropic"
        assert config.model == "claude-3-sonnet"
        assert config.api_key == "test-key"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.timeout == 120
        assert config.retry_attempts == 5
    
    def test_llm_config_validation(self):
        """Test LLM config validation."""
        # Valid provider
        config = LLMConfig(provider="openai")
        assert config.provider == "openai"
        
        # Invalid provider
        with pytest.raises(ValueError, match="Provider must be one of"):
            LLMConfig(provider="invalid_provider")
        
        # Valid temperature range
        config = LLMConfig(temperature=0.0)
        assert config.temperature == 0.0
        
        config = LLMConfig(temperature=2.0)
        assert config.temperature == 2.0
        
        # Invalid temperature
        with pytest.raises(ValueError):
            LLMConfig(temperature=3.0)
        
        with pytest.raises(ValueError):
            LLMConfig(temperature=-1.0)


class TestDatabaseConfig:
    """Test database configuration."""
    
    def test_database_config_defaults(self):
        """Test database config default values."""
        config = DatabaseConfig()
        assert config.url == "sqlite:///agentlabs.db"
        assert config.echo is False
        assert config.pool_size == 5
        assert config.max_overflow == 10
    
    def test_database_config_custom_values(self):
        """Test database config with custom values."""
        config = DatabaseConfig(
            url="postgresql://user:pass@localhost/db",
            echo=True,
            pool_size=10,
            max_overflow=20
        )
        assert config.url == "postgresql://user:pass@localhost/db"
        assert config.echo is True
        assert config.pool_size == 10
        assert config.max_overflow == 20


class TestRedisConfig:
    """Test Redis configuration."""
    
    def test_redis_config_defaults(self):
        """Test Redis config default values."""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.ssl is False
    
    def test_redis_config_custom_values(self):
        """Test Redis config with custom values."""
        config = RedisConfig(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
            ssl=True
        )
        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.db == 1
        assert config.password == "secret"
        assert config.ssl is True


class TestVectorStoreConfig:
    """Test vector store configuration."""
    
    def test_vector_store_config_defaults(self):
        """Test vector store config default values."""
        config = VectorStoreConfig()
        assert config.type == "chroma"
        assert config.path == "./vectorstore"
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.dimension == 384
    
    def test_vector_store_config_custom_values(self):
        """Test vector store config with custom values."""
        config = VectorStoreConfig(
            type="faiss",
            path="/path/to/vectors",
            embedding_model="all-mpnet-base-v2",
            dimension=768
        )
        assert config.type == "faiss"
        assert config.path == "/path/to/vectors"
        assert config.embedding_model == "all-mpnet-base-v2"
        assert config.dimension == 768


class TestLoggingConfig:
    """Test logging configuration."""
    
    def test_logging_config_defaults(self):
        """Test logging config default values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert "time" in config.format
        assert config.file_path is None
        assert config.rotation == "10 MB"
        assert config.retention == "30 days"
        assert config.compression == "gz"
    
    def test_logging_config_custom_values(self):
        """Test logging config with custom values."""
        config = LoggingConfig(
            level="DEBUG",
            format="Custom format",
            file_path="/path/to/logs/app.log",
            rotation="100 MB",
            retention="7 days",
            compression="zip"
        )
        assert config.level == "DEBUG"
        assert config.format == "Custom format"
        assert config.file_path == "/path/to/logs/app.log"
        assert config.rotation == "100 MB"
        assert config.retention == "7 days"
        assert config.compression == "zip"


class TestToolConfig:
    """Test tool configuration."""
    
    def test_tool_config_defaults(self):
        """Test tool config default values."""
        config = ToolConfig()
        assert config.web_search_enabled is True
        assert config.web_search_api_key is None
        assert config.selenium_enabled is True
        assert config.selenium_headless is True
        assert config.data_analysis_enabled is True
    
    def test_tool_config_custom_values(self):
        """Test tool config with custom values."""
        config = ToolConfig(
            web_search_enabled=False,
            web_search_api_key="search-key",
            selenium_enabled=False,
            selenium_headless=False,
            data_analysis_enabled=False
        )
        assert config.web_search_enabled is False
        assert config.web_search_api_key == "search-key"
        assert config.selenium_enabled is False
        assert config.selenium_headless is False
        assert config.data_analysis_enabled is False


class TestSettings:
    """Test main settings."""
    
    def test_settings_defaults(self):
        """Test settings default values."""
        settings = Settings()
        assert settings.app_name == "AgentLabs"
        assert settings.debug is False
        assert settings.data_dir == "./data"
        assert isinstance(settings.llm, LLMConfig)
        assert isinstance(settings.database, DatabaseConfig)
        assert isinstance(settings.redis, RedisConfig)
        assert isinstance(settings.vector_store, VectorStoreConfig)
        assert isinstance(settings.logging, LoggingConfig)
        assert isinstance(settings.tools, ToolConfig)
    
    def test_settings_custom_values(self, tmp_path):
        """Test settings with custom values."""
        custom_data_dir = tmp_path / "custom_data"
        settings = Settings(
            app_name="CustomApp",
            debug=True,
            data_dir=str(custom_data_dir)
        )
        assert settings.app_name == "CustomApp"
        assert settings.debug is True
        assert settings.data_dir == str(custom_data_dir)
    
    def test_settings_data_dir_creation(self, tmp_path):
        """Test that data directory is created."""
        data_dir = tmp_path / "test_data"
        settings = Settings(data_dir=str(data_dir))
        assert data_dir.exists()
        assert data_dir.is_dir()


class TestConfig:
    """Test configuration manager."""
    
    def test_config_init_no_file(self):
        """Test config initialization without file."""
        config = Config()
        assert isinstance(config.settings, Settings)
        assert config.config_path is None
    
    def test_config_init_with_file(self, tmp_path):
        """Test config initialization with file."""
        config_file = tmp_path / "test_config.json"
        config_data = {
            "app_name": "TestApp",
            "debug": True,
            "llm": {
                "provider": "anthropic",
                "model": "claude-3-sonnet"
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        config = Config(str(config_file))
        assert config.settings.app_name == "TestApp"
        assert config.settings.debug is True
        assert config.settings.llm.provider == "anthropic"
        assert config.settings.llm.model == "claude-3-sonnet"
    
    def test_config_save(self, tmp_path):
        """Test config saving."""
        config = Config()
        config.settings.app_name = "SavedApp"
        
        save_path = tmp_path / "saved_config.json"
        config.save_config(str(save_path))
        
        assert save_path.exists()
        
        # Load and verify
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["app_name"] == "SavedApp"
    
    def test_config_update(self):
        """Test config updating."""
        config = Config()
        original_name = config.settings.app_name
        
        updates = {
            "app_name": "UpdatedApp",
            "debug": True
        }
        config.update_config(updates)
        
        assert config.settings.app_name == "UpdatedApp"
        assert config.settings.debug is True
        assert config.settings.app_name != original_name
    
    def test_config_getters(self):
        """Test config getter methods."""
        config = Config()
        
        llm_config = config.get_llm_config()
        assert isinstance(llm_config, LLMConfig)
        
        db_config = config.get_database_config()
        assert isinstance(db_config, DatabaseConfig)
        
        redis_config = config.get_redis_config()
        assert isinstance(redis_config, RedisConfig)
        
        vector_config = config.get_vector_store_config()
        assert isinstance(vector_config, VectorStoreConfig)
        
        logging_config = config.get_logging_config()
        assert isinstance(logging_config, LoggingConfig)
        
        tool_config = config.get_tool_config()
        assert isinstance(tool_config, ToolConfig)
    
    def test_config_environment_variables(self):
        """Test config loading from environment variables."""
        with patch.dict('os.environ', {
            'AGENTLABS_APP_NAME': 'EnvApp',
            'AGENTLABS_DEBUG': 'true',
            'AGENTLABS_LLM__PROVIDER': 'anthropic',
            'AGENTLABS_LLM__MODEL': 'claude-3-sonnet'
        }):
            config = Config()
            assert config.settings.app_name == "EnvApp"
            assert config.settings.debug is True
            assert config.settings.llm.provider == "anthropic"
            assert config.settings.llm.model == "claude-3-sonnet" 