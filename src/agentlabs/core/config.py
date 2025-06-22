"""
Configuration management for AgentLabs framework.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import json
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: str = Field(default="openai", description="LLM provider (openai, anthropic, local)")
    model: str = Field(default="gpt-4", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for API calls")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=4096, gt=0, description="Maximum tokens to generate")
    timeout: int = Field(default=60, gt=0, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, ge=0, description="Number of retry attempts")
    
    @validator('provider')
    def validate_provider(cls, v):
        valid_providers = ['openai', 'anthropic', 'local']
        if v not in valid_providers:
            raise ValueError(f'Provider must be one of {valid_providers}')
        return v


class DatabaseConfig(BaseModel):
    """Configuration for database connections."""
    url: str = Field(default="sqlite:///agentlabs.db", description="Database URL")
    echo: bool = Field(default=False, description="Enable SQL echo")
    pool_size: int = Field(default=5, gt=0, description="Connection pool size")
    max_overflow: int = Field(default=10, ge=0, description="Maximum overflow connections")


class RedisConfig(BaseModel):
    """Configuration for Redis connections."""
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, gt=0, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    ssl: bool = Field(default=False, description="Enable SSL connection")


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    type: str = Field(default="chroma", description="Vector store type (chroma, faiss)")
    path: str = Field(default="./vectorstore", description="Path to vector store")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    dimension: int = Field(default=384, gt=0, description="Embedding dimension")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}", description="Log format")
    file_path: Optional[str] = Field(default=None, description="Log file path")
    rotation: str = Field(default="10 MB", description="Log rotation size")
    retention: str = Field(default="30 days", description="Log retention period")
    compression: str = Field(default="gz", description="Log compression format")


class ToolConfig(BaseModel):
    """Configuration for tools."""
    web_search_enabled: bool = Field(default=True, description="Enable web search tool")
    web_search_api_key: Optional[str] = Field(default=None, description="Web search API key")
    selenium_enabled: bool = Field(default=True, description="Enable Selenium for web scraping")
    selenium_headless: bool = Field(default=True, description="Run Selenium in headless mode")
    data_analysis_enabled: bool = Field(default=True, description="Enable data analysis tools")


class Settings(BaseSettings):
    """Main settings class for AgentLabs."""
    
    # Core settings
    app_name: str = Field(default="AgentLabs", description="Application name")
    debug: bool = Field(default=False, description="Enable debug mode")
    data_dir: str = Field(default="./data", description="Data directory path")
    
    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")
    redis: RedisConfig = Field(default_factory=RedisConfig, description="Redis configuration")
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig, description="Vector store configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    tools: ToolConfig = Field(default_factory=ToolConfig, description="Tool configuration")
    
    # Agent settings
    default_agent_timeout: int = Field(default=300, gt=0, description="Default agent execution timeout")
    max_concurrent_agents: int = Field(default=5, gt=0, description="Maximum concurrent agents")
    
    # Project settings
    max_project_tasks: int = Field(default=100, gt=0, description="Maximum tasks per project")
    project_timeout: int = Field(default=3600, gt=0, description="Project execution timeout")
    
    # Workflow settings
    max_workflow_steps: int = Field(default=50, gt=0, description="Maximum workflow steps")
    workflow_timeout: int = Field(default=7200, gt=0, description="Workflow execution timeout")
    
    class Config:
        env_prefix = "AGENTLABS_"
        env_nested_delimiter = "__"
        case_sensitive = False
    
    @validator('data_dir')
    def create_data_dir(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class Config:
    """Configuration manager for AgentLabs."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.settings = self._load_settings()
    
    def _load_settings(self) -> Settings:
        """Load settings from file and environment."""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            return Settings(**config_data)
        return Settings()
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        save_path = path or self.config_path or "agentlabs_config.json"
        config_dict = self.settings.dict()
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        current_dict = self.settings.dict()
        current_dict.update(updates)
        self.settings = Settings(**current_dict)
    
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return self.settings.llm
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.settings.database
    
    def get_redis_config(self) -> RedisConfig:
        """Get Redis configuration."""
        return self.settings.redis
    
    def get_vector_store_config(self) -> VectorStoreConfig:
        """Get vector store configuration."""
        return self.settings.vector_store
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.settings.logging
    
    def get_tool_config(self) -> ToolConfig:
        """Get tool configuration."""
        return self.settings.tools 