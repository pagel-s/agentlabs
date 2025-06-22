"""
Configuration management for AgentLabs framework.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
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
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    timeout: int = Field(default=60, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    
    @validator('api_key', pre=True, always=True)
    def validate_api_key(cls, v, values):
        if values.get('provider') in ['openai', 'anthropic'] and not v:
            raise ValueError(f"API key is required for {values.get('provider')} provider")
        return v


class DatabaseConfig(BaseModel):
    """Configuration for database connections."""
    
    url: str = Field(default="sqlite:///agentlabs.db", description="Database URL")
    echo: bool = Field(default=False, description="Enable SQL echo")
    pool_size: int = Field(default=5, gt=0, description="Connection pool size")
    max_overflow: int = Field(default=10, ge=0, description="Maximum overflow connections")
    pool_timeout: int = Field(default=30, gt=0, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, gt=0, description="Pool recycle time in seconds")


class RedisConfig(BaseModel):
    """Configuration for Redis connections."""
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, gt=0, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    ssl: bool = Field(default=False, description="Enable SSL connection")
    timeout: int = Field(default=5, gt=0, description="Connection timeout in seconds")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    decode_responses: bool = Field(default=True, description="Decode responses to strings")


class VectorStoreConfig(BaseModel):
    """Configuration for vector store connections."""
    
    type: str = Field(default="chroma", description="Vector store type (chroma, faiss, pinecone)")
    path: Optional[str] = Field(default=None, description="Path for local vector stores")
    collection_name: str = Field(default="agentlabs", description="Collection name")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    dimension: int = Field(default=384, gt=0, description="Vector dimension")
    distance_metric: str = Field(default="cosine", description="Distance metric")
    
    # Pinecone specific
    api_key: Optional[str] = Field(default=None, description="Pinecone API key")
    environment: Optional[str] = Field(default=None, description="Pinecone environment")
    
    # Chroma specific
    persist_directory: Optional[str] = Field(default="./chroma_db", description="Chroma persist directory")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Log format string"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_size: str = Field(default="10 MB", description="Maximum log file size")
    rotation: str = Field(default="1 day", description="Log rotation interval")
    retention: str = Field(default="30 days", description="Log retention period")
    compression: str = Field(default="gz", description="Log compression format")


class Config(BaseSettings):
    """Main configuration class for AgentLabs."""
    
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
    
    # Agent settings
    default_agent_timeout: int = Field(default=300, gt=0, description="Default agent execution timeout")
    default_agent_retries: int = Field(default=3, ge=0, description="Default agent retry attempts")
    max_concurrent_agents: int = Field(default=10, gt=0, description="Maximum concurrent agents")
    
    # Tool settings
    tool_timeout: int = Field(default=60, gt=0, description="Default tool execution timeout")
    web_search_enabled: bool = Field(default=True, description="Enable web search tools")
    browser_enabled: bool = Field(default=True, description="Enable browser automation tools")
    
    # Project settings
    max_project_tasks: int = Field(default=100, gt=0, description="Maximum tasks per project")
    project_timeout: int = Field(default=3600, gt=0, description="Project execution timeout")
    
    # Workflow settings
    max_workflow_steps: int = Field(default=50, gt=0, description="Maximum steps per workflow")
    workflow_timeout: int = Field(default=7200, gt=0, description="Workflow execution timeout")
    
    class Config:
        env_prefix = "AGENTLABS_"
        env_nested_delimiter = "__"
        case_sensitive = False
        
    @validator('data_dir')
    def create_data_dir(cls, v):
        """Create data directory if it doesn't exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return self.llm
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.database
    
    def get_redis_config(self) -> RedisConfig:
        """Get Redis configuration."""
        return self.redis
    
    def get_vector_store_config(self) -> VectorStoreConfig:
        """Get vector store configuration."""
        return self.vector_store
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.logging
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls()
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Config':
        """Create configuration from file."""
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to file."""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _config
    _config = config


def load_config(file_path: Optional[str] = None) -> Config:
    """Load configuration from file or environment."""
    if file_path and Path(file_path).exists():
        config = Config.from_file(file_path)
    else:
        config = Config.from_env()
    
    set_config(config)
    return config 