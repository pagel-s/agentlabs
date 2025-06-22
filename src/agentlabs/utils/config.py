"""Configuration management for AgentLabs framework."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """Configuration for LLM providers."""
    
    model_config = SettingsConfigDict(env_prefix="LLM_")
    
    provider: str = Field(default="openai", description="LLM provider (openai, anthropic, local)")
    model: str = Field(default="gpt-4", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key")
    base_url: Optional[str] = Field(default=None, description="Base URL for API")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=4096, gt=0, description="Maximum tokens to generate")
    timeout: int = Field(default=60, gt=0, description="Request timeout in seconds")
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate LLM provider."""
        valid_providers = ["openai", "anthropic", "local", "ollama"]
        if v.lower() not in valid_providers:
            raise ValueError(f"Provider must be one of {valid_providers}")
        return v.lower()


class DatabaseConfig(BaseSettings):
    """Configuration for database connections."""
    
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    url: str = Field(default="sqlite:///agentlabs.db", description="Database URL")
    echo: bool = Field(default=False, description="Enable SQL echo")
    pool_size: int = Field(default=5, gt=0, description="Connection pool size")
    max_overflow: int = Field(default=10, gt=0, description="Max overflow connections")


class RedisConfig(BaseSettings):
    """Configuration for Redis cache."""
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    url: str = Field(default="redis://localhost:6379", description="Redis URL")
    db: int = Field(default=0, ge=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    timeout: int = Field(default=5, gt=0, description="Connection timeout")


class VectorStoreConfig(BaseSettings):
    """Configuration for vector stores."""
    
    model_config = SettingsConfigDict(env_prefix="VECTOR_")
    
    provider: str = Field(default="chroma", description="Vector store provider")
    path: str = Field(default="./vectorstore", description="Vector store path")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    dimension: int = Field(default=384, gt=0, description="Embedding dimension")


class LoggingConfig(BaseSettings):
    """Configuration for logging."""
    
    model_config = SettingsConfigDict(env_prefix="LOG_")
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="{time} | {level} | {name}:{function}:{line} | {message}", description="Log format")
    file: Optional[str] = Field(default=None, description="Log file path")
    rotation: str = Field(default="10 MB", description="Log rotation size")
    retention: str = Field(default="30 days", description="Log retention period")


class Settings(BaseSettings):
    """Main settings for AgentLabs framework."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Core settings
    app_name: str = Field(default="AgentLabs", description="Application name")
    debug: bool = Field(default=False, description="Debug mode")
    data_dir: Path = Field(default=Path("./data"), description="Data directory")
    cache_dir: Path = Field(default=Path("./cache"), description="Cache directory")
    output_dir: Path = Field(default=Path("./output"), description="Output directory")
    
    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")
    redis: RedisConfig = Field(default_factory=RedisConfig, description="Redis configuration")
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig, description="Vector store configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    
    # Agent settings
    max_concurrent_agents: int = Field(default=10, gt=0, description="Maximum concurrent agents")
    agent_timeout: int = Field(default=300, gt=0, description="Agent execution timeout in seconds")
    retry_attempts: int = Field(default=3, ge=0, description="Number of retry attempts")
    
    # Research settings
    default_research_timeout: int = Field(default=3600, gt=0, description="Default research timeout")
    max_iterations: int = Field(default=100, gt=0, description="Maximum research iterations")
    save_intermediate_results: bool = Field(default=True, description="Save intermediate results")
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize settings and create directories."""
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.logging.file:
            log_file = Path(self.logging.file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if self.vectorstore.path:
            vector_path = Path(self.vectorstore.path)
            vector_path.mkdir(parents=True, exist_ok=True)
    
    def get_database_url(self) -> str:
        """Get database URL with proper formatting."""
        if self.database.url.startswith("sqlite"):
            return self.database.url
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Get Redis URL with proper formatting."""
        return self.redis.url
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls()
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "Settings":
        """Create settings from file."""
        return cls(_env_file=str(file_path))


# Global settings instance
settings = Settings() 