"""LLM provider system for AgentLabs framework."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import openai
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field

from ..utils.config import LLMConfig
from ..utils.logging import LoggerMixin, LoggedClass, log_async_function_call, log_async_execution_time


@dataclass
class LLMMessage:
    """Represents a message in a conversation."""
    role: str
    content: str
    name: Optional[str] = None


@dataclass
class LLMResponse:
    """Represents a response from an LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider(ABC, LoggedClass):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        """Initialize LLM provider."""
        super().__init__()
        self.config = config
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.timeout = config.timeout
        self.retry_attempts = config.retry_attempts
        self.log_info(f"Initialized {self.__class__.__name__} with model {config.model}")
    
    @abstractmethod
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_stream(self, messages: List[LLMMessage], **kwargs) -> Any:
        """Generate a streaming response from the LLM."""
        pass
    
    def _validate_messages(self, messages: List[LLMMessage]) -> None:
        """Validate message format."""
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        for msg in messages:
            if not isinstance(msg, LLMMessage):
                raise ValueError("All messages must be LLMMessage instances")
            if not msg.role or not msg.content:
                raise ValueError("Message role and content cannot be empty")


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""
    
    def __init__(self, config: LLMConfig):
        """Initialize OpenAI provider."""
        super().__init__(config)
        
        if not config.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )
        
        self.langchain_model = ChatOpenAI(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=config.api_key,
            openai_api_base=config.base_url,
        )
    
    @log_async_function_call
    @log_async_execution_time
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate a response using OpenAI API."""
        self._validate_messages(messages)
        
        # Convert messages to OpenAI format
        openai_messages = [
            {
                "role": msg.role,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {})
            }
            for msg in messages
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                metadata={"id": response.id}
            )
        
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def generate_stream(self, messages: List[LLMMessage], **kwargs) -> Any:
        """Generate a streaming response using OpenAI API."""
        self._validate_messages(messages)
        
        openai_messages = [
            {
                "role": msg.role,
                "content": msg.content,
                **({"name": msg.name} if msg.name else {})
            }
            for msg in messages
        ]
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            self.logger.error(f"OpenAI streaming error: {str(e)}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider."""
    
    def __init__(self, config: LLMConfig):
        """Initialize Anthropic provider."""
        super().__init__(config)
        
        if not config.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.client = anthropic.AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        self.langchain_model = ChatAnthropic(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            anthropic_api_key=config.api_key,
        )
    
    @log_async_function_call
    @log_async_execution_time
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate a response using Anthropic API."""
        self._validate_messages(messages)
        
        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                system=system_message,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **kwargs
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                } if response.usage else None,
                finish_reason=response.stop_reason,
                metadata={"id": response.id}
            )
        
        except Exception as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    async def generate_stream(self, messages: List[LLMMessage], **kwargs) -> Any:
        """Generate a streaming response using Anthropic API."""
        self._validate_messages(messages)
        
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        try:
            stream = await self.client.messages.create(
                model=self.model,
                messages=anthropic_messages,
                system=system_message,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.type == "content_block_delta":
                    yield chunk.delta.text
        
        except Exception as e:
            self.logger.error(f"Anthropic streaming error: {str(e)}")
            raise


class LocalProvider(LLMProvider):
    """Local LLM provider (placeholder for future implementation)."""
    
    def __init__(self, config: LLMConfig):
        """Initialize local provider."""
        super().__init__(config)
        self.logger.warning("Local provider is not yet implemented")
    
    async def generate(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate a response using local model (placeholder)."""
        self._validate_messages(messages)
        raise NotImplementedError("Local provider is not yet implemented")
    
    async def generate_stream(self, messages: List[LLMMessage], **kwargs) -> Any:
        """Generate a streaming response using local model (placeholder)."""
        self._validate_messages(messages)
        raise NotImplementedError("Local provider is not yet implemented")


class LLMFactory:
    """Factory for creating LLM providers."""
    
    _providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "local": LocalProvider
    }
    
    @classmethod
    def create_provider(cls, config: LLMConfig) -> LLMProvider:
        """
        Create an LLM provider based on configuration.
        
        Args:
            config: LLM configuration
            
        Returns:
            LLM provider instance
            
        Raises:
            ValueError: If provider type is not supported
        """
        provider_class = cls._providers.get(config.provider.lower())
        if not provider_class:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        return provider_class(config)
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """
        Register a custom provider.
        
        Args:
            name: Provider name
            provider_class: Provider class
        """
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return list(cls._providers.keys()) 