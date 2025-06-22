"""Memory system for AgentLabs framework."""

import asyncio
import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import hashlib
import sqlite3
import redis.asyncio as redis
from pydantic import BaseModel

from ..utils.logging import LoggerMixin, LoggedClass, log_async_function_call, log_async_execution_time


@dataclass
class MemoryItem:
    """A single memory item."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if the memory item has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


@dataclass
class ContextData:
    """Context data for a session."""
    session_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class Memory(ABC, LoggedClass):
    """Abstract base class for memory implementations."""
    
    def __init__(self, max_size: Optional[int] = None, ttl: Optional[int] = None):
        super().__init__()
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from memory."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in memory."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from memory."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists in memory."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all memory."""
        pass
    
    @abstractmethod
    async def keys(self) -> List[str]:
        """Get all keys in memory."""
        pass
    
    async def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get context data for a session."""
        return await self.get(f"context:{session_id}")
    
    async def save_context(self, session_id: str, context_data: Dict[str, Any]) -> bool:
        """Save context data for a session."""
        return await self.set(f"context:{session_id}", context_data)
    
    async def delete_context(self, session_id: str) -> bool:
        """Delete context data for a session."""
        return await self.delete(f"context:{session_id}")


class InMemoryMemory(Memory):
    """In-memory memory implementation."""
    
    def __init__(self, max_size: Optional[int] = None, ttl: Optional[int] = None):
        super().__init__(max_size, ttl)
        self._storage: Dict[str, MemoryItem] = {}
        self._lock = asyncio.Lock()
    
    @log_async_function_call
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from memory."""
        async with self._lock:
            if key not in self._storage:
                return None
            
            item = self._storage[key]
            if item.is_expired():
                del self._storage[key]
                return None
            
            return item.value
    
    @log_async_function_call
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in memory."""
        async with self._lock:
            # Check max size
            if self.max_size and len(self._storage) >= self.max_size:
                # Remove oldest item
                oldest_key = min(self._storage.keys(), key=lambda k: self._storage[k].created_at)
                del self._storage[oldest_key]
            
            # Calculate expiration
            expires_at = None
            if ttl is not None:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            elif self.ttl is not None:
                expires_at = datetime.utcnow() + timedelta(seconds=self.ttl)
            
            # Create memory item
            item = MemoryItem(
                key=key,
                value=value,
                expires_at=expires_at
            )
            
            self._storage[key] = item
            return True
    
    @log_async_function_call
    async def delete(self, key: str) -> bool:
        """Delete a value from memory."""
        async with self._lock:
            if key in self._storage:
                del self._storage[key]
                return True
            return False
    
    @log_async_function_call
    async def exists(self, key: str) -> bool:
        """Check if a key exists in memory."""
        async with self._lock:
            if key not in self._storage:
                return False
            
            item = self._storage[key]
            if item.is_expired():
                del self._storage[key]
                return False
            
            return True
    
    @log_async_function_call
    async def clear(self) -> bool:
        """Clear all memory."""
        async with self._lock:
            self._storage.clear()
            return True
    
    @log_async_function_call
    async def keys(self) -> List[str]:
        """Get all keys in memory."""
        async with self._lock:
            # Remove expired items
            expired_keys = [
                key for key, item in self._storage.items()
                if item.is_expired()
            ]
            for key in expired_keys:
                del self._storage[key]
            
            return list(self._storage.keys())


class FileMemory(Memory):
    """File-based memory implementation."""
    
    def __init__(self, file_path: str, max_size: Optional[int] = None, ttl: Optional[int] = None):
        super().__init__(max_size, ttl)
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
    
    def _load_data(self) -> Dict[str, MemoryItem]:
        """Load data from file."""
        if not self.file_path.exists():
            return {}
        
        try:
            with open(self.file_path, 'rb') as f:
                data = pickle.load(f)
                # Convert dict items back to MemoryItem objects
                return {k: MemoryItem(**v) if isinstance(v, dict) else v for k, v in data.items()}
        except Exception as e:
            self.logger.error(f"Error loading memory file: {str(e)}")
            return {}
    
    def _save_data(self, data: Dict[str, MemoryItem]) -> None:
        """Save data to file."""
        try:
            # Convert MemoryItem objects to dict for serialization
            serializable_data = {}
            for k, v in data.items():
                if isinstance(v, MemoryItem):
                    serializable_data[k] = {
                        'key': v.key,
                        'value': v.value,
                        'created_at': v.created_at,
                        'expires_at': v.expires_at,
                        'metadata': v.metadata
                    }
                else:
                    serializable_data[k] = v
            
            with open(self.file_path, 'wb') as f:
                pickle.dump(serializable_data, f)
        except Exception as e:
            self.logger.error(f"Error saving memory file: {str(e)}")
    
    @log_async_function_call
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from memory."""
        async with self._lock:
            data = self._load_data()
            
            if key not in data:
                return None
            
            item = data[key]
            if item.is_expired():
                del data[key]
                self._save_data(data)
                return None
            
            return item.value
    
    @log_async_function_call
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in memory."""
        async with self._lock:
            data = self._load_data()
            
            # Check max size
            if self.max_size and len(data) >= self.max_size:
                # Remove oldest item
                oldest_key = min(data.keys(), key=lambda k: data[k].created_at)
                del data[oldest_key]
            
            # Calculate expiration
            expires_at = None
            if ttl is not None:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            elif self.ttl is not None:
                expires_at = datetime.utcnow() + timedelta(seconds=self.ttl)
            
            # Create memory item
            item = MemoryItem(
                key=key,
                value=value,
                expires_at=expires_at
            )
            
            data[key] = item
            self._save_data(data)
            return True
    
    @log_async_function_call
    async def delete(self, key: str) -> bool:
        """Delete a value from memory."""
        async with self._lock:
            data = self._load_data()
            
            if key in data:
                del data[key]
                self._save_data(data)
                return True
            return False
    
    @log_async_function_call
    async def exists(self, key: str) -> bool:
        """Check if a key exists in memory."""
        async with self._lock:
            data = self._load_data()
            
            if key not in data:
                return False
            
            item = data[key]
            if item.is_expired():
                del data[key]
                self._save_data(data)
                return False
            
            return True
    
    @log_async_function_call
    async def clear(self) -> bool:
        """Clear all memory."""
        async with self._lock:
            self._save_data({})
            return True
    
    @log_async_function_call
    async def keys(self) -> List[str]:
        """Get all keys in memory."""
        async with self._lock:
            data = self._load_data()
            
            # Remove expired items
            expired_keys = [
                key for key, item in data.items()
                if item.is_expired()
            ]
            for key in expired_keys:
                del data[key]
            
            if expired_keys:
                self._save_data(data)
            
            return list(data.keys())


class RedisMemory(Memory):
    """Redis-based memory implementation."""
    
    def __init__(self, redis_url: str, max_size: Optional[int] = None, ttl: Optional[int] = None):
        super().__init__(max_size, ttl)
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
    
    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url)
        return self._redis
    
    @log_async_function_call
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from memory."""
        try:
            redis_client = await self._get_redis()
            value = await redis_client.get(key)
            if value is None:
                return None
            
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value.decode('utf-8')
        
        except Exception as e:
            self.logger.error(f"Redis get error: {str(e)}")
            return None
    
    @log_async_function_call
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in memory."""
        try:
            redis_client = await self._get_redis()
            
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            # Set expiration
            expiration = ttl or self.ttl
            
            if expiration:
                await redis_client.setex(key, expiration, serialized_value)
            else:
                await redis_client.set(key, serialized_value)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Redis set error: {str(e)}")
            return False
    
    @log_async_function_call
    async def delete(self, key: str) -> bool:
        """Delete a value from memory."""
        try:
            redis_client = await self._get_redis()
            result = await redis_client.delete(key)
            return result > 0
        
        except Exception as e:
            self.logger.error(f"Redis delete error: {str(e)}")
            return False
    
    @log_async_function_call
    async def exists(self, key: str) -> bool:
        """Check if a key exists in memory."""
        try:
            redis_client = await self._get_redis()
            return await redis_client.exists(key) > 0
        
        except Exception as e:
            self.logger.error(f"Redis exists error: {str(e)}")
            return False
    
    @log_async_function_call
    async def clear(self) -> bool:
        """Clear all memory."""
        try:
            redis_client = await self._get_redis()
            await redis_client.flushdb()
            return True
        
        except Exception as e:
            self.logger.error(f"Redis clear error: {str(e)}")
            return False
    
    @log_async_function_call
    async def keys(self) -> List[str]:
        """Get all keys in memory."""
        try:
            redis_client = await self._get_redis()
            keys = await redis_client.keys('*')
            return [key.decode('utf-8') for key in keys]
        
        except Exception as e:
            self.logger.error(f"Redis keys error: {str(e)}")
            return []
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


class MemoryManager:
    """Manager for multiple memory instances."""
    
    def __init__(self):
        self.memories: Dict[str, Memory] = {}
        self.logger = LoggedClass().logger
    
    def register_memory(self, name: str, memory: Memory) -> None:
        """Register a memory instance."""
        self.memories[name] = memory
        self.logger.info(f"Registered memory: {name}")
    
    def get_memory(self, name: str) -> Optional[Memory]:
        """Get a memory instance by name."""
        return self.memories.get(name)
    
    def list_memories(self) -> List[str]:
        """List all registered memories."""
        return list(self.memories.keys())
    
    def remove_memory(self, name: str) -> bool:
        """Remove a memory instance."""
        if name in self.memories:
            del self.memories[name]
            self.logger.info(f"Removed memory: {name}")
            return True
        return False
    
    async def clear_all(self) -> None:
        """Clear all memories."""
        for memory in self.memories.values():
            await memory.clear()
        self.logger.info("Cleared all memories")
    
    async def close(self) -> None:
        """Close all memory connections."""
        for memory in self.memories.values():
            if hasattr(memory, 'close'):
                await memory.close()
        self.logger.info("Closed all memory connections")


# Memory factory
def create_memory(memory_type: str, **kwargs) -> Memory:
    """Create a memory instance based on type."""
    if memory_type == "in_memory":
        return InMemoryMemory(**kwargs)
    elif memory_type == "file":
        return FileMemory(**kwargs)
    elif memory_type == "redis":
        return RedisMemory(**kwargs)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")


# Global memory store instance
memory_store = create_memory("in_memory") 