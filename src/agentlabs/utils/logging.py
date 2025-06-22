"""
Logging utilities for AgentLabs framework.
"""

import sys
import functools
import time
from typing import Optional, Callable, Any
from pathlib import Path
from loguru import logger
from .config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """
    Setup logging configuration using loguru.
    
    Args:
        config: Logging configuration
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format=config.format,
        level=config.level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler if specified
    if config.file_path:
        log_path = Path(config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            config.file_path,
            format=config.format,
            level=config.level,
            rotation=config.rotation,
            retention=config.retention,
            compression=config.compression,
            backtrace=True,
            diagnose=True
        )


def get_logger(name: str) -> "logger":
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self):
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


def log_function_call(func: Callable) -> Callable:
    """
    Decorator to log function calls with parameters and return values.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger_name = get_logger(func.__module__)
        
        # Log function call
        logger_name.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger_name.info(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger_name.error(f"Function {func.__name__} failed with error: {str(e)}")
            raise
    
    return wrapper


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger_name = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger_name.info(f"Function {func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger_name.error(f"Function {func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper


class LoggedClass:
    """Base class that provides logging capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(self.__class__.__name__)
    
    @property
    def logger(self):
        """Get logger for this class."""
        return self._logger


def log_async_function_call(func: Callable) -> Callable:
    """
    Decorator to log async function calls with parameters and return values.
    
    Args:
        func: Async function to decorate
        
    Returns:
        Decorated async function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger_name = get_logger(func.__module__)
        
        # Log function call
        logger_name.info(f"Calling async {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = await func(*args, **kwargs)
            logger_name.info(f"Async function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger_name.error(f"Async function {func.__name__} failed with error: {str(e)}")
            raise
    
    return wrapper


def log_async_execution_time(func: Callable) -> Callable:
    """
    Decorator to log async function execution time.
    
    Args:
        func: Async function to decorate
        
    Returns:
        Decorated async function
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger_name = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger_name.info(f"Async function {func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger_name.error(f"Async function {func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper 