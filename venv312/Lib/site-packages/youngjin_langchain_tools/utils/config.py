# youngjin_langchain_tools/utils/config.py
"""
Configuration management for youngjin-langchain-tools.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel


class LibraryConfig(BaseModel):
    """Global configuration for the library."""
    
    verbose: bool = False
    cache_enabled: bool = True
    timeout: int = 30
    
    class Config:
        extra = "allow"


# Global configuration instance
_config: LibraryConfig = LibraryConfig()


def configure(
    verbose: Optional[bool] = None,
    cache_enabled: Optional[bool] = None,
    timeout: Optional[int] = None,
    **kwargs: Any,
) -> LibraryConfig:
    """
    Configure global library settings.
    
    Args:
        verbose: Enable verbose output.
        cache_enabled: Enable caching.
        timeout: Default timeout in seconds.
        **kwargs: Additional configuration options.
        
    Returns:
        The updated configuration object.
        
    Example:
        >>> configure(verbose=True, cache_enabled=False)
        >>> configure(custom_option="value")
    """
    global _config
    
    updates: Dict[str, Any] = {}
    
    if verbose is not None:
        updates["verbose"] = verbose
    if cache_enabled is not None:
        updates["cache_enabled"] = cache_enabled
    if timeout is not None:
        updates["timeout"] = timeout
    
    updates.update(kwargs)
    
    _config = LibraryConfig(**{**_config.model_dump(), **updates})
    return _config


def get_config() -> LibraryConfig:
    """
    Get the current global configuration.
    
    Returns:
        The current configuration object.
    """
    return _config
