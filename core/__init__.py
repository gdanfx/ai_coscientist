"""
AI Co-Scientist Core Module

This package contains the core infrastructure for the AI Co-Scientist system,
including configuration management and shared data structures.
"""

from .config import (
    Config,
    DatabaseConfig,
    ModelConfig,
    AgentConfig,
    SystemConfig,
    get_config,
    reload_config,
    PUBMED_EMAIL,
    RATE_LIMIT_DELAY,
    MAX_RESULTS_PER_SOURCE,
    TOTAL_MAX_RESULTS,
    WEB_SEARCH_DELAY
)

__all__ = [
    'Config',
    'DatabaseConfig', 
    'ModelConfig',
    'AgentConfig',
    'SystemConfig',
    'get_config',
    'reload_config',
    'PUBMED_EMAIL',
    'RATE_LIMIT_DELAY',
    'MAX_RESULTS_PER_SOURCE',
    'TOTAL_MAX_RESULTS',
    'WEB_SEARCH_DELAY'
]