"""
AI Co-Scientist Utilities Module

This package contains shared utility functions and classes that are used
across multiple agents in the AI Co-Scientist system.
"""

# Import with error handling for optional dependencies
try:
    from .llm_client import LLMClient, create_llm_client
    LLM_CLIENT_AVAILABLE = True
except ImportError:
    LLM_CLIENT_AVAILABLE = False

try:
    from .literature_search import (
        search_pubmed,
        search_arxiv, 
        search_crossref,
        search_web_academic,
        multi_source_literature_search
    )
    LITERATURE_SEARCH_AVAILABLE = True
except ImportError:
    LITERATURE_SEARCH_AVAILABLE = False

# Text processing is always available (basic functionality)
from .text_processing import TextAnalyzer, create_analyzer

__all__ = [
    # Text Processing (always available)
    'TextAnalyzer',
    'create_analyzer'
]

# Add conditional exports
if LLM_CLIENT_AVAILABLE:
    __all__.extend(['LLMClient', 'create_llm_client'])

if LITERATURE_SEARCH_AVAILABLE:
    __all__.extend([
        'search_pubmed',
        'search_arxiv',
        'search_crossref', 
        'search_web_academic',
        'multi_source_literature_search'
    ])