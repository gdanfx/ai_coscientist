"""
AI Co-Scientist Configuration Management

This module handles loading and validation of environment variables and configuration
settings for the AI Co-Scientist system.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded environment variables from {env_path}")
else:
    logger.info("No .env file found, using system environment variables")

@dataclass
class DatabaseConfig:
    """Configuration for literature search databases"""
    pubmed_email: str
    crossref_email: Optional[str] = None
    rate_limit_delay: float = 1.0
    web_search_delay: float = 3.0
    max_results_per_source: int = 3
    total_max_results: int = 12

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash-exp"
    gemini_temperature: float = 0.7
    gemini_max_tokens: int = 4000
    sentence_transformer_model: str = "all-MiniLM-L6-v2"

@dataclass
class AgentConfig:
    """Configuration for agent-specific settings"""
    elo_k_factor: float = 32
    initial_elo_rating: float = 1200
    proximity_distance_threshold: float = 0.25

@dataclass
class SystemConfig:
    """Configuration for system-wide settings"""
    log_level: str = "INFO"
    cache_enabled: bool = True
    debug_mode: bool = False
    use_mock_llm: bool = False

@dataclass
class Config:
    """Main configuration class containing all sub-configurations"""
    database: DatabaseConfig
    model: ModelConfig
    agent: AgentConfig
    system: SystemConfig

    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables"""
        return cls(
            database=DatabaseConfig(
                pubmed_email=get_required_env('PUBMED_EMAIL'),
                crossref_email=get_optional_env('CROSSREF_EMAIL'),
                rate_limit_delay=get_float_env('RATE_LIMIT_DELAY', 1.0),
                web_search_delay=get_float_env('WEB_SEARCH_DELAY', 3.0),
                max_results_per_source=get_int_env('MAX_RESULTS_PER_SOURCE', 3),
                total_max_results=get_int_env('TOTAL_MAX_RESULTS', 12)
            ),
            model=ModelConfig(
                gemini_api_key=get_required_env('GOOGLE_API_KEY'),
                gemini_model=get_env('GEMINI_MODEL', 'gemini-2.5-flash-lite-preview-06-17'),
                gemini_temperature=get_float_env('GEMINI_TEMPERATURE', 0.7),
                gemini_max_tokens=get_int_env('GEMINI_MAX_TOKENS', 4000),
                sentence_transformer_model=get_env('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
            ),
            agent=AgentConfig(
                elo_k_factor=get_float_env('ELO_K_FACTOR', 32),
                initial_elo_rating=get_float_env('INITIAL_ELO_RATING', 1200),
                proximity_distance_threshold=get_float_env('PROXIMITY_DISTANCE_THRESHOLD', 0.25)
            ),
            system=SystemConfig(
                log_level=get_env('LOG_LEVEL', 'INFO'),
                cache_enabled=get_bool_env('CACHE_ENABLED', True),
                debug_mode=get_bool_env('DEBUG_MODE', False),
                use_mock_llm=get_bool_env('USE_MOCK_LLM', False)
            )
        )

    def validate(self) -> None:
        """Validate the configuration"""
        errors = []

        # Validate email format for PubMed
        if not self.database.pubmed_email or '@' not in self.database.pubmed_email:
            errors.append("PUBMED_EMAIL must be a valid email address")

        # Validate API key
        if not self.model.gemini_api_key or self.model.gemini_api_key == 'your_gemini_api_key_here':
            errors.append("GOOGLE_API_KEY must be set to a valid API key")

        # Validate numeric ranges
        if not 0 <= self.model.gemini_temperature <= 2:
            errors.append("GEMINI_TEMPERATURE must be between 0 and 2")

        if self.model.gemini_max_tokens <= 0:
            errors.append("GEMINI_MAX_TOKENS must be positive")

        if self.database.rate_limit_delay < 0:
            errors.append("RATE_LIMIT_DELAY must be non-negative")

        if self.database.max_results_per_source <= 0:
            errors.append("MAX_RESULTS_PER_SOURCE must be positive")

        if not 0 <= self.agent.proximity_distance_threshold <= 1:
            errors.append("PROXIMITY_DISTANCE_THRESHOLD must be between 0 and 1")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

        logger.info("Configuration validation passed")

def get_env(key: str, default: str = None) -> str:
    """Get environment variable with optional default"""
    return os.getenv(key, default)

def get_required_env(key: str) -> str:
    """Get required environment variable, raise error if not found"""
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return value

def get_optional_env(key: str) -> Optional[str]:
    """Get optional environment variable, return None if not found"""
    return os.getenv(key)

def get_int_env(key: str, default: int) -> int:
    """Get integer environment variable with default"""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer value for {key}: {value}, using default {default}")
        return default

def get_float_env(key: str, default: float) -> float:
    """Get float environment variable with default"""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float value for {key}: {value}, using default {default}")
        return default

def get_bool_env(key: str, default: bool) -> bool:
    """Get boolean environment variable with default"""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')

# Global configuration instance
_config: Optional[Config] = None

def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config.from_env()
        _config.validate()
        
        # Configure logging level
        logging.getLogger().setLevel(getattr(logging, _config.system.log_level.upper()))
        
    return _config

def reload_config() -> Config:
    """Reload configuration from environment variables"""
    global _config
    _config = None
    return get_config()

# Configuration constants for backward compatibility
def get_pubmed_email() -> str:
    """Get PubMed email from configuration"""
    return get_config().database.pubmed_email

def get_rate_limit_delay() -> float:
    """Get rate limit delay from configuration"""
    return get_config().database.rate_limit_delay

def get_max_results_per_source() -> int:
    """Get max results per source from configuration"""
    return get_config().database.max_results_per_source

def get_gemini_api_key() -> str:
    """Get Gemini API key from configuration"""
    return get_config().model.gemini_api_key

def get_gemini_model() -> str:
    """Get Gemini model name from configuration"""
    return get_config().model.gemini_model

# Export commonly used constants
PUBMED_EMAIL = get_pubmed_email
RATE_LIMIT_DELAY = get_rate_limit_delay
MAX_RESULTS_PER_SOURCE = get_max_results_per_source
TOTAL_MAX_RESULTS = lambda: get_config().database.total_max_results
WEB_SEARCH_DELAY = lambda: get_config().database.web_search_delay

if __name__ == "__main__":
    # Test configuration loading
    try:
        config = get_config()
        print("✅ Configuration loaded successfully")
        print(f"📧 PubMed Email: {config.database.pubmed_email}")
        print(f"🤖 Gemini Model: {config.model.gemini_model}")
        print(f"📊 Log Level: {config.system.log_level}")
        print(f"🔄 Cache Enabled: {config.system.cache_enabled}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("\n💡 Make sure to:")
        print("   1. Copy .env.example to .env")
        print("   2. Fill in your API keys and email address")
        print("   3. Set GOOGLE_API_KEY and PUBMED_EMAIL")