"""
LLM Client Utility

Provides a unified interface for interacting with language models with
retry logic, error handling, and rate limiting.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio
from functools import wraps

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.caches import BaseCache
from langchain_core.callbacks import Callbacks

from core.config import get_config

logger = logging.getLogger(__name__)

# Fix Pydantic model rebuild issue
try:
    ChatGoogleGenerativeAI.model_rebuild()
except Exception as e:
    logger.warning(f"Could not rebuild ChatGoogleGenerativeAI model: {e}")

# Global rate limiter for LLM calls
import threading
from collections import deque

class GlobalRateLimiter:
    def __init__(self, max_calls_per_minute: int = 3500):  # Conservative limit under 4000/min
        self.max_calls = max_calls_per_minute
        self.calls = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            while self.calls and now - self.calls[0] >= 60:
                self.calls.popleft()
            
            # If we're at the limit, wait
            if len(self.calls) >= self.max_calls:
                sleep_time = 60 - (now - self.calls[0]) + 1  # Wait until oldest call is >60s old
                logger.info(f"Rate limit reached ({len(self.calls)}/{self.max_calls} calls/min), waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
                return self.wait_if_needed()  # Recursive call to re-check
            
            # Record this call
            self.calls.append(now)

# Global rate limiter instance
_global_rate_limiter = GlobalRateLimiter()

@dataclass
class LLMResponse:
    """Standardized response from LLM"""
    content: str
    model: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    error: Optional[str] = None

@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True

def retry_with_exponential_backoff(retry_config: RetryConfig = None):
    """Decorator for exponential backoff retry logic"""
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retry_config.max_retries:
                        logger.error(f"Failed after {retry_config.max_retries} retries: {e}")
                        raise e
                    
                    # Check if it's a rate limit error (429) for longer delay
                    is_rate_limit = "429" in str(e) or "rate limit" in str(e).lower() or "quota" in str(e).lower()
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    
                    # For rate limit errors, use longer delay
                    if is_rate_limit:
                        delay = min(delay * 2, 60.0)  # Up to 1 minute for rate limits
                        logger.warning(f"Rate limit detected: {e}. Using extended delay of {delay:.2f}s")
                    
                    # Add jitter to prevent thundering herd
                    if retry_config.jitter:
                        import random
                        delay *= (0.5 + 0.5 * random.random())
                    
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
            
            return None
        return wrapper
    return decorator

class LLMClient:
    """Unified LLM client with retry logic and error handling"""
    
    def __init__(self, model_name: str = None, temperature: float = None, 
                 max_tokens: int = None, retry_config: RetryConfig = None):
        """
        Initialize LLM client
        
        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            retry_config: Configuration for retry logic
        """
        self.config = get_config()
        
        # Use config defaults if not specified
        self.model_name = model_name or self.config.model.gemini_model
        self.temperature = temperature or self.config.model.gemini_temperature
        self.max_tokens = max_tokens or self.config.model.gemini_max_tokens
        self.retry_config = retry_config or RetryConfig()
        
        # Initialize the underlying LLM
        self._llm = None
        self._initialize_llm()
        
        logger.info(f"LLM Client initialized with model: {self.model_name}")
    
    def _initialize_llm(self):
        """Initialize the underlying LLM instance"""
        try:
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                google_api_key=self.config.model.gemini_api_key
            )
            logger.debug("Successfully initialized Gemini LLM")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    @retry_with_exponential_backoff()
    def invoke(self, messages: Union[str, List[Union[str, tuple, BaseMessage]]]) -> LLMResponse:
        """
        Invoke the LLM with retry logic and rate limiting
        
        Args:
            messages: Single message string, list of messages, or BaseMessage objects
            
        Returns:
            LLMResponse object with content and metadata
        """
        # Apply global rate limiting before making the call
        _global_rate_limiter.wait_if_needed()
        
        start_time = time.time()
        
        try:
            # Normalize messages to the expected format
            normalized_messages = self._normalize_messages(messages)
            
            # Invoke the LLM
            response = self._llm.invoke(normalized_messages)
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.content,
                model=self.model_name,
                response_time=response_time
            )
            
        except Exception as e:
            error_msg = f"LLM invocation failed: {str(e)}"
            logger.error(error_msg)
            return LLMResponse(
                content="",
                model=self.model_name,
                response_time=time.time() - start_time,
                error=error_msg
            )
    
    def _normalize_messages(self, messages: Union[str, List[Union[str, tuple, BaseMessage]]]) -> List[BaseMessage]:
        """Normalize various message formats to BaseMessage objects"""
        if isinstance(messages, str):
            return [HumanMessage(content=messages)]
        
        if not isinstance(messages, list):
            raise ValueError("Messages must be a string or list")
        
        normalized = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                normalized.append(msg)
            elif isinstance(msg, str):
                normalized.append(HumanMessage(content=msg))
            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                if role.lower() in ['human', 'user']:
                    normalized.append(HumanMessage(content=content))
                elif role.lower() in ['system', 'assistant']:
                    normalized.append(SystemMessage(content=content))
                else:
                    normalized.append(HumanMessage(content=content))
            else:
                raise ValueError(f"Unsupported message format: {type(msg)}")
        
        return normalized
    
    async def ainvoke(self, messages: Union[str, List[Union[str, tuple, BaseMessage]]]) -> LLMResponse:
        """Async version of invoke"""
        # For now, just run sync version in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, messages)
    
    def batch_invoke(self, message_batches: List[Union[str, List[Union[str, tuple, BaseMessage]]]]) -> List[LLMResponse]:
        """
        Process multiple message batches
        
        Args:
            message_batches: List of message batches to process
            
        Returns:
            List of LLMResponse objects
        """
        responses = []
        for i, messages in enumerate(message_batches):
            logger.debug(f"Processing batch {i+1}/{len(message_batches)}")
            response = self.invoke(messages)
            responses.append(response)
            
            # Add small delay between batches to respect rate limits
            if i < len(message_batches) - 1:
                time.sleep(self.config.database.rate_limit_delay)
        
        return responses
    
    def is_healthy(self) -> bool:
        """Check if the LLM client is healthy"""
        try:
            test_response = self.invoke("test")
            return test_response.error is None
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'retry_config': self.retry_config.__dict__,
            'is_healthy': self.is_healthy()
        }

class MockLLMClient(LLMClient):
    """Mock LLM client for testing"""
    
    def __init__(self, responses: List[str] = None):
        """Initialize mock client with predefined responses"""
        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.model_name = "mock-model"
        self.temperature = 0.7
        self.max_tokens = 1000
        self.retry_config = RetryConfig()
        
        logger.info("Mock LLM Client initialized")
    
    def invoke(self, messages: Union[str, List[Union[str, tuple, BaseMessage]]]) -> LLMResponse:
        """Return mock response"""
        response_content = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        
        return LLMResponse(
            content=response_content,
            model=self.model_name,
            response_time=0.1
        )
    
    def is_healthy(self) -> bool:
        return True

def create_llm_client(use_mock: bool = None, **kwargs) -> LLMClient:
    """
    Factory function to create LLM client
    
    Args:
        use_mock: Whether to use mock client (defaults to config setting)
        **kwargs: Additional arguments for LLM client
        
    Returns:
        LLMClient instance
    """
    config = get_config()
    
    if use_mock is None:
        use_mock = config.system.use_mock_llm
    
    if use_mock:
        return MockLLMClient(**kwargs)
    else:
        return LLMClient(**kwargs)

# Global client instance for backward compatibility
_global_client: Optional[LLMClient] = None

def get_global_llm_client() -> LLMClient:
    """Get or create global LLM client instance"""
    global _global_client
    if _global_client is None:
        _global_client = create_llm_client()
    return _global_client

def reset_global_llm_client():
    """Reset the global LLM client (useful for testing)"""
    global _global_client
    _global_client = None

if __name__ == "__main__":
    # Test the LLM client
    try:
        client = create_llm_client()
        
        # Test basic invoke
        response = client.invoke("What is 2+2?")
        print(f"✅ LLM Response: {response.content[:100]}")
        print(f"📊 Response time: {response.response_time:.2f}s")
        
        # Test health check
        print(f"🔍 Client healthy: {client.is_healthy()}")
        
        # Test stats
        stats = client.get_stats()
        print(f"📈 Client stats: {stats}")
        
    except Exception as e:
        print(f"❌ LLM Client test failed: {e}")
        print("\n💡 Make sure to:")
        print("   1. Set GOOGLE_API_KEY in your .env file")
        print("   2. Check your internet connection")
        print("   3. Verify API key permissions")