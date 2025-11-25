"""
Embedding cache with improved hashing and memory management.
"""
import hashlib
from typing import Optional

import numpy as np

from ..config import config
from ..logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """LRU cache for embeddings with memory management."""
    
    def __init__(self, max_size: int = None):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.cache: dict = {}
        self.max_size = max_size or config.embedding_cache_size
        self.access_order: list = []
        logger.debug(f"Initialized embedding cache with max_size={self.max_size}")
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate stable cache key using SHA256.
        
        Args:
            text: Text to generate key for
        
        Returns:
            SHA256 hash as hex string
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.
        
        Args:
            text: Text to get embedding for
        
        Returns:
            Cached embedding or None if not found
        """
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            # Move to end (most recently used)
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            return self.cache[cache_key]
        return None
    
    def set(self, text: str, embedding: np.ndarray) -> None:
        """
        Set embedding in cache with LRU eviction.
        
        Args:
            text: Text key
            embedding: Embedding vector to cache
        """
        cache_key = self._get_cache_key(text)
        
        # Evict if cache is full
        if len(self.cache) >= self.max_size and cache_key not in self.cache:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
            logger.debug(f"Evicted embedding from cache (LRU): {lru_key[:8]}...")
        
        self.cache[cache_key] = embedding
        if cache_key not in self.access_order:
            self.access_order.append(cache_key)
    
    def clear(self) -> None:
        """Clear cache and force garbage collection."""
        self.cache.clear()
        self.access_order.clear()
        import gc
        gc.collect()
        logger.info("Embedding cache cleared")
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

