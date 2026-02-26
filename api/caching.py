"""Caching utilities using Redis for the API."""

import json
import os
import pickle
from functools import wraps
from typing import Any, Callable, Optional, Union

import redis
from fastapi import Request

from api.logging_config import get_logger

logger = get_logger(__name__)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() == "true"


class RedisCache:
    """Redis cache manager."""

    def __init__(self) -> None:
        """Initialize Redis connection."""
        self._client: Optional[redis.Redis] = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Redis. Skipped if CACHE_ENABLED=false."""
        if os.getenv("CACHE_ENABLED", "true").lower() == "false":
            logger.info("redis_cache_disabled", reason="CACHE_ENABLED=false")
            return False
        try:
            self._client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                ssl=REDIS_SSL,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            self._client.ping()
            self._connected = True
            logger.info("redis_connected", host=REDIS_HOST, port=REDIS_PORT)
            return True
        except redis.ConnectionError as e:
            logger.warning("redis_connection_failed", error=str(e))
            self._connected = False
            return False
        except Exception as e:
            logger.error("redis_connection_error", error=str(e))
            self._connected = False
            return False

    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        if not self._connected or self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except redis.ConnectionError:
            self._connected = False
            return False

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.is_connected():
            return None

        try:
            data = self._client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error("redis_get_error", key=key, error=str(e))
            return None

    def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None,
    ) -> bool:
        """Set value in cache with optional expiration (seconds)."""
        if not self.is_connected():
            return False

        try:
            serialized = pickle.dumps(value)
            self._client.set(key, serialized, ex=expire)
            return True
        except Exception as e:
            logger.error("redis_set_error", key=key, error=str(e))
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self.is_connected():
            return False

        try:
            self._client.delete(key)
            return True
        except Exception as e:
            logger.error("redis_delete_error", key=key, error=str(e))
            return False

    def flush(self) -> bool:
        """Clear all cache."""
        if not self.is_connected():
            return False

        try:
            self._client.flushdb()
            return True
        except Exception as e:
            logger.error("redis_flush_error", error=str(e))
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get Redis stats."""
        if not self.is_connected():
            return {"connected": False}

        try:
            info = self._client.info()
            return {
                "connected": True,
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.error("redis_stats_error", error=str(e))
            return {"connected": True, "error": str(e)}


# Global cache instance
cache = RedisCache()


def cached(
    expire: int = 300,
    key_prefix: str = "cache",
    key_func: Optional[Callable[..., str]] = None,
) -> Callable:
    """Decorator to cache function results.

    Args:
        expire: Cache expiration time in seconds.
        key_prefix: Prefix for cache key.
        key_func: Optional function to generate cache key from arguments.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                cache_key = f"{key_prefix}:{key_func(*args, **kwargs)}"
            else:
                # Default key generation
                key_parts = [key_prefix, func.__name__]
                for arg in args:
                    if isinstance(arg, (str, int, float, bool)):
                        key_parts.append(str(arg))
                    elif isinstance(arg, dict):
                        key_parts.append(json.dumps(arg, sort_keys=True))
                for k, v in sorted(kwargs.items()):
                    if isinstance(v, (str, int, float, bool)):
                        key_parts.append(f"{k}={v}")
                cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug("cache_hit", key=cache_key)
                return cached_value

            # Call function
            result = await func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, expire=expire)
            logger.debug("cache_miss", key=cache_key)

            return result

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                cache_key = f"{key_prefix}:{key_func(*args, **kwargs)}"
            else:
                key_parts = [key_prefix, func.__name__]
                for arg in args:
                    if isinstance(arg, (str, int, float, bool)):
                        key_parts.append(str(arg))
                    elif isinstance(arg, dict):
                        key_parts.append(json.dumps(arg, sort_keys=True))
                for k, v in sorted(kwargs.items()):
                    if isinstance(v, (str, int, float, bool)):
                        key_parts.append(f"{k}={v}")
                cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug("cache_hit", key=cache_key)
                return cached_value

            # Call function
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, expire=expire)
            logger.debug("cache_miss", key=cache_key)

            return result

        return async_wrapper if func.__code__.co_flags & 0x80 else sync_wrapper
    return decorator


def invalidate_cache(pattern: str) -> bool:
    """Invalidate cache keys matching pattern."""
    if not cache.is_connected():
        return False

    try:
        keys = cache._client.keys(pattern)
        if keys:
            cache._client.delete(*keys)
        return True
    except Exception as e:
        logger.error("cache_invalidation_error", pattern=pattern, error=str(e))
        return False


class PredictionCache:
    """Specialized cache for predictions."""

    PREFIX = "prediction"
    DEFAULT_EXPIRE = 3600  # 1 hour

    @staticmethod
    def generate_key(features: dict[str, Any]) -> str:
        """Generate cache key from features."""
        # Create deterministic key from features
        sorted_features = json.dumps(features, sort_keys=True)
        import hashlib
        feature_hash = hashlib.md5(sorted_features.encode()).hexdigest()
        return f"{PredictionCache.PREFIX}:{feature_hash}"

    @staticmethod
    def get(features: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Get cached prediction."""
        key = PredictionCache.generate_key(features)
        return cache.get(key)

    @staticmethod
    def set(features: dict[str, Any], prediction: dict[str, Any], expire: int = DEFAULT_EXPIRE) -> bool:
        """Cache prediction."""
        key = PredictionCache.generate_key(features)
        return cache.set(key, prediction, expire=expire)

    @staticmethod
    def invalidate() -> bool:
        """Invalidate all predictions."""
        return invalidate_cache(f"{PredictionCache.PREFIX}:*")
