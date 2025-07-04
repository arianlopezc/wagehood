"""
API Authentication and Rate Limiting

This module provides authentication and rate limiting functionality
for the Wagehood real-time trading API.
"""

import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from collections import defaultdict
import logging

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manages API key authentication and validation."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize API key manager.
        
        Args:
            redis_client: Optional Redis client for key storage
        """
        self.redis_client = redis_client
        self._local_keys: Dict[str, Dict[str, Any]] = {}
        
        # Default admin key (should be changed in production)
        self._create_default_keys()
    
    def _create_default_keys(self):
        """Create default API keys for development."""
        # Generate secure admin key
        admin_key = secrets.token_urlsafe(32)
        self._local_keys[admin_key] = {
            "name": "admin",
            "permissions": ["read", "write", "admin"],
            "rate_limit": 1000,  # requests per minute
            "created_at": datetime.now(),
            "active": True
        }
        
        # Generate read-only key
        readonly_key = secrets.token_urlsafe(32) 
        self._local_keys[readonly_key] = {
            "name": "readonly",
            "permissions": ["read"],
            "rate_limit": 100,
            "created_at": datetime.now(),
            "active": True
        }
        
        logger.info(f"Default API keys created:")
        logger.info(f"Admin key: {admin_key}")
        logger.info(f"Read-only key: {readonly_key}")
    
    def create_key(self, name: str, permissions: list, rate_limit: int = 100) -> str:
        """
        Create a new API key.
        
        Args:
            name: Key name/description
            permissions: List of permissions (read, write, admin)
            rate_limit: Rate limit in requests per minute
            
        Returns:
            Generated API key
        """
        api_key = secrets.token_urlsafe(32)
        key_data = {
            "name": name,
            "permissions": permissions,
            "rate_limit": rate_limit,
            "created_at": datetime.now(),
            "active": True
        }
        
        if self.redis_client:
            # Store in Redis
            try:
                self.redis_client.hset(
                    f"api_key:{api_key}",
                    mapping={k: str(v) for k, v in key_data.items()}
                )
                logger.info(f"API key created in Redis: {name}")
            except Exception as e:
                logger.error(f"Failed to store API key in Redis: {e}")
                # Fall back to local storage
                self._local_keys[api_key] = key_data
        else:
            # Store locally
            self._local_keys[api_key] = key_data
        
        logger.info(f"API key created: {name} ({api_key[:8]}...)")
        return api_key
    
    def validate_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate an API key and return key information.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Key information if valid, None otherwise
        """
        if not api_key:
            return None
        
        # Try Redis first
        if self.redis_client:
            try:
                key_data = self.redis_client.hgetall(f"api_key:{api_key}")
                if key_data:
                    # Convert back to proper types
                    return {
                        "name": key_data.get("name", ""),
                        "permissions": key_data.get("permissions", "").split(","),
                        "rate_limit": int(key_data.get("rate_limit", 100)),
                        "created_at": datetime.fromisoformat(key_data.get("created_at", "")),
                        "active": key_data.get("active", "true").lower() == "true"
                    }
            except Exception as e:
                logger.error(f"Failed to validate key in Redis: {e}")
        
        # Fall back to local storage
        return self._local_keys.get(api_key)
    
    def revoke_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if key was revoked successfully
        """
        if self.redis_client:
            try:
                result = self.redis_client.delete(f"api_key:{api_key}")
                if result:
                    logger.info(f"API key revoked from Redis: {api_key[:8]}...")
                    return True
            except Exception as e:
                logger.error(f"Failed to revoke key in Redis: {e}")
        
        # Fall back to local storage
        if api_key in self._local_keys:
            del self._local_keys[api_key]
            logger.info(f"API key revoked locally: {api_key[:8]}...")
            return True
        
        return False
    
    def list_keys(self) -> list:
        """
        List all active API keys.
        
        Returns:
            List of key information (without actual keys)
        """
        keys = []
        
        # Get from Redis
        if self.redis_client:
            try:
                pattern = "api_key:*"
                for key in self.redis_client.scan_iter(match=pattern):
                    key_data = self.redis_client.hgetall(key)
                    if key_data.get("active", "true").lower() == "true":
                        keys.append({
                            "name": key_data.get("name", ""),
                            "permissions": key_data.get("permissions", "").split(","),
                            "rate_limit": int(key_data.get("rate_limit", 100)),
                            "created_at": key_data.get("created_at", "")
                        })
            except Exception as e:
                logger.error(f"Failed to list keys from Redis: {e}")
        
        # Add local keys
        for api_key, key_data in self._local_keys.items():
            if key_data.get("active", True):
                keys.append({
                    "name": key_data["name"],
                    "permissions": key_data["permissions"],
                    "rate_limit": key_data["rate_limit"],
                    "created_at": key_data["created_at"].isoformat()
                })
        
        return keys


class RateLimiter:
    """Rate limiting functionality using sliding window algorithm."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize rate limiter.
        
        Args:
            redis_client: Optional Redis client for distributed rate limiting
        """
        self.redis_client = redis_client
        self._local_requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, key: str, limit: int, window: int = 60) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            key: Rate limiting key (e.g., API key or IP)
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        current_time = time.time()
        window_start = current_time - window
        
        if self.redis_client:
            return self._redis_rate_limit(key, limit, window, current_time, window_start)
        else:
            return self._local_rate_limit(key, limit, window, current_time, window_start)
    
    def _redis_rate_limit(self, key: str, limit: int, window: int, 
                         current_time: float, window_start: float) -> tuple[bool, Dict[str, Any]]:
        """Redis-based distributed rate limiting."""
        try:
            redis_key = f"rate_limit:{key}"
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(redis_key, 0, window_start)
            
            # Count current requests
            pipe.zcard(redis_key)
            
            # Add current request
            pipe.zadd(redis_key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(redis_key, window * 2)
            
            # Execute pipeline
            results = pipe.execute()
            current_requests = results[1]
            
            # Check if allowed
            allowed = current_requests < limit
            
            if not allowed:
                # Remove the request we just added
                self.redis_client.zrem(redis_key, str(current_time))
            
            # Get remaining requests and reset time
            remaining = max(0, limit - current_requests - (1 if allowed else 0))
            reset_time = current_time + window
            
            return allowed, {
                "limit": limit,
                "remaining": remaining,
                "reset": reset_time,
                "retry_after": window if not allowed else None
            }
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fall back to local rate limiting
            return self._local_rate_limit(key, limit, window, current_time, window_start)
    
    def _local_rate_limit(self, key: str, limit: int, window: int,
                         current_time: float, window_start: float) -> tuple[bool, Dict[str, Any]]:
        """Local in-memory rate limiting."""
        # Clean old entries
        self._local_requests[key] = [
            req_time for req_time in self._local_requests[key] 
            if req_time > window_start
        ]
        
        # Check if allowed
        current_requests = len(self._local_requests[key])
        allowed = current_requests < limit
        
        if allowed:
            self._local_requests[key].append(current_time)
        
        # Calculate remaining and reset time
        remaining = max(0, limit - current_requests - (1 if allowed else 0))
        reset_time = current_time + window
        
        return allowed, {
            "limit": limit,
            "remaining": remaining,
            "reset": reset_time,
            "retry_after": window if not allowed else None
        }


# Global instances
api_key_manager = APIKeyManager()
rate_limiter = RateLimiter()

# Security scheme
security = HTTPBearer(auto_error=False)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user.
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        User information from API key
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Validate API key
    key_info = api_key_manager.validate_key(credentials.credentials)
    if not key_info or not key_info.get("active", True):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return key_info


def require_permission(permission: str):
    """
    Dependency factory for requiring specific permissions.
    
    Args:
        permission: Required permission (read, write, admin)
        
    Returns:
        Dependency function
    """
    def permission_dependency(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        if permission not in user.get("permissions", []):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        return user
    
    return permission_dependency


def check_rate_limit(request: Request, user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Dependency to check rate limits.
    
    Args:
        request: FastAPI request object
        user: Current authenticated user
        
    Returns:
        User information if rate limit allows
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    # Use API key as rate limit key
    rate_limit_key = f"user:{user['name']}"
    user_limit = user.get("rate_limit", 100)
    
    allowed, rate_info = rate_limiter.is_allowed(rate_limit_key, user_limit)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": str(rate_info["remaining"]),
                "X-RateLimit-Reset": str(int(rate_info["reset"])),
                "Retry-After": str(int(rate_info["retry_after"] or 60))
            }
        )
    
    # Add rate limit headers to response (this should be done in middleware)
    request.state.rate_limit_headers = {
        "X-RateLimit-Limit": str(rate_info["limit"]),
        "X-RateLimit-Remaining": str(rate_info["remaining"]),
        "X-RateLimit-Reset": str(int(rate_info["reset"]))
    }
    
    return user


# Convenience dependencies
require_read = require_permission("read")
require_write = require_permission("write") 
require_admin = require_permission("admin")