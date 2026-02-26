"""Authentication and authorization utilities for the API."""

import os
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
API_KEY_NAME = "X-API-Key"

# Security schemes
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
http_bearer = HTTPBearer(auto_error=False)


class TokenData(BaseModel):
    """Token data model."""

    username: Optional[str] = None
    scopes: list[str] = []


class User(BaseModel):
    """User model."""

    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    scopes: list[str] = []


class UserInDB(User):
    """User model with hashed password."""

    hashed_password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password to compare against
        
    Returns:
        bool: True if password matches, False otherwise
        
    Note:
        Returns False for passwords >72 bytes (bcrypt 5.x limit) or malformed hashes.
    """
    try:
        password_bytes = plain_password.encode('utf-8')
        # bcrypt 5.x raises ValueError for passwords >72 bytes
        if len(password_bytes) > 72:
            return False
        return bcrypt.checkpw(password_bytes, hashed_password.encode('utf-8'))
    except (ValueError, AttributeError, TypeError):
        # Return False for malformed/invalid hashed passwords or encoding errors
        return False


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt.
    
    Args:
        password: Plain text password to hash
        
    Returns:
        str: Hashed password
        
    Raises:
        ValueError: If password exceeds 72 bytes (bcrypt limit)
    """
    password_bytes = password.encode('utf-8')
    # bcrypt 5.x raises ValueError for passwords >72 bytes
    if len(password_bytes) > 72:
        raise ValueError(
            f"Password too long: {len(password_bytes)} bytes (max 72 bytes for bcrypt)"
        )
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode('utf-8')


# In-memory user database (use proper database in production)
USERS_DB: dict[str, UserInDB] = {
    "admin": UserInDB(
        username="admin",
        email="admin@example.com",
        full_name="Admin User",
        hashed_password=get_password_hash("admin123"),  # Change in production!
        scopes=["read", "write", "admin"],
    ),
    "user": UserInDB(
        username="user",
        email="user@example.com",
        full_name="Regular User",
        hashed_password=get_password_hash("user123"),  # Change in production!
        scopes=["read"],
    ),
}

# In-memory API keys (use proper secret storage in production)
API_KEYS: dict[str, dict[str, Any]] = {
    "test-api-key-12345": {
        "user": "admin",
        "scopes": ["read", "write"],
        "name": "Test API Key",
    },
}


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database."""
    return USERS_DB.get(username)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(
    data: dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """Verify a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise JWTError("Username not found in token")
        scopes = payload.get("scopes", [])
        return TokenData(username=username, scopes=scopes)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_api_key(
    api_key: str = Security(api_key_header),
) -> Optional[dict[str, Any]]:
    """Validate API key."""
    if api_key is None:
        return None

    if api_key in API_KEYS:
        return API_KEYS[api_key]

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
    )


async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Security(http_bearer),
) -> User:
    """Get current user from JWT token."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_data = verify_token(credentials.credentials)
    user = get_user(token_data.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    if user.disabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is disabled",
        )
    return user


async def get_current_user_optional(
    api_key: Optional[dict[str, Any]] = Security(get_api_key),
    token_user: User = Depends(get_current_user_from_token),
) -> Optional[User]:
    """Get current user from either API key or token."""
    if api_key:
        username = api_key.get("user")
        user = get_user(username)
        if user:
            return user
    return token_user if token_user else None


async def get_current_active_user(
    current_user: User = Depends(get_current_user_from_token),
) -> User:
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user",
        )
    return current_user


class RoleChecker:
    """Role-based permission checker."""

    def __init__(self, allowed_roles: list[str]):
        """Initialize with allowed roles."""
        self.allowed_roles = allowed_roles

    def __call__(self, user: User = Depends(get_current_active_user)) -> User:
        """Check if user has required role."""
        if not any(role in user.scopes for role in self.allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied",
            )
        return user


# Permission checkers
require_admin = RoleChecker(["admin"])
require_write = RoleChecker(["write", "admin"])
require_read = RoleChecker(["read", "write", "admin"])


def check_scopes(required_scopes: list[str]):
    """Decorator to check scopes."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get user from kwargs or args
            user = kwargs.get("current_user")
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission denied",
                )

            user_scopes = set(user.scopes)
            required = set(required_scopes)
            if not required.issubset(user_scopes):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions",
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator
