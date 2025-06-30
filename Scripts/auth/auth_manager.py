from typing import Dict, List, Optional, Set, Union
from datetime import datetime, timedelta
import uuid
import jwt
from passlib.hash import bcrypt
from fastapi import Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
import logging
from enum import Enum
from pydantic import BaseModel, Field
from dataclasses import dataclass
import json
import asyncio
import redis.asyncio as redis
from contextlib import asynccontextmanager

# Import AuditLogger components for type hinting and use
try:
    from ..security.audit_logger import AuditLogger, AuditEvent # Relative import if audit_logger is in security
except ImportError:
    # Fallback for potential standalone use or different structure, though less ideal
    AuditLogger = None
    AuditEvent = None

logger = logging.getLogger(__name__)

class Permission(str, Enum):
    # Document permissions
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    
    # Search permissions
    SEARCH_BASIC = "search:basic"
    SEARCH_ADVANCED = "search:advanced"
    
    # Admin permissions
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
    ADMIN_FULL = "admin:full"
    
    # API permissions
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_EXECUTE = "api:execute"

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    READER = "reader"
    API = "api"

class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    hashed_password: str
    roles: List[Role]
    permissions: Set[Permission]
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    metadata: Dict = Field(default_factory=dict)

class APIKey(BaseModel):
    key: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    user_id: str
    permissions: Set[Permission]
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    metadata: Dict = Field(default_factory=dict)

@dataclass
class AuthConfig:
    secret_key: str
    token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30
    api_key_prefix: str = "rag-"
    redis_url: str = "redis://localhost:6379/0"

class AuthManager:
    def __init__(self, config: AuthConfig):
        self.config = config
        self.redis = redis.Redis.from_url(config.redis_url)
        self._role_permissions = self._initialize_role_permissions()
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self.api_key_header = APIKeyHeader(name="X-API-Key")

    def _initialize_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """Initialize default role permissions"""
        return {
            Role.ADMIN: {p for p in Permission},
            Role.USER: {
                Permission.DOCUMENT_READ,
                Permission.DOCUMENT_WRITE,
                Permission.SEARCH_BASIC,
                Permission.SEARCH_ADVANCED,
                Permission.API_READ,
                Permission.API_EXECUTE
            },
            Role.READER: {
                Permission.DOCUMENT_READ,
                Permission.SEARCH_BASIC,
                Permission.API_READ
            },
            Role.API: {
                Permission.API_READ,
                Permission.API_EXECUTE
            }
        }

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: List[Role],
        audit_logger: Optional[AuditLogger] = None, # Added
        ip_address: Optional[str] = None, # Added
        performed_by_user_id: Optional[str] = "system" # ID of user performing creation, system if self-signup/admin
    ) -> User:
        """Create a new user"""
        if len(password) < self.config.password_min_length:
            raise ValueError(
                f"Password must be at least {self.config.password_min_length} "
                "characters long"
            )

        # Hash password
        hashed_password = bcrypt.hash(password)
        
        # Calculate permissions based on roles
        permissions = set()
        for role in roles:
            permissions.update(self._role_permissions[role])

        # Create user
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            roles=roles,
            permissions=permissions
        )

        # Store in Redis
        await self.redis.hset(
            f"users",
            user.id,
            user.json()
        )

        if audit_logger and AuditEvent:
            audit_logger.log_event(AuditEvent(
                event_type="user_creation",
                user_id=performed_by_user_id, # User performing the action
                action="User Created",
                resource_type="user",
                resource_id=user.id, # ID of the created user
                status="success",
                ip_address=ip_address,
                metadata={"username": user.username, "roles": [r.value for r in user.roles]}
            ))
        return user

    async def authenticate_user(
        self,
        username: str,
        password: str,
        audit_logger: Optional[AuditLogger] = None, # Added
        ip_address: Optional[str] = None # Added
    ) -> Optional[User]:
        """Authenticate a user with username and password"""
        # Check failed attempts
        failed_key = f"auth:failed:{username}"
        failed_attempts = await self.redis.get(failed_key)
        
        if failed_attempts and int(failed_attempts) >= self.config.max_failed_attempts:
            raise HTTPException(
                status_code=403,
                detail="Account temporarily locked due to too many failed attempts"
            )

        # Get user
        users = await self.redis.hgetall("users")
        user = None
        
        for user_data in users.values():
            u = User.parse_raw(user_data)
            if u.username == username:
                user = u
                break

        if not user or user.status != UserStatus.ACTIVE:
            await self._record_failed_attempt(username)
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )

        # Verify password
        if not bcrypt.verify(password, user.hashed_password):
            await self._record_failed_attempt(username)
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )

        # Update last login
        user.last_login = datetime.utcnow()
        await self.redis.hset(
            "users",
            user.id,
            user.json()
        )

        # Clear failed attempts
        await self.redis.delete(failed_key)

        if audit_logger and AuditEvent:
            audit_logger.log_event(AuditEvent(
                event_type="user_login_success",
                user_id=user.id, # The user who logged in
                action="User Login",
                resource_type="user_session",
                resource_id=user.id, # Could be a session ID if generated here
                status="success",
                ip_address=ip_address,
                metadata={"username": username}
            ))
        return user

    async def _record_failed_attempt(self, username: str, audit_logger: Optional[AuditLogger] = None, ip_address: Optional[str] = None):
        """Record a failed login attempt and log audit event."""
        if audit_logger and AuditEvent:
            audit_logger.log_event(AuditEvent(
                event_type="user_login_failure",
                user_id=username, # Attempted username
                action="User Login Attempt Failed",
                resource_type="user_session",
                resource_id=username,
                status="failure",
                ip_address=ip_address,
                metadata={"username": username, "reason": "Invalid credentials or inactive user"}
            ))

        key = f"auth:failed:{username}"
        await self.redis.incr(key)
        await self.redis.expire(
            key,
            timedelta(minutes=self.config.lockout_duration_minutes)
        )

    async def create_api_key(
        self,
        user_id: str, # The user for whom the key is created
        name: str,
        permissions: Optional[Set[Permission]] = None,
        expires_in_days: Optional[int] = None,
        audit_logger: Optional[AuditLogger] = None, # Added
        ip_address: Optional[str] = None, # Added - IP of the requester
        performed_by_user_id: Optional[str] = None # User ID of who is performing this action
    ) -> APIKey:
        """Create a new API key for a user"""
        # Get user for whom the key is being created
        user_data = await self.redis.hget("users", user_id)
        if not user_data:
            raise ValueError(f"User {user_id} not found")
        
        user = User.parse_raw(user_data)

        # Validate permissions
        if permissions:
            invalid_permissions = permissions - user.permissions
            if invalid_permissions:
                raise ValueError(
                    f"User does not have the following permissions: "
                    f"{invalid_permissions}"
                )
        else:
            permissions = user.permissions

        # Create API key
        api_key = APIKey(
            name=name,
            user_id=user_id,
            permissions=permissions,
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days)
            if expires_in_days else None
        )

        # Store in Redis
        await self.redis.hset(
            "api_keys",
            api_key.key,
            api_key.json()
        )

        if audit_logger and AuditEvent:
            # If performed_by_user_id is not provided, it implies the user_id themselves created it, or system default
            actor_id = performed_by_user_id if performed_by_user_id else user_id
            audit_logger.log_event(AuditEvent(
                event_type="api_key_creation",
                user_id=actor_id,
                action="API Key Created",
                resource_type="api_key",
                resource_id=api_key.key, # Or a hash of it if key is too sensitive for direct logging
                status="success",
                ip_address=ip_address,
                metadata={
                    "target_user_id": user_id,
                    "key_name": name,
                    "permissions_granted": [p.value for p in permissions] if permissions else "user_default",
                    "expires_days": expires_in_days if expires_in_days else "never"
                }
            ))
        return api_key

    async def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate an API key and return associated data"""
        # Get API key data
        key_data = await self.redis.hget("api_keys", api_key)
        if not key_data:
            return None

        key = APIKey.parse_raw(key_data)

        # Check expiration
        if key.expires_at and key.expires_at < datetime.utcnow():
            await self.redis.hdel("api_keys", api_key)
            return None

        # Update last used
        key.last_used = datetime.utcnow()
        await self.redis.hset(
            "api_keys",
            api_key,
            key.json()
        )

        return key

    def create_access_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token"""
        expires = datetime.utcnow() + (
            expires_delta
            if expires_delta
            else timedelta(minutes=self.config.token_expire_minutes)
        )

        to_encode = {
            "sub": user.id,
            "exp": expires,
            "permissions": list(user.permissions),
            "roles": user.roles
        }

        return jwt.encode(
            to_encode,
            self.config.secret_key,
            algorithm="HS256"
        )

    async def get_current_user(
        self,
        token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))
    ) -> User:
        """Get current user from JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=["HS256"]
            )
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authentication token"
                )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Could not validate token"
            )

        # Get user
        user_data = await self.redis.hget("users", user_id)
        if not user_data:
            raise HTTPException(
                status_code=401,
                detail="User not found"
            )

        return User.parse_raw(user_data)

    def check_permission(
        self,
        user: User,
        required_permission: Permission
    ) -> bool:
        """Check if user has required permission"""
        return required_permission in user.permissions

    def require_permission(self, permission: Permission):
        """Dependency for requiring specific permission"""
        async def check_permission(
            user: User = Depends(self.get_current_user)
        ):
            if not self.check_permission(user, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {permission}"
                )
            return user
        return check_permission

    async def revoke_api_key(self, api_key: str):
        """Revoke an API key"""
        await self.redis.hdel("api_keys", api_key)

    async def update_user_permissions(
        self,
        user_id: str,
        permissions: Set[Permission]
    ):
        """Update user permissions"""
        user_data = await self.redis.hget("users", user_id)
        if not user_data:
            raise ValueError(f"User {user_id} not found")

        user = User.parse_raw(user_data)
        user.permissions = permissions

        await self.redis.hset(
            "users",
            user_id,
            user.json()
        )

    @asynccontextmanager
    async def rate_limit(
        self,
        key: str,
        limit: int,
        window: int
    ):
        """Rate limiting context manager"""
        current = await self.redis.get(f"ratelimit:{key}")
        
        if current and int(current) >= limit:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )

        try:
            await self.redis.incr(f"ratelimit:{key}")
            await self.redis.expire(
                f"ratelimit:{key}",
                window
            )
            yield
        finally:
            pass

    async def cleanup_expired_tokens(self):
        """Clean up expired refresh tokens and API keys"""
        # Clean up expired API keys
        all_keys = await self.redis.hgetall("api_keys")
        now = datetime.utcnow()
        
        for key, data in all_keys.items():
            api_key = APIKey.parse_raw(data)
            if api_key.expires_at and api_key.expires_at < now:
                await self.redis.hdel("api_keys", key)
