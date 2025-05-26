"""
Authentication and authorization for the API Gateway.

This module provides OAuth2 authentication and authorization for the API Gateway,
including token management and user authentication.
"""
import os
import json
import logging
import time
import jwt
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from pathlib import Path
import secrets
import hashlib
import base64
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# User model
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    scopes: List[str] = []
    hashed_password: str = None
    api_keys: List[Dict[str, str]] = []

class UserManager:
    """Manager for user authentication and authorization."""
    
    def __init__(self, users_file: str = "data/users.json", secret_key: str = None):
        """Initialize the user manager.
        
        Args:
            users_file: Path to users data file
            secret_key: Secret key for JWT tokens
        """
        self.users_file = Path(users_file)
        self.users_dir = self.users_file.parent
        self.users_dir.mkdir(parents=True, exist_ok=True)
        
        # Create users file if it doesn't exist
        if not self.users_file.exists():
            with open(self.users_file, "w") as f:
                json.dump({"users": []}, f)
        
        # Load users
        self.users = self._load_users()
        
        # Generate secret key if not provided
        if secret_key:
            self.secret_key = secret_key
        else:
            key_file = self.users_dir / "jwt_secret.key"
            if key_file.exists():
                with open(key_file, "r") as f:
                    self.secret_key = f.read().strip()
            else:
                self.secret_key = secrets.token_hex(32)
                with open(key_file, "w") as f:
                    f.write(self.secret_key)
        
        # Create encryption key for sensitive data
        self.crypto_key = self._get_crypto_key()
    
    def _get_crypto_key(self):
        """Get or create encryption key for sensitive data."""
        key_file = self.users_dir / "crypto.key"
        if key_file.exists():
            with open(key_file, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            return key
    
    def _load_users(self):
        """Load users from file.
        
        Returns:
            Dictionary of users by username
        """
        try:
            with open(self.users_file, "r") as f:
                data = json.load(f)
            
            users = {}
            for user_data in data.get("users", []):
                user = User(**user_data)
                users[user.username] = user
            
            return users
        
        except Exception as e:
            logger.error(f"Failed to load users: {e}")
            return {}
    
    def _save_users(self):
        """Save users to file."""
        try:
            data = {"users": [user.dict() for user in self.users.values()]}
            
            with open(self.users_file, "w") as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save users: {e}")
    
    def get_user(self, username: str):
        """Get a user by username.
        
        Args:
            username: Username
            
        Returns:
            User if found, None otherwise
        """
        return self.users.get(username)
    
    def verify_password(self, plain_password: str, hashed_password: str):
        """Verify a password.
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password is correct, False otherwise
        """
        # In a real app, use a proper password hashing library
        password_hash = hashlib.sha256(plain_password.encode()).hexdigest()
        return password_hash == hashed_password
    
    def get_password_hash(self, password: str):
        """Hash a password.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        # In a real app, use a proper password hashing library
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate_user(self, username: str, password: str):
        """Authenticate a user.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            User if authentication successful, None otherwise
        """
        user = self.get_user(username)
        if not user:
            return None
        
        if not self.verify_password(password, user.hashed_password):
            return None
        
        if user.disabled:
            return None
        
        return user
    
    def create_user(
        self, 
        username: str,
        password: str,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        scopes: List[str] = ["read"]
    ):
        """Create a new user.
        
        Args:
            username: Username
            password: Plain text password
            email: Optional email
            full_name: Optional full name
            scopes: List of scopes
            
        Returns:
            Created user
        """
        if username in self.users:
            raise ValueError(f"User '{username}' already exists")
        
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            scopes=scopes,
            hashed_password=self.get_password_hash(password)
        )
        
        self.users[username] = user
        self._save_users()
        
        return user
    
    def update_user(
        self, 
        username: str,
        email: Optional[str] = None,
        full_name: Optional[str] = None,
        disabled: Optional[bool] = None,
        scopes: Optional[List[str]] = None
    ):
        """Update a user.
        
        Args:
            username: Username
            email: Optional email
            full_name: Optional full name
            disabled: Optional disabled flag
            scopes: Optional list of scopes
            
        Returns:
            Updated user
        """
        user = self.get_user(username)
        if not user:
            raise ValueError(f"User '{username}' not found")
        
        if email is not None:
            user.email = email
        
        if full_name is not None:
            user.full_name = full_name
        
        if disabled is not None:
            user.disabled = disabled
        
        if scopes is not None:
            user.scopes = scopes
        
        self._save_users()
        
        return user
    
    def delete_user(self, username: str):
        """Delete a user.
        
        Args:
            username: Username
            
        Returns:
            True if user was deleted, False otherwise
        """
        if username not in self.users:
            return False
        
        del self.users[username]
        self._save_users()
        
        return True
    
    def change_password(self, username: str, password: str):
        """Change a user's password.
        
        Args:
            username: Username
            password: New plain text password
            
        Returns:
            True if password was changed, False otherwise
        """
        user = self.get_user(username)
        if not user:
            return False
        
        user.hashed_password = self.get_password_hash(password)
        self._save_users()
        
        return True
    
    def create_api_key(self, username: str, name: str = None, scopes: List[str] = None):
        """Create an API key for a user.
        
        Args:
            username: Username
            name: Optional name for the API key
            scopes: Optional list of scopes
            
        Returns:
            API key
        """
        user = self.get_user(username)
        if not user:
            raise ValueError(f"User '{username}' not found")
        
        # Generate API key
        api_key = secrets.token_hex(16)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Use user's scopes if not specified
        if scopes is None:
            scopes = user.scopes
        
        # Add API key to user
        key_data = {
            "name": name or f"Key {len(user.api_keys) + 1}",
            "hash": key_hash,
            "scopes": scopes,
            "created_at": datetime.now().isoformat()
        }
        
        if not user.api_keys:
            user.api_keys = []
        
        user.api_keys.append(key_data)
        self._save_users()
        
        return api_key
    
    def validate_api_key(self, api_key: str):
        """Validate an API key.
        
        Args:
            api_key: API key
            
        Returns:
            User if API key is valid, None otherwise
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        for username, user in self.users.items():
            for key_data in user.api_keys:
                if key_data.get("hash") == key_hash:
                    return user
        
        return None
    
    def create_access_token(
        self, 
        username: str,
        scopes: List[str] = None,
        expires_delta: timedelta = None
    ):
        """Create an access token.
        
        Args:
            username: Username
            scopes: Optional list of scopes
            expires_delta: Optional expiration time
            
        Returns:
            Access token
        """
        user = self.get_user(username)
        if not user:
            raise ValueError(f"User '{username}' not found")
        
        # Use user's scopes if not specified
        if scopes is None:
            scopes = user.scopes
        
        # Set expiration time
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        # Create token data
        token_data = {
            "sub": username,
            "exp": expire,
            "scopes": scopes
        }
        
        # Create JWT token
        encoded_jwt = jwt.encode(token_data, self.secret_key, algorithm="HS256")
        
        return encoded_jwt
    
    def validate_token(self, token: str, required_scopes: List[str] = None):
        """Validate an access token.
        
        Args:
            token: Access token
            required_scopes: Optional list of required scopes
            
        Returns:
            User if token is valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            username = payload.get("sub")
            
            if not username:
                return None
            
            user = self.get_user(username)
            if not user or user.disabled:
                return None
            
            # Check scopes
            token_scopes = payload.get("scopes", [])
            if required_scopes:
                for scope in required_scopes:
                    if scope not in token_scopes:
                        return None
            
            return user
        
        except jwt.PyJWTError:
            return None


# OAuth2 token model
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    scope: str

# OAuth2 configuration
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="auth/token",
    scopes={
        "admin": "Full access to API Gateway",
        "read": "Read-only access to API Gateway",
        "write": "Write access to API Gateway",
        "service": "Access to specific services"
    }
)

class OAuth2Provider:
    """OAuth2 authentication provider."""
    
    def __init__(self, user_manager: UserManager):
        """Initialize the OAuth2 provider.
        
        Args:
            user_manager: User manager instance
        """
        self.user_manager = user_manager
    
    async def get_current_user(
        self, 
        security_scopes: SecurityScopes,
        token: str = Depends(oauth2_scheme)
    ):
        """Get the current user from an OAuth token.
        
        Args:
            security_scopes: Security scopes
            token: OAuth token
            
        Returns:
            User if token is valid
        """
        # Define required scopes
        if security_scopes.scopes:
            authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
        else:
            authenticate_value = "Bearer"
        
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": authenticate_value},
        )
        
        # Validate token
        user = self.user_manager.validate_token(token, security_scopes.scopes)
        if not user:
            raise credentials_exception
        
        return user
    
    def create_access_token(
        self, 
        username: str,
        scopes: List[str] = None,
        expires_delta: timedelta = None
    ):
        """Create an access token.
        
        Args:
            username: Username
            scopes: Optional list of scopes
            expires_delta: Optional expiration time
            
        Returns:
            Token model
        """
        # Default expiration time
        if not expires_delta:
            expires_delta = timedelta(minutes=30)
        
        # Create token
        access_token = self.user_manager.create_access_token(
            username=username,
            scopes=scopes,
            expires_delta=expires_delta
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=expires_delta.total_seconds(),
            scope=" ".join(scopes) if scopes else ""
        )
