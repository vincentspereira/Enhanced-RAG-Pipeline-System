"""
OAuth2 authentication for the API Hub.

This module provides OAuth2 authentication functionality for the API Hub,
including token management, auth flows, and provider integrations.
"""
import os
import json
import logging
import time
import requests
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
import secrets
import base64
import hashlib
import jwt
from jwt.exceptions import InvalidTokenError
from fastapi import Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2AuthorizationCodeBearer, SecurityScopes
from fastapi.responses import RedirectResponse
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OAuth2Provider:
    """Base class for OAuth2 providers."""
    
    def __init__(self, 
                 name: str,
                 client_id: str,
                 client_secret: str,
                 authorize_url: str,
                 token_url: str,
                 userinfo_url: str = None,
                 redirect_uri: str = None,
                 scopes: List[str] = None):
        """Initialize OAuth2 provider.
        
        Args:
            name: Provider name (e.g., 'github', 'google')
            client_id: OAuth client ID
            client_secret: OAuth client secret
            authorize_url: Authorization endpoint URL
            token_url: Token endpoint URL
            userinfo_url: User info endpoint URL
            redirect_uri: Redirect URI for authorization flow
            scopes: List of scopes to request
        """
        self.name = name
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorize_url = authorize_url
        self.token_url = token_url
        self.userinfo_url = userinfo_url
        self.redirect_uri = redirect_uri
        self.scopes = scopes or []
        
    def get_authorization_url(self, state: str = None, **kwargs) -> str:
        """Get authorization URL for OAuth flow.
        
        Args:
            state: State parameter for security
            **kwargs: Additional parameters for authorization URL
            
        Returns:
            Authorization URL
        """
        if state is None:
            state = secrets.token_urlsafe(32)
            
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': ' '.join(self.scopes),
            'state': state,
            'response_type': 'code',
            **kwargs
        }
        
        # Build query string
        query = '&'.join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())
        return f"{self.authorize_url}?{query}"
    
    async def exchange_code_for_token(self, code: str, **kwargs) -> Dict[str, Any]:
        """Exchange authorization code for token.
        
        Args:
            code: Authorization code from redirect
            **kwargs: Additional parameters for token request
            
        Returns:
            Token response
        """
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': code,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'authorization_code',
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.token_url, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Token exchange error: {error_text}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"Failed to exchange code for token: {error_text}"
                    )
                
                return await response.json()
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New token response
        """
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.token_url, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Token refresh error: {error_text}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"Failed to refresh token: {error_text}"
                    )
                
                return await response.json()
    
    async def get_user_info(self, token: str) -> Dict[str, Any]:
        """Get user information from provider.
        
        Args:
            token: Access token
            
        Returns:
            User information
        """
        if not self.userinfo_url:
            raise ValueError(f"User info URL not configured for provider {self.name}")
        
        headers = {'Authorization': f"Bearer {token}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.userinfo_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"User info error: {error_text}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"Failed to get user info: {error_text}"
                    )
                
                return await response.json()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert provider to dictionary.
        
        Returns:
            Provider as dictionary
        """
        return {
            'name': self.name,
            'client_id': self.client_id,
            'authorize_url': self.authorize_url,
            'token_url': self.token_url,
            'userinfo_url': self.userinfo_url,
            'redirect_uri': self.redirect_uri,
            'scopes': self.scopes
        }


class GithubOAuth2Provider(OAuth2Provider):
    """GitHub OAuth2 provider."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, scopes: List[str] = None):
        """Initialize GitHub OAuth2 provider.
        
        Args:
            client_id: GitHub OAuth client ID
            client_secret: GitHub OAuth client secret
            redirect_uri: Redirect URI for authorization flow
            scopes: List of scopes to request
        """
        super().__init__(
            name='github',
            client_id=client_id,
            client_secret=client_secret,
            authorize_url='https://github.com/login/oauth/authorize',
            token_url='https://github.com/login/oauth/access_token',
            userinfo_url='https://api.github.com/user',
            redirect_uri=redirect_uri,
            scopes=scopes or ['read:user', 'user:email']
        )
        
    async def get_user_info(self, token: str) -> Dict[str, Any]:
        """Get user information from GitHub.
        
        Args:
            token: Access token
            
        Returns:
            User information
        """
        # Get basic user info
        headers = {
            'Authorization': f"Bearer {token}",
            'Accept': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            # Get user profile
            async with session.get(self.userinfo_url, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"GitHub user info error: {error_text}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"Failed to get GitHub user info: {error_text}"
                    )
                
                user_data = await response.json()
                
            # Get user emails
            if 'user:email' in self.scopes:
                async with session.get('https://api.github.com/user/emails', headers=headers) as response:
                    if response.status == 200:
                        emails = await response.json()
                        primary_email = next((email for email in emails if email.get('primary')), emails[0] if emails else None)
                        if primary_email:
                            user_data['email'] = primary_email.get('email')
        
        return user_data


class GoogleOAuth2Provider(OAuth2Provider):
    """Google OAuth2 provider."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, scopes: List[str] = None):
        """Initialize Google OAuth2 provider.
        
        Args:
            client_id: Google OAuth client ID
            client_secret: Google OAuth client secret
            redirect_uri: Redirect URI for authorization flow
            scopes: List of scopes to request
        """
        super().__init__(
            name='google',
            client_id=client_id,
            client_secret=client_secret,
            authorize_url='https://accounts.google.com/o/oauth2/auth',
            token_url='https://oauth2.googleapis.com/token',
            userinfo_url='https://www.googleapis.com/oauth2/v3/userinfo',
            redirect_uri=redirect_uri,
            scopes=scopes or ['openid', 'email', 'profile']
        )
        
    def get_authorization_url(self, state: str = None, **kwargs) -> str:
        """Get Google authorization URL with additional parameters.
        
        Args:
            state: State parameter for security
            **kwargs: Additional parameters for authorization URL
            
        Returns:
            Authorization URL
        """
        # Add additional Google-specific parameters
        return super().get_authorization_url(
            state=state,
            access_type='offline',  # Get refresh token
            prompt='consent',  # Force consent screen
            **kwargs
        )


class OAuth2Manager:
    """Manager for OAuth2 providers and tokens."""
    
    def __init__(self, token_dir: str = "data/oauth_tokens"):
        """Initialize OAuth2 manager.
        
        Args:
            token_dir: Directory for storing tokens
        """
        self.token_dir = token_dir
        os.makedirs(token_dir, exist_ok=True)
        
        # JWT settings for internal tokens
        self.jwt_secret = os.environ.get('JWT_SECRET') or secrets.token_hex(32)
        self.jwt_algorithm = 'HS256'
        self.jwt_expires = 24 * 60 * 60  # 24 hours in seconds
        
        # Registered providers
        self.providers: Dict[str, OAuth2Provider] = {}
        
    def register_provider(self, provider: OAuth2Provider):
        """Register an OAuth2 provider.
        
        Args:
            provider: OAuth2 provider instance
        """
        self.providers[provider.name] = provider
        logger.info(f"Registered OAuth2 provider: {provider.name}")
        
    def get_provider(self, name: str) -> Optional[OAuth2Provider]:
        """Get OAuth2 provider by name.
        
        Args:
            name: Provider name
            
        Returns:
            OAuth2Provider instance or None if not found
        """
        return self.providers.get(name)
    
    def list_providers(self) -> List[Dict[str, Any]]:
        """List all registered providers.
        
        Returns:
            List of provider dictionaries
        """
        return [provider.to_dict() for provider in self.providers.values()]
    
    async def handle_callback(self, provider_name: str, code: str, state: str = None) -> Dict[str, Any]:
        """Handle OAuth2 callback.
        
        Args:
            provider_name: Provider name
            code: Authorization code
            state: State parameter
            
        Returns:
            User information and tokens
        """
        provider = self.get_provider(provider_name)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown provider: {provider_name}"
            )
        
        # Exchange code for token
        token_response = await provider.exchange_code_for_token(code)
        access_token = token_response.get('access_token')
        
        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Failed to get access token"
            )
        
        # Get user info
        user_info = await provider.get_user_info(access_token)
        
        # Save tokens
        self._save_oauth_tokens(
            provider_name=provider_name,
            user_id=str(user_info.get('id') or user_info.get('sub')),
            tokens=token_response
        )
        
        # Generate JWT token
        jwt_token = self.create_jwt_token(
            provider=provider_name,
            user_id=str(user_info.get('id') or user_info.get('sub')),
            user_info=user_info
        )
        
        return {
            'user': user_info,
            'provider': provider_name,
            'token': jwt_token
        }
    
    def _save_oauth_tokens(self, provider_name: str, user_id: str, tokens: Dict[str, Any]):
        """Save OAuth2 tokens.
        
        Args:
            provider_name: Provider name
            user_id: User ID
            tokens: Token data
        """
        # Create tokens directory if it doesn't exist
        provider_dir = os.path.join(self.token_dir, provider_name)
        os.makedirs(provider_dir, exist_ok=True)
        
        # Add timestamp
        tokens['timestamp'] = datetime.now().isoformat()
        
        # Save tokens
        token_path = os.path.join(provider_dir, f"{user_id}.json")
        with open(token_path, 'w') as f:
            json.dump(tokens, f, indent=2)
    
    def get_oauth_tokens(self, provider_name: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get OAuth2 tokens.
        
        Args:
            provider_name: Provider name
            user_id: User ID
            
        Returns:
            Token data or None if not found
        """
        token_path = os.path.join(self.token_dir, provider_name, f"{user_id}.json")
        
        if not os.path.exists(token_path):
            return None
            
        try:
            with open(token_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading tokens: {e}")
            return None
    
    def create_jwt_token(self, provider: str, user_id: str, user_info: Dict[str, Any]) -> str:
        """Create JWT token.
        
        Args:
            provider: Provider name
            user_id: User ID
            user_info: User information
            
        Returns:
            JWT token
        """
        now = datetime.utcnow()
        expiry = now + timedelta(seconds=self.jwt_expires)
        
        payload = {
            'sub': user_id,
            'provider': provider,
            'name': user_info.get('name', ''),
            'email': user_info.get('email', ''),
            'iat': now,
            'exp': expiry
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Token payload
            
        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}"
            )
    
    async def refresh_oauth_token(self, provider_name: str, user_id: str) -> Dict[str, Any]:
        """Refresh OAuth2 token.
        
        Args:
            provider_name: Provider name
            user_id: User ID
            
        Returns:
            New token data
        """
        provider = self.get_provider(provider_name)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown provider: {provider_name}"
            )
        
        # Get existing tokens
        tokens = self.get_oauth_tokens(provider_name, user_id)
        if not tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No tokens found"
            )
        
        refresh_token = tokens.get('refresh_token')
        if not refresh_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No refresh token found"
            )
        
        # Refresh token
        new_tokens = await provider.refresh_token(refresh_token)
        
        # If refresh token is not included in response, keep the old one
        if 'refresh_token' not in new_tokens:
            new_tokens['refresh_token'] = refresh_token
        
        # Save new tokens
        self._save_oauth_tokens(provider_name, user_id, new_tokens)
        
        return new_tokens


# OAuth2 security scheme for FastAPI
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="",  # Will be set dynamically
    tokenUrl="token",
)

# OAuth2 manager instance
_oauth2_manager = None

def get_oauth2_manager() -> OAuth2Manager:
    """Get OAuth2 manager instance.
    
    Returns:
        OAuth2Manager instance
    """
    global _oauth2_manager
    if _oauth2_manager is None:
        _oauth2_manager = OAuth2Manager()
    return _oauth2_manager


async def get_current_user(security_scopes: SecurityScopes, token: str = Depends(oauth2_scheme)):
    """Get current user from token.
    
    Args:
        security_scopes: Security scopes
        token: JWT token
        
    Returns:
        User information
    """
    oauth2_manager = get_oauth2_manager()
    
    try:
        payload = oauth2_manager.validate_jwt_token(token)
        
        # Check if token has expired
        if datetime.utcnow() > datetime.fromtimestamp(payload['exp']):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        
        # Get provider and user ID
        provider = payload.get('provider')
        user_id = payload.get('sub')
        
        if not provider or not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Check if user has necessary scopes
        if security_scopes.scopes:
            # You can implement scope checking here
            pass
        
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {str(e)}"
        )
