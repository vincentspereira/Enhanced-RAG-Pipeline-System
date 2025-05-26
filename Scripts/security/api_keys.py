from typing import Optional, List, Dict
from datetime import datetime, timedelta
import uuid
import secrets
import hashlib
from pydantic import BaseModel
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Session, relationship

from .rbac import Base, User

class APIKey(Base):
    __tablename__ = 'api_keys'
    
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    key_hash = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    user_id = Column(UUID, ForeignKey('users.id'))
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime, nullable=True)
    use_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    key_metadata = Column(JSONB)
    
    user = relationship('User')

class APIKeyCreate(BaseModel):
    name: str
    user_id: uuid.UUID
    expires_in_days: Optional[int] = None
    key_metadata: Optional[Dict] = None

class APIKeyResponse(BaseModel):
    id: uuid.UUID
    key: str  # Only returned once upon creation
    name: str
    expires_at: Optional[datetime]
    created_at: datetime
    key_metadata: Optional[Dict]

class APIKeyManager:
    def __init__(self, rbac_manager):
        self.rbac_manager = rbac_manager
        self.engine = rbac_manager.engine
        Base.metadata.create_all(self.engine)
    
    def create_api_key(self, data: APIKeyCreate) -> APIKeyResponse:
        """Create a new API key for a user."""
        # Generate a secure random key
        key = f"sk_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(key)
        
        with Session(self.engine) as session:
            # Calculate expiration if specified
            expires_at = None
            if data.expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=data.expires_in_days)
            
            # Create API key record
            api_key = APIKey(
                key_hash=key_hash,
                name=data.name,
                user_id=data.user_id,
                expires_at=expires_at,
                key_metadata=data.key_metadata
            )
            
            session.add(api_key)
            session.commit()
            
            return APIKeyResponse(
                id=api_key.id,
                key=key,  # Only time the raw key is returned
                name=api_key.name,
                expires_at=api_key.expires_at,
                created_at=api_key.created_at,
                key_metadata=api_key.key_metadata
            )
    
    def validate_api_key(self, key: str) -> Optional[uuid.UUID]:
        """Validate an API key and return the associated user ID if valid."""
        key_hash = self._hash_key(key)
        
        with Session(self.engine) as session:
            api_key = session.query(APIKey).filter_by(
                key_hash=key_hash,
                is_active=True
            ).first()
            
            if not api_key:
                return None
            
            # Check expiration
            if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                return None
            
            # Update usage statistics
            api_key.last_used_at = datetime.utcnow()
            api_key.use_count += 1
            session.commit()
            
            return api_key.user_id
    
    def get_user_api_keys(self, user_id: uuid.UUID) -> List[Dict]:
        """Get all API keys for a user (excluding the actual keys)."""
        with Session(self.engine) as session:
            api_keys = session.query(APIKey).filter_by(user_id=user_id).all()
            
            return [
                {
                    'id': key.id,
                    'name': key.name,
                    'expires_at': key.expires_at,
                    'is_active': key.is_active,
                    'last_used_at': key.last_used_at,
                    'use_count': key.use_count,
                    'created_at': key.created_at,
                    'key_metadata': key.key_metadata
                }
                for key in api_keys
            ]
    
    def revoke_api_key(self, key_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Revoke an API key."""
        with Session(self.engine) as session:
            api_key = session.query(APIKey).filter_by(
                id=key_id,
                user_id=user_id
            ).first()
            
            if not api_key:
                return False
            
            api_key.is_active = False
            session.commit()
            return True
    
    def _hash_key(self, key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def rotate_api_key(self, key_id: uuid.UUID, user_id: uuid.UUID) -> Optional[APIKeyResponse]:
        """Rotate an API key while preserving its settings."""
        with Session(self.engine) as session:
            old_key = session.query(APIKey).filter_by(
                id=key_id,
                user_id=user_id,
                is_active=True
            ).first()
            
            if not old_key:
                return None
            
            # Create new key with same settings
            new_key = self.create_api_key(APIKeyCreate(
                name=f"{old_key.name} (rotated)",
                user_id=user_id,
                expires_in_days=(
                    (old_key.expires_at - datetime.utcnow()).days
                    if old_key.expires_at
                    else None
                ),
                key_metadata=old_key.key_metadata
            ))
            
            # Revoke old key
            old_key.is_active = False
            session.commit()
            
            return new_key
