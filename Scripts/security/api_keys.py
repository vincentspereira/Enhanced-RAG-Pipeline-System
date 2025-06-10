from typing import Optional, List, Dict
from datetime import datetime, timedelta
import uuid
import secrets
import hashlib
from pydantic import BaseModel
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID # Keep UUID
from sqlalchemy.orm import Session, relationship

from .rbac import Base, User, RBACManager # Added RBACManager

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
    key_metadata = Column(JSON)
    
    user = relationship('User')

class APIKeyCreate(BaseModel):
    name: str
    user_id: uuid.UUID
    expires_in_days: Optional[int] = None
    key_metadata: Optional[Dict] = None

class APIKeyResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID # Added user_id
    key: str  # Only returned once upon creation
    name: str
    expires_at: Optional[datetime]
    created_at: datetime
    key_metadata: Optional[Dict]

class APIKeyManager:
    def __init__(self, rbac_manager: RBACManager): # Type hint for clarity
        self.rbac_manager = rbac_manager
        self.engine = rbac_manager.engine
        # Base.metadata.create_all(self.engine) # Defer, should be handled by RBACManager or once globally

    def create_tables_if_needed(self): # Added for consistency, though APIKey uses RBAC's Base
        """Creates tables if they don't exist. Should be called once if APIKey has own tables or RBAC didn't cover it."""
        Base.metadata.create_all(self.engine) # APIKey model uses Base from rbac.py

    def create_api_key(self, data: APIKeyCreate, session: Optional[Session] = None) -> APIKeyResponse:
        """Create a new API key for a user. Uses provided session if available."""
        if session:
            return self._create_api_key_with_session(data, session)
        else:
            with Session(self.engine) as new_session:
                return self._create_api_key_with_session(data, new_session)

    def _create_api_key_with_session(self, data: APIKeyCreate, session: Session) -> APIKeyResponse:
        # Generate a secure random key
        key = f"sk_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(key)
        
        # Calculate expiration if specified
        expires_at = None
        if data.expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=data.expires_in_days)

        # Create API key record (Corrected Indentation)
        api_key = APIKey(
            key_hash=key_hash,
            name=data.name,
            user_id=data.user_id,
            expires_at=expires_at,
            key_metadata=data.key_metadata
        )

        session.add(api_key)
        session.commit() # Commit if we created the session, or let caller commit

        return APIKeyResponse(
                id=api_key.id,
                user_id=api_key.user_id, # Added user_id
                key=key,  # Only time the raw key is returned
                name=api_key.name,
                expires_at=api_key.expires_at,
                created_at=api_key.created_at,
                key_metadata=api_key.key_metadata
            )
    
    def validate_api_key(self, key: str, session: Optional[Session] = None) -> Optional[uuid.UUID]:
        """Validate an API key. Uses provided session if available."""
        if session:
            return self._validate_api_key_with_session(key, session)
        else:
            with Session(self.engine) as new_session:
                return self._validate_api_key_with_session(key, new_session)

    def _validate_api_key_with_session(self, key: str, session: Session) -> Optional[uuid.UUID]:
        key_hash = self._hash_key(key)
        
        api_key = session.query(APIKey).filter_by(
            key_hash=key_hash,
            is_active=True
        ).first()

        # Corrected Indentation for the rest of the method
        if not api_key:
            return None

        # Check expiration
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return None

        # Update usage statistics
        api_key.last_used_at = datetime.utcnow()
        api_key.use_count += 1
        session.commit() # Commit if we created the session

        return api_key.user_id

    def get_user_api_keys(self, user_id: uuid.UUID, session: Optional[Session] = None) -> List[Dict]:
        """Get all API keys for a user. Uses provided session if available."""
        if session:
            return self._get_user_api_keys_with_session(user_id, session)
        else:
            with Session(self.engine) as new_session:
                return self._get_user_api_keys_with_session(user_id, new_session)

    def _get_user_api_keys_with_session(self, user_id: uuid.UUID, session: Session) -> List[Dict]:
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
    
    def revoke_api_key(self, key_id: uuid.UUID, user_id: uuid.UUID, session: Optional[Session] = None) -> bool:
        """Revoke an API key. Uses provided session if available."""
        if session:
            return self._revoke_api_key_with_session(key_id, user_id, session)
        else:
            with Session(self.engine) as new_session:
                return self._revoke_api_key_with_session(key_id, user_id, new_session)

    def _revoke_api_key_with_session(self, key_id: uuid.UUID, user_id: uuid.UUID, session: Session) -> bool:
        api_key = session.query(APIKey).filter_by(
            id=key_id,
            user_id=user_id
        ).first()

        # Corrected Indentation for the rest of the method
        if not api_key:
            return False

        api_key.is_active = False
        session.commit() # Commit if we created the session
        return True
    
    def _hash_key(self, key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def rotate_api_key(self, key_id: uuid.UUID, user_id: uuid.UUID, session: Optional[Session] = None) -> Optional[APIKeyResponse]:
        """Rotate an API key. Uses provided session if available."""
        if session:
            return self._rotate_api_key_with_session(key_id, user_id, session)
        else:
            with Session(self.engine) as new_session:
                return self._rotate_api_key_with_session(key_id, user_id, new_session)

    def _rotate_api_key_with_session(self, key_id: uuid.UUID, user_id: uuid.UUID, session: Session) -> Optional[APIKeyResponse]:
        old_key = session.query(APIKey).filter_by(
            id=key_id,
            user_id=user_id,
            is_active=True
        ).first()

        # Corrected Indentation for the rest of the method
        if not old_key:
            return None

        # Create new key with same settings
        # Pass the current session to ensure atomicity if create_api_key uses it
        new_key_response = self.create_api_key(APIKeyCreate(
            name=f"{old_key.name} (rotated)",
            user_id=user_id,
            expires_in_days=(
                (old_key.expires_at - datetime.utcnow()).days
                if old_key.expires_at
                else None
            ),
            key_metadata=old_key.key_metadata
        ), session=session) # Pass session here

        # Revoke old key
        old_key.is_active = False
        # The commit for new_key creation (if session was passed) and old_key revocation
        # should ideally be handled together. If create_api_key committed using the passed session,
        # this commit is fine. Otherwise, this might be a separate transaction.
        # For now, assuming create_api_key handles its commit with the passed session.
        session.commit()

        return new_key_response
