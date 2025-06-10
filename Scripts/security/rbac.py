from enum import Enum
from typing import List, Dict, Optional, Set
from pydantic import BaseModel, EmailStr
from datetime import datetime
import uuid
import bcrypt
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID # Keep UUID for postgresql compatibility if needed elsewhere
from sqlalchemy import JSON # Use generic JSON for broader compatibility

Base = declarative_base()

# Role-Permission Association Table
role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', UUID, ForeignKey('roles.id')),
    Column('permission_id', UUID, ForeignKey('permissions.id'))
)

class Permission(Base):
    __tablename__ = 'permissions'
    
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    description = Column(String)
    resource = Column(String, nullable=False)  # e.g., 'documents', 'users', 'analytics'
    action = Column(String, nullable=False)    # e.g., 'read', 'write', 'delete'
    conditions = Column(JSON)                 # Optional conditions for fine-grained control
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Role(Base):
    __tablename__ = 'roles'
    
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    description = Column(String)
    is_system_role = Column(Boolean, default=False)  # True for built-in roles
    permissions = relationship('Permission', secondary=role_permissions)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class User(Base):
    __tablename__ = 'users'
    
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role_id = Column(UUID, ForeignKey('roles.id'))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    role = relationship('Role')

# Pydantic models for API
class PermissionCreate(BaseModel):
    name: str
    description: Optional[str]
    resource: str
    action: str
    conditions: Optional[Dict] = None

class RoleCreate(BaseModel):
    name: str
    description: Optional[str]
    permission_ids: List[uuid.UUID]

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    role_id: uuid.UUID

class RBACManager:
    def __init__(self, engine): # Accept engine directly
        self.engine = engine
        # Base.metadata.create_all(self.engine) # Defer this to be called once
        # self._initialize_default_roles() # Defer this

    def create_tables_if_needed(self):
        """Creates tables if they don't exist. Should be called once."""
        Base.metadata.create_all(self.engine)

    def initialize_default_roles_if_needed(self):
        """Initialize default roles and permissions. Should be called once."""
        with Session(self.engine) as session:
            if session.query(Role).filter_by(name='admin').first(): # Check if already initialized
                return

            # Create default permissions
            default_permissions = [
                Permission(
                    name='read_documents',
                    description='Read access to documents',
                    resource='documents',
                    action='read'
                ),
                Permission(
                    name='write_documents',
                    description='Write access to documents',
                    resource='documents',
                    action='write'
                ),
                Permission(
                    name='manage_users',
                    description='Manage user accounts',
                    resource='users',
                    action='manage'
                ),
                # Add more default permissions as needed
            ]
            
            # Create default roles
            admin_role = Role(
                name='admin',
                description='Administrator with full access',
                is_system_role=True
            )
            user_role = Role(
                name='user',
                description='Regular user with basic access',
                is_system_role=True
            )
            readonly_role = Role(
                name='readonly',
                description='Read-only access to documents',
                is_system_role=True
            )
            
            # Add permissions to roles
            admin_role.permissions = default_permissions
            user_role.permissions = [p for p in default_permissions 
                                  if p.name in ['read_documents', 'write_documents']]
            readonly_role.permissions = [p for p in default_permissions 
                                      if p.name == 'read_documents']
            
            # Save to database if they don't exist
            for perm in default_permissions:
                if not session.query(Permission).filter_by(name=perm.name).first():
                    session.add(perm)
            
            for role in [admin_role, user_role, readonly_role]:
                if not session.query(Role).filter_by(name=role.name).first():
                    session.add(role)
            
            session.commit()
    
    def create_user(self, user_data: UserCreate, session: Optional[Session] = None) -> User:
        """Create a new user with the specified role. Uses provided session if available."""
        if session:
            return self._create_user_with_session(user_data, session)
        else:
            with Session(self.engine) as new_session:
                return self._create_user_with_session(user_data, new_session)

    def _create_user_with_session(self, user_data: UserCreate, session: Session) -> User:
        # Hash password
        password_hash = bcrypt.hashpw(
            user_data.password.encode(), bcrypt.gensalt()
        ).decode()

        user = User( # Corrected indentation
            username=user_data.username,
            email=user_data.email,
            password_hash=password_hash,
            role_id=user_data.role_id
        )
        session.add(user)
        session.commit() # Commit if we created the session, or let caller commit if session was passed
        return user

    def check_permission(self, user_id: uuid.UUID, resource: str, action: str, session: Optional[Session] = None) -> bool:
        """Check if a user has permission. Uses provided session if available."""
        if session:
            return self._check_permission_with_session(user_id, resource, action, session)
        else:
            with Session(self.engine) as new_session:
                return self._check_permission_with_session(user_id, resource, action, new_session)

    def _check_permission_with_session(self, user_id: uuid.UUID, resource: str, action: str, session: Session) -> bool:
        user = session.query(User).filter_by(id=user_id).first()
        if not user or not user.is_active:
            return False

        # Get all permissions for the user's role (Corrected indentation)
        permissions = user.role.permissions

        # Check if any permission matches the requested access
        return any(
            p.resource == resource and p.action == action
            for p in permissions
        )

    def get_user_permissions(self, user_id: uuid.UUID, session: Optional[Session] = None) -> List[Dict]:
        """Get all permissions for a user. Uses provided session if available."""
        if session:
            return self._get_user_permissions_with_session(user_id, session)
        else:
            with Session(self.engine) as new_session:
                return self._get_user_permissions_with_session(user_id, new_session)

    def _get_user_permissions_with_session(self, user_id: uuid.UUID, session: Session) -> List[Dict]:
        user = session.query(User).filter_by(id=user_id).first()
        if not user:
            return []

        # Corrected indentation
        return [
            {
                'name': p.name,
                    'resource': p.resource,
                    'action': p.action,
                    'conditions': p.conditions
                }
                for p in user.role.permissions
            ]
