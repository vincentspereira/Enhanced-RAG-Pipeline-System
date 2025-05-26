from enum import Enum
from typing import List, Dict, Optional, Set
from pydantic import BaseModel, EmailStr
from datetime import datetime
import uuid
import bcrypt
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB

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
    conditions = Column(JSONB)                 # Optional conditions for fine-grained control
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
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize default roles and permissions."""
        with Session(self.engine) as session:
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
    
    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user with the specified role."""
        with Session(self.engine) as session:
            # Hash password
            password_hash = bcrypt.hashpw(
                user_data.password.encode(), bcrypt.gensalt()
            ).decode()
            
            user = User(
                username=user_data.username,
                email=user_data.email,
                password_hash=password_hash,
                role_id=user_data.role_id
            )
            session.add(user)
            session.commit()
            return user
    
    def check_permission(self, user_id: uuid.UUID, resource: str, action: str) -> bool:
        """Check if a user has permission to perform an action on a resource."""
        with Session(self.engine) as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user or not user.is_active:
                return False
            
            # Get all permissions for the user's role
            permissions = user.role.permissions
            
            # Check if any permission matches the requested access
            return any(
                p.resource == resource and p.action == action
                for p in permissions
            )
    
    def get_user_permissions(self, user_id: uuid.UUID) -> List[Dict]:
        """Get all permissions for a user."""
        with Session(self.engine) as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                return []
            
            return [
                {
                    'name': p.name,
                    'resource': p.resource,
                    'action': p.action,
                    'conditions': p.conditions
                }
                for p in user.role.permissions
            ]
