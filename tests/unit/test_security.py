import pytest
import uuid
from datetime import datetime, timedelta
from Scripts.security.rbac import RBACManager, User, Role, Permission
from Scripts.security.api_keys import APIKeyManager
from Scripts.security.audit_logger import AuditLogger
from Scripts.security.encryption import DataEncryption
from Scripts.security.secure_storage import SecureStorage

@pytest.mark.asyncio
class TestRBAC:
    async def test_create_user(self, db_session, test_config):
        rbac = RBACManager(test_config["database_url"])
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "TestPass123!",
            "role_id": uuid.uuid4()  # We'll need to create a role first
        }
        
        # Create a role first
        with db_session as session:
            role = Role(name="test_role", description="Test Role")
            session.add(role)
            session.commit()
            user_data["role_id"] = role.id
        
        user = rbac.create_user(user_data)
        assert user.username == user_data["username"]
        assert user.email == user_data["email"]
        assert user.role_id == user_data["role_id"]

    async def test_check_permission(self, db_session, test_config):
        rbac = RBACManager(test_config["database_url"])
        
        # Create role with permission
        with db_session as session:
            permission = Permission(
                name="test_permission",
                resource="test_resource",
                action="read"
            )
            role = Role(
                name="test_role",
                permissions=[permission]
            )
            session.add(role)
            session.commit()
            
            user = User(
                username="testuser",
                email="test@example.com",
                password_hash="hash",
                role_id=role.id
            )
            session.add(user)
            session.commit()
            
            # Test permission check
            has_permission = rbac.check_permission(
                user.id, "test_resource", "read"
            )
            assert has_permission is True
            
            # Test non-existent permission
            has_permission = rbac.check_permission(
                user.id, "test_resource", "write"
            )
            assert has_permission is False

@pytest.mark.asyncio
class TestAPIKeys:
    async def test_create_api_key(self, db_session, test_config):
        rbac = RBACManager(test_config["database_url"])
        api_manager = APIKeyManager(rbac)
        
        # Create test user
        with db_session as session:
            role = Role(name="test_role")
            session.add(role)
            session.commit()
            
            user = User(
                username="testuser",
                email="test@example.com",
                password_hash="hash",
                role_id=role.id
            )
            session.add(user)
            session.commit()
            
            # Create API key
            key_data = {
                "user_id": user.id,
                "name": "Test Key",
                "expiry": datetime.utcnow() + timedelta(days=30)
            }
            
            key_response = api_manager.create_api_key(key_data)
            assert key_response.key is not None
            assert key_response.user_id == user.id
            
            # Validate the key
            user_id = api_manager.validate_api_key(key_response.key)
            assert user_id == user.id

@pytest.mark.asyncio
class TestEncryption:
    async def test_document_encryption(self, temp_storage):
        encryption = DataEncryption(str(temp_storage / "test_key"))
        
        test_content = b"Test content for encryption"
        test_metadata = {"type": "test", "size": len(test_content)}
        
        # Encrypt
        encrypted_data = encryption.encrypt_document(test_content, test_metadata)
        assert "content" in encrypted_data
        assert "metadata" in encrypted_data
        assert encrypted_data["content"] != test_content
        
        # Decrypt
        decrypted_data = encryption.decrypt_document(encrypted_data)
        assert decrypted_data["content"] == test_content
        assert decrypted_data["metadata"]["type"] == test_metadata["type"]

@pytest.mark.asyncio
class TestSecureStorage:
    async def test_store_document(self, test_config, temp_storage, db_session):
        # Initialize dependencies
        encryption = DataEncryption(str(temp_storage / "test_key"))
        rbac = RBACManager(test_config["database_url"])
        
        # Create test user with permissions
        with db_session as session:
            permission = Permission(
                name="doc_write",
                resource="documents",
                action="write"
            )
            role = Role(
                name="test_role",
                permissions=[permission]
            )
            session.add(role)
            session.commit()
            
            user = User(
                username="testuser",
                email="test@example.com",
                password_hash="hash",
                role_id=role.id
            )
            session.add(user)
            session.commit()
            
            # Initialize secure storage
            storage = SecureStorage(
                str(temp_storage),
                encryption,
                None,  # We'll mock the audit logger
                rbac
            )
            
            # Store document
            content = b"Test document content"
            doc_id = storage.store_document(
                user.id,
                content,
                "test.txt",
                "text/plain",
                {"test_key": "test_value"}
            )
            
            assert doc_id is not None
            
            # Retrieve document
            doc = storage.retrieve_document(user.id, doc_id)
            assert doc.content_type == "text/plain"
            assert doc.metadata["test_key"] == "test_value"

@pytest.mark.asyncio
class TestAuditLogger:
    async def test_log_event(self, temp_log_dir, test_config):
        logger = AuditLogger({"log_dir": str(temp_log_dir)})
        
        event = {
            "event_type": "test_event",
            "user_id": str(uuid.uuid4()),
            "action": "test",
            "resource_type": "test_resource",
            "resource_id": str(uuid.uuid4())
        }
        
        # Log event
        logger.log_event(event)
        
        # Verify log file exists and contains event
        log_file = temp_log_dir / "audit.log"
        assert log_file.exists()
        
        log_content = log_file.read_text()
        assert event["event_type"] in log_content
        assert event["user_id"] in log_content
