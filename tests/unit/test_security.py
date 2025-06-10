import pytest
import uuid
from datetime import datetime, timedelta
from Scripts.security.rbac import RBACManager, User, Role, Permission
from Scripts.security.api_keys import APIKeyManager
from Scripts.security.audit_logger import AuditLogger
from Scripts.security.encryption import DataEncryption
from Scripts.security.secure_storage import SecureStorage

class TestRBAC: # Removed @pytest.mark.asyncio
    def test_create_user(self, db_engine, db_session): # Changed to def, added db_engine
        rbac = RBACManager(engine=db_engine) # Use db_engine
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
        
        from Scripts.security.rbac import UserCreate # Ensure UserCreate is imported
        user_create_model = UserCreate(**user_data)
        user = rbac.create_user(user_create_model, session=db_session) # Pass session
        assert user.username == user_data["username"]
        assert user.email == user_data["email"]
        assert user.role_id == user_data["role_id"]

    def test_check_permission(self, db_engine, db_session): # Changed to def, added db_engine
        rbac = RBACManager(engine=db_engine) # Use db_engine
        
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
                user.id, "test_resource", "read", session=db_session # Pass session
            )
            assert has_permission is True
            
            # Test non-existent permission
            has_permission = rbac.check_permission(
                user.id, "test_resource", "write", session=db_session # Pass session
            )
            assert has_permission is False

class TestAPIKeys: # Removed @pytest.mark.asyncio
    def test_create_api_key(self, db_engine, db_session): # Changed to def, added db_engine
        rbac = RBACManager(engine=db_engine) # Use db_engine
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
            
            from Scripts.security.api_keys import APIKeyCreate # Ensure APIKeyCreate is imported
            api_key_create_model = APIKeyCreate(**key_data)
            key_response = api_manager.create_api_key(api_key_create_model, session=db_session) # Pass session
            assert key_response.key is not None
            assert key_response.user_id == user.id
            
            # Validate the key
            user_id = api_manager.validate_api_key(key_response.key, session=db_session) # Pass session
            assert user_id == user.id

class TestEncryption: # Removed @pytest.mark.asyncio
    def test_document_encryption(self, temp_storage): # Changed to def
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

class TestSecureStorage: # Removed @pytest.mark.asyncio
    def test_store_document(self, db_engine, temp_storage, db_session): # Changed to def, added db_engine
        # Initialize dependencies
        encryption = DataEncryption(str(temp_storage / "test_key"))
        rbac = RBACManager(engine=db_engine) # Use db_engine
        
        # Create test user with permissions
        with db_session as session:
            write_permission = Permission(
                name="doc_write",
                resource="documents",
                action="write"
            )
            read_permission = Permission( # Add read permission
                name="doc_read",
                resource="documents",
                action="read"
            )
            role = Role(
                name="test_role",
                permissions=[write_permission, read_permission] # Assign both permissions
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
            from unittest.mock import MagicMock # Ensure MagicMock is imported
            mock_audit_logger = MagicMock(spec=AuditLogger) # Create a mock audit logger

            storage = SecureStorage(
                str(temp_storage),
                encryption,
                mock_audit_logger,  # Pass the mock audit logger
                rbac
            )
            
            # Store document
            content = b"Test document content"
            # SecureStorage.store_document itself uses rbac_manager.check_permission, which now takes a session
            # This means SecureStorage.store_document might also need to accept a session, or
            # rbac_manager.check_permission needs to be able to create its own if session is None (which it does)
            # For now, let's assume the internal session creation in check_permission is acceptable if not passed.
            # However, to be fully correct for tests, SecureStorage methods should also accept the session.
            # For this change, we focus on what's directly passed. The PermissionError might persist if the
            # session used by check_permission inside store_document is different or doesn't see test setup.
            # Let's call check_permission directly first in the test with the test session.

            assert rbac.check_permission(user.id, "documents", "write", session=db_session) # Verify permission with test session

            doc_id = storage.store_document(
                user.id,
                content,
                "test.txt",
                "text/plain",
                {"test_key": "test_value"},
                session=db_session # Pass session
            )
            
            assert doc_id is not None
            
            # Retrieve document
            doc = storage.retrieve_document(user.id, doc_id, session=db_session) # Pass session
            assert doc.content_type == "text/plain"
            assert doc.metadata["test_key"] == "test_value"

class TestAuditLogger: # Removed @pytest.mark.asyncio
    def test_log_event(self, temp_log_dir): # Changed to def, removed test_config as it's not used
        logger = AuditLogger({"log_dir": str(temp_log_dir)})
        
        # Import AuditEvent here or at the top of the file
        from Scripts.security.audit_logger import AuditEvent

        event_data = {
            "event_type": "test_event",
            "user_id": str(uuid.uuid4()),
            "action": "test",
            "resource_type": "test_resource",
            "resource_id": str(uuid.uuid4()),
            "metadata": {"key": "value"} # Ensure metadata is not None
        }
        audit_event_obj = AuditEvent(**event_data)
        
        # Log event
        logger.log_event(audit_event_obj)
        
        # Allow some time for the event to be processed by the worker thread
        import time
        time.sleep(0.1) # Adjust as necessary

        # Verify log file exists and contains event
        log_file = temp_log_dir / "audit.log"
        assert log_file.exists()
        
        log_content = log_file.read_text()
        assert event_data["event_type"] in log_content # Check against original dict
        assert event_data["user_id"] in log_content
