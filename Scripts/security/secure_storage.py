import os
from typing import Optional, Dict, Any, BinaryIO, Union
from datetime import datetime
import uuid
import hashlib
from pathlib import Path
import shutil
import json

from sqlalchemy.orm import Session # Added import for Session
from .encryption import DataEncryption
from .audit_logger import AuditLogger, AuditEvent
from .rbac import RBACManager

class SecureDocument:
    def __init__(self, 
                 document_id: str,
                 content_type: str,
                 size: int,
                 metadata: Dict[str, Any],
                 encrypted_path: str,
                 created_at: datetime,
                 created_by: str,
                 last_modified_at: datetime,
                 last_modified_by: str):
        self.document_id = document_id
        self.content_type = content_type
        self.size = size
        self.metadata = metadata
        self.encrypted_path = encrypted_path
        self.created_at = created_at
        self.created_by = created_by
        self.last_modified_at = last_modified_at
        self.last_modified_by = last_modified_by

class SecureStorage:
    def __init__(self, 
                 base_path: str,
                 encryption: DataEncryption,
                 audit_logger: AuditLogger,
                 rbac_manager: RBACManager):
        """Initialize secure storage system."""
        self.base_path = Path(base_path)
        self.docs_path = self.base_path / "documents"
        self.metadata_path = self.base_path / "metadata"
        self.encryption = encryption
        self.audit_logger = audit_logger
        self.rbac_manager = rbac_manager
        
        # Create necessary directories
        self.docs_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)

    def _get_document_path(self, doc_id: str) -> Path:
        """Get the path for a document's encrypted content."""
        return self.docs_path / f"{doc_id}.enc"

    def _get_metadata_path(self, doc_id: str) -> Path:
        """Get the path for a document's metadata."""
        return self.metadata_path / f"{doc_id}.json"

    def _calculate_hash(self, file_obj: BinaryIO) -> str:
        """Calculate SHA-256 hash of file content."""
        sha256 = hashlib.sha256()
        for chunk in iter(lambda: file_obj.read(4096), b''):
            sha256.update(chunk)
        return sha256.hexdigest()

    def store_document(self,
                      user_id: uuid.UUID,
                      content: Union[bytes, BinaryIO],
                      filename: str,
                      content_type: str,
                      metadata: Optional[Dict[str, Any]] = None,
                      session: Optional[Session] = None) -> str: # Added session
        """
        Store a document securely with encryption and audit logging.
        Returns the document ID.
        """
        # Check permissions
        if not self.rbac_manager.check_permission(user_id, 'documents', 'write', session=session): # Pass session
            raise PermissionError("User does not have write permission")

        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        try:
            # Prepare content
            if isinstance(content, bytes):
                file_content = content
                content_size = len(content)
            else:
                content.seek(0)
                file_content = content.read()
                content_size = content.tell()
                content.seek(0)
            
            # Calculate hash
            content_hash = hashlib.sha256(file_content).hexdigest()
            
            # Encrypt and store document
            encrypted_data = self.encryption.encrypt_document(
                file_content,
                {"filename": filename, "content_type": content_type}
            )
            
            doc_path = self._get_document_path(doc_id)
            with open(doc_path, 'wb') as f:
                f.write(encrypted_data["content"])
            
            # Store metadata
            doc_metadata = {
                "document_id": doc_id,
                "filename": filename,
                "content_type": content_type,
                "size": content_size,
                "hash": content_hash,
                "user_metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "created_by": str(user_id),
                "last_modified_at": datetime.utcnow().isoformat(),
                "last_modified_by": str(user_id)
            }
            
            metadata_path = self._get_metadata_path(doc_id)
            with open(metadata_path, 'w') as f:
                json.dump(doc_metadata, f)
            
            # Log the event
            self.audit_logger.log_event(AuditEvent(
                event_type="document_upload",
                user_id=str(user_id),
                action="create",
                resource_type="document",
                resource_id=doc_id,
                metadata={
                    "filename": filename,
                    "content_type": content_type,
                    "size": content_size
                }
            ))
            
            return doc_id
            
        except Exception as e:
            # Log failure
            self.audit_logger.log_event(AuditEvent(
                event_type="document_upload",
                user_id=str(user_id),
                action="create",
                resource_type="document",
                resource_id=doc_id,
                status="failure",
                metadata={"error": str(e)}
            ))
            raise

    def retrieve_document(self, 
                         user_id: uuid.UUID, 
                         doc_id: str,
                         session: Optional[Session] = None) -> SecureDocument: # Added session
        """
        Retrieve a document's metadata and content.
        Returns a SecureDocument object.
        """
        # Check permissions
        if not self.rbac_manager.check_permission(user_id, 'documents', 'read', session=session): # Pass session
            raise PermissionError("User does not have read permission")

        try:
            # Get metadata
            metadata_path = self._get_metadata_path(doc_id)
            if not metadata_path.exists():
                raise FileNotFoundError(f"Document {doc_id} not found")
                
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get encrypted content
            doc_path = self._get_document_path(doc_id)
            with open(doc_path, 'rb') as f:
                encrypted_data = {
                    "content": f.read(),
                    "metadata": None  # Metadata stored separately
                }
            
            # Log access
            self.audit_logger.log_event(AuditEvent(
                event_type="document_access",
                user_id=str(user_id),
                action="read",
                resource_type="document",
                resource_id=doc_id
            ))
            
            return SecureDocument(
                document_id=doc_id,
                content_type=metadata["content_type"],
                size=metadata["size"],
                metadata=metadata["user_metadata"],
                encrypted_path=str(doc_path),
                created_at=datetime.fromisoformat(metadata["created_at"]),
                created_by=metadata["created_by"],
                last_modified_at=datetime.fromisoformat(metadata["last_modified_at"]),
                last_modified_by=metadata["last_modified_by"]
            )
            
        except Exception as e:
            # Log failure
            self.audit_logger.log_event(AuditEvent(
                event_type="document_access",
                user_id=str(user_id),
                action="read",
                resource_type="document",
                resource_id=doc_id,
                status="failure",
                metadata={"error": str(e)}
            ))
            raise

    def update_document(self,
                       user_id: uuid.UUID,
                       doc_id: str,
                       content: Optional[Union[bytes, BinaryIO]] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       session: Optional[Session] = None) -> None: # Added session
        """
        Update a document's content and/or metadata.
        """
        # Check permissions
        if not self.rbac_manager.check_permission(user_id, 'documents', 'write', session=session): # Pass session
            raise PermissionError("User does not have write permission")

        try:
            # Get existing metadata
            metadata_path = self._get_metadata_path(doc_id)
            if not metadata_path.exists():
                raise FileNotFoundError(f"Document {doc_id} not found")
                
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
            
            # Update content if provided
            if content is not None:
                if isinstance(content, bytes):
                    file_content = content
                    content_size = len(content)
                else:
                    content.seek(0)
                    file_content = content.read()
                    content_size = content.tell()
                    content.seek(0)
                
                # Calculate new hash
                content_hash = hashlib.sha256(file_content).hexdigest()
                
                # Encrypt and store new content
                encrypted_data = self.encryption.encrypt_document(
                    file_content,
                    {"filename": existing_metadata["filename"], 
                     "content_type": existing_metadata["content_type"]}
                )
                
                doc_path = self._get_document_path(doc_id)
                with open(doc_path, 'wb') as f:
                    f.write(encrypted_data["content"])
                    
                # Update metadata
                existing_metadata["size"] = content_size
                existing_metadata["hash"] = content_hash
            
            # Update metadata if provided
            if metadata is not None:
                existing_metadata["user_metadata"].update(metadata)
            
            # Update modification timestamp
            existing_metadata["last_modified_at"] = datetime.utcnow().isoformat()
            existing_metadata["last_modified_by"] = str(user_id)
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(existing_metadata, f)
            
            # Log the update
            self.audit_logger.log_event(AuditEvent(
                event_type="document_update",
                user_id=str(user_id),
                action="update",
                resource_type="document",
                resource_id=doc_id,
                metadata={
                    "content_updated": content is not None,
                    "metadata_updated": metadata is not None
                }
            ))
            
        except Exception as e:
            # Log failure
            self.audit_logger.log_event(AuditEvent(
                event_type="document_update",
                user_id=str(user_id),
                action="update",
                resource_type="document",
                resource_id=doc_id,
                status="failure",
                metadata={"error": str(e)}
            ))
            raise

    def delete_document(self, user_id: uuid.UUID, doc_id: str, session: Optional[Session] = None) -> None: # Added session
        """
        Delete a document and its metadata.
        """
        # Check permissions
        if not self.rbac_manager.check_permission(user_id, 'documents', 'delete', session=session): # Pass session
            raise PermissionError("User does not have delete permission")

        try:
            doc_path = self._get_document_path(doc_id)
            metadata_path = self._get_metadata_path(doc_id)
            
            # Verify document exists
            if not metadata_path.exists():
                raise FileNotFoundError(f"Document {doc_id} not found")
            
            # Delete files
            if doc_path.exists():
                doc_path.unlink()
            metadata_path.unlink()
            
            # Log deletion
            self.audit_logger.log_event(AuditEvent(
                event_type="document_deletion",
                user_id=str(user_id),
                action="delete",
                resource_type="document",
                resource_id=doc_id
            ))
            
        except Exception as e:
            # Log failure
            self.audit_logger.log_event(AuditEvent(
                event_type="document_deletion",
                user_id=str(user_id),
                action="delete",
                resource_type="document",
                resource_id=doc_id,
                status="failure",
                metadata={"error": str(e)}
            ))
            raise

    def list_documents(self, 
                      user_id: uuid.UUID, 
                      filters: Optional[Dict[str, Any]] = None,
                      session: Optional[Session] = None) -> list[SecureDocument]: # Added session
        """
        List documents with optional filtering.
        """
        # Check permissions
        if not self.rbac_manager.check_permission(user_id, 'documents', 'list', session=session): # Pass session
            raise PermissionError("User does not have list permission")

        try:
            documents = []
            
            for metadata_file in self.metadata_path.glob("*.json"):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Apply filters if any
                if filters:
                    matches = True
                    for key, value in filters.items():
                        if key in metadata and metadata[key] != value:
                            matches = False
                            break
                    if not matches:
                        continue
                
                doc = SecureDocument(
                    document_id=metadata["document_id"],
                    content_type=metadata["content_type"],
                    size=metadata["size"],
                    metadata=metadata["user_metadata"],
                    encrypted_path=str(self._get_document_path(metadata["document_id"])),
                    created_at=datetime.fromisoformat(metadata["created_at"]),
                    created_by=metadata["created_by"],
                    last_modified_at=datetime.fromisoformat(metadata["last_modified_at"]),
                    last_modified_by=metadata["last_modified_by"]
                )
                documents.append(doc)
            
            # Log list operation
            self.audit_logger.log_event(AuditEvent(
                event_type="document_list",
                user_id=str(user_id),
                action="list",
                resource_type="documents",
                metadata={"filters": filters}
            ))
            
            return documents
            
        except Exception as e:
            # Log failure
            self.audit_logger.log_event(AuditEvent(
                event_type="document_list",
                user_id=str(user_id),
                action="list",
                resource_type="documents",
                status="failure",
                metadata={"error": str(e)}
            ))
            raise
