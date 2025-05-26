from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import base64
import os
from typing import Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

class DataEncryption:
    def __init__(self, key_path: str = None):
        self.key_path = key_path or os.path.join(os.path.dirname(__file__), 'encryption.key')
        self._initialize_key()
        
    def _initialize_key(self):
        """Initialize or load encryption key."""
        if os.path.exists(self.key_path):
            with open(self.key_path, 'rb') as key_file:
                key = key_file.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_path, 'wb') as key_file:
                key_file.write(key)
        
        self.fernet = Fernet(key)
        
    def encrypt_document(self, content: bytes, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, bytes]:
        """Encrypt document content and metadata."""
        try:
            encrypted_content = self.fernet.encrypt(content)
            
            if metadata:
                metadata_bytes = json.dumps(metadata).encode()
                encrypted_metadata = self.fernet.encrypt(metadata_bytes)
            else:
                encrypted_metadata = None
                
            return {
                "content": encrypted_content,
                "metadata": encrypted_metadata,
                "timestamp": base64.b64encode(str(os.path.getmtime(self.key_path)).encode())
            }
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise
            
    def decrypt_document(self, encrypted_data: Dict[str, bytes]) -> Dict[str, Any]:
        """Decrypt document content and metadata."""
        try:
            decrypted_content = self.fernet.decrypt(encrypted_data["content"])
            
            result = {"content": decrypted_content}
            
            if encrypted_data.get("metadata"):
                metadata_bytes = self.fernet.decrypt(encrypted_data["metadata"])
                result["metadata"] = json.loads(metadata_bytes.decode())
                
            return result
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise
            
    def rotate_key(self):
        """Rotate encryption key for security."""
        new_key = Fernet.generate_key()
        new_fernet = Fernet(new_key)
        
        # Backup old key
        backup_path = f"{self.key_path}.backup"
        with open(self.key_path, 'rb') as current_key_file:
            with open(backup_path, 'wb') as backup_key_file:
                backup_key_file.write(current_key_file.read())
        
        # Save new key
        with open(self.key_path, 'wb') as key_file:
            key_file.write(new_key)
            
        self.fernet = new_fernet
