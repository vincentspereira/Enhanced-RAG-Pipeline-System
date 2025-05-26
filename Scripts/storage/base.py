from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, BinaryIO, Optional

class StorageProvider(ABC):
    """Abstract base class for storage providers."""

    @abstractmethod
    async def upload_file(self, file_path: Union[str, Path], destination_path: str) -> str:
        """Upload a file to storage.
        
        Args:
            file_path: Local file path to upload
            destination_path: Path in storage where file should be saved
            
        Returns:
            str: URL or path to access the uploaded file
        """
        pass

    @abstractmethod
    async def download_file(self, storage_path: str, local_path: Union[str, Path]) -> Path:
        """Download a file from storage.
        
        Args:
            storage_path: Path in storage to download from
            local_path: Local path to save the file to
            
        Returns:
            Path: Path to the downloaded file
        """
        pass

    @abstractmethod
    async def delete_file(self, storage_path: str) -> bool:
        """Delete a file from storage.
        
        Args:
            storage_path: Path in storage to delete
            
        Returns:
            bool: True if deletion was successful
        """
        pass

    @abstractmethod
    async def list_files(self, prefix: str = "") -> list[str]:
        """List files in storage with optional prefix filter.
        
        Args:
            prefix: Optional prefix to filter results
            
        Returns:
            list[str]: List of file paths in storage
        """
        pass

    @abstractmethod
    async def get_file_metadata(self, storage_path: str) -> dict:
        """Get metadata for a file in storage.
        
        Args:
            storage_path: Path to file in storage
            
        Returns:
            dict: File metadata including size, type, last modified, etc.
        """
        pass