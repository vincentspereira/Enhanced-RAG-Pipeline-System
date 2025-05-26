from typing import List, Dict, Any, Optional, Union, BinaryIO
from abc import ABC, abstractmethod
import boto3
from azure.storage.blob import BlobServiceClient
from google.cloud import storage
import os
import logging
from dataclasses import dataclass
import tempfile
from pathlib import Path
import json
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class StorageConfig:
    provider: str = "s3"  # "s3", "azure", or "gcp"
    
    # AWS S3 settings
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    s3_bucket: Optional[str] = None
    s3_region: Optional[str] = None
    
    # Azure Blob Storage settings
    azure_connection_string: Optional[str] = None
    azure_container: Optional[str] = None
    
    # Google Cloud Storage settings
    gcp_project_id: Optional[str] = None
    gcp_bucket: Optional[str] = None
    gcp_credentials_path: Optional[str] = None
    
    # Common settings
    chunk_size: int = 1024 * 1024  # 1MB
    max_workers: int = 4
    enable_compression: bool = True
    cache_duration: int = 3600  # 1 hour
    retry_attempts: int = 3
    timeout: int = 30

class CloudStorage(ABC):
    @abstractmethod
    async def upload_file(self, local_path: str, remote_path: str) -> bool:
        pass

    @abstractmethod
    async def download_file(self, remote_path: str, local_path: str) -> bool:
        pass

    @abstractmethod
    async def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_file(self, remote_path: str) -> bool:
        pass

    @abstractmethod
    async def file_exists(self, remote_path: str) -> bool:
        pass

class S3Storage(CloudStorage):
    def __init__(self, config: StorageConfig):
        self.config = config
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            region_name=config.s3_region
        )

    async def upload_file(self, local_path: str, remote_path: str) -> bool:
        try:
            with open(local_path, 'rb') as file:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.s3.upload_fileobj(
                        file,
                        self.config.s3_bucket,
                        remote_path
                    )
                )
            return True
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False

    async def download_file(self, remote_path: str, local_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as file:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.s3.download_fileobj(
                        self.config.s3_bucket,
                        remote_path,
                        file
                    )
                )
            return True
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False

    async def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            files = []
            
            async for page in self._paginate(paginator, Bucket=self.config.s3_bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    files.append({
                        'name': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'etag': obj['ETag']
                    })
            return files
        except Exception as e:
            logger.error(f"S3 list failed: {e}")
            return []

    async def delete_file(self, remote_path: str) -> bool:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3.delete_object(
                    Bucket=self.config.s3_bucket,
                    Key=remote_path
                )
            )
            return True
        except Exception as e:
            logger.error(f"S3 delete failed: {e}")
            return False

    async def file_exists(self, remote_path: str) -> bool:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3.head_object(
                    Bucket=self.config.s3_bucket,
                    Key=remote_path
                )
            )
            return True
        except:
            return False

    async def _paginate(self, paginator, **kwargs):
        """Helper method to handle S3 pagination"""
        for page in paginator.paginate(**kwargs):
            yield page

class AzureStorage(CloudStorage):
    def __init__(self, config: StorageConfig):
        self.config = config
        self.blob_service = BlobServiceClient.from_connection_string(
            config.azure_connection_string
        )
        self.container = self.blob_service.get_container_client(
            config.azure_container
        )

    async def upload_file(self, local_path: str, remote_path: str) -> bool:
        try:
            blob_client = self.container.get_blob_client(remote_path)
            with open(local_path, "rb") as data:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: blob_client.upload_blob(
                        data,
                        overwrite=True
                    )
                )
            return True
        except Exception as e:
            logger.error(f"Azure upload failed: {e}")
            return False

    async def download_file(self, remote_path: str, local_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob_client = self.container.get_blob_client(remote_path)
            with open(local_path, "wb") as file:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: blob_client.download_blob().readinto(file)
                )
            return True
        except Exception as e:
            logger.error(f"Azure download failed: {e}")
            return False

    async def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        try:
            files = []
            async for blob in self._list_blobs(prefix):
                files.append({
                    'name': blob.name,
                    'size': blob.size,
                    'last_modified': blob.last_modified.isoformat(),
                    'etag': blob.etag
                })
            return files
        except Exception as e:
            logger.error(f"Azure list failed: {e}")
            return []

    async def delete_file(self, remote_path: str) -> bool:
        try:
            blob_client = self.container.get_blob_client(remote_path)
            await asyncio.get_event_loop().run_in_executor(
                None,
                blob_client.delete_blob
            )
            return True
        except Exception as e:
            logger.error(f"Azure delete failed: {e}")
            return False

    async def file_exists(self, remote_path: str) -> bool:
        try:
            blob_client = self.container.get_blob_client(remote_path)
            await asyncio.get_event_loop().run_in_executor(
                None,
                blob_client.get_blob_properties
            )
            return True
        except:
            return False

    async def _list_blobs(self, prefix: str):
        """Helper method to handle Azure blob listing"""
        for blob in self.container.list_blobs(name_starts_with=prefix):
            yield blob

class GCPStorage(CloudStorage):
    def __init__(self, config: StorageConfig):
        self.config = config
        self.client = storage.Client.from_service_account_json(
            config.gcp_credentials_path
        ) if config.gcp_credentials_path else storage.Client()
        self.bucket = self.client.bucket(config.gcp_bucket)

    async def upload_file(self, local_path: str, remote_path: str) -> bool:
        try:
            blob = self.bucket.blob(remote_path)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: blob.upload_from_filename(local_path)
            )
            return True
        except Exception as e:
            logger.error(f"GCP upload failed: {e}")
            return False

    async def download_file(self, remote_path: str, local_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob = self.bucket.blob(remote_path)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: blob.download_to_filename(local_path)
            )
            return True
        except Exception as e:
            logger.error(f"GCP download failed: {e}")
            return False

    async def list_files(self, prefix: str = "") -> List[Dict[str, Any]]:
        try:
            files = []
            blobs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: list(self.bucket.list_blobs(prefix=prefix))
            )
            
            for blob in blobs:
                files.append({
                    'name': blob.name,
                    'size': blob.size,
                    'last_modified': blob.updated.isoformat(),
                    'etag': blob.etag
                })
            return files
        except Exception as e:
            logger.error(f"GCP list failed: {e}")
            return []

    async def delete_file(self, remote_path: str) -> bool:
        try:
            blob = self.bucket.blob(remote_path)
            await asyncio.get_event_loop().run_in_executor(
                None,
                blob.delete
            )
            return True
        except Exception as e:
            logger.error(f"GCP delete failed: {e}")
            return False

    async def file_exists(self, remote_path: str) -> bool:
        try:
            blob = self.bucket.blob(remote_path)
            return await asyncio.get_event_loop().run_in_executor(
                None,
                blob.exists
            )
        except:
            return False

class StorageManager:
    def __init__(self, config: StorageConfig):
        self.config = config
        self.storage = self._initialize_storage()
        self._cache = {}
        self._cache_timestamps = {}

    def _initialize_storage(self) -> CloudStorage:
        """Initialize the appropriate storage provider"""
        if self.config.provider == "s3":
            return S3Storage(self.config)
        elif self.config.provider == "azure":
            return AzureStorage(self.config)
        elif self.config.provider == "gcp":
            return GCPStorage(self.config)
        else:
            raise ValueError(f"Unsupported storage provider: {self.config.provider}")

    async def upload_file(
        self,
        local_path: str,
        remote_path: str,
        compress: bool = None
    ) -> bool:
        """Upload a file to cloud storage with optional compression"""
        if compress is None:
            compress = self.config.enable_compression

        try:
            if compress:
                import gzip
                temp_path = tempfile.mktemp()
                with open(local_path, 'rb') as f_in:
                    with gzip.open(temp_path, 'wb') as f_out:
                        f_out.writelines(f_in)
                success = await self.storage.upload_file(temp_path, remote_path + '.gz')
                os.remove(temp_path)
            else:
                success = await self.storage.upload_file(local_path, remote_path)

            if success:
                self._invalidate_cache(remote_path)
            return success

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    async def download_file(
        self,
        remote_path: str,
        local_path: str,
        use_cache: bool = True
    ) -> bool:
        """Download a file from cloud storage with caching"""
        try:
            # Check if file is in cache and still valid
            if use_cache and remote_path in self._cache:
                cache_time = self._cache_timestamps[remote_path]
                if datetime.now() - cache_time < timedelta(seconds=self.config.cache_duration):
                    with open(local_path, 'wb') as f:
                        f.write(self._cache[remote_path])
                    return True

            # Handle compressed files
            if await self.storage.file_exists(remote_path + '.gz'):
                temp_path = tempfile.mktemp()
                success = await self.storage.download_file(remote_path + '.gz', temp_path)
                if success:
                    import gzip
                    with gzip.open(temp_path, 'rb') as f_in:
                        with open(local_path, 'wb') as f_out:
                            f_out.writelines(f_in)
                    os.remove(temp_path)
            else:
                success = await self.storage.download_file(remote_path, local_path)

            # Update cache
            if success and use_cache:
                with open(local_path, 'rb') as f:
                    self._cache[remote_path] = f.read()
                self._cache_timestamps[remote_path] = datetime.now()

            return success

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def _invalidate_cache(self, remote_path: str = None):
        """Invalidate cache for a specific file or all files"""
        if remote_path:
            self._cache.pop(remote_path, None)
            self._cache_timestamps.pop(remote_path, None)
        else:
            self._cache.clear()
            self._cache_timestamps.clear()

    async def list_files(
        self,
        prefix: str = "",
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """List files in cloud storage"""
        try:
            files = await self.storage.list_files(prefix)
            if not recursive:
                # Filter out files in subdirectories
                files = [f for f in files if '/' not in f['name'][len(prefix):]]
            return files
        except Exception as e:
            logger.error(f"List files failed: {e}")
            return []

    async def delete_file(self, remote_path: str) -> bool:
        """Delete a file from cloud storage"""
        try:
            # Try to delete both compressed and uncompressed versions
            success = await self.storage.delete_file(remote_path)
            compressed_success = await self.storage.delete_file(remote_path + '.gz')
            
            if success or compressed_success:
                self._invalidate_cache(remote_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    async def copy_file(
        self,
        source_path: str,
        dest_path: str,
        source_provider: Optional[str] = None
    ) -> bool:
        """Copy a file between storage providers or within the same provider"""
        try:
            # If source provider is different, initialize it
            source_storage = self.storage
            if source_provider and source_provider != self.config.provider:
                temp_config = StorageConfig(provider=source_provider)
                source_storage = StorageManager(temp_config)._initialize_storage()

            # Download to temporary file
            temp_path = tempfile.mktemp()
            success = await source_storage.download_file(source_path, temp_path)
            if not success:
                return False

            # Upload to destination
            success = await self.storage.upload_file(temp_path, dest_path)
            os.remove(temp_path)
            
            if success:
                self._invalidate_cache(dest_path)
            return success

        except Exception as e:
            logger.error(f"Copy failed: {e}")
            return False

    async def get_file_metadata(self, remote_path: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a file"""
        try:
            files = await self.storage.list_files(remote_path)
            return next((f for f in files if f['name'] == remote_path), None)
        except Exception as e:
            logger.error(f"Get metadata failed: {e}")
            return None
