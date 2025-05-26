import os
from pathlib import Path
from typing import Union, Optional
import asyncio
import logging

from .base import StorageProvider

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceNotFoundError
except ImportError:
    BlobServiceClient = None

try:
    from google.cloud import storage
    from google.api_core import exceptions as gcp_exceptions
except ImportError:
    storage = None

logger = logging.getLogger(__name__)

class S3StorageProvider(StorageProvider):
    """AWS S3 storage provider implementation."""
    
    def __init__(self, bucket_name: str, aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None, region_name: Optional[str] = None):
        if not boto3:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
            
        self.bucket_name = bucket_name
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

    async def upload_file(self, file_path: Union[str, Path], destination_path: str) -> str:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, 
                lambda: self.s3.upload_file(str(file_path), self.bucket_name, destination_path)
            )
            return f"s3://{self.bucket_name}/{destination_path}"
        except ClientError as e:
            logger.error(f"Error uploading to S3: {e}")
            raise

class AzureBlobStorageProvider(StorageProvider):
    """Azure Blob storage provider implementation."""
    
    def __init__(self, connection_string: str, container_name: str):
        if not BlobServiceClient:
            raise ImportError("azure-storage-blob is required. Install with: pip install azure-storage-blob")
            
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)

    async def upload_file(self, file_path: Union[str, Path], destination_path: str) -> str:
        try:
            blob_client = self.container_client.get_blob_client(destination_path)
            with open(str(file_path), "rb") as data:
                await blob_client.upload_blob(data)
            return blob_client.url
        except Exception as e:
            logger.error(f"Error uploading to Azure Blob: {e}")
            raise

class GCSStorageProvider(StorageProvider):
    """Google Cloud Storage provider implementation."""
    
    def __init__(self, bucket_name: str, credentials_path: Optional[str] = None):
        if not storage:
            raise ImportError("google-cloud-storage is required. Install with: pip install google-cloud-storage")
            
        self.bucket_name = bucket_name
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    async def upload_file(self, file_path: Union[str, Path], destination_path: str) -> str:
        try:
            blob = self.bucket.blob(destination_path)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, blob.upload_from_filename, str(file_path))
            return f"gs://{self.bucket_name}/{destination_path}"
        except Exception as e:
            logger.error(f"Error uploading to GCS: {e}")
            raise

# Factory function to create appropriate storage provider
def create_storage_provider(provider_type: str, **kwargs) -> StorageProvider:
    """Create a storage provider instance based on the provider type.
    
    Args:
        provider_type: One of 's3', 'azure', or 'gcs'
        **kwargs: Provider-specific configuration options
        
    Returns:
        StorageProvider: Configured storage provider instance
    """
    providers = {
        's3': S3StorageProvider,
        'azure': AzureBlobStorageProvider,
        'gcs': GCSStorageProvider
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unsupported storage provider: {provider_type}")
        
    return providers[provider_type](**kwargs)