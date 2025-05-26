import os
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import tarfile
import json
import logging
from typing import Dict, Any, List, Optional
import boto3
from botocore.exceptions import ClientError
import asyncio
import aiofiles
import qdrant_client
from sqlalchemy import create_engine, text
import schedule
import time
import threading

class BackupManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backup_dir = Path(config.get('backup_dir', 'backups'))
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
        # Initialize storage client (if configured)
        self.s3_client = self._init_s3_client() if config.get('use_s3') else None
        
    def _setup_logging(self):
        self.logger = logging.getLogger("backup_manager")
        log_dir = Path(self.config.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_dir / "backup.log")
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def _init_s3_client(self):
        """Initialize S3 client for cloud backups."""
        return boto3.client(
            's3',
            aws_access_key_id=self.config['aws_access_key'],
            aws_secret_access_key=self.config['aws_secret_key'],
            region_name=self.config.get('aws_region', 'us-east-1')
        )
        
    async def backup_database(self) -> str:
        """Create a backup of the database."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_file = self.backup_dir / f"db_backup_{timestamp}.sql"
        
        try:
            # Use pg_dump for PostgreSQL backup
            db_url = self.config['database_url']
            if 'postgresql' in db_url:
                cmd = f"pg_dump {db_url} > {backup_file}"
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    raise Exception(f"Database backup failed: {stderr.decode()}")
            
            self.logger.info(f"Database backup created: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            self.logger.error(f"Database backup failed: {str(e)}")
            raise

    async def backup_qdrant(self) -> str:
        """Create a backup of Qdrant collections."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_dir = self.backup_dir / f"qdrant_backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            client = qdrant_client.QdrantClient(
                url=self.config['qdrant_url'],
                api_key=self.config.get('qdrant_api_key')
            )
            
            # Get all collections
            collections = client.get_collections()
            
            # Backup each collection
            for collection in collections:
                collection_backup = backup_dir / f"{collection.name}.json"
                # Snapshot the collection
                snapshot_path = await self._create_collection_snapshot(client, collection.name)
                # Copy snapshot to backup directory
                shutil.copy2(snapshot_path, collection_backup)
            
            # Create tar archive
            backup_file = self.backup_dir / f"qdrant_backup_{timestamp}.tar.gz"
            with tarfile.open(backup_file, "w:gz") as tar:
                tar.add(backup_dir, arcname=os.path.basename(backup_dir))
            
            # Clean up temporary directory
            shutil.rmtree(backup_dir)
            
            self.logger.info(f"Qdrant backup created: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            self.logger.error(f"Qdrant backup failed: {str(e)}")
            raise

    async def _create_collection_snapshot(self, client: qdrant_client.QdrantClient, collection_name: str) -> str:
        """Create a snapshot of a Qdrant collection."""
        try:
            snapshot = client.snapshot(collection_name=collection_name)
            return snapshot.snapshot_path
        except Exception as e:
            self.logger.error(f"Failed to create snapshot for collection {collection_name}: {str(e)}")
            raise

    async def backup_files(self) -> str:
        """Backup document files and metadata."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_file = self.backup_dir / f"files_backup_{timestamp}.tar.gz"
        
        try:
            with tarfile.open(backup_file, "w:gz") as tar:
                # Backup document storage
                docs_dir = Path(self.config['document_storage_path'])
                if docs_dir.exists():
                    tar.add(docs_dir, arcname="documents")
                
                # Backup metadata
                metadata_dir = Path(self.config['metadata_storage_path'])
                if metadata_dir.exists():
                    tar.add(metadata_dir, arcname="metadata")
            
            self.logger.info(f"Files backup created: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            self.logger.error(f"Files backup failed: {str(e)}")
            raise

    async def upload_to_s3(self, file_path: str) -> bool:
        """Upload backup file to S3."""
        if not self.s3_client:
            return False
            
        try:
            bucket = self.config['s3_bucket']
            key = f"backups/{os.path.basename(file_path)}"
            
            self.s3_client.upload_file(file_path, bucket, key)
            self.logger.info(f"Backup uploaded to S3: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"S3 upload failed: {str(e)}")
            return False

    async def cleanup_old_backups(self, max_age_days: int = 30):
        """Remove old backup files."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            
            for backup_file in self.backup_dir.glob("*"):
                if backup_file.is_file():
                    file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        backup_file.unlink()
                        self.logger.info(f"Removed old backup: {backup_file}")
                        
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {str(e)}")

    async def run_backup(self):
        """Run a complete backup."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Create backup manifest
            manifest = {
                "timestamp": timestamp,
                "files": {}
            }
            
            # Run all backups
            db_backup = await self.backup_database()
            manifest["files"]["database"] = os.path.basename(db_backup)
            
            qdrant_backup = await self.backup_qdrant()
            manifest["files"]["qdrant"] = os.path.basename(qdrant_backup)
            
            files_backup = await self.backup_files()
            manifest["files"]["files"] = os.path.basename(files_backup)
            
            # Save manifest
            manifest_file = self.backup_dir / f"backup_manifest_{timestamp}.json"
            async with aiofiles.open(manifest_file, 'w') as f:
                await f.write(json.dumps(manifest, indent=2))
            
            # Upload to S3 if configured
            if self.s3_client:
                for backup_file in [db_backup, qdrant_backup, files_backup, manifest_file]:
                    await self.upload_to_s3(backup_file)
            
            # Cleanup old backups
            await self.cleanup_old_backups(
                max_age_days=self.config.get('backup_retention_days', 30)
            )
            
            self.logger.info(f"Backup completed successfully: {timestamp}")
            return manifest
            
        except Exception as e:
            self.logger.error(f"Backup failed: {str(e)}")
            raise

    def schedule_backups(self, schedule_config: Dict[str, Any]):
        """Schedule automated backups."""
        def run_backup_job():
            asyncio.run(self.run_backup())
        
        # Schedule daily backup
        if schedule_config.get('daily'):
            schedule.every().day.at(schedule_config['daily']).do(run_backup_job)
        
        # Schedule weekly backup
        if schedule_config.get('weekly'):
            schedule.every().week.at(schedule_config['weekly']).do(run_backup_job)
        
        # Run scheduler in a separate thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
