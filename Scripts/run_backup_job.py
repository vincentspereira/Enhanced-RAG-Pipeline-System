import os
import asyncio
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

# Assuming BackupManager is in Scripts/monitoring/backup_manager.py
# Adjust import path if necessary based on how this script is run or packaged.
# For now, assume it can be imported if PYTHONPATH includes /app (common in Docker)
from monitoring.backup_manager import BackupManager

# Setup basic logging for the job
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Log to stdout/stderr, which K8s will capture
    ]
)
logger = logging.getLogger("run_backup_job")

DEFAULT_CONFIG_PATH = "/app/config/backup_job_config.yaml" # Default path for mounted ConfigMap

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads backup job configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded backup configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config {config_path}: {e}")
        raise

def get_env_var(var_name: str, is_required: bool = True, default_value: Any = None) -> Optional[str]:
    """Helper to get environment variables, with logging."""
    value = os.getenv(var_name)
    if value is None:
        if is_required and default_value is None:
            logger.error(f"Required environment variable {var_name} is not set.")
            raise ValueError(f"Missing required environment variable: {var_name}")
        logger.info(f"Environment variable {var_name} not set, using default: {default_value}")
        return default_value
    logger.info(f"Environment variable {var_name} loaded.")
    return value

async def main():
    logger.info("Starting backup job...")

    try:
        # Load non-sensitive config from file (mounted ConfigMap)
        config_file_path = get_env_var("BACKUP_CONFIG_FILE_PATH", is_required=False, default_value=DEFAULT_CONFIG_PATH)
        backup_job_config = load_config(config_file_path)

        # Prepare BackupManager configuration, prioritizing environment variables for secrets
        manager_config: Dict[str, Any] = {
            "backup_dir": backup_job_config.get("backup_dir", "/app/backups_data"), # Ensure this path is writable in the pod
            "log_dir": backup_job_config.get("log_dir", "/app/logs/backup_job_logs"), # Ensure this path is writable

            # Database - Assuming PostgreSQL for now as per BackupManager's current capability
            # For other DBs, BackupManager would need extension.
            "database_url": get_env_var("DB_URL_FOR_BACKUP", is_required=False), # e.g., postgresql://user:pass@host:port/db

            # Qdrant
            "qdrant_url": get_env_var("QDRANT_URL_FOR_BACKUP", is_required=True), # e.g., http://qdrant-service.rag-pipeline.svc.cluster.local:6333
            "qdrant_api_key": get_env_var("QDRANT_API_KEY_FOR_BACKUP", is_required=False),

            # File paths to backup (these should be paths accessible within the backup pod if different from app pod)
            # Or, if backing up PVs, this script would need to mount them.
            # Current BackupManager assumes paths are directly accessible.
            # For K8s, it's often better if the app itself places data in a backup-able location or if DB tools are used.
            "document_storage_path": backup_job_config.get("document_storage_path", "/app/documents_to_backup"),
            "metadata_storage_path": backup_job_config.get("metadata_storage_path", "/app/metadata_to_backup"),

            # S3 Configuration
            "use_s3": backup_job_config.get("use_s3", False),
            "aws_access_key": get_env_var("AWS_ACCESS_KEY_ID_FOR_BACKUP", is_required=backup_job_config.get("use_s3", False)),
            "aws_secret_key": get_env_var("AWS_SECRET_ACCESS_KEY_FOR_BACKUP", is_required=backup_job_config.get("use_s3", False)),
            "aws_region": backup_job_config.get("aws_region", "us-east-1"),
            "s3_bucket": backup_job_config.get("s3_bucket_name"), # Ensure this is set in config if use_s3 is true

            "backup_retention_days": backup_job_config.get("backup_retention_days", 30)
        }

        if manager_config["use_s3"] and not manager_config["s3_bucket"]:
            logger.error("S3 backups are enabled but S3_BUCKET_NAME is not configured.")
            raise ValueError("S3_BUCKET_NAME must be set if use_s3 is true.")

        # Create directories if they don't exist (for local backup storage within the pod)
        Path(manager_config["backup_dir"]).mkdir(parents=True, exist_ok=True)
        Path(manager_config["log_dir"]).mkdir(parents=True, exist_ok=True)


        logger.info(f"Initializing BackupManager with config: {{backup_dir='{manager_config['backup_dir']}', use_s3='{manager_config['use_s3']}', ...}}")
        backup_manager = BackupManager(config=manager_config)

        logger.info("Running backup process...")
        manifest = await backup_manager.run_backup()

        if manifest:
            logger.info(f"Backup process completed successfully. Manifest: {manifest}")
        else:
            logger.error("Backup process completed but returned no manifest.")

    except ValueError as ve: # Catch configuration errors
        logger.error(f"Configuration error: {ve}")
        # Consider exiting with a specific error code for K8s to identify config issues
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during the backup job: {e}", exc_info=True)
        # Exit with a non-zero code to indicate failure to K8s
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
