# Disaster Recovery Plan

This document outlines the disaster recovery (DR) procedures and strategies for the system.

**Purpose:** To ensure business continuity by enabling the recovery of critical systems and data in the event of a disaster.

## 1. Introduction

*   **Scope:**
    *   [Systems and data covered by this DR plan]
*   **Objectives:**
    *   Recovery Time Objective (RTO)
    *   Recovery Point Objective (RPO)
*   **Assumptions:**
    *   [Any assumptions made in the formulation of this plan]

## 2. Roles and Responsibilities

*   **DR Team:**
    *   [List of team members and their roles during a DR event]
*   **Communication Plan:**
    *   [How communication will be managed during a disaster]

## 3. Backup Procedures

*   **Data Backup:**
    *   **Databases:**
        *   [Backup frequency, method (e.g., snapshots, dumps), and storage location]
        *   [Retention policies]
    *   **Application Data:**
        *   [Backup procedures for persistent application data]
    *   **Configuration Data:**
        *   Infrastructure configurations (Kubernetes manifests, Helm charts, Dockerfiles) are version-controlled in Git.
        *   Application configurations (`config.yaml`, environment variables) should be managed securely, potentially using Kubernetes ConfigMaps/Secrets, which are also version-controlled or backed up as part of cluster state.

    *   **Relational Databases (e.g., PostgreSQL - if used and managed by `BackupManager`):**
        *   **Backup Method**: Utilizes `pg_dump` for logical backups, executed by `Scripts/monitoring/backup_manager.py`.
        *   **Frequency**: Configurable (e.g., daily). See `BackupManager` scheduling.
        *   **Storage**: Local backup files (`.sql`) are created in the `backup_manager`'s configured directory. These can be automatically uploaded to S3 (or other cloud storage) if `BackupManager` is configured for it.
        *   **Encryption**: Placeholder for encryption exists in `BackupManager` before cloud upload. Key management for encryption is crucial.
        *   **Retention**: Managed by `BackupManager`'s cleanup logic (local) and S3 lifecycle policies (cloud).
        *   **Restoration**:
            1.  Download the required `.sql` backup file from S3 or local backup storage.
            2.  If encrypted, decrypt the file.
            3.  Provision a new PostgreSQL instance.
            4.  Restore using `psql -U <user> -d <database> -f <backup_file.sql>`.
            5.  Verify data integrity.
        *   **Point-in-Time Recovery (PITR)**: `pg_dump` provides a point-in-time snapshot. True PITR for PostgreSQL requires continuous WAL archiving, which is an advanced DB administration setup beyond the current `BackupManager`'s direct scope but can complement these backups.

    *   **Vector Database (Qdrant)**:
        *   **Context**: Qdrant stores document embeddings and metadata, critical for the RAG functionality.
        *   **Recommended Backup Methods**:
            1.  **Qdrant Snapshots (Primary Method)**:
                *   Qdrant provides API endpoints for creating and managing snapshots of collections (`POST /collections/{collection_name}/snapshots`) or the entire instance (`POST /snapshots`).
                *   These snapshots are stored on the Qdrant node's filesystem (on its PersistentVolume in Kubernetes).
                *   Snapshots capture a consistent state of the data.
            2.  **Filesystem-Level Backups (Volume Snapshots - Secondary/Complementary)**:
                *   If Qdrant uses PersistentVolumes (PVs) in Kubernetes, the underlying storage provider (e.g., AWS EBS, GCE Persistent Disk) usually offers volume snapshot capabilities.
                *   **Caution**: For data consistency, application-aware snapshots (like Qdrant's own) are preferred. If using volume snapshots, it's best if Qdrant can be quiesced or if the snapshot is taken during low activity, though Qdrant is designed to recover from unclean shutdowns.
        *   **Conceptual Backup Strategy & Procedure (using Qdrant Snapshots)**:
            *   **Frequency**:
                *   Full Snapshots: Daily (or more frequently based on RPO and data change rate).
            *   **Automation**: Implement a Kubernetes CronJob.
            *   **Steps within CronJob**:
                1.  **Trigger Snapshot**: The CronJob pod executes a script (e.g., Python with `requests` or `curl`) to call the Qdrant snapshot API for the relevant collection(s) (e.g., `documents`).
                    ```bash
                    # Example: curl -X POST http://qdrant-service.default.svc.cluster.local:6333/collections/documents/snapshots \
                    # -H "api-key: <your-qdrant-api-key-if-any>"
                    # (Adjust service URL as needed)
                    ```
                2.  **Retrieve Snapshot from Qdrant Volume**: The snapshot API response typically includes the name/path of the snapshot file(s) on Qdrant's PV. The CronJob pod then needs to copy these files from the Qdrant pod's volume. This can be achieved by:
                    *   Having shared access to the PV if architecture allows (less common for CronJobs).
                    *   Using `kubectl cp` from the CronJob pod to copy from the Qdrant pod (requires `kubectl` in the CronJob image and appropriate RBAC permissions).
                    *   Having an agent or sidecar on the Qdrant pod that can push snapshots to backup storage.
                3.  **Store Snapshot Externally**: Upload the retrieved snapshot files to a durable, external backup storage location (e.g., AWS S3, Google Cloud Storage, Azure Blob Storage). Use versioning and encryption on the backup storage.
                4.  **Manage Snapshots in Qdrant**: Periodically, list snapshots in Qdrant (`GET /collections/{collection_name}/snapshots`) and delete older ones from Qdrant's local storage (`DELETE /collections/{collection_name}/snapshots/{snapshot_name}`) once they are securely backed up externally, to manage disk space on Qdrant's PV.
        *   **Retention Policy (External Backups)**:
            *   Define based on RPO and compliance (e.g., keep daily for 7 days, weekly for 4 weeks, monthly for 6 months).
        *   **Conceptual Restoration Strategy**:
            1.  **Provision New Qdrant Instance**: If the primary instance is lost.
            2.  **Retrieve Snapshot**: Download the required snapshot from external backup storage.
            3.  **Place Snapshot**: Make the snapshot file(s) accessible to the new Qdrant instance (e.g., by copying onto its PV).
            4.  **Restore via API**:
                *   The exact Qdrant API endpoint to restore a snapshot might depend on the version and how snapshots are managed (e.g., `POST /collections/{collection_name}/snapshots/upload` or by placing snapshot in a recovery directory and using a recovery mode).
                *   Consult the official Qdrant documentation for the specific API call to recover a collection from a snapshot file. The `BackupManager` currently backs up snapshot files; restoration is a manual Qdrant admin task using these files.
            5.  **Verify**: Check collection existence, point counts, and perform sample search queries.

    *   **AI Models (Ollama)**:
        *   **Context**: Ollama downloads and stores large language models locally, typically within its container's filesystem (e.g., `/root/.ollama` by default), which should be mapped to a PersistentVolume (PV) when deployed in Kubernetes (as done in `ollama-statefulset.yaml`).
        *   **Backup Strategy**:
            1.  **PersistentVolume Snapshots**: The primary method is to take regular snapshots of the PV where Ollama stores its models. This should be done using the storage provider's snapshot capabilities (e.g., AWS EBS snapshots, GCE Persistent Disk snapshots, Ceph RBD snapshots, etc.).
            2.  **Model File Backup (Alternative/Complementary)**: If direct PV snapshotting is complex or for added safety, a script could periodically list models (`ollama list`) and then copy the model blobs from Ollama's data directory (inside the pod, from its PV) to an external backup storage (e.g., S3). This is more complex to manage consistently.
        *   **Frequency**: Depends on how often new models are pulled or fine-tuned (if applicable). If models are relatively static after initial setup, less frequent backups (e.g., weekly or after major model changes) might suffice for the model files themselves.
        *   **Storage**: Backups (PV snapshots or model files) should be stored durably and securely, ideally in a different region.
        *   **Restoration Strategy**:
            1.  **Provision New Ollama Instance with PV**: Ensure a new Ollama instance is set up with a PV.
            2.  **Restore PV from Snapshot**: Restore the PV from the chosen snapshot.
            3.  **Start Ollama**: Ollama should then recognize the models present on its restored volume.
            4.  **(If using file backup)**: Copy model files back to the new Ollama instance's data directory on its PV, then restart Ollama. It should rescan and register the models.
            5.  **Verification**: List models (`ollama list`) and test inference with a key model.

    *   **Message Queue (RabbitMQ)**:
        *   **Context**: If RabbitMQ is used with durable queues and persistent messages, its data (message store, configurations, user metadata) is stored on a PersistentVolume (if persistence is enabled in its Helm chart, e.g., Bitnami's RabbitMQ chart allows this).
        *   **Backup Strategy**:
            1.  **PersistentVolume Snapshots**: If RabbitMQ persistence is enabled and uses PVs, take regular snapshots of these PVs using the storage provider's tools. This is the most straightforward way to back up the entire state, including messages in durable queues.
            2.  **RabbitMQ Definitions Export**: RabbitMQ's management plugin allows exporting definitions (users, vhosts, queues, exchanges, policies, etc.) as a JSON file. This should be done regularly and stored externally. This backs up the *structure and configuration* but not the messages themselves.
                *   Can be automated via `rabbitmqadmin` CLI or HTTP API calls.
            3.  **Message Backup (Application-Level or Shovel/Federation - Advanced)**: For critical messages that cannot be lost and where PV snapshots might have limitations (e.g., RPO too high), consider:
                *   Application-level backup: Store critical messages in a separate durable database before or after queuing.
                *   RabbitMQ Shovel or Federation plugins to replicate messages to another RabbitMQ cluster or a different message system in another location (more of a high-availability/DR replication strategy than simple backup).
        *   **Frequency**:
            *   PV Snapshots: Daily or based on RPO for message data.
            *   Definitions Export: Daily or after any configuration change.
        *   **Storage**: Store PV snapshots and definition JSON files securely and durably externally.
        *   **Restoration Strategy**:
            1.  **Provision New RabbitMQ Instance**: With PVs if message persistence is key.
            2.  **Restore PVs from Snapshot**: If PV snapshots were taken.
            3.  **Start RabbitMQ**.
            4.  **Import Definitions**: If restoring to a fresh instance or if definitions are suspect, import the last known good definitions JSON via the management UI or `rabbitmqadmin`.
            5.  **Verification**: Check vhosts, queues, exchanges, user permissions. Test message publishing and consumption.

    *   **Cache (Redis)**:
        *   **Context**: Redis is currently used by the RAG Query Service for caching Qdrant search results and LLM-generated answers to improve performance and reduce load on downstream services.
        *   **Default Assumption (Volatile Cache)**:
            *   For its current use case as a cache, data stored in Redis is considered volatile and can be rebuilt or repopulated by the application if Redis is restarted or data is lost.
            *   If Redis is deployed without persistence enabled (e.g., default for many simple Helm chart setups for caching, or using `emptyDir` volumes in K8s), no specific data backup procedures are strictly necessary for disaster recovery *of the cache content itself*. The primary DR concern is the quick redeployment/availability of the Redis service.
        *   **Backup Strategy (If Redis Persistence is Enabled and Deemed Critical)**:
            *   If, in the future, Redis is used to store more critical, non-transient data, or if even cached data is very expensive to regenerate and warrants persistence, then persistence must be enabled in the Redis deployment (e.g., RDB snapshots + AOF logging, configured with PVs).
            *   **Methods**:
                1.  **Redis RDB Snapshots**: Redis can be configured to periodically save snapshots of its dataset to an `.rdb` file.
                    *   These RDB files, stored on Redis's PV, would then need to be backed up to external, durable storage (e.g., S3, GCS). This can be done via a K8s CronJob that copies the RDB file.
                2.  **Append-Only File (AOF)**: Provides better durability than RDB snapshots alone. The AOF log can also be backed up.
                3.  **PersistentVolume (PV) Snapshots**: If Redis is using a PV for its data directory, the underlying storage provider's PV snapshot mechanism is the most common way to back up its state.
            *   **Frequency**: Depends on the criticality of the data if Redis is used beyond a simple cache (e.g., daily PV snapshots).
            *   **Storage**: External, durable, secure storage.
        *   **Restoration Strategy (If Persistent Data was Backed Up)**:
            1.  **Provision New Redis Instance**: With PVs configured for persistence.
            2.  **Restore PV from Snapshot**: If PV snapshots were used.
            3.  **Or, Restore RDB file**: If RDB file backups were taken, place the `.rdb` file in the new Redis instance's data directory before starting Redis. Redis will load it on startup.
            4.  **Start Redis**.
            5.  **Verification**: Check connectivity and if data (if any was expected to persist) is present.
        *   **Current System Implication**: For the RAG Query Service cache, if Redis restarts without persisted data, the service will experience cache misses, leading to increased latency and load on Qdrant/Ollama until the cache repopulates. This performance degradation is the primary impact, not data loss in the traditional sense for a cache. The DR plan should focus on rapidly restoring Redis *service availability*.

*   **Infrastructure Backup:**
    *   [Backup of IaC scripts (Terraform, Ansible)]
    *   [Backup of Kubernetes cluster state (etcd backups)]
*   **Verification:**
    *   [Procedures for regularly verifying backup integrity and restorability]

## 4. Recovery Procedures

*   **Declaration of Disaster:**
    *   [Criteria for declaring a disaster]
    *   [Process for initiating the DR plan]
*   **Failover to DR Site/Region:**
    *   [Step-by-step instructions for failing over to the DR environment]
    *   [DNS updates, traffic redirection]
*   **Data Restoration:**
    *   [Procedures for restoring data from backups]
    *   [Order of restoration for different systems]
*   **Application Recovery:**
    *   [Steps to bring applications online in the DR environment]
*   **Verification and Testing:**
    *   [How to verify that systems are functioning correctly post-recovery]

## 5. Incident Response Playbooks

*   **Playbook 1: Data Center Outage**
    *   **Detection:** Monitoring alerts indicate loss of connectivity to the primary data center/region.
    *   **Initial Assessment:** DR Team Lead verifies the outage scope and impact. Estimate time to recovery of primary site.
    *   **Decision:** If RTO for primary site recovery is exceeded, DR Team Lead declares a disaster and initiates failover.
    *   **Execution:**
        1.  Execute infrastructure failover scripts/procedures (e.g., promote DR Kubernetes cluster, update DNS to DR IP addresses).
        2.  Restore critical databases (PostgreSQL, Qdrant) from the latest available backups in the DR region (see data restoration procedures above).
        3.  Deploy/scale up application services in the DR Kubernetes cluster using Helm charts, pointing to restored data sources.
        4.  Restore application data from backups if necessary.
        5.  Perform system health checks and functional testing in the DR environment.
        6.  Communicate status to stakeholders.
    *   **Post-Recovery:** Monitor DR environment stability. Plan for failback to primary site once it's restored and stable.
*   **Playbook 2: Qdrant Vector Database Data Loss/Corruption**
    *   **Detection:** Application errors related to vector search, inability to query Qdrant, monitoring alerts for Qdrant health or data consistency.
    *   **Initial Assessment:**
        1.  Verify the extent of data loss/corruption (specific collections, all collections).
        2.  Check Qdrant logs for error messages.
        3.  Attempt basic recovery steps if suggested by Qdrant logs (e.g., restarting pods).
    *   **Decision:** If data is confirmed lost/corrupted and basic recovery fails, proceed with restoration from backup.
    *   **Execution (Restore from Qdrant Snapshot):**
        1.  Identify the latest known good snapshot of the affected Qdrant collection(s) from backup storage (e.g., S3).
        2.  (If applicable) Scale down or temporarily stop application services that write to/read from the affected Qdrant collection to prevent further issues or inconsistent reads during restoration.
        3.  Download the snapshot file(s) to a location accessible by the Qdrant cluster or a utility pod.
        4.  Follow Qdrant's official procedure for restoring a collection from a snapshot. This might involve:
            *   Ensuring the target collection does not exist or is empty.
            *   Using a Qdrant API endpoint (e.g., related to `snapshots/recover` or by placing snapshot in a specific recovery directory and restarting/configuring Qdrant). Refer to Qdrant documentation for the precise method for your version.
        5.  Once Qdrant confirms restoration is complete, verify the collection's health, point count, and perform sample queries.
        6.  (If applicable) Scale up or restart application services.
        7.  Monitor application logs and Qdrant metrics closely post-restoration.
    *   **Post-Recovery:**
        1.  Investigate the root cause of data loss/corruption.
        2.  Review backup frequency and snapshot integrity if the RPO was not met.
*   **Playbook 3: Major Relational Database (e.g., PostgreSQL) Data Corruption**
    *   [Specific steps to take, similar to Qdrant: identify backup, stop writes, restore using `psql`, verify, restart services]
*   **Playbook 4: Ransomware Attack**
    *   [Isolate affected systems, engage security team, determine blast radius, restore from immutable backups to a clean environment, forensic analysis]
*   *[Add more playbooks as needed for different scenarios]*

## 6. Capacity Planning

*   **Current Capacity:**
    *   [Overview of current resource capacity in primary and DR environments]
*   **Scalability:**
    *   [Strategies for scaling resources in the DR environment if needed]
*   **Monitoring:**
    *   [How capacity is monitored to ensure DR readiness]
*   **Forecasting:**
    *   [Methods for forecasting future capacity needs]

## 7. DR Drills and Testing

*   **Frequency:**
    *   [How often DR drills are conducted]
*   **Types of Drills:**
    *   [Tabletop exercises, partial failover tests, full failover tests]
*   **Post-Drill Review:**
    *   [Process for reviewing drill results and updating the DR plan]

## 8. Plan Maintenance

*   **Review Cycle:**
    *   [How often the DR plan is reviewed and updated]
*   **Update Triggers:**
    *   [Events that trigger a review/update (e.g., major system changes, post-incident)]

*TODO: Fill in with specific details, contact lists, RTO/RPO values, and precise procedures for each section.*
