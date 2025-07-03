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
        *   [How infrastructure and application configurations are backed up (e.g., Git, S3)]

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
            4.  **Restore via API**: Use Qdrant's API to restore the collection from the snapshot file(s). (Refer to specific Qdrant version documentation for exact API endpoints and procedures, as this might involve uploading the snapshot through an API or placing it in a predefined recovery directory).
            5.  **Verify**: Check data integrity and search functionality.

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
        *   **Current System Implication**: For the RAG Query Service cache, if Redis restarts without persisted data (i.e., persistence is disabled in its deployment), the service will experience cache misses. This leads to increased latency and higher load on Qdrant and Ollama until the cache repopulates. This performance degradation is the primary impact, not critical data loss. The DR plan for a non-persistent cache focuses on rapidly restoring Redis *service availability* (e.g., ensuring the Redis deployment/StatefulSet can restart quickly).
        *   **If using Bitnami Redis Helm Chart with Persistence**: If `redis.enabled: true` in the parent chart and the Bitnami subchart is configured with `persistence.enabled: true` (e.g., for a more durable cache or other uses), it will use a PersistentVolumeClaim (PVC). In this case, the DR strategy for Redis data aligns with general PV snapshot procedures:
            *   Regularly snapshot the PV used by Redis.
            *   Restoration involves restoring the PV from the snapshot and ensuring the new Redis pod/StatefulSet attaches to it.

### 3.y Application & System Configurations Backup

*   **Context**: This covers the application code, Kubernetes manifests, Helm charts, Dockerfiles, and any critical non-sensitive configuration files not managed as K8s Secrets.
*   **Backup Strategy**:
    *   **Version Control (Git)**: All code, Dockerfiles, Kubernetes YAML manifests (`deployment/local_k8s/`), Helm chart sources (`charts/rag-system/`), and general documentation (`docs/`, `documentation/`) MUST be stored in a Git repository (e.g., GitHub, GitLab).
    *   **Regular Commits & Pushes**: Developers should commit and push changes frequently to a central Git repository.
    *   **Branching Strategy**: A sound branching strategy (e.g., Gitflow, GitHub Flow) should be used to manage features, releases, and hotfixes.
    *   **Backup of Git Repository**: The Git hosting provider (e.g., GitHub) is responsible for the primary backup of the repository itself. For extra precaution, organizations might implement their own periodic clones/backups of critical repositories to a separate storage location.
*   **Restoration Strategy**:
    *   **Code & Manifests**: `git clone` the repository to the desired commit/branch/tag.
    *   **Deployment**: Use the cloned Helm charts (`helm install/upgrade`) or Kubernetes manifests (`kubectl apply`) to redeploy the application and its configuration.
    *   **Sensitive Configurations (K8s Secrets)**: As noted elsewhere, Kubernetes Secrets (for API keys, passwords) should be managed and backed up according to their own secure procedures (e.g., backed up as part of etcd backups, or managed via tools like Sealed Secrets or Vault which have their own DR processes). This Git repository should primarily store the *definitions* of how these secrets are used (e.g., Helm templates referencing secret names), not the secret data itself.

*   **Infrastructure Backup:**
    *   [Backup of IaC scripts (Terraform, Ansible for K8s cluster provisioning itself, if applicable)]
    *   [Backup of Kubernetes cluster state (etcd backups) - This is critical for full cluster DR]
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
    *   [Specific steps to take]
*   **Playbook 2: Major Data Corruption**
    *   [Specific steps to take]
*   **Playbook 3: Ransomware Attack**
    *   [Specific steps to take]
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
