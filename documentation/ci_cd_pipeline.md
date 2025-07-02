# CI/CD Pipeline Architecture (GitHub Actions)

This document describes the architecture of the Continuous Integration and Continuous Deployment (CI/CD) pipeline implemented using GitHub Actions. The workflow file is located at `.github/workflows/ci-cd.yml`.

## 1. Workflow Triggers

The CI/CD pipeline is triggered on the following GitHub events:

*   **Push**: When code is pushed to the `main` or `develop` branches.
*   **Pull Request**: When a pull request is opened or updated targeting the `main` or `develop` branches.

## 2. Jobs Overview

The pipeline consists of several jobs that run in parallel or sequentially based on their dependencies (`needs` keyword).

### 2.1. `test` Job

*   **Purpose**: Runs unit tests across multiple Python versions.
*   **Strategy**:
    *   Runs on `ubuntu-latest`.
    *   Uses a matrix strategy for Python versions: `3.9`, `3.10`, `3.11`.
*   **Services**:
    *   `qdrant`: Starts a Qdrant service (`qdrant/qdrant:latest`) available for tests.
*   **Key Steps**:
    1.  `actions/checkout@v3`: Checks out the repository code.
    2.  `actions/setup-python@v4`: Sets up the specified Python version from the matrix.
    3.  `actions/cache@v3`: Caches pip packages to speed up dependency installation.
    4.  `Install dependencies`: Installs project dependencies from `requirements.txt` and testing tools (`pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-timeout`).
    5.  `Run tests`: Executes tests using `pytest tests/ --cov=Scripts --cov-report=xml -v`. This runs tests from the `tests/` directory, generates coverage reports for the `Scripts` directory in XML format, and uses verbose output.
    6.  `Upload coverage to Codecov`: Uses `codecov/codecov-action@v3` to upload the `coverage.xml` report to Codecov.

### 2.2. `integration_test` Job

*   **Purpose**: Runs integration tests that may require multiple services.
*   **Strategy**: Runs on `ubuntu-latest` with Python `3.10`.
*   **Services**:
    *   `postgres`: Starts a PostgreSQL service (`postgres:latest`) with predefined credentials and database.
    *   `qdrant`: Starts a Qdrant service (`qdrant/qdrant:latest`).
*   **Key Steps**:
    1.  `actions/checkout@v3`: Checks out code.
    2.  `actions/setup-python@v4`: Sets up Python `3.10`.
    3.  `Install dependencies`: Installs project and testing dependencies.
    4.  `Run tests`: Executes tests using `pytest --cov=. --cov-report=xml`. Environment variables `DATABASE_URL` and `QDRANT_URL` are set for the tests to connect to the services.
    5.  `Upload coverage report`: Uploads `coverage.xml` to Codecov.

### 2.3. `lint` Job

*   **Purpose**: Performs static code analysis (linting and formatting checks).
*   **Strategy**: Runs on `ubuntu-latest` with Python `3.10`.
*   **Key Steps**:
    1.  `actions/checkout@v3`: Checks out code.
    2.  `actions/setup-python@v4`: Sets up Python `3.10`.
    3.  `Install linting dependencies`: Installs `flake8`, `black`, `isort`, `mypy`.
    4.  `Run linters`: Executes `flake8 .`, `black . --check`, `isort . --check`, and `mypy .`.

### 2.4. `security-scan` Job

*   **Purpose**: Performs security vulnerability scanning.
*   **Strategy**: Runs on `ubuntu-latest`.
*   **Key Steps**:
    1.  `actions/checkout@v3`: Checks out code.
    2.  `Run security scan`: Uses `snyk/actions/python@master` to scan the project. Requires `SNYK_TOKEN` secret. Scans with a high severity threshold.

### 2.5. `api_hub_test` Job

*   **Purpose**: Runs tests specific to the API Hub components.
*   **Needs**: Depends on the successful completion of the `test` job.
*   **Strategy**: Runs on `ubuntu-latest` with Python `3.10`.
*   **Services**:
    *   `qdrant`: Starts a Qdrant service.
*   **Key Steps**:
    1.  Checkout, Python setup, dependency installation (including API Hub specific test tools).
    2.  `Run API Hub tests`: Executes `pytest tests/simplified_api_hub_test.py tests/integration_api_hub_test.py tests/e2e_api_hub_test.py -v --cov=Scripts.api_v1 --cov-report=xml`.
    3.  `Upload API Hub coverage report`: Uploads `coverage.xml` to Codecov.

### 2.6. `vector_search_test` Job

*   **Purpose**: Runs tests specific to vector search functionality.
*   **Needs**: Depends on the successful completion of the `test` job.
*   **Strategy**: Runs on `ubuntu-latest` with Python `3.10`.
*   **Services**:
    *   `qdrant`: Starts a Qdrant service.
*   **Key Steps**:
    1.  Checkout, Python setup, dependency installation.
    2.  `Run Vector Search tests`: Executes `pytest tests/test_vector_search.py -v --cov=Scripts.vector_search --cov-report=xml`.
    3.  `Upload Vector Search coverage report`: Uploads `coverage.xml` to Codecov.

### 2.7. `performance_test` Job

*   **Purpose**: Runs performance tests.
*   **Needs**: Depends on successful completion of `api_hub_test` and `vector_search_test` jobs.
*   **Strategy**: Runs on `ubuntu-latest` with Python `3.10`.
*   **Services**:
    *   `qdrant`: Starts a Qdrant service.
*   **Key Steps**:
    1.  Checkout, Python setup, dependency installation (including `pytest-benchmark`).
    2.  `Run Performance tests`: Executes `pytest tests/test_performance.py -v`.

### 2.8. `build-and-push` Job

*   **Purpose**: Builds a Docker image and pushes it to Docker Hub.
*   **Condition**: Only runs if the event is a push to the `main` branch.
*   **Needs**: Depends on successful completion of `test`, `lint`, and `security-scan` jobs.
*   **Strategy**: Runs on `ubuntu-latest`.
*   **Key Steps**:
    1.  `actions/checkout@v3`: Checks out code.
    2.  `docker/setup-buildx-action@v2`: Sets up Docker Buildx.
    3.  `docker/login-action@v2`: Logs in to Docker Hub using `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets.
    4.  `docker/build-push-action@v4`: Builds the Docker image from the current context (`.`) and pushes it.
        *   Tags: `user/rag-pipeline:latest` and `user/rag-pipeline:${{ github.sha }}` (Note: `user/rag-pipeline` should be replaced with the actual Docker Hub repository name).
        *   Caching: Uses registry caching (`user/rag-pipeline:buildcache`) to speed up builds.

### 2.9. `deploy` Job

*   **Purpose**: Placeholder for deploying the application to a production environment.
*   **Condition**: Only runs if the event is a push to the `main` branch.
*   **Needs**: Depends on successful completion of `lint`, `integration_test`, `api_hub_test`, `vector_search_test`, and `performance_test` jobs.
*   **Strategy**: Runs on `ubuntu-latest`.
*   **Key Steps**:
    1.  `actions/checkout@v3`: Checks out code.
    2.  Python setup and dependency installation.
    3.  `Deploy to production`: Currently an echo command stating "Deploying to production environment" and executes `bash deployment/deploy.sh`. The actual deployment logic would be in `deployment/deploy.sh`.

## 3. Deployment Strategy (Conceptual based on `deploy` job)

*   The `deploy` job suggests a script-based deployment (`deployment/deploy.sh`).
*   The actual deployment strategy (e.g., to Kubernetes, serverless, etc.) would be detailed within that script.
*   Rollback procedures and specific strategies like blue-green deployments are not explicitly defined in the GitHub Actions workflow itself but would need to be implemented within the `deploy.sh` script or the target deployment platform's tooling.

## 4. Future Enhancements (Placeholder)

*   More sophisticated environment management (e.g., staging, production).
*   Integration with Infrastructure as Code (IaC) tools like Terraform or Ansible.
*   Automated rollback procedures based on health checks post-deployment.
*   Implementation of blue-green or canary deployment strategies.

*This document reflects the CI/CD pipeline as defined in `.github/workflows/ci-cd.yml` at the time of writing. It should be updated as the workflow evolves.*
