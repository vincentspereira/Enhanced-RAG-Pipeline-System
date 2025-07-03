# Automated Test Plan

## 1. Introduction

This document outlines the automated testing strategy for the RAG (Retrieval Augmented Generation) system. The goal is to ensure code quality, system reliability, and prevent regressions through a comprehensive suite of automated tests integrated into the CI/CD pipeline.

## 2. Scope of Automated Testing

Automated tests cover the following key areas:

*   **Unit Tests:** Verify the correctness of individual functions, methods, and classes in isolation.
*   **Integration Tests:** Test the interactions between different components or microservices within the system.
*   **End-to-End (E2E) Tests:** Validate complete user workflows and system functionalities from an external perspective.
*   **Performance Tests:** Measure and assert the performance characteristics (e.g., latency, throughput) of critical system components under load.
*   **API Contract Tests:** (Implicitly covered by E2E and Integration tests) Ensure API endpoints adhere to their defined contracts (request/response schemas, status codes).

## 3. Testing Tools and Frameworks

*   **Primary Testing Framework:** `pytest` is used for writing and running all types of Python-based tests.
*   **Mocking:** `pytest-mock` (based on `unittest.mock`) for creating mocks and stubs.
*   **Asynchronous Testing:** `pytest-asyncio` for testing `async` code.
*   **Code Coverage:** `pytest-cov` (using `coverage.py`) to measure test coverage.
*   **Performance Testing:** `pytest-benchmark` for micro-benchmarks, and custom scripts/load generators (e.g., Locust, k6 - conceptual for now) for system-level performance tests.
*   **CI/CD Integration:** GitHub Actions orchestrates the execution of automated tests.

## 4. Test Structure and Location

*   All tests are located in the `/tests` directory at the root of the repository.
*   **Unit Tests:** `tests/unit/`
    *   Subdirectories often mirror the structure of the `Scripts/` directory.
    *   Example: `tests/unit/test_api_hub.py`
*   **Integration Tests:** `tests/integration/`
    *   Focus on interactions between services or components.
    *   Example: `tests/integration/test_api_hub_integration.py`, `tests/integration/test_doc_processing_service.py`
*   **End-to-End Tests:** `tests/e2e/`
    *   Test full system flows.
    *   Example: `tests/e2e/test_e2e_workflows.py`
*   **Performance Tests:** `tests/test_performance.py`
    *   Contains tests designed to measure response times, throughput, etc.
*   **Specialized Tests:**
    *   `tests/test_vector_search.py`: Specific tests for vector search capabilities.
    *   `tests/simplified_api_hub_test.py`, `tests/e2e_api_hub_test.py`: Additional API Hub tests.
*   **Fixtures:** Common test fixtures are defined in `tests/conftest.py`.

## 5. Test Execution

### Local Execution

*   Developers can run tests locally using `pytest` or the provided `Scripts/run_tests.py` script.
*   Refer to `docs/TESTING_GUIDE.md` for detailed instructions on setting up the environment and running tests.
*   Example commands:
    *   Run all tests: `pytest`
    *   Run specific test file: `pytest tests/unit/test_example.py`
    *   Run tests with coverage: `pytest --cov=Scripts`

### CI/CD Pipeline Execution

*   Automated tests are executed automatically on every push and pull request to `main` and `develop` branches via GitHub Actions (`.github/workflows/ci-cd.yml`).
*   The CI pipeline includes jobs for:
    *   Unit tests (across multiple Python versions).
    *   Integration tests (with service dependencies like PostgreSQL, Qdrant).
    *   Linting and static analysis.
    *   Security scans.
    *   API Hub specific tests.
    *   Vector search tests.
    *   Performance tests.
*   Builds and deployments are gated by the success of these test stages.
*   Test coverage reports (e.g., from Codecov) are generated and uploaded.

## 6. Test Coverage

*   **Target:** Aim for a high level of unit test coverage for critical business logic and utility functions.
*   **Measurement:** Code coverage is measured using `pytest-cov` and reported to Codecov.
*   **Review:** Coverage reports are reviewed as part of the code review process for pull requests. While 100% coverage is not always practical or meaningful, significant drops in coverage or lack of coverage for new critical features should be addressed.

## 7. Performance Testing Strategy

*   **Micro-benchmarks:** `pytest-benchmark` can be used for fine-grained performance testing of specific functions or code paths.
*   **System-Level Performance Tests:** `tests/test_performance.py` contains initial performance tests.
*   **Load Testing (Conceptual):** For comprehensive load testing, tools like k6 or Locust would be used to simulate realistic user loads against deployed environments. This includes:
    *   Defining key user scenarios and API endpoints to target.
    *   Establishing baseline performance metrics.
    *   Running tests regularly to detect performance regressions.
    *   Testing scalability by increasing load and observing system behavior (see `documentation/testing_phase1.md`).

## 8. Test Data Management

*   **Unit Tests:** Use mocked data or small, self-contained test data.
*   **Integration/E2E Tests:** May require pre-populated test databases or specific test files. Fixtures in `tests/conftest.py` help manage this setup. Test data should be version-controlled or generated if simple. Avoid dependencies on external, mutable data sources where possible for test stability.

## 9. Reporting and Analysis

*   **CI/CD:** GitHub Actions provides immediate feedback on test success/failure for each run.
*   **Coverage Reports:** Codecov provides detailed coverage reports and trend analysis.
*   **Manual Test Report Summary:** `docs/TEST_SUMMARY_REPORT.md` (to be updated periodically with overall status).
*   The `Scripts/run_tests.py` script can generate local JSON and Markdown test reports.

## 10. Future Enhancements

*   Increase test coverage, especially for newer modules.
*   Expand E2E test scenarios to cover more complex user workflows.
*   Implement more sophisticated load and stress testing using dedicated tools.
*   Integrate contract testing for microservice APIs (e.g., using Pact).
*   Automate testing for backup and restore procedures.
*   Visual regression testing if UI components are added.

This plan will be reviewed and updated periodically as the system evolves.
