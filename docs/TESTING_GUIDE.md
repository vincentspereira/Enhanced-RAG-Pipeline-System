# Testing Guide

This guide provides instructions for running tests on the Qdrant RAG system, including the API Hub, Vector Search, and other components.

## Prerequisites

- Python 3.10 or higher
- Git
- Access to project repository

## Setting Up the Test Environment

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Qdrant
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv test_venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```powershell
     .\test_venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source test_venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   
   # Install additional test dependencies
   pip install pytest pytest-asyncio pytest-mock httpx requests PyJWT pyyaml slowapi
   ```

## Test Structure

The project's tests are organized into the following categories:

- **Unit Tests**: Located in `tests/unit/` - Test individual components in isolation
- **Integration Tests**: Located in `tests/integration/` - Test interactions between components
- **End-to-End Tests**: Located in `tests/e2e/` - Test complete workflows
- **Performance Tests**: Located in `tests/` (test_performance.py) - Test system performance under load
- **Vector Search Tests**: Located in `tests/` (test_vector_search.py) - Test vector database functionality

## Running Tests

### Running All Tests

To run all tests:

```bash
python -m pytest
```

### Running Specific Test Categories

#### Unit Tests

```bash
python -m pytest tests/unit/
```

#### API Hub Tests

```bash
python -m pytest tests/simplified_api_hub_test.py
```

#### Integration Tests

```bash
python -m pytest tests/integration/
```

#### End-to-End Tests

```bash
python -m pytest tests/e2e/
```

#### Vector Search Tests

```bash
python -m pytest tests/test_vector_search.py
```

#### Performance Tests

```bash
python -m pytest tests/test_performance.py
```

### Running Tests with Verbosity

For more detailed test output:

```bash
python -m pytest -v tests/unit/
```

### Running Tests with Coverage

To run tests with coverage reporting:

```bash
# Install coverage dependencies
pip install pytest-cov

# Run tests with coverage
python -m pytest --cov=Scripts tests/
```

To generate an HTML coverage report:

```bash
python -m pytest --cov=Scripts --cov-report=html tests/
```

The HTML report will be available in the `htmlcov` directory.

## Writing New Tests

### Unit Tests

When writing new unit tests, follow these guidelines:

1. Create test files in the `tests/unit/` directory
2. Name test files with the prefix `test_`
3. Use pytest fixtures for setup and teardown
4. Use mocking to isolate the component being tested
5. Write descriptive test names starting with `test_`

Example:

```python
import pytest
from unittest.mock import MagicMock, patch

def test_some_function():
    # Arrange
    # Act
    # Assert
    assert result == expected_value
```

### Integration Tests

For integration tests:

1. Create test files in the `tests/integration/` directory
2. Use pytest fixtures to set up test environments
3. Test interactions between real components
4. Clean up resources after tests

### End-to-End Tests

For end-to-end tests:

1. Create test files in the `tests/e2e/` directory
2. Simulate real user workflows
3. Test the system as a whole
4. Use appropriate cleanup to avoid test interference

## Test Fixtures

Common test fixtures are defined in `tests/conftest.py`. These include:

- `test_config`: Test configuration
- `db_engine`: Database engine fixture
- `db_session`: Database session fixture
- `qdrant_client`: Qdrant client fixture
- `temp_storage`: Temporary storage fixture
- `mock_openai`: Mocked OpenAI client
- `mock_s3`: Mocked S3 client

You can use these fixtures in your tests by adding them as parameters to your test functions:

```python
def test_something(db_session, qdrant_client):
    # Your test using the fixtures
    pass
```

## Troubleshooting Common Test Issues

### Missing Dependencies

If you encounter missing dependencies, install them with:

```bash
pip install <dependency-name>
```

### Database Connection Issues

For database connection issues:

1. Check that the test database URL is correctly configured
2. Ensure the database server is running
3. Verify credentials if applicable

### Qdrant Connection Issues

For Qdrant connection issues:

1. Make sure Qdrant is running
2. Check the connection URL in the test configuration
3. Verify that the test collection can be created

### Slow Tests

If tests are running slowly:

1. Use pytest's `-xvs` flags to see which tests are slow
2. Consider using mocks for external services
3. Use pytest's parallelization with `pytest-xdist`:
   ```bash
   pip install pytest-xdist
   python -m pytest -n auto
   ```

## Continuous Integration

The project uses GitHub Actions for continuous integration. The CI workflow:

1. Runs on every push and pull request
2. Sets up the test environment
3. Runs all tests
4. Generates and publishes coverage reports
5. Fails if tests fail or coverage is below threshold

## Test Reports

Test reports can be found in:

- Latest test runs: GitHub Actions
- Test summary: `docs/TEST_SUMMARY_REPORT.md`
- Coverage reports: CI artifacts or local `htmlcov` directory

## Contact

For questions about testing or to report test failures, contact:

- Project Maintainer: [maintainer@example.com](mailto:maintainer@example.com)
- Testing Team: [testing@example.com](mailto:testing@example.com)
