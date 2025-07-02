# Contributing to Enhanced RAG Pipeline

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process
We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes (see "Running Tests" section below).
5. Make sure your code lints.
6. Issue that pull request!

## Running Tests

This project uses `pytest` for automated testing. Tests are located in the `tests/` directory, categorized into `unit/` and `integration/`.

To run all tests locally:
1. Ensure you have installed all dependencies, including testing dependencies, from `requirements.txt` (ideally in a Python virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
2. Navigate to the root of the repository.
3. Run pytest:
   ```bash
   pytest
   ```
   Or, for more verbose output:
   ```bash
   pytest -v
   ```
To run specific tests:
   ```bash
   pytest tests/unit/test_your_module.py
   pytest tests/integration/test_your_service.py
   ```

Integration tests might require certain services (like Qdrant, Redis, or the application services themselves) to be running. Refer to the specific test files or `DEPLOY_LOCAL_TEST_GUIDE.md` for details on their prerequisites. Unit tests are designed to run in isolation, often using mocks.

## Pull Request Process
1. Update the README.md with details of changes to the interface
2. Update the requirements.txt with any new dependencies
3. Update the documentation with any new configuration or API changes
4. The PR will be merged once you have the sign-off of maintainers

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project.

## Report bugs using GitHub's [issue tracker]
We use GitHub issues to track public bugs. Report a bug by [opening a new issue]().

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License
By contributing, you agree that your contributions will be licensed under its MIT License.
