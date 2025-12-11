---
date created: 2025-12-11
date updated: 2025-12-11 14:20 UTC
---

# Contributing to Squadron

Thank you for your interest in contributing to Squadron! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- A GitHub account

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/squadron.git
   cd squadron
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

5. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

## Development Workflow

### Creating a Branch

Create a new branch for your feature or bugfix:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

### Making Changes

1. Write your code following our [Code Style](#code-style) guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Run the test suite to ensure everything passes

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=squadron

# Run specific test file
pytest tests/test_llm_providers.py

# Run with verbose output
pytest -v
```

### Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for linting issues
ruff check src/squadron

# Auto-fix issues
ruff check --fix src/squadron

# Format code
ruff format src/squadron
```

#### Style Guidelines

- **Type hints**: Required for all public functions and methods
- **Docstrings**: Use Google-style docstrings for all public APIs
- **Line length**: Maximum 100 characters
- **Imports**: Sorted with `isort` (handled by Ruff)

Example:

```python
async def create_agent(
    name: str,
    config: AgentConfig | None = None,
    memory: Memory | None = None,
) -> Agent:
    """
    Create a new agent instance.
    
    Args:
        name: Unique identifier for the agent
        config: Optional configuration settings
        memory: Optional memory backend
        
    Returns:
        A configured Agent instance
        
    Raises:
        ValueError: If name is empty
    """
    if not name:
        raise ValueError("Agent name cannot be empty")
    ...
```

### Type Checking

```bash
mypy src/squadron
```

### Committing Changes

Write clear, concise commit messages:

```bash
git commit -m "Add support for streaming responses in HuggingFace provider"
```

Follow these conventions:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Keep the first line under 72 characters
- Reference issues when applicable ("Fix #123: ...")

### Submitting a Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub

3. **Fill out the PR template** with:
   - Description of changes
   - Related issues
   - Testing performed
   - Screenshots (if applicable)

4. **Wait for review** - maintainers will review your PR and may request changes

## What to Contribute

### Good First Issues

Look for issues labeled `good first issue` - these are ideal for newcomers.

### Areas We Need Help

- **Documentation**: Improving docs, adding examples
- **Testing**: Increasing test coverage
- **Bug fixes**: Fixing reported issues
- **New providers**: Adding LLM provider integrations
- **Tool packs**: Creating new pre-built tool collections

### Feature Requests

Before implementing a large feature:
1. Check existing issues to avoid duplicates
2. Open an issue to discuss the feature
3. Wait for maintainer feedback before starting work

## Project Structure

```
squadron/
├── src/squadron/
│   ├── core/           # Agent runtime and orchestration
│   ├── memory/         # Memory backends
│   ├── reasoning/      # LATS and MCTS implementation
│   ├── connectivity/   # MCP and A2A protocols
│   ├── governance/     # Evaluation and safety
│   ├── evolution/      # Self-improvement
│   ├── llm/            # LLM provider abstraction
│   └── tools/          # Pre-built tool packs
├── tests/              # Test suites
├── examples/           # Working examples
└── docs/               # Documentation
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source structure (e.g., `src/squadron/llm/` → `tests/test_llm_providers.py`)
- Use pytest fixtures for common setup
- Mock external services (APIs, databases)

### Test Categories

```python
# Unit test
def test_message_creation():
    msg = LLMMessage.user("Hello")
    assert msg.role == MessageRole.USER

# Async test
@pytest.mark.asyncio
async def test_generate_response():
    llm = create_llm(model="gpt-4o")
    response = await llm.generate([LLMMessage.user("Hi")])
    assert response.content

# Integration test (mark appropriately)
@pytest.mark.integration
async def test_full_agent_run():
    ...
```

## Documentation

### Updating Documentation

- Keep README.md up to date with new features
- Add docstrings to all public APIs
- Update examples when APIs change
- Add type hints for better IDE support

### Building Documentation

```bash
# If using mkdocs
mkdocs serve
```

## Release Process

Releases are managed by maintainers. The process:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a release tag
4. GitHub Actions builds and publishes to PyPI

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email security@squadron.ai (do not open public issues)

## Recognition

Contributors are recognized in:
- The CONTRIBUTORS.md file
- Release notes
- The project README (for significant contributions)

Thank you for contributing to Squadron! 🚀
