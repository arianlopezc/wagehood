# Project Memory - Wagehood

This file contains project-specific instructions and standards for Claude Code to follow when working on this codebase.

## Project Planning
- All project plans, design documents, and architectural decisions must be saved as markdown files in the `.local/` folder
- Create the `.local/` folder if it doesn't exist
- Use descriptive filenames for plan documents (e.g., `.local/feature-design.md`, `.local/refactoring-plan.md`)

## Python Coding Standards

### Code Style
- Use 4 spaces for indentation (no tabs)
- Follow PEP 8 style guide
- Maximum line length: 79 characters
- Use type hints for all function parameters and return values
- Use descriptive variable and function names with `lowercase_with_underscores`
- Class names should use `CapitalizedWords` (PascalCase)
- Constants should use `UPPERCASE_WITH_UNDERSCORES`

### Import Organization
- Group imports in this order: standard library, third-party, local imports
- Sort imports alphabetically within each group
- Use absolute imports when possible

### Documentation
- Write docstrings for all public modules, functions, classes, and methods
- Use Google-style docstrings format
- Include type information, parameters, return values, and exceptions in docstrings

## Testing Requirements
- Use `pytest` as the testing framework
- Place tests in a `tests/` directory mirroring the source structure
- Name test files with `test_` prefix
- Aim for minimum 80% code coverage
- Write tests for edge cases and error conditions
- Run tests with: `pytest`
- Run tests with coverage: `pytest --cov=src tests/`

## Code Quality Tools
- Format code with: `black .`
- Sort imports with: `isort .`
- Check code style with: `flake8 .`
- Type check with: `mypy .`
- Comprehensive linting with: `pylint src/`

## Security Practices
- Never hardcode passwords, API keys, or secrets in code
- Use environment variables for sensitive configuration
- Always validate user inputs
- Use parameterized queries for database operations

## Error Handling
- Use specific exception types when possible
- Always log errors with appropriate context
- Use try-except blocks for operations that may fail
- Include cleanup code in finally blocks when needed

## Virtual Environment
- Always work within a virtual environment
- Use `python -m venv venv` to create
- Activate with `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- Track dependencies in `requirements.txt`

## Project-Specific Commands
```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src tests/

# Format all code
black .

# Sort all imports
isort .

# Run type checking
mypy .

# Check code style
flake8 .

# Run comprehensive linting
pylint src/
```

## Development Workflow
1. Create feature branch from main
2. Write tests for new functionality
3. Implement the feature
4. Run all quality checks (tests, formatting, linting)
5. Ensure all checks pass before submitting PR
6. Update documentation as needed