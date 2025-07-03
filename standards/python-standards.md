# Python Coding Standards

## Code Style
- Use 4 spaces for indentation (no tabs)
- Follow PEP 8 style guide
- Maximum line length: 79 characters
- Use type hints for all function parameters and return values
- Use descriptive variable and function names with `lowercase_with_underscores`
- Class names should use `CapitalizedWords` (PascalCase)
- Constants should use `UPPERCASE_WITH_UNDERSCORES`

## Import Organization
- Group imports in this order: standard library, third-party, local imports
- Sort imports alphabetically within each group
- Use absolute imports when possible

## Documentation
- Write docstrings for all public modules, functions, classes, and methods
- Use Google-style docstrings format
- Include type information, parameters, return values, and exceptions in docstrings

## Error Handling
- Use specific exception types when possible
- Always log errors with appropriate context
- Use try-except blocks for operations that may fail
- Include cleanup code in finally blocks when needed

## Security Practices
- Never hardcode passwords, API keys, or secrets in code
- Use environment variables for sensitive configuration
- Always validate user inputs
- Use parameterized queries for database operations

## Virtual Environment
- Always work within a virtual environment
- Use `python -m venv venv` to create
- Activate with `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- Track dependencies in `requirements.txt`