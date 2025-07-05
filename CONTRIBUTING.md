# Contributing to Wagehood

Thank you for your interest in contributing to Wagehood! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect different viewpoints and experiences

## How to Contribute

### Reporting Issues

1. Check existing issues to avoid duplicates
2. Use issue templates when available
3. Provide clear descriptions and steps to reproduce
4. Include relevant system information

### Submitting Pull Requests

1. **Fork the repository** and create a feature branch
2. **Write clear commit messages** following conventional commits
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run tests locally** before submitting
6. **Create a pull request** with a clear description

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/wagehood.git
cd wagehood

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Run tests
pytest

# Run validation
python scripts/run_comprehensive_validation.py
```

### Coding Standards

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Write docstrings for all public functions
- Keep functions focused and modular
- Use meaningful variable names

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting
- Include appropriate test coverage
- Test edge cases and error conditions

### Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update CLI help text as needed
- Include examples for new features

## Pull Request Process

1. **Branch Naming**: Use descriptive names like `feature/add-new-indicator` or `fix/calculation-bug`
2. **Commits**: Use clear, concise commit messages
3. **Testing**: Ensure all tests pass
4. **Documentation**: Update relevant docs
5. **Review**: Address reviewer feedback promptly

## Areas for Contribution

### High Priority
- Additional trading strategies
- Performance optimizations
- Test coverage improvements
- Documentation enhancements

### Feature Ideas
- New technical indicators
- Additional data providers
- Portfolio management features
- Additional backtesting features
- Machine learning integrations

### Good First Issues
Look for issues labeled "good first issue" for beginner-friendly tasks.

## Questions?

Feel free to open an issue for questions or discussions about potential contributions.