# Testing Guidelines

## Testing Requirements
- Use `pytest` as the testing framework
- Place tests in a `tests/` directory mirroring the source structure
- Name test files with `test_` prefix
- Aim for minimum 80% code coverage
- Write tests for edge cases and error conditions

## Test Organization
- Mirror source code structure in tests directory
- Group related tests in classes when appropriate
- Use descriptive test function names that explain what is being tested
- One assertion per test when possible

## Testing Best Practices
- Write tests before or alongside code (TDD/BDD)
- Each test should test one thing
- Tests should be independent and idempotent
- Use descriptive test names that explain what is being tested
- Mock external dependencies
- Test edge cases and error conditions

## Test Fixtures
- Use pytest fixtures for reusable test data
- Keep fixtures focused and minimal
- Document complex fixtures
- Prefer fixtures over setup/teardown methods

## Coverage Goals
- Minimum 80% code coverage
- 100% coverage for critical business logic
- Document why uncovered code is acceptable (if applicable)