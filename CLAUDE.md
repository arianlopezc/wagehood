# Project Memory - Wagehood

This file contains project-specific instructions and standards for Claude Code to follow when working on this codebase.

## Core Standards
@standards/python-standards.md
@standards/testing-guidelines.md

## Project Planning
- All project plans, design documents, and architectural decisions must be saved as markdown files in the `.local/` folder
- Create the `.local/` folder if it doesn't exist
- Use descriptive filenames for plan documents (e.g., `.local/feature-design.md`, `.local/refactoring-plan.md`)

## Documentation Guidelines
- **Private Documentation**: Development notes, analysis, and internal design documents go in `.local/` folder (gitignored)
- **Public Documentation**: User-facing documentation, API guides, and publicly linked docs go in `docs/` folder (version controlled)
- **Rule**: Any documentation referenced in public links (Discord channels, README, etc.) MUST be in `docs/` folder
- **Examples**: 
  - Discord integration guides → `docs/discord-integration-complete.md`
  - Configuration guides → `docs/CONFIGURATION_GUIDE.md`
  - API documentation → `docs/api-reference.md`
- **Migration**: When documentation graduates from private to public, move from `.local/` to `docs/` and update all links

## All Project Commands

### Testing Commands
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/[test_file.py]

# Run tests with coverage report
pytest --cov=src tests/

# Run tests in verbose mode
pytest -v

# Run tests matching pattern
pytest -k "pattern"
```

### Code Quality Commands
```bash
# Format all code
black .

# Check formatting without changes
black . --check

# Sort all imports
isort .

# Check import sorting without changes
isort . --check-only

# Check code style
flake8 .

# Run type checking
mypy .

# Run comprehensive linting
pylint src/
```

### Git Review Commands (Read-Only)
```bash
# Show current changes
git status

# Review uncommitted changes
git diff

# Review staged changes
git diff --staged

# View recent commits
git log --oneline -10

# Show changes in last commit
git show HEAD
```

### Virtual Environment Commands
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Linux/Mac)
source venv/bin/activate

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Update requirements file
pip freeze > requirements.txt
```

## Git Policy
- **NEVER commit without explicit user approval**
- **NEVER push to remote repository unless specifically requested**
- Use `git status` and `git diff` to help user review changes
- All commits must be reviewed and approved by the user
- When user requests commits, provide clear, descriptive commit messages

## Development Workflow
1. Create feature branch from main
2. Write tests for new functionality
3. Implement the feature
4. Run all quality checks (tests, formatting, linting)
5. Ensure all checks pass before submitting PR
6. Update documentation as needed

## Problem-Solving Methodology (ADOEOR Cycle)
When tackling any complex problem, follow the ADOEOR cycle:

1. **ANALYZE** - Before writing code:
   - Understand the current state, desired state, and constraints
   - Identify root causes and blockers
   - Save analysis in `.local/problem-analysis-[feature].md`

2. **DECOMPOSE** - Break down complex tasks:
   - Separate by domains (database, auth, API, frontend)
   - Identify dependencies between components
   - Create task breakdown in `.local/task-decomposition-[feature].md`

3. **ORCHESTRATE** - Plan execution strategy:
   - Choose execution pattern: sequential (for dependencies) or parallel (for independent tasks)
   - Define validation checkpoints between major steps
   - Document strategy in `.local/orchestration-plan-[feature].md`

4. **EXECUTE** - Implement with validation:
   - Complete one component fully before moving to the next
   - Run tests after each component completion
   - Track changes but DO NOT commit without user approval

5. **OPTIMIZE** - Measure and improve:
   - Track execution time and test results
   - Document performance metrics in `.local/performance-metrics.md`
   - Refactor based on bottlenecks identified

6. **RECURSE** - Learn and document:
   - Update `.local/lessons-learned.md` with insights
   - Store successful patterns for future use
   - Add new commands or workflows to CLAUDE.md

## Task Specialization Guidelines
Assign tasks based on domain expertise:
- **Database tasks**: Schema changes, migrations, query optimization
- **Auth tasks**: Authentication, authorization, security implementations
- **API tasks**: Endpoint creation, validation, routing
- **Testing tasks**: Test creation, coverage improvement, E2E tests
- **Infrastructure tasks**: Caching, rate limiting, performance optimization

## Validation Protocol
Implement multi-layer validation:
1. **Self-validation**: After each change, run relevant tests
2. **Cross-validation**: Review changes with `git diff` (for user review)
3. **System validation**: Run full test suite before marking task complete
4. **Performance validation**: Check that changes don't degrade performance

## Performance Tracking
Track and document metrics:
- Execution time for each major operation
- Test coverage percentage changes
- Number of tests added/modified
- Code quality metrics (linting warnings, type errors)
- Save metrics in `.local/metrics/[date]-[feature].md`

## Failure Recovery Protocol
When encountering failures:
1. Document the failure in `.local/failures/[date]-[issue].md`
2. Analyze root cause before attempting fixes
3. Create minimal reproduction case
4. Fix with targeted approach
5. Add tests to prevent regression
6. Update documentation with solution

## Knowledge Persistence
After completing any significant task:
- Document successful patterns in `.local/patterns/[pattern-name].md`
- Update this CLAUDE.md with new insights or commands
- Create reusable code snippets in `.local/snippets/`
- Maintain a `.local/architecture-decisions.md` log

## Continuous Improvement
- Review `.local/` folder weekly for optimization opportunities
- Consolidate repeated patterns into reusable workflows
- Update test scenarios based on discovered edge cases
- Refine task decomposition strategies based on outcomes

## Documentation Maintenance Protocol

### Documentation Update Requirements
After ANY code changes, implementation, or feature additions, documentation MUST be updated:

1. **Main README.md Updates:**
   - Update feature descriptions for any new functionality
   - Add new CLI commands and examples
   - Update performance metrics and strategy information
   - Ensure Quick Start guide reflects current system capabilities
   - Update architecture diagrams if system components change

2. **CLI Documentation Updates:**
   - Update CLI_DOCUMENTATION.md for any new commands
   - Add examples for new command options or flags
   - Update troubleshooting section for new error scenarios
   - Ensure all command help text is accurate and comprehensive

3. **Technical Documentation:**
   - Create API documentation for new classes and methods
   - Update configuration guides for new parameters
   - Document performance characteristics of new features
   - Update deployment instructions if infrastructure changes

4. **Code Documentation:**
   - Update docstrings for modified functions and classes
   - Add inline comments for complex logic
   - Update type hints and parameter descriptions
   - Ensure strategy explanations match implementation

### Documentation Validation Process
Before marking any task as complete:

1. **Accuracy Check:**
   - Verify all examples work as documented
   - Test all CLI commands and options
   - Confirm all configuration options are valid
   - Validate performance claims with actual data

2. **Completeness Check:**
   - Ensure all new features are documented
   - Check that examples cover common use cases
   - Verify troubleshooting covers likely issues
   - Confirm best practices are up-to-date

3. **Consistency Check:**
   - Ensure terminology is consistent across all docs
   - Check that formatting follows project standards
   - Verify cross-references between documents are accurate
   - Ensure version numbers and dates are current

### CLI Update Protocol
After every implementation or change that affects CLI:

1. **Command Updates:**
   - Update help text for any modified commands
   - Add new commands to CLI_DOCUMENTATION.md
   - Update command examples in README.md
   - Test all documented examples

2. **Error Handling:**
   - Update error messages to be user-friendly
   - Add troubleshooting entries for new error scenarios
   - Ensure error messages guide users to solutions
   - Document common failure modes and fixes

3. **Performance Documentation:**
   - Document performance characteristics of new features
   - Update capacity and scaling guidelines
   - Add monitoring and optimization recommendations
   - Include resource usage information

### Documentation File Structure
Maintain these documentation files:

- **README.md:** Main project documentation with overview, quick start, and examples
- **CLI_DOCUMENTATION.md:** Comprehensive CLI reference with all commands and examples
- **CLAUDE.md:** This file - project-specific instructions and standards
- **CONFIGURATION_GUIDE.md:** Detailed configuration reference (create when needed)
- **.local/:** Technical design documents and architectural decisions

### Documentation Language Standards
- **Avoid pretentious or marketing language** that makes unproven claims
- **Do not use words like**: "professional-grade", "enterprise-level", "industry-leading", "cutting-edge", "world-class", "revolutionary", "state-of-the-art"
- **Use factual, measured language**: "comprehensive", "multi-strategy", "configurable", "efficient", "functional"
- **Focus on capabilities, not subjective quality claims**
- **Be precise about what the system actually does vs. aspirational goals**
- **Use technical accuracy over promotional language**
- **State facts, not opinions about quality or superiority**

### Documentation Automation
- Use consistent formatting and markdown standards
- Include practical examples for all features
- Maintain a changelog of major documentation updates
- Keep troubleshooting sections current with user feedback
- Ensure all claims can be verified or measured

## User Review Checkpoints
Always pause for user review at these points:
- Before making any commits
- After completing major components
- Before running any commands that modify project structure
- When about to make breaking changes
- Before pushing anything to remote repositories
- **After any documentation updates** (ensure accuracy and completeness)
- After implementing new CLI features or API changes