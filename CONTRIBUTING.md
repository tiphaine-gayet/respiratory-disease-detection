# Contributing Guide

Thank you for contributing to this project!
Please follow these guidelines to keep the codebase clean, consistent, and secure.

## Branching Strategy

- Create a new branch for each feature or fix:

```bash
git checkout -b feature/your-feature-name
```

- Never push directly to `main`
- Use clear and descriptive branch names

## Pull Requests

- Open a Pull Request (PR) to merge into `main`
- Provide a clear description of:
- what you did
- why it was needed
- Keep PRs small and focused

## Code Quality & Security

This project uses automated checks on every push and pull request:

- **Ruff** → ensures code quality (syntax, structure, unused imports)
- **Bandit** → detects Python security issues
- **CodeQL** → performs advanced static analysis

!!! !!! All checks need to be passed in order to merge to main.

You can activate local pre-commit checks with :

```bash
pre-commit install
```

## Coding Guidelines

- Write clear and readable code
- Use meaningful variable and function names
- Avoid unused imports and dead code
- Keep functions small and modular

## Quick Tips

- Always fetch the latest changes before starting, and rebase your branch if necessary with :

```bash
git fetch origin
git rebase origin/main
```

- Test your code before pushing
- Fix lint/security issues if they are critical

## Communication

- Ask questions if something is unclear
- Keep commits and PRs understandable for the team

**Thanks for contributing!**
