# Contributing to SPEED

Thank you for your interest in contributing to SPEED! This guide will help you get started.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AndersGMadsen/SPEED.git
   cd SPEED
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   pip install -r requirements_dev.txt
   pip install -e .
   ```

## Code Style

- We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting.
- Run `ruff check .` to check for issues.
- Run `ruff format .` to auto-format.

## Running Tests

```bash
pytest tests/ -v
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes and add tests if applicable.
3. Ensure all tests pass: `pytest tests/ -v`
4. Run the linter: `ruff check .`
5. Submit a pull request with a clear description of the changes.

## Configuration Files

When adding or modifying dataset configs in `configs/`, follow the existing YAML structure. See `configs/pretrain/example.yaml` for a complete reference with all parameters.

## Reporting Issues

Please report bugs and feature requests on the [GitHub Issues](https://github.com/AndersGMadsen/SPEED/issues) page.
