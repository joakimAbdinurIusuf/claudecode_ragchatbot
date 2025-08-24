#!/bin/bash

# Pre-commit quality check script
# Runs formatting and all quality checks before committing

set -e  # Exit on first error

echo "🚀 Running pre-commit checks..."
echo

# Change to project root
cd "$(dirname "$0")/.."

# Format code first
echo "📝 Auto-formatting code..."
uv run black .
uv run ruff check --fix .
echo "✅ Code formatted"
echo

# Run quality checks
echo "🔍 Running quality checks..."
uv run black --check --diff .
uv run ruff check .
uv run mypy backend/ main.py
echo "✅ All checks passed"
echo

# Run tests
echo "🧪 Running tests..."
cd backend
uv run python tests/run_tests.py
cd ..
echo "✅ Tests passed"
echo

echo "🎉 Pre-commit checks completed successfully!"
echo "Ready to commit! 🚀"