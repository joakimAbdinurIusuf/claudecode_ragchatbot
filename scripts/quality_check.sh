#!/bin/bash

# Quality check script for the project
# Runs all code quality tools in sequence

set -e  # Exit on first error

echo "🔍 Running code quality checks..."
echo

# Change to project root
cd "$(dirname "$0")/.."

# Run black formatter check
echo "📝 Checking code formatting with black..."
uv run black --check --diff .
echo "✅ Black formatting check passed"
echo

# Run ruff linter
echo "🔧 Running ruff linter..."
uv run ruff check .
echo "✅ Ruff linting passed"
echo

# Run mypy type checker
echo "🔍 Running mypy type checker..."
uv run mypy backend/ main.py
echo "✅ Mypy type checking passed"
echo

echo "🎉 All quality checks passed!"