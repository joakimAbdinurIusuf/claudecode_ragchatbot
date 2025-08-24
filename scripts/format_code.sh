#!/bin/bash

# Code formatting script
# Automatically formats all code with black and fixes ruff issues

set -e  # Exit on first error

echo "🔧 Formatting code..."
echo

# Change to project root
cd "$(dirname "$0")/.."

# Run black formatter
echo "📝 Formatting code with black..."
uv run black .
echo "✅ Black formatting completed"
echo

# Fix auto-fixable ruff issues
echo "🔧 Fixing auto-fixable issues with ruff..."
uv run ruff check --fix .
echo "✅ Ruff auto-fixes completed"
echo

echo "🎉 Code formatting completed!"