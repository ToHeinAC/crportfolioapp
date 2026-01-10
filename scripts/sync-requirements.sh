#!/bin/bash
# Sync pyproject.toml dependencies to requirements.txt for Streamlit Cloud deployment
# UV is the source of truth, requirements.txt is auto-generated

set -e

cd "$(dirname "$0")/.."

echo "Syncing dependencies from pyproject.toml to requirements.txt..."

# Export and filter out the editable install line (-e .) which Streamlit Cloud doesn't support
uv export --no-dev --no-hashes | grep -v "^-e \." > requirements.txt

echo "requirements.txt has been updated from pyproject.toml"
echo ""
echo "First 20 lines:"
head -20 requirements.txt
