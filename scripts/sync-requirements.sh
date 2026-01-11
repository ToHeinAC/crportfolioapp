#!/bin/bash
# Sync pyproject.toml dependencies to requirements.txt for Streamlit Cloud deployment
# UV is the source of truth, requirements.txt is auto-generated

set -e

cd "$(dirname "$0")/.."

echo "Syncing dependencies from pyproject.toml to requirements.txt..."

# Export dependencies from uv, then clean up for Streamlit Cloud:
# - Remove editable install line (-e .)
# - Remove comment lines (starting with #)
# - Remove indented comment lines (e.g., "    # via streamlit")
# - Remove blank lines
# - Remove platform-specific markers that may cause issues
uv export --no-dev --no-hashes | \
    grep -v "^-e \." | \
    grep -v "^#" | \
    grep -v "^[[:space:]]*#" | \
    grep -v "^[[:space:]]*$" \
    > requirements.txt

echo "requirements.txt has been updated from pyproject.toml"
echo ""
echo "Package count: $(wc -l < requirements.txt | tr -d ' ')"
echo ""
echo "First 20 packages:"
head -20 requirements.txt
