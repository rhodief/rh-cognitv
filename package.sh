#!/bin/bash

# Script to build and optionally publish the package.
#
# Usage:
#   ./package.sh              # build only
#   ./package.sh --publish    # build + upload to PyPI
#   ./package.sh --test-pypi  # build + upload to Test PyPI
#   ./package.sh --help       # show help

set -e

# ── Help ──────────────────────────────────────
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    cat <<'EOF'
RH_COGNITV — Package Builder
=============================

USAGE
  ./package.sh              Build the package (sdist + wheel)
  ./package.sh --publish    Build and upload to PyPI
  ./package.sh --test-pypi  Build and upload to Test PyPI
  ./package.sh --help       Show this help message

BUILD
  Uses `python -m build` which reads pyproject.toml.
  Outputs:
    dist/rh_cognitv-<version>-py3-none-any.whl   (wheel)
    dist/rh_cognitv-<version>.tar.gz              (sdist)

PUBLISH TO PYPI
  1. Get an API token at https://pypi.org/manage/account/token/
  2. Set up credentials:
       cp .pypirc-template ~/.pypirc
       chmod 600 ~/.pypirc
     Then replace the placeholder tokens in ~/.pypirc.
     Note: ~/.pypirc lives in your HOME directory, NOT in the project.
  3. Run:
       ./package.sh --publish        # real PyPI
       ./package.sh --test-pypi      # test PyPI (try this first)

  Or manually:
       twine upload dist/*                          # PyPI
       twine upload --repository testpypi dist/*    # Test PyPI

INSTALL FROM PYPI (once published)
  pip install rh_cognitv

INSTALL FROM GITHUB (no PyPI needed)
  pip install git+https://github.com/rhodief/rh-cognitv.git           # latest main
  pip install git+https://github.com/rhodief/rh-cognitv.git@branch    # specific branch
  pip install git+https://github.com/rhodief/rh-cognitv.git@v0.1.0    # specific tag

INSTALL LOCALLY (editable / development)
  pip install -e .              # editable install from project root
  pip install -e ".[dev]"       # with dev extras (if defined)

EOF
    exit 0
fi

# ── Build ─────────────────────────────────────

echo "=========================================="
echo "  RH_COGNITV - Package Builder"
echo "=========================================="

# Clean previous builds
echo ""
echo "[1/4] Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Install build dependencies
echo ""
echo "[2/4] Checking build dependencies..."
pip install --quiet --upgrade pip build twine

# Build the package
echo ""
echo "[3/4] Building the package..."
python -m build

# Show results
echo ""
echo "[4/4] Build complete!"
echo ""
echo "Generated files in dist/:"
ls -la dist/

# Publish if requested
if [[ "$1" == "--publish" ]]; then
    echo ""
    echo "Publishing to PyPI..."
    twine upload dist/*
elif [[ "$1" == "--test-pypi" ]]; then
    echo ""
    echo "Publishing to Test PyPI..."
    twine upload --repository testpypi dist/*
fi

echo ""
echo "=========================================="
echo "  Done!"
echo "=========================================="
echo ""
echo "Run ./package.sh --help for publish & install instructions."
echo ""
