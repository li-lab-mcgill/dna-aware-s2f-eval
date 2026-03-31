#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="cglm"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found in PATH." >&2
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Creating conda environment: ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" python=3.10 pip
else
    echo "Conda environment already exists: ${ENV_NAME}"
fi

echo "Upgrading pip in ${ENV_NAME}"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip

echo "Installing Python dependencies from requirements.txt"
conda run -n "${ENV_NAME}" python -m pip install -r "${REPO_ROOT}/requirements.txt"

echo "Environment setup is complete."
echo "Next step: conda activate ${ENV_NAME}"
