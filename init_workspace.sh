#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="${REPO_ROOT}/workspace"

mkdir -p \
  "${WORKSPACE_DIR}/datasets" \
  "${WORKSPACE_DIR}/models" \
  "${WORKSPACE_DIR}/logs" \
  "${WORKSPACE_DIR}/plots"


echo "Workspace prepared under ${WORKSPACE_DIR}"
echo "Next step: Download paired datasets and checkpoints with python download_zenodo.py"
echo "Checked-in configs live under ${REPO_ROOT}/configs"
