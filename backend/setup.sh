#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if ! command -v python >/dev/null 2>&1; then
  echo "python is required on PATH" >&2
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
python -m pip install -e "${SCRIPT_DIR}"
