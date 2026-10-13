#!/usr/bin/env bash
# Shared config access; safe to source from any working directory.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
config_value() { "${PYTHON}" "${PROJECT_DIR}/config.py" "$@"; }
