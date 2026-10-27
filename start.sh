#!/usr/bin/env sh
# Start Swish (installs dependencies on the first run only).
cd "$(dirname "$0")" && exec python3 start.py "$@"
