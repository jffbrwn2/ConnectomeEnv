#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. No credentials or network in verification runs.
image=${CONNECTOME_IMAGE:-connectome-eval:0.1.0rc2}
docker build -f deploy/portable/Dockerfile -t "$image" .
docker run --rm --network none --entrypoint python "$image" -m proofreading.runtime
docker run --rm --network none --entrypoint python "$image" -m proofreading.smoke --out /tmp/smoke
docker run --rm --network none --entrypoint python -v "$PWD:/repo:ro" -e PYTHONPATH=/repo \
  "$image" -m pytest /repo/tests/proofreading -q --disable-warnings
docker run --rm --network none --entrypoint pip "$image" check
