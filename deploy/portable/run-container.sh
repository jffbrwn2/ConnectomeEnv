#!/usr/bin/env bash
set -euo pipefail
# Required mounts are absolute paths. Secrets are passed by environment name only.
: "${CONNECTOME_DATA:?Set the local data directory}"
: "${CONNECTOME_OUTPUT:?Set the local output directory}"
: "${CONNECTOME_CONFIG:?Set the experiment JSON file}"
: "${CONNECTOME_CACHE:?Set the persistent mesh cache directory}"
runner=proofreading.experiment
if [[ "${CONNECTOME_RECOVER:-0}" == 1 ]]; then runner=proofreading.resilient; fi
exec docker run --rm --init --entrypoint python \
  --user "$(id -u):$(id -g)" \
  --memory "${CONNECTOME_MEMORY:-16g}" --cpus "${CONNECTOME_CPUS:-4}" \
  --mount "type=bind,src=$CONNECTOME_DATA,dst=/data,readonly" \
  --mount "type=bind,src=$CONNECTOME_OUTPUT,dst=/results" \
  --mount "type=bind,src=$CONNECTOME_CACHE,dst=/cache" \
  --mount "type=bind,src=$CONNECTOME_CONFIG,dst=/work/config.json,readonly" \
  -e OPENAI_API_KEY -e ANTHROPIC_API_KEY -e OPENROUTER_API_KEY -e MODEL_API_KEY -e GEMINI_API_KEY -e CAVE_API_TOKEN \
  -e CONNECTOME_WORLD_ROOT -e CONNECTOME_PATH_MAP -e CONNECTOME_MOONSHOT_BUDGET_USD -e MOONSHOT_API_KEY \
  -e CONNECTOME_CACHE_DIRECTORY=/cache/streaming \
  "${CONNECTOME_IMAGE:-connectome-eval:0.1.0rc2}" -m "$runner" /work/config.json "$@"
