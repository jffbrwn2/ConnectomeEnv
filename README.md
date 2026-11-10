# ConnectomeEnv

A visual-agent environment for connectome proofreading: inspect electron microscopy (EM), explore 3D neuron segments, and correct segmentation errors using model tool calls.

## Two evaluation modes

- **ConnectomeBench2 problems:** restore a recorded historical segmentation, let an agent inspect and edit it, and score the resulting state against hidden references.
- **Open-world reconstruction:** explore and edit a starting fragment, with scoring when reference data are supplied.

Both modes use the same inspection, navigation, editing, and undo interface. The runtime saves observations, actions, model responses, and scores for review.

## Try the environment

The tested platform is Linux x86-64 with Docker. This synthetic smoke test renders, edits, and scores both modes without credentials, network access, or model API charges:

```bash
git clone https://github.com/jffbrwn2/ConnectomeEnv.git
cd ConnectomeEnv
docker build -f deploy/portable/Dockerfile -t connectome-eval:0.1.0rc2 .
docker run --rm --network none --entrypoint python \
  connectome-eval:0.1.0rc2 -m proofreading.smoke --out /tmp/smoke
```

The synthetic test checks software behavior; it is not a biological benchmark result.

## Real-data benchmark

We have prepared and verified **1,000 validation examples: 500 split errors and 500 merge errors**, retaining the original 200-example benchmark. Download the frozen bundle from the [v0.1.0rc2 release](https://github.com/jffbrwn2/ConnectomeEnv/releases/tag/v0.1.0rc2). It is approximately 2 GB; allow additional space for extraction, cached imagery, and results.

Each example packages its starting location, historical source configuration, and frozen supervoxel scoring references. It does **not** package the full EM or spatial segmentation volume. The environment streams imagery, segmentation, and meshes as needed and caches them locally.

Real-data runs require internet access, authorized CAVE access to `minnie65_public`, and a vision/tool-capable model endpoint. Users supply their own credentials. Model API keys and CAVE tokens are never included in the repository or sent as part of model observations. The original data sources retain their own access conditions and terms.

## Download and check the benchmark

Run these commands from the cloned repository. They require Docker, Python 3,
`curl`, and `sha256sum`. Your user must have permission to run Docker.

```bash
mkdir -p downloads data results cache
curl -fL --retry 3 -o downloads/cb2-1000.tar.gz \
  https://github.com/jffbrwn2/ConnectomeEnv/releases/download/v0.1.0rc2/cb2-1000.tar.gz
curl -fL --retry 3 -o downloads/SHA256SUMS \
  https://github.com/jffbrwn2/ConnectomeEnv/releases/download/v0.1.0rc2/SHA256SUMS
(cd downloads && sha256sum -c SHA256SUMS)
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --entrypoint python -v "$PWD:/work" connectome-eval:0.1.0rc2 \
  -m proofreading.bundle unpack /work/downloads/cb2-1000.tar.gz --out /work/data/cb2-1000
python3 scripts/configure_benchmark.py data/cb2-1000 --out config.json
```

This selects **two problems**, one split error and one merge error. It preserves
the bundle's benchmark objective and budget. Repeat `--case CASE_ID` to select
specific problems. The configuration generator refuses to overwrite an existing
file. Keep the bundle unchanged; create configurations outside it.

For real data, obtain an authorized CAVE token from your data provider. Enter it
without putting its value in command history:

```bash
read -rsp 'CAVE token: ' CAVE_API_TOKEN; echo
export CAVE_API_TOKEN
export CONNECTOME_DATA="$PWD/data"
export CONNECTOME_OUTPUT="$PWD/results"
export CONNECTOME_CACHE="$PWD/cache"
export CONNECTOME_CONFIG="$PWD/config.json"
export CONNECTOME_IMAGE=connectome-eval:0.1.0rc2
bash deploy/portable/run-container.sh --out /results/check --preflight
```

A successful preflight loads both historical scenes, renders their initial views,
and checks frozen scoring references. It makes **zero model calls**. Results and
observations are saved under `results/check`. Use a new output directory when
repeating a check. Data and cache mounts are independent of the source checkout.

## Run your model

The default configuration uses a `preflight-only` placeholder model. Generate a
separate configuration with your provider and an actual vision/tool-capable model
ID before running inference:

```bash
python3 scripts/configure_benchmark.py data/cb2-1000 \
  --out model-config.json --provider responses --model YOUR_MODEL_ID
export CONNECTOME_CONFIG="$PWD/model-config.json"
# Set the matching model API key through your shell or secret manager.
bash deploy/portable/run-container.sh --out /results/run-001
```

Supported provider names are `responses`, `anthropic`, `gemini`, `openrouter`, and
`chat` (a compatible endpoint, supplied using `--endpoint`). They use
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, and
`MODEL_API_KEY`, respectively. Actual inference incurs your provider's charges.
The runner sends observations to the endpoint, applies tool actions, and records
and scores the trajectory. Generic endpoint compatibility does not guarantee
that every model can use the environment.

The Python package also exposes `connectome-eval`, `connectome-resume`,
`connectome-bundle`, `connectome-score`, and `connectome-results`. Docker includes
the package and its rendering dependencies, so no host Python environment is
needed for the commands above.

Compute and model hosting are independent. The environment can run on a Linux workstation or cloud VM; it does not require MIT Engaging, Slurm, or ChatGPT. CPU rendering is supported. A 4-core machine with 16–32 GiB RAM and persistent SSD is a starting point, not a guarantee for every scene. Self-hosted models require their own inference resources.

## Evaluation boundaries

Scoring uses a fixed local region and frozen references. Later segmentation is a timestamped reference, not a guarantee of fully correct human proofreading. Unknown reference material remains unverified. Wrong additions inside the scoring region are included in the volume accounting; common scoring scope must be aligned when comparing model cohorts. Evaluator files are withheld from hosted model conversations; arbitrary local agent code needs a separate sandbox that cannot access those files.

## Verified setup

On 2026-11-10, a fresh public clone, uncached dependency build, anonymous release download, full bundle verification, and real-data preflight on one split-error and one merge-error case passed on Linux x86-64. Both cases rendered and scored with an empty cache and a CAVE token, running as a non-root user. No hosted model calls were made. See the [validation record](docs/setup-validation.json).

## Development

This is a clean public source snapshot with fresh history. It excludes private experiment logs, credentials, model checkpoints, and development history. Run `bash deploy/portable/check-release.sh` to build and verify the container.

## License

The software is available under the [MIT License](LICENSE). This license does not relicense the source microscopy data or external model services.
