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

We have prepared and verified **1,000 validation examples: 500 split errors and 500 merge errors**, retaining the original 200-example benchmark. The downloadable data bundle is **not yet published in this repository**. The runtime can be installed and tested now; the prepared benchmark requires that separate bundle.

Each example packages its starting location, historical source configuration, and frozen supervoxel scoring references. It does **not** package the full EM or spatial segmentation volume. The environment streams imagery, segmentation, and meshes as needed and caches them locally.

Real-data runs require internet access, authorized CAVE access to `minnie65_public`, and a vision/tool-capable model endpoint. Users supply their own credentials. Model API keys and CAVE tokens are never included in the repository or sent as part of model observations. The original data sources retain their own access conditions and terms.

## Running a model

The Python package exposes `connectome-eval`, `connectome-resume`, `connectome-bundle`, `connectome-score`, and `connectome-results`. Given an installed package, a benchmark bundle, credentials, and a configuration:

```bash
connectome-eval config.json --out results/check --preflight
connectome-eval config.json --out results/run-001
```

Preflight validates references and renders the starting view without calling a model. A normal run sends observations to the configured endpoint, applies tool actions, and records and scores the trajectory. Use separate output directories for separate runs.

Compute and model hosting are independent. The environment can run on a Linux workstation or cloud VM; it does not require MIT Engaging, Slurm, or ChatGPT. CPU rendering is supported. A 4-core machine with 16–32 GiB RAM and persistent SSD is a starting point, not a guarantee for every scene. Self-hosted models require their own inference resources.

## Evaluation boundaries

Scoring uses a fixed local region and frozen references. Later segmentation is a timestamped reference, not a guarantee of fully correct human proofreading. Unknown reference material remains unverified. Wrong additions inside the scoring region are included in the volume accounting; common scoring scope must be aligned when comparing model cohorts. Evaluator files are withheld from hosted model conversations; arbitrary local agent code needs a separate sandbox that cannot access those files.

## Development

This is a clean public source snapshot with fresh history. It excludes private experiment logs, credentials, model checkpoints, and development history. Run `bash deploy/portable/check-release.sh` to build and verify the container.

## License

The software is available under the [MIT License](LICENSE). This license does not relicense the source microscopy data or external model services.
