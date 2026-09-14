#!/usr/bin/env bash
# Tier 1: run all notebooks against the Colab CPU-runtime Docker image.
#
# This reproduces Colab's preinstalled package set locally, offline, for a fast
# regression loop (no Google account, no GPU). It does NOT catch GPU-path bugs or
# anything that needs real Colab Secrets / a browser — see scripts/test-colab-cli.sh
# (tier 2) and colab-mcp (tier 3) for those.
#
# Mechanism (verified 2026-09-14 against a real container run of notebook 4, which
# executed cleanly with zero errors in ~7 minutes):
#   The image's exposed port 8080 is NOT a plain Jupyter kernel gateway — it's a
#   Colab-proprietary "connect to local runtime" layer that expects a browser
#   handshake with an Origin matching colab.(sandbox|research).google.com (the same
#   mechanism colab-mcp drives). It rejects a direct jupyter_client/nbclient network
#   connection. The image DOES ship a full jupyter/nbconvert/nbclient install
#   internally, so instead of fighting that network layer, we bind-mount the repo
#   into the container and run `jupyter nbconvert --execute` via `docker exec`,
#   entirely inside the container's own Python environment. No network kernel-gateway
#   auth is needed this way.
#
# Expected skips on this tier (documented, not bugs):
#   - notebook 3 (3-convolutional-neural-networks.ipynb): stops at the HAM10000
#     download, since no Kaggle credentials are supplied to this container.
#   - notebook 2 (2-pytorch-lightning.ipynb): the wandb.sweep() cell requires a
#     logged-in W&B account; it will fail here unless WANDB_API_KEY/.netrc is
#     made available to the container before running this script.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"

IMAGE="europe-docker.pkg.dev/colab-images/public/cpu-runtime"
CONTAINER_NAME="idl-colab-cpu-runtime"
OUTPUT_DIR="colab-test-output"

# notebook path -> nbconvert ExecutePreprocessor timeout in seconds
declare -A NOTEBOOKS=(
  ["1-neural-networks.ipynb"]=300
  ["answers/1-neural-networks.ipynb"]=300
  ["2-pytorch-lightning.ipynb"]=1800
  ["3-convolutional-neural-networks.ipynb"]=1800
  ["4-recurrent-neural-networks.ipynb"]=600
)
NOTEBOOK_ORDER=(
  "1-neural-networks.ipynb"
  "answers/1-neural-networks.ipynb"
  "2-pytorch-lightning.ipynb"
  "3-convolutional-neural-networks.ipynb"
  "4-recurrent-neural-networks.ipynb"
)

mkdir -p "$OUTPUT_DIR"

echo "==> Pulling $IMAGE (multi-GB, first run only)"
docker pull "$IMAGE"

echo "==> Starting Colab CPU runtime container with repo bind-mounted read-only"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER_NAME" \
  -v "${REPO_DIR}:/content/repo:ro" \
  "$IMAGE" >/dev/null
docker exec "$CONTAINER_NAME" mkdir -p /tmp/out

if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "==> WANDB_API_KEY is set in the environment; will export it into the container for notebook 2"
fi

for nb in "${NOTEBOOK_ORDER[@]}"; do
  out_name="$(basename "${nb%.ipynb}")_output.ipynb"
  timeout="${NOTEBOOKS[$nb]}"
  echo "==> Executing $nb (timeout ${timeout}s) -> ${OUTPUT_DIR}/${out_name}"
  docker exec \
    ${WANDB_API_KEY:+-e "WANDB_API_KEY=${WANDB_API_KEY}"} \
    "$CONTAINER_NAME" \
    jupyter nbconvert --to notebook --execute \
      --output "/tmp/out/${out_name}" \
      --ExecutePreprocessor.timeout="$timeout" \
      --ExecutePreprocessor.allow_errors=True \
      "/content/repo/${nb}"
  docker cp "${CONTAINER_NAME}:/tmp/out/${out_name}" "${OUTPUT_DIR}/${out_name}"
done

echo "==> Stopping container"
docker rm -f "$CONTAINER_NAME" >/dev/null

echo "==> Scanning outputs for errors"
for nb in "${NOTEBOOK_ORDER[@]}"; do
  out_name="$(basename "${nb%.ipynb}")_output.ipynb"
  out_path="${OUTPUT_DIR}/${out_name}"
  errors=$(uv run --with nbformat python - "$out_path" <<'PYEOF'
import sys
import nbformat

nb = nbformat.read(sys.argv[1], as_version=4)
for i, cell in enumerate(nb.cells):
    if cell.get("cell_type") != "code":
        continue
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            print(f"cell {i}: {output.get('ename', '?')}: {output.get('evalue', '?')}")
PYEOF
)
  if [ -n "$errors" ]; then
    echo "  [ERROR]  $nb"
    echo "$errors" | sed 's/^/           /'
  else
    echo "  [OK]     $nb"
  fi
done

echo "Done. Reminder: notebook 3's dataset cell and notebook 2's sweep cell are"
echo "expected to error on this tier unless credentials were exported beforehand."
