#!/usr/bin/env bash
# Tier 2: run all notebooks against a real Colab VM with a T4 GPU, via google-colab-cli.
# This is the pass/fail gate — it's the only tier that exercises the real GPU path,
# a full HAM10000 download + both trainings in notebook 3, and notebook 2's wandb sweep.
#
# Prerequisites (see PLAN-compute.md P1-P7):
#   - `uv tool install google-colab-cli --with "jupyter_kernel_client<1.0.0"`, OAuth completed
#     (`colab sessions` responds). NOTE: as of google-colab-cli 0.6.0, the plain `uv tool install
#     google-colab-cli` pulls jupyter_kernel_client 1.0.2, which renamed KernelClient ->
#     JupyterKernelClient and breaks `colab exec` with `AttributeError: module
#     'jupyter_kernel_client' has no attribute 'KernelClient'`. Pin jupyter_kernel_client<1.0.0
#     until upstream fixes this.
#   - Kaggle credentials locally, EITHER:
#       ~/.kaggle/access_token (newer `kaggle login` browser-OAuth flow), OR
#       ~/.kaggle/kaggle.json (classic username+key token), chmod 600
#     Both are recognized by the `kaggle` CLI/API on the remote VM the same way they are locally.
#   - ~/.netrc present locally (from `wandb login`), or WANDB_API_KEY exported
#   - HAM10000 dataset terms accepted on the Kaggle dataset page
#
# Verified behavior (smoke-tested against a live CPU session, 2026-09-14):
#   - `colab exec` does NOT stop on a cell error — it continues executing subsequent cells and
#     still writes a complete output notebook. So a notebook with a genuinely-failing cell won't
#     abort the whole run; we rely on the post-hoc error scan below to catch it.
#   - The executed notebook is written next to the SOURCE file (dirname of the -f path), not the
#     invocation's cwd, as "<name>_output.ipynb" — this matters for answers/1-neural-networks.ipynb,
#     whose output lands in answers/, not the repo root.
#   - Default --timeout is only 30s; real training cells need much more, set per-notebook below.
#   - `colab upload` to /root/.kaggle/<file> fails with a 500 Internal Server Error on a fresh VM
#     because /root/.kaggle/ doesn't exist yet (confirmed live against a real T4 session,
#     2026-09-14) — colab-cli has no raw remote-shell command, so the fix is exec'ing a one-cell
#     notebook that does `os.makedirs(...)` first, which is what the setup step below does.
set -euo pipefail

cd "$(dirname "$0")/.."

SESSION="idl"
OUTPUT_DIR="colab-test-output"
LOG_FILE="colab-run-log.md"

# notebook path -> exec timeout in seconds (generous; adjust after first real timing data)
declare -A NOTEBOOKS=(
  ["1-neural-networks.ipynb"]=300
  ["2-pytorch-lightning.ipynb"]=3600
  ["4-recurrent-neural-networks.ipynb"]=600
  ["answers/1-neural-networks.ipynb"]=300
  ["3-convolutional-neural-networks.ipynb"]=7200
)
# Run notebook 3 last: biggest download, longest training.
NOTEBOOK_ORDER=(
  "1-neural-networks.ipynb"
  "2-pytorch-lightning.ipynb"
  "4-recurrent-neural-networks.ipynb"
  "answers/1-neural-networks.ipynb"
  "3-convolutional-neural-networks.ipynb"
)

# Flattens a notebook path into a unique output filename, e.g.
# "answers/1-neural-networks.ipynb" -> "answers-1-neural-networks_output.ipynb".
# Using plain basename here would collide 1-neural-networks.ipynb with
# answers/1-neural-networks.ipynb (confirmed live 2026-09-14: the second overwrote the first).
out_name_for() { local flat; flat="$(echo "${1%.ipynb}" | tr '/' '-')"; echo "${flat}_output.ipynb"; }

mkdir -p "$OUTPUT_DIR"

echo "==> Starting Colab session '$SESSION' with a T4 GPU"
colab new -s "$SESSION" --gpu T4

echo "==> Ensuring /root/.kaggle exists on the VM (colab upload 500s if the parent dir is missing)"
MKDIR_NB="$(mktemp -t mkdir-kaggle-XXXXXX.ipynb)"
uv run --with nbformat python3 - "$MKDIR_NB" <<'PYEOF'
import sys
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_code_cell("import os; os.makedirs('/root/.kaggle', exist_ok=True)")]
nbf.write(nb, sys.argv[1])
PYEOF
colab exec -s "$SESSION" -f "$MKDIR_NB"
rm -f "$MKDIR_NB" "$(dirname "$MKDIR_NB")/$(basename "${MKDIR_NB%.ipynb}")_output.ipynb"

echo "==> Uploading credentials"
if [ -f "$HOME/.kaggle/access_token" ]; then
  colab upload -s "$SESSION" "$HOME/.kaggle/access_token" /root/.kaggle/access_token
elif [ -f "$HOME/.kaggle/kaggle.json" ]; then
  colab upload -s "$SESSION" "$HOME/.kaggle/kaggle.json" /root/.kaggle/kaggle.json
else
  echo "  WARNING: no ~/.kaggle/access_token or ~/.kaggle/kaggle.json found locally; notebook 3's download will fail."
fi
if [ -f "$HOME/.netrc" ]; then
  colab upload -s "$SESSION" "$HOME/.netrc" /root/.netrc
else
  echo "  WARNING: ~/.netrc not found locally; notebook 2's wandb.sweep() will fail."
fi

echo "==> Executing notebooks (notebook 3 runs last: biggest download, longest training)"
for nb in "${NOTEBOOK_ORDER[@]}"; do
  echo "  -- $nb (timeout ${NOTEBOOKS[$nb]}s)"
  colab exec -s "$SESSION" -f "$nb" --timeout "${NOTEBOOKS[$nb]}"
done

echo "==> Collecting session log"
colab log -s "$SESSION" -o "$LOG_FILE"

echo "==> Stopping session"
colab stop -s "$SESSION"

echo "==> Moving *_output.ipynb into ${OUTPUT_DIR}/"
for nb in "${NOTEBOOK_ORDER[@]}"; do
  # colab exec writes "<name>_output.ipynb" next to the SOURCE file, not the invocation cwd —
  # matters for answers/1-neural-networks.ipynb, whose output lands in answers/.
  src_produced="$(dirname "$nb")/$(basename "${nb%.ipynb}")_output.ipynb"
  out_name="$(out_name_for "$nb")"
  if [ -f "$src_produced" ]; then
    mv "$src_produced" "${OUTPUT_DIR}/${out_name}"
  fi
done

echo "==> Scanning outputs for errors"
FAILED=0
for nb in "${NOTEBOOK_ORDER[@]}"; do
  out_name="$(out_name_for "$nb")"
  out_path="${OUTPUT_DIR}/${out_name}"
  if [ ! -f "$out_path" ]; then
    echo "  [MISSING] $nb — no output file at ${out_path}"
    FAILED=1
    continue
  fi
  errors=$(uv run --with nbformat python - "$out_path" <<'PYEOF'
import sys
import nbformat

nb = nbformat.read(sys.argv[1], as_version=4)
for i, cell in enumerate(nb.cells):
    if cell.get("cell_type") != "code":
        continue
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            ename = output.get("ename", "?")
            evalue = output.get("evalue", "?")
            print(f"cell {i}: {ename}: {evalue}")
PYEOF
)
  if [ -n "$errors" ]; then
    echo "  [ERROR]  $nb"
    echo "$errors" | sed 's/^/           /'
    FAILED=1
  else
    echo "  [OK]     $nb"
  fi
done

if [ "$FAILED" -ne 0 ]; then
  echo "One or more notebooks had unexpected errors. See ${OUTPUT_DIR}/ and ${LOG_FILE}."
  exit 1
fi

echo "All notebooks executed cleanly on the real Colab T4 runtime."
