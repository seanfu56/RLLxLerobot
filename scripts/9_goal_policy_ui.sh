#!/bin/bash
# Start the Streamlit goal-conditioned policy page on the Ubuntu PC.
#
# Unlike the plain policy page (scripts/9_policy_ui.sh) this one runs a video
# model first: it shows the current camera frame to Cosmos3-Nano or to the
# src/diffusion cascade, displays the predicted video, and lets the operator
# pick which of its frames the policy should aim at. The checkpoint must have
# been trained with `python src/policy/train.py --goal-conditioned`.
#
#   bash scripts/9_goal_policy_ui.sh
#   PORT=8504 bash scripts/9_goal_policy_ui.sh
#
# Then forward the port and open http://127.0.0.1:8504 on the laptop:
#   ssh -L 8504:127.0.0.1:8504 user@ubuntu-pc
#
# For the Cosmos3 source, start its server first, in its own environment:
#   python src/cosmos/serve.py --adapter models/cosmos3_nano_lora/pick-can-all/adapter
# The src/diffusion source needs no server; it loads in the hardware process.
#
# Never run this page and the teleoperation page against the same arm at the
# same time: each owns its own CAN connection and they would fight for it.

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/seanfu/miniconda3/envs/piper/bin/python}"
port="${PORT:-8504}"
address="${ADDRESS:-127.0.0.1}"
gpu="${CUDA_VISIBLE_DEVICES:-0}"

hf_home="${repo_root}/outputs/cache/huggingface"
python_path="${repo_root}/src:${repo_root}/lerobot/src:${repo_root}/plugins/lerobot_robot_piper/src"

if [[ ! -x "${python_bin}" ]]; then
    echo "Python environment not found: ${python_bin}" >&2
    exit 1
fi

if ! "${python_bin}" -c "import streamlit" >/dev/null 2>&1; then
    echo "Streamlit is not installed in ${python_bin}." >&2
    echo "  ${python_bin} -m pip install streamlit" >&2
    exit 1
fi

echo "Model root:   ${repo_root}/models/simple (goal-conditioned runs only)"
echo "Page:         src/ui/goal_policy_app.py on ${address}:${port}"
echo "Forward with: ssh -L ${port}:127.0.0.1:${port} \$USER@\$(hostname)"

cd "${repo_root}"

HF_HOME="${hf_home}" \
PYTHONPATH="${python_path}" \
CUDA_VISIBLE_DEVICES="${gpu}" \
"${python_bin}" -m streamlit run src/ui/goal_policy_app.py \
    --server.address="${address}" \
    --server.port="${port}" \
    --server.enableStaticServing=true \
    --server.headless=true
