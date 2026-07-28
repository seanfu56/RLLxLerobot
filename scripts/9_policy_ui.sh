#!/bin/bash
# Start the Streamlit policy-inference page on the Ubuntu PC.
#
# The page picks a checkpoint under models/sweeps, loads it into a dedicated
# hardware process, and runs timed closed-loop rollouts on the Piper. It is the
# inference counterpart of the teleoperation page (src/ui/app.py) and uses the
# same Piper follower settings as scripts/9_eval_policy_lab.sh.
#
#   bash scripts/9_policy_ui.sh
#   PORT=8503 bash scripts/9_policy_ui.sh
#
# Then forward the port and open http://127.0.0.1:8503 on the laptop:
#   ssh -L 8503:127.0.0.1:8503 user@ubuntu-pc
#
# Never run this page and the teleoperation page against the same arm at the
# same time: each owns its own CAN connection and they would fight for it.

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/seanfu/miniconda3/envs/piper/bin/python}"
port="${PORT:-8503}"
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

echo "Model root:   ${repo_root}/models/sweeps"
echo "Page:         src/ui/policy_app.py on ${address}:${port}"
echo "Forward with: ssh -L ${port}:127.0.0.1:${port} \$USER@\$(hostname)"

cd "${repo_root}"

HF_HOME="${hf_home}" \
PYTHONPATH="${python_path}" \
CUDA_VISIBLE_DEVICES="${gpu}" \
"${python_bin}" -m streamlit run src/ui/policy_app.py \
    --server.address="${address}" \
    --server.port="${port}" \
    --server.enableStaticServing=true \
    --server.headless=true
