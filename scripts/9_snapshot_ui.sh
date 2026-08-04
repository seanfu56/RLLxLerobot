#!/bin/bash
# Start the Streamlit camera-snapshot page on the Ubuntu PC.
#
# The page opens one USB camera, shows it live, and writes single frames to
# outputs/snapshots. It touches no arm: no CAN bus, no leader, no LeRobot
# plugins, so it is safe to run while nothing else is connected - and it is the
# quickest way to produce a conditioning frame for the video models or a goal
# image for src/ui/goal_policy_app.py.
#
#   bash scripts/9_snapshot_ui.sh
#   PORT=8505 bash scripts/9_snapshot_ui.sh
#   DEVICE=/dev/cam_overhead bash scripts/9_snapshot_ui.sh   # informational
#
# Then forward the port and open http://127.0.0.1:8505 on the laptop:
#   ssh -L 8505:127.0.0.1:8505 user@ubuntu-pc
#
# Only one page can hold a given camera at a time. Stop the camera here before
# starting the teleoperation page against the same device.

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/seanfu/miniconda3/envs/piper/bin/python}"
port="${PORT:-8505}"
address="${ADDRESS:-127.0.0.1}"

python_path="${repo_root}/src"

if [[ ! -x "${python_bin}" ]]; then
    echo "Python environment not found: ${python_bin}" >&2
    exit 1
fi

if ! "${python_bin}" -c "import streamlit, cv2" >/dev/null 2>&1; then
    echo "Streamlit and OpenCV are both required in ${python_bin}." >&2
    echo "  ${python_bin} -m pip install -r src/ui/requirements.txt" >&2
    exit 1
fi

echo "Snapshots:    ${repo_root}/outputs/snapshots"
echo "Cameras:      $(ls /dev/cam_* /dev/video* 2>/dev/null | tr '\n' ' ')"
echo "Page:         src/ui/snapshot_app.py on ${address}:${port}"
echo "Forward with: ssh -L ${port}:127.0.0.1:${port} \$USER@\$(hostname)"

cd "${repo_root}"

PYTHONPATH="${python_path}" \
"${python_bin}" -m streamlit run src/ui/snapshot_app.py \
    --server.address="${address}" \
    --server.port="${port}" \
    --server.enableStaticServing=true \
    --server.headless=true
