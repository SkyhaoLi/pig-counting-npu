#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ASCEND_CANDIDATES=(
  "/usr/local/Ascend/ascend-toolkit/set_env.sh"
  "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
)

echo "== Pig Counter board bootstrap =="
echo "Workdir: ${SCRIPT_DIR}"

for candidate in "${ASCEND_CANDIDATES[@]}"; do
  if [[ -f "${candidate}" ]]; then
    # shellcheck disable=SC1090
    source "${candidate}"
    echo "Loaded Ascend env: ${candidate}"
    break
  fi
done

if ! python3 -c "import acl" >/dev/null 2>&1; then
  echo "ERROR: python3 cannot import acl. Check CANN / pyACL installation." >&2
  exit 1
fi

if ! python3 -c "import cv2, numpy" >/dev/null 2>&1; then
  echo "ERROR: python3 cannot import cv2 or numpy." >&2
  echo "Install them first, for example:" >&2
  echo "  python3 -m pip install --user numpy opencv-python" >&2
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/models" "${SCRIPT_DIR}/datasets" "${SCRIPT_DIR}/output" "${SCRIPT_DIR}/logs"

if [[ ! -f "${SCRIPT_DIR}/models/yolov8n_pig_fp16.om" ]]; then
  echo "ERROR: model missing: ${SCRIPT_DIR}/models/yolov8n_pig_fp16.om" >&2
  exit 1
fi

python3 -m py_compile \
  "${SCRIPT_DIR}/npu_detector.py" \
  "${SCRIPT_DIR}/track_and_count_npu.py" \
  "${SCRIPT_DIR}/web_monitor.py" \
  "${SCRIPT_DIR}/batch_run_npu.py"

echo
echo "Bootstrap OK."
echo "Quick test:"
echo "  cd ${SCRIPT_DIR}"
echo "  python3 web_monitor.py --video datasets/group4/1-12头.mp4 --om models/yolov8n_pig_fp16.om"
