#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBODOJO_DIR="${ROOT_DIR}/third_party/RoboDojo"
POLICY_DIR="${ROOT_DIR}/third_party/XPolicyLab/policy/GPT_Policy"
GPU_IDS="${ROBODOJO_GPU_IDS:-0,1,2,3,4,5,6,7}"
SIM_ENV="${ROBODOJO_SIM_ENV:-RoboDojo}"
MODE="${1:-smoke}"
ICL_MODE="${ROBODOJO_ICL_MODE:-video+action}"

if [[ "${MODE}" == "smoke" || "${MODE}" == "benchmark" ]]; then
  if [[ $# -gt 0 ]]; then
    shift
  fi
else
  echo "Usage: $0 [smoke|benchmark] [RoboDojo sweep options]" >&2
  exit 2
fi

if [[ "${1:-}" == "--icl-mode" ]]; then
  [[ $# -ge 2 ]] || { echo "--icl-mode requires video+action, video, or none" >&2; exit 2; }
  ICL_MODE="$2"
  shift 2
fi
case "${ICL_MODE}" in
  video+action|video|none) ;;
  *) echo "Invalid ICL mode: ${ICL_MODE}" >&2; exit 2 ;;
esac

# RoboDojo requires at least one selected task per worker. Trim the default
# eight-GPU list for an explicit small --only selection; this keeps one-task
# smoke tests convenient while preserving full parallelism for larger sweeps.
selected_count=""
for ((arg_index = 1; arg_index <= $#; arg_index++)); do
  if [[ "${!arg_index}" == "--only" ]]; then
    next_index=$((arg_index + 1))
    only_value="${!next_index:-}"
    IFS=',' read -r -a selected_tasks <<< "${only_value}"
    selected_count="${#selected_tasks[@]}"
    break
  fi
done
if [[ -n "${selected_count}" ]]; then
  IFS=',' read -r -a gpu_list <<< "${GPU_IDS}"
  if (( selected_count > 0 && selected_count < ${#gpu_list[@]} )); then
    GPU_IDS="$(IFS=,; echo "${gpu_list[*]:0:selected_count}")"
  fi
fi

[[ -x "${ROOT_DIR}/.venv/bin/python" ]] || {
  echo "Missing ${ROOT_DIR}/.venv; run ./scripts/setup_robodojo.sh first." >&2
  exit 1
}
[[ -x "${HOME}/miniconda3/envs/${SIM_ENV}/bin/python" ]] || {
  echo "Missing RoboDojo simulator environment: ${SIM_ENV}" >&2
  echo "Run ./scripts/setup_robodojo.sh --install-sim-deps first." >&2
  exit 1
}
[[ -x "${POLICY_DIR}/setup_eval_env_client.sh" ]] || {
  echo "Missing generated GPT-Policy adapter; run ./scripts/install_robodojo_policy.sh." >&2
  exit 1
}

export PATH="${HOME}/miniconda3/bin:${HOME}/miniconda3/condabin:${PATH}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export ROBODOJO_ICL_MODE="${ICL_MODE}"

# RoboDojo's embedded runtime weights are keyed by arx_x5. The generated
# simulator client maps this to the calibration-complete gpt_policy_x5
# overlay before launching Isaac Sim.
exec bash "${ROBODOJO_DIR}/scripts/robodojo.sh" "${MODE}" \
  --policy-dir "${POLICY_DIR}" \
  --ckpt none \
  --policy-env "${ROOT_DIR}/.venv" \
  --eval-env "${SIM_ENV}" \
  --env-cfg arx_x5 \
  --gpu-ids "${GPU_IDS}" \
  "$@"
