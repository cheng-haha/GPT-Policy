#!/usr/bin/env bash
set -euo pipefail

# One entry point for preparing the policy host and launching a RoboDojo run.
# Isaac Sim dependencies are opt-in because they are large and GPU/driver
# specific. Pass --install-sim-deps on the simulator host when they are absent.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP_SCRIPT="${ROOT_DIR}/scripts/setup_robodojo.sh"
EVAL_SCRIPT="${ROOT_DIR}/scripts/eval_robodojo_8gpu.sh"
VENV_DIR="${GPT_POLICY_VENV:-${ROOT_DIR}/.venv}"
INSTALL_SIM_DEPS=0
SKIP_SETUP=0

usage() {
  cat <<'EOF'
Usage: scripts/run_robodojo_eval.sh [options] [smoke|benchmark] [RoboDojo options]

Prepare the local policy environment when needed, then launch the evaluation.

Options:
  --install-sim-deps  Also install Isaac Sim/IsaacLab/CuRobo dependencies.
  --skip-setup        Do not run setup; fail if required files are missing.
  -h, --help          Show this help.

Examples:
  scripts/run_robodojo_eval.sh smoke --only push_T --eval-num 1
  scripts/run_robodojo_eval.sh --install-sim-deps smoke --only push_T --eval-num 1
  scripts/run_robodojo_eval.sh smoke --icl-mode video+action --dimension memory --eval-num 1
EOF
}

eval_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-sim-deps) INSTALL_SIM_DEPS=1; shift ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) eval_args+=("$1"); shift ;;
  esac
done

if (( SKIP_SETUP == 0 )); then
  setup_args=()
  (( INSTALL_SIM_DEPS )) && setup_args+=(--install-sim-deps)
  # setup_robodojo.sh is idempotent and also regenerates the policy adapter
  # after a checkout or upstream patch changes.
  if (( INSTALL_SIM_DEPS )) || [[ ! -x "${VENV_DIR}/bin/python" \
      || ! -d "${ROOT_DIR}/third_party/RoboDojo/.git" \
      || ! -d "${ROOT_DIR}/third_party/XPolicyLab/.git" \
      || ! -x "${ROOT_DIR}/third_party/XPolicyLab/policy/GPT_Policy/setup_eval_env_client.sh" ]]; then
    bash "${SETUP_SCRIPT}" "${setup_args[@]}"
  fi
else
  [[ -x "${VENV_DIR}/bin/python" ]] || { echo "Missing ${VENV_DIR}; remove --skip-setup or run setup_robodojo.sh." >&2; exit 1; }
fi

source "${VENV_DIR}/bin/activate"
export PATH="${HOME}/.codex/packages/standalone/releases/0.155.1-x86_64-unknown-linux-musl/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
exec bash "${EVAL_SCRIPT}" "${eval_args[@]}"
