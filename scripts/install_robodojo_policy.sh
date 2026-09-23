#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XPOLICYLAB_DIR="${ROOT_DIR}/third_party/XPolicyLab"
GPT_CONFIG="${ROOT_DIR}/configs/examples/robodojo.json"
CALIBRATION="${ROOT_DIR}/configs/examples/robodojo_calibration.json"
POLICY_NAME="GPT_Policy"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --xpolicylab-dir) XPOLICYLAB_DIR="$2"; shift 2 ;;
    --config) GPT_CONFIG="$2"; shift 2 ;;
    --calibration) CALIBRATION="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--xpolicylab-dir DIR] [--config FILE] [--calibration FILE]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
XPOLICYLAB_DIR="$(realpath "$XPOLICYLAB_DIR")"
GPT_CONFIG="$(realpath "$GPT_CONFIG")"
CALIBRATION="$(realpath "$CALIBRATION")"
ROBODOJO_DIR="${XPOLICYLAB_DIR}/../RoboDojo"
CAMERA_PATCH="${ROOT_DIR}/scripts/patches/robodojo-camera-calibration.patch"
ACTION_PATCH="${ROOT_DIR}/scripts/patches/robodojo-action-execution.patch"
LAYOUT_PATCH="${ROOT_DIR}/scripts/patches/robodojo-layout-selection.patch"
TASK_PATCH="${ROOT_DIR}/scripts/patches/robodojo-task-source-alignment.patch"
REQUEST_PATCH="${ROOT_DIR}/scripts/patches/xpolicylab-request-wait.patch"
if git -C "$ROBODOJO_DIR" apply --reverse --check "$CAMERA_PATCH" 2>/dev/null; then
  printf '%s\n' 'RoboDojo camera calibration patch is already applied.'
elif git -C "$ROBODOJO_DIR" apply --check "$CAMERA_PATCH"; then
  git -C "$ROBODOJO_DIR" apply "$CAMERA_PATCH"
else
  echo 'RoboDojo camera source differs from the pinned version; review the calibration patch before installing.' >&2
  exit 1
fi
if git -C "$ROBODOJO_DIR" apply --reverse --check --unidiff-zero "$ACTION_PATCH" 2>/dev/null; then
  printf '%s\n' 'RoboDojo action execution patch is already applied.'
elif git -C "$ROBODOJO_DIR" apply --check --unidiff-zero "$ACTION_PATCH"; then
  git -C "$ROBODOJO_DIR" apply --unidiff-zero "$ACTION_PATCH"
else
  echo 'RoboDojo evaluator source differs from the pinned version; review the action execution patch before installing.' >&2
  exit 1
fi
if git -C "$ROBODOJO_DIR" apply --reverse --check --unidiff-zero "$LAYOUT_PATCH" 2>/dev/null; then
  printf '%s\n' 'RoboDojo single-layout selection patch is already applied.'
elif git -C "$ROBODOJO_DIR" apply --check --unidiff-zero "$LAYOUT_PATCH"; then
  git -C "$ROBODOJO_DIR" apply --unidiff-zero "$LAYOUT_PATCH"
else
  echo 'RoboDojo seed manager differs from the pinned version; review the layout patch before installing.' >&2
  exit 1
fi
if git -C "$ROBODOJO_DIR" apply --reverse --check --unidiff-zero "$TASK_PATCH" 2>/dev/null; then
  printf '%s\n' 'RoboDojo task instructions match the published panel source.'
elif git -C "$ROBODOJO_DIR" apply --check --unidiff-zero "$TASK_PATCH"; then
  git -C "$ROBODOJO_DIR" apply --unidiff-zero "$TASK_PATCH"
else
  echo 'RoboDojo task instructions differ from the pinned version; review the task source patch before installing.' >&2
  exit 1
fi
if git -C "$XPOLICYLAB_DIR" apply --reverse --check "$REQUEST_PATCH" 2>/dev/null; then
  printf '%s\n' 'XPolicyLab unbounded request wait patch is already applied.'
elif git -C "$XPOLICYLAB_DIR" apply --check "$REQUEST_PATCH"; then
  git -C "$XPOLICYLAB_DIR" apply "$REQUEST_PATCH"
else
  echo 'XPolicyLab source differs from the pinned version; review the request wait patch before installing.' >&2
  exit 1
fi
BASE_ENV_CFG="${XPOLICYLAB_DIR}/../RoboDojo/env_cfg/arx_x5.yml"
OVERLAY_ENV_CFG="${XPOLICYLAB_DIR}/../RoboDojo/env_cfg/gpt_policy_x5.yml"
if [[ -f "$BASE_ENV_CFG" ]]; then
  sed -e 's/intrinsic_matrix: false/intrinsic_matrix: true/' \
      -e 's/extrinsic_matrix: false/extrinsic_matrix: true/' \
      "$BASE_ENV_CFG" >"$OVERLAY_ENV_CFG"
fi
FRANKA_ENV_CFG="${XPOLICYLAB_DIR}/../RoboDojo/env_cfg/gpt_policy_franka.yml"
cat >"${FRANKA_ENV_CFG}" <<'EOF'
config_name: gpt_policy_franka
config:
  sim: sim_config
  scene: default
  robot: franka
  camera: camera_config
observation:
  collect_freq: 25
  robot:
    joint_states: true
    world_ee_state: true
  vision:
    approximate_depth: false
    depth: false
    intrinsic_matrix: true
    extrinsic_matrix: true
    shape: true
robots:
  - {
    robot_type: arm,
    robot_name: franka,
    coupled: False,
    default_root_pos: [0.0, 0.6, 0.765],
    default_root_rot: [0.707, 0, 0, -0.707],
    grasp_perfect_direction: "back"
  }
EOF
cat >"${XPOLICYLAB_DIR}/../RoboDojo/env_cfg/robot/franka.yml" <<'EOF'
robots:
  - {
    robot_type: arm,
    robot_name: franka,
    coupled: False,
    default_root_pos: [0.0, 0.6, 0.765],
    default_root_rot: [0.707, 0, 0, -0.707],
    grasp_perfect_direction: "back"
  }
EOF
POLICY_DIR="${XPOLICYLAB_DIR}/policy/${POLICY_NAME}"
mkdir -p "$POLICY_DIR"
printf '%s\n' '"""GPT-Policy adapter for RoboDojo/XPolicyLab."""' >"${POLICY_DIR}/__init__.py"
cat >"${POLICY_DIR}/model.py" <<'EOF'
from XPolicyLab.model_template import ModelTemplate
from gpt_policy.robodojo.model import GPTPolicyModel

class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.impl = GPTPolicyModel(model_cfg)
    def reset(self):
        self.impl.reset()
    def update_obs(self, obs):
        self.impl.update_obs(obs)
    def update_obs_batch(self, obs_list):
        self.impl.update_obs_batch(obs_list)
    def is_episode_done(self):
        return self.impl.is_episode_done()
    def clear_terminal_decision(self, feedback=None):
        self.impl.clear_terminal_decision(feedback)
    def is_terminal_give_up(self):
        return self.impl.is_terminal_give_up()
    def is_execution_blocked(self):
        return self.impl.is_execution_blocked()
    def get_action(self):
        return self.impl.get_action()
    def get_action_batch(self, env_idx_list=None):
        return self.impl.get_action_batch(env_idx_list)
EOF
cat >"${POLICY_DIR}/deploy.py" <<'EOF'
from gpt_policy.robodojo.deploy import eval_one_episode, eval_one_episode_batch
EOF
cat >"${POLICY_DIR}/deploy.yml" <<EOF
policy_name: ${POLICY_NAME}
model: gpt-6-astra
protocol: ws
request_timeout_s: null
request_warning_s: 120
host: localhost
port: 19000
env_cfg_type: gpt_policy_x5
action_type: ee
eval_batch: false
gpt_policy_config: $(realpath "$GPT_CONFIG")
calibration_manifest: $(realpath "$CALIBRATION")
arms: [left, right]
icl_enabled: true
icl_mode: video+action
icl_dataset_root: /mnt/data/cpfs/b5/post_train_data/robodojo_sim
icl_cache_dir: ${ROOT_DIR}/var/cache/robodojo_icl
icl_keyframes: 6
EOF
cat >"${POLICY_DIR}/setup_eval_policy_server.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
PORT="\${9:-19000}"
HOST="\${10:-0.0.0.0}"
export ROBODOJO_TASK_NAME="\${2:-}"
cd "$(realpath "$XPOLICYLAB_DIR")"
export PYTHONPATH="$(realpath "$ROOT_DIR/src"):\$PWD\${PYTHONPATH:+:\$PYTHONPATH}"
if ! command -v codex >/dev/null 2>&1; then
  for candidate in /mnt/workspace/.local/share/*/export_bin/codex; do
    if [[ -x "\$candidate" ]]; then
      export PATH="\$(dirname "\$candidate"):\$PATH"
      break
    fi
  done
fi
# The Codex plugin may leave an older npm wrapper earlier in PATH while the
# authenticated standalone release is newer. Prefer the standalone binary
# when it is present so RoboDojo and GPT-Policy use the same model capability
# that the codex command itself advertises after an upgrade.
for candidate in /root/.codex/packages/standalone/releases/*/bin; do
  if [[ -x "\$candidate/codex" ]]; then
    export PATH="\$candidate:\$PATH"
  fi
done
command -v codex >/dev/null 2>&1 || {
  echo "Codex CLI not found; install/authenticate Codex or set PATH before starting the server." >&2
  exit 1
}
exec "${ROOT_DIR}/.venv/bin/python" -m client_server.ws.model_server \\
  --config-path "$(realpath "${POLICY_DIR}/deploy.yml")" --host "\${HOST}" --port "\${PORT}"
EOF
chmod +x "${POLICY_DIR}/setup_eval_policy_server.sh"
cat >"${POLICY_DIR}/setup_eval_env_client.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
BENCH="\${1}"; TASK="\${2}"; CKPT="\${3}"; ENV_CFG="\${4}"
ACTION="\${5}"; SEED="\${6}"; ENV_GPU="\${7}"; EVAL_ENV="\${8}"
ADDITIONAL_INFO="\${9}"; PORT="\${10}"; HOST="\${11:-localhost}"

# Keep the calibration-complete overlay even when the generic RoboDojo sweep
# passes its default environment configuration.
ENV_CFG="gpt_policy_x5"
if [[ -d "\${HOME}/miniconda3/envs/\${EVAL_ENV}/bin" ]]; then
  export PATH="\${HOME}/miniconda3/envs/\${EVAL_ENV}/bin:\${PATH}"
fi
exec bash "${ROOT_DIR}/third_party/RoboDojo/scripts/eval_policy.sh" \\
  --dataset_name "\${BENCH}" --task_name "\${TASK}" \\
  --env_cfg_type "\${ENV_CFG}" --policy_name GPT_Policy \\
  --host "\${HOST}" --port "\${PORT}" --protocol ws \\
  --root_dir "${ROOT_DIR}/third_party/RoboDojo" --device_id "\${ENV_GPU}" \\
  --additional_info "\${ADDITIONAL_INFO}" --seed "\${SEED}"
EOF
chmod +x "${POLICY_DIR}/setup_eval_env_client.sh"
cat >"${POLICY_DIR}/eval.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
# RoboDojo's simulator-side shell scripts invoke python3 directly. Use the
# simulator conda environment for those helpers (the policy server itself
# remains in GPT-Policy's local virtual environment).
if [[ -d "\${HOME}/miniconda3/envs/RoboDojo/bin" ]]; then
  export PATH="\${HOME}/miniconda3/envs/RoboDojo/bin:\${PATH}"
fi
BENCH="\${1}"; TASK="\${2}"; CKPT="\${3}"; ENV_CFG="\${4}"
ACTION="\${5}"; SEED="\${6}"; POLICY_GPU="\${7}"; ENV_GPU="\${8}"
# The generated overlay enables simulator intrinsics/extrinsics and is the
# only env config whose observation contract is calibration-complete.
ENV_CFG="gpt_policy_x5"
PORT="\$("${ROOT_DIR}/.venv/bin/python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
"${POLICY_DIR}/setup_eval_policy_server.sh" "\${BENCH}" "\${TASK}" "\${CKPT}" "\${ENV_CFG}" "\${ACTION}" "\${SEED}" "\${POLICY_GPU}" RoboDojo "\${PORT}" 127.0.0.1 &
SERVER_PID=\$!
cleanup() { kill "\${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT
set +e
bash "${ROOT_DIR}/third_party/RoboDojo/scripts/robodojo.sh" client \\
  --task "\${TASK}" --env-cfg "\${ENV_CFG}" --policy-name GPT_Policy \\
  --policy-host 127.0.0.1 --policy-port "\${PORT}" --env-gpu "\${ENV_GPU}"
STATUS=\$?
set -e
exit "\${STATUS}"
EOF
chmod +x "${POLICY_DIR}/eval.sh"
echo "Installed ${POLICY_NAME} wrapper at ${POLICY_DIR}"

# Generate a single-arm Franka profile from the same adapter. The transport
# and calibration plumbing are shared; only the embodiment, DOF, arm key and
# environment overlay differ.
FRANKA_POLICY_DIR="${XPOLICYLAB_DIR}/policy/GPT_Policy_Franka"
mkdir -p "${FRANKA_POLICY_DIR}"
cp -a "${POLICY_DIR}/__init__.py" "${POLICY_DIR}/model.py" "${POLICY_DIR}/deploy.py" \
  "${POLICY_DIR}/setup_eval_policy_server.sh" "${POLICY_DIR}/setup_eval_env_client.sh" \
  "${POLICY_DIR}/eval.sh" "${FRANKA_POLICY_DIR}/"
cat >"${FRANKA_POLICY_DIR}/deploy.yml" <<EOF
policy_name: GPT_Policy_Franka
model: gpt-6-astra
protocol: ws
request_timeout_s: null
request_warning_s: 120
host: localhost
port: 19000
env_cfg_type: gpt_policy_franka
action_type: ee
eval_batch: false
gpt_policy_config: $(realpath "${ROOT_DIR}/configs/examples/robodojo_franka.json")
calibration_manifest: $(realpath "${ROOT_DIR}/configs/examples/robodojo_franka_calibration.json")
robot_model: Franka
dof: 7
arms: [franka]
EOF
sed -i 's/GPT_Policy/GPT_Policy_Franka/g; s/gpt_policy_x5/gpt_policy_franka/g' "${FRANKA_POLICY_DIR}/eval.sh"
sed -i 's/GPT_Policy/GPT_Policy_Franka/g; s/gpt_policy_x5/gpt_policy_franka/g' "${FRANKA_POLICY_DIR}/setup_eval_env_client.sh"
sed -i 's/GPT_Policy/GPT_Policy_Franka/g' "${FRANKA_POLICY_DIR}/setup_eval_policy_server.sh"
chmod +x "${FRANKA_POLICY_DIR}"/*.sh
echo "Installed Franka wrapper at ${FRANKA_POLICY_DIR}"
