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
    def get_action(self):
        return self.impl.get_action()
    def get_action_batch(self, env_idx_list=None):
        return self.impl.get_action_batch(env_idx_list)
EOF
cat >"${POLICY_DIR}/deploy.py" <<'EOF'
def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        for action in actions:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break

def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError("GPT-Policy RoboDojo adapter requires eval_batch=false")
EOF
cat >"${POLICY_DIR}/deploy.yml" <<EOF
policy_name: ${POLICY_NAME}
model: gpt-6-astra
protocol: ws
host: localhost
port: 19000
env_cfg_type: gpt_policy_x5
action_type: ee
eval_batch: false
gpt_policy_config: $(realpath "$GPT_CONFIG")
calibration_manifest: $(realpath "$CALIBRATION")
arms: [left, right]
EOF
cat >"${POLICY_DIR}/setup_eval_policy_server.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
PORT="\${9:-19000}"
HOST="\${10:-0.0.0.0}"
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
# that `codex` itself advertises after an upgrade.
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
cat >"${POLICY_DIR}/eval.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
# RoboDojo's simulator-side shell scripts invoke `python3` directly. Use the
# simulator conda environment for those helpers (the policy server itself
# remains in GPT-Policy's .venv).
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
exec bash "${ROOT_DIR}/third_party/RoboDojo/scripts/robodojo.sh" client \\
  --task "\${TASK}" --env-cfg "\${ENV_CFG}" --policy-name GPT_Policy \\
  --policy-host 127.0.0.1 --policy-port "\${PORT}" --env-gpu "\${ENV_GPU}"
EOF
chmod +x "${POLICY_DIR}/eval.sh"
echo "Installed ${POLICY_NAME} wrapper at ${POLICY_DIR}"

# Generate a single-arm Franka profile from the same adapter. The transport
# and calibration plumbing are shared; only the embodiment, DOF, arm key and
# environment overlay differ.
FRANKA_POLICY_DIR="${XPOLICYLAB_DIR}/policy/GPT_Policy_Franka"
mkdir -p "${FRANKA_POLICY_DIR}"
cp -a "${POLICY_DIR}/__init__.py" "${POLICY_DIR}/model.py" "${POLICY_DIR}/deploy.py" \
  "${POLICY_DIR}/setup_eval_policy_server.sh" "${POLICY_DIR}/eval.sh" "${FRANKA_POLICY_DIR}/"
cat >"${FRANKA_POLICY_DIR}/deploy.yml" <<EOF
policy_name: GPT_Policy_Franka
model: gpt-6-astra
protocol: ws
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
sed -i 's/GPT_Policy/GPT_Policy_Franka/g' "${FRANKA_POLICY_DIR}/setup_eval_policy_server.sh"
chmod +x "${FRANKA_POLICY_DIR}"/*.sh
echo "Installed Franka wrapper at ${FRANKA_POLICY_DIR}"
