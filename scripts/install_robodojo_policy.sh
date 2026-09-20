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
cat >"${POLICY_DIR}/deploy.yml" <<EOF
policy_name: ${POLICY_NAME}
protocol: ws
host: localhost
port: 19000
env_cfg_type: arx_x5
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
exec "${ROOT_DIR}/.venv/bin/python" -m client_server.ws.model_server \\
  --config-path "$(realpath "${POLICY_DIR}/deploy.yml")" --host "\${HOST}" --port "\${PORT}"
EOF
chmod +x "${POLICY_DIR}/setup_eval_policy_server.sh"
echo "Installed ${POLICY_NAME} wrapper at ${POLICY_DIR}"
