#!/usr/bin/env bash
set -euo pipefail

# Bootstrap source checkouts and a Python environment without silently
# downloading Isaac Sim (which is large, GPU/driver-specific, and licensed).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${GPT_POLICY_VENV:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_SIM_DEPS=0
# The container's global pip.conf points at an unavailable internal mirror.
# Allow an explicit override, but use public PyPI by default for a fresh venv.
export PIP_INDEX_URL="${GPT_POLICY_PYPI_INDEX:-https://pypi.org/simple}"

usage() {
  cat <<'EOF'
Usage: scripts/setup_robodojo.sh [--install-sim-deps]

Creates/uses .venv, installs GPT-Policy, and checks out pinned RoboDojo and
XPolicyLab revisions. --install-sim-deps also runs RoboDojo's upstream
installer; it may install Isaac Sim 5.1, Isaac Lab, CuRobo and large CUDA
packages and should only be used on the simulator host.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --install-sim-deps) INSTALL_SIM_DEPS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("RoboDojo requires Python 3.11 or newer")
PY

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
VENV_PYTHON="$VENV_DIR/bin/python"
# The base image already contains a Miniconda installation outside PATH.
# Make the upstream installer reuse it instead of attempting a second install.
if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
  export PATH="${HOME}/miniconda3/bin:${HOME}/miniconda3/condabin:${PATH}"
fi
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -e "$ROOT_DIR"

checkout() {
  local name="$1" url="$2" commit="$3" dest="$ROOT_DIR/third_party/$1"
  if [[ ! -d "$dest/.git" ]]; then
    git clone "$url" "$dest"
  fi
  git -C "$dest" fetch --depth 1 origin "$commit"
  git -C "$dest" checkout --detach "$commit"
  echo "$name: $(git -C "$dest" rev-parse HEAD)"
}

checkout RoboDojo https://github.com/RoboDojo-Benchmark/RoboDojo.git \
  726e9aabfaa642203722eb126f5eaf0f37f3e1ad
checkout XPolicyLab https://github.com/XPolicyLab/XPolicyLab.git \
  e75f56b57d68561bf223faaf1198908a401602b4

# Install the lightweight XPolicyLab transport/client dependencies into the
# same venv. Isaac Sim itself remains opt-in below.
"$VENV_PYTHON" -m pip install -e "$ROOT_DIR/third_party/XPolicyLab"

bash "$ROOT_DIR/scripts/install_robodojo_policy.sh" \
  --xpolicylab-dir "$ROOT_DIR/third_party/XPolicyLab"

if [[ "$INSTALL_SIM_DEPS" == 1 ]]; then
  export OMNI_KIT_ACCEPT_EULA=YES
  bash "$ROOT_DIR/third_party/RoboDojo/scripts/install.sh" --install
  # The upstream installer can leave an editable IsaacLab entry pointing at
  # the checkout path used on its build machine. Rebind it to this checkout
  # so the RoboDojo client can import isaaclab from the current workspace.
  ROBO_DOJO_PYTHON="${HOME}/miniconda3/envs/RoboDojo/bin/python"
  ISAACLAB_SOURCE="$ROOT_DIR/third_party/RoboDojo/third_party/IsaacLab/source/isaaclab"
  if [[ -x "$ROBO_DOJO_PYTHON" && -f "$ISAACLAB_SOURCE/pyproject.toml" ]]; then
    PIP_INDEX_URL="$PIP_INDEX_URL" "$ROBO_DOJO_PYTHON" -m pip install \
      --no-build-isolation -e "$ISAACLAB_SOURCE"
  fi
else
  echo "Source checkouts ready. Isaac Sim dependencies not installed."
  echo "Run with --install-sim-deps on the GPU simulator host when ready."
fi

# Generate the adapter again after the upstream installer.  Its submodule
# setup may reset the separate XPolicyLab checkout while initializing policy
# sources.
bash "$ROOT_DIR/scripts/install_robodojo_policy.sh" \
  --xpolicylab-dir "$ROOT_DIR/third_party/XPolicyLab"

# RoboDojo's client resolves policy deploy adapters relative to its own root.
# Reuse the separately pinned XPolicyLab checkout instead of maintaining a
# second copy under RoboDojo/XPolicyLab. Do this after the upstream installer,
# which may initialize its own submodule at that path.
if [[ -d "$ROOT_DIR/third_party/RoboDojo/XPolicyLab" && ! -e "$ROOT_DIR/third_party/RoboDojo/XPolicyLab/policy" ]]; then
  rmdir "$ROOT_DIR/third_party/RoboDojo/XPolicyLab"
  ln -s ../XPolicyLab "$ROOT_DIR/third_party/RoboDojo/XPolicyLab"
elif [[ -d "$ROOT_DIR/third_party/RoboDojo/XPolicyLab" ]]; then
  # If RoboDojo initialized its own XPolicyLab submodule, copy only our
  # generated adapter into that checkout; upstream policy sources stay intact.
  mkdir -p "$ROOT_DIR/third_party/RoboDojo/XPolicyLab/policy"
  rm -rf "$ROOT_DIR/third_party/RoboDojo/XPolicyLab/policy/GPT_Policy"
  cp -a "$ROOT_DIR/third_party/XPolicyLab/policy/GPT_Policy" \
    "$ROOT_DIR/third_party/RoboDojo/XPolicyLab/policy/GPT_Policy"
fi
