#!/usr/bin/env bash
# Run three ARX X5 RoboDojo tasks in round-robin order, four episodes each.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELECTED_RUNNER="${ROOT_DIR}/scripts/run_robodojo_selected.sh"
GPU_ID=0
ICL_MODE=video+action
MODE=smoke
DRY_RUN=0
SKIP_SETUP=0

usage() {
  cat <<'EOF'
Usage: scripts/run_robodojo_round_robin.sh [options]

Runs exactly four rounds in this order (12 episodes total):
  build_tower -> organize_table -> put_bottles_into_dustbin

Options:
  --gpu ID          GPU for the simulator and policy server (default: 0).
  --icl-mode MODE   video+action, video, or none (default: video+action).
  --mode MODE       smoke or benchmark (default: smoke).
  --skip-setup      Reuse the existing environment.
  --dry-run         Print the 12 task invocations without launching them.
  -h, --help        Show this help.

Example:
  scripts/run_robodojo_round_robin.sh --gpu 0 --dry-run
EOF
}

fail() { echo "[robodojo round-robin] $*" >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --gpu|--icl-mode|--mode)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"
      case "$1" in
        --gpu) GPU_ID="$2" ;;
        --icl-mode) ICL_MODE="$2" ;;
        --mode) MODE="$2" ;;
      esac
      shift 2
      ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown option: $1" ;;
  esac
done

[[ "$GPU_ID" =~ ^[0-9]+$ ]] || fail "--gpu must be a non-negative integer: $GPU_ID"
case "$ICL_MODE" in video+action|video|none) ;; *) fail "invalid --icl-mode: $ICL_MODE" ;; esac
case "$MODE" in smoke|benchmark) ;; *) fail "invalid --mode: $MODE" ;; esac

tasks=(build_tower organize_table put_bottles_into_dustbin)
export ROBODOJO_GPU_IDS="$GPU_ID"
attempted=0
for round in 1 2 3 4; do
  for task in "${tasks[@]}"; do
    cmd=("${SELECTED_RUNNER}" --tasks "$task" --count 1 --gpu "$GPU_ID"
      --icl-mode "$ICL_MODE" --mode "$MODE")
    (( SKIP_SETUP || attempted > 0 )) && cmd+=(--skip-setup)
    (( DRY_RUN )) && cmd+=(--dry-run)
    printf '[robodojo round-robin] round=%s/4 task=%s\n' "$round" "$task"
    "${cmd[@]}"
    ((attempted += 1))
  done
done
printf '[robodojo round-robin] scheduled %s episodes across 4 rounds\n' "$attempted"
