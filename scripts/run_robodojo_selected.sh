#!/usr/bin/env bash
# Run selected ARX X5 RoboDojo tasks in order, with an episode count per task.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${ROBODOJO_SELECTED_RUNNER:-${ROOT_DIR}/scripts/run_robodojo_eval.sh}"

# Demonstration-backed tasks that use only the ARX X5 dual-arm scene.
AVAILABLE=(
  organize_table
  arrange_largest_number
  pack_objects_into_box
  classify_objects
  build_tower
  fold_clothes
  put_bottles_into_dustbin
)
tasks_csv="$(IFS=,; echo "${AVAILABLE[*]}")"
tasks_option_seen=0
counts_csv=""
default_count=1
gpu_id=0
icl_mode=video+action
mode=smoke
skip_setup=0
dry_run=0

usage() {
  cat <<'EOF'
Usage: scripts/run_robodojo_selected.sh [TASK_LIST] [options]

Runs tasks sequentially on one GPU. TASK_LIST is one comma-separated argument.
Task success is recorded in RoboDojo's native _result.json; a nonzero runner
exit means the evaluation did not complete and stops this sequence.
--count and --counts specify evaluation
episodes per task; they do not start a new simulator process for each episode.

Options:
  --tasks a,b,c       Alternative way to pass TASK_LIST (default: all seven).
  --count N           Default episodes per selected task (default: 1).
  --counts a=N,b=N    Per-task episode counts; override --count.
  --gpu ID            GPU for the simulator and policy server (default: 0).
  --icl-mode MODE      video+action, video, or none (default: video+action).
  --mode MODE          smoke or benchmark (default: smoke).
  --skip-setup        Use existing environment without running setup.
  --dry-run           Show the commands without starting evaluation.
  -h, --help          Show this help.

Available tasks:
  organize_table, arrange_largest_number, pack_objects_into_box,
  classify_objects, build_tower, fold_clothes, put_bottles_into_dustbin

Examples:
  scripts/run_robodojo_selected.sh build_tower,organize_table,fold_clothes
  scripts/run_robodojo_selected.sh --tasks fold_clothes,pack_objects_into_box --count 3
  scripts/run_robodojo_selected.sh --counts fold_clothes=2,build_tower=4
  scripts/run_robodojo_selected.sh --tasks arrange_largest_number --count 2 --dry-run
EOF
}

fail() { echo "[robodojo selected] $*" >&2; exit 2; }
value() {
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || fail "$1 requires a value"
}
positive_integer() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "$1 must be a positive integer: $2"
}
known_task() {
  local candidate="$1" available
  for available in "${AVAILABLE[@]}"; do
    [[ "$candidate" == "$available" ]] && return 0
  done
  fail "unknown task '$candidate' (choose from: ${AVAILABLE[*]})"
}

while (( $# )); do
  case "$1" in
    --tasks)
      value "$@"
      (( tasks_option_seen == 0 )) || fail "task list was specified more than once"
      tasks_csv="$2"
      tasks_option_seen=1
      shift 2
      ;;
    --count) value "$@"; default_count="$2"; shift 2 ;;
    --counts) value "$@"; counts_csv="$2"; shift 2 ;;
    --gpu) value "$@"; gpu_id="$2"; shift 2 ;;
    --icl-mode) value "$@"; icl_mode="$2"; shift 2 ;;
    --mode) value "$@"; mode="$2"; shift 2 ;;
    --skip-setup) skip_setup=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) fail "unknown option: $1" ;;
    *)
      (( tasks_option_seen == 0 )) || fail "task list was specified more than once"
      tasks_csv="$1"
      tasks_option_seen=1
      shift
      ;;
  esac
done

positive_integer --count "$default_count"
[[ "$gpu_id" =~ ^[0-9]+$ ]] || fail "--gpu must be a non-negative integer: $gpu_id"
case "$icl_mode" in video+action|video|none) ;; *) fail "invalid --icl-mode: $icl_mode" ;; esac
case "$mode" in smoke|benchmark) ;; *) fail "invalid --mode: $mode" ;; esac

IFS=, read -r -a tasks <<< "$tasks_csv"
[[ ${#tasks[@]} -gt 0 && "$tasks_csv" != *, && "$tasks_csv" != *,,* ]] || fail "--tasks must be a non-empty comma-separated list"
declare -A selected=() counts=()
for task in "${tasks[@]}"; do
  known_task "$task"
  [[ ! -v selected["$task"] ]] || fail "duplicate task: $task"
  selected["$task"]=1
done

if [[ -n "$counts_csv" ]]; then
  [[ "$counts_csv" != *, && "$counts_csv" != *,,* ]] || fail "invalid --counts list"
  IFS=, read -r -a count_specs <<< "$counts_csv"
  for spec in "${count_specs[@]}"; do
    [[ "$spec" == *=* ]] || fail "expected task=N in --counts: $spec"
    task="${spec%%=*}"
    count="${spec#*=}"
    known_task "$task"
    [[ -v selected["$task"] ]] || fail "--counts task is not selected: $task"
    [[ ! -v counts["$task"] ]] || fail "duplicate count for task: $task"
    positive_integer "--counts $task" "$count"
    counts["$task"]="$count"
  done
fi

export ROBODOJO_GPU_IDS="$gpu_id"
completed_tasks=()
for task in "${tasks[@]}"; do
  count="${counts[$task]:-$default_count}"
  cmd=("$RUNNER")
  (( skip_setup )) && cmd+=(--skip-setup)
  cmd+=("$mode" --icl-mode "$icl_mode" --only "$task" --eval-num "$count")
  printf '[robodojo selected] task=%s episodes=%s gpu=%s\n' "$task" "$count" "$gpu_id"
  if (( dry_run )); then
    printf 'ROBODOJO_GPU_IDS=%q' "$gpu_id"
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    if "${cmd[@]}"; then
      completed_tasks+=("$task")
      printf '[robodojo selected] evaluated: %s\n' "$task"
    else
      status=$?
      printf '[robodojo selected] runner error: %s (exit %s); stopping\n' "$task" "$status" >&2
      exit "$status"
    fi
  fi
  skip_setup=1
done

if (( dry_run == 0 )); then
  printf '[robodojo selected] evaluated=%s total=%s; read native _result.json for success and score\n' \
    "${#completed_tasks[@]}" "${#tasks[@]}"
fi
