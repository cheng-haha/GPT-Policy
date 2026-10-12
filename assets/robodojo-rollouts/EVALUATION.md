# Evaluation protocol

## Fixed settings and scenes

- gpt-6-astra / medium; Codex 0.160.0; one JSON tool decision per model completion.
- Native control horizons; wall limit 9,000 seconds; unlimited model decisions.
- No live image window or host conversation replay. Recoverable API failures
  retry the pending decision in the same model conversation.
- Native IK; 0.15-rad joint increment; move/hold, early arrival, pass-through,
  and distributed closed-gripper transport enabled.
- RoboDojo pin: 726e9aabfaa642203722eb126f5eaf0f37f3e1ad. Dependencies, actual runtime hashes,
  asset manifests and case-specific prompts are supplied.
- Hidden simulator object states and stage scores are not supplied to the policy.
  Native success conditions and action/path validation are unchanged.

- classify_objects: standard 0, standard 1, standard 2, standard 3, standard 4.
- classify_objects_by_language: standard 0, standard 1, standard 2, standard 3, standard 4.
- organize_table: standard 0, standard 1, standard 2, standard 3, standard 4.
- imitate_sorting_sequence: standard 0, standard 1, standard 2, standard 3, standard 5.
- arrange_largest_number: standard 0, standard 1, random 0, random 1, random 2.
- pack_objects_into_box: standard 0, standard 1, random 0, random 1, random 2.
- build_tower: standard 0, standard 1, standard 2, standard 3, standard 4.
- make_kong: standard 0, standard 1, standard 2, standard 3, standard 4.
- fold_clothes: standard 0, standard 1, random 0, random 1, random 2.
- put_bottles_into_dustbin: standard 0, standard 1, standard 2, standard 3, standard 4.

Imitation uses layout 5 in place of the previously native-invalid layout 4;
it is therefore not the same complete 50-scene set as a 0–4 panel.
Three generalization tasks use standard 0–1 plus random 0–2.

## Outcomes and score

Success requires complete=true, valid_for_success_rate=true and native_success=true.
All 50 selected cases are valid native terminal outcomes. Mean score is the
arithmetic mean of all five native 0–100 final process scores per task.
The overall value is the mean of ten task means, equal here to the 50-case mean.
Every case includes its original on-run score trace; no score was replay-reconstructed.
Make Kong has binary native scoring in this pin. Intermediate visible progress
is not assigned invented points.

## Run provenance and demonstration scope

The original cohort and explicitly authorized replacement cohorts contain 69
recorded attempts: 50 selected, 10 superseded valid outcomes and 9 infrastructure
interruptions. These are cohort counts, not the total development compute.
The five packing repeats and five Make Kong v4 outcomes were selected together
regardless of score; the valid imitation layout-0 recovery is retained.
See selection_history.json and per-case provenance. No original attempt was deleted.

The selected runtime commits are 65d16dac and 3d21435. Their runtime difference
handles exhausted recoverable provider retries while preserving the current
conversation and simulation. Make Kong uses its v4 demonstration. Recorded
prompts, action schema and numerical controls are available for inspection.

These task/layout families were used during prompt, context and execution
development.
One historical expert demonstration per task is fixed across its five scenes.
Language classification explicitly relabels Classify episode 77; random variants
reuse their standard task's context. Historical expert success is not current
policy success. The expert dataset does not provide original seed/layout metadata;
overlap with evaluation layouts has not been verified.

Official asset manifests cover task-scoped resources, not every file in a
global asset tree. Local-only unused extras are named in each audit and excluded
from the reproduction dependency list. Loaded layers confirm the official X5
ARX.usd; the high-friction alternate robot was not loaded.

## Comparison and efficiency

The [reference report, section 6.3](https://anonymous-report-421.github.io/public-website/?view=1) counts selected runs,
cached input tokens and physical time separately. We use those categories and
provide raw per-case and per-call records. An executed action segment here is one
move_eef_chunk result with at least one acknowledged control; partial execution
counts once, validation-only rejection counts zero. Waypoints and all model
decisions are separately reported because a chunk can contain several targets.

Our ICL demonstration, medium effort, tool protocol, selected replacement history
and imitation scene differ from the reference's xhigh setup. Published baseline
values are attributed references, not baselines rerun under this harness.
Missing failed-call token usage prevents a complete total-cost comparison.
Shorter physical time alone is not evidence of faster end-to-end completion.
