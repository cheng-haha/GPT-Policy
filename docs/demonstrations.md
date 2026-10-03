# Demonstration context: prepare, inspect, and execute

`gpt-policy` turns a historical demonstration into ordered text and images for the robot agent's first turn. By default, it includes recorded robot states and actions when the source contains them. The current task and live observations guide execution; historical trajectories are planning references, not commands to replay.

## Quick start

Run these commands from the project directory with `gpt-policy` available in the environment. Replace `/path/to/demo.json` with an existing reviewed demonstration.

```bash
# 1. Generate context for inspection, without opening cameras or the robot.
gpt-policy "Unscrew and remove the bottle cap, leaving the bottle standing securely on the table." \
  --demo /path/to/demo.json \
  --prepare-only var/prepared/bottle-auto

# 2. Inspect the effective mode and saved text/image context.
python -m json.tool var/prepared/bottle-auto/input-videos/video-000/provenance.json
python -m json.tool var/prepared/bottle-auto/input.json

# 3. Execute using that prepared context. This starts a physical robot task.
gpt-policy --input-json var/prepared/bottle-auto/input.json
```

The preparation destination must be a **new directory**. A reviewed `demo.json` is expanded offline, without model calls. Preparing a raw video or a new recorded run requires FFmpeg and can call the configured vision selector. `--prepare-only` skips task naming, robot decisions, and hardware initialization.

## How context is generated

1. **Resolve the request.** Task text, `--demo`, and JSON media blocks are normalized into the same input representation. Explicit mode choices are applied, and `auto` is resolved from the source data.
2. **Load or select keyframes.** Reviewed JSON retains its approved frames and available views. New video/recordings go through frame extraction and visual selection; recorded runs can also supply aligned state and action data.
3. **Build historical context.** Each demonstration becomes a historical-reference notice, source metadata, chronological keyframe descriptions and images, and an end-of-demonstration marker. `video+action` adds available states, coordinate-frame information, alignment, and sampled action rows. Surrounding user text and image blocks keep their order.
4. **Send it with the first observation.** The agent receives the prepared text/images together with the current task, measured robot state, and live camera observations. Historical context stays separate from current observations.

The model receives actual image bytes and numeric text, not just a path to `demo.json` or `actions.jsonl`. Action rows use a shared `action_sample_encoding.columns` header and per-stage `action.sample_rows`. They retain segment endpoints, periodic samples and gripper-command transitions. `null` means missing data; `"="` repeats the preceding value within the same action segment. Original imported action segments remain in `actions.jsonl` for inspection; the model gets their compact representation.

Views come from the source. A top-only demonstration remains top-only; a reviewed three-view demonstration can provide top and both wrist images at each keyframe. For example, 14 keyframes with three distinct images each produce 42 images. Writing “three-view” in a prompt does not create missing views. Reviewed input is limited to 1–24 keyframes and at most 48 unique images per demonstration; oversized input is rejected rather than silently trimmed.

## Choose an input source

| Source | How to load it | Available context |
| --- | --- | --- |
| Reviewed `demo.json`, or a directory containing it | Inline path or `--demo` | Approved frames, views and any recorded state/action fields |
| Current project recording directory | Inline path or `--demo` | Video/keyframe selection; `states.jsonl` enables recorded state/action context |
| Exported YAM episode directory | Inline path or `--demo` | Exported video; state/action messages from `episode.mcap` |
| Ordinary video file | Inline path or `--demo` | Images and annotations; no numeric robot actions |
| Request JSON with `content` | `--input-json` or inline path | Text, images and unexpanded `video` blocks |
| Prepared `input.json` | `--input-json` or inline path | Already-expanded text/images, with the chosen mode frozen |

Examples below execute a task. Add `--prepare-only var/prepared/new-name` to inspect inputs first.

```bash
# A path directly in the task text.
gpt-policy "Use /path/to/demo.json as a reference. Unscrew and remove the bottle cap, leaving the bottle standing securely on the table."

# The same source supplied explicitly.
gpt-policy "Unscrew and remove the bottle cap, leaving the bottle standing securely on the table." \
  --demo /path/to/demo.json
```

CLI paths are relative to the current working directory; media paths inside request JSON are relative to that JSON file. Both support `~`. Quote paths containing spaces inside the instruction, for example `"Use '/path/to/my demo/demo.json' as a reference. ..."`. One demonstration can be referenced inline; combine multiple demonstrations with JSON. `--demo` or `--input-json` bypasses automatic path extraction from the CLI task text.

`--input-json` expects a **request JSON or prepared input**, not a raw demonstration schema. Load a raw `demo.json` with `--demo` or an inline path. A request can be saved as `request_json/bottle.json`:

```json
{
  "instruction": "Unscrew and remove the bottle cap, leaving the bottle standing securely on the table.",
  "model": "gpt-6-astra",
  "content": [
    "Adapt the historical demonstration to the current observations.",
    {"video": "/path/to/demo.json", "mode": "auto"}
  ]
}
```

```bash
gpt-policy --input-json request_json/bottle.json --prepare-only var/prepared/bottle-json
```

Use a `video` block to load a demonstration from request JSON. A path mentioned only in its `instruction` text is not automatically expanded. Each video block may have its own mode; omitting `mode` means `auto`.

## Select a mode

| Requested mode | Resolved behavior | State/action data in model context |
| --- | --- | --- |
| Omitted, or `auto` | Inspect the source and choose `video` or `video+action` | Included when available |
| `video` | Keep images and annotations; omit state, action and numeric alignment fields | Omitted even when available |
| `video+action` | Include available recorded state/action fields; reject a source with neither | Included; missing channels are not invented |

Automatic selection checks nonempty `state`/`action` fields in reviewed keyframes, actual state rows in project recordings, or supported state/action messages in a YAM MCAP episode. It does not infer data availability from the task name, filename, or wording such as “robot states and action trajectories”. An ordinary video cannot acquire actions from nearby sidecar files; supply the complete recording directory instead.

For a reviewed demonstration containing numeric data, the following two commands produce the same demonstration context:

```bash
gpt-policy "Follow the demonstration using the current scene." --demo /path/to/demo.json \
  --prepare-only var/prepared/demo-auto
gpt-policy "Follow the demonstration using the current scene." --demo /path/to/demo.json \
  --demo-mode video+action --prepare-only var/prepared/demo-action
```

To prepare an image-only version of the same source:

```bash
gpt-policy "Follow the demonstration using the current scene." --demo /path/to/demo.json \
  --demo-mode video --prepare-only var/prepared/demo-video
```

Mode priority is:

`--demo-mode` → explicit mode words in the current CLI instruction → each JSON video's `mode` → `auto`.

Explicit wording such as `Use video mode.` or `Use video+action mode.` is supported; conflicting mode declarations in the CLI instruction are rejected unless the flag overrides them. Generic wording such as “video demonstration” does not force image-only mode. Mode words embedded in paths are ignored. When loading a request JSON, its own instruction text is not reinterpreted as a mode override: use the media block's `mode` field.

An old request with `"mode": "video"` remains image-only. Change it to `"mode": "auto"`, remove that field, or override it for this run:

```bash
gpt-policy --input-json request_json/bottle.json --demo-mode auto \
  --prepare-only var/prepared/bottle-json-auto
```

**A prepared `input.json` contains text and images, not unresolved video blocks.** Adding `--demo-mode` to that file does not change or remove its historical state/action text. To change modes, re-prepare the original demonstration or the saved `request.json` into a new directory. For an explicit no-demonstration run, use `--input-json` with a text-only request; a free-form task alone can otherwise match a saved request containing a demonstration.

## See which mode was actually loaded

Preparation and normal execution print the **resolved** mode for each expanded demonstration. `auto` therefore appears as `video` or `video+action`, not as a third model-input format. Example output for a reviewed bottle demonstration:

```text
Demonstration ready (video+action): 13 keyframes, 13 state frames, 205 action samples
```

The same source with `--demo-mode video` prints:

```text
Demonstration ready (video): 13 keyframes, 0 state frames, 0 action samples
WARNING: video mode omitted available demonstration states/actions. Use --demo-mode video+action to include them.
```

Counts depend on the source. `keyframes` counts timestamps, not camera images. `state frames` counts keyframes with states; `action samples` counts compact samples included in context, not every raw sample. A source containing states but no action samples can still resolve to `video+action`.

Read the same information for every demonstration in a prepared package:

```bash
python - var/prepared/bottle-auto <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
for path in sorted((root / "input-videos").glob("video-*/provenance.json")):
    data = json.loads(path.read_text())
    fields = ("mode", "keyframes", "unique_images", "state_keyframes",
              "action_keyframes", "input_action_samples", "output_action_samples",
              "numeric_data_omitted")
    print(path.parent.name, json.dumps({k: data[k] for k in fields}, indent=2))
PY
```

The historical-reference text in `input.json` also states `Input mode: video.` or `Input mode: video+action.`. Normal runs log preparation metadata as `video_preprocessed` events. Loading an already-prepared input does not expand a video again, so it does not print a new `Demonstration ready` line; inspect the saved input or its original provenance. `--check` checks configuration without preparing a demonstration; use `--prepare-only` to inspect demonstration mode and counts.

## Files to inspect and reuse

| File in the preparation directory | Meaning |
| --- | --- |
| `request.json` | Original task/media request with absolute source paths and resolved modes; requires the original source to re-prepare |
| `input.json` and its referenced images | Expanded historical text/images for the first turn; portable when the package is moved together |
| `input-videos/video-000/demo.json` | Exported keyframes, image references, and mode-specific state/action data |
| `input-videos/video-000/actions.jsonl` | Imported action segments before context compression; created only in `video+action` |
| `input-videos/video-000/provenance.json` | Source/image hashes, resolved mode, state/action counts, and selection/compression provenance |

Additional demonstrations use `video-001`, `video-002`, and so on. Action fields are historical references; commanded targets and measured feedback retain their separate meanings. Missing measurements are not reconstructed as recorded motion.

To regenerate the context in another mode, use the raw request:

```bash
gpt-policy --input-json var/prepared/bottle-auto/request.json \
  --demo-mode video --prepare-only var/prepared/bottle-video
```

To execute the exact prepared task and demonstration context, use its expanded input:

```bash
gpt-policy --input-json var/prepared/bottle-auto/input.json
```

These execute commands load the configured machine/agent, open its cameras and robot, obtain new observations, and run the normal control loop. Matching prepared context does not imply identical live observations or identical model decisions across physical runs.

Normal free-form commands can use the task-selection model to reuse an existing `request_json/*.json`. Explicitly referenced demonstration paths and resolved modes filter incompatible candidates before that selection. `--input-json` uses the specified request directly and bypasses task lookup; `--prepare-only` also skips the task-selection model.

Demonstration mode and the live-image history window are independent. The supplied Codex profile disables image-count refresh by default (`live_image_window: null`); see [runtime configuration](runtime.md). Generated requests and `var/` artifacts remain local and are excluded from Git. Source demonstration files remain at the locations supplied by the user.

## Code entry points

- [Request normalization](../src/gpt_policy/input/request.py) and [automatic mode resolution](../src/gpt_policy/input/demo_mode.py).
- [Demonstration expansion and action text](../src/gpt_policy/input/demonstration.py).
- [Model-interface text and image serialization](../src/gpt_policy/harness/input_content.py).
- [CLI preparation, mode/count display and execution](../src/gpt_policy/main.py).
