# Runtime configuration

## Installation

From the repository root, create and activate a Python 3.10+ virtual environment, then install the selected backend:

```bash
python -m pip install -r requirements/yam.txt
# ARX: python -m pip install -r requirements/arx.txt
```

These requirements use the optional extras in `pyproject.toml`; the YAM SDK revision is pinned there. FFmpeg/ffprobe and authenticated provider CLIs are external tools. Install `.[dev,simulation]` to run the [offline regression suite](../tests/README.md), which covers configuration, context loading, camera controls, recording and simulation execution.

## Machine profiles

Use the [machine profile instructions](../configs/machines/README.md). The deployed `yambox`, `arx247` and `arx248` profiles and their measured calibration are included. The separate `configs/examples/` profiles still contain placeholder geometry for new installations. `--check` validates structure; it does not verify a physical calibration or connection. The default `configs/default.json` selects YAM through `machines/yambox.json`; the Codex agent uses `gpt-6-astra`. Host-specific overrides and credentials stay outside Git.

```bash
gpt-policy --check
gpt-policy --config configs/machine.local.json --check
# Included deployment profile:
gpt-policy --machine yambox --check
```

## Live image window

Image-count-triggered thread refresh is disabled in both simulation and hardware.
Omitted values and `live_image_window: null` resolve to off. Any non-null value
(including programmatic AgentConfig construction and `--live-window N`) fails
before model or hardware startup. `--no-live-window` remains a compatible no-op.
There is no 12-turn history truncation. Codex retains its native conversation
and manages its own context; full input and execution evidence remains on disk.
Exceptional provider-overload recovery is separate and remains enabled.

`--check`, run configuration and run-start metadata report
`live_image_window: null` and `live_window_enabled: false`.

## Camera exposure and gain

ARX and YAM examples include `camera_controls: {"exposure_us": 12000, "gain": 16}`. Adjust these for your devices and lighting, or set `camera_controls` to `null` to leave camera controls unchanged. The V4L2 path uses D405 controls; the RealSense path applies controls through its RGB sensors. At each task start, manual exposure is applied and read back before warmup, recording and robot startup. Unsupported/out-of-range settings or a readback mismatch fail startup; white-balance settings are preserved.

## Recording layout

Default runs use `var/runs/<provider>/<task-category>/<ICL>/<timestamp-task>_<outcome>/`, where provider is `gpt`, `claude` or `kimi` and ICL is `none_icl`, `video_icl` or `video_action_icl`. The sibling ICL directories are created together. `runtime.task_category` can supply a stable category; `runtime.record_dir` overrides the complete destination. Pre-run initialization failures go under `_initialization_failed`.

Recorded files include configuration, events, status, transcript, usage and available video/state streams. Human review determines the final success/failed label; an unrated model completion remains unreviewed. All `var/` contents, caches and generated request archives are ignored by Git. The existing `request_json/plain_text.json` default task is tracked because the ARX robot configuration references it.

## Public synchronization scope

The public runtime includes live-window controls, camera controls, recording layout, generic evaluation, core offline tests and the deployed machine/calibration profiles. Task-specific plug prompts, the fast/slow hybrid policy, credentials, private experiment plans/history and generated run data are excluded. General control prompts and provider integrations remain available.
