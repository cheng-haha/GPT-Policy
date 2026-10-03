# Runtime configuration

## Installation

From the repository root, create and activate a Python 3.10+ virtual environment, then install the selected backend:

```bash
python -m pip install -r requirements/yam.txt
# ARX: python -m pip install -r requirements/arx.txt
```

These requirements use the optional extras in `pyproject.toml`; the YAM SDK revision is pinned there. FFmpeg/ffprobe and authenticated provider CLIs are external tools. The offline test suite does not require robot SDKs; install `.[dev]` to run it. Tests for optional SDKs/media tools skip when those tools are absent.

## Machine profiles

Use the [machine profile instructions](../configs/machines/README.md). The example profiles inherit portable robot/motion defaults and placeholder geometry. Replace serials, interfaces, intrinsics and camera transforms with your measurements. `--check` validates structure; it does not verify a physical calibration or connection. Site profiles, local overrides and credentials stay outside Git.

```bash
gpt-policy --check
gpt-policy --config configs/machine.local.json --check
# Or configs/machines/lab-yam.json:
gpt-policy --machine lab-yam --check
```

## Live image window

The shipped `configs/agents/codex.json` sets `live_image_window` to `null`, so image-count-triggered session refresh is off by default. Provider-overload recovery remains active. Custom profiles that omit this field retain the compatibility fallback of 8; set it explicitly to `null` for the same default behavior.

```bash
gpt-policy --check --no-live-window
gpt-policy --check --live-window 8
```

The flags apply to Codex, are mutually exclusive, and affect only this invocation. An enabled window must be an integer of at least 3. `--check`, `config.json`, `status.json` and the run-start event record the effective `live_image_window` and `live_window_enabled`. The window counts live observation groups, not demonstration views or total tokens.

## Camera exposure and gain

ARX and YAM examples include `camera_controls: {"exposure_us": 12000, "gain": 16}`. Adjust these for your devices and lighting, or set `camera_controls` to `null` to leave camera controls unchanged. The V4L2 path uses D405 controls; the RealSense path applies controls through its RGB sensors. At each task start, manual exposure is applied and read back before warmup, recording and robot startup. Unsupported/out-of-range settings or a readback mismatch fail startup; white-balance settings are preserved.

## Recording layout

Default runs use `var/runs/<provider>/<task-category>/<ICL>/<timestamp-task>_<outcome>/`, where provider is `gpt`, `claude` or `kimi` and ICL is `none_icl`, `video_icl` or `video_action_icl`. The sibling ICL directories are created together. `runtime.task_category` can supply a stable category; `runtime.record_dir` overrides the complete destination. Pre-run initialization failures go under `_initialization_failed`.

Recorded files include configuration, events, status, transcript, usage and available video/state streams. Human review determines the final success/failed label; an unrated model completion remains unreviewed. All `var/` contents, caches and generated request archives are ignored by Git.

## Public synchronization scope

The public runtime includes live-window controls, camera controls, recording layout, generic evaluation and portable tests. Task-specific plug prompts, the fast/slow hybrid policy, machine calibration/serials, private experiment plans/history and generated run data are excluded. General control prompts and provider integrations remain available.
