# RoboDojo evaluation

Run GPT-Policy on the ten RoboDojo tasks using the released demonstration
contexts. The host uses the same context compiler and one-tool-per-decision
loop as the hardware backend; Isaac runs in a separate Python process.

## 1. Install

Use Python 3.10+ for the host:

```bash
python -m pip install -e '.[simulation]'
```

Install **Isaac Sim 5.1** and [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)
in a separate environment. Pin RoboDojo to
`726e9aabfaa642203722eb126f5eaf0f37f3e1ad` and initialize its submodules:

```bash
git -C /path/to/RoboDojo checkout 726e9aabfaa642203722eb126f5eaf0f37f3e1ad
git -C /path/to/RoboDojo submodule update --init --recursive
```

Download the [official assets](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo)
at revision `35efbc7dedfdbeeb6e95fb749bd885d73d483e41` into that checkout,
keeping a single `Assets/` directory. Install FFmpeg with `libx264` for video
recording. Install and authenticate Codex separately; the evaluation used
`codex-cli 0.160.0`.

## 2. Download demonstrations

**Hugging Face:** [SII-CDZ/gpt-policy-robodojo-contexts](https://huggingface.co/datasets/SII-CDZ/gpt-policy-robodojo-contexts).

Download the fixed context release:

```bash
hf download SII-CDZ/gpt-policy-robodojo-contexts --repo-type dataset \
  --local-dir data/robodojo-release
```

Each task has one fixed demonstration: **16 timestamps, three views per
timestamp, 48 images**, with annotations and action references. Standard and
random variants share their task's demonstration. No training or re-extraction
is needed.

| Bundle path | Contents |
| --- | --- |
| `contexts/<task>/prepared/input.json` and `keyframes/` | Model-ready demonstration |
| `contexts/<task>/agent_context.json` | Recorded prompt and tool schema |
| `protocol/reproduction_config.json` | Execution settings and context mappings |
| `protocol/suite.json` | The 50 evaluation cases |
| `protocol/context_bundle.json` and `protocol/assets/` | Context and asset checksums |

Load `prepared/input.json` through the configuration below. The backend supplies
the live prompt and tool schema; do not append `agent_context.json` again.

## 3. Configure and check

```bash
python data/robodojo-release/configure.py \
  --bundle data/robodojo-release \
  --robodojo-root /path/to/RoboDojo \
  --simulation-python /path/to/isaac-env/bin/python \
  --codex-bin /path/to/codex \
  --output-root var/robodojo \
  --config configs/robodojo.local.json

gpt-policy sim --config configs/robodojo.local.json \
  --suite data/robodojo-release/protocol/suite.json --check
```

Configuration verifies the ten contexts and 480 images, then resolves local
paths. `--check` verifies dependencies, layouts and assets without starting
Isaac or calling a model. Local configuration is ignored by Git.

GPU IDs refer to physical NVIDIA devices. Set `render_gpu_indices` if Vulkan
uses a different device order. For an optional camera/control smoke test:

```bash
gpt-policy sim --config configs/robodojo.local.json \
  --task classify_objects --seed 0 --layout 0 --gpus 0 --smoke
```

## 4. Evaluate on eight GPUs

```bash
gpt-policy sim --config configs/robodojo.local.json \
  --suite data/robodojo-release/protocol/suite.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --output var/robodojo/native50
```

This queues **ten tasks × five scenes**, with at most one simulator per GPU.
Use fewer GPU IDs for a smaller machine. Use a new output directory for each
run; `--resume` schedules only cases that have not started.

The profile uses `gpt-6-astra / medium`, native control-step limits, unlimited
model decisions, a 9,000-second wall limit, and `live_image_window: null`.

| Task | Scenes, seed 0 | Control limit |
| --- | --- | ---: |
| Classify objects | standard 0–4 | 1,100 |
| Classify by language | standard 0–4 | 1,100 |
| Organize table | standard 0–4 | 1,000 |
| Imitate sorting sequence | standard 0, 1, 2, 3, 5 | 1,600 |
| Arrange largest number | standard 0–1; random 0–2 | 1,050 |
| Pack objects | standard 0–1; random 0–2 | 1,300 |
| Build tower | standard 0–4 | 1,050 |
| Make Kong | standard 0–4 | 600 |
| Fold clothes | standard 0–1; random 0–2 | 500 |
| Bottles into dustbin | standard 0–4 | 700 |

For one episode, replace `--suite ...` with
`--task classify_objects --seed 0 --layout 0` and select one GPU.
Use `--prepare-only var/prepared/classify` to inspect the model input offline.

## Execution

Actions specify X5 **link6 poses in the shared environment frame**, in metres
and unit xyzw quaternions. Hardware fingertip TCP offsets are not used.
The adapter supplies robot/tool geometry and camera calibration.

`move_eef_chunk` accepts up to 64 dual-arm waypoints. Each segment is bounded
by 5 cm and 0.35 rad, with 1–5 native controls per waypoint. The released profile
uses native IK with a 0.15-rad joint-increment bound, tracking feedback,
pass-through waypoints and distributed closed-gripper transport.

`move` may finish at measured arrival; `hold` retains its requested controls
for contact, release and settling. Intermediate moves can pass through after
reaching 3 mm / 0.03 rad tolerance; endpoints retain the stability check.
Every executed control counts toward the native horizon.

The demonstration enters the first turn. Later turns append current three-view
RGB, robot state and execution feedback in the same conversation. Recoverable
provider errors retry the pending decision without replaying executed actions.
Simulation pauses while the model is thinking.

## Results

Each episode saves `result.json`, numerical controls, native scores, token
usage and three H.264 MP4 videos. Success requires a complete, valid native
terminal with `native_success=true`. Mean score averages all five native
0–100 scores per task, including failures. Report infrastructure interruptions
separately.

```bash
python scripts/audit_robodojo_run.py /path/to/episode
# Verify the downloaded demonstrations, prompts, paths and checksums:
python data/robodojo-release/verify.py data/robodojo-release
```

Physical time is controls × 0.04 seconds; model waiting contributes to wall
time. Token accounting separates cached input, uncached input and output.
This dataset contains demonstrations and the fixed evaluation configuration;
new run outputs are written to your selected output directory. Model sampling
and simulator execution can vary between runs; matching inputs and settings
does not guarantee identical outcomes.
