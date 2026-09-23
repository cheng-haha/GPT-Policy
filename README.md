<div align="center">
  <img src="docs/assets/gpt-policy-wordmark-v1.png" alt="GPT-Policy wordmark" width="720" />
</div>

<div align="center">
  <img src="docs/assets/gpt-policy-teaser.png" alt="GPT-Policy overview" width="100%" />
</div>

# GPT-Policy: In-Context Robot Learning with VLM Agents

This repository is the public implementation of GPT-Policy, a closed-loop control framework that connects a fixed vision-language model (VLM) to robot tools. At deployment time, the agent can use demonstrations, goal images, interaction history, and execution feedback without gradient updates or task-specific parameter changes.

The implementation currently provides hardware adapters for two robot arm platforms: **ARX X5** and **I2RT/YAM**.

[![Paper project](https://img.shields.io/badge/paper%20project-GPT--Policy-EA4C89)](https://cheng-haha.github.io/GPT-Policy/)
[![ArXiv](https://img.shields.io/badge/arXiv-2609.19138-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.19138)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg?logo=readthedocs&logoColor=white)](https://cheng-haha.github.io/GPT-Policy/paper.pdf?v=20260915-repository-rename)
[![X](https://img.shields.io/badge/X-Post-000000?logo=x&logoColor=white)](https://x.com/z_code68632/status/2098397364725895387)

**Paper:** [*In-Context Robot Learning with VLM Agents*](https://github.com/cheng-haha/GPT-Policy/blob/main/docs/GPT-Policy.pdf)  \
**Authors:** Dongzhou Cheng, [Taoran Yi](https://taoranyi.com/), Ye Fang, Xingwu Zhang, Fan Feng, Yixuan Li, Gengxiong Zhuang, Rongze Wang, Shuai Yang, Wei Song, Weizhi Xue, Minyan Wu, Jie Gui, Jiaqi Wang, and Tong Wu.

<details>
<summary>Abstract</summary>

> Enabling robots to adapt to unfamiliar environments as readily as humans remains a moonshot goal of embodied AI. No finite collection of demonstrations can cover every task and situation a robot will encounter, making the ability to learn from context at deployment essential for generalization. Such in-context learning (ICL), however, remains largely beyond the reach of existing robotic policies. The broad agentic capabilities of commercial vision-language models (VLMs), such as GPT-6 Astra, raise a compelling question: can these models learn from demonstrations, examples, and interaction feedback, then translate that information into executable and verifiable robot behavior from a new initial state without gradient updates or persistent changes to task-specific parameters? We introduce GPT-Policy, a general-agent framework for in-context robot learning. GPT-Policy integrates a context compiler that preserves task-relevant visual transitions, a VLM that proposes robot-tool actions, and a constrained controller that verifies and executes each action and reports its outcome. We evaluate its reliability and limitations through task success and efficiency metrics, matched comparisons across models, and controlled context ablations. In real-robot trials, human video demonstrations improve task completion even without robot action labels, while aligned action references yield further gains on contact-sensitive tasks. These findings position GPT-Policy as a step toward robot adaptation through in-context learning, providing an empirical foundation for translating the general-purpose capabilities of VLMs into physical behavior and clarifying the challenges that must be overcome for reliable deployment.

</details>

## News

- 🚀 **[2026/09/16]** The [paper](https://cheng-haha.github.io/GPT-Policy/paper.pdf?v=20260915-repository-rename), [project page](https://cheng-haha.github.io/GPT-Policy/), and [code](https://github.com/cheng-haha/GPT-Policy) are now publicly available!
- 🎥 **[2026/09/11]** The first robot cases and [demonstrations](https://x.com/z_code68632/status/2098397364725895387) are added to the project page!

## Demos

Four synchronized views of three GPT-6 Astra robot runs. Click any preview to open its MP4.

<p align="center">
  <a href="docs/assets/plug-insertion-top-and-right-wrist.mp4"><img src="assets/videos/plug-top-view-preview.gif" alt="Plug insertion from top camera" width="48%" /></a>
  <a href="docs/assets/plug-insertion-top-and-right-wrist.mp4"><img src="assets/videos/plug-right-wrist-view-preview.gif" alt="Plug insertion from right wrist camera" width="48%" /></a>
  <br />
  <sub><b>Plug insertion · top view</b> &nbsp;&nbsp;&nbsp;&nbsp; <b>Plug insertion · right wrist view</b></sub>
</p>

<p align="center">
  <a href="assets/videos/gpt6-sprite-retrieval-5s.mp4"><img src="assets/videos/gpt6-sprite-retrieval-preview.gif" alt="Robot retrieving Sprite bottle" width="48%" /></a>
  <a href="assets/videos/gpt6-bottle-opening-5s.mp4"><img src="assets/videos/gpt6-bottle-opening-preview.gif" alt="Robot unscrewing a bottle cap" width="48%" /></a>
  <br />
  <sub><b>Sprite retrieval</b> · search and place the bottle &nbsp;&nbsp;&nbsp;&nbsp; <b>Bottle opening</b> · unscrew and separate the cap</sub>
</p>

The [full experiment gallery](https://cheng-haha.github.io/GPT-Policy/#results) includes the other tasks and context comparisons.

## Method overview

<div align="center">
  <img src="docs/assets/gpt-policy-overview.png" alt="GPT-Policy closed-loop architecture" width="100%" />
</div>

GPT-Policy builds one model input from the task, live camera/state observations, task references, and the previous tool result. The VLM emits one structured request; the selected adapter validates and executes it, then returns fresh observations and feedback for the next decision. Adapters support Cartesian targets and waypoint sequences, sequential IK checks, backend-specific timing, gripper control, and append-only run recording.

The available context types are:

- **Human Video:** a visual procedure that can transfer across embodiments.
- **Robot Video / Video + Action:** robot interactions, arm roles, and aligned motion references.
- **Target Image:** the desired object arrangement, position, and spacing.
- **Self History:** earlier observations, actions, results, and discovered subgoals.
- **Human-Robot Interaction:** live intent, pointing, corrections, and turn-taking.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The base install is hardware-free. Install only the backend you need:

```bash
source .venv/bin/activate
python scripts/install_drivers.py arx
python scripts/install_drivers.py yam
python scripts/install_drivers.py realsense
# or: python -m pip install -e '.[arx,realsense]'
```

Agent CLIs are external dependencies. Install and authenticate the provider you select; credentials are stored outside this repository.

### Optional RoboDojo simulation

RoboDojo and XPolicyLab are kept as pinned third-party checkouts, with their
own licenses and histories. To prepare the policy-side environment and source
trees:

```bash
./scripts/setup_robodojo.sh
source .venv/bin/activate
```

This step does not download Isaac Sim or CUDA packages. On the dedicated GPU
simulator host, run `./scripts/setup_robodojo.sh --install-sim-deps` to invoke
RoboDojo's upstream installer. The simulator and policy process communicate
through XPolicyLab's WebSocket contract, so they may run on separate machines.
The current tested simulator stack is Isaac Sim 5.1/IsaacLab 0.54.3 on an
NVIDIA driver with CUDA support; the policy host only needs the Python virtual
environment and an authenticated Codex CLI.
See [third_party/README.md](third_party/README.md) and
[THIRD_PARTY.md](THIRD_PARTY.md) for pinned revisions and license notices.

The adapter implementation lives under `src/gpt_policy/robodojo/`. Use
`configs/examples/robodojo.json` as the starting configuration. Its
calibration manifest specifies the embodiment's grasp-center offset and a
nominal base pose. The patched simulator supplies actual base poses,
environment origins, and camera matrices on every observation; these take
precedence over static values.

Camera geometry uses an explicit coordinate convention: RoboDojo emits USD
camera-to-world poses (+X right, +Y up, -Z forward), while pixel rays use
OpenCV optical axes (+X right, +Y down, +Z forward). The bridge applies
`diag(1,-1,-1,1)` on the camera side before converting into an arm base frame.
Model observations include both labeled raw extrinsics and
`base_from_optical_camera`; do not apply a second axis flip to the latter.
Calibration manifests default to `camera_extrinsic_axes: "usd"`; use
`"opencv"` only for extrinsics that have already been converted upstream.

`install_robodojo_policy.sh` also applies the versioned camera-export patch in
`scripts/patches/`: sensor poses come from live Fabric transforms, not stale
USD transforms or camera housings. Standard pinhole vertical aperture is
matched to the render resolution's square pixels, so SDK intrinsics and
rendered images agree. The bridge still supports independent `fx` and `fy`.
TCP commands and observations apply the configured grasp-center offset:
X5 is 145 mm along `link6` +X (mapped to TCP +Z); Franka is 102 mm along
`panda_hand` +Z. These are configured grasp centers, not raw flange origins
or a claim that the distal mesh boundary is exactly at the TCP.
After upgrading an existing checkout, rerun the installer and restart the
simulator and policy server. Camera regression checks:

```bash
.venv/bin/python -m pytest -q tests/test_robodojo_calibration.py tests/test_robodojo_adapter.py
/root/miniconda3/envs/RoboDojo/bin/python scripts/verify_robodojo_camera_calibration.py
/root/miniconda3/envs/RoboDojo/bin/python scripts/verify_robodojo_scene_calibration.py \
  --env-cfg gpt_policy_x5 --seeds 0,1,2 --output var/runs/robodojo/calibration_x5
/root/miniconda3/envs/RoboDojo/bin/python scripts/verify_robodojo_scene_calibration.py \
  --env-cfg gpt_policy_franka --seeds 0,1,2 --output var/runs/robodojo/calibration_franka
```

The offline tests replay three wrist-camera baselines from a recorded Push-T
failure. The live check compares SDK-projected known landmarks with the
bridge's back-projected rays. The scene test exercises the actual Push-T
environment, multiple joint configurations and resets, live camera/base/TCP
transforms, rendered markers, temporal triangulation, and small IK-driven
motions. It saves numerical checks and RGB evidence. The existing policy's
5-degree parallax criterion distinguishes usable rendered triangulation
from separately reported weak-baseline diagnostics. These tests do not call
a VLM, move physical hardware, or expose diagnostic ground truth to a policy.

When `/mnt/data/cpfs/b5/post_train_data/robodojo_sim` is available, the
generated RoboDojo policy wrapper also enables bounded in-context examples.
It matches the live task instruction to the corresponding dataset, reads one
historical episode's Parquet state/action trajectory, and extracts six
synchronized keyframes from the three camera videos into
`var/cache/robodojo_icl/`. Only that compact summary and those keyframes are
sent to Codex on the first decision; the full 62 GB dataset is never loaded
into the policy context. Set `ROBODOJO_ICL_ROOT` to another dataset location,
or set `icl_enabled: false` in the generated `deploy.yml` to disable it.

The three rollout modes use the same evaluator. `video+action` supplies the
historical keyframes and compressed state/action samples; `video` supplies
only the historical keyframes; `none` disables historical examples and leaves
the normal task prompt plus live observations:

`smoke` is the quick sweep mode: it defaults to one episode per selected task
and is intended for checking that the simulator, policy server, camera
calibration, and context path are working. `benchmark` uses the task-defined
episode counts and is intended for a full run. `--only`, `--dimension`,
`--all`, and `--dry-run` are passed through to RoboDojo's sweep runner.

The unified runner prepares the Python environment, pinned source checkouts,
generated policy adapter, and (when explicitly requested) simulator
dependencies before launching the sweep:

```bash
# Prepare lightweight dependencies automatically; default context is video + action.
./scripts/run_robodojo_eval.sh smoke --icl-mode video+action --only push_T --eval-num 1

# On a simulator host that still needs Isaac Sim/IsaacLab/CuRobo:
./scripts/run_robodojo_eval.sh --install-sim-deps smoke --only push_T --eval-num 1
```

The lower-level runner remains available when the environment is already set up:

```bash
# Default: video + action context.
./scripts/eval_robodojo_8gpu.sh smoke --icl-mode video+action --only push_T --eval-num 1

# Visual demonstration only.
./scripts/eval_robodojo_8gpu.sh smoke --icl-mode video --only push_T --eval-num 1

# Prompt/live-observation baseline.
./scripts/eval_robodojo_8gpu.sh smoke --icl-mode none --only push_T --eval-num 1
```

For a multi-GPU sweep, pass a comma-separated worker list. Each selected task
is assigned to one worker/GPU and the workers run concurrently:

```bash
ROBODOJO_GPU_IDS=0,1,2 \
./scripts/eval_robodojo_8gpu.sh smoke --icl-mode video+action \
  --only push_T,build_tower,insert_key --eval-num 1
```

Each worker starts its own Isaac Sim client, GPT-Policy WebSocket server, and
Codex app-server process. This is task-level parallelism: one task is not
split across eight GPUs. Use `--dry-run` to inspect the GPU-to-task assignment
without launching the simulator.

For a sequential evaluation of the seven demonstration-backed tasks that use
the ARX dual-arm scene, use the selected-task runner. Pass one comma-separated
task list; its order determines execution order. A completed task failure is
recorded in RoboDojo's native `_result.json` and does not stop later tasks;
an infrastructure or runner failure does. `--count` sets episodes
per task and `--counts` overrides individual tasks:

```bash
./scripts/run_robodojo_selected.sh build_tower,organize_table,fold_clothes
./scripts/run_robodojo_selected.sh --tasks fold_clothes,arrange_largest_number \
  --count 3 --counts arrange_largest_number=5 --gpu 0
./scripts/run_robodojo_selected.sh --dry-run
```

For the ten-task subset used by GPT-as-Policy, use the frozen 50-case panel in
`configs/robodojo_panel50.json`. The panel contains five published layout IDs
per task: two standard and three `_random` for number arrangement, packing,
and clothes folding; five standard for the other seven tasks. The runner
checks all 50 layout hashes against the published panel before launching any
model or simulator process. It runs 13 native RoboDojo `benchmark` invocations
and checks each `_result.json` for the exact expected layout IDs. A failed task
is a completed case; a missing native result stops the panel without counting
that case as a failure.

```bash
# Verify layout identity and inspect the commands without starting a simulator.
python3 scripts/run_robodojo_panel50.py --dry-run --gpu 0 --icl-mode none

# Run the no-demonstration comparison condition; choose a stable run ID for resume.
python3 scripts/run_robodojo_panel50.py --gpu 0 --icl-mode none \
  --run-id xingwu_panel50_01

# Resume after an interrupted group. Completed groups are not repeated.
python3 scripts/run_robodojo_panel50.py --gpu 0 --icl-mode none \
  --run-id xingwu_panel50_01 --resume
```

`--icl-mode video+action` preserves GPT-Policy's current ICL setting, but the
local dataset has no demonstration for `classify_objects_by_language`; the
runner reports that gap. Add `--require-icl` to refuse a mixed ICL condition.
The panel aligns task layouts and native scoring with the published case set;
the policy and its EEF controller remain GPT-Policy's own method. The active
ARX X5 config uses RoboDojo's upstream action interpolation and physics.

The policy has no default TCP minimum height (`z: [null, 0.45]` in arm-base
coordinates), allowing cloth contact approaches below 40 mm. Other workspace
bounds and the maximum step distance remain in force. Effective bounds are
included in GPT observations; passing these checks does not guarantee a grasp
or collision-free motion.
`scripts/verify_robodojo_cloth_contact.py --output PATH --height 0.025` runs
a deterministic approach/close/lift diagnostic in the RoboDojo Python environment.

After setup, generate the XPolicyLab policy wrapper. This creates both the
dual-arm ARX X5 profile and the single-arm Franka profile:

```bash
./scripts/install_robodojo_policy.sh
```

GPT-Policy uses the native Codex app-server interface. GPT-6-Astra requires a
recent Codex release (0.155.1 or newer in the tested setup) and an existing
`codex login`; the generated server wrapper prefers the authenticated
standalone release when an older npm wrapper is also present. Verify the
active binary with:

```bash
codex login status
/root/.codex/packages/standalone/releases/0.155.1-x86_64-unknown-linux-musl/bin/codex --version
```

On the policy host, start the WebSocket server (the server does not require
Isaac Sim):

```bash
third_party/XPolicyLab/policy/GPT_Policy/setup_eval_policy_server.sh \
  RoboDojo push_T none gpt_policy_x5 ee 0 0 RoboDojo 19000 0.0.0.0
```

On the Isaac Sim host, point RoboDojo's client at that server:

```bash
bash third_party/RoboDojo/scripts/robodojo.sh client \
  --task push_T --env-cfg gpt_policy_x5 --policy-name GPT_Policy \
  --policy-host POLICY_HOST --policy-port 19000 --env-gpu 0
```

Use `eval_batch=false` for the current GPT-Policy closed-loop VLM adapter;
it intentionally processes one environment at a time so every new image,
state, calibration matrix, and execution result reaches the same model turn.

For a local one-episode smoke evaluation (the first Isaac Sim startup can take
about a minute), use the generated convenience script:

```bash
source .venv/bin/activate
export PATH=/root/.codex/packages/standalone/releases/0.155.1-x86_64-unknown-linux-musl/bin:$PATH
export EVAL_NUM=1
bash third_party/XPolicyLab/policy/GPT_Policy/eval.sh \
  RoboDojo push_T none gpt_policy_x5 ee 0 0 0
```

The simulator-side transcript is written by the command's caller; for the
repository's long-running runs, use `var/runs/robodojo/`. GPT-Policy's own
structured observations, calibration context, decisions, and timing continue
to use the usual `var/runs/gpt/` recording format. The RoboDojo evaluator, not
the policy, assigns the final task success label.

List the canonical tasks and capability dimensions before launching a larger
run:

```bash
bash third_party/RoboDojo/scripts/robodojo.sh tasks
bash third_party/RoboDojo/scripts/robodojo.sh dimensions
```

Eight-GPU smoke and benchmark sweeps use RoboDojo's runtime-weighted task
partitioner. Each GPU runs an independent Isaac Sim client and GPT-Policy
server, so the closed-loop adapter remains single-environment while tasks run
in parallel:

```bash
# One episode per selected task on GPUs 0-7.
./scripts/eval_robodojo_8gpu.sh smoke --dimension memory --eval-num 1

# Full task-defined episode counts on GPUs 0-7.
./scripts/eval_robodojo_8gpu.sh benchmark --all --eval-num native

# Validate task assignment and commands without starting Isaac Sim.
./scripts/eval_robodojo_8gpu.sh smoke --all --eval-num 1 --dry-run
```

运行中的 RoboDojo worker 可以用监控脚本查看 GPU、任务、seed、当前步数和结果状态：

```bash
python scripts/monitor_robodojo.py
python scripts/monitor_robodojo.py --watch --interval 5
python scripts/monitor_robodojo.py --watch --running-only
```

`--watch` 会持续刷新；按 `Ctrl-C` 退出监控，不会停止评测进程。

修改 RoboDojo action/TCP 转换后，可以先运行固定低位侧推的无仿真诊断：

```bash
python scripts/verify_robodojo_tcp_path.py
```

Set `ROBODOJO_GPU_IDS` to a comma-separated subset and
`ROBODOJO_SIM_ENV` if the simulator environment has a different name. The
policy process always uses this repository's `.venv`; RoboDojo's Isaac Sim
runtime remains in its isolated simulator environment because Isaac Sim pins
packages that conflict with the policy transport dependencies.

If Astra temporarily reports `Selected model is at capacity`, the adapter
retries the same turn with bounded backoff. A persistent capacity error is a
provider-side condition; it should be distinguished from simulator, asset,
or action-schema failures in the run log.

### Franka profile

The RoboDojo bridge also provides a single-arm Franka profile. Generate it
alongside the X5 profile with `scripts/install_robodojo_policy.sh`; it uses
`gpt_policy_franka`, `GPT_Policy_Franka`, seven arm joints, and the runtime
camera intrinsic/extrinsic matrices emitted by RoboDojo. Launch it with:

```bash
bash third_party/RoboDojo/scripts/robodojo.sh client \
  --task push_T --env-cfg gpt_policy_franka \
  --policy-dir XPolicyLab/policy/GPT_Policy_Franka \
  --policy-host 127.0.0.1 --policy-port 19000 --env-gpu 0
```

The Franka profile is single-arm; the mixed X5/X5/Franka competition scene is
not yet exposed as one three-arm GPT-Policy action schema.

## Quick start

For the default ARX profile, edit the placeholders in `configs/default.json` once, then run a task directly:

```bash
source .venv/bin/activate
gpt-policy "pick up the red block"
```

The command resolves `configs/default.json` automatically. To validate the profile without opening hardware or a model session:

```bash
gpt-policy --check
```

For another machine or provider, pass an explicit profile:

```bash
cp configs/examples/yam-local.json configs/my-machine.json
# edit interfaces, camera serials, and measured calibration
gpt-policy --config configs/my-machine.json "pick up the red block"
```

An input package can contain text, images, videos, or reviewed demonstrations:

```bash
gpt-policy --input-json task.json
```

Each motion is planned from fresh feedback, checked with per-sample IK, and recorded as an append-only run directory. `Ctrl+C` requests software cancellation and cleanup; it does not replace a hardware emergency stop.

## Results from the paper

<p>🎯 <b>Target Image / Self History / Human–Robot Interaction:</b> 100% success on each of the six tasks evaluated under these conditions.</p>
<p>👀 <b>Human Video:</b> Success rate improves from 0% to 67% on both towel and notebook pickup.</p>
<p>🤖 <b>Robot Video + Action:</b> Success rate improves from 0% to 100% on bottle opening and from 0% to 67% on plug reinsertion.</p>

## Repository layout

```text
src/gpt_policy/       protocol, input preparation, planning, recording, adapters
configs/default.json  default sanitized ARX profile for `gpt-policy "..."`
configs/agents/       provider examples
configs/examples/     local-machine templates
scripts/              opt-in driver installation
tests/                offline protocol and configuration tests
docs/assets/          figures used in this README
```

The public tree excludes deployment hosts, private prompts, real credentials, run recordings, site-specific calibration, and evaluation history. The paper's physical demonstration records and complete evaluation environment are not included by implication.

## Development

```bash
python -m pytest -q
python -m compileall -q src
```

## TODO

- [x] Release the GPT-Policy pipeline for real-world ARX robots, including the complete harness and format adapters for different context types.
- [x] Release the YAM pipeline with hardware integration and flexible context support.
- [x] Release the RoboDojo policy bridge, calibration-aware context, X5/Franka
  profiles, and reproducible single-episode evaluation entry point.
- [ ] Complete and publish the full RoboDojo task/seed score matrix.
- [ ] Optimize the agent harness for context construction, feedback, and execution efficiency.

## Limitations and safety

This is a research control loop. The integrator must verify calibration, workspace limits, collision behavior, camera placement, provider configuration, and emergency-stop procedures before energizing a robot. IK acceptance and a model completion message do not establish collision-free motion or physical task success. The project license is intentionally pending; redistribution and commercial use are not granted by this preview.

See [THIRD_PARTY.md](THIRD_PARTY.md) for third-party notices and optional SDK sources.

## Citation

```bibtex
@article{cheng2026incontextrobotlearningvlm,
  title={In-Context Robot Learning with VLM Agents},
  author={Dongzhou Cheng and Taoran Yi and Ye Fang and Xingwu Zhang and Fan Feng and Yixuan Li and Gengxiong Zhuang and Rongze Wang and Shuai Yang and Wei Song and Weizhi Xue and Minyan Wu and Jie Gui and Jiaqi Wang and Tong Wu},
  journal={arxiv:2609.19138},
  year={2026}
}
```

## Acknowledgements

GPT-Policy integrates optional ARX, I2RT/YAM, RealSense, and provider CLI interfaces, with reference to [RoboCurve's inspect-robots project](https://github.com/robocurve/inspect-robots). Please see [THIRD_PARTY.md](THIRD_PARTY.md) before redistributing a deployment that includes external SDKs.
