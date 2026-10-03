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
**Authors:** Dongzhou Cheng, [Taoran Yi](https://taoranyi.com/), [Ye Fang](https://aleafy.github.io/), Xingwu Zhang, Fan Feng, Yixuan Li, Gengxiong Zhuang, Rongze Wang, Shuai Yang, Wei Song, Weizhi Xue, Minyan Wu, Jie Gui, Jiaqi Wang, and Tong Wu.

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
python scripts/install_drivers.py yam
python scripts/install_drivers.py realsense
# ARX instead: python scripts/install_drivers.py arx
# YAM alternative: python -m pip install -e '.[yam,realsense]'
```

Agent CLIs are external dependencies. Install and authenticate the provider you select; credentials are stored outside this repository.

## Quick start

The default is **YAM (`yambox`) + Codex (`gpt-6-astra`)**. Run from the project directory after installation. For different hardware, configure a [machine profile](configs/machines/README.md) first.

### 1. Check the configuration

```bash
source .venv/bin/activate
gpt-policy --check
```

The default reports `machine: yambox`, `backend: yam`, and `hardware_opened: false`. Local or explicit profile overrides take precedence.

### 2. Run with a robot demonstration

Replace the path with your reviewed `demo.json`. This command starts the robot task:

```bash
gpt-policy "Use the robot demonstration as a reference. Unscrew and remove the bottle cap, leaving the bottle standing securely on the table." \
  --demo /path/to/robot-demonstration/demo.json
```

The default `auto` mode includes recorded states and actions when available. No extra `--demo-mode video+action` is needed.

### 3. Reuse the saved task

Use the filename printed by `Request:`, for example:

```bash
gpt-policy --input-json request_json/unscrew-remove-bottle-cap.json
```

Task requests live in `request_json/`; generated context, images and recordings live in `var/`.

| Need | Documentation |
| --- | --- |
| Choose `auto`, `video`, or `video+action` | [Demonstration modes](docs/demonstrations.md#select-a-mode) |
| Preview context and see the loaded mode | [Context inspection](docs/demonstrations.md#see-which-mode-was-actually-loaded) |
| Change machines, cameras, or live-window settings | [Runtime configuration](docs/runtime.md) |

## Results from the paper

<p>🎯 <b>Target Image / Self History / Human–Robot Interaction:</b> 100% success on each of the six tasks evaluated under these conditions.</p>
<p>👀 <b>Human Video:</b> Success rate improves from 0% to 67% on both towel and notebook pickup.</p>
<p>🤖 <b>Robot Video + Action:</b> Success rate improves from 0% to 100% on bottle opening and from 0% to 67% on plug reinsertion.</p>

## Repository layout

```text
src/gpt_policy/       protocol, input preparation, planning, recording, adapters
configs/default.json  default YAM/yambox profile for `gpt-policy "..."`
configs/agents/       provider examples
configs/examples/     local-machine templates
configs/machines/     yambox, arx247 and arx248 deployment profiles
configs/calibration/  measured intrinsics and camera transforms
configs/robots/       portable robot and motion defaults
requirements/         installation entry points for each backend
scripts/              driver installation, evaluation and provider checks
tests/                core offline configuration/runtime regression tests
docs/assets/          figures used in this README
```

The public tree includes the deployed YAM/ARX machine profiles and their measured calibration. It excludes private task prompts, real credentials, run recordings and evaluation history. The paper's physical demonstration records and complete evaluation environment are not included by implication.

## Development

The public repository keeps a focused [offline regression suite](tests/README.md) for configuration, live-window behavior, camera controls and recording layout.

```bash
python -m pytest -q
python -m compileall -q src
```

## TODO

- [x] Release the GPT-Policy pipeline for real-world ARX robots, including the complete harness and format adapters for different context types.
- [x] Release the YAM pipeline with hardware integration and flexible context support.
- [ ] Release the RoboDojo simulation pipeline for reproducible evaluation.
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
