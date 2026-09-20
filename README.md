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
See [third_party/README.md](third_party/README.md) and
[THIRD_PARTY.md](THIRD_PARTY.md) for pinned revisions and license notices.

The adapter implementation lives under `src/gpt_policy/robodojo/`. Use
`configs/examples/robodojo.json` as the starting configuration. Its
calibration manifest must be populated from the active RoboDojo camera manager
and USD robot frames before evaluation; the identity/empty values in the
example are only a schema, not a valid calibrated run.

After setup, generate the XPolicyLab policy wrapper:

```bash
./scripts/install_robodojo_policy.sh
```

On the policy host, start the WebSocket server (the server does not require
Isaac Sim):

```bash
third_party/XPolicyLab/policy/GPT_Policy/setup_eval_policy_server.sh \
  RoboDojo push_T none arx_x5 ee 0 0 RoboDojo 19000 0.0.0.0
```

On the Isaac Sim host, point RoboDojo's client at that server:

```bash
bash third_party/RoboDojo/scripts/robodojo.sh client \
  --task push_T --env-cfg arx_x5 --policy-name GPT_Policy \
  --policy-host POLICY_HOST --policy-port 19000 --env-gpu 0
```

Use `eval_batch=false` for the current GPT-Policy closed-loop VLM adapter;
it intentionally processes one environment at a time so every new image,
state, calibration matrix, and execution result reaches the same model turn.

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
