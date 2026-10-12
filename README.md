# In-Context Robot Learning with VLM Agents

Static project page for In-Context Robot Learning with VLM Agents. The design follows the editorial rhythm of the OpenWAM project page while using the manuscript's own content, figures, color system, and identity.

The interactive demo gallery groups recordings by task and lets visitors switch between model/context configurations. One representative trial is shown per condition; long robot runs are encoded at 20x or 30x speed for web delivery, as labeled on each video.

A matched-condition comparison pairs the GPT-6 Astra red-towel run with two Claude Fable 5.1 batches using the same instruction, eight-frame human-video context, and head-camera view.

A second interactive comparison covers four Kimi K3 runs under text-only, no-demonstration conditions. Three task controls reuse one matched GPT-6 Astra reference per distinct task; the two Kimi K3 runs for removing fruit from a plate are grouped behind a run selector. Interrupted runs are counted as failures, and the known prompt/content mismatch is disclosed in the module. All comparison videos use the head camera and are encoded at 20x speed.

## Preview

Open `index.html` directly, or serve the folder with any static web server.
For local video seeking, use the included server with byte-range support:

```bash
python3 scripts/serve_preview.py --port 8766
```

Open http://127.0.0.1:8766/#simulation-rollouts.

## RoboDojo simulation videos

The simulation section includes a task/layout explorer inspired by
[RoboICL's rollout explorer](https://mosi-ai.github.io/RoboICL-GPT6-Astra.github.io/).
Its implementation is in `simulation-rollouts.js` and `simulation-rollouts.css`.
All 50 selected episodes from `gpt-policy-robodojo-10tasks-20261011` are included,
with their original standard/random layout identities and outcomes.

Three 640 × 480 camera recordings are joined left wrist → overhead → right wrist
into one 1920 × 480 H.264 MP4. There is one frame per control step at 25 fps,
including the initial frame. This is control-step playback, not wall-clock time.
The synchronized panel shows public numerical EEF requests and native evaluation
scores; it does not reproduce model reasoning. Native scores were not provided
to the policy. No new evaluation is run to build the gallery.

Build with Python 3.11+ and FFmpeg, using a local copy of the public release:

```bash
python3 scripts/build_simulation_gallery.py /path/to/public-release assets/robodojo-rollouts
```

The builder expects `tasks/`, `LICENSE.txt`, `THIRD_PARTY.md`, and
`protocol/EVALUATION.md` (or a root-level `EVALUATION.md`). It retains every selected outcome, checks
control/frame alignment, and verifies 36/50 successes and a mean score of 80.20.
Source hashes and original case identities are in the generated manifest.
The original release must remain separate from the output directory.
Non-motion calls and rejected motions do not advance the recorded video; the
panel displays executed numerical targets, with the original decision index.

`manifest.js` and per-case `trace.js` support both direct file opening and HTTP.
Task links in the heatmap open the corresponding video. Query parameters
`task`, `layout`, `variant`, and `speed` preserve a selected example.
The release's evaluation protocol and attribution files are retained in
`assets/robodojo-rollouts/`; their terms are unchanged.

## Deploy

GitHub Pages publishes this website from the root (`/`) of the `page` branch in `cheng-haha/GPT-Policy`, with no build step.

Published website: https://cheng-haha.github.io/GPT-Policy/

The website layout and assets are based on `cheng-haha/in-context-robot-learning` at commit `7f59dbc7b94506506f7a61997b6773c3203f3a79` (Update project page from latest manuscript). This is the final corrected version before the subsequent page redesign, including the VLM Agents title, revised abstract, synchronized video summaries, and experiment-table corrections.

## Source material

- Manuscript: `paper.pdf`
- Figure assets: extracted from the supplied manuscript PDF
- Demo configuration and recordings: project experiment log retrieved from Feishu on 14 September 2026

## Manuscript synchronization

Website text and the 16-row context-results table were synchronized with the live Overleaf abstract, introduction, method, experiments, discussion, and conclusion on 15 September 2026. English and Chinese text are updated together. The existing demo recordings remain illustrative runs; aggregate results and the model-comparison discussion follow the manuscript.

## Figure synchronization

All eight main-text figures and the downloadable `paper.pdf` now come from the Overleaf build compiled on 15 September 2026 at 13:49 PDT. Figure 1 replaces the cover photo; Figures 2–3 illustrate the method; Figures 4–7 accompany the results; Figure 8 accompanies the discussion. Each figure opens at full resolution and has English/Chinese captions. `assets/paper-figures/manifest.json` records source pages and the source PDF checksum. Figures are rendered directly from the PDF without changing their annotations or colors.
