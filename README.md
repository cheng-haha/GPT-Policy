# In-Context Robot Learning with General Agents

Static project page for In-Context Robot Learning with General Agents. The design follows the editorial rhythm of the OpenWAM project page while using the manuscript's own content, figures, color system, and identity.

The interactive demo gallery groups recordings by task and lets visitors switch between model/context configurations. One representative trial is shown per condition; long robot runs are encoded at 20x or 30x speed for web delivery, as labeled on each video.

A matched-condition comparison pairs the GPT-6 Astra red-towel run with two Claude Fable 5.1 batches using the same instruction, eight-frame human-video context, and head-camera view.

A second interactive comparison covers four Kimi K3 runs under text-only, no-demonstration conditions. Three task controls reuse one matched GPT-6 Astra reference per distinct task; the two Kimi K3 runs for removing fruit from a plate are grouped behind a run selector. Interrupted runs are counted as failures, and the known prompt/content mismatch is disclosed in the module. All comparison videos use the head camera and are encoded at 20x speed.

## Preview

Open `index.html` directly, or serve the folder with any static web server.

## Deploy

GitHub Pages publishes this website from the root (`/`) of the `page` branch in `cheng-haha/GPT-Policy-Eval`, with no build step.

Published website: https://cheng-haha.github.io/GPT-Policy-Eval/

The website and assets match the last update on 14 September 2026 in `cheng-haha/in-context-robot-learning`: commit `473e7537d8d3e7128199274c555b67d4e4bf1e7d` (Add bilingual language toggle).

## Source material

- Manuscript: `paper.pdf`
- Figure assets: extracted from the supplied manuscript PDF
- Demo configuration and recordings: project experiment log retrieved from Feishu on 14 September 2026
