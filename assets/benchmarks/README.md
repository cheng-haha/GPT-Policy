# RoboDojo charts and paper results

## Paper version

The local website was updated from `GPT_Policy_iclr2027__Copy_ (6).pdf`, compiled **2026-10-11 14:51:05 PDT**, 47 pages. It was already present in Downloads and is now the site's `paper.pdf`. The Overleaf project is https://www.overleaf.com/project/6ab3f98275f46d738116b160. The browser project did not respond to content reads during this update, so this is the latest locally available export, not a newly fetched Overleaf revision. The exact SHA-256 is recorded in `benchmark-data.js`.

- Paper Table 2, page 8: simulation progress scores.
- Paper Table 5, page 20: simulation success rates.
- Paper Table 1, page 6: real-robot success, mean decisions/time and sample standard deviations.
- Paper Table 3, page 9: demonstration-information, calibrated-localization and action-chunking ablations.

## Confirmed simulation data (takes precedence over the PDF)

The owner supplied an updated screenshot of paper **Table 2: One-shot RoboDojo results** on 2026-10-11 and explicitly confirmed it as authoritative. The earlier local PDF export is stale for this table. **Do not replace these simulation results with values from that export.**

GPT-Policy: **80.20 / 100**, **36/50 successes (72%)**. Per-task score / success (%):

| Task | Score | Success (%) |
| --- | ---: | ---: |
| Organize table | 65 | 20 |
| Classify objects by language | 88 | 80 |
| Imitate sorting sequence | 100 | 100 |
| Arrange largest number | 100 | 100 |
| Build tower | 46 | 40 |
| Fold clothes | 80 | 80 |
| Put bottles into dustbin | 73 | 60 |
| Pack objects into box | 70 | 60 |
| Make Kong | 80 | 80 |
| Classify objects | 100 | 100 |

## Baseline alignment (2026-10-11)

The original baseline cells and aggregate values reproduce the [GPT-as-Policy reference report](https://anonymous-report-421.github.io/public-website/?view=1). This supersedes those baseline values in the earlier owner-supplied Table 2; **the confirmed GPT-Policy column remains unchanged**. RoboICL is sourced separately below.

`gpt-as-policy-reference.json` preserves the exact model/task values, scene mixtures and upstream provenance hashes used for the update. The original report data were read directly from the page's embedded JSON. Only relevant result data are retained here.

| Baseline | Reported Score | Reported SR (%) |
| --- | ---: | ---: |
| Galaxea G0.5 | 38.2608 | 30.9464 |
| OpenWAM-α | 32.7322 | 23.0662 |
| π₀.₅ | 24.4300 | 15.6666 |
| GPT 6 Astra (direct) | 37.8125 | 26 |
| π₀.₅ + GPT 6 Astra (hybrid) | 62.6000 | 48 |

For official models, arrange-largest-number, pack-objects-into-box and fold-clothes combine two standard and three randomized scenes, matching the scene mixture in our release. Other tasks use five standard scenes. The report then averages the ten tasks equally.

Direct's reported Score is the mean of its 48 available score records (37.8125), rather than the unweighted mean of its ten task means (38.5). Its success rate is 13/50. We retain the original reported aggregate rather than silently recomputing it. Hybrid uses its 50 score records. GPT-Policy uses ten equally weighted tasks with five scenes per task (80.20 Score; 36/50 successes, 72%).

Other baseline models also use the reference report's exact values. These are published results, not reruns under our evaluation protocol. Only presentation is rounded to two decimals; all calculations retain the source precision.

## RoboICL comparison (2026-10-12)

The [RoboICL ten-task panel](https://mosi-ai.github.io/RoboICL-GPT6-Astra.github.io/#results-and-episodes) reports a one-shot mean score of **60.60**. Its language-classification score reuses the zero-shot result, marked †. This is distinct from the source's 30-task aggregate.

`roboicl-reference.json` records all ten scores and the public rollout evidence for seven reconstructed success-rate estimates, matching our paper Table 2. Estimates are marked ≈ in the heatmap. They use the first five available layouts in ID order (four for Mahjong), whose means match the ten-task scores. The source does not explicitly identify the ten-task panel's rollout subset; these are estimates, not officially reported panel SRs. Number arrangement, clothes folding, packing, and the overall SR remain unavailable (null in data, — on the page). The missing overall SR gets no bar and is not treated as zero.

The CSV distinguishes reported scores, estimated SRs, and unavailable values. The chart uses the existing OpenAI icon to identify RoboICL's GPT-6 Astra backbone, not as a RoboICL organization logo.

## Reference Table 1 implementation

`assets/benchmarks/gpt-as-policy-heatmap.css` copies the actual table, legend and metric-switch declarations from the reference page's inline stylesheet. `robodojo.js` ports the reference's Table 1 DOM structure and its `X9` color function into the site's vanilla JavaScript renderer. The React data-report wrapper is not needed here. `robodojo.css` isolates the copied styles from this site's global table typography and adjusts the width for the added RoboICL column.

Column order is the reference's original seven models, followed by RoboICL and GPT-Policy: DM0.5, GalaxeaVLA (G0.5), Xiaomi-Robotics-1, OpenWAM-α, π₀.₅, Hybrid, Direct, RoboICL (1-shot), **GPT-Policy**. The task order also follows the reference. There are no header logos, highlighted column borders or custom legend treatment. The color scale remains 0–100 for both metrics. The table now spans 840 px to accommodate the added column; small screens scroll within the table, retaining a sticky task column.

`benchmark-data.js` is the single source shared by the bar chart, heatmap and per-task details. `robodojo.js` supports EN/ZH and synchronizes both metric switches. The heatmap has 10 task rows, nine model columns and one overall row.

- `robodojo-tasks.csv`: the confirmed GPT-Policy task data and aggregate.
- `robodojo-comparison.csv`: both metrics and provenance for all displayed methods, with explicitly labeled reported aggregates.
- `paper-real-robot-results.json`: the earlier PDF's real-robot Tables 1 and 3, unchanged by this simulation correction.

No timeout/native-failure fields are displayed. The aggregate bar chart stays beneath Figure 1; the heatmap and per-task details are in the Simulation experiments section. The bar background is white and GPT-Policy's bar is charcoal (#202124). The bar-chart model logos and font subsets were extracted from the reference page on 2026-10-07; ownership remains with their respective owners. MORPHI's logo is the user-provided image.

## Figure refresh

The website's overview, framework and context-compiler images were rendered at 6× from the same PDF, preserving proportions:

- `assets/paper-figures/context-overview.png`: paper Figure 1, page 2, crop [107,65,505,328] points.
- `assets/method-overview.png`: paper Figure 2, page 4, crop [108,80,505,337] points.
- `assets/paper-figures/context-compiler.png`: paper Figure 6, page 15, crop [112,230,500,404] points. It is website Figure 3, replacing the execution-harness image. The source figure is noted in its caption.
