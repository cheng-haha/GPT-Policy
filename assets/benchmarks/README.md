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

The displayed G0.5, OpenWAM-α, π₀.₅, GPT-as-Policy Direct and Hybrid results also follow this confirmed Table 2. In particular, the reported means are G0.5 **38.78 / 31.33%**, OpenWAM-α **33.57 / 23.80%**, π₀.₅ **24.90 / 16%**, Direct **38.50 / 26%**, and Hybrid **62.60 / 48%**. Keep reported mean precision rather than recomputing it from rounded cells.

## Additional published references

Source: [GPT 6 Astra as an Embodied Policy](https://anonymous-report-421.github.io/public-website/?view=1), retrieved 2026-10-11. DM0.5, Xiaomi R1 and the additional official models in the aggregate bar chart retain the report's ten-task subset results. These are published results, not local reruns; each baseline retains its evaluation protocol.

## Reference Table 1 implementation

`assets/benchmarks/gpt-as-policy-heatmap.css` copies the actual table, legend and metric-switch declarations from the reference page's inline stylesheet. `robodojo.js` ports the reference's Table 1 DOM structure and its `X9` color function into the site's vanilla JavaScript renderer. The React data-report wrapper is not needed here. `robodojo.css` isolates the copied styles from this site's global table typography and constrains the component to the reference's 760 px width.

Column order is the reference's original seven models, followed by GPT-Policy: DM0.5, GalaxeaVLA (G0.5), Xiaomi-Robotics-1, OpenWAM-α, π₀.₅, Hybrid, Direct, **GPT-Policy**. The task order also follows the reference. No additional model families, header logos, highlighted column borders or custom legend treatment are added. The color scale remains 0–100 for both metrics. Small screens scroll within the table, retaining a sticky task column.

`benchmark-data.js` is the single source shared by the bar chart, heatmap and per-task details. `robodojo.js` supports EN/ZH and synchronizes both metric switches. The heatmap has 10 task rows, eight model columns and one mean row.

- `robodojo-tasks.csv`: the confirmed GPT-Policy task data and aggregate.
- `robodojo-comparison.csv`: both metrics and provenance for all displayed methods.
- `paper-real-robot-results.json`: the earlier PDF's real-robot Tables 1 and 3, unchanged by this simulation correction.

No timeout/native-failure fields are displayed. The section stays beneath Figure 1. The bar background is white and GPT-Policy's bar is charcoal (#202124). The bar-chart model logos and font subsets were extracted from the reference page on 2026-10-07; ownership remains with their respective owners. MORPHI's logo is the user-provided image.

## Figure refresh

The website's overview, framework and execution-harness images were rendered at 6× from the same PDF, preserving proportions:

- `assets/paper-figures/context-overview.png`: paper Figure 1, page 2, crop [107,65,505,328] points.
- `assets/method-overview.png`: paper Figure 2, page 4, crop [108,80,505,337] points.
- `assets/paper-figures/execution-harness.png`: paper Figure 7, page 17, crop [112,80,497,246] points. It remains website Figure 3, with the source figure noted in its caption.
