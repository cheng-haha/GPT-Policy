# RoboDojo charts and paper results

## Paper version

The local website was updated from `GPT_Policy_iclr2027__Copy_ (6).pdf`, compiled **2026-10-11 14:51:05 PDT**, 47 pages. It was already present in Downloads and is now the site's `paper.pdf`. The Overleaf project is https://www.overleaf.com/project/6ab3f98275f46d738116b160. The browser project did not respond to content reads during this update, so this is the latest locally available export, not a newly fetched Overleaf revision. The exact SHA-256 is recorded in `benchmark-data.js`.

- Paper Table 2, page 8: simulation progress scores.
- Paper Table 5, page 20: simulation success rates.
- Paper Table 1, page 6: real-robot success, mean decisions/time and sample standard deviations.
- Paper Table 3, page 9: demonstration-information, calibrated-localization and action-chunking ablations.

GPT-Policy's current simulation aggregate is **63.00 / 100**, **26/50 successes (52%)**, with one demonstration per task and five episodes per task. This supersedes the earlier owner-supplied table (80.20 and 72%).

## Additional published references

Source: [GPT 6 Astra as an Embodied Policy](https://anonymous-report-421.github.io/public-website/?view=1), retrieved 2026-10-11. The source's Table 1 includes DM0.5, Galaxea G0.5, Xiaomi R1 and OpenWAM-α (its overall leaderboard top four), π₀.₅, GPT-as-Policy Direct and Hybrid. These are reported values, not local reruns. Extra official-model rows in the aggregate bar chart retain the reference report's ten-task subset values.

### Source precedence and aggregation

- Use paper Tables 2 and 5 wherever present. Add the paper's RoboDawn and both RoboICL conditions.
- DM0.5 and Xiaomi R1 per-task scores, and official-model success rates absent from the paper, come from the reference report. Galaxea and OpenWAM scores come from the paper; their success rates come from the reference report.
- π₀.₅ follows the paper's three-seed means and equal standard/random weighting for number arrangement, folding and packing: reported overall score **24.90**, success **16.0%**. These differ from the reference report's weighting.
- Direct's paper mean is **38.50**, the equal average of task scores, rather than the reference report's episode-pooled 37.8125. Its success rate is **26%**.
- Retain reported aggregate precision. Do not reconstruct a paper aggregate from rounded cells (π₀.₅'s displayed cells round differently).
- RoboICL success rates are unavailable: show a neutral **—**, and omit those methods from the success-rate bars. Never treat missing data as zero.
- RoboICL's one-shot language-classification entry uses its zero-shot score (44), marked †, because the demonstration is unavailable.
- Baselines retain their original evaluation protocols; the visible captions identify source and aggregation differences.

## Files and rendering

`benchmark-data.js` is the data source shared by the horizontal chart, heatmap and per-task detail table. Tasks are joined by stable IDs, not source-table order. `robodojo.js` renders all three, supports EN/ZH, uses a fixed 0–100 color scale, and synchronizes the two compact metric controls. The heatmap has 10 task rows, 11 model/condition columns and one reported-mean row. Darker blue denotes higher values. Text switches between black and white for contrast; missing data has a neutral background. GPT-Policy has an outlined column and MORPHI + OpenAI marks.

- `robodojo-tasks.csv`: GPT-Policy per-task data and aggregate.
- `robodojo-comparison.csv`: both metrics and provenance for all displayed methods.
- `paper-real-robot-results.json`: transcription of paper Tables 1 and 3. Existing demonstration videos are retained; the statistics beside them now reflect Table 1's updated aggregate results, not just the selected video.

No timeout/native-failure fields are displayed.

The section remains directly beneath Figure 1. The bar background stays white and GPT-Policy's bar stays charcoal (#202124). The original model logos and font subsets were extracted from the reference page on 2026-10-07; ownership stays with the respective owners. MORPHI's logo is the original user-provided image. New paper baselines are identified by text where no verified logo was supplied.

## Figure refresh

The website's overview, framework and execution-harness images were rendered at 6× from the same PDF, preserving proportions:

- `assets/paper-figures/context-overview.png`: paper Figure 1, page 2, crop [107,65,505,328] points.
- `assets/method-overview.png`: paper Figure 2, page 4, crop [108,80,505,337] points.
- `assets/paper-figures/execution-harness.png`: paper Figure 7, page 17, crop [112,80,497,246] points. It remains website Figure 3, with the source figure noted in its caption.
