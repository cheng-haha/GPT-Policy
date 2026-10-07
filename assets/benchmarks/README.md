# RoboDojo opening chart

## GPT-Policy

The ten task rows in `robodojo-tasks.csv` were transcribed from the evaluation-table screenshot supplied by the project owner on 2026-10-07. The chart calculates the aggregate from the task rows: mean score **85.90 / 100**, success **39/50 (78%)**. Each task has five trials. Scores and success rates are distinct metrics. The displayed table and CSV include task names, scores, and success counts/rates.

The screenshot does not specify seeds, scene mixture, control budgets, or a context condition. The page does not assign such settings or claim matched evaluation against the published references.

## Published references

Source: [GPT 6 Astra as an Embodied Policy](https://anonymous-report-421.github.io/public-website/?view=1), inspected 2026-10-07. The opening RoboDojo mean-score chart and the Results success-rate chart report the following values for their selected ten-task subset:

| Method | Mean score | Success rate (%) |
| --- | ---: | ---: |
| π₀.₅ + GPT 6 Astra | 62.60 | 48 |
| Galaxea G0.5 | 38.26 | 30.95 |
| GPT 6 Astra (direct) | 37.81 | 26 |
| Xiaomi R1 | 33.97 | 25.51 |
| OpenWAM-α | 32.73 | 23.07 |
| DM0.5 | 31.82 | 22.04 |
| Meituan R0 | 28.33 | 17.88 |
| Spatial Forcing | 24.96 | 17.15 |
| π₀.₅ | 24.43 | 15.67 |
| StarVLA-PI | 22.36 | 16.17 |
| InternVLA A1.5 | 21.74 | 15.03 |
| Hy-VLA 0.5 | 21.16 | 13.94 |

The source recomputes official-model aggregates for its selected task subset and scene mixture. These are not full-benchmark leaderboard values, and are not reruns of the GPT-Policy evaluation. The visible chart caption keeps this distinction explicit.

The chart is implemented locally in `robodojo.js` and `robodojo.css`, following the reference's horizontal bars, blue panel, highlighted method, and numeric labels. The existing site's language control switches both the chart and task table between English and Chinese. Both chart metrics use a fixed 0–100 axis and sort by the selected metric.

The MORPHI logo is the original 540×540 PNG supplied by the project owner. It is displayed with its aspect ratio preserved in the GPT-Policy row.


## Reference-style revision

The chart sits directly after Figure 1 and its caption. Its full-width light-blue panel, 4 px blue top border, right-aligned heading, 38.667 px rows, 20 px bars, bottom axis, and model-icon column follow the reference opening chart. The compact options menu retains mean-score/success-rate switching.

The RoboDojo and model logos under `logos/`, and the chart font subsets under `fonts/`, are original embedded assets from the reference page, retrieved on 2026-10-07. Logo correspondence follows that page: Physical Intelligence plus OpenAI for the hybrid, Galaxea, OpenAI for direct GPT, Xiaomi, OpenWAM, DM, Meituan, Spatial Forcing, Physical Intelligence, StarVLA, InternVLA, and Hunyuan for Hy-VLA. They identify the reported models; ownership remains with their respective owners. The MORPHI logo comes from the project owner, as noted above.
