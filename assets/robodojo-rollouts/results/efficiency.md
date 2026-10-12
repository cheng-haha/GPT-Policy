# Execution and token accounting

Categories follow [GPT-as-Policy §6.3](https://anonymous-report-421.github.io/public-website/?view=1), accessed 2026-10-11.
Reference columns reproduce their published figures; they were not rerun here.

| Metric | GPT-Policy, selected 50 | Reference hybrid | Reference direct |
|---|---:|---:|---:|
| Success rate | 72% | 48% | 26% |
| Actual control steps | 39,692 | 42,750 | 38,221 |
| Executed action segments | 5,340 | 3,776 | 7,729 |
| Mean physical duration (s) | 31.75 | 34.20 | 30.58 |
| Total recorded tokens (cached input included) | 855,443,119* | 624,762,828 | 1,132,343,772 |
| Cached input tokens | 815,661,184* | 607,555,840 | 1,107,349,760 |
| Uncached input tokens | 36,906,672* | 15,901,963 | 22,907,448 |
| Output tokens | 2,875,263* | 1,305,025 | 2,086,564 |

**\* Recorded lower bound, not a complete token bill:** 42
failed calls across 10 selected episodes did not return
usage. All 6,162 completed calls did. Unknown usage is
retained as null in per-call evidence, not fabricated or estimated. Reasoning tokens
are already part of output and are not counted twice. These figures exclude
earlier unselected attempts and demonstration preparation/development compute.

An executed segment is one motion chunk that acknowledged at least one control,
including partially executed chunks. There were 444
zero-control motion calls and 378 other tool decisions.
All 6,162 tool decisions and 17,645 executed
waypoints are available separately. A chunk is not a single control or waypoint.

Physical duration is control_steps × 0.04 seconds; it excludes model waits.
The first video frame is the initial observation, so video frame counts are
controls + 1. Neither token counts nor physical duration represent account charges
or wall-clock completion time.

Our medium-effort ICL/chunk harness differs from the reference's xhigh direct/hybrid
setups and their selected scene/retry histories. This descriptive table does not
isolate a causal token or success improvement.

- [Per-episode CSV](efficiency_episodes.csv)
- [Per-task CSV](efficiency_tasks.csv)
- [Aggregate JSON](efficiency.json)
