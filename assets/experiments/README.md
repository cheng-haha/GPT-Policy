# Additional task demos

Three representative successful runs were added from the [experiment record](https://morphicrobot.feishu.cn/docx/XELDdCILeoYJJFxXHdPc1svgnbe), revision 1516. Selected videos are updated when needed to match their documented reference inputs. Human-video tasks show the human demonstration alongside representative robot runs with and without human-video context; robot-demonstration tasks compare no video, video, and video + action; the remaining tasks retain one robot demo each.

| Task | Condition | Selected run | Decisions | Full task time |
| --- | --- | --- | --- | --- |
| Notebook pickup | Human video | Trial 1 | 42 | 684.2 s |
| Bottle opening | Robot video + action | Trial 1 | 62 | 907.522 s |
| Plug removal and reinsertion | Robot video + action | Trial 1 | 46 | 611 s |

The displayed success rates and mean costs summarize all three trials in each condition. Each clip is a single representative run. Videos use the top-camera recording at 20×, encoded as fast-start H.264 MP4 at 24 fps. Playback duration differs from full task elapsed time.

The bottle and plug process summaries were derived from model_decision notes retrieved through SSH from yam, using the run directories specified in the experiment record. They are summaries of the recorded agent sequence. Their timeline markers were aligned using the source top-camera frame timestamps (10 fps before acceleration); the frame counts match the downloaded source durations exactly. Existing task summaries retain their original clips and timing.

Notebook comparison adds its original real-time human demonstration and the first no-demonstration trial (90 decisions, 1316.3 s, gave up). The existing towel demonstration and no-demonstration clip are reused.

See manifest.json for source hashes, source and encoded durations, and file sizes.

## Robot-demonstration comparisons

| Task | Condition | Selected trial | Outcome |
| --- | --- | --- | --- |
| Bottle opening | None | 3 | Budget exhausted |
| Bottle opening | Robot video | 2 | Success |
| Bottle opening | Robot video + action | 1 | Success |
| Plug reinsertion | None | 1 | Gave up |
| Plug reinsertion | Robot video | 1 | Gave up |
| Plug reinsertion | Robot video + action | 1 | Success |

Each condition shows one illustrative trial with its trial number and outcome, while the success fraction and means cover all three trials. The bottle no-video example includes a full 100-decision attempt; the video-only example illustrates one of its two successful trials. Existing action-aligned runs and their matched summaries are unchanged. These selected examples do not replace the aggregate statistics.

## Goal-image alignment

The goal-image examples use blocks trial 3 and fruit trial 1 from the experiment record (revision 1516), paired with the target references shown in paper Figure 6. All three trials for each task were compared visually: blocks trial 3 has a straighter top row and more closely aligned stem in its final frame; fruit trial 1 keeps all four objects visible without gripper occlusion. These are qualitative selections for clear presentation. The blocks reference is the original portrait image; its top row is green, green, blue, with yellow and orange down the stem. This replaces the older video whose color arrangement differed. The fruit reference is unchanged and matches the source image byte for byte; its video now shows the complete first recorded trial, including the unobstructed final arrangement.

| Task | Selected trial | Decisions | Full task time | Playback |
| --- | --- | --- | --- | --- |
| Arrange blocks into a T | 3 | 66 | 13 min 47 s | 20× |
| Arrange four fruits | 1 | 33 | 9 min 10 s | 20× |

The website keeps the aggregate success rate and mean costs across all three trials. Each displayed clip is labeled with its selected trial number. Target images retain their original aspect ratios and open at full size when clicked.
