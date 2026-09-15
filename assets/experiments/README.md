# Additional task demos

Three representative successful runs were added from the [experiment record](https://morphicrobot.feishu.cn/docx/XELDdCILeoYJJFxXHdPc1svgnbe), revision 1516. Existing task videos remain unchanged. Human-video tasks show the human demonstration alongside representative robot runs with and without human-video context; other tasks retain one robot demo each.

| Task | Condition | Selected run | Decisions | Full task time |
| --- | --- | --- | --- | --- |
| Notebook pickup | Human video | Trial 1 | 42 | 684.2 s |
| Bottle opening | Robot video + action | Trial 1 | 62 | 907.522 s |
| Plug removal and reinsertion | Robot video + action | Trial 1 | 46 | 611 s |

The displayed success rates and mean costs summarize all three trials in each condition. Each clip is a single representative run. Videos use the top-camera recording at 20×, encoded as fast-start H.264 MP4 at 24 fps. Playback duration differs from full task elapsed time.

The bottle and plug process summaries were derived from model_decision notes retrieved through SSH from yam, using the run directories specified in the experiment record. They are summaries of the recorded agent sequence. Their timeline markers were aligned using the source top-camera frame timestamps (10 fps before acceleration); the frame counts match the downloaded source durations exactly. Existing task summaries retain their original clips and timing.

Notebook comparison adds its original real-time human demonstration and the first no-demonstration trial (90 decisions, 1316.3 s, gave up). The existing towel demonstration and no-demonstration clip are reused.

See manifest.json for source hashes, source and encoded durations, and file sizes.
