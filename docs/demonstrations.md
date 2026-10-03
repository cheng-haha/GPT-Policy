# Demonstration inputs

Pass a reviewed `demo.json`, a recorded run directory, or a supported video with `--demo`. Relative media paths in input JSON are resolved from that JSON file.

```bash
gpt-policy "Repeat the demonstrated task" --demo /path/to/demo
gpt-policy "Repeat the demonstrated task" --demo /path/to/demo --demo-mode video+action
```

The default is `auto`: reviewed JSON keyframes containing state/action data use `video+action`; image-only demonstrations and standalone videos use `video`. Recorded runs are checked for actual state rows, and YAM episodes for state/action topic messages. Task names, filenames, and generic wording such as "video demonstration" do not determine the mode. Invalid sources fail instead of silently dropping data.

The same rule applies to inline demonstration paths and JSON video blocks with omitted `mode` or `"mode": "auto"`. Priority is `--demo-mode` > explicit mode words in the current CLI instruction > each JSON block's mode > auto. Explicit `video` still preserves image-only experiments. Change an old JSON block to `"mode": "auto"`, remove its mode, or pass `--demo-mode auto` to enable detection. Resolved modes are saved in `request.json` and used to filter incompatible saved tasks before task matching.

`video` supplies selected demonstration images and annotations. `video+action` also requires recorded action/state information; a standalone video cannot supply action data. Demonstration views follow the source: a three-camera record can provide three views, while a top-only record supplies one. The runtime does not force all demonstrations to three views.

Freeze a reusable input package before opening hardware:

```bash
gpt-policy "Repeat the demonstrated task" --demo /path/to/demo \
  --prepare-only var/prepared/example
gpt-policy --input-json var/prepared/example/input.json
```

The destination must not already exist. Preparing raw video can call the configured selection model and requires FFmpeg; reviewed inputs can be reused without repeating raw-video selection. Inspect the saved input and media before a physical run. Preparation alone opens no robot hardware.

Demonstration context and the live-image history window are separate. The public profile disables live-image-count refresh by default; see [runtime configuration](runtime.md). Prepared inputs, demonstration media and run recordings remain local under `var/` unless you explicitly publish a separate dataset.
