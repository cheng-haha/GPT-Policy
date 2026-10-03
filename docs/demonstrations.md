# Demonstration inputs

Pass a reviewed `demo.json`, a recorded run directory, or a supported video with `--demo`. Relative media paths in input JSON are resolved from that JSON file.

```bash
gpt-policy "Repeat the demonstrated task" --demo /path/to/demo --demo-mode video
gpt-policy "Repeat the demonstrated task" --demo /path/to/demo --demo-mode video+action
```

`video` supplies selected demonstration images and annotations. `video+action` also requires recorded action/state information; a standalone video cannot supply action data. Demonstration views follow the source: a three-camera record can provide three views, while a top-only record supplies one. The runtime does not force all demonstrations to three views.

Freeze a reusable input package before opening hardware:

```bash
gpt-policy "Repeat the demonstrated task" --demo /path/to/demo \
  --demo-mode video --prepare-only var/prepared/example
gpt-policy --input-json var/prepared/example/input.json
```

The destination must not already exist. Preparing raw video can call the configured selection model and requires FFmpeg; reviewed inputs can be reused without repeating raw-video selection. Inspect the saved input and media before a physical run. Preparation alone opens no robot hardware.

Demonstration context and the live-image history window are separate. The public profile disables live-image-count refresh by default; see [runtime configuration](runtime.md). Prepared inputs, demonstration media and run recordings remain local under `var/` unless you explicitly publish a separate dataset.
