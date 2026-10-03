# Local machine profiles

Machine names are arbitrary: `--machine lab-yam` loads `configs/machines/lab-yam.json`.
Create that ignored local file with `{"extends":"../examples/yam-local.json","machine":"lab-yam"}`, then supply the interfaces, serials, measured intrinsics and camera transforms for your machine.
For ARX, inherit `../examples/arx-local.json`. Robot defaults are in `configs/robots/`.

Alternatively create the ignored `configs/machine.local.json` with `{"extends":"examples/yam-local.json"}` and your overrides; the CLI selects it automatically. Do not move a template to a different directory without adjusting its relative `extends` paths.

`--check` validates structure without hardware. Placeholder matrices/serials are not a physical calibration or connection check.
