# Machine profiles

The repository includes the deployed `yambox`, `arx247` and `arx248` profiles, including camera serials/device paths, CAN interfaces, camera controls, scene notes and motion settings. Their measured camera intrinsics, distortion coefficients and hand-eye/top-camera transforms are in `configs/calibration/`; shared robot defaults are in `configs/robots/`.

```bash
gpt-policy --machine yambox --check
gpt-policy --machine arx247 --check
gpt-policy --machine arx248 --check
gpt-policy --machine yambox "pick up the red block"
```

`--check` opens no hardware or model session. These measurements describe the named installations; they must match the hardware and camera placement you use. The default `configs/default.json` remains the generic ARX example.

To select a named machine by default, create an ignored `configs/machine.local.json` containing:

```json
{"extends":"machines/yambox.json"}
```

The main yam checkout uses an equivalent symlink to `machines/yambox.json`. Keep host selection in this local file rather than changing the shared default for every machine.

For a new installation, create `configs/machines/lab-yam.json` inheriting `../examples/yam-local.json` and supply measured calibration and hardware settings. Files named `*.local.json` remain ignored; regular machine profiles and calibration files can be committed. Adjust relative `extends` paths when moving a file.

The ARX robot default also references the tracked `request_json/plain_text.json` example task. Explicit task text or `--input-json` overrides it; other generated request archives remain ignored. Passwords, API keys and provider credentials belong outside these hardware profiles.
