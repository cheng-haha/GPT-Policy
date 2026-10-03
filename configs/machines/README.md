# Machine profiles

The repository includes the deployed `yambox`, `arx247` and `arx248` profiles, including camera serials/device paths, CAN interfaces, camera controls, scene notes and motion settings. Their measured camera intrinsics, distortion coefficients and hand-eye/top-camera transforms are in `configs/calibration/`; shared robot defaults are in `configs/robots/`.

```bash
gpt-policy --machine yambox --check
gpt-policy --machine arx247 --check
gpt-policy --machine arx248 --check
gpt-policy --machine yambox "pick up the red block"
```

`--check` opens no hardware or model session. These measurements describe the named installations; they must match the hardware and camera placement you use. The default `configs/default.json` inherits `machines/yambox.json`, selecting the deployed YAM configuration and Codex agent.

To override the default on another host, create an ignored `configs/machine.local.json`, for example:

```json
{"extends":"machines/arx247.json"}
```

The main yam checkout selects `machines/yambox.json` through its local machine selector. Existing local selectors and configuration environment variables still override the public default. Use `--machine arx247` or `--machine arx248` for a single run, or `--config` for a custom profile.

For a new installation, create `configs/machines/lab-yam.json` inheriting `../examples/yam-local.json` and supply measured calibration and hardware settings. Files named `*.local.json` remain ignored; regular machine profiles and calibration files can be committed. Adjust relative `extends` paths when moving a file.

The ARX robot default also references the tracked `request_json/plain_text.json` example task. Explicit task text or `--input-json` overrides it; other generated request archives remain ignored. Passwords, API keys and provider credentials belong outside these hardware profiles.
