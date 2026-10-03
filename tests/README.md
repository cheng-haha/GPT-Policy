# Core offline regression tests

The public repository keeps nine test modules focused on the supported runtime configuration:

- `test_config.py`, `test_public_profiles.py`, `test_settings_runtime.py`: agent/runtime configuration, deployed machine profiles and their input dependencies.
- `harness/test_live_window_cli.py`, `harness/test_codex_context.py`: live-window defaults/overrides, recorded settings, history refresh and overload recovery.
- `test_d405_controls.py`, `test_realsense_controls.py`, `test_camera_warmup.py`: manual exposure/gain, readback and cleanup when camera startup fails.
- `test_recording_layout.py`: task/ICL directories, explicit recording locations and initialization-failure logs.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

These tests use simulated devices/providers and temporary files. They do not move robots, open cameras or call model providers. Hardware-specific, historical and extended evaluation/protocol test suites remain in the main development project and are excluded from this public test directory. Keep helpers self-contained when maintaining this subset.
