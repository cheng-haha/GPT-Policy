# RoboDojo calibration validation

## Scope and outcome

The repaired bridge passes **1,434 numerical checks** in live Isaac Sim
Push-T scenes: 1,110 for dual-arm X5 and 324 for Franka. Each profile is
tested with seeds 0, 1, and 2 and five joint configurations per seed.
There are 60 camera captures, 300 independently rendered marker detections,
and nine small IK-driven TCP moves. No VLM is called. Diagnostic markers
and simulator object coordinates are never supplied to a policy.

| Maximum measured error | X5 | Franka |
| --- | ---: | ---: |
| Rendered marker reprojection | 0.389 px | 0.134 px |
| Rendered temporal triangulation, parallax >= 5 degrees | 3.056 mm | Not applicable: fixed head camera only |
| Executed TCP translation after IK and simulation settling | 3.494 mm | 0.154 mm |
| Executed TCP rotation | 0.007382 rad | 0.000295 rad |

All 300 rendered markers are detected. The geometric pose round trips are
below 7e-16 in maximum matrix-element error. Those algebraic checks are
supplementary: the independently rendered pixels, not a forward/inverse
round trip alone, establish image-to-geometry agreement.

The standalone SDK projection check also passes, with maximum ray error
7.55e-8 m. Calibration, adapter, and tool-catalog unit tests pass (26 tests
and 11 subtests). The full repository suite reports 81 passed and two
unrelated configuration-path failures: `tests/test_config.py` resolves
`configs` one directory above the repository in two existing tests.

## Corrections verified

- Convert USD camera axes (+X right, +Y up, -Z forward) into optical axes
  (+X right, +Y down, +Z forward) before pixel back-projection.
- Read actual imaging-sensor transforms from live Fabric, not stale USD
  poses or camera-housing poses. A USD-only check missed the discrepancy
  after moving the arms; rendered markers exposed it.
- Match standard pinhole vertical aperture to the render aspect ratio.
  The head camera's original configured vertical aperture disagreed with
  the square-pixel projection actually used by tiled rendering. Read
  intrinsics from the now-consistent active sensor.
- Read live robot base transforms and environment origins. Normalize
  quaternion computations in float64 instead of relying on rounded
  configuration quaternions or float32 rotation matrices.
- Apply source-link-to-grasp-center TCP transformations in both directions,
  including execution feedback. X5 uses 145 mm along link6 +X, reoriented
  to TCP +Z; Franka uses 102 mm along panda_hand +Z. These are configured
  grasp centers, not the source-link origin or the exact distal mesh edge.
- Give the model labeled source matrices, already-converted optical/base
  matrices, and explicit pixel/reference-frame semantics. Historical
  demonstration EE poses are labeled as source-link poses, not live TCPs.

No runtime parallax, residual, workspace, or stopping threshold was tightened
or relaxed to obtain these results.

## Reproduction and artifacts

Run `scripts/install_robodojo_policy.sh` to apply the versioned simulator
patch, then restart existing simulator/policy processes. With the simulator
environment prepared, run:

```bash
.venv/bin/python -m pytest -q tests/test_robodojo_calibration.py tests/test_robodojo_adapter.py tests/test_tools_catalog.py
/root/miniconda3/envs/RoboDojo/bin/python scripts/verify_robodojo_camera_calibration.py
/root/miniconda3/envs/RoboDojo/bin/python scripts/verify_robodojo_scene_calibration.py --env-cfg gpt_policy_x5 --seeds 0,1,2 --output var/runs/robodojo/calibration_scene_x5_verified
/root/miniconda3/envs/RoboDojo/bin/python scripts/verify_robodojo_scene_calibration.py --env-cfg gpt_policy_franka --seeds 0,1,2 --output var/runs/robodojo/calibration_scene_franka_verified
```

Each scene run saves `report.json` and RGB PNGs. The report includes every
check, tolerance, camera matrix, landmark coordinate, projected pixel, and
detected pixel. Weak-baseline triangulations are recorded separately, not
counted as policy-usable depth estimates. The offline fixture additionally
replays the original failed Push-T wrist observations without needing Isaac
Sim or private run recordings.

## Interpretation limits

These measurements validate the tested single-environment X5 and Franka
profiles, sensors, transforms, resets, and sampled joint configurations.
They do not establish zero error for arbitrary poses or camera models,
physical-hardware calibration, collision safety, or Push-T task success.
Rasterization, pixel annotation, and limited parallax still bound practical
depth accuracy. Franka's current fixed-head-only profile cannot obtain a
temporal camera-motion baseline merely by moving its arm.
