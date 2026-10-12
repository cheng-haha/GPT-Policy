"""Native one-control engineering smoke, never a success-rate episode."""

from pathlib import Path

from .io import write_json
from .runtime import Simulator


def run_smoke(config, evidence, output, gpu):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", evidence)
    result = {
        "complete": False,
        "valid_for_success_rate": False,
        "native_success": None,
        "native_score": None,
        "status": "initialization_error",
    }
    try:
        with Simulator(config, evidence["case"], output / "sim", gpu) as sim:
            obs = sim.observation
            targets = {
                arm: {
                    "pose_xyzquat": obs["arms"][arm]["link6_pose_xyzquat"],
                    "gripper": obs["arms"][arm]["gripper"],
                }
                for arm in ("left", "right")
            }
            receipt = sim.move(
                {
                    "observation_id": obs["observation_id"],
                    "note": "Engineering hold at measured pose.",
                    "waypoints": [dict(**targets, steps=1)],
                }
            )
            write_json(
                output / "smoke.json",
                {
                    "views": list(receipt["observation"]["images"]),
                    "acknowledged_controls": receipt["acknowledged_controls"],
                    "native_step": receipt["end_step"],
                    "passed": receipt["acknowledged_controls"] == 1
                    and receipt["end_step"] == 1,
                },
            )
            result = sim.finish("engineering_smoke")
    except BaseException as error:  # noqa: BLE001 - retain failed engineering attempt
        result.update(
            status="interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "runtime_error",
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        write_json(output / "result.json", result)
    return result
