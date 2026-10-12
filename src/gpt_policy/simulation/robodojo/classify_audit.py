"""Post-episode diagnostics only; never used for actions or native scoring."""

import logging

from .io import write_json


def classify_terminal_conditions(env):
    """Read the native classify predicates after the episode has ended.

    The containment groups allow any category in each basket. The native task
    separately pairs cat0/1/2 with basket0/1/2 for its height checks. Preserve
    that distinction rather than silently assigning a category to a basket.
    Calls below are read-only native predicates; no check queue is advanced.
    """
    if env.num_envs != 1 or not bool(env.end_flag[0]):
        raise ValueError("Condition audit requires one ended native environment")
    parser = env.reward_manager.func_parser
    categories = [parser.get_label_by_prefix(f"cat{i}")[0] for i in range(3)]
    if not all(categories):
        raise ValueError("Missing native category labels")
    baskets = [f"basket{i}" for i in range(3)]
    containment = []
    for basket in baskets:
        groups = []
        for index, labels in enumerate(categories):
            objects = [
                {
                    "label": label,
                    "passed": bool(
                        parser.is_A_in_B(
                            {"env_idx": 0, "label_A": label, "label_B": basket}
                        )
                    ),
                }
                for label in labels
            ]
            groups.append(
                {
                    "category": f"cat{index}",
                    "objects": objects,
                    "passed": all(row["passed"] for row in objects),
                }
            )
        containment.append(
            {
                "basket": basket,
                "categories": groups,
                "passed": any(group["passed"] for group in groups),
            }
        )
    height = []
    for index, (labels, basket) in enumerate(zip(categories, baskets)):
        objects = [
            {
                "label": label,
                "passed": bool(
                    parser.is_A_z_lower_than_B_bbox_zmax(
                        {
                            "env_idx": 0,
                            "label_A": label,
                            "label_B": basket,
                            "z_threshold": 0.01,
                        }
                    )
                ),
            }
            for label in labels
        ]
        height.append(
            {
                "category": f"cat{index}",
                "reference_basket": basket,
                "z_threshold_m": 0.01,
                "objects": objects,
                "passed": all(row["passed"] for row in objects),
            }
        )
    home = bool(
        parser.all_robot_back_to_origin(
            {"env_idx": 0, "pos_threshold": 0.15, "rot_threshold": 20}
        )
    )
    return {
        "task": "classify_objects",
        "native_control_steps": int(env.take_action_cnt[0]),
        "scope": "Post-terminal read-only predicates; diagnostic only, not a replacement outcome",
        "containment": containment,
        "height": height,
        "robot_home": {
            "passed": home,
            "per_axis_position_threshold_m": 0.15,
            "angle_threshold_degrees": 20,
        },
    }


def write_classify_terminal_audit(env, output):
    """Best effort: a diagnostic failure must not affect the episode result."""
    path = output / "terminal_conditions.json"
    try:
        if path.exists():
            return
        try:
            data = classify_terminal_conditions(env)
        except Exception as error:  # noqa: BLE001 - preserve terminal evidence across native/provider failures
            data = {
                "task": "classify_objects",
                "audit_available": False,
                "error_type": type(error).__name__,
            }
        write_json(path, data)
    except Exception as error:  # noqa: BLE001 - preserve terminal evidence across native/provider failures
        logging.getLogger(__name__).warning(
            "Could not write terminal condition audit: %s", type(error).__name__
        )
