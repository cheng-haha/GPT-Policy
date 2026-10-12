"""Native registration and outcome eligibility. No model success overrides."""

from .io import require


def register_native_evaluation(env):
    env.run_reward()
    if hasattr(env, "get_score"):
        env.get_score()
    names = ("check_list", "final_check_list", "trigger_check_list")
    counts = {
        name: [len(group) for group in getattr(env.reward_manager, name)]
        for name in names
    }
    require(
        all(len(counts[name]) == env.num_envs for name in names),
        "Native condition group count mismatch",
    )
    require(
        all(sum(counts[name][i] for name in names) > 0 for i in range(env.num_envs)),
        "Native task completion conditions are empty",
    )
    support = []
    if getattr(env, "interact", False) and hasattr(env, "query_support_arm_traj"):
        support = list(env.get_running_env_idx_list())
        for index in support:
            env.query_support_arm_traj(env_idx=index)
    return {
        "registered": True,
        "condition_group_counts": counts,
        "support_envs": support,
    }


def native_stop_reason(env):
    """Honor the official invalid-scene flag without calling it a native failure."""
    if 0 in getattr(env, "unstable_envs", set()):
        return "invalid_native_layout"
    if bool(env.end_flag[0]):
        return "native_terminal"
    return None


def native_process_score_100(env, poisoned=False):
    """Read official process score without advancing reward checks or physics.

    env.success starts True before completion, so it cannot alone imply 100.
    This host-only diagnostic also survives a wall-time/API interruption.
    """
    if poisoned or 0 in getattr(env, "unstable_envs", set()):
        return None
    if bool(env.end_flag[0]) and bool(env.success[0]):
        return 100.0
    return (
        float(env.reward_manager.get_score()[0])
        if hasattr(env, "get_score")
        else 0.0
    )


def outcome(env, steps, reason, poisoned=False):
    complete = bool(env.end_flag[0])
    invalid = 0 in getattr(env, "unstable_envs", set())
    eligible = complete and not invalid and not poisoned
    success = bool(env.success[0]) if eligible else None
    process_score = native_process_score_100(env, poisoned)
    score = process_score / 100 if eligible else None
    return {
        "complete": complete,
        "valid_for_success_rate": eligible,
        "native_success": success,
        "native_score": score,
        # Diagnostic only when incomplete; does not grant native outcome eligibility.
        "native_process_score_100": process_score,
        "reason": reason,
        "status": (
            "invalid_native_layout"
            if invalid
            else "native_completed"
            if eligible
            else "budget_censored"
            if reason in ("decision_budget", "wall_time_budget")
            else "incomplete"
        ),
        "native_control_steps": steps,
        "native_step_limit": int(env.step_lim),
        "evaluation_budget": getattr(env, "_gpt_policy_budget", {
            "protocol": "native_controls", "standard_native_horizon": True,
            "original_native_step_limit": int(env.step_lim),
        }),
    }
