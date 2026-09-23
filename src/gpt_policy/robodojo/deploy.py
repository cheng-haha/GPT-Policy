"""RoboDojo deployment loop with explicit environment termination."""


def _completion_feedback(env):
    """Read push_T's current reward predicates without changing reward state."""
    if getattr(env, "task_name", None) != "push_T":
        return {"task": getattr(env, "task_name", None), "checks": None}
    reward = env.reward_manager
    checks = {
        "t_orientation_within_7_deg": reward.is_qpos_close(
            label_A="t", label_B="target_t", dis_threshold=7
        ),
        "t_xy_distance_under_0_007_m": reward.is_AB_xy_distance_within_threshold(
            label_A="t", label_B="target_t", threshold=0.007
        ),
        "both_arms_at_origin": reward.all_robot_back_to_origin(),
    }
    return {
        "task": "push_T",
        "checks": {name: bool(reward.check_once(check, 0)) for name, check in checks.items()},
        "episode_valid": bool(env.success[0]),
    }


def _finish_policy_exit(env, force_failure: bool = False):
    # success is initialized to True by RoboDojo while an episode is running.
    # Only is_episode_end() can turn a reward into confirmed task completion.
    env.is_episode_end()
    running = env.get_running_env_idx_list()
    if running and force_failure:
        for index in running:
            env.success[index] = False
            env.end_flag[index] = True
        env.get_obs_batch(env_idx_list=running, last_frame=True)
        return True
    return not running


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        # Keep the two counters explicit: GPT-Policy counts model decisions,
        # while RoboDojo counts submitted environment actions.  A chunked
        # policy action may expand into multiple environment actions, so the
        # policy must receive the simulator counter instead of guessing it.
        obs = TASK_ENV.get_obs()
        obs["task_name"] = TASK_ENV.task_name
        obs["env_step"] = int(TASK_ENV.take_action_cnt[0])
        obs["success"] = bool(TASK_ENV.success[0])
        obs["end_flag"] = bool(TASK_ENV.end_flag[0])
        model_client.call(func_name="update_obs", obs=obs)
        if model_client.call(func_name="is_episode_done"):
            # ``done`` is only a request from the policy.  RoboDojo remains
            # authoritative: keep the episode alive when its reward checker
            # has not confirmed success yet.
            give_up = model_client.call(func_name="is_terminal_give_up")
            blocked = model_client.call(func_name="is_execution_blocked")
            confirmed = _finish_policy_exit(TASK_ENV, force_failure=give_up or blocked)
            if not confirmed and not give_up and not blocked:
                model_client.call(func_name="clear_terminal_decision", obs=_completion_feedback(TASK_ENV))
                continue
            break
        actions = model_client.call(func_name="get_action")
        if model_client.call(func_name="is_episode_done"):
            give_up = model_client.call(func_name="is_terminal_give_up")
            blocked = model_client.call(func_name="is_execution_blocked")
            confirmed = _finish_policy_exit(TASK_ENV, force_failure=give_up or blocked)
            if not confirmed and not give_up and not blocked:
                model_client.call(func_name="clear_terminal_decision", obs=_completion_feedback(TASK_ENV))
                continue
            break
        for action in actions:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError("GPT-Policy RoboDojo adapter requires eval_batch=false")
