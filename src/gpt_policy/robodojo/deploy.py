"""RoboDojo deployment loop with explicit environment termination."""


def _finish_policy_exit(env, force_failure=False):
    # success is initialized to True by RoboDojo while an episode is running.
    # Only is_episode_end() can turn a reward into confirmed task completion.
    env.is_episode_end()
    running = env.get_running_env_idx_list()
    for index in running:
        if force_failure or not env.end_flag[index]:
            env.success[index] = False
        env.end_flag[index] = True
    if running:
        env.get_obs_batch(env_idx_list=running, last_frame=True)


def _execute_native_actions(env, actions):
    """Execute one model decision and settle only after its final waypoint."""
    if not actions:
        return None
    cfg = env.eval_cfg.get("action_execution", {})
    wait_until_settled = bool(cfg.get("wait_until_settled", False))
    settle_after_trajectory = bool(cfg.get("settle_after_trajectory", True))
    executed_final = False
    try:
        for index, action in enumerate(actions):
            final = index == len(actions) - 1
            if wait_until_settled and settle_after_trajectory:
                cfg["wait_until_settled"] = final
            env.take_action(action)
            executed_final = final
            if env.is_episode_end():
                break
    finally:
        if wait_until_settled and settle_after_trajectory:
            cfg["wait_until_settled"] = True

    if not wait_until_settled or not executed_final:
        return None
    feedback = getattr(env, "_last_ik_feedback", [None])[0]
    execution = feedback.get("execution") if isinstance(feedback, dict) else None
    if isinstance(execution, dict):
        if execution.get("settled") is not True:
            print(f"[action_execution] final target still reported unsettled after wait: {execution}")
        return None
    return {
        "accepted": False,
        "executed": True,
        "reason": "missing_settle_feedback",
        "execution": execution,
        "explanation": (
            "The final trajectory target did not provide settle feedback; "
            "no post-action observation was sent to the policy."
        ),
    }


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        if model_client.call(func_name="is_episode_done"):
            _finish_policy_exit(TASK_ENV)
            break
        actions = model_client.call(func_name="get_action")
        if model_client.call(func_name="is_episode_done"):
            _finish_policy_exit(TASK_ENV)
            break
        failure = _execute_native_actions(TASK_ENV, actions)
        if failure is not None:
            TASK_ENV._gpt_policy_control_feedback = failure
            print(f"[action_execution] {failure['reason']}: {failure['execution']}")
            _finish_policy_exit(TASK_ENV, force_failure=True)


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError("GPT-Policy RoboDojo adapter requires eval_batch=false")
