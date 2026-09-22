"""RoboDojo deployment loop with explicit environment termination."""


def _finish_policy_exit(env):
    # success is initialized to True by RoboDojo while an episode is running.
    # Only is_episode_end() can turn a reward into confirmed task completion.
    env.is_episode_end()
    running = env.get_running_env_idx_list()
    for index in running:
        env.success[index] = False
        env.end_flag[index] = True
    if running:
        env.get_obs_batch(env_idx_list=running, last_frame=True)


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
        for action in actions:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError("GPT-Policy RoboDojo adapter requires eval_batch=false")
