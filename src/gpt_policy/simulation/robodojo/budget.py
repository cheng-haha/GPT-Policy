"""Explicit episode budgets; never replace native success predicates."""

import math
import time

UNBOUNDED_CONTROLS = 10**9


class EpisodeWallTimeExceeded(TimeoutError):
    pass


def budget_config(config):
    control = config.get("control_step_limit", "native")
    if control is not None and control != "native":
        raise ValueError("control_step_limit must be 'native' or null")
    wall = config.get("episode_wall_time_s")
    if wall is not None and (
        isinstance(wall, bool) or not isinstance(wall, (int, float))
        or not math.isfinite(wall) or wall <= 0
    ):
        raise ValueError("episode_wall_time_s must be null or a finite positive number")
    if control is None and wall is None:
        raise ValueError("Unlimited controls require episode_wall_time_s")
    return {"control_step_limit": control, "episode_wall_time_s": wall}


class EpisodeBudget:
    def __init__(self, env, config):
        self.config = budget_config(config)
        self.original_limit = int(env.step_lim)
        self.extended = self.config["control_step_limit"] is None
        if self.extended:
            env.step_lim = UNBOUNDED_CONTROLS
        wall = self.config["episode_wall_time_s"]
        self.started = time.monotonic()
        self.deadline = None if wall is None else self.started + wall
        self.evidence = {
            **self.config,
            "protocol": "extended_controls" if self.extended else "native_controls",
            "standard_native_horizon": not self.extended,
            "original_native_step_limit": self.original_limit,
            "effective_native_step_limit": int(env.step_lim),
            "started_at_s": time.time(),
            "deadline_monotonic": self.deadline,
            "success_predicates": "unchanged_native",
        }
        env._gpt_policy_budget = self.evidence

    def expired(self):
        return self.deadline is not None and time.monotonic() >= self.deadline

    def remaining_wall_time(self):
        return None if self.deadline is None else max(0.0, self.deadline - time.monotonic())
