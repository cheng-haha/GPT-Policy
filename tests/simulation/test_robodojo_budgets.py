"""Extended horizon preserves native predicates and bounded actions."""

from types import SimpleNamespace

import pytest

from gpt_policy.simulation.robodojo.budget import EpisodeBudget, budget_config
from gpt_policy.simulation.robodojo.config import simulation_agent_config
from gpt_policy.simulation.robodojo.evaluation import outcome
from gpt_policy.simulation.robodojo.tools import agent_context


def test_extended_horizon_changes_only_control_budget(monkeypatch):
    clock = [20.0]
    monkeypatch.setattr("gpt_policy.simulation.robodojo.budget.time.monotonic", lambda: clock[0])
    checks = object()
    env = SimpleNamespace(step_lim=1100, end_flag=[False], success=[True],
                          reward_manager=checks, unstable_envs=set())
    budget = EpisodeBudget(env, {"control_step_limit": None, "episode_wall_time_s": 9000})
    assert env.step_lim == 10**9 and budget.original_limit == 1100
    assert env.reward_manager is checks and env.end_flag == [False]
    assert not budget.expired() and budget.remaining_wall_time() == 9000
    result = outcome(env, 1500, "wall_time_budget")
    assert result["status"] == "budget_censored"
    assert result["native_success"] is None and not result["valid_for_success_rate"]
    assert not result["evaluation_budget"]["standard_native_horizon"]
    clock[0] += 9000
    assert budget.expired() and budget.remaining_wall_time() == 0
    env.end_flag[0] = True
    assert outcome(env, 1500, "native_terminal")["native_success"] is True


def test_default_horizon_and_unlimited_prompt_keep_action_and_success_contract():
    env = SimpleNamespace(step_lim=1100)
    budget = EpisodeBudget(env, {})
    assert env.step_lim == 1100 and not budget.expired()
    native = agent_context()
    extended = agent_context(extended_controls=True)
    assert native.tools == extended.tools
    assert native.output_schema == extended.output_schema
    assert "remaining_controls is null" in extended.instructions
    assert "have no fixed count limit" in extended.instructions
    assert "both robot arms back near their initial pose" in extended.instructions


@pytest.mark.parametrize("values", [
    {"control_step_limit": None},
    {"control_step_limit": 1100},
    {"episode_wall_time_s": 0},
    {"episode_wall_time_s": True},
    {"episode_wall_time_s": float("nan")},
    {"episode_wall_time_s": float("inf")},
])
def test_invalid_budget_fails_before_resources(values):
    with pytest.raises(ValueError):
        budget_config(values)


def test_simulation_profile_inherits_disabled_window_and_rejects_override():
    config = {"model": "test", "effort": "medium", "codex_bin": "test"}
    assert simulation_agent_config(config).live_image_window is None
    with pytest.raises(ValueError, match="disabled"):
        simulation_agent_config({**config, "live_image_window": 8})


@pytest.mark.parametrize("stop", ["wall", "invalid"])
def test_native_move_stops_between_acknowledged_controls(tmp_path, monkeypatch, stop):
    import numpy as np

    from gpt_policy.simulation.robodojo.environment import NativeSession

    clock = [0.0]
    monkeypatch.setattr("gpt_policy.simulation.robodojo.budget.time.monotonic", lambda: clock[0])
    env = SimpleNamespace(step_lim=1100, end_flag=[False], take_action_cnt=[0], unstable_envs=set())
    budget = EpisodeBudget(env, {"control_step_limit": None, "episode_wall_time_s": 2})
    pose = [0, 0, 1, 0, 0, 0, 1]
    session = object.__new__(NativeSession)
    session.env, session.budget, session.output = env, budget, tmp_path
    session.finished, session.step_id = False, 0
    session.early_arrival, session.last_gripper_command = False, None
    session.pass_through = False
    session.execution = {"ik_backend": "bounded_dls", "chunk_feedback": False}
    session.obs = {"observation_id": "e:0", "step_id": 0,
                   "arms": {a: {"link6_pose_xyzquat": pose, "gripper": 1}
                            for a in ("left", "right")}}
    session.kinematics = SimpleNamespace(
        target=lambda _: {"action": np.zeros(14), "diagnostics": {}})
    def take_action(command):
        env.take_action_cnt[0] += 1
        clock[0] += 1
        if stop == "invalid" and env.take_action_cnt[0] == 2:
            env.unstable_envs.add(0)
    env.take_action = take_action
    def observe():
        session.obs.update(step_id=session.step_id, observation_id=f"e:{session.step_id}")
    session.observe = observe
    reasons = []
    session.finish = reasons.append
    args = {"observation_id": "e:0", "note": "hold",
            "waypoints": [dict(steps=5, **{a: {"pose_xyzquat": pose, "gripper": 1}
                                         for a in ("left", "right")})]}
    receipt = session.move(args)
    assert receipt["acknowledged_controls"] == env.take_action_cnt[0] == 2
    assert receipt["wall_time_budget_exhausted"]
    assert reasons == (["invalid_native_layout"] if stop == "invalid" else [])
    assert env.end_flag == [False]
