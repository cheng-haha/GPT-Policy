"""Official score units and censored-score preservation, without new success claims."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gpt_policy.simulation.robodojo.evaluation import native_process_score_100, outcome


def scored_env(*, complete=False, success=True, score=50, invalid=False):
    return SimpleNamespace(
        end_flag=[complete], success=[success], step_lim=1000,
        unstable_envs={0} if invalid else set(),
        get_score=Mock(side_effect=AssertionError("must not re-register scoring")),
        reward_manager=SimpleNamespace(
            get_score=Mock(return_value=[score]),
            get_reward=Mock(side_effect=AssertionError("must not advance predicates")),
        ),
    )


def test_initial_success_flag_is_not_full_credit():
    env = scored_env(score=25)
    assert native_process_score_100(env) == 25
    env.get_score.assert_not_called()
    env.reward_manager.get_reward.assert_not_called()


@pytest.mark.parametrize("reason", ["wall_time_budget", "runtime_error", "decision_budget"])
def test_interrupted_score_is_saved_without_promoting_native_outcome(reason):
    env = scored_env(score=75)
    result = outcome(env, 12, reason)
    assert result["native_process_score_100"] == 75
    assert result["native_score"] is None and result["native_success"] is None
    assert not result["valid_for_success_rate"] and not result["complete"]
    assert env.end_flag == [False] and env.success == [True]


@pytest.mark.parametrize("success,partial,expected", [(True, 30, 100), (False, 30, 30)])
def test_completed_outcome_preserves_normalized_compatibility(success, partial, expected):
    env = scored_env(complete=True, success=success, score=partial)
    result = outcome(env, 12, "native_terminal")
    assert result["native_score"] == expected / 100
    assert result["native_process_score_100"] == expected
    assert result["valid_for_success_rate"]


@pytest.mark.parametrize("invalid,poisoned", [(True, False), (False, True)])
def test_invalid_or_uncertain_score_is_not_publishable(invalid, poisoned):
    env = scored_env(complete=True, invalid=invalid)
    assert native_process_score_100(env, poisoned) is None
    env.reward_manager.get_score.assert_not_called()


def test_no_task_score_function_means_binary_scoring():
    env = scored_env()
    del env.get_score
    assert native_process_score_100(env) == 0
    env.end_flag = [True]
    assert native_process_score_100(env) == 100
    env.reward_manager.get_score.assert_not_called()


def test_native_invalid_scene_stops_without_becoming_policy_failure():
    from gpt_policy.simulation.robodojo.evaluation import native_stop_reason

    env = scored_env(invalid=True)
    assert native_stop_reason(env) == "invalid_native_layout"
    result = outcome(env, 12, native_stop_reason(env))
    assert result["status"] == "invalid_native_layout"
    assert result["complete"] is False
    assert result["native_success"] is None
    assert result["native_process_score_100"] is None
    env.unstable_envs.clear()
    assert native_stop_reason(env) is None
    env.end_flag[0] = True
    assert native_stop_reason(env) == "native_terminal"
