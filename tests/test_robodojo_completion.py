import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gpt_policy.robodojo.deploy import _completion_feedback, eval_one_episode
from gpt_policy.robodojo.model import GPTPolicyModel


class _Reward:
    def is_qpos_close(self, **kwargs):
        return ("is_qpos_close", kwargs)

    def is_AB_xy_distance_within_threshold(self, **kwargs):
        return ("is_AB_xy_distance_within_threshold", kwargs)

    def all_robot_back_to_origin(self):
        return ("all_robot_back_to_origin", {})

    def check_once(self, check, env_idx):
        assert env_idx == 0
        return check[0] != "all_robot_back_to_origin"


class RoboDojoCompletionTest(unittest.TestCase):
    def test_rejected_done_reports_only_unmet_home_condition(self):
        env = SimpleNamespace(task_name="push_T", reward_manager=_Reward(), success=[True])
        feedback = _completion_feedback(env)
        self.assertEqual(feedback["checks"], {
            "t_orientation_within_7_deg": True,
            "t_xy_distance_under_0_007_m": True,
            "both_arms_at_origin": False,
        })

        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model._terminal_reason = "terminal.done"
        model._terminal_kind = "terminal.done"
        model._home_tcp_targets = {"left": [0.1, 0.2, 0.3, 0, 0, 0, 1]}
        model.step = 3
        model.latest_frame = {"env_step": 4}
        events = []
        model._trace_event = lambda name, payload: events.append((name, payload))
        model.clear_terminal_decision(feedback)
        result = json.loads(model.previous)["result"]
        self.assertFalse(result["accepted"])
        self.assertIn("both_arms_at_origin", result["reason"])
        self.assertNotIn("t_xy_distance_under_0_007_m", result["reason"])
        self.assertEqual(result["completion_feedback"], feedback)
        self.assertEqual(result["home_tcp_targets"], model._home_tcp_targets)
        self.assertEqual(events[-1][1]["completion_feedback"], feedback)

    def test_push_t_prompt_includes_home_target_and_success_rules(self):
        model = GPTPolicyModel.__new__(GPTPolicyModel)
        model.model_cfg = {}
        model.arms = ("left", "right")
        model.robot_model = "X5"
        model.settings = {"runtime": {"task_name": "different_task"}}
        model.latest_frame = {"task_name": "push_T"}
        model.catalog = object()
        model._home_tcp_targets = {"left": [0.1, 0.2, 0.3, 0, 0, 0, 1]}
        contexts = []
        model.agent = SimpleNamespace(start=contexts.append)
        with patch("gpt_policy.robodojo.model.instructions", return_value="base"), \
             patch("gpt_policy.robodojo.model.output_schema", return_value={}), \
             patch("gpt_policy.robodojo.model.tool_schemas", return_value=[]):
            model._start("Push the T")
        prompt = contexts[0].instructions
        self.assertIn("BOTH arms must return", prompt)
        self.assertIn("0.007 m", prompt)
        self.assertIn("[0.1, 0.2, 0.3, 0, 0, 0, 1]", prompt)

        model.latest_frame = {"task_name": "other_task"}
        with patch("gpt_policy.robodojo.model.instructions", return_value="base"), \
             patch("gpt_policy.robodojo.model.output_schema", return_value={}), \
             patch("gpt_policy.robodojo.model.tool_schemas", return_value=[]):
            model._start("Other task")
        self.assertEqual(contexts[-1].instructions, "base")

    def test_deploy_forwards_reward_feedback_on_done_rejection(self):
        class Env:
            task_name = "push_T"
            reward_manager = _Reward()
            success = [True]
            end_flag = [False]
            take_action_cnt = [0]
            ended = False

            def is_episode_end(self):
                return self.ended

            def get_running_env_idx_list(self):
                return [0]

            def get_obs(self):
                return {"state": {}}

        env = Env()
        calls = []

        class Client:
            def call(self, func_name, obs=None):
                calls.append((func_name, obs))
                if func_name == "is_episode_done":
                    return any(name == "get_action" for name, _ in calls)
                if func_name in {"is_terminal_give_up", "is_execution_blocked"}:
                    return False
                if func_name == "get_action":
                    return [{"dummy": True}]
                if func_name == "clear_terminal_decision":
                    env.ended = True

        with patch("gpt_policy.robodojo.deploy._finish_policy_exit", return_value=False):
            eval_one_episode(env, Client())
        feedback = next(obs for name, obs in calls if name == "clear_terminal_decision")
        self.assertFalse(feedback["checks"]["both_arms_at_origin"])
        self.assertTrue(feedback["checks"]["t_xy_distance_under_0_007_m"])
        self.assertEqual(next(obs for name, obs in calls if name == "update_obs")["task_name"], "push_T")

    def test_repeated_unconfirmed_done_cannot_loop_forever(self):
        class Env:
            task_name = "organize_table"
            success = [True]
            end_flag = [False]
            take_action_cnt = [0]

            def is_episode_end(self):
                return self.end_flag[0]

            def get_running_env_idx_list(self):
                return [] if self.end_flag[0] else [0]

            def get_obs(self):
                return {}

            def get_obs_batch(self, **kwargs):
                return []

        class Client:
            def __init__(self):
                self.cleared = 0

            def call(self, func_name, **kwargs):
                if func_name == "is_episode_done":
                    return True
                if func_name in {"is_terminal_give_up", "is_execution_blocked"}:
                    return False
                if func_name == "clear_terminal_decision":
                    self.cleared += 1

        env, client = Env(), Client()
        eval_one_episode(env, client)
        self.assertFalse(env.success[0])
        self.assertTrue(env.end_flag[0])
        self.assertEqual(client.cleared, 2)


if __name__ == "__main__":
    unittest.main()
