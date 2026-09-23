import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from gpt_policy.robodojo.model import GPTPolicyModel


class RoboDojoTraceNamingTest(unittest.TestCase):
    def test_launcher_task_names_trace_before_first_observation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "calibration.json"
            calibration.write_text("{}", encoding="utf-8")
            config = root / "config.json"
            config.write_text("{}", encoding="utf-8")
            cfg = {
                "gpt_policy_config": str(config),
                "calibration_manifest": str(calibration),
                "trace_dir": str(root / "traces"),
                "icl_enabled": False,
            }
            with patch.dict("os.environ", {"ROBODOJO_TASK_NAME": "imitate_sorting_sequence"}), \
                 patch("gpt_policy.robodojo.model.load_settings", return_value={"runtime": {"task_name": "push_T"}}), \
                 patch("gpt_policy.robodojo.model.RoboDojoCalibration.from_manifest"), \
                 patch("gpt_policy.robodojo.model.RoboDojoAdapter"), \
                 patch("gpt_policy.robodojo.model.load_tool_catalog"), \
                 patch("gpt_policy.robodojo.model.agent_config", return_value=SimpleNamespace(type="codex", model="gpt-6-astra")), \
                 patch("gpt_policy.robodojo.model.create_agent"):
                model = GPTPolicyModel(cfg)
            try:
                self.assertIn("-imitate_sorting_sequence-pid", model.trace_dir.name)
                event = json.loads((model.trace_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(event["task_name"], "imitate_sorting_sequence")
                self.assertIsNone(model._workspace_context()["tcp_bounds_m"]["z"][0])
            finally:
                model.trace_events.close()


if __name__ == "__main__":
    unittest.main()
