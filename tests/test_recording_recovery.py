import io
import json

import pytest
from PIL import Image

from gpt_policy.hardware.camera import CapturedImage
from gpt_policy.recording.mp4 import MjpegMp4Writer
from gpt_policy.recording.recover import recover
from gpt_policy.recording.trace import RunRecorder


@pytest.mark.parametrize("terminal", ["done", "give_up"])
@pytest.mark.parametrize("human,task_status,suffix", [("success", "completed", "success"), ("failed", "failed", "failed"), (None, "unreviewed", "unreviewed")])
@pytest.mark.parametrize("run_status", ["failed", "interrupted"])
def test_finalization_fault_preserves_human_verdict_and_model_conclusion(tmp_path, terminal, human, task_status, suffix, run_status):
    recorder = RunRecorder(tmp_path / "run", {})
    recorder.write("terminal", {"step": 35, "name": terminal, "arguments": {"summary": "finished"}})
    if human:
        recorder.write("human_evaluation", {"outcome": human})
    root = recorder.close(run_status, "home did not finish")
    assert root.name == f"run_{suffix}"
    status = json.loads((root / "status.json").read_text())
    assert status["state"] == run_status
    assert status["task_status"] == task_status
    assert status["outcome"] == suffix
    assert status["model_outcome"] == ("success" if terminal == "done" else "give_up")
    assert status["human_outcome"] == human
    assert status["outcome_source"] == ("human" if human else "unreviewed")
    assert status["error"] == "home did not finish"
    event = json.loads((root / "events.jsonl").read_text().splitlines()[-1])
    assert event["task_status"] == task_status
    assert event["status"] == run_status


def test_outcome_rename_never_overwrites_an_existing_run(tmp_path):
    root = tmp_path / "run"
    existing = tmp_path / "run_success"
    existing.mkdir()
    (existing / "preserve.txt").write_text("previous run")
    recorder = RunRecorder(root, {"task_name": "pick-fruit"})
    recorder.write("human_evaluation", {"outcome": "success"})
    with pytest.raises(FileExistsError, match="already exists"):
        recorder.close("completed")
    assert (existing / "preserve.txt").read_text() == "previous run"
    assert (root / "status.json").exists()
    assert recorder._stream.closed


def test_model_retry_observations_preserve_all_images_and_audit_events(tmp_path):
    recorder = RunRecorder(tmp_path / 'run', {})
    for attempt in range(3):
        if attempt:
            recorder.write('model_retry', {'step':37, 'retry':attempt, 'code':'serverOverloaded'})
        image = CapturedImage('top', f'frame-{attempt}'.encode(), 'image/jpeg', 1, 1, attempt)
        recorder.observation(37, f'observation-{attempt}', {}, [], {'top':image}, attempt=attempt)
    recorder.write('model_retry_exhausted', {'step':37, 'reason':'retry_limit'})
    root = recorder.close('failed', 'at capacity')
    assert (root/'frames/step-00037-top.jpg').read_bytes() == b'frame-0'
    assert (root/'frames/step-00037-retry-01-top.jpg').read_bytes() == b'frame-1'
    assert (root/'frames/step-00037-retry-02-top.jpg').read_bytes() == b'frame-2'
    events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    observations = [e for e in events if e['event']=='observation']
    assert [e.get('attempt',0) for e in observations] == [0,1,2]
    assert len({e['images'][0]['path'] for e in observations}) == 3
    transcript = json.loads((root/'transcript.json').read_text())
    retry_events = [e['content']['event'] for e in transcript if isinstance(e.get('content'),dict)]
    assert retry_events == ['model_retry','model_retry','model_retry_exhausted']


def test_recover_unfinalized_mp4_without_touching_source(tmp_path):
    data = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(data, format="JPEG")
    writer = MjpegMp4Writer(tmp_path / "left.mp4", 16, 16, 10)
    writer.append(data.getvalue())
    row = {"camera": "left", "jpeg_offset": writer.offsets[0], "jpeg_size": writer.sizes[0], "width": 16, "height": 16, "fps": 10}
    (tmp_path / "video-frames.jsonl").write_text(json.dumps(row) + '\n{"partial":')
    writer.stream.close()  # simulate a killed process; no moov written
    original = (tmp_path / "left.mp4").read_bytes()
    assert b"moov" not in original
    output = recover(tmp_path, "left")
    assert b"moov" in output.read_bytes()
    assert b"co64" in output.read_bytes()
    assert (tmp_path / "left.mp4").read_bytes() == original
    with pytest.raises(FileExistsError):
        recover(tmp_path, "left")
