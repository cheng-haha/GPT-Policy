import pytest


@pytest.fixture(autouse=True)
def isolated_request_archive(tmp_path, monkeypatch):
    from gpt_policy import main
    monkeypatch.setattr(main, "REQUEST_DIRECTORY", tmp_path / "request_json")
    monkeypatch.setattr(main, "select_task_request", lambda run, *_: ("pick-up-fruit", None, run))
