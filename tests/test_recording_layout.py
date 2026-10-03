from pathlib import Path

import pytest

from gpt_policy.input import ImagePart, RunInput, TextPart, VideoPart
from gpt_policy.recording.layout import icl_directory, recording_directory, task_category
from gpt_policy.recording.trace import RunRecorder


@pytest.mark.parametrize('mode,expected', [(None, 'none_icl'), ('video', 'video_icl'), ('video+action', 'video_action_icl')])
def test_explicit_input_overrides_action_wording_and_source_name(mode, expected):
    parts = (VideoPart(Path('demo-video-actions.json'), mode=mode),) if mode else ()
    run = RunInput('Use robot states and action trajectories. In the current scene, unscrew the bottle cap.', 'test', parts)
    assert icl_directory(run) == expected
    assert task_category(run, 'input') == '拧瓶盖'


@pytest.mark.parametrize('mode,expected', [('video', 'video_icl'), ('video+action', 'video_action_icl')])
def test_portable_prepared_input_uses_mode_marker(mode, expected):
    run = RunInput('insert the plug', 'test', (TextPart(f'Historical demonstration. Input mode: {mode}.'), ImagePart(Path('frame.jpg'))))
    assert icl_directory(run) == expected
    assert icl_directory(RunInput('place blocks like this', 'test', (ImagePart(Path('target.jpg')),))) == 'none_icl'


@pytest.mark.parametrize('goal,expected', [
    ('Use bottle-cap-demo.json as reference. In the current scene, remove the plug and insert it back into the same socket.', '拔出并重新插回插头'),
    ('Use robot state/action data. In the current scene, insert the loose plug into the power strip.', '插入插排'),
    ('拔出插头，再重新插回插排', '拔出并重新插回插头'),
    ('拧开瓶盖', '拧瓶盖'),
    ('把所有笔放到笔筒里', '把所有笔放入笔筒'),
    ('拔掉胶棒的盖帽', '拔下固体胶盖子'),
])
def test_current_task_is_independent_of_demo_title(goal, expected):
    assert task_category(RunInput(goal, 'test'), 'input') == expected


def test_unknown_task_and_override_are_safe_and_stable():
    run = RunInput('a new task', 'test')
    assert task_category(run, 'new-task-deadbeef') == 'new-task'
    assert task_category(run, 'input').startswith('未分类任务-')
    assert task_category(run, 'input') != task_category(RunInput('another task', 'test'), 'input')
    assert task_category(run, 'input', '自定义任务') == '自定义任务'
    with pytest.raises(ValueError):
        task_category(run, 'input', '../escape')


@pytest.mark.parametrize('status,human', [('completed', 'success'), ('give_up', 'failed'), ('interrupted', None)])
def test_finalized_recording_stays_in_classified_parent(tmp_path, status, human):
    root = tmp_path / 'gpt'
    path = recording_directory(root, '拧瓶盖', 'video_icl', '20260924-trial')
    assert {p.name for p in path.parent.parent.iterdir()} == {'none_icl', 'video_icl', 'video_action_icl'}
    recorder = RunRecorder(path, {'task_category': '拧瓶盖', 'icl_type': 'video_icl'})
    if human:
        recorder.write("human_evaluation", {"outcome": human})
    final = recorder.close(status)
    assert final.parent == root / '拧瓶盖' / 'video_icl'
    assert (final / 'status.json').is_file()
    assert not path.exists()


@pytest.mark.parametrize('mode,explicit', [('video', False), ('video+action', False), (None, False), ('video', True)])
def test_cli_routes_before_hardware_and_honors_explicit_record_dir(tmp_path, monkeypatch, mode, explicit):
    import json
    from unittest.mock import Mock
    from gpt_policy import main as app
    from gpt_policy.harness.config import AgentConfig

    monkeypatch.chdir(tmp_path)
    demo = tmp_path / 'demo.json'
    demo.write_text('{}')
    content = ['unscrew and remove the bottle cap']
    if mode:
        content.append({'video': str(demo), 'mode': mode})
    request = tmp_path / 'input.json'
    request.write_text(json.dumps({'instruction': content[0], 'content': content}))
    runtime = {'input_json': str(request)}
    if explicit:
        runtime['record_dir'] = str(tmp_path / 'custom')
    monkeypatch.setattr(app, 'load_settings', lambda *_: {'runtime': runtime})
    monkeypatch.setattr(app, 'agent_config', lambda *_: AgentConfig(model='test'))
    monkeypatch.setattr(app, 'preflight_agent', lambda *_: None)
    recorder = Mock(side_effect=InterruptedError('stop before hardware'))
    cameras = Mock()
    monkeypatch.setattr(app, 'RunRecorder', recorder)
    monkeypatch.setattr(app, 'CameraSet', cameras)
    monkeypatch.setattr('sys.argv', ['gpt-policy', '--input-json', str(request)])
    with pytest.raises(InterruptedError, match='stop before hardware'):
        app.main()
    path, metadata = recorder.call_args.args
    condition = {None: 'none_icl', 'video': 'video_icl', 'video+action': 'video_action_icl'}[mode]
    assert metadata['task_category'] == '拧瓶盖'
    assert metadata['icl_type'] == condition
    if explicit:
        assert path == tmp_path / 'custom'
        assert not (tmp_path / 'var/runs').exists()
    else:
        assert path.parent == Path('var/runs/gpt/拧瓶盖') / condition
    cameras.assert_not_called()


def test_cli_naming_failure_logs_do_not_clutter_root(tmp_path, monkeypatch):
    from gpt_policy import main as app
    from gpt_policy.harness.config import AgentConfig
    from gpt_policy.harness.usage import model_call

    def fail(*_):
        with model_call('codex', 'test') as call:
            call.update(request_started=True)
            raise RuntimeError('naming failed')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, 'load_settings', lambda *_: {})
    monkeypatch.setattr(app, 'agent_config', lambda *_: AgentConfig(model='test'))
    monkeypatch.setattr(app, 'preflight_agent', lambda *_: None)
    monkeypatch.setattr(app, 'select_task_request', fail)
    monkeypatch.setattr('sys.argv', ['gpt-policy', 'new task'])
    with pytest.raises(RuntimeError, match='naming failed'):
        app.main()
    assert len(list((tmp_path / 'var/runs/gpt/_initialization_failed').glob('*/usage.json'))) == 1
    assert {p.name for p in (tmp_path / 'var/runs/gpt').iterdir()} == {'_initialization_failed'}
