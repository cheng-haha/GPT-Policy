"""Model overload recovery must never replay a physical action."""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from gpt_policy.hardware.motion_control import MotionFault
from gpt_policy.harness.codex import CodexAppServer
from gpt_policy.harness.config import AgentConfig
from gpt_policy.harness.errors import AgentOverloadedError, AgentTimeoutError, AgentUsageLimitError
from gpt_policy.harness.models import AgentContext
from gpt_policy.harness.providers.codex import CodexSession
from gpt_policy.harness.protocol import instructions, observation
from gpt_policy.harness.usage import collect_usage
from gpt_policy.input import TextPart
from gpt_policy.input.request import RunInput
from gpt_policy.runtime.runner import run_loop
from gpt_policy.settings import runtime_config


MOVE = {'name':'move_to', 'arguments':{'note':'move once'}}
DONE = {'name':'done', 'arguments':{'summary':'done'}}


def overloaded():
    return AgentOverloadedError('Selected model is at capacity.', provider='codex', code='serverOverloaded')


@pytest.fixture
def rig(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr('gpt_policy.runtime.runner.time.monotonic', lambda: clock[0])
    monkeypatch.setattr('gpt_policy.runtime.runner.random.uniform', lambda *args: 0)

    def advance(seconds):
        clock[0] += seconds

    wait = Mock(side_effect=advance)
    monkeypatch.setattr('gpt_policy.runtime.runner.wait_with_health', wait)
    robot, cameras, video, agent, executor, recorder, display = [Mock() for _ in range(7)]
    robot.state.side_effect = lambda: {'sample_time':clock[0]}
    cameras.describe.return_value = []
    video.snapshot.side_effect = lambda **kw: {'top':object()}
    executor.execute.return_value = {'completed_motion':True}
    executor.is_terminal.side_effect = lambda name: name == 'done'
    robot.return_home.return_value = {'home':{'settle':{'settled':True}}}
    display.waiting.side_effect = lambda *args: nullcontext()
    observe = Mock(side_effect=lambda task,state,cameras,previous,step: json.dumps({'state':state,'previous':previous,'step':step}))
    runtime = runtime_config({'runtime':{'max_decisions':3}})
    run_input = RunInput('task', 'model', (TextPart('demonstration'),))

    def run(**kwargs):
        return run_loop(runtime, run_input, robot, cameras, video, agent, executor,
                        recorder, observe, display=display, **kwargs)

    return SimpleNamespace(clock=clock, advance=advance, wait=wait, robot=robot, cameras=cameras,
        video=video, agent=agent, executor=executor, recorder=recorder, observe=observe,
        display=display, run=run, run_input=run_input)


def events(rig, kind):
    return [c.args[1] for c in rig.recorder.write.call_args_list if c.args[0]==kind]


def test_recovery_recaptures_same_step_without_repeating_previous_motion(rig):
    rig.agent.decide.side_effect = [MOVE, overloaded(), MOVE, DONE]
    assert rig.run() == 'completed'
    assert rig.agent.decide.call_count == 4
    assert rig.executor.execute.call_count == 2
    assert rig.video.snapshot.call_count == 4
    assert [c.args[-1] for c in rig.observe.call_args_list] == [0,1,1,2]
    turns = [c.args[0] for c in rig.agent.decide.call_args_list]
    assert turns[1].images is not turns[2].images
    assert json.loads(turns[2].observation)['state']['sample_time'] == 2
    assert json.loads(turns[1].observation)['previous'] == json.loads(turns[2].observation)['previous']
    assert turns[0].content is rig.run_input.content
    assert all(t.content is None for t in turns[1:])
    assert rig.recorder.observation.call_args_list[2].kwargs == {'attempt':1}
    assert [e['step'] for e in events(rig,'model_decision')] == [0,1,2]
    assert [e['step'] for e in events(rig,'execution_result')] == [0,1]
    retry = events(rig,'model_retry')[0]
    assert retry['step']==1 and retry['retry']==1 and retry['code']=='serverOverloaded'
    rig.robot.return_home.assert_called_once()


def test_initial_turn_retry_preserves_demonstration_and_budget(rig):
    rig.agent.decide.side_effect = [overloaded(), DONE]
    assert rig.run() == 'completed'
    assert [c.args[-1] for c in rig.observe.call_args_list] == [0,0]
    assert all(c.args[0].content is rig.run_input.content for c in rig.agent.decide.call_args_list)
    rig.executor.execute.assert_not_called()


def test_retry_limit_does_not_execute_or_home(rig):
    error = overloaded()
    rig.agent.decide.side_effect = error
    with pytest.raises(AgentOverloadedError) as caught:
        rig.run()
    assert caught.value is error
    assert rig.agent.decide.call_count == 21
    assert [c.args[0] for c in rig.wait.call_args_list] == [2,4,8] + [8] * 17
    assert len(events(rig,'model_retry')) == 20
    assert events(rig,'model_retry_exhausted')[0]['reason'] == 'retry_limit'
    assert not events(rig,'model_decision')
    rig.executor.execute.assert_not_called()
    rig.robot.return_home.assert_not_called()


@pytest.mark.parametrize('late_answer', [True,False])
def test_total_recovery_window_bounds_silent_or_late_provider(rig, late_answer):
    calls = []

    class SilentQueue:
        def get(self, timeout=None):
            rig.advance(timeout)
            raise queue.Empty

    native = object.__new__(CodexAppServer)
    native._events = SilentQueue()

    def decide(turn):
        calls.append(turn)
        if len(calls)==1:
            raise overloaded()
        if late_answer:
            rig.advance(301)
            return DONE
        return native._receive()

    rig.agent.decide.side_effect = decide
    with pytest.raises(AgentTimeoutError):
        rig.run()
    assert len(calls)==2
    assert rig.clock[0] <= (303 if late_answer else 300.11)
    assert events(rig,'model_retry_exhausted')[0]['reason'] == 'recovery_deadline'
    assert not events(rig,'model_decision')
    rig.executor.execute.assert_not_called()
    rig.robot.return_home.assert_not_called()


def test_each_decision_has_an_independent_recovery_budget(rig):
    rig.agent.decide.side_effect = [overloaded(), MOVE, overloaded(), DONE]
    assert rig.run() == 'completed'
    assert [(e['step'],e['retry']) for e in events(rig,'model_retry')] == [(0,1),(1,1)]
    assert [c.args[0] for c in rig.wait.call_args_list] == [2,2]
    rig.executor.execute.assert_called_once()


@pytest.mark.parametrize('error', [RuntimeError('connection lost'), KeyboardInterrupt(),
                                  AgentUsageLimitError('Quota exhausted', provider='codex', code='usageLimitExceeded'),
                                  MotionFault({'reason':'control_thread_stopped'})])
def test_other_failures_are_not_retried(rig,error):
    rig.agent.decide.side_effect = error
    with pytest.raises(type(error)) as caught:
        rig.run()
    assert caught.value is error
    rig.agent.decide.assert_called_once()
    rig.wait.assert_not_called()
    rig.executor.execute.assert_not_called()


def test_health_fault_stops_recovery_before_another_model_call(rig):
    rig.agent.decide.side_effect = [overloaded(), DONE]
    fault = MotionFault({'reason':'motor_overtemperature'})

    def health():
        if rig.agent.decide.call_count:
            raise fault

    with pytest.raises(MotionFault) as caught:
        rig.run(health_check=health)
    assert caught.value is fault
    rig.agent.decide.assert_called_once()
    rig.wait.assert_not_called()
    rig.executor.execute.assert_not_called()


@pytest.mark.parametrize('bimanual', [False,True])
def test_retry_polls_real_state_even_if_recorder_health_check_is_noop(rig,bimanual):
    temperature = {'over_limit':{'temp_mos':[{'motor_index':1,'value_c':85}]}, 'limit_c':80}

    def state():
        arm = {'temperature_over_limit':temperature if rig.clock[0]>=2 else None}
        return {'arms':{'left':{},'right':arm}} if bimanual else arm

    rig.robot.state.side_effect = state
    rig.agent.decide.side_effect = [overloaded(), DONE]
    with pytest.raises(MotionFault) as caught:
        rig.run(health_check=lambda: None)
    assert caught.value.details['reason']=='motor_overtemperature'
    rig.agent.decide.assert_called_once()
    rig.executor.execute.assert_not_called()


def test_stale_or_stopped_driver_aborts_recovery(rig):
    def state():
        if rig.clock[0]>=2:
            raise RuntimeError('I2RT feedback has not updated for over one second')
        return {}

    rig.robot.state.side_effect = state
    rig.agent.decide.side_effect = [overloaded(), DONE]
    with pytest.raises(RuntimeError, match='not updated'):
        rig.run()
    rig.agent.decide.assert_called_once()
    rig.executor.execute.assert_not_called()


@pytest.mark.parametrize('overload_turn', [1, 2])
def test_native_overload_to_runner_recovery_and_usage_are_end_to_end(rig, overload_turn):
    raw_state = {
        'joint_positions_rad': [0] * 6, 'joint_velocities_rad_s': [0] * 6,
        'joint_torques_nm': [0] * 6, 'tcp_xyzrpy': [.2, 0, .1, 0, 0, 0],
        'gripper_position_m': .04, 'gripper_velocity_m_s': 0, 'gripper_torque_nm': .2,
        'temperature_mos_c': [42.375] * 7, 'temperature_rotor_c': [53.875] * 7,
        'temperature_limit_c': 80.0, 'temperature_over_limit': None,
    }
    rig.robot.state.side_effect = lambda: deepcopy(raw_state)
    rig.executor.execute.return_value = deepcopy(raw_state)  # Full result, not compact motion feedback.
    rig.observe.side_effect = observation
    rig.video.snapshot.side_effect = lambda **kwargs: {}
    native = object.__new__(CodexAppServer)
    native.model, native.effort, native.thread_id = 'gpt-test', 'medium', None
    native.convert_camera_images_to_jpeg, native.camera_jpeg_quality = False,85
    native.project_root = Path('/robot')
    native._events = queue.Queue()
    counts = {'thread':0,'turn':0}

    def request(method, params):
        if method=='thread/start':
            counts['thread'] += 1
            return {'thread':{'id':f'thread-{counts["thread"]}'}}
        if method=='thread/unsubscribe':
            return {}
        assert method=='turn/start'
        counts['turn'] += 1
        turn_id = f'turn-{counts["turn"]}'
        failed = counts['turn']==overload_turn
        wire = MOVE if counts['turn']<=2 else DONE
        native._events.put({'method':'item/completed','params':{'threadId':native.thread_id,
            'turnId':turn_id,'item':{'type':'agentMessage','text':json.dumps(wire)}}})
        native._events.put({'method':'turn/completed','params':{'threadId':native.thread_id,
            'turn':{'id':turn_id,'status':'failed' if failed else 'completed',
                    'error':{'codexErrorInfo':'serverOverloaded','message':'at capacity'} if failed else None}}})
        return {'turn':{'id':turn_id}}

    native._request = Mock(side_effect=request)
    with patch('gpt_policy.harness.providers.codex.CodexAppServer', return_value=native):
        session = CodexSession(AgentConfig(),'gpt-test',False,85)
    session.start(AgentContext(instructions('YAM', 'can0', 6, settings={'backend':'yam'}),[],{}))
    rig.agent.decide.side_effect = session.decide
    with collect_usage() as ledger:
        assert rig.run() == 'completed'
    assert counts == {'thread':2,'turn':3}
    assert len(ledger.calls)==3 and ledger.summary()['failed_calls']==1
    assert [c['thread_id'] for c in ledger.calls] == [
        'thread-1', 'thread-2' if overload_turn==1 else 'thread-1', 'thread-2',
    ]
    rig.executor.execute.assert_called_once()
    assert len(session._history)==2
    assert len(events(rig,'model_retry'))==1
    # Inspect the actual RPC input, including rebuilt history, not just a helper's output.
    requests = [c.args[1] for c in native._request.call_args_list if c.args[0] in {'thread/start','turn/start'}]
    for request in requests:
        text = json.dumps(request)
        for term in ('temperature_', '42.375', '53.875', 'serverOverloaded', 'model_retry',
                     'delay_s', 'recovery_timeout_s'):
            assert term not in text
    retry_request = [c.args[1] for c in native._request.call_args_list if c.args[0]=='turn/start'][overload_turn]
    assert 'HISTORICAL EXECUTION RECORD' in json.dumps(retry_request['input'])
    assert 'demonstration' in json.dumps(retry_request['input'])
    if overload_turn==2:
        assert 'previous_result' in json.dumps(retry_request['input'])
    for call in rig.recorder.observation.call_args_list:
        assert call.args[2]['temperature_rotor_c'] == [53.875] * 7
    assert events(rig,'execution_result')[0]['result'] == raw_state
