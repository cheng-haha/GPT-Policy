"""Shutdown races with a blocked send; native SDK coverage opens no devices."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from gpt_policy.hardware.yam import YamRobot
from gpt_policy.hardware.motion_control import MotionControl


@pytest.mark.parametrize("native", [False, True], ids=["fake", "pinned-sdk"])
def test_close_drains_can_sender_before_closing_socket(native):
    if native:
        ChainBase = pytest.importorskip("i2rt.motor_drivers.dm_driver").DMChainCanInterface
        DriverBase = pytest.importorskip("i2rt.robots.motor_chain_robot").MotorChainRobot
    else:
        class ChainBase:
            def start_thread(self):
                threading.Thread(target=self._set_torques_and_update_state).start()

            def _set_torques_and_update_state(self):
                with self._rate_recorder:
                    while self.running:
                        self._set_commands(self.commands)

            def close(self):
                self.running = False
                self.motor_interface.close()

        class DriverBase:
            def close(self):
                self._stop_event.set()
                self._server_thread.join()
                self.motor_chain.close()

    entered, release, stopped, bus_closed = [threading.Event() for _ in range(4)]
    trace = []

    class RateRecorder:
        def __enter__(self): return self
        def track(self): pass
        def __exit__(self, *_): trace.append("sender_exited")

    class Bus:
        def send(self):
            entered.set()
            assert release.wait(5), "test failed to release blocked CAN send"
            if bus_closed.is_set():
                raise ValueError("file descriptor cannot be a negative integer (-1)")
            trace.append("batch_sent")

        def close(self):
            trace.append("bus_closed")
            bus_closed.set()

        def motor_off(self, motor_id):
            assert not bus_closed.is_set()
            assert "sender_exited" in trace and "producer_exited" in trace
            trace.append(f"motor_off_{motor_id}")

    class Chain(ChainBase):
        @property
        def running(self): return self._running

        @running.setter
        def running(self, value):
            self._running = value
            if not value:
                stopped.set()

        def _set_commands(self, _):
            self.motor_interface.send()
            return []

    # Bypass constructors: only the real start/loop/close methods run, with a
    # fake transport. No motor factory, socket, calibration or movement.
    chain = object.__new__(Chain)
    chain.motor_interface = Bus()
    chain.motor_list = [(i, "fake") for i in range(1, 8)]
    chain.running = True
    chain.start_thread_flag = False
    chain.state = []
    chain.commands = []
    chain.command_lock = threading.Lock()
    chain.state_lock = threading.Lock()
    chain._rate_recorder = RateRecorder()
    chain._report_interval = 30
    chain.enable_auto_recovery = False
    chain.same_bus_device_driver = None
    chain._update_absolute_positions = lambda _: None
    chain.start_thread()
    assert entered.wait(2)
    sender = next(t for t in threading.enumerate()
                  if getattr(getattr(t, "_target", None), "__self__", None) is chain)

    driver = object.__new__(DriverBase)
    driver.motor_chain = chain
    driver._stop_event = threading.Event()

    def produce():
        driver._stop_event.wait()
        trace.append("producer_exited")

    driver._server_thread = threading.Thread(target=produce)
    driver._server_thread.start()

    # A peer CAN sender with the same thread name must not be stopped/joined.
    peer = SimpleNamespace(release=threading.Event())
    peer_thread = threading.Thread(target=peer.release.wait, name=sender.name)
    peer_thread.start()

    arm = object.__new__(YamRobot)
    arm.driver = driver
    arm._closed = False
    arm._driver_closed = False
    arm.motion = MotionControl()
    arm._stop = arm.motion.stopped
    arm._pool = ThreadPoolExecutor(max_workers=1)
    pool = ThreadPoolExecutor(max_workers=1)
    closing = pool.submit(arm.close)
    try:
        assert stopped.wait(2)
        assert not bus_closed.wait(.05), "CAN closed while its sender was inside send()"
        assert not closing.done()
        release.set()
        closing.result(timeout=2)
        assert not sender.is_alive()
        assert not driver._server_thread.is_alive()
        assert peer_thread.is_alive()
        assert trace == ["producer_exited", "batch_sent", "sender_exited", "bus_closed"]
        arm.close()
        assert trace.count("bus_closed") == 1
    finally:
        release.set()
        peer.release.set()
        driver._stop_event.set()
        chain.running = False
        sender.join(timeout=2)
        driver._server_thread.join(timeout=2)
        peer_thread.join(timeout=2)
        pool.shutdown(wait=True)
        arm._pool.shutdown(wait=True)
