"""`WarehouseArm`: the arm cell (arm.cell) as a DimOS module, stepping in real time on its own thread.

    in   arm_command      String         "<from> <to>", reach targets from layout.json: conveyor, pick_table, pallet-2
    out  arm_joint_state  JointState     the iiwa's seven joints, 25 Hz
    out  tote_pose        PoseStamped    the free tote, 25 Hz
    out  arm_status       String         what the arm is doing, and "done <from> <to> x y z" / "error ..."

    dimos topic send /arm_command 'String("conveyor pick_table")'
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.String import String
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class WarehouseArmConfig(ModuleConfig):
    tote_id: str = "tote-09"  # the tote at the conveyor's end
    speed: float = 1.0  # x real time
    publish_every_steps: int = 20  # 25 Hz at 2 ms steps


class WarehouseArm(Module):
    config: WarehouseArmConfig
    arm_command: In[String]
    arm_joint_state: Out[JointState]
    tote_pose: Out[PoseStamped]
    arm_status: Out[String]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cmds: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._cell: Any = None
        self._steps = 0
        self._wall0 = 0.0
        self._sim0 = 0.0

    @rpc
    def start(self) -> None:
        super().start()
        from dimos_worlds.arm.cell import ArmCell

        self._cell = ArmCell(self.config.tote_id)
        self._cell.on_step.append(self._after_step)
        self.register_disposable(Disposable(self.arm_command.subscribe(lambda m: self._cmds.put(str(m.data)))))
        threading.Thread(target=self._run_arm, name="warehouse-arm", daemon=True).start()
        self._status("ready")

    @rpc
    def stop(self) -> None:
        self._stop.set()
        super().stop()

    def _status(self, text: str) -> None:
        logger.info(f"arm: {text}")
        self.arm_status.publish(String(text))

    def _run_arm(self) -> None:
        cell = self._cell
        self._wall0, self._sim0 = time.monotonic(), cell.sim_time
        while not self._stop.is_set():
            try:
                cmd = self._cmds.get_nowait()
            except queue.Empty:
                cell.step()
                continue
            parts = cmd.replace(",", " ").replace("->", " ").split()
            if len(parts) != 2:
                self._status(f"error: want '<from> <to>', got {cmd!r}")
                continue
            src, dst = parts
            try:
                p = cell.pick_place(src, dst, log=self._status)
                self._status(f"done {src} {dst} {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}")
            except Exception as e:  # out of reach, no tote there: report, keep running
                self._status(f"error: {e}")

    def _after_step(self, cell: Any) -> None:
        self._steps += 1
        if self._steps % self.config.publish_every_steps == 0:
            ts = time.time()
            m = cell.model
            self.arm_joint_state.publish(JointState(
                ts=ts, name=[m.joint(j).name for j in cell.joints],
                position=[float(v) for v in cell.q()],
                velocity=[float(v) for v in cell.data.qvel[cell.dadr]]))
            p, q = cell.tote_pose()
            self.tote_pose.publish(PoseStamped(ts=ts, frame_id="world", position=Vector3(*map(float, p)),
                                               orientation=Quaternion(float(q[1]), float(q[2]), float(q[3]),
                                                                      float(q[0]))))
            # pace to speed x real time
            ahead = (cell.sim_time - self._sim0) / self.config.speed - (time.monotonic() - self._wall0)
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -1.0:
                self._wall0, self._sim0 = time.monotonic(), cell.sim_time
        if self._stop.is_set():
            raise SystemExit
