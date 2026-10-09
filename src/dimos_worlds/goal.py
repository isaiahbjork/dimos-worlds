"""Send a navigation goal to a running blueprint, after the topic has found its peers.

    dimos-worlds-goal /goal_request 12 11.4            # go2-warehouse
    dimos-worlds-goal /g1/goal_request 6 11 --yaw 0    # warehouse-fleet, one topic per robot

`dimos topic send` (DimOS 0.0.14) publishes as soon as its session opens, before peer discovery; on Linux that
goal is lost every time (verified on Ubuntu 22.04), on macOS sometimes. This waits for discovery, then publishes the
same PoseStamped on the same transport. Prints what it sent; the planner logs `Got new goal` when it lands.

    dimos-worlds-send /arm_command 'String("conveyor pick_table")'   # any message, same syntax as dimos topic send
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from dimos.core.transport_factory import make_transport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dimos-worlds-goal", description=__doc__.split("\n")[0])
    ap.add_argument("topic", help="goal topic, e.g. /goal_request or /go2/goal_request")
    ap.add_argument("x", type=float, help="metres, world frame")
    ap.add_argument("y", type=float, help="metres, world frame")
    ap.add_argument("--yaw", type=float, default=0.0, help="final heading, radians (default 0 = +x)")
    ap.add_argument("--wait", type=float, default=3.0, help="seconds to wait for peer discovery (default 3)")
    a = ap.parse_args(argv)
    goal = PoseStamped(frame_id="world", position=Vector3(a.x, a.y, 0.0),
                       orientation=Quaternion(0.0, 0.0, math.sin(a.yaw / 2), math.cos(a.yaw / 2)))
    transport = make_transport(a.topic, PoseStamped)
    time.sleep(a.wait)
    transport.broadcast(None, goal)
    time.sleep(1.0)  # let it flush before the session closes
    print(f"sent to {a.topic}: {goal}")
    return 0


def send_main(argv: list[str] | None = None) -> int:
    """`dimos topic send TOPIC EXPR` with the same discovery wait."""
    from dimos.cli.topic import _build_eval_context  # DimOS's own message namespace for EXPR

    ap = argparse.ArgumentParser(prog="dimos-worlds-send", description=send_main.__doc__)
    ap.add_argument("topic")
    ap.add_argument("message", help='e.g. \'String("conveyor pick_table")\'')
    ap.add_argument("--wait", type=float, default=3.0, help="seconds to wait for peer discovery (default 3)")
    a = ap.parse_args(argv)
    msg = eval(a.message, _build_eval_context())  # noqa: S307  (the user's own command line, as dimos topic send)
    transport = make_transport(a.topic, type(msg))
    time.sleep(a.wait)
    transport.broadcast(None, msg)
    time.sleep(1.0)
    print(f"sent to {a.topic}: {getattr(msg, 'data', msg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
