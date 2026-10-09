"""The iiwa arm cell: Menagerie model on the pedestal, positional IK, a tote picked off the conveyor and placed."""
from __future__ import annotations

import numpy as np
import pytest

from dimos_worlds.arm.cell import TARGETS, ArmCell
from dimos_worlds.warehouse import cell as wh


@pytest.fixture(scope="module")
def arm() -> ArmCell:
    return ArmCell()


def test_arm_stands_on_the_pedestal_and_the_tote_rests_on_the_conveyor(arm: ArmCell) -> None:
    m = arm.model
    base = arm.data.xpos[m.body("iiwa/base").id]
    assert np.allclose(base, wh.ARM_BASE, atol=1e-6)
    assert m.nu == 7 and len(arm.joints) == 7
    p, _ = arm.tote_pose()
    assert np.allclose(p[:2], TARGETS["conveyor"][:2], atol=0.01)
    assert p[2] + wh.TOTE_SIZE[2] / 2 == pytest.approx(TARGETS["conveyor"][2], abs=0.01)


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_ik_reaches_every_target_with_the_tool_down(arm: ArmCell, name: str) -> None:
    x, y, z = TARGETS[name]
    for lift in (0.0, wh.ARM_CLEARANCE_M):
        q, err = arm.solve((x, y, z + lift))
        assert err < 2e-3
        assert np.all(q >= arm.lo) and np.all(q <= arm.hi)


def test_pick_from_conveyor_place_on_pick_table() -> None:
    cell = ArmCell()
    steps: list[str] = []
    final = cell.pick_place("conveyor", "pick_table", log=steps.append)
    tx, ty, tz = TARGETS["pick_table"]
    assert np.hypot(final[0] - tx, final[1] - ty) < 0.02
    assert final[2] + wh.TOTE_SIZE[2] / 2 == pytest.approx(tz, abs=0.01)  # resting on the table, not floating
    _, quat = cell.tote_pose()
    assert abs(quat[0]) > 0.999  # upright and still square to the table (it started at yaw 0)
    assert np.allclose(cell.q(), [0.0, 0.785398, 0.0, -1.5708, 0.0, 0.0, 0.0], atol=0.02)  # back home
    assert any("gripped" in s for s in steps) and any("released" in s for s in steps)
    # and on to the low pallet
    final = cell.pick_place("pick_table", "pallet-2")
    px, py, pz = TARGETS["pallet-2"]
    assert np.hypot(final[0] - px, final[1] - py) < 0.02
    assert final[2] + wh.TOTE_SIZE[2] / 2 == pytest.approx(pz, abs=0.01)


def test_out_of_reach_and_missing_tote_are_refused() -> None:
    cell = ArmCell()
    with pytest.raises(ValueError):
        cell.move_tip((wh.ARM_BASE[0] + 1.5, wh.ARM_BASE[1], 1.0))
    with pytest.raises(RuntimeError):
        cell.pick_place("pick_table", "conveyor")  # the tote is on the conveyor, not the table
