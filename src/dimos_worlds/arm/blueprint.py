"""`dimos run dimos-worlds.warehouse-arm`: the iiwa arm cell on its own, or next to the fleet:

    dimos --viewer none run dimos-worlds.warehouse-fleet dimos-worlds.warehouse-arm
    dimos topic send /arm_command 'String("conveyor pick_table")'
    dimos topic echo /arm_status String
"""
from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect

from dimos_worlds.arm.module import WarehouseArm

warehouse_arm = autoconnect(WarehouseArm.blueprint())
