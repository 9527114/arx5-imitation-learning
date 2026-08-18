import time
from multiprocessing.managers import SharedMemoryManager

import click
import numpy as np

from arx5_collector.input import SpaceMouseTeleop


@click.command()
@click.option("--duration", default=20.0, show_default=True, type=float)
@click.option("--frequency", "-f", default=30.0, show_default=True, type=float)
@click.option("--deadzone", default=0.02, show_default=True, type=float)
def main(duration, frequency, deadzone):
    dt = 1.0 / frequency
    target_pose = np.zeros(6, dtype=np.float64)
    target_gripper = 0.04
    gripper_width = 0.088

    with SharedMemoryManager() as shm_manager:
        teleop = SpaceMouseTeleop(
            shm_manager=shm_manager,
            deadzone=deadzone,
            smoothing_window=0,
        )
        teleop.start()
        print("Move SpaceMouse or press buttons. Ctrl+C to stop.")
        t_end = time.monotonic() + duration
        try:
            while time.monotonic() < t_end:
                command = teleop.update(
                    target_pose=target_pose,
                    target_gripper_pos=target_gripper,
                    dt=dt,
                    gripper_width=gripper_width,
                )
                target_pose = command.action
                target_gripper = command.gripper_action
                print(
                    "motion",
                    np.array2string(command.raw_motion, precision=3),
                    f"left={command.left_pressed}",
                    f"right={command.right_pressed}",
                    "target",
                    np.array2string(target_pose, precision=4),
                    f"gripper={target_gripper:.5f}",
                )
                time.sleep(dt)
        except KeyboardInterrupt:
            pass
        finally:
            teleop.stop()


if __name__ == "__main__":
    main()

