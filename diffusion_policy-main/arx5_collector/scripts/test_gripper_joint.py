import time

import click

from arx5_collector.sdk_path import ensure_arx5_sdk_path

ensure_arx5_sdk_path()

import arx5_interface as arx5
from arx5_local_config import apply_local_controller_gain, apply_local_robot_config


@click.command()
@click.option("--model", default="X5", show_default=True)
@click.option("--interface", default="can1", show_default=True)
@click.option("--close-pos", default=0.0, show_default=True, type=float)
@click.option("--open-pos", default=0.03, show_default=True, type=float)
@click.option("--hold-sec", default=8.0, show_default=True, type=float)
@click.option("--step", default=0.0002, show_default=True, type=float)
@click.option("--frequency", default=50.0, show_default=True, type=float)
@click.option(
    "--sequence",
    default="both",
    show_default=True,
    type=click.Choice(["close", "open", "both"]),
)
def main(model, interface, close_pos, open_pos, hold_sec, step, frequency, sequence):
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    apply_local_robot_config(robot_config)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    controller_config.background_send_recv = True

    controller = arx5.Arx5JointController(robot_config, controller_config, interface)
    apply_local_controller_gain(controller)
    gain = controller.get_gain()
    print("gripper_open_readout", robot_config.gripper_open_readout)
    print("gripper_width", robot_config.gripper_width)
    print("gripper_torque_max", robot_config.gripper_torque_max)
    print("gripper_kp", gain.gripper_kp, "gripper_kd", gain.gripper_kd)

    try:
        state = controller.get_joint_state()
        hold_q = state.pos().copy()
        current = float(state.gripper_pos)
        print(f"initial gripper_pos {current:.5f}")

        dt = 1.0 / frequency
        if sequence == "close":
            targets = [("close", close_pos)]
        elif sequence == "open":
            targets = [("open", open_pos)]
        else:
            targets = [("close", close_pos), ("open", open_pos)]
        for name, pos in targets:
            direction = "opening" if pos > current else "closing"
            print(f"command {name}: gripper_pos={pos:.5f} ({direction} direction)")
            start = time.time()
            last_print = 0.0
            while time.time() - start < hold_sec:
                if abs(pos - current) <= step:
                    current = pos
                elif pos > current:
                    current += step
                else:
                    current -= step

                cmd = arx5.JointState(robot_config.joint_dof)
                cmd.pos()[:] = hold_q
                cmd.gripper_pos = current
                controller.set_joint_cmd(cmd)

                state = controller.get_joint_state()
                joint_cmd = controller.get_joint_cmd()
                now = time.time()
                if now - last_print > 0.2:
                    print(
                        f" actual {state.gripper_pos:.5f}"
                        f" target {joint_cmd.gripper_pos:.5f}"
                        f" cmd {current:.5f}"
                        f" torque {state.gripper_torque:.3f}",
                        end="\r",
                    )
                    last_print = now
                time.sleep(dt)
            print("")
    finally:
        controller.set_to_damping()


if __name__ == "__main__":
    main()
