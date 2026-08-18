import time

import click

from arx5_collector.robot import Arx5Robot


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
@click.option("--command-mode", default="cmd", show_default=True, type=click.Choice(["cmd", "traj"]))
def main(model, interface, close_pos, open_pos, hold_sec, step, frequency, sequence, command_mode):
    with Arx5Robot(
        model=model,
        interface=interface,
        reset_to_home=False,
        command_mode=command_mode,
    ) as robot:
        gain = robot.controller.get_gain()
        print("gripper_torque_max", robot.robot_config.gripper_torque_max)
        print("gripper_kp", gain.gripper_kp, "gripper_kd", gain.gripper_kd)
        state = robot.get_state()
        pose = state["TargetTCPPose"].copy()
        print("initial gripper_pos", state["gripper_pos"], "target", state["target_gripper_pos"])

        current = float(state["gripper_pos"][0])
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
                robot.schedule_waypoint(pose, gripper_pos=current)
                state = robot.get_state()
                now = time.time()
                if now - last_print > 0.2:
                    print(
                        f" actual {state['gripper_pos'][0]:.5f}"
                        f" target {state['target_gripper_pos'][0]:.5f}"
                        f" cmd {current:.5f}",
                        end="\r",
                    )
                    last_print = now
                time.sleep(dt)
            print("")


if __name__ == "__main__":
    main()
