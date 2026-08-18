import click

from arx5_collector.robot import Arx5Robot


@click.command()
@click.option("--model", default="X5", show_default=True)
@click.option("--interface", default="can1", show_default=True)
def main(model, interface):
    with Arx5Robot(model=model, interface=interface, reset_to_home=False) as robot:
        gain = robot.controller.get_gain()
        print("gripper_kp", gain.gripper_kp)
        print("gripper_kd", gain.gripper_kd)
        state = robot.get_state()
        for key, value in state.items():
            print(key, value)


if __name__ == "__main__":
    main()
