import os
import sys

import click

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import arx5_interface as arx5
from arx5_local_config import GRIPPER_OPEN_READOUT, GRIPPER_WIDTH


@click.command()
@click.argument("model")
@click.argument("interface")
@click.option(
    "--open-readout",
    type=float,
    default=GRIPPER_OPEN_READOUT,
    show_default=True,
    help="Known fully-open gripper motor readout after zeroing.",
)
@click.option(
    "--width",
    type=float,
    default=GRIPPER_WIDTH,
    show_default=True,
    help="Known fully-open gripper width in meters.",
)
@click.option(
    "--negative-startup",
    is_flag=True,
    help="Use negative open_readout only to bypass stale negative startup readout.",
)
def main(model: str, interface: str, open_readout: float, width: float, negative_startup: bool):
    """
    Prepare gripper zero after robot power-cycle.

    By default this script starts with the normal positive open_readout. If the
    robot was power-cycled and fails startup with a small negative gripper
    position, rerun with --negative-startup to enter the SDK calibration routine.
    """
    if open_readout == 0:
        raise click.ClickException("--open-readout must be non-zero.")

    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    if negative_startup:
        robot_config.gripper_open_readout = -abs(open_readout)
    else:
        robot_config.gripper_open_readout = abs(open_readout)
    robot_config.gripper_width = width

    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    controller_config.gravity_compensation = False

    click.echo("Starting gripper zero preparation.")
    click.echo(f"Using startup readout: {robot_config.gripper_open_readout}")
    click.echo(f"Normal configured readout should remain positive: {abs(open_readout)}")
    click.echo("")
    click.echo("Follow the SDK prompts:")
    click.echo("  1. Fully close the gripper, then press Enter.")
    click.echo("  2. Fully open the gripper, then press Enter.")
    click.echo("")

    joint_controller = arx5.Arx5JointController(
        robot_config, controller_config, interface
    )
    joint_controller.calibrate_gripper()

    click.echo("")
    click.echo("Gripper zero preparation finished.")
    click.echo(
        "If the printed fully-open readout differs from "
        f"{abs(open_readout)}, update arx5_local_config.py."
    )
    click.echo(
        "If startup fails with a small negative gripper position, rerun this "
        "script with --negative-startup."
    )


if __name__ == "__main__":
    main()
