import click

from arx5_collector.sdk_path import ensure_arx5_sdk_path

ensure_arx5_sdk_path()

import arx5_interface as arx5
from arx5_local_config import (
    GRIPPER_OPEN_READOUT,
    GRIPPER_WIDTH,
    apply_local_controller_gain,
    apply_local_robot_config,
)


def make_robot_config(model, negative_startup=False):
    robot_config = arx5.RobotConfigFactory.get_instance().get_config(model)
    apply_local_robot_config(robot_config)
    if negative_startup:
        robot_config.gripper_open_readout = -abs(GRIPPER_OPEN_READOUT)
    else:
        robot_config.gripper_open_readout = abs(GRIPPER_OPEN_READOUT)
    return robot_config


def make_joint_controller(model, interface, negative_startup=False):
    robot_config = make_robot_config(model, negative_startup=negative_startup)
    controller_config = arx5.ControllerConfigFactory.get_instance().get_config(
        "joint_controller", robot_config.joint_dof
    )
    controller_config.gravity_compensation = False
    controller_config.background_send_recv = True
    controller = arx5.Arx5JointController(robot_config, controller_config, interface)
    apply_local_controller_gain(controller)
    return controller, robot_config


def check_expected_gripper(pos, width, expected_gripper):
    if expected_gripper == "any":
        return True
    if expected_gripper == "open":
        return pos > width * 0.7
    if expected_gripper == "closed":
        return pos < width * 0.3
    raise ValueError(f"Unsupported expected gripper state: {expected_gripper}")


def run_sanity_check(model, interface, expected_gripper):
    controller = None
    try:
        controller, robot_config = make_joint_controller(model, interface)
        state = controller.get_joint_state()
        gain = controller.get_gain()
        pos = float(state.gripper_pos)
        click.echo("Robot startup sanity check passed.")
        click.echo(f"gripper_open_readout {robot_config.gripper_open_readout}")
        click.echo(f"gripper_width {robot_config.gripper_width}")
        click.echo(f"gripper_torque_max {robot_config.gripper_torque_max}")
        click.echo(f"gripper_kp {gain.gripper_kp} gripper_kd {gain.gripper_kd}")
        click.echo(f"gripper_pos {pos:.5f}")
        click.echo(f"gripper_torque {state.gripper_torque:.3f}")
        if not check_expected_gripper(pos, robot_config.gripper_width, expected_gripper):
            click.echo("")
            click.echo(
                "WARNING: SDK gripper_pos does not match the physical gripper state "
                f"you declared with --expected-gripper {expected_gripper}."
            )
            click.echo(
                "This usually means the gripper zero was set at the wrong physical position."
            )
            return False
        return True
    except Exception as exc:
        click.echo("Robot startup sanity check failed.")
        click.echo(str(exc))
        return False
    finally:
        if controller is not None:
            controller.set_to_damping()


def run_gripper_calibration(model, interface, negative_startup):
    click.echo("")
    click.echo("Starting semi-automatic gripper zero preparation.")
    click.echo(f"negative_startup {negative_startup}")
    click.echo(f"configured positive open_readout {abs(GRIPPER_OPEN_READOUT)}")
    click.echo(f"configured gripper_width {GRIPPER_WIDTH}")
    click.echo("")
    click.echo("Follow the SDK prompts exactly:")
    click.echo("  1. When asked to fully close: physically close the gripper, then press Enter.")
    click.echo("  2. When asked to fully open: physically open the gripper, then press Enter.")
    click.echo("")

    controller = None
    try:
        controller, _ = make_joint_controller(
            model, interface, negative_startup=negative_startup
        )
        controller.calibrate_gripper()
    finally:
        if controller is not None:
            controller.set_to_damping()


@click.command()
@click.option("--model", default="X5", show_default=True)
@click.option("--interface", default="can1", show_default=True)
@click.option(
    "--expected-gripper",
    default="any",
    show_default=True,
    type=click.Choice(["any", "open", "closed"]),
    help="Physical gripper state before running this script.",
)
@click.option(
    "--force-calibrate",
    is_flag=True,
    help="Skip accepting the startup check and run gripper zero calibration.",
)
@click.option(
    "--negative-startup",
    is_flag=True,
    help="Use negative open_readout only when normal startup fails with negative gripper_pos.",
)
@click.option(
    "--no-recheck",
    is_flag=True,
    help="Do not run a second sanity check after calibration.",
)
def main(
    model,
    interface,
    expected_gripper,
    force_calibrate,
    negative_startup,
    no_recheck,
):
    ok = False
    if not force_calibrate:
        ok = run_sanity_check(model, interface, expected_gripper)
        if ok:
            click.echo("")
            click.echo("Robot session is ready.")
            return

    click.echo("")
    if not force_calibrate:
        click.echo("Sanity check did not pass. Calibration is recommended.")
        if not click.confirm("Run gripper zero calibration now?", default=True):
            raise click.ClickException("Robot session is not ready.")

    run_gripper_calibration(model, interface, negative_startup=negative_startup)

    if no_recheck:
        click.echo("")
        click.echo("Calibration finished. Recheck was skipped.")
        return

    click.echo("")
    click.echo("Rechecking startup after calibration.")
    if run_sanity_check(model, interface, expected_gripper):
        click.echo("")
        click.echo("Robot session is ready.")
    else:
        raise click.ClickException(
            "Recheck failed. If startup failed with negative gripper_pos, rerun "
            "with --negative-startup. If the physical state does not match the "
            "SDK reading, rerun with --force-calibrate and follow the prompts."
        )


if __name__ == "__main__":
    main()
