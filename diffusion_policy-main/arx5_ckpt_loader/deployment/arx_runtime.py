import time


class PolicyActionScheduler:
    """ARX5-aware safety wrapper for policy-generated EEF actions."""

    def __init__(
        self,
        robot,
        command_latency: float,
        tracking_guard: bool = True,
        log_interval: float = 0.5,
    ):
        self.robot = robot
        self.command_latency = float(command_latency)
        self.tracking_guard = bool(tracking_guard)
        self.log_interval = float(log_interval)
        self._last_guard_print = 0.0

    def schedule(self, pose, gripper_pos, target_time=None):
        if target_time is None:
            target_time = time.time() + self.command_latency
        if self.tracking_guard and not self.robot.tracking_is_healthy():
            pos_error, rot_error, state = self.robot.tracking_error()
            pose = state["ActualTCPPose"]
            gripper_pos = float(state["gripper_pos"][0])
            now = time.time()
            if now - self._last_guard_print >= self.log_interval:
                self._last_guard_print = now
                print(
                    "tracking guard hold:",
                    f"pos_error={pos_error:.4f}",
                    f"rot_error={rot_error:.4f}",
                )
        self.robot.schedule_waypoint(
            pose,
            target_time=target_time,
            gripper_pos=gripper_pos,
        )
