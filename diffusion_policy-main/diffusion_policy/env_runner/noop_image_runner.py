from typing import Dict

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class NoopImageRunner(BaseImageRunner):
    """Offline-only runner for real datasets without a simulator rollout."""

    def run(self, policy: BaseImagePolicy) -> Dict:
        return {}
