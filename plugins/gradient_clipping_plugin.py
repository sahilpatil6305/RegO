from avalanche.training.plugins import SupervisedPlugin
from torch.nn.utils import clip_grad_norm_

class GradientClippingPlugin(SupervisedPlugin):
    """
    Plugin to clip gradients before optimizer step to prevent exploding gradients.
    """
    def __init__(self, max_norm=1.0):
        super().__init__()
        self.max_norm = max_norm
        print(f"[GradientClipping] Initialized with max_norm={max_norm}")

    def before_update(self, strategy, **kwargs):
        """Clip gradients before optimizer update"""
        clip_grad_norm_(strategy.model.parameters(), self.max_norm)
