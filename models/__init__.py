"""
Local model definitions for standalone dehazing inference.
Supports: MSFA-DeNet (v1) and MSFA-DeNet v2.
"""

from .msfa_denet import MSFANetLite
from .msfa_denet_v2 import MSFADeNetV2
from .image_utils import load_image, save_image, to_tensor

# ── Model Registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "msfa_denet": MSFANetLite,
    "msfa_denet_v2": MSFADeNetV2,
}

# Both models predict clean images directly (no transmission → physics step)
DIRECT_MODELS = {"msfa_denet", "msfa_denet_v2"}


def build_model(arch_type: str, **kwargs):
    """
    Build a model instance from architecture type string.

    Args:
        arch_type: One of 'msfa_denet' or 'msfa_denet_v2'.
        **kwargs:  Architecture-specific keyword args (e.g. channels=64).

    Returns:
        An nn.Module model instance.
    """
    if arch_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: '{arch_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[arch_type]
    try:
        return model_class(**kwargs)
    except TypeError:
        return model_class()
