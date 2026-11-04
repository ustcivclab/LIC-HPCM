import math
from pathlib import Path

import torch

from .models.HPCM_1B import HPCM as HPCM1B
from .models.HPCM_Base import HPCM as HPCMBase
from .models.HPCM_Large import HPCM as HPCMLarge

__all__ = ["hpcm2025_base", "hpcm2025_large", "hpcm2025_1b"]

CHECKPOINT_ROOT = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "lic-hpcm"

model_architectures: dict[str, type] = {
    "hpcm2025-base": HPCMBase,
    "hpcm2025-large": HPCMLarge,
    "hpcm2025-1b": HPCM1B,
}

# quality level -> lambda, ordered from lowest (level 1, fewest bits) to
# highest quality, matching compressai's quality-level convention.
model_lambdas: dict[str, dict[str, dict[int, str]]] = {
    "hpcm2025-base": {
        "mse": {1: "0.0018", 2: "0.0035", 3: "0.0067", 4: "0.013", 5: "0.025", 6: "0.0483"},
        "ms-ssim": {1: "2.4", 2: "4.58", 3: "8.73", 4: "16.64", 5: "31.73", 6: "60.5"},
    },
    "hpcm2025-large": {
        "mse": {1: "0.0018", 2: "0.0035", 3: "0.0067", 4: "0.013", 5: "0.025", 6: "0.0483"},
        "ms-ssim": {1: "2.4", 2: "4.58", 3: "8.73", 4: "16.64", 5: "31.73", 6: "60.5"},
    },
    "hpcm2025-1b": {
        "mse": {6: "0.0483"},
    },
}

cfgs: dict[str, dict[int, tuple]] = {
    name: {q: () for q in lams["mse"]} for name, lams in model_lambdas.items()
}


def get_scale_table(min=0.12, max=64, levels=60):
    """Returns table of logarithmically scaled."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def _checkpoint_path(architecture: str, metric: str, quality: int) -> Path | None:
    lam = model_lambdas.get(architecture, {}).get(metric, {}).get(quality)
    if lam is None:
        return None
    variant = architecture.replace("hpcm2025-", "")
    return CHECKPOINT_ROOT / variant / metric / f"{lam}.pth.tar"


def _load_model(architecture, metric, quality, pretrained=False, progress=True, **kwargs):
    if architecture not in model_architectures:
        raise ValueError(f'Invalid architecture name "{architecture}"')

    if quality not in cfgs[architecture]:
        raise ValueError(f'Invalid quality value "{quality}"')

    model = model_architectures[architecture](*cfgs[architecture][quality], **kwargs)

    if pretrained:
        checkpoint_path = _checkpoint_path(architecture, metric, quality)
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise RuntimeError(f"Pre-trained model not found at {checkpoint_path}")

        # `progress` is kept for signature-compatibility with compressai.zoo;
        # checkpoints are loaded from a local cache, not downloaded.
        del progress
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # strict=False: the released checkpoints carry no adaptive_params_list,
        # which keeps its init values.
        model.load_state_dict(state_dict, strict=False)
        model.entropy_bottleneck.update()
        model.entropy_conditional.update_scale_table(get_scale_table())
        model.eval()

    return model


def hpcm2025_base(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Hierarchical Progressive Context Modeling (Base) from Y. Li, H. Zhang,
    L. Li, D. Liu: `"Learned Image Compression with Hierarchical Progressive
    Context Modeling" <https://arxiv.org/abs/2507.19125>`_, ICCV 2025.

    Args:
        quality (int): Quality levels (1: lowest, highest: 6)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model loaded from
            the local checkpoint cache (``~/.cache/torch/hub/checkpoints/lic-hpcm``)
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 6:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 6)')

    return _load_model("hpcm2025-base", metric, quality, pretrained, progress, **kwargs)


def hpcm2025_large(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""Hierarchical Progressive Context Modeling (Large) from Y. Li, H. Zhang,
    L. Li, D. Liu: `"Learned Image Compression with Hierarchical Progressive
    Context Modeling" <https://arxiv.org/abs/2507.19125>`_, ICCV 2025.

    Args:
        quality (int): Quality levels (1: lowest, highest: 6)
        metric (str): Optimized metric, choose from ('mse', 'ms-ssim')
        pretrained (bool): If True, returns a pre-trained model loaded from
            the local checkpoint cache (``~/.cache/torch/hub/checkpoints/lic-hpcm``)
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse", "ms-ssim"):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 6:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 6)')

    return _load_model("hpcm2025-large", metric, quality, pretrained, progress, **kwargs)


def hpcm2025_1b(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""HPCM-1B: scaled-up Hierarchical Progressive Context Modeling from Y. Li,
    H. Zhang, L. Li, D. Liu: `"Scaling Learned Image Compression Models up to 1
    Billion" <https://arxiv.org/abs/2508.09075>`_.

    Only one checkpoint (MSE, lambda=0.0483, quality=1) is currently released.

    Args:
        quality (int): Quality level (only 1 is available so far)
        metric (str): Optimized metric, choose from ('mse',)
        pretrained (bool): If True, returns a pre-trained model loaded from
            the local checkpoint cache (``~/.cache/torch/hub/checkpoints/lic-hpcm``)
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if metric not in ("mse",):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 1:
        raise ValueError(f'Invalid quality "{quality}", only level 1 is available')

    return _load_model("hpcm2025-1b", metric, quality, pretrained, progress, **kwargs)
