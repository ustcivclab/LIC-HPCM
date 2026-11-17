import torch
from torch import nn

from ..entropy_models import GGMConditional, GGMEntropyBottleneck
from ..layers.layers import (
    Downsample21,
    Downsample32,
    Downsample_32,
    MaskSample1,
    MaskSample2,
    MaskSample3,
    Resample33,
    Upsample12,
    Upsample23,
)

# Checkpoints saved before the entropy models were refactored keep the
# hyperprior parameters at the top level, plus the CDF buffers, which are only
# derived from them and are rebuilt by update().
_LEGACY_PARAMS = {
    "means_hyper": "entropy_bottleneck.means",
    "scales_hyper": "entropy_bottleneck.scales",
}
_LEGACY_PREFIX = "entropy_estimation."
_LEGACY_BUFFERS = (
    "scale_table",
    "quantized_cdf_z",
    "cdf_length_z",
    "offset_z",
    "quantized_cdf_y",
    "cdf_length_y",
    "offset_y",
)


class BB(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.entropy_bottleneck = GGMEntropyBottleneck(N)
        self.entropy_conditional = GGMConditional()

        self.downsample32 = Downsample32()
        self.downsample21 = Downsample21()
        self.upsample12 = Upsample12()
        self.upsample23 = Upsample23()
        self.downsample_32 = Downsample_32()
        self.resample33 = Resample33()
        self.masksample3 = MaskSample3()
        self.masksample2 = MaskSample2()
        self.masksample1 = MaskSample1()

    def load_state_dict(self, state_dict, strict=True):
        """Rename the pre-refactor keys and drop the derived CDF buffers."""
        state_dict = {k: v for k, v in state_dict.items() if k not in _LEGACY_BUFFERS}
        for old, new in _LEGACY_PARAMS.items():
            if old in state_dict:
                state_dict[new] = state_dict.pop(old)
        for key in [k for k in state_dict if k.startswith(_LEGACY_PREFIX)]:
            value = state_dict.pop(key)
            suffix = key.removeprefix(_LEGACY_PREFIX)
            for module in ("entropy_bottleneck", "entropy_conditional"):
                state_dict[f"{module}.{suffix}"] = value

        return super().load_state_dict(state_dict, strict)

    def process_with_mask(self, y, scales, means, mask_sample, i, write=False):
        """One context step: (symbols if write else residual, y_hat, scales_hat)."""
        scales_hat = mask_sample(scales, i)
        y_res = mask_sample(y - means, i)
        means_hat = mask_sample(means, i)
        if write:
            y_q = self.entropy_conditional.quantize(y_res, "symbols")
            return y_q, y_q + means_hat, scales_hat
        y_round, _ = self.entropy_conditional(y_res)
        return y_res, y_round + means_hat, scales_hat
    
    @staticmethod
    def combine_for_writing_s1(x):
        return sum(x.chunk(2, 1))
    
    @staticmethod
    def combine_for_writing_s2(x):
        return sum(x.chunk(4, 1))

    @staticmethod
    def combine_for_writing_s3(x):
        return sum(x.chunk(8, 1))
