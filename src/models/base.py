import torch
from torch import nn

from ..entropy_models import ste_round, GGMConditional, GGMEntropyBottleneck

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

        self.g_a = None
        self.g_s = None
        self.h_a = None
        self.h_s = None

        self.entropy_bottleneck = GGMEntropyBottleneck(N)
        self.entropy_conditional = GGMConditional()

        self.mask_for_two_part = {}
        self.mask_for_four_part = {}
        self.mask_for_eight_part = {}
        self.mask_for_rec_s2 = {}

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

    def process_with_mask(self, y, scales, means, mask):
        scales_hat = torch.where(mask, scales, 0.0)
        means_hat = torch.where(mask, means, 0.0)
        y_res = torch.where(mask, y - means_hat, 0.0)
        if self.training:
            y_q = ste_round(y_res)
        else:
            y_q = torch.round(y_res)
        y_hat = y_q + means_hat
        return y_res, y_q, y_hat, scales_hat
    
    def get_one_channel_two_parts_mask(self, height, width, device):
        micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=torch.bool, device=device)
        mask_0 = micro_mask.repeat(height // 2, width // 2)
        return torch.stack((mask_0, ~mask_0))
    
    def get_mask_two_parts(self, channel, height, width, device):
        curr_mask_str = f"{channel}x{width}x{height}"
        if curr_mask_str not in self.mask_for_two_part:
            assert not isinstance(channel, int) or channel % 2 == 0
            m = self.get_one_channel_two_parts_mask(height, width, device)
            m = m.unsqueeze(1).expand(2, channel // 2, height, width)
            self.mask_for_two_part[curr_mask_str] = torch.stack((
                torch.cat((m[0], m[1]), dim=0),
                torch.cat((m[1], m[0]), dim=0),
            ))
        return self.mask_for_two_part[curr_mask_str].to(device)
    
    def get_one_channel_four_parts_mask(self, height, width, device):
        patterns = (((1, 0), (0, 0)), ((0, 1), (0, 0)), ((0, 0), (1, 0)), ((0, 0), (0, 1)))
        micro_mask = torch.tensor(patterns, dtype=torch.bool, device=device)
        mask = micro_mask.repeat(1, (height + 1) // 2, (width + 1) // 2)
        return mask[:, :height, :width]

    def get_mask_four_parts(self, channel, height, width, device):
        curr_mask_str = f"{channel}x{width}x{height}"
        if curr_mask_str not in self.mask_for_four_part:
            assert not isinstance(channel, int) or channel % 4 == 0
            m = self.get_one_channel_four_parts_mask(height, width, device)
            m = m.unsqueeze(1).expand(4, channel // 4, height, width)
            perms = [(0, 1, 2, 3), (3, 2, 1, 0), (2, 3, 0, 1), (1, 0, 3, 2)]
            self.mask_for_four_part[curr_mask_str] = torch.stack([
                torch.cat([m[i] for i in perm], dim=0) for perm in perms
            ])
        return self.mask_for_four_part[curr_mask_str].to(device)
    
    def get_one_channel_eight_parts_mask(self, height, width, device):
        patten_list = [((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0)), ((0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)), ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0)), \
                       ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0)), ((0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0)), ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0))]
        micro_mask = torch.tensor(patten_list, dtype=torch.bool, device=device).repeat(1, 2, 2)
        mask = micro_mask.repeat(1, (height + 1) // 8, (width + 1) // 8)
        return mask[:, :height, :width]
    
    def get_one_channel_eight_parts_mask_for_s1(self, height, width, device):
        patten_list = [((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0))]
        micro_mask = torch.tensor(patten_list, dtype=torch.bool, device=device)
        mask = micro_mask.repeat(1, (height + 1) // 4, (width + 1) // 4)
        return mask[:, :height, :width]
    
    def get_one_channel_eight_parts_mask_for_s2(self, height, width, device):
        patten_list = [((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0)), ((0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)), ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0)), \
                       ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0)), ((0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0)), \
                       ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0)), ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0))]
        micro_mask = torch.tensor(patten_list, dtype=torch.bool, device=device)
        mask = micro_mask.repeat(1, (height + 1) // 4, (width + 1) // 4)
        return mask[:, :height, :width]

    def get_mask_eight_parts(self, channel, height, width, device):
        curr_mask_str = f"{channel}x{width}x{height}"
        if curr_mask_str not in self.mask_for_eight_part:
            assert not isinstance(channel, int) or channel % 8 == 0
            m = self.get_one_channel_eight_parts_mask(height, width, device)
            m = m.unsqueeze(1).expand(8, channel // 8, height, width)
            cat_list = [
                [0, 2, 4, 6, 1, 3, 5, 7],
                [1, 3, 5, 7, 0, 2, 4, 6],
                [2, 4, 6, 0, 3, 5, 7, 1],
                [3, 5, 7, 1, 2, 4, 6, 0],
                [4, 6, 0, 2, 5, 7, 1, 3],
                [5, 7, 1, 3, 4, 6, 0, 2],
                [6, 0, 2, 4, 7, 1, 3, 5],
                [7, 1, 3, 5, 6, 0, 2, 4],
            ]
            self.mask_for_eight_part[curr_mask_str] = torch.stack([
                torch.cat([m[i] for i in perm], dim=0) for perm in cat_list
            ])
        return self.mask_for_eight_part[curr_mask_str].to(device)
    
    def get_mask_for_s1(self, height, width, device):
        indices = [0, 2, 4, 6, 1, 3, 5, 7]
        return self.get_one_channel_eight_parts_mask_for_s1(height, width, device)[indices]
    
    def get_mask_for_rec_s2(self, height, width, device):
        curr_mask_str = f"{width}x{height}"
        if curr_mask_str not in self.mask_for_rec_s2:
            self.mask_for_rec_s2[curr_mask_str] = self.get_one_channel_four_parts_mask(height, width, device)
        return self.mask_for_rec_s2[curr_mask_str].to(device)
    
    def get_mask_for_s2(self, height, width, device):
        mask = self.get_one_channel_eight_parts_mask_for_s2(height, width, device)
        indices_1 = [0, 2, 4, 6, 1, 3, 5, 7]
        indices_2 = [1, 3, 5, 7, 0, 2, 4, 6]
        return mask[indices_1] | mask[indices_2]

    @staticmethod
    def merge_mask_parts(mask, channel):
        # Part p of the mask covers the channels [p*C//P, (p+1)*C//P).
        return mask[:, None].expand(-1, channel // mask.shape[0], -1, -1).flatten(0, 1)
    
    def get_s2_hyper_with_mask(self, y, mask_s1, mask_s2, mask_rec_s2):
        mask = self.merge_mask_parts(mask_s2 & ~mask_s1, y.shape[1]).expand_as(y)
        # expand rather than a hand-built shape: a traced shape is an uniterable Proxy
        mask_rec_s2 = self.merge_mask_parts(mask_rec_s2, y.shape[1]).expand(y.shape[0], -1, -1, -1)
        return torch.masked_scatter(y.new_zeros(mask_rec_s2.shape), ~mask_rec_s2, y[mask].reshape(-1))
    
    def get_s3_hyper_with_mask(self, common_params, mask):
        merged = self.merge_mask_parts(mask, common_params.shape[1])
        return common_params.masked_fill(merged, 0.0)
    
    def get_s1_s2_with_mask(self, y, mask):
        # Every part keeps the same fraction of positions, so a single select
        # on the merged mask matches the per-part chunks.
        mask = self.merge_mask_parts(mask, y.shape[1])
        batch, height, width = y.shape[0], y.shape[2] // 2, y.shape[3] // 2
        return y.masked_select(mask).view(batch, -1, height, width)
    
    def recon_for_s2_s3(self, y_curr, mask):
        mask = self.merge_mask_parts(mask, y_curr.shape[1]).expand(y_curr.shape[0], -1, -1, -1)
        return torch.masked_scatter(y_curr.new_zeros(mask.shape), mask, y_curr.reshape(-1))
    
    @staticmethod
    def combine_for_writing_s1(x):
        return sum(x.chunk(2, 1))
    
    @staticmethod
    def combine_for_writing_s2(x):
        return sum(x.chunk(4, 1))

    @staticmethod
    def combine_for_writing_s3(x):
        return sum(x.chunk(8, 1))
