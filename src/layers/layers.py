import torch
from torch import nn


class Sample(nn.Module):
    r"""Sample"""


class MaskSample3(Sample):
    def __init__(self):
        super().__init__()
        self.micro_mask = nn.Parameter(
            torch.tensor(((1, 0), (0, 1)), dtype=torch.bool),
            requires_grad=False,
        )

    def get_mask_two_parts(self, channel, height, width):
        assert not isinstance(channel, int) or channel % 2 == 0
        mask_0 = self.micro_mask.repeat(height // 2, width // 2)
        mask = torch.stack((mask_0, ~mask_0))
        mask = mask.unsqueeze(1).expand(2, channel // 2, height, width)
        mask_for_two_part = torch.stack((
            torch.cat((mask[0], mask[1]), dim=0),
            torch.cat((mask[1], mask[0]), dim=0),
        ))
        return mask_for_two_part

    def forward(self, y: torch.Tensor, i: int) -> torch.Tensor:
        _, C, H, W = y.shape
        mask_list = self.get_mask_two_parts(C, H, W)
        return torch.where(mask_list[i], y, 0)


class MaskSample1(Sample):
    def __init__(self):
        super().__init__()
        self.micro_mask = nn.Parameter(
            torch.tensor(
                (
                    ((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0)),
                    ((0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)),
                    ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)),
                    ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0)),
                    ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0)),
                    ((0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0)),
                    ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0)),
                    ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0)),
                ),
                dtype=torch.bool,
            ),
            requires_grad=False,
        )

    def get_mask_eight_parts(self, channel, height, width):
        assert not isinstance(channel, int) or channel % 8 == 0
        mask = self.micro_mask.repeat(1, (height + 3) // 4, (width + 3) // 4)
        mask = mask[:, :height, :width]
        mask = mask.unsqueeze(1).expand(8, channel // 8, height, width)
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
        mask_for_eight_part = torch.stack([
            torch.cat([mask[i] for i in perm], dim=0) for perm in cat_list
        ])
        return mask_for_eight_part

    def forward(self, y: torch.Tensor, i: int) -> torch.Tensor:
        _, C, H, W = y.shape
        mask_list = self.get_mask_eight_parts(C, H, W)
        return torch.where(mask_list[i], y, 0)


class MaskSample2Base(Sample):
    def __init__(self):
        super().__init__()
        self.micro_mask = nn.Parameter(
            torch.tensor(
                (
                    ((1, 0), (0, 0)),
                    ((0, 1), (0, 0)),
                    ((0, 0), (1, 0)),
                    ((0, 0), (0, 1)),
                ),
                dtype=torch.bool,
            ),
            requires_grad=False,
        )

    def get_one_channel_four_parts_mask(self, height, width):
        mask = self.micro_mask.repeat(1, (height + 1) // 2, (width + 1) // 2)
        return mask[:, :height, :width]


class MaskSample2(MaskSample2Base):
    def get_mask_four_parts(self, channel, height, width):
        assert not isinstance(channel, int) or channel % 4 == 0
        m = self.get_one_channel_four_parts_mask(height, width)
        m = m.unsqueeze(1).expand(4, channel // 4, height, width)
        perms = [(0, 1, 2, 3), (3, 2, 1, 0), (2, 3, 0, 1), (1, 0, 3, 2)]
        mask_for_four_part = torch.stack([
            torch.cat([m[i] for i in perm], dim=0) for perm in perms
        ])
        return mask_for_four_part

    def forward(self, y: torch.Tensor, i: int) -> torch.Tensor:
        _, C, H, W = y.shape
        mask_list = self.get_mask_four_parts(C, H, W)
        return torch.where(mask_list[i], y, 0)


class Resample(Sample):
    @staticmethod
    def merge_mask_parts(mask, channel):
        # Part p of the mask covers the channels [p*C//P, (p+1)*C//P).
        return (
            mask[:, None]
            .expand(-1, channel // mask.shape[0], -1, -1)
            .flatten(0, 1)
        )

    def get_s1_s2_with_mask(self, y, mask):
        # Every part keeps the same fraction of positions, so a single select
        # on the merged mask matches the per-part chunks.
        mask = self.merge_mask_parts(mask, y.shape[1])
        batch, height, width = y.shape[0], y.shape[2] // 2, y.shape[3] // 2
        return y.masked_select(mask).view(batch, -1, height, width)

    def recon_for_s2_s3(self, y_curr, mask):
        mask = self.merge_mask_parts(mask, y_curr.shape[1]).expand(
            y_curr.shape[0], -1, -1, -1
        )
        return torch.masked_scatter(
            y_curr.new_zeros(mask.shape), mask, y_curr.reshape(-1)
        )


class Downsample21(MaskSample2Base, Resample):
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        _, _, H, W = y.shape
        mask_list_rec_s2 = self.get_one_channel_four_parts_mask(H, W)
        return self.get_s1_s2_with_mask(y, mask_list_rec_s2)


class Upsample12(MaskSample2Base, Resample):
    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        _, _, H, W = y_hat.shape
        mask_list_rec_s2 = self.get_one_channel_four_parts_mask(2 * H, 2 * W)
        y_hat = self.recon_for_s2_s3(y_hat, mask_list_rec_s2)
        return y_hat


class DownsampleBase(Resample):
    def __init__(self):
        super().__init__()
        self.s2_mask = nn.Parameter(
            torch.tensor(
                (
                    ((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0)),
                    ((0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)),
                    ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)),
                    ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0)),
                    ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0)),
                    ((0, 0, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0)),
                    ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 1, 0)),
                    ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (1, 0, 0, 0)),
                ),
                dtype=torch.bool,
            ),
            requires_grad=False,
        )

    def get_mask_for_s2(self, height, width):
        mask = self.s2_mask.repeat(1, (height + 1) // 4, (width + 1) // 4)
        indices_1 = [0, 2, 4, 6, 1, 3, 5, 7]
        indices_2 = [1, 3, 5, 7, 0, 2, 4, 6]
        mask = mask[:, :height, :width]
        return mask[indices_1] | mask[indices_2]


class Downsample32(DownsampleBase):
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        _, _, H, W = y.shape
        mask_list_s2 = self.get_mask_for_s2(H, W)
        return self.get_s1_s2_with_mask(y, mask_list_s2)


class Upsample23(DownsampleBase):
    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        _, _, H, W = y_hat.shape
        mask_list_s2 = self.get_mask_for_s2(2 * H, 2 * W)
        y_hat = self.recon_for_s2_s3(y_hat, mask_list_s2)
        return y_hat


class Resample33(DownsampleBase):
    def get_s3_hyper_with_mask(self, common_params, mask):
        merged = self.merge_mask_parts(mask, common_params.shape[1])
        return common_params.masked_fill(merged, 0.0)

    def forward(self, scales_all: torch.Tensor) -> torch.Tensor:
        _, _, H, W = scales_all.shape
        mask_list_s2 = self.get_mask_for_s2(H, W)
        scales_s3 = self.get_s3_hyper_with_mask(scales_all, mask_list_s2)
        return scales_s3


class Downsample_32(MaskSample2Base, DownsampleBase):
    def __init__(self):
        super().__init__()
        self.s1_mask = nn.Parameter(
            torch.tensor(
                (
                    ((1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
                    ((0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
                    ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
                    ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0)),
                    ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
                    ((0, 0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
                    ((0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
                    ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
                ),
                dtype=torch.bool,
            ),
            requires_grad=False,
        )

    def get_mask_for_s1(self, height, width):
        mask = self.s1_mask.repeat(1, (height + 1) // 4, (width + 1) // 4)
        indices = [0, 2, 4, 6, 1, 3, 5, 7]
        return mask[:, :height, :width][indices]

    def get_s2_hyper_with_mask(self, y, mask_s1, mask_s2, mask_rec_s2):
        mask = self.merge_mask_parts(mask_s2 & ~mask_s1, y.shape[1]).expand_as(
            y
        )
        # expand rather than a hand-built shape: a traced shape is an uniterable Proxy
        mask_rec_s2 = self.merge_mask_parts(mask_rec_s2, y.shape[1]).expand(
            y.shape[0], -1, -1, -1
        )
        return torch.masked_scatter(
            y.new_zeros(mask_rec_s2.shape), ~mask_rec_s2, y[mask].reshape(-1)
        )

    def forward(self, scales_all: torch.Tensor) -> torch.Tensor:
        _, _, H, W = scales_all.shape
        mask_list_s1 = self.get_mask_for_s1(H, W)
        mask_list_s2 = self.get_mask_for_s2(H, W)
        mask_list_rec_s2 = self.get_one_channel_four_parts_mask(H // 2, W // 2)
        scales_s2 = self.get_s2_hyper_with_mask(
            scales_all, mask_list_s1, mask_list_s2, mask_list_rec_s2
        )
        return scales_s2
