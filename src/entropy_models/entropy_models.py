"""Generalized-normal (GGM) entropy models on top of CompressAI's `EntropyModel`.

The latent is quantized on an integer grid relative to a per-channel mean, and
only the residual symbols are arithmetic coded — the same scheme CompressAI's
`EntropyBottleneck`/`GaussianConditional` implement with a Gaussian/spline
distribution, here with a generalized normal of shape `beta`.
"""

from collections.abc import Callable

import scipy.stats
import torch
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyModel
from compressai.ops import LowerBound
from torch import Tensor
from torchao.quantization.quant_primitives import _Round

ste_round: Callable[[torch.Tensor], torch.Tensor] = _Round.apply


class generalnormalcdf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, beta, y):
        ctx.absy = torch.pow(torch.abs(y), beta)
        ctx.beta = beta
        output = (1 + torch.sign(y) * torch.special.gammainc(1 / beta, ctx.absy)) / 2
        ctx.device = y.device
        return output

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.absy
        beta = ctx.beta
        y_grad = torch.exp(-y - torch.special.gammaln(1 / beta)) * beta / 2
        return None, grad_output * y_grad


class GGMEntropyModel(EntropyModel):
    r"""Base class for the generalized-normal discrete entropy models.

    Args:
        beta: shape of the generalized normal distribution
        scale_bound: lower bound applied to every scale, both when computing
            likelihoods and when building the CDF tables
        tail_mass: probability mass reserved for the CDF tails
    """

    def __init__(self, beta: float = 1.5, scale_bound: float = 0.12,
                 tail_mass: float = 1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scale_lower_bound = LowerBound(scale_bound)
        self.tail_mass = float(tail_mass)
        self.register_buffer("beta", torch.Tensor([beta]))

    def _standardized_quantile(self, quantile) -> float:
        return scipy.stats.gennorm.ppf(quantile, float(self.beta))

    def _cdf(self, values: Tensor) -> Tensor:
        return generalnormalcdf.apply(self.beta.to(values.device), values)

    def _likelihood(self, inputs: Tensor, scales: Tensor,
                    means: Tensor | None = None) -> Tensor:
        half = float(0.5)
        values = inputs - means if means is not None else inputs
        scales = self.scale_lower_bound(scales)
        upper = self._cdf((values + half) / scales)
        lower = self._cdf((values - half) / scales)
        return upper - lower

    def likelihood(self, inputs: Tensor, scales: Tensor,
                   means: Tensor | None = None) -> Tensor:
        return self.likelihood_lower_bound(self._likelihood(inputs, scales, means))

    def quantize_noise(self, inputs: Tensor) -> Tensor:
        """Training-time stand-in for quantization: the symbol replaced by U(-0.5, 0.5).

        Not `self.quantize(inputs, "noise")`: compressai draws with
        `rand_like`, we draw with `uniform_`, and the two are not bitwise equal,
        so switching would move every training likelihood (and its gradient).
        """
        half = float(0.5)
        return inputs + torch.empty_like(inputs).uniform_(-half, half)

    def _build_tables(self, scales: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantized CDFs, one row per scale (channel, or scale-table entry).

        `gammainc` rounds differently on CPU and CUDA, and a CDF is only
        correct as built on one device, so the tables are always computed on
        CPU and moved to the module's device afterwards. That keeps `update()`
        independent of when the model is moved to a GPU. The scale lower bound
        is applied by value for the same reason: the incoming scales and this
        module need not share a device.
        """
        device = self.beta.device
        bound = float(self.scale_lower_bound.bound)
        scales = scales.detach().float().view(-1).clamp_min(bound).cpu()

        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(scales * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        samples = torch.arange(max_length, device=scales.device).int() - pmf_center[:, None]
        samples_scale = scales.unsqueeze(1)

        upper = self._cdf((samples + 0.5) / samples_scale)
        lower = self._cdf((samples - 0.5) / samples_scale)
        pmf = upper - lower
        tails = lower[:, :1] + (1 - upper[:, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tails, pmf_length, max_length)
        tables = (quantized_cdf, pmf_length + 2, -pmf_center)
        return tuple(t.to(device) for t in tables)

    def _apply_tables(self, scales: Tensor) -> None:
        self._quantized_cdf, self._cdf_length, self._offset = self._build_tables(scales)


class GGMEntropyBottleneck(GGMEntropyModel):
    r"""Entropy bottleneck with a generalized-normal prior over the hyper latent.

    `means` shifts the quantization grid, so `round(x - means)` is coded and
    `means` is added back on both sides.
    """

    def __init__(self, channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = int(channels)
        self.means = torch.nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scales = torch.nn.Parameter(torch.ones(1, channels, 1, 1))

    def update(self) -> bool:
        self._apply_tables(self.scales)
        return True

    def forward(self, x: Tensor, training: bool | None = None) -> tuple[Tensor, Tensor]:
        if training is None:
            training = self.training

        res = x - self.means
        if training:
            x_hat = ste_round(res) + self.means
            return x_hat, self.likelihood(self.quantize_noise(res), self.scales)

        symbols = torch.round(res)
        return symbols + self.means, self.likelihood(symbols, self.scales)

    def build_indexes(self, size: tuple[int, ...]) -> Tensor:
        dims = len(size)
        batch, channels = size[0], size[1]

        view_dims = [1] * dims
        view_dims[1] = -1
        indexes = torch.arange(channels).view(*view_dims)

        return indexes.repeat(batch, 1, *size[2:]).int()

    def _get_means(self, batch: int) -> Tensor:
        return self.means.detach().expand(batch, self.channels, 1, 1)

    def compress(self, x: Tensor) -> list[bytes]:
        indexes = self.build_indexes(x.size())
        return super().compress(x, indexes, self._get_means(x.size(0)))

    def decompress(self, strings: list[bytes], size: tuple[int, ...]) -> Tensor:
        batch = len(strings)
        output_size = (batch, self.channels, *size)
        indexes = self.build_indexes(output_size).to(self._quantized_cdf.device)
        return super().decompress(strings, indexes, torch.float32, self._get_means(batch))


class GGMConditional(GGMEntropyModel):
    r"""Per-pixel generalized-normal conditional model, indexed by a scale table.

    The progressive scheme codes every context step of an image into one rANS
    stream, so coding is exposed as an accumulate-then-flush `compress_parts`
    and a resumable decoder handle per image (`open_streams` + one
    `decompress_part` call per context step).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("scale_table", torch.Tensor())

    def update_scale_table(self, scale_table: Tensor) -> bool:
        self.scale_table = scale_table
        return self.update()

    def update(self) -> bool:
        if self.scale_table.numel() == 0:
            raise ValueError("No scale table. Run update_scale_table() first")
        self._apply_tables(self.scale_table)
        return True

    def build_indexes(self, scales: Tensor) -> Tensor:
        table = self.scale_table[:-1].to(scales.device).view(1, 1, 1, 1, -1)
        return (scales.unsqueeze(-1) > table).sum(-1)

    def forward(self, x: Tensor, means: Tensor | None = None,
                training: bool | None = None) -> Tensor:
        """Round x onto the integer grid, relative to means when given."""
        if training is None:
            training = self.training
        res = x - means if means is not None else x
        if training:
            return ste_round(res)
        return torch.round(res)

    def likelihoods(self, inputs: Tensor, scales: Tensor,
                    training: bool | None = None) -> Tensor:
        if training is None:
            training = self.training
        if training:
            return self.likelihood(self.quantize_noise(inputs), scales)
        return self.likelihood(torch.round(inputs), scales)

    def _coder_tables(self):
        cdf_length = self._cdf_length.reshape(-1).int().tolist()
        cdf = [row[:length] for row, length
               in zip(self._quantized_cdf.tolist(), cdf_length)]
        return cdf, cdf_length, self._offset.reshape(-1).int().tolist()

    def compress_parts(self, symbols_list: list[Tensor],
                       indexes_list: list[Tensor]) -> list[bytes]:
        """Code every context step of one image into a single rANS stream."""
        cdf, cdf_length, offset = self._coder_tables()
        strings = []
        for parts, index_parts in zip(zip(*symbols_list), zip(*indexes_list)):
            encoder = BufferedRansEncoder()
            for symbols, indexes in zip(parts, index_parts):
                encoder.encode_with_indexes(
                    symbols.reshape(-1).int().tolist(),
                    indexes.reshape(-1).int().tolist(),
                    cdf, cdf_length, offset,
                )
            strings.append(encoder.flush())
        return strings

    def open_streams(self, strings: list[bytes]) -> list[RansDecoder]:
        decoders = []
        for string in strings:
            decoder = RansDecoder()
            decoder.set_stream(string)
            decoders.append(decoder)
        return decoders

    def decompress_part(self, streams: list[RansDecoder], indexes: Tensor) -> Tensor:
        cdf, cdf_length, offset = self._coder_tables()
        values = [
            stream.decode_stream(
                part.reshape(-1).int().tolist(), cdf, cdf_length, offset)
            for stream, part in zip(streams, indexes)
        ]
        return torch.tensor(values, device=indexes.device,
                            dtype=torch.float32).reshape(indexes.size())
