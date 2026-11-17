import math
import torch
from torch import nn
import torch.nn.functional as F

from .base import BB as basemodel
from ..layers import PConvRB, conv2x2_down, deconv2x2_up, DWConvRB, conv1x1, conv4x4_down, deconv4x4_up

class g_a(nn.Module):
    def __init__(self):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4
        
        self.branch = nn.Sequential(
            conv4x4_down(3,96),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(96,192),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(192,384),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(384,320),
        )

    def forward(self, x):
        return self.branch(x)

class g_s(nn.Module):
    def __init__(self):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4

        self.branch = nn.Sequential(
            deconv2x2_up(320,384),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(384, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv2x2_up(384,192),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(192, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv2x2_up(192,96),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(96, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv4x4_up(96,3),
        )

    def forward(self, x):
        out = self.branch(x)
        return out
    
class h_a(nn.Module):
    def __init__(self):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4

        self.branch = nn.Sequential(
            PConvRB(320, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(320,256),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            conv2x2_down(256,256),
        )

    def forward(self, x):
        out = self.branch(x)
        return out

class h_s(nn.Module):
    def __init__(self):
        super().__init__()
        
        mlp_ratio = 4
        partial_ratio = 4

        self.branch = nn.Sequential(
            deconv2x2_up(256,256),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            PConvRB(256, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
            deconv2x2_up(256,320*2),
            PConvRB(320*2, mlp_ratio = mlp_ratio, partial_ratio = partial_ratio),
        )

    def forward(self, x):
        out = self.branch(x)
        return out

class y_spatial_prior_s1_s2(nn.Module):
    def __init__(self, M):
        super().__init__()
        
        self.branch_1 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
        )
        self.branch_2 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
            conv1x1(3*M,2*M),
        )

    def forward(self, x):
        return self.branch_2(self.branch_1(x))

class y_spatial_prior_s3(nn.Module):
    def __init__(self, M):
        super().__init__()
        
        self.branch_1 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
            DWConvRB(M*3),
            DWConvRB(M*3),
        )
        self.branch_2 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
            DWConvRB(M*3),
            conv1x1(3*M,2*M),
        )

    def forward(self, x):
        return self.branch_2(self.branch_1(x))

class CrossAttentionCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, window_size, kernel_size, num_heads=32):
        super(CrossAttentionCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        self.conv_q = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.conv_out = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

        self.window_size = window_size

    def forward(self, x_t, h_prev):
        x_t_init = x_t
        win = self.window_size
        b, _, h, w = x_t_init.shape

        def to_windows(t):
            return t.reshape(b, t.shape[1], h // win, win, w // win, win).permute(0, 2, 4, 1, 3, 5).reshape(-1, t.shape[1], win, win)

        x_t = to_windows(x_t)
        h_prev = to_windows(h_prev)
        batch_size, C, H, W = x_t.size()
        q = self.conv_q(x_t)
        k = self.conv_k(h_prev)
        v = self.conv_v(h_prev)

        q = q.view(batch_size, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(batch_size, self.num_heads, self.head_dim, H * W)
        v = v.view(batch_size, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        attn_scores = torch.matmul(q, k) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous().view(batch_size, self.hidden_dim, H, W)
        attn_output = attn_output.reshape(b, h // win, w // win, self.hidden_dim, win, win).permute(0, 3, 1, 4, 2, 5).reshape(b, self.hidden_dim, h, w)

        h_t = attn_output + self.conv_out(x_t_init)

        return h_t

class HPCM(basemodel):
    def __init__(self, M=320, N=256):
        super().__init__(N)
        
        self.g_a = g_a()
        self.g_s = g_s()

        self.h_a = h_a()
        self.h_s = h_s()

        self.y_spatial_prior_adaptor_list_s1 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(1))
        self.y_spatial_prior_adaptor_list_s2 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(3))
        self.y_spatial_prior_adaptor_list_s3 = nn.ModuleList(conv1x1(3*M,3*M) for _ in range(6))
        self.y_spatial_prior_s1_s2 = y_spatial_prior_s1_s2(M)
        self.y_spatial_prior_s3 = y_spatial_prior_s3(M)

        self.attn_s1 = CrossAttentionCell(320*2, 320*2, window_size=4, kernel_size=1)
        self.attn_s2 = CrossAttentionCell(320*2, 320*2, window_size=8, kernel_size=1)
        self.attn_s3 = CrossAttentionCell(320*2, 320*2, window_size=8, kernel_size=1)
        
        self.context_net = nn.ModuleList(conv1x1(2*M,2*M) for _ in range(2))

    def forward(self, x, training=None):
        if training is None:
            training=self.training 
        
        y = self.g_a(x)
        z = self.h_a(y)
        
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        params = self.h_s(z_hat)
        y_res, y_hat, scales_y = self.forward_hpcm(y, params, 
                                        self.y_spatial_prior_adaptor_list_s1, self.y_spatial_prior_s1_s2, 
                                        self.y_spatial_prior_adaptor_list_s2, self.y_spatial_prior_s1_s2, 
                                        self.y_spatial_prior_adaptor_list_s3, self.y_spatial_prior_s3, 
                                        self.context_net, 
                                        )

        x_hat = self.g_s(y_hat)
        
        y_likelihoods = self.entropy_conditional.likelihoods(y_res, scales_y)
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
    def forward_hpcm(self, y, common_params, 
                              y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                              y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                              y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                              context_net, write=False):
        ############### 2-step scale-1 (s1) (4× downsample) coding
        y_s1 = self.downsample_31(y)
        y_s2 = self.downsample32(y)

        # get scales_s1 and means_s1, same as getting s1 and s2
        scales_all, means_all = common_params.chunk(2,1)
        scales_s1 = self.downsample_31(scales_all)
        means_s1 = self.downsample_31(means_all)
        common_params_s1 = torch.cat((scales_s1, means_s1), dim=1)
        context_next = common_params_s1

        y_coded_list_s1 = []
        scale_list_s1 = []
        y_hat = None
        scales_hat = None
        y_res = None

        for i in range(2):
            if i == 0:
                y_coded, y_hat_step, s_hat = self.process_with_mask(y_s1, scales_s1, means_s1, self.masksample3, i, write)
            else:
                y_hat_so_far = y_hat
                params = torch.cat((context_next, y_hat_so_far), dim=1)
                context = y_spatial_prior_s1(y_spatial_prior_adaptor_list_s1[i - 1](params))
                context_next = self.attn_s1(context, context_next)
                scales, means = context.chunk(2, 1)
                y_coded, y_hat_step, s_hat = self.process_with_mask(y_s1, scales, means, self.masksample3, i, write)
            if write:
                y_coded_list_s1.append(y_coded)
            scale_list_s1.append(s_hat)
            y_hat = y_hat_step if y_hat is None else y_hat + y_hat_step
            scales_hat = s_hat if scales_hat is None else scales_hat + s_hat
            if not write:
                y_res = y_coded if y_res is None else y_res + y_coded

        if write:
            y_q_write_list_s1 = [self.combine_for_writing_s1(y_coded) for y_coded in y_coded_list_s1]
            scales_hat_write_list_s1 = [self.combine_for_writing_s1(s) for s in scale_list_s1]

        # up-scaling to s2
        y_hat = self.upsample12(y_hat)
        scales_hat = self.upsample12(scales_hat)
        if not write:
            y_res = self.upsample12(y_res)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.upsample12(context_next_scales)
        context_next_means = self.upsample12(context_next_means)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 4-step scale-2 (s2) (2× downsample) coding

        scales_s2 = self.downsample_32(scales_all)
        means_s2 = self.downsample_32(means_all)
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context += common_params_s2
        context_next = context_net[0](context)
        
        y_coded_list_s2 = []
        scale_list_s2 = [scales_hat]

        for i in range(3):
            y_hat_so_far = y_hat
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = y_spatial_prior_s2(y_spatial_prior_adaptor_list_s2[i - 1](params))
            context_next = self.attn_s2(context, context_next)
            scales, means = context.chunk(2, 1)
            y_coded, y_hat_step, s_hat = self.process_with_mask(y_s2, scales, means, self.masksample2, i + 1, write)
            if write:
                y_coded_list_s2.append(y_coded)
            scale_list_s2.append(s_hat)
            y_hat = y_hat + y_hat_step
            scales_hat = scales_hat + s_hat
            if not write:
                y_res = y_res + y_coded

        if write:
            y_q_write_list_s2 = [self.combine_for_writing_s2(y_coded) for y_coded in y_coded_list_s2]
            scales_hat_write_list_s2 = [self.combine_for_writing_s2(s) for s in scale_list_s2[1:]]

        # up-scaling to s3
        y_hat = self.upsample23(y_hat)
        scales_hat = self.upsample23(scales_hat)
        if not write:
            y_res = self.upsample23(y_res)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.upsample23(context_next_scales)
        context_next_means = self.upsample23(context_next_means)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 8-step scale-3 (s3) coding

        scales_s3 = self.resample33(scales_all)
        means_s3 = self.resample33(means_all)
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context += common_params_s3
        context_next = context_net[1](context)

        y_coded_list_s3 = []
        scale_list_s3 = [scales_hat]

        for i in range(6):
            y_hat_so_far = y_hat
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = y_spatial_prior_s3(y_spatial_prior_adaptor_list_s3[i - 1](params))
            context_next = self.attn_s3(context, context_next)
            scales, means = context.chunk(2, 1)
            y_coded, y_hat_step, s_hat = self.process_with_mask(y, scales, means, self.masksample1, i + 2, write)
            if write:
                y_coded_list_s3.append(y_coded)
            scale_list_s3.append(s_hat)
            y_hat = y_hat + y_hat_step
            scales_hat = scales_hat + s_hat
            if not write:
                y_res = y_res + y_coded

        if write:
            y_q_write_list_s3 = [self.combine_for_writing_s3(y_coded) for y_coded in y_coded_list_s3]
            scales_hat_write_list_s3 = [self.combine_for_writing_s3(s) for s in scale_list_s3[1:]]

            return y_q_write_list_s1 + y_q_write_list_s2 + y_q_write_list_s3, scales_hat_write_list_s1 + scales_hat_write_list_s2 + scales_hat_write_list_s3

        return y_res, y_hat, scales_hat
    
    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)
        y_q_write_list, scales_hat_write_list = self.compress_hpcm(
            y, params,
            self.y_spatial_prior_adaptor_list_s1, self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s2, self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s3, self.y_spatial_prior_s3,
            self.context_net,
        )

        indexes = [self.entropy_conditional.build_indexes(s) for s in scales_hat_write_list]
        y_strings = self.entropy_conditional.compress_parts(y_q_write_list, indexes)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)

        params = self.h_s(z_hat)
        streams = self.entropy_conditional.open_streams(strings[0])
        y_hat = self.decompress_hpcm(
            params,
            self.y_spatial_prior_adaptor_list_s1, self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s2, self.y_spatial_prior_s1_s2,
            self.y_spatial_prior_adaptor_list_s3, self.y_spatial_prior_s3,
            self.context_net, streams,
        )

        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def compress_hpcm(self, y, common_params, 
                              y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                              y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                              y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                              context_net
                              ):
        return self.forward_hpcm(y, common_params, 
                              y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                              y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                              y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                              context_net, write=True
                              )

    def decompress_hpcm(self, common_params, 
                                y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                                y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                                y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                                context_net, streams
                                ):
        ############### 2-step resolution-1 (s1) (4× downsample) coding

        # get scales_s1 and means_s1
        scales_all, means_all = common_params.chunk(2,1)
        scales_s1 = self.downsample_31(scales_all)
        means_s1 = self.downsample_31(means_all)
        common_params_s1 = torch.cat((scales_s1, means_s1), dim=1)
        context_next = common_params_s1

        for i in range(2):
            if i == 0:
                scales_r = self.combine_for_writing_s1(self.masksample3(scales_s1, i))
                indexes_r = self.entropy_conditional.build_indexes(scales_r)
                y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
                y_hat_curr_step = self.masksample3(
                    torch.cat([y_q_r for _ in range(2)], dim=1) + means_s1, i
                )
                y_hat_so_far = y_hat_curr_step
            else:
                params = torch.cat((context_next, y_hat_so_far), dim=1)
                context = y_spatial_prior_s1(y_spatial_prior_adaptor_list_s1[i - 1](params))
                context_next = self.attn_s1(context, context_next)
                scales, means = context.chunk(2, 1)
                scales_r = self.combine_for_writing_s1(self.masksample3(scales, i))
                indexes_r = self.entropy_conditional.build_indexes(scales_r)
                y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
                y_hat_curr_step = self.masksample3(
                    torch.cat([y_q_r for _ in range(2)], dim=1) + means, i
                )
                y_hat_so_far = y_hat_so_far + y_hat_curr_step
        
        y_hat_so_far = self.upsample12(y_hat_so_far)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.upsample12(context_next_scales)
        context_next_means = self.upsample12(context_next_means)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 4-step resolution-2 (s2) (2× downsample) coding
        scales_s2 = self.downsample_32(scales_all)
        means_s2 = self.downsample_32(means_all)
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context += common_params_s2
        context_next = context_net[0](context)

        for i in range(3):
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = y_spatial_prior_s2(y_spatial_prior_adaptor_list_s2[i - 1](params))
            context_next = self.attn_s2(context, context_next)
            scales, means = context.chunk(2, 1)
            scales_r = self.combine_for_writing_s2(self.masksample2(scales, i + 1))
            indexes_r = self.entropy_conditional.build_indexes(scales_r)
            y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
            y_hat_curr_step = self.masksample2(
                torch.cat([y_q_r for _ in range(4)], dim=1) + means, i + 1
            )
            y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat_so_far = self.upsample23(y_hat_so_far)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.upsample23(context_next_scales)
        context_next_means = self.upsample23(context_next_means)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 8-step resolution-3 (s3) coding
        scales_s3 = self.resample33(scales_all)
        means_s3 = self.resample33(means_all)
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context += common_params_s3
        context_next = context_net[1](context)

        for i in range(6):
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context = y_spatial_prior_s3(y_spatial_prior_adaptor_list_s3[i - 1](params))
            context_next = self.attn_s3(context, context_next)
            scales, means = context.chunk(2, 1)
            scales_r = self.combine_for_writing_s3(self.masksample1(scales, i + 2))
            indexes_r = self.entropy_conditional.build_indexes(scales_r)
            y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
            y_hat_curr_step = self.masksample1(
                torch.cat([y_q_r for _ in range(8)], dim=1) + means, i + 2
            )
            y_hat_so_far = y_hat_so_far + y_hat_curr_step
        
        y_hat = y_hat_so_far

        return y_hat
