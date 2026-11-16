import torch
from torch import nn

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
        )
        self.branch_2 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
            conv1x1(3*M,2*M),
        )

    def forward(self, x):
        return self.branch_2(self.branch_1(x))

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
        # get y_s2 first
        y_s2 = self.downsample32(y)
        # get y_s1 from y_s2
        y_s1 = self.downsample21(y_s2)

        # get scales_s1 and means_s1, same as getting s1 and s2
        scales_all, means_all = common_params.chunk(2,1)
        scales_s2 = self.downsample32(scales_all)
        scales_s1 = self.downsample21(scales_s2)
        means_s2 = self.downsample32(means_all)
        means_s1 = self.downsample21(means_s2)
        common_params_s1 = torch.cat((scales_s1, means_s1), dim=1)
        context_next = common_params_s1

        mask_list = self.masksample3(torch.ones_like(scales_s1[:1], dtype=torch.bool)).bool()
        y_coded_list_s1 = []
        y_hat_list_s1 = []
        scale_list_s1 = []

        for i in range(2):
            if i == 0:
                y_coded, y_hat_step, s_hat = self.process_with_mask(y_s1, scales_s1, means_s1, mask_list[i], write)
            else:
                y_hat_so_far = torch.sum(torch.stack(y_hat_list_s1), dim=0)
                params = torch.cat((context_next, y_hat_so_far), dim=1)
                context_next = y_spatial_prior_s1(y_spatial_prior_adaptor_list_s1[i - 1](params))
                scales, means = context_next.chunk(2, 1)
                y_coded, y_hat_step, s_hat = self.process_with_mask(y_s1, scales, means, mask_list[i], write)
            y_coded_list_s1.append(y_coded)
            y_hat_list_s1.append(y_hat_step)
            scale_list_s1.append(s_hat)

        y_hat = torch.sum(torch.stack(y_hat_list_s1), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s1), dim=0)

        if write:
            y_q_write_list_s1 = [self.combine_for_writing_s1(y_coded) for y_coded in y_coded_list_s1]
            scales_hat_write_list_s1 = [self.combine_for_writing_s1(s) for s in scale_list_s1]
        else:
            y_res = torch.sum(torch.stack(y_coded_list_s1), dim=0)

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
        
        mask_list = self.masksample2(torch.ones_like(scales_s2[:1], dtype=torch.bool)).bool()[1:]
        y_coded_list_s2 = [] if write else [y_res]
        y_hat_list_s2 = [y_hat]
        scale_list_s2 = [scales_hat]

        for i in range(3):
            y_hat_so_far = torch.sum(torch.stack(y_hat_list_s2), dim=0)
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context_next = y_spatial_prior_s2(y_spatial_prior_adaptor_list_s2[i - 1](params))
            scales, means = context_next.chunk(2, 1)
            y_coded, y_hat_step, s_hat = self.process_with_mask(y_s2, scales, means, mask_list[i], write)
            y_coded_list_s2.append(y_coded)
            y_hat_list_s2.append(y_hat_step)
            scale_list_s2.append(s_hat)

        y_hat = torch.sum(torch.stack(y_hat_list_s2), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s2), dim=0)

        if write:
            y_q_write_list_s2 = [self.combine_for_writing_s2(y_coded) for y_coded in y_coded_list_s2]
            scales_hat_write_list_s2 = [self.combine_for_writing_s2(s) for s in scale_list_s2[1:]]
        else:
            y_res = torch.sum(torch.stack(y_coded_list_s2), dim=0)

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

        mask_list = self.masksample1(torch.ones_like(scales_all[:1], dtype=torch.bool)).bool()[2:]
        y_coded_list_s3 = [] if write else [y_res]
        y_hat_list_s3 = [y_hat]
        scale_list_s3 = [scales_hat]

        for i in range(6):
            y_hat_so_far = torch.sum(torch.stack(y_hat_list_s3), dim=0)
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context_next = y_spatial_prior_s3(y_spatial_prior_adaptor_list_s3[i - 1](params))
            scales, means = context_next.chunk(2, 1)
            y_coded, y_hat_step, s_hat = self.process_with_mask(y, scales, means, mask_list[i], write)
            y_coded_list_s3.append(y_coded)
            y_hat_list_s3.append(y_hat_step)
            scale_list_s3.append(s_hat)

        y_hat = torch.sum(torch.stack(y_hat_list_s3), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s3), dim=0)

        if write:
            y_q_write_list_s3 = [self.combine_for_writing_s3(y_coded) for y_coded in y_coded_list_s3]
            scales_hat_write_list_s3 = [self.combine_for_writing_s3(s) for s in scale_list_s3[1:]]

            return y_q_write_list_s1 + y_q_write_list_s2 + y_q_write_list_s3, scales_hat_write_list_s1 + scales_hat_write_list_s2 + scales_hat_write_list_s3

        y_res = torch.sum(torch.stack(y_coded_list_s3), dim=0)
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
        scales_s2 = self.downsample32(scales_all)
        scales_s1 = self.downsample21(scales_s2)
        means_s2 = self.downsample32(means_all)
        means_s1 = self.downsample21(means_s2)
        common_params_s1 = torch.cat((scales_s1, means_s1), dim=1)
        context_next = common_params_s1

        mask_list = self.masksample3(torch.ones_like(scales_s1[:1], dtype=torch.bool)).bool()

        for i in range(2):
            if i == 0:
                scales_r = self.combine_for_writing_s1(torch.where(mask_list[i], scales_s1, 0))
                indexes_r = self.entropy_conditional.build_indexes(scales_r)
                y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
                y_hat_curr_step = torch.where(mask_list[i], torch.cat([y_q_r for _ in range(2)], dim=1) + means_s1, 0)
                y_hat_so_far = y_hat_curr_step
            else:
                params = torch.cat((context_next, y_hat_so_far), dim=1)
                context_next = y_spatial_prior_s1(y_spatial_prior_adaptor_list_s1[i - 1](params))
                scales, means = context_next.chunk(2, 1)
                scales_r = self.combine_for_writing_s1(torch.where(mask_list[i], scales, 0))
                indexes_r = self.entropy_conditional.build_indexes(scales_r)
                y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
                y_hat_curr_step = torch.where(mask_list[i], torch.cat([y_q_r for _ in range(2)], dim=1) + means, 0)
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

        mask_list = self.masksample2(torch.ones_like(scales_s2[:1], dtype=torch.bool)).bool()[1:]

        for i in range(3):
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context_next = y_spatial_prior_s2(y_spatial_prior_adaptor_list_s2[i - 1](params))
            scales, means = context_next.chunk(2, 1)
            scales_r = self.combine_for_writing_s2(torch.where(mask_list[i], scales, 0))
            indexes_r = self.entropy_conditional.build_indexes(scales_r)
            y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
            y_hat_curr_step = torch.where(mask_list[i], torch.cat([y_q_r for _ in range(4)], dim=1) + means, 0)
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

        mask_list = self.masksample1(torch.ones_like(scales_all[:1], dtype=torch.bool)).bool()[2:]

        for i in range(6):
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context_next = y_spatial_prior_s3(y_spatial_prior_adaptor_list_s3[i - 1](params))
            scales, means = context_next.chunk(2, 1)
            scales_r = self.combine_for_writing_s3(torch.where(mask_list[i], scales, 0))
            indexes_r = self.entropy_conditional.build_indexes(scales_r)
            y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
            y_hat_curr_step = torch.where(mask_list[i], torch.cat([y_q_r for _ in range(8)], dim=1) + means, 0)
            y_hat_so_far = y_hat_so_far + y_hat_curr_step
        
        y_hat = y_hat_so_far

        return y_hat
