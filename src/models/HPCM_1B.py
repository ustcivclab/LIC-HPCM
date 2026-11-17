from collections import OrderedDict

import torch
from torch import nn

from .base import BB as basemodel
from ..layers import PConvRB, conv2x2_down, deconv2x2_up, DWConvRB, conv1x1

class g_a(nn.Module):
    def __init__(self, chs, num_layers, M, partial_ratio=2, mlp_ratio=2, act=nn.LeakyReLU):
        super().__init__()
        self.g_a_down1 = conv2x2_down(3, chs[0])
        self.g_a_chunk1 = nn.Sequential(*[
            PConvRB(chs[0], partial_ratio, mlp_ratio) for i in range(num_layers[3])]
        )

        self.g_a_down2 = conv2x2_down(chs[0], chs[1])
        self.g_a_chunk2 = nn.Sequential(*[
            PConvRB(chs[1], partial_ratio, mlp_ratio) for i in range(num_layers[2])]
        )
        
        self.g_a_down3 = conv2x2_down(chs[1], chs[2])
        self.g_a_chunk3 = nn.Sequential(*[
            PConvRB(chs[2], partial_ratio, mlp_ratio) for i in range(num_layers[1])]
        )
        
        self.g_a_down4 = conv2x2_down(chs[2], chs[3])
        self.g_a_chunk4 = nn.Sequential(*[
            PConvRB(chs[3], partial_ratio, mlp_ratio) for i in range(num_layers[0])]
        )

    def forward(self, x):
        x = self.g_a_down1(x)
        x = self.g_a_chunk1(x)
        x = self.g_a_down2(x)
        x = self.g_a_chunk2(x)
        x = self.g_a_down3(x)
        x = self.g_a_chunk3(x)
        x = self.g_a_down4(x)
        y = self.g_a_chunk4(x)
        return y
    
class g_s(nn.Module):
    def __init__(self, chs, num_layers, M, partial_ratio=2, mlp_ratio=2, act=nn.LeakyReLU):
        super().__init__()
        
        self.g_s_chunk4 = nn.Sequential(*[
            PConvRB(chs[3], partial_ratio, mlp_ratio) for i in range(num_layers[3])]
        )
        self.g_s_up4 = deconv2x2_up(chs[3], chs[2])
        
        self.g_s_chunk3 = nn.Sequential(*[
            PConvRB(chs[2], partial_ratio, mlp_ratio) for i in range(num_layers[2])]
        )
        self.g_s_up3 = deconv2x2_up(chs[2], chs[1])
        
        self.g_s_chunk2 = nn.Sequential(*[
            PConvRB(chs[1], partial_ratio, mlp_ratio) for i in range(num_layers[1])]
        )
        self.g_s_up2 = deconv2x2_up(chs[1], chs[0])
        
        self.g_s_chunk1 = nn.Sequential(*[
            PConvRB(chs[0], partial_ratio, mlp_ratio) for i in range(num_layers[0])]
        )
        self.g_s_up1 = deconv2x2_up(chs[0], 3)

    def forward(self, y):
        y = self.g_s_chunk4(y)
        y = self.g_s_up4(y)
        y = self.g_s_chunk3(y)
        y = self.g_s_up3(y)
        y = self.g_s_chunk2(y)
        y = self.g_s_up2(y)
        y = self.g_s_chunk1(y)
        x = self.g_s_up1(y)
        return x

class y_spatial_prior_s1_s2(nn.Module):
    def __init__(self, M):
        super().__init__()
        
        self.branch_1 = nn.Sequential(
            DWConvRB(M*3),
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
            DWConvRB(M*2),
            DWConvRB(M*2),
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
            DWConvRB(M*3),
            DWConvRB(M*3),
        )
        self.branch_2 = nn.Sequential(
            DWConvRB(M*3),
            DWConvRB(M*3),
            DWConvRB(M*3),
            DWConvRB(M*3),
            conv1x1(3*M,2*M),
            DWConvRB(M*2),
            DWConvRB(M*2),
        )

    def forward(self, x):
        return self.branch_2(self.branch_1(x))

class HPCM(basemodel):
    def __init__(self, M=512, N=256):
        super().__init__(N)
        
        chs = [192, 384, 768, 512, 512]
        num_layers = [3,3,9,3,3]
        mlp_ratio = 4
        partial_ratio = 2
        
        self.g_a = g_a(chs, num_layers, M, partial_ratio, mlp_ratio)

        self.h_a = nn.Sequential(
            PConvRB(M, partial_ratio, mlp_ratio),
            PConvRB(M, partial_ratio, mlp_ratio),
            PConvRB(M, partial_ratio, mlp_ratio),
            PConvRB(M, partial_ratio, mlp_ratio),
            conv2x2_down(M, chs[4]),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            conv2x2_down(chs[4], N),
        )
        
        chs = [192, 384, 768, 512, 512]
        num_layers = [3,3,9,3,3]
        mlp_ratio = 4
        partial_ratio = 2

        self.h_s = nn.Sequential(
            deconv2x2_up(N, chs[4]),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            PConvRB(chs[4], partial_ratio, mlp_ratio),
            deconv2x2_up(chs[4], M*2),
            PConvRB(M*2, partial_ratio, mlp_ratio),
            PConvRB(M*2, partial_ratio, mlp_ratio),
            PConvRB(M*2, partial_ratio, mlp_ratio),
            PConvRB(M*2, partial_ratio, mlp_ratio),
        )

        self.g_s = g_s(chs, num_layers, M, partial_ratio, mlp_ratio)

        self.y_spatial_prior_adaptor_list_s1 = nn.ModuleList(
            nn.Sequential(
                DWConvRB(M*3), 
                DWConvRB(M*3), 
                DWConvRB(M*3), 
                DWConvRB(M*3), 
                DWConvRB(M*3), 
            ) 
            for _ in range(1)
        )
        self.y_spatial_prior_s1_s2 = y_spatial_prior_s1_s2(M)
        self.y_spatial_prior_adaptor_list_s2 = nn.ModuleList(
            nn.Sequential(
                DWConvRB(M*3), 
                DWConvRB(M*3), 
                DWConvRB(M*3), 
                DWConvRB(M*3), 
                DWConvRB(M*3), 
            ) 
            for _ in range(3)
        )
        self.y_spatial_prior_adaptor_list_s3 = nn.ModuleList(
            nn.Sequential(
                DWConvRB(M*3), 
                DWConvRB(M*3), 
                DWConvRB(M*3), 
                DWConvRB(M*3), 
                DWConvRB(M*3), 
                DWConvRB(M*3), 
                DWConvRB(M*3), 
            ) 
            for _ in range(6)
        )
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

    def forward_hpcm(self, y, common_params, 
                              y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                              y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                              y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                              context_net, write=False):
        ############### 2-step resolution-1 (s1) (4× downsample) coding
        y_s1 = self.downsample_31(y)
        y_s2 = self.downsample32(y)

        # same as getting s1 and s2
        scales_all, means_all = common_params.chunk(2,1)
        scales = self.downsample_31(scales_all)
        means = self.downsample_31(means_all)
        common_params_s1 = torch.cat((scales, means), dim=1)
        context = common_params_s1

        y_coded_list_s1 = []
        scale_list_s1 = []
        y_hat = None
        scales_hat = None
        y_res = None

        for i in range(2):
            if i == 0:
                y_coded, y_hat_step, s_hat = self.process_with_mask(y_s1, scales, means, self.masksample3, i, write)
            else:
                y_hat_so_far = y_hat
                params = torch.cat((context, y_hat_so_far), dim=1)
                context = y_spatial_prior_s1(y_spatial_prior_adaptor_list_s1[i - 1](params))
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

        y_hat = self.upsample12(y_hat)
        scales_hat = self.upsample12(scales_hat)
        if not write:
            y_res = self.upsample12(y_res)

        context_next_scales, context_next_means = context.chunk(2, 1)
        context_next_scales = self.upsample12(context_next_scales)
        context_next_means = self.upsample12(context_next_means)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 4-step resolution-2 (s2) (2× downsample) coding

        scales_s2 = self.downsample_32(scales_all)
        means_s2 = self.downsample_32(means_all)
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context += common_params_s2
        context = context_net[0](context)
        
        y_coded_list_s2 = []
        scale_list_s2 = [scales_hat]

        for i in range(3):
            y_hat_so_far = y_hat
            params = torch.cat((context, y_hat_so_far), dim=1)
            context = y_spatial_prior_s2(y_spatial_prior_adaptor_list_s2[i - 1](params))
            scales, means = context.chunk(2, 1)
            y_coded, y_hat_step, s_hat = self.process_with_mask(y_s2, scales, means, self.masksample2, i + 1, write)
            if write:
                y_coded_list_s2.append(y_coded)
            scale_list_s2.append(s_hat)
            y_hat = y_hat + y_hat_step
            scales_hat = scales_hat + s_hat
            if not write:
                y_res = y_res + y_coded

        # TODO
        if write:
            y_q_write_list_s2 = [self.combine_for_writing_s2(y_coded) for y_coded in y_coded_list_s2]
            scales_hat_write_list_s2 = [self.combine_for_writing_s2(s) for s in scale_list_s2[1:]]

        y_hat = self.upsample23(y_hat)
        scales_hat = self.upsample23(scales_hat)
        if not write:
            y_res = self.upsample23(y_res)

        context_next_scales, context_next_means = context.chunk(2, 1)
        context_next_scales = self.upsample23(context_next_scales)
        context_next_means = self.upsample23(context_next_means)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 8-step resolution-3 (s3) coding

        scales_s3 = self.resample33(scales_all)
        means_s3 = self.resample33(means_all)
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context += common_params_s3
        context = context_net[1](context)

        y_coded_list_s3 = []
        scale_list_s3 = [scales_hat]

        for i in range(6):
            y_hat_so_far = y_hat
            params = torch.cat((context, y_hat_so_far), dim=1)
            context = y_spatial_prior_s3(y_spatial_prior_adaptor_list_s3[i - 1](params))
            scales, means = context.chunk(2, 1)
            y_coded, y_hat_step, s_hat = self.process_with_mask(y, scales, means, self.masksample1, i + 2, write)
            if write:
                y_coded_list_s3.append(y_coded)
            scale_list_s3.append(s_hat)
            y_hat = y_hat + y_hat_step
            scales_hat = scales_hat + s_hat
            if not write:
                y_res = y_res + y_coded

        # TODO
        if write:
            y_q_write_list_s3 = [self.combine_for_writing_s3(y_coded) for y_coded in y_coded_list_s3]
            scales_hat_write_list_s3 = [self.combine_for_writing_s3(s) for s in scale_list_s3[1:]]

            return y_q_write_list_s1 + y_q_write_list_s2 + y_q_write_list_s3, scales_hat_write_list_s1 + scales_hat_write_list_s2 + scales_hat_write_list_s3

        return y_res, y_hat, scales_hat
    
    def compress_hpcm(self, y, common_params, 
                              y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                              y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                              y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                              context_net,
                              ):
        return self.forward_hpcm(y, common_params, 
                              y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                              y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                              y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                              context_net,
                              write=True
                              )

    def decompress_hpcm(self, common_params, 
                                y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                                y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                                y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                                context_net, streams
                                ):
        ############### 2-step scale-1 (s1) (4× downsample) decoding

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
                context_next = y_spatial_prior_s1(y_spatial_prior_adaptor_list_s1[i - 1](params))
                scales, means = context_next.chunk(2, 1)

                scales_r = self.combine_for_writing_s1(self.masksample3(scales, i))
                indexes_r = self.entropy_conditional.build_indexes(scales_r)
                y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
                y_hat_curr_step = self.masksample3(
                    torch.cat([y_q_r for _ in range(2)], dim=1) + means, i
                )
                y_hat_so_far = y_hat_so_far + y_hat_curr_step

        # up-scaling to s2
        y_hat_so_far = self.upsample12(y_hat_so_far)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.upsample12(context_next_scales)
        context_next_means = self.upsample12(context_next_means)
        context_next = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 4-step scale-2 (s2) (2× downsample) decoding

        scales_s2 = self.downsample_32(scales_all)
        means_s2 = self.downsample_32(means_all)
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context_next = context_next + common_params_s2
        context_next = context_net[0](context_next)

        for i in range(3):
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context_next = y_spatial_prior_s2(y_spatial_prior_adaptor_list_s2[i - 1](params))
            scales, means = context_next.chunk(2, 1)

            scales_r = self.combine_for_writing_s2(self.masksample2(scales, i + 1))
            indexes_r = self.entropy_conditional.build_indexes(scales_r)
            y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
            y_hat_curr_step = self.masksample2(
                torch.cat([y_q_r for _ in range(4)], dim=1) + means, i + 1
            )
            y_hat_so_far = y_hat_so_far + y_hat_curr_step

        # up-scaling to s3
        y_hat_so_far = self.upsample23(y_hat_so_far)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.upsample23(context_next_scales)
        context_next_means = self.upsample23(context_next_means)
        context_next = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 6-step scale-3 (s3) coding
        scales_s3 = self.resample33(scales_all)
        means_s3 = self.resample33(means_all)
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context_next = context_next + common_params_s3
        context_next = context_net[1](context_next)

        for i in range(6):
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context_next = y_spatial_prior_s3(y_spatial_prior_adaptor_list_s3[i - 1](params))
            scales, means = context_next.chunk(2, 1)

            scales_r = self.combine_for_writing_s3(self.masksample1(scales, i + 2))
            indexes_r = self.entropy_conditional.build_indexes(scales_r)
            y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
            y_hat_curr_step = self.masksample1(
                torch.cat([y_q_r for _ in range(8)], dim=1) + means, i + 2
            )
            y_hat_so_far = y_hat_so_far + y_hat_curr_step

        return y_hat_so_far
