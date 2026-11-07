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

    def forward(self, x, quant_step):
        return self.branch_2(self.branch_1(x)*quant_step)

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

    def forward(self, x, quant_step):
        return self.branch_2(self.branch_1(x)*quant_step)

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

        self.adaptive_params_list = nn.ParameterList([
            nn.Parameter(torch.ones((1, M*3, 1, 1)), requires_grad=True) for _ in range(10)
        ])

        self.context_net = nn.ModuleList(conv1x1(2*M,2*M) for _ in range(2))
        
    def forward(self, x, training=None):
        if training is None:
            training=self.training 
            
        y = self.g_a(x)
        z = self.h_a(y)
        
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        params = self.h_s(z_hat)
        y_res, y_q, y_hat, scales_y = self.forward_hpcm(y, params, 
                                        self.y_spatial_prior_adaptor_list_s1, self.y_spatial_prior_s1_s2, 
                                        self.y_spatial_prior_adaptor_list_s2, self.y_spatial_prior_s1_s2, 
                                        self.y_spatial_prior_adaptor_list_s3, self.y_spatial_prior_s3, 
                                        self.adaptive_params_list, self.context_net, 
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
            self.adaptive_params_list, self.context_net,
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
            self.adaptive_params_list, self.context_net, streams,
        )

        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}

    def forward_hpcm(self, y, common_params, 
                              y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                              y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                              y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                              adaptive_params_list, context_net, write=False):
        B, C, H, W = y.size()
        device = common_params.device

        ############### 2-step resolution-1 (s1) (4× downsample) coding
        # get s2 first
        mask_list_s2 = self.get_mask_for_s2(B, C, H, W, device)
        y_s2 = self.get_s1_s2_with_mask(y, mask_list_s2)
        # get s1 from s2
        mask_list_rec_s2 = self.get_mask_for_rec_s2(B, C, H // 2, W // 2, device)
        y_s1 = self.get_s1_s2_with_mask(y_s2, mask_list_rec_s2)

        # same as getting s1 and s2
        scales_all, means_all = common_params.chunk(2,1)
        scales_s2 = self.get_s1_s2_with_mask(scales_all, mask_list_s2)
        scales = self.get_s1_s2_with_mask(scales_s2, mask_list_rec_s2)
        means_s2 = self.get_s1_s2_with_mask(means_all, mask_list_s2)
        means = self.get_s1_s2_with_mask(means_s2, mask_list_rec_s2)
        common_params_s1 = torch.cat((scales, means), dim=1)
        context = common_params_s1

        mask_list = self.get_mask_two_parts(B, C, H // 4, W // 4, device)
        y_res_list_s1 = []
        y_q_list_s1 = []
        y_hat_list_s1 = []
        scale_list_s1 = []

        for i in range(2):
            if i == 0:
                y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y_s1, scales, means, mask_list[i])
                y_res_list_s1.append(y_res_0)
                y_q_list_s1.append(y_q_0)
                y_hat_list_s1.append(y_hat_0)
                scale_list_s1.append(s_hat_0)
            else:
                y_hat_so_far = torch.sum(torch.stack(y_hat_list_s1), dim=0)
                params = torch.cat((context, y_hat_so_far), dim=1)
                context = y_spatial_prior_s1(y_spatial_prior_adaptor_list_s1[i - 1](params), adaptive_params_list[i - 1])
                scales, means = context.chunk(2, 1)
                y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y_s1, scales, means, mask_list[i])
                y_res_list_s1.append(y_res_1)
                y_q_list_s1.append(y_q_1)
                y_hat_list_s1.append(y_hat_1)
                scale_list_s1.append(s_hat_1)
        
        y_res = torch.sum(torch.stack(y_res_list_s1), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s1), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s1), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s1), dim=0)

        if write:
            y_q_write_list_s1 = [self.combine_for_writing_s1(y_q_list_s1[i]) for i in range(len(y_q_list_s1))]
            scales_hat_write_list_s1 = [self.combine_for_writing_s1(scale_list_s1[i]) for i in range(len(scale_list_s1))]
        
        y_res = self.recon_for_s2_s3(y_res, mask_list_rec_s2)
        y_q = self.recon_for_s2_s3(y_q, mask_list_rec_s2)
        y_hat = self.recon_for_s2_s3(y_hat, mask_list_rec_s2)
        scales_hat = self.recon_for_s2_s3(scales_hat, mask_list_rec_s2)

        context_next_scales, context_next_means = context.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_rec_s2)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_rec_s2)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 4-step resolution-2 (s2) (2× downsample) coding

        mask_list_s1 = self.get_mask_for_s1(B, C, H, W, device)
        scales_s2 = self.get_s2_hyper_with_mask(scales_all, mask_list_s1, mask_list_s2, mask_list_rec_s2)
        means_s2 = self.get_s2_hyper_with_mask(means_all, mask_list_s1, mask_list_s2, mask_list_rec_s2)
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context += common_params_s2
        context = context_net[0](context)
        
        mask_list = self.get_mask_four_parts(B, C, H // 2, W // 2, device)[1:]
        y_res_list_s2 = []
        y_q_list_s2 = []
        y_hat_list_s2 = []
        scale_list_s2 = []
        y_res_list_s2.append(y_res)
        y_q_list_s2.append(y_q)
        y_hat_list_s2.append(y_hat)
        scale_list_s2.append(scales_hat)

        for i in range(3):
            y_hat_so_far = torch.sum(torch.stack(y_hat_list_s2), dim=0)
            params = torch.cat((context, y_hat_so_far), dim=1)
            context = y_spatial_prior_s2(y_spatial_prior_adaptor_list_s2[i - 1](params), adaptive_params_list[i + 1])
            scales, means = context.chunk(2, 1)
            y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y_s2, scales, means, mask_list[i])
            y_res_list_s2.append(y_res_1)
            y_q_list_s2.append(y_q_1)
            y_hat_list_s2.append(y_hat_1)
            scale_list_s2.append(s_hat_1)
        
        y_res = torch.sum(torch.stack(y_res_list_s2), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s2), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s2), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s2), dim=0)

        # TODO
        if write:
            y_q_write_list_s2 = [self.combine_for_writing_s2(y_q_list_s2[i]) for i in range(1, len(y_q_list_s2))]
            scales_hat_write_list_s2 = [self.combine_for_writing_s2(scale_list_s2[i]) for i in range(1, len(scale_list_s2))]
       
        y_res = self.recon_for_s2_s3(y_res, mask_list_s2)
        y_q = self.recon_for_s2_s3(y_q, mask_list_s2)
        y_hat = self.recon_for_s2_s3(y_hat, mask_list_s2)
        scales_hat = self.recon_for_s2_s3(scales_hat, mask_list_s2)

        context_next_scales, context_next_means = context.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_s2)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_s2)
        context = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 8-step resolution-3 (s3) coding

        scales_s3 = self.get_s3_hyper_with_mask(scales_all, mask_list_s2)
        means_s3 = self.get_s3_hyper_with_mask(means_all, mask_list_s2)
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context += common_params_s3
        context = context_net[1](context)

        mask_list = self.get_mask_eight_parts(B, C, H, W, device)[2:]
        y_res_list_s3 = []
        y_q_list_s3 = []
        y_hat_list_s3 = []
        scale_list_s3 = []
        y_res_list_s3.append(y_res)
        y_q_list_s3.append(y_q)
        y_hat_list_s3.append(y_hat)
        scale_list_s3.append(scales_hat)

        for i in range(6):
            y_hat_so_far = torch.sum(torch.stack(y_hat_list_s3), dim=0)
            params = torch.cat((context, y_hat_so_far), dim=1)
            context = y_spatial_prior_s3(y_spatial_prior_adaptor_list_s3[i - 1](params), adaptive_params_list[i + 4])
            scales, means = context.chunk(2, 1)
            y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_list[i])
            y_res_list_s3.append(y_res_1)
            y_q_list_s3.append(y_q_1)
            y_hat_list_s3.append(y_hat_1)
            scale_list_s3.append(s_hat_1)

        y_res = torch.sum(torch.stack(y_res_list_s3), dim=0)
        y_q = torch.sum(torch.stack(y_q_list_s3), dim=0)
        y_hat = torch.sum(torch.stack(y_hat_list_s3), dim=0)
        scales_hat = torch.sum(torch.stack(scale_list_s3), dim=0)

        # TODO
        if write:
            y_q_write_list_s3 = [self.combine_for_writing_s3(y_q_list_s3[i]) for i in range(1, len(y_q_list_s3))]
            scales_hat_write_list_s3 = [self.combine_for_writing_s3(scale_list_s3[i]) for i in range(1, len(scale_list_s3))]

            return y_q_write_list_s1 + y_q_write_list_s2 + y_q_write_list_s3, scales_hat_write_list_s1 + scales_hat_write_list_s2 + scales_hat_write_list_s3

        return y_res, y_q, y_hat, scales_hat
    
    def compress_hpcm(self, y, common_params, 
                              y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                              y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                              y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                              adaptive_params_list, context_net,
                              ):
        return self.forward_hpcm(y, common_params, 
                              y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                              y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                              y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                              adaptive_params_list, context_net,
                              write=True
                              )

    def decompress_hpcm(self, common_params, 
                                y_spatial_prior_adaptor_list_s1, y_spatial_prior_s1, 
                                y_spatial_prior_adaptor_list_s2, y_spatial_prior_s2, 
                                y_spatial_prior_adaptor_list_s3, y_spatial_prior_s3, 
                                adaptive_params_list, context_net, streams
                                ):
        scales_all, means_all = common_params.chunk(2,1)
        device = means_all.device
        B, C, H, W = means_all.size()

        ############### 2-step scale-1 (s1) (4× downsample) decoding
        mask_list_s2 = self.get_mask_for_s2(B, C, H, W, device)
        mask_list_rec_s2 = self.get_mask_for_rec_s2(B, C, H // 2, W // 2, device)

        scales_s2 = self.get_s1_s2_with_mask(scales_all, mask_list_s2)
        scales_s1 = self.get_s1_s2_with_mask(scales_s2, mask_list_rec_s2)
        means_s2 = self.get_s1_s2_with_mask(means_all, mask_list_s2)
        means_s1 = self.get_s1_s2_with_mask(means_s2, mask_list_rec_s2)
        common_params_s1 = torch.cat((scales_s1, means_s1), dim=1)
        context_next = common_params_s1

        mask_list = self.get_mask_two_parts(B, C, H // 4, W // 4, device)

        for i in range(2):
            if i == 0:
                scales_r = self.combine_for_writing_s1(torch.where(mask_list[i], scales_s1, 0.0))
                indexes_r = self.entropy_conditional.build_indexes(scales_r)
                y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
                y_hat_curr_step = torch.where(mask_list[i], torch.cat([y_q_r for _ in range(2)], dim=1) + means_s1, 0.0)
                y_hat_so_far = y_hat_curr_step
            else:
                params = torch.cat((context_next, y_hat_so_far), dim=1)
                context_next = y_spatial_prior_s1(y_spatial_prior_adaptor_list_s1[i - 1](params), adaptive_params_list[i - 1])
                scales, means = context_next.chunk(2, 1)

                scales_r = self.combine_for_writing_s1(torch.where(mask_list[i], scales, 0.0))
                indexes_r = self.entropy_conditional.build_indexes(scales_r)
                y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
                y_hat_curr_step = torch.where(mask_list[i], torch.cat([y_q_r for _ in range(2)], dim=1) + means, 0.0)
                y_hat_so_far = y_hat_so_far + y_hat_curr_step

        # up-scaling to s2
        y_hat_so_far = self.recon_for_s2_s3(y_hat_so_far, mask_list_rec_s2)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_rec_s2)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_rec_s2)
        context_next = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 4-step scale-2 (s2) (2× downsample) decoding
        mask_list_s1 = self.get_mask_for_s1(B, C, H, W, device)

        scales_s2 = self.get_s2_hyper_with_mask(scales_all, mask_list_s1, mask_list_s2, mask_list_rec_s2)
        means_s2 = self.get_s2_hyper_with_mask(means_all, mask_list_s1, mask_list_s2, mask_list_rec_s2)
        common_params_s2 = torch.cat((scales_s2, means_s2), dim=1)
        context_next = context_next + common_params_s2
        context_next = context_net[0](context_next)

        mask_list = self.get_mask_four_parts(B, C, H // 2, W // 2, device)[1:]

        for i in range(3):
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context_next = y_spatial_prior_s2(y_spatial_prior_adaptor_list_s2[i - 1](params), adaptive_params_list[i + 1])
            scales, means = context_next.chunk(2, 1)

            scales_r = self.combine_for_writing_s2(torch.where(mask_list[i], scales, 0.0))
            indexes_r = self.entropy_conditional.build_indexes(scales_r)
            y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
            y_hat_curr_step = torch.where(mask_list[i], torch.cat([y_q_r for _ in range(4)], dim=1) + means, 0.0)
            y_hat_so_far = y_hat_so_far + y_hat_curr_step

        # up-scaling to s3
        y_hat_so_far = self.recon_for_s2_s3(y_hat_so_far, mask_list_s2)

        context_next_scales, context_next_means = context_next.chunk(2, 1)
        context_next_scales = self.recon_for_s2_s3(context_next_scales, mask_list_s2)
        context_next_means = self.recon_for_s2_s3(context_next_means, mask_list_s2)
        context_next = torch.cat((context_next_scales, context_next_means), dim=1)

        ############### 6-step scale-3 (s3) coding
        scales_s3 = self.get_s3_hyper_with_mask(scales_all, mask_list_s2)
        means_s3 = self.get_s3_hyper_with_mask(means_all, mask_list_s2)
        common_params_s3 = torch.cat((scales_s3, means_s3), dim=1)
        context_next = context_next + common_params_s3
        context_next = context_net[1](context_next)

        mask_list = self.get_mask_eight_parts(B, C, H, W, device)[2:]

        for i in range(6):
            params = torch.cat((context_next, y_hat_so_far), dim=1)
            context_next = y_spatial_prior_s3(y_spatial_prior_adaptor_list_s3[i - 1](params), adaptive_params_list[i + 4])
            scales, means = context_next.chunk(2, 1)

            scales_r = self.combine_for_writing_s3(torch.where(mask_list[i], scales, 0.0))
            indexes_r = self.entropy_conditional.build_indexes(scales_r)
            y_q_r = self.entropy_conditional.decompress_part(streams, indexes_r)
            y_hat_curr_step = torch.where(mask_list[i], torch.cat([y_q_r for _ in range(8)], dim=1) + means, 0.0)
            y_hat_so_far = y_hat_so_far + y_hat_curr_step

        return y_hat_so_far