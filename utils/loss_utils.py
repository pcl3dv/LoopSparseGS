#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchmetrics.functional.regression import pearson_corrcoef

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def compute_pearson_loss(mono_depth, rendered_depth):
    mono_depth = mono_depth.reshape(-1, 1)
    rendered_depth = rendered_depth.reshape(-1, 1)
    mono_depth_loss = min(
        (1 - pearson_corrcoef( - mono_depth, rendered_depth)),
        (1 - pearson_corrcoef(1 / (mono_depth + 200.), rendered_depth))
        )
    return mono_depth_loss

def compute_patch_pearson_loss(mono_depth, rendered_depth, p_l, stride=4):
    patch_mono_depth = mono_depth.unfold(0, p_l, stride).unfold(1, p_l, stride).reshape(-1, p_l*p_l).T
    patch_rendered_depth = rendered_depth.unfold(0, p_l, stride).unfold(1, p_l, stride).reshape(-1, p_l*p_l).T
    
    mono_mask = (torch.sum(patch_mono_depth < 1e-5, 0) == p_l*p_l)
    if mono_mask.sum()>0:
        mono_loss = torch.min(
            (1 - pearson_corrcoef( - patch_mono_depth[:, ~mono_mask], patch_rendered_depth[:, ~mono_mask])),
            (1 - pearson_corrcoef(1 / (patch_mono_depth[:, ~mono_mask] + 200.), patch_rendered_depth[:, ~mono_mask]))
            )
        mask_loss = 0.001 * ((patch_rendered_depth[:, mono_mask] - patch_rendered_depth[:, mono_mask].mean(0))**2).mean(0)
        mono_depth_loss = torch.concat((mono_loss,  mask_loss)).mean()
    else:
        mono_depth_loss = torch.min(
            (1 - pearson_corrcoef( - patch_mono_depth, patch_rendered_depth)),
            (1 - pearson_corrcoef(1 / (patch_mono_depth + 200.), patch_rendered_depth))
            ).mean()
    return mono_depth_loss
    
def compute_depth_loss(colmap_depth, rendered_depth):
    rendered_depth = rendered_depth.reshape(-1, 1)
    colmap_depth = colmap_depth.reshape(-1, 1)
    valid_depth_mask = colmap_depth > 0.
    
    depth_loss = l1_loss(colmap_depth[valid_depth_mask], rendered_depth[valid_depth_mask])
    return depth_loss


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def tv_loss(x, weight=1.0):
    # batch_size = x.size(0)
    h_x = x.size(0)
    w_x = x.size(1)
    count_h = (h_x - 1) * w_x
    count_w = h_x * (w_x - 1)
    
    h_tv = torch.pow(x[1:, :] - x[:h_x-1, :], 2).sum()
    w_tv = torch.pow(x[:, 1:] - x[:, :w_x-1], 2).sum()
    
    return weight * 2 * (h_tv / count_h + w_tv / count_w)