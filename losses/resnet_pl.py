"""High-receptive-field perceptual loss (LaMa's resnet_pl), vendored.

ADE20k-segmentation ResNet50-dilated features; the dilated receptive field
rewards globally-consistent structure (LaMa tab.3: FID 5.69 vs 6.29 for VGG19).
Expects inputs in [0, 1].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_ade20k import ModelBuilder

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None]
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None]

WEIGHTS_PATH = '/home/jincheng/Mural/mural_project/lama_mat_comparison/torch_home'


class ResNetPL(nn.Module):
    def __init__(self, weight=1, weights_path=WEIGHTS_PATH,
                 arch_encoder='resnet50dilated', segmentation=True):
        super().__init__()
        self.impl = ModelBuilder.get_encoder(weights_path=weights_path,
                                             arch_encoder=arch_encoder,
                                             arch_decoder='ppm_deepsup',
                                             fc_dim=2048,
                                             segmentation=segmentation)
        self.impl.eval()
        for w in self.impl.parameters():
            w.requires_grad_(False)
        self.weight = weight

    def forward(self, pred, target):
        pred = (pred - IMAGENET_MEAN.to(pred)) / IMAGENET_STD.to(pred)
        target = (target - IMAGENET_MEAN.to(target)) / IMAGENET_STD.to(target)
        pred_feats = self.impl(pred, return_feature_maps=True)
        target_feats = self.impl(target, return_feature_maps=True)
        return torch.stack([F.mse_loss(p, t) for p, t in zip(pred_feats, target_feats)]).sum() * self.weight
