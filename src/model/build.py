"""
Adapted from: https://github.com/mariodoebler/test-time-adaptation/blob/main/classification/models/model.py
"""

from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from torchvision import models
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from collections import OrderedDict
import torch
from src.model.resnet_domainnet126 import ResNetDomainNet126
    
class ImageNormalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float],
        std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std
    
def normalize_model(model: nn.Module, mean: Tuple[float, float, float],
    std: Tuple[float, float, float]) -> nn.Module:
    layers = OrderedDict([
        ('normalize', ImageNormalizer(mean, std)),
        ('model', model)
    ])
    return nn.Sequential(layers)

def build_model(cfg):
    dts = cfg.CORRUPTION.DATASET.split("_")[0]
    if dts in ["cifar10c", "cifar100c"]:
        base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                                "cifar10" if dts == "cifar10c" else "cifar100", ThreatModel.corruptions).cuda()
    elif dts in ["domainnet126"]:
        base_model = ResNetDomainNet126(
             arch=cfg.MODEL.ARCH, 
             checkpoint_path=cfg.CKPT_PATH, 
             num_classes=cfg.CORRUPTION.NUM_CLASS)
    elif dts in ["imagenetc", "ccc"]:
        mu = (0.485, 0.456, 0.406)
        sigma = (0.229, 0.224, 0.225)

        base_model = normalize_model(models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        ), mu, sigma)
    else:
        raise NotImplementedError()

    return base_model
