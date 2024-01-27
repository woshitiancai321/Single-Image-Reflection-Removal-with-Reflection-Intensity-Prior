"""
Custom Norm wrappers to enable sync BN, regular BN and for weight initialization
"""
import torch.nn as nn
#from config import cfg
from utils2.attr_dict import AttrDict
from apex import amp

__C = AttrDict()
cfg = __C
__C.MODEL = AttrDict()
__C.MODEL.BN = 'regularnorm'
#__C.MODEL.BNFUNC = None
__C.MODEL.BNFUNC = nn.BatchNorm2d
def Norm2d(in_channels):
    """
    Custom Norm Function to allow flexible switching
    """
    layer = getattr(cfg.MODEL, 'BNFUNC')
    normalization_layer = layer(in_channels)
    return normalization_layer


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


@amp.float_function
def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=True)
