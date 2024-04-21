import torch
import torch.nn as nn
import torch.nn.init as init

import sys
sys.path.append("/gpfs/home6/scur0004/.local/lib/python3.6/site-packages")

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Swish



import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        """ EfficientNet Encoder """
        self.eff_encoder = EfficientNet_Backbone(output_layer='_avg_pooling')  

        """ Decoder """
        self.d1 = decoder_block(1280, 512)  
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)

        """ Segmentation output """
        self.outputs = nn.Sequential(
            nn.Conv2d(128, 34, kernel_size=1, padding=0),
            nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
        )

    def forward(self, inputs):
        """ Encoder """
        features = self.eff_encoder(inputs)

        """ Decoder """
        d1 = self.d1(features)  
        d2 = self.d2(d1)  
        d3 = self.d3(d2)  

        """ Segmentation output """
        outputs = self.outputs(d3)  

        return outputs



# Depthwise convs
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        # Depth-wise convolution
        self.depthwise_conv = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c)
        self.depthwise_bn = nn.BatchNorm2d(in_c)
        
        # Point-wise convolution
        self.pointwise_conv = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.pointwise_bn = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        # Depth-wise convolution
        x = self.depthwise_conv(inputs)
        x = self.depthwise_bn(x)
        x = self.relu(x)

        # Point-wise convolution
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
#https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/README.md#usage
class EfficientNet_Backbone(nn.Module):
    def __init__(self, output_layer=None):
        super(EfficientNet_Backbone, self).__init__()
        self.backbone_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.output_layer = output_layer

        # Remove layers after the output layer
        if self.output_layer is not None:
            self.remove_layers()

    def remove_layers(self):
        if self.output_layer in self.backbone_model._modules:
            output_layer_index = list(self.backbone_model._modules.keys()).index(self.output_layer)

            for layer_name in list(self.backbone_model._modules.keys())[output_layer_index+1:]:
                delattr(self.backbone_model, layer_name)

    def forward(self, x):
        # Swish activation function
        self.backbone_model._swish = Swish()
        x = self.backbone_model.extract_features(x)
        return x

#1x1 convs
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        # 1x1 Convolution to reduce the number of channels
        self.reduce_channels = nn.Conv2d(in_c, out_c, kernel_size=1)

        # Convolutional block followed by max pooling
        self.conv = conv_block(out_c, out_c)  
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        # Reduce channels using 1x1 convolution
        x = self.reduce_channels(inputs)

        # Convolutional block
        x = self.conv(x)

        # Max pooling
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c, out_c)

    def forward(self, inputs):
        x = self.up(inputs)
        x = self.conv(x)

        return x
