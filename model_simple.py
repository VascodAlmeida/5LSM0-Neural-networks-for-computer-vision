import torch
import torch.nn as nn
import torch.nn.init as init

""" segmentation model example
"""

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e4 = encoder_block(64, 128)

        """ Bottleneck """
        self.b = conv_block(128, 256)

        """ Decoder """
        self.d1 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Sequential(
            nn.Conv2d(64, 34, kernel_size=1, padding=0),
            nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
        )



    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s4, p4 = self.e4(p1)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d4 = self.d4(d1, s1)

        """ Segmentation output """
        outputs = self.outputs(d4)

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
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x
