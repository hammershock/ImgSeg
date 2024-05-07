"""

Unet Inference with ONLY Numpy

The NumPyTorch Implementation is good,
It is almost Pytorch-style so the model can be easily to be transplanted between torch and numpytorch.

@author hammershock
@email zhanghanmo@bupt.edu.cn
@Github https://github.com/hammershock/ImgSeg
@date 2024.5.7
"""
import numpy as np

import NumPyTorch.nn as nn
import NumPyTorch as torch
import NumPyTorch.nn.functional as F

import cv2


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # 1024
        return self.conv(x)


class UNet(nn.Module):
    """
    Example:
    (Dummy Test)
    >>> model = UNet()
    >>> dummy_imgs = torch.rand(4, 3, 256, 256)  # Batch size, channels, height, width
    >>> dummy_masks = torch.randint(0, 2, (4, 1, 256, 256))  # Binary mask
    >>> model = model.to('cpu')
    >>> outputs = model(dummy_imgs)
    >>> outputs.shape == dummy_masks.shape
    >>> outputs.shape[-2:] == dummy_imgs.shape[-2:] == dummy_masks.shape[-2:]
    True
    """

    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = ConvBlock(3, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.down4 = ConvBlock(256, 512)
        self.down5 = ConvBlock(512, 1024)

        self.pool = nn.MaxPool2d(2)  # Change to MaxPool for simplifying

        self.up4 = UpConvBlock(1024, 512)
        self.up3 = UpConvBlock(512, 256)
        self.up2 = UpConvBlock(256, 128)
        self.up1 = UpConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, 34, kernel_size=1)

    def forward(self, x):
        # Downsampling path
        d1 = self.down1(x)  # 64
        d2 = self.down2(self.pool(d1))  # 128
        d3 = self.down3(self.pool(d2))  # 256
        d4 = self.down4(self.pool(d3))  # 512
        d5 = self.down5(self.pool(d4))  # 1024

        # Upsampling path
        u4 = self.up4(d5, d4)
        u3 = self.up3(u4, d3)
        u2 = self.up2(u3, d2)
        u1 = self.up1(u2, d1)

        return self.final_conv(u1)


class FlexibleUNet(nn.Module):
    """
    A configurable U-Net architecture that allows adjusting the depth and channel expansion factor.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels in the output image.
        depth (int): Number of levels in the U-Net, including the bottom level.
        wf (int): Number of filters in the first layer is `wf`, it gets doubled every down step.

    Example:
    >>> model = FlexibleUNet(in_channels=3, out_channels=34, depth=5, wf=6)
    >>> dummy_imgs = torch.rand(4, 3, 256, 256)
    >>> dummy_masks = torch.randint(0, 2, (4, 1, 256, 256))
    >>> model = model.to('cpu')
    >>> outputs = model(dummy_imgs)
    >>> torch.save(model.state_dict(), f'./test_model.pth')
    >>> outputs.shape[-2:] == dummy_masks.shape[-2:]
    True
    """

    def __init__(self, in_channels=3, out_channels=34, depth=5, wf=6):
        super(FlexibleUNet, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        prev_channels = in_channels
        for i in range(depth):
            features = wf * (2 ** i)  # [1, 2, 4, 8, 16, 32, 128] * wf
            self.down_path.append(ConvBlock(prev_channels, features))
            prev_channels = features

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            features = wf * (2 ** i)
            self.up_path.append(UpConvBlock(prev_channels, features))
            prev_channels = features

        self.pool = nn.MaxPool2d(2)
        self.final_conv = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            if i != len(self.down_path) - 1:
                x = self.pool(x)

        blocks = blocks[::-1]  # Reverse the blocks list
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[i + 1])

        return self.final_conv(x)


def normalize(image, mean=None, std=None):
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    if std is None:
        std = np.array([0.229, 0.224, 0.225])
    image = (image - mean[:, None, None]) / std[:, None, None]  # Normalize image
    return image


label_colors = np.random.randint(0, 255, (34, 3), dtype=np.uint8)  # Same color mapping as before


def decode_segmap(image, nc=34):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


if __name__ == '__main__':
    # state_dict = torch.load_pytorch('./models/unet_5_6.pth')
    # print(state_dict.keys())
    # torch.save(state_dict, './models/unet_5_6.pkl')
    state_dict = torch.load('./models/unet.pkl')
    # model = FlexibleUNet(3, 34, depth=5, wf=6)
    model = UNet()
    model.load_state_dict(state_dict)
    image = cv2.imread('./val_samples/data/frankfurt_000000_000294_leftImg8bit.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 256)).transpose((2, 0, 1)) / 255.0
    image = image[np.newaxis, :, :, :]
    image = normalize(image)
    outputs = model(image.astype(np.float16))

    probabilities = F.softmax(outputs, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)

    image_show = decode_segmap(predicted_classes[0])
    print(predicted_classes.shape)
    cv2.imshow('image', image_show)
    cv2.waitKey(0)



