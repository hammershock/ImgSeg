"""
A Practice of Image Segmentation, on dataset 'Cityscapes'
@author hammershock
@email zhanghanmo@bupt.edu.cn
@Github https://github.com/hammershock/ImgSeg
@date 2024.5.7
"""

import os
from collections import defaultdict
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from inference_numpy import decode_segmap


def load_data(train_dir: str, label_dir: str) -> Dict[str, Dict[str, list]]:
    """

    :param train_dir:
    :param label_dir:
    :return:

    Example:
    >>> data = load_data(train_dir='./leftImg8bit_trainvaltest/', label_dir='./gtFine_trainvaltest')
    >>> train_data = data['train']
    >>> key, data_label_pair = next(iter(train_data.items()))  # first
    >>> key
    'jena_000078_000019'
    >>> data_label_pair[0]  # image path
    './leftImg8bit_trainvaltest/leftImg8bit/train/jena/jena_000078_000019_leftImg8bit.png'
    >>> data_label_pair[1]  # label path
    './gtFine_trainvaltest/gtFine/train/jena/jena_000078_000019_gtFine_labelIds.png'
    """
    data = {name: defaultdict(list) for name in ['train', 'val', 'test']}
    for i, directory in enumerate([train_dir, label_dir]):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if not file.endswith('.png'):  # filter other files, i.e. "README" and "LICENSE"
                    continue
                if i == 1 and not file.endswith('labelIds.png'):
                    continue

                parts = root.split("/")

                if parts[-1][0] == '.':  # filter the invisible folders, i.e. ".ipynb_checkpoints"
                    continue
                data_type = parts[-2]  # test, train, val
                city_name = parts[-1]  # (unused)
                filepath = os.path.join(root, file)

                # the header of image name, like "munich_000276_000019", we use it as the dict key
                base_name = '_'.join(file.split('_')[:3])

                # images and labels are restored in pairs
                data[data_type][base_name].append(filepath)  # full path to the image

    return data


class ConvBlock(nn.Module):
    """
    This class defines a convolutional block with two convolution layers,
    each followed by batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.

    Examples:
    >>> conv_block = ConvBlock(3, 16)
    >>> x = torch.randn(1, 3, 64, 64)
    >>> output = conv_block(x)
    >>> output.shape
    torch.Size([1, 16, 64, 64])
    """
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
    """
    This class defines an up-sampling block using a transposed convolution followed by concatenation with a skip connection and a convolution block.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor after up-sampling.

    Examples:
    >>> upconv = UpConvBlock(3, 8)
    >>> x = torch.randn(1, 3, 64, 64)
    >>> skip = torch.randn(1, 8, 128, 128)
    >>> output = upconv(x, skip)
    >>> output.shape
    torch.Size([1, 8, 128, 128])
    """
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        """

        :param x: (b, in, size, size)
        :param skip: (b, out, 2 * size, 2 * size)
        :return:
        """
        x = self.up(x)  # out
        x = torch.cat([x, skip], dim=1)  # concat along channel axis
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
    True
    >>> outputs.shape[-2:] == dummy_imgs.shape[-2:]
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
        d1 = self.down1(x)  # (64, size)
        d2 = self.down2(self.pool(d1))  # (128, size/2)
        d3 = self.down3(self.pool(d2))  # (256, size/4)
        d4 = self.down4(self.pool(d3))  # (512, size/8)
        d5 = self.down5(self.pool(d4))  # (1024, size/16)

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


class CityscapesDataset(Dataset):
    cache = {}

    def __init__(self, data_dict: dict):
        super(CityscapesDataset, self).__init__()
        self.data = list(data_dict.values())

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def normalize(self, image):
        image = (image - self.mean[:, None, None]) / self.std[:, None, None]  # Normalize image
        return image

    def load_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = (image / 255.0).astype(np.float32)  # Normalize to [0, 1]
        image = np.transpose(image, (2, 0, 1))  # Change HWC to CHW
        image = self.normalize(image)
        return torch.from_numpy(image).float()

    def load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {path}")
        mask = cv2.resize(mask, (512, 256), interpolation=cv2.INTER_LINEAR)
        mask = mask.astype(np.int64)  # Ensure mask is integer type
        return torch.from_numpy(mask).long()  # Convert numpy array to torch tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, mask_path = self.data[idx]

        # Check cache first
        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)

        # mask = torch.squeeze(mask, 0)  # 移除单一维度
        # mask = mask.long()
        return image, mask


def train(model, train_loader, optimizer, criterion, device, epoch, writer, save_path):
    model.train()
    running_loss = 0.0
    p_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for i, (images, masks) in enumerate(p_bar):
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loss_show = running_loss / (i + 1)
        writer.add_scalar('Training Loss', loss_show, epoch * len(train_loader) + i)
        p_bar.set_postfix(running_loss=loss_show)

    torch.save(model.state_dict(), save_path)
    return running_loss / len(train_loader)


def validate(model, data_loader, device, writer, epoch, n_validate=None):
    model.eval()  # Set model to evaluation mode
    total_intersection, total_union = 0, 0
    num_classes = 34  # assuming 34 classes including background
    kwargs = {"total": n_validate} if n_validate is not None else {}
    with torch.no_grad():  # Disable gradient computation
        for i, (images, masks) in enumerate(tqdm(data_loader, desc='validating', **kwargs)):
            if n_validate is not None and i > n_validate:
                break
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)

            images = predicted_classes.cpu().detach().numpy()  # .squeeze()
            for image in images:
                # Convert predicted classes to color images
                image_show = decode_segmap(image)

                # Convert Numpy array to PIL Image and then to Tensor for TensorBoard
                image_show = Image.fromarray(image_show)
                image_show = transforms.ToTensor()(image_show)
                writer.add_image(f'Validation/Image_{i}', image_show, epoch)

                # Calculate intersection and union per batch
                intersection, union = calculate_intersection_and_union(predicted_classes, masks, num_classes)
                total_intersection += intersection
                total_union += union

        # Calculate mIoU
        miou = total_intersection / total_union
        writer.add_scalar('Validation/mIoU', miou.mean(), epoch)  # log mean IoU to TensorBoard


def calculate_intersection_and_union(pred, target, num_classes):
    """ Calculate intersection and union for multi-class segmentation. """
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)

    for class_ in range(num_classes):
        pred_inds = (pred == class_)
        target_inds = (target == class_)
        intersection[class_] = (pred_inds & target_inds).sum()
        union[class_] = (pred_inds | target_inds).sum()

    return intersection, union


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        ce_loss = self.ce_loss(inputs, targets)
        return self.alpha * dice_loss + self.beta * ce_loss


if __name__ == "__main__":
    data = load_data(train_dir='./leftImg8bit_trainvaltest/', label_dir='./gtFine_trainvaltest')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = CityscapesDataset(data['train'])
    val_set = CityscapesDataset(data['val'])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=10, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=10, persistent_workers=True)

    model_depth = 5
    wf = 6
    model = FlexibleUNet(in_channels=3, out_channels=34, depth=model_depth, wf=wf).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification

    epochs = 120
    model_path = f'./models/unet_{model_depth}_{wf}.pth'

    os.makedirs(os.path.split(model_path)[0], exist_ok=True)

    try: model.load_state_dict(torch.load(model_path, map_location=device))
    except: pass

    writer = SummaryWriter('../../tf-logs/cityscapes_experiment')

    for epoch in range(epochs):
        # Perform one epoch of training
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, writer, model_path)
        validate(model, val_loader, device, writer, epoch, n_validate=100)

    writer.close()
