import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    size_divisor = 16

    def __init__(self, in_channels: int = 1, n_classes: int = 3, base: int = 32):
        super().__init__()
        self.n_classes = n_classes

        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, n_classes, 1)

    @staticmethod
    def _cat(up, skip):
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="nearest")
        return torch.cat([up, skip], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(self._cat(self.up4(b), e4))
        d3 = self.dec3(self._cat(self.up3(d4), e3))
        d2 = self.dec2(self._cat(self.up2(d3), e2))
        d1 = self.dec1(self._cat(self.up1(d2), e1))

        return self.head(d1)


class UNetResNet34(nn.Module):
    size_divisor = 32

    def __init__(self, in_channels: int = 1, n_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.n_classes = n_classes

        net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        rgb = net.conv1.weight.data
        conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        if in_channels == 1:
            conv1.weight.data = rgb.sum(dim=1, keepdim=True)
        elif in_channels == 3:
            conv1.weight.data = rgb

        self.stem = nn.Sequential(conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = DoubleConv(32, 32)
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        e1 = self.stem(x)
        e2 = self.layer1(self.pool(e1))
        e3 = self.layer2(e2)
        e4 = self.layer3(e3)
        b = self.layer4(e4)

        d4 = self.dec4(UNet._cat(self.up4(b), e4))
        d3 = self.dec3(UNet._cat(self.up3(d4), e3))
        d2 = self.dec2(UNet._cat(self.up2(d3), e2))
        d1 = self.dec1(UNet._cat(self.up1(d2), e1))
        d0 = self.dec0(self.up0(d1))

        return self.head(d0)


_ARCHS = {"unet": UNet, "resnet34": UNetResNet34}


def build_model(arch: str = "unet", **kwargs) -> nn.Module:
    if arch not in _ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(_ARCHS)}")
    return _ARCHS[arch](**kwargs)


def load_model(path, device=None, **kwargs) -> nn.Module:
    ckpt = torch.load(path, map_location=device or "cpu")
    arch = ckpt.get("arch", "unet")
    if arch == "resnet34":
        kwargs.setdefault("pretrained", False)
    model = build_model(arch, **kwargs)
    model.load_state_dict(ckpt["model"])
    return model.to(device) if device is not None else model
