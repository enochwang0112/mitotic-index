import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

_ARCHS = {"resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1)}


class MitosisClassifier(nn.Module):
    def __init__(self, arch: str = "resnet18", n_classes: int = 2, pretrained: bool = True):
        super().__init__()
        if arch not in _ARCHS:
            raise ValueError(f"unknown arch {arch!r}; choose from {sorted(_ARCHS)}")
        ctor, weights = _ARCHS[arch]
        net = ctor(weights=weights if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, n_classes)
        self.arch = arch
        self.n_classes = n_classes
        self.net = net

    def forward(self, x):
        return self.net(x)


def build_classifier(arch: str = "resnet18", n_classes: int = 2, pretrained: bool = True) -> MitosisClassifier:
    return MitosisClassifier(arch=arch, n_classes=n_classes, pretrained=pretrained)


def load_classifier(path, device=None, **kwargs) -> MitosisClassifier:
    ckpt = torch.load(path, map_location=device or "cpu")
    kwargs.setdefault("pretrained", False)
    kwargs.setdefault("arch", ckpt.get("arch", "resnet18"))
    kwargs.setdefault("n_classes", ckpt.get("n_classes", 2))
    model = build_classifier(**kwargs)
    model.load_state_dict(ckpt["model"])
    return model.to(device) if device is not None else model
