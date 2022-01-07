from typing import Dict

import re
from torch import nn
import torchvision
import timm
from .models import SmallResnet, Resnet, WideResnet, Mockup

def get_model(config: Dict) -> nn.Module:
    if config.get("mock", 0) == 1:
        return Mockup(config["num_classes"])
    pretrained = config.get("pretrained", False)
    num_classes = config.get("num_classes")
    if hasattr(torchvision.models, config["name"]):
        model_function = getattr(torchvision.models, config["name"])
        model = model_function(pretrained=pretrained)
        if num_classes:
            model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
        return model
    match = re.fullmatch("SmallResnet(\d+)\.(\d+)", config["name"])
    if match is not None:
        return SmallResnet(int(match.group(1)), config["num_classes"], int(match.group(2)))
    match = re.fullmatch("Resnet(\d+)\.(\d+)", config["name"])
    if match is not None:
        return Resnet(int(match.group(1)), config["num_classes"], int(match.group(2)))
    match = re.fullmatch("WideResnet(\d+)\.(\d+)", config["name"])
    if match is not None:
        return WideResnet(int(match.group(1)), config["num_classes"], int(match.group(2)))
    if config["name"].startswith("TheirWideResNet"):
        from .models.wrn_mixup_model import wrn28_10
        return wrn28_10(config["num_classes"], loss_type='softmax')#, wrap=config.get("wrap", False))
    if config["name"].startswith("TheirResNet"):
        from .models.resnet_mixup_model import resnet18
        return resnet18(num_classes=config["num_classes"])

    model = timm.create_model(config["name"], pretrained=pretrained, num_classes=num_classes)
    return model