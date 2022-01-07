from typing import Dict, Tuple, Union
import torch

from torchvision import transforms, datasets

from src.data.cifar_fs import CifarFS
from src.data.mini_image_net import MiniImageNet
from src.data.features_dataset import FeaturesDataset

from .few_shot_dataset import FewShotDataset

def get_dataset(config: Dict, path: str, meta_split: str, train_test_split: bool = False, model_path: str = None) -> Union[FewShotDataset, Tuple[FewShotDataset, FewShotDataset]]:
    name = config["name"]
    resolution = config["resolution"]
    if config.get('extract_features', 1) == 0:
        name = "FeaturesDataset"
        path = model_path
    def prepare_dataset(name: str, train: bool, resolution: int):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            if resolution < 100:
                scale = (0.5, 1)
            else:
                scale = (0.1, 1)
            preprocess = transforms.Compose([
                transforms.RandomResizedCrop(resolution, scale),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            preprocess = transforms.Compose([
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                normalize
            ])
        return {
            'CifarFS': lambda: CifarFS(f'{path}', train=train, transform=preprocess, meta_split=meta_split),
            'MiniImageNet': lambda: MiniImageNet(f'{path}', train=train, transform=preprocess, meta_split=meta_split),
            'FeaturesDataset': lambda: FeaturesDataset(f'{path}', transform=None, meta_split=meta_split),
        }[name]
    
    if train_test_split:
        return prepare_dataset(name,
                           True,
                           resolution)(), \
               prepare_dataset(name,
                           False,
                           resolution)()
    else:
        return prepare_dataset(name,
                           None,
                           resolution)()
