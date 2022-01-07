import os
import pickle
from typing import Any, Callable, Optional, Tuple
import numpy as np
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from sklearn.model_selection import train_test_split

from .simple_dataset import SimpleDataset

class FewShotDataset(VisionDataset):
    base_folder = None

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super(FewShotDataset, self).__init__(root, transform=transform,
                                      target_transform=target_transform)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)

    def sample_random_dataset(self, num_classes: int, num_known_samples_per_class: int, num_unknown_samples_per_class) -> SimpleDataset:
        def sample_without_replacement(low: int, high: int, size: int=1):
            return np.random.choice(np.arange(low, high), size=size, replace=False)
        
        classes = sample_without_replacement(0, self.total_classes, num_classes)
        data = []
        targets = []
        known_mask = []
        for j, class_id in enumerate(classes):
            ids = sample_without_replacement(self.class_ranges[class_id], self.class_ranges[class_id+1], num_known_samples_per_class+num_unknown_samples_per_class)
            for i, idx in enumerate(ids):
                data.append(self.data[idx])
                targets.append(j)
                if i < num_known_samples_per_class:
                    known_mask.append(1)
                else:
                    known_mask.append(0)
        
        return SimpleDataset(data, targets, known_mask, transform=self.transform, target_transform=self.target_transform)