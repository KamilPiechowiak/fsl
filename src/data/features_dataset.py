import os
from typing import Callable, Optional

import numpy as np
from .few_shot_dataset import FewShotDataset

from src.utils.file_utils import read_pickle

class FeaturesDataset(FewShotDataset):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            meta_split: str = 'train'
    ) -> None:
        super(FeaturesDataset, self).__init__(root, transform, target_transform)

        filename =  os.path.join(root, f'{meta_split}_features.pickle')
        data = read_pickle(filename)
        self.targets = [np.full(shape=len(data[key]), fill_value=key)
                    for key in data]
        self.data = [features for key in data for features in data[key]]

        self.class_ranges = [0]
        self.targets = []
        self.data = []

        for class_num, key in enumerate(data):
            entry = data[key]
            self.data.extend(entry)
            self.targets.extend([class_num]*len(entry))
            self.class_ranges.append(self.class_ranges[-1]+len(entry))
        
        self.total_classes = len(self.class_ranges)-1