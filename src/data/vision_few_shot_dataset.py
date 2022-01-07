import os
import pickle
from typing import Any, Callable, Optional
from .few_shot_dataset import FewShotDataset
from sklearn.model_selection import train_test_split


class VisionFewShotDataset(FewShotDataset):
    base_folder = None

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            meta_split: str = 'train',
            random_state: int = 0
    ) -> None:
        """
        Args:
            train - True, False or None (do not split)
            meta_split - few shot learning split, one of {train, val, test} 
        """

        super(FewShotDataset, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        with open(os.path.join(self.root, self.base_folder, 'splits', f'{meta_split}.txt')) as file:
            downloaded_list = file.read().splitlines()

        self.data: Any = []
        self.targets = []
        self.class_ranges = [0]
        self.total_classes = len(downloaded_list)

        # one file per class
        for class_num, file_name in enumerate(downloaded_list):
            file_path = os.path.join(self.root, self.base_folder, 'data', f"{file_name}.pickle")
            with open(file_path, 'rb') as f:
                entry = pickle.load(f)
                self.data.extend(entry)
                self.targets.extend([class_num]*len(entry))
                self.class_ranges.append(self.class_ranges[-1]+len(entry))
        
        if train is not None:
            data_train, data_test, targets_train, targets_test = train_test_split(self.data, self.targets, test_size=0.2, random_state=random_state)
            if train:
                self.data, self.targets = data_train, targets_train
            else:
                self.data, self.targets = data_test, targets_test