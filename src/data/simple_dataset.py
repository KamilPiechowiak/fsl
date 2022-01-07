from typing import Any, Callable, List, Optional, Tuple
import numpy as np
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    base_folder = None

    def __init__(
            self,
            X: List[Any],
            y: List[int],
            known_mask: List[int],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            meta_split - few shot learning split, one of {train, val, test} 
        """
        self.X = X
        self.y = np.array(y)
        self.known_mask = np.array(known_mask)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.X[index], self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.known_mask[index] == 0:
            target = -1

        return img, target


    def __len__(self) -> int:
        return len(self.X)

    def get_true_labels(self) -> List[int]:
        return self.y[self.known_mask == 0]