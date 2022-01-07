from abc import ABC
from typing import List, Optional
import numpy as np

class Classifier(ABC):

    def __init__(self) -> None:
        pass

    def fit_predict(self, features: np.ndarray, y: List[Optional[int]]) -> np.ndarray:
        raise NotImplementedError()

    def predicte(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def save(self, path: str) -> None:
        raise NotImplementedError()

    def load(self, path: str) -> None:
        raise NotImplementedError()

    def split_features(self, features, labels):
        features_s = []
        labels_s = []
        features_q = []
        for f, y in zip(features, labels):
            if y == -1:
                features_q.append(f)
            else:
                features_s.append(f)
                labels_s.append(y)
        features_q = np.array(features_q)
        features_s = np.array(features_s)
        labels_s = np.array(labels_s)
        return features_s, labels_s, features_q
