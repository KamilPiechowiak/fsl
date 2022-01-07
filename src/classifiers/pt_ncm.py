from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import logging

from .pt import PT

class PT_NCM(PT):
    def __init__(self, preprocess=True, eps=1e-6) -> None:
        super(PT_NCM, self).__init__()
        self.perform_preprocessing = preprocess
        self.eps = eps

    def preprocess(self, features: np.ndarray) -> np.ndarray:
        if self.perform_preprocessing:
            assert np.min(features)+self.eps > 0
            y = features**self.beta
            return y / (y**2).sum(axis=1, keepdims = True)**0.5
        else:
            return features


    def adjust(self, features_q: np.ndarray, features_s: np.ndarray, labels_s: np.ndarray, initial_class_centers: np.ndarray, classes_counts: np.ndarray, q: int) -> np.ndarray:
        scores = np.dot(features_q, self.W)
        return np.argmax(scores, axis=1)
