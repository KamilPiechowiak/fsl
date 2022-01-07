from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import logging
from sklearn.cluster import KMeans

from .pt import PT

class PT_Kmeans(PT):
    def __init__(self) -> None:
        super(PT_Kmeans, self).__init__()

    def adjust(self, features_q: np.ndarray, features_s: np.ndarray, labels_s: np.ndarray, initial_class_centers: np.ndarray, classes_counts: np.ndarray, q: int) -> np.ndarray:
        n_classes = initial_class_centers.shape[1]
        k_means = KMeans(n_classes, init=(initial_class_centers/classes_counts.reshape(1, -1)).T, n_init=1)
        y_pred = k_means.fit_predict(np.concatenate((features_q, features_s), axis=0))
        return y_pred[:len(features_q)]