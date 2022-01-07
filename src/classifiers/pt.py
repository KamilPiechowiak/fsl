from typing import List, Optional
import numpy as np

from .classifier import Classifier

class PT(Classifier):
    def __init__(self, beta: float = 0.5, eps: float = 1e-6, alpha: float = 0.4, n_steps: int = 30, lambda_: int = 10) -> None:
        self.beta = beta
        self.eps = eps
        self.alpha = alpha
        self.n_steps = n_steps
        self.lambda_ = lambda_

    def preprocess(self, features: np.ndarray) -> np.ndarray:
        assert np.min(features)+self.eps > 0
        y = features**self.beta
        return y / (y**2).sum(axis=1, keepdims = True)**0.5

    def sinkhorn(self, L: np.ndarray, p: int, q: int):
        M = np.exp(-self.lambda_*L)
        for _ in range(50):
            M = M/M.sum(axis=1, keepdims=True)*p
            M = M/M.sum(axis=0, keepdims=True)*q
        return M

    def fit_predict(self, features: np.ndarray, labels: List[Optional[int]]) -> np.ndarray:
        features = self.preprocess(features)
        features_dim = features.shape[1]
        features_s, labels_s, features_q = self.split_features(features, labels)

        classes, classes_counts = np.unique(labels_s, return_counts=True)
        n_classes = classes.size
        q = features_q.shape[0]/n_classes
        self.W = np.zeros((features_dim, n_classes))
        for f, y in zip(features_s, labels_s):
            self.W[:,y] += f
        initial_class_centers = self.W.copy()
        self.W = self.W/classes_counts.reshape(1, -1)

        return self.adjust(features_q, features_s, labels_s, initial_class_centers, classes_counts, q)

    def adjust(self, features_q: np.ndarray, features_s: np.ndarray, labels_s: np.ndarray, initial_class_centers: np.ndarray, classes_counts: np.ndarray, q: int) -> np.ndarray:
        for _ in range(self.n_steps+1):
            L = ((np.expand_dims(features_q, axis=1) - np.expand_dims(self.W.T, axis=0))**2).sum(axis=2) # unknown_samples x n_classes
            M = self.sinkhorn(L, 1, q) # unknown_samples x n_classes
            W_cand = (np.dot(features_q.T, M) + initial_class_centers) / (M.sum(axis=0) + classes_counts) # features_dim x n_classes
            self.W = (1-self.alpha)*self.W + self.alpha*W_cand
            # self.W /= (self.W**2).sum(axis=0, keepdims = True)**0.5

        return M.argmax(axis=1)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        features = self.preprocess(features)
        scores = np.dot(features, self.W)
        return np.argmax(scores, axis=1)

