from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import logging

from .pt import PT

class PEME(PT):
    def __init__(self, beta: float = 0.5, eps: float = 1e-6, alpha: float = 0.1, n_steps: int = 20, lambda_: int = 8.5, epochs: int = 40, momentum: float = 0.8) -> None:
        super(PEME, self).__init__(beta, eps, alpha, n_steps, lambda_)
        self.epochs = epochs
        self.momentum = momentum

    def preprocess(self, features: np.ndarray, training_mean: np.ndarray = None) -> np.ndarray:
        assert np.min(features)+self.eps > 0
        p = features**self.beta
        e = p / (p**2).sum(axis=1, keepdims = True)**0.5
        if training_mean is not None:
            m = e - training_mean
        else:
            m = e - e.mean(axis=0)
        return m / (m**2).sum(axis=1, keepdims = True)**0.5

    def sinkhorn(self, L: np.ndarray, p: int, q: int, labels_s: np.ndarray):
        M = np.exp(-self.lambda_*L)
        M[:len(labels_s),:] = 0
        M[np.arange(len(labels_s)), labels_s] = 1 # FIXME samples with known class
        for _ in range(50):
            M = M/M.sum(axis=1, keepdims=True)*p
            mask = (M.sum(axis=0) < q)
            M[:,mask] = M[:,mask]/(M[:,mask].sum(axis=0))*q
        return M

    def adjust(self, features_q: np.ndarray, features_s: np.ndarray, labels_s: np.ndarray, initial_class_centers: np.ndarray, classes_counts: np.ndarray, q: int) -> np.ndarray:
        features = np.concatenate([features_s, features_q], axis=0)
        estimated_min_samples_per_class = np.min(classes_counts)
        for _ in range(self.n_steps+1):
            self.W /= (self.W**2).sum(axis=0, keepdims = True)**0.5
            L = 1 - np.dot(features, self.W) # n_samples x n_classes
            M = self.sinkhorn(L, 1, estimated_min_samples_per_class, labels_s) # n_samples x n_classes
            self.W = np.dot(features.T, M) / M.sum(axis=0) # features_dim x n_classes
            # if np.isnan(self.W.std(axis=0)[0]):
            #     print(M)
            #     exit(0)
            if self.epochs > 0:
                self.fine_tune_weights(M, features)
            res = np.argmax(M, axis=1)
            estimated_min_samples_per_class = np.min(np.unique(res, return_counts=True)[1])
        return res[len(labels_s):]
        

    def fine_tune_weights(self, M: np.ndarray, features: np.ndarray) -> None:
        kappa = torch.Tensor(1).float()
        kappa.requires_grad = True
        W = torch.from_numpy(self.W).float()
        W.requires_grad = True
        features = torch.from_numpy(features).float()
        M = torch.from_numpy(M).float()
        # print(torch.argmax(M, dim=1))
        optimizer = torch.optim.SGD([W, kappa], lr=self.alpha, momentum=self.momentum)
        logging.debug(f"max W's value before opt: {W.abs().max()}")
        logging.debug("-----------------------------")
        for _ in range(self.epochs):
            y_predicted = kappa * (features @ W) / (W**2).sum(dim=0)**0.5
            # print((torch.argmax(y_predicted, dim=1) == torch.argmax(M, dim=1)).float().mean())
            loss = F.cross_entropy(y_predicted, M)
            if loss.detach().cpu() > 10:
                continue
            loss.backward()
            logging.debug(loss)
            optimizer.step()
            optimizer.zero_grad()
        logging.debug(f"max W's value after opt: {W.abs().max()}")
        self.W = W.detach().numpy()