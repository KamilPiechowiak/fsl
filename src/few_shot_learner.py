from typing import Dict
import torch
import numpy as np
from .feature_extractors import FeatureExtractor, feature_extractor
from .classifiers import Classifier, get_classifier

class FewShotLearner:
    def __init__(self, feature_extractor: FeatureExtractor, classifier: Classifier) -> None:
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def fit_predict(self, dataset: torch.utils.data.DataLoader, extract_features=True) -> np.ndarray:
        if extract_features:
            features = self.feature_extractor.extract_features(dataset)
        else:
            features = []
            for x, _ in dataset:
                features.extend([xs.numpy() for xs in x])
            features = np.stack(features, axis=0)
        # features = np.maximum(features, -features) # FIXME 
        labels = []
        for _, y in dataset:
            labels.extend([ys.item() for ys in y])
        return self.classifier.fit_predict(features, labels)
    
    @staticmethod
    def from_config(config: Dict) -> 'FewShotLearner':
        feature_extractor = FeatureExtractor(config)
        classifier = get_classifier(config["classifier"])
        return FewShotLearner(feature_extractor, classifier)