from typing import Dict

from src.classifiers.pt_kmeans import PT_Kmeans
from .classifier import Classifier
from .pt import PT
from .peme import PEME
from .pt_ncm import PT_NCM
from .pt_kmeans import PT_Kmeans

def get_classifier(config: Dict) -> Classifier:
    config = config.copy()
    name = config["name"]
    del config["name"]
    return {
        "PT": lambda: PT(**config),
        "PEME": lambda: PEME(**config),
        "PT_NCM": lambda: PT_NCM(**config),
        "PT_Kmeans": lambda: PT_Kmeans(**config)
    }[name]()