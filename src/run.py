import argparse
import os
import random
from typing import List, Dict
import logging
from tqdm import tqdm

import numpy as np
import torch
from src.classifiers.classifiers_factory import get_classifier
from src.data.few_shot_dataset import FewShotDataset
from src.data import get_dataset
from src.feature_extractors import FeatureExtractor, DistributedFeatureExtractor
from src.few_shot_learner import FewShotLearner

from src.utils.file_utils import read_json, save_json, save_pickle, read_pickle
from src.utils.path_utils import get_path

def run(config: Dict) -> None:
    if config["stage"] == "pretrain":
        if config.get("distributed", 0) == 1:
            feature_extractor = DistributedFeatureExtractor(config)
        else:
            feature_extractor = FeatureExtractor(config)
        feature_extractor.train()
    elif config["stage"] == "compute_features":
        dataset: FewShotDataset = get_dataset(
            config["dataset"],
            config["datasets_path"],
            config["meta_split"]
        )
        feature_extractor = FeatureExtractor(config)
        feature_extractor.load('best.pt')
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = config["batch_size"],
            shuffle = False,
            num_workers = config["num_workers"],
        )
        labels = np.concatenate([y for _, y in loader])
        features = feature_extractor.extract_features(loader)
        features_dict = {}
        for feature, label in zip(features, labels):
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(feature)
        
        meta_split = config["meta_split"]
        save_pickle(features_dict, os.path.join(feature_extractor.path, f"{meta_split}_features.pickle"))
    else:
        feature_extractor = FeatureExtractor(config)
        if config["model"].get("pretrained", 0) != 1 and config["dataset"].get("extract_features", 1) == 1:
            feature_extractor.load("best.pt")
        classifier = get_classifier(config["classifier"])
        few_shot_learner = FewShotLearner(feature_extractor, classifier)

        dataset: FewShotDataset = get_dataset(
            config["dataset"],
            config["datasets_path"],
            config["meta_split"],
            model_path = feature_extractor.path
        )
        
        accuracies = []
        for _ in tqdm(range(config["classification"]["num_iterations"])):
            small_dataset = dataset.sample_random_dataset(
                config["classification"]["num_classes"],
                config["classification"]["num_known_samples_per_class"],
                config["classification"]["num_unknown_samples_per_class"]
            )
            loader = torch.utils.data.DataLoader(
                small_dataset,
                batch_size = config["batch_size"],
                shuffle = False,
                num_workers = config["num_workers"],
            )
            y_predicted = np.array(few_shot_learner.fit_predict(loader, extract_features=config["dataset"].get("extract_features", True)))
            y_true = np.array(small_dataset.get_true_labels())
            accuracies+= [(y_predicted == y_true).mean()]
            if accuracies[-1] < 0.25:
                print(y_predicted, y_true)
        print(accuracies)
        print(np.mean(accuracies))
        save_json(accuracies, os.path.join(config["results_path"], 'evaluation', get_path(config), "accuracies.json"))
    
def create_tasks(config: Dict) -> List[Dict]:
    tasks = []
    for task in config["tasks"]:
        tasks += [{
            **config["general"].copy(),
            **task.copy()
        }] * task.get("repeat", 1)
    return tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Path to config file")
    parser.add_argument('nodeid', help="Id of the node")
    parser.add_argument('numnodes', help="Number of the nodes")
    parser.add_argument('--log', default="INFO", help="Logging level")
    args = parser.parse_args()
    tasks = create_tasks(read_json(args.config))
    node_id, num_nodes = int(args.nodeid), int(args.numnodes)

    logging.basicConfig(level=getattr(logging, args.log.upper(), None), format=f"{node_id}: %(asctime)s %(message)s")

    RANDOM_STATE = node_id
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    for i in range(node_id, len(tasks), num_nodes):
        logging.info(f"Running task {i}")
        tasks[i]["node_id"] = node_id
        tasks[i]["num_nodes"] = num_nodes
        run(tasks[i])