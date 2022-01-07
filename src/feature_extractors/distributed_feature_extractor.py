import logging
import math
import os
import shutil
from typing import Callable, Dict
import numpy as np
import torch
from torch import nn
import torch.distributed as dist

from src.metrics.stats_reporter import StatsReporter
from src.data import get_dataset
from .models_factory import get_model
from src.utils.path_utils import get_model_path

class DistributedFeatureExtractor:
    
    def __init__(self, config: Dict):
        torch.multiprocessing.set_start_method('forkserver')
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=config["num_nodes"],
            rank=config["node_id"]
        )
        logging.info(f"Number of gpus: {torch.cuda.device_count()}")
        self.DEVICE = "cpu"
        if torch.cuda.is_available():
            self.DEVICE = "cuda:0"
        self.model = get_model(config["model"]).to(self.DEVICE)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
            device_ids=[0],
            output_device=0,
        )
        self.config = config
        self.path = os.path.join(config["results_path"], get_model_path(config))

    def train(self) -> None:
        rotation = self.config.get("rotation", 0)
        if rotation:
            self.rotation_classifier = nn.Linear(self.config["features_dim"], 4).to(self.DEVICE)
            self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}, {'params': self.rotation_classifier.parameters()}], lr=self.config['learning_rate'])
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        lf = lambda x: (1 + math.cos(x * math.pi / self.config['scheduler_mocked_epochs'])) / 2 * 0.9 + 0.1
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        loss_func = nn.CrossEntropyLoss()
        metrics = {
            'loss': loss_func,
            'acc': lambda input, target: (torch.max(input, 1)[1] == target).sum() / float(target.shape[0]),
        }
        if self.config["node_id"] == 0:
            self.stats_reporter = StatsReporter(metrics, self.path)
            if rotation:
                self.stats_reporter.add_metrics(["loss_class", "loss_rotation"])

        if self.config["model"].get("load_path") is not None:
            self.load("best.pt", self.config["model"].get("load_path"))
        
        train_dataset, val_dataset = get_dataset(
            self.config["dataset"],
            self.config["datasets_path"],
            self.config["meta_split"],
            train_test_split=True
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=self.config["num_nodes"],
            rank=self.config["node_id"],
            shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=self.config["num_nodes"],
            rank=self.config["node_id"],
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = self.config["batch_size"],
            sampler = train_sampler,
            num_workers = self.config["num_workers"],
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = self.config["batch_size"],
            sampler = val_sampler,
            num_workers = self.config["num_workers"],
            drop_last=True,
        )

        bestLoss = 1e10
        for epoch in range(self.config["epochs"]):
            logging.info(f"Epoch {epoch}")
            train_sampler.set_epoch(epoch)
            self.model.train()
            self.single_epoch(
                train_loader,
                loss_func,
                metrics,
                gradient_accumulation=self.config.get("gradient_accumulation", 1),
                is_training=True)

            with torch.no_grad():
                self.model.eval()
                loss = self.single_epoch(
                    val_loader,
                    loss_func,
                    metrics,
                    gradient_accumulation=self.config.get("gradient_accumulation", 1),
                    is_training=False)['loss']

                if self.config["node_id"] == 0 and loss < bestLoss:
                    bestLoss = loss
                    self.save('best.pt')
            if self.config["node_id"] == 0:
                self.save('current.pt')
            if epoch % self.config['persist_state_every'] == self.config['persist_state_every'] - 1 and os.path.exists(f'{self.path}/best.pt'):
                shutil.copy(f'{self.path}/best.pt', f'{epoch}_checkpoint.pt')

            self.scheduler.step()

    def single_epoch(self, loader: torch.utils.data.DataLoader, loss_func: Callable, metrics: Dict[str, Callable]={}, gradient_accumulation: int=1, is_training: bool=False) -> None:
        metric_values = {}
        for metric in metrics.keys():
            metric_values[metric] = []
        samples = []

        if is_training:
            mixup_alpha = self.config.get("mixup_alpha", None)
            rotation = self.config.get("rotation", 0)
            if rotation:
                metric_values["loss_class"] = []
                metric_values["loss_rotation"] = []
        else:
            mixup_alpha = None
            rotation = 0

        for i, (x, y) in enumerate(loader):
            x, y = x.to(self.DEVICE), y.to(self.DEVICE)
            logging.info(f"{x.shape}, {y.shape}")

            if mixup_alpha is not None:
                lambda_ = np.random.beta(mixup_alpha, mixup_alpha)
                _, y_pred, y_a, y_b = self.model(x, y, lambda_)
                loss = (1-lambda_)*loss_func(y_pred, y_a) + lambda_*loss_func(y_pred, y_b)
                if is_training:
                    loss.backward()
                
            if rotation == 1:
                n = x.shape[0]
                if mixup_alpha is not None:
                    n//=4 # if rotation is not a single training objective take only
                x_new, y_class, y_rotation = [], [], []
                import matplotlib.pyplot as plt
                for x_i, y_i in zip(x[:n], y[:n]):
                    for j in range(4):
                        x_new.append(x_i)
                        x_i = x_i.transpose(1, 2).flip(2)
                        y_class.append(y_i)
                        y_rotation.append(torch.tensor(j))
                x_new = torch.stack(x_new, dim=0)
                y_class = torch.stack(y_class, dim=0)
                y_rotation = torch.stack(y_rotation, dim=0).to(self.DEVICE)
                features, y_class_pred = self.model(x_new)
                y_rotation_pred = self.rotation_classifier(features)
                loss_class = loss_func(y_class_pred, y_class)
                loss_rotation = loss_func(y_rotation_pred, y_rotation)
                loss = (loss_class+loss_rotation)/2
                if is_training:
                    loss.backward()
                metric_values["loss_class"].append(loss_class.cpu().detach())
                metric_values["loss_rotation"].append(loss_rotation.cpu().detach())
                if mixup_alpha is None:
                    y_pred = y_class_pred[::4]
            if rotation == 0 and mixup_alpha is None:
                _, y_pred = self.model(x)
                loss = loss_func(y_pred, y)
                if is_training:
                    loss.backward()
            logging.info(f"Loss: {loss.cpu().detach()}")
            for metric, f in metrics.items():
                metric_values[metric].append(f(y_pred, y).cpu().detach())
            samples.append(x.shape[0])
            if is_training:
                if i%gradient_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        logging.info(f"Iter: {i}")

        if is_training and i%gradient_accumulation != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.config["node_id"] == 0:
            samples = np.array(samples)
            metric_keys = list(metric_values.keys())
            for key in metric_keys:
                metric_values[key] = np.sum(np.array(metric_values[key])*samples)/np.sum(samples)
            self.stats_reporter.update(metric_values, is_training=is_training)

        return metric_values

    def extract_features(self, dataset: torch.utils.data.DataLoader) -> np.ndarray:
        self.model.eval()
        if hasattr(self.model, "extract_features"):
            feature_extractor = self.model.extract_features
        else:
            raise RuntimeError(f"{self.model} has no extract_features method")
        with torch.no_grad():
            features = []
            from tqdm import tqdm
            for x, y in tqdm(dataset):
                features += [feature_extractor(x).numpy().reshape(len(y),-1)]
        return np.concatenate(features, axis=0)

    def save(self, filename: str) -> None:
        path = os.path.join(self.path, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, path)

    def load(self, filename: str, path=None) -> None:
        if path is None:
            path = self.path
        checkpoint = torch.load(os.path.join(path, filename), map_location=self.DEVICE)
        model_state = checkpoint['model']
        if list(model_state.keys())[0].startswith('module') == False:
            from collections import OrderedDict
            new_model_state = OrderedDict()
            for key in model_state.keys():
                new_model_state[f"module.{key}"] = model_state[key]
            model_state = new_model_state
        self.model.load_state_dict(model_state, strict=True)
        if hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if hasattr(self, "scheduler"):
            self.scheduler.load_state_dict(checkpoint['scheduler'])