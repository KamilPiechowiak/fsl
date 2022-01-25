# Few-Shot Learning

This repository contains a reimplementation of the following paper: [Squeezing Backbone Feature Distributions to the Max for Efficient Few-Shot Learning](https://arxiv.org/pdf/2110.09446v1.pdf) \[1\]. Algorithms presented in this work are an extension of the methods used in the following paper: [Leveraging the Feature Distribution
in Transfer-based Few-Shot Learning](https://arxiv.org/pdf/2006.03806v3.pdf) \[2\]. That is why we also implemented algorithms from the second paper.
In Few-Shot Learning the dataset is divided into 3 subsets: *D_meta-train*, *D_meta-val*, *D_meta-test*. These subsets contain pairwise disjoint sets of classes. *D_meta-train* contains labeled samples used to pretrain the backbone model (in this scenario a neural network). *D_meta-val* is used to create smaller subsets to adjust the few-shot learning algorithm. For this purpose, *n* classes are sampled and for each class *s* training and *q* test examples are sampled. The task is to use *ns* training samples to adapt a classifer to correctly recognize *nq* samples. *D_meta-test* is used to evaluate the performance of the model on the previously unseen data.

In the aforementioned papers, the backbone network is trained on *D_meta-train* and later used as a feature extractor. Few-shot classification is performed using features computed with the backbone and then applying various algorithms on a set of latent vectors.

## Backbone training
We reimplemented two backbone networks [ResNet18](src/feature_extractors/models/resnet.py) and [WideResnet26](src/feature_extractors/models/wide_resnet.py). The authors have also used ResNet12, but there is neither description nor the code of the architecture available. The question about ResNet12 on [github](https://github.com/yhu01/PT-MAP/issues/26) remains unanswered.
At first, we thought that the networks were trained using standard classification objective with the cross entropy error. However that approach did not give satisfactory results. We found that the authors followed [Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://arxiv.org/pdf/1907.12087.pdf) \[3\]. The backbones' training consists of two phases. In the first stage, each input image is rotated by different angles and the auxiliary goal of the model is to predict the rotation angle. A set of 4 rotation angles was used: {0°, 90°, 180°, 270°}. Additionally, classification loss is used. In the second phase, model is fine-tuned with Manifold Mixup. This is a modification of a mixup augmentation technique which can be applied not only to the input of the network but also to the input of any layer. Along with Manifold Mixup loss, rotation and classification losses are also used. The goal of the Manifold Mixup is to create nicely separated groups for different classes, so that when new classes arrive they have sparse regions between clusters in the feature space.
The training of a model is implemented in [feature_extrator.py](src/feature_extractors/feature_extractor.py).

## Features computation
Computing features for each sample is the most computationally expensive step in the implemented few-shot classification algorithms. That is why we precompute features for every image in the *D_meta-val* and *D_meta-test* subsets in advance and then use them as inputs to the classification algorithm.

## Few-shot classification
We implemented classification algorithms present in both papers: [PT+NCM](src/classifiers/pt_ncm.py), [PT+K-means](src/classifiers/pt_kmeans.py), [PT+MAP](src/classifiers/pt.py), [PEME-BMS](src/classifiers/peme.py). We originally planned to reproduce values computed by the authors in tables 1 and 2 in paper \[1\], but as more algorithms were implemented we should be able to reproduce parts of table 5 in \[2\] and table 4 in \[1\].

## Training
We execute training on a SLURM cluster. Example script is available [here](scripts/run.sh). Unfortunately, due to computational costs we had to constrain ourselves to 2 datasets (CifarFS and MiniImageNet).

## Results
We trained both backbones on CifarFS and MiniImageNet. The table below compares the results obtained using our backbones in two different training regimes. We marked them with suffixes:
- MM - training with manifold_mixup (no rotation)
- S2M2 - training with rotation and then manifold_mixup+rotation

The authors made the WideResNet's weights and features available for both datasets. We use them in our comparison. We also include the results reported in the paper. We use the following suffixes in this case:
- Weights - we use weights provided by the authors, but compute the feature vectors ourselves. We then apply algorithms on the feature vectors.
- Features - we use feature vectors provided by the authors and apply algorithms on them.
- Paper - values provided by the authors in the paper. We just copy  values from the paper.

All experiments were performed using *n=5* (classes), *q=15* (unlabelled samples per class) and *s=1 or 5* (labelled samples per class). Number in the header indicates the number of labelled samples per class. Every experiment (corresponding to a single cell in the table below) was repeated 10000 times. We also included 95% confidence intervals in the table (as did the authors of the original paper). To see the full table content you may need to scroll. Results are also avaiable in a separate file [results.csv](results.csv).

| dataset      | model                 | NCM-1      | NCM-5      | PT_NCM-1   | PT_NCM-5   | PT_Kmeans-1 | PT_Kmeans-5 | PT-1       | PT-5       | PEME-1     | PEME-5     |
|--------------|-----------------------|------------|------------|------------|------------|-------------|-------------|------------|------------|------------|------------|
| CifarFS      | Resnet18-MM           | 57.1±0.22  | 71.91±0.19 | 59.55±0.21 | 76.48±0.18 | 66.97±0.26  | 79.5±0.17   | 71.69±0.26 | 81.33±0.17 | 74.59±0.29 | 67.45±0.42 |
| CifarFS      | Resnet18-S2M2         | 65.46±0.22 | 78.45±0.18 | 67.68±0.22 | 81.82±0.16 | 75.15±0.24  | 84.3±0.16   | 80.7±0.26  | 85.95±0.16 | 80.27±0.27 | 85.46±0.16 |
| CifarFS      | Resnet18-Paper        | 56.4±0     | 78.3±0     | 71.41±0.22 | 85.5±0.15  | 79.97±0.23  | 86.74±0.16  | 84.8±0.25  | 88.55±0.16 | 84.16±0.24 | 89.39±0.15 |
| CifarFS      | WideResnet26-MM       | 50.49±0.23 | 64.24±0.22 | 63.32±0.21 | 79.41±0.17 | 71.48±0.25  | 84.16±0.16  | 74.41±0.23 | 85.14±0.15 | 81.19±0.27 | 86.54±0.15 |
| CifarFS      | WideResnet26-S2M2     | 54.76±0.25 | 70.53±0.23 | 65.38±0.22 | 80.58±0.17 | 73.67±0.25  | 84.56±0.16  | 78.78±0.24 | 86.03±0.16 | 82.0±0.26  | 86.67±0.16 |
| CifarFS      | WideResnet26-Weights  | 70.45±0.22 | 83.91±0.17 | 72.72±0.21 | 85.59±0.16 | 81.98±0.24  | 88.65±0.15  | 85.34±0.22 | 89.74±0.15 | 86.16±0.24 | 89.57±0.15 |
| CifarFS      | WideResnet26-Features | 71.92±0.22 | 84.42±0.17 | 73.24±0.21 | 85.87±0.16 | 81.98±0.23  | 88.78±0.15  | 85.72±0.23 | 89.8±0.15  | 86.03±0.23 | 89.55±0.15 |
| CifarFS      | WideResnet26-Paper    | 68.93±0    | 86.81±0    | 74.64±0.21 | 87.64±0.15 | 83.69±0.22  | 89.19±0.15  | 87.69±0.23 | 90.68±0.15 | 86.93±0.23 | 91.18±0.15 |
| MiniImageNet | Resnet18-MM           | 43.77±0.18 | 56.35±0.16 | 48.72±0.19 | 64.0±0.16  | 53.34±0.23  | 67.46±0.17  | 57.72±0.28 | 69.4±0.18  | 57.34±0.29 | 62.42±0.3  |
| MiniImageNet | Resnet18-S2M2         | 48.77±0.19 | 63.39±0.18 | 54.38±0.2  | 70.32±0.16 | 60.31±0.23  | 75.27±0.16  | 62.55±0.22 | 75.83±0.16 | 67.91±0.29 | 77.66±0.18 |
| MiniImageNet | Resnet18-Paper        | 47.63±0    | 72.89±0    | 62.5±0.2   | 82.17±0.14 | 73.08±0.22  | 84.67±0.14  | 80.0±0.27  | 86.96±0.14 | 79.3±0.27  | 87.94±0.14 |
| MiniImageNet | WideResnet26-MM       | 44.16±0.18 | 57.24±0.17 | 51.38±0.19 | 66.09±0.16 | 56.04±0.24  | 70.63±0.17  | 60.67±0.27 | 72.31±0.18 | 61.01±0.3  | 72.24±0.19 |
| MiniImageNet | WideResnet26-S2M2     | 48.47±0.19 | 61.51±0.18 | 55.66±0.2  | 70.5±0.16  | 60.81±0.24  | 76.06±0.16  | 65.76±0.26 | 77.32±0.16 | 67.67±0.29 | 77.75±0.17 |
| MiniImageNet | WideResnet26-Weights  | 41.31±0.21 | 53.11±0.23 | 53.83±0.2  | 67.7±0.17  | 60.06±0.24  | 76.08±0.17  | 59.73±0.2  | 74.48±0.16 | 67.8±0.3   | 52.85±0.58 |
| MiniImageNet | WideResnet26-Features | 46.65±0.25 | 57.89±0.26 | 63.23±0.2  | 76.31±0.17 | 70.51±0.24  | 85.51±0.13  | 70.65±0.2  | 84.53±0.13 | 81.47±0.26 | 45.54±0.65 |
| MiniImageNet | WideResnet26-Paper    | 55.31±0    | 78.33±0    | 65.35±0.2  | 83.87±0.13 | 76.67±0.22  | 86.73±0.13  | 82.92±0.26 | 88.82±0.13 | 82.07±0.25 | 89.51±0.13 |

In the table we can see that changing the training regime from manifold mixup to S2M2 provided improvement in an accuracy (up to 10 percentage points in some settings). On CifarFS we obtain lower values than those reported in the paper (by about 4 percentage points). The difference is probably caused by shorter training. We trained for 100+100 epochs compared to 400+100 epochs.
We also compare results obtained with weights or features provided by the authors. We use our implementation of few-shot classification algorithms on them and get very similar results. This shows that our implementation is correct with high confidence.

On MiniImageNet we also get values lower than the authors. Again we trained our backbones for less epochs (ResNet for 100+100, WideResNet for 100+36) than the authors (400+100). This dataset was computationally more demanding than CifarFS. Shorter training is reflected in the results. However, in the case of WideResNet we usually obtain better results than with WideResNet with authors' weights. It's interesting that the results obtained with authors' weights and authors' features are very different. Theoretically, they should be the same. Features should have been computed using those weights. One explanation could be that the authors included wrong checkpoint of the model. Another possibility is that they manipulated the feature vectors after computation. Still, the authors report in the paper values even higher than those we obtained with their features. That could indicate that there's an error in our implementations, but good results on CifarFS make it less probable. The difference occurs also on simple algorithms like NCM (nearest class mean). Making such a mistake that the algorithm works but in a slightly worse way is not a straightforward task (if there was a mistake, simple algorithm should work very badly - there is not much place for errors).
