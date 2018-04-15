# pytorch-magnet-loss
PyTorch implementation of ["Metric Learning with Adaptive Density Discrimination"](https://arxiv.org/pdf/1511.05939.pdf) by Rippel, Paluri, Dollar, and Bourdev (ICLR 2016).

## To Do

-   [ ] Clean up code and get basic pairwise running with Oxford-IIIT
-   [ ] Get set of experiments run by Magnet Loss and write proposal with set of ideas
-   [ ] Test pairwise loss
-   [ ] Implement Triplet Loss
-   [ ] Implement Magnet Loss
-   [ ] Get complete results on Oxford-IIIT before moving to other datasets.


## Software Dependencies

- PyTorch 3.1
- Torch Vision
- sklearn
- protobuf
- tensorboardX

### Implementations to write

- [ ] Model
    - [x] Inception V3
    - [ ] GoogLeNet -- what's actually used by the paper, but seems weird and irrelevant
- [ ] Losses
    - [x] Pairwise Loss
    - [ ] Triplet Loss
    - [ ] Magnet Loss
        - [ ]  Neighborhood Sampling
        - [ ] Cluster index
- [ ] Dataset Wrappers
    - [ ] ImageNet
    - [x] Oxford-IIIT Pet
    - [ ] Stanford Dogs
    - [ ] Oxford 102 Flowers

## Introduction

This is a ["Metric Learning with Adaptive Density Discrimination"](https://arxiv.org/pdf/1511.05939.pdf) implementation in PyTorch.

## Implementation

### Datasets

#### Oxford IIIT

The Oxford-IIIT Pet Dataset consists of 37 classes (25 Dog Breeds and 12 Cat Breeds). There are roughly 200 images for each class.
The images have a large variations in scale, pose and lighting.
All images have an associated ground truth annotation of breed, head ROI, and pixel level tri-map segmentation.


### Procedure Details

#### Cluster Indexing

In the paper, the authors do not fully specify how they performed the book-keeping for the indexing.
In order to simplify the implementation, the following changes were made that should not affect the actual procedure being done.

1.  Rather than implement the sampling procedure described in Section 3.2 Component #1,
uniform sampling of the clusters ins performed and the losses are instead weighted based on the estimate $p_i$.

2. All the sampling and loss calculations are performed within the forward pass of the loss calculation per batch,
where the batch size is much larger than that determined by the paper's calculations (DxM).
The larger batch size minimizes the difficulty of directly sampling the smaller batches,
without affecting the calculation required.

3. This method of sampling does not implement the importance sampling correlated with L_1 as mentioned in the paper,
however, through weighting the cluster specific losses with the importance term that would have been used for sampling.
