# pytorch-magnet-loss
PyTorch implementation of ["Metric Learning with Adaptive Density Discrimination"](https://arxiv.org/pdf/1511.05939.pdf) by Rippel, Paluri, Dollar, and Bourdev (ICLR 2016).

## To Do

-   [ ] Clean up code and get basic pairwise running with Oxford-IIIT
-   [ ] Get set of experiments run by Magnet Loss and write proposal with set of ideas
-   [ ] Test pairwise loss
-   [ ] Implement Triplet Loss
-   [ ] Implement Magnet Loss
-   [ ] Get complete results on Oxford-IIIT before moving to other datasets.

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

This is a ["Metric Learning with Adaptive Density Discrimination"](https://arxiv.org/pdf/1511.05939.pdf).

