# pytorch-magnet-loss
PyTorch implementation of ["Metric Learning with Adaptive Density Discrimination"](https://arxiv.org/pdf/1511.05939.pdf) by Rippel, Paluri, Dollar, and Bourdev (ICLR 2016).

## To Do

1. [ ] Clean up code and get basic pairwise running with Oxford-IIIT
2. [ ] Get set of experiments run by Magnet Loss and write proposal with set of ideas
3. [ ] Test pairwise loss
4. [ ] Implement Triplet Loss
5. [ ] Implement Magnet Loss
6. [ ] Get complete results on Oxford-IIIT before moving to other datasets.

### Losses

- [ ] Losses
    - [x] Pairwise Loss
    - [ ] Triplet Loss
    - [ ] Magnet Loss
        - [ ]  Neighborhood Sampling
        - [ ] Cluster index
- [ ] Dataset Wrappers
    - [x] Oxford-IIIT Pet
    - [ ] Stanford Dogs
    - [ ] Oxford 102 Flowers

## Introduction

This is a ["Metric Learning with Adaptive Density Discrimination"](https://arxiv.org/pdf/1511.05939.pdf).

## Citation

This is an implementation of [ ["Metric Learning with Adaptive Density Discrimination"](https://arxiv.org/pdf/1511.05939.pdf),
so please cite the respective papers if you use this code in any published work.
