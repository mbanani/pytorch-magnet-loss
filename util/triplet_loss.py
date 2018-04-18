"""
A PyTorch implmentation of the KL-Divergence Loss as described in (https://arxiv.org/abs/1511.06321)

Lua Implementation (not inspected yet TODO) (https://github.com/yenchanghsu/NNclustering/blob/master/BatchKLDivCriterion.lua)

"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from IPython import embed

class triplet_loss(nn.Module):
    def __init__(self, alpha = 7.18):
        super(triplet_loss, self).__init__()

        self.alpha  = alpha

    def forward(self, outputs, clusters):
        """
        :param  indices     The index of each embedding
        :param  outputs     The set of embeddings
        :param  clusters    Cluster assignments for each index
        :return Loss        Magnet loss calculated for current batch
        """

        assert not clusters.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

        _min_float  = 1e-6
        num_instances = 0.0

        outputs     = outputs.float()
        clusters    = clusters.cpu().data.numpy()
        batch_size  = outputs.size(0)

        loss        = torch.zeros(1)

        # If GPU is available compute loss on it
        loss    = loss.cuda()
        loss    = torch.autograd.Variable(loss).cuda()


        ######################### Cluster Assignments ##########################

        # Generate a set of clusters in the batch
        # and the local indices corresponding to each of those clusters
        # batch_clusters = { cluster_number : [ local_indices] }
        # TODO fix later!!!  -- for now assiming indices are irrelevant!
        batch_clusters = {}
        for i in range(0, len(clusters)):
            if clusters[i] in batch_clusters.keys():
                batch_clusters[clusters[i]].append(i)
            else:
                batch_clusters[clusters[i]] = [i]


        ######################### Cluster Assignments ##########################
        old_clusters = list(batch_clusters.keys())
        clusters = []
        # remove clusters with less than D instances TODO
        for c in old_clusters:
            if len(batch_clusters[c]) >= 2:
                clusters.append(c)

        ########################## CALCULATE THE LOSS #########################
        instances_1 = []
        instances_2 = []
        instances_3 = []

        for m in range(0, len(clusters)):
            c = clusters[m]
            for d1 in range(0, len(batch_clusters[c]) - 1):
                for d2 in range(d1+1, len(batch_clusters[c])):
                    ins_i1  = batch_clusters[c][d1]
                    ins_i2  = batch_clusters[c][d2]

                    for mN in range(0, len(clusters)):
                        if mN != m:
                            cN = clusters[mN]
                            for dN in range(0, len(batch_clusters[cN])):
                                ins_iN  = batch_clusters[cN][dN]
                                instances_1.append(ins_i1)
                                instances_2.append(ins_i2)
                                instances_3.append(ins_iN)


        return ((outputs[instances_1] - outputs[instances_2]).norm(p=2, dim = 1) + self.alpha - (outputs[instances_1] - outputs[instances_3]).norm(p=2, dim = 1)).clamp(min = 0.0).mean()
