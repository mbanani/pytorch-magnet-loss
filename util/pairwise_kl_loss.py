"""
A PyTorch implmentation of the KL-Divergence Loss as described in (https://arxiv.org/abs/1511.06321)

Lua Implementation (not inspected yet TODO) (https://github.com/yenchanghsu/NNclustering/blob/master/BatchKLDivCriterion.lua)

"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from IPython import embed

class pairwise_kl_loss(nn.Module):
    def __init__(self, margin = 2, hinge = True):
        super(pairwise_kl_loss, self).__init__()
        self.hinge = hinge
        self.margin = margin

    def forward(self, preds, labels):
        """
        :param  preds   Probability predictions over N classes (batch_size, N)
        :param  labels  Probability labels over N classes (batch_size, N)
        :return Loss    Pairwise KL-Divergence loss.
        """

        assert not labels.requires_grad, \
            "nn criterions don't compute the gradient w.r.t. targets - please " \
            "mark these variables as volatile or not requiring gradients"

        _min_float  = 1e-6

        preds       = preds.float().cpu()
        labels      = labels.float().cpu()
        batch_size  = preds.size(0)

        loss        = torch.zeros(1)

        # If GPU is available compute loss on it
        #if torch.cuda.is_available():
        #    loss        = loss.cuda()

        loss        = torch.autograd.Variable(loss)

        # Calculate pairwise for all possible (N * (N-1))/2 pairs
        s_list_a = []
        s_list_b = []
        d_list_a = []
        d_list_b = []


        for inst_a in range(batch_size):
            for inst_b in range(batch_size):

                if inst_a != inst_b:
                    if labels[inst_a].data[0] == labels[inst_b].data[0]:
                        s_list_a.append(inst_a)
                        s_list_b.append(inst_b)

                    else:
                        d_list_a.append(inst_a)
                        d_list_b.append(inst_b)

                    # END IF

                # END IF

            # END FOR

        # END FOR

        assert len(s_list_a) == len(s_list_b)
        assert len(d_list_a) == len(d_list_b)

        # For similar pairs calculate KL divergence
        if len(s_list_a) > 0:
            s_list_a = torch.from_numpy(np.asarray(s_list_a)).long()
            s_list_b = torch.from_numpy(np.asarray(s_list_b)).long()

            loss += F.kl_div(preds[s_list_a].clamp( min = _min_float).log(), preds[s_list_b], size_average = False)

        # END IF

        # For dissimilar pairs compute KL divergence, compare against margin and compute final loss
        custom_KL   = torch.zeros(len(d_list_a))

        # If GPU is available compute loss on it
        #custom_KL   = custom_KL.cuda()

        custom_KL   = torch.autograd.Variable(custom_KL)

        if len(d_list_a) > 0:
            d_list_a = torch.from_numpy(np.asarray(d_list_a)).long()
            d_list_b = torch.from_numpy(np.asarray(d_list_b)).long()

            custom_KL += (preds[d_list_a] * (preds[d_list_a].clamp( min = _min_float).log() - preds[d_list_b].clamp( min = _min_float).log())).sum(dim = 1)

            for index in range(len(d_list_a)):
                if custom_KL[index].data[0] < self.margin:
                    loss += self.margin - custom_KL[index]

                # END IF

            # END FOR

        # END IF

        num_pairs = batch_size * (batch_size - 1) / 2.
        loss      = loss / num_pairs

        # Finally place the computed loss on the GPU to be compatible with training model on GPU
        return loss.cuda()
