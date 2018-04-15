"""
    A set of evaluation metrics for clustering

    NMI, Purity and RI are calculated using equations in the following link:
    https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

    Hungarian Algorithm obtained from https://github.com/tdedecko/hungarian-algorithm

"""

import numpy    as np

from sklearn    import cluster
from IPython    import embed

class softkNN_metrics(object):

    def __init__(self, sigma = 1.0, num_classes = 37):
        self.reset()
        self.sigma          = sigma
        self.num_classes    = num_classes

    def reset(self):
        self.embeddings = []
        self.labels     = []

    def update_dict(self, embeddings, labels):


    def get_softmax_probs(self):
        # First calculate an N^2 matrix for distances
        softmax_probs = np.zeros((len(self.embeddings), self.num_classes))
        distances = np.zeros((len(self.embeddings), len(self.embeddings)))

        # Inefficient for now,
        # assuming potentially having 2 sets of embeddings (one annotated, and one is not)
        for i in range(0, len(self.embeddings)):
            for j in range(0, len(self.embeddings)):
                d = np.linalg.norm(self.embeddings[i] - self.embeddings[j])
                distances[i][j] = d


        for i in range(0, len(self.embeddings)):
            elements = list(np.argsort(distances[i])[0:129])
            elements.remove(i)

            # find the top 128 elements
            close_distances = distances[i][elements]
            close_labels    = [self.labels[e] for e in elements]

            # calculate softmax values for all and sum for each
            for j in range(0, len(close_labels)):
                softmax_probs[i][close_labels[j]] += np.exp(-0.5 * close_distances[j] / self.sigma)

            softmax_probs[i] = softmax_probs[i] / np.sum(softmax_probs[i])

        return softmax_probs
    def metrics(self, unique = False):
        softmax_probs = self.get_softmax_probs()

        # Calculate top1/top5/ etc etc
