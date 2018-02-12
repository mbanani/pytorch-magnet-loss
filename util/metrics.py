"""
    A set of evaluation metrics for clustering

    NMI, Purity and RI are calculated using equations in the following link:
    https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

    Hungarian Algorithm obtained from https://github.com/tdedecko/hungarian-algorithm

"""

import numpy    as np

from sklearn    import cluster
from .hungarian import Hungarian
from IPython    import embed

def confusion_matrix_st(preds, labels):
        num_inst   = len(preds)
        num_labels = np.max(labels) + 1

        conf_matrix = np.zeros( (num_labels, num_labels))

        for i in range(0, num_inst):
            gt_i = labels[i]
            pr_i = preds[i]
            conf_matrix[gt_i, pr_i] = conf_matrix[gt_i, pr_i] + 1

        return conf_matrix

def confusion_matrix(preds, labels):
        num_inst   = preds.shape[0]
        num_labels = preds.shape[1]
        preds      = np.argmax(preds, axis=1)

        conf_matrix = np.zeros( (num_labels, num_labels))

        for i in range(0, num_inst):
            gt_i = labels[i]
            pr_i = preds[i]
            conf_matrix[gt_i, pr_i] = conf_matrix[gt_i, pr_i] + 1

        return conf_matrix

def correct_confusion_matrix(conf_matrix):
    num_labels = conf_matrix.shape[0]
    new_conf = np.zeros( (num_labels, num_labels) )
    hungarian = Hungarian(conf_matrix, is_profit_matrix = True)
    hungarian.calculate()
    results = hungarian.get_results()
    correct = np.zeros(num_labels)
    for i in range(0, num_labels):
        for j in range(0, num_labels):
            new_conf[results[i][0], results[j][0]] = conf_matrix[results[i][0], results[j][1]]
    print new_conf
    return new_conf

def get_hungarian_results(conf_matrix):
    hungarian = Hungarian(conf_matrix, is_profit_matrix = True)
    hungarian.calculate()
    results = hungarian.get_results()
    return results

def calculate_accuracy(conf_matrix):
    num_labels = conf_matrix.shape[0]
    hungarian = Hungarian(conf_matrix, is_profit_matrix = True)
    hungarian.calculate()
    results = hungarian.get_results()

    correct = np.zeros(num_labels)
    for i in range(0, num_labels):
        correct[results[i][0]] = conf_matrix[results[i][0], results[i][1]]

    overall_accuracy = np.sum(correct) / np.sum(conf_matrix)
    return overall_accuracy, correct

def calculate_purity(conf_matrix):
    # purity = 1/N * sum_{c = clusters} max_{t= classes} (union of cluster_t and class_t)
    num_inst    = np.sum(conf_matrix).astype(float)
    best_unions = np.amax(conf_matrix, axis = 0)
    return np.sum(best_unions) / num_inst

def calculate_NMI(conf_matrix):
    num_labels = conf_matrix.shape[0]
    _min_float = 1e-12

    conf_matrix = conf_matrix.astype(float)
    total_inst  = np.sum(conf_matrix)

    prob_matrix = conf_matrix / total_inst
    P_gt        = np.sum(prob_matrix, axis=1)
    P_pred      = np.sum(prob_matrix, axis=0)

    # Clip for numerical stability
    prob_matrix = np.clip(prob_matrix, a_min = _min_float, a_max = None)
    P_gt        = np.clip(P_gt, a_min = _min_float, a_max = None)
    P_pred      = np.clip(P_pred, a_min = _min_float, a_max = None)


    mutual_info = 0.0

    for i in range(0, num_labels):
        for j in range(0, num_labels):
            mutual_info += prob_matrix[i,j] * np.log(prob_matrix[i,j] / (P_gt[i] * P_pred[j]))

    entropy_gt   = -1. * np.sum(P_gt * np.log(P_gt))
    entropy_pred = -1. * np.sum(P_pred * np.log(P_pred))

    NMI = 2. * mutual_info / (entropy_pred + entropy_gt)

    return NMI

def calculate_spectral_clustering_accuracy(np_activations, np_labels, num_clusters):

    activations = np_activations[0]
    labels      = np_labels[0]

    # Compute spectral clustering accuracy
    for i in range(1, len(np_activations)):
        activations = np.concatenate((activations, np_activations[i]), axis=0)
        labels      = np.concatenate((labels, np_labels[i]), axis=0)

    # END FOR

    # Setup of spectral clustering algorithm
    algorithm = cluster.SpectralClustering(n_clusters=num_clusters)

    # Obtain predictions from spectral clustering model
    predictions = algorithm.fit_predict(activations)

    # Compute confusion matrix from predictions
    conf_matrix = confusion_matrix_st(predictions, labels)

    # Use hungarian algorithm to compare cluster labels vs. true class labels
    hungarian = Hungarian(conf_matrix, is_profit_matrix = True)
    hungarian.calculate()
    results = hungarian.get_results()

    correct = np.zeros(num_clusters)

    for i in range(0, num_clusters):
        correct[results[i][0]] = conf_matrix[results[i][0], results[i][1]]

    # END FOR

    overall_accuracy = np.sum(correct) / np.sum(conf_matrix)

    return overall_accuracy

def calculate_kmeanspp_clustering_accuracy(np_activations, np_labels, num_clusters):

    activations = np_activations[0]
    labels      = np_labels[0]

    # Compute spectral clustering accuracy
    for i in range(1, len(np_activations)):
        activations = np.concatenate((activations, np_activations[i]), axis=0)
        labels      = np.concatenate((labels, np_labels[i]), axis=0)

    # END FOR

    # Setup of kmeans++ clustering algorithm
    algorithm = cluster.KMeans(n_clusters=num_clusters, init='k-means++')

    # Obtain predictions from spectral clustering model
    predictions = algorithm.fit_predict(activations)

    # Compute confusion matrix from predictions
    conf_matrix = confusion_matrix_st(predictions, labels)

    # Use hungarian algorithm to compare cluster labels vs. true class labels
    hungarian = Hungarian(conf_matrix, is_profit_matrix = True)
    hungarian.calculate()
    results = hungarian.get_results()

    correct = np.zeros(num_clusters)

    for i in range(0, num_clusters):
        correct[results[i][0]] = conf_matrix[results[i][0], results[i][1]]

    # END FOR

    overall_accuracy = np.sum(correct) / np.sum(conf_matrix)

    return overall_accuracy

def cluster_metrics(conf_matrix, np_activations, np_labels):
    accuracy, _     = calculate_accuracy(conf_matrix)

    return accuracy

def cluster_metrics_eval(conf_matrix, np_activations, np_labels):
    NMI             = calculate_NMI(conf_matrix)
    purity          = calculate_purity(conf_matrix)
    accuracy, _     = calculate_accuracy(conf_matrix)
    s_accuracy      = calculate_spectral_clustering_accuracy(np_activations, np_labels, conf_matrix.shape[0])
    k_accuracy      = calculate_kmeanspp_clustering_accuracy(np_activations, np_labels, conf_matrix.shape[0])

    return accuracy, purity, NMI, s_accuracy, k_accuracy
