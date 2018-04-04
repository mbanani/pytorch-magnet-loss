import numpy                as numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot    as plt

from sklearn.manifold   import TSNE
import time


def tSNE_visual(activations, labels, show=False, save_path=None):
    print("Calculating tSNE for array of size: ", activations.shape)
    t_start = time.time()
    tsne = TSNE()
    active_tsne = tsne.fit_transform(activations)
    elapsed_time = time.time() - t_start
    print("tSNE calculated! time elapsed: %.3f seconds" % elapsed_time)
    fig = plt.figure(1, (10, 10))
    plt.scatter(active_tsne[:, 0], active_tsne[:, 1], c=labels)

    if show:
        fig.show()
    if save_path:
        fig.savefig(save_path)
