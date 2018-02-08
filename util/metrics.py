import numpy as np
import scipy.misc

from IPython import embed

class metrics(object):

    def __init__(self):


        self.results_pred   = []
        self.results_label  = []


    """
        Updates the keypoint dictionary
        params:     obj_class       object class                    (batch_size)
                    predictions     predictions for each image      (batch_size x 360*NumClasses)
                    labels          labels for each image           (batch_size x 360*NumClasses)
    """
    def update_dict(self, predictions, labels):
        """Log a scalar variable."""
        if type(predictions) == int:
            predictions = [predictions]
        if type(labels) == int:
            labels = [labels]


        for i in range(0, len(obj_class)):
            start_index = self.class_ranges[obj_class[i]]
            self.results_pred.append( [ np.argmax(predictions[0][i, start_index:start_index+360]),
                                        np.argmax(predictions[1][i, start_index:start_index+360]),
                                        np.argmax(predictions[2][i, start_index:start_index+360])])
            self.results_label.append( [ np.argmax(labels[0][i, start_index:start_index+360]),
                                         np.argmax(labels[1][i, start_index:start_index+360]),
                                         np.argmax(labels[2][i, start_index:start_index+360])])

    def metrics(self):
        pass
