import numpy as np
import math

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    idx = y_pred[np.arange(n), y_true]
    cel = -np.mean(np.log(idx))

    return cel