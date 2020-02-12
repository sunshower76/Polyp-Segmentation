import numpy as np

def get_dice(pred, target):
    # in the case of input shape is (1,H,W)
    eps = 1e-5
    inter = np.dot(pred.reshape(-1), target.reshape(-1))
    union = np.sum(pred) + np.sum(target) + eps

    t = (2 * inter + eps) / union
    return t