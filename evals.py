import torch
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.metrics import roc_auc_score, average_precision_score


class SSDk:
    def __init__(self, train_features, train_labels):
        self.known_inlier = train_features[train_labels == 0]
        self.known_outlier = train_features[train_labels == 1]

    def get_dist(self, z, known, shrunkcov=False):
        if shrunkcov:
            print("Using ledoit-wolf covariance estimator.")
            cov = lambda x: ledoit_wolf(x)[0]
        else:
            cov = lambda x: np.cov(x.T, bias=True)

        dist = np.sum(
            (z - np.mean(known, axis=0, keepdims=True))
            * (
                np.linalg.pinv(cov(known)).dot(
                    (z - np.mean(known, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )

        return dist

    def get_score(self, z):
        d_in = self.get_dist(z, self.known_inlier)
        d_out = self.get_dist(z, self.known_outlier, shrunkcov=True)
        s = d_in - d_out
        return s


def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = roc_auc_score(labels, data)
    return auroc


def get_pr_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr = average_precision_score(labels, data)
    return aupr


def get_fpr(xin, xood, tpr=95):
    return np.sum(xood < np.percentile(xin, tpr)) / len(xood)
