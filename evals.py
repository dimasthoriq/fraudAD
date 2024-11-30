import torch
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import get_features


class SSDk:
    def __init__(self, train_features, train_labels):
        self.known_inlier = train_features[train_labels == 0]
        self.known_outlier = train_features[train_labels == 1]

    def get_dist(self, z, known, shrunkcov=False):
        if shrunkcov:
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


class SAD:
    def __init__(self, train_features, train_labels):
        self.known_inlier = train_features[train_labels == 0]
        self.known_outlier = train_features[train_labels == 1]
        self.epsilon = 1e-6
        c = np.mean(train_features[train_labels == 0], axis=0)
        c[(abs(c) < self.epsilon) & (c < 0)] = -self.epsilon
        c[(abs(c) < self.epsilon) & (c > 0)] = self.epsilon
        self.c = c

    def m_dist(self, z, known, shrunkcov=False):
        if shrunkcov:
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

    def get_mahalanobis_score(self, z):
        d_in = self.m_dist(z, self.known_inlier)
        d_out = self.m_dist(z, self.known_outlier, shrunkcov=True)
        s = d_in - d_out
        return s

    def get_score(self, z, c=None):
        if c is None:
            c = self.c
        elif isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        dist = np.sum((z - c) ** 2, axis=1)
        return dist


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


def evaluate(model, train_loader, test_loader, method):
    train_f, train_l = get_features(model, train_loader)
    test_f, test_l = get_features(model, test_loader)

    if method == 'ssd':
        ssd = SSDk(train_f, train_l)
        pred = ssd.get_score(test_f)
    else:
        sad = SAD(train_f, train_l)
        if method == 'sad':
            pred = sad.get_score(test_f)
        else:
            pred = sad.get_mahalanobis_score(test_f)

    ap = get_pr_sklearn(pred[test_l == 0], pred[test_l == 1])
    fpr = get_fpr(pred[test_l == 0], pred[test_l == 1])
    roc = get_roc_sklearn(pred[test_l == 0], pred[test_l == 1])

    return ap, fpr, roc
