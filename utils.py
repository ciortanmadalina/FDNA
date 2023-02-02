import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import accuracy_score, auc, roc_curve, confusion_matrix


def label_matching(y, pred):
    """_summary_

    Args:
        y (_type_): _description_
        pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    aucs = []
    pred_to_gt = {}
    for cluster_label in np.unique(pred):
        pidx = np.where(pred == cluster_label)[0]
        
        gt_label = Counter(y[pidx]).most_common()[0][0]
        cidx = np.where(y == gt_label)[0]
        pred_to_gt[cluster_label] = gt_label

        binary_y = np.zeros(len(y)).astype(int)
        binary_pred = np.zeros(len(pred)).astype(int)

        binary_y[cidx] = 1
        binary_pred[pidx] =1

        fpr, tpr, thresholds = roc_curve(binary_y, binary_pred, pos_label = 1)

        aucs.append(auc(fpr, tpr))
        print(f"Predicted cluster {cluster_label} corresponds to gt {gt_label} and has AUC {aucs[-1]}")
    mean_auc = np.mean(aucs)
    print(f"Average aucs {mean_auc}")
    return pred_to_gt

def evaluate(y, pred):
    """Evaluates the predicted clustering on the test set using a one vs rest AUC,
    ARI and a confusion matrix

    Args:
        y (_type_): _description_
        pred (_type_): _description_
    """
    print("ARI score: ", adjusted_rand_score(y, pred))
    aucs = []
    for c in np.unique(y):
        cidx = np.where(y == c)[0]
        pidx = np.where(pred == c)[0]
        binary_y = np.zeros(len(y)).astype(int)
        binary_pred = np.zeros(len(pred)).astype(int)
        binary_y[cidx] = 1
        binary_pred[pidx] = 1

        fpr, tpr, thresholds = roc_curve(binary_y, binary_pred, pos_label=1)
        aucs.append(auc(fpr, tpr))
        print(f"Ground truth class {c} has AUC {aucs[-1]}")
    mean_auc = np.mean(aucs)
    print(f"Average aucs {mean_auc}")
    print('Confusion matrix')
    print(confusion_matrix(y, pred))