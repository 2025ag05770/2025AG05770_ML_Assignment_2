import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba, classes):
    # y_true / y_pred are label numbers, y_proba columns follow the order of classes
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # AUC needs every class to be present in y_true, otherwise give NaN
    if set(classes).issubset(set(y_true)):
        auc = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=classes
        )
    else:
        auc = np.nan

    return {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
    }
