import numpy as np

from metrics_utils import compute_metrics

classes = np.array([0, 1, 2])
y_true = np.array([0, 1, 2, 0, 1, 2])


def one_hot(labels):
    return np.eye(3)[labels]


# test 1: perfect predictions
pred = y_true.copy()
res = compute_metrics(y_true, pred, one_hot(pred), classes)
print("perfect:", res)
for name in res:
    assert res[name] == 1.0, name

# test 2: every prediction wrong
pred = (y_true + 1) % 3
res = compute_metrics(y_true, pred, one_hot(pred), classes)
print("all wrong:", res)
assert res["Accuracy"] == 0.0
assert res["MCC"] < 0

# test 3: class 2 is missing from y_true, so AUC should be NaN
y_true2 = np.array([0, 1, 0, 1])
pred2 = y_true2.copy()
res = compute_metrics(y_true2, pred2, one_hot(pred2), classes)
print("missing class:", res)
assert np.isnan(res["AUC"])
assert res["Accuracy"] == 1.0

print("all tests passed")
