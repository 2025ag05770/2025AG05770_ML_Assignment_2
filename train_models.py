import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from explore_data import TARGET, load_data
from metrics_utils import compute_metrics

MODEL_DIR = "model"


def get_models():
    # every model gets a scaler first, features are on very different scales
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "SVM (RBF)": SVC(kernel="rbf", C=10, probability=True, random_state=42),
    }
    pipes = {}
    for name in models:
        pipes[name] = Pipeline(
            [("scaler", StandardScaler()), ("clf", models[name])]
        )
    return pipes


def get_filename(name):
    # "SVM (RBF)" -> "svm_rbf.joblib"
    temp = name.lower().replace("(", "").replace(")", "").replace(" ", "_")
    return temp + ".joblib"


def split_data(df):
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    # stratify keeps the class percentages same in train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    return x_train, x_test, y_train, y_test


def main():
    df = load_data()
    print("data loaded:", df.shape)

    x_train, x_test, y_train, y_test = split_data(df)
    print("train:", x_train.shape, "test:", x_test.shape)

    # models need numbers, so change class names to 0..6
    le = LabelEncoder()
    y_train_num = le.fit_transform(y_train)
    y_test_num = le.transform(y_test)
    classes = list(range(len(le.classes_)))

    os.makedirs(MODEL_DIR, exist_ok=True)

    rows = []
    models = get_models()
    for name in models:
        model = models[name]
        print("training", name)
        model.fit(x_train, y_train_num)

        pred = model.predict(x_test)
        prob = model.predict_proba(x_test)
        res = compute_metrics(y_test_num, pred, prob, classes)
        print(res)

        row = {"Model": name}
        row.update(res)
        rows.append(row)

        joblib.dump(model, os.path.join(MODEL_DIR, get_filename(name)))

    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    # hold-out test set with the real class names, the app can use it
    test_df = x_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv("test_data.csv", index=False)

    print(result_df.round(4))


if __name__ == "__main__":
    main()
