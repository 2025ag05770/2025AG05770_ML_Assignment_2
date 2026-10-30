import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from explore_data import TARGET
from metrics_utils import compute_metrics
from train_models import get_filename

MODEL_DIR = "model"
TEST_FILE = "test_data.csv"

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")
st.title("Dry Bean Classifier")
st.write("Upload a CSV with the 16 bean features and a `Class` column, pick a model and see the metrics.")


@st.cache_resource
def load_models():
    names = pd.read_csv(MODEL_DIR + "/metrics.csv")["Model"].tolist()
    models = {}
    for name in names:
        models[name] = joblib.load(MODEL_DIR + "/" + get_filename(name))
    le = joblib.load(MODEL_DIR + "/label_encoder.joblib")
    return models, le


models, le = load_models()
features = list(models[list(models)[0]].feature_names_in_)

# sidebar
st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
use_test = st.sidebar.checkbox("Use bundled test data", value=False)
model_name = st.sidebar.selectbox("Model", list(models))

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_test:
    df = pd.read_csv(TEST_FILE)
else:
    st.info("Upload a CSV or tick 'Use bundled test data' in the sidebar.")
    st.stop()

# check columns and labels
missing = [c for c in features + [TARGET] if c not in df.columns]
if len(missing) > 0:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

unknown = sorted(set(df[TARGET]) - set(le.classes_))
if len(unknown) > 0:
    st.error("Unknown labels in Class column: " + ", ".join(str(u) for u in unknown))
    st.stop()

st.subheader("Data preview")
st.write(f"{len(df)} rows")
st.dataframe(df.head(10))

# predict
x = df[features]
y_true = le.transform(df[TARGET])
model = models[model_name]
pred = model.predict(x)
prob = model.predict_proba(x)
classes = list(range(len(le.classes_)))
res = compute_metrics(y_true, pred, prob, classes)

st.subheader("Metrics for " + model_name)
cols = st.columns(6)
i = 0
for key in res:
    if pd.isna(res[key]):
        text = "n/a"
    else:
        text = f"{res[key]:.4f}"
    cols[i].metric(key, text)
    i = i + 1

# confusion matrix (all 7 classes are always shown, even if a class is missing in the file)
st.subheader("Confusion matrix")
view = st.radio("View", ["Counts", "Row-normalised"], horizontal=True)
cm = confusion_matrix(y_true, pred, labels=classes)
if view == "Counts":
    cm_show = cm
    fmt = "d"
else:
    row_sum = cm.sum(axis=1, keepdims=True)
    # rows with no samples stay 0 instead of dividing by zero
    cm_show = np.divide(cm, row_sum, out=np.zeros(cm.shape), where=row_sum > 0)
    fmt = ".2f"
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm_show, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
plt.close(fig)

# per-class report
st.subheader("Classification report")
rep = classification_report(
    y_true, pred, labels=classes, target_names=le.classes_,
    output_dict=True, zero_division=0,
)
rep_df = pd.DataFrame(rep).T
rep_df = rep_df.drop(index="accuracy", errors="ignore")
rep_df["support"] = rep_df["support"].astype(int)
st.dataframe(rep_df.round(4))

# weakest class: lowest recall, only among classes that are in the file
class_rows = rep_df.loc[list(le.classes_)]
present = class_rows[class_rows["support"] > 0]
weakest = present["recall"].idxmin()
st.info(f"Weakest class by recall: {weakest} (recall {present.loc[weakest, 'recall']:.4f}, "
        f"{present.loc[weakest, 'support']} samples)")


# download predictions
st.subheader("Download predictions")
out_df = df.copy()
out_df["Predicted"] = le.inverse_transform(pred)
st.download_button(
    "Download predictions CSV",
    out_df.to_csv(index=False).encode(),
    file_name="predictions.csv",
    mime="text/csv",
)


