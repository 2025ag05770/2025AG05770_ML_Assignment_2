import joblib
import pandas as pd
import streamlit as st

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
