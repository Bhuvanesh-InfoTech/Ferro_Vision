import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st
import utils as U

st.set_page_config(page_title="Ferro Vision", page_icon="🛠️", layout="wide")

st.title("🛠️ Ferro Vision")
st.subheader("RUL Forecasting & Visual Defect Detection")
st.write(
    "Deep learning for industrial asset reliability: predict how many operating cycles a turbofan engine "
    "has left, and flag defective cast components from a photo."
)

c1, c2, c3 = st.columns(3)
c1.markdown("**1 · EDA**\n\nExplore the turbofan sensor data and the casting images.")
c2.markdown("**2 · Regression**\n\nANN predicts RUL from one cycle; RNN / LSTM predict the RUL curve over a 30-cycle window.")
c3.markdown("**3 · Image Classification**\n\nUpload a casting photo and get Defective / Normal with a confidence score.")
st.caption("Use the sidebar to switch pages.")

st.divider()
st.subheader("Project status")

checks = {
    "Turbofan data": U.TURBOFAN_CSV,
    "Scaler": U.ARTIFACT_DIR / "scaler.pkl",
    "Regression config": U.ARTIFACT_DIR / "config.json",
    "ANN model": U.MODEL_DIR / "ann.keras",
    "RNN model": U.MODEL_DIR / "rnn.keras",
    "LSTM model": U.MODEL_DIR / "lstm.keras",
    "Image-model config": U.ARTIFACT_DIR / "class_config.json",
    "Custom CNN": U.MODEL_DIR / "custom_cnn.keras",
    "VGG16": U.MODEL_DIR / "vgg16.keras",
    "MobileNetV2": U.MODEL_DIR / "mobilenetv2.keras",
}
cols = st.columns(5)
for i, (name, path) in enumerate(checks.items()):
    cols[i % 5].markdown(f"{'✅' if path.exists() else '❌'} {name}")
if not all(p.exists() for p in checks.values()):
    st.info("Some files are missing. Run the notebooks to generate models and artifacts; "
            "the turbofan CSV goes in `data/turbofan.csv`.")

reg, clf = U.load_results("regression_results.csv"), U.load_results("classification_results.csv")
if reg is not None or clf is not None:
    st.divider()
    st.subheader("Model results (test set)")
    if reg is not None:
        st.markdown("**RUL regression**")
        st.dataframe(reg.round(3))
    if clf is not None:
        st.markdown("**Defect classification**")
        st.dataframe(clf.round(4))
