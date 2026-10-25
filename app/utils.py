"""Shared helpers for the Ferro Vision Streamlit app."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, MODEL_DIR, ARTIFACT_DIR = ROOT / "data", ROOT / "models", ROOT / "artifacts"
TURBOFAN_CSV = DATA_DIR / "turbofan.csv"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ----------------------------------------------------------------------------- loading
def stop_missing(message, files):
    """Show a friendly error listing the missing files, then stop the page."""
    st.error(message)
    st.markdown("Missing:\n" + "\n".join(f"- `{f}`" for f in files))
    st.info("Run the notebooks first (they create these files), then refresh this page.")
    st.stop()


@st.cache_data
def load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_results(name):
    path = ARTIFACT_DIR / name
    return pd.read_csv(path, index_col=0) if path.exists() else None


@st.cache_data
def load_turbofan():
    """Turbofan CSV with lower-cased columns, sorted by engine and cycle (None if the file is absent)."""
    if not TURBOFAN_CSV.exists():
        return None
    df = pd.read_csv(TURBOFAN_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    if "rul" not in df.columns:
        df["rul"] = df.groupby("engine_id")["cycle"].transform("max") - df["cycle"]
    return df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


@st.cache_resource
def load_scaler():
    path = ARTIFACT_DIR / "scaler.pkl"
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def load_model(filename):
    from tensorflow import keras  # imported lazily so the Home/EDA pages stay fast
    path = MODEL_DIR / filename
    return keras.models.load_model(path, compile=False) if path.exists() else None


def model_exists(filename):
    return (MODEL_DIR / filename).exists()


def _find_split(root, name):
    if not root.exists():
        return None
    for dirpath, dirnames, _ in os.walk(root):
        if name in dirnames:
            cand = Path(dirpath) / name
            if any(p.is_dir() for p in cand.iterdir()):
                return cand
    return None


@st.cache_data
def casting_index():
    """{'normal': [paths], 'defective': [paths]} for the training images, or None if not found."""
    folder = None
    for root in (DATA_DIR / "casting_split", DATA_DIR / "casting_data"):
        folder = _find_split(root, "train")
        if folder:
            break
    if folder is None:
        return None
    out = {"normal": [], "defective": []}
    for cls_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        label = "defective" if "def" in cls_dir.name.lower() else "normal"
        out[label] += [str(f) for f in sorted(cls_dir.iterdir()) if f.suffix.lower() in IMG_EXT]
    return out


# ----------------------------------------------------------------------------- regression
def final_step_rmse(results, model):
    """Test RMSE of a model on the final step of each window (from artifacts/regression_results.csv)."""
    if results is None or "RMSE" not in results.columns:
        return None
    rows = [(str(i).strip().lower(), i) for i in results.index]
    for key, idx in rows:
        if key.startswith(model.lower()) and "final" in key:
            return float(results.loc[idx, "RMSE"])
    for key, idx in rows:
        if key.startswith(model.lower()):
            return float(results.loc[idx, "RMSE"])
    return None


def health_status(rul):
    """Illustrative traffic-light bands - tune these to your own maintenance policy."""
    if rul >= 150:
        return "Healthy", "green"
    if rul >= 60:
        return "Monitor", "orange"
    return "Critical - plan maintenance", "red"


def ann_predict(model, scaler, feature_cols, raw_df):
    X = scaler.transform(raw_df[feature_cols].astype(float))
    return model.predict(X.astype("float32"), verbose=0).ravel()


def prepare_window(df, feature_cols, window):
    """Validate an uploaded/selected history and return its last `window` rows (or raise ValueError)."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(df) < window:
        raise ValueError(f"Need at least {window} cycles, got {len(df)}.")
    if df[feature_cols].isnull().any().any():
        raise ValueError("The data contains missing values.")
    return df.tail(window).reset_index(drop=True)


def seq_predict(model, scaler, feature_cols, window_df):
    X = scaler.transform(window_df[feature_cols].astype(float))[None].astype("float32")
    return model.predict(X, verbose=0)[0, :, 0]


# ----------------------------------------------------------------------------- images
def predict_image(model, pil_image, size):
    """Probability that a casting image is defective. Models expect raw RGB pixels in [0, 255]."""
    import tensorflow as tf
    arr = np.asarray(pil_image.convert("RGB"), dtype="float32")
    arr = tf.image.resize(arr, (size, size), method="bilinear").numpy()[None]
    return float(model.predict(arr, verbose=0)[0, 0])
