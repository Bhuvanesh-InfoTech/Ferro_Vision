import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st
from PIL import Image
import utils as U

st.set_page_config(page_title="Image Classification · Ferro Vision", page_icon="🔍", layout="wide")
st.title("🔍 Casting Defect Detection")
st.write("Upload a top-view photo of a pump-impeller casting. The model returns **Defective** or **Normal** with a confidence score.")

cfg = U.load_json(U.ARTIFACT_DIR / "class_config.json")
if cfg is None:
    U.stop_missing("Image-model configuration not found.", ["artifacts/class_config.json"])

sizes = cfg.get("input_sizes", {cfg["best_model"]: cfg["best_model_input_size"]})
available = [f for f in sizes if U.model_exists(f)]
if not available:
    U.stop_missing("No image models found.", [f"models/{f}" for f in sizes])

pretty = {"custom_cnn.keras": "Custom CNN", "vgg16.keras": "VGG16", "mobilenetv2.keras": "MobileNetV2"}
best = cfg["best_model"] if cfg["best_model"] in available else available[0]

with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Model", available, index=available.index(best),
                          format_func=lambda f: pretty.get(f, f) + (" (best)" if f == cfg["best_model"] else ""))
    threshold = st.slider("Decision threshold", 0.05, 0.95, float(cfg.get("threshold", 0.5)), 0.05,
                          help="Probability above which a part is flagged Defective. Lower it to miss fewer defects (more false alarms).")
    st.caption(f"Input size: {sizes[choice]}×{sizes[choice]} px")

file = st.file_uploader("Casting image", type=["jpg", "jpeg", "png", "bmp"])

if file is None:
    st.info("Upload an image to classify it.")
else:
    try:
        image = Image.open(file).convert("RGB")
    except Exception:
        st.error("Could not read that file as an image.")
        st.stop()

    model = U.load_model(choice)
    with st.spinner("Analysing..."):
        p_def = U.predict_image(model, image, sizes[choice])

    is_def = p_def >= threshold
    label = "Defective" if is_def else "Normal"
    confidence = p_def if is_def else 1 - p_def

    left, right = st.columns([1, 1])
    left.image(image, caption=f"Uploaded image ({image.size[0]}×{image.size[1]})")
    with right:
        if is_def:
            st.error(f"### ❌ {label}")
        else:
            st.success(f"### ✅ {label}")
        st.metric("Confidence", f"{confidence:.1%}")
        st.progress(float(p_def), text=f"Probability defective: {p_def:.1%}")
        st.caption(f"Model: {pretty.get(choice, choice)} · threshold {threshold:.2f}")
        if 0.35 <= p_def <= 0.65:
            st.warning("Borderline score. Consider a manual inspection of this part.")
