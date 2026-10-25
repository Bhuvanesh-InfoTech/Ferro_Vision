import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
import utils as U

st.set_page_config(page_title="EDA · Ferro Vision", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")

dataset = st.radio("Dataset", ["Turbofan sensor data", "Casting image data"], horizontal=True)


@st.cache_data
def sensor_correlations(df):
    feats = [c for c in df.columns if c.startswith("sensor") or c.startswith("op_setting")]
    return df[feats + ["rul"]].corr(method="spearman")["rul"].drop("rul").sort_values(key=abs, ascending=False)


# ================================================================== turbofan
if dataset == "Turbofan sensor data":
    df = U.load_turbofan()
    if df is None:
        U.stop_missing("Turbofan data not found.", ["data/turbofan.csv"])

    sensors = [c for c in df.columns if c.startswith("sensor")]
    life = df.groupby("engine_id")["cycle"].max()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Engines", df["engine_id"].nunique())
    m3.metric("Sensors", len(sensors))
    m4.metric("Median engine life", f"{int(life.median())} cycles")

    st.subheader("Summary statistics")
    st.dataframe(df.drop(columns=["engine_id"]).describe().T.round(3))

    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(df, x="rul", nbins=50, title="RUL distribution (all rows)"))
    c2.plotly_chart(px.histogram(life.reset_index(), x="cycle", nbins=30, title="Engine lifetime (max cycle per engine)"))

    st.subheader("Sensor trends across cycles")
    corr = sensor_correlations(df)
    default_sensors = [c for c in corr.index if c.startswith("sensor")][:3]
    cc1, cc2 = st.columns([1, 3])
    engine = cc1.selectbox("Engine", sorted(df["engine_id"].unique()))
    chosen = cc2.multiselect("Sensors", sensors, default=default_sensors)
    if chosen:
        g = df[df["engine_id"] == engine]
        long = g.melt(id_vars="cycle", value_vars=chosen, var_name="sensor", value_name="value")
        fig = px.line(long, x="cycle", y="value", color="sensor", facet_row="sensor", height=200 * len(chosen) + 80)
        fig.update_yaxes(matches=None)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)
    else:
        st.info("Pick at least one sensor.")

    st.subheader("Which features move with RUL?")
    bar = corr.reset_index()
    bar.columns = ["feature", "spearman"]
    st.plotly_chart(px.bar(bar.sort_values("spearman"), x="spearman", y="feature", orientation="h",
                           title="Spearman correlation with RUL", height=max(350, 22 * len(bar))))
    st.caption("Sensors with the largest absolute correlation carry the most degradation signal.")

# ================================================================== casting
else:
    index = U.casting_index()
    if index is None:
        U.stop_missing("Casting images not found.", ["data/casting_data/train/...", "(or data/casting_split/train/...)"])

    counts = {k: len(v) for k, v in index.items()}
    total = sum(counts.values())
    m1, m2, m3 = st.columns(3)
    m1.metric("Training images", f"{total:,}")
    m2.metric("Normal", f"{counts['normal']:,}")
    m3.metric("Defective", f"{counts['defective']:,}", f"{counts['defective'] / total:.0%} of total", delta_color="off")

    st.plotly_chart(px.bar(x=list(counts.keys()), y=list(counts.values()), color=list(counts.keys()),
                           color_discrete_map={"normal": "#4c72b0", "defective": "#c44e52"},
                           labels={"x": "class", "y": "images"}, title="Class balance (training images)"))

    st.subheader("Sample images")
    if "eda_seed" not in st.session_state:
        st.session_state["eda_seed"] = 0
    st.button("🔀 Show different images", on_click=lambda: st.session_state.update(eda_seed=st.session_state["eda_seed"] + 1))
    rng = np.random.RandomState(st.session_state["eda_seed"])
    for label in ("normal", "defective"):
        st.markdown(f"**{label.capitalize()}**")
        paths = rng.choice(index[label], size=min(5, len(index[label])), replace=False)
        cols = st.columns(5)
        for col, p in zip(cols, paths):
            col.image(Image.open(p))

    sample = Image.open(index["normal"][0])
    st.caption(f"Image size: {sample.size[0]}×{sample.size[1]} px, mode {sample.mode}.")
