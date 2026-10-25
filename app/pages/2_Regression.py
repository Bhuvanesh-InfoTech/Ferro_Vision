import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import utils as U

st.set_page_config(page_title="Regression · Ferro Vision", page_icon="📉", layout="wide")
st.title("📉 Remaining Useful Life (RUL) Prediction")

# ------------------------------------------------------------------ load artifacts
cfg = U.load_json(U.ARTIFACT_DIR / "config.json")
scaler = U.load_scaler()
if cfg is None or scaler is None:
    U.stop_missing("Regression artifacts not found.", ["artifacts/config.json", "artifacts/scaler.pkl"])

feature_cols, window = cfg["feature_cols"], int(cfg["window"])
results = U.load_results("regression_results.csv")
df = U.load_turbofan()
means = dict(zip(feature_cols, scaler.mean_))

models = {}
for label, fname in [("ANN", "ann.keras"), ("RNN", "rnn.keras"), ("LSTM", "lstm.keras")]:
    if U.model_exists(fname):
        models[label] = U.load_model(fname)
if not models:
    U.stop_missing("No regression models found.", ["models/ann.keras", "models/rnn.keras", "models/lstm.keras"])

st.sidebar.markdown("### Typical test error (RMSE)")
for name in ("ANN", "RNN", "LSTM"):
    r = U.final_step_rmse(results, name)
    st.sidebar.write(f"**{name}**: {'±' + format(r, '.1f') + ' cycles' if r else 'n/a'}")
st.sidebar.caption("From the notebook's test engines, final step of each window.")

tab1, tab2 = st.tabs(["Single cycle → ANN", f"Window of {window} cycles → RNN / LSTM"])

# ================================================================== tab 1: ANN
with tab1:
    st.markdown("Enter one cycle's operating settings and sensor readings (raw, unscaled values). "
                "The ANN predicts the cycles remaining from this snapshot alone.")

    for f in feature_cols:                       # defaults = training-data means
        st.session_state.setdefault(f"in_{f}", float(means[f]))

    if df is not None:
        with st.container(border=True):
            st.markdown("**Optional: load a real cycle from the dataset**")
            a, b, c = st.columns([1, 1, 1])
            eng = a.selectbox("Engine", sorted(df["engine_id"].unique()), key="ann_engine")
            max_c = int(df.loc[df["engine_id"] == eng, "cycle"].max())
            cyc = b.number_input("Cycle", min_value=1, max_value=max_c, value=max(1, max_c // 2))

            def load_sample(eng, cyc):
                row = df[(df["engine_id"] == eng) & (df["cycle"] == cyc)].iloc[0]
                for f in feature_cols:
                    st.session_state[f"in_{f}"] = float(row[f])
                st.session_state["sample_actual"] = float(row["rul"])

            c.write("")
            c.button("Load this cycle", on_click=load_sample, args=(eng, int(cyc)))

    with st.expander("Sensor and setting values", expanded=True):
        cols = st.columns(4)
        for i, f in enumerate(feature_cols):
            cols[i % 4].number_input(f, key=f"in_{f}", format="%.4f", step=0.01)

    if "ANN" not in models:
        st.warning("`models/ann.keras` not found.")
    elif st.button("Predict RUL", type="primary"):
        raw = pd.DataFrame([{f: st.session_state[f"in_{f}"] for f in feature_cols}])
        rul = max(float(U.ann_predict(models["ANN"], scaler, feature_cols, raw)[0]), 0.0)
        status, color = U.health_status(rul)

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted RUL", f"{rul:.0f} cycles")
        m2.markdown(f"**Status**\n\n:{color}[{status}]")
        err = U.final_step_rmse(results, "ANN")
        if err:
            m3.metric("Likely range", f"{max(rul - err, 0):.0f} – {rul + err:.0f}", help="Prediction ± typical test RMSE (approximate).")
        st.progress(min(rul / 300, 1.0), text="Remaining life (bar scaled to 300 cycles)")
        if st.session_state.get("sample_actual") is not None:
            actual = st.session_state["sample_actual"]
            st.caption(f"Actual RUL of the loaded dataset cycle: {actual:.0f} (error {abs(rul - actual):.0f}). "
                       "Only meaningful if you haven't edited the values.")
        st.caption("Status bands are illustrative (≥150 healthy, 60-150 monitor, <60 critical).")

# ================================================================== tab 2: RNN / LSTM
with tab2:
    st.markdown(f"Provide the **{window} most recent cycles** of an engine. The sequence models return one RUL estimate "
                "for every cycle in the window, so you can see the trend, not just a single number.")
    options = ["Upload CSV"] + (["Demo window from dataset"] if df is not None else [])
    source = st.radio("Input", options, horizontal=True, index=len(options) - 1)

    template = pd.DataFrame([{**{"cycle": i + 1}, **{f: means[f] for f in feature_cols}} for i in range(window)])
    window_df, actual_curve, error = None, None, None

    if source == "Upload CSV":
        st.download_button("⬇️ Download CSV template", template.to_csv(index=False), "window_template.csv", "text/csv")
        st.caption(f"Required columns: {', '.join(feature_cols)} (optional: `cycle`, `rul`). Case-insensitive.")
        up = st.file_uploader("Recent cycles (CSV)", type=["csv"])
        if up is not None:
            try:
                raw = pd.read_csv(up)
                window_df = U.prepare_window(raw, feature_cols, window)
                if "rul" in window_df.columns:
                    actual_curve = window_df["rul"].to_numpy()
            except Exception as e:
                error = str(e)
    else:
        a, b = st.columns(2)
        eng = a.selectbox("Engine", sorted(df["engine_id"].unique()), key="win_engine")
        max_c = int(df.loc[df["engine_id"] == eng, "cycle"].max())
        end = b.number_input("Window ends at cycle", min_value=window, max_value=max_c, value=min(max_c, max(window, max_c // 2)))
        hist = df[(df["engine_id"] == eng) & (df["cycle"] <= end)]
        window_df = U.prepare_window(hist, feature_cols, window)
        actual_curve = window_df["rul"].to_numpy()

    if error:
        st.error(error)
    elif window_df is None:
        st.info("Upload a CSV or choose the demo window to see predictions.")
    else:
        cycles = window_df["cycle"].to_numpy() if "cycle" in window_df.columns else np.arange(1, window + 1)
        preds = {}
        for name in ("RNN", "LSTM"):
            if name in models:
                preds[name] = np.maximum(U.seq_predict(models[name], scaler, feature_cols, window_df), 0)
        if "ANN" in models:   # per-row ANN estimate for context
            preds["ANN (row by row)"] = np.maximum(U.ann_predict(models["ANN"], scaler, feature_cols, window_df), 0)
        if not preds:
            st.warning("No sequence models found (`models/rnn.keras`, `models/lstm.keras`).")
            st.stop()

        rmse = {n: U.final_step_rmse(results, n.split()[0]) for n in preds}
        best = min((n for n in preds if rmse[n]), key=lambda n: rmse[n], default=None)

        fig = go.Figure()
        if actual_curve is not None:
            fig.add_scatter(x=cycles, y=actual_curve, name="Actual RUL", line=dict(color="black", dash="dash"))
        if best:  # confidence band around the most accurate model
            hi, lo = preds[best] + rmse[best], np.maximum(preds[best] - rmse[best], 0)
            fig.add_scatter(x=np.r_[cycles, cycles[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", mode="none",
                            fillcolor="rgba(70,130,180,0.18)", name=f"{best} ± test RMSE", hoverinfo="skip")
        for name, y in preds.items():
            fig.add_scatter(x=cycles, y=y, name=name, mode="lines+markers", marker=dict(size=4))
        fig.update_layout(title="Predicted RUL at every cycle of the window", xaxis_title="cycle",
                          yaxis_title="RUL (cycles)", legend=dict(orientation="h", y=-0.2), height=450)
        st.plotly_chart(fig)

        st.subheader("Prediction at the latest cycle")
        rows = []
        for name, y in preds.items():
            r = rmse[name]
            rows.append({"Model": name, "Predicted RUL": round(float(y[-1]), 1),
                         "Typical error (±)": round(r, 1) if r else None,
                         "Likely range": f"{max(y[-1] - r, 0):.0f} – {y[-1] + r:.0f}" if r else "n/a"})
        st.dataframe(pd.DataFrame(rows).set_index("Model"))

        if "RNN" in preds and "LSTM" in preds:
            gap = abs(preds["RNN"][-1] - preds["LSTM"][-1])
            st.markdown(f"**Model agreement:** RNN and LSTM differ by **{gap:.1f} cycles** at the latest cycle. "
                        "A small gap suggests a more reliable estimate; a large gap means treat the number with caution.")
        if best:
            status, color = U.health_status(float(preds[best][-1]))
            st.markdown(f"**Asset status ({best}):** :{color}[{status}]")
        st.caption("The shaded band is the model's average test error (RMSE), an approximate context for uncertainty, "
                   "not a statistical guarantee.")
