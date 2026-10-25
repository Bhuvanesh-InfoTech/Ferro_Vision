# Ferro Vision: RUL Forecasting & Visual Defect Detection

An end-to-end deep learning project for industrial asset reliability, built with TensorFlow/Keras.
It combines **prognostics** (predicting failure before it happens) with **quality inspection**
(catching defects at the point of manufacture).

## Problem statement
Unplanned failure of rotating and mechanical assets causes costly downtime and safety risk.
This project tackles three complementary tasks:

1. **RUL regression:** predict how many operating cycles a turbofan engine has left, from a single
   cycle's sensor snapshot (ANN).
2. **RUL sequence-to-sequence regression:** predict RUL at every step of a 30-cycle window (RNN, LSTM)
   and test whether temporal context beats the static ANN baseline.
3. **Visual defect classification:** classify pump-impeller castings as Defective or Normal using a
   custom CNN and two fine-tuned pretrained backbones (VGG16, MobileNetV2).

## Business use cases
- **Predictive maintenance:** schedule maintenance before failure and reduce unplanned downtime.
- **Operational risk reduction:** flag assets approaching end-of-life early.
- **Automated quality inspection:** replace manual visual inspection on the production line.
- **Cost optimisation:** move from fixed-interval to condition-based maintenance.
- **Asset health monitoring:** feed predictions into a live dashboard.

## Datasets
| Dataset | Size | Task |
|---|---|---|
| Turbofan engine degradation (Industrial Risk and Reliability Assessment) | 61,249 rows × 27 columns, 249 engines | RUL regression |
| Casting product images | 7,348 images, 300×300 px (train 6,633 / test 715) | Binary classification |

## Approach
**RUL regression (Notebook 1)**
- Cleaning: null and duplicate checks, cycle-sequence validation, constant-sensor removal.
- Split **by engine_id** (70/15/15) to prevent leakage; scaler fit on training data only.
- Sliding windows of 30 cycles with a per-step RUL target for the RNN and LSTM.
- Fair comparison: sequence models scored on all steps and on the final step of each window,
  and the ANN scored on the same final-step cycles.
- Stable sequence training: output bias initialised to the mean RUL and gradient clipping.

**Defect classification (Notebook 2)**
- Integrity checks: corrupt files, duplicate and train/test overlap detection, class balance.
- Stratified 80/20 train/validation split; the test set is evaluated once at the end.
- Augmentation (flip, rotation, zoom, brightness) and preprocessing are built into each saved model.
- Transfer learning in two stages: train the new head, then fine-tune the top layers at a low learning rate.
- Class weights to offset the mild class imbalance.

## Results
### RUL regression (test engines, final step of each window)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| ANN | 38.27 | 53.28 | 0.614 |
| RNN | 38.74 | 54.62 | 0.595 |
| LSTM | *fill* | *fill* | *fill* |

### Defect classification (test set, 715 images)
| Model | Accuracy |
|---|---|
| Custom CNN | 97.34% |
| VGG16 | 99.58% |
| MobileNetV2 | 99.44% |

## Key insights
- Sensor readings drift steadily with degradation, and a handful of sensors carry most of the signal.
- *(Write your finding on whether temporal context helped: compare ANN vs RNN vs LSTM.)*
- Pretrained backbones clearly beat the custom CNN, and MobileNetV2 is almost as accurate as VGG16
  at a fraction of the size.
- 64 normal test images also appear in the training set (a dataset property). Validation accuracy
  (clean split) matches test accuracy, so the effect appears small.

## Streamlit app
Multi-page app with real-time predictions:
- **EDA:** dataset selector, summary statistics, RUL distribution, sensor trends, class balance, sample images.
- **Regression:** single-cycle ANN prediction, and RNN/LSTM per-step RUL curve from a 30-cycle window
  (CSV upload or demo window) with an error band and model-agreement indicator.
- **Image classification:** upload a casting photo and get Defective/Normal with a confidence score.

## Tech stack
Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Plotly, Matplotlib, Seaborn, Streamlit.