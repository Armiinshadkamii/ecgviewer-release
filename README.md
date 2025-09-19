# ECG Feature Extraction and Classification App

This application provides a modular, clinical-grade pipeline for ECG signal processing, feature extraction, and heartbeat classification. It includes a Tkinter-based GUI for interactive visualization and classification of ECG segments, powered by a pre-trained machine learning model.

---

## 🧠 Overview

The app performs the following tasks:

- Loads ECG signals from CSV or WFDB formats
- Detects R-peaks and segments heartbeats
- Extracts clinically relevant features (RR interval, QRS duration, P-wave width, etc.)
- Classifies each heartbeat using a trained ExtraTreesClassifier model
- Displays results and confidence scores in a responsive GUI

---

## 🖥️ GUI Features

- **Upload File**: Load ECG data from CSV or WFDB `.hea` files
- **Classify Heartbeat**: Start classification of ECG segments
- **Pause/Continue**: Control the classification flow
- **Segment Navigation**: Browse previous/next heartbeat segments
- **Live Plot**: Visualize ECG waveform with Matplotlib
- **Results Panel**: View predicted class, confidence, and extracted features

The GUI is built using `ttkbootstrap` for a modern dark-themed interface and `matplotlib` for waveform visualization. Classification runs asynchronously to keep the UI responsive.

---

## 🧬 Feature Extraction

Each heartbeat segment is analyzed to extract:

- `rr_interval`: Time between successive R-peaks
- `r_peak_amp`: Amplitude of the R-peak
- `qrs_duration`: Duration of the QRS complex
- `p_wave_width`: Width of the P-wave
- `min_voltage`, `max_voltage`, `mean_voltage`, `std_voltage`: Signal statistics

These features are computed using slope analysis, moving averages, and statistical summaries. The `ECGFeatureExtractor` class handles this logic.

---

## 🤖 Classification

The classifier uses a pre-trained ExtraTreesClassifier (trained by me on mit-bih database) model with:

- **Scaler**: `StandardScaler` for feature normalization
- **Encoder**: `LabelEncoder` for label transformation
- **Model**: `KNeighborsClassifier` trained on balanced ECG feature data

### Supported Labels

- `N`: Normal beat
- `V`: Ventricular
- `S`: Supraventricular
- `F`: Fusion
- `Q`: Unknown
- `U`: Unknown

The classifier returns both the predicted label and a confidence score (0–100%).

---

## 📦 Data Sources

Supported formats:

- **CSV**: Plain signal values, one column, no header
- **WFDB**: MIT-BIH `.hea` files (converted internally)

The `ECGDataLoader` handles both formats and automatically detects R-peaks using `scipy.signal.find_peaks`.

---

## 🧪 Model Evaluation

The classifier module includes:

- `get_score(x, y)`: Returns accuracy score on test data
- `display_confusion_matrix(x, y)`: Plots confusion matrix using `matplotlib`

These utilities help validate model performance and interpret classification results.

---

## 🧰 Developer Notes

- Modular architecture with separation of concerns
- Threaded data loading to prevent UI blocking
- Compatible with Windows and Linux
- Uses `joblib` for model persistence and `pandas` for feature handling
- UI state management includes pause/resume logic and segment navigation
- Includes `requirements.txt` for reproducible setup



