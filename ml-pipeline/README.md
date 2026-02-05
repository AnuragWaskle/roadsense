# RoadSense ML Pipeline

This directory contains the Python scripts to train the Pothole Detection model (TCN-BiLSTM).

## 1. Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Data Preparation
We use multiple sources (Kaggle, accelerometer.xyz, etc.). 
1. Download the datasets from the links provided in the main `task.md` or Implementation Plan.
2. Place the raw CSV files in `raw_data/`.
3. The training script expects a common schema: `timestamp, ax, ay, az, gx, gy, gz, label`.
   - `label`: 0 (Smooth), 1 (Pothole), 2 (SpeedBump)

## 3. Training
Run the training script:
```bash
cd src
python train.py
```
- If no data is found in `raw_data/`, it will generate **Synthetic Data** for verification.
- The model will be saved to `models/final/road_sense_model.h5`.
- A TFLite model will be exported to `models/final/road_sense_model.tflite`.

## 4. Key Files
- `src/model.py`: TCN-BiLSTM Keras model definition.
- `src/preprocessing.py`: Sliding window (128 samples) & Feature extraction.
- `src/train.py`: Main loop (Load -> Split -> Train -> Save -> Convert).
