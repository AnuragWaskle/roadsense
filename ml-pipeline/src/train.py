import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from preprocessing import load_and_preprocess_data, create_synthetic_data
from model import build_tcn_bilstm_model

# Config
RAW_DATA_DIR = "../raw_data"
MODEL_SAVE_PATH = "../models/final/road_sense_model.h5"
TFLITE_SAVE_PATH = "../models/final/road_sense_model.tflite"

def train():
    # 1. Load Data
    print("Loading data...")
    X, y = load_and_preprocess_data(RAW_DATA_DIR)
    
    if len(X) == 0:
        print("No data found in raw_data. Generating SYNTHETIC data for demonstration.")
        X, y = create_synthetic_data(num_samples=500)
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    
    # 3. Build Model
    model = build_tcn_bilstm_model(input_shape=(128, 6), num_classes=3)
    
    # 4. Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=10, # Short for demo
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    
    # 5. Save Keras Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # 6. Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization/Quantization (Optional but recommended for mobile)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(TFLITE_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {TFLITE_SAVE_PATH}")

if __name__ == "__main__":
    train()
