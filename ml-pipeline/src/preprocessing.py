import pandas as pd
import numpy as np
import os
from scipy import stats

# Constants
WINDOW_SIZE = 128  # ~2.56 seconds at 50Hz
STEP_SIZE = 64     # 50% overlap
SAMPLING_RATE = 50 # Hz

LABELS = {
    'Smooth': 0,
    'Pothole': 1,
    'SpeedBump': 2
}

def load_and_preprocess_data(data_dir):
    """
    Loads CSV files from data_dir, resamples to 50Hz, and creates sliding windows.
    Expected CSV columns: timestamp, ax, ay, az, gx, gy, gz, label
    """
    X = []
    y = []
    
    print(f"Scanning {data_dir} for CSV files...")
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Returning empty arrays.")
        return np.array(X), np.array(y)

    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(filepath)
                print(f"Processing {filename}...")
                
                # TODO: formatting/resampling logic here depending on specific dataset structure
                # For now, assuming data is already somewhat clean or synthetic
                
                # Check for required columns
                required_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'label']
                if not all(col in df.columns for col in required_cols):
                    print(f"Skipping {filename}: Missing columns")
                    continue
                
                # Create Windows
                for i in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
                    window = df.iloc[i : i + WINDOW_SIZE]
                    
                    # Features
                    features = window[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
                    X.append(features)
                    
                    # Label (Majority vote in window)
                    labels = window['label'].values
                    mode_label = stats.mode(labels)[0]
                    
                    # Handle Scalar vs Array return type of stats.mode depending on scipy version
                    if isinstance(mode_label, np.ndarray):
                        mode_label = mode_label[0]
                        
                    y.append(mode_label)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return np.array(X), np.array(y)

def create_synthetic_data(num_samples=1000):
    """Generate synthetic data for testing the pipeline"""
    print("Generating synthetic data...")
    X = np.random.randn(num_samples, WINDOW_SIZE, 6)
    y = np.random.randint(0, 3, size=(num_samples,))
    return X, y

if __name__ == "__main__":
    # Test
    X, y = create_synthetic_data(10)
    print(f"Generated shape: X={X.shape}, y={y.shape}")
