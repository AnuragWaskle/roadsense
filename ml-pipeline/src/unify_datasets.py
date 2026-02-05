import pandas as pd
import os

# Helper to Unify Datasets into the standard format:
# timestamp, ax, ay, az, gx, gy, gz, label
# Label Mapping: 0=Smooth, 1=Pothole, 2=SpeedBump

INPUT_DIR = "../raw_downloads"
OUTPUT_DIR = "../raw_data"

def normalize_kaggle_dataset(file_path):
    """
    Example normalizer for Kaggle Pothole dataset.
    Adjust column names based on specific dataset.
    """
    try:
        df = pd.read_csv(file_path)
        # Rename columns to standard (example mapping, verify with actual CSV)
        # Assumes dataset has generic names like 'Time', 'AccX', etc.
        # This is a template - MODIFY based on the actual downloaded file headers.
        
        # Example transformation:
        # df = df.rename(columns={'Time': 'timestamp', 'AccX': 'ax', ...})
        
        # Add label if missing or map string labels to int
        # df['label'] = 1 # Force label if file is pure pothole data
        
        output_path = os.path.join(OUTPUT_DIR, "normalized_" + os.path.basename(file_path))
        df.to_csv(output_path, index=False)
        print(f"Normalized {file_path} -> {output_path}")
    except Exception as e:
        print(f"Error normalizing {file_path}: {e}")

if __name__ == "__main__":
    # Create normalized folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Example usage:
    # normalize_kaggle_dataset("../raw_downloads/dataset_1.csv")
    print("Unification script ready. Uncomment logic to process specific files.")
