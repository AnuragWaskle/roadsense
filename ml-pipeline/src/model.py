import tensorflow as tf
from tensorflow.keras import layers, models

def build_tcn_bilstm_model(input_shape=(128, 6), num_classes=3):
    """
    Builds a TCN-BiLSTM model for pothole detection.
    
    Args:
        input_shape: (time_steps, features) -> Default (128, 6)
        num_classes: Number of output classes -> Default 3 (Smooth, Pothole, SpeedBump)
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # 1. TCN Block (Temporal Convolutional Network)
    # Using dilated convolutions to capture long-range dependencies
    x = layers.Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu', padding='causal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu', padding='causal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(filters=64, kernel_size=3, dilation_rate=4, activation='relu', padding='causal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # 2. BiLSTM Block (Bidirectional Long Short-Term Memory)
    # To capture context from both past and future (within the window)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)
    
    # 3. Classification Head
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="RoadSense_TCN_BiLSTM")
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    model = build_tcn_bilstm_model()
    model.summary()
