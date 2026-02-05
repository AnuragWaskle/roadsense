# RoadSense

RoadSense is a crowdsourced road quality monitoring system using smartphone sensors and deep learning.

## Project Structure

This monorepo is organized into four main components:

### 📱 `/mobile`
**Tech Stack:** React Native (Expo), TensorFlow Lite, React Native Maps.
- `app/`: Expo Router application screens.
- `src/services/`:
    - `sensor.service.ts`: Handles Accelerometer/Gyroscope data collection at 50Hz.
    - `tflite.service.ts`: Bridge to the TCN-BiLSTM .tflite model for inference.
    - `background-tasks/`: Logic for running data collection in the background.

### 🌐 `/web`
**Tech Stack:** React.js, Vite, Leaflet.js, Tailwind CSS.
- `src/components/Map/`:
    - `MapContainer.jsx`: Main Leaflet map instance.
    - `HeatmapLayer.jsx`: Visualizes pothole density.
- `src/pages/`: Admin dashboard views.

### 🧠 `/ml-pipeline`
**Tech Stack:** Python, TensorFlow/Keras.
- `raw_data/`: Sensor datasets (Kaggle/Collected).
- `processed_data/`: Windowed time-series data.
- `models/`:
    - `final/`: Exported .tflite models for the mobile app.
- `src/`: Training scripts for the TCN-BiLSTM model.

### 🗄️ `/backend`
**Tech Stack:** Supabase (PostgreSQL + PostGIS).
- `supabase/migrations/`: SQL definitions for geospatial tables.
- `supabase/functions/`: Edge functions.

## Implementation Phases
1. **ML Pipeline:** Train and export the model.
2. **Mobile:** specific sensor collection and TFLite integration.
3. **Integration:** Connect ML model to real-time mobile data.
4. **Web:** Visualize results.
