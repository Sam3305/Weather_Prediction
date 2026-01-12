# Weather Prediction System Using Neural Networks
## Technical Report

### Executive Summary
This project implements a neural network-based weather prediction system that forecasts next-day temperature from recent weather conditions and seasonal signal using `synthetic_weather.csv`. The system demonstrates an end-to-end ML pipeline using TensorFlow/Keras for modeling and Streamlit for deployment.

### Technical Architecture

1. **Data Processing**
   - Input features standardized using StandardScaler (`scaler_X.joblib`)
   - Target variable standardized with a separate StandardScaler (`scaler_y.joblib`) and inverse-transformed to °C at inference
   - Dataset: `synthetic_weather.csv`

2. **Model Architecture**
   - Framework: TensorFlow 2.x with Keras API
   - Architecture: Sequential Neural Network
   - Input Features (7): `temp_lag1`, `temp_lag2`, `humidity`, `wind_speed`, `pressure`, `precip`, `day_of_year`
   - Output: 1 node (next-day temperature, °C)
   - Optimizer: Adam
   - Loss Function: Mean Squared Error

3. **Implementation**
   - Development Environment:
     - Python 3.10
     - Jupyter Notebook (`Weather_Prediction_NN.ipynb`)
     - Key Libraries: TensorFlow, scikit-learn, pandas, numpy
   - Deployment:
     - Streamlit web interface (`app.py`)
   - Data Pipeline:
     - Feature scaling → Model prediction → Inverse transformation via `scaler_y`

### Results and Evaluation
- Training Performance:
  - Mean Absolute Error and RMSE are computed in the notebook on the test split
- Model Validation:
  - Visual inspection (Actual vs Predicted plot)

### Deployment
- Web Application Features:
  - Interactive input fields for the 7 model features
  - Real-time predictions and delta display
- System Requirements:
  - Python 3.8+
  - Dependencies listed in `requirements.txt`

### Model Output Processing
- Predictions are produced in °C by inverse-transforming the model’s standardized output with `scaler_y`
- Output is clipped to a broad physical range [-50°C, 50°C] only to prevent outliers

### Validation
- Input validation ensures parameters are within physical limits
- Warning system flags large predicted day-over-day changes

### Notes on Inputs and Lags
- The model expects `temp_lag1` and `temp_lag2`. When unavailable, the app allows users to provide them. If users do not know these values, using the current temperature as a proxy reduces accuracy.
- `day_of_year` captures seasonality and defaults to today’s value but can be adjusted.

### Limitations
- Based on synthetic data
- Limited features; no geolocation or broader meteorological context
- Heuristic handling of missing lag temperatures may reduce accuracy
- No integration with external weather APIs

### Future Improvements
- Incorporate real weather datasets and location context
- Use time-series models and better lag/rolling features
- Uncertainty estimation and calibration
- External API integration and richer visualizations