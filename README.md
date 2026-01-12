# Weather Prediction System

A machine learning project using neural networks to predict next-day temperature based on current weather parameters.

## Input Parameters
- Temperature (°C)
- Humidity (%)
- Pressure (hPa)
- Wind Speed (m/s)
- Precipitation (mm)
- Yesterday Temperature (°C)
- 2 Days Ago Temperature (°C)
- Day of Year (1-366)

## Output
The model predicts the next-day temperature in degrees Celsius (°C).

## Project Structure
- `Weather_Prediction_NN.ipynb`: Jupyter notebook containing model development
- `clean_dataset.py`: Cleans synthetic dataset (e.g., precipitation realism)
- `train_model.py`: Deterministic training script to (re)train and save artifacts
- `app.py`: Streamlit web interface for predictions
- `weather_model.h5`: Trained neural network model
- `scaler_X.joblib`: StandardScaler for input features
- `scaler_y.joblib`: StandardScaler for output temperature
- `synthetic_weather.csv`: Training dataset

## Setup and Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

## Model Details
- Type: Neural Network (implemented in TensorFlow/Keras)
- Inputs (7): temp_lag1, temp_lag2, humidity, wind_speed, pressure, precip, day_of_year
- Output: Next-day temperature (°C)
- Scalers: StandardScaler used for feature normalization and target inverse-transform

## Data Cleaning (make synthetic precip more realistic)
1) Clean the dataset:
```bash
python clean_dataset.py --input synthetic_weather.csv --output synthetic_weather_clean.csv
```
2) Retrain on the cleaned dataset:
```bash
python train_model.py --data synthetic_weather_clean.csv
```
3) Verify console output shows:
- Dataset path (clean file)
- Input features: 7
- Reasonable MAE/RMSE

## Data
The project uses synthetic weather data (`synthetic_weather.csv`) for training and testing the model.

## Key Components
### Scalers
- **scaler_X.joblib**: StandardScaler for normalizing input features
- **scaler_y.joblib**: StandardScaler for target variable; transforms predictions back to °C

### Model
- **weather_model.h5**: Trained neural network saved in HDF5 format

### Prediction Pipeline
1. Inputs → `scaler_X` → normalized features
2. Normalized features → model → normalized prediction
3. Normalized prediction → `scaler_y.inverse_transform` → final prediction (°C)

## Retraining and Verification
To ensure the model matches the app features and was trained correctly, retrain with:
```bash
python train_model.py
```
This will:
- Validate the dataset columns exist
- Build lag features (`temp_lag1`, `temp_lag2`) and `day_of_year`
- Train with early stopping
- Print test MAE/RMSE and input feature count (should be 7)
- Save: `weather_model.h5`, `scaler_X.joblib`, `scaler_y.joblib`

After retraining, restart the app:
```bash
streamlit run app.py
```

## Prediction Guidelines
- **Normal Change:** ±10°C
- **Significant Change:** >±15°C
- **Extreme Change:** >±20°C

### Confidence Levels
- **High:** Temperature change within normal range
- **Medium:** Significant temperature change
- **Low:** Extreme temperature change

### Example Output
```
Input: Current=20.0°C, Yesterday=20.0°C, 2-Days-Ago=19.0°C, humidity=50%, wind=5 m/s, pressure=1013 hPa, precip=0 mm, day_of_year=150
Predicted: 24.1°C (+4.1°C)
Confidence: High
```

## Limitations
- Trained on synthetic data; real-world generalization is limited
- Using current temperature as a proxy for unknown lags reduces accuracy; provide realistic lag values when possible
- No location context; predictions are not site-specific