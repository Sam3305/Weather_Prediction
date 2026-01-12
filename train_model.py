import os
import argparse

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Expect columns: date,temp,humidity,wind_speed,pressure,precip
    required = {'date', 'temp', 'humidity', 'wind_speed', 'pressure', 'precip'}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Dataset missing required columns: {missing}")

    df['temp_lag1'] = df['temp'].shift(1)
    df['temp_lag2'] = df['temp'].shift(2)
    df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear
    # Target: next-day temperature
    df['target_temp'] = df['temp'].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


def get_matrices(df: pd.DataFrame):
    feature_cols = [
        'temp_lag1',
        'temp_lag2',
        'humidity',
        'wind_speed',
        'pressure',
        'precip',
        'day_of_year',
    ]
    X = df[feature_cols].values.astype(float)
    y = df['target_temp'].values.astype(float)
    return X, y


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def main():
    parser = argparse.ArgumentParser(description='Train weather model.')
    parser.add_argument(
        '--data', default='synthetic_weather.csv', help='Path to dataset CSV'
    )
    args = parser.parse_args()

    rng_seed = 42
    tf.random.set_seed(rng_seed)
    np.random.seed(rng_seed)

    data_path = args.data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    df_feat = build_features(df)
    X, y = get_matrices(df_feat)

    # Non-shuffled split to preserve temporal structure
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    model = build_model(input_dim=X_train.shape[1])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Evaluate
    y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    mae = float(np.mean(np.abs(y_pred - y_test)))
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))

    print(f"Dataset: {data_path}")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Test MAE: {mae:.3f} °C, RMSE: {rmse:.3f} °C")

    # Save artifacts
    model.save('weather_model.h5')
    dump(scaler_X, 'scaler_X.joblib')
    dump(scaler_y, 'scaler_y.joblib')
    print(
        'Model and scalers saved: weather_model.h5, scaler_X.joblib, '
        'scaler_y.joblib'
    )


if __name__ == '__main__':
    main()
