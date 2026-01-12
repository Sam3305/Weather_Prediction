import streamlit as st
import numpy as np
import tensorflow as tf
from joblib import load
from datetime import datetime


@st.cache_resource
def load_model():
    def mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    model = tf.keras.models.load_model(
        'weather_model.h5', custom_objects={'mse': mse}
    )
    scaler_X = load('scaler_X.joblib')
    scaler_y = load('scaler_y.joblib')
    return model, scaler_X, scaler_y


def validate_prediction(current_temp, predicted_temp):
    """Validate temperature prediction and return appropriate warning level"""
    temp_change = predicted_temp - current_temp
    if abs(temp_change) > 20:
        return (
            "high",
            "🔴 Very large temperature change predicted. Please verify local "
            "weather conditions.",
        )
    elif abs(temp_change) > 15:
        return "medium", "🟡 Significant temperature change predicted."
    elif abs(temp_change) > 10:
        return "low", "ℹ️ Moderate temperature change predicted."
    return "normal", None


def main():
    st.title('Weather Prediction System')
    st.markdown(
        """
        This system predicts next-day temperature based on current weather
        conditions. Normal temperature changes are typically within ±10°C.
        """
    )

    try:
        model, scaler_X, scaler_y = load_model()

        # Inputs aligned with the trained model features
        # Model expects: [temp_lag1, temp_lag2, humidity, wind_speed, pressure,
        # precip, day_of_year]
        col1, col2 = st.columns(2)

        with col1:
            temperature = st.number_input(
                'Current Temperature (°C)', -50.0, 50.0, 20.0
            )
            humidity = st.number_input('Humidity (%)', 0.0, 100.0, 50.0)
            pressure = st.number_input('Pressure (hPa)', 900.0, 1100.0, 1013.0)
            wind_speed = st.number_input('Wind Speed (m/s)', 0.0, 50.0, 5.0)

        with col2:
            precipitation = st.number_input(
                'Precipitation (mm)',
                0.0,
                100.0,
                0.0,
            )
            # Heuristic for missing lag temps: assume same as current temp
            temp_lag1 = st.number_input(
                'Yesterday Temperature (°C, if unknown use current)',
                -50.0,
                50.0,
                float(temperature),
            )
            temp_lag2 = st.number_input(
                '2 Days Ago Temperature (°C, if unknown use current)',
                -50.0,
                50.0,
                float(temperature),
            )

        # Compute day_of_year from today (user can override)
        default_doy = datetime.utcnow().timetuple().tm_yday
        day_of_year = st.slider(
            'Day of Year',
            1,
            366,
            default_doy,
            help='Used to capture seasonality',
        )

        if st.button(
            'Predict Weather', help='Generate next-day temperature forecast'
        ):
            with st.spinner('Generating prediction...'):
                # Arrange inputs to match scaler/model training order
                model_features = np.array(
                    [[
                        temp_lag1,
                        temp_lag2,
                        humidity,
                        wind_speed,
                        pressure,
                        precipitation,
                        day_of_year,
                    ]],
                    dtype=float,
                )

                input_scaled = scaler_X.transform(model_features)
                prediction_scaled = model.predict(input_scaled, verbose=0)
                prediction = scaler_y.inverse_transform(prediction_scaled)
                final_temp = float(np.clip(prediction[0][0], -50.0, 50.0))

                warning_level, warning_message = validate_prediction(
                    temperature, final_temp
                )

                st.success('Prediction generated successfully!')

                # Temperature display
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        'Current Temperature',
                        f'{temperature:.1f}°C',
                        help='Input temperature',
                    )
                with col2:
                    st.metric(
                        'Predicted Next-Day Temperature',
                        f'{final_temp:.1f}°C',
                        f'{final_temp - temperature:+.1f}°C',
                        help=(
                            'Predicted temperature (+ indicates warming, '
                            '- indicates cooling)'
                        ),
                    )

                if warning_message:
                    if warning_level == 'high':
                        st.error(warning_message)
                    elif warning_level == 'medium':
                        st.warning(warning_message)
                    else:
                        st.info(warning_message)

                # Confidence indicator
                confidence = (
                    'Low' if warning_level == 'high' else 'Medium'
                    if warning_level == 'medium' else 'High'
                )
                st.markdown(f"**Prediction Confidence:** {confidence}")

                st.caption(
                    'Note: When past temperatures are unknown, the app uses '
                    'your current temperature for lag features, which reduces '
                    'accuracy. For best results, provide realistic lag values.'
                )

    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
        st.error('Please check if model files exist in the correct location.')
        st.info(
            'Required files: weather_model.h5, scaler_X.joblib, '
            'scaler_y.joblib'
        )


if __name__ == '__main__':
    main()