import streamlit as st
import numpy as np
from keras.models import load_model
from datetime import datetime, timedelta
import os
from utils import calculate_lengths, create_dataset

# Title
st.title("Menstrual Cycle Predictor")

# Input form
st.markdown("### Enter at least 2 past periods (start and end dates):")
user_input = st.text_area("Enter one period per line as: YYYY-MM-DD,YYYY-MM-DD", height=150)

if user_input:
    try:
        # Parse user input
        lines = user_input.strip().split("\n")
        user_periods = [tuple(line.strip().split(",")) for line in lines if line.strip()]

        # Validate format
        for start, end in user_periods:
            datetime.strptime(start, "%Y-%m-%d")
            datetime.strptime(end, "%Y-%m-%d")

        if len(user_periods) < 2:
            st.warning("Please enter at least 2 valid periods.")
        else:
            # Calculate lengths
            cycle_lengths, m_lengths = calculate_lengths(user_periods)
            data = np.array([[c, m] for c, m in zip(cycle_lengths, m_lengths)])

            steps = min(3, len(data))
            if len(data) >= steps:
                train_x, _ = create_dataset(data, steps)
                last_seq = np.expand_dims(data[-steps:], axis=0)

                # Load model
                model_path = os.path.join("models", "lstm_trained_model.h5")
                model = load_model(model_path)

                # Predict
                y_pred = model.predict(last_seq, verbose=0)[0]
                next_cycle = int(round(y_pred[0]))
                next_menstruation = int(round(y_pred[1]))

                last_period_start = datetime.strptime(user_periods[-1][0], "%Y-%m-%d")
                predicted_start = last_period_start + timedelta(days=next_cycle)
                predicted_end = predicted_start + timedelta(days=next_menstruation - 1)

                st.success(f"**Predicted Next Period Start:** {predicted_start.strftime('%Y-%m-%d')}")
                st.success(f"**Predicted Next Period End:** {predicted_end.strftime('%Y-%m-%d')}")
                st.info(f"Cycle Length: {next_cycle} days\n\nMenstruation Length: {next_menstruation} days")
            else:
                st.warning("Not enough data to form prediction sequence. Please enter more periods.")

    except Exception as e:
        st.error(f"Invalid input format or error occurred: {e}")
