# streamlit_app.py — Streamlit frontend to use trained model with user input periods

import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from keras.models import load_model
from utils import calculate_lengths, create_dataset

st.set_page_config(page_title="Cycle Predictor", layout="centered")
st.title("📅 Menstrual Cycle Predictor")

st.markdown("Enter at least 2 periods (start and end dates) to predict your next cycle.")

# Input form
with st.form("period_input_form"):
    n_periods = st.number_input("How many past periods do you want to enter?", min_value=2, max_value=20, value=3)
    periods = []
    for i in range(n_periods):
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input(f"Start Date {i+1}", key=f"start_{i}")
        with col2:
            end = st.date_input(f"End Date {i+1}", key=f"end_{i}")
        if start and end:
            periods.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))

    submitted = st.form_submit_button("Predict Next Period")

if submitted:
    if len(periods) < 2:
        st.error("❗ Please enter at least 2 valid period entries.")
    else:
        # Preprocessing
        cycle_lengths, m_lengths = calculate_lengths(periods)
        data = np.array([[c, m] for c, m in zip(cycle_lengths, m_lengths)])

        if len(data) < 3:
            st.error("⚠️ Need at least 3 entries to form a valid sequence for prediction.")
        else:
            # Dataset
            train_x, _ = create_dataset(data, steps=3)
            last_seq = np.expand_dims(data[-3:], axis=0)

            # Load model
            model = load_model("models/lstm_trained_model.h5")
            y_pred = model.predict(last_seq, verbose=0)[0]

            next_cycle = int(round(y_pred[0]))
            next_mens = int(round(y_pred[1]))

            last_start = datetime.strptime(periods[-1][0], "%Y-%m-%d")
            predicted_start = last_start + timedelta(days=next_cycle)
            predicted_end = predicted_start + timedelta(days=next_mens - 1)

            st.success(f"✅ Predicted Start Date: {predicted_start.strftime('%Y-%m-%d')}")
            st.info(f"Predicted End Date: {predicted_end.strftime('%Y-%m-%d')}")
            st.markdown(f"**Cycle Length**: {next_cycle} days")
            st.markdown(f"**Menstruation Length**: {next_mens} days")
