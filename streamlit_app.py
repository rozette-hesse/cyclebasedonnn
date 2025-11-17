import streamlit as st
import numpy as np
from keras.models import load_model
from datetime import datetime, timedelta
from utils import calculate_lengths, create_dataset

# Load pre-trained model
model = load_model("models/lstm_trained_model.h5")

st.title("Menstrual Cycle Predictor")
st.write("Enter at least 3 periods to predict the next one.")

# Input
periods = []
num_periods = st.number_input("How many past periods do you want to input?", min_value=3, max_value=10, value=3)

with st.form("period_form"):
    for i in range(num_periods):
        st.subheader(f"Period #{i+1}")
        start = st.date_input(f"Start Date {i+1}", key=f"start_{i}")
        end = st.date_input(f"End Date {i+1}", key=f"end_{i}")
        periods.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    submitted = st.form_submit_button("Predict Next Period")

if submitted:
    try:
        # Prepare data
        cycle_lengths, m_lengths = calculate_lengths(periods)
        data = np.array([[c, m] for c, m in zip(cycle_lengths, m_lengths)])

        if len(data) < 3:
            st.error("You need to input at least 4 periods to compute 3 sequences.")
        else:
            x, _ = create_dataset(data, 3)
            last_seq = np.expand_dims(data[-3:], axis=0)
            y_pred = model.predict(last_seq, verbose=0)[0]

            next_cycle = int(round(y_pred[0]))
            next_menstruation = int(round(y_pred[1]))

            last_start = datetime.strptime(periods[-1][0], "%Y-%m-%d")
            pred_start = last_start + timedelta(days=next_cycle)
            pred_end = pred_start + timedelta(days=next_menstruation - 1)

            st.success(f"Predicted Next Period Start: {pred_start.strftime('%Y-%m-%d')}")
            st.success(f"Predicted End: {pred_end.strftime('%Y-%m-%d')}")
            st.info(f"Cycle Length: {next_cycle} days | Menstruation: {next_menstruation} days")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
