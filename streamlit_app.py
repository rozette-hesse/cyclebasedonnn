# streamlit_app.py

import streamlit as st
import numpy as np
from keras.models import load_model
from datetime import datetime, timedelta
from utils import calculate_lengths, create_dataset

st.title("Menstrual Cycle Predictor")

# === USER INPUT ===
st.write("Enter at least **4** period ranges (start and end):")

dates = []
for i in range(4):
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input(f"Start Date {i+1}", key=f"start_{i}")
    with col2:
        end = st.date_input(f"End Date {i+1}", key=f"end_{i}")
    if start and end and start <= end:
        dates.append((str(start), str(end)))

if len(dates) < 4:
    st.warning("Please enter at least 4 valid period entries to make a prediction.")
    st.stop()

# === PREPROCESS ===
cycle_lengths, m_lengths = calculate_lengths(dates)
data = np.array([[c, m] for c, m in zip(cycle_lengths, m_lengths)])
X, _ = create_dataset(data, 3)
X = np.expand_dims(data[-3:], axis=0)

# === LOAD MODEL AND PREDICT ===
model = load_model("models/lstm_trained_model.h5")
pred = model.predict(X, verbose=0)[0]
predicted_cycle = int(round(pred[0]))
predicted_mens = int(round(pred[1]))

last_period_start = datetime.strptime(dates[-1][0], "%Y-%m-%d")
predicted_start = last_period_start + timedelta(days=predicted_cycle)
predicted_end = predicted_start + timedelta(days=predicted_mens - 1)

# === DISPLAY RESULTS ===
st.subheader("Prediction")
st.write(f"**Next Period Start:** {predicted_start.strftime('%Y-%m-%d')}")
st.write(f"**Next Period End:** {predicted_end.strftime('%Y-%m-%d')}")
st.write(f"**Predicted Cycle Length:** {predicted_cycle} days")
st.write(f"**Predicted Menstruation Length:** {predicted_mens} days")
