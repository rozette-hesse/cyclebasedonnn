import streamlit as st
from keras.models import load_model
from keras.losses import MeanSquaredError
from datetime import datetime, timedelta
import numpy as np

# Load trained model
model = load_model("best_model.h5", compile=False)
model.compile(optimizer="adam", loss=MeanSquaredError())

st.title("🌸 Menstrual Cycle Predictor")
st.write("Enter **at least 4** period start and end dates.")

# Date input section
period_data = []
for i in range(1, 5):
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input(f"🩸 Period {i} Start", key=f"start_{i}")
    with col2:
        end = st.date_input(f"🩸 Period {i} End", key=f"end_{i}")
    if start and end and start <= end:
        period_data.append((start, end))

more_periods = st.checkbox("➕ Add more period data")
i = 5
while more_periods:
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input(f"Period {i} Start", key=f"start_{i}")
    with col2:
        end = st.date_input(f"Period {i} End", key=f"end_{i}")
    if start and end and start <= end:
        period_data.append((start, end))
    i += 1
    if not st.checkbox("Add another?"):
        break

# Sort dates
period_data = sorted(period_data, key=lambda x: x[0], reverse=False)

# Prediction logic
if len(period_data) >= 4:
    user_data = []
    for i in range(len(period_data) - 1):
        start = period_data[i][0]
        end = period_data[i][1]
        next_start = period_data[i + 1][0]
        cycle_length = (next_start - start).days
        mens_length = (end - start).days + 1
        user_data.append([cycle_length, mens_length])

    user_input = np.array(user_data[-3:]).reshape((1, 3, 2))
    predicted_cycle, predicted_mens = model.predict(user_input)[0]

    # Calculate current status
    last_period_start = period_data[-1][0]
    today = datetime.now().date()
    days_since_last = (today - last_period_start).days
    upcoming_cycle_start = last_period_start + timedelta(days=round(predicted_cycle))

    if days_since_last < predicted_mens:
        phase = "Menstrual Phase"
    elif days_since_last < predicted_cycle / 2:
        phase = "Follicular Phase"
    elif days_since_last < predicted_cycle * 0.75:
        phase = "Ovulation Phase"
    elif days_since_last < predicted_cycle:
        phase = "Luteal Phase"
    else:
        phase = "Likely New Cycle"

    # Show output
    st.subheader("🔍 Prediction Results")
    st.markdown(f"📅 **Predicted Cycle Length**: `{round(predicted_cycle)} days`")
    st.markdown(f"🩸 **Predicted Menstruation Length**: `{round(predicted_mens)} days`")
    st.markdown(f"📍 **Today is Day {days_since_last}** of your current cycle.")
    st.markdown(f"🌗 **Current Phase**: `{phase}`")
    st.markdown(f"📆 **Next Expected Period Start**: `{upcoming_cycle_start}`")
else:
    st.warning("❗ Please enter at least 4 period records to get a prediction.")
