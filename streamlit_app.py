# === streamlit_app.py ===
import streamlit as st
import numpy as np
from keras.models import load_model
from datetime import datetime, timedelta
from utils import build_dataset, evaluate_predictions

# Load pre-trained model
model = load_model("models/lstm_trained_model.h5")

# Streamlit UI
st.title("Cycle Predictor App")
st.markdown("Upload your period log file with start and end dates (CSV: start_date,end_date)")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        # Read uploaded data
        data = uploaded_file.read().decode("utf-8").strip().split("\n")
        periods = [tuple(line.strip().split(",")) for line in data if line.strip()]

        if len(periods) < 4:
            st.warning("Please upload at least 4 periods for accurate prediction.")
        else:
            # Preprocess
            x, y, last_known_period = build_dataset(periods)
            y_pred = model.predict(x[-1:])[0]
            next_cycle = int(round(y_pred[0]))
            next_menstruation = int(round(y_pred[1]))

            # Calculate prediction
            last_start_date = datetime.strptime(last_known_period, "%Y-%m-%d")
            predicted_start = last_start_date + timedelta(days=next_cycle)
            predicted_end = predicted_start + timedelta(days=next_menstruation - 1)

            st.success(f"Predicted Next Period Start: {predicted_start.date()}")
            st.info(f"Predicted End: {predicted_end.date()}")

            # Accuracy report (on full dataset)
            all_preds = model.predict(x)
            predicted = [[int(round(p[0])), int(round(p[1]))] for p in all_preds]
            cycle_acc, length_acc = evaluate_predictions(y, predicted)

            st.write(f"**Accuracy on Past Data**")
            st.write(f"Cycle Length Accuracy: {cycle_acc:.2f}%")
            st.write(f"Menstruation Length Accuracy: {length_acc:.2f}%")

    except Exception as e:
        st.error(f"There was an error processing the file: {e}")
