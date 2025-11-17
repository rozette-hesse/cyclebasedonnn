import streamlit as st
import pandas as pd
from datetime import datetime
from keras.models import load_model
import numpy as np
from scripts.utils import build_dataset, evaluate_predictions

st.set_page_config(page_title="Cycle Predictor AI", layout="centered")
st.title("🩸 AI-Powered Menstrual Cycle Predictor")

uploaded_file = st.file_uploader("Upload a file with your period start and end dates (CSV or TXT with columns: start_date, end_date)", type=["txt", "csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep="\t" if uploaded_file.name.endswith(".txt") else ",")
    st.write("### Preview:")
    st.dataframe(df.head())

    if "start_date" not in df.columns or "end_date" not in df.columns:
        st.error("File must contain 'start_date' and 'end_date' columns")
    else:
        with st.spinner("Preparing dataset and predicting..."):
            try:
                # Convert to datetime and sort
                df["start_date"] = pd.to_datetime(df["start_date"])
                df["end_date"] = pd.to_datetime(df["end_date"])
                df = df.sort_values("start_date")

                # Calculate cycle and menstruation lengths
                df["cycle_length"] = df["start_date"].shift(-1) - df["start_date"]
                df["cycle_length"] = df["cycle_length"].dt.days
                df["menstruation_length"] = (df["end_date"] - df["start_date"]).dt.days + 1
                df.dropna(inplace=True)

                # Prepare input data
                X, y, _, _ = build_dataset(df[["cycle_length", "menstruation_length"]].values.tolist())

                model = load_model("models/lstm_4000.h5")
                y_pred = model.predict(X[-1:])
                pred_cycle, pred_mens = map(int, map(round, y_pred[0]))

                last_period_start = df["start_date"].iloc[-1]
                predicted_start = last_period_start + pd.Timedelta(days=pred_cycle)
                predicted_end = predicted_start + pd.Timedelta(days=pred_mens - 1)

                st.success("Prediction complete!")
                st.markdown(f"**Predicted Next Period Start:** {predicted_start.strftime('%Y-%m-%d')}")
                st.markdown(f"**Predicted End Date:** {predicted_end.strftime('%Y-%m-%d')}")
                st.markdown(f"**Predicted Duration:** {pred_mens} days")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

else:
    st.info("Please upload a valid period history file.")
