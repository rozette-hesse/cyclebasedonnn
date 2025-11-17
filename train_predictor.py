# train_predictor.py

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import metrics
from utils import load_synthetic_data, evaluate_predictions

# === LOAD DATA ===
train_x, train_y, test_x, test_y = load_synthetic_data("synthetic_data_with_dates.txt")

# === MODEL SETUP ===
n_steps = train_x.shape[1]
n_features = train_x.shape[2]
model = Sequential()
model.add(LSTM(100, activation="relu", return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation="relu"))
model.add(Dense(n_features))
model.compile(optimizer="adam", loss="mse", metrics=[metrics.mae, metrics.mape])

# === TRAIN ===
model.fit(train_x, train_y, epochs=200, verbose=1)
model.save("models/lstm_trained_model.h5")

# === EVALUATE ===
preds = model.predict(test_x)
rounded_preds = [[int(round(x[0])), int(round(x[1]))] for x in preds]
accuracy = evaluate_predictions(test_y, rounded_preds)

print("Cycle length accuracy:", round(accuracy[0], 3))
print("Menstruation length accuracy:", round(accuracy[1], 3))
