# Standalone LSTM Menstrual Cycle Predictor with User-Input File (like synthetic_data.txt)

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import metrics
from datetime import datetime

# === USER FILE INPUT SECTION ===
# File format: each line has "start_date,end_date" in YYYY-MM-DD format

def load_period_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    periods = [tuple(line.strip().split(",")) for line in lines if line.strip()]
    return periods

user_periods = load_period_file("user_periods.txt")  # Update this with your actual file path

# === PREPROCESSING ===
def calculate_lengths(periods):
    cycle_lengths = []
    m_lengths = []
    for i in range(1, len(periods)):
        prev_start = datetime.strptime(periods[i-1][0], "%Y-%m-%d")
        curr_start = datetime.strptime(periods[i][0], "%Y-%m-%d")
        curr_end = datetime.strptime(periods[i][1], "%Y-%m-%d")

        cycle_lengths.append((curr_start - prev_start).days)
        m_lengths.append((curr_end - curr_start).days + 1)
    return cycle_lengths, m_lengths

cycle_lengths, m_lengths = calculate_lengths(user_periods)
data = np.array([[c, m] for c, m in zip(cycle_lengths, m_lengths)])

# === DATASET CREATION ===
def create_dataset(sequence, steps=3):
    x, y = [], []
    for i in range(len(sequence) - steps):
        x.append(sequence[i:i+steps])
        y.append(sequence[i+steps])
    return np.array(x), np.array(y)

train_x, train_y = create_dataset(data, 3)

# === MODEL SETUP ===
n_steps = train_x.shape[1]
n_features = train_x.shape[2]
model = Sequential()
model.add(LSTM(100, activation="relu", return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation="relu"))
model.add(Dense(n_features))
model.compile(optimizer="adam", loss="mse", metrics=[metrics.mae, metrics.mape])

# === TRAIN MODEL ===
model.fit(train_x, train_y, epochs=100, verbose=2)

# === PREDICT NEXT CYCLE ===
last_seq = np.expand_dims(data[-3:], axis=0)
y_pred = model.predict(last_seq, verbose=0)[0]
next_cycle = int(round(y_pred[0]))
next_menstruation = int(round(y_pred[1]))

last_period_start = datetime.strptime(user_periods[-1][0], "%Y-%m-%d")
predicted_start = last_period_start + np.timedelta64(next_cycle, 'D')
predicted_end = predicted_start + np.timedelta64(next_menstruation - 1, 'D')

print("Predicted Next Period Start:", predicted_start.astype(datetime).strftime("%Y-%m-%d"))
print("Predicted Next Period End:", predicted_end.astype(datetime).strftime("%Y-%m-%d"))
print("Cycle Length:", next_cycle, "days")
print("Menstruation Length:", next_menstruation, "days")
