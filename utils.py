import numpy as np
from datetime import datetime, timedelta

# === FOR USER DATE INPUT PROCESSING ===
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

def create_dataset(sequence, steps=3):
    x, y = [], []
    for i in range(len(sequence) - steps):
        x.append(sequence[i:i+steps])
        y.append(sequence[i+steps])
    return np.array(x), np.array(y)

# === SYNTHETIC FILE LOADER ===
def load_synthetic_data(file):
    periods = []
    with open(file, 'r') as f:
        for line in f:
            periods.append([int(x) for x in line.strip().split()])
    
    x, y = [], []
    for i in range(len(periods) - 3):
        x.append([periods[i], periods[i+1], periods[i+2]])
        y.append(periods[i+3])

    x = np.array(x)
    y = np.array(y)
    return x, y, x, y

# === EVALUATION ===
def evaluate_predictions(test_y, predictions):
    correct_cycle = sum(int(y[0]) == int(p[0]) for y, p in zip(test_y, predictions))
    correct_mens = sum(int(y[1]) == int(p[1]) for y, p in zip(test_y, predictions))
    total = len(test_y)
    return correct_cycle / total, correct_mens / total

# === OPTIONAL: ORIGINAL FUNCTIONS ===
def read_period_file(file):
    """For .txt files from Period Tracker apps."""
    period_cal = []
    periods = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            newline = line.split("\t")
            if line.strip().split("\t")[1] == "Period Starts":
                period_cal.append([])
                period_cal[-1].append(datetime.strptime(newline[0], "%d %b, %Y"))
            elif line.strip().split("\t")[1] == "Period Ends":
                period_cal[-1].append(datetime.strptime(newline[0], "%d %b, %Y"))

    for period in period_cal[1:]:
        num = period_cal.index(period)
        if num > 0:
            lengths = []
            lengths.append(period_cal[num][0])
            lengths.append((period_cal[num][0] - period_cal[num - 1][0]).days)
            lengths.append((period_cal[num][1] - period_cal[num][0]).days + 1)
            periods.append(lengths)

    return periods

def print_predictions(last_known_period, predictions):
    next_periods = [[
        last_known_period + timedelta(days = predictions[0][0]),
        last_known_period + timedelta(days = predictions[0][0] + predictions[0][1]),
        predictions[0][1]
    ]]
    for period in predictions[1:]:
        last_period = next_periods[-1]
        next_periods.append([
            last_period[0] + timedelta(days = period[0]),
            last_period[0] + timedelta(days = period[0] + period[1]),
            period[1]
        ])
    
    for num, period in enumerate(next_periods):
        print(str(num) + ". From " + period[0].strftime('%d.%m.%Y') + \
              " to " + period[1].strftime('%d.%m.%Y') + ", length: " + str(period[2]))
    
    return next_periods
