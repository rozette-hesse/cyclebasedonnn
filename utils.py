import numpy as np
from datetime import datetime

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

def load_synthetic_data(file):
    periods = []
    with open(file, 'r') as f:
        for line in f:
            cycle, menstruation = map(int, line.strip().split())
            periods.append([cycle, menstruation])

    x, y = [], []
    for i in range(len(periods) - 3):
        x.append([periods[i], periods[i+1], periods[i+2]])
        y.append(periods[i+3])

    x = np.array(x)
    y = np.array(y)
    return x, y, x, y  # for compatibility

def evaluate_predictions(test_y, predictions):
    correct_cycle = sum(int(y[0]) == int(p[0]) for y, p in zip(test_y, predictions))
    correct_mens = sum(int(y[1]) == int(p[1]) for y, p in zip(test_y, predictions))
    total = len(test_y)
    return correct_cycle / total, correct_mens / total
