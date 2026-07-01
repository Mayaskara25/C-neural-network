#!/usr/bin/env python3
import csv
import math
import os
import random

SEED = 42
TRAIN_ROWS = 500
TEST_ROWS = 100
NOISE_SIGMA = 0.01
L1 = 1.0
L2 = 0.8
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "forward_kinematics")


def position_for(a1, a2):
    x = L1 * math.cos(a1) + L2 * math.cos(a1 + a2)
    y = L1 * math.sin(a1) + L2 * math.sin(a1 + a2)
    return x + random.gauss(0, NOISE_SIGMA), y + random.gauss(0, NOISE_SIGMA)


def generate_rows(n):
    rows = []
    for _ in range(n):
        a1 = random.uniform(0, 2 * math.pi)
        a2 = random.uniform(-math.pi, math.pi)
        x, y = position_for(a1, a2)
        rows.append((round(a1, 5), round(a2, 5), round(x, 5), round(y, 5)))
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a1", "a2", "x", "y"])
        writer.writerows(rows)


def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = generate_rows(TRAIN_ROWS + TEST_ROWS)
    random.shuffle(all_rows)
    train_rows = all_rows[:TRAIN_ROWS]
    test_rows = all_rows[TRAIN_ROWS:]

    write_csv(os.path.join(OUT_DIR, "train.csv"), train_rows)
    write_csv(os.path.join(OUT_DIR, "test.csv"), test_rows)

    print(f"wrote {len(train_rows)} rows to {OUT_DIR}/train.csv")
    print(f"wrote {len(test_rows)} rows to {OUT_DIR}/test.csv")


if __name__ == "__main__":
    main()
