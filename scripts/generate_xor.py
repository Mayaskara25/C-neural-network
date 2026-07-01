#!/usr/bin/env python3
import csv
import os
import random

SEED = 42
REPLICAS_PER_CORNER = 50
TRAIN_PER_CORNER = 40
TEST_PER_CORNER = 10
JITTER_SIGMA = 0.1
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "xor")

CORNERS = [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]


def jittered_rows(x1, x2, y, n):
    rows = []
    for _ in range(n):
        jx1 = x1 + random.gauss(0, JITTER_SIGMA)
        jx2 = x2 + random.gauss(0, JITTER_SIGMA)
        rows.append((round(jx1, 4), round(jx2, 4), y))
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "x2", "y"])
        writer.writerows(rows)


def main():
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_rows = []
    test_rows = []
    for x1, x2, y in CORNERS:
        replicas = jittered_rows(x1, x2, y, REPLICAS_PER_CORNER)
        random.shuffle(replicas)
        train_rows.extend(replicas[:TRAIN_PER_CORNER])
        test_rows.extend(replicas[TRAIN_PER_CORNER:TRAIN_PER_CORNER + TEST_PER_CORNER])

    random.shuffle(train_rows)
    random.shuffle(test_rows)

    write_csv(os.path.join(OUT_DIR, "train.csv"), train_rows)
    write_csv(os.path.join(OUT_DIR, "test.csv"), test_rows)

    print(f"wrote {len(train_rows)} rows to {OUT_DIR}/train.csv")
    print(f"wrote {len(test_rows)} rows to {OUT_DIR}/test.csv")


if __name__ == "__main__":
    main()
