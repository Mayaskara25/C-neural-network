#!/usr/bin/env python3
import csv
import os
import random

SEED = 42
TRAIN_ROWS = 400
TEST_ROWS = 100
NOISE_SIGMA = 3000.0
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "house_price")


def price_for(size, rooms, age):
    return 50 * size + 10000 * rooms - 200 * age + random.gauss(0, NOISE_SIGMA)


def generate_rows(n):
    rows = []
    for _ in range(n):
        size = random.uniform(500, 3000)
        rooms = random.randint(1, 6)
        age = random.uniform(0, 50)
        price = price_for(size, rooms, age)
        rows.append((round(size, 2), rooms, round(age, 2), round(price, 2)))
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size", "rooms", "age", "price"])
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
