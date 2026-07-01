CC = gcc
CFLAGS = -Wall -O2 -Iinclude
LDFLAGS = -lm

SRCS = $(wildcard src/*.c)

trainer_test: tests/trainer_test.c $(SRCS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

backprop_test: tests/backprop_test.c $(SRCS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

layer_test: tests/layer_test.c $(SRCS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

model_test: tests/model_test.c $(SRCS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

matrix_test: tests/matrix_test.c $(SRCS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

csv_test: tests/csv_test.c $(SRCS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

house_price_test: tests/house_price_test.c $(SRCS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

xor_test: tests/xor_test.c $(SRCS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

load_model_test: tests/load_model_test.c $(SRCS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

forward_kinematics_test: tests/forward_kinematics_test.c $(SRCS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

all: trainer_test backprop_test layer_test model_test matrix_test csv_test house_price_test xor_test load_model_test forward_kinematics_test

clean:
	rm -f trainer_test backprop_test layer_test model_test matrix_test csv_test house_price_test xor_test load_model_test forward_kinematics_test

.PHONY: all clean
