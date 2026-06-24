CC = gcc
CFLAGS = -Wall -Iinclude
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

all: trainer_test backprop_test layer_test model_test matrix_test csv_test

clean:
	rm -f trainer_test backprop_test layer_test model_test matrix_test csv_test

.PHONY: all clean
