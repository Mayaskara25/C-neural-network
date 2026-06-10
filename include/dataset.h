#ifndef DATASET_H
#define DATASET_H
#include "matrix.h"

typedef struct {
  matrix *inputs;
  matrix *outputs;
  int num_examples;
  int capacity;
  int input_size;
  int output_size;
} Dataset;

Dataset* create_dataset(int input_size , int output_size);
void add_example(Dataset *ds , matrix input , matrix output);
void free_dataset(Dataset *ds);
#endif