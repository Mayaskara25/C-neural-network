#ifndef DATALOADER_H
#define DATALOADER_H
#include "dataset.h"
#include "matrix.h"

typedef struct {
  matrix *inputs;
  matrix *outputs;
  int size;
}Batch;

typedef struct {
  Dataset *ds;
  int batch_size;
  int current_pos;
  int *indices;
}DataLoader;

DataLoader* create_dataloader(Dataset *ds , int batch_size );
void reset_dataloader(DataLoader *dl);
int has_next_batch(DataLoader *dl);
Batch next_batch(DataLoader *dl);
void free_batch(Batch b);
void free_dataloader(DataLoader *dl);
#endif