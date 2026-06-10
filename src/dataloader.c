#include "dataloader.h"
#include <stdlib.h>
#include <stdio.h>

static void shuffle(DataLoader *dl){
  for( int i  = dl->ds->num_examples - 1 ; i > 0  ; i--) {
    int j = rand() % (i+1) ;
    int temp = dl->indices[i];
    dl->indices[i] = dl->indices[j];
    dl->indices[j] = temp;
  }
}
// i dont know just how many create__ functions are there in this code...........
DataLoader* create_dataloader(Dataset *ds , int batch_size) {
  DataLoader *dl = (DataLoader*)malloc(sizeof(DataLoader));
  if(dl == NULL ){ fprintf(stderr , "Error :  dataloader malloc failed") ; return NULL ; }
  dl->ds = ds;
  dl->batch_size = batch_size;
  dl->current_pos = 0;
  dl->indices = (int*)malloc(ds->num_examples * sizeof(int));  
  if(dl->indices ==NULL) {fprintf(stderr , "Error :  indices malloc failed") ; return NULL ;}
  for ( int i  = 0 ;  i  < dl->ds->num_examples ; i++) dl->indices[i] = i;
  shuffle(dl);
  return dl;
}

void reset_dataloader(DataLoader *dl){ dl->current_pos = 0 ; shuffle(dl); }

int has_next_batch(DataLoader *dl) { return dl->current_pos < dl->ds->num_examples ;}

Batch next_batch(DataLoader *dl){
  int remaining = dl->ds->num_examples - dl->current_pos ;
  int size = remaining < dl->batch_size ? remaining : dl->batch_size ;
  Batch b;
  b.size = size ;
  b.inputs = (matrix*)malloc(size * sizeof(matrix) );
  b.outputs = (matrix*)malloc(size * sizeof(matrix) );
  // i have to make a malloc error failed fxn
  for (int i = 0; i < size; i++) {
        int idx      = dl->indices[dl->current_pos + i];
        b.inputs[i]  = dl->ds->inputs[idx];   // reference, not copy
        b.outputs[i] = dl->ds->outputs[idx];  // reference, not copy
        //i may be causing issues by making it a reference and not copy --- plz read this in future so that you can point out where the code is failing.
  }
  dl->current_pos += size;
  return b;
}
void free_batch(Batch b){
  // only free the arrays, NOT the matrices inside
    // those are owned by Dataset
    free(b.inputs);
    free(b.outputs);
}
void free_dataloader(DataLoader *dl){
  free(dl->indices);
  free(dl);
}
