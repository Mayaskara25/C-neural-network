#include "dataset.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#define DATASET_INIT_CAPACITY 8

Dataset* create_dataset(int input_size , int output_size){
  Dataset *ds =(Dataset*)malloc(sizeof(Dataset));
  if( ds == NULL) { fprintf(stderr , "Error :dataset malloc failed\n"); return NULL;}
  ds->inputs = (matrix*)malloc(DATASET_INIT_CAPACITY * sizeof(matrix));
  ds->outputs = (matrix*)malloc(DATASET_INIT_CAPACITY * sizeof(matrix));
   //printf("DEBUG create_dataset: inputs=%p outputs=%p\n",
   //        (void*)ds->inputs, (void*)ds->outputs);
   // if (ds->inputs == NULL || ds->outputs == NULL) {
   //     fprintf(stderr, "inputs/outputs malloc failed\n"); return NULL;
   // }
  ds->input_size = input_size;
  ds->output_size = output_size;
  ds->capacity = DATASET_INIT_CAPACITY;
  ds->num_examples = 0;
  return ds; 
}
void add_example(Dataset *ds , matrix input , matrix output) {
  //printf("DEBUG add_example called: num=%d cap=%d inputs_ptr=%p\n",
  //         ds->num_examples, ds->capacity, (void*)ds->inputs);
  if(ds->num_examples == ds->capacity) {
    ds->capacity *= 2;
    ds->inputs = (matrix*)realloc(ds->inputs ,ds->capacity*sizeof(matrix));
    ds->outputs = (matrix*)realloc(ds->outputs ,ds->capacity*sizeof(matrix));
    if(!ds->inputs || !ds->outputs) { fprintf(stderr , "error : dataset realloc failed") ; return;}
    
    //printf("DEBUG add_example: stored input rows=%d cols=%d\n",
    //       ds->inputs[ds->num_examples].rows,
    //       ds->inputs[ds->num_examples].cols);

    
  }
  ds->inputs[ds->num_examples] = copy_mat(input);
  ds->outputs[ds->num_examples] = copy_mat(output);
  ds->num_examples++ ;
}
void free_dataset(Dataset *ds){
  for( int  i  = 0 ; i < ds->num_examples ; i++ ){
    free_mat(&ds->inputs[i]);
    free_mat(&ds->outputs[i]);
  }
  free(ds->inputs);
  free(ds->outputs);
  free(ds);
}