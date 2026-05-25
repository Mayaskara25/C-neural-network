#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "layer.h"
DenseLayer* create_dense_layer(int input_size , int output_size , float (*activation)(float) , float (*activation_derivative) (float)) {
    DenseLayer *d = (DenseLayer*)malloc(sizeof(DenseLayer));
    if( d==NULL) { fprintf(stderr , "Error : layer malloc failed") ; return NULL ;}
    d->weights = create_mat(output_size , input_size);
    d->bias = create_mat(output_size , 1);
    d->input = create_mat(input_size , 1);
    d->output = create_mat(output_size , 1);
    d->z  = create_mat(output_size , 1);
    d->activation = activation;
    d->activation_derivative = activation_derivative;
    xavier_init_mat(d->weights , d->input.rows);
    zero_mat(d->bias);
    return d;
}
 void forward_pass( matrix input , DenseLayer *layer){
     if(layer->input.rows != input.rows || layer->input.cols != input.cols){
         fprintf(stderr , "Input matrix incompatable with layer's input");
         return;
     }
     free_mat(&layer->input);
     layer->input = copy_mat(input);
     matrix temp = mul_mat(layer->weights , layer->input);
     free_mat(&layer->z);
     layer->z = add_mat(temp, layer->bias );
     free_mat(&temp);
     free_mat(&layer->output);
     layer->output = other_op_mat(layer->z , layer->activation);

 }

 void free_dense_layer(DenseLayer *layer) {
    free_mat(&layer->weights);
    free_mat(&layer->bias);
    free_mat(&layer->input);
    free_mat(&layer->output);
    free_mat(&layer->z);
    free(layer);
 }
