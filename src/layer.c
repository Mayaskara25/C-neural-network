#include <stdio.h>
#include <stdlib.h>
#include "activation.h"
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
    d->dW.data = NULL ; d->dW.rows = 0 ; d->dW.cols = 0 ;
    d->db.data = NULL ; d->db.rows = 0 ; d->db.cols = 0;
    d->dx.data = NULL ; d->dx.rows = 0 ; d->dx.cols = 0;
    d->accumulated_db.data =NULL ;d->accumulated_db.rows = 0 ; d->accumulated_db.cols = 0;
    d->accumulated_dW.data = NULL ;d->accumulated_dW.rows = 0 ; d->accumulated_dW.cols = 0;
    d->delta.data = NULL ; d->delta.rows = 0 ; d->delta.cols = 0;
    d->activation = activation;
    d->activation_derivative = activation_derivative;
    if(activation == relu_activation){
        he_init_mat(d->weights, d->input.rows);
    } else {
      xavier_init_mat(d->weights , d->input.rows);
    }
    zero_mat(d->bias);
    return d;
    //ik there is an issue in create_dense_layer() - the way matrices are initialised is broken --BUT IT WORKS SO DONT TOUCH IT;
    //[problem - some matices are init with create_mat but other are just initialised to zero dont change it coz ill have to make changes further 
    //in i dont know how many other functions....its too late now]
}
 void forward_pass( matrix input , DenseLayer *layer){
    // printf("DEBUG forward_pass: input rows=%d cols=%d | layer->input rows=%d cols=%d\n",
    //       input.rows, input.cols, layer->input.rows, layer->input.cols);
    // if(layer->input.rows != input.rows || layer->input.cols != input.cols){
    //     fprintf(stderr , "Input matrix incompatable with layer's input");
    //     return;
    // }
     free_mat(&layer->input);
     layer->input = copy_mat(input);
     matrix temp = mul_mat(layer->weights , layer->input);
     free_mat(&layer->z);
     layer->z = add_mat(temp, layer->bias );
     free_mat(&temp);
     free_mat(&layer->output);
     layer->output = other_op_mat(layer->z , layer->activation);

 }
 void backward_pass(matrix upstream_gradient , DenseLayer *layer){
     //for delta
  matrix act_deriv = other_op_mat(layer->z,layer->activation_derivative);
  free_mat(&layer->delta);
  layer->delta = elemul_mat(upstream_gradient, act_deriv);
  free_mat(&act_deriv);
  //for dW
  matrix x_T = transpose_mat(layer->input);
  free_mat(&layer->dW);
  layer->dW = mul_mat(layer->delta, x_T);
  free_mat(&x_T);
  //for db
  free_mat(&layer->db);
  layer->db = copy_mat(layer->delta);
  //for dx
  matrix weight_T = transpose_mat(layer->weights);
  free_mat(&layer->dx);
  layer->dx = mul_mat(weight_T, layer->delta);
  free_mat(&weight_T);

 }
 void zero_accumulators(DenseLayer *layer) {
    if(layer->accumulated_dW.data == NULL){
        layer->accumulated_db = create_mat(layer->bias.rows , 1);
        layer->accumulated_dW = create_mat(layer->weights.rows , layer->weights.cols);
    }
    zero_mat(layer->accumulated_db);
    zero_mat(layer->accumulated_dW);
 }

 void accumulate_gradients(DenseLayer *layer){
    for ( int i = 0 ; i < layer->dW.rows ; i++){
        for( int j = 0 ; j < layer->dW.cols ; j++){
            mat_at(layer->accumulated_dW , i , j) += mat_at(layer->dW , i , j );
        }
    }
    for( int i = 0 ;  i  < layer->db.rows ; i++){
        mat_at(layer->accumulated_db , i, 0) += mat_at(layer->db , i , 0);
    }
 }
 void free_dense_layer(DenseLayer *layer) {
    free_mat(&layer->weights);
    free_mat(&layer->bias);
    free_mat(&layer->input);
    free_mat(&layer->output);
    free_mat(&layer->z);
    free_mat(&layer->dW);
    free_mat(&layer->db);
    free_mat(&layer->dx);
    free_mat(&layer->delta);
    free_mat(&layer->accumulated_db);
    free_mat(&layer->accumulated_dW);
    free(layer);
 }
