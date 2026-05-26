#ifndef LAYER_H
#define LAYER_H
#include "matrix.h"
typedef struct {
    matrix weights;
    matrix bias;
    matrix output;
    matrix input;
    matrix z; //thi is for storing the preactivation values
    matrix delta;
    matrix dW;
    matrix db;
    matrix dx;
    float (*activation)(float);
    float (*activation_derivative)(float);
}DenseLayer;

DenseLayer* create_dense_layer(int input_size , int output_size , float (*activation)(float) , float (*activation_deriative) (float));
void forward_pass(matrix input , DenseLayer *layer);
void backward_pass(matrix upstream_gradient, DenseLayer *layer);
void free_dense_layer(DenseLayer *layer);
#endif
