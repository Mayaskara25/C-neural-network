#ifndef MODEL_H
#define MODEL_H
#include "matrix.h"
#include "layer.h"
#define INIT_CAPACITY 2


typedef struct{
    DenseLayer **layer;
    int num_of_layers;
    int capacity;
} NeuralNetwork;

NeuralNetwork* create_neural_network();
void add_layer(NeuralNetwork* network , DenseLayer *layer);
matrix const* forward_network(matrix input , NeuralNetwork *network);
void free_neural_network(NeuralNetwork *network);
#endif
