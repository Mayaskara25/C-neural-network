#include "matrix.h"
#include "model.h"
#include <stdio.h>
#include <stdlib.h>

NeuralNetwork* create_neural_network(){
    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->layer = NULL;
    nn->num_of_layers = 0;
    nn->capacity = INIT_CAPACITY;
    return nn;
    }
void add_layer(NeuralNetwork *network, DenseLayer *layer){
    if(network->layer == NULL){
    DenseLayer** temp1 = (DenseLayer**)malloc(INIT_CAPACITY*sizeof(DenseLayer*));
     if(temp1 != NULL) {
         network->layer = temp1;
     }
     else {
         fprintf(stderr, "ERROR : Couldnt add layer as first malloc failed ");
         return;
    }
    }
    else if(network->num_of_layers == network->capacity){
        DenseLayer** temp = (DenseLayer**)realloc(network->layer , (network->capacity+1)* sizeof(DenseLayer*));
        if(temp != NULL) {
            network->layer = temp;
            network->capacity++;

        }
        else {
            fprintf(stderr,"ERROR : couldnt add the layer as realloc failed");
            return;
        }
    }
        network->layer[network->num_of_layers] = layer;
        network->num_of_layers++;
}

matrix const* forward_network(matrix input, NeuralNetwork *network){
    matrix current_ans = input;
    for( int i = 0 ; i < network->num_of_layers ; i++){
        forward_pass(current_ans, network->layer[i]);
        current_ans = network->layer[i]->output;
    }
    return &network->layer[network->num_of_layers -1]->output;
}

void free_neural_network(NeuralNetwork *network){
    for( int i = 0 ; i < network->num_of_layers ; i++){
        free_dense_layer(network->layer[i]);
    }
    free(network->layer);
    free(network);
}
