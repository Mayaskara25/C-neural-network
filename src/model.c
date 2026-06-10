#include "layer.h"
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
void backward_network(matrix loss_grad, NeuralNetwork *network){
    matrix current_grad = loss_grad;
    for ( int i = network->num_of_layers-1 ; i >= 0 ;i--){
        backward_pass(current_grad,network->layer[i]);
        current_grad = network->layer[i]->dx;
    }
}
void update_weights(NeuralNetwork *network, float learning_rate){
    for( int i = network->num_of_layers - 1 ; i >=0 ; i-- ){
        matrix delta_W = scalarmul_mat(network->layer[i]->dW, learning_rate);
        matrix new_weight = sub_mat(network->layer[i]->weights, delta_W);
        free_mat(&network->layer[i]->weights);
        network->layer[i]->weights = new_weight;
        free_mat(&delta_W);

        matrix delta_b = scalarmul_mat(network->layer[i]->db , learning_rate);
        matrix new_bias = sub_mat(network->layer[i]->bias, delta_b);
        free_mat(&network->layer[i]->bias);
        network->layer[i]->bias = new_bias;
        free_mat(&delta_b);
    }
}

void zero_network_accumulators(NeuralNetwork *nn) {
    for( int i = 0 ;  i < nn->num_of_layers ; i++) zero_accumulators(nn->layer[i]); 
}
void accumulate_network_gradients(NeuralNetwork *nn){
    for (int i = 0; i < nn->num_of_layers; i++)  accumulate_gradients(nn->layer[i]);      
}

void update_weights_batch(NeuralNetwork *nn, float lr, int batch_size) {
    for( int i = 0 ; i  < nn->num_of_layers ; i++) {
        DenseLayer *layer =nn->layer[i];
        for( int r = 0 ; r < layer->weights.rows ; r++){
            for( int c = 0 ; c < layer->weights.cols ; c++){
                mat_at(layer->weights , r ,c) -= lr*(mat_at(layer->accumulated_dW , r ,c)/(float)batch_size);
            }
        }
        for( int r = 0 ; r < layer->bias.rows ; r++){
            mat_at(layer->bias ,r , 0) -= lr*(mat_at(layer->accumulated_db , r , 0 ) / (float)batch_size);
        }
    }
}
void free_neural_network(NeuralNetwork *network){
    for( int i = 0 ; i < network->num_of_layers ; i++){
        free_dense_layer(network->layer[i]);
    }
    free(network->layer);
    free(network);
}
