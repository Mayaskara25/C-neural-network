#include "trainer.h"
#include "activation.h"
#include "layer.h"
#include "matrix.h"
#include "loss.h"
#include "model.h"
#include <stdio.h>


static float input_max = 0;
static float output_max = 0;

ActivationID get_activationId(float (*activation)(float)) {

    if(activation == sigmoid_activation) return ACTIVATION_SIGMOID;
    if(activation == relu_activation) return ACTIVATION_RELU;
    if( activation == tanh_activation) return ACTIVATION_TANH;
    else { return ACTIVATION_LINEAR; }
}
void get_activation_fxn(ActivationID id, float (**activation)(float), float (**activation_derivative)(float)) {
    switch(id){
        case ACTIVATION_SIGMOID:
            *activation = sigmoid_activation;
            *activation_derivative = sigmoid_derivative;
            break;
        case ACTIVATION_RELU:
            *activation = relu_activation;
            *activation_derivative = relu_derivative;
            break;
        case ACTIVATION_TANH:
            *activation = tanh_activation;
            *activation_derivative = tanh_derivative;
            break;
        case ACTIVATION_LINEAR:
            *activation = no_activation;
            *activation_derivative = no_activ_derivative;
            break;
    }
}

void train(NeuralNetwork *network, matrix X, matrix Y, int epoches, float learning_rate) {

    input_max = find_max(X);
    output_max = find_max(Y);
    matrix normal_X = scalarmul_mat(X, 1.0f/input_max);
    matrix normal_Y = scalarmul_mat(Y, 1.0f/output_max);
    printf("starting to train.....\n");
    for ( int i = 0 ; i < epoches ; i++) {
        matrix const* output = forward_network(normal_X, network);
        float loss = mse(*output,normal_Y);
        matrix loss_grad = mse_derivative(*output, normal_Y);
        backward_network(loss_grad, network);
        update_weights(network,learning_rate);
        free_mat(&loss_grad);
    }
    printf("training complete\n");
    free_mat(&normal_X);
    free_mat(&normal_Y);
}

matrix predict(matrix X, NeuralNetwork *network){
    matrix normal_X = scalarmul_mat(X, 1.0f/input_max);
    matrix const* output = forward_network(normal_X, network);
    matrix answer_temp = copy_mat(*output);
    matrix answer = scalarmul_mat(answer_temp, output_max);
    free_mat(&normal_X);
    free_mat(&answer_temp);
    return answer;
}

void save_model(NeuralNetwork *network , char *path){
    FILE *f = fopen(path, "wb");
    if(f == NULL){
        fprintf(stderr, "Error saving model : couldnt open file");
    }
        int magic = 0xFAFF;
        fwrite(&magic, sizeof(int), 1 , f);


        fwrite(&network->num_of_layers , sizeof(float) , 1 , f);
        fwrite(&input_max, sizeof(float), 1, f);
        fwrite(&output_max, sizeof(float), 1 , f);
        for(int i = 0 ; i < network->num_of_layers ; i++){
            DenseLayer *layer = network->layer[i];
            int input_size = layer->weights.cols;
            int output_size = layer->weights.rows;
            ActivationID actID = get_activationId(layer->activation);

            fwrite(&input_size , sizeof(int) , 1 , f);
            fwrite(&output_size , sizeof(int) , 1 , f);
            fwrite(&actID, sizeof(int), 1 , f);

            int num_weights = input_size * output_size;
            int num_bias = output_size;
            fwrite(&layer->weights.data , sizeof(float) , num_weights , f);
            fwrite(&layer->bias, sizeof(float), num_bias, f);
        }
        fclose(f);
        printf("model saved to %s : " , path);
}
NeuralNetwork* load_model(char* path){
    FILE *f = fopen(path,"rb");
    if(f == NULL) {
        fprintf(stderr , "Error loading model ; failed to open saved file\n");
    }
    int magic;
    fread(&magic , sizeof(int) , 1 , f);
    if(magic != 0xFAFF) {
        fprintf(stderr,"Error : invalid model file\n");
        fclose(f);
        return NULL;
    }

    int num_of_layers;
    fread(&num_of_layers,sizeof(int), 1 , f);
    fread(&input_max , sizeof(int) , 1 , f);
    fread(&output_max , sizeof(int) , 1 , f);

    NeuralNetwork* nn = create_neural_network();

    for( int i = 0 ; i  < num_of_layers ; i++){
        int input_size , output_size;
        ActivationID actID;

        fread(&input_size , sizeof(int) , 1 , f);
        fread(&output_size , sizeof(int) , 1 , f);
        fread(&actID, sizeof(int), 1 , f);

        float (*activation)(float);
        float (*activation_derivative)(float);
        get_activation_fxn(actID, &activation, &activation_derivative);

        DenseLayer  *layer = create_dense_layer(input_size,  output_size, activation,activation_derivative );
        int num_weights = input_size*output_size;
        int num_bias = output_size;
        fread(&layer->weights.data , sizeof(float) , num_weights , f);
        fread(&layer->bias.data , sizeof(float), num_bias , f);
        add_layer(nn, layer);
    }
    fclose(f);
    printf("model loaded from %s\n" , path);
    return nn;
}
