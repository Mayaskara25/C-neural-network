#include <stdio.h>
#include <stdlib.h>
#include "trainer.h"
#include "activation.h"
#include "layer.h"
#include "matrix.h"
#include "loss.h"
#include "model.h"
#include "dataloader.h"
#include "dataset.h"



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
TrainedModel* create_trained_model(NeuralNetwork *network) {
    TrainedModel *m = (TrainedModel*)malloc(sizeof(TrainedModel));
    if( m == NULL) { fprintf(stderr , "error : trained model malloc failed") ; return NULL ; }
    m->network = network;
    m->input_max = 1.0f;
    m->output_max = 1.0f;
    return m;
}
void free_trained_model(TrainedModel *model){
    free_neural_network(model->network);
    free(model);
}
void normalize_dataset(Dataset *ds , TrainedModel *model){
    //this is to find max across all input matrices
    float in_max = 0.0f;
    for( int i = 0 ; i < ds->num_examples ; i++){
        float m = find_max(ds->inputs[i]);
        if( m > in_max)  in_max = m;
    }
    //same for output
    float out_max = 0.0f;
    for( int i = 0 ; i < ds->num_examples ; i++){
        float m = find_max(ds->outputs[i]);
        if( m > out_max)  out_max = m;
    }
    model->input_max = in_max;
    model->output_max = out_max;
    //normalizing - the issue with find_mat persists ; go to matrix.c and fix it
    for( int i = 0 ; i < ds->num_examples ; i++){
        for( int r = 0 ; r < ds->inputs[i].rows ; r++ ){
            mat_at(ds->inputs[i] , r, 0) /= in_max;
        }
        for( int r = 0 ; r < ds->outputs[i].rows ; r++){
            mat_at(ds->outputs[i] , r, 0) /= out_max;
        }
    }
}
void train(TrainedModel *model , DataLoader *dl , int epoches , float lr){
    
    // DEBUG — print first example dimensions
    //printf("DEBUG: first input  rows=%d cols=%d\n",
    //       dl->ds->inputs[0].rows, dl->ds->inputs[0].cols);
    //printf("DEBUG: first output rows=%d cols=%d\n",
    //       dl->ds->outputs[0].rows, dl->ds->outputs[0].cols);
    //printf("DEBUG: layer[0] input rows=%d cols=%d\n",
    //       model->network->layer[0]->input.rows,
    //       model->network->layer[0]->input.cols);
    // rest of train...

    printf("Starting to train");
    for( int epoch = 0 ; epoch < epoches ; epoch++){
        reset_dataloader(dl);
        while(has_next_batch(dl)){
            Batch b = next_batch(dl);
            zero_network_accumulators(model->network);
            for(int i = 0 ; i < b.size ; i++){
                matrix const* output =  forward_network(b.inputs[i] , model->network);
                matrix loss_grad = mse_derivative(*output , b.outputs[i]);
                backward_network(loss_grad , model->network);
                accumulate_network_gradients(model->network);
                free_mat(&loss_grad);
            }
            update_weights_batch(model->network , lr , b.size);
            free_batch(b);
        }
        if( epoch % 100 == 0){
            matrix const* out = forward_network(dl->ds->inputs[0] , model->network);
            float loss = mse(*out , dl->ds->outputs[0]);
            printf("epoches %d loss : %f\n" , epoch , loss);
        }
    }
    printf("training complete");

}
matrix predict(TrainedModel *model, matrix X) {
    matrix norm_X = scalarmul_mat(X, 1.0f / model->input_max);
    matrix const* output = forward_network(norm_X, model->network);
    matrix answer_temp = copy_mat(*output);
    matrix answer = scalarmul_mat(answer_temp, model->output_max);
    free_mat(&norm_X);
    free_mat(&answer_temp);
    return answer;
}
void save_model(TrainedModel *model , char *path){
    FILE *f = fopen(path , "wb");
    if( f == NULL) { fprintf(stderr , " error : could not open file to save model") ; return ;}
    int magic = 0xFAFF;
    fwrite(&model ,sizeof(int) , 1 , f);
    fwrite(&model->network->num_of_layers , sizeof(int) , 1 , f);
    fwrite(&model->input_max , sizeof(float), 1 , f);
    fwrite(&model->output_max , sizeof(float), 1 , f);
    for( int i = 0 ; i < model->network->num_of_layers ; i++){
        DenseLayer *layer = model->network->layer[i];
        int input_size = layer->weights.cols;
        int output_size = layer->weights.rows;
        ActivationID actID = get_activationId(layer->activation);

        fwrite(&input_size , sizeof(int),1,f);
        fwrite(&output_size , sizeof(int) , 1 , f);
        fwrite(&actID , sizeof(int) , 1 , f);
        fwrite(layer->weights.data , sizeof(float) , input_size * output_size , f);
        fwrite(layer->bias.data , sizeof(int), output_size , f);
    }
    fclose(f);
    printf("model saved\n");
}
NeuralNetwork* load_model_network(char* path , TrainedModel **out_model){
    FILE *f = fopen(path , "rb");
    if( f == NULL) { fprintf(stderr , " error : could not open file to read model") ; return NULL;}
    int magic;
    fread(&magic , sizeof(int) , 1 , f);
    if( magic != 0xFAFF ) { fprintf(stderr , " error : invalid model file") ; fclose(f) ; return NULL ;}
    int num_of_layers;
    float input_max , output_max;
    fread(&num_of_layers , sizeof(int) , 1 , f);
    fread(&input_max , sizeof(float) , 1 , f);
    fread(&output_max , sizeof(float) ,1 , f);
    NeuralNetwork *nn = create_neural_network();
    for( int i = 0 ; i < num_of_layers ; i++) {
        int input_size , output_size ;
        ActivationID actID ;
        fread(&input_max , sizeof(float) , 1, f);
        fread(&output_max , sizeof(float), 1, f);
        fread(&actID , sizeof(int), 1 , f);
        float (*activation)(float);
        float (*activation_derivative)(float);
        get_activation_fxn(actID , &activation , &activation_derivative);

        DenseLayer *layer =create_dense_layer(input_size , output_size , activation , activation_derivative);
        fread(&layer->weights.data , sizeof(float) , input_size*output_size ,f );
        fread(&layer->bias.data , sizeof(float) , output_size , f);
        add_layer(nn , layer);
    }
    fclose(f);

    TrainedModel* model = create_trained_model(nn);
    model->input_max = input_max;
    model->output_max = output_max;
    *out_model = model;
    printf("model loaded from %s\n", path);
    return nn;

}
/*
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
    return nn; */

