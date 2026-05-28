#ifndef TRAINER_H
#define TRAINER_H
#include "matrix.h"
#include "model.h"

typedef enum {
    ACTIVATION_SIGMOID = 0,
    ACTIVATION_RELU = 1 ,
    ACTIVATION_TANH = 2,
    ACTIVATION_LINEAR = 3
} ActivationID;

ActivationID get_activationId(float(*activation)(float));
void get_activation_fxn(ActivationID id , float (**activation)(float) , float (**activation_derivative)(float));
void train(NeuralNetwork *network , matrix X , matrix Y , int epoches , float learning_rate);
matrix predict (matrix X , NeuralNetwork *network );
void save_model(NeuralNetwork *network , char* path);
NeuralNetwork* load_model(char* path);
#endif
