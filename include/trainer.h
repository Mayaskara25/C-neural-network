#ifndef TRAINER_H
#define TRAINER_H
#include "matrix.h"
#include "dataloader.h"
#include "dataset.h"
#include "model.h"

typedef enum {
    ACTIVATION_SIGMOID = 0,
    ACTIVATION_RELU = 1 ,
    ACTIVATION_TANH = 2,
    ACTIVATION_LINEAR = 3
} ActivationID;

typedef struct {
    NeuralNetwork *network;
    matrix input_max;
    matrix output_max;
} TrainedModel ;
TrainedModel* create_trained_model(NeuralNetwork *network);
void free_trained_model(TrainedModel *model);
void normalize_dataset(Dataset *ds , TrainedModel *model);
void train(TrainedModel *model , DataLoader *dl , int epoches , float lr);
matrix predict(TrainedModel *model , matrix X);
void save_model(TrainedModel *model , char *path);
NeuralNetwork* load_model_network(char* path , TrainedModel **out_model);
ActivationID get_activationId(float(*activation)(float));   
void get_activation_fxn(ActivationID id, float (**activation)(float), float (**derivative)(float));

/* this is old stuff , this is for stochastic gradient descent - when i made this there wasnt any TrainedModel , no Dataset , no Dataloaer ; check earlier commits to access how it worked.
I am removing these older functions from trainer.c as well ;
void train(NeuralNetwork *network , matrix X , matrix Y , int epoches , float learning_rate);
matrix predict (matrix X , NeuralNetwork *network );
void save_model(NeuralNetwork *network , char* path);
NeuralNetwork* load_model(char* path); */
#endif
