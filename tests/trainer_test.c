#include "trainer.h"
#include "model.h"
#include "matrix.h"
#include "layer.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "activation.h"

int main() {
    srand(time(NULL));
    matrix input  = create_mat(1, 1);
    matrix output = create_mat(1, 1);

    NeuralNetwork *nn = create_neural_network();
    DenseLayer *l1 = create_dense_layer(1, 8, sigmoid_activation, sigmoid_derivative);
    DenseLayer *l2 = create_dense_layer(8, 8, sigmoid_activation, sigmoid_derivative);
    DenseLayer *l3 = create_dense_layer(8, 1, sigmoid_activation, sigmoid_derivative);
    add_layer(nn, l1);
    add_layer(nn, l2);
    add_layer(nn, l3);


    for (int epoch = 0; epoch < 1000; epoch++) {
        for (int x = 1; x <= 20; x++) {
            mat_at(input,  0, 0) = (float)x;
            mat_at(output, 0, 0) = (float)(x * x);
            train(nn, input, output, 1, 0.01);
        }
    }
   float guess ;
   printf("enter number to predict:\n");
   scanf("%f" , &guess);
   mat_at(input, 0, 0) = guess;
   matrix predict_mat = predict(input, nn);
   printf("predicted output:\n");
   print_mat(predict_mat);

free_neural_network(nn);
free_mat(&input);
free_mat(&output);
free_mat(&predict_mat);
return 0;
}
