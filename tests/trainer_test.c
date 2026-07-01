#include "matrix.h"
#include "layer.h"
#include "activation.h"
#include "model.h"
#include "trainer.h"
#include "dataset.h"
#include "dataloader.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main() {
    srand(time(NULL));

    // build dataset — y = x² for x = 1..20
    Dataset *ds = create_dataset(1, 1);
    for (int x = 1; x <= 20; x++) {
        matrix input  = create_mat(1, 1);
        matrix output = create_mat(1, 1);
        mat_at(input,  0, 0) = (float)x;
        mat_at(output, 0, 0) = (float)(x * x);
        add_example(ds, input, output);
        free_mat(&input);
        free_mat(&output);
    }
    printf("DEBUG trainer_test: ds->num_examples=%d\n", ds->num_examples);
    printf("DEBUG trainer_test: ds->inputs[0] rows=%d cols=%d val=%f\n",
       ds->inputs[0].rows, ds->inputs[0].cols,
       mat_at(ds->inputs[0], 0, 0));

    // build network
    NeuralNetwork *nn = create_neural_network();
    add_layer(nn, create_dense_layer(1, 8, sigmoid_activation, sigmoid_derivative));
    add_layer(nn, create_dense_layer(8, 8, sigmoid_activation, sigmoid_derivative));
    add_layer(nn, create_dense_layer(8, 1, sigmoid_activation, sigmoid_derivative));

    // create trained model and normalize dataset once
    TrainedModel *model = create_trained_model(nn);
    normalize_dataset(ds, model);

    // create dataloader with batch size 4
    DataLoader *dl = create_dataloader(ds, 4);

    // train
    train(model, dl, 10000, 0.1, LOSS_MSE);

    // save
    save_model(model, "model.bin");

    // test predictions for x = 21..30
    printf("\nTesting predictions:\n");
    for (int x = 21; x <= 30; x++) {
        matrix test_input = create_mat(1, 1);
        mat_at(test_input, 0, 0) = (float)x;

        matrix predicted = predict(model, test_input);
        float expected   = (float)(x * x);
        float pred_val   = mat_at(predicted, 0, 0);

        printf("x=%d: predicted=%.2f expected=%.2f error=%.2f%%\n",
               x, pred_val, expected,
               fabsf(pred_val - expected) / expected * 100.0f);

        free_mat(&test_input);
        free_mat(&predicted);
    }
    matrix temp = create_mat(1,1);
    mat_at(temp , 0 , 0) = (float)17.543;
    matrix pout = predict(model , temp);
    printf("predicted output : %f" , mat_at(pout, 0 , 0) );

    // cleanup
    free_mat(&temp);
    free_mat(&pout);

    free_dataloader(dl);
    free_dataset(ds);
    free_trained_model(model);
    return 0;
}