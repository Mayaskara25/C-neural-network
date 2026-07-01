#include "matrix.h"
#include "layer.h"
#include "activation.h"
#include "model.h"
#include "trainer.h"
#include "dataset.h"
#include "dataloader.h"
#include "csv.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main() {
    srand(time(NULL));

    CSV *train_csv = read_csv("data/forward_kinematics/train.csv", 1);
    if (train_csv == NULL) {
        fprintf(stderr, "run scripts/generate_forward_kinematics.py first\n");
        return 1;
    }
    Dataset *train_ds = csv_to_dataset(train_csv, 2, 2);
    free_csv(train_csv);

    CSV *test_csv = read_csv("data/forward_kinematics/test.csv", 1);
    if (test_csv == NULL) {
        fprintf(stderr, "run scripts/generate_forward_kinematics.py first\n");
        free_dataset(train_ds);
        return 1;
    }
    Dataset *test_ds = csv_to_dataset(test_csv, 2, 2);
    free_csv(test_csv);

    printf("train examples: %d, test examples: %d\n", train_ds->num_examples, test_ds->num_examples);

    NeuralNetwork *nn = create_neural_network();
    add_layer(nn, create_dense_layer(2, 16, relu_activation, relu_derivative));
    add_layer(nn, create_dense_layer(16, 16, relu_activation, relu_derivative));
    add_layer(nn, create_dense_layer(16, 2, no_activation, no_activ_derivative));

    TrainedModel *model = create_trained_model(nn);
    normalize_dataset(train_ds, model);

    DataLoader *dl = create_dataloader(train_ds, 16);
    train(model, dl, 3000, 0.02);

    printf("\nTesting on held-out set:\n");
    float total_x_error = 0.0f, total_y_error = 0.0f;
    for (int i = 0; i < test_ds->num_examples; i++) {
        matrix predicted = predict(model, test_ds->inputs[i]);
        float expected_x = mat_at(test_ds->outputs[i], 0, 0);
        float expected_y = mat_at(test_ds->outputs[i], 1, 0);
        float pred_x = mat_at(predicted, 0, 0);
        float pred_y = mat_at(predicted, 1, 0);
        float x_error = fabsf(pred_x - expected_x);
        float y_error = fabsf(pred_y - expected_y);
        total_x_error += x_error;
        total_y_error += y_error;

        if (i < 10) {
            printf("a1=%.2f a2=%.2f: predicted=(%.3f,%.3f) expected=(%.3f,%.3f) abs_error=(%.3f,%.3f)\n",
                   mat_at(test_ds->inputs[i], 0, 0), mat_at(test_ds->inputs[i], 1, 0),
                   pred_x, pred_y, expected_x, expected_y, x_error, y_error);
        }
        free_mat(&predicted);
    }
    printf("\naverage test error: x=%.4f y=%.4f over %d held-out examples\n",
           total_x_error / test_ds->num_examples, total_y_error / test_ds->num_examples,
           test_ds->num_examples);

    free_dataloader(dl);
    free_dataset(train_ds);
    free_dataset(test_ds);
    free_trained_model(model);
    return 0;
}
