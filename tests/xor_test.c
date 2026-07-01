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

    CSV *train_csv = read_csv("data/xor/train.csv", 1);
    if (train_csv == NULL) {
        fprintf(stderr, "run scripts/generate_xor.py first\n");
        return 1;
    }
    Dataset *train_ds = csv_to_dataset(train_csv, 2, 1);
    free_csv(train_csv);

    CSV *test_csv = read_csv("data/xor/test.csv", 1);
    if (test_csv == NULL) {
        fprintf(stderr, "run scripts/generate_xor.py first\n");
        free_dataset(train_ds);
        return 1;
    }
    Dataset *test_ds = csv_to_dataset(test_csv, 2, 1);
    free_csv(test_csv);

    printf("train examples: %d, test examples: %d\n", train_ds->num_examples, test_ds->num_examples);

    NeuralNetwork *nn = create_neural_network();
    add_layer(nn, create_dense_layer(2, 8, sigmoid_activation, sigmoid_derivative));
    add_layer(nn, create_dense_layer(8, 1, sigmoid_activation, sigmoid_derivative));

    TrainedModel *model = create_trained_model(nn);
    normalize_dataset(train_ds, model);

    DataLoader *dl = create_dataloader(train_ds, 16);
    train(model, dl, 3000, 0.4, LOSS_MSE);

    printf("\nTesting on held-out set:\n");
    int correct = 0;
    float total_error = 0.0f;
    for (int i = 0; i < test_ds->num_examples; i++) {
        matrix predicted = predict(model, test_ds->inputs[i]);
        float expected = mat_at(test_ds->outputs[i], 0, 0);
        float pred_val = mat_at(predicted, 0, 0);
        int predicted_class = pred_val >= 0.5f ? 1 : 0;
        int expected_class = expected >= 0.5f ? 1 : 0;
        if (predicted_class == expected_class) correct++;
        total_error += fabsf(pred_val - expected);

        if (i < 10) {
            printf("x1=%.2f x2=%.2f: predicted=%.3f expected=%.0f class=%d %s\n",
                   mat_at(test_ds->inputs[i], 0, 0), mat_at(test_ds->inputs[i], 1, 0),
                   pred_val, expected, predicted_class,
                   predicted_class == expected_class ? "correct" : "WRONG");
        }
        free_mat(&predicted);
    }
    float accuracy = 100.0f * correct / test_ds->num_examples;
    printf("\ntest accuracy: %.1f%% (%d/%d), average abs error: %.4f\n",
           accuracy, correct, test_ds->num_examples, total_error / test_ds->num_examples);
    printf("XOR test: %s\n", accuracy >= 90.0f ? "PASSED" : "FAILED");

    free_dataloader(dl);
    free_dataset(train_ds);
    free_dataset(test_ds);
    free_trained_model(model);
    return accuracy >= 90.0f ? 0 : 1;
}
