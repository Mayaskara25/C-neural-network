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

    CSV *train_csv = read_csv("data/house_price/train.csv", 1);
    if (train_csv == NULL) {
        fprintf(stderr, "run scripts/generate_house_prices.py first\n");
        return 1;
    }
    Dataset *train_ds = csv_to_dataset(train_csv, 3, 1);
    free_csv(train_csv);

    CSV *test_csv = read_csv("data/house_price/test.csv", 1);
    if (test_csv == NULL) {
        fprintf(stderr, "run scripts/generate_house_prices.py first\n");
        free_dataset(train_ds);
        return 1;
    }
    Dataset *test_ds = csv_to_dataset(test_csv, 3, 1);
    free_csv(test_csv);

    printf("train examples: %d, test examples: %d\n", train_ds->num_examples, test_ds->num_examples);

    NeuralNetwork *nn = create_neural_network();
    add_layer(nn, create_dense_layer(3, 8, relu_activation, relu_derivative));
    add_layer(nn, create_dense_layer(8, 8, relu_activation, relu_derivative));
    add_layer(nn, create_dense_layer(8, 1, no_activation, no_activ_derivative));

    TrainedModel *model = create_trained_model(nn);
    normalize_dataset(train_ds, model);

    DataLoader *dl = create_dataloader(train_ds, 16);
    train(model, dl, 1000, 0.005, LOSS_MSE);

    save_model(model, "house_price_model.bin");

    printf("\nTesting on held-out set:\n");
    float total_pct_error = 0.0f;
    for (int i = 0; i < test_ds->num_examples; i++) {
        matrix predicted = predict(model, test_ds->inputs[i]);
        float expected = mat_at(test_ds->outputs[i], 0, 0);
        float pred_val = mat_at(predicted, 0, 0);
        float pct_error = fabsf(pred_val - expected) / expected * 100.0f;
        total_pct_error += pct_error;

        if (i < 10) {
            printf("size=%.0f rooms=%.0f age=%.0f: predicted=%.2f expected=%.2f error=%.2f%%\n",
                   mat_at(test_ds->inputs[i], 0, 0), mat_at(test_ds->inputs[i], 1, 0),
                   mat_at(test_ds->inputs[i], 2, 0), pred_val, expected, pct_error);
        }
        free_mat(&predicted);
    }
    printf("\naverage test error: %.2f%% over %d held-out examples\n",
           total_pct_error / test_ds->num_examples, test_ds->num_examples);

    free_dataloader(dl);
    free_dataset(train_ds);
    free_dataset(test_ds);
    free_trained_model(model);
    return 0;
}
