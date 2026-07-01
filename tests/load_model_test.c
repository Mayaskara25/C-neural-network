#include "matrix.h"
#include "trainer.h"
#include "dataset.h"
#include "csv.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    TrainedModel *model = NULL;
    NeuralNetwork *nn = load_model_network("house_price_model.bin", &model);
    if (nn == NULL) {
        fprintf(stderr, "run house_price_test first to produce house_price_model.bin\n");
        return 1;
    }

    CSV *test_csv = read_csv("data/house_price/test.csv", 1);
    if (test_csv == NULL) {
        fprintf(stderr, "run scripts/generate_house_prices.py first\n");
        free_trained_model(model);
        return 1;
    }
    Dataset *test_ds = csv_to_dataset(test_csv, 3, 1);
    free_csv(test_csv);

    printf("test examples: %d\n", test_ds->num_examples);

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

    free_dataset(test_ds);
    free_trained_model(model);
    return 0;
}
