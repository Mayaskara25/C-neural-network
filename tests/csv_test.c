#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "csv.h"
#include "matrix.h"
#include "dataset.h"

static int tests_passed = 0;
static int tests_failed = 0;

static void check(int cond, const char *name) {
    if (cond) {
        printf("  PASS: %s\n", name);
        tests_passed++;
    } else {
        printf("  FAIL: %s\n", name);
        tests_failed++;
    }
}

static void test_round_trip(void) {
    printf("\n--- test: write-then-read round trip ---\n");

    CSV *csv = create_csv(3, 4);
    csv->has_header = 1;
    csv->header = (char**)malloc(4 * sizeof(char*));
    csv->header[0] = strdup("area");
    csv->header[1] = strdup("beds");
    csv->header[2] = strdup("baths");
    csv->header[3] = strdup("price");

    csv_at(*csv, 0, 0) = 1500; csv_at(*csv, 0, 1) = 3; csv_at(*csv, 0, 2) = 2;   csv_at(*csv, 0, 3) = 300000;
    csv_at(*csv, 1, 0) = 2000; csv_at(*csv, 1, 1) = 4; csv_at(*csv, 1, 2) = 2.5f; csv_at(*csv, 1, 3) = 450000;
    csv_at(*csv, 2, 0) = 800;  csv_at(*csv, 2, 1) = 1; csv_at(*csv, 2, 2) = 1;   csv_at(*csv, 2, 3) = 150000;

    write_csv(csv, "test_round_trip.csv");

    CSV *loaded = read_csv("test_round_trip.csv", 1);
    check(loaded != NULL, "file loaded");
    check(loaded->rows == 3, "rows == 3");
    check(loaded->cols == 4, "cols == 4");
    check(loaded->has_header == 1, "has_header == 1");
    check(strcmp(loaded->header[0], "area") == 0, "header[0] == area");
    check(strcmp(loaded->header[3], "price") == 0, "header[3] == price");
    check(csv_at(*loaded, 0, 0) == 1500, "data[0][0] == 1500");
    check(csv_at(*loaded, 1, 2) == 2.5f, "data[1][2] == 2.5");
    check(csv_at(*loaded, 2, 3) == 150000, "data[2][3] == 150000");

    printf("  loaded csv:\n");
    print_csv(loaded);

    free_csv(csv);
    free_csv(loaded);
    remove("test_round_trip.csv");
}

static void test_no_header(void) {
    printf("\n--- test: read without header ---\n");

    FILE *f = fopen("test_no_header.csv", "w");
    fprintf(f, "1,2,3\n4,5,6\n7,8,9\n");
    fclose(f);

    CSV *csv = read_csv("test_no_header.csv", 0);
    check(csv != NULL, "file loaded");
    check(csv->rows == 3, "rows == 3");
    check(csv->cols == 3, "cols == 3");
    check(csv->has_header == 0, "has_header == 0");
    check(csv->header == NULL, "header == NULL");
    check(csv_at(*csv, 0, 0) == 1, "data[0][0] == 1");
    check(csv_at(*csv, 2, 2) == 9, "data[2][2] == 9");

    free_csv(csv);
    remove("test_no_header.csv");
}

static void test_csv_to_dataset(void) {
    printf("\n--- test: csv_to_dataset ---\n");

    CSV *csv = create_csv(10, 5);
    for (int r = 0; r < 10; r++)
        for (int c = 0; c < 5; c++)
            csv_at(*csv, r, c) = (float)(r * 10 + c);

    Dataset *ds = csv_to_dataset(csv, 3, 2);
    check(ds != NULL, "dataset created");
    check(ds->num_examples == 10, "num_examples == 10");
    check(ds->input_size == 3, "input_size == 3");
    check(ds->output_size == 2, "output_size == 2");

    check(mat_at(ds->inputs[0], 0, 0) == 0, "inputs[0][0] == col0 val");
    check(mat_at(ds->inputs[0], 1, 0) == 1, "inputs[0][1] == col1 val");
    check(mat_at(ds->inputs[0], 2, 0) == 2, "inputs[0][2] == col2 val");
    check(mat_at(ds->outputs[0], 0, 0) == 3, "outputs[0][0] == col3 val");
    check(mat_at(ds->outputs[0], 1, 0) == 4, "outputs[0][1] == col4 val");

    check(mat_at(ds->inputs[9], 0, 0) == 90, "inputs[9][0] == row9 col0");
    check(mat_at(ds->outputs[9], 1, 0) == 94, "outputs[9][1] == row9 col4");

    free_dataset(ds);
    free_csv(csv);
}

static void test_skipped_columns(void) {
    printf("\n--- test: csv_to_dataset with skipped middle columns ---\n");

    CSV *csv = create_csv(3, 6);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 6; c++)
            csv_at(*csv, r, c) = (float)(r * 10 + c);

    Dataset *ds = csv_to_dataset(csv, 2, 1);
    check(ds != NULL, "dataset created");
    check(ds->input_size == 2, "input_size == 2");
    check(ds->output_size == 1, "output_size == 1");

    check(mat_at(ds->inputs[0], 0, 0) == 0, "input col 0");
    check(mat_at(ds->inputs[0], 1, 0) == 1, "input col 1");
    check(mat_at(ds->outputs[0], 0, 0) == 5, "output from last col (5)");
    check(mat_at(ds->inputs[2], 0, 0) == 20, "row2 input col 0");
    check(mat_at(ds->outputs[2], 0, 0) == 25, "row2 output from last col");

    free_dataset(ds);
    free_csv(csv);
}

static void test_errors(void) {
    printf("\n--- test: error cases ---\n");

    CSV *bad = read_csv("nonexistent_file.csv", 1);
    check(bad == NULL, "nonexistent file returns NULL");

    CSV *csv = create_csv(5, 4);
    Dataset *ds = csv_to_dataset(csv, 10, 10);
    check(ds == NULL, "input_cols + output_cols > cols returns NULL");
    free_csv(csv);
}

int main(void) {
    test_round_trip();
    test_no_header();
    test_csv_to_dataset();
    test_skipped_columns();
    test_errors();

    printf("\n=== %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
