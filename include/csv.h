#ifndef CSV_H
#define CSV_H
#include "dataset.h"
#include "matrix.h"

#define csv_at(csv, r, c) ((csv).data[(r) * (csv).cols + (c)])

typedef struct {
    float *data;
    int rows;
    int cols;
    char **header;
    int has_header;
} CSV;

CSV* create_csv(int rows, int cols);
CSV* read_csv(const char *path, int has_header);
void write_csv(const CSV *csv, const char *path);
void print_csv(const CSV *csv);
void free_csv(CSV *csv);
Dataset* csv_to_dataset(const CSV *csv, int input_cols, int output_cols);

#endif
