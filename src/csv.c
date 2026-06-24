#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "csv.h"
#include "matrix.h"
#include "dataset.h"

#define CSV_INIT_CAPACITY 64
#define CSV_LINE_BUFFER 4096

static void strip_newline(char *line) {
    int len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
        line[--len] = '\0';
}

static int count_cols(const char *line) {
    int count = 1;
    for (int i = 0; line[i] != '\0'; i++)
        if (line[i] == ',') count++;
    return count;
}

static char** parse_header(const char *line, int cols) {
    char **header = (char**)malloc(cols * sizeof(char*));
    if (header == NULL) return NULL;
    char *copy = strdup(line);
    char *token = strtok(copy, ",");
    for (int i = 0; i < cols && token != NULL; i++) {
        header[i] = strdup(token);
        token = strtok(NULL, ",");
    }
    free(copy);
    return header;
}

CSV* create_csv(int rows, int cols) {
    CSV *csv = (CSV*)malloc(sizeof(CSV));
    if (csv == NULL) { fprintf(stderr, "error: csv malloc failed\n"); return NULL; }
    csv->rows = rows;
    csv->cols = cols;
    csv->data = (float*)calloc(rows * cols, sizeof(float));
    if (csv->data == NULL && rows * cols > 0) {
        fprintf(stderr, "error: csv data malloc failed\n");
        free(csv);
        return NULL;
    }
    csv->header = NULL;
    csv->has_header = 0;
    return csv;
}

CSV* read_csv(const char *path, int has_header) {
    FILE *f = fopen(path, "r");
    if (f == NULL) { fprintf(stderr, "error: could not open %s\n", path); return NULL; }

    char line[CSV_LINE_BUFFER];
    int cols = 0;
    int capacity = CSV_INIT_CAPACITY;
    int row = 0;
    char **header = NULL;

    if (fgets(line, CSV_LINE_BUFFER, f) == NULL) {
        fclose(f);
        return create_csv(0, 0);
    }
    strip_newline(line);
    cols = count_cols(line);

    if (has_header) {
        header = parse_header(line, cols);
    } else {
        float *data = (float*)malloc(capacity * cols * sizeof(float));
        char *copy = strdup(line);
        char *token = strtok(copy, ",");
        for (int c = 0; c < cols && token != NULL; c++) {
            data[row * cols + c] = strtof(token, NULL);
            token = strtok(NULL, ",");
        }
        free(copy);
        row++;

        while (fgets(line, CSV_LINE_BUFFER, f) != NULL) {
            strip_newline(line);
            if (line[0] == '\0') continue;
            if (row >= capacity) {
                capacity *= 2;
                data = (float*)realloc(data, capacity * cols * sizeof(float));
            }
            copy = strdup(line);
            token = strtok(copy, ",");
            for (int c = 0; c < cols && token != NULL; c++) {
                data[row * cols + c] = strtof(token, NULL);
                token = strtok(NULL, ",");
            }
            free(copy);
            row++;
        }
        fclose(f);

        CSV *csv = (CSV*)malloc(sizeof(CSV));
        csv->rows = row;
        csv->cols = cols;
        csv->data = data;
        csv->header = NULL;
        csv->has_header = 0;
        return csv;
    }

    float *data = (float*)malloc(capacity * cols * sizeof(float));

    while (fgets(line, CSV_LINE_BUFFER, f) != NULL) {
        strip_newline(line);
        if (line[0] == '\0') continue;
        if (row >= capacity) {
            capacity *= 2;
            data = (float*)realloc(data, capacity * cols * sizeof(float));
        }
        char *copy = strdup(line);
        char *token = strtok(copy, ",");
        for (int c = 0; c < cols && token != NULL; c++) {
            data[row * cols + c] = strtof(token, NULL);
            token = strtok(NULL, ",");
        }
        free(copy);
        row++;
    }
    fclose(f);

    CSV *csv = (CSV*)malloc(sizeof(CSV));
    csv->rows = row;
    csv->cols = cols;
    csv->data = data;
    csv->header = header;
    csv->has_header = 1;
    return csv;
}

void write_csv(const CSV *csv, const char *path) {
    FILE *f = fopen(path, "w");
    if (f == NULL) { fprintf(stderr, "error: could not open %s for writing\n", path); return; }

    if (csv->has_header && csv->header != NULL) {
        for (int c = 0; c < csv->cols; c++) {
            if (c > 0) fprintf(f, ",");
            fprintf(f, "%s", csv->header[c]);
        }
        fprintf(f, "\n");
    }

    for (int r = 0; r < csv->rows; r++) {
        for (int c = 0; c < csv->cols; c++) {
            if (c > 0) fprintf(f, ",");
            fprintf(f, "%.7g", csv_at(*csv, r, c));
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void print_csv(const CSV *csv) {
    if (csv->has_header && csv->header != NULL) {
        for (int c = 0; c < csv->cols; c++) {
            if (c > 0) printf("\t");
            printf("%s", csv->header[c]);
        }
        printf("\n");
    }
    for (int r = 0; r < csv->rows; r++) {
        for (int c = 0; c < csv->cols; c++) {
            if (c > 0) printf("\t");
            printf("%.4f", csv_at(*csv, r, c));
        }
        printf("\n");
    }
    printf("[%d rows x %d cols]\n", csv->rows, csv->cols);
}

void free_csv(CSV *csv) {
    if (csv == NULL) return;
    if (csv->header != NULL) {
        for (int c = 0; c < csv->cols; c++)
            free(csv->header[c]);
        free(csv->header);
    }
    free(csv->data);
    free(csv);
}

Dataset* csv_to_dataset(const CSV *csv, int input_cols, int output_cols) {
    if (input_cols + output_cols > csv->cols) {
        fprintf(stderr, "error: input_cols(%d) + output_cols(%d) > csv cols(%d)\n",
                input_cols, output_cols, csv->cols);
        return NULL;
    }
    if (input_cols <= 0 || output_cols <= 0) {
        fprintf(stderr, "error: input_cols and output_cols must be > 0\n");
        return NULL;
    }

    Dataset *ds = create_dataset(input_cols, output_cols);
    int out_start = csv->cols - output_cols;

    for (int r = 0; r < csv->rows; r++) {
        matrix input = create_mat(input_cols, 1);
        matrix output = create_mat(output_cols, 1);

        for (int c = 0; c < input_cols; c++)
            mat_at(input, c, 0) = csv_at(*csv, r, c);
        for (int c = 0; c < output_cols; c++)
            mat_at(output, c, 0) = csv_at(*csv, r, out_start + c);

        add_example(ds, input, output);
        free_mat(&input);
        free_mat(&output);
    }
    return ds;
}
