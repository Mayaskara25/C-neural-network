#ifndef MATRIX_H
#define MATRIX_H
#include <stdio.h>
#define mat_at(m,r,c)(m.data[(r)*(m.cols) + (c)])
#define mat_error(msg) do { \
fprintf(stderr , "%s\n" , msg); \
matrix _err = { 0 , 0 , NULL};  \
return _err; \
} while (0)

typedef struct {
    int rows;
    int cols;
    float *data;
} matrix;

void print_mat(matrix a);
void input_mat(matrix a);

matrix create_mat(int row , int col);
matrix copy_mat(matrix a);
void free_mat(matrix *a);

matrix add_mat(matrix  a, matrix b);
matrix sub_mat(matrix a , matrix b);
matrix scalarmul_mat(matrix a , float b);
matrix mul_mat(matrix a , matrix b);
matrix transpose_mat(matrix a);

void rand_init_mat(matrix a);
void zero_mat(matrix a);
void xavier_init_mat(matrix a , int input_size); //this gives rand in range of +- 1/sqrtf(input_size) ; apprantly it is used for sigmoid and tanh acctivation;
void he_init_mat(matrix a , int input_size); //this gives in range +- sqrtf(2/input_size) ; its used with ReLU;

matrix other_op_mat(matrix a , float (*fn)(float));
matrix elemul_mat(matrix a , matrix b );

float find_max( matrix a);

#endif
