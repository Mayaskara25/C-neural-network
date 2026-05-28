#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "matrix.h"

void print_mat(matrix a){
    for( int i  = 0 ; i < a.rows ; i++) {
        for ( int j = 0 ; j < a.cols ; j++) {
            printf("%f\t" , mat_at(a,i,j));
        }
        printf("\n");
    }
}
matrix create_mat(int row , int col) {
    matrix m;
    m.rows = row;
    m.cols = col;
    m.data = (float*)malloc((row*col)*sizeof(float));
    if (m.data == NULL) {
        fprintf(stderr , "Error = create_mat failed as malloc failed");
        m.rows = 0 ; m.cols = 0;
        return m;
    }
    return m;
}
matrix copy_mat(matrix a) {
    matrix c = create_mat(a.rows , a.cols);
    for( int i = 0 ; i < (a.rows * a.cols ) ; i++) c.data[i] = a.data[i];
    return c;
}

void free_mat(matrix *a) {
    if(a->data != NULL) {
    free(a->data);
    a->data =NULL;
    a->rows =0;
    a->cols=0;
    }
}


void zero_mat(matrix a) {
    for( int i = 0; i < a.rows ; i++){
        for( int j = 0 ; j < a.cols ; j++) mat_at(a,i,j) = 0;
    }
}

void rand_init_mat(matrix a){
    for(int i  = 0 ; i < a.rows ; i++) {
        for( int j = 0 ; j < a.cols ; j++){
            mat_at(a,i,j) = (  ( (float)rand()/(float)RAND_MAX ) * 2.0f ) -1.0f ;
        }
    }
}

void xavier_init_mat(matrix a , int input_size) {
    float limit = 1.0f / sqrtf(input_size);
    for(int i = 0 ; i < a.rows ; i++){
        for(int j = 0; j < a.cols ; j++){
            mat_at(a,i,j) =  ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f)*limit;
        }
    }
}
void he_init_mat(matrix a , int input_size) {
    float limit = sqrtf(2.0f / input_size);
    for(int i = 0 ; i < a.rows ; i++){
        for(int j = 0; j < a.cols ; j++){
            mat_at(a,i,j) =  ((((float)rand()/(float)RAND_MAX)*2.0f)-1.0f)*limit;
        }
    }
}


matrix add_mat(matrix a , matrix b) {
    if(a.rows != b.rows || a.cols != b.cols) {
      mat_error("matrices not compatable for addition");
    }
    matrix c = create_mat(a.rows , a.cols);
    for( int i = 0 ; i < a.rows ; i++){
        for( int j = 0; j < a.cols ; j++) {
            mat_at(c,i,j) = mat_at(a,i,j) + mat_at(b,i,j);
        }
    }
    return c;
}

matrix sub_mat(matrix a , matrix b) {
    if(a.rows != b.rows || a.cols != b.cols) {
        mat_error("matrices not compatable for subtraction");
    }
    matrix c = create_mat(a.rows , a.cols);
    for( int i = 0 ; i < a.rows ; i++){
        for( int j = 0; j < a.cols ; j++) {
            mat_at(c,i,j) = mat_at(a,i,j) - mat_at(b,i,j);
        }
    }
    return c;
}

matrix scalarmul_mat(matrix a , float b){
    matrix c = create_mat(a.rows , a.cols);
    for( int i = 0 ; i < a.rows ; i++){
        for( int j = 0; j < a.cols ; j++) {
            mat_at(c,i,j) = mat_at(a,i,j)*b;
        }
    }
    return c;
}
matrix mul_mat(matrix a , matrix b){
    if( a.cols != b.rows) {
        mat_error("matrices not compatable for multiplication");
    }
        matrix c = create_mat(a.rows , b.cols);
        zero_mat(c);
        for (int i = 0 ; i < c.rows ; i++) {
            for ( int j = 0 ; j < c.cols ; j++){
                for( int k = 0 ; k < a.cols ; k++) {
                    mat_at(c,i,j) += ( mat_at(a,i,k) * mat_at(b,k,j) );
                }
            }
        }
    return c;
}

matrix transpose_mat(matrix a) {
    matrix c =  create_mat(a.cols , a.rows);
    for( int i = 0 ; i < c.rows ; i++){
        for( int j = 0; j < c.cols ; j++){
            mat_at(c,i,j) = mat_at(a,j,i);
        }
    }
    return c;
}

matrix elemul_mat( matrix a , matrix b) {
    if(a.rows != b.rows || a.cols != b.cols) {
        mat_error("matrices not compatable for element wise multiplication");
    }
    matrix c = create_mat(a.rows , a.cols);
    for( int i = 0 ; i < a.rows ; i++){
        for( int j = 0; j < a.cols ; j++) {
            mat_at(c,i,j) = mat_at(a,i,j) * mat_at(b,i,j);
        }
    }
    return c;
}
matrix other_op_mat(matrix a , float (*fn)(float)) {
    matrix c = create_mat( a.rows , a.cols);
    for ( int i = 0 ; i < c.rows ; i++ ) {
        for ( int j = 0 ; j < c.cols ; j++) {
            mat_at(c,i,j) = fn(mat_at(a,i,j));
        }
    }
    return c;
}

float find_max(matrix a){
    float max = mat_at(a, 0, 0);
    for( int i = 0 ; i < a.rows ; i++){
        for( int j = 0 ; j < a.cols ; j++){
            if(mat_at(a, i, j) > max){
                max = mat_at(a, i, j);
            }
        }
    }
    return max;;
}
