#include <stdio.h>
#include "matrix.h"
#include "activation.h"
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));
    matrix a = create_mat(2,2);
    for( int i = 0 ; i < a.rows ; i++) {
        for ( int j= 0 ; j < a.cols ;j++) {
            printf("..");
            scanf("%f" ,&mat_at(a,i,j));
        }
    }
    print_mat(a);
    printf("\n");

    matrix b = create_mat(2,2);
    for( int i = 0 ; i < b.rows ; i++) {
        for ( int j= 0 ; j < b.cols ;j++) {
            printf("..");
            scanf("%f" ,&mat_at(b,i,j));
        }
    }
    print_mat(b);
    printf("\n");

    matrix c = other_op_mat(a , sigmoid_activation);
    print_mat(c);
    printf("\n");

    matrix d = other_op_mat(a , relu_activation);
    print_mat(d);
    printf("\n");

    matrix e = other_op_mat(a , tanh_activation);
    print_mat(e);
    printf("\n");

    matrix f = other_op_mat(a , sigmoid_derivative);
    print_mat(f);
    printf("\n");

    matrix g = other_op_mat(a , tanh_derivative);
    print_mat(g);
    printf("\n");





    /*matrix c = add_mat(a,b);
    print_mat(c);
    printf("\n");
    matrix d = sub_mat(a,b);
    print_mat(d);
    printf("\n");
    matrix e = mul_mat(a,b);
    print_mat(e);
    printf("\n");
    matrix f = scalarmul_mat(a ,2.0);
    print_mat(f);
    printf("\n");
    matrix g = elemul_mat(a,b);
    print_mat(g);
    printf("\n"); */



    free_mat(&a);
    free_mat(&b);
    free_mat(&c);
    free_mat(&d);
    free_mat(&e);
    free_mat(&f);
    free_mat(&g);

    return 0;



}
