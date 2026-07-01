#include <stdio.h>
#include "matrix.h"
#include "activation.h"
#include "layer.h"
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));
    printf("input matrix:\n");
    matrix a = create_mat(3,1);
    input_mat(a);
    print_mat(a);
    printf("\n");
DenseLayer *layer = create_dense_layer(3, 5, sigmoid_activation , sigmoid_derivative);
forward_pass(a,layer);
print_mat(layer->output);
free_dense_layer(layer);
free_mat(&a);
printf("\n");
return 0;
}
