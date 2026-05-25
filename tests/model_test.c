#include "matrix.h"
#include "layer.h"
#include "activation.h"
#include "model.h"
#include <time.h>
#include <stdlib.h>
int main() {
    srand(time(NULL));
        printf("input matrix:\n");
        matrix a = create_mat(3,1);
        for( int i = 0 ; i < a.rows ; i++) {
            for ( int j= 0 ; j < a.cols ;j++) {
                scanf("%f" ,&mat_at(a,i,j));
            }
        }
        print_mat(a);
        printf("\n");
        NeuralNetwork* nn = create_neural_network();
        DenseLayer* l1 = create_dense_layer(3, 4, sigmoid_activation, sigmoid_derivative);
        add_layer(nn,l1 );
        DenseLayer* l2 = create_dense_layer(4, 4,sigmoid_activation , sigmoid_derivative);
        add_layer(nn, l2);
        DenseLayer* l3 = create_dense_layer(4, 4,sigmoid_activation , sigmoid_derivative);
        add_layer(nn, l3);
        matrix const* final_output = forward_network(a,nn);
        print_mat(*final_output);
        free_neural_network(nn);
        free_mat(&a);

return 0;
}
