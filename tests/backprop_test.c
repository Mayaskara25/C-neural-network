#include "loss.h"
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
        printf("output matrix:\n");
        matrix o = create_mat(4, 1);
        for( int i = 0 ; i < o.rows ; i++) {
            for ( int j= 0 ; j < o.cols ;j++) {
                scanf("%f" ,&mat_at(o,i,j));
            }
        }


        NeuralNetwork* nn = create_neural_network();
        DenseLayer* l1 = create_dense_layer(3, 4, sigmoid_activation, sigmoid_derivative);
        add_layer(nn,l1 );
        DenseLayer* l2 = create_dense_layer(4, 4,sigmoid_activation , sigmoid_derivative);
        add_layer(nn, l2);
        DenseLayer* l3 = create_dense_layer(4, 4,sigmoid_activation , sigmoid_derivative);
        add_layer(nn, l3);
        DenseLayer* l4 = create_dense_layer(4, 4 , no_activation , no_activ_derivative);
        add_layer(nn, l4);

        for ( int i = 0 ; i < 10000;  i++){
            matrix const* output = forward_network(a,nn);
            float loss = mse(*output,o);
            if( i%1000 == 0) {
            printf("loss at round %d:%f\n",i + 1, loss);
            }
            matrix loss_grad = mse_derivative(*output,o);
            backward_network(loss_grad, nn);
            update_weights(nn,0.1);
            free_mat(&loss_grad);
            if( i == 9999) {
                matrix k = copy_mat(*output);
                print_mat(k);
                free_mat(&k);
            }
        }
        //print_mat(*output);
        free_neural_network(nn);
        free_mat(&a);
        free_mat(&o);

return 0;
}
