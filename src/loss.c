#include "matrix.h"
#include <stdio.h>
#include "loss.h"


float mse(matrix y_predicted, matrix y_actual) {
    if(y_actual.cols != y_predicted.cols || y_actual.rows != y_predicted.rows){
        fprintf(stderr , "ERROR : Invlaid argument matrices for mse calculation");
        return 1;
    }
    matrix diff = sub_mat(y_actual, y_predicted);
    matrix sqDiff = elemul_mat(diff, diff);
    float total = 0;
    for( int i = 0 ; i < sqDiff.rows ; i++){
        for( int j = 0; j < sqDiff.cols ; j++){
            total+=mat_at(sqDiff, i, j);
        }
    }
    float mse = ( total / (float)(y_actual.rows*y_actual.cols) );
    free_mat(&diff);
    free_mat(&sqDiff);
    return mse;
}

matrix mse_derivative(matrix y_predicted, matrix y_actual){
    if(y_actual.cols != y_predicted.cols || y_actual.rows != y_predicted.rows){
        mat_error("ERROR : Invlaid argument matrices for mse calculation");
    }
    float coeff = -2.0f /( y_actual.rows*y_actual.cols );
    matrix diff = sub_mat(y_actual, y_predicted); //ORDER IS IMPORTANT , I.E IT CANT BE " ACTUAL , PREDICTED "-> y-y' not y'-y
    matrix ans = scalarmul_mat(diff, coeff );
    free_mat(&diff);
    return ans;
}
