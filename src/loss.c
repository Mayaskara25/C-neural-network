#include "matrix.h"
#include <stdio.h>
#include <math.h>
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

float mae(matrix y_predicted, matrix y_actual) {
    if(y_actual.cols != y_predicted.cols || y_actual.rows != y_predicted.rows){
        fprintf(stderr , "ERROR : Invlaid argument matrices for mae calculation");
        return 1;
    }
    matrix diff = sub_mat(y_predicted, y_actual); // e = y_pred - y_actual
    float total = 0;
    for( int i = 0 ; i < diff.rows ; i++){
        for( int j = 0; j < diff.cols ; j++){
            total += fabsf(mat_at(diff, i, j));
        }
    }
    float ans = total / (float)(y_actual.rows*y_actual.cols);
    free_mat(&diff);
    return ans;
}

matrix mae_derivative(matrix y_predicted, matrix y_actual){
    if(y_actual.cols != y_predicted.cols || y_actual.rows != y_predicted.rows){
        mat_error("ERROR : Invlaid argument matrices for mae calculation");
    }
    int N = y_actual.rows * y_actual.cols;
    matrix diff = sub_mat(y_predicted, y_actual); // e = y_pred - y_actual
    matrix ans = create_mat(diff.rows, diff.cols);
    for( int i = 0 ; i < diff.rows ; i++){
        for( int j = 0 ; j < diff.cols ; j++){
            float e = mat_at(diff, i, j);
            float sign = (e > 0.0f) ? 1.0f : (e < 0.0f ? -1.0f : 0.0f);
            mat_at(ans, i, j) = sign / (float)N;
        }
    }
    free_mat(&diff);
    return ans;
}

float huber(matrix y_predicted, matrix y_actual) {
    if(y_actual.cols != y_predicted.cols || y_actual.rows != y_predicted.rows){
        fprintf(stderr , "ERROR : Invlaid argument matrices for huber calculation");
        return 1;
    }
    const float delta = 1.0f;
    matrix diff = sub_mat(y_predicted, y_actual);
    float total = 0;
    for( int i = 0 ; i < diff.rows ; i++){
        for( int j = 0 ; j < diff.cols ; j++){
            float e = mat_at(diff, i, j);
            float ae = fabsf(e);
            total += (ae <= delta) ? (0.5f * e * e) : (delta * (ae - 0.5f * delta));
        }
    }
    float ans = total / (float)(y_actual.rows*y_actual.cols);
    free_mat(&diff);
    return ans;
}

matrix huber_derivative(matrix y_predicted, matrix y_actual){
    if(y_actual.cols != y_predicted.cols || y_actual.rows != y_predicted.rows){
        mat_error("ERROR : Invlaid argument matrices for huber calculation");
    }
    const float delta = 1.0f;
    int N = y_actual.rows * y_actual.cols;
    matrix diff = sub_mat(y_predicted, y_actual);
    matrix ans = create_mat(diff.rows, diff.cols);
    for( int i = 0 ; i < diff.rows ; i++){
        for( int j = 0 ; j < diff.cols ; j++){
            float e = mat_at(diff, i, j);
            if(fabsf(e) <= delta){
                mat_at(ans, i, j) = e / (float)N;
            } else {
                float sign = (e > 0.0f) ? 1.0f : -1.0f;
                mat_at(ans, i, j) = (delta * sign) / (float)N;
            }
        }
    }
    free_mat(&diff);
    return ans;
}

float binary_cross_entropy(matrix y_predicted, matrix y_actual) {
    if(y_actual.cols != y_predicted.cols || y_actual.rows != y_predicted.rows){
        fprintf(stderr , "ERROR : Invlaid argument matrices for binary_cross_entropy calculation");
        return 1;
    }
    const float eps = 1e-7f;
    float total = 0;
    for( int i = 0 ; i < y_predicted.rows ; i++){
        for( int j = 0 ; j < y_predicted.cols ; j++){
            float p = mat_at(y_predicted, i, j);
            if(p < eps) p = eps;
            if(p > 1.0f - eps) p = 1.0f - eps;
            float y = mat_at(y_actual, i, j);
            total += -( y * logf(p) + (1.0f - y) * logf(1.0f - p) );
        }
    }
    return total / (float)(y_actual.rows*y_actual.cols);
}

matrix binary_cross_entropy_derivative(matrix y_predicted, matrix y_actual){
    if(y_actual.cols != y_predicted.cols || y_actual.rows != y_predicted.rows){
        mat_error("ERROR : Invlaid argument matrices for binary_cross_entropy calculation");
    }
    const float eps = 1e-7f;
    int N = y_actual.rows * y_actual.cols;
    matrix ans = create_mat(y_predicted.rows, y_predicted.cols);
    for( int i = 0 ; i < y_predicted.rows ; i++){
        for( int j = 0 ; j < y_predicted.cols ; j++){
            float p = mat_at(y_predicted, i, j);
            if(p < eps) p = eps;
            if(p > 1.0f - eps) p = 1.0f - eps;
            float y = mat_at(y_actual, i, j);
            // raw dL/dy_pred (NOT fused with sigmoid'): (p - y) / (p*(1-p)) / N
            mat_at(ans, i, j) = ( (p - y) / (p * (1.0f - p)) ) / (float)N;
        }
    }
    return ans;
}

void get_loss_fxn(LossID id, float (**loss)(matrix, matrix), matrix (**loss_derivative)(matrix, matrix)){
    switch(id){
        case LOSS_MSE:
            *loss = mse;
            *loss_derivative = mse_derivative;
            break;
        case LOSS_MAE:
            *loss = mae;
            *loss_derivative = mae_derivative;
            break;
        case LOSS_HUBER:
            *loss = huber;
            *loss_derivative = huber_derivative;
            break;
        default:
            fprintf(stderr , "error : unknown loss id %d, falling back to mse\n" , id);
            *loss = mse;
            *loss_derivative = mse_derivative;
            break;
    }
}
