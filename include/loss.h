#ifndef LOSS_H
#define LOSS_H
#include "matrix.h"

typedef enum {
    LOSS_MSE = 0,
    LOSS_MAE = 1,
    LOSS_HUBER = 2
} LossID;

float mse(matrix y_predicted , matrix y_actual); //mse = (1/N)summation(y-y')^2
matrix mse_derivative(matrix y_predicted , matrix y_actual); //differentiate the above term for mse wrt y'
                                                            //d/dy'(mse) = -(2/N)(y-y');

float mae(matrix y_predicted , matrix y_actual); //mae = (1/N)summation|y_pred - y_actual|
matrix mae_derivative(matrix y_predicted , matrix y_actual); //d/dy_pred(mae) = sign(y_pred - y_actual) / N

float huber(matrix y_predicted , matrix y_actual); //delta hardcoded 1.0f
matrix huber_derivative(matrix y_predicted , matrix y_actual);
// let e = y_pred - y_actual, delta = 1.0f:
//   |e| <= delta : dL/dy_pred = e/N
//   |e| >  delta : dL/dy_pred = (delta/N)*sign(e)

// NOT wired into LossID/get_loss_fxn/train() yet -- keep for later.
// Requires clamped y_pred in [1e-7, 1-1e-7]. Returns RAW (unfused) dL/dy_pred;
// backward_pass() already multiplies by the output layer's activation_derivative,
// so do NOT pre-multiply by sigmoid'(z) here.
// Meaningful only when the output layer uses sigmoid activation.
float binary_cross_entropy(matrix y_predicted , matrix y_actual);
matrix binary_cross_entropy_derivative(matrix y_predicted , matrix y_actual);

void get_loss_fxn(LossID id, float (**loss)(matrix, matrix), matrix (**loss_derivative)(matrix, matrix));
#endif
