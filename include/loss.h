#ifndef LOSS_H
#define LOSS_H
#include "matrix.h"


float mse(matrix y_predicted , matrix y_actual); //mse = (1/N)summation(y-y')^2
matrix mse_derivative(matrix y_predicted , matrix y_actual); //differentiate the above term for mse wrt y'
                                                            //d/dy'(mse) = -(2/N)(y-y');
#endif
