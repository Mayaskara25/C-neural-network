#include <stdio.h>
#include <math.h>
#include "activation.h"

 float sigmoid_activation(float a) {return ( 1.0f / (1.0f + expf(-a))); }
 float sigmoid_derivative(float a) { return ( ( expf(-a)) / powf( (1.0 + expf(-a)) , 2 ) ); }
    //another formula -> d/dx(sigmoid(a)) = sigmoid(a) * (1 - sigmoid(a))

 float relu_activation(float a){ return ( a > 0  ? a : 0 ); }
 float relu_derivative(float a ){ return (a > 0 ? 1 : 0 ) ;}

 float tanh_activation(float a){ return (  (expf(a) - expf(-a)) / (expf(a) + expf(-a)) ) ; }
 float tanh_derivative(float a){ return  ( (4.0f) / powf( ( expf(a) + expf(-a) ) , 2 ) ) ; }
    //another formula -> d/dx(tanh(a)) = 1 -(tanh(a))^2 ;
