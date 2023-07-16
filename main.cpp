#include <iostream>
#include <autograd/autograd.h>
#include <Eigen/core>
#include <autograd/variablematrix.h> 
//add transpose and continuously product
int main()

{
	ATtensor tensor_a;
	tensor_a.resize(1, 1);
	ATtensor tensor_b=tensor_a;
	ATtensor tensor_c = tensor_a;
	ATtensor tensord = tensor_a + tensor_b + tensor_c;
	printTensor(tensord);
}
