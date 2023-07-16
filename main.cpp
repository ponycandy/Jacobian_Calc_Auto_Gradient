#include <iostream>
#include <autograd/autograd.h>
#include <Eigen/core>
#include <autograd/variablematrix.h> 
//add transpose and continuously product
int main()

{
	ATtensor tensor_a;
	tensor_a.resize(1, 2);
	tensor_a.setvalue(1);
	ATtensor tensor_b=tensor_a;
	ATtensor tensor_c = 4 * tensor_a.Transpose();
	//Transpose会把梯度传到原张量
	Eigen::MatrixXd mat(2, 2);
	GetJacobian(tensor_c, tensor_a, mat);
	std::cout << mat << std::endl;
}
