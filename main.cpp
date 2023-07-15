#include <iostream>
#include <autograd/autograd.h>
#include <Eigen/core>
#include <autograd/variablematrix.h> 

int main()

{
	ATtensor tensor_a;
	tensor_a.resize(1, 2);
	tensor_a.setvalue(2);
	//���ǳ�ʼ��Ϊȫ0Ԫ��,��ʱ���ṩ������ʼ��������
	Atvariable_Ptr ptrat = autograd::variable(0);
	printTensor(tensor_a);
	ATtensor tensor_b;
	tensor_b.resize(2, 1);
	tensor_b.setvalue(1);
	ATtensor tensor_c = tensor_a * tensor_b;
	printTensor(tensor_c);
	autograd::run_backward(*tensor_c[0][0]);

	std::cout << "after BP, tensor_a \n" << std::endl;
	printTensor(tensor_a);
	std::cout << "after BP, tensor_b \n" << std::endl;
	printTensor(tensor_b);
}
