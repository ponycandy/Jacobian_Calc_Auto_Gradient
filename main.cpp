#include <iostream>
#include <autograd/autograd.h>
#include <Eigen/core>
#include <autograd/variablematrix.h> 

int main()

{
	//Test + tensor
	ATtensor tensora;
	tensora.resize(1, 1);
	tensora.setvalue(1);
	tensora = tensora + tensora;
	autograd::run_backward(*tensora[0][0]);
	printTensor(tensora);
	//如果反向梯度发生了顶替，那么图就会失效,tensor_a以后参与的构成节点的后面所有链接都会失效
	ATtensor tensorb;
	tensorb.resize(1, 1);
	tensorb.setvalue(1);
	tensora = tensora + tensorb;
	autograd::run_backward(*tensora[0][0]);
	printTensor(tensora);
	printTensor(tensorb);
	//但是，矩阵相乘是包含了顶替的（自叠加）,
	ATtensor tensorc;
	tensorc.resize(1, 1);
	tensorc.setvalue(1);
	ATtensor tensord;
	tensord.resize(1, 1);
	tensord.setvalue(1);
	ATtensor tensore = tensorc * tensord;
	std::cout << "* test" << std::endl;
	autograd::run_backward(*tensore[0][0]);
	printTensor(tensorc);
	printTensor(tensord);
	//那如果是长乘法呢？会有失效的风险吗?
	ATtensor tensor1;
	tensor1.resize(2, 2);
	tensor1.setvalue(0.4);
	ATtensor tensor2;
	tensor2.resize(2, 2);
	tensor2.setvalue(0.5);
	ATtensor tensor3 = tensor1 * tensor2;
	std::cout << "\\\\\\\\\\\\\\\\\kjhjk" << std::endl;
	autograd::run_backward(*tensor3[0][0]);
	printTensor(tensor1);
	printTensor(tensor2);
	std::cout << "\\\\\\\\\\\\\\\\\kjhjk" << std::endl;
	autograd::run_backward(*tensor3[1][1]);
	printTensor(tensor1);
	printTensor(tensor2);
	//目前来看不会有任何问题，不知道为什么
	//数字乘法测试
	ATtensor tensorp;
	tensorp.resize(2, 2);
	tensorp.setvalue(1);
	ATtensor tensorq = (4 * tensorp ) ; //不允许连续乘法，比如：(4 * tensorp ) *4 就会报错，不知道为啥
	std::cout << "***********************" << std::endl;
	autograd::run_backward(*tensorq[1][1]);
	printTensor(tensorp);
	printTensor(tensorq);
	//矩阵加法
	ATtensor tensorl;
	tensorl.resize(2, 2);
	tensorl.setvalue(1);
	Eigen::MatrixXd mat;
	mat.resize(2, 2);
	mat.setOnes();
	ATtensor m_tensor = tensorl + mat;
	std::cout << "sauiodhqwhdiu" << std::endl;
	autograd::run_backward(*m_tensor[1][1]);
	printTensor(m_tensor);
	printTensor(tensorl);
}
