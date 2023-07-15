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
	//��������ݶȷ����˶��棬��ôͼ�ͻ�ʧЧ,tensor_a�Ժ����Ĺ��ɽڵ�ĺ����������Ӷ���ʧЧ
	ATtensor tensorb;
	tensorb.resize(1, 1);
	tensorb.setvalue(1);
	tensora = tensora + tensorb;
	autograd::run_backward(*tensora[0][0]);
	printTensor(tensora);
	printTensor(tensorb);
	//���ǣ���������ǰ����˶���ģ��Ե��ӣ�,
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
	//������ǳ��˷��أ�����ʧЧ�ķ�����?
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
	//Ŀǰ�����������κ����⣬��֪��Ϊʲô
	//���ֳ˷�����
	ATtensor tensorp;
	tensorp.resize(2, 2);
	tensorp.setvalue(1);
	ATtensor tensorq = (4 * tensorp ) ; //�����������˷������磺(4 * tensorp ) *4 �ͻᱨ����֪��Ϊɶ
	std::cout << "***********************" << std::endl;
	autograd::run_backward(*tensorq[1][1]);
	printTensor(tensorp);
	printTensor(tensorq);
	//����ӷ�
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
