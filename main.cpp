#include <iostream>
#include <autograd/autograd.h>
#include <autograd/variablematrix.h>
int main()

{
	/*std::shared_ptr<autograd::Variable> xvar = autograd::variable(1);
	Eigen::Matrix<std::shared_ptr<autograd::Variable>, 1, 1> mat1;

	mat1 << xvar;

	Eigen::Matrix<std::shared_ptr<autograd::Variable>, 2, 1> mat;

	mat << 4 * xvar, 2 * xvar;

	std::vector<std::shared_ptr<autograd::Variable>> var {xvar};
	std::vector<std::shared_ptr<autograd::Variable>> value {mat(0, 0), mat(1, 0)};

	Eigen::MatrixXd jac;
	jac.resize(2, 1);
	GetJacobian(value, var, jac);
	GetJacobian(mat, mat1, jac);
	std::cout << jac << std::endl;*/

	//Eigen::Matrix<std::shared_ptr<autograd::Variable>, row, col> mat2;
	//�����޷�ʹ��Eigentensor���ͣ�Ŀǰֻ��ֱ��ʹ��vector�����������ſɱ�
	//���ԣ�����vector��Ŭ�������ⲿ���У�
	return 0;
}
