#include <iostream>
#include <autograd/autograd.h>
#include <autograd/variablematrix.h>
int main()

{
	Eigen::Matrix<std::shared_ptr<double>, Eigen::Dynamic, Eigen::Dynamic> mat;
	mat.resize(2, 1);
	std::shared_ptr<double> mat01(new double(2));
	mat << mat01, mat01;
	std::shared_ptr<double> value = mat(1, 0);
	double a = *value;
	std::cout << a << std::endl;
	return 0;
}
