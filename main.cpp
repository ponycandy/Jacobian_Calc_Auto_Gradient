#include <iostream>
#include <autograd/autograd.h>
#include <Eigen/core>
#include <autograd/variablematrix.h> 

int main()

{
	//no error
	//Eigen::Matrix<std::shared_ptr<double>, Eigen::Dynamic, Eigen::Dynamic> mat;
	//mat.resize(2, 1);
	//std::shared_ptr<double> mat01(new double(2));
	//mat << mat01, mat01;
	//std::shared_ptr<double> value = mat(1, 0);
	//double a = *value;
	//std::cout << a << std::endl;

	//no error
	//Eigen::Matrix<std::shared_ptr<Mdouble>, Eigen::Dynamic, Eigen::Dynamic> mat;
	//mat.resize(2, 1);
	//std::shared_ptr<Mdouble> mat01(new Mdouble);
	//mat (0,0)= mat01;


	//failed
	std::shared_ptr<autograd::Variable> xvar = autograd::variable(1);
	std::shared_ptr<autograd::Variable> xvar2 = autograd::variable(2);

	Eigen::Matrix<std::shared_ptr<autograd::Variable>, Eigen::Dynamic, Eigen::Dynamic> mat;
	mat.resize(1, 1);
	mat(0, 0) = xvar;
	/*mat << autograd::variable(1), autograd::variable(2);*/
	Eigen::Matrix<std::shared_ptr<autograd::Variable>, Eigen::Dynamic, Eigen::Dynamic> mat2;
	mat2.resize(1, 1);
	mat2(0, 0) = xvar2;

	mat2=mat;

	std::cout << mat2(0, 0)->value_ << std::endl;
	autograd::run_backward(*mat2(0, 0));
	std::cout << mat(0, 0)->grad_ << std::endl;
	return 0;

}
