#include <iostream>
#include <autograd/autograd.h>
#include <Eigen/core>
//#include <autograd/variablematrix.h> 
struct Mdouble:public std::enable_shared_from_this<Mdouble>
{
	int a = 0;
};
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
	
	Eigen::Matrix<std::shared_ptr<autograd::Variable>, Eigen::Dynamic, Eigen::Dynamic> mat;
	mat.resize(2, 1);
	std::shared_ptr<autograd::Variable> xvar(new autograd::Variable);
	std::shared_ptr<autograd::Variable> xvar2(new autograd::Variable);
	mat(0, 0)=xvar;
	mat + mat;


	return 0;
	//问题出在这里：一旦尝试客制化乘法的时候，就会出现问题了！
}
