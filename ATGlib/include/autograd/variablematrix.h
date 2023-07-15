#if !defined(__VARIABLEMATRIX_H)
#define __VARIABLEMATRIX_H
#include "Eigen/core"
#include "autograd/autograd.h"
//typedef Eigen::Matrix<std::shared_ptr<autograd::Variable>, Eigen::Dynamic, Eigen::Dynamic> ATGTensor;
//typedef std::vector<std::shared_ptr<autograd::Variable>> ATGvector;


//void GetJacobian(ATGvector& leftside, ATGvector& upside, Eigen::MatrixXd& returnmat);
//void GetJacobian(ATGTensor& leftside, ATGTensor& upside, Eigen::MatrixXd& returnmat);
typedef std::shared_ptr<autograd::Variable> Atvariable_Ptr;
typedef autograd::Variable Atvariable;

class ATtensor
{
public:
	ATtensor();
	~ATtensor();
	void resize(int nrows, int ncols);
	void setvalue(double value);
	int rows;
	int cols;
	int totalsize;
	Atvariable_Ptr* operator[](int i);
	std::vector<Atvariable_Ptr> data;
	
};


//一些重要的符号重载

ATtensor operator*(ATtensor& a, double b);

ATtensor operator*(double b, ATtensor& a);


ATtensor operator/(ATtensor& a, double b);


ATtensor operator*(ATtensor& b, ATtensor& a);

ATtensor operator+(ATtensor& b, ATtensor& a);

//常数矩阵的乘法...这个还是有必要的


//debug调试工具

void printTensor(ATtensor& a);

#endif 
