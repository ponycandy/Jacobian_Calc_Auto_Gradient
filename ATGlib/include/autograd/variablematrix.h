#if !defined(__VARIABLEMATRIX_H)
#define __VARIABLEMATRIX_H
#include "Eigen/core"
#include "autograd/autograd.h"




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

//常数矩阵的运算
ATtensor operator*(ATtensor& b, Eigen::MatrixXd& a);
ATtensor operator*(Eigen::MatrixXd& b ,ATtensor& a);
ATtensor operator+(ATtensor& b, Eigen::MatrixXd& a);
ATtensor operator+(Eigen::MatrixXd& b, ATtensor& a);
//debug调试工具

void printTensor(ATtensor& a);

//雅可比求解
void GetJacobian(ATtensor& leftside, ATtensor& upside, Eigen::MatrixXd& returnmat);
//void GetJacobian(ATGTensor& leftside, ATGTensor& upside, Eigen::MatrixXd& returnmat);

#endif 
