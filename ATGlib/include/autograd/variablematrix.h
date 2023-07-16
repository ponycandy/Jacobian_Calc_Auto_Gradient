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
	const Atvariable_Ptr* operator[](int i) const;
	std::vector<Atvariable_Ptr> data;
	
};


//Some important symbol overloading

ATtensor operator*(const ATtensor& a, const double b);

ATtensor operator*(const double b, const ATtensor& a);


ATtensor operator/(const ATtensor& a, const double b);


ATtensor operator*(const ATtensor& b, const ATtensor& a);

ATtensor operator+(const ATtensor& b, const ATtensor& a);

//Operation of constant matrix
ATtensor operator*(const ATtensor& b, const Eigen::MatrixXd& a);
ATtensor operator*(const Eigen::MatrixXd& b , const ATtensor& a);
ATtensor operator+(const ATtensor& b, const Eigen::MatrixXd& a);
ATtensor operator+(const Eigen::MatrixXd& b, const ATtensor& a);
//debugging tool

void printTensor(ATtensor& a);

//Jacobi solution
void GetJacobian(ATtensor& leftside, ATtensor& upside, Eigen::MatrixXd& returnmat);
//void GetJacobian(ATGTensor& leftside, ATGTensor& upside, Eigen::MatrixXd& returnmat);

#endif 
