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


//һЩ��Ҫ�ķ�������

ATtensor operator*(ATtensor& a, double b);

ATtensor operator*(double b, ATtensor& a);


ATtensor operator/(ATtensor& a, double b);


ATtensor operator*(ATtensor& b, ATtensor& a);

ATtensor operator+(ATtensor& b, ATtensor& a);

//�������������
ATtensor operator*(ATtensor& b, Eigen::MatrixXd& a);
ATtensor operator*(Eigen::MatrixXd& b ,ATtensor& a);
ATtensor operator+(ATtensor& b, Eigen::MatrixXd& a);
ATtensor operator+(Eigen::MatrixXd& b, ATtensor& a);
//debug���Թ���

void printTensor(ATtensor& a);

//�ſɱ����
void GetJacobian(ATtensor& leftside, ATtensor& upside, Eigen::MatrixXd& returnmat);
//void GetJacobian(ATGTensor& leftside, ATGTensor& upside, Eigen::MatrixXd& returnmat);

#endif 
