#include "autograd/variablematrix.h"
#include <iostream>

//void GetJacobian(ATGvector& leftside, ATGvector& upside, Eigen::MatrixXd& returnmat)
//{
//	int rows = leftside.size();
//	int cols = upside.size();
//	int i = 0;
//	int j = 0;
//	for (auto x : leftside)
//	{
//		autograd::run_backward(*x);
//		for (auto y : upside)
//		{
//			returnmat(i, j) = y->grad_;
//			y->grad_ = 0;
//			j++;
//		}
//		j = 0;
//		i++;
//	}
//
//}
//
//void GetJacobian(ATGTensor& leftside, ATGTensor& upside, Eigen::MatrixXd& returnmat)
//{
//	int rows = leftside.rows();
//	int cols = upside.rows();
//	ATGvector lft;
//	for (size_t i = 0; i < rows; i++)
//	{
//		lft.push_back(leftside(i,0));
//	}
//	ATGvector upd;
//	for (int  i = 0; i < cols; i++)
//	{
//		upd.push_back(upside(i, 0));
//	}
//	GetJacobian(lft, upd, returnmat);
//}

ATtensor::ATtensor()
{
}

ATtensor::~ATtensor()
{
	//delete[] data;
}

void ATtensor::resize(int nrows, int ncols)
{
	rows = nrows;
	cols = ncols;
	totalsize = rows * cols;
	data.reserve(totalsize);
	for (int i = 1; i <= totalsize; i++)
	{
		Atvariable_Ptr ptr = autograd::variable(0);
		data.push_back(ptr);
	}
}
void ATtensor::setvalue(double value)
{
	for (int i = 1; i <= totalsize; i++)
	{
		data[i - 1]->value_ = value;
	}
}
Atvariable_Ptr* ATtensor::operator[](int i) 
{
    // check if the index is valid
    if (i < 0 || i >= rows) 
    {
        throw std::out_of_range("Invalid row index");
    }
    // return the pointer to the row
    return data.data() + i * cols;
}

ATtensor operator*( ATtensor& a, double b)
{
	ATtensor newtensor;
	newtensor.resize(a.rows, a.cols);
	int i = 0;
	for (auto x : a.data)
	{
		newtensor.data[i] = b * x;
		i++;
	}
    return newtensor;
}
ATtensor operator*(double b,  ATtensor& a)
{
	ATtensor newtensor;
	newtensor.resize(a.rows, a.cols);
	int i = 0;
	for (auto x : a.data)
	{
		newtensor.data[i] = b * x;
		i++;
	}
	return newtensor;
}
ATtensor operator/(ATtensor& a, double b)
{
	ATtensor newtensor;
	newtensor.resize(a.rows, a.cols);
	int i = 0;
	for (auto x : a.data)
	{
		newtensor.data[i] = x / b;
		i++;
	}
	return newtensor;
}
//张量积
ATtensor operator*( ATtensor& b,  ATtensor& a)
{
	ATtensor newtensor;
	newtensor.resize(b.rows, a.cols);
	if (b.cols != a.rows)
	{
		throw std::out_of_range("matrix multiplication is not fit");
	}
	int brows = b.rows;
	int bcols = b.cols;
	int acols = a.cols;
	for (int i=0;i< brows;i++)
	{
		for (int j = 0; j < acols; j++)
		{
			//newtensor[i][j]=b[i][j] * a[j][i];
			for (int k = 0; k < bcols; k++)
			{
				newtensor[i][j] = newtensor[i][j]+(b[i][k] * a[k][j]);
			}
		}
	}
	return newtensor;
}

//张量和
ATtensor operator+(ATtensor& b, ATtensor& a)
{
	ATtensor newtensor;
	newtensor.resize(b.rows, a.cols);
	if ((b.rows != a.rows) || (b.cols != a.cols))
	{
		throw std::out_of_range("matrix size not same!");
	}
	int i = 0;
	int brows = b.rows;
	int aclos = a.cols;
	for (int i = 0; i < brows; i++)
	{
		for (int j = 0; j < aclos; j++)
		{
			newtensor[i][j] = b[i][j] + a[i][j];
		}
	}
	return newtensor;
}

void printTensor(ATtensor& a)
{
	std::string loginfo = "value \n";


	int arows = a.rows;
	int acols = a.cols;
	for (int i = 0; i < arows; i++)
	{
		for (int j = 0; j < acols; j++)
		{
			
			loginfo.append(std::to_string(a[i][j]->value_));
			loginfo.append("  ");

		}
		loginfo.append(" \n");
	}
	std::cout << loginfo << std::endl;


	std::string loginfo1 = "gradient \n";



	for (int i = 0; i < arows; i++)
	{
		for (int j = 0; j < acols; j++)
		{

			loginfo1.append(std::to_string(a[i][j]->grad_));
			loginfo1.append("  ");

		}
		loginfo1.append(" \n");
	}
	std::cout << loginfo1 << std::endl;
}
