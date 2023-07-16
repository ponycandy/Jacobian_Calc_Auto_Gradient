#include "autograd/variablematrix.h"
#include <iostream>


ATtensor::ATtensor()
{
}

ATtensor::~ATtensor()
{

}

ATtensor ATtensor::Transpose()
{
	ATtensor newtensor;
	newtensor.cols = rows;
	newtensor.rows = cols;
	newtensor.data = data;
	return newtensor;
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

const Atvariable_Ptr* ATtensor::operator[](int i) const
{
	// check if the index is valid
	if (i < 0 || i >= rows) {
		throw std::out_of_range("Invalid row index");
	}
	// return the pointer to the row
	return data.data() + i * cols;
}

ATtensor operator*(const ATtensor& a, const double b)
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
ATtensor operator*(const double b, const ATtensor& a)
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
ATtensor operator/(const ATtensor& a, const double b)
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
ATtensor operator*(const ATtensor& b, const  ATtensor& a)
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
ATtensor operator+(const ATtensor& b, const ATtensor& a)
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
//常数矩阵
ATtensor operator*(const ATtensor& b, const Eigen::MatrixXd& a)
{
	ATtensor newtensor;
	int brows = b.rows;
	int bcols = b.cols;
	int acols = a.cols();
	newtensor.resize(b.rows, acols);
	if (b.cols != a.rows())
	{
		throw std::out_of_range("matrix multiplication is not fit");
	}

	for (int i = 0; i < brows; i++)
	{
		for (int j = 0; j < acols; j++)
		{
			//newtensor[i][j]=b[i][j] * a[j][i];
			for (int k = 0; k < bcols; k++)
			{
				newtensor[i][j] = newtensor[i][j] + (b[i][k] * a(k,j));
			}
		}
	}
	return newtensor;
}
ATtensor operator*(const Eigen::MatrixXd& b, const ATtensor& a)
{
	ATtensor newtensor;
	int brows = b.rows();
	int bcols = b.cols();
	int acols = a.cols;
	newtensor.resize(b.rows(), acols);
	if (b.cols() != a.rows)
	{
		throw std::out_of_range("matrix multiplication is not fit");
	}

	for (int i = 0; i < brows; i++)
	{
		for (int j = 0; j < acols; j++)
		{
			for (int k = 0; k < bcols; k++)
			{
				newtensor[i][j] = newtensor[i][j] + (b(i,k) * a[k][j]);
			}
		}
	}
	return newtensor;
}
ATtensor operator+(const ATtensor& b, const Eigen::MatrixXd& a)
{
	ATtensor newtensor;
	newtensor.resize(b.rows, a.cols());
	if ((b.rows != a.rows()) || (b.cols != a.cols()))
	{
		throw std::out_of_range("matrix size not same!");
	}
	int i = 0;
	int brows = b.rows;
	int aclos = a.cols();
	for (int i = 0; i < brows; i++)
	{
		for (int j = 0; j < aclos; j++)
		{
			newtensor[i][j] = b[i][j] + a(i,j);
		}
	}
	return newtensor;
}
ATtensor operator+(const Eigen::MatrixXd& b, const ATtensor& a)
{
	ATtensor newtensor;
	newtensor.resize(b.rows(), a.cols);
	if ((b.rows() != a.rows) || (b.cols() != a.cols))
	{
		throw std::out_of_range("matrix size not same!");
	}
	int i = 0;
	int brows = b.rows();
	int aclos = a.cols;
	for (int i = 0; i < brows; i++)
	{
		for (int j = 0; j < aclos; j++)
		{
			newtensor[i][j] = b(i,j) + a[i][j];
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

void GetJacobian(ATtensor& leftside, ATtensor& upside, Eigen::MatrixXd& returnmat)
{
//考虑到如果矩阵已经匹配了尺寸，每次都在内部resize会比较消耗内存，所以总是在外部resize
	int rows = leftside.totalsize; 
	int cols = upside.totalsize;
	int i = 0;
	int j = 0;
	for (auto x : leftside.data)
	{
		autograd::run_backward(*x);
		for (auto y : upside.data)
		{
			returnmat(i, j) = y->grad_;
			y->grad_ = 0;
			j++;
		}
		j = 0;
		i++;
	}
}
