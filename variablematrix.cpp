#include "autograd/variablematrix.h"

void GetJacobian(std::vector<std::shared_ptr<autograd::Variable>>& leftside, std::vector<std::shared_ptr<autograd::Variable>>& upside, Eigen::MatrixXd& returnmat)
{
	int rows = leftside.size();
	int cols = upside.size();
	int i = 0;
	int j = 0;
	for (auto x : leftside)
	{
		autograd::run_backward(*x);
		for (auto y : upside)
		{
			returnmat(i, j) = y->grad_;
			y->grad_ = 0;
			j++;
		}
		j = 0;
		i++;
	}

}