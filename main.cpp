

#include <iostream>
#include <autograd/autograd.h>
using namespace autograd;
int main()

{
	
	//std::shared_ptr<Variable> e = variable(std::exp(1));
	//std::shared_ptr<Variable> x = variable(3.0);
	//std::shared_ptr<Variable> z = e ^ x;
	std::shared_ptr<Variable> p = variable(3.0);
	auto dd =  7*p;
	auto uu = dd*7;
	//ASSERT_FLOAT_EQ(z->value_, std::exp(3.0));
	//
	//ASSERT_FLOAT_EQ(x->grad_, std::exp(3.0));
	//autograd::run_backward(*z);
	//std::cout << x->grad_ << std::endl;
	//
	//
	autograd::run_backward(*uu);
	std::cout << p->grad_ << std::endl;
	return 0;
}