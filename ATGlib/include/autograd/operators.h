#if !defined(__OPERATORS_H__)
#define __OPERATORS_H__

#include "autograd/autograd.h"

namespace autograd {

class AddBackward : public Node {
public:
  variable_list apply(variable_list &&grads) override;
};

class SubBackward : public Node {
public:
  variable_list apply(variable_list &&grads) override;
};

class AccumulateGrad : public Node {
public:
  variable_list apply(variable_list &&grads) override;
  std::weak_ptr<Variable> variable_;
};

class MulBackward : public Node {
public:
  variable_list apply(variable_list &&grads) override;
  std::shared_ptr<Variable> other_;
  std::shared_ptr<Variable> self_;
};

class DivBackward : public Node {
public:
  variable_list apply(variable_list &&grads) override;
  std::shared_ptr<Variable> other_;
  std::shared_ptr<Variable> self_;
};

class PowBackward : public Node {
public:
  variable_list apply(variable_list &&grads) override;
  std::shared_ptr<Variable> other_;
  std::shared_ptr<Variable> self_;
};

class LogBackward : public Node {
public:
  variable_list apply(variable_list &&grads) override;
  std::shared_ptr<Variable> self_;
};


class ReLUBackward : public Node {
public:
  variable_list apply(variable_list &&grads) override;
  std::shared_ptr<Variable> self_;
};

class NegBackward : public Node {
public:
  variable_list apply(variable_list &&grads) override;
};
//以下是我的添加：
class AddBackwardConstant : public Node {
public:
	variable_list apply(variable_list&& grads) override;
};
class MulBackwardConstant : public Node {
public:
	variable_list apply(variable_list&& grads) override;
	double other_;  //总是将常数值赋给others
	std::shared_ptr<Variable> self_;
};
} // namespace autograd

#endif // __OPERATORS_H__
