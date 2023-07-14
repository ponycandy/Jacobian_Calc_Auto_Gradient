#include "autograd/variable.h"
#include "autograd/operators.h"

#include <cmath>
#include <memory>

namespace autograd {

std::shared_ptr<Variable> variable(float v) {
  return std::make_shared<Variable>(v);
}

void Variable::set_gradient_edge(Edge &&gradient_edge) {
  gradient_edge_ = gradient_edge;
}

Edge Variable::gradient_edge() {
  if (!gradient_edge_.grad_fn() && requires_grad_) {
    std::shared_ptr<AccumulateGrad> grad_fn =
        std::make_shared<AccumulateGrad>();
    grad_fn->variable_ = shared_from_this();
    grad_fn->add_input_nr();
    gradient_edge_.set_grad_fn(grad_fn);
  }
  return gradient_edge_;
}

std::shared_ptr<Variable> Variable::detach() {
  std::shared_ptr<Variable> variable = std::make_shared<Variable>(value_);
  variable->requires_grad_ = false;
  return variable;
}

std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs) {
  std::shared_ptr<AddBackward> grad_fn = std::make_shared<AddBackward>();
  grad_fn->add_input_nr();
  auto result = std::make_shared<Variable>(lhs->value_ + rhs->value_);
  result->set_gradient_edge({grad_fn, 0});
  grad_fn->add_next_edge(lhs->gradient_edge());
  grad_fn->add_next_edge(rhs->gradient_edge());
  return result;
}

std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs) {
  std::shared_ptr<SubBackward> grad_fn = std::make_shared<SubBackward>();
  grad_fn->add_input_nr();
  auto result = std::make_shared<Variable>(lhs->value_ - rhs->value_);
  result->set_gradient_edge({grad_fn, 0});
  grad_fn->add_next_edge(lhs->gradient_edge());
  grad_fn->add_next_edge(rhs->gradient_edge());
  return result;
}

std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs) {
  std::shared_ptr<MulBackward> grad_fn = std::make_shared<MulBackward>();
  grad_fn->self_ = lhs;
  grad_fn->other_ = rhs;
  grad_fn->add_input_nr();
  auto result = std::make_shared<Variable>(lhs->value_ * rhs->value_);
  result->set_gradient_edge({grad_fn, 0});
  grad_fn->add_next_edge(lhs->gradient_edge());
  grad_fn->add_next_edge(rhs->gradient_edge());
  return result;
}

std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs) {
  std::shared_ptr<DivBackward> grad_fn = std::make_shared<DivBackward>();
  grad_fn->self_ = lhs;
  grad_fn->other_ = rhs;
  grad_fn->add_input_nr();
  auto result = std::make_shared<Variable>(lhs->value_ / rhs->value_);
  result->set_gradient_edge({grad_fn, 0});
  grad_fn->add_next_edge(lhs->gradient_edge());
  grad_fn->add_next_edge(rhs->gradient_edge());
  return result;
}

std::shared_ptr<Variable> operator^(std::shared_ptr<Variable> lhs,
                                    std::shared_ptr<Variable> rhs) {
  std::shared_ptr<PowBackward> grad_fn = std::make_shared<PowBackward>();
  grad_fn->self_ = lhs;
  grad_fn->other_ = rhs;
  grad_fn->add_input_nr();
  auto result = std::make_shared<Variable>(std::pow(lhs->value_, rhs->value_));
  result->set_gradient_edge({grad_fn, 0});
  grad_fn->add_next_edge(lhs->gradient_edge());
  grad_fn->add_next_edge(rhs->gradient_edge());
  return result;
}

std::shared_ptr<Variable> Variable::log() {
  std::shared_ptr<LogBackward> grad_fn = std::make_shared<LogBackward>();
  grad_fn->self_ = shared_from_this();
  grad_fn->add_input_nr();
  auto result = std::make_shared<Variable>(std::log(value_));
  result->set_gradient_edge({grad_fn, 0});
  grad_fn->add_next_edge(gradient_edge());
  return result;
}

std::shared_ptr<Variable> Variable::relu() {
  std::shared_ptr<ReLUBackward> grad_fn = std::make_shared<ReLUBackward>();
  grad_fn->self_ = shared_from_this();
  grad_fn->add_input_nr();
  auto result = std::make_shared<Variable>(std::max(value_, 0.0f));
  result->set_gradient_edge({grad_fn, 0});
  grad_fn->add_next_edge(gradient_edge());
  return result;
}

std::shared_ptr<Variable> Variable::sigmoid() {
  auto one = std::make_shared<Variable>(1.0);
  auto e = std::make_shared<Variable>(std::exp(1.0));
  return one / (one + (e ^ (-shared_from_this())));
}

std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> var) {
  std::shared_ptr<NegBackward> grad_fn = std::make_shared<NegBackward>();
  grad_fn->add_input_nr();
  auto result = std::make_shared<Variable>(-var->value_);
  result->set_gradient_edge({grad_fn, 0});
  grad_fn->add_next_edge(var->gradient_edge());
  return result;
}

//以下是我的定义：
std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> lhs, double rhs)
{
    std::shared_ptr<AddBackwardConstant> grad_fn = std::make_shared<AddBackwardConstant>();
    grad_fn->add_input_nr();
    auto result = std::make_shared<Variable>(lhs->value_ + rhs);
    result->set_gradient_edge({ grad_fn, 0 });
    grad_fn->add_next_edge(lhs->gradient_edge());
    return result;
}
std::shared_ptr<Variable> operator+(double lhs, std::shared_ptr<Variable> rhs)
{
    std::shared_ptr<AddBackwardConstant> grad_fn = std::make_shared<AddBackwardConstant>();
    grad_fn->add_input_nr();
    auto result = std::make_shared<Variable>(rhs->value_ + lhs);
    result->set_gradient_edge({ grad_fn, 0 });
    grad_fn->add_next_edge(rhs->gradient_edge());
    return result;
}
std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> lhs, double rhs)
{
    std::shared_ptr<AddBackwardConstant> grad_fn = std::make_shared<AddBackwardConstant>();
    grad_fn->add_input_nr();
    auto result = std::make_shared<Variable>(lhs->value_ - rhs);
    result->set_gradient_edge({ grad_fn, 0 });//这里暂时没有搞明白原理
    grad_fn->add_next_edge(lhs->gradient_edge());
    return result;
}
std::shared_ptr<Variable> operator-(double lhs, std::shared_ptr<Variable> rhs)
{
    std::shared_ptr<NegBackward> grad_fn = std::make_shared<NegBackward>();
    grad_fn->add_input_nr();
    auto result = std::make_shared<Variable>(lhs - rhs->value_);
    result->set_gradient_edge({ grad_fn, 0 });
    grad_fn->add_next_edge(rhs->gradient_edge());
    return result;
}
std::shared_ptr<Variable> operator*(double lhs, std::shared_ptr<Variable> rhs)
{
    std::shared_ptr<MulBackwardConstant> grad_fn = std::make_shared<MulBackwardConstant>();//MulBackward中存了值用来反向传播
    grad_fn->self_ = rhs;
    grad_fn->other_ = lhs;
    grad_fn->add_input_nr();
    auto result = std::make_shared<Variable>(lhs* rhs->value_);
    result->set_gradient_edge({ grad_fn, 0 });
    grad_fn->add_next_edge(rhs->gradient_edge());
    return result;
}
std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> lhs, double rhs)
{
    std::shared_ptr<MulBackwardConstant> grad_fn = std::make_shared<MulBackwardConstant>();
    grad_fn->self_ = lhs;
    grad_fn->other_ = rhs;
    grad_fn->add_input_nr();
    auto result = std::make_shared<Variable>(rhs * lhs->value_);
    result->set_gradient_edge({ grad_fn, 0 });
    grad_fn->add_next_edge(lhs->gradient_edge());
    return result;
}
} // namespace autograd