#include "autograd/operators.h"
#include <cmath>

namespace autograd {

variable_list AccumulateGrad::apply(variable_list &&grads) {
  auto grad = grads[0].value_;
  if (auto ptr = variable_.lock()) {
    ptr->grad_ += grad;
  }
  return variable_list();
}

variable_list AddBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_;
  variable_list grads_input{grad, grad};
  return grads_input;
}

variable_list MulBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_;
  variable_list grads_input{other_->value_ * grad, self_->value_ * grad};
  return grads_input;
}

variable_list DivBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_;
  variable_list grads_input{1.0f / other_->value_ * grad,
                            -self_->value_ / (other_->value_ * other_->value_) *
                                grad};
  return grads_input;
}

variable_list SubBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_;
  variable_list grads_input{grad, -grad};
  return grads_input;
}

variable_list PowBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_;
  auto xvalue = self_->value_;
  auto yvalue = other_->value_;
  variable_list grads_input{grad * yvalue * std::pow(xvalue, yvalue - 1),
                            grad * std::pow(xvalue, yvalue) * std::log(xvalue)};
  return grads_input;
}

variable_list LogBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_;
  auto value = self_->value_;
  variable_list grads_input{grad / value};
  return grads_input;
}

variable_list ReLUBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_;
  auto value = self_->value_;
  float grad_value = value >= 0 ? grad : 0.0f;
  variable_list grads_input{grad_value};
  return grads_input;
}

variable_list NegBackward::apply(variable_list &&grads) {
  auto grad = grads[0].value_;
  variable_list grads_input{-grad};
  return grads_input;
}
//以下为我的添加：
variable_list AddBackwardConstant::apply(variable_list&& grads) {
    auto grad = grads[0].value_;  
    //这一项是指求导图到目前为止所积累的导数值，所以输出应该是本节点导数乘以这个值
    //上面的NegBackward就是一个例子
    variable_list grads_input{ grad };
    //有几项variable变量参与这个运算,grads_input就有几项，对应顺序没有关系，系统会遍历每个variable的梯度
    // 左操作数和右操作数哪个在前无所谓
    //定义对自己的导数：按照本符号定义的输入顺序，处在第一个位置上的变量就是自己
    return grads_input;
    //这个应该可以通用给左+和右+常数
}
variable_list MulBackwardConstant::apply(variable_list&& grads) {
    auto grad = grads[0].value_;
    variable_list grads_input{ other_* grad};
    return grads_input;
}
variable_list DivBackwardConstant::apply(variable_list&& grads) {
    auto grad = grads[0].value_;
    variable_list grads_input{ -other_ /(self_->value_* self_->value_)* grad };
    return grads_input;
}
variable_list DivBackwardConstant_R::apply(variable_list&& grads) {
    auto grad = grads[0].value_;
    variable_list grads_input{ 1/other_ * grad };
    return grads_input;
}
variable_list PowBackwardConstant::apply(variable_list&& grads) {
    auto grad = grads[0].value_;
    variable_list grads_input{ grad *  std::pow(other_,self_->value_)*std::log(other_)};
    return grads_input;
}

variable_list PowBackwardConstant_R::apply(variable_list&& grads) {
    auto grad = grads[0].value_;
    variable_list grads_input{ grad * other_* std::pow(self_->value_,other_-1) };
    return grads_input;
}

} // namespace autograd
