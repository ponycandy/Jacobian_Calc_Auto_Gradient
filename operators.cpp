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
//����Ϊ�ҵ���ӣ�
variable_list AddBackwardConstant::apply(variable_list&& grads) {
    auto grad = grads[0].value_;  
    //��һ����ָ��ͼ��ĿǰΪֹ�����۵ĵ���ֵ���������Ӧ���Ǳ��ڵ㵼���������ֵ
    //�����NegBackward����һ������
    variable_list grads_input{ grad };
    //�м���variable���������������,grads_input���м����Ӧ˳��û�й�ϵ��ϵͳ�����ÿ��variable���ݶ�
    // ����������Ҳ������ĸ���ǰ����ν
    //������Լ��ĵ��������ձ����Ŷ��������˳�򣬴��ڵ�һ��λ���ϵı��������Լ�
    return grads_input;
    //���Ӧ�ÿ���ͨ�ø���+����+����
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
