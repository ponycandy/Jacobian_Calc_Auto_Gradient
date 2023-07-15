#include <iostream>
#include <autograd/autograd.h>
#include <autograd/variablematrix.h>
int main()

{
	std::shared_ptr<autograd::Variable> xvar = autograd::variable(1);
	Eigen::Matrix<std::shared_ptr<autograd::Variable>, 2, 1> mat;
	
	mat<< 4*xvar,2*xvar;

	std::vector<std::shared_ptr<autograd::Variable>> var {xvar};
	std::vector<std::shared_ptr<autograd::Variable>> value {mat(0,0),mat(1,0)};

	Eigen::MatrixXd jac;
	jac.resize(2, 1);
	GetJacobian(value, var,jac);

	std::cout << jac << std::endl;

	return 0;
}
//#include <iostream>
//#include <cmath>
//#include <complex>
//
//#include <Eigen/Dense>
//
//class MyDouble {
//public:
//    double value;
//    MyDouble() : value() {};
//    MyDouble(double val) : value(val) {};
//
//    template<typename T>
//    MyDouble& operator+=(T rhs) {
//        value = static_cast<double>(value + rhs);
//        return *this;
//    }
//
//    template<typename T>
//    MyDouble& operator-=(const T& rhs) {
//        value = static_cast<double>(value - rhs);
//        return *this;
//    }
//
//    template<typename T>
//    MyDouble& operator*=(T rhs) {
//        value = static_cast<double>(value * rhs);
//        return *this;
//    }
//
//    template<typename T>
//    MyDouble& operator/=(T rhs) {
//        value = static_cast<double>(value / rhs);
//        return *this;
//    }
//
//    MyDouble operator-() const {
//        return -value;
//    }
//
//    friend std::ostream& operator<<(std::ostream& out, const MyDouble& val) {
//        out << val.value << " m";
//        return out;
//    }
//
//    explicit operator double() {
//        return value;
//    }
//};
//
//#define OVERLOAD_OPERATOR(op,ret) ret operator op(const MyDouble &lhs, const MyDouble &rhs) { \
//        return lhs.value op rhs.value; \
//    }
//
//OVERLOAD_OPERATOR(+, MyDouble)
//OVERLOAD_OPERATOR(-, MyDouble)
//OVERLOAD_OPERATOR(*, MyDouble)
//OVERLOAD_OPERATOR(/ , MyDouble)
//
//OVERLOAD_OPERATOR(> , bool)
//OVERLOAD_OPERATOR(< , bool)
//    OVERLOAD_OPERATOR(>= , bool)
//    OVERLOAD_OPERATOR(<= , bool)
//    OVERLOAD_OPERATOR(== , bool)
//    OVERLOAD_OPERATOR(!= , bool)
//    //上面定义加减乘除
//
//    MyDouble sqrt(MyDouble val) {
//    return std::sqrt(val.value);
//}
//MyDouble abs(MyDouble val) {
//    return std::abs(val.value);
//}
//MyDouble abs2(MyDouble val) {
//    return val * val;
//}
//bool isfinite(const MyDouble&) { return true; }
//
//namespace Eigen {
//    template<> struct NumTraits<MyDouble>
//    : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
//    {
//        typedef MyDouble Real;
//        typedef MyDouble NonInteger;
//        typedef MyDouble Nested;
//        enum {
//            IsComplex = 0,
//            IsInteger = 0,
//            IsSigned = 1,
//            RequireInitialization = 0,
//            ReadCost = 1,
//            AddCost = 3,
//            MulCost = 3
//        };
//    };
//
//    template<typename BinaryOp>
//    struct ScalarBinaryOpTraits<MyDouble, double, BinaryOp> { typedef MyDouble ReturnType; };
//
//    template<typename BinaryOp>
//    struct ScalarBinaryOpTraits<double, MyDouble, BinaryOp> { typedef MyDouble ReturnType; };
//}
//
//int main() {
//    Eigen::Matrix<MyDouble, 2, 2> test;
//    test << 1, 2, 3, 4;
//
//    Eigen::Matrix<double, 2, 2> reference;
//    reference << 1, 2, 3, 4;
//
//    MyDouble a = 3;
//    a += 2;
//    a = 2 + a;
//    a = a + 2;
//    a -= 2;
//    a -= MyDouble(3);
//
//    a = a / a;
//
//    test = test * 2;
//
//    //std::complex<MyDouble> complexTest(3, 4);
//    //complexTest *= 2;
//
//    //Eigen::EigenSolver<Eigen::Matrix<MyDouble, 2, 2>> solver(test);
//    //Eigen::EigenSolver<Eigen::Matrix<double, 2, 2>> refSolver(reference);
//    //std::cout << "MyDouble:" << std::endl;
//    //std::cout << test.trace() << std::endl;
//    //std::cout << solver.eigenvalues() << std::endl;
//    //std::cout << "\nRefernce:" << std::endl;
//    //std::cout << reference.trace() << std::endl;
//    //std::cout << refSolver.eigenvalues() << std::endl;
//}