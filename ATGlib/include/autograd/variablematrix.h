#if !defined(__VARIABLEMATRIX_H)
#define __VARIABLEMATRIX_H
#include "Eigen/core"
#include "autograd/autograd.h"
typedef Eigen::Matrix<std::shared_ptr<autograd::Variable>, Eigen::Dynamic, Eigen::Dynamic> ATGTensor;
typedef std::vector<std::shared_ptr<autograd::Variable>> ATGvector;
namespace Eigen 
{
    template<> struct NumTraits<ATGTensor>
    : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
    {
        typedef ATGTensor Real;
        typedef ATGTensor NonInteger;
        typedef ATGTensor Nested;
        enum {
            IsComplex = 0,
            IsInteger = 0,
            IsSigned = 1,
            RequireInitialization = 1,
            ReadCost = 1,
            AddCost = 3,
            MulCost = 3
        };
    };
    template<typename BinaryOp>
    struct ScalarBinaryOpTraits<ATGTensor, double, BinaryOp> { typedef ATGTensor ReturnType; };

    template<typename BinaryOp>
    struct ScalarBinaryOpTraits<double, ATGTensor, BinaryOp> { typedef ATGTensor ReturnType; };
}

void GetJacobian(ATGvector& leftside, ATGvector& upside, Eigen::MatrixXd& returnmat);
void GetJacobian(ATGTensor& leftside, ATGTensor& upside, Eigen::MatrixXd& returnmat);

#endif 
