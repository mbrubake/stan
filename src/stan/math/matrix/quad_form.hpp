#ifndef __STAN__MATH__MATRIX__QUAD_FORM_HPP__
#define __STAN__MATH__MATRIX__QUAD_FORM_HPP__

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>

namespace stan {
  namespace math {
    template<typename DerivedA,typename DerivedB>
    inline typename
    boost::enable_if_c< boost::is_same<typename DerivedA::Scalar,double>::value &&
                        boost::is_same<typename DerivedB::Scalar,double>::value,
                        Eigen::Matrix<double,DerivedB::ColsAtCompileTime,DerivedB::ColsAtCompileTime> >::type
    quad_form(const Eigen::MatrixBase<DerivedA> &A,
              const Eigen::MatrixBase<DerivedB> &B)
    {
      validate_square(A,"quad_form");
      validate_multiplicable(A,B,"quad_form");
      return B.transpose()*A*B;
    }
    
    template<typename DerivedA,typename DerivedB>
    inline typename
    boost::enable_if_c< boost::is_same<typename DerivedA::Scalar,double>::value &&
                        boost::is_same<typename DerivedB::Scalar,double>::value,
                        Eigen::Matrix<double,DerivedB::ColsAtCompileTime,DerivedB::ColsAtCompileTime> >::type
    inv_quad_form(const Eigen::MatrixBase<DerivedA> &A,
                  const Eigen::MatrixBase<DerivedB> &B)
    {
      validate_square(A,"inv_quad_form");
      validate_multiplicable(A,B,"inv_quad_form");
      return B.transpose()*A.qr().solve(B);
    }
    
    template<typename DerivedA,typename DerivedB>
    inline typename
    boost::enable_if_c< boost::is_same<typename DerivedA::Scalar,double>::value &&
                        boost::is_same<typename DerivedB::Scalar,double>::value,
                        Eigen::Matrix<double,DerivedB::ColsAtCompileTime,DerivedB::ColsAtCompileTime> >::type
    inv_quad_form_spd(const Eigen::MatrixBase<DerivedA> &A,
                      const Eigen::MatrixBase<DerivedB> &B)
    {
      validate_square(A,"inv_quad_form");
      validate_multiplicable(A,B,"inv_quad_form");
      return B.transpose()*A.ldlt().solve(B);
    }
  }
}