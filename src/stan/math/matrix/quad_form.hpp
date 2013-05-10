#ifndef __STAN__MATH__MATRIX__QUAD_FORM_HPP__
#define __STAN__MATH__MATRIX__QUAD_FORM_HPP__

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>

namespace stan {
  namespace math {
    /**
     * Compute B^T A B
     **/
    template<typename DerivedA,typename DerivedB>
    inline typename
    boost::enable_if_c< boost::is_same<typename DerivedA::Scalar,double>::value &&
                        boost::is_same<typename DerivedB::Scalar,double>::value && DerivedB::ColsAtCompileTime != 1,
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
                        boost::is_same<typename DerivedB::Scalar,double>::value && DerivedB::ColsAtCompileTime == 1,
                        double >::type
    quad_form(const Eigen::MatrixBase<DerivedA> &A,
              const Eigen::MatrixBase<DerivedB> &B)
    {
      validate_square(A,"quad_form");
      validate_multiplicable(A,B,"quad_form");
      return B.dot(A*B);
    }
    
    /**
     * Compute B^T A^-1 B
     **/
    template<typename DerivedA,typename DerivedB>
    inline typename
    boost::enable_if_c< boost::is_same<typename DerivedA::Scalar,double>::value &&
                        boost::is_same<typename DerivedB::Scalar,double>::value && DerivedB::ColsAtCompileTime != 1,
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
                        boost::is_same<typename DerivedB::Scalar,double>::value && DerivedB::ColsAtCompileTime == 1,
                        double >::type
    inv_quad_form(const Eigen::MatrixBase<DerivedA> &A,
                  const Eigen::MatrixBase<DerivedB> &B)
    {
      validate_square(A,"inv_quad_form");
      validate_multiplicable(A,B,"inv_quad_form");
      return B.dot(A.qr().solve(B));
    }
    
    /**
     * Compute B^T A^-1 B where A is a symmetric, positive definite matrix.
     **/
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
    
    /**
     * Compute trace(B^T A^-1 B).
     **/
    template<typename DerivedD, typename DerivedA,typename DerivedB>
    inline typename
    boost::enable_if_c< boost::is_same<typename DerivedA::Scalar,double>::value &&
                        boost::is_same<typename DerivedB::Scalar,double>::value,
                        double >::type
    trace_inv_quad_form(const Eigen::MatrixBase<DerivedA> &A,
                        const Eigen::MatrixBase<DerivedB> &B)
    {
      validate_square(A,"trace_inv_quad_form");
      validate_multiplicable(A,B,"trace_inv_quad_form");
      return (B.transpose()*A.qr().solve(B)).trace();
    }
    
    /**
     * Compute trace(D^-1 B^T A^-1 B).
     **/
    template<typename DerivedD, typename DerivedA,typename DerivedB>
    inline typename
    boost::enable_if_c< boost::is_same<typename DerivedD::Scalar,double>::value &&
                        boost::is_same<typename DerivedA::Scalar,double>::value &&
                        boost::is_same<typename DerivedB::Scalar,double>::value,
                        double >::type
    trace_inv_quad_form(const Eigen::MatrixBase<DerivedB> &D,
                        const Eigen::MatrixBase<DerivedA> &A,
                        const Eigen::MatrixBase<DerivedB> &B)
    {
      validate_square(A,"trace_inv_quad_form");
      validate_square(D,"trace_inv_quad_form");
      validate_multiplicable(A,B,"trace_inv_quad_form");
      validate_multiplicable(B,D,"trace_inv_quad_form");
      return D.qr().solve(B.transpose()*A.qr().solve(B)).trace();
    }
  }
}

#endif

