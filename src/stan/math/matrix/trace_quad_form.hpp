#ifndef __STAN__MATH__MATRIX__TRACE_QUAD_FORM_HPP__
#define __STAN__MATH__MATRIX__TRACE_QUAD_FORM_HPP__

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>

namespace stan {
  namespace math {
    /**
     * Compute trace(B^T A B).
     **/
    template<typename DerivedD, typename DerivedA,typename DerivedB>
    inline typename
    boost::enable_if_c< boost::is_same<typename DerivedA::Scalar,double>::value &&
                        boost::is_same<typename DerivedB::Scalar,double>::value,
                        double >::type
    trace_quad_form(const Eigen::MatrixBase<DerivedA> &A,
                    const Eigen::MatrixBase<DerivedB> &B)
    {
      validate_square(A,"trace_quad_form");
      validate_multiplicable(A,B,"trace_quad_form");
      return (B.transpose()*A*B).trace();
    }
    
    /**
     * Compute trace(D B^T A B).
     **/
    template<typename DerivedD, typename DerivedA,typename DerivedB>
    inline typename
    boost::enable_if_c< boost::is_same<typename DerivedD::Scalar,double>::value &&
                        boost::is_same<typename DerivedA::Scalar,double>::value &&
                        boost::is_same<typename DerivedB::Scalar,double>::value,
                        double >::type
    trace_gen_quad_form(const Eigen::MatrixBase<DerivedD> &D,
                        const Eigen::MatrixBase<DerivedA> &A,
                        const Eigen::MatrixBase<DerivedB> &B)
    {
      validate_square(A,"trace_gen_quad_form");
      validate_square(D,"trace_gen_quad_form");
      validate_multiplicable(A,B,"trace_gen_quad_form");
      validate_multiplicable(B,D,"trace_gen_quad_form");
      return (D*B.transpose()*A*B).trace();
    }
  }
}

#endif

