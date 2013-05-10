#ifndef __STAN__MATH__MATRIX__INVERSE_SPD_HPP__
#define __STAN__MATH__MATRIX__INVERSE_SPD_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_square.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the inverse of the specified symmetric, pos/neg-definite matrix.
     * @param m Specified matrix.
     * @return Inverse of the matrix.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    inverse_spd(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      validate_square(m,"inverse_spd");
      return m.ldlt().solve(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>::Identity(m.rows(),m.cols()));
    }

  }
}
#endif
