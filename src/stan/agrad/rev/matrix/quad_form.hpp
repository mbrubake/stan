#ifndef __STAN__AGRAD__REV__MATRIX__QUAD_FORM_HPP__
#define __STAN__AGRAD__REV__MATRIX__QUAD_FORM_HPP__

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>

namespace stan {
  namespace agrad {
    namespace {
      template<typename TA,int RA,int CA,typename TB,int RB,int CB>
      class quad_form_vari_alloc : public chainable_alloc {
      protected:
        static inline void computeC(const Eigen::Matrix<var,RA,CA> &A,
                                    const Eigen::Matrix<var,RB,CB> &B,
                                    Eigen::Matrix<var,CB,CB> &C) {
          size_t i,j;
          Eigen::Matrix<double,RA,CA> Ad(A.rows(),A.cols());
          for (j = 0; j < A.cols(); j++)
            for (i = 0; i < A.rows(); i++)
              Ad(i,j) = A(i,j).vi_->val_;

          Eigen::Matrix<double,RB,CB> Bd(B.rows(),B.cols());
          for (j = 0; j < B.cols(); j++)
            for (i = 0; i < B.rows(); i++)
              Bd(i,j) = B(i,j).vi_->val_;
          
          Eigen::Matrix<double,CB,CB> Cd(Bd.transpose()*Ad*Bd);
          for (j = 0; j < C.cols(); j++)
            for (i = 0; i < C.rows(); i++)
              C(i,j) = var(new vari(Cd(i,j),false));
        }
        static inline void computeC(const Eigen::Matrix<double,RA,CA> &Ad,
                                    const Eigen::Matrix<var,RB,CB> &B,
                                    Eigen::Matrix<var,CB,CB> &C) {
          size_t i,j;
          Eigen::Matrix<double,RB,CB> Bd(B.rows(),B.cols());
          for (j = 0; j < B.cols(); j++)
            for (i = 0; i < B.rows(); i++)
              Bd(i,j) = B(i,j).vi_->val_;
          
          Eigen::Matrix<double,CB,CB> Cd(Bd.transpose()*Ad*Bd);
          for (j = 0; j < C.cols(); j++)
            for (i = 0; i < C.rows(); i++)
              C(i,j) = var(new vari(Cd(i,j),false));
        }
        static inline void computeC(const Eigen::Matrix<var,RA,CA> &A,
                                    const Eigen::Matrix<double,RB,CB> &Bd,
                                    Eigen::Matrix<var,CB,CB> &C) {
          size_t i,j;
          Eigen::Matrix<double,RA,CA> Ad(A.rows(),A.cols());
          for (j = 0; j < A.cols(); j++)
            for (i = 0; i < A.rows(); i++)
              Ad(i,j) = A(i,j).vi_->val_;

          Eigen::Matrix<double,CB,CB> Cd(Bd.transpose()*Ad*Bd);
          for (j = 0; j < C.cols(); j++)
            for (i = 0; i < C.rows(); i++)
              C(i,j) = var(new vari(Cd(i,j),false));
        }
        
      public:
        quad_form_vari_alloc(const Eigen::Matrix<TA,RA,CA> &A,
                             const Eigen::Matrix<TB,RB,CB> &B)
        : _A(A), _B(B), _C(_B.cols(),_B.cols())
        {
          computeC(_A,_B,_C);
        }
        
        Eigen::Matrix<TA,RA,CA>  _A;
        Eigen::Matrix<TB,RB,CB>  _B;
        Eigen::Matrix<var,CB,CB> _C;
      };
      
      template<typename TA,int RA,int CA,typename TB,int RB,int CB>
      class quad_form_vari : public vari {
      protected:
        static inline void chainA(Eigen::Matrix<var,RA,CA> &A, 
                                  const Eigen::Matrix<double,RB,CB> &Bd,
                                  const Eigen::Matrix<double,CB,CB> &adjC)
        {
          size_t i,j;
          Eigen::Matrix<double,RA,CA>     adjA(Bd*adjC*Bd.transpose());
          for (j = 0; j < A.cols(); j++)
            for (i = 0; i < A.rows(); i++)
              A(i,j).vi_->adj_ += adjA(i,j);
        }
        static inline void chainB(Eigen::Matrix<var,RB,CB> &B, 
                                  const Eigen::Matrix<double,RA,CA> &Ad,
                                  const Eigen::Matrix<double,RB,CB> &Bd,
                                  const Eigen::Matrix<double,CB,CB> &adjC)
        {
          size_t i,j;
          Eigen::Matrix<double,RA,CA>     adjB(Ad*Bd*adjC.transpose() + Ad.transpose()*Bd*adjC);
          for (j = 0; j < B.cols(); j++)
            for (i = 0; i < B.rows(); i++)
              B(i,j).vi_->adj_ += adjB(i,j);
        }
        
        static inline void chainAB(Eigen::Matrix<var,RA,CA> &A,
                                   Eigen::Matrix<var,RB,CB> &B,
                                   const Eigen::Matrix<double,CB,CB> &adjC)
        {
          size_t i,j;
          Eigen::Matrix<double,RA,CA> Ad(A.rows(),A.cols());
          Eigen::Matrix<double,RB,CB> Bd(B.rows(),B.cols());
          
          for (j = 0; j < B.cols(); j++)
            for (i = 0; i < B.rows(); i++)
              Bd(i,j) = B(i,j).vi_->val_;
          
          for (j = 0; j < A.cols(); j++)
            for (i = 0; i < A.rows(); i++)
              Ad(i,j) = A(i,j).vi_->val_;
          
          chainA(A,Bd,adjC);
          chainB(B,Ad,Bd,adjC);
        }
        
        static inline void chainAB(Eigen::Matrix<double,RA,CA> &A,
                                   Eigen::Matrix<var,RB,CB> &B,
                                   const Eigen::Matrix<double,CB,CB> &adjC)
        {
          size_t i,j;
          Eigen::Matrix<double,RB,CB> Bd(B.rows(),B.cols());
          
          for (j = 0; j < B.cols(); j++)
            for (i = 0; i < B.rows(); i++)
              Bd(i,j) = B(i,j).vi_->val_;
          
          chainB(B,A,Bd,adjC);
        }
        
        static inline void chainAB(Eigen::Matrix<var,RA,CA> &A,
                                   Eigen::Matrix<double,RB,CB> &B,
                                   const Eigen::Matrix<double,CB,CB> &adjC)
        {
          size_t i,j;
          Eigen::Matrix<double,RA,CA> Ad(A.rows(),A.cols());

          for (j = 0; j < A.cols(); j++)
            for (i = 0; i < A.rows(); i++)
              Ad(i,j) = A(i,j).vi_->val_;
          
          chainA(A,B,adjC);
        }

      public:
        quad_form_vari(const Eigen::Matrix<TA,RA,CA> &A,
                       const Eigen::Matrix<TB,RB,CB> &B)
        : vari(0.0) {
          _impl = new quad_form_vari_alloc<TA,RA,CA,TB,RB,CB>(A,B);
        }
        
        virtual void chain() {
          size_t i,j;
          Eigen::Matrix<double,CB,CB> adjC(_impl->_C.rows(),_impl->_C.cols());
          
          for (j = 0; j < _impl->_C.cols(); j++)
            for (i = 0; i < _impl->_C.rows(); i++)
              adjC(i,j) = _impl->_C(i,j).vi_->adj_;
          
          chainAB(_impl->_A,_impl->_B,adjC);
        };

        quad_form_vari_alloc<TA,RA,CA,TB,RB,CB> *_impl;
      };
    }
    
    template<typename TA,int RA,int CA,typename TB,int RB,int CB>
    inline typename
    boost::enable_if_c< (boost::is_same<TA,var>::value ||
                         boost::is_same<TB,var>::value) && CB != 1,
                        Eigen::Matrix<var,CB,CB> >::type
    quad_form(const Eigen::Matrix<TA,RA,CA> &A,
              const Eigen::Matrix<TB,RB,CB> &B)
    {
      stan::math::validate_square(A,"quad_form");
      stan::math::validate_multiplicable(A,B,"quad_form");
      
      quad_form_vari<TA,RA,CA,TB,RB,CB> *baseVari = new quad_form_vari<TA,RA,CA,TB,RB,CB>(A,B);
      
      return baseVari->_impl->_C;
    }
    template<typename TA,int RA,int CA,typename TB,int RB,int CB>
    inline typename
    boost::enable_if_c< (boost::is_same<TA,var>::value ||
                         boost::is_same<TB,var>::value) && CB == 1,
                        var >::type
    quad_form(const Eigen::Matrix<TA,RA,CA> &A,
              const Eigen::Matrix<TB,RB,CB> &B)
    {
      stan::math::validate_square(A,"quad_form");
      stan::math::validate_multiplicable(A,B,"quad_form");
      
      quad_form_vari<TA,RA,CA,TB,RB,CB> *baseVari = new quad_form_vari<TA,RA,CA,TB,RB,CB>(A,B);
      
      return baseVari->_impl->_C(0,0);
    }
  }
}

#endif
