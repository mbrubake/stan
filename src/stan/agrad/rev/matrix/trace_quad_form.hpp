#ifndef __STAN__AGRAD__REV__MATRIX__TRACE_QUAD_FORM_HPP__
#define __STAN__AGRAD__REV__MATRIX__TRACE_QUAD_FORM_HPP__

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/math/matrix/trace_quad_form.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>

namespace stan {
  namespace agrad {
    namespace {
      template<int R,int C>
      inline 
      const Eigen::Matrix<double,R,C> &to_val_mat(const Eigen::Matrix<double,R,C> &M) {
        return M;
      }
      
      template<int R,int C>
      inline Eigen::Matrix<double,R,C> to_val_mat(const Eigen::Matrix<var,R,C> &M) {
        size_t i,j;
        Eigen::Matrix<double,R,C> Md(M.rows(),M.cols());
        for (j = 0; j < M.cols(); j++)
          for (i = 0; i < M.rows(); i++)
            Md(i,j) = M(i,j).vi_->val_;
        return Md;
      }
      
      template<typename TA,int RA,int CA,typename TB,int RB,int CB>
      class trace_quad_form_vari_alloc : public chainable_alloc {
      public:
        trace_quad_form_vari_alloc(const Eigen::Matrix<TA,RA,CA> &A,
                                   const Eigen::Matrix<TB,RB,CB> &B)
        : _A(A), _B(B)
        { }
        
        double compute() {
          return stan::math::trace_quad_form(to_val_mat(_A),
                                             to_val_mat(_B));
        }
        
        Eigen::Matrix<TA,RA,CA>  _A;
        Eigen::Matrix<TB,RB,CB>  _B;
      };
      
      template<typename TA,int RA,int CA,typename TB,int RB,int CB>
      class trace_quad_form_vari : public vari {
      protected:
        static inline void chainA(Eigen::Matrix<var,RA,CA> &A, 
                                  const Eigen::Matrix<double,RB,CB> &Bd,
                                  const double &adjC)
        {
          size_t i,j;
          Eigen::Matrix<double,RA,CA>     adjA(adjC*Bd*Bd.transpose());
          for (j = 0; j < A.cols(); j++)
            for (i = 0; i < A.rows(); i++)
              A(i,j).vi_->adj_ += adjA(i,j);
        }
        static inline void chainB(Eigen::Matrix<var,RB,CB> &B, 
                                  const Eigen::Matrix<double,RA,CA> &Ad,
                                  const Eigen::Matrix<double,RB,CB> &Bd,
                                  const double &adjC)
        {
          size_t i,j;
          Eigen::Matrix<double,RA,CA>     adjB(adjC*(Ad + Ad.transpose())*Bd);
          for (j = 0; j < B.cols(); j++)
            for (i = 0; i < B.rows(); i++)
              B(i,j).vi_->adj_ += adjB(i,j);
        }
        
        static inline void chainAB(Eigen::Matrix<var,RA,CA> &A,
                                   Eigen::Matrix<var,RB,CB> &B,
                                   const double &adjC)
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
                                   const double &adjC)
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
                                   const double &adjC)
        {
          size_t i,j;
          Eigen::Matrix<double,RA,CA> Ad(A.rows(),A.cols());

          for (j = 0; j < A.cols(); j++)
            for (i = 0; i < A.rows(); i++)
              Ad(i,j) = A(i,j).vi_->val_;
          
          chainA(A,B,adjC);
        }

      public:
        trace_quad_form_vari(trace_quad_form_vari_alloc<TA,RA,CA,TB,RB,CB> *impl)
        : vari(impl->compute()), _impl(impl) { }
        
        virtual void chain() {
          chainAB(_impl->_A,_impl->_B,adj_);
        };

        trace_quad_form_vari_alloc<TA,RA,CA,TB,RB,CB> *_impl;
      };
    }
    
    template<typename TA,int RA,int CA,typename TB,int RB,int CB>
    inline typename
    boost::enable_if_c< boost::is_same<TA,var>::value ||
                        boost::is_same<TB,var>::value,
                        var >::type
    trace_quad_form(const Eigen::Matrix<TA,RA,CA> &A,
                    const Eigen::Matrix<TB,RB,CB> &B)
    {
      stan::math::validate_square(A,"trace_quad_form");
      stan::math::validate_multiplicable(A,B,"trace_quad_form");
      
      trace_quad_form_vari_alloc<TA,RA,CA,TB,RB,CB> *baseVari = new trace_quad_form_vari_alloc<TA,RA,CA,TB,RB,CB>(A,B);
      
      return var(new trace_quad_form_vari<TA,RA,CA,TB,RB,CB>(baseVari));
    }

    namespace {
      template<typename TD,int RD,int CD,
               typename TA,int RA,int CA,
               typename TB,int RB,int CB>
      class trace_gen_quad_form_vari_alloc : public chainable_alloc {
      public:
        trace_gen_quad_form_vari_alloc(const Eigen::Matrix<TD,RD,CD> &D,
                                       const Eigen::Matrix<TA,RA,CA> &A,
                                       const Eigen::Matrix<TB,RB,CB> &B)
        : _D(D), _A(A), _B(B)
        { }
        
        double compute() {
          return stan::math::trace_gen_quad_form(to_val_mat(_D),
                                                 to_val_mat(_A),
                                                 to_val_mat(_B));
        }
        
        Eigen::Matrix<TD,RD,CD>  _D;
        Eigen::Matrix<TA,RA,CA>  _A;
        Eigen::Matrix<TB,RB,CB>  _B;
      };
      
      template<typename TD,int RD,int CD,
               typename TA,int RA,int CA,
               typename TB,int RB,int CB>
      class trace_gen_quad_form_vari : public vari {
      protected:
        static inline void computeAdjoints(const double &adj,
                                           const Eigen::Matrix<double,RD,CD> &D,
                                           const Eigen::Matrix<double,RA,CA> &A,
                                           const Eigen::Matrix<double,RB,CB> &B,
                                           Eigen::Matrix<var,RD,CD> *varD,
                                           Eigen::Matrix<var,RA,CA> *varA,
                                           Eigen::Matrix<var,RB,CB> *varB)
        {
          Eigen::Matrix<double,CA,CB> AtB;
          Eigen::Matrix<double,RA,CB> BD;
          if (varB || varA)
            BD.noalias() = B*D;
          if (varA || varD)
            AtB.noalias() = A.transpose()*B;
          
          if (varB) {
            Eigen::Matrix<double,RB,CB> adjB(A*BD + AtB*D.transpose());
            size_t i,j;
            for (j = 0; j < B.cols(); j++)
              for (i = 0; i < B.rows(); i++)
                (*varB)(i,j).vi_->adj_ += adjB(i,j);
          }
          if (varA) {
            Eigen::Matrix<double,RA,CA> adjA(B*BD.transpose());
            size_t i,j;
            for (j = 0; j < A.cols(); j++)
              for (i = 0; i < A.rows(); i++)
                (*varA)(i,j).vi_->adj_ += adjA(i,j);
          }
          if (varD) {
            Eigen::Matrix<double,RD,CD> adjD(B.transpose()*AtB);
            size_t i,j;
            for (j = 0; j < D.cols(); j++)
              for (i = 0; i < D.rows(); i++)
                (*varD)(i,j).vi_->adj_ += adjD(i,j);
          }
        }

        
      public:
        trace_gen_quad_form_vari(trace_gen_quad_form_vari_alloc<TD,RD,CD,TA,RA,CA,TB,RB,CB> *impl)
        : vari(impl->compute()), _impl(impl) { }
        
        virtual void chain() {
          computeAdjoints(adj_,
                          to_val_mat(_impl->_D),
                          to_val_mat(_impl->_A),
                          to_val_mat(_impl->_B),
                          (Eigen::Matrix<var,RD,CD>*)(boost::is_same<TD,var>::value?(&_impl->_D):NULL),
                          (Eigen::Matrix<var,RA,CA>*)(boost::is_same<TA,var>::value?(&_impl->_A):NULL),
                          (Eigen::Matrix<var,RB,CB>*)(boost::is_same<TB,var>::value?(&_impl->_B):NULL));
        }
        
        trace_gen_quad_form_vari_alloc<TD,RD,CD,TA,RA,CA,TB,RB,CB> *_impl;
      };
    }
    
    template<typename TD,int RD,int CD,
             typename TA,int RA,int CA,
             typename TB,int RB,int CB>
    inline typename
    boost::enable_if_c< boost::is_same<TD,var>::value ||
                        boost::is_same<TA,var>::value ||
                        boost::is_same<TB,var>::value,
                        var >::type
    trace_gen_quad_form(const Eigen::Matrix<TD,RD,CD> &D,
                        const Eigen::Matrix<TA,RA,CA> &A,
                        const Eigen::Matrix<TB,RB,CB> &B)
    {
      stan::math::validate_square(A,"trace_gen_quad_form");
      stan::math::validate_square(D,"trace_gen_quad_form");
      stan::math::validate_multiplicable(A,B,"trace_gen_quad_form");
      stan::math::validate_multiplicable(B,D,"trace_gen_quad_form");
      
      trace_gen_quad_form_vari_alloc<TD,RD,CD,TA,RA,CA,TB,RB,CB> *baseVari = new trace_gen_quad_form_vari_alloc<TD,RD,CD,TA,RA,CA,TB,RB,CB>(D,A,B);
      
      return var(new trace_gen_quad_form_vari<TD,RD,CD,TA,RA,CA,TB,RB,CB>(baseVari));
    }
  }
}

#endif
