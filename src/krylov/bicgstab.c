/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * BiCGSTAB bicgstab
 *
 *****************************************************************************/

#include "krylov.h"
#include "_nalu_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABFunctionsCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_BiCGSTABFunctions *
nalu_hypre_BiCGSTABFunctionsCreate(
   void *     (*CreateVector)  ( void *vvector ),
   NALU_HYPRE_Int  (*DestroyVector) ( void *vvector ),
   void *     (*MatvecCreate)  ( void *A, void *x ),
   NALU_HYPRE_Int  (*Matvec)        ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A,
                                 void *x, NALU_HYPRE_Complex beta, void *y ),
   NALU_HYPRE_Int  (*MatvecDestroy) ( void *matvec_data ),
   NALU_HYPRE_Real (*InnerProd)     ( void *x, void *y ),
   NALU_HYPRE_Int  (*CopyVector)    ( void *x, void *y ),
   NALU_HYPRE_Int  (*ClearVector)   ( void *x ),
   NALU_HYPRE_Int  (*ScaleVector)   ( NALU_HYPRE_Complex alpha, void *x ),
   NALU_HYPRE_Int  (*Axpy)          ( NALU_HYPRE_Complex alpha, void *x, void *y ),
   NALU_HYPRE_Int  (*CommInfo)      ( void *A, NALU_HYPRE_Int *my_id,
                                 NALU_HYPRE_Int *num_procs ),
   NALU_HYPRE_Int  (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   NALU_HYPRE_Int  (*Precond)       ( void *vdata, void *A, void *b, void *x )
)
{
   nalu_hypre_BiCGSTABFunctions * bicgstab_functions;
   bicgstab_functions = (nalu_hypre_BiCGSTABFunctions *)
                        nalu_hypre_CTAlloc( nalu_hypre_BiCGSTABFunctions,  1, NALU_HYPRE_MEMORY_HOST);

   bicgstab_functions->CreateVector = CreateVector;
   bicgstab_functions->DestroyVector = DestroyVector;
   bicgstab_functions->MatvecCreate = MatvecCreate;
   bicgstab_functions->Matvec = Matvec;
   bicgstab_functions->MatvecDestroy = MatvecDestroy;
   bicgstab_functions->InnerProd = InnerProd;
   bicgstab_functions->CopyVector = CopyVector;
   bicgstab_functions->ClearVector = ClearVector;
   bicgstab_functions->ScaleVector = ScaleVector;
   bicgstab_functions->Axpy = Axpy;
   bicgstab_functions->CommInfo = CommInfo;
   bicgstab_functions->precond_setup = PrecondSetup;
   bicgstab_functions->precond = Precond;

   return bicgstab_functions;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_BiCGSTABCreate( nalu_hypre_BiCGSTABFunctions * bicgstab_functions )
{
   nalu_hypre_BiCGSTABData *bicgstab_data;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   bicgstab_data = nalu_hypre_CTAlloc( nalu_hypre_BiCGSTABData,  1, NALU_HYPRE_MEMORY_HOST);
   bicgstab_data->functions = bicgstab_functions;

   /* set defaults */
   (bicgstab_data -> tol)            = 1.0e-06;
   (bicgstab_data -> min_iter)       = 0;
   (bicgstab_data -> max_iter)       = 1000;
   (bicgstab_data -> stop_crit)      = 0; /* rel. residual norm */
   (bicgstab_data -> a_tol)          = 0.0;
   (bicgstab_data -> precond_data)   = NULL;
   (bicgstab_data -> logging)        = 0;
   (bicgstab_data -> print_level)    = 0;
   (bicgstab_data -> hybrid)         = 0;
   (bicgstab_data -> p)              = NULL;
   (bicgstab_data -> q)              = NULL;
   (bicgstab_data -> r)              = NULL;
   (bicgstab_data -> r0)             = NULL;
   (bicgstab_data -> s)              = NULL;
   (bicgstab_data -> v)             = NULL;
   (bicgstab_data -> matvec_data)    = NULL;
   (bicgstab_data -> norms)          = NULL;
   (bicgstab_data -> log_file_name)  = NULL;

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return (void *) bicgstab_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABDestroy( void *bicgstab_vdata )
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData *)bicgstab_vdata;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   if (bicgstab_data)
   {
      nalu_hypre_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;
      if ( (bicgstab_data -> norms) != NULL )
      {
         nalu_hypre_TFree(bicgstab_data -> norms, NALU_HYPRE_MEMORY_HOST);
      }

      (*(bicgstab_functions->MatvecDestroy))(bicgstab_data -> matvec_data);

      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> r);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> r0);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> s);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> v);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> p);
      (*(bicgstab_functions->DestroyVector))(bicgstab_data -> q);

      nalu_hypre_TFree(bicgstab_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(bicgstab_functions, NALU_HYPRE_MEMORY_HOST);
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return (nalu_hypre_error_flag);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABSetup( void *bicgstab_vdata,
                     void *A,
                     void *b,
                     void *x         )
{
   nalu_hypre_BiCGSTABData      *bicgstab_data      = (nalu_hypre_BiCGSTABData *)bicgstab_vdata;
   nalu_hypre_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;

   NALU_HYPRE_Int            max_iter         = (bicgstab_data -> max_iter);
   NALU_HYPRE_Int          (*precond_setup)(void*, void*, void*,
                                       void*) = (bicgstab_functions -> precond_setup);
   void          *precond_data     = (bicgstab_data -> precond_data);

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   (bicgstab_data -> A) = A;

   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/

   if ((bicgstab_data -> p) == NULL)
   {
      (bicgstab_data -> p) = (*(bicgstab_functions->CreateVector))(b);
   }
   if ((bicgstab_data -> q) == NULL)
   {
      (bicgstab_data -> q) = (*(bicgstab_functions->CreateVector))(b);
   }
   if ((bicgstab_data -> r) == NULL)
   {
      (bicgstab_data -> r) = (*(bicgstab_functions->CreateVector))(b);
   }
   if ((bicgstab_data -> r0) == NULL)
   {
      (bicgstab_data -> r0) = (*(bicgstab_functions->CreateVector))(b);
   }
   if ((bicgstab_data -> s) == NULL)
   {
      (bicgstab_data -> s) = (*(bicgstab_functions->CreateVector))(b);
   }
   if ((bicgstab_data -> v) == NULL)
   {
      (bicgstab_data -> v) = (*(bicgstab_functions->CreateVector))(b);
   }

   if ((bicgstab_data -> matvec_data) == NULL)
      (bicgstab_data -> matvec_data) =
         (*(bicgstab_functions->MatvecCreate))(A, x);

   precond_setup(precond_data, A, b, x);

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((bicgstab_data->logging) > 0 || (bicgstab_data->print_level) > 0)
   {
      if ((bicgstab_data -> norms) != NULL)
      {
         nalu_hypre_TFree (bicgstab_data -> norms, NALU_HYPRE_MEMORY_HOST);
      }
      (bicgstab_data -> norms) = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  max_iter + 1, NALU_HYPRE_MEMORY_HOST);
   }
   if ((bicgstab_data -> print_level) > 0)
   {
      if ((bicgstab_data -> log_file_name) == NULL)
      {
         (bicgstab_data -> log_file_name) = (char*)"bicgstab.out.log";
      }
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABSolve
 *-------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABSolve(void  *bicgstab_vdata,
                    void  *A,
                    void  *b,
                    void  *x)
{
   nalu_hypre_BiCGSTABData      *bicgstab_data      = (nalu_hypre_BiCGSTABData*)bicgstab_vdata;
   nalu_hypre_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;

   NALU_HYPRE_Int               min_iter     = (bicgstab_data -> min_iter);
   NALU_HYPRE_Int           max_iter     = (bicgstab_data -> max_iter);
   NALU_HYPRE_Int           stop_crit    = (bicgstab_data -> stop_crit);
   NALU_HYPRE_Int           hybrid    = (bicgstab_data -> hybrid);
   NALU_HYPRE_Real       r_tol     = (bicgstab_data -> tol);
   NALU_HYPRE_Real       cf_tol       = (bicgstab_data -> cf_tol);
   void             *matvec_data  = (bicgstab_data -> matvec_data);
   NALU_HYPRE_Real        a_tol        = (bicgstab_data -> a_tol);



   void             *r            = (bicgstab_data -> r);
   void             *r0           = (bicgstab_data -> r0);
   void             *s            = (bicgstab_data -> s);
   void             *v           = (bicgstab_data -> v);
   void             *p            = (bicgstab_data -> p);
   void             *q            = (bicgstab_data -> q);

   NALU_HYPRE_Int              (*precond)(void*, void*, void*, void*)   = (bicgstab_functions -> precond);
   NALU_HYPRE_Int               *precond_data = (NALU_HYPRE_Int*)(bicgstab_data -> precond_data);

   /* logging variables */
   NALU_HYPRE_Int             logging        = (bicgstab_data -> logging);
   NALU_HYPRE_Int             print_level    = (bicgstab_data -> print_level);
   NALU_HYPRE_Real     *norms          = (bicgstab_data -> norms);
   /*   char           *log_file_name  = (bicgstab_data -> log_file_name);
     FILE           *fp; */

   NALU_HYPRE_Int        iter;
   NALU_HYPRE_Int        my_id, num_procs;
   NALU_HYPRE_Real alpha, beta, gamma, epsilon, temp, res, r_norm, b_norm;
   NALU_HYPRE_Real epsmac = NALU_HYPRE_REAL_MIN;
   NALU_HYPRE_Real ieee_check = 0.;
   NALU_HYPRE_Real cf_ave_0 = 0.0;
   NALU_HYPRE_Real cf_ave_1 = 0.0;
   NALU_HYPRE_Real weight;
   NALU_HYPRE_Real r_norm_0;
   NALU_HYPRE_Real den_norm;
   NALU_HYPRE_Real gamma_numer;
   NALU_HYPRE_Real gamma_denom;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   (bicgstab_data -> converged) = 0;

   (*(bicgstab_functions->CommInfo))(A, &my_id, &num_procs);
   if (logging > 0 || print_level > 0)
   {
      norms          = (bicgstab_data -> norms);
      /* log_file_name  = (bicgstab_data -> log_file_name);
         fp = fopen(log_file_name,"w"); */
   }

   /* initialize work arrays */
   (*(bicgstab_functions->CopyVector))(b, r0);

   /* compute initial residual */

   (*(bicgstab_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r0);
   (*(bicgstab_functions->CopyVector))(r0, r);
   (*(bicgstab_functions->CopyVector))(r0, p);

   b_norm = nalu_hypre_sqrt((*(bicgstab_functions->InnerProd))(b, b));

   /* Since it does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (b_norm != 0.) { ieee_check = b_norm / b_norm; } /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0 || print_level > 0)
      {
         nalu_hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
         nalu_hypre_printf("ERROR -- nalu_hypre_BiCGSTABSolve: INFs and/or NaNs detected in input.\n");
         nalu_hypre_printf("User probably placed non-numerics in supplied b.\n");
         nalu_hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         nalu_hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
      NALU_HYPRE_ANNOTATE_FUNC_END;

      return nalu_hypre_error_flag;
   }

   res = (*(bicgstab_functions->InnerProd))(r0, r0);
   r_norm = nalu_hypre_sqrt(res);
   r_norm_0 = r_norm;

   /* Since it does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (r_norm != 0.) { ieee_check = r_norm / r_norm; } /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (logging > 0 || print_level > 0)
      {
         nalu_hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
         nalu_hypre_printf("ERROR -- nalu_hypre_BiCGSTABSolve: INFs and/or NaNs detected in input.\n");
         nalu_hypre_printf("User probably placed non-numerics in supplied A or x_0.\n");
         nalu_hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         nalu_hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
      }

      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
      NALU_HYPRE_ANNOTATE_FUNC_END;

      return nalu_hypre_error_flag;
   }

   if (logging > 0 || print_level > 0)
   {
      norms[0] = r_norm;
      if (print_level > 0 && my_id == 0)
      {
         nalu_hypre_printf("L2 norm of b: %e\n", b_norm);
         if (b_norm == 0.0)
         {
            nalu_hypre_printf("Rel_resid_norm actually contains the residual norm\n");
         }
         nalu_hypre_printf("Initial L2 norm of residual: %e\n", r_norm);
      }
   }
   iter = 0;

   if (b_norm > 0.0)
   {
      /* convergence criterion |r_i| <= r_tol*|b| if |b| > 0 */
      den_norm = b_norm;
   }
   else
   {
      /* convergence criterion |r_i| <= r_tol*|r0| if |b| = 0 */
      den_norm = r_norm;
   };

   /* convergence criterion |r_i| <= r_tol/a_tol , absolute residual norm*/
   if (stop_crit)
   {
      if (a_tol == 0.0) /* this is for backwards compatibility
                           (accomodating setting stop_crit to 1, but not setting a_tol) -
                           eventually we will get rid of the stop_crit flag as with GMRES */
      {
         epsilon = r_tol;
      }
      else
      {
         epsilon = a_tol;   /* this means new interface fcn called */
      }

   }
   else /* default convergence test (stop_crit = 0)*/
   {

      /* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
      den_norm = |r_0| or |b|
      note: default for a_tol is 0.0, so relative residual criteria is used unless
            user also specifies a_tol or sets r_tol = 0.0, which means absolute
            tol only is checked  */

      epsilon = nalu_hypre_max(a_tol, r_tol * den_norm);

   }


   if (print_level > 0 && my_id == 0)
   {
      if (b_norm > 0.0)
      {
         nalu_hypre_printf("=============================================\n\n");
         nalu_hypre_printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
         nalu_hypre_printf("-----    ------------    ---------- ------------\n");
      }
      else
      {
         nalu_hypre_printf("=============================================\n\n");
         nalu_hypre_printf("Iters     resid.norm     conv.rate\n");
         nalu_hypre_printf("-----    ------------    ----------\n");

      }
   }

   (bicgstab_data -> num_iterations) = iter;
   if (b_norm > 0.0)
   {
      (bicgstab_data -> rel_residual_norm) = r_norm / b_norm;
   }
   /* check for convergence before starting */
   if (r_norm == 0.0)
   {
      NALU_HYPRE_ANNOTATE_FUNC_END;
      return nalu_hypre_error_flag;
   }
   else if (r_norm <= epsilon && iter >= min_iter)
   {
      if (print_level > 0 && my_id == 0)
      {
         nalu_hypre_printf("\n\n");
         nalu_hypre_printf("Tolerance and min_iter requirements satisfied by initial data.\n");
         nalu_hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
      }
      (bicgstab_data -> converged) = 1;
      NALU_HYPRE_ANNOTATE_FUNC_END;

      return nalu_hypre_error_flag;
   }
   /* Start BiCGStab iterations */
   while (iter < max_iter)
   {
      iter++;

      (*(bicgstab_functions->ClearVector))(v);
      precond(precond_data, A, p, v);
      (*(bicgstab_functions->Matvec))(matvec_data, 1.0, A, v, 0.0, q);
      temp = (*(bicgstab_functions->InnerProd))(r0, q);
      if (nalu_hypre_abs(temp) >= epsmac)
      {
         alpha = res / temp;
      }
      else
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "BiCGSTAB broke down!! divide by near zero\n");
         NALU_HYPRE_ANNOTATE_FUNC_END;

         return nalu_hypre_error_flag;
      }
      (*(bicgstab_functions->Axpy))(alpha, v, x);
      (*(bicgstab_functions->Axpy))(-alpha, q, r);
      (*(bicgstab_functions->ClearVector))(v);
      precond(precond_data, A, r, v);
      (*(bicgstab_functions->Matvec))(matvec_data, 1.0, A, v, 0.0, s);
      /* Handle case when gamma = 0.0/0.0 as 0.0 and not NAN */
      gamma_numer = (*(bicgstab_functions->InnerProd))(r, s);
      gamma_denom = (*(bicgstab_functions->InnerProd))(s, s);
      if ((gamma_numer == 0.0) && (gamma_denom == 0.0))
      {
         gamma = 0.0;
      }
      else
      {
         gamma = gamma_numer / gamma_denom;
      }
      (*(bicgstab_functions->Axpy))(gamma, v, x);
      (*(bicgstab_functions->Axpy))(-gamma, s, r);
      /* residual is now updated, must immediately check for convergence */
      r_norm = nalu_hypre_sqrt((*(bicgstab_functions->InnerProd))(r, r));
      if (logging > 0 || print_level > 0)
      {
         norms[iter] = r_norm;
      }
      if (print_level > 0 && my_id == 0)
      {
         if (b_norm > 0.0)
            nalu_hypre_printf("% 5d    %e    %f   %e\n", iter, norms[iter],
                         norms[iter] / norms[iter - 1], norms[iter] / b_norm);
         else
            nalu_hypre_printf("% 5d    %e    %f\n", iter, norms[iter],
                         norms[iter] / norms[iter - 1]);
      }
      /* check for convergence, evaluate actual residual */
      if (r_norm <= epsilon && iter >= min_iter)
      {
         (*(bicgstab_functions->CopyVector))(b, r);
         (*(bicgstab_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);
         r_norm = nalu_hypre_sqrt((*(bicgstab_functions->InnerProd))(r, r));
         if (r_norm <= epsilon)
         {
            if (print_level > 0 && my_id == 0)
            {
               nalu_hypre_printf("\n\n");
               nalu_hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
            }
            (bicgstab_data -> converged) = 1;
            break;
         }
      }
      /*--------------------------------------------------------------------
       * Optional test to see if adequate progress is being made.
       * The average convergence factor is recorded and compared
       * against the tolerance 'cf_tol'. The weighting factor is
       * intended to pay more attention to the test when an accurate
       * estimate for average convergence factor is available.
       *--------------------------------------------------------------------*/
      if (cf_tol > 0.0)
      {
         cf_ave_0 = cf_ave_1;
         cf_ave_1 = nalu_hypre_pow( r_norm / r_norm_0, 1.0 / (2.0 * iter));

         weight   = nalu_hypre_abs(cf_ave_1 - cf_ave_0);
         weight   = weight / nalu_hypre_max(cf_ave_1, cf_ave_0);
         weight   = 1.0 - weight;
         if (weight * cf_ave_1 > cf_tol) { break; }
      }

      if (nalu_hypre_abs(res) >= epsmac)
      {
         beta = 1.0 / res;
      }
      else
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "BiCGSTAB broke down!! res=0 \n");
         NALU_HYPRE_ANNOTATE_FUNC_END;

         return nalu_hypre_error_flag;
      }
      res = (*(bicgstab_functions->InnerProd))(r0, r);
      beta *= res;
      (*(bicgstab_functions->Axpy))(-gamma, q, p);
      if (nalu_hypre_abs(gamma) >= epsmac)
      {
         (*(bicgstab_functions->ScaleVector))((beta * alpha / gamma), p);
      }
      else
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "BiCGSTAB broke down!! gamma=0 \n");
         NALU_HYPRE_ANNOTATE_FUNC_END;

         return nalu_hypre_error_flag;
      }
      (*(bicgstab_functions->Axpy))(1.0, r, p);
   } /* end while loop */

   (bicgstab_data -> num_iterations) = iter;
   if (b_norm > 0.0)
   {
      (bicgstab_data -> rel_residual_norm) = r_norm / b_norm;
   }
   if (b_norm == 0.0)
   {
      (bicgstab_data -> rel_residual_norm) = r_norm;
   }

   if (iter >= max_iter && r_norm > epsilon && epsilon > 0 && hybrid != -1) { nalu_hypre_error(NALU_HYPRE_ERROR_CONV); }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABSetTol( void   *bicgstab_vdata,
                      NALU_HYPRE_Real  tol       )
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;

   (bicgstab_data -> tol) = tol;

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABSetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABSetAbsoluteTol( void   *bicgstab_vdata,
                              NALU_HYPRE_Real  a_tol       )
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;

   (bicgstab_data -> a_tol) = a_tol;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABSetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABSetConvergenceFactorTol( void   *bicgstab_vdata,
                                       NALU_HYPRE_Real  cf_tol       )
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;

   (bicgstab_data -> cf_tol) = cf_tol;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABSetMinIter( void *bicgstab_vdata,
                          NALU_HYPRE_Int   min_iter  )
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;

   (bicgstab_data -> min_iter) = min_iter;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABSetMaxIter( void *bicgstab_vdata,
                          NALU_HYPRE_Int   max_iter  )
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;

   (bicgstab_data -> max_iter) = max_iter;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABSetStopCrit( void   *bicgstab_vdata,
                           NALU_HYPRE_Int  stop_crit       )
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;

   (bicgstab_data -> stop_crit) = stop_crit;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABSetPrecond( void  *bicgstab_vdata,
                          NALU_HYPRE_Int  (*precond)(void*, void*, void*, void*),
                          NALU_HYPRE_Int  (*precond_setup)(void*, void*, void*, void*),
                          void  *precond_data )
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;
   nalu_hypre_BiCGSTABFunctions *bicgstab_functions = bicgstab_data->functions;


   (bicgstab_functions -> precond)        = precond;
   (bicgstab_functions -> precond_setup)  = precond_setup;
   (bicgstab_data -> precond_data)   = precond_data;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABGetPrecond( void         *bicgstab_vdata,
                          NALU_HYPRE_Solver *precond_data_ptr )
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;

   *precond_data_ptr = (NALU_HYPRE_Solver)(bicgstab_data -> precond_data);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABSetLogging( void *bicgstab_vdata,
                          NALU_HYPRE_Int   logging)
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;

   (bicgstab_data -> logging) = logging;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BiCGSTABSetHybrid( void *bicgstab_vdata,
                         NALU_HYPRE_Int   logging)
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;

   (bicgstab_data -> hybrid) = logging;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABSetPrintLevel( void *bicgstab_vdata,
                             NALU_HYPRE_Int   print_level)
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;

   (bicgstab_data -> print_level) = print_level;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABGetConverged
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABGetConverged( void *bicgstab_vdata,
                            NALU_HYPRE_Int  *converged )
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;

   *converged = (bicgstab_data -> converged);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABGetNumIterations( void *bicgstab_vdata,
                                NALU_HYPRE_Int  *num_iterations )
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;

   *num_iterations = (bicgstab_data -> num_iterations);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABGetFinalRelativeResidualNorm( void   *bicgstab_vdata,
                                            NALU_HYPRE_Real *relative_residual_norm )
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;

   *relative_residual_norm = (bicgstab_data -> rel_residual_norm);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BiCGSTABGetResidual( void   *bicgstab_vdata,
                           void **residual )
{
   nalu_hypre_BiCGSTABData *bicgstab_data = (nalu_hypre_BiCGSTABData  *)bicgstab_vdata;

   *residual = (bicgstab_data -> r);

   return nalu_hypre_error_flag;
}
