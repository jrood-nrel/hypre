/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Preconditioned conjugate gradient (Omin) functions
 *
 *****************************************************************************/

/* This was based on the pcg.c formerly in struct_ls, with
   changes (GetPrecond and stop_crit) for compatibility with the pcg.c
   in parcsr_ls and elsewhere.  Incompatibilities with the
   parcsr_ls version:
   - logging is different; no attempt has been made to be the same
   - treatment of b=0 in Ax=b is different: this returns x=0; the parcsr
   version iterates with a special stopping criterion
*/

#include "krylov.h"
#include "_nalu_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGFunctionsCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_PCGFunctions *
nalu_hypre_PCGFunctionsCreate(
   void *       (*CAlloc)        ( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location ),
   NALU_HYPRE_Int    (*Free)          ( void *ptr ),
   NALU_HYPRE_Int    (*CommInfo)      ( void  *A, NALU_HYPRE_Int   *my_id,
                                   NALU_HYPRE_Int   *num_procs ),
   void *       (*CreateVector)  ( void *vector ),
   NALU_HYPRE_Int    (*DestroyVector) ( void *vector ),
   void *       (*MatvecCreate)  ( void *A, void *x ),
   NALU_HYPRE_Int    (*Matvec)        ( void *matvec_data, NALU_HYPRE_Complex alpha, void *A,
                                   void *x, NALU_HYPRE_Complex beta, void *y ),
   NALU_HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
   NALU_HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
   NALU_HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
   NALU_HYPRE_Int    (*ClearVector)   ( void *x ),
   NALU_HYPRE_Int    (*ScaleVector)   ( NALU_HYPRE_Complex alpha, void *x ),
   NALU_HYPRE_Int    (*Axpy)          ( NALU_HYPRE_Complex alpha, void *x, void *y ),
   NALU_HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
   NALU_HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
)
{
   nalu_hypre_PCGFunctions * pcg_functions;
   pcg_functions = (nalu_hypre_PCGFunctions *)
                   CAlloc( 1, sizeof(nalu_hypre_PCGFunctions), NALU_HYPRE_MEMORY_HOST );

   pcg_functions->CAlloc = CAlloc;
   pcg_functions->Free = Free;
   pcg_functions->CommInfo = CommInfo;
   pcg_functions->CreateVector = CreateVector;
   pcg_functions->DestroyVector = DestroyVector;
   pcg_functions->MatvecCreate = MatvecCreate;
   pcg_functions->Matvec = Matvec;
   pcg_functions->MatvecDestroy = MatvecDestroy;
   pcg_functions->InnerProd = InnerProd;
   pcg_functions->CopyVector = CopyVector;
   pcg_functions->ClearVector = ClearVector;
   pcg_functions->ScaleVector = ScaleVector;
   pcg_functions->Axpy = Axpy;
   /* default preconditioner must be set here but can be changed later... */
   pcg_functions->precond_setup = PrecondSetup;
   pcg_functions->precond       = Precond;

   return pcg_functions;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_PCGCreate( nalu_hypre_PCGFunctions *pcg_functions )
{
   nalu_hypre_PCGData *pcg_data;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   pcg_data = nalu_hypre_CTAllocF(nalu_hypre_PCGData, 1, pcg_functions, NALU_HYPRE_MEMORY_HOST);

   pcg_data -> functions = pcg_functions;

   /* set defaults */
   (pcg_data -> tol)          = 1.0e-06;
   (pcg_data -> atolf)        = 0.0;
   (pcg_data -> cf_tol)       = 0.0;
   (pcg_data -> a_tol)        = 0.0;
   (pcg_data -> rtol)         = 0.0;
   (pcg_data -> max_iter)     = 1000;
   (pcg_data -> two_norm)     = 0;
   (pcg_data -> rel_change)   = 0;
   (pcg_data -> recompute_residual) = 0;
   (pcg_data -> recompute_residual_p) = 0;
   (pcg_data -> stop_crit)    = 0;
   (pcg_data -> converged)    = 0;
   (pcg_data -> hybrid)       = 0;
   (pcg_data -> owns_matvec_data ) = 1;
   (pcg_data -> matvec_data)  = NULL;
   (pcg_data -> precond_data) = NULL;
   (pcg_data -> print_level)  = 0;
   (pcg_data -> logging)      = 0;
   (pcg_data -> norms)        = NULL;
   (pcg_data -> rel_norms)    = NULL;
   (pcg_data -> p)            = NULL;
   (pcg_data -> s)            = NULL;
   (pcg_data -> r)            = NULL;

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return (void *) pcg_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGDestroy( void *pcg_vdata )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   if (pcg_data)
   {
      nalu_hypre_PCGFunctions *pcg_functions = pcg_data->functions;
      if ( (pcg_data -> norms) != NULL )
      {
         nalu_hypre_TFreeF( pcg_data -> norms, pcg_functions );
         pcg_data -> norms = NULL;
      }
      if ( (pcg_data -> rel_norms) != NULL )
      {
         nalu_hypre_TFreeF( pcg_data -> rel_norms, pcg_functions );
         pcg_data -> rel_norms = NULL;
      }
      if ( pcg_data -> matvec_data != NULL && pcg_data->owns_matvec_data )
      {
         (*(pcg_functions->MatvecDestroy))(pcg_data -> matvec_data);
         pcg_data -> matvec_data = NULL;
      }
      if ( pcg_data -> p != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> p);
         pcg_data -> p = NULL;
      }
      if ( pcg_data -> s != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> s);
         pcg_data -> s = NULL;
      }
      if ( pcg_data -> r != NULL )
      {
         (*(pcg_functions->DestroyVector))(pcg_data -> r);
         pcg_data -> r = NULL;
      }
      nalu_hypre_TFreeF( pcg_data, pcg_functions );
      nalu_hypre_TFreeF( pcg_functions, pcg_functions );
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return (nalu_hypre_error_flag);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_PCGGetResidual( void *pcg_vdata, void **residual )
{
   /* returns a pointer to the residual vector */

   nalu_hypre_PCGData  *pcg_data     =  (nalu_hypre_PCGData *)pcg_vdata;
   *residual = pcg_data->r;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetup( void *pcg_vdata,
                void *A,
                void *b,
                void *x         )
{
   nalu_hypre_PCGData *pcg_data =  (nalu_hypre_PCGData *)pcg_vdata;
   nalu_hypre_PCGFunctions *pcg_functions = pcg_data->functions;
   NALU_HYPRE_Int            max_iter         = (pcg_data -> max_iter);
   NALU_HYPRE_Int          (*precond_setup)(void*, void*, void*, void*) = (pcg_functions -> precond_setup);
   void          *precond_data     = (pcg_data -> precond_data);

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   (pcg_data -> A) = A;

   /*--------------------------------------------------
    * The arguments for CreateVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/

   if ( pcg_data -> p != NULL )
   {
      (*(pcg_functions->DestroyVector))(pcg_data -> p);
   }
   (pcg_data -> p) = (*(pcg_functions->CreateVector))(x);

   if ( pcg_data -> s != NULL )
   {
      (*(pcg_functions->DestroyVector))(pcg_data -> s);
   }
   (pcg_data -> s) = (*(pcg_functions->CreateVector))(x);

   if ( pcg_data -> r != NULL )
   {
      (*(pcg_functions->DestroyVector))(pcg_data -> r);
   }
   (pcg_data -> r) = (*(pcg_functions->CreateVector))(b);

   if ( pcg_data -> matvec_data != NULL && pcg_data->owns_matvec_data )
   {
      (*(pcg_functions->MatvecDestroy))(pcg_data -> matvec_data);
   }
   (pcg_data -> matvec_data) = (*(pcg_functions->MatvecCreate))(A, x);

   precond_setup(precond_data, A, b, x);

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ( (pcg_data->logging) > 0  || (pcg_data->print_level) > 0 )
   {
      if ( (pcg_data -> norms) != NULL )
      {
         nalu_hypre_TFreeF( pcg_data -> norms, pcg_functions );
      }
      (pcg_data -> norms)     = nalu_hypre_CTAllocF( NALU_HYPRE_Real, max_iter + 1,
                                                pcg_functions, NALU_HYPRE_MEMORY_HOST);

      if ( (pcg_data -> rel_norms) != NULL )
      {
         nalu_hypre_TFreeF( pcg_data -> rel_norms, pcg_functions );
      }
      (pcg_data -> rel_norms) = nalu_hypre_CTAllocF( NALU_HYPRE_Real, max_iter + 1,
                                                pcg_functions, NALU_HYPRE_MEMORY_HOST );
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSolve
 *--------------------------------------------------------------------------
 *
 * We use the following convergence test as the default (see Ashby, Holst,
 * Manteuffel, and Saylor):
 *
 *       ||e||_A                           ||r||_C
 *       -------  <=  [kappa_A(C*A)]^(1/2) -------  < tol
 *       ||x||_A                           ||b||_C
 *
 * where we let (for the time being) kappa_A(CA) = 1.
 * We implement the test as:
 *
 *       gamma = <C*r,r>/<C*b,b>  <  (tol^2) = eps
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSolve( void *pcg_vdata,
                void *A,
                void *b,
                void *x         )
{
   nalu_hypre_PCGData  *pcg_data     =  (nalu_hypre_PCGData *)pcg_vdata;
   nalu_hypre_PCGFunctions *pcg_functions = pcg_data->functions;

   NALU_HYPRE_Real      r_tol        = (pcg_data -> tol);
   NALU_HYPRE_Real      a_tol        = (pcg_data -> a_tol);
   NALU_HYPRE_Real      atolf        = (pcg_data -> atolf);
   NALU_HYPRE_Real      cf_tol       = (pcg_data -> cf_tol);
   NALU_HYPRE_Real      rtol         = (pcg_data -> rtol);
   NALU_HYPRE_Int       max_iter     = (pcg_data -> max_iter);
   NALU_HYPRE_Int       two_norm     = (pcg_data -> two_norm);
   NALU_HYPRE_Int       rel_change   = (pcg_data -> rel_change);
   NALU_HYPRE_Int       recompute_residual   = (pcg_data -> recompute_residual);
   NALU_HYPRE_Int       recompute_residual_p = (pcg_data -> recompute_residual_p);
   NALU_HYPRE_Int       stop_crit    = (pcg_data -> stop_crit);
   NALU_HYPRE_Int       hybrid       = (pcg_data -> hybrid);
   /*
      NALU_HYPRE_Int             converged    = (pcg_data -> converged);
   */
   void           *p            = (pcg_data -> p);
   void           *s            = (pcg_data -> s);
   void           *r            = (pcg_data -> r);
   void           *matvec_data  = (pcg_data -> matvec_data);
   NALU_HYPRE_Int     (*precond)(void*, void*, void*, void*)   = (pcg_functions -> precond);
   void           *precond_data = (pcg_data -> precond_data);
   NALU_HYPRE_Int       print_level  = (pcg_data -> print_level);
   NALU_HYPRE_Int       logging      = (pcg_data -> logging);
   NALU_HYPRE_Real     *norms        = (pcg_data -> norms);
   NALU_HYPRE_Real     *rel_norms    = (pcg_data -> rel_norms);

   NALU_HYPRE_Real      alpha, beta;
   NALU_HYPRE_Real      gamma, gamma_old;
   NALU_HYPRE_Real      bi_prod, eps;
   NALU_HYPRE_Real      pi_prod, xi_prod;
   NALU_HYPRE_Real      ieee_check = 0.;

   NALU_HYPRE_Real      i_prod = 0.0;
   NALU_HYPRE_Real      i_prod_0 = 0.0;
   NALU_HYPRE_Real      cf_ave_0 = 0.0;
   NALU_HYPRE_Real      cf_ave_1 = 0.0;
   NALU_HYPRE_Real      weight;
   NALU_HYPRE_Real      ratio;

   NALU_HYPRE_Real      guard_zero_residual, sdotp;
   NALU_HYPRE_Int       tentatively_converged = 0;
   NALU_HYPRE_Int       recompute_true_residual = 0;

   NALU_HYPRE_Int       i = 0;
   NALU_HYPRE_Int       my_id, num_procs;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   (pcg_data -> converged) = 0;

   (*(pcg_functions->CommInfo))(A, &my_id, &num_procs);

   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/

   guard_zero_residual = 0.0;

   /*-----------------------------------------------------------------------
    * Start pcg solve
    *-----------------------------------------------------------------------*/

   /* compute eps */
   if (two_norm)
   {
      /* bi_prod = <b,b> */
      bi_prod = (*(pcg_functions->InnerProd))(b, b);
      if (print_level > 1 && my_id == 0)
      {
         nalu_hypre_printf("<b,b>: %e\n", bi_prod);
      }
   }
   else
   {
      /* bi_prod = <C*b,b> */
      (*(pcg_functions->ClearVector))(p);
      precond(precond_data, A, b, p);
      bi_prod = (*(pcg_functions->InnerProd))(p, b);
      if (print_level > 1 && my_id == 0)
      {
         nalu_hypre_printf("<C*b,b>: %e\n", bi_prod);
      }
   };

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (bi_prod != 0.) { ieee_check = bi_prod / bi_prod; } /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (print_level > 0 || logging > 0)
      {
         nalu_hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
         nalu_hypre_printf("ERROR -- nalu_hypre_PCGSolve: INFs and/or NaNs detected in input.\n");
         nalu_hypre_printf("User probably placed non-numerics in supplied b.\n");
         nalu_hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         nalu_hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
      NALU_HYPRE_ANNOTATE_FUNC_END;

      return nalu_hypre_error_flag;
   }

   eps = r_tol * r_tol; /* note: this may be re-assigned below */
   if ( bi_prod > 0.0 )
   {
      if ( stop_crit && !rel_change && atolf <= 0 ) /* pure absolute tolerance */
      {
         eps = eps / bi_prod;
         /* Note: this section is obsolete.  Aside from backwards comatability
            concerns, we could delete the stop_crit parameter and related code,
            using tol & atolf instead. */
      }
      else if ( atolf > 0 ) /* mixed relative and absolute tolerance */
      {
         bi_prod += atolf;
      }
      else /* DEFAULT (stop_crit and atolf exist for backwards compatibilty
              and are not in the reference manual) */
      {
         /* convergence criteria:  <C*r,r>  <= max( a_tol^2, r_tol^2 * <C*b,b> )
             note: default for a_tol is 0.0, so relative residual criteria is used unless
             user specifies a_tol, or sets r_tol = 0.0, which means absolute
             tol only is checked  */
         eps = nalu_hypre_max(r_tol * r_tol, a_tol * a_tol / bi_prod);

      }
   }
   else    /* bi_prod==0.0: the rhs vector b is zero */
   {
      /* Set x equal to zero and return */
      (*(pcg_functions->CopyVector))(b, x);
      if (logging > 0 || print_level > 0)
      {
         norms[0]     = 0.0;
         rel_norms[i] = 0.0;
      }
      NALU_HYPRE_ANNOTATE_FUNC_END;

      return nalu_hypre_error_flag;
      /* In this case, for the original parcsr pcg, the code would take special
         action to force iterations even though the exact value was known. */
   };

   /* r = b - Ax */
   (*(pcg_functions->CopyVector))(b, r);

   (*(pcg_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);

   //nalu_hypre_ParVectorUpdateHost(r);
   /* p = C*r */
   (*(pcg_functions->ClearVector))(p);
   precond(precond_data, A, r, p);

   /* gamma = <r,p> */
   gamma = (*(pcg_functions->InnerProd))(r, p);

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (gamma != 0.) { ieee_check = gamma / gamma; } /* INF -> NaN conversion */
   if (ieee_check != ieee_check)
   {
      /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
         for ieee_check self-equality works on all IEEE-compliant compilers/
         machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
         by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
         found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
      if (print_level > 0 || logging > 0)
      {
         nalu_hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
         nalu_hypre_printf("ERROR -- nalu_hypre_PCGSolve: INFs and/or NaNs detected in input.\n");
         nalu_hypre_printf("User probably placed non-numerics in supplied A or x_0.\n");
         nalu_hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         nalu_hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
      }
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
      NALU_HYPRE_ANNOTATE_FUNC_END;

      return nalu_hypre_error_flag;
   }

   /* Set initial residual norm */
   if ( logging > 0 || print_level > 0 || cf_tol > 0.0 )
   {
      if (two_norm)
      {
         i_prod_0 = (*(pcg_functions->InnerProd))(r, r);
      }
      else
      {
         i_prod_0 = gamma;
      }

      if ( logging > 0 || print_level > 0 ) { norms[0] = sqrt(i_prod_0); }
   }
   if ( print_level > 1 && my_id == 0 )
   {
      nalu_hypre_printf("\n\n");
      if (two_norm)
      {
         if ( stop_crit && !rel_change && atolf == 0 ) /* pure absolute tolerance */
         {
            nalu_hypre_printf("Iters       ||r||_2     conv.rate\n");
            nalu_hypre_printf("-----    ------------   ---------\n");
         }
         else
         {
            nalu_hypre_printf("Iters       ||r||_2     conv.rate  ||r||_2/||b||_2\n");
            nalu_hypre_printf("-----    ------------   ---------  ------------ \n");
         }
      }
      else  /* !two_norm */
      {
         nalu_hypre_printf("Iters       ||r||_C     conv.rate  ||r||_C/||b||_C\n");
         nalu_hypre_printf("-----    ------------    ---------  ------------ \n");
      }
      /* nalu_hypre_printf("% 5d    %e\n", i, norms[i]); */
   }

   while ((i + 1) <= max_iter)
   {
      /*--------------------------------------------------------------------
       * the core CG calculations...
       *--------------------------------------------------------------------*/
      i++;

      /* At user request, periodically recompute the residual from the formula
         r = b - A x (instead of using the recursive definition). Note that this
         is potentially expensive and can lead to degraded convergence (since it
         essentially a "restarted CG"). */
      recompute_true_residual = recompute_residual_p && !(i % recompute_residual_p);

      /* s = A*p */
      (*(pcg_functions->Matvec))(matvec_data, 1.0, A, p, 0.0, s);

      /* alpha = gamma / <s,p> */
      sdotp = (*(pcg_functions->InnerProd))(s, p);
      if ( sdotp == 0.0 )
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_CONV, "Zero sdotp value in PCG");
         if (i == 1) { i_prod = i_prod_0; }
         break;
      }
      alpha = gamma / sdotp;
      if (! (alpha > NALU_HYPRE_REAL_MIN) )
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_CONV, "Subnormal alpha value in PCG");
         if (i == 1) { i_prod = i_prod_0; }
         break;
      }

      gamma_old = gamma;

      /* x = x + alpha*p */
      (*(pcg_functions->Axpy))(alpha, p, x);

      /* r = r - alpha*s */
      if ( !recompute_true_residual )
      {
         (*(pcg_functions->Axpy))(-alpha, s, r);
      }
      else
      {
         if (print_level > 1 && my_id == 0)
         {
            nalu_hypre_printf("Recomputing the residual...\n");
         }
         (*(pcg_functions->CopyVector))(b, r);
         (*(pcg_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);
      }

      /* residual-based stopping criteria: ||r_new-r_old|| < rtol ||b|| */
      if (rtol && two_norm)
      {
         /* use that r_new-r_old = alpha * s */
         NALU_HYPRE_Real drob2 = alpha * alpha * (*(pcg_functions->InnerProd))(s, s) / bi_prod;
         if ( drob2 < rtol * rtol )
         {
            if (print_level > 1 && my_id == 0)
            {
               nalu_hypre_printf("\n\n||r_old-r_new||/||b||: %e\n", sqrt(drob2));
            }
            break;
         }
      }

      /* s = C*r */
      (*(pcg_functions->ClearVector))(s);
      precond(precond_data, A, r, s);

      /* gamma = <r,s> */
      gamma = (*(pcg_functions->InnerProd))(r, s);

      /* residual-based stopping criteria: ||r_new-r_old||_C < rtol ||b||_C */
      if (rtol && !two_norm)
      {
         /* use that ||r_new-r_old||_C^2 = (r_new ,C r_new) + (r_old, C r_old) */
         NALU_HYPRE_Real r2ob2 = (gamma + gamma_old) / bi_prod;
         if ( r2ob2 < rtol * rtol)
         {
            if (print_level > 1 && my_id == 0)
            {
               nalu_hypre_printf("\n\n||r_old-r_new||_C/||b||_C: %e\n", sqrt(r2ob2));
            }
            break;
         }
      }

      /* set i_prod for convergence test */
      if (two_norm)
      {
         i_prod = (*(pcg_functions->InnerProd))(r, r);
      }
      else
      {
         i_prod = gamma;
      }

      /*--------------------------------------------------------------------
       * optional output
       *--------------------------------------------------------------------*/
#if 0
      if (two_norm)
         nalu_hypre_printf("Iter (%d): ||r||_2 = %e, ||r||_2/||b||_2 = %e\n",
                      i, sqrt(i_prod), (bi_prod ? sqrt(i_prod / bi_prod) : 0));
      else
         nalu_hypre_printf("Iter (%d): ||r||_C = %e, ||r||_C/||b||_C = %e\n",
                      i, sqrt(i_prod), (bi_prod ? sqrt(i_prod / bi_prod) : 0));
#endif

      /* print norm info */
      if ( logging > 0 || print_level > 0 )
      {
         norms[i]     = sqrt(i_prod);
         rel_norms[i] = bi_prod ? sqrt(i_prod / bi_prod) : 0;
      }
      if ( print_level > 1 && my_id == 0 )
      {
         if (two_norm)
         {
            if ( stop_crit && !rel_change && atolf == 0 )  /* pure absolute tolerance */
            {
               nalu_hypre_printf("% 5d    %e    %f\n", i, norms[i],
                            norms[i] / norms[i - 1] );
            }
            else
            {
               nalu_hypre_printf("% 5d    %e    %f    %e\n", i, norms[i],
                            norms[i] / norms[i - 1], rel_norms[i] );
            }
         }
         else
         {
            nalu_hypre_printf("% 5d    %e    %f    %e\n", i, norms[i],
                         norms[i] / norms[i - 1], rel_norms[i] );
         }
      }


      /*--------------------------------------------------------------------
       * check for convergence
       *--------------------------------------------------------------------*/
      if (i_prod / bi_prod < eps)  /* the basic convergence test */
      {
         tentatively_converged = 1;
      }
      if ( tentatively_converged && recompute_residual )
         /* At user request, don't trust the convergence test until we've recomputed
            the residual from scratch.  This is expensive in the usual case where an
            the norm is the energy norm.
            This calculation is coded on the assumption that r's accuracy is only a
            concern for problems where CG takes many iterations. */
      {
         /* r = b - Ax */
         (*(pcg_functions->CopyVector))(b, r);
         (*(pcg_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);

         /* set i_prod for convergence test */
         if (two_norm)
         {
            i_prod = (*(pcg_functions->InnerProd))(r, r);
         }
         else
         {
            /* s = C*r */
            (*(pcg_functions->ClearVector))(s);
            precond(precond_data, A, r, s);
            /* iprod = gamma = <r,s> */
            i_prod = (*(pcg_functions->InnerProd))(r, s);
         }
         if (i_prod / bi_prod >= eps) { tentatively_converged = 0; }
      }
      if ( tentatively_converged && rel_change && (i_prod > guard_zero_residual ))
         /* At user request, don't treat this as converged unless x didn't change
            much in the last iteration. */
      {
         pi_prod = (*(pcg_functions->InnerProd))(p, p);
         xi_prod = (*(pcg_functions->InnerProd))(x, x);
         ratio = alpha * alpha * pi_prod / xi_prod;
         if (ratio >= eps) { tentatively_converged = 0; }
      }
      if ( tentatively_converged )
         /* we've passed all the convergence tests, it's for real */
      {
         (pcg_data -> converged) = 1;
         break;
      }

      if (! (gamma > NALU_HYPRE_REAL_MIN) )
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_CONV, "Subnormal gamma value in PCG");

         break;
      }
      /* ... gamma should be >=0.  IEEE subnormal numbers are < 2**(-1022)=2.2e-308
         (and >= 2**(-1074)=4.9e-324).  So a gamma this small means we're getting
         dangerously close to subnormal or zero numbers (usually if gamma is small,
         so will be other variables).  Thus further calculations risk a crash.
         Such small gamma generally means no hope of progress anyway. */

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
         if (! (i_prod_0 > NALU_HYPRE_REAL_MIN) )
         {
            /* i_prod_0 is zero, or (almost) subnormal, yet i_prod wasn't small
               enough to pass the convergence test.  Therefore initial guess was good,
               and we're just calculating garbage - time to bail out before the
               next step, which will be a divide by zero (or close to it). */
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_CONV, "Subnormal i_prod value in PCG");

            break;
         }
         cf_ave_1 = pow( i_prod / i_prod_0, 1.0 / (2.0 * i) );

         weight   = fabs(cf_ave_1 - cf_ave_0);
         weight   = weight / nalu_hypre_max(cf_ave_1, cf_ave_0);
         weight   = 1.0 - weight;
#if 0
         nalu_hypre_printf("I = %d: cf_new = %e, cf_old = %e, weight = %e\n",
                      i, cf_ave_1, cf_ave_0, weight );
#endif
         if (weight * cf_ave_1 > cf_tol) { break; }
      }

      /*--------------------------------------------------------------------
       * back to the core CG calculations
       *--------------------------------------------------------------------*/

      /* beta = gamma / gamma_old */
      beta = gamma / gamma_old;

      /* p = s + beta p */
      if ( !recompute_true_residual )
      {
         (*(pcg_functions->ScaleVector))(beta, p);
         (*(pcg_functions->Axpy))(1.0, s, p);
      }
      else
      {
         (*(pcg_functions->CopyVector))(s, p);
      }
   }

   /*--------------------------------------------------------------------
    * Finish up with some outputs.
    *--------------------------------------------------------------------*/

   if ( print_level > 1 && my_id == 0 )
   {
      nalu_hypre_printf("\n\n");
   }

   if (i >= max_iter && (i_prod / bi_prod) >= eps && eps > 0 && hybrid != -1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_CONV, "Reached max iterations in PCG before convergence");
   }

   (pcg_data -> num_iterations) = i;
   if (bi_prod > 0.0)
   {
      (pcg_data -> rel_residual_norm) = sqrt(i_prod / bi_prod);
   }
   else /* actually, we'll never get here... */
   {
      (pcg_data -> rel_residual_norm) = 0.0;
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetTol, nalu_hypre_PCGGetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetTol( void   *pcg_vdata,
                 NALU_HYPRE_Real  tol       )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   (pcg_data -> tol) = tol;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PCGGetTol( void   *pcg_vdata,
                 NALU_HYPRE_Real * tol       )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   *tol = (pcg_data -> tol);

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetAbsoluteTol, nalu_hypre_PCGGetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetAbsoluteTol( void   *pcg_vdata,
                         NALU_HYPRE_Real  a_tol       )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   (pcg_data -> a_tol) = a_tol;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PCGGetAbsoluteTol( void   *pcg_vdata,
                         NALU_HYPRE_Real * a_tol       )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   *a_tol = (pcg_data -> a_tol);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetAbsoluteTolFactor, nalu_hypre_PCGGetAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetAbsoluteTolFactor( void   *pcg_vdata,
                               NALU_HYPRE_Real  atolf   )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   (pcg_data -> atolf) = atolf;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PCGGetAbsoluteTolFactor( void   *pcg_vdata,
                               NALU_HYPRE_Real  * atolf   )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   *atolf = (pcg_data -> atolf);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetResidualTol, nalu_hypre_PCGGetResidualTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetResidualTol( void   *pcg_vdata,
                         NALU_HYPRE_Real  rtol   )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   (pcg_data -> rtol) = rtol;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PCGGetResidualTol( void   *pcg_vdata,
                         NALU_HYPRE_Real  * rtol   )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   *rtol = (pcg_data -> rtol);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetConvergenceFactorTol, nalu_hypre_PCGGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetConvergenceFactorTol( void   *pcg_vdata,
                                  NALU_HYPRE_Real  cf_tol   )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   (pcg_data -> cf_tol) = cf_tol;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PCGGetConvergenceFactorTol( void   *pcg_vdata,
                                  NALU_HYPRE_Real * cf_tol   )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   *cf_tol = (pcg_data -> cf_tol);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetMaxIter, nalu_hypre_PCGGetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetMaxIter( void *pcg_vdata,
                     NALU_HYPRE_Int   max_iter  )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   (pcg_data -> max_iter) = max_iter;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PCGGetMaxIter( void *pcg_vdata,
                     NALU_HYPRE_Int * max_iter  )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;


   *max_iter = (pcg_data -> max_iter);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetTwoNorm, nalu_hypre_PCGGetTwoNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetTwoNorm( void *pcg_vdata,
                     NALU_HYPRE_Int   two_norm  )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;


   (pcg_data -> two_norm) = two_norm;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PCGGetTwoNorm( void *pcg_vdata,
                     NALU_HYPRE_Int * two_norm  )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;


   *two_norm = (pcg_data -> two_norm);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetRelChange, nalu_hypre_PCGGetRelChange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetRelChange( void *pcg_vdata,
                       NALU_HYPRE_Int   rel_change  )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;


   (pcg_data -> rel_change) = rel_change;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PCGGetRelChange( void *pcg_vdata,
                       NALU_HYPRE_Int * rel_change  )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;


   *rel_change = (pcg_data -> rel_change);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetRecomputeResidual, nalu_hypre_PCGGetRecomputeResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetRecomputeResidual( void *pcg_vdata,
                               NALU_HYPRE_Int   recompute_residual  )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;


   (pcg_data -> recompute_residual) = recompute_residual;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PCGGetRecomputeResidual( void *pcg_vdata,
                               NALU_HYPRE_Int * recompute_residual  )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;


   *recompute_residual = (pcg_data -> recompute_residual);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetRecomputeResidualP, nalu_hypre_PCGGetRecomputeResidualP
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetRecomputeResidualP( void *pcg_vdata,
                                NALU_HYPRE_Int   recompute_residual_p  )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   (pcg_data -> recompute_residual_p) = recompute_residual_p;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PCGGetRecomputeResidualP( void *pcg_vdata,
                                NALU_HYPRE_Int * recompute_residual_p  )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   *recompute_residual_p = (pcg_data -> recompute_residual_p);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetStopCrit, nalu_hypre_PCGGetStopCrit
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetStopCrit( void *pcg_vdata,
                      NALU_HYPRE_Int   stop_crit  )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;


   (pcg_data -> stop_crit) = stop_crit;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PCGGetStopCrit( void *pcg_vdata,
                      NALU_HYPRE_Int * stop_crit  )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;


   *stop_crit = (pcg_data -> stop_crit);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGGetPrecond( void         *pcg_vdata,
                     NALU_HYPRE_Solver *precond_data_ptr )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;


   *precond_data_ptr = (NALU_HYPRE_Solver)(pcg_data -> precond_data);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetPrecond( void  *pcg_vdata,
                     NALU_HYPRE_Int  (*precond)(void*, void*, void*, void*),
                     NALU_HYPRE_Int  (*precond_setup)(void*, void*, void*, void*),
                     void  *precond_data )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;
   nalu_hypre_PCGFunctions *pcg_functions = pcg_data->functions;


   (pcg_functions -> precond)       = precond;
   (pcg_functions -> precond_setup) = precond_setup;
   (pcg_data -> precond_data)  = precond_data;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetPrintLevel, nalu_hypre_PCGGetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetPrintLevel( void *pcg_vdata,
                        NALU_HYPRE_Int   level)
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;


   (pcg_data -> print_level) = level;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PCGGetPrintLevel( void *pcg_vdata,
                        NALU_HYPRE_Int * level)
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;


   *level = (pcg_data -> print_level);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGSetLogging, nalu_hypre_PCGGetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGSetLogging( void *pcg_vdata,
                     NALU_HYPRE_Int   level)
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   (pcg_data -> logging) = level;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PCGGetLogging( void *pcg_vdata,
                     NALU_HYPRE_Int * level)
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   *level = (pcg_data -> logging);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PCGSetHybrid( void *pcg_vdata,
                    NALU_HYPRE_Int   level)
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   (pcg_data -> hybrid) = level;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGGetNumIterations( void *pcg_vdata,
                           NALU_HYPRE_Int  *num_iterations )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   *num_iterations = (pcg_data -> num_iterations);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGGetConverged
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGGetConverged( void *pcg_vdata,
                       NALU_HYPRE_Int  *converged)
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   *converged = (pcg_data -> converged);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGPrintLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGPrintLogging( void *pcg_vdata,
                       NALU_HYPRE_Int   myid)
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   NALU_HYPRE_Int            num_iterations  = (pcg_data -> num_iterations);
   NALU_HYPRE_Int            print_level     = (pcg_data -> print_level);
   NALU_HYPRE_Real    *norms           = (pcg_data -> norms);
   NALU_HYPRE_Real    *rel_norms       = (pcg_data -> rel_norms);

   NALU_HYPRE_Int            i;

   if (myid == 0)
   {
      if (print_level > 0)
      {
         for (i = 0; i < num_iterations; i++)
         {
            nalu_hypre_printf("Residual norm[%d] = %e   ", i, norms[i]);
            nalu_hypre_printf("Relative residual norm[%d] = %e\n", i, rel_norms[i]);
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PCGGetFinalRelativeResidualNorm( void   *pcg_vdata,
                                       NALU_HYPRE_Real *relative_residual_norm )
{
   nalu_hypre_PCGData *pcg_data = (nalu_hypre_PCGData *)pcg_vdata;

   NALU_HYPRE_Real     rel_residual_norm = (pcg_data -> rel_residual_norm);

   *relative_residual_norm = rel_residual_norm;

   return nalu_hypre_error_flag;
}
