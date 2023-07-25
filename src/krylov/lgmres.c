/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * LGMRES lgmres
 *
 *****************************************************************************/

#include "krylov.h"
#include "_nalu_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESFunctionsCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_LGMRESFunctions *
nalu_hypre_LGMRESFunctionsCreate(
   void *       (*CAlloc)        ( size_t count, size_t elt_size, NALU_HYPRE_MemoryLocation location ),
   NALU_HYPRE_Int    (*Free)          ( void *ptr ),
   NALU_HYPRE_Int    (*CommInfo)      ( void  *A, NALU_HYPRE_Int   *my_id,
                                   NALU_HYPRE_Int   *num_procs ),
   void *       (*CreateVector)  ( void *vector ),
   void *       (*CreateVectorArray)  ( NALU_HYPRE_Int size, void *vectors ),
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
   nalu_hypre_LGMRESFunctions * lgmres_functions;
   lgmres_functions = (nalu_hypre_LGMRESFunctions *)
                      CAlloc( 1, sizeof(nalu_hypre_LGMRESFunctions), NALU_HYPRE_MEMORY_HOST );

   lgmres_functions->CAlloc = CAlloc;
   lgmres_functions->Free = Free;
   lgmres_functions->CommInfo = CommInfo; /* not in PCGFunctionsCreate */
   lgmres_functions->CreateVector = CreateVector;
   lgmres_functions->CreateVectorArray = CreateVectorArray; /* not in PCGFunctionsCreate */
   lgmres_functions->DestroyVector = DestroyVector;
   lgmres_functions->MatvecCreate = MatvecCreate;
   lgmres_functions->Matvec = Matvec;
   lgmres_functions->MatvecDestroy = MatvecDestroy;
   lgmres_functions->InnerProd = InnerProd;
   lgmres_functions->CopyVector = CopyVector;
   lgmres_functions->ClearVector = ClearVector;
   lgmres_functions->ScaleVector = ScaleVector;
   lgmres_functions->Axpy = Axpy;
   /* default preconditioner must be set here but can be changed later... */
   lgmres_functions->precond_setup = PrecondSetup;
   lgmres_functions->precond       = Precond;

   return lgmres_functions;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_LGMRESCreate( nalu_hypre_LGMRESFunctions *lgmres_functions )
{
   nalu_hypre_LGMRESData *lgmres_data;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   lgmres_data = nalu_hypre_CTAllocF(nalu_hypre_LGMRESData, 1, lgmres_functions, NALU_HYPRE_MEMORY_HOST);
   lgmres_data->functions = lgmres_functions;

   /* set defaults */
   (lgmres_data -> k_dim)          = 20;
   (lgmres_data -> tol)            = 1.0e-06;
   (lgmres_data -> cf_tol)         = 0.0;
   (lgmres_data -> a_tol)          = 0.0; /* abs. residual tol */
   (lgmres_data -> min_iter)       = 0;
   (lgmres_data -> max_iter)       = 1000;
   (lgmres_data -> rel_change)     = 0;
   (lgmres_data -> stop_crit)      = 0; /* rel. residual norm */
   (lgmres_data -> converged)      = 0;
   (lgmres_data -> precond_data)   = NULL;
   (lgmres_data -> print_level)    = 0;
   (lgmres_data -> logging)        = 0;
   (lgmres_data -> p)              = NULL;
   (lgmres_data -> r)              = NULL;
   (lgmres_data -> w)              = NULL;
   (lgmres_data -> w_2)            = NULL;
   (lgmres_data -> matvec_data)    = NULL;
   (lgmres_data -> norms)          = NULL;
   (lgmres_data -> log_file_name)  = NULL;

   /* lgmres specific */
   (lgmres_data -> aug_dim)         = 2;
   (lgmres_data -> approx_constant) = 1;

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return (void *) lgmres_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESDestroy( void *lgmres_vdata )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;
   NALU_HYPRE_Int i;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;
   if (lgmres_data)
   {
      nalu_hypre_LGMRESFunctions *lgmres_functions = lgmres_data->functions;
      if ( (lgmres_data->logging > 0) || (lgmres_data->print_level) > 0 )
      {
         if ( (lgmres_data -> norms) != NULL )
         {
            nalu_hypre_TFreeF( lgmres_data -> norms, lgmres_functions );
         }
      }

      if ( (lgmres_data -> matvec_data) != NULL )
      {
         (*(lgmres_functions->MatvecDestroy))(lgmres_data -> matvec_data);
      }

      if ( (lgmres_data -> r) != NULL )
      {
         (*(lgmres_functions->DestroyVector))(lgmres_data -> r);
      }
      if ( (lgmres_data -> w) != NULL )
      {
         (*(lgmres_functions->DestroyVector))(lgmres_data -> w);
      }
      if ( (lgmres_data -> w_2) != NULL )
      {
         (*(lgmres_functions->DestroyVector))(lgmres_data -> w_2);
      }


      if ( (lgmres_data -> p) != NULL )
      {
         for (i = 0; i < (lgmres_data -> k_dim + 1); i++)
         {
            if ( (lgmres_data -> p)[i] != NULL )
            {
               (*(lgmres_functions->DestroyVector))( (lgmres_data -> p) [i]);
            }
         }
         nalu_hypre_TFreeF( lgmres_data->p, lgmres_functions );
      }

      /* lgmres mod */
      if ( (lgmres_data -> aug_vecs) != NULL )
      {
         for (i = 0; i < (lgmres_data -> aug_dim + 1); i++)
         {
            if ( (lgmres_data -> aug_vecs)[i] != NULL )
            {
               (*(lgmres_functions->DestroyVector))( (lgmres_data -> aug_vecs) [i]);
            }
         }
         nalu_hypre_TFreeF( lgmres_data->aug_vecs, lgmres_functions );
      }
      if ( (lgmres_data -> a_aug_vecs) != NULL )
      {
         for (i = 0; i < (lgmres_data -> aug_dim); i++)
         {
            if ( (lgmres_data -> a_aug_vecs)[i] != NULL )
            {
               (*(lgmres_functions->DestroyVector))( (lgmres_data -> a_aug_vecs) [i]);
            }
         }
         nalu_hypre_TFreeF( lgmres_data->a_aug_vecs, lgmres_functions );
      }
      /*---*/

      nalu_hypre_TFreeF(lgmres_data->aug_order, lgmres_functions);



      nalu_hypre_TFreeF( lgmres_data, lgmres_functions );
      nalu_hypre_TFreeF( lgmres_functions, lgmres_functions );
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_LGMRESGetResidual( void *lgmres_vdata, void **residual )
{
   nalu_hypre_LGMRESData  *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;
   *residual = lgmres_data->r;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESSetup( void *lgmres_vdata,
                   void *A,
                   void *b,
                   void *x         )
{
   nalu_hypre_LGMRESData *lgmres_data     = (nalu_hypre_LGMRESData *)lgmres_vdata;
   nalu_hypre_LGMRESFunctions *lgmres_functions = lgmres_data->functions;

   NALU_HYPRE_Int            k_dim            = (lgmres_data -> k_dim);
   NALU_HYPRE_Int            max_iter         = (lgmres_data -> max_iter);
   NALU_HYPRE_Int          (*precond_setup)(void*, void*, void*, void*) = (lgmres_functions->precond_setup);
   void          *precond_data     = (lgmres_data -> precond_data);

   NALU_HYPRE_Int            rel_change       = (lgmres_data -> rel_change);

   /* lgmres mod */
   NALU_HYPRE_Int            aug_dim          = (lgmres_data -> aug_dim);

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   (lgmres_data -> A) = A;

   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/

   if ((lgmres_data -> p) == NULL)
   {
      (lgmres_data -> p) = (void**)(*(lgmres_functions->CreateVectorArray))(k_dim + 1, x);
   }
   if ((lgmres_data -> r) == NULL)
   {
      (lgmres_data -> r) = (*(lgmres_functions->CreateVector))(b);
   }
   if ((lgmres_data -> w) == NULL)
   {
      (lgmres_data -> w) = (*(lgmres_functions->CreateVector))(b);
   }

   if (rel_change)
   {
      if ((lgmres_data -> w_2) == NULL)
      {
         (lgmres_data -> w_2) = (*(lgmres_functions->CreateVector))(b);
      }
   }

   /* lgmres mod */
   if ((lgmres_data -> aug_vecs) == NULL)
   {
      (lgmres_data -> aug_vecs) = (void**)(*(lgmres_functions->CreateVectorArray))(aug_dim + 1,
                                                                                   x);   /* one extra */
   }
   if ((lgmres_data -> a_aug_vecs) == NULL)
   {
      (lgmres_data -> a_aug_vecs) = (void**)(*(lgmres_functions->CreateVectorArray))(aug_dim, x);
   }
   if ((lgmres_data -> aug_order) == NULL)
   {
      (lgmres_data -> aug_order) = nalu_hypre_CTAllocF(NALU_HYPRE_Int, aug_dim, lgmres_functions,
                                                  NALU_HYPRE_MEMORY_HOST);
   }
   /*---*/


   if ((lgmres_data -> matvec_data) == NULL)
   {
      (lgmres_data -> matvec_data) = (*(lgmres_functions->MatvecCreate))(A, x);
   }

   precond_setup(precond_data, A, b, x);

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ( (lgmres_data->logging) > 0 || (lgmres_data->print_level) > 0 )
   {
      if ((lgmres_data -> norms) == NULL)
      {
         (lgmres_data -> norms) = nalu_hypre_CTAllocF(NALU_HYPRE_Real, max_iter + 1, lgmres_functions,
                                                 NALU_HYPRE_MEMORY_HOST);
      }
   }
   if ( (lgmres_data->print_level) > 0 )
   {
      if ((lgmres_data -> log_file_name) == NULL)
      {
         (lgmres_data -> log_file_name) = (char*)"lgmres.out.log";
      }
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESSolve

   Note: no rel. change capability

 *-------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESSolve(void  *lgmres_vdata,
                  void  *A,
                  void  *b,
                  void  *x)
{
   nalu_hypre_LGMRESData  *lgmres_data   = (nalu_hypre_LGMRESData *)lgmres_vdata;
   nalu_hypre_LGMRESFunctions *lgmres_functions = lgmres_data->functions;
   NALU_HYPRE_Int           k_dim        = (lgmres_data -> k_dim);
   NALU_HYPRE_Int               min_iter     = (lgmres_data -> min_iter);
   NALU_HYPRE_Int           max_iter     = (lgmres_data -> max_iter);
   NALU_HYPRE_Real       r_tol        = (lgmres_data -> tol);
   NALU_HYPRE_Real       cf_tol       = (lgmres_data -> cf_tol);
   NALU_HYPRE_Real        a_tol        = (lgmres_data -> a_tol);
   void             *matvec_data  = (lgmres_data -> matvec_data);

   void             *r            = (lgmres_data -> r);
   void             *w            = (lgmres_data -> w);


   void            **p            = (lgmres_data -> p);

   /* lgmres  mod*/
   void          **aug_vecs       = (lgmres_data ->aug_vecs);
   void          **a_aug_vecs     = (lgmres_data ->a_aug_vecs);
   NALU_HYPRE_Int            *aug_order      = (lgmres_data->aug_order);
   NALU_HYPRE_Int             aug_dim        = (lgmres_data -> aug_dim);
   NALU_HYPRE_Int             approx_constant =  (lgmres_data ->approx_constant);
   NALU_HYPRE_Int             it_arnoldi, aug_ct, it_total, ii, order, it_aug;
   NALU_HYPRE_Int             spot = 0;
   NALU_HYPRE_Real      tmp_norm, r_norm_last;
   /*---*/

   NALU_HYPRE_Int              (*precond)(void*, void*, void*, void*)   = (lgmres_functions -> precond);
   NALU_HYPRE_Int               *precond_data = (NALU_HYPRE_Int*)(lgmres_data -> precond_data);

   NALU_HYPRE_Int             print_level    = (lgmres_data -> print_level);
   NALU_HYPRE_Int             logging        = (lgmres_data -> logging);

   NALU_HYPRE_Real     *norms          = (lgmres_data -> norms);

   NALU_HYPRE_Int        break_value = 0;
   NALU_HYPRE_Int         i, j, k;
   NALU_HYPRE_Real *rs, **hh, *c, *s;
   NALU_HYPRE_Int        iter;
   NALU_HYPRE_Int        my_id, num_procs;
   NALU_HYPRE_Real epsilon, gamma, t, r_norm, b_norm, den_norm;

   NALU_HYPRE_Real epsmac = 1.e-16;
   NALU_HYPRE_Real ieee_check = 0.;

   NALU_HYPRE_Real cf_ave_0 = 0.0;
   NALU_HYPRE_Real cf_ave_1 = 0.0;
   NALU_HYPRE_Real weight;
   NALU_HYPRE_Real r_norm_0;

   /* We are not checking rel. change for now... */
   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   (lgmres_data -> converged) = 0;
   /*-----------------------------------------------------------------------
    * With relative change convergence test on, it is possible to attempt
    * another iteration with a zero residual. This causes the parameter
    * alpha to go NaN. The guard_zero_residual parameter is to circumvent
    * this. Perhaps it should be set to something non-zero (but small).
    *-----------------------------------------------------------------------*/

   (*(lgmres_functions->CommInfo))(A, &my_id, &num_procs);
   if ( logging > 0 || print_level > 0 )
   {
      norms          = (lgmres_data -> norms);
      /* not used yet      log_file_name  = (lgmres_data -> log_file_name);*/
      /* fp = fopen(log_file_name,"w"); */
   }

   /* initialize work arrays  - lgmres includes aug_dim*/
   rs = nalu_hypre_CTAllocF(NALU_HYPRE_Real, k_dim + 1 + aug_dim, lgmres_functions, NALU_HYPRE_MEMORY_HOST);
   c = nalu_hypre_CTAllocF(NALU_HYPRE_Real, k_dim + aug_dim, lgmres_functions, NALU_HYPRE_MEMORY_HOST);
   s = nalu_hypre_CTAllocF(NALU_HYPRE_Real, k_dim + aug_dim, lgmres_functions, NALU_HYPRE_MEMORY_HOST);

   /* lgmres mod. - need non-modified hessenberg to avoid aug_dim matvecs */
   hh = nalu_hypre_CTAllocF(NALU_HYPRE_Real*, k_dim + aug_dim + 1, lgmres_functions, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < k_dim + aug_dim + 1; i++)
   {
      hh[i] = nalu_hypre_CTAllocF(NALU_HYPRE_Real, k_dim + aug_dim, lgmres_functions, NALU_HYPRE_MEMORY_HOST);
   }

   (*(lgmres_functions->CopyVector))(b, p[0]);

   /* compute initial residual */
   (*(lgmres_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, p[0]);

   b_norm = nalu_hypre_sqrt((*(lgmres_functions->InnerProd))(b, b));

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
         nalu_hypre_printf("\n\nERROR detected by Hypre ... BEGIN\n");
         nalu_hypre_printf("ERROR -- nalu_hypre_LGMRESSolve: INFs and/or NaNs detected in input.\n");
         nalu_hypre_printf("User probably placed non-numerics in supplied b.\n");
         nalu_hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         nalu_hypre_printf("ERROR detected by Hypre ... END\n\n\n");
      }
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
      NALU_HYPRE_ANNOTATE_FUNC_END;

      return nalu_hypre_error_flag;
   }

   r_norm = nalu_hypre_sqrt((*(lgmres_functions->InnerProd))(p[0], p[0]));
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
         nalu_hypre_printf("\n\nERROR detected by Hypre ... BEGIN\n");
         nalu_hypre_printf("ERROR -- nalu_hypre_LGMRESSolve: INFs and/or NaNs detected in input.\n");
         nalu_hypre_printf("User probably placed non-numerics in supplied A or x_0.\n");
         nalu_hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         nalu_hypre_printf("ERROR detected by Hypre ... END\n\n\n");
      }
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
      NALU_HYPRE_ANNOTATE_FUNC_END;

      return nalu_hypre_error_flag;
   }

   if ( logging > 0 || print_level > 0)
   {
      norms[0] = r_norm;
      if ( print_level > 1 && my_id == 0 )
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
      /* convergence criterion |r_i|/|b| <= accuracy if |b| > 0 */
      den_norm = b_norm;
   }
   else
   {
      /* convergence criterion |r_i|/|r0| <= accuracy if |b| = 0 */
      den_norm = r_norm;
   };

   /* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
       den_norm = |r_0| or |b|
       note: default for a_tol is 0.0, so relative residual criteria is used unless
             user specifies a_tol, or sets r_tol = 0.0, which means absolute
             tol only is checked  */

   epsilon = nalu_hypre_max(a_tol, r_tol * den_norm);

   /* so now our stop criteria is |r_i| <= epsilon */


   if ( print_level > 1 && my_id == 0 )
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

      };
   }



   /*lgmres initialization */
   for (ii = 0; ii < aug_dim; ii++)
   {
      aug_order[ii] = 0;
   }
   aug_ct = 0; /* number of aug. vectors available */



   /* outer iteration cycle */
   while (iter < max_iter)
   {
      /* initialize first term of hessenberg system */

      rs[0] = r_norm;
      if (r_norm == 0.0)
      {
         nalu_hypre_TFreeF(c, lgmres_functions);
         nalu_hypre_TFreeF(s, lgmres_functions);
         nalu_hypre_TFreeF(rs, lgmres_functions);
         for (i = 0; i < k_dim + aug_dim + 1; i++)
         {
            nalu_hypre_TFreeF(hh[i], lgmres_functions);
         }

         nalu_hypre_TFreeF(hh, lgmres_functions);
         NALU_HYPRE_ANNOTATE_FUNC_END;

         return nalu_hypre_error_flag;
      }

      /* see if we are already converged and
         should print the final norm and exit */
      if (r_norm <= epsilon && iter >= min_iter)
      {
         (*(lgmres_functions->CopyVector))(b, r);
         (*(lgmres_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);
         r_norm = nalu_hypre_sqrt((*(lgmres_functions->InnerProd))(r, r));
         if (r_norm  <= epsilon)
         {
            if ( print_level > 1 && my_id == 0)
            {
               nalu_hypre_printf("\n\n");
               nalu_hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
            }
            break;
         }
         else if ( print_level > 0 && my_id == 0)
         {
            nalu_hypre_printf("false convergence 1\n");
         }

      }

      t = 1.0 / r_norm;
      r_norm_last = r_norm;

      (*(lgmres_functions->ScaleVector))(t, p[0]);
      i = 0;

      /* lgmres mod: determine number of arnoldi steps to take */
      /* if approx_constant then we keep the space the same size
         even if we don't have the full number of aug vectors yet*/
      if (approx_constant)
      {
         it_arnoldi = k_dim - aug_ct;
      }
      else
      {
         it_arnoldi = k_dim - aug_dim;
      }
      it_total =  it_arnoldi + aug_ct;
      it_aug = 0; /* keep track of augmented iterations */


      /***RESTART CYCLE (right-preconditioning) ***/
      while (i < it_total && iter < max_iter)
      {
         i++;
         iter++;
         (*(lgmres_functions->ClearVector))(r);


         /*LGMRES_MOD: decide whether this is an arnoldi step or an aug step */
         if ( i <= it_arnoldi)
         {
            /* Arnoldi */
            precond(precond_data, A, p[i - 1], r);
            (*(lgmres_functions->Matvec))(matvec_data, 1.0, A, r, 0.0, p[i]);
         }
         else
         {
            /*lgmres aug step */
            it_aug ++;
            order = i - it_arnoldi - 1; /* which aug step (note i starts at 1) - aug order number at 0*/
            for (ii = 0; ii < aug_dim; ii++)
            {
               if (aug_order[ii] == order)
               {
                  spot = ii;
                  break; /* must have this because there will be duplicates before aug_ct = aug_dim */
               }
            }
            /* copy a_aug_vecs[spot] to p[i] */
            (*(lgmres_functions->CopyVector))(a_aug_vecs[spot], p[i]);

            /*note: an alternate implementation choice would be to only save the AUGVECS and
              not A_AUGVEC and then apply the PC here to the augvec */
         }
         /*---*/

         /* modified Gram_Schmidt */
         for (j = 0; j < i; j++)
         {
            hh[j][i - 1] = (*(lgmres_functions->InnerProd))(p[j], p[i]);
            (*(lgmres_functions->Axpy))(-hh[j][i - 1], p[j], p[i]);
         }
         t = nalu_hypre_sqrt((*(lgmres_functions->InnerProd))(p[i], p[i]));
         hh[i][i - 1] = t;
         if (t != 0.0)
         {
            t = 1.0 / t;
            (*(lgmres_functions->ScaleVector))(t, p[i]);
         }


         /* done with modified Gram_schmidt and Arnoldi step.
            update factorization of hh */
         for (j = 1; j < i; j++)
         {
            t = hh[j - 1][i - 1];
            hh[j - 1][i - 1] = s[j - 1] * hh[j][i - 1] + c[j - 1] * t;
            hh[j][i - 1] = -s[j - 1] * t + c[j - 1] * hh[j][i - 1];
         }
         t = hh[i][i - 1] * hh[i][i - 1];
         t += hh[i - 1][i - 1] * hh[i - 1][i - 1];
         gamma = nalu_hypre_sqrt(t);
         if (gamma == 0.0) { gamma = epsmac; }
         c[i - 1] = hh[i - 1][i - 1] / gamma;
         s[i - 1] = hh[i][i - 1] / gamma;
         rs[i] = -hh[i][i - 1] * rs[i - 1];
         rs[i] /=  gamma;
         rs[i - 1] = c[i - 1] * rs[i - 1];
         /* determine residual norm */
         hh[i - 1][i - 1] = s[i - 1] * hh[i][i - 1] + c[i - 1] * hh[i - 1][i - 1];
         r_norm = nalu_hypre_abs(rs[i]);

         /* print ? */
         if ( print_level > 0 )
         {
            norms[iter] = r_norm;
            if ( print_level > 1 && my_id == 0 )
            {
               if (b_norm > 0.0)
                  nalu_hypre_printf("% 5d    %e    %f   %e\n", iter,
                               norms[iter], norms[iter] / norms[iter - 1],
                               norms[iter] / b_norm);
               else
                  nalu_hypre_printf("% 5d    %e    %f\n", iter, norms[iter],
                               norms[iter] / norms[iter - 1]);
            }
         }
         /*convergence factor tolerance */
         if (cf_tol > 0.0)
         {
            cf_ave_0 = cf_ave_1;
            cf_ave_1 = nalu_hypre_pow( r_norm / r_norm_0, 1.0 / (2.0 * iter));

            weight   = nalu_hypre_abs(cf_ave_1 - cf_ave_0);
            weight   = weight / nalu_hypre_max(cf_ave_1, cf_ave_0);
            weight   = 1.0 - weight;
#if 0
            nalu_hypre_printf("I = %d: cf_new = %e, cf_old = %e, weight = %e\n",
                         i, cf_ave_1, cf_ave_0, weight );
#endif
            if (weight * cf_ave_1 > cf_tol)
            {
               break_value = 1;
               break;
            }
         }
         /* should we exit the restart cycle? (conv. check) */
         if (r_norm <= epsilon && iter >= min_iter)
         {
            break;
         }


      } /*** end of restart cycle ***/

      /* now compute solution, first solve upper triangular system */

      if (break_value) { break; }

      rs[i - 1] = rs[i - 1] / hh[i - 1][i - 1];
      for (k = i - 2; k >= 0; k--)
      {
         t = 0.0;
         for (j = k + 1; j < i; j++)
         {
            t -= hh[k][j] * rs[j];
         }
         t += rs[k];
         rs[k] = t / hh[k][k];
      }
      /* form linear combination of p's to get solution */
      /* put the new aug_vector in aug_vecs[aug_dim]  - a temp position*/
      /* i = number of iterations */
      /* it_aug = number of augmented iterations */
      /* it_arnoldi = number of arnoldi iterations */


      /*check if exited early before all arnoldi its */
      if (it_arnoldi > i) { it_arnoldi = i; }


      if (!it_aug)
      {
         (*(lgmres_functions->CopyVector))(p[i - 1], w);
         (*(lgmres_functions->ScaleVector))(rs[i - 1], w);
         for (j = i - 2; j >= 0; j--)
         {
            (*(lgmres_functions->Axpy))(rs[j], p[j], w);
         }
      }
      else /* need some of the augvecs */
      {
         (*(lgmres_functions->CopyVector))(p[0], w);
         (*(lgmres_functions->ScaleVector))(rs[0], w);

         /* reg. arnoldi directions */
         for (j = 1; j < it_arnoldi; j++) /*first one already done */
         {
            (*(lgmres_functions->Axpy))(rs[j], p[j], w);
         }

         /* augment directions */
         for (ii = 0; ii < it_aug; ii++)
         {
            for (j = 0; j < aug_dim; j++)
            {
               if (aug_order[j] == ii)
               {
                  spot = j;
                  break; /* must have this because there will be
                            * duplicates before aug_ct = aug_dim */
               }
            }
            (*(lgmres_functions->Axpy))(rs[it_arnoldi + ii], aug_vecs[spot], w);
         }
      }


      /* grab the new aug vector before the prec*/
      (*(lgmres_functions->CopyVector))(w, aug_vecs[aug_dim]);

      (*(lgmres_functions->ClearVector))(r);
      /* find correction (in r) (un-wind precond.)*/
      precond(precond_data, A, w, r);

      /* update current solution x (in x) */
      (*(lgmres_functions->Axpy))(1.0, r, x);


      /* check for convergence by evaluating the actual residual */
      if (r_norm <= epsilon && iter >= min_iter)
      {
         /* calculate actual residual norm*/
         (*(lgmres_functions->CopyVector))(b, r);
         (*(lgmres_functions->Matvec))(matvec_data, -1.0, A, x, 1.0, r);
         r_norm = nalu_hypre_sqrt( (*(lgmres_functions->InnerProd))(r, r) );

         if (r_norm <= epsilon)
         {
            if ( print_level > 1 && my_id == 0 )
            {
               nalu_hypre_printf("\n\n");
               nalu_hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
            }
            (lgmres_data -> converged) = 1;
            break;
         }
         else /* conv. has not occurred, according to true residual */
         {
            if ( print_level > 0 && my_id == 0)
            {
               nalu_hypre_printf("false convergence 2\n");
            }
            (*(lgmres_functions->CopyVector))(r, p[0]);
            i = 0;
         }
      } /* end of convergence check */

      /* compute residual vector and continue loop */

      /* copy r0 (not scaled) to w*/
      (*(lgmres_functions->CopyVector))(p[0], w);
      (*(lgmres_functions->ScaleVector))(r_norm_last, w);


      for (j = i ; j > 0; j--)
      {
         rs[j - 1] = -s[j - 1] * rs[j];
         rs[j] = c[j - 1] * rs[j];
      }

      if (i) { (*(lgmres_functions->Axpy))(rs[i] - 1.0, p[i], p[i]); }
      for (j = i - 1 ; j > 0; j--)
      {
         (*(lgmres_functions->Axpy))(rs[j], p[j], p[i]);
      }

      if (i)
      {
         (*(lgmres_functions->Axpy))(rs[0] - 1.0, p[0], p[0]);
         (*(lgmres_functions->Axpy))(1.0, p[i], p[0]);
      }

      /* lgmres mod */
      /* collect aug vector and A*augvector for future restarts -
         only if we will be restarting (i.e. this cycle performed it_total
         iterations). ordering starts at 0.*/
      if (aug_dim > 0)
      {
         if (!aug_ct)
         {
            spot = 0;
            aug_ct++;
         }
         else if (aug_ct < aug_dim)
         {
            spot = aug_ct;
            aug_ct++;
         }
         else
         {
            /* truncate - already have aug_dim number of vectors*/
            for (ii = 0; ii < aug_dim; ii++)
            {
               if (aug_order[ii] == (aug_dim - 1))
               {
                  spot = ii;
               }
            }
         }
         /* aug_vecs[aug_dim] contains new aug vector */
         (*(lgmres_functions->CopyVector))(aug_vecs[aug_dim], aug_vecs[spot]);
         /*need to normalize */
         tmp_norm = nalu_hypre_sqrt((*(lgmres_functions->InnerProd))(aug_vecs[spot], aug_vecs[spot]));

         tmp_norm = 1.0 / tmp_norm;
         (*(lgmres_functions->ScaleVector))(tmp_norm, aug_vecs[spot]);

         /*set new aug vector to order 0  - move all others back one */
         for (ii = 0; ii < aug_dim; ii++)
         {
            aug_order[ii]++;
         }
         aug_order[spot] = 0;

         /*now add the A*aug vector to A_AUGVEC(spot) - this is
          * independ. of preconditioning type*/
         /* A*augvec = V*H*y  = r0-rm   (r0 is in w and rm is in p[0])*/
         (*(lgmres_functions->CopyVector))( w, a_aug_vecs[spot]);
         (*(lgmres_functions->ScaleVector))(- 1.0, a_aug_vecs[spot]); /* -r0*/
         (*(lgmres_functions->Axpy))(1.0, p[0], a_aug_vecs[spot]); /* rm - r0 */
         (*(lgmres_functions->ScaleVector))(-tmp_norm, a_aug_vecs[spot]); /* r0-rm /norm */

      }

   } /* END of iteration while loop */


   if ( print_level > 1 && my_id == 0 )
   {
      nalu_hypre_printf("\n\n");
   }

   (lgmres_data -> num_iterations) = iter;
   if (b_norm > 0.0)
   {
      (lgmres_data -> rel_residual_norm) = r_norm / b_norm;
   }
   if (b_norm == 0.0)
   {
      (lgmres_data -> rel_residual_norm) = r_norm;
   }

   if (iter >= max_iter && r_norm > epsilon && epsilon > 0) { nalu_hypre_error(NALU_HYPRE_ERROR_CONV); }


   nalu_hypre_TFreeF(c, lgmres_functions);
   nalu_hypre_TFreeF(s, lgmres_functions);
   nalu_hypre_TFreeF(rs, lgmres_functions);

   for (i = 0; i < k_dim + 1 + aug_dim; i++)
   {
      nalu_hypre_TFreeF(hh[i], lgmres_functions);
   }
   nalu_hypre_TFreeF(hh, lgmres_functions);

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESSetKDim, nalu_hypre_LGMRESGetKDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESSetKDim( void   *lgmres_vdata,
                     NALU_HYPRE_Int   k_dim )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   (lgmres_data -> k_dim) = k_dim;

   return nalu_hypre_error_flag;

}

NALU_HYPRE_Int
nalu_hypre_LGMRESGetKDim( void   *lgmres_vdata,
                     NALU_HYPRE_Int * k_dim )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   *k_dim = (lgmres_data -> k_dim);

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESSetAugDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESSetAugDim( void   *lgmres_vdata,
                       NALU_HYPRE_Int   aug_dim )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;

   if (aug_dim < 0) { aug_dim = 0; } /* must be positive */

   if (aug_dim > (lgmres_data -> k_dim) - 1) /* must be be <= (restart size-1) */
   {
      while (aug_dim > (lgmres_data -> k_dim) - 1)
      {
         aug_dim--;
      }

      aug_dim = (((0) < (aug_dim)) ? (aug_dim) : (0));

   }
   (lgmres_data -> aug_dim) = aug_dim;

   return nalu_hypre_error_flag;
}
NALU_HYPRE_Int
nalu_hypre_LGMRESGetAugDim( void   *lgmres_vdata,
                       NALU_HYPRE_Int * aug_dim )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   *aug_dim = (lgmres_data -> aug_dim);

   return nalu_hypre_error_flag;
}



/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESSetTol, nalu_hypre_LGMRESGetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESSetTol( void   *lgmres_vdata,
                    NALU_HYPRE_Real  tol       )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   (lgmres_data -> tol) = tol;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_LGMRESGetTol( void   *lgmres_vdata,
                    NALU_HYPRE_Real  * tol      )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   *tol = (lgmres_data -> tol);

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESSetAbsoluteTol, nalu_hypre_LGMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESSetAbsoluteTol( void   *lgmres_vdata,
                            NALU_HYPRE_Real  a_tol       )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   (lgmres_data -> a_tol) = a_tol;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_LGMRESGetAbsoluteTol( void   *lgmres_vdata,
                            NALU_HYPRE_Real  * a_tol      )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   *a_tol = (lgmres_data -> a_tol);

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESSetConvergenceFactorTol, nalu_hypre_LGMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESSetConvergenceFactorTol( void   *lgmres_vdata,
                                     NALU_HYPRE_Real  cf_tol       )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   (lgmres_data -> cf_tol) = cf_tol;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_LGMRESGetConvergenceFactorTol( void   *lgmres_vdata,
                                     NALU_HYPRE_Real * cf_tol       )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   *cf_tol = (lgmres_data -> cf_tol);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESSetMinIter, nalu_hypre_LGMRESGetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESSetMinIter( void *lgmres_vdata,
                        NALU_HYPRE_Int   min_iter  )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   (lgmres_data -> min_iter) = min_iter;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_LGMRESGetMinIter( void *lgmres_vdata,
                        NALU_HYPRE_Int * min_iter  )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   *min_iter = (lgmres_data -> min_iter);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESSetMaxIter, nalu_hypre_LGMRESGetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESSetMaxIter( void *lgmres_vdata,
                        NALU_HYPRE_Int   max_iter  )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   (lgmres_data -> max_iter) = max_iter;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_LGMRESGetMaxIter( void *lgmres_vdata,
                        NALU_HYPRE_Int * max_iter  )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   *max_iter = (lgmres_data -> max_iter);

   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESSetStopCrit, nalu_hypre_LGMRESGetStopCrit
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESSetStopCrit( void   *lgmres_vdata,
                         NALU_HYPRE_Int  stop_crit       )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   (lgmres_data -> stop_crit) = stop_crit;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_LGMRESGetStopCrit( void   *lgmres_vdata,
                         NALU_HYPRE_Int * stop_crit       )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   *stop_crit = (lgmres_data -> stop_crit);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESSetPrecond( void  *lgmres_vdata,
                        NALU_HYPRE_Int  (*precond)(void*, void*, void*, void*),
                        NALU_HYPRE_Int  (*precond_setup)(void*, void*, void*, void*),
                        void  *precond_data )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;
   nalu_hypre_LGMRESFunctions *lgmres_functions = lgmres_data->functions;


   (lgmres_functions -> precond)        = precond;
   (lgmres_functions -> precond_setup)  = precond_setup;
   (lgmres_data -> precond_data)   = precond_data;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESGetPrecond( void         *lgmres_vdata,
                        NALU_HYPRE_Solver *precond_data_ptr )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   *precond_data_ptr = (NALU_HYPRE_Solver)(lgmres_data -> precond_data);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESSetPrintLevel, nalu_hypre_LGMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESSetPrintLevel( void *lgmres_vdata,
                           NALU_HYPRE_Int   level)
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   (lgmres_data -> print_level) = level;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_LGMRESGetPrintLevel( void *lgmres_vdata,
                           NALU_HYPRE_Int * level)
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   *level = (lgmres_data -> print_level);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESSetLogging, nalu_hypre_LGMRESGetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESSetLogging( void *lgmres_vdata,
                        NALU_HYPRE_Int   level)
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   (lgmres_data -> logging) = level;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_LGMRESGetLogging( void *lgmres_vdata,
                        NALU_HYPRE_Int * level)
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   *level = (lgmres_data -> logging);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESGetNumIterations( void *lgmres_vdata,
                              NALU_HYPRE_Int  *num_iterations )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   *num_iterations = (lgmres_data -> num_iterations);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESGetConverged
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESGetConverged( void *lgmres_vdata,
                          NALU_HYPRE_Int  *converged )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   *converged = (lgmres_data -> converged);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_LGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_LGMRESGetFinalRelativeResidualNorm( void   *lgmres_vdata,
                                          NALU_HYPRE_Real *relative_residual_norm )
{
   nalu_hypre_LGMRESData *lgmres_data = (nalu_hypre_LGMRESData *)lgmres_vdata;


   *relative_residual_norm = (lgmres_data -> rel_residual_norm);

   return nalu_hypre_error_flag;
}
