/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "sys_pfmg.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSolve( void                 *sys_pfmg_vdata,
                    nalu_hypre_SStructMatrix  *A_in,
                    nalu_hypre_SStructVector  *b_in,
                    nalu_hypre_SStructVector  *x_in         )
{
   nalu_hypre_SysPFMGData       *sys_pfmg_data = (nalu_hypre_SysPFMGData*)sys_pfmg_vdata;

   nalu_hypre_SStructPMatrix *A;
   nalu_hypre_SStructPVector *b;
   nalu_hypre_SStructPVector *x;

   NALU_HYPRE_Real            tol             = (sys_pfmg_data -> tol);
   NALU_HYPRE_Int             max_iter        = (sys_pfmg_data -> max_iter);
   NALU_HYPRE_Int             rel_change      = (sys_pfmg_data -> rel_change);
   NALU_HYPRE_Int             zero_guess      = (sys_pfmg_data -> zero_guess);
   NALU_HYPRE_Int             num_pre_relax   = (sys_pfmg_data -> num_pre_relax);
   NALU_HYPRE_Int             num_post_relax  = (sys_pfmg_data -> num_post_relax);
   NALU_HYPRE_Int             num_levels      = (sys_pfmg_data -> num_levels);
   nalu_hypre_SStructPMatrix  **A_l           = (sys_pfmg_data -> A_l);
   nalu_hypre_SStructPMatrix  **P_l           = (sys_pfmg_data -> P_l);
   nalu_hypre_SStructPMatrix  **RT_l          = (sys_pfmg_data -> RT_l);
   nalu_hypre_SStructPVector  **b_l           = (sys_pfmg_data -> b_l);
   nalu_hypre_SStructPVector  **x_l           = (sys_pfmg_data -> x_l);
   nalu_hypre_SStructPVector  **r_l           = (sys_pfmg_data -> r_l);
   nalu_hypre_SStructPVector  **e_l           = (sys_pfmg_data -> e_l);
   void                **relax_data_l    = (sys_pfmg_data -> relax_data_l);
   void                **matvec_data_l   = (sys_pfmg_data -> matvec_data_l);
   void                **restrict_data_l = (sys_pfmg_data -> restrict_data_l);
   void                **interp_data_l   = (sys_pfmg_data -> interp_data_l);
   NALU_HYPRE_Int             logging         = (sys_pfmg_data -> logging);
   NALU_HYPRE_Real           *norms           = (sys_pfmg_data -> norms);
   NALU_HYPRE_Real           *rel_norms       = (sys_pfmg_data -> rel_norms);
   NALU_HYPRE_Int            *active_l        = (sys_pfmg_data -> active_l);

   NALU_HYPRE_Real            b_dot_b, r_dot_r, eps = 0;
   NALU_HYPRE_Real            e_dot_e = 0, x_dot_x = 1;

   NALU_HYPRE_Int             i, l;

#if DEBUG
   char                  filename[255];
#endif

   /*-----------------------------------------------------
    * Initialize some things and deal with special cases
    *-----------------------------------------------------*/

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;
   nalu_hypre_BeginTiming(sys_pfmg_data -> time_index);

   /*-----------------------------------------------------
    * Refs to A,x,b (the PMatrix & PVectors within
    * the input SStructMatrix & SStructVectors)
    *-----------------------------------------------------*/
   nalu_hypre_SStructPMatrixRef(nalu_hypre_SStructMatrixPMatrix(A_in, 0), &A);
   nalu_hypre_SStructPVectorRef(nalu_hypre_SStructVectorPVector(b_in, 0), &b);
   nalu_hypre_SStructPVectorRef(nalu_hypre_SStructVectorPVector(x_in, 0), &x);


   nalu_hypre_SStructPMatrixDestroy(A_l[0]);
   nalu_hypre_SStructPVectorDestroy(b_l[0]);
   nalu_hypre_SStructPVectorDestroy(x_l[0]);
   nalu_hypre_SStructPMatrixRef(A, &A_l[0]);
   nalu_hypre_SStructPVectorRef(b, &b_l[0]);
   nalu_hypre_SStructPVectorRef(x, &x_l[0]);


   (sys_pfmg_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         nalu_hypre_SStructPVectorSetConstantValues(x, 0.0);
      }

      nalu_hypre_EndTiming(sys_pfmg_data -> time_index);
      NALU_HYPRE_ANNOTATE_FUNC_END;

      return nalu_hypre_error_flag;
   }

   /* part of convergence check */
   if (tol > 0.0)
   {
      /* eps = (tol^2) */
      nalu_hypre_SStructPInnerProd(b_l[0], b_l[0], &b_dot_b);
      eps = tol * tol;

      /* if rhs is zero, return a zero solution */
      if (b_dot_b == 0.0)
      {
         nalu_hypre_SStructPVectorSetConstantValues(x, 0.0);
         if (logging > 0)
         {
            norms[0]     = 0.0;
            rel_norms[0] = 0.0;
         }

         nalu_hypre_EndTiming(sys_pfmg_data -> time_index);
         NALU_HYPRE_ANNOTATE_FUNC_END;

         return nalu_hypre_error_flag;
      }
   }

   /*-----------------------------------------------------
    * Do V-cycles:
    *   For each index l, "fine" = l, "coarse" = (l+1)
    *-----------------------------------------------------*/

   for (i = 0; i < max_iter; i++)
   {
      /*--------------------------------------------------
       * Down cycle
       *--------------------------------------------------*/
      NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(0);

      /* fine grid pre-relaxation */
      nalu_hypre_SysPFMGRelaxSetPreRelax(relax_data_l[0]);
      nalu_hypre_SysPFMGRelaxSetMaxIter(relax_data_l[0], num_pre_relax);
      nalu_hypre_SysPFMGRelaxSetZeroGuess(relax_data_l[0], zero_guess);
      nalu_hypre_SysPFMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
      zero_guess = 0;

      /* compute fine grid residual (b - Ax) */
      nalu_hypre_SStructPCopy(b_l[0], r_l[0]);
      nalu_hypre_SStructPMatvecCompute(matvec_data_l[0],
                                  -1.0, A_l[0], x_l[0], 1.0, r_l[0]);

      /* convergence check */
      if (tol > 0.0)
      {
         nalu_hypre_SStructPInnerProd(r_l[0], r_l[0], &r_dot_r);

         if (logging > 0)
         {
            norms[i] = nalu_hypre_sqrt(r_dot_r);
            if (b_dot_b > 0)
            {
               rel_norms[i] = nalu_hypre_sqrt(r_dot_r / b_dot_b);
            }
            else
            {
               rel_norms[i] = 0.0;
            }
         }

         /* always do at least 1 V-cycle */
         if ((r_dot_r / b_dot_b < eps) && (i > 0))
         {
            if ( ((rel_change) && (e_dot_e / x_dot_x) < eps) || (!rel_change) )
            {
               NALU_HYPRE_ANNOTATE_MGLEVEL_END(0);
               break;
            }
         }
      }

      if (num_levels > 1)
      {
         /* restrict fine grid residual */
         nalu_hypre_SysSemiRestrict(restrict_data_l[0], RT_l[0], r_l[0], b_l[1]);
#if DEBUG
         nalu_hypre_sprintf(filename, "zout_xdown.%02d", 0);
         nalu_hypre_SStructPVectorPrint(filename, x_l[0], 0);
         nalu_hypre_sprintf(filename, "zout_rdown.%02d", 0);
         nalu_hypre_SStructPVectorPrint(filename, r_l[0], 0);
         nalu_hypre_sprintf(filename, "zout_b.%02d", 1);
         nalu_hypre_SStructPVectorPrint(filename, b_l[1], 0);
#endif
         NALU_HYPRE_ANNOTATE_MGLEVEL_END(0);

         for (l = 1; l <= (num_levels - 2); l++)
         {
            if (active_l[l])
            {
               NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);

               /* pre-relaxation */
               nalu_hypre_SysPFMGRelaxSetPreRelax(relax_data_l[l]);
               nalu_hypre_SysPFMGRelaxSetMaxIter(relax_data_l[l], num_pre_relax);
               nalu_hypre_SysPFMGRelaxSetZeroGuess(relax_data_l[l], 1);
               nalu_hypre_SysPFMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

               /* compute residual (b - Ax) */
               nalu_hypre_SStructPCopy(b_l[l], r_l[l]);
               nalu_hypre_SStructPMatvecCompute(matvec_data_l[l],
                                           -1.0, A_l[l], x_l[l], 1.0, r_l[l]);
            }
            else
            {
               /* inactive level, set x=0, so r=(b-Ax)=b */
               nalu_hypre_SStructPVectorSetConstantValues(x_l[l], 0.0);
               nalu_hypre_SStructPCopy(b_l[l], r_l[l]);
            }

            /* restrict residual */
            nalu_hypre_SysSemiRestrict(restrict_data_l[l],
                                  RT_l[l], r_l[l], b_l[l + 1]);
#if DEBUG
            nalu_hypre_sprintf(filename, "zout_xdown.%02d", l);
            nalu_hypre_SStructPVectorPrint(filename, x_l[l], 0);
            nalu_hypre_sprintf(filename, "zout_rdown.%02d", l);
            nalu_hypre_SStructPVectorPrint(filename, r_l[l], 0);
            nalu_hypre_sprintf(filename, "zout_RT.%02d", l);
            nalu_hypre_SStructPMatrixPrint(filename, RT_l[l], 0);
            nalu_hypre_sprintf(filename, "zout_b.%02d", l + 1);
            nalu_hypre_SStructPVectorPrint(filename, b_l[l + 1], 0);
#endif
            NALU_HYPRE_ANNOTATE_MGLEVEL_END(l);
         }

         /*--------------------------------------------------
          * Bottom
          *--------------------------------------------------*/
         NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(num_levels - 1);

         nalu_hypre_SysPFMGRelaxSetZeroGuess(relax_data_l[l], 1);
         nalu_hypre_SysPFMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
#if DEBUG
         nalu_hypre_sprintf(filename, "zout_xbottom.%02d", l);
         nalu_hypre_SStructPVectorPrint(filename, x_l[l], 0);
#endif

         /*--------------------------------------------------
          * Up cycle
          *--------------------------------------------------*/

         for (l = (num_levels - 2); l >= 1; l--)
         {
            /* interpolate error and correct (x = x + Pe_c) */
            nalu_hypre_SysSemiInterp(interp_data_l[l], P_l[l], x_l[l + 1], e_l[l]);
            nalu_hypre_SStructPAxpy(1.0, e_l[l], x_l[l]);
            NALU_HYPRE_ANNOTATE_MGLEVEL_END(l + 1);
#if DEBUG
            nalu_hypre_sprintf(filename, "zout_eup.%02d", l);
            nalu_hypre_SStructPVectorPrint(filename, e_l[l], 0);
            nalu_hypre_sprintf(filename, "zout_xup.%02d", l);
            nalu_hypre_SStructPVectorPrint(filename, x_l[l], 0);
#endif
            NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(l);

            if (active_l[l])
            {
               /* post-relaxation */
               nalu_hypre_SysPFMGRelaxSetPostRelax(relax_data_l[l]);
               nalu_hypre_SysPFMGRelaxSetMaxIter(relax_data_l[l], num_post_relax);
               nalu_hypre_SysPFMGRelaxSetZeroGuess(relax_data_l[l], 0);
               nalu_hypre_SysPFMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
            }
         }

         /* interpolate error and correct on fine grid (x = x + Pe_c) */
         nalu_hypre_SysSemiInterp(interp_data_l[0], P_l[0], x_l[1], e_l[0]);
         nalu_hypre_SStructPAxpy(1.0, e_l[0], x_l[0]);
         NALU_HYPRE_ANNOTATE_MGLEVEL_END(1);
#if DEBUG
         nalu_hypre_sprintf(filename, "zout_eup.%02d", 0);
         nalu_hypre_SStructPVectorPrint(filename, e_l[0], 0);
         nalu_hypre_sprintf(filename, "zout_xup.%02d", 0);
         nalu_hypre_SStructPVectorPrint(filename, x_l[0], 0);
#endif
         NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(0);
      }

      /* part of convergence check */
      if ((tol > 0.0) && (rel_change))
      {
         if (num_levels > 1)
         {
            nalu_hypre_SStructPInnerProd(e_l[0], e_l[0], &e_dot_e);
            nalu_hypre_SStructPInnerProd(x_l[0], x_l[0], &x_dot_x);
         }
      }
      /* fine grid post-relaxation */
      nalu_hypre_SysPFMGRelaxSetPostRelax(relax_data_l[0]);
      nalu_hypre_SysPFMGRelaxSetMaxIter(relax_data_l[0], num_post_relax);
      nalu_hypre_SysPFMGRelaxSetZeroGuess(relax_data_l[0], 0);
      nalu_hypre_SysPFMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
      (sys_pfmg_data -> num_iterations) = (i + 1);

      NALU_HYPRE_ANNOTATE_MGLEVEL_END(0);
   }

   /*-----------------------------------------------------
    * Destroy Refs to A,x,b (the PMatrix & PVectors within
    * the input SStructMatrix & SStructVectors).
    *-----------------------------------------------------*/
   nalu_hypre_SStructPMatrixDestroy(A);
   nalu_hypre_SStructPVectorDestroy(x);
   nalu_hypre_SStructPVectorDestroy(b);

   nalu_hypre_EndTiming(sys_pfmg_data -> time_index);
   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}
