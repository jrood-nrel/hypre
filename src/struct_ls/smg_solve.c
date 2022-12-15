/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "smg.h"

#define DEBUG 0

/*--------------------------------------------------------------------------
 * This is the main solve routine for the Schaffer multigrid method.
 * This solver works for 1D, 2D, or 3D linear systems.  The dimension
 * is determined by the nalu_hypre_StructStencilNDim argument of the matrix
 * stencil.  The nalu_hypre_StructGridNDim argument of the matrix grid is
 * allowed to be larger than the dimension of the solver, and in fact,
 * this feature is used in the smaller-dimensional solves required
 * in the relaxation method for both the 2D and 3D algorithms.  This
 * allows one to do multiple 2D or 1D solves in parallel (e.g., multiple
 * 2D solves, where the 2D problems are "stacked" planes in 3D).
 * The only additional requirement is that the linear system(s) data
 * be contiguous in memory.
 *
 * Notes:
 * - Iterations are counted as follows: 1 iteration consists of a
 *   V-cycle plus an extra pre-relaxation.  If the number of MG levels
 *   is equal to 1, then only the extra pre-relaxation step is done at
 *   each iteration.  When the solver exits because the maximum number
 *   of iterations is reached, the last extra pre-relaxation is not done.
 *   This allows one to use the solver as a preconditioner for conjugate
 *   gradient and insure symmetry.
 * - nalu_hypre_SMGRelax is the relaxation routine.  There are different "data"
 *   structures for each call to reflect different arguments and parameters.
 *   One important parameter sets whether or not an initial guess of zero
 *   is to be used in the relaxation.
 * - nalu_hypre_SMGResidual computes the residual, b - Ax.
 * - nalu_hypre_SemiRestrict restricts the residual to the coarse grid.
 * - nalu_hypre_SemiInterp interpolates the coarse error and adds it to the
 *   fine grid solution.
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSolve( void               *smg_vdata,
                nalu_hypre_StructMatrix *A,
                nalu_hypre_StructVector *b,
                nalu_hypre_StructVector *x         )
{

   nalu_hypre_SMGData        *smg_data = (nalu_hypre_SMGData        *)smg_vdata;

   NALU_HYPRE_Real            tol             = (smg_data -> tol);
   NALU_HYPRE_Int             max_iter        = (smg_data -> max_iter);
   NALU_HYPRE_Int             rel_change      = (smg_data -> rel_change);
   NALU_HYPRE_Int             zero_guess      = (smg_data -> zero_guess);
   NALU_HYPRE_Int             num_levels      = (smg_data -> num_levels);
   NALU_HYPRE_Int             num_pre_relax   = (smg_data -> num_pre_relax);
   NALU_HYPRE_Int             num_post_relax  = (smg_data -> num_post_relax);
   nalu_hypre_IndexRef        base_index      = (smg_data -> base_index);
   nalu_hypre_IndexRef        base_stride     = (smg_data -> base_stride);
   nalu_hypre_StructMatrix  **A_l             = (smg_data -> A_l);
   nalu_hypre_StructMatrix  **PT_l            = (smg_data -> PT_l);
   nalu_hypre_StructMatrix  **R_l             = (smg_data -> R_l);
   nalu_hypre_StructVector  **b_l             = (smg_data -> b_l);
   nalu_hypre_StructVector  **x_l             = (smg_data -> x_l);
   nalu_hypre_StructVector  **r_l             = (smg_data -> r_l);
   nalu_hypre_StructVector  **e_l             = (smg_data -> e_l);
   void                **relax_data_l    = (smg_data -> relax_data_l);
   void                **residual_data_l = (smg_data -> residual_data_l);
   void                **restrict_data_l = (smg_data -> restrict_data_l);
   void                **interp_data_l   = (smg_data -> interp_data_l);
   NALU_HYPRE_Int             logging         = (smg_data -> logging);
   NALU_HYPRE_Real           *norms           = (smg_data -> norms);
   NALU_HYPRE_Real           *rel_norms       = (smg_data -> rel_norms);

   NALU_HYPRE_Real            b_dot_b = 0, r_dot_r, eps = 0;
   NALU_HYPRE_Real            e_dot_e = 0, x_dot_x = 1;

   NALU_HYPRE_Int             i, l;

#if DEBUG
   char                  filename[255];
#endif

   /*-----------------------------------------------------
    * Initialize some things and deal with special cases
    *-----------------------------------------------------*/

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;
   nalu_hypre_BeginTiming(smg_data -> time_index);

   nalu_hypre_StructMatrixDestroy(A_l[0]);
   nalu_hypre_StructVectorDestroy(b_l[0]);
   nalu_hypre_StructVectorDestroy(x_l[0]);
   A_l[0] = nalu_hypre_StructMatrixRef(A);
   b_l[0] = nalu_hypre_StructVectorRef(b);
   x_l[0] = nalu_hypre_StructVectorRef(x);

   (smg_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         nalu_hypre_StructVectorSetConstantValues(x, 0.0);
      }

      nalu_hypre_EndTiming(smg_data -> time_index);
      NALU_HYPRE_ANNOTATE_FUNC_END;

      return nalu_hypre_error_flag;
   }

   /* part of convergence check */
   if (tol > 0.0)
   {
      /* eps = (tol^2) */
      b_dot_b = nalu_hypre_StructInnerProd(b_l[0], b_l[0]);
      eps = tol * tol;

      /* if rhs is zero, return a zero solution */
      if (b_dot_b == 0.0)
      {
         nalu_hypre_StructVectorSetConstantValues(x, 0.0);
         if (logging > 0)
         {
            norms[0]     = 0.0;
            rel_norms[0] = 0.0;
         }

         nalu_hypre_EndTiming(smg_data -> time_index);
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

      /* fine grid pre-relaxation */
      if (num_levels > 1)
      {
         nalu_hypre_SMGRelaxSetRegSpaceRank(relax_data_l[0], 0, 0);
         nalu_hypre_SMGRelaxSetRegSpaceRank(relax_data_l[0], 1, 1);
      }
      nalu_hypre_SMGRelaxSetMaxIter(relax_data_l[0], num_pre_relax);
      nalu_hypre_SMGRelaxSetZeroGuess(relax_data_l[0], zero_guess);
      nalu_hypre_SMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);
      zero_guess = 0;

      /* compute fine grid residual (b - Ax) */
      nalu_hypre_SMGResidual(residual_data_l[0], A_l[0], x_l[0], b_l[0], r_l[0]);

      /* convergence check */
      if (tol > 0.0)
      {
         r_dot_r = nalu_hypre_StructInnerProd(r_l[0], r_l[0]);
         if (logging > 0)
         {
            norms[i] = sqrt(r_dot_r);
            if (b_dot_b > 0)
            {
               rel_norms[i] = sqrt(r_dot_r / b_dot_b);
            }
            else
            {
               rel_norms[i] = 0.0;
            }
         }

         /* always do at least 1 V-cycle */
         if ((r_dot_r / b_dot_b < eps) && (i > 0))
         {
            if (rel_change)
            {
               if ((e_dot_e / x_dot_x) < eps)
               {
                  break;
               }
            }
            else
            {
               break;
            }
         }
      }

      if (num_levels > 1)
      {
         /* restrict fine grid residual */
         nalu_hypre_SemiRestrict(restrict_data_l[0], R_l[0], r_l[0], b_l[1]);
#if DEBUG
         if (nalu_hypre_StructStencilNDim(nalu_hypre_StructMatrixStencil(A)) == 3)
         {
            nalu_hypre_sprintf(filename, "zout_xdown.%02d", 0);
            nalu_hypre_StructVectorPrint(filename, x_l[0], 0);
            nalu_hypre_sprintf(filename, "zout_rdown.%02d", 0);
            nalu_hypre_StructVectorPrint(filename, r_l[0], 0);
            nalu_hypre_sprintf(filename, "zout_b.%02d", 1);
            nalu_hypre_StructVectorPrint(filename, b_l[1], 0);
         }
#endif
         for (l = 1; l <= (num_levels - 2); l++)
         {
            /* pre-relaxation */
            nalu_hypre_SMGRelaxSetRegSpaceRank(relax_data_l[l], 0, 0);
            nalu_hypre_SMGRelaxSetRegSpaceRank(relax_data_l[l], 1, 1);
            nalu_hypre_SMGRelaxSetMaxIter(relax_data_l[l], num_pre_relax);
            nalu_hypre_SMGRelaxSetZeroGuess(relax_data_l[l], 1);
            nalu_hypre_SMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);

            /* compute residual (b - Ax) */
            nalu_hypre_SMGResidual(residual_data_l[l],
                              A_l[l], x_l[l], b_l[l], r_l[l]);

            /* restrict residual */
            nalu_hypre_SemiRestrict(restrict_data_l[l], R_l[l], r_l[l], b_l[l + 1]);
#if DEBUG
            if (nalu_hypre_StructStencilNDim(nalu_hypre_StructMatrixStencil(A)) == 3)
            {
               nalu_hypre_sprintf(filename, "zout_xdown.%02d", l);
               nalu_hypre_StructVectorPrint(filename, x_l[l], 0);
               nalu_hypre_sprintf(filename, "zout_rdown.%02d", l);
               nalu_hypre_StructVectorPrint(filename, r_l[l], 0);
               nalu_hypre_sprintf(filename, "zout_b.%02d", l + 1);
               nalu_hypre_StructVectorPrint(filename, b_l[l + 1], 0);
            }
#endif
         }

         /*--------------------------------------------------
          * Bottom
          *--------------------------------------------------*/

         nalu_hypre_SMGRelaxSetZeroGuess(relax_data_l[l], 1);
         nalu_hypre_SMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
#if DEBUG
         if (nalu_hypre_StructStencilNDim(nalu_hypre_StructMatrixStencil(A)) == 3)
         {
            nalu_hypre_sprintf(filename, "zout_xbottom.%02d", l);
            nalu_hypre_StructVectorPrint(filename, x_l[l], 0);
         }
#endif

         /*--------------------------------------------------
          * Up cycle
          *--------------------------------------------------*/

         for (l = (num_levels - 2); l >= 1; l--)
         {
            /* interpolate error and correct (x = x + Pe_c) */
            nalu_hypre_SemiInterp(interp_data_l[l], PT_l[l], x_l[l + 1], e_l[l]);
            nalu_hypre_StructAxpy(1.0, e_l[l], x_l[l]);
#if DEBUG
            if (nalu_hypre_StructStencilNDim(nalu_hypre_StructMatrixStencil(A)) == 3)
            {
               nalu_hypre_sprintf(filename, "zout_eup.%02d", l);
               nalu_hypre_StructVectorPrint(filename, e_l[l], 0);
               nalu_hypre_sprintf(filename, "zout_xup.%02d", l);
               nalu_hypre_StructVectorPrint(filename, x_l[l], 0);
            }
#endif
            /* post-relaxation */
            nalu_hypre_SMGRelaxSetRegSpaceRank(relax_data_l[l], 0, 1);
            nalu_hypre_SMGRelaxSetRegSpaceRank(relax_data_l[l], 1, 0);
            nalu_hypre_SMGRelaxSetMaxIter(relax_data_l[l], num_post_relax);
            nalu_hypre_SMGRelaxSetZeroGuess(relax_data_l[l], 0);
            nalu_hypre_SMGRelax(relax_data_l[l], A_l[l], b_l[l], x_l[l]);
         }

         /* interpolate error and correct on fine grid (x = x + Pe_c) */
         nalu_hypre_SemiInterp(interp_data_l[0], PT_l[0], x_l[1], e_l[0]);
         nalu_hypre_SMGAxpy(1.0, e_l[0], x_l[0], base_index, base_stride);
#if DEBUG
         if (nalu_hypre_StructStencilNDim(nalu_hypre_StructMatrixStencil(A)) == 3)
         {
            nalu_hypre_sprintf(filename, "zout_eup.%02d", 0);
            nalu_hypre_StructVectorPrint(filename, e_l[0], 0);
            nalu_hypre_sprintf(filename, "zout_xup.%02d", 0);
            nalu_hypre_StructVectorPrint(filename, x_l[0], 0);
         }
#endif
      }

      /* part of convergence check */
      if ((tol > 0.0) && (rel_change))
      {
         if (num_levels > 1)
         {
            e_dot_e = nalu_hypre_StructInnerProd(e_l[0], e_l[0]);
            x_dot_x = nalu_hypre_StructInnerProd(x_l[0], x_l[0]);
         }
         else
         {
            e_dot_e = 0.0;
            x_dot_x = 1.0;
         }
      }

      /* fine grid post-relaxation */
      if (num_levels > 1)
      {
         nalu_hypre_SMGRelaxSetRegSpaceRank(relax_data_l[0], 0, 1);
         nalu_hypre_SMGRelaxSetRegSpaceRank(relax_data_l[0], 1, 0);
      }
      nalu_hypre_SMGRelaxSetMaxIter(relax_data_l[0], num_post_relax);
      nalu_hypre_SMGRelaxSetZeroGuess(relax_data_l[0], 0);
      nalu_hypre_SMGRelax(relax_data_l[0], A_l[0], b_l[0], x_l[0]);

      (smg_data -> num_iterations) = (i + 1);
   }

   nalu_hypre_EndTiming(smg_data -> time_index);
   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}
