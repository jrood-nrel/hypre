/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellSolve- note that there is no input operator Aee. We assume
 * that maxwell_vdata has the exact operators. This prevents the need to
 * to recompute Ann in the solve phase. However, we do allow the f_edge &
 * u_edge to change per call.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MaxwellSolve2( void                *maxwell_vdata,
                     nalu_hypre_SStructMatrix *A_in,
                     nalu_hypre_SStructVector *f,
                     nalu_hypre_SStructVector *u )
{
   nalu_hypre_MaxwellData     *maxwell_data = (nalu_hypre_MaxwellData     *)maxwell_vdata;

   nalu_hypre_ParVector       *f_edge;
   nalu_hypre_ParVector       *u_edge;

   NALU_HYPRE_Int              max_iter     = maxwell_data-> max_iter;
   NALU_HYPRE_Real             tol          = maxwell_data-> tol;
   NALU_HYPRE_Int              rel_change   = maxwell_data-> rel_change;
   NALU_HYPRE_Int              zero_guess   = maxwell_data-> zero_guess;
   NALU_HYPRE_Int              npre_relax   = maxwell_data-> num_pre_relax;
   NALU_HYPRE_Int              npost_relax  = maxwell_data-> num_post_relax;

   nalu_hypre_ParCSRMatrix   **Ann_l        = maxwell_data-> Ann_l;
   nalu_hypre_ParCSRMatrix   **Pn_l         = maxwell_data-> Pn_l;
   nalu_hypre_ParCSRMatrix   **RnT_l        = maxwell_data-> RnT_l;
   nalu_hypre_ParVector      **bn_l         = maxwell_data-> bn_l;
   nalu_hypre_ParVector      **xn_l         = maxwell_data-> xn_l;
   nalu_hypre_ParVector      **resn_l       = maxwell_data-> resn_l;
   nalu_hypre_ParVector      **en_l         = maxwell_data-> en_l;
   nalu_hypre_ParVector      **nVtemp2_l    = maxwell_data-> nVtemp2_l;
   NALU_HYPRE_Int            **nCF_marker_l = maxwell_data-> nCF_marker_l;
   NALU_HYPRE_Real            *nrelax_weight = maxwell_data-> nrelax_weight;
   NALU_HYPRE_Real            *nomega       = maxwell_data-> nomega;
   NALU_HYPRE_Int              nrelax_type  = maxwell_data-> nrelax_type;
   NALU_HYPRE_Int              node_numlevs = maxwell_data-> node_numlevels;

   nalu_hypre_ParCSRMatrix    *Tgrad        = maxwell_data-> Tgrad;
   nalu_hypre_ParCSRMatrix    *T_transpose  = maxwell_data-> T_transpose;

   nalu_hypre_ParCSRMatrix   **Aee_l        = maxwell_data-> Aee_l;
   nalu_hypre_IJMatrix       **Pe_l         = maxwell_data-> Pe_l;
   nalu_hypre_IJMatrix       **ReT_l        = maxwell_data-> ReT_l;
   nalu_hypre_ParVector      **be_l         = maxwell_data-> be_l;
   nalu_hypre_ParVector      **xe_l         = maxwell_data-> xe_l;
   nalu_hypre_ParVector      **rese_l       = maxwell_data-> rese_l;
   nalu_hypre_ParVector      **ee_l         = maxwell_data-> ee_l;
   nalu_hypre_ParVector      **eVtemp2_l    = maxwell_data-> eVtemp2_l;
   NALU_HYPRE_Int            **eCF_marker_l = maxwell_data-> eCF_marker_l;
   NALU_HYPRE_Real            *erelax_weight = maxwell_data-> erelax_weight;
   NALU_HYPRE_Real            *eomega       = maxwell_data-> eomega;
   NALU_HYPRE_Int              erelax_type  = maxwell_data-> erelax_type;
   NALU_HYPRE_Int              edge_numlevs = maxwell_data-> edge_numlevels;

   NALU_HYPRE_Int            **BdryRanks_l  = maxwell_data-> BdryRanks_l;
   NALU_HYPRE_Int             *BdryRanksCnts_l = maxwell_data-> BdryRanksCnts_l;

   NALU_HYPRE_Int              logging      = maxwell_data-> logging;
   NALU_HYPRE_Real            *norms        = maxwell_data-> norms;
   NALU_HYPRE_Real            *rel_norms    = maxwell_data-> rel_norms;

   NALU_HYPRE_Int              relax_local, cycle_param;

   NALU_HYPRE_Real             b_dot_b = 0, r_dot_r, eps = 0;
   NALU_HYPRE_Real             e_dot_e, x_dot_x;

   NALU_HYPRE_Int              i, j;
   NALU_HYPRE_Int              level;

   /* added for the relaxation routines */
   nalu_hypre_ParVector *ze = NULL;

   if (nalu_hypre_NumThreads() > 1)
   {
      /* Aee is always bigger than Ann */

      ze = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[0]),
                                 nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[0]),
                                 nalu_hypre_ParCSRMatrixRowStarts(Aee_l[0]));
      nalu_hypre_ParVectorInitialize(ze);
   }

   nalu_hypre_BeginTiming(maxwell_data-> time_index);

   nalu_hypre_SStructVectorConvert(f, &f_edge);
   nalu_hypre_SStructVectorConvert(u, &u_edge);
   nalu_hypre_ParVectorZeroBCValues(f_edge, BdryRanks_l[0], BdryRanksCnts_l[0]);
   nalu_hypre_ParVectorZeroBCValues(u_edge, BdryRanks_l[0], BdryRanksCnts_l[0]);
   be_l[0] = f_edge;
   xe_l[0] = u_edge;

   /* the nodal fine vectors: xn= 0. bn= T'*(be- Aee*xe) is updated in the cycle. */
   nalu_hypre_ParVectorSetConstantValues(xn_l[0], 0.0);

   relax_local = 0;
   cycle_param = 0;

   (maxwell_data-> num_iterations) = 0;
   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         nalu_hypre_ParVectorSetConstantValues(xe_l[0], 0.0);
      }

      nalu_hypre_EndTiming(maxwell_data -> time_index);

      return nalu_hypre_error_flag;
   }

   /* part of convergence check */
   if (tol > 0.0)
   {
      /* eps = (tol^2) */
      b_dot_b = nalu_hypre_ParVectorInnerProd(be_l[0], be_l[0]);
      eps = tol * tol;

      /* if rhs is zero, return a zero solution */
      if (b_dot_b == 0.0)
      {
         nalu_hypre_ParVectorSetConstantValues(xe_l[0], 0.0);
         if (logging > 0)
         {
            norms[0]     = 0.0;
            rel_norms[0] = 0.0;
         }

         nalu_hypre_EndTiming(maxwell_data -> time_index);

         return nalu_hypre_error_flag;
      }
   }

   /*-----------------------------------------------------
    * Do V-cycles:
    * For each index l, "fine" = l, "coarse" = (l-1)
    *
    *   solution update:
    *      edge_sol= edge_sol + T*node_sol
    *-----------------------------------------------------*/
   for (i = 0; i < max_iter; i++)
   {
      /* compute fine grid residual & nodal rhs. */
      nalu_hypre_ParVectorCopy(be_l[0], rese_l[0]);
      nalu_hypre_ParCSRMatrixMatvec(-1.0, Aee_l[0], xe_l[0], 1.0, rese_l[0]);
      nalu_hypre_ParVectorZeroBCValues(rese_l[0], BdryRanks_l[0], BdryRanksCnts_l[0]);
      nalu_hypre_ParCSRMatrixMatvec(1.0, T_transpose, rese_l[0], 0.0, bn_l[0]);

      /* convergence check */
      if (tol > 0.0)
      {
         r_dot_r = nalu_hypre_ParVectorInnerProd(rese_l[0], rese_l[0]);

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

      nalu_hypre_ParVectorCopy(bn_l[0], resn_l[0]);
      nalu_hypre_ParCSRMatrixMatvec(-1.0, Ann_l[0], xn_l[0], 1.0, resn_l[0]);
      r_dot_r = nalu_hypre_ParVectorInnerProd(resn_l[0], resn_l[0]);

      for (level = 0; level <= node_numlevs - 2; level++)
      {
         /*-----------------------------------------------
          * Down cycle
          *-----------------------------------------------*/
         for (j = 0; j < npre_relax; j++)
         {
            nalu_hypre_BoomerAMGRelaxIF(Ann_l[level],
                                   bn_l[level],
                                   nCF_marker_l[level],
                                   nrelax_type,
                                   relax_local,
                                   cycle_param,
                                   nrelax_weight[level],
                                   nomega[level],
                                   NULL,
                                   xn_l[level],
                                   nVtemp2_l[level],
                                   ze);
         }  /*for (j = 0; j < npre_relax; j++) */

         /* compute residuals */
         nalu_hypre_ParVectorCopy(bn_l[level], resn_l[level]);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, Ann_l[level], xn_l[level],
                                  1.0, resn_l[level]);

         /* restrict residuals */
         nalu_hypre_ParCSRMatrixMatvecT(1.0, RnT_l[level], resn_l[level],
                                   0.0, bn_l[level + 1]);

         /* zero off initial guess for the next level */
         nalu_hypre_ParVectorSetConstantValues(xn_l[level + 1], 0.0);

      }  /* for (level = 0; level<= node_numlevs-2; level++) */

      /* coarsest node solve */
      level = node_numlevs - 1;
      nalu_hypre_BoomerAMGRelaxIF(Ann_l[level],
                             bn_l[level],
                             nCF_marker_l[level],
                             nrelax_type,
                             relax_local,
                             cycle_param,
                             nrelax_weight[level],
                             nomega[level],
                             NULL,
                             xn_l[level],
                             nVtemp2_l[level],
                             ze);

      /*---------------------------------------------------------------------
       *  Cycle up the levels.
       *---------------------------------------------------------------------*/
      for (level = (node_numlevs - 2); level >= 1; level--)
      {
         nalu_hypre_ParCSRMatrixMatvec(1.0, Pn_l[level], xn_l[level + 1], 0.0,
                                  en_l[level]);
         nalu_hypre_ParVectorAxpy(1.0, en_l[level], xn_l[level]);

         /* post smooth */
         for (j = 0; j < npost_relax; j++)
         {
            nalu_hypre_BoomerAMGRelaxIF(Ann_l[level],
                                   bn_l[level],
                                   nCF_marker_l[level],
                                   nrelax_type,
                                   relax_local,
                                   cycle_param,
                                   nrelax_weight[level],
                                   nomega[level],
                                   NULL,
                                   xn_l[level],
                                   nVtemp2_l[level],
                                   ze);
         }
      }   /* for (level = (en_numlevs - 2); level>= 1; level--) */

      /* interpolate error and correct on finest grids */
      nalu_hypre_ParCSRMatrixMatvec(1.0, Pn_l[0], xn_l[1], 0.0, en_l[0]);
      nalu_hypre_ParVectorAxpy(1.0, en_l[0], xn_l[0]);

      for (j = 0; j < npost_relax; j++)
      {
         nalu_hypre_BoomerAMGRelaxIF(Ann_l[0],
                                bn_l[0],
                                nCF_marker_l[0],
                                nrelax_type,
                                relax_local,
                                cycle_param,
                                nrelax_weight[0],
                                nomega[0],
                                NULL,
                                xn_l[0],
                                nVtemp2_l[0],
                                ze);
      }  /* for (j = 0; j < npost_relax; j++) */
      nalu_hypre_ParVectorCopy(bn_l[0], resn_l[0]);
      nalu_hypre_ParCSRMatrixMatvec(-1.0, Ann_l[0], xn_l[0], 1.0, resn_l[0]);

      /* add the gradient solution component to xe_l[0] */
      nalu_hypre_ParCSRMatrixMatvec(1.0, Tgrad, xn_l[0], 1.0, xe_l[0]);

      nalu_hypre_ParVectorCopy(be_l[0], rese_l[0]);
      nalu_hypre_ParCSRMatrixMatvec(-1.0, Aee_l[0], xe_l[0], 1.0, rese_l[0]);
      r_dot_r = nalu_hypre_ParVectorInnerProd(rese_l[0], rese_l[0]);

      for (level = 0; level <= edge_numlevs - 2; level++)
      {
         /*-----------------------------------------------
          * Down cycle
          *-----------------------------------------------*/
         for (j = 0; j < npre_relax; j++)
         {
            nalu_hypre_BoomerAMGRelaxIF(Aee_l[level],
                                   be_l[level],
                                   eCF_marker_l[level],
                                   erelax_type,
                                   relax_local,
                                   cycle_param,
                                   erelax_weight[level],
                                   eomega[level],
                                   NULL,
                                   xe_l[level],
                                   eVtemp2_l[level],
                                   ze);
         }  /*for (j = 0; j < npre_relax; j++) */

         /* compute residuals */
         nalu_hypre_ParVectorCopy(be_l[level], rese_l[level]);
         nalu_hypre_ParCSRMatrixMatvec(-1.0, Aee_l[level], xe_l[level],
                                  1.0, rese_l[level]);

         /* restrict residuals */
         nalu_hypre_ParCSRMatrixMatvecT(1.0,
                                   (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(ReT_l[level]),
                                   rese_l[level], 0.0, be_l[level + 1]);
         nalu_hypre_ParVectorZeroBCValues(be_l[level + 1], BdryRanks_l[level + 1],
                                     BdryRanksCnts_l[level + 1]);

         /* zero off initial guess for the next level */
         nalu_hypre_ParVectorSetConstantValues(xe_l[level + 1], 0.0);

      }  /* for (level = 1; level<= edge_numlevels-2; level++) */

      /* coarsest edge solve */
      level = edge_numlevs - 1;
      for (j = 0; j < npre_relax; j++)
      {
         nalu_hypre_BoomerAMGRelaxIF(Aee_l[level],
                                be_l[level],
                                eCF_marker_l[level],
                                erelax_type,
                                relax_local,
                                cycle_param,
                                erelax_weight[level],
                                eomega[level],
                                NULL,
                                xe_l[level],
                                eVtemp2_l[level],
                                ze);
      }

      /*---------------------------------------------------------------------
       *  Up cycle.
       *---------------------------------------------------------------------*/
      for (level = (edge_numlevs - 2); level >= 1; level--)
      {
         nalu_hypre_ParCSRMatrixMatvec(1.0,
                                  (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(Pe_l[level]),
                                  xe_l[level + 1], 0.0, ee_l[level]);
         nalu_hypre_ParVectorZeroBCValues(ee_l[level], BdryRanks_l[level],
                                     BdryRanksCnts_l[level]);
         nalu_hypre_ParVectorAxpy(1.0, ee_l[level], xe_l[level]);

         /* post smooth */
         for (j = 0; j < npost_relax; j++)
         {
            nalu_hypre_BoomerAMGRelaxIF(Aee_l[level],
                                   be_l[level],
                                   eCF_marker_l[level],
                                   erelax_type,
                                   relax_local,
                                   cycle_param,
                                   erelax_weight[level],
                                   eomega[level],
                                   NULL,
                                   xe_l[level],
                                   eVtemp2_l[level],
                                   ze);
         }
      }  /* for (level = (edge_numlevs - 2); level>= 1; level--) */

      /* interpolate error and correct on finest grids */
      nalu_hypre_ParCSRMatrixMatvec(1.0,
                               (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(Pe_l[0]),
                               xe_l[1], 0.0, ee_l[0]);
      nalu_hypre_ParVectorZeroBCValues(ee_l[0], BdryRanks_l[0],
                                  BdryRanksCnts_l[0]);
      nalu_hypre_ParVectorAxpy(1.0, ee_l[0], xe_l[0]);

      for (j = 0; j < npost_relax; j++)
      {
         nalu_hypre_BoomerAMGRelaxIF(Aee_l[0],
                                be_l[0],
                                eCF_marker_l[0],
                                erelax_type,
                                relax_local,
                                cycle_param,
                                erelax_weight[0],
                                eomega[0],
                                NULL,
                                xe_l[0],
                                eVtemp2_l[0],
                                ze);
      }  /* for (j = 0; j < npost_relax; j++) */

      e_dot_e = nalu_hypre_ParVectorInnerProd(ee_l[0], ee_l[0]);
      x_dot_x = nalu_hypre_ParVectorInnerProd(xe_l[0], xe_l[0]);

      nalu_hypre_ParVectorCopy(be_l[0], rese_l[0]);
      nalu_hypre_ParCSRMatrixMatvec(-1.0, Aee_l[0], xe_l[0], 1.0, rese_l[0]);

      (maxwell_data -> num_iterations) = (i + 1);
   }

   nalu_hypre_EndTiming(maxwell_data -> time_index);

   nalu_hypre_ParVectorDestroy(ze);

   return nalu_hypre_error_flag;
}

