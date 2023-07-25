/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Relaxation scheme
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGRelax
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelaxIF( nalu_hypre_ParCSRMatrix *A,
                        nalu_hypre_ParVector    *f,
                        NALU_HYPRE_Int          *cf_marker,
                        NALU_HYPRE_Int           relax_type,
                        NALU_HYPRE_Int           relax_order,
                        NALU_HYPRE_Int           cycle_param,
                        NALU_HYPRE_Real          relax_weight,
                        NALU_HYPRE_Real          omega,
                        NALU_HYPRE_Real         *l1_norms,
                        nalu_hypre_ParVector    *u,
                        nalu_hypre_ParVector    *Vtemp,
                        nalu_hypre_ParVector    *Ztemp )
{
   NALU_HYPRE_Int i, Solve_err_flag = 0;
   NALU_HYPRE_Int relax_points[2];

   if (relax_order == 1 && cycle_param < 3)
   {
      if (cycle_param < 2)
      {
         /* CF down cycle */
         relax_points[0] =  1;
         relax_points[1] = -1;
      }
      else
      {
         /* FC up cycle */
         relax_points[0] = -1;
         relax_points[1] =  1;
      }

      for (i = 0; i < 2; i++)
      {
         Solve_err_flag = nalu_hypre_BoomerAMGRelax(A, f, cf_marker, relax_type, relax_points[i],
                                               relax_weight, omega, l1_norms, u, Vtemp, Ztemp);
      }
   }
   else
   {
      Solve_err_flag = nalu_hypre_BoomerAMGRelax(A, f, cf_marker, relax_type, 0, relax_weight, omega,
                                            l1_norms, u, Vtemp, Ztemp);
   }

   return Solve_err_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRRelax_L1_Jacobi (same as the one in AMS, but this allows CF)
 * u_new = u_old + w D^{-1}(f - A u), where D_ii = ||A(i,:)||_1
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_ParCSRRelax_L1_Jacobi( nalu_hypre_ParCSRMatrix *A,
                             nalu_hypre_ParVector    *f,
                             NALU_HYPRE_Int          *cf_marker,
                             NALU_HYPRE_Int           relax_points,
                             NALU_HYPRE_Real          relax_weight,
                             NALU_HYPRE_Real         *l1_norms,
                             nalu_hypre_ParVector    *u,
                             nalu_hypre_ParVector    *Vtemp )

{
   return nalu_hypre_BoomerAMGRelax(A, f, cf_marker, 18, relax_points, relax_weight, 0.0, l1_norms, u,
                               Vtemp, NULL);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGRelax_FCFJacobi
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelax_FCFJacobi( nalu_hypre_ParCSRMatrix *A,
                                nalu_hypre_ParVector    *f,
                                NALU_HYPRE_Int          *cf_marker,
                                NALU_HYPRE_Real          relax_weight,
                                nalu_hypre_ParVector    *u,
                                nalu_hypre_ParVector    *Vtemp)
{
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int relax_points[3];
   NALU_HYPRE_Int relax_type = 0;

   relax_points[0] = -1; /*F */
   relax_points[1] =  1; /*C */
   relax_points[2] = -1; /*F */

   /* cf == NULL --> size == 0 */
   if (cf_marker == NULL)
   {
      nalu_hypre_assert(nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A)) == 0);
   }

   for (i = 0; i < 3; i++)
   {
      nalu_hypre_BoomerAMGRelax(A, f, cf_marker, relax_type, relax_points[i],
                           relax_weight, 0.0, NULL, u, Vtemp, NULL);
   }

   return nalu_hypre_error_flag;
}



