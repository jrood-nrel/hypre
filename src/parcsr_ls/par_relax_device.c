/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGRelaxHybridGaussSeidelDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelaxHybridGaussSeidelDevice( nalu_hypre_ParCSRMatrix *A,
                                             nalu_hypre_ParVector    *f,
                                             NALU_HYPRE_Int          *cf_marker,
                                             NALU_HYPRE_Int           relax_points,
                                             NALU_HYPRE_Real          relax_weight,
                                             NALU_HYPRE_Real          omega,
                                             NALU_HYPRE_Real         *l1_norms,
                                             nalu_hypre_ParVector    *u,
                                             nalu_hypre_ParVector    *Vtemp,
                                             nalu_hypre_ParVector    *Ztemp,
                                             NALU_HYPRE_Int           GS_order,
                                             NALU_HYPRE_Int           Symm )
{
   /* Vtemp, Ztemp have the fine-grid size. Create two shell vectors that have the correct size */
   nalu_hypre_ParVector *w1 = nalu_hypre_ParVectorCloneShallow(f);
   nalu_hypre_ParVector *w2 = nalu_hypre_ParVectorCloneShallow(u);

   nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(w1)) = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(
                                                                          Vtemp));
   nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(w2)) = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(
                                                                          Ztemp));

   if (Symm)
   {
      /* V = f - A*u */
      nalu_hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, u, 1.0, f, w1);

      /* Z = L^{-1}*V */
      nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('L', 0, nalu_hypre_ParCSRMatrixDiag(A), l1_norms,
                                              nalu_hypre_ParVectorLocalVector(w1),
                                              nalu_hypre_ParVectorLocalVector(w2));

      /* u = u + w*Z */
      nalu_hypre_ParVectorAxpy(relax_weight, w2, u);

      /* Note: only update V from local change of u, i.e., V = V - w*A_diag*Z_local */
      nalu_hypre_CSRMatrixMatvec(-relax_weight, nalu_hypre_ParCSRMatrixDiag(A),
                            nalu_hypre_ParVectorLocalVector(w2), 1.0,
                            nalu_hypre_ParVectorLocalVector(w1));

      /* Z = U^{-1}*V */
      nalu_hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, nalu_hypre_ParCSRMatrixDiag(A), l1_norms,
                                              nalu_hypre_ParVectorLocalVector(w1),
                                              nalu_hypre_ParVectorLocalVector(w2));

      /* u = u + w*Z */
      nalu_hypre_ParVectorAxpy(relax_weight, w2, u);
   }
   else
   {
      const char uplo = GS_order > 0 ? 'L' : 'U';
      /* V = f - A*u */
      nalu_hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A, u, 1.0, f, w1);

      /* Z = L^{-1}*V or Z = U^{-1}*V */
      nalu_hypre_CSRMatrixTriLowerUpperSolveDevice(uplo, 0, nalu_hypre_ParCSRMatrixDiag(A), l1_norms,
                                              nalu_hypre_ParVectorLocalVector(w1),
                                              nalu_hypre_ParVectorLocalVector(w2));

      /* u = u + w*Z */
      nalu_hypre_ParVectorAxpy(relax_weight, w2, u);
   }

   nalu_hypre_ParVectorDestroy(w1);
   nalu_hypre_ParVectorDestroy(w2);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGRelaxTwoStageGaussSeidelDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGRelaxTwoStageGaussSeidelDevice ( nalu_hypre_ParCSRMatrix *A,
                                                nalu_hypre_ParVector    *f,
                                                NALU_HYPRE_Real          relax_weight,
                                                NALU_HYPRE_Real          omega,
                                                NALU_HYPRE_Real         *A_diag_diag,
                                                nalu_hypre_ParVector    *u,
                                                nalu_hypre_ParVector    *r,
                                                nalu_hypre_ParVector    *z,
                                                NALU_HYPRE_Int           num_inner_iters)
{
   nalu_hypre_CSRMatrix *A_diag       = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int        num_rows     = nalu_hypre_CSRMatrixNumRows(A_diag);

   nalu_hypre_Vector    *u_local      = nalu_hypre_ParVectorLocalVector(u);
   nalu_hypre_Vector    *r_local      = nalu_hypre_ParVectorLocalVector(r);
   nalu_hypre_Vector    *z_local      = nalu_hypre_ParVectorLocalVector(z);

   NALU_HYPRE_Int        u_vecstride  = nalu_hypre_VectorVectorStride(u_local);
   NALU_HYPRE_Int        r_vecstride  = nalu_hypre_VectorVectorStride(r_local);
   NALU_HYPRE_Int        z_vecstride  = nalu_hypre_VectorVectorStride(z_local);
   NALU_HYPRE_Complex   *u_data       = nalu_hypre_VectorData(u_local);
   NALU_HYPRE_Complex   *r_data       = nalu_hypre_VectorData(r_local);
   NALU_HYPRE_Complex   *z_data       = nalu_hypre_VectorData(z_local);

   NALU_HYPRE_Int        num_vectors  = nalu_hypre_VectorNumVectors(r_local);
   NALU_HYPRE_Complex    multiplier   = 1.0;
   NALU_HYPRE_Int        i;

   nalu_hypre_GpuProfilingPushRange("BoomerAMGRelaxTwoStageGaussSeidelDevice");

   /* Sanity checks */
   nalu_hypre_assert(u_vecstride == num_rows);
   nalu_hypre_assert(r_vecstride == num_rows);
   nalu_hypre_assert(z_vecstride == num_rows);

   // 0) r = relax_weight * (f - A * u)
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(-relax_weight, A, u, relax_weight, f, r);

   // 1) z = r/D, u = u + z
   hypreDevice_DiagScaleVector2(num_vectors, num_rows, A_diag_diag,
                                r_data, 1.0, z_data, u_data, 1);
   multiplier *= -1.0;

   for (i = 0; i < num_inner_iters; i++)
   {
      // 2) r = L * z
      nalu_hypre_CSRMatrixSpMVDevice(0, 1.0, A_diag, z_local, 0.0, r_local, -2);

      // 3) z = r/D, u = u + m * z
      hypreDevice_DiagScaleVector2(num_vectors, num_rows, A_diag_diag,
                                   r_data, multiplier, z_data, u_data,
                                   (num_inner_iters > i + 1));
      multiplier *= -1.0;
   }

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

#endif /* #if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) */
