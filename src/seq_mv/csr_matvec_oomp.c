/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for nalu_hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_DEVICE_OPENMP)

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixMatvec
 *--------------------------------------------------------------------------*/

/* y[offset:end] = alpha*A[offset:end,:]*x + beta*b[offset:end] */
NALU_HYPRE_Int
nalu_hypre_CSRMatrixMatvecOMPOffload( NALU_HYPRE_Int        trans,
                                 NALU_HYPRE_Complex    alpha,
                                 nalu_hypre_CSRMatrix *A,
                                 nalu_hypre_Vector    *x,
                                 NALU_HYPRE_Complex    beta,
                                 nalu_hypre_Vector    *y,
                                 NALU_HYPRE_Int        offset )
{
   nalu_hypre_CSRMatrix *B;

   if (trans)
   {
      nalu_hypre_CSRMatrixTransposeDevice(A, &B, 1);

      /* NALU_HYPRE_CUDA_CALL(cudaDeviceSynchronize()); */
   }
   else
   {
      B = A;
   }

   NALU_HYPRE_Int      A_nrows  = nalu_hypre_CSRMatrixNumRows(B);
   NALU_HYPRE_Complex *A_data   = nalu_hypre_CSRMatrixData(B);
   NALU_HYPRE_Int     *A_i      = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int     *A_j      = nalu_hypre_CSRMatrixJ(B);
   NALU_HYPRE_Complex *x_data   = nalu_hypre_VectorData(x);
   NALU_HYPRE_Complex *y_data   = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int      i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(A_data, A_i, A_j, y_data, x_data)
   for (i = offset; i < A_nrows; i++)
   {
      NALU_HYPRE_Complex tempx = 0.0;
      NALU_HYPRE_Int j;
      for (j = A_i[i]; j < A_i[i + 1]; j++)
      {
         tempx += A_data[j] * x_data[A_j[j]];
      }
      y_data[i] = alpha * tempx + beta * y_data[i];
   }

   /* NALU_HYPRE_CUDA_CALL(cudaDeviceSynchronize()); */

   return nalu_hypre_error_flag;
}

#endif /* #if defined(NALU_HYPRE_USING_DEVICE_OPENMP) */

