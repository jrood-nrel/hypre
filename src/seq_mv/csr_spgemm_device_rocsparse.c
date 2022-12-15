/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "_nalu_hypre_utilities.hpp"
#include "seq_mv.hpp"

#if defined(NALU_HYPRE_USING_HIP) && defined(NALU_HYPRE_USING_ROCSPARSE)

NALU_HYPRE_Int
hypreDevice_CSRSpGemmRocsparse(NALU_HYPRE_Int           m,
                               NALU_HYPRE_Int           k,
                               NALU_HYPRE_Int           n,
                               rocsparse_mat_descr descrA,
                               NALU_HYPRE_Int           nnzA,
                               NALU_HYPRE_Int          *d_ia,
                               NALU_HYPRE_Int          *d_ja,
                               NALU_HYPRE_Complex      *d_a,
                               rocsparse_mat_descr descrB,
                               NALU_HYPRE_Int           nnzB,
                               NALU_HYPRE_Int          *d_ib,
                               NALU_HYPRE_Int          *d_jb,
                               NALU_HYPRE_Complex      *d_b,
                               rocsparse_mat_descr descrC,
                               rocsparse_mat_info  infoC,
                               NALU_HYPRE_Int          *nnzC_out,
                               NALU_HYPRE_Int         **d_ic_out,
                               NALU_HYPRE_Int         **d_jc_out,
                               NALU_HYPRE_Complex     **d_c_out)
{
   NALU_HYPRE_Int  *d_ic, *d_jc, baseC, nnzC;
   NALU_HYPRE_Int  *d_ja_sorted, *d_jb_sorted;
   NALU_HYPRE_Complex *d_c, *d_a_sorted, *d_b_sorted;

   d_a_sorted  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzA, NALU_HYPRE_MEMORY_DEVICE);
   d_b_sorted  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzB, NALU_HYPRE_MEMORY_DEVICE);
   d_ja_sorted = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nnzA, NALU_HYPRE_MEMORY_DEVICE);
   d_jb_sorted = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nnzB, NALU_HYPRE_MEMORY_DEVICE);

   rocsparse_handle handle = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());

   rocsparse_operation transA = rocsparse_operation_none;
   rocsparse_operation transB = rocsparse_operation_none;

   /* Copy the unsorted over as the initial "sorted" */
   nalu_hypre_TMemcpy(d_ja_sorted, d_ja, NALU_HYPRE_Int,     nnzA, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(d_a_sorted,  d_a,  NALU_HYPRE_Complex, nnzA, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(d_jb_sorted, d_jb, NALU_HYPRE_Int,     nnzB, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(d_b_sorted,  d_b,  NALU_HYPRE_Complex, nnzB, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   /* For rocSPARSE, the CSR SpGEMM implementation does not require the columns to be sorted! */
   /* RL: for matrices with long rows, it seemed that the sorting is still needed */
#if 0
   nalu_hypre_SortCSRRocsparse(m, k, nnzA, descrA, d_ia, d_ja_sorted, d_a_sorted);
   nalu_hypre_SortCSRRocsparse(k, n, nnzB, descrB, d_ib, d_jb_sorted, d_b_sorted);
#endif

   // nnzTotalDevHostPtr points to host memory
   NALU_HYPRE_Int *nnzTotalDevHostPtr = &nnzC;
   NALU_HYPRE_ROCSPARSE_CALL( rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host) );

   d_ic = nalu_hypre_TAlloc(NALU_HYPRE_Int, m + 1, NALU_HYPRE_MEMORY_DEVICE);

   // For rocsparse, we need an extra buffer for computing the
   // csrgemmnnz and the csrgemm
   //
   // Once the buffer is allocated, we can use the same allocated
   // buffer for both the csrgemm_nnz and csrgemm
   //
   // Note that rocsparse csrgemms do: C = \alpha*A*B +\beta*D
   // So we hardcode \alpha=1, D to nothing, and pass NULL for beta
   // to indicate \beta = 0 to match the cusparse behavior.
   NALU_HYPRE_Complex alpha = 1.0;

   size_t rs_buffer_size = 0;
   void *rs_buffer;

   NALU_HYPRE_ROCSPARSE_CALL( nalu_hypre_rocsparse_csrgemm_buffer_size(handle,
                                                             transA, transB,
                                                             m, n, k,
                                                             &alpha, // \alpha = 1
                                                             descrA, nnzA, d_ia, d_ja_sorted,
                                                             descrB, nnzB, d_ib, d_jb_sorted,
                                                             NULL, // \beta = 0
                                                             NULL,   0,    NULL, NULL, // D is nothing
                                                             infoC, &rs_buffer_size) );

   rs_buffer = nalu_hypre_TAlloc(char, rs_buffer_size, NALU_HYPRE_MEMORY_DEVICE);

   // Note that rocsparse csrgemms do: C = \alpha*A*B +\beta*D
   // So we hardcode \alpha=1, D to nothing, and \beta = 0
   // to match the cusparse behavior
   NALU_HYPRE_ROCSPARSE_CALL( rocsparse_csrgemm_nnz(handle, transA, transB,
                                               m, n, k,
                                               descrA, nnzA, d_ia, d_ja_sorted,
                                               descrB, nnzB, d_ib, d_jb_sorted,
                                               NULL,   0,    NULL, NULL, // D is nothing
                                               descrC,       d_ic, nnzTotalDevHostPtr,
                                               infoC, rs_buffer) );

   if (NULL != nnzTotalDevHostPtr)
   {
      nnzC = *nnzTotalDevHostPtr;
   }
   else
   {
      nalu_hypre_TMemcpy(&nnzC,  d_ic + m, NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(&baseC, d_ic,     NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      nnzC -= baseC;
   }

   d_jc = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nnzC, NALU_HYPRE_MEMORY_DEVICE);
   d_c  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzC, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_ROCSPARSE_CALL( nalu_hypre_rocsparse_csrgemm(handle, transA, transB,
                                                 m, n, k,
                                                 &alpha, // alpha = 1
                                                 descrA, nnzA, d_a_sorted, d_ia, d_ja_sorted,
                                                 descrB, nnzB, d_b_sorted, d_ib, d_jb_sorted,
                                                 NULL, // beta = 0
                                                 NULL,   0,    NULL,       NULL, NULL, // D is nothing
                                                 descrC,       d_c, d_ic, d_jc,
                                                 infoC, rs_buffer) );

   // Free up the memory needed by rocsparse
   nalu_hypre_TFree(rs_buffer, NALU_HYPRE_MEMORY_DEVICE);

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out  = d_c;
   *nnzC_out = nnzC;

   nalu_hypre_TFree(d_a_sorted,  NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_b_sorted,  NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_ja_sorted, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_jb_sorted, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}

#endif // defined(NALU_HYPRE_USING_HIP) && defined(NALU_HYPRE_USING_ROCSPARSE)
