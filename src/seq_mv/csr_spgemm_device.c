/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"
#include "seq_mv.hpp"

#if defined(NALU_HYPRE_USING_GPU)

NALU_HYPRE_Int
hypreDevice_CSRSpGemm(nalu_hypre_CSRMatrix  *A,
                      nalu_hypre_CSRMatrix  *B,
                      nalu_hypre_CSRMatrix **C_ptr)
{
   NALU_HYPRE_Complex    *d_a  = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int        *d_ia = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int        *d_ja = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int         m    = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int         k    = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int         nnza = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Complex    *d_b  = nalu_hypre_CSRMatrixData(B);
   NALU_HYPRE_Int        *d_ib = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int        *d_jb = nalu_hypre_CSRMatrixJ(B);
   NALU_HYPRE_Int         n    = nalu_hypre_CSRMatrixNumCols(B);
   NALU_HYPRE_Int         nnzb = nalu_hypre_CSRMatrixNumNonzeros(B);
   NALU_HYPRE_Complex    *d_c;
   NALU_HYPRE_Int        *d_ic;
   NALU_HYPRE_Int        *d_jc;
   NALU_HYPRE_Int         nnzC;
   nalu_hypre_CSRMatrix  *C;

   *C_ptr = C = nalu_hypre_CSRMatrixCreate(m, n, 0);
   nalu_hypre_CSRMatrixMemoryLocation(C) = NALU_HYPRE_MEMORY_DEVICE;

   /* trivial case */
   if (nnza == 0 || nnzb == 0)
   {
      nalu_hypre_CSRMatrixI(C) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, m + 1, NALU_HYPRE_MEMORY_DEVICE);

      return nalu_hypre_error_flag;
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPGEMM] -= nalu_hypre_MPI_Wtime();
#endif

#ifdef NALU_HYPRE_SPGEMM_TIMING
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   NALU_HYPRE_Real ta = nalu_hypre_MPI_Wtime();
#endif

   /* use CUSPARSE or rocSPARSE*/
   if (nalu_hypre_HandleSpgemmUseVendor(nalu_hypre_handle()))
   {
#if defined(NALU_HYPRE_USING_CUSPARSE)
      hypreDevice_CSRSpGemmCusparse(m, k, n,
                                    nalu_hypre_CSRMatrixGPUMatDescr(A), nnza, d_ia, d_ja, d_a,
                                    nalu_hypre_CSRMatrixGPUMatDescr(B), nnzb, d_ib, d_jb, d_b,
                                    nalu_hypre_CSRMatrixGPUMatDescr(C), &nnzC, &d_ic, &d_jc, &d_c);
#elif defined(NALU_HYPRE_USING_ROCSPARSE)
      hypreDevice_CSRSpGemmRocsparse(m, k, n,
                                     nalu_hypre_CSRMatrixGPUMatDescr(A), nnza, d_ia, d_ja, d_a,
                                     nalu_hypre_CSRMatrixGPUMatDescr(B), nnzb, d_ib, d_jb, d_b,
                                     nalu_hypre_CSRMatrixGPUMatDescr(C), nalu_hypre_CSRMatrixGPUMatInfo(C), &nnzC, &d_ic, &d_jc, &d_c);
#elif defined(NALU_HYPRE_USING_ONEMKLSPARSE)
      hypreDevice_CSRSpGemmOnemklsparse( m, k, n,
                                         nalu_hypre_CSRMatrixGPUMatHandle(A), nnza, d_ia, d_ja, d_a,
                                         nalu_hypre_CSRMatrixGPUMatHandle(B), nnzb, d_ib, d_jb, d_b,
                                         nalu_hypre_CSRMatrixGPUMatHandle(C), &nnzC, &d_ic, &d_jc, &d_c);
#else
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Attempting to use device sparse matrix library for SpGEMM without having compiled support for it!\n");
#endif
   }
   else
   {
      d_a  = nalu_hypre_CSRMatrixPatternOnly(A) ? NULL : d_a;
      d_b  = nalu_hypre_CSRMatrixPatternOnly(B) ? NULL : d_b;

      NALU_HYPRE_Int *d_rc = nalu_hypre_TAlloc(NALU_HYPRE_Int, m, NALU_HYPRE_MEMORY_DEVICE);
      const NALU_HYPRE_Int alg = nalu_hypre_HandleSpgemmAlgorithm(nalu_hypre_handle());

      if (nalu_hypre_HandleSpgemmNumBin(nalu_hypre_handle()) == 0)
      {
         hypreDevice_CSRSpGemmBinnedGetBlockNumDim();
      }

      if (alg == 1)
      {
         hypreDevice_CSRSpGemmRownnz
         (m, k, n, nnza, d_ia, d_ja, d_ib, d_jb, 0 /* without input rc */, d_rc);

         hypreDevice_CSRSpGemmNumerWithRownnzUpperbound
         (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, 1, &d_ic, &d_jc, &d_c, &nnzC);
      }
      else /* if (alg == 3) */
      {
         const NALU_HYPRE_Int row_est_mtd = nalu_hypre_HandleSpgemmRownnzEstimateMethod(nalu_hypre_handle());

         hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, row_est_mtd);

         NALU_HYPRE_Int rownnz_exact;

         hypreDevice_CSRSpGemmRownnzUpperbound
         (m, k, n, d_ia, d_ja, d_ib, d_jb, 1 /* with input rc */, d_rc, &rownnz_exact);

         hypreDevice_CSRSpGemmNumerWithRownnzUpperbound
         (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, rownnz_exact, &d_ic, &d_jc, &d_c, &nnzC);
      }

      nalu_hypre_TFree(d_rc, NALU_HYPRE_MEMORY_DEVICE);
   }

#ifdef NALU_HYPRE_SPGEMM_TIMING
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   NALU_HYPRE_Real tb = nalu_hypre_MPI_Wtime() - ta;
   NALU_HYPRE_SPGEMM_PRINT("SpGemm time %f\n", tb);
#endif

   nalu_hypre_CSRMatrixNumNonzeros(C) = nnzC;
   nalu_hypre_CSRMatrixI(C)           = d_ic;
   nalu_hypre_CSRMatrixJ(C)           = d_jc;
   nalu_hypre_CSRMatrixData(C)        = d_c;

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPGEMM] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}

#endif /* defined(NALU_HYPRE_USING_GPU) */

