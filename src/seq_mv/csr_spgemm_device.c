/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "csr_spgemm_device.h"
#include "seq_mv.hpp"

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)

NALU_HYPRE_Int
hypreDevice_CSRSpGemm(hypre_CSRMatrix  *A,
                      hypre_CSRMatrix  *B,
                      hypre_CSRMatrix **C_ptr)
{
   NALU_HYPRE_Complex    *d_a  = hypre_CSRMatrixData(A);
   NALU_HYPRE_Int        *d_ia = hypre_CSRMatrixI(A);
   NALU_HYPRE_Int        *d_ja = hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int         m    = hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int         k    = hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int         nnza = hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Complex    *d_b  = hypre_CSRMatrixData(B);
   NALU_HYPRE_Int        *d_ib = hypre_CSRMatrixI(B);
   NALU_HYPRE_Int        *d_jb = hypre_CSRMatrixJ(B);
   NALU_HYPRE_Int         n    = hypre_CSRMatrixNumCols(B);
   NALU_HYPRE_Int         nnzb = hypre_CSRMatrixNumNonzeros(B);
   NALU_HYPRE_Complex    *d_c;
   NALU_HYPRE_Int        *d_ic;
   NALU_HYPRE_Int        *d_jc;
   NALU_HYPRE_Int         nnzC;
   hypre_CSRMatrix  *C;

   *C_ptr = C = hypre_CSRMatrixCreate(m, n, 0);
   hypre_CSRMatrixMemoryLocation(C) = NALU_HYPRE_MEMORY_DEVICE;

   /* trivial case */
   if (nnza == 0 || nnzb == 0)
   {
      hypre_CSRMatrixI(C) = hypre_CTAlloc(NALU_HYPRE_Int, m + 1, NALU_HYPRE_MEMORY_DEVICE);

      return hypre_error_flag;
   }

#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_SPGEMM] -= hypre_MPI_Wtime();
#endif

#ifdef NALU_HYPRE_SPGEMM_TIMING
   hypre_ForceSyncComputeStream(hypre_handle());
   NALU_HYPRE_Real ta = hypre_MPI_Wtime();
#endif

   /* use CUSPARSE or rocSPARSE*/
   if (hypre_HandleSpgemmUseVendor(hypre_handle()))
   {
#if defined(NALU_HYPRE_USING_CUSPARSE)
      hypreDevice_CSRSpGemmCusparse(m, k, n,
                                    hypre_CSRMatrixGPUMatDescr(A), nnza, d_ia, d_ja, d_a,
                                    hypre_CSRMatrixGPUMatDescr(B), nnzb, d_ib, d_jb, d_b,
                                    hypre_CSRMatrixGPUMatDescr(C), &nnzC, &d_ic, &d_jc, &d_c);
#elif defined(NALU_HYPRE_USING_ROCSPARSE)
      hypreDevice_CSRSpGemmRocsparse(m, k, n,
                                     hypre_CSRMatrixGPUMatDescr(A), nnza, d_ia, d_ja, d_a,
                                     hypre_CSRMatrixGPUMatDescr(B), nnzb, d_ib, d_jb, d_b,
                                     hypre_CSRMatrixGPUMatDescr(C), hypre_CSRMatrixGPUMatInfo(C), &nnzC, &d_ic, &d_jc, &d_c);
#elif defined(NALU_HYPRE_USING_ONEMKLSPARSE)
      hypreDevice_CSRSpGemmOnemklsparse(m, k, n,
                                        hypre_CSRMatrixGPUMatHandle(A), nnza, d_ia, d_ja, d_a,
                                        hypre_CSRMatrixGPUMatHandle(B), nnzb, d_ib, d_jb, d_b,
                                        hypre_CSRMatrixGPUMatHandle(C), &nnzC, &d_ic, &d_jc, &d_c);
#else
      hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Attempting to use device sparse matrix library for SpGEMM without having compiled support for it!\n");
#endif
   }
   else
   {
      /* WM: todo - sycl implementation when not using oneMKL sparse */
#if !defined(NALU_HYPRE_USING_SYCL)
      d_a  = hypre_CSRMatrixPatternOnly(A) ? NULL : d_a;
      d_b  = hypre_CSRMatrixPatternOnly(B) ? NULL : d_b;

      NALU_HYPRE_Int *d_rc = hypre_TAlloc(NALU_HYPRE_Int, m, NALU_HYPRE_MEMORY_DEVICE);
      const NALU_HYPRE_Int alg = hypre_HandleSpgemmAlgorithm(hypre_handle());

      if (hypre_HandleSpgemmNumBin(hypre_handle()) == 0)
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
         const NALU_HYPRE_Int row_est_mtd = hypre_HandleSpgemmRownnzEstimateMethod(hypre_handle());

         hypreDevice_CSRSpGemmRownnzEstimate(m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, row_est_mtd);

         NALU_HYPRE_Int rownnz_exact;

         hypreDevice_CSRSpGemmRownnzUpperbound
         (m, k, n, d_ia, d_ja, d_ib, d_jb, 1 /* with input rc */, d_rc, &rownnz_exact);

         hypreDevice_CSRSpGemmNumerWithRownnzUpperbound
         (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, rownnz_exact, &d_ic, &d_jc, &d_c, &nnzC);
      }

      hypre_TFree(d_rc, NALU_HYPRE_MEMORY_DEVICE);
#endif /* !defined(NALU_HYPRE_USING_SYCL) */
   }

#ifdef NALU_HYPRE_SPGEMM_TIMING
   hypre_ForceSyncComputeStream(hypre_handle());
   NALU_HYPRE_Real tb = hypre_MPI_Wtime() - ta;
   NALU_HYPRE_SPGEMM_PRINT("SpGemm time %f\n", tb);
#endif

   hypre_CSRMatrixNumNonzeros(C) = nnzC;
   hypre_CSRMatrixI(C)           = d_ic;
   hypre_CSRMatrixJ(C)           = d_jc;
   hypre_CSRMatrixData(C)        = d_c;

#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_SPGEMM] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

#endif /* defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL) */

