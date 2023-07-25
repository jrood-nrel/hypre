/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "seq_mv.h"
#include "csr_spgemm_device.h"

#if defined(NALU_HYPRE_USING_GPU)

/*
 * d_rc: input: nnz (upper bound) of each row
 * exact_rownnz: if d_rc is exact
 */
NALU_HYPRE_Int
hypreDevice_CSRSpGemmNumerWithRownnzUpperboundNoBin( NALU_HYPRE_Int       m,
                                                     NALU_HYPRE_Int       k,
                                                     NALU_HYPRE_Int       n,
                                                     NALU_HYPRE_Int      *d_ia,
                                                     NALU_HYPRE_Int      *d_ja,
                                                     NALU_HYPRE_Complex  *d_a,
                                                     NALU_HYPRE_Int      *d_ib,
                                                     NALU_HYPRE_Int      *d_jb,
                                                     NALU_HYPRE_Complex  *d_b,
                                                     NALU_HYPRE_Int      *d_rc,
                                                     NALU_HYPRE_Int       exact_rownnz,
                                                     NALU_HYPRE_Int     **d_ic_out,
                                                     NALU_HYPRE_Int     **d_jc_out,
                                                     NALU_HYPRE_Complex **d_c_out,
                                                     NALU_HYPRE_Int      *nnzC_out )
{
   constexpr NALU_HYPRE_Int SHMEM_HASH_SIZE = NUMER_HASH_SIZE[NALU_HYPRE_SPGEMM_DEFAULT_BIN];
   constexpr NALU_HYPRE_Int GROUP_SIZE = T_GROUP_SIZE[NALU_HYPRE_SPGEMM_DEFAULT_BIN];
   const NALU_HYPRE_Int BIN = NALU_HYPRE_SPGEMM_DEFAULT_BIN;

#ifdef NALU_HYPRE_SPGEMM_PRINTF
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_Int max_rc = NALU_HYPRE_ONEDPL_CALL(std::reduce, d_rc, d_rc + m, 0,
                                        sycl::maximum<NALU_HYPRE_Int>());
   NALU_HYPRE_Int min_rc = NALU_HYPRE_ONEDPL_CALL(std::reduce, d_rc, d_rc + m, max_rc,
                                        sycl::minimum<NALU_HYPRE_Int>());
#else
   NALU_HYPRE_Int max_rc = NALU_HYPRE_THRUST_CALL(reduce, d_rc, d_rc + m, 0,      thrust::maximum<NALU_HYPRE_Int>());
   NALU_HYPRE_Int min_rc = NALU_HYPRE_THRUST_CALL(reduce, d_rc, d_rc + m, max_rc, thrust::minimum<NALU_HYPRE_Int>());
#endif
   NALU_HYPRE_SPGEMM_PRINT("%s[%d]: max RC %d, min RC %d\n", __FILE__, __LINE__, max_rc, min_rc);
#endif

   /* if rc contains exact rownnz: can allocate the final C=(ic,jc,c) directly;
      if rc contains upper bound : it is a temporary space that is more than enough to store C */
   NALU_HYPRE_Int     *d_ic = nalu_hypre_TAlloc(NALU_HYPRE_Int, m + 1, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int     *d_jc;
   NALU_HYPRE_Complex *d_c;
   NALU_HYPRE_Int      nnzC = -1;

   nalu_hypre_create_ija(m, NULL, d_rc, d_ic, &d_jc, &d_c, &nnzC);

#ifdef NALU_HYPRE_SPGEMM_PRINTF
   NALU_HYPRE_SPGEMM_PRINT("%s[%d]: nnzC %d\n", __FILE__, __LINE__, nnzC);
#endif


   /* even with exact rownnz, still may need global hash, since shared hash is smaller than symbol */
   nalu_hypre_spgemm_numerical_with_rownnz<BIN, SHMEM_HASH_SIZE, GROUP_SIZE, false>
   (m, NULL, k, n, true, exact_rownnz, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, d_ic, d_jc, d_c);

   if (!exact_rownnz)
   {
      hypreDevice_CSRSpGemmNumerPostCopy<T_GROUP_SIZE[5]>(m, d_rc, &nnzC, &d_ic, &d_jc, &d_c);
   }

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out  = d_c;
   *nnzC_out = nnzC;

   return nalu_hypre_error_flag;
}

#define NALU_HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED2(BIN, BIN2, SHMEM_HASH_SIZE, GROUP_SIZE, EXACT_ROWNNZ, GHASH)  \
{                                                                                                                \
   const NALU_HYPRE_Int p = h_bin_ptr[BIN - 1];                                                                       \
   const NALU_HYPRE_Int q = h_bin_ptr[BIN];                                                                           \
   const NALU_HYPRE_Int bs = q - p;                                                                                   \
   if (bs)                                                                                                       \
   {                                                                                                             \
      NALU_HYPRE_SPGEMM_PRINT("bin[%d]: %d rows, p %d, q %d\n", BIN, bs, p, q);                                       \
      nalu_hypre_spgemm_numerical_with_rownnz<BIN2, SHMEM_HASH_SIZE, GROUP_SIZE, true>                                \
         (bs, d_rind + p, k, n, GHASH, EXACT_ROWNNZ, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, d_ic, d_jc, d_c);   \
   }                                                                                                             \
}

#define NALU_HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(BIN, SHMEM_HASH_SIZE, GROUP_SIZE, EXACT_ROWNNZ, GHASH)         \
   NALU_HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED2(BIN, BIN, SHMEM_HASH_SIZE, GROUP_SIZE, EXACT_ROWNNZ, GHASH)

NALU_HYPRE_Int
hypreDevice_CSRSpGemmNumerWithRownnzUpperboundBinned( NALU_HYPRE_Int       m,
                                                      NALU_HYPRE_Int       k,
                                                      NALU_HYPRE_Int       n,
                                                      NALU_HYPRE_Int      *d_ia,
                                                      NALU_HYPRE_Int      *d_ja,
                                                      NALU_HYPRE_Complex  *d_a,
                                                      NALU_HYPRE_Int      *d_ib,
                                                      NALU_HYPRE_Int      *d_jb,
                                                      NALU_HYPRE_Complex  *d_b,
                                                      NALU_HYPRE_Int      *d_rc,
                                                      NALU_HYPRE_Int       exact_rownnz,
                                                      NALU_HYPRE_Int     **d_ic_out,
                                                      NALU_HYPRE_Int     **d_jc_out,
                                                      NALU_HYPRE_Complex **d_c_out,
                                                      NALU_HYPRE_Int      *nnzC_out )
{
   /* if rc contains exact rownnz: can allocate the final C=(ic,jc,c) directly;
      if rc contains upper bound : it is a temporary space that is more than enough to store C */
   NALU_HYPRE_Int     *d_ic = nalu_hypre_TAlloc(NALU_HYPRE_Int, m + 1, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int     *d_jc;
   NALU_HYPRE_Complex *d_c;
   NALU_HYPRE_Int      nnzC = -1;

   nalu_hypre_create_ija(m, NULL, d_rc, d_ic, &d_jc, &d_c, &nnzC);

#ifdef NALU_HYPRE_SPGEMM_PRINTF
   NALU_HYPRE_SPGEMM_PRINT("%s[%d]: nnzC %d\n", __FILE__, __LINE__, nnzC);
#endif

   NALU_HYPRE_Int *d_rind = nalu_hypre_TAlloc(NALU_HYPRE_Int, m, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int  h_bin_ptr[NALU_HYPRE_SPGEMM_MAX_NBIN + 1];
   //NALU_HYPRE_Int  num_bins = nalu_hypre_HandleSpgemmNumBin(nalu_hypre_handle());
   NALU_HYPRE_Int high_bin = nalu_hypre_HandleSpgemmHighestBin(nalu_hypre_handle())[1];
   const bool hbin9 = 9 == high_bin;
   const char s = NUMER_HASH_SIZE[1] / 2, t = 2, u = high_bin;

   nalu_hypre_SpGemmCreateBins(m, s, t, u, d_rc, false, d_rind, h_bin_ptr);

#if 0
   NALU_HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  1, NUMER_HASH_SIZE[ 1], T_GROUP_SIZE[ 1], exact_rownnz,
                                               false);
#endif
   NALU_HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  2, NUMER_HASH_SIZE[ 2], T_GROUP_SIZE[ 2], exact_rownnz,
                                               false);
   NALU_HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  3, NUMER_HASH_SIZE[ 3], T_GROUP_SIZE[ 3], exact_rownnz,
                                               false);
   NALU_HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  4, NUMER_HASH_SIZE[ 4], T_GROUP_SIZE[ 4], exact_rownnz,
                                               false);
   NALU_HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  5, NUMER_HASH_SIZE[ 5], T_GROUP_SIZE[ 5], exact_rownnz,
                                               false);
   NALU_HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  6, NUMER_HASH_SIZE[ 6], T_GROUP_SIZE[ 6], exact_rownnz,
                                               false);
   NALU_HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  7, NUMER_HASH_SIZE[ 7], T_GROUP_SIZE[ 7], exact_rownnz,
                                               false);
   NALU_HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  8, NUMER_HASH_SIZE[ 8], T_GROUP_SIZE[ 8], exact_rownnz,
                                               false);
   NALU_HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED(  9, NUMER_HASH_SIZE[ 9], T_GROUP_SIZE[ 9], exact_rownnz,
                                               hbin9);
   NALU_HYPRE_SPGEMM_NUMERICAL_WITH_ROWNNZ_BINNED( 10, NUMER_HASH_SIZE[10], T_GROUP_SIZE[10], exact_rownnz,
                                              true);

   if (!exact_rownnz)
   {
      hypreDevice_CSRSpGemmNumerPostCopy<T_GROUP_SIZE[5]>(m, d_rc, &nnzC, &d_ic, &d_jc, &d_c);
   }

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out  = d_c;
   *nnzC_out = nnzC;

   nalu_hypre_TFree(d_rind, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
hypreDevice_CSRSpGemmNumerWithRownnzUpperbound( NALU_HYPRE_Int       m,
                                                NALU_HYPRE_Int       k,
                                                NALU_HYPRE_Int       n,
                                                NALU_HYPRE_Int      *d_ia,
                                                NALU_HYPRE_Int      *d_ja,
                                                NALU_HYPRE_Complex  *d_a,
                                                NALU_HYPRE_Int      *d_ib,
                                                NALU_HYPRE_Int      *d_jb,
                                                NALU_HYPRE_Complex  *d_b,
                                                NALU_HYPRE_Int      *d_rc,
                                                NALU_HYPRE_Int       exact_rownnz,
                                                NALU_HYPRE_Int     **d_ic_out,
                                                NALU_HYPRE_Int     **d_jc_out,
                                                NALU_HYPRE_Complex **d_c_out,
                                                NALU_HYPRE_Int      *nnzC_out )

{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPGEMM_NUMERIC] -= nalu_hypre_MPI_Wtime();
#endif

#ifdef NALU_HYPRE_SPGEMM_NVTX
   nalu_hypre_GpuProfilingPushRange("CSRSpGemmNumer");
#endif

#ifdef NALU_HYPRE_SPGEMM_TIMING
   NALU_HYPRE_Real t1 = nalu_hypre_MPI_Wtime();
#endif

   const NALU_HYPRE_Int binned = nalu_hypre_HandleSpgemmBinned(nalu_hypre_handle());

   if (binned)
   {
      hypreDevice_CSRSpGemmNumerWithRownnzUpperboundBinned
      (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, 1, d_ic_out, d_jc_out, d_c_out, nnzC_out);
   }
   else
   {
      hypreDevice_CSRSpGemmNumerWithRownnzUpperboundNoBin
      (m, k, n, d_ia, d_ja, d_a, d_ib, d_jb, d_b, d_rc, 1, d_ic_out, d_jc_out, d_c_out, nnzC_out);
   }

#ifdef NALU_HYPRE_SPGEMM_TIMING
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   NALU_HYPRE_Real t2 = nalu_hypre_MPI_Wtime() - t1;
   NALU_HYPRE_SPGEMM_PRINT("SpGemmNumerical time %f\n", t2);
#endif

#ifdef NALU_HYPRE_SPGEMM_NVTX
   nalu_hypre_GpuProfilingPopRange();
#endif

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPGEMM_NUMERIC] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}

#endif /* defined(NALU_HYPRE_USING_GPU) */
