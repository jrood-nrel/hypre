/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "seq_mv.h"
#include "_nalu_hypre_utilities.hpp"

#if defined(NALU_HYPRE_USING_CUSPARSE)

NALU_HYPRE_Int
hypreDevice_CSRSpTransCusparse(NALU_HYPRE_Int   m,        NALU_HYPRE_Int   n,        NALU_HYPRE_Int       nnzA,
                               NALU_HYPRE_Int  *d_ia,     NALU_HYPRE_Int  *d_ja,     NALU_HYPRE_Complex  *d_aa,
                               NALU_HYPRE_Int **d_ic_out, NALU_HYPRE_Int **d_jc_out, NALU_HYPRE_Complex **d_ac_out,
                               NALU_HYPRE_Int   want_data)
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPTRANS] -= nalu_hypre_MPI_Wtime();
#endif

   cusparseHandle_t handle = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
   cusparseAction_t action = want_data ? CUSPARSE_ACTION_NUMERIC : CUSPARSE_ACTION_SYMBOLIC;
   NALU_HYPRE_Complex *csc_a;
   if (want_data)
   {
      csc_a = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzA,  NALU_HYPRE_MEMORY_DEVICE);
   }
   else
   {
      csc_a = NULL;
      d_aa = NULL;
   }
   NALU_HYPRE_Int *csc_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnzA,  NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int *csc_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, n + 1, NALU_HYPRE_MEMORY_DEVICE);

#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
   size_t bufferSize = 0;
   const cudaDataType data_type = nalu_hypre_HYPREComplexToCudaDataType();

   NALU_HYPRE_CUSPARSE_CALL( cusparseCsr2cscEx2_bufferSize(handle,
                                                      m, n, nnzA,
                                                      d_aa, d_ia, d_ja,
                                                      csc_a, csc_i, csc_j,
                                                      data_type,
                                                      action,
                                                      CUSPARSE_INDEX_BASE_ZERO,
                                                      CUSPARSE_CSR2CSC_ALG1,
                                                      &bufferSize) );

   char *dBuffer = nalu_hypre_TAlloc(char, bufferSize, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_CUSPARSE_CALL( cusparseCsr2cscEx2(handle,
                                           m, n, nnzA,
                                           d_aa, d_ia, d_ja,
                                           csc_a, csc_i, csc_j,
                                           data_type,
                                           action,
                                           CUSPARSE_INDEX_BASE_ZERO,
                                           CUSPARSE_CSR2CSC_ALG1,
                                           dBuffer) );

   nalu_hypre_TFree(dBuffer, NALU_HYPRE_MEMORY_DEVICE);
#else
   NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparse_csr2csc(handle,
                                               m, n, nnzA,
                                               d_aa, d_ia, d_ja,
                                               csc_a, csc_j, csc_i,
                                               action,
                                               CUSPARSE_INDEX_BASE_ZERO) );
#endif /* #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION */

   *d_ic_out = csc_i;
   *d_jc_out = csc_j;
   *d_ac_out = csc_a;

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPTRANS] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}

#endif // #if defined(NALU_HYPRE_USING_CUSPARSE)


#if defined(NALU_HYPRE_USING_ROCSPARSE)
NALU_HYPRE_Int
hypreDevice_CSRSpTransRocsparse(NALU_HYPRE_Int   m,        NALU_HYPRE_Int   n,        NALU_HYPRE_Int       nnzA,
                                NALU_HYPRE_Int  *d_ia,     NALU_HYPRE_Int  *d_ja,     NALU_HYPRE_Complex  *d_aa,
                                NALU_HYPRE_Int **d_ic_out, NALU_HYPRE_Int **d_jc_out, NALU_HYPRE_Complex **d_ac_out,
                                NALU_HYPRE_Int   want_data)
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPTRANS] -= nalu_hypre_MPI_Wtime();
#endif

   rocsparse_handle handle = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
   rocsparse_action action = want_data ? rocsparse_action_numeric : rocsparse_action_symbolic;

   NALU_HYPRE_Complex *csc_a;
   if (want_data)
   {
      csc_a = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzA,  NALU_HYPRE_MEMORY_DEVICE);
   }
   else
   {
      csc_a = NULL;
      d_aa = NULL;
   }
   NALU_HYPRE_Int *csc_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnzA,  NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int *csc_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, n + 1, NALU_HYPRE_MEMORY_DEVICE);

   size_t buffer_size = 0;
   NALU_HYPRE_ROCSPARSE_CALL( rocsparse_csr2csc_buffer_size(handle,
                                                       m, n, nnzA,
                                                       csc_i, csc_j,
                                                       action,
                                                       &buffer_size) );

   void * buffer;
   buffer = nalu_hypre_TAlloc(char, buffer_size, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_ROCSPARSE_CALL( nalu_hypre_rocsparse_csr2csc(handle,
                                                 m, n, nnzA,
                                                 d_aa, d_ia, d_ja,
                                                 csc_a, csc_j, csc_i,
                                                 action,
                                                 rocsparse_index_base_zero,
                                                 buffer) );

   nalu_hypre_TFree(buffer, NALU_HYPRE_MEMORY_DEVICE);

   *d_ic_out = csc_i;
   *d_jc_out = csc_j;
   *d_ac_out = csc_a;

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_SyncCudaDevice(nalu_hypre_handle())
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPTRANS] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}

#endif // #if defined(NALU_HYPRE_USING_ROCSPARSE)

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

NALU_HYPRE_Int
hypreDevice_CSRSpTrans(NALU_HYPRE_Int   m,        NALU_HYPRE_Int   n,        NALU_HYPRE_Int       nnzA,
                       NALU_HYPRE_Int  *d_ia,     NALU_HYPRE_Int  *d_ja,     NALU_HYPRE_Complex  *d_aa,
                       NALU_HYPRE_Int **d_ic_out, NALU_HYPRE_Int **d_jc_out, NALU_HYPRE_Complex **d_ac_out,
                       NALU_HYPRE_Int   want_data)
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPTRANS] -= nalu_hypre_MPI_Wtime();
#endif

   NALU_HYPRE_Int *d_jt, *d_it, *d_pm, *d_ic, *d_jc;
   NALU_HYPRE_Complex *d_ac = NULL;
   NALU_HYPRE_Int *mem_work = nalu_hypre_TAlloc(NALU_HYPRE_Int, 3 * nnzA, NALU_HYPRE_MEMORY_DEVICE);

   /* allocate C */
   d_jc = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnzA, NALU_HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      d_ac = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzA, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* permutation vector */
   //d_pm = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnzA, NALU_HYPRE_MEMORY_DEVICE);
   d_pm = mem_work;

   /* expansion: A's row idx */
   //d_it = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnzA, NALU_HYPRE_MEMORY_DEVICE);
   d_it = d_pm + nnzA;
   hypreDevice_CsrRowPtrsToIndices_v2(m, nnzA, d_ia, d_it);

   /* a copy of col idx of A */
   //d_jt = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnzA, NALU_HYPRE_MEMORY_DEVICE);
   d_jt = d_it + nnzA;
   nalu_hypre_TMemcpy(d_jt, d_ja, NALU_HYPRE_Int, nnzA, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   /* sort: by col */
   NALU_HYPRE_THRUST_CALL(sequence, d_pm, d_pm + nnzA);
   NALU_HYPRE_THRUST_CALL(stable_sort_by_key, d_jt, d_jt + nnzA, d_pm);
   NALU_HYPRE_THRUST_CALL(gather, d_pm, d_pm + nnzA, d_it, d_jc);
   if (want_data)
   {
      NALU_HYPRE_THRUST_CALL(gather, d_pm, d_pm + nnzA, d_aa, d_ac);
   }

   /* convert into ic: row idx --> row ptrs */
   d_ic = hypreDevice_CsrRowIndicesToPtrs(n, nnzA, d_jt);

#ifdef NALU_HYPRE_DEBUG
   NALU_HYPRE_Int nnzC;
   nalu_hypre_TMemcpy(&nnzC, &d_ic[n], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_assert(nnzC == nnzA);
#endif

   /*
   nalu_hypre_TFree(d_jt, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_it, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_pm, NALU_HYPRE_MEMORY_DEVICE);
   */
   nalu_hypre_TFree(mem_work, NALU_HYPRE_MEMORY_DEVICE);

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_ac_out = d_ac;

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPTRANS] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}

#endif /* NALU_HYPRE_USING_CUDA  || defined(NALU_HYPRE_USING_HIP) */

#if defined(NALU_HYPRE_USING_SYCL)
NALU_HYPRE_Int
hypreDevice_CSRSpTrans(NALU_HYPRE_Int   m,        NALU_HYPRE_Int   n,        NALU_HYPRE_Int       nnzA,
                       NALU_HYPRE_Int  *d_ia,     NALU_HYPRE_Int  *d_ja,     NALU_HYPRE_Complex  *d_aa,
                       NALU_HYPRE_Int **d_ic_out, NALU_HYPRE_Int **d_jc_out, NALU_HYPRE_Complex **d_ac_out,
                       NALU_HYPRE_Int   want_data)
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPTRANS] -= nalu_hypre_MPI_Wtime();
#endif

   NALU_HYPRE_Int *d_jt, *d_it, *d_pm, *d_ic, *d_jc;
   NALU_HYPRE_Complex *d_ac = NULL;
   NALU_HYPRE_Int *mem_work = nalu_hypre_TAlloc(NALU_HYPRE_Int, 3 * nnzA, NALU_HYPRE_MEMORY_DEVICE);

   /* allocate C */
   d_jc = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnzA, NALU_HYPRE_MEMORY_DEVICE);
   if (want_data)
   {
      d_ac = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzA, NALU_HYPRE_MEMORY_DEVICE);
   }

   /* permutation vector */
   d_pm = mem_work;

   /* expansion: A's row idx */
   d_it = d_pm + nnzA;
   hypreDevice_CsrRowPtrsToIndices_v2(m, nnzA, d_ia, d_it);

   /* a copy of col idx of A */
   d_jt = d_it + nnzA;
   nalu_hypre_TMemcpy(d_jt, d_ja, NALU_HYPRE_Int, nnzA, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   /* sort: by col */
   oneapi::dpl::counting_iterator<NALU_HYPRE_Int> count(0);
   NALU_HYPRE_ONEDPL_CALL( std::copy,
                      count,
                      count + nnzA,
                      d_pm);

   auto zip_jt_pm = oneapi::dpl::make_zip_iterator(d_jt, d_pm);
   NALU_HYPRE_ONEDPL_CALL( std::stable_sort,
                      zip_jt_pm,
                      zip_jt_pm + nnzA,
   [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); } );

   auto permuted_it = oneapi::dpl::make_permutation_iterator(d_it, d_pm);
   NALU_HYPRE_ONEDPL_CALL( std::copy,
                      permuted_it,
                      permuted_it + nnzA,
                      d_jc );

   if (want_data)
   {
      auto permuted_aa = oneapi::dpl::make_permutation_iterator(d_aa, d_pm);
      NALU_HYPRE_ONEDPL_CALL( std::copy,
                         permuted_aa,
                         permuted_aa + nnzA,
                         d_ac );
   }

   /* convert into ic: row idx --> row ptrs */
   d_ic = hypreDevice_CsrRowIndicesToPtrs(n, nnzA, d_jt);

#ifdef NALU_HYPRE_DEBUG
   NALU_HYPRE_Int nnzC;
   nalu_hypre_TMemcpy(&nnzC, &d_ic[n], NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_assert(nnzC == nnzA);
#endif

   nalu_hypre_TFree(mem_work, NALU_HYPRE_MEMORY_DEVICE);

   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_ac_out = d_ac;

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_SPTRANS] += nalu_hypre_MPI_Wtime();
#endif

   return nalu_hypre_error_flag;
}
#endif // #if defined(NALU_HYPRE_USING_SYCL)
