/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "_nalu_hypre_utilities.hpp"
#include "seq_mv.hpp"

#if defined(NALU_HYPRE_USING_SYCL) && defined(NALU_HYPRE_USING_ONEMKLSPARSE)

NALU_HYPRE_Int
hypreDevice_CSRSpGemmOnemklsparse(NALU_HYPRE_Int                            m,
                                  NALU_HYPRE_Int                            k,
                                  NALU_HYPRE_Int                            n,
                                  oneapi::mkl::sparse::matrix_handle_t handle_A,
                                  NALU_HYPRE_Int                            nnzA,
                                  NALU_HYPRE_Int                           *_d_ia,
                                  NALU_HYPRE_Int                           *_d_ja,
                                  NALU_HYPRE_Complex                       *d_a,
                                  oneapi::mkl::sparse::matrix_handle_t handle_B,
                                  NALU_HYPRE_Int                            nnzB,
                                  NALU_HYPRE_Int                           *_d_ib,
                                  NALU_HYPRE_Int                           *_d_jb,
                                  NALU_HYPRE_Complex                       *d_b,
                                  oneapi::mkl::sparse::matrix_handle_t handle_C,
                                  NALU_HYPRE_Int                           *nnzC_out,
                                  NALU_HYPRE_Int                          **d_ic_out,
                                  NALU_HYPRE_Int                          **d_jc_out,
                                  NALU_HYPRE_Complex                      **d_c_out)
{
   /* Need these conversions in the case of the bigint build */
#if defined(NALU_HYPRE_BIGINT)
   std::int64_t *d_ia      = reinterpret_cast<std::int64_t*>(_d_ia);
   std::int64_t *d_ja      = reinterpret_cast<std::int64_t*>(_d_ja);
   std::int64_t *d_ib      = reinterpret_cast<std::int64_t*>(_d_ib);
   std::int64_t *d_jb      = reinterpret_cast<std::int64_t*>(_d_jb);

   std::int64_t *d_ic, *d_jc = NULL;
   std::int64_t *d_ja_sorted, *d_jb_sorted;

   /* Allocate space for sorted arrays */
   d_ja_sorted = nalu_hypre_TAlloc(std::int64_t,     nnzA, NALU_HYPRE_MEMORY_DEVICE);
   d_jb_sorted = nalu_hypre_TAlloc(std::int64_t,     nnzB, NALU_HYPRE_MEMORY_DEVICE);

   /* Copy the unsorted over as the initial "sorted" */
   nalu_hypre_TMemcpy(d_ja_sorted, d_ja, std::int64_t,     nnzA, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(d_jb_sorted, d_jb, std::int64_t,     nnzB, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
#else
   NALU_HYPRE_Int *d_ia      = _d_ia;
   NALU_HYPRE_Int *d_ja      = _d_ja;
   NALU_HYPRE_Int *d_ib      = _d_ib;
   NALU_HYPRE_Int *d_jb      = _d_jb;

   NALU_HYPRE_Int *d_ic, *d_jc = NULL;
   NALU_HYPRE_Int *d_ja_sorted, *d_jb_sorted;

   /* Allocate space for sorted arrays */
   d_ja_sorted = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nnzA, NALU_HYPRE_MEMORY_DEVICE);
   d_jb_sorted = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nnzB, NALU_HYPRE_MEMORY_DEVICE);

   /* Copy the unsorted over as the initial "sorted" */
   nalu_hypre_TMemcpy(d_ja_sorted, d_ja, NALU_HYPRE_Int,     nnzA, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(d_jb_sorted, d_jb, NALU_HYPRE_Int,     nnzB, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
#endif

   std::int64_t *tmp_size1_h = NULL, *tmp_size1_d = NULL;
   std::int64_t *tmp_size2_h = NULL, *tmp_size2_d = NULL;
   std::int64_t *nnzC_h = NULL, *nnzC_d;
   void *tmp_buffer1 = NULL;
   void *tmp_buffer2 = NULL;
   NALU_HYPRE_Complex *d_c = NULL;
   NALU_HYPRE_Complex *d_a_sorted, *d_b_sorted;

   /* Allocate space for sorted arrays */
   d_a_sorted  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzA, NALU_HYPRE_MEMORY_DEVICE);
   d_b_sorted  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzB, NALU_HYPRE_MEMORY_DEVICE);

   /* Copy the unsorted over as the initial "sorted" */
   nalu_hypre_TMemcpy(d_a_sorted,  d_a,  NALU_HYPRE_Complex, nnzA, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(d_b_sorted,  d_b,  NALU_HYPRE_Complex, nnzB, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   /* sort copies of col indices and data for A and B */
   /* WM: todo - this is currently necessary for correctness of oneMKL's matmat, but this may change in the future? */
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::set_csr_data(handle_A, m, k, oneapi::mkl::index_base::zero,
                                                        d_ia, d_ja_sorted, d_a_sorted) );
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::set_csr_data(handle_B, k, n, oneapi::mkl::index_base::zero,
                                                        d_ib, d_jb_sorted, d_b_sorted) );
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::sort_matrix(*nalu_hypre_HandleComputeStream(nalu_hypre_handle()),
                                                       handle_A, {}).wait() );
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::sort_matrix(*nalu_hypre_HandleComputeStream(nalu_hypre_handle()),
                                                       handle_B, {}).wait() );

   oneapi::mkl::sparse::matmat_descr_t descr = NULL;
   oneapi::mkl::sparse::matmat_request req;

#if defined(NALU_HYPRE_BIGINT)
   d_ic = nalu_hypre_TAlloc(std::int64_t, m + 1, NALU_HYPRE_MEMORY_DEVICE);
#else
   d_ic = nalu_hypre_TAlloc(NALU_HYPRE_Int, m + 1, NALU_HYPRE_MEMORY_DEVICE);
#endif
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::set_csr_data(handle_C, m, n, oneapi::mkl::index_base::zero,
                                                        d_ic, d_jc, d_c) );

   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::init_matmat_descr(&descr) );
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::set_matmat_data(descr,
                                                           oneapi::mkl::sparse::matrix_view_descr::general,
                                                           oneapi::mkl::transpose::nontrans,
                                                           oneapi::mkl::sparse::matrix_view_descr::general,
                                                           oneapi::mkl::transpose::nontrans,
                                                           oneapi::mkl::sparse::matrix_view_descr::general) );

   /* get tmp_buffer1 size for work estimation */
   req = oneapi::mkl::sparse::matmat_request::get_work_estimation_buf_size;
   tmp_size1_d = nalu_hypre_TAlloc(std::int64_t, 1, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::matmat(*nalu_hypre_HandleComputeStream(nalu_hypre_handle()),
                                                  handle_A,
                                                  handle_B,
                                                  handle_C,
                                                  req,
                                                  descr,
                                                  tmp_size1_d,
                                                  NULL,
                                                  {}).wait() );

   /* allocate tmp_buffer1 for work estimation */
   tmp_size1_h = nalu_hypre_TAlloc(std::int64_t, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TMemcpy(tmp_size1_h, tmp_size1_d, std::int64_t, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   tmp_buffer1 = (void*) nalu_hypre_TAlloc(std::uint8_t, *tmp_size1_h, NALU_HYPRE_MEMORY_DEVICE);

   /* do work_estimation */
   req = oneapi::mkl::sparse::matmat_request::work_estimation;
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::matmat(*nalu_hypre_HandleComputeStream(nalu_hypre_handle()),
                                                  handle_A,
                                                  handle_B,
                                                  handle_C,
                                                  req,
                                                  descr,
                                                  tmp_size1_d,
                                                  tmp_buffer1,
                                                  {}).wait() );

   /* get tmp_buffer2 size for computation */
   req = oneapi::mkl::sparse::matmat_request::get_compute_buf_size;
   tmp_size2_d = nalu_hypre_TAlloc(std::int64_t, 1, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::matmat(*nalu_hypre_HandleComputeStream(nalu_hypre_handle()),
                                                  handle_A,
                                                  handle_B,
                                                  handle_C,
                                                  req,
                                                  descr,
                                                  tmp_size2_d,
                                                  NULL,
                                                  {}).wait() );

   /* allocate tmp_buffer2 for computation */
   tmp_size2_h = nalu_hypre_TAlloc(std::int64_t, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TMemcpy(tmp_size2_h, tmp_size2_d, std::int64_t, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
   tmp_buffer2 = (void*) nalu_hypre_TAlloc(std::uint8_t, *tmp_size2_h, NALU_HYPRE_MEMORY_DEVICE);

   /* do the computation */
   req = oneapi::mkl::sparse::matmat_request::compute;
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::matmat(*nalu_hypre_HandleComputeStream(nalu_hypre_handle()),
                                                  handle_A,
                                                  handle_B,
                                                  handle_C,
                                                  req,
                                                  descr,
                                                  tmp_size2_d,
                                                  tmp_buffer2,
                                                  {}).wait() );

   /* get nnzC */
   req = oneapi::mkl::sparse::matmat_request::get_nnz;
   nnzC_d = nalu_hypre_TAlloc(std::int64_t, 1, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::matmat(*nalu_hypre_HandleComputeStream(nalu_hypre_handle()),
                                                  handle_A,
                                                  handle_B,
                                                  handle_C,
                                                  req,
                                                  descr,
                                                  nnzC_d,
                                                  NULL,
                                                  {}).wait() );

   /* allocate col index and data arrays */
   nnzC_h = nalu_hypre_TAlloc(std::int64_t, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TMemcpy(nnzC_h, nnzC_d, std::int64_t, 1, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_BIGINT)
   d_jc = nalu_hypre_TAlloc(std::int64_t, *nnzC_h, NALU_HYPRE_MEMORY_DEVICE);
#else
   d_jc = nalu_hypre_TAlloc(NALU_HYPRE_Int, *nnzC_h, NALU_HYPRE_MEMORY_DEVICE);
#endif
   d_c = nalu_hypre_TAlloc(NALU_HYPRE_Complex, *nnzC_h, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::set_csr_data(handle_C, m, n, oneapi::mkl::index_base::zero,
                                                        d_ic, d_jc, d_c) );

   /* finalize C */
   req = oneapi::mkl::sparse::matmat_request::finalize;
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::matmat(*nalu_hypre_HandleComputeStream(nalu_hypre_handle()),
                                                  handle_A,
                                                  handle_B,
                                                  handle_C,
                                                  req,
                                                  descr,
                                                  NULL,
                                                  NULL,
                                                  {}).wait() );

   /* release the matmat descr */
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::release_matmat_descr(&descr) );

   /* assign the output */
   *nnzC_out = *nnzC_h;
#if defined(NALU_HYPRE_BIGINT)
   *d_ic_out = reinterpret_cast<NALU_HYPRE_Int*>(d_ic);
   *d_jc_out = reinterpret_cast<NALU_HYPRE_Int*>(d_jc);
#else
   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
#endif
   *d_c_out = d_c;

   /* restore the original (unsorted) col indices and data to A and B and free sorted arrays */
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::set_csr_data(handle_A, m, k, oneapi::mkl::index_base::zero,
                                                        d_ia, d_ja_sorted, d_a_sorted) );
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::set_csr_data(handle_B, k, n, oneapi::mkl::index_base::zero,
                                                        d_ib, d_jb_sorted, d_b_sorted) );
   nalu_hypre_TFree(d_a_sorted,  NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_b_sorted,  NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_ja_sorted, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(d_jb_sorted, NALU_HYPRE_MEMORY_DEVICE);

   /* free temporary arrays */
   nalu_hypre_TFree(tmp_size1_h, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(tmp_size1_d, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(tmp_size2_h, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(tmp_size2_d, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(nnzC_h, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nnzC_d, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(tmp_buffer1, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(tmp_buffer2, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}
#endif
