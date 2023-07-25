/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matrix operation functions for nalu_hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "_nalu_hypre_onedpl.hpp"
#include "seq_mv.h"
#include "_nalu_hypre_utilities.hpp"
#include "seq_mv.hpp"

#if defined(NALU_HYPRE_USING_CUSPARSE)  ||\
    defined(NALU_HYPRE_USING_ROCSPARSE) ||\
    defined(NALU_HYPRE_USING_ONEMKLSPARSE)

/*--------------------------------------------------------------------------
 * nalu_hypre_CsrsvDataCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_CsrsvData*
nalu_hypre_CsrsvDataCreate()
{
   nalu_hypre_CsrsvData *data = nalu_hypre_CTAlloc(nalu_hypre_CsrsvData, 1, NALU_HYPRE_MEMORY_HOST);

#if defined(NALU_HYPRE_USING_CUSPARSE)
   NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparseSpSV_createDescr(&nalu_hypre_CsrsvDataInfoL(data)) );
   NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparseSpSV_createDescr(&nalu_hypre_CsrsvDataInfoU(data)) );

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   NALU_HYPRE_ROCSPARSE_CALL( rocsparse_create_mat_info(&(nalu_hypre_CsrsvDataInfoL(data)) ) );
   NALU_HYPRE_ROCSPARSE_CALL( rocsparse_create_mat_info(&(nalu_hypre_CsrsvDataInfoU(data)) ) );
#endif

   nalu_hypre_CsrsvDataAnalyzedL(data) = 0;
   nalu_hypre_CsrsvDataAnalyzedU(data) = 0;

   return data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CsrsvDataDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CsrsvDataDestroy(nalu_hypre_CsrsvData* data)
{
   if (data)
   {
      /* Lower matrix info */
      if (nalu_hypre_CsrsvDataInfoL(data))
      {
#if defined(NALU_HYPRE_USING_CUSPARSE)
         NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparseSpSV_destroyDescr(nalu_hypre_CsrsvDataInfoL(data)) );

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
         NALU_HYPRE_ROCSPARSE_CALL( rocsparse_destroy_mat_info(nalu_hypre_CsrsvDataInfoL(data)) );
#endif
      }

      /* Upper matrix info */
      if (nalu_hypre_CsrsvDataInfoU(data))
      {
#if defined(NALU_HYPRE_USING_CUSPARSE)
         NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparseSpSV_destroyDescr(nalu_hypre_CsrsvDataInfoU(data)) );

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
         NALU_HYPRE_ROCSPARSE_CALL( rocsparse_destroy_mat_info(nalu_hypre_CsrsvDataInfoU(data)) );
#endif
      }

      /* Buffers */
#if defined(NALU_HYPRE_USING_CUSPARSE) && (CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION)
      nalu_hypre_TFree(nalu_hypre_CsrsvDataBufferL(data), NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(nalu_hypre_CsrsvDataBufferU(data), NALU_HYPRE_MEMORY_DEVICE);
#else
      nalu_hypre_TFree(nalu_hypre_CsrsvDataBuffer(data), NALU_HYPRE_MEMORY_DEVICE);
#endif
      nalu_hypre_TFree(nalu_hypre_CsrsvDataMatData(data), NALU_HYPRE_MEMORY_DEVICE);

      /* Free data structure pointer */
      nalu_hypre_TFree(data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_GpuMatDataCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_GpuMatData *
nalu_hypre_GpuMatDataCreate()
{
   nalu_hypre_GpuMatData *data = nalu_hypre_CTAlloc(nalu_hypre_GpuMatData, 1, NALU_HYPRE_MEMORY_HOST);

#if defined(NALU_HYPRE_USING_CUSPARSE)
   cusparseMatDescr_t mat_descr;

   NALU_HYPRE_CUSPARSE_CALL( cusparseCreateMatDescr(&mat_descr) );
   NALU_HYPRE_CUSPARSE_CALL( cusparseSetMatType(mat_descr, CUSPARSE_MATRIX_TYPE_GENERAL) );
   NALU_HYPRE_CUSPARSE_CALL( cusparseSetMatIndexBase(mat_descr, CUSPARSE_INDEX_BASE_ZERO) );
   nalu_hypre_GpuMatDataMatDescr(data) = mat_descr;

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   rocsparse_mat_descr mat_descr;
   rocsparse_mat_info  info;

   NALU_HYPRE_ROCSPARSE_CALL( rocsparse_create_mat_descr(&mat_descr) );
   NALU_HYPRE_ROCSPARSE_CALL( rocsparse_set_mat_type(mat_descr, rocsparse_matrix_type_general) );
   NALU_HYPRE_ROCSPARSE_CALL( rocsparse_set_mat_index_base(mat_descr, rocsparse_index_base_zero) );
   NALU_HYPRE_ROCSPARSE_CALL( rocsparse_create_mat_info(&info) );

   nalu_hypre_GpuMatDataMatDescr(data) = mat_descr;
   nalu_hypre_GpuMatDataMatInfo(data) = info;

#elif defined(NALU_HYPRE_USING_ONEMKLSPARSE)
   oneapi::mkl::sparse::matrix_handle_t mat_handle;
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::init_matrix_handle(&mat_handle) );
   nalu_hypre_GpuMatDataMatHandle(data) = mat_handle;
#endif

   return data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_GPUMatDataSetCSRData
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_GPUMatDataSetCSRData(nalu_hypre_CSRMatrix *matrix)
{

#if defined(NALU_HYPRE_USING_ONEMKLSPARSE)
#if defined(NALU_HYPRE_BIGINT)
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::set_csr_data(nalu_hypre_CSRMatrixGPUMatHandle(matrix),
                                                        nalu_hypre_CSRMatrixNumRows(matrix),
                                                        nalu_hypre_CSRMatrixNumCols(matrix),
                                                        oneapi::mkl::index_base::zero,
                                                        reinterpret_cast<std::int64_t*>(nalu_hypre_CSRMatrixI(matrix)),
                                                        reinterpret_cast<std::int64_t*>(nalu_hypre_CSRMatrixJ(matrix)),
                                                        nalu_hypre_CSRMatrixData(matrix)) );
#else
   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::set_csr_data(nalu_hypre_CSRMatrixGPUMatHandle(matrix),
                                                        nalu_hypre_CSRMatrixNumRows(matrix),
                                                        nalu_hypre_CSRMatrixNumCols(matrix),
                                                        oneapi::mkl::index_base::zero,
                                                        nalu_hypre_CSRMatrixI(matrix),
                                                        nalu_hypre_CSRMatrixJ(matrix),
                                                        nalu_hypre_CSRMatrixData(matrix)) );
#endif
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_GpuMatDataDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_GpuMatDataDestroy(nalu_hypre_GpuMatData *data)
{
   if (data)
   {
#if defined(NALU_HYPRE_USING_CUSPARSE)
      NALU_HYPRE_CUSPARSE_CALL( cusparseDestroyMatDescr(nalu_hypre_GpuMatDataMatDescr(data)) );
      nalu_hypre_TFree(nalu_hypre_GpuMatDataSpMVBuffer(data), NALU_HYPRE_MEMORY_DEVICE);

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
      NALU_HYPRE_ROCSPARSE_CALL( rocsparse_destroy_mat_descr(nalu_hypre_GpuMatDataMatDescr(data)) );
      NALU_HYPRE_ROCSPARSE_CALL( rocsparse_destroy_mat_info(nalu_hypre_GpuMatDataMatInfo(data)) );

#elif defined(NALU_HYPRE_USING_ONEMKLSPARSE)
      NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::release_matrix_handle(&nalu_hypre_GpuMatDataMatHandle(data)) );
#endif
   }

   nalu_hypre_TFree(data, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

#endif /* #if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) || defined(NALU_HYPRE_USING_ONEMKLSPARSE) */

#if defined(NALU_HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixAddDevice
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix*
nalu_hypre_CSRMatrixAddDevice( NALU_HYPRE_Complex    alpha,
                          nalu_hypre_CSRMatrix *A,
                          NALU_HYPRE_Complex    beta,
                          nalu_hypre_CSRMatrix *B )
{
   NALU_HYPRE_Complex    *A_data   = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int        *A_i      = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int        *A_j      = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int         nrows_A  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int         ncols_A  = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int         nnz_A    = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Complex    *B_data   = nalu_hypre_CSRMatrixData(B);
   NALU_HYPRE_Int        *B_i      = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int        *B_j      = nalu_hypre_CSRMatrixJ(B);
   NALU_HYPRE_Int         nrows_B  = nalu_hypre_CSRMatrixNumRows(B);
   NALU_HYPRE_Int         ncols_B  = nalu_hypre_CSRMatrixNumCols(B);
   NALU_HYPRE_Int         nnz_B    = nalu_hypre_CSRMatrixNumNonzeros(B);
   NALU_HYPRE_Complex    *C_data;
   NALU_HYPRE_Int        *C_i;
   NALU_HYPRE_Int        *C_j;
   NALU_HYPRE_Int         nnzC;
   nalu_hypre_CSRMatrix  *C;

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Warning! Incompatible matrix dimensions!\n");

      return NULL;
   }

   hypreDevice_CSRSpAdd(nrows_A, nrows_B, ncols_A, nnz_A, nnz_B,
                        A_i, A_j, alpha, A_data, NULL, B_i, B_j, beta, B_data, NULL, NULL,
                        &nnzC, &C_i, &C_j, &C_data);

   C = nalu_hypre_CSRMatrixCreate(nrows_A, ncols_B, nnzC);
   nalu_hypre_CSRMatrixI(C) = C_i;
   nalu_hypre_CSRMatrixJ(C) = C_j;
   nalu_hypre_CSRMatrixData(C) = C_data;
   nalu_hypre_CSRMatrixMemoryLocation(C) = NALU_HYPRE_MEMORY_DEVICE;

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return C;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixMultiplyDevice
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix*
nalu_hypre_CSRMatrixMultiplyDevice( nalu_hypre_CSRMatrix *A,
                               nalu_hypre_CSRMatrix *B )
{
   NALU_HYPRE_Int         ncols_A  = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int         nrows_B  = nalu_hypre_CSRMatrixNumRows(B);
   nalu_hypre_CSRMatrix  *C;

   if (ncols_A != nrows_B)
   {
      nalu_hypre_printf("Warning! incompatible matrix dimensions!\n");
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Warning! incompatible matrix dimensions!\n");

      return NULL;
   }

   nalu_hypre_GpuProfilingPushRange("CSRMatrixMultiply");

   hypreDevice_CSRSpGemm(A, B, &C);

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   nalu_hypre_GpuProfilingPopRange();

   return C;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixTripleMultiplyDevice
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix*
nalu_hypre_CSRMatrixTripleMultiplyDevice ( nalu_hypre_CSRMatrix *A,
                                      nalu_hypre_CSRMatrix *B,
                                      nalu_hypre_CSRMatrix *C )
{
   nalu_hypre_CSRMatrix *BC  = nalu_hypre_CSRMatrixMultiplyDevice(B, C);
   nalu_hypre_CSRMatrix *ABC = nalu_hypre_CSRMatrixMultiplyDevice(A, BC);

   nalu_hypre_CSRMatrixDestroy(BC);

   return ABC;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixSplitDevice
 *
 * Split CSR matrix B_ext (extended rows of parcsr B) into diag part and
 * offd part corresponding to B.
 *
 * Input  - col_map_offd_B:
 * Output - col_map_offd_C: union of col_map_offd_B and offd-indices of
 *                          Bext_offd
 *          map_B_to_C: mapping from col_map_offd_B to col_map_offd_C
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixSplitDevice( nalu_hypre_CSRMatrix  *B_ext,
                            NALU_HYPRE_BigInt      first_col_diag_B,
                            NALU_HYPRE_BigInt      last_col_diag_B,
                            NALU_HYPRE_Int         num_cols_offd_B,
                            NALU_HYPRE_BigInt     *col_map_offd_B,
                            NALU_HYPRE_Int       **map_B_to_C_ptr,
                            NALU_HYPRE_Int        *num_cols_offd_C_ptr,
                            NALU_HYPRE_BigInt    **col_map_offd_C_ptr,
                            nalu_hypre_CSRMatrix **B_ext_diag_ptr,
                            nalu_hypre_CSRMatrix **B_ext_offd_ptr )
{
   NALU_HYPRE_Int num_rows = nalu_hypre_CSRMatrixNumRows(B_ext);
   NALU_HYPRE_Int B_ext_nnz = nalu_hypre_CSRMatrixNumNonzeros(B_ext);

   NALU_HYPRE_Int *B_ext_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int, B_ext_nnz, NALU_HYPRE_MEMORY_DEVICE);
   hypreDevice_CsrRowPtrsToIndices_v2(num_rows, B_ext_nnz, nalu_hypre_CSRMatrixI(B_ext), B_ext_ii);

   NALU_HYPRE_Int B_ext_diag_nnz;
   NALU_HYPRE_Int B_ext_offd_nnz;
   NALU_HYPRE_Int ierr;

   ierr = nalu_hypre_CSRMatrixSplitDevice_core( 0,
                                           num_rows,
                                           B_ext_nnz,
                                           NULL,
                                           nalu_hypre_CSRMatrixBigJ(B_ext),
                                           NULL,
                                           NULL,
                                           first_col_diag_B,
                                           last_col_diag_B,
                                           num_cols_offd_B,
                                           NULL,
                                           NULL,
                                           NULL,
                                           NULL,
                                           &B_ext_diag_nnz,
                                           NULL,
                                           NULL,
                                           NULL,
                                           NULL,
                                           &B_ext_offd_nnz,
                                           NULL,
                                           NULL,
                                           NULL,
                                           NULL );

   NALU_HYPRE_Int     *B_ext_diag_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     B_ext_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int     *B_ext_diag_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     B_ext_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Complex *B_ext_diag_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, B_ext_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_Int     *B_ext_offd_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int,     B_ext_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int     *B_ext_offd_j  = nalu_hypre_TAlloc(NALU_HYPRE_Int,     B_ext_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Complex *B_ext_offd_a  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, B_ext_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);

   ierr = nalu_hypre_CSRMatrixSplitDevice_core( 1,
                                           num_rows,
                                           B_ext_nnz,
                                           B_ext_ii,
                                           nalu_hypre_CSRMatrixBigJ(B_ext),
                                           nalu_hypre_CSRMatrixData(B_ext),
                                           NULL,
                                           first_col_diag_B,
                                           last_col_diag_B,
                                           num_cols_offd_B,
                                           col_map_offd_B,
                                           map_B_to_C_ptr,
                                           num_cols_offd_C_ptr,
                                           col_map_offd_C_ptr,
                                           &B_ext_diag_nnz,
                                           B_ext_diag_ii,
                                           B_ext_diag_j,
                                           B_ext_diag_a,
                                           NULL,
                                           &B_ext_offd_nnz,
                                           B_ext_offd_ii,
                                           B_ext_offd_j,
                                           B_ext_offd_a,
                                           NULL );

   nalu_hypre_TFree(B_ext_ii, NALU_HYPRE_MEMORY_DEVICE);

   /* convert to row ptrs */
   NALU_HYPRE_Int *B_ext_diag_i = hypreDevice_CsrRowIndicesToPtrs(num_rows, B_ext_diag_nnz, B_ext_diag_ii);
   NALU_HYPRE_Int *B_ext_offd_i = hypreDevice_CsrRowIndicesToPtrs(num_rows, B_ext_offd_nnz, B_ext_offd_ii);

   nalu_hypre_TFree(B_ext_diag_ii, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(B_ext_offd_ii, NALU_HYPRE_MEMORY_DEVICE);

   /* create diag and offd CSR */
   nalu_hypre_CSRMatrix *B_ext_diag = nalu_hypre_CSRMatrixCreate(num_rows,
                                                       last_col_diag_B - first_col_diag_B + 1, B_ext_diag_nnz);
   nalu_hypre_CSRMatrix *B_ext_offd = nalu_hypre_CSRMatrixCreate(num_rows, *num_cols_offd_C_ptr, B_ext_offd_nnz);

   nalu_hypre_CSRMatrixI(B_ext_diag) = B_ext_diag_i;
   nalu_hypre_CSRMatrixJ(B_ext_diag) = B_ext_diag_j;
   nalu_hypre_CSRMatrixData(B_ext_diag) = B_ext_diag_a;
   nalu_hypre_CSRMatrixNumNonzeros(B_ext_diag) = B_ext_diag_nnz;
   nalu_hypre_CSRMatrixMemoryLocation(B_ext_diag) = NALU_HYPRE_MEMORY_DEVICE;

   nalu_hypre_CSRMatrixI(B_ext_offd) = B_ext_offd_i;
   nalu_hypre_CSRMatrixJ(B_ext_offd) = B_ext_offd_j;
   nalu_hypre_CSRMatrixData(B_ext_offd) = B_ext_offd_a;
   nalu_hypre_CSRMatrixNumNonzeros(B_ext_offd) = B_ext_offd_nnz;
   nalu_hypre_CSRMatrixMemoryLocation(B_ext_offd) = NALU_HYPRE_MEMORY_DEVICE;

   *B_ext_diag_ptr = B_ext_diag;
   *B_ext_offd_ptr = B_ext_offd;

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixMergeColMapOffd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixMergeColMapOffd( NALU_HYPRE_Int      num_cols_offd_B,
                                NALU_HYPRE_BigInt  *col_map_offd_B,
                                NALU_HYPRE_Int      B_ext_offd_nnz,
                                NALU_HYPRE_BigInt  *B_ext_offd_bigj,
                                NALU_HYPRE_Int     *num_cols_offd_C_ptr,
                                NALU_HYPRE_BigInt **col_map_offd_C_ptr,
                                NALU_HYPRE_Int    **map_B_to_C_ptr )
{
   /* offd map of B_ext_offd Union col_map_offd_B */
   NALU_HYPRE_BigInt *col_map_offd_C = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, B_ext_offd_nnz + num_cols_offd_B,
                                               NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_TMemcpy(col_map_offd_C, B_ext_offd_bigj, NALU_HYPRE_BigInt, B_ext_offd_nnz,
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_TMemcpy(col_map_offd_C + B_ext_offd_nnz, col_map_offd_B, NALU_HYPRE_BigInt, num_cols_offd_B,
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::sort,
                      col_map_offd_C,
                      col_map_offd_C + B_ext_offd_nnz + num_cols_offd_B );

   NALU_HYPRE_BigInt *new_end = NALU_HYPRE_ONEDPL_CALL( std::unique,
                                              col_map_offd_C,
                                              col_map_offd_C + B_ext_offd_nnz + num_cols_offd_B );
#else
   NALU_HYPRE_THRUST_CALL( sort,
                      col_map_offd_C,
                      col_map_offd_C + B_ext_offd_nnz + num_cols_offd_B );

   NALU_HYPRE_BigInt *new_end = NALU_HYPRE_THRUST_CALL( unique,
                                              col_map_offd_C,
                                              col_map_offd_C + B_ext_offd_nnz + num_cols_offd_B );
#endif

   NALU_HYPRE_Int num_cols_offd_C = new_end - col_map_offd_C;

#if 1
   NALU_HYPRE_BigInt *tmp = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_C, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(tmp, col_map_offd_C, NALU_HYPRE_BigInt, num_cols_offd_C, NALU_HYPRE_MEMORY_DEVICE,
                 NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(col_map_offd_C, NALU_HYPRE_MEMORY_DEVICE);
   col_map_offd_C = tmp;
#else
   col_map_offd_C = nalu_hypre_TReAlloc_v2(col_map_offd_C, NALU_HYPRE_BigInt, B_ext_offd_nnz + num_cols_offd_B,
                                      NALU_HYPRE_Int, num_cols_offd_C, NALU_HYPRE_MEMORY_DEVICE);
#endif

   /* create map from col_map_offd_B */
   NALU_HYPRE_Int *map_B_to_C = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_offd_B, NALU_HYPRE_MEMORY_DEVICE);

   if (num_cols_offd_B)
   {
#if defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                         col_map_offd_C,
                         col_map_offd_C + num_cols_offd_C,
                         col_map_offd_B,
                         col_map_offd_B + num_cols_offd_B,
                         map_B_to_C );
#else
      NALU_HYPRE_THRUST_CALL( lower_bound,
                         col_map_offd_C,
                         col_map_offd_C + num_cols_offd_C,
                         col_map_offd_B,
                         col_map_offd_B + num_cols_offd_B,
                         map_B_to_C );
#endif
   }

   *map_B_to_C_ptr = map_B_to_C;
   *num_cols_offd_C_ptr = num_cols_offd_C;
   *col_map_offd_C_ptr  = col_map_offd_C;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixSplitDevice_core
 *
 * job = 0: query B_ext_diag/offd_nnz; 1: real computation
 *
 * NOTES:
 *   B_ext_ii: NOT row pointers of CSR but row indices of COO
 *   B_ext_bigj: [BigInt] global column indices
 *   B_ext_xata: companion data with B_ext_data; NULL if none
 *   B_ext_diag_ii: memory allocated outside
 *   B_ext_diag_xata: companion with B_ext_diag_data_ptr; NULL if none
 *   B_ext_offd_ii: memory allocated outside
 *   B_ext_offd_xata: companion with B_ext_offd_data_ptr; NULL if none
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixSplitDevice_core( NALU_HYPRE_Int      job,
                                 NALU_HYPRE_Int      num_rows,
                                 NALU_HYPRE_Int      B_ext_nnz,
                                 NALU_HYPRE_Int     *B_ext_ii,
                                 NALU_HYPRE_BigInt  *B_ext_bigj,
                                 NALU_HYPRE_Complex *B_ext_data,
                                 char          *B_ext_xata,
                                 NALU_HYPRE_BigInt   first_col_diag_B,
                                 NALU_HYPRE_BigInt   last_col_diag_B,
                                 NALU_HYPRE_Int      num_cols_offd_B,
                                 NALU_HYPRE_BigInt  *col_map_offd_B,
                                 NALU_HYPRE_Int    **map_B_to_C_ptr,
                                 NALU_HYPRE_Int     *num_cols_offd_C_ptr,
                                 NALU_HYPRE_BigInt **col_map_offd_C_ptr,
                                 NALU_HYPRE_Int     *B_ext_diag_nnz_ptr,
                                 NALU_HYPRE_Int     *B_ext_diag_ii,
                                 NALU_HYPRE_Int     *B_ext_diag_j,
                                 NALU_HYPRE_Complex *B_ext_diag_data,
                                 char          *B_ext_diag_xata,
                                 NALU_HYPRE_Int     *B_ext_offd_nnz_ptr,
                                 NALU_HYPRE_Int     *B_ext_offd_ii,
                                 NALU_HYPRE_Int     *B_ext_offd_j,
                                 NALU_HYPRE_Complex *B_ext_offd_data,
                                 char          *B_ext_offd_xata )
{
   NALU_HYPRE_Int      B_ext_diag_nnz;
   NALU_HYPRE_Int      B_ext_offd_nnz;
   NALU_HYPRE_BigInt  *B_ext_diag_bigj = NULL;
   NALU_HYPRE_BigInt  *B_ext_offd_bigj = NULL;
   NALU_HYPRE_BigInt  *col_map_offd_C;
   NALU_HYPRE_Int     *map_B_to_C = NULL;
   NALU_HYPRE_Int      num_cols_offd_C;

   nalu_hypre_GpuProfilingPushRange("CSRMatrixSplitDevice_core");

   in_range<NALU_HYPRE_BigInt> pred1(first_col_diag_B, last_col_diag_B);

   /* get diag and offd nnz */
   if (job == 0)
   {
      /* query the nnz's */
#if defined(NALU_HYPRE_USING_SYCL)
      B_ext_diag_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                          B_ext_bigj,
                                          B_ext_bigj + B_ext_nnz,
                                          pred1 );
#else
      B_ext_diag_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                          B_ext_bigj,
                                          B_ext_bigj + B_ext_nnz,
                                          pred1 );
#endif
      B_ext_offd_nnz = B_ext_nnz - B_ext_diag_nnz;

      *B_ext_diag_nnz_ptr = B_ext_diag_nnz;
      *B_ext_offd_nnz_ptr = B_ext_offd_nnz;

      nalu_hypre_GpuProfilingPopRange();

      return nalu_hypre_error_flag;
   }
   else
   {
      B_ext_diag_nnz = *B_ext_diag_nnz_ptr;
      B_ext_offd_nnz = *B_ext_offd_nnz_ptr;
   }

   /* copy to diag */
   B_ext_diag_bigj = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, B_ext_diag_nnz, NALU_HYPRE_MEMORY_DEVICE);

   if (B_ext_diag_xata)
   {
#if defined(NALU_HYPRE_USING_SYCL)
      auto first = oneapi::dpl::make_zip_iterator(B_ext_ii, B_ext_bigj, B_ext_data, B_ext_xata);
      auto new_end = hypreSycl_copy_if(
                        first,                                                             /* first   */
                        first + B_ext_nnz,                                                 /* last    */
                        B_ext_bigj,                                                        /* stencil */
                        oneapi::dpl::make_zip_iterator(B_ext_diag_ii, B_ext_diag_bigj, B_ext_diag_data,
                                                       B_ext_diag_xata),                   /* result  */
                        pred1 );
      nalu_hypre_assert( std::get<0>(new_end.base()) == B_ext_diag_ii + B_ext_diag_nnz );
#else
      auto new_end = NALU_HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data,
                                                                     B_ext_xata)),             /* first */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data,
                                                                     B_ext_xata)) + B_ext_nnz, /* last */
                        B_ext_bigj,                                                            /* stencil */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_diag_ii, B_ext_diag_bigj, B_ext_diag_data,
                                                                     B_ext_diag_xata)),        /* result */
                        pred1 );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == B_ext_diag_ii + B_ext_diag_nnz );
#endif
   }
   else
   {
#if defined(NALU_HYPRE_USING_SYCL)
      auto first = oneapi::dpl::make_zip_iterator(B_ext_ii, B_ext_bigj, B_ext_data);
      auto new_end = hypreSycl_copy_if(
                        first,                                                                /* first   */
                        first + B_ext_nnz,                                                    /* last    */
                        B_ext_bigj,                                                           /* stencil */
                        oneapi::dpl::make_zip_iterator(B_ext_diag_ii, B_ext_diag_bigj, B_ext_diag_data),   /* result  */
                        pred1 );
      nalu_hypre_assert( std::get<0>(new_end.base()) == B_ext_diag_ii + B_ext_diag_nnz );
#else
      auto new_end = NALU_HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,
                                                                     B_ext_data)),             /* first */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,
                                                                     B_ext_data)) + B_ext_nnz, /* last */
                        B_ext_bigj,                                                            /* stencil */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_diag_ii, B_ext_diag_bigj,
                                                                     B_ext_diag_data)),        /* result */
                        pred1 );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == B_ext_diag_ii + B_ext_diag_nnz );
#endif
   }

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      B_ext_diag_bigj,
                      B_ext_diag_bigj + B_ext_diag_nnz,
                      B_ext_diag_j,
   [const_val = first_col_diag_B](const auto & x) {return x - const_val;} );
#else
   NALU_HYPRE_THRUST_CALL( transform,
                      B_ext_diag_bigj,
                      B_ext_diag_bigj + B_ext_diag_nnz,
                      thrust::make_constant_iterator(first_col_diag_B),
                      B_ext_diag_j,
                      thrust::minus<NALU_HYPRE_BigInt>());
#endif
   nalu_hypre_TFree(B_ext_diag_bigj, NALU_HYPRE_MEMORY_DEVICE);

   /* copy to offd */
   B_ext_offd_bigj = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, B_ext_offd_nnz, NALU_HYPRE_MEMORY_DEVICE);

   if (B_ext_offd_xata)
   {
#if defined(NALU_HYPRE_USING_SYCL)
      auto first = oneapi::dpl::make_zip_iterator(B_ext_ii, B_ext_bigj, B_ext_data, B_ext_xata);
      auto new_end = hypreSycl_copy_if(
                        first,                                           /* first */
                        first + B_ext_nnz,                               /* last */
                        B_ext_bigj,                                      /* stencil */
                        oneapi::dpl::make_zip_iterator(B_ext_offd_ii, B_ext_offd_bigj, B_ext_offd_data,
                                                       B_ext_offd_xata), /* result */
                        std::not_fn(pred1) );
      nalu_hypre_assert( std::get<0>(new_end.base()) == B_ext_offd_ii + B_ext_offd_nnz );
#else
      auto new_end = NALU_HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data,
                                                                     B_ext_xata)),             /* first */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data,
                                                                     B_ext_xata)) + B_ext_nnz, /* last */
                        B_ext_bigj,                                                            /* stencil */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_offd_ii, B_ext_offd_bigj, B_ext_offd_data,
                                                                     B_ext_offd_xata)),        /* result */
                        thrust::not1(pred1) );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == B_ext_offd_ii + B_ext_offd_nnz );
#endif
   }
   else
   {
#if defined(NALU_HYPRE_USING_SYCL)
      auto first = oneapi::dpl::make_zip_iterator(B_ext_ii, B_ext_bigj, B_ext_data);
      auto new_end = hypreSycl_copy_if(
                        first,                                                              /* first   */
                        first + B_ext_nnz,                                                  /* last    */
                        B_ext_bigj,                                                         /* stencil */
                        oneapi::dpl::make_zip_iterator(B_ext_offd_ii, B_ext_offd_bigj, B_ext_offd_data), /* result  */
                        std::not_fn(pred1) );
      nalu_hypre_assert( std::get<0>(new_end.base()) == B_ext_offd_ii + B_ext_offd_nnz );
#else
      auto new_end = NALU_HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,
                                                                     B_ext_data)),             /* first */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,
                                                                     B_ext_data)) + B_ext_nnz, /* last */
                        B_ext_bigj,                                                            /* stencil */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_offd_ii, B_ext_offd_bigj,
                                                                     B_ext_offd_data)),        /* result */
                        thrust::not1(pred1) );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == B_ext_offd_ii + B_ext_offd_nnz );
#endif
   }

   nalu_hypre_CSRMatrixMergeColMapOffd(num_cols_offd_B, col_map_offd_B, B_ext_offd_nnz, B_ext_offd_bigj,
                                  &num_cols_offd_C, &col_map_offd_C, &map_B_to_C);

#if defined(NALU_HYPRE_USING_SYCL)
   if (num_cols_offd_C > 0 && B_ext_offd_nnz > 0)
   {
      NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                         col_map_offd_C,
                         col_map_offd_C + num_cols_offd_C,
                         B_ext_offd_bigj,
                         B_ext_offd_bigj + B_ext_offd_nnz,
                         B_ext_offd_j );
   }
#else
   NALU_HYPRE_THRUST_CALL( lower_bound,
                      col_map_offd_C,
                      col_map_offd_C + num_cols_offd_C,
                      B_ext_offd_bigj,
                      B_ext_offd_bigj + B_ext_offd_nnz,
                      B_ext_offd_j );
#endif

   nalu_hypre_TFree(B_ext_offd_bigj, NALU_HYPRE_MEMORY_DEVICE);

   *map_B_to_C_ptr = map_B_to_C;
   *num_cols_offd_C_ptr = num_cols_offd_C;
   *col_map_offd_C_ptr  = col_map_offd_C;

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixCompressColumnsDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixCompressColumnsDevice(nalu_hypre_CSRMatrix  *A,
                                     NALU_HYPRE_BigInt     *col_map,
                                     NALU_HYPRE_Int       **col_idx_new_ptr,
                                     NALU_HYPRE_BigInt    **col_map_new_ptr)
{
   NALU_HYPRE_Int  num_cols = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int  nnz      = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int *A_j      = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int *tmp_j    = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnz, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int *tmp_end;
   NALU_HYPRE_Int  num_cols_new;

   nalu_hypre_TMemcpy(tmp_j, A_j, NALU_HYPRE_Int, nnz, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL(std::sort, tmp_j, tmp_j + nnz);
   tmp_end = NALU_HYPRE_ONEDPL_CALL(std::unique, tmp_j, tmp_j + nnz);
#else
   NALU_HYPRE_THRUST_CALL(sort, tmp_j, tmp_j + nnz);
   tmp_end = NALU_HYPRE_THRUST_CALL(unique, tmp_j, tmp_j + nnz);
#endif
   num_cols_new = tmp_end - tmp_j;

   nalu_hypre_assert(num_cols_new <= num_cols);

   if (num_cols_new < num_cols)
   {
      NALU_HYPRE_Int    *offd_mark = NULL;
      NALU_HYPRE_BigInt *col_map_new;

      if (num_cols_new)
      {
         offd_mark = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols, NALU_HYPRE_MEMORY_DEVICE);
      }

      if (col_map_new_ptr)
      {
         col_map_new = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_new, NALU_HYPRE_MEMORY_DEVICE);
      }

#if defined(NALU_HYPRE_USING_SYCL)
      oneapi::dpl::counting_iterator count(0);
      hypreSycl_scatter( count,
                         count + num_cols_new,
                         tmp_j,
                         offd_mark );

      hypreSycl_gather(A_j, A_j + nnz, offd_mark, A_j);

      if (col_map_new_ptr)
      {
         hypreSycl_gather(tmp_j, tmp_j + num_cols_new, col_map, col_map_new);
      }
#else
      NALU_HYPRE_THRUST_CALL( scatter,
                         thrust::counting_iterator<NALU_HYPRE_Int>(0),
                         thrust::counting_iterator<NALU_HYPRE_Int>(num_cols_new),
                         tmp_j,
                         offd_mark );

      NALU_HYPRE_THRUST_CALL(gather, A_j, A_j + nnz, offd_mark, A_j);

      if (col_map_new_ptr)
      {
         NALU_HYPRE_THRUST_CALL(gather, tmp_j, tmp_j + num_cols_new, col_map, col_map_new);
      }
#endif

      nalu_hypre_TFree(offd_mark, NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_CSRMatrixNumCols(A) = num_cols_new;

      if (col_idx_new_ptr)
      {
         *col_idx_new_ptr = tmp_j;
      }
      else
      {
         nalu_hypre_TFree(tmp_j, NALU_HYPRE_MEMORY_DEVICE);
      }

      if (col_map_new_ptr)
      {
         *col_map_new_ptr = col_map_new;
      }
   }
   else
   {
      if (col_idx_new_ptr)
      {
         *col_idx_new_ptr = NULL;
      }

      if (col_map_new_ptr)
      {
         *col_map_new_ptr = NULL;
      }

      nalu_hypre_TFree(tmp_j, NALU_HYPRE_MEMORY_DEVICE);
   }

   return nalu_hypre_error_flag;
}

#endif /* defined(NALU_HYPRE_USING_GPU) */

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixTriLowerUpperSolveDevice_core
 *
 * TODO (VPM): The analysis portion (setup phase) of the triangular solve
 *             is embedded into the vendor libraries wrappers.
 *             Should we create a separate function "Setup" function?
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixTriLowerUpperSolveDevice_core(char             uplo,
                                             NALU_HYPRE_Int        unit_diag,
                                             nalu_hypre_CSRMatrix *A,
                                             NALU_HYPRE_Real      *l1_norms,
                                             nalu_hypre_Vector    *f,
                                             NALU_HYPRE_Int        offset_f,
                                             nalu_hypre_Vector    *u,
                                             NALU_HYPRE_Int        offset_u)
{
   /* Trivial case: no rows */
   if (nalu_hypre_CSRMatrixNumRows(A) <= 0)
   {
      return nalu_hypre_error_flag;
   }

   /* Trivial case: empty rows */
   if (nalu_hypre_CSRMatrixNumNonzeros(A) <= 0)
   {
      return nalu_hypre_error_flag;
   }

   /* Sanity check */
   if (nalu_hypre_CSRMatrixNumRows(A) != nalu_hypre_CSRMatrixNumCols(A))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Triangular matrix solver works only for square matrices!");
      return nalu_hypre_error_flag;
   }

   /* Call vendor specific implementations */
#if defined(NALU_HYPRE_USING_CUSPARSE)
   nalu_hypre_CSRMatrixTriLowerUpperSolveCusparse(uplo, unit_diag, A,
                                             l1_norms,
                                             nalu_hypre_VectorData(f) + offset_f,
                                             nalu_hypre_VectorData(u) + offset_u);
#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   nalu_hypre_CSRMatrixTriLowerUpperSolveRocsparse(uplo, unit_diag, A,
                                              l1_norms,
                                              nalu_hypre_VectorData(f) + offset_f,
                                              nalu_hypre_VectorData(u) + offset_u);
#else
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                     "nalu_hypre_CSRMatrixTriLowerUpperSolveDevice requires configuration with either cuSPARSE or rocSPARSE\n");
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixTriLowerUpperSolveDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixTriLowerUpperSolveDevice(char             uplo,
                                        NALU_HYPRE_Int        unit_diag,
                                        nalu_hypre_CSRMatrix *A,
                                        NALU_HYPRE_Real      *l1_norms,
                                        nalu_hypre_Vector    *f,
                                        nalu_hypre_Vector    *u )
{
   return nalu_hypre_CSRMatrixTriLowerUpperSolveDevice_core(uplo, unit_diag, A, l1_norms, f, 0, u, 0);
}

#endif /* defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) */

#if defined(NALU_HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixAddPartial
 *
 * Adds matrix rows in the CSR matrix B to the CSR Matrix A, where row_nums[i]
 * defines to which row of A the i-th row of B is added, and returns a CSR
 * Matrix C. Repeated row indices are allowed in row_nums
 *
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 *       in A and B. To remove those, use nalu_hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix*
nalu_hypre_CSRMatrixAddPartialDevice( nalu_hypre_CSRMatrix *A,
                                 nalu_hypre_CSRMatrix *B,
                                 NALU_HYPRE_Int       *row_nums)
{
   NALU_HYPRE_Complex    *A_data   = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int        *A_i      = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int        *A_j      = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int         nrows_A  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int         ncols_A  = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int         nnz_A    = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Complex    *B_data   = nalu_hypre_CSRMatrixData(B);
   NALU_HYPRE_Int        *B_i      = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int        *B_j      = nalu_hypre_CSRMatrixJ(B);
   NALU_HYPRE_Int         nrows_B  = nalu_hypre_CSRMatrixNumRows(B);
   NALU_HYPRE_Int         ncols_B  = nalu_hypre_CSRMatrixNumCols(B);
   NALU_HYPRE_Int         nnz_B    = nalu_hypre_CSRMatrixNumNonzeros(B);
   NALU_HYPRE_Complex    *C_data;
   NALU_HYPRE_Int        *C_i;
   NALU_HYPRE_Int        *C_j;
   NALU_HYPRE_Int         nnzC;
   nalu_hypre_CSRMatrix  *C;

   if (ncols_A != ncols_B)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Warning! incompatible matrix dimensions!\n");

      return NULL;
   }

   hypreDevice_CSRSpAdd(nrows_A, nrows_B, ncols_A, nnz_A, nnz_B,
                        A_i, A_j, 1.0, A_data, NULL, B_i, B_j,
                        1.0, B_data, NULL, row_nums,
                        &nnzC, &C_i, &C_j, &C_data);

   C = nalu_hypre_CSRMatrixCreate(nrows_A, ncols_B, nnzC);
   nalu_hypre_CSRMatrixI(C) = C_i;
   nalu_hypre_CSRMatrixJ(C) = C_j;
   nalu_hypre_CSRMatrixData(C) = C_data;
   nalu_hypre_CSRMatrixMemoryLocation(C) = NALU_HYPRE_MEMORY_DEVICE;

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return C;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixColNNzRealDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixColNNzRealDevice( nalu_hypre_CSRMatrix  *A,
                                 NALU_HYPRE_Real       *colnnz)
{
   NALU_HYPRE_Int *A_j      = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int  ncols_A  = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int  nnz_A    = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int *A_j_sorted;
   NALU_HYPRE_Int  num_reduced_col_indices;
   NALU_HYPRE_Int *reduced_col_indices;
   NALU_HYPRE_Int *reduced_col_nnz;

   A_j_sorted = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnz_A, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(A_j_sorted, A_j, NALU_HYPRE_Int, nnz_A, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL(std::sort, A_j_sorted, A_j_sorted + nnz_A);
#else
   NALU_HYPRE_THRUST_CALL(sort, A_j_sorted, A_j_sorted + nnz_A);
#endif

   reduced_col_indices = nalu_hypre_TAlloc(NALU_HYPRE_Int, ncols_A, NALU_HYPRE_MEMORY_DEVICE);
   reduced_col_nnz     = nalu_hypre_TAlloc(NALU_HYPRE_Int, ncols_A, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)

   /* WM: todo - better way to get around lack of constant iterator in DPL? */
   NALU_HYPRE_Int *ones = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnz_A, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_ONEDPL_CALL( std::fill_n, ones, nnz_A, 1 );
   auto new_end = NALU_HYPRE_ONEDPL_CALL( oneapi::dpl::reduce_by_segment,
                                     A_j_sorted,
                                     A_j_sorted + nnz_A,
                                     ones,
                                     reduced_col_indices,
                                     reduced_col_nnz);

   nalu_hypre_TFree(ones, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_assert(new_end.first - reduced_col_indices == new_end.second - reduced_col_nnz);
   num_reduced_col_indices = new_end.first - reduced_col_indices;
#else
   thrust::pair<NALU_HYPRE_Int*, NALU_HYPRE_Int*> new_end =
      NALU_HYPRE_THRUST_CALL(reduce_by_key, A_j_sorted, A_j_sorted + nnz_A,
                        thrust::make_constant_iterator(1),
                        reduced_col_indices,
                        reduced_col_nnz);
   nalu_hypre_assert(new_end.first - reduced_col_indices == new_end.second - reduced_col_nnz);
   num_reduced_col_indices = new_end.first - reduced_col_indices;
#endif


   nalu_hypre_Memset(colnnz, 0, ncols_A * sizeof(NALU_HYPRE_Real), NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::copy, reduced_col_nnz, reduced_col_nnz + num_reduced_col_indices,
                      oneapi::dpl::make_permutation_iterator(colnnz, reduced_col_indices) );
#else
   NALU_HYPRE_THRUST_CALL(scatter, reduced_col_nnz, reduced_col_nnz + num_reduced_col_indices,
                     reduced_col_indices, colnnz);
#endif

   nalu_hypre_TFree(A_j_sorted,          NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(reduced_col_indices, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(reduced_col_nnz,     NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMoveDiagFirst
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRMoveDiagFirst( nalu_hypre_DeviceItem    &item,
                                 NALU_HYPRE_Int      nrows,
                                 NALU_HYPRE_Int     *ia,
                                 NALU_HYPRE_Int     *ja,
                                 NALU_HYPRE_Complex *aa )
{
   NALU_HYPRE_Int row  = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);
   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);

   if (row >= nrows)
   {
      return;
   }

   NALU_HYPRE_Int p = 0, q = 0;

   if (lane < 2)
   {
      p = read_only_load(ia + row + lane);
   }

   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane + 1; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q);
        j += NALU_HYPRE_WARP_SIZE)
   {
      nalu_hypre_int find_diag = j < q && ja[j] == row;

      if (find_diag)
      {
         ja[j] = ja[p];
         ja[p] = row;
         NALU_HYPRE_Complex tmp = aa[p];
         aa[p] = aa[j];
         aa[j] = tmp;
      }

      if ( warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, find_diag) )
      {
         break;
      }
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixMoveDiagFirstDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixMoveDiagFirstDevice( nalu_hypre_CSRMatrix  *A )
{
   NALU_HYPRE_Int      nrows  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Complex *A_data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int     *A_i    = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *A_j    = nalu_hypre_CSRMatrixJ(A);

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

   NALU_HYPRE_GPU_LAUNCH(hypreGPUKernel_CSRMoveDiagFirst, gDim, bDim,
                    nrows, A_i, A_j, A_data);

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixStack2Device
 *
 * return C = [A; B]
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix*
nalu_hypre_CSRMatrixStack2Device(nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B)
{
   nalu_hypre_GpuProfilingPushRange("CSRMatrixStack2");

   nalu_hypre_assert( nalu_hypre_CSRMatrixNumCols(A) == nalu_hypre_CSRMatrixNumCols(B) );

   nalu_hypre_CSRMatrix *C = nalu_hypre_CSRMatrixCreate( nalu_hypre_CSRMatrixNumRows(A) + nalu_hypre_CSRMatrixNumRows(B),
                                               nalu_hypre_CSRMatrixNumCols(A),
                                               nalu_hypre_CSRMatrixNumNonzeros(A) + nalu_hypre_CSRMatrixNumNonzeros(B) );

   NALU_HYPRE_Int     *C_i = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nalu_hypre_CSRMatrixNumRows(C) + 1,
                                     NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int     *C_j = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nalu_hypre_CSRMatrixNumNonzeros(C),
                                     NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Complex *C_a = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nalu_hypre_CSRMatrixNumNonzeros(C),
                                     NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_TMemcpy(C_i, nalu_hypre_CSRMatrixI(A), NALU_HYPRE_Int, nalu_hypre_CSRMatrixNumRows(A) + 1,
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(C_i + nalu_hypre_CSRMatrixNumRows(A) + 1, nalu_hypre_CSRMatrixI(B) + 1, NALU_HYPRE_Int,
                 nalu_hypre_CSRMatrixNumRows(B),
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
#if defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ONEDPL_CALL( std::transform,
                      C_i + nalu_hypre_CSRMatrixNumRows(A) + 1,
                      C_i + nalu_hypre_CSRMatrixNumRows(C) + 1,
                      C_i + nalu_hypre_CSRMatrixNumRows(A) + 1,
   [const_val = nalu_hypre_CSRMatrixNumNonzeros(A)] (const auto & x) {return x + const_val;} );
#else
   NALU_HYPRE_THRUST_CALL( transform,
                      C_i + nalu_hypre_CSRMatrixNumRows(A) + 1,
                      C_i + nalu_hypre_CSRMatrixNumRows(C) + 1,
                      thrust::make_constant_iterator(nalu_hypre_CSRMatrixNumNonzeros(A)),
                      C_i + nalu_hypre_CSRMatrixNumRows(A) + 1,
                      thrust::plus<NALU_HYPRE_Int>() );
#endif

   nalu_hypre_TMemcpy(C_j, nalu_hypre_CSRMatrixJ(A), NALU_HYPRE_Int, nalu_hypre_CSRMatrixNumNonzeros(A),
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(C_j + nalu_hypre_CSRMatrixNumNonzeros(A), nalu_hypre_CSRMatrixJ(B), NALU_HYPRE_Int,
                 nalu_hypre_CSRMatrixNumNonzeros(B),
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_TMemcpy(C_a, nalu_hypre_CSRMatrixData(A), NALU_HYPRE_Complex, nalu_hypre_CSRMatrixNumNonzeros(A),
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(C_a + nalu_hypre_CSRMatrixNumNonzeros(A), nalu_hypre_CSRMatrixData(B), NALU_HYPRE_Complex,
                 nalu_hypre_CSRMatrixNumNonzeros(B),
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_CSRMatrixI(C) = C_i;
   nalu_hypre_CSRMatrixJ(C) = C_j;
   nalu_hypre_CSRMatrixData(C) = C_a;
   nalu_hypre_CSRMatrixMemoryLocation(C) = NALU_HYPRE_MEMORY_DEVICE;

   nalu_hypre_GpuProfilingPopRange();

   return C;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRRowSum
 *
 * type == 0, sum,
 *         1, abs sum (l-1)
 *         2, square sum (l-2)
 *--------------------------------------------------------------------------*/

template<NALU_HYPRE_Int type>
__global__ void
hypreGPUKernel_CSRRowSum( nalu_hypre_DeviceItem    &item,
                          NALU_HYPRE_Int      nrows,
                          NALU_HYPRE_Int     *ia,
                          NALU_HYPRE_Int     *ja,
                          NALU_HYPRE_Complex *aa,
                          NALU_HYPRE_Int     *CF_i,
                          NALU_HYPRE_Int     *CF_j,
                          NALU_HYPRE_Complex *row_sum,
                          NALU_HYPRE_Complex  scal,
                          NALU_HYPRE_Int      set)
{
   NALU_HYPRE_Int row_i = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= nrows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p = 0, q = 0;

   if (lane < 2)
   {
      p = read_only_load(ia + row_i + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   NALU_HYPRE_Complex row_sum_i = 0.0;

   for (NALU_HYPRE_Int j = p + lane; j < q; j += NALU_HYPRE_WARP_SIZE)
   {
      if ( CF_i && CF_j && read_only_load(&CF_i[row_i]) != read_only_load(&CF_j[ja[j]]) )
      {
         continue;
      }

      NALU_HYPRE_Complex aii = aa[j];

      if (type == 0)
      {
         row_sum_i += aii;
      }
      else if (type == 1)
      {
         row_sum_i += nalu_hypre_abs(aii);
      }
      else if (type == 2)
      {
         row_sum_i += aii * aii;
      }
   }

   row_sum_i = warp_reduce_sum(item, row_sum_i);

   if (lane == 0)
   {
      if (set)
      {
         row_sum[row_i] = scal * row_sum_i;
      }
      else
      {
         row_sum[row_i] += scal * row_sum_i;
      }
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixComputeRowSumDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixComputeRowSumDevice( nalu_hypre_CSRMatrix *A,
                                    NALU_HYPRE_Int       *CF_i,
                                    NALU_HYPRE_Int       *CF_j,
                                    NALU_HYPRE_Complex   *row_sum,
                                    NALU_HYPRE_Int        type,
                                    NALU_HYPRE_Complex    scal,
                                    const char      *set_or_add)
{
   NALU_HYPRE_Int      nrows  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Complex *A_data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int     *A_i    = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *A_j    = nalu_hypre_CSRMatrixJ(A);

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

   NALU_HYPRE_Int set = set_or_add[0] == 's';
   if (type == 0)
   {
      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRRowSum<0>, gDim, bDim,
                        nrows, A_i, A_j, A_data,
                        CF_i, CF_j, row_sum, scal, set );
   }
   else if (type == 1)
   {
      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRRowSum<1>, gDim, bDim,
                        nrows, A_i, A_j, A_data,
                        CF_i, CF_j, row_sum, scal, set );
   }
   else if (type == 2)
   {
      NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRRowSum<2>, gDim, bDim,
                        nrows, A_i, A_j, A_data, CF_i, CF_j,
                        row_sum, scal, set );
   }

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMatrixIntersectPattern
 *
 * mark is of size nA
 * diag_option: 1: special treatment for diag entries, mark as -2
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRMatrixIntersectPattern(nalu_hypre_DeviceItem &item,
                                         NALU_HYPRE_Int  n,
                                         NALU_HYPRE_Int  nA,
                                         NALU_HYPRE_Int *rowid,
                                         NALU_HYPRE_Int *colid,
                                         NALU_HYPRE_Int *idx,
                                         NALU_HYPRE_Int *mark,
                                         NALU_HYPRE_Int  diag_option)
{
   NALU_HYPRE_Int i = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i >= n)
   {
      return;
   }

   NALU_HYPRE_Int r1 = read_only_load(&rowid[i]);
   NALU_HYPRE_Int c1 = read_only_load(&colid[i]);
   NALU_HYPRE_Int j = read_only_load(&idx[i]);

   if (0 == diag_option)
   {
      if (j < nA)
      {
         NALU_HYPRE_Int r2 = i < n - 1 ? read_only_load(&rowid[i + 1]) : -1;
         NALU_HYPRE_Int c2 = i < n - 1 ? read_only_load(&colid[i + 1]) : -1;
         if (r1 == r2 && c1 == c2)
         {
            mark[j] = c1;
         }
         else
         {
            mark[j] = -1;
         }
      }
   }
   else if (1 == diag_option)
   {
      if (j < nA)
      {
         if (r1 == c1)
         {
            mark[j] = -2;
         }
         else
         {
            NALU_HYPRE_Int r2 = i < n - 1 ? read_only_load(&rowid[i + 1]) : -1;
            NALU_HYPRE_Int c2 = i < n - 1 ? read_only_load(&colid[i + 1]) : -1;
            if (r1 == r2 && c1 == c2)
            {
               mark[j] = c1;
            }
            else
            {
               mark[j] = -1;
            }
         }
      }
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixIntersectPattern
 *
 * markA: array of size nnz(A), for pattern of (A and B), markA is the
 * column indices as in A_J. Otherwise, mark pattern not in A-B as -1 in markA
 *
 * Note the special treatment for diagonal entries of A (marked as -2)
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixIntersectPattern(nalu_hypre_CSRMatrix *A,
                                nalu_hypre_CSRMatrix *B,
                                NALU_HYPRE_Int       *markA,
                                NALU_HYPRE_Int        diag_opt)
{
   NALU_HYPRE_Int nrows = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int nnzA  = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int nnzB  = nalu_hypre_CSRMatrixNumNonzeros(B);

   NALU_HYPRE_Int *Cii = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnzA + nnzB, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int *Cjj = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnzA + nnzB, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_Int *idx = nalu_hypre_TAlloc(NALU_HYPRE_Int, nnzA + nnzB, NALU_HYPRE_MEMORY_DEVICE);

   hypreDevice_CsrRowPtrsToIndices_v2(nrows, nnzA, nalu_hypre_CSRMatrixI(A), Cii);
   hypreDevice_CsrRowPtrsToIndices_v2(nrows, nnzB, nalu_hypre_CSRMatrixI(B), Cii + nnzA);
   nalu_hypre_TMemcpy(Cjj,        nalu_hypre_CSRMatrixJ(A), NALU_HYPRE_Int, nnzA, NALU_HYPRE_MEMORY_DEVICE,
                 NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(Cjj + nnzA, nalu_hypre_CSRMatrixJ(B), NALU_HYPRE_Int, nnzB, NALU_HYPRE_MEMORY_DEVICE,
                 NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   hypreSycl_sequence(idx, idx + nnzA + nnzB, 0);

   auto zipped_begin = oneapi::dpl::make_zip_iterator(Cii, Cjj, idx);
   NALU_HYPRE_ONEDPL_CALL( std::stable_sort, zipped_begin, zipped_begin + nnzA + nnzB,
                      [](auto lhs, auto rhs)
   {
      if (std::get<0>(lhs) == std::get<0>(rhs))
      {
         return std::get<1>(lhs) < std::get<1>(rhs);
      }
      else
      {
         return std::get<0>(lhs) < std::get<0>(rhs);
      }
   } );
#else
   NALU_HYPRE_THRUST_CALL( sequence, idx, idx + nnzA + nnzB );

   NALU_HYPRE_THRUST_CALL( stable_sort_by_key,
                      thrust::make_zip_iterator(thrust::make_tuple(Cii, Cjj)),
                      thrust::make_zip_iterator(thrust::make_tuple(Cii, Cjj)) + nnzA + nnzB,
                      idx );
#endif

   nalu_hypre_TMemcpy(markA, nalu_hypre_CSRMatrixJ(A), NALU_HYPRE_Int, nnzA, NALU_HYPRE_MEMORY_DEVICE,
                 NALU_HYPRE_MEMORY_DEVICE);

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nnzA + nnzB, "thread", bDim);

   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRMatrixIntersectPattern, gDim, bDim,
                     nnzA + nnzB, nnzA, Cii, Cjj, idx, markA, diag_opt );

   nalu_hypre_TFree(Cii, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(Cjj, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(idx, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRExtractDiag
 *
 * type 0: diag
 *      1: abs diag
 *      2: diag inverse
 *      3: diag inverse sqrt
 *      4: abs diag inverse sqrt
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRExtractDiag( nalu_hypre_DeviceItem    &item,
                               NALU_HYPRE_Int      nrows,
                               NALU_HYPRE_Int     *ia,
                               NALU_HYPRE_Int     *ja,
                               NALU_HYPRE_Complex *aa,
                               NALU_HYPRE_Complex *d,
                               NALU_HYPRE_Int      type)
{
   NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p = 0, q = 0;

   if (lane < 2)
   {
      p = read_only_load(ia + row + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   NALU_HYPRE_Int has_diag = 0;

   for (NALU_HYPRE_Int j = p + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q); j += NALU_HYPRE_WARP_SIZE)
   {
      nalu_hypre_int find_diag = j < q && ja[j] == row;

      if (find_diag)
      {
         if (type == 0)
         {
            d[row] = aa[j];
         }
         else if (type == 1)
         {
            d[row] = nalu_hypre_abs(aa[j]);
         }
         else if (type == 2)
         {
            d[row] = 1.0 / aa[j];
         }
         else if (type == 3)
         {
            d[row] = 1.0 / nalu_hypre_sqrt(aa[j]);
         }
         else if (type == 4)
         {
            d[row] = 1.0 / nalu_hypre_sqrt(nalu_hypre_abs(aa[j]));
         }
      }

      if ( warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, find_diag) )
      {
         has_diag = 1;
         break;
      }
   }

   if (!has_diag && lane == 0)
   {
      d[row] = 0.0;
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixExtractDiagonalDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixExtractDiagonalDevice( nalu_hypre_CSRMatrix *A,
                                      NALU_HYPRE_Complex   *d,
                                      NALU_HYPRE_Int        type)
{
   NALU_HYPRE_Int      nrows  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Complex *A_data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int     *A_i    = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *A_j    = nalu_hypre_CSRMatrixJ(A);

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRExtractDiag, gDim, bDim, nrows,
                     A_i, A_j, A_data, d, type );

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRCheckDiagFirst
 *
 * check if diagonal entry is the first one at each row
 * Return: the number of rows that do not have the first entry as diagonal
 * RL: only check if it's a non-empty row
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRCheckDiagFirst( nalu_hypre_DeviceItem &item,
                                  NALU_HYPRE_Int  nrows,
                                  NALU_HYPRE_Int *ia,
                                  NALU_HYPRE_Int *ja,
                                  NALU_HYPRE_Int *result )
{
   const NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_thread_id<1, 1>(item);
   if (row < nrows)
   {
      result[row] = (ia[row + 1] > ia[row]) && (ja[ia[row]] != row);
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixCheckDiagFirstDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixCheckDiagFirstDevice( nalu_hypre_CSRMatrix *A )
{
   NALU_HYPRE_Int  *A_i      = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int  *A_j      = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int   num_rows = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int  *result;
   NALU_HYPRE_Int   ierr;

   /* Sanity check */
   if (nalu_hypre_CSRMatrixNumRows(A) != nalu_hypre_CSRMatrixNumCols(A))
   {
      return 0;
   }

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nalu_hypre_CSRMatrixNumRows(A), "thread", bDim);

   result = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRCheckDiagFirst, gDim, bDim,
                     num_rows, A_i, A_j, result );

   /* Compute number of rows in which the diagonal is not the first entry */
#if defined(NALU_HYPRE_USING_SYCL)
   ierr = NALU_HYPRE_ONEDPL_CALL( std::reduce,
                             result,
                             result + nalu_hypre_CSRMatrixNumRows(A) );
#else
   ierr = NALU_HYPRE_THRUST_CALL( reduce,
                             result,
                             result + nalu_hypre_CSRMatrixNumRows(A) );
#endif

   nalu_hypre_TFree(result, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return ierr;
}

#endif /* defined(NALU_HYPRE_USING_GPU) */

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMatrixFixZeroDiagDevice
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRMatrixReplaceDiagDevice( nalu_hypre_DeviceItem    &item,
                                           NALU_HYPRE_Complex *new_diag,
                                           NALU_HYPRE_Complex  v,
                                           NALU_HYPRE_Int      nrows,
                                           NALU_HYPRE_Int     *ia,
                                           NALU_HYPRE_Int     *ja,
                                           NALU_HYPRE_Complex *data,
                                           NALU_HYPRE_Real     tol,
                                           NALU_HYPRE_Int     *result )
{
   const NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p = 0, q = 0;
   bool has_diag = false;

   if (lane < 2)
   {
      p = read_only_load(ia + row + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   for (NALU_HYPRE_Int j = p + lane; warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, j < q); j += NALU_HYPRE_WARP_SIZE)
   {
      nalu_hypre_int find_diag = j < q && read_only_load(&ja[j]) == row;

      if (find_diag)
      {
         if (new_diag)
         {
            NALU_HYPRE_Complex d = read_only_load(&new_diag[row]);
            data[j] = nalu_hypre_abs(d) <= tol ? v : d;
         }
         else
         {
            if (nalu_hypre_abs(data[j]) <= tol)
            {
               data[j] = v;
            }
         }
      }

      if ( warp_any_sync(item, NALU_HYPRE_WARP_FULL_MASK, find_diag) )
      {
         has_diag = true;
         break;
      }
   }

   if (result && !has_diag && lane == 0)
   {
      result[row] = 1;
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixReplaceDiagDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixReplaceDiagDevice( nalu_hypre_CSRMatrix *A,
                                  NALU_HYPRE_Complex   *new_diag,
                                  NALU_HYPRE_Complex    v,
                                  NALU_HYPRE_Real       tol )
{
   if (nalu_hypre_CSRMatrixNumRows(A) != nalu_hypre_CSRMatrixNumCols(A))
   {
      return nalu_hypre_error_flag;
   }

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nalu_hypre_CSRMatrixNumRows(A), "warp", bDim);

#if NALU_HYPRE_DEBUG
   NALU_HYPRE_Int *result = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nalu_hypre_CSRMatrixNumRows(A), NALU_HYPRE_MEMORY_DEVICE);
#else
   NALU_HYPRE_Int *result = NULL;
#endif

   NALU_HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRMatrixReplaceDiagDevice, gDim, bDim,
                     new_diag, v, nalu_hypre_CSRMatrixNumRows(A),
                     nalu_hypre_CSRMatrixI(A), nalu_hypre_CSRMatrixJ(A), nalu_hypre_CSRMatrixData(A),
                     tol, result );

#if NALU_HYPRE_DEBUG
   /* the number of structural zero in A */
   NALU_HYPRE_Int num_zeros = NALU_HYPRE_THRUST_CALL( reduce,
                                            result,
                                            result + nalu_hypre_CSRMatrixNumRows(A) );

   nalu_hypre_TFree(result, NALU_HYPRE_MEMORY_DEVICE);

   if (num_zeros)
   {
      nalu_hypre_error_w_msg(num_zeros, "structural zero in nalu_hypre_CSRMatrixReplaceDiagDevice");
   }
#endif

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return nalu_hypre_error_flag;
}

#endif /* defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) */

#if defined(NALU_HYPRE_USING_GPU)

#if defined(NALU_HYPRE_USING_SYCL)
typedef std::tuple<NALU_HYPRE_Int, NALU_HYPRE_Int> Int2;
struct Int2Unequal
{
   bool operator()(const Int2& t) const
   {
      return (std::get<0>(t) != std::get<1>(t));
   }
};
#else
typedef thrust::tuple<NALU_HYPRE_Int, NALU_HYPRE_Int> Int2;
struct Int2Unequal : public thrust::unary_function<Int2, bool>
{
   __host__ __device__
   bool operator()(const Int2& t) const
   {
      return (thrust::get<0>(t) != thrust::get<1>(t));
   }
};
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixRemoveDiagonalDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixRemoveDiagonalDevice(nalu_hypre_CSRMatrix *A)
{
   NALU_HYPRE_Int      nrows  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int      nnz    = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int     *A_i    = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *A_j    = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Complex *A_data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int     *A_ii   = hypreDevice_CsrRowPtrsToIndices(nrows, nnz, A_i);
   NALU_HYPRE_Int      new_nnz;
   NALU_HYPRE_Int     *new_ii;
   NALU_HYPRE_Int     *new_j;
   NALU_HYPRE_Complex *new_data;

#if defined(NALU_HYPRE_USING_SYCL)
   auto zip_ij = oneapi::dpl::make_zip_iterator(A_ii, A_j);
   new_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                zip_ij,
                                zip_ij + nnz,
                                Int2Unequal() );
#else
   new_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)),
                                thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)) + nnz,
                                Int2Unequal() );
#endif

   if (new_nnz == nnz)
   {
      /* no diagonal entries found */
      nalu_hypre_TFree(A_ii, NALU_HYPRE_MEMORY_DEVICE);
      return nalu_hypre_error_flag;
   }

   new_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int, new_nnz, NALU_HYPRE_MEMORY_DEVICE);
   new_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, new_nnz, NALU_HYPRE_MEMORY_DEVICE);

   if (A_data)
   {
      new_data = nalu_hypre_TAlloc(NALU_HYPRE_Complex, new_nnz, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
      auto zip_ija = oneapi::dpl::make_zip_iterator(A_ii, A_j, A_data);
      auto zip_new_ija = oneapi::dpl::make_zip_iterator(new_ii, new_j, new_data);
      auto new_end = hypreSycl_copy_if(
                        zip_ija,
                        zip_ija + nnz,
                        zip_ij,
                        zip_new_ija,
                        Int2Unequal() );

      nalu_hypre_assert( std::get<0>(new_end.base()) == new_ii + new_nnz );
#else
      auto new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)) + nnz,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(new_ii, new_j, new_data)),
                                        Int2Unequal() );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == new_ii + new_nnz );
#endif
   }
   else
   {
      new_data = NULL;

#if defined(NALU_HYPRE_USING_SYCL)
      auto zip_new_ij = oneapi::dpl::make_zip_iterator(new_ii, new_j);
      auto new_end = hypreSycl_copy_if( zip_ij,
                                        zip_ij + nnz,
                                        zip_ij,
                                        zip_new_ij,
                                        Int2Unequal() );

      nalu_hypre_assert( std::get<0>(new_end.base()) == new_ii + new_nnz );
#else
      auto new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)) + nnz,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(new_ii, new_j)),
                                        Int2Unequal() );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == new_ii + new_nnz );
#endif
   }

   nalu_hypre_TFree(A_ii,   NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(A_i,    NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(A_j,    NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(A_data, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_CSRMatrixNumNonzeros(A) = new_nnz;
   nalu_hypre_CSRMatrixI(A) = hypreDevice_CsrRowIndicesToPtrs(nrows, new_nnz, new_ii);
   nalu_hypre_CSRMatrixJ(A) = new_j;
   nalu_hypre_CSRMatrixData(A) = new_data;
   nalu_hypre_TFree(new_ii, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRDiagScale
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRDiagScale( nalu_hypre_DeviceItem    &item,
                             NALU_HYPRE_Int      nrows,
                             NALU_HYPRE_Int     *ia,
                             NALU_HYPRE_Int     *ja,
                             NALU_HYPRE_Complex *aa,
                             NALU_HYPRE_Complex *ld,
                             NALU_HYPRE_Complex *rd)
{
   NALU_HYPRE_Int row = nalu_hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   NALU_HYPRE_Int lane = nalu_hypre_gpu_get_lane_id<1>(item);
   NALU_HYPRE_Int p = 0, q = 0;

   if (lane < 2)
   {
      p = read_only_load(ia + row + lane);
   }
   q = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, p, 0);

   NALU_HYPRE_Complex sl = 1.0;

   if (ld)
   {
      if (!lane)
      {
         sl = read_only_load(ld + row);
      }
      sl = warp_shuffle_sync(item, NALU_HYPRE_WARP_FULL_MASK, sl, 0);
   }

   if (rd)
   {
      for (NALU_HYPRE_Int i = p + lane; i < q; i += NALU_HYPRE_WARP_SIZE)
      {
         const NALU_HYPRE_Int col = read_only_load(ja + i);
         const NALU_HYPRE_Complex sr = read_only_load(rd + col);
         aa[i] = sl * aa[i] * sr;
      }
   }
   else if (sl != 1.0)
   {
      for (NALU_HYPRE_Int i = p + lane; i < q; i += NALU_HYPRE_WARP_SIZE)
      {
         aa[i] = sl * aa[i];
      }
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixDiagScaleDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixDiagScaleDevice( nalu_hypre_CSRMatrix *A,
                                nalu_hypre_Vector    *ld,
                                nalu_hypre_Vector    *rd)
{
   NALU_HYPRE_Int      nrows  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Complex *A_data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int     *A_i    = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *A_j    = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Complex *ldata  = ld ? nalu_hypre_VectorData(ld) : NULL;
   NALU_HYPRE_Complex *rdata  = rd ? nalu_hypre_VectorData(rd) : NULL;

   dim3 bDim = nalu_hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = nalu_hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

   NALU_HYPRE_GPU_LAUNCH(hypreGPUKernel_CSRDiagScale, gDim, bDim,
                    nrows, A_i, A_j, A_data, ldata, rdata);

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * cabsfirst_greaterthan_second_pred
 *
 * This predicate compares first and second element in a tuple in absolute
 * value first is assumed to be complex, second to be real > 0
 *--------------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_SYCL)
struct cabsfirst_greaterthan_second_pred
{
   bool operator()(const std::tuple<NALU_HYPRE_Complex, NALU_HYPRE_Real>& t) const
   {
      const NALU_HYPRE_Complex i = std::get<0>(t);
      const NALU_HYPRE_Real j = std::get<1>(t);

      return nalu_hypre_cabs(i) > j;
   }
};
#else
struct cabsfirst_greaterthan_second_pred : public
   thrust::unary_function<thrust::tuple<NALU_HYPRE_Complex, NALU_HYPRE_Real>, bool>
{
   __host__ __device__
   bool operator()(const thrust::tuple<NALU_HYPRE_Complex, NALU_HYPRE_Real>& t) const
   {
      const NALU_HYPRE_Complex i = thrust::get<0>(t);
      const NALU_HYPRE_Real j = thrust::get<1>(t);

      return nalu_hypre_cabs(i) > j;
   }
};
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixDropSmallEntriesDevice
 *
 * drop the entries that are smaller than:
 *    tol if elmt_tols == null,
 *    elmt_tols[j] otherwise where j = 0...NumNonzeros(A)
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixDropSmallEntriesDevice( nalu_hypre_CSRMatrix *A,
                                       NALU_HYPRE_Real       tol,
                                       NALU_HYPRE_Real      *elmt_tols)
{
   NALU_HYPRE_Int      nrows  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int      nnz    = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int     *A_i    = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *A_j    = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Complex *A_data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int     *A_ii   = NULL;
   NALU_HYPRE_Int      new_nnz = 0;
   NALU_HYPRE_Int     *new_ii;
   NALU_HYPRE_Int     *new_j;
   NALU_HYPRE_Complex *new_data;

#if defined(NALU_HYPRE_USING_SYCL)
   if (elmt_tols == NULL)
   {
      new_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                   A_data,
                                   A_data + nnz,
      [tol] (const auto & x) {return !(x < tol);} );
   }
   else
   {
      new_nnz = NALU_HYPRE_ONEDPL_CALL( std::count_if,
                                   oneapi::dpl::make_zip_iterator(A_data, elmt_tols),
                                   oneapi::dpl::make_zip_iterator(A_data, elmt_tols) + nnz,
                                   cabsfirst_greaterthan_second_pred() );
   }
#else
   if (elmt_tols == NULL)
   {
      new_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                   A_data,
                                   A_data + nnz,
                                   thrust::not1(less_than<NALU_HYPRE_Complex>(tol)) );
   }
   else
   {
      new_nnz = NALU_HYPRE_THRUST_CALL( count_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_data, elmt_tols)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_data, elmt_tols)) + nnz,
                                   cabsfirst_greaterthan_second_pred() );
   }
#endif

   if (new_nnz == nnz)
   {
      nalu_hypre_TFree(A_ii, NALU_HYPRE_MEMORY_DEVICE);
      return nalu_hypre_error_flag;
   }

   if (!A_ii)
   {
      A_ii = hypreDevice_CsrRowPtrsToIndices(nrows, nnz, A_i);
   }
   new_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int, new_nnz, NALU_HYPRE_MEMORY_DEVICE);
   new_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, new_nnz, NALU_HYPRE_MEMORY_DEVICE);
   new_data = nalu_hypre_TAlloc(NALU_HYPRE_Complex, new_nnz, NALU_HYPRE_MEMORY_DEVICE);

#if defined(NALU_HYPRE_USING_SYCL)
   if (elmt_tols == NULL)
   {
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_ii, A_j, A_data),
                                        oneapi::dpl::make_zip_iterator(A_ii, A_j, A_data) + nnz,
                                        A_data,
                                        oneapi::dpl::make_zip_iterator(new_ii, new_j, new_data),
      [tol] (const auto & x) {return !(x < tol);} );

      nalu_hypre_assert( std::get<0>(new_end.base()) == new_ii + new_nnz );
   }
   else
   {
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_ii, A_j, A_data),
                                        oneapi::dpl::make_zip_iterator(A_ii, A_j, A_data) + nnz,
                                        oneapi::dpl::make_zip_iterator(A_data, elmt_tols),
                                        oneapi::dpl::make_zip_iterator(new_ii, new_j, new_data),
                                        cabsfirst_greaterthan_second_pred() );

      nalu_hypre_assert( std::get<0>(new_end.base()) == new_ii + new_nnz );
   }
#else
   if (elmt_tols == NULL)
   {
      auto new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)) + nnz,
                                        A_data,
                                        thrust::make_zip_iterator(thrust::make_tuple(new_ii, new_j, new_data)),
                                        thrust::not1(less_than<NALU_HYPRE_Complex>(tol)) );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == new_ii + new_nnz );
   }
   else
   {
      auto new_end = NALU_HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)) + nnz,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_data, elmt_tols)),
                                        thrust::make_zip_iterator(thrust::make_tuple(new_ii, new_j, new_data)),
                                        cabsfirst_greaterthan_second_pred() );

      nalu_hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == new_ii + new_nnz );
   }
#endif


   nalu_hypre_TFree(A_ii,   NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(A_i,    NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(A_j,    NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(A_data, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_CSRMatrixNumNonzeros(A) = new_nnz;
   nalu_hypre_CSRMatrixI(A) = hypreDevice_CsrRowIndicesToPtrs(nrows, new_nnz, new_ii);
   nalu_hypre_CSRMatrixJ(A) = new_j;
   nalu_hypre_CSRMatrixData(A) = new_data;
   nalu_hypre_TFree(new_ii, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}

#endif /* defined(NALU_HYPRE_USING_GPU) */

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixIdentityDevice
 *
 * A = alp * I
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix *
nalu_hypre_CSRMatrixIdentityDevice(NALU_HYPRE_Int n, NALU_HYPRE_Complex alp)
{
   nalu_hypre_CSRMatrix *A = nalu_hypre_CSRMatrixCreate(n, n, n);

   nalu_hypre_CSRMatrixInitialize_v2(A, 0, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_THRUST_CALL( sequence,
                      nalu_hypre_CSRMatrixI(A),
                      nalu_hypre_CSRMatrixI(A) + n + 1,
                      0  );

   NALU_HYPRE_THRUST_CALL( sequence,
                      nalu_hypre_CSRMatrixJ(A),
                      nalu_hypre_CSRMatrixJ(A) + n,
                      0  );

   NALU_HYPRE_THRUST_CALL( fill,
                      nalu_hypre_CSRMatrixData(A),
                      nalu_hypre_CSRMatrixData(A) + n,
                      alp );

   return A;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixIdentityDevice
 *
 * A = diag(v)
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix *
nalu_hypre_CSRMatrixDiagMatrixFromVectorDevice(NALU_HYPRE_Int n, NALU_HYPRE_Complex *v)
{
   nalu_hypre_CSRMatrix *A = nalu_hypre_CSRMatrixCreate(n, n, n);

   nalu_hypre_CSRMatrixInitialize_v2(A, 0, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_THRUST_CALL( sequence,
                      nalu_hypre_CSRMatrixI(A),
                      nalu_hypre_CSRMatrixI(A) + n + 1,
                      0  );

   NALU_HYPRE_THRUST_CALL( sequence,
                      nalu_hypre_CSRMatrixJ(A),
                      nalu_hypre_CSRMatrixJ(A) + n,
                      0  );

   NALU_HYPRE_THRUST_CALL( copy,
                      v,
                      v + n,
                      nalu_hypre_CSRMatrixData(A) );

   return A;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixIdentityDevice
 *
 * B = diagm(A)
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix *
nalu_hypre_CSRMatrixDiagMatrixFromMatrixDevice(nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int type)
{
   NALU_HYPRE_Int      nrows  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Complex  *diag = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nrows, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_CSRMatrixExtractDiagonalDevice(A, diag, type);

   nalu_hypre_CSRMatrix *diag_mat = nalu_hypre_CSRMatrixDiagMatrixFromVectorDevice(nrows, diag);

   nalu_hypre_TFree(diag, NALU_HYPRE_MEMORY_DEVICE);
   return diag_mat;
}

/*--------------------------------------------------------------------------
 * adj_functor (Used in nalu_hypre_CSRMatrixPermuteDevice)
 *--------------------------------------------------------------------------*/

struct adj_functor : public thrust::unary_function<NALU_HYPRE_Int, NALU_HYPRE_Int>
{
   NALU_HYPRE_Int *ia_;

   adj_functor(NALU_HYPRE_Int *ia)
   {
      ia_ = ia;
   }

   __host__ __device__ NALU_HYPRE_Int operator()(NALU_HYPRE_Int i) const
   {
      return ia_[i + 1] - ia_[i];
   }
};

/*--------------------------------------------------------------------------
 * bii_functor (Used in nalu_hypre_CSRMatrixPermuteDevice)
 *--------------------------------------------------------------------------*/

struct bii_functor
{
   NALU_HYPRE_Int *p_, *ia_, *ib_, *rb_;

   bii_functor(NALU_HYPRE_Int *p, NALU_HYPRE_Int *ia, NALU_HYPRE_Int *ib, NALU_HYPRE_Int *rb)
   {
      p_ = p;
      ia_ = ia;
      ib_ = ib;
      rb_ = rb;
   }

   __host__ __device__ void operator()(NALU_HYPRE_Int i)
   {
      const NALU_HYPRE_Int r = rb_[i];
      rb_[i] = ia_[p_[r]] + i - ib_[r];
   }
};

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixPermuteDevice
 *
 * See nalu_hypre_CSRMatrixPermute.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixPermuteDevice( nalu_hypre_CSRMatrix  *A,
                              NALU_HYPRE_Int        *perm,
                              NALU_HYPRE_Int        *rqperm,
                              nalu_hypre_CSRMatrix  *B )
{
   /* Input matrix */
   NALU_HYPRE_Int         num_rows     = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int         num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int        *A_i          = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int        *A_j          = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Complex    *A_a          = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int        *B_i          = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int        *B_j          = nalu_hypre_CSRMatrixJ(B);
   NALU_HYPRE_Complex    *B_a          = nalu_hypre_CSRMatrixData(B);

   NALU_HYPRE_Int        *B_ii;

   /* Build B_i */
   NALU_HYPRE_THRUST_CALL(gather,
                     perm,
                     perm + num_rows,
                     thrust::make_transform_iterator(thrust::make_counting_iterator(0), adj_functor(A_i)),
                     B_i);
   hypreDevice_IntegerExclusiveScan(num_rows + 1, B_i);

   /* Build B_ii (row indices array) */
   B_ii = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_nonzeros, NALU_HYPRE_MEMORY_DEVICE);
   hypreDevice_CsrRowPtrsToIndices_v2(num_rows, num_nonzeros, B_i, B_ii);
   NALU_HYPRE_THRUST_CALL(for_each,
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(num_nonzeros),
                     bii_functor(perm, A_i, B_i, B_ii));

   /* Build B_j and B_a */
   NALU_HYPRE_THRUST_CALL(gather,
                     B_ii,
                     B_ii + num_nonzeros,
                     thrust::make_zip_iterator(thrust::make_tuple(
                                                  thrust::make_permutation_iterator(rqperm, A_j), A_a)),
                     thrust::make_zip_iterator(thrust::make_tuple(B_j, B_a)));

   /* Free memory */
   nalu_hypre_TFree(B_ii, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}

#endif /* NALU_HYPRE_USING_CUDA || defined(NALU_HYPRE_USING_HIP) */

#if defined(NALU_HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixTransposeDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixTransposeDevice(nalu_hypre_CSRMatrix  *A,
                               nalu_hypre_CSRMatrix **AT_ptr,
                               NALU_HYPRE_Int         data)
{
   NALU_HYPRE_Complex    *A_data   = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int        *A_i      = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int        *A_j      = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int         nrows_A  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int         ncols_A  = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int         nnz_A    = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Complex    *C_data;
   NALU_HYPRE_Int        *C_i;
   NALU_HYPRE_Int        *C_j;
   nalu_hypre_CSRMatrix  *C;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;
   nalu_hypre_GpuProfilingPushRange("CSRMatrixTranspose");

   /* trivial case */
   if (nnz_A == 0)
   {
      C_i =    nalu_hypre_CTAlloc(NALU_HYPRE_Int,     ncols_A + 1, NALU_HYPRE_MEMORY_DEVICE);
      C_j =    nalu_hypre_CTAlloc(NALU_HYPRE_Int,     0,           NALU_HYPRE_MEMORY_DEVICE);
      C_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, 0,           NALU_HYPRE_MEMORY_DEVICE);
   }
   else
   {
      if ( !nalu_hypre_HandleSpTransUseVendor(nalu_hypre_handle()) )
      {
#if defined(NALU_HYPRE_USING_GPU)
         hypreDevice_CSRSpTrans(nrows_A, ncols_A, nnz_A, A_i, A_j, A_data, &C_i, &C_j, &C_data, data);
#endif
      }
      else
      {
#if defined(NALU_HYPRE_USING_CUSPARSE)
         hypreDevice_CSRSpTransCusparse(nrows_A, ncols_A, nnz_A, A_i, A_j, A_data, &C_i, &C_j, &C_data,
                                        data);
#elif defined(NALU_HYPRE_USING_ROCSPARSE)
         hypreDevice_CSRSpTransRocsparse(nrows_A, ncols_A, nnz_A, A_i, A_j, A_data, &C_i, &C_j, &C_data,
                                         data);
#elif defined(NALU_HYPRE_USING_GPU)
         hypreDevice_CSRSpTrans(nrows_A, ncols_A, nnz_A, A_i, A_j, A_data, &C_i, &C_j, &C_data, data);
#endif
      }
   }

   C = nalu_hypre_CSRMatrixCreate(ncols_A, nrows_A, nnz_A);
   nalu_hypre_CSRMatrixI(C) = C_i;
   nalu_hypre_CSRMatrixJ(C) = C_j;
   nalu_hypre_CSRMatrixData(C) = C_data;
   nalu_hypre_CSRMatrixMemoryLocation(C) = NALU_HYPRE_MEMORY_DEVICE;

   *AT_ptr = C;

   nalu_hypre_SyncComputeStream(nalu_hypre_handle());

   nalu_hypre_GpuProfilingPopRange();
   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

#endif /* #if defined(NALU_HYPRE_USING_GPU) */

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixSortRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixSortRow(nalu_hypre_CSRMatrix *A)
{
   nalu_hypre_GpuProfilingPushRange("CSRMatrixSort");

#if defined(NALU_HYPRE_USING_CUSPARSE)
   nalu_hypre_SortCSRCusparse(nalu_hypre_CSRMatrixNumRows(A), nalu_hypre_CSRMatrixNumCols(A),
                         nalu_hypre_CSRMatrixNumNonzeros(A), nalu_hypre_CSRMatrixGPUMatDescr(A),
                         nalu_hypre_CSRMatrixI(A), nalu_hypre_CSRMatrixJ(A), nalu_hypre_CSRMatrixData(A));

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   nalu_hypre_SortCSRRocsparse(nalu_hypre_CSRMatrixNumRows(A), nalu_hypre_CSRMatrixNumCols(A),
                          nalu_hypre_CSRMatrixNumNonzeros(A), nalu_hypre_CSRMatrixGPUMatDescr(A),
                          nalu_hypre_CSRMatrixI(A), nalu_hypre_CSRMatrixJ(A), nalu_hypre_CSRMatrixData(A));
#else
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                     "nalu_hypre_CSRMatrixSortRow only implemented for cuSPARSE/rocSPARSE!\n");
#endif

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

#if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE)

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixSortRowOutOfPlace
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixSortRowOutOfPlace(nalu_hypre_CSRMatrix *A)
{
   NALU_HYPRE_Int     *A_j  = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Complex *A_a  = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int      nnzA = nalu_hypre_CSRMatrixNumNonzeros(A);

   /* if both exist, we assume A has been sorted */
   if (nalu_hypre_CSRMatrixSortedJ(A) && nalu_hypre_CSRMatrixSortedData(A))
   {
      return nalu_hypre_error_flag;
   }

   nalu_hypre_TFree(nalu_hypre_CSRMatrixSortedJ(A), NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(nalu_hypre_CSRMatrixSortedData(A), NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_CSRMatrixSortedJ(A)    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nnzA, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_CSRMatrixSortedData(A) = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzA, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_TMemcpy(nalu_hypre_CSRMatrixSortedJ(A), A_j, NALU_HYPRE_Int, nnzA,
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(nalu_hypre_CSRMatrixSortedData(A), A_a, NALU_HYPRE_Complex, nnzA,
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_CSRMatrixJ(A) = nalu_hypre_CSRMatrixSortedJ(A);
   nalu_hypre_CSRMatrixData(A) = nalu_hypre_CSRMatrixSortedData(A);

   nalu_hypre_CSRMatrixSortRow(A);

   nalu_hypre_CSRMatrixJ(A)    = A_j;
   nalu_hypre_CSRMatrixData(A) = A_a;

   return nalu_hypre_error_flag;
}
#endif /* defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) */

#if defined(NALU_HYPRE_USING_CUSPARSE)

/*--------------------------------------------------------------------------
 * nalu_hypre_SortCSRCusparse
 *
 * Sorts values and column indices in each row in ascending order INPLACE
 *
 * Parameters:
 *   n: Number of rows [in]
 *   m: Number of columns [in]
 *   nnzA: Number of nonzeros [in]
 *   d_ia: row pointers [in/out]
 *   d_ja_sorted: column indices [in/out]
 *   d_a_sorted: coefficients [in/out]
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SortCSRCusparse( NALU_HYPRE_Int            n,
                       NALU_HYPRE_Int            m,
                       NALU_HYPRE_Int            nnzA,
                       cusparseMatDescr_t   descrA,
                       const NALU_HYPRE_Int     *d_ia,
                       NALU_HYPRE_Int           *d_ja_sorted,
                       NALU_HYPRE_Complex       *d_a_sorted )
{
   cusparseHandle_t  cusparsehandle = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
   size_t            pBufferSizeInBytes = 0;
   void             *pBuffer = NULL;
   csru2csrInfo_t    sortInfoA;

   nalu_hypre_GpuProfilingPushRange("SortCSRCusparse");

   NALU_HYPRE_CUSPARSE_CALL( cusparseCreateCsru2csrInfo(&sortInfoA) );
   NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparse_csru2csr_bufferSizeExt(cusparsehandle,
                                                              n, m, nnzA,
                                                              d_a_sorted, d_ia, d_ja_sorted,
                                                              sortInfoA, &pBufferSizeInBytes) );

   pBuffer = nalu_hypre_TAlloc(char, pBufferSizeInBytes, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparse_csru2csr(cusparsehandle,
                                                n, m, nnzA, descrA,
                                                d_a_sorted, d_ia, d_ja_sorted,
                                                sortInfoA, pBuffer) );

   nalu_hypre_TFree(pBuffer, NALU_HYPRE_MEMORY_DEVICE);
   NALU_HYPRE_CUSPARSE_CALL(cusparseDestroyCsru2csrInfo(sortInfoA));

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixTriLowerUpperSolveCusparse
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixTriLowerUpperSolveCusparse(char             uplo,
                                          NALU_HYPRE_Int        unit_diag,
                                          nalu_hypre_CSRMatrix *A,
                                          NALU_HYPRE_Real      *l1_norms,
                                          NALU_HYPRE_Complex   *f_data,
                                          NALU_HYPRE_Complex   *u_data )
{
   NALU_HYPRE_Int              num_rows     = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int              num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int             *A_i          = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int             *A_j          = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Complex         *A_a          = nalu_hypre_CSRMatrixData(A);
   nalu_hypre_CsrsvData       *csrsv_data   = nalu_hypre_CSRMatrixCsrsvData(A);
   NALU_HYPRE_Complex         *A_ma;

   cusparseHandle_t       handle       = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
   cusparseDiagType_t     diag_type    = unit_diag ? CUSPARSE_DIAG_TYPE_UNIT :
                                         CUSPARSE_DIAG_TYPE_NON_UNIT;
   cusparseFillMode_t     fill_mode_L  = CUSPARSE_FILL_MODE_LOWER;
   cusparseFillMode_t     fill_mode_U  = CUSPARSE_FILL_MODE_UPPER;
   cusparseOperation_t    operation    = CUSPARSE_OPERATION_NON_TRANSPOSE;

   NALU_HYPRE_Complex          alpha        = 1.0;

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
   NALU_HYPRE_Int              num_cols     = nalu_hypre_CSRMatrixNumCols(A);
   cusparseSpMatDescr_t   matA;
   cusparseDnVecDescr_t   vecF;
   cusparseDnVecDescr_t   vecU;

   cudaDataType           data_type    = nalu_hypre_HYPREComplexToCudaDataType();
   size_t                 buffer_size;
   char*                  buffer_L;
   char*                  buffer_U;
#else
   cusparseSolvePolicy_t  policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   cusparseMatDescr_t     descr;
   cusparseStatus_t       status;
   NALU_HYPRE_Int             *A_sj;
   nalu_hypre_int              buffer_size;
   char*                  buffer;
   nalu_hypre_int              structural_zero;
   char                   msg[256];

   /* cuSPARSE's legacy API requires sorted rows. Sort and save in CSR's (sj, sa) */
   nalu_hypre_CSRMatrixSortRowOutOfPlace(A);
#endif

   /* setup csrsvdata in CSR: modify the diagonal (once) */
   if (!csrsv_data)
   {
      nalu_hypre_CSRMatrixCsrsvData(A) = nalu_hypre_CsrsvDataCreate();
      csrsv_data = nalu_hypre_CSRMatrixCsrsvData(A);

      nalu_hypre_CsrsvDataMatData(csrsv_data) = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_nonzeros,
                                                        NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_CSRMatrixData(A) = nalu_hypre_CsrsvDataMatData(csrsv_data);

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
      nalu_hypre_TMemcpy(nalu_hypre_CsrsvDataMatData(csrsv_data), A_a, NALU_HYPRE_Complex, num_nonzeros,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
#else
      nalu_hypre_TMemcpy(nalu_hypre_CsrsvDataMatData(csrsv_data), nalu_hypre_CSRMatrixSortedData(A),
                    NALU_HYPRE_Complex, num_nonzeros,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_CSRMatrixJ(A) = nalu_hypre_CSRMatrixSortedJ(A);
#endif

      /* if (l1_norms), replace A's diag with l1_norm, and
       * replace zero diag with inf. so as to skip relaxation for this unknown */
      nalu_hypre_CSRMatrixReplaceDiagDevice(A, l1_norms, INFINITY, 0.0);

      nalu_hypre_CSRMatrixData(A) = A_a;
#if CUSPARSE_VERSION < CUSPARSE_SPSV_VERSION
      nalu_hypre_CSRMatrixJ(A) = A_j;
#endif
   }

   /* Analysis and Solve */
   A_ma = nalu_hypre_CsrsvDataMatData(csrsv_data);

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
   matA = nalu_hypre_CSRMatrixToCusparseSpMat_core(num_rows, num_cols, 0,
                                              num_nonzeros, A_i, A_j, A_ma);
   vecF = nalu_hypre_VectorToCusparseDnVec_core(f_data, num_rows);
   vecU = nalu_hypre_VectorToCusparseDnVec_core(u_data, num_cols);

   NALU_HYPRE_CUSPARSE_CALL( cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE,
                                                  &diag_type, sizeof(cusparseDiagType_t)) );
#else
   A_sj  = nalu_hypre_CSRMatrixSortedJ(A);
   descr = nalu_hypre_CSRMatrixGPUMatDescr(A);
   NALU_HYPRE_CUSPARSE_CALL( cusparseSetMatDiagType(descr, diag_type) );
#endif

   if (uplo == 'L')
   {
#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
      NALU_HYPRE_CUSPARSE_CALL( cusparseSpMatSetAttribute(matA,
                                                     CUSPARSE_SPMAT_FILL_MODE,
                                                     &fill_mode_L,
                                                     sizeof(cusparseFillMode_t)) );
#else
      NALU_HYPRE_CUSPARSE_CALL( cusparseSetMatFillMode(descr, fill_mode_L) );
#endif

      /* TODO (VPM): move the following block to nalu_hypre_CSRMatrixTriSetupCusparse */
      if (!nalu_hypre_CsrsvDataAnalyzedL(csrsv_data))
      {
#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
         NALU_HYPRE_CUSPARSE_CALL( cusparseSpSV_bufferSize(handle, operation,
                                                      &alpha, matA, vecF, vecU, data_type,
                                                      CUSPARSE_SPSV_ALG_DEFAULT,
                                                      nalu_hypre_CsrsvDataInfoL(csrsv_data),
                                                      &buffer_size) );
#else
         NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparse_csrsv2_bufferSize(handle,
                                                               operation,
                                                               num_rows, num_nonzeros, descr,
                                                               A_ma, A_i, A_sj,
                                                               nalu_hypre_CsrsvDataInfoL(csrsv_data),
                                                               &buffer_size) );
#endif

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
         if (nalu_hypre_CsrsvDataBufferSizeL(csrsv_data) < buffer_size)
         {
            buffer_L = nalu_hypre_TReAlloc_v2(nalu_hypre_CsrsvDataBufferL(csrsv_data),
                                         char,
                                         nalu_hypre_CsrsvDataBufferSizeL(csrsv_data),
                                         char,
                                         buffer_size,
                                         NALU_HYPRE_MEMORY_DEVICE);

            nalu_hypre_CsrsvDataBufferL(csrsv_data)     = buffer_L;
            nalu_hypre_CsrsvDataBufferSizeL(csrsv_data) = buffer_size;
         }
#else
         if (nalu_hypre_CsrsvDataBufferSize(csrsv_data) < buffer_size)
         {
            buffer = nalu_hypre_TReAlloc_v2(nalu_hypre_CsrsvDataBuffer(csrsv_data),
                                       char,
                                       nalu_hypre_CsrsvDataBufferSize(csrsv_data),
                                       char,
                                       buffer_size,
                                       NALU_HYPRE_MEMORY_DEVICE);

            nalu_hypre_CsrsvDataBuffer(csrsv_data)     = buffer;
            nalu_hypre_CsrsvDataBufferSize(csrsv_data) = buffer_size;
         }
#endif

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
         NALU_HYPRE_CUSPARSE_CALL( cusparseSpSV_analysis(handle, operation,
                                                    &alpha, matA, vecF, vecU, data_type,
                                                    CUSPARSE_SPSV_ALG_DEFAULT,
                                                    nalu_hypre_CsrsvDataInfoL(csrsv_data),
                                                    nalu_hypre_CsrsvDataBufferL(csrsv_data)) );
#else
         NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparse_csrsv2_analysis(handle,
                                                             operation,
                                                             num_rows, num_nonzeros, descr,
                                                             A_ma, A_i, A_sj,
                                                             nalu_hypre_CsrsvDataInfoL(csrsv_data),
                                                             policy,
                                                             nalu_hypre_CsrsvDataBuffer(csrsv_data)) );

         status = cusparseXcsrsv2_zeroPivot(handle,
                                            nalu_hypre_CsrsvDataInfoL(csrsv_data),
                                            &structural_zero);
         if (CUSPARSE_STATUS_ZERO_PIVOT == status)
         {
            nalu_hypre_sprintf(msg, "A(%d,%d) is missing\n",
                          structural_zero, structural_zero);
            nalu_hypre_error_w_msg(1, msg);
         }
#endif
         nalu_hypre_CsrsvDataAnalyzedL(csrsv_data) = 1;
      }

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
      NALU_HYPRE_CUSPARSE_CALL( cusparseSpSV_solve(handle, operation,
                                              &alpha, matA, vecF, vecU, data_type,
                                              CUSPARSE_SPSV_ALG_DEFAULT,
                                              nalu_hypre_CsrsvDataInfoL(csrsv_data)) );
#else
      NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparse_csrsv2_solve(handle, operation,
                                                       num_rows, num_nonzeros, &alpha, descr,
                                                       A_ma, A_i, A_sj,
                                                       nalu_hypre_CsrsvDataInfoL(csrsv_data),
                                                       f_data, u_data, policy,
                                                       nalu_hypre_CsrsvDataBuffer(csrsv_data)) );
#endif
   }
   else
   {
#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
      NALU_HYPRE_CUSPARSE_CALL( cusparseSpMatSetAttribute(matA,
                                                     CUSPARSE_SPMAT_FILL_MODE,
                                                     &fill_mode_U,
                                                     sizeof(cusparseFillMode_t)) );
#else
      NALU_HYPRE_CUSPARSE_CALL( cusparseSetMatFillMode(descr, fill_mode_U) );
#endif

      /* TODO (VPM): move the following block to nalu_hypre_CSRMatrixTriSetupCusparse */
      if (!nalu_hypre_CsrsvDataAnalyzedU(csrsv_data))
      {
#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
         NALU_HYPRE_CUSPARSE_CALL( cusparseSpSV_bufferSize(handle, operation,
                                                      &alpha, matA, vecF, vecU, data_type,
                                                      CUSPARSE_SPSV_ALG_DEFAULT,
                                                      nalu_hypre_CsrsvDataInfoU(csrsv_data),
                                                      &buffer_size) );
#else
         NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparse_csrsv2_bufferSize(handle,
                                                               operation,
                                                               num_rows, num_nonzeros, descr,
                                                               A_ma, A_i, A_sj,
                                                               nalu_hypre_CsrsvDataInfoU(csrsv_data),
                                                               &buffer_size) );
#endif

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
         if (nalu_hypre_CsrsvDataBufferSizeU(csrsv_data) < buffer_size)
         {
            buffer_U = nalu_hypre_TReAlloc_v2(nalu_hypre_CsrsvDataBufferU(csrsv_data),
                                         char,
                                         nalu_hypre_CsrsvDataBufferSizeU(csrsv_data),
                                         char,
                                         buffer_size,
                                         NALU_HYPRE_MEMORY_DEVICE);

            nalu_hypre_CsrsvDataBufferU(csrsv_data)     = buffer_U;
            nalu_hypre_CsrsvDataBufferSizeU(csrsv_data) = buffer_size;
         }
#else
         if (nalu_hypre_CsrsvDataBufferSize(csrsv_data) < buffer_size)
         {
            buffer = nalu_hypre_TReAlloc_v2(nalu_hypre_CsrsvDataBuffer(csrsv_data),
                                       char,
                                       nalu_hypre_CsrsvDataBufferSize(csrsv_data),
                                       char,
                                       buffer_size,
                                       NALU_HYPRE_MEMORY_DEVICE);

            nalu_hypre_CsrsvDataBuffer(csrsv_data)     = buffer;
            nalu_hypre_CsrsvDataBufferSize(csrsv_data) = buffer_size;
         }
#endif

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
         NALU_HYPRE_CUSPARSE_CALL( cusparseSpSV_analysis(handle, operation,
                                                    &alpha, matA, vecF, vecU, data_type,
                                                    CUSPARSE_SPSV_ALG_DEFAULT,
                                                    nalu_hypre_CsrsvDataInfoU(csrsv_data),
                                                    nalu_hypre_CsrsvDataBufferU(csrsv_data)) );
#else
         NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparse_csrsv2_analysis(handle,
                                                             operation,
                                                             num_rows, num_nonzeros, descr,
                                                             A_ma, A_i, A_sj,
                                                             nalu_hypre_CsrsvDataInfoU(csrsv_data),
                                                             policy,
                                                             nalu_hypre_CsrsvDataBuffer(csrsv_data)) );

         status = cusparseXcsrsv2_zeroPivot(handle,
                                            nalu_hypre_CsrsvDataInfoU(csrsv_data),
                                            &structural_zero);
         if (CUSPARSE_STATUS_ZERO_PIVOT == status)
         {
            nalu_hypre_sprintf(msg, "A(%d,%d) is missing\n",
                          structural_zero, structural_zero);
            nalu_hypre_error_w_msg(1, msg);
         }
#endif
         nalu_hypre_CsrsvDataAnalyzedU(csrsv_data) = 1;
      }

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
      NALU_HYPRE_CUSPARSE_CALL( cusparseSpSV_solve(handle, operation,
                                              &alpha, matA, vecF, vecU, data_type,
                                              CUSPARSE_SPSV_ALG_DEFAULT,
                                              nalu_hypre_CsrsvDataInfoU(csrsv_data)) );
#else
      NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparse_csrsv2_solve(handle,
                                                       operation,
                                                       num_rows, num_nonzeros, &alpha,
                                                       descr, A_ma, A_i, A_sj,
                                                       nalu_hypre_CsrsvDataInfoU(csrsv_data),
                                                       f_data, u_data, policy,
                                                       nalu_hypre_CsrsvDataBuffer(csrsv_data)) );
#endif
   }

   /* Free memory */
#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
   NALU_HYPRE_CUSPARSE_CALL( cusparseDestroySpMat(matA) );
   NALU_HYPRE_CUSPARSE_CALL( cusparseDestroyDnVec(vecF) );
   NALU_HYPRE_CUSPARSE_CALL( cusparseDestroyDnVec(vecU) );
#endif

   return nalu_hypre_error_flag;
}

#elif defined(NALU_HYPRE_USING_ROCSPARSE)

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixTriLowerUpperSolveRocsparse
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixTriLowerUpperSolveRocsparse(char              uplo,
                                           NALU_HYPRE_Int         unit_diag,
                                           nalu_hypre_CSRMatrix  *A,
                                           NALU_HYPRE_Real       *l1_norms,
                                           NALU_HYPRE_Complex    *f_data,
                                           NALU_HYPRE_Complex    *u_data )
{
   NALU_HYPRE_Int            num_rows      = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int            num_nonzeros  = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int           *A_i           = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int           *A_j           = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Complex       *A_a           = nalu_hypre_CSRMatrixData(A);
   nalu_hypre_CsrsvData     *csrsv_data    = nalu_hypre_CSRMatrixCsrsvData(A);

   rocsparse_handle     handle        = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
   rocsparse_mat_descr  descr         = nalu_hypre_CSRMatrixGPUMatDescr(A);
   NALU_HYPRE_Int           *A_sj;
   NALU_HYPRE_Complex       *A_ma;

   rocsparse_status     status;
   rocsparse_diag_type  diag_type     = unit_diag ? rocsparse_diag_type_unit :
                                        rocsparse_diag_type_non_unit;
   NALU_HYPRE_Complex        alpha         = 1.0;
   nalu_hypre_int            structural_zero;
   size_t               buffer_size;
   char                *buffer;
   char                 msg[256];

   /* rocSPARSE requires sorted rows. Sort and save in CSR's (sj, sa) */
   nalu_hypre_CSRMatrixSortRowOutOfPlace(A);

   /* Setup csrsvdata in CSR: modify the diagonal (once) */
   if (!csrsv_data)
   {
      nalu_hypre_CSRMatrixCsrsvData(A) = nalu_hypre_CsrsvDataCreate();
      csrsv_data = nalu_hypre_CSRMatrixCsrsvData(A);

      nalu_hypre_CsrsvDataMatData(csrsv_data) = nalu_hypre_TAlloc(NALU_HYPRE_Complex,
                                                        num_nonzeros,
                                                        NALU_HYPRE_MEMORY_DEVICE);

      nalu_hypre_CSRMatrixData(A) = nalu_hypre_CsrsvDataMatData(csrsv_data);
      nalu_hypre_TMemcpy(nalu_hypre_CsrsvDataMatData(csrsv_data), nalu_hypre_CSRMatrixSortedData(A),
                    NALU_HYPRE_Complex, num_nonzeros,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_CSRMatrixJ(A) = nalu_hypre_CSRMatrixSortedJ(A);

      /* if (l1_norms), replace A's diag with l1_norm, and
       * replace zero diag with inf. so as to skip relaxation for this unknown */
      nalu_hypre_CSRMatrixReplaceDiagDevice(A, l1_norms, INFINITY, 0.0);

      nalu_hypre_CSRMatrixData(A) = A_a;
      nalu_hypre_CSRMatrixJ(A)    = A_j;
   }

   /* Analysis and Solve */
   buffer = nalu_hypre_CsrsvDataBuffer(csrsv_data);
   A_ma   = nalu_hypre_CsrsvDataMatData(csrsv_data);
   A_sj   = nalu_hypre_CSRMatrixSortedJ(A);

   /* Set matrix diagonal type */
   NALU_HYPRE_ROCSPARSE_CALL( rocsparse_set_mat_diag_type(descr, diag_type) );

   if (uplo == 'L')
   {
      NALU_HYPRE_ROCSPARSE_CALL( rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower) );

      /* TODO (VPM): move the following block to nalu_hypre_CSRMatrixTriSetupRocsparse */
      if (!nalu_hypre_CsrsvDataAnalyzedL(csrsv_data))
      {
         NALU_HYPRE_ROCSPARSE_CALL( nalu_hypre_rocsparse_csrsv_buffer_size(handle,
                                                                 rocsparse_operation_none,
                                                                 num_rows, num_nonzeros, descr,
                                                                 A_ma, A_i, A_sj,
                                                                 nalu_hypre_CsrsvDataInfoL(csrsv_data),
                                                                 &buffer_size) );

         if (nalu_hypre_CsrsvDataBufferSize(csrsv_data) < buffer_size)
         {
            buffer = nalu_hypre_TReAlloc_v2(nalu_hypre_CsrsvDataBuffer(csrsv_data),
                                       char, nalu_hypre_CsrsvDataBufferSize(csrsv_data),
                                       char, buffer_size,
                                       NALU_HYPRE_MEMORY_DEVICE);

            nalu_hypre_CsrsvDataBuffer(csrsv_data)     = buffer;
            nalu_hypre_CsrsvDataBufferSize(csrsv_data) = buffer_size;
         }

         NALU_HYPRE_ROCSPARSE_CALL( nalu_hypre_rocsparse_csrsv_analysis(handle, rocsparse_operation_none,
                                                              num_rows, num_nonzeros, descr,
                                                              A_ma, A_i, A_sj,
                                                              nalu_hypre_CsrsvDataInfoL(csrsv_data),
                                                              rocsparse_analysis_policy_reuse,
                                                              rocsparse_solve_policy_auto,
                                                              buffer) );

         status = rocsparse_csrsv_zero_pivot(handle, descr,
                                             nalu_hypre_CsrsvDataInfoL(csrsv_data),
                                             &structural_zero);
         if (rocsparse_status_zero_pivot == status)
         {
            nalu_hypre_sprintf(msg,
                          "nalu_hypre_CSRMatrixTriLowerUpperSolveRocsparse A(%d,%d) is missing\n",
                          structural_zero, structural_zero);
            nalu_hypre_error_w_msg(1, msg);
         }
         nalu_hypre_CsrsvDataAnalyzedL(csrsv_data) = 1;
      }

      NALU_HYPRE_ROCSPARSE_CALL( nalu_hypre_rocsparse_csrsv_solve(handle, rocsparse_operation_none,
                                                        num_rows, num_nonzeros, &alpha,
                                                        descr, A_ma, A_i, A_sj,
                                                        nalu_hypre_CsrsvDataInfoL(csrsv_data),
                                                        f_data, u_data,
                                                        rocsparse_solve_policy_auto,
                                                        buffer) );
   }
   else
   {
      NALU_HYPRE_ROCSPARSE_CALL( rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_upper) );

      /* TODO (VPM): move the following block to nalu_hypre_CSRMatrixTriSetupRocsparse */
      if (!nalu_hypre_CsrsvDataAnalyzedU(csrsv_data))
      {
         NALU_HYPRE_ROCSPARSE_CALL( nalu_hypre_rocsparse_csrsv_buffer_size(handle,
                                                                 rocsparse_operation_none,
                                                                 num_rows, num_nonzeros, descr,
                                                                 A_ma, A_i, A_sj,
                                                                 nalu_hypre_CsrsvDataInfoU(csrsv_data),
                                                                 &buffer_size) );

         if (nalu_hypre_CsrsvDataBufferSize(csrsv_data) < buffer_size)
         {
            buffer = nalu_hypre_TReAlloc_v2(nalu_hypre_CsrsvDataBuffer(csrsv_data),
                                       char, nalu_hypre_CsrsvDataBufferSize(csrsv_data),
                                       char, buffer_size,
                                       NALU_HYPRE_MEMORY_DEVICE);

            nalu_hypre_CsrsvDataBuffer(csrsv_data) = buffer;
            nalu_hypre_CsrsvDataBufferSize(csrsv_data) = buffer_size;
         }

         NALU_HYPRE_ROCSPARSE_CALL( nalu_hypre_rocsparse_csrsv_analysis(handle, rocsparse_operation_none,
                                                              num_rows, num_nonzeros, descr,
                                                              A_ma, A_i, A_sj,
                                                              nalu_hypre_CsrsvDataInfoU(csrsv_data),
                                                              rocsparse_analysis_policy_reuse,
                                                              rocsparse_solve_policy_auto,
                                                              buffer) );

         status = rocsparse_csrsv_zero_pivot(handle, descr,
                                             nalu_hypre_CsrsvDataInfoU(csrsv_data),
                                             &structural_zero);
         if (rocsparse_status_zero_pivot == status)
         {
            nalu_hypre_sprintf(msg,
                          "nalu_hypre_CSRMatrixTriLowerUpperSolveRocsparse A(%d,%d) is missing\n",
                          structural_zero, structural_zero);
            nalu_hypre_error_w_msg(1, msg);
         }
         nalu_hypre_CsrsvDataAnalyzedU(csrsv_data) = 1;
      }

      NALU_HYPRE_ROCSPARSE_CALL( nalu_hypre_rocsparse_csrsv_solve(handle, rocsparse_operation_none,
                                                        num_rows, num_nonzeros, &alpha, descr,
                                                        A_ma, A_i, A_sj,
                                                        nalu_hypre_CsrsvDataInfoU(csrsv_data),
                                                        f_data, u_data,
                                                        rocsparse_solve_policy_auto,
                                                        buffer) );
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SortCSRRocsparse
 *
 * @brief This functions sorts values and column indices in each row in
 *        ascending order OUT-OF-PLACE
 *
 * @param[in] n Number of rows
 * @param[in] m Number of columns
 * @param[in] nnzA Number of nonzeroes
 * @param[in] *d_ia (Unsorted) Row indices
 * @param[in,out] *d_ja_sorted On Start: Unsorted column indices.
 *                             On return: Sorted column indices
 * @param[in,out] *d_a_sorted On Start: Unsorted values.
 *                            On Return: Sorted values corresponding with
 *                                       column indices
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SortCSRRocsparse( NALU_HYPRE_Int            n,
                        NALU_HYPRE_Int            m,
                        NALU_HYPRE_Int            num_nonzeros,
                        rocsparse_mat_descr  descrA,
                        const NALU_HYPRE_Int     *d_ia,
                        NALU_HYPRE_Int           *d_ja_sorted,
                        NALU_HYPRE_Complex       *d_a_sorted )
{
   rocsparse_handle  handle = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
   size_t            pBufferSizeInBytes = 0;
   void             *pBuffer = NULL;
   NALU_HYPRE_Int        *P = NULL;
   NALU_HYPRE_Complex    *d_a_tmp;

   // FIXME: There is not in-place version of csr sort in rocSPARSE currently, so we make
   //        a temporary copy of the data for gthr, sort that, and then copy the sorted values
   //        back to the array being returned. Where there is an in-place version available,
   //        we should use it.
   d_a_tmp  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_nonzeros, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_ROCSPARSE_CALL( rocsparse_csrsort_buffer_size(handle, n, m, num_nonzeros,
                                                       d_ia, d_ja_sorted,
                                                       &pBufferSizeInBytes) );

   pBuffer = nalu_hypre_TAlloc(char, pBufferSizeInBytes, NALU_HYPRE_MEMORY_DEVICE);
   P       = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_nonzeros, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_ROCSPARSE_CALL( rocsparse_create_identity_permutation(handle, num_nonzeros, P) );
   NALU_HYPRE_ROCSPARSE_CALL( rocsparse_csrsort(handle, n, m, num_nonzeros, descrA, d_ia,
                                           d_ja_sorted, P, pBuffer) );
   NALU_HYPRE_ROCSPARSE_CALL( nalu_hypre_rocsparse_gthr(handle, num_nonzeros, d_a_sorted, d_a_tmp, P,
                                              rocsparse_index_base_zero) );

   nalu_hypre_TFree(pBuffer, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(P, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_TMemcpy(d_a_sorted, d_a_tmp, NALU_HYPRE_Complex, num_nonzeros,
                 NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_TFree(d_a_tmp, NALU_HYPRE_MEMORY_DEVICE);

   return nalu_hypre_error_flag;
}
#endif // #if defined(NALU_HYPRE_USING_CUSPARSE) #elif defined(NALU_HYPRE_USING_ROCSPARSE)

#if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE)

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixILU0
 *
 * TODO (VPM): Change this function's name to nalu_hypre_ILU0SetupDevice?
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixILU0(nalu_hypre_CSRMatrix *A)
{
   /* Input matrix data */
   NALU_HYPRE_Int                 num_rows          = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int                 num_cols          = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int                 num_nonzeros      = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int                *A_i               = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int                *A_j               = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Complex            *A_data            = nalu_hypre_CSRMatrixData(A);

   /* Vendor math sparse libraries data */
#if defined(NALU_HYPRE_USING_CUSPARSE)
   csrilu02Info_t            matA_info        = NULL;
   cusparseHandle_t          handle           = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
   cusparseMatDescr_t        descr            = nalu_hypre_CSRMatrixGPUMatDescr(A);
   cusparseSolvePolicy_t     analysis_policy  = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   cusparseSolvePolicy_t     solve_policy     = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   cusparseStatus_t          status;
   NALU_HYPRE_Int                 buffer_size;

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   rocsparse_mat_info        matA_info        = NULL;
   rocsparse_handle          handle           = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
   rocsparse_mat_descr       descr            = nalu_hypre_CSRMatrixGPUMatDescr(A);
   rocsparse_analysis_policy analysis_policy  = rocsparse_analysis_policy_reuse;
   rocsparse_solve_policy    solve_policy     = rocsparse_solve_policy_auto;
   rocsparse_status          status;
   size_t                    buffer_size;
#endif

   void                     *buffer           = NULL;
   NALU_HYPRE_Int                 zero_pivot;
   char                      errmsg[1024];

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;
   nalu_hypre_GpuProfilingPushRange("CSRMatrixILU0");

   /* Sanity check */
   if (num_rows != num_cols)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Not a square matrix!");
      return nalu_hypre_error_flag;
   }

   /*-------------------------------------------------------------------------------------
    * 1. Sort columns inside each row first, we can't assume that's sorted
    *-------------------------------------------------------------------------------------*/

   nalu_hypre_CSRMatrixSortRow(A);

   /*-------------------------------------------------------------------------------------
    * 2. Create info for ilu setup and solve
    *-------------------------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_CUSPARSE)
   NALU_HYPRE_CUSPARSE_CALL(cusparseCreateCsrilu02Info(&matA_info));

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   NALU_HYPRE_ROCSPARSE_CALL(rocsparse_create_mat_info(&matA_info));

#endif

   /*-------------------------------------------------------------------------------------
    * 3. Get work array size
    *-------------------------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_CUSPARSE)
   NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrilu02_bufferSize(handle, num_rows, num_nonzeros,
                                                          descr, A_data, A_i, A_j,
                                                          matA_info, &buffer_size));
#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   NALU_HYPRE_ROCSPARSE_CALL(nalu_hypre_rocsparse_csrilu0_buffer_size(handle, num_rows, num_nonzeros,
                                                            descr, A_data, A_i, A_j,
                                                            matA_info, &buffer_size));
#endif

   /*-------------------------------------------------------------------------------------
    * 4. Create work array on the device
    *-------------------------------------------------------------------------------------*/

   buffer = nalu_hypre_TAlloc(char, buffer_size, NALU_HYPRE_MEMORY_DEVICE);

   /*-------------------------------------------------------------------------------------
    * 5.1 Perform the analysis
    *-------------------------------------------------------------------------------------*/

   nalu_hypre_GpuProfilingPushRange("Analysis");
#if defined(NALU_HYPRE_USING_CUSPARSE)
   NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrilu02_analysis(handle, num_rows, num_nonzeros,
                                                        descr, A_data, A_i, A_j,
                                                        matA_info, analysis_policy, buffer));

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   NALU_HYPRE_ROCSPARSE_CALL(nalu_hypre_rocsparse_csrilu0_analysis(handle, num_rows, num_nonzeros,
                                                         descr, A_data, A_i, A_j,
                                                         matA_info, analysis_policy,
                                                         solve_policy, buffer));
#endif
   nalu_hypre_GpuProfilingPopRange();

   /*-------------------------------------------------------------------------------------
    * 5.2. Check for zero pivots
    *-------------------------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_CUSPARSE)
   status = cusparseXcsrilu02_zeroPivot(handle, matA_info, &zero_pivot);
   if (status == CUSPARSE_STATUS_ZERO_PIVOT)
   {
      nalu_hypre_sprintf(errmsg, "nalu_hypre_ILU: found zero pivot at A(%d, %d) after analysis\n",
                    zero_pivot, zero_pivot);
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, errmsg);
      return nalu_hypre_error_flag;
   }
   else if (status != CUSPARSE_STATUS_SUCCESS)
   {
      nalu_hypre_sprintf(errmsg, "cuSPARSE ERROR (code = %d, %s) at %s:%d\n",
                    status, cusparseGetErrorString(status), __FILE__, __LINE__);
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, errmsg);
      return nalu_hypre_error_flag;
   }

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   status = rocsparse_csrsv_zero_pivot(handle, descr, matA_info, &zero_pivot);
   if (status == rocsparse_status_zero_pivot)
   {
      nalu_hypre_sprintf(errmsg, "nalu_hypre_ILU: found zero pivot at A(%d, %d) after analysis\n",
                    zero_pivot, zero_pivot);
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, errmsg);
      return nalu_hypre_error_flag;
   }
   else if (status != rocsparse_status_success)
   {
      nalu_hypre_sprintf(errmsg, "rocSPARSE ERROR (code = %d) at %s:%d\n",
                    status, __FILE__, __LINE__);
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, errmsg);
      return nalu_hypre_error_flag;
   }
#endif

   /*-------------------------------------------------------------------------------------
    * 6.1 Compute the numerical factorization
    *-------------------------------------------------------------------------------------*/

   nalu_hypre_GpuProfilingPushRange("Factorization");
#if defined(NALU_HYPRE_USING_CUSPARSE)
   NALU_HYPRE_CUSPARSE_CALL(nalu_hypre_cusparse_csrilu02(handle, num_rows, num_nonzeros,
                                               descr, A_data, A_i, A_j,
                                               matA_info, solve_policy, buffer));
#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   NALU_HYPRE_ROCSPARSE_CALL(nalu_hypre_rocsparse_csrilu0(handle, num_rows, num_nonzeros,
                                                descr, A_data, A_i, A_j,
                                                matA_info, solve_policy, buffer));
#endif
   nalu_hypre_GpuProfilingPopRange();

   /*-------------------------------------------------------------------------------------
    * 6.2 Check for zero pivots
    *-------------------------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_CUSPARSE)
   status = cusparseXcsrilu02_zeroPivot(handle, matA_info, &zero_pivot);
   if (status == CUSPARSE_STATUS_ZERO_PIVOT)
   {
      nalu_hypre_sprintf(errmsg, "nalu_hypre_ILU: found zero pivot at A(%d, %d) after factorization\n",
                    zero_pivot, zero_pivot);
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, errmsg);
      return nalu_hypre_error_flag;
   }
   else if (status != CUSPARSE_STATUS_SUCCESS)
   {
      nalu_hypre_sprintf(errmsg, "cuSPARSE ERROR (code = %d, %s) at %s:%d\n",
                    status, cusparseGetErrorString(status), __FILE__, __LINE__);
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, errmsg);
      return nalu_hypre_error_flag;
   }

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   status = rocsparse_csrsv_zero_pivot(handle, descr, matA_info, &zero_pivot);
   if (status == rocsparse_status_zero_pivot)
   {
      nalu_hypre_sprintf(errmsg, "nalu_hypre_ILU: found zero pivot at A(%d, %d) after factorization\n",
                    zero_pivot, zero_pivot);
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, errmsg);
      return nalu_hypre_error_flag;
   }
   else if (status != rocsparse_status_success)
   {
      nalu_hypre_sprintf(errmsg, "rocSPARSE ERROR (code = %d) at %s:%d\n",
                    status, __FILE__, __LINE__);
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, errmsg);
      return nalu_hypre_error_flag;
   }
#endif

   /*-------------------------------------------------------------------------------------
    * 7. Free memory
    *-------------------------------------------------------------------------------------*/

#if defined(NALU_HYPRE_USING_CUSPARSE)
   NALU_HYPRE_CUSPARSE_CALL(cusparseDestroyCsrilu02Info(matA_info));

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
   NALU_HYPRE_ROCSPARSE_CALL(rocsparse_destroy_mat_info(matA_info));
#endif

   /* Free buffer */
   nalu_hypre_TFree(buffer, NALU_HYPRE_MEMORY_DEVICE);

   nalu_hypre_GpuProfilingPopRange();
   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

#endif /* #if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) */

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixSpMVAnalysisDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixSpMVAnalysisDevice(nalu_hypre_CSRMatrix *matrix)
{
#if defined(NALU_HYPRE_USING_ROCSPARSE)
   NALU_HYPRE_ExecutionPolicy  exec = nalu_hypre_GetExecPolicy1( nalu_hypre_CSRMatrixMemoryLocation(matrix) );
   rocsparse_handle       handle = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      NALU_HYPRE_ROCSPARSE_CALL( nalu_hypre_rocsparse_csrmv_analysis(handle,
                                                           rocsparse_operation_none,
                                                           nalu_hypre_CSRMatrixNumRows(matrix),
                                                           nalu_hypre_CSRMatrixNumCols(matrix),
                                                           nalu_hypre_CSRMatrixNumNonzeros(matrix),
                                                           nalu_hypre_CSRMatrixGPUMatDescr(matrix),
                                                           nalu_hypre_CSRMatrixData(matrix),
                                                           nalu_hypre_CSRMatrixI(matrix),
                                                           nalu_hypre_CSRMatrixJ(matrix),
                                                           nalu_hypre_CSRMatrixGPUMatInfo(matrix)) );
   }
#endif /* #if defined(NALU_HYPRE_USING_ROCSPARSE) */

   return nalu_hypre_error_flag;
}
