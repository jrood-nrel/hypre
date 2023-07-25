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
#include "seq_mv.hpp"

#if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)

#if CUSPARSE_VERSION >= CUSPARSE_NEWSPMM_VERSION
#define NALU_HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_SPMV_CSR_ALG2
#define NALU_HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_SPMM_CSR_ALG3

#elif CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
#define NALU_HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_CSRMV_ALG2
#define NALU_HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_SPMM_CSR_ALG1

#else
#define NALU_HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_CSRMV_ALG2
#define NALU_HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_CSRMM_ALG1
#endif

/* y = alpha * A * x + beta * y
 * This function is supposed to be only used inside the other functions in this file
 */
static inline NALU_HYPRE_Int
nalu_hypre_CSRMatrixMatvecDevice2( NALU_HYPRE_Int        trans,
                              NALU_HYPRE_Complex    alpha,
                              nalu_hypre_CSRMatrix *A,
                              nalu_hypre_Vector    *x,
                              NALU_HYPRE_Complex    beta,
                              nalu_hypre_Vector    *y,
                              NALU_HYPRE_Int        offset )
{
   /* Sanity check */
   if (nalu_hypre_VectorData(x) == nalu_hypre_VectorData(y))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "ERROR::x and y are the same pointer in nalu_hypre_CSRMatrixMatvecDevice2");
   }

#if defined(NALU_HYPRE_USING_CUSPARSE)  || \
    defined(NALU_HYPRE_USING_ROCSPARSE) || \
    defined(NALU_HYPRE_USING_ONEMKLSPARSE)

   /* Input variables */
   NALU_HYPRE_Int  num_vectors_x      = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int  num_vectors_y      = nalu_hypre_VectorNumVectors(y);

   /* Local variables */
   NALU_HYPRE_Int  use_vendor = nalu_hypre_HandleSpMVUseVendor(nalu_hypre_handle());

#if defined(NALU_HYPRE_USING_CUSPARSE) && CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
   NALU_HYPRE_Int  multivec_storage_x = nalu_hypre_VectorMultiVecStorageMethod(x);
   NALU_HYPRE_Int  multivec_storage_y = nalu_hypre_VectorMultiVecStorageMethod(y);

   /* Force use of hypre's SpMV for row-wise multivectors */
   if ((num_vectors_x > 1 && multivec_storage_x == 1) ||
       (num_vectors_y > 1 && multivec_storage_y == 1))
   {
      use_vendor = 0;
   }
#else
   /* TODO - enable cuda 10, rocsparse, and onemkle sparse support for multi-vectors */
   if (num_vectors_x > 1 || num_vectors_y > 1)
   {
      use_vendor = 0;
   }
#endif

   if (use_vendor)
   {
#if defined(NALU_HYPRE_USING_CUSPARSE)
      nalu_hypre_CSRMatrixMatvecCusparse(trans, alpha, A, x, beta, y, offset);

#elif defined(NALU_HYPRE_USING_ROCSPARSE)
      nalu_hypre_CSRMatrixMatvecRocsparse(trans, alpha, A, x, beta, y, offset);

#elif defined(NALU_HYPRE_USING_ONEMKLSPARSE)
      nalu_hypre_CSRMatrixMatvecOnemklsparse(trans, alpha, A, x, beta, y, offset);
#endif
   }
   else
#endif // defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) ...
   {
#if defined(NALU_HYPRE_USING_GPU)
      nalu_hypre_CSRMatrixSpMVDevice(trans, alpha, A, x, beta, y, 0);

#elif defined(NALU_HYPRE_USING_DEVICE_OPENMP)
      nalu_hypre_CSRMatrixMatvecOMPOffload(trans, alpha, A, x, beta, y, offset);
#endif
   }

   return nalu_hypre_error_flag;
}

/* y = alpha * A * x + beta * b */
NALU_HYPRE_Int
nalu_hypre_CSRMatrixMatvecDevice( NALU_HYPRE_Int        trans,
                             NALU_HYPRE_Complex    alpha,
                             nalu_hypre_CSRMatrix *A,
                             nalu_hypre_Vector    *x,
                             NALU_HYPRE_Complex    beta,
                             nalu_hypre_Vector    *b,
                             nalu_hypre_Vector    *y,
                             NALU_HYPRE_Int        offset )
{
   //nalu_hypre_GpuProfilingPushRange("CSRMatrixMatvec");
   NALU_HYPRE_Int   num_vectors = nalu_hypre_VectorNumVectors(x);

   // TODO: RL: do we need offset > 0 at all?
   nalu_hypre_assert(offset == 0);

   // VPM: offset > 0 does not work with multivectors. Remove offset? See comment above
   nalu_hypre_assert(!(offset != 0 && num_vectors > 1));
   nalu_hypre_assert(num_vectors > 0);

   NALU_HYPRE_Int nx = trans ? nalu_hypre_CSRMatrixNumRows(A) : nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int ny = trans ? nalu_hypre_CSRMatrixNumCols(A) : nalu_hypre_CSRMatrixNumRows(A);

   //RL: Note the "<=", since the vectors sometimes can be temporary work spaces that have
   //    large sizes than the needed (such as in par_cheby.c)
   nalu_hypre_assert(ny <= nalu_hypre_VectorSize(y));
   nalu_hypre_assert(nx <= nalu_hypre_VectorSize(x));
   nalu_hypre_assert(ny <= nalu_hypre_VectorSize(b));

   //nalu_hypre_CSRMatrixPrefetch(A, NALU_HYPRE_MEMORY_DEVICE);
   //nalu_hypre_SeqVectorPrefetch(x, NALU_HYPRE_MEMORY_DEVICE);
   //nalu_hypre_SeqVectorPrefetch(b, NALU_HYPRE_MEMORY_DEVICE);
   //if (nalu_hypre_VectorData(b) != nalu_hypre_VectorData(y))
   //{
   //   nalu_hypre_SeqVectorPrefetch(y, NALU_HYPRE_MEMORY_DEVICE);
   //}

   if (nalu_hypre_VectorData(b) != nalu_hypre_VectorData(y))
   {
      nalu_hypre_TMemcpy( nalu_hypre_VectorData(y) + offset,
                     nalu_hypre_VectorData(b) + offset,
                     NALU_HYPRE_Complex,
                     (ny - offset) * num_vectors,
                     nalu_hypre_VectorMemoryLocation(y),
                     nalu_hypre_VectorMemoryLocation(b) );

   }

   if (nalu_hypre_CSRMatrixNumNonzeros(A) <= 0 || alpha == 0.0)
   {
      nalu_hypre_SeqVectorScale(beta, y);
   }
   else
   {
      nalu_hypre_CSRMatrixMatvecDevice2(trans, alpha, A, x, beta, y, offset);
   }

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncComputeStream(nalu_hypre_handle());
#endif

   //nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

#if defined(NALU_HYPRE_USING_CUSPARSE)
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixMatvecCusparseNewAPI
 *
 * Sparse Matrix/(Multi)Vector interface to cusparse's API 11
 *
 * Note: The descriptor variables are not saved to allow for generic input
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixMatvecCusparseNewAPI( NALU_HYPRE_Int        trans,
                                     NALU_HYPRE_Complex    alpha,
                                     nalu_hypre_CSRMatrix *A,
                                     nalu_hypre_Vector    *x,
                                     NALU_HYPRE_Complex    beta,
                                     nalu_hypre_Vector    *y,
                                     NALU_HYPRE_Int        offset )
{
   /* Input variables */
   NALU_HYPRE_Int         num_vectors = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int         num_cols    = trans ? nalu_hypre_CSRMatrixNumRows(A) : nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int         num_rows    = trans ? nalu_hypre_CSRMatrixNumCols(A) : nalu_hypre_CSRMatrixNumRows(A);
   nalu_hypre_CSRMatrix  *AT;
   nalu_hypre_CSRMatrix  *B;

   /* SpMV data */
   size_t                    bufferSize = 0;
   char                     *dBuffer    = nalu_hypre_CSRMatrixGPUMatSpMVBuffer(A);
   cusparseHandle_t          handle     = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
   const cudaDataType        data_type  = nalu_hypre_HYPREComplexToCudaDataType();
   const cusparseIndexType_t index_type = nalu_hypre_HYPREIntToCusparseIndexType();

   /* Local cusparse descriptor variables */
   cusparseSpMatDescr_t      matA;
   cusparseDnVecDescr_t      vecX, vecY;
   cusparseDnMatDescr_t      matX, matY;

   /* We handle the transpose explicitly to ensure the same output each run
    * and for potential performance improvement memory for AT */
   if (trans)
   {
      nalu_hypre_CSRMatrixTransposeDevice(A, &AT, 1);
      B = AT;
   }
   else
   {
      B = A;
   }

   /* Create cuSPARSE vector data structures */
   matA = nalu_hypre_CSRMatrixToCusparseSpMat(B, offset);
   if (num_vectors == 1)
   {
      vecX = nalu_hypre_VectorToCusparseDnVec(x, 0, num_cols);
      vecY = nalu_hypre_VectorToCusparseDnVec(y, offset, num_rows - offset);
   }
   else
   {
      matX = nalu_hypre_VectorToCusparseDnMat(x);
      matY = nalu_hypre_VectorToCusparseDnMat(y);
   }

   if (!dBuffer)
   {
      if (num_vectors == 1)
      {
         NALU_HYPRE_CUSPARSE_CALL( cusparseSpMV_bufferSize(handle,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      &alpha,
                                                      matA,
                                                      vecX,
                                                      &beta,
                                                      vecY,
                                                      data_type,
                                                      NALU_HYPRE_CUSPARSE_SPMV_ALG,
                                                      &bufferSize) );
      }
      else
      {
         NALU_HYPRE_CUSPARSE_CALL( cusparseSpMM_bufferSize(handle,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      &alpha,
                                                      matA,
                                                      matX,
                                                      &beta,
                                                      matY,
                                                      data_type,
                                                      NALU_HYPRE_CUSPARSE_SPMM_ALG,
                                                      &bufferSize) );
      }

      dBuffer = nalu_hypre_TAlloc(char, bufferSize, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_CSRMatrixGPUMatSpMVBuffer(A) = dBuffer;

#if CUSPARSE_VERSION >= CUSPARSE_NEWSPMM_VERSION
      if (num_vectors > 1)
      {
         NALU_HYPRE_CUSPARSE_CALL( cusparseSpMM_preprocess(handle,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      &alpha,
                                                      matA,
                                                      matX,
                                                      &beta,
                                                      matY,
                                                      data_type,
                                                      NALU_HYPRE_CUSPARSE_SPMM_ALG,
                                                      dBuffer) );
      }
#endif
   }

   if (num_vectors == 1)
   {
      NALU_HYPRE_CUSPARSE_CALL( cusparseSpMV(handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        matA,
                                        vecX,
                                        &beta,
                                        vecY,
                                        data_type,
                                        NALU_HYPRE_CUSPARSE_SPMV_ALG,
                                        dBuffer) );
   }
   else
   {
      NALU_HYPRE_CUSPARSE_CALL( cusparseSpMM(handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        matA,
                                        matX,
                                        &beta,
                                        matY,
                                        data_type,
                                        NALU_HYPRE_CUSPARSE_SPMM_ALG,
                                        dBuffer) );
   }

#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_SyncComputeStream(nalu_hypre_handle());
#endif

   /* Free memory */
   NALU_HYPRE_CUSPARSE_CALL( cusparseDestroySpMat(matA) );
   if (num_vectors == 1)
   {
      NALU_HYPRE_CUSPARSE_CALL( cusparseDestroyDnVec(vecX) );
      NALU_HYPRE_CUSPARSE_CALL( cusparseDestroyDnVec(vecY) );
   }
   else
   {
      NALU_HYPRE_CUSPARSE_CALL( cusparseDestroyDnMat(matX) );
      NALU_HYPRE_CUSPARSE_CALL( cusparseDestroyDnMat(matY) );
   }
   if (trans)
   {
      nalu_hypre_CSRMatrixDestroy(AT);
   }

   return nalu_hypre_error_flag;
}

#else // #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

NALU_HYPRE_Int
nalu_hypre_CSRMatrixMatvecCusparseOldAPI( NALU_HYPRE_Int        trans,
                                     NALU_HYPRE_Complex    alpha,
                                     nalu_hypre_CSRMatrix *A,
                                     nalu_hypre_Vector    *x,
                                     NALU_HYPRE_Complex    beta,
                                     nalu_hypre_Vector    *y,
                                     NALU_HYPRE_Int        offset )
{
#ifdef NALU_HYPRE_BIGINT
#error "ERROR: cusparse old API should not be used when bigint is enabled!"
#endif
   cusparseHandle_t handle = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
   cusparseMatDescr_t descr = nalu_hypre_CSRMatrixGPUMatDescr(A);
   nalu_hypre_CSRMatrix *B;

   if (trans)
   {
      nalu_hypre_CSRMatrixTransposeDevice(A, &B, 1);
   }
   else
   {
      B = A;
   }

   NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparse_csrmv(handle,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             nalu_hypre_CSRMatrixNumRows(B) - offset,
                                             nalu_hypre_CSRMatrixNumCols(B),
                                             nalu_hypre_CSRMatrixNumNonzeros(B),
                                             &alpha,
                                             descr,
                                             nalu_hypre_CSRMatrixData(B),
                                             nalu_hypre_CSRMatrixI(B) + offset,
                                             nalu_hypre_CSRMatrixJ(B),
                                             nalu_hypre_VectorData(x),
                                             &beta,
                                             nalu_hypre_VectorData(y) + offset) );

   if (trans)
   {
      nalu_hypre_CSRMatrixDestroy(B);
   }

   return nalu_hypre_error_flag;
}

#endif // #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

NALU_HYPRE_Int
nalu_hypre_CSRMatrixMatvecCusparse( NALU_HYPRE_Int        trans,
                               NALU_HYPRE_Complex    alpha,
                               nalu_hypre_CSRMatrix *A,
                               nalu_hypre_Vector    *x,
                               NALU_HYPRE_Complex    beta,
                               nalu_hypre_Vector    *y,
                               NALU_HYPRE_Int        offset )
{
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
   /* Luke E: The generic API is techinically supported on 10.1,10.2 as a preview,
    * with Dscrmv being deprecated. However, there are limitations.
    * While in Cuda < 11, there are specific mentions of using csr2csc involving
    * transposed matrix products with dcsrm*,
    * they are not present in SpMV interface.
    */
   nalu_hypre_CSRMatrixMatvecCusparseNewAPI(trans, alpha, A, x, beta, y, offset);

#else
   nalu_hypre_CSRMatrixMatvecCusparseOldAPI(trans, alpha, A, x, beta, y, offset);
#endif

   return nalu_hypre_error_flag;
}

#endif // #if defined(NALU_HYPRE_USING_CUSPARSE)

#if defined(NALU_HYPRE_USING_ROCSPARSE)
NALU_HYPRE_Int
nalu_hypre_CSRMatrixMatvecRocsparse( NALU_HYPRE_Int        trans,
                                NALU_HYPRE_Complex    alpha,
                                nalu_hypre_CSRMatrix *A,
                                nalu_hypre_Vector    *x,
                                NALU_HYPRE_Complex    beta,
                                nalu_hypre_Vector    *y,
                                NALU_HYPRE_Int        offset )
{
   rocsparse_handle handle = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
   rocsparse_mat_descr descr = nalu_hypre_CSRMatrixGPUMatDescr(A);
   rocsparse_mat_info info = nalu_hypre_CSRMatrixGPUMatInfo(A);

   nalu_hypre_CSRMatrix *B;

   if (trans)
   {
      nalu_hypre_CSRMatrixTransposeDevice(A, &B, 1);
   }
   else
   {
      B = A;
   }

   NALU_HYPRE_ROCSPARSE_CALL( nalu_hypre_rocsparse_csrmv(handle,
                                               rocsparse_operation_none,
                                               nalu_hypre_CSRMatrixNumRows(B) - offset,
                                               nalu_hypre_CSRMatrixNumCols(B),
                                               nalu_hypre_CSRMatrixNumNonzeros(B),
                                               &alpha,
                                               descr,
                                               nalu_hypre_CSRMatrixData(B),
                                               nalu_hypre_CSRMatrixI(B) + offset,
                                               nalu_hypre_CSRMatrixJ(B),
                                               info,
                                               nalu_hypre_VectorData(x),
                                               &beta,
                                               nalu_hypre_VectorData(y) + offset) );

   if (trans)
   {
      nalu_hypre_CSRMatrixDestroy(B);
   }

   return nalu_hypre_error_flag;
}
#endif // #if defined(NALU_HYPRE_USING_ROCSPARSE)

#if defined(NALU_HYPRE_USING_ONEMKLSPARSE)
NALU_HYPRE_Int
nalu_hypre_CSRMatrixMatvecOnemklsparse( NALU_HYPRE_Int        trans,
                                   NALU_HYPRE_Complex    alpha,
                                   nalu_hypre_CSRMatrix *A,
                                   nalu_hypre_Vector    *x,
                                   NALU_HYPRE_Complex    beta,
                                   nalu_hypre_Vector    *y,
                                   NALU_HYPRE_Int        offset )
{
   sycl::queue *compute_queue = nalu_hypre_HandleComputeStream(nalu_hypre_handle());
   nalu_hypre_CSRMatrix *AT;
   oneapi::mkl::sparse::matrix_handle_t matA_handle = nalu_hypre_CSRMatrixGPUMatHandle(A);
   nalu_hypre_GPUMatDataSetCSRData(A);

   if (trans)
   {
      nalu_hypre_CSRMatrixTransposeDevice(A, &AT, 1);
      nalu_hypre_GPUMatDataSetCSRData(AT);
      matA_handle = nalu_hypre_CSRMatrixGPUMatHandle(AT);
   }

   NALU_HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::gemv(*compute_queue,
                                                oneapi::mkl::transpose::nontrans,
                                                alpha,
                                                matA_handle,
                                                nalu_hypre_VectorData(x),
                                                beta,
                                                nalu_hypre_VectorData(y) + offset).wait() );

   if (trans)
   {
      nalu_hypre_CSRMatrixDestroy(AT);
   }

   return nalu_hypre_error_flag;
}
#endif // #if defined(NALU_HYPRE_USING_ROCSPARSE)

#endif // #if defined(NALU_HYPRE_USING_GPU) || defined(NALU_HYPRE_USING_DEVICE_OPENMP)
