/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "_nalu_hypre_utilities.hpp"
#include "seq_mv.hpp"
#include "csr_spgemm_device.h"

#if defined(NALU_HYPRE_USING_CUDA) && defined(NALU_HYPRE_USING_CUSPARSE)

NALU_HYPRE_Int
hypreDevice_CSRSpGemmCusparse(NALU_HYPRE_Int          m,
                              NALU_HYPRE_Int          k,
                              NALU_HYPRE_Int          n,
                              cusparseMatDescr_t descr_A,
                              NALU_HYPRE_Int          nnzA,
                              NALU_HYPRE_Int         *d_ia,
                              NALU_HYPRE_Int         *d_ja,
                              NALU_HYPRE_Complex     *d_a,
                              cusparseMatDescr_t descr_B,
                              NALU_HYPRE_Int          nnzB,
                              NALU_HYPRE_Int         *d_ib,
                              NALU_HYPRE_Int         *d_jb,
                              NALU_HYPRE_Complex     *d_b,
                              cusparseMatDescr_t descr_C,
                              NALU_HYPRE_Int         *nnzC_out,
                              NALU_HYPRE_Int        **d_ic_out,
                              NALU_HYPRE_Int        **d_jc_out,
                              NALU_HYPRE_Complex    **d_c_out)
{
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
   hypreDevice_CSRSpGemmCusparseGenericAPI(m, k, n,
                                           nnzA, d_ia, d_ja, d_a,
                                           nnzB, d_ib, d_jb, d_b,
                                           nnzC_out, d_ic_out, d_jc_out, d_c_out);
#else
   hypreDevice_CSRSpGemmCusparseOldAPI(m, k, n,
                                       descr_A, nnzA, d_ia, d_ja, d_a,
                                       descr_B, nnzB, d_ib, d_jb, d_b,
                                       descr_C, nnzC_out, d_ic_out, d_jc_out, d_c_out);
#endif
   return nalu_hypre_error_flag;
}

#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

/*
 * @brief Uses Cusparse to calculate a sparse-matrix x sparse-matrix product in CSRS format. Supports Cusparse generic API (11+)
 *
 * @param[in] m Number of rows of A,C
 * @param[in] k Number of columns of B,C
 * @param[in] n Number of columns of A, number of rows of B
 * @param[in] nnzA Number of nonzeros in A
 * @param[in] *d_ia Array containing the row pointers of A
 * @param[in] *d_ja Array containing the column indices of A
 * @param[in] *d_a Array containing values of A
 * @param[in] nnzB Number of nonzeros in B
 * @param[in] *d_ib Array containing the row pointers of B
 * @param[in] *d_jb Array containing the column indices of B
 * @param[in] *d_b Array containing values of B
 * @param[out] *nnzC_out Pointer to address with number of nonzeros in C
 * @param[out] *d_ic_out Array containing the row pointers of C
 * @param[out] *d_jc_out Array containing the column indices of C
 * @param[out] *d_c_out Array containing values of C
 */

NALU_HYPRE_Int
hypreDevice_CSRSpGemmCusparseGenericAPI(NALU_HYPRE_Int       m,
                                        NALU_HYPRE_Int       k,
                                        NALU_HYPRE_Int       n,
                                        NALU_HYPRE_Int       nnzA,
                                        NALU_HYPRE_Int      *d_ia,
                                        NALU_HYPRE_Int      *d_ja,
                                        NALU_HYPRE_Complex  *d_a,
                                        NALU_HYPRE_Int       nnzB,
                                        NALU_HYPRE_Int      *d_ib,
                                        NALU_HYPRE_Int      *d_jb,
                                        NALU_HYPRE_Complex  *d_b,
                                        NALU_HYPRE_Int      *nnzC_out,
                                        NALU_HYPRE_Int     **d_ic_out,
                                        NALU_HYPRE_Int     **d_jc_out,
                                        NALU_HYPRE_Complex **d_c_out)
{
   cusparseHandle_t cusparsehandle = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());

   //Initialize the descriptors for the mats
   cusparseSpMatDescr_t matA = nalu_hypre_CSRMatrixToCusparseSpMat_core(m, k, 0, nnzA, d_ia, d_ja, d_a);
   cusparseSpMatDescr_t matB = nalu_hypre_CSRMatrixToCusparseSpMat_core(k, n, 0, nnzB, d_ib, d_jb, d_b);
   cusparseSpMatDescr_t matC = nalu_hypre_CSRMatrixToCusparseSpMat_core(m, n, 0, 0,    NULL, NULL, NULL);
   cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
   cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

   /* Create the SpGEMM Descriptor */
   cusparseSpGEMMDescr_t spgemmDesc;
   NALU_HYPRE_CUSPARSE_CALL( cusparseSpGEMM_createDescr(&spgemmDesc) );

   cudaDataType computeType = nalu_hypre_HYPREComplexToCudaDataType();
   NALU_HYPRE_Complex alpha = 1.0;
   NALU_HYPRE_Complex beta = 0.0;
   size_t bufferSize1;
   size_t bufferSize2;
   void *dBuffer1 = NULL;
   void *dBuffer2 = NULL;

#ifdef NALU_HYPRE_SPGEMM_TIMING
   NALU_HYPRE_Real t1, t2;
#endif

#ifdef NALU_HYPRE_SPGEMM_TIMING
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   t1 = nalu_hypre_MPI_Wtime();
#endif

   /* Do work estimation */
   NALU_HYPRE_CUSPARSE_CALL( cusparseSpGEMM_workEstimation(cusparsehandle, opA, opB,
                                                      &alpha, matA, matB, &beta, matC,
                                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                      spgemmDesc, &bufferSize1, NULL) );
   dBuffer1 = nalu_hypre_TAlloc(char, bufferSize1, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_CUSPARSE_CALL( cusparseSpGEMM_workEstimation(cusparsehandle, opA, opB,
                                                      &alpha, matA, matB, &beta, matC,
                                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                      spgemmDesc, &bufferSize1, dBuffer1) );

#ifdef NALU_HYPRE_SPGEMM_TIMING
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   t2 = nalu_hypre_MPI_Wtime() - t1;
   nalu_hypre_printf("WorkEst %f\n", t2);
#endif

#ifdef NALU_HYPRE_SPGEMM_TIMING
   t1 = nalu_hypre_MPI_Wtime();
#endif

   /* Do computation */
   NALU_HYPRE_CUSPARSE_CALL( cusparseSpGEMM_compute(cusparsehandle, opA, opB,
                                               &alpha, matA, matB, &beta, matC,
                                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                                               spgemmDesc, &bufferSize2, NULL) );

   dBuffer2  = nalu_hypre_TAlloc(char, bufferSize2, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_CUSPARSE_CALL( cusparseSpGEMM_compute(cusparsehandle, opA, opB,
                                               &alpha, matA, matB, &beta, matC,
                                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                                               spgemmDesc, &bufferSize2, dBuffer2) );

#ifdef NALU_HYPRE_SPGEMM_TIMING
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   t2 = nalu_hypre_MPI_Wtime() - t1;
   nalu_hypre_printf("Compute %f\n", t2);
#endif

#ifdef NALU_HYPRE_SPGEMM_TIMING
   t1 = nalu_hypre_MPI_Wtime();
#endif

   /* Required by cusparse api (as of 11) to be int64_t */
   int64_t C_num_rows, C_num_cols, nnzC;
   NALU_HYPRE_Int *d_ic, *d_jc;
   NALU_HYPRE_Complex *d_c;

   /* Get required information for C */
   NALU_HYPRE_CUSPARSE_CALL( cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &nnzC) );

   nalu_hypre_assert(C_num_rows == m);
   nalu_hypre_assert(C_num_cols == n);

   d_ic = nalu_hypre_TAlloc(NALU_HYPRE_Int,     C_num_rows + 1, NALU_HYPRE_MEMORY_DEVICE);
   d_jc = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nnzC,         NALU_HYPRE_MEMORY_DEVICE);
   d_c  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzC,         NALU_HYPRE_MEMORY_DEVICE);

   /* Setup the required descriptor for C */
   NALU_HYPRE_CUSPARSE_CALL(cusparseCsrSetPointers(matC, d_ic, d_jc, d_c));

   /* Copy the data into C */
   NALU_HYPRE_CUSPARSE_CALL(cusparseSpGEMM_copy( cusparsehandle, opA, opB,
                                            &alpha, matA, matB, &beta, matC,
                                            computeType, CUSPARSE_SPGEMM_DEFAULT,
                                            spgemmDesc) );

#ifdef NALU_HYPRE_SPGEMM_TIMING
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   t2 = nalu_hypre_MPI_Wtime() - t1;
   nalu_hypre_printf("Copy %f\n", t2);
#endif

   /* Cleanup the data */
   NALU_HYPRE_CUSPARSE_CALL( cusparseSpGEMM_destroyDescr(spgemmDesc) );
   NALU_HYPRE_CUSPARSE_CALL( cusparseDestroySpMat(matA) );
   NALU_HYPRE_CUSPARSE_CALL( cusparseDestroySpMat(matB) );
   NALU_HYPRE_CUSPARSE_CALL( cusparseDestroySpMat(matC) );

   nalu_hypre_TFree(dBuffer1, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TFree(dBuffer2, NALU_HYPRE_MEMORY_DEVICE);

   /* Assign the output */
   *nnzC_out = nnzC;
   *d_ic_out = d_ic;
   *d_jc_out = d_jc;
   *d_c_out = d_c;

   return nalu_hypre_error_flag;
}

#else

NALU_HYPRE_Int
hypreDevice_CSRSpGemmCusparseOldAPI(NALU_HYPRE_Int          m,
                                    NALU_HYPRE_Int          k,
                                    NALU_HYPRE_Int          n,
                                    cusparseMatDescr_t descr_A,
                                    NALU_HYPRE_Int          nnzA,
                                    NALU_HYPRE_Int         *d_ia,
                                    NALU_HYPRE_Int         *d_ja,
                                    NALU_HYPRE_Complex     *d_a,
                                    cusparseMatDescr_t descr_B,
                                    NALU_HYPRE_Int          nnzB,
                                    NALU_HYPRE_Int         *d_ib,
                                    NALU_HYPRE_Int         *d_jb,
                                    NALU_HYPRE_Complex     *d_b,
                                    cusparseMatDescr_t descr_C,
                                    NALU_HYPRE_Int         *nnzC_out,
                                    NALU_HYPRE_Int        **d_ic_out,
                                    NALU_HYPRE_Int        **d_jc_out,
                                    NALU_HYPRE_Complex    **d_c_out)
{
   NALU_HYPRE_Int  *d_ic, *d_jc, baseC, nnzC;
   NALU_HYPRE_Int  *d_ja_sorted, *d_jb_sorted;
   NALU_HYPRE_Complex *d_c, *d_a_sorted, *d_b_sorted;

#ifdef NALU_HYPRE_SPGEMM_TIMING
   NALU_HYPRE_Real t1, t2;
#endif

#ifdef NALU_HYPRE_SPGEMM_TIMING
   t1 = nalu_hypre_MPI_Wtime();
#endif

   /* Allocate space for sorted arrays */
   d_a_sorted  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzA, NALU_HYPRE_MEMORY_DEVICE);
   d_b_sorted  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzB, NALU_HYPRE_MEMORY_DEVICE);
   d_ja_sorted = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nnzA, NALU_HYPRE_MEMORY_DEVICE);
   d_jb_sorted = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nnzB, NALU_HYPRE_MEMORY_DEVICE);

   cusparseHandle_t cusparsehandle = nalu_hypre_HandleCusparseHandle(nalu_hypre_handle());
   cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
   cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;

   /* Copy the unsorted over as the initial "sorted" */
   nalu_hypre_TMemcpy(d_ja_sorted, d_ja, NALU_HYPRE_Int,     nnzA, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(d_a_sorted,  d_a,  NALU_HYPRE_Complex, nnzA, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(d_jb_sorted, d_jb, NALU_HYPRE_Int,     nnzB, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_TMemcpy(d_b_sorted,  d_b,  NALU_HYPRE_Complex, nnzB, NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_DEVICE);

   /* Sort each of the CSR matrices */
   nalu_hypre_SortCSRCusparse(m, k, nnzA, descr_A, d_ia, d_ja_sorted, d_a_sorted);
   nalu_hypre_SortCSRCusparse(k, n, nnzB, descr_B, d_ib, d_jb_sorted, d_b_sorted);

#ifdef NALU_HYPRE_SPGEMM_TIMING
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   t2 = nalu_hypre_MPI_Wtime() - t1;
   nalu_hypre_printf("sort %f\n", t2);
#endif

#ifdef NALU_HYPRE_SPGEMM_TIMING
   t1 = nalu_hypre_MPI_Wtime();
#endif

   // nnzTotalDevHostPtr points to host memory
   NALU_HYPRE_Int *nnzTotalDevHostPtr = &nnzC;
   NALU_HYPRE_CUSPARSE_CALL( cusparseSetPointerMode(cusparsehandle, CUSPARSE_POINTER_MODE_HOST) );

   d_ic = nalu_hypre_TAlloc(NALU_HYPRE_Int, m + 1, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_CUSPARSE_CALL( cusparseXcsrgemmNnz(cusparsehandle, transA, transB,
                                            m, n, k,
                                            descr_A, nnzA, d_ia, d_ja_sorted,
                                            descr_B, nnzB, d_ib, d_jb_sorted,
                                            descr_C,       d_ic, nnzTotalDevHostPtr ) );

   /* RL: this if is always true (code copied from cusparse manual */
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

#ifdef NALU_HYPRE_SPGEMM_TIMING
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   t2 = nalu_hypre_MPI_Wtime() - t1;
   nalu_hypre_printf("csrgemmNnz %f\n", t2);
#endif

#ifdef NALU_HYPRE_SPGEMM_TIMING
   t1 = nalu_hypre_MPI_Wtime();
#endif

   d_jc = nalu_hypre_TAlloc(NALU_HYPRE_Int,     nnzC, NALU_HYPRE_MEMORY_DEVICE);
   d_c  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnzC, NALU_HYPRE_MEMORY_DEVICE);

   NALU_HYPRE_CUSPARSE_CALL( nalu_hypre_cusparse_csrgemm(cusparsehandle, transA, transB, m, n, k,
                                               descr_A, nnzA, d_a_sorted, d_ia, d_ja_sorted,
                                               descr_B, nnzB, d_b_sorted, d_ib, d_jb_sorted,
                                               descr_C,       d_c, d_ic, d_jc) );

#ifdef NALU_HYPRE_SPGEMM_TIMING
   nalu_hypre_ForceSyncComputeStream(nalu_hypre_handle());
   t2 = nalu_hypre_MPI_Wtime() - t1;
   nalu_hypre_printf("csrgemm %f\n", t2);
#endif

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

#endif /* #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION */
#endif /* #if defined(NALU_HYPRE_USING_CUDA) && defined(NALU_HYPRE_USING_CUSPARSE) */

