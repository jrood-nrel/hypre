/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#ifndef SEQ_MV_HPP
#define SEQ_MV_HPP

#ifdef __cplusplus
extern "C" {
#endif

#if defined(NALU_HYPRE_USING_CUSPARSE)
cusparseSpMatDescr_t nalu_hypre_CSRMatrixToCusparseSpMat(const nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int offset);

cusparseSpMatDescr_t nalu_hypre_CSRMatrixToCusparseSpMat_core( NALU_HYPRE_Int n, NALU_HYPRE_Int m,
                                                          NALU_HYPRE_Int offset, NALU_HYPRE_Int nnz, NALU_HYPRE_Int *i, NALU_HYPRE_Int *j, NALU_HYPRE_Complex *data);

cusparseDnVecDescr_t nalu_hypre_VectorToCusparseDnVec(const nalu_hypre_Vector *x, NALU_HYPRE_Int offset,
                                                 NALU_HYPRE_Int size_override);

cusparseDnMatDescr_t nalu_hypre_VectorToCusparseDnMat(const nalu_hypre_Vector *x);

NALU_HYPRE_Int hypreDevice_CSRSpGemmCusparseOldAPI(NALU_HYPRE_Int m, NALU_HYPRE_Int k, NALU_HYPRE_Int n,
                                              cusparseMatDescr_t descr_A, NALU_HYPRE_Int nnzA, NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja, NALU_HYPRE_Complex *d_a,
                                              cusparseMatDescr_t descr_B, NALU_HYPRE_Int nnzB, NALU_HYPRE_Int *d_ib, NALU_HYPRE_Int *d_jb, NALU_HYPRE_Complex *d_b,
                                              cusparseMatDescr_t descr_C, NALU_HYPRE_Int *nnzC_out, NALU_HYPRE_Int **d_ic_out, NALU_HYPRE_Int **d_jc_out,
                                              NALU_HYPRE_Complex **d_c_out);

NALU_HYPRE_Int hypreDevice_CSRSpGemmCusparse(NALU_HYPRE_Int m, NALU_HYPRE_Int k, NALU_HYPRE_Int n,
                                        cusparseMatDescr_t descr_A, NALU_HYPRE_Int nnzA, NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja, NALU_HYPRE_Complex *d_a,
                                        cusparseMatDescr_t descr_B, NALU_HYPRE_Int nnzB, NALU_HYPRE_Int *d_ib, NALU_HYPRE_Int *d_jb, NALU_HYPRE_Complex *d_b,
                                        cusparseMatDescr_t descr_C, NALU_HYPRE_Int *nnzC_out, NALU_HYPRE_Int **d_ic_out, NALU_HYPRE_Int **d_jc_out,
                                        NALU_HYPRE_Complex **d_c_out);

void nalu_hypre_SortCSRCusparse( NALU_HYPRE_Int n, NALU_HYPRE_Int m, NALU_HYPRE_Int nnzA, cusparseMatDescr_t descrA,
                            const NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja_sorted, NALU_HYPRE_Complex *d_a_sorted );
#endif

#if defined(NALU_HYPRE_USING_ROCSPARSE)
NALU_HYPRE_Int hypreDevice_CSRSpGemmRocsparse(NALU_HYPRE_Int m, NALU_HYPRE_Int k, NALU_HYPRE_Int n,
                                         rocsparse_mat_descr descrA, NALU_HYPRE_Int nnzA, NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja, NALU_HYPRE_Complex *d_a,
                                         rocsparse_mat_descr descrB, NALU_HYPRE_Int nnzB, NALU_HYPRE_Int *d_ib, NALU_HYPRE_Int *d_jb, NALU_HYPRE_Complex *d_b,
                                         rocsparse_mat_descr descrC, rocsparse_mat_info infoC, NALU_HYPRE_Int *nnzC_out, NALU_HYPRE_Int **d_ic_out,
                                         NALU_HYPRE_Int **d_jc_out, NALU_HYPRE_Complex **d_c_out);

void nalu_hypre_SortCSRRocsparse( NALU_HYPRE_Int n, NALU_HYPRE_Int m, NALU_HYPRE_Int nnzA, rocsparse_mat_descr descrA,
                             const NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja_sorted, NALU_HYPRE_Complex *d_a_sorted );
#endif

#if defined(NALU_HYPRE_USING_ONEMKLSPARSE)
NALU_HYPRE_Int hypreDevice_CSRSpGemmOnemklsparse(NALU_HYPRE_Int m, NALU_HYPRE_Int k, NALU_HYPRE_Int n,
                                            oneapi::mkl::sparse::matrix_handle_t handle_A,
                                            NALU_HYPRE_Int nnzA, NALU_HYPRE_Int *d_ia, NALU_HYPRE_Int *d_ja, NALU_HYPRE_Complex *d_a,
                                            oneapi::mkl::sparse::matrix_handle_t handle_B,
                                            NALU_HYPRE_Int nnzB, NALU_HYPRE_Int *d_ib, NALU_HYPRE_Int *d_jb, NALU_HYPRE_Complex *d_b,
                                            oneapi::mkl::sparse::matrix_handle_t handle_C,
                                            NALU_HYPRE_Int *nnzC_out, NALU_HYPRE_Int **d_ic_out, NALU_HYPRE_Int **d_jc_out, NALU_HYPRE_Complex **d_c_out);
#endif

#ifdef __cplusplus
}
#endif

#endif

