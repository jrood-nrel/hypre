/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"
#include "_nalu_hypre_utilities.hpp"
#include "seq_mv.hpp"

#if defined(NALU_HYPRE_USING_CUSPARSE)
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
/*
 * @brief Creates a cuda csr descriptor for a raw CSR matrix
 * @param[in] n Number of rows
 * @param[in] m Number of columns
 * @param[in] offset the first row considered
 * @param[in] nnz Number of nonzeroes
 * @param[in] *i Row indices
 * @param[in] *j Colmn indices
 * @param[in] *data Values
 * @return Descriptor
 */
cusparseSpMatDescr_t
nalu_hypre_CSRMatrixToCusparseSpMat_core( NALU_HYPRE_Int      n,
                                     NALU_HYPRE_Int      m,
                                     NALU_HYPRE_Int      offset,
                                     NALU_HYPRE_Int      nnz,
                                     NALU_HYPRE_Int     *i,
                                     NALU_HYPRE_Int     *j,
                                     NALU_HYPRE_Complex *data)
{
   const cudaDataType        data_type  = nalu_hypre_HYPREComplexToCudaDataType();
   const cusparseIndexType_t index_type = nalu_hypre_HYPREIntToCusparseIndexType();
   const cusparseIndexBase_t index_base = CUSPARSE_INDEX_BASE_ZERO;

   cusparseSpMatDescr_t matA;

   /*
   nalu_hypre_assert( (nalu_hypre_CSRMatrixNumRows(A) - offset != 0) &&
                 (nalu_hypre_CSRMatrixNumCols(A) != 0) &&
                 (nalu_hypre_CSRMatrixNumNonzeros(A) != 0) &&
                 "Matrix has no nonzeros");
   */

   NALU_HYPRE_CUSPARSE_CALL( cusparseCreateCsr(&matA,
                                          n - offset,
                                          m,
                                          nnz,
                                          i + offset,
                                          j,
                                          data,
                                          index_type,
                                          index_type,
                                          index_base,
                                          data_type) );

   return matA;
}

/*
 * @brief Creates a cuSPARSE CSR descriptor from a nalu_hypre_CSRMatrix
 * @param[in] *A Pointer to nalu_hypre_CSRMatrix
 * @param[in] offset Row offset
 * @return cuSPARSE CSR Descriptor
 * @warning Assumes CSRMatrix has base 0
 */
cusparseSpMatDescr_t
nalu_hypre_CSRMatrixToCusparseSpMat(const nalu_hypre_CSRMatrix *A,
                               NALU_HYPRE_Int        offset)
{
   return nalu_hypre_CSRMatrixToCusparseSpMat_core( nalu_hypre_CSRMatrixNumRows(A),
                                               nalu_hypre_CSRMatrixNumCols(A),
                                               offset,
                                               nalu_hypre_CSRMatrixNumNonzeros(A),
                                               nalu_hypre_CSRMatrixI(A),
                                               nalu_hypre_CSRMatrixJ(A),
                                               nalu_hypre_CSRMatrixData(A) );
}

/*
 * @brief Creates a cuSPARSE dense vector descriptor from a nalu_hypre_Vector
 * @param[in] *x Pointer to a nalu_hypre_Vector
 * @param[in] offset Row offset
 * @return cuSPARSE dense vector descriptor
 * @warning Assumes CSRMatrix uses doubles for values
 */
cusparseDnVecDescr_t
nalu_hypre_VectorToCusparseDnVec_core(NALU_HYPRE_Complex *x_data,
                                 NALU_HYPRE_Int      n)
{
   const cudaDataType data_type = nalu_hypre_HYPREComplexToCudaDataType();

   cusparseDnVecDescr_t vecX;

   NALU_HYPRE_CUSPARSE_CALL( cusparseCreateDnVec(&vecX,
                                            n,
                                            x_data,
                                            data_type) );
   return vecX;
}

cusparseDnVecDescr_t
nalu_hypre_VectorToCusparseDnVec(const nalu_hypre_Vector *x,
                            NALU_HYPRE_Int           offset,
                            NALU_HYPRE_Int           size_override)
{
   return nalu_hypre_VectorToCusparseDnVec_core(nalu_hypre_VectorData(x) + offset,
                                           size_override >= 0 ? size_override : nalu_hypre_VectorSize(x) - offset);
}

/*
 * @brief Creates a cuSPARSE dense matrix descriptor from a nalu_hypre_Vector
 * @param[in] *x Pointer to a nalu_hypre_Vector
 * @return cuSPARSE dense matrix descriptor
 * @warning Assumes CSRMatrix uses doubles for values
 */
cusparseDnMatDescr_t
nalu_hypre_VectorToCusparseDnMat_core(NALU_HYPRE_Complex *x_data,
                                 NALU_HYPRE_Int      nrow,
                                 NALU_HYPRE_Int      ncol,
                                 NALU_HYPRE_Int      order)
{

   cudaDataType          data_type = nalu_hypre_HYPREComplexToCudaDataType();
   cusparseDnMatDescr_t  matX;

   NALU_HYPRE_CUSPARSE_CALL( cusparseCreateDnMat(&matX,
                                            nrow,
                                            ncol,
                                            (order == 0) ? nrow : ncol,
                                            x_data,
                                            data_type,
                                            (order == 0) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW) );
   return matX;
}

cusparseDnMatDescr_t
nalu_hypre_VectorToCusparseDnMat(const nalu_hypre_Vector *x)
{
   return nalu_hypre_VectorToCusparseDnMat_core(nalu_hypre_VectorData(x),
                                           nalu_hypre_VectorSize(x),
                                           nalu_hypre_VectorNumVectors(x),
                                           nalu_hypre_VectorMultiVecStorageMethod(x));
}

#endif // #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
#endif // #if defined(NALU_HYPRE_USING_CUSPARSE)
