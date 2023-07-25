/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef nalu_hypre_CSR_MATRIX_HEADER
#define nalu_hypre_CSR_MATRIX_HEADER

#if defined(NALU_HYPRE_USING_CUSPARSE)  ||\
    defined(NALU_HYPRE_USING_ROCSPARSE) ||\
    defined(NALU_HYPRE_USING_ONEMKLSPARSE)
struct nalu_hypre_CsrsvData;
typedef struct nalu_hypre_CsrsvData nalu_hypre_CsrsvData;

struct nalu_hypre_GpuMatData;
typedef struct nalu_hypre_GpuMatData nalu_hypre_GpuMatData;
#endif

/*--------------------------------------------------------------------------
 * CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int            *i;
   NALU_HYPRE_Int            *j;
   NALU_HYPRE_BigInt         *big_j;
   NALU_HYPRE_Int             num_rows;
   NALU_HYPRE_Int             num_cols;
   NALU_HYPRE_Int             num_nonzeros;
   nalu_hypre_int            *i_short;
   nalu_hypre_int            *j_short;
   NALU_HYPRE_Int             owns_data;       /* Does the CSRMatrix create/destroy `data', `i', `j'? */
   NALU_HYPRE_Int             pattern_only;    /* 1: data array is ignored, and assumed to be all 1's */
   NALU_HYPRE_Complex        *data;
   NALU_HYPRE_Int            *rownnz;          /* for compressing rows in matrix multiplication  */
   NALU_HYPRE_Int             num_rownnz;
   NALU_HYPRE_MemoryLocation  memory_location; /* memory location of arrays i, j, data */

#if defined(NALU_HYPRE_USING_CUSPARSE)  ||\
    defined(NALU_HYPRE_USING_ROCSPARSE) ||\
    defined(NALU_HYPRE_USING_ONEMKLSPARSE)
   NALU_HYPRE_Int            *sorted_j;        /* some cusparse routines require sorted CSR */
   NALU_HYPRE_Complex        *sorted_data;
   nalu_hypre_CsrsvData      *csrsv_data;
   nalu_hypre_GpuMatData     *mat_data;
#endif
} nalu_hypre_CSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_CSRMatrixData(matrix)                 ((matrix) -> data)
#define nalu_hypre_CSRMatrixI(matrix)                    ((matrix) -> i)
#define nalu_hypre_CSRMatrixJ(matrix)                    ((matrix) -> j)
#define nalu_hypre_CSRMatrixBigJ(matrix)                 ((matrix) -> big_j)
#define nalu_hypre_CSRMatrixNumRows(matrix)              ((matrix) -> num_rows)
#define nalu_hypre_CSRMatrixNumCols(matrix)              ((matrix) -> num_cols)
#define nalu_hypre_CSRMatrixNumNonzeros(matrix)          ((matrix) -> num_nonzeros)
#define nalu_hypre_CSRMatrixRownnz(matrix)               ((matrix) -> rownnz)
#define nalu_hypre_CSRMatrixNumRownnz(matrix)            ((matrix) -> num_rownnz)
#define nalu_hypre_CSRMatrixOwnsData(matrix)             ((matrix) -> owns_data)
#define nalu_hypre_CSRMatrixPatternOnly(matrix)          ((matrix) -> pattern_only)
#define nalu_hypre_CSRMatrixMemoryLocation(matrix)       ((matrix) -> memory_location)

#if defined(NALU_HYPRE_USING_CUSPARSE)  ||\
    defined(NALU_HYPRE_USING_ROCSPARSE) ||\
    defined(NALU_HYPRE_USING_ONEMKLSPARSE)
#define nalu_hypre_CSRMatrixSortedJ(matrix)              ((matrix) -> sorted_j)
#define nalu_hypre_CSRMatrixSortedData(matrix)           ((matrix) -> sorted_data)
#define nalu_hypre_CSRMatrixCsrsvData(matrix)            ((matrix) -> csrsv_data)
#define nalu_hypre_CSRMatrixGPUMatData(matrix)           ((matrix) -> mat_data)
#endif

NALU_HYPRE_Int nalu_hypre_CSRMatrixGetLoadBalancedPartitionBegin( nalu_hypre_CSRMatrix *A );
NALU_HYPRE_Int nalu_hypre_CSRMatrixGetLoadBalancedPartitionEnd( nalu_hypre_CSRMatrix *A );

/*--------------------------------------------------------------------------
 * CSR Boolean Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int    *i;
   NALU_HYPRE_Int    *j;
   NALU_HYPRE_BigInt *big_j;
   NALU_HYPRE_Int     num_rows;
   NALU_HYPRE_Int     num_cols;
   NALU_HYPRE_Int     num_nonzeros;
   NALU_HYPRE_Int     owns_data;

} nalu_hypre_CSRBooleanMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Boolean Matrix structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_CSRBooleanMatrix_Get_I(matrix)        ((matrix)->i)
#define nalu_hypre_CSRBooleanMatrix_Get_J(matrix)        ((matrix)->j)
#define nalu_hypre_CSRBooleanMatrix_Get_BigJ(matrix)     ((matrix)->big_j)
#define nalu_hypre_CSRBooleanMatrix_Get_NRows(matrix)    ((matrix)->num_rows)
#define nalu_hypre_CSRBooleanMatrix_Get_NCols(matrix)    ((matrix)->num_cols)
#define nalu_hypre_CSRBooleanMatrix_Get_NNZ(matrix)      ((matrix)->num_nonzeros)
#define nalu_hypre_CSRBooleanMatrix_Get_OwnsData(matrix) ((matrix)->owns_data)

#endif
