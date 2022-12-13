/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
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

#ifndef hypre_CSR_MATRIX_HEADER
#define hypre_CSR_MATRIX_HEADER

#if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) || defined(NALU_HYPRE_USING_ONEMKLSPARSE)
struct hypre_CsrsvData;
typedef struct hypre_CsrsvData hypre_CsrsvData;
#endif

#if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) || defined(NALU_HYPRE_USING_ONEMKLSPARSE)
struct hypre_GpuMatData;
typedef struct hypre_GpuMatData hypre_GpuMatData;
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
   hypre_int            *i_short;
   hypre_int            *j_short;
   NALU_HYPRE_Int             owns_data;       /* Does the CSRMatrix create/destroy `data', `i', `j'? */
   NALU_HYPRE_Int             pattern_only;    /* if 1, data array is ignored, and assumed to be all 1's */
   NALU_HYPRE_Complex        *data;
   NALU_HYPRE_Int            *rownnz;          /* for compressing rows in matrix multiplication  */
   NALU_HYPRE_Int             num_rownnz;
   NALU_HYPRE_MemoryLocation  memory_location; /* memory location of arrays i, j, data */
#if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) || defined(NALU_HYPRE_USING_ONEMKLSPARSE)
   NALU_HYPRE_Int            *sorted_j;        /* some cusparse routines require sorted CSR */
   NALU_HYPRE_Complex        *sorted_data;
   hypre_CsrsvData      *csrsv_data;
   hypre_GpuMatData     *mat_data;
#endif
} hypre_CSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRMatrixData(matrix)                 ((matrix) -> data)
#define hypre_CSRMatrixI(matrix)                    ((matrix) -> i)
#define hypre_CSRMatrixJ(matrix)                    ((matrix) -> j)
#define hypre_CSRMatrixBigJ(matrix)                 ((matrix) -> big_j)
#define hypre_CSRMatrixNumRows(matrix)              ((matrix) -> num_rows)
#define hypre_CSRMatrixNumCols(matrix)              ((matrix) -> num_cols)
#define hypre_CSRMatrixNumNonzeros(matrix)          ((matrix) -> num_nonzeros)
#define hypre_CSRMatrixRownnz(matrix)               ((matrix) -> rownnz)
#define hypre_CSRMatrixNumRownnz(matrix)            ((matrix) -> num_rownnz)
#define hypre_CSRMatrixOwnsData(matrix)             ((matrix) -> owns_data)
#define hypre_CSRMatrixPatternOnly(matrix)          ((matrix) -> pattern_only)
#define hypre_CSRMatrixMemoryLocation(matrix)       ((matrix) -> memory_location)

#if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) || defined(NALU_HYPRE_USING_ONEMKLSPARSE)
#define hypre_CSRMatrixSortedJ(matrix)              ((matrix) -> sorted_j)
#define hypre_CSRMatrixSortedData(matrix)           ((matrix) -> sorted_data)
#define hypre_CSRMatrixCsrsvData(matrix)            ((matrix) -> csrsv_data)
#define hypre_CSRMatrixGPUMatData(matrix)           ((matrix) -> mat_data)
#endif

NALU_HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionBegin( hypre_CSRMatrix *A );
NALU_HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionEnd( hypre_CSRMatrix *A );

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

} hypre_CSRBooleanMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Boolean Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRBooleanMatrix_Get_I(matrix)        ((matrix)->i)
#define hypre_CSRBooleanMatrix_Get_J(matrix)        ((matrix)->j)
#define hypre_CSRBooleanMatrix_Get_BigJ(matrix)     ((matrix)->big_j)
#define hypre_CSRBooleanMatrix_Get_NRows(matrix)    ((matrix)->num_rows)
#define hypre_CSRBooleanMatrix_Get_NCols(matrix)    ((matrix)->num_cols)
#define hypre_CSRBooleanMatrix_Get_NNZ(matrix)      ((matrix)->num_nonzeros)
#define hypre_CSRBooleanMatrix_Get_OwnsData(matrix) ((matrix)->owns_data)

#endif

