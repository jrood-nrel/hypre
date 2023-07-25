/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Multiblock Matrix data structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_MULTIBLOCK_MATRIX_HEADER
#define nalu_hypre_MULTIBLOCK_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * Multiblock Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_Int             num_submatrices;
   NALU_HYPRE_Int            *submatrix_types;
   void                **submatrices;

} nalu_hypre_MultiblockMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the Multiblock Matrix structure
 *--------------------------------------------------------------------------*/

#define nalu_hypre_MultiblockMatrixSubmatrices(matrix)        ((matrix) -> submatrices)
#define nalu_hypre_MultiblockMatrixNumSubmatrices(matrix)     ((matrix) -> num_submatrices)
#define nalu_hypre_MultiblockMatrixSubmatrixTypes(matrix)     ((matrix) -> submatrix_types)

#define nalu_hypre_MultiblockMatrixSubmatrix(matrix,j) (nalu_hypre_MultiblockMatrixSubmatrices\
(matrix)[j])
#define nalu_hypre_MultiblockMatrixSubmatrixType(matrix,j) (nalu_hypre_MultiblockMatrixSubmatrixTypes\
(matrix)[j])

#endif

