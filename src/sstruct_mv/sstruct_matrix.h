/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the hypre_SStructMatrix structures
 *
 *****************************************************************************/

#ifndef hypre_SSTRUCT_MATRIX_HEADER
#define hypre_SSTRUCT_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * hypre_SStructMatrix:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;
   hypre_SStructPGrid     *pgrid;
   hypre_SStructStencil  **stencils;     /* nvar array of stencils */

   NALU_HYPRE_Int               nvars;
   NALU_HYPRE_Int             **smaps;
   hypre_StructStencil  ***sstencils;    /* nvar x nvar array of sstencils */
   hypre_StructMatrix   ***smatrices;    /* nvar x nvar array of smatrices */
   NALU_HYPRE_Int             **symmetric;    /* Stencil entries symmetric?
                                          * (nvar x nvar array) */

   /* temporary storage for SetValues routines */
   NALU_HYPRE_Int               sentries_size;
   NALU_HYPRE_Int              *sentries;

   NALU_HYPRE_Int               accumulated;  /* AddTo values accumulated? */

   NALU_HYPRE_Int               ref_count;

} hypre_SStructPMatrix;

typedef struct hypre_SStructMatrix_struct
{
   MPI_Comm                comm;
   NALU_HYPRE_Int               ndim;
   hypre_SStructGraph     *graph;
   NALU_HYPRE_Int            ***splits;   /* S/U-matrix split for each stencil */

   /* S-matrix info */
   NALU_HYPRE_Int               nparts;
   hypre_SStructPMatrix  **pmatrices;
   NALU_HYPRE_Int            ***symmetric;    /* Stencil entries symmetric?
                                          * (nparts x nvar x nvar array) */

   /* U-matrix info */
   NALU_HYPRE_IJMatrix          ijmatrix;
   hypre_ParCSRMatrix     *parcsrmatrix;

   /* temporary storage for SetValues routines */
   NALU_HYPRE_Int               entries_size;
   NALU_HYPRE_Int              *Sentries;
   NALU_HYPRE_Int              *Uentries;

   NALU_HYPRE_Int               tmp_size;     /* size of the following 3 */
   NALU_HYPRE_BigInt           *tmp_row_coords;
   NALU_HYPRE_BigInt           *tmp_col_coords;
   NALU_HYPRE_Complex          *tmp_coeffs;
   NALU_HYPRE_BigInt           *d_tmp_row_coords;
   NALU_HYPRE_BigInt           *d_tmp_col_coords;
   NALU_HYPRE_Complex          *d_tmp_coeffs;

   NALU_HYPRE_Int               ns_symmetric; /* Non-stencil entries symmetric? */
   NALU_HYPRE_Int               global_size;  /* Total number of nonzero coeffs */

   NALU_HYPRE_Int               ref_count;

   /* GEC0902   adding an object type to the matrix  */
   NALU_HYPRE_Int               object_type;

} hypre_SStructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructMatrix
 *--------------------------------------------------------------------------*/

#define hypre_SStructMatrixComm(mat)                 ((mat) -> comm)
#define hypre_SStructMatrixNDim(mat)                 ((mat) -> ndim)
#define hypre_SStructMatrixGraph(mat)                ((mat) -> graph)
#define hypre_SStructMatrixSplits(mat)               ((mat) -> splits)
#define hypre_SStructMatrixSplit(mat, p, v)          ((mat) -> splits[p][v])
#define hypre_SStructMatrixNParts(mat)               ((mat) -> nparts)
#define hypre_SStructMatrixPMatrices(mat)            ((mat) -> pmatrices)
#define hypre_SStructMatrixPMatrix(mat, part)        ((mat) -> pmatrices[part])
#define hypre_SStructMatrixSymmetric(mat)            ((mat) -> symmetric)
#define hypre_SStructMatrixIJMatrix(mat)             ((mat) -> ijmatrix)
#define hypre_SStructMatrixParCSRMatrix(mat)         ((mat) -> parcsrmatrix)
#define hypre_SStructMatrixEntriesSize(mat)          ((mat) -> entries_size)
#define hypre_SStructMatrixSEntries(mat)             ((mat) -> Sentries)
#define hypre_SStructMatrixUEntries(mat)             ((mat) -> Uentries)
#define hypre_SStructMatrixTmpSize(mat)              ((mat) -> tmp_size)
#define hypre_SStructMatrixTmpRowCoords(mat)         ((mat) -> tmp_row_coords)
#define hypre_SStructMatrixTmpColCoords(mat)         ((mat) -> tmp_col_coords)
#define hypre_SStructMatrixTmpCoeffs(mat)            ((mat) -> tmp_coeffs)
#define hypre_SStructMatrixTmpRowCoordsDevice(mat)   ((mat) -> d_tmp_row_coords)
#define hypre_SStructMatrixTmpColCoordsDevice(mat)   ((mat) -> d_tmp_col_coords)
#define hypre_SStructMatrixTmpCoeffsDevice(mat)      ((mat) -> d_tmp_coeffs)
#define hypre_SStructMatrixNSSymmetric(mat)          ((mat) -> ns_symmetric)
#define hypre_SStructMatrixGlobalSize(mat)           ((mat) -> global_size)
#define hypre_SStructMatrixRefCount(mat)             ((mat) -> ref_count)
#define hypre_SStructMatrixObjectType(mat)           ((mat) -> object_type)

/*--------------------------------------------------------------------------
 * Accessor macros: hypre_SStructPMatrix
 *--------------------------------------------------------------------------*/

#define hypre_SStructPMatrixComm(pmat)              ((pmat) -> comm)
#define hypre_SStructPMatrixPGrid(pmat)             ((pmat) -> pgrid)
#define hypre_SStructPMatrixNDim(pmat) \
hypre_SStructPGridNDim(hypre_SStructPMatrixPGrid(pmat))
#define hypre_SStructPMatrixStencils(pmat)          ((pmat) -> stencils)
#define hypre_SStructPMatrixNVars(pmat)             ((pmat) -> nvars)
#define hypre_SStructPMatrixStencil(pmat, var)      ((pmat) -> stencils[var])
#define hypre_SStructPMatrixSMaps(pmat)             ((pmat) -> smaps)
#define hypre_SStructPMatrixSMap(pmat, var)         ((pmat) -> smaps[var])
#define hypre_SStructPMatrixSStencils(pmat)         ((pmat) -> sstencils)
#define hypre_SStructPMatrixSStencil(pmat, vi, vj) \
((pmat) -> sstencils[vi][vj])
#define hypre_SStructPMatrixSMatrices(pmat)         ((pmat) -> smatrices)
#define hypre_SStructPMatrixSMatrix(pmat, vi, vj)  \
((pmat) -> smatrices[vi][vj])
#define hypre_SStructPMatrixSymmetric(pmat)         ((pmat) -> symmetric)
#define hypre_SStructPMatrixSEntriesSize(pmat)      ((pmat) -> sentries_size)
#define hypre_SStructPMatrixSEntries(pmat)          ((pmat) -> sentries)
#define hypre_SStructPMatrixAccumulated(pmat)       ((pmat) -> accumulated)
#define hypre_SStructPMatrixRefCount(pmat)          ((pmat) -> ref_count)

#endif
