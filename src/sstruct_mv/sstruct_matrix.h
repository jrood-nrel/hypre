/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the nalu_hypre_SStructMatrix structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_SSTRUCT_MATRIX_HEADER
#define nalu_hypre_SSTRUCT_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructMatrix:
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;
   nalu_hypre_SStructPGrid     *pgrid;
   nalu_hypre_SStructStencil  **stencils;     /* nvar array of stencils */

   NALU_HYPRE_Int               nvars;
   NALU_HYPRE_Int             **smaps;
   nalu_hypre_StructStencil  ***sstencils;    /* nvar x nvar array of sstencils */
   nalu_hypre_StructMatrix   ***smatrices;    /* nvar x nvar array of smatrices */
   NALU_HYPRE_Int             **symmetric;    /* Stencil entries symmetric?
                                          * (nvar x nvar array) */

   /* temporary storage for SetValues routines */
   NALU_HYPRE_Int               sentries_size;
   NALU_HYPRE_Int              *sentries;

   NALU_HYPRE_Int               accumulated;  /* AddTo values accumulated? */

   NALU_HYPRE_Int               ref_count;

} nalu_hypre_SStructPMatrix;

typedef struct nalu_hypre_SStructMatrix_struct
{
   MPI_Comm                comm;
   NALU_HYPRE_Int               ndim;
   nalu_hypre_SStructGraph     *graph;
   NALU_HYPRE_Int            ***splits;   /* S/U-matrix split for each stencil */

   /* S-matrix info */
   NALU_HYPRE_Int               nparts;
   nalu_hypre_SStructPMatrix  **pmatrices;
   NALU_HYPRE_Int            ***symmetric;    /* Stencil entries symmetric?
                                          * (nparts x nvar x nvar array) */

   /* U-matrix info */
   NALU_HYPRE_IJMatrix          ijmatrix;
   nalu_hypre_ParCSRMatrix     *parcsrmatrix;

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

} nalu_hypre_SStructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructMatrix
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructMatrixComm(mat)                 ((mat) -> comm)
#define nalu_hypre_SStructMatrixNDim(mat)                 ((mat) -> ndim)
#define nalu_hypre_SStructMatrixGraph(mat)                ((mat) -> graph)
#define nalu_hypre_SStructMatrixSplits(mat)               ((mat) -> splits)
#define nalu_hypre_SStructMatrixSplit(mat, p, v)          ((mat) -> splits[p][v])
#define nalu_hypre_SStructMatrixNParts(mat)               ((mat) -> nparts)
#define nalu_hypre_SStructMatrixPMatrices(mat)            ((mat) -> pmatrices)
#define nalu_hypre_SStructMatrixPMatrix(mat, part)        ((mat) -> pmatrices[part])
#define nalu_hypre_SStructMatrixSymmetric(mat)            ((mat) -> symmetric)
#define nalu_hypre_SStructMatrixIJMatrix(mat)             ((mat) -> ijmatrix)
#define nalu_hypre_SStructMatrixParCSRMatrix(mat)         ((mat) -> parcsrmatrix)
#define nalu_hypre_SStructMatrixEntriesSize(mat)          ((mat) -> entries_size)
#define nalu_hypre_SStructMatrixSEntries(mat)             ((mat) -> Sentries)
#define nalu_hypre_SStructMatrixUEntries(mat)             ((mat) -> Uentries)
#define nalu_hypre_SStructMatrixTmpSize(mat)              ((mat) -> tmp_size)
#define nalu_hypre_SStructMatrixTmpRowCoords(mat)         ((mat) -> tmp_row_coords)
#define nalu_hypre_SStructMatrixTmpColCoords(mat)         ((mat) -> tmp_col_coords)
#define nalu_hypre_SStructMatrixTmpCoeffs(mat)            ((mat) -> tmp_coeffs)
#define nalu_hypre_SStructMatrixTmpRowCoordsDevice(mat)   ((mat) -> d_tmp_row_coords)
#define nalu_hypre_SStructMatrixTmpColCoordsDevice(mat)   ((mat) -> d_tmp_col_coords)
#define nalu_hypre_SStructMatrixTmpCoeffsDevice(mat)      ((mat) -> d_tmp_coeffs)
#define nalu_hypre_SStructMatrixNSSymmetric(mat)          ((mat) -> ns_symmetric)
#define nalu_hypre_SStructMatrixGlobalSize(mat)           ((mat) -> global_size)
#define nalu_hypre_SStructMatrixRefCount(mat)             ((mat) -> ref_count)
#define nalu_hypre_SStructMatrixObjectType(mat)           ((mat) -> object_type)

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_SStructPMatrix
 *--------------------------------------------------------------------------*/

#define nalu_hypre_SStructPMatrixComm(pmat)              ((pmat) -> comm)
#define nalu_hypre_SStructPMatrixPGrid(pmat)             ((pmat) -> pgrid)
#define nalu_hypre_SStructPMatrixNDim(pmat) \
nalu_hypre_SStructPGridNDim(nalu_hypre_SStructPMatrixPGrid(pmat))
#define nalu_hypre_SStructPMatrixStencils(pmat)          ((pmat) -> stencils)
#define nalu_hypre_SStructPMatrixNVars(pmat)             ((pmat) -> nvars)
#define nalu_hypre_SStructPMatrixStencil(pmat, var)      ((pmat) -> stencils[var])
#define nalu_hypre_SStructPMatrixSMaps(pmat)             ((pmat) -> smaps)
#define nalu_hypre_SStructPMatrixSMap(pmat, var)         ((pmat) -> smaps[var])
#define nalu_hypre_SStructPMatrixSStencils(pmat)         ((pmat) -> sstencils)
#define nalu_hypre_SStructPMatrixSStencil(pmat, vi, vj) \
((pmat) -> sstencils[vi][vj])
#define nalu_hypre_SStructPMatrixSMatrices(pmat)         ((pmat) -> smatrices)
#define nalu_hypre_SStructPMatrixSMatrix(pmat, vi, vj)  \
((pmat) -> smatrices[vi][vj])
#define nalu_hypre_SStructPMatrixSymmetric(pmat)         ((pmat) -> symmetric)
#define nalu_hypre_SStructPMatrixSEntriesSize(pmat)      ((pmat) -> sentries_size)
#define nalu_hypre_SStructPMatrixSEntries(pmat)          ((pmat) -> sentries)
#define nalu_hypre_SStructPMatrixAccumulated(pmat)       ((pmat) -> accumulated)
#define nalu_hypre_SStructPMatrixRefCount(pmat)          ((pmat) -> ref_count)

#endif
