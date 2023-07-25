/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the nalu_hypre_StructMatrix structures
 *
 *****************************************************************************/

#ifndef nalu_hypre_STRUCT_MATRIX_HEADER
#define nalu_hypre_STRUCT_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrix:
 *--------------------------------------------------------------------------*/

typedef struct nalu_hypre_StructMatrix_struct
{
   MPI_Comm              comm;

   nalu_hypre_StructGrid     *grid;
   nalu_hypre_StructStencil  *user_stencil;
   nalu_hypre_StructStencil  *stencil;
   NALU_HYPRE_Int             num_values;                /* Number of "stored" coefficients */

   nalu_hypre_BoxArray       *data_space;

   NALU_HYPRE_MemoryLocation  memory_location;           /* memory location of data */
   NALU_HYPRE_Complex        *data;                      /* Pointer to variable matrix data */
   NALU_HYPRE_Complex        *data_const;                /* Pointer to constant matrix data */
   NALU_HYPRE_Complex       **stencil_data;              /* Pointer for each stencil */
   NALU_HYPRE_Int             data_alloced;              /* Boolean used for freeing data */
   NALU_HYPRE_Int             data_size;                 /* Size of variable matrix data */
   NALU_HYPRE_Int             data_const_size;           /* Size of constant matrix data */
   NALU_HYPRE_Int           **data_indices;              /* num-boxes by stencil-size array
                                                       of indices into the data array.
                                                       data_indices[b][s] is the starting
                                                       index of matrix data corresponding
                                                       to box b and stencil coefficient s */
   NALU_HYPRE_Int             constant_coefficient;      /* normally 0; set to 1 for
                                                       constant coefficient matrices
                                                       or 2 for constant coefficient
                                                       with variable diagonal */

   NALU_HYPRE_Int             symmetric;                 /* Is the matrix symmetric */
   NALU_HYPRE_Int            *symm_elements;             /* Which elements are "symmetric" */
   NALU_HYPRE_Int             num_ghost[2 * NALU_HYPRE_MAXDIM]; /* Num ghost layers in each direction */

   NALU_HYPRE_BigInt          global_size;               /* Total number of nonzero coeffs */

   nalu_hypre_CommPkg        *comm_pkg;                  /* Info on how to update ghost data */

   NALU_HYPRE_Int             ref_count;

} nalu_hypre_StructMatrix;

/*--------------------------------------------------------------------------
 * Accessor macros: nalu_hypre_StructMatrix
 *--------------------------------------------------------------------------*/

#define nalu_hypre_StructMatrixComm(matrix)                ((matrix) -> comm)
#define nalu_hypre_StructMatrixGrid(matrix)                ((matrix) -> grid)
#define nalu_hypre_StructMatrixUserStencil(matrix)         ((matrix) -> user_stencil)
#define nalu_hypre_StructMatrixStencil(matrix)             ((matrix) -> stencil)
#define nalu_hypre_StructMatrixNumValues(matrix)           ((matrix) -> num_values)
#define nalu_hypre_StructMatrixDataSpace(matrix)           ((matrix) -> data_space)
#define nalu_hypre_StructMatrixMemoryLocation(matrix)      ((matrix) -> memory_location)
#define nalu_hypre_StructMatrixData(matrix)                ((matrix) -> data)
#define nalu_hypre_StructMatrixDataConst(matrix)           ((matrix) -> data_const)
#define nalu_hypre_StructMatrixStencilData(matrix)         ((matrix) -> stencil_data)
#define nalu_hypre_StructMatrixDataAlloced(matrix)         ((matrix) -> data_alloced)
#define nalu_hypre_StructMatrixDataSize(matrix)            ((matrix) -> data_size)
#define nalu_hypre_StructMatrixDataConstSize(matrix)       ((matrix) -> data_const_size)
#define nalu_hypre_StructMatrixDataIndices(matrix)         ((matrix) -> data_indices)
#define nalu_hypre_StructMatrixConstantCoefficient(matrix) ((matrix) -> constant_coefficient)
#define nalu_hypre_StructMatrixSymmetric(matrix)           ((matrix) -> symmetric)
#define nalu_hypre_StructMatrixSymmElements(matrix)        ((matrix) -> symm_elements)
#define nalu_hypre_StructMatrixNumGhost(matrix)            ((matrix) -> num_ghost)
#define nalu_hypre_StructMatrixGlobalSize(matrix)          ((matrix) -> global_size)
#define nalu_hypre_StructMatrixCommPkg(matrix)             ((matrix) -> comm_pkg)
#define nalu_hypre_StructMatrixRefCount(matrix)            ((matrix) -> ref_count)

#define nalu_hypre_StructMatrixNDim(matrix) \
nalu_hypre_StructGridNDim(nalu_hypre_StructMatrixGrid(matrix))

#define nalu_hypre_StructMatrixBox(matrix, b) \
nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(matrix), b)

#define nalu_hypre_StructMatrixBoxData(matrix, b, s) \
(nalu_hypre_StructMatrixStencilData(matrix)[s] + nalu_hypre_StructMatrixDataIndices(matrix)[b][s])

#define nalu_hypre_StructMatrixBoxDataValue(matrix, b, s, index) \
(nalu_hypre_StructMatrixBoxData(matrix, b, s) + \
 nalu_hypre_BoxIndexRank(nalu_hypre_StructMatrixBox(matrix, b), index))

#define nalu_hypre_CCStructMatrixBoxDataValue(matrix, b, s, index) \
(nalu_hypre_StructMatrixBoxData(matrix, b, s) + \
 nalu_hypre_CCBoxIndexRank(nalu_hypre_StructMatrixBox(matrix, b), index))

#endif
