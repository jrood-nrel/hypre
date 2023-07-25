/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_StructMatrix class.
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixCreateMask
 *    This routine returns the matrix, `mask', containing pointers to
 *    some of the data in the input matrix `matrix'.  This can be useful,
 *    for example, to construct "splittings" of a matrix for use in
 *    iterative methods.  The key note here is that the matrix `mask' does
 *    NOT contain a copy of the data in `matrix', but it can be used as
 *    if it were a normal StructMatrix object.
 *
 *    Notes:
 *    (1) Only the stencil, data_indices, and global_size components of the
 *        StructMatrix structure are modified.
 *    (2) PrintStructMatrix will not correctly print the stencil-to-data
 *        correspondence.
 *--------------------------------------------------------------------------*/

nalu_hypre_StructMatrix *
nalu_hypre_StructMatrixCreateMask( nalu_hypre_StructMatrix *matrix,
                              NALU_HYPRE_Int           num_stencil_indices,
                              NALU_HYPRE_Int          *stencil_indices     )
{
   NALU_HYPRE_Int             ndim = nalu_hypre_StructMatrixNDim(matrix);
   nalu_hypre_StructMatrix   *mask;

   nalu_hypre_StructStencil  *stencil;
   nalu_hypre_Index          *stencil_shape;
   NALU_HYPRE_Int             stencil_size;
   NALU_HYPRE_Complex       **stencil_data;
   nalu_hypre_Index          *mask_stencil_shape;
   NALU_HYPRE_Int             mask_stencil_size;
   NALU_HYPRE_Complex       **mask_stencil_data;

   nalu_hypre_BoxArray       *data_space;
   NALU_HYPRE_Int           **data_indices;
   NALU_HYPRE_Int           **mask_data_indices;

   NALU_HYPRE_Int             i, j;

   stencil       = nalu_hypre_StructMatrixStencil(matrix);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   stencil_size  = nalu_hypre_StructStencilSize(stencil);
   stencil_data  = nalu_hypre_StructMatrixStencilData(matrix);

   mask = nalu_hypre_CTAlloc(nalu_hypre_StructMatrix, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_StructMatrixComm(mask) = nalu_hypre_StructMatrixComm(matrix);

   nalu_hypre_StructGridRef(nalu_hypre_StructMatrixGrid(matrix),
                       &nalu_hypre_StructMatrixGrid(mask));

   nalu_hypre_StructMatrixUserStencil(mask) =
      nalu_hypre_StructStencilRef(nalu_hypre_StructMatrixUserStencil(matrix));

   mask_stencil_size  = num_stencil_indices;
   mask_stencil_shape = nalu_hypre_CTAlloc(nalu_hypre_Index, num_stencil_indices, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_stencil_indices; i++)
   {
      nalu_hypre_CopyIndex(stencil_shape[stencil_indices[i]],
                      mask_stencil_shape[i]);
   }
   nalu_hypre_StructMatrixStencil(mask) =
      nalu_hypre_StructStencilCreate(nalu_hypre_StructStencilNDim(stencil),
                                mask_stencil_size,
                                mask_stencil_shape);

   nalu_hypre_StructMatrixNumValues(mask) = nalu_hypre_StructMatrixNumValues(matrix);

   nalu_hypre_StructMatrixDataSpace(mask) =
      nalu_hypre_BoxArrayDuplicate(nalu_hypre_StructMatrixDataSpace(matrix));

   nalu_hypre_StructMatrixMemoryLocation(mask) = nalu_hypre_StructMatrixMemoryLocation(matrix);

   nalu_hypre_StructMatrixData(mask) = nalu_hypre_StructMatrixData(matrix);
   nalu_hypre_StructMatrixDataConst(mask) = nalu_hypre_StructMatrixDataConst(matrix);

   nalu_hypre_StructMatrixDataAlloced(mask) = 0;
   nalu_hypre_StructMatrixDataSize(mask) = nalu_hypre_StructMatrixDataSize(matrix);
   nalu_hypre_StructMatrixDataConstSize(mask) = nalu_hypre_StructMatrixDataConstSize(matrix);
   data_space   = nalu_hypre_StructMatrixDataSpace(matrix);
   data_indices = nalu_hypre_StructMatrixDataIndices(matrix);
   mask_data_indices = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(data_space), NALU_HYPRE_MEMORY_HOST);
   mask_stencil_data  = nalu_hypre_TAlloc(NALU_HYPRE_Complex*, mask_stencil_size, NALU_HYPRE_MEMORY_HOST);
   if (nalu_hypre_BoxArraySize(data_space) > 0)
   {
      mask_data_indices[0] = nalu_hypre_TAlloc(NALU_HYPRE_Int,
                                          num_stencil_indices * nalu_hypre_BoxArraySize(data_space),
                                          NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_ForBoxI(i, data_space)
   {
      mask_data_indices[i] = mask_data_indices[0] + num_stencil_indices * i;
      for (j = 0; j < num_stencil_indices; j++)
      {
         mask_data_indices[i][j] = data_indices[i][stencil_indices[j]];
      }
   }
   for (i = 0; i < mask_stencil_size; i++)
   {
      mask_stencil_data[i] = stencil_data[stencil_indices[i]];
   }
   nalu_hypre_StructMatrixStencilData(mask) = mask_stencil_data;

   nalu_hypre_StructMatrixDataIndices(mask) = mask_data_indices;

   nalu_hypre_StructMatrixSymmetric(mask) = nalu_hypre_StructMatrixSymmetric(matrix);

   nalu_hypre_StructMatrixSymmElements(mask) = nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_size, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < stencil_size; i++)
   {
      nalu_hypre_StructMatrixSymmElements(mask)[i] =
         nalu_hypre_StructMatrixSymmElements(matrix)[i];
   }

   for (i = 0; i < 2 * ndim; i++)
   {
      nalu_hypre_StructMatrixNumGhost(mask)[i] =
         nalu_hypre_StructMatrixNumGhost(matrix)[i];
   }

   nalu_hypre_StructMatrixGlobalSize(mask) =
      nalu_hypre_StructGridGlobalSize(nalu_hypre_StructMatrixGrid(mask)) *
      mask_stencil_size;

   nalu_hypre_StructMatrixCommPkg(mask) = NULL;

   nalu_hypre_StructMatrixRefCount(mask) = 1;

   return mask;
}

