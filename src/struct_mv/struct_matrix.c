/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_StructMatrix class.
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"
#include "_nalu_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixExtractPointerByIndex
 *    Returns pointer to data for stencil entry coresponding to
 *    `index' in `matrix'. If the index does not exist in the matrix's
 *    stencil, the NULL pointer is returned.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Complex *
nalu_hypre_StructMatrixExtractPointerByIndex( nalu_hypre_StructMatrix *matrix,
                                         NALU_HYPRE_Int           b,
                                         nalu_hypre_Index         index  )
{
   nalu_hypre_StructStencil   *stencil;
   NALU_HYPRE_Int              rank;

   stencil = nalu_hypre_StructMatrixStencil(matrix);
   rank = nalu_hypre_StructStencilElementRank( stencil, index );

   if ( rank >= 0 )
   {
      return nalu_hypre_StructMatrixBoxData(matrix, b, rank);
   }
   else
   {
      return NULL;  /* error - invalid index */
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_StructMatrix *
nalu_hypre_StructMatrixCreate( MPI_Comm             comm,
                          nalu_hypre_StructGrid    *grid,
                          nalu_hypre_StructStencil *user_stencil )
{
   NALU_HYPRE_Int            ndim = nalu_hypre_StructGridNDim(grid);
   nalu_hypre_StructMatrix  *matrix;
   NALU_HYPRE_Int            i;

   matrix = nalu_hypre_CTAlloc(nalu_hypre_StructMatrix, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_StructMatrixComm(matrix)        = comm;
   nalu_hypre_StructGridRef(grid, &nalu_hypre_StructMatrixGrid(matrix));
   nalu_hypre_StructMatrixUserStencil(matrix) = nalu_hypre_StructStencilRef(user_stencil);
   nalu_hypre_StructMatrixDataAlloced(matrix) = 1;
   nalu_hypre_StructMatrixRefCount(matrix)    = 1;

   /* set defaults */
   nalu_hypre_StructMatrixSymmetric(matrix) = 0;
   nalu_hypre_StructMatrixConstantCoefficient(matrix) = 0;
   for (i = 0; i < 2 * ndim; i++)
   {
      nalu_hypre_StructMatrixNumGhost(matrix)[i] = nalu_hypre_StructGridNumGhost(grid)[i];
   }

   nalu_hypre_StructMatrixMemoryLocation(matrix) = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixRef
 *--------------------------------------------------------------------------*/

nalu_hypre_StructMatrix *
nalu_hypre_StructMatrixRef( nalu_hypre_StructMatrix *matrix )
{
   nalu_hypre_StructMatrixRefCount(matrix) ++;

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixDestroy( nalu_hypre_StructMatrix *matrix )
{
   if (matrix)
   {
      nalu_hypre_StructMatrixRefCount(matrix) --;
      if (nalu_hypre_StructMatrixRefCount(matrix) == 0)
      {
         if (nalu_hypre_StructMatrixDataAlloced(matrix))
         {
            nalu_hypre_TFree(nalu_hypre_StructMatrixData(matrix), nalu_hypre_StructMatrixMemoryLocation(matrix));
            nalu_hypre_TFree(nalu_hypre_StructMatrixDataConst(matrix), NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(nalu_hypre_StructMatrixStencilData(matrix), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_CommPkgDestroy(nalu_hypre_StructMatrixCommPkg(matrix));
         if (nalu_hypre_BoxArraySize(nalu_hypre_StructMatrixDataSpace(matrix)) > 0)
         {
            nalu_hypre_TFree(nalu_hypre_StructMatrixDataIndices(matrix)[0], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(nalu_hypre_StructMatrixDataIndices(matrix), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_BoxArrayDestroy(nalu_hypre_StructMatrixDataSpace(matrix));
         nalu_hypre_TFree(nalu_hypre_StructMatrixSymmElements(matrix), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_StructStencilDestroy(nalu_hypre_StructMatrixUserStencil(matrix));
         nalu_hypre_StructStencilDestroy(nalu_hypre_StructMatrixStencil(matrix));
         nalu_hypre_StructGridDestroy(nalu_hypre_StructMatrixGrid(matrix));
         nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixInitializeShell
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixInitializeShell( nalu_hypre_StructMatrix *matrix )
{
   NALU_HYPRE_Int             ndim = nalu_hypre_StructMatrixNDim(matrix);
   nalu_hypre_StructGrid     *grid = nalu_hypre_StructMatrixGrid(matrix);

   nalu_hypre_StructStencil  *user_stencil;
   nalu_hypre_StructStencil  *stencil;
   nalu_hypre_Index          *stencil_shape;
   NALU_HYPRE_Int             stencil_size;
   NALU_HYPRE_Complex       **stencil_data;
   NALU_HYPRE_Int             num_values;
   NALU_HYPRE_Int            *symm_elements;
   NALU_HYPRE_Int             constant_coefficient;

   NALU_HYPRE_Int            *num_ghost;
   NALU_HYPRE_Int             extra_ghost[2 * NALU_HYPRE_MAXDIM];

   nalu_hypre_BoxArray       *data_space;
   nalu_hypre_BoxArray       *boxes;
   nalu_hypre_Box            *box;
   nalu_hypre_Box            *data_box;

   NALU_HYPRE_Int           **data_indices;
   NALU_HYPRE_Int             data_size;
   NALU_HYPRE_Int             data_const_size;
   NALU_HYPRE_Int             data_box_volume;

   NALU_HYPRE_Int             i, j, d;

   /*-----------------------------------------------------------------------
    * Set up stencil and num_values:
    *
    * If the matrix is symmetric, then the stencil is a "symmetrized"
    * version of the user's stencil.  If the matrix is not symmetric,
    * then the stencil is the same as the user's stencil.
    *
    * The `symm_elements' array is used to determine what data is
    * explicitely stored (symm_elements[i] < 0) and what data does is
    * not explicitely stored (symm_elements[i] >= 0), but is instead
    * stored as the transpose coefficient at a neighboring grid point.
    *-----------------------------------------------------------------------*/

   if (nalu_hypre_StructMatrixStencil(matrix) == NULL)
   {
      user_stencil = nalu_hypre_StructMatrixUserStencil(matrix);

      if (nalu_hypre_StructMatrixSymmetric(matrix))
      {
         /* store only symmetric stencil entry data */
         nalu_hypre_StructStencilSymmetrize(user_stencil, &stencil, &symm_elements);
         num_values = ( nalu_hypre_StructStencilSize(stencil) + 1 ) / 2;
      }
      else
      {
         /* store all stencil entry data */
         stencil = nalu_hypre_StructStencilRef(user_stencil);
         num_values = nalu_hypre_StructStencilSize(stencil);
         symm_elements = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_values, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_values; i++)
         {
            symm_elements[i] = -1;
         }
      }

      nalu_hypre_StructMatrixStencil(matrix)      = stencil;
      nalu_hypre_StructMatrixSymmElements(matrix) = symm_elements;
      nalu_hypre_StructMatrixNumValues(matrix)    = num_values;
   }

   /*-----------------------------------------------------------------------
    * Set ghost-layer size for symmetric storage
    *   - All stencil coeffs are to be available at each point in the
    *     grid, as well as in the user-specified ghost layer.
    *-----------------------------------------------------------------------*/

   num_ghost     = nalu_hypre_StructMatrixNumGhost(matrix);
   stencil       = nalu_hypre_StructMatrixStencil(matrix);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   stencil_size  = nalu_hypre_StructStencilSize(stencil);
   symm_elements = nalu_hypre_StructMatrixSymmElements(matrix);

   stencil_data  = nalu_hypre_TAlloc(NALU_HYPRE_Complex*, stencil_size, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_StructMatrixStencilData(matrix) = stencil_data;

   for (d = 0; d < 2 * ndim; d++)
   {
      extra_ghost[d] = 0;
   }

   for (i = 0; i < stencil_size; i++)
   {
      if (symm_elements[i] >= 0)
      {
         for (d = 0; d < ndim; d++)
         {
            extra_ghost[2 * d]     = nalu_hypre_max( extra_ghost[2 * d],
                                                -nalu_hypre_IndexD(stencil_shape[i], d) );
            extra_ghost[2 * d + 1] = nalu_hypre_max( extra_ghost[2 * d + 1],
                                                nalu_hypre_IndexD(stencil_shape[i], d) );
         }
      }
   }

   for (d = 0; d < ndim; d++)
   {
      num_ghost[2 * d]     += extra_ghost[2 * d];
      num_ghost[2 * d + 1] += extra_ghost[2 * d + 1];
   }

   /*-----------------------------------------------------------------------
    * Set up data_space
    *-----------------------------------------------------------------------*/

   if (nalu_hypre_StructMatrixDataSpace(matrix) == NULL)
   {
      boxes = nalu_hypre_StructGridBoxes(grid);
      data_space = nalu_hypre_BoxArrayCreate(nalu_hypre_BoxArraySize(boxes), ndim);

      nalu_hypre_ForBoxI(i, boxes)
      {
         box = nalu_hypre_BoxArrayBox(boxes, i);
         data_box = nalu_hypre_BoxArrayBox(data_space, i);

         nalu_hypre_CopyBox(box, data_box);
         for (d = 0; d < ndim; d++)
         {
            nalu_hypre_BoxIMinD(data_box, d) -= num_ghost[2 * d];
            nalu_hypre_BoxIMaxD(data_box, d) += num_ghost[2 * d + 1];
         }
      }

      nalu_hypre_StructMatrixDataSpace(matrix) = data_space;
   }

   /*-----------------------------------------------------------------------
    * Set up data_indices array and data-size
    *-----------------------------------------------------------------------*/

   if (nalu_hypre_StructMatrixDataIndices(matrix) == NULL)
   {
      data_space = nalu_hypre_StructMatrixDataSpace(matrix);
      data_indices = nalu_hypre_TAlloc(NALU_HYPRE_Int *, nalu_hypre_BoxArraySize(data_space),
                                  NALU_HYPRE_MEMORY_HOST);
      if (nalu_hypre_BoxArraySize(data_space) > 0)
      {
         data_indices[0] = nalu_hypre_TAlloc(NALU_HYPRE_Int, stencil_size * nalu_hypre_BoxArraySize(data_space),
                                        NALU_HYPRE_MEMORY_HOST);
      }
      constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(matrix);

      data_size = 0;
      data_const_size = 0;
      if ( constant_coefficient == 0 )
      {
         nalu_hypre_ForBoxI(i, data_space)
         {
            data_box = nalu_hypre_BoxArrayBox(data_space, i);
            data_box_volume  = nalu_hypre_BoxVolume(data_box);

            data_indices[i] = data_indices[0] + stencil_size * i;

            /* set pointers for "stored" coefficients */
            for (j = 0; j < stencil_size; j++)
            {
               if (symm_elements[j] < 0)
               {
                  data_indices[i][j] = data_size;
                  data_size += data_box_volume;
               }
            }

            /* set pointers for "symmetric" coefficients */
            for (j = 0; j < stencil_size; j++)
            {
               if (symm_elements[j] >= 0)
               {
                  data_indices[i][j] = data_indices[i][symm_elements[j]] +
                                       nalu_hypre_BoxOffsetDistance(data_box, stencil_shape[j]);
               }
            }
         }
      }
      else if ( constant_coefficient == 1 )
      {
         nalu_hypre_ForBoxI(i, data_space)
         {
            data_box = nalu_hypre_BoxArrayBox(data_space, i);
            data_box_volume  = nalu_hypre_BoxVolume(data_box);

            data_indices[i] = data_indices[0] + stencil_size * i;
            /* set pointers for "stored" coefficients */
            for (j = 0; j < stencil_size; j++)
            {
               if (symm_elements[j] < 0)
               {
                  data_indices[i][j] = data_const_size;
                  ++data_const_size;
               }
            }

            /* set pointers for "symmetric" coefficients */
            for (j = 0; j < stencil_size; j++)
            {
               if (symm_elements[j] >= 0)
               {
                  data_indices[i][j] = data_indices[i][symm_elements[j]];
               }
            }
         }
      }
      else
      {
         nalu_hypre_assert( constant_coefficient == 2 );
         data_const_size += stencil_size;
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if (nalu_hypre_StructGridDataLocation(grid) == NALU_HYPRE_MEMORY_HOST)
         {
            /* in this case, "data" is put on host using the space of
             * "data_const". so, "data" need to be shifted by the size of
             * const coeff */
            data_size += stencil_size;/* all constant coeffs at the beginning */
         }
#endif
         /* ... this allocates a little more space than is absolutely necessary */
         nalu_hypre_ForBoxI(i, data_space)
         {
            data_box = nalu_hypre_BoxArrayBox(data_space, i);
            data_box_volume  = nalu_hypre_BoxVolume(data_box);

            data_indices[i] = data_indices[0] + stencil_size * i;
            /* set pointers for "stored" coefficients */
            for (j = 0; j < stencil_size; j++)
            {
               if (symm_elements[j] < 0)
               {
                  /* diagonal, variable coefficient */
                  if (nalu_hypre_IndexEqual(stencil_shape[j], 0, ndim))
                  {
                     data_indices[i][j] = data_size;
                     data_size += data_box_volume;
                  }
                  /* off-diagonal, constant coefficient */
                  else
                  {
                     data_indices[i][j] = j;
                  }
               }
            }

            /* set pointers for "symmetric" coefficients */
            for (j = 0; j < stencil_size; j++)
            {
               if (symm_elements[j] >= 0)
               {
                  /* diagonal, variable coefficient */
                  if (nalu_hypre_IndexEqual(stencil_shape[j], 0, ndim))
                  {
                     data_indices[i][j] = data_indices[i][symm_elements[j]] +
                                          nalu_hypre_BoxOffsetDistance(data_box, stencil_shape[j]);
                  }
                  /* off-diagonal, constant coefficient */
                  else
                  {
                     data_indices[i][j] = data_indices[i][symm_elements[j]];
                  }
               }
            }
         }
      }

      nalu_hypre_StructMatrixDataIndices(matrix) = data_indices;

      /*-----------------------------------------------------------------------
       * if data location has not been set outside, set up the data location
       * based on the total number of
       *-----------------------------------------------------------------------*/
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      if (nalu_hypre_StructGridDataLocation(grid) == NALU_HYPRE_MEMORY_HOST)
      {
         data_const_size = data_size + data_const_size;
         data_size       = 0;
      }
#endif
      nalu_hypre_StructMatrixDataSize(matrix)      = data_size;
      nalu_hypre_StructMatrixDataConstSize(matrix) = data_const_size;

      /*
      if (nalu_hypre_BoxArraySize(data_space) > 0)
      {
      nalu_hypre_StructMatrixDataDeviceIndices(matrix) = data_indices[0];
      }
      */
   }

   /*-----------------------------------------------------------------------
    * Set total number of nonzero coefficients
    * For constant coefficients, this is unrelated to the amount of data
    * actually stored.
    *-----------------------------------------------------------------------*/

   nalu_hypre_StructMatrixGlobalSize(matrix) = nalu_hypre_StructGridGlobalSize(grid) * stencil_size;

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixInitializeData
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixInitializeData( nalu_hypre_StructMatrix *matrix,
                                  NALU_HYPRE_Complex      *data,
                                  NALU_HYPRE_Complex      *data_const)
{
   NALU_HYPRE_Int             ndim = nalu_hypre_StructMatrixNDim(matrix);
   NALU_HYPRE_Int constant_coefficient;
   nalu_hypre_StructStencil  *stencil;
   nalu_hypre_Index          *stencil_shape;
   NALU_HYPRE_Complex       **stencil_data;
   NALU_HYPRE_Int stencil_size, i;
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_StructGrid     *grid = nalu_hypre_StructMatrixGrid(matrix);
#endif
   nalu_hypre_StructMatrixData(matrix) = data;
   nalu_hypre_StructMatrixDataConst(matrix) = data_const;
   nalu_hypre_StructMatrixDataAlloced(matrix) = 0;

   stencil       = nalu_hypre_StructMatrixStencil(matrix);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   stencil_size  = nalu_hypre_StructStencilSize(stencil);
   stencil_data  = nalu_hypre_StructMatrixStencilData(matrix);

   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(matrix);

   if (constant_coefficient == 0)
   {
      for (i = 0; i < stencil_size; i++)
      {
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if (nalu_hypre_StructGridDataLocation(grid) != NALU_HYPRE_MEMORY_HOST)
         {
            stencil_data[i] = nalu_hypre_StructMatrixData(matrix);
         }
         else
         {
            stencil_data[i] = nalu_hypre_StructMatrixDataConst(matrix);
         }
#else
         stencil_data[i] = nalu_hypre_StructMatrixData(matrix);
#endif
      }
   }
   else if (constant_coefficient == 1)
   {
      for (i = 0; i < stencil_size; i++)
      {
         stencil_data[i] = nalu_hypre_StructMatrixDataConst(matrix);
      }
   }
   else
   {
      for (i = 0; i < stencil_size; i++)
      {
         /* diagonal, variable coefficient */
         if (nalu_hypre_IndexEqual(stencil_shape[i], 0, ndim))
         {
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
            if (nalu_hypre_StructGridDataLocation(grid) != NALU_HYPRE_MEMORY_HOST)
            {
               stencil_data[i] = nalu_hypre_StructMatrixData(matrix);
            }
            else
            {
               stencil_data[i] = nalu_hypre_StructMatrixDataConst(matrix);
            }
#else
            stencil_data[i] = nalu_hypre_StructMatrixData(matrix);
#endif
         }
         /* off-diagonal, constant coefficient */
         else
         {
            stencil_data[i] = nalu_hypre_StructMatrixDataConst(matrix);
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixInitialize
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_StructMatrixInitialize( nalu_hypre_StructMatrix *matrix )
{
   NALU_HYPRE_Complex *data;
   NALU_HYPRE_Complex *data_const;

   nalu_hypre_StructMatrixInitializeShell(matrix);

   data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nalu_hypre_StructMatrixDataSize(matrix),
                        nalu_hypre_StructMatrixMemoryLocation(matrix));
   data_const = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nalu_hypre_StructMatrixDataConstSize(matrix),
                              NALU_HYPRE_MEMORY_HOST);


   nalu_hypre_StructMatrixInitializeData(matrix, data, data_const);
   nalu_hypre_StructMatrixDataAlloced(matrix) = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *
 * should not be called to set a constant-coefficient part of the matrix,
 *   call nalu_hypre_StructMatrixSetConstantValues instead
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixSetValues( nalu_hypre_StructMatrix *matrix,
                             nalu_hypre_Index         grid_index,
                             NALU_HYPRE_Int           num_stencil_indices,
                             NALU_HYPRE_Int          *stencil_indices,
                             NALU_HYPRE_Complex      *values,
                             NALU_HYPRE_Int           action,
                             NALU_HYPRE_Int           boxnum,
                             NALU_HYPRE_Int           outside )
{
   nalu_hypre_BoxArray      *grid_boxes;
   nalu_hypre_Box           *grid_box;
   nalu_hypre_Index          center_index;
   nalu_hypre_StructStencil *stencil;
   NALU_HYPRE_Int            center_rank;
   NALU_HYPRE_Int           *symm_elements;
   NALU_HYPRE_Int            constant_coefficient;
   NALU_HYPRE_Complex       *matp;
   NALU_HYPRE_Int            i, s, istart, istop;
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_StructMatrixMemoryLocation(matrix);
#endif

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(matrix);
   symm_elements        = nalu_hypre_StructMatrixSymmElements(matrix);

   if (outside > 0)
   {
      grid_boxes = nalu_hypre_StructMatrixDataSpace(matrix);
   }
   else
   {
      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(matrix));
   }

   if (boxnum < 0)
   {
      istart = 0;
      istop  = nalu_hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   center_rank = 0;
   if ( constant_coefficient == 2 )
   {
      nalu_hypre_SetIndex(center_index, 0);
      stencil = nalu_hypre_StructMatrixStencil(matrix);
      center_rank = nalu_hypre_StructStencilElementRank( stencil, center_index );
   }

   for (i = istart; i < istop; i++)
   {
      grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);

      if (nalu_hypre_IndexInBox(grid_index, grid_box))
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            /* only set stored stencil values */
            if (symm_elements[stencil_indices[s]] < 0)
            {
               if ( (constant_coefficient == 1) ||
                    (constant_coefficient == 2 && stencil_indices[s] != center_rank) )
               {
                  /* call SetConstantValues instead */
                  nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
                  matp = nalu_hypre_StructMatrixBoxData(matrix, i, stencil_indices[s]);
               }
               else /* variable coefficient, constant_coefficient=0 */
               {
                  matp = nalu_hypre_StructMatrixBoxDataValue(matrix, i, stencil_indices[s], grid_index);
               }

#if defined(NALU_HYPRE_USING_GPU)
               if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
               {
                  if (action > 0)
                  {
#define DEVICE_VAR is_device_ptr(matp,values)
                     nalu_hypre_LoopBegin(1, k)
                     {
                        *matp += values[s];
                     }
                     nalu_hypre_LoopEnd()
#undef DEVICE_VAR
                  }
                  else if (action > -1)
                  {
                     nalu_hypre_TMemcpy(matp, values + s, NALU_HYPRE_Complex, 1, memory_location, memory_location);
                  }
                  else /* action < 0 */
                  {
                     nalu_hypre_TMemcpy(values + s, matp, NALU_HYPRE_Complex, 1, memory_location, memory_location);
                  }
               }
               else
#endif
               {
                  if (action > 0)
                  {
                     *matp += values[s];
                  }
                  else if (action > -1)
                  {
                     *matp = values[s];
                  }
                  else /* action < 0 */
                  {
                     values[s] = *matp;
                  }
               }
            }
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out
 *
 * should not be called to set a constant-coefficient part of the matrix,
 *   call nalu_hypre_StructMatrixSetConstantValues instead
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixSetBoxValues( nalu_hypre_StructMatrix *matrix,
                                nalu_hypre_Box          *set_box,
                                nalu_hypre_Box          *value_box,
                                NALU_HYPRE_Int           num_stencil_indices,
                                NALU_HYPRE_Int          *stencil_indices,
                                NALU_HYPRE_Complex      *values,
                                NALU_HYPRE_Int           action,
                                NALU_HYPRE_Int           boxnum,
                                NALU_HYPRE_Int           outside )
{
   nalu_hypre_BoxArray      *grid_boxes;
   nalu_hypre_Box           *grid_box;
   nalu_hypre_Box           *int_box;
   nalu_hypre_Index          center_index;
   nalu_hypre_StructStencil *stencil;
   NALU_HYPRE_Int            center_rank;

   NALU_HYPRE_Int           *symm_elements;
   nalu_hypre_BoxArray      *data_space;
   nalu_hypre_Box           *data_box;
   nalu_hypre_IndexRef       data_start;
   nalu_hypre_Index          data_stride;
   NALU_HYPRE_Int            datai;
   NALU_HYPRE_Complex       *datap;
   NALU_HYPRE_Int            constant_coefficient;

   nalu_hypre_Box           *dval_box;
   nalu_hypre_Index          dval_start;
   nalu_hypre_Index          dval_stride;
   NALU_HYPRE_Int            dvali;

   nalu_hypre_Index          loop_size;

   NALU_HYPRE_Int            i, s, istart, istop;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(matrix);
   symm_elements        = nalu_hypre_StructMatrixSymmElements(matrix);

   if (outside > 0)
   {
      grid_boxes = nalu_hypre_StructMatrixDataSpace(matrix);
   }
   else
   {
      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(matrix));
   }
   data_space = nalu_hypre_StructMatrixDataSpace(matrix);

   if (boxnum < 0)
   {
      istart = 0;
      istop  = nalu_hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   nalu_hypre_SetIndex(data_stride, 1);

   int_box = nalu_hypre_BoxCreate(nalu_hypre_StructMatrixNDim(matrix));
   dval_box = nalu_hypre_BoxDuplicate(value_box);
   nalu_hypre_BoxIMinD(dval_box, 0) *= num_stencil_indices;
   nalu_hypre_BoxIMaxD(dval_box, 0) *= num_stencil_indices;
   nalu_hypre_BoxIMaxD(dval_box, 0) += num_stencil_indices - 1;
   nalu_hypre_SetIndex(dval_stride, 1);
   nalu_hypre_IndexD(dval_stride, 0) = num_stencil_indices;

   for (i = istart; i < istop; i++)
   {
      grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);
      data_box = nalu_hypre_BoxArrayBox(data_space, i);

      nalu_hypre_IntersectBoxes(set_box, grid_box, int_box);

      /* if there was an intersection */
      if (nalu_hypre_BoxVolume(int_box))
      {
         data_start = nalu_hypre_BoxIMin(int_box);
         nalu_hypre_CopyIndex(data_start, dval_start);
         nalu_hypre_IndexD(dval_start, 0) *= num_stencil_indices;

         if ( constant_coefficient == 2 )
         {
            nalu_hypre_SetIndex(center_index, 0);
            stencil = nalu_hypre_StructMatrixStencil(matrix);
            center_rank = nalu_hypre_StructStencilElementRank( stencil, center_index );
         }

         for (s = 0; s < num_stencil_indices; s++)
         {
            /* only set stored stencil values */
            if (symm_elements[stencil_indices[s]] < 0)
            {
               datap = nalu_hypre_StructMatrixBoxData(matrix, i, stencil_indices[s]);

               if ( (constant_coefficient == 1) ||
                    (constant_coefficient == 2 && stencil_indices[s] != center_rank ))
                  /* datap has only one data point for a given i and s */
               {
                  /* should have called SetConstantValues */
                  nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
                  nalu_hypre_BoxGetSize(int_box, loop_size);

                  if (action > 0)
                  {
                     datai = nalu_hypre_CCBoxIndexRank(data_box, data_start);
                     dvali = nalu_hypre_BoxIndexRank(dval_box, dval_start);
                     datap[datai] += values[dvali];
                  }
                  else if (action > -1)
                  {
                     datai = nalu_hypre_CCBoxIndexRank(data_box, data_start);
                     dvali = nalu_hypre_BoxIndexRank(dval_box, dval_start);
                     datap[datai] = values[dvali];
                  }
                  else
                  {
                     datai = nalu_hypre_CCBoxIndexRank(data_box, data_start);
                     dvali = nalu_hypre_BoxIndexRank(dval_box, dval_start);
                     values[dvali] = datap[datai];
                     if (action == -2)
                     {
                        datap[datai] = 0;
                     }
                  }

               }
               else   /* variable coefficient: constant_coefficient==0
                         or diagonal with constant_coefficient==2   */
               {
#define DEVICE_VAR is_device_ptr(datap,values)
                  nalu_hypre_BoxGetSize(int_box, loop_size);

                  if (action > 0)
                  {
                     nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(matrix), loop_size,
                                         data_box, data_start, data_stride, datai,
                                         dval_box, dval_start, dval_stride, dvali);
                     {
                        datap[datai] += values[dvali];
                     }
                     nalu_hypre_BoxLoop2End(datai, dvali);
                  }
                  else if (action > -1)
                  {
                     nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(matrix), loop_size,
                                         data_box, data_start, data_stride, datai,
                                         dval_box, dval_start, dval_stride, dvali);
                     {
                        datap[datai] = values[dvali];
                     }
                     nalu_hypre_BoxLoop2End(datai, dvali);
                  }
                  else if (action == -2)
                  {
                     nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(matrix), loop_size,
                                         data_box, data_start, data_stride, datai,
                                         dval_box, dval_start, dval_stride, dvali);
                     {
                        values[dvali] = datap[datai];
                        datap[datai] = 0;
                     }
                     nalu_hypre_BoxLoop2End(datai, dvali);
                  }
                  else
                  {
                     nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(matrix), loop_size,
                                         data_box, data_start, data_stride, datai,
                                         dval_box, dval_start, dval_stride, dvali);
                     {
                        values[dvali] = datap[datai];
                     }
                     nalu_hypre_BoxLoop2End(datai, dvali);
                  }
#undef DEVICE_VAR
               }
            } /* end if (symm_elements) */

            nalu_hypre_IndexD(dval_start, 0) ++;
         }
      }
   }

   nalu_hypre_BoxDestroy(int_box);
   nalu_hypre_BoxDestroy(dval_box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 * (action =-2): get values and zero out (not implemented, just gets values)
 * should be called to set a constant-coefficient part of the matrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixSetConstantValues( nalu_hypre_StructMatrix *matrix,
                                     NALU_HYPRE_Int       num_stencil_indices,
                                     NALU_HYPRE_Int      *stencil_indices,
                                     NALU_HYPRE_Complex  *values,
                                     NALU_HYPRE_Int       action )
{
   nalu_hypre_BoxArray     *boxes;
   nalu_hypre_Box          *box;
   nalu_hypre_Index        center_index;
   nalu_hypre_StructStencil  *stencil;
   NALU_HYPRE_Int          center_rank;
   NALU_HYPRE_Int          constant_coefficient;

   NALU_HYPRE_Complex      *matp;

   NALU_HYPRE_Int           i, s;

   boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(matrix));
   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(matrix);

   if ( constant_coefficient == 1 )
   {
      nalu_hypre_ForBoxI(i, boxes)
      {
         box = nalu_hypre_BoxArrayBox(boxes, i);
         if (action > 0)
         {
            for (s = 0; s < num_stencil_indices; s++)
            {
               matp = nalu_hypre_StructMatrixBoxData(matrix, i,
                                                stencil_indices[s]);
               *matp += values[s];
            }
         }
         else if (action > -1)
         {
            for (s = 0; s < num_stencil_indices; s++)
            {
               matp = nalu_hypre_StructMatrixBoxData(matrix, i,
                                                stencil_indices[s]);
               *matp = values[s];
            }
         }
         else  /* action < 0 */
         {
            for (s = 0; s < num_stencil_indices; s++)
            {
               matp = nalu_hypre_StructMatrixBoxData(matrix, i,
                                                stencil_indices[s]);
               values[s] = *matp;
            }
         }
      }
   }
   else if ( constant_coefficient == 2 )
   {
      nalu_hypre_SetIndex(center_index, 0);
      stencil = nalu_hypre_StructMatrixStencil(matrix);
      center_rank = nalu_hypre_StructStencilElementRank( stencil, center_index );
      if ( action > 0 )
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            if ( stencil_indices[s] == center_rank )
            {
               /* center (diagonal), like constant_coefficient==0
                  We consider it an error, but do the best we can. */
               nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
               nalu_hypre_ForBoxI(i, boxes)
               {
                  box = nalu_hypre_BoxArrayBox(boxes, i);
                  nalu_hypre_StructMatrixSetBoxValues( matrix, box, box,
                                                  num_stencil_indices,
                                                  stencil_indices,
                                                  values, action, -1, 0 );
               }
            }
            else
            {
               /* non-center, like constant_coefficient==1 */
               matp = nalu_hypre_StructMatrixBoxData(matrix, 0,
                                                stencil_indices[s]);
               *matp += values[s];
            }
         }
      }
      else if ( action > -1 )
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            if ( stencil_indices[s] == center_rank )
            {
               /* center (diagonal), like constant_coefficient==0
                  We consider it an error, but do the best we can. */
               nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
               nalu_hypre_ForBoxI(i, boxes)
               {
                  box = nalu_hypre_BoxArrayBox(boxes, i);
                  nalu_hypre_StructMatrixSetBoxValues( matrix, box, box,
                                                  num_stencil_indices,
                                                  stencil_indices,
                                                  values, action, -1, 0 );
               }
            }
            else
            {
               /* non-center, like constant_coefficient==1 */
               matp = nalu_hypre_StructMatrixBoxData(matrix, 0,
                                                stencil_indices[s]);
               *matp += values[s];
            }
         }
      }
      else  /* action<0 */
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            if ( stencil_indices[s] == center_rank )
            {
               /* center (diagonal), like constant_coefficient==0
                  We consider it an error, but do the best we can. */
               nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
               nalu_hypre_ForBoxI(i, boxes)
               {
                  box = nalu_hypre_BoxArrayBox(boxes, i);
                  nalu_hypre_StructMatrixSetBoxValues( matrix, box, box,
                                                  num_stencil_indices,
                                                  stencil_indices,
                                                  values, -1, -1, 0 );
               }
            }
            else
            {
               /* non-center, like constant_coefficient==1 */
               matp = nalu_hypre_StructMatrixBoxData(matrix, 0,
                                                stencil_indices[s]);
               values[s] = *matp;
            }
         }
      }
   }
   else /* constant_coefficient==0 */
   {
      /* We consider this an error, but do the best we can. */
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
      nalu_hypre_ForBoxI(i, boxes)
      {
         box = nalu_hypre_BoxArrayBox(boxes, i);
         nalu_hypre_StructMatrixSetBoxValues( matrix, box, box,
                                         num_stencil_indices, stencil_indices,
                                         values, action, -1, 0 );
      }
   }
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (outside > 0): clear values possibly outside of the grid extents
 * (outside = 0): clear values only inside the grid extents
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixClearValues( nalu_hypre_StructMatrix *matrix,
                               nalu_hypre_Index         grid_index,
                               NALU_HYPRE_Int           num_stencil_indices,
                               NALU_HYPRE_Int          *stencil_indices,
                               NALU_HYPRE_Int           boxnum,
                               NALU_HYPRE_Int           outside )
{
   nalu_hypre_BoxArray      *grid_boxes;
   nalu_hypre_Box           *grid_box;

   NALU_HYPRE_Complex       *matp;

   NALU_HYPRE_Int            i, s, istart, istop;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   if (outside > 0)
   {
      grid_boxes = nalu_hypre_StructMatrixDataSpace(matrix);
   }
   else
   {
      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(matrix));
   }

   if (boxnum < 0)
   {
      istart = 0;
      istop  = nalu_hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   /*-----------------------------------------------------------------------
    * Clear the matrix coefficients
    *-----------------------------------------------------------------------*/

   for (i = istart; i < istop; i++)
   {
      grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);

      if (nalu_hypre_IndexInBox(grid_index, grid_box))
      {
         for (s = 0; s < num_stencil_indices; s++)
         {
            matp = nalu_hypre_StructMatrixBoxDataValue(matrix, i, stencil_indices[s],
                                                  grid_index);
            *matp = 0.0;
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (outside > 0): clear values possibly outside of the grid extents
 * (outside = 0): clear values only inside the grid extents
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixClearBoxValues( nalu_hypre_StructMatrix *matrix,
                                  nalu_hypre_Box          *clear_box,
                                  NALU_HYPRE_Int           num_stencil_indices,
                                  NALU_HYPRE_Int          *stencil_indices,
                                  NALU_HYPRE_Int           boxnum,
                                  NALU_HYPRE_Int           outside )
{
   nalu_hypre_BoxArray      *grid_boxes;
   nalu_hypre_Box           *grid_box;
   nalu_hypre_Box           *int_box;

   NALU_HYPRE_Int           *symm_elements;
   nalu_hypre_BoxArray      *data_space;
   nalu_hypre_Box           *data_box;
   nalu_hypre_IndexRef       data_start;
   nalu_hypre_Index          data_stride;
   NALU_HYPRE_Complex       *datap;

   nalu_hypre_Index          loop_size;

   NALU_HYPRE_Int            i, s, istart, istop;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   if (outside > 0)
   {
      grid_boxes = nalu_hypre_StructMatrixDataSpace(matrix);
   }
   else
   {
      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(matrix));
   }
   data_space = nalu_hypre_StructMatrixDataSpace(matrix);

   if (boxnum < 0)
   {
      istart = 0;
      istop  = nalu_hypre_BoxArraySize(grid_boxes);
   }
   else
   {
      istart = boxnum;
      istop  = istart + 1;
   }

   /*-----------------------------------------------------------------------
    * Clear the matrix coefficients
    *-----------------------------------------------------------------------*/

   nalu_hypre_SetIndex(data_stride, 1);

   symm_elements = nalu_hypre_StructMatrixSymmElements(matrix);

   int_box = nalu_hypre_BoxCreate(nalu_hypre_StructMatrixNDim(matrix));

   for (i = istart; i < istop; i++)
   {
      grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);
      data_box = nalu_hypre_BoxArrayBox(data_space, i);

      nalu_hypre_IntersectBoxes(clear_box, grid_box, int_box);

      /* if there was an intersection */
      if (nalu_hypre_BoxVolume(int_box))
      {
         data_start = nalu_hypre_BoxIMin(int_box);

         for (s = 0; s < num_stencil_indices; s++)
         {
            /* only clear stencil entries that are explicitly stored */
            if (symm_elements[stencil_indices[s]] < 0)
            {
               datap = nalu_hypre_StructMatrixBoxData(matrix, i,
                                                 stencil_indices[s]);

               nalu_hypre_BoxGetSize(int_box, loop_size);

#define DEVICE_VAR is_device_ptr(datap)
               nalu_hypre_BoxLoop1Begin(nalu_hypre_StructMatrixNDim(matrix), loop_size,
                                   data_box, data_start, data_stride, datai);
               {
                  datap[datai] = 0.0;
               }
               nalu_hypre_BoxLoop1End(datai);
#undef DEVICE_VAR
            }
         }
      }
   }

   nalu_hypre_BoxDestroy(int_box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixAssemble( nalu_hypre_StructMatrix *matrix )
{
   NALU_HYPRE_Int              ndim = nalu_hypre_StructMatrixNDim(matrix);
   NALU_HYPRE_Int             *num_ghost = nalu_hypre_StructMatrixNumGhost(matrix);

   NALU_HYPRE_Int              comm_num_values, mat_num_values, constant_coefficient;
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_Int              stencil_size;
   nalu_hypre_StructStencil   *stencil;
#endif
   nalu_hypre_CommInfo        *comm_info;
   nalu_hypre_CommPkg         *comm_pkg;

   nalu_hypre_CommHandle      *comm_handle;

   NALU_HYPRE_Complex         *matrix_data = nalu_hypre_StructMatrixData(matrix);

   NALU_HYPRE_Complex         *matrix_data_comm = matrix_data;

   /* BEGIN - variables for ghost layer identity code below */
   nalu_hypre_StructGrid      *grid;
   nalu_hypre_BoxManager      *boxman;
   nalu_hypre_BoxArray        *data_space;
   nalu_hypre_BoxArrayArray   *boundary_boxes;
   nalu_hypre_BoxArray        *boundary_box_a;
   nalu_hypre_BoxArray        *entry_box_a;
   nalu_hypre_BoxArray        *tmp_box_a;
   nalu_hypre_Box             *data_box;
   nalu_hypre_Box             *boundary_box;
   nalu_hypre_Box             *entry_box;
   nalu_hypre_BoxManEntry    **entries;
   nalu_hypre_Index            loop_size;
   nalu_hypre_Index            index;
   nalu_hypre_IndexRef         start;
   nalu_hypre_Index            stride;
   NALU_HYPRE_Complex         *datap;
   NALU_HYPRE_Int              i, j, ei;
   NALU_HYPRE_Int              num_entries;
   /* End - variables for ghost layer identity code below */

   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient( matrix );

   /*-----------------------------------------------------------------------
    * Set ghost zones along the domain boundary to the identity to enable code
    * simplifications elsewhere in hypre (e.g., CyclicReduction).
    *
    * Intersect each data box with the BoxMan to get neighbors, then subtract
    * the neighbors from the box to get the boundary boxes.
    *-----------------------------------------------------------------------*/

   if ( constant_coefficient != 1 )
   {
      data_space = nalu_hypre_StructMatrixDataSpace(matrix);
      grid       = nalu_hypre_StructMatrixGrid(matrix);
      boxman     = nalu_hypre_StructGridBoxMan(grid);

      boundary_boxes = nalu_hypre_BoxArrayArrayCreate(
                          nalu_hypre_BoxArraySize(data_space), ndim);
      entry_box_a    = nalu_hypre_BoxArrayCreate(0, ndim);
      tmp_box_a      = nalu_hypre_BoxArrayCreate(0, ndim);
      nalu_hypre_ForBoxI(i, data_space)
      {
         /* copy data box to boundary_box_a */
         boundary_box_a = nalu_hypre_BoxArrayArrayBoxArray(boundary_boxes, i);
         nalu_hypre_BoxArraySetSize(boundary_box_a, 1);
         boundary_box = nalu_hypre_BoxArrayBox(boundary_box_a, 0);
         nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(data_space, i), boundary_box);

         nalu_hypre_BoxManIntersect(boxman,
                               nalu_hypre_BoxIMin(boundary_box),
                               nalu_hypre_BoxIMax(boundary_box),
                               &entries, &num_entries);

         /* put neighbor boxes into entry_box_a */
         nalu_hypre_BoxArraySetSize(entry_box_a, num_entries);
         for (ei = 0; ei < num_entries; ei++)
         {
            entry_box = nalu_hypre_BoxArrayBox(entry_box_a, ei);
            nalu_hypre_BoxManEntryGetExtents(entries[ei],
                                        nalu_hypre_BoxIMin(entry_box),
                                        nalu_hypre_BoxIMax(entry_box));
         }
         nalu_hypre_TFree(entries, NALU_HYPRE_MEMORY_HOST);

         /* subtract neighbor boxes (entry_box_a) from data box (boundary_box_a) */
         nalu_hypre_SubtractBoxArrays(boundary_box_a, entry_box_a, tmp_box_a);
      }
      nalu_hypre_BoxArrayDestroy(entry_box_a);
      nalu_hypre_BoxArrayDestroy(tmp_box_a);

      /* set boundary ghost zones to the identity equation */

      nalu_hypre_SetIndex(index, 0);
      nalu_hypre_SetIndex(stride, 1);
      data_space = nalu_hypre_StructMatrixDataSpace(matrix);
      nalu_hypre_ForBoxI(i, data_space)
      {
         datap = nalu_hypre_StructMatrixExtractPointerByIndex(matrix, i, index);

         if (datap)
         {
            data_box = nalu_hypre_BoxArrayBox(data_space, i);
            boundary_box_a = nalu_hypre_BoxArrayArrayBoxArray(boundary_boxes, i);
            nalu_hypre_ForBoxI(j, boundary_box_a)
            {
               boundary_box = nalu_hypre_BoxArrayBox(boundary_box_a, j);
               start = nalu_hypre_BoxIMin(boundary_box);

               nalu_hypre_BoxGetSize(boundary_box, loop_size);

#define DEVICE_VAR is_device_ptr(datap)
               nalu_hypre_BoxLoop1Begin(nalu_hypre_StructMatrixNDim(matrix), loop_size,
                                   data_box, start, stride, datai);
               {
                  datap[datai] = 1.0;
               }
               nalu_hypre_BoxLoop1End(datai);
#undef DEVICE_VAR
            }
         }
      }

      nalu_hypre_BoxArrayArrayDestroy(boundary_boxes);
   }

   /*-----------------------------------------------------------------------
    * If the CommPkg has not been set up, set it up
    *
    * The matrix data array is assumed to have two segments - an initial
    * segment of data constant over all space, followed by a segment with
    * comm_num_values matrix entries for each mesh element.  The mesh-dependent
    * data is, of course, the only part relevent to communications.
    * For constant_coefficient==0, all the data is mesh-dependent.
    * For constant_coefficient==1, all  data is constant.
    * For constant_coefficient==2, both segments are non-null.
    *-----------------------------------------------------------------------*/

   mat_num_values = nalu_hypre_StructMatrixNumValues(matrix);

   if ( constant_coefficient == 0 )
   {
      comm_num_values = mat_num_values;
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      if (nalu_hypre_StructGridDataLocation(grid) == NALU_HYPRE_MEMORY_HOST)
      {
         matrix_data_comm = nalu_hypre_StructMatrixDataConst(matrix);
      }
#endif
   }
   else if ( constant_coefficient == 1 )
   {
      comm_num_values = 0;
   }
   else /* constant_coefficient==2 */
   {
      comm_num_values = 1;
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      if (nalu_hypre_StructGridDataLocation(grid) == NALU_HYPRE_MEMORY_HOST)
      {
         stencil = nalu_hypre_StructMatrixStencil(matrix);
         stencil_size  = nalu_hypre_StructStencilSize(stencil);
         matrix_data_comm = nalu_hypre_StructMatrixDataConst(matrix) + stencil_size;
      }
#endif
   }

   comm_pkg = nalu_hypre_StructMatrixCommPkg(matrix);

   if (!comm_pkg)
   {
      nalu_hypre_CreateCommInfoFromNumGhost(nalu_hypre_StructMatrixGrid(matrix),
                                       num_ghost, &comm_info);
      nalu_hypre_CommPkgCreate(comm_info,
                          nalu_hypre_StructMatrixDataSpace(matrix),
                          nalu_hypre_StructMatrixDataSpace(matrix),
                          comm_num_values, NULL, 0,
                          nalu_hypre_StructMatrixComm(matrix), &comm_pkg);
      nalu_hypre_CommInfoDestroy(comm_info);

      nalu_hypre_StructMatrixCommPkg(matrix) = comm_pkg;
   }

   /*-----------------------------------------------------------------------
    * Update the ghost data
    * This takes care of the communication needs of all known functions
    * referencing the matrix.
    *
    * At present this is the only place where matrix data gets communicated.
    * However, comm_pkg is kept as long as the matrix is, in case some
    * future version hypre has a use for it - e.g. if the user replaces
    * a matrix with a very similar one, we may not want to recompute comm_pkg.
    *-----------------------------------------------------------------------*/

   if ( constant_coefficient != 1 )
   {
      nalu_hypre_InitializeCommunication( comm_pkg,
                                     matrix_data_comm,
                                     matrix_data_comm, 0, 0,
                                     &comm_handle );
      nalu_hypre_FinalizeCommunication( comm_handle );
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixSetNumGhost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixSetNumGhost( nalu_hypre_StructMatrix *matrix,
                               NALU_HYPRE_Int          *num_ghost )
{
   NALU_HYPRE_Int  d, ndim = nalu_hypre_StructMatrixNDim(matrix);

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_StructMatrixNumGhost(matrix)[2 * d]     = num_ghost[2 * d];
      nalu_hypre_StructMatrixNumGhost(matrix)[2 * d + 1] = num_ghost[2 * d + 1];
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixSetConstantCoefficient
 * deprecated in user interface, in favor of SetConstantEntries.
 * left here for internal use
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixSetConstantCoefficient( nalu_hypre_StructMatrix *matrix,
                                          NALU_HYPRE_Int          constant_coefficient )
{
   nalu_hypre_StructMatrixConstantCoefficient(matrix) = constant_coefficient;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixSetConstantEntries
 * - nentries is the number of array entries
 * - Each NALU_HYPRE_Int entries[i] is an index into the shape array of the stencil
 *   of the matrix
 * In the present version, only three possibilites are recognized:
 * - no entries constant                 (constant_coefficient==0)
 * - all entries constant                (constant_coefficient==1)
 * - all but the diagonal entry constant (constant_coefficient==2)
 * If something else is attempted, this function will return a nonzero error.
 * In the present version, if this function is called more than once, only
 * the last call will take effect.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int  nalu_hypre_StructMatrixSetConstantEntries( nalu_hypre_StructMatrix *matrix,
                                                 NALU_HYPRE_Int           nentries,
                                                 NALU_HYPRE_Int          *entries )
{
   /* We make an array offdconst corresponding to the stencil's shape array,
      and use "entries" to fill it with flags - 1 for constant, 0 otherwise.
      By counting the nonzeros in offdconst, and by checking whether its
      diagonal entry is nonzero, we can distinguish among the three
      presently legal values of constant_coefficient, and detect input errors.
      We do not need to treat duplicates in "entries" as an error condition.
   */
   nalu_hypre_StructStencil *stencil = nalu_hypre_StructMatrixUserStencil(matrix);
   /* ... Stencil doesn't exist yet */
   NALU_HYPRE_Int stencil_size  = nalu_hypre_StructStencilSize(stencil);
   NALU_HYPRE_Int *offdconst = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  stencil_size, NALU_HYPRE_MEMORY_HOST);
   /* ... note: CTAlloc initializes to 0 (normally it works by calling calloc) */
   NALU_HYPRE_Int nconst = 0;
   NALU_HYPRE_Int constant_coefficient, diag_rank;
   nalu_hypre_Index diag_index;
   NALU_HYPRE_Int i, j;

   for ( i = 0; i < nentries; ++i )
   {
      offdconst[ entries[i] ] = 1;
   }

   for ( j = 0; j < stencil_size; ++j )
   {
      nconst += offdconst[j];
   }

   if ( nconst <= 0 )
   {
      constant_coefficient = 0;
   }
   else if ( nconst >= stencil_size )
   {
      constant_coefficient = 1;
   }
   else
   {
      nalu_hypre_SetIndex(diag_index, 0);
      diag_rank = nalu_hypre_StructStencilElementRank( stencil, diag_index );
      if ( offdconst[diag_rank] == 0 )
      {
         constant_coefficient = 2;
         if ( nconst != (stencil_size - 1) )
         {
            nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
         }
      }
      else
      {
         constant_coefficient = 0;
         nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
      }
   }

   nalu_hypre_StructMatrixSetConstantCoefficient( matrix, constant_coefficient );

   nalu_hypre_TFree(offdconst, NALU_HYPRE_MEMORY_HOST);
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixClearGhostValues( nalu_hypre_StructMatrix *matrix )
{
   NALU_HYPRE_Int             ndim = nalu_hypre_StructMatrixNDim(matrix);
   nalu_hypre_Box            *m_data_box;

   NALU_HYPRE_Complex        *mp;

   nalu_hypre_StructStencil  *stencil;
   NALU_HYPRE_Int            *symm_elements;
   nalu_hypre_BoxArray       *boxes;
   nalu_hypre_Box            *box;
   nalu_hypre_BoxArray       *diff_boxes;
   nalu_hypre_Box            *diff_box;
   nalu_hypre_Index           loop_size;
   nalu_hypre_IndexRef        start;
   nalu_hypre_Index           unit_stride;

   NALU_HYPRE_Int             i, j, s;

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   nalu_hypre_SetIndex(unit_stride, 1);

   stencil = nalu_hypre_StructMatrixStencil(matrix);
   symm_elements = nalu_hypre_StructMatrixSymmElements(matrix);
   boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(matrix));
   diff_boxes = nalu_hypre_BoxArrayCreate(0, ndim);
   nalu_hypre_ForBoxI(i, boxes)
   {
      box        = nalu_hypre_BoxArrayBox(boxes, i);
      m_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(matrix), i);
      nalu_hypre_BoxArraySetSize(diff_boxes, 0);
      nalu_hypre_SubtractBoxes(m_data_box, box, diff_boxes);

      for (s = 0; s < nalu_hypre_StructStencilSize(stencil); s++)
      {
         /* only clear stencil entries that are explicitly stored */
         if (symm_elements[s] < 0)
         {
            mp = nalu_hypre_StructMatrixBoxData(matrix, i, s);
            nalu_hypre_ForBoxI(j, diff_boxes)
            {
               diff_box = nalu_hypre_BoxArrayBox(diff_boxes, j);
               start = nalu_hypre_BoxIMin(diff_box);

               nalu_hypre_BoxGetSize(diff_box, loop_size);

#define DEVICE_VAR is_device_ptr(mp)
               nalu_hypre_BoxLoop1Begin(nalu_hypre_StructMatrixNDim(matrix), loop_size,
                                   m_data_box, start, unit_stride, mi);
               {
                  mp[mi] = 0.0;
               }
               nalu_hypre_BoxLoop1End(mi);
#undef DEVICE_VAR
            }
         }
      }
   }
   nalu_hypre_BoxArrayDestroy(diff_boxes);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixPrintData
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixPrintData( FILE               *file,
                             nalu_hypre_StructMatrix *matrix,
                             NALU_HYPRE_Int           all )
{
   NALU_HYPRE_Int             ndim            = nalu_hypre_StructMatrixNDim(matrix);
   NALU_HYPRE_Int             num_values      = nalu_hypre_StructMatrixNumValues(matrix);
   NALU_HYPRE_Int             ctecoef         = nalu_hypre_StructMatrixConstantCoefficient(matrix);
   nalu_hypre_StructGrid     *grid            = nalu_hypre_StructMatrixGrid(matrix);
   nalu_hypre_StructStencil  *stencil         = nalu_hypre_StructMatrixStencil(matrix);
   NALU_HYPRE_Int             stencil_size    = nalu_hypre_StructStencilSize(stencil);
   NALU_HYPRE_Int            *symm_elements   = nalu_hypre_StructMatrixSymmElements(matrix);
   nalu_hypre_BoxArray       *data_space      = nalu_hypre_StructMatrixDataSpace(matrix);
   NALU_HYPRE_Int             data_size       = nalu_hypre_StructMatrixDataSize(matrix);
   nalu_hypre_BoxArray       *grid_boxes      = nalu_hypre_StructGridBoxes(grid);
   NALU_HYPRE_Complex        *data            = nalu_hypre_StructMatrixData(matrix);
   NALU_HYPRE_MemoryLocation  memory_location = nalu_hypre_StructMatrixMemoryLocation(matrix);
   nalu_hypre_BoxArray       *boxes;
   nalu_hypre_Index           center_index;
   NALU_HYPRE_Int             center_rank;
   NALU_HYPRE_Complex        *h_data;

   /* Allocate/Point to data on the host memory */
   if (nalu_hypre_GetActualMemLocation(memory_location) != nalu_hypre_MEMORY_HOST)
   {
      h_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, data_size, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(h_data, data, NALU_HYPRE_Complex, data_size,
                    NALU_HYPRE_MEMORY_HOST, memory_location);
   }
   else
   {
      h_data = data;
   }

   /* Print ghost data (all) also or only real data? */
   boxes = (all) ? data_space : grid_boxes;

   /* Print data to file */
   if (ctecoef == 1)
   {
      nalu_hypre_PrintCCBoxArrayData(file, boxes, data_space, num_values, h_data);
   }
   else if (ctecoef == 2)
   {
      nalu_hypre_SetIndex(center_index, 0);
      center_rank = nalu_hypre_StructStencilElementRank(stencil, center_index);

      nalu_hypre_PrintCCVDBoxArrayData(file, boxes, data_space, num_values,
                                  center_rank, stencil_size, symm_elements,
                                  ndim, h_data);
   }
   else
   {
      nalu_hypre_PrintBoxArrayData(file, boxes, data_space, num_values,
                              ndim, h_data);
   }

   /* Free memory */
   if (nalu_hypre_GetActualMemLocation(memory_location) != nalu_hypre_MEMORY_HOST)
   {
      nalu_hypre_TFree(h_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixReadData
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixReadData( FILE               *file,
                            nalu_hypre_StructMatrix *matrix )
{
   NALU_HYPRE_Int             ndim            = nalu_hypre_StructMatrixNDim(matrix);
   NALU_HYPRE_Int             num_values      = nalu_hypre_StructMatrixNumValues(matrix);
   NALU_HYPRE_Int             ctecoef         = nalu_hypre_StructMatrixConstantCoefficient(matrix);
   nalu_hypre_StructGrid     *grid            = nalu_hypre_StructMatrixGrid(matrix);
   nalu_hypre_StructStencil  *stencil         = nalu_hypre_StructMatrixStencil(matrix);
   NALU_HYPRE_Int             stencil_size    = nalu_hypre_StructStencilSize(stencil);
   NALU_HYPRE_Int             symmetric       = nalu_hypre_StructMatrixSymmetric(matrix);
   nalu_hypre_BoxArray       *data_space      = nalu_hypre_StructMatrixDataSpace(matrix);
   nalu_hypre_BoxArray       *boxes           = nalu_hypre_StructGridBoxes(grid);
   NALU_HYPRE_Complex        *data            = nalu_hypre_StructMatrixData(matrix);
   NALU_HYPRE_Int             data_size       = nalu_hypre_StructMatrixDataSize(matrix);
   NALU_HYPRE_MemoryLocation  memory_location = nalu_hypre_StructMatrixMemoryLocation(matrix);
   NALU_HYPRE_Complex        *h_data;
   NALU_HYPRE_Int             real_stencil_size;

   /* Allocate/Point to data on the host memory */
   if (nalu_hypre_GetActualMemLocation(memory_location) != nalu_hypre_MEMORY_HOST)
   {
      h_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, data_size, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      h_data = data;
   }

   /* real_stencil_size is the stencil size of the matrix after it's fixed up
      by the call (if any) of nalu_hypre_StructStencilSymmetrize from
      nalu_hypre_StructMatrixInitializeShell.*/
   if (symmetric)
   {
      real_stencil_size = 2 * stencil_size - 1;
   }
   else
   {
      real_stencil_size = stencil_size;
   }

   /* Read data from file */
   if (ctecoef == 0)
   {
      nalu_hypre_ReadBoxArrayData(file, boxes, data_space,
                             num_values, ndim, h_data);
   }
   else
   {
      nalu_hypre_assert(ctecoef <= 2);
      nalu_hypre_ReadBoxArrayData_CC(file, boxes, data_space,
                                stencil_size, real_stencil_size,
                                ctecoef, ndim, h_data);
   }

   /* Move data to the device memory if necessary and free host data */
   if (nalu_hypre_GetActualMemLocation(memory_location) != nalu_hypre_MEMORY_HOST)
   {
      nalu_hypre_TMemcpy(data, h_data, NALU_HYPRE_Complex, data_size,
                    memory_location, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(h_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixPrint( const char         *filename,
                         nalu_hypre_StructMatrix *matrix,
                         NALU_HYPRE_Int           all      )
{
   FILE                 *file;
   char                  new_filename[255];

   nalu_hypre_StructGrid     *grid;

   nalu_hypre_StructStencil  *stencil;
   nalu_hypre_Index          *stencil_shape;
   NALU_HYPRE_Int             stencil_size;

   NALU_HYPRE_Int             ndim, num_values;

   NALU_HYPRE_Int            *symm_elements;

   NALU_HYPRE_Int             i, j, d;
   NALU_HYPRE_Int             myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

   nalu_hypre_MPI_Comm_rank(nalu_hypre_StructMatrixComm(matrix), &myid);

   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      nalu_hypre_printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   /*----------------------------------------
    * Print header info
    *----------------------------------------*/

   nalu_hypre_fprintf(file, "StructMatrix\n");

   nalu_hypre_fprintf(file, "\nSymmetric: %d\n", nalu_hypre_StructMatrixSymmetric(matrix));
   nalu_hypre_fprintf(file, "\nConstantCoefficient: %d\n",
                 nalu_hypre_StructMatrixConstantCoefficient(matrix));

   /* print grid info */
   nalu_hypre_fprintf(file, "\nGrid:\n");
   grid = nalu_hypre_StructMatrixGrid(matrix);
   nalu_hypre_StructGridPrint(file, grid);

   /* print stencil info */
   nalu_hypre_fprintf(file, "\nStencil:\n");
   stencil = nalu_hypre_StructMatrixStencil(matrix);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);

   ndim = nalu_hypre_StructMatrixNDim(matrix);
   num_values = nalu_hypre_StructMatrixNumValues(matrix);
   symm_elements = nalu_hypre_StructMatrixSymmElements(matrix);
   nalu_hypre_fprintf(file, "%d\n", num_values);
   stencil_size = nalu_hypre_StructStencilSize(stencil);
   j = 0;
   for (i = 0; i < stencil_size; i++)
   {
      if (symm_elements[i] < 0)
      {
         /* Print line of the form: "%d: %d %d %d\n" */
         nalu_hypre_fprintf(file, "%d:", j++);
         for (d = 0; d < ndim; d++)
         {
            nalu_hypre_fprintf(file, " %d", nalu_hypre_IndexD(stencil_shape[i], d));
         }
         nalu_hypre_fprintf(file, "\n");
      }
   }

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   nalu_hypre_fprintf(file, "\nData:\n");
   nalu_hypre_StructMatrixPrintData(file, matrix, all);

   /*----------------------------------------
    * Close file
    *----------------------------------------*/

   fflush(file);
   fclose(file);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixRead
 *--------------------------------------------------------------------------*/

nalu_hypre_StructMatrix *
nalu_hypre_StructMatrixRead( MPI_Comm    comm,
                        const char *filename,
                        NALU_HYPRE_Int  *num_ghost )
{
   FILE                 *file;
   char                  new_filename[255];

   nalu_hypre_StructMatrix   *matrix;

   nalu_hypre_StructGrid     *grid;
   NALU_HYPRE_Int             ndim;

   nalu_hypre_StructStencil  *stencil;
   nalu_hypre_Index          *stencil_shape;
   NALU_HYPRE_Int             stencil_size;
   NALU_HYPRE_Int             symmetric;
   NALU_HYPRE_Int             constant_coefficient;

   NALU_HYPRE_Int             i, d, idummy;

   NALU_HYPRE_Int             myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

   nalu_hypre_MPI_Comm_rank(comm, &myid );

   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      nalu_hypre_printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   /*----------------------------------------
    * Read header info
    *----------------------------------------*/

   nalu_hypre_fscanf(file, "StructMatrix\n");

   nalu_hypre_fscanf(file, "\nSymmetric: %d\n", &symmetric);
   nalu_hypre_fscanf(file, "\nConstantCoefficient: %d\n", &constant_coefficient);

   /* read grid info */
   nalu_hypre_fscanf(file, "\nGrid:\n");
   nalu_hypre_StructGridRead(comm, file, &grid);

   /* read stencil info */
   nalu_hypre_fscanf(file, "\nStencil:\n");
   ndim = nalu_hypre_StructGridNDim(grid);
   nalu_hypre_fscanf(file, "%d\n", &stencil_size);
   stencil_shape = nalu_hypre_CTAlloc(nalu_hypre_Index,  stencil_size, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < stencil_size; i++)
   {
      /* Read line of the form: "%d: %d %d %d\n" */
      nalu_hypre_fscanf(file, "%d:", &idummy);
      for (d = 0; d < ndim; d++)
      {
         nalu_hypre_fscanf(file, " %d", &nalu_hypre_IndexD(stencil_shape[i], d));
      }
      nalu_hypre_fscanf(file, "\n");
   }
   stencil = nalu_hypre_StructStencilCreate(ndim, stencil_size, stencil_shape);

   /*----------------------------------------
    * Initialize the matrix
    *----------------------------------------*/

   matrix = nalu_hypre_StructMatrixCreate(comm, grid, stencil);
   nalu_hypre_StructMatrixSymmetric(matrix) = symmetric;
   nalu_hypre_StructMatrixConstantCoefficient(matrix) = constant_coefficient;
   nalu_hypre_StructMatrixSetNumGhost(matrix, num_ghost);
   nalu_hypre_StructMatrixInitialize(matrix);

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   nalu_hypre_fscanf(file, "\nData:\n");
   nalu_hypre_StructMatrixReadData(file, matrix);

   /*----------------------------------------
    * Assemble the matrix
    *----------------------------------------*/

   nalu_hypre_StructMatrixAssemble(matrix);

   /*----------------------------------------
    * Close file
    *----------------------------------------*/

   fclose(file);

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatrixMigrate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixMigrate( nalu_hypre_StructMatrix *from_matrix,
                           nalu_hypre_StructMatrix *to_matrix   )
{
   nalu_hypre_CommInfo        *comm_info;
   nalu_hypre_CommPkg         *comm_pkg;
   nalu_hypre_CommHandle      *comm_handle;

   NALU_HYPRE_Int              constant_coefficient, comm_num_values;
   NALU_HYPRE_Int              stencil_size, mat_num_values;
   nalu_hypre_StructStencil   *stencil;

   NALU_HYPRE_Complex         *matrix_data_from = nalu_hypre_StructMatrixData(from_matrix);
   NALU_HYPRE_Complex         *matrix_data_to = nalu_hypre_StructMatrixData(to_matrix);
   NALU_HYPRE_Complex         *matrix_data_comm_from = matrix_data_from;
   NALU_HYPRE_Complex         *matrix_data_comm_to = matrix_data_to;

   /*------------------------------------------------------
    * Set up nalu_hypre_CommPkg
    *------------------------------------------------------*/

   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient( from_matrix );
   nalu_hypre_assert( constant_coefficient == nalu_hypre_StructMatrixConstantCoefficient( to_matrix ) );

   mat_num_values = nalu_hypre_StructMatrixNumValues(from_matrix);
   nalu_hypre_assert( mat_num_values == nalu_hypre_StructMatrixNumValues(to_matrix) );

   if ( constant_coefficient == 0 )
   {
      comm_num_values = mat_num_values;
   }
   else if ( constant_coefficient == 1 )
   {
      comm_num_values = 0;
   }
   else /* constant_coefficient==2 */
   {
      comm_num_values = 1;
      stencil = nalu_hypre_StructMatrixStencil(from_matrix);
      stencil_size = nalu_hypre_StructStencilSize(stencil);
      nalu_hypre_assert(stencil_size ==
                   nalu_hypre_StructStencilSize( nalu_hypre_StructMatrixStencil(to_matrix) ) );
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      if (nalu_hypre_StructGridDataLocation(nalu_hypre_StructMatrixGrid(from_matrix)) == NALU_HYPRE_MEMORY_HOST)
      {
         stencil = nalu_hypre_StructMatrixStencil(from_matrix);
         stencil_size  = nalu_hypre_StructStencilSize(stencil);
         matrix_data_comm_from = nalu_hypre_StructMatrixDataConst(from_matrix) + stencil_size;
         stencil = nalu_hypre_StructMatrixStencil(to_matrix);
         stencil_size  = nalu_hypre_StructStencilSize(stencil);
         matrix_data_comm_to = nalu_hypre_StructMatrixDataConst(to_matrix) + stencil_size;
      }
#endif
   }

   nalu_hypre_CreateCommInfoFromGrids(nalu_hypre_StructMatrixGrid(from_matrix),
                                 nalu_hypre_StructMatrixGrid(to_matrix),
                                 &comm_info);
   nalu_hypre_CommPkgCreate(comm_info,
                       nalu_hypre_StructMatrixDataSpace(from_matrix),
                       nalu_hypre_StructMatrixDataSpace(to_matrix),
                       comm_num_values, NULL, 0,
                       nalu_hypre_StructMatrixComm(from_matrix), &comm_pkg);
   nalu_hypre_CommInfoDestroy(comm_info);
   /* is this correct for periodic? */

   /*-----------------------------------------------------------------------
    * Migrate the matrix data
    *-----------------------------------------------------------------------*/

   if ( constant_coefficient != 1 )
   {
      nalu_hypre_InitializeCommunication( comm_pkg,
                                     matrix_data_comm_from,
                                     matrix_data_comm_to, 0, 0,
                                     &comm_handle );
      nalu_hypre_FinalizeCommunication( comm_handle );
   }
   nalu_hypre_CommPkgDestroy(comm_pkg);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * clears matrix stencil coefficients reaching outside of the physical boundaries
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatrixClearBoundary( nalu_hypre_StructMatrix *matrix)
{
   NALU_HYPRE_Int            ndim = nalu_hypre_StructMatrixNDim(matrix);
   NALU_HYPRE_Complex       *data;
   nalu_hypre_BoxArray      *grid_boxes;
   nalu_hypre_BoxArray      *data_space;
   /*nalu_hypre_Box           *box;*/
   nalu_hypre_Box           *grid_box;
   nalu_hypre_Box           *data_box;
   nalu_hypre_Box           *tmp_box;
   nalu_hypre_Index         *shape;
   nalu_hypre_Index          stencil_element;
   nalu_hypre_Index          loop_size;
   nalu_hypre_IndexRef       start;
   nalu_hypre_Index          stride;
   nalu_hypre_StructGrid    *grid;
   nalu_hypre_StructStencil *stencil;
   nalu_hypre_BoxArray      *boundary;

   NALU_HYPRE_Int           i, i2, j;

   /*-----------------------------------------------------------------------
    * Set the matrix coefficients
    *-----------------------------------------------------------------------*/

   grid = nalu_hypre_StructMatrixGrid(matrix);
   stencil = nalu_hypre_StructMatrixStencil(matrix);
   grid_boxes = nalu_hypre_StructGridBoxes(grid);
   ndim = nalu_hypre_StructStencilNDim(stencil);
   data_space = nalu_hypre_StructMatrixDataSpace(matrix);
   nalu_hypre_SetIndex(stride, 1);
   shape = nalu_hypre_StructStencilShape(stencil);

   for (j = 0; j < nalu_hypre_StructStencilSize(stencil); j++)
   {
      nalu_hypre_CopyIndex(shape[j], stencil_element);
      if (!nalu_hypre_IndexEqual(stencil_element, 0, ndim))
      {
         nalu_hypre_ForBoxI(i, grid_boxes)
         {
            grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);
            data_box = nalu_hypre_BoxArrayBox(data_space, i);
            boundary = nalu_hypre_BoxArrayCreate( 0, ndim );
            nalu_hypre_GeneralBoxBoundaryIntersect(grid_box, grid, stencil_element,
                                              boundary);
            data = nalu_hypre_StructMatrixBoxData(matrix, i, j);
            nalu_hypre_ForBoxI(i2, boundary)
            {
               tmp_box = nalu_hypre_BoxArrayBox(boundary, i2);
               nalu_hypre_BoxGetSize(tmp_box, loop_size);
               start = nalu_hypre_BoxIMin(tmp_box);
#define DEVICE_VAR is_device_ptr(data)
               nalu_hypre_BoxLoop1Begin(ndim, loop_size, data_box, start, stride, ixyz);
               {
                  data[ixyz] = 0.0;
               }
               nalu_hypre_BoxLoop1End(ixyz);
#undef DEVICE_VAR
            }
            nalu_hypre_BoxArrayDestroy(boundary);
         }
      }
   }

   return nalu_hypre_error_flag;
}
