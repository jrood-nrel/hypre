/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_StructVector class.
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"
#include "_nalu_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

nalu_hypre_StructVector *
nalu_hypre_StructVectorCreate( MPI_Comm          comm,
                          nalu_hypre_StructGrid *grid )
{
   NALU_HYPRE_Int            ndim = nalu_hypre_StructGridNDim(grid);
   nalu_hypre_StructVector  *vector;
   NALU_HYPRE_Int            i;

   vector = nalu_hypre_CTAlloc(nalu_hypre_StructVector, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_StructVectorComm(vector)           = comm;
   nalu_hypre_StructGridRef(grid, &nalu_hypre_StructVectorGrid(vector));
   nalu_hypre_StructVectorDataAlloced(vector)    = 1;
   nalu_hypre_StructVectorBGhostNotClear(vector) = 0;
   nalu_hypre_StructVectorRefCount(vector)       = 1;

   /* set defaults */
   for (i = 0; i < 2 * ndim; i++)
   {
      nalu_hypre_StructVectorNumGhost(vector)[i] = nalu_hypre_StructGridNumGhost(grid)[i];
   }

   nalu_hypre_StructVectorMemoryLocation(vector) = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());

   return vector;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

nalu_hypre_StructVector *
nalu_hypre_StructVectorRef( nalu_hypre_StructVector *vector )
{
   nalu_hypre_StructVectorRefCount(vector) ++;

   return vector;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorDestroy( nalu_hypre_StructVector *vector )
{
   if (vector)
   {
      nalu_hypre_StructVectorRefCount(vector) --;
      if (nalu_hypre_StructVectorRefCount(vector) == 0)
      {
         if (nalu_hypre_StructVectorDataAlloced(vector))
         {
            nalu_hypre_TFree(nalu_hypre_StructVectorData(vector), nalu_hypre_StructVectorMemoryLocation(vector));
         }

         nalu_hypre_TFree(nalu_hypre_StructVectorDataIndices(vector), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_BoxArrayDestroy(nalu_hypre_StructVectorDataSpace(vector));
         nalu_hypre_StructGridDestroy(nalu_hypre_StructVectorGrid(vector));
         nalu_hypre_TFree(vector, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorInitializeShell( nalu_hypre_StructVector *vector )
{
   NALU_HYPRE_Int             ndim = nalu_hypre_StructVectorNDim(vector);
   nalu_hypre_StructGrid     *grid;

   NALU_HYPRE_Int            *num_ghost;

   nalu_hypre_BoxArray       *data_space;
   nalu_hypre_BoxArray       *boxes;
   nalu_hypre_Box            *box;
   nalu_hypre_Box            *data_box;

   NALU_HYPRE_Int            *data_indices;
   NALU_HYPRE_Int             data_size;

   NALU_HYPRE_Int             i, d;

   /*-----------------------------------------------------------------------
    * Set up data_space
    *-----------------------------------------------------------------------*/

   grid = nalu_hypre_StructVectorGrid(vector);

   if (nalu_hypre_StructVectorDataSpace(vector) == NULL)
   {
      num_ghost = nalu_hypre_StructVectorNumGhost(vector);

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

      nalu_hypre_StructVectorDataSpace(vector) = data_space;
   }

   /*-----------------------------------------------------------------------
    * Set up data_indices array and data_size
    *-----------------------------------------------------------------------*/

   if (nalu_hypre_StructVectorDataIndices(vector) == NULL)
   {
      data_space = nalu_hypre_StructVectorDataSpace(vector);
      data_indices = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(data_space), NALU_HYPRE_MEMORY_HOST);

      data_size = 0;
      nalu_hypre_ForBoxI(i, data_space)
      {
         data_box = nalu_hypre_BoxArrayBox(data_space, i);

         data_indices[i] = data_size;
         data_size += nalu_hypre_BoxVolume(data_box);
      }

      nalu_hypre_StructVectorDataIndices(vector) = data_indices;

      nalu_hypre_StructVectorDataSize(vector)    = data_size;

   }

   /*-----------------------------------------------------------------------
    * Set total number of nonzero coefficients
    *-----------------------------------------------------------------------*/

   nalu_hypre_StructVectorGlobalSize(vector) = nalu_hypre_StructGridGlobalSize(grid);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorInitializeData( nalu_hypre_StructVector *vector,
                                  NALU_HYPRE_Complex      *data)
{
   nalu_hypre_StructVectorData(vector) = data;
   nalu_hypre_StructVectorDataAlloced(vector) = 0;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorInitialize( nalu_hypre_StructVector *vector )
{
   NALU_HYPRE_Complex *data;

   nalu_hypre_StructVectorInitializeShell(vector);

   data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nalu_hypre_StructVectorDataSize(vector),
                        nalu_hypre_StructVectorMemoryLocation(vector));

   nalu_hypre_StructVectorInitializeData(vector, data);
   nalu_hypre_StructVectorDataAlloced(vector) = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (action > 0): add-to values
 * (action = 0): set values
 * (action < 0): get values
 *
 * (outside > 0): set values possibly outside of the grid extents
 * (outside = 0): set values only inside the grid extents
 *
 * NOTE: Getting and setting values outside of the grid extents requires care,
 * as these values may be stored in multiple ghost zone locations.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorSetValues( nalu_hypre_StructVector *vector,
                             nalu_hypre_Index         grid_index,
                             NALU_HYPRE_Complex      *values,
                             NALU_HYPRE_Int           action,
                             NALU_HYPRE_Int           boxnum,
                             NALU_HYPRE_Int           outside    )
{
   nalu_hypre_BoxArray      *grid_boxes;
   nalu_hypre_Box           *grid_box;
   NALU_HYPRE_Complex       *vecp;
   NALU_HYPRE_Int            i, istart, istop;
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_StructVectorMemoryLocation(vector);
#endif

   if (outside > 0)
   {
      grid_boxes = nalu_hypre_StructVectorDataSpace(vector);
   }
   else
   {
      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(vector));
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

   for (i = istart; i < istop; i++)
   {
      grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);

      if (nalu_hypre_IndexInBox(grid_index, grid_box))
      {
         vecp = nalu_hypre_StructVectorBoxDataValue(vector, i, grid_index);

#if defined(NALU_HYPRE_USING_GPU)
         if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
         {
            if (action > 0)
            {
#define DEVICE_VAR is_device_ptr(vecp,values)
               nalu_hypre_LoopBegin(1, k)
               {
                  *vecp += *values;
               }
               nalu_hypre_LoopEnd()
#undef DEVICE_VAR
            }
            else if (action > -1)
            {
               nalu_hypre_TMemcpy(vecp, values, NALU_HYPRE_Complex, 1, memory_location, memory_location);
            }
            else /* action < 0 */
            {
               nalu_hypre_TMemcpy(values, vecp, NALU_HYPRE_Complex, 1, memory_location, memory_location);
            }
         }
         else
#endif
         {
            if (action > 0)
            {
               *vecp += *values;
            }
            else if (action > -1)
            {
               *vecp = *values;
            }
            else /* action < 0 */
            {
               *values = *vecp;
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
 *
 * (outside > 0): set values possibly outside of the grid extents
 * (outside = 0): set values only inside the grid extents
 *
 * NOTE: Getting and setting values outside of the grid extents requires care,
 * as these values may be stored in multiple ghost zone locations.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorSetBoxValues( nalu_hypre_StructVector *vector,
                                nalu_hypre_Box          *set_box,
                                nalu_hypre_Box          *value_box,
                                NALU_HYPRE_Complex      *values,
                                NALU_HYPRE_Int           action,
                                NALU_HYPRE_Int           boxnum,
                                NALU_HYPRE_Int           outside )
{
   nalu_hypre_BoxArray     *grid_boxes;
   nalu_hypre_Box          *grid_box;
   nalu_hypre_Box          *int_box;

   nalu_hypre_BoxArray     *data_space;
   nalu_hypre_Box          *data_box;
   nalu_hypre_IndexRef      data_start;
   nalu_hypre_Index         data_stride;
   NALU_HYPRE_Complex      *datap;

   nalu_hypre_Box          *dval_box;
   nalu_hypre_Index         dval_start;
   nalu_hypre_Index         dval_stride;

   nalu_hypre_Index         loop_size;

   NALU_HYPRE_Int           i, istart, istop;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   if (outside > 0)
   {
      grid_boxes = nalu_hypre_StructVectorDataSpace(vector);
   }
   else
   {
      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(vector));
   }
   data_space = nalu_hypre_StructVectorDataSpace(vector);

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
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   nalu_hypre_SetIndex(data_stride, 1);

   int_box = nalu_hypre_BoxCreate(nalu_hypre_StructVectorNDim(vector));
   dval_box = nalu_hypre_BoxDuplicate(value_box);
   nalu_hypre_SetIndex(dval_stride, 1);

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

         datap = nalu_hypre_StructVectorBoxData(vector, i);

         nalu_hypre_BoxGetSize(int_box, loop_size);

#define DEVICE_VAR is_device_ptr(datap,values)
         if (action > 0)
         {
            nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(vector), loop_size,
                                data_box, data_start, data_stride, datai,
                                dval_box, dval_start, dval_stride, dvali);
            {
               datap[datai] += values[dvali];
            }
            nalu_hypre_BoxLoop2End(datai, dvali);
         }
         else if (action > -1)
         {
            nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(vector), loop_size,
                                data_box, data_start, data_stride, datai,
                                dval_box, dval_start, dval_stride, dvali);
            {
               datap[datai] = values[dvali];
            }
            nalu_hypre_BoxLoop2End(datai, dvali);
         }
         else /* action < 0 */
         {
            nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(vector), loop_size,
                                data_box, data_start, data_stride, datai,
                                dval_box, dval_start, dval_stride, dvali);
            {
               values[dvali] = datap[datai];
            }
            nalu_hypre_BoxLoop2End(datai, dvali);
         }
#undef DEVICE_VAR
      }
   }

   nalu_hypre_BoxDestroy(int_box);
   nalu_hypre_BoxDestroy(dval_box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * (outside > 0): clear values possibly outside of the grid extents
 * (outside = 0): clear values only inside the grid extents
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorClearValues( nalu_hypre_StructVector *vector,
                               nalu_hypre_Index         grid_index,
                               NALU_HYPRE_Int           boxnum,
                               NALU_HYPRE_Int           outside    )
{
   nalu_hypre_BoxArray      *grid_boxes;
   nalu_hypre_Box           *grid_box;
   NALU_HYPRE_Complex       *vecp;
   NALU_HYPRE_Int            i, istart, istop;
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_StructVectorMemoryLocation(vector);
#endif

   if (outside > 0)
   {
      grid_boxes = nalu_hypre_StructVectorDataSpace(vector);
   }
   else
   {
      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(vector));
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

   for (i = istart; i < istop; i++)
   {
      grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);

      if (nalu_hypre_IndexInBox(grid_index, grid_box))
      {
         vecp = nalu_hypre_StructVectorBoxDataValue(vector, i, grid_index);

#if defined(NALU_HYPRE_USING_GPU)
         if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
         {
#define DEVICE_VAR is_device_ptr(vecp)
            nalu_hypre_LoopBegin(1, k)
            {
               *vecp = 0.0;
            }
            nalu_hypre_LoopEnd()
#undef DEVICE_VAR
         }
         else
#endif
         {
            *vecp = 0.0;
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
nalu_hypre_StructVectorClearBoxValues( nalu_hypre_StructVector *vector,
                                  nalu_hypre_Box          *clear_box,
                                  NALU_HYPRE_Int           boxnum,
                                  NALU_HYPRE_Int           outside )
{
   nalu_hypre_BoxArray     *grid_boxes;
   nalu_hypre_Box          *grid_box;
   nalu_hypre_Box          *int_box;

   nalu_hypre_BoxArray     *data_space;
   nalu_hypre_Box          *data_box;
   nalu_hypre_IndexRef      data_start;
   nalu_hypre_Index         data_stride;
   NALU_HYPRE_Complex      *datap;

   nalu_hypre_Index         loop_size;

   NALU_HYPRE_Int           i, istart, istop;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   if (outside > 0)
   {
      grid_boxes = nalu_hypre_StructVectorDataSpace(vector);
   }
   else
   {
      grid_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(vector));
   }
   data_space = nalu_hypre_StructVectorDataSpace(vector);

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
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   nalu_hypre_SetIndex(data_stride, 1);

   int_box = nalu_hypre_BoxCreate(nalu_hypre_StructVectorNDim(vector));

   for (i = istart; i < istop; i++)
   {
      grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);
      data_box = nalu_hypre_BoxArrayBox(data_space, i);

      nalu_hypre_IntersectBoxes(clear_box, grid_box, int_box);

      /* if there was an intersection */
      if (nalu_hypre_BoxVolume(int_box))
      {
         data_start = nalu_hypre_BoxIMin(int_box);

         datap = nalu_hypre_StructVectorBoxData(vector, i);

         nalu_hypre_BoxGetSize(int_box, loop_size);

#define DEVICE_VAR is_device_ptr(datap)
         nalu_hypre_BoxLoop1Begin(nalu_hypre_StructVectorNDim(vector), loop_size,
                             data_box, data_start, data_stride, datai);
         {
            datap[datai] = 0.0;
         }
         nalu_hypre_BoxLoop1End(datai);
#undef DEVICE_VAR
      }
   }

   nalu_hypre_BoxDestroy(int_box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorClearAllValues( nalu_hypre_StructVector *vector )
{
   NALU_HYPRE_Complex *data      = nalu_hypre_StructVectorData(vector);
   NALU_HYPRE_Int      data_size = nalu_hypre_StructVectorDataSize(vector);
   nalu_hypre_Index    imin, imax;
   nalu_hypre_Box     *box;

   box = nalu_hypre_BoxCreate(1);
   nalu_hypre_IndexD(imin, 0) = 1;
   nalu_hypre_IndexD(imax, 0) = data_size;
   nalu_hypre_BoxSetExtents(box, imin, imax);

#define DEVICE_VAR is_device_ptr(data)
   nalu_hypre_BoxLoop1Begin(1, imax,
                       box, imin, imin, datai);
   {
      data[datai] = 0.0;
   }
   nalu_hypre_BoxLoop1End(datai);
#undef DEVICE_VAR

   nalu_hypre_BoxDestroy(box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorSetNumGhost( nalu_hypre_StructVector *vector,
                               NALU_HYPRE_Int          *num_ghost )
{
   NALU_HYPRE_Int  d, ndim = nalu_hypre_StructVectorNDim(vector);

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_StructVectorNumGhost(vector)[2 * d]     = num_ghost[2 * d];
      nalu_hypre_StructVectorNumGhost(vector)[2 * d + 1] = num_ghost[2 * d + 1];
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorSetDataSize(nalu_hypre_StructVector *vector,
                              NALU_HYPRE_Int          *data_size,
                              NALU_HYPRE_Int          *data_host_size)
{
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_StructGrid     *grid = nalu_hypre_StructVectorGrid(vector);
   if (nalu_hypre_StructGridDataLocation(grid) != NALU_HYPRE_MEMORY_HOST)
   {
      *data_size += nalu_hypre_StructVectorDataSize(vector);
   }
   else
   {
      *data_host_size += nalu_hypre_StructVectorDataSize(vector);
   }
#else
   *data_size += nalu_hypre_StructVectorDataSize(vector);
#endif
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorAssemble( nalu_hypre_StructVector *vector )
{
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * copies data from x to y
 * y has its own data array, so this is a deep copy in that sense.
 * The grid and other size information are not copied - they are
 * assumed to have already been set up to be consistent.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorCopy( nalu_hypre_StructVector *x,
                        nalu_hypre_StructVector *y )
{
   nalu_hypre_Box          *x_data_box;

   NALU_HYPRE_Complex      *xp, *yp;

   nalu_hypre_BoxArray     *boxes;
   nalu_hypre_Box          *box;
   nalu_hypre_Index         loop_size;
   nalu_hypre_IndexRef      start;
   nalu_hypre_Index         unit_stride;

   NALU_HYPRE_Int           i;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   nalu_hypre_SetIndex(unit_stride, 1);

   boxes = nalu_hypre_StructGridBoxes( nalu_hypre_StructVectorGrid(x) );
   nalu_hypre_ForBoxI(i, boxes)
   {
      box   = nalu_hypre_BoxArrayBox(boxes, i);
      start = nalu_hypre_BoxIMin(box);

      x_data_box =
         nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);
      xp = nalu_hypre_StructVectorBoxData(x, i);
      yp = nalu_hypre_StructVectorBoxData(y, i);

      nalu_hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(yp,xp)
      nalu_hypre_BoxLoop1Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                          x_data_box, start, unit_stride, vi);
      {
         yp[vi] = xp[vi];
      }
      nalu_hypre_BoxLoop1End(vi);
#undef DEVICE_VAR
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorSetConstantValues( nalu_hypre_StructVector *vector,
                                     NALU_HYPRE_Complex       values )
{
   nalu_hypre_Box          *v_data_box;

   NALU_HYPRE_Complex      *vp;

   nalu_hypre_BoxArray     *boxes;
   nalu_hypre_Box          *box;
   nalu_hypre_Index         loop_size;
   nalu_hypre_IndexRef      start;
   nalu_hypre_Index         unit_stride;

   NALU_HYPRE_Int           i;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   nalu_hypre_SetIndex(unit_stride, 1);

   boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(vector));
   nalu_hypre_ForBoxI(i, boxes)
   {
      box      = nalu_hypre_BoxArrayBox(boxes, i);
      start = nalu_hypre_BoxIMin(box);

      v_data_box =
         nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(vector), i);
      vp = nalu_hypre_StructVectorBoxData(vector, i);

      nalu_hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(vp)
      nalu_hypre_BoxLoop1Begin(nalu_hypre_StructVectorNDim(vector), loop_size,
                          v_data_box, start, unit_stride, vi);
      {
         vp[vi] = values;
      }
      nalu_hypre_BoxLoop1End(vi);
#undef DEVICE_VAR
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Takes a function pointer of the form:  NALU_HYPRE_Complex  f(i,j,k)
 * RDF: This function doesn't appear to be used anywhere.
 *--------------------------------------------------------------------------*/

/* ONLY3D */

NALU_HYPRE_Int
nalu_hypre_StructVectorSetFunctionValues( nalu_hypre_StructVector *vector,
                                     NALU_HYPRE_Complex     (*fcn)(NALU_HYPRE_Int, NALU_HYPRE_Int, NALU_HYPRE_Int) )
{
   nalu_hypre_Box          *v_data_box;

   NALU_HYPRE_Complex      *vp;

   nalu_hypre_BoxArray     *boxes;
   nalu_hypre_Box          *box;
   nalu_hypre_Index         loop_size;
   nalu_hypre_IndexRef      start;
   nalu_hypre_Index         unit_stride;

   NALU_HYPRE_Int           b, i, j, k;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   nalu_hypre_SetIndex(unit_stride, 1);

   boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(vector));
   nalu_hypre_ForBoxI(b, boxes)
   {
      box      = nalu_hypre_BoxArrayBox(boxes, b);
      start = nalu_hypre_BoxIMin(box);

      v_data_box =
         nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(vector), b);
      vp = nalu_hypre_StructVectorBoxData(vector, b);

      nalu_hypre_BoxGetSize(box, loop_size);

      i = nalu_hypre_IndexD(start, 0);
      j = nalu_hypre_IndexD(start, 1);
      k = nalu_hypre_IndexD(start, 2);

      nalu_hypre_SerialBoxLoop1Begin(nalu_hypre_StructVectorNDim(vector), loop_size,
                                v_data_box, start, unit_stride, vi)
      {
         vp[vi] = fcn(i, j, k);
         i++;
         j++;
         k++;
      }
      nalu_hypre_SerialBoxLoop1End(vi)
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorClearGhostValues( nalu_hypre_StructVector *vector )
{
   NALU_HYPRE_Int           ndim = nalu_hypre_StructVectorNDim(vector);
   nalu_hypre_Box          *v_data_box;

   NALU_HYPRE_Complex      *vp;

   nalu_hypre_BoxArray     *boxes;
   nalu_hypre_Box          *box;
   nalu_hypre_BoxArray     *diff_boxes;
   nalu_hypre_Box          *diff_box;
   nalu_hypre_Index         loop_size;
   nalu_hypre_IndexRef      start;
   nalu_hypre_Index         unit_stride;

   NALU_HYPRE_Int           i, j;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   nalu_hypre_SetIndex(unit_stride, 1);

   boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(vector));
   diff_boxes = nalu_hypre_BoxArrayCreate(0, ndim);
   nalu_hypre_ForBoxI(i, boxes)
   {
      box        = nalu_hypre_BoxArrayBox(boxes, i);
      v_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(vector), i);
      nalu_hypre_BoxArraySetSize(diff_boxes, 0);
      nalu_hypre_SubtractBoxes(v_data_box, box, diff_boxes);

      vp = nalu_hypre_StructVectorBoxData(vector, i);
      nalu_hypre_ForBoxI(j, diff_boxes)
      {
         diff_box = nalu_hypre_BoxArrayBox(diff_boxes, j);
         start = nalu_hypre_BoxIMin(diff_box);

         nalu_hypre_BoxGetSize(diff_box, loop_size);

#define DEVICE_VAR is_device_ptr(vp)
         nalu_hypre_BoxLoop1Begin(nalu_hypre_StructVectorNDim(vector), loop_size,
                             v_data_box, start, unit_stride, vi);
         {
            vp[vi] = 0.0;
         }
         nalu_hypre_BoxLoop1End(vi);
#undef DEVICE_VAR
      }
   }
   nalu_hypre_BoxArrayDestroy(diff_boxes);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * clears vector values on the physical boundaries
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorClearBoundGhostValues( nalu_hypre_StructVector *vector,
                                         NALU_HYPRE_Int           force )
{
   NALU_HYPRE_Int           ndim = nalu_hypre_StructVectorNDim(vector);
   NALU_HYPRE_Complex      *vp;
   nalu_hypre_BoxArray     *boxes;
   nalu_hypre_Box          *box;
   nalu_hypre_Box          *v_data_box;
   nalu_hypre_Index         loop_size;
   nalu_hypre_IndexRef      start;
   nalu_hypre_Index         stride;
   nalu_hypre_Box          *bbox;
   nalu_hypre_StructGrid   *grid;
   nalu_hypre_BoxArray     *boundary_boxes;
   nalu_hypre_BoxArray     *array_of_box;
   nalu_hypre_BoxArray     *work_boxarray;

   NALU_HYPRE_Int           i, i2;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   /* Only clear if not clear already or if force argument is set */
   if (nalu_hypre_StructVectorBGhostNotClear(vector) || force)
   {
      grid = nalu_hypre_StructVectorGrid(vector);
      boxes = nalu_hypre_StructGridBoxes(grid);
      nalu_hypre_SetIndex(stride, 1);

      nalu_hypre_ForBoxI(i, boxes)
      {
         box        = nalu_hypre_BoxArrayBox(boxes, i);
         boundary_boxes = nalu_hypre_BoxArrayCreate( 0, ndim );
         v_data_box =
            nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(vector), i);
         nalu_hypre_BoxBoundaryG( v_data_box, grid, boundary_boxes );
         vp = nalu_hypre_StructVectorBoxData(vector, i);

         /* box is a grid box, no ghost zones.
            v_data_box is vector data box, may or may not have ghost zones
            To get only ghost zones, subtract box from boundary_boxes.   */
         work_boxarray = nalu_hypre_BoxArrayCreate( 0, ndim );
         array_of_box = nalu_hypre_BoxArrayCreate( 1, ndim );
         nalu_hypre_BoxArrayBoxes(array_of_box)[0] = *box;
         nalu_hypre_SubtractBoxArrays( boundary_boxes, array_of_box, work_boxarray );

         nalu_hypre_ForBoxI(i2, boundary_boxes)
         {
            bbox       = nalu_hypre_BoxArrayBox(boundary_boxes, i2);
            nalu_hypre_BoxGetSize(bbox, loop_size);
            start = nalu_hypre_BoxIMin(bbox);
#define DEVICE_VAR is_device_ptr(vp)
            nalu_hypre_BoxLoop1Begin(nalu_hypre_StructVectorNDim(vector), loop_size,
                                v_data_box, start, stride, vi);
            {
               vp[vi] = 0.0;
            }
            nalu_hypre_BoxLoop1End(vi);
#undef DEVICE_VAR
         }
         nalu_hypre_BoxArrayDestroy(boundary_boxes);
         nalu_hypre_BoxArrayDestroy(work_boxarray);
         nalu_hypre_BoxArrayDestroy(array_of_box);
      }

      nalu_hypre_StructVectorBGhostNotClear(vector) = 0;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorScaleValues( nalu_hypre_StructVector *vector, NALU_HYPRE_Complex factor )
{
   NALU_HYPRE_Complex    *data;

   nalu_hypre_Index       imin;
   nalu_hypre_Index       imax;
   nalu_hypre_Box        *box;
   nalu_hypre_Index       loop_size;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   box = nalu_hypre_BoxCreate(nalu_hypre_StructVectorNDim(vector));
   nalu_hypre_SetIndex(imin, 1);
   nalu_hypre_SetIndex(imax, 1);
   nalu_hypre_IndexD(imax, 0) = nalu_hypre_StructVectorDataSize(vector);
   nalu_hypre_BoxSetExtents(box, imin, imax);
   data = nalu_hypre_StructVectorData(vector);
   nalu_hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(data)
   nalu_hypre_BoxLoop1Begin(nalu_hypre_StructVectorNDim(vector), loop_size,
                       box, imin, imin, datai);
   {
      data[datai] *= factor;
   }
   nalu_hypre_BoxLoop1End(datai);
#undef DEVICE_VAR

   nalu_hypre_BoxDestroy(box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

nalu_hypre_CommPkg *
nalu_hypre_StructVectorGetMigrateCommPkg( nalu_hypre_StructVector *from_vector,
                                     nalu_hypre_StructVector *to_vector   )
{
   nalu_hypre_CommInfo        *comm_info;
   nalu_hypre_CommPkg         *comm_pkg;

   /*------------------------------------------------------
    * Set up nalu_hypre_CommPkg
    *------------------------------------------------------*/

   nalu_hypre_CreateCommInfoFromGrids(nalu_hypre_StructVectorGrid(from_vector),
                                 nalu_hypre_StructVectorGrid(to_vector),
                                 &comm_info);
   nalu_hypre_CommPkgCreate(comm_info,
                       nalu_hypre_StructVectorDataSpace(from_vector),
                       nalu_hypre_StructVectorDataSpace(to_vector), 1, NULL, 0,
                       nalu_hypre_StructVectorComm(from_vector), &comm_pkg);
   nalu_hypre_CommInfoDestroy(comm_info);
   /* is this correct for periodic? */

   return comm_pkg;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorMigrate( nalu_hypre_CommPkg      *comm_pkg,
                           nalu_hypre_StructVector *from_vector,
                           nalu_hypre_StructVector *to_vector   )
{
   nalu_hypre_CommHandle      *comm_handle;

   /*-----------------------------------------------------------------------
    * Migrate the vector data
    *-----------------------------------------------------------------------*/

   nalu_hypre_InitializeCommunication(comm_pkg,
                                 nalu_hypre_StructVectorData(from_vector),
                                 nalu_hypre_StructVectorData(to_vector), 0, 0,
                                 &comm_handle);
   nalu_hypre_FinalizeCommunication(comm_handle);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructVectorPrintData
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorPrintData( FILE               *file,
                             nalu_hypre_StructVector *vector,
                             NALU_HYPRE_Int           all )
{
   NALU_HYPRE_Int            ndim            = nalu_hypre_StructVectorNDim(vector);
   nalu_hypre_StructGrid    *grid            = nalu_hypre_StructVectorGrid(vector);
   nalu_hypre_BoxArray      *grid_boxes      = nalu_hypre_StructGridBoxes(grid);
   nalu_hypre_BoxArray      *data_space      = nalu_hypre_StructVectorDataSpace(vector);
   NALU_HYPRE_Int            data_size       = nalu_hypre_StructVectorDataSize(vector);
   NALU_HYPRE_Complex       *data            = nalu_hypre_StructVectorData(vector);
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_StructVectorMemoryLocation(vector);
   nalu_hypre_BoxArray      *boxes;
   NALU_HYPRE_Complex       *h_data;

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
   nalu_hypre_PrintBoxArrayData(file, boxes, data_space, 1, ndim, h_data);

   /* Free memory */
   if (nalu_hypre_GetActualMemLocation(memory_location) != nalu_hypre_MEMORY_HOST)
   {
      nalu_hypre_TFree(h_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructVectorReadData
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorReadData( FILE               *file,
                            nalu_hypre_StructVector *vector )
{
   NALU_HYPRE_Int            ndim            = nalu_hypre_StructVectorNDim(vector);
   nalu_hypre_StructGrid    *grid            = nalu_hypre_StructVectorGrid(vector);
   nalu_hypre_BoxArray      *grid_boxes      = nalu_hypre_StructGridBoxes(grid);
   nalu_hypre_BoxArray      *data_space      = nalu_hypre_StructVectorDataSpace(vector);
   NALU_HYPRE_Int            data_size       = nalu_hypre_StructVectorDataSize(vector);
   NALU_HYPRE_Complex       *data            = nalu_hypre_StructVectorData(vector);
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_StructVectorMemoryLocation(vector);
   NALU_HYPRE_Complex       *h_data;

   /* Allocate/Point to data on the host memory */
   if (nalu_hypre_GetActualMemLocation(memory_location) != nalu_hypre_MEMORY_HOST)
   {
      h_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, data_size, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      h_data = data;
   }

   /* Read data from file */
   nalu_hypre_ReadBoxArrayData(file, grid_boxes, data_space, 1, ndim, h_data);

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
 * nalu_hypre_StructVectorPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructVectorPrint( const char         *filename,
                         nalu_hypre_StructVector *vector,
                         NALU_HYPRE_Int           all      )
{
   FILE              *file;
   char               new_filename[255];

   nalu_hypre_StructGrid  *grid;
   NALU_HYPRE_Int          myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

   nalu_hypre_MPI_Comm_rank(nalu_hypre_StructVectorComm(vector), &myid);
   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      nalu_hypre_printf("Error: can't open output file %s\n", new_filename);
      nalu_hypre_error_in_arg(1);

      return nalu_hypre_error_flag;
   }

   /*----------------------------------------
    * Print header info
    *----------------------------------------*/

   nalu_hypre_fprintf(file, "StructVector\n");

   /* print grid info */
   nalu_hypre_fprintf(file, "\nGrid:\n");
   grid = nalu_hypre_StructVectorGrid(vector);
   nalu_hypre_StructGridPrint(file, grid);

   /*----------------------------------------
    * Print data
    *----------------------------------------*/

   nalu_hypre_fprintf(file, "\nData:\n");
   nalu_hypre_StructVectorPrintData(file, vector, all);

   /*----------------------------------------
    * Close file
    *----------------------------------------*/

   fflush(file);
   fclose(file);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructVectorRead
 *--------------------------------------------------------------------------*/

nalu_hypre_StructVector *
nalu_hypre_StructVectorRead( MPI_Comm    comm,
                        const char *filename,
                        NALU_HYPRE_Int  *num_ghost )
{
   FILE                 *file;
   char                  new_filename[255];

   nalu_hypre_StructVector   *vector;
   nalu_hypre_StructGrid     *grid;

   NALU_HYPRE_Int             myid;

   /*----------------------------------------
    * Open file
    *----------------------------------------*/

   nalu_hypre_MPI_Comm_rank(comm, &myid);
   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      nalu_hypre_printf("Error: can't open input file %s\n", new_filename);
      nalu_hypre_error_in_arg(2);
      exit(1);
   }

   /*----------------------------------------
    * Read header info
    *----------------------------------------*/

   nalu_hypre_fscanf(file, "StructVector\n");

   /* read grid info */
   nalu_hypre_fscanf(file, "\nGrid:\n");
   nalu_hypre_StructGridRead(comm, file, &grid);

   /*----------------------------------------
    * Initialize the vector
    *----------------------------------------*/

   vector = nalu_hypre_StructVectorCreate(comm, grid);
   nalu_hypre_StructVectorSetNumGhost(vector, num_ghost);
   nalu_hypre_StructVectorInitialize(vector);

   /*----------------------------------------
    * Read data
    *----------------------------------------*/

   nalu_hypre_fscanf(file, "\nData:\n");
   nalu_hypre_StructVectorReadData(file, vector);

   /*----------------------------------------
    * Assemble the vector
    *----------------------------------------*/

   nalu_hypre_StructVectorAssemble(vector);

   /*----------------------------------------
    * Close file
    *----------------------------------------*/

   fclose(file);

   return vector;
}

/*--------------------------------------------------------------------------
 * The following is used only as a debugging aid.
 *
 * nalu_hypre_StructVectorClone
 * Returns a complete copy of x - a deep copy, with its own copy of the data.
 *--------------------------------------------------------------------------*/

nalu_hypre_StructVector *
nalu_hypre_StructVectorClone(nalu_hypre_StructVector *x)
{
   MPI_Comm             comm            = nalu_hypre_StructVectorComm(x);
   nalu_hypre_StructGrid    *grid            = nalu_hypre_StructVectorGrid(x);
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_StructVectorMemoryLocation(x);
   nalu_hypre_BoxArray      *data_space      = nalu_hypre_StructVectorDataSpace(x);
   NALU_HYPRE_Int           *data_indices    = nalu_hypre_StructVectorDataIndices(x);
   NALU_HYPRE_Int            data_size       = nalu_hypre_StructVectorDataSize(x);
   NALU_HYPRE_Int            ndim            = nalu_hypre_StructGridNDim(grid);
   NALU_HYPRE_Int            data_space_size = nalu_hypre_BoxArraySize(data_space);
   nalu_hypre_StructVector  *y               = nalu_hypre_StructVectorCreate(comm, grid);
   NALU_HYPRE_Int            i;

   nalu_hypre_StructVectorDataSize(y)    = data_size;
   nalu_hypre_StructVectorDataSpace(y)   = nalu_hypre_BoxArrayDuplicate(data_space);
   nalu_hypre_StructVectorData(y)        = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, data_size, memory_location);
   nalu_hypre_StructVectorDataIndices(y) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, data_space_size, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < data_space_size; i++)
   {
      nalu_hypre_StructVectorDataIndices(y)[i] = data_indices[i];
   }

   nalu_hypre_StructVectorCopy( x, y );

   for (i = 0; i < 2 * ndim; i++)
   {
      nalu_hypre_StructVectorNumGhost(y)[i] = nalu_hypre_StructVectorNumGhost(x)[i];
   }

   nalu_hypre_StructVectorBGhostNotClear(y) = nalu_hypre_StructVectorBGhostNotClear(x);
   nalu_hypre_StructVectorGlobalSize(y) = nalu_hypre_StructVectorGlobalSize(x);

   return y;
}
