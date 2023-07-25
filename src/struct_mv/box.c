/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_Box class:
 *   Basic class functions.
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"

/*==========================================================================
 * Member functions: nalu_hypre_Index
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SetIndex( nalu_hypre_Index  index,
                NALU_HYPRE_Int    val )
{
   NALU_HYPRE_Int d;

   for (d = 0; d < NALU_HYPRE_MAXDIM; d++)
   {
      nalu_hypre_IndexD(index, d) = val;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CopyIndex( nalu_hypre_Index  in_index,
                 nalu_hypre_Index  out_index )
{
   NALU_HYPRE_Int d;

   for (d = 0; d < NALU_HYPRE_MAXDIM; d++)
   {
      nalu_hypre_IndexD(out_index, d) = nalu_hypre_IndexD(in_index, d);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CopyToCleanIndex( nalu_hypre_Index  in_index,
                        NALU_HYPRE_Int    ndim,
                        nalu_hypre_Index  out_index )
{
   NALU_HYPRE_Int d;
   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_IndexD(out_index, d) = nalu_hypre_IndexD(in_index, d);
   }
   for (d = ndim; d < NALU_HYPRE_MAXDIM; d++)
   {
      nalu_hypre_IndexD(out_index, d) = 0;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IndexEqual( nalu_hypre_Index  index,
                  NALU_HYPRE_Int    val,
                  NALU_HYPRE_Int    ndim )
{
   NALU_HYPRE_Int d, equal;

   equal = 1;
   for (d = 0; d < ndim; d++)
   {
      if (nalu_hypre_IndexD(index, d) != val)
      {
         equal = 0;
         break;
      }
   }

   return equal;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IndexMin( nalu_hypre_Index  index,
                NALU_HYPRE_Int    ndim )
{
   NALU_HYPRE_Int d, min;

   min = nalu_hypre_IndexD(index, 0);
   for (d = 1; d < ndim; d++)
   {
      if (nalu_hypre_IndexD(index, d) < min)
      {
         min = nalu_hypre_IndexD(index, d);
      }
   }

   return min;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IndexMax( nalu_hypre_Index  index,
                NALU_HYPRE_Int    ndim )
{
   NALU_HYPRE_Int d, max;

   max = nalu_hypre_IndexD(index, 0);
   for (d = 1; d < ndim; d++)
   {
      if (nalu_hypre_IndexD(index, d) < max)
      {
         max = nalu_hypre_IndexD(index, d);
      }
   }

   return max;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AddIndexes( nalu_hypre_Index  index1,
                  nalu_hypre_Index  index2,
                  NALU_HYPRE_Int    ndim,
                  nalu_hypre_Index  result )
{
   NALU_HYPRE_Int d;

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_IndexD(result, d) = nalu_hypre_IndexD(index1, d) + nalu_hypre_IndexD(index2, d);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SubtractIndexes( nalu_hypre_Index  index1,
                       nalu_hypre_Index  index2,
                       NALU_HYPRE_Int    ndim,
                       nalu_hypre_Index  result )
{
   NALU_HYPRE_Int d;

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_IndexD(result, d) = nalu_hypre_IndexD(index1, d) - nalu_hypre_IndexD(index2, d);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IndexesEqual( nalu_hypre_Index  index1,
                    nalu_hypre_Index  index2,
                    NALU_HYPRE_Int    ndim )
{
   NALU_HYPRE_Int d, equal;

   equal = 1;
   for (d = 0; d < ndim; d++)
   {
      if (nalu_hypre_IndexD(index1, d) != nalu_hypre_IndexD(index2, d))
      {
         equal = 0;
         break;
      }
   }

   return equal;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IndexPrint( FILE        *file,
                  NALU_HYPRE_Int    ndim,
                  nalu_hypre_Index  index )
{
   NALU_HYPRE_Int d;

   nalu_hypre_fprintf(file, "[%d", nalu_hypre_IndexD(index, 0));
   for (d = 1; d < ndim; d++)
   {
      nalu_hypre_fprintf(file, " %d", nalu_hypre_IndexD(index, d));
   }
   nalu_hypre_fprintf(file, "]");

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IndexRead( FILE        *file,
                 NALU_HYPRE_Int    ndim,
                 nalu_hypre_Index  index )
{
   NALU_HYPRE_Int d;

   nalu_hypre_fscanf(file, "[%d", &nalu_hypre_IndexD(index, 0));
   for (d = 1; d < ndim; d++)
   {
      nalu_hypre_fscanf(file, " %d", &nalu_hypre_IndexD(index, d));
   }
   nalu_hypre_fscanf(file, "]");

   for (d = ndim; d < NALU_HYPRE_MAXDIM; d++)
   {
      nalu_hypre_IndexD(index, d) = 0;
   }

   return nalu_hypre_error_flag;
}

/*==========================================================================
 * Member functions: nalu_hypre_Box
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

nalu_hypre_Box *
nalu_hypre_BoxCreate( NALU_HYPRE_Int  ndim )
{
   nalu_hypre_Box *box;

   box = nalu_hypre_CTAlloc(nalu_hypre_Box,  1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_BoxNDim(box) = ndim;

   return box;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxDestroy( nalu_hypre_Box *box )
{
   if (box)
   {
      nalu_hypre_TFree(box, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This is used to initialize ndim when the box has static storage
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxInit( nalu_hypre_Box *box,
               NALU_HYPRE_Int  ndim )
{
   nalu_hypre_BoxNDim(box) = ndim;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxSetExtents( nalu_hypre_Box  *box,
                     nalu_hypre_Index imin,
                     nalu_hypre_Index imax )
{
   nalu_hypre_CopyIndex(imin, nalu_hypre_BoxIMin(box));
   nalu_hypre_CopyIndex(imax, nalu_hypre_BoxIMax(box));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CopyBox( nalu_hypre_Box  *box1,
               nalu_hypre_Box  *box2 )
{
   nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(box1), nalu_hypre_BoxIMin(box2));
   nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(box1), nalu_hypre_BoxIMax(box2));
   nalu_hypre_BoxNDim(box2) = nalu_hypre_BoxNDim(box1);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return a duplicate box.
 *--------------------------------------------------------------------------*/

nalu_hypre_Box *
nalu_hypre_BoxDuplicate( nalu_hypre_Box *box )
{
   nalu_hypre_Box  *new_box;

   new_box = nalu_hypre_BoxCreate(nalu_hypre_BoxNDim(box));
   nalu_hypre_CopyBox(box, new_box);

   return new_box;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxVolume( nalu_hypre_Box *box )
{
   NALU_HYPRE_Int volume, d, ndim = nalu_hypre_BoxNDim(box);

   volume = 1;
   for (d = 0; d < ndim; d++)
   {
      volume *= nalu_hypre_BoxSizeD(box, d);
   }

   return volume;
}

/*--------------------------------------------------------------------------
 * To prevent overflow when needed
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real
nalu_hypre_doubleBoxVolume( nalu_hypre_Box *box )
{
   NALU_HYPRE_Real    volume;
   NALU_HYPRE_Int d, ndim = nalu_hypre_BoxNDim(box);

   volume = 1.0;
   for (d = 0; d < ndim; d++)
   {
      volume *= nalu_hypre_BoxSizeD(box, d);
   }

   return volume;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IndexInBox( nalu_hypre_Index   index,
                  nalu_hypre_Box    *box )
{
   NALU_HYPRE_Int d, inbox, ndim = nalu_hypre_BoxNDim(box);

   inbox = 1;
   for (d = 0; d < ndim; d++)
   {
      if (!nalu_hypre_IndexDInBox(index, d, box))
      {
         inbox = 0;
         break;
      }
   }

   return inbox;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxGetSize( nalu_hypre_Box   *box,
                  nalu_hypre_Index  size )
{
   NALU_HYPRE_Int d, ndim = nalu_hypre_BoxNDim(box);

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_IndexD(size, d) = nalu_hypre_BoxSizeD(box, d);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxGetStrideSize( nalu_hypre_Box   *box,
                        nalu_hypre_Index  stride,
                        nalu_hypre_Index  size   )
{
   NALU_HYPRE_Int  d, s, ndim = nalu_hypre_BoxNDim(box);

   for (d = 0; d < ndim; d++)
   {
      s = nalu_hypre_BoxSizeD(box, d);
      if (s > 0)
      {
         s = (s - 1) / nalu_hypre_IndexD(stride, d) + 1;
      }
      nalu_hypre_IndexD(size, d) = s;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxGetStrideVolume( nalu_hypre_Box   *box,
                          nalu_hypre_Index  stride,
                          NALU_HYPRE_Int   *volume_ptr )
{
   NALU_HYPRE_Int  volume, d, s, ndim = nalu_hypre_BoxNDim(box);

   volume = 1;
   for (d = 0; d < ndim; d++)
   {
      s = nalu_hypre_BoxSizeD(box, d);
      if (s > 0)
      {
         s = (s - 1) / nalu_hypre_IndexD(stride, d) + 1;
      }
      volume *= s;
   }

   *volume_ptr = volume;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Returns the rank of an index into a multi-D box where the assumed ordering is
 * dimension 0 first, then dimension 1, etc.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxIndexRank( nalu_hypre_Box   *box,
                    nalu_hypre_Index  index )
{
   NALU_HYPRE_Int  rank, size, d, ndim = nalu_hypre_BoxNDim(box);

   rank = 0;
   size = 1;
   for (d = 0; d < ndim; d++)
   {
      rank += (nalu_hypre_IndexD(index, d) - nalu_hypre_BoxIMinD(box, d)) * size;
      size *= nalu_hypre_BoxSizeD(box, d);
   }

   return rank;
}

/*--------------------------------------------------------------------------
 * Computes an index into a multi-D box from a rank where the assumed ordering
 * is dimension 0 first, then dimension 1, etc.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxRankIndex( nalu_hypre_Box   *box,
                    NALU_HYPRE_Int    rank,
                    nalu_hypre_Index  index )
{
   NALU_HYPRE_Int  d, r, s, ndim = nalu_hypre_BoxNDim(box);

   r = rank;
   s = nalu_hypre_BoxVolume(box);
   for (d = ndim - 1; d >= 0; d--)
   {
      s = s / nalu_hypre_BoxSizeD(box, d);
      nalu_hypre_IndexD(index, d) = r / s;
      nalu_hypre_IndexD(index, d) += nalu_hypre_BoxIMinD(box, d);
      r = r % s;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Returns the distance of an index offset in a multi-D box where the assumed
 * ordering is dimension 0 first, then dimension 1, etc.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxOffsetDistance( nalu_hypre_Box   *box,
                         nalu_hypre_Index  index )
{
   NALU_HYPRE_Int  dist, size, d, ndim = nalu_hypre_BoxNDim(box);

   dist = 0;
   size = 1;
   for (d = 0; d < ndim; d++)
   {
      dist += nalu_hypre_IndexD(index, d) * size;
      size *= nalu_hypre_BoxSizeD(box, d);
   }

   return dist;
}

/*--------------------------------------------------------------------------
 * Shift a box by a positive shift
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxShiftPos( nalu_hypre_Box   *box,
                   nalu_hypre_Index  shift )
{
   NALU_HYPRE_Int  d, ndim = nalu_hypre_BoxNDim(box);

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_BoxIMinD(box, d) += nalu_hypre_IndexD(shift, d);
      nalu_hypre_BoxIMaxD(box, d) += nalu_hypre_IndexD(shift, d);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Shift a box by a negative shift
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxShiftNeg( nalu_hypre_Box   *box,
                   nalu_hypre_Index  shift )
{
   NALU_HYPRE_Int  d, ndim = nalu_hypre_BoxNDim(box);

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_BoxIMinD(box, d) -= nalu_hypre_IndexD(shift, d);
      nalu_hypre_BoxIMaxD(box, d) -= nalu_hypre_IndexD(shift, d);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Grow a box outward in each dimension as specified by index
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxGrowByIndex( nalu_hypre_Box   *box,
                      nalu_hypre_Index  index )
{
   nalu_hypre_IndexRef  imin = nalu_hypre_BoxIMin(box);
   nalu_hypre_IndexRef  imax = nalu_hypre_BoxIMax(box);
   NALU_HYPRE_Int       ndim = nalu_hypre_BoxNDim(box);
   NALU_HYPRE_Int       d, i;

   for (d = 0; d < ndim; d++)
   {
      i = nalu_hypre_IndexD(index, d);
      nalu_hypre_IndexD(imin, d) -= i;
      nalu_hypre_IndexD(imax, d) += i;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Grow a box outward by val in each dimension
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxGrowByValue( nalu_hypre_Box  *box,
                      NALU_HYPRE_Int   val )
{
   NALU_HYPRE_Int  *imin = nalu_hypre_BoxIMin(box);
   NALU_HYPRE_Int  *imax = nalu_hypre_BoxIMax(box);
   NALU_HYPRE_Int   ndim = nalu_hypre_BoxNDim(box);
   NALU_HYPRE_Int  d;

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_IndexD(imin, d) -= val;
      nalu_hypre_IndexD(imax, d) += val;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Grow a box as specified by array
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxGrowByArray( nalu_hypre_Box  *box,
                      NALU_HYPRE_Int  *array )
{
   NALU_HYPRE_Int  *imin = nalu_hypre_BoxIMin(box);
   NALU_HYPRE_Int  *imax = nalu_hypre_BoxIMax(box);
   NALU_HYPRE_Int   ndim = nalu_hypre_BoxNDim(box);
   NALU_HYPRE_Int   d;

   for (d = 0; d < ndim; d++)
   {
      imin[d] -= array[2 * d];
      imax[d] += array[2 * d + 1];
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Print a box to file
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxPrint( FILE      *file,
                nalu_hypre_Box *box )
{
   NALU_HYPRE_Int   ndim = nalu_hypre_BoxNDim(box);
   NALU_HYPRE_Int   d;

   nalu_hypre_fprintf(file, "(%d", nalu_hypre_BoxIMinD(box, 0));
   for (d = 1; d < ndim; d++)
   {
      nalu_hypre_fprintf(file, ", %d", nalu_hypre_BoxIMinD(box, d));
   }
   nalu_hypre_fprintf(file, ") x (%d", nalu_hypre_BoxIMaxD(box, 0));
   for (d = 1; d < ndim; d++)
   {
      nalu_hypre_fprintf(file, ", %d", nalu_hypre_BoxIMaxD(box, d));
   }
   nalu_hypre_fprintf(file, ")");

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Read a box from file
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxRead( FILE       *file,
               NALU_HYPRE_Int   ndim,
               nalu_hypre_Box **box_ptr )
{
   nalu_hypre_Box  *box;
   NALU_HYPRE_Int   d;

   /* Don't create a new box if the output box already exists */
   if (*box_ptr)
   {
      box = *box_ptr;
      nalu_hypre_BoxInit(box, ndim);
   }
   else
   {
      box = nalu_hypre_BoxCreate(ndim);
   }

   nalu_hypre_fscanf(file, "(%d", &nalu_hypre_BoxIMinD(box, 0));
   for (d = 1; d < ndim; d++)
   {
      nalu_hypre_fscanf(file, ", %d", &nalu_hypre_BoxIMinD(box, d));
   }
   nalu_hypre_fscanf(file, ") x (%d", &nalu_hypre_BoxIMaxD(box, 0));
   for (d = 1; d < ndim; d++)
   {
      nalu_hypre_fscanf(file, ", %d", &nalu_hypre_BoxIMaxD(box, d));
   }
   nalu_hypre_fscanf(file, ")");

   *box_ptr = box;

   return nalu_hypre_error_flag;
}

/*==========================================================================
 * Member functions: nalu_hypre_BoxArray
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

nalu_hypre_BoxArray *
nalu_hypre_BoxArrayCreate( NALU_HYPRE_Int size,
                      NALU_HYPRE_Int ndim )
{
   NALU_HYPRE_Int       i;
   nalu_hypre_Box      *box;
   nalu_hypre_BoxArray *box_array;

   box_array = nalu_hypre_TAlloc(nalu_hypre_BoxArray,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_BoxArrayBoxes(box_array)     = nalu_hypre_CTAlloc(nalu_hypre_Box,  size, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_BoxArraySize(box_array)      = size;
   nalu_hypre_BoxArrayAllocSize(box_array) = size;
   nalu_hypre_BoxArrayNDim(box_array)      = ndim;
   for (i = 0; i < size; i++)
   {
      box = nalu_hypre_BoxArrayBox(box_array, i);
      nalu_hypre_BoxNDim(box) = ndim;
   }

   return box_array;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxArrayDestroy( nalu_hypre_BoxArray *box_array )
{
   if (box_array)
   {
      nalu_hypre_TFree(nalu_hypre_BoxArrayBoxes(box_array), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(box_array, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxArraySetSize( nalu_hypre_BoxArray  *box_array,
                       NALU_HYPRE_Int        size      )
{
   NALU_HYPRE_Int  alloc_size;

   alloc_size = nalu_hypre_BoxArrayAllocSize(box_array);

   if (size > alloc_size)
   {
      NALU_HYPRE_Int  i, old_alloc_size, ndim = nalu_hypre_BoxArrayNDim(box_array);
      nalu_hypre_Box *box;

      old_alloc_size = alloc_size;
      alloc_size = size + nalu_hypre_BoxArrayExcess;
      nalu_hypre_BoxArrayBoxes(box_array) =
         nalu_hypre_TReAlloc(nalu_hypre_BoxArrayBoxes(box_array),  nalu_hypre_Box,  alloc_size, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_BoxArrayAllocSize(box_array) = alloc_size;

      for (i = old_alloc_size; i < alloc_size; i++)
      {
         box = nalu_hypre_BoxArrayBox(box_array, i);
         nalu_hypre_BoxNDim(box) = ndim;
      }
   }

   nalu_hypre_BoxArraySize(box_array) = size;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return a duplicate box_array.
 *--------------------------------------------------------------------------*/

nalu_hypre_BoxArray *
nalu_hypre_BoxArrayDuplicate( nalu_hypre_BoxArray *box_array )
{
   nalu_hypre_BoxArray  *new_box_array;

   NALU_HYPRE_Int        i;

   new_box_array = nalu_hypre_BoxArrayCreate(
                      nalu_hypre_BoxArraySize(box_array), nalu_hypre_BoxArrayNDim(box_array));
   nalu_hypre_ForBoxI(i, box_array)
   {
      nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(box_array, i),
                    nalu_hypre_BoxArrayBox(new_box_array, i));
   }

   return new_box_array;
}

/*--------------------------------------------------------------------------
 * Append box to the end of box_array.
 * The box_array may be empty.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AppendBox( nalu_hypre_Box      *box,
                 nalu_hypre_BoxArray *box_array )
{
   NALU_HYPRE_Int  size;

   size = nalu_hypre_BoxArraySize(box_array);
   nalu_hypre_BoxArraySetSize(box_array, (size + 1));
   nalu_hypre_CopyBox(box, nalu_hypre_BoxArrayBox(box_array, size));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Delete box from box_array.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_DeleteBox( nalu_hypre_BoxArray *box_array,
                 NALU_HYPRE_Int       index     )
{
   NALU_HYPRE_Int  i;

   for (i = index; i < nalu_hypre_BoxArraySize(box_array) - 1; i++)
   {
      nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(box_array, i + 1),
                    nalu_hypre_BoxArrayBox(box_array, i));
   }

   nalu_hypre_BoxArraySize(box_array) --;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Deletes boxes corrsponding to indices from box_array.
 * Assumes indices are in ascending order. (AB 11/04)
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_DeleteMultipleBoxes( nalu_hypre_BoxArray *box_array,
                           NALU_HYPRE_Int*  indices,
                           NALU_HYPRE_Int num )
{
   NALU_HYPRE_Int  i, j, start, array_size;

   if (num < 1)
   {
      return nalu_hypre_error_flag;
   }

   array_size =  nalu_hypre_BoxArraySize(box_array);
   start = indices[0];
   j = 0;

   for (i = start; (i + j) < array_size; i++)
   {
      if (j < num)
      {
         while ((i + j) == indices[j]) /* see if deleting consecutive items */
         {
            j++; /*increase the shift*/
            if (j == num) { break; }
         }
      }

      if ( (i + j) < array_size) /* if deleting the last item then no moving */
      {
         nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(box_array, i + j),
                       nalu_hypre_BoxArrayBox(box_array, i));
      }
   }

   nalu_hypre_BoxArraySize(box_array) = array_size - num;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Append box_array_0 to the end of box_array_1.
 * The box_array_1 may be empty.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AppendBoxArray( nalu_hypre_BoxArray *box_array_0,
                      nalu_hypre_BoxArray *box_array_1 )
{
   NALU_HYPRE_Int  size, size_0;
   NALU_HYPRE_Int  i;

   size   = nalu_hypre_BoxArraySize(box_array_1);
   size_0 = nalu_hypre_BoxArraySize(box_array_0);
   nalu_hypre_BoxArraySetSize(box_array_1, (size + size_0));

   /* copy box_array_0 boxes into box_array_1 */
   for (i = 0; i < size_0; i++)
   {
      nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(box_array_0, i),
                    nalu_hypre_BoxArrayBox(box_array_1, size + i));
   }

   return nalu_hypre_error_flag;
}

/*==========================================================================
 * Member functions: nalu_hypre_BoxArrayArray
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

nalu_hypre_BoxArrayArray *
nalu_hypre_BoxArrayArrayCreate( NALU_HYPRE_Int size,
                           NALU_HYPRE_Int ndim )
{
   nalu_hypre_BoxArrayArray  *box_array_array;
   NALU_HYPRE_Int             i;

   box_array_array = nalu_hypre_CTAlloc(nalu_hypre_BoxArrayArray,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_BoxArrayArrayBoxArrays(box_array_array) =
      nalu_hypre_CTAlloc(nalu_hypre_BoxArray *,  size, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < size; i++)
   {
      nalu_hypre_BoxArrayArrayBoxArray(box_array_array, i) =
         nalu_hypre_BoxArrayCreate(0, ndim);
   }
   nalu_hypre_BoxArrayArraySize(box_array_array) = size;
   nalu_hypre_BoxArrayArrayNDim(box_array_array) = ndim;

   return box_array_array;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoxArrayArrayDestroy( nalu_hypre_BoxArrayArray *box_array_array )
{
   NALU_HYPRE_Int  i;

   if (box_array_array)
   {
      nalu_hypre_ForBoxArrayI(i, box_array_array)
      nalu_hypre_BoxArrayDestroy(
         nalu_hypre_BoxArrayArrayBoxArray(box_array_array, i));

      nalu_hypre_TFree(nalu_hypre_BoxArrayArrayBoxArrays(box_array_array), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(box_array_array, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return a duplicate box_array_array.
 *--------------------------------------------------------------------------*/

nalu_hypre_BoxArrayArray *
nalu_hypre_BoxArrayArrayDuplicate( nalu_hypre_BoxArrayArray *box_array_array )
{
   nalu_hypre_BoxArrayArray  *new_box_array_array;
   nalu_hypre_BoxArray      **new_box_arrays;
   NALU_HYPRE_Int             new_size;

   nalu_hypre_BoxArray      **box_arrays;
   NALU_HYPRE_Int             i;

   new_size = nalu_hypre_BoxArrayArraySize(box_array_array);
   new_box_array_array = nalu_hypre_BoxArrayArrayCreate(
                            new_size, nalu_hypre_BoxArrayArrayNDim(box_array_array));

   if (new_size)
   {
      new_box_arrays = nalu_hypre_BoxArrayArrayBoxArrays(new_box_array_array);
      box_arrays     = nalu_hypre_BoxArrayArrayBoxArrays(box_array_array);

      for (i = 0; i < new_size; i++)
      {
         nalu_hypre_AppendBoxArray(box_arrays[i], new_box_arrays[i]);
      }
   }

   return new_box_array_array;
}
