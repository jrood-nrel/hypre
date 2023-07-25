/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_Box class:
 *   Box algebra functions.
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * Intersect box1 and box2.
 * If the boxes do not intersect, the result is a box with zero volume.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IntersectBoxes( nalu_hypre_Box *box1,
                      nalu_hypre_Box *box2,
                      nalu_hypre_Box *ibox )
{
   NALU_HYPRE_Int d, ndim = nalu_hypre_BoxNDim(box1);

   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_BoxIMinD(ibox, d) =
         nalu_hypre_max(nalu_hypre_BoxIMinD(box1, d), nalu_hypre_BoxIMinD(box2, d));
      nalu_hypre_BoxIMaxD(ibox, d) =
         nalu_hypre_min(nalu_hypre_BoxIMaxD(box1, d), nalu_hypre_BoxIMaxD(box2, d));
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Compute (box1 - box2) and append result to box_array.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SubtractBoxes( nalu_hypre_Box      *box1,
                     nalu_hypre_Box      *box2,
                     nalu_hypre_BoxArray *box_array )
{
   NALU_HYPRE_Int   d, size, maxboxes, ndim = nalu_hypre_BoxNDim(box1);
   nalu_hypre_Box  *box;
   nalu_hypre_Box  *rembox;

   /*------------------------------------------------------
    * Set the box array size to the maximum possible,
    * plus one, to have space for the remainder box.
    *------------------------------------------------------*/

   maxboxes = 2 * ndim;

   size = nalu_hypre_BoxArraySize(box_array);
   nalu_hypre_BoxArraySetSize(box_array, (size + maxboxes + 1));

   /*------------------------------------------------------
    * Subtract the boxes by cutting box1 in x, y, then z
    *------------------------------------------------------*/

   rembox = nalu_hypre_BoxArrayBox(box_array, (size + maxboxes));
   nalu_hypre_CopyBox(box1, rembox);

   for (d = 0; d < ndim; d++)
   {
      /* if the boxes do not intersect, the subtraction is trivial */
      if ( (nalu_hypre_BoxIMinD(box2, d) > nalu_hypre_BoxIMaxD(rembox, d)) ||
           (nalu_hypre_BoxIMaxD(box2, d) < nalu_hypre_BoxIMinD(rembox, d)) )
      {
         size = nalu_hypre_BoxArraySize(box_array) - maxboxes - 1;
         nalu_hypre_CopyBox(box1, nalu_hypre_BoxArrayBox(box_array, size));
         size++;
         break;
      }

      /* update the box array */
      else
      {
         if ( nalu_hypre_BoxIMinD(box2, d) > nalu_hypre_BoxIMinD(rembox, d) )
         {
            box = nalu_hypre_BoxArrayBox(box_array, size);
            nalu_hypre_CopyBox(rembox, box);
            nalu_hypre_BoxIMaxD(box, d) = nalu_hypre_BoxIMinD(box2, d) - 1;
            nalu_hypre_BoxIMinD(rembox, d) = nalu_hypre_BoxIMinD(box2, d);
            if ( nalu_hypre_BoxVolume(box) > 0 ) { size++; }
         }
         if ( nalu_hypre_BoxIMaxD(box2, d) < nalu_hypre_BoxIMaxD(rembox, d) )
         {
            box = nalu_hypre_BoxArrayBox(box_array, size);
            nalu_hypre_CopyBox(rembox, box);
            nalu_hypre_BoxIMinD(box, d) = nalu_hypre_BoxIMaxD(box2, d) + 1;
            nalu_hypre_BoxIMaxD(rembox, d) = nalu_hypre_BoxIMaxD(box2, d);
            if ( nalu_hypre_BoxVolume(box) > 0 ) { size++; }
         }
      }
   }
   nalu_hypre_BoxArraySetSize(box_array, size);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Compute (box_array1 - box_array2) and replace box_array1 with result.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SubtractBoxArrays( nalu_hypre_BoxArray *box_array1,
                         nalu_hypre_BoxArray *box_array2,
                         nalu_hypre_BoxArray *tmp_box_array )
{
   nalu_hypre_BoxArray *diff_boxes     = box_array1;
   nalu_hypre_BoxArray *new_diff_boxes = tmp_box_array;
   nalu_hypre_BoxArray  box_array;
   nalu_hypre_Box      *box1;
   nalu_hypre_Box      *box2;
   NALU_HYPRE_Int       i, k;

   nalu_hypre_ForBoxI(i, box_array2)
   {
      box2 = nalu_hypre_BoxArrayBox(box_array2, i);

      /* compute new_diff_boxes = (diff_boxes - box2) */
      nalu_hypre_BoxArraySetSize(new_diff_boxes, 0);
      nalu_hypre_ForBoxI(k, diff_boxes)
      {
         box1 = nalu_hypre_BoxArrayBox(diff_boxes, k);
         nalu_hypre_SubtractBoxes(box1, box2, new_diff_boxes);
      }

      /* swap internals of diff_boxes and new_diff_boxes */
      box_array       = *new_diff_boxes;
      *new_diff_boxes = *diff_boxes;
      *diff_boxes     = box_array;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NOTE: Avoid using - this only works for ndim < 4
 *
 * Compute the union of all boxes.
 *
 * To compute the union, we first construct a logically rectangular,
 * variably spaced, 3D grid called block.  Each cell (i,j,k) of block
 * corresponds to a box with extents given by
 *
 *   iminx = block_index[0][i]
 *   iminy = block_index[1][j]
 *   iminz = block_index[2][k]
 *   imaxx = block_index[0][i+1] - 1
 *   imaxy = block_index[1][j+1] - 1
 *   imaxz = block_index[2][k+1] - 1
 *
 * The size of block is given by
 *
 *   sizex = block_sz[0]
 *   sizey = block_sz[1]
 *   sizez = block_sz[2]
 *
 * We initially set all cells of block that are part of the union to
 *
 *   factor[2] + factor[1] + factor[0]
 *
 * where
 *
 *   factor[0] = 1;
 *   factor[1] = (block_sz[0] + 1);
 *   factor[2] = (block_sz[1] + 1) * factor[1];
 *
 * The cells of block are then "joined" in x first, then y, then z.
 * The result is that each nonzero entry of block corresponds to a
 * box in the union with extents defined by factoring the entry, then
 * indexing into the block_index array.
 *
 * Note: Special care has to be taken for boxes of size 0.
 *
 *--------------------------------------------------------------------------*/

/* ONLY3D */

NALU_HYPRE_Int
nalu_hypre_UnionBoxes( nalu_hypre_BoxArray *boxes )
{
   nalu_hypre_Box       *box;

   NALU_HYPRE_Int       *block_index[3];
   NALU_HYPRE_Int        block_sz[3], block_volume;
   NALU_HYPRE_Int       *block;
   NALU_HYPRE_Int        index;
   NALU_HYPRE_Int        size;
   NALU_HYPRE_Int        factor[3];

   NALU_HYPRE_Int        iminmax[2], imin[3], imax[3];
   NALU_HYPRE_Int        ii[3], dd[3];
   NALU_HYPRE_Int        join;
   NALU_HYPRE_Int        i_tmp0, i_tmp1;
   NALU_HYPRE_Int        ioff, joff, koff;
   NALU_HYPRE_Int        bi, d, i, j, k;

   NALU_HYPRE_Int        index_not_there;

   /*------------------------------------------------------
    * If the size of boxes is less than 2, return
    *------------------------------------------------------*/

   if (nalu_hypre_BoxArraySize(boxes) < 2)
   {
      return nalu_hypre_error_flag;
   }

   /*------------------------------------------------------
    * Set up the block_index array
    *------------------------------------------------------*/

   i_tmp0 = 2 * nalu_hypre_BoxArraySize(boxes);
   block_index[0] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  3 * i_tmp0, NALU_HYPRE_MEMORY_HOST);
   block_sz[0] = 0;
   for (d = 1; d < 3; d++)
   {
      block_index[d] = block_index[d - 1] + i_tmp0;
      block_sz[d] = 0;
   }

   nalu_hypre_ForBoxI(bi, boxes)
   {
      box = nalu_hypre_BoxArrayBox(boxes, bi);

      for (d = 0; d < 3; d++)
      {
         iminmax[0] = nalu_hypre_BoxIMinD(box, d);
         iminmax[1] = nalu_hypre_BoxIMaxD(box, d) + 1;

         for (i = 0; i < 2; i++)
         {
            /* find the new index position in the block_index array */
            index_not_there = 1;
            for (j = 0; j < block_sz[d]; j++)
            {
               if (iminmax[i] <= block_index[d][j])
               {
                  if (iminmax[i] == block_index[d][j])
                  {
                     index_not_there = 0;
                  }
                  break;
               }
            }

            /* if the index is already there, don't add it again */
            if (index_not_there)
            {
               for (k = block_sz[d]; k > j; k--)
               {
                  block_index[d][k] = block_index[d][k - 1];
               }
               block_index[d][j] = iminmax[i];
               block_sz[d]++;
            }
         }
      }
   }

   for (d = 0; d < 3; d++)
   {
      block_sz[d]--;
   }
   block_volume = block_sz[0] * block_sz[1] * block_sz[2];

   /*------------------------------------------------------
    * Set factor values
    *------------------------------------------------------*/

   factor[0] = 1;
   factor[1] = (block_sz[0] + 1);
   factor[2] = (block_sz[1] + 1) * factor[1];

   /*------------------------------------------------------
    * Set up the block array
    *------------------------------------------------------*/

   block = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  block_volume, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ForBoxI(bi, boxes)
   {
      box = nalu_hypre_BoxArrayBox(boxes, bi);

      /* find the block_index indices corresponding to the current box */
      for (d = 0; d < 3; d++)
      {
         j = 0;

         while (nalu_hypre_BoxIMinD(box, d) != block_index[d][j])
         {
            j++;
         }
         imin[d] = j;

         while (nalu_hypre_BoxIMaxD(box, d) + 1 != block_index[d][j])
         {
            j++;
         }
         imax[d] = j;
      }

      /* note: boxes of size zero will not be added to block */
      for (k = imin[2]; k < imax[2]; k++)
      {
         for (j = imin[1]; j < imax[1]; j++)
         {
            for (i = imin[0]; i < imax[0]; i++)
            {
               index = ((k) * block_sz[1] + j) * block_sz[0] + i;

               block[index] = factor[2] + factor[1] + factor[0];
            }
         }
      }
   }

   /*------------------------------------------------------
    * Join block array in x, then y, then z
    *
    * Notes:
    *   - ii[0], ii[1], and ii[2] correspond to indices
    *     in x, y, and z respectively.
    *   - dd specifies the order in which to loop over
    *     the three dimensions.
    *------------------------------------------------------*/

   for (d = 0; d < 3; d++)
   {
      switch (d)
      {
         case 0: /* join in x */
            dd[0] = 0;
            dd[1] = 1;
            dd[2] = 2;
            break;

         case 1: /* join in y */
            dd[0] = 1;
            dd[1] = 0;
            dd[2] = 2;
            break;

         case 2: /* join in z */
            dd[0] = 2;
            dd[1] = 1;
            dd[2] = 0;
            break;
      }

      for (ii[dd[2]] = 0; ii[dd[2]] < block_sz[dd[2]]; ii[dd[2]]++)
      {
         for (ii[dd[1]] = 0; ii[dd[1]] < block_sz[dd[1]]; ii[dd[1]]++)
         {
            join = 0;
            for (ii[dd[0]] = 0; ii[dd[0]] < block_sz[dd[0]]; ii[dd[0]]++)
            {
               index = ((ii[2]) * block_sz[1] + ii[1]) * block_sz[0] + ii[0];

               if ((join) && (block[index] == i_tmp1))
               {
                  block[index]  = 0;
                  block[i_tmp0] += factor[dd[0]];
               }
               else
               {
                  if (block[index])
                  {
                     i_tmp0 = index;
                     i_tmp1 = block[index];
                     join  = 1;
                  }
                  else
                  {
                     join = 0;
                  }
               }
            }
         }
      }
   }

   /*------------------------------------------------------
    * Set up the boxes BoxArray
    *------------------------------------------------------*/

   size = 0;
   for (index = 0; index < block_volume; index++)
   {
      if (block[index])
      {
         size++;
      }
   }
   nalu_hypre_BoxArraySetSize(boxes, size);

   index = 0;
   size = 0;
   for (k = 0; k < block_sz[2]; k++)
   {
      for (j = 0; j < block_sz[1]; j++)
      {
         for (i = 0; i < block_sz[0]; i++)
         {
            if (block[index])
            {
               ioff = (block[index] % factor[1])            ;
               joff = (block[index] % factor[2]) / factor[1];
               koff = (block[index]            ) / factor[2];

               box = nalu_hypre_BoxArrayBox(boxes, size);
               nalu_hypre_BoxIMinD(box, 0) = block_index[0][i];
               nalu_hypre_BoxIMinD(box, 1) = block_index[1][j];
               nalu_hypre_BoxIMinD(box, 2) = block_index[2][k];
               nalu_hypre_BoxIMaxD(box, 0) = block_index[0][i + ioff] - 1;
               nalu_hypre_BoxIMaxD(box, 1) = block_index[1][j + joff] - 1;
               nalu_hypre_BoxIMaxD(box, 2) = block_index[2][k + koff] - 1;

               size++;
            }

            index++;
         }
      }
   }

   /*---------------------------------------------------------
    * Clean up and return
    *---------------------------------------------------------*/

   nalu_hypre_TFree(block_index[0], NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(block, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NOTE: Avoid using - this only works for ndim < 4
 *
 * Compute the union of all boxes such that the minimum number of boxes is
 * generated. Accomplished by making six calls to nalu_hypre_UnionBoxes and then
 * taking the union that has the least no. of boxes. The six calls union in the
 * order xzy, yzx, yxz, zxy, zyx, xyz
 *--------------------------------------------------------------------------*/

/* ONLY3D */

NALU_HYPRE_Int
nalu_hypre_MinUnionBoxes( nalu_hypre_BoxArray *boxes )
{
   nalu_hypre_BoxArrayArray     *rotated_array;
   nalu_hypre_BoxArray          *rotated_boxes;
   nalu_hypre_Box               *box, *rotated_box;
   nalu_hypre_Index              lower, upper;

   NALU_HYPRE_Int                i, j, size, min_size, array;

   size = nalu_hypre_BoxArraySize(boxes);
   rotated_box = nalu_hypre_CTAlloc(nalu_hypre_Box,  1, NALU_HYPRE_MEMORY_HOST);
   rotated_array = nalu_hypre_BoxArrayArrayCreate(5, nalu_hypre_BoxArrayNDim(boxes));

   for (i = 0; i < 5; i++)
   {
      rotated_boxes = nalu_hypre_BoxArrayArrayBoxArray(rotated_array, i);
      switch (i)
      {
         case 0:
            for (j = 0; j < size; j++)
            {
               box = nalu_hypre_BoxArrayBox(boxes, j);
               nalu_hypre_SetIndex3(lower, nalu_hypre_BoxIMin(box)[0],  nalu_hypre_BoxIMin(box)[2],
                               nalu_hypre_BoxIMin(box)[1]);
               nalu_hypre_SetIndex3(upper, nalu_hypre_BoxIMax(box)[0],  nalu_hypre_BoxIMax(box)[2],
                               nalu_hypre_BoxIMax(box)[1]);
               nalu_hypre_BoxSetExtents(rotated_box, lower, upper);
               nalu_hypre_AppendBox(rotated_box, rotated_boxes);
            }
            nalu_hypre_UnionBoxes(rotated_boxes);
            break;

         case 1:
            for (j = 0; j < size; j++)
            {
               box = nalu_hypre_BoxArrayBox(boxes, j);
               nalu_hypre_SetIndex3(lower, nalu_hypre_BoxIMin(box)[1],  nalu_hypre_BoxIMin(box)[2],
                               nalu_hypre_BoxIMin(box)[0]);
               nalu_hypre_SetIndex3(upper, nalu_hypre_BoxIMax(box)[1],  nalu_hypre_BoxIMax(box)[2],
                               nalu_hypre_BoxIMax(box)[0]);
               nalu_hypre_BoxSetExtents(rotated_box, lower, upper);
               nalu_hypre_AppendBox(rotated_box, rotated_boxes);
            }
            nalu_hypre_UnionBoxes(rotated_boxes);
            break;

         case 2:
            for (j = 0; j < size; j++)
            {
               box = nalu_hypre_BoxArrayBox(boxes, j);
               nalu_hypre_SetIndex3(lower, nalu_hypre_BoxIMin(box)[1],  nalu_hypre_BoxIMin(box)[0],
                               nalu_hypre_BoxIMin(box)[2]);
               nalu_hypre_SetIndex3(upper, nalu_hypre_BoxIMax(box)[1],  nalu_hypre_BoxIMax(box)[0],
                               nalu_hypre_BoxIMax(box)[2]);
               nalu_hypre_BoxSetExtents(rotated_box, lower, upper);
               nalu_hypre_AppendBox(rotated_box, rotated_boxes);
            }
            nalu_hypre_UnionBoxes(rotated_boxes);
            break;

         case 3:
            for (j = 0; j < size; j++)
            {
               box = nalu_hypre_BoxArrayBox(boxes, j);
               nalu_hypre_SetIndex3(lower, nalu_hypre_BoxIMin(box)[2],  nalu_hypre_BoxIMin(box)[0],
                               nalu_hypre_BoxIMin(box)[1]);
               nalu_hypre_SetIndex3(upper, nalu_hypre_BoxIMax(box)[2],  nalu_hypre_BoxIMax(box)[0],
                               nalu_hypre_BoxIMax(box)[1]);
               nalu_hypre_BoxSetExtents(rotated_box, lower, upper);
               nalu_hypre_AppendBox(rotated_box, rotated_boxes);
            }
            nalu_hypre_UnionBoxes(rotated_boxes);
            break;

         case 4:
            for (j = 0; j < size; j++)
            {
               box = nalu_hypre_BoxArrayBox(boxes, j);
               nalu_hypre_SetIndex3(lower, nalu_hypre_BoxIMin(box)[2],  nalu_hypre_BoxIMin(box)[1],
                               nalu_hypre_BoxIMin(box)[0]);
               nalu_hypre_SetIndex3(upper, nalu_hypre_BoxIMax(box)[2],  nalu_hypre_BoxIMax(box)[1],
                               nalu_hypre_BoxIMax(box)[0]);
               nalu_hypre_BoxSetExtents(rotated_box, lower, upper);
               nalu_hypre_AppendBox(rotated_box, rotated_boxes);
            }
            nalu_hypre_UnionBoxes(rotated_boxes);
            break;

      } /*switch(i) */
   }    /* for (i= 0; i< 5; i++) */
   nalu_hypre_TFree(rotated_box, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_UnionBoxes(boxes);

   array = 5;
   min_size = nalu_hypre_BoxArraySize(boxes);

   for (i = 0; i < 5; i++)
   {
      rotated_boxes = nalu_hypre_BoxArrayArrayBoxArray(rotated_array, i);
      if (nalu_hypre_BoxArraySize(rotated_boxes) < min_size)
      {
         min_size = nalu_hypre_BoxArraySize(rotated_boxes);
         array = i;
      }
   }

   /* copy the box_array with the minimum number of boxes to boxes */
   if (array != 5)
   {
      rotated_boxes = nalu_hypre_BoxArrayArrayBoxArray(rotated_array, array);
      nalu_hypre_BoxArraySize(boxes) = min_size;

      switch (array)
      {
         case 0:
            for (j = 0; j < min_size; j++)
            {
               rotated_box = nalu_hypre_BoxArrayBox(rotated_boxes, j);
               nalu_hypre_SetIndex3(lower, nalu_hypre_BoxIMin(rotated_box)[0],
                               nalu_hypre_BoxIMin(rotated_box)[2],
                               nalu_hypre_BoxIMin(rotated_box)[1]);
               nalu_hypre_SetIndex3(upper, nalu_hypre_BoxIMax(rotated_box)[0],
                               nalu_hypre_BoxIMax(rotated_box)[2],
                               nalu_hypre_BoxIMax(rotated_box)[1]);

               nalu_hypre_BoxSetExtents( nalu_hypre_BoxArrayBox(boxes, j), lower, upper);
            }
            break;

         case 1:
            for (j = 0; j < min_size; j++)
            {
               rotated_box = nalu_hypre_BoxArrayBox(rotated_boxes, j);
               nalu_hypre_SetIndex3(lower, nalu_hypre_BoxIMin(rotated_box)[2],
                               nalu_hypre_BoxIMin(rotated_box)[0],
                               nalu_hypre_BoxIMin(rotated_box)[1]);
               nalu_hypre_SetIndex3(upper, nalu_hypre_BoxIMax(rotated_box)[2],
                               nalu_hypre_BoxIMax(rotated_box)[0],
                               nalu_hypre_BoxIMax(rotated_box)[1]);

               nalu_hypre_BoxSetExtents( nalu_hypre_BoxArrayBox(boxes, j), lower, upper);
            }
            break;

         case 2:
            for (j = 0; j < min_size; j++)
            {
               rotated_box = nalu_hypre_BoxArrayBox(rotated_boxes, j);
               nalu_hypre_SetIndex3(lower, nalu_hypre_BoxIMin(rotated_box)[1],
                               nalu_hypre_BoxIMin(rotated_box)[0],
                               nalu_hypre_BoxIMin(rotated_box)[2]);
               nalu_hypre_SetIndex3(upper, nalu_hypre_BoxIMax(rotated_box)[1],
                               nalu_hypre_BoxIMax(rotated_box)[0],
                               nalu_hypre_BoxIMax(rotated_box)[2]);

               nalu_hypre_BoxSetExtents( nalu_hypre_BoxArrayBox(boxes, j), lower, upper);
            }
            break;

         case 3:
            for (j = 0; j < min_size; j++)
            {
               rotated_box = nalu_hypre_BoxArrayBox(rotated_boxes, j);
               nalu_hypre_SetIndex3(lower, nalu_hypre_BoxIMin(rotated_box)[1],
                               nalu_hypre_BoxIMin(rotated_box)[2],
                               nalu_hypre_BoxIMin(rotated_box)[0]);
               nalu_hypre_SetIndex3(upper, nalu_hypre_BoxIMax(rotated_box)[1],
                               nalu_hypre_BoxIMax(rotated_box)[2],
                               nalu_hypre_BoxIMax(rotated_box)[0]);

               nalu_hypre_BoxSetExtents( nalu_hypre_BoxArrayBox(boxes, j), lower, upper);
            }
            break;

         case 4:
            for (j = 0; j < min_size; j++)
            {
               rotated_box = nalu_hypre_BoxArrayBox(rotated_boxes, j);
               nalu_hypre_SetIndex3(lower, nalu_hypre_BoxIMin(rotated_box)[2],
                               nalu_hypre_BoxIMin(rotated_box)[1],
                               nalu_hypre_BoxIMin(rotated_box)[0]);
               nalu_hypre_SetIndex3(upper, nalu_hypre_BoxIMax(rotated_box)[2],
                               nalu_hypre_BoxIMax(rotated_box)[1],
                               nalu_hypre_BoxIMax(rotated_box)[0]);

               nalu_hypre_BoxSetExtents( nalu_hypre_BoxArrayBox(boxes, j), lower, upper);
            }
            break;

      }   /* switch(array) */
   }      /* if (array != 5) */

   nalu_hypre_BoxArrayArrayDestroy(rotated_array);

   return nalu_hypre_error_flag;
}
