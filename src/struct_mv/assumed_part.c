/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* This is code for the struct assumed partition - AHB 6/05 */

#include "_nalu_hypre_struct_mv.h"

/* these are for debugging */
#define REGION_STAT 0
#define NO_REFINE   0
#define REFINE_INFO 0

/* Note: Functions used only in this file (not elsewhere) to determine the
 * partition have names that start with nalu_hypre_AP */

/*--------------------------------------------------------------------------
 * Computes the product of the first ndim index values.  Returns 1 if ndim = 0.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IndexProd( nalu_hypre_Index  index,
                 NALU_HYPRE_Int    ndim )
{
   NALU_HYPRE_Int  d, prod;

   prod = 1;
   for (d = 0; d < ndim; d++)
   {
      prod *= nalu_hypre_IndexD(index, d);
   }

   return prod;
}

/*--------------------------------------------------------------------------
 * Computes an index into a multi-D box of size bsize[0] x bsize[1] x ... from a
 * rank where the assumed ordering is dimension 0 first, then dimension 1, etc.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_IndexFromRank( NALU_HYPRE_Int    rank,
                     nalu_hypre_Index  bsize,
                     nalu_hypre_Index  index,
                     NALU_HYPRE_Int    ndim )
{
   NALU_HYPRE_Int  d, r, s;

   r = rank;
   for (d = ndim - 1; d >= 0; d--)
   {
      s = nalu_hypre_IndexProd(bsize, d);
      nalu_hypre_IndexD(index, d) = r / s;
      r = r % s;
   }

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * Given a region, subdivide the region equally a specified number of times.
 * For dimension d, each "level" is a subdivison of 2^d.  The box_array is
 * adjusted to have space for l(2^d)^level boxes.  We are bisecting each
 * dimension (level) times.
 *
 * We may want to add min size parameter for dimension of results regions
 * (currently 2), i.e., don't bisect a dimension if it will be smaller than 2
 * grid points, for example.
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_APSubdivideRegion( nalu_hypre_Box      *region,
                         NALU_HYPRE_Int       ndim,
                         NALU_HYPRE_Int       level,
                         nalu_hypre_BoxArray *box_array,
                         NALU_HYPRE_Int      *num_new_boxes )
{
   NALU_HYPRE_Int    i, j,  width, sz, dv, total;
   NALU_HYPRE_Int    extra, points, count;
   NALU_HYPRE_Int   *partition[NALU_HYPRE_MAXDIM];

   NALU_HYPRE_Int    min_gridpts; /* This should probably be an input parameter */

   nalu_hypre_Index  isize, index, div;
   nalu_hypre_Box   *box;

   /* if level = 0 then no dividing */
   if (!level)
   {
      nalu_hypre_BoxArraySetSize(box_array, 1);
      nalu_hypre_CopyBox(region, nalu_hypre_BoxArrayBox(box_array, 0));
      *num_new_boxes = 1;
      return nalu_hypre_error_flag;
   }

   /* Get the size of the box in each dimension */
   nalu_hypre_BoxGetSize(region, isize);

   /* div = num of regions in each dimension */

   /* Figure out the number of regions.  Make sure the sizes will contain the
      min number of gridpoints, or divide less in that dimension.  We require at
      least min_gridpts in a region dimension. */

   min_gridpts = 4;

   total = 1;
   for (i = 0; i < ndim; i++)
   {
      dv = 1;
      sz = nalu_hypre_IndexD(isize, i);
      for (j = 0; j < level; j++)
      {
         if (sz >= 2 * dv * min_gridpts) /* Cut each dim in half */
         {
            dv *= 2;
         }
      }

      /* Space for each partition */
      partition[i] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  dv + 1, NALU_HYPRE_MEMORY_HOST);
      /* Total number of regions to create */
      total = total * dv;

      nalu_hypre_IndexD(div, i) = dv;
   }
   *num_new_boxes = total;

   /* Prepare box array */
   nalu_hypre_BoxArraySetSize(box_array, total);

   /* Divide each dimension */
   for (i = 0; i < ndim; i++)
   {
      dv = nalu_hypre_IndexD(div, i);
      partition[i][0] =  nalu_hypre_BoxIMinD(region, i);
      /* Count grid points */
      points = nalu_hypre_IndexD(isize, i);
      width =  points / dv;
      extra =  points % dv;
      for (j = 1; j < dv; j++)
      {
         partition[i][j] = partition[i][j - 1] + width;
         if (j <= extra)
         {
            partition[i][j]++;
         }
      }
      partition[i][dv] = nalu_hypre_BoxIMaxD(region, i) + 1;
   }

   count = 0;
   nalu_hypre_SerialBoxLoop0Begin(ndim, div);
   {
      box = nalu_hypre_BoxArrayBox(box_array, count);
      zypre_BoxLoopGetIndex(index);
      for (i = 0; i < ndim; i++)
      {
         j = nalu_hypre_IndexD(index, i);
         nalu_hypre_BoxIMinD(box, i) = partition[i][j];
         nalu_hypre_BoxIMaxD(box, i) = partition[i][j + 1] - 1;
      }
      count++;
   }
   nalu_hypre_SerialBoxLoop0End();

   /* clean up */
   for (i = 0; i < ndim; i++)
   {
      nalu_hypre_TFree(partition[i], NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * Given a list of regions, find out how many of *my* boxes are contained in
 * each region.
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_APFindMyBoxesInRegions( nalu_hypre_BoxArray *region_array,
                              nalu_hypre_BoxArray *my_box_array,
                              NALU_HYPRE_Int     **p_count_array,
                              NALU_HYPRE_Real    **p_vol_array )
{
   NALU_HYPRE_Int      ndim = nalu_hypre_BoxArrayNDim(region_array);
   NALU_HYPRE_Int      i, j, d;
   NALU_HYPRE_Int      num_boxes, num_regions;
   NALU_HYPRE_Int     *count_array;
   NALU_HYPRE_Real    *vol_array;
   nalu_hypre_Box     *my_box, *result_box, *grow_box, *region;
   nalu_hypre_Index    grow_index;

   num_boxes =  nalu_hypre_BoxArraySize(my_box_array);
   num_regions = nalu_hypre_BoxArraySize(region_array);

   count_array = *p_count_array;
   vol_array = *p_vol_array;

   /* May need to add some sorting to make this more efficient, though we
      shouldn't have many regions */

   /* Note: a box can be in more than one region */

   result_box = nalu_hypre_BoxCreate(ndim);
   grow_box = nalu_hypre_BoxCreate(ndim);

   for (i = 0; i < num_regions; i++)
   {
      count_array[i] = 0;
      vol_array[i] = 0.0;

      region = nalu_hypre_BoxArrayBox(region_array, i);

      for (j = 0; j < num_boxes; j++)
      {
         my_box = nalu_hypre_BoxArrayBox(my_box_array, j);
         /* Check if its a zero volume box.  If so, it still need to be counted,
            so expand until volume is non-zero, then intersect. */
         if (nalu_hypre_BoxVolume(my_box) == 0)
         {
            nalu_hypre_CopyBox(my_box, grow_box);
            for (d = 0; d < ndim; d++)
            {
               if (!nalu_hypre_BoxSizeD(my_box, d))
               {
                  nalu_hypre_IndexD(grow_index, d) =
                     (nalu_hypre_BoxIMinD(my_box, d) - nalu_hypre_BoxIMaxD(my_box, d) + 1) / 2;
               }
               else
               {
                  nalu_hypre_IndexD(grow_index, d) = 0;
               }
            }
            /* Expand the grow box (leave our box untouched) */
            nalu_hypre_BoxGrowByIndex(grow_box, grow_index);
            /* Do they intersect? */
            nalu_hypre_IntersectBoxes(grow_box, region, result_box);
         }
         else
         {
            /* Do they intersect? */
            nalu_hypre_IntersectBoxes(my_box, region, result_box);
         }
         if (nalu_hypre_BoxVolume(result_box) > 0)
         {
            count_array[i]++;
            vol_array[i] += (NALU_HYPRE_Real) nalu_hypre_BoxVolume(result_box);
         }
      }
   }

   /* clean up */
   nalu_hypre_BoxDestroy(result_box);
   nalu_hypre_BoxDestroy(grow_box);

   /* output */
   *p_count_array = count_array;
   *p_vol_array = vol_array;

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * Given a list of regions, find out how many global boxes are contained in each
 * region.  Assumes that p_count_array and p_vol_array have been allocated.
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_APGetAllBoxesInRegions( nalu_hypre_BoxArray *region_array,
                              nalu_hypre_BoxArray *my_box_array,
                              NALU_HYPRE_Int     **p_count_array,
                              NALU_HYPRE_Real    **p_vol_array,
                              MPI_Comm        comm )
{
   NALU_HYPRE_Int    i;
   NALU_HYPRE_Int   *count_array;
   NALU_HYPRE_Int    num_regions;
   NALU_HYPRE_Int   *send_buf_count;
   NALU_HYPRE_Real  *send_buf_vol;
   NALU_HYPRE_Real  *vol_array;
   NALU_HYPRE_Real  *dbl_vol_and_count;

   count_array = *p_count_array;
   vol_array = *p_vol_array;

   /* First get a count and volume of my boxes in each region */
   num_regions = nalu_hypre_BoxArraySize(region_array);

   send_buf_count = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_regions, NALU_HYPRE_MEMORY_HOST);
   send_buf_vol = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_regions * 2,
                                NALU_HYPRE_MEMORY_HOST); /* allocate NALU_HYPRE_Real */

   dbl_vol_and_count =  nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_regions * 2,
                                      NALU_HYPRE_MEMORY_HOST); /* allocate NALU_HYPRE_Real */

   nalu_hypre_APFindMyBoxesInRegions( region_array, my_box_array, &send_buf_count,
                                 &send_buf_vol);


   /* Copy ints to doubles so we can do one Allreduce */
   for (i = 0; i < num_regions; i++)
   {
      send_buf_vol[num_regions + i] = (NALU_HYPRE_Real) send_buf_count[i];
   }

   nalu_hypre_MPI_Allreduce(send_buf_vol, dbl_vol_and_count, num_regions * 2,
                       NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);

   /* Unpack */
   for (i = 0; i < num_regions; i++)
   {
      vol_array[i] = dbl_vol_and_count[i];
      count_array[i] = (NALU_HYPRE_Int) dbl_vol_and_count[num_regions + i];
   }

   /* Clean up */
   nalu_hypre_TFree(send_buf_count, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_buf_vol, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dbl_vol_and_count, NALU_HYPRE_MEMORY_HOST);

   /* Output */
   *p_count_array = count_array;
   *p_vol_array = vol_array;

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * Given a list of regions, shrink regions according to min and max extents.
 * These regions should all be non-empty at the global level.
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_APShrinkRegions( nalu_hypre_BoxArray *region_array,
                       nalu_hypre_BoxArray *my_box_array,
                       MPI_Comm        comm )
{
   NALU_HYPRE_Int     ndim, ndim2;
   NALU_HYPRE_Int     i, j, d, ii;
   NALU_HYPRE_Int     num_boxes, num_regions;
   NALU_HYPRE_Int    *indices, *recvbuf;
   NALU_HYPRE_Int     count = 0;

   nalu_hypre_Box    *my_box, *result_box, *grow_box, *region;
   nalu_hypre_Index   grow_index, imin, imax;

   ndim  = nalu_hypre_BoxArrayNDim(my_box_array);
   ndim2 = 2 * ndim;

   num_boxes   = nalu_hypre_BoxArraySize(my_box_array);
   num_regions = nalu_hypre_BoxArraySize(region_array);

   indices = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_regions * ndim2, NALU_HYPRE_MEMORY_HOST);
   recvbuf = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_regions * ndim2, NALU_HYPRE_MEMORY_HOST);

   result_box = nalu_hypre_BoxCreate(ndim);

   /* Allocate a grow box */
   grow_box = nalu_hypre_BoxCreate(ndim);

   /* Look locally at my boxes */
   /* For each region */
   for (i = 0; i < num_regions; i++)
   {
      count = 0; /* Number of my boxes in this region */

      /* Get the region box */
      region = nalu_hypre_BoxArrayBox(region_array, i);

      /* Go through each of my local boxes */
      for (j = 0; j < num_boxes; j++)
      {
         my_box = nalu_hypre_BoxArrayBox(my_box_array, j);

         /* Check if its a zero volume box.  If so, it still needs to be
            checked, so expand until volume is nonzero, then intersect. */
         if (nalu_hypre_BoxVolume(my_box) == 0)
         {
            nalu_hypre_CopyBox(my_box, grow_box);
            for (d = 0; d < ndim; d++)
            {
               if (!nalu_hypre_BoxSizeD(my_box, d))
               {
                  nalu_hypre_IndexD(grow_index, d) =
                     (nalu_hypre_BoxIMinD(my_box, d) - nalu_hypre_BoxIMaxD(my_box, d) + 1) / 2;
               }
               else
               {
                  nalu_hypre_IndexD(grow_index, d) = 0;
               }
            }
            /* Grow the grow box (leave our box untouched) */
            nalu_hypre_BoxGrowByIndex(grow_box, grow_index);
            /* Do they intersect? */
            nalu_hypre_IntersectBoxes(grow_box, region, result_box);
         }
         else
         {
            /* Do they intersect? */
            nalu_hypre_IntersectBoxes( my_box, region, result_box);
         }

         if (nalu_hypre_BoxVolume(result_box) > 0) /* They intersect */
         {
            if (!count) /* Set min and max for first box */
            {
               ii = i * ndim2;
               for (d = 0; d < ndim; d++)
               {
                  indices[ii + d] = nalu_hypre_BoxIMinD(result_box, d);
                  indices[ii + ndim + d] = nalu_hypre_BoxIMaxD(result_box, d);
               }
            }

            count++;

            /* Boxes intersect, so get max and min extents of the result box
               (this keeps the bounds inside the region) */
            ii = i * ndim2;
            for (d = 0; d < ndim; d++)
            {
               indices[ii + d] = nalu_hypre_min(indices[ii + d],
                                           nalu_hypre_BoxIMinD(result_box, d));
               indices[ii + ndim + d] = nalu_hypre_max(indices[ii + ndim + d],
                                                  nalu_hypre_BoxIMaxD(result_box, d));
            }
         }
      }

      /* If we had no boxes in that region, set the min to the max extents of
         the region and the max to the min! */
      if (!count)
      {
         ii = i * ndim2;
         for (d = 0; d < ndim; d++)
         {
            indices[ii + d] = nalu_hypre_BoxIMaxD(region, d);
            indices[ii + ndim + d] = nalu_hypre_BoxIMinD(region, d);
         }
      }

      /* Negate max indices for the Allreduce */
      /* Note: min(x)= -max(-x) */
      ii = i * ndim2;
      for (d = 0; d < ndim; d++)
      {
         indices[ii + ndim + d] = -indices[ii + ndim + d];
      }
   }

   /* Do an Allreduce on size and volume to get the global information */
   nalu_hypre_MPI_Allreduce(indices, recvbuf, num_regions * ndim2, NALU_HYPRE_MPI_INT,
                       nalu_hypre_MPI_MIN, comm);

   /* Unpack the "shrunk" regions */
   /* For each region */
   for (i = 0; i < num_regions; i++)
   {
      /* Get the region box */
      region = nalu_hypre_BoxArrayBox(region_array, i);

      /* Resize the box */
      ii = i * ndim2;
      for (d = 0; d < ndim; d++)
      {
         nalu_hypre_IndexD(imin, d) =  recvbuf[ii + d];
         nalu_hypre_IndexD(imax, d) = -recvbuf[ii + ndim + d];
      }

      nalu_hypre_BoxSetExtents(region, imin, imax );

      /* Add: check to see whether any shrinking is actually occuring */
   }

   /* Clean up */
   nalu_hypre_TFree(recvbuf, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(indices, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_BoxDestroy(result_box);
   nalu_hypre_BoxDestroy(grow_box);

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * Given a list of regions, eliminate empty regions.
 *
 * region_array = assumed partition regions
 * count_array  = number of global boxes in each region
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_APPruneRegions( nalu_hypre_BoxArray *region_array,
                      NALU_HYPRE_Int     **p_count_array,
                      NALU_HYPRE_Real    **p_vol_array )
{
   NALU_HYPRE_Int   i, j;
   NALU_HYPRE_Int   num_regions;
   NALU_HYPRE_Int   count;
   NALU_HYPRE_Int   *delete_indices;

   NALU_HYPRE_Int   *count_array;
   NALU_HYPRE_Real  *vol_array;

   count_array = *p_count_array;
   vol_array = *p_vol_array;

   num_regions = nalu_hypre_BoxArraySize(region_array);
   delete_indices = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_regions, NALU_HYPRE_MEMORY_HOST);
   count = 0;

   /* Delete regions with zero elements */
   for (i = 0; i < num_regions; i++)
   {
      if (count_array[i] == 0)
      {
         delete_indices[count++] = i;
      }
   }

   nalu_hypre_DeleteMultipleBoxes(region_array, delete_indices, count);

   /* Adjust count and volume arrays */
   if (count > 0)
   {
      j = 0;
      for (i = delete_indices[0]; (i + j) < num_regions; i++)
      {
         if (j < count)
         {
            while ((i + j) == delete_indices[j])
            {
               j++; /* Increase the shift */
               if (j == count) { break; }
            }
         }
         vol_array[i] = vol_array[i + j];
         count_array[i] = count_array[i + j];
      }
   }

   /* Clean up */
   nalu_hypre_TFree(delete_indices, NALU_HYPRE_MEMORY_HOST);

   /* Return variables */
   *p_count_array = count_array;
   *p_vol_array = vol_array;

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * Given a list of regions, and corresponding volumes contained in regions
 * subdivide some of the regions that are not full enough.
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_APRefineRegionsByVol( nalu_hypre_BoxArray *region_array,
                            NALU_HYPRE_Real     *vol_array,
                            NALU_HYPRE_Int       max_regions,
                            NALU_HYPRE_Real      gamma,
                            NALU_HYPRE_Int       ndim,
                            NALU_HYPRE_Int      *return_code,
                            MPI_Comm        comm )
{
   NALU_HYPRE_Int          i, count, loop;
   NALU_HYPRE_Int          num_regions, init_num_regions;
   NALU_HYPRE_Int         *delete_indices;

   NALU_HYPRE_Real        *fraction_full;
   NALU_HYPRE_Int         *order;
   NALU_HYPRE_Int          myid, num_procs, est_size;
   NALU_HYPRE_Int          new1;

   nalu_hypre_BoxArray    *tmp_array;
   nalu_hypre_Box         *box;

   nalu_hypre_MPI_Comm_rank(comm, &myid);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   num_regions = nalu_hypre_BoxArraySize(region_array);

   if (!num_regions)
   {
      /* No regions, so no subdividing */
      *return_code = 1;
      return nalu_hypre_error_flag;
   }

   fraction_full = nalu_hypre_CTAlloc(NALU_HYPRE_Real,   num_regions, NALU_HYPRE_MEMORY_HOST);
   order = nalu_hypre_CTAlloc(NALU_HYPRE_Int,   num_regions, NALU_HYPRE_MEMORY_HOST);
   delete_indices = nalu_hypre_CTAlloc(NALU_HYPRE_Int,   num_regions, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_regions; i++)
   {
      box = nalu_hypre_BoxArrayBox(region_array, i);
      fraction_full[i] = vol_array[i] / nalu_hypre_doubleBoxVolume(box);
      order[i] = i; /* This is what order to access the boxes */
   }

   /* Want to refine the regions starting with those that are the least full */
   /* Sort the fraction AND the index */
   nalu_hypre_qsort2(order, fraction_full, 0, num_regions - 1);

   /* Now we can subdivide any that are not full enough */
   /* When this is called, we know that size < max_regions */
   /* It is ok to subdivde such that we have slightly more regions than
      max_region, but we do not want more regions than processors */

   tmp_array = nalu_hypre_BoxArrayCreate(0, ndim);
   count = 0; /* How many regions subdivided */
   loop = 0; /* Counts the loop number */
   init_num_regions = num_regions;
   /* All regions are at least gamma full and no subdividing occured */
   *return_code = 1;

   while (fraction_full[loop] < gamma)
   {
      /* Some subdividing occurred */
      *return_code = 2;

      /* We can't let the number of regions exceed the number of processors.
         Only an issue for small proc numbers. */
      est_size = num_regions + nalu_hypre_pow2(ndim) - 1;
      if (est_size > num_procs)
      {
         if (loop == 0)
         {
            /* Some are less than gamma full, but we cannot further subdivide
               due to max processors limit (no subdividing occured) */
            *return_code = 4;
         }

         else
         {
            /* Some subdividing occured, but there are some regions less than
               gamma full (max reached) that were not subdivided */
            *return_code = 3;
         }

         break;
      }

      box = nalu_hypre_BoxArrayBox(region_array, order[loop]);
      nalu_hypre_APSubdivideRegion(box, ndim, 1, tmp_array, &new1);

      if (new1 > 1) /* If new = 1, then no subdividing occured */
      {
         num_regions = num_regions + new1 - 1; /* The orginal will be deleted */

         delete_indices[count] = order[loop];
         count++; /* Number of regions subdivided */

         /* Append tmp_array to region_array */
         nalu_hypre_AppendBoxArray(tmp_array, region_array);
      }

      /* If we are on the last region */
      if  ((loop + 1) == init_num_regions)
      {
         break;
      }

      /* Clear tmp_array for next loop */
      nalu_hypre_BoxArraySetSize(tmp_array, 0);

      /* If we now have too many regions, don't want to subdivide anymore */
      if (num_regions >= max_regions)
      {
         /* See if next regions satifies gamma */
         if (fraction_full[order[loop + 1]] > gamma)
         {
            /* All regions less than gamma full have been subdivided (and we
               have reached max) */
            *return_code = 5;
         }
         else
         {
            /* Some regions less than gamma full (but max is reached) */
            *return_code = 3;
         }
         break;
      }

      loop++; /* Increment to repeat loop */
   }

   if (count == 0 )
   {
      /* No refining occured so don't do any more */
      *return_code = 1;
   }
   else
   {
      /* We subdivided count regions */
      /* Delete the old regions */
      nalu_hypre_qsort0(delete_indices, 0, count - 1); /* Put deleted indices in asc order */
      nalu_hypre_DeleteMultipleBoxes( region_array, delete_indices, count );
   }

   /* TO DO: number of regions intact (beginning of region array is intact) -
      may return this eventually */
   /* regions_intact = init_num_regions - count; */

   /* Clean up */
   nalu_hypre_TFree(fraction_full, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(order, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(delete_indices, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_BoxArrayDestroy(tmp_array);

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * Construct an assumed partition
 *
 * 8/06 - Changed the assumption that the local boxes have boxnums 0 to
 * num(local_boxes)-1 (now need to pass in ids).
 *
 * 10/06 - Changed.  No longer need to deal with negative boxes as this is used
 * through the box manager.
 *
 * 3/6 - Don't allow more regions than boxes (unless global boxes = 0) and don't
 * partition into more procs than global number of boxes.
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_StructAssumedPartitionCreate(
   NALU_HYPRE_Int                 ndim,
   nalu_hypre_Box                *bounding_box,
   NALU_HYPRE_Real                global_boxes_size,
   NALU_HYPRE_Int                 global_num_boxes,
   nalu_hypre_BoxArray           *local_boxes,
   NALU_HYPRE_Int                *local_boxnums,
   NALU_HYPRE_Int                 max_regions,
   NALU_HYPRE_Int                 max_refinements,
   NALU_HYPRE_Real                gamma,
   MPI_Comm                  comm,
   nalu_hypre_StructAssumedPart **p_assumed_partition )
{
   NALU_HYPRE_Int          i, j, d;
   NALU_HYPRE_Int          size;
   NALU_HYPRE_Int          myid, num_procs;
   NALU_HYPRE_Int          num_proc_partitions;
   NALU_HYPRE_Int          count_array_size;
   NALU_HYPRE_Int         *count_array = NULL;
   NALU_HYPRE_Real        *vol_array = NULL, one_volume, dbl_vol;
   NALU_HYPRE_Int          return_code;
   NALU_HYPRE_Int          num_refine;
   NALU_HYPRE_Int          total_boxes, proc_count, max_position;
   NALU_HYPRE_Int         *proc_array = NULL;
   NALU_HYPRE_Int          initial_level;
   NALU_HYPRE_Int          dmax;
   NALU_HYPRE_Real         width, wmin, wmax;
   NALU_HYPRE_Real         rn_cubes, rn_cube_procs, rn_cube_divs, rdiv;

   nalu_hypre_Index        div_index;
   nalu_hypre_BoxArray    *region_array;
   nalu_hypre_Box         *box, *grow_box;

   nalu_hypre_StructAssumedPart *assumed_part;

   NALU_HYPRE_Int   proc_alloc, count, box_count;
   NALU_HYPRE_Int   max_response_size;
   NALU_HYPRE_Int  *response_buf = NULL, *response_buf_starts = NULL;
   NALU_HYPRE_Int  *tmp_proc_ids = NULL, *tmp_box_nums = NULL, *tmp_box_inds = NULL;
   NALU_HYPRE_Int  *proc_array_starts = NULL;

   nalu_hypre_BoxArray              *my_partition;
   nalu_hypre_DataExchangeResponse  response_obj;

   NALU_HYPRE_Int  *contact_boxinfo;
   NALU_HYPRE_Int  index;


   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &myid);

   /* Special case where there are no boxes in the grid */
   if (global_num_boxes == 0)
   {
      region_array = nalu_hypre_BoxArrayCreate(0, ndim);
      assumed_part = nalu_hypre_TAlloc(nalu_hypre_StructAssumedPart,  1, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_StructAssumedPartNDim(assumed_part) = ndim;
      nalu_hypre_StructAssumedPartRegions(assumed_part) = region_array;
      nalu_hypre_StructAssumedPartNumRegions(assumed_part) = 0;
      nalu_hypre_StructAssumedPartDivisions(assumed_part) =  NULL;
      nalu_hypre_StructAssumedPartProcPartitions(assumed_part) =
         nalu_hypre_CTAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_StructAssumedPartProcPartition(assumed_part, 0) = 0;
      nalu_hypre_StructAssumedPartMyPartition(assumed_part) =  NULL;
      nalu_hypre_StructAssumedPartMyPartitionBoxes(assumed_part)
         = nalu_hypre_BoxArrayCreate(0, ndim);
      nalu_hypre_StructAssumedPartMyPartitionIdsAlloc(assumed_part) = 0;
      nalu_hypre_StructAssumedPartMyPartitionIdsSize(assumed_part) = 0;
      nalu_hypre_StructAssumedPartMyPartitionNumDistinctProcs(assumed_part) = 0;
      nalu_hypre_StructAssumedPartMyPartitionBoxnums(assumed_part) = NULL;
      nalu_hypre_StructAssumedPartMyPartitionProcIds(assumed_part) = NULL;
      *p_assumed_partition = assumed_part;

      return nalu_hypre_error_flag;
   }
   /* End special case of zero boxes */

   /* FIRST DO ALL THE GLOBAL PARTITION INFO */

   /* Initially divide the bounding box */

   if (!nalu_hypre_BoxVolume(bounding_box) && global_num_boxes)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Bounding box has zero volume AND there are grid boxes");
   }

   /* First modify any input parameters if necessary */

   /* Don't want the number of regions exceeding the number of processors */
   /* Note: This doesn't change the value in the caller's code */
   max_regions = nalu_hypre_min(num_procs, max_regions);

   /* Don't want more regions than boxes either */
   if (global_num_boxes) { max_regions = nalu_hypre_min(global_num_boxes, max_regions); }

   /* Start with a region array of size 0 */
   region_array = nalu_hypre_BoxArrayCreate(0, ndim);

   /* If the bounding box is sufficiently covered by boxes, then we will just
      have one region (the bounding box), otherwise we will subdivide */

   one_volume = nalu_hypre_doubleBoxVolume(bounding_box);

   if ( ((global_boxes_size / one_volume) > gamma) ||
        (global_num_boxes > one_volume) || (global_num_boxes == 0) )
   {
      /* Don't bother with any refinements.  We are full enough, or we have a
         small bounding box and we are not full because of empty boxes */
      initial_level = 0;
      max_refinements = 0;
   }
   else
   {
      /* Could be an input parameter, but 1 division is probably sufficient */
      initial_level = 1;

      /* Start with the specified intial_levels for the original domain, unless
         we have a smaller number of procs */
      for (i = 0; i < initial_level; i++)
      {
         if ( nalu_hypre_pow2(initial_level * ndim) > num_procs) { initial_level --; }

         /* Not be able to do any refinements due to the number of processors */
         if (!initial_level) { max_refinements = 0; }
      }
   }

#if NO_REFINE
   max_refinements = 0;
   initial_level = 0;
#endif

#if REFINE_INFO
   if (myid == 0)
   {
      nalu_hypre_printf("gamma =  %g\n", gamma);
      nalu_hypre_printf("max_regions =  %d\n", max_regions);
      nalu_hypre_printf("max_refinements =  %d\n", max_refinements);
      nalu_hypre_printf("initial level =  %d\n", initial_level);
   }
#endif

   /* Divide the bounding box */
   nalu_hypre_APSubdivideRegion(bounding_box, ndim, initial_level, region_array, &size);
   /* If no subdividing occured (because too small) then don't try to refine */
   if (initial_level > 0 && size == 1) { max_refinements = 0; }

   /* Need space for count and volume */
   size = nalu_hypre_BoxArraySize(region_array);
   count_array_size = size; /* Memory allocation size */
   count_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int,   size, NALU_HYPRE_MEMORY_HOST);
   vol_array =  nalu_hypre_CTAlloc(NALU_HYPRE_Real,   size, NALU_HYPRE_MEMORY_HOST);

   /* How many boxes are in each region (global count) and what is the volume */
   nalu_hypre_APGetAllBoxesInRegions(region_array, local_boxes, &count_array,
                                &vol_array, comm);

   /* Don't do any initial prune and shrink if we have only one region and we
      can't do any refinements */

   if ( !(size == 1 && max_refinements == 0))
   {
      /* Get rid of regions with no boxes (and adjust count and vol arrays) */
      nalu_hypre_APPruneRegions( region_array, &count_array, &vol_array);

      /* Shrink the extents */
      nalu_hypre_APShrinkRegions( region_array, local_boxes, comm);
   }

   /* Keep track of refinements */
   num_refine = 0;

   /* Now we can keep refining by dividing the regions that are not full enough
      and eliminating empty regions */
   while ( (nalu_hypre_BoxArraySize(region_array) < max_regions) &&
           (num_refine < max_refinements) )
   {
      num_refine++;

      /* Calculate how full the regions are and subdivide the least full */

      size = nalu_hypre_BoxArraySize(region_array);

      /* Divide regions that are not full enough */
      nalu_hypre_APRefineRegionsByVol(region_array, vol_array, max_regions,
                                 gamma, ndim, &return_code, comm);

      /* 1 = all regions are at least gamma full - no subdividing occured */
      /* 4 = no subdividing occured due to num_procs limit on regions */
      if (return_code == 1 || return_code == 4)
      {
         break;
      }
      /* This is extraneous I think */
      if (size == nalu_hypre_BoxArraySize(region_array))
      {
         /* No dividing occured - exit the loop */
         break;
      }

      size = nalu_hypre_BoxArraySize(region_array);
      if (size >  count_array_size)
      {
         count_array = nalu_hypre_TReAlloc(count_array,  NALU_HYPRE_Int,   size, NALU_HYPRE_MEMORY_HOST);
         vol_array =  nalu_hypre_TReAlloc(vol_array,  NALU_HYPRE_Real,   size, NALU_HYPRE_MEMORY_HOST);
         count_array_size = size;
      }

      /* FUTURE MOD: Just count and prune and shrink in the modified regions
         from refineRegionsByVol. These are the last regions in the array. */

      /* Num boxes are in each region (global count) and what the volume is */
      nalu_hypre_APGetAllBoxesInRegions(region_array, local_boxes, &count_array,
                                   &vol_array, comm);

      /* Get rid of regions with no boxes (and adjust count and vol arrays) */
      nalu_hypre_APPruneRegions(region_array, &count_array, &vol_array);

      /* Shrink the extents */
      nalu_hypre_APShrinkRegions(region_array, local_boxes, comm);

      /* These may be ok after pruning, but if no pruning then exit the loop */
      /* 5 = all regions < gamma full were subdivided and max reached */
      /* 3 = some regions were divided (not all that needed) and max reached */
      if ( (return_code == 3 || return_code == 5)
           && size == nalu_hypre_BoxArraySize(region_array) )
      {
         break;
      }

   }
   /* End of refinements */

   /* Error checking */
   if (global_num_boxes)
   {
      nalu_hypre_ForBoxI(i, region_array)
      {
         if (nalu_hypre_BoxVolume(nalu_hypre_BoxArrayBox(region_array, i)) == 0)
         {
            nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
            nalu_hypre_error_w_msg(
               NALU_HYPRE_ERROR_GENERIC,
               "A region has zero volume (this should never happen)!");
         }
      }
   }

#if REGION_STAT
   if (myid == 0)
   {
      nalu_hypre_printf("myid = %d, %d REGIONS (after refining %d times\n",
                   myid, nalu_hypre_BoxArraySize(region_array), num_refine);

      nalu_hypre_ForBoxI(i, region_array)
      {
         box = nalu_hypre_BoxArrayBox(region_array, i);
         nalu_hypre_printf("myid = %d, %d:  (%d, %d, %d)  x  (%d, %d, %d)\n",
                      myid, i,
                      nalu_hypre_BoxIMinX(box),
                      nalu_hypre_BoxIMinY(box),
                      nalu_hypre_BoxIMinZ(box),
                      nalu_hypre_BoxIMaxX(box),
                      nalu_hypre_BoxIMaxY(box),
                      nalu_hypre_BoxIMaxZ(box));
      }
   }
#endif

   nalu_hypre_TFree(vol_array, NALU_HYPRE_MEMORY_HOST);

   /* ------------------------------------------------------------------------*/

   /* Now we have the regions - construct the assumed partition */

   size = nalu_hypre_BoxArraySize(region_array);
   assumed_part = nalu_hypre_TAlloc(nalu_hypre_StructAssumedPart,  1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_StructAssumedPartNDim(assumed_part) = ndim;
   nalu_hypre_StructAssumedPartRegions(assumed_part) = region_array;
   /* The above is aliased, so don't destroy region_array in this function */
   nalu_hypre_StructAssumedPartNumRegions(assumed_part) = size;
   nalu_hypre_StructAssumedPartDivisions(assumed_part) =
      nalu_hypre_CTAlloc(nalu_hypre_Index,  size, NALU_HYPRE_MEMORY_HOST);

   /* First determine which processors (how many) to assign to each region */
   proc_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
   /* This is different than the total number of boxes as some boxes can be in
      more than one region */
   total_boxes = 0;
   proc_count = 0;
   d = -1;
   max_position = -1;
   /* Calculate total number of boxes in the regions */
   for (i = 0; i < size; i++)
   {
      total_boxes += count_array[i];
   }
   /* Calculate the fraction of actual boxes in each region, multiplied by total
      number of proc partitons desired, put result in proc_array to assign each
      region a number of processors proportional to the fraction of boxes */

   /* 3/6 - Limit the number of proc partitions to no larger than the total
      boxes in the regions (at coarse levels, may be many more procs than boxes,
      so this should minimize some communication). */
   num_proc_partitions = nalu_hypre_min(num_procs, total_boxes);

   for (i = 0; i < size; i++)
   {
      if (!total_boxes) /* In case there are no boxes in a grid */
      {
         proc_array[i] = 0;
      }
      else
      {
         proc_array[i] = (NALU_HYPRE_Int)
                         nalu_hypre_round( ((NALU_HYPRE_Real)count_array[i] / (NALU_HYPRE_Real)total_boxes) *
                                      (NALU_HYPRE_Real) num_proc_partitions );
      }

      box =  nalu_hypre_BoxArrayBox(region_array, i);
      dbl_vol = nalu_hypre_doubleBoxVolume(box);

      /* Can't have any zeros! */
      if (!proc_array[i]) { proc_array[i] = 1; }

      if (dbl_vol < (NALU_HYPRE_Real) proc_array[i])
      {
         /* Don't let the number of procs be greater than the volume.  If true,
            then safe to cast back to NALU_HYPRE_Int and vol doesn't overflow. */
         proc_array[i] = (NALU_HYPRE_Int) dbl_vol;
      }

      proc_count += proc_array[i];
      if (d < proc_array[i])
      {
         d = proc_array[i];
         max_position = i;
      }

      /*If (myid == 0) nalu_hypre_printf("proc array[%d] = %d\n", i, proc_array[i]);*/
   }

   nalu_hypre_TFree(count_array, NALU_HYPRE_MEMORY_HOST);

   /* Adjust such that num_proc_partitions = proc_count (they should be close) */
   /* A processor is only assigned to ONE region */

   /* If we need a few more processors assigned in proc_array for proc_count to
      equal num_proc_partitions (it is ok if we have fewer procs in proc_array
      due to volume constraints) */
   while (num_proc_partitions > proc_count)
   {
      proc_array[max_position]++;

      if ( (NALU_HYPRE_Real) proc_array[max_position] >
           nalu_hypre_doubleBoxVolume(nalu_hypre_BoxArrayBox(region_array, max_position)) )
      {
         proc_array[max_position]--;
         break; /* Some processors won't get assigned partitions */
      }
      proc_count++;
   }

   /* If we we need fewer processors in proc_array */
   i = 0;
   while (num_proc_partitions < proc_count)
   {
      if (proc_array[max_position] != 1)
      {
         proc_array[max_position]--;
      }
      else
      {
         while (i < size && proc_array[i] <= 1) /* size is the number of regions */
         {
            i++;
         }
         proc_array[i]--;
      }
      proc_count--;
   }
   /* The above logic would be flawed IF we allowed more regions than
      processors, but this is not allowed! */

   /* Now we have the number of processors in each region so create the
      processor partition */
   /* size = # of regions */
   nalu_hypre_StructAssumedPartProcPartitions(assumed_part) =
      nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size + 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_StructAssumedPartProcPartition(assumed_part, 0) = 0;
   for (i = 0; i < size; i++)
   {
      nalu_hypre_StructAssumedPartProcPartition(assumed_part, i + 1) =
         nalu_hypre_StructAssumedPartProcPartition(assumed_part, i) + proc_array[i];
   }

   /* Now determine the NUMBER of divisions in the x, y amd z dir according
      to the number or processors assigned to the region */

   /* FOR EACH REGION */
   for (i = 0; i < size; i++)
   {
      proc_count = proc_array[i];
      box = nalu_hypre_BoxArrayBox(region_array, i);

      /* Find min width and max width dimensions */
      dmax = 0;
      wmin = wmax = nalu_hypre_BoxSizeD(box, 0);
      for (d = 1; d < ndim; d++)
      {
         width = nalu_hypre_BoxSizeD(box, d);
         if (width < wmin)
         {
            wmin = width;
         }
         else if (width > wmax)
         {
            dmax = d;
            wmax = width;
         }
      }

      /* Notation (all real numbers):
         rn_cubes      - number of wmin-width cubes in the region
         rn_cube_procs - number of procs per wmin-width cube
         rn_cube_divs  - number of divs per wmin-width cube */

      /* After computing the above, each div_index[d] is set by first flooring
         rn_cube_divs, then div_index[dmax] is incremented until we have more
         partitions than processors. */

      rn_cubes = nalu_hypre_doubleBoxVolume(box) / pow(wmin, ndim);
      rn_cube_procs = proc_count / rn_cubes;
      rn_cube_divs = pow(rn_cube_procs, (1.0 / (NALU_HYPRE_Real)ndim));

      for (d = 0; d < ndim; d++)
      {
         width = nalu_hypre_BoxSizeD(box, d);
         rdiv = rn_cube_divs * (width / wmin);
         /* Add a small number to compensate for roundoff issues */
         nalu_hypre_IndexD(div_index, d) = (NALU_HYPRE_Int) floor(rdiv + 1.0e-6);
         /* Make sure div_index[d] is at least 1 */
         nalu_hypre_IndexD(div_index, d) = nalu_hypre_max(nalu_hypre_IndexD(div_index, d), 1);
      }

      /* Decrease div_index to ensure no more than 2 partitions per processor.
       * This is only needed when div_index[d] is adjusted to 1 above. */
      while (nalu_hypre_IndexProd(div_index, ndim) >= 2 * proc_count)
      {
         /* Decrease the max dimension by a factor of 2 without going below 1 */
         nalu_hypre_IndexD(div_index, dmax) = (nalu_hypre_IndexD(div_index, dmax) + 1) / 2;
         for (d = 0; d < ndim; d++)
         {
            if (nalu_hypre_IndexD(div_index, d) > nalu_hypre_IndexD(div_index, dmax))
            {
               dmax = d;
            }
         }
      }

      /* Increment div_index[dmax] to ensure more partitions than processors.
         This can never result in more than 2 partitions per processor. */
      while (nalu_hypre_IndexProd(div_index, ndim) < proc_count)
      {
         nalu_hypre_IndexD(div_index, dmax) ++;
      }

      nalu_hypre_CopyIndex(div_index, nalu_hypre_StructAssumedPartDivision(assumed_part, i));

#if REGION_STAT
      if ( myid == 0 )
      {
         nalu_hypre_printf("region = %d, proc_count = %d, divisions = [", i, proc_count);
         for (d = 0; d < ndim; d++)
         {
            nalu_hypre_printf(" %d", nalu_hypre_IndexD(div_index, d));
         }
         nalu_hypre_printf("]\n");
      }
#endif
   } /* End of FOR EACH REGION loop */

   /* NOW WE HAVE COMPLETED GLOBAL INFO - START FILLING IN LOCAL INFO */

   /* We need to populate the assumed partition object with info specific to
      each processor, like which assumed partition we own, which boxes are in
      that region, etc. */

   /* Figure out my partition region and put it in the assumed_part structure */
   nalu_hypre_StructAssumedPartMyPartition(assumed_part) = nalu_hypre_BoxArrayCreate(2, ndim);
   my_partition = nalu_hypre_StructAssumedPartMyPartition(assumed_part);
   nalu_hypre_StructAssumedPartitionGetRegionsFromProc(assumed_part, myid, my_partition);
#if 0
   nalu_hypre_ForBoxI(i, my_partition)
   {
      box = nalu_hypre_BoxArrayBox(my_partition, i);
      nalu_hypre_printf("myid = %d: MY ASSUMED Partitions (%d):  (%d, %d, %d)  x  "
                   "(%d, %d, %d)\n",
                   myid, i,
                   nalu_hypre_BoxIMinX(box),
                   nalu_hypre_BoxIMinY(box),
                   nalu_hypre_BoxIMinZ(box),
                   nalu_hypre_BoxIMaxX(box),
                   nalu_hypre_BoxIMaxY(box),
                   nalu_hypre_BoxIMaxZ(box));
   }
#endif

   /* Find out which boxes are in my partition: Look through my boxes, figure
      out which assumed parition (AP) they fall in and contact that processor.
      Use the exchange data functionality for this. */

   proc_alloc = nalu_hypre_pow2(ndim);
   proc_array = nalu_hypre_TReAlloc(proc_array,  NALU_HYPRE_Int,  proc_alloc, NALU_HYPRE_MEMORY_HOST);

   /* Probably there will mostly be one proc per box */
   /* Don't want to allocate too much memory here */
   size = 1.2 * nalu_hypre_BoxArraySize(local_boxes);

   /* Each local box may live on multiple procs in the assumed partition */
   tmp_proc_ids = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST); /* local box proc ids */
   tmp_box_nums = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST); /* local box boxnum */
   tmp_box_inds = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST); /* local box array index */

   proc_count = 0;
   count = 0; /* Current number of procs */
   grow_box = nalu_hypre_BoxCreate(ndim);

   nalu_hypre_ForBoxI(i, local_boxes)
   {
      box = nalu_hypre_BoxArrayBox(local_boxes, i);

      nalu_hypre_StructAssumedPartitionGetProcsFromBox(
         assumed_part, box, &proc_count, &proc_alloc, &proc_array);
      /* Do we need more storage? */
      if ((count + proc_count) > size)
      {
         size = size + proc_count + 1.2 * (nalu_hypre_BoxArraySize(local_boxes) - i);
         /* nalu_hypre_printf("myid = %d, *adjust* alloc size = %d\n", myid, size);*/
         tmp_proc_ids = nalu_hypre_TReAlloc(tmp_proc_ids,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
         tmp_box_nums = nalu_hypre_TReAlloc(tmp_box_nums,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
         tmp_box_inds = nalu_hypre_TReAlloc(tmp_box_inds,  NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
      }
      for (j = 0; j < proc_count; j++)
      {
         tmp_proc_ids[count] = proc_array[j];
         tmp_box_nums[count] = local_boxnums[i];
         tmp_box_inds[count] = i;
         count++;
      }
   }

   nalu_hypre_BoxDestroy(grow_box);

   /* Now we have two arrays: tmp_proc_ids and tmp_box_nums.  These are
      corresponding box numbers and proc ids.  We need to sort the processor ids
      and then create a new buffer to send to the exchange data function. */

   /* Sort the proc_ids */
   nalu_hypre_qsort3i(tmp_proc_ids, tmp_box_nums, tmp_box_inds, 0, count - 1);

   /* Use proc_array for the processor ids to contact.  Use box array to get our
      boxes and then pass the array only (not the structure) to exchange data. */
   box_count = count;

   contact_boxinfo = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  box_count * (1 + 2 * ndim), NALU_HYPRE_MEMORY_HOST);

   proc_array = nalu_hypre_TReAlloc(proc_array,  NALU_HYPRE_Int,  box_count, NALU_HYPRE_MEMORY_HOST);
   proc_array_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  box_count + 1, NALU_HYPRE_MEMORY_HOST);
   proc_array_starts[0] = 0;

   proc_count = 0;
   index = 0;

   if (box_count)
   {
      proc_array[0] = tmp_proc_ids[0];

      contact_boxinfo[index++] = tmp_box_nums[0];
      box = nalu_hypre_BoxArrayBox(local_boxes, tmp_box_inds[0]);
      for (d = 0; d < ndim; d++)
      {
         contact_boxinfo[index++] = nalu_hypre_BoxIMinD(box, d);
         contact_boxinfo[index++] = nalu_hypre_BoxIMaxD(box, d);
      }
      proc_count++;
   }

   for (i = 1; i < box_count; i++)
   {
      if (tmp_proc_ids[i]  != proc_array[proc_count - 1])
      {
         proc_array[proc_count] = tmp_proc_ids[i];
         proc_array_starts[proc_count] = i;
         proc_count++;
      }

      /* These boxes are not copied in a particular order */

      contact_boxinfo[index++] = tmp_box_nums[i];
      box = nalu_hypre_BoxArrayBox(local_boxes, tmp_box_inds[i]);
      for (d = 0; d < ndim; d++)
      {
         contact_boxinfo[index++] = nalu_hypre_BoxIMinD(box, d);
         contact_boxinfo[index++] = nalu_hypre_BoxIMaxD(box, d);
      }
   }
   proc_array_starts[proc_count] = box_count;

   /* Clean up */
   nalu_hypre_TFree(tmp_proc_ids, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(tmp_box_nums, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(tmp_box_inds, NALU_HYPRE_MEMORY_HOST);

   /* EXCHANGE DATA */

   /* Prepare to populate the local info in the assumed partition */
   nalu_hypre_StructAssumedPartMyPartitionBoxes(assumed_part)
      = nalu_hypre_BoxArrayCreate(box_count, ndim);
   nalu_hypre_BoxArraySetSize(nalu_hypre_StructAssumedPartMyPartitionBoxes(assumed_part), 0);
   nalu_hypre_StructAssumedPartMyPartitionIdsSize(assumed_part) = 0;
   nalu_hypre_StructAssumedPartMyPartitionIdsAlloc(assumed_part) = box_count;
   nalu_hypre_StructAssumedPartMyPartitionProcIds(assumed_part)
      = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  box_count, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_StructAssumedPartMyPartitionBoxnums(assumed_part)
      = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  box_count, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_StructAssumedPartMyPartitionNumDistinctProcs(assumed_part) = 0;

   /* Set up for exchanging data */
   /* The response we expect is just a confirmation */
   response_buf = NULL;
   response_buf_starts = NULL;

   /* Response object */
   response_obj.fill_response = nalu_hypre_APFillResponseStructAssumedPart;
   response_obj.data1 = assumed_part; /* Where we keep info from contacts */
   response_obj.data2 = NULL;

   max_response_size = 0; /* No response data - just confirmation */

   nalu_hypre_DataExchangeList(proc_count, proc_array,
                          contact_boxinfo, proc_array_starts,
                          (1 + 2 * ndim)*sizeof(NALU_HYPRE_Int),
                          sizeof(NALU_HYPRE_Int), &response_obj, max_response_size, 1,
                          comm, (void**) &response_buf, &response_buf_starts);

   nalu_hypre_TFree(proc_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(proc_array_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(response_buf, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(response_buf_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(contact_boxinfo, NALU_HYPRE_MEMORY_HOST);

   /* Return vars */
   *p_assumed_partition = assumed_part;

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * Destroy the assumed partition.
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_StructAssumedPartitionDestroy( nalu_hypre_StructAssumedPart *assumed_part )
{
   if (assumed_part)
   {
      nalu_hypre_BoxArrayDestroy( nalu_hypre_StructAssumedPartRegions(assumed_part));
      nalu_hypre_TFree(nalu_hypre_StructAssumedPartProcPartitions(assumed_part), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_StructAssumedPartDivisions(assumed_part), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_BoxArrayDestroy( nalu_hypre_StructAssumedPartMyPartition(assumed_part));
      nalu_hypre_BoxArrayDestroy( nalu_hypre_StructAssumedPartMyPartitionBoxes(assumed_part));
      nalu_hypre_TFree(nalu_hypre_StructAssumedPartMyPartitionProcIds(assumed_part), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree( nalu_hypre_StructAssumedPartMyPartitionBoxnums(assumed_part), NALU_HYPRE_MEMORY_HOST);

      /* This goes last! */
      nalu_hypre_TFree(assumed_part, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * fillResponseStructAssumedPart
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_APFillResponseStructAssumedPart(
   void      *p_recv_contact_buf,
   NALU_HYPRE_Int  contact_size,
   NALU_HYPRE_Int  contact_proc,
   void      *ro,
   MPI_Comm   comm,
   void     **p_send_response_buf,
   NALU_HYPRE_Int *response_message_size )
{
   NALU_HYPRE_Int    ndim, size, alloc_size, myid, i, d, index;
   NALU_HYPRE_Int   *ids, *boxnums;
   NALU_HYPRE_Int   *recv_contact_buf;

   nalu_hypre_Box   *box;

   nalu_hypre_BoxArray              *part_boxes;
   nalu_hypre_DataExchangeResponse  *response_obj = (nalu_hypre_DataExchangeResponse  *)ro;
   nalu_hypre_StructAssumedPart     *assumed_part = (nalu_hypre_StructAssumedPart     *)response_obj->data1;

   /* Initialize stuff */
   nalu_hypre_MPI_Comm_rank(comm, &myid );

   ndim = nalu_hypre_StructAssumedPartNDim(assumed_part);
   part_boxes =  nalu_hypre_StructAssumedPartMyPartitionBoxes(assumed_part);
   ids = nalu_hypre_StructAssumedPartMyPartitionProcIds(assumed_part);
   boxnums = nalu_hypre_StructAssumedPartMyPartitionBoxnums(assumed_part);

   size =  nalu_hypre_StructAssumedPartMyPartitionIdsSize(assumed_part);
   alloc_size = nalu_hypre_StructAssumedPartMyPartitionIdsAlloc(assumed_part);

   recv_contact_buf = (NALU_HYPRE_Int * ) p_recv_contact_buf;

   /* Increment how many procs have contacted us */
   nalu_hypre_StructAssumedPartMyPartitionNumDistinctProcs(assumed_part)++;

   /* Check to see if we need to allocate more space for ids and boxnums */
   if ((size + contact_size) > alloc_size)
   {
      alloc_size = size + contact_size;
      ids = nalu_hypre_TReAlloc(ids,  NALU_HYPRE_Int,  alloc_size, NALU_HYPRE_MEMORY_HOST);
      boxnums = nalu_hypre_TReAlloc(boxnums,  NALU_HYPRE_Int,  alloc_size, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_StructAssumedPartMyPartitionIdsAlloc(assumed_part) = alloc_size;
   }

   box = nalu_hypre_BoxCreate(ndim);

   /* Populate our assumed partition according to boxes received */
   index = 0;
   for (i = 0; i < contact_size; i++)
   {
      ids[size + i] = contact_proc; /* Set the proc id */
      boxnums[size + i] = recv_contact_buf[index++];
      for (d = 0; d < ndim; d++)
      {
         nalu_hypre_BoxIMinD(box, d) = recv_contact_buf[index++];
         nalu_hypre_BoxIMaxD(box, d) = recv_contact_buf[index++];
      }

      nalu_hypre_AppendBox(box, part_boxes);
   }
   /* Adjust the size of the proc ids*/
   nalu_hypre_StructAssumedPartMyPartitionIdsSize(assumed_part) = size + contact_size;

   /* In case more memory was allocated we have to assign these pointers back */
   nalu_hypre_StructAssumedPartMyPartitionBoxes(assumed_part) = part_boxes;
   nalu_hypre_StructAssumedPartMyPartitionProcIds(assumed_part) = ids;
   nalu_hypre_StructAssumedPartMyPartitionBoxnums(assumed_part) = boxnums;

   /* Output - no message to return (confirmation) */
   *response_message_size = 0;

   nalu_hypre_BoxDestroy(box);

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * Given a processor id, get that processor's assumed region(s).
 *
 * At most a processor has 2 assumed regions.  Pass in a BoxArray of size 2.
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_StructAssumedPartitionGetRegionsFromProc(
   nalu_hypre_StructAssumedPart *assumed_part,
   NALU_HYPRE_Int                proc_id,
   nalu_hypre_BoxArray          *assumed_regions )
{
   NALU_HYPRE_Int   *proc_partitions;
   NALU_HYPRE_Int    ndim, i, d;
   NALU_HYPRE_Int    in_region, proc_count, proc_start, num_partitions;
   NALU_HYPRE_Int    part_num, width, extra;
   NALU_HYPRE_Int    adj_proc_id;
   NALU_HYPRE_Int    num_assumed, num_regions;

   nalu_hypre_Box   *region, *box;
   nalu_hypre_Index  div, divindex, rsize, imin, imax;
   NALU_HYPRE_Int    divi;

   ndim = nalu_hypre_StructAssumedPartNDim(assumed_part);
   num_regions = nalu_hypre_StructAssumedPartNumRegions(assumed_part);
   proc_partitions = nalu_hypre_StructAssumedPartProcPartitions(assumed_part);

   /* Check if this processor owns an assumed region.  It is rare that it won't
      (only if # procs > bounding box or # procs > global #boxes). */

   if (proc_id >= proc_partitions[num_regions])
   {
      /* Owns no boxes */
      num_assumed = 0;
   }
   else
   {
      /* Which partition region am I in? */
      in_region = 0;
      if (num_regions > 1)
      {
         while (proc_id >= proc_partitions[in_region + 1])
         {
            in_region++;
         }
      }

      /* First processor in the range */
      proc_start = proc_partitions[in_region];
      /* How many processors in that region? */
      proc_count = proc_partitions[in_region + 1] - proc_partitions[in_region];
      /* Get the region */
      region = nalu_hypre_BoxArrayBox(nalu_hypre_StructAssumedPartRegions(assumed_part),
                                 in_region);
      /* Size of the regions */
      nalu_hypre_BoxGetSize(region, rsize);
      /* Get the divisions in each dimension */
      nalu_hypre_CopyIndex(nalu_hypre_StructAssumedPartDivision(assumed_part, in_region),
                      div);

      /* Calculate the assumed partition(s) (at most 2) that I own */

      num_partitions = nalu_hypre_IndexProd(div, ndim);
      /* How many procs have 2 partitions instead of one*/
      extra =  num_partitions % proc_count;

      /* Adjust the proc number to range from 0 to (proc_count-1) */
      adj_proc_id = proc_id - proc_start;

      /* The region is divided into num_partitions partitions according to the
         number of divisions in each direction.  Some processors may own more
         than one partition (up to 2).  These partitions are numbered by
         dimension 0 first, then dimension 1, etc.  From the partition number,
         we can calculate the processor id. */

      /* Get my partition number */
      if (adj_proc_id < extra)
      {
         part_num = adj_proc_id * 2;
         num_assumed = 2;
      }
      else
      {
         part_num = extra + adj_proc_id;
         num_assumed = 1;
      }
   }

   /* Make sure BoxArray has been allocated for num_assumed boxes */
   nalu_hypre_BoxArraySetSize(assumed_regions, num_assumed);

   for (i = 0; i < num_assumed; i++)
   {
      nalu_hypre_IndexFromRank(part_num + i, div, divindex, ndim);

      for (d = ndim - 1; d >= 0; d--)
      {
         width = nalu_hypre_IndexD(rsize, d) / nalu_hypre_IndexD(div, d);
         extra = nalu_hypre_IndexD(rsize, d) % nalu_hypre_IndexD(div, d);

         divi = nalu_hypre_IndexD(divindex, d);
         nalu_hypre_IndexD(imin, d) = divi * width + nalu_hypre_min(divi, extra);
         divi = nalu_hypre_IndexD(divindex, d) + 1;
         nalu_hypre_IndexD(imax, d) = divi * width + nalu_hypre_min(divi, extra) - 1;

         /* Change relative coordinates to absolute */
         nalu_hypre_IndexD(imin, d) +=  nalu_hypre_BoxIMinD(region, d);
         nalu_hypre_IndexD(imax, d) +=  nalu_hypre_BoxIMinD(region, d);
      }

      /* Set the assumed region*/
      box = nalu_hypre_BoxArrayBox(assumed_regions, i);
      nalu_hypre_BoxSetExtents(box, imin, imax);
   }

   return nalu_hypre_error_flag;
}

/******************************************************************************
 * Given a box, which processor(s) assumed partition does the box intersect.
 *
 * proc_array should be allocated to size_alloc_proc_array
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_StructAssumedPartitionGetProcsFromBox(
   nalu_hypre_StructAssumedPart *assumed_part,
   nalu_hypre_Box               *box,
   NALU_HYPRE_Int               *num_proc_array,
   NALU_HYPRE_Int               *size_alloc_proc_array,
   NALU_HYPRE_Int              **p_proc_array )
{
   NALU_HYPRE_Int       ndim = nalu_hypre_StructAssumedPartNDim(assumed_part);

   NALU_HYPRE_Int       i, d, p, q, r, myid;
   NALU_HYPRE_Int       num_regions, in_regions, this_region, proc_count, proc_start;
   NALU_HYPRE_Int       adj_proc_id, extra, num_partitions;
   NALU_HYPRE_Int       width;

   NALU_HYPRE_Int      *proc_array, proc_array_count;
   NALU_HYPRE_Int      *which_regions;
   NALU_HYPRE_Int      *proc_ids, num_proc_ids, size_proc_ids, ncorners;

   nalu_hypre_Box      *region;
   nalu_hypre_Box      *result_box, *part_box, *part_dbox;
   nalu_hypre_Index     div, rsize, stride, loop_size;
   nalu_hypre_IndexRef  start;
   nalu_hypre_BoxArray *region_array;
   NALU_HYPRE_Int      *proc_partitions;

   /* Need myid only for the nalu_hypre_printf statement */
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid);

   proc_array = *p_proc_array;
   region_array = nalu_hypre_StructAssumedPartRegions(assumed_part);
   num_regions = nalu_hypre_StructAssumedPartNumRegions(assumed_part);
   proc_partitions = nalu_hypre_StructAssumedPartProcPartitions(assumed_part);

   /* First intersect the box to find out which region(s) it lies in, then
      determine which processor owns the assumed part of these regions(s) */

   result_box = nalu_hypre_BoxCreate(ndim);
   part_box = nalu_hypre_BoxCreate(ndim);
   part_dbox = nalu_hypre_BoxCreate(ndim);
   which_regions = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_regions, NALU_HYPRE_MEMORY_HOST);

   /* The number of corners in a box is a good initial size for proc_ids */
   ncorners = nalu_hypre_pow2(ndim);
   size_proc_ids = ncorners;
   proc_ids = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size_proc_ids, NALU_HYPRE_MEMORY_HOST);
   num_proc_ids = 0;

   /* which partition region(s) am i in? */
   in_regions = 0;
   for (i = 0; i < num_regions; i++)
   {
      region = nalu_hypre_BoxArrayBox(region_array, i);
      nalu_hypre_IntersectBoxes(box, region, result_box);
      if (  nalu_hypre_BoxVolume(result_box) > 0 )
      {
         which_regions[in_regions] = i;
         in_regions++;
      }
   }

#if 0
   if (in_regions == 0)
   {
      /* 9/16/10 - In nalu_hypre_SStructGridAssembleBoxManagers we grow boxes by 1
         before we gather boxes because of shared variables, so we can get the
         situation that the gather box is outside of the assumed region. */

      if (nalu_hypre_BoxVolume(box) > 0)
      {
         nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
         nalu_hypre_printf("MY_ID = %d Error: positive volume box (%d, %d, %d) x "
                      "(%d, %d, %d)  not in any assumed regions! (this should never"
                      " happen)\n",
                      myid,
                      nalu_hypre_BoxIMinX(box),
                      nalu_hypre_BoxIMinY(box),
                      nalu_hypre_BoxIMinZ(box),
                      nalu_hypre_BoxIMaxX(box),
                      nalu_hypre_BoxIMaxY(box),
                      nalu_hypre_BoxIMaxZ(box));
      }
   }
#endif

   /* For each region, who is assumed to own this box?  Add the proc number to
      proc array. */
   for (r = 0; r < in_regions; r++)
   {
      /* Initialization for this particular region */
      this_region = which_regions[r];
      region = nalu_hypre_BoxArrayBox(region_array, this_region);
      /* First processor in the range */
      proc_start = proc_partitions[this_region];
      /* How many processors in that region? */
      proc_count = proc_partitions[this_region + 1] - proc_start;
      /* Size of the regions */
      nalu_hypre_BoxGetSize(region, rsize);
      /* Get the divisons in each dimension */
      nalu_hypre_CopyIndex(nalu_hypre_StructAssumedPartDivision(assumed_part, this_region),
                      div);

      /* Intersect box with region */
      nalu_hypre_IntersectBoxes(box, region, result_box);

      /* Compute part_box (the intersected assumed partitions) from result_box.
         Start part index number from 1 for convenience in BoxLoop below. */
      for (d = 0; d < ndim; d++)
      {
         width = nalu_hypre_IndexD(rsize, d) / nalu_hypre_IndexD(div, d);
         extra = nalu_hypre_IndexD(rsize, d) % nalu_hypre_IndexD(div, d);

         /* imin component, shifted by region imin */
         i = nalu_hypre_BoxIMinD(result_box, d) - nalu_hypre_BoxIMinD(region, d);
         p = i / (width + 1);
         if (p < extra)
         {
            nalu_hypre_BoxIMinD(part_box, d) = p + 1;
         }
         else
         {
            q = (i - extra * (width + 1)) / width;
            nalu_hypre_BoxIMinD(part_box, d) = extra + q + 1;
         }

         /* imax component, shifted by region imin  */
         i = nalu_hypre_BoxIMaxD(result_box, d) - nalu_hypre_BoxIMinD(region, d);
         p = i / (width + 1);
         if (p < extra)
         {
            nalu_hypre_BoxIMaxD(part_box, d) = p + 1;
         }
         else
         {
            q = (i - extra * (width + 1)) / width;
            nalu_hypre_BoxIMaxD(part_box, d) = extra + q + 1;
         }
      }

      /* Number of partitions in this region? */
      num_partitions = nalu_hypre_IndexProd(div, ndim);
      /* How many procs have 2 partitions instead of one*/
      extra =  num_partitions % proc_count;

      /* Compute part_num for each index in part_box and get proc_ids */
      start = nalu_hypre_BoxIMin(part_box);
      nalu_hypre_SetIndex(stride, 1);
      nalu_hypre_BoxGetSize(part_box, loop_size);
      nalu_hypre_BoxSetExtents(part_dbox, stride, div);
      nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size, part_dbox, start, stride, part_num);
      {
         /*convert the partition number to a processor number*/
         if (part_num < (2 * extra))
         {
            adj_proc_id = part_num / 2 ;
         }
         else
         {
            adj_proc_id =  extra + (part_num - 2 * extra);
         }

         if (num_proc_ids == size_proc_ids)
         {
            size_proc_ids += ncorners;
            proc_ids = nalu_hypre_TReAlloc(proc_ids,  NALU_HYPRE_Int,  size_proc_ids, NALU_HYPRE_MEMORY_HOST);
         }

         proc_ids[num_proc_ids] = adj_proc_id + proc_start;
         num_proc_ids++;
      }
      nalu_hypre_SerialBoxLoop1End(part_num);

   } /*end of for each region loop*/

   if (in_regions)
   {
      /* Determine unique proc_ids (could be duplicates due to a processor
         owning more than one partiton in a region).  Sort the array. */
      nalu_hypre_qsort0(proc_ids, 0, num_proc_ids - 1);

      /* Make sure we have enough space from proc_array */
      if (*size_alloc_proc_array < num_proc_ids)
      {
         proc_array = nalu_hypre_TReAlloc(proc_array,  NALU_HYPRE_Int,  num_proc_ids, NALU_HYPRE_MEMORY_HOST);
         *size_alloc_proc_array = num_proc_ids;
      }

      /* Put unique values in proc_array */
      proc_array[0] = proc_ids[0]; /* There will be at least one processor id */
      proc_array_count = 1;
      for (i = 1; i < num_proc_ids; i++)
      {
         if  (proc_ids[i] != proc_array[proc_array_count - 1])
         {
            proc_array[proc_array_count] = proc_ids[i];
            proc_array_count++;
         }
      }
   }
   else /* No processors for this box */
   {
      proc_array_count = 0;
   }

   /* Return variables */
   *p_proc_array = proc_array;
   *num_proc_array = proc_array_count;

   /* Clean up*/
   nalu_hypre_BoxDestroy(result_box);
   nalu_hypre_BoxDestroy(part_box);
   nalu_hypre_BoxDestroy(part_dbox);
   nalu_hypre_TFree(which_regions, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(proc_ids, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

#if 0
/******************************************************************************
 * UNFINISHED
 *
 * Create a new assumed partition by coarsening another assumed partition.
 *
 * Unfinished because of a problem: Can't figure out what the new id is since
 * the zero boxes drop out, and we don't have all of the boxes from a particular
 * processor in the AP.  This may not be a problem any longer (see [issue708]).
 *****************************************************************************/

NALU_HYPRE_Int
nalu_hypre_StructCoarsenAP(nalu_hypre_StructAssumedPart  *ap,
                      nalu_hypre_Index               index,
                      nalu_hypre_Index               stride,
                      nalu_hypre_StructAssumedPart **new_ap_ptr )
{
   NALU_HYPRE_Int num_regions;

   nalu_hypre_BoxArray *coarse_boxes;
   nalu_hypre_BoxArray *fine_boxes;
   nalu_hypre_BoxArray *regions_array;
   nalu_hypre_Box      *box, *new_box;

   nalu_hypre_StructAssumedPartition *new_ap;

   /* Create new ap and copy global description information */
   new_ap = nalu_hypre_TAlloc(nalu_hypre_StructAssumedPart,  1, NALU_HYPRE_MEMORY_HOST);

   num_regions = nalu_hypre_StructAssumedPartNumRegions(ap);
   regions_array = nalu_hypre_BoxArrayCreate(num_regions, ndim);

   nalu_hypre_StructAssumedPartRegions(new_ap) = regions_array;
   nalu_hypre_StructAssumedPartNumRegions(new_ap) = num_regions;
   nalu_hypre_StructAssumedPartProcPartitions(new_ap) =
      nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_regions + 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_StructAssumedPartDivisions(new_ap) =
      nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_regions, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_StructAssumedPartProcPartitions(new_ap)[0] = 0;

   for (i = 0; i < num_regions; i++)
   {
      box =  nalu_hypre_BoxArrayBox(nalu_hypre_StructAssumedPartRegions(ap), i);

      nalu_hypre_CopyBox(box, nalu_hypre_BoxArrayBox(regions_array, i));

      nalu_hypre_StructAssumedPartDivision(new_ap, i) =
         nalu_hypre_StructAssumedPartDivision(new_ap, i);

      nalu_hypre_StructAssumedPartProcPartition(new_ap, i + 1) =
         nalu_hypre_StructAssumedPartProcPartition(ap, i + 1);
   }

   /* Copy my partition (at most 2 boxes)*/
   nalu_hypre_StructAssumedPartMyPartition(new_ap) = nalu_hypre_BoxArrayCreate(2, ndim);
   for (i = 0; i < 2; i++)
   {
      box     = nalu_hypre_BoxArrayBox(nalu_hypre_StructAssumedPartMyPartition(ap), i);
      new_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructAssumedPartMyPartition(new_ap), i);
      nalu_hypre_CopyBox(box, new_box);
   }

   /* Create space for the boxes, ids and boxnums */
   size = nalu_hypre_StructAssumedPartMyPartitionIdsSize(ap);

   nalu_hypre_StructAssumedPartMyPartitionProcIds(new_ap) =
      nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_StructAssumedPartMyPartitionBoxnums(new_ap) =
      nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_StructAssumedPartMyPartitionBoxes(new_ap)
      = nalu_hypre_BoxArrayCreate(size, ndim);

   nalu_hypre_StructAssumedPartMyPartitionIdsAlloc(new_ap) = size;
   nalu_hypre_StructAssumedPartMyPartitionIdsSize(new_ap) = size;

   /* Coarsen and copy the boxes.  Need to prune size 0 boxes. */
   coarse_boxes = nalu_hypre_StructAssumedPartMyPartitionBoxes(new_ap);
   fine_boxes =  nalu_hypre_StructAssumedPartMyPartitionBoxes(ap);

   new_box = nalu_hypre_BoxCreate(ndim);

   nalu_hypre_ForBoxI(i, fine_boxes)
   {
      box =  nalu_hypre_BoxArrayBox(fine_boxes, i);
      nalu_hypre_CopyBox(box, new_box);
      nalu_hypre_StructCoarsenBox(new_box, index, stride);
   }

   /* Unfinished because of a problem: Can't figure out what the new id is since
      the zero boxes drop out, and we don't have all of the boxes from a
      particular processor in the AP */

   /* nalu_hypre_StructAssumedPartMyPartitionNumDistinctProcs(new_ap) */

   *new_ap_ptr = new_ap;

   return nalu_hypre_error_flag;
}
#endif
