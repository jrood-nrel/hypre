/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_StructGrid class.
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"

#define DEBUG 0

#if DEBUG
char       filename[255];
FILE      *file;
NALU_HYPRE_Int  my_rank;
#endif

static NALU_HYPRE_Int time_index = 0;

/*--------------------------------------------------------------------------
 * nalu_hypre_StructGridCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridCreate( MPI_Comm           comm,
                        NALU_HYPRE_Int          ndim,
                        nalu_hypre_StructGrid **grid_ptr)
{
   nalu_hypre_StructGrid    *grid;
   NALU_HYPRE_Int           i;

   grid = nalu_hypre_TAlloc(nalu_hypre_StructGrid,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_StructGridComm(grid)        = comm;
   nalu_hypre_StructGridNDim(grid)        = ndim;
   nalu_hypre_StructGridBoxes(grid)       = nalu_hypre_BoxArrayCreate(0, ndim);
   nalu_hypre_StructGridIDs(grid)         = NULL;

   nalu_hypre_SetIndex(nalu_hypre_StructGridMaxDistance(grid), 8);

   nalu_hypre_StructGridBoundingBox(grid) = NULL;
   nalu_hypre_StructGridLocalSize(grid)   = 0;
   nalu_hypre_StructGridGlobalSize(grid)  = 0;
   nalu_hypre_SetIndex(nalu_hypre_StructGridPeriodic(grid), 0);
   nalu_hypre_StructGridRefCount(grid)     = 1;
   nalu_hypre_StructGridBoxMan(grid)       = NULL;

   nalu_hypre_StructGridNumPeriods(grid)   = 1;
   nalu_hypre_StructGridPShifts(grid)     = NULL;

   nalu_hypre_StructGridGhlocalSize(grid)  = 0;
   for (i = 0; i < 2 * ndim; i++)
   {
      nalu_hypre_StructGridNumGhost(grid)[i] = 1;
   }

   *grid_ptr = grid;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructGridRef
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridRef( nalu_hypre_StructGrid  *grid,
                     nalu_hypre_StructGrid **grid_ref)
{
   nalu_hypre_StructGridRefCount(grid) ++;
   *grid_ref = grid;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructGridDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridDestroy( nalu_hypre_StructGrid *grid )
{
   if (grid)
   {
      nalu_hypre_StructGridRefCount(grid) --;
      if (nalu_hypre_StructGridRefCount(grid) == 0)
      {
         nalu_hypre_BoxDestroy(nalu_hypre_StructGridBoundingBox(grid));
         nalu_hypre_TFree(nalu_hypre_StructGridIDs(grid), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_BoxArrayDestroy(nalu_hypre_StructGridBoxes(grid));

         nalu_hypre_BoxManDestroy(nalu_hypre_StructGridBoxMan(grid));
         nalu_hypre_TFree( nalu_hypre_StructGridPShifts(grid), NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_TFree(grid, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_StructGridSetPeriodic
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridSetPeriodic( nalu_hypre_StructGrid  *grid,
                             nalu_hypre_Index        periodic)
{
   nalu_hypre_CopyIndex(periodic, nalu_hypre_StructGridPeriodic(grid));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructGridSetExtents
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridSetExtents( nalu_hypre_StructGrid  *grid,
                            nalu_hypre_Index        ilower,
                            nalu_hypre_Index        iupper )
{
   nalu_hypre_Box   *box;

   box = nalu_hypre_BoxCreate(nalu_hypre_StructGridNDim(grid));
   nalu_hypre_BoxSetExtents(box, ilower, iupper);
   nalu_hypre_AppendBox(box, nalu_hypre_StructGridBoxes(grid));
   nalu_hypre_BoxDestroy(box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructGridSetBoxes
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridSetBoxes( nalu_hypre_StructGrid *grid,
                          nalu_hypre_BoxArray   *boxes )
{

   nalu_hypre_TFree(nalu_hypre_StructGridBoxes(grid), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_StructGridBoxes(grid) = boxes;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructGridSetBoundingBox
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridSetBoundingBox( nalu_hypre_StructGrid *grid,
                                nalu_hypre_Box   *new_bb )
{

   nalu_hypre_BoxDestroy(nalu_hypre_StructGridBoundingBox(grid));
   nalu_hypre_StructGridBoundingBox(grid) = nalu_hypre_BoxDuplicate(new_bb);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructGridSetIDs
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridSetIDs( nalu_hypre_StructGrid *grid,
                        NALU_HYPRE_Int   *ids )
{
   nalu_hypre_TFree(nalu_hypre_StructGridIDs(grid), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_StructGridIDs(grid) = ids;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructGridSetBoxManager
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridSetBoxManager( nalu_hypre_StructGrid *grid,
                               nalu_hypre_BoxManager *boxman )
{

   nalu_hypre_TFree(nalu_hypre_StructGridBoxMan(grid), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_StructGridBoxMan(grid) = boxman;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructGridSetMaxDistance
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridSetMaxDistance( nalu_hypre_StructGrid *grid,
                                nalu_hypre_Index dist )
{
   nalu_hypre_CopyIndex(dist, nalu_hypre_StructGridMaxDistance(grid));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * New - nalu_hypre_StructGridAssemble
 * AHB 9/06
 * New assemble routine that uses the BoxManager structure
 *
 *   Notes:
 *   1. No longer need a different assemble for the assumed partition case
 *   2. if this is called from StructCoarsen, then the Box Manager has already
 *   been created, and ids have been set
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridAssemble( nalu_hypre_StructGrid *grid )
{

   NALU_HYPRE_Int d, k, p, i;

   NALU_HYPRE_Int is_boxman;
   NALU_HYPRE_Int size, ghostsize;
   NALU_HYPRE_Int num_local_boxes;
   NALU_HYPRE_Int myid, num_procs;
   NALU_HYPRE_BigInt global_size;
   NALU_HYPRE_Int max_nentries;
   NALU_HYPRE_Int info_size;
   NALU_HYPRE_Int num_periods;

   NALU_HYPRE_Int *ids = NULL;
   NALU_HYPRE_Int  iperiodic, notcenter;

   NALU_HYPRE_Int  sendbuf6[2 * NALU_HYPRE_MAXDIM], recvbuf6[2 * NALU_HYPRE_MAXDIM];

   nalu_hypre_Box  *box;
   nalu_hypre_Box  *ghostbox;
   nalu_hypre_Box  *grow_box;
   nalu_hypre_Box  *periodic_box;
   nalu_hypre_Box  *result_box;

   nalu_hypre_Index min_index, max_index, loop_size;
   nalu_hypre_Index *pshifts;
   nalu_hypre_IndexRef pshift;

   void *entry_info = NULL;

   /*  initialize info from the grid */
   MPI_Comm             comm         = nalu_hypre_StructGridComm(grid);
   NALU_HYPRE_Int            ndim         = nalu_hypre_StructGridNDim(grid);
   nalu_hypre_BoxArray      *local_boxes  = nalu_hypre_StructGridBoxes(grid);
   nalu_hypre_IndexRef       max_distance = nalu_hypre_StructGridMaxDistance(grid);
   nalu_hypre_Box           *bounding_box = nalu_hypre_StructGridBoundingBox(grid);
   nalu_hypre_IndexRef       periodic     = nalu_hypre_StructGridPeriodic(grid);
   nalu_hypre_BoxManager    *boxman       = nalu_hypre_StructGridBoxMan(grid);
   NALU_HYPRE_Int           *numghost     = nalu_hypre_StructGridNumGhost(grid);

   if (!time_index)
   {
      time_index = nalu_hypre_InitializeTiming("StructGridAssemble");
   }

   nalu_hypre_BeginTiming(time_index);

   /* other initializations */
   num_local_boxes = nalu_hypre_BoxArraySize(local_boxes);

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &myid);

   /* has the box manager been created? */
   if (boxman == NULL)
   {
      is_boxman = 0;
   }
   else
   {
      is_boxman = 1;
   }

   /* are the ids known? (these may have been set in coarsen)  - if not we need
      to set them */
   if (nalu_hypre_StructGridIDs(grid) == NULL)
   {
      ids = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_local_boxes, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_local_boxes; i++)
      {
         ids[i] = i;
      }
      nalu_hypre_StructGridIDs(grid) = ids;
   }
   else
   {
      ids = nalu_hypre_StructGridIDs(grid);
   }

   /******** calculate the periodicity information ****************/

   box = nalu_hypre_BoxCreate(ndim);
   for (d = 0; d < ndim; d++)
   {
      iperiodic = nalu_hypre_IndexD(periodic, d) ? 1 : 0;
      nalu_hypre_BoxIMinD(box, d) = -iperiodic;
      nalu_hypre_BoxIMaxD(box, d) =  iperiodic;
   }
   num_periods = nalu_hypre_BoxVolume(box);

   pshifts = nalu_hypre_CTAlloc(nalu_hypre_Index,  num_periods, NALU_HYPRE_MEMORY_HOST);
   pshift = pshifts[0];
   nalu_hypre_SetIndex(pshift, 0);
   if (num_periods > 1)
   {
      p = 1;
      nalu_hypre_BoxGetSize(box, loop_size);
      nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
      {
         pshift = pshifts[p];
         zypre_BoxLoopGetIndex(pshift);
         nalu_hypre_AddIndexes(pshift, nalu_hypre_BoxIMin(box), ndim, pshift);
         notcenter = 0;
         for (d = 0; d < ndim; d++)
         {
            nalu_hypre_IndexD(pshift, d) *= nalu_hypre_IndexD(periodic, d);
            if (nalu_hypre_IndexD(pshift, d))
            {
               notcenter = 1;
            }
         }
         if (notcenter)
         {
            p++;
         }
      }
      nalu_hypre_SerialBoxLoop0End();
   }
   nalu_hypre_BoxDestroy(box);

   nalu_hypre_StructGridNumPeriods(grid) = num_periods;
   nalu_hypre_StructGridPShifts(grid)    = pshifts;

   /********calculate local size and the ghost size **************/

   size = 0;
   ghostsize = 0;
   ghostbox = nalu_hypre_BoxCreate(ndim);

   nalu_hypre_ForBoxI(i, local_boxes)
   {
      box = nalu_hypre_BoxArrayBox(local_boxes, i);
      size +=  nalu_hypre_BoxVolume(box);

      nalu_hypre_CopyBox(box, ghostbox);
      nalu_hypre_BoxGrowByArray(ghostbox, numghost);
      ghostsize += nalu_hypre_BoxVolume(ghostbox);
   }

   nalu_hypre_StructGridLocalSize(grid) = size;
   nalu_hypre_StructGridGhlocalSize(grid) = ghostsize;
   nalu_hypre_BoxDestroy(ghostbox);

   /* if the box manager has been created then we don't need to do the
    * following (because it was done through the coarsening routine) */
   if (!is_boxman)
   {
      /*************** set the global size *****************/

      NALU_HYPRE_BigInt big_size = (NALU_HYPRE_BigInt)size;
      nalu_hypre_MPI_Allreduce(&big_size, &global_size, 1, NALU_HYPRE_MPI_BIG_INT,
                          nalu_hypre_MPI_SUM, comm);
      nalu_hypre_StructGridGlobalSize(grid) = global_size; /* TO DO: this NALU_HYPRE_Int
                                                       * could overflow! (used
                                                       * to calc flops) */

      /*************** set bounding box ***********/

      bounding_box = nalu_hypre_BoxCreate(ndim);

      if (num_local_boxes)
      {
         /* initialize min and max index*/
         box = nalu_hypre_BoxArrayBox(local_boxes, 0);
         for (d = 0; d < ndim; d++)
         {
            nalu_hypre_IndexD(min_index, d) =  nalu_hypre_BoxIMinD(box, d);
            nalu_hypre_IndexD(max_index, d) =  nalu_hypre_BoxIMaxD(box, d);
         }

         nalu_hypre_ForBoxI(i, local_boxes)
         {
            box = nalu_hypre_BoxArrayBox(local_boxes, i);


            /* find min and max box extents */
            for (d = 0; d < ndim; d++)
            {
               nalu_hypre_IndexD(min_index, d) = nalu_hypre_min( nalu_hypre_IndexD(min_index, d),
                                                       nalu_hypre_BoxIMinD(box, d));
               nalu_hypre_IndexD(max_index, d) = nalu_hypre_max( nalu_hypre_IndexD(max_index, d),
                                                       nalu_hypre_BoxIMaxD(box, d));
            }
         }
         /*set bounding box (this is still based on local info only) */
         nalu_hypre_BoxSetExtents(bounding_box, min_index, max_index);

      }
      else /* no boxes owned*/
      {
         /* initialize min and max */
         for (d = 0; d < ndim; d++)
         {
            nalu_hypre_BoxIMinD(bounding_box, d) =  nalu_hypre_pow2(30);
            nalu_hypre_BoxIMaxD(bounding_box, d) = -nalu_hypre_pow2(30);
         }
      }
      /* set the extra dimensions of the bounding box to zero */
      for (d = ndim; d < NALU_HYPRE_MAXDIM; d++)
      {
         nalu_hypre_BoxIMinD(bounding_box, d) = 0;
         nalu_hypre_BoxIMaxD(bounding_box, d) = 0;
      }

      /* communication needed for the bounding box */
      /* pack buffer */
      for (d = 0; d < ndim; d++)
      {
         sendbuf6[d] = nalu_hypre_BoxIMinD(bounding_box, d);
         sendbuf6[d + ndim] = -nalu_hypre_BoxIMaxD(bounding_box, d);
      }
      nalu_hypre_MPI_Allreduce(sendbuf6, recvbuf6, 2 * ndim, NALU_HYPRE_MPI_INT,
                          nalu_hypre_MPI_MIN, comm);
      /* unpack buffer */
      for (d = 0; d < ndim; d++)
      {
         nalu_hypre_BoxIMinD(bounding_box, d) = recvbuf6[d];
         nalu_hypre_BoxIMaxD(bounding_box, d) = -recvbuf6[d + ndim];
      }

      nalu_hypre_StructGridBoundingBox(grid) = bounding_box;

      /*************** create a box manager *****************/
      max_nentries =  num_local_boxes + 20;
      info_size = 0; /* we don't need an info object */
      nalu_hypre_BoxManCreate(max_nentries, info_size, ndim, bounding_box,
                         comm, &boxman);

      /******** populate the box manager with my local boxes and gather neighbor
                information  ******/

      grow_box = nalu_hypre_BoxCreate(ndim);
      result_box = nalu_hypre_BoxCreate(ndim);
      periodic_box = nalu_hypre_BoxCreate(ndim);

      /* now loop through each local box */
      nalu_hypre_ForBoxI(i, local_boxes)
      {
         box = nalu_hypre_BoxArrayBox(local_boxes, i);
         /* add entry for each local box (the id is the boxnum, and should be
            sequential */
         nalu_hypre_BoxManAddEntry( boxman, nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box),
                               myid, i, entry_info );

         /* now expand box by max_distance or larger and gather entries */
         nalu_hypre_CopyBox(box, grow_box);
         nalu_hypre_BoxGrowByIndex(grow_box, max_distance);
         nalu_hypre_BoxManGatherEntries(boxman, nalu_hypre_BoxIMin(grow_box),
                                   nalu_hypre_BoxIMax(grow_box));

         /* now repeat for any periodic boxes - by shifting the grow_box*/
         for (k = 1; k < num_periods; k++) /* k=0 is original box */
         {
            nalu_hypre_CopyBox(grow_box, periodic_box);
            pshift = pshifts[k];
            nalu_hypre_BoxShiftPos(periodic_box, pshift);

            /* see if the shifted box intersects the domain */
            nalu_hypre_IntersectBoxes(periodic_box, bounding_box, result_box);
            /* if so, call gather entries */
            if (nalu_hypre_BoxVolume(result_box) > 0)
            {
               nalu_hypre_BoxManGatherEntries(boxman, nalu_hypre_BoxIMin(periodic_box),
                                         nalu_hypre_BoxIMax(periodic_box));
            }
         }
      }/* end of for each local box */

      nalu_hypre_BoxDestroy(periodic_box);
      nalu_hypre_BoxDestroy(grow_box);
      nalu_hypre_BoxDestroy(result_box);

   } /* end of if (!is_boxman) */

   /* boxman was created, but need to get additional neighbor info */
   else if ( nalu_hypre_IndexEqual(max_distance, 0, ndim) )
   {
      /* pick a new max distance and set in grid*/
      nalu_hypre_SetIndex(nalu_hypre_StructGridMaxDistance(grid), 2);
      max_distance =  nalu_hypre_StructGridMaxDistance(grid);

      grow_box = nalu_hypre_BoxCreate(ndim);
      result_box = nalu_hypre_BoxCreate(ndim);
      periodic_box = nalu_hypre_BoxCreate(ndim);

      /* now loop through each local box */
      nalu_hypre_ForBoxI(i, local_boxes)
      {
         box = nalu_hypre_BoxArrayBox(local_boxes, i);

         /* now expand box by max_distance or larger and gather entries */
         nalu_hypre_CopyBox(box, grow_box);
         nalu_hypre_BoxGrowByIndex(grow_box, max_distance);
         nalu_hypre_BoxManGatherEntries(boxman, nalu_hypre_BoxIMin(grow_box),
                                   nalu_hypre_BoxIMax(grow_box));

         /* now repeat for any periodic boxes - by shifting the grow_box*/
         for (k = 1; k < num_periods; k++) /* k=0 is original box */
         {
            nalu_hypre_CopyBox(grow_box, periodic_box);
            pshift = pshifts[k];
            nalu_hypre_BoxShiftPos(periodic_box, pshift);

            /* see if the shifted box intersects the domain */
            nalu_hypre_IntersectBoxes(periodic_box, bounding_box, result_box);
            /* if so, call gather entries */
            if (nalu_hypre_BoxVolume(result_box) > 0)
            {
               nalu_hypre_BoxManGatherEntries(boxman, nalu_hypre_BoxIMin(periodic_box),
                                         nalu_hypre_BoxIMax(periodic_box));
            }
         }
      }/* end of for each local box */

      nalu_hypre_BoxDestroy(periodic_box);
      nalu_hypre_BoxDestroy(grow_box);
      nalu_hypre_BoxDestroy(result_box);
   }

   /***************Assemble the box manager *****************/

   nalu_hypre_BoxManAssemble(boxman);

   nalu_hypre_StructGridBoxMan(grid) = boxman;

   nalu_hypre_EndTiming(time_index);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_GatherAllBoxes
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_GatherAllBoxes(MPI_Comm         comm,
                     nalu_hypre_BoxArray  *boxes,
                     NALU_HYPRE_Int        ndim,
                     nalu_hypre_BoxArray **all_boxes_ptr,
                     NALU_HYPRE_Int      **all_procs_ptr,
                     NALU_HYPRE_Int       *first_local_ptr)
{
   nalu_hypre_BoxArray    *all_boxes;
   NALU_HYPRE_Int         *all_procs;
   NALU_HYPRE_Int          first_local;
   NALU_HYPRE_Int          all_boxes_size;

   nalu_hypre_Box         *box;
   nalu_hypre_Index        imin;
   nalu_hypre_Index        imax;

   NALU_HYPRE_Int          num_all_procs, my_rank;

   NALU_HYPRE_Int         *sendbuf;
   NALU_HYPRE_Int          sendcount;
   NALU_HYPRE_Int         *recvbuf;
   NALU_HYPRE_Int         *recvcounts;
   NALU_HYPRE_Int         *displs;
   NALU_HYPRE_Int          recvbuf_size;
   NALU_HYPRE_Int          item_size;

   NALU_HYPRE_Int          i, p, b, d;

   /*-----------------------------------------------------
    * Accumulate the box info
    *-----------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(comm, &num_all_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_rank);

   /* compute recvcounts and displs */
   item_size = 2 * ndim + 1;
   sendcount = item_size * nalu_hypre_BoxArraySize(boxes);
   recvcounts =  nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_all_procs, NALU_HYPRE_MEMORY_HOST);
   displs = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_all_procs, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_MPI_Allgather(&sendcount, 1, NALU_HYPRE_MPI_INT,
                       recvcounts, 1, NALU_HYPRE_MPI_INT, comm);
   displs[0] = 0;
   recvbuf_size = recvcounts[0];
   for (p = 1; p < num_all_procs; p++)
   {
      displs[p] = displs[p - 1] + recvcounts[p - 1];
      recvbuf_size += recvcounts[p];
   }

   /* allocate sendbuf and recvbuf */
   sendbuf = nalu_hypre_TAlloc(NALU_HYPRE_Int,  sendcount, NALU_HYPRE_MEMORY_HOST);
   recvbuf =  nalu_hypre_TAlloc(NALU_HYPRE_Int,  recvbuf_size, NALU_HYPRE_MEMORY_HOST);

   /* put local box extents and process number into sendbuf */
   i = 0;
   for (b = 0; b < nalu_hypre_BoxArraySize(boxes); b++)
   {
      sendbuf[i++] = my_rank;

      box = nalu_hypre_BoxArrayBox(boxes, b);
      for (d = 0; d < ndim; d++)
      {
         sendbuf[i++] = nalu_hypre_BoxIMinD(box, d);
         sendbuf[i++] = nalu_hypre_BoxIMaxD(box, d);
      }
   }

   /* get global grid info */
   nalu_hypre_MPI_Allgatherv(sendbuf, sendcount, NALU_HYPRE_MPI_INT,
                        recvbuf, recvcounts, displs, NALU_HYPRE_MPI_INT, comm);

   /* sort recvbuf by process rank? */

   /*-----------------------------------------------------
    * Create all_boxes, etc.
    *-----------------------------------------------------*/

   /* unpack recvbuf box info */
   all_boxes_size = recvbuf_size / item_size;
   all_boxes   = nalu_hypre_BoxArrayCreate(all_boxes_size, ndim);
   all_procs   = nalu_hypre_TAlloc(NALU_HYPRE_Int,  all_boxes_size, NALU_HYPRE_MEMORY_HOST);
   first_local = -1;
   i = 0;
   b = 0;
   box = nalu_hypre_BoxCreate(ndim);
   while (i < recvbuf_size)
   {
      all_procs[b] = recvbuf[i++];
      for (d = 0; d < ndim; d++)
      {
         nalu_hypre_IndexD(imin, d) = recvbuf[i++];
         nalu_hypre_IndexD(imax, d) = recvbuf[i++];
      }
      nalu_hypre_BoxSetExtents(box, imin, imax);
      nalu_hypre_CopyBox(box, nalu_hypre_BoxArrayBox(all_boxes, b));

      if ((first_local < 0) && (all_procs[b] == my_rank))
      {
         first_local = b;
      }

      b++;
   }
   nalu_hypre_BoxDestroy(box);

   /*-----------------------------------------------------
    * Return
    *-----------------------------------------------------*/

   nalu_hypre_TFree(sendbuf, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recvbuf, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recvcounts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(displs, NALU_HYPRE_MEMORY_HOST);

   *all_boxes_ptr   = all_boxes;
   *all_procs_ptr   = all_procs;
   *first_local_ptr = first_local;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ComputeBoxnums
 *
 * It is assumed that, for any process number in 'procs', all of that
 * processes local boxes appear in the 'boxes' array.
 *
 * It is assumed that the boxes in 'boxes' are ordered by associated
 * process number then by their local ordering on that process.
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ComputeBoxnums(nalu_hypre_BoxArray *boxes,
                     NALU_HYPRE_Int      *procs,
                     NALU_HYPRE_Int     **boxnums_ptr)
{

   NALU_HYPRE_Int         *boxnums;
   NALU_HYPRE_Int          num_boxes;
   NALU_HYPRE_Int          p, b, boxnum;

   /*-----------------------------------------------------
    *-----------------------------------------------------*/

   num_boxes = nalu_hypre_BoxArraySize(boxes);
   boxnums = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_boxes, NALU_HYPRE_MEMORY_HOST);

   p = -1;
   for (b = 0; b < num_boxes; b++)
   {
      /* start boxnum count at zero for each new process */
      if (procs[b] != p)
      {
         boxnum = 0;
         p = procs[b];
      }
      boxnums[b] = boxnum;
      boxnum++;
   }

   *boxnums_ptr = boxnums;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructGridPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridPrint( FILE             *file,
                       nalu_hypre_StructGrid *grid )
{

   nalu_hypre_BoxArray  *boxes;
   nalu_hypre_Box       *box;

   NALU_HYPRE_Int        i, d, ndim;

   ndim = nalu_hypre_StructGridNDim(grid);
   nalu_hypre_fprintf(file, "%d\n", ndim);

   boxes = nalu_hypre_StructGridBoxes(grid);
   nalu_hypre_fprintf(file, "%d\n", nalu_hypre_BoxArraySize(boxes));

   /* Print lines of the form: "%d:  (%d, %d, %d)  x  (%d, %d, %d)\n" */
   nalu_hypre_ForBoxI(i, boxes)
   {
      box = nalu_hypre_BoxArrayBox(boxes, i);
      nalu_hypre_fprintf(file, "%d:  (%d", i, nalu_hypre_BoxIMinD(box, 0));
      for (d = 1; d < ndim; d++)
      {
         nalu_hypre_fprintf(file, ", %d", nalu_hypre_BoxIMinD(box, d));
      }
      nalu_hypre_fprintf(file, ")  x  (%d", nalu_hypre_BoxIMaxD(box, 0));
      for (d = 1; d < ndim; d++)
      {
         nalu_hypre_fprintf(file, ", %d", nalu_hypre_BoxIMaxD(box, d));
      }
      nalu_hypre_fprintf(file, ")\n");
   }
   /* Print line of the form: "Periodic: %d %d %d\n" */
   nalu_hypre_fprintf(file, "\nPeriodic:");
   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_fprintf(file, " %d", nalu_hypre_StructGridPeriodic(grid)[d]);
   }
   nalu_hypre_fprintf(file, "\n");

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructGridRead
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridRead( MPI_Comm           comm,
                      FILE              *file,
                      nalu_hypre_StructGrid **grid_ptr )
{

   nalu_hypre_StructGrid *grid;

   nalu_hypre_Index       ilower;
   nalu_hypre_Index       iupper;
   nalu_hypre_IndexRef    periodic;

   NALU_HYPRE_Int         ndim;
   NALU_HYPRE_Int         num_boxes;

   NALU_HYPRE_Int         i, d, idummy;

   nalu_hypre_fscanf(file, "%d\n", &ndim);
   nalu_hypre_StructGridCreate(comm, ndim, &grid);

   nalu_hypre_fscanf(file, "%d\n", &num_boxes);

   /* Read lines of the form: "%d:  (%d, %d, %d)  x  (%d, %d, %d)\n" */
   for (i = 0; i < num_boxes; i++)
   {
      nalu_hypre_fscanf(file, "%d:  (%d", &idummy, &nalu_hypre_IndexD(ilower, 0));
      for (d = 1; d < ndim; d++)
      {
         nalu_hypre_fscanf(file, ", %d", &nalu_hypre_IndexD(ilower, d));
      }
      nalu_hypre_fscanf(file, ")  x  (%d", &nalu_hypre_IndexD(iupper, 0));
      for (d = 1; d < ndim; d++)
      {
         nalu_hypre_fscanf(file, ", %d", &nalu_hypre_IndexD(iupper, d));
      }
      nalu_hypre_fscanf(file, ")\n");

      nalu_hypre_StructGridSetExtents(grid, ilower, iupper);
   }

   periodic = nalu_hypre_StructGridPeriodic(grid);

   /* Read line of the form: "Periodic: %d %d %d\n" */
   nalu_hypre_fscanf(file, "Periodic:");
   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_fscanf(file, " %d", &nalu_hypre_IndexD(periodic, d));
   }
   nalu_hypre_fscanf(file, "\n");

   nalu_hypre_StructGridAssemble(grid);

   *grid_ptr = grid;

   return nalu_hypre_error_flag;
}

/*------------------------------------------------------------------------------
 * GEC0902  nalu_hypre_StructGridSetNumGhost
 *
 * the purpose is to set num ghost in the structure grid. It is identical
 * to the function that is used in the structure vector entity.
 *-----------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridSetNumGhost( nalu_hypre_StructGrid *grid, NALU_HYPRE_Int  *num_ghost )
{
   NALU_HYPRE_Int  i, ndim = nalu_hypre_StructGridNDim(grid);

   for (i = 0; i < 2 * ndim; i++)
   {
      nalu_hypre_StructGridNumGhost(grid)[i] = num_ghost[i];
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructGridGetMaxBoxSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructGridGetMaxBoxSize( nalu_hypre_StructGrid *grid )
{
   nalu_hypre_BoxArray   *boxes;
   nalu_hypre_Box        *box;
   NALU_HYPRE_Int         i, max_box_size = 0;

   boxes = nalu_hypre_StructGridBoxes(grid);
   nalu_hypre_ForBoxI(i, boxes)
   {
      box = nalu_hypre_BoxArrayBox(nalu_hypre_StructGridBoxes(grid), i);
      max_box_size = nalu_hypre_max(max_box_size, nalu_hypre_BoxVolume(box));
   }

   return max_box_size;
}

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
NALU_HYPRE_Int
nalu_hypre_StructGridSetDataLocation( NALU_HYPRE_StructGrid grid, NALU_HYPRE_MemoryLocation data_location )
{
   nalu_hypre_StructGridDataLocation(grid) = data_location;

   return nalu_hypre_error_flag;
}

#endif
