/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#define TIME_DEBUG 0

#if TIME_DEBUG
static NALU_HYPRE_Int s_coarsen_num = 0;
#endif


#include "_nalu_hypre_struct_ls.h"

#define DEBUG 0

#if DEBUG
char       filename[255];
FILE      *file;
static NALU_HYPRE_Int debug_count = 0;
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMapFineToCoarse
 *
 * NOTE: findex and cindex are indexes on the fine and coarse index space, and
 * do not stand for "F-pt index" and "C-pt index".
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMapFineToCoarse( nalu_hypre_Index findex,
                             nalu_hypre_Index index,
                             nalu_hypre_Index stride,
                             nalu_hypre_Index cindex )
{
   nalu_hypre_IndexX(cindex) =
      (nalu_hypre_IndexX(findex) - nalu_hypre_IndexX(index)) / nalu_hypre_IndexX(stride);
   nalu_hypre_IndexY(cindex) =
      (nalu_hypre_IndexY(findex) - nalu_hypre_IndexY(index)) / nalu_hypre_IndexY(stride);
   nalu_hypre_IndexZ(cindex) =
      (nalu_hypre_IndexZ(findex) - nalu_hypre_IndexZ(index)) / nalu_hypre_IndexZ(stride);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMapCoarseToFine
 *
 * NOTE: findex and cindex are indexes on the fine and coarse index space, and
 * do not stand for "F-pt index" and "C-pt index".
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMapCoarseToFine( nalu_hypre_Index cindex,
                             nalu_hypre_Index index,
                             nalu_hypre_Index stride,
                             nalu_hypre_Index findex )
{
   nalu_hypre_IndexX(findex) =
      nalu_hypre_IndexX(cindex) * nalu_hypre_IndexX(stride) + nalu_hypre_IndexX(index);
   nalu_hypre_IndexY(findex) =
      nalu_hypre_IndexY(cindex) * nalu_hypre_IndexY(stride) + nalu_hypre_IndexY(index);
   nalu_hypre_IndexZ(findex) =
      nalu_hypre_IndexZ(cindex) * nalu_hypre_IndexZ(stride) + nalu_hypre_IndexZ(index);

   return nalu_hypre_error_flag;
}

#define nalu_hypre_StructCoarsenBox(box, index, stride)                      \
   nalu_hypre_ProjectBox(box, index, stride);                                \
   nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(box), index, stride,       \
                               nalu_hypre_BoxIMin(box));                     \
   nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(box), index, stride,       \
                               nalu_hypre_BoxIMax(box))

/*--------------------------------------------------------------------------
 * New version of nalu_hypre_StructCoarsen that uses the BoxManager (AHB 12/06)
 *
 * This routine coarsens the grid, 'fgrid', by the coarsening factor, 'stride',
 * using the index mapping in 'nalu_hypre_StructMapFineToCoarse'.
 *
 *  1.  A coarse grid is created with boxes that result from coarsening the fine
 *  grid boxes, bounding box, and periodicity information.
 *
 *  2. If "sufficient" neighbor information exists in the fine grid to be
 *  transferred to the coarse grid, then the coarse grid box manager can be
 *  created by simply coarsening all of the entries in the fine grid manager.
 *  ("Sufficient" is determined by checking max_distance in the fine grid.)
 *
 *  3.  Otherwise, neighbor information will be collected during the
 *  StructGridAssemble according to the choosen value of max_distance for the
 *  coarse grid.
 *
 *   4. We do not need a separate version for the assumed partition case
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructCoarsen( nalu_hypre_StructGrid  *fgrid,
                     nalu_hypre_Index        index,
                     nalu_hypre_Index        stride,
                     NALU_HYPRE_Int          prune,
                     nalu_hypre_StructGrid **cgrid_ptr )
{
   nalu_hypre_StructGrid *cgrid;

   MPI_Comm          comm;
   NALU_HYPRE_Int         ndim;

   nalu_hypre_BoxArray   *my_boxes;

   nalu_hypre_Index       periodic;
   nalu_hypre_Index       ilower, iupper;

   nalu_hypre_Box        *box;
   nalu_hypre_Box        *new_box;
   nalu_hypre_Box        *bounding_box;

   NALU_HYPRE_Int         i, j, myid, count;
   NALU_HYPRE_Int         info_size, max_nentries;
   NALU_HYPRE_Int         num_entries;
   NALU_HYPRE_Int        *fids, *cids;
   nalu_hypre_Index       new_dist;
   nalu_hypre_IndexRef    max_distance;
   NALU_HYPRE_Int         proc, id;
   NALU_HYPRE_Int         coarsen_factor, known;
   NALU_HYPRE_Int         num, last_proc;
#if 0
   nalu_hypre_StructAssumedPart *fap = NULL, *cap = NULL;
#endif
   nalu_hypre_BoxManager   *fboxman, *cboxman;

   nalu_hypre_BoxManEntry *entries;
   nalu_hypre_BoxManEntry  *entry;

   void               *entry_info = NULL;

#if TIME_DEBUG
   NALU_HYPRE_Int tindex;
   char new_title[80];
   nalu_hypre_sprintf(new_title, "Coarsen.%d", s_coarsen_num);
   tindex = nalu_hypre_InitializeTiming(new_title);
   s_coarsen_num++;

   nalu_hypre_BeginTiming(tindex);
#endif

   nalu_hypre_SetIndex(ilower, 0);
   nalu_hypre_SetIndex(iupper, 0);

   /* get relevant information from the fine grid */
   fids = nalu_hypre_StructGridIDs(fgrid);
   fboxman = nalu_hypre_StructGridBoxMan(fgrid);
   comm  = nalu_hypre_StructGridComm(fgrid);
   ndim  = nalu_hypre_StructGridNDim(fgrid);
   max_distance = nalu_hypre_StructGridMaxDistance(fgrid);

   /* initial */
   nalu_hypre_MPI_Comm_rank(comm, &myid );

   /* create new coarse grid */
   nalu_hypre_StructGridCreate(comm, ndim, &cgrid);

   /* coarsen my boxes and create the coarse grid ids (same as fgrid) */
   my_boxes = nalu_hypre_BoxArrayDuplicate(nalu_hypre_StructGridBoxes(fgrid));
   cids = nalu_hypre_TAlloc(NALU_HYPRE_Int,   nalu_hypre_BoxArraySize(my_boxes), NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nalu_hypre_BoxArraySize(my_boxes); i++)
   {
      box = nalu_hypre_BoxArrayBox(my_boxes, i);
      nalu_hypre_StructCoarsenBox(box, index, stride);
      cids[i] = fids[i];
   }

   /* prune? */
   /* zero volume boxes are needed when forming P and P^T */
   if (prune)
   {
      count = 0;
      nalu_hypre_ForBoxI(i, my_boxes)
      {
         box = nalu_hypre_BoxArrayBox(my_boxes, i);
         if (nalu_hypre_BoxVolume(box))
         {
            nalu_hypre_CopyBox(box, nalu_hypre_BoxArrayBox(my_boxes, count));
            cids[count] = cids[i];
            count++;
         }
      }
      nalu_hypre_BoxArraySetSize(my_boxes, count);
   }

   /* set coarse grid boxes */
   nalu_hypre_StructGridSetBoxes(cgrid, my_boxes);

   /* set coarse grid ids */
   nalu_hypre_StructGridSetIDs(cgrid, cids);

   /* adjust periodicity and set for the coarse grid */
   nalu_hypre_CopyIndex(nalu_hypre_StructGridPeriodic(fgrid), periodic);
   for (i = 0; i < ndim; i++)
   {
      nalu_hypre_IndexD(periodic, i) /= nalu_hypre_IndexD(stride, i);
   }
   nalu_hypre_StructGridSetPeriodic(cgrid, periodic);

   /* Check the max_distance value of the fine grid to determine whether we will
      need to re-gather information in the assemble.  If we need to re-gather,
      then the max_distance will be set to (0,0,0).  Either way, we will create
      and populate the box manager with the information from the fine grid.

      Note: if all global info is already known for a grid, the we do not need
      to re-gather regardless of the max_distance values. */

   for (i = 0; i < ndim; i++)
   {
      coarsen_factor = nalu_hypre_IndexD(stride, i);
      nalu_hypre_IndexD(new_dist, i) = nalu_hypre_IndexD(max_distance, i) / coarsen_factor;
   }
   for (i = ndim; i < 3; i++)
   {
      nalu_hypre_IndexD(new_dist, i) = 2;
   }

   nalu_hypre_BoxManGetAllGlobalKnown (fboxman, &known );


   /* large enough - don't need to re-gather */
   if ( (nalu_hypre_IndexMin(new_dist, ndim) > 1) || known )
   {
      /* update new max distance value */
      if (!known) /* only need to change if global info is not known */
      {
         nalu_hypre_StructGridSetMaxDistance(cgrid, new_dist);
      }
   }
   else  /* not large enough - set max_distance to 0 - neighbor info will be
            collected during the assemble */
   {
      nalu_hypre_SetIndex(new_dist, 0);
      nalu_hypre_StructGridSetMaxDistance(cgrid, new_dist);
   }

   /* update the new bounding box */
   bounding_box = nalu_hypre_BoxDuplicate(nalu_hypre_StructGridBoundingBox(fgrid));
   nalu_hypre_StructCoarsenBox(bounding_box, index, stride);

   nalu_hypre_StructGridSetBoundingBox(cgrid, bounding_box);

   /* create a box manager for the coarse grid */
   info_size = nalu_hypre_BoxManEntryInfoSize(fboxman);
   max_nentries =  nalu_hypre_BoxManMaxNEntries(fboxman);
   nalu_hypre_BoxManCreate(max_nentries, info_size, ndim, bounding_box,
                      comm, &cboxman);

   nalu_hypre_BoxDestroy(bounding_box);

   /* update all global known */
   nalu_hypre_BoxManSetAllGlobalKnown(cboxman, known );

   /* now get the entries from the fgrid box manager, coarsen, and add to the
      coarse grid box manager (note: my boxes have already been coarsened) */

   nalu_hypre_BoxManGetAllEntries( fboxman, &num_entries, &entries);

   new_box = nalu_hypre_BoxCreate(ndim);
   num = 0;
   last_proc = -1;

   /* entries are sorted by (proc, id) pairs - may not have entries for all
      processors, but for each processor represented, we do have all of its
      boxes.  We will keep them sorted in the new box manager - to avoid
      re-sorting */
   for (i = 0; i < num_entries; i++)
   {
      entry = &entries[i];
      proc = nalu_hypre_BoxManEntryProc(entry);

      if  (proc != myid) /* not my boxes */
      {
         nalu_hypre_BoxManEntryGetExtents(entry, ilower, iupper);
         nalu_hypre_BoxSetExtents(new_box, ilower, iupper);
         nalu_hypre_StructCoarsenBox(new_box, index, stride);
         id =  nalu_hypre_BoxManEntryId(entry);
         /* if there is pruning we need to adjust the ids if any boxes drop out
            (we want these ids sequential - no gaps) - and zero boxes are not
            kept in the box manager */
         if (prune)
         {
            if (proc != last_proc)
            {
               num = 0;
               last_proc = proc;
            }
            if (nalu_hypre_BoxVolume(new_box))
            {

               nalu_hypre_BoxManAddEntry( cboxman, nalu_hypre_BoxIMin(new_box),
                                     nalu_hypre_BoxIMax(new_box), proc, num,
                                     entry_info);
               num++;
            }
         }
         else /* no pruning - just use id (note that size zero boxes will not be
                 saved in the box manager, so we will have gaps in the box
                 numbers) */
         {
            nalu_hypre_BoxManAddEntry( cboxman, nalu_hypre_BoxIMin(new_box),
                                  nalu_hypre_BoxIMax(new_box), proc, id,
                                  entry_info);
         }
      }
      else /* my boxes */
         /* add my coarse grid boxes to the coarse grid box manager (have
            already been pruned if necessary) - re-number the entry ids to be
            sequential (this is the box number, really) */
      {
         if (proc != last_proc) /* just do this once (the first myid) */
         {
            nalu_hypre_ForBoxI(j, my_boxes)
            {
               box = nalu_hypre_BoxArrayBox(my_boxes, j);
               nalu_hypre_BoxManAddEntry( cboxman, nalu_hypre_BoxIMin(box),
                                     nalu_hypre_BoxIMax(box), myid, j,
                                     entry_info );
            }
            last_proc = proc;
         }
      }
   } /* loop through entries */

   /* these entries are sorted */
   nalu_hypre_BoxManSetIsEntriesSort(cboxman, 1 );

   nalu_hypre_BoxDestroy(new_box);

#if 0
   /* if there is an assumed partition in the fg, then coarsen those boxes as
      well and add to cg */
   nalu_hypre_BoxManGetAssumedPartition ( fboxman, &fap);

   if (fap)
   {
      /* coarsen fap to get cap */

      /* set cap */
      nalu_hypre_BoxManSetAssumedPartition (cboxman, cap);
   }
#endif

   /* assign new box manager */
   nalu_hypre_StructGridSetBoxManager(cgrid, cboxman);

   /* finally... assemble the new coarse grid */
   nalu_hypre_StructGridAssemble(cgrid);

   /* return the coarse grid */
   *cgrid_ptr = cgrid;

#if TIME_DEBUG
   nalu_hypre_EndTiming(tindex);
#endif

   return nalu_hypre_error_flag;
}

#undef nalu_hypre_StructCoarsenBox
