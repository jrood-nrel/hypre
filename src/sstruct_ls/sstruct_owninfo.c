/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

NALU_HYPRE_Int
nalu_hypre_SStructIndexScaleF_C( nalu_hypre_Index findex,
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

   return 0;
}


NALU_HYPRE_Int
nalu_hypre_SStructIndexScaleC_F( nalu_hypre_Index cindex,
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

   return 0;
}
/*--------------------------------------------------------------------------
 * nalu_hypre_SStructOwnInfo: Given a fgrid, coarsen each fbox and find the
 * coarsened boxes that belong on my current processor. These are my own_boxes.
 *--------------------------------------------------------------------------*/

nalu_hypre_SStructOwnInfoData *
nalu_hypre_SStructOwnInfo( nalu_hypre_StructGrid  *fgrid,
                      nalu_hypre_StructGrid  *cgrid,
                      nalu_hypre_BoxManager  *cboxman,
                      nalu_hypre_BoxManager  *fboxman,
                      nalu_hypre_Index        rfactor )
{
   nalu_hypre_SStructOwnInfoData *owninfo_data;

   MPI_Comm                  comm = nalu_hypre_SStructVectorComm(fgrid);
   NALU_HYPRE_Int                 ndim = nalu_hypre_StructGridNDim(fgrid);

   nalu_hypre_BoxArray           *grid_boxes;
   nalu_hypre_BoxArray           *intersect_boxes;
   nalu_hypre_BoxArray           *tmp_boxarray;

   nalu_hypre_Box                *grid_box, scaled_box;
   nalu_hypre_Box                 boxman_entry_box;

   nalu_hypre_BoxManEntry       **boxman_entries;
   NALU_HYPRE_Int                 nboxman_entries;

   nalu_hypre_BoxArrayArray      *own_boxes;
   NALU_HYPRE_Int               **own_cboxnums;

   nalu_hypre_BoxArrayArray      *own_composite_cboxes;

   nalu_hypre_Index               ilower, iupper, index;

   NALU_HYPRE_Int                 myproc, proc;

   NALU_HYPRE_Int                 cnt;
   NALU_HYPRE_Int                 i, j, k, mod;

   nalu_hypre_BoxInit(&scaled_box, ndim);
   nalu_hypre_BoxInit(&boxman_entry_box, ndim);

   nalu_hypre_ClearIndex(index);
   nalu_hypre_MPI_Comm_rank(comm, &myproc);

   owninfo_data = nalu_hypre_CTAlloc(nalu_hypre_SStructOwnInfoData,  1, NALU_HYPRE_MEMORY_HOST);

   /*------------------------------------------------------------------------
    * Create the structured ownbox patterns.
    *
    *   own_boxes are obtained by intersecting this proc's fgrid boxes
    *   with cgrid's box_man. Intersecting BoxManEntries on this proc
    *   will give the own_boxes.
    *------------------------------------------------------------------------*/
   grid_boxes    = nalu_hypre_StructGridBoxes(fgrid);

   own_boxes   = nalu_hypre_BoxArrayArrayCreate(nalu_hypre_BoxArraySize(grid_boxes), ndim);
   own_cboxnums = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(grid_boxes), NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ForBoxI(i, grid_boxes)
   {
      grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);

      /*---------------------------------------------------------------------
       * Find the boxarray that is owned. BoxManIntersect returns
       * the full extents of the boxes that intersect with the given box.
       * We further need to intersect each box in the list with the given
       * box to determine the actual box that is owned.
       *---------------------------------------------------------------------*/
      nalu_hypre_SStructIndexScaleF_C(nalu_hypre_BoxIMin(grid_box), index,
                                 rfactor, nalu_hypre_BoxIMin(&scaled_box));
      nalu_hypre_SStructIndexScaleF_C(nalu_hypre_BoxIMax(grid_box), index,
                                 rfactor, nalu_hypre_BoxIMax(&scaled_box));

      nalu_hypre_BoxManIntersect(cboxman, nalu_hypre_BoxIMin(&scaled_box),
                            nalu_hypre_BoxIMax(&scaled_box), &boxman_entries,
                            &nboxman_entries);

      cnt = 0;
      for (j = 0; j < nboxman_entries; j++)
      {
         nalu_hypre_SStructBoxManEntryGetProcess(boxman_entries[j], &proc);
         if (proc == myproc)
         {
            cnt++;
         }
      }
      own_cboxnums[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt, NALU_HYPRE_MEMORY_HOST);

      cnt = 0;
      for (j = 0; j < nboxman_entries; j++)
      {
         nalu_hypre_SStructBoxManEntryGetProcess(boxman_entries[j], &proc);

         /* determine the chunk of the boxman_entries[j] box that is needed */
         nalu_hypre_BoxManEntryGetExtents(boxman_entries[j], ilower, iupper);
         nalu_hypre_BoxSetExtents(&boxman_entry_box, ilower, iupper);
         nalu_hypre_IntersectBoxes(&boxman_entry_box, &scaled_box, &boxman_entry_box);

         if (proc == myproc)
         {
            nalu_hypre_SStructBoxManEntryGetBoxnum(boxman_entries[j], &own_cboxnums[i][cnt]);
            nalu_hypre_AppendBox(&boxman_entry_box,
                            nalu_hypre_BoxArrayArrayBoxArray(own_boxes, i));
            cnt++;
         }
      }
      nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);
   }  /* nalu_hypre_ForBoxI(i, grid_boxes) */

   (owninfo_data -> size)     = nalu_hypre_BoxArraySize(grid_boxes);
   (owninfo_data -> own_boxes) = own_boxes;
   (owninfo_data -> own_cboxnums) = own_cboxnums;

   /*------------------------------------------------------------------------
    *   own_composite_cboxes are obtained by intersecting this proc's cgrid
    *   boxes with fgrid's box_man. For each cbox, subtracting all the
    *   intersecting boxes from all processors will give the
    *   own_composite_cboxes.
    *------------------------------------------------------------------------*/
   grid_boxes = nalu_hypre_StructGridBoxes(cgrid);
   own_composite_cboxes = nalu_hypre_BoxArrayArrayCreate(nalu_hypre_BoxArraySize(grid_boxes), ndim);
   (owninfo_data -> own_composite_size) = nalu_hypre_BoxArraySize(grid_boxes);

   tmp_boxarray = nalu_hypre_BoxArrayCreate(0, ndim);
   nalu_hypre_ForBoxI(i, grid_boxes)
   {
      grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);
      nalu_hypre_AppendBox(grid_box,
                      nalu_hypre_BoxArrayArrayBoxArray(own_composite_cboxes, i));

      nalu_hypre_ClearIndex(index);
      nalu_hypre_SStructIndexScaleC_F(nalu_hypre_BoxIMin(grid_box), index,
                                 rfactor, nalu_hypre_BoxIMin(&scaled_box));
      nalu_hypre_SetIndex3(index, rfactor[0] - 1, rfactor[1] - 1, rfactor[2] - 1);
      nalu_hypre_SStructIndexScaleC_F(nalu_hypre_BoxIMax(grid_box), index,
                                 rfactor, nalu_hypre_BoxIMax(&scaled_box));

      nalu_hypre_BoxManIntersect(fboxman, nalu_hypre_BoxIMin(&scaled_box),
                            nalu_hypre_BoxIMax(&scaled_box), &boxman_entries,
                            &nboxman_entries);

      nalu_hypre_ClearIndex(index);
      intersect_boxes = nalu_hypre_BoxArrayCreate(0, ndim);
      for (j = 0; j < nboxman_entries; j++)
      {
         nalu_hypre_BoxManEntryGetExtents(boxman_entries[j], ilower, iupper);
         nalu_hypre_BoxSetExtents(&boxman_entry_box, ilower, iupper);
         nalu_hypre_IntersectBoxes(&boxman_entry_box, &scaled_box, &boxman_entry_box);

         /* contract the intersection box so that only the cnodes in the
            intersection box are included. */
         for (k = 0; k < ndim; k++)
         {
            mod = nalu_hypre_BoxIMin(&boxman_entry_box)[k] % rfactor[k];
            if (mod)
            {
               nalu_hypre_BoxIMin(&boxman_entry_box)[k] += rfactor[k] - mod;
            }
         }

         nalu_hypre_SStructIndexScaleF_C(nalu_hypre_BoxIMin(&boxman_entry_box), index,
                                    rfactor, nalu_hypre_BoxIMin(&boxman_entry_box));
         nalu_hypre_SStructIndexScaleF_C(nalu_hypre_BoxIMax(&boxman_entry_box), index,
                                    rfactor, nalu_hypre_BoxIMax(&boxman_entry_box));
         nalu_hypre_AppendBox(&boxman_entry_box, intersect_boxes);
      }

      nalu_hypre_SubtractBoxArrays(nalu_hypre_BoxArrayArrayBoxArray(own_composite_cboxes, i),
                              intersect_boxes, tmp_boxarray);
      nalu_hypre_MinUnionBoxes(nalu_hypre_BoxArrayArrayBoxArray(own_composite_cboxes, i));

      nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_BoxArrayDestroy(intersect_boxes);
   }
   nalu_hypre_BoxArrayDestroy(tmp_boxarray);

   (owninfo_data -> own_composite_cboxes) = own_composite_cboxes;

   return owninfo_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructOwnInfoDataDestroy
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_SStructOwnInfoDataDestroy(nalu_hypre_SStructOwnInfoData *owninfo_data)
{
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int i;

   if (owninfo_data)
   {
      if (owninfo_data -> own_boxes)
      {
         nalu_hypre_BoxArrayArrayDestroy( (owninfo_data -> own_boxes) );
      }

      for (i = 0; i < (owninfo_data -> size); i++)
      {
         if (owninfo_data -> own_cboxnums[i])
         {
            nalu_hypre_TFree(owninfo_data -> own_cboxnums[i], NALU_HYPRE_MEMORY_HOST);
         }
      }
      nalu_hypre_TFree(owninfo_data -> own_cboxnums, NALU_HYPRE_MEMORY_HOST);

      if (owninfo_data -> own_composite_cboxes)
      {
         nalu_hypre_BoxArrayArrayDestroy( (owninfo_data -> own_composite_cboxes) );
      }
   }

   nalu_hypre_TFree(owninfo_data, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

