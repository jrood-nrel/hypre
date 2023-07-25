/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructSendInfo: Given a fgrid, coarsen each fbox and find the
 * coarsened boxes that must be sent, the procs that they must be sent to,
 * and the remote boxnums of these sendboxes.
 *--------------------------------------------------------------------------*/

nalu_hypre_SStructSendInfoData *
nalu_hypre_SStructSendInfo( nalu_hypre_StructGrid      *fgrid,
                       nalu_hypre_BoxManager      *cboxman,
                       nalu_hypre_Index            rfactor )
{
   nalu_hypre_SStructSendInfoData *sendinfo_data;

   MPI_Comm                   comm = nalu_hypre_StructGridComm(fgrid);
   NALU_HYPRE_Int                  ndim = nalu_hypre_StructGridNDim(fgrid);

   nalu_hypre_BoxArray            *grid_boxes;
   nalu_hypre_Box                 *grid_box, cbox;
   nalu_hypre_Box                 *intersect_box, boxman_entry_box;

   nalu_hypre_BoxManEntry        **boxman_entries;
   NALU_HYPRE_Int                  nboxman_entries;

   nalu_hypre_BoxArrayArray       *send_boxes;
   NALU_HYPRE_Int                **send_processes;
   NALU_HYPRE_Int                **send_remote_boxnums;

   nalu_hypre_Index                ilower, iupper, index;

   NALU_HYPRE_Int                  myproc, proc;

   NALU_HYPRE_Int                  cnt;
   NALU_HYPRE_Int                  i, j;

   nalu_hypre_BoxInit(&cbox, ndim);
   nalu_hypre_BoxInit(&boxman_entry_box, ndim);

   nalu_hypre_ClearIndex(index);
   nalu_hypre_MPI_Comm_rank(comm, &myproc);

   sendinfo_data = nalu_hypre_CTAlloc(nalu_hypre_SStructSendInfoData,  1, NALU_HYPRE_MEMORY_HOST);

   /*------------------------------------------------------------------------
    * Create the structured sendbox patterns.
    *
    *   send_boxes are obtained by intersecting this proc's fgrid boxes
    *   with cgrid's box_man. Intersecting BoxManEntries not on this proc
    *   will give boxes that we will need to send data to- i.e., we scan
    *   through the boxes of grid and find the processors that own a chunk
    *   of it.
    *------------------------------------------------------------------------*/
   intersect_box = nalu_hypre_BoxCreate(ndim);
   grid_boxes   = nalu_hypre_StructGridBoxes(fgrid);

   send_boxes = nalu_hypre_BoxArrayArrayCreate(nalu_hypre_BoxArraySize(grid_boxes), ndim);
   send_processes = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(grid_boxes), NALU_HYPRE_MEMORY_HOST);
   send_remote_boxnums = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(grid_boxes),
                                       NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ForBoxI(i, grid_boxes)
   {
      grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);

      /*---------------------------------------------------------------------
       * Find the boxarray that must be sent. BoxManIntersect returns
       * the full extents of the boxes that intersect with the given box.
       * We further need to intersect each box in the list with the given
       * box to determine the actual box that needs to be sent.
       *---------------------------------------------------------------------*/
      nalu_hypre_SStructIndexScaleF_C(nalu_hypre_BoxIMin(grid_box), index,
                                 rfactor, nalu_hypre_BoxIMin(&cbox));
      nalu_hypre_SStructIndexScaleF_C(nalu_hypre_BoxIMax(grid_box), index,
                                 rfactor, nalu_hypre_BoxIMax(&cbox));

      nalu_hypre_BoxManIntersect(cboxman, nalu_hypre_BoxIMin(&cbox), nalu_hypre_BoxIMax(&cbox),
                            &boxman_entries, &nboxman_entries);

      cnt = 0;
      for (j = 0; j < nboxman_entries; j++)
      {
         nalu_hypre_SStructBoxManEntryGetProcess(boxman_entries[j], &proc);
         if (proc != myproc)
         {
            cnt++;
         }
      }
      send_processes[i]     = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt, NALU_HYPRE_MEMORY_HOST);
      send_remote_boxnums[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt, NALU_HYPRE_MEMORY_HOST);

      cnt = 0;
      for (j = 0; j < nboxman_entries; j++)
      {
         nalu_hypre_SStructBoxManEntryGetProcess(boxman_entries[j], &proc);

         /* determine the chunk of the boxman_entries[j] box that is needed */
         nalu_hypre_BoxManEntryGetExtents(boxman_entries[j], ilower, iupper);
         nalu_hypre_BoxSetExtents(&boxman_entry_box, ilower, iupper);
         nalu_hypre_IntersectBoxes(&boxman_entry_box, &cbox, &boxman_entry_box);

         if (proc != myproc)
         {
            send_processes[i][cnt]     = proc;
            nalu_hypre_SStructBoxManEntryGetBoxnum(boxman_entries[j],
                                              &send_remote_boxnums[i][cnt]);
            nalu_hypre_AppendBox(&boxman_entry_box,
                            nalu_hypre_BoxArrayArrayBoxArray(send_boxes, i));
            cnt++;
         }
      }
      nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);
   }  /* nalu_hypre_ForBoxI(i, grid_boxes) */

   nalu_hypre_BoxDestroy(intersect_box);

   (sendinfo_data -> size)               = nalu_hypre_BoxArraySize(grid_boxes);
   (sendinfo_data -> send_boxes)         = send_boxes;
   (sendinfo_data -> send_procs)         = send_processes;
   (sendinfo_data -> send_remote_boxnums) = send_remote_boxnums;

   return sendinfo_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructSendInfoDataDestroy
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_SStructSendInfoDataDestroy(nalu_hypre_SStructSendInfoData *sendinfo_data)
{
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int i;

   if (sendinfo_data)
   {
      if (sendinfo_data -> send_boxes)
      {
         nalu_hypre_BoxArrayArrayDestroy( (sendinfo_data -> send_boxes) );
      }

      for (i = 0; i < (sendinfo_data -> size); i++)
      {
         if (sendinfo_data -> send_procs[i])
         {
            nalu_hypre_TFree(sendinfo_data -> send_procs[i], NALU_HYPRE_MEMORY_HOST);
         }

         if (sendinfo_data -> send_remote_boxnums[i])
         {
            nalu_hypre_TFree(sendinfo_data -> send_remote_boxnums[i], NALU_HYPRE_MEMORY_HOST);
         }
      }
      nalu_hypre_TFree(sendinfo_data -> send_procs, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(sendinfo_data -> send_remote_boxnums, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(sendinfo_data, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

