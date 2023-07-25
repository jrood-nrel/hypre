/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructRecvInfo: For each processor, for each cbox of its cgrid,
 * refine it and find out which processors owe this cbox. Coarsen these
 * fine recv boxes and store them.
 *--------------------------------------------------------------------------*/

nalu_hypre_SStructRecvInfoData *
nalu_hypre_SStructRecvInfo( nalu_hypre_StructGrid      *cgrid,
                       nalu_hypre_BoxManager      *fboxman,
                       nalu_hypre_Index            rfactor )
{
   nalu_hypre_SStructRecvInfoData *recvinfo_data;

   MPI_Comm                   comm = nalu_hypre_StructGridComm(cgrid);
   NALU_HYPRE_Int                  ndim = nalu_hypre_StructGridNDim(cgrid);

   nalu_hypre_BoxArray            *grid_boxes;
   nalu_hypre_Box                 *grid_box, fbox;
   nalu_hypre_Box                 *intersect_box, boxman_entry_box;

   nalu_hypre_BoxManEntry        **boxman_entries;
   NALU_HYPRE_Int                  nboxman_entries;

   nalu_hypre_BoxArrayArray       *recv_boxes;
   NALU_HYPRE_Int                **recv_processes;

   nalu_hypre_Index                ilower, iupper, index1, index2;

   NALU_HYPRE_Int                  myproc, proc;

   NALU_HYPRE_Int                  cnt;
   NALU_HYPRE_Int                  i, j;

   nalu_hypre_BoxInit(&fbox, ndim);
   nalu_hypre_BoxInit(&boxman_entry_box, ndim);

   nalu_hypre_ClearIndex(index1);
   nalu_hypre_SetIndex3(index2, rfactor[0] - 1, rfactor[1] - 1, rfactor[2] - 1);

   nalu_hypre_MPI_Comm_rank(comm, &myproc);

   recvinfo_data = nalu_hypre_CTAlloc(nalu_hypre_SStructRecvInfoData,  1, NALU_HYPRE_MEMORY_HOST);

   /*------------------------------------------------------------------------
    * Create the structured recvbox patterns.
    *   recv_boxes are obtained by intersecting this proc's cgrid boxes
    *   with the fine fboxman. Intersecting BoxManEntries not on this proc
    *   will give the boxes that we will be receiving some data from. To
    *   get the exact receiving box extents, we need to take an intersection.
    *   Since only coarse data is communicated, these intersection boxes
    *   must be coarsened.
    *------------------------------------------------------------------------*/
   intersect_box = nalu_hypre_BoxCreate(ndim);
   grid_boxes   = nalu_hypre_StructGridBoxes(cgrid);

   recv_boxes = nalu_hypre_BoxArrayArrayCreate(nalu_hypre_BoxArraySize(grid_boxes), ndim);
   recv_processes = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(grid_boxes), NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ForBoxI(i, grid_boxes)
   {
      grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);

      nalu_hypre_SStructIndexScaleC_F(nalu_hypre_BoxIMin(grid_box), index1,
                                 rfactor, nalu_hypre_BoxIMin(&fbox));
      nalu_hypre_SStructIndexScaleC_F(nalu_hypre_BoxIMax(grid_box), index2,
                                 rfactor, nalu_hypre_BoxIMax(&fbox));

      nalu_hypre_BoxManIntersect(fboxman, nalu_hypre_BoxIMin(&fbox), nalu_hypre_BoxIMax(&fbox),
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
      recv_processes[i]     = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt, NALU_HYPRE_MEMORY_HOST);

      cnt = 0;
      for (j = 0; j < nboxman_entries; j++)
      {
         nalu_hypre_SStructBoxManEntryGetProcess(boxman_entries[j], &proc);

         /* determine the chunk of the boxman_entries[j] box that is needed */
         nalu_hypre_BoxManEntryGetExtents(boxman_entries[j], ilower, iupper);
         nalu_hypre_BoxSetExtents(&boxman_entry_box, ilower, iupper);
         nalu_hypre_IntersectBoxes(&boxman_entry_box, &fbox, &boxman_entry_box);

         if (proc != myproc)
         {
            recv_processes[i][cnt] = proc;
            nalu_hypre_SStructIndexScaleF_C(nalu_hypre_BoxIMin(&boxman_entry_box), index1,
                                       rfactor, nalu_hypre_BoxIMin(&boxman_entry_box));
            nalu_hypre_SStructIndexScaleF_C(nalu_hypre_BoxIMax(&boxman_entry_box), index1,
                                       rfactor, nalu_hypre_BoxIMax(&boxman_entry_box));
            nalu_hypre_AppendBox(&boxman_entry_box,
                            nalu_hypre_BoxArrayArrayBoxArray(recv_boxes, i));
            cnt++;
         }
      }
      nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);
   }  /* nalu_hypre_ForBoxI(i, grid_boxes) */

   nalu_hypre_BoxDestroy(intersect_box);

   (recvinfo_data -> size)      = nalu_hypre_BoxArraySize(grid_boxes);
   (recvinfo_data -> recv_boxes) = recv_boxes;
   (recvinfo_data -> recv_procs) = recv_processes;

   return recvinfo_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructRecvInfoDataDestroy
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_SStructRecvInfoDataDestroy(nalu_hypre_SStructRecvInfoData *recvinfo_data)
{
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int i;

   if (recvinfo_data)
   {
      if (recvinfo_data -> recv_boxes)
      {
         nalu_hypre_BoxArrayArrayDestroy( (recvinfo_data -> recv_boxes) );
      }

      for (i = 0; i < (recvinfo_data -> size); i++)
      {
         if (recvinfo_data -> recv_procs[i])
         {
            nalu_hypre_TFree(recvinfo_data -> recv_procs[i], NALU_HYPRE_MEMORY_HOST);
         }

      }
      nalu_hypre_TFree(recvinfo_data -> recv_procs, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(recvinfo_data, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

