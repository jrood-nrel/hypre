/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * Note that send_coords, recv_coords, send_dirs, recv_dirs may be NULL to
 * represent an identity transform.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CommInfoCreate( nalu_hypre_BoxArrayArray  *send_boxes,
                      nalu_hypre_BoxArrayArray  *recv_boxes,
                      NALU_HYPRE_Int           **send_procs,
                      NALU_HYPRE_Int           **recv_procs,
                      NALU_HYPRE_Int           **send_rboxnums,
                      NALU_HYPRE_Int           **recv_rboxnums,
                      nalu_hypre_BoxArrayArray  *send_rboxes,
                      nalu_hypre_BoxArrayArray  *recv_rboxes,
                      NALU_HYPRE_Int             boxes_match,
                      nalu_hypre_CommInfo      **comm_info_ptr )
{
   nalu_hypre_CommInfo  *comm_info;

   comm_info = nalu_hypre_TAlloc(nalu_hypre_CommInfo,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_CommInfoNDim(comm_info)          = nalu_hypre_BoxArrayArrayNDim(send_boxes);
   nalu_hypre_CommInfoSendBoxes(comm_info)     = send_boxes;
   nalu_hypre_CommInfoRecvBoxes(comm_info)     = recv_boxes;
   nalu_hypre_CommInfoSendProcesses(comm_info) = send_procs;
   nalu_hypre_CommInfoRecvProcesses(comm_info) = recv_procs;
   nalu_hypre_CommInfoSendRBoxnums(comm_info)  = send_rboxnums;
   nalu_hypre_CommInfoRecvRBoxnums(comm_info)  = recv_rboxnums;
   nalu_hypre_CommInfoSendRBoxes(comm_info)    = send_rboxes;
   nalu_hypre_CommInfoRecvRBoxes(comm_info)    = recv_rboxes;

   nalu_hypre_CommInfoNumTransforms(comm_info)  = 0;
   nalu_hypre_CommInfoCoords(comm_info)         = NULL;
   nalu_hypre_CommInfoDirs(comm_info)           = NULL;
   nalu_hypre_CommInfoSendTransforms(comm_info) = NULL;
   nalu_hypre_CommInfoRecvTransforms(comm_info) = NULL;

   nalu_hypre_CommInfoBoxesMatch(comm_info)    = boxes_match;
   nalu_hypre_SetIndex(nalu_hypre_CommInfoSendStride(comm_info), 1);
   nalu_hypre_SetIndex(nalu_hypre_CommInfoRecvStride(comm_info), 1);

   *comm_info_ptr = comm_info;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CommInfoSetTransforms( nalu_hypre_CommInfo  *comm_info,
                             NALU_HYPRE_Int        num_transforms,
                             nalu_hypre_Index     *coords,
                             nalu_hypre_Index     *dirs,
                             NALU_HYPRE_Int      **send_transforms,
                             NALU_HYPRE_Int      **recv_transforms )
{
   nalu_hypre_CommInfoNumTransforms(comm_info)  = num_transforms;
   nalu_hypre_CommInfoCoords(comm_info)         = coords;
   nalu_hypre_CommInfoDirs(comm_info)           = dirs;
   nalu_hypre_CommInfoSendTransforms(comm_info) = send_transforms;
   nalu_hypre_CommInfoRecvTransforms(comm_info) = recv_transforms;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CommInfoGetTransforms( nalu_hypre_CommInfo  *comm_info,
                             NALU_HYPRE_Int       *num_transforms,
                             nalu_hypre_Index    **coords,
                             nalu_hypre_Index    **dirs )
{
   *num_transforms = nalu_hypre_CommInfoNumTransforms(comm_info);
   *coords         = nalu_hypre_CommInfoCoords(comm_info);
   *dirs           = nalu_hypre_CommInfoDirs(comm_info);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CommInfoProjectSend( nalu_hypre_CommInfo  *comm_info,
                           nalu_hypre_Index      index,
                           nalu_hypre_Index      stride )
{
   nalu_hypre_ProjectBoxArrayArray(nalu_hypre_CommInfoSendBoxes(comm_info),
                              index, stride);
   nalu_hypre_ProjectBoxArrayArray(nalu_hypre_CommInfoSendRBoxes(comm_info),
                              index, stride);
   nalu_hypre_CopyIndex(stride, nalu_hypre_CommInfoSendStride(comm_info));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CommInfoProjectRecv( nalu_hypre_CommInfo  *comm_info,
                           nalu_hypre_Index      index,
                           nalu_hypre_Index      stride )
{
   nalu_hypre_ProjectBoxArrayArray(nalu_hypre_CommInfoRecvBoxes(comm_info),
                              index, stride);
   nalu_hypre_ProjectBoxArrayArray(nalu_hypre_CommInfoRecvRBoxes(comm_info),
                              index, stride);
   nalu_hypre_CopyIndex(stride, nalu_hypre_CommInfoRecvStride(comm_info));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CommInfoDestroy( nalu_hypre_CommInfo  *comm_info )
{
   NALU_HYPRE_Int           **processes;
   NALU_HYPRE_Int           **rboxnums;
   NALU_HYPRE_Int           **transforms;
   NALU_HYPRE_Int             i, size;

   if (comm_info)
   {
      size = nalu_hypre_BoxArrayArraySize(nalu_hypre_CommInfoSendBoxes(comm_info));
      nalu_hypre_BoxArrayArrayDestroy(nalu_hypre_CommInfoSendBoxes(comm_info));
      processes = nalu_hypre_CommInfoSendProcesses(comm_info);
      for (i = 0; i < size; i++)
      {
         nalu_hypre_TFree(processes[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(processes, NALU_HYPRE_MEMORY_HOST);
      rboxnums = nalu_hypre_CommInfoSendRBoxnums(comm_info);
      if (rboxnums != NULL)
      {
         for (i = 0; i < size; i++)
         {
            nalu_hypre_TFree(rboxnums[i], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(rboxnums, NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_BoxArrayArrayDestroy(nalu_hypre_CommInfoSendRBoxes(comm_info));
      transforms = nalu_hypre_CommInfoSendTransforms(comm_info);
      if (transforms != NULL)
      {
         for (i = 0; i < size; i++)
         {
            nalu_hypre_TFree(transforms[i], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(transforms, NALU_HYPRE_MEMORY_HOST);
      }

      size = nalu_hypre_BoxArrayArraySize(nalu_hypre_CommInfoRecvBoxes(comm_info));
      nalu_hypre_BoxArrayArrayDestroy(nalu_hypre_CommInfoRecvBoxes(comm_info));
      processes = nalu_hypre_CommInfoRecvProcesses(comm_info);
      for (i = 0; i < size; i++)
      {
         nalu_hypre_TFree(processes[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(processes, NALU_HYPRE_MEMORY_HOST);
      rboxnums = nalu_hypre_CommInfoRecvRBoxnums(comm_info);
      if (rboxnums != NULL)
      {
         for (i = 0; i < size; i++)
         {
            nalu_hypre_TFree(rboxnums[i], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(rboxnums, NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_BoxArrayArrayDestroy(nalu_hypre_CommInfoRecvRBoxes(comm_info));
      transforms = nalu_hypre_CommInfoRecvTransforms(comm_info);
      if (transforms != NULL)
      {
         for (i = 0; i < size; i++)
         {
            nalu_hypre_TFree(transforms[i], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(transforms, NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_TFree(nalu_hypre_CommInfoCoords(comm_info), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_CommInfoDirs(comm_info), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(comm_info, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NEW version that uses the box manager to find neighbors boxes.
 * AHB 9/06
 *
 * Return descriptions of communications patterns for a given
 * grid-stencil computation.  These patterns are defined by
 * intersecting the data dependencies of each box (including data
 * dependencies within the box) with its neighbor boxes.
 *
 * An inconsistent ordering of the boxes in the send/recv data regions
 * is returned.  That is, the ordering of the boxes on process p for
 * receives from process q is not guaranteed to be the same as the
 * ordering of the boxes on process q for sends to process p.
 *
 * The routine uses a grow-the-box-and-intersect-with-neighbors style
 * algorithm.
 *
 * 1. The basic algorithm:
 *
 * The basic algorithm is as follows, with one additional optimization
 * discussed below that helps to minimize the number of communications
 * that are done with neighbors (e.g., consider a 7-pt stencil and the
 * difference between doing 26 communications versus 6):
 *
 * To compute send/recv regions, do
 *
 *   for i = local box
 *   {
 *      gbox_i = grow box i according to stencil
 *
 *      //find neighbors of i
 *      call BoxManIntersect on gbox_i (and periodic gbox_i)
 *
 *      // receives
 *      for j = neighbor box of i
 *      {
 *         intersect gbox_i with box j and add to recv region
 *      }
 *
 *      // sends
 *      for j = neighbor box of i
 *      {
 *         gbox_j = grow box j according to stencil
 *         intersect gbox_j with box i and add to send region
 *      }
 *   }
 *
 *   (Note: no ordering is assumed)
 *
 * 2. Optimization on basic algorithm:
 *
 * Before looping over the neighbors in the above algorithm, do a
 * preliminary sweep through the neighbors to select a subset of
 * neighbors to do the intersections with.  To select the subset,
 * compute a so-called "distance index" and check the corresponding
 * entry in the so-called "stencil grid" to decide whether or not to
 * use the box.
 *
 * The "stencil grid" is a 3x3x3 grid in 3D that is built from the
 * stencil as follows:
 *
 *   // assume for simplicity that i,j,k are -1, 0, or 1
 *   for each stencil entry (i,j,k)
 *   {
 *      mark all stencil grid entries in (1,1,1) x (1+i,1+j,1+k)
 *      // here (1,1,1) is the "center" entry in the stencil grid
 *   }
 *
 *
 * 3. Complications with periodicity:
 *
 * When periodicity is on, it is possible to have a box-pair region
 * (the description of a communication pattern between two boxes) that
 * consists of more than one box.
 *
 * 4.  Box Manager
 *
 *   The box manager is used to determine neighbors.  It is assumed
 *   that the grid's box manager contains sufficient neighbor
 *   information.
 *
 * NOTES:
 *
 *    A. No concept of data ownership is assumed.  As a result,
 *       redundant communication patterns can be produced when the grid
 *       boxes overlap.
 *
 *    B. Boxes in the send and recv regions do not need to be in any
 *       particular order (including those that are periodic).
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CreateCommInfoFromStencil( nalu_hypre_StructGrid      *grid,
                                 nalu_hypre_StructStencil   *stencil,
                                 nalu_hypre_CommInfo       **comm_info_ptr )
{
   NALU_HYPRE_Int              ndim = nalu_hypre_StructGridNDim(grid);
   NALU_HYPRE_Int              i, j, k, d, m, s, si;

   nalu_hypre_BoxArrayArray   *send_boxes;
   nalu_hypre_BoxArrayArray   *recv_boxes;

   NALU_HYPRE_Int            **send_procs;
   NALU_HYPRE_Int            **recv_procs;
   NALU_HYPRE_Int            **send_rboxnums;
   NALU_HYPRE_Int            **recv_rboxnums;
   nalu_hypre_BoxArrayArray   *send_rboxes;
   nalu_hypre_BoxArrayArray   *recv_rboxes;

   nalu_hypre_BoxArray        *local_boxes;
   NALU_HYPRE_Int              num_boxes;

   nalu_hypre_BoxManager      *boxman;

   nalu_hypre_Index           *stencil_shape;
   nalu_hypre_IndexRef         stencil_offset;
   nalu_hypre_IndexRef         pshift;

   nalu_hypre_Box             *box;
   nalu_hypre_Box             *hood_box;
   nalu_hypre_Box             *grow_box;
   nalu_hypre_Box             *extend_box;
   nalu_hypre_Box             *int_box;
   nalu_hypre_Box             *periodic_box;

   nalu_hypre_Box             *stencil_box, *sbox; /* extents of the stencil grid */
   NALU_HYPRE_Int             *stencil_grid;
   NALU_HYPRE_Int              grow[NALU_HYPRE_MAXDIM][2];

   nalu_hypre_BoxManEntry    **entries;
   nalu_hypre_BoxManEntry     *entry;

   NALU_HYPRE_Int              num_entries;
   nalu_hypre_BoxArray        *neighbor_boxes = NULL;
   NALU_HYPRE_Int             *neighbor_procs = NULL;
   NALU_HYPRE_Int             *neighbor_ids = NULL;
   NALU_HYPRE_Int             *neighbor_shifts = NULL;
   NALU_HYPRE_Int              neighbor_count;
   NALU_HYPRE_Int              neighbor_alloc;

   nalu_hypre_Index            ilower, iupper;

   nalu_hypre_BoxArray        *send_box_array;
   nalu_hypre_BoxArray        *recv_box_array;
   nalu_hypre_BoxArray        *send_rbox_array;
   nalu_hypre_BoxArray        *recv_rbox_array;

   nalu_hypre_Box            **cboxes;
   nalu_hypre_Box             *cboxes_mem;
   NALU_HYPRE_Int             *cboxes_neighbor_location;
   NALU_HYPRE_Int              num_cboxes, cbox_alloc;

   nalu_hypre_Index            istart, istop, sgindex;
   nalu_hypre_IndexRef         start;
   nalu_hypre_Index            loop_size, stride;

   NALU_HYPRE_Int              num_periods, loc, box_id, id, proc_id;
   NALU_HYPRE_Int              myid;

   MPI_Comm               comm;

   /*------------------------------------------------------
    * Initializations
    *------------------------------------------------------*/

   nalu_hypre_SetIndex(ilower, 0);
   nalu_hypre_SetIndex(iupper, 0);
   nalu_hypre_SetIndex(istart, 0);
   nalu_hypre_SetIndex(istop, 0);
   nalu_hypre_SetIndex(sgindex, 0);

   local_boxes = nalu_hypre_StructGridBoxes(grid);
   num_boxes   = nalu_hypre_BoxArraySize(local_boxes);
   num_periods = nalu_hypre_StructGridNumPeriods(grid);

   boxman = nalu_hypre_StructGridBoxMan(grid);
   comm   = nalu_hypre_StructGridComm(grid);

   nalu_hypre_MPI_Comm_rank(comm, &myid);

   stencil_box = nalu_hypre_BoxCreate(ndim);
   nalu_hypre_SetIndex(nalu_hypre_BoxIMin(stencil_box), 0);
   nalu_hypre_SetIndex(nalu_hypre_BoxIMax(stencil_box), 2);

   /* Set initial values to zero */
   stencil_grid = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxVolume(stencil_box), NALU_HYPRE_MEMORY_HOST);

   sbox = nalu_hypre_BoxCreate(ndim);
   nalu_hypre_SetIndex(stride, 1);

   /*------------------------------------------------------
    * Compute the "grow" information from the stencil
    *------------------------------------------------------*/

   stencil_shape = nalu_hypre_StructStencilShape(stencil);

   for (d = 0; d < ndim; d++)
   {
      grow[d][0] = 0;
      grow[d][1] = 0;
   }

   for (s = 0; s < nalu_hypre_StructStencilSize(stencil); s++)
   {
      stencil_offset = stencil_shape[s];

      for (d = 0; d < ndim; d++)
      {
         m = stencil_offset[d];

         istart[d] = 1;
         istop[d]  = 1;

         if (m < 0)
         {
            istart[d] = 0;
            grow[d][0] = nalu_hypre_max(grow[d][0], -m);
         }
         else if (m > 0)
         {
            istop[d] = 2;
            grow[d][1] = nalu_hypre_max(grow[d][1],  m);
         }
      }

      /* update stencil grid from the grow_stencil */
      nalu_hypre_BoxSetExtents(sbox, istart, istop);
      start = nalu_hypre_BoxIMin(sbox);
      nalu_hypre_BoxGetSize(sbox, loop_size);

      nalu_hypre_SerialBoxLoop1Begin(ndim, loop_size,
                                stencil_box, start, stride, si);
      {
         stencil_grid[si] = 1;
      }
      nalu_hypre_SerialBoxLoop1End(si);
   }

   /*------------------------------------------------------
    * Compute send/recv boxes and procs for each local box
    *------------------------------------------------------*/

   /* initialize: for each local box, we create an array of send/recv info */

   send_boxes = nalu_hypre_BoxArrayArrayCreate(num_boxes, ndim);
   recv_boxes = nalu_hypre_BoxArrayArrayCreate(num_boxes, ndim);
   send_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  num_boxes, NALU_HYPRE_MEMORY_HOST);
   recv_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  num_boxes, NALU_HYPRE_MEMORY_HOST);

   /* Remote boxnums and boxes describe data on the opposing processor, so some
      shifting of boxes is needed below for periodic neighbor boxes.  Remote box
      info is also needed for receives to allow for reverse communication. */
   send_rboxnums = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  num_boxes, NALU_HYPRE_MEMORY_HOST);
   send_rboxes   = nalu_hypre_BoxArrayArrayCreate(num_boxes, ndim);
   recv_rboxnums = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  num_boxes, NALU_HYPRE_MEMORY_HOST);
   recv_rboxes   = nalu_hypre_BoxArrayArrayCreate(num_boxes, ndim);

   grow_box = nalu_hypre_BoxCreate(nalu_hypre_StructGridNDim(grid));
   extend_box = nalu_hypre_BoxCreate(nalu_hypre_StructGridNDim(grid));
   int_box  = nalu_hypre_BoxCreate(nalu_hypre_StructGridNDim(grid));
   periodic_box =  nalu_hypre_BoxCreate(nalu_hypre_StructGridNDim(grid));

   /* storage we will use and keep track of the neighbors */
   neighbor_alloc = 30; /* initial guess at max size */
   neighbor_boxes = nalu_hypre_BoxArrayCreate(neighbor_alloc, ndim);
   neighbor_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  neighbor_alloc, NALU_HYPRE_MEMORY_HOST);
   neighbor_ids = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  neighbor_alloc, NALU_HYPRE_MEMORY_HOST);
   neighbor_shifts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  neighbor_alloc, NALU_HYPRE_MEMORY_HOST);

   /* storage we will use to collect all of the intersected boxes (the send and
      recv regions for box i (this may not be enough in the case of periodic
      boxes, so we will have to check) */
   cbox_alloc =  nalu_hypre_BoxManNEntries(boxman);

   cboxes_neighbor_location = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cbox_alloc, NALU_HYPRE_MEMORY_HOST);
   cboxes = nalu_hypre_CTAlloc(nalu_hypre_Box *,  cbox_alloc, NALU_HYPRE_MEMORY_HOST);
   cboxes_mem = nalu_hypre_CTAlloc(nalu_hypre_Box,  cbox_alloc, NALU_HYPRE_MEMORY_HOST);

   /******* loop through each local box **************/

   for (i = 0; i < num_boxes; i++)
   {
      /* get the box */
      box = nalu_hypre_BoxArrayBox(local_boxes, i);
      box_id = i;

      /* grow box local i according to the stencil*/
      nalu_hypre_CopyBox(box, grow_box);
      for (d = 0; d < ndim; d++)
      {
         nalu_hypre_BoxIMinD(grow_box, d) -= grow[d][0];
         nalu_hypre_BoxIMaxD(grow_box, d) += grow[d][1];
      }

      /* extend_box - to find the list of potential neighbors, we need to grow
         the local box a bit differently in case, for example, the stencil grows
         in one dimension [0] and not the other [1] */
      nalu_hypre_CopyBox(box, extend_box);
      for (d = 0; d < ndim; d++)
      {
         nalu_hypre_BoxIMinD(extend_box, d) -= nalu_hypre_max(grow[d][0], grow[d][1]);
         nalu_hypre_BoxIMaxD(extend_box, d) += nalu_hypre_max(grow[d][0], grow[d][1]);
      }

      /*------------------------------------------------
       * Determine the neighbors of box i
       *------------------------------------------------*/

      /* Do this by intersecting the extend box with the BoxManager.
         We must also check for periodic neighbors. */

      neighbor_count = 0;
      nalu_hypre_BoxArraySetSize(neighbor_boxes, 0);
      /* shift the box by each period (k=0 is original box) */
      for (k = 0; k < num_periods; k++)
      {
         nalu_hypre_CopyBox(extend_box, periodic_box);
         pshift = nalu_hypre_StructGridPShift(grid, k);
         nalu_hypre_BoxShiftPos(periodic_box, pshift);

         /* get the intersections */
         nalu_hypre_BoxManIntersect(boxman, nalu_hypre_BoxIMin(periodic_box),
                               nalu_hypre_BoxIMax(periodic_box),
                               &entries, &num_entries);

         /* note: do we need to remove the intersection with our original box?
            no if periodic, yes if non-periodic (k=0) */

         /* unpack entries (first check storage) */
         if (neighbor_count + num_entries > neighbor_alloc)
         {
            neighbor_alloc = neighbor_count + num_entries + 5;
            neighbor_procs = nalu_hypre_TReAlloc(neighbor_procs,  NALU_HYPRE_Int,
                                            neighbor_alloc, NALU_HYPRE_MEMORY_HOST);
            neighbor_ids = nalu_hypre_TReAlloc(neighbor_ids,  NALU_HYPRE_Int,  neighbor_alloc, NALU_HYPRE_MEMORY_HOST);
            neighbor_shifts = nalu_hypre_TReAlloc(neighbor_shifts,  NALU_HYPRE_Int,
                                             neighbor_alloc, NALU_HYPRE_MEMORY_HOST);
         }
         /* check storage for the array */
         nalu_hypre_BoxArraySetSize(neighbor_boxes, neighbor_count + num_entries);
         /* now unpack */
         for (j = 0; j < num_entries; j++)
         {
            entry = entries[j];
            proc_id = nalu_hypre_BoxManEntryProc(entry);
            id = nalu_hypre_BoxManEntryId(entry);
            /* don't keep box i in the non-periodic case*/
            if (!k)
            {
               if ((myid == proc_id) && (box_id == id))
               {
                  continue;
               }
            }

            nalu_hypre_BoxManEntryGetExtents(entry, ilower, iupper);
            nalu_hypre_BoxSetExtents(nalu_hypre_BoxArrayBox(neighbor_boxes, neighbor_count),
                                ilower, iupper);
            /* shift the periodic boxes (needs to be the opposite of above) */
            if (k)
            {
               nalu_hypre_BoxShiftNeg(
                  nalu_hypre_BoxArrayBox(neighbor_boxes, neighbor_count), pshift);
            }

            neighbor_procs[neighbor_count] = proc_id;
            neighbor_ids[neighbor_count] = id;
            neighbor_shifts[neighbor_count] = k;
            neighbor_count++;
         }
         nalu_hypre_BoxArraySetSize(neighbor_boxes, neighbor_count);

         nalu_hypre_TFree(entries, NALU_HYPRE_MEMORY_HOST);

      } /* end of loop through periods k */

      /* Now we have a list of all of the neighbors for box i! */

      /* note: we don't want/need to remove duplicates - they should have
         different intersections (TO DO: put more thought into if there are ever
         any exceptions to this? - the intersection routine already eliminates
         duplicates - so what i mean is eliminating duplicates from multiple
         intersection calls in periodic case)  */

      /*------------------------------------------------
       * Compute recv_box_array for box i
       *------------------------------------------------*/

      /* check size of storage for cboxes */
      /* let's make sure that we have enough storage in case each neighbor
         produces a send/recv region */
      if (neighbor_count > cbox_alloc)
      {
         cbox_alloc = neighbor_count;
         cboxes_neighbor_location = nalu_hypre_TReAlloc(cboxes_neighbor_location,
                                                   NALU_HYPRE_Int,  cbox_alloc, NALU_HYPRE_MEMORY_HOST);
         cboxes = nalu_hypre_TReAlloc(cboxes,  nalu_hypre_Box *,  cbox_alloc, NALU_HYPRE_MEMORY_HOST);
         cboxes_mem = nalu_hypre_TReAlloc(cboxes_mem,  nalu_hypre_Box,  cbox_alloc, NALU_HYPRE_MEMORY_HOST);
      }

      /* Loop through each neighbor box.  If the neighbor box intersects the
         grown box i (grown according to our stencil), then the intersection is
         a recv region.  If the neighbor box was shifted to handle periodicity,
         we need to (positive) shift it back. */

      num_cboxes = 0;

      for (k = 0; k < neighbor_count; k++)
      {
         hood_box = nalu_hypre_BoxArrayBox(neighbor_boxes, k);
         /* check the stencil grid to see if it makes sense to intersect */
         for (d = 0; d < ndim; d++)
         {
            sgindex[d] = 1;

            s = nalu_hypre_BoxIMinD(hood_box, d) - nalu_hypre_BoxIMaxD(box, d);
            if (s > 0)
            {
               sgindex[d] = 2;
            }
            s = nalu_hypre_BoxIMinD(box, d) - nalu_hypre_BoxIMaxD(hood_box, d);
            if (s > 0)
            {
               sgindex[d] = 0;
            }
         }
         /* it makes sense only if we have at least one non-zero entry */
         si = nalu_hypre_BoxIndexRank(stencil_box, sgindex);
         if (stencil_grid[si])
         {
            /* intersect - result is int_box */
            nalu_hypre_IntersectBoxes(grow_box, hood_box, int_box);
            /* if we have a positive volume box, this is a recv region */
            if (nalu_hypre_BoxVolume(int_box))
            {
               /* keep track of which neighbor: k... */
               cboxes_neighbor_location[num_cboxes] = k;
               cboxes[num_cboxes] = &cboxes_mem[num_cboxes];
               /* keep the intersected box */
               nalu_hypre_CopyBox(int_box, cboxes[num_cboxes]);
               num_cboxes++;
            }
         }
      } /* end of loop through each neighbor */

      /* create recv_box_array and recv_procs for box i */
      recv_box_array = nalu_hypre_BoxArrayArrayBoxArray(recv_boxes, i);
      nalu_hypre_BoxArraySetSize(recv_box_array, num_cboxes);
      recv_procs[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cboxes, NALU_HYPRE_MEMORY_HOST);
      recv_rboxnums[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cboxes, NALU_HYPRE_MEMORY_HOST);
      recv_rbox_array = nalu_hypre_BoxArrayArrayBoxArray(recv_rboxes, i);
      nalu_hypre_BoxArraySetSize(recv_rbox_array, num_cboxes);

      for (m = 0; m < num_cboxes; m++)
      {
         loc = cboxes_neighbor_location[m];
         recv_procs[i][m] = neighbor_procs[loc];
         recv_rboxnums[i][m] = neighbor_ids[loc];
         nalu_hypre_CopyBox(cboxes[m], nalu_hypre_BoxArrayBox(recv_box_array, m));

         /* if periodic, positive shift before copying to the rbox_array */
         if (neighbor_shifts[loc]) /* periodic if shift != 0 */
         {
            pshift = nalu_hypre_StructGridPShift(grid, neighbor_shifts[loc]);
            nalu_hypre_BoxShiftPos(cboxes[m], pshift);
         }
         nalu_hypre_CopyBox(cboxes[m], nalu_hypre_BoxArrayBox(recv_rbox_array, m));

         cboxes[m] = NULL;
      }

      /*------------------------------------------------
       * Compute send_box_array for box i
       *------------------------------------------------*/

      /* Loop through each neighbor box.  If the grown neighbor box intersects
         box i, then the intersection is a send region.  If the neighbor box was
         shifted to handle periodicity, we need to (positive) shift it back. */

      num_cboxes = 0;

      for (k = 0; k < neighbor_count; k++)
      {
         hood_box = nalu_hypre_BoxArrayBox(neighbor_boxes, k);
         /* check the stencil grid to see if it makes sense to intersect */
         for (d = 0; d < ndim; d++)
         {
            sgindex[d] = 1;

            s = nalu_hypre_BoxIMinD(box, d) - nalu_hypre_BoxIMaxD(hood_box, d);
            if (s > 0)
            {
               sgindex[d] = 2;
            }
            s = nalu_hypre_BoxIMinD(hood_box, d) - nalu_hypre_BoxIMaxD(box, d);
            if (s > 0)
            {
               sgindex[d] = 0;
            }
         }
         /* it makes sense only if we have at least one non-zero entry */
         si = nalu_hypre_BoxIndexRank(stencil_box, sgindex);
         if (stencil_grid[si])
         {
            /* grow the neighbor box and intersect */
            nalu_hypre_CopyBox(hood_box, grow_box);
            for (d = 0; d < ndim; d++)
            {
               nalu_hypre_BoxIMinD(grow_box, d) -= grow[d][0];
               nalu_hypre_BoxIMaxD(grow_box, d) += grow[d][1];
            }
            nalu_hypre_IntersectBoxes(box, grow_box, int_box);
            /* if we have a positive volume box, this is a send region */
            if (nalu_hypre_BoxVolume(int_box))
            {
               /* keep track of which neighbor: k... */
               cboxes_neighbor_location[num_cboxes] = k;
               cboxes[num_cboxes] = &cboxes_mem[num_cboxes];
               /* keep the intersected box */
               nalu_hypre_CopyBox(int_box, cboxes[num_cboxes]);
               num_cboxes++;
            }
         }
      }/* end of loop through neighbors */

      /* create send_box_array and send_procs for box i */
      send_box_array = nalu_hypre_BoxArrayArrayBoxArray(send_boxes, i);
      nalu_hypre_BoxArraySetSize(send_box_array, num_cboxes);
      send_procs[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cboxes, NALU_HYPRE_MEMORY_HOST);
      send_rboxnums[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cboxes, NALU_HYPRE_MEMORY_HOST);
      send_rbox_array = nalu_hypre_BoxArrayArrayBoxArray(send_rboxes, i);
      nalu_hypre_BoxArraySetSize(send_rbox_array, num_cboxes);

      for (m = 0; m < num_cboxes; m++)
      {
         loc = cboxes_neighbor_location[m];
         send_procs[i][m] = neighbor_procs[loc];
         send_rboxnums[i][m] = neighbor_ids[loc];
         nalu_hypre_CopyBox(cboxes[m], nalu_hypre_BoxArrayBox(send_box_array, m));

         /* if periodic, positive shift before copying to the rbox_array */
         if (neighbor_shifts[loc]) /* periodic if shift != 0 */
         {
            pshift = nalu_hypre_StructGridPShift(grid, neighbor_shifts[loc]);
            nalu_hypre_BoxShiftPos(cboxes[m], pshift);
         }
         nalu_hypre_CopyBox(cboxes[m], nalu_hypre_BoxArrayBox(send_rbox_array, m));

         cboxes[m] = NULL;
      }
   } /* end of loop through each local box */

   /* clean up */
   nalu_hypre_TFree(neighbor_procs, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(neighbor_ids, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(neighbor_shifts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_BoxArrayDestroy(neighbor_boxes);

   nalu_hypre_TFree(cboxes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(cboxes_mem, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(cboxes_neighbor_location, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_BoxDestroy(grow_box);
   nalu_hypre_BoxDestroy(int_box);
   nalu_hypre_BoxDestroy(periodic_box);
   nalu_hypre_BoxDestroy(extend_box);

   nalu_hypre_BoxDestroy(stencil_box);
   nalu_hypre_BoxDestroy(sbox);
   nalu_hypre_TFree(stencil_grid, NALU_HYPRE_MEMORY_HOST);

   /*------------------------------------------------------
    * Return
    *------------------------------------------------------*/

   nalu_hypre_CommInfoCreate(send_boxes, recv_boxes, send_procs, recv_procs,
                        send_rboxnums, recv_rboxnums, send_rboxes, recv_rboxes,
                        1, comm_info_ptr);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return descriptions of communications patterns for a given grid
 * based on a specified number of "ghost zones".  These patterns are
 * defined by building a stencil and calling CommInfoFromStencil.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CreateCommInfoFromNumGhost( nalu_hypre_StructGrid      *grid,
                                  NALU_HYPRE_Int             *num_ghost,
                                  nalu_hypre_CommInfo       **comm_info_ptr )
{
   NALU_HYPRE_Int             ndim = nalu_hypre_StructGridNDim(grid);
   nalu_hypre_StructStencil  *stencil;
   nalu_hypre_Index          *stencil_shape;
   nalu_hypre_Box            *box;
   nalu_hypre_Index           ii, loop_size;
   nalu_hypre_IndexRef        start;
   NALU_HYPRE_Int             i, d, size;

   size = (NALU_HYPRE_Int)(pow(3.0, ndim) + 0.5);
   stencil_shape = nalu_hypre_CTAlloc(nalu_hypre_Index,  size, NALU_HYPRE_MEMORY_HOST);
   box = nalu_hypre_BoxCreate(ndim);
   for (d = 0; d < ndim; d++)
   {
      nalu_hypre_BoxIMinD(box, d) = -(num_ghost[2 * d]   ? 1 : 0);
      nalu_hypre_BoxIMaxD(box, d) =  (num_ghost[2 * d + 1] ? 1 : 0);
   }

   size = 0;
   start = nalu_hypre_BoxIMin(box);
   nalu_hypre_BoxGetSize(box, loop_size);
   nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
   {
      zypre_BoxLoopGetIndex(ii);
      for (d = 0; d < ndim; d++)
      {
         i = ii[d] + start[d];
         if (i < 0)
         {
            stencil_shape[size][d] = -num_ghost[2 * d];
         }
         else if (i > 0)
         {
            stencil_shape[size][d] =  num_ghost[2 * d + 1];
         }
      }
      size++;
   }
   nalu_hypre_SerialBoxLoop0End();

   nalu_hypre_BoxDestroy(box);

   stencil = nalu_hypre_StructStencilCreate(ndim, size, stencil_shape);
   nalu_hypre_CreateCommInfoFromStencil(grid, stencil, comm_info_ptr);
   nalu_hypre_StructStencilDestroy(stencil);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Return descriptions of communications patterns for migrating data
 * from one grid distribution to another.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CreateCommInfoFromGrids( nalu_hypre_StructGrid      *from_grid,
                               nalu_hypre_StructGrid      *to_grid,
                               nalu_hypre_CommInfo       **comm_info_ptr )
{
   nalu_hypre_BoxArrayArray     *send_boxes;
   nalu_hypre_BoxArrayArray     *recv_boxes;
   NALU_HYPRE_Int              **send_procs;
   NALU_HYPRE_Int              **recv_procs;
   NALU_HYPRE_Int              **send_rboxnums;
   NALU_HYPRE_Int              **recv_rboxnums;
   nalu_hypre_BoxArrayArray     *send_rboxes;
   nalu_hypre_BoxArrayArray     *recv_rboxes;

   nalu_hypre_BoxArrayArray     *comm_boxes;
   NALU_HYPRE_Int              **comm_procs;
   NALU_HYPRE_Int              **comm_boxnums;
   nalu_hypre_BoxArray          *comm_box_array;
   nalu_hypre_Box               *comm_box;

   nalu_hypre_StructGrid        *local_grid;
   nalu_hypre_StructGrid        *remote_grid;

   nalu_hypre_BoxArray          *local_boxes;
   nalu_hypre_BoxArray          *remote_boxes;
   nalu_hypre_BoxArray          *remote_all_boxes;
   NALU_HYPRE_Int               *remote_all_procs;
   NALU_HYPRE_Int               *remote_all_boxnums;
   NALU_HYPRE_Int                remote_first_local;

   nalu_hypre_Box               *local_box;
   nalu_hypre_Box               *remote_box;

   NALU_HYPRE_Int                i, j, k, r, ndim;

   /*------------------------------------------------------
    * Set up communication info
    *------------------------------------------------------*/

   ndim = nalu_hypre_StructGridNDim(from_grid);

   for (r = 0; r < 2; r++)
   {
      switch (r)
      {
         case 0:
            local_grid  = from_grid;
            remote_grid = to_grid;
            break;

         case 1:
            local_grid  = to_grid;
            remote_grid = from_grid;
            break;
      }

      /*---------------------------------------------------
       * Compute comm_boxes and comm_procs
       *---------------------------------------------------*/

      local_boxes  = nalu_hypre_StructGridBoxes(local_grid);
      remote_boxes = nalu_hypre_StructGridBoxes(remote_grid);
      nalu_hypre_GatherAllBoxes(nalu_hypre_StructGridComm(remote_grid), remote_boxes, ndim,
                           &remote_all_boxes,
                           &remote_all_procs,
                           &remote_first_local);
      nalu_hypre_ComputeBoxnums(remote_all_boxes, remote_all_procs,
                           &remote_all_boxnums);

      comm_boxes = nalu_hypre_BoxArrayArrayCreate(nalu_hypre_BoxArraySize(local_boxes), ndim);
      comm_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(local_boxes), NALU_HYPRE_MEMORY_HOST);
      comm_boxnums = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(local_boxes), NALU_HYPRE_MEMORY_HOST);

      comm_box = nalu_hypre_BoxCreate(ndim);
      nalu_hypre_ForBoxI(i, local_boxes)
      {
         local_box = nalu_hypre_BoxArrayBox(local_boxes, i);

         comm_box_array = nalu_hypre_BoxArrayArrayBoxArray(comm_boxes, i);
         comm_procs[i] =
            nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(remote_all_boxes), NALU_HYPRE_MEMORY_HOST);
         comm_boxnums[i] =
            nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(remote_all_boxes), NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_ForBoxI(j, remote_all_boxes)
         {
            remote_box = nalu_hypre_BoxArrayBox(remote_all_boxes, j);

            nalu_hypre_IntersectBoxes(local_box, remote_box, comm_box);
            if (nalu_hypre_BoxVolume(comm_box))
            {
               k = nalu_hypre_BoxArraySize(comm_box_array);
               comm_procs[i][k] = remote_all_procs[j];
               comm_boxnums[i][k] = remote_all_boxnums[j];

               nalu_hypre_AppendBox(comm_box, comm_box_array);
            }
         }

         comm_procs[i] =
            nalu_hypre_TReAlloc(comm_procs[i],
                           NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(comm_box_array), NALU_HYPRE_MEMORY_HOST);
         comm_boxnums[i] =
            nalu_hypre_TReAlloc(comm_boxnums[i],
                           NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(comm_box_array), NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_BoxDestroy(comm_box);

      nalu_hypre_BoxArrayDestroy(remote_all_boxes);
      nalu_hypre_TFree(remote_all_procs, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(remote_all_boxnums, NALU_HYPRE_MEMORY_HOST);

      switch (r)
      {
         case 0:
            send_boxes = comm_boxes;
            send_procs = comm_procs;
            send_rboxnums = comm_boxnums;
            send_rboxes = nalu_hypre_BoxArrayArrayDuplicate(comm_boxes);
            break;

         case 1:
            recv_boxes = comm_boxes;
            recv_procs = comm_procs;
            recv_rboxnums = comm_boxnums;
            recv_rboxes = nalu_hypre_BoxArrayArrayDuplicate(comm_boxes);
            break;
      }
   }

   nalu_hypre_CommInfoCreate(send_boxes, recv_boxes, send_procs, recv_procs,
                        send_rboxnums, recv_rboxnums, send_rboxes, recv_rboxes,
                        1, comm_info_ptr);

   return nalu_hypre_error_flag;
}
