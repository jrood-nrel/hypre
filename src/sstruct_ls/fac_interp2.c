/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * OpenMP Problems
 *
 * Not sure about performace yet, so leaving the '#if 1' blocks below.
 *
 ******************************************************************************/

/******************************************************************************
 *  FAC composite level interpolation.
 *  Identity interpolation of values away from underlying refinement patches;
 *  linear inside patch.
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "fac.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_FacSemiInterpData data structure
 *--------------------------------------------------------------------------*/
typedef struct
{
   NALU_HYPRE_Int             nvars;
   NALU_HYPRE_Int             ndim;
   nalu_hypre_Index           stride;

   nalu_hypre_SStructPVector *recv_cvectors;
   NALU_HYPRE_Int           **recv_boxnum_map;   /* mapping between the boxes of the
                                               recv_grid and the given grid */
   nalu_hypre_BoxArrayArray **identity_arrayboxes;
   nalu_hypre_BoxArrayArray **ownboxes;
   NALU_HYPRE_Int          ***own_cboxnums;

   nalu_hypre_CommPkg       **interlevel_comm;
   nalu_hypre_CommPkg       **gnodes_comm_pkg;

   NALU_HYPRE_Real          **weights;

} nalu_hypre_FacSemiInterpData2;

/*--------------------------------------------------------------------------
 * nalu_hypre_FacSemiInterpCreate
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_FacSemiInterpCreate2( void **fac_interp_vdata_ptr )
{
   NALU_HYPRE_Int                 ierr = 0;
   nalu_hypre_FacSemiInterpData2  *fac_interp_data;

   fac_interp_data = nalu_hypre_CTAlloc(nalu_hypre_FacSemiInterpData2,  1, NALU_HYPRE_MEMORY_HOST);
   *fac_interp_vdata_ptr = (void *) fac_interp_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FacSemiInterpDestroy
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_FacSemiInterpDestroy2( void *fac_interp_vdata)
{
   NALU_HYPRE_Int                 ierr = 0;

   nalu_hypre_FacSemiInterpData2 *fac_interp_data = (nalu_hypre_FacSemiInterpData2 *)fac_interp_vdata;
   NALU_HYPRE_Int                 i, j, size;

   if (fac_interp_data)
   {
      nalu_hypre_SStructPVectorDestroy(fac_interp_data-> recv_cvectors);

      for (i = 0; i < (fac_interp_data-> nvars); i++)
      {
         nalu_hypre_TFree(fac_interp_data -> recv_boxnum_map[i], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_BoxArrayArrayDestroy(fac_interp_data -> identity_arrayboxes[i]);

         size = nalu_hypre_BoxArrayArraySize(fac_interp_data -> ownboxes[i]);
         nalu_hypre_BoxArrayArrayDestroy(fac_interp_data -> ownboxes[i]);
         for (j = 0; j < size; j++)
         {
            nalu_hypre_TFree(fac_interp_data -> own_cboxnums[i][j], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(fac_interp_data -> own_cboxnums[i], NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_CommPkgDestroy(fac_interp_data -> gnodes_comm_pkg[i]);
         nalu_hypre_CommPkgDestroy(fac_interp_data -> interlevel_comm[i]);

      }
      nalu_hypre_TFree(fac_interp_data -> recv_boxnum_map, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_interp_data -> identity_arrayboxes, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_interp_data -> ownboxes, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_interp_data -> own_cboxnums, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(fac_interp_data -> gnodes_comm_pkg, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fac_interp_data -> interlevel_comm, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < (fac_interp_data -> ndim); i++)
      {
         nalu_hypre_TFree(fac_interp_data -> weights[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(fac_interp_data -> weights, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(fac_interp_data, NALU_HYPRE_MEMORY_HOST);
   }
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FacSemiInterpSetup2:
 * Note that an intermediate coarse SStruct_PVector is used in interpolating
 * the interlevel communicated data (coarse data). The data in these
 * intermediate vectors will be interpolated to the fine grid.
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_FacSemiInterpSetup2( void                 *fac_interp_vdata,
                           nalu_hypre_SStructVector  *e,
                           nalu_hypre_SStructPVector *ec,
                           nalu_hypre_Index           rfactors)
{
   NALU_HYPRE_Int                 ierr = 0;

   nalu_hypre_FacSemiInterpData2 *fac_interp_data = (nalu_hypre_FacSemiInterpData2 *)fac_interp_vdata;
   NALU_HYPRE_Int                 part_fine = 1;
   NALU_HYPRE_Int                 part_crse = 0;

   nalu_hypre_CommPkg           **gnodes_comm_pkg;
   nalu_hypre_CommPkg           **interlevel_comm;
   nalu_hypre_CommInfo           *comm_info;

   nalu_hypre_SStructPVector     *recv_cvectors;
   nalu_hypre_SStructPGrid       *recv_cgrid;
   NALU_HYPRE_Int               **recv_boxnum_map;
   nalu_hypre_SStructGrid        *temp_grid;

   nalu_hypre_SStructPGrid       *pgrid;

   nalu_hypre_SStructPVector     *ef = nalu_hypre_SStructVectorPVector(e, part_fine);
   nalu_hypre_StructVector       *e_var, *s_rc, *s_cvector;

   nalu_hypre_BoxArrayArray     **identity_arrayboxes;
   nalu_hypre_BoxArrayArray     **ownboxes;

   nalu_hypre_BoxArrayArray     **send_boxes, *send_rboxes;
   NALU_HYPRE_Int              ***send_processes;
   NALU_HYPRE_Int              ***send_remote_boxnums;

   nalu_hypre_BoxArrayArray     **recv_boxes, *recv_rboxes;
   NALU_HYPRE_Int              ***recv_processes;
   NALU_HYPRE_Int              ***recv_remote_boxnums;

   nalu_hypre_BoxArray           *boxarray;
   nalu_hypre_BoxArray           *tmp_boxarray, *intersect_boxes;
   nalu_hypre_Box                 box, scaled_box;
   NALU_HYPRE_Int              ***own_cboxnums;

   nalu_hypre_BoxManager         *boxman1;
   nalu_hypre_BoxManEntry       **boxman_entries;
   NALU_HYPRE_Int                 nboxman_entries;

   NALU_HYPRE_Int                 nvars = nalu_hypre_SStructPVectorNVars(ef);
   NALU_HYPRE_Int                 vars;

   nalu_hypre_Index               zero_index, index;
   nalu_hypre_Index               ilower, iupper;
   NALU_HYPRE_Int                *num_ghost;

   NALU_HYPRE_Int                 ndim, i, j, k, fi, ci;
   NALU_HYPRE_Int                 cnt1, cnt2;
   NALU_HYPRE_Int                 proc, myproc, tot_procs;
   NALU_HYPRE_Int                 num_values;

   NALU_HYPRE_Real              **weights;
   NALU_HYPRE_Real                refine_factors_2recp[3];
   nalu_hypre_Index               refine_factors_half;

   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myproc);
   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &tot_procs);

   ndim = nalu_hypre_SStructPGridNDim(nalu_hypre_SStructPVectorPGrid(ef));
   nalu_hypre_SetIndex3(zero_index, 0, 0, 0);

   nalu_hypre_BoxInit(&box, ndim);
   nalu_hypre_BoxInit(&scaled_box, ndim);

   /*------------------------------------------------------------------------
    * Intralevel communication structures-
    * A communication pkg must be created for each StructVector. Stencils
    * are needed in creating the packages- we are assuming that the same
    * stencil pattern for each StructVector, i.e., linear interpolation for
    * each variable.
    *------------------------------------------------------------------------*/
   gnodes_comm_pkg = nalu_hypre_CTAlloc(nalu_hypre_CommPkg *,  nvars, NALU_HYPRE_MEMORY_HOST);
   for (vars = 0; vars < nvars; vars++)
   {
      e_var = nalu_hypre_SStructPVectorSVector(ec, vars);
      num_ghost = nalu_hypre_StructVectorNumGhost(e_var);

      nalu_hypre_CreateCommInfoFromNumGhost(nalu_hypre_StructVectorGrid(e_var),
                                       num_ghost, &comm_info);
      nalu_hypre_CommPkgCreate(comm_info,
                          nalu_hypre_StructVectorDataSpace(e_var),
                          nalu_hypre_StructVectorDataSpace(e_var),
                          1, NULL, 0, nalu_hypre_StructVectorComm(e_var),
                          &gnodes_comm_pkg[vars]);
      nalu_hypre_CommInfoDestroy(comm_info);
   }

   (fac_interp_data -> ndim)           = ndim;
   (fac_interp_data -> nvars)          = nvars;
   (fac_interp_data -> gnodes_comm_pkg) = gnodes_comm_pkg;
   nalu_hypre_CopyIndex(rfactors, (fac_interp_data -> stride));

   /*------------------------------------------------------------------------
    * Interlevel communication structures.
    *
    * Algorithm for identity_boxes: For each cbox on this processor, refine
    * it and intersect it with the fmap.
    *    (cbox - all coarsened fmap_intersect boxes)= identity chunks
    * for cbox.
    *
    * Algorithm for own_boxes (fullwgted boxes on this processor): For each
    * fbox, coarsen it and boxmap intersect it with cmap.
    *   (cmap_intersect boxes on myproc)= ownboxes
    * for this fbox.
    *
    * Algorithm for recv_box: For each fbox, coarsen it and boxmap intersect
    * it with cmap.
    *   (cmap_intersect boxes off_proc)= unstretched recv_boxes.
    * These boxes are stretched by one in each direction so that the ghostlayer
    * is also communicated. However, the recv_grid will consists of the
    * unstretched boxes so that overlapping does not occur.
    *--------------------------------------------------------------------------*/
   identity_arrayboxes = nalu_hypre_CTAlloc(nalu_hypre_BoxArrayArray *,  nvars, NALU_HYPRE_MEMORY_HOST);

   pgrid = nalu_hypre_SStructPVectorPGrid(ec);
   nalu_hypre_ClearIndex(index);
   for (i = 0; i < ndim; i++)
   {
      index[i] = rfactors[i] - 1;
   }

   tmp_boxarray = nalu_hypre_BoxArrayCreate(0, ndim);
   for (vars = 0; vars < nvars; vars++)
   {
      boxman1 = nalu_hypre_SStructGridBoxManager(nalu_hypre_SStructVectorGrid(e),
                                            part_fine, vars);
      boxarray = nalu_hypre_StructGridBoxes(nalu_hypre_SStructPGridSGrid(pgrid, vars));
      identity_arrayboxes[vars] = nalu_hypre_BoxArrayArrayCreate(nalu_hypre_BoxArraySize(boxarray), ndim);

      nalu_hypre_ForBoxI(ci, boxarray)
      {
         box = *nalu_hypre_BoxArrayBox(boxarray, ci);
         nalu_hypre_AppendBox(&box,
                         nalu_hypre_BoxArrayArrayBoxArray(identity_arrayboxes[vars], ci));

         nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMin(&box), zero_index,
                                     rfactors, nalu_hypre_BoxIMin(&scaled_box));
         nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMax(&box), index,
                                     rfactors, nalu_hypre_BoxIMax(&scaled_box));

         nalu_hypre_BoxManIntersect(boxman1, nalu_hypre_BoxIMin(&scaled_box),
                               nalu_hypre_BoxIMax(&scaled_box), &boxman_entries,
                               &nboxman_entries);

         intersect_boxes = nalu_hypre_BoxArrayCreate(0, ndim);
         for (i = 0; i < nboxman_entries; i++)
         {
            nalu_hypre_BoxManEntryGetExtents(boxman_entries[i], ilower, iupper);
            nalu_hypre_BoxSetExtents(&box, ilower, iupper);
            nalu_hypre_IntersectBoxes(&box, &scaled_box, &box);

            /* contract this refined box so that only the coarse nodes on this
               processor will be subtracted. */
            for (j = 0; j < ndim; j++)
            {
               k = nalu_hypre_BoxIMin(&box)[j] % rfactors[j];
               if (k)
               {
                  nalu_hypre_BoxIMin(&box)[j] += rfactors[j] - k;
               }
            }

            nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(&box), zero_index,
                                        rfactors, nalu_hypre_BoxIMin(&box));
            nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(&box), zero_index,
                                        rfactors, nalu_hypre_BoxIMax(&box));
            nalu_hypre_AppendBox(&box, intersect_boxes);
         }

         nalu_hypre_SubtractBoxArrays(nalu_hypre_BoxArrayArrayBoxArray(identity_arrayboxes[vars], ci),
                                 intersect_boxes, tmp_boxarray);
         nalu_hypre_MinUnionBoxes(nalu_hypre_BoxArrayArrayBoxArray(identity_arrayboxes[vars], ci));

         nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_BoxArrayDestroy(intersect_boxes);
      }
   }
   nalu_hypre_BoxArrayDestroy(tmp_boxarray);
   fac_interp_data -> identity_arrayboxes = identity_arrayboxes;

   /*--------------------------------------------------------------------------
    * fboxes are coarsened. For each coarsened fbox, we need a boxarray of
    * recvboxes or ownboxes.
    *--------------------------------------------------------------------------*/
   ownboxes = nalu_hypre_CTAlloc(nalu_hypre_BoxArrayArray *,  nvars, NALU_HYPRE_MEMORY_HOST);
   own_cboxnums = nalu_hypre_CTAlloc(NALU_HYPRE_Int **,  nvars, NALU_HYPRE_MEMORY_HOST);

   recv_boxes = nalu_hypre_CTAlloc(nalu_hypre_BoxArrayArray *,  nvars, NALU_HYPRE_MEMORY_HOST);
   recv_processes = nalu_hypre_CTAlloc(NALU_HYPRE_Int **,  nvars, NALU_HYPRE_MEMORY_HOST);

   /* dummy pointer for CommInfoCreate */
   recv_remote_boxnums = nalu_hypre_CTAlloc(NALU_HYPRE_Int **,  nvars, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ClearIndex(index);
   for (i = 0; i < ndim; i++)
   {
      index[i] = 1;
   }

   for (vars = 0; vars < nvars; vars++)
   {
      boxman1 = nalu_hypre_SStructGridBoxManager(nalu_hypre_SStructVectorGrid(e),
                                            part_crse, vars);
      pgrid = nalu_hypre_SStructPVectorPGrid(ef);
      boxarray = nalu_hypre_StructGridBoxes(nalu_hypre_SStructPGridSGrid(pgrid, vars));

      ownboxes[vars] = nalu_hypre_BoxArrayArrayCreate(nalu_hypre_BoxArraySize(boxarray), ndim);
      own_cboxnums[vars] = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(boxarray), NALU_HYPRE_MEMORY_HOST);
      recv_boxes[vars]    = nalu_hypre_BoxArrayArrayCreate(nalu_hypre_BoxArraySize(boxarray), ndim);
      recv_processes[vars] = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(boxarray), NALU_HYPRE_MEMORY_HOST);
      recv_remote_boxnums[vars] = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(boxarray),
                                                NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_ForBoxI(fi, boxarray)
      {
         box = *nalu_hypre_BoxArrayBox(boxarray, fi);

         /*--------------------------------------------------------------------
          * Adjust this box so that only the coarse nodes inside the fine box
          * are extracted.
          *--------------------------------------------------------------------*/
         for (j = 0; j < ndim; j++)
         {
            k = nalu_hypre_BoxIMin(&box)[j] % rfactors[j];
            if (k)
            {
               nalu_hypre_BoxIMin(&box)[j] += rfactors[j] - k;
            }
         }

         nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(&box), zero_index,
                                     rfactors, nalu_hypre_BoxIMin(&scaled_box));
         nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(&box), zero_index,
                                     rfactors, nalu_hypre_BoxIMax(&scaled_box));

         nalu_hypre_BoxManIntersect(boxman1, nalu_hypre_BoxIMin(&scaled_box),
                               nalu_hypre_BoxIMax(&scaled_box), &boxman_entries, &nboxman_entries);

         cnt1 = 0; cnt2 = 0;
         for (i = 0; i < nboxman_entries; i++)
         {
            nalu_hypre_SStructBoxManEntryGetProcess(boxman_entries[i], &proc);
            if (proc == myproc)
            {
               cnt1++;
            }
            else
            {
               cnt2++;
            }
         }

         own_cboxnums[vars][fi]  = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt1, NALU_HYPRE_MEMORY_HOST);
         recv_processes[vars][fi] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt2, NALU_HYPRE_MEMORY_HOST);
         recv_remote_boxnums[vars][fi] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt2, NALU_HYPRE_MEMORY_HOST);

         cnt1 = 0; cnt2 = 0;
         for (i = 0; i < nboxman_entries; i++)
         {
            nalu_hypre_BoxManEntryGetExtents(boxman_entries[i], ilower, iupper);
            nalu_hypre_BoxSetExtents(&box, ilower, iupper);
            nalu_hypre_IntersectBoxes(&box, &scaled_box, &box);

            nalu_hypre_SStructBoxManEntryGetProcess(boxman_entries[i], &proc);
            if (proc == myproc)
            {
               nalu_hypre_AppendBox(&box,
                               nalu_hypre_BoxArrayArrayBoxArray(ownboxes[vars], fi));
               nalu_hypre_SStructBoxManEntryGetBoxnum(boxman_entries[i],
                                                 &own_cboxnums[vars][fi][cnt1]);
               cnt1++;
            }
            else
            {
               /* extend the box so all the required data for interpolation is recvd. */
               nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&box), index, 3,
                                     nalu_hypre_BoxIMin(&box));
               nalu_hypre_AddIndexes(nalu_hypre_BoxIMax(&box), index, 3, nalu_hypre_BoxIMax(&box));

               nalu_hypre_AppendBox(&box,
                               nalu_hypre_BoxArrayArrayBoxArray(recv_boxes[vars], fi));
               recv_processes[vars][fi][cnt2] = proc;
               cnt2++;
            }
         }
         nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);
      }  /* nalu_hypre_ForBoxI(fi, boxarray) */
   }     /* for (vars= 0; vars< nvars; vars++) */

   (fac_interp_data -> ownboxes) = ownboxes;
   (fac_interp_data -> own_cboxnums) = own_cboxnums;

   /*--------------------------------------------------------------------------
    * With the recv'ed boxes form a SStructPGrid and a SStructGrid. The
    * SStructGrid is needed to generate a box_manager (so that a local box ordering
    * for the remote_boxnums are obtained). Record the recv_boxnum/fbox_num
    * mapping. That is, we interpolate a recv_box l to a fine box m, generally
    * l != m since the recv_grid and fgrid do not agree.
    *--------------------------------------------------------------------------*/
   NALU_HYPRE_SStructGridCreate(nalu_hypre_SStructPVectorComm(ec),
                           ndim, 1, &temp_grid);
   nalu_hypre_SStructPGridCreate(nalu_hypre_SStructPVectorComm(ec), ndim, &recv_cgrid);
   recv_boxnum_map = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nvars, NALU_HYPRE_MEMORY_HOST);

   cnt2 = 0;
   nalu_hypre_ClearIndex(index);
   for (i = 0; i < ndim; i++)
   {
      index[i] = 1;
   }
   for (vars = 0; vars < nvars; vars++)
   {
      cnt1 = 0;
      nalu_hypre_ForBoxArrayI(i, recv_boxes[vars])
      {
         boxarray = nalu_hypre_BoxArrayArrayBoxArray(recv_boxes[vars], i);
         cnt1 += nalu_hypre_BoxArraySize(boxarray);
      }
      recv_boxnum_map[vars] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt1, NALU_HYPRE_MEMORY_HOST);

      cnt1 = 0;
      nalu_hypre_ForBoxArrayI(i, recv_boxes[vars])
      {
         boxarray = nalu_hypre_BoxArrayArrayBoxArray(recv_boxes[vars], i);
         nalu_hypre_ForBoxI(j, boxarray)
         {
            box = *nalu_hypre_BoxArrayBox(boxarray, j);

            /* contract the box its actual size. */
            nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&box), index, 3, nalu_hypre_BoxIMin(&box));
            nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMax(&box), index, 3,
                                  nalu_hypre_BoxIMax(&box));

            nalu_hypre_SStructPGridSetExtents(recv_cgrid,
                                         nalu_hypre_BoxIMin(&box),
                                         nalu_hypre_BoxIMax(&box));

            NALU_HYPRE_SStructGridSetExtents(temp_grid, 0,
                                        nalu_hypre_BoxIMin(&box),
                                        nalu_hypre_BoxIMax(&box));

            recv_boxnum_map[vars][cnt1] = i; /* record the fbox num. i */
            cnt1++;
            cnt2++;
         }
      }
   }

   /*------------------------------------------------------------------------
    * When there are no boxes to communicate, set the temp_grid to have a
    * box of size zero. This is needed so that this SStructGrid can be
    * assembled. This is done only when this only one processor.
    *------------------------------------------------------------------------*/
   if (cnt2 == 0)
   {
      /* min_index > max_index so that the box has volume zero. */
      nalu_hypre_BoxSetExtents(&box, index, zero_index);
      nalu_hypre_SStructPGridSetExtents(recv_cgrid,
                                   nalu_hypre_BoxIMin(&box),
                                   nalu_hypre_BoxIMax(&box));

      NALU_HYPRE_SStructGridSetExtents(temp_grid, 0,
                                  nalu_hypre_BoxIMin(&box),
                                  nalu_hypre_BoxIMax(&box));
   }

   NALU_HYPRE_SStructGridSetVariables(temp_grid, 0,
                                 nalu_hypre_SStructPGridNVars(pgrid),
                                 nalu_hypre_SStructPGridVarTypes(pgrid));
   NALU_HYPRE_SStructGridAssemble(temp_grid);
   nalu_hypre_SStructPGridSetVariables(recv_cgrid, nvars,
                                  nalu_hypre_SStructPGridVarTypes(pgrid) );
   nalu_hypre_SStructPGridAssemble(recv_cgrid);

   nalu_hypre_SStructPVectorCreate(nalu_hypre_SStructPGridComm(recv_cgrid), recv_cgrid,
                              &recv_cvectors);
   nalu_hypre_SStructPVectorInitialize(recv_cvectors);
   nalu_hypre_SStructPVectorAssemble(recv_cvectors);

   fac_interp_data -> recv_cvectors  = recv_cvectors;
   fac_interp_data -> recv_boxnum_map = recv_boxnum_map;

   /* pgrid recv_cgrid no longer needed. */
   nalu_hypre_SStructPGridDestroy(recv_cgrid);

   /*------------------------------------------------------------------------
    * Send_boxes.
    * Algorithm for send_boxes: For each cbox on this processor, box_map
    * intersect it with temp_grid's map.
    *   (intersection boxes off-proc)= send_boxes for this cbox.
    * Note that the send_boxes will be stretched to include the ghostlayers.
    * This guarantees that all the data required for linear interpolation
    * will be on the processor. Also, note that the remote_boxnums are
    * with respect to the recv_cgrid box numbering.
    *--------------------------------------------------------------------------*/
   send_boxes = nalu_hypre_CTAlloc(nalu_hypre_BoxArrayArray *,  nvars, NALU_HYPRE_MEMORY_HOST);
   send_processes = nalu_hypre_CTAlloc(NALU_HYPRE_Int **,  nvars, NALU_HYPRE_MEMORY_HOST);
   send_remote_boxnums = nalu_hypre_CTAlloc(NALU_HYPRE_Int **,  nvars, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ClearIndex(index);
   for (i = 0; i < ndim; i++)
   {
      index[i] = 1;
   }
   for (vars = 0; vars < nvars; vars++)
   {
      /*-------------------------------------------------------------------
       * send boxes: intersect with temp_grid that has all the recv boxes-
       * These local box_nums may not be the same as the local box_nums of
       * the coarse grid.
       *-------------------------------------------------------------------*/
      boxman1 = nalu_hypre_SStructGridBoxManager(temp_grid, 0, vars);
      pgrid = nalu_hypre_SStructPVectorPGrid(ec);
      boxarray = nalu_hypre_StructGridBoxes(nalu_hypre_SStructPGridSGrid(pgrid, vars));

      send_boxes[vars] = nalu_hypre_BoxArrayArrayCreate(nalu_hypre_BoxArraySize(boxarray), ndim);
      send_processes[vars] = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(boxarray), NALU_HYPRE_MEMORY_HOST);
      send_remote_boxnums[vars] = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(boxarray),
                                                NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_ForBoxI(ci, boxarray)
      {
         box = *nalu_hypre_BoxArrayBox(boxarray, ci);
         nalu_hypre_BoxSetExtents(&scaled_box, nalu_hypre_BoxIMin(&box), nalu_hypre_BoxIMax(&box));

         nalu_hypre_BoxManIntersect(boxman1, nalu_hypre_BoxIMin(&scaled_box),
                               nalu_hypre_BoxIMax(&scaled_box), &boxman_entries, &nboxman_entries);

         cnt1 = 0;
         for (i = 0; i < nboxman_entries; i++)
         {
            nalu_hypre_SStructBoxManEntryGetProcess(boxman_entries[i], &proc);
            if (proc != myproc)
            {
               cnt1++;
            }
         }
         send_processes[vars][ci]     = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt1, NALU_HYPRE_MEMORY_HOST);
         send_remote_boxnums[vars][ci] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt1, NALU_HYPRE_MEMORY_HOST);

         cnt1 = 0;
         for (i = 0; i < nboxman_entries; i++)
         {
            nalu_hypre_BoxManEntryGetExtents(boxman_entries[i], ilower, iupper);
            nalu_hypre_BoxSetExtents(&box, ilower, iupper);
            nalu_hypre_IntersectBoxes(&box, &scaled_box, &box);

            nalu_hypre_SStructBoxManEntryGetProcess(boxman_entries[i], &proc);
            if (proc != myproc)
            {
               /* strech the box */
               nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&box), index, 3,
                                     nalu_hypre_BoxIMin(&box));
               nalu_hypre_AddIndexes(nalu_hypre_BoxIMax(&box), index, 3, nalu_hypre_BoxIMax(&box));

               nalu_hypre_AppendBox(&box,
                               nalu_hypre_BoxArrayArrayBoxArray(send_boxes[vars], ci));

               send_processes[vars][ci][cnt1] = proc;
               nalu_hypre_SStructBoxManEntryGetBoxnum(
                  boxman_entries[i], &send_remote_boxnums[vars][ci][cnt1]);
               cnt1++;
            }
         }

         nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);
      }  /* nalu_hypre_ForBoxI(ci, boxarray) */
   }    /* for (vars= 0; vars< nvars; vars++) */

   /*--------------------------------------------------------------------------
    * Can disgard temp_grid now- only needed it's box_man info,
    *--------------------------------------------------------------------------*/
   NALU_HYPRE_SStructGridDestroy(temp_grid);

   /*--------------------------------------------------------------------------
    * Can create the interlevel_comm.
    *--------------------------------------------------------------------------*/
   interlevel_comm = nalu_hypre_CTAlloc(nalu_hypre_CommPkg *,  nvars, NALU_HYPRE_MEMORY_HOST);

   num_values = 1;
   for (vars = 0; vars < nvars; vars++)
   {
      s_rc = nalu_hypre_SStructPVectorSVector(ec, vars);

      s_cvector = nalu_hypre_SStructPVectorSVector(recv_cvectors, vars);
      send_rboxes = nalu_hypre_BoxArrayArrayDuplicate(send_boxes[vars]);
      recv_rboxes = nalu_hypre_BoxArrayArrayDuplicate(recv_boxes[vars]);

      nalu_hypre_CommInfoCreate(send_boxes[vars], recv_boxes[vars],
                           send_processes[vars], recv_processes[vars],
                           send_remote_boxnums[vars], recv_remote_boxnums[vars],
                           send_rboxes, recv_rboxes, 1, &comm_info);

      nalu_hypre_CommPkgCreate(comm_info,
                          nalu_hypre_StructVectorDataSpace(s_rc),
                          nalu_hypre_StructVectorDataSpace(s_cvector),
                          num_values, NULL, 0,
                          nalu_hypre_StructVectorComm(s_rc),
                          &interlevel_comm[vars]);
      nalu_hypre_CommInfoDestroy(comm_info);
   }
   nalu_hypre_TFree(send_boxes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_boxes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_processes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_processes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_remote_boxnums, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_remote_boxnums, NALU_HYPRE_MEMORY_HOST);

   (fac_interp_data -> interlevel_comm) = interlevel_comm;

   /* interpolation weights */
   weights = nalu_hypre_TAlloc(NALU_HYPRE_Real *,  ndim, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < ndim; i++)
   {
      weights[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  rfactors[i] + 1, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_ClearIndex(refine_factors_half);
   /*   nalu_hypre_ClearIndex(refine_factors_2recp);*/
   for (i = 0; i < ndim; i++)
   {
      refine_factors_half[i] = rfactors[i] / 2;
      refine_factors_2recp[i] = 1.0 / (2.0 * rfactors[i]);
   }

   for (i = 0; i < ndim; i++)
   {
      for (j = 0; j <= refine_factors_half[i]; j++)
      {
         weights[i][j] = refine_factors_2recp[i] * (rfactors[i] + 2 * j - 1.0);
      }

      for (j = (refine_factors_half[i] + 1); j <= rfactors[i]; j++)
      {
         weights[i][j] = refine_factors_2recp[i] * (2 * j - rfactors[i] - 1.0);
      }
   }
   (fac_interp_data -> weights) = weights;


   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_FAC_IdentityInterp2(void                 *  fac_interp_vdata,
                          nalu_hypre_SStructPVector *  xc,
                          nalu_hypre_SStructVector  *  e)
{
   nalu_hypre_FacSemiInterpData2 *interp_data = (nalu_hypre_FacSemiInterpData2 *)fac_interp_vdata;
   nalu_hypre_BoxArrayArray     **identity_boxes = interp_data-> identity_arrayboxes;

   NALU_HYPRE_Int               part_crse = 0;

   NALU_HYPRE_Int               ierr     = 0;

   /*-----------------------------------------------------------------------
    * Compute e at coarse points (injection).
    * The pgrid of xc is the same as the part_csre pgrid of e.
    *-----------------------------------------------------------------------*/
   nalu_hypre_SStructPartialPCopy(xc,
                             nalu_hypre_SStructVectorPVector(e, part_crse),
                             identity_boxes);

   return ierr;
}

/*-------------------------------------------------------------------------
 * Linear interpolation. Interpolate the vector first by interpolating the
 * values in ownboxes and then values in recv_cvectors (the interlevel
 * communicated data).
 *-------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_FAC_WeightedInterp2(void                  *fac_interp_vdata,
                          nalu_hypre_SStructPVector  *xc,
                          nalu_hypre_SStructVector   *e_parts)
{
   NALU_HYPRE_Int ierr = 0;

   nalu_hypre_FacSemiInterpData2 *interp_data = (nalu_hypre_FacSemiInterpData2 *)fac_interp_vdata;

   nalu_hypre_CommPkg          **comm_pkg       = interp_data-> gnodes_comm_pkg;
   nalu_hypre_CommPkg          **interlevel_comm = interp_data-> interlevel_comm;
   nalu_hypre_SStructPVector    *recv_cvectors  = interp_data-> recv_cvectors;
   NALU_HYPRE_Int              **recv_boxnum_map = interp_data-> recv_boxnum_map;
   nalu_hypre_BoxArrayArray    **ownboxes       = interp_data-> ownboxes;
   NALU_HYPRE_Int             ***own_cboxnums   = interp_data-> own_cboxnums;
   NALU_HYPRE_Real             **weights        = interp_data-> weights;
   NALU_HYPRE_Int                ndim           = interp_data-> ndim;

   nalu_hypre_CommHandle       *comm_handle;

   nalu_hypre_IndexRef          stride;  /* refinement factors */

   nalu_hypre_SStructPVector   *e;

   nalu_hypre_StructGrid       *fgrid;
   nalu_hypre_BoxArray         *fgrid_boxes;
   nalu_hypre_Box              *fbox;
   nalu_hypre_BoxArrayArray    *own_cboxes;
   nalu_hypre_BoxArray         *own_abox;
   nalu_hypre_Box              *ownbox;
   NALU_HYPRE_Int             **var_boxnums;
   NALU_HYPRE_Int              *cboxnums;

   nalu_hypre_Box              *xc_dbox;
   nalu_hypre_Box              *e_dbox;

   nalu_hypre_Box               refined_box, intersect_box;


   nalu_hypre_StructVector     *xc_var;
   nalu_hypre_StructVector     *e_var;
   nalu_hypre_StructVector     *recv_var;

   NALU_HYPRE_Real           ***xcp;
   NALU_HYPRE_Real           ***ep;

   nalu_hypre_Index             loop_size, lindex;
   nalu_hypre_Index             start, start_offset;
   nalu_hypre_Index             startc;
   nalu_hypre_Index             stridec;
   nalu_hypre_Index             refine_factors;
   nalu_hypre_Index             refine_factors_half;
   nalu_hypre_Index             intersect_size;
   nalu_hypre_Index             zero_index, temp_index1, temp_index2;

   NALU_HYPRE_Int               fi, bi;
   NALU_HYPRE_Int               nvars, var;

   NALU_HYPRE_Int               i, j, k, offset_ip1, offset_jp1, offset_kp1;
   NALU_HYPRE_Int               ishift, jshift, kshift;
   NALU_HYPRE_Int               ptr_ishift, ptr_jshift, ptr_kshift;
   NALU_HYPRE_Int               imax, jmax, kmax;
   NALU_HYPRE_Int               jsize, ksize;

   NALU_HYPRE_Int               part_fine = 1;

   NALU_HYPRE_Real              xweight1, xweight2;
   NALU_HYPRE_Real              yweight1, yweight2;
   NALU_HYPRE_Real              zweight1, zweight2;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   nalu_hypre_BoxInit(&refined_box, ndim);
   nalu_hypre_BoxInit(&intersect_box, ndim);

   stride        = (interp_data -> stride);

   nalu_hypre_SetIndex3(zero_index, 0, 0, 0);
   nalu_hypre_CopyIndex(stride, refine_factors);
   for (i = ndim; i < 3; i++)
   {
      refine_factors[i] = 1;
   }
   nalu_hypre_SetIndex3(stridec, 1, 1, 1);
   for (i = 0; i < ndim; i++)
   {
      refine_factors_half[i] = refine_factors[i] / 2;
   }

   /*-----------------------------------------------------------------------
    * Compute e in the refined patch. But first communicate the coarse
    * data. Will need a ghostlayer communication on the given level and an
    * interlevel communication between levels.
    *-----------------------------------------------------------------------*/
   nvars =  nalu_hypre_SStructPVectorNVars(xc);
   for (var = 0; var < nvars; var++)
   {
      xc_var = nalu_hypre_SStructPVectorSVector(xc, var);
      nalu_hypre_InitializeCommunication(comm_pkg[var],
                                    nalu_hypre_StructVectorData(xc_var),
                                    nalu_hypre_StructVectorData(xc_var), 0, 0,
                                    &comm_handle);
      nalu_hypre_FinalizeCommunication(comm_handle);

      if (recv_cvectors != NULL)
      {
         recv_var = nalu_hypre_SStructPVectorSVector(recv_cvectors, var);
         nalu_hypre_InitializeCommunication(interlevel_comm[var],
                                       nalu_hypre_StructVectorData(xc_var),
                                       nalu_hypre_StructVectorData(recv_var), 0, 0,
                                       &comm_handle);
         nalu_hypre_FinalizeCommunication(comm_handle);
      }
   }

   e =  nalu_hypre_SStructVectorPVector(e_parts, part_fine);

   /*-----------------------------------------------------------------------
    * Allocate memory for the data pointers. Assuming linear interpolation.
    * We stride through the refinement patch by the refinement factors, and
    * so we must have pointers to the intermediate fine nodes=> ep will
    * be size refine_factors[2]*refine_factors[1]. This holds for all
    * dimensions since refine_factors[i]= 1 for i>= ndim.
    * Note that we need 3 coarse nodes per coordinate direction for the
    * interpolating. This is dimensional dependent:
    *   ndim= 3     kplane= 0,1,2 & jplane= 0,1,2    **ptr size [3][3]
    *   ndim= 2     kplane= 0     & jplane= 0,1,2    **ptr size [1][3]
    *   ndim= 1     kplane= 0     & jplane= 0        **ptr size [1][1]
    *-----------------------------------------------------------------------*/
   ksize = 3;
   jsize = 3;
   if (ndim < 3)
   {
      ksize = 1;
   }
   if (ndim < 2)
   {
      jsize = 1;
   }

   xcp  = nalu_hypre_TAlloc(NALU_HYPRE_Real **,  ksize, NALU_HYPRE_MEMORY_HOST);
   ep   = nalu_hypre_TAlloc(NALU_HYPRE_Real **,  refine_factors[2], NALU_HYPRE_MEMORY_HOST);

   for (k = 0; k < refine_factors[2]; k++)
   {
      ep[k] = nalu_hypre_TAlloc(NALU_HYPRE_Real *,  refine_factors[1], NALU_HYPRE_MEMORY_HOST);
   }

   for (k = 0; k < ksize; k++)
   {
      xcp[k] = nalu_hypre_TAlloc(NALU_HYPRE_Real *,  jsize, NALU_HYPRE_MEMORY_HOST);
   }

   for (var = 0; var < nvars; var++)
   {
      xc_var = nalu_hypre_SStructPVectorSVector(xc, var);
      e_var = nalu_hypre_SStructPVectorSVector(e, var);

      fgrid      = nalu_hypre_StructVectorGrid(e_var);
      fgrid_boxes = nalu_hypre_StructGridBoxes(fgrid);

      own_cboxes = ownboxes[var];
      var_boxnums = own_cboxnums[var];

      /*--------------------------------------------------------------------
       * Interpolate the own_box coarse grid values.
       *--------------------------------------------------------------------*/
      nalu_hypre_ForBoxI(fi, fgrid_boxes)
      {
         fbox = nalu_hypre_BoxArrayBox(fgrid_boxes, fi);

         e_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(e_var), fi);
         own_abox = nalu_hypre_BoxArrayArrayBoxArray(own_cboxes, fi);
         cboxnums = var_boxnums[fi];

         /*--------------------------------------------------------------------
          * Get the ptrs for the fine struct_vectors.
          *--------------------------------------------------------------------*/
         for (k = 0; k < refine_factors[2]; k++)
         {
            for (j = 0; j < refine_factors[1]; j++)
            {
               nalu_hypre_SetIndex3(temp_index1, 0, j, k);
               ep[k][j] = nalu_hypre_StructVectorBoxData(e_var, fi) +
                          nalu_hypre_BoxOffsetDistance(e_dbox, temp_index1);
            }
         }

         nalu_hypre_ForBoxI(bi, own_abox)
         {
            ownbox = nalu_hypre_BoxArrayBox(own_abox, bi);
            nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMin(ownbox), zero_index,
                                        refine_factors, nalu_hypre_BoxIMin(&refined_box));
            nalu_hypre_ClearIndex(temp_index1);
            for (j = 0; j < ndim; j++)
            {
               temp_index1[j] = refine_factors[j] - 1;
            }
            nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMax(ownbox), temp_index1,
                                        refine_factors, nalu_hypre_BoxIMax(&refined_box));
            nalu_hypre_IntersectBoxes(fbox, &refined_box, &intersect_box);

            xc_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(xc_var),
                                        cboxnums[bi]);

            /*-----------------------------------------------------------------
             * Get ptrs for the crse struct_vectors. For linear interpolation
             * and arbitrary refinement factors, we need to point to the correct
             * coarse grid nodes. Note that the ownboxes were created so that
             * only the coarse nodes inside a fbox are contained in ownbox.
             * Since we loop over the fine intersect box, we need to refine
             * ownbox.
             *-----------------------------------------------------------------*/
            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&intersect_box), start);
            nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(&intersect_box), intersect_size);
            for (i = 0; i < 3; i++)
            {
               intersect_size[i] -= (start[i] - 1);
            }

            /*------------------------------------------------------------------
             * The fine intersection box may not be divisible by the refinement
             * factor. This means that the interpolated coarse nodes and their
             * wieghts must be carefully determined. We accomplish this using the
             * offset away from a fine index that is divisible by the factor.
             * Because the ownboxes were created so that only coarse nodes
             * completely in the fbox are included, start is always divisible
             * by refine_factors. We do the calculation anyways for future changes.
             *------------------------------------------------------------------*/
            nalu_hypre_ClearIndex(start_offset);
            for (i = 0; i < ndim; i++)
            {
               start_offset[i] = start[i] % refine_factors[i];
            }

            ptr_kshift = 0;
            if ( (start[2] % refine_factors[2] < refine_factors_half[2]) && ndim == 3 )
            {
               ptr_kshift = -1;
            }

            ptr_jshift = 0;
            if ( start[1] % refine_factors[1] < refine_factors_half[1] && ndim >= 2 )
            {
               ptr_jshift = -1;
            }

            ptr_ishift = 0;
            if ( start[0] % refine_factors[0] < refine_factors_half[0] )
            {
               ptr_ishift = -1;
            }

            for (k = 0; k < ksize; k++)
            {
               for (j = 0; j < jsize; j++)
               {
                  nalu_hypre_SetIndex3(temp_index2, ptr_ishift, j + ptr_jshift, k + ptr_kshift);
                  xcp[k][j] = nalu_hypre_StructVectorBoxData(xc_var, cboxnums[bi]) +
                              nalu_hypre_BoxOffsetDistance(xc_dbox, temp_index2);
               }
            }

            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(ownbox), startc);
            nalu_hypre_BoxGetSize(ownbox, loop_size);

            nalu_hypre_SerialBoxLoop2Begin(ndim, loop_size,
                                      e_dbox,  start,  stride,  ei,
                                      xc_dbox, startc, stridec, xci);
            {
               /*--------------------------------------------------------
                * Linear interpolation. Determine the weights and the
                * correct coarse grid values to be weighted. All fine
                * values in an agglomerated coarse cell or in the remainder
                * agglomerated coarse cells are determined. The upper
                * extents are needed.
                *--------------------------------------------------------*/
               zypre_BoxLoopGetIndex(lindex);
               imax = nalu_hypre_min( (intersect_size[0] - lindex[0] * stride[0]),
                                 refine_factors[0] );
               jmax = nalu_hypre_min( (intersect_size[1] - lindex[1] * stride[1]),
                                 refine_factors[1]);
               kmax = nalu_hypre_min( (intersect_size[2] - lindex[2] * stride[2]),
                                 refine_factors[2]);

               for (k = 0; k < kmax; k++)
               {
                  if (ndim == 3)
                  {
                     offset_kp1 = start_offset[2] + k + 1;

                     if (ptr_kshift == -1)
                     {
                        if (offset_kp1 <= refine_factors_half[2])
                        {
                           zweight2 = weights[2][offset_kp1];
                           kshift = 0;
                        }
                        else
                        {
                           kshift = 1;
                           if (offset_kp1 >  refine_factors_half[2] &&
                               offset_kp1 <= refine_factors[2])
                           {
                              zweight2 = weights[2][offset_kp1];
                           }
                           else
                           {
                              zweight2 = weights[2][offset_kp1 - refine_factors[2]];
                           }
                        }
                        zweight1 = 1.0 - zweight2;
                     }

                     else
                     {
                        if (offset_kp1 > refine_factors_half[2] &&
                            offset_kp1 <= refine_factors[2])
                        {
                           zweight2 = weights[2][offset_kp1];
                           kshift = 0;
                        }
                        else
                        {
                           kshift = 0;
                           offset_kp1 -= refine_factors[2];
                           if (offset_kp1 > 0 && offset_kp1 <= refine_factors_half[2])
                           {
                              zweight2 = weights[2][offset_kp1];
                           }
                           else
                           {
                              zweight2 = weights[2][offset_kp1];
                              kshift  = 1;
                           }
                        }
                        zweight1 = 1.0 - zweight2;
                     }
                  }     /* if (ndim == 3) */

                  for (j = 0; j < jmax; j++)
                  {
                     if (ndim >= 2)
                     {
                        offset_jp1 = start_offset[1] + j + 1;

                        if (ptr_jshift == -1)
                        {
                           if (offset_jp1 <= refine_factors_half[1])
                           {
                              yweight2 = weights[1][offset_jp1];
                              jshift = 0;
                           }
                           else
                           {
                              jshift = 1;
                              if (offset_jp1 >  refine_factors_half[1] &&
                                  offset_jp1 <= refine_factors[1])
                              {
                                 yweight2 = weights[1][offset_jp1];
                              }
                              else
                              {
                                 yweight2 = weights[1][offset_jp1 - refine_factors[1]];
                              }
                           }
                           yweight1 = 1.0 - yweight2;
                        }

                        else
                        {
                           if (offset_jp1 > refine_factors_half[1] &&
                               offset_jp1 <= refine_factors[1])
                           {
                              yweight2 = weights[1][offset_jp1];
                              jshift = 0;
                           }
                           else
                           {
                              jshift = 0;
                              offset_jp1 -= refine_factors[1];
                              if (offset_jp1 > 0 && offset_jp1 <= refine_factors_half[1])
                              {
                                 yweight2 = weights[1][offset_jp1];
                              }
                              else
                              {
                                 yweight2 = weights[1][offset_jp1];
                                 jshift  = 1;
                              }
                           }
                           yweight1 = 1.0 - yweight2;
                        }
                     }     /* if (ndim >= 2) */

                     for (i = 0; i < imax; i++)
                     {
                        offset_ip1 = start_offset[0] + i + 1;

                        if (ptr_ishift == -1)
                        {
                           if (offset_ip1 <= refine_factors_half[0])
                           {
                              xweight2 = weights[0][offset_ip1];
                              ishift = 0;
                           }
                           else
                           {
                              ishift = 1;
                              if (offset_ip1 >  refine_factors_half[0] &&
                                  offset_ip1 <= refine_factors[0])
                              {
                                 xweight2 = weights[0][offset_ip1];
                              }
                              else
                              {
                                 xweight2 = weights[0][offset_ip1 - refine_factors[0]];
                              }
                           }
                           xweight1 = 1.0 - xweight2;
                        }

                        else
                        {
                           if (offset_ip1 > refine_factors_half[0] &&
                               offset_ip1 <= refine_factors[0])
                           {
                              xweight2 = weights[0][offset_ip1];
                              ishift = 0;
                           }
                           else
                           {
                              ishift = 0;
                              offset_ip1 -= refine_factors[0];
                              if (offset_ip1 > 0 && offset_ip1 <= refine_factors_half[0])
                              {
                                 xweight2 = weights[0][offset_ip1];
                              }
                              else
                              {
                                 xweight2 = weights[0][offset_ip1];
                                 ishift  = 1;
                              }
                           }
                           xweight1 = 1.0 - xweight2;
                        }

                        if (ndim == 3)
                        {
                           ep[k][j][ei + i] = zweight1 * (
                                                 yweight1 * (
                                                    xweight1 * xcp[kshift][jshift][ishift + xci] +
                                                    xweight2 * xcp[kshift][jshift][ishift + xci + 1])
                                                 + yweight2 * (
                                                    xweight1 * xcp[kshift][jshift + 1][ishift + xci] +
                                                    xweight2 * xcp[kshift][jshift + 1][ishift + xci + 1]) )
                                              + zweight2 * (
                                                 yweight1 * (
                                                    xweight1 * xcp[kshift + 1][jshift][ishift + xci] +
                                                    xweight2 * xcp[kshift + 1][jshift][ishift + xci + 1])
                                                 + yweight2 * (
                                                    xweight1 * xcp[kshift + 1][jshift + 1][ishift + xci] +
                                                    xweight2 * xcp[kshift + 1][jshift + 1][ishift + xci + 1]) );
                        }
                        else if (ndim == 2)
                        {
                           ep[0][j][ei + i] = yweight1 * (
                                                 xweight1 * xcp[0][jshift][ishift + xci] +
                                                 xweight2 * xcp[0][jshift][ishift + xci + 1]);
                           ep[0][j][ei + i] += yweight2 * (
                                                  xweight1 * xcp[0][jshift + 1][ishift + xci] +
                                                  xweight2 * xcp[0][jshift + 1][ishift + xci + 1]);
                        }
                        else
                        {
                           ep[0][0][ei + i] = xweight1 * xcp[0][0][ishift + xci] +
                                              xweight2 * xcp[0][0][ishift + xci + 1];
                        }
                     }      /* for (i= 0; i< imax; i++) */
                  }         /* for (j= 0; j< jmax; j++) */
               }            /* for (k= 0; k< kmax; k++) */
            }
            nalu_hypre_SerialBoxLoop2End(ei, xci);

         }/* nalu_hypre_ForBoxI(bi, own_abox) */
      }   /* nalu_hypre_ForBoxArray(fi, fgrid_boxes) */

      /*--------------------------------------------------------------------
       * Interpolate the off-processor coarse grid values. These are the
       * recv_cvector values. We will use the ownbox ptrs.
       * recv_vector is non-null even when it has a grid with zero-volume
       * boxes.
       *--------------------------------------------------------------------*/
      recv_var = nalu_hypre_SStructPVectorSVector(recv_cvectors, var);
      own_abox = nalu_hypre_StructGridBoxes(nalu_hypre_StructVectorGrid(recv_var));
      cboxnums = recv_boxnum_map[var];

      nalu_hypre_ForBoxI(bi, own_abox)
      {
         ownbox = nalu_hypre_BoxArrayBox(own_abox, bi);

         /*check for boxes of volume zero- i.e., recv_cvectors is really null.*/
         if (nalu_hypre_BoxVolume(ownbox))
         {
            xc_dbox = nalu_hypre_BoxArrayBox(
                         nalu_hypre_StructVectorDataSpace(recv_var), bi);

            fi = cboxnums[bi];
            fbox  = nalu_hypre_BoxArrayBox(fgrid_boxes, fi);
            e_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(e_var), fi);

            /*------------------------------------------------------------------
             * Get the ptrs for the fine struct_vectors.
             *------------------------------------------------------------------*/
            for (k = 0; k < refine_factors[2]; k++)
            {
               for (j = 0; j < refine_factors[1]; j++)
               {
                  nalu_hypre_SetIndex3(temp_index1, 0, j, k);
                  ep[k][j] = nalu_hypre_StructVectorBoxData(e_var, fi) +
                             nalu_hypre_BoxOffsetDistance(e_dbox, temp_index1);
               }
            }

            nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMin(ownbox), zero_index,
                                        refine_factors, nalu_hypre_BoxIMin(&refined_box));
            nalu_hypre_ClearIndex(temp_index1);
            for (j = 0; j < ndim; j++)
            {
               temp_index1[j] = refine_factors[j] - 1;
            }
            nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMax(ownbox), temp_index1,
                                        refine_factors, nalu_hypre_BoxIMax(&refined_box));
            nalu_hypre_IntersectBoxes(fbox, &refined_box, &intersect_box);

            /*-----------------------------------------------------------------
             * Get ptrs for the crse struct_vectors. For linear interpolation
             * and arbitrary refinement factors, we need to point to the correct
             * coarse grid nodes. Note that the ownboxes were created so that
             * only the coarse nodes inside a fbox are contained in ownbox.
             * Since we loop over the fine intersect box, we need to refine
             * ownbox.
             *-----------------------------------------------------------------*/
            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&intersect_box), start);
            nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(&intersect_box), intersect_size);
            for (i = 0; i < 3; i++)
            {
               intersect_size[i] -= (start[i] - 1);
            }

            /*------------------------------------------------------------------
             * The fine intersection box may not be divisible by the refinement
             * factor. This means that the interpolated coarse nodes and their
             * weights must be carefully determined. We accomplish this using the
             * offset away from a fine index that is divisible by the factor.
             * Because the ownboxes were created so that only coarse nodes
             * completely in the fbox are included, start is always divisible
             * by refine_factors. We do the calculation anyways for future changes.
             *------------------------------------------------------------------*/
            nalu_hypre_ClearIndex(start_offset);
            for (i = 0; i < ndim; i++)
            {
               start_offset[i] = start[i] % refine_factors[i];
            }

            ptr_kshift = 0;
            if ((start[2] % refine_factors[2] < refine_factors_half[2]) && ndim == 3)
            {
               ptr_kshift = -1;
            }

            ptr_jshift = 0;
            if ((start[1] % refine_factors[1] < refine_factors_half[1]) && ndim >= 2)
            {
               ptr_jshift = -1;
            }

            ptr_ishift = 0;
            if ( start[0] % refine_factors[0] < refine_factors_half[0] )
            {
               ptr_ishift = -1;
            }

            for (k = 0; k < ksize; k++)
            {
               for (j = 0; j < jsize; j++)
               {
                  nalu_hypre_SetIndex3(temp_index2,
                                  ptr_ishift, j + ptr_jshift, k + ptr_kshift);
                  xcp[k][j] = nalu_hypre_StructVectorBoxData(recv_var, bi) +
                              nalu_hypre_BoxOffsetDistance(xc_dbox, temp_index2);
               }
            }

            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(ownbox), startc);
            nalu_hypre_BoxGetSize(ownbox, loop_size);

            nalu_hypre_SerialBoxLoop2Begin(ndim, loop_size,
                                      e_dbox,  start,  stride,  ei,
                                      xc_dbox, startc, stridec, xci);
            {
               /*--------------------------------------------------------
                * Linear interpolation. Determine the weights and the
                * correct coarse grid values to be weighted. All fine
                * values in an agglomerated coarse cell or in the remainder
                * agglomerated coarse cells are determined. The upper
                * extents are needed.
                *--------------------------------------------------------*/
               zypre_BoxLoopGetIndex(lindex);
               imax = nalu_hypre_min( (intersect_size[0] - lindex[0] * stride[0]),
                                 refine_factors[0] );
               jmax = nalu_hypre_min( (intersect_size[1] - lindex[1] * stride[1]),
                                 refine_factors[1]);
               kmax = nalu_hypre_min( (intersect_size[2] - lindex[2] * stride[2]),
                                 refine_factors[2]);

               for (k = 0; k < kmax; k++)
               {
                  if (ndim == 3)
                  {
                     offset_kp1 = start_offset[2] + k + 1;

                     if (ptr_kshift == -1)
                     {
                        if (offset_kp1 <= refine_factors_half[2])
                        {
                           zweight2 = weights[2][offset_kp1];
                           kshift = 0;
                        }
                        else
                        {
                           kshift = 1;
                           if (offset_kp1 >  refine_factors_half[2] &&
                               offset_kp1 <= refine_factors[2])
                           {
                              zweight2 = weights[2][offset_kp1];
                           }
                           else
                           {
                              zweight2 = weights[2][offset_kp1 - refine_factors[2]];
                           }
                        }
                        zweight1 = 1.0 - zweight2;
                     }

                     else
                     {
                        if (offset_kp1 > refine_factors_half[2] &&
                            offset_kp1 <= refine_factors[2])
                        {
                           zweight2 = weights[2][offset_kp1];
                           kshift = 0;
                        }
                        else
                        {
                           kshift = 0;
                           offset_kp1 -= refine_factors[2];
                           if (offset_kp1 > 0 && offset_kp1 <= refine_factors_half[2])
                           {
                              zweight2 = weights[2][offset_kp1];
                           }
                           else
                           {
                              zweight2 = weights[2][offset_kp1];
                              kshift  = 1;
                           }
                        }
                        zweight1 = 1.0 - zweight2;
                     }
                  }     /* if (ndim == 3) */

                  for (j = 0; j < jmax; j++)
                  {
                     if (ndim >= 2)
                     {
                        offset_jp1 = start_offset[1] + j + 1;

                        if (ptr_jshift == -1)
                        {
                           if (offset_jp1 <= refine_factors_half[1])
                           {
                              yweight2 = weights[1][offset_jp1];
                              jshift = 0;
                           }
                           else
                           {
                              jshift = 1;
                              if (offset_jp1 >  refine_factors_half[1] &&
                                  offset_jp1 <= refine_factors[1])
                              {
                                 yweight2 = weights[1][offset_jp1];
                              }
                              else
                              {
                                 yweight2 = weights[1][offset_jp1 - refine_factors[1]];
                              }
                           }
                           yweight1 = 1.0 - yweight2;
                        }

                        else
                        {
                           if (offset_jp1 > refine_factors_half[1] &&
                               offset_jp1 <= refine_factors[1])
                           {
                              yweight2 = weights[1][offset_jp1];
                              jshift = 0;
                           }
                           else
                           {
                              jshift = 0;
                              offset_jp1 -= refine_factors[1];
                              if (offset_jp1 > 0 && offset_jp1 <= refine_factors_half[1])
                              {
                                 yweight2 = weights[1][offset_jp1];
                              }
                              else
                              {
                                 yweight2 = weights[1][offset_jp1];
                                 jshift  = 1;
                              }
                           }
                           yweight1 = 1.0 - yweight2;
                        }
                     }  /* if (ndim >= 2) */

                     for (i = 0; i < imax; i++)
                     {
                        offset_ip1 = start_offset[0] + i + 1;

                        if (ptr_ishift == -1)
                        {
                           if (offset_ip1 <= refine_factors_half[0])
                           {
                              xweight2 = weights[0][offset_ip1];
                              ishift = 0;
                           }
                           else
                           {
                              ishift = 1;
                              if (offset_ip1 >  refine_factors_half[0] &&
                                  offset_ip1 <= refine_factors[0])
                              {
                                 xweight2 = weights[0][offset_ip1];
                              }
                              else
                              {
                                 xweight2 = weights[0][offset_ip1 - refine_factors[0]];
                              }
                           }
                           xweight1 = 1.0 - xweight2;
                        }

                        else
                        {
                           if (offset_ip1 > refine_factors_half[0] &&
                               offset_ip1 <= refine_factors[0])
                           {
                              xweight2 = weights[0][offset_ip1];
                              ishift = 0;
                           }
                           else
                           {
                              ishift = 0;
                              offset_ip1 -= refine_factors[0];
                              if (offset_ip1 > 0 && offset_ip1 <= refine_factors_half[0])
                              {
                                 xweight2 = weights[0][offset_ip1];
                              }
                              else
                              {
                                 xweight2 = weights[0][offset_ip1];
                                 ishift  = 1;
                              }
                           }
                           xweight1 = 1.0 - xweight2;
                        }


                        if (ndim == 3)
                        {
                           ep[k][j][ei + i] = zweight1 * (
                                                 yweight1 * (
                                                    xweight1 * xcp[kshift][jshift][ishift + xci] +
                                                    xweight2 * xcp[kshift][jshift][ishift + xci + 1])
                                                 + yweight2 * (
                                                    xweight1 * xcp[kshift][jshift + 1][ishift + xci] +
                                                    xweight2 * xcp[kshift][jshift + 1][ishift + xci + 1]) )
                                              + zweight2 * (
                                                 yweight1 * (
                                                    xweight1 * xcp[kshift + 1][jshift][ishift + xci] +
                                                    xweight2 * xcp[kshift + 1][jshift][ishift + xci + 1])
                                                 + yweight2 * (
                                                    xweight1 * xcp[kshift + 1][jshift + 1][ishift + xci] +
                                                    xweight2 * xcp[kshift + 1][jshift + 1][ishift + xci + 1]) );
                        }
                        else if (ndim == 2)
                        {
                           ep[0][j][ei + i] = yweight1 * (
                                                 xweight1 * xcp[0][jshift][ishift + xci] +
                                                 xweight2 * xcp[0][jshift][ishift + xci + 1]);
                           ep[0][j][ei + i] += yweight2 * (
                                                  xweight1 * xcp[0][jshift + 1][ishift + xci] +
                                                  xweight2 * xcp[0][jshift + 1][ishift + xci + 1]);
                        }

                        else
                        {
                           ep[0][0][ei + i] = xweight1 * xcp[0][0][ishift + xci] +
                                              xweight2 * xcp[0][0][ishift + xci + 1];
                        }

                     }      /* for (i= 0; i< imax; i++) */
                  }         /* for (j= 0; j< jmax; j++) */
               }            /* for (k= 0; k< kmax; k++) */
            }
            nalu_hypre_SerialBoxLoop2End(ei, xci);

         }  /* if (nalu_hypre_BoxVolume(ownbox)) */
      }     /* nalu_hypre_ForBoxI(bi, own_abox) */
   }         /* for (var= 0; var< nvars; var++)*/

   for (k = 0; k < ksize; k++)
   {
      nalu_hypre_TFree(xcp[k], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(xcp, NALU_HYPRE_MEMORY_HOST);

   for (k = 0; k < refine_factors[2]; k++)
   {
      nalu_hypre_TFree(ep[k], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(ep, NALU_HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/
   return ierr;
}
