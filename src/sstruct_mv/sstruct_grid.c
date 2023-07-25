/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_SStructGrid class.
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"

/*==========================================================================
 * SStructVariable routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* ONLY3D for non-cell and non-node variable types */

NALU_HYPRE_Int
nalu_hypre_SStructVariableGetOffset( NALU_HYPRE_SStructVariable  vartype,
                                NALU_HYPRE_Int              ndim,
                                nalu_hypre_Index            varoffset )
{
   NALU_HYPRE_Int d;

   switch (vartype)
   {
      case NALU_HYPRE_SSTRUCT_VARIABLE_CELL:
         nalu_hypre_SetIndex(varoffset, 0);
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_NODE:
         nalu_hypre_SetIndex(varoffset, 1);
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_XFACE:
         nalu_hypre_SetIndex3(varoffset, 1, 0, 0);
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_YFACE:
         nalu_hypre_SetIndex3(varoffset, 0, 1, 0);
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_ZFACE:
         nalu_hypre_SetIndex3(varoffset, 0, 0, 1);
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_XEDGE:
         nalu_hypre_SetIndex3(varoffset, 0, 1, 1);
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_YEDGE:
         nalu_hypre_SetIndex3(varoffset, 1, 0, 1);
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_ZEDGE:
         nalu_hypre_SetIndex3(varoffset, 1, 1, 0);
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_UNDEFINED:
         break;
   }
   for (d = ndim; d < NALU_HYPRE_MAXDIM; d++)
   {
      nalu_hypre_IndexD(varoffset, d) = 0;
   }

   return nalu_hypre_error_flag;
}

/*==========================================================================
 * SStructPGrid routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPGridCreate( MPI_Comm             comm,
                          NALU_HYPRE_Int            ndim,
                          nalu_hypre_SStructPGrid **pgrid_ptr )
{
   nalu_hypre_SStructPGrid  *pgrid;
   nalu_hypre_StructGrid    *sgrid;
   NALU_HYPRE_Int            t;

   pgrid = nalu_hypre_TAlloc(nalu_hypre_SStructPGrid,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SStructPGridComm(pgrid)             = comm;
   nalu_hypre_SStructPGridNDim(pgrid)             = ndim;
   nalu_hypre_SStructPGridNVars(pgrid)            = 0;
   nalu_hypre_SStructPGridCellSGridDone(pgrid)    = 0;
   nalu_hypre_SStructPGridVarTypes(pgrid)         = NULL;

   for (t = 0; t < 8; t++)
   {
      nalu_hypre_SStructPGridVTSGrid(pgrid, t)     = NULL;
      nalu_hypre_SStructPGridVTIBoxArray(pgrid, t) = NULL;
   }
   NALU_HYPRE_StructGridCreate(comm, ndim, &sgrid);
   nalu_hypre_SStructPGridCellSGrid(pgrid) = sgrid;

   nalu_hypre_SStructPGridPNeighbors(pgrid) = nalu_hypre_BoxArrayCreate(0, ndim);
   nalu_hypre_SStructPGridPNborOffsets(pgrid) = NULL;

   nalu_hypre_SStructPGridLocalSize(pgrid)  = 0;
   nalu_hypre_SStructPGridGlobalSize(pgrid) = 0;

   /* GEC0902 ghost addition to the grid    */
   nalu_hypre_SStructPGridGhlocalSize(pgrid)   = 0;

   nalu_hypre_SetIndex(nalu_hypre_SStructPGridPeriodic(pgrid), 0);

   *pgrid_ptr = pgrid;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPGridDestroy( nalu_hypre_SStructPGrid *pgrid )
{
   nalu_hypre_StructGrid **sgrids;
   nalu_hypre_BoxArray   **iboxarrays;
   NALU_HYPRE_Int          t;

   if (pgrid)
   {
      sgrids     = nalu_hypre_SStructPGridSGrids(pgrid);
      iboxarrays = nalu_hypre_SStructPGridIBoxArrays(pgrid);
      nalu_hypre_TFree(nalu_hypre_SStructPGridVarTypes(pgrid), NALU_HYPRE_MEMORY_HOST);
      for (t = 0; t < 8; t++)
      {
         NALU_HYPRE_StructGridDestroy(sgrids[t]);
         nalu_hypre_BoxArrayDestroy(iboxarrays[t]);
      }
      nalu_hypre_BoxArrayDestroy(nalu_hypre_SStructPGridPNeighbors(pgrid));
      nalu_hypre_TFree(nalu_hypre_SStructPGridPNborOffsets(pgrid), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(pgrid, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPGridSetExtents( nalu_hypre_SStructPGrid  *pgrid,
                              nalu_hypre_Index          ilower,
                              nalu_hypre_Index          iupper )
{
   nalu_hypre_StructGrid *sgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);

   NALU_HYPRE_StructGridSetExtents(sgrid, ilower, iupper);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPGridSetCellSGrid( nalu_hypre_SStructPGrid  *pgrid,
                                nalu_hypre_StructGrid    *cell_sgrid )
{
   nalu_hypre_SStructPGridCellSGrid(pgrid) = cell_sgrid;
   nalu_hypre_SStructPGridCellSGridDone(pgrid) = 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPGridSetVariables( nalu_hypre_SStructPGrid    *pgrid,
                                NALU_HYPRE_Int              nvars,
                                NALU_HYPRE_SStructVariable *vartypes )
{
   nalu_hypre_SStructVariable  *new_vartypes;
   NALU_HYPRE_Int               i;

   nalu_hypre_TFree(nalu_hypre_SStructPGridVarTypes(pgrid), NALU_HYPRE_MEMORY_HOST);

   new_vartypes = nalu_hypre_TAlloc(nalu_hypre_SStructVariable,  nvars, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nvars; i++)
   {
      new_vartypes[i] = vartypes[i];
   }

   nalu_hypre_SStructPGridNVars(pgrid)    = nvars;
   nalu_hypre_SStructPGridVarTypes(pgrid) = new_vartypes;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPGridSetPNeighbor( nalu_hypre_SStructPGrid  *pgrid,
                                nalu_hypre_Box           *pneighbor_box,
                                nalu_hypre_Index          pnbor_offset )
{
   nalu_hypre_BoxArray  *pneighbors    = nalu_hypre_SStructPGridPNeighbors(pgrid);
   nalu_hypre_Index     *pnbor_offsets = nalu_hypre_SStructPGridPNborOffsets(pgrid);
   NALU_HYPRE_Int        size          = nalu_hypre_BoxArraySize(pneighbors);
   NALU_HYPRE_Int        memchunk      = 10;

   nalu_hypre_AppendBox(pneighbor_box, pneighbors);
   if ((size % memchunk) == 0)
   {
      pnbor_offsets = nalu_hypre_TReAlloc(pnbor_offsets,  nalu_hypre_Index,  (size + memchunk), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_SStructPGridPNborOffsets(pgrid) = pnbor_offsets;
   }
   nalu_hypre_CopyIndex(pnbor_offset, pnbor_offsets[size]);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * 11/06 AHB - modified to use the box manager
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPGridAssemble( nalu_hypre_SStructPGrid  *pgrid )
{
   MPI_Comm               comm          = nalu_hypre_SStructPGridComm(pgrid);
   NALU_HYPRE_Int              ndim          = nalu_hypre_SStructPGridNDim(pgrid);
   NALU_HYPRE_Int              nvars         = nalu_hypre_SStructPGridNVars(pgrid);
   NALU_HYPRE_SStructVariable *vartypes      = nalu_hypre_SStructPGridVarTypes(pgrid);
   nalu_hypre_StructGrid     **sgrids        = nalu_hypre_SStructPGridSGrids(pgrid);
   nalu_hypre_BoxArray       **iboxarrays    = nalu_hypre_SStructPGridIBoxArrays(pgrid);
   nalu_hypre_BoxArray        *pneighbors    = nalu_hypre_SStructPGridPNeighbors(pgrid);
   nalu_hypre_Index           *pnbor_offsets = nalu_hypre_SStructPGridPNborOffsets(pgrid);
   nalu_hypre_IndexRef         periodic      = nalu_hypre_SStructPGridPeriodic(pgrid);

   nalu_hypre_StructGrid      *cell_sgrid;
   nalu_hypre_IndexRef         cell_imax;
   nalu_hypre_StructGrid      *sgrid;
   nalu_hypre_BoxArray        *iboxarray;
   nalu_hypre_BoxManager      *boxman;
   nalu_hypre_BoxArray        *hood_boxes;
   NALU_HYPRE_Int              hood_first_local;
   NALU_HYPRE_Int              hood_num_local;
   nalu_hypre_BoxArray        *nbor_boxes;
   nalu_hypre_BoxArray        *diff_boxes;
   nalu_hypre_BoxArray        *tmp_boxes;
   nalu_hypre_BoxArray        *boxes;
   nalu_hypre_Box             *box;
   nalu_hypre_Index            varoffset;
   NALU_HYPRE_Int              pneighbors_size, vneighbors_size;

   NALU_HYPRE_Int              t, var, i, j, d, valid;

   /*-------------------------------------------------------------
    * set up the uniquely distributed sgrids for each vartype
    *-------------------------------------------------------------*/

   cell_sgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);
   NALU_HYPRE_StructGridSetPeriodic(cell_sgrid, periodic);
   if (!nalu_hypre_SStructPGridCellSGridDone(pgrid))
   {
      NALU_HYPRE_StructGridAssemble(cell_sgrid);
   }

   /* this is used to truncate boxes when periodicity is on */
   cell_imax = nalu_hypre_BoxIMax(nalu_hypre_StructGridBoundingBox(cell_sgrid));

   /* get neighbor info from the struct grid box manager */
   boxman     = nalu_hypre_StructGridBoxMan(cell_sgrid);
   hood_boxes =  nalu_hypre_BoxArrayCreate(0, ndim);
   nalu_hypre_BoxManGetAllEntriesBoxes(boxman, hood_boxes);
   hood_first_local = nalu_hypre_BoxManFirstLocal(boxman);
   hood_num_local   = nalu_hypre_BoxManNumMyEntries(boxman);

   pneighbors_size = nalu_hypre_BoxArraySize(pneighbors);

   /* Add one since hood_first_local can be -1 */
   nbor_boxes = nalu_hypre_BoxArrayCreate(
                   pneighbors_size + hood_first_local + hood_num_local + 1, ndim);
   diff_boxes = nalu_hypre_BoxArrayCreate(0, ndim);
   tmp_boxes  = nalu_hypre_BoxArrayCreate(0, ndim);

   for (var = 0; var < nvars; var++)
   {
      t = vartypes[var];

      if ((t > 0) && (sgrids[t] == NULL))
      {
         NALU_HYPRE_StructGridCreate(comm, ndim, &sgrid);
         nalu_hypre_StructGridSetNumGhost(sgrid, nalu_hypre_StructGridNumGhost(cell_sgrid));
         boxes = nalu_hypre_BoxArrayCreate(0, ndim);
         nalu_hypre_SStructVariableGetOffset((nalu_hypre_SStructVariable) t,
                                        ndim, varoffset);

         /* create nbor_boxes for this variable type */
         vneighbors_size = 0;
         for (i = 0; i < pneighbors_size; i++)
         {
            box = nalu_hypre_BoxArrayBox(nbor_boxes, vneighbors_size);
            nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(pneighbors, i), box);
            nalu_hypre_SStructCellBoxToVarBox(box, pnbor_offsets[i], varoffset, &valid);
            /* only add pneighbor boxes for valid variable types*/
            if (valid)
            {
               vneighbors_size++;
            }
         }
         for (i = 0; i < (hood_first_local + hood_num_local); i++)
         {
            box = nalu_hypre_BoxArrayBox(nbor_boxes, vneighbors_size + i);
            nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(hood_boxes, i), box);
            nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(box), varoffset,
                                  nalu_hypre_BoxNDim(box), nalu_hypre_BoxIMin(box));
         }

         /* boxes = (local boxes - neighbors with smaller ID - vneighbors) */
         for (i = 0; i < hood_num_local; i++)
         {
            j = vneighbors_size + hood_first_local + i;
            nalu_hypre_BoxArraySetSize(diff_boxes, 1);
            nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(nbor_boxes, j),
                          nalu_hypre_BoxArrayBox(diff_boxes, 0));
            nalu_hypre_BoxArraySetSize(nbor_boxes, j);

            nalu_hypre_SubtractBoxArrays(diff_boxes, nbor_boxes, tmp_boxes);
            nalu_hypre_AppendBoxArray(diff_boxes, boxes);
         }

         /* truncate if necessary when periodic */
         for (d = 0; d < ndim; d++)
         {
            if (nalu_hypre_IndexD(periodic, d) && nalu_hypre_IndexD(varoffset, d))
            {
               nalu_hypre_ForBoxI(i, boxes)
               {
                  box = nalu_hypre_BoxArrayBox(boxes, i);
                  if (nalu_hypre_BoxIMaxD(box, d) == nalu_hypre_IndexD(cell_imax, d))
                  {
                     nalu_hypre_BoxIMaxD(box, d) --;
                  }
               }
            }
         }
         NALU_HYPRE_StructGridSetPeriodic(sgrid, periodic);

         nalu_hypre_StructGridSetBoxes(sgrid, boxes);
         NALU_HYPRE_StructGridAssemble(sgrid);

         sgrids[t] = sgrid;
      }
   }

   nalu_hypre_BoxArrayDestroy(hood_boxes);

   nalu_hypre_BoxArrayDestroy(nbor_boxes);
   nalu_hypre_BoxArrayDestroy(diff_boxes);
   nalu_hypre_BoxArrayDestroy(tmp_boxes);

   /*-------------------------------------------------------------
    * compute iboxarrays
    *-------------------------------------------------------------*/

   for (t = 0; t < 8; t++)
   {
      sgrid = sgrids[t];
      if (sgrid != NULL)
      {
         iboxarray = nalu_hypre_BoxArrayDuplicate(nalu_hypre_StructGridBoxes(sgrid));

         nalu_hypre_SStructVariableGetOffset((nalu_hypre_SStructVariable) t,
                                        ndim, varoffset);
         nalu_hypre_ForBoxI(i, iboxarray)
         {
            /* grow the boxes */
            box = nalu_hypre_BoxArrayBox(iboxarray, i);
            nalu_hypre_BoxGrowByIndex(box, varoffset);
         }

         iboxarrays[t] = iboxarray;
      }
   }

   /*-------------------------------------------------------------
    * set up the size info
    * GEC0902 addition of the local ghost size for pgrid.At first pgridghlocalsize=0
    *-------------------------------------------------------------*/

   for (var = 0; var < nvars; var++)
   {
      sgrid = nalu_hypre_SStructPGridSGrid(pgrid, var);
      nalu_hypre_SStructPGridLocalSize(pgrid)  += nalu_hypre_StructGridLocalSize(sgrid);
      nalu_hypre_SStructPGridGlobalSize(pgrid) += nalu_hypre_StructGridGlobalSize(sgrid);
      nalu_hypre_SStructPGridGhlocalSize(pgrid) += nalu_hypre_StructGridGhlocalSize(sgrid);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructPGridGetMaxBoxSize( nalu_hypre_SStructPGrid *pgrid )
{
   NALU_HYPRE_Int         nvars = nalu_hypre_SStructPGridNVars(pgrid);
   NALU_HYPRE_Int         var;
   nalu_hypre_StructGrid *sgrid;
   NALU_HYPRE_Int         max_box_size = 0;

   for (var = 0; var < nvars; var++)
   {
      sgrid = nalu_hypre_SStructPGridSGrid(pgrid, var);
      max_box_size = nalu_hypre_max(max_box_size, nalu_hypre_StructGridGetMaxBoxSize(sgrid));
   }

   return max_box_size;
}

/*==========================================================================
 * SStructGrid routines
 *==========================================================================*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGridRef( nalu_hypre_SStructGrid  *grid,
                      nalu_hypre_SStructGrid **grid_ref)
{
   nalu_hypre_SStructGridRefCount(grid) ++;
   *grid_ref = grid;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This replaces nalu_hypre_SStructGridAssembleMaps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGridAssembleBoxManagers( nalu_hypre_SStructGrid *grid )
{
   MPI_Comm                   comm        = nalu_hypre_SStructGridComm(grid);
   NALU_HYPRE_Int                  ndim        = nalu_hypre_SStructGridNDim(grid);
   NALU_HYPRE_Int                  nparts      = nalu_hypre_SStructGridNParts(grid);
   NALU_HYPRE_Int                  local_size  = nalu_hypre_SStructGridLocalSize(grid);
   nalu_hypre_BoxManager        ***managers;
   nalu_hypre_SStructBoxManInfo    info_obj;
   nalu_hypre_SStructPGrid        *pgrid;
   NALU_HYPRE_Int                  nvars;
   nalu_hypre_StructGrid          *sgrid;
   nalu_hypre_Box                 *bounding_box;

   NALU_HYPRE_Int                 offsets[2];

   nalu_hypre_SStructBoxManInfo   *entry_info;

   nalu_hypre_BoxManEntry         *all_entries, *entry;
   NALU_HYPRE_Int                  num_entries;
   nalu_hypre_IndexRef             entry_imin;
   nalu_hypre_IndexRef             entry_imax;

   NALU_HYPRE_Int                  nprocs, myproc, proc;
   NALU_HYPRE_Int                  part, var, b, local_ct;

   nalu_hypre_Box                 *ghostbox, *box;
   NALU_HYPRE_Int                 * num_ghost;
   NALU_HYPRE_Int                  ghoffsets[2];
   NALU_HYPRE_Int                  ghlocal_size  = nalu_hypre_SStructGridGhlocalSize(grid);

   NALU_HYPRE_Int                  info_size;
   NALU_HYPRE_Int                  box_offset, ghbox_offset;

   /*------------------------------------------------------
    * Build box manager info for grid boxes
    *------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(comm, &nprocs);
   nalu_hypre_MPI_Comm_rank(comm, &myproc);

   /*find offset and ghost offsets */
   {
      NALU_HYPRE_Int scan_recv;

      /* offsets */

      nalu_hypre_MPI_Scan(
         &local_size, &scan_recv, 1, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_SUM, comm);
      /* first point in my range */
      offsets[0] = scan_recv - local_size;
      /* first point in next proc's range */
      offsets[1] = scan_recv;

      nalu_hypre_SStructGridStartRank(grid) = offsets[0];

      /* ghost offsets */
      nalu_hypre_MPI_Scan(
         &ghlocal_size, &scan_recv, 1, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_SUM, comm);
      /* first point in my range */
      ghoffsets[0] = scan_recv - ghlocal_size;
      /* first point in next proc's range */
      ghoffsets[1] = scan_recv;

      nalu_hypre_SStructGridGhstartRank(grid) = ghoffsets[0];
   }

   /* allocate a box manager for each part and variable -
      copy the local box info from the underlying sgrid boxmanager*/

   managers = nalu_hypre_TAlloc(nalu_hypre_BoxManager **,  nparts, NALU_HYPRE_MEMORY_HOST);

   /* first offsets */
   box_offset =  offsets[0];
   ghbox_offset =  ghoffsets[0];

   info_size = sizeof(nalu_hypre_SStructBoxManInfo);

   /* storage for the entry info is allocated and kept in the box
      manager - so here we just write over the info_obj and then
      it is copied in AddEntry */
   entry_info = &info_obj;

   /* this is the same for all the info objects */
   nalu_hypre_SStructBoxManInfoType(entry_info) = nalu_hypre_SSTRUCT_BOXMAN_INFO_DEFAULT;

   box = nalu_hypre_BoxCreate(ndim);
   ghostbox = nalu_hypre_BoxCreate(ndim);

   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);

      managers[part] = nalu_hypre_TAlloc(nalu_hypre_BoxManager *,  nvars, NALU_HYPRE_MEMORY_HOST);

      for (var = 0; var < nvars; var++)
      {
         sgrid = nalu_hypre_SStructPGridSGrid(pgrid, var);

         /* get all the entires from the sgrid. for the local boxes, we will
          * calculate the info and add to the box manager - the rest we will
          * gather (because we cannot calculate the info for them) */

         nalu_hypre_BoxManGetAllEntries(nalu_hypre_StructGridBoxMan(sgrid),
                                   &num_entries, &all_entries);

         bounding_box = nalu_hypre_StructGridBoundingBox(sgrid);

         /* need to create a box manager and then later give it the bounding box
            for gather entries call */

         nalu_hypre_BoxManCreate(
            nalu_hypre_BoxManNumMyEntries(nalu_hypre_StructGridBoxMan(sgrid)),
            info_size, nalu_hypre_StructGridNDim(sgrid), bounding_box,
            nalu_hypre_StructGridComm(sgrid), &managers[part][var]);

         /* each sgrid has num_ghost */

         num_ghost = nalu_hypre_StructGridNumGhost(sgrid);
         nalu_hypre_BoxManSetNumGhost(managers[part][var], num_ghost);

         /* loop through the all of the entries - for the local boxes
          * populate the info object and add to Box Manager- recall
          * that all of the boxes array belong to the calling proc */

         local_ct = 0;
         for (b = 0; b < num_entries; b++)
         {
            entry = &all_entries[b];

            proc = nalu_hypre_BoxManEntryProc(entry);

            entry_imin = nalu_hypre_BoxManEntryIMin(entry);
            entry_imax = nalu_hypre_BoxManEntryIMax(entry);
            nalu_hypre_BoxSetExtents( box, entry_imin, entry_imax );

            if (proc == myproc)
            {
               nalu_hypre_SStructBoxManInfoOffset(entry_info) = box_offset;
               nalu_hypre_SStructBoxManInfoGhoffset(entry_info) = ghbox_offset;
               nalu_hypre_BoxManAddEntry(managers[part][var],
                                    entry_imin, entry_imax,
                                    myproc, local_ct, entry_info);

               /* update offset */
               box_offset += nalu_hypre_BoxVolume(box);

               /* grow box to compute volume with ghost */
               nalu_hypre_CopyBox(box, ghostbox);
               nalu_hypre_BoxGrowByArray(ghostbox, num_ghost);

               /* update offset */
               ghbox_offset += nalu_hypre_BoxVolume(ghostbox);

               local_ct++;
            }
            else /* not a local box */
            {
               nalu_hypre_BoxManGatherEntries(managers[part][var],
                                         entry_imin, entry_imax);
            }
         }

         /* call the assemble later */

      } /* end of variable loop */
   } /* end of part loop */

   {
      /* need to do a gather entries on neighbor information so that we have
         what we need for the NborBoxManagers function */

      /* these neighbor boxes are much larger than the data that we care about,
         so first we need to intersect them with the grid and just pass the
         intersected box into the Box Manager */

      nalu_hypre_SStructNeighbor    *vneighbor;
      NALU_HYPRE_Int                 b, i;
      nalu_hypre_Box                *vbox;
      NALU_HYPRE_Int               **nvneighbors = nalu_hypre_SStructGridNVNeighbors(grid);
      nalu_hypre_SStructNeighbor  ***vneighbors  = nalu_hypre_SStructGridVNeighbors(grid);
      NALU_HYPRE_Int                *coord, *dir;
      nalu_hypre_Index               imin0, imin1;
      NALU_HYPRE_Int                 nbor_part, nbor_var;
      nalu_hypre_IndexRef            max_distance;
      nalu_hypre_Box                *grow_box;
      nalu_hypre_Box                *int_box;
      nalu_hypre_Box                *nbor_box;
      nalu_hypre_BoxManager         *box_man;
      nalu_hypre_BoxArray           *local_boxes;

      grow_box = nalu_hypre_BoxCreate(ndim);
      int_box = nalu_hypre_BoxCreate(ndim);
      nbor_box =  nalu_hypre_BoxCreate(ndim);

      local_boxes = nalu_hypre_BoxArrayCreate(0, ndim);

      for (part = 0; part < nparts; part++)
      {
         pgrid = nalu_hypre_SStructGridPGrid(grid, part);
         nvars = nalu_hypre_SStructPGridNVars(pgrid);

         for (var = 0; var < nvars; var++)
         {
            sgrid = nalu_hypre_SStructPGridSGrid(pgrid, var);
            max_distance = nalu_hypre_StructGridMaxDistance(sgrid);

            /* now loop through my boxes, grow them, and intersect with all of
             * the neighbors */

            box_man = nalu_hypre_StructGridBoxMan(sgrid);
            nalu_hypre_BoxManGetLocalEntriesBoxes(box_man, local_boxes);

            nalu_hypre_ForBoxI(i, local_boxes)
            {
               nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(local_boxes, i), grow_box);
               nalu_hypre_BoxGrowByIndex(grow_box, max_distance);

               /* loop through neighbors */
               for (b = 0; b < nvneighbors[part][var]; b++)
               {
                  vneighbor = &vneighbors[part][var][b];
                  vbox = nalu_hypre_SStructNeighborBox(vneighbor);

                  /* grow neighbor box by 1 to account for shared parts */
                  nalu_hypre_CopyBox(vbox, nbor_box);
                  nalu_hypre_BoxGrowByValue(nbor_box, 1);

                  nbor_part = nalu_hypre_SStructNeighborPart(vneighbor);

                  coord = nalu_hypre_SStructNeighborCoord(vneighbor);
                  dir   = nalu_hypre_SStructNeighborDir(vneighbor);

                  /* find intersection of neighbor and my local box */
                  nalu_hypre_IntersectBoxes(grow_box, nbor_box, int_box);
                  if (nalu_hypre_BoxVolume(int_box) > 0)
                  {
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(vbox), imin0);
                     nalu_hypre_CopyIndex(nalu_hypre_SStructNeighborILower(vneighbor), imin1);

                     /* map int_box to neighbor part index space */
                     nalu_hypre_SStructBoxToNborBox(int_box, imin0, imin1, coord, dir);
                     nalu_hypre_SStructVarToNborVar(grid, part, var, coord, &nbor_var);

                     nalu_hypre_BoxManGatherEntries(
                        managers[nbor_part][nbor_var],
                        nalu_hypre_BoxIMin(int_box), nalu_hypre_BoxIMax(int_box));
                  }
               } /* end neighbor loop */
            } /* end local box loop */
         }
      }
      nalu_hypre_BoxDestroy(grow_box);
      nalu_hypre_BoxDestroy(int_box);
      nalu_hypre_BoxDestroy(nbor_box);
      nalu_hypre_BoxArrayDestroy(local_boxes);
   }

   /* now call the assembles */
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);

      for (var = 0; var < nvars; var++)
      {
         nalu_hypre_BoxManAssemble(managers[part][var]);
      }
   }

   nalu_hypre_BoxDestroy(ghostbox);
   nalu_hypre_BoxDestroy(box);

   nalu_hypre_SStructGridBoxManagers(grid) = managers;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGridAssembleNborBoxManagers( nalu_hypre_SStructGrid *grid )
{
   NALU_HYPRE_Int                    ndim        = nalu_hypre_SStructGridNDim(grid);
   NALU_HYPRE_Int                    nparts      = nalu_hypre_SStructGridNParts(grid);
   NALU_HYPRE_Int                  **nvneighbors = nalu_hypre_SStructGridNVNeighbors(grid);
   nalu_hypre_SStructNeighbor     ***vneighbors  = nalu_hypre_SStructGridVNeighbors(grid);
   nalu_hypre_SStructNeighbor       *vneighbor;
   nalu_hypre_SStructPGrid          *pgrid;
   NALU_HYPRE_Int                    nvars;
   nalu_hypre_StructGrid            *sgrid;

   nalu_hypre_BoxManager          ***nbor_managers;
   nalu_hypre_SStructBoxManNborInfo *nbor_info, *peri_info;
   nalu_hypre_SStructBoxManInfo     *entry_info;
   nalu_hypre_BoxManEntry          **entries, *all_entries, *entry;
   NALU_HYPRE_Int                    nentries;

   nalu_hypre_Box                   *nbor_box, *box, *int_box, *ghbox;
   NALU_HYPRE_Int                   *coord, *dir;
   nalu_hypre_Index                  imin0, imin1;
   NALU_HYPRE_BigInt                 nbor_offset, nbor_ghoffset;
   NALU_HYPRE_Int                    nbor_proc, nbor_boxnum, nbor_part, nbor_var;
   nalu_hypre_IndexRef               pshift;
   NALU_HYPRE_Int                    num_periods, k;
   NALU_HYPRE_Int                    proc;
   nalu_hypre_Index                  nbor_ilower;
   NALU_HYPRE_Int                    c[NALU_HYPRE_MAXDIM], *num_ghost, *stride, *ghstride;
   NALU_HYPRE_Int                    part, var, b, i, d, info_size;

   nalu_hypre_Box                   *bounding_box;

   /*------------------------------------------------------
    * Create a box manager for the neighbor boxes
    *------------------------------------------------------*/

   bounding_box = nalu_hypre_BoxCreate(ndim);

   nbor_box = nalu_hypre_BoxCreate(ndim);
   box = nalu_hypre_BoxCreate(ndim);
   int_box = nalu_hypre_BoxCreate(ndim);
   ghbox = nalu_hypre_BoxCreate(ndim);
   /* nbor_info is copied into the box manager */
   nbor_info = nalu_hypre_TAlloc(nalu_hypre_SStructBoxManNborInfo,  1, NALU_HYPRE_MEMORY_HOST);
   peri_info = nalu_hypre_CTAlloc(nalu_hypre_SStructBoxManNborInfo,  1, NALU_HYPRE_MEMORY_HOST);

   nbor_managers = nalu_hypre_TAlloc(nalu_hypre_BoxManager **,  nparts, NALU_HYPRE_MEMORY_HOST);

   info_size = sizeof(nalu_hypre_SStructBoxManNborInfo);

   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);

      nbor_managers[part] = nalu_hypre_TAlloc(nalu_hypre_BoxManager *,  nvars, NALU_HYPRE_MEMORY_HOST);

      for (var = 0; var < nvars; var++)
      {
         sgrid = nalu_hypre_SStructPGridSGrid(pgrid, var);
         nalu_hypre_CopyBox( nalu_hypre_StructGridBoundingBox(sgrid), bounding_box);
         /* The bounding_box is only needed if BoxManGatherEntries() is called,
          * but we don't gather anything currently for the neighbor boxman, so
          * the next bit of code is not needed right now. */
#if 0
         {
            MPI_Comm     comm        = nalu_hypre_SStructGridComm(grid);
            nalu_hypre_Box   *vbox;
            nalu_hypre_Index  min_index, max_index;
            NALU_HYPRE_Int    d;
            NALU_HYPRE_Int    sendbuf6[2 * NALU_HYPRE_MAXDIM], recvbuf6[2 * NALU_HYPRE_MAXDIM];
            nalu_hypre_CopyToCleanIndex( nalu_hypre_BoxIMin(bounding_box), ndim, min_index);
            nalu_hypre_CopyToCleanIndex( nalu_hypre_BoxIMax(bounding_box), ndim, max_index);

            for (b = 0; b < nvneighbors[part][var]; b++)
            {
               vneighbor = &vneighbors[part][var][b];
               vbox = nalu_hypre_SStructNeighborBox(vneighbor);
               /* find min and max box extents */
               for (d = 0; d < ndim; d++)
               {
                  nalu_hypre_IndexD(min_index, d) =
                     nalu_hypre_min(nalu_hypre_IndexD(min_index, d), nalu_hypre_BoxIMinD(vbox, d));
                  nalu_hypre_IndexD(max_index, d) =
                     nalu_hypre_max(nalu_hypre_IndexD(max_index, d), nalu_hypre_BoxIMaxD(vbox, d));
               }
            }
            /* this is based on local info - all procs need to have
             * the same bounding box!  */
            nalu_hypre_BoxSetExtents( bounding_box, min_index, max_index);

            /* communication needed for the bounding box */
            /* pack buffer */
            for (d = 0; d < ndim; d++)
            {
               sendbuf6[d] = nalu_hypre_BoxIMinD(bounding_box, d);
               sendbuf6[d + ndim] = -nalu_hypre_BoxIMaxD(bounding_box, d);
            }
            nalu_hypre_MPI_Allreduce(
               sendbuf6, recvbuf6, 2 * ndim, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_MIN, comm);
            /* unpack buffer */
            for (d = 0; d < ndim; d++)
            {
               nalu_hypre_BoxIMinD(bounding_box, d) = recvbuf6[d];
               nalu_hypre_BoxIMaxD(bounding_box, d) = -recvbuf6[d + ndim];
            }
         }
#endif
         /* Here we want to create a new manager for the neighbor information
          * (instead of adding to the current and reassembling).  This uses a
          * lower bound for the actual box manager size. */

         nalu_hypre_BoxManCreate(nvneighbors[part][var], info_size, ndim,
                            nalu_hypre_StructGridBoundingBox(sgrid),
                            nalu_hypre_StructGridComm(sgrid),
                            &nbor_managers[part][var]);

         /* Compute entries and add to the neighbor box manager */
         for (b = 0; b < nvneighbors[part][var]; b++)
         {
            vneighbor = &vneighbors[part][var][b];

            nalu_hypre_CopyBox(nalu_hypre_SStructNeighborBox(vneighbor), nbor_box);
            nbor_part = nalu_hypre_SStructNeighborPart(vneighbor);
            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(nalu_hypre_SStructNeighborBox(vneighbor)), imin0);
            nalu_hypre_CopyIndex(nalu_hypre_SStructNeighborILower(vneighbor), imin1);
            coord = nalu_hypre_SStructNeighborCoord(vneighbor);
            dir   = nalu_hypre_SStructNeighborDir(vneighbor);

            /* Intersect neighbor boxes with appropriate PGrid */

            /* map to neighbor part index space */
            nalu_hypre_SStructBoxToNborBox(nbor_box, imin0, imin1, coord, dir);
            nalu_hypre_SStructVarToNborVar(grid, part, var, coord, &nbor_var);

            nalu_hypre_SStructGridIntersect(grid, nbor_part, nbor_var, nbor_box, 0,
                                       &entries, &nentries);

            for (i = 0; i < nentries; i++)
            {
               nalu_hypre_BoxManEntryGetExtents(entries[i], nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box));
               nalu_hypre_IntersectBoxes(nbor_box, box, int_box);

               /* map back from neighbor part index space */
               nalu_hypre_SStructNborBoxToBox(int_box, imin0, imin1, coord, dir);

               nalu_hypre_SStructIndexToNborIndex(
                  nalu_hypre_BoxIMin(int_box), imin0, imin1, coord, dir, ndim, nbor_ilower);

               nalu_hypre_SStructBoxManEntryGetProcess(entries[i], &nbor_proc);
               nalu_hypre_SStructBoxManEntryGetBoxnum(entries[i], &nbor_boxnum);
               nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entries[i], nbor_ilower, &nbor_offset);
               nalu_hypre_SStructBoxManEntryGetGlobalGhrank(entries[i], nbor_ilower, &nbor_ghoffset);
               num_ghost = nalu_hypre_BoxManEntryNumGhost(entries[i]);

               /* Set up the neighbor info. */
               nalu_hypre_SStructBoxManInfoType(nbor_info) = nalu_hypre_SSTRUCT_BOXMAN_INFO_NEIGHBOR;
               nalu_hypre_SStructBoxManInfoOffset(nbor_info) = nbor_offset;
               nalu_hypre_SStructBoxManInfoGhoffset(nbor_info) = nbor_ghoffset;
               nalu_hypre_SStructBoxManNborInfoProc(nbor_info) = nbor_proc;
               nalu_hypre_SStructBoxManNborInfoBoxnum(nbor_info) = nbor_boxnum;
               nalu_hypre_SStructBoxManNborInfoPart(nbor_info) = nbor_part;
               nalu_hypre_CopyIndex(nbor_ilower, nalu_hypre_SStructBoxManNborInfoILower(nbor_info));
               nalu_hypre_CopyIndex(coord, nalu_hypre_SStructBoxManNborInfoCoord(nbor_info));
               nalu_hypre_CopyIndex(dir, nalu_hypre_SStructBoxManNborInfoDir(nbor_info));
               /* This computes strides in the local index-space, so they
                * may be negative.  Want `c' to map from the neighbor
                * index-space back. */
               for (d = 0; d < ndim; d++)
               {
                  c[coord[d]] = d;
               }
               nalu_hypre_CopyBox(box, ghbox);
               nalu_hypre_BoxGrowByArray(ghbox, num_ghost);
               stride   = nalu_hypre_SStructBoxManNborInfoStride(nbor_info);
               ghstride = nalu_hypre_SStructBoxManNborInfoGhstride(nbor_info);
               stride[c[0]]   = 1;
               ghstride[c[0]] = 1;
               for (d = 1; d < ndim; d++)
               {
                  stride[c[d]]   = nalu_hypre_BoxSizeD(box, d - 1)   * stride[c[d - 1]];
                  ghstride[c[d]] = nalu_hypre_BoxSizeD(ghbox, d - 1) * ghstride[c[d - 1]];
               }
               for (d = 0; d < ndim; d++)
               {
                  stride[c[d]]   *= dir[c[d]];
                  ghstride[c[d]] *= dir[c[d]];
               }

               /* Here the ids need to be unique.  Cannot use the boxnum.
                  A negative number lets the box manager assign the id. */
               nalu_hypre_BoxManAddEntry(nbor_managers[part][var],
                                    nalu_hypre_BoxIMin(int_box),
                                    nalu_hypre_BoxIMax(int_box),
                                    nbor_proc, -1, nbor_info);

            } /* end of entries loop */

            nalu_hypre_TFree(entries, NALU_HYPRE_MEMORY_HOST);

         } /* end of vneighbor box loop */

         /* RDF: Add periodic boxes to the neighbor box managers.
          *
          * Compute a local bounding box and grow by max_distance, shift the
          * boxman boxes (local and non-local to allow for periodicity of a box
          * with itself) and intersect them with the grown local bounding box.
          * If there is a nonzero intersection, add the shifted box to the
          * neighbor boxman.  The only reason for doing the intersect is to
          * reduce the number of boxes that we add. */

         num_periods = nalu_hypre_StructGridNumPeriods(sgrid);
         if ((num_periods > 1) && (nalu_hypre_StructGridNumBoxes(sgrid)))
         {
            nalu_hypre_BoxArray  *boxes = nalu_hypre_StructGridBoxes(sgrid);

            /* Compute a local bounding box */
            nalu_hypre_CopyBox(nalu_hypre_BoxArrayBox(boxes, 0), bounding_box);
            nalu_hypre_ForBoxI(i, boxes)
            {
               for (d = 0; d < nalu_hypre_StructGridNDim(sgrid); d++)
               {
                  nalu_hypre_BoxIMinD(bounding_box, d) =
                     nalu_hypre_min(nalu_hypre_BoxIMinD(bounding_box, d),
                               nalu_hypre_BoxIMinD(nalu_hypre_BoxArrayBox(boxes, i), d));
                  nalu_hypre_BoxIMaxD(bounding_box, d) =
                     nalu_hypre_max(nalu_hypre_BoxIMaxD(bounding_box, d),
                               nalu_hypre_BoxIMaxD(nalu_hypre_BoxArrayBox(boxes, i), d));
               }
            }
            /* Grow the bounding box by max_distance */
            nalu_hypre_BoxGrowByIndex(bounding_box, nalu_hypre_StructGridMaxDistance(sgrid));

            nalu_hypre_BoxManGetAllEntries(nalu_hypre_SStructGridBoxManager(grid, part, var),
                                      &nentries, &all_entries);

            for (b = 0; b < nentries; b++)
            {
               entry = &all_entries[b];

               proc = nalu_hypre_BoxManEntryProc(entry);

               nalu_hypre_BoxManEntryGetInfo(entry, (void **) &entry_info);
               nalu_hypre_SStructBoxManInfoType(peri_info) =
                  nalu_hypre_SStructBoxManInfoType(entry_info);
               nalu_hypre_SStructBoxManInfoOffset(peri_info) =
                  nalu_hypre_SStructBoxManInfoOffset(entry_info);
               nalu_hypre_SStructBoxManInfoGhoffset(peri_info) =
                  nalu_hypre_SStructBoxManInfoGhoffset(entry_info);

               for (k = 1; k < num_periods; k++) /* k = 0 is original box */
               {
                  pshift = nalu_hypre_StructGridPShift(sgrid, k);
                  nalu_hypre_BoxSetExtents(box, nalu_hypre_BoxManEntryIMin(entry),
                                      nalu_hypre_BoxManEntryIMax(entry));
                  nalu_hypre_BoxShiftPos(box, pshift);

                  nalu_hypre_IntersectBoxes(box, bounding_box, int_box);
                  if (nalu_hypre_BoxVolume(int_box) > 0)
                  {
                     nalu_hypre_BoxManAddEntry(nbor_managers[part][var],
                                          nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box),
                                          proc, -1, peri_info);
                  }
               }
            }
         }

         nalu_hypre_BoxManAssemble(nbor_managers[part][var]);

      } /* end of variables loop */

   } /* end of part loop */

   nalu_hypre_SStructGridNborBoxManagers(grid) = nbor_managers;

   nalu_hypre_TFree(nbor_info, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(peri_info, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_BoxDestroy(nbor_box);
   nalu_hypre_BoxDestroy(box);
   nalu_hypre_BoxDestroy(int_box);
   nalu_hypre_BoxDestroy(ghbox);

   nalu_hypre_BoxDestroy(bounding_box);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine computes the inter-part communication information for updating
 * shared variable data.
 *
 * It grows each local box according to vartype and intersects with the BoxManager
 * to get map entries.  Then, for each of the neighbor-type entries, it grows
 * either the local box or the neighbor box based on which one is the "owner"
 * (the part number determines this).
 *
 * NEW Approach
 *
 * Loop over the vneighbor boxes.  Let pi = my part and pj = vneighbor part.
 * The part with the smaller ID owns the data, so (pi < pj) means that shared
 * vneighbor data overlaps with pi's data and pj's ghost, and (pi > pj) means
 * that shared vneighbor data overlaps with pj's data and pi's ghost.
 *
 * Intersect each vneighbor box with the BoxManager for the owner part (either
 * pi or pj) and intersect a grown vneighbor box with the BoxManager for the
 * non-owner part.  This produces two lists of boxes on the two different parts
 * that share data.  The remainder of the routine loops over these two lists,
 * intersecting the boxes appropriately with the vneighbor box to determine send
 * and receive communication info.  For convenience, the information is put into
 * a 4D "matrix" based on pi, pj, vi (variable on part pi), and vj.  The upper
 * "triangle" (given by pi < pj) stores the send information and the lower
 * triangle stores the receive information.
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGridCreateCommInfo( nalu_hypre_SStructGrid  *grid )
{
   NALU_HYPRE_Int                ndim = nalu_hypre_SStructGridNDim(grid);
   NALU_HYPRE_Int                nparts = nalu_hypre_SStructGridNParts(grid);
   nalu_hypre_SStructPGrid     **pgrids = nalu_hypre_SStructGridPGrids(grid);
   NALU_HYPRE_Int              **nvneighbors = nalu_hypre_SStructGridNVNeighbors(grid);
   nalu_hypre_SStructNeighbor ***vneighbors  = nalu_hypre_SStructGridVNeighbors(grid);
   nalu_hypre_SStructNeighbor   *vneighbor;
   nalu_hypre_SStructCommInfo  **vnbor_comm_info;
   NALU_HYPRE_Int                vnbor_ncomms;
   nalu_hypre_SStructCommInfo   *comm_info;
   NALU_HYPRE_SStructVariable   *vartypes;
   nalu_hypre_Index              varoffset;

   typedef struct
   {
      nalu_hypre_BoxArrayArray    *boxes;
      nalu_hypre_BoxArrayArray    *rboxes;
      NALU_HYPRE_Int             **procs;
      NALU_HYPRE_Int             **rboxnums;
      NALU_HYPRE_Int             **transforms;
      NALU_HYPRE_Int              *num_transforms; /* reference to num transforms */
      nalu_hypre_Index            *coords;
      nalu_hypre_Index            *dirs;

   } CInfo;

   nalu_hypre_IndexRef           coord, dir;

   CInfo                  **cinfo_a;  /* array of size (nparts^2)(maxvars^2) */
   CInfo                   *cinfo, *send_cinfo, *recv_cinfo;
   NALU_HYPRE_Int                cinfoi, cinfoj, maxvars;
   nalu_hypre_BoxArray          *cbox_a;
   nalu_hypre_BoxArray          *crbox_a;
   NALU_HYPRE_Int               *cproc_a;
   NALU_HYPRE_Int               *crboxnum_a;
   NALU_HYPRE_Int               *ctransform_a;
   NALU_HYPRE_Int               *cnum_transforms;
   nalu_hypre_Index             *ccoords;
   nalu_hypre_Index             *cdirs;

   nalu_hypre_SStructPGrid      *pgrid;

   nalu_hypre_BoxManEntry      **pi_entries, **pj_entries;
   nalu_hypre_BoxManEntry       *pi_entry,    *pj_entry;
   NALU_HYPRE_Int                npi_entries,  npj_entries;

   nalu_hypre_Box               *vn_box, *pi_box, *pj_box, *int_box, *int_rbox;
   nalu_hypre_Index              imin0, imin1;

   NALU_HYPRE_Int                nvars, size, pi_proc, myproc;
   NALU_HYPRE_Int                pi, pj, vi, vj, ei, ej, ni, bi, ti;

   nalu_hypre_MPI_Comm_rank(nalu_hypre_SStructGridComm(grid), &myproc);

   vn_box = nalu_hypre_BoxCreate(ndim);
   pi_box = nalu_hypre_BoxCreate(ndim);
   pj_box = nalu_hypre_BoxCreate(ndim);
   int_box = nalu_hypre_BoxCreate(ndim);
   int_rbox = nalu_hypre_BoxCreate(ndim);

   /* initialize cinfo_a array */
   maxvars = 0;
   for (pi = 0; pi < nparts; pi++)
   {
      nvars = nalu_hypre_SStructPGridNVars(pgrids[pi]);
      if ( maxvars < nvars )
      {
         maxvars = nvars;
      }
   }
   cinfo_a = nalu_hypre_CTAlloc(CInfo *,  nparts * nparts * maxvars * maxvars, NALU_HYPRE_MEMORY_HOST);

   /* loop over local boxes and compute send/recv CommInfo */

   vnbor_ncomms = 0;
   /* for each part */
   for (pi = 0; pi < nparts; pi++)
   {
      pgrid  = pgrids[pi];
      nvars = nalu_hypre_SStructPGridNVars(pgrid);
      vartypes = nalu_hypre_SStructPGridVarTypes(pgrid);

      /* for each variable */
      for (vi = 0; vi < nvars; vi++)
      {
         nalu_hypre_SStructVariableGetOffset(vartypes[vi], ndim, varoffset);

         /* for each vneighbor box */
         for (ni = 0; ni < nvneighbors[pi][vi]; ni++)
         {
            vneighbor = &vneighbors[pi][vi][ni];
            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(nalu_hypre_SStructNeighborBox(vneighbor)), imin0);
            nalu_hypre_CopyIndex(nalu_hypre_SStructNeighborILower(vneighbor), imin1);
            coord = nalu_hypre_SStructNeighborCoord(vneighbor);
            dir   = nalu_hypre_SStructNeighborDir(vneighbor);

            pj = nalu_hypre_SStructNeighborPart(vneighbor);
            nalu_hypre_SStructVarToNborVar(grid, pi, vi, coord, &vj);

            /* intersect with grid for part pi */
            nalu_hypre_CopyBox(nalu_hypre_SStructNeighborBox(vneighbor), vn_box);
            /* always grow the vneighbor box */
            nalu_hypre_BoxGrowByIndex(vn_box, varoffset);
            nalu_hypre_SStructGridIntersect(grid, pi, vi, vn_box, 0, &pi_entries, &npi_entries);

            /* intersect with grid for part pj */
            nalu_hypre_CopyBox(nalu_hypre_SStructNeighborBox(vneighbor), vn_box);
            /* always grow the vneighbor box */
            nalu_hypre_BoxGrowByIndex(vn_box, varoffset);
            /* map vneighbor box to part pj index space */
            nalu_hypre_SStructBoxToNborBox(vn_box, imin0, imin1, coord, dir);
            nalu_hypre_SStructGridIntersect(grid, pj, vj, vn_box, 0, &pj_entries, &npj_entries);

            /* loop over pi and pj entries */
            for (ei = 0; ei < npi_entries; ei++)
            {
               pi_entry = pi_entries[ei];
               /* only concerned with pi boxes on my processor */
               nalu_hypre_SStructBoxManEntryGetProcess(pi_entry, &pi_proc);
               if (pi_proc != myproc)
               {
                  continue;
               }
               nalu_hypre_BoxManEntryGetExtents(
                  pi_entry, nalu_hypre_BoxIMin(pi_box), nalu_hypre_BoxIMax(pi_box));

               /* if pi is not the owner, grow pi_box to compute recv boxes */
               if (pi > pj)
               {
                  nalu_hypre_BoxGrowByIndex(pi_box, varoffset);
               }

               for (ej = 0; ej < npj_entries; ej++)
               {
                  pj_entry = pj_entries[ej];
                  nalu_hypre_BoxManEntryGetExtents(
                     pj_entry, nalu_hypre_BoxIMin(pj_box), nalu_hypre_BoxIMax(pj_box));
                  /* map pj_box to part pi index space */
                  nalu_hypre_SStructNborBoxToBox(pj_box, imin0, imin1, coord, dir);

                  /* if pj is not the owner, grow pj_box to compute send boxes */
                  if (pj > pi)
                  {
                     nalu_hypre_BoxGrowByIndex(pj_box, varoffset);
                  }

                  /* intersect the pi and pj boxes */
                  nalu_hypre_IntersectBoxes(pi_box, pj_box, int_box);

                  /* if there is an intersection, compute communication info */
                  if (nalu_hypre_BoxVolume(int_box))
                  {
                     cinfoi = (((pi) * maxvars + vi) * nparts + pj) * maxvars + vj;
                     cinfoj = (((pj) * maxvars + vj) * nparts + pi) * maxvars + vi;

                     /* allocate CommInfo arguments if needed */
                     if (cinfo_a[cinfoi] == NULL)
                     {
                        NALU_HYPRE_Int  i_num_boxes = nalu_hypre_StructGridNumBoxes(
                                                    nalu_hypre_SStructPGridSGrid(pgrids[pi], vi));
                        NALU_HYPRE_Int  j_num_boxes = nalu_hypre_StructGridNumBoxes(
                                                    nalu_hypre_SStructPGridSGrid(pgrids[pj], vj));

                        cnum_transforms = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
                        ccoords = nalu_hypre_CTAlloc(nalu_hypre_Index,  nvneighbors[pi][vi], NALU_HYPRE_MEMORY_HOST);
                        cdirs   = nalu_hypre_CTAlloc(nalu_hypre_Index,  nvneighbors[pi][vi], NALU_HYPRE_MEMORY_HOST);

                        cinfo = nalu_hypre_TAlloc(CInfo,  1, NALU_HYPRE_MEMORY_HOST);
                        (cinfo->boxes) = nalu_hypre_BoxArrayArrayCreate(i_num_boxes, ndim);
                        (cinfo->rboxes) = nalu_hypre_BoxArrayArrayCreate(i_num_boxes, ndim);
                        (cinfo->procs) = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  i_num_boxes, NALU_HYPRE_MEMORY_HOST);
                        (cinfo->rboxnums) = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  i_num_boxes, NALU_HYPRE_MEMORY_HOST);
                        (cinfo->transforms) = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  i_num_boxes, NALU_HYPRE_MEMORY_HOST);
                        (cinfo->num_transforms) = cnum_transforms;
                        (cinfo->coords) = ccoords;
                        (cinfo->dirs) = cdirs;
                        cinfo_a[cinfoi] = cinfo;

                        cinfo = nalu_hypre_TAlloc(CInfo,  1, NALU_HYPRE_MEMORY_HOST);
                        (cinfo->boxes) = nalu_hypre_BoxArrayArrayCreate(j_num_boxes, ndim);
                        (cinfo->rboxes) = nalu_hypre_BoxArrayArrayCreate(j_num_boxes, ndim);
                        (cinfo->procs) = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  j_num_boxes, NALU_HYPRE_MEMORY_HOST);
                        (cinfo->rboxnums) = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  j_num_boxes, NALU_HYPRE_MEMORY_HOST);
                        (cinfo->transforms) = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  j_num_boxes, NALU_HYPRE_MEMORY_HOST);
                        (cinfo->num_transforms) = cnum_transforms;
                        (cinfo->coords) = ccoords;
                        (cinfo->dirs) = cdirs;
                        cinfo_a[cinfoj] = cinfo;

                        vnbor_ncomms++;
                     }

                     cinfo = cinfo_a[cinfoi];


                     nalu_hypre_SStructBoxManEntryGetBoxnum(pi_entry, &bi);

                     cbox_a = nalu_hypre_BoxArrayArrayBoxArray((cinfo->boxes), bi);
                     crbox_a = nalu_hypre_BoxArrayArrayBoxArray((cinfo->rboxes), bi);

                     /* Since cinfo is unique for each (pi,vi,pj,vj), we can use
                      * the remote (proc, boxnum) to determine duplicates */
                     {
                        NALU_HYPRE_Int  j, proc, boxnum, duplicate = 0;

                        nalu_hypre_SStructBoxManEntryGetProcess(pj_entry, &proc);
                        nalu_hypre_SStructBoxManEntryGetBoxnum(pj_entry, &boxnum);
                        cproc_a = (cinfo->procs[bi]);
                        crboxnum_a = (cinfo->rboxnums[bi]);
                        nalu_hypre_ForBoxI(j, cbox_a)
                        {
                           if ( (proc == cproc_a[j]) && (boxnum == crboxnum_a[j]) )
                           {
                              duplicate = 1;
                           }
                        }
                        if (duplicate)
                        {
                           continue;
                        }
                     }

                     size = nalu_hypre_BoxArraySize(cbox_a);
                     /* Allocate in chunks of 10 ('size' grows by 1) */
                     if (size % 10 == 0)
                     {
                        (cinfo->procs[bi]) =
                           nalu_hypre_TReAlloc((cinfo->procs[bi]),  NALU_HYPRE_Int,  size + 10, NALU_HYPRE_MEMORY_HOST);
                        (cinfo->rboxnums[bi]) =
                           nalu_hypre_TReAlloc((cinfo->rboxnums[bi]),  NALU_HYPRE_Int,  size + 10, NALU_HYPRE_MEMORY_HOST);
                        (cinfo->transforms[bi]) =
                           nalu_hypre_TReAlloc((cinfo->transforms[bi]),  NALU_HYPRE_Int,  size + 10, NALU_HYPRE_MEMORY_HOST);
                     }
                     cproc_a = (cinfo->procs[bi]);
                     crboxnum_a = (cinfo->rboxnums[bi]);
                     ctransform_a = (cinfo->transforms[bi]);
                     cnum_transforms = (cinfo->num_transforms);
                     ccoords = (cinfo->coords);
                     cdirs = (cinfo->dirs);

                     /* map intersection box to part pj index space */
                     nalu_hypre_CopyBox(int_box, int_rbox);
                     nalu_hypre_SStructBoxToNborBox(int_rbox, imin0, imin1, coord, dir);

                     nalu_hypre_AppendBox(int_box, cbox_a);
                     nalu_hypre_AppendBox(int_rbox, crbox_a);
                     nalu_hypre_SStructBoxManEntryGetProcess(pj_entry, &cproc_a[size]);
                     nalu_hypre_SStructBoxManEntryGetBoxnum(pj_entry, &crboxnum_a[size]);
                     /* search for transform */
                     for (ti = 0; ti < *cnum_transforms; ti++)
                     {
                        if ( nalu_hypre_IndexesEqual(coord, ccoords[ti], ndim) &&
                             nalu_hypre_IndexesEqual(dir, cdirs[ti], ndim) )
                        {
                           break;
                        }
                     }
                     /* set transform */
                     if (ti >= *cnum_transforms)
                     {
                        nalu_hypre_CopyIndex(coord, ccoords[ti]);
                        nalu_hypre_CopyIndex(dir, cdirs[ti]);
                        (*cnum_transforms)++;
                     }
                     ctransform_a[size] = ti;

                  } /* end of if intersection box */
               } /* end of ej entries loop */
            } /* end of ei entries loop */
            nalu_hypre_TFree(pj_entries, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(pi_entries, NALU_HYPRE_MEMORY_HOST);
         } /* end of ni vneighbor box loop */
      } /* end of vi variable loop */
   } /* end of pi part loop */

   /* loop through the upper triangle and create vnbor_comm_info */
   vnbor_comm_info = nalu_hypre_TAlloc(nalu_hypre_SStructCommInfo *,  vnbor_ncomms, NALU_HYPRE_MEMORY_HOST);
   vnbor_ncomms = 0;
   for (pi = 0; pi < nparts; pi++)
   {
      for (vi = 0; vi < maxvars; vi++)
      {
         for (pj = (pi + 1); pj < nparts; pj++)
         {
            for (vj = 0; vj < maxvars; vj++)
            {
               cinfoi = (((pi) * maxvars + vi) * nparts + pj) * maxvars + vj;

               if (cinfo_a[cinfoi] != NULL)
               {
                  comm_info = nalu_hypre_TAlloc(nalu_hypre_SStructCommInfo,  1, NALU_HYPRE_MEMORY_HOST);

                  cinfoj = (((pj) * maxvars + vj) * nparts + pi) * maxvars + vi;
                  send_cinfo = cinfo_a[cinfoi];
                  recv_cinfo = cinfo_a[cinfoj];

                  /* send/recv boxes may not match (2nd to last argument) */
                  nalu_hypre_CommInfoCreate(
                     (send_cinfo->boxes), (recv_cinfo->boxes),
                     (send_cinfo->procs), (recv_cinfo->procs),
                     (send_cinfo->rboxnums), (recv_cinfo->rboxnums),
                     (send_cinfo->rboxes), (recv_cinfo->rboxes),
                     0, &nalu_hypre_SStructCommInfoCommInfo(comm_info));
                  nalu_hypre_CommInfoSetTransforms(
                     nalu_hypre_SStructCommInfoCommInfo(comm_info),
                     *(send_cinfo->num_transforms),
                     (send_cinfo->coords), (send_cinfo->dirs),
                     (send_cinfo->transforms), (recv_cinfo->transforms));
                  nalu_hypre_TFree(send_cinfo->num_transforms, NALU_HYPRE_MEMORY_HOST);

                  nalu_hypre_SStructCommInfoSendPart(comm_info) = pi;
                  nalu_hypre_SStructCommInfoRecvPart(comm_info) = pj;
                  nalu_hypre_SStructCommInfoSendVar(comm_info) = vi;
                  nalu_hypre_SStructCommInfoRecvVar(comm_info) = vj;

                  vnbor_comm_info[vnbor_ncomms] = comm_info;
#if 0
                  {
                     /* debugging print */
                     nalu_hypre_BoxArrayArray *boxaa;
                     nalu_hypre_BoxArray      *boxa;
                     nalu_hypre_Box           *box;
                     NALU_HYPRE_Int            i, j, d, **procs, **rboxs;

                     boxaa = (comm_info->comm_info->send_boxes);
                     procs = (comm_info->comm_info->send_processes);
                     rboxs = (comm_info->comm_info->send_rboxnums);
                     nalu_hypre_ForBoxArrayI(i, boxaa)
                     {
                        nalu_hypre_printf("%d: (pi,vi:pj,vj) = (%d,%d:%d,%d), ncomm = %d, send box = %d, (proc,rbox: ...) =",
                                     myproc, pi, vi, pj, vj, vnbor_ncomms, i);
                        boxa = nalu_hypre_BoxArrayArrayBoxArray(boxaa, i);
                        nalu_hypre_ForBoxI(j, boxa)
                        {
                           box = nalu_hypre_BoxArrayBox(boxa, j);
                           nalu_hypre_printf(" (%d,%d: ", procs[i][j], rboxs[i][j]);
                           for (d = 0; d < ndim; d++)
                           {
                              nalu_hypre_printf(" %d", nalu_hypre_BoxIMinD(box, d));
                           }
                           nalu_hypre_printf(" x");
                           for (d = 0; d < ndim; d++)
                           {
                              nalu_hypre_printf(" %d", nalu_hypre_BoxIMaxD(box, d));
                           }
                           nalu_hypre_printf(")");
                        }
                        nalu_hypre_printf("\n");
                     }
                     boxaa = (comm_info->comm_info->recv_boxes);
                     procs = (comm_info->comm_info->recv_processes);
                     rboxs = (comm_info->comm_info->recv_rboxnums);
                     nalu_hypre_ForBoxArrayI(i, boxaa)
                     {
                        nalu_hypre_printf("%d: (pi,vi:pj,vj) = (%d,%d:%d,%d), ncomm = %d, recv box = %d, (proc,rbox: ...) =",
                                     myproc, pi, vi, pj, vj, vnbor_ncomms, i);
                        boxa = nalu_hypre_BoxArrayArrayBoxArray(boxaa, i);
                        nalu_hypre_ForBoxI(j, boxa)
                        {
                           box = nalu_hypre_BoxArrayBox(boxa, j);
                           nalu_hypre_printf(" (%d,%d: ", procs[i][j], rboxs[i][j]);
                           for (d = 0; d < ndim; d++)
                           {
                              nalu_hypre_printf(" %d", nalu_hypre_BoxIMinD(box, d));
                           }
                           nalu_hypre_printf(" x");
                           for (d = 0; d < ndim; d++)
                           {
                              nalu_hypre_printf(" %d", nalu_hypre_BoxIMaxD(box, d));
                           }
                           nalu_hypre_printf(")");
                        }
                        nalu_hypre_printf("\n");
                     }
                     fflush(stdout);
                  }
#endif
                  vnbor_ncomms++;
               }
            }
         }
      }
   }
   nalu_hypre_SStructGridVNborCommInfo(grid) = vnbor_comm_info;
   nalu_hypre_SStructGridVNborNComms(grid) = vnbor_ncomms;

   size = nparts * nparts * maxvars * maxvars;
   for (cinfoi = 0; cinfoi < size; cinfoi++)
   {
      nalu_hypre_TFree(cinfo_a[cinfoi], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(cinfo_a, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_BoxDestroy(vn_box);
   nalu_hypre_BoxDestroy(pi_box);
   nalu_hypre_BoxDestroy(pj_box);
   nalu_hypre_BoxDestroy(int_box);
   nalu_hypre_BoxDestroy(int_rbox);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine returns a NULL 'entry_ptr' if an entry is not found
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGridFindBoxManEntry( nalu_hypre_SStructGrid  *grid,
                                  NALU_HYPRE_Int           part,
                                  nalu_hypre_Index         index,
                                  NALU_HYPRE_Int           var,
                                  nalu_hypre_BoxManEntry **entry_ptr )
{
   NALU_HYPRE_Int nentries;

   nalu_hypre_BoxManEntry **entries;

   nalu_hypre_BoxManIntersect (  nalu_hypre_SStructGridBoxManager(grid, part, var),
                            index, index, &entries, &nentries);

   /* we should only get a single entry returned */
   if (nentries > 1)
   {
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
      *entry_ptr = NULL;
   }
   else if (nentries == 0)
   {
      *entry_ptr = NULL;
   }
   else
   {
      *entry_ptr = entries[0];
   }

   /* remove the entries array (NULL or allocated in the intersect routine) */
   nalu_hypre_TFree(entries, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGridFindNborBoxManEntry( nalu_hypre_SStructGrid  *grid,
                                      NALU_HYPRE_Int           part,
                                      nalu_hypre_Index         index,
                                      NALU_HYPRE_Int           var,
                                      nalu_hypre_BoxManEntry **entry_ptr )
{
   NALU_HYPRE_Int nentries;

   nalu_hypre_BoxManEntry **entries;

   nalu_hypre_BoxManIntersect (  nalu_hypre_SStructGridNborBoxManager(grid, part, var),
                            index, index, &entries, &nentries);

   /* we should only get a single entry returned */
   if (nentries >  1)
   {
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
      *entry_ptr = NULL;
   }
   else if (nentries == 0)
   {
      *entry_ptr = NULL;
   }
   else
   {
      *entry_ptr = entries[0];
   }

   /* remove the entries array (NULL or allocated in the intersect routine) */
   nalu_hypre_TFree(entries, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGridBoxProcFindBoxManEntry( nalu_hypre_SStructGrid  *grid,
                                         NALU_HYPRE_Int           part,
                                         NALU_HYPRE_Int           var,
                                         NALU_HYPRE_Int           box,
                                         NALU_HYPRE_Int           proc,
                                         nalu_hypre_BoxManEntry **entry_ptr )
{
   nalu_hypre_BoxManGetEntry(nalu_hypre_SStructGridBoxManager(grid, part, var),
                        proc, box, entry_ptr);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructBoxManEntryGetCSRstrides(  nalu_hypre_BoxManEntry *entry,
                                        nalu_hypre_Index        strides )
{
   nalu_hypre_SStructBoxManInfo *entry_info;

   nalu_hypre_BoxManEntryGetInfo(entry, (void **) &entry_info);

   if (nalu_hypre_SStructBoxManInfoType(entry_info) == nalu_hypre_SSTRUCT_BOXMAN_INFO_DEFAULT)
   {
      NALU_HYPRE_Int    d, ndim = nalu_hypre_BoxManEntryNDim(entry);
      nalu_hypre_Index  imin;
      nalu_hypre_Index  imax;

      nalu_hypre_BoxManEntryGetExtents(entry, imin, imax);

      strides[0] = 1;
      for (d = 1; d < ndim; d++)
      {
         strides[d] = nalu_hypre_IndexD(imax, d - 1) - nalu_hypre_IndexD(imin, d - 1) + 1;
         strides[d] *= strides[d - 1];
      }
   }
   else
   {
      nalu_hypre_SStructBoxManNborInfo *entry_ninfo;

      entry_ninfo = (nalu_hypre_SStructBoxManNborInfo *) entry_info;

      nalu_hypre_CopyIndex(nalu_hypre_SStructBoxManNborInfoStride(entry_ninfo), strides);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * GEC1002 addition for a ghost stride calculation
 * same function except that you modify imin, imax with the ghost and
 * when the info is type nmapinfo you pull the ghoststrides.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructBoxManEntryGetGhstrides( nalu_hypre_BoxManEntry *entry,
                                      nalu_hypre_Index        strides )
{
   nalu_hypre_SStructBoxManInfo *entry_info;
   NALU_HYPRE_Int               *numghost;

   nalu_hypre_BoxManEntryGetInfo(entry, (void **) &entry_info);

   if (nalu_hypre_SStructBoxManInfoType(entry_info) == nalu_hypre_SSTRUCT_BOXMAN_INFO_DEFAULT)
   {
      NALU_HYPRE_Int    d, ndim = nalu_hypre_BoxManEntryNDim(entry);
      nalu_hypre_Index  imin;
      nalu_hypre_Index  imax;

      nalu_hypre_BoxManEntryGetExtents(entry, imin, imax);

      /* getting the ghost from the mapentry to modify imin, imax */

      numghost = nalu_hypre_BoxManEntryNumGhost(entry);

      for (d = 0; d < ndim; d++)
      {
         imax[d] += numghost[2 * d + 1];
         imin[d] -= numghost[2 * d];
      }

      /* imin, imax modified now and calculation identical.  */

      strides[0] = 1;
      for (d = 1; d < ndim; d++)
      {
         strides[d] = nalu_hypre_IndexD(imax, d - 1) - nalu_hypre_IndexD(imin, d - 1) + 1;
         strides[d] *= strides[d - 1];
      }
   }
   else
   {
      nalu_hypre_SStructBoxManNborInfo *entry_ninfo;
      /* now get the ghost strides using the macro   */
      entry_ninfo = (nalu_hypre_SStructBoxManNborInfo *) entry_info;
      nalu_hypre_CopyIndex(nalu_hypre_SStructBoxManNborInfoGhstride(entry_ninfo), strides);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructBoxManEntryGetGlobalCSRank( nalu_hypre_BoxManEntry *entry,
                                         nalu_hypre_Index        index,
                                         NALU_HYPRE_BigInt      *rank_ptr )
{
   NALU_HYPRE_Int                ndim = nalu_hypre_BoxManEntryNDim(entry);
   nalu_hypre_SStructBoxManInfo *entry_info;
   nalu_hypre_Index              imin;
   nalu_hypre_Index              imax;
   nalu_hypre_Index              strides;
   NALU_HYPRE_BigInt             offset;
   NALU_HYPRE_Int                d;

   nalu_hypre_BoxManEntryGetInfo(entry, (void **) &entry_info);
   nalu_hypre_BoxManEntryGetExtents(entry, imin, imax);
   offset = nalu_hypre_SStructBoxManInfoOffset(entry_info);

   nalu_hypre_SStructBoxManEntryGetCSRstrides(entry, strides);

   *rank_ptr = offset;
   for (d = 0; d < ndim; d++)
   {
      *rank_ptr += (NALU_HYPRE_BigInt)((nalu_hypre_IndexD(index, d) - nalu_hypre_IndexD(imin, d)) * strides[d]);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * GEC1002 a way to get the rank when you are in the presence of ghosts
 * It could have been a function pointer but this is safer. It computes
 * the ghost rank by using ghoffset, ghstrides and imin is modified
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructBoxManEntryGetGlobalGhrank( nalu_hypre_BoxManEntry *entry,
                                         nalu_hypre_Index        index,
                                         NALU_HYPRE_BigInt      *rank_ptr )
{
   NALU_HYPRE_Int                 ndim = nalu_hypre_BoxManEntryNDim(entry);
   nalu_hypre_SStructBoxManInfo  *entry_info;
   nalu_hypre_Index               imin;
   nalu_hypre_Index               imax;
   nalu_hypre_Index               ghstrides;
   NALU_HYPRE_BigInt              ghoffset;
   NALU_HYPRE_Int                 *numghost = nalu_hypre_BoxManEntryNumGhost(entry);
   NALU_HYPRE_Int                 d;
   NALU_HYPRE_Int                 info_type;

   nalu_hypre_BoxManEntryGetInfo(entry, (void **) &entry_info);
   nalu_hypre_BoxManEntryGetExtents(entry, imin, imax);
   ghoffset = nalu_hypre_SStructBoxManInfoGhoffset(entry_info);
   info_type = nalu_hypre_SStructBoxManInfoType(entry_info);

   nalu_hypre_SStructBoxManEntryGetGhstrides(entry, ghstrides);

   /* GEC shifting the imin according to the ghosts when you have a default info
    * When you have a neighbor info, you do not need to shift the imin since
    * the ghoffset for neighbor info has factored in the ghost presence during
    * the neighbor info assemble phase   */

   if (info_type == nalu_hypre_SSTRUCT_BOXMAN_INFO_DEFAULT)
   {
      for (d = 0; d < ndim; d++)
      {
         imin[d] -= numghost[2 * d];
      }
   }

   *rank_ptr = ghoffset;
   for (d = 0; d < ndim; d++)
   {
      *rank_ptr += (NALU_HYPRE_BigInt)((nalu_hypre_IndexD(index, d) - nalu_hypre_IndexD(imin, d)) * ghstrides[d]);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructBoxManEntryGetProcess( nalu_hypre_BoxManEntry *entry,
                                    NALU_HYPRE_Int         *proc_ptr )
{
   *proc_ptr = nalu_hypre_BoxManEntryProc(entry);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * For neighbors, the boxnum is in the info, otherwise it is the same
 * as the id.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructBoxManEntryGetBoxnum( nalu_hypre_BoxManEntry *entry,
                                   NALU_HYPRE_Int         *id_ptr )
{
   nalu_hypre_SStructBoxManNborInfo *info;

   nalu_hypre_BoxManEntryGetInfo(entry, (void **) &info);

   if (nalu_hypre_SStructBoxManInfoType(info) ==
       nalu_hypre_SSTRUCT_BOXMAN_INFO_NEIGHBOR)
      /* get from the info object */
   {
      *id_ptr = nalu_hypre_SStructBoxManNborInfoBoxnum(info);
   }
   else /* use id from the entry */
   {
      *id_ptr = nalu_hypre_BoxManEntryId(entry);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructBoxManEntryGetPart( nalu_hypre_BoxManEntry *entry,
                                 NALU_HYPRE_Int          part,
                                 NALU_HYPRE_Int         *part_ptr )
{
   nalu_hypre_SStructBoxManNborInfo *info;

   nalu_hypre_BoxManEntryGetInfo(entry, (void **) &info);

   if (nalu_hypre_SStructBoxManInfoType(info) == nalu_hypre_SSTRUCT_BOXMAN_INFO_NEIGHBOR)
   {
      *part_ptr = nalu_hypre_SStructBoxManNborInfoPart(info);
   }
   else
   {
      *part_ptr = part;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Mapping Notes:
 *
 *   coord maps Box index-space to NborBox index-space.  That is, `coord[d]' is
 *   the dimension in the NborBox index-space, and `d' is the dimension in the
 *   Box index-space.
 *
 *   dir also works on the Box index-space.  That is, `dir[d]' is the direction
 *   (positive or negative) of dimension `coord[d]' in the NborBox index-space,
 *   relative to the positive direction of dimension `d' in the Box index-space.
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructIndexToNborIndex( nalu_hypre_Index  index,
                               nalu_hypre_Index  root,
                               nalu_hypre_Index  nbor_root,
                               nalu_hypre_Index  coord,
                               nalu_hypre_Index  dir,
                               NALU_HYPRE_Int    ndim,
                               nalu_hypre_Index  nbor_index )
{
   NALU_HYPRE_Int  d, nd;

   for (d = 0; d < ndim; d++)
   {
      nd = coord[d];
      nbor_index[nd] = nbor_root[nd] + (index[d] - root[d]) * dir[d];
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructBoxToNborBox( nalu_hypre_Box   *box,
                           nalu_hypre_Index  root,
                           nalu_hypre_Index  nbor_root,
                           nalu_hypre_Index  coord,
                           nalu_hypre_Index  dir )
{
   NALU_HYPRE_Int   *imin = nalu_hypre_BoxIMin(box);
   NALU_HYPRE_Int   *imax = nalu_hypre_BoxIMax(box);
   NALU_HYPRE_Int    ndim = nalu_hypre_BoxNDim(box);
   nalu_hypre_Index  nbor_imin, nbor_imax;
   NALU_HYPRE_Int    d;

   nalu_hypre_SStructIndexToNborIndex(imin, root, nbor_root, coord, dir, ndim, nbor_imin);
   nalu_hypre_SStructIndexToNborIndex(imax, root, nbor_root, coord, dir, ndim, nbor_imax);

   for (d = 0; d < ndim; d++)
   {
      imin[d] = nalu_hypre_min(nbor_imin[d], nbor_imax[d]);
      imax[d] = nalu_hypre_max(nbor_imin[d], nbor_imax[d]);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * See "Mapping Notes" in comment for `nalu_hypre_SStructBoxToNborBox'.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructNborIndexToIndex( nalu_hypre_Index  nbor_index,
                               nalu_hypre_Index  root,
                               nalu_hypre_Index  nbor_root,
                               nalu_hypre_Index  coord,
                               nalu_hypre_Index  dir,
                               NALU_HYPRE_Int    ndim,
                               nalu_hypre_Index  index )
{
   NALU_HYPRE_Int  d, nd;

   for (d = 0; d < ndim; d++)
   {
      nd = coord[d];
      index[d] = root[d] + (nbor_index[nd] - nbor_root[nd]) * dir[d];
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructNborBoxToBox( nalu_hypre_Box   *nbor_box,
                           nalu_hypre_Index  root,
                           nalu_hypre_Index  nbor_root,
                           nalu_hypre_Index  coord,
                           nalu_hypre_Index  dir )
{
   NALU_HYPRE_Int   *nbor_imin = nalu_hypre_BoxIMin(nbor_box);
   NALU_HYPRE_Int   *nbor_imax = nalu_hypre_BoxIMax(nbor_box);
   NALU_HYPRE_Int    ndim = nalu_hypre_BoxNDim(nbor_box);
   nalu_hypre_Index  imin, imax;
   NALU_HYPRE_Int    d;

   nalu_hypre_SStructNborIndexToIndex(nbor_imin, root, nbor_root, coord, dir, ndim, imin);
   nalu_hypre_SStructNborIndexToIndex(nbor_imax, root, nbor_root, coord, dir, ndim, imax);

   for (d = 0; d < ndim; d++)
   {
      nbor_imin[d] = nalu_hypre_min(imin[d], imax[d]);
      nbor_imax[d] = nalu_hypre_max(imin[d], imax[d]);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *
 * Assumptions:
 *
 * 1. Variables and variable types are the same on neighboring parts
 * 2. Variable types are listed in order as follows:
 *       Face - XFACE, YFACE, ZFACE
 *       Edge - XEDGE, YEDGE, ZEDGE
 * 3. If the coordinate transformation is not the identity, then all ndim
 *    variable types must exist on the grid.
 *
 *--------------------------------------------------------------------------*/

/* ONLY3D for non-cell and non-node variable types */

NALU_HYPRE_Int
nalu_hypre_SStructVarToNborVar( nalu_hypre_SStructGrid  *grid,
                           NALU_HYPRE_Int           part,
                           NALU_HYPRE_Int           var,
                           NALU_HYPRE_Int          *coord,
                           NALU_HYPRE_Int          *nbor_var_ptr)
{
   nalu_hypre_SStructPGrid     *pgrid   = nalu_hypre_SStructGridPGrid(grid, part);
   NALU_HYPRE_SStructVariable   vartype = nalu_hypre_SStructPGridVarType(pgrid, var);

   switch (vartype)
   {
      case NALU_HYPRE_SSTRUCT_VARIABLE_XFACE:
      case NALU_HYPRE_SSTRUCT_VARIABLE_XEDGE:
         *nbor_var_ptr = var + (coord[0]  );
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_YFACE:
      case NALU_HYPRE_SSTRUCT_VARIABLE_YEDGE:
         *nbor_var_ptr = var + (coord[1] - 1);
         break;
      case NALU_HYPRE_SSTRUCT_VARIABLE_ZFACE:
      case NALU_HYPRE_SSTRUCT_VARIABLE_ZEDGE:
         *nbor_var_ptr = var + (coord[2] - 2);
         break;
      default:
         *nbor_var_ptr = var;
         break;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * GEC0902 a function that will set the ghost in each of the sgrids
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGridSetNumGhost( nalu_hypre_SStructGrid  *grid, NALU_HYPRE_Int *num_ghost )
{
   NALU_HYPRE_Int             ndim   = nalu_hypre_SStructGridNDim(grid);
   NALU_HYPRE_Int             nparts = nalu_hypre_SStructGridNParts(grid);
   NALU_HYPRE_Int             part, i, t;
   nalu_hypre_SStructPGrid   *pgrid;
   nalu_hypre_StructGrid     *sgrid;

   for (i = 0; i < 2 * ndim; i++)
   {
      nalu_hypre_SStructGridNumGhost(grid)[i] = num_ghost[i];
   }

   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);

      for (t = 0; t < 8; t++)
      {
         sgrid = nalu_hypre_SStructPGridVTSGrid(pgrid, t);
         if (sgrid != NULL)
         {
            nalu_hypre_StructGridSetNumGhost(sgrid, num_ghost);
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * GEC1002 a function that will select the right way to calculate the rank
 * depending on the matrix type. It is an extension to the usual GetGlobalRank
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructBoxManEntryGetGlobalRank( nalu_hypre_BoxManEntry *entry,
                                       nalu_hypre_Index        index,
                                       NALU_HYPRE_BigInt      *rank_ptr,
                                       NALU_HYPRE_Int          type)
{
   if (type == NALU_HYPRE_PARCSR)
   {
      nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, index, rank_ptr);
   }
   if (type == NALU_HYPRE_SSTRUCT || type == NALU_HYPRE_STRUCT)
   {
      nalu_hypre_SStructBoxManEntryGetGlobalGhrank(entry, index, rank_ptr);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * GEC1002 a function that will select the right way to calculate the strides
 * depending on the matrix type. It is an extension to the usual strides
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructBoxManEntryGetStrides(nalu_hypre_BoxManEntry   *entry,
                                   nalu_hypre_Index          strides,
                                   NALU_HYPRE_Int            type)
{
   if (type == NALU_HYPRE_PARCSR)
   {
      nalu_hypre_SStructBoxManEntryGetCSRstrides(entry, strides);
   }
   if (type == NALU_HYPRE_SSTRUCT || type == NALU_HYPRE_STRUCT)
   {
      nalu_hypre_SStructBoxManEntryGetGhstrides(entry, strides);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *  A function to determine the local variable box numbers that underlie
 *  a cellbox with local box number boxnum. Only returns local box numbers
 *  of myproc.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructBoxNumMap(nalu_hypre_SStructGrid        *grid,
                       NALU_HYPRE_Int                 part,
                       NALU_HYPRE_Int                 boxnum,
                       NALU_HYPRE_Int               **num_varboxes_ptr,
                       NALU_HYPRE_Int              ***map_ptr)
{
   nalu_hypre_SStructPGrid    *pgrid   = nalu_hypre_SStructGridPGrid(grid, part);
   nalu_hypre_StructGrid      *cellgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);
   nalu_hypre_StructGrid      *vargrid;
   nalu_hypre_BoxArray        *boxes;
   nalu_hypre_Box             *cellbox, vbox, *box, intersect_box;
   NALU_HYPRE_SStructVariable *vartypes = nalu_hypre_SStructPGridVarTypes(pgrid);

   NALU_HYPRE_Int              ndim    = nalu_hypre_SStructGridNDim(grid);
   NALU_HYPRE_Int              nvars   = nalu_hypre_SStructPGridNVars(pgrid);
   nalu_hypre_Index            varoffset;

   NALU_HYPRE_Int             *num_boxes;
   NALU_HYPRE_Int            **var_boxnums;
   NALU_HYPRE_Int             *temp;

   NALU_HYPRE_Int              i, j, k, var;

   nalu_hypre_BoxInit(&vbox, ndim);
   nalu_hypre_BoxInit(&intersect_box, ndim);
   cellbox = nalu_hypre_StructGridBox(cellgrid, boxnum);

   /* ptrs to store var_box map info */
   num_boxes  = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nvars, NALU_HYPRE_MEMORY_HOST);
   var_boxnums = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nvars, NALU_HYPRE_MEMORY_HOST);

   /* intersect the cellbox with the var_boxes */
   for (var = 0; var < nvars; var++)
   {
      vargrid = nalu_hypre_SStructPGridSGrid(pgrid, var);
      boxes  = nalu_hypre_StructGridBoxes(vargrid);
      temp   = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(boxes), NALU_HYPRE_MEMORY_HOST);

      /* map cellbox to a variable box */
      nalu_hypre_CopyBox(cellbox, &vbox);

      i = vartypes[var];
      nalu_hypre_SStructVariableGetOffset((nalu_hypre_SStructVariable) i,
                                     ndim, varoffset);
      nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&vbox), varoffset, ndim,
                            nalu_hypre_BoxIMin(&vbox));

      /* loop over boxes to see if they intersect with vbox */
      nalu_hypre_ForBoxI(i, boxes)
      {
         box = nalu_hypre_BoxArrayBox(boxes, i);
         nalu_hypre_IntersectBoxes(&vbox, box, &intersect_box);
         if (nalu_hypre_BoxVolume(&intersect_box))
         {
            temp[i]++;
            num_boxes[var]++;
         }
      }

      /* record local var box numbers */
      if (num_boxes[var])
      {
         var_boxnums[var] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_boxes[var], NALU_HYPRE_MEMORY_HOST);
      }
      else
      {
         var_boxnums[var] = NULL;
      }

      j = 0;
      k = nalu_hypre_BoxArraySize(boxes);
      for (i = 0; i < k; i++)
      {
         if (temp[i])
         {
            var_boxnums[var][j] = i;
            j++;
         }
      }
      nalu_hypre_TFree(temp, NALU_HYPRE_MEMORY_HOST);

   }  /* for (var= 0; var< nvars; var++) */

   *num_varboxes_ptr = num_boxes;
   *map_ptr = var_boxnums;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *  A function to extract all the local var box numbers underlying the
 *  cellgrid boxes.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructCellGridBoxNumMap(nalu_hypre_SStructGrid        *grid,
                               NALU_HYPRE_Int                 part,
                               NALU_HYPRE_Int              ***num_varboxes_ptr,
                               NALU_HYPRE_Int             ****map_ptr)
{
   nalu_hypre_SStructPGrid    *pgrid    = nalu_hypre_SStructGridPGrid(grid, part);
   nalu_hypre_StructGrid      *cellgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);
   nalu_hypre_BoxArray        *cellboxes = nalu_hypre_StructGridBoxes(cellgrid);

   NALU_HYPRE_Int            **num_boxes;
   NALU_HYPRE_Int           ***var_boxnums;

   NALU_HYPRE_Int              i, ncellboxes;

   ncellboxes = nalu_hypre_BoxArraySize(cellboxes);

   num_boxes  = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  ncellboxes, NALU_HYPRE_MEMORY_HOST);
   var_boxnums = nalu_hypre_TAlloc(NALU_HYPRE_Int **,  ncellboxes, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ForBoxI(i, cellboxes)
   {
      nalu_hypre_SStructBoxNumMap(grid,
                             part,
                             i,
                             &num_boxes[i],
                             &var_boxnums[i]);
   }

   *num_varboxes_ptr = num_boxes;
   *map_ptr = var_boxnums;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Converts a cell-based box with offset to a variable-based box.  The argument
 * valid is a boolean that specifies the status of the conversion.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructCellBoxToVarBox( nalu_hypre_Box   *box,
                              nalu_hypre_Index  offset,
                              nalu_hypre_Index  varoffset,
                              NALU_HYPRE_Int   *valid )
{
   nalu_hypre_IndexRef imin = nalu_hypre_BoxIMin(box);
   nalu_hypre_IndexRef imax = nalu_hypre_BoxIMax(box);
   NALU_HYPRE_Int      ndim = nalu_hypre_BoxNDim(box);
   NALU_HYPRE_Int      d, off;

   *valid = 1;
   for (d = 0; d < ndim; d++)
   {
      off = nalu_hypre_IndexD(offset, d);
      if ( (nalu_hypre_IndexD(varoffset, d) == 0) && (off != 0) )
      {
         *valid = 0;
         break;
      }
      if (off < 0)
      {
         nalu_hypre_IndexD(imin, d) -= 1;
         nalu_hypre_IndexD(imax, d) -= 1;
      }
      else if (off == 0)
      {
         nalu_hypre_IndexD(imin, d) -= nalu_hypre_IndexD(varoffset, d);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Intersects with either the grid's boxman or neighbor boxman.
 *
 * action = 0   intersect only with my box manager
 * action = 1   intersect only with my neighbor box manager
 * action < 0   intersect with both box managers
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGridIntersect( nalu_hypre_SStructGrid   *grid,
                            NALU_HYPRE_Int            part,
                            NALU_HYPRE_Int            var,
                            nalu_hypre_Box           *box,
                            NALU_HYPRE_Int            action,
                            nalu_hypre_BoxManEntry ***entries_ptr,
                            NALU_HYPRE_Int           *nentries_ptr )
{
   nalu_hypre_BoxManEntry **entries, **tentries;
   NALU_HYPRE_Int           nentries, ntentries, i;
   nalu_hypre_BoxManager   *boxman;

   if (action < 0)
   {
      boxman = nalu_hypre_SStructGridBoxManager(grid, part, var);
      nalu_hypre_BoxManIntersect(boxman, nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box),
                            &entries, &nentries);
      boxman = nalu_hypre_SStructGridNborBoxManager(grid, part, var);
      nalu_hypre_BoxManIntersect(boxman, nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box),
                            &tentries, &ntentries);
      entries = nalu_hypre_TReAlloc(entries,  nalu_hypre_BoxManEntry *,
                               (nentries + ntentries), NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < ntentries; i++)
      {
         entries[nentries + i] = tentries[i];
      }
      nentries += ntentries;
      nalu_hypre_TFree(tentries, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      if (action == 0)
      {
         boxman = nalu_hypre_SStructGridBoxManager(grid, part, var);
      }
      else
      {
         boxman = nalu_hypre_SStructGridNborBoxManager(grid, part, var);
      }
      nalu_hypre_BoxManIntersect(boxman, nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box),
                            &entries, &nentries);
   }

   *entries_ptr  = entries;
   *nentries_ptr = nentries;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGridGetMaxBoxSize( nalu_hypre_SStructGrid *grid )
{
   NALU_HYPRE_Int            nparts = nalu_hypre_SStructGridNParts(grid);
   NALU_HYPRE_Int            part;
   nalu_hypre_SStructPGrid  *pgrid;
   NALU_HYPRE_Int            max_box_size = 0;

   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      max_box_size = nalu_hypre_max(max_box_size, nalu_hypre_SStructPGridGetMaxBoxSize(pgrid));
   }

   return max_box_size;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGridPrint( FILE              *file,
                        nalu_hypre_SStructGrid *grid )
{
   /* Grid variables */
   NALU_HYPRE_Int               ndim = nalu_hypre_SStructGridNDim(grid);
   NALU_HYPRE_Int               nparts = nalu_hypre_SStructGridNParts(grid);
   NALU_HYPRE_Int              *nneighbors = nalu_hypre_SStructGridNNeighbors(grid);
   nalu_hypre_SStructNeighbor **neighbors  = nalu_hypre_SStructGridNeighbors(grid);
   nalu_hypre_Index           **nbor_offsets = nalu_hypre_SStructGridNborOffsets(grid);
   nalu_hypre_IndexRef          nbor_offset;
   nalu_hypre_IndexRef          coord, dir, ilomap;
   NALU_HYPRE_Int               npart;
   nalu_hypre_SStructNeighbor  *neighbor;
   nalu_hypre_SStructPGrid     *pgrid;
   nalu_hypre_StructGrid       *sgrid;
   nalu_hypre_BoxArray         *boxes;
   nalu_hypre_Box              *box;
   NALU_HYPRE_SStructVariable  *vartypes;
   NALU_HYPRE_Int              *num_ghost;
   nalu_hypre_IndexRef          periodic;

   /* Local variables */
   NALU_HYPRE_Int               i;
   NALU_HYPRE_Int               part, var;
   NALU_HYPRE_Int               nvars;
   NALU_HYPRE_Int               nboxes;

   /* Print basic info */
   nalu_hypre_fprintf(file, "\nGridCreate: %d %d\n\n", ndim, nparts);

   /* Print number of boxes per part */
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      sgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);
      boxes = nalu_hypre_StructGridBoxes(sgrid);
      nboxes = nalu_hypre_BoxArraySize(boxes);

      nalu_hypre_fprintf(file, "GridNumBoxes: %d %d\n", part, nboxes);
   }

   /* Print boxes per part */
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      sgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);
      boxes = nalu_hypre_StructGridBoxes(sgrid);

      nalu_hypre_ForBoxI(i, boxes)
      {
         box = nalu_hypre_BoxArrayBox(boxes, i);

         nalu_hypre_fprintf(file, "\nGridSetExtents: (%d, %d): ", part, i);
         nalu_hypre_BoxPrint(file, box);
      }
   }
   nalu_hypre_fprintf(file, "\n\n");

   /* Print variable info per part */
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);
      vartypes = nalu_hypre_SStructPGridVarTypes(pgrid);

      nalu_hypre_fprintf(file, "GridSetVariables: %d %d ", part, nvars);
      nalu_hypre_fprintf(file, "[%d", vartypes[0]);
      for (var = 1; var < nvars; var++)
      {
         nalu_hypre_fprintf(file, " %d", vartypes[var]);
      }
      nalu_hypre_fprintf(file, "]\n");
   }
   nalu_hypre_fprintf(file, "\n");

   /* Print ghost info */
   num_ghost = nalu_hypre_SStructGridNumGhost(grid);
   nalu_hypre_fprintf(file, "GridSetNumGhost:");
   for (i = 0; i < 2 * ndim; i++)
   {
      nalu_hypre_fprintf(file, " %d", num_ghost[i]);
   }
   nalu_hypre_fprintf(file, "\n");

   /* Print periodic data per part */
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      periodic = nalu_hypre_SStructPGridPeriodic(pgrid);

      nalu_hypre_fprintf(file, "\nGridSetPeriodic: %d ", part);
      nalu_hypre_IndexPrint(file, ndim, periodic);
   }
   nalu_hypre_fprintf(file, "\n\n");

   /* GridSetFEMOrdering */

   /* GridSetSharedPart and GridSetNeighborPart data */
   for (part = 0; part < nparts; part++)
   {
      nalu_hypre_fprintf(file, "GridNumNeighbors: %d %d\n", part, nneighbors[part]);
      for (i = 0; i < nneighbors[part]; i++)
      {
         neighbor = &neighbors[part][i];
         nbor_offset = nbor_offsets[part][i];
         box = nalu_hypre_SStructNeighborBox(neighbor);
         npart = nalu_hypre_SStructNeighborPart(neighbor);
         coord = nalu_hypre_SStructNeighborCoord(neighbor);
         dir = nalu_hypre_SStructNeighborDir(neighbor);
         ilomap = nalu_hypre_SStructNeighborILower(neighbor);

         /* Print SStructNeighbor info */
         nalu_hypre_fprintf(file, "GridNeighborInfo: ");
         nalu_hypre_BoxPrint(file, box);
         nalu_hypre_fprintf(file, " ");
         nalu_hypre_IndexPrint(file, ndim, nbor_offset);
         nalu_hypre_fprintf(file, " %d ", npart);
         nalu_hypre_IndexPrint(file, ndim, coord);
         nalu_hypre_fprintf(file, " ");
         nalu_hypre_IndexPrint(file, ndim, dir);
         nalu_hypre_fprintf(file, " ");
         nalu_hypre_IndexPrint(file, ndim, ilomap);
         nalu_hypre_fprintf(file, "\n");
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructGridRead
 *
 * This function reads a semi-structured grid from file. This is used mainly
 * for debugging purposes.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SStructGridRead( MPI_Comm            comm,
                       FILE               *file,
                       nalu_hypre_SStructGrid **grid_ptr )
{
   /* Grid variables */
   NALU_HYPRE_SStructGrid       grid;
   NALU_HYPRE_SStructVariable  *vartypes;
   NALU_HYPRE_Int               num_ghost[2 * NALU_HYPRE_MAXDIM];
   nalu_hypre_Index           **nbor_offsets;
   NALU_HYPRE_Int              *nneighbors;
   nalu_hypre_SStructNeighbor **neighbors;
   nalu_hypre_SStructNeighbor  *neighbor;
   nalu_hypre_Index             periodic;

   /* Local variables */
   NALU_HYPRE_Int               ndim;
   NALU_HYPRE_Int               b, d, i, j;
   NALU_HYPRE_Int               part;
   NALU_HYPRE_Int               nparts, nvars;
   NALU_HYPRE_Int               nboxes;
   NALU_HYPRE_Int              *nboxes_array;
   nalu_hypre_Box              *box;

   nalu_hypre_fscanf(file, "\nGridCreate: %d %d\n\n", &ndim, &nparts);
   NALU_HYPRE_SStructGridCreate(comm, ndim, nparts, &grid);

   /* Allocate memory */
   nboxes_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nparts, NALU_HYPRE_MEMORY_HOST);
   box = nalu_hypre_BoxCreate(ndim);

   /* Read number of boxes per part */
   for (i = 0; i < nparts; i++)
   {
      nalu_hypre_fscanf(file, "GridNumBoxes: %d %d\n", &part, &nboxes);
      nboxes_array[part] = nboxes;
   }
   nalu_hypre_fscanf(file, "\n");

   /* Read boxes per part */
   for (i = 0; i < nparts; i++)
   {
      for (j = 0; j < nboxes_array[i]; j++)
      {
         nalu_hypre_fscanf(file, "\nGridSetExtents: (%d, %d): ", &part, &b);
         nalu_hypre_BoxRead(file, ndim, &box);

         NALU_HYPRE_SStructGridSetExtents(grid, part, nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box));
      }
   }
   nalu_hypre_fscanf(file, "\n\n");

   /* Read variable info per part */
   for (i = 0; i < nparts; i++)
   {
      nalu_hypre_fscanf(file, "GridSetVariables: %d %d ", &part, &nvars);
      vartypes = nalu_hypre_CTAlloc(nalu_hypre_SStructVariable, nvars, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_fscanf(file, "[%d", &vartypes[0]);
      for (j = 1; j < nvars; j++)
      {
         nalu_hypre_fscanf(file, " %d", &vartypes[j]);
      }
      nalu_hypre_fscanf(file, "]\n");
      NALU_HYPRE_SStructGridSetVariables(grid, part, nvars, vartypes);
      nalu_hypre_TFree(vartypes, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_fscanf(file, "\n");

   /* Read ghost info */
   nalu_hypre_fscanf(file, "GridSetNumGhost:");
   for (i = 0; i < 2 * ndim; i++)
   {
      nalu_hypre_fscanf(file, " %d", &num_ghost[i]);
   }
   nalu_hypre_fscanf(file, "\n");

   /* Read periodic data per part */
   for (i = 0; i < nparts; i++)
   {
      nalu_hypre_fscanf(file, "\nGridSetPeriodic: %d ", &part);
      nalu_hypre_IndexRead(file, ndim, periodic);

      NALU_HYPRE_SStructGridSetPeriodic(grid, part, periodic);
   }
   nalu_hypre_fscanf(file, "\n\n");

   /* GridSetFEMOrdering */

   /* GridSetSharedPart and GridSetNeighborPart data */
   nneighbors = nalu_hypre_SStructGridNNeighbors(grid);
   neighbors  = nalu_hypre_SStructGridNeighbors(grid);
   nbor_offsets = nalu_hypre_SStructGridNborOffsets(grid);
   for (part = 0; part < nparts; part++)
   {
      nalu_hypre_fscanf(file, "GridNumNeighbors: %d %d\n", &part, &nneighbors[part]);
      neighbors[part] = nalu_hypre_TAlloc(nalu_hypre_SStructNeighbor, nneighbors[part], NALU_HYPRE_MEMORY_HOST);
      nbor_offsets[part] = nalu_hypre_TAlloc(nalu_hypre_Index, nneighbors[part], NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < nneighbors[part]; i++)
      {
         neighbor = &neighbors[part][i];

         /* Read SStructNeighbor info */
         nalu_hypre_fscanf(file, "GridNeighborInfo: ");
         nalu_hypre_BoxRead(file, ndim, &box);
         nalu_hypre_CopyBox(box, nalu_hypre_SStructNeighborBox(neighbor));
         nalu_hypre_fscanf(file, " ");
         nalu_hypre_IndexRead(file, ndim, nbor_offsets[part][i]);
         nalu_hypre_fscanf(file, " %d ", &nalu_hypre_SStructNeighborPart(neighbor));
         nalu_hypre_IndexRead(file, ndim, nalu_hypre_SStructNeighborCoord(neighbor));
         nalu_hypre_fscanf(file, " ");
         nalu_hypre_IndexRead(file, ndim, nalu_hypre_SStructNeighborDir(neighbor));
         nalu_hypre_fscanf(file, " ");
         nalu_hypre_IndexRead(file, ndim, nalu_hypre_SStructNeighborILower(neighbor));
         nalu_hypre_fscanf(file, "\n");

         for (d = ndim; d < NALU_HYPRE_MAXDIM; d++)
         {
            nalu_hypre_IndexD(nalu_hypre_SStructNeighborCoord(neighbor), d) = d;
            nalu_hypre_IndexD(nalu_hypre_SStructNeighborDir(neighbor), d) = 1;
         }
      }
   }

   /* Assemble grid */
   NALU_HYPRE_SStructGridAssemble(grid);

   /* Free memory */
   nalu_hypre_TFree(nboxes_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_BoxDestroy(box);

   *grid_ptr = grid;

   return nalu_hypre_error_flag;
}
