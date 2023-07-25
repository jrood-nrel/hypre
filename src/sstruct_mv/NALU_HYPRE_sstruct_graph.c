/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_SStructGraph interface
 *
 *****************************************************************************/

#include "_nalu_hypre_sstruct_mv.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGraphCreate( MPI_Comm             comm,
                          NALU_HYPRE_SStructGrid    grid,
                          NALU_HYPRE_SStructGraph  *graph_ptr )
{
   nalu_hypre_SStructGraph     *graph;
   NALU_HYPRE_Int               nparts;
   nalu_hypre_SStructStencil ***stencils;
   nalu_hypre_SStructPGrid    **pgrids;
   NALU_HYPRE_Int              *fem_nsparse;
   NALU_HYPRE_Int             **fem_sparse_i;
   NALU_HYPRE_Int             **fem_sparse_j;
   NALU_HYPRE_Int             **fem_entries;
   NALU_HYPRE_Int               nvars;
   NALU_HYPRE_Int               part, var;

   graph = nalu_hypre_TAlloc(nalu_hypre_SStructGraph,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SStructGraphComm(graph) = comm;
   nalu_hypre_SStructGraphNDim(graph) = nalu_hypre_SStructGridNDim(grid);
   nalu_hypre_SStructGridRef(grid, &nalu_hypre_SStructGraphGrid(graph));
   nalu_hypre_SStructGridRef(grid, &nalu_hypre_SStructGraphDomainGrid(graph));
   nparts = nalu_hypre_SStructGridNParts(grid);
   nalu_hypre_SStructGraphNParts(graph) = nparts;
   pgrids = nalu_hypre_SStructGridPGrids(grid);
   stencils = nalu_hypre_TAlloc(nalu_hypre_SStructStencil **,  nparts, NALU_HYPRE_MEMORY_HOST);
   fem_nsparse  = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nparts, NALU_HYPRE_MEMORY_HOST);
   fem_sparse_i = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nparts, NALU_HYPRE_MEMORY_HOST);
   fem_sparse_j = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nparts, NALU_HYPRE_MEMORY_HOST);
   fem_entries  = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nparts, NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      nvars = nalu_hypre_SStructPGridNVars(pgrids[part]);
      stencils[part]  = nalu_hypre_TAlloc(nalu_hypre_SStructStencil *,  nvars, NALU_HYPRE_MEMORY_HOST);
      fem_nsparse[part]  = 0;
      fem_sparse_i[part] = NULL;
      fem_sparse_j[part] = NULL;
      fem_entries[part]  = NULL;
      for (var = 0; var < nvars; var++)
      {
         stencils[part][var] = NULL;
      }
   }
   nalu_hypre_SStructGraphStencils(graph)   = stencils;
   nalu_hypre_SStructGraphFEMNSparse(graph) = fem_nsparse;
   nalu_hypre_SStructGraphFEMSparseJ(graph) = fem_sparse_i;
   nalu_hypre_SStructGraphFEMSparseI(graph) = fem_sparse_j;
   nalu_hypre_SStructGraphFEMEntries(graph) = fem_entries;

   nalu_hypre_SStructGraphNUVEntries(graph) = 0;
   nalu_hypre_SStructGraphIUVEntries(graph) = NULL;
   nalu_hypre_SStructGraphUVEntries(graph)  = NULL;
   nalu_hypre_SStructGraphUVESize(graph)    = 0;
   nalu_hypre_SStructGraphUEMaxSize(graph)  = 0;
   nalu_hypre_SStructGraphUVEOffsets(graph) = NULL;

   nalu_hypre_SStructGraphRefCount(graph)   = 1;
   nalu_hypre_SStructGraphObjectType(graph) = NALU_HYPRE_SSTRUCT;

   nalu_hypre_SStructGraphEntries(graph)    = NULL;
   nalu_hypre_SStructNGraphEntries(graph)   = 0;
   nalu_hypre_SStructAGraphEntries(graph)   = 0;

   *graph_ptr = graph;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGraphDestroy( NALU_HYPRE_SStructGraph graph )
{
   NALU_HYPRE_Int                 nparts;
   nalu_hypre_SStructPGrid      **pgrids;
   nalu_hypre_SStructStencil   ***stencils;
   NALU_HYPRE_Int                *fem_nsparse;
   NALU_HYPRE_Int               **fem_sparse_i;
   NALU_HYPRE_Int               **fem_sparse_j;
   NALU_HYPRE_Int               **fem_entries;
   NALU_HYPRE_Int                 nUventries;
   NALU_HYPRE_Int                *iUventries;
   nalu_hypre_SStructUVEntry    **Uventries;
   nalu_hypre_SStructUVEntry     *Uventry;
   NALU_HYPRE_BigInt            **Uveoffsets;
   nalu_hypre_SStructGraphEntry **graph_entries;
   NALU_HYPRE_Int                 nvars;
   NALU_HYPRE_Int                 part, var, i;

   if (graph)
   {
      nalu_hypre_SStructGraphRefCount(graph) --;
      if (nalu_hypre_SStructGraphRefCount(graph) == 0)
      {
         nparts   = nalu_hypre_SStructGraphNParts(graph);
         pgrids   = nalu_hypre_SStructGraphPGrids(graph);
         stencils = nalu_hypre_SStructGraphStencils(graph);
         fem_nsparse  = nalu_hypre_SStructGraphFEMNSparse(graph);
         fem_sparse_i = nalu_hypre_SStructGraphFEMSparseJ(graph);
         fem_sparse_j = nalu_hypre_SStructGraphFEMSparseI(graph);
         fem_entries  = nalu_hypre_SStructGraphFEMEntries(graph);
         nUventries = nalu_hypre_SStructGraphNUVEntries(graph);
         iUventries = nalu_hypre_SStructGraphIUVEntries(graph);
         Uventries  = nalu_hypre_SStructGraphUVEntries(graph);
         Uveoffsets = nalu_hypre_SStructGraphUVEOffsets(graph);
         for (part = 0; part < nparts; part++)
         {
            nvars = nalu_hypre_SStructPGridNVars(pgrids[part]);
            for (var = 0; var < nvars; var++)
            {
               NALU_HYPRE_SStructStencilDestroy(stencils[part][var]);
            }
            nalu_hypre_TFree(stencils[part], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(fem_sparse_i[part], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(fem_sparse_j[part], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(fem_entries[part], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(Uveoffsets[part], NALU_HYPRE_MEMORY_HOST);
         }
         NALU_HYPRE_SStructGridDestroy(nalu_hypre_SStructGraphGrid(graph));
         NALU_HYPRE_SStructGridDestroy(nalu_hypre_SStructGraphDomainGrid(graph));
         nalu_hypre_TFree(stencils, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(fem_nsparse, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(fem_sparse_i, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(fem_sparse_j, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(fem_entries, NALU_HYPRE_MEMORY_HOST);
         /* RDF: THREAD? */
         for (i = 0; i < nUventries; i++)
         {
            Uventry = Uventries[iUventries[i]];
            if (Uventry)
            {
               nalu_hypre_TFree(nalu_hypre_SStructUVEntryUEntries(Uventry), NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(Uventry, NALU_HYPRE_MEMORY_HOST);
            }
            Uventries[iUventries[i]] = NULL;
         }
         nalu_hypre_TFree(iUventries, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(Uventries, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(Uveoffsets, NALU_HYPRE_MEMORY_HOST);
         graph_entries = nalu_hypre_SStructGraphEntries(graph);
         for (i = 0; i < nalu_hypre_SStructNGraphEntries(graph); i++)
         {
            nalu_hypre_TFree(graph_entries[i], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(graph_entries, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(graph, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGraphSetDomainGrid( NALU_HYPRE_SStructGraph graph,
                                 NALU_HYPRE_SStructGrid  domain_grid)
{
   /* This should only decrement a reference counter */
   NALU_HYPRE_SStructGridDestroy(nalu_hypre_SStructGraphDomainGrid(graph));
   nalu_hypre_SStructGridRef(domain_grid, &nalu_hypre_SStructGraphDomainGrid(graph));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGraphSetStencil( NALU_HYPRE_SStructGraph   graph,
                              NALU_HYPRE_Int            part,
                              NALU_HYPRE_Int            var,
                              NALU_HYPRE_SStructStencil stencil )
{
   nalu_hypre_SStructStencilRef(stencil, &nalu_hypre_SStructGraphStencil(graph, part, var));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGraphSetFEM( NALU_HYPRE_SStructGraph graph,
                          NALU_HYPRE_Int          part )
{
   if (!nalu_hypre_SStructGraphFEMPNSparse(graph, part))
   {
      nalu_hypre_SStructGraphFEMPNSparse(graph, part) = -1;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGraphSetFEMSparsity( NALU_HYPRE_SStructGraph  graph,
                                  NALU_HYPRE_Int           part,
                                  NALU_HYPRE_Int           nsparse,
                                  NALU_HYPRE_Int          *sparsity )
{
   NALU_HYPRE_Int          *fem_sparse_i;
   NALU_HYPRE_Int          *fem_sparse_j;
   NALU_HYPRE_Int           s;

   nalu_hypre_SStructGraphFEMPNSparse(graph, part) = nsparse;
   fem_sparse_i = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nsparse, NALU_HYPRE_MEMORY_HOST);
   fem_sparse_j = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nsparse, NALU_HYPRE_MEMORY_HOST);
   for (s = 0; s < nsparse; s++)
   {
      fem_sparse_i[s] = sparsity[2 * s];
      fem_sparse_j[s] = sparsity[2 * s + 1];
   }
   nalu_hypre_SStructGraphFEMPSparseI(graph, part) = fem_sparse_i;
   nalu_hypre_SStructGraphFEMPSparseJ(graph, part) = fem_sparse_j;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *   THIS IS FOR A NON-OVERLAPPING GRID GRAPH.
 *
 *   Now we just keep track of calls to this function and do all the "work"
 *   in the assemble.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGraphAddEntries( NALU_HYPRE_SStructGraph   graph,
                              NALU_HYPRE_Int            part,
                              NALU_HYPRE_Int           *index,
                              NALU_HYPRE_Int            var,
                              NALU_HYPRE_Int            to_part,
                              NALU_HYPRE_Int           *to_index,
                              NALU_HYPRE_Int            to_var )
{
   nalu_hypre_SStructGrid        *grid      = nalu_hypre_SStructGraphGrid(graph);
   NALU_HYPRE_Int                 ndim      = nalu_hypre_SStructGridNDim(grid);

   nalu_hypre_SStructGraphEntry **entries   = nalu_hypre_SStructGraphEntries(graph);
   nalu_hypre_SStructGraphEntry  *new_entry;

   NALU_HYPRE_Int                 n_entries = nalu_hypre_SStructNGraphEntries(graph);
   NALU_HYPRE_Int                 a_entries = nalu_hypre_SStructAGraphEntries(graph);

   /* check storage */
   if (!a_entries)
   {
      a_entries = 1000;
      entries = nalu_hypre_TAlloc(nalu_hypre_SStructGraphEntry *,  a_entries, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_SStructAGraphEntries(graph) = a_entries;
      nalu_hypre_SStructGraphEntries(graph) = entries;
   }
   else if (n_entries >= a_entries)
   {
      a_entries += 1000;
      entries = nalu_hypre_TReAlloc(entries,  nalu_hypre_SStructGraphEntry *,  a_entries, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_SStructAGraphEntries(graph) = a_entries;
      nalu_hypre_SStructGraphEntries(graph) = entries;
   }

   /*save parameters to a new entry */

   new_entry = nalu_hypre_TAlloc(nalu_hypre_SStructGraphEntry,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SStructGraphEntryPart(new_entry) = part;
   nalu_hypre_SStructGraphEntryToPart(new_entry) = to_part;

   nalu_hypre_SStructGraphEntryVar(new_entry) = var;
   nalu_hypre_SStructGraphEntryToVar(new_entry) = to_var;

   nalu_hypre_CopyToCleanIndex(index, ndim, nalu_hypre_SStructGraphEntryIndex(new_entry));
   nalu_hypre_CopyToCleanIndex(
      to_index, ndim, nalu_hypre_SStructGraphEntryToIndex(new_entry));

   entries[n_entries] = new_entry;

   /* update count */
   n_entries++;
   nalu_hypre_SStructNGraphEntries(graph) = n_entries;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This routine mainly computes the column numbers for the non-stencil
 * graph entries (i.e., those created by GraphAddEntries calls).
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGraphAssemble( NALU_HYPRE_SStructGraph graph )
{

   MPI_Comm                  comm        = nalu_hypre_SStructGraphComm(graph);
   NALU_HYPRE_Int                 ndim        = nalu_hypre_SStructGraphNDim(graph);
   nalu_hypre_SStructGrid        *grid        = nalu_hypre_SStructGraphGrid(graph);
   nalu_hypre_SStructGrid        *dom_grid    = nalu_hypre_SStructGraphDomainGrid(graph);
   NALU_HYPRE_Int                 nparts      = nalu_hypre_SStructGraphNParts(graph);
   nalu_hypre_SStructStencil   ***stencils    = nalu_hypre_SStructGraphStencils(graph);
   NALU_HYPRE_Int                 nUventries;
   NALU_HYPRE_Int                *iUventries;
   nalu_hypre_SStructUVEntry    **Uventries;
   NALU_HYPRE_Int                 Uvesize;
   NALU_HYPRE_BigInt            **Uveoffsets;
   NALU_HYPRE_Int                 type        = nalu_hypre_SStructGraphObjectType(graph);
   nalu_hypre_SStructGraphEntry **add_entries = nalu_hypre_SStructGraphEntries(graph);
   NALU_HYPRE_Int                 n_add_entries = nalu_hypre_SStructNGraphEntries(graph);

   nalu_hypre_SStructPGrid       *pgrid;
   nalu_hypre_StructGrid         *sgrid;
   NALU_HYPRE_Int                 nvars;
   nalu_hypre_BoxArray           *boxes;
   nalu_hypre_Box                *box;
   NALU_HYPRE_Int                 vol, d;

   nalu_hypre_SStructGraphEntry  *new_entry;
   nalu_hypre_SStructUVEntry     *Uventry;
   NALU_HYPRE_Int                 nUentries;
   nalu_hypre_SStructUEntry      *Uentries;
   NALU_HYPRE_Int                 to_part;
   nalu_hypre_IndexRef            to_index;
   NALU_HYPRE_Int                 to_var;
   NALU_HYPRE_Int                 to_boxnum;
   NALU_HYPRE_Int                 to_proc;
   NALU_HYPRE_BigInt              Uverank, rank;
   nalu_hypre_BoxManEntry        *boxman_entry;

   NALU_HYPRE_Int                 nprocs, myproc;
   NALU_HYPRE_Int                 part, var;
   nalu_hypre_IndexRef            index;
   NALU_HYPRE_Int                 i, j;

   /* may need to re-do box managers for the AP*/
   nalu_hypre_BoxManager        ***managers = nalu_hypre_SStructGridBoxManagers(grid);
   nalu_hypre_BoxManager        ***new_managers = NULL;
   nalu_hypre_BoxManager          *orig_boxman;
   nalu_hypre_BoxManager          *new_boxman;

   NALU_HYPRE_Int                  global_n_add_entries;
   NALU_HYPRE_Int                  is_gather, k;

   nalu_hypre_BoxManEntry         *all_entries, *entry;
   NALU_HYPRE_Int                  num_entries;
   void                      *info;
   nalu_hypre_Box                 *bbox, *new_box;
   nalu_hypre_Box               ***new_gboxes, *new_gbox;
   NALU_HYPRE_Int                 *num_ghost;

   /*---------------------------------------------------------
    *  If AP, then may need to redo the box managers
    *
    *  Currently using bounding boxes based on the indexes in add_entries to
    *  determine which boxes to gather in the box managers.  We refer to these
    *  bounding boxes as "gather boxes" here (new_gboxes).  This should work
    *  well in most cases, but it does have the potential to cause lots of grid
    *  boxes to be gathered (hence lots of communication).
    *
    *  A better algorithm would use more care in computing gather boxes that
    *  aren't "too big", while not computing "too many" either (which can also
    *  be slow).  One approach might be to compute an octree with leaves that
    *  have the same volume as the maximum grid box volume.  The leaves would
    *  then serve as the gather boxes.  The number of gather boxes would then be
    *  on the order of the number of local grid boxes (assuming the add_entries
    *  are local, which is generally how they should be used).
    *---------------------------------------------------------*/

   new_box = nalu_hypre_BoxCreate(ndim);

   /* if any processor has added entries, then all need to participate */

   nalu_hypre_MPI_Allreduce(&n_add_entries, &global_n_add_entries,
                       1, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_SUM, comm);

   if (global_n_add_entries > 0 )
   {
      /* create new managers */
      new_managers = nalu_hypre_TAlloc(nalu_hypre_BoxManager **,  nparts, NALU_HYPRE_MEMORY_HOST);
      new_gboxes = nalu_hypre_TAlloc(nalu_hypre_Box **,  nparts, NALU_HYPRE_MEMORY_HOST);

      for (part = 0; part < nparts; part++)
      {
         pgrid = nalu_hypre_SStructGridPGrid(grid, part);
         nvars = nalu_hypre_SStructPGridNVars(pgrid);

         new_managers[part] = nalu_hypre_TAlloc(nalu_hypre_BoxManager *,  nvars, NALU_HYPRE_MEMORY_HOST);
         new_gboxes[part] = nalu_hypre_TAlloc(nalu_hypre_Box *,  nvars, NALU_HYPRE_MEMORY_HOST);

         for (var = 0; var < nvars; var++)
         {
            sgrid = nalu_hypre_SStructPGridSGrid(pgrid, var);

            orig_boxman = managers[part][var];
            bbox =  nalu_hypre_BoxManBoundingBox(orig_boxman);

            nalu_hypre_BoxManCreate(nalu_hypre_BoxManNEntries(orig_boxman),
                               nalu_hypre_BoxManEntryInfoSize(orig_boxman),
                               nalu_hypre_StructGridNDim(sgrid), bbox,
                               nalu_hypre_StructGridComm(sgrid),
                               &new_managers[part][var]);
            /* create gather box with flipped bounding box extents */
            new_gboxes[part][var] = nalu_hypre_BoxCreate(ndim);
            nalu_hypre_BoxSetExtents(new_gboxes[part][var],
                                nalu_hypre_BoxIMax(bbox), nalu_hypre_BoxIMin(bbox));


            /* need to set the num ghost for new manager also */
            num_ghost = nalu_hypre_StructGridNumGhost(sgrid);
            nalu_hypre_BoxManSetNumGhost(new_managers[part][var], num_ghost);
         }
      } /* end loop over parts */

      /* now go through the local add entries */
      for (j = 0; j < n_add_entries; j++)
      {
         new_entry = add_entries[j];

         /* check part, var, index, to_part, to_var, to_index */
         for (k = 0; k < 2; k++)
         {
            switch (k)
            {
               case 0:
                  part =  nalu_hypre_SStructGraphEntryPart(new_entry);
                  var = nalu_hypre_SStructGraphEntryVar(new_entry);
                  index = nalu_hypre_SStructGraphEntryIndex(new_entry);
                  break;
               case 1:
                  part =  nalu_hypre_SStructGraphEntryToPart(new_entry) ;
                  var =  nalu_hypre_SStructGraphEntryToVar(new_entry);
                  index = nalu_hypre_SStructGraphEntryToIndex(new_entry);
                  break;
            }

            /* if the index is not within the bounds of the struct grid bounding
               box (which has been set in the box manager) then there should not
               be a coupling here (doesn't make sense) */

            new_boxman = new_managers[part][var];
            new_gbox = new_gboxes[part][var];
            bbox =  nalu_hypre_BoxManBoundingBox(new_boxman);

            if (nalu_hypre_IndexInBox(index, bbox) != 0)
            {
               /* compute new gather box extents based on index */
               for (d = 0; d < ndim; d++)
               {
                  nalu_hypre_BoxIMinD(new_gbox, d) =
                     nalu_hypre_min(nalu_hypre_BoxIMinD(new_gbox, d), nalu_hypre_IndexD(index, d));
                  nalu_hypre_BoxIMaxD(new_gbox, d) =
                     nalu_hypre_max(nalu_hypre_BoxIMaxD(new_gbox, d), nalu_hypre_IndexD(index, d));
               }
            }
         }
      }

      /* Now go through the managers and if gather has been called (on any
         processor) then populate the new manager with the entries from the old
         manager and then assemble and delete the old manager. */
      for (part = 0; part < nparts; part++)
      {
         pgrid = nalu_hypre_SStructGridPGrid(grid, part);
         nvars = nalu_hypre_SStructPGridNVars(pgrid);

         for (var = 0; var < nvars; var++)
         {
            new_boxman = new_managers[part][var];
            new_gbox = new_gboxes[part][var];

            /* call gather if non-empty gather box */
            if (nalu_hypre_BoxVolume(new_gbox) > 0)
            {
               nalu_hypre_BoxManGatherEntries(
                  new_boxman, nalu_hypre_BoxIMin(new_gbox), nalu_hypre_BoxIMax(new_gbox));
            }

            /* check to see if gather was called by some processor */
            nalu_hypre_BoxManGetGlobalIsGatherCalled(new_boxman, comm, &is_gather);
            if (is_gather)
            {
               /* copy orig boxman information to the new boxman*/

               orig_boxman = managers[part][var];

               nalu_hypre_BoxManGetAllEntries(orig_boxman, &num_entries, &all_entries);

               for (j = 0; j < num_entries; j++)
               {
                  entry = &all_entries[j];

                  nalu_hypre_BoxManEntryGetInfo(entry, &info);

                  nalu_hypre_BoxManAddEntry(new_boxman,
                                       nalu_hypre_BoxManEntryIMin(entry),
                                       nalu_hypre_BoxManEntryIMax(entry),
                                       nalu_hypre_BoxManEntryProc(entry),
                                       nalu_hypre_BoxManEntryId(entry),
                                       info);
               }

               /* call assemble for new boxmanager*/
               nalu_hypre_BoxManAssemble(new_boxman);

               /* TEMP for testing
                  if (nalu_hypre_BoxManNEntries(new_boxman) != num_entries)
                  {
                  nalu_hypre_MPI_Comm_rank(comm, &myproc);
                  nalu_hypre_printf("myid = %d, new_entries = %d, old entries = %d\n", myproc, nalu_hypre_BoxManNEntries(new_boxman), num_entries);
                  } */

               /* destroy old manager */
               nalu_hypre_BoxManDestroy (managers[part][var]);
            }
            else /* no gather called */
            {
               /*leave the old manager (so destroy the new one)  */
               nalu_hypre_BoxManDestroy(new_boxman);

               /*copy the old to the new */
               new_managers[part][var] = managers[part][var];
            }

            nalu_hypre_BoxDestroy(new_gboxes[part][var]);
         } /* end of var loop */
         nalu_hypre_TFree(managers[part], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(new_gboxes[part], NALU_HYPRE_MEMORY_HOST);
      } /* end of part loop */
      nalu_hypre_TFree(managers, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(new_gboxes, NALU_HYPRE_MEMORY_HOST);

      /* assign the new ones */
      nalu_hypre_SStructGridBoxManagers(grid) = new_managers;
   }

   /* clean up */
   nalu_hypre_BoxDestroy(new_box);

   /* end of AP stuff */

   nalu_hypre_MPI_Comm_size(comm, &nprocs);
   nalu_hypre_MPI_Comm_rank(comm, &myproc);

   /*---------------------------------------------------------
    * Set up UVEntries and iUventries
    *---------------------------------------------------------*/

   /* first set up Uvesize and Uveoffsets */

   Uvesize = 0;
   Uveoffsets = nalu_hypre_TAlloc(NALU_HYPRE_BigInt *,  nparts, NALU_HYPRE_MEMORY_HOST);
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);
      Uveoffsets[part] = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  nvars, NALU_HYPRE_MEMORY_HOST);
      for (var = 0; var < nvars; var++)
      {
         Uveoffsets[part][var] = Uvesize;
         sgrid = nalu_hypre_SStructPGridSGrid(pgrid, var);
         boxes = nalu_hypre_StructGridBoxes(sgrid);
         nalu_hypre_ForBoxI(i, boxes)
         {
            box = nalu_hypre_BoxArrayBox(boxes, i);
            vol = 1;
            for (d = 0; d < ndim; d++)
            {
               vol *= (nalu_hypre_BoxSizeD(box, d) + 2);
            }
            Uvesize += vol;
         }
      }
   }
   nalu_hypre_SStructGraphUVESize(graph)    = Uvesize;
   nalu_hypre_SStructGraphUVEOffsets(graph) = Uveoffsets;

   /* now set up nUventries, iUventries, and Uventries */

   iUventries = nalu_hypre_TAlloc(NALU_HYPRE_Int,  n_add_entries, NALU_HYPRE_MEMORY_HOST);
   Uventries = nalu_hypre_CTAlloc(nalu_hypre_SStructUVEntry *,  Uvesize, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SStructGraphIUVEntries(graph) = iUventries;
   nalu_hypre_SStructGraphUVEntries(graph)  = Uventries;

   nUventries = 0;

   /* go through each entry that was added */
   for (j = 0; j < n_add_entries; j++)
   {
      new_entry = add_entries[j];

      part =  nalu_hypre_SStructGraphEntryPart(new_entry);
      var = nalu_hypre_SStructGraphEntryVar(new_entry);
      index = nalu_hypre_SStructGraphEntryIndex(new_entry);
      to_part =  nalu_hypre_SStructGraphEntryToPart(new_entry) ;
      to_var =  nalu_hypre_SStructGraphEntryToVar(new_entry);
      to_index = nalu_hypre_SStructGraphEntryToIndex(new_entry);

      /* compute location (rank) for Uventry */
      nalu_hypre_SStructGraphGetUVEntryRank(graph, part, var, index, &Uverank);

      if (Uverank > -1)
      {
         iUventries[nUventries] = Uverank;

         if (Uventries[Uverank] == NULL)
         {
            Uventry = nalu_hypre_TAlloc(nalu_hypre_SStructUVEntry,  1, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_SStructUVEntryPart(Uventry) = part;
            nalu_hypre_CopyIndex(index, nalu_hypre_SStructUVEntryIndex(Uventry));
            nalu_hypre_SStructUVEntryVar(Uventry) = var;
            nalu_hypre_SStructGridFindBoxManEntry(grid, part, index, var, &boxman_entry);
            nalu_hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index, &rank, type);
            nalu_hypre_SStructUVEntryRank(Uventry) = rank;
            nUentries = 1;
            Uentries = nalu_hypre_TAlloc(nalu_hypre_SStructUEntry,  nUentries, NALU_HYPRE_MEMORY_HOST);
         }
         else
         {
            Uventry = Uventries[Uverank];
            nUentries = nalu_hypre_SStructUVEntryNUEntries(Uventry) + 1;
            Uentries = nalu_hypre_SStructUVEntryUEntries(Uventry);
            Uentries = nalu_hypre_TReAlloc(Uentries,  nalu_hypre_SStructUEntry,  nUentries, NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_SStructUVEntryNUEntries(Uventry) = nUentries;
         nalu_hypre_SStructUVEntryUEntries(Uventry)  = Uentries;
         nalu_hypre_SStructGraphUEMaxSize(graph) =
            nalu_hypre_max(nalu_hypre_SStructGraphUEMaxSize(graph), nUentries);

         i = nUentries - 1;
         nalu_hypre_SStructUVEntryToPart(Uventry, i) = to_part;
         nalu_hypre_CopyIndex(to_index, nalu_hypre_SStructUVEntryToIndex(Uventry, i));
         nalu_hypre_SStructUVEntryToVar(Uventry, i) = to_var;

         nalu_hypre_SStructGridFindBoxManEntry(
            dom_grid, to_part, to_index, to_var, &boxman_entry);
         nalu_hypre_SStructBoxManEntryGetBoxnum(boxman_entry, &to_boxnum);
         nalu_hypre_SStructUVEntryToBoxnum(Uventry, i) = to_boxnum;
         nalu_hypre_SStructBoxManEntryGetProcess(boxman_entry, &to_proc);
         nalu_hypre_SStructUVEntryToProc(Uventry, i) = to_proc;
         nalu_hypre_SStructBoxManEntryGetGlobalRank(
            boxman_entry, to_index, &rank, type);
         nalu_hypre_SStructUVEntryToRank(Uventry, i) = rank;

         Uventries[Uverank] = Uventry;

         nUventries++;
         nalu_hypre_SStructGraphNUVEntries(graph) = nUventries;

         nalu_hypre_SStructGraphUVEntries(graph) = Uventries;
      }
   } /* end of loop through add entries */

   /*---------------------------------------------------------
    * Set up the FEM stencil information
    *---------------------------------------------------------*/

   for (part = 0; part < nparts; part++)
   {
      /* only do this if SetFEM was called */
      if (nalu_hypre_SStructGraphFEMPNSparse(graph, part))
      {
         NALU_HYPRE_Int     fem_nsparse  = nalu_hypre_SStructGraphFEMPNSparse(graph, part);
         NALU_HYPRE_Int    *fem_sparse_i = nalu_hypre_SStructGraphFEMPSparseI(graph, part);
         NALU_HYPRE_Int    *fem_sparse_j = nalu_hypre_SStructGraphFEMPSparseJ(graph, part);
         NALU_HYPRE_Int    *fem_entries  = nalu_hypre_SStructGraphFEMPEntries(graph, part);
         NALU_HYPRE_Int     fem_nvars    = nalu_hypre_SStructGridFEMPNVars(grid, part);
         NALU_HYPRE_Int    *fem_vars     = nalu_hypre_SStructGridFEMPVars(grid, part);
         nalu_hypre_Index  *fem_offsets  = nalu_hypre_SStructGridFEMPOffsets(grid, part);
         nalu_hypre_Index   offset;
         NALU_HYPRE_Int     s, iv, jv, d, nvars, entry;
         NALU_HYPRE_Int    *stencil_sizes;
         nalu_hypre_Index **stencil_offsets;
         NALU_HYPRE_Int   **stencil_vars;

         nvars = nalu_hypre_SStructPGridNVars(nalu_hypre_SStructGridPGrid(grid, part));

         /* build default full sparsity pattern if nothing set by user */
         if (fem_nsparse < 0)
         {
            fem_nsparse = fem_nvars * fem_nvars;
            fem_sparse_i = nalu_hypre_TAlloc(NALU_HYPRE_Int,  fem_nsparse, NALU_HYPRE_MEMORY_HOST);
            fem_sparse_j = nalu_hypre_TAlloc(NALU_HYPRE_Int,  fem_nsparse, NALU_HYPRE_MEMORY_HOST);
            s = 0;
            for (i = 0; i < fem_nvars; i++)
            {
               for (j = 0; j < fem_nvars; j++)
               {
                  fem_sparse_i[s] = i;
                  fem_sparse_j[s] = j;
                  s++;
               }
            }
            nalu_hypre_SStructGraphFEMPNSparse(graph, part) = fem_nsparse;
            nalu_hypre_SStructGraphFEMPSparseI(graph, part) = fem_sparse_i;
            nalu_hypre_SStructGraphFEMPSparseJ(graph, part) = fem_sparse_j;
         }

         fem_entries = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  fem_nsparse, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_SStructGraphFEMPEntries(graph, part) = fem_entries;

         stencil_sizes   = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nvars, NALU_HYPRE_MEMORY_HOST);
         stencil_offsets = nalu_hypre_CTAlloc(nalu_hypre_Index *,  nvars, NALU_HYPRE_MEMORY_HOST);
         stencil_vars    = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nvars, NALU_HYPRE_MEMORY_HOST);
         for (iv = 0; iv < nvars; iv++)
         {
            stencil_offsets[iv] = nalu_hypre_CTAlloc(nalu_hypre_Index,  fem_nvars * fem_nvars, NALU_HYPRE_MEMORY_HOST);
            stencil_vars[iv]    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  fem_nvars * fem_nvars, NALU_HYPRE_MEMORY_HOST);
         }

         for (s = 0; s < fem_nsparse; s++)
         {
            i = fem_sparse_i[s];
            j = fem_sparse_j[s];
            iv = fem_vars[i];
            jv = fem_vars[j];

            /* shift off-diagonal offset by diagonal */
            for (d = 0; d < ndim; d++)
            {
               offset[d] = fem_offsets[j][d] - fem_offsets[i][d];
            }

            /* search stencil_offsets */
            for (entry = 0; entry < stencil_sizes[iv]; entry++)
            {
               /* if offset is already in the stencil, break */
               if ( nalu_hypre_IndexesEqual(offset, stencil_offsets[iv][entry], ndim)
                    && (jv == stencil_vars[iv][entry]) )
               {
                  break;
               }
            }
            /* if this is a new stencil offset, add it to the stencil */
            if (entry == stencil_sizes[iv])
            {
               for (d = 0; d < ndim; d++)
               {
                  stencil_offsets[iv][entry][d] = offset[d];
               }
               stencil_vars[iv][entry] = jv;
               stencil_sizes[iv]++;
            }

            fem_entries[s] = entry;
         }

         /* set up the stencils */
         for (iv = 0; iv < nvars; iv++)
         {
            NALU_HYPRE_SStructStencilDestroy(stencils[part][iv]);
            NALU_HYPRE_SStructStencilCreate(ndim, stencil_sizes[iv],
                                       &stencils[part][iv]);
            for (entry = 0; entry < stencil_sizes[iv]; entry++)
            {
               NALU_HYPRE_SStructStencilSetEntry(stencils[part][iv], entry,
                                            stencil_offsets[iv][entry],
                                            stencil_vars[iv][entry]);
            }
         }

         /* free up temporary stuff */
         for (iv = 0; iv < nvars; iv++)
         {
            nalu_hypre_TFree(stencil_offsets[iv], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(stencil_vars[iv], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(stencil_sizes, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(stencil_offsets, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(stencil_vars, NALU_HYPRE_MEMORY_HOST);
      }
   }

   /*---------------------------------------------------------
    * Sort the iUventries array and eliminate duplicates.
    *---------------------------------------------------------*/

   if (nUventries > 1)
   {
      nalu_hypre_qsort0(iUventries, 0, nUventries - 1);

      j = 1;
      for (i = 1; i < nUventries; i++)
      {
         if (iUventries[i] > iUventries[i - 1])
         {
            iUventries[j] = iUventries[i];
            j++;
         }
      }
      nUventries = j;
      nalu_hypre_SStructGraphNUVEntries(graph) = nUventries;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGraphSetObjectType( NALU_HYPRE_SStructGraph  graph,
                                 NALU_HYPRE_Int           type )
{
   nalu_hypre_SStructGraphObjectType(graph) = type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGraphPrint( FILE *file, NALU_HYPRE_SStructGraph graph )
{
   NALU_HYPRE_Int                 type = nalu_hypre_SStructGraphObjectType(graph);
   NALU_HYPRE_Int                 ndim = nalu_hypre_SStructGraphNDim(graph);
   NALU_HYPRE_Int                 nentries = nalu_hypre_SStructNGraphEntries(graph);
   nalu_hypre_SStructGraphEntry **entries = nalu_hypre_SStructGraphEntries(graph);
   NALU_HYPRE_Int                 part, to_part;
   NALU_HYPRE_Int                 var, to_var;
   nalu_hypre_IndexRef            index, to_index;

   NALU_HYPRE_Int                 i;

   /* Print auxiliary info */
   nalu_hypre_fprintf(file, "GraphSetObjectType: %d\n", type);

   /* Print SStructGraphEntry info */
   nalu_hypre_fprintf(file, "GraphNumEntries: %d", nentries);
   for (i = 0; i < nentries; i++)
   {
      part = nalu_hypre_SStructGraphEntryPart(entries[i]);
      var = nalu_hypre_SStructGraphEntryVar(entries[i]);
      index = nalu_hypre_SStructGraphEntryIndex(entries[i]);
      to_part = nalu_hypre_SStructGraphEntryToPart(entries[i]);
      to_var = nalu_hypre_SStructGraphEntryToVar(entries[i]);
      to_index = nalu_hypre_SStructGraphEntryToIndex(entries[i]);

      nalu_hypre_fprintf(file, "\nGraphAddEntries: %d %d ", part, var);
      nalu_hypre_IndexPrint(file, ndim, index);
      nalu_hypre_fprintf(file, " %d %d ", to_part, to_var);
      nalu_hypre_IndexPrint(file, ndim, to_index);
   }
   nalu_hypre_fprintf(file, "\n");

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGraphRead( FILE                  *file,
                        NALU_HYPRE_SStructGrid      grid,
                        NALU_HYPRE_SStructStencil **stencils,
                        NALU_HYPRE_SStructGraph    *graph_ptr )
{
   MPI_Comm                  comm = nalu_hypre_SStructGridComm(grid);
   NALU_HYPRE_Int                 nparts = nalu_hypre_SStructGridNParts(grid);
   NALU_HYPRE_Int                 ndim = nalu_hypre_SStructGridNDim(grid);

   NALU_HYPRE_SStructGraph        graph;
   nalu_hypre_SStructGraphEntry **entries;
   nalu_hypre_SStructPGrid       *pgrid;
   NALU_HYPRE_Int                 nentries;
   NALU_HYPRE_Int                 a_entries;
   NALU_HYPRE_Int                 part, to_part;
   NALU_HYPRE_Int                 var, to_var;
   nalu_hypre_Index               index, to_index;

   NALU_HYPRE_Int                 type;
   NALU_HYPRE_Int                 nvars;
   NALU_HYPRE_Int                 i;

   /* Create graph */
   NALU_HYPRE_SStructGraphCreate(comm, grid, &graph);

   /* Read auxiliary info */
   nalu_hypre_fscanf(file, "GraphSetObjectType: %d\n", &type);
   NALU_HYPRE_SStructGraphSetObjectType(graph, type);

   /* Set stencils */
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);

      for (var = 0; var < nvars; var++)
      {
         NALU_HYPRE_SStructGraphSetStencil(graph, part, var, stencils[part][var]);
      }
   }

   /* TODO: NALU_HYPRE_SStructGraphSetFEM */
   /* TODO: NALU_HYPRE_SStructGraphSetFEMSparsity */

   /* Read SStructGraphEntry info */
   nalu_hypre_fscanf(file, "GraphNumEntries: %d", &nentries);
   a_entries = nentries + 1;
   nalu_hypre_SStructAGraphEntries(graph) = a_entries;
   entries = nalu_hypre_CTAlloc(nalu_hypre_SStructGraphEntry *, a_entries, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SStructGraphEntries(graph) = entries;
   for (i = 0; i < nentries; i++)
   {
      nalu_hypre_fscanf(file, "\nGraphAddEntries: %d %d ", &part, &var);
      nalu_hypre_IndexRead(file, ndim, index);
      nalu_hypre_fscanf(file, " %d %d ", &to_part, &to_var);
      nalu_hypre_IndexRead(file, ndim, to_index);

      NALU_HYPRE_SStructGraphAddEntries(graph, part, index, var, to_part, to_index, to_var);
   }
   nalu_hypre_fscanf(file, "\n");

   *graph_ptr = graph;

   return nalu_hypre_error_flag;
}
