/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * OpenMP Problems
 *
 * Need to fix the way these variables are set and incremented in loops:
 *   i, nrows (only where they are listed at the end of SMP_PRIVATE)
 *
 * Are private static arrays a problem?
 *
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_Maxwell_Grad.c
 *   Forms a node-to-edge gradient operator. Looping over the
 *   edge grid so that each processor fills up only its own rows. Each
 *   processor will have its processor interface nodal ranks.
 *   Loops over two types of boxes, interior of grid boxes and boundary
 *   of boxes. Algo:
 *       find all nodal and edge physical boundary points and set
 *       the appropriate flag to be 0 at a boundary dof.
 *       set -1's in value array
 *       for each edge box,
 *       for interior
 *       {
 *          connect edge ijk (row) to nodes (col) connected to this edge
 *          and change -1 to 1 if needed;
 *       }
 *       for boundary layers
 *       {
 *          if edge not on the physical boundary connect only the nodes
 *          that are not on the physical boundary
 *       }
 *       set parcsr matrix with values;
 *
 * Note that the nodes that are on the processor interface can be
 * on the physical boundary. But the off-proc edges connected to this
 * type of node will be a physical boundary edge.
 *
 *--------------------------------------------------------------------------*/
nalu_hypre_ParCSRMatrix *
nalu_hypre_Maxwell_Grad(nalu_hypre_SStructGrid *grid)
{
   MPI_Comm               comm = (grid ->  comm);

   NALU_HYPRE_IJMatrix         T_grad;
   nalu_hypre_ParCSRMatrix    *parcsr_grad;
   NALU_HYPRE_Int              matrix_type = NALU_HYPRE_PARCSR;

   nalu_hypre_SStructGrid     *node_grid, *edge_grid;

   nalu_hypre_SStructPGrid    *pgrid;
   nalu_hypre_StructGrid      *var_grid;
   nalu_hypre_BoxArray        *boxes, *tmp_box_array1, *tmp_box_array2;
   nalu_hypre_BoxArray        *edge_boxes, *cell_boxes;
   nalu_hypre_Box             *box, *cell_box;
   nalu_hypre_Box              layer, interior_box;
   nalu_hypre_Box             *box_piece;

   nalu_hypre_BoxManager      *boxman;
   nalu_hypre_BoxManEntry     *entry;

   NALU_HYPRE_BigInt          *inode, *jedge;
   NALU_HYPRE_Int              nrows, nnodes, *nflag, *eflag, *ncols;
   NALU_HYPRE_Real            *vals;

   nalu_hypre_Index            index;
   nalu_hypre_Index            loop_size, start, lindex;
   nalu_hypre_Index            shift, shift2;
   nalu_hypre_Index           *offsets, *varoffsets;

   NALU_HYPRE_Int              nparts = nalu_hypre_SStructGridNParts(grid);
   NALU_HYPRE_Int              ndim  = nalu_hypre_SStructGridNDim(grid);

   NALU_HYPRE_SStructVariable  vartype_node, *vartype_edges;
   NALU_HYPRE_SStructVariable *vartypes;

   NALU_HYPRE_Int              nvars, part;

   NALU_HYPRE_BigInt           m;
   NALU_HYPRE_Int              i, j, k, n, d;
   NALU_HYPRE_Int             *direction, ndirection;

   NALU_HYPRE_BigInt           ilower, iupper;
   NALU_HYPRE_BigInt           jlower, jupper;

   NALU_HYPRE_BigInt           start_rank1, start_rank2, rank;
   NALU_HYPRE_Int              myproc;

   NALU_HYPRE_MemoryLocation   memory_location;

   nalu_hypre_BoxInit(&layer, ndim);
   nalu_hypre_BoxInit(&interior_box, ndim);

   nalu_hypre_MPI_Comm_rank(comm, &myproc);

   nalu_hypre_ClearIndex(shift);
   for (i = 0; i < ndim; i++)
   {
      nalu_hypre_IndexD(shift, i) = -1;
   }

   /* To get the correct ranks, separate node & edge grids must be formed.
      Note that the edge vars must be ordered the same way as is in grid.*/
   NALU_HYPRE_SStructGridCreate(comm, ndim, nparts, &node_grid);
   NALU_HYPRE_SStructGridCreate(comm, ndim, nparts, &edge_grid);

   vartype_node = NALU_HYPRE_SSTRUCT_VARIABLE_NODE;
   vartype_edges = nalu_hypre_TAlloc(NALU_HYPRE_SStructVariable,  ndim, NALU_HYPRE_MEMORY_HOST);

   /* Assuming the same edge variable types on all parts */
   pgrid   = nalu_hypre_SStructGridPGrid(grid, 0);
   vartypes = nalu_hypre_SStructPGridVarTypes(pgrid);
   nvars   = nalu_hypre_SStructPGridNVars(pgrid);

   k = 0;
   for (i = 0; i < nvars; i++)
   {
      j = vartypes[i];
      switch (j)
      {
         case 2:
         {
            vartype_edges[k] = NALU_HYPRE_SSTRUCT_VARIABLE_XFACE;
            k++;
            break;
         }

         case 3:
         {
            vartype_edges[k] = NALU_HYPRE_SSTRUCT_VARIABLE_YFACE;
            k++;
            break;
         }

         case 5:
         {
            vartype_edges[k] = NALU_HYPRE_SSTRUCT_VARIABLE_XEDGE;
            k++;
            break;
         }

         case 6:
         {
            vartype_edges[k] = NALU_HYPRE_SSTRUCT_VARIABLE_YEDGE;
            k++;
            break;
         }

         case 7:
         {
            vartype_edges[k] = NALU_HYPRE_SSTRUCT_VARIABLE_ZEDGE;
            k++;
            break;
         }

      }  /* switch(j) */
   }     /* for (i= 0; i< nvars; i++) */

   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      var_grid = nalu_hypre_SStructPGridCellSGrid(pgrid) ;

      boxes = nalu_hypre_StructGridBoxes(var_grid);
      nalu_hypre_ForBoxI(j, boxes)
      {
         box = nalu_hypre_BoxArrayBox(boxes, j);
         NALU_HYPRE_SStructGridSetExtents(node_grid, part,
                                     nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box));
         NALU_HYPRE_SStructGridSetExtents(edge_grid, part,
                                     nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box));
      }
      NALU_HYPRE_SStructGridSetVariables(node_grid, part, 1, &vartype_node);
      NALU_HYPRE_SStructGridSetVariables(edge_grid, part, ndim, vartype_edges);
   }
   NALU_HYPRE_SStructGridAssemble(node_grid);
   NALU_HYPRE_SStructGridAssemble(edge_grid);

   /* CREATE IJ_MATRICES- need to find the size of each one. Notice that the row
      and col ranks of these matrices can be created using only grid information.
      Grab the first part, first variable, first box, and lower index (lower rank);
      Grab the last part, last variable, last box, and upper index (upper rank). */

   /* Grad: node(col) -> edge(row). Same for 2-d and 3-d */
   /* lower rank */
   part = 0;
   i   = 0;

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(edge_grid, part, 0, i, myproc, &entry);
   pgrid   = nalu_hypre_SStructGridPGrid(edge_grid, part);
   var_grid = nalu_hypre_SStructPGridSGrid(pgrid, 0);
   boxes   = nalu_hypre_StructGridBoxes(var_grid);
   box     = nalu_hypre_BoxArrayBox(boxes, 0);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(box), &ilower);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(node_grid, part, 0, i, myproc, &entry);
   pgrid   = nalu_hypre_SStructGridPGrid(node_grid, part);
   var_grid = nalu_hypre_SStructPGridSGrid(pgrid, 0);
   boxes   = nalu_hypre_StructGridBoxes(var_grid);
   box     = nalu_hypre_BoxArrayBox(boxes, 0);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(box), &jlower);

   /* upper rank */
   part = nparts - 1;

   pgrid   = nalu_hypre_SStructGridPGrid(edge_grid, part);
   nvars   = nalu_hypre_SStructPGridNVars(pgrid);
   var_grid = nalu_hypre_SStructPGridSGrid(pgrid, nvars - 1);
   boxes   = nalu_hypre_StructGridBoxes(var_grid);
   box     = nalu_hypre_BoxArrayBox(boxes, nalu_hypre_BoxArraySize(boxes) - 1);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(edge_grid, part, nvars - 1,
                                           nalu_hypre_BoxArraySize(boxes) - 1, myproc,
                                           &entry);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(box), &iupper);

   pgrid   = nalu_hypre_SStructGridPGrid(node_grid, part);
   nvars   = nalu_hypre_SStructPGridNVars(pgrid);
   var_grid = nalu_hypre_SStructPGridSGrid(pgrid, nvars - 1);
   boxes   = nalu_hypre_StructGridBoxes(var_grid);
   box     = nalu_hypre_BoxArrayBox(boxes, nalu_hypre_BoxArraySize(boxes) - 1);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(node_grid, part, nvars - 1,
                                           nalu_hypre_BoxArraySize(boxes) - 1, myproc,
                                           &entry);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(box), &jupper);

   NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &T_grad);
   NALU_HYPRE_IJMatrixSetObjectType(T_grad, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJMatrixInitialize(T_grad);

   memory_location = nalu_hypre_IJMatrixMemoryLocation(T_grad);

   /*------------------------------------------------------------------------------
    * fill up the parcsr matrix.
    *------------------------------------------------------------------------------*/

   /* count the no. of rows. Make sure repeated nodes along the boundaries are counted.*/
   nrows = 0;
   nnodes = 0;
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(edge_grid, part);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);
      for (m = 0; m < nvars; m++)
      {
         var_grid = nalu_hypre_SStructPGridSGrid(pgrid, m);
         boxes   = nalu_hypre_StructGridBoxes(var_grid);
         nalu_hypre_ForBoxI(j, boxes)
         {
            box = nalu_hypre_BoxArrayBox(boxes, j);
            /* make slightly bigger to handle any shared nodes */
            nalu_hypre_CopyBox(box, &layer);
            nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&layer), shift, 3, nalu_hypre_BoxIMin(&layer));
            nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMax(&layer), shift, 3, nalu_hypre_BoxIMax(&layer));
            nrows += nalu_hypre_BoxVolume(&layer);
         }
      }

      pgrid = nalu_hypre_SStructGridPGrid(node_grid, part);
      var_grid = nalu_hypre_SStructPGridSGrid(pgrid, 0); /* only one variable grid */
      boxes   = nalu_hypre_StructGridBoxes(var_grid);
      nalu_hypre_ForBoxI(j, boxes)
      {
         box = nalu_hypre_BoxArrayBox(boxes, j);
         /* make slightly bigger to handle any shared nodes */
         nalu_hypre_CopyBox(box, &layer);
         nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(&layer), shift, 3, nalu_hypre_BoxIMin(&layer));
         nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMax(&layer), shift, 3, nalu_hypre_BoxIMax(&layer));
         nnodes += nalu_hypre_BoxVolume(&layer);
      }
   }

   eflag = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows, NALU_HYPRE_MEMORY_HOST);
   nflag = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnodes, NALU_HYPRE_MEMORY_HOST);

   /* Set eflag to have the number of nodes connected to an edge (2) and
      nflag to have the number of edges connect to a node. */
   for (i = 0; i < nrows; i++)
   {
      eflag[i] = 2;
   }
   j = 2 * ndim;
   for (i = 0; i < nnodes; i++)
   {
      nflag[i] = j;
   }

   /* Determine physical boundary points. Get the rank and set flag[rank]= 0.
      This will boundary dof, i.e., flag[rank]= 0 will flag a boundary dof. */

   start_rank1 = nalu_hypre_SStructGridStartRank(node_grid);
   start_rank2 = nalu_hypre_SStructGridStartRank(edge_grid);
   for (part = 0; part < nparts; part++)
   {
      /* node flag */
      pgrid   = nalu_hypre_SStructGridPGrid(node_grid, part);
      var_grid = nalu_hypre_SStructPGridSGrid(pgrid, 0);
      boxes   = nalu_hypre_StructGridBoxes(var_grid);
      boxman     = nalu_hypre_SStructGridBoxManager(node_grid, part, 0);

      nalu_hypre_ForBoxI(j, boxes)
      {
         box = nalu_hypre_BoxArrayBox(boxes, j);
         nalu_hypre_BoxManGetEntry(boxman, myproc, j, &entry);
         i = nalu_hypre_BoxVolume(box);

         tmp_box_array1 = nalu_hypre_BoxArrayCreate(0, ndim);
         nalu_hypre_BoxBoundaryG(box, var_grid, tmp_box_array1);

         for (m = 0; m < nalu_hypre_BoxArraySize(tmp_box_array1); m++)
         {
            box_piece = nalu_hypre_BoxArrayBox(tmp_box_array1, m);
            if (nalu_hypre_BoxVolume(box_piece) < i)
            {
               nalu_hypre_BoxGetSize(box_piece, loop_size);
               nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(box_piece), start);

               nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
               {
                  zypre_BoxLoopGetIndex(lindex);
                  nalu_hypre_SetIndex3(index, lindex[0], lindex[1], lindex[2]);
                  nalu_hypre_AddIndexes(index, start, 3, index);

                  nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, index,
                                                        &rank, matrix_type);
                  nflag[rank - start_rank1] = 0;
               }
               nalu_hypre_SerialBoxLoop0End();
            }  /* if (nalu_hypre_BoxVolume(box_piece) < i) */

         }  /* for (m= 0; m< nalu_hypre_BoxArraySize(tmp_box_array1); m++) */
         nalu_hypre_BoxArrayDestroy(tmp_box_array1);

      }  /* nalu_hypre_ForBoxI(j, boxes) */

      /*-----------------------------------------------------------------
       * edge flag. Since we want only the edges that completely lie
       * on a boundary, whereas the boundary extraction routines mark
       * edges that touch the boundary, we need to call the boundary
       * routines in appropriate directions:
       *    2-d horizontal edges (y faces)- search in j directions
       *    2-d vertical edges (x faces)  - search in i directions
       *    3-d x edges                   - search in j,k directions
       *    3-d y edges                   - search in i,k directions
       *    3-d z edges                   - search in i,j directions
       *-----------------------------------------------------------------*/
      pgrid    = nalu_hypre_SStructGridPGrid(edge_grid, part);
      nvars    = nalu_hypre_SStructPGridNVars(pgrid);
      direction = nalu_hypre_TAlloc(NALU_HYPRE_Int,  2, NALU_HYPRE_MEMORY_HOST); /* only two directions at most */
      for (m = 0; m < nvars; m++)
      {
         var_grid = nalu_hypre_SStructPGridSGrid(pgrid, m);
         boxes   = nalu_hypre_StructGridBoxes(var_grid);
         boxman  = nalu_hypre_SStructGridBoxManager(edge_grid, part, m);

         j = vartype_edges[m];
         switch (j)
         {
            case 2: /* x faces, 2d */
            {
               ndirection  = 1;
               direction[0] = 0;
               break;
            }

            case 3: /* y faces, 2d */
            {
               ndirection  = 1;
               direction[0] = 1;
               break;
            }

            case 5: /* x edges, 3d */
            {
               ndirection  = 2;
               direction[0] = 1;
               direction[1] = 2;
               break;
            }

            case 6: /* y edges, 3d */
            {
               ndirection  = 2;
               direction[0] = 0;
               direction[1] = 2;
               break;
            }

            case 7: /* z edges, 3d */
            {
               ndirection  = 2;
               direction[0] = 0;
               direction[1] = 1;
               break;
            }
         }  /* switch(j) */

         nalu_hypre_ForBoxI(j, boxes)
         {
            box = nalu_hypre_BoxArrayBox(boxes, j);
            nalu_hypre_BoxManGetEntry(boxman, myproc, j, &entry);
            i = nalu_hypre_BoxVolume(box);

            for (d = 0; d < ndirection; d++)
            {
               tmp_box_array1 = nalu_hypre_BoxArrayCreate(0, ndim);
               tmp_box_array2 = nalu_hypre_BoxArrayCreate(0, ndim);
               nalu_hypre_BoxBoundaryDG(box, var_grid, tmp_box_array1,
                                   tmp_box_array2, direction[d]);

               for (k = 0; k < nalu_hypre_BoxArraySize(tmp_box_array1); k++)
               {
                  box_piece = nalu_hypre_BoxArrayBox(tmp_box_array1, k);
                  if (nalu_hypre_BoxVolume(box_piece) < i)
                  {
                     nalu_hypre_BoxGetSize(box_piece, loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(box_piece), start);

                     nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(index, lindex[0], lindex[1], lindex[2]);
                        nalu_hypre_AddIndexes(index, start, 3, index);

                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, index,
                                                              &rank, matrix_type);
                        eflag[rank - start_rank2] = 0;
                     }
                     nalu_hypre_SerialBoxLoop0End();
                  }  /* if (nalu_hypre_BoxVolume(box_piece) < i) */
               }     /* for (k= 0; k< nalu_hypre_BoxArraySize(tmp_box_array1); k++) */

               nalu_hypre_BoxArrayDestroy(tmp_box_array1);

               for (k = 0; k < nalu_hypre_BoxArraySize(tmp_box_array2); k++)
               {
                  box_piece = nalu_hypre_BoxArrayBox(tmp_box_array2, k);
                  if (nalu_hypre_BoxVolume(box_piece) < i)
                  {
                     nalu_hypre_BoxGetSize(box_piece, loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(box_piece), start);

                     nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(index, lindex[0], lindex[1], lindex[2]);
                        nalu_hypre_AddIndexes(index, start, 3, index);

                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, index,
                                                              &rank, matrix_type);
                        eflag[rank - start_rank2] = 0;
                     }
                     nalu_hypre_SerialBoxLoop0End();
                  }  /* if (nalu_hypre_BoxVolume(box_piece) < i) */
               }     /* for (k= 0; k< nalu_hypre_BoxArraySize(tmp_box_array2); k++) */
               nalu_hypre_BoxArrayDestroy(tmp_box_array2);
            }  /* for (d= 0; d< ndirection; d++) */

         }  /* nalu_hypre_ForBoxI(j, boxes) */
      }     /* for (m= 0; m< nvars; m++) */

      nalu_hypre_TFree(direction, NALU_HYPRE_MEMORY_HOST);
   }  /* for (part= 0; part< nparts; part++) */

   /* set vals. Will have more memory than is needed- extra allotted
      for repeated nodes. */
   inode = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nrows, memory_location);
   ncols = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows, memory_location);

   /* each row can have at most two columns */
   k = 2 * nrows;
   jedge = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, k, memory_location);
   vals = nalu_hypre_TAlloc(NALU_HYPRE_Real, k, memory_location);
   for (i = 0; i < k; i++)
   {
      vals[i] = -1.0;
   }

   /* to get the correct col connection to each node, we need to offset
      index ijk. Determine these. Assuming the same var ordering for each
      part. Note that these are not the variable offsets. */
   offsets   = nalu_hypre_TAlloc(nalu_hypre_Index,  ndim, NALU_HYPRE_MEMORY_HOST);
   varoffsets = nalu_hypre_TAlloc(nalu_hypre_Index,  ndim, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < ndim; i++)
   {
      j = vartype_edges[i];
      nalu_hypre_SStructVariableGetOffset(vartype_edges[i], ndim, varoffsets[i]);
      switch (j)
      {
         case 2:
         {
            nalu_hypre_SetIndex3(offsets[i], 0, 1, 0);
            break;
         }

         case 3:
         {
            nalu_hypre_SetIndex3(offsets[i], 1, 0, 0);
            break;
         }

         case 5:
         {
            nalu_hypre_SetIndex3(offsets[i], 1, 0, 0);
            break;
         }

         case 6:
         {
            nalu_hypre_SetIndex3(offsets[i], 0, 1, 0);
            break;
         }

         case 7:
         {
            nalu_hypre_SetIndex3(offsets[i], 0, 0, 1);
            break;
         }
      }   /*  switch(j) */
   }     /* for (i= 0; i< ndim; i++) */

   nrows = 0; i = 0;
   for (part = 0; part < nparts; part++)
   {
      /* grab boxarray for node rank extracting later */
      pgrid       = nalu_hypre_SStructGridPGrid(node_grid, part);
      var_grid    = nalu_hypre_SStructPGridSGrid(pgrid, 0);

      /* grab edge structures */
      pgrid     = nalu_hypre_SStructGridPGrid(edge_grid, part);

      /* the cell-centred reference box is used to get the correct
         interior edge box. For parallel distribution of the edge
         grid, simple contraction of the edge box does not get the
         correct interior edge box. Need to contract the cell box. */
      var_grid = nalu_hypre_SStructPGridCellSGrid(pgrid);
      cell_boxes = nalu_hypre_StructGridBoxes(var_grid);

      nvars     = nalu_hypre_SStructPGridNVars(pgrid);
      for (n = 0; n < nvars; n++)
      {
         var_grid  = nalu_hypre_SStructPGridSGrid(pgrid, n);
         edge_boxes = nalu_hypre_StructGridBoxes(var_grid);

         nalu_hypre_ForBoxI(j, edge_boxes)
         {
            box = nalu_hypre_BoxArrayBox(edge_boxes, j);
            cell_box = nalu_hypre_BoxArrayBox(cell_boxes, j);

            nalu_hypre_CopyBox(cell_box, &interior_box);

            /* shrink the cell_box to get the interior cell_box. All
               edges in the interior box should be on this proc. */
            nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&interior_box), shift, 3,
                                  nalu_hypre_BoxIMin(&interior_box));

            nalu_hypre_AddIndexes(nalu_hypre_BoxIMax(&interior_box), shift, 3,
                             nalu_hypre_BoxIMax(&interior_box));

            /* offset this to the variable interior box */
            nalu_hypre_CopyBox(&interior_box, &layer);
            nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&layer), varoffsets[n], 3,
                                  nalu_hypre_BoxIMin(&layer));

            nalu_hypre_BoxGetSize(&layer, loop_size);
            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&layer), start);

            /* Interior box- loop over each edge and find the row rank and
               then the column ranks for the connected nodes. Change the
               appropriate values to 1. */
            nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
            {
               zypre_BoxLoopGetIndex(lindex);
               nalu_hypre_SetIndex3(index, lindex[0], lindex[1], lindex[2]);
               nalu_hypre_AddIndexes(index, start, 3, index);

               /* edge ijk connected to nodes ijk & ijk-offsets. Interior edges
                  and so no boundary edges to consider. */
               nalu_hypre_SStructGridFindBoxManEntry(edge_grid, part, index, n,
                                                &entry);
               nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, index, &m, matrix_type);
               inode[nrows] = m;

               nalu_hypre_SStructGridFindBoxManEntry(node_grid, part, index, 0,
                                                &entry);
               nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, index, &m, matrix_type);
               jedge[i] = m;
               vals[i] = 1.0; /* change only this connection */
               i++;

               nalu_hypre_SubtractIndexes(index, offsets[n], 3, index);
               nalu_hypre_SStructGridFindBoxManEntry(node_grid, part, index, 0,
                                                &entry);
               nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, index, &m, matrix_type);
               jedge[i] = m;
               i++;

               ncols[nrows] = 2;
               nrows++;
            }
            nalu_hypre_SerialBoxLoop0End();

            /* now the boundary layers. To cases to consider: is the
               edge totally on the boundary or is the edge connected
               to the boundary. Need to check eflag & nflag. */
            for (d = 0; d < ndim; d++)
            {
               /*shift the layer box in the correct direction and distance.
                 distance= nalu_hypre_BoxIMax(box)[d]-nalu_hypre_BoxIMin(box)[d]+1-1
                 = nalu_hypre_BoxIMax(box)[d]-nalu_hypre_BoxIMin(box)[d] */
               nalu_hypre_ClearIndex(shift2);
               shift2[d] = nalu_hypre_BoxIMax(box)[d] - nalu_hypre_BoxIMin(box)[d];

               /* ndirection= 0 negative; ndirection= 1 positive */
               for (ndirection = 0; ndirection < 2; ndirection++)
               {
                  nalu_hypre_CopyBox(box, &layer);

                  if (ndirection)
                  {
                     nalu_hypre_BoxShiftPos(&layer, shift2);
                  }
                  else
                  {
                     nalu_hypre_BoxShiftNeg(&layer, shift2);
                  }

                  nalu_hypre_IntersectBoxes(box, &layer, &layer);
                  nalu_hypre_BoxGetSize(&layer, loop_size);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&layer), start);

                  nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
                  {
                     zypre_BoxLoopGetIndex(lindex);
                     nalu_hypre_SetIndex3(index, lindex[0], lindex[1], lindex[2]);
                     nalu_hypre_AddIndexes(index, start, 3, index);

                     /* edge ijk connects to nodes ijk & ijk+offsets. */
                     nalu_hypre_SStructGridFindBoxManEntry(edge_grid, part, index, n,
                                                      &entry);
                     nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, index, &m,
                                                           matrix_type);

                     /* check if the edge lies on the boundary & if not
                        check if the connecting node is on the boundary. */
                     if (eflag[m - start_rank2])
                     {
                        inode[nrows] = m;
                        /* edge not completely on the boundary. One connecting
                           node must be in the interior. */
                        nalu_hypre_SStructGridFindBoxManEntry(node_grid, part, index, 0,
                                                         &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, index, &m,
                                                              matrix_type);

                        /* check if node on my processor. If not, the node must
                           be in the interior (draw a diagram to see this). */
                        if (m >= start_rank1 && m <= jupper)
                        {
                           /* node on proc. Now check if on the boundary. */
                           if (nflag[m - start_rank1]) /* interior node */
                           {
                              jedge[i] = m;
                              vals[i] = 1.0;
                              i++;

                              ncols[nrows]++;
                           }
                        }
                        else  /* node off-proc */
                        {
                           jedge[i] = m;
                           vals[i] = 1.0;
                           i++;

                           ncols[nrows]++;
                        }

                        /* ijk+offsets */
                        nalu_hypre_SubtractIndexes(index, offsets[n], 3, index);
                        nalu_hypre_SStructGridFindBoxManEntry(node_grid, part, index, 0,
                                                         &entry);
                        nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, index, &m,
                                                              matrix_type);
                        /* boundary checks again */
                        if (m >= start_rank1 && m <= jupper)
                        {
                           /* node on proc. Now check if on the boundary. */
                           if (nflag[m - start_rank1]) /* interior node */
                           {
                              jedge[i] = m;
                              i++;
                              ncols[nrows]++;
                           }
                        }
                        else  /* node off-proc */
                        {
                           jedge[i] = m;
                           i++;
                           ncols[nrows]++;
                        }

                        nrows++; /* must have at least one node connection */
                     }  /* if (eflag[m-start_rank2]) */

                  }
                  nalu_hypre_SerialBoxLoop0End();
               }  /* for (ndirection= 0; ndirection< 2; ndirection++) */
            }     /* for (d= 0; d< ndim; d++) */

         }  /* nalu_hypre_ForBoxI(j, boxes) */
      }     /* for (n= 0; n< nvars; n++) */
   }        /* for (part= 0; part< nparts; part++) */

   nalu_hypre_TFree(offsets, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(varoffsets, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(vartype_edges, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_SStructGridDestroy(node_grid);
   NALU_HYPRE_SStructGridDestroy(edge_grid);

   NALU_HYPRE_IJMatrixSetValues(T_grad, nrows, ncols,
                           (const NALU_HYPRE_BigInt*) inode, (const NALU_HYPRE_BigInt*) jedge,
                           (const NALU_HYPRE_Real*) vals);
   NALU_HYPRE_IJMatrixAssemble(T_grad);

   nalu_hypre_TFree(eflag, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nflag, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(ncols, memory_location);
   nalu_hypre_TFree(inode, memory_location);
   nalu_hypre_TFree(jedge, memory_location);
   nalu_hypre_TFree(vals, memory_location);

   parcsr_grad = (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(T_grad);
   NALU_HYPRE_IJMatrixSetObjectType(T_grad, -1);
   NALU_HYPRE_IJMatrixDestroy(T_grad);

   return  parcsr_grad;
}
