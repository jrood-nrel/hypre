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
 *   tot_nsendRowsNcols, send_ColsData_alloc, tot_sendColsData
 *
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellOffProcRowCreate
 *--------------------------------------------------------------------------*/
nalu_hypre_MaxwellOffProcRow *
nalu_hypre_MaxwellOffProcRowCreate(NALU_HYPRE_Int ncols)
{
   nalu_hypre_MaxwellOffProcRow  *OffProcRow;
   NALU_HYPRE_BigInt             *cols;
   NALU_HYPRE_Real               *data;

   OffProcRow = nalu_hypre_CTAlloc(nalu_hypre_MaxwellOffProcRow,  1, NALU_HYPRE_MEMORY_HOST);
   (OffProcRow -> ncols) = ncols;

   cols = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  ncols, NALU_HYPRE_MEMORY_HOST);
   data = nalu_hypre_TAlloc(NALU_HYPRE_Real,  ncols, NALU_HYPRE_MEMORY_HOST);

   (OffProcRow -> cols) = cols;
   (OffProcRow -> data) = data;

   return OffProcRow;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellOffProcRowDestroy
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_MaxwellOffProcRowDestroy(void *OffProcRow_vdata)
{
   nalu_hypre_MaxwellOffProcRow  *OffProcRow = (nalu_hypre_MaxwellOffProcRow  *)OffProcRow_vdata;
   NALU_HYPRE_Int                 ierr = 0;

   if (OffProcRow)
   {
      nalu_hypre_TFree(OffProcRow -> cols, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(OffProcRow -> data, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(OffProcRow, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SStructSharedDOF_ParcsrMatRowsComm
 *   Given a sstruct_grid & parcsr matrix with rows corresponding to the
 *   sstruct_grid, determine and extract the rows that must be communicated.
 *   These rows are for shared dof that geometrically lie on processor
 *   boundaries but internally are stored on one processor.
 *   Algo:
 *       for each cellbox
 *         RECVs:
 *          i)  stretch the cellbox to the variable box
 *          ii) in the appropriate (dof-dependent) direction, take the
 *              boundary and boxman_intersect to extract boxmanentries
 *              that contain these boundary edges.
 *          iii)loop over the boxmanentries and see if they belong
 *              on this proc or another proc
 *                 a) if belong on another proc, these are the recvs:
 *                    count and prepare the communication buffers and
 *                    values.
 *
 *         SENDs:
 *          i)  form layer of cells that is one layer off cellbox
 *              (stretches in the appropriate direction)
 *          ii) boxman_intersect with the cellgrid boxman
 *          iii)loop over the boxmanentries and see if they belong
 *              on this proc or another proc
 *                 a) if belong on another proc, these are the sends:
 *                    count and prepare the communication buffers and
 *                    values.
 *
 * Note: For the recv data, the dof can come from only one processor.
 *       For the send data, the dof can go to more than one processor
 *       (the same dof is on the boundary of several cells).
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_SStructSharedDOF_ParcsrMatRowsComm( nalu_hypre_SStructGrid    *grid,
                                          nalu_hypre_ParCSRMatrix   *A,
                                          NALU_HYPRE_Int            *num_offprocrows_ptr,
                                          nalu_hypre_MaxwellOffProcRow ***OffProcRows_ptr)
{
   MPI_Comm             A_comm = nalu_hypre_ParCSRMatrixComm(A);
   MPI_Comm          grid_comm = nalu_hypre_SStructGridComm(grid);

   NALU_HYPRE_Int       matrix_type = NALU_HYPRE_PARCSR;

   NALU_HYPRE_Int            nparts = nalu_hypre_SStructGridNParts(grid);
   NALU_HYPRE_Int            ndim  = nalu_hypre_SStructGridNDim(grid);

   nalu_hypre_SStructGrid     *cell_ssgrid;

   nalu_hypre_SStructPGrid    *pgrid;
   nalu_hypre_StructGrid      *cellgrid;
   nalu_hypre_BoxArray        *cellboxes;
   nalu_hypre_Box             *box, *cellbox, vbox, boxman_entry_box;

   nalu_hypre_Index            loop_size, start, lindex;
   NALU_HYPRE_BigInt           start_rank, end_rank, rank;

   NALU_HYPRE_Int              i, j, k, m, n, t, part, var, nvars;

   NALU_HYPRE_SStructVariable *vartypes;
   NALU_HYPRE_Int              nbdry_slabs;
   nalu_hypre_BoxArray        *recv_slabs, *send_slabs;
   nalu_hypre_Index            varoffset;

   nalu_hypre_BoxManager     **boxmans, *cell_boxman;
   nalu_hypre_BoxManEntry    **boxman_entries, *entry;
   NALU_HYPRE_Int              nboxman_entries;

   nalu_hypre_Index            ilower, iupper, index;

   NALU_HYPRE_Int              proc, nprocs, myproc;
   NALU_HYPRE_Int             *SendToProcs, *RecvFromProcs;
   NALU_HYPRE_Int            **send_RowsNcols;       /* buffer for rows & ncols */
   NALU_HYPRE_Int             *send_RowsNcols_alloc;
   NALU_HYPRE_Int             *send_ColsData_alloc;
   NALU_HYPRE_Int             *tot_nsendRowsNcols, *tot_sendColsData;
   NALU_HYPRE_Real           **vals;  /* buffer for cols & data */

   NALU_HYPRE_BigInt          *col_inds;
   NALU_HYPRE_Real            *values;

   nalu_hypre_MPI_Request     *requests;
   nalu_hypre_MPI_Status      *status;
   NALU_HYPRE_Int            **rbuffer_RowsNcols;
   NALU_HYPRE_Real           **rbuffer_ColsData;
   NALU_HYPRE_Int              num_sends, num_recvs;

   nalu_hypre_MaxwellOffProcRow **OffProcRows;
   NALU_HYPRE_Int                *starts;

   NALU_HYPRE_Int              ierr = 0;

   nalu_hypre_BoxInit(&vbox, ndim);
   nalu_hypre_BoxInit(&boxman_entry_box, ndim);

   nalu_hypre_MPI_Comm_rank(A_comm, &myproc);
   nalu_hypre_MPI_Comm_size(grid_comm, &nprocs);

   start_rank = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   end_rank  = nalu_hypre_ParCSRMatrixLastRowIndex(A);

   /* need a cellgrid boxman to determine the send boxes -> only the cell dofs
      are unique so a boxman intersect can be used to get the edges that
      must be sent. */
   NALU_HYPRE_SStructGridCreate(grid_comm, ndim, nparts, &cell_ssgrid);
   vartypes = nalu_hypre_CTAlloc(NALU_HYPRE_SStructVariable,  1, NALU_HYPRE_MEMORY_HOST);
   vartypes[0] = NALU_HYPRE_SSTRUCT_VARIABLE_CELL;

   for (i = 0; i < nparts; i++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, i);
      cellgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);

      cellboxes = nalu_hypre_StructGridBoxes(cellgrid);
      nalu_hypre_ForBoxI(j, cellboxes)
      {
         box = nalu_hypre_BoxArrayBox(cellboxes, j);
         NALU_HYPRE_SStructGridSetExtents(cell_ssgrid, i,
                                     nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box));
      }
      NALU_HYPRE_SStructGridSetVariables(cell_ssgrid, i, 1, vartypes);
   }
   NALU_HYPRE_SStructGridAssemble(cell_ssgrid);
   nalu_hypre_TFree(vartypes, NALU_HYPRE_MEMORY_HOST);

   /* box algebra to determine communication */
   SendToProcs    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs, NALU_HYPRE_MEMORY_HOST);
   RecvFromProcs  = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs, NALU_HYPRE_MEMORY_HOST);

   send_RowsNcols      = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nprocs, NALU_HYPRE_MEMORY_HOST);
   send_RowsNcols_alloc = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nprocs, NALU_HYPRE_MEMORY_HOST);
   send_ColsData_alloc = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nprocs, NALU_HYPRE_MEMORY_HOST);
   vals                = nalu_hypre_TAlloc(NALU_HYPRE_Real *,  nprocs, NALU_HYPRE_MEMORY_HOST);
   tot_nsendRowsNcols  = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs, NALU_HYPRE_MEMORY_HOST);
   tot_sendColsData    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < nprocs; i++)
   {
      send_RowsNcols[i] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  1000, NALU_HYPRE_MEMORY_HOST); /* initial allocation */
      send_RowsNcols_alloc[i] = 1000;

      vals[i] = nalu_hypre_TAlloc(NALU_HYPRE_Real,  2000, NALU_HYPRE_MEMORY_HOST); /* initial allocation */
      send_ColsData_alloc[i] = 2000;
   }

   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      nvars = nalu_hypre_SStructPGridNVars(pgrid);
      vartypes = nalu_hypre_SStructPGridVarTypes(pgrid);

      cellgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);
      cellboxes = nalu_hypre_StructGridBoxes(cellgrid);

      boxmans = nalu_hypre_TAlloc(nalu_hypre_BoxManager *,  nvars, NALU_HYPRE_MEMORY_HOST);
      for (t = 0; t < nvars; t++)
      {
         boxmans[t] = nalu_hypre_SStructGridBoxManager(grid, part, t);
      }
      cell_boxman = nalu_hypre_SStructGridBoxManager(cell_ssgrid, part, 0);

      nalu_hypre_ForBoxI(j, cellboxes)
      {
         cellbox = nalu_hypre_BoxArrayBox(cellboxes, j);

         for (t = 0; t < nvars; t++)
         {
            var = vartypes[t];
            nalu_hypre_SStructVariableGetOffset((nalu_hypre_SStructVariable) var,
                                           ndim, varoffset);

            /* form the variable cellbox */
            nalu_hypre_CopyBox(cellbox, &vbox);
            nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&vbox), varoffset, 3,
                                  nalu_hypre_BoxIMin(&vbox));

            /* boundary layer box depends on variable type */
            switch (var)
            {
               case 1:  /* node based */
               {
                  nbdry_slabs = 6;
                  recv_slabs = nalu_hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  /* slab in the +/- i,j,k directions */
                  box = nalu_hypre_BoxArrayBox(recv_slabs, 0);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMin(box)[0] = nalu_hypre_BoxIMax(box)[0];

                  box = nalu_hypre_BoxArrayBox(recv_slabs, 1);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMax(box)[0] = nalu_hypre_BoxIMin(box)[0];

                  /* need to contract the slab in the i direction to avoid repeated
                     counting of some nodes. */
                  box = nalu_hypre_BoxArrayBox(recv_slabs, 2);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMin(box)[1] = nalu_hypre_BoxIMax(box)[1];

                  nalu_hypre_BoxIMin(box)[0]++; /* contract */
                  nalu_hypre_BoxIMax(box)[0]--; /* contract */

                  box = nalu_hypre_BoxArrayBox(recv_slabs, 3);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMax(box)[1] = nalu_hypre_BoxIMin(box)[1];

                  nalu_hypre_BoxIMin(box)[0]++; /* contract */
                  nalu_hypre_BoxIMax(box)[0]--; /* contract */

                  /* need to contract the slab in the i & j directions to avoid repeated
                     counting of some nodes. */
                  box = nalu_hypre_BoxArrayBox(recv_slabs, 4);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMin(box)[2] = nalu_hypre_BoxIMax(box)[2];

                  nalu_hypre_BoxIMin(box)[0]++; /* contract */
                  nalu_hypre_BoxIMax(box)[0]--; /* contract */
                  nalu_hypre_BoxIMin(box)[1]++; /* contract */
                  nalu_hypre_BoxIMax(box)[1]--; /* contract */

                  box = nalu_hypre_BoxArrayBox(recv_slabs, 5);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMax(box)[2] = nalu_hypre_BoxIMin(box)[2];

                  nalu_hypre_BoxIMin(box)[0]++; /* contract */
                  nalu_hypre_BoxIMax(box)[0]--; /* contract */
                  nalu_hypre_BoxIMin(box)[1]++; /* contract */
                  nalu_hypre_BoxIMax(box)[1]--; /* contract */

                  /* send boxes are cell-based stretching out of cellbox - i.e., cells
                     that have these edges as boundary */
                  send_slabs = nalu_hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  box = nalu_hypre_BoxArrayBox(send_slabs, 0);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMax(box)[0]++;
                  nalu_hypre_BoxIMin(box)[0] = nalu_hypre_BoxIMax(box)[0];

                  nalu_hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  nalu_hypre_BoxIMin(box)[2]--;

                  nalu_hypre_BoxIMax(box)[1]++; /* stretch one layer +/- j*/
                  nalu_hypre_BoxIMin(box)[1]--;


                  box = nalu_hypre_BoxArrayBox(send_slabs, 1);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMin(box)[0]--;
                  nalu_hypre_BoxIMax(box)[0] = nalu_hypre_BoxIMin(box)[0];

                  nalu_hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  nalu_hypre_BoxIMin(box)[2]--;

                  nalu_hypre_BoxIMax(box)[1]++; /* stretch one layer +/- j*/
                  nalu_hypre_BoxIMin(box)[1]--;


                  box = nalu_hypre_BoxArrayBox(send_slabs, 2);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMax(box)[1]++;
                  nalu_hypre_BoxIMin(box)[1] = nalu_hypre_BoxIMax(box)[1];

                  nalu_hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  nalu_hypre_BoxIMin(box)[2]--;

                  box = nalu_hypre_BoxArrayBox(send_slabs, 3);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMin(box)[1]--;
                  nalu_hypre_BoxIMax(box)[1] = nalu_hypre_BoxIMin(box)[1];

                  nalu_hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  nalu_hypre_BoxIMin(box)[2]--;


                  box = nalu_hypre_BoxArrayBox(send_slabs, 4);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMax(box)[2]++;
                  nalu_hypre_BoxIMin(box)[2] = nalu_hypre_BoxIMax(box)[2];


                  box = nalu_hypre_BoxArrayBox(send_slabs, 5);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMin(box)[2]--;
                  nalu_hypre_BoxIMax(box)[2] = nalu_hypre_BoxIMin(box)[2];

                  break;
               }

               case 2:  /* x-face based */
               {
                  nbdry_slabs = 2;
                  recv_slabs = nalu_hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  /* slab in the +/- i direction */
                  box = nalu_hypre_BoxArrayBox(recv_slabs, 0);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMin(box)[0] = nalu_hypre_BoxIMax(box)[0];

                  box = nalu_hypre_BoxArrayBox(recv_slabs, 1);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMax(box)[0] = nalu_hypre_BoxIMin(box)[0];

                  /* send boxes are cell-based stretching out of cellbox - i.e., cells
                     that have these edges as boundary */
                  send_slabs = nalu_hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  box = nalu_hypre_BoxArrayBox(send_slabs, 0);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMax(box)[0]++;
                  nalu_hypre_BoxIMin(box)[0] = nalu_hypre_BoxIMax(box)[0];

                  box = nalu_hypre_BoxArrayBox(send_slabs, 1);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMin(box)[0]--;
                  nalu_hypre_BoxIMax(box)[0] = nalu_hypre_BoxIMin(box)[0];

                  break;
               }

               case 3:  /* y-face based */
               {
                  nbdry_slabs = 2;
                  recv_slabs = nalu_hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  /* slab in the +/- j direction */
                  box = nalu_hypre_BoxArrayBox(recv_slabs, 0);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMin(box)[1] = nalu_hypre_BoxIMax(box)[1];

                  box = nalu_hypre_BoxArrayBox(recv_slabs, 1);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMax(box)[1] = nalu_hypre_BoxIMin(box)[1];

                  /* send boxes are cell-based stretching out of cellbox - i.e., cells
                     that have these edges as boundary */
                  send_slabs = nalu_hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  box = nalu_hypre_BoxArrayBox(send_slabs, 0);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMax(box)[1]++;
                  nalu_hypre_BoxIMin(box)[1] = nalu_hypre_BoxIMax(box)[1];

                  box = nalu_hypre_BoxArrayBox(send_slabs, 1);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMin(box)[1]--;
                  nalu_hypre_BoxIMax(box)[1] = nalu_hypre_BoxIMin(box)[1];

                  break;
               }

               case 4:  /* z-face based */
               {
                  nbdry_slabs = 2;
                  recv_slabs = nalu_hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  /* slab in the +/- k direction */
                  box = nalu_hypre_BoxArrayBox(recv_slabs, 0);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMin(box)[2] = nalu_hypre_BoxIMax(box)[2];

                  box = nalu_hypre_BoxArrayBox(recv_slabs, 1);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMax(box)[2] = nalu_hypre_BoxIMin(box)[2];

                  /* send boxes are cell-based stretching out of cellbox - i.e., cells
                     that have these edges as boundary */
                  send_slabs = nalu_hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  box = nalu_hypre_BoxArrayBox(send_slabs, 0);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMax(box)[2]++;
                  nalu_hypre_BoxIMin(box)[2] = nalu_hypre_BoxIMax(box)[2];

                  box = nalu_hypre_BoxArrayBox(send_slabs, 1);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMin(box)[2]--;
                  nalu_hypre_BoxIMax(box)[2] = nalu_hypre_BoxIMin(box)[2];

                  break;
               }

               case 5:  /* x-edge based */
               {
                  nbdry_slabs = 4;
                  recv_slabs = nalu_hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  /* slab in the +/- j & k direction */
                  box = nalu_hypre_BoxArrayBox(recv_slabs, 0);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMin(box)[1] = nalu_hypre_BoxIMax(box)[1];

                  box = nalu_hypre_BoxArrayBox(recv_slabs, 1);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMax(box)[1] = nalu_hypre_BoxIMin(box)[1];

                  /* need to contract the slab in the j direction to avoid repeated
                     counting of some x-edges. */
                  box = nalu_hypre_BoxArrayBox(recv_slabs, 2);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMin(box)[2] = nalu_hypre_BoxIMax(box)[2];

                  nalu_hypre_BoxIMin(box)[1]++; /* contract */
                  nalu_hypre_BoxIMax(box)[1]--; /* contract */

                  box = nalu_hypre_BoxArrayBox(recv_slabs, 3);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMax(box)[2] = nalu_hypre_BoxIMin(box)[2];

                  nalu_hypre_BoxIMin(box)[1]++; /* contract */
                  nalu_hypre_BoxIMax(box)[1]--; /* contract */

                  /* send boxes are cell-based stretching out of cellbox - i.e., cells
                     that have these edges as boundary */
                  send_slabs = nalu_hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  box = nalu_hypre_BoxArrayBox(send_slabs, 0);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMax(box)[1]++;
                  nalu_hypre_BoxIMin(box)[1] = nalu_hypre_BoxIMax(box)[1];

                  nalu_hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  nalu_hypre_BoxIMin(box)[2]--;

                  box = nalu_hypre_BoxArrayBox(send_slabs, 1);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMin(box)[1]--;
                  nalu_hypre_BoxIMax(box)[1] = nalu_hypre_BoxIMin(box)[1];

                  nalu_hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  nalu_hypre_BoxIMin(box)[2]--;

                  box = nalu_hypre_BoxArrayBox(send_slabs, 2);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMax(box)[2]++;
                  nalu_hypre_BoxIMin(box)[2] = nalu_hypre_BoxIMax(box)[2];

                  box = nalu_hypre_BoxArrayBox(send_slabs, 3);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMin(box)[2]--;
                  nalu_hypre_BoxIMax(box)[2] = nalu_hypre_BoxIMin(box)[2];

                  break;
               }

               case 6:  /* y-edge based */
               {
                  nbdry_slabs = 4;
                  recv_slabs = nalu_hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  /* slab in the +/- i & k direction */
                  box = nalu_hypre_BoxArrayBox(recv_slabs, 0);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMin(box)[0] = nalu_hypre_BoxIMax(box)[0];

                  box = nalu_hypre_BoxArrayBox(recv_slabs, 1);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMax(box)[0] = nalu_hypre_BoxIMin(box)[0];

                  /* need to contract the slab in the i direction to avoid repeated
                     counting of some y-edges. */
                  box = nalu_hypre_BoxArrayBox(recv_slabs, 2);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMin(box)[2] = nalu_hypre_BoxIMax(box)[2];

                  nalu_hypre_BoxIMin(box)[0]++; /* contract */
                  nalu_hypre_BoxIMax(box)[0]--; /* contract */

                  box = nalu_hypre_BoxArrayBox(recv_slabs, 3);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMax(box)[2] = nalu_hypre_BoxIMin(box)[2];

                  nalu_hypre_BoxIMin(box)[0]++; /* contract */
                  nalu_hypre_BoxIMax(box)[0]--; /* contract */

                  /* send boxes are cell-based stretching out of cellbox - i.e., cells
                     that have these edges as boundary */
                  send_slabs = nalu_hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  box = nalu_hypre_BoxArrayBox(send_slabs, 0);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMax(box)[0]++;
                  nalu_hypre_BoxIMin(box)[0] = nalu_hypre_BoxIMax(box)[0];

                  nalu_hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  nalu_hypre_BoxIMin(box)[2]--;

                  box = nalu_hypre_BoxArrayBox(send_slabs, 1);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMin(box)[0]--;
                  nalu_hypre_BoxIMax(box)[0] = nalu_hypre_BoxIMin(box)[0];

                  nalu_hypre_BoxIMax(box)[2]++; /* stretch one layer +/- k*/
                  nalu_hypre_BoxIMin(box)[2]--;

                  box = nalu_hypre_BoxArrayBox(send_slabs, 2);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMax(box)[2]++;
                  nalu_hypre_BoxIMin(box)[2] = nalu_hypre_BoxIMax(box)[2];

                  box = nalu_hypre_BoxArrayBox(send_slabs, 3);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMin(box)[2]--;
                  nalu_hypre_BoxIMax(box)[2] = nalu_hypre_BoxIMin(box)[2];

                  break;
               }

               case 7:  /* z-edge based */
               {
                  nbdry_slabs = 4;
                  recv_slabs = nalu_hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  /* slab in the +/- i & j direction */
                  box = nalu_hypre_BoxArrayBox(recv_slabs, 0);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMin(box)[0] = nalu_hypre_BoxIMax(box)[0];

                  box = nalu_hypre_BoxArrayBox(recv_slabs, 1);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMax(box)[0] = nalu_hypre_BoxIMin(box)[0];

                  /* need to contract the slab in the i direction to avoid repeated
                     counting of some z-edges. */
                  box = nalu_hypre_BoxArrayBox(recv_slabs, 2);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMin(box)[1] = nalu_hypre_BoxIMax(box)[1];

                  nalu_hypre_BoxIMin(box)[0]++; /* contract */
                  nalu_hypre_BoxIMax(box)[0]--; /* contract */

                  box = nalu_hypre_BoxArrayBox(recv_slabs, 3);
                  nalu_hypre_CopyBox(&vbox, box);
                  nalu_hypre_BoxIMax(box)[1] = nalu_hypre_BoxIMin(box)[1];

                  nalu_hypre_BoxIMin(box)[0]++; /* contract */
                  nalu_hypre_BoxIMax(box)[0]--; /* contract */

                  /* send boxes are cell-based stretching out of cellbox - i.e., cells
                     that have these edges as boundary */
                  send_slabs = nalu_hypre_BoxArrayCreate(nbdry_slabs, ndim);

                  box = nalu_hypre_BoxArrayBox(send_slabs, 0);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMax(box)[1]++;
                  nalu_hypre_BoxIMin(box)[1] = nalu_hypre_BoxIMax(box)[1];

                  nalu_hypre_BoxIMax(box)[0]++; /* stretch one layer +/- i*/
                  nalu_hypre_BoxIMin(box)[0]--;

                  box = nalu_hypre_BoxArrayBox(send_slabs, 1);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMin(box)[1]--;
                  nalu_hypre_BoxIMax(box)[1] = nalu_hypre_BoxIMin(box)[1];

                  nalu_hypre_BoxIMax(box)[0]++; /* stretch one layer +/- i*/
                  nalu_hypre_BoxIMin(box)[0]--;

                  box = nalu_hypre_BoxArrayBox(send_slabs, 2);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMax(box)[0]++;
                  nalu_hypre_BoxIMin(box)[0] = nalu_hypre_BoxIMax(box)[0];

                  box = nalu_hypre_BoxArrayBox(send_slabs, 3);
                  nalu_hypre_CopyBox(cellbox, box);
                  nalu_hypre_BoxIMin(box)[0]--;
                  nalu_hypre_BoxIMax(box)[0] = nalu_hypre_BoxIMin(box)[0];

                  break;
               }

            }  /* switch(var) */

            /* determine no. of recv rows */
            for (i = 0; i < nbdry_slabs; i++)
            {
               box = nalu_hypre_BoxArrayBox(recv_slabs, i);
               nalu_hypre_BoxManIntersect(boxmans[t], nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box),
                                     &boxman_entries, &nboxman_entries);

               for (m = 0; m < nboxman_entries; m++)
               {
                  nalu_hypre_SStructBoxManEntryGetProcess(boxman_entries[m], &proc);
                  if (proc != myproc)
                  {
                     nalu_hypre_BoxManEntryGetExtents(boxman_entries[m], ilower, iupper);
                     nalu_hypre_BoxSetExtents(&boxman_entry_box, ilower, iupper);
                     nalu_hypre_IntersectBoxes(&boxman_entry_box, box, &boxman_entry_box);

                     RecvFromProcs[proc] += nalu_hypre_BoxVolume(&boxman_entry_box);
                  }
               }
               nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);

               /* determine send rows. Note the cell_boxman */
               box = nalu_hypre_BoxArrayBox(send_slabs, i);
               nalu_hypre_BoxManIntersect(cell_boxman, nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box),
                                     &boxman_entries, &nboxman_entries);

               for (m = 0; m < nboxman_entries; m++)
               {
                  nalu_hypre_SStructBoxManEntryGetProcess(boxman_entries[m], &proc);
                  if (proc != myproc)
                  {
                     nalu_hypre_BoxManEntryGetExtents(boxman_entries[m], ilower, iupper);
                     nalu_hypre_BoxSetExtents(&boxman_entry_box, ilower, iupper);
                     nalu_hypre_IntersectBoxes(&boxman_entry_box, box, &boxman_entry_box);

                     /* not correct box piece right now. Need to determine
                        the correct var box - extend to var_box and then intersect
                        with vbox */
                     nalu_hypre_SubtractIndexes(nalu_hypre_BoxIMin(&boxman_entry_box),
                                           varoffset, 3,
                                           nalu_hypre_BoxIMin(&boxman_entry_box));
                     nalu_hypre_IntersectBoxes(&boxman_entry_box, &vbox, &boxman_entry_box);

                     SendToProcs[proc] += 2 * nalu_hypre_BoxVolume(&boxman_entry_box);
                     /* check to see if sufficient memory allocation for send_rows */
                     if (SendToProcs[proc] > send_RowsNcols_alloc[proc])
                     {
                        send_RowsNcols_alloc[proc] = SendToProcs[proc];
                        send_RowsNcols[proc] =
                           nalu_hypre_TReAlloc(send_RowsNcols[proc],  NALU_HYPRE_Int,
                                          send_RowsNcols_alloc[proc], NALU_HYPRE_MEMORY_HOST);
                     }

                     nalu_hypre_BoxGetSize(&boxman_entry_box, loop_size);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&boxman_entry_box), start);

                     nalu_hypre_SerialBoxLoop0Begin(ndim, loop_size);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        nalu_hypre_SetIndex3(index, lindex[0], lindex[1], lindex[2]);
                        nalu_hypre_AddIndexes(index, start, 3, index);

                        nalu_hypre_SStructGridFindBoxManEntry(grid, part, index, t,
                                                         &entry);
                        if (entry)
                        {
                           nalu_hypre_SStructBoxManEntryGetGlobalRank(entry, index,
                                                                 &rank, matrix_type);

                           /* index may still be off myproc because vbox was formed
                              by expanding the cellbox to the variable box without
                              checking (difficult) the whole expanded box is on myproc */
                           if (rank <= end_rank && rank >= start_rank)
                           {
                              send_RowsNcols[proc][tot_nsendRowsNcols[proc]] = rank;
                              tot_nsendRowsNcols[proc]++;

                              NALU_HYPRE_ParCSRMatrixGetRow((NALU_HYPRE_ParCSRMatrix) A, rank, &n,
                                                       &col_inds, &values);
                              send_RowsNcols[proc][tot_nsendRowsNcols[proc]] = n;
                              tot_nsendRowsNcols[proc]++;

                              /* check if sufficient memory allocation in the data arrays */
                              if ( (tot_sendColsData[proc] + 2 * n) > send_ColsData_alloc[proc] )
                              {
                                 send_ColsData_alloc[proc] += 2000;
                                 vals[proc] = nalu_hypre_TReAlloc(vals[proc],  NALU_HYPRE_Real,
                                                             send_ColsData_alloc[proc], NALU_HYPRE_MEMORY_HOST);
                              }
                              for (k = 0; k < n; k++)
                              {
                                 vals[proc][tot_sendColsData[proc]] = (NALU_HYPRE_Real) col_inds[k];
                                 tot_sendColsData[proc]++;
                                 vals[proc][tot_sendColsData[proc]] = values[k];
                                 tot_sendColsData[proc]++;
                              }
                              NALU_HYPRE_ParCSRMatrixRestoreRow((NALU_HYPRE_ParCSRMatrix) A, rank, &n,
                                                           &col_inds, &values);
                           }  /* if (rank <= end_rank && rank >= start_rank) */
                        }     /* if (entry) */
                     }
                     nalu_hypre_SerialBoxLoop0End();

                  }  /* if (proc != myproc) */
               }     /* for (m= 0; m< nboxman_entries; m++) */
               nalu_hypre_TFree(boxman_entries, NALU_HYPRE_MEMORY_HOST);

            }  /* for (i= 0; i< nbdry_slabs; i++) */
            nalu_hypre_BoxArrayDestroy(send_slabs);
            nalu_hypre_BoxArrayDestroy(recv_slabs);

         }  /* for (t= 0; t< nvars; t++) */
      }     /* nalu_hypre_ForBoxI(j, cellboxes) */
      nalu_hypre_TFree(boxmans, NALU_HYPRE_MEMORY_HOST);
   }  /* for (part= 0; part< nparts; part++) */

   NALU_HYPRE_SStructGridDestroy(cell_ssgrid);

   num_sends = 0;
   num_recvs = 0;
   k = 0;
   starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nprocs; i++)
   {
      starts[i + 1] = starts[i] + RecvFromProcs[i];
      if (RecvFromProcs[i])
      {
         num_recvs++;
         k += RecvFromProcs[i];
      }

      if (tot_sendColsData[i])
      {
         num_sends++;
      }
   }
   OffProcRows = nalu_hypre_TAlloc(nalu_hypre_MaxwellOffProcRow *,  k, NALU_HYPRE_MEMORY_HOST);
   *num_offprocrows_ptr = k;

   requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request,  num_sends + num_recvs, NALU_HYPRE_MEMORY_HOST);
   status  = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,  num_sends + num_recvs, NALU_HYPRE_MEMORY_HOST);

   /* send row size data */
   j = 0;
   rbuffer_RowsNcols = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nprocs, NALU_HYPRE_MEMORY_HOST);
   rbuffer_ColsData = nalu_hypre_TAlloc(NALU_HYPRE_Real *,  nprocs, NALU_HYPRE_MEMORY_HOST);

   for (proc = 0; proc < nprocs; proc++)
   {
      if (RecvFromProcs[proc])
      {
         rbuffer_RowsNcols[proc] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  2 * RecvFromProcs[proc], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_MPI_Irecv(rbuffer_RowsNcols[proc], 2 * RecvFromProcs[proc], NALU_HYPRE_MPI_INT,
                         proc, 0, grid_comm, &requests[j++]);
      }  /* if (RecvFromProcs[proc]) */

   }     /* for (proc= 0; proc< nprocs; proc++) */

   for (proc = 0; proc < nprocs; proc++)
   {
      if (tot_nsendRowsNcols[proc])
      {
         nalu_hypre_MPI_Isend(send_RowsNcols[proc], tot_nsendRowsNcols[proc], NALU_HYPRE_MPI_INT, proc,
                         0, grid_comm, &requests[j++]);
      }
   }

   nalu_hypre_MPI_Waitall(j, requests, status);

   /* unpack data */
   for (proc = 0; proc < nprocs; proc++)
   {
      send_RowsNcols_alloc[proc] = 0;
      if (RecvFromProcs[proc])
      {
         m = 0; ;
         for (i = 0; i < RecvFromProcs[proc]; i++)
         {
            /* rbuffer_RowsNcols[m] has the row & rbuffer_RowsNcols[m+1] the col size */
            OffProcRows[starts[proc] + i] = nalu_hypre_MaxwellOffProcRowCreate(rbuffer_RowsNcols[proc][m + 1]);
            (OffProcRows[starts[proc] + i] -> row)  = rbuffer_RowsNcols[proc][m];
            (OffProcRows[starts[proc] + i] -> ncols) = rbuffer_RowsNcols[proc][m + 1];

            send_RowsNcols_alloc[proc] += rbuffer_RowsNcols[proc][m + 1];
            m += 2;
         }

         rbuffer_ColsData[proc] = nalu_hypre_TAlloc(NALU_HYPRE_Real,  2 * send_RowsNcols_alloc[proc],
                                               NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(rbuffer_RowsNcols[proc], NALU_HYPRE_MEMORY_HOST);
      }
   }

   nalu_hypre_TFree(rbuffer_RowsNcols, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);

   requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request,  num_sends + num_recvs, NALU_HYPRE_MEMORY_HOST);
   status  = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,  num_sends + num_recvs, NALU_HYPRE_MEMORY_HOST);

   /* send row data */
   j = 0;
   for (proc = 0; proc < nprocs; proc++)
   {
      if (RecvFromProcs[proc])
      {
         nalu_hypre_MPI_Irecv(rbuffer_ColsData[proc], 2 * send_RowsNcols_alloc[proc], NALU_HYPRE_MPI_REAL,
                         proc, 1, grid_comm, &requests[j++]);
      }  /* if (RecvFromProcs[proc]) */
   }     /* for (proc= 0; proc< nprocs; proc++) */

   for (proc = 0; proc < nprocs; proc++)
   {
      if (tot_sendColsData[proc])
      {
         nalu_hypre_MPI_Isend(vals[proc], tot_sendColsData[proc], NALU_HYPRE_MPI_REAL, proc,
                         1, grid_comm, &requests[j++]);
      }
   }

   nalu_hypre_MPI_Waitall(j, requests, status);

   /* unpack data */
   for (proc = 0; proc < nprocs; proc++)
   {
      if (RecvFromProcs[proc])
      {
         k = 0;
         for (i = 0; i < RecvFromProcs[proc]; i++)
         {
            col_inds = (OffProcRows[starts[proc] + i] -> cols);
            values  = (OffProcRows[starts[proc] + i] -> data);
            m       = (OffProcRows[starts[proc] + i] -> ncols);

            for (t = 0; t < m; t++)
            {
               col_inds[t] = (NALU_HYPRE_Int) rbuffer_ColsData[proc][k++];
               values[t]  = rbuffer_ColsData[proc][k++];
            }
         }
         nalu_hypre_TFree(rbuffer_ColsData[proc], NALU_HYPRE_MEMORY_HOST);
      }  /* if (RecvFromProcs[proc]) */

   }     /* for (proc= 0; proc< nprocs; proc++) */
   nalu_hypre_TFree(rbuffer_ColsData, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
   for (proc = 0; proc < nprocs; proc++)
   {
      nalu_hypre_TFree(send_RowsNcols[proc], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(vals[proc], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(send_RowsNcols, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(vals, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(tot_sendColsData, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(tot_nsendRowsNcols, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_ColsData_alloc, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_RowsNcols_alloc, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(SendToProcs, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(RecvFromProcs, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(starts, NALU_HYPRE_MEMORY_HOST);

   *OffProcRows_ptr = OffProcRows;

   return ierr;
}
