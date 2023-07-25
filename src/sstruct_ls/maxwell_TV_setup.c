/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * OpenMP Problems
 *
 * Are private static arrays a problem?
 *
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "maxwell_TV.h"
#include "par_amg.h"

#define DEBUG 0
/*--------------------------------------------------------------------------
 * nalu_hypre_MaxwellTV_Setup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MaxwellTV_Setup(void                 *maxwell_vdata,
                      nalu_hypre_SStructMatrix  *Aee_in,
                      nalu_hypre_SStructVector  *b_in,
                      nalu_hypre_SStructVector  *x_in)
{
   nalu_hypre_MaxwellData     *maxwell_TV_data = (nalu_hypre_MaxwellData     *)maxwell_vdata;

   MPI_Comm               comm = nalu_hypre_SStructMatrixComm(Aee_in);

   nalu_hypre_SStructGraph    *graph = nalu_hypre_SStructMatrixGraph(Aee_in);
   nalu_hypre_SStructGrid     *grid = nalu_hypre_SStructGraphGrid(graph);
   nalu_hypre_Index           *rfactor_in = (maxwell_TV_data-> rfactor);
   nalu_hypre_ParCSRMatrix    *T         = (maxwell_TV_data-> Tgrad);

   nalu_hypre_SStructMatrix   *Ann;
   NALU_HYPRE_IJMatrix         Aen;
   nalu_hypre_SStructVector   *bn;
   nalu_hypre_SStructVector   *xn;

   nalu_hypre_ParCSRMatrix    *Aee  = nalu_hypre_SStructMatrixParCSRMatrix(Aee_in);
   nalu_hypre_ParCSRMatrix    *T_transpose;
   nalu_hypre_ParCSRMatrix    *transpose;
   nalu_hypre_ParCSRMatrix    *parcsr_mat;
   NALU_HYPRE_Int              size, *size_ptr;
   NALU_HYPRE_BigInt          *col_inds;
   NALU_HYPRE_Real            *values;

   nalu_hypre_ParVector       *parvector_x;
   nalu_hypre_ParVector       *parvector_b;

   nalu_hypre_ParCSRMatrix   **Aen_l;

   void                  *amg_vdata;
   nalu_hypre_ParAMGData      *amg_data;
   nalu_hypre_ParCSRMatrix   **Ann_l;
   nalu_hypre_ParCSRMatrix   **Pn_l;
   nalu_hypre_ParCSRMatrix   **RnT_l;
   nalu_hypre_ParVector      **bn_l;
   nalu_hypre_ParVector      **xn_l;
   nalu_hypre_ParVector      **resn_l;
   nalu_hypre_ParVector      **en_l;
   nalu_hypre_ParVector      **nVtemp_l;
   nalu_hypre_ParVector      **nVtemp2_l;
   NALU_HYPRE_Int            **nCF_marker_l;
   NALU_HYPRE_Real            *nrelax_weight;
   NALU_HYPRE_Real            *nomega;
   NALU_HYPRE_Int              nrelax_type;
   NALU_HYPRE_Int              node_numlevels;

   nalu_hypre_ParCSRMatrix   **Aee_l;
   nalu_hypre_IJMatrix       **Pe_l;
   nalu_hypre_IJMatrix       **ReT_l;
   nalu_hypre_ParVector      **be_l;
   nalu_hypre_ParVector      **xe_l;
   nalu_hypre_ParVector      **rese_l;
   nalu_hypre_ParVector      **ee_l;
   nalu_hypre_ParVector      **eVtemp_l;
   nalu_hypre_ParVector      **eVtemp2_l;
   NALU_HYPRE_Real            *erelax_weight;
   NALU_HYPRE_Real            *eomega;
   NALU_HYPRE_Int            **eCF_marker_l;
   NALU_HYPRE_Int              erelax_type;

#if 0
   /* objects needed to fine the edge relaxation parameters */
   NALU_HYPRE_Int              relax_type;
   NALU_HYPRE_Int             *relax_types;
   void                  *e_amg_vdata;
   nalu_hypre_ParAMGData      *e_amgData;
   NALU_HYPRE_Int              numCGSweeps = 10;
   NALU_HYPRE_Int            **amg_CF_marker;
   nalu_hypre_ParCSRMatrix   **A_array;
#endif

   nalu_hypre_SStructGrid     *node_grid;
   nalu_hypre_SStructGraph    *node_graph;

   NALU_HYPRE_Int             *coarsen;
   nalu_hypre_SStructGrid    **egrid_l;
   nalu_hypre_SStructGrid     *edge_grid, *face_grid, *cell_grid;
   nalu_hypre_SStructGrid    **topological_edge, **topological_face, **topological_cell;

   NALU_HYPRE_Int            **BdryRanks_l;
   NALU_HYPRE_Int             *BdryRanksCnts_l;

   nalu_hypre_SStructPGrid    *pgrid;
   nalu_hypre_StructGrid      *sgrid;

   nalu_hypre_BoxArray        *boxes, *tmp_box_array;
   nalu_hypre_Box             *box, *box_piece, *contract_box;
   nalu_hypre_BoxArray        *cboxes;

   NALU_HYPRE_SStructVariable *vartypes, *vartype_edges, *vartype_faces, *vartype_cell;
   nalu_hypre_SStructStencil **Ann_stencils;

   nalu_hypre_MaxwellOffProcRow **OffProcRows;
   NALU_HYPRE_Int                 num_OffProcRows;

   nalu_hypre_Index            rfactor;
   nalu_hypre_Index            index, cindex, shape, loop_size, start, lindex;
   NALU_HYPRE_Int              stencil_size;
   NALU_HYPRE_Int              matrix_type = NALU_HYPRE_PARCSR;

   NALU_HYPRE_Int              ndim = nalu_hypre_SStructMatrixNDim(Aee_in);
   NALU_HYPRE_Int              nparts, part, vars, nboxes, lev_nboxes;

   NALU_HYPRE_Int              nrows;
   NALU_HYPRE_BigInt           rank, start_rank, *jnode, *inode;
   NALU_HYPRE_Int             *flag, *ncols;
   NALU_HYPRE_BigInt          *flag2;
   NALU_HYPRE_Real            *vals;

   NALU_HYPRE_Int              i, j, k, l, m;
   NALU_HYPRE_BigInt           big_i, *big_i_ptr;

   nalu_hypre_BoxManager      *node_boxman;
   nalu_hypre_BoxManEntry     *entry;
   NALU_HYPRE_Int              kstart = 0, kend = 0;
   NALU_HYPRE_BigInt           ilower, iupper;
   NALU_HYPRE_BigInt           jlower, jupper;
   NALU_HYPRE_Int              myproc;

   NALU_HYPRE_BigInt           first_local_row, last_local_row;
   NALU_HYPRE_BigInt           first_local_col, last_local_col;

   NALU_HYPRE_Int              edge_maxlevels, edge_numlevels, en_numlevels;

   NALU_HYPRE_Int              constant_coef =  maxwell_TV_data -> constant_coef;
   NALU_HYPRE_Int              trueV = 1;
   NALU_HYPRE_Int              falseV = 0;

   NALU_HYPRE_Int              ierr = 0;
#if DEBUG
   /*char                  filename[255];*/
#endif

   NALU_HYPRE_MemoryLocation   memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(Aee);

   nalu_hypre_MPI_Comm_rank(comm, &myproc);

   (maxwell_TV_data -> ndim) = ndim;

   /* Adjust rfactor so that the correct dimension is used */
   for (i = ndim; i < 3; i++)
   {
      rfactor_in[0][i] = 1;
   }
   nalu_hypre_CopyIndex(rfactor_in[0], rfactor);

   /*---------------------------------------------------------------------
    * Set up matrices Ann, Aen.
    *
    * Forming the finest node matrix: We are assuming the Aee_in is in the
    * parcsr data structure, the stencil structure for the node is the
    * 9 or 27 point fem pattern, etc.
    *
    * Need to form the grid, graph, etc. for these matrices.
    *---------------------------------------------------------------------*/
   nparts = nalu_hypre_SStructMatrixNParts(Aee_in);
   NALU_HYPRE_SStructGridCreate(comm, ndim, nparts, &node_grid);

   /* grids can be constructed from the cell-centre grid of Aee_in */
   vartypes = nalu_hypre_CTAlloc(NALU_HYPRE_SStructVariable,  1, NALU_HYPRE_MEMORY_HOST);
   vartypes[0] = NALU_HYPRE_SSTRUCT_VARIABLE_NODE;

   for (i = 0; i < nparts; i++)
   {
      pgrid = nalu_hypre_SStructPMatrixPGrid(nalu_hypre_SStructMatrixPMatrix(Aee_in, i));
      sgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);

      boxes = nalu_hypre_StructGridBoxes(sgrid);
      nalu_hypre_ForBoxI(j, boxes)
      {
         box = nalu_hypre_BoxArrayBox(boxes, j);
         NALU_HYPRE_SStructGridSetExtents(node_grid, i,
                                     nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box));
      }

      NALU_HYPRE_SStructGridSetVariables(node_grid, i, 1, vartypes);
   }
   NALU_HYPRE_SStructGridAssemble(node_grid);

   /* Ann stencils & graph */
   stencil_size = 1;
   for (i = 0; i < ndim; i++)
   {
      stencil_size *= 3;
   }

   Ann_stencils = nalu_hypre_CTAlloc(nalu_hypre_SStructStencil *,  1, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_SStructStencilCreate(ndim, stencil_size, &Ann_stencils[0]);

   vars = 0; /* scalar equation, node-to-node */
   if (ndim > 2)
   {
      kstart = -1;
      kend  =  2;
   }
   else if (ndim == 2)
   {
      kstart = 0;
      kend  = 1;
   }

   m = 0;
   for (k = kstart; k < kend; k++)
   {
      for (j = -1; j < 2; j++)
      {
         for (i = -1; i < 2; i++)
         {
            nalu_hypre_SetIndex3(shape, i, j, k);
            NALU_HYPRE_SStructStencilSetEntry(Ann_stencils[0], m, shape, vars);
            m++;
         }
      }
   }

   NALU_HYPRE_SStructGraphCreate(comm, node_grid, &node_graph);
   for (part = 0; part < nparts; part++)
   {
      NALU_HYPRE_SStructGraphSetStencil(node_graph, part, 0, Ann_stencils[0]);
   }
   NALU_HYPRE_SStructGraphAssemble(node_graph);

   NALU_HYPRE_SStructMatrixCreate(comm, node_graph, &Ann);
   NALU_HYPRE_SStructMatrixSetObjectType(Ann, NALU_HYPRE_PARCSR);
   NALU_HYPRE_SStructMatrixInitialize(Ann);

   /* Aen is constructed as an IJ matrix. Constructing it as a sstruct_matrix
    * would make it a square matrix. */
   part = 0;
   i   = 0;

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(node_grid, part, 0, i, myproc, &entry);
   pgrid = nalu_hypre_SStructGridPGrid(node_grid, part);
   vartypes[0] = NALU_HYPRE_SSTRUCT_VARIABLE_NODE;
   j = vartypes[0];
   sgrid = nalu_hypre_SStructPGridVTSGrid(pgrid, j);
   boxes = nalu_hypre_StructGridBoxes(sgrid);
   box  = nalu_hypre_BoxArrayBox(boxes, 0);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(box), &jlower);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(grid, part, 0, i, myproc, &entry);
   pgrid = nalu_hypre_SStructGridPGrid(grid, part);
   /* grab the first edge variable type */
   vartypes[0] = nalu_hypre_SStructPGridVarType(pgrid, 0);
   j = vartypes[0];
   sgrid = nalu_hypre_SStructPGridVTSGrid(pgrid, j);
   boxes = nalu_hypre_StructGridBoxes(sgrid);
   box  = nalu_hypre_BoxArrayBox(boxes, 0);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMin(box), &ilower);

   part = nparts - 1;
   pgrid = nalu_hypre_SStructGridPGrid(node_grid, part);
   vartypes[0] = NALU_HYPRE_SSTRUCT_VARIABLE_NODE;
   j = vartypes[0];
   sgrid = nalu_hypre_SStructPGridVTSGrid(pgrid, j);
   boxes = nalu_hypre_StructGridBoxes(sgrid);
   box  = nalu_hypre_BoxArrayBox(boxes, nalu_hypre_BoxArraySize(boxes) - 1);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(node_grid, part, 0,
                                           nalu_hypre_BoxArraySize(boxes) - 1,
                                           myproc, &entry);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(box), &jupper);

   pgrid = nalu_hypre_SStructGridPGrid(grid, part);
   vars = nalu_hypre_SStructPGridNVars(pgrid);
   vartypes[0] = nalu_hypre_SStructPGridVarType(pgrid, vars - 1);
   j = vartypes[0];
   sgrid = nalu_hypre_SStructPGridVTSGrid(pgrid, j);
   boxes = nalu_hypre_StructGridBoxes(sgrid);
   box  = nalu_hypre_BoxArrayBox(boxes, nalu_hypre_BoxArraySize(boxes) - 1);
   nalu_hypre_TFree(vartypes, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_SStructGridBoxProcFindBoxManEntry(grid, part, vars - 1,
                                           nalu_hypre_BoxArraySize(boxes) - 1,
                                           myproc, &entry);
   nalu_hypre_SStructBoxManEntryGetGlobalCSRank(entry, nalu_hypre_BoxIMax(box), &iupper);

   NALU_HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, &Aen);
   NALU_HYPRE_IJMatrixSetObjectType(Aen, NALU_HYPRE_PARCSR);
   NALU_HYPRE_IJMatrixInitialize(Aen);

   /* setup the Aen & Ann using matrix-matrix products
    * Aen's parscr matrix has not been formed yet-> fill up ij_matrix */
   parcsr_mat = nalu_hypre_ParMatmul(Aee, T);
   NALU_HYPRE_ParCSRMatrixGetLocalRange((NALU_HYPRE_ParCSRMatrix) parcsr_mat,
                                   &first_local_row, &last_local_row,
                                   &first_local_col, &last_local_col);

   size_ptr  = nalu_hypre_TAlloc(NALU_HYPRE_Int,    1, memory_location);
   big_i_ptr = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, 1, memory_location);

   for (big_i = first_local_row; big_i <= last_local_row; big_i++)
   {
      NALU_HYPRE_ParCSRMatrixGetRow((NALU_HYPRE_ParCSRMatrix) parcsr_mat,
                               big_i, &size, &col_inds, &values);

      size_ptr[0]  = size;
      big_i_ptr[0] = big_i;

      //RL: this is very slow when using on device
      NALU_HYPRE_IJMatrixSetValues(Aen, 1, size_ptr, big_i_ptr, (const NALU_HYPRE_BigInt *) col_inds,
                              (const NALU_HYPRE_Real *) values);

      NALU_HYPRE_ParCSRMatrixRestoreRow((NALU_HYPRE_ParCSRMatrix) parcsr_mat,
                                   big_i, &size, &col_inds, &values);
   }
   nalu_hypre_ParCSRMatrixDestroy(parcsr_mat);
   NALU_HYPRE_IJMatrixAssemble(Aen);

   /* Ann's parscr matrix has not been formed yet-> fill up ij_matrix */
   nalu_hypre_ParCSRMatrixTranspose(T, &T_transpose, 1);
   parcsr_mat = nalu_hypre_ParMatmul(T_transpose,
                                (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(Aen));
   NALU_HYPRE_ParCSRMatrixGetLocalRange((NALU_HYPRE_ParCSRMatrix) parcsr_mat,
                                   &first_local_row, &last_local_row,
                                   &first_local_col, &last_local_col);

   for (big_i = first_local_row; big_i <= last_local_row; big_i++)
   {
      NALU_HYPRE_ParCSRMatrixGetRow((NALU_HYPRE_ParCSRMatrix) parcsr_mat,
                               big_i, &size, &col_inds, &values);

      size_ptr[0]  = size;
      big_i_ptr[0] = big_i;

      //RL: this is very slow when using on device
      NALU_HYPRE_IJMatrixSetValues(nalu_hypre_SStructMatrixIJMatrix(Ann),
                              1, size_ptr, big_i_ptr, (const NALU_HYPRE_BigInt *) col_inds,
                              (const NALU_HYPRE_Real *) values);

      NALU_HYPRE_ParCSRMatrixRestoreRow((NALU_HYPRE_ParCSRMatrix) parcsr_mat,
                                   big_i, &size, &col_inds, &values);
   }
   nalu_hypre_ParCSRMatrixDestroy(parcsr_mat);

   nalu_hypre_TFree(size_ptr,  memory_location);
   nalu_hypre_TFree(big_i_ptr, memory_location);

   /* set the physical boundary points to identity */
   nrows = 0;
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(node_grid, part);
      sgrid = nalu_hypre_SStructPGridSGrid(pgrid, 0);
      nrows += nalu_hypre_StructGridLocalSize(sgrid);
   }

   flag = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows, NALU_HYPRE_MEMORY_HOST);
   flag2 = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nrows, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nrows; i++)
   {
      flag[i] = 1;
   }

   /* Determine physical boundary points. Get the rank and set flag[rank]= rank.
      This will boundary point, i.e., ncols[rank]> 0 will flag a boundary point. */
   start_rank = nalu_hypre_SStructGridStartRank(node_grid);
   for (part = 0; part < nparts; part++)
   {
      pgrid   = nalu_hypre_SStructGridPGrid(node_grid, part);
      sgrid   = nalu_hypre_SStructPGridSGrid(pgrid, 0);
      boxes   = nalu_hypre_StructGridBoxes(sgrid);
      node_boxman = nalu_hypre_SStructGridBoxManager(node_grid, part, 0);

      nalu_hypre_ForBoxI(j, boxes)
      {
         box = nalu_hypre_BoxArrayBox(boxes, j);
         nalu_hypre_BoxManGetEntry(node_boxman, myproc, j, &entry);
         i = nalu_hypre_BoxVolume(box);

         tmp_box_array = nalu_hypre_BoxArrayCreate(0, ndim);
         ierr        += nalu_hypre_BoxBoundaryG(box, sgrid, tmp_box_array);

         for (m = 0; m < nalu_hypre_BoxArraySize(tmp_box_array); m++)
         {
            box_piece = nalu_hypre_BoxArrayBox(tmp_box_array, m);
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
                  flag[(NALU_HYPRE_Int)(rank - start_rank)] = 0;
                  flag2[(NALU_HYPRE_Int)(rank - start_rank)] = rank;
               }
               nalu_hypre_SerialBoxLoop0End();
            }  /* if (nalu_hypre_BoxVolume(box_piece) < i) */
         }  /* for (m= 0; m< nalu_hypre_BoxArraySize(tmp_box_array); m++) */
         nalu_hypre_BoxArrayDestroy(tmp_box_array);
      }  /* nalu_hypre_ForBoxI(j, boxes) */
   }     /* for (part= 0; part< nparts; part++) */

   /* set up boundary identity */
   j = 0;
   for (i = 0; i < nrows; i++)
   {
      if (!flag[i])
      {
         j++;
      }
   }

   inode = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, j, memory_location);
   ncols = nalu_hypre_CTAlloc(NALU_HYPRE_Int,    j, memory_location);
   jnode = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, j, memory_location);
   vals = nalu_hypre_TAlloc(NALU_HYPRE_Real,    j, memory_location);

   j = 0;
   for (i = 0; i < nrows; i++)
   {
      if (!flag[i])
      {
         ncols[j] = 1;
         inode[j] = flag2[i];
         jnode[j] = flag2[i];
         vals[j] = 1.0;
         j++;
      }
   }
   nalu_hypre_TFree(flag, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(flag2, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_IJMatrixSetValues(nalu_hypre_SStructMatrixIJMatrix(Ann),
                           j, ncols, (const NALU_HYPRE_BigInt*) inode,
                           (const NALU_HYPRE_BigInt*) jnode, (const NALU_HYPRE_Real*) vals);
   nalu_hypre_TFree(ncols, memory_location);
   nalu_hypre_TFree(inode, memory_location);
   nalu_hypre_TFree(jnode, memory_location);
   nalu_hypre_TFree(vals,  memory_location);

   NALU_HYPRE_SStructMatrixAssemble(Ann);
#if DEBUG
   NALU_HYPRE_SStructMatrixPrint("sstruct.out.Ann",  Ann, 0);
   NALU_HYPRE_IJMatrixPrint(Aen, "driver.out.Aen");
#endif

   /* setup bn & xn using matvec. Assemble first and then perform matvec to get
      the nodal rhs and initial guess. */
   NALU_HYPRE_SStructVectorCreate(comm, node_grid, &bn);
   NALU_HYPRE_SStructVectorSetObjectType(bn, NALU_HYPRE_PARCSR);
   NALU_HYPRE_SStructVectorInitialize(bn);
   NALU_HYPRE_SStructVectorAssemble(bn);

   nalu_hypre_SStructVectorConvert(b_in, &parvector_x);
   /*NALU_HYPRE_SStructVectorGetObject((NALU_HYPRE_SStructVector) b_in, (void **) &parvector_x);*/
   NALU_HYPRE_SStructVectorGetObject((NALU_HYPRE_SStructVector) bn, (void **) &parvector_b);
   nalu_hypre_ParCSRMatrixMatvec(1.0, T_transpose, parvector_x, 0.0, parvector_b);

   NALU_HYPRE_SStructVectorCreate(comm, node_grid, &xn);
   NALU_HYPRE_SStructVectorSetObjectType(xn, NALU_HYPRE_PARCSR);
   NALU_HYPRE_SStructVectorInitialize(xn);
   NALU_HYPRE_SStructVectorAssemble(xn);

   nalu_hypre_SStructVectorConvert(x_in, &parvector_x);
   /*NALU_HYPRE_SStructVectorGetObject((NALU_HYPRE_SStructVector) x_in, (void **) &parvector_x);*/
   NALU_HYPRE_SStructVectorGetObject((NALU_HYPRE_SStructVector) xn, (void **) &parvector_b);
   nalu_hypre_ParCSRMatrixMatvec(1.0, T_transpose, parvector_x, 0.0, parvector_b);

   /* Destroy the node grid and graph. This only decrements reference counters. */
   NALU_HYPRE_SStructGridDestroy(node_grid);
   NALU_HYPRE_SStructGraphDestroy(node_graph);

   /* create the multigrid components for the nodal matrix using amg. We need
      to extract the nodal mg components to form the system mg components. */
   amg_vdata = (void *) nalu_hypre_BoomerAMGCreate();
   nalu_hypre_BoomerAMGSetStrongThreshold(amg_vdata, 0.25);
   nalu_hypre_BoomerAMGSetup(amg_vdata,
                        nalu_hypre_SStructMatrixParCSRMatrix(Ann),
                        nalu_hypre_SStructVectorParVector(bn),
                        nalu_hypre_SStructVectorParVector(xn));
   {
      amg_data = (nalu_hypre_ParAMGData*) amg_vdata;

      node_numlevels = nalu_hypre_ParAMGDataNumLevels(amg_data);

      Ann_l   = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix *,  node_numlevels, NALU_HYPRE_MEMORY_HOST);
      Pn_l    = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix *,  node_numlevels, NALU_HYPRE_MEMORY_HOST);
      RnT_l   = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix *,  node_numlevels, NALU_HYPRE_MEMORY_HOST);
      bn_l    = nalu_hypre_CTAlloc(nalu_hypre_ParVector*,  node_numlevels, NALU_HYPRE_MEMORY_HOST);
      xn_l    = nalu_hypre_CTAlloc(nalu_hypre_ParVector*,  node_numlevels, NALU_HYPRE_MEMORY_HOST);
      resn_l  = nalu_hypre_CTAlloc(nalu_hypre_ParVector*,  node_numlevels, NALU_HYPRE_MEMORY_HOST);
      en_l    = nalu_hypre_CTAlloc(nalu_hypre_ParVector*,  node_numlevels, NALU_HYPRE_MEMORY_HOST);
      nVtemp_l = nalu_hypre_CTAlloc(nalu_hypre_ParVector*,  node_numlevels, NALU_HYPRE_MEMORY_HOST);
      nVtemp2_l = nalu_hypre_CTAlloc(nalu_hypre_ParVector*,  node_numlevels, NALU_HYPRE_MEMORY_HOST);

      /* relaxation parameters */
      nCF_marker_l = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  node_numlevels, NALU_HYPRE_MEMORY_HOST);
      nrelax_weight = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  node_numlevels, NALU_HYPRE_MEMORY_HOST);
      nomega       = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  node_numlevels, NALU_HYPRE_MEMORY_HOST);
      nrelax_type  = 6;  /* fast parallel hybrid */

      for (i = 0; i < node_numlevels; i++)
      {
         Ann_l[i] = (nalu_hypre_ParAMGDataAArray(amg_data))[i];
         Pn_l[i] = nalu_hypre_ParAMGDataPArray(amg_data)[i];
         RnT_l[i] = nalu_hypre_ParAMGDataRArray(amg_data)[i];

         bn_l[i] = nalu_hypre_ParAMGDataFArray(amg_data)[i];
         xn_l[i] = nalu_hypre_ParAMGDataUArray(amg_data)[i];

         /* create temporary vectors */
         resn_l[i] = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Ann_l[i]),
                                           nalu_hypre_ParCSRMatrixGlobalNumRows(Ann_l[i]),
                                           nalu_hypre_ParCSRMatrixRowStarts(Ann_l[i]));
         nalu_hypre_ParVectorInitialize(resn_l[i]);

         en_l[i] = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Ann_l[i]),
                                         nalu_hypre_ParCSRMatrixGlobalNumRows(Ann_l[i]),
                                         nalu_hypre_ParCSRMatrixRowStarts(Ann_l[i]));
         nalu_hypre_ParVectorInitialize(en_l[i]);

         nVtemp_l[i] = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Ann_l[i]),
                                             nalu_hypre_ParCSRMatrixGlobalNumRows(Ann_l[i]),
                                             nalu_hypre_ParCSRMatrixRowStarts(Ann_l[i]));
         nalu_hypre_ParVectorInitialize(nVtemp_l[i]);

         nVtemp2_l[i] = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Ann_l[i]),
                                              nalu_hypre_ParCSRMatrixGlobalNumRows(Ann_l[i]),
                                              nalu_hypre_ParCSRMatrixRowStarts(Ann_l[i]));
         nalu_hypre_ParVectorInitialize(nVtemp2_l[i]);

         if (nalu_hypre_ParAMGDataCFMarkerArray(amg_data)[i])
         {
            nCF_marker_l[i] = nalu_hypre_IntArrayData(nalu_hypre_ParAMGDataCFMarkerArray(amg_data)[i]);
         }
         else
         {
            nCF_marker_l[i] = NULL;
         }
         nrelax_weight[i] = nalu_hypre_ParAMGDataRelaxWeight(amg_data)[i];
         nomega[i]       = nalu_hypre_ParAMGDataOmega(amg_data)[i];
      }
   }
   (maxwell_TV_data -> Ann_stencils)    = Ann_stencils;
   (maxwell_TV_data -> T_transpose)     = T_transpose;
   (maxwell_TV_data -> Ann)             = Ann;
   (maxwell_TV_data -> Aen)             = Aen;
   (maxwell_TV_data -> bn)              = bn;
   (maxwell_TV_data -> xn)              = xn;

   (maxwell_TV_data -> amg_vdata)       = amg_vdata;
   (maxwell_TV_data -> Ann_l)           = Ann_l;
   (maxwell_TV_data -> Pn_l)            = Pn_l;
   (maxwell_TV_data -> RnT_l)           = RnT_l;
   (maxwell_TV_data -> bn_l)            = bn_l;
   (maxwell_TV_data -> xn_l)            = xn_l;
   (maxwell_TV_data -> resn_l)          = resn_l;
   (maxwell_TV_data -> en_l)            = en_l;
   (maxwell_TV_data -> nVtemp_l)        = nVtemp_l;
   (maxwell_TV_data -> nVtemp2_l)       = nVtemp2_l;
   (maxwell_TV_data -> nCF_marker_l)    = nCF_marker_l;
   (maxwell_TV_data -> nrelax_weight)   = nrelax_weight;
   (maxwell_TV_data -> nomega)          = nomega;
   (maxwell_TV_data -> nrelax_type)     = nrelax_type;
   (maxwell_TV_data -> node_numlevels)  = node_numlevels;

   /* coarsen the edge matrix. Will coarsen uniformly since we have no
    * scheme to semi-coarsen. That is, coarsen wrt to rfactor, with
    * rfactor[i] > 1 for i < ndim.
    * Determine the number of levels for the edge problem */
   cboxes = nalu_hypre_BoxArrayCreate(0, ndim);
   coarsen = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nparts, NALU_HYPRE_MEMORY_HOST);
   edge_maxlevels = 0;
   for (part = 0; part < nparts; part++)
   {
      pgrid = nalu_hypre_SStructGridPGrid(grid, part);
      sgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);

      box = nalu_hypre_BoxDuplicate(nalu_hypre_StructGridBoundingBox(sgrid));
      nalu_hypre_AppendBox(box, cboxes);
      /* since rfactor[i]>1, the following i will be an upper bound of
         the number of levels. */
      i  = nalu_hypre_Log2(nalu_hypre_BoxSizeD(box, 0)) + 2 +
           nalu_hypre_Log2(nalu_hypre_BoxSizeD(box, 1)) + 2 +
           nalu_hypre_Log2(nalu_hypre_BoxSizeD(box, 2)) + 2;

      nalu_hypre_BoxDestroy(box);
      /* the following allows some of the parts to have volume zero grids */
      edge_maxlevels = nalu_hypre_max(edge_maxlevels, i);
      coarsen[part] = trueV;
   }

   if ((maxwell_TV_data-> edge_maxlevels) > 0)
   {
      edge_maxlevels = nalu_hypre_min(edge_maxlevels,
                                 (maxwell_TV_data -> edge_maxlevels));
   }

   (maxwell_TV_data -> edge_maxlevels) = edge_maxlevels;

   /* form the edge grids: coarsen the cell grid on each part and then
      set the boxes of these grids to be the boxes of the sstruct_grid. */
   egrid_l   = nalu_hypre_TAlloc(nalu_hypre_SStructGrid *,  edge_maxlevels, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SStructGridRef(grid, &egrid_l[0]);

   /* form the topological grids for the topological matrices. */

   /* Assuming same variable ordering on all parts */
   pgrid = nalu_hypre_SStructGridPGrid(grid, 0);

   NALU_HYPRE_SStructGridCreate(comm, ndim, nparts, &edge_grid);
   vartype_edges = nalu_hypre_CTAlloc(NALU_HYPRE_SStructVariable,  ndim, NALU_HYPRE_MEMORY_HOST);
   if (ndim > 2)
   {
      NALU_HYPRE_SStructGridCreate(comm, ndim, nparts, &face_grid);
      vartype_faces = nalu_hypre_CTAlloc(NALU_HYPRE_SStructVariable,  ndim, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < 3; i++)
      {
         vartype_edges[2] = nalu_hypre_SStructPGridVarType(pgrid, i);
         j = vartype_edges[2];

         switch (j)
         {
            case 5:
            {
               vartype_edges[i] = NALU_HYPRE_SSTRUCT_VARIABLE_XEDGE;
               vartype_faces[i] = NALU_HYPRE_SSTRUCT_VARIABLE_XFACE;
               break;
            }
            case 6:
            {
               vartype_edges[i] = NALU_HYPRE_SSTRUCT_VARIABLE_YEDGE;
               vartype_faces[i] = NALU_HYPRE_SSTRUCT_VARIABLE_YFACE;
               break;
            }
            case 7:
            {
               vartype_edges[i] = NALU_HYPRE_SSTRUCT_VARIABLE_ZEDGE;
               vartype_faces[i] = NALU_HYPRE_SSTRUCT_VARIABLE_ZFACE;
               break;
            }

         }  /* switch(j) */
      }     /* for (i= 0; i< 3; i++) */
   }
   else
   {
      for (i = 0; i < 2; i++)
      {
         vartype_edges[1] = nalu_hypre_SStructPGridVarType(pgrid, i);
         j = vartype_edges[1];

         switch (j)
         {
            case 2:
            {
               vartype_edges[i] = NALU_HYPRE_SSTRUCT_VARIABLE_XFACE;
               break;
            }
            case 3:
            {
               vartype_edges[i] = NALU_HYPRE_SSTRUCT_VARIABLE_YFACE;
               break;
            }
         }  /* switch(j) */
      }     /* for (i= 0; i< 3; i++) */
   }

   NALU_HYPRE_SStructGridCreate(comm, ndim, nparts, &cell_grid);
   vartype_cell = nalu_hypre_CTAlloc(NALU_HYPRE_SStructVariable,  1, NALU_HYPRE_MEMORY_HOST);
   vartype_cell[0] = NALU_HYPRE_SSTRUCT_VARIABLE_CELL;

   for (i = 0; i < nparts; i++)
   {
      pgrid = nalu_hypre_SStructPMatrixPGrid(nalu_hypre_SStructMatrixPMatrix(Aee_in, i));
      sgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);

      boxes = nalu_hypre_StructGridBoxes(sgrid);
      nalu_hypre_ForBoxI(j, boxes)
      {
         box = nalu_hypre_BoxArrayBox(boxes, j);
         NALU_HYPRE_SStructGridSetExtents(edge_grid, i,
                                     nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box));
         NALU_HYPRE_SStructGridSetExtents(cell_grid, i,
                                     nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box));
         if (ndim > 2)
         {
            NALU_HYPRE_SStructGridSetExtents(face_grid, i,
                                        nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMax(box));
         }
      }
      NALU_HYPRE_SStructGridSetVariables(edge_grid, i, ndim, vartype_edges);
      NALU_HYPRE_SStructGridSetVariables(cell_grid, i, 1, vartype_cell);

      if (ndim > 2)
      {
         NALU_HYPRE_SStructGridSetVariables(face_grid, i, ndim, vartype_faces);
      }
   }

   NALU_HYPRE_SStructGridAssemble(edge_grid);
   topological_edge   = nalu_hypre_TAlloc(nalu_hypre_SStructGrid *,  edge_maxlevels, NALU_HYPRE_MEMORY_HOST);
   topological_edge[0] = edge_grid;

   NALU_HYPRE_SStructGridAssemble(cell_grid);
   topological_cell   = nalu_hypre_TAlloc(nalu_hypre_SStructGrid *,  edge_maxlevels, NALU_HYPRE_MEMORY_HOST);
   topological_cell[0] = cell_grid;

   if (ndim > 2)
   {
      NALU_HYPRE_SStructGridAssemble(face_grid);
      topological_face = nalu_hypre_TAlloc(nalu_hypre_SStructGrid *,  edge_maxlevels, NALU_HYPRE_MEMORY_HOST);
      topological_face[0] = face_grid;
   }

   /*--------------------------------------------------------------------------
    * to determine when to stop coarsening, we check the cell bounding boxes
    * of the level egrid. After each coarsening, the bounding boxes are
    * replaced by the generated coarse egrid cell bounding boxes.
    *--------------------------------------------------------------------------*/
   nalu_hypre_SetIndex3(cindex, 0, 0, 0);
   j = 0; /* j tracks the number of parts that have been coarsened away */
   edge_numlevels = 1;

   for (l = 0; ; l++)
   {
      NALU_HYPRE_SStructGridCreate(comm, ndim, nparts, &egrid_l[l + 1]);
      NALU_HYPRE_SStructGridCreate(comm, ndim, nparts, &topological_edge[l + 1]);
      NALU_HYPRE_SStructGridCreate(comm, ndim, nparts, &topological_cell[l + 1]);
      if (ndim > 2)
      {
         NALU_HYPRE_SStructGridCreate(comm, ndim, nparts, &topological_face[l + 1]);
      }

      /* coarsen the non-zero bounding boxes only if we have some. */
      nboxes = 0;
      if (j < nparts)
      {
         for (part = 0; part < nparts; part++)
         {
            pgrid = nalu_hypre_SStructGridPGrid(egrid_l[l], part);
            sgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);

            if (coarsen[part])
            {
               box = nalu_hypre_BoxArrayBox(cboxes, part);
               m = trueV;
               for (i = 0; i < ndim; i++)
               {
                  if ( nalu_hypre_BoxIMaxD(box, i) < nalu_hypre_BoxIMinD(box, i) )
                  {
                     m = falseV;
                     break;
                  }
               }

               if (m)
               {
                  /*   MAY NEED TO CHECK THE FOLLOWING MORE CAREFULLY: */
                  /* should we decrease this bounding box so that we get the
                     correct coarse bounding box? Recall that we will decrease
                     each box of the cell_grid so that exact rfactor divisibility
                     is attained. Project does not automatically perform this.
                     E.g., consider a grid with only one box whose width
                     does not divide by rfactor, but it contains beginning and
                     ending indices that are divisible by rfactor. Then an extra
                     coarse grid layer is given by project. */

                  contract_box = nalu_hypre_BoxContraction(box, sgrid, rfactor);
                  nalu_hypre_CopyBox(contract_box, box);
                  nalu_hypre_BoxDestroy(contract_box);

                  nalu_hypre_ProjectBox(box, cindex, rfactor);
                  nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(box), cindex,
                                              rfactor, nalu_hypre_BoxIMin(box));
                  nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(box), cindex,
                                              rfactor, nalu_hypre_BoxIMax(box));

                  /* build the coarse edge grids. Only fill up box extents.
                     The boxes of the grid may be contracted. Note that the
                     box projection may not perform the contraction. */
                  k = 0;
                  nalu_hypre_CoarsenPGrid(egrid_l[l], cindex, rfactor, part,
                                     egrid_l[l + 1], &k);

                  /* build the topological grids */
                  nalu_hypre_CoarsenPGrid(topological_edge[l], cindex, rfactor, part,
                                     topological_edge[l + 1], &i);
                  nalu_hypre_CoarsenPGrid(topological_cell[l], cindex, rfactor, part,
                                     topological_cell[l + 1], &i);
                  if (ndim > 2)
                  {
                     nalu_hypre_CoarsenPGrid(topological_face[l], cindex, rfactor,
                                        part, topological_face[l + 1], &i);
                  }
                  nboxes += k;
               }
               else
               {
                  /* record empty, coarsened-away part */
                  coarsen[part] = falseV;
                  /* set up a dummy box so this grid can be destroyed */
                  NALU_HYPRE_SStructGridSetExtents(egrid_l[l + 1], part,
                                              nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMin(box));

                  NALU_HYPRE_SStructGridSetExtents(topological_edge[l + 1], part,
                                              nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMin(box));

                  NALU_HYPRE_SStructGridSetExtents(topological_cell[l + 1], part,
                                              nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMin(box));

                  if (ndim > 2)
                  {
                     NALU_HYPRE_SStructGridSetExtents(topological_face[l + 1], part,
                                                 nalu_hypre_BoxIMin(box), nalu_hypre_BoxIMin(box));
                  }
                  j++;
               }

            }  /* if (coarsen[part]) */

            vartypes = nalu_hypre_SStructPGridVarTypes(
                          nalu_hypre_SStructGridPGrid(egrid_l[l], part));
            NALU_HYPRE_SStructGridSetVariables(egrid_l[l + 1], part, ndim,
                                          vartypes);

            NALU_HYPRE_SStructGridSetVariables(topological_edge[l + 1], part, ndim,
                                          vartype_edges);
            NALU_HYPRE_SStructGridSetVariables(topological_cell[l + 1], part, 1,
                                          vartype_cell);
            if (ndim > 2)
            {
               NALU_HYPRE_SStructGridSetVariables(topological_face[l + 1], part, ndim,
                                             vartype_faces);
            }
         }  /* for (part= 0; part< nparts; part++) */
      }     /* if (j < nparts) */

      NALU_HYPRE_SStructGridAssemble(egrid_l[l + 1]);
      NALU_HYPRE_SStructGridAssemble(topological_edge[l + 1]);
      NALU_HYPRE_SStructGridAssemble(topological_cell[l + 1]);
      if (ndim > 2)
      {
         NALU_HYPRE_SStructGridAssemble(topological_face[l + 1]);
      }

      lev_nboxes = 0;
      nalu_hypre_MPI_Allreduce(&nboxes, &lev_nboxes, 1, NALU_HYPRE_MPI_INT, nalu_hypre_MPI_SUM,
                          nalu_hypre_SStructGridComm(egrid_l[l + 1]));

      if (lev_nboxes)  /* there were coarsen boxes */
      {
         edge_numlevels++;
      }

      else
      {
         /* no coarse boxes. Trigger coarsening completed and destroy the
            cgrids corresponding to this level. */
         j = nparts;
      }

      /* extract the cell bounding boxes */
      if (j < nparts)
      {
         for (part = 0; part < nparts; part++)
         {
            if (coarsen[part])
            {
               pgrid = nalu_hypre_SStructGridPGrid(egrid_l[l + 1], part);
               sgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);

               box = nalu_hypre_BoxDuplicate(nalu_hypre_StructGridBoundingBox(sgrid));
               nalu_hypre_CopyBox(box, nalu_hypre_BoxArrayBox(cboxes, part));
               nalu_hypre_BoxDestroy(box);
            }
         }
      }

      else
      {
         NALU_HYPRE_SStructGridDestroy(egrid_l[l + 1]);
         NALU_HYPRE_SStructGridDestroy(topological_edge[l + 1]);
         NALU_HYPRE_SStructGridDestroy(topological_cell[l + 1]);
         if (ndim > 2)
         {
            NALU_HYPRE_SStructGridDestroy(topological_face[l + 1]);
         }
         break;
      }
   }
   (maxwell_TV_data -> egrid_l) = egrid_l;

   nalu_hypre_Maxwell_PhysBdy(egrid_l, edge_numlevels, rfactor,
                         &BdryRanks_l, &BdryRanksCnts_l);

   (maxwell_TV_data -> BdryRanks_l)    = BdryRanks_l;
   (maxwell_TV_data -> BdryRanksCnts_l) = BdryRanksCnts_l;

   nalu_hypre_BoxArrayDestroy(cboxes);
   nalu_hypre_TFree(coarsen, NALU_HYPRE_MEMORY_HOST);
   /* okay to de-allocate vartypes now */
   nalu_hypre_TFree(vartype_edges, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(vartype_cell, NALU_HYPRE_MEMORY_HOST);
   if (ndim > 2)
   {
      nalu_hypre_TFree(vartype_faces, NALU_HYPRE_MEMORY_HOST);
   }


   /* Aen matrices are defined for min(edge_numlevels, node_numlevels). */
   en_numlevels = nalu_hypre_min(edge_numlevels, node_numlevels);
   (maxwell_TV_data -> en_numlevels)  = en_numlevels;
   (maxwell_TV_data -> edge_numlevels) = edge_numlevels;

   Aee_l = nalu_hypre_TAlloc(nalu_hypre_ParCSRMatrix *,  edge_numlevels, NALU_HYPRE_MEMORY_HOST);
   Aen_l = nalu_hypre_TAlloc(nalu_hypre_ParCSRMatrix *,  en_numlevels, NALU_HYPRE_MEMORY_HOST);

   /* Pe_l are defined to be IJ matrices rather than directly parcsr. This
      was done so that in the topological formation, some of the ij matrix
      routines can be used. */
   Pe_l    = nalu_hypre_TAlloc(nalu_hypre_IJMatrix  *,  edge_numlevels - 1, NALU_HYPRE_MEMORY_HOST);
   ReT_l   = nalu_hypre_TAlloc(nalu_hypre_IJMatrix  *,  edge_numlevels - 1, NALU_HYPRE_MEMORY_HOST);

   be_l    = nalu_hypre_TAlloc(nalu_hypre_ParVector *,  edge_numlevels, NALU_HYPRE_MEMORY_HOST);
   xe_l    = nalu_hypre_TAlloc(nalu_hypre_ParVector *,  edge_numlevels, NALU_HYPRE_MEMORY_HOST);
   rese_l  = nalu_hypre_TAlloc(nalu_hypre_ParVector *,  edge_numlevels, NALU_HYPRE_MEMORY_HOST);
   ee_l    = nalu_hypre_TAlloc(nalu_hypre_ParVector *,  edge_numlevels, NALU_HYPRE_MEMORY_HOST);
   eVtemp_l = nalu_hypre_TAlloc(nalu_hypre_ParVector *,  edge_numlevels, NALU_HYPRE_MEMORY_HOST);
   eVtemp2_l = nalu_hypre_TAlloc(nalu_hypre_ParVector *,  edge_numlevels, NALU_HYPRE_MEMORY_HOST);

   Aee_l[0] = nalu_hypre_SStructMatrixParCSRMatrix(Aee_in);
   Aen_l[0] = (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(Aen),
              be_l[0] = nalu_hypre_SStructVectorParVector(b_in);
   xe_l[0] = nalu_hypre_SStructVectorParVector(x_in);

   rese_l[0] =
      nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[0]),
                            nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[0]),
                            nalu_hypre_ParCSRMatrixRowStarts(Aee_l[0]));
   nalu_hypre_ParVectorInitialize(rese_l[0]);

   ee_l[0] =
      nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[0]),
                            nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[0]),
                            nalu_hypre_ParCSRMatrixRowStarts(Aee_l[0]));
   nalu_hypre_ParVectorInitialize(ee_l[0]);

   eVtemp_l[0] =
      nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[0]),
                            nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[0]),
                            nalu_hypre_ParCSRMatrixRowStarts(Aee_l[0]));
   nalu_hypre_ParVectorInitialize(eVtemp_l[0]);

   eVtemp2_l[0] =
      nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[0]),
                            nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[0]),
                            nalu_hypre_ParCSRMatrixRowStarts(Aee_l[0]));
   nalu_hypre_ParVectorInitialize(eVtemp2_l[0]);

   for (l = 0; l < (en_numlevels - 1); l++)
   {
      if (l < edge_numlevels) /* create edge operators */
      {
         if (!constant_coef)
         {
            void             *PTopology_vdata;
            nalu_hypre_PTopology  *PTopology;

            nalu_hypre_CreatePTopology(&PTopology_vdata);
            if (ndim > 2)
            {
               Pe_l[l] = nalu_hypre_Maxwell_PTopology(topological_edge[l],
                                                 topological_edge[l + 1],
                                                 topological_face[l],
                                                 topological_face[l + 1],
                                                 topological_cell[l],
                                                 topological_cell[l + 1],
                                                 Aee_l[l],
                                                 rfactor,
                                                 PTopology_vdata);
            }
            else
            {
               /* two-dim case: edges= faces but stored in edge grid */
               Pe_l[l] = nalu_hypre_Maxwell_PTopology(topological_edge[l],
                                                 topological_edge[l + 1],
                                                 topological_edge[l],
                                                 topological_edge[l + 1],
                                                 topological_cell[l],
                                                 topological_cell[l + 1],
                                                 Aee_l[l],
                                                 rfactor,
                                                 PTopology_vdata);
            }

            PTopology = (nalu_hypre_PTopology*)PTopology_vdata;

            /* extract off-processors rows of Pe_l[l]. Needed for amge.*/
            nalu_hypre_SStructSharedDOF_ParcsrMatRowsComm(egrid_l[l],
                                                     (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(Pe_l[l]),
                                                     &num_OffProcRows,
                                                     &OffProcRows);

            if (ndim == 3)
            {
               nalu_hypre_ND1AMGeInterpolation(Aee_l[l],
                                          (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Element_iedge),
                                          (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Face_iedge),
                                          (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Edge_iedge),
                                          (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Element_Face),
                                          (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Element_Edge),
                                          num_OffProcRows,
                                          OffProcRows,
                                          Pe_l[l]);
            }
            else
            {
               nalu_hypre_ND1AMGeInterpolation(Aee_l[l],
                                          (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Element_iedge),
                                          (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Edge_iedge),
                                          (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Edge_iedge),
                                          (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Element_Edge),
                                          (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Element_Edge),
                                          num_OffProcRows,
                                          OffProcRows,
                                          Pe_l[l]);
            }

            nalu_hypre_DestroyPTopology(PTopology_vdata);

            for (i = 0; i < num_OffProcRows; i++)
            {
               nalu_hypre_MaxwellOffProcRowDestroy((void *) OffProcRows[i]);
            }
            nalu_hypre_TFree(OffProcRows, NALU_HYPRE_MEMORY_HOST);
         }

         else
         {
            Pe_l[l] = nalu_hypre_Maxwell_PNedelec(topological_edge[l],
                                             topological_edge[l + 1],
                                             rfactor);
         }
#if DEBUG
#endif


         ReT_l[l] = Pe_l[l];
         nalu_hypre_BoomerAMGBuildCoarseOperator(
            (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(Pe_l[l]),
            Aee_l[l],
            (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(Pe_l[l]),
            &Aee_l[l + 1]);

         /* zero off boundary points */
         nalu_hypre_ParCSRMatrixEliminateRowsCols(Aee_l[l + 1],
                                             BdryRanksCnts_l[l + 1],
                                             BdryRanks_l[l + 1]);

         nalu_hypre_ParCSRMatrixTranspose(
            (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(Pe_l[l]),
            &transpose, 1);
         parcsr_mat = nalu_hypre_ParMatmul(transpose, Aen_l[l]);
         Aen_l[l + 1] = nalu_hypre_ParMatmul(parcsr_mat, Pn_l[l]);
         nalu_hypre_ParCSRMatrixDestroy(parcsr_mat);
         nalu_hypre_ParCSRMatrixDestroy(transpose);

         xe_l[l + 1] =
            nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[l + 1]),
                                  nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[l + 1]),
                                  nalu_hypre_ParCSRMatrixRowStarts(Aee_l[l + 1]));
         nalu_hypre_ParVectorInitialize(xe_l[l + 1]);

         be_l[l + 1] =
            nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[l + 1]),
                                  nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[l + 1]),
                                  nalu_hypre_ParCSRMatrixRowStarts(Aee_l[l + 1]));
         nalu_hypre_ParVectorInitialize(be_l[l + 1]);

         rese_l[l + 1] =
            nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[l + 1]),
                                  nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[l + 1]),
                                  nalu_hypre_ParCSRMatrixRowStarts(Aee_l[l + 1]));
         nalu_hypre_ParVectorInitialize(rese_l[l + 1]);

         ee_l[l + 1] =
            nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[l + 1]),
                                  nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[l + 1]),
                                  nalu_hypre_ParCSRMatrixRowStarts(Aee_l[l + 1]));
         nalu_hypre_ParVectorInitialize(ee_l[l + 1]);

         eVtemp_l[l + 1] =
            nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[l + 1]),
                                  nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[l + 1]),
                                  nalu_hypre_ParCSRMatrixRowStarts(Aee_l[l + 1]));
         nalu_hypre_ParVectorInitialize(eVtemp_l[l + 1]);

         eVtemp2_l[l + 1] =
            nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[l + 1]),
                                  nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[l + 1]),
                                  nalu_hypre_ParCSRMatrixRowStarts(Aee_l[l + 1]));
         nalu_hypre_ParVectorInitialize(eVtemp2_l[l + 1]);

      }  /* if (l < edge_numlevels) */
   }     /* for (l = 0; l < (en_numlevels - 1); l++) */

   /* possible to have more edge levels */
   for (l = (en_numlevels - 1); l < (edge_numlevels - 1); l++)
   {
      if (!constant_coef)
      {
         void             *PTopology_vdata;
         nalu_hypre_PTopology  *PTopology;

         nalu_hypre_CreatePTopology(&PTopology_vdata);
         if (ndim > 2)
         {
            Pe_l[l] = nalu_hypre_Maxwell_PTopology(topological_edge[l],
                                              topological_edge[l + 1],
                                              topological_face[l],
                                              topological_face[l + 1],
                                              topological_cell[l],
                                              topological_cell[l + 1],
                                              Aee_l[l],
                                              rfactor,
                                              PTopology_vdata);
         }
         else
         {
            Pe_l[l] = nalu_hypre_Maxwell_PTopology(topological_edge[l],
                                              topological_edge[l + 1],
                                              topological_edge[l],
                                              topological_edge[l + 1],
                                              topological_cell[l],
                                              topological_cell[l + 1],
                                              Aee_l[l],
                                              rfactor,
                                              PTopology_vdata);
         }

         PTopology = (nalu_hypre_PTopology*)PTopology_vdata;

         /* extract off-processors rows of Pe_l[l]. Needed for amge.*/
         nalu_hypre_SStructSharedDOF_ParcsrMatRowsComm(egrid_l[l],
                                                  (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(Pe_l[l]),
                                                  &num_OffProcRows,
                                                  &OffProcRows);
         if (ndim == 3)
         {
            nalu_hypre_ND1AMGeInterpolation(Aee_l[l],
                                       (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Element_iedge),
                                       (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Face_iedge),
                                       (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Edge_iedge),
                                       (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Element_Face),
                                       (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Element_Edge),
                                       num_OffProcRows,
                                       OffProcRows,
                                       Pe_l[l]);
         }
         else
         {
            nalu_hypre_ND1AMGeInterpolation(Aee_l[l],
                                       (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Element_iedge),
                                       (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Edge_iedge),
                                       (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Edge_iedge),
                                       (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Element_Edge),
                                       (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(PTopology -> Element_Edge),
                                       num_OffProcRows,
                                       OffProcRows,
                                       Pe_l[l]);
         }

         nalu_hypre_DestroyPTopology(PTopology_vdata);
         for (i = 0; i < num_OffProcRows; i++)
         {
            nalu_hypre_MaxwellOffProcRowDestroy((void *) OffProcRows[i]);
         }
         nalu_hypre_TFree(OffProcRows, NALU_HYPRE_MEMORY_HOST);
      }

      else
      {
         Pe_l[l] = nalu_hypre_Maxwell_PNedelec(topological_edge[l],
                                          topological_edge[l + 1],
                                          rfactor);
      }

      ReT_l[l] = Pe_l[l];
      nalu_hypre_BoomerAMGBuildCoarseOperator(
         (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(Pe_l[l]),
         Aee_l[l],
         (nalu_hypre_ParCSRMatrix *) nalu_hypre_IJMatrixObject(Pe_l[l]),
         &Aee_l[l + 1]);

      /* zero off boundary points */
      nalu_hypre_ParCSRMatrixEliminateRowsCols(Aee_l[l + 1],
                                          BdryRanksCnts_l[l + 1],
                                          BdryRanks_l[l + 1]);

      xe_l[l + 1] =
         nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[l + 1]),
                               nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[l + 1]),
                               nalu_hypre_ParCSRMatrixRowStarts(Aee_l[l + 1]));
      nalu_hypre_ParVectorInitialize(xe_l[l + 1]);

      be_l[l + 1] =
         nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[l + 1]),
                               nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[l + 1]),
                               nalu_hypre_ParCSRMatrixRowStarts(Aee_l[l + 1]));
      nalu_hypre_ParVectorInitialize(be_l[l + 1]);

      ee_l[l + 1] =
         nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[l + 1]),
                               nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[l + 1]),
                               nalu_hypre_ParCSRMatrixRowStarts(Aee_l[l + 1]));
      nalu_hypre_ParVectorInitialize(ee_l[l + 1]);

      rese_l[l + 1] =
         nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[l + 1]),
                               nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[l + 1]),
                               nalu_hypre_ParCSRMatrixRowStarts(Aee_l[l + 1]));
      nalu_hypre_ParVectorInitialize(rese_l[l + 1]);

      eVtemp_l[l + 1] =
         nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[l + 1]),
                               nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[l + 1]),
                               nalu_hypre_ParCSRMatrixRowStarts(Aee_l[l + 1]));
      nalu_hypre_ParVectorInitialize(eVtemp_l[l + 1]);

      eVtemp2_l[l + 1] =
         nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(Aee_l[l + 1]),
                               nalu_hypre_ParCSRMatrixGlobalNumRows(Aee_l[l + 1]),
                               nalu_hypre_ParCSRMatrixRowStarts(Aee_l[l + 1]));
      nalu_hypre_ParVectorInitialize(eVtemp2_l[l + 1]);
   }

   /* Can delete all topological grids. Not even referenced in IJMatrices. */
   for (l = 0; l < edge_numlevels; l++)
   {
      NALU_HYPRE_SStructGridDestroy(topological_edge[l]);
      NALU_HYPRE_SStructGridDestroy(topological_cell[l]);
      if (ndim > 2)
      {
         NALU_HYPRE_SStructGridDestroy(topological_face[l]);
      }
   }
   nalu_hypre_TFree(topological_edge, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(topological_cell, NALU_HYPRE_MEMORY_HOST);
   if (ndim > 2)
   {
      nalu_hypre_TFree(topological_face, NALU_HYPRE_MEMORY_HOST);
   }

#if DEBUG
#endif

   (maxwell_TV_data -> Aee_l)    = Aee_l;
   (maxwell_TV_data -> Aen_l)    = Aen_l;
   (maxwell_TV_data -> Pe_l)     = Pe_l;
   (maxwell_TV_data -> ReT_l)    = ReT_l;
   (maxwell_TV_data -> xe_l)     = xe_l;
   (maxwell_TV_data -> be_l)     = be_l;
   (maxwell_TV_data -> ee_l)     = ee_l;
   (maxwell_TV_data -> rese_l)   = rese_l;
   (maxwell_TV_data -> eVtemp_l) = eVtemp_l;
   (maxwell_TV_data -> eVtemp2_l) = eVtemp2_l;

   /*-----------------------------------------------------
    * Determine relaxation parameters for edge problems.
    * Needed for quick parallel over/under-relaxation.
    *-----------------------------------------------------*/
   erelax_type  = 2;
   erelax_weight = nalu_hypre_TAlloc(NALU_HYPRE_Real,  edge_numlevels, NALU_HYPRE_MEMORY_HOST);
   eomega       = nalu_hypre_TAlloc(NALU_HYPRE_Real,  edge_numlevels, NALU_HYPRE_MEMORY_HOST);
   eCF_marker_l = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  edge_numlevels, NALU_HYPRE_MEMORY_HOST);

#if 0
   relax_type = 6; /* SSOR */
   for (l = 0; l < 1; l++)
   {
      erelax_weight[l] = 1.0;
      eCF_marker_l[l] = NULL;

      e_amg_vdata = (void *) nalu_hypre_BoomerAMGCreate();
      e_amgData = e_amg_vdata;

      relax_types = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  2, NALU_HYPRE_MEMORY_HOST);
      relax_types[1] = relax_type;

      amg_CF_marker = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  1, NALU_HYPRE_MEMORY_HOST);
      A_array      = nalu_hypre_TAlloc(nalu_hypre_ParCSRMatrix *,  1, NALU_HYPRE_MEMORY_HOST);

      amg_CF_marker[0] = NULL;
      A_array[0]      = Aee_l[l];

      (e_amgData -> CF_marker_array)   = amg_CF_marker;
      (e_amgData -> A_array)           = A_array;
      (e_amgData -> Vtemp )            = eVtemp_l[l];
      (e_amgData -> grid_relax_type)   = relax_types;
      (e_amgData -> smooth_num_levels) = 0;
      (e_amgData -> smooth_type)       = 0;
      nalu_hypre_BoomerAMGCGRelaxWt((void *) e_amgData, 0, numCGSweeps, &eomega[l]);

      nalu_hypre_TFree((e_amgData -> A_array), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree((e_amgData -> CF_marker_array), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree((e_amgData -> grid_relax_type), NALU_HYPRE_MEMORY_HOST);
      (e_amgData -> A_array) = NULL;
      (e_amgData -> Vtemp ) = NULL;
      (e_amgData -> CF_marker_array) = NULL;
      (e_amgData -> grid_relax_type) = NULL;
      nalu_hypre_TFree(e_amg_vdata, NALU_HYPRE_MEMORY_HOST);
      eomega[l] = 1.0;
   }
#endif

   for (l = 0; l < edge_numlevels; l++)
   {
      erelax_weight[l] = 1.0;
      eomega[l] = 1.0;
      eCF_marker_l[l] = NULL;
   }
   (maxwell_TV_data ->  erelax_type)  = erelax_type;
   (maxwell_TV_data ->  erelax_weight) = erelax_weight;
   (maxwell_TV_data ->  eomega)       = eomega;
   (maxwell_TV_data ->  eCF_marker_l) = eCF_marker_l;


   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ((maxwell_TV_data -> logging) > 0)
   {
      i = (maxwell_TV_data -> max_iter);
      (maxwell_TV_data -> norms)     = nalu_hypre_TAlloc(NALU_HYPRE_Real,  i, NALU_HYPRE_MEMORY_HOST);
      (maxwell_TV_data -> rel_norms) = nalu_hypre_TAlloc(NALU_HYPRE_Real,  i, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_CoarsenPGrid( nalu_hypre_SStructGrid  *fgrid,
                    nalu_hypre_Index         index,
                    nalu_hypre_Index         stride,
                    NALU_HYPRE_Int           part,
                    nalu_hypre_SStructGrid  *cgrid,
                    NALU_HYPRE_Int          *nboxes)
{
   NALU_HYPRE_Int ierr = 0;

   nalu_hypre_SStructPGrid *pgrid = nalu_hypre_SStructGridPGrid(fgrid, part);
   nalu_hypre_StructGrid   *sgrid = nalu_hypre_SStructPGridCellSGrid(pgrid);

   nalu_hypre_BoxArray     *boxes;
   nalu_hypre_Box          *box, *contract_box;
   NALU_HYPRE_Int           i;

   /*-----------------------------------------
    * Set the coarse sgrid
    *-----------------------------------------*/
   boxes = nalu_hypre_BoxArrayDuplicate(nalu_hypre_StructGridBoxes(sgrid));
   for (i = 0; i < nalu_hypre_BoxArraySize(boxes); i++)
   {
      box = nalu_hypre_BoxArrayBox(boxes, i);

      /* contract box so that divisible by stride */
      contract_box = nalu_hypre_BoxContraction(box, sgrid, stride);
      nalu_hypre_ProjectBox(contract_box, index, stride);

      nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(contract_box), index, stride,
                                  nalu_hypre_BoxIMin(contract_box));
      nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(contract_box), index, stride,
                                  nalu_hypre_BoxIMax(contract_box));

      /* set box even if zero volume but don't count it */
      NALU_HYPRE_SStructGridSetExtents(cgrid, part,
                                  nalu_hypre_BoxIMin(contract_box),
                                  nalu_hypre_BoxIMax(contract_box));

      if ( nalu_hypre_BoxVolume(contract_box) )
      {
         *nboxes = *nboxes + 1;
      }
      nalu_hypre_BoxDestroy(contract_box);
   }
   nalu_hypre_BoxArrayDestroy(boxes);

   return ierr;
}



/*--------------------------------------------------------------------------
 *  Contracts a box so that the resulting box divides evenly into rfactor.
 *  Contraction is done in the (+) or (-) direction that does not have
 *  neighbor boxes, or if both directions have neighbor boxes, the (-) side
 *  is contracted.
 *  Modified to use box manager AHB 11/06
 *--------------------------------------------------------------------------*/

nalu_hypre_Box *
nalu_hypre_BoxContraction( nalu_hypre_Box           *box,
                      nalu_hypre_StructGrid    *sgrid,
                      nalu_hypre_Index          rfactor )
{

   nalu_hypre_BoxManager    *boxman = nalu_hypre_StructGridBoxMan(sgrid);

   nalu_hypre_BoxArray      *neighbor_boxes = NULL;
   nalu_hypre_Box           *nbox;
   nalu_hypre_Box           *contracted_box;
   nalu_hypre_Box           *shifted_box;
   nalu_hypre_Box            intersect_box;

   NALU_HYPRE_Int            ndim = nalu_hypre_StructGridNDim(sgrid);

   nalu_hypre_Index          remainder, box_width;
   NALU_HYPRE_Int            i, j, k, p;
   NALU_HYPRE_Int            npos, nneg;


   /* get the boxes out of the box manager - use these as the neighbor boxes */
   neighbor_boxes = nalu_hypre_BoxArrayCreate(0, ndim);
   nalu_hypre_BoxManGetAllEntriesBoxes( boxman, neighbor_boxes);

   nalu_hypre_BoxInit(&intersect_box, ndim);

   contracted_box = nalu_hypre_BoxCreate(ndim);

   nalu_hypre_ClearIndex(remainder);
   p = 0;
   for (i = 0; i < ndim; i++)
   {
      j = nalu_hypre_BoxIMax(box)[i] - nalu_hypre_BoxIMin(box)[i] + 1;
      box_width[i] = j;
      k = j % rfactor[i];

      if (k)
      {
         remainder[i] = k;
         p++;
      }
   }

   nalu_hypre_CopyBox(box, contracted_box);
   if (p)
   {
      shifted_box = nalu_hypre_BoxCreate(ndim);
      for (i = 0; i < ndim; i++)
      {
         if (remainder[i])   /* non-divisible in the i'th direction */
         {
            /* shift box in + & - directions to determine which side to
               contract. */
            nalu_hypre_CopyBox(box, shifted_box);
            nalu_hypre_BoxIMax(shifted_box)[i] += box_width[i];
            nalu_hypre_BoxIMin(shifted_box)[i] += box_width[i];

            npos = 0;
            nalu_hypre_ForBoxI(k, neighbor_boxes)
            {
               nbox = nalu_hypre_BoxArrayBox(neighbor_boxes, k);
               nalu_hypre_IntersectBoxes(shifted_box, nbox, &intersect_box);
               if (nalu_hypre_BoxVolume(&intersect_box))
               {
                  npos++;
               }
            }

            nalu_hypre_CopyBox(box, shifted_box);
            nalu_hypre_BoxIMax(shifted_box)[i] -= box_width[i];
            nalu_hypre_BoxIMin(shifted_box)[i] -= box_width[i];

            nneg = 0;
            nalu_hypre_ForBoxI(k, neighbor_boxes)
            {
               nbox = nalu_hypre_BoxArrayBox(neighbor_boxes, k);
               nalu_hypre_IntersectBoxes(shifted_box, nbox, &intersect_box);
               if (nalu_hypre_BoxVolume(&intersect_box))
               {
                  nneg++;
               }
            }

            if ( (npos) || ( (!npos) && (!nneg) ) )
            {
               /* contract - direction */
               nalu_hypre_BoxIMin(contracted_box)[i] += remainder[i];
            }
            else
            {
               if (nneg)
               {
                  /* contract + direction */
                  nalu_hypre_BoxIMax(contracted_box)[i] -= remainder[i];
               }
            }

         }  /* if (remainder[i]) */
      }     /* for (i= 0; i< ndim; i++) */

      nalu_hypre_BoxDestroy(shifted_box);
   }  /* if (p) */

   nalu_hypre_BoxArrayDestroy(neighbor_boxes);

   return contracted_box;
}
