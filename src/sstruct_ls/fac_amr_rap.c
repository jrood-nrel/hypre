/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"
#include "fac.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_AMR_RAP:  Forms the coarse operators for all amr levels.
 * Given an amr composite matrix, the coarse grid operator is produced.
 * Nesting of amr levels is not assumed. Communication of chunks of the
 * coarse grid operator is performed.
 *
 * Note: The sstruct_grid of A and fac_A are the same. These are kept the
 * same so that the row ranks are the same. However, the generated
 * coarse-grid operators are re-distributed so that each processor has its
 * operator on its grid.
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_AMR_RAP( nalu_hypre_SStructMatrix  *A,
               nalu_hypre_Index          *rfactors,
               nalu_hypre_SStructMatrix **fac_A_ptr )
{

   MPI_Comm                     comm         = nalu_hypre_SStructMatrixComm(A);
   NALU_HYPRE_Int                    ndim         = nalu_hypre_SStructMatrixNDim(A);
   NALU_HYPRE_Int                    nparts       = nalu_hypre_SStructMatrixNParts(A);
   nalu_hypre_SStructGraph          *graph        = nalu_hypre_SStructMatrixGraph(A);
   NALU_HYPRE_IJMatrix               ij_A         = nalu_hypre_SStructMatrixIJMatrix(A);
   NALU_HYPRE_Int                    matrix_type  = nalu_hypre_SStructMatrixObjectType(A);

   nalu_hypre_SStructGrid           *grid         = nalu_hypre_SStructGraphGrid(graph);
   NALU_HYPRE_Int                    nUventries   = nalu_hypre_SStructGraphNUVEntries(graph);
   NALU_HYPRE_Int                   *iUventries   = nalu_hypre_SStructGraphIUVEntries(graph);
   nalu_hypre_SStructUVEntry       **Uventries    = nalu_hypre_SStructGraphUVEntries(graph);
   nalu_hypre_SStructUVEntry        *Uventry;
   NALU_HYPRE_Int                    nUentries;

   nalu_hypre_CommPkg               *amrA_comm_pkg;
   nalu_hypre_CommHandle            *comm_handle;

   nalu_hypre_SStructMatrix         *fac_A;
   nalu_hypre_SStructPMatrix        *pmatrix, *fac_pmatrix;
   nalu_hypre_StructMatrix          *smatrix, *fac_smatrix;
   nalu_hypre_Box                   *smatrix_dbox, *fac_smatrix_dbox;
   NALU_HYPRE_Real                  *smatrix_vals, *fac_smatrix_vals;

   nalu_hypre_SStructGrid           *fac_grid;
   nalu_hypre_SStructGraph          *fac_graph;
   nalu_hypre_SStructPGrid          *f_pgrid, *c_pgrid;
   nalu_hypre_StructGrid            *fgrid, *cgrid;
   nalu_hypre_BoxArray              *grid_boxes, *cgrid_boxes;
   nalu_hypre_Box                   *grid_box;
   nalu_hypre_Box                    scaled_box;

   nalu_hypre_SStructPGrid          *temp_pgrid;
   nalu_hypre_SStructStencil       **temp_sstencils;
   nalu_hypre_SStructPMatrix        *temp_pmatrix;

   nalu_hypre_SStructOwnInfoData  ***owninfo;
   nalu_hypre_SStructRecvInfoData   *recvinfo;
   nalu_hypre_SStructSendInfoData   *sendinfo;
   nalu_hypre_BoxArrayArray         *own_composite_cboxes, *own_boxes;
   nalu_hypre_BoxArray              *own_composite_cbox;
   NALU_HYPRE_Int                  **own_cboxnums;

   nalu_hypre_BoxManager            *fboxman, *cboxman;
   nalu_hypre_BoxManEntry           *boxman_entry;
   nalu_hypre_Index                  ilower;

   NALU_HYPRE_Real                  *values;
   NALU_HYPRE_Int                   *ncols, tot_cols;
   NALU_HYPRE_BigInt                *rows, *cols;

   nalu_hypre_SStructStencil        *stencils;
   nalu_hypre_Index                  stencil_shape, loop_size;
   NALU_HYPRE_Int                    stencil_size, *stencil_vars;

   nalu_hypre_Index                  index, stride, zero_index;
   NALU_HYPRE_Int                    nvars, var1, var2, part, cbox;
   NALU_HYPRE_Int                    i, j, k, size;

   NALU_HYPRE_Int                    myid;
   NALU_HYPRE_Int                    ierr = 0;

   nalu_hypre_MPI_Comm_rank(comm, &myid);
   nalu_hypre_ClearIndex(zero_index);

   nalu_hypre_BoxInit(&scaled_box, ndim);

   nalu_hypre_SStructGraphRef(graph, &fac_graph);
   fac_grid = nalu_hypre_SStructGraphGrid(fac_graph);
   NALU_HYPRE_SStructMatrixCreate(comm, fac_graph, &fac_A);
   NALU_HYPRE_SStructMatrixInitialize(fac_A);

   /*--------------------------------------------------------------------------
    * Copy all A's unstructured data and structured data that are not processed
    * into fac_A. Since the grids are the same for both matrices, the ranks
    * are also the same. Thus, the rows, cols, etc. for the IJ_matrix are
    * the same.
    *--------------------------------------------------------------------------*/
   ncols = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nUventries, NALU_HYPRE_MEMORY_HOST);
   rows = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nUventries, NALU_HYPRE_MEMORY_HOST);

   tot_cols = 0;
   for (i = 0; i < nUventries; i++)
   {
      Uventry = Uventries[iUventries[i]];
      tot_cols += nalu_hypre_SStructUVEntryNUEntries(Uventry);
   }
   cols = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  tot_cols, NALU_HYPRE_MEMORY_HOST);

   k    = 0;
   for (i = 0; i < nUventries; i++)
   {
      Uventry = Uventries[iUventries[i]];
      part   = nalu_hypre_SStructUVEntryPart(Uventry);
      nalu_hypre_CopyIndex(nalu_hypre_SStructUVEntryIndex(Uventry), index);
      var1     = nalu_hypre_SStructUVEntryVar(Uventry);
      nUentries = nalu_hypre_SStructUVEntryNUEntries(Uventry);

      ncols[i] = nUentries;
      nalu_hypre_SStructGridFindBoxManEntry(grid, part, index, var1, &boxman_entry);
      nalu_hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index, &rows[i], matrix_type);

      for (j = 0; j < nUentries; j++)
      {
         cols[k++] = nalu_hypre_SStructUVEntryToRank(Uventry, j);
      }
   }

   values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  tot_cols, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_IJMatrixGetValues(ij_A, nUventries, ncols, rows, cols, values);

   NALU_HYPRE_IJMatrixSetValues(nalu_hypre_SStructMatrixIJMatrix(fac_A), nUventries,
                           ncols, (const NALU_HYPRE_BigInt *) rows, (const NALU_HYPRE_BigInt *) cols,
                           (const NALU_HYPRE_Real *) values);
   nalu_hypre_TFree(ncols, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(rows, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(cols, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   owninfo = nalu_hypre_CTAlloc(nalu_hypre_SStructOwnInfoData  **,  nparts, NALU_HYPRE_MEMORY_HOST);
   for (part = (nparts - 1); part > 0; part--)
   {
      f_pgrid = nalu_hypre_SStructGridPGrid(fac_grid, part);
      c_pgrid = nalu_hypre_SStructGridPGrid(fac_grid, part - 1);

      nvars  = nalu_hypre_SStructPGridNVars(f_pgrid);
      owninfo[part] = nalu_hypre_CTAlloc(nalu_hypre_SStructOwnInfoData   *,  nvars, NALU_HYPRE_MEMORY_HOST);

      for (var1 = 0; var1 < nvars; var1++)
      {
         fboxman = nalu_hypre_SStructGridBoxManager(fac_grid, part, var1);
         cboxman = nalu_hypre_SStructGridBoxManager(fac_grid, part - 1, var1);

         fgrid = nalu_hypre_SStructPGridSGrid(f_pgrid, var1);
         cgrid = nalu_hypre_SStructPGridSGrid(c_pgrid, var1);

         owninfo[part][var1] = nalu_hypre_SStructOwnInfo(fgrid, cgrid, cboxman, fboxman,
                                                    rfactors[part]);
      }
   }

   nalu_hypre_SetIndex3(stride, 1, 1, 1);
   for (part = (nparts - 1); part > 0; part--)
   {
      f_pgrid = nalu_hypre_SStructGridPGrid(fac_grid, part);
      c_pgrid = nalu_hypre_SStructGridPGrid(fac_grid, part - 1);
      nvars  = nalu_hypre_SStructPGridNVars(f_pgrid);

      for (var1 = 0; var1 < nvars; var1++)
      {
         fgrid     = nalu_hypre_SStructPGridSGrid(f_pgrid, var1);
         cgrid     = nalu_hypre_SStructPGridSGrid(c_pgrid, var1);
         grid_boxes = nalu_hypre_StructGridBoxes(fgrid);

         stencils = nalu_hypre_SStructGraphStencil(graph, part, var1);
         stencil_size = nalu_hypre_SStructStencilSize(stencils);
         stencil_vars = nalu_hypre_SStructStencilVars(stencils);

         if (part == (nparts - 1)) /* copy all fine data */
         {
            pmatrix    = nalu_hypre_SStructMatrixPMatrix(A, part);
            fac_pmatrix = nalu_hypre_SStructMatrixPMatrix(fac_A, part);
            nalu_hypre_ForBoxI(i, grid_boxes)
            {
               grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);
               nalu_hypre_BoxGetSize(grid_box, loop_size);
               nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(grid_box), ilower);

               for (j = 0; j < stencil_size; j++)
               {
                  var2       = stencil_vars[j];
                  smatrix    = nalu_hypre_SStructPMatrixSMatrix(pmatrix, var1, var2);
                  fac_smatrix = nalu_hypre_SStructPMatrixSMatrix(fac_pmatrix, var1, var2);

                  smatrix_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(smatrix),
                                                   i);
                  fac_smatrix_dbox =
                     nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(fac_smatrix), i);

                  nalu_hypre_CopyIndex(nalu_hypre_SStructStencilEntry(stencils, j), stencil_shape);
                  smatrix_vals = nalu_hypre_StructMatrixExtractPointerByIndex(smatrix,
                                                                         i,
                                                                         stencil_shape);
                  fac_smatrix_vals = nalu_hypre_StructMatrixExtractPointerByIndex(fac_smatrix,
                                                                             i,
                                                                             stencil_shape);

#define DEVICE_VAR is_device_ptr(fac_smatrix_vals,smatrix_vals)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      smatrix_dbox, ilower, stride, iA,
                                      fac_smatrix_dbox, ilower, stride, iAc);
                  {
                     fac_smatrix_vals[iAc] = smatrix_vals[iA];
                  }
                  nalu_hypre_BoxLoop2End(iA, iAc);
#undef DEVICE_VAR

               }  /* for (j = 0; j < stencil_size; j++) */
            }     /* nalu_hypre_ForBoxI(i, grid_boxes) */
         }        /* if (part == (nparts-1)) */

         /*----------------------------------------------------------------------
          *  Copy all coarse data not underlying a fbox and on this processor-
          *  i.e., the own_composite_cbox data.
          *----------------------------------------------------------------------*/
         pmatrix    = nalu_hypre_SStructMatrixPMatrix(A, part - 1);
         fac_pmatrix = nalu_hypre_SStructMatrixPMatrix(fac_A, part - 1);

         own_composite_cboxes = nalu_hypre_SStructOwnInfoDataCompositeCBoxes(owninfo[part][var1]);

         stencils = nalu_hypre_SStructGraphStencil(graph, part - 1, var1);
         stencil_size = nalu_hypre_SStructStencilSize(stencils);
         stencil_vars = nalu_hypre_SStructStencilVars(stencils);

         nalu_hypre_ForBoxArrayI(i, own_composite_cboxes)
         {
            own_composite_cbox = nalu_hypre_BoxArrayArrayBoxArray(own_composite_cboxes, i);
            nalu_hypre_ForBoxI(j, own_composite_cbox)
            {
               grid_box = nalu_hypre_BoxArrayBox(own_composite_cbox, j);
               nalu_hypre_BoxGetSize(grid_box, loop_size);
               nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(grid_box), ilower);

               for (k = 0; k < stencil_size; k++)
               {
                  var2       = stencil_vars[k];
                  smatrix    = nalu_hypre_SStructPMatrixSMatrix(pmatrix, var1, var2);
                  fac_smatrix = nalu_hypre_SStructPMatrixSMatrix(fac_pmatrix, var1, var2);

                  smatrix_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(smatrix),
                                                   i);
                  fac_smatrix_dbox =
                     nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(fac_smatrix), i);

                  nalu_hypre_CopyIndex(nalu_hypre_SStructStencilEntry(stencils, k), stencil_shape);
                  smatrix_vals = nalu_hypre_StructMatrixExtractPointerByIndex(smatrix,
                                                                         i,
                                                                         stencil_shape);
                  fac_smatrix_vals = nalu_hypre_StructMatrixExtractPointerByIndex(fac_smatrix,
                                                                             i,
                                                                             stencil_shape);

#define DEVICE_VAR is_device_ptr(fac_smatrix_vals,smatrix_vals)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      smatrix_dbox, ilower, stride, iA,
                                      fac_smatrix_dbox, ilower, stride, iAc);
                  {
                     fac_smatrix_vals[iAc] = smatrix_vals[iA];
                  }
                  nalu_hypre_BoxLoop2End(iA, iAc);
#undef DEVICE_VAR

               }  /* for (k = 0; k< stencil_size; k++) */
            }      /* nalu_hypre_ForBoxI(j, own_composite_cbox) */
         }          /* nalu_hypre_ForBoxArrayI(i, own_composite_cboxes) */

      }  /* for (var1= 0; var1< nvars; var1++) */
   }     /* for (part= (nparts-1); part> 0; part--) */

   /*--------------------------------------------------------------------------
    * All possible data has been copied into fac_A- i.e., the original amr
    * composite operator. Now we need to coarsen away the fboxes and the
    * interface connections.
    *
    * Algo.:
    *   Loop from the finest amr_level to amr_level 1
    *   {
    *      1) coarsen the cf connections to get stencil connections from
    *         the coarse nodes to the coarsened fbox nodes.
    *      2) coarsen the fboxes and the fc connections. These are coarsened
    *         into a temp SStruct_PMatrix whose grid is the coarsened fgrid.
    *      3) copy all coarsened data that belongs on this processor and
    *         communicate any that belongs to another processor.
    *   }
    *--------------------------------------------------------------------------*/
   for (part = (nparts - 1); part >= 1; part--)
   {
      nalu_hypre_AMR_CFCoarsen(A, fac_A, rfactors[part], part);

      /*-----------------------------------------------------------------------
       *  Create the temp SStruct_PMatrix for coarsening away the level= part
       *  boxes.
       *-----------------------------------------------------------------------*/
      f_pgrid = nalu_hypre_SStructGridPGrid(fac_grid, part);
      c_pgrid = nalu_hypre_SStructGridPGrid(fac_grid, part - 1);
      grid_boxes = nalu_hypre_SStructPGridCellIBoxArray(f_pgrid);

      nalu_hypre_SStructPGridCreate(nalu_hypre_SStructGridComm(f_pgrid),
                               ndim, &temp_pgrid);

      /*coarsen the fboxes.*/
      for (i = 0; i < nalu_hypre_BoxArraySize(grid_boxes); i++)
      {
         grid_box = nalu_hypre_BoxArrayBox(grid_boxes, i);
         nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(grid_box), zero_index,
                                     rfactors[part], nalu_hypre_BoxIMin(&scaled_box));
         nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(grid_box), zero_index,
                                     rfactors[part], nalu_hypre_BoxIMax(&scaled_box));

         nalu_hypre_SStructPGridSetExtents(temp_pgrid,
                                      nalu_hypre_BoxIMin(&scaled_box),
                                      nalu_hypre_BoxIMax(&scaled_box));
      }

      nvars  = nalu_hypre_SStructPGridNVars(f_pgrid);
      nalu_hypre_SStructPGridSetVariables(temp_pgrid, nvars,
                                     nalu_hypre_SStructPGridVarTypes(f_pgrid));
      nalu_hypre_SStructPGridAssemble(temp_pgrid);

      /* reference the sstruct_stencil of fac_pmatrix- to be used in temp_pmatrix */
      temp_sstencils = nalu_hypre_CTAlloc(nalu_hypre_SStructStencil *,  nvars, NALU_HYPRE_MEMORY_HOST);
      fac_pmatrix = nalu_hypre_SStructMatrixPMatrix(fac_A, part - 1);
      for (i = 0; i < nvars; i++)
      {
         nalu_hypre_SStructStencilRef(nalu_hypre_SStructPMatrixStencil(fac_pmatrix, i),
                                 &temp_sstencils[i]);
      }

      nalu_hypre_SStructPMatrixCreate(nalu_hypre_SStructPMatrixComm(fac_pmatrix),
                                 temp_pgrid,
                                 temp_sstencils,
                                 &temp_pmatrix);
      nalu_hypre_SStructPMatrixInitialize(temp_pmatrix);

      nalu_hypre_AMR_FCoarsen(A, fac_A, temp_pmatrix, rfactors[part], part);

      /*-----------------------------------------------------------------------
       * Extract the own_box data (boxes of coarsen data of this processor).
       *-----------------------------------------------------------------------*/
      fac_pmatrix = nalu_hypre_SStructMatrixPMatrix(fac_A, part - 1);
      for (var1 = 0; var1 < nvars; var1++)
      {
         stencils = nalu_hypre_SStructGraphStencil(graph, part - 1, var1);
         stencil_size = nalu_hypre_SStructStencilSize(stencils);
         stencil_vars = nalu_hypre_SStructStencilVars(stencils);

         own_boxes = nalu_hypre_SStructOwnInfoDataOwnBoxes(owninfo[part][var1]);
         own_cboxnums = nalu_hypre_SStructOwnInfoDataOwnBoxNums(owninfo[part][var1]);
         size = nalu_hypre_SStructOwnInfoDataSize(owninfo[part][var1]);

         /* loop over all the cbox chunks */
         for (i = 0; i < size; i++)
         {
            cgrid_boxes = nalu_hypre_BoxArrayArrayBoxArray(own_boxes, i);
            nalu_hypre_ForBoxI(j, cgrid_boxes)
            {
               grid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, j);
               nalu_hypre_BoxGetSize(grid_box, loop_size);
               nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(grid_box), ilower);

               cbox = own_cboxnums[i][j];

               for (k = 0; k < stencil_size; k++)
               {
                  var2 = stencil_vars[k];
                  smatrix = nalu_hypre_SStructPMatrixSMatrix(temp_pmatrix, var1, var2);
                  fac_smatrix = nalu_hypre_SStructPMatrixSMatrix(fac_pmatrix, var1, var2);

                  /*---------------------------------------------------------------
                   * note: the cbox number of the temp_grid is the same as the
                   * fbox number, whereas the cbox numbers of the fac_grid is in
                   * own_cboxnums- i.e., numbers i & cbox, respectively.
                   *---------------------------------------------------------------*/
                  smatrix_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(smatrix),
                                                   i);
                  fac_smatrix_dbox =
                     nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(fac_smatrix), cbox);

                  nalu_hypre_CopyIndex(nalu_hypre_SStructStencilEntry(stencils, k), stencil_shape);
                  smatrix_vals =
                     nalu_hypre_StructMatrixExtractPointerByIndex(smatrix,
                                                             i,
                                                             stencil_shape);
                  fac_smatrix_vals =
                     nalu_hypre_StructMatrixExtractPointerByIndex(fac_smatrix,
                                                             cbox,
                                                             stencil_shape);

#define DEVICE_VAR is_device_ptr(fac_smatrix_vals,smatrix_vals)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      smatrix_dbox, ilower, stride, iA,
                                      fac_smatrix_dbox, ilower, stride, iAc);
                  {
                     fac_smatrix_vals[iAc] = smatrix_vals[iA];
                  }
                  nalu_hypre_BoxLoop2End(iA, iAc);
#undef DEVICE_VAR

               }  /* for (k = 0; k < stencil_size; k++) */
            }     /* nalu_hypre_ForBoxI(j, cgrid_boxes) */
         }        /* for (i= 0; i< size; i++) */

         nalu_hypre_SStructOwnInfoDataDestroy(owninfo[part][var1]);
      }           /* for (var1= 0; var1< nvars; var1++) */

      nalu_hypre_TFree(owninfo[part], NALU_HYPRE_MEMORY_HOST);

      /*-----------------------------------------------------------------------
       * Communication of off-process coarse data. A communication pkg is
       * needed. Thus, compute the communication info- sendboxes, recvboxes,
       * etc.
       *-----------------------------------------------------------------------*/
      for (var1 = 0; var1 < nvars; var1++)
      {
         fboxman = nalu_hypre_SStructGridBoxManager(fac_grid, part, var1);
         cboxman = nalu_hypre_SStructGridBoxManager(fac_grid, part - 1, var1);

         fgrid = nalu_hypre_SStructPGridSGrid(f_pgrid, var1);
         cgrid = nalu_hypre_SStructPGridSGrid(c_pgrid, var1);

         sendinfo = nalu_hypre_SStructSendInfo(fgrid, cboxman, rfactors[part]);
         recvinfo = nalu_hypre_SStructRecvInfo(cgrid, fboxman, rfactors[part]);

         /*-------------------------------------------------------------------
          * need to check this for more than one variable- are the comm. info
          * for this sgrid okay for cross-variable matrices?
          *-------------------------------------------------------------------*/
         for (var2 = 0; var2 < nvars; var2++)
         {
            fac_smatrix = nalu_hypre_SStructPMatrixSMatrix(fac_pmatrix, var1, var2);
            smatrix    = nalu_hypre_SStructPMatrixSMatrix(temp_pmatrix, var1, var2);

            nalu_hypre_SStructAMRInterCommunication(sendinfo,
                                               recvinfo,
                                               nalu_hypre_StructMatrixDataSpace(smatrix),
                                               nalu_hypre_StructMatrixDataSpace(fac_smatrix),
                                               nalu_hypre_StructMatrixNumValues(smatrix),
                                               comm,
                                               &amrA_comm_pkg);

            nalu_hypre_InitializeCommunication(amrA_comm_pkg,
                                          nalu_hypre_StructMatrixData(smatrix),
                                          nalu_hypre_StructMatrixData(fac_smatrix), 0, 0,
                                          &comm_handle);
            nalu_hypre_FinalizeCommunication(comm_handle);

            nalu_hypre_CommPkgDestroy(amrA_comm_pkg);
         }

         nalu_hypre_SStructSendInfoDataDestroy(sendinfo);
         nalu_hypre_SStructRecvInfoDataDestroy(recvinfo);

      }  /* for (var1= 0; var1< nvars; var1++) */

      nalu_hypre_SStructPGridDestroy(temp_pgrid);
      nalu_hypre_SStructPMatrixDestroy(temp_pmatrix);

   }  /* for (part= 0; part< nparts; part++) */

   nalu_hypre_TFree(owninfo, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_SStructMatrixAssemble(fac_A);

   *fac_A_ptr = fac_A;
   return ierr;
}
