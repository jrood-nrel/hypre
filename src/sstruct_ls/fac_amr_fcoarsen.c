/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * OpenMP Problems
 *
 * Need to fix the way these variables are set and incremented in loops:
 *   vals
 *
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"
#include "fac.h"

#define MapStencilRank(stencil, rank)           \
   {                                            \
      NALU_HYPRE_Int ii,jj,kk;                       \
      ii = nalu_hypre_IndexX(stencil);               \
      jj = nalu_hypre_IndexY(stencil);               \
      kk = nalu_hypre_IndexZ(stencil);               \
      if (ii==-1)                               \
         ii=2;                                  \
      if (jj==-1)                               \
         jj=2;                                  \
      if (kk==-1)                               \
         kk=2;                                  \
      rank = ii + 3*jj + 9*kk;                  \
   }

#define InverseMapStencilRank(rank, stencil)    \
   {                                            \
      NALU_HYPRE_Int ij,ii,jj,kk;                    \
      ij = (rank%9);                            \
      ii = (ij%3);                              \
      jj = (ij-ii)/3;                           \
      kk = (rank-3*jj-ii)/9;                    \
      if (ii==2)                                \
         ii= -1;                                \
      if (jj==2)                                \
         jj= -1;                                \
      if (kk==2)                                \
         kk= -1;                                \
      nalu_hypre_SetIndex3(stencil, ii, jj, kk);     \
   }


#define AbsStencilShape(stencil, abs_shape)                     \
   {                                                            \
      NALU_HYPRE_Int ii,jj,kk;                                       \
      ii = nalu_hypre_IndexX(stencil);                               \
      jj = nalu_hypre_IndexY(stencil);                               \
      kk = nalu_hypre_IndexZ(stencil);                               \
      abs_shape= nalu_hypre_abs(ii) + nalu_hypre_abs(jj) + nalu_hypre_abs(kk); \
   }

/*--------------------------------------------------------------------------
 * nalu_hypre_AMR_FCoarsen: Coarsen the fbox and f/c connections. Forms the
 * coarse operator by averaging neighboring connections in the refinement
 * patch.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AMR_FCoarsen( nalu_hypre_SStructMatrix  *   A,
                    nalu_hypre_SStructMatrix  *   fac_A,
                    nalu_hypre_SStructPMatrix *   A_crse,
                    nalu_hypre_Index              refine_factors,
                    NALU_HYPRE_Int                level )

{
   nalu_hypre_Box               fine_box;
   nalu_hypre_Box               intersect_box;

   MPI_Comm                comm       = nalu_hypre_SStructMatrixComm(A);

   nalu_hypre_SStructGraph     *graph      = nalu_hypre_SStructMatrixGraph(A);
   NALU_HYPRE_Int               graph_type = nalu_hypre_SStructGraphObjectType(graph);
   nalu_hypre_SStructGrid      *grid       = nalu_hypre_SStructGraphGrid(graph);
   NALU_HYPRE_IJMatrix          ij_A       = nalu_hypre_SStructMatrixIJMatrix(A);
   NALU_HYPRE_Int               matrix_type = nalu_hypre_SStructMatrixObjectType(A);
   NALU_HYPRE_Int               ndim       = nalu_hypre_SStructMatrixNDim(A);

   nalu_hypre_SStructPMatrix   *A_pmatrix  = nalu_hypre_SStructMatrixPMatrix(fac_A, level);

   nalu_hypre_StructMatrix     *smatrix_var;
   nalu_hypre_StructStencil    *stencils, *stencils_last;
   NALU_HYPRE_Int               stencil_size, stencil_last_size;
   nalu_hypre_Index             stencil_shape_i, stencil_last_shape_i;
   nalu_hypre_Index             loop_size;
   nalu_hypre_Box               loop_box;
   NALU_HYPRE_Real            **a_ptrs;
   nalu_hypre_Box              *A_dbox;

   NALU_HYPRE_Int               part_crse = level - 1;
   NALU_HYPRE_Int               part_fine = level;

   nalu_hypre_StructMatrix     *crse_smatrix;
   NALU_HYPRE_Real             *crse_ptr;
   NALU_HYPRE_Real            **crse_ptrs;
   nalu_hypre_Box              *crse_dbox;

   nalu_hypre_StructGrid       *cgrid;
   nalu_hypre_BoxArray         *cgrid_boxes;
   nalu_hypre_Box              *cgrid_box;
   nalu_hypre_Index             cstart;
   nalu_hypre_Index             fstart, fend;
   nalu_hypre_Index             stridec, stridef;

   nalu_hypre_StructGrid       *fgrid;
   nalu_hypre_BoxArray         *fgrid_boxes;
   nalu_hypre_Box              *fgrid_box;
   nalu_hypre_BoxArray       ***fgrid_crse_extents;
   nalu_hypre_BoxArray       ***fbox_interior;
   nalu_hypre_BoxArrayArray  ***fbox_bdy;
   NALU_HYPRE_Int            ***interior_fboxi;
   NALU_HYPRE_Int            ***bdy_fboxi;
   NALU_HYPRE_Int            ***cboxi_fboxes;
   NALU_HYPRE_Int             **cboxi_fcnt;

   nalu_hypre_BoxArray         *fbox_interior_ci, *fbox_bdy_ci_fi;
   nalu_hypre_BoxArrayArray    *fbox_bdy_ci;
   NALU_HYPRE_Int              *interior_fboxi_ci;
   NALU_HYPRE_Int              *bdy_fboxi_ci;

   NALU_HYPRE_Int               centre;

   nalu_hypre_BoxArray         *data_space;

   NALU_HYPRE_Int               ci, fi, arrayi;
   NALU_HYPRE_Int               max_stencil_size = 27;
   NALU_HYPRE_Int               trueV = 1;
   NALU_HYPRE_Int               falseV = 0;
   NALU_HYPRE_Int               found, sort;
   NALU_HYPRE_Int               stencil_marker;
   NALU_HYPRE_Int              *stencil_ranks, *rank_stencils;
   NALU_HYPRE_Int              *stencil_contrib_cnt;
   NALU_HYPRE_Int             **stencil_contrib_i;
   NALU_HYPRE_Real            **weight_contrib_i;
   NALU_HYPRE_Real              weights[4] = {1.0, 0.25, 0.125, 0.0625};
   NALU_HYPRE_Real              sum;
   NALU_HYPRE_Int               abs_stencil_shape;
   nalu_hypre_Box             **shift_box;
   nalu_hypre_Box               coarse_cell_box;
   NALU_HYPRE_Int               volume_coarse_cell_box;
   NALU_HYPRE_Int              *volume_shift_box;
   NALU_HYPRE_Int               max_contribut_size, stencil_i;
   NALU_HYPRE_BigInt            startrank, rank;
   NALU_HYPRE_Real             *vals, *vals2;

   NALU_HYPRE_Int               i, j, k, l, m, n, ll, kk, jj;
   NALU_HYPRE_Int               nvars, var1, var2, var2_start;
   NALU_HYPRE_Int               iA_shift_z, iA_shift_zy, iA_shift_zyx;

   nalu_hypre_Index             lindex;
   nalu_hypre_Index             index1, index2;
   nalu_hypre_Index             index_temp;

   NALU_HYPRE_Int             **box_graph_indices;
   NALU_HYPRE_Int              *box_graph_cnts;
   NALU_HYPRE_Int              *box_ranks, *box_ranks_cnt, *box_to_ranks_cnt;
   NALU_HYPRE_Int              *cdata_space_ranks, *box_starts, *box_ends;
   NALU_HYPRE_Int              *box_connections;
   NALU_HYPRE_Int             **coarse_contrib_Uv;
   NALU_HYPRE_Int              *fine_interface_ranks;
   NALU_HYPRE_Int               nUventries = nalu_hypre_SStructGraphNUVEntries(graph);
   NALU_HYPRE_Int              *iUventries  = nalu_hypre_SStructGraphIUVEntries(graph);
   nalu_hypre_SStructUVEntry  **Uventries   = nalu_hypre_SStructGraphUVEntries(graph);
   nalu_hypre_SStructUVEntry   *Uventry;
   NALU_HYPRE_Int               nUentries, cnt1;
   nalu_hypre_Index             index, *cindex, *Uv_cindex;
   NALU_HYPRE_Int               box_array_size, cbox_array_size;

   NALU_HYPRE_Int               nrows;
   NALU_HYPRE_BigInt            to_rank;
   NALU_HYPRE_Int              *ncols;
   NALU_HYPRE_BigInt           *rows, *cols;
   NALU_HYPRE_Int             **interface_max_stencil_ranks;
   NALU_HYPRE_Int             **interface_max_stencil_cnt;
   NALU_HYPRE_Int             **interface_rank_stencils;
   NALU_HYPRE_Int             **interface_stencil_ranks;
   NALU_HYPRE_Int              *coarse_stencil_cnt;
   NALU_HYPRE_Real             *stencil_vals;
   NALU_HYPRE_Int              *common_rank_stencils, *common_stencil_ranks;
   NALU_HYPRE_Int              *common_stencil_i;
   nalu_hypre_BoxManEntry      *boxman_entry;

   NALU_HYPRE_Int              *temp1, *temp2;
   NALU_HYPRE_Real             *temp3;
   NALU_HYPRE_Real              sum_contrib, scaling;

   NALU_HYPRE_Int             **OffsetA;

   NALU_HYPRE_Int              *parents;
   NALU_HYPRE_Int              *parents_cnodes;

   NALU_HYPRE_Int               myid;

   nalu_hypre_MPI_Comm_rank(comm, &myid);

   nalu_hypre_BoxInit(&fine_box, ndim);
   nalu_hypre_BoxInit(&intersect_box, ndim);
   nalu_hypre_BoxInit(&loop_box, ndim);
   nalu_hypre_BoxInit(&coarse_cell_box, ndim);

   /*--------------------------------------------------------------------------
    * Task: Coarsen the fbox and f/c connections to form the coarse grid
    * operator inside the fgrid.
    *--------------------------------------------------------------------------*/

   if (graph_type == NALU_HYPRE_SSTRUCT)
   {
      startrank   = nalu_hypre_SStructGridGhstartRank(grid);
   }
   if (graph_type == NALU_HYPRE_PARCSR)
   {
      startrank   = nalu_hypre_SStructGridStartRank(grid);
   }

   /*--------------------------------------------------------------------------
    * Fine grid strides by the refinement factors.
    *--------------------------------------------------------------------------*/
   nalu_hypre_SetIndex3(stridec, 1, 1, 1);
   for (i = 0; i < ndim; i++)
   {
      stridef[i] = refine_factors[i];
   }
   for (i = ndim; i < 3; i++)
   {
      stridef[i] = 1;
   }

   /*--------------------------------------------------------------------------
    * Scaling for averaging row sum.
    *--------------------------------------------------------------------------*/
   scaling = 1.0;
   for (i = 0; i < ndim - 2; i++)
   {
      scaling *= refine_factors[0];
   }

   /*--------------------------------------------------------------------------
    *  Determine the coarsened fine grid- fgrid_crse_extents.
    *  These are between fpart= level and cpart= (level-1). The
    *  fgrid_crse_extents will be indexed by cboxes- the boxarray of coarsened
    *  fboxes FULLY in a given cbox.
    *
    *  Also, determine the interior and boundary boxes of each fbox. Having
    *  these will allow us to determine the f/c interface nodes without
    *  extensive checking. These are also indexed by the cboxes.
    *    fgrid_interior- for each cbox, we have a collection of child fboxes,
    *                    each leading to an interior=> boxarray
    *    fgrid_bdy     - for each cbox, we have a collection of child fboxes,
    *                    each leading to a boxarray of bdies=> boxarrayarray.
    *  Because we need to know the fbox id for these boxarray/boxarrayarray,
    *  we will need one for each fbox.
    *
    *  And, determine which cboxes contain a given fbox. That is, given a
    *  fbox, find all cboxes that contain a chunk of it.
    *--------------------------------------------------------------------------*/
   nvars    =  nalu_hypre_SStructPMatrixNVars(A_pmatrix);

   fgrid_crse_extents      = nalu_hypre_TAlloc(nalu_hypre_BoxArray **,  nvars, NALU_HYPRE_MEMORY_HOST);
   fbox_interior           = nalu_hypre_TAlloc(nalu_hypre_BoxArray **,  nvars, NALU_HYPRE_MEMORY_HOST);
   fbox_bdy                = nalu_hypre_TAlloc(nalu_hypre_BoxArrayArray **,  nvars, NALU_HYPRE_MEMORY_HOST);
   interior_fboxi          = nalu_hypre_TAlloc(NALU_HYPRE_Int **,  nvars, NALU_HYPRE_MEMORY_HOST);
   bdy_fboxi               = nalu_hypre_TAlloc(NALU_HYPRE_Int **,  nvars, NALU_HYPRE_MEMORY_HOST);
   cboxi_fboxes            = nalu_hypre_TAlloc(NALU_HYPRE_Int **,  nvars, NALU_HYPRE_MEMORY_HOST);
   cboxi_fcnt              = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nvars, NALU_HYPRE_MEMORY_HOST);

   for (var1 = 0; var1 < nvars; var1++)
   {
      cgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructPMatrixPGrid(A_crse), var1);
      cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);
      fgrid_crse_extents[var1] = nalu_hypre_TAlloc(nalu_hypre_BoxArray *,
                                              nalu_hypre_BoxArraySize(cgrid_boxes), NALU_HYPRE_MEMORY_HOST);
      fbox_interior[var1] = nalu_hypre_TAlloc(nalu_hypre_BoxArray *,
                                         nalu_hypre_BoxArraySize(cgrid_boxes), NALU_HYPRE_MEMORY_HOST);
      fbox_bdy[var1]     = nalu_hypre_TAlloc(nalu_hypre_BoxArrayArray *,
                                        nalu_hypre_BoxArraySize(cgrid_boxes), NALU_HYPRE_MEMORY_HOST);
      interior_fboxi[var1] = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(cgrid_boxes),
                                          NALU_HYPRE_MEMORY_HOST);
      bdy_fboxi[var1]     = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(cgrid_boxes),
                                         NALU_HYPRE_MEMORY_HOST);

      fgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructPMatrixPGrid(A_pmatrix), var1);
      fgrid_boxes = nalu_hypre_StructGridBoxes(fgrid);

      cboxi_fboxes[var1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  nalu_hypre_BoxArraySize(fgrid_boxes),
                                         NALU_HYPRE_MEMORY_HOST);
      cboxi_fcnt[var1]  = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(fgrid_boxes), NALU_HYPRE_MEMORY_HOST);

      /*-----------------------------------------------------------------------
       *  Determine the fine grid boxes that are underlying a coarse grid box.
       *  Coarsen the indices to determine the looping extents of these
       *  boxes. Also, find the looping extents for the extended coarsened
       *  boxes, and the interior and boundary extents of a fine_grid box.
       *  The fine_grid boxes must be adjusted so that only the coarse nodes
       *  inside these boxes are included. Only the lower bound needs to be
       *  adjusted.
       *-----------------------------------------------------------------------*/
      nalu_hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(cgrid_box), cstart);

         cnt1 = 0;
         temp1 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(fgrid_boxes), NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_ClearIndex(index_temp);
         nalu_hypre_ForBoxI(fi, fgrid_boxes)
         {
            fgrid_box = nalu_hypre_BoxArrayBox(fgrid_boxes, fi);
            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(fgrid_box), fstart);
            for (i = 0; i < ndim; i++)
            {
               j = fstart[i] % refine_factors[i];
               if (j)
               {
                  fstart[i] += refine_factors[i] - j;
               }
            }

            nalu_hypre_StructMapFineToCoarse(fstart, index_temp,
                                        refine_factors, nalu_hypre_BoxIMin(&fine_box));
            nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(fgrid_box), index_temp,
                                        refine_factors, nalu_hypre_BoxIMax(&fine_box));

            nalu_hypre_IntersectBoxes(&fine_box, cgrid_box, &intersect_box);
            if (nalu_hypre_BoxVolume(&intersect_box) > 0)
            {
               temp1[cnt1++] = fi;
            }
         }

         fgrid_crse_extents[var1][ci] = nalu_hypre_BoxArrayCreate(cnt1, ndim);
         fbox_interior[var1][ci]  = nalu_hypre_BoxArrayCreate(cnt1, ndim);
         fbox_bdy[var1][ci]       = nalu_hypre_BoxArrayArrayCreate(cnt1, ndim);
         interior_fboxi[var1][ci] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt1, NALU_HYPRE_MEMORY_HOST);
         bdy_fboxi[var1][ci]      = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt1, NALU_HYPRE_MEMORY_HOST);

         for (fi = 0; fi < cnt1; fi++)
         {
            fgrid_box = nalu_hypre_BoxArrayBox(fgrid_boxes, temp1[fi]);
            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(fgrid_box), fstart);
            nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(fgrid_box), fend);

            /*--------------------------------------------------------------------
             * record which sides will be adjusted- fstart adjustments will
             * decrease the box size, whereas fend adjustments will increase the
             * box size. Since we fstart decreases the box size, we cannot
             * have an f/c interface at an adjusted fstart end. fend may
             * correspond to an f/c interface whether it has been adjusted or not.
             *--------------------------------------------------------------------*/
            nalu_hypre_SetIndex3(index1, 1, 1, 1);
            for (i = 0; i < ndim; i++)
            {
               j = fstart[i] % refine_factors[i];
               if (j)
               {
                  fstart[i] += refine_factors[i] - j;
                  index1[i] = 0;
               }

               j = fend[i] % refine_factors[i];
               if (refine_factors[i] - 1 - j)
               {
                  fend[i] += (refine_factors[i] - 1) - j;
               }
            }

            nalu_hypre_StructMapFineToCoarse(fstart, index_temp,
                                        refine_factors, nalu_hypre_BoxIMin(&fine_box));
            nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(fgrid_box), index_temp,
                                        refine_factors, nalu_hypre_BoxIMax(&fine_box));
            nalu_hypre_IntersectBoxes(&fine_box, cgrid_box, &intersect_box);

            nalu_hypre_CopyBox(&intersect_box,
                          nalu_hypre_BoxArrayBox(fgrid_crse_extents[var1][ci], fi));

            /*--------------------------------------------------------------------
             * adjust the fine intersect_box so that we get the interior and
             * boundaries separately.
             *--------------------------------------------------------------------*/
            nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMin(&intersect_box), index_temp,
                                        refine_factors, nalu_hypre_BoxIMin(&fine_box));

            /* the following index2 shift for ndim<3 is no problem since
               refine_factors[j]= 1 for j>=ndim. */
            nalu_hypre_SetIndex3(index2, refine_factors[0] - 1, refine_factors[1] - 1,
                            refine_factors[2] - 1);
            nalu_hypre_StructMapCoarseToFine(nalu_hypre_BoxIMax(&intersect_box), index2,
                                        refine_factors, nalu_hypre_BoxIMax(&fine_box));

            nalu_hypre_SetIndex3(index2, 1, 1, 1);
            nalu_hypre_CopyBox(&fine_box, &loop_box);
            for (i = 0; i < ndim; i++)
            {
               nalu_hypre_BoxIMin(&loop_box)[i] += refine_factors[i] * index1[i];
               nalu_hypre_BoxIMax(&loop_box)[i] -= refine_factors[i] * index2[i];
            }
            nalu_hypre_CopyBox(&loop_box,
                          nalu_hypre_BoxArrayBox(fbox_interior[var1][ci], fi));
            interior_fboxi[var1][ci][fi] = temp1[fi];

            nalu_hypre_SubtractBoxes(&fine_box, &loop_box,
                                nalu_hypre_BoxArrayArrayBoxArray(fbox_bdy[var1][ci], fi));
            bdy_fboxi[var1][ci][fi] = temp1[fi];
         }
         nalu_hypre_TFree(temp1, NALU_HYPRE_MEMORY_HOST);

      }  /* nalu_hypre_ForBoxI(ci, cgrid_boxes) */

      /*--------------------------------------------------------------------
       * Determine the cboxes that contain a chunk of a given fbox.
       *--------------------------------------------------------------------*/
      nalu_hypre_ForBoxI(fi, fgrid_boxes)
      {
         fgrid_box = nalu_hypre_BoxArrayBox(fgrid_boxes, fi);
         nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(fgrid_box), fstart);
         for (i = 0; i < ndim; i++)
         {
            j = fstart[i] % refine_factors[i];
            if (j)
            {
               fstart[i] += refine_factors[i] - j;
            }
         }

         nalu_hypre_StructMapFineToCoarse(fstart, index_temp,
                                     refine_factors, nalu_hypre_BoxIMin(&fine_box));
         nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(fgrid_box), index_temp,
                                     refine_factors, nalu_hypre_BoxIMax(&fine_box));

         temp1 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(cgrid_boxes), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ForBoxI(i, cgrid_boxes)
         {
            cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, i);
            nalu_hypre_IntersectBoxes(&fine_box, cgrid_box, &intersect_box);
            if (nalu_hypre_BoxVolume(&intersect_box) > 0)
            {
               temp1[cboxi_fcnt[var1][fi]] = i;
               cboxi_fcnt[var1][fi]++;
            }
         }

         cboxi_fboxes[var1][fi] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  cboxi_fcnt[var1][fi], NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < cboxi_fcnt[var1][fi]; i++)
         {
            cboxi_fboxes[var1][fi][i] = temp1[i];
         }
         nalu_hypre_TFree(temp1, NALU_HYPRE_MEMORY_HOST);
      }
   }     /* for (var1= 0; var1< nvars; var1++) */

   /*--------------------------------------------------------------------------
    *  STEP 1:
    *        COMPUTE THE COARSE LEVEL OPERATOR INSIDE OF A REFINED BOX.
    *
    *  We assume that the coarse and fine grid variables are of the same type.
    *
    *  Coarse stencils in the refinement patches are obtained by averaging the
    *  fine grid coefficients. Since we are assuming cell-centred discretization,
    *  we apply a weighted averaging of ONLY the fine grid coefficients along
    *  interfaces of adjacent agglomerated coarse cells.
    *
    *  Since the stencil pattern is assumed arbitrary, we must determine the
    *  stencil pattern of each var1-var2 struct_matrix to get the correct
    *  contributing stencil coefficients, averaging weights, etc.
    *--------------------------------------------------------------------------*/

   /*--------------------------------------------------------------------------
    *  Agglomerated coarse cell info. These are needed in defining the looping
    *  extents for averaging- i.e., we loop over extents determined by the
    *  size of the agglomerated coarse cell.
    *  Note that the agglomerated coarse cell is constructed correctly for
    *  any dimensions (1, 2, or 3).
    *--------------------------------------------------------------------------*/
   nalu_hypre_ClearIndex(index_temp);
   nalu_hypre_CopyIndex(index_temp, nalu_hypre_BoxIMin(&coarse_cell_box));
   nalu_hypre_SetIndex3(index_temp, refine_factors[0] - 1, refine_factors[1] - 1,
                   refine_factors[2] - 1 );
   nalu_hypre_CopyIndex(index_temp, nalu_hypre_BoxIMax(&coarse_cell_box));

   volume_coarse_cell_box = nalu_hypre_BoxVolume(&coarse_cell_box);


   /*--------------------------------------------------------------------------
    * Offsets in y & z directions for refinement patches. These will be used
    * for pointing to correct coarse stencil location.
    *--------------------------------------------------------------------------*/
   OffsetA =  nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  2, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < 2; i++)
   {
      OffsetA[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  refine_factors[i + 1], NALU_HYPRE_MEMORY_HOST);
   }

   /*--------------------------------------------------------------------------
    *  Stencil contribution cnts, weights, etc are computed only if we have
    *  a new stencil pattern. If the pattern is the same, the previously
    *  computed stencil contribution cnts, weights, etc can be used.
    *
    *  Mark the stencil_marker so that the first time the stencil is non-null,
    *  the stencil contribution cnts, weights, etc are computed.
    *--------------------------------------------------------------------------*/
   stencil_marker = trueV;
   for (var1 = 0; var1 < nvars; var1++)
   {
      cgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructPMatrixPGrid(A_crse), var1);
      cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);

      fgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructPMatrixPGrid(A_pmatrix), var1);
      fgrid_boxes = nalu_hypre_StructGridBoxes(fgrid);


      for (var2 = 0; var2 < nvars; var2++)
      {
         stencils = nalu_hypre_SStructPMatrixSStencil(A_crse, var1, var2);
         if (stencils != NULL)
         {
            stencil_size = nalu_hypre_StructStencilSize(stencils);

            /*-----------------------------------------------------------------
             * When stencil_marker== true, form the stencil contributions cnts,
             * weights, etc. This occurs for the first non-null stencil or
             * when the stencil shape of the current non-null stencil has a
             * different stencil shape from that of the latest non-null stencil.
             *
             * But when  stencil_marker== false, we must check to see if we
             * need new stencil contributions cnts, weights, etc. Thus, find
             * the latest non-null stencil for comparison.
             *-----------------------------------------------------------------*/
            if (stencil_marker == falseV)
            {
               /* search for the first previous non-null stencil */
               found     = falseV;
               var2_start = var2 - 1;
               for (j = var1; j >= 0; j--)
               {
                  for (i = var2_start; i >= 0; i--)
                  {
                     stencils_last = nalu_hypre_SStructPMatrixSStencil(A_crse, j, i);
                     if (stencils_last != NULL)
                     {
                        found = trueV;
                        break;
                     }
                  }
                  if (found)
                  {
                     break;
                  }
                  else
                  {
                     var2_start = nvars - 1;
                  }
               }

               /*--------------------------------------------------------------
                * Compare the stencil shape.
                *--------------------------------------------------------------*/
               stencil_last_size = nalu_hypre_StructStencilSize(stencils_last);
               if (stencil_last_size != stencil_size)
               {
                  stencil_marker = trueV;
                  break;
               }
               else
               {
                  found = falseV;
                  for (i = 0; i < stencil_size; i++)
                  {
                     nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils, i),
                                     stencil_shape_i);
                     nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils_last, i),
                                     stencil_last_shape_i);

                     nalu_hypre_SetIndex3(index_temp,
                                     stencil_shape_i[0] - stencil_last_shape_i[0],
                                     stencil_shape_i[1] - stencil_last_shape_i[1],
                                     stencil_shape_i[2] - stencil_last_shape_i[2]);

                     AbsStencilShape(index_temp, abs_stencil_shape);
                     if (abs_stencil_shape)
                     {
                        found = trueV;
                        stencil_marker = trueV;
                        nalu_hypre_TFree(stencil_contrib_cnt, NALU_HYPRE_MEMORY_HOST);
                        nalu_hypre_TFree(stencil_ranks, NALU_HYPRE_MEMORY_HOST);
                        for (i = 0; i < stencil_size; i++)
                        {
                           nalu_hypre_BoxDestroy(shift_box[i]);
                        }
                        nalu_hypre_TFree(shift_box, NALU_HYPRE_MEMORY_HOST);
                        nalu_hypre_TFree(volume_shift_box, NALU_HYPRE_MEMORY_HOST);
                        nalu_hypre_TFree(vals, NALU_HYPRE_MEMORY_HOST);

                        for (j = 1; j < max_stencil_size; j++)
                        {
                           stencil_i = rank_stencils[j];
                           if (stencil_i != -1)
                           {
                              nalu_hypre_TFree(stencil_contrib_i[stencil_i], NALU_HYPRE_MEMORY_HOST);
                              nalu_hypre_TFree(weight_contrib_i[stencil_i], NALU_HYPRE_MEMORY_HOST);
                           }
                        }
                        nalu_hypre_TFree(stencil_contrib_i, NALU_HYPRE_MEMORY_HOST);
                        nalu_hypre_TFree(weight_contrib_i, NALU_HYPRE_MEMORY_HOST);
                        nalu_hypre_TFree(rank_stencils, NALU_HYPRE_MEMORY_HOST);
                     }

                     if (found)
                     {
                        break;
                     }
                  }   /* for (i= 0; i< stencil_size; i++) */
               }      /* else */
            }         /* if (stencil_marker == false) */

            /*-----------------------------------------------------------------
             *  If stencil_marker==true, form the contribution structures.
             *  Since the type of averaging is determined by the stencil shapes,
             *  we need a ranking of the stencil shape to allow for easy
             *  determination.
             *
             *  top:  14  12  13    centre:  5  3  4     bottom 23   21   22
             *        11   9  10             2  0  1            20   18   19
             *        17  15  16             8  6  7            26   24   25
             *
             *  for stencil of max. size 27.
             *
             *  stencil_contrib_cnt[i]=  no. of fine stencils averaged to
             *                           form stencil entry i.
             *  stencil_contrib_i[i]  =  rank of fine stencils contributing
             *                           to form stencil entry i.
             *  weight_contrib_i[i]   =  array of weights for weighting
             *                           the contributions to stencil entry i.
             *  stencil_ranks[i]      =  rank of stencil entry i.
             *  rank_stencils[i]      =  stencil entry of rank i.
             *-----------------------------------------------------------------*/

            if (stencil_marker == trueV)
            {

               /* mark stencil_marker for the next stencil */
               stencil_marker = falseV;

               stencil_contrib_cnt = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  stencil_size, NALU_HYPRE_MEMORY_HOST);
               stencil_contrib_i  = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  stencil_size, NALU_HYPRE_MEMORY_HOST);
               weight_contrib_i   = nalu_hypre_TAlloc(NALU_HYPRE_Real *,  stencil_size, NALU_HYPRE_MEMORY_HOST);
               stencil_ranks      = nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_size, NALU_HYPRE_MEMORY_HOST);
               rank_stencils      = nalu_hypre_TAlloc(NALU_HYPRE_Int,  max_stencil_size, NALU_HYPRE_MEMORY_HOST);
               shift_box          = nalu_hypre_TAlloc(nalu_hypre_Box *,  stencil_size, NALU_HYPRE_MEMORY_HOST);
               volume_shift_box   = nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_size, NALU_HYPRE_MEMORY_HOST);

               for (i = 0; i < max_stencil_size; i++)
               {
                  rank_stencils[i] = -1;
                  if (i < stencil_size)
                  {
                     stencil_ranks[i] = -1;
                  }
               }

               /*-----------------------------------------------------------------
                *  Get mappings between stencil entries and ranks and vice versa;
                *  fine grid looping extents for averaging of the fine coefficients;
                *  and the number of fine grid values to be averaged.
                *  Note that the shift_boxes are constructed correctly for any
                *  dimensions. For j>=ndim,
                *  nalu_hypre_BoxIMin(shift_box[i])[j]=nalu_hypre_BoxIMax(shift_box[i])[j]= 0.
                *-----------------------------------------------------------------*/
               for (i = 0; i < stencil_size; i++)
               {
                  shift_box[i] = nalu_hypre_BoxCreate(ndim);
                  nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils, i),
                                  stencil_shape_i);
                  MapStencilRank(stencil_shape_i, j);
                  stencil_ranks[i] = j;
                  rank_stencils[stencil_ranks[i]] = i;

                  nalu_hypre_SetIndex3(nalu_hypre_BoxIMin(shift_box[i]),
                                  (refine_factors[0] - 1)*stencil_shape_i[0],
                                  (refine_factors[1] - 1)*stencil_shape_i[1],
                                  (refine_factors[2] - 1)*stencil_shape_i[2]);

                  nalu_hypre_AddIndexes(nalu_hypre_BoxIMin(shift_box[i]),
                                   nalu_hypre_BoxIMax(&coarse_cell_box), 3,
                                   nalu_hypre_BoxIMax(shift_box[i]));

                  nalu_hypre_IntersectBoxes(&coarse_cell_box, shift_box[i], shift_box[i]);

                  volume_shift_box[i] = nalu_hypre_BoxVolume(shift_box[i]);
               }

               /*-----------------------------------------------------------------
                *  Derive the contribution info.
                *  The above rank table is used to determine the direction indices.
                *  Weight construction procedure valid for any dimensions.
                *-----------------------------------------------------------------*/

               /* east */
               stencil_i = rank_stencils[1];
               if (stencil_i != -1)
               {
                  stencil_contrib_cnt[stencil_i]++;
                  for (i = 4; i <= 7; i += 3)
                  {
                     if (rank_stencils[i] != -1)       /* ne or se */
                     {
                        stencil_contrib_cnt[stencil_i]++;
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 1; i <= 7; i += 3)
                        {
                           if (rank_stencils[j * 9 + i] != -1) /* bottom or top planes */
                           {
                              stencil_contrib_cnt[stencil_i]++;
                           }
                        }
                     }
                  }
                  max_contribut_size = stencil_contrib_cnt[stencil_i];
               }

               /* fill up the east contribution stencil indices */
               if (stencil_i != -1)
               {
                  stencil_contrib_i[stencil_i] =
                     nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                  weight_contrib_i[stencil_i] =
                     nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                  sum = 0.0;
                  k = 0;

                  stencil_contrib_i[stencil_i][k] = stencil_i;
                  AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                   abs_stencil_shape );
                  weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                  sum += weights[abs_stencil_shape];

                  for (i = 4; i <= 7; i += 3)
                  {
                     if (rank_stencils[i] != -1)
                     {
                        stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                        AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[i]),
                                        abs_stencil_shape );
                        weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                        sum += weights[abs_stencil_shape];
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 1; i <= 7; i += 3)
                        {
                           if (rank_stencils[j * 9 + i] != -1)
                           {
                              stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + i];

                              AbsStencilShape(
                                 nalu_hypre_StructStencilElement(stencils, rank_stencils[j * 9 + i]),
                                 abs_stencil_shape );
                              weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                              sum += weights[abs_stencil_shape];
                           }
                        }
                     }
                  }

                  for (i = 0; i < k ; i++)
                  {
                     weight_contrib_i[stencil_i][i] /= sum;
                  }
               }


               /* west */
               stencil_i = rank_stencils[2];
               if (stencil_i != -1)
               {
                  stencil_contrib_cnt[stencil_i]++;
                  for (i = 5; i <= 8; i += 3)
                  {
                     if (rank_stencils[i] != -1)       /* nw or sw */
                     {
                        stencil_contrib_cnt[stencil_i]++;
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 2; i <= 8; i += 3)
                        {
                           if (rank_stencils[j * 9 + i] != -1) /* bottom or top planes */
                           {
                              stencil_contrib_cnt[stencil_i]++;
                           }
                        }
                     }
                  }
                  max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                  stencil_contrib_cnt[stencil_i] );
               }

               if (stencil_i != -1)
               {
                  stencil_contrib_i[stencil_i] =
                     nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                  weight_contrib_i[stencil_i] =
                     nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                  sum = 0.0;
                  k = 0;

                  stencil_contrib_i[stencil_i][k] = stencil_i;
                  AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                   abs_stencil_shape );
                  weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                  sum += weights[abs_stencil_shape];

                  for (i = 5; i <= 8; i += 3)
                  {
                     if (rank_stencils[i] != -1)
                     {
                        stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                        AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[i]),
                                        abs_stencil_shape );
                        weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                        sum += weights[abs_stencil_shape];
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 2; i <= 8; i += 3)
                        {
                           if (rank_stencils[j * 9 + i] != -1)
                           {
                              stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + i];

                              AbsStencilShape(
                                 nalu_hypre_StructStencilElement(stencils, rank_stencils[j * 9 + i]),
                                 abs_stencil_shape );
                              weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                              sum += weights[abs_stencil_shape];
                           }
                        }
                     }
                  }

                  for (i = 0; i < k ; i++)
                  {
                     weight_contrib_i[stencil_i][i] /= sum;
                  }
               }


               /* north */
               stencil_i = rank_stencils[3];
               if (stencil_i != -1)
               {
                  stencil_contrib_cnt[stencil_i]++;
                  for (i = 4; i <= 5; i++)
                  {
                     if (rank_stencils[i] != -1)       /* ne or nw */
                     {
                        stencil_contrib_cnt[stencil_i]++;
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 3; i <= 5; i++)
                        {
                           if (rank_stencils[j * 9 + i] != -1) /* bottom or top planes */
                           {
                              stencil_contrib_cnt[stencil_i]++;
                           }
                        }
                     }
                  }
                  max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                  stencil_contrib_cnt[stencil_i] );
               }

               if (stencil_i != -1)
               {
                  stencil_contrib_i[stencil_i] =
                     nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                  weight_contrib_i[stencil_i] =
                     nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                  sum = 0.0;
                  k = 0;

                  stencil_contrib_i[stencil_i][k] = stencil_i;
                  AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                   abs_stencil_shape );
                  weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                  sum += weights[abs_stencil_shape];

                  for (i = 4; i <= 5; i++)
                  {
                     if (rank_stencils[i] != -1)
                     {
                        stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                        AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[i]),
                                        abs_stencil_shape );
                        weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                        sum += weights[abs_stencil_shape];
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 3; i <= 5; i++)
                        {
                           if (rank_stencils[j * 9 + i] != -1)
                           {
                              stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + i];

                              AbsStencilShape(
                                 nalu_hypre_StructStencilElement(stencils, rank_stencils[j * 9 + i]),
                                 abs_stencil_shape );
                              weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                              sum += weights[abs_stencil_shape];
                           }
                        }
                     }
                  }

                  for (i = 0; i < k ; i++)
                  {
                     weight_contrib_i[stencil_i][i] /= sum;
                  }
               }

               /* south */
               stencil_i = rank_stencils[6];
               if (stencil_i != -1)
               {
                  stencil_contrib_cnt[stencil_i]++;
                  for (i = 7; i <= 8; i++)
                  {
                     if (rank_stencils[i] != -1)       /* ne or nw */
                     {
                        stencil_contrib_cnt[stencil_i]++;
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 6; i <= 8; i++)
                        {
                           if (rank_stencils[j * 9 + i] != -1) /* bottom or top planes */
                           {
                              stencil_contrib_cnt[stencil_i]++;
                           }
                        }
                     }
                  }
                  max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                  stencil_contrib_cnt[stencil_i] );
               }


               if (stencil_i != -1)
               {
                  stencil_contrib_i[stencil_i] =
                     nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                  weight_contrib_i[stencil_i] =
                     nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                  sum = 0.0;
                  k = 0;

                  stencil_contrib_i[stencil_i][k] = stencil_i;
                  AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                   abs_stencil_shape );
                  weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                  sum += weights[abs_stencil_shape];

                  for (i = 7; i <= 8; i++)
                  {
                     if (rank_stencils[i] != -1)
                     {
                        stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                        AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[i]),
                                        abs_stencil_shape );
                        weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                        sum += weights[abs_stencil_shape];
                     }
                  }

                  if (ndim > 2)
                  {
                     for (j = 1; j <= 2; j++)
                     {
                        for (i = 6; i <= 8; i++)
                        {
                           if (rank_stencils[j * 9 + i] != -1)
                           {
                              stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + i];

                              AbsStencilShape(
                                 nalu_hypre_StructStencilElement(stencils, rank_stencils[j * 9 + i]),
                                 abs_stencil_shape );
                              weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                              sum += weights[abs_stencil_shape];
                           }
                        }
                     }
                  }

                  for (i = 0; i < k ; i++)
                  {
                     weight_contrib_i[stencil_i][i] /= sum;
                  }
               }

               /*-----------------------------------------------------------------
                *  If only 2-d, extract the corner indices.
                *-----------------------------------------------------------------*/
               if (ndim == 2)
               {
                  /* corners: ne  & nw */
                  for (i = 4; i <= 5; i++)
                  {
                     stencil_i = rank_stencils[i];
                     if (stencil_i != -1)
                     {
                        stencil_contrib_cnt[stencil_i]++;
                        stencil_contrib_i[stencil_i] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
                        weight_contrib_i[stencil_i] =  nalu_hypre_TAlloc(NALU_HYPRE_Real,  1, NALU_HYPRE_MEMORY_HOST);
                        stencil_contrib_i[stencil_i][0] = stencil_i;
                        weight_contrib_i[stencil_i][0] = weights[0];
                     }
                  }

                  /* corners: se  & sw */
                  for (i = 7; i <= 8; i++)
                  {
                     stencil_i = rank_stencils[i];
                     if (stencil_i != -1)
                     {
                        stencil_contrib_cnt[stencil_i]++;
                        stencil_contrib_i[stencil_i] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
                        weight_contrib_i[stencil_i] =  nalu_hypre_TAlloc(NALU_HYPRE_Real,  1, NALU_HYPRE_MEMORY_HOST);
                        stencil_contrib_i[stencil_i][0] = stencil_i;
                        weight_contrib_i[stencil_i][0] = weights[0];
                     }
                  }
               }

               /*-----------------------------------------------------------------
                *  Additional directions for 3-dim case
                *-----------------------------------------------------------------*/
               if (ndim > 2)
               {
                  /* sides: top */
                  stencil_i = rank_stencils[9];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 1; i <= 8; i++)
                     {
                        if (rank_stencils[9 + i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 1; i <= 8; i++)
                     {
                        if (rank_stencils[9 + i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[9 + i];

                           AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[9 + i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* sides: bottom */
                  stencil_i = rank_stencils[18];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 1; i <= 8; i++)
                     {
                        if (rank_stencils[18 + i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 1; i <= 8; i++)
                     {
                        if (rank_stencils[18 + i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[18 + i];

                           AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[18 + i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: cne */
                  stencil_i = rank_stencils[4];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 4] != -1) /* bottom or top planes */
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 4] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + 4];

                           AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[j * 9 + 4]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: cse */
                  stencil_i = rank_stencils[7];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 7] != -1) /* bottom or top planes */
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 7] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + 7];

                           AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[j * 9 + 7]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: cnw */
                  stencil_i = rank_stencils[5];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 5] != -1) /* bottom or top planes */
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 5] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + 5];

                           AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[j * 9 + 5]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: csw */
                  stencil_i = rank_stencils[8];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 8] != -1) /* bottom or top planes */
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (j = 1; j <= 2; j++)
                     {
                        if (rank_stencils[j * 9 + 8] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[j * 9 + 8];

                           AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[j * 9 + 8]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: top east */
                  stencil_i = rank_stencils[10];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[10 + i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[10 + i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[10 + i];

                           AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[10 + i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: top west */
                  stencil_i = rank_stencils[11];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[11 + i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[11 + i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[11 + i];

                           AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[11 + i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: top north */
                  stencil_i = rank_stencils[12];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 13; i <= 14; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 13; i <= 14; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                           AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: top south*/
                  stencil_i = rank_stencils[15];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 16; i <= 17; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 16; i <= 17; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                           AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: bottom east */
                  stencil_i = rank_stencils[19];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[19 + i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[19 + i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[19 + i];

                           AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[19 + i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: bottom west */
                  stencil_i = rank_stencils[20];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[20 + i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 3; i <= 6; i += 3)
                     {
                        if (rank_stencils[20 + i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[20 + i];

                           AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[20 + i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: bottom north */
                  stencil_i = rank_stencils[21];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 22; i <= 23; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 22; i <= 23; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                           AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* edges: bottom south*/
                  stencil_i = rank_stencils[24];
                  if (stencil_i != -1)
                  {
                     stencil_contrib_cnt[stencil_i]++;
                     for (i = 25; i <= 26; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                        }
                     }
                     max_contribut_size = nalu_hypre_max( max_contribut_size,
                                                     stencil_contrib_cnt[stencil_i] );
                  }

                  if (stencil_i != -1)
                  {
                     stencil_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     weight_contrib_i[stencil_i] =
                        nalu_hypre_TAlloc(NALU_HYPRE_Real,  stencil_contrib_cnt[stencil_i], NALU_HYPRE_MEMORY_HOST);
                     sum = 0.0;
                     k = 0;

                     stencil_contrib_i[stencil_i][k] = stencil_i;
                     AbsStencilShape( nalu_hypre_StructStencilElement(stencils, stencil_i),
                                      abs_stencil_shape );
                     weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                     sum += weights[abs_stencil_shape];

                     for (i = 25; i <= 26; i++)
                     {
                        if (rank_stencils[i] != -1)
                        {
                           stencil_contrib_i[stencil_i][k] = rank_stencils[i];

                           AbsStencilShape(nalu_hypre_StructStencilElement(stencils, rank_stencils[i]),
                                           abs_stencil_shape );
                           weight_contrib_i[stencil_i][k++] = weights[abs_stencil_shape];
                           sum += weights[abs_stencil_shape];
                        }
                     }

                     for (i = 0; i < k ; i++)
                     {
                        weight_contrib_i[stencil_i][i] /= sum;
                     }
                  }

                  /* corners*/
                  for (j = 1; j <= 2; j++)
                  {
                     for (i = 4; i <= 5; i++)
                     {
                        stencil_i = rank_stencils[9 * j + i];
                        if (stencil_i != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                           stencil_contrib_i[stencil_i] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
                           weight_contrib_i[stencil_i] =  nalu_hypre_TAlloc(NALU_HYPRE_Real,  1, NALU_HYPRE_MEMORY_HOST);
                           stencil_contrib_i[stencil_i][0] = stencil_i;
                           weight_contrib_i[stencil_i][0] = weights[0];
                        }
                     }
                     for (i = 7; i <= 8; i++)
                     {
                        stencil_i = rank_stencils[9 * j + i];
                        if (stencil_i != -1)
                        {
                           stencil_contrib_cnt[stencil_i]++;
                           stencil_contrib_i[stencil_i] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
                           weight_contrib_i[stencil_i] =  nalu_hypre_TAlloc(NALU_HYPRE_Real,  1, NALU_HYPRE_MEMORY_HOST);
                           stencil_contrib_i[stencil_i][0] = stencil_i;
                           weight_contrib_i[stencil_i][0] = weights[0];
                        }
                     }
                  }

               }       /* if ndim > 2 */
               /*-----------------------------------------------------------------
                *  Allocate for the temporary vector used in computing the
                *  averages.
                *-----------------------------------------------------------------*/
               vals = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  max_contribut_size, NALU_HYPRE_MEMORY_HOST);

               /*-----------------------------------------------------------------
                *  coarse grid stencil contributor structures have been formed.
                *-----------------------------------------------------------------*/
            }   /* if (stencil_marker == true) */

            /*---------------------------------------------------------------------
             *  Loop over gridboxes to average stencils
             *---------------------------------------------------------------------*/
            smatrix_var = nalu_hypre_SStructPMatrixSMatrix(A_pmatrix, var1, var2);
            crse_smatrix = nalu_hypre_SStructPMatrixSMatrix(A_crse, var1, var2);

            /*---------------------------------------------------------------------
             *  data ptrs to extract and fill in data.
             *---------------------------------------------------------------------*/
            a_ptrs   = nalu_hypre_TAlloc(NALU_HYPRE_Real *,  stencil_size, NALU_HYPRE_MEMORY_HOST);
            crse_ptrs = nalu_hypre_TAlloc(NALU_HYPRE_Real *,  stencil_size, NALU_HYPRE_MEMORY_HOST);

            nalu_hypre_ForBoxI(ci, cgrid_boxes)
            {
               cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);
               fbox_interior_ci = fbox_interior[var1][ci];
               fbox_bdy_ci      = fbox_bdy[var1][ci];
               interior_fboxi_ci = interior_fboxi[var1][ci];
               bdy_fboxi_ci     = bdy_fboxi[var1][ci];

               crse_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(crse_smatrix),
                                             ci);
               /*------------------------------------------------------------------
                * grab the correct coarse grid pointers. These are the parent base
                * grids.
                *------------------------------------------------------------------*/
               for (i = 0; i < stencil_size; i++)
               {
                  nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils, i), stencil_shape_i);
                  crse_ptrs[i] = nalu_hypre_StructMatrixExtractPointerByIndex(crse_smatrix,
                                                                         ci,
                                                                         stencil_shape_i);
               }
               /*------------------------------------------------------------------
                *  Loop over the interior of each patch inside cgrid_box.
                *------------------------------------------------------------------*/
               nalu_hypre_ForBoxI(fi, fbox_interior_ci)
               {
                  fgrid_box = nalu_hypre_BoxArrayBox(fbox_interior_ci, fi);
                  /*--------------------------------------------------------------
                   * grab the fine grid ptrs & create the offsets for the fine
                   * grid ptrs.
                   *--------------------------------------------------------------*/
                  A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(smatrix_var),
                                             interior_fboxi_ci[fi]);
                  for (i = 0; i < stencil_size; i++)
                  {
                     nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils, i),
                                     stencil_shape_i);
                     a_ptrs[i] =
                        nalu_hypre_StructMatrixExtractPointerByIndex(smatrix_var,
                                                                interior_fboxi_ci[fi],
                                                                stencil_shape_i);
                  }

                  /*---------------------------------------------------------------
                   *  Compute the offsets for pointing to the correct data.
                   *  Note that for 1-d, OffsetA[j][i]= 0. Therefore, this ptr
                   *  will be correct for 1-d.
                   *---------------------------------------------------------------*/
                  for (j = 0; j < 2; j++)
                  {
                     OffsetA[j][0] = 0;
                     for (i = 1; i < refine_factors[j + 1]; i++)
                     {
                        if (j == 0)
                        {
                           nalu_hypre_SetIndex3(index_temp, 0, i, 0);
                        }
                        else
                        {
                           nalu_hypre_SetIndex3(index_temp, 0, 0, i);
                        }
                        OffsetA[j][i] = nalu_hypre_BoxOffsetDistance(A_dbox, index_temp);
                     }
                  }

                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(fgrid_box), fstart);
                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(fgrid_box), fend);

                  /* coarsen the interior patch box*/
                  nalu_hypre_ClearIndex(index_temp);
                  nalu_hypre_StructMapFineToCoarse(fstart, index_temp, stridef,
                                              nalu_hypre_BoxIMin(&fine_box));
                  nalu_hypre_StructMapFineToCoarse(fend, index_temp, stridef,
                                              nalu_hypre_BoxIMax(&fine_box));

                  nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&fine_box), cstart);

                  /*----------------------------------------------------------------
                   * Loop over interior grid box.
                   *----------------------------------------------------------------*/

                  nalu_hypre_BoxGetSize(&fine_box, loop_size);

                  nalu_hypre_SerialBoxLoop2Begin(ndim, loop_size,
                                            A_dbox, fstart, stridef, iA,
                                            crse_dbox, cstart, stridec, iAc);
                  {
                     for (i = 0; i < stencil_size; i++)
                     {
                        rank =  stencil_ranks[i];

                        /*------------------------------------------------------------
                         *  Loop over refinement agglomeration making up a coarse cell
                         *  when a non-centre stencil.
                         *------------------------------------------------------------*/
                        if (rank)
                        {
                           /*--------------------------------------------------------
                            *  Loop over refinement agglomeration extents making up a
                            *  a coarse cell.
                            *--------------------------------------------------------*/
                           nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(shift_box[i]), index1);
                           nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(shift_box[i]), index2);

                           for (m = 0; m < stencil_contrib_cnt[i]; m++)
                           {
                              vals[m] = 0.0;
                           }

                           /*--------------------------------------------------------
                            * For 1-d, index1[l]= index2[l]= 0, l>=1. So
                            *    iA_shift_zyx= j,
                            * which is correct. Similarly, 2-d is correct.
                            *--------------------------------------------------------*/
                           for (l = index1[2]; l <= index2[2]; l++)
                           {
                              iA_shift_z = iA + OffsetA[1][l];
                              for (k = index1[1]; k <= index2[1]; k++)
                              {
                                 iA_shift_zy = iA_shift_z + OffsetA[0][k];
                                 for (j = index1[0]; j <= index2[0]; j++)
                                 {
                                    iA_shift_zyx = iA_shift_zy + j;

                                    for (m = 0; m < stencil_contrib_cnt[i]; m++)
                                    {
                                       stencil_i = stencil_contrib_i[i][m];
                                       vals[m] += a_ptrs[stencil_i][iA_shift_zyx];
                                    }
                                 }
                              }
                           }
                           /*----------------------------------------------------------
                            *  average & weight the contributions and place into coarse
                            *  stencil entry.
                            *----------------------------------------------------------*/
                           crse_ptrs[i][iAc] = 0.0;
                           for (m = 0; m < stencil_contrib_cnt[i]; m++)
                           {
                              crse_ptrs[i][iAc] += vals[m] * weight_contrib_i[i][m];
                           }
                           crse_ptrs[i][iAc] /= volume_shift_box[i];

                        }  /* if (rank) */
                     }     /* for i */

                     /*------------------------------------------------------------------
                      *  centre stencil:
                      *  The centre stencil is computed so that the row sum is equal to
                      *  the sum of the row sums of the fine matrix. Uses the computed
                      *  coarse off-diagonal stencils.
                      *
                      *  No fine-coarse interface for the interior boxes.
                      *------------------------------------------------------------------*/
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&coarse_cell_box), index1);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(&coarse_cell_box), index2);

                     sum = 0.0;
                     for (l = index1[2]; l <= index2[2]; l++)
                     {
                        iA_shift_z = iA + OffsetA[1][l];
                        for (k = index1[1]; k <= index2[1]; k++)
                        {
                           iA_shift_zy = iA_shift_z + OffsetA[0][k];
                           for (j = index1[0]; j <= index2[0]; j++)
                           {
                              iA_shift_zyx = iA_shift_zy + j;
                              for (m = 0; m < stencil_size; m++)
                              {
                                 sum += a_ptrs[m][iA_shift_zyx];
                              }
                           }
                        }
                     }

                     /*---------------------------------------------------------------
                      * coarse centre coefficient- when away from the fine-coarse
                      * interface, the centre coefficient is the sum of the
                      * off-diagonal components.
                      *---------------------------------------------------------------*/
                     sum /= scaling;
                     for (m = 0; m < stencil_size; m++)
                     {
                        rank = stencil_ranks[m];
                        if (rank)
                        {
                           sum -= crse_ptrs[m][iAc];
                        }
                     }
                     crse_ptrs[ rank_stencils[0] ][iAc] = sum;
                  }
                  nalu_hypre_SerialBoxLoop2End(iA, iAc);
               }    /* end nalu_hypre_ForBoxI(fi, fbox_interior_ci) */

               /*------------------------------------------------------------------
                *  Loop over the boundaries of each patch inside cgrid_box.
                *------------------------------------------------------------------*/
               nalu_hypre_ForBoxArrayI(arrayi, fbox_bdy_ci)
               {
                  fbox_bdy_ci_fi = nalu_hypre_BoxArrayArrayBoxArray(fbox_bdy_ci, arrayi);
                  nalu_hypre_ForBoxI(fi, fbox_bdy_ci_fi)
                  {
                     fgrid_box = nalu_hypre_BoxArrayBox(fbox_bdy_ci_fi, fi);

                     /*-----------------------------------------------------------
                      * grab the fine grid ptrs & create the offsets for the fine
                      * grid ptrs.
                      *-----------------------------------------------------------*/
                     A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(smatrix_var),
                                                bdy_fboxi_ci[arrayi]);
                     for (i = 0; i < stencil_size; i++)
                     {
                        nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils, i),
                                        stencil_shape_i);
                        a_ptrs[i] =
                           nalu_hypre_StructMatrixExtractPointerByIndex(smatrix_var,
                                                                   bdy_fboxi_ci[arrayi],
                                                                   stencil_shape_i);
                     }

                     /*--------------------------------------------------------------
                      *  Compute the offsets for pointing to the correct data.
                      *--------------------------------------------------------------*/
                     for (j = 0; j < 2; j++)
                     {
                        OffsetA[j][0] = 0;
                        for (i = 1; i < refine_factors[j + 1]; i++)
                        {
                           if (j == 0)
                           {
                              nalu_hypre_SetIndex3(index_temp, 0, i, 0);
                           }
                           else
                           {
                              nalu_hypre_SetIndex3(index_temp, 0, 0, i);
                           }
                           OffsetA[j][i] = nalu_hypre_BoxOffsetDistance(A_dbox, index_temp);
                        }
                     }

                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(fgrid_box), fstart);
                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(fgrid_box), fend);

                     /* coarsen the patch box*/
                     nalu_hypre_ClearIndex(index_temp);
                     nalu_hypre_StructMapFineToCoarse(fstart, index_temp, stridef,
                                                 nalu_hypre_BoxIMin(&fine_box));
                     nalu_hypre_StructMapFineToCoarse(fend, index_temp, stridef,
                                                 nalu_hypre_BoxIMax(&fine_box));

                     nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&fine_box), cstart);

                     /*--------------------------------------------------------------
                      * Loop over boundary grid box.
                      *--------------------------------------------------------------*/

                     nalu_hypre_BoxGetSize(&fine_box, loop_size);

                     nalu_hypre_SerialBoxLoop2Begin(ndim, loop_size,
                                               A_dbox, fstart, stridef, iA,
                                               crse_dbox, cstart, stridec, iAc);
                     {
                        zypre_BoxLoopGetIndex(lindex);
                        for (i = 0; i < stencil_size; i++)
                        {
                           rank =  stencil_ranks[i];

                           /*--------------------------------------------------------
                            * Loop over refinement agglomeration making up a coarse
                            * cell when a non-centre stencil.
                            *--------------------------------------------------------*/
                           if (rank)
                           {
                              /*-----------------------------------------------------
                               * Loop over refinement agglomeration extents making up
                               * a coarse cell.
                               *-----------------------------------------------------*/
                              nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(shift_box[i]), index1);
                              nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(shift_box[i]), index2);

                              for (m = 0; m < stencil_contrib_cnt[i]; m++)
                              {
                                 vals[m] = 0.0;
                              }

                              for (l = index1[2]; l <= index2[2]; l++)
                              {
                                 iA_shift_z = iA + OffsetA[1][l];
                                 for (k = index1[1]; k <= index2[1]; k++)
                                 {
                                    iA_shift_zy = iA_shift_z + OffsetA[0][k];
                                    for (j = index1[0]; j <= index2[0]; j++)
                                    {
                                       iA_shift_zyx = iA_shift_zy + j;

                                       for (m = 0; m < stencil_contrib_cnt[i]; m++)
                                       {
                                          stencil_i = stencil_contrib_i[i][m];
                                          vals[m] += a_ptrs[stencil_i][iA_shift_zyx];
                                       }
                                    }
                                 }
                              }
                              /*---------------------------------------------------------
                               *  average & weight the contributions and place into coarse
                               *  stencil entry.
                               *---------------------------------------------------------*/
                              crse_ptrs[i][iAc] = 0.0;
                              for (m = 0; m < stencil_contrib_cnt[i]; m++)
                              {
                                 crse_ptrs[i][iAc] += vals[m] * weight_contrib_i[i][m];
                              }
                              crse_ptrs[i][iAc] /= volume_shift_box[i];

                           }  /* if (rank) */
                        }     /* for i */

                        /*---------------------------------------------------------------
                         *  centre stencil:
                         *  The centre stencil is computed so that the row sum is equal to
                         *  th sum of the row sums of the fine matrix. Uses the computed
                         *  coarse off-diagonal stencils.
                         *
                         *  Along the fine-coarse interface, we need to add the unstructured
                         *  connections.
                         *---------------------------------------------------------------*/
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&coarse_cell_box), index1);
                        nalu_hypre_CopyIndex(nalu_hypre_BoxIMax(&coarse_cell_box), index2);

                        temp3 = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  volume_coarse_cell_box, NALU_HYPRE_MEMORY_HOST);

                        /*---------------------------------------------------------------
                         *  iA_shift_zyx is computed correctly for 1 & 2-d. Also,
                         *  ll= 0 for 2-d, and ll= kk= 0 for 1-d. Correct ptrs.
                         *---------------------------------------------------------------*/
                        for (l = index1[2]; l <= index2[2]; l++)
                        {
                           iA_shift_z = iA + OffsetA[1][l];
                           ll        = l * refine_factors[1] * refine_factors[0];
                           for (k = index1[1]; k <= index2[1]; k++)
                           {
                              iA_shift_zy = iA_shift_z + OffsetA[0][k];
                              kk         = ll + k * refine_factors[0];
                              for (j = index1[0]; j <= index2[0]; j++)
                              {
                                 iA_shift_zyx = iA_shift_zy + j;
                                 jj          = kk + j;
                                 for (m = 0; m < stencil_size; m++)
                                 {
                                    temp3[jj] += a_ptrs[m][iA_shift_zyx];
                                 }
                              }
                           }
                        }

                        /*------------------------------------------------------------
                         * extract all unstructured connections. Note that we extract
                         * from sstruct_matrix A, which already has been assembled.
                         *------------------------------------------------------------*/
                        if (nUventries > 0)
                        {
                           temp2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  volume_coarse_cell_box, NALU_HYPRE_MEMORY_HOST);
                           cnt1 = 0;
                           for (l = index1[2]; l <= index2[2]; l++)
                           {
                              ll = l * refine_factors[1] * refine_factors[0];
                              for (k = index1[1]; k <= index2[1]; k++)
                              {
                                 kk = ll + k * refine_factors[0];
                                 for (j = index1[0]; j <= index2[0]; j++)
                                 {
                                    jj = kk + j;

                                    nalu_hypre_SetIndex3(index_temp,
                                                    j + lindex[0]*stridef[0],
                                                    k + lindex[1]*stridef[1],
                                                    l + lindex[2]*stridef[2]);
                                    nalu_hypre_AddIndexes(fstart, index_temp, 3, index_temp);

                                    nalu_hypre_SStructGridFindBoxManEntry(grid, part_fine, index_temp,
                                                                     var1, &boxman_entry);
                                    nalu_hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index_temp,
                                                                          &rank, matrix_type);

                                    found = falseV;
                                    i = nalu_hypre_SStructGraphIUVEntry(graph, 0);
                                    m = nalu_hypre_SStructGraphIUVEntry(graph, nUventries - 1);
                                    if ((rank - startrank) >= i && (rank - startrank) <= m)
                                    {
                                       found = trueV;
                                    }

                                    if (found)
                                    {
                                       Uventry = nalu_hypre_SStructGraphUVEntry(graph, rank - startrank);

                                       if (Uventry != NULL)
                                       {
                                          nUentries = nalu_hypre_SStructUVEntryNUEntries(Uventry);

                                          m = 0;
                                          for (i = 0; i < nUentries; i++)
                                          {
                                             if (nalu_hypre_SStructUVEntryToPart(Uventry, i) == part_crse)
                                             {
                                                m++;
                                             }
                                          }  /* for (i= 0; i< nUentries; i++) */

                                          temp2[jj] = m;
                                          cnt1    += m;

                                       }  /* if (Uventry != NULL) */
                                    }     /* if (found) */

                                 }   /* for (j= index1[0]; j<= index2[0]; j++) */
                              }      /* for (k= index1[1]; k<= index2[1]; k++) */
                           }         /* for (l= index1[2]; l<= index2[2]; l++) */

                           ncols = nalu_hypre_TAlloc(NALU_HYPRE_Int,  cnt1, NALU_HYPRE_MEMORY_HOST);
                           for (l = 0; l < cnt1; l++)
                           {
                              ncols[l] = 1;
                           }

                           rows = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  cnt1, NALU_HYPRE_MEMORY_HOST);
                           cols = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  cnt1, NALU_HYPRE_MEMORY_HOST);
                           vals2 = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  cnt1, NALU_HYPRE_MEMORY_HOST);

                           cnt1 = 0;
                           for (l = index1[2]; l <= index2[2]; l++)
                           {
                              ll = l * refine_factors[1] * refine_factors[0];
                              for (k = index1[1]; k <= index2[1]; k++)
                              {
                                 kk = ll + k * refine_factors[0];
                                 for (j = index1[0]; j <= index2[0]; j++)
                                 {
                                    jj = kk + j;

                                    nalu_hypre_SetIndex3(index_temp,
                                                    j + lindex[0]*stridef[0],
                                                    k + lindex[1]*stridef[1],
                                                    l + lindex[2]*stridef[2]);
                                    nalu_hypre_AddIndexes(fstart, index_temp, 3, index_temp);

                                    nalu_hypre_SStructGridFindBoxManEntry(grid, part_fine, index_temp,
                                                                     var1, &boxman_entry);
                                    nalu_hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index_temp,
                                                                          &rank, matrix_type);

                                    found = falseV;
                                    if (nUventries > 0)
                                    {
                                       i = nalu_hypre_SStructGraphIUVEntry(graph, 0);
                                       m = nalu_hypre_SStructGraphIUVEntry(graph, nUventries - 1);
                                       if ((NALU_HYPRE_Int)(rank - startrank) >= i && (NALU_HYPRE_Int)(rank - startrank) <= m)
                                       {
                                          found = trueV;
                                       }
                                    }

                                    if (found)
                                    {
                                       Uventry = nalu_hypre_SStructGraphUVEntry(graph, (NALU_HYPRE_Int)(rank - startrank));

                                       if (Uventry != NULL)
                                       {
                                          nUentries = nalu_hypre_SStructUVEntryNUEntries(Uventry);
                                          for (i = 0; i < nUentries; i++)
                                          {
                                             if (nalu_hypre_SStructUVEntryToPart(Uventry, i) == part_crse)
                                             {
                                                rows[cnt1] = rank;
                                                cols[cnt1++] = nalu_hypre_SStructUVEntryToRank(Uventry, i);
                                             }

                                          }  /* for (i= 0; i< nUentries; i++) */
                                       }     /* if (Uventry != NULL) */
                                    }        /* if (found) */

                                 }   /* for (j= index1[0]; j<= index2[0]; j++) */
                              }      /* for (k= index1[1]; k<= index2[1]; k++) */
                           }         /* for (l= index1[2]; l<= index2[2]; l++) */

                           NALU_HYPRE_IJMatrixGetValues(ij_A, cnt1, ncols, rows, cols, vals2);

                           cnt1 = 0;
                           for (l = index1[2]; l <= index2[2]; l++)
                           {
                              ll = l * refine_factors[1] * refine_factors[0];
                              for (k = index1[1]; k <= index2[1]; k++)
                              {
                                 kk = ll + k * refine_factors[0];
                                 for (j = index1[0]; j <= index2[0]; j++)
                                 {
                                    jj = kk + j;
                                    for (m = 0; m < temp2[jj]; m++)
                                    {
                                       temp3[jj] += vals2[cnt1];
                                       cnt1++;
                                    }
                                    temp2[jj] = 0; /* zero off for next time */
                                 }       /* for (j= index1[0]; j<= index2[0]; j++) */
                              }           /* for (k= index1[1]; k<= index2[1]; k++) */
                           }              /* for (l= index1[2]; l<= index2[2]; l++) */

                           nalu_hypre_TFree(ncols, NALU_HYPRE_MEMORY_HOST);
                           nalu_hypre_TFree(rows, NALU_HYPRE_MEMORY_HOST);
                           nalu_hypre_TFree(cols, NALU_HYPRE_MEMORY_HOST);
                           nalu_hypre_TFree(vals2, NALU_HYPRE_MEMORY_HOST);
                           nalu_hypre_TFree(temp2, NALU_HYPRE_MEMORY_HOST);

                        }   /* if Uventries > 0 */

                        sum = 0.0;
                        for (l = index1[2]; l <= index2[2]; l++)
                        {
                           ll = l * refine_factors[1] * refine_factors[0];
                           for (k = index1[1]; k <= index2[1]; k++)
                           {
                              kk = ll + k * refine_factors[0];
                              for (j = index1[0]; j <= index2[0]; j++)
                              {
                                 jj = kk + j;
                                 sum += temp3[jj];
                              }
                           }
                        }

                        sum /= scaling;
                        crse_ptrs[ rank_stencils[0] ][iAc] = sum;

                        nalu_hypre_TFree(temp3, NALU_HYPRE_MEMORY_HOST);

                     }
                     nalu_hypre_SerialBoxLoop2End(iA, iAc);

                  }  /* nalu_hypre_ForBoxI(fi, fbox_bdy_ci_fi) */
               }      /* nalu_hypre_ForBoxArrayI(arrayi, fbox_bdy_ci) */
            }          /* nalu_hypre_ForBoxI(ci, cgrid_boxes) */

            nalu_hypre_TFree(a_ptrs, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(crse_ptrs, NALU_HYPRE_MEMORY_HOST);

         }    /* if (stencils != NULL) */
      }       /* end var2 */
   }          /* end var1 */

   if (stencil_contrib_cnt)
   {
      nalu_hypre_TFree(stencil_contrib_cnt, NALU_HYPRE_MEMORY_HOST);
   }
   if (stencil_ranks)
   {
      nalu_hypre_TFree(stencil_ranks, NALU_HYPRE_MEMORY_HOST);
   }
   if (volume_shift_box)
   {
      nalu_hypre_TFree(volume_shift_box, NALU_HYPRE_MEMORY_HOST);
   }
   if (vals)
   {
      nalu_hypre_TFree(vals, NALU_HYPRE_MEMORY_HOST);
   }

   if (shift_box)
   {
      for (j = 0; j < stencil_size; j++)
      {
         if (shift_box[j])
         {
            nalu_hypre_BoxDestroy(shift_box[j]);
         }
      }
      nalu_hypre_TFree(shift_box, NALU_HYPRE_MEMORY_HOST);
   }

   if (stencil_contrib_i)
   {
      for (j = 1; j < max_stencil_size; j++)
      {
         stencil_i = rank_stencils[j];
         if (stencil_i != -1)
         {
            if (stencil_contrib_i[stencil_i])
            {
               nalu_hypre_TFree(stencil_contrib_i[stencil_i], NALU_HYPRE_MEMORY_HOST);
            }
         }
      }
      nalu_hypre_TFree(stencil_contrib_i, NALU_HYPRE_MEMORY_HOST);
   }

   if (weight_contrib_i)
   {
      for (j = 1; j < max_stencil_size; j++)
      {
         stencil_i = rank_stencils[j];
         if (stencil_i != -1)
         {
            if (weight_contrib_i[stencil_i])
            {
               nalu_hypre_TFree(weight_contrib_i[stencil_i], NALU_HYPRE_MEMORY_HOST);
            }
         }
      }
      nalu_hypre_TFree(weight_contrib_i, NALU_HYPRE_MEMORY_HOST);
   }

   if (rank_stencils)
   {
      nalu_hypre_TFree(rank_stencils, NALU_HYPRE_MEMORY_HOST);
   }

   if (OffsetA)
   {
      for (j = 0; j < 2; j++)
      {
         if (OffsetA[j])
         {
            nalu_hypre_TFree(OffsetA[j], NALU_HYPRE_MEMORY_HOST);
         }
      }
      nalu_hypre_TFree(OffsetA, NALU_HYPRE_MEMORY_HOST);
   }

   /*--------------------------------------------------------------------------
    *  STEP 2:
    *
    *  Interface coarsening: fine-to-coarse connections. We are
    *  assuming that only like-variables couple along interfaces.
    *
    *  The task is to coarsen all the fine-to-coarse unstructured
    *  connections and to compute coarse coefficients along the
    *  interfaces (coarse-to-fine coefficients are obtained from these
    *  computed values assuming symmetry). This involves
    *      1) scanning over the graph entries to find the locations of
    *         the unstructure connections;
    *      2) determining the stencil shape of the coarsened connections;
    *      3) averaging the unstructured coefficients to compute
    *         coefficient entries for the interface stencils;
    *      4) determining the weights of the interface stencil coefficients
    *         to construct the structured coarse grid matrix along the
    *         interfaces.
    *
    *  We perform this task by
    *      1) scanning over the graph entries to group the locations
    *         of the fine-to-coarse connections wrt the boxes of the
    *         fine grid. Temporary vectors storing the Uventries indices
    *         and the number of connections for each box will be created;
    *      2) for each fine grid box, group the fine-to-coarse connections
    *         with respect to the connected coarse nodes. Temporary vectors
    *         storing the Uventry indices and the Uentry indices for each
    *         coarse node will be created (i.e., for a fixed coarse grid node,
    *         record the fine node Uventries indices that connect to this
    *         coarse node and Uentry index of the Uventry that contains
    *         this coarse node.). The grouping is accomplished comparing the
    *         ranks of the coarse nodes;
    *      3) using the Uventries and Uentry indices for each coarse node,
    *         "coarsen" the fine grid connections to this coarse node to
    *         create interface stencils (wrt to the coarse nodes- i.e.,
    *         the centre of the stencil is at a coarse node). Also, find
    *         the IJ rows and columns corresponding to all the fine-to-coarse
    *         connections in a box, and extract the  unstructured coefficients;
    *      4) looping over all coarse grid nodes connected to a fixed fine box,
    *         compute the arithmetically averaged interface stencils;
    *      5) compare the underlying coarse grid structured stencil shape
    *         to the interface stencil shape to determine how to weight the
    *         averaged interface stencil coefficients.
    *
    *  EXCEPTION: A NODE CAN CONTAIN ONLY UNSTRUCTURED CONNECTIONS
    *  BETWEEN ONLY TWO AMR LEVELS- I.E., WE CANNOT HAVE A NODE THAT
    *  IS ON THE INTERFACE OF MORE THAN TWO AMR LEVELS. CHANGES TO
    *  HANDLE THIS LATTER CASE WILL INVOLVE THE SEARCH FOR f/c
    *  CONNECTIONS.
    *-----------------------------------------------------------------*/
   if (nUventries > 0)
   {
      nvars    =  nalu_hypre_SStructPMatrixNVars(A_pmatrix);

      for (var1 = 0; var1 < nvars; var1++)
      {
         /*-----------------------------------------------------------------
          *  Yank out the structured stencils for this variable (only like
          *  variables considered) and find their ranks.
          *-----------------------------------------------------------------*/
         stencils    = nalu_hypre_SStructPMatrixSStencil(A_crse, var1, var1);
         stencil_size = nalu_hypre_StructStencilSize(stencils);

         stencil_ranks = nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_size, NALU_HYPRE_MEMORY_HOST);
         rank_stencils = nalu_hypre_TAlloc(NALU_HYPRE_Int,  max_stencil_size, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < stencil_size; i++)
         {
            nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils, i),
                            stencil_shape_i);
            MapStencilRank( stencil_shape_i, stencil_ranks[i] );
            rank_stencils[ stencil_ranks[i] ] = i;
         }
         /*-----------------------------------------------------------------
          *  qsort the ranks into ascending order
          *-----------------------------------------------------------------*/
         nalu_hypre_qsort0(stencil_ranks, 0, stencil_size - 1);

         crse_smatrix = nalu_hypre_SStructPMatrixSMatrix(A_crse, var1, var1);
         cgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructPMatrixPGrid(A_crse), var1);
         cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);

         fgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructPMatrixPGrid(A_pmatrix), var1);
         fgrid_boxes = nalu_hypre_StructGridBoxes(fgrid);

         box_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(fgrid_boxes), NALU_HYPRE_MEMORY_HOST);
         box_ends  = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_BoxArraySize(fgrid_boxes), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_SStructGraphFindSGridEndpts(graph, part_fine, var1, myid,
                                           0, box_starts);
         nalu_hypre_SStructGraphFindSGridEndpts(graph, part_fine, var1, myid,
                                           1, box_ends);

         /*-----------------------------------------------------------------
          *  Step 1: scanning over the graph entries to group the locations
          *          of the unstructured connections wrt to fine grid boxes.
          *
          *  Count the components that couple for each box.
          *
          *  box_graph_indices[fi]=   array of Uventries indices in box fi.
          *  box_graph_cnts[fi]   =   number of Uventries in box fi.
          *  cdata_space_rank[ci] =   begin offset rank of coarse data_space
          *                           box ci.
          *-----------------------------------------------------------------*/
         box_array_size   = nalu_hypre_BoxArraySize(fgrid_boxes);
         cbox_array_size  = nalu_hypre_BoxArraySize(cgrid_boxes);
         box_graph_indices = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  box_array_size, NALU_HYPRE_MEMORY_HOST);
         box_graph_cnts   = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  box_array_size, NALU_HYPRE_MEMORY_HOST);

         data_space = nalu_hypre_StructMatrixDataSpace(crse_smatrix);
         cdata_space_ranks = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cbox_array_size, NALU_HYPRE_MEMORY_HOST);
         cdata_space_ranks[0] = 0;
         for (i = 1; i < cbox_array_size; i++)
         {
            cdata_space_ranks[i] = cdata_space_ranks[i - 1] +
                                   nalu_hypre_BoxVolume(nalu_hypre_BoxArrayBox(data_space, i - 1));
         }

         /*-----------------------------------------------------------------
          *  Scanning obtained by searching iUventries between the start
          *  and end of a fine box. Binary search used to find the interval
          *  between these two endpts. Index (-1) returned if no interval
          *  bounds found. Note that if start has positive index, then end
          *  must have a positive index also.
          *-----------------------------------------------------------------*/
         for (fi = 0; fi < box_array_size; fi++)
         {
            i = nalu_hypre_LowerBinarySearch(iUventries, box_starts[fi], nUventries);
            if (i >= 0)
            {
               j = nalu_hypre_UpperBinarySearch(iUventries, box_ends[fi], nUventries);
               box_graph_indices[fi] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  j - i + 1, NALU_HYPRE_MEMORY_HOST);

               for (k = 0; k < (j - i + 1); k++)
               {
                  Uventry = nalu_hypre_SStructGraphUVEntry(graph,
                                                      iUventries[i + k]);

                  for (m = 0; m < nalu_hypre_SStructUVEntryNUEntries(Uventry); m++)
                  {
                     if (nalu_hypre_SStructUVEntryToPart(Uventry, m) == part_crse)
                     {
                        box_graph_indices[fi][box_graph_cnts[fi]] = iUventries[i + k];
                        box_graph_cnts[fi]++;
                        break;
                     }
                  }  /* for (m= 0; m< nalu_hypre_SStructUVEntryNUEntries(Uventry); m++) */
               }     /* for (k= 0; k< (j-i+1); k++) */
            }        /* if (i >= 0) */
         }           /* for (fi= 0; fi< box_array_size; fi++) */

         /*-----------------------------------------------------------------
          *  Step 2:
          *  Determine and group the fine-to-coarse connections in a box.
          *  Grouped according to the coarsened fine grid interface nodes.
          *
          *  box_ranks              = ranks of coarsened fine grid interface
          *                           nodes.
          *  box_connections        = counter for the distinct coarsened fine
          *                           grid interface nodes. This can be
          *                           used to group all the Uventries of a
          *                           coarsened fine grid node.
          *  cindex[l]              = the nalu_hypre_Index of coarsen node l.
          *  parents_cnodes[l]      = parent box that contains the coarsened
          *                           fine grid interface node l.
          *  fine_interface_ranks[l]= rank of coarsened fine grid interface
          *                           node l.
          *  box_ranks_cnt[l]       = counter for no. of Uventries for
          *                           coarsened node l.
          *  coarse_contrib_Uv[l]   = Uventry indices for Uventries that
          *                           contain fine-to-coarse connections of
          *                           coarse node l.
          *-----------------------------------------------------------------*/
         for (fi = 0; fi < box_array_size; fi++)
         {
            /*-------------------------------------------------------------
             * Determine the coarse data ptrs corresponding to fine box fi.
             * These are needed in assigning the averaged unstructured
             * coefficients.
             *
             * Determine how many distinct coarse grid nodes are in the
             * unstructured connection for a given box. Each node has a
             * structures.
             *
             * temp1 & temp2 are linked lists vectors used for grouping the
             * Uventries for a given coarse node.
             *-------------------------------------------------------------*/
            box_ranks       = nalu_hypre_TAlloc(NALU_HYPRE_Int,  box_graph_cnts[fi], NALU_HYPRE_MEMORY_HOST);
            box_connections = nalu_hypre_TAlloc(NALU_HYPRE_Int,  box_graph_cnts[fi], NALU_HYPRE_MEMORY_HOST);
            parents         = nalu_hypre_TAlloc(NALU_HYPRE_Int,  box_graph_cnts[fi], NALU_HYPRE_MEMORY_HOST);
            temp1           = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  box_graph_cnts[fi] + 1, NALU_HYPRE_MEMORY_HOST);
            temp2           = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  box_graph_cnts[fi], NALU_HYPRE_MEMORY_HOST);
            Uv_cindex       = nalu_hypre_TAlloc(nalu_hypre_Index,  box_graph_cnts[fi], NALU_HYPRE_MEMORY_HOST);

            /*-------------------------------------------------------------
             * determine the parent box of this fgrid_box.
             *-------------------------------------------------------------*/
            nalu_hypre_ClearIndex(index_temp);
            for (i = 0; i < box_graph_cnts[fi]; i++)
            {
               Uventry = Uventries[box_graph_indices[fi][i]];

               /*-------------------------------------------------------------
                * Coarsen the fine grid interface nodes and then get their
                * ranks. The correct coarse grid is needed to determine the
                * correct data_box.
                * Save the rank of the coarsened index & the parent box id.
                *-------------------------------------------------------------*/
               nalu_hypre_CopyIndex(nalu_hypre_SStructUVEntryIndex(Uventry), index);
               nalu_hypre_StructMapFineToCoarse(index, index_temp, stridef, Uv_cindex[i]);
               nalu_hypre_BoxSetExtents(&fine_box, Uv_cindex[i], Uv_cindex[i]);

               for (j = 0; j < cboxi_fcnt[var1][fi]; j++)
               {
                  ci = cboxi_fboxes[var1][fi][j];
                  cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);
                  nalu_hypre_IntersectBoxes(&fine_box, cgrid_box, &intersect_box);
                  if (nalu_hypre_BoxVolume(&intersect_box) > 0)
                  {
                     break;
                  }
               }

               parents[i]  = ci;
               box_ranks[i] = cdata_space_ranks[ci] +
                              nalu_hypre_BoxIndexRank(nalu_hypre_BoxArrayBox(data_space, ci),
                                                 Uv_cindex[i]);
            }

            /*---------------------------------------------------------------
             * Determine and "group" the Uventries using the box_ranks.
             * temp2 stores the Uventries indices for a coarsen node.
             *---------------------------------------------------------------*/
            cnt1 = 0;
            j   = 0;
            temp1[cnt1] = j;

            for (i = 0; i < box_graph_cnts[fi]; i++)
            {
               if (box_ranks[i] != -1)
               {
                  k                 = box_ranks[i];
                  box_connections[i] = cnt1;
                  temp2[j++]        = box_graph_indices[fi][i];

                  for (l = i + 1; l < box_graph_cnts[fi]; l++)
                  {
                     if (box_ranks[l] == k)
                     {
                        box_connections[l] = cnt1;
                        temp2[j++]        = box_graph_indices[fi][l];
                        box_ranks[l]      = -1;
                     }
                  }
                  cnt1++;
                  temp1[cnt1] = j;
               }
            }

            /*-----------------------------------------------------------------
             *  Store the graph entry info and other index info for each coarse
             *  grid node.
             *-----------------------------------------------------------------*/
            parents_cnodes      = nalu_hypre_TAlloc(NALU_HYPRE_Int,  cnt1, NALU_HYPRE_MEMORY_HOST);
            fine_interface_ranks = nalu_hypre_TAlloc(NALU_HYPRE_Int,  cnt1, NALU_HYPRE_MEMORY_HOST);
            box_ranks_cnt       = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt1, NALU_HYPRE_MEMORY_HOST);
            coarse_contrib_Uv   = nalu_hypre_TAlloc(NALU_HYPRE_Int *,  cnt1, NALU_HYPRE_MEMORY_HOST);
            cindex              = nalu_hypre_TAlloc(nalu_hypre_Index,  cnt1, NALU_HYPRE_MEMORY_HOST);

            for (i = 0; i < box_graph_cnts[fi]; i++)
            {
               if (box_ranks[i] != -1)
               {
                  j                      = box_connections[i];
                  parents_cnodes[j]      = parents[i];
                  fine_interface_ranks[j] =
                     nalu_hypre_BoxIndexRank(nalu_hypre_BoxArrayBox(data_space, parents[i]),
                                        Uv_cindex[i]);
                  nalu_hypre_CopyIndex(Uv_cindex[i], cindex[j]);

                  box_ranks_cnt[j]       = temp1[j + 1] - temp1[j];
                  coarse_contrib_Uv[j]   = nalu_hypre_TAlloc(NALU_HYPRE_Int,  box_ranks_cnt[j], NALU_HYPRE_MEMORY_HOST);

                  l                      = temp1[j];
                  for (k = 0; k < box_ranks_cnt[j]; k++)
                  {
                     coarse_contrib_Uv[j][k] = temp2[l + k];
                  }
               }
            }

            if (box_ranks)
            {
               nalu_hypre_TFree(box_ranks, NALU_HYPRE_MEMORY_HOST);
            }
            if (box_connections)
            {
               nalu_hypre_TFree(box_connections, NALU_HYPRE_MEMORY_HOST);
            }
            if (parents)
            {
               nalu_hypre_TFree(parents, NALU_HYPRE_MEMORY_HOST);
            }
            if (temp1)
            {
               nalu_hypre_TFree(temp1, NALU_HYPRE_MEMORY_HOST);
            }
            if (temp2)
            {
               nalu_hypre_TFree(temp2, NALU_HYPRE_MEMORY_HOST);
            }
            if (Uv_cindex)
            {
               nalu_hypre_TFree(Uv_cindex, NALU_HYPRE_MEMORY_HOST);
            }

            /*------------------------------------------------------------------------
             *  Step 3:
             *  Create the interface stencils.
             *
             *   interface_max_stencil_ranks[i] =  stencil_shape rank for each coarse
             *                                     Uentry connection of coarsened node
             *                                     i (i.e., the stencil_shape ranks of
             *                                     the interface stencils at node i).
             *   interface_max_stencil_cnt[i][m]=  counter for number of Uentries
             *                                     that describes a connection which
             *                                     coarsens into stencil_shape rank m.
             *   coarse_stencil_cnts[i]         =  counter for the no. of distinct
             *                                     interface stencil_shapes (i.e., the
             *                                     no. entries of the interface stencil).
             *   interface_stencil_ranks[i][l]  =  stencil_shape rank for interface
             *                                     stencil entry l, for coarse node i.
             *   interface_rank_stencils[i][j]  =  interface stencil entry for
             *                                     stencil_shape rank j, for node i.
             *------------------------------------------------------------------------*/

            /*-----------------------------------------------------------------
             *  Extract rows & cols info for extracting data from IJ matrix.
             *  Extract for all connections for a box.
             *-----------------------------------------------------------------*/
            nalu_hypre_ClearIndex(index_temp);

            nrows = 0;
            box_to_ranks_cnt =  nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt1, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < cnt1; i++)
            {
               for (j = 0; j < box_ranks_cnt[i]; j++)
               {
                  Uventry  = Uventries[ coarse_contrib_Uv[i][j] ];
                  for (k = 0; k < nalu_hypre_SStructUVEntryNUEntries(Uventry); k++)
                  {
                     if (nalu_hypre_SStructUVEntryToPart(Uventry, k) == part_crse)
                     {
                        box_to_ranks_cnt[i]++;
                     }
                  }
               }
               nrows += box_to_ranks_cnt[i];
            }

            ncols = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nrows, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < nrows; i++)
            {
               ncols[i] = 1;
            }

            rows =  nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  nrows, NALU_HYPRE_MEMORY_HOST);
            cols =  nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  nrows, NALU_HYPRE_MEMORY_HOST);
            vals =  nalu_hypre_CTAlloc(NALU_HYPRE_Real,  nrows, NALU_HYPRE_MEMORY_HOST);

            interface_max_stencil_ranks =  nalu_hypre_TAlloc(NALU_HYPRE_Int *,  cnt1, NALU_HYPRE_MEMORY_HOST);
            interface_max_stencil_cnt  =  nalu_hypre_TAlloc(NALU_HYPRE_Int *,  cnt1, NALU_HYPRE_MEMORY_HOST);
            interface_rank_stencils    =  nalu_hypre_TAlloc(NALU_HYPRE_Int *,  cnt1, NALU_HYPRE_MEMORY_HOST);
            interface_stencil_ranks    =  nalu_hypre_TAlloc(NALU_HYPRE_Int *,  cnt1, NALU_HYPRE_MEMORY_HOST);
            coarse_stencil_cnt         =  nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt1, NALU_HYPRE_MEMORY_HOST);

            k = 0;
            for (i = 0; i < cnt1; i++)
            {
               /*-----------------------------------------------------------------
                * for each coarse interface node, we get a stencil. We compute only
                * the ranks assuming a maximum size stencil of 27.
                *-----------------------------------------------------------------*/
               interface_max_stencil_ranks[i] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  box_to_ranks_cnt[i], NALU_HYPRE_MEMORY_HOST);
               interface_max_stencil_cnt[i]  = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_stencil_size, NALU_HYPRE_MEMORY_HOST);

               /*-----------------------------------------------------------------
                * conjugate the coarse node index for determining the stencil
                * shapes for the Uentry connections.
                *-----------------------------------------------------------------*/
               nalu_hypre_CopyIndex(cindex[i], index1);
               nalu_hypre_SetIndex3(index1, -index1[0], -index1[1], -index1[2]);

               n = 0;
               for (j = 0; j < box_ranks_cnt[i]; j++)
               {
                  /*--------------------------------------------------------------
                   * extract the row rank for a given Uventry. Note that these
                   * are the ranks in the grid of A. Therefore, we grab the index
                   * from the nested_graph Uventry to determine the global rank.
                   * With the rank, find the corresponding Uventry of the graph
                   * of A. The to_ranks now can be extracted out.
                   *--------------------------------------------------------------*/
                  Uventry = Uventries[ coarse_contrib_Uv[i][j] ];
                  nalu_hypre_CopyIndex(nalu_hypre_SStructUVEntryIndex(Uventry), index);

                  nalu_hypre_SStructGridFindBoxManEntry(grid, part_fine, index, var1, &boxman_entry);
                  nalu_hypre_SStructBoxManEntryGetGlobalRank(boxman_entry, index, &rank, matrix_type);

                  Uventry = nalu_hypre_SStructGraphUVEntry(graph, rank - startrank);
                  nUentries = nalu_hypre_SStructUVEntryNUEntries(Uventry);

                  for (l = 0; l < nUentries; l++)
                  {
                     if (nalu_hypre_SStructUVEntryToPart(Uventry, l) == part_crse)
                     {
                        to_rank  = nalu_hypre_SStructUVEntryToRank(Uventry, l);
                        rows[k]  = rank;
                        cols[k++] = to_rank;

                        /*---------------------------------------------------------
                         * compute stencil shape for this Uentry.
                         *---------------------------------------------------------*/
                        nalu_hypre_CopyIndex( nalu_hypre_SStructUVEntryToIndex(Uventry, l),
                                         index );
                        nalu_hypre_AddIndexes(index, index1, 3, index2);

                        MapStencilRank(index2, m);
                        interface_max_stencil_ranks[i][n++] = m;
                        interface_max_stencil_cnt[i][m]++;
                     }
                  }
               }
               nalu_hypre_TFree(coarse_contrib_Uv[i], NALU_HYPRE_MEMORY_HOST);

               /*-----------------------------------------------------------------
                * Determine only the distinct stencil ranks for coarse node i.
                *-----------------------------------------------------------------*/
               l = 0;
               for (j = 0; j < max_stencil_size; j++)
               {
                  if (interface_max_stencil_cnt[i][j])
                  {
                     l++;
                  }
               }

               coarse_stencil_cnt[i] = l;
               interface_stencil_ranks[i] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  l, NALU_HYPRE_MEMORY_HOST);
               interface_rank_stencils[i] = nalu_hypre_TAlloc(NALU_HYPRE_Int,  max_stencil_size, NALU_HYPRE_MEMORY_HOST);

               /*-----------------------------------------------------------------
                * For each stencil rank, assign one of the stencil_shape_i index.
                *-----------------------------------------------------------------*/
               l = 0;
               for (j = 0; j < max_stencil_size; j++)
               {
                  if (interface_max_stencil_cnt[i][j])
                  {
                     interface_rank_stencils[i][j] = l;
                     interface_stencil_ranks[i][l] = j;
                     l++;
                  }
               }
            }   /* for (i= 0; i< cnt1; i++) */

            nalu_hypre_TFree(coarse_contrib_Uv, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(box_ranks_cnt, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(cindex, NALU_HYPRE_MEMORY_HOST);

            /*-----------------------------------------------------------------
             * Extract data from IJ matrix
             *-----------------------------------------------------------------*/
            NALU_HYPRE_IJMatrixGetValues(ij_A, nrows, ncols, rows, cols, vals);

            nalu_hypre_TFree(ncols, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(rows, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(cols, NALU_HYPRE_MEMORY_HOST);

            /*-----------------------------------------------------------------
             *  Steps 4 & 5:
             *  Compute the arithmetically averaged interface stencils,
             *  and determine the interface stencil weights.
             *
             *    stencil_vals[l]       = averaged stencil coeff for interface
             *                            stencil entry l.
             *    common_rank_stencils  = final structured coarse stencil entries
             *                            for the stencil_shapes that the
             *                            interface stencils must collapse to.
             *    common_stencil_ranks  = final structured coarse stencil_shape
             *                            ranks for the stencil_shapes that the
             *                            interface stencils must collapse to.
             *    common_stencil_i      = stencil entry of the interface stencil
             *                            corresponding to the common
             *                            stencil_shape.
             *-----------------------------------------------------------------*/
            k = 0;
            for (i = 0; i < cnt1; i++)
            {
               stencil_vals = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  coarse_stencil_cnt[i], NALU_HYPRE_MEMORY_HOST);

               /*-----------------------------------------------------------------
                * Compute the arithmetic stencil averages for coarse node i.
                *-----------------------------------------------------------------*/
               for (j = 0; j < box_to_ranks_cnt[i]; j++)
               {
                  m = interface_max_stencil_ranks[i][j];
                  l = interface_rank_stencils[i][m];
                  stencil_vals[l] += vals[k] / interface_max_stencil_cnt[i][m];
                  k++;
               }
               nalu_hypre_TFree(interface_max_stencil_ranks[i], NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(interface_max_stencil_cnt[i], NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(interface_rank_stencils[i], NALU_HYPRE_MEMORY_HOST);

               /*-----------------------------------------------------------------
                * Determine which stencil has to be formed. This is accomplished
                * by comparing the coarse grid stencil ranks with the computed
                * interface stencil ranks. We qsort (if there are more than one
                * rank) the ranks to give quick comparisons. Note that we need
                * to swap the elements of stencil_vals & fine_interface_ranks[i]'s
                * accordingly.
                *-----------------------------------------------------------------*/

               sort = falseV;
               for (j = 0; j < (coarse_stencil_cnt[i] - 1); j++)
               {
                  if (interface_stencil_ranks[i][j] > interface_stencil_ranks[i][j + 1])
                  {
                     sort = trueV;
                     break;
                  }
               }

               if ( (coarse_stencil_cnt[i] > 1) && (sort == trueV) )
               {
                  temp1 = nalu_hypre_TAlloc(NALU_HYPRE_Int,  coarse_stencil_cnt[i], NALU_HYPRE_MEMORY_HOST);
                  for (j = 0; j < coarse_stencil_cnt[i]; j++)
                  {
                     temp1[j] = j;
                  }

                  nalu_hypre_qsort1(interface_stencil_ranks[i], (NALU_HYPRE_Real *) temp1, 0,
                               coarse_stencil_cnt[i] - 1);

                  /*---------------------------------------------------------------
                   * swap the stencil_vals to agree with the rank swapping.
                   *---------------------------------------------------------------*/
                  temp3  = nalu_hypre_TAlloc(NALU_HYPRE_Real,  coarse_stencil_cnt[i], NALU_HYPRE_MEMORY_HOST);
                  for (j = 0; j < coarse_stencil_cnt[i]; j++)
                  {
                     m         = temp1[j];
                     temp3[j]  = stencil_vals[m];
                  }
                  for (j = 0; j < coarse_stencil_cnt[i]; j++)
                  {
                     stencil_vals[j] = temp3[j];
                  }

                  nalu_hypre_TFree(temp1, NALU_HYPRE_MEMORY_HOST);
                  nalu_hypre_TFree(temp3, NALU_HYPRE_MEMORY_HOST);
               }

               /*-----------------------------------------------------------------
                * Compute the weights for the averaged stencil contributions.
                * We need to convert the ranks back to stencil_shapes and then
                * find the abs of the stencil shape.
                *-----------------------------------------------------------------*/
               temp3 = nalu_hypre_TAlloc(NALU_HYPRE_Real,  coarse_stencil_cnt[i], NALU_HYPRE_MEMORY_HOST);
               for (j = 0; j < coarse_stencil_cnt[i]; j++)
               {
                  InverseMapStencilRank(interface_stencil_ranks[i][j], index_temp);
                  AbsStencilShape(index_temp, abs_stencil_shape);
                  temp3[j] = weights[abs_stencil_shape];
               }

               /*-----------------------------------------------------------------
                * Compare the coarse stencil and the interface stencil and
                * extract the common stencil shapes.
                * WE ARE ASSUMING THAT THE COARSE INTERFACE STENCIL HAS SOME
                * COMMON STENCIL SHAPE WITH THE COARSE STENCIL.
                *-----------------------------------------------------------------*/
               common_rank_stencils = nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_size, NALU_HYPRE_MEMORY_HOST);
               common_stencil_ranks = nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_size, NALU_HYPRE_MEMORY_HOST);
               common_stencil_i    = nalu_hypre_TAlloc(NALU_HYPRE_Int,  stencil_size, NALU_HYPRE_MEMORY_HOST);

               l = 0;
               m = 0;
               for (j = 0; j < stencil_size; j++)
               {
                  while (  (l < coarse_stencil_cnt[i])
                           && (stencil_ranks[j] > interface_stencil_ranks[i][l]) )
                  {
                     l++;
                  }

                  if (l >= coarse_stencil_cnt[i])
                  {
                     break;
                  }
                  /*--------------------------------------------------------------
                   * Check if a common stencil shape rank has been found.
                   *--------------------------------------------------------------*/
                  if (   (stencil_ranks[j] == interface_stencil_ranks[i][l])
                         && (l < coarse_stencil_cnt[i]) )
                  {
                     common_rank_stencils[m] = rank_stencils[ stencil_ranks[j] ];
                     common_stencil_ranks[m] = stencil_ranks[j];
                     common_stencil_i[m++]  = l;
                     l++;
                  }
               }
               /*-----------------------------------------------------------------
                * Find the contribution and weights for the averaged stencils.
                *-----------------------------------------------------------------*/
               for (j = 0; j < m; j++)
               {
                  nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(
                                     stencils, common_rank_stencils[j]),
                                  stencil_shape_i);
                  AbsStencilShape(stencil_shape_i, abs_stencil_shape);

                  crse_ptr = nalu_hypre_StructMatrixExtractPointerByIndex(crse_smatrix,
                                                                     parents_cnodes[i],
                                                                     stencil_shape_i);

                  /*-----------------------------------------------------------------
                   *  For a compact stencil (e.g., -1 <= nalu_hypre_Index[i] <= 1, i= 0-2),
                   *  the value of abs_stencil_shape can be used to determine the
                   *  stencil:
                   *     abs_stencil_shape=   3   only corners in 3-d
                   *                          2   corners in 2-d; or the centre plane
                   *                              in 3-d, or e,w,n,s of the bottom
                   *                              or top plane in 3-d
                   *                          1   e,w in 1-d; or e,w,n,s in 2-d;
                   *                              or the centre plane in 3-d,
                   *                              or c of the bottom or top plane
                   *                              in 3-d
                   *                          0   c in 1-d, 2-d, or 3-d.
                   *-----------------------------------------------------------------*/

                  switch (abs_stencil_shape)
                  {
                     case 3:    /* corners of 3-d stencil */

                        l = common_stencil_i[j];
                        crse_ptr[fine_interface_ranks[i]] = stencil_vals[l];

                        break;


                     case 2:    /* corners in 2-d or edges in 3-d */

                        if (ndim == 2)
                        {
                           l = common_stencil_i[j];
                           crse_ptr[fine_interface_ranks[i]] = stencil_vals[l];
                        }

                        else if (ndim == 3)
                        {
                           /*----------------------------------------------------------
                            * The edge values are weighted sums of the averaged
                            * coefficients. The weights and averaged coefficients must
                            * be found. The contributions are found using the stencil
                            * ranks and the stencil ordering
                            * top: 14  12  13  centre:  5  3  4  bottom 23   21   22
                            *      11   9  10           2  0  1         20   18   19
                            *      17  15  16           8  6  7         26   24   25
                            *----------------------------------------------------------*/
                           l    =  common_stencil_ranks[j];
                           temp1 =  nalu_hypre_TAlloc(NALU_HYPRE_Int,  2, NALU_HYPRE_MEMORY_HOST);

                           switch (l)
                           {
                              case 4:   /* centre plane ne */

                                 temp1[0] = 13;
                                 temp1[1] = 22;
                                 break;

                              case 5:   /* centre plane nw */

                                 temp1[0] = 14;
                                 temp1[1] = 23;
                                 break;

                              case 7:   /* centre plane se */

                                 temp1[0] = 16;
                                 temp1[1] = 25;
                                 break;

                              case 8:   /* centre plane sw */

                                 temp1[0] = 17;
                                 temp1[1] = 26;
                                 break;

                              case 10:   /* top plane e */

                                 temp1[0] = 13;
                                 temp1[1] = 16;
                                 break;

                              case 11:   /* top plane w */

                                 temp1[0] = 14;
                                 temp1[1] = 17;
                                 break;

                              case 12:   /* top plane n */

                                 temp1[0] = 13;
                                 temp1[1] = 14;
                                 break;

                              case 15:   /* top plane s */

                                 temp1[0] = 16;
                                 temp1[1] = 17;
                                 break;

                              case 19:   /* bottom plane e */

                                 temp1[0] = 22;
                                 temp1[1] = 25;
                                 break;

                              case 20:   /* bottom plane w */

                                 temp1[0] = 23;
                                 temp1[1] = 26;
                                 break;

                              case 21:   /* bottom plane n */

                                 temp1[0] = 22;
                                 temp1[1] = 23;
                                 break;

                              case 24:   /* bottom plane s */

                                 temp1[0] = 25;
                                 temp1[1] = 26;
                                 break;
                           }


                           /*-------------------------------------------------------
                            *  Add up the weighted contributions of the interface
                            *  stencils. This involves searching the ranks of
                            *  interface_stencil_ranks. The weights must be averaged.
                            *-------------------------------------------------------*/

                           l = common_stencil_i[j];
                           sum = temp3[l];
                           sum_contrib = sum * stencil_vals[l];

                           n = 1;
                           for (l = 0; l < 2; l++)
                           {
                              while (  (n < coarse_stencil_cnt[i])
                                       && (interface_stencil_ranks[i][n] < temp1[l]) )
                              {
                                 n++;
                              }

                              if (n >= coarse_stencil_cnt[i])
                              {
                                 break;
                              }

                              if (interface_stencil_ranks[i][n] == temp1[l])
                              {
                                 sum += temp3[n];
                                 sum_contrib += temp3[n] * stencil_vals[n];
                                 n++;
                              }
                           }

                           sum_contrib /= sum;   /* average out the weights */
                           l = common_stencil_i[j];
                           crse_ptr[fine_interface_ranks[i]] = sum_contrib;

                           nalu_hypre_TFree(temp1, NALU_HYPRE_MEMORY_HOST);

                        }    /* else if (ndim == 3) */

                        break;

                     case 1:     /* e,w in 1-d, or edges in 2-d, or faces in 3-d */

                        if (ndim == 1)
                        {
                           l = common_stencil_i[j];
                           crse_ptr[fine_interface_ranks[i]] = stencil_vals[l];
                        }

                        else if (ndim == 2)
                        {
                           l    =  common_stencil_ranks[j];
                           temp1 =  nalu_hypre_TAlloc(NALU_HYPRE_Int,  2, NALU_HYPRE_MEMORY_HOST);

                           switch (l)
                           {
                              case 1:   /* e */

                                 temp1[0] = 4;
                                 temp1[1] = 7;
                                 break;

                              case 2:   /* w */

                                 temp1[0] = 5;
                                 temp1[1] = 8;
                                 break;

                              case 3:   /* n */

                                 temp1[0] = 4;
                                 temp1[1] = 5;
                                 break;

                              case 6:   /* s */

                                 temp1[0] = 7;
                                 temp1[1] = 8;
                                 break;
                           }

                           /*-------------------------------------------------------
                            *  Add up the weighted contributions of the interface
                            *  stencils.
                            *-------------------------------------------------------*/

                           l = common_stencil_i[j];
                           sum = temp3[l];
                           sum_contrib = sum * stencil_vals[l];

                           n = 1;
                           for (l = 0; l < 2; l++)
                           {
                              while (  (n < coarse_stencil_cnt[i])
                                       && (interface_stencil_ranks[i][n] < temp1[l]) )
                              {
                                 n++;
                              }

                              if (n >= coarse_stencil_cnt[i])
                              {
                                 break;
                              }

                              if (interface_stencil_ranks[i][n] == temp1[l])
                              {
                                 sum += temp3[n];
                                 sum_contrib += temp3[n] * stencil_vals[n];
                                 n++;
                              }
                           }

                           sum_contrib /= sum;   /* average out the weights */
                           l = common_stencil_i[j];
                           crse_ptr[fine_interface_ranks[i]] = sum_contrib;

                           nalu_hypre_TFree(temp1, NALU_HYPRE_MEMORY_HOST);

                        }   /* else if (ndim == 2) */

                        else /* 3-d */
                        {
                           l    =  common_stencil_ranks[j];
                           temp1 =  nalu_hypre_TAlloc(NALU_HYPRE_Int,  8, NALU_HYPRE_MEMORY_HOST);

                           switch (l)
                           {
                              case 1:   /* centre plane e */

                                 temp1[0] = 4;
                                 temp1[1] = 7;
                                 temp1[2] = 10;
                                 temp1[3] = 13;
                                 temp1[4] = 16;
                                 temp1[5] = 19;
                                 temp1[6] = 22;
                                 temp1[7] = 25;
                                 break;

                              case 2:   /* centre plane w */

                                 temp1[0] = 5;
                                 temp1[1] = 8;
                                 temp1[2] = 11;
                                 temp1[3] = 14;
                                 temp1[4] = 17;
                                 temp1[5] = 20;
                                 temp1[6] = 23;
                                 temp1[7] = 26;
                                 break;

                              case 3:   /* centre plane n */

                                 temp1[0] = 4;
                                 temp1[1] = 5;
                                 temp1[2] = 12;
                                 temp1[3] = 13;
                                 temp1[4] = 14;
                                 temp1[5] = 21;
                                 temp1[6] = 22;
                                 temp1[7] = 23;
                                 break;

                              case 6:   /* centre plane s */

                                 temp1[0] = 7;
                                 temp1[1] = 8;
                                 temp1[2] = 15;
                                 temp1[3] = 16;
                                 temp1[4] = 17;
                                 temp1[5] = 24;
                                 temp1[6] = 25;
                                 temp1[7] = 26;
                                 break;

                              case 9:   /* top plane c */

                                 for (n = 0; n < 8; n++)
                                 {
                                    temp1[n] = 10 + n;
                                 }
                                 break;

                              case 18:   /* bottom plane c */

                                 for (n = 0; n < 8; n++)
                                 {
                                    temp1[n] = 19 + n;
                                 }
                                 break;

                           }

                           /*-------------------------------------------------------
                            *  Add up the weighted contributions of the interface
                            *  stencils.
                            *-------------------------------------------------------*/

                           l = common_stencil_i[j];
                           sum = temp3[l];
                           sum_contrib = sum * stencil_vals[l];

                           n = 1;
                           for (l = 0; l < 8; l++)
                           {
                              while (   (n < coarse_stencil_cnt[i])
                                        && (interface_stencil_ranks[i][n] < temp1[l]) )
                              {
                                 n++;
                              }

                              if (n >= coarse_stencil_cnt[i])
                              {
                                 break;
                              }

                              if (interface_stencil_ranks[i][n] == temp1[l])
                              {
                                 sum += temp3[n];
                                 sum_contrib += temp3[n] * stencil_vals[n];
                                 n++;
                              }
                           }

                           sum_contrib /= sum;   /* average out the weights */
                           l = common_stencil_i[j];
                           crse_ptr[fine_interface_ranks[i]] = sum_contrib;

                           nalu_hypre_TFree(temp1, NALU_HYPRE_MEMORY_HOST);

                        }    /* else */

                        break;

                  }   /* switch(abs_stencil_shape) */
               }       /* for (j= 0; j< m; j++) */

               nalu_hypre_TFree(interface_stencil_ranks[i], NALU_HYPRE_MEMORY_HOST);

               nalu_hypre_TFree(stencil_vals, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(temp3, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(common_rank_stencils, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(common_stencil_ranks, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(common_stencil_ranks, NALU_HYPRE_MEMORY_HOST);
               nalu_hypre_TFree(common_stencil_i, NALU_HYPRE_MEMORY_HOST);

            }          /* for (i= 0; i< cnt1; i++) */

            nalu_hypre_TFree(box_to_ranks_cnt, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(interface_max_stencil_ranks, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(interface_max_stencil_cnt, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(interface_rank_stencils, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(interface_stencil_ranks, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(coarse_stencil_cnt, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(fine_interface_ranks, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(parents_cnodes, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(vals, NALU_HYPRE_MEMORY_HOST);

            /*-----------------------------------------------------------
             *  Box fi is completed.
             *-----------------------------------------------------------*/
         }     /* for (fi= 0; fi< box_array_size; fi++) */

         nalu_hypre_TFree(stencil_ranks, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(rank_stencils, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(cdata_space_ranks, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(box_graph_cnts, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < box_array_size; i++)
         {
            if (box_graph_indices[i])
            {
               nalu_hypre_TFree(box_graph_indices[i], NALU_HYPRE_MEMORY_HOST);
            }
         }
         nalu_hypre_TFree(box_graph_indices, NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_TFree(box_starts, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(box_ends, NALU_HYPRE_MEMORY_HOST);
      }  /* for (var1= 0; var1< nvars; var1++) */
   }    /* if (nUventries > 0) */


   /*--------------------------------------------------------------------------
    *  STEP 3:
    *        Coarsened f/c interface coefficients can be used to create the
    *        centre components along the coarsened f/c nodes now. Loop over
    *        the coarsened fbox_bdy's and set the centre stencils.
    *--------------------------------------------------------------------------*/
   nalu_hypre_ClearIndex(index_temp);
   for (var1 = 0; var1 < nvars; var1++)
   {
      /* only like variables couple. */
      smatrix_var  = nalu_hypre_SStructPMatrixSMatrix(A_crse, var1, var1);
      stencils     = nalu_hypre_SStructPMatrixSStencil(A_crse, var1, var1);
      stencil_size = nalu_hypre_StructStencilSize(stencils);
      a_ptrs       = nalu_hypre_TAlloc(NALU_HYPRE_Real *,  stencil_size, NALU_HYPRE_MEMORY_HOST);

      rank_stencils = nalu_hypre_TAlloc(NALU_HYPRE_Int,  max_stencil_size, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < stencil_size; i++)
      {
         nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils, i),
                         stencil_shape_i);
         MapStencilRank(stencil_shape_i, rank);
         rank_stencils[rank] = i;
      }
      centre = rank_stencils[0];

      cgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructPMatrixPGrid(A_crse), var1);
      cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);

      nalu_hypre_ForBoxI(ci, cgrid_boxes)
      {
         A_dbox     = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(smatrix_var), ci);
         fbox_bdy_ci = fbox_bdy[var1][ci];

         for (i = 0; i < stencil_size; i++)
         {
            nalu_hypre_CopyIndex(nalu_hypre_StructStencilElement(stencils, i),
                            stencil_shape_i);
            a_ptrs[i] = nalu_hypre_StructMatrixExtractPointerByIndex(smatrix_var,
                                                                ci,
                                                                stencil_shape_i);
         }

         /*------------------------------------------------------------------
          * Loop over the boundaries of each patch inside cgrid_box ci.
          * These patch boxes must be coarsened to get the correct extents.
          *------------------------------------------------------------------*/
         nalu_hypre_ForBoxArrayI(arrayi, fbox_bdy_ci)
         {
            fbox_bdy_ci_fi = nalu_hypre_BoxArrayArrayBoxArray(fbox_bdy_ci, arrayi);
            nalu_hypre_ForBoxI(fi, fbox_bdy_ci_fi)
            {
               fgrid_box = nalu_hypre_BoxArrayBox(fbox_bdy_ci_fi, fi);
               nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(fgrid_box), index_temp,
                                           stridef, nalu_hypre_BoxIMin(&fine_box));
               nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(fgrid_box), index_temp,
                                           stridef, nalu_hypre_BoxIMax(&fine_box));

               nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(&fine_box), cstart);
               nalu_hypre_BoxGetSize(&fine_box, loop_size);

#define DEVICE_VAR is_device_ptr(a_ptrs)
               nalu_hypre_BoxLoop1Begin(ndim, loop_size,
                                   A_dbox, cstart, stridec, iA);
               {
                  NALU_HYPRE_Int i;
                  for (i = 0; i < stencil_size; i++)
                  {
                     if (i != centre)
                     {
                        a_ptrs[centre][iA] -= a_ptrs[i][iA];
                     }
                  }
               }
               nalu_hypre_BoxLoop1End(iA);
#undef DEVICE_VAR

            }  /* nalu_hypre_ForBoxI(fi, fbox_bdy_ci_fi) */
         }      /* nalu_hypre_ForBoxArrayI(arrayi, fbox_bdy_ci) */
      }          /* nalu_hypre_ForBoxI(ci, cgrid_boxes) */

      nalu_hypre_TFree(a_ptrs, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(rank_stencils, NALU_HYPRE_MEMORY_HOST);

   }  /* for (var1= 0; var1< nvars; var1++) */

   for (var1 = 0; var1 < nvars; var1++)
   {
      cgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructPMatrixPGrid(A_crse), var1);
      cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);

      fgrid = nalu_hypre_SStructPGridSGrid(nalu_hypre_SStructPMatrixPGrid(A_pmatrix), var1);
      fgrid_boxes = nalu_hypre_StructGridBoxes(fgrid);

      nalu_hypre_ForBoxI(ci, cgrid_boxes)
      {
         nalu_hypre_BoxArrayDestroy(fgrid_crse_extents[var1][ci]);
         nalu_hypre_BoxArrayDestroy(fbox_interior[var1][ci]);
         nalu_hypre_BoxArrayArrayDestroy(fbox_bdy[var1][ci]);
         nalu_hypre_TFree(interior_fboxi[var1][ci], NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(bdy_fboxi[var1][ci], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(fgrid_crse_extents[var1], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fbox_interior[var1], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(fbox_bdy[var1], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(interior_fboxi[var1], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(bdy_fboxi[var1], NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_ForBoxI(fi, fgrid_boxes)
      {
         nalu_hypre_TFree(cboxi_fboxes[var1][fi], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(cboxi_fboxes[var1], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(cboxi_fcnt[var1], NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(fgrid_crse_extents, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fbox_interior, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fbox_bdy, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(interior_fboxi, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(bdy_fboxi, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(cboxi_fboxes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(cboxi_fcnt, NALU_HYPRE_MEMORY_HOST);

   return 0;
}

