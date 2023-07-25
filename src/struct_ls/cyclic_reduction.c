/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 * Cyclic reduction algorithm (coded as if it were a 1D MG method)
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"

#define DEBUG 0

/*--------------------------------------------------------------------------
 * Macros
 *--------------------------------------------------------------------------*/

#define nalu_hypre_CycRedSetCIndex(base_index, base_stride, level, cdir, cindex) \
   {                                                                    \
      if (level > 0)                                                    \
         nalu_hypre_SetIndex3(cindex, 0, 0, 0);                              \
      else                                                              \
         nalu_hypre_CopyIndex(base_index,  cindex);                          \
      nalu_hypre_IndexD(cindex, cdir) += 0;                                  \
   }

#define nalu_hypre_CycRedSetFIndex(base_index, base_stride, level, cdir, findex) \
   {                                                                    \
      if (level > 0)                                                    \
         nalu_hypre_SetIndex3(findex, 0, 0, 0);                              \
      else                                                              \
         nalu_hypre_CopyIndex(base_index,  findex);                          \
      nalu_hypre_IndexD(findex, cdir) += 1;                                  \
   }

#define nalu_hypre_CycRedSetStride(base_index, base_stride, level, cdir, stride) \
   {                                                                    \
      if (level > 0)                                                    \
         nalu_hypre_SetIndex3(stride, 1, 1, 1);                              \
      else                                                              \
         nalu_hypre_CopyIndex(base_stride, stride);                          \
      nalu_hypre_IndexD(stride, cdir) *= 2;                                  \
   }

/*--------------------------------------------------------------------------
 * nalu_hypre_CyclicReductionData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm              comm;

   NALU_HYPRE_Int             num_levels;

   NALU_HYPRE_Int             ndim;
   NALU_HYPRE_Int             cdir;         /* coarsening direction */
   nalu_hypre_Index           base_index;
   nalu_hypre_Index           base_stride;

   nalu_hypre_StructGrid    **grid_l;

   nalu_hypre_BoxArray       *base_points;
   nalu_hypre_BoxArray      **fine_points_l;

   NALU_HYPRE_MemoryLocation  memory_location; /* memory location of data */
   NALU_HYPRE_Real           *data;
   NALU_HYPRE_Real           *data_const;
   nalu_hypre_StructMatrix  **A_l;
   nalu_hypre_StructVector  **x_l;

   nalu_hypre_ComputePkg    **down_compute_pkg_l;
   nalu_hypre_ComputePkg    **up_compute_pkg_l;

   NALU_HYPRE_Int             time_index;
   NALU_HYPRE_BigInt          solve_flops;
   NALU_HYPRE_Int             max_levels;
} nalu_hypre_CyclicReductionData;

/*--------------------------------------------------------------------------
 * nalu_hypre_CyclicReductionCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_CyclicReductionCreate( MPI_Comm  comm )
{
   nalu_hypre_CyclicReductionData *cyc_red_data;

   cyc_red_data = nalu_hypre_CTAlloc(nalu_hypre_CyclicReductionData,  1, NALU_HYPRE_MEMORY_HOST);

   (cyc_red_data -> comm) = comm;
   (cyc_red_data -> ndim) = 3;
   (cyc_red_data -> cdir) = 0;
   (cyc_red_data -> time_index)  = nalu_hypre_InitializeTiming("CyclicReduction");
   (cyc_red_data -> max_levels)  = -1;

   /* set defaults */
   nalu_hypre_SetIndex3((cyc_red_data -> base_index), 0, 0, 0);
   nalu_hypre_SetIndex3((cyc_red_data -> base_stride), 1, 1, 1);

   (cyc_red_data -> memory_location) = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());

   return (void *) cyc_red_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CycRedCreateCoarseOp
 *
 * NOTE: This routine assumes that domain boundary ghost zones (i.e., ghost
 * zones that do not intersect the grid) have the identity equation in them.
 * This is currently insured by the MatrixAssemble routine.
 *--------------------------------------------------------------------------*/

nalu_hypre_StructMatrix *
nalu_hypre_CycRedCreateCoarseOp( nalu_hypre_StructMatrix *A,
                            nalu_hypre_StructGrid   *coarse_grid,
                            NALU_HYPRE_Int           cdir        )
{
   NALU_HYPRE_Int              ndim = nalu_hypre_StructMatrixNDim(A);
   nalu_hypre_StructMatrix    *Ac;
   nalu_hypre_Index           *Ac_stencil_shape;
   nalu_hypre_StructStencil   *Ac_stencil;
   NALU_HYPRE_Int              Ac_stencil_size;
   NALU_HYPRE_Int              Ac_num_ghost[] = {0, 0, 0, 0, 0, 0};

   NALU_HYPRE_Int              i;
   NALU_HYPRE_Int              stencil_rank;

   /*-----------------------------------------------
    * Define Ac_stencil
    *-----------------------------------------------*/

   stencil_rank = 0;

   /*-----------------------------------------------
    * non-symmetric case:
    *
    * 3 point fine grid stencil produces 3 point Ac
    *-----------------------------------------------*/

   if (!nalu_hypre_StructMatrixSymmetric(A))
   {
      Ac_stencil_size = 3;
      Ac_stencil_shape = nalu_hypre_CTAlloc(nalu_hypre_Index,  Ac_stencil_size, NALU_HYPRE_MEMORY_HOST);
      for (i = -1; i < 2; i++)
      {
         /* Storage for 3 elements (c,w,e) */
         nalu_hypre_SetIndex3(Ac_stencil_shape[stencil_rank], 0, 0, 0);
         nalu_hypre_IndexD(Ac_stencil_shape[stencil_rank], cdir) = i;
         stencil_rank++;
      }
   }

   /*-----------------------------------------------
    * symmetric case:
    *
    * 3 point fine grid stencil produces 3 point Ac
    *
    * Only store the lower triangular part + diagonal = 2 entries,
    * lower triangular means the lower triangular part on the matrix
    * in the standard lexicalgraphic ordering.
    *-----------------------------------------------*/

   else
   {
      Ac_stencil_size = 2;
      Ac_stencil_shape = nalu_hypre_CTAlloc(nalu_hypre_Index,  Ac_stencil_size, NALU_HYPRE_MEMORY_HOST);
      for (i = -1; i < 1; i++)
      {

         /* Storage for 2 elements in (c,w) */
         nalu_hypre_SetIndex3(Ac_stencil_shape[stencil_rank], 0, 0, 0);
         nalu_hypre_IndexD(Ac_stencil_shape[stencil_rank], cdir) = i;
         stencil_rank++;
      }
   }

   Ac_stencil = nalu_hypre_StructStencilCreate(ndim, Ac_stencil_size, Ac_stencil_shape);

   Ac = nalu_hypre_StructMatrixCreate(nalu_hypre_StructMatrixComm(A),
                                 coarse_grid, Ac_stencil);

   nalu_hypre_StructStencilDestroy(Ac_stencil);

   /*-----------------------------------------------
    * Coarse operator in symmetric iff fine operator is
    *-----------------------------------------------*/

   nalu_hypre_StructMatrixSymmetric(Ac) = nalu_hypre_StructMatrixSymmetric(A);

   /*-----------------------------------------------
    * Set number of ghost points
    *-----------------------------------------------*/

   Ac_num_ghost[2 * cdir] = 1;
   if (!nalu_hypre_StructMatrixSymmetric(A))
   {
      Ac_num_ghost[2 * cdir + 1] = 1;
   }
   nalu_hypre_StructMatrixSetNumGhost(Ac, Ac_num_ghost);

   nalu_hypre_StructMatrixInitializeShell(Ac);

   return Ac;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CycRedSetupCoarseOp
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CycRedSetupCoarseOp( nalu_hypre_StructMatrix *A,
                           nalu_hypre_StructMatrix *Ac,
                           nalu_hypre_Index         cindex,
                           nalu_hypre_Index         cstride,
                           NALU_HYPRE_Int           cdir )
{
   nalu_hypre_Index             index;

   nalu_hypre_StructGrid       *fgrid;
   NALU_HYPRE_Int              *fgrid_ids;
   nalu_hypre_StructGrid       *cgrid;
   nalu_hypre_BoxArray         *cgrid_boxes;
   NALU_HYPRE_Int              *cgrid_ids;
   nalu_hypre_Box              *cgrid_box;
   nalu_hypre_IndexRef          cstart;
   nalu_hypre_Index             stridec;
   nalu_hypre_Index             fstart;
   nalu_hypre_IndexRef          stridef;
   nalu_hypre_Index             loop_size;

   NALU_HYPRE_Int               fi, ci;

   nalu_hypre_Box              *A_dbox;
   nalu_hypre_Box              *Ac_dbox;

   NALU_HYPRE_Real             *a_cc, *a_cw, *a_ce;
   NALU_HYPRE_Real             *ac_cc, *ac_cw, *ac_ce;

   NALU_HYPRE_Int               offsetA;

   stridef = cstride;
   nalu_hypre_SetIndex3(stridec, 1, 1, 1);

   fgrid = nalu_hypre_StructMatrixGrid(A);
   fgrid_ids = nalu_hypre_StructGridIDs(fgrid);

   cgrid = nalu_hypre_StructMatrixGrid(Ac);
   cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);
   cgrid_ids = nalu_hypre_StructGridIDs(cgrid);

   fi = 0;
   nalu_hypre_ForBoxI(ci, cgrid_boxes)
   {
      while (fgrid_ids[fi] != cgrid_ids[ci])
      {
         fi++;
      }

      cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

      cstart = nalu_hypre_BoxIMin(cgrid_box);
      nalu_hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

      A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), fi);
      Ac_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(Ac), ci);

      /*-----------------------------------------------
       * Extract pointers for 3-point fine grid operator:
       *
       * a_cc is pointer for center coefficient
       * a_cw is pointer for west coefficient
       * a_ce is pointer for east coefficient
       *-----------------------------------------------*/

      nalu_hypre_SetIndex3(index, 0, 0, 0);
      a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      nalu_hypre_IndexD(index, cdir) = -1;
      a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      nalu_hypre_IndexD(index, cdir) = 1;
      a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, fi, index);

      /*-----------------------------------------------
       * Extract pointers for coarse grid operator - always 3-point:
       *
       * If A is symmetric so is Ac.  We build only the
       * lower triangular part (plus diagonal).
       *
       * ac_cc is pointer for center coefficient (etc.)
       *-----------------------------------------------*/

      nalu_hypre_SetIndex3(index, 0, 0, 0);
      ac_cc = nalu_hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);

      nalu_hypre_IndexD(index, cdir) = -1;
      ac_cw = nalu_hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);

      if (!nalu_hypre_StructMatrixSymmetric(A))
      {
         nalu_hypre_IndexD(index, cdir) = 1;
         ac_ce = nalu_hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);
      }

      /*-----------------------------------------------
       * Define offsets for fine grid stencil and interpolation
       *
       * In the BoxLoop below I assume iA and iP refer
       * to data associated with the point which we are
       * building the stencil for.  The below offsets
       * are used in refering to data associated with
       * other points.
       *-----------------------------------------------*/

      nalu_hypre_SetIndex3(index, 0, 0, 0);
      nalu_hypre_IndexD(index, cdir) = 1;
      offsetA = nalu_hypre_BoxOffsetDistance(A_dbox, index);

      /*-----------------------------------------------
       * non-symmetric case
       *-----------------------------------------------*/

      if (!nalu_hypre_StructMatrixSymmetric(A))
      {
         nalu_hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(ac_cw,a_cw,a_cc,ac_cc,a_ce,ac_ce)
         nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                             A_dbox, fstart, stridef, iA,
                             Ac_dbox, cstart, stridec, iAc);
         {
            NALU_HYPRE_Int iAm1 = iA - offsetA;
            NALU_HYPRE_Int iAp1 = iA + offsetA;

            ac_cw[iAc] = -a_cw[iA] * a_cw[iAm1] / a_cc[iAm1];

            ac_cc[iAc] = a_cc[iA] - a_cw[iA] * a_ce[iAm1] / a_cc[iAm1] -
                         a_ce[iA] * a_cw[iAp1] / a_cc[iAp1];

            ac_ce[iAc] = -a_ce[iA] * a_ce[iAp1] / a_cc[iAp1];

         }
         nalu_hypre_BoxLoop2End(iA, iAc);
#undef DEVICE_VAR
      }

      /*-----------------------------------------------
       * symmetric case
       *-----------------------------------------------*/

      else
      {
         nalu_hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(ac_cw,a_cw,a_cc,ac_cc,a_ce)
         nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                             A_dbox, fstart, stridef, iA,
                             Ac_dbox, cstart, stridec, iAc);
         {
            NALU_HYPRE_Int iAm1 = iA - offsetA;
            NALU_HYPRE_Int iAp1 = iA + offsetA;

            ac_cw[iAc] = -a_cw[iA] * a_cw[iAm1] / a_cc[iAm1];

            ac_cc[iAc] = a_cc[iA] - a_cw[iA] * a_ce[iAm1] / a_cc[iAm1] -
                         a_ce[iA] * a_cw[iAp1] / a_cc[iAp1];
         }
         nalu_hypre_BoxLoop2End(iA, iAc);
#undef DEVICE_VAR
      }

   } /* end ForBoxI */

   nalu_hypre_StructMatrixAssemble(Ac);

   /*-----------------------------------------------------------------------
    * Collapse stencil in periodic direction on coarsest grid.
    *-----------------------------------------------------------------------*/

   if (nalu_hypre_IndexD(nalu_hypre_StructGridPeriodic(cgrid), cdir) == 1)
   {
      nalu_hypre_ForBoxI(ci, cgrid_boxes)
      {
         cgrid_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

         cstart = nalu_hypre_BoxIMin(cgrid_box);

         Ac_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(Ac), ci);

         /*-----------------------------------------------
          * Extract pointers for coarse grid operator - always 3-point:
          *
          * If A is symmetric so is Ac.  We build only the
          * lower triangular part (plus diagonal).
          *
          * ac_cc is pointer for center coefficient (etc.)
          *-----------------------------------------------*/

         nalu_hypre_SetIndex3(index, 0, 0, 0);
         ac_cc = nalu_hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);

         nalu_hypre_IndexD(index, cdir) = -1;
         ac_cw = nalu_hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);

         if (!nalu_hypre_StructMatrixSymmetric(A))
         {
            nalu_hypre_IndexD(index, cdir) = 1;
            ac_ce = nalu_hypre_StructMatrixExtractPointerByIndex(Ac, ci, index);
         }

         /*-----------------------------------------------
          * non-symmetric case
          *-----------------------------------------------*/

         if (!nalu_hypre_StructMatrixSymmetric(A))
         {
            nalu_hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(ac_cc,ac_cw,ac_ce)
            nalu_hypre_BoxLoop1Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                Ac_dbox, cstart, stridec, iAc);
            {
               ac_cc[iAc] += (ac_cw[iAc] + ac_ce[iAc]);
               ac_cw[iAc]  =  0.0;
               ac_ce[iAc]  =  0.0;
            }
            nalu_hypre_BoxLoop1End(iAc);
#undef DEVICE_VAR
         }

         /*-----------------------------------------------
          * symmetric case
          *-----------------------------------------------*/

         else
         {
            nalu_hypre_BoxGetSize(cgrid_box, loop_size);

#define DEVICE_VAR is_device_ptr(ac_cc,ac_cw)
            nalu_hypre_BoxLoop1Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                Ac_dbox, cstart, stridec, iAc);
            {
               ac_cc[iAc] += (2.0 * ac_cw[iAc]);
               ac_cw[iAc]  =  0.0;
            }
            nalu_hypre_BoxLoop1End(iAc);
#undef DEVICE_VAR
         }

      } /* end ForBoxI */

   }

   nalu_hypre_StructMatrixAssemble(Ac);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CyclicReductionSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CyclicReductionSetup( void               *cyc_red_vdata,
                            nalu_hypre_StructMatrix *A,
                            nalu_hypre_StructVector *b,
                            nalu_hypre_StructVector *x             )
{
   nalu_hypre_CyclicReductionData *cyc_red_data = (nalu_hypre_CyclicReductionData *) cyc_red_vdata;

   MPI_Comm                comm        = (cyc_red_data -> comm);
   NALU_HYPRE_Int               cdir        = (cyc_red_data -> cdir);
   nalu_hypre_IndexRef          base_index  = (cyc_red_data -> base_index);
   nalu_hypre_IndexRef          base_stride = (cyc_red_data -> base_stride);

   NALU_HYPRE_Int               num_levels;
   NALU_HYPRE_Int               max_levels = -1;
   nalu_hypre_StructGrid      **grid_l;
   nalu_hypre_BoxArray         *base_points;
   nalu_hypre_BoxArray        **fine_points_l;
   NALU_HYPRE_Real             *data;
   NALU_HYPRE_Real             *data_const;
   NALU_HYPRE_Int               data_size = 0;
   NALU_HYPRE_Int               data_size_const = 0;
   nalu_hypre_StructMatrix    **A_l;
   nalu_hypre_StructVector    **x_l;
   nalu_hypre_ComputePkg      **down_compute_pkg_l;
   nalu_hypre_ComputePkg      **up_compute_pkg_l;
   nalu_hypre_ComputeInfo      *compute_info;

   nalu_hypre_Index             cindex;
   nalu_hypre_Index             findex;
   nalu_hypre_Index             stride;

   nalu_hypre_StructGrid       *grid;
   nalu_hypre_Box              *cbox;
   NALU_HYPRE_Int               l;
   NALU_HYPRE_Int               flop_divisor;
   NALU_HYPRE_Int               x_num_ghost[] = {0, 0, 0, 0, 0, 0};

   NALU_HYPRE_MemoryLocation    memory_location = nalu_hypre_StructMatrixMemoryLocation(A);

   /*-----------------------------------------------------
    * Set up coarse grids
    *-----------------------------------------------------*/

   grid = nalu_hypre_StructMatrixGrid(A);

   /* Compute a preliminary num_levels value based on the grid */
   cbox = nalu_hypre_BoxDuplicate(nalu_hypre_StructGridBoundingBox(grid));
   num_levels = nalu_hypre_Log2(nalu_hypre_BoxSizeD(cbox, cdir)) + 2;
   if (cyc_red_data -> max_levels > 0)
   {
      max_levels = (cyc_red_data -> max_levels);
   }


   grid_l    = nalu_hypre_TAlloc(nalu_hypre_StructGrid *,  num_levels, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_StructGridRef(grid, &grid_l[0]);

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   data_location = nalu_hypre_StructGridDataLocation(grid);
#endif
   for (l = 0; ; l++)
   {
      /* set cindex and stride */
      nalu_hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      nalu_hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* check to see if we should coarsen */
      if ( nalu_hypre_BoxIMinD(cbox, cdir) == nalu_hypre_BoxIMaxD(cbox, cdir) ||
           (l == (max_levels - 1)))
      {
         /* stop coarsening */
         break;
      }

      /* coarsen cbox */
      nalu_hypre_ProjectBox(cbox, cindex, stride);
      nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMin(cbox), cindex, stride,
                                  nalu_hypre_BoxIMin(cbox));
      nalu_hypre_StructMapFineToCoarse(nalu_hypre_BoxIMax(cbox), cindex, stride,
                                  nalu_hypre_BoxIMax(cbox));

      /* coarsen the grid */
      nalu_hypre_StructCoarsen(grid_l[l], cindex, stride, 1, &grid_l[l + 1]);
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      nalu_hypre_StructGridDataLocation(grid_l[l + 1]) = data_location;
#endif
   }
   num_levels = l + 1;

   /* free up some things */
   nalu_hypre_BoxDestroy(cbox);

   (cyc_red_data -> ndim)            = nalu_hypre_StructGridNDim(grid);
   (cyc_red_data -> num_levels)      = num_levels;
   (cyc_red_data -> grid_l)          = grid_l;

   /*-----------------------------------------------------
    * Set up base points
    *-----------------------------------------------------*/

   base_points = nalu_hypre_BoxArrayDuplicate(nalu_hypre_StructGridBoxes(grid_l[0]));
   nalu_hypre_ProjectBoxArray(base_points, base_index, base_stride);

   (cyc_red_data -> base_points) = base_points;

   /*-----------------------------------------------------
    * Set up fine points
    *-----------------------------------------------------*/

   fine_points_l   = nalu_hypre_TAlloc(nalu_hypre_BoxArray *,   num_levels, NALU_HYPRE_MEMORY_HOST);

   for (l = 0; l < (num_levels - 1); l++)
   {
      nalu_hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      nalu_hypre_CycRedSetFIndex(base_index, base_stride, l, cdir, findex);
      nalu_hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      fine_points_l[l] = nalu_hypre_BoxArrayDuplicate(nalu_hypre_StructGridBoxes(grid_l[l]));
      nalu_hypre_ProjectBoxArray(fine_points_l[l], findex, stride);
   }

   fine_points_l[l] = nalu_hypre_BoxArrayDuplicate(nalu_hypre_StructGridBoxes(grid_l[l]));
   if (num_levels == 1)
   {
      nalu_hypre_ProjectBoxArray(fine_points_l[l], base_index, base_stride);
   }

   (cyc_red_data -> fine_points_l)   = fine_points_l;

   /*-----------------------------------------------------
    * Set up matrix and vector structures
    *-----------------------------------------------------*/

   A_l  = nalu_hypre_TAlloc(nalu_hypre_StructMatrix *,  num_levels, NALU_HYPRE_MEMORY_HOST);
   x_l  = nalu_hypre_TAlloc(nalu_hypre_StructVector *,  num_levels, NALU_HYPRE_MEMORY_HOST);

   A_l[0] = nalu_hypre_StructMatrixRef(A);
   x_l[0] = nalu_hypre_StructVectorRef(x);

   x_num_ghost[2 * cdir]     = 1;
   x_num_ghost[2 * cdir + 1] = 1;

   for (l = 0; l < (num_levels - 1); l++)
   {
      A_l[l + 1] = nalu_hypre_CycRedCreateCoarseOp(A_l[l], grid_l[l + 1], cdir);
      //nalu_hypre_StructMatrixInitializeShell(A_l[l+1]);
      data_size += nalu_hypre_StructMatrixDataSize(A_l[l + 1]);
      data_size_const += nalu_hypre_StructMatrixDataConstSize(A_l[l + 1]);

      x_l[l + 1] = nalu_hypre_StructVectorCreate(comm, grid_l[l + 1]);
      nalu_hypre_StructVectorSetNumGhost(x_l[l + 1], x_num_ghost);
      nalu_hypre_StructVectorInitializeShell(x_l[l + 1]);
      nalu_hypre_StructVectorSetDataSize(x_l[l + 1], &data_size, &data_size_const);
   }

   data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, data_size, memory_location);
   data_const = nalu_hypre_CTAlloc(NALU_HYPRE_Real, data_size_const, NALU_HYPRE_MEMORY_HOST);

   (cyc_red_data -> memory_location) = memory_location;
   (cyc_red_data -> data) = data;
   (cyc_red_data -> data_const) = data_const;

   for (l = 0; l < (num_levels - 1); l++)
   {
      nalu_hypre_StructMatrixInitializeData(A_l[l + 1], data, data_const);
      data += nalu_hypre_StructMatrixDataSize(A_l[l + 1]);
      data_const += nalu_hypre_StructMatrixDataConstSize(A_l[l + 1]);

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
      if (data_location != NALU_HYPRE_MEMORY_HOST)
      {
         nalu_hypre_StructVectorInitializeData(x_l[l + 1], data);
         nalu_hypre_StructVectorAssemble(x_l[l + 1]);
         data += nalu_hypre_StructVectorDataSize(x_l[l + 1]);
      }
      else
      {
         nalu_hypre_StructVectorInitializeData(x_l[l + 1], data_const);
         nalu_hypre_StructVectorAssemble(x_l[l + 1]);
         data_const += nalu_hypre_StructVectorDataSize(x_l[l + 1]);
      }
#else
      nalu_hypre_StructVectorInitializeData(x_l[l + 1], data);
      nalu_hypre_StructVectorAssemble(x_l[l + 1]);
      data += nalu_hypre_StructVectorDataSize(x_l[l + 1]);
#endif
   }

   (cyc_red_data -> A_l)  = A_l;
   (cyc_red_data -> x_l)  = x_l;

   /*-----------------------------------------------------
    * Set up coarse grid operators
    *-----------------------------------------------------*/

   for (l = 0; l < (num_levels - 1); l++)
   {
      nalu_hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      nalu_hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      nalu_hypre_CycRedSetupCoarseOp(A_l[l], A_l[l + 1], cindex, stride, cdir);
   }

   /*----------------------------------------------------------
    * Set up compute packages
    *----------------------------------------------------------*/

   down_compute_pkg_l = nalu_hypre_TAlloc(nalu_hypre_ComputePkg *,  (num_levels - 1), NALU_HYPRE_MEMORY_HOST);
   up_compute_pkg_l   = nalu_hypre_TAlloc(nalu_hypre_ComputePkg *,  (num_levels - 1), NALU_HYPRE_MEMORY_HOST);

   for (l = 0; l < (num_levels - 1); l++)
   {
      nalu_hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      nalu_hypre_CycRedSetFIndex(base_index, base_stride, l, cdir, findex);
      nalu_hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* down-cycle */
      nalu_hypre_CreateComputeInfo(grid_l[l], nalu_hypre_StructMatrixStencil(A_l[l]),
                              &compute_info);
      nalu_hypre_ComputeInfoProjectSend(compute_info, findex, stride);
      nalu_hypre_ComputeInfoProjectRecv(compute_info, findex, stride);
      nalu_hypre_ComputeInfoProjectComp(compute_info, cindex, stride);
      nalu_hypre_ComputePkgCreate(compute_info,
                             nalu_hypre_StructVectorDataSpace(x_l[l]), 1,
                             grid_l[l], &down_compute_pkg_l[l]);

      /* up-cycle */
      nalu_hypre_CreateComputeInfo(grid_l[l], nalu_hypre_StructMatrixStencil(A_l[l]),
                              &compute_info);
      nalu_hypre_ComputeInfoProjectSend(compute_info, cindex, stride);
      nalu_hypre_ComputeInfoProjectRecv(compute_info, cindex, stride);
      nalu_hypre_ComputeInfoProjectComp(compute_info, findex, stride);
      nalu_hypre_ComputePkgCreate(compute_info,
                             nalu_hypre_StructVectorDataSpace(x_l[l]), 1,
                             grid_l[l], &up_compute_pkg_l[l]);
   }

   (cyc_red_data -> down_compute_pkg_l) = down_compute_pkg_l;
   (cyc_red_data -> up_compute_pkg_l)   = up_compute_pkg_l;

   /*-----------------------------------------------------
    * Compute solve flops
    *-----------------------------------------------------*/

   flop_divisor = (nalu_hypre_IndexX(base_stride) *
                   nalu_hypre_IndexY(base_stride) *
                   nalu_hypre_IndexZ(base_stride)  );
   (cyc_red_data -> solve_flops) =
      nalu_hypre_StructVectorGlobalSize(x_l[0]) / 2 / (NALU_HYPRE_BigInt)flop_divisor;
   (cyc_red_data -> solve_flops) +=
      5 * nalu_hypre_StructVectorGlobalSize(x_l[0]) / 2 / (NALU_HYPRE_BigInt)flop_divisor;
   for (l = 1; l < (num_levels - 1); l++)
   {
      (cyc_red_data -> solve_flops) +=
         10 * nalu_hypre_StructVectorGlobalSize(x_l[l]) / 2;
   }

   if (num_levels > 1)
   {
      (cyc_red_data -> solve_flops) +=
         nalu_hypre_StructVectorGlobalSize(x_l[l]) / 2;
   }


   /*-----------------------------------------------------
    * Finalize some things
    *-----------------------------------------------------*/

#if DEBUG
   {
      char  filename[255];

      /* debugging stuff */
      for (l = 0; l < num_levels; l++)
      {
         nalu_hypre_sprintf(filename, "yout_A.%02d", l);
         nalu_hypre_StructMatrixPrint(filename, A_l[l], 0);
      }
   }
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CyclicReduction
 *
 * The solution vectors on each level are also used to store the
 * right-hand-side data.  We can do this because of the red-black
 * nature of the algorithm and the fact that the method is exact,
 * allowing one to assume initial guesses of zero on all grid levels.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CyclicReduction( void               *cyc_red_vdata,
                       nalu_hypre_StructMatrix *A,
                       nalu_hypre_StructVector *b,
                       nalu_hypre_StructVector *x             )
{
   nalu_hypre_CyclicReductionData *cyc_red_data = (nalu_hypre_CyclicReductionData *)cyc_red_vdata;

   NALU_HYPRE_Int             num_levels      = (cyc_red_data -> num_levels);
   NALU_HYPRE_Int             cdir            = (cyc_red_data -> cdir);
   nalu_hypre_IndexRef        base_index      = (cyc_red_data -> base_index);
   nalu_hypre_IndexRef        base_stride     = (cyc_red_data -> base_stride);
   nalu_hypre_BoxArray       *base_points     = (cyc_red_data -> base_points);
   nalu_hypre_BoxArray      **fine_points_l   = (cyc_red_data -> fine_points_l);
   nalu_hypre_StructMatrix  **A_l             = (cyc_red_data -> A_l);
   nalu_hypre_StructVector  **x_l             = (cyc_red_data -> x_l);
   nalu_hypre_ComputePkg    **down_compute_pkg_l = (cyc_red_data -> down_compute_pkg_l);
   nalu_hypre_ComputePkg    **up_compute_pkg_l   = (cyc_red_data -> up_compute_pkg_l);

   nalu_hypre_StructGrid     *fgrid;
   NALU_HYPRE_Int            *fgrid_ids;
   nalu_hypre_StructGrid     *cgrid;
   nalu_hypre_BoxArray       *cgrid_boxes;
   NALU_HYPRE_Int            *cgrid_ids;

   nalu_hypre_CommHandle     *comm_handle;

   nalu_hypre_BoxArrayArray  *compute_box_aa;
   nalu_hypre_BoxArray       *compute_box_a;
   nalu_hypre_Box            *compute_box;

   nalu_hypre_Box            *A_dbox;
   nalu_hypre_Box            *x_dbox;
   nalu_hypre_Box            *b_dbox;
   nalu_hypre_Box            *xc_dbox;

   NALU_HYPRE_Real           *Ap, *Awp, *Aep;
   NALU_HYPRE_Real           *xp, *xwp, *xep;
   NALU_HYPRE_Real           *bp;
   NALU_HYPRE_Real           *xcp;

   nalu_hypre_Index           cindex;
   nalu_hypre_Index           stride;

   nalu_hypre_Index           index;
   nalu_hypre_Index           loop_size;
   nalu_hypre_Index           start;
   nalu_hypre_Index           startc;
   nalu_hypre_Index           stridec;

   NALU_HYPRE_Int             compute_i, fi, ci, j, l;

   nalu_hypre_BeginTiming(cyc_red_data -> time_index);


   /*--------------------------------------------------
    * Initialize some things
    *--------------------------------------------------*/

   nalu_hypre_SetIndex3(stridec, 1, 1, 1);

   nalu_hypre_StructMatrixDestroy(A_l[0]);
   nalu_hypre_StructVectorDestroy(x_l[0]);
   A_l[0] = nalu_hypre_StructMatrixRef(A);
   x_l[0] = nalu_hypre_StructVectorRef(x);

   /*--------------------------------------------------
    * Copy b into x
    *--------------------------------------------------*/

   compute_box_a = base_points;
   nalu_hypre_ForBoxI(fi, compute_box_a)
   {
      compute_box = nalu_hypre_BoxArrayBox(compute_box_a, fi);

      x_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), fi);
      b_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(b), fi);

      xp = nalu_hypre_StructVectorBoxData(x, fi);
      bp = nalu_hypre_StructVectorBoxData(b, fi);

      nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(compute_box), start);
      nalu_hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);

#define DEVICE_VAR is_device_ptr(xp,bp)
      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                          x_dbox, start, base_stride, xi,
                          b_dbox, start, base_stride, bi);
      {
         xp[xi] = bp[bi];
      }
      nalu_hypre_BoxLoop2End(xi, bi);
#undef DEVICE_VAR
   }

   /*--------------------------------------------------
    * Down cycle:
    *
    * 1) Do an F-relaxation sweep with zero initial guess
    * 2) Compute and inject residual at C-points
    *    - computations are at C-points
    *    - communications are at F-points
    *
    * Notes:
    * - Before these two steps are executed, the
    * fine-grid solution vector contains the right-hand-side.
    * - After these two steps are executed, the fine-grid
    * solution vector contains the right-hand side at
    * C-points and the current solution approximation at
    * F-points.  The coarse-grid solution vector contains
    * the restricted (injected) fine-grid residual.
    *--------------------------------------------------*/

   for (l = 0; l < num_levels - 1 ; l++)
   {
      /* set cindex and stride */
      nalu_hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      nalu_hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* Step 1 */
      compute_box_a = fine_points_l[l];
      nalu_hypre_ForBoxI(fi, compute_box_a)
      {
         compute_box = nalu_hypre_BoxArrayBox(compute_box_a, fi);

         A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A_l[l]), fi);
         x_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x_l[l]), fi);

         nalu_hypre_SetIndex3(index, 0, 0, 0);
         Ap = nalu_hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
         xp = nalu_hypre_StructVectorBoxData(x_l[l], fi);

         nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(compute_box), start);
         nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(xp,Ap)
         nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                             A_dbox, start, stride, Ai,
                             x_dbox, start, stride, xi);
         {
            xp[xi] /= Ap[Ai];
         }
         nalu_hypre_BoxLoop2End(Ai, xi);
#undef DEVICE_VAR
      }

      /* Step 2 */
      fgrid = nalu_hypre_StructVectorGrid(x_l[l]);
      fgrid_ids = nalu_hypre_StructGridIDs(fgrid);
      cgrid = nalu_hypre_StructVectorGrid(x_l[l + 1]);
      cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);
      cgrid_ids = nalu_hypre_StructGridIDs(cgrid);

      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
            {
               xp = nalu_hypre_StructVectorData(x_l[l]);
               nalu_hypre_InitializeIndtComputations(down_compute_pkg_l[l], xp,
                                                &comm_handle);
               compute_box_aa = nalu_hypre_ComputePkgIndtBoxes(down_compute_pkg_l[l]);
            }
            break;

            case 1:
            {
               nalu_hypre_FinalizeIndtComputations(comm_handle);
               compute_box_aa = nalu_hypre_ComputePkgDeptBoxes(down_compute_pkg_l[l]);
            }
            break;
         }

         fi = 0;
         nalu_hypre_ForBoxI(ci, cgrid_boxes)
         {
            while (fgrid_ids[fi] != cgrid_ids[ci])
            {
               fi++;
            }

            compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, fi);

            A_dbox  = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A_l[l]), fi);
            x_dbox  = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x_l[l]), fi);
            xc_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x_l[l + 1]), ci);

            xp  = nalu_hypre_StructVectorBoxData(x_l[l], fi);
            xcp = nalu_hypre_StructVectorBoxData(x_l[l + 1], ci);

            nalu_hypre_SetIndex3(index, 0, 0, 0);
            nalu_hypre_IndexD(index, cdir) = -1;
            Awp = nalu_hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
            xwp = nalu_hypre_StructVectorBoxData(x_l[l], fi);
            //RL:PTR_OFFSET
            NALU_HYPRE_Int xwp_offset = nalu_hypre_BoxOffsetDistance(x_dbox, index);

            nalu_hypre_SetIndex3(index, 0, 0, 0);
            nalu_hypre_IndexD(index, cdir) = 1;
            Aep = nalu_hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
            xep = nalu_hypre_StructVectorBoxData(x_l[l], fi);
            NALU_HYPRE_Int xep_offset = nalu_hypre_BoxOffsetDistance(x_dbox, index);

            nalu_hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

               nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(compute_box), start);
               nalu_hypre_StructMapFineToCoarse(start, cindex, stride, startc);

               nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(xcp,xp,Awp,xwp,Aep,xep)
               nalu_hypre_BoxLoop3Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                                   A_dbox, start, stride, Ai,
                                   x_dbox, start, stride, xi,
                                   xc_dbox, startc, stridec, xci);
               {
                  xcp[xci] = xp[xi] - Awp[Ai] * xwp[xi + xwp_offset] -
                             Aep[Ai] * xep[xi + xep_offset];
               }
               nalu_hypre_BoxLoop3End(Ai, xi, xci);
#undef DEVICE_VAR
            }
         }
      }
   }
   /*--------------------------------------------------
    * Coarsest grid:
    *
    * Do an F-relaxation sweep with zero initial guess
    *
    * This is the same as step 1 in above, but is
    * broken out as a sepecial case to add a check
    * for zero diagonal that can occur for singlar
    * problems like the full Neumann problem.
    *--------------------------------------------------*/
   /* set cindex and stride */
   nalu_hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
   nalu_hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

   compute_box_a = fine_points_l[l];
   nalu_hypre_ForBoxI(fi, compute_box_a)
   {
      compute_box = nalu_hypre_BoxArrayBox(compute_box_a, fi);

      A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A_l[l]), fi);
      x_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x_l[l]), fi);

      nalu_hypre_SetIndex3(index, 0, 0, 0);
      Ap = nalu_hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
      xp = nalu_hypre_StructVectorBoxData(x_l[l], fi);

      nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(compute_box), start);
      nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(xp,Ap)
      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                          A_dbox, start, stride, Ai,
                          x_dbox, start, stride, xi);
      {
         if (Ap[Ai] != 0.0)
         {
            xp[xi] /= Ap[Ai];
         }
      }
      nalu_hypre_BoxLoop2End(Ai, xi);
#undef DEVICE_VAR
   }

   /*--------------------------------------------------
    * Up cycle:
    *
    * 1) Inject coarse error into fine-grid solution
    *    vector (this is the solution at the C-points)
    * 2) Do an F-relaxation sweep on Ax = 0 and update
    *    solution at F-points
    *    - computations are at F-points
    *    - communications are at C-points
    *--------------------------------------------------*/

   for (l = (num_levels - 2); l >= 0; l--)
   {
      /* set cindex and stride */
      nalu_hypre_CycRedSetCIndex(base_index, base_stride, l, cdir, cindex);
      nalu_hypre_CycRedSetStride(base_index, base_stride, l, cdir, stride);

      /* Step 1 */
      fgrid = nalu_hypre_StructVectorGrid(x_l[l]);
      fgrid_ids = nalu_hypre_StructGridIDs(fgrid);
      cgrid = nalu_hypre_StructVectorGrid(x_l[l + 1]);
      cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);
      cgrid_ids = nalu_hypre_StructGridIDs(cgrid);

      fi = 0;
      nalu_hypre_ForBoxI(ci, cgrid_boxes)
      {
         while (fgrid_ids[fi] != cgrid_ids[ci])
         {
            fi++;
         }

         compute_box = nalu_hypre_BoxArrayBox(cgrid_boxes, ci);

         nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(compute_box), startc);
         nalu_hypre_StructMapCoarseToFine(startc, cindex, stride, start);

         x_dbox  = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x_l[l]), fi);
         xc_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x_l[l + 1]), ci);

         xp  = nalu_hypre_StructVectorBoxData(x_l[l], fi);
         xcp = nalu_hypre_StructVectorBoxData(x_l[l + 1], ci);

         nalu_hypre_BoxGetSize(compute_box, loop_size);

#define DEVICE_VAR is_device_ptr(xp,xcp)
         nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                             x_dbox, start, stride, xi,
                             xc_dbox, startc, stridec, xci);
         {
            xp[xi] = xcp[xci];
         }
         nalu_hypre_BoxLoop2End(xi, xci);
#undef DEVICE_VAR
      }

      /* Step 2 */
      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
            {
               xp = nalu_hypre_StructVectorData(x_l[l]);
               nalu_hypre_InitializeIndtComputations(up_compute_pkg_l[l], xp,
                                                &comm_handle);
               compute_box_aa = nalu_hypre_ComputePkgIndtBoxes(up_compute_pkg_l[l]);
            }
            break;

            case 1:
            {
               nalu_hypre_FinalizeIndtComputations(comm_handle);
               compute_box_aa = nalu_hypre_ComputePkgDeptBoxes(up_compute_pkg_l[l]);
            }
            break;
         }

         nalu_hypre_ForBoxArrayI(fi, compute_box_aa)
         {
            compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, fi);

            A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A_l[l]), fi);
            x_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x_l[l]), fi);

            nalu_hypre_SetIndex3(index, 0, 0, 0);
            Ap = nalu_hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
            xp = nalu_hypre_StructVectorBoxData(x_l[l], fi);

            nalu_hypre_SetIndex3(index, 0, 0, 0);
            nalu_hypre_IndexD(index, cdir) = -1;
            Awp = nalu_hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
            //RL PTROFFSET
            xwp = nalu_hypre_StructVectorBoxData(x_l[l], fi);
            NALU_HYPRE_Int xwp_offset = nalu_hypre_BoxOffsetDistance(x_dbox, index);

            nalu_hypre_SetIndex3(index, 0, 0, 0);
            nalu_hypre_IndexD(index, cdir) = 1;
            Aep = nalu_hypre_StructMatrixExtractPointerByIndex(A_l[l], fi, index);
            xep = nalu_hypre_StructVectorBoxData(x_l[l], fi);
            NALU_HYPRE_Int xep_offset = nalu_hypre_BoxOffsetDistance(x_dbox, index);

            nalu_hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

               nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(compute_box), start);
               nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(xp,Awp,Aep,Ap)
               nalu_hypre_BoxLoop2Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                                   A_dbox, start, stride, Ai,
                                   x_dbox, start, stride, xi);
               {
                  xp[xi] -= (Awp[Ai] * xp[xi + xwp_offset] + Aep[Ai] * xp[xi + xep_offset]) / Ap[Ai];
               }
               nalu_hypre_BoxLoop2End(Ai, xi);
#undef DEVICE_VAR
            }
         }
      }
   }

   /*-----------------------------------------------------
    * Finalize some things
    *-----------------------------------------------------*/

   nalu_hypre_IncFLOPCount(cyc_red_data -> solve_flops);
   nalu_hypre_EndTiming(cyc_red_data -> time_index);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CyclicReductionSetBase
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CyclicReductionSetBase( void        *cyc_red_vdata,
                              nalu_hypre_Index  base_index,
                              nalu_hypre_Index  base_stride )
{
   nalu_hypre_CyclicReductionData *cyc_red_data = (nalu_hypre_CyclicReductionData *)cyc_red_vdata;
   NALU_HYPRE_Int                d;

   for (d = 0; d < 3; d++)
   {
      nalu_hypre_IndexD((cyc_red_data -> base_index),  d) =
         nalu_hypre_IndexD(base_index,  d);
      nalu_hypre_IndexD((cyc_red_data -> base_stride), d) =
         nalu_hypre_IndexD(base_stride, d);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CyclicReductionSetCDir
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CyclicReductionSetCDir( void        *cyc_red_vdata,
                              NALU_HYPRE_Int    cdir )
{
   nalu_hypre_CyclicReductionData *cyc_red_data = (nalu_hypre_CyclicReductionData *)cyc_red_vdata;

   (cyc_red_data -> cdir) = cdir;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CyclicReductionDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CyclicReductionDestroy( void *cyc_red_vdata )
{
   nalu_hypre_CyclicReductionData *cyc_red_data = (nalu_hypre_CyclicReductionData *)cyc_red_vdata;

   NALU_HYPRE_Int l;

   if (cyc_red_data)
   {
      NALU_HYPRE_MemoryLocation memory_location = cyc_red_data -> memory_location;

      nalu_hypre_BoxArrayDestroy(cyc_red_data -> base_points);
      nalu_hypre_StructGridDestroy(cyc_red_data -> grid_l[0]);
      nalu_hypre_StructMatrixDestroy(cyc_red_data -> A_l[0]);
      nalu_hypre_StructVectorDestroy(cyc_red_data -> x_l[0]);
      for (l = 0; l < ((cyc_red_data -> num_levels) - 1); l++)
      {
         nalu_hypre_StructGridDestroy(cyc_red_data -> grid_l[l + 1]);
         nalu_hypre_BoxArrayDestroy(cyc_red_data -> fine_points_l[l]);
         nalu_hypre_StructMatrixDestroy(cyc_red_data -> A_l[l + 1]);
         nalu_hypre_StructVectorDestroy(cyc_red_data -> x_l[l + 1]);
         nalu_hypre_ComputePkgDestroy(cyc_red_data -> down_compute_pkg_l[l]);
         nalu_hypre_ComputePkgDestroy(cyc_red_data -> up_compute_pkg_l[l]);
      }
      nalu_hypre_BoxArrayDestroy(cyc_red_data -> fine_points_l[l]);
      nalu_hypre_TFree(cyc_red_data -> data, memory_location);
      nalu_hypre_TFree(cyc_red_data -> grid_l, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(cyc_red_data -> fine_points_l, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(cyc_red_data -> A_l, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(cyc_red_data -> x_l, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(cyc_red_data -> down_compute_pkg_l, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(cyc_red_data -> up_compute_pkg_l, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_FinalizeTiming(cyc_red_data -> time_index);
      nalu_hypre_TFree(cyc_red_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CyclicReductionDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CyclicReductionSetMaxLevel( void   *cyc_red_vdata,
                                  NALU_HYPRE_Int   max_level  )
{
   nalu_hypre_CyclicReductionData *cyc_red_data = (nalu_hypre_CyclicReductionData *)cyc_red_vdata;
   (cyc_red_data -> max_levels) = max_level;

   return nalu_hypre_error_flag;
}
