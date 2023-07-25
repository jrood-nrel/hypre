/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGInterpData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   nalu_hypre_StructMatrix *P;
   nalu_hypre_ComputePkg   *compute_pkg;
   nalu_hypre_Index         cindex;
   nalu_hypre_Index         findex;
   nalu_hypre_Index         stride;
   nalu_hypre_Index         strideP;

   NALU_HYPRE_Int           time_index;

} nalu_hypre_SparseMSGInterpData;

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGInterpCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_SparseMSGInterpCreate( void )
{
   nalu_hypre_SparseMSGInterpData *interp_data;

   interp_data = nalu_hypre_CTAlloc(nalu_hypre_SparseMSGInterpData,  1, NALU_HYPRE_MEMORY_HOST);
   (interp_data -> time_index)  = nalu_hypre_InitializeTiming("SparseMSGInterp");

   return (void *) interp_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGInterpSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGInterpSetup( void               *interp_vdata,
                            nalu_hypre_StructMatrix *P,
                            nalu_hypre_StructVector *xc,
                            nalu_hypre_StructVector *e,
                            nalu_hypre_Index         cindex,
                            nalu_hypre_Index         findex,
                            nalu_hypre_Index         stride,
                            nalu_hypre_Index         strideP       )
{
   nalu_hypre_SparseMSGInterpData   *interp_data = (nalu_hypre_SparseMSGInterpData   *)interp_vdata;

   nalu_hypre_StructGrid       *grid;
   nalu_hypre_StructStencil    *stencil;

   nalu_hypre_ComputeInfo      *compute_info;
   nalu_hypre_ComputePkg       *compute_pkg;

   NALU_HYPRE_Int               ierr = 0;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = nalu_hypre_StructVectorGrid(e);
   stencil = nalu_hypre_StructMatrixStencil(P);

   nalu_hypre_CreateComputeInfo(grid, stencil, &compute_info);
   nalu_hypre_ComputeInfoProjectSend(compute_info, cindex, stride);
   nalu_hypre_ComputeInfoProjectRecv(compute_info, cindex, stride);
   nalu_hypre_ComputeInfoProjectComp(compute_info, findex, stride);
   nalu_hypre_ComputePkgCreate(compute_info, nalu_hypre_StructVectorDataSpace(e), 1,
                          grid, &compute_pkg);

   /*----------------------------------------------------------
    * Set up the interp data structure
    *----------------------------------------------------------*/

   (interp_data -> P) = nalu_hypre_StructMatrixRef(P);
   (interp_data -> compute_pkg) = compute_pkg;
   nalu_hypre_CopyIndex(cindex, (interp_data -> cindex));
   nalu_hypre_CopyIndex(findex, (interp_data -> findex));
   nalu_hypre_CopyIndex(stride, (interp_data -> stride));
   nalu_hypre_CopyIndex(strideP, (interp_data -> strideP));

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGInterp:
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGInterp( void               *interp_vdata,
                       nalu_hypre_StructMatrix *P,
                       nalu_hypre_StructVector *xc,
                       nalu_hypre_StructVector *e            )
{
   NALU_HYPRE_Int ierr = 0;

   nalu_hypre_SparseMSGInterpData   *interp_data = (nalu_hypre_SparseMSGInterpData   *)interp_vdata;

   nalu_hypre_ComputePkg       *compute_pkg;
   nalu_hypre_IndexRef          cindex;
   nalu_hypre_IndexRef          findex;
   nalu_hypre_IndexRef          stride;
   nalu_hypre_IndexRef          strideP;

   nalu_hypre_StructGrid       *fgrid;
   NALU_HYPRE_Int              *fgrid_ids;
   nalu_hypre_StructGrid       *cgrid;
   nalu_hypre_BoxArray         *cgrid_boxes;
   NALU_HYPRE_Int              *cgrid_ids;

   nalu_hypre_CommHandle       *comm_handle;

   nalu_hypre_BoxArrayArray    *compute_box_aa;
   nalu_hypre_BoxArray         *compute_box_a;
   nalu_hypre_Box              *compute_box;

   nalu_hypre_Box              *P_dbox;
   nalu_hypre_Box              *xc_dbox;
   nalu_hypre_Box              *e_dbox;

   NALU_HYPRE_Real             *Pp0, *Pp1;
   NALU_HYPRE_Real             *xcp;
   NALU_HYPRE_Real             *ep, *ep0, *ep1;

   nalu_hypre_Index             loop_size;
   nalu_hypre_Index             start;
   nalu_hypre_Index             startc;
   nalu_hypre_Index             startP;
   nalu_hypre_Index             stridec;

   nalu_hypre_StructStencil    *stencil;
   nalu_hypre_Index            *stencil_shape;

   NALU_HYPRE_Int               compute_i, fi, ci, j;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   nalu_hypre_BeginTiming(interp_data -> time_index);

   compute_pkg   = (interp_data -> compute_pkg);
   cindex        = (interp_data -> cindex);
   findex        = (interp_data -> findex);
   stride        = (interp_data -> stride);
   strideP       = (interp_data -> strideP);

   stencil       = nalu_hypre_StructMatrixStencil(P);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);

   nalu_hypre_SetIndex3(stridec, 1, 1, 1);

   /*-----------------------------------------------------------------------
    * Compute e at coarse points (injection)
    *-----------------------------------------------------------------------*/

   fgrid = nalu_hypre_StructVectorGrid(e);
   fgrid_ids = nalu_hypre_StructGridIDs(fgrid);
   cgrid = nalu_hypre_StructVectorGrid(xc);
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

      e_dbox  = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(e), fi);
      xc_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(xc), ci);

      ep  = nalu_hypre_StructVectorBoxData(e, fi);
      xcp = nalu_hypre_StructVectorBoxData(xc, ci);

      nalu_hypre_BoxGetSize(compute_box, loop_size);

#define DEVICE_VAR is_device_ptr(ep,xcp)
      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(P), loop_size,
                          e_dbox,  start,  stride,  ei,
                          xc_dbox, startc, stridec, xci);
      {
         ep[ei] = xcp[xci];
      }
      nalu_hypre_BoxLoop2End(ei, xci);
#undef DEVICE_VAR
   }

   /*-----------------------------------------------------------------------
    * Compute e at fine points
    *-----------------------------------------------------------------------*/

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch (compute_i)
      {
         case 0:
         {
            ep = nalu_hypre_StructVectorData(e);
            nalu_hypre_InitializeIndtComputations(compute_pkg, ep, &comm_handle);
            compute_box_aa = nalu_hypre_ComputePkgIndtBoxes(compute_pkg);
         }
         break;

         case 1:
         {
            nalu_hypre_FinalizeIndtComputations(comm_handle);
            compute_box_aa = nalu_hypre_ComputePkgDeptBoxes(compute_pkg);
         }
         break;
      }

      nalu_hypre_ForBoxArrayI(fi, compute_box_aa)
      {
         compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, fi);

         P_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(P), fi);
         e_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(e), fi);

         Pp0 = nalu_hypre_StructMatrixBoxData(P, fi, 0);
         Pp1 = nalu_hypre_StructMatrixBoxData(P, fi, 1);
         ep  = nalu_hypre_StructVectorBoxData(e, fi);
         ep0 = ep + nalu_hypre_BoxOffsetDistance(e_dbox, stencil_shape[0]);
         ep1 = ep + nalu_hypre_BoxOffsetDistance(e_dbox, stencil_shape[1]);

         nalu_hypre_ForBoxI(j, compute_box_a)
         {
            compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(compute_box), start);
            nalu_hypre_StructMapFineToCoarse(start,  findex, stride,  startc);
            nalu_hypre_StructMapCoarseToFine(startc, cindex, strideP, startP);

            nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(ep,Pp0,ep0,Pp1,ep1)
            nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(P), loop_size,
                                P_dbox, startP, strideP, Pi,
                                e_dbox, start,  stride,  ei);
            {
               ep[ei] =  (Pp0[Pi] * ep0[ei] +
                          Pp1[Pi] * ep1[ei]);
            }
            nalu_hypre_BoxLoop2End(Pi, ei);
#undef DEVICE_VAR
         }
      }
   }

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   nalu_hypre_IncFLOPCount(3 * nalu_hypre_StructVectorGlobalSize(xc));
   nalu_hypre_EndTiming(interp_data -> time_index);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGInterpDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGInterpDestroy( void *interp_vdata )
{
   NALU_HYPRE_Int ierr = 0;

   nalu_hypre_SparseMSGInterpData *interp_data = (nalu_hypre_SparseMSGInterpData   *)interp_vdata;

   if (interp_data)
   {
      nalu_hypre_StructMatrixDestroy(interp_data -> P);
      nalu_hypre_ComputePkgDestroy(interp_data -> compute_pkg);
      nalu_hypre_FinalizeTiming(interp_data -> time_index);
      nalu_hypre_TFree(interp_data, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}

