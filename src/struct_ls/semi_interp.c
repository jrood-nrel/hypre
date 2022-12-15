/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   nalu_hypre_StructMatrix *P;
   NALU_HYPRE_Int           P_stored_as_transpose;
   nalu_hypre_ComputePkg   *compute_pkg;
   nalu_hypre_Index         cindex;
   nalu_hypre_Index         findex;
   nalu_hypre_Index         stride;

   NALU_HYPRE_Int           time_index;

} nalu_hypre_SemiInterpData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_SemiInterpCreate( )
{
   nalu_hypre_SemiInterpData *interp_data;

   interp_data = nalu_hypre_CTAlloc(nalu_hypre_SemiInterpData,  1, NALU_HYPRE_MEMORY_HOST);
   (interp_data -> time_index)  = nalu_hypre_InitializeTiming("SemiInterp");

   return (void *) interp_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SemiInterpSetup( void               *interp_vdata,
                       nalu_hypre_StructMatrix *P,
                       NALU_HYPRE_Int           P_stored_as_transpose,
                       nalu_hypre_StructVector *xc,
                       nalu_hypre_StructVector *e,
                       nalu_hypre_Index         cindex,
                       nalu_hypre_Index         findex,
                       nalu_hypre_Index         stride       )
{
   nalu_hypre_SemiInterpData   *interp_data = (nalu_hypre_SemiInterpData   *)interp_vdata;

   nalu_hypre_StructGrid       *grid;
   nalu_hypre_StructStencil    *stencil;

   nalu_hypre_ComputeInfo      *compute_info;
   nalu_hypre_ComputePkg       *compute_pkg;

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
   (interp_data -> P_stored_as_transpose) = P_stored_as_transpose;
   (interp_data -> compute_pkg) = compute_pkg;
   nalu_hypre_CopyIndex(cindex, (interp_data -> cindex));
   nalu_hypre_CopyIndex(findex, (interp_data -> findex));
   nalu_hypre_CopyIndex(stride, (interp_data -> stride));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SemiInterp( void               *interp_vdata,
                  nalu_hypre_StructMatrix *P,
                  nalu_hypre_StructVector *xc,
                  nalu_hypre_StructVector *e            )
{
   nalu_hypre_SemiInterpData   *interp_data = (nalu_hypre_SemiInterpData   *)interp_vdata;

   NALU_HYPRE_Int               P_stored_as_transpose;
   nalu_hypre_ComputePkg       *compute_pkg;
   nalu_hypre_IndexRef          cindex;
   nalu_hypre_IndexRef          findex;
   nalu_hypre_IndexRef          stride;

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

   NALU_HYPRE_Int               Pi;
   NALU_HYPRE_Int               constant_coefficient;

   NALU_HYPRE_Real             *Pp0, *Pp1;
   NALU_HYPRE_Real             *xcp;
   NALU_HYPRE_Real             *ep;

   nalu_hypre_Index             loop_size;
   nalu_hypre_Index             start;
   nalu_hypre_Index             startc;
   nalu_hypre_Index             stridec;

   nalu_hypre_StructStencil    *stencil;
   nalu_hypre_Index            *stencil_shape;

   NALU_HYPRE_Int               compute_i, fi, ci, j;
   nalu_hypre_StructVector     *xc_tmp;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   nalu_hypre_BeginTiming(interp_data -> time_index);

   P_stored_as_transpose = (interp_data -> P_stored_as_transpose);
   compute_pkg   = (interp_data -> compute_pkg);
   cindex        = (interp_data -> cindex);
   findex        = (interp_data -> findex);
   stride        = (interp_data -> stride);

   stencil       = nalu_hypre_StructMatrixStencil(P);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(P);
   nalu_hypre_assert( constant_coefficient == 0 || constant_coefficient == 1 );
   /* ... constant_coefficient==2 for P shouldn't happen, see
      nalu_hypre_PFMGCreateInterpOp in pfmg_setup_interp.c */

   if (constant_coefficient) { nalu_hypre_StructVectorClearBoundGhostValues(e, 0); }

   nalu_hypre_SetIndex3(stridec, 1, 1, 1);

   /*-----------------------------------------------------------------------
    * Compute e at coarse points (injection)
    *-----------------------------------------------------------------------*/

   fgrid = nalu_hypre_StructVectorGrid(e);
   fgrid_ids = nalu_hypre_StructGridIDs(fgrid);
   cgrid = nalu_hypre_StructVectorGrid(xc);
   cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);
   cgrid_ids = nalu_hypre_StructGridIDs(cgrid);

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_MemoryLocation data_location_f = nalu_hypre_StructGridDataLocation(fgrid);
   NALU_HYPRE_MemoryLocation data_location_c = nalu_hypre_StructGridDataLocation(cgrid);

   if (data_location_f != data_location_c)
   {
      xc_tmp = nalu_hypre_StructVectorCreate(nalu_hypre_MPI_COMM_WORLD, cgrid);
      nalu_hypre_StructVectorSetNumGhost(xc_tmp, nalu_hypre_StructVectorNumGhost(xc));
      nalu_hypre_StructGridDataLocation(cgrid) = data_location_f;
      nalu_hypre_StructVectorInitialize(xc_tmp);
      nalu_hypre_StructVectorAssemble(xc_tmp);
      nalu_hypre_TMemcpy(nalu_hypre_StructVectorData(xc_tmp), nalu_hypre_StructVectorData(xc), NALU_HYPRE_Complex,
                    nalu_hypre_StructVectorDataSize(xc), NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      xc_tmp = xc;
   }
#else
   xc_tmp = xc;
#endif
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
      xcp = nalu_hypre_StructVectorBoxData(xc_tmp, ci);

      nalu_hypre_BoxGetSize(compute_box, loop_size);

#define DEVICE_VAR is_device_ptr(ep,xcp)
      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(P), loop_size,
                          e_dbox, start, stride, ei,
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

         //RL:PTROFFSET
         NALU_HYPRE_Int Pp1_offset = 0, ep0_offset, ep1_offset;
         if (P_stored_as_transpose)
         {
            if ( constant_coefficient )
            {
               Pp0 = nalu_hypre_StructMatrixBoxData(P, fi, 1);
               Pp1 = nalu_hypre_StructMatrixBoxData(P, fi, 0);
               Pp1_offset = -nalu_hypre_CCBoxOffsetDistance(P_dbox, stencil_shape[0]);
            }
            else
            {
               Pp0 = nalu_hypre_StructMatrixBoxData(P, fi, 1);
               Pp1 = nalu_hypre_StructMatrixBoxData(P, fi, 0);
               Pp1_offset = -nalu_hypre_BoxOffsetDistance(P_dbox, stencil_shape[0]);
            }
         }
         else
         {
            Pp0 = nalu_hypre_StructMatrixBoxData(P, fi, 0);
            Pp1 = nalu_hypre_StructMatrixBoxData(P, fi, 1);
         }
         ep  = nalu_hypre_StructVectorBoxData(e, fi);
         ep0_offset = nalu_hypre_BoxOffsetDistance(e_dbox, stencil_shape[0]);
         ep1_offset = nalu_hypre_BoxOffsetDistance(e_dbox, stencil_shape[1]);

         nalu_hypre_ForBoxI(j, compute_box_a)
         {
            compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

            nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(compute_box), start);
            nalu_hypre_StructMapFineToCoarse(start, findex, stride, startc);

            nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

            if ( constant_coefficient )
            {
               NALU_HYPRE_Complex Pp0val, Pp1val;
               Pi = nalu_hypre_CCBoxIndexRank( P_dbox, startc );
               Pp0val = Pp0[Pi];
               Pp1val = Pp1[Pi + Pp1_offset];

#define DEVICE_VAR is_device_ptr(ep)
               nalu_hypre_BoxLoop1Begin(nalu_hypre_StructMatrixNDim(P), loop_size,
                                   e_dbox, start, stride, ei);
               {
                  ep[ei] =  (Pp0val * ep[ei + ep0_offset] +
                             Pp1val * ep[ei + ep1_offset]);
               }
               nalu_hypre_BoxLoop1End(ei);
#undef DEVICE_VAR
            }
            else
            {
#define DEVICE_VAR is_device_ptr(ep,Pp0,Pp1)
               nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(P), loop_size,
                                   P_dbox, startc, stridec, Pi,
                                   e_dbox, start, stride, ei);
               {
                  ep[ei] =  (Pp0[Pi]            * ep[ei + ep0_offset] +
                             Pp1[Pi + Pp1_offset] * ep[ei + ep1_offset]);
               }
               nalu_hypre_BoxLoop2End(Pi, ei);
#undef DEVICE_VAR
            }
         }
      }
   }
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   if (data_location_f != data_location_c)
   {
      nalu_hypre_StructVectorDestroy(xc_tmp);
      nalu_hypre_StructGridDataLocation(cgrid) = data_location_c;
   }
#endif
   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   nalu_hypre_IncFLOPCount(3 * nalu_hypre_StructVectorGlobalSize(xc));
   nalu_hypre_EndTiming(interp_data -> time_index);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SemiInterpDestroy( void *interp_vdata )
{
   nalu_hypre_SemiInterpData *interp_data = (nalu_hypre_SemiInterpData   *)interp_vdata;

   if (interp_data)
   {
      nalu_hypre_StructMatrixDestroy(interp_data -> P);
      nalu_hypre_ComputePkgDestroy(interp_data -> compute_pkg);
      nalu_hypre_FinalizeTiming(interp_data -> time_index);
      nalu_hypre_TFree(interp_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

