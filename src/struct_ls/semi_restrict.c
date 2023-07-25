/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

typedef struct
{
   nalu_hypre_StructMatrix *R;
   NALU_HYPRE_Int           R_stored_as_transpose;
   nalu_hypre_ComputePkg   *compute_pkg;
   nalu_hypre_Index         cindex;
   nalu_hypre_Index         stride;

   NALU_HYPRE_Int           time_index;

} nalu_hypre_SemiRestrictData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_SemiRestrictCreate( void )
{
   nalu_hypre_SemiRestrictData *restrict_data;

   restrict_data = nalu_hypre_CTAlloc(nalu_hypre_SemiRestrictData,  1, NALU_HYPRE_MEMORY_HOST);

   (restrict_data -> time_index)  = nalu_hypre_InitializeTiming("SemiRestrict");

   return (void *) restrict_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SemiRestrictSetup( void               *restrict_vdata,
                         nalu_hypre_StructMatrix *R,
                         NALU_HYPRE_Int           R_stored_as_transpose,
                         nalu_hypre_StructVector *r,
                         nalu_hypre_StructVector *rc,
                         nalu_hypre_Index         cindex,
                         nalu_hypre_Index         findex,
                         nalu_hypre_Index         stride                )
{
   nalu_hypre_SemiRestrictData *restrict_data = (nalu_hypre_SemiRestrictData *)restrict_vdata;

   nalu_hypre_StructGrid       *grid;
   nalu_hypre_StructStencil    *stencil;

   nalu_hypre_ComputeInfo      *compute_info;
   nalu_hypre_ComputePkg       *compute_pkg;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = nalu_hypre_StructVectorGrid(r);
   stencil = nalu_hypre_StructMatrixStencil(R);

   nalu_hypre_CreateComputeInfo(grid, stencil, &compute_info);
   nalu_hypre_ComputeInfoProjectSend(compute_info, findex, stride);
   nalu_hypre_ComputeInfoProjectRecv(compute_info, findex, stride);
   nalu_hypre_ComputeInfoProjectComp(compute_info, cindex, stride);
   nalu_hypre_ComputePkgCreate(compute_info, nalu_hypre_StructVectorDataSpace(r), 1,
                          grid, &compute_pkg);

   /*----------------------------------------------------------
    * Set up the restrict data structure
    *----------------------------------------------------------*/

   (restrict_data -> R) = nalu_hypre_StructMatrixRef(R);
   (restrict_data -> R_stored_as_transpose) = R_stored_as_transpose;
   (restrict_data -> compute_pkg) = compute_pkg;
   nalu_hypre_CopyIndex(cindex, (restrict_data -> cindex));
   nalu_hypre_CopyIndex(stride, (restrict_data -> stride));

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SemiRestrict( void               *restrict_vdata,
                    nalu_hypre_StructMatrix *R,
                    nalu_hypre_StructVector *r,
                    nalu_hypre_StructVector *rc             )
{
   nalu_hypre_SemiRestrictData *restrict_data = (nalu_hypre_SemiRestrictData *)restrict_vdata;

   NALU_HYPRE_Int               R_stored_as_transpose;
   nalu_hypre_ComputePkg       *compute_pkg;
   nalu_hypre_IndexRef          cindex;
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

   nalu_hypre_Box              *R_dbox;
   nalu_hypre_Box              *r_dbox;
   nalu_hypre_Box              *rc_dbox;

   NALU_HYPRE_Int               Ri;
   NALU_HYPRE_Int               constant_coefficient;

   NALU_HYPRE_Real             *Rp0, *Rp1;
   NALU_HYPRE_Real             *rp;
   NALU_HYPRE_Real             *rcp;

   nalu_hypre_Index             loop_size;
   nalu_hypre_IndexRef          start;
   nalu_hypre_Index             startc;
   nalu_hypre_Index             stridec;

   nalu_hypre_StructStencil    *stencil;
   nalu_hypre_Index            *stencil_shape;

   NALU_HYPRE_Int               compute_i, fi, ci, j;
   nalu_hypre_StructVector     *rc_tmp;
   /*-----------------------------------------------------------------------
    * Initialize some things.
    *-----------------------------------------------------------------------*/

   nalu_hypre_BeginTiming(restrict_data -> time_index);

   R_stored_as_transpose = (restrict_data -> R_stored_as_transpose);
   compute_pkg   = (restrict_data -> compute_pkg);
   cindex        = (restrict_data -> cindex);
   stride        = (restrict_data -> stride);

   stencil       = nalu_hypre_StructMatrixStencil(R);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(R);
   nalu_hypre_assert( constant_coefficient == 0 || constant_coefficient == 1 );
   /* ... if A has constant_coefficient==2, R has constant_coefficient==0 */

   if (constant_coefficient) { nalu_hypre_StructVectorClearBoundGhostValues(r, 0); }

   nalu_hypre_SetIndex3(stridec, 1, 1, 1);

   /*--------------------------------------------------------------------
    * Restrict the residual.
    *--------------------------------------------------------------------*/

   fgrid = nalu_hypre_StructVectorGrid(r);
   fgrid_ids = nalu_hypre_StructGridIDs(fgrid);
   cgrid = nalu_hypre_StructVectorGrid(rc);
   cgrid_boxes = nalu_hypre_StructGridBoxes(cgrid);
   cgrid_ids = nalu_hypre_StructGridIDs(cgrid);

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_MemoryLocation data_location_f = nalu_hypre_StructGridDataLocation(fgrid);
   NALU_HYPRE_MemoryLocation data_location_c = nalu_hypre_StructGridDataLocation(cgrid);

   if (data_location_f != data_location_c)
   {
      rc_tmp = nalu_hypre_StructVectorCreate(nalu_hypre_MPI_COMM_WORLD, cgrid);
      nalu_hypre_StructVectorSetNumGhost(rc_tmp, nalu_hypre_StructVectorNumGhost(rc));
      nalu_hypre_StructGridDataLocation(cgrid) = data_location_f;
      nalu_hypre_StructVectorInitialize(rc_tmp);
      nalu_hypre_StructVectorAssemble(rc_tmp);
   }
   else
   {
      rc_tmp = rc;
   }
#else
   rc_tmp = rc;
#endif

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch (compute_i)
      {
         case 0:
         {
            rp = nalu_hypre_StructVectorData(r);
            nalu_hypre_InitializeIndtComputations(compute_pkg, rp, &comm_handle);
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

      fi = 0;
      nalu_hypre_ForBoxI(ci, cgrid_boxes)
      {
         while (fgrid_ids[fi] != cgrid_ids[ci])
         {
            fi++;
         }

         compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, fi);

         R_dbox  = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(R),  fi);
         r_dbox  = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(r),  fi);
         rc_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(rc), ci);

         // RL: PTROFFSET
         NALU_HYPRE_Int Rp0_offset = 0, rp0_offset, rp1_offset;

         if (R_stored_as_transpose)
         {
            if ( constant_coefficient )
            {
               Rp0 = nalu_hypre_StructMatrixBoxData(R, fi, 1);
               Rp1 = nalu_hypre_StructMatrixBoxData(R, fi, 0);
               Rp0_offset = -nalu_hypre_CCBoxOffsetDistance(R_dbox, stencil_shape[1]);
            }
            else
            {
               Rp0 = nalu_hypre_StructMatrixBoxData(R, fi, 1);
               Rp1 = nalu_hypre_StructMatrixBoxData(R, fi, 0);
               Rp0_offset = -nalu_hypre_BoxOffsetDistance(R_dbox, stencil_shape[1]);
            }
         }
         else
         {
            Rp0 = nalu_hypre_StructMatrixBoxData(R, fi, 0);
            Rp1 = nalu_hypre_StructMatrixBoxData(R, fi, 1);
         }
         rp  = nalu_hypre_StructVectorBoxData(r, fi);
         rp0_offset = nalu_hypre_BoxOffsetDistance(r_dbox, stencil_shape[0]);
         rp1_offset = nalu_hypre_BoxOffsetDistance(r_dbox, stencil_shape[1]);
         rcp = nalu_hypre_StructVectorBoxData(rc_tmp, ci);

         nalu_hypre_ForBoxI(j, compute_box_a)
         {
            compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

            start  = nalu_hypre_BoxIMin(compute_box);
            nalu_hypre_StructMapFineToCoarse(start, cindex, stride, startc);

            nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

            if ( constant_coefficient )
            {
               NALU_HYPRE_Complex Rp0val, Rp1val;
               Ri = nalu_hypre_CCBoxIndexRank( R_dbox, startc );

               Rp0val = Rp0[Ri + Rp0_offset];
               Rp1val = Rp1[Ri];
#define DEVICE_VAR is_device_ptr(rcp,rp)
               nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(R), loop_size,
                                   r_dbox,  start,  stride,  ri,
                                   rc_dbox, startc, stridec, rci);
               {
                  rcp[rci] = rp[ri] + (Rp0val * rp[ri + rp0_offset] +
                                       Rp1val * rp[ri + rp1_offset]);
               }
               nalu_hypre_BoxLoop2End(ri, rci);
#undef DEVICE_VAR
            }
            else
            {
#define DEVICE_VAR is_device_ptr(rcp,rp,Rp0,Rp1)
               nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(R), loop_size,
                                   R_dbox,  startc, stridec, Ri,
                                   r_dbox,  start,  stride,  ri,
                                   rc_dbox, startc, stridec, rci);
               {
                  rcp[rci] = rp[ri] + (Rp0[Ri + Rp0_offset] * rp[ri + rp0_offset] +
                                       Rp1[Ri]            * rp[ri + rp1_offset]);
               }
               nalu_hypre_BoxLoop3End(Ri, ri, rci);
#undef DEVICE_VAR
            }
         }
      }
   }
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   if (data_location_f != data_location_c)
   {
      nalu_hypre_TMemcpy(nalu_hypre_StructVectorData(rc), nalu_hypre_StructVectorData(rc_tmp), NALU_HYPRE_Complex,
                    nalu_hypre_StructVectorDataSize(rc_tmp), NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_StructVectorDestroy(rc_tmp);
      nalu_hypre_StructGridDataLocation(cgrid) = data_location_c;
   }
#endif
   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   nalu_hypre_IncFLOPCount(4 * nalu_hypre_StructVectorGlobalSize(rc));
   nalu_hypre_EndTiming(restrict_data -> time_index);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SemiRestrictDestroy( void *restrict_vdata )
{
   nalu_hypre_SemiRestrictData *restrict_data = (nalu_hypre_SemiRestrictData *)restrict_vdata;

   if (restrict_data)
   {
      nalu_hypre_StructMatrixDestroy(restrict_data -> R);
      nalu_hypre_ComputePkgDestroy(restrict_data -> compute_pkg);
      nalu_hypre_FinalizeTiming(restrict_data -> time_index);
      nalu_hypre_TFree(restrict_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

