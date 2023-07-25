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
   nalu_hypre_Index          base_index;
   nalu_hypre_Index          base_stride;

   nalu_hypre_StructMatrix  *A;
   nalu_hypre_StructVector  *x;
   nalu_hypre_StructVector  *b;
   nalu_hypre_StructVector  *r;
   nalu_hypre_BoxArray      *base_points;
   nalu_hypre_ComputePkg    *compute_pkg;

   NALU_HYPRE_Int            time_index;
   NALU_HYPRE_BigInt         flops;

} nalu_hypre_SMGResidualData;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_SMGResidualCreate( void )
{
   nalu_hypre_SMGResidualData *residual_data;

   residual_data = nalu_hypre_CTAlloc(nalu_hypre_SMGResidualData,  1, NALU_HYPRE_MEMORY_HOST);

   (residual_data -> time_index)  = nalu_hypre_InitializeTiming("SMGResidual");

   /* set defaults */
   nalu_hypre_SetIndex3((residual_data -> base_index), 0, 0, 0);
   nalu_hypre_SetIndex3((residual_data -> base_stride), 1, 1, 1);

   return (void *) residual_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGResidualSetup( void               *residual_vdata,
                        nalu_hypre_StructMatrix *A,
                        nalu_hypre_StructVector *x,
                        nalu_hypre_StructVector *b,
                        nalu_hypre_StructVector *r              )
{
   nalu_hypre_SMGResidualData  *residual_data = (nalu_hypre_SMGResidualData  *)residual_vdata;

   nalu_hypre_IndexRef          base_index  = (residual_data -> base_index);
   nalu_hypre_IndexRef          base_stride = (residual_data -> base_stride);

   nalu_hypre_StructGrid       *grid;
   nalu_hypre_StructStencil    *stencil;

   nalu_hypre_BoxArray         *base_points;
   nalu_hypre_ComputeInfo      *compute_info;
   nalu_hypre_ComputePkg       *compute_pkg;

   /*----------------------------------------------------------
    * Set up base points and the compute package
    *----------------------------------------------------------*/

   grid    = nalu_hypre_StructMatrixGrid(A);
   stencil = nalu_hypre_StructMatrixStencil(A);

   base_points = nalu_hypre_BoxArrayDuplicate(nalu_hypre_StructGridBoxes(grid));
   nalu_hypre_ProjectBoxArray(base_points, base_index, base_stride);

   nalu_hypre_CreateComputeInfo(grid, stencil, &compute_info);
   nalu_hypre_ComputeInfoProjectComp(compute_info, base_index, base_stride);
   nalu_hypre_ComputePkgCreate(compute_info, nalu_hypre_StructVectorDataSpace(x), 1,
                          grid, &compute_pkg);

   /*----------------------------------------------------------
    * Set up the residual data structure
    *----------------------------------------------------------*/

   (residual_data -> A)           = nalu_hypre_StructMatrixRef(A);
   (residual_data -> x)           = nalu_hypre_StructVectorRef(x);
   (residual_data -> b)           = nalu_hypre_StructVectorRef(b);
   (residual_data -> r)           = nalu_hypre_StructVectorRef(r);
   (residual_data -> base_points) = base_points;
   (residual_data -> compute_pkg) = compute_pkg;

   /*-----------------------------------------------------
    * Compute flops
    *-----------------------------------------------------*/

   (residual_data -> flops) =
      (nalu_hypre_StructMatrixGlobalSize(A) + nalu_hypre_StructVectorGlobalSize(x)) /
      (NALU_HYPRE_BigInt)(nalu_hypre_IndexX(base_stride) *
                     nalu_hypre_IndexY(base_stride) *
                     nalu_hypre_IndexZ(base_stride)  );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGResidual( void               *residual_vdata,
                   nalu_hypre_StructMatrix *A,
                   nalu_hypre_StructVector *x,
                   nalu_hypre_StructVector *b,
                   nalu_hypre_StructVector *r              )
{
   nalu_hypre_SMGResidualData  *residual_data = (nalu_hypre_SMGResidualData  *)residual_vdata;

   nalu_hypre_IndexRef          base_stride = (residual_data -> base_stride);
   nalu_hypre_BoxArray         *base_points = (residual_data -> base_points);
   nalu_hypre_ComputePkg       *compute_pkg = (residual_data -> compute_pkg);

   nalu_hypre_CommHandle       *comm_handle;

   nalu_hypre_BoxArrayArray    *compute_box_aa;
   nalu_hypre_BoxArray         *compute_box_a;
   nalu_hypre_Box              *compute_box;

   nalu_hypre_Box              *A_data_box;
   nalu_hypre_Box              *x_data_box;
   nalu_hypre_Box              *b_data_box;
   nalu_hypre_Box              *r_data_box;

   NALU_HYPRE_Real             *Ap;
   NALU_HYPRE_Real             *xp;
   NALU_HYPRE_Real             *bp;
   NALU_HYPRE_Real             *rp;

   nalu_hypre_Index             loop_size;
   nalu_hypre_IndexRef          start;

   nalu_hypre_StructStencil    *stencil;
   nalu_hypre_Index            *stencil_shape;
   NALU_HYPRE_Int               stencil_size;

   NALU_HYPRE_Int               compute_i, i, j, si;

   nalu_hypre_BeginTiming(residual_data -> time_index);

   /*-----------------------------------------------------------------------
    * Compute residual r = b - Ax
    *-----------------------------------------------------------------------*/

   stencil       = nalu_hypre_StructMatrixStencil(A);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   stencil_size  = nalu_hypre_StructStencilSize(stencil);

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch (compute_i)
      {
         case 0:
         {
            xp = nalu_hypre_StructVectorData(x);
            nalu_hypre_InitializeIndtComputations(compute_pkg, xp, &comm_handle);
            compute_box_aa = nalu_hypre_ComputePkgIndtBoxes(compute_pkg);

            /*----------------------------------------
             * Copy b into r
             *----------------------------------------*/

            compute_box_a = base_points;
            nalu_hypre_ForBoxI(i, compute_box_a)
            {
               compute_box = nalu_hypre_BoxArrayBox(compute_box_a, i);
               start = nalu_hypre_BoxIMin(compute_box);

               b_data_box =
                  nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(b), i);
               r_data_box =
                  nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(r), i);

               bp = nalu_hypre_StructVectorBoxData(b, i);
               rp = nalu_hypre_StructVectorBoxData(r, i);

               nalu_hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);

#define DEVICE_VAR is_device_ptr(rp,bp)
               nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                   b_data_box, start, base_stride, bi,
                                   r_data_box, start, base_stride, ri);
               {
                  rp[ri] = bp[bi];
               }
               nalu_hypre_BoxLoop2End(bi, ri);
#undef DEVICE_VAR
            }
         }
         break;

         case 1:
         {
            nalu_hypre_FinalizeIndtComputations(comm_handle);
            compute_box_aa = nalu_hypre_ComputePkgDeptBoxes(compute_pkg);
         }
         break;
      }

      /*--------------------------------------------------------------------
       * Compute r -= A*x
       *--------------------------------------------------------------------*/

      nalu_hypre_ForBoxArrayI(i, compute_box_aa)
      {
         compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

         A_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), i);
         x_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);
         r_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(r), i);

         rp = nalu_hypre_StructVectorBoxData(r, i);

         nalu_hypre_ForBoxI(j, compute_box_a)
         {
            compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

            start  = nalu_hypre_BoxIMin(compute_box);

            for (si = 0; si < stencil_size; si++)
            {
               Ap = nalu_hypre_StructMatrixBoxData(A, i, si);
               xp = nalu_hypre_StructVectorBoxData(x, i);
               //RL:PTROFFSET
               NALU_HYPRE_Int xp_off = nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[si]);

               nalu_hypre_BoxGetStrideSize(compute_box, base_stride,
                                      loop_size);

#define DEVICE_VAR is_device_ptr(rp,Ap,xp)
               nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                   A_data_box, start, base_stride, Ai,
                                   x_data_box, start, base_stride, xi,
                                   r_data_box, start, base_stride, ri);
               {
                  rp[ri] -= Ap[Ai] * xp[xi + xp_off];
               }
               nalu_hypre_BoxLoop3End(Ai, xi, ri);
#undef DEVICE_VAR
            }
         }
      }
   }

   nalu_hypre_IncFLOPCount(residual_data -> flops);
   nalu_hypre_EndTiming(residual_data -> time_index);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGResidualSetBase( void        *residual_vdata,
                          nalu_hypre_Index  base_index,
                          nalu_hypre_Index  base_stride )
{
   nalu_hypre_SMGResidualData *residual_data = (nalu_hypre_SMGResidualData  *)residual_vdata;
   NALU_HYPRE_Int              d;

   for (d = 0; d < 3; d++)
   {
      nalu_hypre_IndexD((residual_data -> base_index),  d)
         = nalu_hypre_IndexD(base_index,  d);
      nalu_hypre_IndexD((residual_data -> base_stride), d)
         = nalu_hypre_IndexD(base_stride, d);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGResidualDestroy( void *residual_vdata )
{
   nalu_hypre_SMGResidualData *residual_data = (nalu_hypre_SMGResidualData  *)residual_vdata;

   if (residual_data)
   {
      nalu_hypre_StructMatrixDestroy(residual_data -> A);
      nalu_hypre_StructVectorDestroy(residual_data -> x);
      nalu_hypre_StructVectorDestroy(residual_data -> b);
      nalu_hypre_StructVectorDestroy(residual_data -> r);
      nalu_hypre_BoxArrayDestroy(residual_data -> base_points);
      nalu_hypre_ComputePkgDestroy(residual_data -> compute_pkg );
      nalu_hypre_FinalizeTiming(residual_data -> time_index);
      nalu_hypre_TFree(residual_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

