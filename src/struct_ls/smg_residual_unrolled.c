/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_SMGResidualData data structure
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
 * nalu_hypre_SMGResidualCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_SMGResidualCreate( )
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
 * nalu_hypre_SMGResidualSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGResidualSetup( void               *residual_vdata,
                        nalu_hypre_StructMatrix *A,
                        nalu_hypre_StructVector *x,
                        nalu_hypre_StructVector *b,
                        nalu_hypre_StructVector *r              )
{
   NALU_HYPRE_Int ierr;

   nalu_hypre_SMGResidualData  *residual_data = residual_vdata;

   nalu_hypre_IndexRef          base_index  = (residual_data -> base_index);
   nalu_hypre_IndexRef          base_stride = (residual_data -> base_stride);
   nalu_hypre_Index             unit_stride;

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

   nalu_hypre_SetIndex3(unit_stride, 1, 1, 1);

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

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SMGResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGResidual( void               *residual_vdata,
                   nalu_hypre_StructMatrix *A,
                   nalu_hypre_StructVector *x,
                   nalu_hypre_StructVector *b,
                   nalu_hypre_StructVector *r              )
{
   NALU_HYPRE_Int ierr;

   nalu_hypre_SMGResidualData  *residual_data = residual_vdata;

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

   NALU_HYPRE_Int               Ai;
   NALU_HYPRE_Int               xi;
   NALU_HYPRE_Int               bi;
   NALU_HYPRE_Int               ri;

   NALU_HYPRE_Real             *Ap0;
   NALU_HYPRE_Real             *xp0;
   NALU_HYPRE_Real             *bp;
   NALU_HYPRE_Real             *rp;

   nalu_hypre_Index             loop_size;
   nalu_hypre_IndexRef          start;

   nalu_hypre_StructStencil    *stencil;
   nalu_hypre_Index            *stencil_shape;
   NALU_HYPRE_Int               stencil_size;

   NALU_HYPRE_Int               compute_i, i, j, si;

   NALU_HYPRE_Real        *Ap1, *Ap2;
   NALU_HYPRE_Real        *Ap3, *Ap4;
   NALU_HYPRE_Real        *Ap5, *Ap6;
   NALU_HYPRE_Real        *Ap7, *Ap8, *Ap9;
   NALU_HYPRE_Real        *Ap10, *Ap11, *Ap12, *Ap13, *Ap14;
   NALU_HYPRE_Real        *Ap15, *Ap16, *Ap17, *Ap18;
   NALU_HYPRE_Real        *Ap19, *Ap20, *Ap21, *Ap22, *Ap23, *Ap24, *Ap25, *Ap26;
   NALU_HYPRE_Real        *xp1, *xp2;
   NALU_HYPRE_Real        *xp3, *xp4;
   NALU_HYPRE_Real        *xp5, *xp6;
   NALU_HYPRE_Real        *xp7, *xp8, *xp9;
   NALU_HYPRE_Real        *xp10, *xp11, *xp12, *xp13, *xp14;
   NALU_HYPRE_Real        *xp15, *xp16, *xp17, *xp18;
   NALU_HYPRE_Real        *xp19, *xp20, *xp21, *xp22, *xp23, *xp24, *xp25, *xp26;

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
            xp0 = nalu_hypre_StructVectorData(x);
            nalu_hypre_InitializeIndtComputations(compute_pkg, xp0, &comm_handle);
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

         /*--------------------------------------------------------------
          * Switch statement to direct control (based on stencil size) to
          * code to get pointers and offsets fo A and x.
          *--------------------------------------------------------------*/

         switch (stencil_size)
         {
            case 1:

               Ap0 = nalu_hypre_StructMatrixBoxData(A, i, 0);
               xp0 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);

               break;

            case 3:

               Ap0 = nalu_hypre_StructMatrixBoxData(A, i, 0);
               Ap1 = nalu_hypre_StructMatrixBoxData(A, i, 1);
               Ap2 = nalu_hypre_StructMatrixBoxData(A, i, 2);

               xp0 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);
               xp1 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[1]);
               xp2 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[2]);

               break;

            case 5:

               Ap0 = nalu_hypre_StructMatrixBoxData(A, i, 0);
               Ap1 = nalu_hypre_StructMatrixBoxData(A, i, 1);
               Ap2 = nalu_hypre_StructMatrixBoxData(A, i, 2);
               Ap3 = nalu_hypre_StructMatrixBoxData(A, i, 3);
               Ap4 = nalu_hypre_StructMatrixBoxData(A, i, 4);

               xp0 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);
               xp1 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[1]);
               xp2 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[2]);
               xp3 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[3]);
               xp4 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[4]);

               break;

            case 7:

               Ap0 = nalu_hypre_StructMatrixBoxData(A, i, 0);
               Ap1 = nalu_hypre_StructMatrixBoxData(A, i, 1);
               Ap2 = nalu_hypre_StructMatrixBoxData(A, i, 2);
               Ap3 = nalu_hypre_StructMatrixBoxData(A, i, 3);
               Ap4 = nalu_hypre_StructMatrixBoxData(A, i, 4);
               Ap5 = nalu_hypre_StructMatrixBoxData(A, i, 5);
               Ap6 = nalu_hypre_StructMatrixBoxData(A, i, 6);

               xp0 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);
               xp1 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[1]);
               xp2 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[2]);
               xp3 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[3]);
               xp4 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[4]);
               xp5 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[5]);
               xp6 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[6]);

               break;

            case 9:

               Ap0 = nalu_hypre_StructMatrixBoxData(A, i, 0);
               Ap1 = nalu_hypre_StructMatrixBoxData(A, i, 1);
               Ap2 = nalu_hypre_StructMatrixBoxData(A, i, 2);
               Ap3 = nalu_hypre_StructMatrixBoxData(A, i, 3);
               Ap4 = nalu_hypre_StructMatrixBoxData(A, i, 4);
               Ap5 = nalu_hypre_StructMatrixBoxData(A, i, 5);
               Ap6 = nalu_hypre_StructMatrixBoxData(A, i, 6);
               Ap7 = nalu_hypre_StructMatrixBoxData(A, i, 7);
               Ap8 = nalu_hypre_StructMatrixBoxData(A, i, 8);

               xp0 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);
               xp1 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[1]);
               xp2 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[2]);
               xp3 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[3]);
               xp4 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[4]);
               xp5 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[5]);
               xp6 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[6]);
               xp7 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[7]);
               xp8 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[8]);

               break;

            case 15:

               Ap0 = nalu_hypre_StructMatrixBoxData(A, i, 0);
               Ap1 = nalu_hypre_StructMatrixBoxData(A, i, 1);
               Ap2 = nalu_hypre_StructMatrixBoxData(A, i, 2);
               Ap3 = nalu_hypre_StructMatrixBoxData(A, i, 3);
               Ap4 = nalu_hypre_StructMatrixBoxData(A, i, 4);
               Ap5 = nalu_hypre_StructMatrixBoxData(A, i, 5);
               Ap6 = nalu_hypre_StructMatrixBoxData(A, i, 6);
               Ap7 = nalu_hypre_StructMatrixBoxData(A, i, 7);
               Ap8 = nalu_hypre_StructMatrixBoxData(A, i, 8);
               Ap9 = nalu_hypre_StructMatrixBoxData(A, i, 9);
               Ap10 = nalu_hypre_StructMatrixBoxData(A, i, 10);
               Ap11 = nalu_hypre_StructMatrixBoxData(A, i, 11);
               Ap12 = nalu_hypre_StructMatrixBoxData(A, i, 12);
               Ap13 = nalu_hypre_StructMatrixBoxData(A, i, 13);
               Ap14 = nalu_hypre_StructMatrixBoxData(A, i, 14);

               xp0 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);
               xp1 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[1]);
               xp2 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[2]);
               xp3 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[3]);
               xp4 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[4]);
               xp5 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[5]);
               xp6 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[6]);
               xp7 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[7]);
               xp8 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[8]);
               xp9 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[9]);
               xp10 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[10]);
               xp11 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[11]);
               xp12 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[12]);
               xp13 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[13]);
               xp14 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[14]);

               break;

            case 19:

               Ap0 = nalu_hypre_StructMatrixBoxData(A, i, 0);
               Ap1 = nalu_hypre_StructMatrixBoxData(A, i, 1);
               Ap2 = nalu_hypre_StructMatrixBoxData(A, i, 2);
               Ap3 = nalu_hypre_StructMatrixBoxData(A, i, 3);
               Ap4 = nalu_hypre_StructMatrixBoxData(A, i, 4);
               Ap5 = nalu_hypre_StructMatrixBoxData(A, i, 5);
               Ap6 = nalu_hypre_StructMatrixBoxData(A, i, 6);
               Ap7 = nalu_hypre_StructMatrixBoxData(A, i, 7);
               Ap8 = nalu_hypre_StructMatrixBoxData(A, i, 8);
               Ap9 = nalu_hypre_StructMatrixBoxData(A, i, 9);
               Ap10 = nalu_hypre_StructMatrixBoxData(A, i, 10);
               Ap11 = nalu_hypre_StructMatrixBoxData(A, i, 11);
               Ap12 = nalu_hypre_StructMatrixBoxData(A, i, 12);
               Ap13 = nalu_hypre_StructMatrixBoxData(A, i, 13);
               Ap14 = nalu_hypre_StructMatrixBoxData(A, i, 14);
               Ap15 = nalu_hypre_StructMatrixBoxData(A, i, 15);
               Ap16 = nalu_hypre_StructMatrixBoxData(A, i, 16);
               Ap17 = nalu_hypre_StructMatrixBoxData(A, i, 17);
               Ap18 = nalu_hypre_StructMatrixBoxData(A, i, 18);

               xp0 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);
               xp1 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[1]);
               xp2 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[2]);
               xp3 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[3]);
               xp4 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[4]);
               xp5 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[5]);
               xp6 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[6]);
               xp7 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[7]);
               xp8 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[8]);
               xp9 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[9]);
               xp10 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[10]);
               xp11 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[11]);
               xp12 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[12]);
               xp13 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[13]);
               xp14 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[14]);
               xp15 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[15]);
               xp16 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[16]);
               xp17 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[17]);
               xp18 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[18]);

               break;

            case 27:

               Ap0 = nalu_hypre_StructMatrixBoxData(A, i, 0);
               Ap1 = nalu_hypre_StructMatrixBoxData(A, i, 1);
               Ap2 = nalu_hypre_StructMatrixBoxData(A, i, 2);
               Ap3 = nalu_hypre_StructMatrixBoxData(A, i, 3);
               Ap4 = nalu_hypre_StructMatrixBoxData(A, i, 4);
               Ap5 = nalu_hypre_StructMatrixBoxData(A, i, 5);
               Ap6 = nalu_hypre_StructMatrixBoxData(A, i, 6);
               Ap7 = nalu_hypre_StructMatrixBoxData(A, i, 7);
               Ap8 = nalu_hypre_StructMatrixBoxData(A, i, 8);
               Ap9 = nalu_hypre_StructMatrixBoxData(A, i, 9);
               Ap10 = nalu_hypre_StructMatrixBoxData(A, i, 10);
               Ap11 = nalu_hypre_StructMatrixBoxData(A, i, 11);
               Ap12 = nalu_hypre_StructMatrixBoxData(A, i, 12);
               Ap13 = nalu_hypre_StructMatrixBoxData(A, i, 13);
               Ap14 = nalu_hypre_StructMatrixBoxData(A, i, 14);
               Ap15 = nalu_hypre_StructMatrixBoxData(A, i, 15);
               Ap16 = nalu_hypre_StructMatrixBoxData(A, i, 16);
               Ap17 = nalu_hypre_StructMatrixBoxData(A, i, 17);
               Ap18 = nalu_hypre_StructMatrixBoxData(A, i, 18);
               Ap19 = nalu_hypre_StructMatrixBoxData(A, i, 19);
               Ap20 = nalu_hypre_StructMatrixBoxData(A, i, 20);
               Ap21 = nalu_hypre_StructMatrixBoxData(A, i, 21);
               Ap22 = nalu_hypre_StructMatrixBoxData(A, i, 22);
               Ap23 = nalu_hypre_StructMatrixBoxData(A, i, 23);
               Ap24 = nalu_hypre_StructMatrixBoxData(A, i, 24);
               Ap25 = nalu_hypre_StructMatrixBoxData(A, i, 25);
               Ap26 = nalu_hypre_StructMatrixBoxData(A, i, 26);

               xp0 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[0]);
               xp1 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[1]);
               xp2 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[2]);
               xp3 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[3]);
               xp4 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[4]);
               xp5 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[5]);
               xp6 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[6]);
               xp7 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[7]);
               xp8 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[8]);
               xp9 = nalu_hypre_StructVectorBoxData(x, i) +
                     nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[9]);
               xp10 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[10]);
               xp11 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[11]);
               xp12 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[12]);
               xp13 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[13]);
               xp14 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[14]);
               xp15 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[15]);
               xp16 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[16]);
               xp17 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[17]);
               xp18 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[18]);
               xp19 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[19]);
               xp20 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[20]);
               xp21 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[21]);
               xp22 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[22]);
               xp23 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[23]);
               xp24 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[24]);
               xp25 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[25]);
               xp26 = nalu_hypre_StructVectorBoxData(x, i) +
                      nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[26]);

               break;

            default:
               ;
         }

         nalu_hypre_ForBoxI(j, compute_box_a)
         {
            compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

            start  = nalu_hypre_BoxIMin(compute_box);

            /*------------------------------------------------------
             * Switch statement to direct control to appropriate
             * box loop depending on stencil size
             *------------------------------------------------------*/

            switch (stencil_size)
            {

               case 1:

                  nalu_hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);

#define DEVICE_VAR is_device_ptr(rp,Ap0,xp0)
                  nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                      A_data_box, start, base_stride, Ai,
                                      x_data_box, start, base_stride, xi,
                                      r_data_box, start, base_stride, ri);
                  {

                     rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi];

                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, ri);
#undef DEVICE_VAR

                  break;

               case 3:

                  nalu_hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);

#define DEVICE_VAR is_device_ptr(rp,Ap0,xp0,Ap1,xp1,Ap2,xp2)
                  nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                      A_data_box, start, base_stride, Ai,
                                      x_data_box, start, base_stride, xi,
                                      r_data_box, start, base_stride, ri);
                  {

                     rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi]
                              - Ap1[Ai] * xp1[xi]
                              - Ap2[Ai] * xp2[xi];

                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, ri);
#undef DEVICE_VAR

                  break;

               case 5:

                  nalu_hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);

#define DEVICE_VAR is_device_ptr(rp,Ap0,xp0,Ap1,xp1,Ap2,xp2,Ap3,xp3,Ap4,xp4)
                  nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                      A_data_box, start, base_stride, Ai,
                                      x_data_box, start, base_stride, xi,
                                      r_data_box, start, base_stride, ri);
                  {

                     rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi]
                              - Ap1[Ai] * xp1[xi]
                              - Ap2[Ai] * xp2[xi]
                              - Ap3[Ai] * xp3[xi]
                              - Ap4[Ai] * xp4[xi];

                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, ri);
#undef DEVICE_VAR

                  break;

               case 7:

                  nalu_hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);

#define DEVICE_VAR is_device_ptr(rp,Ap0,xp0,Ap1,xp1,Ap2,xp2,Ap3,xp3,Ap4,xp4,Ap5,xp5,Ap6,xp6)
                  nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                      A_data_box, start, base_stride, Ai,
                                      x_data_box, start, base_stride, xi,
                                      r_data_box, start, base_stride, ri);
                  {

                     rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi]
                              - Ap1[Ai] * xp1[xi]
                              - Ap2[Ai] * xp2[xi]
                              - Ap3[Ai] * xp3[xi]
                              - Ap4[Ai] * xp4[xi]
                              - Ap5[Ai] * xp5[xi]
                              - Ap6[Ai] * xp6[xi];

                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, ri);
#undef DEVICE_VAR

                  break;

               case 9:

                  nalu_hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);

#define DEVICE_VAR is_device_ptr(rp,Ap0,xp0,Ap1,xp1,Ap2,xp2,Ap3,xp3,Ap4,xp4,Ap5,xp5,Ap6,xp6,Ap7,xp7,Ap8,xp8)
                  nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                      A_data_box, start, base_stride, Ai,
                                      x_data_box, start, base_stride, xi,
                                      r_data_box, start, base_stride, ri);
                  {

                     rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi]
                              - Ap1[Ai] * xp1[xi]
                              - Ap2[Ai] * xp2[xi]
                              - Ap3[Ai] * xp3[xi]
                              - Ap4[Ai] * xp4[xi]
                              - Ap5[Ai] * xp5[xi]
                              - Ap6[Ai] * xp6[xi]
                              - Ap7[Ai] * xp7[xi]
                              - Ap8[Ai] * xp8[xi];

                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, ri);
#undef DEVICE_VAR

                  break;

               case 15:

                  nalu_hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);

#define DEVICE_VAR is_device_ptr(rp,Ap0,xp0,Ap1,xp1,Ap2,xp2,Ap3,xp3,Ap4,xp4,Ap5,xp5,Ap6,xp6,Ap7,xp7,Ap8,xp8,Ap9,xp9,Ap10,xp10,Ap11,xp11,Ap12,xp12,Ap13,xp13,Ap14,xp14)
                  nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                      A_data_box, start, base_stride, Ai,
                                      x_data_box, start, base_stride, xi,
                                      r_data_box, start, base_stride, ri);
                  {

                     rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi]
                              - Ap1[Ai] * xp1[xi]
                              - Ap2[Ai] * xp2[xi]
                              - Ap3[Ai] * xp3[xi]
                              - Ap4[Ai] * xp4[xi]
                              - Ap5[Ai] * xp5[xi]
                              - Ap6[Ai] * xp6[xi]
                              - Ap7[Ai] * xp7[xi]
                              - Ap8[Ai] * xp8[xi]
                              - Ap9[Ai] * xp9[xi]
                              - Ap10[Ai] * xp10[xi]
                              - Ap11[Ai] * xp11[xi]
                              - Ap12[Ai] * xp12[xi]
                              - Ap13[Ai] * xp13[xi]
                              - Ap14[Ai] * xp14[xi];

                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, ri);
#undef DEVICE_VAR

                  break;

               case 19:

                  nalu_hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);

#define DEVICE_VAR is_device_ptr(rp,Ap0,xp0,Ap1,xp1,Ap2,xp2,Ap3,xp3,Ap4,xp4,Ap5,xp5,Ap6,xp6,Ap7,xp7,Ap8,xp8,Ap9,xp9,Ap10,xp10,Ap11,xp11,Ap12,xp12,Ap13,xp13,Ap14,xp14,Ap15,xp15,Ap16,xp16,Ap17,xp17,Ap18,xp18)
                  nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                      A_data_box, start, base_stride, Ai,
                                      x_data_box, start, base_stride, xi,
                                      r_data_box, start, base_stride, ri);
                  {

                     rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi]
                              - Ap1[Ai] * xp1[xi]
                              - Ap2[Ai] * xp2[xi]
                              - Ap3[Ai] * xp3[xi]
                              - Ap4[Ai] * xp4[xi]
                              - Ap5[Ai] * xp5[xi]
                              - Ap6[Ai] * xp6[xi]
                              - Ap7[Ai] * xp7[xi]
                              - Ap8[Ai] * xp8[xi]
                              - Ap9[Ai] * xp9[xi]
                              - Ap10[Ai] * xp10[xi]
                              - Ap11[Ai] * xp11[xi]
                              - Ap12[Ai] * xp12[xi]
                              - Ap13[Ai] * xp13[xi]
                              - Ap14[Ai] * xp14[xi]
                              - Ap15[Ai] * xp15[xi]
                              - Ap16[Ai] * xp16[xi]
                              - Ap17[Ai] * xp17[xi]
                              - Ap18[Ai] * xp18[xi];

                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, ri);
#undef DEVICE_VAR

                  break;

               case 27:

                  nalu_hypre_BoxGetStrideSize(compute_box, base_stride, loop_size);

#define DEVICE_VAR is_device_ptr(rp,Ap0,xp0,Ap1,xp1,Ap2,xp2,Ap3,xp3,Ap4,xp4,Ap5,xp5,Ap6,xp6,Ap7,xp7,Ap8,xp8,Ap9,xp9,Ap10,xp10,Ap11,xp11,Ap12,xp12,Ap13,xp13,Ap14,xp14,Ap15,xp15,Ap16,xp16,Ap17,xp17,Ap18,xp18,Ap19,xp19,Ap20,xp20,Ap21,xp21,Ap22,xp22,Ap23,xp23,Ap24,xp24,Ap25,xp25,Ap26,xp26)
                  nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                      A_data_box, start, base_stride, Ai,
                                      x_data_box, start, base_stride, xi,
                                      r_data_box, start, base_stride, ri);
                  {

                     rp[ri] = rp[ri]
                              - Ap0[Ai] * xp0[xi]
                              - Ap1[Ai] * xp1[xi]
                              - Ap2[Ai] * xp2[xi]
                              - Ap3[Ai] * xp3[xi]
                              - Ap4[Ai] * xp4[xi]
                              - Ap5[Ai] * xp5[xi]
                              - Ap6[Ai] * xp6[xi]
                              - Ap7[Ai] * xp7[xi]
                              - Ap8[Ai] * xp8[xi]
                              - Ap9[Ai] * xp9[xi]
                              - Ap10[Ai] * xp10[xi]
                              - Ap11[Ai] * xp11[xi]
                              - Ap12[Ai] * xp12[xi]
                              - Ap13[Ai] * xp13[xi]
                              - Ap14[Ai] * xp14[xi]
                              - Ap15[Ai] * xp15[xi]
                              - Ap16[Ai] * xp16[xi]
                              - Ap17[Ai] * xp17[xi]
                              - Ap18[Ai] * xp18[xi]
                              - Ap19[Ai] * xp19[xi]
                              - Ap20[Ai] * xp20[xi]
                              - Ap21[Ai] * xp21[xi]
                              - Ap22[Ai] * xp22[xi]
                              - Ap23[Ai] * xp23[xi]
                              - Ap24[Ai] * xp24[xi]
                              - Ap25[Ai] * xp25[xi]
                              - Ap26[Ai] * xp26[xi];

                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, ri);
#undef DEVICE_VAR

                  break;

               default:

                  for (si = 0; si < stencil_size; si++)
                  {
                     Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si);
                     xp0 = nalu_hypre_StructVectorBoxData(x, i) +
                           nalu_hypre_BoxOffsetDistance(x_data_box, stencil_shape[si]);

                     nalu_hypre_BoxGetStrideSize(compute_box, base_stride,
                                            loop_size);

#define DEVICE_VAR is_device_ptr(rp,Ap0,xp0)
                     nalu_hypre_BoxLoop3Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                         A_data_box, start, base_stride, Ai,
                                         x_data_box, start, base_stride, xi,
                                         r_data_box, start, base_stride, ri);
                     {
                        rp[ri] -= Ap0[Ai] * xp0[xi];
                     }
                     nalu_hypre_BoxLoop3End(Ai, xi, ri);
#undef DEVICE_VAR
                  }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   nalu_hypre_IncFLOPCount(residual_data -> flops);
   nalu_hypre_EndTiming(residual_data -> time_index);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SMGResidualSetBase
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGResidualSetBase( void        *residual_vdata,
                          nalu_hypre_Index  base_index,
                          nalu_hypre_Index  base_stride )
{
   nalu_hypre_SMGResidualData *residual_data = residual_vdata;
   NALU_HYPRE_Int              d;
   NALU_HYPRE_Int              ierr = 0;

   for (d = 0; d < 3; d++)
   {
      nalu_hypre_IndexD((residual_data -> base_index),  d)
         = nalu_hypre_IndexD(base_index,  d);
      nalu_hypre_IndexD((residual_data -> base_stride), d)
         = nalu_hypre_IndexD(base_stride, d);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SMGResidualDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGResidualDestroy( void *residual_vdata )
{
   NALU_HYPRE_Int ierr;

   nalu_hypre_SMGResidualData *residual_data = residual_vdata;

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

   return ierr;
}

