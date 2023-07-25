/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Structured matrix-vector multiply routine
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_mv.h"
#include "_nalu_hypre_struct_mv.hpp"

/* this currently cannot be greater than 7 */
#ifdef MAX_DEPTH
#undef MAX_DEPTH
#endif
#define MAX_DEPTH 7

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatvecData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   nalu_hypre_StructMatrix  *A;
   nalu_hypre_StructVector  *x;
   nalu_hypre_ComputePkg    *compute_pkg;

} nalu_hypre_StructMatvecData;

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatvecCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_StructMatvecCreate( void )
{
   nalu_hypre_StructMatvecData *matvec_data;

   matvec_data = nalu_hypre_CTAlloc(nalu_hypre_StructMatvecData,  1, NALU_HYPRE_MEMORY_HOST);

   return (void *) matvec_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatvecSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatvecSetup( void               *matvec_vdata,
                         nalu_hypre_StructMatrix *A,
                         nalu_hypre_StructVector *x            )
{
   nalu_hypre_StructMatvecData  *matvec_data = (nalu_hypre_StructMatvecData  *)matvec_vdata;

   nalu_hypre_StructGrid        *grid;
   nalu_hypre_StructStencil     *stencil;
   nalu_hypre_ComputeInfo       *compute_info;
   nalu_hypre_ComputePkg        *compute_pkg;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = nalu_hypre_StructMatrixGrid(A);
   stencil = nalu_hypre_StructMatrixStencil(A);

   nalu_hypre_CreateComputeInfo(grid, stencil, &compute_info);
   nalu_hypre_ComputePkgCreate(compute_info, nalu_hypre_StructVectorDataSpace(x), 1,
                          grid, &compute_pkg);

   /*----------------------------------------------------------
    * Set up the matvec data structure
    *----------------------------------------------------------*/

   (matvec_data -> A)           = nalu_hypre_StructMatrixRef(A);
   (matvec_data -> x)           = nalu_hypre_StructVectorRef(x);
   (matvec_data -> compute_pkg) = compute_pkg;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatvecCompute
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatvecCompute( void               *matvec_vdata,
                           NALU_HYPRE_Complex       alpha,
                           nalu_hypre_StructMatrix *A,
                           nalu_hypre_StructVector *x,
                           NALU_HYPRE_Complex       beta,
                           nalu_hypre_StructVector *y            )
{
   nalu_hypre_StructMatvecData  *matvec_data = (nalu_hypre_StructMatvecData  *)matvec_vdata;

   nalu_hypre_ComputePkg        *compute_pkg;

   nalu_hypre_CommHandle        *comm_handle;

   nalu_hypre_BoxArrayArray     *compute_box_aa;
   nalu_hypre_Box               *y_data_box;

   NALU_HYPRE_Complex           *xp;
   NALU_HYPRE_Complex           *yp;

   nalu_hypre_BoxArray          *boxes;
   nalu_hypre_Box               *box;
   nalu_hypre_Index              loop_size;
   nalu_hypre_IndexRef           start;
   nalu_hypre_IndexRef           stride;

   NALU_HYPRE_Int                constant_coefficient;

   NALU_HYPRE_Complex            temp;
   NALU_HYPRE_Int                compute_i, i;

   nalu_hypre_StructVector      *x_tmp = NULL;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(A);
   if (constant_coefficient) { nalu_hypre_StructVectorClearBoundGhostValues(x, 0); }

   compute_pkg = (matvec_data -> compute_pkg);

   stride = nalu_hypre_ComputePkgStride(compute_pkg);

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
      boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(A));
      nalu_hypre_ForBoxI(i, boxes)
      {
         box   = nalu_hypre_BoxArrayBox(boxes, i);
         start = nalu_hypre_BoxIMin(box);

         y_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(y), i);
         yp = nalu_hypre_StructVectorBoxData(y, i);

         nalu_hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(yp)
         nalu_hypre_BoxLoop1Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                             y_data_box, start, stride, yi);
         {
            yp[yi] *= beta;
         }
         nalu_hypre_BoxLoop1End(yi);
#undef DEVICE_VAR
      }

      return nalu_hypre_error_flag;
   }

   if (x == y)
   {
      x_tmp = nalu_hypre_StructVectorClone(y);
      x = x_tmp;
   }
   /*-----------------------------------------------------------------------
    * Do (alpha != 0.0) computation
    *-----------------------------------------------------------------------*/

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch (compute_i)
      {
         case 0:
         {
            xp = nalu_hypre_StructVectorData(x);
            nalu_hypre_InitializeIndtComputations(compute_pkg, xp, &comm_handle);
            compute_box_aa = nalu_hypre_ComputePkgIndtBoxes(compute_pkg);

            /*--------------------------------------------------------------
             * initialize y= (beta/alpha)*y normally (where everything
             * is multiplied by alpha at the end),
             * beta*y for constant coefficient (where only Ax gets multiplied by alpha)
             *--------------------------------------------------------------*/

            if ( constant_coefficient == 1 )
            {
               temp = beta;
            }
            else
            {
               temp = beta / alpha;
            }
            if (temp != 1.0)
            {
               boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(A));
               nalu_hypre_ForBoxI(i, boxes)
               {
                  box   = nalu_hypre_BoxArrayBox(boxes, i);
                  start = nalu_hypre_BoxIMin(box);

                  y_data_box =
                     nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(y), i);
                  yp = nalu_hypre_StructVectorBoxData(y, i);

#define DEVICE_VAR is_device_ptr(yp)
                  if (temp == 0.0)
                  {
                     nalu_hypre_BoxGetSize(box, loop_size);

                     nalu_hypre_BoxLoop1Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                                         y_data_box, start, stride, yi);
                     {
                        yp[yi] = 0.0;
                     }
                     nalu_hypre_BoxLoop1End(yi);
                  }
                  else
                  {
                     nalu_hypre_BoxGetSize(box, loop_size);

                     nalu_hypre_BoxLoop1Begin(nalu_hypre_StructVectorNDim(x), loop_size,
                                         y_data_box, start, stride, yi);
                     {
                        yp[yi] *= temp;
                     }
                     nalu_hypre_BoxLoop1End(yi);
                  }
#undef DEVICE_VAR
               }
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
       * y += A*x
       *--------------------------------------------------------------------*/

      switch ( constant_coefficient )
      {
         case 0:
         {
            nalu_hypre_StructMatvecCC0( alpha, A, x, y, compute_box_aa, stride );
            break;
         }
         case 1:
         {
            nalu_hypre_StructMatvecCC1( alpha, A, x, y, compute_box_aa, stride );
            break;
         }
         case 2:
         {
            nalu_hypre_StructMatvecCC2( alpha, A, x, y, compute_box_aa, stride );
            break;
         }
      }

   }

   if (x_tmp)
   {
      nalu_hypre_StructVectorDestroy(x_tmp);
      x = y;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatvecCC0
 * core of struct matvec computation, for the case constant_coefficient==0
 * (all coefficients are variable)
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_StructMatvecCC0( NALU_HYPRE_Complex       alpha,
                                 nalu_hypre_StructMatrix *A,
                                 nalu_hypre_StructVector *x,
                                 nalu_hypre_StructVector *y,
                                 nalu_hypre_BoxArrayArray     *compute_box_aa,
                                 nalu_hypre_IndexRef           stride
                               )
{
   NALU_HYPRE_Int i, j, si;
   NALU_HYPRE_Complex           *Ap0;
   NALU_HYPRE_Complex           *Ap1;
   NALU_HYPRE_Complex           *Ap2;
   NALU_HYPRE_Complex           *Ap3;
   NALU_HYPRE_Complex           *Ap4;
   NALU_HYPRE_Complex           *Ap5;
   NALU_HYPRE_Complex           *Ap6;
   NALU_HYPRE_Int                xoff0;
   NALU_HYPRE_Int                xoff1;
   NALU_HYPRE_Int                xoff2;
   NALU_HYPRE_Int                xoff3;
   NALU_HYPRE_Int                xoff4;
   NALU_HYPRE_Int                xoff5;
   NALU_HYPRE_Int                xoff6;
   nalu_hypre_BoxArray          *compute_box_a;
   nalu_hypre_Box               *compute_box;

   nalu_hypre_Box               *A_data_box;
   nalu_hypre_Box               *x_data_box;
   nalu_hypre_StructStencil     *stencil;
   nalu_hypre_Index             *stencil_shape;
   NALU_HYPRE_Int                stencil_size;

   nalu_hypre_Box               *y_data_box;
   NALU_HYPRE_Complex           *xp;
   NALU_HYPRE_Complex           *yp;
   NALU_HYPRE_Int                depth;
   nalu_hypre_Index              loop_size;
   nalu_hypre_IndexRef           start;
   NALU_HYPRE_Int                ndim;

   stencil       = nalu_hypre_StructMatrixStencil(A);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   stencil_size  = nalu_hypre_StructStencilSize(stencil);
   ndim          = nalu_hypre_StructVectorNDim(x);

   nalu_hypre_ForBoxArrayI(i, compute_box_aa)
   {
      compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

      A_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), i);
      x_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);
      y_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(y), i);

      xp = nalu_hypre_StructVectorBoxData(x, i);
      yp = nalu_hypre_StructVectorBoxData(y, i);

      nalu_hypre_ForBoxI(j, compute_box_a)
      {
         compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

         nalu_hypre_BoxGetSize(compute_box, loop_size);
         start  = nalu_hypre_BoxIMin(compute_box);

         /* unroll up to depth MAX_DEPTH */
         for (si = 0; si < stencil_size; si += MAX_DEPTH)
         {
            depth = nalu_hypre_min(MAX_DEPTH, (stencil_size - si));
            switch (depth)
            {
               case 7:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = nalu_hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = nalu_hypre_StructMatrixBoxData(A, i, si + 4);
                  Ap5 = nalu_hypre_StructMatrixBoxData(A, i, si + 5);
                  Ap6 = nalu_hypre_StructMatrixBoxData(A, i, si + 6);

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);
                  xoff5 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 5]);
                  xoff6 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 6]);

#define DEVICE_VAR is_device_ptr(yp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,xp)
                  nalu_hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, stride, Ai,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        Ap0[Ai] * xp[xi + xoff0] +
                        Ap1[Ai] * xp[xi + xoff1] +
                        Ap2[Ai] * xp[xi + xoff2] +
                        Ap3[Ai] * xp[xi + xoff3] +
                        Ap4[Ai] * xp[xi + xoff4] +
                        Ap5[Ai] * xp[xi + xoff5] +
                        Ap6[Ai] * xp[xi + xoff6];
                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR

                  break;

               case 6:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = nalu_hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = nalu_hypre_StructMatrixBoxData(A, i, si + 4);
                  Ap5 = nalu_hypre_StructMatrixBoxData(A, i, si + 5);

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);
                  xoff5 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 5]);

#define DEVICE_VAR is_device_ptr(yp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,xp)
                  nalu_hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, stride, Ai,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        Ap0[Ai] * xp[xi + xoff0] +
                        Ap1[Ai] * xp[xi + xoff1] +
                        Ap2[Ai] * xp[xi + xoff2] +
                        Ap3[Ai] * xp[xi + xoff3] +
                        Ap4[Ai] * xp[xi + xoff4] +
                        Ap5[Ai] * xp[xi + xoff5];
                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR

                  break;

               case 5:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = nalu_hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = nalu_hypre_StructMatrixBoxData(A, i, si + 4);

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);

#define DEVICE_VAR is_device_ptr(yp,Ap0,Ap1,Ap2,Ap3,Ap4,xp)
                  nalu_hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, stride, Ai,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        Ap0[Ai] * xp[xi + xoff0] +
                        Ap1[Ai] * xp[xi + xoff1] +
                        Ap2[Ai] * xp[xi + xoff2] +
                        Ap3[Ai] * xp[xi + xoff3] +
                        Ap4[Ai] * xp[xi + xoff4];
                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR

                  break;

               case 4:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = nalu_hypre_StructMatrixBoxData(A, i, si + 3);

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);

#define DEVICE_VAR is_device_ptr(yp,Ap0,Ap1,Ap2,Ap3,xp)
                  nalu_hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, stride, Ai,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        Ap0[Ai] * xp[xi + xoff0] +
                        Ap1[Ai] * xp[xi + xoff1] +
                        Ap2[Ai] * xp[xi + xoff2] +
                        Ap3[Ai] * xp[xi + xoff3];
                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR

                  break;

               case 3:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);

#define DEVICE_VAR is_device_ptr(yp,Ap0,Ap1,Ap2,xp)
                  nalu_hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, stride, Ai,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        Ap0[Ai] * xp[xi + xoff0] +
                        Ap1[Ai] * xp[xi + xoff1] +
                        Ap2[Ai] * xp[xi + xoff2];
                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR

                  break;

               case 2:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);

#define DEVICE_VAR is_device_ptr(yp,Ap0,Ap1,xp)
                  nalu_hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, stride, Ai,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        Ap0[Ai] * xp[xi + xoff0] +
                        Ap1[Ai] * xp[xi + xoff1];
                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR

                  break;

               case 1:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);

#define DEVICE_VAR is_device_ptr(yp,Ap0,xp)
                  nalu_hypre_BoxLoop3Begin(ndim, loop_size,
                                      A_data_box, start, stride, Ai,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        Ap0[Ai] * xp[xi + xoff0];
                  }
                  nalu_hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR

                  break;
            }
         }

         if (alpha != 1.0)
         {
#define DEVICE_VAR is_device_ptr(yp)
            nalu_hypre_BoxLoop1Begin(ndim, loop_size,
                                y_data_box, start, stride, yi);
            {
               yp[yi] *= alpha;
            }
            nalu_hypre_BoxLoop1End(yi);
#undef DEVICE_VAR
         }
      }
   }

   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatvecCC1
 * core of struct matvec computation, for the case constant_coefficient==1
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_StructMatvecCC1( NALU_HYPRE_Complex       alpha,
                                 nalu_hypre_StructMatrix *A,
                                 nalu_hypre_StructVector *x,
                                 nalu_hypre_StructVector *y,
                                 nalu_hypre_BoxArrayArray     *compute_box_aa,
                                 nalu_hypre_IndexRef           stride
                               )
{
   NALU_HYPRE_Int i, j, si;
   NALU_HYPRE_Complex           *Ap0;
   NALU_HYPRE_Complex           *Ap1;
   NALU_HYPRE_Complex           *Ap2;
   NALU_HYPRE_Complex           *Ap3;
   NALU_HYPRE_Complex           *Ap4;
   NALU_HYPRE_Complex           *Ap5;
   NALU_HYPRE_Complex           *Ap6;
   NALU_HYPRE_Complex           AAp0;
   NALU_HYPRE_Complex           AAp1;
   NALU_HYPRE_Complex           AAp2;
   NALU_HYPRE_Complex           AAp3;
   NALU_HYPRE_Complex           AAp4;
   NALU_HYPRE_Complex           AAp5;
   NALU_HYPRE_Complex           AAp6;
   NALU_HYPRE_Int                xoff0;
   NALU_HYPRE_Int                xoff1;
   NALU_HYPRE_Int                xoff2;
   NALU_HYPRE_Int                xoff3;
   NALU_HYPRE_Int                xoff4;
   NALU_HYPRE_Int                xoff5;
   NALU_HYPRE_Int                xoff6;
   NALU_HYPRE_Int                Ai;

   nalu_hypre_BoxArray          *compute_box_a;
   nalu_hypre_Box               *compute_box;

   nalu_hypre_Box               *x_data_box;
   nalu_hypre_StructStencil     *stencil;
   nalu_hypre_Index             *stencil_shape;
   NALU_HYPRE_Int                stencil_size;

   nalu_hypre_Box               *y_data_box;
   NALU_HYPRE_Complex           *xp;
   NALU_HYPRE_Complex           *yp;
   NALU_HYPRE_Int                depth;
   nalu_hypre_Index              loop_size;
   nalu_hypre_IndexRef           start;
   NALU_HYPRE_Int                ndim;

   stencil       = nalu_hypre_StructMatrixStencil(A);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   stencil_size  = nalu_hypre_StructStencilSize(stencil);
   ndim          = nalu_hypre_StructVectorNDim(x);

   nalu_hypre_ForBoxArrayI(i, compute_box_aa)
   {
      compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

      x_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);
      y_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(y), i);

      xp = nalu_hypre_StructVectorBoxData(x, i);
      yp = nalu_hypre_StructVectorBoxData(y, i);

      nalu_hypre_ForBoxI(j, compute_box_a)
      {
         compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

         nalu_hypre_BoxGetSize(compute_box, loop_size);
         start  = nalu_hypre_BoxIMin(compute_box);

         Ai = 0;

         /* unroll up to depth MAX_DEPTH */
         for (si = 0; si < stencil_size; si += MAX_DEPTH)
         {
            depth = nalu_hypre_min(MAX_DEPTH, (stencil_size - si));
            switch (depth)
            {
               case 7:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = nalu_hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = nalu_hypre_StructMatrixBoxData(A, i, si + 4);
                  Ap5 = nalu_hypre_StructMatrixBoxData(A, i, si + 5);
                  Ap6 = nalu_hypre_StructMatrixBoxData(A, i, si + 6);
                  AAp0 = Ap0[Ai] * alpha;
                  AAp1 = Ap1[Ai] * alpha;
                  AAp2 = Ap2[Ai] * alpha;
                  AAp3 = Ap3[Ai] * alpha;
                  AAp4 = Ap4[Ai] * alpha;
                  AAp5 = Ap5[Ai] * alpha;
                  AAp6 = Ap6[Ai] * alpha;

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);
                  xoff5 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 5]);
                  xoff6 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 6]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3] +
                        AAp4 * xp[xi + xoff4] +
                        AAp5 * xp[xi + xoff5] +
                        AAp6 * xp[xi + xoff6];
                  }
                  nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 6:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = nalu_hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = nalu_hypre_StructMatrixBoxData(A, i, si + 4);
                  Ap5 = nalu_hypre_StructMatrixBoxData(A, i, si + 5);
                  AAp0 = Ap0[Ai] * alpha;
                  AAp1 = Ap1[Ai] * alpha;
                  AAp2 = Ap2[Ai] * alpha;
                  AAp3 = Ap3[Ai] * alpha;
                  AAp4 = Ap4[Ai] * alpha;
                  AAp5 = Ap5[Ai] * alpha;

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);
                  xoff5 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 5]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3] +
                        AAp4 * xp[xi + xoff4] +
                        AAp5 * xp[xi + xoff5];
                  }
                  nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 5:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = nalu_hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = nalu_hypre_StructMatrixBoxData(A, i, si + 4);
                  AAp0 = Ap0[Ai] * alpha;
                  AAp1 = Ap1[Ai] * alpha;
                  AAp2 = Ap2[Ai] * alpha;
                  AAp3 = Ap3[Ai] * alpha;
                  AAp4 = Ap4[Ai] * alpha;

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3] +
                        AAp4 * xp[xi + xoff4];
                  }
                  nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 4:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = nalu_hypre_StructMatrixBoxData(A, i, si + 3);
                  AAp0 = Ap0[Ai] * alpha;
                  AAp1 = Ap1[Ai] * alpha;
                  AAp2 = Ap2[Ai] * alpha;
                  AAp3 = Ap3[Ai] * alpha;

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3];
                  }
                  nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 3:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);
                  AAp0 = Ap0[Ai] * alpha;
                  AAp1 = Ap1[Ai] * alpha;
                  AAp2 = Ap2[Ai] * alpha;

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2];
                  }
                  nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 2:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  AAp0 = Ap0[Ai] * alpha;
                  AAp1 = Ap1[Ai] * alpha;

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1];
                  }
                  nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 1:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  AAp0 = Ap0[Ai] * alpha;

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0];
                  }
                  nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
            }
         }
      }
   }

   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatvecCC2
 * core of struct matvec computation, for the case constant_coefficient==2
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_StructMatvecCC2( NALU_HYPRE_Complex       alpha,
                                 nalu_hypre_StructMatrix *A,
                                 nalu_hypre_StructVector *x,
                                 nalu_hypre_StructVector *y,
                                 nalu_hypre_BoxArrayArray     *compute_box_aa,
                                 nalu_hypre_IndexRef           stride
                               )
{
   NALU_HYPRE_Int i, j, si;
   NALU_HYPRE_Complex           *Ap0;
   NALU_HYPRE_Complex           *Ap1;
   NALU_HYPRE_Complex           *Ap2;
   NALU_HYPRE_Complex           *Ap3;
   NALU_HYPRE_Complex           *Ap4;
   NALU_HYPRE_Complex           *Ap5;
   NALU_HYPRE_Complex           *Ap6;
   NALU_HYPRE_Complex           AAp0;
   NALU_HYPRE_Complex           AAp1;
   NALU_HYPRE_Complex           AAp2;
   NALU_HYPRE_Complex           AAp3;
   NALU_HYPRE_Complex           AAp4;
   NALU_HYPRE_Complex           AAp5;
   NALU_HYPRE_Complex           AAp6;
   NALU_HYPRE_Int                xoff0;
   NALU_HYPRE_Int                xoff1;
   NALU_HYPRE_Int                xoff2;
   NALU_HYPRE_Int                xoff3;
   NALU_HYPRE_Int                xoff4;
   NALU_HYPRE_Int                xoff5;
   NALU_HYPRE_Int                xoff6;
   NALU_HYPRE_Int                si_center, center_rank;
   nalu_hypre_Index              center_index;
   NALU_HYPRE_Int                Ai_CC;
   nalu_hypre_BoxArray          *compute_box_a;
   nalu_hypre_Box               *compute_box;

   nalu_hypre_Box               *A_data_box;
   nalu_hypre_Box               *x_data_box;
   nalu_hypre_StructStencil     *stencil;
   nalu_hypre_Index             *stencil_shape;
   NALU_HYPRE_Int                stencil_size;

   nalu_hypre_Box               *y_data_box;
   NALU_HYPRE_Complex           *xp;
   NALU_HYPRE_Complex           *yp;
   NALU_HYPRE_Int                depth;
   nalu_hypre_Index              loop_size;
   nalu_hypre_IndexRef           start;
   NALU_HYPRE_Int                ndim;
   NALU_HYPRE_Complex            zero[1] = {0};

   stencil       = nalu_hypre_StructMatrixStencil(A);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   stencil_size  = nalu_hypre_StructStencilSize(stencil);
   ndim          = nalu_hypre_StructVectorNDim(x);

   nalu_hypre_ForBoxArrayI(i, compute_box_aa)
   {
      compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

      A_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), i);
      x_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);
      y_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(y), i);

      xp = nalu_hypre_StructVectorBoxData(x, i);
      yp = nalu_hypre_StructVectorBoxData(y, i);

      nalu_hypre_ForBoxI(j, compute_box_a)
      {
         compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

         nalu_hypre_BoxGetSize(compute_box, loop_size);
         start  = nalu_hypre_BoxIMin(compute_box);

         Ai_CC = nalu_hypre_CCBoxIndexRank( A_data_box, start );

         /* Find the stencil index for the center of the stencil, which
            makes the matrix diagonal.  This is the variable coefficient
            part of the matrix, so will get different treatment...*/
         nalu_hypre_SetIndex(center_index, 0);
         center_rank = nalu_hypre_StructStencilElementRank( stencil, center_index );
         si_center = center_rank;

         /* unroll up to depth MAX_DEPTH
            Only the constant coefficient part of the matrix is referenced here,
            the center (variable) coefficient part is deferred. */
         for (si = 0; si < stencil_size; si += MAX_DEPTH)
         {
            depth = nalu_hypre_min(MAX_DEPTH, (stencil_size - si));
            switch (depth)
            {
               case 7:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = nalu_hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = nalu_hypre_StructMatrixBoxData(A, i, si + 4);
                  Ap5 = nalu_hypre_StructMatrixBoxData(A, i, si + 5);
                  Ap6 = nalu_hypre_StructMatrixBoxData(A, i, si + 6);
                  if ( (0 <= si_center - si) && (si_center - si < 7) )
                  {
                     switch ( si_center - si )
                     {
                        case 0: Ap0 = zero; break;
                        case 1: Ap1 = zero; break;
                        case 2: Ap2 = zero; break;
                        case 3: Ap3 = zero; break;
                        case 4: Ap4 = zero; break;
                        case 5: Ap5 = zero; break;
                        case 6: Ap6 = zero; break;
                     }
                  }

                  AAp0 = Ap0[Ai_CC];
                  AAp1 = Ap1[Ai_CC];
                  AAp2 = Ap2[Ai_CC];
                  AAp3 = Ap3[Ai_CC];
                  AAp4 = Ap4[Ai_CC];
                  AAp5 = Ap5[Ai_CC];
                  AAp6 = Ap6[Ai_CC];


                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);
                  xoff5 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 5]);
                  xoff6 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 6]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3] +
                        AAp4 * xp[xi + xoff4] +
                        AAp5 * xp[xi + xoff5] +
                        AAp6 * xp[xi + xoff6];
                  }
                  nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR

                  break;

               case 6:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = nalu_hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = nalu_hypre_StructMatrixBoxData(A, i, si + 4);
                  Ap5 = nalu_hypre_StructMatrixBoxData(A, i, si + 5);
                  if ( (0 <= si_center - si) && (si_center - si < 6) )
                  {
                     switch ( si_center - si )
                     {
                        case 0: Ap0 = zero; break;
                        case 1: Ap1 = zero; break;
                        case 2: Ap2 = zero; break;
                        case 3: Ap3 = zero; break;
                        case 4: Ap4 = zero; break;
                        case 5: Ap5 = zero; break;
                     }
                  }
                  AAp0 = Ap0[Ai_CC];
                  AAp1 = Ap1[Ai_CC];
                  AAp2 = Ap2[Ai_CC];
                  AAp3 = Ap3[Ai_CC];
                  AAp4 = Ap4[Ai_CC];
                  AAp5 = Ap5[Ai_CC];

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);
                  xoff5 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 5]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3] +
                        AAp4 * xp[xi + xoff4] +
                        AAp5 * xp[xi + xoff5];
                  }
                  nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 5:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = nalu_hypre_StructMatrixBoxData(A, i, si + 3);
                  Ap4 = nalu_hypre_StructMatrixBoxData(A, i, si + 4);
                  if ( (0 <= si_center - si) && (si_center - si < 5) )
                  {
                     switch ( si_center - si )
                     {
                        case 0: Ap0 = zero; break;
                        case 1: Ap1 = zero; break;
                        case 2: Ap2 = zero; break;
                        case 3: Ap3 = zero; break;
                        case 4: Ap4 = zero; break;
                     }
                  }
                  AAp0 = Ap0[Ai_CC];
                  AAp1 = Ap1[Ai_CC];
                  AAp2 = Ap2[Ai_CC];
                  AAp3 = Ap3[Ai_CC];
                  AAp4 = Ap4[Ai_CC];

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);
                  xoff4 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 4]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3] +
                        AAp4 * xp[xi + xoff4];
                  }
                  nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 4:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);
                  Ap3 = nalu_hypre_StructMatrixBoxData(A, i, si + 3);
                  if ( (0 <= si_center - si) && (si_center - si < 4) )
                  {
                     switch ( si_center - si )
                     {
                        case 0: Ap0 = zero; break;
                        case 1: Ap1 = zero; break;
                        case 2: Ap2 = zero; break;
                        case 3: Ap3 = zero; break;
                     }
                  }
                  AAp0 = Ap0[Ai_CC];
                  AAp1 = Ap1[Ai_CC];
                  AAp2 = Ap2[Ai_CC];
                  AAp3 = Ap3[Ai_CC];

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);
                  xoff3 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 3]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2] +
                        AAp3 * xp[xi + xoff3];
                  }
                  nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 3:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  Ap2 = nalu_hypre_StructMatrixBoxData(A, i, si + 2);
                  if ( (0 <= si_center - si) && (si_center - si < 3) )
                  {
                     switch ( si_center - si )
                     {
                        case 0: Ap0 = zero; break;
                        case 1: Ap1 = zero; break;
                        case 2: Ap2 = zero; break;
                     }
                  }
                  AAp0 = Ap0[Ai_CC];
                  AAp1 = Ap1[Ai_CC];
                  AAp2 = Ap2[Ai_CC];

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);
                  xoff2 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 2]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1] +
                        AAp2 * xp[xi + xoff2];
                  }
                  nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 2:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si + 1);
                  if ( (0 <= si_center - si) && (si_center - si < 2) )
                  {
                     switch ( si_center - si )
                     {
                        case 0: Ap0 = zero; break;
                        case 1: Ap1 = zero; break;
                     }
                  }
                  AAp0 = Ap0[Ai_CC];
                  AAp1 = Ap1[Ai_CC];

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);
                  xoff1 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 1]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0] +
                        AAp1 * xp[xi + xoff1];
                  }
                  nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
                  break;

               case 1:
                  Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si + 0);
                  if ( si_center - si == 0 )
                  {
                     Ap0 = zero;
                  }
                  AAp0 = Ap0[Ai_CC];

                  xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                                  stencil_shape[si + 0]);

#define DEVICE_VAR is_device_ptr(yp,xp)
                  nalu_hypre_BoxLoop2Begin(ndim, loop_size,
                                      x_data_box, start, stride, xi,
                                      y_data_box, start, stride, yi);
                  {
                     yp[yi] +=
                        AAp0 * xp[xi + xoff0];
                  }
                  nalu_hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR

                  break;
            }
         }

         Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si_center);
         xoff0 = nalu_hypre_BoxOffsetDistance(x_data_box,
                                         stencil_shape[si_center]);
         if (alpha != 1.0 )
         {
#define DEVICE_VAR is_device_ptr(yp,Ap0,xp)
            nalu_hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                y_data_box, start, stride, yi);
            {
               yp[yi] = alpha * ( yp[yi] +
                                  Ap0[Ai] * xp[xi + xoff0] );
            }
            nalu_hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR
         }
         else
         {
#define DEVICE_VAR is_device_ptr(yp,Ap0,xp)
            nalu_hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, start, stride, Ai,
                                x_data_box, start, stride, xi,
                                y_data_box, start, stride, yi);
            {
               yp[yi] +=
                  Ap0[Ai] * xp[xi + xoff0];
            }
            nalu_hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR
         }

      }
   }

   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatvecDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatvecDestroy( void *matvec_vdata )
{
   nalu_hypre_StructMatvecData *matvec_data = (nalu_hypre_StructMatvecData *)matvec_vdata;

   if (matvec_data)
   {
      nalu_hypre_StructMatrixDestroy(matvec_data -> A);
      nalu_hypre_StructVectorDestroy(matvec_data -> x);
      nalu_hypre_ComputePkgDestroy(matvec_data -> compute_pkg );
      nalu_hypre_TFree(matvec_data, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_StructMatvec
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_StructMatvec( NALU_HYPRE_Complex       alpha,
                    nalu_hypre_StructMatrix *A,
                    nalu_hypre_StructVector *x,
                    NALU_HYPRE_Complex       beta,
                    nalu_hypre_StructVector *y     )
{
   void *matvec_data;

   matvec_data = nalu_hypre_StructMatvecCreate();
   nalu_hypre_StructMatvecSetup(matvec_data, A, x);
   nalu_hypre_StructMatvecCompute(matvec_data, alpha, A, x, beta, y);
   nalu_hypre_StructMatvecDestroy(matvec_data);

   return nalu_hypre_error_flag;
}
