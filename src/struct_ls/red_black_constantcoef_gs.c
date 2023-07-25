/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"
#include "red_black_gs.h"

#ifndef nalu_hypre_abs
#define nalu_hypre_abs(a)  (((a)>0) ? (a) : -(a))
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_RedBlackConstantCoefGS( void               *relax_vdata,
                              nalu_hypre_StructMatrix *A,
                              nalu_hypre_StructVector *b,
                              nalu_hypre_StructVector *x )
{
   nalu_hypre_RedBlackGSData  *relax_data = (nalu_hypre_RedBlackGSData  *)relax_vdata;

   NALU_HYPRE_Int              max_iter    = (relax_data -> max_iter);
   NALU_HYPRE_Int              zero_guess  = (relax_data -> zero_guess);
   NALU_HYPRE_Int              rb_start    = (relax_data -> rb_start);
   NALU_HYPRE_Int              diag_rank   = (relax_data -> diag_rank);
   nalu_hypre_ComputePkg      *compute_pkg = (relax_data -> compute_pkg);
   NALU_HYPRE_Int              ndim = nalu_hypre_StructMatrixNDim(A);

   nalu_hypre_CommHandle      *comm_handle;

   nalu_hypre_BoxArrayArray   *compute_box_aa;
   nalu_hypre_BoxArray        *compute_box_a;
   nalu_hypre_Box             *compute_box;

   nalu_hypre_Box             *A_dbox;
   nalu_hypre_Box             *b_dbox;
   nalu_hypre_Box             *x_dbox;

   NALU_HYPRE_Int              Ai, Astart, Ani, Anj;
   NALU_HYPRE_Int              bstart, bni, bnj;
   NALU_HYPRE_Int              xstart, xni, xnj;
   NALU_HYPRE_Int              xoff0, xoff1, xoff2, xoff3, xoff4, xoff5;

   NALU_HYPRE_Real            *Ap;
   NALU_HYPRE_Real            *App;
   NALU_HYPRE_Real            *bp;
   NALU_HYPRE_Real            *xp;

   /* constant coefficient */
   NALU_HYPRE_Int              constant_coeff = nalu_hypre_StructMatrixConstantCoefficient(A);
   NALU_HYPRE_Real             App0, App1, App2, App3, App4, App5, AApd;

   nalu_hypre_IndexRef         start;
   nalu_hypre_Index            loop_size;

   nalu_hypre_StructStencil   *stencil;
   nalu_hypre_Index           *stencil_shape;
   NALU_HYPRE_Int              stencil_size;
   NALU_HYPRE_Int              offd[6];

   NALU_HYPRE_Int              iter, rb, redblack, d;
   NALU_HYPRE_Int              compute_i, i, j;
   NALU_HYPRE_Int              ni, nj, nk;

   /*----------------------------------------------------------
    * Initialize some things and deal with special cases
    *----------------------------------------------------------*/

   nalu_hypre_BeginTiming(relax_data -> time_index);

   nalu_hypre_StructMatrixDestroy(relax_data -> A);
   nalu_hypre_StructVectorDestroy(relax_data -> b);
   nalu_hypre_StructVectorDestroy(relax_data -> x);
   (relax_data -> A) = nalu_hypre_StructMatrixRef(A);
   (relax_data -> x) = nalu_hypre_StructVectorRef(x);
   (relax_data -> b) = nalu_hypre_StructVectorRef(b);

   (relax_data -> num_iterations) = 0;

   /* if max_iter is zero, return */
   if (max_iter == 0)
   {
      /* if using a zero initial guess, return zero */
      if (zero_guess)
      {
         nalu_hypre_StructVectorSetConstantValues(x, 0.0);
      }

      nalu_hypre_EndTiming(relax_data -> time_index);
      return nalu_hypre_error_flag;
   }
   else
   {
      stencil       = nalu_hypre_StructMatrixStencil(A);
      stencil_shape = nalu_hypre_StructStencilShape(stencil);
      stencil_size  = nalu_hypre_StructStencilSize(stencil);

      /* get off-diag entry ranks ready */
      i = 0;
      for (j = 0; j < stencil_size; j++)
      {
         if (j != diag_rank)
         {
            offd[i] = j;
            i++;
         }
      }
   }

   nalu_hypre_StructVectorClearBoundGhostValues(x, 0);

   /*----------------------------------------------------------
    * Do zero_guess iteration
    *----------------------------------------------------------*/

   rb = rb_start;
   iter = 0;

   if (zero_guess)
   {
      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
            {
               compute_box_aa = nalu_hypre_ComputePkgIndtBoxes(compute_pkg);
            }
            break;

            case 1:
            {
               compute_box_aa = nalu_hypre_ComputePkgDeptBoxes(compute_pkg);
            }
            break;
         }

         nalu_hypre_ForBoxArrayI(i, compute_box_aa)
         {
            compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), i);
            b_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(b), i);
            x_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);

            Ap = nalu_hypre_StructMatrixBoxData(A, i, diag_rank);
            bp = nalu_hypre_StructVectorBoxData(b, i);
            xp = nalu_hypre_StructVectorBoxData(x, i);

            nalu_hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

               start  = nalu_hypre_BoxIMin(compute_box);
               nalu_hypre_BoxGetSize(compute_box, loop_size);

               /* Are we relaxing index start or start+(1,0,0)? */
               redblack = rb;
               for (d = 0; d < ndim; d++)
               {
                  redblack += nalu_hypre_IndexD(start, d);
               }
               redblack = nalu_hypre_abs(redblack) % 2;

               bstart = nalu_hypre_BoxIndexRank(b_dbox, start);
               xstart = nalu_hypre_BoxIndexRank(x_dbox, start);
               ni = nalu_hypre_IndexX(loop_size);
               nj = nalu_hypre_IndexY(loop_size);
               nk = nalu_hypre_IndexZ(loop_size);
               bni = nalu_hypre_BoxSizeX(b_dbox);
               xni = nalu_hypre_BoxSizeX(x_dbox);
               bnj = nalu_hypre_BoxSizeY(b_dbox);
               xnj = nalu_hypre_BoxSizeY(x_dbox);
               if (ndim < 3)
               {
                  nk = 1;
                  if (ndim < 2)
                  {
                     nj = 1;
                  }
               }

               if (constant_coeff == 1)
               {
                  Ai = nalu_hypre_CCBoxIndexRank(A_dbox, start);
                  AApd = 1.0 / Ap[Ai];

                  nalu_hypre_RedBlackLoopInit();

#define DEVICE_VAR is_device_ptr(xp,bp)
                  nalu_hypre_RedBlackConstantcoefLoopBegin(ni, nj, nk, redblack,
                                                      bstart, bni, bnj, bi,
                                                      xstart, xni, xnj, xi);
                  {
                     xp[xi] = bp[bi] * AApd;
                  }
                  nalu_hypre_RedBlackConstantcoefLoopEnd();
#undef DEVICE_VAR
               }

               else      /* variable coefficient diag */
               {
                  Astart = nalu_hypre_BoxIndexRank(A_dbox, start);
                  Ani = nalu_hypre_BoxSizeX(A_dbox);
                  Anj = nalu_hypre_BoxSizeY(A_dbox);

                  nalu_hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp,Ap)
                  nalu_hypre_RedBlackLoopBegin(ni, nj, nk, redblack,
                                          Astart, Ani, Anj, Ai,
                                          bstart, bni, bnj, bi,
                                          xstart, xni, xnj, xi);
                  {
                     xp[xi] = bp[bi] / Ap[Ai];
                  }
                  nalu_hypre_RedBlackLoopEnd();
#undef DEVICE_VAR
               }

            }
         }
      }

      rb = (rb + 1) % 2;
      iter++;
   }

   /*----------------------------------------------------------
    * Do regular iterations
    *----------------------------------------------------------*/

   while (iter < 2 * max_iter)
   {
      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch (compute_i)
         {
            case 0:
            {
               xp = nalu_hypre_StructVectorData(x);
               nalu_hypre_InitializeIndtComputations(compute_pkg, xp, &comm_handle);
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

         nalu_hypre_ForBoxArrayI(i, compute_box_aa)
         {
            compute_box_a = nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), i);
            b_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(b), i);
            x_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);

            Ap = nalu_hypre_StructMatrixBoxData(A, i, diag_rank);
            bp = nalu_hypre_StructVectorBoxData(b, i);
            xp = nalu_hypre_StructVectorBoxData(x, i);

            nalu_hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

               start  = nalu_hypre_BoxIMin(compute_box);
               nalu_hypre_BoxGetSize(compute_box, loop_size);

               /* Are we relaxing index start or start+(1,0,0)? */
               redblack = rb;
               for (d = 0; d < ndim; d++)
               {
                  redblack += nalu_hypre_IndexD(start, d);
               }
               redblack = nalu_hypre_abs(redblack) % 2;

               bstart = nalu_hypre_BoxIndexRank(b_dbox, start);
               xstart = nalu_hypre_BoxIndexRank(x_dbox, start);
               ni = nalu_hypre_IndexX(loop_size);
               nj = nalu_hypre_IndexY(loop_size);
               nk = nalu_hypre_IndexZ(loop_size);
               bni = nalu_hypre_BoxSizeX(b_dbox);
               xni = nalu_hypre_BoxSizeX(x_dbox);
               bnj = nalu_hypre_BoxSizeY(b_dbox);
               xnj = nalu_hypre_BoxSizeY(x_dbox);
               Ai = nalu_hypre_CCBoxIndexRank(A_dbox, start);
               if (ndim < 3)
               {
                  nk = 1;
                  if (ndim < 2)
                  {
                     nj = 1;
                  }
               }

               switch (stencil_size)
               {
                  case 7:
                     App = nalu_hypre_StructMatrixBoxData(A, i, offd[5]);
                     App5 = App[Ai];
                     App = nalu_hypre_StructMatrixBoxData(A, i, offd[4]);
                     App4 = App[Ai];
                     xoff5 = nalu_hypre_BoxOffsetDistance(
                                x_dbox, stencil_shape[offd[5]]);
                     xoff4 = nalu_hypre_BoxOffsetDistance(
                                x_dbox, stencil_shape[offd[4]]);

                  case 5:
                     App = nalu_hypre_StructMatrixBoxData(A, i, offd[3]);
                     App3 = App[Ai];
                     App = nalu_hypre_StructMatrixBoxData(A, i, offd[2]);
                     App2 = App[Ai];
                     xoff3 = nalu_hypre_BoxOffsetDistance(
                                x_dbox, stencil_shape[offd[3]]);
                     xoff2 = nalu_hypre_BoxOffsetDistance(
                                x_dbox, stencil_shape[offd[2]]);

                  case 3:
                     App = nalu_hypre_StructMatrixBoxData(A, i, offd[1]);
                     App1 = App[Ai];
                     App = nalu_hypre_StructMatrixBoxData(A, i, offd[0]);
                     App0 = App[Ai];
                     xoff1 = nalu_hypre_BoxOffsetDistance(
                                x_dbox, stencil_shape[offd[1]]);
                     xoff0 = nalu_hypre_BoxOffsetDistance(
                                x_dbox, stencil_shape[offd[0]]);
                     break;
               }

               if (constant_coeff == 1)
               {
                  AApd = 1 / Ap[Ai];

                  switch (stencil_size)
                  {
                     case 7:
                        nalu_hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp)
                        nalu_hypre_RedBlackConstantcoefLoopBegin(ni, nj, nk, redblack,
                                                            bstart, bni, bnj, bi,
                                                            xstart, xni, xnj, xi);
                        {
                           xp[xi] =
                              (bp[bi] -
                               App0 * xp[xi + xoff0] -
                               App1 * xp[xi + xoff1] -
                               App2 * xp[xi + xoff2] -
                               App3 * xp[xi + xoff3] -
                               App4 * xp[xi + xoff4] -
                               App5 * xp[xi + xoff5]) * AApd;
                        }
                        nalu_hypre_RedBlackConstantcoefLoopEnd();
#undef DEVICE_VAR

                        break;

                     case 5:
                        nalu_hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp)
                        nalu_hypre_RedBlackConstantcoefLoopBegin(ni, nj, nk, redblack,
                                                            bstart, bni, bnj, bi,
                                                            xstart, xni, xnj, xi);
                        {
                           xp[xi] =
                              (bp[bi] -
                               App0 * xp[xi + xoff0] -
                               App1 * xp[xi + xoff1] -
                               App2 * xp[xi + xoff2] -
                               App3 * xp[xi + xoff3]) * AApd;
                        }
                        nalu_hypre_RedBlackConstantcoefLoopEnd();
#undef DEVICE_VAR
                        break;

                     case 3:
                        nalu_hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp)
                        nalu_hypre_RedBlackConstantcoefLoopBegin(ni, nj, nk, redblack,
                                                            bstart, bni, bnj, bi,
                                                            xstart, xni, xnj, xi);
                        {
                           xp[xi] =
                              (bp[bi] -
                               App0 * xp[xi + xoff0] -
                               App1 * xp[xi + xoff1]) * AApd;
                        }
                        nalu_hypre_RedBlackConstantcoefLoopEnd();
#undef DEVICE_VAR
                        break;
                  }

               }  /* if (constant_coeff == 1) */

               else /* variable diagonal */
               {
                  Astart = nalu_hypre_BoxIndexRank(A_dbox, start);
                  Ani = nalu_hypre_BoxSizeX(A_dbox);
                  Anj = nalu_hypre_BoxSizeY(A_dbox);

                  switch (stencil_size)
                  {
                     case 7:
                        nalu_hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp,Ap)
                        nalu_hypre_RedBlackLoopBegin(ni, nj, nk, redblack,
                                                Astart, Ani, Anj, Ai,
                                                bstart, bni, bnj, bi,
                                                xstart, xni, xnj, xi);
                        {
                           xp[xi] =
                              (bp[bi] -
                               App0 * xp[xi + xoff0] -
                               App1 * xp[xi + xoff1] -
                               App2 * xp[xi + xoff2] -
                               App3 * xp[xi + xoff3] -
                               App4 * xp[xi + xoff4] -
                               App5 * xp[xi + xoff5]) / Ap[Ai];
                        }
                        nalu_hypre_RedBlackLoopEnd();
#undef DEVICE_VAR
                        break;

                     case 5:
                        nalu_hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp,Ap)
                        nalu_hypre_RedBlackLoopBegin(ni, nj, nk, redblack,
                                                Astart, Ani, Anj, Ai,
                                                bstart, bni, bnj, bi,
                                                xstart, xni, xnj, xi);
                        {
                           xp[xi] =
                              (bp[bi] -
                               App0 * xp[xi + xoff0] -
                               App1 * xp[xi + xoff1] -
                               App2 * xp[xi + xoff2] -
                               App3 * xp[xi + xoff3]) / Ap[Ai];
                        }
                        nalu_hypre_RedBlackLoopEnd();
#undef DEVICE_VAR
                        break;

                     case 3:
                        nalu_hypre_RedBlackLoopInit();
#define DEVICE_VAR is_device_ptr(xp,bp,Ap)
                        nalu_hypre_RedBlackLoopBegin(ni, nj, nk, redblack,
                                                Astart, Ani, Anj, Ai,
                                                bstart, bni, bnj, bi,
                                                xstart, xni, xnj, xi);
                        {
                           xp[xi] =
                              (bp[bi] -
                               App0 * xp[xi + xoff0] -
                               App1 * xp[xi + xoff1]) / Ap[Ai];
                        }
                        nalu_hypre_RedBlackLoopEnd();
#undef DEVICE_VAR
                        break;

                  }  /* switch(stencil_size) */
               }     /* else */
            }
         }
      }

      rb = (rb + 1) % 2;
      iter++;
   }

   (relax_data -> num_iterations) = iter / 2;

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   nalu_hypre_IncFLOPCount(relax_data -> flops);
   nalu_hypre_EndTiming(relax_data -> time_index);

   return nalu_hypre_error_flag;
}


