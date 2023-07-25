/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"
#include "pfmg.h"

#ifdef MAX_DEPTH
#undef MAX_DEPTH
#endif
#define MAX_DEPTH 7

/* 2: the most explicit implementation, a function for each stencil size */
#define CC0_IMPLEMENTATION 2

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

nalu_hypre_StructMatrix *
nalu_hypre_PFMGCreateInterpOp( nalu_hypre_StructMatrix *A,
                          nalu_hypre_StructGrid   *cgrid,
                          NALU_HYPRE_Int           cdir,
                          NALU_HYPRE_Int           rap_type )
{
   nalu_hypre_StructMatrix   *P;

   nalu_hypre_StructStencil  *stencil;
   nalu_hypre_Index          *stencil_shape;
   NALU_HYPRE_Int             stencil_size;
   NALU_HYPRE_Int             stencil_dim;

   NALU_HYPRE_Int             num_ghost[] = {1, 1, 1, 1, 1, 1};

   NALU_HYPRE_Int             i;
   NALU_HYPRE_Int             constant_coefficient;

   /* set up stencil */
   stencil_size = 2;
   stencil_dim = nalu_hypre_StructStencilNDim(nalu_hypre_StructMatrixStencil(A));
   stencil_shape = nalu_hypre_CTAlloc(nalu_hypre_Index,  stencil_size, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < stencil_size; i++)
   {
      nalu_hypre_SetIndex3(stencil_shape[i], 0, 0, 0);
   }
   nalu_hypre_IndexD(stencil_shape[0], cdir) = -1;
   nalu_hypre_IndexD(stencil_shape[1], cdir) =  1;
   stencil =
      nalu_hypre_StructStencilCreate(stencil_dim, stencil_size, stencil_shape);

   /* set up matrix */
   P = nalu_hypre_StructMatrixCreate(nalu_hypre_StructMatrixComm(A), cgrid, stencil);
   nalu_hypre_StructMatrixSetNumGhost(P, num_ghost);

   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(A);
   if ( constant_coefficient == 2 )
   {
      if ( rap_type == 0 )
         /* A has variable diagonal, which will force all P coefficients to be variable */
      {
         nalu_hypre_StructMatrixSetConstantCoefficient(P, 0 );
      }
      else
      {
         /* We will force P to be 0.5's everywhere, ignoring A. */
         nalu_hypre_StructMatrixSetConstantCoefficient(P, 1);
      }
   }
   else
   {
      /* constant_coefficient = 0 or 1: A is entirely constant or entirely
         variable coefficient */
      nalu_hypre_StructMatrixSetConstantCoefficient( P, constant_coefficient );
   }

   nalu_hypre_StructStencilDestroy(stencil);

   return P;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetupInterpOp( nalu_hypre_StructMatrix *A,
                         NALU_HYPRE_Int           cdir,
                         nalu_hypre_Index         findex,
                         nalu_hypre_Index         stride,
                         nalu_hypre_StructMatrix *P,
                         NALU_HYPRE_Int           rap_type )
{
   nalu_hypre_BoxArray        *compute_boxes;
   nalu_hypre_Box             *compute_box;

   nalu_hypre_Box             *A_dbox;
   nalu_hypre_Box             *P_dbox;

   NALU_HYPRE_Real            *Pp0, *Pp1;
   NALU_HYPRE_Int              constant_coefficient;

   nalu_hypre_StructStencil   *stencil;
   nalu_hypre_Index           *stencil_shape;
   NALU_HYPRE_Int              stencil_size;
   nalu_hypre_StructStencil   *P_stencil;
   nalu_hypre_Index           *P_stencil_shape;

   NALU_HYPRE_Int              Pstenc0, Pstenc1;

   nalu_hypre_Index            loop_size;
   nalu_hypre_Index            start;
   nalu_hypre_IndexRef         startc;
   nalu_hypre_Index            stridec;

   NALU_HYPRE_Int              i, si;

   NALU_HYPRE_Int              si0, si1;
   NALU_HYPRE_Int              mrk0, mrk1;
   NALU_HYPRE_Int              d;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   stencil       = nalu_hypre_StructMatrixStencil(A);
   stencil_shape = nalu_hypre_StructStencilShape(stencil);
   stencil_size  = nalu_hypre_StructStencilSize(stencil);

   P_stencil       = nalu_hypre_StructMatrixStencil(P);
   P_stencil_shape = nalu_hypre_StructStencilShape(P_stencil);

   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(A);

   /*----------------------------------------------------------
    * Find stencil enties in A corresponding to P
    *----------------------------------------------------------*/

   si0 = -1;
   si1 = -1;
   for (si = 0; si < stencil_size; si++)
   {
      mrk0 = 0;
      mrk1 = 0;
      for (d = 0; d < nalu_hypre_StructStencilNDim(stencil); d++)
      {
         if (nalu_hypre_IndexD(stencil_shape[si], d) ==
             nalu_hypre_IndexD(P_stencil_shape[0], d))
         {
            mrk0++;
         }
         if (nalu_hypre_IndexD(stencil_shape[si], d) ==
             nalu_hypre_IndexD(P_stencil_shape[1], d))
         {
            mrk1++;
         }
      }
      if (mrk0 == nalu_hypre_StructStencilNDim(stencil))
      {
         si0 = si;
      }
      if (mrk1 == nalu_hypre_StructStencilNDim(stencil))
      {
         si1 = si;
      }
   }

   nalu_hypre_SetIndex3(stridec, 1, 1, 1);

   /*----------------------------------------------------------
    * Compute P
    *----------------------------------------------------------*/

   compute_boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(P));
   nalu_hypre_ForBoxI(i, compute_boxes)
   {
      compute_box = nalu_hypre_BoxArrayBox(compute_boxes, i);

      A_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), i);
      P_dbox = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(P), i);

      Pp0 = nalu_hypre_StructMatrixBoxData(P, i, 0);
      Pp1 = nalu_hypre_StructMatrixBoxData(P, i, 1);

      Pstenc0 = nalu_hypre_IndexD(P_stencil_shape[0], cdir);
      Pstenc1 = nalu_hypre_IndexD(P_stencil_shape[1], cdir);

      startc  = nalu_hypre_BoxIMin(compute_box);
      nalu_hypre_StructMapCoarseToFine(startc, findex, stride, start);

      nalu_hypre_BoxGetStrideSize(compute_box, stridec, loop_size);

      if ( constant_coefficient == 1 )
         /* all coefficients are constant */
      {
         nalu_hypre_PFMGSetupInterpOp_CC1
         ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
           P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, si0, si1 );
      }

      else if ( constant_coefficient == 2 )
         /* all coefficients are constant except the diagonal is variable */
      {
         nalu_hypre_PFMGSetupInterpOp_CC2
         ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
           P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, si0, si1 );
      }

      else
         /* constant_coefficient == 0 , all coefficients in A vary */
      {
#if CC0_IMPLEMENTATION <= 1
         nalu_hypre_PFMGSetupInterpOp_CC0
         ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
           P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, si0, si1 );
#else
         switch (stencil_size)
         {
            case 5:
               nalu_hypre_PFMGSetupInterpOp_CC0_SS5
               ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                 P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, P_stencil_shape );
               break;
            case 9:
               nalu_hypre_PFMGSetupInterpOp_CC0_SS9
               ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                 P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, P_stencil_shape );
               break;
            case 7:
               nalu_hypre_PFMGSetupInterpOp_CC0_SS7
               ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                 P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, P_stencil_shape );
               break;
            case 15:
               nalu_hypre_PFMGSetupInterpOp_CC0_SS15
               ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                 P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, P_stencil_shape );
               break;
            case 19:
               nalu_hypre_PFMGSetupInterpOp_CC0_SS19
               ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                 P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, P_stencil_shape );
               break;
            case 27:
               nalu_hypre_PFMGSetupInterpOp_CC0_SS27
               ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                 P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, P_stencil_shape );
               break;
            default:
               /*
               nalu_hypre_PFMGSetupInterpOp_CC0
                  ( i, A, A_dbox, cdir, stride, stridec, start, startc, loop_size,
                    P_dbox, Pstenc0, Pstenc1, Pp0, Pp1, rap_type, si0, si1 );
                */

               nalu_hypre_printf("hypre error: unsupported stencil size %d\n", stencil_size);
               nalu_hypre_MPI_Abort(nalu_hypre_MPI_COMM_WORLD, 1);
         }
#endif
      }
   }

#if 0
   nalu_hypre_StructMatrixAssemble(P);
#else
   nalu_hypre_StructInterpAssemble(A, P, 0, cdir, findex, stride);
#endif

   return nalu_hypre_error_flag;
}

#if CC0_IMPLEMENTATION == 0

NALU_HYPRE_Int
nalu_hypre_PFMGSetupInterpOp_CC0
( NALU_HYPRE_Int           i, /* box index */
  nalu_hypre_StructMatrix *A,
  nalu_hypre_Box          *A_dbox,
  NALU_HYPRE_Int           cdir,
  nalu_hypre_Index         stride,
  nalu_hypre_Index         stridec,
  nalu_hypre_Index         start,
  nalu_hypre_IndexRef      startc,
  nalu_hypre_Index         loop_size,
  nalu_hypre_Box          *P_dbox,
  NALU_HYPRE_Int           Pstenc0,
  NALU_HYPRE_Int           Pstenc1,
  NALU_HYPRE_Real         *Pp0,
  NALU_HYPRE_Real         *Pp1,
  NALU_HYPRE_Int           rap_type,
  NALU_HYPRE_Int           si0,
  NALU_HYPRE_Int           si1 )
{
   nalu_hypre_StructStencil *stencil = nalu_hypre_StructMatrixStencil(A);
   nalu_hypre_Index         *stencil_shape = nalu_hypre_StructStencilShape(stencil);
   NALU_HYPRE_Int            stencil_size = nalu_hypre_StructStencilSize(stencil);
   NALU_HYPRE_Int            warning_cnt = 0;
   NALU_HYPRE_Int            data_location = nalu_hypre_StructGridDataLocation(nalu_hypre_StructMatrixGrid(A));
   NALU_HYPRE_Int          **data_indices = nalu_hypre_StructMatrixDataIndices(A);
   NALU_HYPRE_Complex       *matrixA_data = nalu_hypre_StructMatrixData(A);
   NALU_HYPRE_Int           *data_indices_boxi_d;
   nalu_hypre_Index         *stencil_shape_d;
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_StructMatrixMemoryLocation(A);

   if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
   {
      data_indices_boxi_d = nalu_hypre_TAlloc(NALU_HYPRE_Int, stencil_size, memory_location);
      stencil_shape_d = nalu_hypre_TAlloc(nalu_hypre_Index, stencil_size, memory_location);
      nalu_hypre_TMemcpy(data_indices_boxi_d, data_indices[i], NALU_HYPRE_Int, stencil_size, memory_location,
                    NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TMemcpy(stencil_shape_d, stencil_shape, nalu_hypre_Index, stencil_size, memory_location,
                    NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      data_indices_boxi_d = data_indices[i];
      stencil_shape_d = stencil_shape;
   }

#define DEVICE_VAR is_device_ptr(Pp0,Pp1,matrixA_data,stencil_shape_d,data_indices_boxi_d)
   nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start,  stride,  Ai,
                       P_dbox, startc, stridec, Pi);
   {
      NALU_HYPRE_Int si, mrk0, mrk1, Astenc;
      NALU_HYPRE_Real center;
      NALU_HYPRE_Real *Ap;

      center  = 0.0;
      Pp0[Pi] = 0.0;
      Pp1[Pi] = 0.0;
      mrk0 = 0;
      mrk1 = 0;

      for (si = 0; si < stencil_size; si++)
      {
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         if (data_location != NALU_HYPRE_MEMORY_HOST)
         {
            Ap     = matrixA_data + data_indices_boxi_d[si];
            Astenc = nalu_hypre_IndexD(stencil_shape_d[si], cdir);
         }
         else
         {
            Ap     = nalu_hypre_StructMatrixBoxData(A, i, si);
            Astenc = nalu_hypre_IndexD(stencil_shape[si], cdir);
         }
#else
         Ap     = matrixA_data + data_indices_boxi_d[si];
         Astenc = nalu_hypre_IndexD(stencil_shape_d[si], cdir);
#endif

         if (Astenc == 0)
         {
            center += Ap[Ai];
         }
         else if (Astenc == Pstenc0)
         {
            Pp0[Pi] -= Ap[Ai];
         }
         else if (Astenc == Pstenc1)
         {
            Pp1[Pi] -= Ap[Ai];
         }

         if (si == si0 && Ap[Ai] == 0.0)
         {
            mrk0++;
         }
         if (si == si1 && Ap[Ai] == 0.0)
         {
            mrk1++;
         }
      }

      if (!center)
      {
         //warning_cnt++;
         Pp0[Pi] = 0.0;
         Pp1[Pi] = 0.0;
      }
      else
      {
         Pp0[Pi] /= center;
         Pp1[Pi] /= center;
      }

      /*----------------------------------------------
       * Set interpolation weight to zero, if stencil
       * entry in same direction is zero. Prevents
       * interpolation and operator stencils reaching
       * outside domain.
       *----------------------------------------------*/
      if (mrk0 != 0)
      {
         Pp0[Pi] = 0.0;
      }
      if (mrk1 != 0)
      {
         Pp1[Pi] = 0.0;
      }
   }
   nalu_hypre_BoxLoop2End(Ai, Pi);
#undef DEVICE_VAR

   if (warning_cnt)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Warning 0 center in interpolation. Setting interp = 0.");
   }

   if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_TFree(data_indices_boxi_d, memory_location);
      nalu_hypre_TFree(stencil_shape_d, memory_location);
   }

   return nalu_hypre_error_flag;
}

#endif

#if CC0_IMPLEMENTATION == 1

NALU_HYPRE_Int
nalu_hypre_PFMGSetupInterpOp_CC0
( NALU_HYPRE_Int           i, /* box index */
  nalu_hypre_StructMatrix *A,
  nalu_hypre_Box          *A_dbox,
  NALU_HYPRE_Int           cdir,
  nalu_hypre_Index         stride,
  nalu_hypre_Index         stridec,
  nalu_hypre_Index         start,
  nalu_hypre_IndexRef      startc,
  nalu_hypre_Index         loop_size,
  nalu_hypre_Box          *P_dbox,
  NALU_HYPRE_Int           Pstenc0,
  NALU_HYPRE_Int           Pstenc1,
  NALU_HYPRE_Real         *Pp0,
  NALU_HYPRE_Real         *Pp1,
  NALU_HYPRE_Int           rap_type,
  NALU_HYPRE_Int           si0,
  NALU_HYPRE_Int           si1 )
{
   nalu_hypre_StructStencil   *stencil = nalu_hypre_StructMatrixStencil(A);
   nalu_hypre_Index           *stencil_shape = nalu_hypre_StructStencilShape(stencil);
   NALU_HYPRE_Int              stencil_size = nalu_hypre_StructStencilSize(stencil);
   NALU_HYPRE_Int              warning_cnt = 0;
   NALU_HYPRE_Int              dim, si, loop_length = 1, Astenc;
   NALU_HYPRE_Real            *Ap, *center, *Ap0, *Ap1;
   NALU_HYPRE_MemoryLocation   memory_location = nalu_hypre_StructMatrixMemoryLocation(A);

   for (dim = 0; dim < nalu_hypre_StructMatrixNDim(A); dim++)
   {
      loop_length *= loop_size[dim];
   }
   center = nalu_hypre_CTAlloc(NALU_HYPRE_Real, loop_length, memory_location);

   for (si = 0; si < stencil_size; si++)
   {
      Ap     = nalu_hypre_StructMatrixBoxData(A, i, si);
      Astenc = nalu_hypre_IndexD(stencil_shape[si], cdir);

      if (Astenc == 0)
      {
#define DEVICE_VAR is_device_ptr(center, Ap)
         nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                             A_dbox, start,  stride,  Ai,
                             P_dbox, startc, stridec, Pi)
         center[idx] += Ap[Ai];
         nalu_hypre_BoxLoop2End(Ai, Pi)
#undef DEVICE_VAR
      }
      else if (Astenc == Pstenc0)
      {
#define DEVICE_VAR is_device_ptr(Pp0, Ap)
         nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                             A_dbox, start,  stride,  Ai,
                             P_dbox, startc, stridec, Pi)
         Pp0[Pi] -= Ap[Ai];
         nalu_hypre_BoxLoop2End(Ai, Pi)
#undef DEVICE_VAR
      }
      else if (Astenc == Pstenc1)
      {
#define DEVICE_VAR is_device_ptr(Pp1, Ap)
         nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                             A_dbox, start,  stride,  Ai,
                             P_dbox, startc, stridec, Pi)
         Pp1[Pi] -= Ap[Ai];
         nalu_hypre_BoxLoop2End(Ai, Pi)
#undef DEVICE_VAR
      }
   }

   Ap0 = nalu_hypre_StructMatrixBoxData(A, i, si0);
   Ap1 = nalu_hypre_StructMatrixBoxData(A, i, si1);
#define DEVICE_VAR is_device_ptr(center, Pp0, Pp1, Ap0, Ap1)
   nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start,  stride,  Ai,
                       P_dbox, startc, stridec, Pi)
   NALU_HYPRE_Real cval = center[idx];
   if (Ap0[Ai] == 0.0 || cval == 0.0)
   {
      Pp0[Pi] = 0.0;
   }
   else
   {
      Pp0[Pi] /= cval;
   }

   if (Ap1[Ai] == 0.0 || cval == 0.0)
   {
      Pp1[Pi] = 0.0;
   }
   else
   {
      Pp1[Pi] /= cval;
   }
   nalu_hypre_BoxLoop2End(Ai, Pi)
#undef DEVICE_VAR

   if (warning_cnt)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Warning 0 center in interpolation. Setting interp = 0.");
   }

   nalu_hypre_TFree(center, memory_location);

   return nalu_hypre_error_flag;
}

#endif


NALU_HYPRE_Int
nalu_hypre_PFMGSetupInterpOp_CC1
( NALU_HYPRE_Int           i, /* box index, doesn't matter */
  nalu_hypre_StructMatrix *A,
  nalu_hypre_Box          *A_dbox,
  NALU_HYPRE_Int           cdir,
  nalu_hypre_Index         stride,
  nalu_hypre_Index         stridec,
  nalu_hypre_Index         start,
  nalu_hypre_IndexRef      startc,
  nalu_hypre_Index         loop_size,
  nalu_hypre_Box          *P_dbox,
  NALU_HYPRE_Int           Pstenc0,
  NALU_HYPRE_Int           Pstenc1,
  NALU_HYPRE_Real         *Pp0,
  NALU_HYPRE_Real         *Pp1,
  NALU_HYPRE_Int           rap_type,
  NALU_HYPRE_Int           si0,
  NALU_HYPRE_Int           si1 )
{
   NALU_HYPRE_Int              si;
   NALU_HYPRE_Int              Ai, Pi;
   NALU_HYPRE_Real            *Ap;
   NALU_HYPRE_Real             center;
   NALU_HYPRE_Int              Astenc;
   NALU_HYPRE_Int              mrk0, mrk1;
   nalu_hypre_StructStencil   *stencil = nalu_hypre_StructMatrixStencil(A);
   nalu_hypre_Index           *stencil_shape = nalu_hypre_StructStencilShape(stencil);
   NALU_HYPRE_Int              stencil_size = nalu_hypre_StructStencilSize(stencil);
   NALU_HYPRE_Int              warning_cnt = 0;

   Ai = nalu_hypre_CCBoxIndexRank(A_dbox, start );
   Pi = nalu_hypre_CCBoxIndexRank(P_dbox, startc);

   center  = 0.0;
   Pp0[Pi] = 0.0;
   Pp1[Pi] = 0.0;
   mrk0 = 0;
   mrk1 = 0;

   for (si = 0; si < stencil_size; si++)
   {
      Ap     = nalu_hypre_StructMatrixBoxData(A, i, si);
      Astenc = nalu_hypre_IndexD(stencil_shape[si], cdir);

      if (Astenc == 0)
      {
         center += Ap[Ai];
      }
      else if (Astenc == Pstenc0)
      {
         Pp0[Pi] -= Ap[Ai];
      }
      else if (Astenc == Pstenc1)
      {
         Pp1[Pi] -= Ap[Ai];
      }

      if (si == si0 && Ap[Ai] == 0.0)
      {
         mrk0++;
      }
      if (si == si1 && Ap[Ai] == 0.0)
      {
         mrk1++;
      }
   }
   if (!center)
   {
      warning_cnt++;
      Pp0[Pi] = 0.0;
      Pp1[Pi] = 0.0;
   }
   else
   {
      Pp0[Pi] /= center;
      Pp1[Pi] /= center;
   }

   /*----------------------------------------------
    * Set interpolation weight to zero, if stencil
    * entry in same direction is zero.
    * For variable coefficients, this was meant to prevent
    * interpolation and operator stencils from reaching
    * outside the domain.
    * For constant coefficients it will hardly ever happen
    * (means the stencil point shouldn't have been defined there)
    * but it's possible and then it would still make sense to
    * do this.
    *----------------------------------------------*/
   if (mrk0 != 0)
   {
      Pp0[Pi] = 0.0;
   }
   if (mrk1 != 0)
   {
      Pp1[Pi] = 0.0;
   }

   if (warning_cnt)
   {
      nalu_hypre_error_w_msg(
         NALU_HYPRE_ERROR_GENERIC,
         "Warning 0 center in interpolation. Setting interp = 0.");
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGSetupInterpOp_CC2
( NALU_HYPRE_Int           i, /* box index */
  nalu_hypre_StructMatrix *A,
  nalu_hypre_Box          *A_dbox,
  NALU_HYPRE_Int           cdir,
  nalu_hypre_Index         stride,
  nalu_hypre_Index         stridec,
  nalu_hypre_Index         start,
  nalu_hypre_IndexRef      startc,
  nalu_hypre_Index         loop_size,
  nalu_hypre_Box          *P_dbox,
  NALU_HYPRE_Int           Pstenc0,
  NALU_HYPRE_Int           Pstenc1,
  NALU_HYPRE_Real         *Pp0,
  NALU_HYPRE_Real         *Pp1,
  NALU_HYPRE_Int           rap_type,
  NALU_HYPRE_Int           si0,
  NALU_HYPRE_Int           si1 )
{
   NALU_HYPRE_Int              si;
   NALU_HYPRE_Int              Ai;
   NALU_HYPRE_Int              Pi;
   NALU_HYPRE_Real            *Ap;
   NALU_HYPRE_Real             P0, P1;
   NALU_HYPRE_Real             center_offd;
   NALU_HYPRE_Int              Astenc;
   NALU_HYPRE_Int              mrk0_offd, mrk1_offd;
   nalu_hypre_StructStencil   *stencil = nalu_hypre_StructMatrixStencil(A);
   nalu_hypre_Index           *stencil_shape = nalu_hypre_StructStencilShape(stencil);
   NALU_HYPRE_Int              stencil_size = nalu_hypre_StructStencilSize(stencil);
   nalu_hypre_Index            diag_index;
   NALU_HYPRE_Int              diag_rank;
   NALU_HYPRE_Int              warning_cnt = 0;

   nalu_hypre_SetIndex3(diag_index, 0, 0, 0);
   diag_rank = nalu_hypre_StructStencilElementRank(stencil, diag_index);

   if ( rap_type != 0 )
   {
      /* simply force P to be constant coefficient, all 0.5's */
      Pi = nalu_hypre_CCBoxIndexRank(P_dbox, startc);
      Pp0[Pi] = 0.5;
      Pp1[Pi] = 0.5;
   }
   else
   {
      /* Most coeffients of A go into P like for constant_coefficient=1.
         But P is entirely variable coefficient, because the diagonal of A is
         variable, and hence "center" below is variable. So we use the constant
         coefficient calculation to initialize the diagonal's variable
         coefficient calculation (which is like constant_coefficient=0). */
      Ai = nalu_hypre_CCBoxIndexRank(A_dbox, start );

      center_offd  = 0.0;
      P0 = 0.0;
      P1 = 0.0;
      mrk0_offd = 0;
      mrk1_offd = 0;

      for (si = 0; si < stencil_size; si++)
      {
         if ( si != diag_rank )
         {
            Ap = nalu_hypre_StructMatrixBoxData(A, i, si);
            Astenc = nalu_hypre_IndexD(stencil_shape[si], cdir);

            if (Astenc == 0)
            {
               center_offd += Ap[Ai];
            }
            else if (Astenc == Pstenc0)
            {
               P0 -= Ap[Ai];
            }
            else if (Astenc == Pstenc1)
            {
               P1 -= Ap[Ai];
            }

            if (si == si0 && Ap[Ai] == 0.0)
            {
               mrk0_offd++;
            }
            if (si == si1 && Ap[Ai] == 0.0)
            {
               mrk1_offd++;
            }
         }
      }

      si = diag_rank;

      NALU_HYPRE_Real *Ap = nalu_hypre_StructMatrixBoxData(A, i, si);

#define DEVICE_VAR is_device_ptr(Pp0,Pp1,Ap)
      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          A_dbox, start, stride, Ai,
                          P_dbox, startc, stridec, Pi);
      {
         NALU_HYPRE_Int   mrk0, mrk1;
         NALU_HYPRE_Real  center;
         NALU_HYPRE_Real  p0val, p1val;

         p0val = P0;
         p1val = P1;
         center = center_offd;
         mrk0 = mrk0_offd;
         mrk1 = mrk1_offd;

         /* RL: Astenc is only needed for assertion, comment out
            Astenc = nalu_hypre_IndexD(stencil_shape[si], cdir);
            nalu_hypre_assert( Astenc==0 );
         */

         center += Ap[Ai];

         //if (si == si0 && Ap[Ai] == 0.0)
         //   mrk0++;
         //if (si == si1 && Ap[Ai] == 0.0)
         //   mrk1++;

         if (!center)
         {
            //warning_cnt++;
            p0val = 0.0;
            p1val = 0.0;
         }
         else
         {
            p0val /= center;
            p1val /= center;
         }

         /*----------------------------------------------
          * Set interpolation weight to zero, if stencil
          * entry in same direction is zero. Prevents
          * interpolation and operator stencils reaching
          * outside domain.
          *----------------------------------------------*/
         if (mrk0 != 0)
         {
            p0val = 0.0;
         }
         if (mrk1 != 0)
         {
            p1val = 0.0;
         }
         Pp0[Pi] = p0val;
         Pp1[Pi] = p1val;

      }
      nalu_hypre_BoxLoop2End(Ai, Pi);
#undef DEVICE_VAR
   }

   if (warning_cnt)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Warning 0 center in interpolation. Setting interp = 0.");
   }

   return nalu_hypre_error_flag;
}

#if CC0_IMPLEMENTATION > 1

NALU_HYPRE_Int
nalu_hypre_PFMGSetupInterpOp_CC0_SS5
( NALU_HYPRE_Int           i, /* box index */
  nalu_hypre_StructMatrix *A,
  nalu_hypre_Box          *A_dbox,
  NALU_HYPRE_Int           cdir,
  nalu_hypre_Index         stride,
  nalu_hypre_Index         stridec,
  nalu_hypre_Index         start,
  nalu_hypre_IndexRef      startc,
  nalu_hypre_Index         loop_size,
  nalu_hypre_Box          *P_dbox,
  NALU_HYPRE_Int           Pstenc0,
  NALU_HYPRE_Int           Pstenc1,
  NALU_HYPRE_Real         *Pp0,
  NALU_HYPRE_Real         *Pp1,
  NALU_HYPRE_Int           rap_type,
  nalu_hypre_Index        *P_stencil_shape )
{
   //nalu_hypre_StructStencil   *stencil = nalu_hypre_StructMatrixStencil(A);
   //nalu_hypre_Index           *stencil_shape = nalu_hypre_StructStencilShape(stencil);
   //NALU_HYPRE_Int              stencil_size = nalu_hypre_StructStencilSize(stencil);
   //NALU_HYPRE_Int              warning_cnt= 0;

   nalu_hypre_Index            index;
   NALU_HYPRE_Real            *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   NALU_HYPRE_Real            *p0, *p1;

   p0 = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[0]);
   p1 = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[1]);

   /*-----------------------------------------------------------------
    * Extract pointers for 5-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index, 0, 0, 0);
   a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 0, 0);
   a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 0, 0);
   a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, -1, 0);
   a_cs = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 1, 0);
   a_cn = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

#define DEVICE_VAR is_device_ptr(a_cc,a_cs,a_cn,a_cw,a_ce,Pp0,Pp1,p0,p1)
   nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start, stride, Ai,
                       P_dbox, startc, stridec, Pi);
   {
      NALU_HYPRE_Real center, left, right;

      switch (cdir)
      {
         case 0:
            center = a_cc[Ai] + a_cs[Ai] + a_cn[Ai];
            left   = -a_cw[Ai];
            right  = -a_ce[Ai];
            break;
         case 1:
            center = a_cc[Ai] + a_cw[Ai] + a_ce[Ai];
            left   = -a_cs[Ai];
            right  = -a_cn[Ai];
            break;
      }

      if (!center)
      {
         //warning_cnt++;
         Pp0[Pi] = 0.0;
         Pp1[Pi] = 0.0;
      }
      else
      {
         switch (Pstenc0)
         {
            case -1:
               Pp0[Pi] = left / center;
               break;
            case  1:
               Pp0[Pi] = right / center;
               break;
         }

         switch (Pstenc1)
         {
            case -1:
               Pp1[Pi] = left / center;
               break;
            case  1:
               Pp1[Pi] = right / center;
               break;
         }
      }

      if (p0[Ai] == 0.0) { Pp0[Pi] = 0.0; }
      if (p1[Ai] == 0.0) { Pp1[Pi] = 0.0; }
      /*----------------------------------------------
       * Set interpolation weight to zero, if stencil
       * entry in same direction is zero. Prevents
       * interpolation and operator stencils reaching
       * outside domain.
       *----------------------------------------------*/
      //if (mrk0 != 0)
      //   Pp0[Pi] = 0.0;
      //if (mrk1 != 0)
      //   Pp1[Pi] = 0.0;
   }
   nalu_hypre_BoxLoop2End(Ai, Pi);
#undef DEVICE_VAR

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGSetupInterpOp_CC0_SS9
( NALU_HYPRE_Int           i, /* box index */
  nalu_hypre_StructMatrix *A,
  nalu_hypre_Box          *A_dbox,
  NALU_HYPRE_Int           cdir,
  nalu_hypre_Index         stride,
  nalu_hypre_Index         stridec,
  nalu_hypre_Index         start,
  nalu_hypre_IndexRef      startc,
  nalu_hypre_Index         loop_size,
  nalu_hypre_Box          *P_dbox,
  NALU_HYPRE_Int           Pstenc0,
  NALU_HYPRE_Int           Pstenc1,
  NALU_HYPRE_Real         *Pp0,
  NALU_HYPRE_Real         *Pp1,
  NALU_HYPRE_Int           rap_type,
  nalu_hypre_Index        *P_stencil_shape )
{
   //nalu_hypre_StructStencil   *stencil = nalu_hypre_StructMatrixStencil(A);
   //nalu_hypre_Index           *stencil_shape = nalu_hypre_StructStencilShape(stencil);
   //NALU_HYPRE_Int              stencil_size = nalu_hypre_StructStencilSize(stencil);
   //NALU_HYPRE_Int              warning_cnt= 0;

   nalu_hypre_Index            index;
   NALU_HYPRE_Real            *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   NALU_HYPRE_Real            *a_csw, *a_cse, *a_cne, *a_cnw;
   NALU_HYPRE_Real            *p0, *p1;

   p0 = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[0]);
   p1 = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[1]);
   /*-----------------------------------------------------------------
    * Extract pointers for 5-point grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient
    * a_ce is pointer for east coefficient
    * a_cs is pointer for south coefficient
    * a_cn is pointer for north coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index, 0, 0, 0);
   a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 0, 0);
   a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 0, 0);
   a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, -1, 0);
   a_cs = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 1, 0);
   a_cn = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 9-point grid operator:
    *
    * a_csw is pointer for southwest coefficient
    * a_cse is pointer for southeast coefficient
    * a_cnw is pointer for northwest coefficient
    * a_cne is pointer for northeast coefficient
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index, -1, -1, 0);
   a_csw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, -1, 0);
   a_cse = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 1, 0);
   a_cnw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 1, 0);
   a_cne = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

#define DEVICE_VAR is_device_ptr(a_cc,a_cs,a_cn,a_cw,a_csw,a_cnw,a_ce,a_cse,a_cne,Pp0,Pp1,p0,p1)
   nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start, stride, Ai,
                       P_dbox, startc, stridec, Pi);
   {
      NALU_HYPRE_Real center, left, right;

      switch (cdir)
      {
         case 0:
            center = a_cc[Ai] +  a_cs[Ai] +  a_cn[Ai];
            left   = -a_cw[Ai] - a_csw[Ai] - a_cnw[Ai];
            right  = -a_ce[Ai] - a_cse[Ai] - a_cne[Ai];
            break;
         case 1:
            center = a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai];
            left   = -a_cs[Ai] - a_csw[Ai] - a_cse[Ai];
            right  = -a_cn[Ai] - a_cnw[Ai] - a_cne[Ai];
            break;
      };

      if (!center)
      {
         //warning_cnt++;
         Pp0[Pi] = 0.0;
         Pp1[Pi] = 0.0;
      }
      else
      {
         switch (Pstenc0)
         {
            case -1:
               Pp0[Pi] = left / center;
               Pp1[Pi] = right / center;
               break;
            case 1:
               Pp0[Pi] = right / center;
               Pp1[Pi] = left / center;
               break;
         };
         /*
            switch (Pstenc1)
            {
            case -1:
            Pp1[Pi] = left/center;break;
            case 1:
            Pp1[Pi] = right/center;break;
            };
            */
      }

      if (p0[Ai] == 0.0) { Pp0[Pi] = 0.0; }
      if (p1[Ai] == 0.0) { Pp1[Pi] = 0.0; }
   }
   nalu_hypre_BoxLoop2End(Ai, Pi);
#undef DEVICE_VAR

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGSetupInterpOp_CC0_SS7
( NALU_HYPRE_Int           i, /* box index */
  nalu_hypre_StructMatrix *A,
  nalu_hypre_Box          *A_dbox,
  NALU_HYPRE_Int           cdir,
  nalu_hypre_Index         stride,
  nalu_hypre_Index         stridec,
  nalu_hypre_Index         start,
  nalu_hypre_IndexRef      startc,
  nalu_hypre_Index         loop_size,
  nalu_hypre_Box          *P_dbox,
  NALU_HYPRE_Int           Pstenc0,
  NALU_HYPRE_Int           Pstenc1,
  NALU_HYPRE_Real         *Pp0,
  NALU_HYPRE_Real         *Pp1,
  NALU_HYPRE_Int           rap_type,
  nalu_hypre_Index        *P_stencil_shape )
{
   //nalu_hypre_StructStencil   *stencil = nalu_hypre_StructMatrixStencil(A);
   //nalu_hypre_Index           *stencil_shape = nalu_hypre_StructStencilShape(stencil);
   //NALU_HYPRE_Int              stencil_size = nalu_hypre_StructStencilSize(stencil);
   //NALU_HYPRE_Int              warning_cnt= 0;

   nalu_hypre_Index            index;
   NALU_HYPRE_Real            *a_cc, *a_cw, *a_ce, *a_cs, *a_cn, *a_ac, *a_bc;
   NALU_HYPRE_Real            *p0, *p1;

   p0 = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[0]);
   p1 = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[1]);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index, 0, 0, 0);
   a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 0, 0);
   a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 0, 0);
   a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, -1, 0);
   a_cs = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 1, 0);
   a_cn = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 0, 1);
   a_ac = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 0, -1);
   a_bc = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

#define DEVICE_VAR is_device_ptr(a_cc,a_cs,a_cn,a_ac,a_bc,a_cw,a_ce,Pp0,Pp1,p0,p1)
   nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start, stride, Ai,
                       P_dbox, startc, stridec, Pi);
   {
      NALU_HYPRE_Real center, left, right;

      switch (cdir)
      {
         case 0:
            center = a_cc[Ai] +  a_cs[Ai] + a_cn[Ai] + a_ac[Ai] + a_bc[Ai];
            left   = -a_cw[Ai];
            right  = -a_ce[Ai];
            break;
         case 1:
            center = a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai] + a_ac[Ai] + a_bc[Ai] ;
            left   = -a_cs[Ai];
            right  = -a_cn[Ai];
            break;
         case 2:
            center = a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai] + a_cs[Ai] + a_cn[Ai] ;
            left   = -a_bc[Ai];
            right  = -a_ac[Ai];
            break;
      };

      if (!center)
      {
         Pp0[Pi] = 0.0;
         Pp1[Pi] = 0.0;
      }
      else
      {
         switch (Pstenc0)
         {
            case -1:
               Pp0[Pi] = left / center;
               Pp1[Pi] = right / center;
               break;
            case 1:
               Pp0[Pi] = right / center;
               Pp1[Pi] = left / center;
               break;
         };
         /*
            switch (Pstenc1)
            {
            case -1:
            Pp1[Pi] = left/center;break;
            case 1:
            Pp1[Pi] = right/center;break;
            };
            */
      }

      if (p0[Ai] == 0.0) { Pp0[Pi] = 0.0; }
      if (p1[Ai] == 0.0) { Pp1[Pi] = 0.0; }

      //printf("%d: %d, Pp0[%d] = %e, Pp1 = %e, %e, %e, %e, cc=%e, cw=%e, ce=%e, cs=%e, cn=%e, bc=%e, ac=%e \n",Ai,cdir, Pi,Pp0[Pi],Pp1[Pi],center, left, right,
      //     a_cc[Ai],a_cw[Ai],a_ce[Ai],a_cs[Ai],a_cn[Ai],a_bc[Ai],a_ac[Ai]);
   }
   nalu_hypre_BoxLoop2End(Ai, Pi);
#undef DEVICE_VAR

   return nalu_hypre_error_flag;
}


NALU_HYPRE_Int
nalu_hypre_PFMGSetupInterpOp_CC0_SS15
( NALU_HYPRE_Int           i, /* box index */
  nalu_hypre_StructMatrix *A,
  nalu_hypre_Box          *A_dbox,
  NALU_HYPRE_Int           cdir,
  nalu_hypre_Index         stride,
  nalu_hypre_Index         stridec,
  nalu_hypre_Index         start,
  nalu_hypre_IndexRef      startc,
  nalu_hypre_Index         loop_size,
  nalu_hypre_Box          *P_dbox,
  NALU_HYPRE_Int           Pstenc0,
  NALU_HYPRE_Int           Pstenc1,
  NALU_HYPRE_Real         *Pp0,
  NALU_HYPRE_Real         *Pp1,
  NALU_HYPRE_Int           rap_type,
  nalu_hypre_Index        *P_stencil_shape )
{
   nalu_hypre_Index           index;
   NALU_HYPRE_Int             stencil_type15;
   NALU_HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn, *a_ac, *a_bc;
   NALU_HYPRE_Real           *a_aw, *a_ae, *a_as, *a_an, *a_bw, *a_be, *a_bs, *a_bn;
   NALU_HYPRE_Real           *a_csw, *a_cse, *a_cnw, *a_cne;
   NALU_HYPRE_Real           *p0, *p1;

   p0 = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[0]);
   p1 = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[1]);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index, 0, 0, 0);
   a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 0, 0);
   a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 0, 0);
   a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, -1, 0);
   a_cs = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 1, 0);
   a_cn = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 0, 1);
   a_ac = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 0, -1);
   a_bc = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 15-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index, -1, 0, 1);
   a_aw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 0, 1);
   a_ae = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, -1, 1);
   a_as = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 1, 1);
   a_an = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 0, -1);
   a_bw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 0, -1);
   a_be = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, -1, -1);
   a_bs = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 1, -1);
   a_bn = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, -1, 0);
   a_csw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, -1, 0);
   a_cse = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 1, 0);
   a_cnw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 1, 0);
   a_cne = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   if (a_csw)
   {
      if (a_as)
      {
         stencil_type15 = 1;
      }
      else
      {
         stencil_type15 = 0;
      }
   }
   else
   {
      stencil_type15 = 2;
   }

   //printf("loop_size %d %d %d, cdir %d, %p %p %p %p %p %p %p %p %p %p %p %p %p %p %p\n", loop_size[0], loop_size[1], loop_size[2], cdir, a_cc, a_cw, a_ce, a_ac, a_bc, a_cs, a_as, a_bs, a_csw, a_cse, a_cn, a_an, a_bn, a_cnw, a_cne);

#define DEVICE_VAR is_device_ptr(a_cc,a_cs,a_cn,a_ac,a_bc,a_as,a_an,a_bs,a_bn,a_cw,a_aw,a_bw,a_ce,a_ae,a_be,a_cnw,a_cne,a_csw,a_cse,Pp0,Pp1,p0,p1)
   if (stencil_type15 == 0)
   {
      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          A_dbox, start, stride, Ai,
                          P_dbox, startc, stridec, Pi);
      {
         NALU_HYPRE_Real center, left, right;

         switch (cdir)
         {
            case 0:
               center =  a_cc[Ai] + a_cs[Ai] + a_cn[Ai] +  a_ac[Ai] +  a_bc[Ai];
               left   = -a_cw[Ai] - a_aw[Ai] - a_bw[Ai] - a_csw[Ai] - a_cnw[Ai];
               right  = -a_ce[Ai] - a_ae[Ai] - a_be[Ai] - a_cse[Ai] - a_cne[Ai];
               break;
            case 1:
               center =  a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai] +  a_ac[Ai] +  a_aw[Ai] + a_ae[Ai] +
                         a_bc[Ai] +  a_bw[Ai] +  a_be[Ai];
               left   = -a_cs[Ai] - a_csw[Ai] - a_cse[Ai]; /* front */
               right  = -a_cn[Ai] - a_cnw[Ai] - a_cne[Ai]; /* back */
               break;
            case 2:
               center =   a_cc[Ai] +  a_cw[Ai] +   a_ce[Ai] +  a_cs[Ai] + a_cn[Ai] +
                          a_csw[Ai] + a_cse[Ai] +  a_cnw[Ai] - a_cne[Ai];
               left   =  -a_bc[Ai] -  a_bw[Ai] -   a_be[Ai]; /* below */
               right  =  -a_ac[Ai] -  a_aw[Ai] -   a_ae[Ai]; /* above */
               break;
         }

         if (!center)
         {
            Pp0[Pi] = 0.0;
            Pp1[Pi] = 0.0;
         }
         else
         {
            switch (Pstenc0)
            {
               case -1:
                  Pp0[Pi] = left  / center;
                  Pp1[Pi] = right / center;
                  break;
               case 1:
                  Pp0[Pi] = right / center;
                  Pp1[Pi] = left  / center;
                  break;
            }
         }

         if (p0[Ai] == 0.0)
         {
            Pp0[Pi] = 0.0;
         }
         if (p1[Ai] == 0.0)
         {
            Pp1[Pi] = 0.0;
         }
      }
      nalu_hypre_BoxLoop2End(Ai, Pi);
   }
   else if (stencil_type15 == 1)
   {
      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          A_dbox, start, stride, Ai,
                          P_dbox, startc, stridec, Pi);
      {
         NALU_HYPRE_Real center, left, right;

         switch (cdir)
         {
            case 0:
               center =  a_cc[Ai] + a_cs[Ai] + a_cn[Ai] +  a_ac[Ai] +  a_as[Ai] + a_an[Ai] +
                         a_bc[Ai] + a_bs[Ai] + a_bn[Ai];
               left   = -a_cw[Ai] - a_csw[Ai] - a_cnw[Ai];
               right  = -a_ce[Ai] - a_cse[Ai] - a_cne[Ai];
               break;
            case 1:
               center =  a_cc[Ai] + a_cw[Ai] + a_ce[Ai] +  a_ac[Ai] +  a_bc[Ai];
               left   = -a_cs[Ai] - a_as[Ai] - a_bs[Ai] - a_csw[Ai] - a_cse[Ai]; /* front */
               right  = -a_cn[Ai] - a_an[Ai] - a_bn[Ai] - a_cnw[Ai] - a_cne[Ai]; /* back */
               break;
            case 2:
               center =  a_cc[Ai] + a_cw[Ai] + a_ce[Ai] + a_cs[Ai] + a_cn[Ai] +
                         a_csw[Ai] + a_cse[Ai] + a_cnw[Ai] + a_cne[Ai];
               left   = -a_bc[Ai] - a_bs[Ai] - a_bn[Ai]; /* below */
               right  = -a_ac[Ai] - a_as[Ai] - a_an[Ai]; /* above */
               break;
         }

         if (!center)
         {
            Pp0[Pi] = 0.0;
            Pp1[Pi] = 0.0;
         }
         else
         {
            switch (Pstenc0)
            {
               case -1:
                  Pp0[Pi] = left  / center;
                  Pp1[Pi] = right / center;
                  break;
               case 1:
                  Pp0[Pi] = right / center;
                  Pp1[Pi] = left  / center;
                  break;
            }
         }

         if (p0[Ai] == 0.0)
         {
            Pp0[Pi] = 0.0;
         }
         if (p1[Ai] == 0.0)
         {
            Pp1[Pi] = 0.0;
         }
      }
      nalu_hypre_BoxLoop2End(Ai, Pi);
   }
   else
   {
      nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                          A_dbox, start, stride, Ai,
                          P_dbox, startc, stridec, Pi);
      {
         NALU_HYPRE_Real center, left, right;

         switch (cdir)
         {
            case 0:
               center =  a_cc[Ai] + a_cs[Ai] + a_cn[Ai] +  a_ac[Ai] + a_as[Ai] + a_an[Ai] +
                         a_bc[Ai] + a_bs[Ai] + a_bn[Ai];
               left   = -a_cw[Ai] - a_aw[Ai] - a_bw[Ai];
               right  = -a_ce[Ai] - a_ae[Ai] - a_be[Ai];
               break;
            case 1:
               center =  a_cc[Ai] + a_cw[Ai] + a_ce[Ai] +  a_ac[Ai] +  a_aw[Ai] + a_ae[Ai] +
                         a_bc[Ai] + a_bw[Ai] + a_be[Ai];
               left   = -a_cs[Ai] - a_as[Ai] - a_bs[Ai]; /* front */
               right  = -a_cn[Ai] - a_an[Ai] - a_bn[Ai]; /* back */
               break;
            case 2:
               center =  a_cc[Ai] + a_cw[Ai] + a_ce[Ai] + a_cs[Ai] + a_cn[Ai];
               left   = -a_bc[Ai] - a_bw[Ai] - a_be[Ai] - a_bs[Ai] - a_bn[Ai]; /* below */
               right  = -a_ac[Ai] - a_aw[Ai] - a_ae[Ai] - a_as[Ai] - a_an[Ai]; /* above */
               break;
         }

         if (!center)
         {
            Pp0[Pi] = 0.0;
            Pp1[Pi] = 0.0;
         }
         else
         {
            switch (Pstenc0)
            {
               case -1:
                  Pp0[Pi] = left  / center;
                  Pp1[Pi] = right / center;
                  break;
               case 1:
                  Pp0[Pi] = right / center;
                  Pp1[Pi] = left  / center;
                  break;
            }
         }

         if (p0[Ai] == 0.0)
         {
            Pp0[Pi] = 0.0;
         }
         if (p1[Ai] == 0.0)
         {
            Pp1[Pi] = 0.0;
         }
      }
      nalu_hypre_BoxLoop2End(Ai, Pi);
   }
#undef DEVICE_VAR

   return nalu_hypre_error_flag;
}


NALU_HYPRE_Int
nalu_hypre_PFMGSetupInterpOp_CC0_SS19
( NALU_HYPRE_Int           i, /* box index */
  nalu_hypre_StructMatrix *A,
  nalu_hypre_Box          *A_dbox,
  NALU_HYPRE_Int           cdir,
  nalu_hypre_Index         stride,
  nalu_hypre_Index         stridec,
  nalu_hypre_Index         start,
  nalu_hypre_IndexRef      startc,
  nalu_hypre_Index         loop_size,
  nalu_hypre_Box          *P_dbox,
  NALU_HYPRE_Int           Pstenc0,
  NALU_HYPRE_Int           Pstenc1,
  NALU_HYPRE_Real         *Pp0,
  NALU_HYPRE_Real         *Pp1,
  NALU_HYPRE_Int           rap_type,
  nalu_hypre_Index        *P_stencil_shape )
{
   //nalu_hypre_StructStencil   *stencil = nalu_hypre_StructMatrixStencil(A);
   // nalu_hypre_Index           *stencil_shape = nalu_hypre_StructStencilShape(stencil);
   //NALU_HYPRE_Int              stencil_size = nalu_hypre_StructStencilSize(stencil);
   //NALU_HYPRE_Int              warning_cnt= 0;

   nalu_hypre_Index            index;
   NALU_HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn, *a_ac, *a_bc;
   NALU_HYPRE_Real           *a_csw, *a_cse, *a_cne, *a_cnw;
   NALU_HYPRE_Real           *a_aw, *a_ae, *a_as, *a_an, *a_bw, *a_be, *a_bs, *a_bn;
   NALU_HYPRE_Real            *p0, *p1;

   p0 = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[0]);
   p1 = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[1]);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index, 0, 0, 0);
   a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 0, 0);
   a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 0, 0);
   a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, -1, 0);
   a_cs = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 1, 0);
   a_cn = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 0, 1);
   a_ac = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 0, -1);
   a_bc = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 19-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index, -1, 0, 1);
   a_aw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 0, 1);
   a_ae = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, -1, 1);
   a_as = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 1, 1);
   a_an = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 0, -1);
   a_bw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 0, -1);
   a_be = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, -1, -1);
   a_bs = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 1, -1);
   a_bn = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, -1, 0);
   a_csw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, -1, 0);
   a_cse = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 1, 0);
   a_cnw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 1, 0);
   a_cne = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

#define DEVICE_VAR is_device_ptr(a_cc,a_cs,a_cn,a_ac,a_bc,a_as,a_an,a_bs,a_bn,a_cw,a_aw,a_bw,a_csw,a_cnw,a_ce,a_ae,a_be,a_cse,a_cne,Pp0,Pp1,p0,p1)
   nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start, stride, Ai,
                       P_dbox, startc, stridec, Pi);
   {
      NALU_HYPRE_Real center, left, right;

      switch (cdir)
      {
         case 0:
            center = a_cc[Ai] +  a_cs[Ai] + a_cn[Ai] + a_ac[Ai] + a_bc[Ai] + a_as[Ai] + a_an[Ai] + a_bs[Ai] +
                     a_bn[Ai];
            left   = -a_cw[Ai] - a_aw[Ai] - a_bw[Ai] - a_csw[Ai] - a_cnw[Ai];
            right  = -a_ce[Ai] - a_ae[Ai] - a_be[Ai] - a_cse[Ai] - a_cne[Ai];
            break;
         case 1:
            center = a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai] + a_ac[Ai] + a_bc[Ai] + a_aw[Ai] + a_ae[Ai] + a_bw[Ai] +
                     a_be[Ai];
            left   = -a_cs[Ai] - a_as[Ai] - a_bs[Ai] - a_csw[Ai] - a_cse[Ai];
            right  = -a_cn[Ai] - a_an[Ai] - a_bn[Ai] - a_cnw[Ai] - a_cne[Ai];
            break;
         case 2:
            center = a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai] +  a_cs[Ai] + a_cn[Ai] + a_csw[Ai] + a_cse[Ai] + a_cnw[Ai]
                     + a_cne[Ai];
            left   = -a_bc[Ai] - a_bw[Ai] - a_be[Ai] - a_bs[Ai] - a_bn[Ai];
            right  = -a_ac[Ai] - a_aw[Ai] - a_ae[Ai] - a_as[Ai] - a_an[Ai];
            break;
      };

      if (!center)
      {
         Pp0[Pi] = 0.0;
         Pp1[Pi] = 0.0;
      }
      else
      {
         switch (Pstenc0)
         {
            case -1:
               Pp0[Pi] = left / center;
               Pp1[Pi] = right / center;
               break;
            case 1:
               Pp0[Pi] = right / center;
               Pp1[Pi] = left / center;
               break;
         };
         /*
            switch (Pstenc1)
            {
            case -1:
            Pp1[Pi] = left/center;break;
            case 1:
            Pp1[Pi] = right/center;break;
            };
            */
      }

      if (p0[Ai] == 0.0) { Pp0[Pi] = 0.0; }
      if (p1[Ai] == 0.0) { Pp1[Pi] = 0.0; }
      //printf("Pp0[%d] = %e, Pp1 = %e\n",Pi,Pp0[Pi],Pp1[Pi]);
   }
   nalu_hypre_BoxLoop2End(Ai, Pi);
#undef DEVICE_VAR

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_PFMGSetupInterpOp_CC0_SS27
( NALU_HYPRE_Int           i, /* box index */
  nalu_hypre_StructMatrix *A,
  nalu_hypre_Box          *A_dbox,
  NALU_HYPRE_Int           cdir,
  nalu_hypre_Index         stride,
  nalu_hypre_Index         stridec,
  nalu_hypre_Index         start,
  nalu_hypre_IndexRef      startc,
  nalu_hypre_Index         loop_size,
  nalu_hypre_Box          *P_dbox,
  NALU_HYPRE_Int           Pstenc0,
  NALU_HYPRE_Int           Pstenc1,
  NALU_HYPRE_Real         *Pp0,
  NALU_HYPRE_Real         *Pp1,
  NALU_HYPRE_Int           rap_type,
  nalu_hypre_Index        *P_stencil_shape )
{
   //nalu_hypre_StructStencil   *stencil = nalu_hypre_StructMatrixStencil(A);
   //nalu_hypre_Index           *stencil_shape = nalu_hypre_StructStencilShape(stencil);
   //NALU_HYPRE_Int              stencil_size = nalu_hypre_StructStencilSize(stencil);
   //NALU_HYPRE_Int              warning_cnt= 0;

   nalu_hypre_Index            index;
   NALU_HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn, *a_ac, *a_bc;
   NALU_HYPRE_Real           *a_csw, *a_cse, *a_cne, *a_cnw;
   NALU_HYPRE_Real           *a_aw, *a_ae, *a_as, *a_an, *a_bw, *a_be, *a_bs, *a_bn;
   NALU_HYPRE_Real           *a_asw, *a_ase, *a_ane, *a_anw, *a_bsw, *a_bse, *a_bne, *a_bnw;
   NALU_HYPRE_Real            *p0, *p1;

   p0 = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[0]);
   p1 = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, P_stencil_shape[1]);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index, 0, 0, 0);
   a_cc = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 0, 0);
   a_cw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 0, 0);
   a_ce = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, -1, 0);
   a_cs = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 1, 0);
   a_cn = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 0, 1);
   a_ac = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 0, -1);
   a_bc = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 19-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index, -1, 0, 1);
   a_aw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 0, 1);
   a_ae = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, -1, 1);
   a_as = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 1, 1);
   a_an = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 0, -1);
   a_bw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 0, -1);
   a_be = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, -1, -1);
   a_bs = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 0, 1, -1);
   a_bn = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, -1, 0);
   a_csw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, -1, 0);
   a_cse = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 1, 0);
   a_cnw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 1, 0);
   a_cne = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point fine grid operator:
    *
    * a_asw is pointer for southwest coefficient in plane above
    * a_ase is pointer for southeast coefficient in plane above
    * a_anw is pointer for northwest coefficient in plane above
    * a_ane is pointer for northeast coefficient in plane above
    * a_bsw is pointer for southwest coefficient in plane below
    * a_bse is pointer for southeast coefficient in plane below
    * a_bnw is pointer for northwest coefficient in plane below
    * a_bne is pointer for northeast coefficient in plane below
    *-----------------------------------------------------------------*/

   nalu_hypre_SetIndex3(index, -1, -1, 1);
   a_asw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, -1, 1);
   a_ase = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 1, 1);
   a_anw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 1, 1);
   a_ane = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, -1, -1);
   a_bsw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, -1, -1);
   a_bse = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, -1, 1, -1);
   a_bnw = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

   nalu_hypre_SetIndex3(index, 1, 1, -1);
   a_bne = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);

#define DEVICE_VAR is_device_ptr(a_cc,a_cs,a_cn,a_ac,a_bc,a_as,a_an,a_bs,a_bn,a_cw,a_aw,a_bw,a_csw,a_cnw,a_asw,a_anw,a_bsw,a_bnw,a_ce,a_ae,a_be,a_cse,a_cne,a_ase,a_ane,a_bse,a_bne,Pp0,Pp1,p0,p1)
   nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                       A_dbox, start, stride, Ai,
                       P_dbox, startc, stridec, Pi);
   {
      NALU_HYPRE_Real center, left, right;

      switch (cdir)
      {
         case 0:
            center = a_cc[Ai] +  a_cs[Ai] + a_cn[Ai] + a_ac[Ai] + a_bc[Ai] + a_as[Ai] + a_an[Ai] + a_bs[Ai] +
                     a_bn[Ai];
            left   = -a_cw[Ai] - a_aw[Ai] - a_bw[Ai] - a_csw[Ai] - a_cnw[Ai] - a_asw[Ai] - a_anw[Ai] - a_bsw[Ai]
                     - a_bnw[Ai];
            right  = -a_ce[Ai] - a_ae[Ai] - a_be[Ai] - a_cse[Ai] - a_cne[Ai] - a_ase[Ai] - a_ane[Ai] - a_bse[Ai]
                     - a_bne[Ai];
            break;
         case 1:
            center = a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai] + a_ac[Ai] + a_bc[Ai] + a_aw[Ai] + a_ae[Ai] + a_bw[Ai] +
                     a_be[Ai];
            left   = -a_cs[Ai] - a_as[Ai] - a_bs[Ai] - a_csw[Ai] - a_cse[Ai] - a_asw[Ai] - a_ase[Ai] - a_bsw[Ai]
                     - a_bse[Ai];
            right  = -a_cn[Ai] - a_an[Ai] - a_bn[Ai] - a_cnw[Ai] - a_cne[Ai] - a_anw[Ai] - a_ane[Ai] - a_bnw[Ai]
                     - a_bne[Ai];
            break;
         case 2:
            center = a_cc[Ai] +  a_cw[Ai] +  a_ce[Ai] +  a_cs[Ai] + a_cn[Ai] + a_csw[Ai] + a_cse[Ai] + a_cnw[Ai]
                     + a_cne[Ai];
            left   = -a_bc[Ai] - a_bw[Ai] - a_be[Ai] - a_bs[Ai] - a_bn[Ai] - a_bsw[Ai] - a_bse[Ai] - a_bnw[Ai] -
                     a_bne[Ai];
            right  = -a_ac[Ai] - a_aw[Ai] - a_ae[Ai] - a_as[Ai] - a_an[Ai] - a_asw[Ai] - a_ase[Ai] - a_anw[Ai] -
                     a_ane[Ai];
            break;
      };

      if (!center)
      {
         //warning_cnt++;
         Pp0[Pi] = 0.0;
         Pp1[Pi] = 0.0;
      }
      else
      {
         switch (Pstenc0)
         {
            case -1:
               Pp0[Pi] = left / center;
               Pp1[Pi] = right / center;
               break;
            case 1:
               Pp0[Pi] = right / center;
               Pp1[Pi] = left / center;
               break;
         };
         /*
            switch (Pstenc1)
            {
            case -1:
            Pp1[Pi] = left/center;break;
            case 1:
            Pp1[Pi] = right/center;break;
            };
            */
      }

      if (p0[Ai] == 0.0) { Pp0[Pi] = 0.0; }
      if (p1[Ai] == 0.0) { Pp1[Pi] = 0.0; }
      //printf("Pp0[%d] = %e, Pp1 = %e\n",Pi,Pp0[Pi],Pp1[Pi]);
   }
   nalu_hypre_BoxLoop2End(Ai, Pi);
#undef DEVICE_VAR

   return nalu_hypre_error_flag;
}

#endif

