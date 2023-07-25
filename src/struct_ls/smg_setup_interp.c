/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"
#include "smg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

nalu_hypre_StructMatrix *
nalu_hypre_SMGCreateInterpOp( nalu_hypre_StructMatrix *A,
                         nalu_hypre_StructGrid   *cgrid,
                         NALU_HYPRE_Int           cdir  )
{
   nalu_hypre_StructMatrix   *PT;

   nalu_hypre_StructStencil  *stencil;
   nalu_hypre_Index          *stencil_shape;
   NALU_HYPRE_Int             stencil_size;
   NALU_HYPRE_Int             stencil_dim;

   NALU_HYPRE_Int             num_ghost[] = {1, 1, 1, 1, 1, 1};

   NALU_HYPRE_Int             i;

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
   PT = nalu_hypre_StructMatrixCreate(nalu_hypre_StructMatrixComm(A), cgrid, stencil);
   nalu_hypre_StructMatrixSetNumGhost(PT, num_ghost);

   nalu_hypre_StructStencilDestroy(stencil);

   return PT;
}

/*--------------------------------------------------------------------------
 * This routine uses SMGRelax to set up the interpolation operator.
 *
 * To illustrate how it proceeds, consider setting up the the {0, 0, -1}
 * stencil coefficient of P^T.  This coefficient corresponds to the
 * {0, 0, 1} coefficient of P.  Do one sweep of plane relaxation on the
 * fine grid points for the system, A_mask x = b, with initial guess
 * x_0 = all ones and right-hand-side b = all zeros.  The A_mask matrix
 * contains all coefficients of A except for those in the same direction
 * as {0, 0, -1}.
 *
 * The relaxation data for the multigrid algorithm is passed in and used.
 * When this routine returns, the only modified relaxation parameters
 * are MaxIter, RegSpace and PreSpace info, the right-hand-side and
 * solution info.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetupInterpOp( void               *relax_data,
                        nalu_hypre_StructMatrix *A,
                        nalu_hypre_StructVector *b,
                        nalu_hypre_StructVector *x,
                        nalu_hypre_StructMatrix *PT,
                        NALU_HYPRE_Int           cdir,
                        nalu_hypre_Index         cindex,
                        nalu_hypre_Index         findex,
                        nalu_hypre_Index         stride    )
{
   nalu_hypre_StructMatrix   *A_mask;

   nalu_hypre_StructStencil  *A_stencil;
   nalu_hypre_Index          *A_stencil_shape;
   NALU_HYPRE_Int             A_stencil_size;
   nalu_hypre_StructStencil  *PT_stencil;
   nalu_hypre_Index          *PT_stencil_shape;
   NALU_HYPRE_Int             PT_stencil_size;

   NALU_HYPRE_Int            *stencil_indices;
   NALU_HYPRE_Int             num_stencil_indices;

   nalu_hypre_StructGrid     *fgrid;

   nalu_hypre_StructStencil  *compute_pkg_stencil;
   nalu_hypre_Index          *compute_pkg_stencil_shape;
   NALU_HYPRE_Int             compute_pkg_stencil_size = 1;
   NALU_HYPRE_Int             compute_pkg_stencil_dim = 1;
   nalu_hypre_ComputePkg     *compute_pkg;
   nalu_hypre_ComputeInfo    *compute_info;

   nalu_hypre_CommHandle     *comm_handle;

   nalu_hypre_BoxArrayArray  *compute_box_aa;
   nalu_hypre_BoxArray       *compute_box_a;
   nalu_hypre_Box            *compute_box;

   nalu_hypre_Box            *PT_data_box;
   nalu_hypre_Box            *x_data_box;
   NALU_HYPRE_Real           *PTp;
   NALU_HYPRE_Real           *xp;

   nalu_hypre_Index           loop_size;
   nalu_hypre_Index           start;
   nalu_hypre_Index           startc;
   nalu_hypre_Index           stridec;

   NALU_HYPRE_Int             si, sj, d;
   NALU_HYPRE_Int             compute_i, i, j;

   /*--------------------------------------------------------
    * Initialize some things
    *--------------------------------------------------------*/

   nalu_hypre_SetIndex3(stridec, 1, 1, 1);

   fgrid = nalu_hypre_StructMatrixGrid(A);

   A_stencil = nalu_hypre_StructMatrixStencil(A);
   A_stencil_shape = nalu_hypre_StructStencilShape(A_stencil);
   A_stencil_size  = nalu_hypre_StructStencilSize(A_stencil);
   PT_stencil = nalu_hypre_StructMatrixStencil(PT);
   PT_stencil_shape = nalu_hypre_StructStencilShape(PT_stencil);
   PT_stencil_size  = nalu_hypre_StructStencilSize(PT_stencil);

   /* Set up relaxation parameters */
   nalu_hypre_SMGRelaxSetMaxIter(relax_data, 1);
   nalu_hypre_SMGRelaxSetNumPreSpaces(relax_data, 0);
   nalu_hypre_SMGRelaxSetNumRegSpaces(relax_data, 1);
   nalu_hypre_SMGRelaxSetRegSpaceRank(relax_data, 0, 1);

   compute_pkg_stencil_shape =
      nalu_hypre_CTAlloc(nalu_hypre_Index,  compute_pkg_stencil_size, NALU_HYPRE_MEMORY_HOST);
   compute_pkg_stencil = nalu_hypre_StructStencilCreate(compute_pkg_stencil_dim,
                                                   compute_pkg_stencil_size,
                                                   compute_pkg_stencil_shape);

   for (si = 0; si < PT_stencil_size; si++)
   {
      /*-----------------------------------------------------
       * Compute A_mask matrix: This matrix contains all
       * stencil coefficients of A except for the coefficients
       * in the opposite direction of the current P stencil
       * coefficient being computed (same direction for P^T).
       *-----------------------------------------------------*/

      stencil_indices = nalu_hypre_TAlloc(NALU_HYPRE_Int,  A_stencil_size, NALU_HYPRE_MEMORY_HOST);
      num_stencil_indices = 0;
      for (sj = 0; sj < A_stencil_size; sj++)
      {
         if (nalu_hypre_IndexD(A_stencil_shape[sj],  cdir) !=
             nalu_hypre_IndexD(PT_stencil_shape[si], cdir)   )
         {
            stencil_indices[num_stencil_indices] = sj;
            num_stencil_indices++;
         }
      }
      A_mask =
         nalu_hypre_StructMatrixCreateMask(A, num_stencil_indices, stencil_indices);
      nalu_hypre_TFree(stencil_indices, NALU_HYPRE_MEMORY_HOST);

      /*-----------------------------------------------------
       * Do relaxation sweep to compute coefficients
       *-----------------------------------------------------*/

      nalu_hypre_StructVectorClearGhostValues(x);
      nalu_hypre_StructVectorSetConstantValues(x, 1.0);
      nalu_hypre_StructVectorSetConstantValues(b, 0.0);
      nalu_hypre_SMGRelaxSetNewMatrixStencil(relax_data, PT_stencil);
      nalu_hypre_SMGRelaxSetup(relax_data, A_mask, b, x);
      nalu_hypre_SMGRelax(relax_data, A_mask, b, x);

      /*-----------------------------------------------------
       * Free up A_mask matrix
       *-----------------------------------------------------*/

      nalu_hypre_StructMatrixDestroy(A_mask);

      /*-----------------------------------------------------
       * Set up compute package for communication of
       * coefficients from fine to coarse across processor
       * boundaries.
       *-----------------------------------------------------*/

      nalu_hypre_CopyIndex(PT_stencil_shape[si], compute_pkg_stencil_shape[0]);
      nalu_hypre_CreateComputeInfo(fgrid, compute_pkg_stencil, &compute_info);
      nalu_hypre_ComputeInfoProjectSend(compute_info, findex, stride);
      nalu_hypre_ComputeInfoProjectRecv(compute_info, findex, stride);
      nalu_hypre_ComputeInfoProjectComp(compute_info, cindex, stride);
      nalu_hypre_ComputePkgCreate(compute_info, nalu_hypre_StructVectorDataSpace(x), 1,
                             fgrid, &compute_pkg);

      /*-----------------------------------------------------
       * Copy coefficients from x into P^T
       *-----------------------------------------------------*/

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
            compute_box_a =
               nalu_hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

            x_data_box  =
               nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);
            PT_data_box =
               nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(PT), i);

            xp  = nalu_hypre_StructVectorBoxData(x, i);
            PTp = nalu_hypre_StructMatrixBoxData(PT, i, si);

            nalu_hypre_ForBoxI(j, compute_box_a)
            {
               compute_box = nalu_hypre_BoxArrayBox(compute_box_a, j);

               nalu_hypre_CopyIndex(nalu_hypre_BoxIMin(compute_box), start);
               nalu_hypre_StructMapFineToCoarse(start, cindex, stride,
                                           startc);

               /* shift start index to appropriate F-point */
               for (d = 0; d < 3; d++)
               {
                  nalu_hypre_IndexD(start, d) +=
                     nalu_hypre_IndexD(PT_stencil_shape[si], d);
               }

               nalu_hypre_BoxGetStrideSize(compute_box, stride, loop_size);

#define DEVICE_VAR is_device_ptr(PTp,xp)
               nalu_hypre_BoxLoop2Begin(nalu_hypre_StructMatrixNDim(A), loop_size,
                                   x_data_box,  start,  stride,  xi,
                                   PT_data_box, startc, stridec, PTi);
               {
                  PTp[PTi] = xp[xi];
               }
               nalu_hypre_BoxLoop2End(xi, PTi);
#undef DEVICE_VAR
            }
         }
      }

      /*-----------------------------------------------------
       * Free up compute package info
       *-----------------------------------------------------*/

      nalu_hypre_ComputePkgDestroy(compute_pkg);
   }

   /* Tell SMGRelax that the stencil has changed */
   nalu_hypre_SMGRelaxSetNewMatrixStencil(relax_data, PT_stencil);

   nalu_hypre_StructStencilDestroy(compute_pkg_stencil);

#if 0
   nalu_hypre_StructMatrixAssemble(PT);
#else
   nalu_hypre_StructInterpAssemble(A, PT, 1, cdir, cindex, stride);
#endif

   return nalu_hypre_error_flag;
}

