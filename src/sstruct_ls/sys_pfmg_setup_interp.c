/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

nalu_hypre_SStructPMatrix *
nalu_hypre_SysPFMGCreateInterpOp( nalu_hypre_SStructPMatrix *A,
                             nalu_hypre_SStructPGrid   *cgrid,
                             NALU_HYPRE_Int             cdir  )
{
   nalu_hypre_SStructPMatrix  *P;

   nalu_hypre_Index           *stencil_shape;
   NALU_HYPRE_Int              stencil_size;

   NALU_HYPRE_Int              ndim;

   NALU_HYPRE_Int              nvars;
   nalu_hypre_SStructStencil **P_stencils;

   NALU_HYPRE_Int              i, s;

   /* set up stencil_shape */
   stencil_size = 2;
   stencil_shape = nalu_hypre_CTAlloc(nalu_hypre_Index,  stencil_size, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < stencil_size; i++)
   {
      nalu_hypre_SetIndex3(stencil_shape[i], 0, 0, 0);
   }
   nalu_hypre_IndexD(stencil_shape[0], cdir) = -1;
   nalu_hypre_IndexD(stencil_shape[1], cdir) =  1;

   /* set up P_stencils */
   ndim = nalu_hypre_StructStencilNDim(nalu_hypre_SStructPMatrixSStencil(A, 0, 0));
   nvars = nalu_hypre_SStructPMatrixNVars(A);
   P_stencils = nalu_hypre_CTAlloc(nalu_hypre_SStructStencil *,  nvars, NALU_HYPRE_MEMORY_HOST);
   for (s = 0; s < nvars; s++)
   {
      NALU_HYPRE_SStructStencilCreate(ndim, stencil_size, &P_stencils[s]);
      for (i = 0; i < stencil_size; i++)
      {
         NALU_HYPRE_SStructStencilSetEntry(P_stencils[s], i,
                                      stencil_shape[i], s);
      }
   }

   /* create interpolation matrix */
   nalu_hypre_SStructPMatrixCreate(nalu_hypre_SStructPMatrixComm(A), cgrid,
                              P_stencils, &P);

   nalu_hypre_TFree(stencil_shape, NALU_HYPRE_MEMORY_HOST);

   return P;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSetupInterpOp( nalu_hypre_SStructPMatrix *A,
                            NALU_HYPRE_Int             cdir,
                            nalu_hypre_Index           findex,
                            nalu_hypre_Index           stride,
                            nalu_hypre_SStructPMatrix *P      )
{
   NALU_HYPRE_Int              nvars;
   nalu_hypre_StructMatrix    *A_s;
   nalu_hypre_StructMatrix    *P_s;
   NALU_HYPRE_Int              vi;

   nvars = nalu_hypre_SStructPMatrixNVars(A);

   for (vi = 0; vi < nvars; vi++)
   {
      A_s = nalu_hypre_SStructPMatrixSMatrix(A, vi, vi);
      P_s = nalu_hypre_SStructPMatrixSMatrix(P, vi, vi);
      nalu_hypre_PFMGSetupInterpOp(A_s, cdir, findex, stride, P_s, 0);
   }

   return nalu_hypre_error_flag;
}
