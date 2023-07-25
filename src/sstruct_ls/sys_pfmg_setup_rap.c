/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

nalu_hypre_SStructPMatrix *
nalu_hypre_SysPFMGCreateRAPOp( nalu_hypre_SStructPMatrix *R,
                          nalu_hypre_SStructPMatrix *A,
                          nalu_hypre_SStructPMatrix *P,
                          nalu_hypre_SStructPGrid   *coarse_grid,
                          NALU_HYPRE_Int             cdir        )
{
   nalu_hypre_SStructPMatrix    *RAP;
   NALU_HYPRE_Int                ndim;
   NALU_HYPRE_Int                nvars;
   nalu_hypre_SStructVariable    vartype;

   nalu_hypre_SStructStencil **RAP_stencils;

   nalu_hypre_StructMatrix    *RAP_s;
   nalu_hypre_StructMatrix    *R_s;
   nalu_hypre_StructMatrix    *A_s;
   nalu_hypre_StructMatrix    *P_s;

   nalu_hypre_Index          **RAP_shapes;

   nalu_hypre_StructStencil   *sstencil;
   nalu_hypre_Index           *shape;
   NALU_HYPRE_Int              s;
   NALU_HYPRE_Int             *sstencil_sizes;

   NALU_HYPRE_Int              stencil_size;

   nalu_hypre_StructGrid      *cgrid;

   NALU_HYPRE_Int              vi, vj;

   NALU_HYPRE_Int              sten_cntr;

   NALU_HYPRE_Int              P_stored_as_transpose = 0;

   ndim = nalu_hypre_StructStencilNDim(nalu_hypre_SStructPMatrixSStencil(A, 0, 0));
   nvars = nalu_hypre_SStructPMatrixNVars(A);

   vartype = nalu_hypre_SStructPGridVarType(coarse_grid, 0);
   cgrid = nalu_hypre_SStructPGridVTSGrid(coarse_grid, vartype);

   RAP_stencils = nalu_hypre_CTAlloc(nalu_hypre_SStructStencil *,  nvars, NALU_HYPRE_MEMORY_HOST);

   RAP_shapes = nalu_hypre_CTAlloc(nalu_hypre_Index *,  nvars, NALU_HYPRE_MEMORY_HOST);
   sstencil_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nvars, NALU_HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------------
    * Symmetry within a block is exploited, but not symmetry of the form
    * A_{vi,vj} = A_{vj,vi}^T.
    *--------------------------------------------------------------------------*/

   for (vi = 0; vi < nvars; vi++)
   {
      R_s = nalu_hypre_SStructPMatrixSMatrix(R, vi, vi);
      stencil_size = 0;
      for (vj = 0; vj < nvars; vj++)
      {
         A_s = nalu_hypre_SStructPMatrixSMatrix(A, vi, vj);
         P_s = nalu_hypre_SStructPMatrixSMatrix(P, vj, vj);
         sstencil_sizes[vj] = 0;
         if (A_s != NULL)
         {
            RAP_s = nalu_hypre_SemiCreateRAPOp(R_s, A_s, P_s,
                                          cgrid, cdir,
                                          P_stored_as_transpose);
            /* Just want stencil for RAP */
            nalu_hypre_StructMatrixInitializeShell(RAP_s);
            sstencil = nalu_hypre_StructMatrixStencil(RAP_s);
            shape = nalu_hypre_StructStencilShape(sstencil);
            sstencil_sizes[vj] = nalu_hypre_StructStencilSize(sstencil);
            stencil_size += sstencil_sizes[vj];
            RAP_shapes[vj] = nalu_hypre_CTAlloc(nalu_hypre_Index,
                                           sstencil_sizes[vj], NALU_HYPRE_MEMORY_HOST);
            for (s = 0; s < sstencil_sizes[vj]; s++)
            {
               nalu_hypre_CopyIndex(shape[s], RAP_shapes[vj][s]);
            }
            nalu_hypre_StructMatrixDestroy(RAP_s);
         }
      }

      NALU_HYPRE_SStructStencilCreate(ndim, stencil_size, &RAP_stencils[vi]);
      sten_cntr = 0;
      for (vj = 0; vj < nvars; vj++)
      {
         if (sstencil_sizes[vj] > 0)
         {
            for (s = 0; s < sstencil_sizes[vj]; s++)
            {
               NALU_HYPRE_SStructStencilSetEntry(RAP_stencils[vi],
                                            sten_cntr, RAP_shapes[vj][s], vj);
               sten_cntr++;
            }
            nalu_hypre_TFree(RAP_shapes[vj], NALU_HYPRE_MEMORY_HOST);
         }
      }
   }

   /* create RAP Pmatrix */
   nalu_hypre_SStructPMatrixCreate(nalu_hypre_SStructPMatrixComm(A),
                              coarse_grid, RAP_stencils, &RAP);

   nalu_hypre_TFree(RAP_shapes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(sstencil_sizes, NALU_HYPRE_MEMORY_HOST);

   return RAP;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SysPFMGSetupRAPOp( nalu_hypre_SStructPMatrix *R,
                         nalu_hypre_SStructPMatrix *A,
                         nalu_hypre_SStructPMatrix *P,
                         NALU_HYPRE_Int             cdir,
                         nalu_hypre_Index           cindex,
                         nalu_hypre_Index           cstride,
                         nalu_hypre_SStructPMatrix *Ac      )
{
   NALU_HYPRE_Int               nvars;
   NALU_HYPRE_Int               vi, vj;

   nalu_hypre_StructMatrix    *R_s;
   nalu_hypre_StructMatrix    *A_s;
   nalu_hypre_StructMatrix    *P_s;

   nalu_hypre_StructMatrix    *Ac_s;

   NALU_HYPRE_Int              P_stored_as_transpose = 0;

   nvars = nalu_hypre_SStructPMatrixNVars(A);

   /*--------------------------------------------------------------------------
    * Symmetry within a block is exploited, but not symmetry of the form
    * A_{vi,vj} = A_{vj,vi}^T.
    *--------------------------------------------------------------------------*/

   for (vi = 0; vi < nvars; vi++)
   {
      R_s = nalu_hypre_SStructPMatrixSMatrix(R, vi, vi);
      for (vj = 0; vj < nvars; vj++)
      {
         A_s  = nalu_hypre_SStructPMatrixSMatrix(A, vi, vj);
         Ac_s = nalu_hypre_SStructPMatrixSMatrix(Ac, vi, vj);
         P_s  = nalu_hypre_SStructPMatrixSMatrix(P, vj, vj);
         if (A_s != NULL)
         {
            nalu_hypre_SemiBuildRAP(A_s, P_s, R_s, cdir, cindex, cstride,
                               P_stored_as_transpose, Ac_s);
            /* Assemble here? */
            nalu_hypre_StructMatrixAssemble(Ac_s);
         }
      }
   }

   return nalu_hypre_error_flag;
}

