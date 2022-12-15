/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *
 *****************************************************************************/

#include "_nalu_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGCreateRAPOp
 *
 *   Wrapper for 2 and 3d CreateRAPOp routines which set up new coarse
 *   grid structures.
 *--------------------------------------------------------------------------*/

nalu_hypre_StructMatrix *
nalu_hypre_SparseMSGCreateRAPOp( nalu_hypre_StructMatrix *R,
                            nalu_hypre_StructMatrix *A,
                            nalu_hypre_StructMatrix *P,
                            nalu_hypre_StructGrid   *coarse_grid,
                            NALU_HYPRE_Int           cdir        )
{
   nalu_hypre_StructMatrix    *RAP;
   nalu_hypre_StructStencil   *stencil;

   stencil = nalu_hypre_StructMatrixStencil(A);

   switch (nalu_hypre_StructStencilNDim(stencil))
   {
      case 2:
         RAP = nalu_hypre_SparseMSG2CreateRAPOp(R, A, P, coarse_grid, cdir);
         break;

      case 3:
         RAP = nalu_hypre_SparseMSG3CreateRAPOp(R, A, P, coarse_grid, cdir);
         break;
   }

   return RAP;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_SparseMSGSetupRAPOp
 *
 * Wrapper for 2 and 3d, symmetric and non-symmetric routines to calculate
 * entries in RAP. Incomplete error handling at the moment.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SparseMSGSetupRAPOp( nalu_hypre_StructMatrix *R,
                           nalu_hypre_StructMatrix *A,
                           nalu_hypre_StructMatrix *P,
                           NALU_HYPRE_Int           cdir,
                           nalu_hypre_Index         cindex,
                           nalu_hypre_Index         cstride,
                           nalu_hypre_Index         stridePR,
                           nalu_hypre_StructMatrix *Ac       )
{
   NALU_HYPRE_Int ierr = 0;

   nalu_hypre_StructStencil   *stencil;

   stencil = nalu_hypre_StructMatrixStencil(A);

   switch (nalu_hypre_StructStencilNDim(stencil))
   {

      case 2:

         /*--------------------------------------------------------------------
          *    Set lower triangular (+ diagonal) coefficients
          *--------------------------------------------------------------------*/
         ierr = nalu_hypre_SparseMSG2BuildRAPSym(A, P, R, cdir,
                                            cindex, cstride, stridePR, Ac);

         /*--------------------------------------------------------------------
          *    For non-symmetric A, set upper triangular coefficients as well
          *--------------------------------------------------------------------*/
         if (!nalu_hypre_StructMatrixSymmetric(A))
            ierr += nalu_hypre_SparseMSG2BuildRAPNoSym(A, P, R, cdir,
                                                  cindex, cstride, stridePR, Ac);

         break;

      case 3:

         /*--------------------------------------------------------------------
          *    Set lower triangular (+ diagonal) coefficients
          *--------------------------------------------------------------------*/
         ierr = nalu_hypre_SparseMSG3BuildRAPSym(A, P, R, cdir,
                                            cindex, cstride, stridePR, Ac);

         /*--------------------------------------------------------------------
          *    For non-symmetric A, set upper triangular coefficients as well
          *--------------------------------------------------------------------*/
         if (!nalu_hypre_StructMatrixSymmetric(A))
            ierr += nalu_hypre_SparseMSG3BuildRAPNoSym(A, P, R, cdir,
                                                  cindex, cstride, stridePR, Ac);

         break;

   }

   nalu_hypre_StructMatrixAssemble(Ac);

   return ierr;
}

