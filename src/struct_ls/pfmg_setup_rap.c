/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "pfmg.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_PFMGCreateRAPOp
 *
 *   Wrapper for 2 and 3d CreateRAPOp routines which set up new coarse
 *   grid structures.
 *
 *   The parameter rap_type controls which lower level routines are
 *   used.
 *      rap_type = 0   Use optimized code for computing Galerkin operators
 *                     for special, common stencil patterns: 5 & 9 pt in
 *                     2d and 7, 19 & 27 in 3d.
 *      rap_type = 1   Use PARFLOW formula for coarse grid operator. Used
 *                     only with 5pt in 2d and 7pt in 3d.
 *      rap_type = 2   General purpose Galerkin code.
 *--------------------------------------------------------------------------*/

nalu_hypre_StructMatrix *
nalu_hypre_PFMGCreateRAPOp( nalu_hypre_StructMatrix *R,
                       nalu_hypre_StructMatrix *A,
                       nalu_hypre_StructMatrix *P,
                       nalu_hypre_StructGrid   *coarse_grid,
                       NALU_HYPRE_Int           cdir,
                       NALU_HYPRE_Int           rap_type    )
{
   nalu_hypre_StructMatrix    *RAP = NULL;
   nalu_hypre_StructStencil   *stencil;
   NALU_HYPRE_Int              P_stored_as_transpose = 0;
   NALU_HYPRE_Int              constant_coefficient;

   stencil = nalu_hypre_StructMatrixStencil(A);

   if (rap_type == 0)
   {
      switch (nalu_hypre_StructStencilNDim(stencil))
      {
         case 2:
            RAP = nalu_hypre_PFMG2CreateRAPOp(R, A, P, coarse_grid, cdir);
            break;

         case 3:
            RAP = nalu_hypre_PFMG3CreateRAPOp(R, A, P, coarse_grid, cdir);
            break;
      }
   }

   else if (rap_type == 1)
   {
      switch (nalu_hypre_StructStencilNDim(stencil))
      {
         case 2:
            RAP =  nalu_hypre_PFMGCreateCoarseOp5(R, A, P, coarse_grid, cdir);
            break;

         case 3:
            RAP =  nalu_hypre_PFMGCreateCoarseOp7(R, A, P, coarse_grid, cdir);
            break;
      }
   }
   else if (rap_type == 2)
   {
      RAP = nalu_hypre_SemiCreateRAPOp(R, A, P, coarse_grid, cdir,
                                  P_stored_as_transpose);
   }


   constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(A);
   if ( constant_coefficient == 2 && rap_type == 0 )
   {
      /* A has variable diagonal, so, in the Galerkin case, P (and R) is
         entirely variable coefficient.  Thus RAP will be variable coefficient */
      nalu_hypre_StructMatrixSetConstantCoefficient( RAP, 0 );
   }
   else
   {
      nalu_hypre_StructMatrixSetConstantCoefficient( RAP, constant_coefficient );
   }

   return RAP;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PFMGSetupRAPOp
 *
 * Wrapper for 2 and 3d, symmetric and non-symmetric routines to calculate
 * entries in RAP. Incomplete error handling at the moment.
 *
 *   The parameter rap_type controls which lower level routines are
 *   used.
 *      rap_type = 0   Use optimized code for computing Galerkin operators
 *                     for special, common stencil patterns: 5 & 9 pt in
 *                     2d and 7, 19 & 27 in 3d.
 *      rap_type = 1   Use PARFLOW formula for coarse grid operator. Used
 *                     only with 5pt in 2d and 7pt in 3d.
 *      rap_type = 2   General purpose Galerkin code.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PFMGSetupRAPOp( nalu_hypre_StructMatrix *R,
                      nalu_hypre_StructMatrix *A,
                      nalu_hypre_StructMatrix *P,
                      NALU_HYPRE_Int           cdir,
                      nalu_hypre_Index         cindex,
                      nalu_hypre_Index         cstride,
                      NALU_HYPRE_Int           rap_type,
                      nalu_hypre_StructMatrix *Ac      )
{
   NALU_HYPRE_Int              P_stored_as_transpose = 0;
   nalu_hypre_StructStencil   *stencil;

   nalu_hypre_StructMatrix    *Ac_tmp;

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_MemoryLocation data_location_A = nalu_hypre_StructGridDataLocation(nalu_hypre_StructMatrixGrid(A));
   NALU_HYPRE_MemoryLocation data_location_Ac = nalu_hypre_StructGridDataLocation(nalu_hypre_StructMatrixGrid(Ac));
   NALU_HYPRE_Int constant_coefficient = nalu_hypre_StructMatrixConstantCoefficient(Ac);
   if ( data_location_A != data_location_Ac )
   {
      Ac_tmp = nalu_hypre_PFMGCreateRAPOp(R, A, P, nalu_hypre_StructMatrixGrid(Ac), cdir, rap_type);
      nalu_hypre_StructMatrixSymmetric(Ac_tmp) = nalu_hypre_StructMatrixSymmetric(Ac);
      nalu_hypre_StructMatrixConstantCoefficient(Ac_tmp) = nalu_hypre_StructMatrixConstantCoefficient(Ac);
      nalu_hypre_StructGridDataLocation(nalu_hypre_StructMatrixGrid(Ac)) = data_location_A;
      NALU_HYPRE_StructMatrixInitialize(Ac_tmp);
   }
   else
   {
      Ac_tmp = Ac;
   }
#else
   Ac_tmp = Ac;
#endif
   stencil = nalu_hypre_StructMatrixStencil(A);

   if (rap_type == 0)
   {
      switch (nalu_hypre_StructStencilNDim(stencil))
      {
         case 2:
            /*--------------------------------------------------------------------
             *    Set lower triangular (+ diagonal) coefficients
             *--------------------------------------------------------------------*/
            nalu_hypre_PFMG2BuildRAPSym(A, P, R, cdir, cindex, cstride, Ac_tmp);

            /*--------------------------------------------------------------------
             *    For non-symmetric A, set upper triangular coefficients as well
             *--------------------------------------------------------------------*/
            if (!nalu_hypre_StructMatrixSymmetric(A))
            {
               nalu_hypre_PFMG2BuildRAPNoSym(A, P, R, cdir, cindex, cstride, Ac_tmp);
            }

            break;

         case 3:

            /*--------------------------------------------------------------------
             *    Set lower triangular (+ diagonal) coefficients
             *--------------------------------------------------------------------*/
            nalu_hypre_PFMG3BuildRAPSym(A, P, R, cdir, cindex, cstride, Ac_tmp);

            /*--------------------------------------------------------------------
             *    For non-symmetric A, set upper triangular coefficients as well
             *--------------------------------------------------------------------*/
            if (!nalu_hypre_StructMatrixSymmetric(A))
            {
               nalu_hypre_PFMG3BuildRAPNoSym(A, P, R, cdir, cindex, cstride, Ac_tmp);
            }

            break;
      }
   }

   else if (rap_type == 1)
   {
      switch (nalu_hypre_StructStencilNDim(stencil))
      {
         case 2:
            nalu_hypre_PFMGBuildCoarseOp5(A, P, R, cdir, cindex, cstride, Ac_tmp);
            break;

         case 3:
            nalu_hypre_PFMGBuildCoarseOp7(A, P, R, cdir, cindex, cstride, Ac_tmp);
            break;
      }
   }

   else if (rap_type == 2)
   {
      nalu_hypre_SemiBuildRAP(A, P, R, cdir, cindex, cstride,
                         P_stored_as_transpose, Ac_tmp);
   }

   nalu_hypre_StructMatrixAssemble(Ac_tmp);

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   if ( data_location_A != data_location_Ac )
   {
      if (constant_coefficient == 0)
      {
         nalu_hypre_TMemcpy(nalu_hypre_StructMatrixDataConst(Ac), nalu_hypre_StructMatrixData(Ac_tmp), NALU_HYPRE_Complex,
                       nalu_hypre_StructMatrixDataSize(Ac_tmp), NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      }
      else if (constant_coefficient == 1)
      {
         nalu_hypre_TMemcpy(nalu_hypre_StructMatrixDataConst(Ac), nalu_hypre_StructMatrixDataConst(Ac_tmp), NALU_HYPRE_Complex,
                       nalu_hypre_StructMatrixDataConstSize(Ac_tmp), NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
      }
      else if (constant_coefficient == 2)
      {
         nalu_hypre_TMemcpy(nalu_hypre_StructMatrixDataConst(Ac), nalu_hypre_StructMatrixDataConst(Ac_tmp), NALU_HYPRE_Complex,
                       nalu_hypre_StructMatrixDataConstSize(Ac_tmp), NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_StructStencil *stencil_c       = nalu_hypre_StructMatrixStencil(Ac);
         NALU_HYPRE_Int stencil_size  = nalu_hypre_StructStencilSize(stencil_c);
         NALU_HYPRE_Complex       *Acdiag = nalu_hypre_StructMatrixDataConst(Ac) + stencil_size;
         nalu_hypre_TMemcpy(Acdiag, nalu_hypre_StructMatrixData(Ac_tmp), NALU_HYPRE_Complex,
                       nalu_hypre_StructMatrixDataSize(Ac_tmp), NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      }

      nalu_hypre_HandleStructExecPolicy(nalu_hypre_handle()) = data_location_Ac == NALU_HYPRE_MEMORY_DEVICE ?
                                                     NALU_HYPRE_EXEC_DEVICE : NALU_HYPRE_EXEC_HOST;
      nalu_hypre_StructGridDataLocation(nalu_hypre_StructMatrixGrid(Ac)) = data_location_Ac;
      nalu_hypre_StructMatrixAssemble(Ac);
      nalu_hypre_HandleStructExecPolicy(nalu_hypre_handle()) = data_location_A == NALU_HYPRE_MEMORY_DEVICE ?
                                                     NALU_HYPRE_EXEC_DEVICE : NALU_HYPRE_EXEC_HOST;
      nalu_hypre_StructMatrixDestroy(Ac_tmp);
   }
#endif

   return nalu_hypre_error_flag;
}

