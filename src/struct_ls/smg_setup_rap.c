/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "smg.h"

#define OLDRAP 1
#define NEWRAP 0

/*--------------------------------------------------------------------------
 * Wrapper for 2 and 3d CreateRAPOp routines which set up new coarse
 * grid structures.
 *--------------------------------------------------------------------------*/

nalu_hypre_StructMatrix *
nalu_hypre_SMGCreateRAPOp( nalu_hypre_StructMatrix *R,
                      nalu_hypre_StructMatrix *A,
                      nalu_hypre_StructMatrix *PT,
                      nalu_hypre_StructGrid   *coarse_grid )
{
   nalu_hypre_StructMatrix    *RAP;
   nalu_hypre_StructStencil   *stencil;

#if NEWRAP
   NALU_HYPRE_Int              cdir;
   NALU_HYPRE_Int              P_stored_as_transpose = 1;
#endif

   stencil = nalu_hypre_StructMatrixStencil(A);

#if OLDRAP
   switch (nalu_hypre_StructStencilNDim(stencil))
   {
      case 2:
         RAP = nalu_hypre_SMG2CreateRAPOp(R, A, PT, coarse_grid);
         break;

      case 3:
         RAP = nalu_hypre_SMG3CreateRAPOp(R, A, PT, coarse_grid);
         break;
   }
#endif

#if NEWRAP
   switch (nalu_hypre_StructStencilNDim(stencil))
   {
      case 2:
         cdir = 1;
         RAP = nalu_hypre_SemiCreateRAPOp(R, A, PT, coarse_grid, cdir,
                                     P_stored_as_transpose);
         break;

      case 3:
         cdir = 2;
         RAP = nalu_hypre_SemiCreateRAPOp(R, A, PT, coarse_grid, cdir,
                                     P_stored_as_transpose);
         break;
   }
#endif

   return RAP;
}

/*--------------------------------------------------------------------------
 * Wrapper for 2 and 3d, symmetric and non-symmetric routines to calculate
 * entries in RAP. Incomplete error handling at the moment.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_SMGSetupRAPOp( nalu_hypre_StructMatrix *R,
                     nalu_hypre_StructMatrix *A,
                     nalu_hypre_StructMatrix *PT,
                     nalu_hypre_StructMatrix *Ac,
                     nalu_hypre_Index         cindex,
                     nalu_hypre_Index         cstride )
{
#if NEWRAP
   NALU_HYPRE_Int              cdir;
   NALU_HYPRE_Int              P_stored_as_transpose = 1;
#endif

   nalu_hypre_StructStencil   *stencil;
   nalu_hypre_StructMatrix    *Ac_tmp;
#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_MemoryLocation data_location_A = nalu_hypre_StructGridDataLocation(nalu_hypre_StructMatrixGrid(A));
   NALU_HYPRE_MemoryLocation data_location_Ac = nalu_hypre_StructGridDataLocation(nalu_hypre_StructMatrixGrid(Ac));
   if (data_location_A != data_location_Ac)
   {
      Ac_tmp = nalu_hypre_SMGCreateRAPOp(R, A, PT, nalu_hypre_StructMatrixGrid(Ac));
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
#if OLDRAP
   switch (nalu_hypre_StructStencilNDim(stencil))
   {

      case 2:

         /*--------------------------------------------------------------------
          *    Set lower triangular (+ diagonal) coefficients
          *--------------------------------------------------------------------*/
         nalu_hypre_SMG2BuildRAPSym(A, PT, R, Ac_tmp, cindex, cstride);

         /*--------------------------------------------------------------------
          *    For non-symmetric A, set upper triangular coefficients as well
          *--------------------------------------------------------------------*/
         if (!nalu_hypre_StructMatrixSymmetric(A))
         {
            nalu_hypre_SMG2BuildRAPNoSym(A, PT, R, Ac_tmp, cindex, cstride);
            /*-----------------------------------------------------------------
             *    Collapse stencil for periodic probems on coarsest grid.
             *-----------------------------------------------------------------*/
            nalu_hypre_SMG2RAPPeriodicNoSym(Ac_tmp, cindex, cstride);
         }
         else
         {
            /*-----------------------------------------------------------------
             *    Collapse stencil for periodic problems on coarsest grid.
             *-----------------------------------------------------------------*/
            nalu_hypre_SMG2RAPPeriodicSym(Ac_tmp, cindex, cstride);
         }

         break;

      case 3:

         /*--------------------------------------------------------------------
          *    Set lower triangular (+ diagonal) coefficients
          *--------------------------------------------------------------------*/
         nalu_hypre_SMG3BuildRAPSym(A, PT, R, Ac_tmp, cindex, cstride);

         /*--------------------------------------------------------------------
          *    For non-symmetric A, set upper triangular coefficients as well
          *--------------------------------------------------------------------*/
         if (!nalu_hypre_StructMatrixSymmetric(A))
         {
            nalu_hypre_SMG3BuildRAPNoSym(A, PT, R, Ac_tmp, cindex, cstride);
            /*-----------------------------------------------------------------
             *    Collapse stencil for periodic probems on coarsest grid.
             *-----------------------------------------------------------------*/
            nalu_hypre_SMG3RAPPeriodicNoSym(Ac_tmp, cindex, cstride);
         }
         else
         {
            /*-----------------------------------------------------------------
             *    Collapse stencil for periodic problems on coarsest grid.
             *-----------------------------------------------------------------*/
            nalu_hypre_SMG3RAPPeriodicSym(Ac_tmp, cindex, cstride);
         }

         break;

   }
#endif

#if NEWRAP
   switch (nalu_hypre_StructStencilNDim(stencil))
   {

      case 2:
         cdir = 1;
         nalu_hypre_SemiBuildRAP(A, PT, R, cdir, cindex, cstride,
                            P_stored_as_transpose, Ac_tmp);
         break;

      case 3:
         cdir = 2;
         nalu_hypre_SemiBuildRAP(A, PT, R, cdir, cindex, cstride,
                            P_stored_as_transpose, Ac_tmp);
         break;

   }
#endif

   nalu_hypre_StructMatrixAssemble(Ac_tmp);

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   if (data_location_A != data_location_Ac)
   {

      nalu_hypre_TMemcpy(nalu_hypre_StructMatrixDataConst(Ac), nalu_hypre_StructMatrixData(Ac_tmp), NALU_HYPRE_Complex,
                    nalu_hypre_StructMatrixDataSize(Ac_tmp), NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_SetDeviceOff();
      nalu_hypre_StructGridDataLocation(nalu_hypre_StructMatrixGrid(Ac)) = data_location_Ac;
      nalu_hypre_StructMatrixAssemble(Ac);
      nalu_hypre_SetDeviceOn();
      nalu_hypre_StructMatrixDestroy(Ac_tmp);
   }
#endif
   return nalu_hypre_error_flag;
}

