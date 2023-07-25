/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"
#include "_nalu_hypre_struct_mv.hpp"

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGCreate( MPI_Comm comm, NALU_HYPRE_StructSolver *solver )
{
   /* The function names with a PCG in them are in
      struct_ls/pcg_struct.c .  These functions do rather little -
      e.g., cast to the correct type - before calling something else.
      These names should be called, e.g., nalu_hypre_struct_Free, to reduce the
      chance of name conflicts. */
   nalu_hypre_PCGFunctions * pcg_functions =
      nalu_hypre_PCGFunctionsCreate(
         nalu_hypre_StructKrylovCAlloc, nalu_hypre_StructKrylovFree,
         nalu_hypre_StructKrylovCommInfo,
         nalu_hypre_StructKrylovCreateVector,
         nalu_hypre_StructKrylovDestroyVector, nalu_hypre_StructKrylovMatvecCreate,
         nalu_hypre_StructKrylovMatvec, nalu_hypre_StructKrylovMatvecDestroy,
         nalu_hypre_StructKrylovInnerProd, nalu_hypre_StructKrylovCopyVector,
         nalu_hypre_StructKrylovClearVector,
         nalu_hypre_StructKrylovScaleVector, nalu_hypre_StructKrylovAxpy,
         nalu_hypre_StructKrylovIdentitySetup, nalu_hypre_StructKrylovIdentity );

   *solver = ( (NALU_HYPRE_StructSolver) nalu_hypre_PCGCreate( pcg_functions ) );

   return nalu_hypre_error_flag;
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGDestroy( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_PCGDestroy( (void *) solver ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGSetup( NALU_HYPRE_StructSolver solver,
                      NALU_HYPRE_StructMatrix A,
                      NALU_HYPRE_StructVector b,
                      NALU_HYPRE_StructVector x      )
{
   return ( NALU_HYPRE_PCGSetup( (NALU_HYPRE_Solver) solver,
                            (NALU_HYPRE_Matrix) A,
                            (NALU_HYPRE_Vector) b,
                            (NALU_HYPRE_Vector) x ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGSolve( NALU_HYPRE_StructSolver solver,
                      NALU_HYPRE_StructMatrix A,
                      NALU_HYPRE_StructVector b,
                      NALU_HYPRE_StructVector x      )
{
   return ( NALU_HYPRE_PCGSolve( (NALU_HYPRE_Solver) solver,
                            (NALU_HYPRE_Matrix) A,
                            (NALU_HYPRE_Vector) b,
                            (NALU_HYPRE_Vector) x ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGSetTol( NALU_HYPRE_StructSolver solver,
                       NALU_HYPRE_Real         tol    )
{
   return ( NALU_HYPRE_PCGSetTol( (NALU_HYPRE_Solver) solver, tol ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGSetAbsoluteTol( NALU_HYPRE_StructSolver solver,
                               NALU_HYPRE_Real         tol    )
{
   return ( NALU_HYPRE_PCGSetAbsoluteTol( (NALU_HYPRE_Solver) solver, tol ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGSetMaxIter( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_Int          max_iter )
{
   return ( NALU_HYPRE_PCGSetMaxIter( (NALU_HYPRE_Solver) solver, max_iter ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGSetTwoNorm( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_Int          two_norm )
{
   return ( NALU_HYPRE_PCGSetTwoNorm( (NALU_HYPRE_Solver) solver, two_norm ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGSetRelChange( NALU_HYPRE_StructSolver solver,
                             NALU_HYPRE_Int          rel_change )
{
   return ( NALU_HYPRE_PCGSetRelChange( (NALU_HYPRE_Solver) solver, rel_change ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGSetPrecond( NALU_HYPRE_StructSolver         solver,
                           NALU_HYPRE_PtrToStructSolverFcn precond,
                           NALU_HYPRE_PtrToStructSolverFcn precond_setup,
                           NALU_HYPRE_StructSolver         precond_solver )
{
   return ( NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) precond,
                                 (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                 (NALU_HYPRE_Solver) precond_solver ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGSetLogging( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_Int          logging )
{
   return ( NALU_HYPRE_PCGSetLogging( (NALU_HYPRE_Solver) solver, logging ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGSetPrintLevel( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int      print_level )
{
   return ( NALU_HYPRE_PCGSetPrintLevel( (NALU_HYPRE_Solver) solver, print_level ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGGetNumIterations( NALU_HYPRE_StructSolver  solver,
                                 NALU_HYPRE_Int          *num_iterations )
{
   return ( NALU_HYPRE_PCGGetNumIterations( (NALU_HYPRE_Solver) solver, num_iterations ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGGetFinalRelativeResidualNorm( NALU_HYPRE_StructSolver  solver,
                                             NALU_HYPRE_Real         *norm   )
{
   return ( NALU_HYPRE_PCGGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver) solver, norm ) );
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructDiagScaleSetup( NALU_HYPRE_StructSolver solver,
                            NALU_HYPRE_StructMatrix A,
                            NALU_HYPRE_StructVector y,
                            NALU_HYPRE_StructVector x      )
{
   return nalu_hypre_error_flag;
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructDiagScale( NALU_HYPRE_StructSolver solver,
                       NALU_HYPRE_StructMatrix HA,
                       NALU_HYPRE_StructVector Hy,
                       NALU_HYPRE_StructVector Hx      )
{
   nalu_hypre_StructMatrix   *A = (nalu_hypre_StructMatrix *) HA;
   nalu_hypre_StructVector   *y = (nalu_hypre_StructVector *) Hy;
   nalu_hypre_StructVector   *x = (nalu_hypre_StructVector *) Hx;

   nalu_hypre_BoxArray       *boxes;
   nalu_hypre_Box            *box;

   nalu_hypre_Box            *A_data_box;
   nalu_hypre_Box            *y_data_box;
   nalu_hypre_Box            *x_data_box;

   NALU_HYPRE_Real           *Ap;
   NALU_HYPRE_Real           *yp;
   NALU_HYPRE_Real           *xp;

   nalu_hypre_Index           index;
   nalu_hypre_IndexRef        start;
   nalu_hypre_Index           stride;
   nalu_hypre_Index           loop_size;

   NALU_HYPRE_Int             i;

   /* x = D^{-1} y */
   nalu_hypre_SetIndex(stride, 1);
   boxes = nalu_hypre_StructGridBoxes(nalu_hypre_StructMatrixGrid(A));
   nalu_hypre_ForBoxI(i, boxes)
   {
      box = nalu_hypre_BoxArrayBox(boxes, i);

      A_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructMatrixDataSpace(A), i);
      x_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(x), i);
      y_data_box = nalu_hypre_BoxArrayBox(nalu_hypre_StructVectorDataSpace(y), i);

      nalu_hypre_SetIndex(index, 0);
      Ap = nalu_hypre_StructMatrixExtractPointerByIndex(A, i, index);
      xp = nalu_hypre_StructVectorBoxData(x, i);
      yp = nalu_hypre_StructVectorBoxData(y, i);

      start  = nalu_hypre_BoxIMin(box);

      nalu_hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(xp,yp,Ap)
      nalu_hypre_BoxLoop3Begin(nalu_hypre_StructVectorNDim(Hx), loop_size,
                          A_data_box, start, stride, Ai,
                          x_data_box, start, stride, xi,
                          y_data_box, start, stride, yi);
      {
         xp[xi] = yp[yi] / Ap[Ai];
      }
      nalu_hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR
   }

   return nalu_hypre_error_flag;
}

