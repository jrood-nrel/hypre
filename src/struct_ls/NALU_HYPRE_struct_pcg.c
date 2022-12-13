/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGCreate( MPI_Comm comm, NALU_HYPRE_StructSolver *solver )
{
   /* The function names with a PCG in them are in
      struct_ls/pcg_struct.c .  These functions do rather little -
      e.g., cast to the correct type - before calling something else.
      These names should be called, e.g., hypre_struct_Free, to reduce the
      chance of name conflicts. */
   hypre_PCGFunctions * pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_StructKrylovCAlloc, hypre_StructKrylovFree,
         hypre_StructKrylovCommInfo,
         hypre_StructKrylovCreateVector,
         hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
         hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
         hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
         hypre_StructKrylovClearVector,
         hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
         hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );

   *solver = ( (NALU_HYPRE_StructSolver) hypre_PCGCreate( pcg_functions ) );

   return hypre_error_flag;
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructPCGDestroy( NALU_HYPRE_StructSolver solver )
{
   return ( hypre_PCGDestroy( (void *) solver ) );
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
   return hypre_error_flag;
}

/*==========================================================================*/

NALU_HYPRE_Int
NALU_HYPRE_StructDiagScale( NALU_HYPRE_StructSolver solver,
                       NALU_HYPRE_StructMatrix HA,
                       NALU_HYPRE_StructVector Hy,
                       NALU_HYPRE_StructVector Hx      )
{
   hypre_StructMatrix   *A = (hypre_StructMatrix *) HA;
   hypre_StructVector   *y = (hypre_StructVector *) Hy;
   hypre_StructVector   *x = (hypre_StructVector *) Hx;

   hypre_BoxArray       *boxes;
   hypre_Box            *box;

   hypre_Box            *A_data_box;
   hypre_Box            *y_data_box;
   hypre_Box            *x_data_box;

   NALU_HYPRE_Real           *Ap;
   NALU_HYPRE_Real           *yp;
   NALU_HYPRE_Real           *xp;

   hypre_Index           index;
   hypre_IndexRef        start;
   hypre_Index           stride;
   hypre_Index           loop_size;

   NALU_HYPRE_Int             i;

   /* x = D^{-1} y */
   hypre_SetIndex(stride, 1);
   boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);

      A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

      hypre_SetIndex(index, 0);
      Ap = hypre_StructMatrixExtractPointerByIndex(A, i, index);
      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);

      start  = hypre_BoxIMin(box);

      hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(xp,yp,Ap)
      hypre_BoxLoop3Begin(hypre_StructVectorNDim(Hx), loop_size,
                          A_data_box, start, stride, Ai,
                          x_data_box, start, stride, xi,
                          y_data_box, start, stride, yi);
      {
         xp[xi] = yp[yi] / Ap[Ai];
      }
      hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR
   }

   return hypre_error_flag;
}

