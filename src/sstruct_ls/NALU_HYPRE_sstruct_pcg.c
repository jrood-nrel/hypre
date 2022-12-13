/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGCreate( MPI_Comm             comm,
                        NALU_HYPRE_SStructSolver *solver )
{
   hypre_PCGFunctions * pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_SStructKrylovCAlloc, hypre_SStructKrylovFree, hypre_SStructKrylovCommInfo,
         hypre_SStructKrylovCreateVector,
         hypre_SStructKrylovDestroyVector, hypre_SStructKrylovMatvecCreate,
         hypre_SStructKrylovMatvec, hypre_SStructKrylovMatvecDestroy,
         hypre_SStructKrylovInnerProd, hypre_SStructKrylovCopyVector,
         hypre_SStructKrylovClearVector,
         hypre_SStructKrylovScaleVector, hypre_SStructKrylovAxpy,
         hypre_SStructKrylovIdentitySetup, hypre_SStructKrylovIdentity );

   *solver = ( (NALU_HYPRE_SStructSolver) hypre_PCGCreate( pcg_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGDestroy( NALU_HYPRE_SStructSolver solver )
{
   return ( hypre_PCGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetup( NALU_HYPRE_SStructSolver solver,
                       NALU_HYPRE_SStructMatrix A,
                       NALU_HYPRE_SStructVector b,
                       NALU_HYPRE_SStructVector x )
{
   return ( NALU_HYPRE_PCGSetup( (NALU_HYPRE_Solver) solver,
                            (NALU_HYPRE_Matrix) A,
                            (NALU_HYPRE_Vector) b,
                            (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSolve( NALU_HYPRE_SStructSolver solver,
                       NALU_HYPRE_SStructMatrix A,
                       NALU_HYPRE_SStructVector b,
                       NALU_HYPRE_SStructVector x )
{
   return ( NALU_HYPRE_PCGSolve( (NALU_HYPRE_Solver) solver,
                            (NALU_HYPRE_Matrix) A,
                            (NALU_HYPRE_Vector) b,
                            (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetTol( NALU_HYPRE_SStructSolver solver,
                        NALU_HYPRE_Real          tol )
{
   return ( NALU_HYPRE_PCGSetTol( (NALU_HYPRE_Solver) solver, tol ) );
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetAbsoluteTol( NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Real          tol )
{
   return ( NALU_HYPRE_PCGSetAbsoluteTol( (NALU_HYPRE_Solver) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetMaxIter( NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_Int           max_iter )
{
   return ( NALU_HYPRE_PCGSetMaxIter( (NALU_HYPRE_Solver) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetTwoNorm( NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_Int           two_norm )
{
   return ( NALU_HYPRE_PCGSetTwoNorm( (NALU_HYPRE_Solver) solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetRelChange( NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int           rel_change )
{
   return ( NALU_HYPRE_PCGSetRelChange( (NALU_HYPRE_Solver) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetPrecond( NALU_HYPRE_SStructSolver          solver,
                            NALU_HYPRE_PtrToSStructSolverFcn  precond,
                            NALU_HYPRE_PtrToSStructSolverFcn  precond_setup,
                            void                        *precond_data )
{
   return ( NALU_HYPRE_PCGSetPrecond( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_PtrToSolverFcn) precond,
                                 (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                 (NALU_HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetLogging( NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_Int           logging )
{
   return ( NALU_HYPRE_PCGSetLogging( (NALU_HYPRE_Solver) solver, logging ) );
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGSetPrintLevel( NALU_HYPRE_SStructSolver solver,
                               NALU_HYPRE_Int           level )
{
   return ( NALU_HYPRE_PCGSetPrintLevel( (NALU_HYPRE_Solver) solver, level ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGGetNumIterations( NALU_HYPRE_SStructSolver  solver,
                                  NALU_HYPRE_Int           *num_iterations )
{
   return ( NALU_HYPRE_PCGGetNumIterations( (NALU_HYPRE_Solver) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGGetFinalRelativeResidualNorm( NALU_HYPRE_SStructSolver  solver,
                                              NALU_HYPRE_Real          *norm )
{
   return ( NALU_HYPRE_PCGGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver) solver, norm ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGGetResidual( NALU_HYPRE_SStructSolver  solver,
                             void              **residual )
{
   return ( NALU_HYPRE_PCGGetResidual( (NALU_HYPRE_Solver) solver, residual ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructDiagScaleSetup( NALU_HYPRE_SStructSolver solver,
                             NALU_HYPRE_SStructMatrix A,
                             NALU_HYPRE_SStructVector y,
                             NALU_HYPRE_SStructVector x      )
{

   return ( NALU_HYPRE_StructDiagScaleSetup( (NALU_HYPRE_StructSolver) solver,
                                        (NALU_HYPRE_StructMatrix) A,
                                        (NALU_HYPRE_StructVector) y,
                                        (NALU_HYPRE_StructVector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructDiagScale( NALU_HYPRE_SStructSolver solver,
                        NALU_HYPRE_SStructMatrix A,
                        NALU_HYPRE_SStructVector y,
                        NALU_HYPRE_SStructVector x      )
{
   NALU_HYPRE_Int                nparts = hypre_SStructMatrixNParts(A);

   hypre_SStructPMatrix    *pA;
   hypre_SStructPVector    *px;
   hypre_SStructPVector    *py;
   hypre_StructMatrix      *sA;
   hypre_StructVector      *sx;
   hypre_StructVector      *sy;

   NALU_HYPRE_Int part, vi;
   NALU_HYPRE_Int nvars;

   for (part = 0; part < nparts; part++)
   {
      pA = hypre_SStructMatrixPMatrix(A, part);
      px = hypre_SStructVectorPVector(x, part);
      py = hypre_SStructVectorPVector(y, part);
      nvars = hypre_SStructPMatrixNVars(pA);
      for (vi = 0; vi < nvars; vi++)
      {
         sA = hypre_SStructPMatrixSMatrix(pA, vi, vi);
         sx = hypre_SStructPVectorSVector(px, vi);
         sy = hypre_SStructPVectorSVector(py, vi);

         NALU_HYPRE_StructDiagScale( (NALU_HYPRE_StructSolver) solver,
                                (NALU_HYPRE_StructMatrix) sA,
                                (NALU_HYPRE_StructVector) sy,
                                (NALU_HYPRE_StructVector) sx );
      }
   }

   return hypre_error_flag;
}



