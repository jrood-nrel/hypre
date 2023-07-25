/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGCreate( MPI_Comm             comm,
                        NALU_HYPRE_SStructSolver *solver )
{
   nalu_hypre_PCGFunctions * pcg_functions =
      nalu_hypre_PCGFunctionsCreate(
         nalu_hypre_SStructKrylovCAlloc, nalu_hypre_SStructKrylovFree, nalu_hypre_SStructKrylovCommInfo,
         nalu_hypre_SStructKrylovCreateVector,
         nalu_hypre_SStructKrylovDestroyVector, nalu_hypre_SStructKrylovMatvecCreate,
         nalu_hypre_SStructKrylovMatvec, nalu_hypre_SStructKrylovMatvecDestroy,
         nalu_hypre_SStructKrylovInnerProd, nalu_hypre_SStructKrylovCopyVector,
         nalu_hypre_SStructKrylovClearVector,
         nalu_hypre_SStructKrylovScaleVector, nalu_hypre_SStructKrylovAxpy,
         nalu_hypre_SStructKrylovIdentitySetup, nalu_hypre_SStructKrylovIdentity );

   *solver = ( (NALU_HYPRE_SStructSolver) nalu_hypre_PCGCreate( pcg_functions ) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructPCGDestroy( NALU_HYPRE_SStructSolver solver )
{
   return ( nalu_hypre_PCGDestroy( (void *) solver ) );
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
   NALU_HYPRE_Int                nparts = nalu_hypre_SStructMatrixNParts(A);

   nalu_hypre_SStructPMatrix    *pA;
   nalu_hypre_SStructPVector    *px;
   nalu_hypre_SStructPVector    *py;
   nalu_hypre_StructMatrix      *sA;
   nalu_hypre_StructVector      *sx;
   nalu_hypre_StructVector      *sy;

   NALU_HYPRE_Int part, vi;
   NALU_HYPRE_Int nvars;

   for (part = 0; part < nparts; part++)
   {
      pA = nalu_hypre_SStructMatrixPMatrix(A, part);
      px = nalu_hypre_SStructVectorPVector(x, part);
      py = nalu_hypre_SStructVectorPVector(y, part);
      nvars = nalu_hypre_SStructPMatrixNVars(pA);
      for (vi = 0; vi < nvars; vi++)
      {
         sA = nalu_hypre_SStructPMatrixSMatrix(pA, vi, vi);
         sx = nalu_hypre_SStructPVectorSVector(px, vi);
         sy = nalu_hypre_SStructPVectorSVector(py, vi);

         NALU_HYPRE_StructDiagScale( (NALU_HYPRE_StructSolver) solver,
                                (NALU_HYPRE_StructMatrix) sA,
                                (NALU_HYPRE_StructVector) sy,
                                (NALU_HYPRE_StructVector) sx );
      }
   }

   return nalu_hypre_error_flag;
}



