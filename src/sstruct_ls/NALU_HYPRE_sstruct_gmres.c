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
NALU_HYPRE_SStructGMRESCreate( MPI_Comm             comm,
                          NALU_HYPRE_SStructSolver *solver )
{
   hypre_GMRESFunctions * gmres_functions =
      hypre_GMRESFunctionsCreate(
         hypre_SStructKrylovCAlloc, hypre_SStructKrylovFree, hypre_SStructKrylovCommInfo,
         hypre_SStructKrylovCreateVector,
         hypre_SStructKrylovCreateVectorArray,
         hypre_SStructKrylovDestroyVector, hypre_SStructKrylovMatvecCreate,
         hypre_SStructKrylovMatvec, hypre_SStructKrylovMatvecDestroy,
         hypre_SStructKrylovInnerProd, hypre_SStructKrylovCopyVector,
         hypre_SStructKrylovClearVector,
         hypre_SStructKrylovScaleVector, hypre_SStructKrylovAxpy,
         hypre_SStructKrylovIdentitySetup, hypre_SStructKrylovIdentity );

   *solver = ( (NALU_HYPRE_SStructSolver) hypre_GMRESCreate( gmres_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESDestroy( NALU_HYPRE_SStructSolver solver )
{
   return ( hypre_GMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetup( NALU_HYPRE_SStructSolver solver,
                         NALU_HYPRE_SStructMatrix A,
                         NALU_HYPRE_SStructVector b,
                         NALU_HYPRE_SStructVector x )
{
   return ( NALU_HYPRE_GMRESSetup( (NALU_HYPRE_Solver) solver,
                              (NALU_HYPRE_Matrix) A,
                              (NALU_HYPRE_Vector) b,
                              (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSolve( NALU_HYPRE_SStructSolver solver,
                         NALU_HYPRE_SStructMatrix A,
                         NALU_HYPRE_SStructVector b,
                         NALU_HYPRE_SStructVector x )
{
   return ( NALU_HYPRE_GMRESSolve( (NALU_HYPRE_Solver) solver,
                              (NALU_HYPRE_Matrix) A,
                              (NALU_HYPRE_Vector) b,
                              (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetKDim( NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_Int           k_dim )
{
   return ( NALU_HYPRE_GMRESSetKDim( (NALU_HYPRE_Solver) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetTol( NALU_HYPRE_SStructSolver solver,
                          NALU_HYPRE_Real          tol )
{
   return ( NALU_HYPRE_GMRESSetTol( (NALU_HYPRE_Solver) solver, tol ) );
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetAbsoluteTol( NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Real          atol )
{
   return ( NALU_HYPRE_GMRESSetAbsoluteTol( (NALU_HYPRE_Solver) solver, atol ) );
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetMinIter( NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int           min_iter )
{
   return ( NALU_HYPRE_GMRESSetMinIter( (NALU_HYPRE_Solver) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetMaxIter( NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int           max_iter )
{
   return ( NALU_HYPRE_GMRESSetMaxIter( (NALU_HYPRE_Solver) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetStopCrit( NALU_HYPRE_SStructSolver solver,
                               NALU_HYPRE_Int           stop_crit )
{
   return ( NALU_HYPRE_GMRESSetStopCrit( (NALU_HYPRE_Solver) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetPrecond( NALU_HYPRE_SStructSolver          solver,
                              NALU_HYPRE_PtrToSStructSolverFcn  precond,
                              NALU_HYPRE_PtrToSStructSolverFcn  precond_setup,
                              void *          precond_data )
{
   return ( NALU_HYPRE_GMRESSetPrecond( (NALU_HYPRE_Solver) solver,
                                   (NALU_HYPRE_PtrToSolverFcn) precond,
                                   (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                   (NALU_HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetLogging( NALU_HYPRE_SStructSolver solver,
                              NALU_HYPRE_Int           logging )
{
   return ( NALU_HYPRE_GMRESSetLogging( (NALU_HYPRE_Solver) solver, logging ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESSetPrintLevel( NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int           level )
{
   return ( NALU_HYPRE_GMRESSetPrintLevel( (NALU_HYPRE_Solver) solver, level ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESGetNumIterations( NALU_HYPRE_SStructSolver  solver,
                                    NALU_HYPRE_Int           *num_iterations )
{
   return ( NALU_HYPRE_GMRESGetNumIterations( (NALU_HYPRE_Solver) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESGetFinalRelativeResidualNorm( NALU_HYPRE_SStructSolver  solver,
                                                NALU_HYPRE_Real          *norm )
{
   return ( NALU_HYPRE_GMRESGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver) solver, norm ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructGMRESGetResidual( NALU_HYPRE_SStructSolver  solver,
                               void              **residual )
{
   return ( NALU_HYPRE_GMRESGetResidual( (NALU_HYPRE_Solver) solver, residual ) );
}
