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
NALU_HYPRE_SStructBiCGSTABCreate( MPI_Comm             comm,
                             NALU_HYPRE_SStructSolver *solver )
{
   nalu_hypre_BiCGSTABFunctions * bicgstab_functions =
      nalu_hypre_BiCGSTABFunctionsCreate(
         nalu_hypre_SStructKrylovCreateVector,
         nalu_hypre_SStructKrylovDestroyVector, nalu_hypre_SStructKrylovMatvecCreate,
         nalu_hypre_SStructKrylovMatvec, nalu_hypre_SStructKrylovMatvecDestroy,
         nalu_hypre_SStructKrylovInnerProd, nalu_hypre_SStructKrylovCopyVector,
         nalu_hypre_SStructKrylovClearVector,
         nalu_hypre_SStructKrylovScaleVector, nalu_hypre_SStructKrylovAxpy,
         nalu_hypre_SStructKrylovCommInfo,
         nalu_hypre_SStructKrylovIdentitySetup, nalu_hypre_SStructKrylovIdentity );

   *solver = ( (NALU_HYPRE_SStructSolver) nalu_hypre_BiCGSTABCreate( bicgstab_functions ) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABDestroy( NALU_HYPRE_SStructSolver solver )
{
   return ( nalu_hypre_BiCGSTABDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetup( NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_SStructMatrix A,
                            NALU_HYPRE_SStructVector b,
                            NALU_HYPRE_SStructVector x )
{
   return ( NALU_HYPRE_BiCGSTABSetup( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_Matrix) A,
                                 (NALU_HYPRE_Vector) b,
                                 (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSolve( NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_SStructMatrix A,
                            NALU_HYPRE_SStructVector b,
                            NALU_HYPRE_SStructVector x )
{
   return ( NALU_HYPRE_BiCGSTABSolve( (NALU_HYPRE_Solver) solver,
                                 (NALU_HYPRE_Matrix) A,
                                 (NALU_HYPRE_Vector) b,
                                 (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetTol( NALU_HYPRE_SStructSolver solver,
                             NALU_HYPRE_Real          tol )
{
   return ( NALU_HYPRE_BiCGSTABSetTol( (NALU_HYPRE_Solver) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetAbsoluteTol( NALU_HYPRE_SStructSolver solver,
                                     NALU_HYPRE_Real          tol )
{
   return ( NALU_HYPRE_BiCGSTABSetAbsoluteTol( (NALU_HYPRE_Solver) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetMinIter( NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int           min_iter )
{
   return ( NALU_HYPRE_BiCGSTABSetMinIter( (NALU_HYPRE_Solver) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetMaxIter( NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int           max_iter )
{
   return ( NALU_HYPRE_BiCGSTABSetMaxIter( (NALU_HYPRE_Solver) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetStopCrit( NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int           stop_crit )
{
   return ( NALU_HYPRE_BiCGSTABSetStopCrit( (NALU_HYPRE_Solver) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetPrecond( NALU_HYPRE_SStructSolver          solver,
                                 NALU_HYPRE_PtrToSStructSolverFcn  precond,
                                 NALU_HYPRE_PtrToSStructSolverFcn  precond_setup,
                                 void *          precond_data )
{
   return ( NALU_HYPRE_BiCGSTABSetPrecond( (NALU_HYPRE_Solver) solver,
                                      (NALU_HYPRE_PtrToSolverFcn) precond,
                                      (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                      (NALU_HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetLogging( NALU_HYPRE_SStructSolver solver,
                                 NALU_HYPRE_Int           logging )
{
   return ( NALU_HYPRE_BiCGSTABSetLogging( (NALU_HYPRE_Solver) solver, logging ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABSetPrintLevel( NALU_HYPRE_SStructSolver solver,
                                    NALU_HYPRE_Int           print_level )
{
   return ( NALU_HYPRE_BiCGSTABSetPrintLevel( (NALU_HYPRE_Solver) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABGetNumIterations( NALU_HYPRE_SStructSolver  solver,
                                       NALU_HYPRE_Int           *num_iterations )
{
   return ( NALU_HYPRE_BiCGSTABGetNumIterations( (NALU_HYPRE_Solver) solver,
                                            num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm( NALU_HYPRE_SStructSolver  solver,
                                                   NALU_HYPRE_Real          *norm )
{
   return ( NALU_HYPRE_BiCGSTABGetFinalRelativeResidualNorm( (NALU_HYPRE_Solver) solver,
                                                        norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_SStructBiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructBiCGSTABGetResidual( NALU_HYPRE_SStructSolver  solver,
                                  void          **residual)
{
   return ( NALU_HYPRE_BiCGSTABGetResidual( (NALU_HYPRE_Solver) solver, residual ) );
}
