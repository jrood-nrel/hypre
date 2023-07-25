/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_PCG interface
 *
 *****************************************************************************/

#include "krylov.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGCreate: Call class-specific function, e.g. NALU_HYPRE_ParCSRPCGCreate
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGDestroy: Call class-specific function
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetup( NALU_HYPRE_Solver solver,
                NALU_HYPRE_Matrix A,
                NALU_HYPRE_Vector b,
                NALU_HYPRE_Vector x )
{
   return ( nalu_hypre_PCGSetup( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSolve( NALU_HYPRE_Solver solver,
                NALU_HYPRE_Matrix A,
                NALU_HYPRE_Vector b,
                NALU_HYPRE_Vector x )
{
   return ( nalu_hypre_PCGSolve( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetTol, NALU_HYPRE_PCGGetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetTol( NALU_HYPRE_Solver solver,
                 NALU_HYPRE_Real   tol )
{
   return ( nalu_hypre_PCGSetTol( (void *) solver, tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_PCGGetTol( NALU_HYPRE_Solver  solver,
                 NALU_HYPRE_Real   *tol )
{
   return ( nalu_hypre_PCGGetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetAbsoluteTol, NALU_HYPRE_PCGGetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetAbsoluteTol( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Real   a_tol )
{
   return ( nalu_hypre_PCGSetAbsoluteTol( (void *) solver, a_tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_PCGGetAbsoluteTol( NALU_HYPRE_Solver  solver,
                         NALU_HYPRE_Real   *a_tol )
{
   return ( nalu_hypre_PCGGetAbsoluteTol( (void *) solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetResidualTol, NALU_HYPRE_PCGGetResidualTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetResidualTol( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Real   rtol )
{
   return ( nalu_hypre_PCGSetResidualTol( (void *) solver, rtol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_PCGGetResidualTol( NALU_HYPRE_Solver  solver,
                         NALU_HYPRE_Real   *rtol )
{
   return ( nalu_hypre_PCGGetResidualTol( (void *) solver, rtol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetAbsoluteTolFactor, NALU_HYPRE_PCGGetAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetAbsoluteTolFactor( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Real   abstolf )
{
   return ( nalu_hypre_PCGSetAbsoluteTolFactor( (void *) solver, abstolf ) );
}

NALU_HYPRE_Int
NALU_HYPRE_PCGGetAbsoluteTolFactor( NALU_HYPRE_Solver  solver,
                               NALU_HYPRE_Real   *abstolf )
{
   return ( nalu_hypre_PCGGetAbsoluteTolFactor( (void *) solver, abstolf ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetConvergenceFactorTol, NALU_HYPRE_PCGGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetConvergenceFactorTol( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Real   cf_tol )
{
   return nalu_hypre_PCGSetConvergenceFactorTol( (void *) solver,
                                            cf_tol );
}

NALU_HYPRE_Int
NALU_HYPRE_PCGGetConvergenceFactorTol( NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Real   *cf_tol )
{
   return nalu_hypre_PCGGetConvergenceFactorTol( (void *) solver,
                                            cf_tol );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetMaxIter, NALU_HYPRE_PCGGetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetMaxIter( NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Int    max_iter )
{
   return ( nalu_hypre_PCGSetMaxIter( (void *) solver, max_iter ) );
}

NALU_HYPRE_Int
NALU_HYPRE_PCGGetMaxIter( NALU_HYPRE_Solver  solver,
                     NALU_HYPRE_Int    *max_iter )
{
   return ( nalu_hypre_PCGGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetStopCrit, NALU_HYPRE_PCGGetStopCrit
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetStopCrit( NALU_HYPRE_Solver solver,
                      NALU_HYPRE_Int    stop_crit )
{
   return ( nalu_hypre_PCGSetStopCrit( (void *) solver, stop_crit ) );
}

NALU_HYPRE_Int
NALU_HYPRE_PCGGetStopCrit( NALU_HYPRE_Solver  solver,
                      NALU_HYPRE_Int    *stop_crit )
{
   return ( nalu_hypre_PCGGetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetTwoNorm, NALU_HYPRE_PCGGetTwoNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetTwoNorm( NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Int    two_norm )
{
   return ( nalu_hypre_PCGSetTwoNorm( (void *) solver, two_norm ) );
}

NALU_HYPRE_Int
NALU_HYPRE_PCGGetTwoNorm( NALU_HYPRE_Solver  solver,
                     NALU_HYPRE_Int    *two_norm )
{
   return ( nalu_hypre_PCGGetTwoNorm( (void *) solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetRelChange, NALU_HYPRE_PCGGetRelChange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetRelChange( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Int    rel_change )
{
   return ( nalu_hypre_PCGSetRelChange( (void *) solver, rel_change ) );
}

NALU_HYPRE_Int
NALU_HYPRE_PCGGetRelChange( NALU_HYPRE_Solver  solver,
                       NALU_HYPRE_Int    *rel_change )
{
   return ( nalu_hypre_PCGGetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetRecomputeResidual, NALU_HYPRE_PCGGetRecomputeResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetRecomputeResidual( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int    recompute_residual )
{
   return ( nalu_hypre_PCGSetRecomputeResidual( (void *) solver, recompute_residual ) );
}

NALU_HYPRE_Int
NALU_HYPRE_PCGGetRecomputeResidual( NALU_HYPRE_Solver  solver,
                               NALU_HYPRE_Int    *recompute_residual )
{
   return ( nalu_hypre_PCGGetRecomputeResidual( (void *) solver, recompute_residual ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetRecomputeResidualP, NALU_HYPRE_PCGGetRecomputeResidualP
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetRecomputeResidualP( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    recompute_residual_p )
{
   return ( nalu_hypre_PCGSetRecomputeResidualP( (void *) solver, recompute_residual_p ) );
}

NALU_HYPRE_Int
NALU_HYPRE_PCGGetRecomputeResidualP( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int    *recompute_residual_p )
{
   return ( nalu_hypre_PCGGetRecomputeResidualP( (void *) solver, recompute_residual_p ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetPrecond( NALU_HYPRE_Solver         solver,
                     NALU_HYPRE_PtrToSolverFcn precond,
                     NALU_HYPRE_PtrToSolverFcn precond_setup,
                     NALU_HYPRE_Solver         precond_solver )
{
   return ( nalu_hypre_PCGSetPrecond( (void *) solver,
                                 (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond,
                                 (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
                                 (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGGetPrecond( NALU_HYPRE_Solver  solver,
                     NALU_HYPRE_Solver *precond_data_ptr )
{
   return ( nalu_hypre_PCGGetPrecond( (void *)     solver,
                                 (NALU_HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetLogging, NALU_HYPRE_PCGGetLogging
 * SetLogging sets both the print and log level, for backwards compatibility.
 * Soon the SetPrintLevel call should be deleted.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetLogging( NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Int    level )
{
   return ( nalu_hypre_PCGSetLogging( (void *) solver, level ) );
}

NALU_HYPRE_Int
NALU_HYPRE_PCGGetLogging( NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Int        * level )
{
   return ( nalu_hypre_PCGGetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGSetPrintLevel, NALU_HYPRE_PCGGetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGSetPrintLevel( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int    level )
{
   return ( nalu_hypre_PCGSetPrintLevel( (void *) solver, level ) );
}

NALU_HYPRE_Int
NALU_HYPRE_PCGGetPrintLevel( NALU_HYPRE_Solver  solver,
                        NALU_HYPRE_Int    *level )
{
   return ( nalu_hypre_PCGGetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGGetNumIterations( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Int    *num_iterations )
{
   return ( nalu_hypre_PCGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGGetConverged
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGGetConverged( NALU_HYPRE_Solver  solver,
                       NALU_HYPRE_Int    *converged )
{
   return ( nalu_hypre_PCGGetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_PCGGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                       NALU_HYPRE_Real   *norm )
{
   return ( nalu_hypre_PCGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_PCGGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_PCGGetResidual( NALU_HYPRE_Solver   solver,
                                void         *residual )
{
   /* returns a pointer to the residual vector */
   return nalu_hypre_PCGGetResidual( (void *) solver, (void **) residual );
}

