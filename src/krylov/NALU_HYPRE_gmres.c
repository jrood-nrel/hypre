/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_GMRES interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESDestroy
 *--------------------------------------------------------------------------*/
/* to do, not trivial */
/*
NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESDestroy( NALU_HYPRE_Solver solver )
{
   return( hypre_GMRESDestroy( (void *) solver ) );
}
*/

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESSetup( NALU_HYPRE_Solver solver,
                  NALU_HYPRE_Matrix A,
                  NALU_HYPRE_Vector b,
                  NALU_HYPRE_Vector x      )
{
   return ( hypre_GMRESSetup( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESSolve( NALU_HYPRE_Solver solver,
                  NALU_HYPRE_Matrix A,
                  NALU_HYPRE_Vector b,
                  NALU_HYPRE_Vector x      )
{
   return ( hypre_GMRESSolve( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESSetKDim, NALU_HYPRE_GMRESGetKDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESSetKDim( NALU_HYPRE_Solver solver,
                    NALU_HYPRE_Int             k_dim    )
{
   return ( hypre_GMRESSetKDim( (void *) solver, k_dim ) );
}

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetKDim( NALU_HYPRE_Solver solver,
                    NALU_HYPRE_Int           * k_dim    )
{
   return ( hypre_GMRESGetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESSetTol, NALU_HYPRE_GMRESGetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESSetTol( NALU_HYPRE_Solver solver,
                   NALU_HYPRE_Real         tol    )
{
   return ( hypre_GMRESSetTol( (void *) solver, tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetTol( NALU_HYPRE_Solver solver,
                   NALU_HYPRE_Real       * tol    )
{
   return ( hypre_GMRESGetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESSetAbsoluteTol, NALU_HYPRE_GMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESSetAbsoluteTol( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Real         a_tol    )
{
   return ( hypre_GMRESSetAbsoluteTol( (void *) solver, a_tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetAbsoluteTol( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Real       * a_tol    )
{
   return ( hypre_GMRESGetAbsoluteTol( (void *) solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESSetConvergenceFactorTol, NALU_HYPRE_GMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESSetConvergenceFactorTol( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Real         cf_tol    )
{
   return ( hypre_GMRESSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetConvergenceFactorTol( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Real       * cf_tol    )
{
   return ( hypre_GMRESGetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESSetMinIter, NALU_HYPRE_GMRESGetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESSetMinIter( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Int          min_iter )
{
   return ( hypre_GMRESSetMinIter( (void *) solver, min_iter ) );
}

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetMinIter( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Int        * min_iter )
{
   return ( hypre_GMRESGetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESSetMaxIter, NALU_HYPRE_GMRESGetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESSetMaxIter( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Int          max_iter )
{
   return ( hypre_GMRESSetMaxIter( (void *) solver, max_iter ) );
}

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetMaxIter( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Int        * max_iter )
{
   return ( hypre_GMRESGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESSetStopCrit, NALU_HYPRE_GMRESGetStopCrit - OBSOLETE
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESSetStopCrit( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int          stop_crit )
{
   return ( hypre_GMRESSetStopCrit( (void *) solver, stop_crit ) );
}

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetStopCrit( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int        * stop_crit )
{
   return ( hypre_GMRESGetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESSetRelChange, NALU_HYPRE_GMRESGetRelChange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESSetRelChange( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Int                rel_change )
{
   return ( hypre_GMRESSetRelChange( (void *) solver, rel_change ) );
}

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetRelChange( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Int              * rel_change )
{
   return ( hypre_GMRESGetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESSetSkipRealResidualCheck, NALU_HYPRE_GMRESGetSkipRealResidualCheck
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESSetSkipRealResidualCheck( NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int skip_real_r_check )
{
   return ( hypre_GMRESSetSkipRealResidualCheck( (void *) solver, skip_real_r_check ) );
}

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetSkipRealResidualCheck( NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int *skip_real_r_check )
{
   return ( hypre_GMRESGetSkipRealResidualCheck( (void *) solver, skip_real_r_check ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESSetPrecond( NALU_HYPRE_Solver          solver,
                       NALU_HYPRE_PtrToSolverFcn  precond,
                       NALU_HYPRE_PtrToSolverFcn  precond_setup,
                       NALU_HYPRE_Solver          precond_solver )
{
   return ( hypre_GMRESSetPrecond( (void *) solver,
                                   (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond,
                                   (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
                                   (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetPrecond( NALU_HYPRE_Solver  solver,
                       NALU_HYPRE_Solver *precond_data_ptr )
{
   return ( hypre_GMRESGetPrecond( (void *)     solver,
                                   (NALU_HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESSetPrintLevel, NALU_HYPRE_GMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESSetPrintLevel( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int          level )
{
   return ( hypre_GMRESSetPrintLevel( (void *) solver, level ) );
}

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetPrintLevel( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int        * level )
{
   return ( hypre_GMRESGetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESSetLogging, NALU_HYPRE_GMRESGetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESSetLogging( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Int          level )
{
   return ( hypre_GMRESSetLogging( (void *) solver, level ) );
}

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetLogging( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Int        * level )
{
   return ( hypre_GMRESGetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetNumIterations( NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Int                *num_iterations )
{
   return ( hypre_GMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESGetConverged
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetConverged( NALU_HYPRE_Solver  solver,
                         NALU_HYPRE_Int                *converged )
{
   return ( hypre_GMRESGetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_GMRESGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                         NALU_HYPRE_Real         *norm   )
{
   return ( hypre_GMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_GMRESGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_GMRESGetResidual( NALU_HYPRE_Solver solver, void *residual )
{
   /* returns a pointer to the residual vector */
   return hypre_GMRESGetResidual( (void *) solver, (void **) residual );
}

