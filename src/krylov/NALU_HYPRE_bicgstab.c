/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_BiCGSTAB interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABCreate does not exist.  Call the appropriate function which
 * also specifies the vector type, e.g. NALU_HYPRE_ParCSRBiCGSTABCreate
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABDestroy( NALU_HYPRE_Solver solver )
{
   return ( nalu_hypre_BiCGSTABDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABSetup( NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Matrix A,
                     NALU_HYPRE_Vector b,
                     NALU_HYPRE_Vector x      )
{
   return ( nalu_hypre_BiCGSTABSetup( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABSolve( NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Matrix A,
                     NALU_HYPRE_Vector b,
                     NALU_HYPRE_Vector x      )
{
   return ( nalu_hypre_BiCGSTABSolve( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABSetTol( NALU_HYPRE_Solver solver,
                      NALU_HYPRE_Real         tol    )
{
   return ( nalu_hypre_BiCGSTABSetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABSetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABSetAbsoluteTol( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Real         a_tol    )
{
   return ( nalu_hypre_BiCGSTABSetAbsoluteTol( (void *) solver, a_tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABSetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABSetConvergenceFactorTol( NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Real         cf_tol    )
{
   return ( nalu_hypre_BiCGSTABSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABSetMinIter( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int          min_iter )
{
   return ( nalu_hypre_BiCGSTABSetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABSetMaxIter( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int          max_iter )
{
   return ( nalu_hypre_BiCGSTABSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABSetStopCrit( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int          stop_crit )
{
   return ( nalu_hypre_BiCGSTABSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABSetPrecond( NALU_HYPRE_Solver         solver,
                          NALU_HYPRE_PtrToSolverFcn precond,
                          NALU_HYPRE_PtrToSolverFcn precond_setup,
                          NALU_HYPRE_Solver         precond_solver )
{
   return ( nalu_hypre_BiCGSTABSetPrecond( (void *) solver,
                                      (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond,
                                      (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
                                      (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABGetPrecond( NALU_HYPRE_Solver  solver,
                          NALU_HYPRE_Solver *precond_data_ptr )
{
   return ( nalu_hypre_BiCGSTABGetPrecond( (void *)     solver,
                                      (NALU_HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABSetLogging( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int logging)
{
   return ( nalu_hypre_BiCGSTABSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABSetPrintLevel( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int print_level)
{
   return ( nalu_hypre_BiCGSTABSetPrintLevel( (void *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABGetNumIterations( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int                *num_iterations )
{
   return ( nalu_hypre_BiCGSTABGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                            NALU_HYPRE_Real         *norm   )
{
   return ( nalu_hypre_BiCGSTABGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BiCGSTABGetResidual( NALU_HYPRE_Solver  solver,
                           void             *residual  )
{
   return ( nalu_hypre_BiCGSTABGetResidual( (void *) solver, (void **) residual ) );
}
