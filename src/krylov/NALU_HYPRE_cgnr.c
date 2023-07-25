/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_CGNR interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CGNRCreate does not exist.  Call the appropriate function which
 * also specifies the vector type, e.g. NALU_HYPRE_ParCSRCGNRCreate
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CGNRDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CGNRDestroy( NALU_HYPRE_Solver solver )
{
   return ( nalu_hypre_CGNRDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CGNRSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CGNRSetup( NALU_HYPRE_Solver solver,
                 NALU_HYPRE_Matrix A,
                 NALU_HYPRE_Vector b,
                 NALU_HYPRE_Vector x      )
{
   return ( nalu_hypre_CGNRSetup( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CGNRSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CGNRSolve( NALU_HYPRE_Solver solver,
                 NALU_HYPRE_Matrix A,
                 NALU_HYPRE_Vector b,
                 NALU_HYPRE_Vector x      )
{
   return ( nalu_hypre_CGNRSolve( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CGNRSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CGNRSetTol( NALU_HYPRE_Solver solver,
                  NALU_HYPRE_Real         tol    )
{
   return ( nalu_hypre_CGNRSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CGNRSetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CGNRSetMinIter( NALU_HYPRE_Solver solver,
                      NALU_HYPRE_Int                min_iter )
{
   return ( nalu_hypre_CGNRSetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CGNRSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CGNRSetMaxIter( NALU_HYPRE_Solver solver,
                      NALU_HYPRE_Int                max_iter )
{
   return ( nalu_hypre_CGNRSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CGNRSetStopCrit
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CGNRSetStopCrit( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Int                stop_crit )
{
   return ( nalu_hypre_CGNRSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CGNRSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CGNRSetPrecond( NALU_HYPRE_Solver         solver,
                      NALU_HYPRE_PtrToSolverFcn precond,
                      NALU_HYPRE_PtrToSolverFcn precondT,
                      NALU_HYPRE_PtrToSolverFcn precond_setup,
                      NALU_HYPRE_Solver         precond_solver )
{
   return ( nalu_hypre_CGNRSetPrecond( (void *) solver,
                                  (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond,
                                  (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precondT,
                                  (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
                                  (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CGNRGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CGNRGetPrecond( NALU_HYPRE_Solver   solver,
                      NALU_HYPRE_Solver  *precond_data_ptr )
{
   return ( nalu_hypre_CGNRGetPrecond( (void *)         solver,
                                  (NALU_HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CGNRSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CGNRSetLogging( NALU_HYPRE_Solver solver,
                      NALU_HYPRE_Int logging)
{
   return ( nalu_hypre_CGNRSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CGNRGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CGNRGetNumIterations( NALU_HYPRE_Solver  solver,
                            NALU_HYPRE_Int                *num_iterations )
{
   return ( nalu_hypre_CGNRGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CGNRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CGNRGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                        NALU_HYPRE_Real         *norm   )
{
   return ( nalu_hypre_CGNRGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}
