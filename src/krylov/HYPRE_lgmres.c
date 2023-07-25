/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_LGMRES interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESDestroy
 *--------------------------------------------------------------------------*/
/* to do, not trivial */
/*
NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESDestroy( NALU_HYPRE_Solver solver )
{
   return( nalu_hypre_LGMRESDestroy( (void *) solver ) );
}
*/

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESSetup( NALU_HYPRE_Solver solver,
                   NALU_HYPRE_Matrix A,
                   NALU_HYPRE_Vector b,
                   NALU_HYPRE_Vector x      )
{
   return ( nalu_hypre_LGMRESSetup( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESSolve( NALU_HYPRE_Solver solver,
                   NALU_HYPRE_Matrix A,
                   NALU_HYPRE_Vector b,
                   NALU_HYPRE_Vector x      )
{
   return ( nalu_hypre_LGMRESSolve( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESSetKDim, NALU_HYPRE_LGMRESGetKDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESSetKDim( NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Int             k_dim    )
{
   return ( nalu_hypre_LGMRESSetKDim( (void *) solver, k_dim ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LGMRESGetKDim( NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Int           * k_dim    )
{
   return ( nalu_hypre_LGMRESGetKDim( (void *) solver, k_dim ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESSetAugDim, NALU_HYPRE_LGMRESGetAugDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESSetAugDim( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Int             aug_dim    )
{
   return ( nalu_hypre_LGMRESSetAugDim( (void *) solver, aug_dim ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LGMRESGetAugDim( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Int           * aug_dim    )
{
   return ( nalu_hypre_LGMRESGetAugDim( (void *) solver, aug_dim ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESSetTol, NALU_HYPRE_LGMRESGetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESSetTol( NALU_HYPRE_Solver solver,
                    NALU_HYPRE_Real         tol    )
{
   return ( nalu_hypre_LGMRESSetTol( (void *) solver, tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LGMRESGetTol( NALU_HYPRE_Solver solver,
                    NALU_HYPRE_Real       * tol    )
{
   return ( nalu_hypre_LGMRESGetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESSetAbsoluteTol, NALU_HYPRE_LGMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESSetAbsoluteTol( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Real         a_tol    )
{
   return ( nalu_hypre_LGMRESSetAbsoluteTol( (void *) solver, a_tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LGMRESGetAbsoluteTol( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Real       * a_tol    )
{
   return ( nalu_hypre_LGMRESGetAbsoluteTol( (void *) solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESSetConvergenceFactorTol, NALU_HYPRE_LGMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESSetConvergenceFactorTol( NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Real         cf_tol    )
{
   return ( nalu_hypre_LGMRESSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LGMRESGetConvergenceFactorTol( NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Real       * cf_tol    )
{
   return ( nalu_hypre_LGMRESGetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESSetMinIter, NALU_HYPRE_LGMRESGetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESSetMinIter( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int          min_iter )
{
   return ( nalu_hypre_LGMRESSetMinIter( (void *) solver, min_iter ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LGMRESGetMinIter( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int        * min_iter )
{
   return ( nalu_hypre_LGMRESGetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESSetMaxIter, NALU_HYPRE_LGMRESGetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESSetMaxIter( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int          max_iter )
{
   return ( nalu_hypre_LGMRESSetMaxIter( (void *) solver, max_iter ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LGMRESGetMaxIter( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int        * max_iter )
{
   return ( nalu_hypre_LGMRESGetMaxIter( (void *) solver, max_iter ) );
}



/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESSetPrecond( NALU_HYPRE_Solver          solver,
                        NALU_HYPRE_PtrToSolverFcn  precond,
                        NALU_HYPRE_PtrToSolverFcn  precond_setup,
                        NALU_HYPRE_Solver          precond_solver )
{
   return ( nalu_hypre_LGMRESSetPrecond( (void *) solver,
                                    (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond,
                                    (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
                                    (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESGetPrecond( NALU_HYPRE_Solver  solver,
                        NALU_HYPRE_Solver *precond_data_ptr )
{
   return ( nalu_hypre_LGMRESGetPrecond( (void *)     solver,
                                    (NALU_HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESSetPrintLevel, NALU_HYPRE_LGMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESSetPrintLevel( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int          level )
{
   return ( nalu_hypre_LGMRESSetPrintLevel( (void *) solver, level ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LGMRESGetPrintLevel( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int        * level )
{
   return ( nalu_hypre_LGMRESGetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESSetLogging, NALU_HYPRE_LGMRESGetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESSetLogging( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int          level )
{
   return ( nalu_hypre_LGMRESSetLogging( (void *) solver, level ) );
}

NALU_HYPRE_Int
NALU_HYPRE_LGMRESGetLogging( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int        * level )
{
   return ( nalu_hypre_LGMRESGetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESGetNumIterations( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int                *num_iterations )
{
   return ( nalu_hypre_LGMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESGetConverged
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESGetConverged( NALU_HYPRE_Solver  solver,
                          NALU_HYPRE_Int                *converged )
{
   return ( nalu_hypre_LGMRESGetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_LGMRESGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                          NALU_HYPRE_Real         *norm   )
{
   return ( nalu_hypre_LGMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_LGMRESGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_LGMRESGetResidual( NALU_HYPRE_Solver solver, void *residual )
{
   /* returns a pointer to the residual vector */
   return nalu_hypre_LGMRESGetResidual( (void *) solver, (void **) residual );
}

