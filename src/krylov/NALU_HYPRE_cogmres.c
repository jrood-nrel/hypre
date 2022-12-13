/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_COGMRES interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESDestroy
 *--------------------------------------------------------------------------*/
/* to do, not trivial */
/*
NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESDestroy( NALU_HYPRE_Solver solver )
{
   return( hypre_COGMRESDestroy( (void *) solver ) );
}
*/

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESSetup( NALU_HYPRE_Solver solver,
                    NALU_HYPRE_Matrix A,
                    NALU_HYPRE_Vector b,
                    NALU_HYPRE_Vector x      )
{
   return ( hypre_COGMRESSetup( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESSolve( NALU_HYPRE_Solver solver,
                    NALU_HYPRE_Matrix A,
                    NALU_HYPRE_Vector b,
                    NALU_HYPRE_Vector x      )
{
   return ( hypre_COGMRESSolve( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESSetKDim, NALU_HYPRE_COGMRESGetKDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESSetKDim( NALU_HYPRE_Solver solver,
                      NALU_HYPRE_Int             k_dim    )
{
   return ( hypre_COGMRESSetKDim( (void *) solver, k_dim ) );
}

NALU_HYPRE_Int
NALU_HYPRE_COGMRESGetKDim( NALU_HYPRE_Solver solver,
                      NALU_HYPRE_Int           * k_dim    )
{
   return ( hypre_COGMRESGetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESSetUnroll, NALU_HYPRE_COGMRESGetUnroll
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESSetUnroll( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int             unroll    )
{
   return ( hypre_COGMRESSetUnroll( (void *) solver, unroll ) );
}

NALU_HYPRE_Int
NALU_HYPRE_COGMRESGetUnroll( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int           * unroll    )
{
   return ( hypre_COGMRESGetUnroll( (void *) solver, unroll ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESSetCGS, NALU_HYPRE_COGMRESGetCGS
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESSetCGS( NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Int             cgs    )
{
   return ( hypre_COGMRESSetCGS( (void *) solver, cgs ) );
}

NALU_HYPRE_Int
NALU_HYPRE_COGMRESGetCGS( NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Int           * cgs    )
{
   return ( hypre_COGMRESGetCGS( (void *) solver, cgs ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESSetTol, NALU_HYPRE_COGMRESGetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESSetTol( NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Real         tol    )
{
   return ( hypre_COGMRESSetTol( (void *) solver, tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_COGMRESGetTol( NALU_HYPRE_Solver solver,
                     NALU_HYPRE_Real       * tol    )
{
   return ( hypre_COGMRESGetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESSetAbsoluteTol, NALU_HYPRE_COGMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESSetAbsoluteTol( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Real         a_tol    )
{
   return ( hypre_COGMRESSetAbsoluteTol( (void *) solver, a_tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_COGMRESGetAbsoluteTol( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Real       * a_tol    )
{
   return ( hypre_COGMRESGetAbsoluteTol( (void *) solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESSetConvergenceFactorTol, NALU_HYPRE_COGMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESSetConvergenceFactorTol( NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Real         cf_tol    )
{
   return ( hypre_COGMRESSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_COGMRESGetConvergenceFactorTol( NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Real       * cf_tol    )
{
   return ( hypre_COGMRESGetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESSetMinIter, NALU_HYPRE_COGMRESGetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESSetMinIter( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Int          min_iter )
{
   return ( hypre_COGMRESSetMinIter( (void *) solver, min_iter ) );
}

NALU_HYPRE_Int
NALU_HYPRE_COGMRESGetMinIter( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Int        * min_iter )
{
   return ( hypre_COGMRESGetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESSetMaxIter, NALU_HYPRE_COGMRESGetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESSetMaxIter( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Int          max_iter )
{
   return ( hypre_COGMRESSetMaxIter( (void *) solver, max_iter ) );
}

NALU_HYPRE_Int
NALU_HYPRE_COGMRESGetMaxIter( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Int        * max_iter )
{
   return ( hypre_COGMRESGetMaxIter( (void *) solver, max_iter ) );
}



/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESSetPrecond( NALU_HYPRE_Solver          solver,
                         NALU_HYPRE_PtrToSolverFcn  precond,
                         NALU_HYPRE_PtrToSolverFcn  precond_setup,
                         NALU_HYPRE_Solver          precond_solver )
{
   return ( hypre_COGMRESSetPrecond( (void *) solver,
                                     (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond,
                                     (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
                                     (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESGetPrecond( NALU_HYPRE_Solver  solver,
                         NALU_HYPRE_Solver *precond_data_ptr )
{
   return ( hypre_COGMRESGetPrecond( (void *)     solver,
                                     (NALU_HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESSetPrintLevel, NALU_HYPRE_COGMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESSetPrintLevel( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int          level )
{
   return ( hypre_COGMRESSetPrintLevel( (void *) solver, level ) );
}

NALU_HYPRE_Int
NALU_HYPRE_COGMRESGetPrintLevel( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int        * level )
{
   return ( hypre_COGMRESGetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESSetLogging, NALU_HYPRE_COGMRESGetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESSetLogging( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Int          level )
{
   return ( hypre_COGMRESSetLogging( (void *) solver, level ) );
}

NALU_HYPRE_Int
NALU_HYPRE_COGMRESGetLogging( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Int        * level )
{
   return ( hypre_COGMRESGetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESGetNumIterations( NALU_HYPRE_Solver  solver,
                               NALU_HYPRE_Int                *num_iterations )
{
   return ( hypre_COGMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESGetConverged
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESGetConverged( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Int                *converged )
{
   return ( hypre_COGMRESGetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_COGMRESGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                           NALU_HYPRE_Real         *norm   )
{
   return ( hypre_COGMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_COGMRESGetResidual( NALU_HYPRE_Solver solver, void *residual )
{
   /* returns a pointer to the residual vector */
   return hypre_COGMRESGetResidual( (void *) solver, (void **) residual );

}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_COGMRESSetModifyPC
 *--------------------------------------------------------------------------*/


NALU_HYPRE_Int NALU_HYPRE_COGMRESSetModifyPC( NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int (*modify_pc)(NALU_HYPRE_Solver, NALU_HYPRE_Int, NALU_HYPRE_Real) )
{
   return hypre_COGMRESSetModifyPC( (void *) solver, (NALU_HYPRE_Int(*)(void*, NALU_HYPRE_Int,
                                                                   NALU_HYPRE_Real))modify_pc);

}


