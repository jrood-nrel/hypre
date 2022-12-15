/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_FlexGMRES interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESDestroy
 *--------------------------------------------------------------------------*/
/* to do, not trivial */
/*
NALU_HYPRE_Int
NALU_HYPRE_ParCSRFlexGMRESDestroy( NALU_HYPRE_Solver solver )
{
   return( nalu_hypre_FlexGMRESDestroy( (void *) solver ) );
}
*/

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESSetup( NALU_HYPRE_Solver solver,
                      NALU_HYPRE_Matrix A,
                      NALU_HYPRE_Vector b,
                      NALU_HYPRE_Vector x      )
{
   return ( nalu_hypre_FlexGMRESSetup( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESSolve( NALU_HYPRE_Solver solver,
                      NALU_HYPRE_Matrix A,
                      NALU_HYPRE_Vector b,
                      NALU_HYPRE_Vector x      )
{
   return ( nalu_hypre_FlexGMRESSolve( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESSetKDim, NALU_HYPRE_FlexGMRESGetKDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESSetKDim( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int             k_dim    )
{
   return ( nalu_hypre_FlexGMRESSetKDim( (void *) solver, k_dim ) );
}

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESGetKDim( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int           * k_dim    )
{
   return ( nalu_hypre_FlexGMRESGetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESSetTol, NALU_HYPRE_FlexGMRESGetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESSetTol( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Real         tol    )
{
   return ( nalu_hypre_FlexGMRESSetTol( (void *) solver, tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESGetTol( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Real       * tol    )
{
   return ( nalu_hypre_FlexGMRESGetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESSetAbsoluteTol, NALU_HYPRE_FlexGMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESSetAbsoluteTol( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Real         a_tol    )
{
   return ( nalu_hypre_FlexGMRESSetAbsoluteTol( (void *) solver, a_tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESGetAbsoluteTol( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Real       * a_tol    )
{
   return ( nalu_hypre_FlexGMRESGetAbsoluteTol( (void *) solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESSetConvergenceFactorTol, NALU_HYPRE_FlexGMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESSetConvergenceFactorTol( NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Real         cf_tol    )
{
   return ( nalu_hypre_FlexGMRESSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESGetConvergenceFactorTol( NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Real       * cf_tol    )
{
   return ( nalu_hypre_FlexGMRESGetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESSetMinIter, NALU_HYPRE_FlexGMRESGetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESSetMinIter( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int          min_iter )
{
   return ( nalu_hypre_FlexGMRESSetMinIter( (void *) solver, min_iter ) );
}

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESGetMinIter( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int        * min_iter )
{
   return ( nalu_hypre_FlexGMRESGetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESSetMaxIter, NALU_HYPRE_FlexGMRESGetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESSetMaxIter( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int          max_iter )
{
   return ( nalu_hypre_FlexGMRESSetMaxIter( (void *) solver, max_iter ) );
}

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESGetMaxIter( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int        * max_iter )
{
   return ( nalu_hypre_FlexGMRESGetMaxIter( (void *) solver, max_iter ) );
}



/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESSetPrecond( NALU_HYPRE_Solver          solver,
                           NALU_HYPRE_PtrToSolverFcn  precond,
                           NALU_HYPRE_PtrToSolverFcn  precond_setup,
                           NALU_HYPRE_Solver          precond_solver )
{
   return ( nalu_hypre_FlexGMRESSetPrecond( (void *) solver,
                                       (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond,
                                       (NALU_HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
                                       (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESGetPrecond( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Solver *precond_data_ptr )
{
   return ( nalu_hypre_FlexGMRESGetPrecond( (void *)     solver,
                                       (NALU_HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESSetPrintLevel, NALU_HYPRE_FlexGMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESSetPrintLevel( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int          level )
{
   return ( nalu_hypre_FlexGMRESSetPrintLevel( (void *) solver, level ) );
}

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESGetPrintLevel( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int        * level )
{
   return ( nalu_hypre_FlexGMRESGetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESSetLogging, NALU_HYPRE_FlexGMRESGetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESSetLogging( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int          level )
{
   return ( nalu_hypre_FlexGMRESSetLogging( (void *) solver, level ) );
}

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESGetLogging( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int        * level )
{
   return ( nalu_hypre_FlexGMRESGetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESGetNumIterations( NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Int                *num_iterations )
{
   return ( nalu_hypre_FlexGMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESGetConverged
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESGetConverged( NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Int                *converged )
{
   return ( nalu_hypre_FlexGMRESGetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_FlexGMRESGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                             NALU_HYPRE_Real         *norm   )
{
   return ( nalu_hypre_FlexGMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int NALU_HYPRE_FlexGMRESGetResidual( NALU_HYPRE_Solver solver, void *residual )
{
   /* returns a pointer to the residual vector */
   return nalu_hypre_FlexGMRESGetResidual( (void *) solver, (void **) residual );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_FlexGMRESSetModifyPC
 *--------------------------------------------------------------------------*/


NALU_HYPRE_Int NALU_HYPRE_FlexGMRESSetModifyPC( NALU_HYPRE_Solver  solver,
                                      NALU_HYPRE_Int (*modify_pc)(NALU_HYPRE_Solver, NALU_HYPRE_Int, NALU_HYPRE_Real) )

{
   return nalu_hypre_FlexGMRESSetModifyPC( (void *) solver, (NALU_HYPRE_Int(*)(void*, NALU_HYPRE_Int,
                                                                     NALU_HYPRE_Real))modify_pc);

}




