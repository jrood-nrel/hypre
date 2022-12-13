/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCGNRCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCGNRCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   hypre_CGNRFunctions * cgnr_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   cgnr_functions =
      hypre_CGNRFunctionsCreate(
         hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecT,
         hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd,
         hypre_ParKrylovCopyVector, hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup,
         hypre_ParKrylovIdentity, hypre_ParKrylovIdentity );
   *solver = ( (NALU_HYPRE_Solver) hypre_CGNRCreate( cgnr_functions) );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCGNRDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCGNRDestroy( NALU_HYPRE_Solver solver )
{
   return ( hypre_CGNRDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCGNRSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCGNRSetup( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_ParCSRMatrix A,
                       NALU_HYPRE_ParVector b,
                       NALU_HYPRE_ParVector x      )
{
   return ( NALU_HYPRE_CGNRSetup( solver,
                             (NALU_HYPRE_Matrix) A,
                             (NALU_HYPRE_Vector) b,
                             (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCGNRSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCGNRSolve( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_ParCSRMatrix A,
                       NALU_HYPRE_ParVector b,
                       NALU_HYPRE_ParVector x      )
{
   return ( NALU_HYPRE_CGNRSolve( solver,
                             (NALU_HYPRE_Matrix) A,
                             (NALU_HYPRE_Vector) b,
                             (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCGNRSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCGNRSetTol( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Real         tol    )
{
   return ( NALU_HYPRE_CGNRSetTol( solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCGNRSetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCGNRSetMinIter( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int                min_iter )
{
   return ( NALU_HYPRE_CGNRSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCGNRSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCGNRSetMaxIter( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int                max_iter )
{
   return ( NALU_HYPRE_CGNRSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCGNRSetStopCrit
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCGNRSetStopCrit( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int                stop_crit )
{
   return ( NALU_HYPRE_CGNRSetStopCrit( solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCGNRSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCGNRSetPrecond( NALU_HYPRE_Solver         solver,
                            NALU_HYPRE_PtrToParSolverFcn precond,
                            NALU_HYPRE_PtrToParSolverFcn precondT,
                            NALU_HYPRE_PtrToParSolverFcn precond_setup,
                            NALU_HYPRE_Solver         precond_solver )
{
   return ( NALU_HYPRE_CGNRSetPrecond( solver,
                                  (NALU_HYPRE_PtrToSolverFcn) precond,
                                  (NALU_HYPRE_PtrToSolverFcn) precondT,
                                  (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                  precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCGNRGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCGNRGetPrecond( NALU_HYPRE_Solver   solver,
                            NALU_HYPRE_Solver  *precond_data_ptr )
{
   return ( NALU_HYPRE_CGNRGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCGNRSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCGNRSetLogging( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int logging)
{
   return ( NALU_HYPRE_CGNRSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCGNRGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCGNRGetNumIterations( NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Int    *num_iterations )
{
   return ( NALU_HYPRE_CGNRGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                              NALU_HYPRE_Real   *norm   )
{
   return ( NALU_HYPRE_CGNRGetFinalRelativeResidualNorm( solver, norm ) );
}
