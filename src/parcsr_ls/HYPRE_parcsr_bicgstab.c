/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   hypre_BiCGSTABFunctions * bicgstab_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   bicgstab_functions =
      hypre_BiCGSTABFunctionsCreate(
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovCommInfo,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );
   *solver = ( (NALU_HYPRE_Solver) hypre_BiCGSTABCreate( bicgstab_functions) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABDestroy( NALU_HYPRE_Solver solver )
{
   return ( hypre_BiCGSTABDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABSetup( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_ParCSRMatrix A,
                           NALU_HYPRE_ParVector b,
                           NALU_HYPRE_ParVector x      )
{
   return ( NALU_HYPRE_BiCGSTABSetup( solver,
                                 (NALU_HYPRE_Matrix) A,
                                 (NALU_HYPRE_Vector) b,
                                 (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABSolve( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_ParCSRMatrix A,
                           NALU_HYPRE_ParVector b,
                           NALU_HYPRE_ParVector x      )
{
   return ( NALU_HYPRE_BiCGSTABSolve( solver,
                                 (NALU_HYPRE_Matrix) A,
                                 (NALU_HYPRE_Vector) b,
                                 (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABSetTol( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Real         tol    )
{
   return ( NALU_HYPRE_BiCGSTABSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABSetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABSetAbsoluteTol( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Real         a_tol    )
{
   return ( NALU_HYPRE_BiCGSTABSetAbsoluteTol( solver, a_tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABSetMinIter( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int          min_iter )
{
   return ( NALU_HYPRE_BiCGSTABSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABSetMaxIter( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int          max_iter )
{
   return ( NALU_HYPRE_BiCGSTABSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABSetStopCrit( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int          stop_crit )
{
   return ( NALU_HYPRE_BiCGSTABSetStopCrit( solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABSetPrecond( NALU_HYPRE_Solver         solver,
                                NALU_HYPRE_PtrToParSolverFcn precond,
                                NALU_HYPRE_PtrToParSolverFcn precond_setup,
                                NALU_HYPRE_Solver         precond_solver )
{
   return ( NALU_HYPRE_BiCGSTABSetPrecond( solver,
                                      (NALU_HYPRE_PtrToSolverFcn) precond,
                                      (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                      precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABGetPrecond( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Solver *precond_data_ptr )
{
   return ( NALU_HYPRE_BiCGSTABGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABSetLogging( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int logging)
{
   return ( NALU_HYPRE_BiCGSTABSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABSetPrintLevel( NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int print_level)
{
   return ( NALU_HYPRE_BiCGSTABSetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABGetNumIterations( NALU_HYPRE_Solver  solver,
                                      NALU_HYPRE_Int                *num_iterations )
{
   return ( NALU_HYPRE_BiCGSTABGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                                  NALU_HYPRE_Real         *norm   )
{
   return ( NALU_HYPRE_BiCGSTABGetFinalRelativeResidualNorm( solver, norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRBiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRBiCGSTABGetResidual( NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_ParVector *residual)
{
   return ( NALU_HYPRE_BiCGSTABGetResidual( solver, (void *) residual ) );
}
