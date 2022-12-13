/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   hypre_COGMRESFunctions * cogmres_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   cogmres_functions =
      hypre_COGMRESFunctionsCreate(
         hypre_ParKrylovCAlloc, hypre_ParKrylovFree, hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovCreateVectorArray,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovMassInnerProd,
         hypre_ParKrylovMassDotpTwo, hypre_ParKrylovCopyVector,
         //hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy, hypre_ParKrylovMassAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );
   *solver = ( (NALU_HYPRE_Solver) hypre_COGMRESCreate( cogmres_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESDestroy( NALU_HYPRE_Solver solver )
{
   return ( hypre_COGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESSetup( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_ParCSRMatrix A,
                          NALU_HYPRE_ParVector b,
                          NALU_HYPRE_ParVector x      )
{
   return ( NALU_HYPRE_COGMRESSetup( solver,
                                (NALU_HYPRE_Matrix) A,
                                (NALU_HYPRE_Vector) b,
                                (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESSolve( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_ParCSRMatrix A,
                          NALU_HYPRE_ParVector b,
                          NALU_HYPRE_ParVector x      )
{
   return ( NALU_HYPRE_COGMRESSolve( solver,
                                (NALU_HYPRE_Matrix) A,
                                (NALU_HYPRE_Vector) b,
                                (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetKDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESSetKDim( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int             k_dim    )
{
   return ( NALU_HYPRE_COGMRESSetKDim( solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetUnroll
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESSetUnroll( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int             unroll    )
{
   return ( NALU_HYPRE_COGMRESSetUnroll( solver, unroll ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetCGS
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESSetCGS( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int             cgs    )
{
   return ( NALU_HYPRE_COGMRESSetCGS( solver, cgs ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESSetTol( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Real         tol    )
{
   return ( NALU_HYPRE_COGMRESSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESSetAbsoluteTol( NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Real         a_tol    )
{
   return ( NALU_HYPRE_COGMRESSetAbsoluteTol( solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESSetMinIter( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int          min_iter )
{
   return ( NALU_HYPRE_COGMRESSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESSetMaxIter( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int          max_iter )
{
   return ( NALU_HYPRE_COGMRESSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESSetPrecond( NALU_HYPRE_Solver          solver,
                               NALU_HYPRE_PtrToParSolverFcn  precond,
                               NALU_HYPRE_PtrToParSolverFcn  precond_setup,
                               NALU_HYPRE_Solver          precond_solver )
{
   return ( NALU_HYPRE_COGMRESSetPrecond( solver,
                                     (NALU_HYPRE_PtrToSolverFcn) precond,
                                     (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                     precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESGetPrecond( NALU_HYPRE_Solver  solver,
                               NALU_HYPRE_Solver *precond_data_ptr )
{
   return ( NALU_HYPRE_COGMRESGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESSetLogging( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int logging)
{
   return ( NALU_HYPRE_COGMRESSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESSetPrintLevel( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int print_level)
{
   return ( NALU_HYPRE_COGMRESSetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESGetNumIterations( NALU_HYPRE_Solver  solver,
                                     NALU_HYPRE_Int    *num_iterations )
{
   return ( NALU_HYPRE_COGMRESGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                                 NALU_HYPRE_Real   *norm   )
{
   return ( NALU_HYPRE_COGMRESGetFinalRelativeResidualNorm( solver, norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESGetResidual( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_ParVector *residual)
{
   return ( NALU_HYPRE_COGMRESGetResidual( solver, (void *) residual ) );
}
