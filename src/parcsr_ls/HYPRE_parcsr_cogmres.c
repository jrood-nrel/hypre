/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   nalu_hypre_COGMRESFunctions * cogmres_functions;

   if (!solver)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   cogmres_functions =
      nalu_hypre_COGMRESFunctionsCreate(
         nalu_hypre_ParKrylovCAlloc, nalu_hypre_ParKrylovFree, nalu_hypre_ParKrylovCommInfo,
         nalu_hypre_ParKrylovCreateVector,
         nalu_hypre_ParKrylovCreateVectorArray,
         nalu_hypre_ParKrylovDestroyVector, nalu_hypre_ParKrylovMatvecCreate,
         nalu_hypre_ParKrylovMatvec, nalu_hypre_ParKrylovMatvecDestroy,
         nalu_hypre_ParKrylovInnerProd, nalu_hypre_ParKrylovMassInnerProd,
         nalu_hypre_ParKrylovMassDotpTwo, nalu_hypre_ParKrylovCopyVector,
         //nalu_hypre_ParKrylovCopyVector,
         nalu_hypre_ParKrylovClearVector,
         nalu_hypre_ParKrylovScaleVector, nalu_hypre_ParKrylovAxpy, nalu_hypre_ParKrylovMassAxpy,
         nalu_hypre_ParKrylovIdentitySetup, nalu_hypre_ParKrylovIdentity );
   *solver = ( (NALU_HYPRE_Solver) nalu_hypre_COGMRESCreate( cogmres_functions ) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRCOGMRESDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRCOGMRESDestroy( NALU_HYPRE_Solver solver )
{
   return ( nalu_hypre_COGMRESDestroy( (void *) solver ) );
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
