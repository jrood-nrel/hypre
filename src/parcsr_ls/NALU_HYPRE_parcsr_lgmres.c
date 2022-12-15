/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   nalu_hypre_LGMRESFunctions * lgmres_functions;

   if (!solver)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   lgmres_functions =
      nalu_hypre_LGMRESFunctionsCreate(
         nalu_hypre_ParKrylovCAlloc, nalu_hypre_ParKrylovFree, nalu_hypre_ParKrylovCommInfo,
         nalu_hypre_ParKrylovCreateVector,
         nalu_hypre_ParKrylovCreateVectorArray,
         nalu_hypre_ParKrylovDestroyVector, nalu_hypre_ParKrylovMatvecCreate,
         nalu_hypre_ParKrylovMatvec, nalu_hypre_ParKrylovMatvecDestroy,
         nalu_hypre_ParKrylovInnerProd, nalu_hypre_ParKrylovCopyVector,
         nalu_hypre_ParKrylovClearVector,
         nalu_hypre_ParKrylovScaleVector, nalu_hypre_ParKrylovAxpy,
         nalu_hypre_ParKrylovIdentitySetup, nalu_hypre_ParKrylovIdentity );
   *solver = ( (NALU_HYPRE_Solver) nalu_hypre_LGMRESCreate( lgmres_functions ) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESDestroy( NALU_HYPRE_Solver solver )
{
   return ( nalu_hypre_LGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESSetup( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_ParCSRMatrix A,
                         NALU_HYPRE_ParVector b,
                         NALU_HYPRE_ParVector x      )
{
   return ( NALU_HYPRE_LGMRESSetup( solver,
                               (NALU_HYPRE_Matrix) A,
                               (NALU_HYPRE_Vector) b,
                               (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESSolve( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_ParCSRMatrix A,
                         NALU_HYPRE_ParVector b,
                         NALU_HYPRE_ParVector x      )
{
   return ( NALU_HYPRE_LGMRESSolve( solver,
                               (NALU_HYPRE_Matrix) A,
                               (NALU_HYPRE_Vector) b,
                               (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESSetKDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESSetKDim( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int    k_dim    )
{
   return ( NALU_HYPRE_LGMRESSetKDim( solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESSetAugDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESSetAugDim( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int    aug_dim    )
{
   return ( NALU_HYPRE_LGMRESSetAugDim( solver, aug_dim ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESSetTol( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Real   tol    )
{
   return ( NALU_HYPRE_LGMRESSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESSetAbsoluteTol( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Real   a_tol    )
{
   return ( NALU_HYPRE_LGMRESSetAbsoluteTol( solver, a_tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESSetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESSetMinIter( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    min_iter )
{
   return ( NALU_HYPRE_LGMRESSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESSetMaxIter( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    max_iter )
{
   return ( NALU_HYPRE_LGMRESSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESSetPrecond( NALU_HYPRE_Solver          solver,
                              NALU_HYPRE_PtrToParSolverFcn  precond,
                              NALU_HYPRE_PtrToParSolverFcn  precond_setup,
                              NALU_HYPRE_Solver          precond_solver )
{
   return ( NALU_HYPRE_LGMRESSetPrecond( solver,
                                    (NALU_HYPRE_PtrToSolverFcn) precond,
                                    (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                    precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESGetPrecond( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Solver *precond_data_ptr )
{
   return ( NALU_HYPRE_LGMRESGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESSetLogging( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int logging)
{
   return ( NALU_HYPRE_LGMRESSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESSetPrintLevel( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int print_level)
{
   return ( NALU_HYPRE_LGMRESSetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESGetNumIterations( NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int    *num_iterations )
{
   return ( NALU_HYPRE_LGMRESGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                                NALU_HYPRE_Real   *norm   )
{
   return ( NALU_HYPRE_LGMRESGetFinalRelativeResidualNorm( solver, norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRLGMRESGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRLGMRESGetResidual( NALU_HYPRE_Solver  solver,
                               NALU_HYPRE_ParVector *residual)
{
   return ( NALU_HYPRE_LGMRESGetResidual( solver, (void *) residual ) );
}
