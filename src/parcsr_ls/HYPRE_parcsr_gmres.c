/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   hypre_GMRESFunctions * gmres_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   gmres_functions =
      hypre_GMRESFunctionsCreate(
         hypre_ParKrylovCAlloc, hypre_ParKrylovFree, hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovCreateVectorArray,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );
   *solver = ( (NALU_HYPRE_Solver) hypre_GMRESCreate( gmres_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESDestroy( NALU_HYPRE_Solver solver )
{
   return ( hypre_GMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESSetup( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_ParCSRMatrix A,
                        NALU_HYPRE_ParVector b,
                        NALU_HYPRE_ParVector x      )
{
   return ( NALU_HYPRE_GMRESSetup( solver,
                              (NALU_HYPRE_Matrix) A,
                              (NALU_HYPRE_Vector) b,
                              (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESSolve( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_ParCSRMatrix A,
                        NALU_HYPRE_ParVector b,
                        NALU_HYPRE_ParVector x      )
{
   return ( NALU_HYPRE_GMRESSolve( solver,
                              (NALU_HYPRE_Matrix) A,
                              (NALU_HYPRE_Vector) b,
                              (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetKDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESSetKDim( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int             k_dim    )
{
   return ( NALU_HYPRE_GMRESSetKDim( solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESSetTol( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_Real         tol    )
{
   return ( NALU_HYPRE_GMRESSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESSetAbsoluteTol( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Real         a_tol    )
{
   return ( NALU_HYPRE_GMRESSetAbsoluteTol( solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESSetMinIter( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int          min_iter )
{
   return ( NALU_HYPRE_GMRESSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESSetMaxIter( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int          max_iter )
{
   return ( NALU_HYPRE_GMRESSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetStopCrit - OBSOLETE
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESSetStopCrit( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int          stop_crit )
{
   return ( NALU_HYPRE_GMRESSetStopCrit( solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESSetPrecond( NALU_HYPRE_Solver          solver,
                             NALU_HYPRE_PtrToParSolverFcn  precond,
                             NALU_HYPRE_PtrToParSolverFcn  precond_setup,
                             NALU_HYPRE_Solver          precond_solver )
{
   return ( NALU_HYPRE_GMRESSetPrecond( solver,
                                   (NALU_HYPRE_PtrToSolverFcn) precond,
                                   (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                   precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESGetPrecond( NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Solver *precond_data_ptr )
{
   return ( NALU_HYPRE_GMRESGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESSetLogging( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int logging)
{
   return ( NALU_HYPRE_GMRESSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESSetPrintLevel( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int print_level)
{
   return ( NALU_HYPRE_GMRESSetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESGetNumIterations( NALU_HYPRE_Solver  solver,
                                   NALU_HYPRE_Int    *num_iterations )
{
   return ( NALU_HYPRE_GMRESGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                               NALU_HYPRE_Real   *norm   )
{
   return ( NALU_HYPRE_GMRESGetFinalRelativeResidualNorm( solver, norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRGMRESGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRGMRESGetResidual( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_ParVector *residual   )
{
   return ( NALU_HYPRE_GMRESGetResidual( solver, (void *) residual ) );
}

/*--------------------------------------------------------------------------
 * Setup routine for on-processor triangular solve as preconditioning.
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int NALU_HYPRE_ParCSROnProcTriSetup(NALU_HYPRE_Solver       solver,
                                     NALU_HYPRE_ParCSRMatrix HA,
                                     NALU_HYPRE_ParVector    Hy,
                                     NALU_HYPRE_ParVector    Hx)
{
   hypre_ParCSRMatrix *A = (hypre_ParCSRMatrix *) HA;

   // Check for and get topological ordering of matrix
   if (!hypre_ParCSRMatrixProcOrdering(A))
   {
      hypre_CSRMatrix *A_diag  = hypre_ParCSRMatrixDiag(A);
      NALU_HYPRE_Real *A_diag_data  = hypre_CSRMatrixData(A_diag);
      NALU_HYPRE_Int *A_diag_i      = hypre_CSRMatrixI(A_diag);
      NALU_HYPRE_Int *A_diag_j      = hypre_CSRMatrixJ(A_diag);
      NALU_HYPRE_Int n              = hypre_CSRMatrixNumRows(A_diag);
      NALU_HYPRE_Int *proc_ordering = hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
      hypre_topo_sort(A_diag_i, A_diag_j, A_diag_data, proc_ordering, n);
      hypre_ParCSRMatrixProcOrdering(A) = proc_ordering;
   }

   return 0;
}


/*--------------------------------------------------------------------------
 * Solve routine for on-processor triangular solve as preconditioning.
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int NALU_HYPRE_ParCSROnProcTriSolve(NALU_HYPRE_Solver       solver,
                                     NALU_HYPRE_ParCSRMatrix HA,
                                     NALU_HYPRE_ParVector    Hy,
                                     NALU_HYPRE_ParVector    Hx)
{
   hypre_ParCSRMatrix *A = (hypre_ParCSRMatrix *) HA;
   hypre_ParVector    *y = (hypre_ParVector *) Hy;
   hypre_ParVector    *x = (hypre_ParVector *) Hx;
   NALU_HYPRE_Int ierr = 0;
   ierr = hypre_BoomerAMGRelax(A, y, NULL, 10, 0, 1, 1, NULL, x, NULL, NULL);
   return ierr;
}

