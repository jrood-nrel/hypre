/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   nalu_hypre_PCGFunctions * pcg_functions;

   if (!solver)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   pcg_functions =
      nalu_hypre_PCGFunctionsCreate(
         nalu_hypre_ParKrylovCAlloc, nalu_hypre_ParKrylovFree, nalu_hypre_ParKrylovCommInfo,
         nalu_hypre_ParKrylovCreateVector,
         nalu_hypre_ParKrylovDestroyVector, nalu_hypre_ParKrylovMatvecCreate,
         nalu_hypre_ParKrylovMatvec, nalu_hypre_ParKrylovMatvecDestroy,
         nalu_hypre_ParKrylovInnerProd, nalu_hypre_ParKrylovCopyVector,
         nalu_hypre_ParKrylovClearVector,
         nalu_hypre_ParKrylovScaleVector, nalu_hypre_ParKrylovAxpy,
         nalu_hypre_ParKrylovIdentitySetup, nalu_hypre_ParKrylovIdentity );
   *solver = ( (NALU_HYPRE_Solver) nalu_hypre_PCGCreate( pcg_functions ) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGDestroy( NALU_HYPRE_Solver solver )
{
   return ( nalu_hypre_PCGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGSetup( NALU_HYPRE_Solver solver,
                      NALU_HYPRE_ParCSRMatrix A,
                      NALU_HYPRE_ParVector b,
                      NALU_HYPRE_ParVector x      )
{
   return ( NALU_HYPRE_PCGSetup( solver,
                            (NALU_HYPRE_Matrix) A,
                            (NALU_HYPRE_Vector) b,
                            (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGSolve( NALU_HYPRE_Solver solver,
                      NALU_HYPRE_ParCSRMatrix A,
                      NALU_HYPRE_ParVector b,
                      NALU_HYPRE_ParVector x      )
{
   return ( NALU_HYPRE_PCGSolve( solver,
                            (NALU_HYPRE_Matrix) A,
                            (NALU_HYPRE_Vector) b,
                            (NALU_HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGSetTol( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Real   tol    )
{
   return ( NALU_HYPRE_PCGSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGSetAbsoluteTol( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Real   a_tol    )
{
   return ( NALU_HYPRE_PCGSetAbsoluteTol( solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGSetMaxIter( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int    max_iter )
{
   return ( NALU_HYPRE_PCGSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetStopCrit
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGSetStopCrit( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int    stop_crit )
{
   return ( NALU_HYPRE_PCGSetStopCrit( solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGSetTwoNorm( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int    two_norm )
{
   return ( NALU_HYPRE_PCGSetTwoNorm( solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetRelChange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGSetRelChange( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int    rel_change )
{
   return ( NALU_HYPRE_PCGSetRelChange( solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGSetPrecond( NALU_HYPRE_Solver         solver,
                           NALU_HYPRE_PtrToParSolverFcn precond,
                           NALU_HYPRE_PtrToParSolverFcn precond_setup,
                           NALU_HYPRE_Solver         precond_solver )
{
   return ( NALU_HYPRE_PCGSetPrecond( solver,
                                 (NALU_HYPRE_PtrToSolverFcn) precond,
                                 (NALU_HYPRE_PtrToSolverFcn) precond_setup,
                                 precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGGetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGGetPrecond( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Solver *precond_data_ptr )
{
   return ( NALU_HYPRE_PCGGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetPrintLevel
 * an obsolete function; use NALU_HYPRE_PCG* functions instead
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGSetPrintLevel( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int level )
{
   return ( NALU_HYPRE_PCGSetPrintLevel( solver, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGSetLogging
 * an obsolete function; use NALU_HYPRE_PCG* functions instead
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGSetLogging( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int level )
{
   return ( NALU_HYPRE_PCGSetLogging( solver, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGGetNumIterations( NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Int    *num_iterations )
{
   return ( NALU_HYPRE_PCGGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                             NALU_HYPRE_Real   *norm   )
{
   return ( NALU_HYPRE_PCGGetFinalRelativeResidualNorm( solver, norm ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGGetResidual( NALU_HYPRE_Solver  solver,
                            NALU_HYPRE_ParVector *residual   )
{
   return ( NALU_HYPRE_PCGGetResidual( solver, (void *) residual ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRDiagScaleSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRDiagScaleSetup( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_ParVector y,
                            NALU_HYPRE_ParVector x      )
{
   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRDiagScale
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRDiagScale( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_ParCSRMatrix HA,
                       NALU_HYPRE_ParVector Hy,
                       NALU_HYPRE_ParVector Hx      )
{
   return nalu_hypre_ParCSRDiagScaleVector((nalu_hypre_ParCSRMatrix *) HA,
                                      (nalu_hypre_ParVector *)    Hy,
                                      (nalu_hypre_ParVector *)    Hx);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRSymPrecondSetup
 *--------------------------------------------------------------------------*/

/*

NALU_HYPRE_Int
NALU_HYPRE_ParCSRSymPrecondSetup( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_ParCSRMatrix A,
                             NALU_HYPRE_ParVector b,
                             NALU_HYPRE_ParVector x      )
{
   nalu_hypre_ParCSRMatrix *A = (nalu_hypre_ParCSRMatrix *) A;
   nalu_hypre_ParVector    *y = (nalu_hypre_ParVector *) b;
   nalu_hypre_ParVector    *x = (nalu_hypre_ParVector *) x;

   NALU_HYPRE_Real *x_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(x));
   NALU_HYPRE_Real *y_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(y));
   NALU_HYPRE_Real *A_diag = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(A));
   NALU_HYPRE_Real *A_offd = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffD(A));

   NALU_HYPRE_Int i, ierr = 0;
   nalu_hypre_ParCSRMatrix *Asym;
   MPI_Comm comm;
   NALU_HYPRE_Int global_num_rows;
   NALU_HYPRE_Int global_num_cols;
   NALU_HYPRE_Int *row_starts;
   NALU_HYPRE_Int *col_starts;
   NALU_HYPRE_Int num_cols_offd;
   NALU_HYPRE_Int num_nonzeros_diag;
   NALU_HYPRE_Int num_nonzeros_offd;

   Asym = nalu_hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                   row_starts, col_starts, num_cols_offd,
                                   num_nonzeros_diag, num_nonzeros_offd);

   for (i=0; i < nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(x)); i++)
   {
      x_data[i] = y_data[i]/A_data[A_i[i]];
   }

   return ierr;
} */
