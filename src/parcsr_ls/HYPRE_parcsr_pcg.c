/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
   hypre_PCGFunctions * pcg_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_ParKrylovCAlloc, hypre_ParKrylovFree, hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );
   *solver = ( (NALU_HYPRE_Solver) hypre_PCGCreate( pcg_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPCGDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPCGDestroy( NALU_HYPRE_Solver solver )
{
   return ( hypre_PCGDestroy( (void *) solver ) );
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
   return hypre_ParCSRDiagScaleVector((hypre_ParCSRMatrix *) HA,
                                      (hypre_ParVector *)    Hy,
                                      (hypre_ParVector *)    Hx);
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
   hypre_ParCSRMatrix *A = (hypre_ParCSRMatrix *) A;
   hypre_ParVector    *y = (hypre_ParVector *) b;
   hypre_ParVector    *x = (hypre_ParVector *) x;

   NALU_HYPRE_Real *x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   NALU_HYPRE_Real *y_data = hypre_VectorData(hypre_ParVectorLocalVector(y));
   NALU_HYPRE_Real *A_diag = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A));
   NALU_HYPRE_Real *A_offd = hypre_CSRMatrixData(hypre_ParCSRMatrixOffD(A));

   NALU_HYPRE_Int i, ierr = 0;
   hypre_ParCSRMatrix *Asym;
   MPI_Comm comm;
   NALU_HYPRE_Int global_num_rows;
   NALU_HYPRE_Int global_num_cols;
   NALU_HYPRE_Int *row_starts;
   NALU_HYPRE_Int *col_starts;
   NALU_HYPRE_Int num_cols_offd;
   NALU_HYPRE_Int num_nonzeros_diag;
   NALU_HYPRE_Int num_nonzeros_offd;

   Asym = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                   row_starts, col_starts, num_cols_offd,
                                   num_nonzeros_diag, num_nonzeros_offd);

   for (i=0; i < hypre_VectorSize(hypre_ParVectorLocalVector(x)); i++)
   {
      x_data[i] = y_data[i]/A_data[A_i[i]];
   }

   return ierr;
} */
