/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRBiCGSTABL interface
 *
 *****************************************************************************/

#ifndef __CGSTABL__
#define __CGSTABL__

#ifdef __cplusplus
extern "C" {
#endif

extern int NALU_HYPRE_ParCSRBiCGSTABLCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver );

extern int NALU_HYPRE_ParCSRBiCGSTABLDestroy( NALU_HYPRE_Solver solver );

extern int NALU_HYPRE_ParCSRBiCGSTABLSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                               NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );

extern int NALU_HYPRE_ParCSRBiCGSTABLSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );

extern int NALU_HYPRE_ParCSRBiCGSTABLSetTol( NALU_HYPRE_Solver solver, double tol );

extern int NALU_HYPRE_ParCSRBiCGSTABLSetSize( NALU_HYPRE_Solver solver, int size );

extern int NALU_HYPRE_ParCSRBiCGSTABLSetMaxIter( NALU_HYPRE_Solver solver, int max_iter );

extern int NALU_HYPRE_ParCSRBiCGSTABLSetStopCrit( NALU_HYPRE_Solver solver, int stop_crit );

extern int NALU_HYPRE_ParCSRBiCGSTABLSetPrecond( NALU_HYPRE_Solver  solver,
          int (*precond)      (NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void               *precond_data );

extern int NALU_HYPRE_ParCSRBiCGSTABLSetLogging( NALU_HYPRE_Solver solver, int logging);

extern int NALU_HYPRE_ParCSRBiCGSTABLGetNumIterations(NALU_HYPRE_Solver solver,
                                                 int *num_iterations);

extern int NALU_HYPRE_ParCSRBiCGSTABLGetFinalRelativeResidualNorm(NALU_HYPRE_Solver solver,
                                                       double *norm );

#ifdef __cplusplus
}
#endif
#endif

