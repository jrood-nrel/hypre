/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRTFQmr interface
 *
 *****************************************************************************/

#ifndef __TFQMR__
#define __TFQMR__

#ifdef __cplusplus
extern "C" {
#endif

extern int NALU_HYPRE_ParCSRTFQmrCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver );

extern int NALU_HYPRE_ParCSRTFQmrDestroy( NALU_HYPRE_Solver solver );

extern int NALU_HYPRE_ParCSRTFQmrSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                               NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );

extern int NALU_HYPRE_ParCSRTFQmrSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );

extern int NALU_HYPRE_ParCSRTFQmrSetTol( NALU_HYPRE_Solver solver, double tol );

extern int NALU_HYPRE_ParCSRTFQmrSetMaxIter( NALU_HYPRE_Solver solver, int max_iter );

extern int NALU_HYPRE_ParCSRTFQmrSetStopCrit( NALU_HYPRE_Solver solver, int stop_crit );

extern int NALU_HYPRE_ParCSRTFQmrSetPrecond( NALU_HYPRE_Solver  solver,
          int (*precond)      (NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void               *precond_data );

extern int NALU_HYPRE_ParCSRTFQmrSetLogging( NALU_HYPRE_Solver solver, int logging);

extern int NALU_HYPRE_ParCSRTFQmrGetNumIterations(NALU_HYPRE_Solver solver,
                                                 int *num_iterations);

extern int NALU_HYPRE_ParCSRTFQmrGetFinalRelativeResidualNorm(NALU_HYPRE_Solver solver,
                                                       double *norm );

#ifdef __cplusplus
}
#endif
#endif

