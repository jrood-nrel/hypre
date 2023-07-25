/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRLSICG interface
 *
 *****************************************************************************/

#ifndef __LSICG__
#define __LSICG__

#ifdef __cplusplus
extern "C" {
#endif

extern int NALU_HYPRE_ParCSRLSICGCreate(MPI_Comm comm, NALU_HYPRE_Solver *solver);

extern int NALU_HYPRE_ParCSRLSICGDestroy(NALU_HYPRE_Solver solver);

extern int NALU_HYPRE_ParCSRLSICGSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );

extern int NALU_HYPRE_ParCSRLSICGSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );

extern int NALU_HYPRE_ParCSRLSICGSetTol(NALU_HYPRE_Solver solver, double tol);

extern int NALU_HYPRE_ParCSRLSICGSetMaxIter(NALU_HYPRE_Solver solver, int max_iter);

extern int NALU_HYPRE_ParCSRLSICGSetStopCrit(NALU_HYPRE_Solver solver, int stop_crit);

extern int NALU_HYPRE_ParCSRLSICGSetPrecond(NALU_HYPRE_Solver  solver,
          int (*precond)      (NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void *precond_data );

extern int NALU_HYPRE_ParCSRLSICGSetLogging(NALU_HYPRE_Solver solver, int logging);

extern int NALU_HYPRE_ParCSRLSICGGetNumIterations(NALU_HYPRE_Solver solver,
                                             int *num_iterations);

extern int NALU_HYPRE_ParCSRLSICGGetFinalRelativeResidualNorm(NALU_HYPRE_Solver solver,
                                                         double *norm );

#ifdef __cplusplus
}
#endif
#endif

