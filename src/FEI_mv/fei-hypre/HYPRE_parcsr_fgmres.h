/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRFGMRES interface
 *
 *****************************************************************************/

#ifndef __FGMRESH__
#define __FGMRESH__

#ifdef __cplusplus
extern "C" {
#endif

int NALU_HYPRE_ParCSRFGMRESCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver );

int NALU_HYPRE_ParCSRFGMRESDestroy( NALU_HYPRE_Solver solver );

int NALU_HYPRE_ParCSRFGMRESSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                   NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );

int NALU_HYPRE_ParCSRFGMRESSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                   NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );

int NALU_HYPRE_ParCSRFGMRESSetKDim(NALU_HYPRE_Solver solver, int kdim);

int NALU_HYPRE_ParCSRFGMRESSetTol(NALU_HYPRE_Solver solver, double tol);

int NALU_HYPRE_ParCSRFGMRESSetMaxIter(NALU_HYPRE_Solver solver, int max_iter);

int NALU_HYPRE_ParCSRFGMRESSetStopCrit(NALU_HYPRE_Solver solver, int stop_crit);

int NALU_HYPRE_ParCSRFGMRESSetPrecond(NALU_HYPRE_Solver  solver,
          int (*precond)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void *precond_data);

int NALU_HYPRE_ParCSRFGMRESSetLogging(NALU_HYPRE_Solver solver, int logging);

int NALU_HYPRE_ParCSRFGMRESGetNumIterations(NALU_HYPRE_Solver solver,
                                              int *num_iterations);

int NALU_HYPRE_ParCSRFGMRESGetFinalRelativeResidualNorm(NALU_HYPRE_Solver solver,
                                                          double *norm );

int NALU_HYPRE_ParCSRFGMRESUpdatePrecondTolerance(NALU_HYPRE_Solver  solver,
                             int (*set_tolerance)(NALU_HYPRE_Solver sol, double));

#ifdef __cplusplus
}
#endif

#endif

