/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRBiCGS interface
 *
 *****************************************************************************/

#ifndef __BICGS__
#define __BICGS__

#ifdef __cplusplus
extern "C" {
#endif

extern int NALU_HYPRE_ParCSRBiCGSCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver );

extern int NALU_HYPRE_ParCSRBiCGSDestroy( NALU_HYPRE_Solver solver );

extern int NALU_HYPRE_ParCSRBiCGSSetup(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );

extern int NALU_HYPRE_ParCSRBiCGSSolve(NALU_HYPRE_Solver solver,NALU_HYPRE_ParCSRMatrix A,
                                  NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x );

extern int NALU_HYPRE_ParCSRBiCGSSetTol( NALU_HYPRE_Solver solver, double tol );

extern int NALU_HYPRE_ParCSRBiCGSSetMaxIter( NALU_HYPRE_Solver solver, int max_iter );

extern int NALU_HYPRE_ParCSRBiCGSSetStopCrit( NALU_HYPRE_Solver solver, int stop_crit );

extern int NALU_HYPRE_ParCSRBiCGSSetPrecond( NALU_HYPRE_Solver  solver,
          int (*precond)      (NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          int (*precond_setup)(NALU_HYPRE_Solver sol, NALU_HYPRE_ParCSRMatrix matrix,
			       NALU_HYPRE_ParVector b, NALU_HYPRE_ParVector x),
          void                *precond_data );

extern int NALU_HYPRE_ParCSRBiCGSSetLogging( NALU_HYPRE_Solver solver, int logging);

extern int NALU_HYPRE_ParCSRBiCGSGetNumIterations(NALU_HYPRE_Solver solver,
                                             int *num_iterations);

extern int NALU_HYPRE_ParCSRBiCGSGetFinalRelativeResidualNorm(NALU_HYPRE_Solver solver,
                                                         double *norm );

#ifdef __cplusplus
}
#endif
#endif

