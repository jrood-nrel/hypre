/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* NALU_HYPRE_DistributedMatrixPilutSolver.c */
NALU_HYPRE_Int NALU_HYPRE_NewDistributedMatrixPilutSolver (MPI_Comm comm , NALU_HYPRE_DistributedMatrix matrix, NALU_HYPRE_DistributedMatrixPilutSolver *solver );
NALU_HYPRE_Int NALU_HYPRE_FreeDistributedMatrixPilutSolver (NALU_HYPRE_DistributedMatrixPilutSolver in_ptr );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverInitialize (NALU_HYPRE_DistributedMatrixPilutSolver solver );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetMatrix (NALU_HYPRE_DistributedMatrixPilutSolver in_ptr , NALU_HYPRE_DistributedMatrix matrix );
NALU_HYPRE_DistributedMatrix NALU_HYPRE_DistributedMatrixPilutSolverGetMatrix (NALU_HYPRE_DistributedMatrixPilutSolver in_ptr );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetNumLocalRow (NALU_HYPRE_DistributedMatrixPilutSolver in_ptr , NALU_HYPRE_Int FirstLocalRow );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetFactorRowSize (NALU_HYPRE_DistributedMatrixPilutSolver in_ptr , NALU_HYPRE_Int size );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetDropTolerance (NALU_HYPRE_DistributedMatrixPilutSolver in_ptr , NALU_HYPRE_Real tolerance );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetMaxIts (NALU_HYPRE_DistributedMatrixPilutSolver in_ptr , NALU_HYPRE_Int its );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetup (NALU_HYPRE_DistributedMatrixPilutSolver in_ptr );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSolve (NALU_HYPRE_DistributedMatrixPilutSolver in_ptr , NALU_HYPRE_Real *x , NALU_HYPRE_Real *b );
NALU_HYPRE_Int NALU_HYPRE_DistributedMatrixPilutSolverSetLogging( NALU_HYPRE_DistributedMatrixPilutSolver in_ptr, NALU_HYPRE_Int logging );

