/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRPilut interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "./NALU_HYPRE_parcsr_ls.h"

#include "../distributed_matrix/NALU_HYPRE_distributed_matrix_types.h"
#include "../distributed_matrix/NALU_HYPRE_distributed_matrix_protos.h"

#include "../matrix_matrix/NALU_HYPRE_matrix_matrix_protos.h"

#include "../distributed_ls/pilut/NALU_HYPRE_DistributedMatrixPilutSolver_types.h"
#include "../distributed_ls/pilut/NALU_HYPRE_DistributedMatrixPilutSolver_protos.h"

/* Must include implementation definition for ParVector since no data access
  functions are publically provided. AJC, 5/99 */
/* Likewise for Vector. AJC, 5/99 */
#include "../seq_mv/vector.h"

/* AB 8/06 - replace header file */
/* #include "../parcsr_mv/par_vector.h" */
#include "../parcsr_mv/_nalu_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPilutCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPilutCreate( MPI_Comm comm, NALU_HYPRE_Solver *solver )
{
#ifdef NALU_HYPRE_MIXEDINT
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return nalu_hypre_error_flag;
#else

   NALU_HYPRE_NewDistributedMatrixPilutSolver( comm, NULL,
                                          (NALU_HYPRE_DistributedMatrixPilutSolver *) solver);

   NALU_HYPRE_DistributedMatrixPilutSolverInitialize(
      (NALU_HYPRE_DistributedMatrixPilutSolver) solver );

   return nalu_hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPilutDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPilutDestroy( NALU_HYPRE_Solver solver )
{
#ifdef NALU_HYPRE_MIXEDINT
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return nalu_hypre_error_flag;
#else

   NALU_HYPRE_DistributedMatrix mat = NALU_HYPRE_DistributedMatrixPilutSolverGetMatrix(
                                    (NALU_HYPRE_DistributedMatrixPilutSolver) solver );
   if ( mat ) { NALU_HYPRE_DistributedMatrixDestroy( mat ); }

   NALU_HYPRE_FreeDistributedMatrixPilutSolver(
      (NALU_HYPRE_DistributedMatrixPilutSolver) solver );

   return nalu_hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPilutSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPilutSetup( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_ParCSRMatrix A,
                        NALU_HYPRE_ParVector b,
                        NALU_HYPRE_ParVector x      )
{
#ifdef NALU_HYPRE_MIXEDINT
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return nalu_hypre_error_flag;
#else

   NALU_HYPRE_DistributedMatrix matrix;
   NALU_HYPRE_DistributedMatrixPilutSolver distributed_solver =
      (NALU_HYPRE_DistributedMatrixPilutSolver) solver;

   NALU_HYPRE_ConvertParCSRMatrixToDistributedMatrix(
      A, &matrix );

   NALU_HYPRE_DistributedMatrixPilutSolverSetMatrix( distributed_solver, matrix );

   NALU_HYPRE_DistributedMatrixPilutSolverSetup( distributed_solver );

   return nalu_hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPilutSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPilutSolve( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_ParCSRMatrix A,
                        NALU_HYPRE_ParVector b,
                        NALU_HYPRE_ParVector x      )
{
#ifdef NALU_HYPRE_MIXEDINT
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return nalu_hypre_error_flag;
#else

   NALU_HYPRE_Real *rhs, *soln;

   rhs = nalu_hypre_VectorData( nalu_hypre_ParVectorLocalVector( (nalu_hypre_ParVector *)b ) );
   soln = nalu_hypre_VectorData( nalu_hypre_ParVectorLocalVector( (nalu_hypre_ParVector *)x ) );

   NALU_HYPRE_DistributedMatrixPilutSolverSolve(
      (NALU_HYPRE_DistributedMatrixPilutSolver) solver,
      soln, rhs );

   return nalu_hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPilutSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPilutSetMaxIter( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int          max_iter  )
{
#ifdef NALU_HYPRE_MIXEDINT
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return nalu_hypre_error_flag;
#else


   NALU_HYPRE_DistributedMatrixPilutSolverSetMaxIts(
      (NALU_HYPRE_DistributedMatrixPilutSolver) solver, max_iter );

   return nalu_hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPilutSetDropTolerance
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPilutSetDropTolerance( NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Real   tol    )
{
#ifdef NALU_HYPRE_MIXEDINT
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return nalu_hypre_error_flag;
#else

   NALU_HYPRE_DistributedMatrixPilutSolverSetDropTolerance(
      (NALU_HYPRE_DistributedMatrixPilutSolver) solver, tol );

   return nalu_hypre_error_flag;
#endif
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRPilutSetFactorRowSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPilutSetFactorRowSize( NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int       size    )
{
#ifdef NALU_HYPRE_MIXEDINT
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return nalu_hypre_error_flag;
#else

   NALU_HYPRE_DistributedMatrixPilutSolverSetFactorRowSize(
      (NALU_HYPRE_DistributedMatrixPilutSolver) solver, size );

   return nalu_hypre_error_flag;
#endif
}

NALU_HYPRE_Int
NALU_HYPRE_ParCSRPilutSetLogging( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int    logging    )
{
#ifdef NALU_HYPRE_MIXEDINT
   nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Pilut cannot be used in mixedint mode!");
   return nalu_hypre_error_flag;
#else

   NALU_HYPRE_DistributedMatrixPilutSolverSetLogging(
      (NALU_HYPRE_DistributedMatrixPilutSolver) solver, logging );

   return nalu_hypre_error_flag;
#endif
}

