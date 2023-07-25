/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUCreate( NALU_HYPRE_Solver *solver )
{
   if (!solver)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   *solver = ( (NALU_HYPRE_Solver) nalu_hypre_ILUCreate( ) );
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUDestroy( NALU_HYPRE_Solver solver )
{
   return ( nalu_hypre_ILUDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetup( NALU_HYPRE_Solver solver,
                NALU_HYPRE_ParCSRMatrix A,
                NALU_HYPRE_ParVector b,
                NALU_HYPRE_ParVector x      )
{
   return ( nalu_hypre_ILUSetup( (void *) solver,
                            (nalu_hypre_ParCSRMatrix *) A,
                            (nalu_hypre_ParVector *) b,
                            (nalu_hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSolve( NALU_HYPRE_Solver solver,
                NALU_HYPRE_ParCSRMatrix A,
                NALU_HYPRE_ParVector b,
                NALU_HYPRE_ParVector x      )
{
   return ( nalu_hypre_ILUSolve( (void *) solver,
                            (nalu_hypre_ParCSRMatrix *) A,
                            (nalu_hypre_ParVector *) b,
                            (nalu_hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetPrintLevel( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level )
{
   return nalu_hypre_ILUSetPrintLevel( solver, print_level );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetLogging( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging )
{
   return nalu_hypre_ILUSetLogging(solver, logging );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetMaxIter( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter )
{
   return nalu_hypre_ILUSetMaxIter( solver, max_iter );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetTriSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetTriSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_Int tri_solve )
{
   return nalu_hypre_ILUSetTriSolve( solver, tri_solve );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetLowerJacobiIters
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetLowerJacobiIters( NALU_HYPRE_Solver solver, NALU_HYPRE_Int lower_jacobi_iters )
{
   return nalu_hypre_ILUSetLowerJacobiIters( solver, lower_jacobi_iters );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetUpperJacobiIters
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetUpperJacobiIters( NALU_HYPRE_Solver solver, NALU_HYPRE_Int upper_jacobi_iters )
{
   return nalu_hypre_ILUSetUpperJacobiIters( solver, upper_jacobi_iters );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetTol( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol )
{
   return nalu_hypre_ILUSetTol( solver, tol );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetDropThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetDropThreshold( NALU_HYPRE_Solver solver, NALU_HYPRE_Real threshold )
{
   return nalu_hypre_ILUSetDropThreshold( solver, threshold );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetDropThresholdArray
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetDropThresholdArray( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *threshold )
{
   return nalu_hypre_ILUSetDropThresholdArray( solver, threshold );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetNSHDropThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetNSHDropThreshold( NALU_HYPRE_Solver solver, NALU_HYPRE_Real threshold )
{
   return nalu_hypre_ILUSetSchurNSHDropThreshold( solver, threshold );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetNSHDropThresholdArray
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetNSHDropThresholdArray( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *threshold )
{
   return nalu_hypre_ILUSetSchurNSHDropThresholdArray( solver, threshold );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetSchurMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetSchurMaxIter( NALU_HYPRE_Solver solver, NALU_HYPRE_Int ss_max_iter )
{
   return nalu_hypre_ILUSetSchurSolverMaxIter( solver, ss_max_iter );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetMaxNnzPerRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetMaxNnzPerRow( NALU_HYPRE_Solver solver, NALU_HYPRE_Int nzmax )
{
   return nalu_hypre_ILUSetMaxNnzPerRow( solver, nzmax );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetLevelOfFill
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetLevelOfFill( NALU_HYPRE_Solver solver, NALU_HYPRE_Int lfil )
{
   return nalu_hypre_ILUSetLevelOfFill( solver, lfil );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetType( NALU_HYPRE_Solver solver, NALU_HYPRE_Int ilu_type )
{
   return nalu_hypre_ILUSetType( solver, ilu_type );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetLocalReordering
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetLocalReordering(  NALU_HYPRE_Solver solver, NALU_HYPRE_Int ordering_type )
{
   return nalu_hypre_ILUSetLocalReordering(solver, ordering_type);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUGetNumIterations( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations )
{
   return nalu_hypre_ILUGetNumIterations( solver, num_iterations );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUGetFinalRelativeResidualNorm(  NALU_HYPRE_Solver solver, NALU_HYPRE_Real *res_norm )
{
   return nalu_hypre_ILUGetFinalRelativeResidualNorm(solver, res_norm);
}
