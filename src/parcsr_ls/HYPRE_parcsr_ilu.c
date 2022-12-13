/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUCreate( NALU_HYPRE_Solver *solver )
{
   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   *solver = ( (NALU_HYPRE_Solver) hypre_ILUCreate( ) );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUDestroy( NALU_HYPRE_Solver solver )
{
   return ( hypre_ILUDestroy( (void *) solver ) );
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
   return ( hypre_ILUSetup( (void *) solver,
                            (hypre_ParCSRMatrix *) A,
                            (hypre_ParVector *) b,
                            (hypre_ParVector *) x ) );
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
   return ( hypre_ILUSolve( (void *) solver,
                            (hypre_ParCSRMatrix *) A,
                            (hypre_ParVector *) b,
                            (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetPrintLevel( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level )
{
   return hypre_ILUSetPrintLevel( solver, print_level );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetLogging( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging )
{
   return hypre_ILUSetLogging(solver, logging );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetMaxIter( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter )
{
   return hypre_ILUSetMaxIter( solver, max_iter );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetTriSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetTriSolve( NALU_HYPRE_Solver solver, NALU_HYPRE_Int tri_solve )
{
   return hypre_ILUSetTriSolve( solver, tri_solve );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetLowerJacobiIters
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetLowerJacobiIters( NALU_HYPRE_Solver solver, NALU_HYPRE_Int lower_jacobi_iters )
{
   return hypre_ILUSetLowerJacobiIters( solver, lower_jacobi_iters );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetUpperJacobiIters
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetUpperJacobiIters( NALU_HYPRE_Solver solver, NALU_HYPRE_Int upper_jacobi_iters )
{
   return hypre_ILUSetUpperJacobiIters( solver, upper_jacobi_iters );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetTol( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol )
{
   return hypre_ILUSetTol( solver, tol );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetDropThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetDropThreshold( NALU_HYPRE_Solver solver, NALU_HYPRE_Real threshold )
{
   return hypre_ILUSetDropThreshold( solver, threshold );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetDropThresholdArray
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetDropThresholdArray( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *threshold )
{
   return hypre_ILUSetDropThresholdArray( solver, threshold );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetNSHDropThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetNSHDropThreshold( NALU_HYPRE_Solver solver, NALU_HYPRE_Real threshold )
{
   return hypre_ILUSetSchurNSHDropThreshold( solver, threshold );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetNSHDropThresholdArray
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetNSHDropThresholdArray( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *threshold )
{
   return hypre_ILUSetSchurNSHDropThresholdArray( solver, threshold );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetSchurMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetSchurMaxIter( NALU_HYPRE_Solver solver, NALU_HYPRE_Int ss_max_iter )
{
   return hypre_ILUSetSchurSolverMaxIter( solver, ss_max_iter );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetMaxNnzPerRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetMaxNnzPerRow( NALU_HYPRE_Solver solver, NALU_HYPRE_Int nzmax )
{
   return hypre_ILUSetMaxNnzPerRow( solver, nzmax );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetLevelOfFill
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetLevelOfFill( NALU_HYPRE_Solver solver, NALU_HYPRE_Int lfil )
{
   return hypre_ILUSetLevelOfFill( solver, lfil );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetType( NALU_HYPRE_Solver solver, NALU_HYPRE_Int ilu_type )
{
   return hypre_ILUSetType( solver, ilu_type );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUSetLocalReordering
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUSetLocalReordering(  NALU_HYPRE_Solver solver, NALU_HYPRE_Int ordering_type )
{
   return hypre_ILUSetLocalReordering(solver, ordering_type);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUGetNumIterations( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations )
{
   return hypre_ILUGetNumIterations( solver, num_iterations );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ILUGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ILUGetFinalRelativeResidualNorm(  NALU_HYPRE_Solver solver, NALU_HYPRE_Real *res_norm )
{
   return hypre_ILUGetFinalRelativeResidualNorm(solver, res_norm);
}
