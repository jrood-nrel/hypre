/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructJacobiCreate( MPI_Comm            comm,
                          NALU_HYPRE_StructSolver *solver )
{
   *solver = ( (NALU_HYPRE_StructSolver) nalu_hypre_JacobiCreate( comm ) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructJacobiDestroy( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_JacobiDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructJacobiSetup( NALU_HYPRE_StructSolver solver,
                         NALU_HYPRE_StructMatrix A,
                         NALU_HYPRE_StructVector b,
                         NALU_HYPRE_StructVector x      )
{
   return ( nalu_hypre_JacobiSetup( (void *) solver,
                               (nalu_hypre_StructMatrix *) A,
                               (nalu_hypre_StructVector *) b,
                               (nalu_hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructJacobiSolve( NALU_HYPRE_StructSolver solver,
                         NALU_HYPRE_StructMatrix A,
                         NALU_HYPRE_StructVector b,
                         NALU_HYPRE_StructVector x      )
{
   return ( nalu_hypre_JacobiSolve( (void *) solver,
                               (nalu_hypre_StructMatrix *) A,
                               (nalu_hypre_StructVector *) b,
                               (nalu_hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructJacobiSetTol( NALU_HYPRE_StructSolver solver,
                          NALU_HYPRE_Real         tol    )
{
   return ( nalu_hypre_JacobiSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructJacobiGetTol( NALU_HYPRE_StructSolver solver,
                          NALU_HYPRE_Real       * tol    )
{
   return ( nalu_hypre_JacobiGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructJacobiSetMaxIter( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int          max_iter  )
{
   return ( nalu_hypre_JacobiSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructJacobiGetMaxIter( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int        * max_iter  )
{
   return ( nalu_hypre_JacobiGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructJacobiSetZeroGuess( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_JacobiSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructJacobiGetZeroGuess( NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_Int * zeroguess )
{
   return ( nalu_hypre_JacobiGetZeroGuess( (void *) solver, zeroguess ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructJacobiSetNonZeroGuess( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_JacobiSetZeroGuess( (void *) solver, 0 ) );
}


/* NOT YET IMPLEMENTED */

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructJacobiGetNumIterations( NALU_HYPRE_StructSolver  solver,
                                    NALU_HYPRE_Int          *num_iterations )
{
   return ( nalu_hypre_JacobiGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructJacobiGetFinalRelativeResidualNorm( NALU_HYPRE_StructSolver  solver,
                                                NALU_HYPRE_Real         *norm   )
{
   return ( nalu_hypre_JacobiGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}
