/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_StructSparseMSG interface
 *
 *****************************************************************************/

#include "_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGCreate( MPI_Comm comm, NALU_HYPRE_StructSolver *solver )
{
   *solver = ( (NALU_HYPRE_StructSolver) hypre_SparseMSGCreate( comm ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGDestroy( NALU_HYPRE_StructSolver solver )
{
   return ( hypre_SparseMSGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSetup( NALU_HYPRE_StructSolver solver,
                            NALU_HYPRE_StructMatrix A,
                            NALU_HYPRE_StructVector b,
                            NALU_HYPRE_StructVector x      )
{
   return ( hypre_SparseMSGSetup( (void *) solver,
                                  (hypre_StructMatrix *) A,
                                  (hypre_StructVector *) b,
                                  (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSolve( NALU_HYPRE_StructSolver solver,
                            NALU_HYPRE_StructMatrix A,
                            NALU_HYPRE_StructVector b,
                            NALU_HYPRE_StructVector x      )
{
   return ( hypre_SparseMSGSolve( (void *) solver,
                                  (hypre_StructMatrix *) A,
                                  (hypre_StructVector *) b,
                                  (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSetTol( NALU_HYPRE_StructSolver solver,
                             NALU_HYPRE_Real         tol    )
{
   return ( hypre_SparseMSGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSetMaxIter( NALU_HYPRE_StructSolver solver,
                                 NALU_HYPRE_Int          max_iter  )
{
   return ( hypre_SparseMSGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetJump
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSetJump( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int              jump )
{
   return ( hypre_SparseMSGSetJump( (void *) solver, jump ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetRelChange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSetRelChange( NALU_HYPRE_StructSolver solver,
                                   NALU_HYPRE_Int          rel_change  )
{
   return ( hypre_SparseMSGSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetZeroGuess
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSetZeroGuess( NALU_HYPRE_StructSolver solver )
{
   return ( hypre_SparseMSGSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSetNonZeroGuess( NALU_HYPRE_StructSolver solver )
{
   return ( hypre_SparseMSGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSetRelaxType( NALU_HYPRE_StructSolver solver,
                                   NALU_HYPRE_Int          relax_type )
{
   return ( hypre_SparseMSGSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetJacobiWeight
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSetJacobiWeight(NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_Real         weight)
{
   return ( hypre_SparseMSGSetJacobiWeight( (void *) solver, weight) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetNumPreRelax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSetNumPreRelax( NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_Int          num_pre_relax )
{
   return ( hypre_SparseMSGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetNumPostRelax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSetNumPostRelax( NALU_HYPRE_StructSolver solver,
                                      NALU_HYPRE_Int          num_post_relax )
{
   return ( hypre_SparseMSGSetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetNumFineRelax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSetNumFineRelax( NALU_HYPRE_StructSolver solver,
                                      NALU_HYPRE_Int          num_fine_relax )
{
   return ( hypre_SparseMSGSetNumFineRelax( (void *) solver, num_fine_relax) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSetLogging( NALU_HYPRE_StructSolver solver,
                                 NALU_HYPRE_Int          logging )
{
   return ( hypre_SparseMSGSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGSetPrintLevel( NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Int        print_level )
{
   return ( hypre_SparseMSGSetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGGetNumIterations( NALU_HYPRE_StructSolver  solver,
                                       NALU_HYPRE_Int          *num_iterations )
{
   return ( hypre_SparseMSGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_StructSparseMSGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSparseMSGGetFinalRelativeResidualNorm( NALU_HYPRE_StructSolver  solver,
                                                   NALU_HYPRE_Real         *norm   )
{
   return ( hypre_SparseMSGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

