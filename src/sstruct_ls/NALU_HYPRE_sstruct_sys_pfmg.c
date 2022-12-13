/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGCreate( MPI_Comm comm, NALU_HYPRE_SStructSolver *solver )
{
   *solver = ( (NALU_HYPRE_SStructSolver) hypre_SysPFMGCreate( comm ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGDestroy( NALU_HYPRE_SStructSolver solver )
{
   return ( hypre_SysPFMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetup( NALU_HYPRE_SStructSolver  solver,
                           NALU_HYPRE_SStructMatrix A,
                           NALU_HYPRE_SStructVector b,
                           NALU_HYPRE_SStructVector x      )
{
   return ( hypre_SysPFMGSetup( (void *) solver,
                                (hypre_SStructMatrix *) A,
                                (hypre_SStructVector *) b,
                                (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSolve( NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_SStructMatrix A,
                           NALU_HYPRE_SStructVector b,
                           NALU_HYPRE_SStructVector x      )
{
   return ( hypre_SysPFMGSolve( (void *) solver,
                                (hypre_SStructMatrix *) A,
                                (hypre_SStructVector *) b,
                                (hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetTol( NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_Real         tol    )
{
   return ( hypre_SysPFMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetMaxIter( NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Int          max_iter  )
{
   return ( hypre_SysPFMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetRelChange( NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int          rel_change  )
{
   return ( hypre_SysPFMGSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetZeroGuess( NALU_HYPRE_SStructSolver solver )
{
   return ( hypre_SysPFMGSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetNonZeroGuess( NALU_HYPRE_SStructSolver solver )
{
   return ( hypre_SysPFMGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetRelaxType( NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int          relax_type )
{
   return ( hypre_SysPFMGSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetJacobiWeight(NALU_HYPRE_SStructSolver solver,
                                    NALU_HYPRE_Real          weight)
{
   return ( hypre_SysPFMGSetJacobiWeight( (void *) solver, weight) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetNumPreRelax( NALU_HYPRE_SStructSolver solver,
                                    NALU_HYPRE_Int          num_pre_relax )
{
   return ( hypre_SysPFMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetNumPostRelax( NALU_HYPRE_SStructSolver solver,
                                     NALU_HYPRE_Int          num_post_relax )
{
   return ( hypre_SysPFMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetSkipRelax( NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int          skip_relax )
{
   return ( hypre_SysPFMGSetSkipRelax( (void *) solver, skip_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetDxyz( NALU_HYPRE_SStructSolver  solver,
                             NALU_HYPRE_Real         *dxyz   )
{
   return ( hypre_SysPFMGSetDxyz( (void *) solver, dxyz) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetLogging( NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Int          logging )
{
   return ( hypre_SysPFMGSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
*--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetPrintLevel( NALU_HYPRE_SStructSolver solver,
                                   NALU_HYPRE_Int         print_level )
{
   return ( hypre_SysPFMGSetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGGetNumIterations( NALU_HYPRE_SStructSolver  solver,
                                      NALU_HYPRE_Int          *num_iterations )
{
   return ( hypre_SysPFMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm( NALU_HYPRE_SStructSolver  solver,
                                                  NALU_HYPRE_Real         *norm   )
{
   return ( hypre_SysPFMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

