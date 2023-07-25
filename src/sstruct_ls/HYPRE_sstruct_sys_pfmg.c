/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGCreate( MPI_Comm comm, NALU_HYPRE_SStructSolver *solver )
{
   *solver = ( (NALU_HYPRE_SStructSolver) nalu_hypre_SysPFMGCreate( comm ) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGDestroy( NALU_HYPRE_SStructSolver solver )
{
   return ( nalu_hypre_SysPFMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetup( NALU_HYPRE_SStructSolver  solver,
                           NALU_HYPRE_SStructMatrix A,
                           NALU_HYPRE_SStructVector b,
                           NALU_HYPRE_SStructVector x      )
{
   return ( nalu_hypre_SysPFMGSetup( (void *) solver,
                                (nalu_hypre_SStructMatrix *) A,
                                (nalu_hypre_SStructVector *) b,
                                (nalu_hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSolve( NALU_HYPRE_SStructSolver solver,
                           NALU_HYPRE_SStructMatrix A,
                           NALU_HYPRE_SStructVector b,
                           NALU_HYPRE_SStructVector x      )
{
   return ( nalu_hypre_SysPFMGSolve( (void *) solver,
                                (nalu_hypre_SStructMatrix *) A,
                                (nalu_hypre_SStructVector *) b,
                                (nalu_hypre_SStructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetTol( NALU_HYPRE_SStructSolver solver,
                            NALU_HYPRE_Real         tol    )
{
   return ( nalu_hypre_SysPFMGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetMaxIter( NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Int          max_iter  )
{
   return ( nalu_hypre_SysPFMGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetRelChange( NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int          rel_change  )
{
   return ( nalu_hypre_SysPFMGSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetZeroGuess( NALU_HYPRE_SStructSolver solver )
{
   return ( nalu_hypre_SysPFMGSetZeroGuess( (void *) solver, 1 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetNonZeroGuess( NALU_HYPRE_SStructSolver solver )
{
   return ( nalu_hypre_SysPFMGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetRelaxType( NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int          relax_type )
{
   return ( nalu_hypre_SysPFMGSetRelaxType( (void *) solver, relax_type) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetJacobiWeight(NALU_HYPRE_SStructSolver solver,
                                    NALU_HYPRE_Real          weight)
{
   return ( nalu_hypre_SysPFMGSetJacobiWeight( (void *) solver, weight) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetNumPreRelax( NALU_HYPRE_SStructSolver solver,
                                    NALU_HYPRE_Int          num_pre_relax )
{
   return ( nalu_hypre_SysPFMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetNumPostRelax( NALU_HYPRE_SStructSolver solver,
                                     NALU_HYPRE_Int          num_post_relax )
{
   return ( nalu_hypre_SysPFMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetSkipRelax( NALU_HYPRE_SStructSolver solver,
                                  NALU_HYPRE_Int          skip_relax )
{
   return ( nalu_hypre_SysPFMGSetSkipRelax( (void *) solver, skip_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetDxyz( NALU_HYPRE_SStructSolver  solver,
                             NALU_HYPRE_Real         *dxyz   )
{
   return ( nalu_hypre_SysPFMGSetDxyz( (void *) solver, dxyz) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetLogging( NALU_HYPRE_SStructSolver solver,
                                NALU_HYPRE_Int          logging )
{
   return ( nalu_hypre_SysPFMGSetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
*--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGSetPrintLevel( NALU_HYPRE_SStructSolver solver,
                                   NALU_HYPRE_Int         print_level )
{
   return ( nalu_hypre_SysPFMGSetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGGetNumIterations( NALU_HYPRE_SStructSolver  solver,
                                      NALU_HYPRE_Int          *num_iterations )
{
   return ( nalu_hypre_SysPFMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm( NALU_HYPRE_SStructSolver  solver,
                                                  NALU_HYPRE_Real         *norm   )
{
   return ( nalu_hypre_SysPFMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

