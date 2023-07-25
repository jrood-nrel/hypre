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
NALU_HYPRE_StructSMGCreate( MPI_Comm comm, NALU_HYPRE_StructSolver *solver )
{
   *solver = ( (NALU_HYPRE_StructSolver) nalu_hypre_SMGCreate( comm ) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGDestroy( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_SMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGSetup( NALU_HYPRE_StructSolver solver,
                      NALU_HYPRE_StructMatrix A,
                      NALU_HYPRE_StructVector b,
                      NALU_HYPRE_StructVector x      )
{
   return ( nalu_hypre_SMGSetup( (void *) solver,
                            (nalu_hypre_StructMatrix *) A,
                            (nalu_hypre_StructVector *) b,
                            (nalu_hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGSolve( NALU_HYPRE_StructSolver solver,
                      NALU_HYPRE_StructMatrix A,
                      NALU_HYPRE_StructVector b,
                      NALU_HYPRE_StructVector x      )
{
   return ( nalu_hypre_SMGSolve( (void *) solver,
                            (nalu_hypre_StructMatrix *) A,
                            (nalu_hypre_StructVector *) b,
                            (nalu_hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGSetMemoryUse( NALU_HYPRE_StructSolver solver,
                             NALU_HYPRE_Int          memory_use )
{
   return ( nalu_hypre_SMGSetMemoryUse( (void *) solver, memory_use ) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructSMGGetMemoryUse( NALU_HYPRE_StructSolver solver,
                             NALU_HYPRE_Int        * memory_use )
{
   return ( nalu_hypre_SMGGetMemoryUse( (void *) solver, memory_use ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGSetTol( NALU_HYPRE_StructSolver solver,
                       NALU_HYPRE_Real         tol    )
{
   return ( nalu_hypre_SMGSetTol( (void *) solver, tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructSMGGetTol( NALU_HYPRE_StructSolver solver,
                       NALU_HYPRE_Real       * tol    )
{
   return ( nalu_hypre_SMGGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGSetMaxIter( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_Int          max_iter  )
{
   return ( nalu_hypre_SMGSetMaxIter( (void *) solver, max_iter ) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructSMGGetMaxIter( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_Int        * max_iter  )
{
   return ( nalu_hypre_SMGGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGSetRelChange( NALU_HYPRE_StructSolver solver,
                             NALU_HYPRE_Int          rel_change  )
{
   return ( nalu_hypre_SMGSetRelChange( (void *) solver, rel_change ) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructSMGGetRelChange( NALU_HYPRE_StructSolver solver,
                             NALU_HYPRE_Int        * rel_change  )
{
   return ( nalu_hypre_SMGGetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGSetZeroGuess( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_SMGSetZeroGuess( (void *) solver, 1 ) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructSMGGetZeroGuess( NALU_HYPRE_StructSolver solver,
                             NALU_HYPRE_Int * zeroguess )
{
   return ( nalu_hypre_SMGGetZeroGuess( (void *) solver, zeroguess ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGSetNonZeroGuess( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_SMGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * Note that we require at least 1 pre-relax sweep.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGSetNumPreRelax( NALU_HYPRE_StructSolver solver,
                               NALU_HYPRE_Int          num_pre_relax )
{
   return ( nalu_hypre_SMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructSMGGetNumPreRelax( NALU_HYPRE_StructSolver solver,
                               NALU_HYPRE_Int        * num_pre_relax )
{
   return ( nalu_hypre_SMGGetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGSetNumPostRelax( NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_Int          num_post_relax )
{
   return ( nalu_hypre_SMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructSMGGetNumPostRelax( NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_Int        * num_post_relax )
{
   return ( nalu_hypre_SMGGetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGSetLogging( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_Int          logging )
{
   return ( nalu_hypre_SMGSetLogging( (void *) solver, logging) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructSMGGetLogging( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_Int        * logging )
{
   return ( nalu_hypre_SMGGetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGSetPrintLevel( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int  print_level )
{
   return ( nalu_hypre_SMGSetPrintLevel( (void *) solver, print_level) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructSMGGetPrintLevel( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int      * print_level )
{
   return ( nalu_hypre_SMGGetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGGetNumIterations( NALU_HYPRE_StructSolver  solver,
                                 NALU_HYPRE_Int          *num_iterations )
{
   return ( nalu_hypre_SMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructSMGGetFinalRelativeResidualNorm( NALU_HYPRE_StructSolver  solver,
                                             NALU_HYPRE_Real         *norm   )
{
   return ( nalu_hypre_SMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
NALU_HYPRE_Int
NALU_HYPRE_StructSMGSetDeviceLevel( NALU_HYPRE_StructSolver  solver,
                               NALU_HYPRE_Int   device_level  )
{
   return (nalu_hypre_StructSMGSetDeviceLevel( (void *) solver, device_level ));
}
#endif
