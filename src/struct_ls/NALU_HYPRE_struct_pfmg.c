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
NALU_HYPRE_StructPFMGCreate( MPI_Comm comm, NALU_HYPRE_StructSolver *solver )
{
   *solver = ( (NALU_HYPRE_StructSolver) nalu_hypre_PFMGCreate( comm ) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGDestroy( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_PFMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetup( NALU_HYPRE_StructSolver solver,
                       NALU_HYPRE_StructMatrix A,
                       NALU_HYPRE_StructVector b,
                       NALU_HYPRE_StructVector x      )
{
   return ( nalu_hypre_PFMGSetup( (void *) solver,
                             (nalu_hypre_StructMatrix *) A,
                             (nalu_hypre_StructVector *) b,
                             (nalu_hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSolve( NALU_HYPRE_StructSolver solver,
                       NALU_HYPRE_StructMatrix A,
                       NALU_HYPRE_StructVector b,
                       NALU_HYPRE_StructVector x      )
{
   return ( nalu_hypre_PFMGSolve( (void *) solver,
                             (nalu_hypre_StructMatrix *) A,
                             (nalu_hypre_StructVector *) b,
                             (nalu_hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetTol( NALU_HYPRE_StructSolver solver,
                        NALU_HYPRE_Real         tol    )
{
   return ( nalu_hypre_PFMGSetTol( (void *) solver, tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetTol( NALU_HYPRE_StructSolver solver,
                        NALU_HYPRE_Real       * tol    )
{
   return ( nalu_hypre_PFMGGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetMaxIter( NALU_HYPRE_StructSolver solver,
                            NALU_HYPRE_Int          max_iter  )
{
   return ( nalu_hypre_PFMGSetMaxIter( (void *) solver, max_iter ) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetMaxIter( NALU_HYPRE_StructSolver solver,
                            NALU_HYPRE_Int        * max_iter  )
{
   return ( nalu_hypre_PFMGGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetMaxLevels( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int          max_levels  )
{
   return ( nalu_hypre_PFMGSetMaxLevels( (void *) solver, max_levels ) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetMaxLevels( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int        * max_levels  )
{
   return ( nalu_hypre_PFMGGetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetRelChange( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int          rel_change  )
{
   return ( nalu_hypre_PFMGSetRelChange( (void *) solver, rel_change ) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetRelChange( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int        * rel_change  )
{
   return ( nalu_hypre_PFMGGetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetZeroGuess( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_PFMGSetZeroGuess( (void *) solver, 1 ) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetZeroGuess( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int * zeroguess )
{
   return ( nalu_hypre_PFMGGetZeroGuess( (void *) solver, zeroguess ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetNonZeroGuess( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_PFMGSetZeroGuess( (void *) solver, 0 ) );
}

/*--------------------------------------------------------------------------
 * GetJacobiWeight will not return the actual weight
 * if SetJacobiWeight has not been called.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetRelaxType( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int          relax_type )
{
   return ( nalu_hypre_PFMGSetRelaxType( (void *) solver, relax_type) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetRelaxType( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int        * relax_type )
{
   return ( nalu_hypre_PFMGGetRelaxType( (void *) solver, relax_type) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetJacobiWeight(NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_Real         weight)
{
   return ( nalu_hypre_PFMGSetJacobiWeight( (void *) solver, weight) );
}
NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetJacobiWeight(NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_Real        *weight)
{
   return ( nalu_hypre_PFMGGetJacobiWeight( (void *) solver, weight) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetRAPType( NALU_HYPRE_StructSolver solver,
                            NALU_HYPRE_Int          rap_type )
{
   return ( nalu_hypre_PFMGSetRAPType( (void *) solver, rap_type) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetRAPType( NALU_HYPRE_StructSolver solver,
                            NALU_HYPRE_Int        * rap_type )
{
   return ( nalu_hypre_PFMGGetRAPType( (void *) solver, rap_type) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetNumPreRelax( NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_Int          num_pre_relax )
{
   return ( nalu_hypre_PFMGSetNumPreRelax( (void *) solver, num_pre_relax) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetNumPreRelax( NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_Int        * num_pre_relax )
{
   return ( nalu_hypre_PFMGGetNumPreRelax( (void *) solver, num_pre_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetNumPostRelax( NALU_HYPRE_StructSolver solver,
                                 NALU_HYPRE_Int          num_post_relax )
{
   return ( nalu_hypre_PFMGSetNumPostRelax( (void *) solver, num_post_relax) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetNumPostRelax( NALU_HYPRE_StructSolver solver,
                                 NALU_HYPRE_Int        * num_post_relax )
{
   return ( nalu_hypre_PFMGGetNumPostRelax( (void *) solver, num_post_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetSkipRelax( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int          skip_relax )
{
   return ( nalu_hypre_PFMGSetSkipRelax( (void *) solver, skip_relax) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetSkipRelax( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int        * skip_relax )
{
   return ( nalu_hypre_PFMGGetSkipRelax( (void *) solver, skip_relax) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetDxyz( NALU_HYPRE_StructSolver  solver,
                         NALU_HYPRE_Real         *dxyz   )
{
   return ( nalu_hypre_PFMGSetDxyz( (void *) solver, dxyz) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetLogging( NALU_HYPRE_StructSolver solver,
                            NALU_HYPRE_Int          logging )
{
   return ( nalu_hypre_PFMGSetLogging( (void *) solver, logging) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetLogging( NALU_HYPRE_StructSolver solver,
                            NALU_HYPRE_Int        * logging )
{
   return ( nalu_hypre_PFMGGetLogging( (void *) solver, logging) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetPrintLevel( NALU_HYPRE_StructSolver solver,
                               NALU_HYPRE_Int            print_level )
{
   return ( nalu_hypre_PFMGSetPrintLevel( (void *) solver, print_level) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetPrintLevel( NALU_HYPRE_StructSolver solver,
                               NALU_HYPRE_Int          * print_level )
{
   return ( nalu_hypre_PFMGGetPrintLevel( (void *) solver, print_level) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetNumIterations( NALU_HYPRE_StructSolver  solver,
                                  NALU_HYPRE_Int          *num_iterations )
{
   return ( nalu_hypre_PFMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructPFMGGetFinalRelativeResidualNorm( NALU_HYPRE_StructSolver  solver,
                                              NALU_HYPRE_Real         *norm   )
{
   return ( nalu_hypre_PFMGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

#if 0 //defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
NALU_HYPRE_Int
NALU_HYPRE_StructPFMGSetDeviceLevel( NALU_HYPRE_StructSolver  solver,
                                NALU_HYPRE_Int   device_level  )
{
   return ( nalu_hypre_PFMGSetDeviceLevel( (void *) solver, device_level ) );
}
#endif
