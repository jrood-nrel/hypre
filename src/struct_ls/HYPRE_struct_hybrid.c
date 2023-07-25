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
NALU_HYPRE_StructHybridCreate( MPI_Comm comm, NALU_HYPRE_StructSolver *solver )
{
   *solver = ( (NALU_HYPRE_StructSolver) nalu_hypre_HybridCreate( comm ) );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridDestroy( NALU_HYPRE_StructSolver solver )
{
   return ( nalu_hypre_HybridDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetup( NALU_HYPRE_StructSolver solver,
                         NALU_HYPRE_StructMatrix A,
                         NALU_HYPRE_StructVector b,
                         NALU_HYPRE_StructVector x      )
{
   return ( nalu_hypre_HybridSetup( (void *) solver,
                               (nalu_hypre_StructMatrix *) A,
                               (nalu_hypre_StructVector *) b,
                               (nalu_hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSolve( NALU_HYPRE_StructSolver solver,
                         NALU_HYPRE_StructMatrix A,
                         NALU_HYPRE_StructVector b,
                         NALU_HYPRE_StructVector x      )
{
   return ( nalu_hypre_HybridSolve( (void *) solver,
                               (nalu_hypre_StructMatrix *) A,
                               (nalu_hypre_StructVector *) b,
                               (nalu_hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetTol( NALU_HYPRE_StructSolver solver,
                          NALU_HYPRE_Real         tol    )
{
   return ( nalu_hypre_HybridSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetConvergenceTol( NALU_HYPRE_StructSolver solver,
                                     NALU_HYPRE_Real         cf_tol    )
{
   return ( nalu_hypre_HybridSetConvergenceTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetDSCGMaxIter( NALU_HYPRE_StructSolver solver,
                                  NALU_HYPRE_Int          dscg_max_its )
{
   return ( nalu_hypre_HybridSetDSCGMaxIter( (void *) solver, dscg_max_its ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetPCGMaxIter( NALU_HYPRE_StructSolver solver,
                                 NALU_HYPRE_Int          pcg_max_its )
{
   return ( nalu_hypre_HybridSetPCGMaxIter( (void *) solver, pcg_max_its ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetPCGAbsoluteTolFactor( NALU_HYPRE_StructSolver solver,
                                           NALU_HYPRE_Real  pcg_atolf )
{
   return ( nalu_hypre_HybridSetPCGAbsoluteTolFactor( (void *) solver, pcg_atolf ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetTwoNorm( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int          two_norm    )
{
   return ( nalu_hypre_HybridSetTwoNorm( (void *) solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetStopCrit( NALU_HYPRE_StructSolver solver,
                               NALU_HYPRE_Int          stop_crit    )
{
   return ( nalu_hypre_HybridSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetRelChange( NALU_HYPRE_StructSolver solver,
                                NALU_HYPRE_Int          rel_change    )
{
   return ( nalu_hypre_HybridSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetSolverType( NALU_HYPRE_StructSolver solver,
                                 NALU_HYPRE_Int          solver_type    )
{
   return ( nalu_hypre_HybridSetSolverType( (void *) solver, solver_type ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetRecomputeResidual( NALU_HYPRE_StructSolver  solver,
                                        NALU_HYPRE_Int           recompute_residual )
{
   return ( nalu_hypre_HybridSetRecomputeResidual( (void *) solver, recompute_residual ) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructHybridGetRecomputeResidual( NALU_HYPRE_StructSolver  solver,
                                        NALU_HYPRE_Int          *recompute_residual )
{
   return ( nalu_hypre_HybridGetRecomputeResidual( (void *) solver, recompute_residual ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetRecomputeResidualP( NALU_HYPRE_StructSolver  solver,
                                         NALU_HYPRE_Int           recompute_residual_p )
{
   return ( nalu_hypre_HybridSetRecomputeResidualP( (void *) solver, recompute_residual_p ) );
}

NALU_HYPRE_Int
NALU_HYPRE_StructHybridGetRecomputeResidualP( NALU_HYPRE_StructSolver  solver,
                                         NALU_HYPRE_Int          *recompute_residual_p )
{
   return ( nalu_hypre_HybridGetRecomputeResidualP( (void *) solver, recompute_residual_p ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetKDim( NALU_HYPRE_StructSolver solver,
                           NALU_HYPRE_Int          k_dim    )
{
   return ( nalu_hypre_HybridSetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetPrecond( NALU_HYPRE_StructSolver         solver,
                              NALU_HYPRE_PtrToStructSolverFcn precond,
                              NALU_HYPRE_PtrToStructSolverFcn precond_setup,
                              NALU_HYPRE_StructSolver         precond_solver )
{
   return ( nalu_hypre_HybridSetPrecond( (void *) solver,
                                    (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) precond,
                                    (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) precond_setup,
                                    (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetLogging( NALU_HYPRE_StructSolver solver,
                              NALU_HYPRE_Int          logging    )
{
   return ( nalu_hypre_HybridSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridSetPrintLevel( NALU_HYPRE_StructSolver solver,
                                 NALU_HYPRE_Int          print_level    )
{
   return ( nalu_hypre_HybridSetPrintLevel( (void *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridGetNumIterations( NALU_HYPRE_StructSolver solver,
                                    NALU_HYPRE_Int         *num_its    )
{
   return ( nalu_hypre_HybridGetNumIterations( (void *) solver, num_its ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridGetDSCGNumIterations( NALU_HYPRE_StructSolver solver,
                                        NALU_HYPRE_Int         *dscg_num_its )
{
   return ( nalu_hypre_HybridGetDSCGNumIterations( (void *) solver, dscg_num_its ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridGetPCGNumIterations( NALU_HYPRE_StructSolver solver,
                                       NALU_HYPRE_Int         *pcg_num_its )
{
   return ( nalu_hypre_HybridGetPCGNumIterations( (void *) solver, pcg_num_its ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_StructHybridGetFinalRelativeResidualNorm( NALU_HYPRE_StructSolver solver,
                                                NALU_HYPRE_Real        *norm    )
{
   return ( nalu_hypre_HybridGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

