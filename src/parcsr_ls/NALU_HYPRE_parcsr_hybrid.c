/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridCreate( NALU_HYPRE_Solver *solver )
{
   if (!solver)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   *solver = ( (NALU_HYPRE_Solver) nalu_hypre_AMGHybridCreate( ) );
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridDestroy( NALU_HYPRE_Solver solver )
{
   return ( nalu_hypre_AMGHybridDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetup( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_ParCSRMatrix A,
                         NALU_HYPRE_ParVector b,
                         NALU_HYPRE_ParVector x      )
{
   return ( nalu_hypre_AMGHybridSetup( (void *) solver,
                                  (nalu_hypre_ParCSRMatrix *) A,
                                  (nalu_hypre_ParVector *) b,
                                  (nalu_hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSolve( NALU_HYPRE_Solver solver,
                         NALU_HYPRE_ParCSRMatrix A,
                         NALU_HYPRE_ParVector b,
                         NALU_HYPRE_ParVector x      )
{
   return ( nalu_hypre_AMGHybridSolve( (void *) solver,
                                  (nalu_hypre_ParCSRMatrix *) A,
                                  (nalu_hypre_ParVector *) b,
                                  (nalu_hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetTol( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Real   tol    )
{
   return ( nalu_hypre_AMGHybridSetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetAbsoluteTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetAbsoluteTol( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Real   tol    )
{
   return ( nalu_hypre_AMGHybridSetAbsoluteTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetConvergenceTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetConvergenceTol( NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Real   cf_tol    )
{
   return ( nalu_hypre_AMGHybridSetConvergenceTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetDSCGMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetDSCGMaxIter( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    dscg_max_its )
{
   return ( nalu_hypre_AMGHybridSetDSCGMaxIter( (void *) solver, dscg_max_its ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetPCGMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetPCGMaxIter( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    pcg_max_its )
{
   return ( nalu_hypre_AMGHybridSetPCGMaxIter( (void *) solver, pcg_max_its ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetSetupType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetSetupType( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    setup_type )
{
   return ( nalu_hypre_AMGHybridSetSetupType( (void *) solver, setup_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetSolverType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetSolverType( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    solver_type )
{
   return ( nalu_hypre_AMGHybridSetSolverType( (void *) solver, solver_type ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetRecomputeResidual( NALU_HYPRE_Solver  solver,
                                        NALU_HYPRE_Int     recompute_residual )
{
   return ( nalu_hypre_AMGHybridSetRecomputeResidual( (void *) solver, recompute_residual ) );
}

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridGetRecomputeResidual( NALU_HYPRE_Solver  solver,
                                        NALU_HYPRE_Int    *recompute_residual )
{
   return ( nalu_hypre_AMGHybridGetRecomputeResidual( (void *) solver, recompute_residual ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetRecomputeResidualP( NALU_HYPRE_Solver  solver,
                                         NALU_HYPRE_Int     recompute_residual_p )
{
   return ( nalu_hypre_AMGHybridSetRecomputeResidualP( (void *) solver, recompute_residual_p ) );
}

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridGetRecomputeResidualP( NALU_HYPRE_Solver  solver,
                                         NALU_HYPRE_Int    *recompute_residual_p )
{
   return ( nalu_hypre_AMGHybridGetRecomputeResidualP( (void *) solver, recompute_residual_p ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetKDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetKDim( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int    k_dim    )
{
   return ( nalu_hypre_AMGHybridSetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetTwoNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetTwoNorm( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    two_norm    )
{
   return ( nalu_hypre_AMGHybridSetTwoNorm( (void *) solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetStopCrit
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetStopCrit( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int    stop_crit    )
{
   return ( nalu_hypre_AMGHybridSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelChange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetRelChange( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    rel_change    )
{
   return ( nalu_hypre_AMGHybridSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetPrecond
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetPrecond( NALU_HYPRE_Solver         solver,
                              NALU_HYPRE_PtrToParSolverFcn precond,
                              NALU_HYPRE_PtrToParSolverFcn precond_setup,
                              NALU_HYPRE_Solver         precond_solver )
{
   return ( nalu_hypre_AMGHybridSetPrecond( (void *) solver,
                                       (NALU_HYPRE_Int (*)(void*, void*, void*, void*) ) precond,
                                       (NALU_HYPRE_Int (*)(void*, void*, void*, void*) ) precond_setup,
                                       (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetLogging( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    logging    )
{
   return ( nalu_hypre_AMGHybridSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetPrintLevel( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    print_level    )
{
   return ( nalu_hypre_AMGHybridSetPrintLevel( (void *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetStrongThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetStrongThreshold( NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Real   strong_threshold    )
{
   return ( nalu_hypre_AMGHybridSetStrongThreshold( (void *) solver,
                                               strong_threshold ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetMaxRowSum
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetMaxRowSum( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Real   max_row_sum    )
{
   return ( nalu_hypre_AMGHybridSetMaxRowSum( (void *) solver, max_row_sum ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetTruncFactor
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetTruncFactor( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Real   trunc_factor    )
{
   return ( nalu_hypre_AMGHybridSetTruncFactor( (void *) solver, trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetPMaxElmts
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetPMaxElmts( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    p_max    )
{
   return ( nalu_hypre_AMGHybridSetPMaxElmts( (void *) solver, p_max ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetMaxLevels
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetMaxLevels( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    max_levels    )
{
   return ( nalu_hypre_AMGHybridSetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetMeasureType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetMeasureType( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    measure_type    )
{
   return ( nalu_hypre_AMGHybridSetMeasureType( (void *) solver, measure_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetCoarsenType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetCoarsenType( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int    coarsen_type    )
{
   return ( nalu_hypre_AMGHybridSetCoarsenType( (void *) solver, coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetInterpType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetInterpType( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    interp_type    )
{
   return ( nalu_hypre_AMGHybridSetInterpType( (void *) solver, interp_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetCycleType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetCycleType( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    cycle_type    )
{
   return ( nalu_hypre_AMGHybridSetCycleType( (void *) solver, cycle_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNumGridSweeps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetNumGridSweeps( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int   *num_grid_sweeps    )
{
   return ( nalu_hypre_AMGHybridSetNumGridSweeps( (void *) solver, num_grid_sweeps ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetGridRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetGridRelaxType( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int   *grid_relax_type    )
{
   return ( nalu_hypre_AMGHybridSetGridRelaxType( (void *) solver, grid_relax_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetGridRelaxPoints
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetGridRelaxPoints( NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int  **grid_relax_points    )
{
   return ( nalu_hypre_AMGHybridSetGridRelaxPoints( (void *) solver, grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNumSweeps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetNumSweeps( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    num_sweeps    )
{
   return ( nalu_hypre_AMGHybridSetNumSweeps( (void *) solver, num_sweeps ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetCycleNumSweeps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetCycleNumSweeps( NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int    num_sweeps,
                                     NALU_HYPRE_Int    k )
{
   return ( nalu_hypre_AMGHybridSetCycleNumSweeps( (void *) solver, num_sweeps, k ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetRelaxType( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    relax_type    )
{
   return ( nalu_hypre_AMGHybridSetRelaxType( (void *) solver, relax_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetCycleRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetCycleRelaxType( NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int    relax_type,
                                     NALU_HYPRE_Int    k )
{
   return ( nalu_hypre_AMGHybridSetCycleRelaxType( (void *) solver, relax_type, k ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelaxOrder
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetRelaxOrder( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    relax_order    )
{
   return ( nalu_hypre_AMGHybridSetRelaxOrder( (void *) solver, relax_order ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetKeepTranspose
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetKeepTranspose( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    keepT    )
{
   return ( nalu_hypre_AMGHybridSetKeepTranspose( (void *) solver, keepT ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetMaxCoarseSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetMaxCoarseSize( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    max_coarse_size    )
{
   return ( nalu_hypre_AMGHybridSetMaxCoarseSize( (void *) solver, max_coarse_size ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetMinCoarseSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetMinCoarseSize( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    min_coarse_size    )
{
   return ( nalu_hypre_AMGHybridSetMinCoarseSize( (void *) solver, min_coarse_size ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetSeqThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetSeqThreshold( NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int    seq_threshold    )
{
   return ( nalu_hypre_AMGHybridSetSeqThreshold( (void *) solver, seq_threshold ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelaxWt
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetRelaxWt( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Real   relax_wt    )
{
   return ( nalu_hypre_AMGHybridSetRelaxWt( (void *) solver, relax_wt ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetLevelRelaxWt( NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Real   relax_wt,
                                   NALU_HYPRE_Int    level )
{
   return ( nalu_hypre_AMGHybridSetLevelRelaxWt( (void *) solver, relax_wt, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetOuterWt
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetOuterWt( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Real   outer_wt    )
{
   return ( nalu_hypre_AMGHybridSetOuterWt( (void *) solver, outer_wt ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetLevelOuterWt
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetLevelOuterWt( NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Real   outer_wt,
                                   NALU_HYPRE_Int    level )
{
   return ( nalu_hypre_AMGHybridSetLevelOuterWt( (void *) solver, outer_wt, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetRelaxWeight
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetRelaxWeight( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Real  *relax_weight    )
{
   return ( nalu_hypre_AMGHybridSetRelaxWeight( (void *) solver, relax_weight ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetOmega
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetOmega( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Real  *omega    )
{
   return ( nalu_hypre_AMGHybridSetOmega( (void *) solver, omega ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetAggNumLevels
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetAggNumLevels( NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int    agg_num_levels    )
{
   return ( nalu_hypre_AMGHybridSetAggNumLevels( (void *) solver, agg_num_levels ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetAggInterpType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetAggInterpType( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    agg_interp_type    )
{
   return ( nalu_hypre_AMGHybridSetAggInterpType( (void *) solver, agg_interp_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNumPaths
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetNumPaths( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int    num_paths    )
{
   return ( nalu_hypre_AMGHybridSetNumPaths( (void *) solver, num_paths ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNumFunctions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetNumFunctions( NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Int    num_functions    )
{
   return ( nalu_hypre_AMGHybridSetNumFunctions( (void *) solver, num_functions ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNodal
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetNodal( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int    nodal    )
{
   return ( nalu_hypre_AMGHybridSetNodal( (void *) solver, nodal ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetDofFunc
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetDofFunc( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int   *dof_func    )
{
   return ( nalu_hypre_AMGHybridSetDofFunc( (void *) solver, dof_func ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridSetNonGalerkTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridSetNonGalerkinTol( NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Int   nongalerk_num_tol,
                                     NALU_HYPRE_Real  *nongalerkin_tol)
{
   return ( nalu_hypre_AMGHybridSetNonGalerkinTol( (void *) solver, nongalerk_num_tol, nongalerkin_tol ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridGetNumIterations( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int   *num_its    )
{
   return ( nalu_hypre_AMGHybridGetNumIterations( (void *) solver, num_its ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridGetDSCGNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridGetDSCGNumIterations( NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Int   *dscg_num_its )
{
   return ( nalu_hypre_AMGHybridGetDSCGNumIterations( (void *) solver, dscg_num_its ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridGetPCGNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridGetPCGNumIterations( NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int   *pcg_num_its )
{
   return ( nalu_hypre_AMGHybridGetPCGNumIterations( (void *) solver, pcg_num_its ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm( NALU_HYPRE_Solver solver,
                                                NALU_HYPRE_Real  *norm    )
{
   return ( nalu_hypre_AMGHybridGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}


NALU_HYPRE_Int
NALU_HYPRE_ParCSRHybridGetSetupSolveTime( NALU_HYPRE_Solver solver,
                                     NALU_HYPRE_Real  *time    )
{
   return ( nalu_hypre_AMGHybridGetSetupSolveTime( (void *) solver, time ) );
}
