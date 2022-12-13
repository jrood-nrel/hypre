/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRCreate( NALU_HYPRE_Solver *solver )
{
   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   *solver = ( (NALU_HYPRE_Solver) hypre_MGRCreate( ) );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRDestroy( NALU_HYPRE_Solver solver )
{
   return ( hypre_MGRDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetup( NALU_HYPRE_Solver solver,
                NALU_HYPRE_ParCSRMatrix A,
                NALU_HYPRE_ParVector b,
                NALU_HYPRE_ParVector x      )
{
   if (!A)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   return ( hypre_MGRSetup( (void *) solver,
                            (hypre_ParCSRMatrix *) A,
                            (hypre_ParVector *) b,
                            (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSolve( NALU_HYPRE_Solver solver,
                NALU_HYPRE_ParCSRMatrix A,
                NALU_HYPRE_ParVector b,
                NALU_HYPRE_ParVector x      )
{
   if (!A)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   if (!b)
   {
      hypre_error_in_arg(3);
      return hypre_error_flag;
   }

   if (!x)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   return ( hypre_MGRSolve( (void *) solver,
                            (hypre_ParCSRMatrix *) A,
                            (hypre_ParVector *) b,
                            (hypre_ParVector *) x ) );
}

#ifdef NALU_HYPRE_USING_DSUPERLU
/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDirectSolverCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRDirectSolverCreate( NALU_HYPRE_Solver *solver )
{
   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   *solver = ( (NALU_HYPRE_Solver) hypre_MGRDirectSolverCreate( ) );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDirectSolverDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRDirectSolverDestroy( NALU_HYPRE_Solver solver )
{
   return ( hypre_MGRDirectSolverDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDirectSolverSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRDirectSolverSetup( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_ParVector b,
                            NALU_HYPRE_ParVector x      )
{
   return ( hypre_MGRDirectSolverSetup( (void *) solver,
                                        (hypre_ParCSRMatrix *) A,
                                        (hypre_ParVector *) b,
                                        (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRDirectSolverSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRDirectSolverSolve( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_ParCSRMatrix A,
                            NALU_HYPRE_ParVector b,
                            NALU_HYPRE_ParVector x      )
{
   return ( hypre_MGRDirectSolverSolve( (void *) solver,
                                        (hypre_ParCSRMatrix *) A,
                                        (hypre_ParVector *) b,
                                        (hypre_ParVector *) x ) );
}
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCpointsByContiguousBlock
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetCpointsByContiguousBlock( NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Int  block_size,
                                      NALU_HYPRE_Int  max_num_levels,
                                      NALU_HYPRE_BigInt  *idx_array,
                                      NALU_HYPRE_Int  *block_num_coarse_points,
                                      NALU_HYPRE_Int  **block_coarse_indexes)
{
   return ( hypre_MGRSetCpointsByContiguousBlock( (void *) solver, block_size, max_num_levels,
                                                  idx_array, block_num_coarse_points, block_coarse_indexes));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCpointsByBlock
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetCpointsByBlock( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int  block_size,
                            NALU_HYPRE_Int  max_num_levels,
                            NALU_HYPRE_Int *block_num_coarse_points,
                            NALU_HYPRE_Int  **block_coarse_indexes)
{
   return ( hypre_MGRSetCpointsByBlock( (void *) solver, block_size, max_num_levels,
                                        block_num_coarse_points, block_coarse_indexes));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCpointsByPointMarkerArray
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetCpointsByPointMarkerArray( NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int  block_size,
                                       NALU_HYPRE_Int  max_num_levels,
                                       NALU_HYPRE_Int  *num_block_coarse_points,
                                       NALU_HYPRE_Int  **lvl_block_coarse_indexes,
                                       NALU_HYPRE_Int  *point_marker_array)
{
   return ( hypre_MGRSetCpointsByPointMarkerArray( (void *) solver, block_size, max_num_levels,
                                                   num_block_coarse_points, lvl_block_coarse_indexes, point_marker_array));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetNonCpointsToFpoints
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetNonCpointsToFpoints( NALU_HYPRE_Solver solver, NALU_HYPRE_Int nonCptToFptFlag)
{
   return hypre_MGRSetNonCpointsToFpoints((void *) solver, nonCptToFptFlag);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetFSolver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetFSolver(NALU_HYPRE_Solver          solver,
                    NALU_HYPRE_PtrToParSolverFcn  fine_grid_solver_solve,
                    NALU_HYPRE_PtrToParSolverFcn  fine_grid_solver_setup,
                    NALU_HYPRE_Solver          fsolver )
{
   return ( hypre_MGRSetFSolver( (void *) solver,
                                 (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) fine_grid_solver_solve,
                                 (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) fine_grid_solver_setup,
                                 (void *) fsolver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRBuildAff
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRBuildAff(NALU_HYPRE_ParCSRMatrix A,
                  NALU_HYPRE_Int *CF_marker,
                  NALU_HYPRE_Int debug_flag,
                  NALU_HYPRE_ParCSRMatrix *A_ff)
{
   return (hypre_MGRBuildAff(A, CF_marker, debug_flag, A_ff));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCoarseSolver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetCoarseSolver(NALU_HYPRE_Solver          solver,
                         NALU_HYPRE_PtrToParSolverFcn  coarse_grid_solver_solve,
                         NALU_HYPRE_PtrToParSolverFcn  coarse_grid_solver_setup,
                         NALU_HYPRE_Solver          coarse_grid_solver )
{
   return ( hypre_MGRSetCoarseSolver( (void *) solver,
                                      (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) coarse_grid_solver_solve,
                                      (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) coarse_grid_solver_setup,
                                      (void *) coarse_grid_solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetMaxCoarseLevels
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetMaxCoarseLevels( NALU_HYPRE_Solver solver, NALU_HYPRE_Int maxlev )
{
   return hypre_MGRSetMaxCoarseLevels(solver, maxlev);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetBlockSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetBlockSize( NALU_HYPRE_Solver solver, NALU_HYPRE_Int bsize )
{
   return hypre_MGRSetBlockSize(solver, bsize );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetReservedCoarseNodes
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetReservedCoarseNodes( NALU_HYPRE_Solver solver, NALU_HYPRE_Int reserved_coarse_size,
                                 NALU_HYPRE_BigInt *reserved_coarse_indexes )
{
   return hypre_MGRSetReservedCoarseNodes(solver, reserved_coarse_size, reserved_coarse_indexes );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetReservedCpointsLevelToKeep
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetReservedCpointsLevelToKeep( NALU_HYPRE_Solver solver, NALU_HYPRE_Int level)
{
   return hypre_MGRSetReservedCpointsLevelToKeep((void *) solver, level);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetRestrictType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetRestrictType(NALU_HYPRE_Solver solver, NALU_HYPRE_Int restrict_type )
{
   return hypre_MGRSetRestrictType(solver, restrict_type );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelRestrictType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelRestrictType( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *restrict_type )
{
   return hypre_MGRSetLevelRestrictType( solver, restrict_type );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetFRelaxMethod
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetFRelaxMethod(NALU_HYPRE_Solver solver, NALU_HYPRE_Int relax_method )
{
   return hypre_MGRSetFRelaxMethod(solver, relax_method );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelFRelaxMethod
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelFRelaxMethod( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *relax_method )
{
   return hypre_MGRSetLevelFRelaxMethod( solver, relax_method );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelFRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelFRelaxType( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *relax_type )
{
   return hypre_MGRSetLevelFRelaxType( solver, relax_type );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCoarseGridMethod
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetCoarseGridMethod( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *cg_method )
{
   return hypre_MGRSetCoarseGridMethod( solver, cg_method );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelFRelaxNumFunctions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelFRelaxNumFunctions( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_functions )
{
   return hypre_MGRSetLevelFRelaxNumFunctions( solver, num_functions );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetRelaxType(NALU_HYPRE_Solver solver, NALU_HYPRE_Int relax_type )
{
   return hypre_MGRSetRelaxType(solver, relax_type );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetNumRelaxSweeps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetNumRelaxSweeps( NALU_HYPRE_Solver solver, NALU_HYPRE_Int nsweeps )
{
   return hypre_MGRSetNumRelaxSweeps(solver, nsweeps);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelNumRelaxSweeps
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelNumRelaxSweeps( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *nsweeps )
{
   return hypre_MGRSetLevelNumRelaxSweeps(solver, nsweeps);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetInterpType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetInterpType( NALU_HYPRE_Solver solver, NALU_HYPRE_Int interpType )
{
   return hypre_MGRSetInterpType(solver, interpType);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelInterpType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelInterpType( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *interpType )
{
   return hypre_MGRSetLevelInterpType(solver, interpType);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetNumInterpSweeps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetNumInterpSweeps( NALU_HYPRE_Solver solver, NALU_HYPRE_Int nsweeps )
{
   return hypre_MGRSetNumInterpSweeps(solver, nsweeps);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetNumRestrictSweeps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetNumRestrictSweeps( NALU_HYPRE_Solver solver, NALU_HYPRE_Int nsweeps )
{
   return hypre_MGRSetNumRestrictSweeps(solver, nsweeps);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetTruncateCoarseGridThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetTruncateCoarseGridThreshold( NALU_HYPRE_Solver solver, NALU_HYPRE_Real threshold)
{
   return hypre_MGRSetTruncateCoarseGridThreshold( solver, threshold );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetBlockJacobiBlockSize
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetBlockJacobiBlockSize( NALU_HYPRE_Solver solver, NALU_HYPRE_Int blk_size )
{
   return hypre_MGRSetBlockJacobiBlockSize(solver, blk_size);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetFrelaxPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetFrelaxPrintLevel( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level )
{
   return hypre_MGRSetFrelaxPrintLevel( solver, print_level );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetCoarseGridPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetCoarseGridPrintLevel( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level )
{
   return hypre_MGRSetCoarseGridPrintLevel( solver, print_level );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetPrintLevel( NALU_HYPRE_Solver solver, NALU_HYPRE_Int print_level )
{
   return hypre_MGRSetPrintLevel( solver, print_level );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetLogging( NALU_HYPRE_Solver solver, NALU_HYPRE_Int logging )
{
   return hypre_MGRSetLogging(solver, logging );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetMaxIter( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter )
{
   return hypre_MGRSetMaxIter( solver, max_iter );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetTol( NALU_HYPRE_Solver solver, NALU_HYPRE_Real tol )
{
   return hypre_MGRSetTol( solver, tol );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetMaxGlobalsmoothIters
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetMaxGlobalSmoothIters( NALU_HYPRE_Solver solver, NALU_HYPRE_Int max_iter )
{
   return hypre_MGRSetMaxGlobalSmoothIters(solver, max_iter);
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelsmoothIters
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelSmoothIters( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int *smooth_iters )
{
   return hypre_MGRSetLevelSmoothIters(solver, smooth_iters);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetGlobalsmoothType
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetGlobalSmoothType( NALU_HYPRE_Solver solver, NALU_HYPRE_Int iter_type )
{
   return hypre_MGRSetGlobalSmoothType(solver, iter_type);
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetLevelsmoothType
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetLevelSmoothType( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int *smooth_type )
{
   return hypre_MGRSetLevelSmoothType(solver, smooth_type);
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetGlobalSmoothCycle
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_MGRSetGlobalSmoothCycle( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int global_smooth_cycle )
{
   return hypre_MGRSetGlobalSmoothCycle(solver, global_smooth_cycle);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRSetMaxPElmts
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRSetPMaxElmts( NALU_HYPRE_Solver solver, NALU_HYPRE_Int P_max_elmts )
{
   return hypre_MGRSetPMaxElmts(solver, P_max_elmts);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRGetCoarseGridConvergenceFactor
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRGetCoarseGridConvergenceFactor( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *conv_factor )
{
   return hypre_MGRGetCoarseGridConvergenceFactor( solver, conv_factor );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRGetNumIterations( NALU_HYPRE_Solver solver, NALU_HYPRE_Int *num_iterations )
{
   return hypre_MGRGetNumIterations( solver, num_iterations );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MGRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MGRGetFinalRelativeResidualNorm( NALU_HYPRE_Solver solver, NALU_HYPRE_Real *res_norm )
{
   return hypre_MGRGetFinalRelativeResidualNorm(solver, res_norm);
}
