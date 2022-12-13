/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGCreate( NALU_HYPRE_Solver *solver)
{
   if (!solver)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }
   *solver = (NALU_HYPRE_Solver) hypre_BoomerAMGCreate( ) ;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGDestroy( NALU_HYPRE_Solver solver )
{
   return ( hypre_BoomerAMGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetup( NALU_HYPRE_Solver solver,
                      NALU_HYPRE_ParCSRMatrix A,
                      NALU_HYPRE_ParVector b,
                      NALU_HYPRE_ParVector x      )
{
   if (!A)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }

   return ( hypre_BoomerAMGSetup( (void *) solver,
                                  (hypre_ParCSRMatrix *) A,
                                  (hypre_ParVector *) b,
                                  (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSolve( NALU_HYPRE_Solver solver,
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

   return ( hypre_BoomerAMGSolve( (void *) solver,
                                  (hypre_ParCSRMatrix *) A,
                                  (hypre_ParVector *) b,
                                  (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSolveT
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSolveT( NALU_HYPRE_Solver solver,
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

   return ( hypre_BoomerAMGSolveT( (void *) solver,
                                   (hypre_ParCSRMatrix *) A,
                                   (hypre_ParVector *) b,
                                   (hypre_ParVector *) x ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRestriction
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetRestriction( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int    restr_par  )
{
   return ( hypre_BoomerAMGSetRestriction( (void *) solver, restr_par ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetIsTriangular
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetIsTriangular( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    is_triangular  )
{
   return ( hypre_BoomerAMGSetIsTriangular( (void *) solver, is_triangular ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetGMRESSwitchR
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetGMRESSwitchR( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    gmres_switch  )
{
   return ( hypre_BoomerAMGSetGMRESSwitchR( (void *) solver, gmres_switch ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxLevels, NALU_HYPRE_BoomerAMGGetMaxLevels
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetMaxLevels( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int          max_levels  )
{
   return ( hypre_BoomerAMGSetMaxLevels( (void *) solver, max_levels ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetMaxLevels( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int        * max_levels  )
{
   return ( hypre_BoomerAMGGetMaxLevels( (void *) solver, max_levels ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxCoarseSize, NALU_HYPRE_BoomerAMGGetMaxCoarseSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetMaxCoarseSize( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int          max_coarse_size  )
{
   return ( hypre_BoomerAMGSetMaxCoarseSize( (void *) solver, max_coarse_size ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetMaxCoarseSize( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int        * max_coarse_size  )
{
   return ( hypre_BoomerAMGGetMaxCoarseSize( (void *) solver, max_coarse_size ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMinCoarseSize, NALU_HYPRE_BoomerAMGGetMinCoarseSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetMinCoarseSize( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int          min_coarse_size  )
{
   return ( hypre_BoomerAMGSetMinCoarseSize( (void *) solver, min_coarse_size ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetMinCoarseSize( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int        * min_coarse_size  )
{
   return ( hypre_BoomerAMGGetMinCoarseSize( (void *) solver, min_coarse_size ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSeqThreshold, NALU_HYPRE_BoomerAMGGetSeqThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetSeqThreshold( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int          seq_threshold  )
{
   return ( hypre_BoomerAMGSetSeqThreshold( (void *) solver, seq_threshold ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetSeqThreshold( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int        * seq_threshold  )
{
   return ( hypre_BoomerAMGGetSeqThreshold( (void *) solver, seq_threshold ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRedundant, NALU_HYPRE_BoomerAMGGetRedundant
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetRedundant( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int          redundant  )
{
   return ( hypre_BoomerAMGSetRedundant( (void *) solver, redundant ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetRedundant( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int        * redundant  )
{
   return ( hypre_BoomerAMGGetRedundant( (void *) solver, redundant ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRedundant, NALU_HYPRE_BoomerAMGGetRedundant
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetCoarsenCutFactor( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Int    coarsen_cut_factor )
{
   return ( hypre_BoomerAMGSetCoarsenCutFactor( (void *) solver, coarsen_cut_factor ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetCoarsenCutFactor( NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int    *coarsen_cut_factor )
{
   return ( hypre_BoomerAMGGetCoarsenCutFactor( (void *) solver, coarsen_cut_factor ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetStrongThreshold, NALU_HYPRE_BoomerAMGGetStrongThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetStrongThreshold( NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Real   strong_threshold  )
{
   return ( hypre_BoomerAMGSetStrongThreshold( (void *) solver,
                                               strong_threshold ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetStrongThreshold( NALU_HYPRE_Solver solver,
                                   NALU_HYPRE_Real * strong_threshold  )
{
   return ( hypre_BoomerAMGGetStrongThreshold( (void *) solver,
                                               strong_threshold ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetStrongThresholdR( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Real   strong_threshold  )
{
   return ( hypre_BoomerAMGSetStrongThresholdR( (void *) solver,
                                                strong_threshold ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetStrongThresholdR( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Real * strong_threshold  )
{
   return ( hypre_BoomerAMGGetStrongThresholdR( (void *) solver,
                                                strong_threshold ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetFilterThresholdR( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Real   filter_threshold  )
{
   return ( hypre_BoomerAMGSetFilterThresholdR( (void *) solver,
                                                filter_threshold ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetFilterThresholdR( NALU_HYPRE_Solver solver,
                                    NALU_HYPRE_Real * filter_threshold  )
{
   return ( hypre_BoomerAMGGetFilterThresholdR( (void *) solver,
                                                filter_threshold ) );
}


NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetSabs( NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int    Sabs  )
{
   return ( hypre_BoomerAMGSetSabs( (void *) solver,
                                    Sabs ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxRowSum, NALU_HYPRE_BoomerAMGGetMaxRowSum
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetMaxRowSum( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Real   max_row_sum  )
{
   return ( hypre_BoomerAMGSetMaxRowSum( (void *) solver,
                                         max_row_sum ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetMaxRowSum( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Real * max_row_sum  )
{
   return ( hypre_BoomerAMGGetMaxRowSum( (void *) solver,
                                         max_row_sum ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetTruncFactor, NALU_HYPRE_BoomerAMGGetTruncFactor
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetTruncFactor( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Real   trunc_factor  )
{
   return ( hypre_BoomerAMGSetTruncFactor( (void *) solver,
                                           trunc_factor ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetTruncFactor( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Real * trunc_factor  )
{
   return ( hypre_BoomerAMGGetTruncFactor( (void *) solver,
                                           trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetPMaxElmts, NALU_HYPRE_BoomerAMGGetPMaxElmts
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetPMaxElmts( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int   P_max_elmts  )
{
   return ( hypre_BoomerAMGSetPMaxElmts( (void *) solver,
                                         P_max_elmts ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetPMaxElmts( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int   * P_max_elmts  )
{
   return ( hypre_BoomerAMGGetPMaxElmts( (void *) solver,
                                         P_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold, NALU_HYPRE_BoomerAMGGetJacobiTruncThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold( NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Real   jacobi_trunc_threshold  )
{
   return ( hypre_BoomerAMGSetJacobiTruncThreshold( (void *) solver,
                                                    jacobi_trunc_threshold ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetJacobiTruncThreshold( NALU_HYPRE_Solver solver,
                                        NALU_HYPRE_Real * jacobi_trunc_threshold  )
{
   return ( hypre_BoomerAMGGetJacobiTruncThreshold( (void *) solver,
                                                    jacobi_trunc_threshold ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetPostInterpType, NALU_HYPRE_BoomerAMGGetPostInterpType
 *  If >0, specifies something to do to improve a computed interpolation matrix.
 * defaults to 0, for nothing.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetPostInterpType( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int       post_interp_type  )
{
   return ( hypre_BoomerAMGSetPostInterpType( (void *) solver,
                                              post_interp_type ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetPostInterpType( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Int     * post_interp_type  )
{
   return ( hypre_BoomerAMGGetPostInterpType( (void *) solver,
                                              post_interp_type ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSCommPkgSwitch
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetSCommPkgSwitch( NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Real   S_commpkg_switch  )
{
   return 0;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetInterpType( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int          interp_type  )
{
   return ( hypre_BoomerAMGSetInterpType( (void *) solver, interp_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSepWeight
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetSepWeight( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int          sep_weight  )
{
   return ( hypre_BoomerAMGSetSepWeight( (void *) solver, sep_weight ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMinIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetMinIter( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int          min_iter  )
{
   return ( hypre_BoomerAMGSetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetMaxIter( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int          max_iter  )
{
   return ( hypre_BoomerAMGSetMaxIter( (void *) solver, max_iter ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetMaxIter( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int        * max_iter  )
{
   return ( hypre_BoomerAMGGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCoarsenType, NALU_HYPRE_BoomerAMGGetCoarsenType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetCoarsenType( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int          coarsen_type  )
{
   return ( hypre_BoomerAMGSetCoarsenType( (void *) solver, coarsen_type ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetCoarsenType( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int        * coarsen_type  )
{
   return ( hypre_BoomerAMGGetCoarsenType( (void *) solver, coarsen_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMeasureType, NALU_HYPRE_BoomerAMGGetMeasureType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetMeasureType( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int          measure_type  )
{
   return ( hypre_BoomerAMGSetMeasureType( (void *) solver, measure_type ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetMeasureType( NALU_HYPRE_Solver solver,
                               NALU_HYPRE_Int        * measure_type  )
{
   return ( hypre_BoomerAMGGetMeasureType( (void *) solver, measure_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetOldDefault
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetOldDefault( NALU_HYPRE_Solver solver)
{
   NALU_HYPRE_BoomerAMGSetCoarsenType( solver, 6 );
   NALU_HYPRE_BoomerAMGSetInterpType( solver, 0 );
   NALU_HYPRE_BoomerAMGSetPMaxElmts( solver, 0 );
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSetupType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetSetupType( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int          setup_type  )
{
   return ( hypre_BoomerAMGSetSetupType( (void *) solver, setup_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCycleType, NALU_HYPRE_BoomerAMGGetCycleType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetCycleType( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int          cycle_type  )
{
   return ( hypre_BoomerAMGSetCycleType( (void *) solver, cycle_type ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetCycleType( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int        * cycle_type  )
{
   return ( hypre_BoomerAMGGetCycleType( (void *) solver, cycle_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetFCycle, NALU_HYPRE_BoomerAMGGetFCycle
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetFCycle( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int    fcycle  )
{
   return ( hypre_BoomerAMGSetFCycle( (void *) solver, fcycle ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetFCycle( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int   *fcycle  )
{
   return ( hypre_BoomerAMGGetFCycle( (void *) solver, fcycle ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetConvergeType, NALU_HYPRE_BoomerAMGGetConvergeType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetConvergeType( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    type    )
{
   return ( hypre_BoomerAMGSetConvergeType( (void *) solver, type ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetConvergeType( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int   *type    )
{
   return ( hypre_BoomerAMGGetConvergeType( (void *) solver, type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetTol, NALU_HYPRE_BoomerAMGGetTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetTol( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Real   tol    )
{
   return ( hypre_BoomerAMGSetTol( (void *) solver, tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetTol( NALU_HYPRE_Solver solver,
                       NALU_HYPRE_Real * tol    )
{
   return ( hypre_BoomerAMGGetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumGridSweeps
 * DEPRECATED.  There are memory management problems associated with the
 * use of a user-supplied array (who releases it?).
 * Use SetNumSweeps and SetCycleNumSweeps instead.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetNumGridSweeps( NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Int          *num_grid_sweeps  )
{
   return ( hypre_BoomerAMGSetNumGridSweeps( (void *) solver, num_grid_sweeps ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumSweeps
 * There is no corresponding Get function.  Use GetCycleNumSweeps.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetNumSweeps( NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Int          num_sweeps  )
{
   return ( hypre_BoomerAMGSetNumSweeps( (void *) solver, num_sweeps ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCycleNumSweeps, NALU_HYPRE_BoomerAMGGetCycleNumSweeps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetCycleNumSweeps( NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Int          num_sweeps, NALU_HYPRE_Int k  )
{
   return ( hypre_BoomerAMGSetCycleNumSweeps( (void *) solver, num_sweeps, k ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetCycleNumSweeps( NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Int        * num_sweeps, NALU_HYPRE_Int k  )
{
   return ( hypre_BoomerAMGGetCycleNumSweeps( (void *) solver, num_sweeps, k ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGInitGridRelaxation
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGInitGridRelaxation( NALU_HYPRE_Int     **num_grid_sweeps_ptr,
                                   NALU_HYPRE_Int     **grid_relax_type_ptr,
                                   NALU_HYPRE_Int    ***grid_relax_points_ptr,
                                   NALU_HYPRE_Int       coarsen_type,
                                   NALU_HYPRE_Real  **relax_weights_ptr,
                                   NALU_HYPRE_Int       max_levels         )
{
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int *num_grid_sweeps;
   NALU_HYPRE_Int *grid_relax_type;
   NALU_HYPRE_Int **grid_relax_points;
   NALU_HYPRE_Real *relax_weights;

   *num_grid_sweeps_ptr   = hypre_CTAlloc(NALU_HYPRE_Int,  4, NALU_HYPRE_MEMORY_HOST);
   *grid_relax_type_ptr   = hypre_CTAlloc(NALU_HYPRE_Int,  4, NALU_HYPRE_MEMORY_HOST);
   *grid_relax_points_ptr = hypre_CTAlloc(NALU_HYPRE_Int*,  4, NALU_HYPRE_MEMORY_HOST);
   *relax_weights_ptr     = hypre_CTAlloc(NALU_HYPRE_Real,  max_levels, NALU_HYPRE_MEMORY_HOST);

   num_grid_sweeps   = *num_grid_sweeps_ptr;
   grid_relax_type   = *grid_relax_type_ptr;
   grid_relax_points = *grid_relax_points_ptr;
   relax_weights     = *relax_weights_ptr;

   if (coarsen_type == 5)
   {
      /* fine grid */
      num_grid_sweeps[0] = 3;
      grid_relax_type[0] = 3;
      grid_relax_points[0] = hypre_CTAlloc(NALU_HYPRE_Int,  4, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[0][0] = -2;
      grid_relax_points[0][1] = -1;
      grid_relax_points[0][2] = 1;

      /* down cycle */
      num_grid_sweeps[1] = 4;
      grid_relax_type[1] = 3;
      grid_relax_points[1] = hypre_CTAlloc(NALU_HYPRE_Int,  4, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[1][0] = -1;
      grid_relax_points[1][1] = 1;
      grid_relax_points[1][2] = -2;
      grid_relax_points[1][3] = -2;

      /* up cycle */
      num_grid_sweeps[2] = 4;
      grid_relax_type[2] = 3;
      grid_relax_points[2] = hypre_CTAlloc(NALU_HYPRE_Int,  4, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[2][0] = -2;
      grid_relax_points[2][1] = -2;
      grid_relax_points[2][2] = 1;
      grid_relax_points[2][3] = -1;
   }
   else
   {
      /* fine grid */
      num_grid_sweeps[0] = 2;
      grid_relax_type[0] = 3;
      grid_relax_points[0] = hypre_CTAlloc(NALU_HYPRE_Int,  2, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[0][0] = 1;
      grid_relax_points[0][1] = -1;

      /* down cycle */
      num_grid_sweeps[1] = 2;
      grid_relax_type[1] = 3;
      grid_relax_points[1] = hypre_CTAlloc(NALU_HYPRE_Int,  2, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[1][0] = 1;
      grid_relax_points[1][1] = -1;

      /* up cycle */
      num_grid_sweeps[2] = 2;
      grid_relax_type[2] = 3;
      grid_relax_points[2] = hypre_CTAlloc(NALU_HYPRE_Int,  2, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[2][0] = -1;
      grid_relax_points[2][1] = 1;
   }
   /* coarsest grid */
   num_grid_sweeps[3] = 1;
   grid_relax_type[3] = 3;
   grid_relax_points[3] = hypre_CTAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
   grid_relax_points[3][0] = 0;

   for (i = 0; i < max_levels; i++)
   {
      relax_weights[i] = 1.;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetGridRelaxType
 * DEPRECATED.  There are memory management problems associated with the
 * use of a user-supplied array (who releases it?).
 * Use SetRelaxType and SetCycleRelaxType instead.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetGridRelaxType( NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Int          *grid_relax_type  )
{
   return ( hypre_BoomerAMGSetGridRelaxType( (void *) solver, grid_relax_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetRelaxType( NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Int          relax_type  )
{
   return ( hypre_BoomerAMGSetRelaxType( (void *) solver, relax_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCycleRelaxType, NALU_HYPRE_BoomerAMGetCycleRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetCycleRelaxType( NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Int          relax_type, NALU_HYPRE_Int k  )
{
   return ( hypre_BoomerAMGSetCycleRelaxType( (void *) solver, relax_type, k ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetCycleRelaxType( NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Int        * relax_type, NALU_HYPRE_Int k  )
{
   return ( hypre_BoomerAMGGetCycleRelaxType( (void *) solver, relax_type, k ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRelaxOrder
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetRelaxOrder( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int           relax_order)
{
   return ( hypre_BoomerAMGSetRelaxOrder( (void *) solver, relax_order ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetGridRelaxPoints
 * DEPRECATED.  There are memory management problems associated with the
 * use of a user-supplied array (who releases it?).
 * Ulrike Yang suspects that nobody uses this function.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetGridRelaxPoints( NALU_HYPRE_Solver   solver,
                                   NALU_HYPRE_Int          **grid_relax_points  )
{
   return ( hypre_BoomerAMGSetGridRelaxPoints( (void *) solver, grid_relax_points ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRelaxWeight
 * DEPRECATED.  There are memory management problems associated with the
 * use of a user-supplied array (who releases it?).
 * Use SetRelaxWt and SetLevelRelaxWt instead.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetRelaxWeight( NALU_HYPRE_Solver  solver,
                               NALU_HYPRE_Real   *relax_weight  )
{
   return ( hypre_BoomerAMGSetRelaxWeight( (void *) solver, relax_weight ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRelaxWt
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetRelaxWt( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Real    relax_wt  )
{
   return ( hypre_BoomerAMGSetRelaxWt( (void *) solver, relax_wt ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLevelRelaxWt
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetLevelRelaxWt( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Real    relax_wt,
                                NALU_HYPRE_Int         level  )
{
   return ( hypre_BoomerAMGSetLevelRelaxWt( (void *) solver, relax_wt, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetOmega
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetOmega( NALU_HYPRE_Solver  solver,
                         NALU_HYPRE_Real   *omega  )
{
   return ( hypre_BoomerAMGSetOmega( (void *) solver, omega ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetOuterWt
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetOuterWt( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Real    outer_wt  )
{
   return ( hypre_BoomerAMGSetOuterWt( (void *) solver, outer_wt ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLevelOuterWt
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetLevelOuterWt( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Real    outer_wt,
                                NALU_HYPRE_Int         level  )
{
   return ( hypre_BoomerAMGSetLevelOuterWt( (void *) solver, outer_wt, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSmoothType, NALU_HYPRE_BoomerAMGGetSmoothType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetSmoothType( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int       smooth_type )
{
   return ( hypre_BoomerAMGSetSmoothType( (void *) solver, smooth_type ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetSmoothType( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int     * smooth_type )
{
   return ( hypre_BoomerAMGGetSmoothType( (void *) solver, smooth_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSmoothNumLevels, NALU_HYPRE_BoomerAMGGetSmoothNumLevels
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetSmoothNumLevels( NALU_HYPRE_Solver  solver,
                                   NALU_HYPRE_Int       smooth_num_levels  )
{
   return ( hypre_BoomerAMGSetSmoothNumLevels((void *)solver, smooth_num_levels ));
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetSmoothNumLevels( NALU_HYPRE_Solver  solver,
                                   NALU_HYPRE_Int     * smooth_num_levels  )
{
   return ( hypre_BoomerAMGGetSmoothNumLevels((void *)solver, smooth_num_levels ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSmoothNumSweeps, NALU_HYPRE_BoomerAMGGetSmoothNumSweeps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetSmoothNumSweeps( NALU_HYPRE_Solver  solver,
                                   NALU_HYPRE_Int       smooth_num_sweeps  )
{
   return ( hypre_BoomerAMGSetSmoothNumSweeps((void *)solver, smooth_num_sweeps ));
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetSmoothNumSweeps( NALU_HYPRE_Solver  solver,
                                   NALU_HYPRE_Int     * smooth_num_sweeps  )
{
   return ( hypre_BoomerAMGGetSmoothNumSweeps((void *)solver, smooth_num_sweeps ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLogging, NALU_HYPRE_BoomerAMGGetLogging
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetLogging( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int          logging  )
{
   /* This function should be called before Setup.  Logging changes
      may require allocation or freeing of arrays, which is presently
      only done there.
      It may be possible to support logging changes at other times,
      but there is little need.
   */
   return ( hypre_BoomerAMGSetLogging( (void *) solver, logging ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetLogging( NALU_HYPRE_Solver solver,
                           NALU_HYPRE_Int        * logging  )
{
   return ( hypre_BoomerAMGGetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetPrintLevel, NALU_HYPRE_BoomerAMGGetPrintLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetPrintLevel( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int        print_level  )
{
   return ( hypre_BoomerAMGSetPrintLevel( (void *) solver, print_level ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetPrintLevel( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int      * print_level  )
{
   return ( hypre_BoomerAMGGetPrintLevel( (void *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetPrintFileName
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetPrintFileName( NALU_HYPRE_Solver  solver,
                                 const char   *print_file_name  )
{
   return ( hypre_BoomerAMGSetPrintFileName( (void *) solver, print_file_name ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDebugFlag, NALU_HYPRE_BoomerAMGGetDebugFlag
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetDebugFlag( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int          debug_flag  )
{
   return ( hypre_BoomerAMGSetDebugFlag( (void *) solver, debug_flag ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetDebugFlag( NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int        * debug_flag  )
{
   return ( hypre_BoomerAMGGetDebugFlag( (void *) solver, debug_flag ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGGetNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetNumIterations( NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Int          *num_iterations  )
{
   return ( hypre_BoomerAMGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGGetCumNumIterations
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetCumNumIterations( NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int          *cum_num_iterations  )
{
   return ( hypre_BoomerAMGGetCumNumIterations( (void *) solver, cum_num_iterations ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGGetResidual
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetResidual( NALU_HYPRE_Solver solver, NALU_HYPRE_ParVector * residual )
{
   return hypre_BoomerAMGGetResidual( (void *) solver,
                                      (hypre_ParVector **) residual );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm( NALU_HYPRE_Solver  solver,
                                             NALU_HYPRE_Real   *rel_resid_norm  )
{
   return ( hypre_BoomerAMGGetRelResidualNorm( (void *) solver, rel_resid_norm ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetVariant, NALU_HYPRE_BoomerAMGGetVariant
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetVariant( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Int          variant  )
{
   return ( hypre_BoomerAMGSetVariant( (void *) solver, variant ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetVariant( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Int        * variant  )
{
   return ( hypre_BoomerAMGGetVariant( (void *) solver, variant ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetOverlap, NALU_HYPRE_BoomerAMGGetOverlap
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetOverlap( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Int          overlap  )
{
   return ( hypre_BoomerAMGSetOverlap( (void *) solver, overlap ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetOverlap( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Int        * overlap  )
{
   return ( hypre_BoomerAMGGetOverlap( (void *) solver, overlap ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDomainType, NALU_HYPRE_BoomerAMGGetDomainType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetDomainType( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int          domain_type  )
{
   return ( hypre_BoomerAMGSetDomainType( (void *) solver, domain_type ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetDomainType( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int        * domain_type  )
{
   return ( hypre_BoomerAMGGetDomainType( (void *) solver, domain_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight, NALU_HYPRE_BoomerAMGGetSchwarzRlxWeight
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight( NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Real schwarz_rlx_weight)
{
   return ( hypre_BoomerAMGSetSchwarzRlxWeight( (void *) solver,
                                                schwarz_rlx_weight ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetSchwarzRlxWeight( NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Real * schwarz_rlx_weight)
{
   return ( hypre_BoomerAMGGetSchwarzRlxWeight( (void *) solver,
                                                schwarz_rlx_weight ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm( NALU_HYPRE_Solver  solver,
                                     NALU_HYPRE_Int use_nonsymm)
{
   return ( hypre_BoomerAMGSetSchwarzUseNonSymm( (void *) solver,
                                                 use_nonsymm ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSym
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetSym( NALU_HYPRE_Solver  solver,
                       NALU_HYPRE_Int           sym)
{
   return ( hypre_BoomerAMGSetSym( (void *) solver, sym ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetLevel( NALU_HYPRE_Solver  solver,
                         NALU_HYPRE_Int           level)
{
   return ( hypre_BoomerAMGSetLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetThreshold( NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Real    threshold  )
{
   return ( hypre_BoomerAMGSetThreshold( (void *) solver, threshold ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetFilter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetFilter( NALU_HYPRE_Solver  solver,
                          NALU_HYPRE_Real    filter  )
{
   return ( hypre_BoomerAMGSetFilter( (void *) solver, filter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDropTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetDropTol( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Real    drop_tol  )
{
   return ( hypre_BoomerAMGSetDropTol( (void *) solver, drop_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMaxNzPerRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetMaxNzPerRow( NALU_HYPRE_Solver  solver,
                               NALU_HYPRE_Int          max_nz_per_row  )
{
   return ( hypre_BoomerAMGSetMaxNzPerRow( (void *) solver, max_nz_per_row ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetEuclidFile
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetEuclidFile( NALU_HYPRE_Solver  solver,
                              char         *euclidfile)
{
   return ( hypre_BoomerAMGSetEuclidFile( (void *) solver, euclidfile ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetEuLevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetEuLevel( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Int           eu_level)
{
   return ( hypre_BoomerAMGSetEuLevel( (void *) solver, eu_level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetEuSparseA
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetEuSparseA( NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Real    eu_sparse_A  )
{
   return ( hypre_BoomerAMGSetEuSparseA( (void *) solver, eu_sparse_A ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetEuBJ
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetEuBJ( NALU_HYPRE_Solver  solver,
                        NALU_HYPRE_Int         eu_bj)
{
   return ( hypre_BoomerAMGSetEuBJ( (void *) solver, eu_bj ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetILUType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetILUType( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Int         ilu_type)
{
   return ( hypre_BoomerAMGSetILUType( (void *) solver, ilu_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetILULevel
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetILULevel( NALU_HYPRE_Solver  solver,
                            NALU_HYPRE_Int         ilu_lfil)
{
   return ( hypre_BoomerAMGSetILULevel( (void *) solver, ilu_lfil ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetILUMaxRowNnz
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetILUMaxRowNnz( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int         ilu_max_row_nnz)
{
   return ( hypre_BoomerAMGSetILUMaxRowNnz( (void *) solver, ilu_max_row_nnz ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetILUMaxIter
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetILUMaxIter( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int         ilu_max_iter)
{
   return ( hypre_BoomerAMGSetILUMaxIter( (void *) solver, ilu_max_iter ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetILUDroptol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetILUDroptol( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Real        ilu_droptol)
{
   return ( hypre_BoomerAMGSetILUDroptol( (void *) solver, ilu_droptol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetILUTriSolve
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetILUTriSolve( NALU_HYPRE_Solver  solver,
                               NALU_HYPRE_Int        ilu_tri_solve)
{
   return ( hypre_BoomerAMGSetILUTriSolve( (void *) solver, ilu_tri_solve ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetILULowerJacobiIters
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetILULowerJacobiIters( NALU_HYPRE_Solver  solver,
                                       NALU_HYPRE_Int        ilu_lower_jacobi_iters)
{
   return ( hypre_BoomerAMGSetILULowerJacobiIters( (void *) solver, ilu_lower_jacobi_iters ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetILUUpperJacobiIters
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetILUUpperJacobiIters( NALU_HYPRE_Solver  solver,
                                       NALU_HYPRE_Int        ilu_upper_jacobi_iters)
{
   return ( hypre_BoomerAMGSetILUUpperJacobiIters( (void *) solver, ilu_upper_jacobi_iters ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetILULocalReordering
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetILULocalReordering( NALU_HYPRE_Solver  solver,
                                      NALU_HYPRE_Int         ilu_reordering_type)
{
   return ( hypre_BoomerAMGSetILULocalReordering( (void *) solver, ilu_reordering_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetFSAIMaxSteps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetFSAIMaxSteps( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int     max_steps  )
{
   return ( hypre_BoomerAMGSetFSAIMaxSteps( (void *) solver, max_steps ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize( NALU_HYPRE_Solver  solver,
                                   NALU_HYPRE_Int     max_step_size  )
{
   return ( hypre_BoomerAMGSetFSAIMaxStepSize( (void *) solver, max_step_size ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters( NALU_HYPRE_Solver  solver,
                                   NALU_HYPRE_Int     eig_max_iters  )
{
   return ( hypre_BoomerAMGSetFSAIEigMaxIters( (void *) solver, eig_max_iters ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetFSAIKapTolerance
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetFSAIKapTolerance( NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Real    kap_tolerance  )
{
   return ( hypre_BoomerAMGSetFSAIKapTolerance( (void *) solver, kap_tolerance ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumFunctions, NALU_HYPRE_BoomerAMGGetNumFunctions
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetNumFunctions( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int          num_functions  )
{
   return ( hypre_BoomerAMGSetNumFunctions( (void *) solver, num_functions ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetNumFunctions( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int        * num_functions  )
{
   return ( hypre_BoomerAMGGetNumFunctions( (void *) solver, num_functions ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNodal
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetNodal( NALU_HYPRE_Solver  solver,
                         NALU_HYPRE_Int          nodal  )
{
   return ( hypre_BoomerAMGSetNodal( (void *) solver, nodal ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNodalLevels
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetNodalLevels( NALU_HYPRE_Solver  solver,
                               NALU_HYPRE_Int          nodal_levels  )
{
   return ( hypre_BoomerAMGSetNodalLevels( (void *) solver, nodal_levels ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNodalDiag
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetNodalDiag( NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Int          nodal  )
{
   return ( hypre_BoomerAMGSetNodalDiag( (void *) solver, nodal ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetKeepSameSign
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetKeepSameSign( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int     keep_same_sign  )
{
   return ( hypre_BoomerAMGSetKeepSameSign( (void *) solver, keep_same_sign ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDofFunc
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetDofFunc( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Int          *dof_func  )
/* Warning about a possible memory problem: When the BoomerAMG object is destroyed
   in hypre_BoomerAMGDestroy, dof_func aka DofFunc will be destroyed (currently
   line 246 of par_amg.c).  Normally this is what we want.  But if the user provided
   dof_func by calling NALU_HYPRE_BoomerAMGSetDofFunc, this could be an unwanted surprise.
   As hypre is currently commonly used, this situation is likely to be rare. */
{
   return ( hypre_BoomerAMGSetDofFunc( (void *) solver, dof_func ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumPaths
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetNumPaths( NALU_HYPRE_Solver  solver,
                            NALU_HYPRE_Int          num_paths  )
{
   return ( hypre_BoomerAMGSetNumPaths( (void *) solver, num_paths ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggNumLevels
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetAggNumLevels( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int          agg_num_levels  )
{
   return ( hypre_BoomerAMGSetAggNumLevels( (void *) solver, agg_num_levels ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggInterpType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetAggInterpType( NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Int          agg_interp_type  )
{
   return ( hypre_BoomerAMGSetAggInterpType( (void *) solver, agg_interp_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggTruncFactor
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetAggTruncFactor( NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Real    agg_trunc_factor  )
{
   return ( hypre_BoomerAMGSetAggTruncFactor( (void *) solver, agg_trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAddTruncFactor
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetAddTruncFactor( NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Real        add_trunc_factor  )
{
   return ( hypre_BoomerAMGSetMultAddTruncFactor( (void *) solver, add_trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMultAddTruncFactor
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetMultAddTruncFactor( NALU_HYPRE_Solver  solver,
                                      NALU_HYPRE_Real        add_trunc_factor  )
{
   return ( hypre_BoomerAMGSetMultAddTruncFactor( (void *) solver, add_trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAddRelaxWt
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetAddRelaxWt( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Real        add_rlx_wt  )
{
   return ( hypre_BoomerAMGSetAddRelaxWt( (void *) solver, add_rlx_wt ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAddRelaxType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetAddRelaxType( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int        add_rlx_type  )
{
   return ( hypre_BoomerAMGSetAddRelaxType( (void *) solver, add_rlx_type ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggP12TruncFactor
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetAggP12TruncFactor( NALU_HYPRE_Solver  solver,
                                     NALU_HYPRE_Real    agg_P12_trunc_factor  )
{
   return ( hypre_BoomerAMGSetAggP12TruncFactor( (void *) solver, agg_P12_trunc_factor ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggPMaxElmts
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetAggPMaxElmts( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int          agg_P_max_elmts  )
{
   return ( hypre_BoomerAMGSetAggPMaxElmts( (void *) solver, agg_P_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAddPMaxElmts
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetAddPMaxElmts( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int          add_P_max_elmts  )
{
   return ( hypre_BoomerAMGSetMultAddPMaxElmts( (void *) solver, add_P_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts( NALU_HYPRE_Solver  solver,
                                    NALU_HYPRE_Int          add_P_max_elmts  )
{
   return ( hypre_BoomerAMGSetMultAddPMaxElmts( (void *) solver, add_P_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAggP12MaxElmts
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetAggP12MaxElmts( NALU_HYPRE_Solver  solver,
                                  NALU_HYPRE_Int          agg_P12_max_elmts  )
{
   return ( hypre_BoomerAMGSetAggP12MaxElmts( (void *) solver, agg_P12_max_elmts ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps( NALU_HYPRE_Solver  solver,
                                   NALU_HYPRE_Int          num_CR_relax_steps  )
{
   return ( hypre_BoomerAMGSetNumCRRelaxSteps( (void *) solver, num_CR_relax_steps ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCRRate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetCRRate( NALU_HYPRE_Solver  solver,
                          NALU_HYPRE_Real    CR_rate  )
{
   return ( hypre_BoomerAMGSetCRRate( (void *) solver, CR_rate ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCRStrongTh
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetCRStrongTh( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Real    CR_strong_th  )
{
   return ( hypre_BoomerAMGSetCRStrongTh( (void *) solver, CR_strong_th ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetADropTol( NALU_HYPRE_Solver  solver,
                            NALU_HYPRE_Real    A_drop_tol  )
{
   return ( hypre_BoomerAMGSetADropTol( (void *) solver, A_drop_tol ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetADropType( NALU_HYPRE_Solver  solver,
                             NALU_HYPRE_Int     A_drop_type  )
{
   return ( hypre_BoomerAMGSetADropType( (void *) solver, A_drop_type ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetISType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetISType( NALU_HYPRE_Solver  solver,
                          NALU_HYPRE_Int          IS_type  )
{
   return ( hypre_BoomerAMGSetISType( (void *) solver, IS_type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCRUseCG
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetCRUseCG( NALU_HYPRE_Solver  solver,
                           NALU_HYPRE_Int    CR_use_CG  )
{
   return ( hypre_BoomerAMGSetCRUseCG( (void *) solver, CR_use_CG ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetGSMG
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetGSMG( NALU_HYPRE_Solver  solver,
                        NALU_HYPRE_Int        gsmg  )
{
   return ( hypre_BoomerAMGSetGSMG( (void *) solver, gsmg ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNumSamples
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetNumSamples( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int        gsmg  )
{
   return ( hypre_BoomerAMGSetNumSamples( (void *) solver, gsmg ) );
}
/* BM Aug 25, 2006 */

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCGCIts
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetCGCIts (NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int its)
{
   return (hypre_BoomerAMGSetCGCIts ( (void *) solver, its ) );
}

/* BM Oct 23, 2006 */
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetPlotGrids
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetPlotGrids (NALU_HYPRE_Solver solver,
                             NALU_HYPRE_Int plotgrids)
{
   return (hypre_BoomerAMGSetPlotGrids ( (void *) solver, plotgrids ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetPlotFileName
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetPlotFileName (NALU_HYPRE_Solver solver,
                                const char *plotfilename)
{
   return (hypre_BoomerAMGSetPlotFileName ( (void *) solver, plotfilename ) );
}

/* BM Oct 17, 2006 */

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCoordDim
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetCoordDim (NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int coorddim)
{
   return (hypre_BoomerAMGSetCoordDim ( (void *) solver, coorddim ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCoordinates
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetCoordinates (NALU_HYPRE_Solver solver,
                               float *coordinates)
{
   return (hypre_BoomerAMGSetCoordinates ( (void *) solver, coordinates ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGGetGridHierarchy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetGridHierarchy(NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int *cgrid )
{
   return (hypre_BoomerAMGGetGridHierarchy ( (void *) solver, cgrid ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyOrder
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetChebyOrder( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int        order )
{
   return ( hypre_BoomerAMGSetChebyOrder( (void *) solver, order ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyFraction
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetChebyFraction( NALU_HYPRE_Solver  solver,
                                 NALU_HYPRE_Real     ratio )
{
   return ( hypre_BoomerAMGSetChebyFraction( (void *) solver, ratio ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyScale
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetChebyScale( NALU_HYPRE_Solver  solver,
                              NALU_HYPRE_Int     scale )
{
   return ( hypre_BoomerAMGSetChebyScale( (void *) solver, scale ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyVariant
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetChebyVariant( NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int     variant )
{
   return ( hypre_BoomerAMGSetChebyVariant( (void *) solver, variant ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetChebyEigEst
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetChebyEigEst( NALU_HYPRE_Solver  solver,
                               NALU_HYPRE_Int     eig_est )
{
   return ( hypre_BoomerAMGSetChebyEigEst( (void *) solver, eig_est ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpVectors
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetInterpVectors (NALU_HYPRE_Solver solver, NALU_HYPRE_Int num_vectors,
                                 NALU_HYPRE_ParVector *vectors)
{
   return (hypre_BoomerAMGSetInterpVectors ( (void *) solver,
                                             num_vectors,
                                             (hypre_ParVector **) vectors ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpVecVariant
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetInterpVecVariant(NALU_HYPRE_Solver solver, NALU_HYPRE_Int num)

{
   return (hypre_BoomerAMGSetInterpVecVariant ( (void *) solver, num ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpVecQMax
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetInterpVecQMax( NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int       q_max  )
{
   return ( hypre_BoomerAMGSetInterpVecQMax( (void *) solver,
                                             q_max ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpVecAbsQTrunc
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetInterpVecAbsQTrunc( NALU_HYPRE_Solver solver,
                                      NALU_HYPRE_Real   q_trunc  )
{
   return ( hypre_BoomerAMGSetInterpVecAbsQTrunc( (void *) solver,
                                                  q_trunc ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSmoothInterpVectors
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetSmoothInterpVectors( NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    smooth_interp_vectors  )
{
   return ( hypre_BoomerAMGSetSmoothInterpVectors( (void *) solver,
                                                   smooth_interp_vectors) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpRefine
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetInterpRefine( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    num_refine  )
{
   return ( hypre_BoomerAMGSetInterpRefine( (void *) solver,
                                            num_refine ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetInterpVecFirstLevel(
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetInterpVecFirstLevel( NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Int    level  )
{
   return ( hypre_BoomerAMGSetInterpVecFirstLevel( (void *) solver,
                                                   level ) );
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAdditive, NALU_HYPRE_BoomerAMGGetAdditive
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetAdditive( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int    additive  )
{
   return ( hypre_BoomerAMGSetAdditive( (void *) solver, additive ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetAdditive( NALU_HYPRE_Solver solver,
                            NALU_HYPRE_Int  * additive  )
{
   return ( hypre_BoomerAMGGetAdditive( (void *) solver, additive ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetMultAdditive, NALU_HYPRE_BoomerAMGGetMultAdditive
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetMultAdditive( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    mult_additive  )
{
   return ( hypre_BoomerAMGSetMultAdditive( (void *) solver, mult_additive ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetMultAdditive( NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int   *mult_additive  )
{
   return ( hypre_BoomerAMGGetMultAdditive( (void *) solver, mult_additive ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetSimple, NALU_HYPRE_BoomerAMGGetSimple
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetSimple( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int    simple  )
{
   return ( hypre_BoomerAMGSetSimple( (void *) solver, simple ) );
}

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetSimple( NALU_HYPRE_Solver solver,
                          NALU_HYPRE_Int   *simple  )
{
   return ( hypre_BoomerAMGGetSimple( (void *) solver, simple ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetAddLastLvl
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetAddLastLvl( NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    add_last_lvl  )
{
   return ( hypre_BoomerAMGSetAddLastLvl( (void *) solver, add_last_lvl ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNonGalerkinTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetNonGalerkinTol (NALU_HYPRE_Solver solver,
                                  NALU_HYPRE_Real   nongalerkin_tol)
{
   return (hypre_BoomerAMGSetNonGalerkinTol ( (void *) solver, nongalerkin_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol (NALU_HYPRE_Solver solver,
                                       NALU_HYPRE_Real   nongalerkin_tol,
                                       NALU_HYPRE_Int    level)
{
   return (hypre_BoomerAMGSetLevelNonGalerkinTol ( (void *) solver, nongalerkin_tol, level ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetNonGalerkTol
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetNonGalerkTol (NALU_HYPRE_Solver solver,
                                NALU_HYPRE_Int    nongalerk_num_tol,
                                NALU_HYPRE_Real  *nongalerk_tol)
{
   return (hypre_BoomerAMGSetNonGalerkTol ( (void *) solver, nongalerk_num_tol, nongalerk_tol ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetRAP2
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetRAP2 (NALU_HYPRE_Solver solver,
                        NALU_HYPRE_Int    rap2)
{
   return (hypre_BoomerAMGSetRAP2 ( (void *) solver, rap2 ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetModuleRAP2
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetModuleRAP2 (NALU_HYPRE_Solver solver,
                              NALU_HYPRE_Int    mod_rap2)
{
   return (hypre_BoomerAMGSetModuleRAP2 ( (void *) solver, mod_rap2 ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetKeepTranspose
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetKeepTranspose (NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    keepTranspose)
{
   return (hypre_BoomerAMGSetKeepTranspose ( (void *) solver, keepTranspose ) );
}

#ifdef NALU_HYPRE_USING_DSUPERLU
/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetDSLUThreshold
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetDSLUThreshold (NALU_HYPRE_Solver solver,
                                 NALU_HYPRE_Int    slu_threshold)
{
   return (hypre_BoomerAMGSetDSLUThreshold ( (void *) solver, slu_threshold ) );
}
#endif

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCpointsToKeep
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetCpointsToKeep(NALU_HYPRE_Solver  solver,
                                NALU_HYPRE_Int     cpt_coarse_level,
                                NALU_HYPRE_Int     num_cpt_coarse,
                                NALU_HYPRE_BigInt *cpt_coarse_index)
{
   return (hypre_BoomerAMGSetCPoints( (void *) solver, cpt_coarse_level, num_cpt_coarse,
                                      cpt_coarse_index));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCPoints
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetCPoints(NALU_HYPRE_Solver  solver,
                          NALU_HYPRE_Int     cpt_coarse_level,
                          NALU_HYPRE_Int     num_cpt_coarse,
                          NALU_HYPRE_BigInt *cpt_coarse_index)
{
   return (hypre_BoomerAMGSetCPoints( (void *) solver, cpt_coarse_level, num_cpt_coarse,
                                      cpt_coarse_index));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetFPoints
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetFPoints(NALU_HYPRE_Solver   solver,
                          NALU_HYPRE_Int      num_fpt,
                          NALU_HYPRE_BigInt  *fpt_index)
{
   return (hypre_BoomerAMGSetFPoints( (void *) solver,
                                      0, num_fpt,
                                      fpt_index) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetIsolatedFPoints
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetIsolatedFPoints(NALU_HYPRE_Solver   solver,
                                  NALU_HYPRE_Int      num_isolated_fpt,
                                  NALU_HYPRE_BigInt  *isolated_fpt_index)
{
   return (hypre_BoomerAMGSetFPoints( (void *) solver,
                                      1, num_isolated_fpt,
                                      isolated_fpt_index) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGSetCumNnzAP
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGSetCumNnzAP( NALU_HYPRE_Solver  solver,
                            NALU_HYPRE_Real    cum_nnz_AP )
{
   return( hypre_BoomerAMGSetCumNnzAP( (void *) solver, cum_nnz_AP ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_BoomerAMGGetCumNnzAP
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_BoomerAMGGetCumNnzAP( NALU_HYPRE_Solver  solver,
                            NALU_HYPRE_Real   *cum_nnz_AP )
{
   return( hypre_BoomerAMGGetCumNnzAP( (void *) solver, cum_nnz_AP ) );
}

