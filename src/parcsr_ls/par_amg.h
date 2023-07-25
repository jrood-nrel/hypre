/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_ParAMG_DATA_HEADER
#define nalu_hypre_ParAMG_DATA_HEADER

#define CUMNUMIT

#include "par_csr_block_matrix.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ParAMGData
 *--------------------------------------------------------------------------*/

typedef struct
{
   NALU_HYPRE_MemoryLocation  memory_location;   /* memory location of matrices/vectors in AMGData */

   /* setup params */
   NALU_HYPRE_Int      max_levels;
   NALU_HYPRE_Real     strong_threshold;
   NALU_HYPRE_Int      coarsen_cut_factor;
   NALU_HYPRE_Real     strong_thresholdR; /* theta for build R: defines strong F neighbors */
   NALU_HYPRE_Real     filter_thresholdR; /* theta for filtering R  */
   NALU_HYPRE_Real     max_row_sum;
   NALU_HYPRE_Real     trunc_factor;
   NALU_HYPRE_Real     agg_trunc_factor;
   NALU_HYPRE_Real     agg_P12_trunc_factor;
   NALU_HYPRE_Real     jacobi_trunc_threshold;
   NALU_HYPRE_Real     S_commpkg_switch;
   NALU_HYPRE_Real     CR_rate;
   NALU_HYPRE_Real     CR_strong_th;
   NALU_HYPRE_Real     A_drop_tol;
   NALU_HYPRE_Int      A_drop_type;
   NALU_HYPRE_Int      measure_type;
   NALU_HYPRE_Int      setup_type;
   NALU_HYPRE_Int      coarsen_type;
   NALU_HYPRE_Int      P_max_elmts;
   NALU_HYPRE_Int      interp_type;
   NALU_HYPRE_Int      sep_weight;
   NALU_HYPRE_Int      agg_interp_type;
   NALU_HYPRE_Int      agg_P_max_elmts;
   NALU_HYPRE_Int      agg_P12_max_elmts;
   NALU_HYPRE_Int      restr_par;
   NALU_HYPRE_Int      is_triangular;
   NALU_HYPRE_Int      gmres_switch;
   NALU_HYPRE_Int      agg_num_levels;
   NALU_HYPRE_Int      num_paths;
   NALU_HYPRE_Int      post_interp_type;
   NALU_HYPRE_Int      num_CR_relax_steps;
   NALU_HYPRE_Int      IS_type;
   NALU_HYPRE_Int      CR_use_CG;
   NALU_HYPRE_Int      cgc_its;
   NALU_HYPRE_Int      max_coarse_size;
   NALU_HYPRE_Int      min_coarse_size;
   NALU_HYPRE_Int      seq_threshold;
   NALU_HYPRE_Int      redundant;
   NALU_HYPRE_Int      participate;
   NALU_HYPRE_Int      Sabs;

   /* solve params */
   NALU_HYPRE_Int      max_iter;
   NALU_HYPRE_Int      min_iter;
   NALU_HYPRE_Int      fcycle;
   NALU_HYPRE_Int      cycle_type;
   NALU_HYPRE_Int     *num_grid_sweeps;
   NALU_HYPRE_Int     *grid_relax_type;
   NALU_HYPRE_Int    **grid_relax_points;
   NALU_HYPRE_Int      relax_order;
   NALU_HYPRE_Int      user_coarse_relax_type;
   NALU_HYPRE_Int      user_relax_type;
   NALU_HYPRE_Int      user_num_sweeps;
   NALU_HYPRE_Real     user_relax_weight;
   NALU_HYPRE_Real     outer_wt;
   NALU_HYPRE_Real    *relax_weight;
   NALU_HYPRE_Real    *omega;
   NALU_HYPRE_Int      converge_type;
   NALU_HYPRE_Real     tol;
   NALU_HYPRE_Int      partial_cycle_coarsest_level;
   NALU_HYPRE_Int      partial_cycle_control;


   /* problem data */
   nalu_hypre_ParCSRMatrix  *A;
   NALU_HYPRE_Int            num_variables;
   NALU_HYPRE_Int            num_functions;
   NALU_HYPRE_Int            nodal;
   NALU_HYPRE_Int            nodal_levels;
   NALU_HYPRE_Int            nodal_diag;
   NALU_HYPRE_Int            keep_same_sign;
   NALU_HYPRE_Int            num_points;
   nalu_hypre_IntArray      *dof_func;
   NALU_HYPRE_Int           *dof_point;
   NALU_HYPRE_Int           *point_dof_map;

   /* data generated in the setup phase */
   nalu_hypre_ParCSRMatrix **A_array;
   nalu_hypre_ParVector    **F_array;
   nalu_hypre_ParVector    **U_array;
   nalu_hypre_ParCSRMatrix **P_array;
   nalu_hypre_ParCSRMatrix **R_array;
   nalu_hypre_IntArray     **CF_marker_array;
   nalu_hypre_IntArray     **dof_func_array;
   NALU_HYPRE_Int          **dof_point_array;
   NALU_HYPRE_Int          **point_dof_map_array;
   NALU_HYPRE_Int            num_levels;
   nalu_hypre_Vector       **l1_norms;

   /* Block data */
   nalu_hypre_ParCSRBlockMatrix **A_block_array;
   nalu_hypre_ParCSRBlockMatrix **P_block_array;
   nalu_hypre_ParCSRBlockMatrix **R_block_array;

   NALU_HYPRE_Int block_mode;

   /* data for more complex smoothers */
   NALU_HYPRE_Int            smooth_num_levels;
   NALU_HYPRE_Int            smooth_type;
   NALU_HYPRE_Solver        *smoother;
   NALU_HYPRE_Int            smooth_num_sweeps;
   NALU_HYPRE_Int            schw_variant;
   NALU_HYPRE_Int            schw_overlap;
   NALU_HYPRE_Int            schw_domain_type;
   NALU_HYPRE_Real           schwarz_rlx_weight;
   NALU_HYPRE_Int            schwarz_use_nonsymm;
   NALU_HYPRE_Int            ps_sym;
   NALU_HYPRE_Int            ps_level;
   NALU_HYPRE_Int            pi_max_nz_per_row;
   NALU_HYPRE_Int            eu_level;
   NALU_HYPRE_Int            eu_bj;
   NALU_HYPRE_Real           ps_threshold;
   NALU_HYPRE_Real           ps_filter;
   NALU_HYPRE_Real           pi_drop_tol;
   NALU_HYPRE_Real           eu_sparse_A;
   char                *euclidfile;
   NALU_HYPRE_Int            ilu_lfil;
   NALU_HYPRE_Int            ilu_type;
   NALU_HYPRE_Int            ilu_max_row_nnz;
   NALU_HYPRE_Int            ilu_max_iter;
   NALU_HYPRE_Real           ilu_droptol;
   NALU_HYPRE_Int            ilu_tri_solve;
   NALU_HYPRE_Int            ilu_lower_jacobi_iters;
   NALU_HYPRE_Int            ilu_upper_jacobi_iters;
   NALU_HYPRE_Int            ilu_reordering_type;

   NALU_HYPRE_Int            fsai_algo_type;
   NALU_HYPRE_Int            fsai_local_solve_type;
   NALU_HYPRE_Int            fsai_max_steps;
   NALU_HYPRE_Int            fsai_max_step_size;
   NALU_HYPRE_Int            fsai_max_nnz_row;
   NALU_HYPRE_Int            fsai_num_levels;
   NALU_HYPRE_Real           fsai_threshold;
   NALU_HYPRE_Int            fsai_eig_max_iters;
   NALU_HYPRE_Real           fsai_kap_tolerance;

   NALU_HYPRE_Real          *max_eig_est;
   NALU_HYPRE_Real          *min_eig_est;
   NALU_HYPRE_Int            cheby_eig_est;
   NALU_HYPRE_Int            cheby_order;
   NALU_HYPRE_Int            cheby_variant;
   NALU_HYPRE_Int            cheby_scale;
   NALU_HYPRE_Real           cheby_fraction;
   nalu_hypre_Vector       **cheby_ds;
   NALU_HYPRE_Real         **cheby_coefs;

   NALU_HYPRE_Real           cum_nnz_AP;

   /* data needed for non-Galerkin option */
   NALU_HYPRE_Int           nongalerk_num_tol;
   NALU_HYPRE_Real         *nongalerk_tol;
   NALU_HYPRE_Real          nongalerkin_tol;
   NALU_HYPRE_Real         *nongal_tol_array;

   /* data generated in the solve phase */
   nalu_hypre_ParVector   *Vtemp;
   nalu_hypre_Vector      *Vtemp_local;
   NALU_HYPRE_Real        *Vtemp_local_data;
   NALU_HYPRE_Real         cycle_op_count;
   nalu_hypre_ParVector   *Rtemp;
   nalu_hypre_ParVector   *Ptemp;
   nalu_hypre_ParVector   *Ztemp;

   /* fields used by GSMG and LS interpolation */
   NALU_HYPRE_Int          gsmg;        /* nonzero indicates use of GSMG */
   NALU_HYPRE_Int          num_samples; /* number of sample vectors */

   /* log info */
   NALU_HYPRE_Int        logging;
   NALU_HYPRE_Int        num_iterations;
#ifdef CUMNUMIT
   NALU_HYPRE_Int        cum_num_iterations;
#endif
   NALU_HYPRE_Real       rel_resid_norm;
   nalu_hypre_ParVector *residual; /* available if logging>1 */

   /* output params */
   NALU_HYPRE_Int      print_level;
   char           log_file_name[256];
   NALU_HYPRE_Int      debug_flag;

   /* whether to print the constructed coarse grids BM Oct 22, 2006 */
   NALU_HYPRE_Int      plot_grids;
   char           plot_filename[251];

   /* coordinate data BM Oct 17, 2006 */
   NALU_HYPRE_Int      coorddim;
   float         *coordinates;

   /* data for fitting vectors in interpolation */
   NALU_HYPRE_Int          num_interp_vectors;
   NALU_HYPRE_Int          num_levels_interp_vectors; /* not set by user */
   nalu_hypre_ParVector  **interp_vectors;
   nalu_hypre_ParVector ***interp_vectors_array;
   NALU_HYPRE_Int          interp_vec_variant;
   NALU_HYPRE_Int          interp_vec_first_level;
   NALU_HYPRE_Real         interp_vectors_abs_q_trunc;
   NALU_HYPRE_Int          interp_vectors_q_max;
   NALU_HYPRE_Int          interp_refine;
   NALU_HYPRE_Int          smooth_interp_vectors;
   NALU_HYPRE_Real       *expandp_weights; /* currently not set by user */

   /* enable redundant coarse grid solve */
   NALU_HYPRE_Solver         coarse_solver;
   nalu_hypre_ParCSRMatrix  *A_coarse;
   nalu_hypre_ParVector     *f_coarse;
   nalu_hypre_ParVector     *u_coarse;
   MPI_Comm             new_comm;

   /* store matrix, vector and communication info for Gaussian elimination */
   NALU_HYPRE_Int   gs_setup;
   NALU_HYPRE_Real *A_mat, *A_inv;
   NALU_HYPRE_Real *b_vec;
   NALU_HYPRE_Int  *comm_info;

   /* information for multiplication with Lambda - additive AMG */
   NALU_HYPRE_Int      additive;
   NALU_HYPRE_Int      mult_additive;
   NALU_HYPRE_Int      simple;
   NALU_HYPRE_Int      add_last_lvl;
   NALU_HYPRE_Int      add_P_max_elmts;
   NALU_HYPRE_Real     add_trunc_factor;
   NALU_HYPRE_Int      add_rlx_type;
   NALU_HYPRE_Real     add_rlx_wt;
   nalu_hypre_ParCSRMatrix *Lambda;
   nalu_hypre_ParCSRMatrix *Atilde;
   nalu_hypre_ParVector *Rtilde;
   nalu_hypre_ParVector *Xtilde;
   NALU_HYPRE_Real *D_inv;

   /* Use 2 mat-mat-muls instead of triple product*/
   NALU_HYPRE_Int rap2;
   NALU_HYPRE_Int keepTranspose;
   NALU_HYPRE_Int modularized_matmat;

   /* information for preserving indices as coarse grid points */
   NALU_HYPRE_Int      num_C_points;
   NALU_HYPRE_Int      C_points_coarse_level;
   NALU_HYPRE_Int     *C_points_local_marker;
   NALU_HYPRE_BigInt  *C_points_marker;

   /* information for preserving indices as special fine grid points */
   NALU_HYPRE_Int      num_isolated_F_points;
   NALU_HYPRE_BigInt  *isolated_F_points_marker;

   /* information for preserving indices as fine grid points */
   NALU_HYPRE_Int      num_F_points;
   NALU_HYPRE_BigInt  *F_points_marker;

#ifdef NALU_HYPRE_USING_DSUPERLU
   /* Parameters and data for SuperLU_Dist */
   NALU_HYPRE_Int dslu_threshold;
   NALU_HYPRE_Solver dslu_solver;
#endif

} nalu_hypre_ParAMGData;

/*--------------------------------------------------------------------------
 * Accessor functions for the nalu_hypre_AMGData structure
 *--------------------------------------------------------------------------*/

/* setup params */

#define nalu_hypre_ParAMGDataMemoryLocation(amg_data)       ((amg_data) -> memory_location)
#define nalu_hypre_ParAMGDataRestriction(amg_data)          ((amg_data) -> restr_par)
#define nalu_hypre_ParAMGDataIsTriangular(amg_data)         ((amg_data) -> is_triangular)
#define nalu_hypre_ParAMGDataGMRESSwitchR(amg_data)         ((amg_data) -> gmres_switch)
#define nalu_hypre_ParAMGDataMaxLevels(amg_data)            ((amg_data) -> max_levels)
#define nalu_hypre_ParAMGDataCoarsenCutFactor(amg_data)     ((amg_data) -> coarsen_cut_factor)
#define nalu_hypre_ParAMGDataStrongThreshold(amg_data)      ((amg_data) -> strong_threshold)
#define nalu_hypre_ParAMGDataStrongThresholdR(amg_data)     ((amg_data) -> strong_thresholdR)
#define nalu_hypre_ParAMGDataFilterThresholdR(amg_data)     ((amg_data) -> filter_thresholdR)
#define nalu_hypre_ParAMGDataSabs(amg_data)                 ((amg_data) -> Sabs)
#define nalu_hypre_ParAMGDataMaxRowSum(amg_data)            ((amg_data) -> max_row_sum)
#define nalu_hypre_ParAMGDataTruncFactor(amg_data)          ((amg_data) -> trunc_factor)
#define nalu_hypre_ParAMGDataAggTruncFactor(amg_data)       ((amg_data) -> agg_trunc_factor)
#define nalu_hypre_ParAMGDataAggP12TruncFactor(amg_data)    ((amg_data) -> agg_P12_trunc_factor)
#define nalu_hypre_ParAMGDataJacobiTruncThreshold(amg_data) ((amg_data) -> jacobi_trunc_threshold)
#define nalu_hypre_ParAMGDataSCommPkgSwitch(amg_data)       ((amg_data) -> S_commpkg_switch)
#define nalu_hypre_ParAMGDataInterpType(amg_data)           ((amg_data) -> interp_type)
#define nalu_hypre_ParAMGDataSepWeight(amg_data)            ((amg_data) -> sep_weight)
#define nalu_hypre_ParAMGDataAggInterpType(amg_data)        ((amg_data) -> agg_interp_type)
#define nalu_hypre_ParAMGDataCoarsenType(amg_data)          ((amg_data) -> coarsen_type)
#define nalu_hypre_ParAMGDataMeasureType(amg_data)          ((amg_data) -> measure_type)
#define nalu_hypre_ParAMGDataSetupType(amg_data)            ((amg_data) -> setup_type)
#define nalu_hypre_ParAMGDataPMaxElmts(amg_data)            ((amg_data) -> P_max_elmts)
#define nalu_hypre_ParAMGDataAggPMaxElmts(amg_data)         ((amg_data) -> agg_P_max_elmts)
#define nalu_hypre_ParAMGDataAggP12MaxElmts(amg_data)       ((amg_data) -> agg_P12_max_elmts)
#define nalu_hypre_ParAMGDataNumPaths(amg_data)             ((amg_data) -> num_paths)
#define nalu_hypre_ParAMGDataAggNumLevels(amg_data)         ((amg_data) -> agg_num_levels)
#define nalu_hypre_ParAMGDataPostInterpType(amg_data)       ((amg_data) -> post_interp_type)
#define nalu_hypre_ParAMGDataNumCRRelaxSteps(amg_data)      ((amg_data) -> num_CR_relax_steps)
#define nalu_hypre_ParAMGDataCRRate(amg_data)               ((amg_data) -> CR_rate)
#define nalu_hypre_ParAMGDataCRStrongTh(amg_data)           ((amg_data) -> CR_strong_th)
#define nalu_hypre_ParAMGDataADropTol(amg_data)             ((amg_data) -> A_drop_tol)
#define nalu_hypre_ParAMGDataADropType(amg_data)            ((amg_data) -> A_drop_type)
#define nalu_hypre_ParAMGDataISType(amg_data)               ((amg_data) -> IS_type)
#define nalu_hypre_ParAMGDataCRUseCG(amg_data)              ((amg_data) -> CR_use_CG)
#define nalu_hypre_ParAMGDataL1Norms(amg_data)              ((amg_data) -> l1_norms)
#define nalu_hypre_ParAMGDataCGCIts(amg_data)               ((amg_data) -> cgc_its)
#define nalu_hypre_ParAMGDataMaxCoarseSize(amg_data)        ((amg_data) -> max_coarse_size)
#define nalu_hypre_ParAMGDataMinCoarseSize(amg_data)        ((amg_data) -> min_coarse_size)
#define nalu_hypre_ParAMGDataSeqThreshold(amg_data)         ((amg_data) -> seq_threshold)

/* solve params */

#define nalu_hypre_ParAMGDataMinIter(amg_data) ((amg_data)->min_iter)
#define nalu_hypre_ParAMGDataMaxIter(amg_data) ((amg_data)->max_iter)
#define nalu_hypre_ParAMGDataFCycle(amg_data) ((amg_data)->fcycle)
#define nalu_hypre_ParAMGDataCycleType(amg_data) ((amg_data)->cycle_type)
#define nalu_hypre_ParAMGDataConvergeType(amg_data) ((amg_data)->converge_type)
#define nalu_hypre_ParAMGDataTol(amg_data) ((amg_data)->tol)
#define nalu_hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data) ((amg_data)->partial_cycle_coarsest_level)
#define nalu_hypre_ParAMGDataPartialCycleControl(amg_data) ((amg_data)->partial_cycle_control)
#define nalu_hypre_ParAMGDataNumGridSweeps(amg_data) ((amg_data)->num_grid_sweeps)
#define nalu_hypre_ParAMGDataUserCoarseRelaxType(amg_data) ((amg_data)->user_coarse_relax_type)
#define nalu_hypre_ParAMGDataUserRelaxType(amg_data) ((amg_data)->user_relax_type)
#define nalu_hypre_ParAMGDataUserRelaxWeight(amg_data) ((amg_data)->user_relax_weight)
#define nalu_hypre_ParAMGDataUserNumSweeps(amg_data) ((amg_data)->user_num_sweeps)
#define nalu_hypre_ParAMGDataGridRelaxType(amg_data) ((amg_data)->grid_relax_type)
#define nalu_hypre_ParAMGDataGridRelaxPoints(amg_data) ((amg_data)->grid_relax_points)
#define nalu_hypre_ParAMGDataRelaxOrder(amg_data) ((amg_data)->relax_order)
#define nalu_hypre_ParAMGDataRelaxWeight(amg_data) ((amg_data)->relax_weight)
#define nalu_hypre_ParAMGDataOmega(amg_data) ((amg_data)->omega)
#define nalu_hypre_ParAMGDataOuterWt(amg_data) ((amg_data)->outer_wt)

/* problem data parameters */
#define  nalu_hypre_ParAMGDataNumVariables(amg_data)  ((amg_data)->num_variables)
#define nalu_hypre_ParAMGDataNumFunctions(amg_data) ((amg_data)->num_functions)
#define nalu_hypre_ParAMGDataNodal(amg_data) ((amg_data)->nodal)
#define nalu_hypre_ParAMGDataNodalLevels(amg_data) ((amg_data)->nodal_levels)
#define nalu_hypre_ParAMGDataNodalDiag(amg_data) ((amg_data)->nodal_diag)
#define nalu_hypre_ParAMGDataKeepSameSign(amg_data) ((amg_data)->keep_same_sign)
#define nalu_hypre_ParAMGDataNumPoints(amg_data) ((amg_data)->num_points)
#define nalu_hypre_ParAMGDataDofFunc(amg_data) ((amg_data)->dof_func)
#define nalu_hypre_ParAMGDataDofPoint(amg_data) ((amg_data)->dof_point)
#define nalu_hypre_ParAMGDataPointDofMap(amg_data) ((amg_data)->point_dof_map)

/* data generated by the setup phase */
#define nalu_hypre_ParAMGDataCFMarkerArray(amg_data) ((amg_data)-> CF_marker_array)
#define nalu_hypre_ParAMGDataAArray(amg_data) ((amg_data)->A_array)
#define nalu_hypre_ParAMGDataFArray(amg_data) ((amg_data)->F_array)
#define nalu_hypre_ParAMGDataUArray(amg_data) ((amg_data)->U_array)
#define nalu_hypre_ParAMGDataPArray(amg_data) ((amg_data)->P_array)
#define nalu_hypre_ParAMGDataRArray(amg_data) ((amg_data)->R_array)
#define nalu_hypre_ParAMGDataDofFuncArray(amg_data) ((amg_data)->dof_func_array)
#define nalu_hypre_ParAMGDataDofPointArray(amg_data) ((amg_data)->dof_point_array)
#define nalu_hypre_ParAMGDataPointDofMapArray(amg_data) \
((amg_data)->point_dof_map_array)
#define nalu_hypre_ParAMGDataNumLevels(amg_data) ((amg_data)->num_levels)
#define nalu_hypre_ParAMGDataSmoothType(amg_data) ((amg_data)->smooth_type)
#define nalu_hypre_ParAMGDataSmoothNumLevels(amg_data) \
((amg_data)->smooth_num_levels)
#define nalu_hypre_ParAMGDataSmoothNumSweeps(amg_data) \
((amg_data)->smooth_num_sweeps)
#define nalu_hypre_ParAMGDataSmoother(amg_data) ((amg_data)->smoother)
#define nalu_hypre_ParAMGDataVariant(amg_data) ((amg_data)->schw_variant)
#define nalu_hypre_ParAMGDataOverlap(amg_data) ((amg_data)->schw_overlap)
#define nalu_hypre_ParAMGDataDomainType(amg_data) ((amg_data)->schw_domain_type)
#define nalu_hypre_ParAMGDataSchwarzRlxWeight(amg_data) \
((amg_data)->schwarz_rlx_weight)
#define nalu_hypre_ParAMGDataSchwarzUseNonSymm(amg_data) \
((amg_data)->schwarz_use_nonsymm)
#define nalu_hypre_ParAMGDataSym(amg_data) ((amg_data)->ps_sym)
#define nalu_hypre_ParAMGDataLevel(amg_data) ((amg_data)->ps_level)
#define nalu_hypre_ParAMGDataMaxNzPerRow(amg_data) ((amg_data)->pi_max_nz_per_row)
#define nalu_hypre_ParAMGDataThreshold(amg_data) ((amg_data)->ps_threshold)
#define nalu_hypre_ParAMGDataFilter(amg_data) ((amg_data)->ps_filter)
#define nalu_hypre_ParAMGDataDropTol(amg_data) ((amg_data)->pi_drop_tol)
#define nalu_hypre_ParAMGDataEuclidFile(amg_data) ((amg_data)->euclidfile)
#define nalu_hypre_ParAMGDataEuLevel(amg_data) ((amg_data)->eu_level)
#define nalu_hypre_ParAMGDataEuSparseA(amg_data) ((amg_data)->eu_sparse_A)
#define nalu_hypre_ParAMGDataEuBJ(amg_data) ((amg_data)->eu_bj)
#define nalu_hypre_ParAMGDataILUType(amg_data) ((amg_data)->ilu_type)
#define nalu_hypre_ParAMGDataILULevel(amg_data) ((amg_data)->ilu_lfil)
#define nalu_hypre_ParAMGDataILUMaxRowNnz(amg_data) ((amg_data)->ilu_max_row_nnz)
#define nalu_hypre_ParAMGDataILUDroptol(amg_data) ((amg_data)->ilu_droptol)
#define nalu_hypre_ParAMGDataILUTriSolve(amg_data) ((amg_data)->ilu_tri_solve)
#define nalu_hypre_ParAMGDataILULowerJacobiIters(amg_data) ((amg_data)->ilu_lower_jacobi_iters)
#define nalu_hypre_ParAMGDataILUUpperJacobiIters(amg_data) ((amg_data)->ilu_upper_jacobi_iters)
#define nalu_hypre_ParAMGDataILUMaxIter(amg_data) ((amg_data)->ilu_max_iter)
#define nalu_hypre_ParAMGDataILULocalReordering(amg_data) ((amg_data)->ilu_reordering_type)
#define nalu_hypre_ParAMGDataFSAIAlgoType(amg_data) ((amg_data)->fsai_algo_type)
#define nalu_hypre_ParAMGDataFSAILocalSolveType(amg_data) ((amg_data)->fsai_local_solve_type)
#define nalu_hypre_ParAMGDataFSAIMaxSteps(amg_data) ((amg_data)->fsai_max_steps)
#define nalu_hypre_ParAMGDataFSAIMaxStepSize(amg_data) ((amg_data)->fsai_max_step_size)
#define nalu_hypre_ParAMGDataFSAIMaxNnzRow(amg_data) ((amg_data)->fsai_max_nnz_row)
#define nalu_hypre_ParAMGDataFSAINumLevels(amg_data) ((amg_data)->fsai_num_levels)
#define nalu_hypre_ParAMGDataFSAIThreshold(amg_data) ((amg_data)->fsai_threshold)
#define nalu_hypre_ParAMGDataFSAIEigMaxIters(amg_data) ((amg_data)->fsai_eig_max_iters)
#define nalu_hypre_ParAMGDataFSAIKapTolerance(amg_data) ((amg_data)->fsai_kap_tolerance)

#define nalu_hypre_ParAMGDataMaxEigEst(amg_data) ((amg_data)->max_eig_est)
#define nalu_hypre_ParAMGDataMinEigEst(amg_data) ((amg_data)->min_eig_est)
#define nalu_hypre_ParAMGDataChebyOrder(amg_data) ((amg_data)->cheby_order)
#define nalu_hypre_ParAMGDataChebyFraction(amg_data) ((amg_data)->cheby_fraction)
#define nalu_hypre_ParAMGDataChebyEigEst(amg_data) ((amg_data)->cheby_eig_est)
#define nalu_hypre_ParAMGDataChebyVariant(amg_data) ((amg_data)->cheby_variant)
#define nalu_hypre_ParAMGDataChebyScale(amg_data) ((amg_data)->cheby_scale)
#define nalu_hypre_ParAMGDataChebyDS(amg_data) ((amg_data)->cheby_ds)
#define nalu_hypre_ParAMGDataChebyCoefs(amg_data) ((amg_data)->cheby_coefs)

#define nalu_hypre_ParAMGDataCumNnzAP(amg_data)   ((amg_data)->cum_nnz_AP)

/* block */
#define nalu_hypre_ParAMGDataABlockArray(amg_data) ((amg_data)->A_block_array)
#define nalu_hypre_ParAMGDataPBlockArray(amg_data) ((amg_data)->P_block_array)
#define nalu_hypre_ParAMGDataRBlockArray(amg_data) ((amg_data)->R_block_array)

#define nalu_hypre_ParAMGDataBlockMode(amg_data) ((amg_data)->block_mode)


/* data generated in the solve phase */
#define nalu_hypre_ParAMGDataVtemp(amg_data) ((amg_data)->Vtemp)
#define nalu_hypre_ParAMGDataVtempLocal(amg_data) ((amg_data)->Vtemp_local)
#define nalu_hypre_ParAMGDataVtemplocalData(amg_data) ((amg_data)->Vtemp_local_data)
#define nalu_hypre_ParAMGDataCycleOpCount(amg_data) ((amg_data)->cycle_op_count)
#define nalu_hypre_ParAMGDataRtemp(amg_data) ((amg_data)->Rtemp)
#define nalu_hypre_ParAMGDataPtemp(amg_data) ((amg_data)->Ptemp)
#define nalu_hypre_ParAMGDataZtemp(amg_data) ((amg_data)->Ztemp)

/* fields used by GSMG */
#define nalu_hypre_ParAMGDataGSMG(amg_data) ((amg_data)->gsmg)
#define nalu_hypre_ParAMGDataNumSamples(amg_data) ((amg_data)->num_samples)

/* log info data */
#define nalu_hypre_ParAMGDataLogging(amg_data) ((amg_data)->logging)
#define nalu_hypre_ParAMGDataNumIterations(amg_data) ((amg_data)->num_iterations)
#ifdef CUMNUMIT
#define nalu_hypre_ParAMGDataCumNumIterations(amg_data) ((amg_data)->cum_num_iterations)
#endif
#define nalu_hypre_ParAMGDataRelativeResidualNorm(amg_data) ((amg_data)->rel_resid_norm)
#define nalu_hypre_ParAMGDataResidual(amg_data) ((amg_data)->residual)

/* output parameters */
#define nalu_hypre_ParAMGDataPrintLevel(amg_data) ((amg_data)->print_level)
#define nalu_hypre_ParAMGDataLogFileName(amg_data) ((amg_data)->log_file_name)
#define nalu_hypre_ParAMGDataDebugFlag(amg_data)   ((amg_data)->debug_flag)

/* BM Oct 22, 2006 */
#define nalu_hypre_ParAMGDataPlotGrids(amg_data) ((amg_data)->plot_grids)
#define nalu_hypre_ParAMGDataPlotFileName(amg_data) ((amg_data)->plot_filename)

/* coordinates BM Oct 17, 2006 */
#define nalu_hypre_ParAMGDataCoordDim(amg_data) ((amg_data)->coorddim)
#define nalu_hypre_ParAMGDataCoordinates(amg_data) ((amg_data)->coordinates)


#define nalu_hypre_ParAMGNumInterpVectors(amg_data) ((amg_data)->num_interp_vectors)
#define nalu_hypre_ParAMGNumLevelsInterpVectors(amg_data) ((amg_data)->num_levels_interp_vectors)
#define nalu_hypre_ParAMGInterpVectors(amg_data) ((amg_data)->interp_vectors)
#define nalu_hypre_ParAMGInterpVectorsArray(amg_data) ((amg_data)->interp_vectors_array)
#define nalu_hypre_ParAMGInterpVecVariant(amg_data) ((amg_data)->interp_vec_variant)
#define nalu_hypre_ParAMGInterpVecFirstLevel(amg_data) ((amg_data)->interp_vec_first_level)
#define nalu_hypre_ParAMGInterpVecAbsQTrunc(amg_data) ((amg_data)->interp_vectors_abs_q_trunc)
#define nalu_hypre_ParAMGInterpVecQMax(amg_data) ((amg_data)->interp_vectors_q_max)
#define nalu_hypre_ParAMGInterpRefine(amg_data) ((amg_data)->interp_refine)
#define nalu_hypre_ParAMGSmoothInterpVectors(amg_data) ((amg_data)->smooth_interp_vectors)
#define nalu_hypre_ParAMGDataExpandPWeights(amg_data) ((amg_data)->expandp_weights)

#define nalu_hypre_ParAMGDataCoarseSolver(amg_data) ((amg_data)->coarse_solver)
#define nalu_hypre_ParAMGDataACoarse(amg_data) ((amg_data)->A_coarse)
#define nalu_hypre_ParAMGDataFCoarse(amg_data) ((amg_data)->f_coarse)
#define nalu_hypre_ParAMGDataUCoarse(amg_data) ((amg_data)->u_coarse)
#define nalu_hypre_ParAMGDataNewComm(amg_data) ((amg_data)->new_comm)
#define nalu_hypre_ParAMGDataRedundant(amg_data) ((amg_data)->redundant)
#define nalu_hypre_ParAMGDataParticipate(amg_data) ((amg_data)->participate)

#define nalu_hypre_ParAMGDataGSSetup(amg_data) ((amg_data)->gs_setup)
#define nalu_hypre_ParAMGDataAMat(amg_data) ((amg_data)->A_mat)
#define nalu_hypre_ParAMGDataAInv(amg_data) ((amg_data)->A_inv)
#define nalu_hypre_ParAMGDataBVec(amg_data) ((amg_data)->b_vec)
#define nalu_hypre_ParAMGDataCommInfo(amg_data) ((amg_data)->comm_info)

/* additive AMG parameters */
#define nalu_hypre_ParAMGDataAdditive(amg_data) ((amg_data)->additive)
#define nalu_hypre_ParAMGDataMultAdditive(amg_data) ((amg_data)->mult_additive)
#define nalu_hypre_ParAMGDataSimple(amg_data) ((amg_data)->simple)
#define nalu_hypre_ParAMGDataAddLastLvl(amg_data) ((amg_data)->add_last_lvl)
#define nalu_hypre_ParAMGDataMultAddPMaxElmts(amg_data) ((amg_data)->add_P_max_elmts)
#define nalu_hypre_ParAMGDataMultAddTruncFactor(amg_data) ((amg_data)->add_trunc_factor)
#define nalu_hypre_ParAMGDataAddRelaxType(amg_data) ((amg_data)->add_rlx_type)
#define nalu_hypre_ParAMGDataAddRelaxWt(amg_data) ((amg_data)->add_rlx_wt)
#define nalu_hypre_ParAMGDataLambda(amg_data) ((amg_data)->Lambda)
#define nalu_hypre_ParAMGDataAtilde(amg_data) ((amg_data)->Atilde)
#define nalu_hypre_ParAMGDataRtilde(amg_data) ((amg_data)->Rtilde)
#define nalu_hypre_ParAMGDataXtilde(amg_data) ((amg_data)->Xtilde)
#define nalu_hypre_ParAMGDataDinv(amg_data) ((amg_data)->D_inv)

/* non-Galerkin parameters */
#define nalu_hypre_ParAMGDataNonGalerkNumTol(amg_data) ((amg_data)->nongalerk_num_tol)
#define nalu_hypre_ParAMGDataNonGalerkTol(amg_data) ((amg_data)->nongalerk_tol)
#define nalu_hypre_ParAMGDataNonGalerkinTol(amg_data) ((amg_data)->nongalerkin_tol)
#define nalu_hypre_ParAMGDataNonGalTolArray(amg_data) ((amg_data)->nongal_tol_array)

#define nalu_hypre_ParAMGDataRAP2(amg_data) ((amg_data)->rap2)
#define nalu_hypre_ParAMGDataKeepTranspose(amg_data) ((amg_data)->keepTranspose)
#define nalu_hypre_ParAMGDataModularizedMatMat(amg_data) ((amg_data)->modularized_matmat)

/*indices for the dof which will keep coarsening to the coarse level */
#define nalu_hypre_ParAMGDataNumCPoints(amg_data)  ((amg_data)->num_C_points)
#define nalu_hypre_ParAMGDataCPointsLevel(amg_data) ((amg_data)->C_points_coarse_level)
#define nalu_hypre_ParAMGDataCPointsLocalMarker(amg_data) ((amg_data)->C_points_local_marker)
#define nalu_hypre_ParAMGDataCPointsMarker(amg_data) ((amg_data)->C_points_marker)

/* information for preserving indices as special fine grid points */
#define nalu_hypre_ParAMGDataNumIsolatedFPoints(amg_data)     ((amg_data)->num_isolated_F_points)
#define nalu_hypre_ParAMGDataIsolatedFPointsMarker(amg_data)  ((amg_data)->isolated_F_points_marker)

/* information for preserving indices as fine grid points */
#define nalu_hypre_ParAMGDataNumFPoints(amg_data)     ((amg_data)->num_F_points)
#define nalu_hypre_ParAMGDataFPointsMarker(amg_data)  ((amg_data)->F_points_marker)

/* Parameters and data for SuperLU_Dist */
#ifdef NALU_HYPRE_USING_DSUPERLU
#define nalu_hypre_ParAMGDataDSLUThreshold(amg_data) ((amg_data)->dslu_threshold)
#define nalu_hypre_ParAMGDataDSLUSolver(amg_data) ((amg_data)->dslu_solver)
#endif

#endif
