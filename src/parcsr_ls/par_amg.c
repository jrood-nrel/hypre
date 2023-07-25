/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ParAMG functions
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "par_amg.h"
#ifdef NALU_HYPRE_USING_DSUPERLU
#include <math.h>
#include "superlu_ddefs.h"
#endif
/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGCreate
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_BoomerAMGCreate( void )
{
   nalu_hypre_ParAMGData  *amg_data;

   /* setup params */
   NALU_HYPRE_Int    max_levels;
   NALU_HYPRE_Int    max_coarse_size;
   NALU_HYPRE_Int    min_coarse_size;
   NALU_HYPRE_Int    coarsen_cut_factor;
   NALU_HYPRE_Real   strong_threshold;
   NALU_HYPRE_Real   strong_threshold_R;
   NALU_HYPRE_Real   filter_threshold_R;
   NALU_HYPRE_Int    Sabs;
   NALU_HYPRE_Real   max_row_sum;
   NALU_HYPRE_Real   trunc_factor;
   NALU_HYPRE_Real   agg_trunc_factor;
   NALU_HYPRE_Real   agg_P12_trunc_factor;
   NALU_HYPRE_Real   jacobi_trunc_threshold;
   NALU_HYPRE_Real   CR_rate;
   NALU_HYPRE_Real   CR_strong_th;
   NALU_HYPRE_Real   A_drop_tol;
   NALU_HYPRE_Int    A_drop_type;
   NALU_HYPRE_Int    interp_type;
   NALU_HYPRE_Int    sep_weight;
   NALU_HYPRE_Int    coarsen_type;
   NALU_HYPRE_Int    measure_type;
   NALU_HYPRE_Int    setup_type;
   NALU_HYPRE_Int    P_max_elmts;
   NALU_HYPRE_Int    num_functions;
   NALU_HYPRE_Int    nodal, nodal_levels, nodal_diag;
   NALU_HYPRE_Int    keep_same_sign;
   NALU_HYPRE_Int    num_paths;
   NALU_HYPRE_Int    agg_num_levels;
   NALU_HYPRE_Int    agg_interp_type;
   NALU_HYPRE_Int    agg_P_max_elmts;
   NALU_HYPRE_Int    agg_P12_max_elmts;
   NALU_HYPRE_Int    post_interp_type;
   NALU_HYPRE_Int    num_CR_relax_steps;
   NALU_HYPRE_Int    IS_type;
   NALU_HYPRE_Int    CR_use_CG;
   NALU_HYPRE_Int    cgc_its;
   NALU_HYPRE_Int    seq_threshold;
   NALU_HYPRE_Int    redundant;
   NALU_HYPRE_Int    rap2;
   NALU_HYPRE_Int    keepT;
   NALU_HYPRE_Int    modu_rap;

   /* solve params */
   NALU_HYPRE_Int    min_iter;
   NALU_HYPRE_Int    max_iter;
   NALU_HYPRE_Int    fcycle;
   NALU_HYPRE_Int    cycle_type;

   NALU_HYPRE_Int    converge_type;
   NALU_HYPRE_Real   tol;

   NALU_HYPRE_Int    num_sweeps;
   NALU_HYPRE_Int    relax_down;
   NALU_HYPRE_Int    relax_up;
   NALU_HYPRE_Int    relax_coarse;
   NALU_HYPRE_Int    relax_order;
   NALU_HYPRE_Real   relax_wt;
   NALU_HYPRE_Real   outer_wt;
   NALU_HYPRE_Real   nongalerkin_tol;
   NALU_HYPRE_Int    smooth_type;
   NALU_HYPRE_Int    smooth_num_levels;
   NALU_HYPRE_Int    smooth_num_sweeps;

   NALU_HYPRE_Int    variant, overlap, domain_type, schwarz_use_nonsymm;
   NALU_HYPRE_Real   schwarz_rlx_weight;
   NALU_HYPRE_Int    level, sym;
   NALU_HYPRE_Int    eu_level, eu_bj;
   NALU_HYPRE_Int    max_nz_per_row;
   NALU_HYPRE_Real   thresh, filter;
   NALU_HYPRE_Real   drop_tol;
   NALU_HYPRE_Real   eu_sparse_A;
   char    *euclidfile;
   NALU_HYPRE_Int    ilu_lfil;
   NALU_HYPRE_Int    ilu_type;
   NALU_HYPRE_Int    ilu_max_row_nnz;
   NALU_HYPRE_Int    ilu_max_iter;
   NALU_HYPRE_Real   ilu_droptol;
   NALU_HYPRE_Int    ilu_tri_solve;
   NALU_HYPRE_Int    ilu_lower_jacobi_iters;
   NALU_HYPRE_Int    ilu_upper_jacobi_iters;
   NALU_HYPRE_Int    ilu_reordering_type;

   NALU_HYPRE_Int    fsai_algo_type;
   NALU_HYPRE_Int    fsai_local_solve_type;
   NALU_HYPRE_Int    fsai_max_steps;
   NALU_HYPRE_Int    fsai_max_step_size;
   NALU_HYPRE_Int    fsai_max_nnz_row;
   NALU_HYPRE_Int    fsai_num_levels;
   NALU_HYPRE_Real   fsai_threshold;
   NALU_HYPRE_Int    fsai_eig_maxiter;
   NALU_HYPRE_Real   fsai_kap_tolerance;

   NALU_HYPRE_Int cheby_order;
   NALU_HYPRE_Int cheby_eig_est;
   NALU_HYPRE_Int cheby_variant;
   NALU_HYPRE_Int cheby_scale;
   NALU_HYPRE_Real cheby_eig_ratio;

   NALU_HYPRE_Int block_mode;

   NALU_HYPRE_Int    additive;
   NALU_HYPRE_Int    mult_additive;
   NALU_HYPRE_Int    simple;
   NALU_HYPRE_Int    add_last_lvl;
   NALU_HYPRE_Real   add_trunc_factor;
   NALU_HYPRE_Int    add_P_max_elmts;
   NALU_HYPRE_Int    add_rlx_type;
   NALU_HYPRE_Real   add_rlx_wt;

   /* log info */
   NALU_HYPRE_Int    num_iterations;
   NALU_HYPRE_Int    cum_num_iterations;
   NALU_HYPRE_Real   cum_nnz_AP;

   /* output params */
   NALU_HYPRE_Int    print_level;
   NALU_HYPRE_Int    logging;
   /* NALU_HYPRE_Int      cycle_op_count; */
   char     log_file_name[256];
   NALU_HYPRE_Int    debug_flag;

   char     plot_file_name[251] = {0};

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());

   /*-----------------------------------------------------------------------
    * Setup default values for parameters
    *-----------------------------------------------------------------------*/

   /* setup params */
   max_levels = 25;
   max_coarse_size = 9;
   min_coarse_size = 0;
   seq_threshold = 0;
   redundant = 0;
   coarsen_cut_factor = 0;
   strong_threshold = 0.25;
   strong_threshold_R = 0.25;
   filter_threshold_R = 0.0;
   Sabs = 0;
   max_row_sum = 0.9;
   trunc_factor = 0.0;
   agg_trunc_factor = 0.0;
   agg_P12_trunc_factor = 0.0;
   jacobi_trunc_threshold = 0.01;
   sep_weight = 0;
   coarsen_type = 10;
   interp_type = 6;
   measure_type = 0;
   setup_type = 1;
   P_max_elmts = 4;
   agg_P_max_elmts = 0;
   agg_P12_max_elmts = 0;
   num_functions = 1;
   nodal = 0;
   nodal_levels = max_levels;
   nodal_diag = 0;
   keep_same_sign = 0;
   num_paths = 1;
   agg_num_levels = 0;
   post_interp_type = 0;
   agg_interp_type = 4;
   num_CR_relax_steps = 2;
   CR_rate = 0.7;
   CR_strong_th = 0;
   A_drop_tol = 0.0;
   A_drop_type = -1;
   IS_type = 1;
   CR_use_CG = 0;
   cgc_its = 1;

   variant = 0;
   overlap = 1;
   domain_type = 2;
   schwarz_rlx_weight = 1.0;
   smooth_num_sweeps = 1;
   smooth_num_levels = 0;
   smooth_type = 6;
   schwarz_use_nonsymm = 0;

   level = 1;
   sym = 0;
   thresh = 0.1;
   filter = 0.05;
   drop_tol = 0.0001;
   max_nz_per_row = 20;
   euclidfile = NULL;
   eu_level = 0;
   eu_sparse_A = 0.0;
   eu_bj = 0;
   ilu_lfil = 0;
   ilu_type = 0;
   ilu_max_row_nnz = 20;
   ilu_max_iter = 1;
   ilu_droptol = 0.01;
   ilu_tri_solve = 1;
   ilu_lower_jacobi_iters = 5;
   ilu_upper_jacobi_iters = 5;
   ilu_reordering_type = 1;

   /* FSAI smoother params */
#if defined (NALU_HYPRE_USING_CUDA) || defined (NALU_HYPRE_USING_HIP)
   if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
   {
      fsai_algo_type = 3;
   }
   else
#endif
   {
      fsai_algo_type = nalu_hypre_NumThreads() > 4 ? 2 : 1;
   }
   fsai_local_solve_type = 1;
   fsai_max_steps = 5;
   fsai_max_step_size = 3;
   fsai_max_nnz_row = fsai_max_steps * fsai_max_step_size;
   fsai_num_levels = 2;
   fsai_threshold = 0.01;
   fsai_eig_maxiter = 5;
   fsai_kap_tolerance = 0.001;

   /* solve params */
   min_iter  = 0;
   max_iter  = 20;
   fcycle = 0;
   cycle_type = 1;
   converge_type = 0;
   tol = 1.0e-6;

   num_sweeps = 1;
   relax_down = 13;
   relax_up = 14;
   relax_coarse = 9;
   relax_order = 0;
   relax_wt = 1.0;
   outer_wt = 1.0;

   cheby_order = 2;
   cheby_variant = 0;
   cheby_scale = 1;
   cheby_eig_est = 10;
   cheby_eig_ratio = .3;

   block_mode = 0;

   additive = -1;
   mult_additive = -1;
   simple = -1;
   add_last_lvl = -1;
   add_trunc_factor = 0.0;
   add_P_max_elmts = 0;
   add_rlx_type = 18;
   add_rlx_wt = 1.0;

   /* log info */
   num_iterations = 0;
   cum_num_iterations = 0;
   cum_nnz_AP = -1.0;

   /* output params */
   print_level = 0;
   logging = 0;
   nalu_hypre_sprintf(log_file_name, "%s", "amg.out.log");
   /* cycle_op_count = 0; */
   debug_flag = 0;

   nongalerkin_tol = 0.0;

   rap2 = 0;
   keepT = 0;
   modu_rap = 0;

   if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
   {
      keepT           =  1;
      modu_rap        =  1;
      coarsen_type    =  8;
      relax_down      = 18;
      relax_up        = 18;
      agg_interp_type =  7;
   }

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   /*-----------------------------------------------------------------------
    * Create the nalu_hypre_ParAMGData structure and return
    *-----------------------------------------------------------------------*/

   amg_data = nalu_hypre_CTAlloc(nalu_hypre_ParAMGData, 1, NALU_HYPRE_MEMORY_HOST);

   /* memory location will be reset at the setup */
   nalu_hypre_ParAMGDataMemoryLocation(amg_data) = memory_location;

   nalu_hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data) = -1;
   nalu_hypre_ParAMGDataPartialCycleControl(amg_data) = -1;
   nalu_hypre_ParAMGDataMaxLevels(amg_data) =  max_levels;
   nalu_hypre_ParAMGDataUserCoarseRelaxType(amg_data) = 9;
   nalu_hypre_ParAMGDataUserRelaxType(amg_data) = -1;
   nalu_hypre_ParAMGDataUserNumSweeps(amg_data) = -1;
   nalu_hypre_ParAMGDataUserRelaxWeight(amg_data) = relax_wt;
   nalu_hypre_ParAMGDataOuterWt(amg_data) = outer_wt;
   nalu_hypre_BoomerAMGSetMaxCoarseSize(amg_data, max_coarse_size);
   nalu_hypre_BoomerAMGSetMinCoarseSize(amg_data, min_coarse_size);
   nalu_hypre_BoomerAMGSetCoarsenCutFactor(amg_data, coarsen_cut_factor);
   nalu_hypre_BoomerAMGSetStrongThreshold(amg_data, strong_threshold);
   nalu_hypre_BoomerAMGSetStrongThresholdR(amg_data, strong_threshold_R);
   nalu_hypre_BoomerAMGSetFilterThresholdR(amg_data, filter_threshold_R);
   nalu_hypre_BoomerAMGSetSabs(amg_data, Sabs);
   nalu_hypre_BoomerAMGSetMaxRowSum(amg_data, max_row_sum);
   nalu_hypre_BoomerAMGSetTruncFactor(amg_data, trunc_factor);
   nalu_hypre_BoomerAMGSetAggTruncFactor(amg_data, agg_trunc_factor);
   nalu_hypre_BoomerAMGSetAggP12TruncFactor(amg_data, agg_P12_trunc_factor);
   nalu_hypre_BoomerAMGSetJacobiTruncThreshold(amg_data, jacobi_trunc_threshold);
   nalu_hypre_BoomerAMGSetSepWeight(amg_data, sep_weight);
   nalu_hypre_BoomerAMGSetMeasureType(amg_data, measure_type);
   nalu_hypre_BoomerAMGSetCoarsenType(amg_data, coarsen_type);
   nalu_hypre_BoomerAMGSetInterpType(amg_data, interp_type);
   nalu_hypre_BoomerAMGSetSetupType(amg_data, setup_type);
   nalu_hypre_BoomerAMGSetPMaxElmts(amg_data, P_max_elmts);
   nalu_hypre_BoomerAMGSetAggPMaxElmts(amg_data, agg_P_max_elmts);
   nalu_hypre_BoomerAMGSetAggP12MaxElmts(amg_data, agg_P12_max_elmts);
   nalu_hypre_BoomerAMGSetNumFunctions(amg_data, num_functions);
   nalu_hypre_BoomerAMGSetNodal(amg_data, nodal);
   nalu_hypre_BoomerAMGSetNodalLevels(amg_data, nodal_levels);
   nalu_hypre_BoomerAMGSetNodal(amg_data, nodal_diag);
   nalu_hypre_BoomerAMGSetKeepSameSign(amg_data, keep_same_sign);
   nalu_hypre_BoomerAMGSetNumPaths(amg_data, num_paths);
   nalu_hypre_BoomerAMGSetAggNumLevels(amg_data, agg_num_levels);
   nalu_hypre_BoomerAMGSetAggInterpType(amg_data, agg_interp_type);
   nalu_hypre_BoomerAMGSetPostInterpType(amg_data, post_interp_type);
   nalu_hypre_BoomerAMGSetNumCRRelaxSteps(amg_data, num_CR_relax_steps);
   nalu_hypre_BoomerAMGSetCRRate(amg_data, CR_rate);
   nalu_hypre_BoomerAMGSetCRStrongTh(amg_data, CR_strong_th);
   nalu_hypre_BoomerAMGSetADropTol(amg_data, A_drop_tol);
   nalu_hypre_BoomerAMGSetADropType(amg_data, A_drop_type);
   nalu_hypre_BoomerAMGSetISType(amg_data, IS_type);
   nalu_hypre_BoomerAMGSetCRUseCG(amg_data, CR_use_CG);
   nalu_hypre_BoomerAMGSetCGCIts(amg_data, cgc_its);
   nalu_hypre_BoomerAMGSetVariant(amg_data, variant);
   nalu_hypre_BoomerAMGSetOverlap(amg_data, overlap);
   nalu_hypre_BoomerAMGSetSchwarzRlxWeight(amg_data, schwarz_rlx_weight);
   nalu_hypre_BoomerAMGSetSchwarzUseNonSymm(amg_data, schwarz_use_nonsymm);
   nalu_hypre_BoomerAMGSetDomainType(amg_data, domain_type);
   nalu_hypre_BoomerAMGSetSym(amg_data, sym);
   nalu_hypre_BoomerAMGSetLevel(amg_data, level);
   nalu_hypre_BoomerAMGSetThreshold(amg_data, thresh);
   nalu_hypre_BoomerAMGSetFilter(amg_data, filter);
   nalu_hypre_BoomerAMGSetDropTol(amg_data, drop_tol);
   nalu_hypre_BoomerAMGSetMaxNzPerRow(amg_data, max_nz_per_row);
   nalu_hypre_BoomerAMGSetEuclidFile(amg_data, euclidfile);
   nalu_hypre_BoomerAMGSetEuLevel(amg_data, eu_level);
   nalu_hypre_BoomerAMGSetEuSparseA(amg_data, eu_sparse_A);
   nalu_hypre_BoomerAMGSetEuBJ(amg_data, eu_bj);
   nalu_hypre_BoomerAMGSetILUType(amg_data, ilu_type);
   nalu_hypre_BoomerAMGSetILULevel(amg_data, ilu_lfil);
   nalu_hypre_BoomerAMGSetILUMaxRowNnz(amg_data, ilu_max_row_nnz);
   nalu_hypre_BoomerAMGSetILUDroptol(amg_data, ilu_droptol);
   nalu_hypre_BoomerAMGSetILUTriSolve(amg_data, ilu_tri_solve);
   nalu_hypre_BoomerAMGSetILULowerJacobiIters(amg_data, ilu_lower_jacobi_iters);
   nalu_hypre_BoomerAMGSetILUUpperJacobiIters(amg_data, ilu_upper_jacobi_iters);
   nalu_hypre_BoomerAMGSetILUMaxIter(amg_data, ilu_max_iter);
   nalu_hypre_BoomerAMGSetILULocalReordering(amg_data, ilu_reordering_type);
   nalu_hypre_BoomerAMGSetFSAIAlgoType(amg_data, fsai_algo_type);
   nalu_hypre_BoomerAMGSetFSAILocalSolveType(amg_data, fsai_local_solve_type);
   nalu_hypre_BoomerAMGSetFSAIMaxSteps(amg_data, fsai_max_steps);
   nalu_hypre_BoomerAMGSetFSAIMaxStepSize(amg_data, fsai_max_step_size);
   nalu_hypre_BoomerAMGSetFSAIMaxNnzRow(amg_data, fsai_max_nnz_row);
   nalu_hypre_BoomerAMGSetFSAINumLevels(amg_data, fsai_num_levels);
   nalu_hypre_BoomerAMGSetFSAIThreshold(amg_data, fsai_threshold);
   nalu_hypre_BoomerAMGSetFSAIEigMaxIters(amg_data, fsai_eig_maxiter);
   nalu_hypre_BoomerAMGSetFSAIKapTolerance(amg_data, fsai_kap_tolerance);

   nalu_hypre_BoomerAMGSetMinIter(amg_data, min_iter);
   nalu_hypre_BoomerAMGSetMaxIter(amg_data, max_iter);
   nalu_hypre_BoomerAMGSetCycleType(amg_data, cycle_type);
   nalu_hypre_BoomerAMGSetFCycle(amg_data, fcycle);
   nalu_hypre_BoomerAMGSetConvergeType(amg_data, converge_type);
   nalu_hypre_BoomerAMGSetTol(amg_data, tol);
   nalu_hypre_BoomerAMGSetNumSweeps(amg_data, num_sweeps);
   nalu_hypre_BoomerAMGSetCycleRelaxType(amg_data, relax_down, 1);
   nalu_hypre_BoomerAMGSetCycleRelaxType(amg_data, relax_up, 2);
   nalu_hypre_BoomerAMGSetCycleRelaxType(amg_data, relax_coarse, 3);
   nalu_hypre_BoomerAMGSetRelaxOrder(amg_data, relax_order);
   nalu_hypre_BoomerAMGSetRelaxWt(amg_data, relax_wt);
   nalu_hypre_BoomerAMGSetOuterWt(amg_data, outer_wt);
   nalu_hypre_BoomerAMGSetSmoothType(amg_data, smooth_type);
   nalu_hypre_BoomerAMGSetSmoothNumLevels(amg_data, smooth_num_levels);
   nalu_hypre_BoomerAMGSetSmoothNumSweeps(amg_data, smooth_num_sweeps);

   nalu_hypre_BoomerAMGSetChebyOrder(amg_data, cheby_order);
   nalu_hypre_BoomerAMGSetChebyFraction(amg_data, cheby_eig_ratio);
   nalu_hypre_BoomerAMGSetChebyEigEst(amg_data, cheby_eig_est);
   nalu_hypre_BoomerAMGSetChebyVariant(amg_data, cheby_variant);
   nalu_hypre_BoomerAMGSetChebyScale(amg_data, cheby_scale);

   nalu_hypre_BoomerAMGSetNumIterations(amg_data, num_iterations);

   nalu_hypre_BoomerAMGSetAdditive(amg_data, additive);
   nalu_hypre_BoomerAMGSetMultAdditive(amg_data, mult_additive);
   nalu_hypre_BoomerAMGSetSimple(amg_data, simple);
   nalu_hypre_BoomerAMGSetMultAddPMaxElmts(amg_data, add_P_max_elmts);
   nalu_hypre_BoomerAMGSetMultAddTruncFactor(amg_data, add_trunc_factor);
   nalu_hypre_BoomerAMGSetAddRelaxType(amg_data, add_rlx_type);
   nalu_hypre_BoomerAMGSetAddRelaxWt(amg_data, add_rlx_wt);
   nalu_hypre_ParAMGDataAddLastLvl(amg_data) = add_last_lvl;
   nalu_hypre_ParAMGDataLambda(amg_data) = NULL;
   nalu_hypre_ParAMGDataXtilde(amg_data) = NULL;
   nalu_hypre_ParAMGDataRtilde(amg_data) = NULL;
   nalu_hypre_ParAMGDataDinv(amg_data) = NULL;

#ifdef CUMNUMIT
   nalu_hypre_ParAMGDataCumNumIterations(amg_data) = cum_num_iterations;
#endif
   nalu_hypre_BoomerAMGSetPrintLevel(amg_data, print_level);
   nalu_hypre_BoomerAMGSetLogging(amg_data, logging);
   nalu_hypre_BoomerAMGSetPrintFileName(amg_data, log_file_name);
   nalu_hypre_BoomerAMGSetDebugFlag(amg_data, debug_flag);
   nalu_hypre_BoomerAMGSetRestriction(amg_data, 0);
   nalu_hypre_BoomerAMGSetIsTriangular(amg_data, 0);
   nalu_hypre_BoomerAMGSetGMRESSwitchR(amg_data, 64);

   nalu_hypre_BoomerAMGSetGSMG(amg_data, 0);
   nalu_hypre_BoomerAMGSetNumSamples(amg_data, 0);

   nalu_hypre_ParAMGDataAArray(amg_data) = NULL;
   nalu_hypre_ParAMGDataPArray(amg_data) = NULL;
   nalu_hypre_ParAMGDataRArray(amg_data) = NULL;
   nalu_hypre_ParAMGDataCFMarkerArray(amg_data) = NULL;
   nalu_hypre_ParAMGDataVtemp(amg_data)  = NULL;
   nalu_hypre_ParAMGDataRtemp(amg_data)  = NULL;
   nalu_hypre_ParAMGDataPtemp(amg_data)  = NULL;
   nalu_hypre_ParAMGDataZtemp(amg_data)  = NULL;
   nalu_hypre_ParAMGDataFArray(amg_data) = NULL;
   nalu_hypre_ParAMGDataUArray(amg_data) = NULL;
   nalu_hypre_ParAMGDataDofFunc(amg_data) = NULL;
   nalu_hypre_ParAMGDataDofFuncArray(amg_data) = NULL;
   nalu_hypre_ParAMGDataDofPointArray(amg_data) = NULL;
   nalu_hypre_ParAMGDataDofPointArray(amg_data) = NULL;
   nalu_hypre_ParAMGDataPointDofMapArray(amg_data) = NULL;
   nalu_hypre_ParAMGDataSmoother(amg_data) = NULL;
   nalu_hypre_ParAMGDataL1Norms(amg_data) = NULL;

   nalu_hypre_ParAMGDataABlockArray(amg_data) = NULL;
   nalu_hypre_ParAMGDataPBlockArray(amg_data) = NULL;
   nalu_hypre_ParAMGDataRBlockArray(amg_data) = NULL;

   /* this can not be set by the user currently */
   nalu_hypre_ParAMGDataBlockMode(amg_data) = block_mode;

   /* Stuff for Chebyshev smoothing */
   nalu_hypre_ParAMGDataMaxEigEst(amg_data) = NULL;
   nalu_hypre_ParAMGDataMinEigEst(amg_data) = NULL;
   nalu_hypre_ParAMGDataChebyDS(amg_data) = NULL;
   nalu_hypre_ParAMGDataChebyCoefs(amg_data) = NULL;

   /* BM Oct 22, 2006 */
   nalu_hypre_ParAMGDataPlotGrids(amg_data) = 0;
   nalu_hypre_BoomerAMGSetPlotFileName (amg_data, plot_file_name);

   /* BM Oct 17, 2006 */
   nalu_hypre_ParAMGDataCoordDim(amg_data) = 0;
   nalu_hypre_ParAMGDataCoordinates(amg_data) = NULL;

   /* for fitting vectors for interp */
   nalu_hypre_BoomerAMGSetInterpVecVariant(amg_data, 0);
   nalu_hypre_BoomerAMGSetInterpVectors(amg_data, 0, NULL);
   nalu_hypre_ParAMGNumLevelsInterpVectors(amg_data) = max_levels;
   nalu_hypre_ParAMGInterpVectorsArray(amg_data) = NULL;
   nalu_hypre_ParAMGInterpVecQMax(amg_data) = 0;
   nalu_hypre_ParAMGInterpVecAbsQTrunc(amg_data) = 0.0;
   nalu_hypre_ParAMGInterpRefine(amg_data) = 0;
   nalu_hypre_ParAMGInterpVecFirstLevel(amg_data) = 0;
   nalu_hypre_ParAMGNumInterpVectors(amg_data) = 0;
   nalu_hypre_ParAMGSmoothInterpVectors(amg_data) = 0;
   nalu_hypre_ParAMGDataExpandPWeights(amg_data) = NULL;

   /* for redundant coarse grid solve */
   nalu_hypre_ParAMGDataSeqThreshold(amg_data) = seq_threshold;
   nalu_hypre_ParAMGDataRedundant(amg_data) = redundant;
   nalu_hypre_ParAMGDataCoarseSolver(amg_data) = NULL;
   nalu_hypre_ParAMGDataACoarse(amg_data) = NULL;
   nalu_hypre_ParAMGDataFCoarse(amg_data) = NULL;
   nalu_hypre_ParAMGDataUCoarse(amg_data) = NULL;
   nalu_hypre_ParAMGDataNewComm(amg_data) = nalu_hypre_MPI_COMM_NULL;

   /* for Gaussian elimination coarse grid solve */
   nalu_hypre_ParAMGDataGSSetup(amg_data) = 0;
   nalu_hypre_ParAMGDataAMat(amg_data) = NULL;
   nalu_hypre_ParAMGDataAInv(amg_data) = NULL;
   nalu_hypre_ParAMGDataBVec(amg_data) = NULL;
   nalu_hypre_ParAMGDataCommInfo(amg_data) = NULL;

   nalu_hypre_ParAMGDataNonGalerkinTol(amg_data) = nongalerkin_tol;
   nalu_hypre_ParAMGDataNonGalTolArray(amg_data) = NULL;

   nalu_hypre_ParAMGDataRAP2(amg_data)              = rap2;
   nalu_hypre_ParAMGDataKeepTranspose(amg_data)     = keepT;
   nalu_hypre_ParAMGDataModularizedMatMat(amg_data) = modu_rap;

   /* information for preserving indices as coarse grid points */
   nalu_hypre_ParAMGDataCPointsMarker(amg_data)      = NULL;
   nalu_hypre_ParAMGDataCPointsLocalMarker(amg_data) = NULL;
   nalu_hypre_ParAMGDataCPointsLevel(amg_data)       = 0;
   nalu_hypre_ParAMGDataNumCPoints(amg_data)         = 0;

   /* information for preserving indices as special fine grid points */
   nalu_hypre_ParAMGDataIsolatedFPointsMarker(amg_data) = NULL;
   nalu_hypre_ParAMGDataNumIsolatedFPoints(amg_data) = 0;

   nalu_hypre_ParAMGDataCumNnzAP(amg_data) = cum_nnz_AP;

#ifdef NALU_HYPRE_USING_DSUPERLU
   nalu_hypre_ParAMGDataDSLUThreshold(amg_data) = 0;
   nalu_hypre_ParAMGDataDSLUSolver(amg_data) = NULL;
#endif

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return (void *) amg_data;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGDestroy( void *data )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;
   if (amg_data)
   {
      NALU_HYPRE_Int     num_levels = nalu_hypre_ParAMGDataNumLevels(amg_data);
      NALU_HYPRE_Int     smooth_num_levels = nalu_hypre_ParAMGDataSmoothNumLevels(amg_data);
      NALU_HYPRE_Solver *smoother = nalu_hypre_ParAMGDataSmoother(amg_data);
      void         *amg = nalu_hypre_ParAMGDataCoarseSolver(amg_data);
      MPI_Comm      new_comm = nalu_hypre_ParAMGDataNewComm(amg_data);
      NALU_HYPRE_Int    *grid_relax_type = nalu_hypre_ParAMGDataGridRelaxType(amg_data);
      NALU_HYPRE_Int     i;
      NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParAMGDataMemoryLocation(amg_data);

#ifdef NALU_HYPRE_USING_DSUPERLU
      // if (nalu_hypre_ParAMGDataDSLUThreshold(amg_data) > 0)
      if (nalu_hypre_ParAMGDataDSLUSolver(amg_data) != NULL)
      {
         nalu_hypre_SLUDistDestroy(nalu_hypre_ParAMGDataDSLUSolver(amg_data));
         nalu_hypre_ParAMGDataDSLUSolver(amg_data) = NULL;
      }
#endif

      if (nalu_hypre_ParAMGDataMaxEigEst(amg_data))
      {
         nalu_hypre_TFree(nalu_hypre_ParAMGDataMaxEigEst(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParAMGDataMaxEigEst(amg_data) = NULL;
      }
      if (nalu_hypre_ParAMGDataMinEigEst(amg_data))
      {
         nalu_hypre_TFree(nalu_hypre_ParAMGDataMinEigEst(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParAMGDataMinEigEst(amg_data) = NULL;
      }
      if (nalu_hypre_ParAMGDataNumGridSweeps(amg_data))
      {
         nalu_hypre_TFree(nalu_hypre_ParAMGDataNumGridSweeps(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParAMGDataNumGridSweeps(amg_data) = NULL;
      }
      if (grid_relax_type)
      {
         NALU_HYPRE_Int num_levels = nalu_hypre_ParAMGDataNumLevels(amg_data);
         if (grid_relax_type[1] == 15 || grid_relax_type[3] == 15 )
         {
            if (grid_relax_type[1] == 15)
            {
               for (i = 0; i < num_levels; i++)
               {
                  NALU_HYPRE_ParCSRPCGDestroy(smoother[i]);
               }
            }
            if (grid_relax_type[3] == 15 && grid_relax_type[1] != 15)
            {
               NALU_HYPRE_ParCSRPCGDestroy(smoother[num_levels - 1]);
            }
            nalu_hypre_TFree(smoother, NALU_HYPRE_MEMORY_HOST);
         }

         nalu_hypre_TFree(nalu_hypre_ParAMGDataGridRelaxType(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParAMGDataGridRelaxType(amg_data) = NULL;
      }
      if (nalu_hypre_ParAMGDataRelaxWeight(amg_data))
      {
         nalu_hypre_TFree(nalu_hypre_ParAMGDataRelaxWeight(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParAMGDataRelaxWeight(amg_data) = NULL;
      }
      if (nalu_hypre_ParAMGDataOmega(amg_data))
      {
         nalu_hypre_TFree(nalu_hypre_ParAMGDataOmega(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParAMGDataOmega(amg_data) = NULL;
      }
      if (nalu_hypre_ParAMGDataNonGalTolArray(amg_data))
      {
         nalu_hypre_TFree(nalu_hypre_ParAMGDataNonGalTolArray(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParAMGDataNonGalTolArray(amg_data) = NULL;
      }
      if (nalu_hypre_ParAMGDataDofFunc(amg_data))
      {
         nalu_hypre_IntArrayDestroy(nalu_hypre_ParAMGDataDofFunc(amg_data));
         nalu_hypre_ParAMGDataDofFunc(amg_data) = NULL;
      }
      for (i = 1; i < num_levels; i++)
      {
         nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataFArray(amg_data)[i]);
         nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataUArray(amg_data)[i]);

         if (nalu_hypre_ParAMGDataAArray(amg_data)[i])
         {
            nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParAMGDataAArray(amg_data)[i]);
         }

         if (nalu_hypre_ParAMGDataPArray(amg_data)[i - 1])
         {
            nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParAMGDataPArray(amg_data)[i - 1]);
         }

         if (nalu_hypre_ParAMGDataRestriction(amg_data))
         {
            if (nalu_hypre_ParAMGDataRArray(amg_data)[i - 1])
            {
               nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParAMGDataRArray(amg_data)[i - 1]);
            }
         }

         nalu_hypre_IntArrayDestroy(nalu_hypre_ParAMGDataCFMarkerArray(amg_data)[i - 1]);

         /* get rid of any block structures */
         if (nalu_hypre_ParAMGDataABlockArray(amg_data)[i])
         {
            nalu_hypre_ParCSRBlockMatrixDestroy(nalu_hypre_ParAMGDataABlockArray(amg_data)[i]);
         }

         if (nalu_hypre_ParAMGDataPBlockArray(amg_data)[i - 1])
         {
            nalu_hypre_ParCSRBlockMatrixDestroy(nalu_hypre_ParAMGDataPBlockArray(amg_data)[i - 1]);
         }

         /* RL */
         if (nalu_hypre_ParAMGDataRestriction(amg_data))
         {
            if (nalu_hypre_ParAMGDataRBlockArray(amg_data)[i - 1])
            {
               nalu_hypre_ParCSRBlockMatrixDestroy(nalu_hypre_ParAMGDataRBlockArray(amg_data)[i - 1]);
            }
         }
      }
      if (nalu_hypre_ParAMGDataGridRelaxPoints(amg_data))
      {
         for (i = 0; i < 4; i++)
         {
            nalu_hypre_TFree(nalu_hypre_ParAMGDataGridRelaxPoints(amg_data)[i], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(nalu_hypre_ParAMGDataGridRelaxPoints(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParAMGDataGridRelaxPoints(amg_data) = NULL;
      }

      nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParAMGDataLambda(amg_data));

      if (nalu_hypre_ParAMGDataAtilde(amg_data))
      {
         nalu_hypre_ParCSRMatrix *Atilde = nalu_hypre_ParAMGDataAtilde(amg_data);
         nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(Atilde));
         nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(Atilde));
         nalu_hypre_TFree(Atilde, NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataXtilde(amg_data));
      nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataRtilde(amg_data));

      if (nalu_hypre_ParAMGDataL1Norms(amg_data))
      {
         for (i = 0; i < num_levels; i++)
         {
            nalu_hypre_SeqVectorDestroy(nalu_hypre_ParAMGDataL1Norms(amg_data)[i]);
         }
         nalu_hypre_TFree(nalu_hypre_ParAMGDataL1Norms(amg_data), NALU_HYPRE_MEMORY_HOST);
      }

      if (nalu_hypre_ParAMGDataChebyCoefs(amg_data))
      {
         for (i = 0; i < num_levels; i++)
         {
            if (nalu_hypre_ParAMGDataChebyCoefs(amg_data)[i])
            {
               nalu_hypre_TFree(nalu_hypre_ParAMGDataChebyCoefs(amg_data)[i], NALU_HYPRE_MEMORY_HOST);
            }
         }
         nalu_hypre_TFree(nalu_hypre_ParAMGDataChebyCoefs(amg_data), NALU_HYPRE_MEMORY_HOST);
      }

      if (nalu_hypre_ParAMGDataChebyDS(amg_data))
      {
         for (i = 0; i < num_levels; i++)
         {
            nalu_hypre_SeqVectorDestroy(nalu_hypre_ParAMGDataChebyDS(amg_data)[i]);
         }
         nalu_hypre_TFree(nalu_hypre_ParAMGDataChebyDS(amg_data), NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_TFree(nalu_hypre_ParAMGDataDinv(amg_data), NALU_HYPRE_MEMORY_HOST);

      /* get rid of a fine level block matrix */
      if (nalu_hypre_ParAMGDataABlockArray(amg_data))
      {
         if (nalu_hypre_ParAMGDataABlockArray(amg_data)[0])
         {
            nalu_hypre_ParCSRBlockMatrixDestroy(nalu_hypre_ParAMGDataABlockArray(amg_data)[0]);
         }
      }

      /* see comments in par_coarsen.c regarding special case for CF_marker */
      if (num_levels == 1)
      {
         nalu_hypre_IntArrayDestroy(nalu_hypre_ParAMGDataCFMarkerArray(amg_data)[0]);
      }

      nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataVtemp(amg_data));
      nalu_hypre_TFree(nalu_hypre_ParAMGDataFArray(amg_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataUArray(amg_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataAArray(amg_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataABlockArray(amg_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataPBlockArray(amg_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataPArray(amg_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataCFMarkerArray(amg_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataRtemp(amg_data));
      nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataPtemp(amg_data));
      nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataZtemp(amg_data));

      if (nalu_hypre_ParAMGDataDofFuncArray(amg_data))
      {
         for (i = 1; i < num_levels; i++)
         {
            nalu_hypre_IntArrayDestroy(nalu_hypre_ParAMGDataDofFuncArray(amg_data)[i]);
         }
         nalu_hypre_TFree(nalu_hypre_ParAMGDataDofFuncArray(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParAMGDataDofFuncArray(amg_data) = NULL;
      }
      if (nalu_hypre_ParAMGDataRestriction(amg_data))
      {
         nalu_hypre_TFree(nalu_hypre_ParAMGDataRBlockArray(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nalu_hypre_ParAMGDataRArray(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParAMGDataRArray(amg_data) = NULL;
      }
      if (nalu_hypre_ParAMGDataDofPointArray(amg_data))
      {
         for (i = 0; i < num_levels; i++)
         {
            nalu_hypre_TFree(nalu_hypre_ParAMGDataDofPointArray(amg_data)[i], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(nalu_hypre_ParAMGDataDofPointArray(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParAMGDataDofPointArray(amg_data) = NULL;
      }
      if (nalu_hypre_ParAMGDataPointDofMapArray(amg_data))
      {
         for (i = 0; i < num_levels; i++)
         {
            nalu_hypre_TFree(nalu_hypre_ParAMGDataPointDofMapArray(amg_data)[i], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(nalu_hypre_ParAMGDataPointDofMapArray(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParAMGDataPointDofMapArray(amg_data) = NULL;
      }

      if (smooth_num_levels)
      {
         if ( nalu_hypre_ParAMGDataSmoothType(amg_data) == 7 ||
              nalu_hypre_ParAMGDataSmoothType(amg_data) == 17 )
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               NALU_HYPRE_ParCSRPilutDestroy(smoother[i]);
            }
         }
         else if ( nalu_hypre_ParAMGDataSmoothType(amg_data) == 8 ||
                   nalu_hypre_ParAMGDataSmoothType(amg_data) == 18 )
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               NALU_HYPRE_ParCSRParaSailsDestroy(smoother[i]);
            }
         }
         else if ( nalu_hypre_ParAMGDataSmoothType(amg_data) == 9 ||
                   nalu_hypre_ParAMGDataSmoothType(amg_data) == 19 )
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               NALU_HYPRE_EuclidDestroy(smoother[i]);
            }
         }
         else if ( nalu_hypre_ParAMGDataSmoothType(amg_data) == 4 )
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               NALU_HYPRE_FSAIDestroy(smoother[i]);
            }
         }
         else if ( nalu_hypre_ParAMGDataSmoothType(amg_data) == 5 ||
                   nalu_hypre_ParAMGDataSmoothType(amg_data) == 15 )
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               NALU_HYPRE_ILUDestroy(smoother[i]);
            }
         }
         else if ( nalu_hypre_ParAMGDataSmoothType(amg_data) == 6 ||
                   nalu_hypre_ParAMGDataSmoothType(amg_data) == 16 )
         {
            for (i = 0; i < smooth_num_levels; i++)
            {
               NALU_HYPRE_SchwarzDestroy(smoother[i]);
            }
         }
         nalu_hypre_TFree(nalu_hypre_ParAMGDataSmoother(amg_data), NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataResidual(amg_data));
      nalu_hypre_ParAMGDataResidual(amg_data) = NULL;

      if ( nalu_hypre_ParAMGInterpVecVariant(amg_data) > 0 &&
           nalu_hypre_ParAMGNumInterpVectors(amg_data) > 0)
      {
         NALU_HYPRE_Int         num_vecs =  nalu_hypre_ParAMGNumInterpVectors(amg_data);
         nalu_hypre_ParVector **sm_vecs;
         NALU_HYPRE_Int         j, num_il;

         num_il = nalu_hypre_min(nalu_hypre_ParAMGNumLevelsInterpVectors(amg_data), num_levels);

         /* don't destroy lev = 0 - this was user input */
         for (i = 1; i < num_il; i++)
         {
            sm_vecs = nalu_hypre_ParAMGInterpVectorsArray(amg_data)[i];
            for (j = 0; j < num_vecs; j++)
            {
               nalu_hypre_ParVectorDestroy(sm_vecs[j]);
            }
            nalu_hypre_TFree(sm_vecs, NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(nalu_hypre_ParAMGInterpVectorsArray(amg_data), NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_BoomerAMGDestroy(amg);
      nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParAMGDataACoarse(amg_data));
      nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataUCoarse(amg_data));
      nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataFCoarse(amg_data));

      /* destroy input CF_marker data */
      nalu_hypre_TFree(nalu_hypre_ParAMGDataCPointsMarker(amg_data), memory_location);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataCPointsLocalMarker(amg_data), memory_location);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataFPointsMarker(amg_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataIsolatedFPointsMarker(amg_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataAMat(amg_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataAInv(amg_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataBVec(amg_data), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataCommInfo(amg_data), NALU_HYPRE_MEMORY_HOST);

      if (new_comm != nalu_hypre_MPI_COMM_NULL)
      {
         nalu_hypre_MPI_Comm_free(&new_comm);
      }

      nalu_hypre_TFree(amg_data, NALU_HYPRE_MEMORY_HOST);
   }
   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Routines to set the setup phase parameters
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetRestriction( void *data,
                               NALU_HYPRE_Int   restr_par )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /* RL: currently, only 0: R = P^T
    *                     1: AIR
    *                     2: AIR-2
    *                     15: a special version of AIR-2 with less communication cost
    *                     k(k>=3,k!=15): Neumann AIR of degree k-3
    */
   if (restr_par < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataRestriction(amg_data) = restr_par;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetIsTriangular(void *data,
                               NALU_HYPRE_Int is_triangular )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataIsTriangular(amg_data) = is_triangular;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetGMRESSwitchR(void *data,
                               NALU_HYPRE_Int gmres_switch )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataGMRESSwitchR(amg_data) = gmres_switch;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetMaxLevels( void *data,
                             NALU_HYPRE_Int   max_levels )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;
   NALU_HYPRE_Int old_max_levels;
   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (max_levels < 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   old_max_levels = nalu_hypre_ParAMGDataMaxLevels(amg_data);
   if (old_max_levels < max_levels)
   {
      NALU_HYPRE_Real *relax_weight, *omega, *nongal_tol_array;
      NALU_HYPRE_Real relax_wt, outer_wt, nongalerkin_tol;
      NALU_HYPRE_Int i;
      relax_weight = nalu_hypre_ParAMGDataRelaxWeight(amg_data);
      if (relax_weight)
      {
         relax_wt = nalu_hypre_ParAMGDataUserRelaxWeight(amg_data);
         relax_weight = nalu_hypre_TReAlloc(relax_weight,  NALU_HYPRE_Real,  max_levels, NALU_HYPRE_MEMORY_HOST);
         for (i = old_max_levels; i < max_levels; i++)
         {
            relax_weight[i] = relax_wt;
         }
         nalu_hypre_ParAMGDataRelaxWeight(amg_data) = relax_weight;
      }
      omega = nalu_hypre_ParAMGDataOmega(amg_data);
      if (omega)
      {
         outer_wt = nalu_hypre_ParAMGDataOuterWt(amg_data);
         omega = nalu_hypre_TReAlloc(omega,  NALU_HYPRE_Real,  max_levels, NALU_HYPRE_MEMORY_HOST);
         for (i = old_max_levels; i < max_levels; i++)
         {
            omega[i] = outer_wt;
         }
         nalu_hypre_ParAMGDataOmega(amg_data) = omega;
      }
      nongal_tol_array = nalu_hypre_ParAMGDataNonGalTolArray(amg_data);
      if (nongal_tol_array)
      {
         nongalerkin_tol = nalu_hypre_ParAMGDataNonGalerkinTol(amg_data);
         nongal_tol_array = nalu_hypre_TReAlloc(nongal_tol_array,  NALU_HYPRE_Real,  max_levels, NALU_HYPRE_MEMORY_HOST);
         for (i = old_max_levels; i < max_levels; i++)
         {
            nongal_tol_array[i] = nongalerkin_tol;
         }
         nalu_hypre_ParAMGDataNonGalTolArray(amg_data) = nongal_tol_array;
      }
   }
   nalu_hypre_ParAMGDataMaxLevels(amg_data) = max_levels;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetMaxLevels( void *data,
                             NALU_HYPRE_Int *  max_levels )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *max_levels = nalu_hypre_ParAMGDataMaxLevels(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetMaxCoarseSize( void *data,
                                 NALU_HYPRE_Int   max_coarse_size )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (max_coarse_size < 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataMaxCoarseSize(amg_data) = max_coarse_size;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetMaxCoarseSize( void *data,
                                 NALU_HYPRE_Int *  max_coarse_size )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *max_coarse_size = nalu_hypre_ParAMGDataMaxCoarseSize(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetMinCoarseSize( void *data,
                                 NALU_HYPRE_Int   min_coarse_size )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (min_coarse_size < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataMinCoarseSize(amg_data) = min_coarse_size;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetMinCoarseSize( void *data,
                                 NALU_HYPRE_Int *  min_coarse_size )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *min_coarse_size = nalu_hypre_ParAMGDataMinCoarseSize(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetSeqThreshold( void *data,
                                NALU_HYPRE_Int   seq_threshold )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (seq_threshold < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataSeqThreshold(amg_data) = seq_threshold;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetSeqThreshold( void *data,
                                NALU_HYPRE_Int *  seq_threshold )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *seq_threshold = nalu_hypre_ParAMGDataSeqThreshold(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetRedundant( void *data,
                             NALU_HYPRE_Int   redundant )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (redundant < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataRedundant(amg_data) = redundant;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetRedundant( void *data,
                             NALU_HYPRE_Int *  redundant )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *redundant = nalu_hypre_ParAMGDataRedundant(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetCoarsenCutFactor( void       *data,
                                    NALU_HYPRE_Int   coarsen_cut_factor )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (coarsen_cut_factor < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataCoarsenCutFactor(amg_data) = coarsen_cut_factor;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetCoarsenCutFactor( void       *data,
                                    NALU_HYPRE_Int  *coarsen_cut_factor )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *coarsen_cut_factor = nalu_hypre_ParAMGDataCoarsenCutFactor(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetStrongThreshold( void     *data,
                                   NALU_HYPRE_Real    strong_threshold )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (strong_threshold < 0 || strong_threshold > 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataStrongThreshold(amg_data) = strong_threshold;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetStrongThreshold( void     *data,
                                   NALU_HYPRE_Real *  strong_threshold )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *strong_threshold = nalu_hypre_ParAMGDataStrongThreshold(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetStrongThresholdR( void         *data,
                                    NALU_HYPRE_Real    strong_threshold )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (strong_threshold < 0 || strong_threshold > 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataStrongThresholdR(amg_data) = strong_threshold;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetStrongThresholdR( void       *data,
                                    NALU_HYPRE_Real *strong_threshold )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *strong_threshold = nalu_hypre_ParAMGDataStrongThresholdR(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetFilterThresholdR( void         *data,
                                    NALU_HYPRE_Real    filter_threshold )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (filter_threshold < 0 || filter_threshold > 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataFilterThresholdR(amg_data) = filter_threshold;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetFilterThresholdR( void       *data,
                                    NALU_HYPRE_Real *filter_threshold )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *filter_threshold = nalu_hypre_ParAMGDataFilterThresholdR(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetSabs( void         *data,
                        NALU_HYPRE_Int     Sabs )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataSabs(amg_data) = Sabs != 0;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetMaxRowSum( void     *data,
                             NALU_HYPRE_Real    max_row_sum )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (max_row_sum <= 0 || max_row_sum > 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataMaxRowSum(amg_data) = max_row_sum;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetMaxRowSum( void     *data,
                             NALU_HYPRE_Real *  max_row_sum )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *max_row_sum = nalu_hypre_ParAMGDataMaxRowSum(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetTruncFactor( void     *data,
                               NALU_HYPRE_Real    trunc_factor )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (trunc_factor < 0 || trunc_factor >= 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataTruncFactor(amg_data) = trunc_factor;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetTruncFactor( void     *data,
                               NALU_HYPRE_Real *  trunc_factor )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *trunc_factor = nalu_hypre_ParAMGDataTruncFactor(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetPMaxElmts( void     *data,
                             NALU_HYPRE_Int    P_max_elmts )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (P_max_elmts < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataPMaxElmts(amg_data) = P_max_elmts;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetPMaxElmts( void     *data,
                             NALU_HYPRE_Int *  P_max_elmts )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *P_max_elmts = nalu_hypre_ParAMGDataPMaxElmts(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetJacobiTruncThreshold( void     *data,
                                        NALU_HYPRE_Real    jacobi_trunc_threshold )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (jacobi_trunc_threshold < 0 || jacobi_trunc_threshold >= 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataJacobiTruncThreshold(amg_data) = jacobi_trunc_threshold;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetJacobiTruncThreshold( void     *data,
                                        NALU_HYPRE_Real *  jacobi_trunc_threshold )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *jacobi_trunc_threshold = nalu_hypre_ParAMGDataJacobiTruncThreshold(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetPostInterpType( void     *data,
                                  NALU_HYPRE_Int    post_interp_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (post_interp_type < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataPostInterpType(amg_data) = post_interp_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetPostInterpType( void     *data,
                                  NALU_HYPRE_Int  * post_interp_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *post_interp_type = nalu_hypre_ParAMGDataPostInterpType(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetInterpType( void     *data,
                              NALU_HYPRE_Int       interp_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }


   if ((interp_type < 0 || interp_type > 25) && interp_type != 100)

   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataInterpType(amg_data) = interp_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetInterpType( void     *data,
                              NALU_HYPRE_Int *     interp_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *interp_type = nalu_hypre_ParAMGDataInterpType(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetSepWeight( void     *data,
                             NALU_HYPRE_Int       sep_weight )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataSepWeight(amg_data) = sep_weight;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetMinIter( void     *data,
                           NALU_HYPRE_Int       min_iter )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataMinIter(amg_data) = min_iter;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetMinIter( void     *data,
                           NALU_HYPRE_Int *     min_iter )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *min_iter = nalu_hypre_ParAMGDataMinIter(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetMaxIter( void     *data,
                           NALU_HYPRE_Int     max_iter )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (max_iter < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataMaxIter(amg_data) = max_iter;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetMaxIter( void     *data,
                           NALU_HYPRE_Int *   max_iter )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *max_iter = nalu_hypre_ParAMGDataMaxIter(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetCoarsenType( void  *data,
                               NALU_HYPRE_Int    coarsen_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataCoarsenType(amg_data) = coarsen_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetCoarsenType( void  *data,
                               NALU_HYPRE_Int *  coarsen_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *coarsen_type = nalu_hypre_ParAMGDataCoarsenType(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetMeasureType( void  *data,
                               NALU_HYPRE_Int    measure_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataMeasureType(amg_data) = measure_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetMeasureType( void  *data,
                               NALU_HYPRE_Int *  measure_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *measure_type = nalu_hypre_ParAMGDataMeasureType(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetSetupType( void  *data,
                             NALU_HYPRE_Int    setup_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataSetupType(amg_data) = setup_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetSetupType( void  *data,
                             NALU_HYPRE_Int  *  setup_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *setup_type = nalu_hypre_ParAMGDataSetupType(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetCycleType( void  *data,
                             NALU_HYPRE_Int    cycle_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (cycle_type < 0 || cycle_type > 2)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataCycleType(amg_data) = cycle_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetCycleType( void  *data,
                             NALU_HYPRE_Int *  cycle_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *cycle_type = nalu_hypre_ParAMGDataCycleType(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetFCycle( void     *data,
                          NALU_HYPRE_Int fcycle )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataFCycle(amg_data) = fcycle != 0;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetFCycle( void      *data,
                          NALU_HYPRE_Int *fcycle )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *fcycle = nalu_hypre_ParAMGDataFCycle(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetConvergeType( void     *data,
                                NALU_HYPRE_Int type  )
{
   /* type 0: default. relative over ||b||
    *      1:          relative over ||r0||
    */
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /*
   if ()
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   */

   nalu_hypre_ParAMGDataConvergeType(amg_data) = type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetConvergeType( void      *data,
                                NALU_HYPRE_Int *type  )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *type = nalu_hypre_ParAMGDataConvergeType(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetTol( void     *data,
                       NALU_HYPRE_Real    tol  )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (tol < 0 || tol > 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataTol(amg_data) = tol;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetTol( void     *data,
                       NALU_HYPRE_Real *  tol  )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *tol = nalu_hypre_ParAMGDataTol(amg_data);

   return nalu_hypre_error_flag;
}

/* The "Get" function for SetNumSweeps is GetCycleNumSweeps. */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetNumSweeps( void     *data,
                             NALU_HYPRE_Int      num_sweeps )
{
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int *num_grid_sweeps;
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (num_sweeps < 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
      nalu_hypre_ParAMGDataNumGridSweeps(amg_data) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
   }

   num_grid_sweeps = nalu_hypre_ParAMGDataNumGridSweeps(amg_data);

   for (i = 0; i < 3; i++)
   {
      num_grid_sweeps[i] = num_sweeps;
   }
   num_grid_sweeps[3] = 1;

   nalu_hypre_ParAMGDataUserNumSweeps(amg_data) = num_sweeps;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetCycleNumSweeps( void     *data,
                                  NALU_HYPRE_Int      num_sweeps,
                                  NALU_HYPRE_Int      k )
{
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int *num_grid_sweeps;
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (num_sweeps < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if (k < 1 || k > 3)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
      num_grid_sweeps = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < 4; i++)
      {
         num_grid_sweeps[i] = 1;
      }
      nalu_hypre_ParAMGDataNumGridSweeps(amg_data) = num_grid_sweeps;
   }

   nalu_hypre_ParAMGDataNumGridSweeps(amg_data)[k] = num_sweeps;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetCycleNumSweeps( void     *data,
                                  NALU_HYPRE_Int *    num_sweeps,
                                  NALU_HYPRE_Int      k )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (k < 1 || k > 3)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_ParAMGDataNumGridSweeps(amg_data) == NULL)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *num_sweeps = nalu_hypre_ParAMGDataNumGridSweeps(amg_data)[k];

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetNumGridSweeps( void     *data,
                                 NALU_HYPRE_Int      *num_grid_sweeps )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (!num_grid_sweeps)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_ParAMGDataNumGridSweeps(amg_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParAMGDataNumGridSweeps(amg_data), NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_ParAMGDataNumGridSweeps(amg_data) = num_grid_sweeps;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetNumGridSweeps( void     *data,
                                 NALU_HYPRE_Int    ** num_grid_sweeps )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *num_grid_sweeps = nalu_hypre_ParAMGDataNumGridSweeps(amg_data);

   return nalu_hypre_error_flag;
}

/* The "Get" function for SetRelaxType is GetCycleRelaxType. */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetRelaxType( void     *data,
                             NALU_HYPRE_Int      relax_type )
{
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int *grid_relax_type;
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (relax_type < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_ParAMGDataGridRelaxType(amg_data) == NULL)
   {
      nalu_hypre_ParAMGDataGridRelaxType(amg_data) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
   }
   grid_relax_type = nalu_hypre_ParAMGDataGridRelaxType(amg_data);

   for (i = 0; i < 3; i++)
   {
      grid_relax_type[i] = relax_type;
   }
   grid_relax_type[3] = 9;
   nalu_hypre_ParAMGDataUserCoarseRelaxType(amg_data) = 9;
   nalu_hypre_ParAMGDataUserRelaxType(amg_data) = relax_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetCycleRelaxType( void     *data,
                                  NALU_HYPRE_Int      relax_type,
                                  NALU_HYPRE_Int      k )
{
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int *grid_relax_type;
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (k < 1 || k > 3)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   if (relax_type < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_ParAMGDataGridRelaxType(amg_data) == NULL)
   {
      grid_relax_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 4, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < 3; i++)
      {
         grid_relax_type[i] = 3;
      }
      grid_relax_type[3] = 9;
      nalu_hypre_ParAMGDataGridRelaxType(amg_data) = grid_relax_type;
   }

   nalu_hypre_ParAMGDataGridRelaxType(amg_data)[k] = relax_type;
   if (k == 3)
   {
      nalu_hypre_ParAMGDataUserCoarseRelaxType(amg_data) = relax_type;
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetCycleRelaxType( void     *data,
                                  NALU_HYPRE_Int    * relax_type,
                                  NALU_HYPRE_Int      k )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (k < 1 || k > 3)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_ParAMGDataGridRelaxType(amg_data) == NULL)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *relax_type = nalu_hypre_ParAMGDataGridRelaxType(amg_data)[k];

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetRelaxOrder( void     *data,
                              NALU_HYPRE_Int       relax_order)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataRelaxOrder(amg_data) = relax_order;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetRelaxOrder( void     *data,
                              NALU_HYPRE_Int     * relax_order)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *relax_order = nalu_hypre_ParAMGDataRelaxOrder(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetGridRelaxType( void     *data,
                                 NALU_HYPRE_Int      *grid_relax_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (!grid_relax_type)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_ParAMGDataGridRelaxType(amg_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParAMGDataGridRelaxType(amg_data), NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_ParAMGDataGridRelaxType(amg_data) = grid_relax_type;
   nalu_hypre_ParAMGDataUserCoarseRelaxType(amg_data) = grid_relax_type[3];

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetGridRelaxType( void     *data,
                                 NALU_HYPRE_Int    ** grid_relax_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *grid_relax_type = nalu_hypre_ParAMGDataGridRelaxType(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetGridRelaxPoints( void     *data,
                                   NALU_HYPRE_Int      **grid_relax_points )
{
   NALU_HYPRE_Int i;
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (!grid_relax_points)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_ParAMGDataGridRelaxPoints(amg_data))
   {
      for (i = 0; i < 4; i++)
      {
         nalu_hypre_TFree(nalu_hypre_ParAMGDataGridRelaxPoints(amg_data)[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(nalu_hypre_ParAMGDataGridRelaxPoints(amg_data), NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_ParAMGDataGridRelaxPoints(amg_data) = grid_relax_points;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetGridRelaxPoints( void     *data,
                                   NALU_HYPRE_Int    *** grid_relax_points )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *grid_relax_points = nalu_hypre_ParAMGDataGridRelaxPoints(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetRelaxWeight( void     *data,
                               NALU_HYPRE_Real   *relax_weight )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (!relax_weight)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_ParAMGDataRelaxWeight(amg_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParAMGDataRelaxWeight(amg_data), NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_ParAMGDataRelaxWeight(amg_data) = relax_weight;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetRelaxWeight( void     *data,
                               NALU_HYPRE_Real ** relax_weight )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *relax_weight = nalu_hypre_ParAMGDataRelaxWeight(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetRelaxWt( void     *data,
                           NALU_HYPRE_Real    relax_weight )
{
   NALU_HYPRE_Int i, num_levels;
   NALU_HYPRE_Real *relax_weight_array;
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   num_levels = nalu_hypre_ParAMGDataMaxLevels(amg_data);
   if (nalu_hypre_ParAMGDataRelaxWeight(amg_data) == NULL)
   {
      nalu_hypre_ParAMGDataRelaxWeight(amg_data) = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_levels, NALU_HYPRE_MEMORY_HOST);
   }

   relax_weight_array = nalu_hypre_ParAMGDataRelaxWeight(amg_data);
   for (i = 0; i < num_levels; i++)
   {
      relax_weight_array[i] = relax_weight;
   }

   nalu_hypre_ParAMGDataUserRelaxWeight(amg_data) = relax_weight;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetLevelRelaxWt( void    *data,
                                NALU_HYPRE_Real   relax_weight,
                                NALU_HYPRE_Int      level )
{
   NALU_HYPRE_Int i, num_levels;
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;
   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   num_levels = nalu_hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels - 1 || level < 0)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   if (nalu_hypre_ParAMGDataRelaxWeight(amg_data) == NULL)
   {
      nalu_hypre_ParAMGDataRelaxWeight(amg_data) = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_levels, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_levels; i++)
      {
         nalu_hypre_ParAMGDataRelaxWeight(amg_data)[i] = 1.0;
      }
   }

   nalu_hypre_ParAMGDataRelaxWeight(amg_data)[level] = relax_weight;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetLevelRelaxWt( void    *data,
                                NALU_HYPRE_Real * relax_weight,
                                NALU_HYPRE_Int      level )
{
   NALU_HYPRE_Int num_levels;
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;
   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   num_levels = nalu_hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels - 1 || level < 0)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   if (nalu_hypre_ParAMGDataRelaxWeight(amg_data) == NULL)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *relax_weight = nalu_hypre_ParAMGDataRelaxWeight(amg_data)[level];

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetOmega( void     *data,
                         NALU_HYPRE_Real   *omega )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (!omega)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   if (nalu_hypre_ParAMGDataOmega(amg_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParAMGDataOmega(amg_data), NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_ParAMGDataOmega(amg_data) = omega;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetOmega( void     *data,
                         NALU_HYPRE_Real ** omega )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *omega = nalu_hypre_ParAMGDataOmega(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetOuterWt( void     *data,
                           NALU_HYPRE_Real    omega )
{
   NALU_HYPRE_Int i, num_levels;
   NALU_HYPRE_Real *omega_array;
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   num_levels = nalu_hypre_ParAMGDataMaxLevels(amg_data);
   if (nalu_hypre_ParAMGDataOmega(amg_data) == NULL)
   {
      nalu_hypre_ParAMGDataOmega(amg_data) = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_levels, NALU_HYPRE_MEMORY_HOST);
   }

   omega_array = nalu_hypre_ParAMGDataOmega(amg_data);
   for (i = 0; i < num_levels; i++)
   {
      omega_array[i] = omega;
   }
   nalu_hypre_ParAMGDataOuterWt(amg_data) = omega;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetLevelOuterWt( void    *data,
                                NALU_HYPRE_Real   omega,
                                NALU_HYPRE_Int      level )
{
   NALU_HYPRE_Int i, num_levels;
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;
   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   num_levels = nalu_hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels - 1)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   if (nalu_hypre_ParAMGDataOmega(amg_data) == NULL)
   {
      nalu_hypre_ParAMGDataOmega(amg_data) = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_levels, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_levels; i++)
      {
         nalu_hypre_ParAMGDataOmega(amg_data)[i] = 1.0;
      }
   }

   nalu_hypre_ParAMGDataOmega(amg_data)[level] = omega;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetLevelOuterWt( void    *data,
                                NALU_HYPRE_Real * omega,
                                NALU_HYPRE_Int      level )
{
   NALU_HYPRE_Int num_levels;
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;
   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   num_levels = nalu_hypre_ParAMGDataMaxLevels(amg_data);
   if (level > num_levels - 1)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   if (nalu_hypre_ParAMGDataOmega(amg_data) == NULL)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *omega = nalu_hypre_ParAMGDataOmega(amg_data)[level];

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetSmoothType( void     *data,
                              NALU_HYPRE_Int   smooth_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;
   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataSmoothType(amg_data) = smooth_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetSmoothType( void     *data,
                              NALU_HYPRE_Int * smooth_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *smooth_type = nalu_hypre_ParAMGDataSmoothType(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetSmoothNumLevels( void     *data,
                                   NALU_HYPRE_Int   smooth_num_levels )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (smooth_num_levels < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataSmoothNumLevels(amg_data) = smooth_num_levels;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetSmoothNumLevels( void     *data,
                                   NALU_HYPRE_Int * smooth_num_levels )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *smooth_num_levels = nalu_hypre_ParAMGDataSmoothNumLevels(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetSmoothNumSweeps( void     *data,
                                   NALU_HYPRE_Int   smooth_num_sweeps )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (smooth_num_sweeps < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataSmoothNumSweeps(amg_data) = smooth_num_sweeps;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetSmoothNumSweeps( void     *data,
                                   NALU_HYPRE_Int * smooth_num_sweeps )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *smooth_num_sweeps = nalu_hypre_ParAMGDataSmoothNumSweeps(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetLogging( void     *data,
                           NALU_HYPRE_Int       logging )
{
   /* This function should be called before Setup.  Logging changes
      may require allocation or freeing of arrays, which is presently
      only done there.
      It may be possible to support logging changes at other times,
      but there is little need.
   */
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataLogging(amg_data) = logging;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetLogging( void     *data,
                           NALU_HYPRE_Int     * logging )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *logging = nalu_hypre_ParAMGDataLogging(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetPrintLevel( void     *data,
                              NALU_HYPRE_Int print_level )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataPrintLevel(amg_data) = print_level;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetPrintLevel( void     *data,
                              NALU_HYPRE_Int * print_level )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *print_level =  nalu_hypre_ParAMGDataPrintLevel(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetPrintFileName( void       *data,
                                 const char *print_file_name )
{
   nalu_hypre_ParAMGData  *amg_data =  (nalu_hypre_ParAMGData*)data;
   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if ( strlen(print_file_name) > 256 )
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_sprintf(nalu_hypre_ParAMGDataLogFileName(amg_data), "%s", print_file_name);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetPrintFileName( void       *data,
                                 char ** print_file_name )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_sprintf( *print_file_name, "%s", nalu_hypre_ParAMGDataLogFileName(amg_data) );

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetNumIterations( void    *data,
                                 NALU_HYPRE_Int      num_iterations )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataNumIterations(amg_data) = num_iterations;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetDebugFlag( void     *data,
                             NALU_HYPRE_Int       debug_flag )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataDebugFlag(amg_data) = debug_flag;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetDebugFlag( void     *data,
                             NALU_HYPRE_Int     * debug_flag )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *debug_flag = nalu_hypre_ParAMGDataDebugFlag(amg_data);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGSetGSMG
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetGSMG( void       *data,
                        NALU_HYPRE_Int   par )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   amg_data->gsmg = par;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGSetNumSamples
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetNumSamples( void *data,
                              NALU_HYPRE_Int   par )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   amg_data->num_samples = par;

   return nalu_hypre_error_flag;
}

/* BM Aug 25, 2006 */

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetCGCIts( void *data,
                          NALU_HYPRE_Int  its)
{
   NALU_HYPRE_Int ierr = 0;
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) data;

   nalu_hypre_ParAMGDataCGCIts(amg_data) = its;
   return (ierr);
}

/* BM Oct 22, 2006 */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetPlotGrids( void *data,
                             NALU_HYPRE_Int plotgrids)
{
   NALU_HYPRE_Int ierr = 0;
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) data;

   nalu_hypre_ParAMGDataPlotGrids(amg_data) = plotgrids;
   return (ierr);
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetPlotFileName( void       *data,
                                const char *plot_file_name )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;
   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if ( strlen(plot_file_name) > 251 )
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   if (strlen(plot_file_name) == 0 )
   {
      nalu_hypre_sprintf(nalu_hypre_ParAMGDataPlotFileName(amg_data), "%s", "AMGgrids.CF.dat");
   }
   else
   {
      nalu_hypre_sprintf(nalu_hypre_ParAMGDataPlotFileName(amg_data), "%s", plot_file_name);
   }

   return nalu_hypre_error_flag;
}
/* Get the coarse grid hierarchy. Assumes cgrid is preallocated to the size of the local matrix.
 * Adapted from par_amg_setup.c, and simplified by ignoring printing in block mode.
 * We do a memcpy on the final grid hierarchy to avoid modifying user allocated data.
*/
NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetGridHierarchy( void       *data,
                                 NALU_HYPRE_Int *cgrid )
{
   NALU_HYPRE_Int *ibuff = NULL;
   NALU_HYPRE_Int *wbuff, *cbuff, *tmp;
   NALU_HYPRE_Int local_size, lev_size, i, j, level, num_levels, block_mode;
   nalu_hypre_IntArray          *CF_marker_array;
   nalu_hypre_IntArray          *CF_marker_array_host;
   NALU_HYPRE_Int               *CF_marker;

   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;
   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (!cgrid)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   block_mode = nalu_hypre_ParAMGDataBlockMode(amg_data);

   if ( block_mode)
   {
      nalu_hypre_ParCSRBlockMatrix **A_block_array;
      A_block_array = nalu_hypre_ParAMGDataABlockArray(amg_data);
      if (A_block_array == NULL)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Invalid AMG data. AMG setup has not been called!!\n");
         return nalu_hypre_error_flag;
      }

      // get local size and allocate some memory
      local_size = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRBlockMatrixDiag(A_block_array[0]));
      ibuff  = nalu_hypre_CTAlloc(NALU_HYPRE_Int, (2 * local_size), NALU_HYPRE_MEMORY_HOST);
      wbuff  = ibuff;
      cbuff  = ibuff + local_size;

      num_levels = nalu_hypre_ParAMGDataNumLevels(amg_data);
      for (level = (num_levels - 2); level >= 0; level--)
      {
         /* get the CF marker array on the host */
         CF_marker_array = nalu_hypre_ParAMGDataCFMarkerArray(amg_data)[level];
         if (nalu_hypre_GetActualMemLocation(nalu_hypre_IntArrayMemoryLocation(CF_marker_array)) ==
             nalu_hypre_MEMORY_DEVICE)
         {
            CF_marker_array_host = nalu_hypre_IntArrayCloneDeep_v2(CF_marker_array, NALU_HYPRE_MEMORY_HOST);
         }
         else
         {
            CF_marker_array_host = CF_marker_array;
         }
         CF_marker = nalu_hypre_IntArrayData(CF_marker_array_host);

         /* swap pointers */
         tmp = wbuff;
         wbuff = cbuff;
         cbuff = tmp;

         lev_size = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRBlockMatrixDiag(A_block_array[level]));

         for (i = 0, j = 0; i < lev_size; i++)
         {
            /* if a C-point */
            cbuff[i] = 0;
            if (CF_marker[i] > -1)
            {
               cbuff[i] = wbuff[j] + 1;
               j++;
            }
         }

         /* destroy copy host copy if necessary */
         if (nalu_hypre_GetActualMemLocation(nalu_hypre_IntArrayMemoryLocation(CF_marker_array)) ==
             nalu_hypre_MEMORY_DEVICE)
         {
            nalu_hypre_IntArrayDestroy(CF_marker_array_host);
         }
      }
   }
   else
   {
      nalu_hypre_ParCSRMatrix **A_array;
      A_array = nalu_hypre_ParAMGDataAArray(amg_data);
      if (A_array == NULL)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Invalid AMG data. AMG setup has not been called!!\n");
         return nalu_hypre_error_flag;
      }

      // get local size and allocate some memory
      local_size = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A_array[0]));
      wbuff  = nalu_hypre_CTAlloc(NALU_HYPRE_Int, (2 * local_size), NALU_HYPRE_MEMORY_HOST);
      cbuff  = wbuff + local_size;

      num_levels = nalu_hypre_ParAMGDataNumLevels(amg_data);
      for (level = (num_levels - 2); level >= 0; level--)
      {
         /* get the CF marker array on the host */
         CF_marker_array = nalu_hypre_ParAMGDataCFMarkerArray(amg_data)[level];
         if (nalu_hypre_GetActualMemLocation(nalu_hypre_IntArrayMemoryLocation(CF_marker_array)) ==
             nalu_hypre_MEMORY_DEVICE)
         {
            CF_marker_array_host = nalu_hypre_IntArrayCloneDeep_v2(CF_marker_array, NALU_HYPRE_MEMORY_HOST);
         }
         else
         {
            CF_marker_array_host = CF_marker_array;
         }
         CF_marker = nalu_hypre_IntArrayData(CF_marker_array_host);
         /* swap pointers */
         tmp = wbuff;
         wbuff = cbuff;
         cbuff = tmp;

         lev_size = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A_array[level]));

         for (i = 0, j = 0; i < lev_size; i++)
         {
            /* if a C-point */
            cbuff[i] = 0;
            if (CF_marker[i] > -1)
            {
               cbuff[i] = wbuff[j] + 1;
               j++;
            }
         }
         /* destroy copy host copy if necessary */
         if (nalu_hypre_GetActualMemLocation(nalu_hypre_IntArrayMemoryLocation(CF_marker_array)) ==
             nalu_hypre_MEMORY_DEVICE)
         {
            nalu_hypre_IntArrayDestroy(CF_marker_array_host);
         }
      }
   }
   // copy hierarchy into user provided array
   nalu_hypre_TMemcpy(cgrid, cbuff, NALU_HYPRE_Int, local_size, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
   // free memory
   nalu_hypre_TFree(ibuff, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/* BM Oct 17, 2006 */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetCoordDim( void *data,
                            NALU_HYPRE_Int coorddim)
{
   NALU_HYPRE_Int ierr = 0;
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) data;

   nalu_hypre_ParAMGDataCoordDim(amg_data) = coorddim;
   return (ierr);
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetCoordinates( void *data,
                               float *coordinates)
{
   NALU_HYPRE_Int ierr = 0;
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) data;

   nalu_hypre_ParAMGDataCoordinates(amg_data) = coordinates;
   return (ierr);
}

/*--------------------------------------------------------------------------
 * Routines to set the problem data parameters
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetNumFunctions( void     *data,
                                NALU_HYPRE_Int       num_functions )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (num_functions < 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataNumFunctions(amg_data) = num_functions;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetNumFunctions( void     *data,
                                NALU_HYPRE_Int     * num_functions )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *num_functions = nalu_hypre_ParAMGDataNumFunctions(amg_data);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicate whether to use nodal systems function
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetNodal( void     *data,
                         NALU_HYPRE_Int    nodal )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataNodal(amg_data) = nodal;

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * Indicate number of levels for nodal coarsening
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetNodalLevels( void     *data,
                               NALU_HYPRE_Int    nodal_levels )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataNodalLevels(amg_data) = nodal_levels;

   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * Indicate how to treat diag for primary matrix with  nodal systems function
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetNodalDiag( void     *data,
                             NALU_HYPRE_Int    nodal )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataNodalDiag(amg_data) = nodal;

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * Indicate whether to discard same sign coefficients in S for nodal>0
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetKeepSameSign( void      *data,
                                NALU_HYPRE_Int  keep_same_sign )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataKeepSameSign(amg_data) = keep_same_sign;

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * Indicate the degree of aggressive coarsening
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetNumPaths( void     *data,
                            NALU_HYPRE_Int       num_paths )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (num_paths < 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataNumPaths(amg_data) = num_paths;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the number of levels of aggressive coarsening
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetAggNumLevels( void     *data,
                                NALU_HYPRE_Int       agg_num_levels )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (agg_num_levels < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataAggNumLevels(amg_data) = agg_num_levels;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the interpolation used with aggressive coarsening
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetAggInterpType( void     *data,
                                 NALU_HYPRE_Int       agg_interp_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (agg_interp_type < 0 || agg_interp_type > 9)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataAggInterpType(amg_data) = agg_interp_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates max number of elements per row for aggressive coarsening
 * interpolation
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetAggPMaxElmts( void     *data,
                                NALU_HYPRE_Int       agg_P_max_elmts )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (agg_P_max_elmts < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataAggPMaxElmts(amg_data) = agg_P_max_elmts;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates max number of elements per row for smoothed
 * interpolation in mult-additive or simple method
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetMultAddPMaxElmts( void     *data,
                                    NALU_HYPRE_Int       add_P_max_elmts )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (add_P_max_elmts < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataMultAddPMaxElmts(amg_data) = add_P_max_elmts;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates Relaxtion Type for Additive Cycle
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetAddRelaxType( void     *data,
                                NALU_HYPRE_Int       add_rlx_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataAddRelaxType(amg_data) = add_rlx_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates Relaxation Weight for Additive Cycle
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetAddRelaxWt( void     *data,
                              NALU_HYPRE_Real       add_rlx_wt )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataAddRelaxWt(amg_data) = add_rlx_wt;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates max number of elements per row for 1st stage of aggressive
 * coarsening two-stage interpolation
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetAggP12MaxElmts( void     *data,
                                  NALU_HYPRE_Int       agg_P12_max_elmts )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (agg_P12_max_elmts < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataAggP12MaxElmts(amg_data) = agg_P12_max_elmts;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates truncation factor for aggressive coarsening interpolation
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetAggTruncFactor( void     *data,
                                  NALU_HYPRE_Real  agg_trunc_factor )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (agg_trunc_factor < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataAggTruncFactor(amg_data) = agg_trunc_factor;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the truncation factor for smoothed interpolation when using
 * mult-additive or simple method
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetMultAddTruncFactor( void     *data,
                                      NALU_HYPRE_Real      add_trunc_factor )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (add_trunc_factor < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataMultAddTruncFactor(amg_data) = add_trunc_factor;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates truncation factor for 1 stage of aggressive coarsening
 * two stage interpolation
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetAggP12TruncFactor( void     *data,
                                     NALU_HYPRE_Real  agg_P12_trunc_factor )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (agg_P12_trunc_factor < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataAggP12TruncFactor(amg_data) = agg_P12_trunc_factor;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the number of relaxation steps for Compatible relaxation
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetNumCRRelaxSteps( void     *data,
                                   NALU_HYPRE_Int       num_CR_relax_steps )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (num_CR_relax_steps < 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataNumCRRelaxSteps(amg_data) = num_CR_relax_steps;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the desired convergence rate for Compatible relaxation
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetCRRate( void     *data,
                          NALU_HYPRE_Real    CR_rate )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataCRRate(amg_data) = CR_rate;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the desired convergence rate for Compatible relaxation
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetCRStrongTh( void     *data,
                              NALU_HYPRE_Real    CR_strong_th )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataCRStrongTh(amg_data) = CR_strong_th;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates the drop tolerance for A-matrices from the 2nd level of AMG
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetADropTol( void     *data,
                            NALU_HYPRE_Real  A_drop_tol )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataADropTol(amg_data) = A_drop_tol;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetADropType( void      *data,
                             NALU_HYPRE_Int  A_drop_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataADropType(amg_data) = A_drop_type;

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * Indicates which independent set algorithm is used for CR
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetISType( void     *data,
                          NALU_HYPRE_Int      IS_type )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (IS_type < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataISType(amg_data) = IS_type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Indicates whether to use CG for compatible relaxation
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetCRUseCG( void     *data,
                           NALU_HYPRE_Int      CR_use_CG )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataCRUseCG(amg_data) = CR_use_CG;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetNumPoints( void     *data,
                             NALU_HYPRE_Int       num_points )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataNumPoints(amg_data) = num_points;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetDofFunc( void                 *data,
                           NALU_HYPRE_Int            *dof_func)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_IntArrayDestroy(nalu_hypre_ParAMGDataDofFunc(amg_data));
   /* NOTE: size and memory location of nalu_hypre_IntArray will be set during AMG setup */
   if (dof_func == NULL)
   {
      nalu_hypre_ParAMGDataDofFunc(amg_data) = NULL;
   }
   else
   {
      nalu_hypre_ParAMGDataDofFunc(amg_data) = nalu_hypre_IntArrayCreate(-1);
      nalu_hypre_IntArrayData(nalu_hypre_ParAMGDataDofFunc(amg_data)) = dof_func;
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetPointDofMap( void     *data,
                               NALU_HYPRE_Int      *point_dof_map )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_TFree(nalu_hypre_ParAMGDataPointDofMap(amg_data), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParAMGDataPointDofMap(amg_data) = point_dof_map;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetDofPoint( void     *data,
                            NALU_HYPRE_Int      *dof_point )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_TFree(nalu_hypre_ParAMGDataDofPoint(amg_data), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParAMGDataDofPoint(amg_data) = dof_point;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetNumIterations( void     *data,
                                 NALU_HYPRE_Int      *num_iterations )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *num_iterations = nalu_hypre_ParAMGDataNumIterations(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetCumNumIterations( void     *data,
                                    NALU_HYPRE_Int      *cum_num_iterations )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
#ifdef CUMNUMIT
   *cum_num_iterations = nalu_hypre_ParAMGDataCumNumIterations(amg_data);
#endif

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetResidual( void * data, nalu_hypre_ParVector ** resid )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;
   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *resid = nalu_hypre_ParAMGDataResidual( amg_data );
   return nalu_hypre_error_flag;
}


NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetRelResidualNorm( void     *data,
                                   NALU_HYPRE_Real   *rel_resid_norm )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *rel_resid_norm = nalu_hypre_ParAMGDataRelativeResidualNorm(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetVariant( void     *data,
                           NALU_HYPRE_Int       variant)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (variant < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataVariant(amg_data) = variant;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetVariant( void     *data,
                           NALU_HYPRE_Int     * variant)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *variant = nalu_hypre_ParAMGDataVariant(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetOverlap( void     *data,
                           NALU_HYPRE_Int       overlap)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (overlap < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataOverlap(amg_data) = overlap;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetOverlap( void     *data,
                           NALU_HYPRE_Int     * overlap)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *overlap = nalu_hypre_ParAMGDataOverlap(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetDomainType( void     *data,
                              NALU_HYPRE_Int       domain_type)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (domain_type < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataDomainType(amg_data) = domain_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetDomainType( void     *data,
                              NALU_HYPRE_Int     * domain_type)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *domain_type = nalu_hypre_ParAMGDataDomainType(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetSchwarzRlxWeight( void     *data,
                                    NALU_HYPRE_Real schwarz_rlx_weight)
{
   nalu_hypre_ParAMGData  *amg_data =  (nalu_hypre_ParAMGData*)data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataSchwarzRlxWeight(amg_data) = schwarz_rlx_weight;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetSchwarzRlxWeight( void     *data,
                                    NALU_HYPRE_Real   * schwarz_rlx_weight)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *schwarz_rlx_weight = nalu_hypre_ParAMGDataSchwarzRlxWeight(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetSchwarzUseNonSymm( void     *data,
                                     NALU_HYPRE_Int use_nonsymm)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataSchwarzUseNonSymm(amg_data) = use_nonsymm;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetSym( void     *data,
                       NALU_HYPRE_Int       sym)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataSym(amg_data) = sym;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetLevel( void     *data,
                         NALU_HYPRE_Int       level)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataLevel(amg_data) = level;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetThreshold( void     *data,
                             NALU_HYPRE_Real    thresh)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataThreshold(amg_data) = thresh;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetFilter( void     *data,
                          NALU_HYPRE_Real    filter)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataFilter(amg_data) = filter;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetDropTol( void     *data,
                           NALU_HYPRE_Real    drop_tol)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataDropTol(amg_data) = drop_tol;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetMaxNzPerRow( void     *data,
                               NALU_HYPRE_Int       max_nz_per_row)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (max_nz_per_row < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataMaxNzPerRow(amg_data) = max_nz_per_row;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetEuclidFile( void     *data,
                              char     *euclidfile)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataEuclidFile(amg_data) = euclidfile;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetEuLevel( void     *data,
                           NALU_HYPRE_Int      eu_level)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataEuLevel(amg_data) = eu_level;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetEuSparseA( void     *data,
                             NALU_HYPRE_Real    eu_sparse_A)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataEuSparseA(amg_data) = eu_sparse_A;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetEuBJ( void     *data,
                        NALU_HYPRE_Int       eu_bj)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataEuBJ(amg_data) = eu_bj;

   return nalu_hypre_error_flag;
}
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetILUType( void     *data,
                           NALU_HYPRE_Int       ilu_type)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataILUType(amg_data) = ilu_type;

   return nalu_hypre_error_flag;
}
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetILULevel( void     *data,
                            NALU_HYPRE_Int       ilu_lfil)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataILULevel(amg_data) = ilu_lfil;

   return nalu_hypre_error_flag;
}
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetILUDroptol( void     *data,
                              NALU_HYPRE_Real       ilu_droptol)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataILUDroptol(amg_data) = ilu_droptol;

   return nalu_hypre_error_flag;
}
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetILUTriSolve( void     *data,
                               NALU_HYPRE_Int    ilu_tri_solve)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataILUTriSolve(amg_data) = ilu_tri_solve;

   return nalu_hypre_error_flag;
}
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetILULowerJacobiIters( void     *data,
                                       NALU_HYPRE_Int    ilu_lower_jacobi_iters)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataILULowerJacobiIters(amg_data) = ilu_lower_jacobi_iters;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetILUUpperJacobiIters( void     *data,
                                       NALU_HYPRE_Int    ilu_upper_jacobi_iters)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataILUUpperJacobiIters(amg_data) = ilu_upper_jacobi_iters;

   return nalu_hypre_error_flag;
}
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetILUMaxIter( void     *data,
                              NALU_HYPRE_Int       ilu_max_iter)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataILUMaxIter(amg_data) = ilu_max_iter;

   return nalu_hypre_error_flag;
}
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetILUMaxRowNnz( void     *data,
                                NALU_HYPRE_Int       ilu_max_row_nnz)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataILUMaxRowNnz(amg_data) = ilu_max_row_nnz;

   return nalu_hypre_error_flag;
}
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetILULocalReordering( void     *data,
                                      NALU_HYPRE_Int       ilu_reordering_type)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataILULocalReordering(amg_data) = ilu_reordering_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetFSAIAlgoType( void      *data,
                                NALU_HYPRE_Int  fsai_algo_type)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataFSAIAlgoType(amg_data) = fsai_algo_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetFSAILocalSolveType( void      *data,
                                      NALU_HYPRE_Int  fsai_local_solve_type)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataFSAILocalSolveType(amg_data) = fsai_local_solve_type;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetFSAIMaxSteps( void      *data,
                                NALU_HYPRE_Int  fsai_max_steps)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataFSAIMaxSteps(amg_data) = fsai_max_steps;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetFSAIMaxStepSize( void      *data,
                                   NALU_HYPRE_Int  fsai_max_step_size)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataFSAIMaxStepSize(amg_data) = fsai_max_step_size;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetFSAIMaxNnzRow( void      *data,
                                 NALU_HYPRE_Int  fsai_max_nnz_row)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataFSAIMaxNnzRow(amg_data) = fsai_max_nnz_row;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetFSAINumLevels( void      *data,
                                 NALU_HYPRE_Int  fsai_num_levels)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataFSAINumLevels(amg_data) = fsai_num_levels;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetFSAIThreshold( void      *data,
                                 NALU_HYPRE_Real fsai_threshold)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataFSAIThreshold(amg_data) = fsai_threshold;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetFSAIEigMaxIters( void      *data,
                                   NALU_HYPRE_Int  fsai_eig_max_iters)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataFSAIEigMaxIters(amg_data) = fsai_eig_max_iters;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetFSAIKapTolerance( void      *data,
                                    NALU_HYPRE_Real fsai_kap_tolerance)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataFSAIKapTolerance(amg_data) = fsai_kap_tolerance;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetChebyOrder( void     *data,
                              NALU_HYPRE_Int       order)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (order < 1)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataChebyOrder(amg_data) = order;

   return nalu_hypre_error_flag;
}
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetChebyFraction( void     *data,
                                 NALU_HYPRE_Real  ratio)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (ratio <= 0.0 || ratio > 1.0 )
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataChebyFraction(amg_data) = ratio;

   return nalu_hypre_error_flag;
}
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetChebyEigEst( void     *data,
                               NALU_HYPRE_Int     cheby_eig_est)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (cheby_eig_est < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataChebyEigEst(amg_data) = cheby_eig_est;

   return nalu_hypre_error_flag;
}
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetChebyVariant( void     *data,
                                NALU_HYPRE_Int     cheby_variant)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataChebyVariant(amg_data) = cheby_variant;

   return nalu_hypre_error_flag;
}
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetChebyScale( void     *data,
                              NALU_HYPRE_Int     cheby_scale)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataChebyScale(amg_data) = cheby_scale;

   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGSetInterpVectors
 * -used for post-interpolation fitting of smooth vectors
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_BoomerAMGSetInterpVectors(void *solver,
                                          NALU_HYPRE_Int  num_vectors,
                                          nalu_hypre_ParVector **interp_vectors)

{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) solver;
   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGInterpVectors(amg_data) =  interp_vectors;
   nalu_hypre_ParAMGNumInterpVectors(amg_data) = num_vectors;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGSetInterpVectorValues
 * -used for post-interpolation fitting of smooth vectors
 *--------------------------------------------------------------------------*/

/*NALU_HYPRE_Int nalu_hypre_BoomerAMGSetInterpVectorValues(void *solver,
                                    NALU_HYPRE_Int  num_vectors,
                                    NALU_HYPRE_Complex *interp_vector_values)

{
   nalu_hypre_ParAMGData *amg_data = solver;
   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGInterpVectors(amg_data) =  interp_vectors;
   nalu_hypre_ParAMGNumInterpVectors(amg_data) = num_vectors;

   return nalu_hypre_error_flag;
}*/

NALU_HYPRE_Int nalu_hypre_BoomerAMGSetInterpVecVariant(void *solver,
                                             NALU_HYPRE_Int  var)


{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) solver;
   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (var < 1)
   {
      var = 0;
   }
   if (var > 3)
   {
      var = 3;
   }

   nalu_hypre_ParAMGInterpVecVariant(amg_data) = var;

   return nalu_hypre_error_flag;

}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetInterpVecQMax( void     *data,
                                 NALU_HYPRE_Int    q_max)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGInterpVecQMax(amg_data) = q_max;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetInterpVecAbsQTrunc( void     *data,
                                      NALU_HYPRE_Real    q_trunc)
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGInterpVecAbsQTrunc(amg_data) = q_trunc;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int nalu_hypre_BoomerAMGSetSmoothInterpVectors(void *solver,
                                                NALU_HYPRE_Int  smooth_interp_vectors)

{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) solver;
   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGSmoothInterpVectors(amg_data) = smooth_interp_vectors;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetInterpRefine( void     *data,
                                NALU_HYPRE_Int       num_refine )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGInterpRefine(amg_data) = num_refine;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetInterpVecFirstLevel( void     *data,
                                       NALU_HYPRE_Int  level )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGInterpVecFirstLevel(amg_data) = level;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetAdditive( void *data,
                            NALU_HYPRE_Int   additive )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataAdditive(amg_data) = additive;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetAdditive( void *data,
                            NALU_HYPRE_Int *  additive )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *additive = nalu_hypre_ParAMGDataAdditive(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetMultAdditive( void *data,
                                NALU_HYPRE_Int   mult_additive )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataMultAdditive(amg_data) = mult_additive;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetMultAdditive( void *data,
                                NALU_HYPRE_Int *  mult_additive )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *mult_additive = nalu_hypre_ParAMGDataMultAdditive(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetSimple( void *data,
                          NALU_HYPRE_Int   simple )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataSimple(amg_data) = simple;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetSimple( void *data,
                          NALU_HYPRE_Int *  simple )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *simple = nalu_hypre_ParAMGDataSimple(amg_data);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetAddLastLvl( void *data,
                              NALU_HYPRE_Int   add_last_lvl )
{
   nalu_hypre_ParAMGData  *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParAMGDataAddLastLvl(amg_data) = add_last_lvl;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetNonGalerkinTol( void   *data,
                                  NALU_HYPRE_Real nongalerkin_tol)
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) data;
   NALU_HYPRE_Int i, max_num_levels;
   NALU_HYPRE_Real *nongal_tol_array;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (nongalerkin_tol < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   max_num_levels = nalu_hypre_ParAMGDataMaxLevels(amg_data);
   nongal_tol_array = nalu_hypre_ParAMGDataNonGalTolArray(amg_data);

   if (nongal_tol_array == NULL)
   {
      nongal_tol_array = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  max_num_levels, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParAMGDataNonGalTolArray(amg_data) = nongal_tol_array;
   }
   nalu_hypre_ParAMGDataNonGalerkinTol(amg_data) = nongalerkin_tol;

   for (i = 0; i < max_num_levels; i++)
   {
      nongal_tol_array[i] = nongalerkin_tol;
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetLevelNonGalerkinTol( void   *data,
                                       NALU_HYPRE_Real   nongalerkin_tol,
                                       NALU_HYPRE_Int level)
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) data;
   NALU_HYPRE_Real *nongal_tol_array;
   NALU_HYPRE_Int max_num_levels;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (nongalerkin_tol < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   nongal_tol_array = nalu_hypre_ParAMGDataNonGalTolArray(amg_data);
   max_num_levels = nalu_hypre_ParAMGDataMaxLevels(amg_data);

   if (nongal_tol_array == NULL)
   {
      nongal_tol_array = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  max_num_levels, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParAMGDataNonGalTolArray(amg_data) = nongal_tol_array;
   }

   if (level + 1 > max_num_levels)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }

   nongal_tol_array[level] = nongalerkin_tol;
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetNonGalerkTol( void   *data,
                                NALU_HYPRE_Int   nongalerk_num_tol,
                                NALU_HYPRE_Real *nongalerk_tol)
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) data;

   nalu_hypre_ParAMGDataNonGalerkNumTol(amg_data) = nongalerk_num_tol;
   nalu_hypre_ParAMGDataNonGalerkTol(amg_data) = nongalerk_tol;
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetRAP2( void      *data,
                        NALU_HYPRE_Int  rap2 )
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) data;

   nalu_hypre_ParAMGDataRAP2(amg_data) = rap2;
   return nalu_hypre_error_flag;
}


NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetModuleRAP2( void      *data,
                              NALU_HYPRE_Int  mod_rap2 )
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) data;

   nalu_hypre_ParAMGDataModularizedMatMat(amg_data) = mod_rap2;
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetKeepTranspose( void       *data,
                                 NALU_HYPRE_Int   keepTranspose)
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) data;

   nalu_hypre_ParAMGDataKeepTranspose(amg_data) = keepTranspose;
   return nalu_hypre_error_flag;
}

#ifdef NALU_HYPRE_USING_DSUPERLU
NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetDSLUThreshold( void   *data,
                                 NALU_HYPRE_Int   dslu_threshold)
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) data;

   nalu_hypre_ParAMGDataDSLUThreshold(amg_data) = dslu_threshold;
   return nalu_hypre_error_flag;
}
#endif

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetCPoints(void         *data,
                          NALU_HYPRE_Int     cpt_coarse_level,
                          NALU_HYPRE_Int     num_cpt_coarse,
                          NALU_HYPRE_BigInt *cpt_coarse_index)
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) data;

   NALU_HYPRE_BigInt     *C_points_marker = NULL;
   NALU_HYPRE_Int        *C_points_local_marker = NULL;
   NALU_HYPRE_Int         cpt_level;

   if (!amg_data)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Warning! AMG object empty!\n");
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (cpt_coarse_level < 0)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Warning! cpt_coarse_level < 0 !\n");
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   if (num_cpt_coarse < 0)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Warning! num_cpt_coarse < 0 !\n");
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParAMGDataMemoryLocation(amg_data);

   /* free data not previously destroyed */
   if (nalu_hypre_ParAMGDataCPointsLevel(amg_data))
   {
      nalu_hypre_TFree(nalu_hypre_ParAMGDataCPointsMarker(amg_data), memory_location);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataCPointsLocalMarker(amg_data), memory_location);
   }

   /* set Cpoint data */
   if (nalu_hypre_ParAMGDataMaxLevels(amg_data) < cpt_coarse_level)
   {
      cpt_level = nalu_hypre_ParAMGDataNumLevels(amg_data);
   }
   else
   {
      cpt_level = cpt_coarse_level;
   }

   if (cpt_level)
   {
      C_points_marker = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cpt_coarse, memory_location);
      C_points_local_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cpt_coarse, memory_location);

      nalu_hypre_TMemcpy(C_points_marker, cpt_coarse_index, NALU_HYPRE_BigInt, num_cpt_coarse, memory_location,
                    NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_ParAMGDataCPointsMarker(amg_data)      = C_points_marker;
   nalu_hypre_ParAMGDataCPointsLocalMarker(amg_data) = C_points_local_marker;
   nalu_hypre_ParAMGDataNumCPoints(amg_data)         = num_cpt_coarse;
   nalu_hypre_ParAMGDataCPointsLevel(amg_data)       = cpt_level;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetFPoints(void         *data,
                          NALU_HYPRE_Int     isolated,
                          NALU_HYPRE_Int     num_points,
                          NALU_HYPRE_BigInt *indices)
{
   nalu_hypre_ParAMGData   *amg_data = (nalu_hypre_ParAMGData*) data;
   NALU_HYPRE_BigInt       *marker = NULL;
   NALU_HYPRE_Int           i;

   if (!amg_data)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "AMG object empty!\n");
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (num_points < 0)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Warning! negative number of points!\n");
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }


   if ((num_points > 0) && (!indices))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Warning! indices not given!\n");
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }

   /* Set marker data */
   if (num_points > 0)
   {
      marker = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_points, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_points; i++)
      {
         marker[i] = indices[i];
      }
   }

   if (isolated)
   {
      /* Free data not previously destroyed */
      if (nalu_hypre_ParAMGDataIsolatedFPointsMarker(amg_data))
      {
         nalu_hypre_TFree(nalu_hypre_ParAMGDataIsolatedFPointsMarker(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParAMGDataIsolatedFPointsMarker(amg_data) = NULL;
      }

      nalu_hypre_ParAMGDataNumIsolatedFPoints(amg_data)    = num_points;
      nalu_hypre_ParAMGDataIsolatedFPointsMarker(amg_data) = marker;
   }
   else
   {
      /* Free data not previously destroyed */
      if (nalu_hypre_ParAMGDataFPointsMarker(amg_data))
      {
         nalu_hypre_TFree(nalu_hypre_ParAMGDataFPointsMarker(amg_data), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParAMGDataFPointsMarker(amg_data) = NULL;
      }

      nalu_hypre_ParAMGDataNumFPoints(amg_data)    = num_points;
      nalu_hypre_ParAMGDataFPointsMarker(amg_data) = marker;
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSetCumNnzAP( void       *data,
                            NALU_HYPRE_Real  cum_nnz_AP )
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   nalu_hypre_ParAMGDataCumNnzAP(amg_data) = cum_nnz_AP;

   return nalu_hypre_error_flag;
}


NALU_HYPRE_Int
nalu_hypre_BoomerAMGGetCumNnzAP( void       *data,
                            NALU_HYPRE_Real *cum_nnz_AP )
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) data;

   if (!amg_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *cum_nnz_AP = nalu_hypre_ParAMGDataCumNnzAP(amg_data);

   return nalu_hypre_error_flag;
}
