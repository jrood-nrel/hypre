/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface (IJ_matrix interface).
 * Do `driver -help' for usage info.
 * This driver started from the driver for parcsr_linear_solvers, and it
 * works by first building a parcsr matrix as before and then "copying"
 * that matrix row-by-row into the IJMatrix interface. AJC 7/99.
 *--------------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_nalu_hypre_utilities.h"
#include "NALU_HYPRE.h"
#include "NALU_HYPRE_parcsr_mv.h"

#include "NALU_HYPRE_IJ_mv.h"
#include "_nalu_hypre_IJ_mv.h"
#include "NALU_HYPRE_parcsr_ls.h"
#include "_nalu_hypre_parcsr_mv.h"
#include "NALU_HYPRE_krylov.h"

#if defined (NALU_HYPRE_USING_CUDA)
#include <cuda_profiler_api.h>
#endif

/* begin lobpcg */

#define NO_SOLVER -9198

#include <time.h>

#include "NALU_HYPRE_lobpcg.h"

/* max dt */
#define DT_INF 1.0e30
NALU_HYPRE_Int
BuildParIsoLaplacian( NALU_HYPRE_Int argc, char** argv, NALU_HYPRE_ParCSRMatrix *A_ptr );

/* end lobpcg */

#ifdef __cplusplus
extern "C" {
#endif

NALU_HYPRE_Int BuildParFromFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                            NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int ReadParVectorFromFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                 NALU_HYPRE_ParVector *b_ptr );

NALU_HYPRE_Int BuildParLaplacian (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                             NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParSysLaplacian (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParDifConv (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                           NALU_HYPRE_ParCSRMatrix *A_ptr);
NALU_HYPRE_Int BuildParFromOneFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                               NALU_HYPRE_Int num_functions, NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildFuncsFromFiles (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                               NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_Int **dof_func_ptr );
NALU_HYPRE_Int BuildFuncsFromOneFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                 NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_Int **dof_func_ptr );
NALU_HYPRE_Int BuildRhsParFromOneFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                  NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_ParVector *b_ptr );
NALU_HYPRE_Int BuildBigArrayFromOneFile (NALU_HYPRE_Int argc, char *argv [], const char *array_name,
                                    NALU_HYPRE_Int arg_index, NALU_HYPRE_BigInt *partitioning, NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **array_ptr);
NALU_HYPRE_Int BuildParLaplacian9pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParLaplacian27pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                 NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParRotate7pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                             NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParVarDifConv (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                              NALU_HYPRE_ParCSRMatrix *A_ptr, NALU_HYPRE_ParVector *rhs_ptr );
NALU_HYPRE_ParCSRMatrix GenerateSysLaplacian (MPI_Comm comm, NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt ny,
                                         NALU_HYPRE_BigInt nz,
                                         NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Int r,
                                         NALU_HYPRE_Int num_fun, NALU_HYPRE_Real *mtrx, NALU_HYPRE_Real *value);
NALU_HYPRE_ParCSRMatrix GenerateSysLaplacianVCoef (MPI_Comm comm, NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt ny,
                                              NALU_HYPRE_BigInt nz,
                                              NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Int r,
                                              NALU_HYPRE_Int num_fun, NALU_HYPRE_Real *mtrx, NALU_HYPRE_Real *value);
NALU_HYPRE_Int SetSysVcoefValues(NALU_HYPRE_Int num_fun, NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt ny, NALU_HYPRE_BigInt nz,
                            NALU_HYPRE_Real vcx, NALU_HYPRE_Real vcy, NALU_HYPRE_Real vcz, NALU_HYPRE_Int mtx_entry, NALU_HYPRE_Real *values);

NALU_HYPRE_Int BuildParCoordinates (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                               NALU_HYPRE_Int *coorddim_ptr, float **coord_ptr );

extern NALU_HYPRE_Int nalu_hypre_FlexGMRESModifyPCAMGExample(void *precond_data, NALU_HYPRE_Int iterations,
                                                   NALU_HYPRE_Real rel_residual_norm);

extern NALU_HYPRE_Int nalu_hypre_FlexGMRESModifyPCDefault(void *precond_data, NALU_HYPRE_Int iteration,
                                                NALU_HYPRE_Real rel_residual_norm);
#ifdef __cplusplus
}
#endif

nalu_hypre_int
main( nalu_hypre_int argc,
      char *argv[] )
{
   NALU_HYPRE_Int           arg_index;
   NALU_HYPRE_Int           print_usage;
   NALU_HYPRE_Int           sparsity_known = 0;
   NALU_HYPRE_Int           add = 0;
   NALU_HYPRE_Int           check_constant = 0;
   NALU_HYPRE_Int           off_proc = 0;
   NALU_HYPRE_Int           chunk = 0;
   NALU_HYPRE_Int           omp_flag = 0;
   NALU_HYPRE_Int           build_matrix_type;
   NALU_HYPRE_Int           build_matrix_arg_index;
   NALU_HYPRE_Int           build_matrix_M;
   NALU_HYPRE_Int           build_matrix_M_arg_index;
   NALU_HYPRE_Int           build_rhs_type;
   NALU_HYPRE_Int           build_rhs_arg_index;
   NALU_HYPRE_Int           build_src_type;
   NALU_HYPRE_Int           build_src_arg_index;
   NALU_HYPRE_Int           build_x0_type;
   NALU_HYPRE_Int           build_x0_arg_index;
   NALU_HYPRE_Int           build_funcs_type;
   NALU_HYPRE_Int           build_funcs_arg_index;
   NALU_HYPRE_Int           build_fpt_arg_index;
   NALU_HYPRE_Int           build_sfpt_arg_index;
   NALU_HYPRE_Int           build_cpt_arg_index;
   NALU_HYPRE_Int           num_components = 1;
   NALU_HYPRE_Int           solver_id;
   NALU_HYPRE_Int           solver_type = 1;
   NALU_HYPRE_Int           recompute_res = 0;   /* What should be the default here? */
   NALU_HYPRE_Int           ioutdat;
   NALU_HYPRE_Int           poutdat;
   NALU_HYPRE_Int           poutusr = 0; /* if user selects pout */
   NALU_HYPRE_Int           debug_flag;
   NALU_HYPRE_Int           ierr = 0;
   NALU_HYPRE_Int           i, j, c;
   NALU_HYPRE_Int           max_levels = 25;
   NALU_HYPRE_Int           num_iterations;
   NALU_HYPRE_Int           pcg_num_its, dscg_num_its;
   NALU_HYPRE_Int           max_iter = 1000;
   NALU_HYPRE_Int           mg_max_iter = 100;
   NALU_HYPRE_Int           nodal = 0;
   NALU_HYPRE_Int           nodal_diag = 0;
   NALU_HYPRE_Int           keep_same_sign = 0;
   NALU_HYPRE_Real          cf_tol = 0.9;
   NALU_HYPRE_Real          norm;
   NALU_HYPRE_Real          b_dot_b;
   NALU_HYPRE_Real          final_res_norm;
   void               *object;

   NALU_HYPRE_IJMatrix      ij_A = NULL;
   NALU_HYPRE_IJMatrix      ij_M = NULL;
   NALU_HYPRE_IJVector      ij_b = NULL;
   NALU_HYPRE_IJVector      ij_x = NULL;
   NALU_HYPRE_IJVector      *ij_rbm = NULL;

   NALU_HYPRE_ParCSRMatrix  parcsr_A = NULL;
   NALU_HYPRE_ParCSRMatrix  parcsr_M = NULL;
   NALU_HYPRE_ParVector     b = NULL;
   NALU_HYPRE_ParVector     x = NULL;
   NALU_HYPRE_ParVector     *interp_vecs = NULL;
   NALU_HYPRE_ParVector     residual = NULL;
   NALU_HYPRE_ParVector     x0_save = NULL;

   NALU_HYPRE_Solver        amg_solver;
   NALU_HYPRE_Solver        amgdd_solver;
   NALU_HYPRE_Solver        pcg_solver;
   NALU_HYPRE_Solver        amg_precond = NULL;
   NALU_HYPRE_Solver        pcg_precond = NULL;
   NALU_HYPRE_Solver        pcg_precond_gotten;

   NALU_HYPRE_Int           check_residual = 0;
   NALU_HYPRE_Int           num_procs, myid;
   NALU_HYPRE_Int           local_row;
   NALU_HYPRE_Int          *row_sizes;
   NALU_HYPRE_Int          *diag_sizes;
   NALU_HYPRE_Int          *offdiag_sizes;
   NALU_HYPRE_BigInt       *rows;
   NALU_HYPRE_Int           size;
   NALU_HYPRE_Int          *ncols;
   NALU_HYPRE_BigInt       *col_inds;
   NALU_HYPRE_Int          *dof_func;
   NALU_HYPRE_Int           num_functions = 1;
   NALU_HYPRE_Int           num_paths = 1;
   NALU_HYPRE_Int           agg_num_levels = 0;
   NALU_HYPRE_Int           ns_coarse = 1, ns_down = -1, ns_up = -1;

   NALU_HYPRE_Int           time_index;
   MPI_Comm            comm = nalu_hypre_MPI_COMM_WORLD;
   NALU_HYPRE_BigInt        M, N, big_i;
   NALU_HYPRE_Int           local_num_rows, local_num_cols;
   NALU_HYPRE_BigInt        first_local_row, last_local_row;
   NALU_HYPRE_BigInt        first_local_col, last_local_col;
   NALU_HYPRE_BigInt       *partitioning = NULL;
   NALU_HYPRE_Int           variant, overlap, domain_type;
   NALU_HYPRE_Real          schwarz_rlx_weight;
   NALU_HYPRE_Real         *values, val;

   NALU_HYPRE_Int           use_nonsymm_schwarz = 0;
   NALU_HYPRE_Int           test_ij = 0;
   NALU_HYPRE_Int           test_multivec = 0;
   NALU_HYPRE_Int           build_rbm = 0;
   NALU_HYPRE_Int           build_rbm_index = 0;
   NALU_HYPRE_Int           num_interp_vecs = 0;
   NALU_HYPRE_Int           interp_vec_variant = 0;
   NALU_HYPRE_Int           Q_max = 0;
   NALU_HYPRE_Real          Q_trunc = 0;

   const NALU_HYPRE_Real    dt_inf = DT_INF;
   NALU_HYPRE_Real          dt = dt_inf;

   /* solve -Ax = b, for testing SND matrices */
   NALU_HYPRE_Int           negA = 0;

   /* parameters for BoomerAMG */
   NALU_HYPRE_Real     A_drop_tol = 0.0;
   NALU_HYPRE_Int      A_drop_type = -1;
   NALU_HYPRE_Int      coarsen_cut_factor = 0;
   NALU_HYPRE_Real     strong_threshold;
   NALU_HYPRE_Real     strong_thresholdR;
   NALU_HYPRE_Real     filter_thresholdR;
   NALU_HYPRE_Real     trunc_factor;
   NALU_HYPRE_Real     jacobi_trunc_threshold;
   NALU_HYPRE_Real     S_commpkg_switch = 1.0;
   NALU_HYPRE_Real     CR_rate = 0.7;
   NALU_HYPRE_Real     CR_strong_th = 0.0;
   NALU_HYPRE_Int      CR_use_CG = 0;
   NALU_HYPRE_Int      P_max_elmts = 4;
   NALU_HYPRE_Int      cycle_type;
   NALU_HYPRE_Int      fcycle;
   NALU_HYPRE_Int      coarsen_type = 10;
   NALU_HYPRE_Int      measure_type = 0;
   NALU_HYPRE_Int      num_sweeps = 1;
   NALU_HYPRE_Int      IS_type;
   NALU_HYPRE_Int      num_CR_relax_steps = 2;
   NALU_HYPRE_Int      relax_type = -1;
   NALU_HYPRE_Int      add_relax_type = 18;
   NALU_HYPRE_Int      relax_coarse = -1;
   NALU_HYPRE_Int      relax_up = -1;
   NALU_HYPRE_Int      relax_down = -1;
   NALU_HYPRE_Int      relax_order = 0;
   NALU_HYPRE_Int      level_w = -1;
   NALU_HYPRE_Int      level_ow = -1;
   /* NALU_HYPRE_Int    smooth_lev; */
   /* NALU_HYPRE_Int    smooth_rlx = 8; */
   NALU_HYPRE_Int      smooth_type = 6;
   NALU_HYPRE_Int      smooth_num_levels = 0;
   NALU_HYPRE_Int      smooth_num_sweeps = 1;
   NALU_HYPRE_Int      coarse_threshold = 9;
   NALU_HYPRE_Int      min_coarse_size = 0;
   /* redundant coarse grid solve */
   NALU_HYPRE_Int      seq_threshold = 0;
   NALU_HYPRE_Int      redundant = 0;
   /* additive versions */
   NALU_HYPRE_Int    additive = -1;
   NALU_HYPRE_Int    mult_add = -1;
   NALU_HYPRE_Int    simple = -1;
   NALU_HYPRE_Int    add_last_lvl = -1;
   NALU_HYPRE_Int    add_P_max_elmts = 0;
   NALU_HYPRE_Real   add_trunc_factor = 0;
   NALU_HYPRE_Int    rap2     = 0;
   NALU_HYPRE_Int    mod_rap2 = 0;
   NALU_HYPRE_Int    keepTranspose = 0;
#ifdef NALU_HYPRE_USING_DSUPERLU
   NALU_HYPRE_Int    dslu_threshold = -1;
#endif
   NALU_HYPRE_Real   relax_wt;
   NALU_HYPRE_Real   add_relax_wt = 1.0;
   NALU_HYPRE_Real   relax_wt_level;
   NALU_HYPRE_Real   outer_wt;
   NALU_HYPRE_Real   outer_wt_level;
   NALU_HYPRE_Real   tol = 1.e-8, pc_tol = 0.;
   NALU_HYPRE_Real   atol = 0.0;
   NALU_HYPRE_Real   max_row_sum = 1.;
   NALU_HYPRE_Int    converge_type = 0;
   NALU_HYPRE_Int    precon_cycles = 1;

   NALU_HYPRE_Int  cheby_order = 2;
   NALU_HYPRE_Int  cheby_eig_est = 10;
   NALU_HYPRE_Int  cheby_variant = 0;
   NALU_HYPRE_Int  cheby_scale = 1;
   NALU_HYPRE_Real cheby_fraction = .3;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_Int  spmv_use_vendor = 1;
   NALU_HYPRE_Int  use_curand = 1;
#if defined(NALU_HYPRE_USING_CUDA)
   NALU_HYPRE_Int  spgemm_use_vendor = 0;
#else
   NALU_HYPRE_Int  spgemm_use_vendor = 1;
#endif
   NALU_HYPRE_Int  spgemm_alg = 1;
   NALU_HYPRE_Int  spgemm_binned = 0;
   NALU_HYPRE_Int  spgemm_rowest_mtd = 3;
   NALU_HYPRE_Int  spgemm_rowest_nsamples = -1; /* default */
   NALU_HYPRE_Real spgemm_rowest_mult = -1.0; /* default */
#endif
   NALU_HYPRE_Int      nmv = 100;

   /* for CGC BM Aug 25, 2006 */
   NALU_HYPRE_Int      cgcits = 1;
   /* for coordinate plotting BM Oct 24, 2006 */
   NALU_HYPRE_Int      plot_grids = 0;
   NALU_HYPRE_Int      coord_dim  = 3;
   float         *coordinates = NULL;
   char           plot_file_name[256];

   /* parameters for ParaSAILS */
   NALU_HYPRE_Real   sai_threshold = 0.1;
   NALU_HYPRE_Real   sai_filter = 0.1;

   /* parameters for PILUT */
   NALU_HYPRE_Real   drop_tol = -1;
   NALU_HYPRE_Int    nonzeros_to_keep = -1;

   /* parameters for Euclid or ILU smoother in AMG */
   NALU_HYPRE_Real   eu_ilut = 0.0;
   NALU_HYPRE_Real   eu_sparse_A = 0.0;
   NALU_HYPRE_Int    eu_bj = 0;
   NALU_HYPRE_Int    eu_level = -1;
   NALU_HYPRE_Int    eu_stats = 0;
   NALU_HYPRE_Int    eu_mem = 0;
   NALU_HYPRE_Int    eu_row_scale = 0; /* Euclid only */

   /* parameters for GMRES */
   NALU_HYPRE_Int    k_dim;
   /* parameters for COGMRES */
   NALU_HYPRE_Int    cgs = 1;
   NALU_HYPRE_Int    unroll = 0;
   /* parameters for LGMRES */
   NALU_HYPRE_Int    aug_dim;
   /* parameters for GSMG */
   NALU_HYPRE_Int    gsmg_samples = 5;
   /* interpolation */
   NALU_HYPRE_Int    interp_type  = 6; /* default value */
   NALU_HYPRE_Int    post_interp_type  = 0; /* default value */
   /* RL: restriction */
   NALU_HYPRE_Int    restri_type = 0;
   /* aggressive coarsening */
   NALU_HYPRE_Int    agg_interp_type  = 4; /* default value */
   NALU_HYPRE_Int    agg_P_max_elmts  = 0; /* default value */
   NALU_HYPRE_Int    agg_P12_max_elmts  = 0; /* default value */
   NALU_HYPRE_Real   agg_trunc_factor  = 0; /* default value */
   NALU_HYPRE_Real   agg_P12_trunc_factor  = 0; /* default value */

   NALU_HYPRE_Int    print_system = 0;
   NALU_HYPRE_Int    rel_change = 0;
   NALU_HYPRE_Int    second_time = 0;
   NALU_HYPRE_Int    benchmark = 0;

   /* begin lobpcg */
   NALU_HYPRE_Int    hybrid = 1;
   NALU_HYPRE_Int    num_sweep = 1;
   NALU_HYPRE_Int    relax_default = 3;

   NALU_HYPRE_Int  lobpcgFlag = 0;
   NALU_HYPRE_Int  lobpcgGen = 0;
   NALU_HYPRE_Int  constrained = 0;
   NALU_HYPRE_Int  vFromFileFlag = 0;
   NALU_HYPRE_Int  lobpcgSeed = 0;
   NALU_HYPRE_Int  blockSize = 1;
   NALU_HYPRE_Int  verbosity = 1;
   NALU_HYPRE_Int  iterations;
   NALU_HYPRE_Int  maxIterations = 100;
   NALU_HYPRE_Int  checkOrtho = 0;
   NALU_HYPRE_Int  printLevel = 0; /* also c.f. poutdat */
   NALU_HYPRE_Int  two_norm = 1;
   NALU_HYPRE_Int  pcgIterations = 0;
   NALU_HYPRE_Int  pcgMode = 1;
   NALU_HYPRE_Real pcgTol = 1e-2;
   NALU_HYPRE_Real nonOrthF;

   FILE* filePtr;

   mv_MultiVectorPtr eigenvectors = NULL;
   mv_MultiVectorPtr constraints = NULL;
   mv_MultiVectorPtr workspace = NULL;

   NALU_HYPRE_Real* eigenvalues = NULL;

   NALU_HYPRE_Real* residuals;
   utilities_FortranMatrix* residualNorms;
   utilities_FortranMatrix* residualNormsHistory;
   utilities_FortranMatrix* eigenvaluesHistory;
   utilities_FortranMatrix* printBuffer;
   utilities_FortranMatrix* gramXX;
   utilities_FortranMatrix* identity;

   NALU_HYPRE_Solver        lobpcg_solver;

   mv_InterfaceInterpreter* interpreter;
   NALU_HYPRE_MatvecFunctions matvec_fn;

   NALU_HYPRE_IJMatrix      ij_B;
   NALU_HYPRE_ParCSRMatrix  parcsr_B;

   /* end lobpcg */

   /* mgr options */
   NALU_HYPRE_Int mgr_bsize = 1;
   NALU_HYPRE_Int mgr_nlevels = 0;
   NALU_HYPRE_Int mgr_num_reserved_nodes = 0;
   NALU_HYPRE_Int mgr_non_c_to_f = 1;
   NALU_HYPRE_Int mgr_frelax_method = 0;
   NALU_HYPRE_Int *mgr_num_cindexes = NULL;
   NALU_HYPRE_Int **mgr_cindexes = NULL;
   NALU_HYPRE_BigInt *mgr_reserved_coarse_indexes = NULL;
   NALU_HYPRE_Int mgr_relax_type = 0;
   NALU_HYPRE_Int mgr_num_relax_sweeps = 2;
   NALU_HYPRE_Int mgr_interp_type = 2;
   NALU_HYPRE_Int mgr_num_interp_sweeps = 2;
   NALU_HYPRE_Int mgr_gsmooth_type = 0;
   NALU_HYPRE_Int mgr_num_gsmooth_sweeps = 1;
   NALU_HYPRE_Int mgr_restrict_type = 0;
   NALU_HYPRE_Int mgr_num_restrict_sweeps = 0;
   /* end mgr options */

   /* nalu_hypre_ILU options */
   NALU_HYPRE_Int ilu_type = 0;
   NALU_HYPRE_Int ilu_lfil = 0;
   NALU_HYPRE_Int ilu_sm_max_iter = 1;
   NALU_HYPRE_Real ilu_droptol = 1.0e-02;
   NALU_HYPRE_Int ilu_max_row_nnz = 1000;
   NALU_HYPRE_Int ilu_schur_max_iter = 3;
   NALU_HYPRE_Real ilu_nsh_droptol = 1.0e-02;
   /* end hypre ILU options */

   /* nalu_hypre_FSAI options */
   NALU_HYPRE_Int  fsai_algo_type = 1;
   NALU_HYPRE_Int  fsai_max_steps = 10;
   NALU_HYPRE_Int  fsai_max_step_size = 1;
   NALU_HYPRE_Int  fsai_eig_max_iters = 5;
   NALU_HYPRE_Real fsai_kap_tolerance = 1.0e-03;
   /* end hypre FSAI options */

   NALU_HYPRE_Real     *nongalerk_tol = NULL;
   NALU_HYPRE_Int       nongalerk_num_tol = 0;

   /* coasening data */
   NALU_HYPRE_Int     num_cpt = 0;
   NALU_HYPRE_Int     num_fpt = 0;
   NALU_HYPRE_Int     num_isolated_fpt = 0;
   NALU_HYPRE_BigInt *cpt_index = NULL;
   NALU_HYPRE_BigInt *fpt_index = NULL;
   NALU_HYPRE_BigInt *isolated_fpt_index = NULL;

   NALU_HYPRE_BigInt *row_nums = NULL;
   NALU_HYPRE_Int *num_cols = NULL;
   NALU_HYPRE_BigInt *col_nums = NULL;
   NALU_HYPRE_Int i_indx, j_indx, num_rows;
   NALU_HYPRE_Real *data = NULL;

   NALU_HYPRE_Int air = 0;
   NALU_HYPRE_Int **grid_relax_points = NULL;

   /* amg-dd options */
   NALU_HYPRE_Int amgdd_start_level = 0;
   NALU_HYPRE_Int amgdd_padding = 1;
   NALU_HYPRE_Int amgdd_fac_num_relax = 1;
   NALU_HYPRE_Int amgdd_num_comp_cycles = 2;
   NALU_HYPRE_Int amgdd_fac_relax_type = 3;
   NALU_HYPRE_Int amgdd_fac_cycle_type = 1;
   NALU_HYPRE_Int amgdd_num_ghost_layers = 1;

#if defined(NALU_HYPRE_USING_MEMORY_TRACKER)
   NALU_HYPRE_Int print_mem_tracker = 0;
   char mem_tracker_name[NALU_HYPRE_MAX_FILE_NAME_LEN] = {0};
#endif

   /* default execution policy and memory space */
#if defined(NALU_HYPRE_TEST_USING_HOST)
   NALU_HYPRE_MemoryLocation memory_location = NALU_HYPRE_MEMORY_HOST;
   NALU_HYPRE_ExecutionPolicy default_exec_policy = NALU_HYPRE_EXEC_HOST;
   NALU_HYPRE_ExecutionPolicy exec2_policy = NALU_HYPRE_EXEC_HOST;
#else
   NALU_HYPRE_MemoryLocation memory_location = NALU_HYPRE_MEMORY_DEVICE;
   NALU_HYPRE_ExecutionPolicy default_exec_policy = NALU_HYPRE_EXEC_DEVICE;
   NALU_HYPRE_ExecutionPolicy exec2_policy = NALU_HYPRE_EXEC_DEVICE;
#endif

   for (arg_index = 1; arg_index < argc; arg_index ++)
   {
      if ( strcmp(argv[arg_index], "-memory_host") == 0 )
      {
         memory_location = NALU_HYPRE_MEMORY_HOST;
      }
      else if ( strcmp(argv[arg_index], "-memory_device") == 0 )
      {
         memory_location = NALU_HYPRE_MEMORY_DEVICE;
      }
      else if ( strcmp(argv[arg_index], "-exec_host") == 0 )
      {
         default_exec_policy = NALU_HYPRE_EXEC_HOST;
      }
      else if ( strcmp(argv[arg_index], "-exec_device") == 0 )
      {
         default_exec_policy = NALU_HYPRE_EXEC_DEVICE;
      }
      else if ( strcmp(argv[arg_index], "-exec2_host") == 0 )
      {
         exec2_policy = NALU_HYPRE_EXEC_HOST;
      }
      else if ( strcmp(argv[arg_index], "-exec2_device") == 0 )
      {
         exec2_policy = NALU_HYPRE_EXEC_DEVICE;
      }
   }

   if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
   {
      keepTranspose = 1;
      coarsen_type  = 8;
      mod_rap2      = 1;
   }

#ifdef NALU_HYPRE_USING_DEVICE_POOL
   /* device pool allocator */
   nalu_hypre_uint mempool_bin_growth   = 8,
              mempool_min_bin      = 3,
              mempool_max_bin      = 9;
   size_t mempool_max_cached_bytes = 2000LL * 1024 * 1024;
#endif

   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/
   build_matrix_type = 2;
   build_matrix_arg_index = argc;
   build_matrix_M = 0;
   build_matrix_M_arg_index = argc;
   build_rhs_type = 2;
   build_rhs_arg_index = argc;
   build_src_type = -1;
   build_src_arg_index = argc;
   build_x0_type = -1;
   build_x0_arg_index = argc;
   build_funcs_type = 0;
   build_funcs_arg_index = argc;
   build_fpt_arg_index = 0;
   build_sfpt_arg_index = 0;
   build_cpt_arg_index = 0;
   IS_type = 1;
   debug_flag = 0;
   solver_id = 0;
   ioutdat = 3;
   poutdat = 1;
   nalu_hypre_sprintf (plot_file_name, "AMGgrids.CF.dat");

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   print_usage = 0;
   arg_index = 1;

   while ( (arg_index < argc) && (!print_usage) )
   {
      if ( strcmp(argv[arg_index], "-fromfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = -1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-auxfromfile") == 0 )
      {
         arg_index++;
         build_matrix_M           = 1;
         build_matrix_M_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromparcsrfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 0;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-fromonecsrfile") == 0 )
      {
         arg_index++;
         build_matrix_type      = 1;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-laplacian") == 0 )
      {
         arg_index++;
         build_matrix_type      = 2;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-9pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 3;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-27pt") == 0 )
      {
         arg_index++;
         build_matrix_type      = 4;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-difconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 5;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-vardifconv") == 0 )
      {
         arg_index++;
         build_matrix_type      = 6;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rotate") == 0 )
      {
         arg_index++;
         build_matrix_type      = 7;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-test_ij") == 0 )
      {
         arg_index++;
         test_ij = 1;
      }
      else if ( strcmp(argv[arg_index], "-test_multivec") == 0 )
      {
         arg_index++;
         test_multivec = 1;
      }
      else if ( strcmp(argv[arg_index], "-funcsfromonefile") == 0 )
      {
         arg_index++;
         build_funcs_type      = 1;
         build_funcs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-funcsfromfile") == 0 )
      {
         arg_index++;
         build_funcs_type      = 2;
         build_funcs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-exact_size") == 0 )
      {
         arg_index++;
         sparsity_known = 1;
      }
      else if ( strcmp(argv[arg_index], "-storage_low") == 0 )
      {
         arg_index++;
         sparsity_known = 2;
      }
      else if ( strcmp(argv[arg_index], "-add") == 0 )
      {
         arg_index++;
         add = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-chunk") == 0 )
      {
         arg_index++;
         chunk = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-off_proc") == 0 )
      {
         arg_index++;
         off_proc = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-omp") == 0 )
      {
         arg_index++;
         omp_flag = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-check_constant") == 0 )
      {
         arg_index++;
         check_constant = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-concrete_parcsr") == 0 )
      {
         arg_index++;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         /* begin lobpcg */
         if ( strcmp(argv[arg_index], "none") == 0 )
         {
            solver_id = NO_SOLVER;
            arg_index++;
         }
         else /* end lobpcg */
         {
            solver_id = atoi(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-rbm") == 0 )
      {
         arg_index++;
         build_rbm      = 1;
         num_interp_vecs = atoi(argv[arg_index++]);
         build_rbm_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-nc") == 0 )
      {
         arg_index++;
         num_components = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rhsfromfile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 0;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsfromonefile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 1;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsparcsrfile") == 0 )
      {
         arg_index++;
         build_rhs_type      = 7;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsisone") == 0 )
      {
         arg_index++;
         build_rhs_type      = 2;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhsrand") == 0 )
      {
         arg_index++;
         build_rhs_type      = 3;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-xisone") == 0 )
      {
         arg_index++;
         build_rhs_type      = 4;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-rhszero") == 0 )
      {
         arg_index++;
         build_rhs_type      = 5;
         build_rhs_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcfromfile") == 0 )
      {
         arg_index++;
         build_src_type      = 0;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcfromonefile") == 0 )
      {
         arg_index++;
         build_src_type      = 1;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcisone") == 0 )
      {
         arg_index++;
         build_src_type      = 2;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srcrand") == 0 )
      {
         arg_index++;
         build_src_type      = 3;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-srczero") == 0 )
      {
         arg_index++;
         build_src_type      = 4;
         build_rhs_type      = -1;
         build_src_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-x0fromfile") == 0 )
      {
         arg_index++;
         build_x0_type       = 0;
         build_x0_arg_index  = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-x0parcsrfile") == 0 )
      {
         arg_index++;
         build_x0_type      = 7;
         build_x0_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-x0rand") == 0 )
      {
         arg_index++;
         build_x0_type       = 1;
         build_x0_arg_index  = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-CFfromfile") == 0 )
      {
         arg_index++;
         coarsen_type      = 999;
      }
      else if ( strcmp(argv[arg_index], "-Ffromonefile") == 0 )
      {
         arg_index++;
         build_fpt_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-SFfromonefile") == 0 )
      {
         arg_index++;
         build_sfpt_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-Cfromonefile") == 0 )
      {
         arg_index++;
         build_cpt_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-cljp") == 0 )
      {
         arg_index++;
         coarsen_type      = 0;
      }
      else if ( strcmp(argv[arg_index], "-cljp1") == 0 )
      {
         arg_index++;
         coarsen_type      = 7;
      }
      else if ( strcmp(argv[arg_index], "-cgc") == 0 )
      {
         arg_index++;
         coarsen_type      = 21;
         cgcits            = 200;
      }
      else if ( strcmp(argv[arg_index], "-cgce") == 0 )
      {
         arg_index++;
         coarsen_type      = 22;
         cgcits            = 200;
      }
      else if ( strcmp(argv[arg_index], "-pmis") == 0 )
      {
         arg_index++;
         coarsen_type      = 8;
      }
      else if ( strcmp(argv[arg_index], "-pmis1") == 0 )
      {
         arg_index++;
         coarsen_type      = 9;
      }
      else if ( strcmp(argv[arg_index], "-cr1") == 0 )
      {
         arg_index++;
         coarsen_type      = 98;
      }
      else if ( strcmp(argv[arg_index], "-cr") == 0 )
      {
         arg_index++;
         coarsen_type      = 99;
      }
      else if ( strcmp(argv[arg_index], "-crcg") == 0 )
      {
         arg_index++;
         CR_use_CG = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-hmis") == 0 )
      {
         arg_index++;
         coarsen_type      = 10;
      }
      else if ( strcmp(argv[arg_index], "-ruge") == 0 )
      {
         arg_index++;
         coarsen_type      = 1;
      }
      else if ( strcmp(argv[arg_index], "-ruge1p") == 0 )
      {
         arg_index++;
         coarsen_type      = 11;
      }
      else if ( strcmp(argv[arg_index], "-ruge2b") == 0 )
      {
         arg_index++;
         coarsen_type      = 2;
      }
      else if ( strcmp(argv[arg_index], "-ruge3") == 0 )
      {
         arg_index++;
         coarsen_type      = 3;
      }
      else if ( strcmp(argv[arg_index], "-ruge3c") == 0 )
      {
         arg_index++;
         coarsen_type      = 4;
      }
      else if ( strcmp(argv[arg_index], "-rugerlx") == 0 )
      {
         arg_index++;
         coarsen_type      = 5;
      }
      else if ( strcmp(argv[arg_index], "-falgout") == 0 )
      {
         arg_index++;
         coarsen_type      = 6;
      }
      else if ( strcmp(argv[arg_index], "-gm") == 0 )
      {
         arg_index++;
         measure_type      = 1;
      }
      else if ( strcmp(argv[arg_index], "-is") == 0 )
      {
         arg_index++;
         IS_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ncr") == 0 )
      {
         arg_index++;
         num_CR_relax_steps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-crth") == 0 )
      {
         arg_index++;
         CR_rate = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-crst") == 0 )
      {
         arg_index++;
         CR_strong_th = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rlx") == 0 )
      {
         arg_index++;
         relax_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rlx_coarse") == 0 )
      {
         arg_index++;
         relax_coarse = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rlx_down") == 0 )
      {
         arg_index++;
         relax_down = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rlx_up") == 0 )
      {
         arg_index++;
         relax_up = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-smtype") == 0 )
      {
         arg_index++;
         smooth_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-smlv") == 0 )
      {
         arg_index++;
         smooth_num_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxl") == 0 )
      {
         arg_index++;
         max_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dbg") == 0 )
      {
         arg_index++;
         debug_flag = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nf") == 0 )
      {
         arg_index++;
         num_functions = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_nl") == 0 )
      {
         arg_index++;
         agg_num_levels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-npaths") == 0 )
      {
         arg_index++;
         num_paths = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ns") == 0 )
      {
         arg_index++;
         num_sweeps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ns_coarse") == 0 )
      {
         arg_index++;
         ns_coarse = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ns_down") == 0 )
      {
         arg_index++;
         ns_down = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ns_up") == 0 )
      {
         arg_index++;
         ns_up = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sns") == 0 )
      {
         arg_index++;
         smooth_num_sweeps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-max_iter") == 0 )
      {
         arg_index++;
         max_iter = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mg_max_iter") == 0 )
      {
         arg_index++;
         mg_max_iter = atoi(argv[arg_index++]);
      }

      else if ( strcmp(argv[arg_index], "-dt") == 0 )
      {
         arg_index++;
         dt = atof(argv[arg_index++]);
         build_rhs_type = -1;
         if ( build_src_type == -1 ) { build_src_type = 2; }
      }
      else if ( strcmp(argv[arg_index], "-restritype") == 0 )
      {
         arg_index++;
         restri_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
      }
      /* begin lobpcg */
      else if ( strcmp(argv[arg_index], "-lobpcg") == 0 )
      {
         /* use lobpcg */
         arg_index++;
         lobpcgFlag = 1;
      }
      else if ( strcmp(argv[arg_index], "-gen") == 0 )
      {
         /* generalized evp */
         arg_index++;
         lobpcgGen = 1;
      }
      else if ( strcmp(argv[arg_index], "-con") == 0 )
      {
         /* constrained evp */
         arg_index++;
         constrained = 1;
      }
      else if ( strcmp(argv[arg_index], "-orthchk") == 0 )
      {
         /* lobpcg: check orthonormality */
         arg_index++;
         checkOrtho = 1;
      }
      else if ( strcmp(argv[arg_index], "-vfromfile") == 0 )
      {
         /* lobpcg: get initial vectors from file */
         arg_index++;
         vFromFileFlag = 1;
      }
      else if ( strcmp(argv[arg_index], "-vrand") == 0 )
      {
         /* lobpcg: block size */
         arg_index++;
         blockSize = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-seed") == 0 )
      {
         /* lobpcg: seed for srand */
         arg_index++;
         lobpcgSeed = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-itr") == 0 )
      {
         /* lobpcg: max # of iterations */
         arg_index++;
         maxIterations = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-verb") == 0 )
      {
         /* lobpcg: verbosity level */
         arg_index++;
         verbosity = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vout") == 0 )
      {
         /* lobpcg: print level */
         arg_index++;
         printLevel = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgitr") == 0 )
      {
         /* lobpcg: inner pcg iterations */
         arg_index++;
         pcgIterations = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgtol") == 0 )
      {
         /* lobpcg: inner pcg iterations */
         arg_index++;
         pcgTol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pcgmode") == 0 )
      {
         /* lobpcg: initial guess for inner pcg */
         arg_index++;        /* 0: zero, otherwise rhs */
         pcgMode = atoi(argv[arg_index++]);
      }
      /* end lobpcg */
      /* begin mgr options*/
      else if ( strcmp(argv[arg_index], "-mgr_bsize") == 0 )
      {
         /* mgr block size */
         arg_index++;
         mgr_bsize = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mgr_nlevels") == 0 )
      {
         /* mgr number of coarsening levels */
         arg_index++;
         mgr_nlevels = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mgr_non_c_to_f") == 0 )
      {
         /* mgr intermediate coarse grid strategy */
         arg_index++;
         mgr_non_c_to_f = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mgr_num_reserved_nodes") == 0 )
      {
         /* mgr number of reserved nodes to be put on coarsest grid */
         arg_index++;
         mgr_num_reserved_nodes = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mgr_frelax_method") == 0 )
      {
         /* mgr F-relaxation strategy: single/ multi level */
         arg_index++;
         mgr_frelax_method = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mgr_relax_type") == 0 )
      {
         /* relax type for "single level" F-relaxation */
         arg_index++;
         mgr_relax_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mgr_relax_sweeps") == 0 )
      {
         /* number of relaxation sweeps */
         arg_index++;
         mgr_num_relax_sweeps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mgr_interp_type") == 0 )
      {
         /* interpolation type */
         arg_index++;
         mgr_interp_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mgr_interp_sweeps") == 0 )
      {
         /* number of interpolation sweeps*/
         arg_index++;
         mgr_num_interp_sweeps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mgr_gsmooth_type") == 0 )
      {
         /* global smoother type */
         arg_index++;
         mgr_gsmooth_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mgr_gsmooth_sweeps") == 0 )
      {
         /* number of global smooth sweeps*/
         arg_index++;
         mgr_num_gsmooth_sweeps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mgr_restrict_type") == 0 )
      {
         /* restriction type */
         arg_index++;
         mgr_restrict_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mgr_restrict_sweeps") == 0 )
      {
         /* number of restriction sweeps*/
         arg_index++;
         mgr_num_restrict_sweeps = atoi(argv[arg_index++]);
      }
      /* end mgr options */
      /* begin ilu options*/
      else if ( strcmp(argv[arg_index], "-ilu_type") == 0 )
      {
         /* ilu_type */
         arg_index++;
         ilu_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ilu_sm_max_iter") == 0 )
      {
         /* number of iteration when applied as a smoother */
         arg_index++;
         ilu_sm_max_iter = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ilu_lfil") == 0 )
      {
         /* level of fill */
         arg_index++;
         ilu_lfil = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ilu_droptol") == 0 )
      {
         /* drop tolerance */
         arg_index++;
         ilu_droptol = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ilu_max_row_nnz") == 0 )
      {
         /* Max number of nonzeros to keep per row */
         arg_index++;
         ilu_max_row_nnz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ilu_schur_max_iter") == 0 )
      {
         /* Max number of iterations for schur system solver */
         arg_index++;
         ilu_schur_max_iter = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ilu_nsh_droptol") == 0 )
      {
         /* Max number of iterations for schur system solver */
         arg_index++;
         ilu_nsh_droptol = atof(argv[arg_index++]);
      }
      /* end ilu options */
      /* begin FSAI options*/
      else if ( strcmp(argv[arg_index], "-fs_algo_type") == 0 )
      {
         arg_index++;
         fsai_algo_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-fs_max_steps") == 0 )
      {
         arg_index++;
         fsai_max_steps = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-fs_max_step_size") == 0 )
      {
         arg_index++;
         fsai_max_step_size = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-fs_eig_max_iters") == 0 )
      {
         arg_index++;
         fsai_eig_max_iters = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-fs_kap_tol") == 0 )
      {
         arg_index++;
         fsai_kap_tolerance = atof(argv[arg_index++]);
      }
      /* end FSAI options */
#if defined(NALU_HYPRE_USING_GPU)
      else if ( strcmp(argv[arg_index], "-mm_vendor") == 0 )
      {
         arg_index++;
         spgemm_use_vendor = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nmv") == 0 )
      {
         arg_index++;
         nmv = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mv_vendor") == 0 )
      {
         arg_index++;
         spmv_use_vendor = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-spgemm_alg") == 0 )
      {
         arg_index++;
         spgemm_alg  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-spgemm_binned") == 0 )
      {
         arg_index++;
         spgemm_binned  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-spgemm_rowest") == 0 )
      {
         arg_index++;
         spgemm_rowest_mtd  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-spgemm_rowestmult") == 0 )
      {
         arg_index++;
         spgemm_rowest_mult  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-spgemm_rowestnsamples") == 0 )
      {
         arg_index++;
         spgemm_rowest_nsamples  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-use_curand") == 0 )
      {
         arg_index++;
         use_curand = atoi(argv[arg_index++]);
      }
#endif
#ifdef NALU_HYPRE_USING_DEVICE_POOL
      else if ( strcmp(argv[arg_index], "-mempool_growth") == 0 )
      {
         arg_index++;
         mempool_bin_growth = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mempool_minbin") == 0 )
      {
         arg_index++;
         mempool_min_bin = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mempool_maxbin") == 0 )
      {
         arg_index++;
         mempool_max_bin = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mempool_maxcached") == 0 )
      {
         // Give maximum cached in Mbytes.
         arg_index++;
         mempool_max_cached_bytes = atoi(argv[arg_index++]) * 1024LL * 1024LL;
      }
#endif
      else if ( strcmp(argv[arg_index], "-negA") == 0 )
      {
         arg_index++;
         negA = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-second_time") == 0 )
      {
         arg_index++;
         second_time = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-benchmark") == 0 )
      {
         arg_index++;
         benchmark = atoi(argv[arg_index++]);
      }
#if defined(NALU_HYPRE_USING_MEMORY_TRACKER)
      else if ( strcmp(argv[arg_index], "-print_mem_tracker") == 0 )
      {
         arg_index++;
         print_mem_tracker = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mem_tracker_filename") == 0 )
      {
         arg_index++;
         snprintf(mem_tracker_name, NALU_HYPRE_MAX_FILE_NAME_LEN, "%s", argv[arg_index++]);
      }
#endif
      else
      {
         arg_index++;
      }
   }

   /* begin CGC BM Aug 25, 2006 */
   if (coarsen_type == 21 || coarsen_type == 22)
   {
      arg_index = 0;
      while ( (arg_index < argc) && (!print_usage) )
      {
         if ( strcmp(argv[arg_index], "-cgcits") == 0 )
         {
            arg_index++;
            cgcits = atoi(argv[arg_index++]);
         }
         else
         {
            arg_index++;
         }
      }
   }

   /* begin lobpcg */

   if ( solver_id == 0 && lobpcgFlag )
   {
      solver_id = 1;
   }

   /* end lobpcg */

   if (solver_id == 8 || solver_id == 18)
   {
      max_levels = 1;
   }

   /* defaults for BoomerAMG */
   if (solver_id == 0 || solver_id == 1 || solver_id == 3 || solver_id == 5
       || solver_id == 9 || solver_id == 13 || solver_id == 14
       || solver_id == 15 || solver_id == 20 || solver_id == 51 || solver_id == 61
       || solver_id == 16
       || solver_id == 70 || solver_id == 71 || solver_id == 72
       || solver_id == 90 || solver_id == 91)
   {
      strong_threshold = 0.25;
      strong_thresholdR = 0.25;
      filter_thresholdR = 0.00;
      trunc_factor = 0.;
      jacobi_trunc_threshold = 0.01;
      cycle_type = 1;
      fcycle = 0;
      relax_wt = 1.;
      outer_wt = 1.;

      /* for CGNR preconditioned with Boomeramg, only relaxation scheme 0 is
         implemented, i.e. Jacobi relaxation, and needs to be used without CF
         ordering */
      if (solver_id == 5)
      {
         relax_type = 0;
         relax_order = 0;
      }
   }

   /* defaults for Schwarz */
   variant = 0;  /* multiplicative */
   overlap = 1;  /* 1 layer overlap */
   domain_type = 2; /* through agglomeration */
   schwarz_rlx_weight = 1.;

   /* defaults for GMRES */
   k_dim = 5;
   cgs = 1;
   unroll = 0;

   /* defaults for LGMRES - should use a larger k_dim, though*/
   aug_dim = 2;

   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-k") == 0 )
      {
         arg_index++;
         k_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cgs") == 0 )
      {
         arg_index++;
         cgs = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-unroll") == 0 )
      {
         arg_index++;
         unroll = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-check_residual") == 0 )
      {
         arg_index++;
         check_residual = 1;
      }
      else if ( strcmp(argv[arg_index], "-aug") == 0 )
      {
         arg_index++;
         aug_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-w") == 0 )
      {
         arg_index++;
         relax_wt = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-wl") == 0 )
      {
         arg_index++;
         relax_wt_level = atof(argv[arg_index++]);
         level_w = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ow") == 0 )
      {
         arg_index++;
         outer_wt = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-owl") == 0 )
      {
         arg_index++;
         outer_wt_level = atof(argv[arg_index++]);
         level_ow = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sw") == 0 )
      {
         arg_index++;
         schwarz_rlx_weight = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-coarse_th") == 0 )
      {
         arg_index++;
         coarse_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-adroptol") == 0 )
      {
         arg_index++;
         A_drop_tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-adroptype") == 0 )
      {
         arg_index++;
         A_drop_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-min_cs") == 0 )
      {
         arg_index++;
         min_coarse_size  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-seq_th") == 0 )
      {
         arg_index++;
         seq_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-red") == 0 )
      {
         arg_index++;
         redundant  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cutf") == 0 )
      {
         arg_index++;
         coarsen_cut_factor = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-th") == 0 )
      {
         arg_index++;
         strong_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-thR") == 0 )
      {
         arg_index++;
         strong_thresholdR  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-fltr_thR") == 0 )
      {
         arg_index++;
         filter_thresholdR  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-CF") == 0 )
      {
         arg_index++;
         relax_order = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cf") == 0 )
      {
         arg_index++;
         cf_tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-conv_type") == 0 )
      {
         arg_index++;
         converge_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-atol") == 0 )
      {
         arg_index++;
         atol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxrs") == 0 )
      {
         arg_index++;
         max_row_sum  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sai_th") == 0 )
      {
         arg_index++;
         sai_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sai_filt") == 0 )
      {
         arg_index++;
         sai_filter  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-drop_tol") == 0 )
      {
         arg_index++;
         drop_tol  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nonzeros_to_keep") == 0 )
      {
         arg_index++;
         nonzeros_to_keep  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ilut") == 0 )
      {
         arg_index++;
         eu_ilut  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sparseA") == 0 )
      {
         arg_index++;
         eu_sparse_A  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rowScale") == 0 )
      {
         arg_index++;
         eu_row_scale  = 1;
      }
      else if ( strcmp(argv[arg_index], "-level") == 0 )
      {
         arg_index++;
         eu_level  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-bj") == 0 )
      {
         arg_index++;
         eu_bj  = 1;
      }
      else if ( strcmp(argv[arg_index], "-eu_stats") == 0 )
      {
         arg_index++;
         eu_stats  = 1;
      }
      else if ( strcmp(argv[arg_index], "-eu_mem") == 0 )
      {
         arg_index++;
         eu_mem  = 1;
      }
      else if ( strcmp(argv[arg_index], "-tr") == 0 )
      {
         arg_index++;
         trunc_factor  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-Pmx") == 0 )
      {
         arg_index++;
         P_max_elmts  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-interpvecvar") == 0 )
      {
         arg_index++;
         interp_vec_variant  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-Qtr") == 0 )
      {
         arg_index++;
         Q_trunc  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-Qmx") == 0 )
      {
         arg_index++;
         Q_max = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-jtr") == 0 )
      {
         arg_index++;
         jacobi_trunc_threshold  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-Ssw") == 0 )
      {
         arg_index++;
         S_commpkg_switch = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-solver_type") == 0 )
      {
         arg_index++;
         solver_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-recompute") == 0 )
      {
         arg_index++;
         recompute_res = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-iout") == 0 )
      {
         arg_index++;
         ioutdat  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-pout") == 0 )
      {
         arg_index++;
         poutdat  = atoi(argv[arg_index++]);
         poutusr = 1;
      }
      else if ( strcmp(argv[arg_index], "-var") == 0 )
      {
         arg_index++;
         variant  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-use_ns") == 0 )
      {
         arg_index++;
         use_nonsymm_schwarz = 1;
      }
      else if ( strcmp(argv[arg_index], "-ov") == 0 )
      {
         arg_index++;
         overlap  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-dom") == 0 )
      {
         arg_index++;
         domain_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-blk_sm") == 0 )
      {
         arg_index++;
         smooth_num_levels = atoi(argv[arg_index++]);
         overlap = 0;
         smooth_type = 6;
         domain_type = 1;
      }
      else if ( strcmp(argv[arg_index], "-mu") == 0 )
      {
         arg_index++;
         cycle_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-fmg") == 0 )
      {
         arg_index++;
         fcycle  = 1;
      }
      else if ( strcmp(argv[arg_index], "-numsamp") == 0 )
      {
         arg_index++;
         gsmg_samples  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-interptype") == 0 )
      {
         arg_index++;
         interp_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_interp") == 0 )
      {
         arg_index++;
         agg_interp_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_Pmx") == 0 )
      {
         arg_index++;
         agg_P_max_elmts  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_P12_mx") == 0 )
      {
         arg_index++;
         agg_P12_max_elmts  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_tr") == 0 )
      {
         arg_index++;
         agg_trunc_factor  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_P12_tr") == 0 )
      {
         arg_index++;
         agg_P12_trunc_factor  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-postinterptype") == 0 )
      {
         arg_index++;
         post_interp_type  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nodal") == 0 )
      {
         arg_index++;
         nodal  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rel_change") == 0 )
      {
         arg_index++;
         rel_change = 1;
      }
      else if ( strcmp(argv[arg_index], "-nodal_diag") == 0 )
      {
         arg_index++;
         nodal_diag  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-keepSS") == 0 )
      {
         arg_index++;
         keep_same_sign  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cheby_order") == 0 )
      {
         arg_index++;
         cheby_order = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cheby_eig_est") == 0 )
      {
         arg_index++;
         cheby_eig_est = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cheby_variant") == 0 )
      {
         arg_index++;
         cheby_variant = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cheby_scale") == 0 )
      {
         arg_index++;
         cheby_scale = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cheby_fraction") == 0 )
      {
         arg_index++;
         cheby_fraction = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-additive") == 0 )
      {
         arg_index++;
         additive  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mult_add") == 0 )
      {
         arg_index++;
         mult_add  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-simple") == 0 )
      {
         arg_index++;
         simple  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-add_end") == 0 )
      {
         arg_index++;
         add_last_lvl  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-add_Pmx") == 0 )
      {
         arg_index++;
         add_P_max_elmts  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-add_tr") == 0 )
      {
         arg_index++;
         add_trunc_factor  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-add_rlx") == 0 )
      {
         arg_index++;
         add_relax_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-add_w") == 0 )
      {
         arg_index++;
         add_relax_wt = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rap") == 0 )
      {
         arg_index++;
         rap2  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mod_rap2") == 0 )
      {
         arg_index++;
         mod_rap2  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-keepT") == 0 )
      {
         arg_index++;
         keepTranspose  = atoi(argv[arg_index++]);
      }
#ifdef NALU_HYPRE_USING_DSUPERLU
      else if ( strcmp(argv[arg_index], "-dslu_th") == 0 )
      {
         arg_index++;
         dslu_threshold  = atoi(argv[arg_index++]);
      }
#endif
      else if ( strcmp(argv[arg_index], "-nongalerk_tol") == 0 )
      {
         arg_index++;
         nongalerk_num_tol = atoi(argv[arg_index++]);
         nongalerk_tol = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  nongalerk_num_tol, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < nongalerk_num_tol; i++)
         {
            nongalerk_tol[i] = atof(argv[arg_index++]);
         }
      }
      else if ( strcmp(argv[arg_index], "-print") == 0 )
      {
         arg_index++;
         print_system = 1;
      }
      /* BM Oct 23, 2006 */
      else if ( strcmp(argv[arg_index], "-plot_grids") == 0 )
      {
         arg_index++;
         plot_grids = 1;
      }
      else if ( strcmp(argv[arg_index], "-plot_file_name") == 0 )
      {
         arg_index++;
         nalu_hypre_sprintf (plot_file_name, "%s", argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-AIR") == 0 )
      {
         arg_index++;
         air = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-amgdd_start_level") == 0 )
      {
         arg_index++;
         amgdd_start_level = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-amgdd_padding") == 0 )
      {
         arg_index++;
         amgdd_padding = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-amgdd_fac_num_relax") == 0 )
      {
         arg_index++;
         amgdd_fac_num_relax = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-amgdd_num_comp_cycles") == 0 )
      {
         arg_index++;
         amgdd_num_comp_cycles = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-amgdd_fac_relax_type") == 0 )
      {
         arg_index++;
         amgdd_fac_relax_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-amgdd_fac_cycle_type") == 0 )
      {
         arg_index++;
         amgdd_fac_cycle_type = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-amgdd_num_ghost_layers") == 0 )
      {
         arg_index++;
         amgdd_num_ghost_layers = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-precon_cycles") == 0 )
      {
         arg_index++;
         precon_cycles = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /* default settings for AIR alg. */
   if (air)
   {
      restri_type = air;    /* Set Restriction to be AIR */
      interp_type = 100;    /* 1-pt Interp */
      if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
      {
         relax_type = 7;
      }
      else
      {
         relax_type = 0;
      }
      ns_down = 0;
      ns_up = 3;
      /* this is a 2-D 4-by-k array using Double pointers */
      grid_relax_points = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, 4, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[0] = NULL;
      grid_relax_points[1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, ns_down, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[2] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, ns_up, NALU_HYPRE_MEMORY_HOST);
      grid_relax_points[3] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, ns_coarse, NALU_HYPRE_MEMORY_HOST);
      /* down cycle: C */
      for (i = 0; i < ns_down; i++)
      {
         grid_relax_points[1][i] = 0;//1;
      }
      /* up cycle: F */
      //for (i=0; i<ns_up; i++)
      //{
      if (ns_up == 3)
      {
         grid_relax_points[2][0] = -1; // F
         grid_relax_points[2][1] = -1; // F
         grid_relax_points[2][2] =  1; // C
      }
      else if (ns_up == 2)
      {
         grid_relax_points[2][0] = -1; // F
         grid_relax_points[2][1] = -1; // F
      }
      //}
      /* coarse: all */
      for (i = 0; i < ns_coarse; i++)
      {
         grid_relax_points[3][i] = 0;
      }
      coarse_threshold = 20;
      /* does not support aggressive coarsening */
      agg_num_levels = 0;
   }
   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/

   if ( print_usage )
   {
      if ( myid == 0 )
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Usage: %s [<options>]\n", argv[0]);
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -fromfile <filename>       : ");
         nalu_hypre_printf("matrix read from multiple files (IJ format)\n");
         nalu_hypre_printf("  -fromparcsrfile <filename> : ");
         nalu_hypre_printf("matrix read from multiple files (ParCSR format)\n");
         nalu_hypre_printf("  -fromonecsrfile <filename> : ");
         nalu_hypre_printf("matrix read from a single file (CSR format)\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -laplacian [<options>] : build 5pt 2D laplacian problem (default) \n");
         nalu_hypre_printf("  -sysL <num functions>  : build SYSTEMS laplacian 7pt operator\n");
         nalu_hypre_printf("  -9pt [<opts>]          : build 9pt 2D laplacian problem\n");
         nalu_hypre_printf("  -27pt [<opts>]         : build 27pt 3D laplacian problem\n");
         nalu_hypre_printf("  -difconv [<opts>]      : build convection-diffusion problem\n");
         nalu_hypre_printf("    -n <nx> <ny> <nz>    : total problem size \n");
         nalu_hypre_printf("    -P <Px> <Py> <Pz>    : processor topology\n");
         nalu_hypre_printf("    -c <cx> <cy> <cz>    : diffusion coefficients\n");
         nalu_hypre_printf("    -a <ax> <ay> <az>    : convection coefficients\n");
         nalu_hypre_printf("    -atype <type>        : FD scheme for convection \n");
         nalu_hypre_printf("           0=Forward (default)       1=Backward\n");
         nalu_hypre_printf("           2=Centered                3=Upwind\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -exact_size            : inserts immediately into ParCSR structure\n");
         nalu_hypre_printf("  -storage_low           : allocates not enough storage for aux struct\n");
         nalu_hypre_printf("  -concrete_parcsr       : use parcsr matrix type as concrete type\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -rbm <val> <filename>  : rigid body mode vectors\n");
         nalu_hypre_printf("  -nc <val>              : number of components of a vector (multivector)\n");
         nalu_hypre_printf("  -rhsfromfile           : ");
         nalu_hypre_printf("rhs read from multiple files (IJ format)\n");
         nalu_hypre_printf("  -rhsfromonefile        : ");
         nalu_hypre_printf("rhs read from a single file (CSR format)\n");
         nalu_hypre_printf("  -rhsparcsrfile        :  ");
         nalu_hypre_printf("rhs read from multiple files (ParCSR format)\n");
         nalu_hypre_printf("  -Ffromonefile          : ");
         nalu_hypre_printf("list of F points from a single file\n");
         nalu_hypre_printf("  -SFfromonefile          : ");
         nalu_hypre_printf("list of isolated F points from a single file\n");
         nalu_hypre_printf("  -rhsrand               : rhs is random vector\n");
         nalu_hypre_printf("  -rhsisone              : rhs is vector with unit coefficients (default)\n");
         nalu_hypre_printf("  -xisone                : solution of all ones\n");
         nalu_hypre_printf("  -rhszero               : rhs is zero vector\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -dt <val>              : specify finite backward Euler time step\n");
         nalu_hypre_printf("                         :    -rhsfromfile, -rhsfromonefile, -rhsrand,\n");
         nalu_hypre_printf("                         :    -rhsrand, or -xisone will be ignored\n");
         nalu_hypre_printf("  -srcfromfile           : ");
         nalu_hypre_printf("backward Euler source read from multiple files (IJ format)\n");
         nalu_hypre_printf("  -srcfromonefile        : ");
         nalu_hypre_printf("backward Euler source read from a single file (IJ format)\n");
         nalu_hypre_printf("  -srcrand               : ");
         nalu_hypre_printf("backward Euler source is random vector with coefficients in range 0 - 1\n");
         nalu_hypre_printf("  -srcisone              : ");
         nalu_hypre_printf("backward Euler source is vector with unit coefficients (default)\n");
         nalu_hypre_printf("  -srczero               : ");
         nalu_hypre_printf("backward Euler source is zero-vector\n");
         nalu_hypre_printf("  -x0fromfile           : ");
         nalu_hypre_printf("initial guess x0 read from multiple files (IJ format)\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -solver <ID>           : solver ID\n");
         nalu_hypre_printf("       0=AMG               1=AMG-PCG        \n");
         nalu_hypre_printf("       2=DS-PCG            3=AMG-GMRES      \n");
         nalu_hypre_printf("       4=DS-GMRES          5=AMG-CGNR       \n");
         nalu_hypre_printf("       6=DS-CGNR           7=PILUT-GMRES    \n");
         nalu_hypre_printf("       8=ParaSails-PCG     9=AMG-BiCGSTAB   \n");
         nalu_hypre_printf("       10=DS-BiCGSTAB     11=PILUT-BiCGSTAB \n");
         nalu_hypre_printf("       12=Schwarz-PCG     13=GSMG           \n");
         nalu_hypre_printf("       14=GSMG-PCG        15=GSMG-GMRES\n");
         nalu_hypre_printf("       16=AMG-COGMRES     17=DIAG-COGMRES\n");
         nalu_hypre_printf("       18=ParaSails-GMRES\n");
         nalu_hypre_printf("       20=Hybrid solver/ DiagScale, AMG \n");
         nalu_hypre_printf("       31=FSAI-PCG \n");
         nalu_hypre_printf("       43=Euclid-PCG      44=Euclid-GMRES   \n");
         nalu_hypre_printf("       45=Euclid-BICGSTAB 46=Euclid-COGMRES\n");
         nalu_hypre_printf("       47=Euclid-FlexGMRES\n");
         nalu_hypre_printf("       50=DS-LGMRES       51=AMG-LGMRES     \n");
         nalu_hypre_printf("       60=DS-FlexGMRES    61=AMG-FlexGMRES  \n");
         nalu_hypre_printf("       70=MGR             71=MGR-PCG  \n");
         nalu_hypre_printf("       72=MGR-FlexGMRES   73=MGR-BICGSTAB  \n");
         nalu_hypre_printf("       74=MGR-COGMRES  \n");
         nalu_hypre_printf("       80=ILU             81=ILU-GMRES  \n");
         nalu_hypre_printf("       82=ILU-FlexGMRES  \n");
         nalu_hypre_printf("       90=AMG-DD          91=AMG-DD-GMRES  \n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -cljp                 : CLJP coarsening \n");
         nalu_hypre_printf("  -cljp1                : CLJP coarsening, fixed random \n");
         nalu_hypre_printf("  -cgc                  : CGC coarsening \n");
         nalu_hypre_printf("  -cgce                 : CGC-E coarsening \n");
         nalu_hypre_printf("  -pmis                 : PMIS coarsening \n");
         nalu_hypre_printf("  -pmis1                : PMIS coarsening, fixed random \n");
         nalu_hypre_printf("  -hmis                 : HMIS coarsening (default)\n");
         nalu_hypre_printf("  -ruge                 : Ruge-Stueben coarsening (local)\n");
         nalu_hypre_printf("  -ruge1p               : Ruge-Stueben coarsening 1st pass only(local)\n");
         nalu_hypre_printf("  -ruge3                : third pass on boundary\n");
         nalu_hypre_printf("  -ruge3c               : third pass on boundary, keep c-points\n");
         nalu_hypre_printf("  -falgout              : local Ruge_Stueben followed by CLJP\n");
         nalu_hypre_printf("  -gm                   : use global measures\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -interptype  <val>    : set interpolation type\n");
         nalu_hypre_printf("       0=Classical modified interpolation  \n");
         nalu_hypre_printf("       1=least squares interpolation (for GSMG only)  \n");
         nalu_hypre_printf("       0=Classical modified interpolation for hyperbolic PDEs \n");
         nalu_hypre_printf("       3=direct interpolation with separation of weights  \n");
         nalu_hypre_printf("       15=direct interpolation\n");
         nalu_hypre_printf("       4=multipass interpolation  \n");
         nalu_hypre_printf("       5=multipass interpolation with separation of weights  \n");
         nalu_hypre_printf("       6=extended classical modified interpolation (default) \n");
         nalu_hypre_printf("       7=extended (only if no common C neighbor) interpolation  \n");
         nalu_hypre_printf("       8=standard interpolation  \n");
         nalu_hypre_printf("       9=standard interpolation with separation of weights  \n");
         nalu_hypre_printf("      12=FF interpolation  \n");
         nalu_hypre_printf("      13=FF1 interpolation  \n");

         nalu_hypre_printf("      16=use modified unknown interpolation for a system (w/unknown or hybrid approach) \n");
         nalu_hypre_printf("      17=use non-systems interp = 6 for a system (w/unknown or hybrid approach) \n");
         nalu_hypre_printf("      18=use non-systems interp = 8 for a system (w/unknown or hybrid approach) \n");
         nalu_hypre_printf("      19=use non-systems interp = 0 for a system (w/unknown or hybrid approach) \n");

         nalu_hypre_printf("      10=classical block interpolation for nodal systems AMG\n");
         nalu_hypre_printf("      11=classical block interpolation with diagonal blocks for nodal systems AMG\n");
         nalu_hypre_printf("      20=same as 10, but don't add weak connect. to diag \n");
         nalu_hypre_printf("      21=same as 11, but don't add weak connect. to diag \n");
         nalu_hypre_printf("      22=classical block interpolation w/Ruge's variant for nodal systems AMG \n");
         nalu_hypre_printf("      23=same as 22, but use row sums for diag scaling matrices,for nodal systems AMG \n");
         nalu_hypre_printf("      24=direct block interpolation for nodal systems AMG\n");
         nalu_hypre_printf("     100=One point interpolation [a Boolean matrix]\n");
         nalu_hypre_printf("\n");

         /* RL */
         nalu_hypre_printf("  -restritype  <val>    : set restriction type\n");
         nalu_hypre_printf("       0=transpose of the interpolation  \n");
         nalu_hypre_printf("       k=local approximate ideal restriction (AIR-k)  \n");
         nalu_hypre_printf("\n");

         nalu_hypre_printf("  -rlx  <val>            : relaxation type\n");
         nalu_hypre_printf("       0=Weighted Jacobi  \n");
         nalu_hypre_printf("       1=Gauss-Seidel (very slow!)  \n");
         nalu_hypre_printf("       3=Hybrid Gauss-Seidel  \n");
         nalu_hypre_printf("       4=Hybrid backward Gauss-Seidel  \n");
         nalu_hypre_printf("       6=Hybrid symmetric Gauss-Seidel  \n");
         nalu_hypre_printf("       8= symmetric L1-Gauss-Seidel  \n");
         nalu_hypre_printf("       13= forward L1-Gauss-Seidel  \n");
         nalu_hypre_printf("       14= backward L1-Gauss-Seidel  \n");
         nalu_hypre_printf("       15=CG  \n");
         nalu_hypre_printf("       16=Chebyshev  \n");
         nalu_hypre_printf("       17=FCF-Jacobi  \n");
         nalu_hypre_printf("       18=L1-Jacobi (may be used with -CF) \n");
         nalu_hypre_printf("       9=Gauss elimination (use for coarsest grid only)  \n");
         nalu_hypre_printf("       99=Gauss elimination with pivoting (use for coarsest grid only)  \n");
         nalu_hypre_printf("       20= Nodal Weighted Jacobi (for systems only) \n");
         nalu_hypre_printf("       23= Nodal Hybrid Jacobi/Gauss-Seidel (for systems only) \n");
         nalu_hypre_printf("       26= Nodal Hybrid Symmetric Gauss-Seidel  (for systems only)\n");
         nalu_hypre_printf("       29= Nodal Gauss elimination (use for coarsest grid only)  \n");
         nalu_hypre_printf("  -rlx_coarse  <val>       : set relaxation type for coarsest grid\n");
         nalu_hypre_printf("  -rlx_down    <val>       : set relaxation type for down cycle\n");
         nalu_hypre_printf("  -rlx_up      <val>       : set relaxation type for up cycle\n");
         nalu_hypre_printf("  -cheby_order  <val> : set order (1-4) for Chebyshev poly. smoother (default is 2)\n");
         nalu_hypre_printf("  -cheby_fraction <val> : fraction of the spectrum for Chebyshev poly. smoother (default is .3)\n");
         nalu_hypre_printf("  -nodal  <val>            : nodal system type\n");
         nalu_hypre_printf("       0 = Unknown approach \n");
         nalu_hypre_printf("       1 = Frobenius norm  \n");
         nalu_hypre_printf("       2 = Sum of Abs.value of elements  \n");
         nalu_hypre_printf("       3 = Largest magnitude element (includes its sign)  \n");
         nalu_hypre_printf("       4 = Inf. norm  \n");
         nalu_hypre_printf("       5 = One norm  (note: use with block version only) \n");
         nalu_hypre_printf("       6 = Sum of all elements in block  \n");
         nalu_hypre_printf("  -nodal_diag <val>        :how to treat diag elements\n");
         nalu_hypre_printf("       0 = no special treatment \n");
         nalu_hypre_printf("       1 = make diag = neg.sum of the off_diag  \n");
         nalu_hypre_printf("       2 = make diag = neg. of diag \n");
         nalu_hypre_printf("  -ns <val>              : Use <val> sweeps on each level\n");
         nalu_hypre_printf("                           (default C/F down, F/C up, F/C fine\n");
         nalu_hypre_printf("  -ns_coarse  <val>       : set no. of sweeps for coarsest grid\n");
         /* RL restore these */
         nalu_hypre_printf("  -ns_down    <val>       : set no. of sweeps for down cycle\n");
         nalu_hypre_printf("  -ns_up      <val>       : set no. of sweeps for up cycle\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -mu   <val>            : set AMG cycles (1=V, 2=W, etc.)\n");
         nalu_hypre_printf("  -cutf <val>            : set coarsening cut factor for dense rows\n");
         nalu_hypre_printf("  -th   <val>            : set AMG threshold Theta = val \n");
         nalu_hypre_printf("  -tr   <val>            : set AMG interpolation truncation factor = val \n");
         nalu_hypre_printf("  -Pmx  <val>            : set maximal no. of elmts per row for AMG interpolation (default: 4)\n");
         nalu_hypre_printf("  -jtr  <val>            : set truncation threshold for Jacobi interpolation = val \n");
         nalu_hypre_printf("  -Ssw  <val>            : set S-commpkg-switch = val \n");
         nalu_hypre_printf("  -mxrs <val>            : set AMG maximum row sum threshold for dependency weakening \n");
         nalu_hypre_printf("  -nf <val>              : set number of functions for systems AMG\n");
         nalu_hypre_printf("  -numsamp <val>         : set number of sample vectors for GSMG\n");

         nalu_hypre_printf("  -postinterptype <val>  : invokes <val> no. of Jacobi interpolation steps after main interpolation\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -cgcitr <val>          : set maximal number of coarsening iterations for CGC\n");
         nalu_hypre_printf("  -solver_type <val>     : sets solver within Hybrid solver\n");
         nalu_hypre_printf("                         : 1  PCG  (default)\n");
         nalu_hypre_printf("                         : 2  GMRES\n");
         nalu_hypre_printf("                         : 3  BiCGSTAB\n");

         nalu_hypre_printf("  -w   <val>             : set Jacobi relax weight = val\n");
         nalu_hypre_printf("  -k   <val>             : dimension Krylov space for GMRES\n");
         nalu_hypre_printf("  -aug   <val>           : number of augmentation vectors for LGMRES (-k indicates total approx space size)\n");

         nalu_hypre_printf("  -mxl  <val>            : maximum number of levels (AMG, ParaSAILS)\n");
         nalu_hypre_printf("  -tol  <val>            : set solver convergence tolerance = val\n");
         nalu_hypre_printf("  -atol  <val>           : set solver absolute convergence tolerance = val\n");
         nalu_hypre_printf("  -max_iter  <val>       : set max iterations\n");
         nalu_hypre_printf("  -mg_max_iter  <val>    : set max iterations for mg solvers\n");
         nalu_hypre_printf("  -agg_nl  <val>         : set number of aggressive coarsening levels (default:0)\n");
         nalu_hypre_printf("  -np  <val>             : set number of paths of length 2 for aggr. coarsening\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -sai_th   <val>        : set ParaSAILS threshold = val \n");
         nalu_hypre_printf("  -sai_filt <val>        : set ParaSAILS filter = val \n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -level   <val>         : set k in ILU(k) for Euclid \n");
         nalu_hypre_printf("  -bj <val>              : enable block Jacobi ILU for Euclid \n");
         nalu_hypre_printf("  -ilut <val>            : set drop tolerance for ILUT in Euclid\n");
         nalu_hypre_printf("                           Note ILUT is sequential only!\n");
         nalu_hypre_printf("  -sparseA <val>         : set drop tolerance in ILU(k) for Euclid \n");
         nalu_hypre_printf("  -rowScale <val>        : enable row scaling in Euclid \n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -drop_tol  <val>       : set threshold for dropping in PILUT\n");
         nalu_hypre_printf("  -nonzeros_to_keep <val>: number of nonzeros in each row to keep\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -iout <val>            : set output flag\n");
         nalu_hypre_printf("       0=no output    1=matrix stats\n");
         nalu_hypre_printf("       2=cycle stats  3=matrix & cycle stats\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -dbg <val>             : set debug flag\n");
         nalu_hypre_printf("       0=no debugging\n       1=internal timing\n       2=interpolation truncation\n       3=more detailed timing in coarsening routine\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -print                 : print out the system\n");
         nalu_hypre_printf("\n");
         /* begin lobpcg */

         nalu_hypre_printf("LOBPCG options:\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -lobpcg                 : run LOBPCG instead of PCG\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -gen                    : solve generalized EVP with B = Laplacian\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -con                    : solve constrained EVP using 'vectors.*.*'\n");
         nalu_hypre_printf("                            as constraints (see -vout 1 below)\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -solver none            : no HYPRE preconditioner is used\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -itr <val>              : maximal number of LOBPCG iterations\n");
         nalu_hypre_printf("                            (default 100);\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -vrand <val>            : compute <val> eigenpairs using random\n");
         nalu_hypre_printf("                            initial vectors (default 1)\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -seed <val>             : use <val> as the seed for the random\n");
         nalu_hypre_printf("                            number generator(default seed is based\n");
         nalu_hypre_printf("                            on the time of the run)\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -vfromfile              : read initial vectors from files\n");
         nalu_hypre_printf("                            vectors.i.j where i is vector number\n");
         nalu_hypre_printf("                            and j is processor number\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -orthchk                : check eigenvectors for orthonormality\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -verb <val>             : verbosity level\n");
         nalu_hypre_printf("  -verb 0                 : no print\n");
         nalu_hypre_printf("  -verb 1                 : print initial eigenvalues and residuals,\n");
         nalu_hypre_printf("                            the iteration number, the number of\n");
         nalu_hypre_printf("                            non-convergent eigenpairs and final\n");
         nalu_hypre_printf("                            eigenvalues and residuals (default)\n");
         nalu_hypre_printf("  -verb 2                 : print eigenvalues and residuals on each\n");
         nalu_hypre_printf("                            iteration\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -pcgitr <val>           : maximal number of inner PCG iterations\n");
         nalu_hypre_printf("                            for preconditioning (default 1);\n");
         nalu_hypre_printf("                            if <val> = 0 then the preconditioner\n");
         nalu_hypre_printf("                            is applied directly\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -pcgtol <val>           : residual tolerance for inner iterations\n");
         nalu_hypre_printf("                            (default 0.01)\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -vout <val>             : file output level\n");
         nalu_hypre_printf("  -vout 0                 : no files created (default)\n");
         nalu_hypre_printf("  -vout 1                 : write eigenvalues to values.txt, residuals\n");
         nalu_hypre_printf("                            to residuals.txt and eigenvectors to \n");
         nalu_hypre_printf("                            vectors.i.j where i is vector number\n");
         nalu_hypre_printf("                            and j is processor number\n");
         nalu_hypre_printf("  -vout 2                 : in addition to the above, write the\n");
         nalu_hypre_printf("                            eigenvalues history (the matrix whose\n");
         nalu_hypre_printf("                            i-th column contains eigenvalues at\n");
         nalu_hypre_printf("                            (i+1)-th iteration) to val_hist.txt and\n");
         nalu_hypre_printf("                            residuals history to res_hist.txt\n");
         nalu_hypre_printf("\nNOTE: in this test driver LOBPCG only works with solvers 1, 2, 8, 12, 14 and 43\n");
         nalu_hypre_printf("\ndefault solver is 1\n");
         nalu_hypre_printf("\n");

         /* end lobpcg */

         nalu_hypre_printf("  -plot_grids            : print out information for plotting the grids\n");
         nalu_hypre_printf("  -plot_file_name <val>  : file name for plotting output\n");
         nalu_hypre_printf("\n");
         nalu_hypre_printf("  -smtype <val>      :smooth type\n");
         nalu_hypre_printf("  -smlv <val>        :smooth num levels\n");
         nalu_hypre_printf("  -ov <val>          :over lap:\n");
         nalu_hypre_printf("  -dom <val>         :domain type\n");
         nalu_hypre_printf("  -use_ns            : use non-symm schwarz smoother\n");
         nalu_hypre_printf("  -var <val>         : schwarz smoother variant (0-3) \n");
         nalu_hypre_printf("  -blk_sm <val>      : same as '-smtype 6 -ov 0 -dom 1 -smlv <val>'\n");
         nalu_hypre_printf("  -nongalerk_tol <val> <list>    : specify the NonGalerkin drop tolerance\n");
         nalu_hypre_printf("                                   and list contains the values, where last value\n");
         nalu_hypre_printf("                                   in list is repeated if val < num_levels in AMG\n");

         /* MGR options */
         nalu_hypre_printf("  -mgr_bsize   <val>               : set block size = val\n");
         nalu_hypre_printf("  -mgr_nlevels   <val>             : set number of coarsening levels = val\n");
         nalu_hypre_printf("  -mgr_num_reserved_nodes   <val>  : set number of reserved nodes \n");
         nalu_hypre_printf("                                     to be kept till the coarsest grid = val\n");
         nalu_hypre_printf("  -mgr_non_c_to_f   <val>          : set strategy for intermediate coarse grid \n");
         nalu_hypre_printf("  -mgr_non_c_to_f   0              : Allow some non Cpoints to be labeled \n");
         nalu_hypre_printf("                                     Cpoints on intermediate grid \n");
         nalu_hypre_printf("  -mgr_non_c_to_f   1              : set non Cpoints strictly to Fpoints \n");
         nalu_hypre_printf("  -mgr_frelax_method   <val>       : set F-relaxation strategy \n");
         nalu_hypre_printf("  -mgr_frelax_method   0           : Use 'single-level smoother' strategy \n");
         nalu_hypre_printf("                                     for F-relaxation \n");
         nalu_hypre_printf("  -mgr_frelax_method   1           : Use a 'multi-level smoother' strategy \n");
         nalu_hypre_printf("                                     for F-relaxation \n");
         /* end MGR options */
         /* hypre ILU options */
         nalu_hypre_printf("  -ilu_type   <val>                : set ILU factorization type = val\n");
         nalu_hypre_printf("  -ilu_type   0                    : Block Jacobi with ILU(k) variants \n");
         nalu_hypre_printf("  -ilu_type   1                    : Block Jacobi with ILUT \n");
         nalu_hypre_printf("  -ilu_type   10                   : GMRES with ILU(k) variants \n");
         nalu_hypre_printf("  -ilu_type   11                   : GMRES with ILUT \n");
         nalu_hypre_printf("  -ilu_type   20                   : NSH with ILU(k) variants \n");
         nalu_hypre_printf("  -ilu_type   21                   : NSH with ILUT \n");
         nalu_hypre_printf("  -ilu_type   30                   : RAS with ILU(k) variants \n");
         nalu_hypre_printf("  -ilu_type   31                   : RAS with ILUT \n");
         nalu_hypre_printf("  -ilu_type   40                   : ddPQ + GMRES with ILU(k) variants \n");
         nalu_hypre_printf("  -ilu_type   41                   : ddPQ + GMRES with ILUT \n");
         nalu_hypre_printf("  -ilu_type   50                   : GMRES with ILU(0): RAP variant with MILU(0)  \n");
         nalu_hypre_printf("  -ilu_lfil   <val>                : set level of fill (k) for ILU(k) = val\n");
         nalu_hypre_printf("  -ilu_droptol   <val>             : set drop tolerance threshold for ILUT = val \n");
         nalu_hypre_printf("  -ilu_max_row_nnz   <val>         : set max. num of nonzeros to keep per row = val \n");
         nalu_hypre_printf("  -ilu_schur_max_iter   <val>      : set max. num of iteration for GMRES/NSH Schur = val \n");
         nalu_hypre_printf("  -ilu_nsh_droptol   <val>         : set drop tolerance threshold for NSH = val \n");
         nalu_hypre_printf("  -ilu_sm_max_iter   <val>         : set number of iterations when applied as a smmother in AMG = val \n");
         /* end ILU options */
         /* hypre FSAI options */
         nalu_hypre_printf("  -fs_max_steps <val>              : Maximum number of steps for FSAI \n");
         nalu_hypre_printf("  -fs_max_step_size <val>          : Maximum step size for FSAI \n");
         nalu_hypre_printf("  -fs_eig_max_iters <val>          : Number of iterations for computing maximum eigenvalue of preconditioned operator \n");
         nalu_hypre_printf("  -fs_kap_tol <val>                : Kap. grad. reduction theshold for FSAI \n");
         /* end FSAI options */
         /* hypre AMG-DD options */
         nalu_hypre_printf("  -amgdd_start_level   <val>       : set AMG-DD start level = val\n");
         nalu_hypre_printf("  -amgdd_padding   <val>           : set AMG-DD padding = val\n");
         nalu_hypre_printf("  -amgdd_num_ghost_layers   <val>  : set AMG-DD number of ghost layers = val\n");
         nalu_hypre_printf("  -amgdd_fac_num_relax   <val>     : set AMG-DD FAC cycle number of pre/post-relaxations = val\n");
         nalu_hypre_printf("  -amgdd_num_comp_cycles   <val>   : set AMG-DD number of inner FAC cycles = val\n");
         nalu_hypre_printf("  -amgdd_fac_relax_type   <val>    : set AMG-DD FAC relaxation type = val\n");
         nalu_hypre_printf("       0=Weighted Jacobi  \n");
         nalu_hypre_printf("       1=Gauss-Seidel  \n");
         nalu_hypre_printf("       2=Ordered Gauss-Seidel  \n");
         nalu_hypre_printf("       3=CFL1 Jacobi  \n");
         nalu_hypre_printf("  -amgdd_fac_cycle_type   <val>    : set AMG-DD FAC cycle type = val\n");
         nalu_hypre_printf("       1=V-cycle  \n");
         nalu_hypre_printf("       2=W-cycle  \n");
         nalu_hypre_printf("       3=F-cycle  \n");
         /* end AMG-DD options */
      }

      goto final;
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
#if defined(NALU_HYPRE_DEVELOP_STRING) && defined(NALU_HYPRE_DEVELOP_BRANCH)
      nalu_hypre_printf("\nUsing NALU_HYPRE_DEVELOP_STRING: %s (branch %s; the develop branch)\n\n",
                   NALU_HYPRE_DEVELOP_STRING, NALU_HYPRE_DEVELOP_BRANCH);

#elif defined(NALU_HYPRE_DEVELOP_STRING) && !defined(NALU_HYPRE_DEVELOP_BRANCH)
      nalu_hypre_printf("\nUsing NALU_HYPRE_DEVELOP_STRING: %s (branch %s; not the develop branch)\n\n",
                   NALU_HYPRE_DEVELOP_STRING, NALU_HYPRE_BRANCH_NAME);

#elif defined(NALU_HYPRE_RELEASE_VERSION)
      nalu_hypre_printf("\nUsing NALU_HYPRE_RELEASE_VERSION: %s\n\n",
                   NALU_HYPRE_RELEASE_VERSION);
#endif

      nalu_hypre_printf("Running with these driver parameters:\n");
      nalu_hypre_printf("  solver ID    = %d\n\n", solver_id);
   }

   /*-----------------------------------------------------------------
    * GPU Device binding
    * Must be done before NALU_HYPRE_Init() and should not be changed after
    *-----------------------------------------------------------------*/
   nalu_hypre_bind_device(myid, num_procs, nalu_hypre_MPI_COMM_WORLD);

   time_index = nalu_hypre_InitializeTiming("Hypre init");
   nalu_hypre_BeginTiming(time_index);

   /*-----------------------------------------------------------
    * Initialize : must be the first HYPRE function to call
    *-----------------------------------------------------------*/
   NALU_HYPRE_Init();

   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("Hypre init times", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

#ifdef NALU_HYPRE_USING_DEVICE_POOL
   /* To be effective, nalu_hypre_SetCubMemPoolSize must immediately follow NALU_HYPRE_Init */
   NALU_HYPRE_SetGPUMemoryPoolSize( mempool_bin_growth, mempool_min_bin,
                               mempool_max_bin, mempool_max_cached_bytes );
#endif

#if defined(NALU_HYPRE_USING_UMPIRE)
   /* Setup Umpire pools */
   NALU_HYPRE_SetUmpireDevicePoolName("NALU_HYPRE_DEVICE_POOL_TEST");
   NALU_HYPRE_SetUmpireUMPoolName("NALU_HYPRE_UM_POOL_TEST");
   NALU_HYPRE_SetUmpireHostPoolName("NALU_HYPRE_HOST_POOL_TEST");
   NALU_HYPRE_SetUmpirePinnedPoolName("NALU_HYPRE_PINNED_POOL_TEST");
   NALU_HYPRE_SetUmpireDevicePoolSize(4LL * 1024 * 1024 * 1024);
   NALU_HYPRE_SetUmpireUMPoolSize(4LL * 1024 * 1024 * 1024);
   NALU_HYPRE_SetUmpireHostPoolSize(4LL * 1024 * 1024 * 1024);
   NALU_HYPRE_SetUmpirePinnedPoolSize(4LL * 1024 * 1024 * 1024);
#endif

#if defined(NALU_HYPRE_USING_MEMORY_TRACKER)
   nalu_hypre_MemoryTrackerSetPrint(print_mem_tracker);
   if (mem_tracker_name[0]) { nalu_hypre_MemoryTrackerSetFileName(mem_tracker_name); }
#endif

   /* default memory location */
   NALU_HYPRE_SetMemoryLocation(memory_location);

   /* default execution policy */
   NALU_HYPRE_SetExecutionPolicy(default_exec_policy);

#if defined(NALU_HYPRE_USING_GPU)
   ierr = NALU_HYPRE_SetSpMVUseVendor(spmv_use_vendor); nalu_hypre_assert(ierr == 0);
   /* use vendor implementation for SpGEMM */
   ierr = NALU_HYPRE_SetSpGemmUseVendor(spgemm_use_vendor); nalu_hypre_assert(ierr == 0);
   ierr = nalu_hypre_SetSpGemmAlgorithm(spgemm_alg); nalu_hypre_assert(ierr == 0);
   ierr = nalu_hypre_SetSpGemmBinned(spgemm_binned); nalu_hypre_assert(ierr == 0);
   ierr = nalu_hypre_SetSpGemmRownnzEstimateMethod(spgemm_rowest_mtd); nalu_hypre_assert(ierr == 0);
   if (spgemm_rowest_nsamples > 0) { ierr = nalu_hypre_SetSpGemmRownnzEstimateNSamples(spgemm_rowest_nsamples); nalu_hypre_assert(ierr == 0); }
   if (spgemm_rowest_mult > 0.0) { ierr = nalu_hypre_SetSpGemmRownnzEstimateMultFactor(spgemm_rowest_mult); nalu_hypre_assert(ierr == 0); }
   /* use cuRand for PMIS */
   NALU_HYPRE_SetUseGpuRand(use_curand);
#endif

   /*-----------------------------------------------------------
    * Set up matrix
    *-----------------------------------------------------------*/

   if ( myid == 0 && dt != dt_inf)
   {
      nalu_hypre_printf("  Backward Euler time step with dt = %e\n", dt);
      nalu_hypre_printf("  Dirichlet 0 BCs are implicit in the spatial operator\n");
   }

   time_index = nalu_hypre_InitializeTiming("Spatial Operator");
   nalu_hypre_BeginTiming(time_index);
   if ( build_matrix_type == -1 )
   {
      ierr = NALU_HYPRE_IJMatrixRead( argv[build_matrix_arg_index], comm,
                                 NALU_HYPRE_PARCSR, &ij_A );
      if (ierr)
      {
         nalu_hypre_printf("ERROR: Problem reading in the system matrix!\n");
         exit(1);
      }
   }
   else if ( build_matrix_type == 0 )
   {
      BuildParFromFile(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 1 )
   {
      BuildParFromOneFile(argc, argv, build_matrix_arg_index, num_functions,
                          &parcsr_A);
   }
   else if ( build_matrix_type == 2 )
   {
      BuildParLaplacian(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 3 )
   {
      BuildParLaplacian9pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 4 )
   {
      BuildParLaplacian27pt(argc, argv, build_matrix_arg_index, &parcsr_A);

      nalu_hypre_CSRMatrixGpuSpMVAnalysis(nalu_hypre_ParCSRMatrixDiag(parcsr_A));
   }
   else if ( build_matrix_type == 5 )
   {
      BuildParDifConv(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 6 )
   {
      BuildParVarDifConv(argc, argv, build_matrix_arg_index, &parcsr_A, &b);
      build_rhs_type      = 6;
      build_src_type      = 5;
   }
   else if ( build_matrix_type == 7 )
   {
      BuildParRotate7pt(argc, argv, build_matrix_arg_index, &parcsr_A);
   }

   else
   {
      nalu_hypre_printf("You have asked for an unsupported problem with\n");
      nalu_hypre_printf("build_matrix_type = %d.\n", build_matrix_type);
      return (-1);
   }
   /* BM Oct 23, 2006 */
   if (plot_grids)
   {
      if (build_matrix_type > 1 &&  build_matrix_type < 8)
         BuildParCoordinates (argc, argv, build_matrix_arg_index,
                              &coord_dim, &coordinates);
      else
      {
         nalu_hypre_printf("Warning: coordinates are not yet printed for build_matrix_type = %d.\n",
                      build_matrix_type);
      }
   }

   if (build_matrix_type < 0)
   {
      ierr = NALU_HYPRE_IJMatrixGetLocalRange( ij_A,
                                          &first_local_row, &last_local_row,
                                          &first_local_col, &last_local_col );

      local_num_rows = (NALU_HYPRE_Int)(last_local_row - first_local_row + 1);
      local_num_cols = (NALU_HYPRE_Int)(last_local_col - first_local_col + 1);
      ierr += NALU_HYPRE_IJMatrixGetObject( ij_A, &object);
      parcsr_A = (NALU_HYPRE_ParCSRMatrix) object;
   }
   else
   {
      /*-----------------------------------------------------------
       * Copy the parcsr matrix into the IJMatrix through interface calls
       *-----------------------------------------------------------*/
      ierr = NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                              &first_local_row, &last_local_row,
                                              &first_local_col, &last_local_col );

      local_num_rows = (NALU_HYPRE_Int)(last_local_row - first_local_row + 1);
      local_num_cols = (NALU_HYPRE_Int)(last_local_col - first_local_col + 1);
   }
   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("Generate Matrix", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   /* Read matrix to be passed to the preconditioner */
   if (build_matrix_M == 1)
   {
      time_index = nalu_hypre_InitializeTiming("Auxiliary Operator");
      nalu_hypre_BeginTiming(time_index);

      ierr = NALU_HYPRE_IJMatrixRead( argv[build_matrix_M_arg_index], comm,
                                 NALU_HYPRE_PARCSR, &ij_M );
      if (ierr)
      {
         nalu_hypre_printf("ERROR: Problem reading in the auxiliary matrix B!\n");
         exit(1);
      }

      NALU_HYPRE_IJMatrixGetObject(ij_M, &object);
      parcsr_M = (NALU_HYPRE_ParCSRMatrix) object;

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Auxiliary Operator", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();
   }
   else
   {
      parcsr_M = parcsr_A;
   }

   /* Check the ij interface - not necessary if one just wants to test solvers */
   if (test_ij && build_matrix_type > -1)
   {
      nalu_hypre_ParCSRMatrixMigrate(parcsr_A, NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_Int mx_size = 5;
      time_index = nalu_hypre_InitializeTiming("Generate IJ matrix");
      nalu_hypre_BeginTiming(time_index);

      ierr += NALU_HYPRE_ParCSRMatrixGetDims( parcsr_A, &M, &N );

      ierr += NALU_HYPRE_IJMatrixCreate( comm, first_local_row, last_local_row,
                                    first_local_col, last_local_col, &ij_A );

      ierr += NALU_HYPRE_IJMatrixSetObjectType( ij_A, NALU_HYPRE_PARCSR );
      num_rows = local_num_rows;
      if (off_proc)
      {
         if (myid != num_procs - 1)
         {
            num_rows++;
         }
         if (myid)
         {
            num_rows++;
         }
      }
      /* The following shows how to build an IJMatrix if one has only an
         estimate for the row sizes */
      row_nums = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_rows, NALU_HYPRE_MEMORY_HOST);
      num_cols = nalu_hypre_CTAlloc(NALU_HYPRE_Int,    num_rows, NALU_HYPRE_MEMORY_HOST);
      if (sparsity_known == 1)
      {
         diag_sizes    = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rows, NALU_HYPRE_MEMORY_HOST);
         offdiag_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_num_rows, NALU_HYPRE_MEMORY_HOST);
      }
      else
      {
         size = 5;
         if (sparsity_known == 0)
         {
            if (build_matrix_type == 2)
            {
               size = 7;
            }
            if (build_matrix_type == 3)
            {
               size = 9;
            }
            if (build_matrix_type == 4)
            {
               size = 27;
            }
         }
         row_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_rows; i++)
         {
            row_sizes[i] = size;
         }
      }
      local_row = 0;
      if (build_matrix_type == 2)
      {
         mx_size = 7;
      }
      if (build_matrix_type == 3)
      {
         mx_size = 9;
      }
      if (build_matrix_type == 4)
      {
         mx_size = 27;
      }
      col_nums = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, mx_size * num_rows, NALU_HYPRE_MEMORY_HOST);
      data     = nalu_hypre_CTAlloc(NALU_HYPRE_Real,   mx_size * num_rows, NALU_HYPRE_MEMORY_HOST);
      i_indx = 0;
      j_indx = 0;

      if (off_proc && myid)
      {
         num_cols[i_indx]   = 2;
         row_nums[i_indx++] = first_local_row - 1;
         col_nums[j_indx]   = first_local_row - 1;
         data[j_indx++]     = 6.0;
         col_nums[j_indx]   = first_local_row - 2;
         data[j_indx++]     = -1.0;
      }
      for (i = 0; i < local_num_rows; i++)
      {
         row_nums[i_indx] = first_local_row + i;
         ierr += NALU_HYPRE_ParCSRMatrixGetRow(parcsr_A, first_local_row + i, &size, &col_inds, &values);
         num_cols[i_indx++] = size;
         nalu_hypre_TMemcpy(&col_nums[j_indx], &col_inds[0], NALU_HYPRE_BigInt, size, NALU_HYPRE_MEMORY_HOST,
                       NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(&data[j_indx], &values[0], NALU_HYPRE_Real, size, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
         if (sparsity_known == 1)
         {
            for (j = 0; j < size; j++)
            {
               if (col_nums[j_indx + j] < first_local_row || col_nums[j_indx + j] > last_local_row)
               {
                  offdiag_sizes[local_row]++;
               }
               else
               {
                  diag_sizes[local_row]++;
               }
            }
         }
         j_indx += size;
         local_row++;
         ierr += NALU_HYPRE_ParCSRMatrixRestoreRow(parcsr_A, first_local_row + i, &size, &col_inds, &values);
      }

      if (off_proc && myid != num_procs - 1)
      {
         num_cols[i_indx]   = 2;
         row_nums[i_indx++] = last_local_row + 1;
         col_nums[j_indx]   = last_local_row + 2;
         data[j_indx++]     = -1.0;
         col_nums[j_indx]   = last_local_row + 1;
         data[j_indx++]     = 6.0;
      }

      if (sparsity_known == 1)
      {
         ierr += NALU_HYPRE_IJMatrixSetDiagOffdSizes( ij_A, (const NALU_HYPRE_Int *) diag_sizes,
                                                 (const NALU_HYPRE_Int *) offdiag_sizes );
      }
      else
      {
         ierr = NALU_HYPRE_IJMatrixSetRowSizes ( ij_A, (const NALU_HYPRE_Int *) row_sizes );
      }

      ierr += NALU_HYPRE_IJMatrixInitialize_v2( ij_A, memory_location );

      if (omp_flag)
      {
         NALU_HYPRE_IJMatrixSetOMPFlag(ij_A, 1);
      }

      /* move arrays to `memory_location' */
      NALU_HYPRE_Int    *num_cols_h = num_cols;
      NALU_HYPRE_BigInt *row_nums_h = row_nums;
      NALU_HYPRE_BigInt *col_nums_h = col_nums;
      NALU_HYPRE_Real   *data_h     = data;
      if (nalu_hypre_GetActualMemLocation(memory_location) != nalu_hypre_MEMORY_HOST)
      {
         num_cols = nalu_hypre_TAlloc(NALU_HYPRE_Int,    num_rows,         memory_location);
         row_nums = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_rows,         memory_location);
         col_nums = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, mx_size * num_rows, memory_location);
         data     = nalu_hypre_TAlloc(NALU_HYPRE_Real,   mx_size * num_rows, memory_location);

         nalu_hypre_TMemcpy(num_cols, num_cols_h, NALU_HYPRE_Int,    num_rows,         memory_location,
                       NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(row_nums, row_nums_h, NALU_HYPRE_BigInt, num_rows,         memory_location,
                       NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(col_nums, col_nums_h, NALU_HYPRE_BigInt, mx_size * num_rows, memory_location,
                       NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(data,     data_h,     NALU_HYPRE_Real,   mx_size * num_rows, memory_location,
                       NALU_HYPRE_MEMORY_HOST);
      }

      if (chunk)
      {
         if (add)
         {
            ierr += NALU_HYPRE_IJMatrixAddToValues(ij_A, num_rows, num_cols, row_nums,
                                              (const NALU_HYPRE_BigInt *) col_nums,
                                              (const NALU_HYPRE_Real *) data);
         }
         else
         {
            ierr += NALU_HYPRE_IJMatrixSetValues(ij_A, num_rows, num_cols, row_nums,
                                            (const NALU_HYPRE_BigInt *) col_nums,
                                            (const NALU_HYPRE_Real *) data);
         }
      }
      else
      {
         j_indx = 0;
         for (i = 0; i < num_rows; i++)
         {
            if (add)
            {
               ierr += NALU_HYPRE_IJMatrixAddToValues( ij_A, 1, &num_cols[i], &row_nums[i],
                                                  (const NALU_HYPRE_BigInt *) &col_nums[j_indx],
                                                  (const NALU_HYPRE_Real *) &data[j_indx] );
            }
            else
            {
               ierr += NALU_HYPRE_IJMatrixSetValues( ij_A, 1, &num_cols[i], &row_nums[i],
                                                (const NALU_HYPRE_BigInt *) &col_nums[j_indx],
                                                (const NALU_HYPRE_Real *) &data[j_indx] );
            }
            j_indx += num_cols_h[i];
         }
      }
      nalu_hypre_TFree(num_cols_h, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(row_nums_h, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(col_nums_h, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data_h,     NALU_HYPRE_MEMORY_HOST);
      if (nalu_hypre_GetActualMemLocation(memory_location) != nalu_hypre_MEMORY_HOST)
      {
         nalu_hypre_TFree(col_nums, memory_location);
         nalu_hypre_TFree(data,     memory_location);
         nalu_hypre_TFree(row_nums, memory_location);
         nalu_hypre_TFree(num_cols, memory_location);
      }

      if (sparsity_known == 1)
      {
         nalu_hypre_TFree(diag_sizes,    NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(offdiag_sizes, NALU_HYPRE_MEMORY_HOST);
      }
      else
      {
         nalu_hypre_TFree(row_sizes, NALU_HYPRE_MEMORY_HOST);
      }

      ierr += NALU_HYPRE_IJMatrixAssemble( ij_A );

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("IJ Matrix Setup", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      if (ierr)
      {
         nalu_hypre_printf("Error in driver building IJMatrix from parcsr matrix. \n");
         return (-1);
      }

      /* This is to emphasize that one can IJMatrixAddToValues after an
         IJMatrixRead or an IJMatrixAssemble.  After an IJMatrixRead,
         assembly is unnecessary if the sparsity pattern of the matrix is
         not changed somehow.  If one has not used IJMatrixRead, one has
         the opportunity to IJMatrixAddTo before a IJMatrixAssemble.
         This first sets all matrix coefficients to -1 and then adds 7.0
         to the diagonal to restore the original matrix*/

      if (check_constant)
      {
         ierr += NALU_HYPRE_IJMatrixSetConstantValues( ij_A, -1.0 );
      }

      ncols    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     last_local_row - first_local_row + 1, NALU_HYPRE_MEMORY_HOST);
      rows     = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  last_local_row - first_local_row + 1, NALU_HYPRE_MEMORY_HOST);
      col_inds = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  last_local_row - first_local_row + 1, NALU_HYPRE_MEMORY_HOST);
      values   = nalu_hypre_TAlloc(NALU_HYPRE_Real,    last_local_row - first_local_row + 1, NALU_HYPRE_MEMORY_HOST);

      val = 0.0;

      if (check_constant)
      {
         val = 7.0;
      }
      if (dt < dt_inf)
      {
         val += 1. / dt;
      }
      else
      {
         val += 0.0;   /* Use zero to avoid unintentional loss of significance */
      }

      for (big_i = first_local_row; big_i <= last_local_row; big_i++)
      {
         j = (NALU_HYPRE_Int) (big_i - first_local_row);
         ncols[j]    = 1;
         rows[j]     = big_i;
         col_inds[j] = big_i;
         values[j]   = val;
      }

      if (nalu_hypre_GetActualMemLocation(memory_location) != nalu_hypre_MEMORY_HOST)
      {
         NALU_HYPRE_Int    *ncols_h    = ncols;
         NALU_HYPRE_BigInt *rows_h     = rows;
         NALU_HYPRE_BigInt *col_inds_h = col_inds;
         NALU_HYPRE_Real   *values_h   = values;

         ncols    = nalu_hypre_TAlloc(NALU_HYPRE_Int,     last_local_row - first_local_row + 1, memory_location);
         rows     = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  last_local_row - first_local_row + 1, memory_location);
         col_inds = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  last_local_row - first_local_row + 1, memory_location);
         values   = nalu_hypre_TAlloc(NALU_HYPRE_Real,    last_local_row - first_local_row + 1, memory_location);

         nalu_hypre_TMemcpy(ncols,    ncols_h,    NALU_HYPRE_Int,    last_local_row - first_local_row + 1,
                       memory_location, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(rows,     rows_h,     NALU_HYPRE_BigInt, last_local_row - first_local_row + 1,
                       memory_location, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(col_inds, col_inds_h, NALU_HYPRE_BigInt, last_local_row - first_local_row + 1,
                       memory_location, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TMemcpy(values,   values_h,   NALU_HYPRE_Real,   last_local_row - first_local_row + 1,
                       memory_location, NALU_HYPRE_MEMORY_HOST);

         nalu_hypre_TFree(ncols_h,    NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(rows_h,     NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(col_inds_h, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(values_h,   NALU_HYPRE_MEMORY_HOST);
      }

      ierr += NALU_HYPRE_IJMatrixAddToValues( ij_A,
                                         local_num_rows,
                                         /* this is to show one can use NULL if ncols contains all ones */
                                         NULL, /* ncols, */
                                         rows,
                                         (const NALU_HYPRE_BigInt *) col_inds,
                                         (const NALU_HYPRE_Real *) values );

      nalu_hypre_TFree(ncols,    memory_location);
      nalu_hypre_TFree(rows,     memory_location);
      nalu_hypre_TFree(col_inds, memory_location);
      nalu_hypre_TFree(values,   memory_location);

      /* If sparsity pattern is not changed since last IJMatrixAssemble call,
         this should be a no-op */

      ierr += NALU_HYPRE_IJMatrixAssemble( ij_A );

      /*-----------------------------------------------------------
       * Fetch the resulting underlying matrix out
       *-----------------------------------------------------------*/
      ierr += NALU_HYPRE_ParCSRMatrixDestroy(parcsr_A);

      ierr += NALU_HYPRE_IJMatrixGetObject( ij_A, &object);
      parcsr_A = (NALU_HYPRE_ParCSRMatrix) object;
   }

   /*-----------------------------------------------------------
    * Set up the interp vector
    *-----------------------------------------------------------*/
   if (build_rbm)
   {
      char new_file_name[80];
      /* RHS */
      interp_vecs = nalu_hypre_CTAlloc(NALU_HYPRE_ParVector, num_interp_vecs, NALU_HYPRE_MEMORY_HOST);
      ij_rbm = nalu_hypre_CTAlloc(NALU_HYPRE_IJVector, num_interp_vecs, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_interp_vecs; i++)
      {
         nalu_hypre_sprintf(new_file_name, "%s.%d", argv[build_rbm_index], i);
         ierr = NALU_HYPRE_IJVectorRead( new_file_name, nalu_hypre_MPI_COMM_WORLD,
                                    NALU_HYPRE_PARCSR, &ij_rbm[i] );
         ierr = NALU_HYPRE_IJVectorGetObject( ij_rbm[i], &object );
         interp_vecs[i] = (NALU_HYPRE_ParVector) object;
      }
      if (ierr)
      {
         nalu_hypre_printf("ERROR: Problem reading in rbm!\n");
         exit(1);
      }
   }

   /*-----------------------------------------------------------
    * Set up coarsening data
    *-----------------------------------------------------------*/
   if (build_fpt_arg_index || build_sfpt_arg_index || build_cpt_arg_index)
   {
      NALU_HYPRE_ParCSRMatrixGetGlobalRowPartitioning(parcsr_A, 0, &partitioning);

      if (build_fpt_arg_index)
      {
         BuildBigArrayFromOneFile(argc, argv, "Fine points", build_fpt_arg_index,
                                  partitioning, &num_fpt, &fpt_index);
      }

      if (build_sfpt_arg_index)
      {
         BuildBigArrayFromOneFile(argc, argv, "Isolated Fine points", build_sfpt_arg_index,
                                  partitioning, &num_isolated_fpt, &isolated_fpt_index);
      }

      if (build_cpt_arg_index)
      {
         BuildBigArrayFromOneFile(argc, argv, "Coarse points", build_cpt_arg_index,
                                  partitioning, &num_cpt, &cpt_index);
      }

      if (partitioning)
      {
         nalu_hypre_TFree(partitioning, NALU_HYPRE_MEMORY_HOST);
      }
   }

   /*-----------------------------------------------------------
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/
   time_index = nalu_hypre_InitializeTiming("RHS and Initial Guess");
   nalu_hypre_BeginTiming(time_index);

   if (myid == 0)
   {
      nalu_hypre_printf("  Number of vector components: %d\n", num_components);
   }

   if (num_components > 1 && !(build_rhs_type > 1 && build_rhs_type < 6))
   {
      nalu_hypre_printf("num_components > 1 not implemented for this RHS choice!\n");
      nalu_hypre_MPI_Abort(comm, 1);
   }

   if (build_rhs_type == 0)
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector read from file %s\n", argv[build_rhs_arg_index]);
         nalu_hypre_printf("  Initial guess is 0\n");
      }

      /* RHS */
      ierr = NALU_HYPRE_IJVectorRead( argv[build_rhs_arg_index], nalu_hypre_MPI_COMM_WORLD,
                                 NALU_HYPRE_PARCSR, &ij_b );
      if (ierr)
      {
         nalu_hypre_printf("ERROR: Problem reading in the right-hand-side!\n");
         exit(1);
      }
      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);
      NALU_HYPRE_IJVectorAssemble(ij_x);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if (build_rhs_type == 1)
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector read from file %s\n", argv[build_rhs_arg_index]);
         nalu_hypre_printf("  Initial guess is 0\n");
      }

      ij_b = NULL;
      BuildRhsParFromOneFile(argc, argv, build_rhs_arg_index, parcsr_A, &b);

      /* initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);
      NALU_HYPRE_IJVectorAssemble(ij_x);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if (build_rhs_type == 2)
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector has unit coefficients\n");
         nalu_hypre_printf("  Initial guess is 0\n");
      }

      NALU_HYPRE_Complex *values_h = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_rows, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Complex *values_d = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_rows, memory_location);
      for (i = 0; i < local_num_rows; i++)
      {
         values_h[i] = 1.0;
      }
      nalu_hypre_TMemcpy(values_d, values_h, NALU_HYPRE_Complex, local_num_rows,
                    memory_location, NALU_HYPRE_MEMORY_HOST);

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorSetNumComponents(ij_b, num_components);
      NALU_HYPRE_IJVectorInitialize_v2(ij_b, memory_location);
      for (c = 0; c < num_components; c++)
      {
         NALU_HYPRE_IJVectorSetComponent(ij_b, c);
         NALU_HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values_d);
      }
      NALU_HYPRE_IJVectorAssemble(ij_b);
      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      nalu_hypre_Memset(values_d, 0, local_num_rows * sizeof(NALU_HYPRE_Complex), memory_location);
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorSetNumComponents(ij_x, num_components);
      NALU_HYPRE_IJVectorInitialize_v2(ij_x, memory_location);
      for (c = 0; c < num_components; c++)
      {
         NALU_HYPRE_IJVectorSetComponent(ij_x, c);
         NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values_d);
      }
      NALU_HYPRE_IJVectorAssemble(ij_x);
      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;

      nalu_hypre_TFree(values_h, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(values_d, memory_location);
   }
   else if (build_rhs_type == 3)
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector has random coefficients and unit 2-norm\n");
         nalu_hypre_printf("  Initial guess is 0\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorSetNumComponents(ij_b, num_components);
      NALU_HYPRE_IJVectorInitialize(ij_b);
      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* For purposes of this test, NALU_HYPRE_ParVector functions are used, but
         these are not necessary.  For a clean use of the interface, the user
         "should" modify coefficients of ij_x by using functions
         NALU_HYPRE_IJVectorSetValues or NALU_HYPRE_IJVectorAddToValues */

      NALU_HYPRE_ParVectorSetRandomValues(b, 22775);
      NALU_HYPRE_ParVectorInnerProd(b, b, &norm);
      norm = 1. / sqrt(norm);
      ierr = NALU_HYPRE_ParVectorScale(norm, b);

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetNumComponents(ij_x, num_components);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);
      NALU_HYPRE_IJVectorAssemble(ij_x);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if (build_rhs_type == 4)
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector set for solution with unit coefficients\n");
         nalu_hypre_printf("  Initial guess is 0\n");
      }

      NALU_HYPRE_Real *values_h = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_cols, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Real *values_d = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_cols, memory_location);
      for (i = 0; i < local_num_cols; i++)
      {
         values_h[i] = 1.;
      }
      nalu_hypre_TMemcpy(values_d, values_h, NALU_HYPRE_Real, local_num_cols,
                    memory_location, NALU_HYPRE_MEMORY_HOST);

      /* Temporary use of solution vector */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorSetNumComponents(ij_x, num_components);
      NALU_HYPRE_IJVectorInitialize(ij_x);
      for (c = 0; c < num_components; c++)
      {
         NALU_HYPRE_IJVectorSetComponent(ij_x, c);
         NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values_d);
      }
      NALU_HYPRE_IJVectorAssemble(ij_x);
      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;

      nalu_hypre_TFree(values_h, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(values_d, memory_location);

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorSetNumComponents(ij_b, num_components);
      NALU_HYPRE_IJVectorInitialize(ij_b);
      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      NALU_HYPRE_ParCSRMatrixMatvec(1.0, parcsr_A, x, 0.0, b);

      /* Zero initial guess */
      nalu_hypre_IJVectorZeroValues(ij_x);
   }
   else if (build_rhs_type == 5)
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector is 0\n");
         nalu_hypre_printf("  Initial guess has unit coefficients\n");
      }

      NALU_HYPRE_Real *values_h = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_cols, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Real *values_d = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_cols, memory_location);
      for (i = 0; i < local_num_cols; i++)
      {
         values_h[i] = 1.;
      }
      nalu_hypre_TMemcpy(values_d, values_h, NALU_HYPRE_Real, local_num_cols,
                    memory_location, NALU_HYPRE_MEMORY_HOST);

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorSetNumComponents(ij_b, num_components);
      NALU_HYPRE_IJVectorInitialize(ij_b);
      NALU_HYPRE_IJVectorAssemble(ij_b);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetNumComponents(ij_x, num_components);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);
      for (c = 0; c < num_components; c++)
      {
         NALU_HYPRE_IJVectorSetComponent(ij_x, c);
         NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values_d);
      }
      NALU_HYPRE_IJVectorAssemble(ij_x);
      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;

      nalu_hypre_TFree(values_h, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(values_d, memory_location);
   }
   else if (build_rhs_type == 6)
   {
      ij_b = NULL;
   }
   else if (build_rhs_type == 7)
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector read from file %s\n", argv[build_rhs_arg_index]);
         nalu_hypre_printf("  Initial guess is 0\n");
      }

      ij_b = NULL;
      ReadParVectorFromFile(argc, argv, build_rhs_arg_index, &b);

      /* initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);
      NALU_HYPRE_IJVectorAssemble(ij_x);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else
   {
      if (build_rhs_type != -1)
      {
         if (myid == 0)
         {
            nalu_hypre_printf("Error: Invalid build_rhs_type!\n");
         }
         nalu_hypre_MPI_Abort(comm, 1);
      }
   }

   if ( build_src_type == 0)
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  Source vector read from file %s\n", argv[build_src_arg_index]);
         nalu_hypre_printf("  Initial unknown vector in evolution is 0\n");
      }

      ierr = NALU_HYPRE_IJVectorRead( argv[build_src_arg_index], nalu_hypre_MPI_COMM_WORLD,
                                 NALU_HYPRE_PARCSR, &ij_b );
      if (ierr)
      {
         nalu_hypre_printf("ERROR: Problem reading in the right-hand-side!\n");
         exit(1);
      }
      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial unknown vector */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);
      NALU_HYPRE_IJVectorAssemble(ij_x);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if (build_src_type == 1)
   {
      BuildRhsParFromOneFile(argc, argv, build_src_arg_index, parcsr_A, &b);
      ij_b = NULL;

      /* Initial unknown vector */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);
      NALU_HYPRE_IJVectorAssemble(ij_x);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_src_type == 2 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  Source vector has unit coefficients\n");
         nalu_hypre_printf("  Initial unknown vector is 0\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);

      NALU_HYPRE_Real *values_h = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_rows, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Real *values_d = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_rows, memory_location);
      for (i = 0; i < local_num_rows; i++)
      {
         values_h[i] = 1.;
      }
      nalu_hypre_TMemcpy(values_d, values_h, NALU_HYPRE_Real, local_num_rows,
                    memory_location, NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values_d);
      NALU_HYPRE_IJVectorAssemble(ij_b);
      nalu_hypre_TFree(values_h, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(values_d, memory_location);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         0 here) is usually used as the initial guess */
      NALU_HYPRE_IJVectorAssemble(ij_x);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_src_type == 3 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  Source vector has random coefficients in range 0 - 1\n");
         nalu_hypre_printf("  Initial unknown vector is 0\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);

      NALU_HYPRE_Real *values_h = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_rows, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Real *values_d = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_rows, memory_location);
      nalu_hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
      {
         values_h[i] = nalu_hypre_Rand();
      }
      nalu_hypre_TMemcpy(values_d, values_h, NALU_HYPRE_Real, local_num_rows, memory_location, NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values_d);
      NALU_HYPRE_IJVectorAssemble(ij_b);
      nalu_hypre_TFree(values_h, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(values_d, memory_location);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         0 here) is usually used as the initial guess */
      NALU_HYPRE_IJVectorAssemble(ij_x);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_src_type == 4 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  Source vector is 0 \n");
         nalu_hypre_printf("  Initial unknown vector has random coefficients in range 0 - 1\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);

      NALU_HYPRE_Real *values_h = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_rows, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Real *values_d = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_rows, memory_location);
      nalu_hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
      {
         values_h[i] = nalu_hypre_Rand() / dt;
      }
      nalu_hypre_TMemcpy(values_d, values_h, NALU_HYPRE_Real, local_num_rows, memory_location, NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values_d);
      NALU_HYPRE_IJVectorAssemble(ij_b);
      nalu_hypre_TFree(values_h, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(values_d, memory_location);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         random in 0 - 1 here) is usually used as the initial guess */
      values_h = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_cols, NALU_HYPRE_MEMORY_HOST);
      values_d = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_cols, memory_location);
      nalu_hypre_SeedRand(myid);
      for (i = 0; i < local_num_cols; i++)
      {
         values_h[i] = nalu_hypre_Rand();
      }
      nalu_hypre_TMemcpy(values_d, values_h, NALU_HYPRE_Real, local_num_cols, memory_location, NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values_d);
      NALU_HYPRE_IJVectorAssemble(ij_x);
      nalu_hypre_TFree(values_h, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(values_d, memory_location);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_src_type == 5 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  Initial guess is random \n");
      }

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         random in 0 - 1 here) is usually used as the initial guess */
      NALU_HYPRE_Real *values_h = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_cols, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Real *values_d = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_cols, memory_location);
      nalu_hypre_SeedRand(myid + 2747);
      nalu_hypre_SeedRand(myid);
      for (i = 0; i < local_num_cols; i++)
      {
         values_h[i] = nalu_hypre_Rand();
      }
      nalu_hypre_TMemcpy(values_d, values_h, NALU_HYPRE_Real, local_num_cols, memory_location, NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values_d);
      NALU_HYPRE_IJVectorAssemble(ij_x);
      nalu_hypre_TFree(values_h, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(values_d, memory_location);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }

   /* initial guess */
   if ( build_x0_type == 0 )
   {
      /* from file */
      if (myid == 0)
      {
         nalu_hypre_printf("  Initial guess vector read from file %s\n", argv[build_x0_arg_index]);
      }
      /* x0 */
      if (ij_x)
      {
         NALU_HYPRE_IJVectorDestroy(ij_x);
      }
      ierr = NALU_HYPRE_IJVectorRead( argv[build_x0_arg_index], nalu_hypre_MPI_COMM_WORLD,
                                 NALU_HYPRE_PARCSR, &ij_x );
      if (ierr)
      {
         nalu_hypre_printf("ERROR: Problem reading in x0!\n");
         exit(1);
      }
      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if (build_x0_type == 7)
   {
      /* from file */
      if (myid == 0)
      {
         nalu_hypre_printf("  Initial guess vector read from file %s\n", argv[build_x0_arg_index]);
      }

      ReadParVectorFromFile(argc, argv, build_x0_arg_index, &x);
   }
   else if (build_x0_type == 1)
   {
      /* random */
      if (myid == 0)
      {
         nalu_hypre_printf("  Initial guess is random \n");
      }

      if (ij_x)
      {
         NALU_HYPRE_IJVectorDestroy(ij_x);
      }

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         random in 0 - 1 here) is usually used as the initial guess */
      NALU_HYPRE_Real *values_h = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_cols, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Real *values_d = nalu_hypre_CTAlloc(NALU_HYPRE_Real, local_num_cols, memory_location);
      nalu_hypre_SeedRand(myid);
      for (i = 0; i < local_num_cols; i++)
      {
         values_h[i] = nalu_hypre_Rand();
      }
      nalu_hypre_TMemcpy(values_d, values_h, NALU_HYPRE_Real, local_num_cols, memory_location, NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values_d);
      NALU_HYPRE_IJVectorAssemble(ij_x);
      nalu_hypre_TFree(values_h, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(values_d, memory_location);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }

   /*-----------------------------------------------------------
    * Finalize IJVector Setup timings
    *-----------------------------------------------------------*/

   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("IJ Vector Setup", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   if (num_functions > 1)
   {
      dof_func = NULL;
      if (build_funcs_type == 1)
      {
         if (myid == 0)
         {
            nalu_hypre_printf(" Calling BuildFuncsFromOneFile\n");
         }
         BuildFuncsFromOneFile(argc, argv, build_funcs_arg_index, parcsr_A, &dof_func);
      }
      else if (build_funcs_type == 2)
      {
         if (myid == 0)
         {
            nalu_hypre_printf(" Calling BuildFuncsFromFiles\n");
         }
         BuildFuncsFromFiles(argc, argv, build_funcs_arg_index, parcsr_A, &dof_func);
      }
      else
      {
         if (myid == 0)
         {
            nalu_hypre_printf (" Number of functions = %d \n", num_functions);
         }
      }
   }

   /*-----------------------------------------------------------
    * Print out the system and initial guess
    *-----------------------------------------------------------*/

   if (negA)
   {
      nalu_hypre_ParCSRMatrixScale(parcsr_A, -1);
   }

   if (print_system)
   {
      if (ij_A)
      {
         NALU_HYPRE_IJMatrixPrint(ij_A, "IJ.out.A");
      }
      else if (parcsr_A)
      {
         nalu_hypre_ParCSRMatrixPrintIJ(parcsr_A, 0, 0, "IJ.out.A");
      }
      else
      {
         if (!myid)
         {
            nalu_hypre_printf(" Matrix A not found!\n");
         }
      }

      if (parcsr_M != parcsr_A)
      {
         if (ij_M)
         {
            NALU_HYPRE_IJMatrixPrint(ij_M, "IJ.out.M");
         }
         else
         {
            if (!myid)
            {
               nalu_hypre_printf(" Matrix M not found!\n");
            }
         }
      }

      if (ij_b)
      {
         NALU_HYPRE_IJVectorPrint(ij_b, "IJ.out.b");
      }
      else if (b)
      {
         NALU_HYPRE_ParVectorPrint(b, "ParVec.out.b");
      }
      NALU_HYPRE_IJVectorPrint(ij_x, "IJ.out.x0");
   }

   /*-----------------------------------------------------------
    * Migrate the system to the wanted memory space
    *-----------------------------------------------------------*/
   nalu_hypre_ParCSRMatrixMigrate(parcsr_A, nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   nalu_hypre_ParVectorMigrate(b, nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   nalu_hypre_ParVectorMigrate(x, nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   if (build_matrix_M == 1)
   {
      nalu_hypre_ParCSRMatrixMigrate(parcsr_M, nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
   }

   if (benchmark)
   {
      poutusr = 1;
      poutdat = 0;
      second_time = 1;
   }

   /* save the initial guess for the 2nd time */
   if (second_time)
   {
      x0_save = nalu_hypre_ParVectorCloneDeep_v2(x, nalu_hypre_ParVectorMemoryLocation(x));
   }

   /* Compute RHS squared norm */
   if (ij_b)
   {
      NALU_HYPRE_IJVectorInnerProd(ij_b, ij_b, &b_dot_b);
   }
   else if (b)
   {
      NALU_HYPRE_ParVectorInnerProd(b, b, &b_dot_b);
   }
   else
   {
      if (!myid)
      {
         nalu_hypre_printf(" Error: Vector b not set!\n");
      }
      nalu_hypre_MPI_Abort(comm, 1);
   }

   /*-----------------------------------------------------------
    * Test multivector support
    *-----------------------------------------------------------*/

   if (test_multivec && ij_b && num_components > 1)
   {
      NALU_HYPRE_IJVector   ij_bf;
      NALU_HYPRE_Complex   *d_data_full;
      NALU_HYPRE_Real       bf_dot_bf, e_dot_e;
      NALU_HYPRE_Int        num_rows_full = local_num_rows * num_components;
      NALU_HYPRE_BigInt     ilower = first_local_row * num_components;
      NALU_HYPRE_BigInt     iupper = ilower + (NALU_HYPRE_BigInt) num_rows_full;

      /* Allocate memory */
      d_data_full = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, num_rows_full, memory_location);

      /* Get values */
      for (c = 0; c < num_components; c++)
      {
         NALU_HYPRE_IJVectorSetComponent(ij_b, c);
         NALU_HYPRE_IJVectorGetValues(ij_b, local_num_rows, NULL,
                                 &d_data_full[c * local_num_rows]);
      }

      /* Create a single component vector containing all values of b */
      NALU_HYPRE_IJVectorCreate(comm, ilower, iupper, &ij_bf);
      NALU_HYPRE_IJVectorSetObjectType(ij_bf, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_bf);
      NALU_HYPRE_IJVectorSetValues(ij_bf, num_rows_full, NULL, d_data_full);
      NALU_HYPRE_IJVectorAssemble(ij_bf);
      NALU_HYPRE_IJVectorInnerProd(ij_bf, ij_bf, &bf_dot_bf);

      e_dot_e = bf_dot_bf - b_dot_b;
      if (myid == 0)
      {
         nalu_hypre_printf("\nVector/Multivector error = %e\n\n", e_dot_e);
      }

      /* Free memory */
      nalu_hypre_TFree(d_data_full, memory_location);
      NALU_HYPRE_IJVectorDestroy(ij_bf);
   }

   /*-----------------------------------------------------------
    * Perform sparse matrix/vector multiplication
    *-----------------------------------------------------------*/

   if (solver_id == -1)
   {
      NALU_HYPRE_Int num_threads = nalu_hypre_NumThreads();

      if (myid == 0)
      {
         nalu_hypre_printf("Running %d matvecs with A\n", nmv);
         nalu_hypre_printf("\n\n Num MPI tasks = %d\n\n", num_procs);
         nalu_hypre_printf(" Num OpenMP threads = %d\n\n", num_threads);
      }

      NALU_HYPRE_Real tt = nalu_hypre_MPI_Wtime();

      time_index = nalu_hypre_InitializeTiming("MatVec Test");
      nalu_hypre_BeginTiming(time_index);

      for (i = 0; i < nmv; i++)
      {
         NALU_HYPRE_ParCSRMatrixMatvec(1., parcsr_A, x, 0., b);
      }

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("MatVec Test", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      tt = nalu_hypre_MPI_Wtime() - tt;

      if (myid == 0)
      {
         nalu_hypre_printf("Matvec time %.2f (ms)\n", tt * 1000.0);
      }

      goto final;
   }

   /*-----------------------------------------------------------
    * Solve the system using the hybrid solver
    *-----------------------------------------------------------*/

   if (solver_id == 20)
   {
      if (myid == 0) { nalu_hypre_printf("Solver:  AMG\n"); }
      time_index = nalu_hypre_InitializeTiming("AMG_hybrid Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRHybridCreate(&amg_solver);
      NALU_HYPRE_ParCSRHybridSetTol(amg_solver, tol);
      NALU_HYPRE_ParCSRHybridSetAbsoluteTol(amg_solver, atol);
      NALU_HYPRE_ParCSRHybridSetConvergenceTol(amg_solver, cf_tol);
      NALU_HYPRE_ParCSRHybridSetSolverType(amg_solver, solver_type);
      NALU_HYPRE_ParCSRHybridSetRecomputeResidual(amg_solver, recompute_res);
      NALU_HYPRE_ParCSRHybridSetLogging(amg_solver, ioutdat);
      NALU_HYPRE_ParCSRHybridSetPrintLevel(amg_solver, poutdat);
      NALU_HYPRE_ParCSRHybridSetDSCGMaxIter(amg_solver, max_iter);
      NALU_HYPRE_ParCSRHybridSetPCGMaxIter(amg_solver, mg_max_iter);
      NALU_HYPRE_ParCSRHybridSetCoarsenType(amg_solver, coarsen_type);
      NALU_HYPRE_ParCSRHybridSetStrongThreshold(amg_solver, strong_threshold);
      NALU_HYPRE_ParCSRHybridSetTruncFactor(amg_solver, trunc_factor);
      NALU_HYPRE_ParCSRHybridSetPMaxElmts(amg_solver, P_max_elmts);
      NALU_HYPRE_ParCSRHybridSetMaxLevels(amg_solver, max_levels);
      NALU_HYPRE_ParCSRHybridSetMaxRowSum(amg_solver, max_row_sum);
      NALU_HYPRE_ParCSRHybridSetNumSweeps(amg_solver, num_sweeps);
      NALU_HYPRE_ParCSRHybridSetInterpType(amg_solver, interp_type);

      if (relax_type > -1) { NALU_HYPRE_ParCSRHybridSetRelaxType(amg_solver, relax_type); }
      NALU_HYPRE_ParCSRHybridSetAggNumLevels(amg_solver, agg_num_levels);
      NALU_HYPRE_ParCSRHybridSetAggInterpType(amg_solver, agg_interp_type);
      NALU_HYPRE_ParCSRHybridSetNumPaths(amg_solver, num_paths);
      NALU_HYPRE_ParCSRHybridSetNumFunctions(amg_solver, num_functions);
      NALU_HYPRE_ParCSRHybridSetNodal(amg_solver, nodal);
      if (relax_down > -1)
      {
         NALU_HYPRE_ParCSRHybridSetCycleRelaxType(amg_solver, relax_down, 1);
      }
      if (relax_up > -1)
      {
         NALU_HYPRE_ParCSRHybridSetCycleRelaxType(amg_solver, relax_up, 2);
      }
      if (relax_coarse > -1)
      {
         NALU_HYPRE_ParCSRHybridSetCycleRelaxType(amg_solver, relax_coarse, 3);
      }
      NALU_HYPRE_ParCSRHybridSetRelaxOrder(amg_solver, relax_order);
      NALU_HYPRE_ParCSRHybridSetKeepTranspose(amg_solver, keepTranspose);
      NALU_HYPRE_ParCSRHybridSetMaxCoarseSize(amg_solver, coarse_threshold);
      NALU_HYPRE_ParCSRHybridSetMinCoarseSize(amg_solver, min_coarse_size);
      NALU_HYPRE_ParCSRHybridSetSeqThreshold(amg_solver, seq_threshold);
      NALU_HYPRE_ParCSRHybridSetRelaxWt(amg_solver, relax_wt);
      NALU_HYPRE_ParCSRHybridSetOuterWt(amg_solver, outer_wt);
      if (level_w > -1)
      {
         NALU_HYPRE_ParCSRHybridSetLevelRelaxWt(amg_solver, relax_wt_level, level_w);
      }
      if (level_ow > -1)
      {
         NALU_HYPRE_ParCSRHybridSetLevelOuterWt(amg_solver, outer_wt_level, level_ow);
      }
      NALU_HYPRE_ParCSRHybridSetNonGalerkinTol(amg_solver, nongalerk_num_tol, nongalerk_tol);

      NALU_HYPRE_ParCSRHybridSetup(amg_solver, parcsr_M, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("ParCSR Hybrid Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRHybridSolve(amg_solver, parcsr_A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      if (second_time)
      {
         /* run a second time [for timings, to check for memory leaks] */
         NALU_HYPRE_ParVectorSetRandomValues(x, 775);
#if defined(NALU_HYPRE_USING_CURAND) || defined(NALU_HYPRE_USING_ROCRAND)
         nalu_hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
         nalu_hypre_ParVectorCopy(x0_save, x);

         if (myid == 0) { nalu_hypre_printf("Solver:  AMG\n"); }
         time_index = nalu_hypre_InitializeTiming("AMG_hybrid Setup");
         nalu_hypre_BeginTiming(time_index);

         NALU_HYPRE_ParCSRHybridSetup(amg_solver, parcsr_M, b, x);

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         time_index = nalu_hypre_InitializeTiming("ParCSR Hybrid Solve");
         nalu_hypre_BeginTiming(time_index);

         NALU_HYPRE_ParCSRHybridSolve(amg_solver, parcsr_A, b, x);

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         NALU_HYPRE_Real time[4];
         NALU_HYPRE_ParCSRHybridGetSetupSolveTime(amg_solver, time);

         if (myid == 0)
         {
            nalu_hypre_printf("ParCSRHybrid: Setup-Time1 %f  Solve-Time1 %f  Setup-Time2 %f  Solve-Time2 %f\n",
                         time[0], time[1], time[2], time[3]);
         }
      }

      NALU_HYPRE_ParCSRHybridGetNumIterations(amg_solver, &num_iterations);
      NALU_HYPRE_ParCSRHybridGetPCGNumIterations(amg_solver, &pcg_num_its);
      NALU_HYPRE_ParCSRHybridGetDSCGNumIterations(amg_solver, &dscg_num_its);
      NALU_HYPRE_ParCSRHybridGetFinalRelativeResidualNorm(amg_solver,
                                                     &final_res_norm);
      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Iterations = %d\n", num_iterations);
         nalu_hypre_printf("PCG_Iterations = %d\n", pcg_num_its);
         nalu_hypre_printf("DSCG_Iterations = %d\n", dscg_num_its);
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }

      NALU_HYPRE_ParCSRHybridDestroy(amg_solver);
   }
   /*-----------------------------------------------------------
    * Solve the system using AMG
    *-----------------------------------------------------------*/

   if (solver_id == 0 || solver_id == 90)
   {
      if (solver_id == 0)
      {
         if (myid == 0) { nalu_hypre_printf("Solver:  AMG\n"); }
         time_index = nalu_hypre_InitializeTiming("BoomerAMG Setup");
         nalu_hypre_BeginTiming(time_index);

         NALU_HYPRE_BoomerAMGCreate(&amg_solver);
      }
      else if (solver_id == 90)
      {
         if (myid == 0) { nalu_hypre_printf("Solver:  AMG-DD\n"); }
         time_index = nalu_hypre_InitializeTiming("BoomerAMGDD Setup");
         nalu_hypre_BeginTiming(time_index);

         NALU_HYPRE_BoomerAMGDDCreate(&amgdd_solver);
         NALU_HYPRE_BoomerAMGDDGetAMG(amgdd_solver, &amg_solver);

         /* AMG-DD options */
         NALU_HYPRE_BoomerAMGDDSetStartLevel(amgdd_solver, amgdd_start_level);
         NALU_HYPRE_BoomerAMGDDSetPadding(amgdd_solver, amgdd_padding);
         NALU_HYPRE_BoomerAMGDDSetFACNumRelax(amgdd_solver, amgdd_fac_num_relax);
         NALU_HYPRE_BoomerAMGDDSetFACNumCycles(amgdd_solver, amgdd_num_comp_cycles);
         NALU_HYPRE_BoomerAMGDDSetFACRelaxType(amgdd_solver, amgdd_fac_relax_type);
         NALU_HYPRE_BoomerAMGDDSetFACCycleType(amgdd_solver, amgdd_fac_cycle_type);
         NALU_HYPRE_BoomerAMGDDSetNumGhostLayers(amgdd_solver, amgdd_num_ghost_layers);
      }

      if (air)
      {
         /* RL: specify restriction */
         nalu_hypre_assert(restri_type >= 0);
         NALU_HYPRE_BoomerAMGSetRestriction(amg_solver, restri_type); /* 0: P^T, 1: AIR, 2: AIR-2 */
         NALU_HYPRE_BoomerAMGSetGridRelaxPoints(amg_solver, grid_relax_points);
         NALU_HYPRE_BoomerAMGSetStrongThresholdR(amg_solver, strong_thresholdR);
         NALU_HYPRE_BoomerAMGSetFilterThresholdR(amg_solver, filter_thresholdR);
      }

      /* RL */
      NALU_HYPRE_BoomerAMGSetADropTol(amg_solver, A_drop_tol);
      NALU_HYPRE_BoomerAMGSetADropType(amg_solver, A_drop_type);
      /* BM Aug 25, 2006 */
      NALU_HYPRE_BoomerAMGSetCGCIts(amg_solver, cgcits);
      NALU_HYPRE_BoomerAMGSetInterpType(amg_solver, interp_type);
      NALU_HYPRE_BoomerAMGSetRestriction(amg_solver, restri_type); /* 0: P^T, 1: AIR, 2: AIR-2 */
      NALU_HYPRE_BoomerAMGSetPostInterpType(amg_solver, post_interp_type);
      NALU_HYPRE_BoomerAMGSetNumSamples(amg_solver, gsmg_samples);
      NALU_HYPRE_BoomerAMGSetCoarsenType(amg_solver, coarsen_type);
      NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(amg_solver, coarsen_cut_factor);
      NALU_HYPRE_BoomerAMGSetCPoints(amg_solver, max_levels, num_cpt, cpt_index);
      NALU_HYPRE_BoomerAMGSetFPoints(amg_solver, num_fpt, fpt_index);
      NALU_HYPRE_BoomerAMGSetIsolatedFPoints(amg_solver, num_isolated_fpt, isolated_fpt_index);
      NALU_HYPRE_BoomerAMGSetMeasureType(amg_solver, measure_type);
      NALU_HYPRE_BoomerAMGSetConvergeType(amg_solver, converge_type);
      NALU_HYPRE_BoomerAMGSetTol(amg_solver, tol);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(amg_solver, strong_threshold);
      NALU_HYPRE_BoomerAMGSetSeqThreshold(amg_solver, seq_threshold);
      NALU_HYPRE_BoomerAMGSetRedundant(amg_solver, redundant);
      NALU_HYPRE_BoomerAMGSetMaxCoarseSize(amg_solver, coarse_threshold);
      NALU_HYPRE_BoomerAMGSetMinCoarseSize(amg_solver, min_coarse_size);
      NALU_HYPRE_BoomerAMGSetTruncFactor(amg_solver, trunc_factor);
      NALU_HYPRE_BoomerAMGSetPMaxElmts(amg_solver, P_max_elmts);
      NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(amg_solver, jacobi_trunc_threshold);
      NALU_HYPRE_BoomerAMGSetSCommPkgSwitch(amg_solver, S_commpkg_switch);
      /* note: log is written to standard output, not to file */
      NALU_HYPRE_BoomerAMGSetPrintLevel(amg_solver, poutusr ? poutdat : 3);
      //NALU_HYPRE_BoomerAMGSetLogging(amg_solver, 2);
      NALU_HYPRE_BoomerAMGSetPrintFileName(amg_solver, "driver.out.log");
      NALU_HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
      NALU_HYPRE_BoomerAMGSetFCycle(amg_solver, fcycle);
      NALU_HYPRE_BoomerAMGSetNumSweeps(amg_solver, num_sweeps);
      NALU_HYPRE_BoomerAMGSetISType(amg_solver, IS_type);
      NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(amg_solver, num_CR_relax_steps);
      NALU_HYPRE_BoomerAMGSetCRRate(amg_solver, CR_rate);
      NALU_HYPRE_BoomerAMGSetCRStrongTh(amg_solver, CR_strong_th);
      NALU_HYPRE_BoomerAMGSetCRUseCG(amg_solver, CR_use_CG);
      if (relax_type > -1) { NALU_HYPRE_BoomerAMGSetRelaxType(amg_solver, relax_type); }
      if (relax_down > -1)
      {
         NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_down, 1);
      }
      if (relax_up > -1)
      {
         NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_up, 2);
      }
      if (relax_coarse > -1)
      {
         NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_coarse, 3);
      }
      NALU_HYPRE_BoomerAMGSetAddRelaxType(amg_solver, add_relax_type);
      NALU_HYPRE_BoomerAMGSetAddRelaxWt(amg_solver, add_relax_wt);
      NALU_HYPRE_BoomerAMGSetChebyOrder(amg_solver, cheby_order);
      NALU_HYPRE_BoomerAMGSetChebyFraction(amg_solver, cheby_fraction);
      NALU_HYPRE_BoomerAMGSetChebyEigEst(amg_solver, cheby_eig_est);
      NALU_HYPRE_BoomerAMGSetChebyVariant(amg_solver, cheby_variant);
      NALU_HYPRE_BoomerAMGSetChebyScale(amg_solver, cheby_scale);
      NALU_HYPRE_BoomerAMGSetRelaxOrder(amg_solver, relax_order);
      NALU_HYPRE_BoomerAMGSetRelaxWt(amg_solver, relax_wt);
      NALU_HYPRE_BoomerAMGSetOuterWt(amg_solver, outer_wt);
      NALU_HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
      if (level_w > -1)
      {
         NALU_HYPRE_BoomerAMGSetLevelRelaxWt(amg_solver, relax_wt_level, level_w);
      }
      if (level_ow > -1)
      {
         NALU_HYPRE_BoomerAMGSetLevelOuterWt(amg_solver, outer_wt_level, level_ow);
      }
      NALU_HYPRE_BoomerAMGSetSmoothType(amg_solver, smooth_type);
      NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(amg_solver, smooth_num_sweeps);
      NALU_HYPRE_BoomerAMGSetSmoothNumLevels(amg_solver, smooth_num_levels);
      NALU_HYPRE_BoomerAMGSetMaxRowSum(amg_solver, max_row_sum);
      NALU_HYPRE_BoomerAMGSetDebugFlag(amg_solver, debug_flag);
      NALU_HYPRE_BoomerAMGSetVariant(amg_solver, variant);
      NALU_HYPRE_BoomerAMGSetOverlap(amg_solver, overlap);
      NALU_HYPRE_BoomerAMGSetDomainType(amg_solver, domain_type);
      NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm(amg_solver, use_nonsymm_schwarz);

      NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(amg_solver, schwarz_rlx_weight);
      if (eu_level < 0) { eu_level = 0; }
      NALU_HYPRE_BoomerAMGSetEuLevel(amg_solver, eu_level);
      NALU_HYPRE_BoomerAMGSetEuBJ(amg_solver, eu_bj);
      NALU_HYPRE_BoomerAMGSetEuSparseA(amg_solver, eu_sparse_A);
      NALU_HYPRE_BoomerAMGSetILUType(amg_solver, ilu_type);
      NALU_HYPRE_BoomerAMGSetILULevel(amg_solver, ilu_lfil);
      NALU_HYPRE_BoomerAMGSetILUDroptol(amg_solver, ilu_droptol);
      NALU_HYPRE_BoomerAMGSetILUMaxRowNnz(amg_solver, ilu_max_row_nnz);
      NALU_HYPRE_BoomerAMGSetILUMaxIter(amg_solver, ilu_sm_max_iter);
      NALU_HYPRE_BoomerAMGSetFSAIMaxSteps(amg_solver, fsai_max_steps);
      NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize(amg_solver, fsai_max_step_size);
      NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters(amg_solver, fsai_eig_max_iters);
      NALU_HYPRE_BoomerAMGSetFSAIKapTolerance(amg_solver, fsai_kap_tolerance);

      NALU_HYPRE_BoomerAMGSetNumFunctions(amg_solver, num_functions);
      NALU_HYPRE_BoomerAMGSetAggNumLevels(amg_solver, agg_num_levels);
      NALU_HYPRE_BoomerAMGSetAggInterpType(amg_solver, agg_interp_type);
      NALU_HYPRE_BoomerAMGSetAggTruncFactor(amg_solver, agg_trunc_factor);
      NALU_HYPRE_BoomerAMGSetAggP12TruncFactor(amg_solver, agg_P12_trunc_factor);
      NALU_HYPRE_BoomerAMGSetAggPMaxElmts(amg_solver, agg_P_max_elmts);
      NALU_HYPRE_BoomerAMGSetAggP12MaxElmts(amg_solver, agg_P12_max_elmts);
      NALU_HYPRE_BoomerAMGSetNumPaths(amg_solver, num_paths);
      NALU_HYPRE_BoomerAMGSetNodal(amg_solver, nodal);
      NALU_HYPRE_BoomerAMGSetNodalDiag(amg_solver, nodal_diag);
      NALU_HYPRE_BoomerAMGSetKeepSameSign(amg_solver, keep_same_sign);
      NALU_HYPRE_BoomerAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);
      if (ns_down > -1)
      {
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(amg_solver, ns_down,   1);
      }
      if (ns_up > -1)
      {
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(amg_solver, ns_up,     2);
      }
      if (num_functions > 1)
      {
         NALU_HYPRE_BoomerAMGSetDofFunc(amg_solver, dof_func);
      }
      NALU_HYPRE_BoomerAMGSetAdditive(amg_solver, additive);
      NALU_HYPRE_BoomerAMGSetMultAdditive(amg_solver, mult_add);
      NALU_HYPRE_BoomerAMGSetSimple(amg_solver, simple);
      NALU_HYPRE_BoomerAMGSetAddLastLvl(amg_solver, add_last_lvl);
      NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(amg_solver, add_P_max_elmts);
      NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(amg_solver, add_trunc_factor);

      NALU_HYPRE_BoomerAMGSetMaxIter(amg_solver, mg_max_iter);
      NALU_HYPRE_BoomerAMGSetRAP2(amg_solver, rap2);
      NALU_HYPRE_BoomerAMGSetModuleRAP2(amg_solver, mod_rap2);
      NALU_HYPRE_BoomerAMGSetKeepTranspose(amg_solver, keepTranspose);
#ifdef NALU_HYPRE_USING_DSUPERLU
      NALU_HYPRE_BoomerAMGSetDSLUThreshold(amg_solver, dslu_threshold);
#endif
      /*NALU_HYPRE_BoomerAMGSetNonGalerkTol(amg_solver, nongalerk_num_tol, nongalerk_tol);*/
      if (nongalerk_tol)
      {
         NALU_HYPRE_BoomerAMGSetNonGalerkinTol(amg_solver, nongalerk_tol[nongalerk_num_tol - 1]);
         for (i = 0; i < nongalerk_num_tol - 1; i++)
         {
            NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol(amg_solver, nongalerk_tol[i], i);
         }
      }
      if (build_rbm)
      {
         NALU_HYPRE_BoomerAMGSetInterpVectors(amg_solver, num_interp_vecs, interp_vecs);
         NALU_HYPRE_BoomerAMGSetInterpVecVariant(amg_solver, interp_vec_variant);
         NALU_HYPRE_BoomerAMGSetInterpVecQMax(amg_solver, Q_max);
         NALU_HYPRE_BoomerAMGSetInterpVecAbsQTrunc(amg_solver, Q_trunc);
      }

      /* BM Oct 23, 2006 */
      if (plot_grids)
      {
         NALU_HYPRE_BoomerAMGSetPlotGrids (amg_solver, 1);
         NALU_HYPRE_BoomerAMGSetPlotFileName (amg_solver, plot_file_name);
         NALU_HYPRE_BoomerAMGSetCoordDim (amg_solver, coord_dim);
         NALU_HYPRE_BoomerAMGSetCoordinates (amg_solver, coordinates);
      }

#if defined (NALU_HYPRE_USING_GPU)
      nalu_hypre_GpuProfilingPushRange("AMG-Setup-1");
#endif
      if (solver_id == 0)
      {
         NALU_HYPRE_BoomerAMGSetup(amg_solver, parcsr_M, b, x);
      }
      else if (solver_id == 90)
      {
         NALU_HYPRE_BoomerAMGDDSetup(amgdd_solver, parcsr_M, b, x);
      }

#if defined (NALU_HYPRE_USING_GPU)
      nalu_hypre_GpuProfilingPopRange();
#endif

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      if (solver_id == 0)
      {
         time_index = nalu_hypre_InitializeTiming("BoomerAMG Solve");
      }
      else if (solver_id == 90)
      {
         time_index = nalu_hypre_InitializeTiming("BoomerAMG-DD Solve");
      }
      nalu_hypre_BeginTiming(time_index);

#if defined (NALU_HYPRE_USING_GPU)
      nalu_hypre_GpuProfilingPushRange("AMG-Solve-1");
#endif

      if (solver_id == 0)
      {
         NALU_HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);
      }
      else if (solver_id == 90)
      {
         NALU_HYPRE_BoomerAMGDDSolve(amgdd_solver, parcsr_A, b, x);
      }

#if defined (NALU_HYPRE_USING_GPU)
      nalu_hypre_GpuProfilingPopRange();
#endif

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      if (second_time)
      {
         NALU_HYPRE_SetExecutionPolicy(exec2_policy);

         /* run a second time [for timings, to check for memory leaks] */
         NALU_HYPRE_ParVectorSetRandomValues(x, 775);
#if defined(NALU_HYPRE_USING_CURAND) || defined(NALU_HYPRE_USING_ROCRAND)
         nalu_hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
         nalu_hypre_ParVectorCopy(x0_save, x);

#if defined(NALU_HYPRE_USING_CUDA)
         cudaProfilerStart();
#endif

         time_index = nalu_hypre_InitializeTiming("BoomerAMG/AMG-DD Setup2");
         nalu_hypre_BeginTiming(time_index);

#if defined (NALU_HYPRE_USING_GPU)
         nalu_hypre_GpuProfilingPushRange("AMG-Setup-2");
#endif

         if (solver_id == 0)
         {
            NALU_HYPRE_BoomerAMGSetup(amg_solver, parcsr_M, b, x);
         }
         else if (solver_id == 90)
         {
            NALU_HYPRE_BoomerAMGDDSetup(amgdd_solver, parcsr_M, b, x);
         }

#if defined (NALU_HYPRE_USING_GPU)
         nalu_hypre_GpuProfilingPopRange();
#endif

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         time_index = nalu_hypre_InitializeTiming("BoomerAMG/AMG-DD Solve2");
         nalu_hypre_BeginTiming(time_index);

#if defined (NALU_HYPRE_USING_GPU)
         nalu_hypre_GpuProfilingPushRange("AMG-Solve-2");
#endif

         if (solver_id == 0)
         {
            NALU_HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);
         }
         else if (solver_id == 90)
         {
            NALU_HYPRE_BoomerAMGDDSolve(amgdd_solver, parcsr_A, b, x);
         }

#if defined (NALU_HYPRE_USING_GPU)
         nalu_hypre_GpuProfilingPopRange();
#endif

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

#if defined(NALU_HYPRE_USING_CUDA)
         cudaProfilerStop();
#endif
      }

      if (solver_id == 0)
      {
         NALU_HYPRE_BoomerAMGGetNumIterations(amg_solver, &num_iterations);
         NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm(amg_solver, &final_res_norm);
      }
      else if (solver_id == 90)
      {
         NALU_HYPRE_BoomerAMGDDGetNumIterations(amgdd_solver, &num_iterations);
         NALU_HYPRE_BoomerAMGDDGetFinalRelativeResidualNorm(amgdd_solver, &final_res_norm);
      }

      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         if (solver_id == 0)
         {
            nalu_hypre_printf("BoomerAMG Iterations = %d\n", num_iterations);
         }
         else if (solver_id == 90)
         {
            nalu_hypre_printf("BoomerAMG-DD Iterations = %d\n", num_iterations);
         }
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }

      if (solver_id == 0)
      {
         NALU_HYPRE_BoomerAMGDestroy(amg_solver);
      }
      else if (solver_id == 90)
      {
         NALU_HYPRE_BoomerAMGDDDestroy(amgdd_solver);
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using GSMG
    *-----------------------------------------------------------*/

   if (solver_id == 13)
   {
      /* reset some smoother parameters */

      relax_order = 0;

      if (myid == 0) { nalu_hypre_printf("Solver:  GSMG\n"); }
      time_index = nalu_hypre_InitializeTiming("BoomerAMG Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_BoomerAMGCreate(&amg_solver);
      NALU_HYPRE_BoomerAMGSetGSMG(amg_solver, 4); /* specify GSMG */
      /* BM Aug 25, 2006 */
      NALU_HYPRE_BoomerAMGSetCGCIts(amg_solver, cgcits);
      NALU_HYPRE_BoomerAMGSetInterpType(amg_solver, interp_type);
      NALU_HYPRE_BoomerAMGSetPostInterpType(amg_solver, post_interp_type);
      NALU_HYPRE_BoomerAMGSetNumSamples(amg_solver, gsmg_samples);
      NALU_HYPRE_BoomerAMGSetCoarsenType(amg_solver, coarsen_type);
      NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(amg_solver, coarsen_cut_factor);
      NALU_HYPRE_BoomerAMGSetCPoints(amg_solver, max_levels, num_cpt, cpt_index);
      NALU_HYPRE_BoomerAMGSetFPoints(amg_solver, num_fpt, fpt_index);
      NALU_HYPRE_BoomerAMGSetIsolatedFPoints(amg_solver, num_isolated_fpt, isolated_fpt_index);
      NALU_HYPRE_BoomerAMGSetMeasureType(amg_solver, measure_type);
      NALU_HYPRE_BoomerAMGSetTol(amg_solver, tol);
      NALU_HYPRE_BoomerAMGSetStrongThreshold(amg_solver, strong_threshold);
      NALU_HYPRE_BoomerAMGSetSeqThreshold(amg_solver, seq_threshold);
      NALU_HYPRE_BoomerAMGSetRedundant(amg_solver, redundant);
      NALU_HYPRE_BoomerAMGSetMaxCoarseSize(amg_solver, coarse_threshold);
      NALU_HYPRE_BoomerAMGSetMinCoarseSize(amg_solver, min_coarse_size);
      NALU_HYPRE_BoomerAMGSetTruncFactor(amg_solver, trunc_factor);
      NALU_HYPRE_BoomerAMGSetPMaxElmts(amg_solver, P_max_elmts);
      NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(amg_solver, jacobi_trunc_threshold);
      NALU_HYPRE_BoomerAMGSetSCommPkgSwitch(amg_solver, S_commpkg_switch);
      /* note: log is written to standard output, not to file */
      NALU_HYPRE_BoomerAMGSetPrintLevel(amg_solver, 3);
      NALU_HYPRE_BoomerAMGSetPrintFileName(amg_solver, "driver.out.log");
      NALU_HYPRE_BoomerAMGSetMaxIter(amg_solver, mg_max_iter);
      NALU_HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
      NALU_HYPRE_BoomerAMGSetFCycle(amg_solver, fcycle);
      NALU_HYPRE_BoomerAMGSetNumSweeps(amg_solver, num_sweeps);
      if (relax_type > -1) { NALU_HYPRE_BoomerAMGSetRelaxType(amg_solver, relax_type); }
      if (relax_down > -1)
      {
         NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_down, 1);
      }
      if (relax_up > -1)
      {
         NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_up, 2);
      }
      if (relax_coarse > -1)
      {
         NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_coarse, 3);
      }
      NALU_HYPRE_BoomerAMGSetAddRelaxType(amg_solver, add_relax_type);
      NALU_HYPRE_BoomerAMGSetAddRelaxWt(amg_solver, add_relax_wt);
      NALU_HYPRE_BoomerAMGSetChebyOrder(amg_solver, cheby_order);
      NALU_HYPRE_BoomerAMGSetChebyFraction(amg_solver, cheby_fraction);
      NALU_HYPRE_BoomerAMGSetChebyEigEst(amg_solver, cheby_eig_est);
      NALU_HYPRE_BoomerAMGSetChebyVariant(amg_solver, cheby_variant);
      NALU_HYPRE_BoomerAMGSetChebyScale(amg_solver, cheby_scale);
      NALU_HYPRE_BoomerAMGSetRelaxOrder(amg_solver, relax_order);
      NALU_HYPRE_BoomerAMGSetRelaxWt(amg_solver, relax_wt);
      NALU_HYPRE_BoomerAMGSetOuterWt(amg_solver, outer_wt);
      if (level_w > -1)
      {
         NALU_HYPRE_BoomerAMGSetLevelRelaxWt(amg_solver, relax_wt_level, level_w);
      }
      if (level_ow > -1)
      {
         NALU_HYPRE_BoomerAMGSetLevelOuterWt(amg_solver, outer_wt_level, level_ow);
      }
      NALU_HYPRE_BoomerAMGSetSmoothType(amg_solver, smooth_type);
      NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(amg_solver, smooth_num_sweeps);
      NALU_HYPRE_BoomerAMGSetSmoothNumLevels(amg_solver, smooth_num_levels);
      NALU_HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
      NALU_HYPRE_BoomerAMGSetMaxRowSum(amg_solver, max_row_sum);
      NALU_HYPRE_BoomerAMGSetDebugFlag(amg_solver, debug_flag);
      NALU_HYPRE_BoomerAMGSetVariant(amg_solver, variant);
      NALU_HYPRE_BoomerAMGSetOverlap(amg_solver, overlap);
      NALU_HYPRE_BoomerAMGSetDomainType(amg_solver, domain_type);
      NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm(amg_solver, use_nonsymm_schwarz);
      NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(amg_solver, schwarz_rlx_weight);
      if (eu_level < 0) { eu_level = 0; }
      NALU_HYPRE_BoomerAMGSetEuLevel(amg_solver, eu_level);
      NALU_HYPRE_BoomerAMGSetEuBJ(amg_solver, eu_bj);
      NALU_HYPRE_BoomerAMGSetEuSparseA(amg_solver, eu_sparse_A);
      NALU_HYPRE_BoomerAMGSetFSAIMaxSteps(amg_solver, fsai_max_steps);
      NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize(amg_solver, fsai_max_step_size);
      NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters(amg_solver, fsai_eig_max_iters);
      NALU_HYPRE_BoomerAMGSetFSAIKapTolerance(amg_solver, fsai_kap_tolerance);
      NALU_HYPRE_BoomerAMGSetNumFunctions(amg_solver, num_functions);
      NALU_HYPRE_BoomerAMGSetAggNumLevels(amg_solver, agg_num_levels);
      NALU_HYPRE_BoomerAMGSetAggInterpType(amg_solver, agg_interp_type);
      NALU_HYPRE_BoomerAMGSetAggTruncFactor(amg_solver, agg_trunc_factor);
      NALU_HYPRE_BoomerAMGSetAggP12TruncFactor(amg_solver, agg_P12_trunc_factor);
      NALU_HYPRE_BoomerAMGSetAggPMaxElmts(amg_solver, agg_P_max_elmts);
      NALU_HYPRE_BoomerAMGSetAggP12MaxElmts(amg_solver, agg_P12_max_elmts);
      NALU_HYPRE_BoomerAMGSetNumPaths(amg_solver, num_paths);
      NALU_HYPRE_BoomerAMGSetNodal(amg_solver, nodal);
      NALU_HYPRE_BoomerAMGSetNodalDiag(amg_solver, nodal_diag);
      if (num_functions > 1)
      {
         NALU_HYPRE_BoomerAMGSetDofFunc(amg_solver, dof_func);
      }
      NALU_HYPRE_BoomerAMGSetAdditive(amg_solver, additive);
      NALU_HYPRE_BoomerAMGSetMultAdditive(amg_solver, mult_add);
      NALU_HYPRE_BoomerAMGSetSimple(amg_solver, simple);
      NALU_HYPRE_BoomerAMGSetAddLastLvl(amg_solver, add_last_lvl);
      NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(amg_solver, add_P_max_elmts);
      NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(amg_solver, add_trunc_factor);
      NALU_HYPRE_BoomerAMGSetRAP2(amg_solver, rap2);
      NALU_HYPRE_BoomerAMGSetModuleRAP2(amg_solver, mod_rap2);
      NALU_HYPRE_BoomerAMGSetKeepTranspose(amg_solver, keepTranspose);
      if (nongalerk_tol)
      {
         NALU_HYPRE_BoomerAMGSetNonGalerkinTol(amg_solver, nongalerk_tol[nongalerk_num_tol - 1]);
         for (i = 0; i < nongalerk_num_tol - 1; i++)
         {
            NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol(amg_solver, nongalerk_tol[i], i);
         }
      }

      NALU_HYPRE_BoomerAMGSetup(amg_solver, parcsr_M, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("BoomerAMG Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      if (second_time)
      {
         /* run a second time [for timings, to check for memory leaks] */
         NALU_HYPRE_ParVectorSetRandomValues(x, 775);
#if defined(NALU_HYPRE_USING_CURAND) || defined(NALU_HYPRE_USING_ROCRAND)
         nalu_hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
         nalu_hypre_ParVectorCopy(x0_save, x);

         NALU_HYPRE_BoomerAMGSetup(amg_solver, parcsr_M, b, x);
         NALU_HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);
      }

      NALU_HYPRE_BoomerAMGDestroy(amg_solver);
   }

   if (solver_id == 999)
   {
      NALU_HYPRE_IJMatrix ij_N;

      /* use ParaSails preconditioner */
      if (myid == 0) { nalu_hypre_printf("Test ParaSails Build IJMatrix\n"); }

      NALU_HYPRE_IJMatrixPrint(ij_A, "parasails.in");

      NALU_HYPRE_ParaSailsCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);
      NALU_HYPRE_ParaSailsSetParams(pcg_precond, 0., 0);
      NALU_HYPRE_ParaSailsSetFilter(pcg_precond, 0.);
      NALU_HYPRE_ParaSailsSetLogging(pcg_precond, ioutdat);

      NALU_HYPRE_IJMatrixGetObject(ij_A, &object);
      parcsr_A = (NALU_HYPRE_ParCSRMatrix) object;

      NALU_HYPRE_ParaSailsSetup(pcg_precond, parcsr_M, NULL, NULL);
      NALU_HYPRE_ParaSailsBuildIJMatrix(pcg_precond, &ij_N);
      NALU_HYPRE_IJMatrixPrint(ij_M, "parasails.out");

      if (myid == 0) { nalu_hypre_printf("Printed to parasails.out.\n"); }
      exit(0);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------------*/

   /* begin lobpcg */
   if (!lobpcgFlag && (solver_id == 1 || solver_id == 2 || solver_id == 8 ||
                       solver_id == 12 || solver_id == 14 || solver_id == 31 ||
                       solver_id == 43 || solver_id == 71))
      /*end lobpcg */
   {
      time_index = nalu_hypre_InitializeTiming("PCG Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRPCGCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
      NALU_HYPRE_PCGSetMaxIter(pcg_solver, max_iter);
      NALU_HYPRE_PCGSetTol(pcg_solver, tol);
      NALU_HYPRE_PCGSetTwoNorm(pcg_solver, 1);
      NALU_HYPRE_PCGSetRelChange(pcg_solver, rel_change);
      NALU_HYPRE_PCGSetPrintLevel(pcg_solver, ioutdat);
      NALU_HYPRE_PCGSetAbsoluteTol(pcg_solver, atol);
      NALU_HYPRE_PCGSetRecomputeResidual(pcg_solver, recompute_res);

      if (solver_id == 1)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: AMG-PCG\n"); }
         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         /* BM Aug 25, 2006 */
         NALU_HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         NALU_HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(pcg_precond, coarsen_cut_factor);
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetCPoints(pcg_precond, max_levels, num_cpt, cpt_index);
         NALU_HYPRE_BoomerAMGSetFPoints(pcg_precond, num_fpt, fpt_index);
         NALU_HYPRE_BoomerAMGSetIsolatedFPoints(pcg_precond, num_isolated_fpt, isolated_fpt_index);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetSeqThreshold(pcg_precond, seq_threshold);
         NALU_HYPRE_BoomerAMGSetRedundant(pcg_precond, redundant);
         NALU_HYPRE_BoomerAMGSetMaxCoarseSize(pcg_precond, coarse_threshold);
         NALU_HYPRE_BoomerAMGSetMinCoarseSize(pcg_precond, min_coarse_size);
         NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         NALU_HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         NALU_HYPRE_BoomerAMGSetSCommPkgSwitch(pcg_precond, S_commpkg_switch);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, precon_cycles);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetFCycle(pcg_precond, fcycle);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         if (relax_type > -1) { NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type); }
         if (relax_down > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         NALU_HYPRE_BoomerAMGSetAddRelaxType(pcg_precond, add_relax_type);
         NALU_HYPRE_BoomerAMGSetAddRelaxWt(pcg_precond, add_relax_wt);
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         NALU_HYPRE_BoomerAMGSetChebyEigEst(pcg_precond, cheby_eig_est);
         NALU_HYPRE_BoomerAMGSetChebyVariant(pcg_precond, cheby_variant);
         NALU_HYPRE_BoomerAMGSetChebyScale(pcg_precond, cheby_scale);
         NALU_HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         NALU_HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         NALU_HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
         NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         NALU_HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
         NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         NALU_HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         NALU_HYPRE_BoomerAMGSetAggInterpType(pcg_precond, agg_interp_type);
         NALU_HYPRE_BoomerAMGSetAggTruncFactor(pcg_precond, agg_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggP12TruncFactor(pcg_precond, agg_P12_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggPMaxElmts(pcg_precond, agg_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetAggP12MaxElmts(pcg_precond, agg_P12_max_elmts);
         NALU_HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         NALU_HYPRE_BoomerAMGSetNodal(pcg_precond, nodal);
         NALU_HYPRE_BoomerAMGSetNodalDiag(pcg_precond, nodal_diag);
         NALU_HYPRE_BoomerAMGSetKeepSameSign(pcg_precond, keep_same_sign);
         NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm(pcg_precond, use_nonsymm_schwarz);
         NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         if (eu_level < 0) { eu_level = 0; }
         NALU_HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         NALU_HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         NALU_HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         NALU_HYPRE_BoomerAMGSetFSAIMaxSteps(pcg_precond, fsai_max_steps);
         NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize(pcg_precond, fsai_max_step_size);
         NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters(pcg_precond, fsai_eig_max_iters);
         NALU_HYPRE_BoomerAMGSetFSAIKapTolerance(pcg_precond, fsai_kap_tolerance);
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         NALU_HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         NALU_HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         NALU_HYPRE_BoomerAMGSetAddLastLvl(pcg_precond, add_last_lvl);
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetModuleRAP2(pcg_precond, mod_rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
#ifdef NALU_HYPRE_USING_DSUPERLU
         NALU_HYPRE_BoomerAMGSetDSLUThreshold(pcg_precond, dslu_threshold);
#endif
         if (nongalerk_tol)
         {
            NALU_HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         if (build_rbm)
         {
            NALU_HYPRE_BoomerAMGSetInterpVectors(pcg_precond, num_interp_vecs, interp_vecs);
            NALU_HYPRE_BoomerAMGSetInterpVecVariant(pcg_precond, interp_vec_variant);
            NALU_HYPRE_BoomerAMGSetInterpVecQMax(pcg_precond, Q_max);
            NALU_HYPRE_BoomerAMGSetInterpVecAbsQTrunc(pcg_precond, Q_trunc);
         }
         NALU_HYPRE_PCGSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_PCGSetPrecond(pcg_solver,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                             pcg_precond);
      }
      else if (solver_id == 2)
      {

         /* use diagonal scaling as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: DS-PCG\n"); }
         pcg_precond = NULL;

         NALU_HYPRE_PCGSetPrecond(pcg_solver,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                             pcg_precond);
      }
      else if (solver_id == 8)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: ParaSails-PCG\n"); }

         NALU_HYPRE_ParaSailsCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);
         NALU_HYPRE_ParaSailsSetParams(pcg_precond, sai_threshold, max_levels);
         NALU_HYPRE_ParaSailsSetFilter(pcg_precond, sai_filter);
         NALU_HYPRE_ParaSailsSetLogging(pcg_precond, poutdat);

         NALU_HYPRE_PCGSetPrecond(pcg_solver,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSolve,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSetup,
                             pcg_precond);
      }
      else if (solver_id == 12)
      {
         /* use Schwarz preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: Schwarz-PCG\n"); }

         NALU_HYPRE_SchwarzCreate(&pcg_precond);
         NALU_HYPRE_SchwarzSetVariant(pcg_precond, variant);
         NALU_HYPRE_SchwarzSetOverlap(pcg_precond, overlap);
         NALU_HYPRE_SchwarzSetDomainType(pcg_precond, domain_type);
         NALU_HYPRE_SchwarzSetRelaxWeight(pcg_precond, schwarz_rlx_weight);
         NALU_HYPRE_SchwarzSetNonSymm(pcg_precond, use_nonsymm_schwarz);
         NALU_HYPRE_PCGSetPrecond(pcg_solver,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SchwarzSolve,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SchwarzSetup,
                             pcg_precond);
      }
      else if (solver_id == 14)
      {
         /* use GSMG as preconditioner */

         /* reset some smoother parameters */

         /* fine grid */
         relax_order = 0;

         if (myid == 0) { nalu_hypre_printf("Solver: GSMG-PCG\n"); }
         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetGSMG(pcg_precond, 4);
         /* BM Aug 25, 2006 */
         NALU_HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         NALU_HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(pcg_precond, coarsen_cut_factor);
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetCPoints(pcg_precond, max_levels, num_cpt, cpt_index);
         NALU_HYPRE_BoomerAMGSetFPoints(pcg_precond, num_fpt, fpt_index);
         NALU_HYPRE_BoomerAMGSetIsolatedFPoints(pcg_precond, num_isolated_fpt, isolated_fpt_index);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetSeqThreshold(pcg_precond, seq_threshold);
         NALU_HYPRE_BoomerAMGSetRedundant(pcg_precond, redundant);
         NALU_HYPRE_BoomerAMGSetMaxCoarseSize(pcg_precond, coarse_threshold);
         NALU_HYPRE_BoomerAMGSetMinCoarseSize(pcg_precond, min_coarse_size);
         NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         NALU_HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         NALU_HYPRE_BoomerAMGSetSCommPkgSwitch(pcg_precond, S_commpkg_switch);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, precon_cycles);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetFCycle(pcg_precond, fcycle);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         if (relax_type > -1) { NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type); }
         if (relax_down > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         NALU_HYPRE_BoomerAMGSetAddRelaxType(pcg_precond, add_relax_type);
         NALU_HYPRE_BoomerAMGSetAddRelaxWt(pcg_precond, add_relax_wt);
         NALU_HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         NALU_HYPRE_BoomerAMGSetChebyEigEst(pcg_precond, cheby_eig_est);
         NALU_HYPRE_BoomerAMGSetChebyVariant(pcg_precond, cheby_variant);
         NALU_HYPRE_BoomerAMGSetChebyScale(pcg_precond, cheby_scale);
         NALU_HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         NALU_HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
         NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         if (eu_level < 0) { eu_level = 0; }
         NALU_HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         NALU_HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         NALU_HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         NALU_HYPRE_BoomerAMGSetFSAIMaxSteps(pcg_precond, fsai_max_steps);
         NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize(pcg_precond, fsai_max_step_size);
         NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters(pcg_precond, fsai_eig_max_iters);
         NALU_HYPRE_BoomerAMGSetFSAIKapTolerance(pcg_precond, fsai_kap_tolerance);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         NALU_HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
         NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         NALU_HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         NALU_HYPRE_BoomerAMGSetAggInterpType(pcg_precond, agg_interp_type);
         NALU_HYPRE_BoomerAMGSetAggTruncFactor(pcg_precond, agg_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggP12TruncFactor(pcg_precond, agg_P12_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggPMaxElmts(pcg_precond, agg_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetAggP12MaxElmts(pcg_precond, agg_P12_max_elmts);
         NALU_HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         NALU_HYPRE_BoomerAMGSetNodal(pcg_precond, nodal);
         NALU_HYPRE_BoomerAMGSetNodalDiag(pcg_precond, nodal_diag);
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         NALU_HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         NALU_HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         NALU_HYPRE_BoomerAMGSetAddLastLvl(pcg_precond, add_last_lvl);
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetModuleRAP2(pcg_precond, mod_rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
#ifdef NALU_HYPRE_USING_DSUPERLU
         NALU_HYPRE_BoomerAMGSetDSLUThreshold(pcg_precond, dslu_threshold);
#endif
         if (nongalerk_tol)
         {
            NALU_HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         NALU_HYPRE_PCGSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_PCGSetPrecond(pcg_solver,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                             pcg_precond);
      }
      else if (solver_id == 31)
      {
         /* use FSAI preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver: FSAI-PCG\n"); }

         NALU_HYPRE_FSAICreate(&pcg_precond);

         NALU_HYPRE_FSAISetAlgoType(pcg_precond, fsai_algo_type);
         NALU_HYPRE_FSAISetMaxSteps(pcg_precond, fsai_max_steps);
         NALU_HYPRE_FSAISetMaxStepSize(pcg_precond, fsai_max_step_size);
         NALU_HYPRE_FSAISetKapTolerance(pcg_precond, fsai_kap_tolerance);
         NALU_HYPRE_FSAISetMaxIterations(pcg_precond, 1);
         NALU_HYPRE_FSAISetTolerance(pcg_precond, 0.0);
         NALU_HYPRE_FSAISetZeroGuess(pcg_precond, 1);
         NALU_HYPRE_FSAISetPrintLevel(pcg_precond, poutdat);

         NALU_HYPRE_PCGSetPrecond(pcg_solver,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_FSAISolve,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_FSAISetup,
                             pcg_precond);
      }
      else if (solver_id == 43)
      {
         /* use Euclid preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver: Euclid-PCG\n"); }

         NALU_HYPRE_EuclidCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);

         /* note: There are three three methods of setting run-time
            parameters for Euclid: (see NALU_HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally
            parse the command line.
         */
         if (eu_level > -1) { NALU_HYPRE_EuclidSetLevel(pcg_precond, eu_level); }
         if (eu_ilut) { NALU_HYPRE_EuclidSetILUT(pcg_precond, eu_ilut); }
         if (eu_sparse_A) { NALU_HYPRE_EuclidSetSparseA(pcg_precond, eu_sparse_A); }
         if (eu_row_scale) { NALU_HYPRE_EuclidSetRowScale(pcg_precond, eu_row_scale); }
         if (eu_bj) { NALU_HYPRE_EuclidSetBJ(pcg_precond, eu_bj); }
         NALU_HYPRE_EuclidSetStats(pcg_precond, eu_stats);
         NALU_HYPRE_EuclidSetMem(pcg_precond, eu_mem);

         /*NALU_HYPRE_EuclidSetParams(pcg_precond, argc, argv);*/

         NALU_HYPRE_PCGSetPrecond(pcg_solver,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                             pcg_precond);
      }
      else if ( solver_id == 71 )
      {
         /* use MGR preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver:  MGR-PCG\n"); }

         NALU_HYPRE_MGRCreate(&pcg_precond);

         mgr_num_cindexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  mgr_nlevels, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_nlevels; i++)
         {
            /* assume 1 coarse index per level */
            mgr_num_cindexes[i] = 1;
         }
         mgr_cindexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int*,  mgr_nlevels, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_nlevels; i++)
         {
            mgr_cindexes[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  mgr_num_cindexes[i], NALU_HYPRE_MEMORY_HOST);
         }
         for (i = 0; i < mgr_nlevels; i++)
         {
            /* assume coarse point is at index 0 */
            mgr_cindexes[i][0] = 0;
         }
         mgr_reserved_coarse_indexes = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  mgr_num_reserved_nodes,
                                                     NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_num_reserved_nodes; i++)
         {
            /* generate artificial reserved nodes */
            mgr_reserved_coarse_indexes[i] = last_local_row - (NALU_HYPRE_BigInt) i; //2*i+1;
         }

         /* set MGR data by block */
         NALU_HYPRE_MGRSetCpointsByBlock( pcg_precond, mgr_bsize, mgr_nlevels, mgr_num_cindexes, mgr_cindexes);
         /* set reserved coarse nodes */
         if (mgr_num_reserved_nodes) { NALU_HYPRE_MGRSetReservedCoarseNodes(pcg_precond, mgr_num_reserved_nodes, mgr_reserved_coarse_indexes); }

         /* set intermediate coarse grid strategy */
         NALU_HYPRE_MGRSetNonCpointsToFpoints(pcg_precond, mgr_non_c_to_f);
         /* set F relaxation strategy */
         NALU_HYPRE_MGRSetFRelaxMethod(pcg_precond, mgr_frelax_method);
         /* set relax type for single level F-relaxation and post-relaxation */
         NALU_HYPRE_MGRSetRelaxType(pcg_precond, 0);
         NALU_HYPRE_MGRSetNumRelaxSweeps(pcg_precond, 2);
         /* set interpolation type */
         NALU_HYPRE_MGRSetRestrictType(pcg_precond, mgr_restrict_type);
         NALU_HYPRE_MGRSetInterpType(pcg_precond, mgr_interp_type);
         NALU_HYPRE_MGRSetNumInterpSweeps(pcg_precond, 2);
         /* set global smoother */
         NALU_HYPRE_MGRSetGlobalSmoothType(pcg_precond, mgr_gsmooth_type);
         NALU_HYPRE_MGRSetMaxGlobalSmoothIters( pcg_precond, mgr_num_gsmooth_sweeps );
         /* set print level */
         NALU_HYPRE_MGRSetPrintLevel(pcg_precond, 1);
         /* set max iterations */
         NALU_HYPRE_MGRSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_MGRSetTol(pcg_precond, pc_tol);

         /* create AMG coarse grid solver */

         NALU_HYPRE_BoomerAMGCreate(&amg_solver);

         if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
         {
            NALU_HYPRE_BoomerAMGSetInterpType(amg_solver, 18);
            NALU_HYPRE_BoomerAMGSetCoarsenType(amg_solver, 8);
            NALU_HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);
         }
         else
         {
            NALU_HYPRE_BoomerAMGSetInterpType(amg_solver, 0);
            NALU_HYPRE_BoomerAMGSetPostInterpType(amg_solver, post_interp_type);
            NALU_HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
            NALU_HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
            NALU_HYPRE_BoomerAMGSetFCycle(amg_solver, fcycle);
            NALU_HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);
            if (relax_down > -1)
            {
               NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_down, 1);
            }
            if (relax_up > -1)
            {
               NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_up, 2);
            }
            if (relax_coarse > -1)
            {
               NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_coarse, 3);
            }
            NALU_HYPRE_BoomerAMGSetSmoothType(amg_solver, smooth_type);
            NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(amg_solver, smooth_num_sweeps);
         }
         NALU_HYPRE_BoomerAMGSetCGCIts(amg_solver, cgcits);
         NALU_HYPRE_BoomerAMGSetTol(amg_solver, 0.0);
         NALU_HYPRE_BoomerAMGSetPMaxElmts(amg_solver, 0);
         NALU_HYPRE_BoomerAMGSetNumSweeps(amg_solver, num_sweeps);
         NALU_HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);
         NALU_HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxIter(amg_solver, precon_cycles);
         NALU_HYPRE_BoomerAMGSetPrintLevel(amg_solver, 1);

         /* set the MGR coarse solver. Comment out to use default CG solver in MGR */
         NALU_HYPRE_MGRSetCoarseSolver( pcg_precond, NALU_HYPRE_BoomerAMGSolve, NALU_HYPRE_BoomerAMGSetup, amg_solver);

         /* setup MGR-PCG solver */
         NALU_HYPRE_PCGSetPrecond(pcg_solver,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_MGRSolve,
                             (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_MGRSetup,
                             pcg_precond);

      }

      NALU_HYPRE_PCGGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten !=  pcg_precond)
      {
         nalu_hypre_printf("NALU_HYPRE_ParCSRPCGGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         nalu_hypre_printf("NALU_HYPRE_ParCSRPCGGetPrecond got good precond\n");
      }

#if defined (NALU_HYPRE_USING_GPU)
      nalu_hypre_GpuProfilingPushRange("PCG-Setup-1");
#endif
      NALU_HYPRE_PCGSetup(pcg_solver, (NALU_HYPRE_Matrix) parcsr_M,
                     (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x);
#if defined (NALU_HYPRE_USING_GPU)
      nalu_hypre_GpuProfilingPopRange();
#endif
      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("PCG Solve");
      nalu_hypre_BeginTiming(time_index);
#if defined (NALU_HYPRE_USING_GPU)
      nalu_hypre_GpuProfilingPushRange("PCG-Solve-1");
#endif
      NALU_HYPRE_PCGSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                     (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);
#if defined (NALU_HYPRE_USING_GPU)
      nalu_hypre_GpuProfilingPopRange();
#endif
      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      if (second_time)
      {
         NALU_HYPRE_SetExecutionPolicy(exec2_policy);

         /* run a second time [for timings, to check for memory leaks] */
         NALU_HYPRE_ParVectorSetRandomValues(x, 775);
#if defined(NALU_HYPRE_USING_CURAND) || defined(NALU_HYPRE_USING_ROCRAND)
         nalu_hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
         nalu_hypre_ParVectorCopy(x0_save, x);

#if defined(NALU_HYPRE_USING_CUDA)
         cudaProfilerStart();
#endif

         time_index = nalu_hypre_InitializeTiming("PCG Setup");
         nalu_hypre_BeginTiming(time_index);

#if defined (NALU_HYPRE_USING_GPU)
         nalu_hypre_GpuProfilingPushRange("PCG-Setup-2");
#endif

         NALU_HYPRE_PCGSetup(pcg_solver, (NALU_HYPRE_Matrix) parcsr_M,
                        (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x);

#if defined (NALU_HYPRE_USING_GPU)
         nalu_hypre_GpuProfilingPopRange();
#endif

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         time_index = nalu_hypre_InitializeTiming("PCG Solve");
         nalu_hypre_BeginTiming(time_index);

#if defined (NALU_HYPRE_USING_GPU)
         nalu_hypre_GpuProfilingPushRange("PCG-Solve-2");
#endif

         NALU_HYPRE_PCGSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                        (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

#if defined (NALU_HYPRE_USING_GPU)
         nalu_hypre_GpuProfilingPopRange();
#endif

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

#if defined(NALU_HYPRE_USING_CUDA)
         cudaProfilerStop();
#endif
      }

      NALU_HYPRE_PCGGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_PCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

      NALU_HYPRE_ParCSRPCGDestroy(pcg_solver);

      if (solver_id == 1)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      else if (solver_id == 8)
      {
         NALU_HYPRE_ParaSailsDestroy(pcg_precond);
      }
      else if (solver_id == 12)
      {
         NALU_HYPRE_SchwarzDestroy(pcg_precond);
      }
      else if (solver_id == 14)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      else if (solver_id == 31)
      {
         NALU_HYPRE_FSAIDestroy(pcg_precond);
      }
      else if (solver_id == 43)
      {
         NALU_HYPRE_EuclidDestroy(pcg_precond);
      }
      else if (solver_id == 71)
      {
         /* free memory */
         if (mgr_num_cindexes)
         {
            nalu_hypre_TFree(mgr_num_cindexes, NALU_HYPRE_MEMORY_HOST);
         }
         mgr_num_cindexes = NULL;

         if (mgr_reserved_coarse_indexes)
         {
            nalu_hypre_TFree(mgr_reserved_coarse_indexes, NALU_HYPRE_MEMORY_HOST);
         }
         mgr_reserved_coarse_indexes = NULL;

         if (mgr_cindexes)
         {
            for ( i = 0; i < mgr_nlevels; i++)
            {
               if (mgr_cindexes[i])
               {
                  nalu_hypre_TFree(mgr_cindexes[i], NALU_HYPRE_MEMORY_HOST);
               }
            }
            nalu_hypre_TFree(mgr_cindexes, NALU_HYPRE_MEMORY_HOST);
            mgr_cindexes = NULL;
         }

         NALU_HYPRE_BoomerAMGDestroy(amg_solver);
         NALU_HYPRE_MGRDestroy(pcg_precond);
      }

      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }

   }

   /* begin lobpcg */

   /*-----------------------------------------------------------
    * Solve the eigenvalue problem using LOBPCG
    *-----------------------------------------------------------*/
   if ( lobpcgFlag )
   {
      interpreter = nalu_hypre_CTAlloc(mv_InterfaceInterpreter, 1, NALU_HYPRE_MEMORY_HOST);

      NALU_HYPRE_ParCSRSetupInterpreter( interpreter );
      NALU_HYPRE_ParCSRSetupMatvec(&matvec_fn);

      if (myid != 0)
      {
         verbosity = 0;
      }

      if ( lobpcgGen )
      {
         BuildParIsoLaplacian(argc, argv, &parcsr_B);

         ierr = NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_B,
                                                 &first_local_row, &last_local_row,
                                                 &first_local_col, &last_local_col );

         local_num_rows = (NALU_HYPRE_Int)(last_local_row - first_local_row + 1);
         local_num_cols = (NALU_HYPRE_Int)(last_local_col - first_local_col + 1);
         ierr += NALU_HYPRE_ParCSRMatrixGetDims( parcsr_B, &M, &N );

         ierr += NALU_HYPRE_IJMatrixCreate( comm, first_local_row, last_local_row,
                                       first_local_col, last_local_col,
                                       &ij_B );

         ierr += NALU_HYPRE_IJMatrixSetObjectType( ij_B, NALU_HYPRE_PARCSR );

         if (sparsity_known == 1)
         {
            diag_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
            offdiag_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
            local_row = 0;
            for (big_i = first_local_row; big_i <= last_local_row; big_i++)
            {
               ierr += NALU_HYPRE_ParCSRMatrixGetRow( parcsr_B, big_i, &size,
                                                 &col_inds, &values );
               for (j = 0; j < size; j++)
               {
                  if (col_inds[j] < first_local_row || col_inds[j] > last_local_row)
                  {
                     offdiag_sizes[local_row]++;
                  }
                  else
                  {
                     diag_sizes[local_row]++;
                  }
               }
               local_row++;
               ierr += NALU_HYPRE_ParCSRMatrixRestoreRow( parcsr_B, big_i, &size,
                                                     &col_inds, &values );
            }
            ierr += NALU_HYPRE_IJMatrixSetDiagOffdSizes( ij_B,
                                                    (const NALU_HYPRE_Int *) diag_sizes,
                                                    (const NALU_HYPRE_Int *) offdiag_sizes );
            nalu_hypre_TFree(diag_sizes, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(offdiag_sizes, NALU_HYPRE_MEMORY_HOST);

            ierr = NALU_HYPRE_IJMatrixInitialize( ij_B );

            for (big_i = first_local_row; big_i <= last_local_row; big_i++)
            {
               ierr += NALU_HYPRE_ParCSRMatrixGetRow( parcsr_B, big_i, &size,
                                                 &col_inds, &values );

               ierr += NALU_HYPRE_IJMatrixSetValues( ij_B, 1, &size, &big_i,
                                                (const NALU_HYPRE_BigInt *) col_inds,
                                                (const NALU_HYPRE_Real *) values );

               ierr += NALU_HYPRE_ParCSRMatrixRestoreRow( parcsr_B, big_i, &size,
                                                     &col_inds, &values );
            }
         }
         else
         {
            row_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows, NALU_HYPRE_MEMORY_HOST);

            size = 5; /* this is in general too low, and supposed to test
                         the capability of the reallocation of the interface */

            if (sparsity_known == 0) /* tries a more accurate estimate of the
                                        storage */
            {
               if (build_matrix_type == 2) { size = 7; }
               if (build_matrix_type == 3) { size = 9; }
               if (build_matrix_type == 4) { size = 27; }
            }

            for (i = 0; i < local_num_rows; i++)
            {
               row_sizes[i] = size;
            }

            ierr = NALU_HYPRE_IJMatrixSetRowSizes ( ij_B, (const NALU_HYPRE_Int *) row_sizes );

            nalu_hypre_TFree(row_sizes, NALU_HYPRE_MEMORY_HOST);

            ierr = NALU_HYPRE_IJMatrixInitialize( ij_B );

            /* Loop through all locally stored rows and insert them into ij_matrix */
            for (big_i = first_local_row; big_i <= last_local_row; big_i++)
            {
               ierr += NALU_HYPRE_ParCSRMatrixGetRow( parcsr_B, big_i, &size,
                                                 &col_inds, &values );

               ierr += NALU_HYPRE_IJMatrixSetValues( ij_B, 1, &size, &big_i,
                                                (const NALU_HYPRE_BigInt *) col_inds,
                                                (const NALU_HYPRE_Real *) values );

               ierr += NALU_HYPRE_ParCSRMatrixRestoreRow( parcsr_B, big_i, &size,
                                                     &col_inds, &values );
            }
         }

         ierr += NALU_HYPRE_IJMatrixAssemble( ij_B );

         ierr += NALU_HYPRE_ParCSRMatrixDestroy(parcsr_B);

         ierr += NALU_HYPRE_IJMatrixGetObject( ij_B, &object);
         parcsr_B = (NALU_HYPRE_ParCSRMatrix) object;

      } /* if ( lobpcgGen ) */


      if ( pcgIterations > 0 ) /* do inner pcg iterations */
      {
         time_index = nalu_hypre_InitializeTiming("PCG Setup");
         nalu_hypre_BeginTiming(time_index);

         NALU_HYPRE_ParCSRPCGCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
         NALU_HYPRE_PCGSetMaxIter(pcg_solver, pcgIterations);
         NALU_HYPRE_PCGSetTol(pcg_solver, pcgTol);
         NALU_HYPRE_PCGSetTwoNorm(pcg_solver, two_norm);
         NALU_HYPRE_PCGSetRelChange(pcg_solver, 0);
         NALU_HYPRE_PCGSetPrintLevel(pcg_solver, 0);
         NALU_HYPRE_PCGSetRecomputeResidual(pcg_solver, recompute_res);

         NALU_HYPRE_PCGGetPrecond(pcg_solver, &pcg_precond);

         if (solver_id == 1)
         {
            /* use BoomerAMG as preconditioner */
            if (myid == 0) { nalu_hypre_printf("Solver: AMG-PCG\n"); }
            NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
            /* BM Aug 25, 2006 */
            NALU_HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
            NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
            NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
            NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
            NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
            NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(pcg_precond, coarsen_cut_factor);
            NALU_HYPRE_BoomerAMGSetCPoints(pcg_precond, max_levels, num_cpt, cpt_index);
            NALU_HYPRE_BoomerAMGSetFPoints(pcg_precond, num_fpt, fpt_index);
            NALU_HYPRE_BoomerAMGSetIsolatedFPoints(pcg_precond, num_isolated_fpt, isolated_fpt_index);
            NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
            NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
            NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
            NALU_HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
            NALU_HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
            NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
            NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
            NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
            NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, precon_cycles);
            NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
            NALU_HYPRE_BoomerAMGSetFCycle(pcg_precond, fcycle);
            NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
            if (relax_type > -1) { NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type); }
            if (relax_down > -1)
            {
               NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
            }
            if (relax_up > -1)
            {
               NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
            }
            if (relax_coarse > -1)
            {
               NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
            }
            NALU_HYPRE_BoomerAMGSetAddRelaxType(pcg_precond, add_relax_type);
            NALU_HYPRE_BoomerAMGSetAddRelaxWt(pcg_precond, add_relax_wt);
            NALU_HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
            NALU_HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
            NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
            NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
            NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
            NALU_HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
            NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
            NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
            NALU_HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
            NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
            NALU_HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
            NALU_HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
            NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
            NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
            NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
            NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
            if (num_functions > 1)
            {
               NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
            }
            NALU_HYPRE_PCGSetPrecond(pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                pcg_precond);
         }
         else if (solver_id == 2)
         {
            /* use diagonal scaling as preconditioner */
            if (myid == 0) { nalu_hypre_printf("Solver: DS-PCG\n"); }
            pcg_precond = NULL;

            NALU_HYPRE_PCGSetPrecond(pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                                pcg_precond);
         }
         else if (solver_id == 8)
         {
            /* use ParaSails preconditioner */
            if (myid == 0) { nalu_hypre_printf("Solver: ParaSails-PCG\n"); }

            NALU_HYPRE_ParaSailsCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);
            NALU_HYPRE_ParaSailsSetParams(pcg_precond, sai_threshold, max_levels);
            NALU_HYPRE_ParaSailsSetFilter(pcg_precond, sai_filter);
            NALU_HYPRE_ParaSailsSetLogging(pcg_precond, poutdat);

            NALU_HYPRE_PCGSetPrecond(pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSetup,
                                pcg_precond);
         }
         else if (solver_id == 12)
         {
            /* use Schwarz preconditioner */
            if (myid == 0) { nalu_hypre_printf("Solver: Schwarz-PCG\n"); }

            NALU_HYPRE_SchwarzCreate(&pcg_precond);
            NALU_HYPRE_SchwarzSetVariant(pcg_precond, variant);
            NALU_HYPRE_SchwarzSetOverlap(pcg_precond, overlap);
            NALU_HYPRE_SchwarzSetDomainType(pcg_precond, domain_type);
            NALU_HYPRE_SchwarzSetRelaxWeight(pcg_precond, schwarz_rlx_weight);

            NALU_HYPRE_PCGSetPrecond(pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SchwarzSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SchwarzSetup,
                                pcg_precond);
         }
         else if (solver_id == 14)
         {
            /* use GSMG as preconditioner */

            /* reset some smoother parameters */

            num_sweeps = num_sweep;
            relax_type = relax_default;
            relax_order = 0;

            if (myid == 0) { nalu_hypre_printf("Solver: GSMG-PCG\n"); }
            NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
            /* BM Aug 25, 2006 */
            NALU_HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
            NALU_HYPRE_BoomerAMGSetGSMG(pcg_precond, 4);
            NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
            NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
            NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
            NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
            NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(pcg_precond, coarsen_cut_factor);
            NALU_HYPRE_BoomerAMGSetCPoints(pcg_precond, max_levels, num_cpt, cpt_index);
            NALU_HYPRE_BoomerAMGSetFPoints(pcg_precond, num_fpt, fpt_index);
            NALU_HYPRE_BoomerAMGSetIsolatedFPoints(pcg_precond, num_isolated_fpt, isolated_fpt_index);
            NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
            NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
            NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
            NALU_HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
            NALU_HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
            NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
            NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
            NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
            NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, precon_cycles);
            NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
            NALU_HYPRE_BoomerAMGSetFCycle(pcg_precond, fcycle);
            NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
            NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
            NALU_HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
            NALU_HYPRE_BoomerAMGSetAddRelaxType(pcg_precond, add_relax_type);
            NALU_HYPRE_BoomerAMGSetAddRelaxWt(pcg_precond, add_relax_wt);
            NALU_HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
            NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
            NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
            NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
            NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
            NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
            NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
            NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
            NALU_HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
            NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
            NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
            NALU_HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
            NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
            NALU_HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
            NALU_HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
            if (num_functions > 1)
            {
               NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
            }
            NALU_HYPRE_PCGSetPrecond(pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                pcg_precond);
         }
         else if (solver_id == 43)
         {
            /* use Euclid preconditioning */
            if (myid == 0) { nalu_hypre_printf("Solver: Euclid-PCG\n"); }

            NALU_HYPRE_EuclidCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);

            /* note: There are three three methods of setting run-time
             *               parameters for Euclid: (see NALU_HYPRE_parcsr_ls.h); here
             *                             we'll use what I think is simplest: let Euclid internally
             *                                           parse the command line.
             *                                                      */
            NALU_HYPRE_EuclidSetParams(pcg_precond, argc, argv);

            NALU_HYPRE_PCGSetPrecond(pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                                pcg_precond);
         }
         else if (solver_id != NO_SOLVER )
         {
            if ( verbosity )
            {
               nalu_hypre_printf("Solver ID not recognized - running inner PCG iterations without preconditioner\n\n");
            }
         }

         NALU_HYPRE_PCGGetPrecond(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  pcg_precond)
         {
            nalu_hypre_printf("NALU_HYPRE_ParCSRPCGGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            nalu_hypre_printf("NALU_HYPRE_ParCSRPCGGetPrecond got good precond\n");
         }

         /*      NALU_HYPRE_PCGSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_M,
          *                     (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x); */

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         NALU_HYPRE_LOBPCGCreate(interpreter, &matvec_fn, &lobpcg_solver);

         NALU_HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
         NALU_HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
         NALU_HYPRE_LOBPCGSetTol(lobpcg_solver, tol);
         NALU_HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);

         NALU_HYPRE_LOBPCGSetPrecond(lobpcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_PCGSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_PCGSetup,
                                pcg_solver);

         NALU_HYPRE_LOBPCGSetupT(lobpcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                            (NALU_HYPRE_Vector)x);

         NALU_HYPRE_LOBPCGSetup(lobpcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                           (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

         if ( lobpcgGen )
            NALU_HYPRE_LOBPCGSetupB(lobpcg_solver, (NALU_HYPRE_Matrix)parcsr_B,
                               (NALU_HYPRE_Vector)x);

         if ( vFromFileFlag )
         {
            eigenvectors = mv_MultiVectorWrap( interpreter,
                                               NALU_HYPRE_ParCSRMultiVectorRead(nalu_hypre_MPI_COMM_WORLD,
                                                                           interpreter,
                                                                           "vectors" ), 1);
            nalu_hypre_assert( eigenvectors != NULL );
            blockSize = mv_MultiVectorWidth( eigenvectors );
         }
         else
         {
            eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
                                                                 blockSize,
                                                                 x );
            if ( lobpcgSeed )
            {
               mv_MultiVectorSetRandom( eigenvectors, lobpcgSeed );
            }
            else
            {
               mv_MultiVectorSetRandom( eigenvectors, (NALU_HYPRE_Int)time(0) );
            }
         }

         if ( constrained )
         {
            constraints = mv_MultiVectorWrap( interpreter,
                                              NALU_HYPRE_ParCSRMultiVectorRead(nalu_hypre_MPI_COMM_WORLD,
                                                                          interpreter,
                                                                          "vectors" ), 1);
            nalu_hypre_assert( constraints != NULL );
         }

         eigenvalues = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  blockSize, NALU_HYPRE_MEMORY_HOST);

         time_index = nalu_hypre_InitializeTiming("LOBPCG Solve");
         nalu_hypre_BeginTiming(time_index);

         NALU_HYPRE_LOBPCGSolve(lobpcg_solver, constraints, eigenvectors, eigenvalues );

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();


         if ( checkOrtho )
         {

            gramXX = utilities_FortranMatrixCreate();
            identity = utilities_FortranMatrixCreate();

            utilities_FortranMatrixAllocateData( blockSize, blockSize, gramXX );
            utilities_FortranMatrixAllocateData( blockSize, blockSize, identity );

            if ( lobpcgGen )
            {
               workspace = mv_MultiVectorCreateCopy( eigenvectors, 0 );
               nalu_hypre_LOBPCGMultiOperatorB( lobpcg_solver,
                                           mv_MultiVectorGetData(eigenvectors),
                                           mv_MultiVectorGetData(workspace) );
               lobpcg_MultiVectorByMultiVector( eigenvectors, workspace, gramXX );
            }
            else
            {
               lobpcg_MultiVectorByMultiVector( eigenvectors, eigenvectors, gramXX );
            }

            utilities_FortranMatrixSetToIdentity( identity );
            utilities_FortranMatrixAdd( -1, identity, gramXX, gramXX );
            nonOrthF = utilities_FortranMatrixFNorm( gramXX );
            if ( myid == 0 )
            {
               nalu_hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);
            }

            utilities_FortranMatrixDestroy( gramXX );
            utilities_FortranMatrixDestroy( identity );

         }

         if ( printLevel )
         {
            NALU_HYPRE_ParCSRMultiVectorPrint( mv_MultiVectorGetData(eigenvectors), "vectors" );

            if ( myid == 0 )
            {
               if ( (filePtr = fopen("values.txt", "w")) )
               {
                  nalu_hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     nalu_hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
                  }
                  fclose(filePtr);
               }

               if ( (filePtr = fopen("residuals.txt", "w")) )
               {
                  residualNorms = NALU_HYPRE_LOBPCGResidualNorms( lobpcg_solver );
                  residuals = utilities_FortranMatrixValues( residualNorms );
                  nalu_hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     nalu_hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
                  }
                  fclose(filePtr);
               }

               if ( printLevel > 1 )
               {

                  printBuffer = utilities_FortranMatrixCreate();

                  iterations = NALU_HYPRE_LOBPCGIterations( lobpcg_solver );

                  eigenvaluesHistory = NALU_HYPRE_LOBPCGEigenvaluesHistory( lobpcg_solver );
                  utilities_FortranMatrixSelectBlock( eigenvaluesHistory,
                                                      1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "val_hist.txt" );
                  residualNormsHistory = NALU_HYPRE_LOBPCGResidualNormsHistory( lobpcg_solver );
                  utilities_FortranMatrixSelectBlock(residualNormsHistory,
                                                     1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "res_hist.txt" );

                  utilities_FortranMatrixDestroy( printBuffer );
               }
            }
         }

         NALU_HYPRE_LOBPCGDestroy(lobpcg_solver);
         mv_MultiVectorDestroy( eigenvectors );
         if ( constrained )
         {
            mv_MultiVectorDestroy( constraints );
         }
         if ( lobpcgGen )
         {
            mv_MultiVectorDestroy( workspace );
         }
         nalu_hypre_TFree(eigenvalues, NALU_HYPRE_MEMORY_HOST);

         NALU_HYPRE_ParCSRPCGDestroy(pcg_solver);

         if (solver_id == 1)
         {
            NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
         }
         else if (solver_id == 8)
         {
            NALU_HYPRE_ParaSailsDestroy(pcg_precond);
         }
         else if (solver_id == 12)
         {
            NALU_HYPRE_SchwarzDestroy(pcg_precond);
         }
         else if (solver_id == 14)
         {
            NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
         }
         else if (solver_id == 43)
         {
            NALU_HYPRE_EuclidDestroy(pcg_precond);
         }

      }
      else   /* pcgIterations <= 0 --> use the preconditioner directly */
      {

         time_index = nalu_hypre_InitializeTiming("LOBPCG Setup");
         nalu_hypre_BeginTiming(time_index);
         if (myid != 0)
         {
            verbosity = 0;
         }
         NALU_HYPRE_LOBPCGCreate(interpreter, &matvec_fn, &pcg_solver);
         NALU_HYPRE_LOBPCGSetMaxIter(pcg_solver, maxIterations);
         NALU_HYPRE_LOBPCGSetTol(pcg_solver, tol);
         NALU_HYPRE_LOBPCGSetPrintLevel(pcg_solver, verbosity);

         NALU_HYPRE_LOBPCGGetPrecond(pcg_solver, &pcg_precond);

         if (solver_id == 1)
         {
            /* use BoomerAMG as preconditioner */
            if (myid == 0)
            {
               nalu_hypre_printf("Solver: AMG-PCG\n");
            }

            NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
            /* BM Aug 25, 2006 */
            NALU_HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
            NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
            NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
            NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
            NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
            NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(pcg_precond, coarsen_cut_factor);
            NALU_HYPRE_BoomerAMGSetCPoints(pcg_precond, max_levels, num_cpt, cpt_index);
            NALU_HYPRE_BoomerAMGSetFPoints(pcg_precond, num_fpt, fpt_index);
            NALU_HYPRE_BoomerAMGSetIsolatedFPoints(pcg_precond, num_isolated_fpt, isolated_fpt_index);
            NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
            NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
            NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
            NALU_HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
            NALU_HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
            NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
            NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
            NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
            NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, precon_cycles);
            NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
            NALU_HYPRE_BoomerAMGSetFCycle(pcg_precond, fcycle);
            NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
            if (relax_type > -1) { NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type); }
            if (relax_down > -1)
            {
               NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
            }
            if (relax_up > -1)
            {
               NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
            }
            if (relax_coarse > -1)
            {
               NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
            }
            NALU_HYPRE_BoomerAMGSetAddRelaxType(pcg_precond, add_relax_type);
            NALU_HYPRE_BoomerAMGSetAddRelaxWt(pcg_precond, add_relax_wt);
            NALU_HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
            NALU_HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
            NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
            NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
            NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
            NALU_HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
            NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
            NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
            NALU_HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
            NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
            NALU_HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
            NALU_HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
            NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
            NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
            NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
            NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
            if (num_functions > 1)
            {
               NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
            }
            NALU_HYPRE_LOBPCGSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                   pcg_precond);
         }
         else if (solver_id == 2)
         {

            /* use diagonal scaling as preconditioner */
            if (myid == 0)
            {
               nalu_hypre_printf("Solver: DS-PCG\n");
            }

            pcg_precond = NULL;

            NALU_HYPRE_LOBPCGSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
         }
         else if (solver_id == 8)
         {
            /* use ParaSails preconditioner */
            if (myid == 0)
            {
               nalu_hypre_printf("Solver: ParaSails-PCG\n");
            }

            NALU_HYPRE_ParaSailsCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);
            NALU_HYPRE_ParaSailsSetParams(pcg_precond, sai_threshold, max_levels);
            NALU_HYPRE_ParaSailsSetFilter(pcg_precond, sai_filter);
            NALU_HYPRE_ParaSailsSetLogging(pcg_precond, poutdat);

            NALU_HYPRE_LOBPCGSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSetup,
                                   pcg_precond);
         }
         else if (solver_id == 12)
         {
            /* use Schwarz preconditioner */
            if (myid == 0)
            {
               nalu_hypre_printf("Solver: Schwarz-PCG\n");
            }

            NALU_HYPRE_SchwarzCreate(&pcg_precond);
            NALU_HYPRE_SchwarzSetVariant(pcg_precond, variant);
            NALU_HYPRE_SchwarzSetOverlap(pcg_precond, overlap);
            NALU_HYPRE_SchwarzSetDomainType(pcg_precond, domain_type);
            NALU_HYPRE_SchwarzSetRelaxWeight(pcg_precond, schwarz_rlx_weight);

            NALU_HYPRE_LOBPCGSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SchwarzSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_SchwarzSetup,
                                   pcg_precond);
         }
         else if (solver_id == 14)
         {
            /* use GSMG as preconditioner */

            /* reset some smoother parameters */

            num_sweeps = num_sweep;
            relax_type = relax_default;
            relax_order = 0;

            if (myid == 0) { nalu_hypre_printf("Solver: GSMG-PCG\n"); }
            NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
            /* BM Aug 25, 2006 */
            NALU_HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
            NALU_HYPRE_BoomerAMGSetGSMG(pcg_precond, 4);
            NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
            NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
            NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
            NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, (hybrid * coarsen_type));
            NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(pcg_precond, coarsen_cut_factor);
            NALU_HYPRE_BoomerAMGSetCPoints(pcg_precond, max_levels, num_cpt, cpt_index);
            NALU_HYPRE_BoomerAMGSetFPoints(pcg_precond, num_fpt, fpt_index);
            NALU_HYPRE_BoomerAMGSetIsolatedFPoints(pcg_precond, num_isolated_fpt, isolated_fpt_index);
            NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
            NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
            NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
            NALU_HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
            NALU_HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
            NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
            NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
            NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
            NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, precon_cycles);
            NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
            NALU_HYPRE_BoomerAMGSetFCycle(pcg_precond, fcycle);
            NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
            NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
            NALU_HYPRE_BoomerAMGSetAddRelaxType(pcg_precond, add_relax_type);
            NALU_HYPRE_BoomerAMGSetAddRelaxWt(pcg_precond, add_relax_wt);
            NALU_HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
            NALU_HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
            NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
            NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
            NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
            NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
            NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
            NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
            NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
            NALU_HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
            NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
            NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
            NALU_HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
            NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
            NALU_HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
            NALU_HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
            if (num_functions > 1)
            {
               NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
            }

            NALU_HYPRE_LOBPCGSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                   pcg_precond);
         }
         else if (solver_id == 43)
         {
            /* use Euclid preconditioning */
            if (myid == 0)
            {
               nalu_hypre_printf("Solver: Euclid-PCG\n");
            }

            NALU_HYPRE_EuclidCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);

            /* note: There are three three methods of setting run-time
             *       parameters for Euclid: (see NALU_HYPRE_parcsr_ls.h); here we'll
             *       use what I think is simplest: let Euclid internally parse
             *       the command line. */
            NALU_HYPRE_EuclidSetParams(pcg_precond, argc, argv);

            NALU_HYPRE_LOBPCGSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                                   pcg_precond);
         }
         else if (solver_id != NO_SOLVER )
         {
            if ( verbosity )
            {
               nalu_hypre_printf("Solver ID not recognized - running LOBPCG without preconditioner\n\n");
            }
         }

         NALU_HYPRE_LOBPCGGetPrecond(pcg_solver, &pcg_precond_gotten);
         if (pcg_precond_gotten !=  pcg_precond && pcgIterations)
         {
            nalu_hypre_printf("NALU_HYPRE_ParCSRLOBPCGGetPrecond got bad precond\n");
            return (-1);
         }
         else if (myid == 0)
         {
            nalu_hypre_printf("NALU_HYPRE_ParCSRLOBPCGGetPrecond got good precond\n");
         }

         NALU_HYPRE_LOBPCGSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                           (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

         if ( lobpcgGen )
            NALU_HYPRE_LOBPCGSetupB(pcg_solver, (NALU_HYPRE_Matrix)parcsr_B,
                               (NALU_HYPRE_Vector)x);

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         if ( vFromFileFlag )
         {
            eigenvectors = mv_MultiVectorWrap( interpreter,
                                               NALU_HYPRE_ParCSRMultiVectorRead(nalu_hypre_MPI_COMM_WORLD,
                                                                           interpreter,
                                                                           "vectors" ), 1);
            nalu_hypre_assert( eigenvectors != NULL );
            blockSize = mv_MultiVectorWidth( eigenvectors );
         }
         else
         {
            eigenvectors = mv_MultiVectorCreateFromSampleVector( interpreter,
                                                                 blockSize,
                                                                 x );
            if ( lobpcgSeed )
            {
               mv_MultiVectorSetRandom( eigenvectors, lobpcgSeed );
            }
            else
            {
               mv_MultiVectorSetRandom( eigenvectors, (NALU_HYPRE_Int)time(0) );
            }
         }

         if ( constrained )
         {
            constraints = mv_MultiVectorWrap( interpreter,
                                              NALU_HYPRE_ParCSRMultiVectorRead(nalu_hypre_MPI_COMM_WORLD,
                                                                          interpreter,
                                                                          "vectors" ), 1);
            nalu_hypre_assert( constraints != NULL );
         }

         eigenvalues = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  blockSize, NALU_HYPRE_MEMORY_HOST);

         time_index = nalu_hypre_InitializeTiming("LOBPCG Solve");
         nalu_hypre_BeginTiming(time_index);

         NALU_HYPRE_LOBPCGSolve(pcg_solver, constraints, eigenvectors, eigenvalues);

         nalu_hypre_EndTiming(time_index);
         nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
         nalu_hypre_FinalizeTiming(time_index);
         nalu_hypre_ClearTiming();

         if ( checkOrtho )
         {

            gramXX = utilities_FortranMatrixCreate();
            identity = utilities_FortranMatrixCreate();

            utilities_FortranMatrixAllocateData( blockSize, blockSize, gramXX );
            utilities_FortranMatrixAllocateData( blockSize, blockSize, identity );

            if ( lobpcgGen )
            {
               workspace = mv_MultiVectorCreateCopy( eigenvectors, 0 );
               nalu_hypre_LOBPCGMultiOperatorB( pcg_solver,
                                           mv_MultiVectorGetData(eigenvectors),
                                           mv_MultiVectorGetData(workspace) );
               lobpcg_MultiVectorByMultiVector( eigenvectors, workspace, gramXX );
            }
            else
            {
               lobpcg_MultiVectorByMultiVector( eigenvectors, eigenvectors, gramXX );
            }

            utilities_FortranMatrixSetToIdentity( identity );
            utilities_FortranMatrixAdd( -1, identity, gramXX, gramXX );
            nonOrthF = utilities_FortranMatrixFNorm( gramXX );
            if ( myid == 0 )
            {
               nalu_hypre_printf("Non-orthonormality of eigenvectors: %12.5e\n", nonOrthF);
            }

            utilities_FortranMatrixDestroy( gramXX );
            utilities_FortranMatrixDestroy( identity );

         }

         if ( printLevel )
         {
            NALU_HYPRE_ParCSRMultiVectorPrint( mv_MultiVectorGetData(eigenvectors), "vectors" );

            if ( myid == 0 )
            {
               if ( (filePtr = fopen("values.txt", "w")) )
               {
                  nalu_hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     nalu_hypre_fprintf(filePtr, "%22.14e\n", eigenvalues[i]);
                  }
                  fclose(filePtr);
               }

               if ( (filePtr = fopen("residuals.txt", "w")) )
               {
                  residualNorms = NALU_HYPRE_LOBPCGResidualNorms( pcg_solver );
                  residuals = utilities_FortranMatrixValues( residualNorms );
                  nalu_hypre_fprintf(filePtr, "%d\n", blockSize);
                  for ( i = 0; i < blockSize; i++ )
                  {
                     nalu_hypre_fprintf(filePtr, "%22.14e\n", residuals[i]);
                  }
                  fclose(filePtr);
               }

               if ( printLevel > 1 )
               {

                  printBuffer = utilities_FortranMatrixCreate();

                  iterations = NALU_HYPRE_LOBPCGIterations( pcg_solver );

                  eigenvaluesHistory = NALU_HYPRE_LOBPCGEigenvaluesHistory( pcg_solver );
                  utilities_FortranMatrixSelectBlock( eigenvaluesHistory,
                                                      1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "val_hist.txt" );

                  residualNormsHistory = NALU_HYPRE_LOBPCGResidualNormsHistory( pcg_solver );
                  utilities_FortranMatrixSelectBlock(residualNormsHistory,
                                                     1, blockSize, 1, iterations + 1, printBuffer );
                  utilities_FortranMatrixPrint( printBuffer, "res_hist.txt" );

                  utilities_FortranMatrixDestroy( printBuffer );
               }
            }
         }

         if (second_time)
         {
            /* run a second time [for timings, to check for memory leaks] */
            mv_MultiVectorSetRandom( eigenvectors, 775 );
#if defined(NALU_HYPRE_USING_CURAND) || defined(NALU_HYPRE_USING_ROCRAND)
            nalu_hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
            nalu_hypre_ParVectorCopy(x0_save, x);

            NALU_HYPRE_LOBPCGSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                              (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);
            NALU_HYPRE_LOBPCGSolve(pcg_solver, constraints, eigenvectors, eigenvalues );
         }

         NALU_HYPRE_LOBPCGDestroy(pcg_solver);

         if (solver_id == 1)
         {
            NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
         }
         else if (solver_id == 8)
         {
            NALU_HYPRE_ParaSailsDestroy(pcg_precond);
         }
         else if (solver_id == 12)
         {
            NALU_HYPRE_SchwarzDestroy(pcg_precond);
         }
         else if (solver_id == 14)
         {
            NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
         }
         else if (solver_id == 43)
         {
            NALU_HYPRE_EuclidDestroy(pcg_precond);
         }

         mv_MultiVectorDestroy( eigenvectors );
         if ( constrained )
         {
            mv_MultiVectorDestroy( constraints );
         }
         if ( lobpcgGen )
         {
            mv_MultiVectorDestroy( workspace );
         }
         nalu_hypre_TFree(eigenvalues, NALU_HYPRE_MEMORY_HOST);
      } /* if ( pcgIterations > 0 ) */

      nalu_hypre_TFree( interpreter, NALU_HYPRE_MEMORY_HOST);

      if ( lobpcgGen )
      {
         NALU_HYPRE_IJMatrixDestroy(ij_B);
      }

   } /* if ( lobpcgFlag ) */

   /* end lobpcg */

   /*-----------------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   if (solver_id == 3  || solver_id == 4  || solver_id == 7  ||
       solver_id == 15 || solver_id == 18 || solver_id == 44 ||
       solver_id == 81 || solver_id == 91)
   {
      time_index = nalu_hypre_InitializeTiming("GMRES Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRGMRESCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
      NALU_HYPRE_GMRESSetKDim(pcg_solver, k_dim);
      NALU_HYPRE_GMRESSetMaxIter(pcg_solver, max_iter);
      NALU_HYPRE_GMRESSetTol(pcg_solver, tol);
      NALU_HYPRE_GMRESSetAbsoluteTol(pcg_solver, atol);
      NALU_HYPRE_GMRESSetLogging(pcg_solver, 1);
      NALU_HYPRE_GMRESSetPrintLevel(pcg_solver, ioutdat);
      NALU_HYPRE_GMRESSetRelChange(pcg_solver, rel_change);

      if (solver_id == 3 || solver_id == 91)
      {
         if (solver_id == 3)
         {
            /* use BoomerAMG as preconditioner */
            if (myid == 0) { nalu_hypre_printf("Solver: AMG-GMRES\n"); }
            NALU_HYPRE_BoomerAMGCreate(&amg_precond);
         }
         else
         {
            /* use BoomerAMG-DD as preconditioner */
            if (myid == 0) { nalu_hypre_printf("Solver: AMG-DD-GMRES\n"); }

            NALU_HYPRE_BoomerAMGDDCreate(&pcg_precond);
            NALU_HYPRE_BoomerAMGDDGetAMG(pcg_precond, &amg_precond);

            /* AMG-DD options */
            NALU_HYPRE_BoomerAMGDDSetStartLevel(pcg_precond, amgdd_start_level);
            NALU_HYPRE_BoomerAMGDDSetPadding(pcg_precond, amgdd_padding);
            NALU_HYPRE_BoomerAMGDDSetFACNumRelax(pcg_precond, amgdd_fac_num_relax);
            NALU_HYPRE_BoomerAMGDDSetFACNumCycles(pcg_precond, amgdd_num_comp_cycles);
            NALU_HYPRE_BoomerAMGDDSetFACRelaxType(pcg_precond, amgdd_fac_relax_type);
            NALU_HYPRE_BoomerAMGDDSetFACCycleType(pcg_precond, amgdd_fac_cycle_type);
            NALU_HYPRE_BoomerAMGDDSetNumGhostLayers(pcg_precond, amgdd_num_ghost_layers);
         }

         if (air)
         {
            /* RL: specify restriction */
            nalu_hypre_assert(restri_type >= 0);
            NALU_HYPRE_BoomerAMGSetRestriction(amg_precond, restri_type); /* 0: P^T, 1: AIR, 2: AIR-2 */
            NALU_HYPRE_BoomerAMGSetGridRelaxPoints(amg_precond, grid_relax_points);
            NALU_HYPRE_BoomerAMGSetStrongThresholdR(amg_precond, strong_thresholdR);
            NALU_HYPRE_BoomerAMGSetFilterThresholdR(amg_precond, filter_thresholdR);
         }

         NALU_HYPRE_BoomerAMGSetCGCIts(amg_precond, cgcits);
         NALU_HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
         NALU_HYPRE_BoomerAMGSetRestriction(amg_precond, restri_type); /* 0: P^T, 1: AIR, 2: AIR-2 */
         NALU_HYPRE_BoomerAMGSetPostInterpType(amg_precond, post_interp_type);
         NALU_HYPRE_BoomerAMGSetNumSamples(amg_precond, gsmg_samples);
         NALU_HYPRE_BoomerAMGSetTol(amg_precond, pc_tol);
         NALU_HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
         NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(amg_precond, coarsen_cut_factor);
         NALU_HYPRE_BoomerAMGSetCPoints(amg_precond, max_levels, num_cpt, cpt_index);
         NALU_HYPRE_BoomerAMGSetFPoints(amg_precond, num_fpt, fpt_index);
         NALU_HYPRE_BoomerAMGSetIsolatedFPoints(amg_precond, num_isolated_fpt, isolated_fpt_index);
         NALU_HYPRE_BoomerAMGSetMeasureType(amg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(amg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetSeqThreshold(amg_precond, seq_threshold);
         NALU_HYPRE_BoomerAMGSetRedundant(amg_precond, redundant);
         NALU_HYPRE_BoomerAMGSetMaxCoarseSize(amg_precond, coarse_threshold);
         NALU_HYPRE_BoomerAMGSetMinCoarseSize(amg_precond, min_coarse_size);
         NALU_HYPRE_BoomerAMGSetTruncFactor(amg_precond, trunc_factor);
         NALU_HYPRE_BoomerAMGSetPMaxElmts(amg_precond, P_max_elmts);
         NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(amg_precond, jacobi_trunc_threshold);
         NALU_HYPRE_BoomerAMGSetSCommPkgSwitch(amg_precond, S_commpkg_switch);
         NALU_HYPRE_BoomerAMGSetPrintLevel(amg_precond, poutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(amg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(amg_precond, precon_cycles);
         NALU_HYPRE_BoomerAMGSetCycleType(amg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetFCycle(amg_precond, fcycle);
         NALU_HYPRE_BoomerAMGSetNumSweeps(amg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(amg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(amg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(amg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(amg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(amg_precond, CR_use_CG);
         if (relax_type > -1) { NALU_HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type); }
         if (relax_down > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_coarse, 3);
         }
         NALU_HYPRE_BoomerAMGSetAddRelaxType(amg_precond, add_relax_type);
         NALU_HYPRE_BoomerAMGSetAddRelaxWt(amg_precond, add_relax_wt);
         NALU_HYPRE_BoomerAMGSetChebyOrder(amg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(amg_precond, cheby_fraction);
         NALU_HYPRE_BoomerAMGSetChebyEigEst(amg_precond, cheby_eig_est);
         NALU_HYPRE_BoomerAMGSetChebyVariant(amg_precond, cheby_variant);
         NALU_HYPRE_BoomerAMGSetChebyScale(amg_precond, cheby_scale);
         NALU_HYPRE_BoomerAMGSetRelaxOrder(amg_precond, relax_order);
         NALU_HYPRE_BoomerAMGSetRelaxWt(amg_precond, relax_wt);
         NALU_HYPRE_BoomerAMGSetOuterWt(amg_precond, outer_wt);
         if (level_w > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelRelaxWt(amg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelOuterWt(amg_precond, outer_wt_level, level_ow);
         }
         NALU_HYPRE_BoomerAMGSetSmoothType(amg_precond, smooth_type);
         NALU_HYPRE_BoomerAMGSetSmoothNumLevels(amg_precond, smooth_num_levels);
         NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(amg_precond, smooth_num_sweeps);
         NALU_HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(amg_precond, max_row_sum);
         NALU_HYPRE_BoomerAMGSetDebugFlag(amg_precond, debug_flag);
         NALU_HYPRE_BoomerAMGSetNumFunctions(amg_precond, num_functions);
         NALU_HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_num_levels);
         NALU_HYPRE_BoomerAMGSetAggInterpType(amg_precond, agg_interp_type);
         NALU_HYPRE_BoomerAMGSetAggTruncFactor(amg_precond, agg_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggP12TruncFactor(amg_precond, agg_P12_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggPMaxElmts(amg_precond, agg_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetAggP12MaxElmts(amg_precond, agg_P12_max_elmts);
         NALU_HYPRE_BoomerAMGSetNumPaths(amg_precond, num_paths);
         NALU_HYPRE_BoomerAMGSetNodal(amg_precond, nodal);
         NALU_HYPRE_BoomerAMGSetNodalDiag(amg_precond, nodal_diag);
         NALU_HYPRE_BoomerAMGSetVariant(amg_precond, variant);
         NALU_HYPRE_BoomerAMGSetOverlap(amg_precond, overlap);
         NALU_HYPRE_BoomerAMGSetDomainType(amg_precond, domain_type);
         NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm(amg_precond, use_nonsymm_schwarz);
         NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(amg_precond, schwarz_rlx_weight);
         if (eu_level < 0) { eu_level = 0; }
         NALU_HYPRE_BoomerAMGSetEuLevel(amg_precond, eu_level);
         NALU_HYPRE_BoomerAMGSetEuBJ(amg_precond, eu_bj);
         NALU_HYPRE_BoomerAMGSetEuSparseA(amg_precond, eu_sparse_A);
         NALU_HYPRE_BoomerAMGSetFSAIMaxSteps(amg_precond, fsai_max_steps);
         NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize(amg_precond, fsai_max_step_size);
         NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters(amg_precond, fsai_eig_max_iters);
         NALU_HYPRE_BoomerAMGSetFSAIKapTolerance(amg_precond, fsai_kap_tolerance);
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(amg_precond, ns_coarse, 3);
         if (ns_down > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleNumSweeps(amg_precond, ns_down,   1);
         }
         if (ns_up > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleNumSweeps(amg_precond, ns_up,     2);
         }
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(amg_precond, dof_func);
         }
         NALU_HYPRE_BoomerAMGSetAdditive(amg_precond, additive);
         NALU_HYPRE_BoomerAMGSetMultAdditive(amg_precond, mult_add);
         NALU_HYPRE_BoomerAMGSetSimple(amg_precond, simple);
         NALU_HYPRE_BoomerAMGSetAddLastLvl(amg_precond, add_last_lvl);
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(amg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(amg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(amg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetModuleRAP2(amg_precond, mod_rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(amg_precond, keepTranspose);
#ifdef NALU_HYPRE_USING_DSUPERLU
         NALU_HYPRE_BoomerAMGSetDSLUThreshold(amg_precond, dslu_threshold);
#endif
         if (nongalerk_tol)
         {
            NALU_HYPRE_BoomerAMGSetNonGalerkinTol(amg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol(amg_precond, nongalerk_tol[i], i);
            }
         }
         if (build_rbm)
         {
            NALU_HYPRE_BoomerAMGSetInterpVectors(amg_precond, 1, interp_vecs);
            NALU_HYPRE_BoomerAMGSetInterpVecVariant(amg_precond, interp_vec_variant);
            NALU_HYPRE_BoomerAMGSetInterpVecQMax(amg_precond, Q_max);
            NALU_HYPRE_BoomerAMGSetInterpVecAbsQTrunc(amg_precond, Q_trunc);
         }

         NALU_HYPRE_GMRESSetMaxIter(pcg_solver, mg_max_iter);

         if (solver_id == 3)
         {
            NALU_HYPRE_GMRESSetPrecond(pcg_solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                  amg_precond);
         }
         else if (solver_id == 91)
         {
            NALU_HYPRE_GMRESSetPrecond(pcg_solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGDDSolve,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGDDSetup,
                                  pcg_precond);
         }
      }
      else if (solver_id == 4)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: DS-GMRES\n"); }
         pcg_precond = NULL;

         NALU_HYPRE_GMRESSetPrecond(pcg_solver,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                               pcg_precond);
      }
      else if (solver_id == 7)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: PILUT-GMRES\n"); }

         ierr = NALU_HYPRE_ParCSRPilutCreate( nalu_hypre_MPI_COMM_WORLD, &pcg_precond );
         if (ierr)
         {
            nalu_hypre_printf("Error in ParPilutCreate\n");
         }

         NALU_HYPRE_GMRESSetPrecond(pcg_solver,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRPilutSolve,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRPilutSetup,
                               pcg_precond);

         if (drop_tol >= 0 )
            NALU_HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
                                               drop_tol );

         if (nonzeros_to_keep >= 0 )
            NALU_HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
                                               nonzeros_to_keep );
      }
      else if (solver_id == 15)
      {
         /* use GSMG as preconditioner */

         /* reset some smoother parameters */

         relax_order = 0;

         if (myid == 0) { nalu_hypre_printf("Solver: GSMG-GMRES\n"); }
         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetGSMG(pcg_precond, 4);
         NALU_HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         NALU_HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(pcg_precond, coarsen_cut_factor);
         NALU_HYPRE_BoomerAMGSetCPoints(pcg_precond, max_levels, num_cpt, cpt_index);
         NALU_HYPRE_BoomerAMGSetFPoints(pcg_precond, num_fpt, fpt_index);
         NALU_HYPRE_BoomerAMGSetIsolatedFPoints(pcg_precond, num_isolated_fpt, isolated_fpt_index);
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetSeqThreshold(pcg_precond, seq_threshold);
         NALU_HYPRE_BoomerAMGSetRedundant(pcg_precond, redundant);
         NALU_HYPRE_BoomerAMGSetMaxCoarseSize(pcg_precond, coarse_threshold);
         NALU_HYPRE_BoomerAMGSetMinCoarseSize(pcg_precond, min_coarse_size);
         NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         NALU_HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         NALU_HYPRE_BoomerAMGSetSCommPkgSwitch(pcg_precond, S_commpkg_switch);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, precon_cycles);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetFCycle(pcg_precond, fcycle);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         if (relax_type > -1) { NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type); }
         if (relax_down > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         NALU_HYPRE_BoomerAMGSetAddRelaxType(pcg_precond, add_relax_type);
         NALU_HYPRE_BoomerAMGSetAddRelaxWt(pcg_precond, add_relax_wt);
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         NALU_HYPRE_BoomerAMGSetChebyEigEst(pcg_precond, cheby_eig_est);
         NALU_HYPRE_BoomerAMGSetChebyVariant(pcg_precond, cheby_variant);
         NALU_HYPRE_BoomerAMGSetChebyScale(pcg_precond, cheby_scale);
         NALU_HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         NALU_HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         NALU_HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
         NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm(pcg_precond, use_nonsymm_schwarz);
         NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         if (eu_level < 0) { eu_level = 0; }
         NALU_HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         NALU_HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         NALU_HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         NALU_HYPRE_BoomerAMGSetFSAIMaxSteps(pcg_precond, fsai_max_steps);
         NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize(pcg_precond, fsai_max_step_size);
         NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters(pcg_precond, fsai_eig_max_iters);
         NALU_HYPRE_BoomerAMGSetFSAIKapTolerance(pcg_precond, fsai_kap_tolerance);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         NALU_HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
         NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         NALU_HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         NALU_HYPRE_BoomerAMGSetAggInterpType(pcg_precond, agg_interp_type);
         NALU_HYPRE_BoomerAMGSetAggTruncFactor(pcg_precond, agg_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggP12TruncFactor(pcg_precond, agg_P12_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggPMaxElmts(pcg_precond, agg_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetAggP12MaxElmts(pcg_precond, agg_P12_max_elmts);
         NALU_HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         NALU_HYPRE_BoomerAMGSetNodal(pcg_precond, nodal);
         NALU_HYPRE_BoomerAMGSetNodalDiag(pcg_precond, nodal_diag);
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         NALU_HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         NALU_HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         NALU_HYPRE_BoomerAMGSetAddLastLvl(pcg_precond, add_last_lvl);
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetModuleRAP2(pcg_precond, mod_rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
#ifdef NALU_HYPRE_USING_DSUPERLU
         NALU_HYPRE_BoomerAMGSetDSLUThreshold(pcg_precond, dslu_threshold);
#endif
         if (nongalerk_tol)
         {
            NALU_HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         NALU_HYPRE_GMRESSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_GMRESSetPrecond(pcg_solver,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                               pcg_precond);
      }
      else if (solver_id == 18)
      {
         /* use ParaSails preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: ParaSails-GMRES\n"); }

         NALU_HYPRE_ParaSailsCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);
         NALU_HYPRE_ParaSailsSetParams(pcg_precond, sai_threshold, max_levels);
         NALU_HYPRE_ParaSailsSetFilter(pcg_precond, sai_filter);
         NALU_HYPRE_ParaSailsSetLogging(pcg_precond, poutdat);
         NALU_HYPRE_ParaSailsSetSym(pcg_precond, 0);

         NALU_HYPRE_GMRESSetPrecond(pcg_solver,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSolve,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParaSailsSetup,
                               pcg_precond);
      }
      else if (solver_id == 44)
      {
         /* use Euclid preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver: Euclid-GMRES\n"); }

         NALU_HYPRE_EuclidCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);

         if (eu_level > -1) { NALU_HYPRE_EuclidSetLevel(pcg_precond, eu_level); }
         if (eu_ilut) { NALU_HYPRE_EuclidSetILUT(pcg_precond, eu_ilut); }
         if (eu_sparse_A) { NALU_HYPRE_EuclidSetSparseA(pcg_precond, eu_sparse_A); }
         if (eu_row_scale) { NALU_HYPRE_EuclidSetRowScale(pcg_precond, eu_row_scale); }
         if (eu_bj) { NALU_HYPRE_EuclidSetBJ(pcg_precond, eu_bj); }
         NALU_HYPRE_EuclidSetStats(pcg_precond, eu_stats);
         NALU_HYPRE_EuclidSetMem(pcg_precond, eu_mem);
         /*NALU_HYPRE_EuclidSetParams(pcg_precond, argc, argv);*/

         NALU_HYPRE_GMRESSetPrecond (pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                                pcg_precond);
      }
      else if (solver_id == 81)
      {
         /* use nalu_hypre_ILU preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver:  ILU-GMRES\n"); }

         /* create precon */
         NALU_HYPRE_ILUCreate(&pcg_precond);
         NALU_HYPRE_ILUSetType(pcg_precond, ilu_type);
         NALU_HYPRE_ILUSetLevelOfFill(pcg_precond, ilu_lfil);
         /* set print level */
         NALU_HYPRE_ILUSetPrintLevel(pcg_precond, 1);
         /* set max iterations */
         NALU_HYPRE_ILUSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_ILUSetTol(pcg_precond, pc_tol);
         /* set max number of nonzeros per row */
         NALU_HYPRE_ILUSetMaxNnzPerRow(pcg_precond, ilu_max_row_nnz);
         /* set the droptol */
         NALU_HYPRE_ILUSetDropThreshold(pcg_precond, ilu_droptol);
         /* set max iterations for Schur system solve */
         NALU_HYPRE_ILUSetSchurMaxIter( pcg_precond, ilu_schur_max_iter );
         if (ilu_type == 20 || ilu_type == 21)
         {
            NALU_HYPRE_ILUSetNSHDropThreshold( pcg_precond, ilu_nsh_droptol);
         }

         /* setup ILU-GMRES solver */
         NALU_HYPRE_GMRESSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_GMRESSetPrecond(pcg_solver,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ILUSolve,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ILUSetup,
                               pcg_precond);
      }

      NALU_HYPRE_GMRESGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != ((solver_id == 3) ? amg_precond : pcg_precond))
      {
         nalu_hypre_printf("NALU_HYPRE_GMRESGetPrecond got bad precond\n");
         return (-1);
      }
      else
      {
         if (myid == 0)
         {
            nalu_hypre_printf("NALU_HYPRE_GMRESGetPrecond got good precond\n");
         }
      }
      NALU_HYPRE_GMRESSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_M, (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("GMRES Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_GMRESSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      if (check_residual)
      {
         NALU_HYPRE_BigInt *indices_h, *indices_d;
         NALU_HYPRE_Complex *values_h, *values_d;
         NALU_HYPRE_Int num_values = 20;
         NALU_HYPRE_ParCSRGMRESGetResidual(pcg_solver, &residual);
         NALU_HYPRE_ParCSRMatrixGetLocalRange( parcsr_A,
                                          &first_local_row, &last_local_row,
                                          &first_local_col, &last_local_col );
         local_num_rows = (NALU_HYPRE_Int)(last_local_row - first_local_row + 1);
         if (local_num_rows < 20)
         {
            num_values = local_num_rows;
         }
         indices_h = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_values, NALU_HYPRE_MEMORY_HOST);
         values_h = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_values, NALU_HYPRE_MEMORY_HOST);
         indices_d = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_values, memory_location);
         values_d = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_values, memory_location);
         for (i = 0; i < num_values; i++)
         {
            indices_h[i] = first_local_row + i;
         }
         nalu_hypre_TMemcpy(indices_d, indices_h, NALU_HYPRE_BigInt, num_values, memory_location,
                       NALU_HYPRE_MEMORY_HOST);

         NALU_HYPRE_ParVectorGetValues((NALU_HYPRE_ParVector) residual, num_values, indices_d, values_d);

         nalu_hypre_TMemcpy(values_h, values_d, NALU_HYPRE_Complex, num_values, NALU_HYPRE_MEMORY_HOST,
                       memory_location);

         for (i = 0; i < num_values; i++)
         {
            if (myid == 0)
            {
               nalu_hypre_printf("index %d value %e\n", i, values_h[i]);
            }
         }
         nalu_hypre_TFree(indices_h, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(values_h, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(indices_d, memory_location);
         nalu_hypre_TFree(values_d, memory_location);
      }

      if (second_time)
      {
         /* run a second time [for timings, to check for memory leaks] */
         NALU_HYPRE_ParVectorSetRandomValues(x, 775);
#if defined(NALU_HYPRE_USING_CURAND) || defined(NALU_HYPRE_USING_ROCRAND)
         nalu_hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
         nalu_hypre_ParVectorCopy(x0_save, x);

         NALU_HYPRE_GMRESSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_M, (NALU_HYPRE_Vector)b,
                          (NALU_HYPRE_Vector)x);
         NALU_HYPRE_GMRESSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b,
                          (NALU_HYPRE_Vector)x);
      }

      NALU_HYPRE_GMRESGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_GMRESGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

      NALU_HYPRE_ParCSRGMRESDestroy(pcg_solver);

      if (solver_id == 3)
      {
         NALU_HYPRE_BoomerAMGDestroy(amg_precond);
      }
      else if (solver_id == 15)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      else if (solver_id == 7)
      {
         NALU_HYPRE_ParCSRPilutDestroy(pcg_precond);
      }
      else if (solver_id == 18)
      {
         NALU_HYPRE_ParaSailsDestroy(pcg_precond);
      }
      else if (solver_id == 44)
      {
         NALU_HYPRE_EuclidDestroy(pcg_precond);
      }
      else if (solver_id == 81)
      {
         NALU_HYPRE_ILUDestroy(pcg_precond);
      }
      else if (solver_id == 91)
      {
         NALU_HYPRE_BoomerAMGDDDestroy(pcg_precond);
      }

      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("GMRES Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final GMRES Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }
   }
   /*-----------------------------------------------------------
    * Solve the system using LGMRES
    *-----------------------------------------------------------*/

   if (solver_id == 50 || solver_id == 51 )
   {
      time_index = nalu_hypre_InitializeTiming("LGMRES Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRLGMRESCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
      NALU_HYPRE_LGMRESSetKDim(pcg_solver, k_dim);
      NALU_HYPRE_LGMRESSetAugDim(pcg_solver, aug_dim);
      NALU_HYPRE_LGMRESSetMaxIter(pcg_solver, max_iter);
      NALU_HYPRE_LGMRESSetTol(pcg_solver, tol);
      NALU_HYPRE_LGMRESSetAbsoluteTol(pcg_solver, atol);
      NALU_HYPRE_LGMRESSetLogging(pcg_solver, 1);
      NALU_HYPRE_LGMRESSetPrintLevel(pcg_solver, ioutdat);

      if (solver_id == 51)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: AMG-LGMRES\n"); }

         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         NALU_HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(pcg_precond, coarsen_cut_factor);
         NALU_HYPRE_BoomerAMGSetCPoints(pcg_precond, max_levels, num_cpt, cpt_index);
         NALU_HYPRE_BoomerAMGSetFPoints(pcg_precond, num_fpt, fpt_index);
         NALU_HYPRE_BoomerAMGSetIsolatedFPoints(pcg_precond, num_isolated_fpt, isolated_fpt_index);
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetSeqThreshold(pcg_precond, seq_threshold);
         NALU_HYPRE_BoomerAMGSetRedundant(pcg_precond, redundant);
         NALU_HYPRE_BoomerAMGSetMaxCoarseSize(pcg_precond, coarse_threshold);
         NALU_HYPRE_BoomerAMGSetMinCoarseSize(pcg_precond, min_coarse_size);
         NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         NALU_HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         NALU_HYPRE_BoomerAMGSetSCommPkgSwitch(pcg_precond, S_commpkg_switch);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, precon_cycles);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetFCycle(pcg_precond, fcycle);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         if (relax_type > -1) { NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type); }
         if (relax_down > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         NALU_HYPRE_BoomerAMGSetAddRelaxType(pcg_precond, add_relax_type);
         NALU_HYPRE_BoomerAMGSetAddRelaxWt(pcg_precond, add_relax_wt);
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         NALU_HYPRE_BoomerAMGSetChebyEigEst(pcg_precond, cheby_eig_est);
         NALU_HYPRE_BoomerAMGSetChebyVariant(pcg_precond, cheby_variant);
         NALU_HYPRE_BoomerAMGSetChebyScale(pcg_precond, cheby_scale);
         NALU_HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         NALU_HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         NALU_HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
         NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         NALU_HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
         NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         NALU_HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         NALU_HYPRE_BoomerAMGSetAggInterpType(pcg_precond, agg_interp_type);
         NALU_HYPRE_BoomerAMGSetAggTruncFactor(pcg_precond, agg_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggP12TruncFactor(pcg_precond, agg_P12_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggPMaxElmts(pcg_precond, agg_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetAggP12MaxElmts(pcg_precond, agg_P12_max_elmts);
         NALU_HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         NALU_HYPRE_BoomerAMGSetNodal(pcg_precond, nodal);
         NALU_HYPRE_BoomerAMGSetNodalDiag(pcg_precond, nodal_diag);
         NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm(pcg_precond, use_nonsymm_schwarz);
         NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         if (eu_level < 0) { eu_level = 0; }
         NALU_HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         NALU_HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         NALU_HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         NALU_HYPRE_BoomerAMGSetFSAIMaxSteps(pcg_precond, fsai_max_steps);
         NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize(pcg_precond, fsai_max_step_size);
         NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters(pcg_precond, fsai_eig_max_iters);
         NALU_HYPRE_BoomerAMGSetFSAIKapTolerance(pcg_precond, fsai_kap_tolerance);
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         NALU_HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         NALU_HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         NALU_HYPRE_BoomerAMGSetAddLastLvl(pcg_precond, add_last_lvl);
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetModuleRAP2(pcg_precond, mod_rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
#ifdef NALU_HYPRE_USING_DSUPERLU
         NALU_HYPRE_BoomerAMGSetDSLUThreshold(pcg_precond, dslu_threshold);
#endif
         if (nongalerk_tol)
         {
            NALU_HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         NALU_HYPRE_LGMRESSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_LGMRESSetPrecond(pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                pcg_precond);
      }
      else if (solver_id == 50)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: DS-LGMRES\n"); }
         pcg_precond = NULL;

         NALU_HYPRE_LGMRESSetPrecond(pcg_solver,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                                pcg_precond);
      }

      NALU_HYPRE_LGMRESGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
         nalu_hypre_printf("NALU_HYPRE_LGMRESGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         nalu_hypre_printf("NALU_HYPRE_LGMRESGetPrecond got good precond\n");
      }
      NALU_HYPRE_LGMRESSetup
      (pcg_solver, (NALU_HYPRE_Matrix)parcsr_M, (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("LGMRES Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_LGMRESSolve
      (pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_LGMRESGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_LGMRESGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

      NALU_HYPRE_ParCSRLGMRESDestroy(pcg_solver);

      if (solver_id == 51)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }

      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("LGMRES Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final LGMRES Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using FlexGMRES
    *-----------------------------------------------------------*/

   if (solver_id == 60 || solver_id == 61 || solver_id == 72 || solver_id == 82 || solver_id == 47)
   {
      time_index = nalu_hypre_InitializeTiming("FlexGMRES Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRFlexGMRESCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
      NALU_HYPRE_FlexGMRESSetKDim(pcg_solver, k_dim);
      NALU_HYPRE_FlexGMRESSetMaxIter(pcg_solver, max_iter);
      NALU_HYPRE_FlexGMRESSetTol(pcg_solver, tol);
      NALU_HYPRE_FlexGMRESSetAbsoluteTol(pcg_solver, atol);
      NALU_HYPRE_FlexGMRESSetLogging(pcg_solver, 1);
      NALU_HYPRE_FlexGMRESSetPrintLevel(pcg_solver, ioutdat);

      if (solver_id == 61)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: AMG-FlexGMRES\n"); }

         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         NALU_HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(pcg_precond, coarsen_cut_factor);
         NALU_HYPRE_BoomerAMGSetCPoints(pcg_precond, max_levels, num_cpt, cpt_index);
         NALU_HYPRE_BoomerAMGSetFPoints(pcg_precond, num_fpt, fpt_index);
         NALU_HYPRE_BoomerAMGSetIsolatedFPoints(pcg_precond, num_isolated_fpt, isolated_fpt_index);
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetSeqThreshold(pcg_precond, seq_threshold);
         NALU_HYPRE_BoomerAMGSetRedundant(pcg_precond, redundant);
         NALU_HYPRE_BoomerAMGSetMaxCoarseSize(pcg_precond, coarse_threshold);
         NALU_HYPRE_BoomerAMGSetMinCoarseSize(pcg_precond, min_coarse_size);
         NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         NALU_HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         NALU_HYPRE_BoomerAMGSetSCommPkgSwitch(pcg_precond, S_commpkg_switch);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, precon_cycles);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetFCycle(pcg_precond, fcycle);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         if (relax_type > -1) { NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type); }
         if (relax_down > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         NALU_HYPRE_BoomerAMGSetAddRelaxType(pcg_precond, add_relax_type);
         NALU_HYPRE_BoomerAMGSetAddRelaxWt(pcg_precond, add_relax_wt);
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         NALU_HYPRE_BoomerAMGSetChebyEigEst(pcg_precond, cheby_eig_est);
         NALU_HYPRE_BoomerAMGSetChebyVariant(pcg_precond, cheby_variant);
         NALU_HYPRE_BoomerAMGSetChebyScale(pcg_precond, cheby_scale);
         NALU_HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         NALU_HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         NALU_HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
         NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         NALU_HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
         NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         NALU_HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         NALU_HYPRE_BoomerAMGSetAggInterpType(pcg_precond, agg_interp_type);
         NALU_HYPRE_BoomerAMGSetAggTruncFactor(pcg_precond, agg_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggP12TruncFactor(pcg_precond, agg_P12_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggPMaxElmts(pcg_precond, agg_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetAggP12MaxElmts(pcg_precond, agg_P12_max_elmts);
         NALU_HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         NALU_HYPRE_BoomerAMGSetNodal(pcg_precond, nodal);
         NALU_HYPRE_BoomerAMGSetNodalDiag(pcg_precond, nodal_diag);
         NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm(pcg_precond, use_nonsymm_schwarz);
         NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         if (eu_level < 0) { eu_level = 0; }
         NALU_HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         NALU_HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         NALU_HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         NALU_HYPRE_BoomerAMGSetFSAIMaxSteps(pcg_precond, fsai_max_steps);
         NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize(pcg_precond, fsai_max_step_size);
         NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters(pcg_precond, fsai_eig_max_iters);
         NALU_HYPRE_BoomerAMGSetFSAIKapTolerance(pcg_precond, fsai_kap_tolerance);
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         NALU_HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         NALU_HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         NALU_HYPRE_BoomerAMGSetAddLastLvl(pcg_precond, add_last_lvl);
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetModuleRAP2(pcg_precond, mod_rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
#ifdef NALU_HYPRE_USING_DSUPERLU
         NALU_HYPRE_BoomerAMGSetDSLUThreshold(pcg_precond, dslu_threshold);
#endif
         if (nongalerk_tol)
         {
            NALU_HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         NALU_HYPRE_FlexGMRESSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_FlexGMRESSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                   pcg_precond);
      }
      else if ( solver_id == 72 )
      {
         /* use MGR preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver:  MGR-FlexGMRES\n"); }

         NALU_HYPRE_MGRCreate(&pcg_precond);

         mgr_num_cindexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  mgr_nlevels, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_nlevels; i++)
         {
            /* assume 1 coarse index per level */
            mgr_num_cindexes[i] = 1;
         }
         mgr_cindexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int*,  mgr_nlevels, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_nlevels; i++)
         {
            mgr_cindexes[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  mgr_num_cindexes[i], NALU_HYPRE_MEMORY_HOST);
         }
         for (i = 0; i < mgr_nlevels; i++)
         {
            /* assume coarse point is at index 0 */
            mgr_cindexes[i][0] = 0;
         }
         mgr_reserved_coarse_indexes = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  mgr_num_reserved_nodes,
                                                     NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_num_reserved_nodes; i++)
         {
            /* generate artificial reserved nodes */
            mgr_reserved_coarse_indexes[i] = last_local_row - (NALU_HYPRE_BigInt) i; //2*i+1;
         }

         /* set MGR data by block */
         NALU_HYPRE_MGRSetCpointsByBlock( pcg_precond, mgr_bsize, mgr_nlevels, mgr_num_cindexes, mgr_cindexes);
         /* set reserved coarse nodes */
         if (mgr_num_reserved_nodes) { NALU_HYPRE_MGRSetReservedCoarseNodes(pcg_precond, mgr_num_reserved_nodes, mgr_reserved_coarse_indexes); }

         /* set intermediate coarse grid strategy */
         NALU_HYPRE_MGRSetNonCpointsToFpoints(pcg_precond, mgr_non_c_to_f);
         /* set F relaxation strategy */
         NALU_HYPRE_MGRSetFRelaxMethod(pcg_precond, mgr_frelax_method);
         /* set relax type for single level F-relaxation and post-relaxation */
         NALU_HYPRE_MGRSetRelaxType(pcg_precond, mgr_relax_type);
         NALU_HYPRE_MGRSetNumRelaxSweeps(pcg_precond, mgr_num_relax_sweeps);
         /* set interpolation type */
         NALU_HYPRE_MGRSetInterpType(pcg_precond, mgr_interp_type);
         NALU_HYPRE_MGRSetNumInterpSweeps(pcg_precond, mgr_num_interp_sweeps);
         /* set print level */
         NALU_HYPRE_MGRSetPrintLevel(pcg_precond, 1);
         /* set max iterations */
         NALU_HYPRE_MGRSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_MGRSetTol(pcg_precond, pc_tol);
         /* set global smoother */
         NALU_HYPRE_MGRSetGlobalSmoothType(pcg_precond, mgr_gsmooth_type);
         NALU_HYPRE_MGRSetMaxGlobalSmoothIters( pcg_precond, mgr_num_gsmooth_sweeps );

         /* create AMG coarse grid solver */

         NALU_HYPRE_BoomerAMGCreate(&amg_solver);
         if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
         {
            NALU_HYPRE_BoomerAMGSetInterpType(amg_solver, 18);
            NALU_HYPRE_BoomerAMGSetCoarsenType(amg_solver, 8);
            NALU_HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);
         }
         else
         {
            NALU_HYPRE_BoomerAMGSetInterpType(amg_solver, 0);
            NALU_HYPRE_BoomerAMGSetPostInterpType(amg_solver, post_interp_type);
            NALU_HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
            NALU_HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
            NALU_HYPRE_BoomerAMGSetFCycle(amg_solver, fcycle);
            NALU_HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);
            if (relax_down > -1)
            {
               NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_down, 1);
            }
            if (relax_up > -1)
            {
               NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_up, 2);
            }
            if (relax_coarse > -1)
            {
               NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_coarse, 3);
            }
            NALU_HYPRE_BoomerAMGSetSmoothType(amg_solver, smooth_type);
            NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(amg_solver, smooth_num_sweeps);
         }
         NALU_HYPRE_BoomerAMGSetCGCIts(amg_solver, cgcits);
         NALU_HYPRE_BoomerAMGSetTol(amg_solver, 0.0);
         NALU_HYPRE_BoomerAMGSetPMaxElmts(amg_solver, 0);
         NALU_HYPRE_BoomerAMGSetNumSweeps(amg_solver, num_sweeps);
         NALU_HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);
         NALU_HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxIter(amg_solver, precon_cycles);
         NALU_HYPRE_BoomerAMGSetPrintLevel(amg_solver, 1);

         /* set the MGR coarse solver. Comment out to use default CG solver in MGR */
         NALU_HYPRE_MGRSetCoarseSolver( pcg_precond, NALU_HYPRE_BoomerAMGSolve, NALU_HYPRE_BoomerAMGSetup, amg_solver);

         /* setup MGR-PCG solver */
         NALU_HYPRE_FlexGMRESSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_FlexGMRESSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_MGRSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_MGRSetup,
                                   pcg_precond);
      }
      else if (solver_id == 47)
      {
         /* use Euclid preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver: Euclid-FlexGMRES\n"); }

         NALU_HYPRE_EuclidCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);

         if (eu_level > -1) { NALU_HYPRE_EuclidSetLevel(pcg_precond, eu_level); }
         if (eu_ilut) { NALU_HYPRE_EuclidSetILUT(pcg_precond, eu_ilut); }
         if (eu_sparse_A) { NALU_HYPRE_EuclidSetSparseA(pcg_precond, eu_sparse_A); }
         if (eu_row_scale) { NALU_HYPRE_EuclidSetRowScale(pcg_precond, eu_row_scale); }
         if (eu_bj) { NALU_HYPRE_EuclidSetBJ(pcg_precond, eu_bj); }
         NALU_HYPRE_EuclidSetStats(pcg_precond, eu_stats);
         NALU_HYPRE_EuclidSetMem(pcg_precond, eu_mem);
         /*NALU_HYPRE_EuclidSetParams(pcg_precond, argc, argv);*/

         /* setup MGR-PCG solver */
         NALU_HYPRE_FlexGMRESSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_FlexGMRESSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                                   pcg_precond);
      }
      else if (solver_id == 82)
      {
         /* use nalu_hypre_ILU preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver:  ILU-FlexGMRES\n"); }

         /* create precon */
         NALU_HYPRE_ILUCreate(&pcg_precond);
         NALU_HYPRE_ILUSetType(pcg_precond, ilu_type);
         NALU_HYPRE_ILUSetLevelOfFill(pcg_precond, ilu_lfil);
         /* set print level */
         NALU_HYPRE_ILUSetPrintLevel(pcg_precond, 1);
         /* set max iterations */
         NALU_HYPRE_ILUSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_ILUSetTol(pcg_precond, pc_tol);
         /* set max number of nonzeros per row */
         NALU_HYPRE_ILUSetMaxNnzPerRow(pcg_precond, ilu_max_row_nnz);
         /* set the droptol */
         NALU_HYPRE_ILUSetDropThreshold(pcg_precond, ilu_droptol);
         /* set max iterations for Schur system solve */
         NALU_HYPRE_ILUSetSchurMaxIter( pcg_precond, ilu_schur_max_iter );
         NALU_HYPRE_ILUSetNSHDropThreshold( pcg_precond, ilu_nsh_droptol);

         /* setup MGR-PCG solver */
         NALU_HYPRE_FlexGMRESSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_FlexGMRESSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ILUSolve,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ILUSetup,
                                   pcg_precond);
      }
      else if (solver_id == 60)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: DS-FlexGMRES\n"); }
         pcg_precond = NULL;

         NALU_HYPRE_FlexGMRESSetPrecond(pcg_solver,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                   (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                                   pcg_precond);
      }

      NALU_HYPRE_FlexGMRESGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
         nalu_hypre_printf("NALU_HYPRE_FlexGMRESGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         nalu_hypre_printf("NALU_HYPRE_FlexGMRESGetPrecond got good precond\n");
      }


      /* this is optional - could be a user defined one instead (see ex5.c)*/
      NALU_HYPRE_FlexGMRESSetModifyPC( pcg_solver,
                                  (NALU_HYPRE_PtrToModifyPCFcn) nalu_hypre_FlexGMRESModifyPCDefault);


      NALU_HYPRE_FlexGMRESSetup
      (pcg_solver, (NALU_HYPRE_Matrix)parcsr_M, (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("FlexGMRES Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_FlexGMRESSolve
      (pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_FlexGMRESGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_FlexGMRESGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

      NALU_HYPRE_ParCSRFlexGMRESDestroy(pcg_solver);

      if (solver_id == 61)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      else if (solver_id == 72)
      {
         /* free memory */
         if (mgr_num_cindexes)
         {
            nalu_hypre_TFree(mgr_num_cindexes, NALU_HYPRE_MEMORY_HOST);
         }
         mgr_num_cindexes = NULL;

         if (mgr_reserved_coarse_indexes)
         {
            nalu_hypre_TFree(mgr_reserved_coarse_indexes, NALU_HYPRE_MEMORY_HOST);
         }
         mgr_reserved_coarse_indexes = NULL;

         if (mgr_cindexes)
         {
            for ( i = 0; i < mgr_nlevels; i++)
            {
               if (mgr_cindexes[i])
               {
                  nalu_hypre_TFree(mgr_cindexes[i], NALU_HYPRE_MEMORY_HOST);
               }
            }
            nalu_hypre_TFree(mgr_cindexes, NALU_HYPRE_MEMORY_HOST);
            mgr_cindexes = NULL;
         }

         NALU_HYPRE_BoomerAMGDestroy(amg_solver);
         NALU_HYPRE_MGRDestroy(pcg_precond);
      }
      else if (solver_id == 47)
      {
         NALU_HYPRE_EuclidDestroy(pcg_precond);
      }
      else if (solver_id == 82)
      {
         NALU_HYPRE_ILUDestroy(pcg_precond);
      }
      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("FlexGMRES Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final FlexGMRES Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using BiCGSTAB
    *-----------------------------------------------------------*/

   if (solver_id == 9 || solver_id == 10 || solver_id == 11 || solver_id == 45 || solver_id == 73)
   {
      time_index = nalu_hypre_InitializeTiming("BiCGSTAB Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRBiCGSTABCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
      NALU_HYPRE_BiCGSTABSetMaxIter(pcg_solver, max_iter);
      NALU_HYPRE_BiCGSTABSetTol(pcg_solver, tol);
      NALU_HYPRE_BiCGSTABSetAbsoluteTol(pcg_solver, atol);
      NALU_HYPRE_BiCGSTABSetLogging(pcg_solver, ioutdat);
      NALU_HYPRE_BiCGSTABSetPrintLevel(pcg_solver, ioutdat);

      if (solver_id == 9)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: AMG-BiCGSTAB\n"); }
         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         NALU_HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(pcg_precond, coarsen_cut_factor);
         NALU_HYPRE_BoomerAMGSetCPoints(pcg_precond, max_levels, num_cpt, cpt_index);
         NALU_HYPRE_BoomerAMGSetFPoints(pcg_precond, num_fpt, fpt_index);
         NALU_HYPRE_BoomerAMGSetIsolatedFPoints(pcg_precond, num_isolated_fpt, isolated_fpt_index);
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetSeqThreshold(pcg_precond, seq_threshold);
         NALU_HYPRE_BoomerAMGSetRedundant(pcg_precond, redundant);
         NALU_HYPRE_BoomerAMGSetMaxCoarseSize(pcg_precond, coarse_threshold);
         NALU_HYPRE_BoomerAMGSetMinCoarseSize(pcg_precond, min_coarse_size);
         NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         NALU_HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         NALU_HYPRE_BoomerAMGSetSCommPkgSwitch(pcg_precond, S_commpkg_switch);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, precon_cycles);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetFCycle(pcg_precond, fcycle);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         if (relax_type > -1) { NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type); }
         if (relax_down > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         NALU_HYPRE_BoomerAMGSetAddRelaxType(pcg_precond, add_relax_type);
         NALU_HYPRE_BoomerAMGSetAddRelaxWt(pcg_precond, add_relax_wt);
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         NALU_HYPRE_BoomerAMGSetChebyEigEst(pcg_precond, cheby_eig_est);
         NALU_HYPRE_BoomerAMGSetChebyVariant(pcg_precond, cheby_variant);
         NALU_HYPRE_BoomerAMGSetChebyScale(pcg_precond, cheby_scale);
         NALU_HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         NALU_HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         NALU_HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
         NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         NALU_HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
         NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         NALU_HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         NALU_HYPRE_BoomerAMGSetAggInterpType(pcg_precond, agg_interp_type);
         NALU_HYPRE_BoomerAMGSetAggTruncFactor(pcg_precond, agg_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggP12TruncFactor(pcg_precond, agg_P12_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggPMaxElmts(pcg_precond, agg_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetAggP12MaxElmts(pcg_precond, agg_P12_max_elmts);
         NALU_HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         NALU_HYPRE_BoomerAMGSetNodal(pcg_precond, nodal);
         NALU_HYPRE_BoomerAMGSetNodalDiag(pcg_precond, nodal_diag);
         NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm(pcg_precond, use_nonsymm_schwarz);

         NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         if (eu_level < 0) { eu_level = 0; }
         NALU_HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         NALU_HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         NALU_HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         NALU_HYPRE_BoomerAMGSetFSAIMaxSteps(pcg_precond, fsai_max_steps);
         NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize(pcg_precond, fsai_max_step_size);
         NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters(pcg_precond, fsai_eig_max_iters);
         NALU_HYPRE_BoomerAMGSetFSAIKapTolerance(pcg_precond, fsai_kap_tolerance);
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         NALU_HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         NALU_HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         NALU_HYPRE_BoomerAMGSetAddLastLvl(pcg_precond, add_last_lvl);
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetModuleRAP2(pcg_precond, mod_rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
#ifdef NALU_HYPRE_USING_DSUPERLU
         NALU_HYPRE_BoomerAMGSetDSLUThreshold(pcg_precond, dslu_threshold);
#endif
         if (nongalerk_tol)
         {
            NALU_HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         NALU_HYPRE_BiCGSTABSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                  pcg_precond);
      }
      else if (solver_id == 10)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: DS-BiCGSTAB\n"); }
         pcg_precond = NULL;

         NALU_HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                                  pcg_precond);
      }
      else if (solver_id == 11)
      {
         /* use PILUT as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: PILUT-BiCGSTAB\n"); }

         ierr = NALU_HYPRE_ParCSRPilutCreate( nalu_hypre_MPI_COMM_WORLD, &pcg_precond );
         if (ierr)
         {
            nalu_hypre_printf("Error in ParPilutCreate\n");
         }

         NALU_HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRPilutSolve,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRPilutSetup,
                                  pcg_precond);

         NALU_HYPRE_ParCSRPilutSetLogging(pcg_precond, 0);

         if (drop_tol >= 0 )
            NALU_HYPRE_ParCSRPilutSetDropTolerance( pcg_precond,
                                               drop_tol );

         if (nonzeros_to_keep >= 0 )
            NALU_HYPRE_ParCSRPilutSetFactorRowSize( pcg_precond,
                                               nonzeros_to_keep );
      }
      else if (solver_id == 45)
      {
         /* use Euclid preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver: Euclid-BICGSTAB\n"); }

         NALU_HYPRE_EuclidCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);

         /* note: There are three three methods of setting run-time
            parameters for Euclid: (see NALU_HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally
            parse the command line.
         */
         if (eu_level > -1) { NALU_HYPRE_EuclidSetLevel(pcg_precond, eu_level); }
         if (eu_ilut) { NALU_HYPRE_EuclidSetILUT(pcg_precond, eu_ilut); }
         if (eu_sparse_A) { NALU_HYPRE_EuclidSetSparseA(pcg_precond, eu_sparse_A); }
         if (eu_row_scale) { NALU_HYPRE_EuclidSetRowScale(pcg_precond, eu_row_scale); }
         if (eu_bj) { NALU_HYPRE_EuclidSetBJ(pcg_precond, eu_bj); }
         NALU_HYPRE_EuclidSetStats(pcg_precond, eu_stats);
         NALU_HYPRE_EuclidSetMem(pcg_precond, eu_mem);

         /*NALU_HYPRE_EuclidSetParams(pcg_precond, argc, argv);*/

         NALU_HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                                  pcg_precond);
      }
      else if (solver_id == 73)
      {
         /* use MGR preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver:  MGR-BICGSTAB\n"); }

         NALU_HYPRE_MGRCreate(&pcg_precond);

         mgr_num_cindexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int, mgr_nlevels, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_nlevels; i++)
         {
            /* assume 1 coarse index per level */
            mgr_num_cindexes[i] = 1;
         }
         mgr_cindexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, mgr_nlevels, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_nlevels; i++)
         {
            mgr_cindexes[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, mgr_num_cindexes[i], NALU_HYPRE_MEMORY_HOST);
         }
         for (i = 0; i < mgr_nlevels; i++)
         {
            /* assume coarse point is at index 0 */
            mgr_cindexes[i][0] = 2;
         }

         mgr_reserved_coarse_indexes = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, mgr_num_reserved_nodes,
                                                     NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_num_reserved_nodes; i++)
         {
            /* Generate 'artificial' reserved nodes. Assumes these are ordered last in the system */
            mgr_reserved_coarse_indexes[i] = last_local_row - (NALU_HYPRE_BigInt) i; //2*i+1;
            //            nalu_hypre_printf("mgr_reserved_coarse_indexes[i] = %b \n", mgr_reserved_coarse_indexes[i]);
         }

         /* set MGR data by block */
         NALU_HYPRE_MGRSetCpointsByBlock( pcg_precond, mgr_bsize, mgr_nlevels, mgr_num_cindexes, mgr_cindexes);
         /* set reserved coarse nodes */
         if (mgr_num_reserved_nodes) { NALU_HYPRE_MGRSetReservedCoarseNodes(pcg_precond, mgr_num_reserved_nodes, mgr_reserved_coarse_indexes); }

         /* set intermediate coarse grid strategy */
         NALU_HYPRE_MGRSetNonCpointsToFpoints(pcg_precond, mgr_non_c_to_f);
         /* set F relaxation strategy */
         NALU_HYPRE_MGRSetFRelaxMethod(pcg_precond, mgr_frelax_method);
         /* set relax type for single level F-relaxation and post-relaxation */
         NALU_HYPRE_MGRSetRelaxType(pcg_precond, mgr_relax_type);
         NALU_HYPRE_MGRSetNumRelaxSweeps(pcg_precond, mgr_num_relax_sweeps);
         /* set interpolation type */
         NALU_HYPRE_MGRSetRestrictType(pcg_precond, mgr_restrict_type);
         NALU_HYPRE_MGRSetNumRestrictSweeps(pcg_precond, mgr_num_restrict_sweeps);
         NALU_HYPRE_MGRSetInterpType(pcg_precond, mgr_interp_type);
         NALU_HYPRE_MGRSetNumInterpSweeps(pcg_precond, mgr_num_interp_sweeps);
         /* set print level */
         NALU_HYPRE_MGRSetPrintLevel(pcg_precond, 1);
         /* set max iterations */
         NALU_HYPRE_MGRSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_MGRSetTol(pcg_precond, pc_tol);
         /* set global smoother */
         NALU_HYPRE_MGRSetGlobalSmoothType(pcg_precond, mgr_gsmooth_type);
         NALU_HYPRE_MGRSetMaxGlobalSmoothIters( pcg_precond, mgr_num_gsmooth_sweeps );

         /* create AMG coarse grid solver */

         NALU_HYPRE_BoomerAMGCreate(&amg_solver);

         if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
         {
            NALU_HYPRE_BoomerAMGSetInterpType(amg_solver, 18);
            NALU_HYPRE_BoomerAMGSetCoarsenType(amg_solver, 8);
            NALU_HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);
         }
         else
         {
            NALU_HYPRE_BoomerAMGSetInterpType(amg_solver, 0);
            NALU_HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
            NALU_HYPRE_BoomerAMGSetCycleType(amg_solver, 1);
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, 14, 1);
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, 14, 2);
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, 9, 3);
         }
         NALU_HYPRE_BoomerAMGSetTol(amg_solver, pc_tol);
         NALU_HYPRE_BoomerAMGSetPMaxElmts(amg_solver, 0);
         NALU_HYPRE_BoomerAMGSetNumSweeps(amg_solver, 1);
         NALU_HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);
         NALU_HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxIter(amg_solver, precon_cycles);
         NALU_HYPRE_BoomerAMGSetPrintLevel(amg_solver, 1);

         /* set the MGR coarse solver. Comment out to use default CG solver in MGR */
         NALU_HYPRE_MGRSetCoarseSolver( pcg_precond, NALU_HYPRE_BoomerAMGSolve, NALU_HYPRE_BoomerAMGSetup, amg_solver);


         /* setup MGR-BiCGSTAB solver */
         NALU_HYPRE_BiCGSTABSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_BiCGSTABSetPrecond(pcg_solver,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_MGRSolve,
                                  (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_MGRSetup,
                                  pcg_precond);
      }

      NALU_HYPRE_BiCGSTABSetup(pcg_solver, (NALU_HYPRE_Matrix) parcsr_M,
                          (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("BiCGSTAB Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_BiCGSTABSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                          (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      if (second_time)
      {
         /* run a second time [for timings, to check for memory leaks] */
         NALU_HYPRE_ParVectorSetRandomValues(x, 775);
#if defined(NALU_HYPRE_USING_CURAND) || defined(NALU_HYPRE_USING_ROCRAND)
         nalu_hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
         nalu_hypre_ParVectorCopy(x0_save, x);

         NALU_HYPRE_BiCGSTABSetup(pcg_solver, (NALU_HYPRE_Matrix) parcsr_M,
                             (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x);
         NALU_HYPRE_BiCGSTABSolve(pcg_solver, (NALU_HYPRE_Matrix) parcsr_A,
                             (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x);
      }

      NALU_HYPRE_BiCGSTABGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_BiCGSTABGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
      NALU_HYPRE_ParCSRBiCGSTABDestroy(pcg_solver);

      if (solver_id == 9)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }

      if (solver_id == 11)
      {
         NALU_HYPRE_ParCSRPilutDestroy(pcg_precond);
      }
      else if (solver_id == 45)
      {
         NALU_HYPRE_EuclidDestroy(pcg_precond);
      }
      else if (solver_id == 73)
      {
         /* free memory */
         if (mgr_num_cindexes)
         {
            nalu_hypre_TFree(mgr_num_cindexes, NALU_HYPRE_MEMORY_HOST);
         }
         mgr_num_cindexes = NULL;

         if (mgr_reserved_coarse_indexes)
         {
            nalu_hypre_TFree(mgr_reserved_coarse_indexes, NALU_HYPRE_MEMORY_HOST);
         }
         mgr_reserved_coarse_indexes = NULL;

         if (mgr_cindexes)
         {
            for ( i = 0; i < mgr_nlevels; i++)
            {
               if (mgr_cindexes[i])
               {
                  nalu_hypre_TFree(mgr_cindexes[i], NALU_HYPRE_MEMORY_HOST);
               }
            }
            nalu_hypre_TFree(mgr_cindexes, NALU_HYPRE_MEMORY_HOST);
            mgr_cindexes = NULL;
         }

         NALU_HYPRE_BoomerAMGDestroy(amg_solver);
         NALU_HYPRE_MGRDestroy(pcg_precond);
      }
      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("BiCGSTAB Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final BiCGSTAB Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using COGMRES
    *-----------------------------------------------------------*/

   if (solver_id == 16 || solver_id == 17 || solver_id == 46 || solver_id == 74)
   {
      time_index = nalu_hypre_InitializeTiming("COGMRES Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRCOGMRESCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
      NALU_HYPRE_COGMRESSetKDim(pcg_solver, k_dim);
      NALU_HYPRE_COGMRESSetUnroll(pcg_solver, unroll);
      NALU_HYPRE_COGMRESSetCGS(pcg_solver, cgs);
      NALU_HYPRE_COGMRESSetMaxIter(pcg_solver, max_iter);
      NALU_HYPRE_COGMRESSetTol(pcg_solver, tol);
      NALU_HYPRE_COGMRESSetAbsoluteTol(pcg_solver, atol);
      NALU_HYPRE_COGMRESSetLogging(pcg_solver, ioutdat);
      NALU_HYPRE_COGMRESSetPrintLevel(pcg_solver, ioutdat);

      if (solver_id == 16)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: AMG-COGMRES\n"); }
         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         NALU_HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(pcg_precond, coarsen_cut_factor);
         NALU_HYPRE_BoomerAMGSetCPoints(pcg_precond, max_levels, num_cpt, cpt_index);
         NALU_HYPRE_BoomerAMGSetFPoints(pcg_precond, num_fpt, fpt_index);
         NALU_HYPRE_BoomerAMGSetIsolatedFPoints(pcg_precond, num_isolated_fpt, isolated_fpt_index);
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetSeqThreshold(pcg_precond, seq_threshold);
         NALU_HYPRE_BoomerAMGSetRedundant(pcg_precond, redundant);
         NALU_HYPRE_BoomerAMGSetMaxCoarseSize(pcg_precond, coarse_threshold);
         NALU_HYPRE_BoomerAMGSetMinCoarseSize(pcg_precond, min_coarse_size);
         NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         NALU_HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         NALU_HYPRE_BoomerAMGSetSCommPkgSwitch(pcg_precond, S_commpkg_switch);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, precon_cycles);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetFCycle(pcg_precond, fcycle);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         if (relax_type > -1) { NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type); }
         if (relax_down > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         NALU_HYPRE_BoomerAMGSetAddRelaxType(pcg_precond, add_relax_type);
         NALU_HYPRE_BoomerAMGSetAddRelaxWt(pcg_precond, add_relax_wt);
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         NALU_HYPRE_BoomerAMGSetChebyEigEst(pcg_precond, cheby_eig_est);
         NALU_HYPRE_BoomerAMGSetChebyVariant(pcg_precond, cheby_variant);
         NALU_HYPRE_BoomerAMGSetChebyScale(pcg_precond, cheby_scale);
         NALU_HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         NALU_HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         NALU_HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
         NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         NALU_HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
         NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         NALU_HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         NALU_HYPRE_BoomerAMGSetAggInterpType(pcg_precond, agg_interp_type);
         NALU_HYPRE_BoomerAMGSetAggTruncFactor(pcg_precond, agg_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggP12TruncFactor(pcg_precond, agg_P12_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggPMaxElmts(pcg_precond, agg_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetAggP12MaxElmts(pcg_precond, agg_P12_max_elmts);
         NALU_HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         NALU_HYPRE_BoomerAMGSetNodal(pcg_precond, nodal);
         NALU_HYPRE_BoomerAMGSetNodalDiag(pcg_precond, nodal_diag);
         NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         NALU_HYPRE_BoomerAMGSetSchwarzUseNonSymm(pcg_precond, use_nonsymm_schwarz);

         NALU_HYPRE_BoomerAMGSetSchwarzRlxWeight(pcg_precond, schwarz_rlx_weight);
         if (eu_level < 0) { eu_level = 0; }
         NALU_HYPRE_BoomerAMGSetEuLevel(pcg_precond, eu_level);
         NALU_HYPRE_BoomerAMGSetEuBJ(pcg_precond, eu_bj);
         NALU_HYPRE_BoomerAMGSetEuSparseA(pcg_precond, eu_sparse_A);
         NALU_HYPRE_BoomerAMGSetFSAIMaxSteps(pcg_precond, fsai_max_steps);
         NALU_HYPRE_BoomerAMGSetFSAIMaxStepSize(pcg_precond, fsai_max_step_size);
         NALU_HYPRE_BoomerAMGSetFSAIEigMaxIters(pcg_precond, fsai_eig_max_iters);
         NALU_HYPRE_BoomerAMGSetFSAIKapTolerance(pcg_precond, fsai_kap_tolerance);
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         NALU_HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         NALU_HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         NALU_HYPRE_BoomerAMGSetAddLastLvl(pcg_precond, add_last_lvl);
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetModuleRAP2(pcg_precond, mod_rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
#ifdef NALU_HYPRE_USING_DSUPERLU
         NALU_HYPRE_BoomerAMGSetDSLUThreshold(pcg_precond, dslu_threshold);
#endif
         if (nongalerk_tol)
         {
            NALU_HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         NALU_HYPRE_COGMRESSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_COGMRESSetPrecond(pcg_solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                                 pcg_precond);
      }
      else if (solver_id == 17)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: DS-COGMRES\n"); }
         pcg_precond = NULL;

         NALU_HYPRE_COGMRESSetPrecond(pcg_solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                                 pcg_precond);
      }
      else if (solver_id == 46)
      {
         /* use Euclid preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver: Euclid-BICGSTAB\n"); }

         NALU_HYPRE_EuclidCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);

         /* note: There are three three methods of setting run-time
            parameters for Euclid: (see NALU_HYPRE_parcsr_ls.h); here
            we'll use what I think is simplest: let Euclid internally
            parse the command line.
         */
         if (eu_level > -1) { NALU_HYPRE_EuclidSetLevel(pcg_precond, eu_level); }
         if (eu_ilut) { NALU_HYPRE_EuclidSetILUT(pcg_precond, eu_ilut); }
         if (eu_sparse_A) { NALU_HYPRE_EuclidSetSparseA(pcg_precond, eu_sparse_A); }
         if (eu_row_scale) { NALU_HYPRE_EuclidSetRowScale(pcg_precond, eu_row_scale); }
         if (eu_bj) { NALU_HYPRE_EuclidSetBJ(pcg_precond, eu_bj); }
         NALU_HYPRE_EuclidSetStats(pcg_precond, eu_stats);
         NALU_HYPRE_EuclidSetMem(pcg_precond, eu_mem);

         /*NALU_HYPRE_EuclidSetParams(pcg_precond, argc, argv);*/

         NALU_HYPRE_COGMRESSetPrecond(pcg_solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_EuclidSetup,
                                 pcg_precond);
      }
      else if (solver_id == 74)
      {
         /* use MGR preconditioning */
         if (myid == 0) { nalu_hypre_printf("Solver:  MGR-BICGSTAB\n"); }

         NALU_HYPRE_MGRCreate(&pcg_precond);

         mgr_num_cindexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int, mgr_nlevels, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_nlevels; i++)
         {
            /* assume 1 coarse index per level */
            mgr_num_cindexes[i] = 1;
         }
         mgr_cindexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, mgr_nlevels, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_nlevels; i++)
         {
            mgr_cindexes[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, mgr_num_cindexes[i], NALU_HYPRE_MEMORY_HOST);
         }
         for (i = 0; i < mgr_nlevels; i++)
         {
            /* assume coarse point is at index 0 */
            mgr_cindexes[i][0] = 2;
         }

         mgr_reserved_coarse_indexes = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, mgr_num_reserved_nodes,
                                                     NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < mgr_num_reserved_nodes; i++)
         {
            /* Generate 'artificial' reserved nodes. Assumes these are ordered last in the system */
            mgr_reserved_coarse_indexes[i] = last_local_row - (NALU_HYPRE_BigInt) i; //2*i+1;
            //            nalu_hypre_printf("mgr_reserved_coarse_indexes[i] = %b \n", mgr_reserved_coarse_indexes[i]);
         }

         /* set MGR data by block */
         NALU_HYPRE_MGRSetCpointsByBlock( pcg_precond, mgr_bsize, mgr_nlevels, mgr_num_cindexes, mgr_cindexes);
         /* set reserved coarse nodes */
         if (mgr_num_reserved_nodes) { NALU_HYPRE_MGRSetReservedCoarseNodes(pcg_precond, mgr_num_reserved_nodes, mgr_reserved_coarse_indexes); }

         /* set intermediate coarse grid strategy */
         NALU_HYPRE_MGRSetNonCpointsToFpoints(pcg_precond, mgr_non_c_to_f);
         /* set F relaxation strategy */
         NALU_HYPRE_MGRSetFRelaxMethod(pcg_precond, mgr_frelax_method);
         /* set relax type for single level F-relaxation and post-relaxation */
         NALU_HYPRE_MGRSetRelaxType(pcg_precond, mgr_relax_type);
         NALU_HYPRE_MGRSetNumRelaxSweeps(pcg_precond, mgr_num_relax_sweeps);
         /* set interpolation type */
         NALU_HYPRE_MGRSetRestrictType(pcg_precond, mgr_restrict_type);
         NALU_HYPRE_MGRSetNumRestrictSweeps(pcg_precond, mgr_num_restrict_sweeps);
         NALU_HYPRE_MGRSetInterpType(pcg_precond, mgr_interp_type);
         NALU_HYPRE_MGRSetNumInterpSweeps(pcg_precond, mgr_num_interp_sweeps);
         /* set print level */
         NALU_HYPRE_MGRSetPrintLevel(pcg_precond, 1);
         /* set max iterations */
         NALU_HYPRE_MGRSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_MGRSetTol(pcg_precond, pc_tol);
         /* set global smoother */
         NALU_HYPRE_MGRSetGlobalSmoothType(pcg_precond, mgr_gsmooth_type);
         NALU_HYPRE_MGRSetMaxGlobalSmoothIters( pcg_precond, mgr_num_gsmooth_sweeps );

         /* create AMG coarse grid solver */

         NALU_HYPRE_BoomerAMGCreate(&amg_solver);
         NALU_HYPRE_BoomerAMGSetTol(amg_solver, pc_tol);
         NALU_HYPRE_BoomerAMGSetPrintLevel(amg_solver, 1);

         NALU_HYPRE_BoomerAMGSetMaxIter(amg_solver, precon_cycles);

         NALU_HYPRE_BoomerAMGSetCycleType(amg_solver, 1);
         NALU_HYPRE_BoomerAMGSetNumSweeps(amg_solver, 1);
         NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, 14, 1);
         NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, 14, 2);
         NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, 9, 3);
         /* set the MGR coarse solver. Comment out to use default CG solver in MGR */
         NALU_HYPRE_MGRSetCoarseSolver( pcg_precond, NALU_HYPRE_BoomerAMGSolve, NALU_HYPRE_BoomerAMGSetup, amg_solver);


         /* setup MGR-COGMRES solver */
         NALU_HYPRE_COGMRESSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_COGMRESSetPrecond(pcg_solver,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_MGRSolve,
                                 (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_MGRSetup,
                                 pcg_precond);
      }
      NALU_HYPRE_COGMRESSetup(pcg_solver, (NALU_HYPRE_Matrix) parcsr_M,
                         (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("COGMRES Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_COGMRESSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                         (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      if (second_time)
      {
         /* run a second time [for timings, to check for memory leaks] */
         NALU_HYPRE_ParVectorSetRandomValues(x, 775);
#if defined(NALU_HYPRE_USING_CURAND) || defined(NALU_HYPRE_USING_ROCRAND)
         nalu_hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
         nalu_hypre_ParVectorCopy(x0_save, x);

         NALU_HYPRE_COGMRESSetup(pcg_solver, (NALU_HYPRE_Matrix) parcsr_M,
                            (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x);
         NALU_HYPRE_COGMRESSolve(pcg_solver, (NALU_HYPRE_Matrix) parcsr_A,
                            (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x);
      }

      NALU_HYPRE_COGMRESGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_COGMRESGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

      NALU_HYPRE_ParCSRCOGMRESDestroy(pcg_solver);

      if (solver_id == 16)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      else if (solver_id == 46)
      {
         NALU_HYPRE_EuclidDestroy(pcg_precond);
      }
      else if (solver_id == 74)
      {
         /* free memory */
         if (mgr_num_cindexes)
         {
            nalu_hypre_TFree(mgr_num_cindexes, NALU_HYPRE_MEMORY_HOST);
         }
         mgr_num_cindexes = NULL;

         if (mgr_reserved_coarse_indexes)
         {
            nalu_hypre_TFree(mgr_reserved_coarse_indexes, NALU_HYPRE_MEMORY_HOST);
         }
         mgr_reserved_coarse_indexes = NULL;

         if (mgr_cindexes)
         {
            for ( i = 0; i < mgr_nlevels; i++)
            {
               if (mgr_cindexes[i])
               {
                  nalu_hypre_TFree(mgr_cindexes[i], NALU_HYPRE_MEMORY_HOST);
               }
            }
            nalu_hypre_TFree(mgr_cindexes, NALU_HYPRE_MEMORY_HOST);
            mgr_cindexes = NULL;
         }

         NALU_HYPRE_BoomerAMGDestroy(amg_solver);
         NALU_HYPRE_MGRDestroy(pcg_precond);
      }
      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("COGMRES Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final COGMRES Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }
   }
   /*-----------------------------------------------------------
    * Solve the system using CGNR
    *-----------------------------------------------------------*/

   if (solver_id == 5 || solver_id == 6)
   {
      time_index = nalu_hypre_InitializeTiming("CGNR Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_ParCSRCGNRCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_solver);
      NALU_HYPRE_CGNRSetMaxIter(pcg_solver, max_iter);
      NALU_HYPRE_CGNRSetTol(pcg_solver, tol);
      NALU_HYPRE_CGNRSetLogging(pcg_solver, ioutdat);

      if (solver_id == 5)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: AMG-CGNR\n"); }
         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         NALU_HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
         NALU_HYPRE_BoomerAMGSetCoarsenCutFactor(pcg_precond, coarsen_cut_factor);
         NALU_HYPRE_BoomerAMGSetCPoints(pcg_precond, max_levels, num_cpt, cpt_index);
         NALU_HYPRE_BoomerAMGSetFPoints(pcg_precond, num_fpt, fpt_index);
         NALU_HYPRE_BoomerAMGSetIsolatedFPoints(pcg_precond, num_isolated_fpt, isolated_fpt_index);
         NALU_HYPRE_BoomerAMGSetMeasureType(pcg_precond, measure_type);
         NALU_HYPRE_BoomerAMGSetStrongThreshold(pcg_precond, strong_threshold);
         NALU_HYPRE_BoomerAMGSetSeqThreshold(pcg_precond, seq_threshold);
         NALU_HYPRE_BoomerAMGSetRedundant(pcg_precond, redundant);
         NALU_HYPRE_BoomerAMGSetMaxCoarseSize(pcg_precond, coarse_threshold);
         NALU_HYPRE_BoomerAMGSetMinCoarseSize(pcg_precond, min_coarse_size);
         NALU_HYPRE_BoomerAMGSetTruncFactor(pcg_precond, trunc_factor);
         NALU_HYPRE_BoomerAMGSetPMaxElmts(pcg_precond, P_max_elmts);
         NALU_HYPRE_BoomerAMGSetJacobiTruncThreshold(pcg_precond, jacobi_trunc_threshold);
         NALU_HYPRE_BoomerAMGSetSCommPkgSwitch(pcg_precond, S_commpkg_switch);
         NALU_HYPRE_BoomerAMGSetPrintLevel(pcg_precond, poutdat);
         NALU_HYPRE_BoomerAMGSetPrintFileName(pcg_precond, "driver.out.log");
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, precon_cycles);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetFCycle(pcg_precond, fcycle);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         if (relax_type > -1) { NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type); }
         if (relax_down > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_down, 1);
         }
         if (relax_up > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(pcg_precond, relax_coarse, 3);
         }
         NALU_HYPRE_BoomerAMGSetAddRelaxType(pcg_precond, add_relax_type);
         NALU_HYPRE_BoomerAMGSetAddRelaxWt(pcg_precond, add_relax_wt);
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
         NALU_HYPRE_BoomerAMGSetChebyEigEst(pcg_precond, cheby_eig_est);
         NALU_HYPRE_BoomerAMGSetChebyVariant(pcg_precond, cheby_variant);
         NALU_HYPRE_BoomerAMGSetChebyScale(pcg_precond, cheby_scale);
         NALU_HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         NALU_HYPRE_BoomerAMGSetRelaxWt(pcg_precond, relax_wt);
         NALU_HYPRE_BoomerAMGSetOuterWt(pcg_precond, outer_wt);
         if (level_w > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelRelaxWt(pcg_precond, relax_wt_level, level_w);
         }
         if (level_ow > -1)
         {
            NALU_HYPRE_BoomerAMGSetLevelOuterWt(pcg_precond, outer_wt_level, level_ow);
         }
         NALU_HYPRE_BoomerAMGSetSmoothType(pcg_precond, smooth_type);
         NALU_HYPRE_BoomerAMGSetSmoothNumLevels(pcg_precond, smooth_num_levels);
         NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(pcg_precond, smooth_num_sweeps);
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
         NALU_HYPRE_BoomerAMGSetDebugFlag(pcg_precond, debug_flag);
         NALU_HYPRE_BoomerAMGSetNumFunctions(pcg_precond, num_functions);
         NALU_HYPRE_BoomerAMGSetAggNumLevels(pcg_precond, agg_num_levels);
         NALU_HYPRE_BoomerAMGSetAggInterpType(pcg_precond, agg_interp_type);
         NALU_HYPRE_BoomerAMGSetAggTruncFactor(pcg_precond, agg_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggP12TruncFactor(pcg_precond, agg_P12_trunc_factor);
         NALU_HYPRE_BoomerAMGSetAggPMaxElmts(pcg_precond, agg_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetAggP12MaxElmts(pcg_precond, agg_P12_max_elmts);
         NALU_HYPRE_BoomerAMGSetNumPaths(pcg_precond, num_paths);
         NALU_HYPRE_BoomerAMGSetNodal(pcg_precond, nodal);
         NALU_HYPRE_BoomerAMGSetNodalDiag(pcg_precond, nodal_diag);
         NALU_HYPRE_BoomerAMGSetVariant(pcg_precond, variant);
         NALU_HYPRE_BoomerAMGSetOverlap(pcg_precond, overlap);
         NALU_HYPRE_BoomerAMGSetDomainType(pcg_precond, domain_type);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         NALU_HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         NALU_HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         NALU_HYPRE_BoomerAMGSetAddLastLvl(pcg_precond, add_last_lvl);
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetModuleRAP2(pcg_precond, mod_rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
#ifdef NALU_HYPRE_USING_DSUPERLU
         NALU_HYPRE_BoomerAMGSetDSLUThreshold(pcg_precond, dslu_threshold);
#endif
         if (nongalerk_tol)
         {
            NALU_HYPRE_BoomerAMGSetNonGalerkinTol(pcg_precond, nongalerk_tol[nongalerk_num_tol - 1]);
            for (i = 0; i < nongalerk_num_tol - 1; i++)
            {
               NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol(pcg_precond, nongalerk_tol[i], i);
            }
         }
         NALU_HYPRE_CGNRSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_CGNRSetPrecond(pcg_solver,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolveT,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                              pcg_precond);
      }
      else if (solver_id == 6)
      {
         /* use diagonal scaling as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: DS-CGNR\n"); }
         pcg_precond = NULL;

         NALU_HYPRE_CGNRSetPrecond(pcg_solver,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScale,
                              (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_ParCSRDiagScaleSetup,
                              pcg_precond);
      }

      NALU_HYPRE_CGNRGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
         nalu_hypre_printf("NALU_HYPRE_ParCSRCGNRGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         nalu_hypre_printf("NALU_HYPRE_ParCSRCGNRGetPrecond got good precond\n");
      }
      NALU_HYPRE_CGNRSetup(pcg_solver, (NALU_HYPRE_Matrix) parcsr_M,
                      (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("CGNR Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_CGNRSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b,
                      (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      if (second_time)
      {
         /* run a second time [for timings, to check for memory leaks] */
         NALU_HYPRE_ParVectorSetRandomValues(x, 775);
#if defined(NALU_HYPRE_USING_CURAND) || defined(NALU_HYPRE_USING_ROCRAND)
         nalu_hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
         nalu_hypre_ParVectorCopy(x0_save, x);

         NALU_HYPRE_CGNRSetup(pcg_solver, (NALU_HYPRE_Matrix) parcsr_M,
                         (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x);
         NALU_HYPRE_CGNRSolve(pcg_solver, (NALU_HYPRE_Matrix) parcsr_A,
                         (NALU_HYPRE_Vector) b, (NALU_HYPRE_Vector) x);
      }

      NALU_HYPRE_CGNRGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_CGNRGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
      NALU_HYPRE_ParCSRCGNRDestroy(pcg_solver);

      if (solver_id == 5)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      if (myid == 0 /* begin lobpcg */ && !lobpcgFlag /* end lobpcg */)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Solve the system using MGR
    *-----------------------------------------------------------*/

   if (solver_id == 70)
   {
      if (myid == 0) { nalu_hypre_printf("Solver:  MGR\n"); }
      time_index = nalu_hypre_InitializeTiming("MGR Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_Solver mgr_solver;
      NALU_HYPRE_MGRCreate(&mgr_solver);

      mgr_num_cindexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  mgr_nlevels, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < mgr_nlevels; i++)
      {
         /* assume 1 coarse index per level */
         mgr_num_cindexes[i] = 1;
      }
      mgr_cindexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int*,  mgr_nlevels, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < mgr_nlevels; i++)
      {
         mgr_cindexes[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  mgr_num_cindexes[i], NALU_HYPRE_MEMORY_HOST);
      }
      for (i = 0; i < mgr_nlevels; i++)
      {
         /* assume coarse point is at index 0 */
         mgr_cindexes[i][0] = 0;
      }
      mgr_reserved_coarse_indexes = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  mgr_num_reserved_nodes,
                                                  NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < mgr_num_reserved_nodes; i++)
      {
         /* generate artificial reserved nodes */
         mgr_reserved_coarse_indexes[i] = last_local_row - (NALU_HYPRE_BigInt) i; //2*i+1;
      }

      /* set MGR data by block */
      NALU_HYPRE_MGRSetCpointsByBlock( mgr_solver, mgr_bsize, mgr_nlevels, mgr_num_cindexes, mgr_cindexes);
      /* set reserved coarse nodes */
      if (mgr_num_reserved_nodes) { NALU_HYPRE_MGRSetReservedCoarseNodes(mgr_solver, mgr_num_reserved_nodes, mgr_reserved_coarse_indexes); }

      /* set intermediate coarse grid strategy */
      NALU_HYPRE_MGRSetNonCpointsToFpoints(mgr_solver, mgr_non_c_to_f);
      /* set F relaxation strategy */
      NALU_HYPRE_MGRSetFRelaxMethod(mgr_solver, mgr_frelax_method);
      /* set relax type for single level F-relaxation and post-relaxation */
      NALU_HYPRE_MGRSetRelaxType(mgr_solver, mgr_relax_type);
      NALU_HYPRE_MGRSetNumRelaxSweeps(mgr_solver, mgr_num_relax_sweeps);
      /* set interpolation type */
      NALU_HYPRE_MGRSetRestrictType(mgr_solver, mgr_restrict_type);
      NALU_HYPRE_MGRSetNumRestrictSweeps(mgr_solver, mgr_num_restrict_sweeps);
      NALU_HYPRE_MGRSetInterpType(mgr_solver, mgr_interp_type);
      NALU_HYPRE_MGRSetNumInterpSweeps(mgr_solver, mgr_num_interp_sweeps);
      /* set print level */
      NALU_HYPRE_MGRSetPrintLevel(mgr_solver, 3);
      /* set max iterations */
      NALU_HYPRE_MGRSetMaxIter(mgr_solver, max_iter);
      NALU_HYPRE_MGRSetTol(mgr_solver, tol);
      /* set global smoother */
      NALU_HYPRE_MGRSetGlobalSmoothType(mgr_solver, mgr_gsmooth_type);
      NALU_HYPRE_MGRSetMaxGlobalSmoothIters( mgr_solver, mgr_num_gsmooth_sweeps );

      /* create AMG coarse grid solver */

      NALU_HYPRE_BoomerAMGCreate(&amg_solver);
      if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
      {
         NALU_HYPRE_BoomerAMGSetInterpType(amg_solver, 18);
         NALU_HYPRE_BoomerAMGSetCoarsenType(amg_solver, 8);
         NALU_HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);
      }
      else
      {
         NALU_HYPRE_BoomerAMGSetInterpType(amg_solver, 0);
         NALU_HYPRE_BoomerAMGSetPostInterpType(amg_solver, post_interp_type);
         NALU_HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
         NALU_HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
         NALU_HYPRE_BoomerAMGSetFCycle(amg_solver, fcycle);
         NALU_HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);
         if (relax_down > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_down, 1);
         }
         if (relax_up > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_up, 2);
         }
         if (relax_coarse > -1)
         {
            NALU_HYPRE_BoomerAMGSetCycleRelaxType(amg_solver, relax_coarse, 3);
         }
         NALU_HYPRE_BoomerAMGSetSmoothType(amg_solver, smooth_type);
         NALU_HYPRE_BoomerAMGSetSmoothNumSweeps(amg_solver, smooth_num_sweeps);
      }
      NALU_HYPRE_BoomerAMGSetCGCIts(amg_solver, cgcits);
      NALU_HYPRE_BoomerAMGSetTol(amg_solver, tol);
      NALU_HYPRE_BoomerAMGSetPMaxElmts(amg_solver, 0);
      NALU_HYPRE_BoomerAMGSetNumSweeps(amg_solver, num_sweeps);
      NALU_HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);
      NALU_HYPRE_BoomerAMGSetMaxLevels(amg_solver, max_levels);
      if (mgr_nlevels < 1 || mgr_bsize < 2)
      {
         NALU_HYPRE_BoomerAMGSetMaxIter(amg_solver, max_iter);
         NALU_HYPRE_BoomerAMGSetPrintLevel(amg_solver, 3);
      }
      else
      {
         NALU_HYPRE_BoomerAMGSetMaxIter(amg_solver, precon_cycles);
         NALU_HYPRE_BoomerAMGSetTol(amg_solver, 0.0);
         NALU_HYPRE_BoomerAMGSetPrintLevel(amg_solver, 1);
      }
      /* set the MGR coarse solver. Comment out to use default CG solver in MGR */
      NALU_HYPRE_MGRSetCoarseSolver( mgr_solver, NALU_HYPRE_BoomerAMGSolve, NALU_HYPRE_BoomerAMGSetup, amg_solver);

      /* setup MGR solver */
      NALU_HYPRE_MGRSetup(mgr_solver, parcsr_M, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("MGR Solve");
      nalu_hypre_BeginTiming(time_index);

      /* MGR solve */
      NALU_HYPRE_MGRSolve(mgr_solver, parcsr_A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      if (second_time)
      {
         /* run a second time [for timings, to check for memory leaks] */
         NALU_HYPRE_ParVectorSetRandomValues(x, 775);
#if defined(NALU_HYPRE_USING_CURAND) || defined(NALU_HYPRE_USING_ROCRAND)
         nalu_hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
         nalu_hypre_ParVectorCopy(x0_save, x);

         NALU_HYPRE_MGRSetup(mgr_solver, parcsr_M, b, x);
         NALU_HYPRE_MGRSolve(mgr_solver, parcsr_A, b, x);
      }

      NALU_HYPRE_MGRGetNumIterations(mgr_solver, &num_iterations);
      NALU_HYPRE_MGRGetFinalRelativeResidualNorm(mgr_solver, &final_res_norm);

      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("MGR Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }

      /* free memory */
      if (mgr_num_cindexes)
      {
         nalu_hypre_TFree(mgr_num_cindexes, NALU_HYPRE_MEMORY_HOST);
      }
      mgr_num_cindexes = NULL;

      if (mgr_reserved_coarse_indexes)
      {
         nalu_hypre_TFree(mgr_reserved_coarse_indexes, NALU_HYPRE_MEMORY_HOST);
      }
      mgr_reserved_coarse_indexes = NULL;

      if (mgr_cindexes)
      {
         for ( i = 0; i < mgr_nlevels; i++)
         {
            if (mgr_cindexes[i])
            {
               nalu_hypre_TFree(mgr_cindexes[i], NALU_HYPRE_MEMORY_HOST);
            }
         }
         nalu_hypre_TFree(mgr_cindexes, NALU_HYPRE_MEMORY_HOST);
         mgr_cindexes = NULL;
      }

      NALU_HYPRE_BoomerAMGDestroy(amg_solver);
      NALU_HYPRE_MGRDestroy(mgr_solver);
   }

   /*-----------------------------------------------------------
    * Solve the system using nalu_hypre_ILU
    *-----------------------------------------------------------*/

   if (solver_id == 80)
   {
      if (myid == 0) { nalu_hypre_printf("Solver:  nalu_hypre_ILU\n"); }
      time_index = nalu_hypre_InitializeTiming("nalu_hypre_ILU Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_Solver ilu_solver;
      NALU_HYPRE_ILUCreate(&ilu_solver);

      /* set ilu type */
      NALU_HYPRE_ILUSetType(ilu_solver, ilu_type);
      /* set level of fill */
      NALU_HYPRE_ILUSetLevelOfFill(ilu_solver, ilu_lfil);
      /* set print level */
      NALU_HYPRE_ILUSetPrintLevel(ilu_solver, 2);
      /* set max iterations */
      NALU_HYPRE_ILUSetMaxIter(ilu_solver, max_iter);
      /* set max number of nonzeros per row */
      NALU_HYPRE_ILUSetMaxNnzPerRow(ilu_solver, ilu_max_row_nnz);
      /* set the droptol */
      NALU_HYPRE_ILUSetDropThreshold(ilu_solver, ilu_droptol);
      NALU_HYPRE_ILUSetTol(ilu_solver, tol);
      /* set max iterations for Schur system solve */
      NALU_HYPRE_ILUSetSchurMaxIter( ilu_solver, ilu_schur_max_iter );

      /* setting for NSH */
      if (ilu_type == 20 || ilu_type == 21)
      {
         NALU_HYPRE_ILUSetNSHDropThreshold( ilu_solver, ilu_nsh_droptol);
      }


      /* setup nalu_hypre_ILU solver */
      NALU_HYPRE_ILUSetup(ilu_solver, parcsr_M, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("nalu_hypre_ILU Solve");
      nalu_hypre_BeginTiming(time_index);

      /* nalu_hypre_ILU solve */
      NALU_HYPRE_ILUSolve(ilu_solver, parcsr_A, b, x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      if (second_time)
      {
         /* run a second time [for timings, to check for memory leaks] */
         NALU_HYPRE_ParVectorSetRandomValues(x, 775);
#if defined(NALU_HYPRE_USING_CURAND) || defined(NALU_HYPRE_USING_ROCRAND)
         nalu_hypre_ResetDeviceRandGenerator(1234ULL, 0ULL);
#endif
         nalu_hypre_ParVectorCopy(x0_save, x);

         NALU_HYPRE_ILUSetup(ilu_solver, parcsr_M, b, x);
         NALU_HYPRE_ILUSolve(ilu_solver, parcsr_A, b, x);
      }

      NALU_HYPRE_ILUGetNumIterations(ilu_solver, &num_iterations);
      NALU_HYPRE_ILUGetFinalRelativeResidualNorm(ilu_solver, &final_res_norm);

      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("nalu_hypre_ILU Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }

      /* free memory */
      NALU_HYPRE_ILUDestroy(ilu_solver);
   }

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   if (print_system)
   {
      NALU_HYPRE_IJVectorPrint(ij_x, "IJ.out.x");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

final:

   NALU_HYPRE_ParVectorDestroy(x0_save);

   if (test_ij || build_matrix_type == -1)
   {
      if (ij_A) { NALU_HYPRE_IJMatrixDestroy(ij_A); }
   }
   else
   {
      NALU_HYPRE_ParCSRMatrixDestroy(parcsr_A);
   }

   if (build_matrix_M == 1)
   {
      NALU_HYPRE_IJMatrixDestroy(ij_M);
   }

   /* for build_rhs_type = 1, 6 or 7, we did not create ij_b  - just b*/
   if (build_rhs_type == 1 || build_rhs_type == 6 || build_rhs_type == 7)
   {
      NALU_HYPRE_ParVectorDestroy(b);
   }
   else
   {
      if (ij_b) { NALU_HYPRE_IJVectorDestroy(ij_b); }
   }

   if (ij_x) { NALU_HYPRE_IJVectorDestroy(ij_x); }

   if (build_rbm)
   {
      if (ij_rbm)
      {
         for (i = 0; i < num_interp_vecs; i++)
         {
            if (ij_rbm[i]) { NALU_HYPRE_IJVectorDestroy(ij_rbm[i]); }
         }
      }
      nalu_hypre_TFree(ij_rbm, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(interp_vecs, NALU_HYPRE_MEMORY_HOST);
   }
   if (nongalerk_tol)
   {
      nalu_hypre_TFree(nongalerk_tol, NALU_HYPRE_MEMORY_HOST);
   }

   if (cpt_index)
   {
      nalu_hypre_TFree(cpt_index, NALU_HYPRE_MEMORY_HOST);
   }
   if (fpt_index)
   {
      nalu_hypre_TFree(fpt_index, NALU_HYPRE_MEMORY_HOST);
   }
   if (isolated_fpt_index)
   {
      nalu_hypre_TFree(isolated_fpt_index, NALU_HYPRE_MEMORY_HOST);
   }

   /*
      nalu_hypre_FinalizeMemoryDebug();
   */

   /* Finalize Hypre */
   NALU_HYPRE_Finalize();

   /* Finalize MPI */
   nalu_hypre_MPI_Finalize();

#if defined(NALU_HYPRE_USING_MEMORY_TRACKER)
   if (memory_location == NALU_HYPRE_MEMORY_HOST)
   {
      if (nalu_hypre_total_bytes[nalu_hypre_MEMORY_DEVICE] || nalu_hypre_total_bytes[nalu_hypre_MEMORY_UNIFIED])
      {
         nalu_hypre_printf("Error: nonzero GPU memory allocated with the HOST mode\n");
         nalu_hypre_assert(0);
      }
   }
#endif

   /* when using cuda-memcheck --leak-check full, uncomment this */
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_ResetCudaDevice(NULL);
#endif

   return (0);
}

/*----------------------------------------------------------------------
 * Build matrix from file. Expects three files on each processor.
 * filename.D.n contains the diagonal part, filename.O.n contains
 * the offdiagonal part and filename.INFO.n contains global row
 * and column numbers, number of columns of offdiagonal matrix
 * and the mapping of offdiagonal column numbers to global column numbers.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParFromFile( NALU_HYPRE_Int                  argc,
                  char                *argv[],
                  NALU_HYPRE_Int                  arg_index,
                  NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   NALU_HYPRE_ParCSRMatrix A;

   NALU_HYPRE_Int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      nalu_hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  FromFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   NALU_HYPRE_ParCSRMatrixRead(nalu_hypre_MPI_COMM_WORLD, filename, &A);

   *A_ptr = A;

   return (0);
}


/*----------------------------------------------------------------------
 * Build rhs from file. Expects two files on each processor.
 * filename.n contains the data and
 * and filename.INFO.n contains global row
 * numbers
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
ReadParVectorFromFile( NALU_HYPRE_Int            argc,
                       char                *argv[],
                       NALU_HYPRE_Int            arg_index,
                       NALU_HYPRE_ParVector      *b_ptr     )
{
   char               *filename;

   NALU_HYPRE_ParVector b;

   NALU_HYPRE_Int                 myid;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      nalu_hypre_printf("  Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  From ParFile: %s\n", filename);
   }

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   NALU_HYPRE_ParVectorRead(nalu_hypre_MPI_COMM_WORLD, filename, &b);

   *b_ptr = b;

   return (0);
}




/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParLaplacian( NALU_HYPRE_Int                  argc,
                   char                *argv[],
                   NALU_HYPRE_Int                  arg_index,
                   NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_BigInt              nx, ny, nz;
   NALU_HYPRE_Int                 P, Q, R;
   NALU_HYPRE_Real          cx, cy, cz;

   NALU_HYPRE_ParCSRMatrix  A;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 p, q, r;
   NALU_HYPRE_Int                 num_fun = 1;
   NALU_HYPRE_Real         *values;
   NALU_HYPRE_Real         *mtrx;

   NALU_HYPRE_Real          ep = .1;

   NALU_HYPRE_Int                 system_vcoef = 0;
   NALU_HYPRE_Int                 sys_opt = 0;
   NALU_HYPRE_Int                 vcoef_opt = 0;


   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sysL") == 0 )
      {
         arg_index++;
         num_fun = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sysL_opt") == 0 )
      {
         arg_index++;
         sys_opt = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sys_vcoef") == 0 )
      {
         /* have to use -sysL for this to */
         arg_index++;
         system_vcoef = 1;
      }
      else if ( strcmp(argv[arg_index], "-sys_vcoef_opt") == 0 )
      {
         arg_index++;
         vcoef_opt = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ep") == 0 )
      {
         arg_index++;
         ep = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) != num_procs)
   {
      nalu_hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  Laplacian:   num_fun = %d\n", num_fun);
      nalu_hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      nalu_hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      nalu_hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  4, NALU_HYPRE_MEMORY_HOST);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0 * cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0 * cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0 * cz;
   }

   if (num_fun == 1)
      A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian(nalu_hypre_MPI_COMM_WORLD,
                                                 nx, ny, nz, P, Q, R, p, q, r, values);
   else
   {
      mtrx = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_fun * num_fun, NALU_HYPRE_MEMORY_HOST);

      if (num_fun == 2)
      {
         if (sys_opt == 1) /* identity  */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 2)
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 20.0;
         }
         else if (sys_opt == 3) /* similar to barry's talk - ex1 */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 2.0;
            mtrx[2] = 2.0;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 4) /* can use with vcoef to get barry's ex*/
         {
            mtrx[0] = 1.0;
            mtrx[1] = 1.0;
            mtrx[2] = 1.0;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 5) /* barry's talk - ex1 */
         {
            mtrx[0] = 1.0;
            mtrx[1] = 1.1;
            mtrx[2] = 1.1;
            mtrx[3] = 1.0;
         }
         else if (sys_opt == 6) /*  */
         {
            mtrx[0] = 1.1;
            mtrx[1] = 1.0;
            mtrx[2] = 1.0;
            mtrx[3] = 1.1;
         }

         else /* == 0 */
         {
            mtrx[0] = 2;
            mtrx[1] = 1;
            mtrx[2] = 1;
            mtrx[3] = 2;
         }
      }
      else if (num_fun == 3)
      {
         if (sys_opt == 1)
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 0.0;
            mtrx[4] = 1.0;
            mtrx[5] = 0.0;
            mtrx[6] = 0.0;
            mtrx[7] = 0.0;
            mtrx[8] = 1.0;
         }
         else if (sys_opt == 2)
         {
            mtrx[0] = 1.0;
            mtrx[1] = 0.0;
            mtrx[2] = 0.0;
            mtrx[3] = 0.0;
            mtrx[4] = 20.0;
            mtrx[5] = 0.0;
            mtrx[6] = 0.0;
            mtrx[7] = 0.0;
            mtrx[8] = .01;
         }
         else if (sys_opt == 3)
         {
            mtrx[0] = 1.01;
            mtrx[1] = 1;
            mtrx[2] = 0.0;
            mtrx[3] = 1;
            mtrx[4] = 2;
            mtrx[5] = 1;
            mtrx[6] = 0.0;
            mtrx[7] = 1;
            mtrx[8] = 1.01;
         }
         else if (sys_opt == 4) /* barry ex4 */
         {
            mtrx[0] = 3;
            mtrx[1] = 1;
            mtrx[2] = 0.0;
            mtrx[3] = 1;
            mtrx[4] = 4;
            mtrx[5] = 2;
            mtrx[6] = 0.0;
            mtrx[7] = 2;
            mtrx[8] = .25;
         }
         else /* == 0 */
         {
            mtrx[0] = 2.0;
            mtrx[1] = 1.0;
            mtrx[2] = 0.0;
            mtrx[3] = 1.0;
            mtrx[4] = 2.0;
            mtrx[5] = 1.0;
            mtrx[6] = 0.0;
            mtrx[7] = 1.0;
            mtrx[8] = 2.0;
         }

      }
      else if (num_fun == 4)
      {
         mtrx[0] = 1.01;
         mtrx[1] = 1;
         mtrx[2] = 0.0;
         mtrx[3] = 0.0;
         mtrx[4] = 1;
         mtrx[5] = 2;
         mtrx[6] = 1;
         mtrx[7] = 0.0;
         mtrx[8] = 0.0;
         mtrx[9] = 1;
         mtrx[10] = 1.01;
         mtrx[11] = 0.0;
         mtrx[12] = 2;
         mtrx[13] = 1;
         mtrx[14] = 0.0;
         mtrx[15] = 1;
      }

      if (!system_vcoef)
      {
         A = (NALU_HYPRE_ParCSRMatrix) GenerateSysLaplacian(nalu_hypre_MPI_COMM_WORLD,
                                                       nx, ny, nz, P, Q,
                                                       R, p, q, r, num_fun, mtrx, values);
      }
      else
      {
         NALU_HYPRE_Real *mtrx_values;

         mtrx_values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_fun * num_fun * 4, NALU_HYPRE_MEMORY_HOST);

         if (num_fun == 2)
         {
            if (vcoef_opt == 1)
            {
               /* Barry's talk * - must also have sys_opt = 4, all fail */
               mtrx[0] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, .10, 1.0, 0, mtrx_values);

               mtrx[1]  = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, .1, 1.0, 1.0, 1, mtrx_values);

               mtrx[2] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, .01, 1.0, 1.0, 2, mtrx_values);

               mtrx[3] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 2.0, .02, 1.0, 3, mtrx_values);
            }
            else if (vcoef_opt == 2)
            {
               /* Barry's talk * - ex2 - if have sys-opt = 4*/
               mtrx[0] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, .010, 1.0, 0, mtrx_values);

               mtrx[1]  = 200.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 1, mtrx_values);

               mtrx[2] = 200.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 2, mtrx_values);

               mtrx[3] = 1.0;
               SetSysVcoefValues(num_fun, nx, ny, nz, 2.0, .02, 1.0, 3, mtrx_values);
            }
            else if (vcoef_opt == 3) /* use with default sys_opt  - ulrike ex 3*/
            {

               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 3, mtrx_values);
            }
            else if (vcoef_opt == 4) /* use with default sys_opt  - ulrike ex 4*/
            {
               NALU_HYPRE_Real ep2 = ep;

               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, ep * 1.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, ep * 1.0, 1.0, 1.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, ep2 * 1.0, 1.0, 3, mtrx_values);
            }
            else if (vcoef_opt == 5) /* use with default sys_opt  - */
            {
               NALU_HYPRE_Real  alp, beta;
               alp = .001;
               beta = 10;

               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, alp * 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, beta * 1.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, alp * 1.0, 1.0, 1.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, beta * 1.0, 1.0, 3, mtrx_values);
            }
            else  /* = 0 */
            {
               /* mtrx[0] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 1.0, 1.0, 0, mtrx_values);

               /* mtrx[1] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 2.0, 1.0, 1, mtrx_values);

               /* mtrx[2] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 2.0, 1.0, 0.0, 2, mtrx_values);

               /* mtrx[3] */
               SetSysVcoefValues(num_fun, nx, ny, nz, 1.0, 3.0, 1.0, 3, mtrx_values);
            }
         }
         else if (num_fun == 3)
         {
            mtrx[0] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, .01, 1, 0, mtrx_values);

            mtrx[1] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 1, mtrx_values);

            mtrx[2] = 0.0;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 2, mtrx_values);

            mtrx[3] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 3, mtrx_values);

            mtrx[4] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 2, .02, 1, 4, mtrx_values);

            mtrx[5] = 2;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 5, mtrx_values);

            mtrx[6] = 0.0;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 6, mtrx_values);

            mtrx[7] = 2;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1, 1, 1, 7, mtrx_values);

            mtrx[8] = 1;
            SetSysVcoefValues(num_fun, nx, ny, nz, 1.5, .04, 1, 8, mtrx_values);
         }

         A = (NALU_HYPRE_ParCSRMatrix) GenerateSysLaplacianVCoef(nalu_hypre_MPI_COMM_WORLD,
                                                            nx, ny, nz, P, Q,
                                                            R, p, q, r, num_fun, mtrx, mtrx_values);

         nalu_hypre_TFree(mtrx_values, NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_TFree(mtrx, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * returns the sign of a real number
 *  1 : positive
 *  0 : zero
 * -1 : negative
 *----------------------------------------------------------------------*/
static inline NALU_HYPRE_Int sign_double(NALU_HYPRE_Real a)
{
   return ( (0.0 < a) - (0.0 > a) );
}

/*----------------------------------------------------------------------
 * Build standard 7-point convection-diffusion operator
 * Parameters given in command line.
 * Operator:
 *
 *  -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f
 *
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParDifConv( NALU_HYPRE_Int                  argc,
                 char                *argv[],
                 NALU_HYPRE_Int                  arg_index,
                 NALU_HYPRE_ParCSRMatrix  *A_ptr)
{
   NALU_HYPRE_BigInt        nx, ny, nz;
   NALU_HYPRE_Int           P, Q, R;
   NALU_HYPRE_Real          cx, cy, cz;
   NALU_HYPRE_Real          ax, ay, az, atype;
   NALU_HYPRE_Real          hinx, hiny, hinz;
   NALU_HYPRE_Int           sign_prod;

   NALU_HYPRE_ParCSRMatrix  A;

   NALU_HYPRE_Int           num_procs, myid;
   NALU_HYPRE_Int           p, q, r;
   NALU_HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   cx = 1.;
   cy = 1.;
   cz = 1.;

   ax = 1.;
   ay = 1.;
   az = 1.;

   atype = 0;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-c") == 0 )
      {
         arg_index++;
         cx = atof(argv[arg_index++]);
         cy = atof(argv[arg_index++]);
         cz = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-a") == 0 )
      {
         arg_index++;
         ax = atof(argv[arg_index++]);
         ay = atof(argv[arg_index++]);
         az = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-atype") == 0 )
      {
         arg_index++;
         atype = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) != num_procs)
   {
      nalu_hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  Convection-Diffusion: \n");
      nalu_hypre_printf("    -cx Dxx - cy Dyy - cz Dzz + ax Dx + ay Dy + az Dz = f\n");
      nalu_hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      nalu_hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      nalu_hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n", cx, cy, cz);
      nalu_hypre_printf("    (ax, ay, az) = (%f, %f, %f)\n\n", ax, ay, az);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   hinx = 1. / (NALU_HYPRE_Real)(nx + 1);
   hiny = 1. / (NALU_HYPRE_Real)(ny + 1);
   hinz = 1. / (NALU_HYPRE_Real)(nz + 1);

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/
   /* values[7]:
    *    [0]: center
    *    [1]: X-
    *    [2]: Y-
    *    [3]: Z-
    *    [4]: X+
    *    [5]: Y+
    *    [6]: Z+
    */
   values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  7, NALU_HYPRE_MEMORY_HOST);

   values[0] = 0.;

   if (0 == atype) /* forward scheme for conv */
   {
      values[1] = -cx / (hinx * hinx);
      values[2] = -cy / (hiny * hiny);
      values[3] = -cz / (hinz * hinz);
      values[4] = -cx / (hinx * hinx) + ax / hinx;
      values[5] = -cy / (hiny * hiny) + ay / hiny;
      values[6] = -cz / (hinz * hinz) + az / hinz;

      if (nx > 1)
      {
         values[0] += 2.0 * cx / (hinx * hinx) - 1.*ax / hinx;
      }
      if (ny > 1)
      {
         values[0] += 2.0 * cy / (hiny * hiny) - 1.*ay / hiny;
      }
      if (nz > 1)
      {
         values[0] += 2.0 * cz / (hinz * hinz) - 1.*az / hinz;
      }
   }
   else if (1 == atype) /* backward scheme for conv */
   {
      values[1] = -cx / (hinx * hinx) - ax / hinx;
      values[2] = -cy / (hiny * hiny) - ay / hiny;
      values[3] = -cz / (hinz * hinz) - az / hinz;
      values[4] = -cx / (hinx * hinx);
      values[5] = -cy / (hiny * hiny);
      values[6] = -cz / (hinz * hinz);

      if (nx > 1)
      {
         values[0] += 2.0 * cx / (hinx * hinx) + 1.*ax / hinx;
      }
      if (ny > 1)
      {
         values[0] += 2.0 * cy / (hiny * hiny) + 1.*ay / hiny;
      }
      if (nz > 1)
      {
         values[0] += 2.0 * cz / (hinz * hinz) + 1.*az / hinz;
      }
   }
   else if (3 == atype) /* upwind scheme */
   {
      sign_prod = sign_double(cx) * sign_double(ax);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[1] = -cx / (hinx * hinx) - ax / hinx;
         values[4] = -cx / (hinx * hinx);
         if (nx > 1)
         {
            values[0] += 2.0 * cx / (hinx * hinx) + 1.*ax / hinx;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[1] = -cx / (hinx * hinx);
         values[4] = -cx / (hinx * hinx) + ax / hinx;
         if (nx > 1)
         {
            values[0] += 2.0 * cx / (hinx * hinx) - 1.*ax / hinx;
         }
      }

      sign_prod = sign_double(cy) * sign_double(ay);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[2] = -cy / (hiny * hiny) - ay / hiny;
         values[5] = -cy / (hiny * hiny);
         if (ny > 1)
         {
            values[0] += 2.0 * cy / (hiny * hiny) + 1.*ay / hiny;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[2] = -cy / (hiny * hiny);
         values[5] = -cy / (hiny * hiny) + ay / hiny;
         if (ny > 1)
         {
            values[0] += 2.0 * cy / (hiny * hiny) - 1.*ay / hiny;
         }
      }

      sign_prod = sign_double(cz) * sign_double(az);
      if (sign_prod == 1) /* same sign use back scheme */
      {
         values[3] = -cz / (hinz * hinz) - az / hinz;
         values[6] = -cz / (hinz * hinz);
         if (nz > 1)
         {
            values[0] += 2.0 * cz / (hinz * hinz) + 1.*az / hinz;
         }
      }
      else /* diff sign use forward scheme */
      {
         values[3] = -cz / (hinz * hinz);
         values[6] = -cz / (hinz * hinz) + az / hinz;
         if (nz > 1)
         {
            values[0] += 2.0 * cz / (hinz * hinz) - 1.*az / hinz;
         }
      }
   }
   else /* centered difference scheme */
   {
      values[1] = -cx / (hinx * hinx) - ax / (2.*hinx);
      values[2] = -cy / (hiny * hiny) - ay / (2.*hiny);
      values[3] = -cz / (hinz * hinz) - az / (2.*hinz);
      values[4] = -cx / (hinx * hinx) + ax / (2.*hinx);
      values[5] = -cy / (hiny * hiny) + ay / (2.*hiny);
      values[6] = -cz / (hinz * hinz) + az / (2.*hinz);

      if (nx > 1)
      {
         values[0] += 2.0 * cx / (hinx * hinx);
      }
      if (ny > 1)
      {
         values[0] += 2.0 * cy / (hiny * hiny);
      }
      if (nz > 1)
      {
         values[0] += 2.0 * cz / (hinz * hinz);
      }
   }

   A = (NALU_HYPRE_ParCSRMatrix) GenerateDifConv(nalu_hypre_MPI_COMM_WORLD,
                                            nx, ny, nz, P, Q, R, p, q, r, values);

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build matrix from one file on Proc. 0. Expects matrix to be in
 * CSR format. Distributes matrix across processors giving each about
 * the same number of rows.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParFromOneFile( NALU_HYPRE_Int                  argc,
                     char                *argv[],
                     NALU_HYPRE_Int                  arg_index,
                     NALU_HYPRE_Int                  num_functions,
                     NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   char               *filename;

   NALU_HYPRE_CSRMatrix  A_CSR = NULL;
   NALU_HYPRE_BigInt       *row_part = NULL;
   NALU_HYPRE_BigInt       *col_part = NULL;

   NALU_HYPRE_Int          myid, numprocs;
   NALU_HYPRE_Int          i, rest, size, num_nodes, num_dofs;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );
   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &numprocs );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      nalu_hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * Generate the matrix
       *-----------------------------------------------------------*/

      A_CSR = NALU_HYPRE_CSRMatrixRead(filename);
   }

   if (myid == 0 && num_functions > 1)
   {
      NALU_HYPRE_CSRMatrixGetNumRows(A_CSR, &num_dofs);
      num_nodes = num_dofs / num_functions;
      if (num_dofs == num_functions * num_nodes)
      {
         row_part = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  numprocs + 1, NALU_HYPRE_MEMORY_HOST);

         row_part[0] = 0;
         size = num_nodes / numprocs;
         rest = num_nodes - size * numprocs;
         for (i = 0; i < rest; i++)
         {
            row_part[i + 1] = row_part[i] + (size + 1) * num_functions;
         }
         for (i = rest; i < numprocs; i++)
         {
            row_part[i + 1] = row_part[i] + size * num_functions;
         }

         col_part = row_part;
      }
   }

   NALU_HYPRE_CSRMatrixToParCSRMatrix(nalu_hypre_MPI_COMM_WORLD, A_CSR, row_part, col_part, A_ptr);

   if (myid == 0)
   {
      NALU_HYPRE_CSRMatrixDestroy(A_CSR);
   }

   return (0);
}

/*----------------------------------------------------------------------
 * Build Function array from files on different processors
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildFuncsFromFiles(    NALU_HYPRE_Int                  argc,
                        char                *argv[],
                        NALU_HYPRE_Int                  arg_index,
                        NALU_HYPRE_ParCSRMatrix   parcsr_A,
                        NALU_HYPRE_Int                **dof_func_ptr     )
{
   /*----------------------------------------------------------------------
    * Build Function array from files on different processors
    *----------------------------------------------------------------------*/

   nalu_hypre_printf (" Feature is not implemented yet!\n");
   return (0);

}

/*----------------------------------------------------------------------
 * Build Function array from file on master process
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildFuncsFromOneFile(  NALU_HYPRE_Int                  argc,
                        char                *argv[],
                        NALU_HYPRE_Int                  arg_index,
                        NALU_HYPRE_ParCSRMatrix   parcsr_A,
                        NALU_HYPRE_Int                **dof_func_ptr     )
{
   char           *filename;

   NALU_HYPRE_Int             myid, num_procs;
   NALU_HYPRE_Int             first_row_index;
   NALU_HYPRE_Int             last_row_index;
   NALU_HYPRE_BigInt         *partitioning;
   NALU_HYPRE_Int            *dof_func;
   NALU_HYPRE_Int            *dof_func_local;
   NALU_HYPRE_Int             i, j;
   NALU_HYPRE_Int             local_size;
   NALU_HYPRE_Int             global_size;
   nalu_hypre_MPI_Request    *requests;
   nalu_hypre_MPI_Status     *status, status0;
   MPI_Comm              comm;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   comm = nalu_hypre_MPI_COMM_WORLD;
   nalu_hypre_MPI_Comm_rank(comm, &myid );
   nalu_hypre_MPI_Comm_size(comm, &num_procs );

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      nalu_hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      FILE *fp;
      nalu_hypre_printf("  Funcs FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * read in the data
       *-----------------------------------------------------------*/
      fp = fopen(filename, "r");

      nalu_hypre_fscanf(fp, "%d", &global_size);
      dof_func = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  global_size, NALU_HYPRE_MEMORY_HOST);

      for (j = 0; j < global_size; j++)
      {
         nalu_hypre_fscanf(fp, "%d", &dof_func[j]);
      }

      fclose(fp);
   }

   NALU_HYPRE_ParCSRMatrixGetGlobalRowPartitioning(parcsr_A, 0, &partitioning);
   first_row_index = nalu_hypre_ParCSRMatrixFirstRowIndex(parcsr_A);
   last_row_index  = nalu_hypre_ParCSRMatrixLastRowIndex(parcsr_A);
   local_size      = last_row_index - first_row_index + 1;
   dof_func_local = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_size, NALU_HYPRE_MEMORY_HOST);
   if (myid == 0)
   {
      requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request, num_procs - 1, NALU_HYPRE_MEMORY_HOST);
      status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status, num_procs - 1, NALU_HYPRE_MEMORY_HOST);
      for (i = 1; i < num_procs; i++)
      {
         nalu_hypre_MPI_Isend(&dof_func[partitioning[i]],
                         (partitioning[i + 1] - partitioning[i]),
                         NALU_HYPRE_MPI_INT, i, 0, comm, &requests[i - 1]);
      }
      for (i = 0; i < local_size; i++)
      {
         dof_func_local[i] = dof_func[i];
      }
      nalu_hypre_MPI_Waitall(num_procs - 1, requests, status);
      nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      nalu_hypre_MPI_Recv(dof_func_local, local_size, NALU_HYPRE_MPI_INT, 0, 0, comm, &status0);
   }

   *dof_func_ptr = dof_func_local;

   if (myid == 0) { nalu_hypre_TFree(dof_func, NALU_HYPRE_MEMORY_HOST); }

   if (partitioning) { nalu_hypre_TFree(partitioning, NALU_HYPRE_MEMORY_HOST); }

   return (0);
}

/*----------------------------------------------------------------------
 * Build Rhs from one file on Proc. 0. Distributes vector across processors
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildRhsParFromOneFile( NALU_HYPRE_Int                  argc,
                        char                *argv[],
                        NALU_HYPRE_Int                  arg_index,
                        NALU_HYPRE_ParCSRMatrix   parcsr_A,
                        NALU_HYPRE_ParVector     *b_ptr     )
{
   char           *filename;
   NALU_HYPRE_Int       myid;
   NALU_HYPRE_BigInt   *partitioning;
   NALU_HYPRE_ParVector b;
   NALU_HYPRE_Vector    b_CSR = NULL;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );
   partitioning = nalu_hypre_ParCSRMatrixRowStarts(parcsr_A);

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/

   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      nalu_hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  Rhs FromFile: %s\n", filename);

      /*-----------------------------------------------------------
       * Generate the matrix
       *-----------------------------------------------------------*/

      b_CSR = NALU_HYPRE_VectorRead(filename);
   }
   NALU_HYPRE_VectorToParVector(nalu_hypre_MPI_COMM_WORLD, b_CSR, partitioning, &b);

   *b_ptr = b;

   NALU_HYPRE_VectorDestroy(b_CSR);

   return (0);
}

/*----------------------------------------------------------------------
 * Build Rhs from one file on Proc. 0. Distributes vector across processors
 * giving each about using the distribution of the matrix A.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildBigArrayFromOneFile( NALU_HYPRE_Int            argc,
                          char                *argv[],
                          const char          *array_name,
                          NALU_HYPRE_Int            arg_index,
                          NALU_HYPRE_BigInt        *partitioning,
                          NALU_HYPRE_Int           *size,
                          NALU_HYPRE_BigInt       **array_ptr )
{
   MPI_Comm        comm = nalu_hypre_MPI_COMM_WORLD;
   char           *filename;
   FILE           *fp;
   NALU_HYPRE_Int       myid;
   NALU_HYPRE_Int       num_procs;
   NALU_HYPRE_Int       global_size;
   NALU_HYPRE_BigInt   *global_array;
   NALU_HYPRE_BigInt   *array;
   NALU_HYPRE_BigInt   *send_buffer;
   NALU_HYPRE_Int      *send_counts = NULL;
   NALU_HYPRE_Int      *displs;
   NALU_HYPRE_Int      *array_procs;
   NALU_HYPRE_Int       j, jj, proc;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/
   nalu_hypre_MPI_Comm_rank(comm, &myid);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   if (arg_index < argc)
   {
      filename = argv[arg_index];
   }
   else
   {
      if (myid == 0)
      {
         nalu_hypre_printf("Error: No filename specified \n");
      }
      nalu_hypre_MPI_Abort(comm, 1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/
   if (myid == 0)
   {
      nalu_hypre_printf("  %s array FromFile: %s\n", array_name, filename);

      /*-----------------------------------------------------------
       * Read data
       *-----------------------------------------------------------*/
      fp = fopen(filename, "r");

      nalu_hypre_fscanf(fp, "%d", &global_size);
      global_array = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, global_size, NALU_HYPRE_MEMORY_HOST);
      for (j = 0; j < global_size; j++)
      {
         nalu_hypre_fscanf(fp, "%d", &global_array[j]);
      }

      fclose(fp);
   }

   /*-----------------------------------------------------------
    * Distribute data
    *-----------------------------------------------------------*/
   if (myid == 0)
   {
      send_counts = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_procs, NALU_HYPRE_MEMORY_HOST);
      displs      = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_procs, NALU_HYPRE_MEMORY_HOST);
      array_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int, global_size, NALU_HYPRE_MEMORY_HOST);
      send_buffer = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, global_size, NALU_HYPRE_MEMORY_HOST);
      for (j = 0; j < global_size; j++)
      {
         for (proc = 0; proc < (num_procs + 1); proc++)
         {
            if (global_array[j] < partitioning[proc])
            {
               proc--; break;
            }
         }

         if (proc < num_procs)
         {
            send_counts[proc]++;
            array_procs[j] = proc;
         }
         else
         {
            array_procs[j] = -1; // Not found
         }
      }

      for (proc = 0; proc < (num_procs - 1); proc++)
      {
         displs[proc + 1] = displs[proc] + send_counts[proc];
      }
   }
   nalu_hypre_MPI_Scatter(send_counts, 1, NALU_HYPRE_MPI_INT, size, 1, NALU_HYPRE_MPI_INT, 0, comm);

   if (myid == 0)
   {
      for (proc = 0; proc < num_procs; proc++)
      {
         send_counts[proc] = 0;
      }

      for (j = 0; j < global_size; j++)
      {
         proc = array_procs[j];
         if (proc > -1)
         {
            jj = displs[proc] + send_counts[proc];
            send_buffer[jj] = global_array[j];
            send_counts[proc]++;
         }
      }
   }

   array = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, *size, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_MPI_Scatterv(send_buffer, send_counts, displs, NALU_HYPRE_MPI_BIG_INT,
                      array, *size, NALU_HYPRE_MPI_BIG_INT, 0, comm);
   *array_ptr = array;

   /* Free memory */
   if (myid == 0)
   {
      nalu_hypre_TFree(send_counts, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(send_buffer, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(displs, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(array_procs, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(global_array, NALU_HYPRE_MEMORY_HOST);
   }

   return 0;
}

/*----------------------------------------------------------------------
 * Build standard 9-point laplacian in 2D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParLaplacian9pt( NALU_HYPRE_Int            argc,
                      char                *argv[],
                      NALU_HYPRE_Int            arg_index,
                      NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_BigInt              nx, ny;
   NALU_HYPRE_Int                 P, Q;

   NALU_HYPRE_ParCSRMatrix  A;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 p, q;
   NALU_HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;

   P  = 1;
   Q  = num_procs;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q) != num_procs)
   {
      nalu_hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  Laplacian 9pt:\n");
      nalu_hypre_printf("    (nx, ny) = (%b, %b)\n", nx, ny);
      nalu_hypre_printf("    (Px, Py) = (%d, %d)\n\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */
   p = myid % P;
   q = ( myid - p) / P;

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  2, NALU_HYPRE_MEMORY_HOST);

   values[1] = -1.;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0;
   }
   if (ny > 1)
   {
      values[0] += 2.0;
   }
   if (nx > 1 && ny > 1)
   {
      values[0] += 4.0;
   }

   A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian9pt(nalu_hypre_MPI_COMM_WORLD,
                                                 nx, ny, P, Q, p, q, values);

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}
/*----------------------------------------------------------------------
 * Build 27-point laplacian in 3D,
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParLaplacian27pt( NALU_HYPRE_Int            argc,
                       char                *argv[],
                       NALU_HYPRE_Int            arg_index,
                       NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_BigInt              nx, ny, nz;
   NALU_HYPRE_Int                 P, Q, R;

   NALU_HYPRE_ParCSRMatrix  A;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 p, q, r;
   NALU_HYPRE_Real         *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) != num_procs)
   {
      nalu_hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  Laplacian_27pt:\n");
      nalu_hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      nalu_hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n\n", P,  Q,  R);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  2, NALU_HYPRE_MEMORY_HOST);

   values[0] = 26.0;
   if (nx == 1 || ny == 1 || nz == 1)
   {
      values[0] = 8.0;
   }
   if (nx * ny == 1 || nx * nz == 1 || ny * nz == 1)
   {
      values[0] = 2.0;
   }
   values[1] = -1.;

   A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian27pt(nalu_hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, values);

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}


/*----------------------------------------------------------------------
 * Build 7-point in 2D
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParRotate7pt( NALU_HYPRE_Int            argc,
                   char                *argv[],
                   NALU_HYPRE_Int            arg_index,
                   NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_BigInt              nx, ny;
   NALU_HYPRE_Int                 P, Q;

   NALU_HYPRE_ParCSRMatrix  A;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 p, q;
   NALU_HYPRE_Real          eps, alpha;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;

   P  = 1;
   Q  = num_procs;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-alpha") == 0 )
      {
         arg_index++;
         alpha  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-eps") == 0 )
      {
         arg_index++;
         eps  = atof(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q) != num_procs)
   {
      nalu_hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  Rotate 7pt:\n");
      nalu_hypre_printf("    alpha = %f, eps = %f\n", alpha, eps);
      nalu_hypre_printf("    (nx, ny) = (%b, %b)\n", nx, ny);
      nalu_hypre_printf("    (Px, Py) = (%d, %d)\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q from P,Q and myid */
   p = myid % P;
   q = ( myid - p) / P;

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   A = (NALU_HYPRE_ParCSRMatrix) GenerateRotate7pt(nalu_hypre_MPI_COMM_WORLD,
                                              nx, ny, P, Q, p, q, alpha, eps);

   *A_ptr = A;

   return (0);
}

/*----------------------------------------------------------------------
 * Build standard 7-point difference operator using centered differences
 *
 *  eps*(a(x,y,z) ux)x + (b(x,y,z) uy)y + (c(x,y,z) uz)z
 *  d(x,y,z) ux + e(x,y,z) uy + f(x,y,z) uz + g(x,y,z) u
 *
 *  functions a,b,c,d,e,f,g need to be defined inside par_vardifconv.c
 *
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParVarDifConv( NALU_HYPRE_Int            argc,
                    char                *argv[],
                    NALU_HYPRE_Int            arg_index,
                    NALU_HYPRE_ParCSRMatrix  *A_ptr,
                    NALU_HYPRE_ParVector     *rhs_ptr     )
{
   NALU_HYPRE_BigInt              nx, ny, nz;
   NALU_HYPRE_Int                 P, Q, R;

   NALU_HYPRE_ParCSRMatrix  A;
   NALU_HYPRE_ParVector  rhs;

   NALU_HYPRE_Int           num_procs, myid;
   NALU_HYPRE_Int           p, q, r;
   NALU_HYPRE_Int           type;
   NALU_HYPRE_Real          eps;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;
   P  = 1;
   Q  = num_procs;
   R  = 1;
   eps = 1.0;

   /* type: 0   : default FD;
    *       1-3 : FD and examples 1-3 in Ruge-Stuben paper */
   type = 0;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-eps") == 0 )
      {
         arg_index++;
         eps  = atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-vardifconvRS") == 0 )
      {
         arg_index++;
         type = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Check a few things
    *-----------------------------------------------------------*/

   if ((P * Q * R) != num_procs)
   {
      nalu_hypre_printf("Error: Invalid number of processors or processor topology \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  ell PDE: eps = %f\n", eps);
      nalu_hypre_printf("    Dx(aDxu) + Dy(bDyu) + Dz(cDzu) + d Dxu + e Dyu + f Dzu  + g u= f\n");
      nalu_hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      nalu_hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
   }
   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   if (0 == type)
   {
      A = (NALU_HYPRE_ParCSRMatrix) GenerateVarDifConv(nalu_hypre_MPI_COMM_WORLD,
                                                  nx, ny, nz, P, Q, R, p, q, r, eps, &rhs);
   }
   else
   {
      A = (NALU_HYPRE_ParCSRMatrix) GenerateRSVarDifConv(nalu_hypre_MPI_COMM_WORLD,
                                                    nx, ny, nz, P, Q, R, p, q, r, eps, &rhs,
                                                    type);
   }

   *A_ptr = A;
   *rhs_ptr = rhs;

   return (0);
}

/**************************************************************************/

NALU_HYPRE_Int SetSysVcoefValues(NALU_HYPRE_Int num_fun, NALU_HYPRE_BigInt nx, NALU_HYPRE_BigInt ny, NALU_HYPRE_BigInt nz,
                            NALU_HYPRE_Real vcx,
                            NALU_HYPRE_Real vcy, NALU_HYPRE_Real vcz, NALU_HYPRE_Int mtx_entry, NALU_HYPRE_Real *values)
{


   NALU_HYPRE_Int sz = num_fun * num_fun;

   values[1 * sz + mtx_entry] = -vcx;
   values[2 * sz + mtx_entry] = -vcy;
   values[3 * sz + mtx_entry] = -vcz;
   values[0 * sz + mtx_entry] = 0.0;

   if (nx > 1)
   {
      values[0 * sz + mtx_entry] += 2.0 * vcx;
   }
   if (ny > 1)
   {
      values[0 * sz + mtx_entry] += 2.0 * vcy;
   }
   if (nz > 1)
   {
      values[0 * sz + mtx_entry] += 2.0 * vcz;
   }

   return 0;

}

/*----------------------------------------------------------------------
 * Build coordinates for 1D/2D/3D
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParCoordinates( NALU_HYPRE_Int            argc,
                     char                *argv[],
                     NALU_HYPRE_Int            arg_index,
                     NALU_HYPRE_Int           *coorddim_ptr,
                     float              **coord_ptr     )
{
   NALU_HYPRE_BigInt              nx, ny, nz;
   NALU_HYPRE_Int                 P, Q, R;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 p, q, r;

   NALU_HYPRE_Int                 coorddim;
   float               *coordinates;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 10;
   ny = 10;
   nz = 10;

   P  = 1;
   Q  = num_procs;
   R  = 1;

   /*-----------------------------------------------------------
    * Parse command line
    *-----------------------------------------------------------*/
   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-P") == 0 )
      {
         arg_index++;
         P  = atoi(argv[arg_index++]);
         Q  = atoi(argv[arg_index++]);
         R  = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the coordinates
    *-----------------------------------------------------------*/

   coorddim = 3;
   if (nx < 2) { coorddim--; }
   if (ny < 2) { coorddim--; }
   if (nz < 2) { coorddim--; }

   if (coorddim > 0)
      coordinates = GenerateCoordinates (nalu_hypre_MPI_COMM_WORLD,
                                         nx, ny, nz, P, Q, R, p, q, r, coorddim);
   else
   {
      coordinates = NULL;
   }

   *coorddim_ptr = coorddim;
   *coord_ptr = coordinates;
   return (0);
}


/* begin lobpcg */

/*----------------------------------------------------------------------
 * Build standard 7-point laplacian in 3D.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParIsoLaplacian( NALU_HYPRE_Int argc, char** argv, NALU_HYPRE_ParCSRMatrix *A_ptr )
{

   NALU_HYPRE_BigInt              nx, ny, nz;
   NALU_HYPRE_Real          cx, cy, cz;

   NALU_HYPRE_Int                 P, Q, R;

   NALU_HYPRE_ParCSRMatrix  A;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 p, q, r;
   NALU_HYPRE_Real         *values;

   NALU_HYPRE_Int arg_index;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   P  = 1;
   Q  = num_procs;
   R  = 1;

   nx = 10;
   ny = 10;
   nz = 10;

   cx = 1.0;
   cy = 1.0;
   cz = 1.0;


   arg_index = 0;
   while (arg_index < argc)
   {
      if ( strcmp(argv[arg_index], "-n") == 0 )
      {
         arg_index++;
         nx = atoi(argv[arg_index++]);
         ny = atoi(argv[arg_index++]);
         nz = atoi(argv[arg_index++]);
      }
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  Laplacian:\n");
      nalu_hypre_printf("    (nx, ny, nz) = (%b, %b, %b)\n", nx, ny, nz);
      nalu_hypre_printf("    (Px, Py, Pz) = (%d, %d, %d)\n", P,  Q,  R);
      nalu_hypre_printf("    (cx, cy, cz) = (%f, %f, %f)\n\n", cx, cy, cz);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   /* compute p,q,r from P,Q,R and myid */
   p = myid % P;
   q = (( myid - p) / P) % Q;
   r = ( myid - p - P * q) / ( P * Q );

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  4, NALU_HYPRE_MEMORY_HOST);

   values[1] = -cx;
   values[2] = -cy;
   values[3] = -cz;

   values[0] = 0.;
   if (nx > 1)
   {
      values[0] += 2.0 * cx;
   }
   if (ny > 1)
   {
      values[0] += 2.0 * cy;
   }
   if (nz > 1)
   {
      values[0] += 2.0 * cz;
   }

   A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian(nalu_hypre_MPI_COMM_WORLD,
                                              nx, ny, nz, P, Q, R, p, q, r, values);

   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

   *A_ptr = A;

   return (0);
}

/* end lobpcg */
