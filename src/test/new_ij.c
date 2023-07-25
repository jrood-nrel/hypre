/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
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
#include "NALU_HYPRE_parcsr_ls.h"
#include "_nalu_hypre_parcsr_mv.h"
#include "NALU_HYPRE_krylov.h"

#ifdef __cplusplus
extern "C" {
#endif

NALU_HYPRE_Int BuildParFromFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                            NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParRhsFromFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                               NALU_HYPRE_ParVector *b_ptr );

NALU_HYPRE_Int BuildParLaplacian (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                             NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParSysLaplacian (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParDifConv (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                           NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParFromOneFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                               NALU_HYPRE_Int num_functions, NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildFuncsFromFiles (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                               NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_Int **dof_func_ptr );
NALU_HYPRE_Int BuildFuncsFromOneFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                 NALU_HYPRE_ParCSRMatrix A, NALU_HYPRE_Int **dof_func_ptr );
NALU_HYPRE_Int BuildRhsParFromOneFile (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                  NALU_HYPRE_Int *partitioning, NALU_HYPRE_ParVector *b_ptr );
NALU_HYPRE_Int BuildParLaplacian9pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParLaplacian27pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                                 NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParRotate7pt (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                             NALU_HYPRE_ParCSRMatrix *A_ptr );
NALU_HYPRE_Int BuildParVarDifConv (NALU_HYPRE_Int argc, char *argv [], NALU_HYPRE_Int arg_index,
                              NALU_HYPRE_ParCSRMatrix *A_ptr, NALU_HYPRE_ParVector *rhs_ptr );
NALU_HYPRE_ParCSRMatrix GenerateSysLaplacian (MPI_Comm comm, NALU_HYPRE_Int nx, NALU_HYPRE_Int ny, NALU_HYPRE_Int nz,
                                         NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Int r,
                                         NALU_HYPRE_Int num_fun, NALU_HYPRE_Real *mtrx, NALU_HYPRE_Real *value);
NALU_HYPRE_ParCSRMatrix GenerateSysLaplacianVCoef (MPI_Comm comm, NALU_HYPRE_Int nx, NALU_HYPRE_Int ny,
                                              NALU_HYPRE_Int nz,
                                              NALU_HYPRE_Int P, NALU_HYPRE_Int Q, NALU_HYPRE_Int R, NALU_HYPRE_Int p, NALU_HYPRE_Int q, NALU_HYPRE_Int r,
                                              NALU_HYPRE_Int num_fun, NALU_HYPRE_Real *mtrx, NALU_HYPRE_Real *value);
NALU_HYPRE_Int SetSysVcoefValues(NALU_HYPRE_Int num_fun, NALU_HYPRE_Int nx, NALU_HYPRE_Int ny, NALU_HYPRE_Int nz,
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
#define SECOND_TIME 0

nalu_hypre_int
main( nalu_hypre_int argc,
      char *argv[] )
{
   NALU_HYPRE_Int                 arg_index;
   NALU_HYPRE_Int                 print_usage;
   NALU_HYPRE_Int                 sparsity_known = 0;
   NALU_HYPRE_Int                 add = 0;
   NALU_HYPRE_Int                 off_proc = 0;
   NALU_HYPRE_Int                 chunk = 0;
   NALU_HYPRE_Int                 omp_flag = 0;
   NALU_HYPRE_Int                 build_matrix_type;
   NALU_HYPRE_Int                 build_matrix_arg_index;
   NALU_HYPRE_Int                 build_rhs_type;
   NALU_HYPRE_Int                 build_rhs_arg_index;
   NALU_HYPRE_Int                 build_src_type;
   NALU_HYPRE_Int                 build_src_arg_index;
   NALU_HYPRE_Int                 build_funcs_type;
   NALU_HYPRE_Int                 build_funcs_arg_index;
   NALU_HYPRE_Int                 solver_id;
   NALU_HYPRE_Int                 solver_type = 1;
   NALU_HYPRE_Int                 ioutdat;
   NALU_HYPRE_Int                 poutdat;
   NALU_HYPRE_Int                 debug_flag;
   NALU_HYPRE_Int                 ierr = 0;
   NALU_HYPRE_Int                 i, j;
   NALU_HYPRE_Int                 max_levels = 25;
   NALU_HYPRE_Int                 num_iterations;
   NALU_HYPRE_Int                 pcg_num_its, dscg_num_its;
   NALU_HYPRE_Int                 max_iter = 1000;
   NALU_HYPRE_Int                 mg_max_iter = 100;
   NALU_HYPRE_Int                 nodal = 0;
   NALU_HYPRE_Int                 nodal_diag = 0;
   NALU_HYPRE_Real          cf_tol = 0.9;
   NALU_HYPRE_Real          norm;
   NALU_HYPRE_Real          final_res_norm;
   void               *object;

   NALU_HYPRE_IJMatrix      ij_A;
   NALU_HYPRE_IJVector      ij_b;
   NALU_HYPRE_IJVector      ij_x;
   NALU_HYPRE_IJVector      *ij_rbm;

   NALU_HYPRE_ParCSRMatrix  parcsr_A;
   NALU_HYPRE_ParVector     b;
   NALU_HYPRE_ParVector     x;
   NALU_HYPRE_ParVector     *interp_vecs = NULL;

   NALU_HYPRE_Solver        amg_solver;
   NALU_HYPRE_Solver        pcg_solver;
   NALU_HYPRE_Solver        pcg_precond = NULL, pcg_precond_gotten;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 local_row;
   NALU_HYPRE_Int                *row_sizes;
   NALU_HYPRE_Int                *diag_sizes;
   NALU_HYPRE_Int                *offdiag_sizes;
   NALU_HYPRE_Int                *rows;
   NALU_HYPRE_Int                 size;
   NALU_HYPRE_Int                *ncols;
   NALU_HYPRE_Int                *col_inds;
   NALU_HYPRE_Int                *dof_func;
   NALU_HYPRE_Int             num_functions = 1;
   NALU_HYPRE_Int             num_paths = 1;
   NALU_HYPRE_Int             agg_num_levels = 0;
   NALU_HYPRE_Int             ns_coarse = 1;

   NALU_HYPRE_Int             time_index;
   MPI_Comm            comm = nalu_hypre_MPI_COMM_WORLD;
   NALU_HYPRE_Int M, N;
   NALU_HYPRE_Int first_local_row, last_local_row, local_num_rows;
   NALU_HYPRE_Int first_local_col, last_local_col, local_num_cols;
   NALU_HYPRE_Int variant, overlap, domain_type;
   NALU_HYPRE_Real schwarz_rlx_weight;
   NALU_HYPRE_Real *values, val;

   NALU_HYPRE_Int use_nonsymm_schwarz = 0;
   NALU_HYPRE_Int test_ij = 0;
   NALU_HYPRE_Int build_rbm = 0;
   NALU_HYPRE_Int build_rbm_index = 0;
   NALU_HYPRE_Int num_interp_vecs = 0;
   NALU_HYPRE_Int interp_vec_variant = 0;
   NALU_HYPRE_Int Q_max = 0;
   NALU_HYPRE_Real Q_trunc = 0;

   const NALU_HYPRE_Real dt_inf = 1.e40;
   NALU_HYPRE_Real dt = dt_inf;

   /* parameters for BoomerAMG */
   NALU_HYPRE_Real   strong_threshold;
   NALU_HYPRE_Real   trunc_factor;
   NALU_HYPRE_Real   jacobi_trunc_threshold;
   NALU_HYPRE_Real   S_commpkg_switch = 1.0;
   NALU_HYPRE_Real   CR_rate = 0.7;
   NALU_HYPRE_Real   CR_strong_th = 0.0;
   NALU_HYPRE_Int      CR_use_CG = 0;
   NALU_HYPRE_Int      P_max_elmts = 0;
   NALU_HYPRE_Int      cycle_type;
   NALU_HYPRE_Int      coarsen_type = 6;
   NALU_HYPRE_Int      measure_type = 0;
   NALU_HYPRE_Int      num_sweeps = 1;
   NALU_HYPRE_Int      IS_type;
   NALU_HYPRE_Int      num_CR_relax_steps = 2;
   NALU_HYPRE_Int      relax_type;
   NALU_HYPRE_Int      relax_coarse = -1;
   NALU_HYPRE_Int      relax_up = -1;
   NALU_HYPRE_Int      relax_down = -1;
   NALU_HYPRE_Int      relax_order = 1;
   NALU_HYPRE_Int      level_w = -1;
   NALU_HYPRE_Int      level_ow = -1;
   /* NALU_HYPRE_Int       smooth_lev; */
   /* NALU_HYPRE_Int       smooth_rlx = 8; */
   NALU_HYPRE_Int       smooth_type = 6;
   NALU_HYPRE_Int       smooth_num_levels = 0;
   NALU_HYPRE_Int      smooth_num_sweeps = 1;
   NALU_HYPRE_Int      coarse_threshold = 9;
   NALU_HYPRE_Int      min_coarse_size = 0;
   /* redundant coarse grid solve */
   NALU_HYPRE_Int      seq_threshold = 0;
   NALU_HYPRE_Int      redundant = 0;
   /* additive versions */
   NALU_HYPRE_Int additive = -1;
   NALU_HYPRE_Int mult_add = -1;
   NALU_HYPRE_Int simple = -1;
   NALU_HYPRE_Int add_P_max_elmts = 0;
   NALU_HYPRE_Real add_trunc_factor = 0;

   NALU_HYPRE_Int    rap2 = 0;
   NALU_HYPRE_Int    keepTranspose = 0;
   NALU_HYPRE_Real   relax_wt;
   NALU_HYPRE_Real   relax_wt_level;
   NALU_HYPRE_Real   outer_wt;
   NALU_HYPRE_Real   outer_wt_level;
   NALU_HYPRE_Real   tol = 1.e-8, pc_tol = 0.;
   NALU_HYPRE_Real   atol = 0.0;
   NALU_HYPRE_Real   max_row_sum = 1.;

   NALU_HYPRE_Int cheby_order = 2;
   NALU_HYPRE_Real cheby_fraction = .3;

   /* for CGC BM Aug 25, 2006 */
   NALU_HYPRE_Int      cgcits = 1;
   /* for coordinate plotting BM Oct 24, 2006 */
   NALU_HYPRE_Int      plot_grids = 0;
   NALU_HYPRE_Int      coord_dim  = 3;
   float    *coordinates = NULL;
   char    plot_file_name[256];

   /* parameters for ParaSAILS */
   NALU_HYPRE_Real   sai_threshold = 0.1;
   NALU_HYPRE_Real   sai_filter = 0.1;

   /* parameters for PILUT */
   NALU_HYPRE_Real   drop_tol = -1;
   NALU_HYPRE_Int      nonzeros_to_keep = -1;

   /* parameters for Euclid or ILU smoother in AMG */
   NALU_HYPRE_Real   eu_ilut = 0.0;
   NALU_HYPRE_Real   eu_sparse_A = 0.0;
   NALU_HYPRE_Int       eu_bj = 0;
   NALU_HYPRE_Int       eu_level = -1;
   NALU_HYPRE_Int       eu_stats = 0;
   NALU_HYPRE_Int       eu_mem = 0;
   NALU_HYPRE_Int       eu_row_scale = 0; /* Euclid only */

   /* parameters for GMRES */
   NALU_HYPRE_Int       k_dim;
   /* parameters for LGMRES */
   NALU_HYPRE_Int       aug_dim;
   /* parameters for GSMG */
   NALU_HYPRE_Int      gsmg_samples = 5;
   /* interpolation */
   NALU_HYPRE_Int      interp_type  = 0; /* default value */
   NALU_HYPRE_Int      post_interp_type  = 0; /* default value */
   /* aggressive coarsening */
   NALU_HYPRE_Int      agg_interp_type  = 4; /* default value */
   NALU_HYPRE_Int      agg_P_max_elmts  = 0; /* default value */
   NALU_HYPRE_Int      agg_P12_max_elmts  = 0; /* default value */
   NALU_HYPRE_Real   agg_trunc_factor  = 0; /* default value */
   NALU_HYPRE_Real   agg_P12_trunc_factor  = 0; /* default value */

   NALU_HYPRE_Int      print_system = 0;

   NALU_HYPRE_Int rel_change = 0;

   NALU_HYPRE_Real     *nongalerk_tol = NULL;
   NALU_HYPRE_Int       nongalerk_num_tol = 0;

   NALU_HYPRE_Int *row_nums = NULL;
   NALU_HYPRE_Int *num_cols = NULL;
   NALU_HYPRE_Int *col_nums = NULL;
   NALU_HYPRE_Int i_indx, j_indx, num_rows;
   NALU_HYPRE_Real *data = NULL;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   /* Initialize MPI */
   nalu_hypre_MPI_Init(&argc, &argv);

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   build_matrix_type = 2;
   build_matrix_arg_index = argc;
   build_rhs_type = 2;
   build_rhs_arg_index = argc;
   build_src_type = -1;
   build_src_arg_index = argc;
   build_funcs_type = 0;
   build_funcs_arg_index = argc;
   relax_type = 3;
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
      else if ( strcmp(argv[arg_index], "-concrete_parcsr") == 0 )
      {
         arg_index++;
         build_matrix_arg_index = arg_index;
      }
      else if ( strcmp(argv[arg_index], "-solver") == 0 )
      {
         arg_index++;
         solver_id = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rbm") == 0 )
      {
         arg_index++;
         build_rbm      = 1;
         num_interp_vecs = atoi(argv[arg_index++]);
         build_rbm_index = arg_index;
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
         CR_rate = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-crst") == 0 )
      {
         arg_index++;
         CR_strong_th = (NALU_HYPRE_Real)atof(argv[arg_index++]);
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
         dt = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         build_rhs_type = -1;
         if ( build_src_type == -1 ) { build_src_type = 2; }
      }
      else if ( strcmp(argv[arg_index], "-help") == 0 )
      {
         print_usage = 1;
      }
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

   if (solver_id == 8 || solver_id == 18)
   {
      max_levels = 1;
   }

   /* defaults for BoomerAMG */
   if (solver_id == 0 || solver_id == 1 || solver_id == 3 || solver_id == 5
       || solver_id == 9 || solver_id == 13 || solver_id == 14
       || solver_id == 15 || solver_id == 20 || solver_id == 51 || solver_id == 61)
   {
      strong_threshold = 0.25;
      trunc_factor = 0.;
      jacobi_trunc_threshold = 0.01;
      cycle_type = 1;
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
      else if ( strcmp(argv[arg_index], "-aug") == 0 )
      {
         arg_index++;
         aug_dim = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-w") == 0 )
      {
         arg_index++;
         relax_wt = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-wl") == 0 )
      {
         arg_index++;
         relax_wt_level = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         level_w = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ow") == 0 )
      {
         arg_index++;
         outer_wt = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-owl") == 0 )
      {
         arg_index++;
         outer_wt_level = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         level_ow = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sw") == 0 )
      {
         arg_index++;
         schwarz_rlx_weight = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-coarse_th") == 0 )
      {
         arg_index++;
         coarse_threshold  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-min_cs") == 0 )
      {
         arg_index++;
         min_coarse_size  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-seq_th") == 0 )
      {
         arg_index++;
         seq_threshold  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-red") == 0 )
      {
         arg_index++;
         redundant  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-th") == 0 )
      {
         arg_index++;
         strong_threshold  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-CF") == 0 )
      {
         arg_index++;
         relax_order = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cf") == 0 )
      {
         arg_index++;
         cf_tol  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-tol") == 0 )
      {
         arg_index++;
         tol  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-atol") == 0 )
      {
         arg_index++;
         atol  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-mxrs") == 0 )
      {
         arg_index++;
         max_row_sum  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sai_th") == 0 )
      {
         arg_index++;
         sai_threshold  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sai_filt") == 0 )
      {
         arg_index++;
         sai_filter  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-drop_tol") == 0 )
      {
         arg_index++;
         drop_tol  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nonzeros_to_keep") == 0 )
      {
         arg_index++;
         nonzeros_to_keep  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-ilut") == 0 )
      {
         arg_index++;
         eu_ilut  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-sparseA") == 0 )
      {
         arg_index++;
         eu_sparse_A  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
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
         trunc_factor  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
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
         Q_trunc  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-Qmx") == 0 )
      {
         arg_index++;
         Q_max = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-jtr") == 0 )
      {
         arg_index++;
         jacobi_trunc_threshold  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-Ssw") == 0 )
      {
         arg_index++;
         S_commpkg_switch = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-solver_type") == 0 )
      {
         arg_index++;
         solver_type  = atoi(argv[arg_index++]);
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
         agg_trunc_factor  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-agg_P12_tr") == 0 )
      {
         arg_index++;
         agg_P12_trunc_factor  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-cheby_order") == 0 )
      {
         arg_index++;
         cheby_order = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-cheby_fraction") == 0 )
      {
         arg_index++;
         cheby_fraction = (NALU_HYPRE_Real)atof(argv[arg_index++]);
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
      else if ( strcmp(argv[arg_index], "-add_Pmx") == 0 )
      {
         arg_index++;
         add_P_max_elmts  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-add_tr") == 0 )
      {
         arg_index++;
         add_trunc_factor  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-rap") == 0 )
      {
         arg_index++;
         rap2  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-keepT") == 0 )
      {
         arg_index++;
         keepTranspose  = atoi(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-nongalerk_tol") == 0 )
      {
         arg_index++;
         nongalerk_num_tol = atoi(argv[arg_index++]);
         nongalerk_tol = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  nongalerk_num_tol, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < nongalerk_num_tol; i++)
         {
            nongalerk_tol[i] = (NALU_HYPRE_Real)atof(argv[arg_index++]);
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
      else
      {
         arg_index++;
      }
   }

   /*-----------------------------------------------------------
    * Print usage info
    *-----------------------------------------------------------*/

   if ( (print_usage) && (myid == 0) )
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
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -exact_size            : inserts immediately into ParCSR structure\n");
      nalu_hypre_printf("  -storage_low           : allocates not enough storage for aux struct\n");
      nalu_hypre_printf("  -concrete_parcsr       : use parcsr matrix type as concrete type\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -rhsfromfile           : ");
      nalu_hypre_printf("rhs read from multiple files (IJ format)\n");
      nalu_hypre_printf("  -rhsfromonefile        : ");
      nalu_hypre_printf("rhs read from a single file (CSR format)\n");
      nalu_hypre_printf("  -rhsparcsrfile        :  ");
      nalu_hypre_printf("rhs read from multiple files (ParCSR format)\n");
      nalu_hypre_printf("  -rhsrand               : rhs is random vector\n");
      nalu_hypre_printf("  -rhsisone              : rhs is vector with unit components (default)\n");
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
      nalu_hypre_printf("backward Euler source is random vector with components in range 0 - 1\n");
      nalu_hypre_printf("  -srcisone              : ");
      nalu_hypre_printf("backward Euler source is vector with unit components (default)\n");
      nalu_hypre_printf("  -srczero               : ");
      nalu_hypre_printf("backward Euler source is zero-vector\n");
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
      nalu_hypre_printf("       18=ParaSails-GMRES\n");
      nalu_hypre_printf("       20=Hybrid solver/ DiagScale, AMG \n");
      nalu_hypre_printf("       43=Euclid-PCG      44=Euclid-GMRES   \n");
      nalu_hypre_printf("       45=Euclid-BICGSTAB\n");
      nalu_hypre_printf("       50=DS-LGMRES         51=AMG-LGMRES     \n");
      nalu_hypre_printf("       60=DS-FlexGMRES         61=AMG-FlexGMRES     \n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -cljp                 : CLJP coarsening \n");
      nalu_hypre_printf("  -cljp1                : CLJP coarsening, fixed random \n");
      nalu_hypre_printf("  -cgc                  : CGC coarsening \n");
      nalu_hypre_printf("  -cgce                 : CGC-E coarsening \n");
      nalu_hypre_printf("  -pmis                 : PMIS coarsening \n");
      nalu_hypre_printf("  -pmis1                : PMIS coarsening, fixed random \n");
      nalu_hypre_printf("  -hmis                 : HMIS coarsening \n");
      nalu_hypre_printf("  -ruge                 : Ruge-Stueben coarsening (local)\n");
      nalu_hypre_printf("  -ruge1p               : Ruge-Stueben coarsening 1st pass only(local)\n");
      nalu_hypre_printf("  -ruge3                : third pass on boundary\n");
      nalu_hypre_printf("  -ruge3c               : third pass on boundary, keep c-points\n");
      nalu_hypre_printf("  -falgout              : local Ruge_Stueben followed by CLJP\n");
      nalu_hypre_printf("  -gm                   : use global measures\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -interptype  <val>    : set interpolation type\n");
      nalu_hypre_printf("       0=Classical modified interpolation (default)  \n");
      nalu_hypre_printf("       1=least squares interpolation (for GSMG only)  \n");
      nalu_hypre_printf("       0=Classical modified interpolation for hyperbolic PDEs \n");
      nalu_hypre_printf("       3=direct interpolation with separation of weights  \n");
      nalu_hypre_printf("       4=multipass interpolation  \n");
      nalu_hypre_printf("       5=multipass interpolation with separation of weights  \n");
      nalu_hypre_printf("       6=extended classical modified interpolation  \n");
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
      nalu_hypre_printf("  -ns_down    <val>       : set no. of sweeps for down cycle\n");
      nalu_hypre_printf("  -ns_up      <val>       : set no. of sweeps for up cycle\n");
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  -mu   <val>            : set AMG cycles (1=V, 2=W, etc.)\n");
      nalu_hypre_printf("  -th   <val>            : set AMG threshold Theta = val \n");
      nalu_hypre_printf("  -tr   <val>            : set AMG interpolation truncation factor = val \n");
      nalu_hypre_printf("  -Pmx  <val>            : set maximal no. of elmts per row for AMG interpolation \n");
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
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("Running with these driver parameters:\n");
      nalu_hypre_printf("  solver ID    = %d\n\n", solver_id);
   }

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
   }
   else if ( build_matrix_type == 5 )
   {
      BuildParDifConv(argc, argv, build_matrix_arg_index, &parcsr_A);
   }
   else if ( build_matrix_type == 6 )
   {
      BuildParVarDifConv(argc, argv, build_matrix_arg_index, &parcsr_A, &b);
      /*NALU_HYPRE_ParCSRMatrixPrint(parcsr_A,"mat100");*/
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

      local_num_rows = last_local_row - first_local_row + 1;
      local_num_cols = last_local_col - first_local_col + 1;
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

      local_num_rows = last_local_row - first_local_row + 1;
      local_num_cols = last_local_col - first_local_col + 1;
   }
   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("Generate Matrix", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   /* Check the ij interface - not necessary if one just wants to test solvers */
   if (test_ij && build_matrix_type > -1)
   {
      NALU_HYPRE_Int mx_size = 5;
      time_index = nalu_hypre_InitializeTiming("Generate IJ matrix");
      nalu_hypre_BeginTiming(time_index);

      ierr += NALU_HYPRE_ParCSRMatrixGetDims( parcsr_A, &M, &N );

      ierr += NALU_HYPRE_IJMatrixCreate( comm, first_local_row, last_local_row,
                                    first_local_col, last_local_col,
                                    &ij_A );

      ierr += NALU_HYPRE_IJMatrixSetObjectType( ij_A, NALU_HYPRE_PARCSR );
      num_rows = local_num_rows;
      if (off_proc)
      {
         if (myid != num_procs - 1) { num_rows++; }
         if (myid) { num_rows++; }
      }
      /* The following shows how to build an IJMatrix if one has only an
         estimate for the row sizes */
      row_nums = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_rows, NALU_HYPRE_MEMORY_HOST);
      num_cols = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_rows, NALU_HYPRE_MEMORY_HOST);
      if (sparsity_known == 1)
      {
         diag_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
         offdiag_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
      }
      else
      {
         size = 5;
         if (sparsity_known == 0)
         {
            if (build_matrix_type == 2) { size = 7; }
            if (build_matrix_type == 3) { size = 9; }
            if (build_matrix_type == 4) { size = 27; }
         }
         row_sizes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_rows, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_rows; i++)
         {
            row_sizes[i] = size;
         }
      }
      local_row = 0;
      if (build_matrix_type == 2) { mx_size = 7; }
      if (build_matrix_type == 3) { mx_size = 9; }
      if (build_matrix_type == 4) { mx_size = 27; }
      col_nums = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  mx_size * num_rows, NALU_HYPRE_MEMORY_HOST);
      data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  mx_size * num_rows, NALU_HYPRE_MEMORY_HOST);
      i_indx = 0;
      j_indx = 0;
      if (off_proc && myid)
      {
         num_cols[i_indx] = 2;
         row_nums[i_indx++] = first_local_row - 1;
         col_nums[j_indx] = first_local_row - 1;
         data[j_indx++] = 6.;
         col_nums[j_indx] = first_local_row - 2;
         data[j_indx++] = -1;
      }
      for (i = 0; i < local_num_rows; i++)
      {
         row_nums[i_indx] = first_local_row + i;
         ierr += NALU_HYPRE_ParCSRMatrixGetRow( parcsr_A, first_local_row + i, &size,
                                           &col_inds, &values);
         num_cols[i_indx++] = size;
         for (j = 0; j < size; j++)
         {
            col_nums[j_indx] = col_inds[j];
            data[j_indx++] = values[j];
            if (sparsity_known == 1)
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
         }
         local_row++;
         ierr += NALU_HYPRE_ParCSRMatrixRestoreRow( parcsr_A, first_local_row + i, &size,
                                               &col_inds, &values );
      }
      if (off_proc && myid != num_procs - 1)
      {
         num_cols[i_indx] = 2;
         row_nums[i_indx++] = last_local_row + 1;
         col_nums[j_indx] = last_local_row + 2;
         data[j_indx++] = -1.;
         col_nums[j_indx] = last_local_row + 1;
         data[j_indx++] = 6;
      }

      /*ierr += NALU_HYPRE_IJMatrixSetRowSizes ( ij_A, (const NALU_HYPRE_Int *) num_cols );*/
      if (sparsity_known == 1)
         ierr += NALU_HYPRE_IJMatrixSetDiagOffdSizes( ij_A, (const NALU_HYPRE_Int *) diag_sizes,
                                                 (const NALU_HYPRE_Int *) offdiag_sizes );
      else
      {
         ierr = NALU_HYPRE_IJMatrixSetRowSizes ( ij_A, (const NALU_HYPRE_Int *) row_sizes );
      }

      ierr += NALU_HYPRE_IJMatrixInitialize( ij_A );

      if (omp_flag) { NALU_HYPRE_IJMatrixSetOMPFlag(ij_A, 1); }

      if (chunk)
      {
         if (add)
            ierr += NALU_HYPRE_IJMatrixAddToValues(ij_A, num_rows, num_cols, row_nums,
                                              (const NALU_HYPRE_Int *) col_nums,
                                              (const NALU_HYPRE_Real *) data);
         else
            ierr += NALU_HYPRE_IJMatrixSetValues(ij_A, num_rows, num_cols, row_nums,
                                            (const NALU_HYPRE_Int *) col_nums,
                                            (const NALU_HYPRE_Real *) data);
      }
      else
      {
         j_indx = 0;
         for (i = 0; i < num_rows; i++)
         {
            if (add)
               ierr += NALU_HYPRE_IJMatrixAddToValues( ij_A, 1, &num_cols[i], &row_nums[i],
                                                  (const NALU_HYPRE_Int *) &col_nums[j_indx],
                                                  (const NALU_HYPRE_Real *) &data[j_indx] );
            else
               ierr += NALU_HYPRE_IJMatrixSetValues( ij_A, 1, &num_cols[i], &row_nums[i],
                                                (const NALU_HYPRE_Int *) &col_nums[j_indx],
                                                (const NALU_HYPRE_Real *) &data[j_indx] );
            j_indx += num_cols[i];
         }
      }
      nalu_hypre_TFree(col_nums, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(row_nums, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(num_cols, NALU_HYPRE_MEMORY_HOST);
      if (sparsity_known == 1)
      {
         nalu_hypre_TFree(diag_sizes, NALU_HYPRE_MEMORY_HOST);
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
         the opportunity to IJMatrixAddTo before a IJMatrixAssemble. */

      ncols    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  last_local_row - first_local_row + 1, NALU_HYPRE_MEMORY_HOST);
      rows     = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  last_local_row - first_local_row + 1, NALU_HYPRE_MEMORY_HOST);
      col_inds = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  last_local_row - first_local_row + 1, NALU_HYPRE_MEMORY_HOST);
      values   = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  last_local_row - first_local_row + 1, NALU_HYPRE_MEMORY_HOST);

      if (dt < dt_inf)
      {
         val = 1. / dt;
      }
      else
      {
         val = 0.;   /* Use zero to avoid unintentional loss of significance */
      }

      for (i = first_local_row; i <= last_local_row; i++)
      {
         j = i - first_local_row;
         rows[j] = i;
         ncols[j] = 1;
         col_inds[j] = i;
         values[j] = val;
      }

      ierr += NALU_HYPRE_IJMatrixAddToValues( ij_A,
                                         local_num_rows,
                                         ncols, rows,
                                         (const NALU_HYPRE_Int *) col_inds,
                                         (const NALU_HYPRE_Real *) values );

      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(col_inds, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(rows, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(ncols, NALU_HYPRE_MEMORY_HOST);

      /* If sparsity pattern is not changed since last IJMatrixAssemble call,
         this should be a no-op */

      ierr += NALU_HYPRE_IJMatrixAssemble( ij_A );

      /*-----------------------------------------------------------
       * Fetch the resulting underlying matrix out
       *-----------------------------------------------------------*/
      if (build_matrix_type > -1)
      {
         ierr += NALU_HYPRE_ParCSRMatrixDestroy(parcsr_A);
      }

      ierr += NALU_HYPRE_IJMatrixGetObject( ij_A, &object);
      parcsr_A = (NALU_HYPRE_ParCSRMatrix) object;

   }

   /*-----------------------------------------------------------
    * Set up the interp vector
    *-----------------------------------------------------------*/

   if ( build_rbm)
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
    * Set up the RHS and initial guess
    *-----------------------------------------------------------*/

   time_index = nalu_hypre_InitializeTiming("RHS and Initial Guess");
   nalu_hypre_BeginTiming(time_index);

   if ( build_rhs_type == 0 )
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

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 1 )
   {

#if 0
      nalu_hypre_printf("build_rhs_type == 1 not currently implemented\n");
      return (-1);
#else
      /* RHS - this has not been tested for multiple processors*/
      BuildRhsParFromOneFile(argc, argv, build_rhs_arg_index, NULL, &b);

      nalu_hypre_printf("  Initial guess is 0\n");

      ij_b = NULL;

      /* initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;

#endif
   }
   else if ( build_rhs_type == 7 )
   {

      /* rhs */
      BuildParRhsFromFile(argc, argv, build_rhs_arg_index, &b);

      nalu_hypre_printf("  Initial guess is 0\n");

      ij_b = NULL;

      /* initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }

   else if ( build_rhs_type == 2 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector has unit components\n");
         nalu_hypre_printf("  Initial guess is 0\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = 1.0;
      }
      NALU_HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 3 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector has random components and unit 2-norm\n");
         nalu_hypre_printf("  Initial guess is 0\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);
      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* For purposes of this test, NALU_HYPRE_ParVector functions are used, but
         these are not necessary.  For a clean use of the interface, the user
         "should" modify components of ij_x by using functions
         NALU_HYPRE_IJVectorSetValues or NALU_HYPRE_IJVectorAddToValues */

      NALU_HYPRE_ParVectorSetRandomValues(b, 22775);
      NALU_HYPRE_ParVectorInnerProd(b, b, &norm);
      norm = 1. / nalu_hypre_sqrt(norm);
      ierr = NALU_HYPRE_ParVectorScale(norm, b);

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_rhs_type == 4 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector set for solution with unit components\n");
         nalu_hypre_printf("  Initial guess is 0\n");
      }

      /* Temporary use of solution vector */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 1.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);
      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      NALU_HYPRE_ParCSRMatrixMatvec(1., parcsr_A, x, 0., b);

      /* Initial guess */
      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
   }
   else if ( build_rhs_type == 5 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  RHS vector is 0\n");
         nalu_hypre_printf("  Initial guess has unit components\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 1.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }

   if ( build_src_type == 0 )
   {
#if 0
      /* RHS */
      BuildRhsParFromFile(argc, argv, build_src_arg_index, &b);
#endif

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

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_src_type == 1 )
   {
      nalu_hypre_printf("build_src_type == 1 not currently implemented\n");
      return (-1);

#if 0
      BuildRhsParFromOneFile(argc, argv, build_src_arg_index, part_b, &b);
#endif
   }
   else if ( build_src_type == 2 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  Source vector has unit components\n");
         nalu_hypre_printf("  Initial unknown vector is 0\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = 1.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         0 here) is usually used as the initial guess */
      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_src_type == 3 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  Source vector has random components in range 0 - 1\n");
         nalu_hypre_printf("  Initial unknown vector is 0\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);
      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_rows, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = nalu_hypre_Rand();
      }

      NALU_HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         0 here) is usually used as the initial guess */
      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = 0.;
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }
   else if ( build_src_type == 4 )
   {
      if (myid == 0)
      {
         nalu_hypre_printf("  Source vector is 0 \n");
         nalu_hypre_printf("  Initial unknown vector has random components in range 0 - 1\n");
      }

      /* RHS */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_row, last_local_row, &ij_b);
      NALU_HYPRE_IJVectorSetObjectType(ij_b, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_b);

      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_SeedRand(myid);
      for (i = 0; i < local_num_rows; i++)
      {
         values[i] = nalu_hypre_Rand() / dt;
      }
      NALU_HYPRE_IJVectorSetValues(ij_b, local_num_rows, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_b, &object );
      b = (NALU_HYPRE_ParVector) object;

      /* Initial guess */
      NALU_HYPRE_IJVectorCreate(nalu_hypre_MPI_COMM_WORLD, first_local_col, last_local_col, &ij_x);
      NALU_HYPRE_IJVectorSetObjectType(ij_x, NALU_HYPRE_PARCSR);
      NALU_HYPRE_IJVectorInitialize(ij_x);

      /* For backward Euler the previous backward Euler iterate (assumed
         random in 0 - 1 here) is usually used as the initial guess */
      values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  local_num_cols, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_SeedRand(myid);
      for (i = 0; i < local_num_cols; i++)
      {
         values[i] = nalu_hypre_Rand();
      }
      NALU_HYPRE_IJVectorSetValues(ij_x, local_num_cols, NULL, values);
      nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);

      ierr = NALU_HYPRE_IJVectorGetObject( ij_x, &object );
      x = (NALU_HYPRE_ParVector) object;
   }

   nalu_hypre_EndTiming(time_index);
   nalu_hypre_PrintTiming("IJ Vector Setup", nalu_hypre_MPI_COMM_WORLD);
   nalu_hypre_FinalizeTiming(time_index);
   nalu_hypre_ClearTiming();

   if (num_functions > 1)
   {
      dof_func = NULL;
      if (build_funcs_type == 1)
      {
         BuildFuncsFromOneFile(argc, argv, build_funcs_arg_index, parcsr_A, &dof_func);
      }
      else if (build_funcs_type == 2)
      {
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

   if (print_system)
   {
      NALU_HYPRE_IJMatrixPrint(ij_A, "IJ.out.A");
      NALU_HYPRE_IJVectorPrint(ij_b, "IJ.out.b");
      NALU_HYPRE_IJVectorPrint(ij_x, "IJ.out.x0");

      /* NALU_HYPRE_ParCSRMatrixPrint( parcsr_A, "new_mat.A" );*/
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
      NALU_HYPRE_ParCSRHybridSetRelaxType(amg_solver, relax_type);
      NALU_HYPRE_ParCSRHybridSetAggNumLevels(amg_solver, agg_num_levels);
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

      NALU_HYPRE_ParCSRHybridSetup(amg_solver, parcsr_A, b, x);

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

   if (solver_id == 0)
   {
      if (myid == 0) { nalu_hypre_printf("Solver:  AMG\n"); }
      time_index = nalu_hypre_InitializeTiming("BoomerAMG Setup");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_BoomerAMGCreate(&amg_solver);
      /* BM Aug 25, 2006 */
      NALU_HYPRE_BoomerAMGSetCGCIts(amg_solver, cgcits);
      NALU_HYPRE_BoomerAMGSetInterpType(amg_solver, interp_type);
      NALU_HYPRE_BoomerAMGSetPostInterpType(amg_solver, post_interp_type);
      NALU_HYPRE_BoomerAMGSetNumSamples(amg_solver, gsmg_samples);
      NALU_HYPRE_BoomerAMGSetCoarsenType(amg_solver, coarsen_type);
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
      NALU_HYPRE_BoomerAMGSetCycleType(amg_solver, cycle_type);
      NALU_HYPRE_BoomerAMGSetNumSweeps(amg_solver, num_sweeps);
      NALU_HYPRE_BoomerAMGSetISType(amg_solver, IS_type);
      NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(amg_solver, num_CR_relax_steps);
      NALU_HYPRE_BoomerAMGSetCRRate(amg_solver, CR_rate);
      NALU_HYPRE_BoomerAMGSetCRStrongTh(amg_solver, CR_strong_th);
      NALU_HYPRE_BoomerAMGSetCRUseCG(amg_solver, CR_use_CG);
      NALU_HYPRE_BoomerAMGSetRelaxType(amg_solver, relax_type);
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
      NALU_HYPRE_BoomerAMGSetChebyOrder(amg_solver, cheby_order);
      NALU_HYPRE_BoomerAMGSetChebyFraction(amg_solver, cheby_fraction);
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
      NALU_HYPRE_BoomerAMGSetCycleNumSweeps(amg_solver, ns_coarse, 3);
      if (num_functions > 1)
      {
         NALU_HYPRE_BoomerAMGSetDofFunc(amg_solver, dof_func);
      }
      NALU_HYPRE_BoomerAMGSetAdditive(amg_solver, additive);
      NALU_HYPRE_BoomerAMGSetMultAdditive(amg_solver, mult_add);
      NALU_HYPRE_BoomerAMGSetSimple(amg_solver, simple);
      NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(amg_solver, add_P_max_elmts);
      NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(amg_solver, add_trunc_factor);

      NALU_HYPRE_BoomerAMGSetMaxIter(amg_solver, mg_max_iter);
      NALU_HYPRE_BoomerAMGSetRAP2(amg_solver, rap2);
      NALU_HYPRE_BoomerAMGSetKeepTranspose(amg_solver, keepTranspose);
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

      NALU_HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);

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

      NALU_HYPRE_BoomerAMGGetNumIterations(amg_solver, &num_iterations);
      NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm(amg_solver, &final_res_norm);

      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("BoomerAMG Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      NALU_HYPRE_ParVectorSetRandomValues(x, 775);
      NALU_HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);
      NALU_HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);
#endif

      NALU_HYPRE_BoomerAMGDestroy(amg_solver);
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
      NALU_HYPRE_BoomerAMGSetNumSweeps(amg_solver, num_sweeps);
      NALU_HYPRE_BoomerAMGSetRelaxType(amg_solver, relax_type);
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
      NALU_HYPRE_BoomerAMGSetChebyOrder(amg_solver, cheby_order);
      NALU_HYPRE_BoomerAMGSetChebyFraction(amg_solver, cheby_fraction);
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
      NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(amg_solver, add_P_max_elmts);
      NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(amg_solver, add_trunc_factor);
      NALU_HYPRE_BoomerAMGSetRAP2(amg_solver, rap2);
      NALU_HYPRE_BoomerAMGSetKeepTranspose(amg_solver, keepTranspose);
      if (nongalerk_tol)
      {
         NALU_HYPRE_BoomerAMGSetNonGalerkinTol(amg_solver, nongalerk_tol[nongalerk_num_tol - 1]);
         for (i = 0; i < nongalerk_num_tol - 1; i++)
         {
            NALU_HYPRE_BoomerAMGSetLevelNonGalerkinTol(amg_solver, nongalerk_tol[i], i);
         }
      }

      NALU_HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);

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

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      NALU_HYPRE_ParVectorSetRandomValues(x, 775);
      NALU_HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, b, x);
      NALU_HYPRE_BoomerAMGSolve(amg_solver, parcsr_A, b, x);
#endif

      NALU_HYPRE_BoomerAMGDestroy(amg_solver);
   }

   if (solver_id == 999)
   {
      NALU_HYPRE_IJMatrix ij_M;
      NALU_HYPRE_ParCSRMatrix  parcsr_mat;

      /* use ParaSails preconditioner */
      if (myid == 0) { nalu_hypre_printf("Test ParaSails Build IJMatrix\n"); }

      NALU_HYPRE_IJMatrixPrint(ij_A, "parasails.in");

      NALU_HYPRE_ParaSailsCreate(nalu_hypre_MPI_COMM_WORLD, &pcg_precond);
      NALU_HYPRE_ParaSailsSetParams(pcg_precond, 0., 0);
      NALU_HYPRE_ParaSailsSetFilter(pcg_precond, 0.);
      NALU_HYPRE_ParaSailsSetLogging(pcg_precond, ioutdat);

      NALU_HYPRE_IJMatrixGetObject( ij_A, &object);
      parcsr_mat = (NALU_HYPRE_ParCSRMatrix) object;

      NALU_HYPRE_ParaSailsSetup(pcg_precond, parcsr_mat, NULL, NULL);
      NALU_HYPRE_ParaSailsBuildIJMatrix(pcg_precond, &ij_M);
      NALU_HYPRE_IJMatrixPrint(ij_M, "parasails.out");

      if (myid == 0) { nalu_hypre_printf("Printed to parasails.out.\n"); }
      exit(0);
   }

   /*-----------------------------------------------------------
    * Solve the system using PCG
    *-----------------------------------------------------------*/

   if (solver_id == 1 || solver_id == 2 || solver_id == 8 ||
       solver_id == 12 || solver_id == 14 || solver_id == 43)
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
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
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
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
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
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         NALU_HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         NALU_HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
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
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
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
         NALU_HYPRE_BoomerAMGSetRelaxOrder(pcg_precond, relax_order);
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
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
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
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
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
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

      NALU_HYPRE_PCGSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                     (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("PCG Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_PCGSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                     (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_PCGGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_PCGGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      NALU_HYPRE_ParVectorSetRandomValues(x, 775);
      NALU_HYPRE_PCGSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                     (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);
      NALU_HYPRE_PCGSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                     (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);
#endif

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

      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }

   }

   /*-----------------------------------------------------------
    * Solve the system using GMRES
    *-----------------------------------------------------------*/

   if (solver_id == 3 || solver_id == 4 || solver_id == 7 ||
       solver_id == 15 || solver_id == 18 || solver_id == 44)
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

      if (solver_id == 3)
      {
         /* use BoomerAMG as preconditioner */
         if (myid == 0) { nalu_hypre_printf("Solver: AMG-GMRES\n"); }

         NALU_HYPRE_BoomerAMGCreate(&pcg_precond);
         NALU_HYPRE_BoomerAMGSetCGCIts(pcg_precond, cgcits);
         NALU_HYPRE_BoomerAMGSetInterpType(pcg_precond, interp_type);
         NALU_HYPRE_BoomerAMGSetPostInterpType(pcg_precond, post_interp_type);
         NALU_HYPRE_BoomerAMGSetNumSamples(pcg_precond, gsmg_samples);
         NALU_HYPRE_BoomerAMGSetTol(pcg_precond, pc_tol);
         NALU_HYPRE_BoomerAMGSetCoarsenType(pcg_precond, coarsen_type);
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
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
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
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
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
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         NALU_HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         NALU_HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
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
            NALU_HYPRE_BoomerAMGSetInterpVectors(pcg_precond, 1, interp_vecs);
            NALU_HYPRE_BoomerAMGSetInterpVecVariant(pcg_precond, interp_vec_variant);
            NALU_HYPRE_BoomerAMGSetInterpVecQMax(pcg_precond, Q_max);
            NALU_HYPRE_BoomerAMGSetInterpVecAbsQTrunc(pcg_precond, Q_trunc);
         }
         NALU_HYPRE_GMRESSetMaxIter(pcg_solver, mg_max_iter);
         NALU_HYPRE_GMRESSetPrecond(pcg_solver,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSolve,
                               (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BoomerAMGSetup,
                               pcg_precond);
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
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
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
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
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
         NALU_HYPRE_BoomerAMGSetMaxLevels(pcg_precond, max_levels);
         NALU_HYPRE_BoomerAMGSetMaxRowSum(pcg_precond, max_row_sum);
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
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
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

      NALU_HYPRE_GMRESGetPrecond(pcg_solver, &pcg_precond_gotten);
      if (pcg_precond_gotten != pcg_precond)
      {
         nalu_hypre_printf("NALU_HYPRE_GMRESGetPrecond got bad precond\n");
         return (-1);
      }
      else if (myid == 0)
      {
         nalu_hypre_printf("NALU_HYPRE_GMRESGetPrecond got good precond\n");
      }
      NALU_HYPRE_GMRESSetup
      (pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Setup phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      time_index = nalu_hypre_InitializeTiming("GMRES Solve");
      nalu_hypre_BeginTiming(time_index);

      NALU_HYPRE_GMRESSolve
      (pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

      nalu_hypre_EndTiming(time_index);
      nalu_hypre_PrintTiming("Solve phase times", nalu_hypre_MPI_COMM_WORLD);
      nalu_hypre_FinalizeTiming(time_index);
      nalu_hypre_ClearTiming();

      NALU_HYPRE_GMRESGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_GMRESGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
#if SECOND_TIME
      /* run a second time to check for memory leaks */
      NALU_HYPRE_ParVectorSetRandomValues(x, 775);
      NALU_HYPRE_GMRESSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b,
                       (NALU_HYPRE_Vector)x);
      NALU_HYPRE_GMRESSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b,
                       (NALU_HYPRE_Vector)x);
#endif

      NALU_HYPRE_ParCSRGMRESDestroy(pcg_solver);

      if (solver_id == 3 || solver_id == 15)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }

      if (solver_id == 7)
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
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
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
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
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
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         NALU_HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         NALU_HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
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
      (pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

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

   if (solver_id == 60 || solver_id == 61 )
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
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
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
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
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
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         NALU_HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         NALU_HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
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
      (pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

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

   if (solver_id == 9 || solver_id == 10 || solver_id == 11 || solver_id == 45)
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
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetISType(pcg_precond, IS_type);
         NALU_HYPRE_BoomerAMGSetNumCRRelaxSteps(pcg_precond, num_CR_relax_steps);
         NALU_HYPRE_BoomerAMGSetCRRate(pcg_precond, CR_rate);
         NALU_HYPRE_BoomerAMGSetCRStrongTh(pcg_precond, CR_strong_th);
         NALU_HYPRE_BoomerAMGSetCRUseCG(pcg_precond, CR_use_CG);
         NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
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
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
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
         NALU_HYPRE_BoomerAMGSetCycleNumSweeps(pcg_precond, ns_coarse, 3);
         if (num_functions > 1)
         {
            NALU_HYPRE_BoomerAMGSetDofFunc(pcg_precond, dof_func);
         }
         NALU_HYPRE_BoomerAMGSetAdditive(pcg_precond, additive);
         NALU_HYPRE_BoomerAMGSetMultAdditive(pcg_precond, mult_add);
         NALU_HYPRE_BoomerAMGSetSimple(pcg_precond, simple);
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
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

      NALU_HYPRE_BiCGSTABSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                          (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);

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

      NALU_HYPRE_BiCGSTABGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_BiCGSTABGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);
#if SECOND_TIME
      /* run a second time to check for memory leaks */
      NALU_HYPRE_ParVectorSetRandomValues(x, 775);
      NALU_HYPRE_BiCGSTABSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                          (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);
      NALU_HYPRE_BiCGSTABSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A,
                          (NALU_HYPRE_Vector)b, (NALU_HYPRE_Vector)x);
#endif

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

      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("BiCGSTAB Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final BiCGSTAB Relative Residual Norm = %e\n", final_res_norm);
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
         NALU_HYPRE_BoomerAMGSetMaxIter(pcg_precond, 1);
         NALU_HYPRE_BoomerAMGSetCycleType(pcg_precond, cycle_type);
         NALU_HYPRE_BoomerAMGSetNumSweeps(pcg_precond, num_sweeps);
         NALU_HYPRE_BoomerAMGSetRelaxType(pcg_precond, relax_type);
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
         NALU_HYPRE_BoomerAMGSetChebyOrder(pcg_precond, cheby_order);
         NALU_HYPRE_BoomerAMGSetChebyFraction(pcg_precond, cheby_fraction);
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
         NALU_HYPRE_BoomerAMGSetMultAddPMaxElmts(pcg_precond, add_P_max_elmts);
         NALU_HYPRE_BoomerAMGSetMultAddTruncFactor(pcg_precond, add_trunc_factor);
         NALU_HYPRE_BoomerAMGSetRAP2(pcg_precond, rap2);
         NALU_HYPRE_BoomerAMGSetKeepTranspose(pcg_precond, keepTranspose);
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
      NALU_HYPRE_CGNRSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b,
                      (NALU_HYPRE_Vector)x);

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

      NALU_HYPRE_CGNRGetNumIterations(pcg_solver, &num_iterations);
      NALU_HYPRE_CGNRGetFinalRelativeResidualNorm(pcg_solver, &final_res_norm);

#if SECOND_TIME
      /* run a second time to check for memory leaks */
      NALU_HYPRE_ParVectorSetRandomValues(x, 775);
      NALU_HYPRE_CGNRSetup(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b,
                      (NALU_HYPRE_Vector)x);
      NALU_HYPRE_CGNRSolve(pcg_solver, (NALU_HYPRE_Matrix)parcsr_A, (NALU_HYPRE_Vector)b,
                      (NALU_HYPRE_Vector)x);
#endif

      NALU_HYPRE_ParCSRCGNRDestroy(pcg_solver);

      if (solver_id == 5)
      {
         NALU_HYPRE_BoomerAMGDestroy(pcg_precond);
      }
      if (myid == 0)
      {
         nalu_hypre_printf("\n");
         nalu_hypre_printf("Iterations = %d\n", num_iterations);
         nalu_hypre_printf("Final Relative Residual Norm = %e\n", final_res_norm);
         nalu_hypre_printf("\n");
      }
   }

   /*-----------------------------------------------------------
    * Print the solution and other info
    *-----------------------------------------------------------*/

   /* RDF: Why is this here? */
   if (!(build_rhs_type == 1 || build_rhs_type == 7))
   {
      NALU_HYPRE_IJVectorGetObjectType(ij_b, &j);
   }

   if (print_system)
   {
      NALU_HYPRE_IJVectorPrint(ij_x, "IJ.out.x");
   }

   /*-----------------------------------------------------------
    * Finalize things
    *-----------------------------------------------------------*/

   if (test_ij || build_matrix_type == -1) { NALU_HYPRE_IJMatrixDestroy(ij_A); }
   else { NALU_HYPRE_ParCSRMatrixDestroy(parcsr_A); }

   /* for build_rhs_type = 1 or 7, we did not create ij_b  - just b*/
   if (build_rhs_type == 1 || build_rhs_type == 7)
   {
      NALU_HYPRE_ParVectorDestroy(b);
   }
   else
   {
      NALU_HYPRE_IJVectorDestroy(ij_b);
   }

   NALU_HYPRE_IJVectorDestroy(ij_x);

   if (build_rbm)
   {
      for (i = 0; i < num_interp_vecs; i++)
      {
         NALU_HYPRE_IJVectorDestroy(ij_rbm[i]);
      }
      nalu_hypre_TFree(ij_rbm, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(interp_vecs, NALU_HYPRE_MEMORY_HOST);
   }
   if (nongalerk_tol) { nalu_hypre_TFree(nongalerk_tol, NALU_HYPRE_MEMORY_HOST); }

   nalu_hypre_MPI_Finalize();

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
BuildParRhsFromFile( NALU_HYPRE_Int                  argc,
                     char                *argv[],
                     NALU_HYPRE_Int                  arg_index,
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
      nalu_hypre_printf("Error: No filename specified \n");
      exit(1);
   }

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  RhsFromParFile: %s\n", filename);
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
   NALU_HYPRE_Int                 nx, ny, nz;
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
         cx = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         cy = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         cz = (NALU_HYPRE_Real)atof(argv[arg_index++]);
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
         ep = (NALU_HYPRE_Real)atof(argv[arg_index++]);
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
      nalu_hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
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
                 NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_Int                 nx, ny, nz;
   NALU_HYPRE_Int                 P, Q, R;
   NALU_HYPRE_Real          cx, cy, cz;
   NALU_HYPRE_Real          ax, ay, az;
   NALU_HYPRE_Real          hinx, hiny, hinz;

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

   cx = 1.;
   cy = 1.;
   cz = 1.;

   ax = 1.;
   ay = 1.;
   az = 1.;

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
         cx = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         cy = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         cz = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-a") == 0 )
      {
         arg_index++;
         ax = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         ay = (NALU_HYPRE_Real)atof(argv[arg_index++]);
         az = (NALU_HYPRE_Real)atof(argv[arg_index++]);
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
      nalu_hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
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

   hinx = 1. / (nx + 1);
   hiny = 1. / (ny + 1);
   hinz = 1. / (nz + 1);

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  7, NALU_HYPRE_MEMORY_HOST);

   values[1] = -cx / (hinx * hinx);
   values[2] = -cy / (hiny * hiny);
   values[3] = -cz / (hinz * hinz);
   values[4] = -cx / (hinx * hinx) + ax / hinx;
   values[5] = -cy / (hiny * hiny) + ay / hiny;
   values[6] = -cz / (hinz * hinz) + az / hinz;

   values[0] = 0.;
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

   NALU_HYPRE_ParCSRMatrix  A;
   NALU_HYPRE_CSRMatrix  A_CSR = NULL;

   NALU_HYPRE_Int                 myid, numprocs;
   NALU_HYPRE_Int                 i, rest, size, num_nodes, num_dofs;
   NALU_HYPRE_Int            *row_part;
   NALU_HYPRE_Int            *col_part;

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

   row_part = NULL;
   col_part = NULL;
   if (myid == 0 && num_functions > 1)
   {
      NALU_HYPRE_CSRMatrixGetNumRows(A_CSR, &num_dofs);
      num_nodes = num_dofs / num_functions;
      if (num_dofs != num_functions * num_nodes)
      {
         row_part = NULL;
         col_part = NULL;
      }
      else
      {
         row_part = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  numprocs + 1, NALU_HYPRE_MEMORY_HOST);
         row_part[0] = 0;
         size = num_nodes / numprocs;
         rest = num_nodes - size * numprocs;
         for (i = 0; i < numprocs; i++)
         {
            row_part[i + 1] = row_part[i] + size * num_functions;
            if (i < rest) { row_part[i + 1] += num_functions; }
         }
         col_part = row_part;
      }
   }

   NALU_HYPRE_CSRMatrixToParCSRMatrix(nalu_hypre_MPI_COMM_WORLD, A_CSR, row_part, col_part, &A);

   *A_ptr = A;

   if (myid == 0) { NALU_HYPRE_CSRMatrixDestroy(A_CSR); }

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


NALU_HYPRE_Int
BuildFuncsFromOneFile(  NALU_HYPRE_Int                  argc,
                        char                *argv[],
                        NALU_HYPRE_Int                  arg_index,
                        NALU_HYPRE_ParCSRMatrix   parcsr_A,
                        NALU_HYPRE_Int                **dof_func_ptr     )
{
   char           *filename;

   NALU_HYPRE_Int             myid, num_procs;
   NALU_HYPRE_Int            *partitioning;
   NALU_HYPRE_Int            *dof_func;
   NALU_HYPRE_Int            *dof_func_local;
   NALU_HYPRE_Int             i, j;
   NALU_HYPRE_Int             local_size, global_size;
   nalu_hypre_MPI_Request   *requests;
   nalu_hypre_MPI_Status    *status, status0;
   MPI_Comm    comm;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   comm = nalu_hypre_MPI_COMM_WORLD;
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );
   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );

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
   NALU_HYPRE_ParCSRMatrixGetRowPartitioning(parcsr_A, &partitioning);
   local_size = partitioning[myid + 1] - partitioning[myid];
   dof_func_local = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_size, NALU_HYPRE_MEMORY_HOST);

   if (myid == 0)
   {
      requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request, num_procs - 1, NALU_HYPRE_MEMORY_HOST);
      status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status, num_procs - 1, NALU_HYPRE_MEMORY_HOST);
      j = 0;
      for (i = 1; i < num_procs; i++)
         nalu_hypre_MPI_Isend(&dof_func[partitioning[i]],
                         partitioning[i + 1] - partitioning[i],
                         NALU_HYPRE_MPI_INT, i, 0, comm, &requests[j++]);
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
                        NALU_HYPRE_Int                 *partitioning,
                        NALU_HYPRE_ParVector     *b_ptr     )
{
   char           *filename;

   NALU_HYPRE_ParVector b;
   NALU_HYPRE_Vector    b_CSR = NULL;

   NALU_HYPRE_Int             myid;

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
 * Build standard 9-point laplacian in 2D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int
BuildParLaplacian9pt( NALU_HYPRE_Int                  argc,
                      char                *argv[],
                      NALU_HYPRE_Int                  arg_index,
                      NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_Int                 nx, ny;
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
      nalu_hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
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
BuildParLaplacian27pt( NALU_HYPRE_Int                  argc,
                       char                *argv[],
                       NALU_HYPRE_Int                  arg_index,
                       NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_Int                 nx, ny, nz;
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
      nalu_hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
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
BuildParRotate7pt( NALU_HYPRE_Int                  argc,
                   char                *argv[],
                   NALU_HYPRE_Int                  arg_index,
                   NALU_HYPRE_ParCSRMatrix  *A_ptr     )
{
   NALU_HYPRE_Int                 nx, ny;
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
         alpha  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
      }
      else if ( strcmp(argv[arg_index], "-eps") == 0 )
      {
         arg_index++;
         eps  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
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
      nalu_hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
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
BuildParVarDifConv( NALU_HYPRE_Int                  argc,
                    char                *argv[],
                    NALU_HYPRE_Int                  arg_index,
                    NALU_HYPRE_ParCSRMatrix  *A_ptr,
                    NALU_HYPRE_ParVector  *rhs_ptr     )
{
   NALU_HYPRE_Int                 nx, ny, nz;
   NALU_HYPRE_Int                 P, Q, R;

   NALU_HYPRE_ParCSRMatrix  A;
   NALU_HYPRE_ParVector  rhs;

   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 p, q, r;
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
         eps  = (NALU_HYPRE_Real)atof(argv[arg_index++]);
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
      nalu_hypre_printf("    (nx, ny, nz) = (%d, %d, %d)\n", nx, ny, nz);
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

   A = (NALU_HYPRE_ParCSRMatrix) GenerateVarDifConv(nalu_hypre_MPI_COMM_WORLD,
                                               nx, ny, nz, P, Q, R, p, q, r, eps, &rhs);

   *A_ptr = A;
   *rhs_ptr = rhs;

   return (0);
}

/**************************************************************************/


NALU_HYPRE_Int SetSysVcoefValues(NALU_HYPRE_Int num_fun, NALU_HYPRE_Int nx, NALU_HYPRE_Int ny, NALU_HYPRE_Int nz,
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
BuildParCoordinates( NALU_HYPRE_Int                  argc,
                     char                *argv[],
                     NALU_HYPRE_Int                  arg_index,
                     NALU_HYPRE_Int                 *coorddim_ptr,
                     float               **coord_ptr     )
{
   NALU_HYPRE_Int                 nx, ny, nz;
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



