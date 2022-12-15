/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef nalu_hypre_ParMGR_DATA_HEADER
#define nalu_hypre_ParMGR_DATA_HEADER
/*--------------------------------------------------------------------------
 * nalu_hypre_ParMGRData
 *--------------------------------------------------------------------------*/
typedef struct
{
   // block data
   NALU_HYPRE_Int  block_size;
   NALU_HYPRE_Int  *block_num_coarse_indexes;
   NALU_HYPRE_Int  *point_marker_array;
   NALU_HYPRE_Int  **block_cf_marker;

   // initial setup data (user provided)
   NALU_HYPRE_Int num_coarse_levels;
   NALU_HYPRE_Int *num_coarse_per_level;
   NALU_HYPRE_Int **level_coarse_indexes;

   //general data
   NALU_HYPRE_Int max_num_coarse_levels;
   nalu_hypre_ParCSRMatrix **A_array;
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_ParCSRMatrix **P_FF_array;
#endif
   nalu_hypre_ParCSRMatrix **P_array;
   nalu_hypre_ParCSRMatrix **RT_array;
   nalu_hypre_ParCSRMatrix *RAP;
   nalu_hypre_IntArray    **CF_marker_array;
   NALU_HYPRE_Int **coarse_indices_lvls;
   nalu_hypre_ParVector    **F_array;
   nalu_hypre_ParVector    **U_array;
   nalu_hypre_ParVector    *residual;
   NALU_HYPRE_Real    *rel_res_norms;

   nalu_hypre_ParCSRMatrix  **A_ff_array;
   nalu_hypre_ParVector    **F_fine_array;
   nalu_hypre_ParVector    **U_fine_array;
   NALU_HYPRE_Solver **aff_solver;
   NALU_HYPRE_Int   (*fine_grid_solver_setup)(void*, void*, void*, void*);
   NALU_HYPRE_Int   (*fine_grid_solver_solve)(void*, void*, void*, void*);

   NALU_HYPRE_Real   max_row_sum;
   NALU_HYPRE_Int    num_interp_sweeps;
   NALU_HYPRE_Int    num_restrict_sweeps;
   //NALU_HYPRE_Int    interp_type;
   NALU_HYPRE_Int    *interp_type;
   NALU_HYPRE_Int    *restrict_type;
   NALU_HYPRE_Real   strong_threshold;
   NALU_HYPRE_Real   trunc_factor;
   NALU_HYPRE_Real   S_commpkg_switch;
   NALU_HYPRE_Int    P_max_elmts;
   NALU_HYPRE_Int    num_iterations;

   nalu_hypre_Vector **l1_norms;
   NALU_HYPRE_Real    final_rel_residual_norm;
   NALU_HYPRE_Real    tol;
   NALU_HYPRE_Real    relax_weight;
   NALU_HYPRE_Int     relax_type;
   NALU_HYPRE_Int     logging;
   NALU_HYPRE_Int     print_level;
   NALU_HYPRE_Int     frelax_print_level;
   NALU_HYPRE_Int     cg_print_level;
   NALU_HYPRE_Int     max_iter;
   NALU_HYPRE_Int     relax_order;
   NALU_HYPRE_Int     *num_relax_sweeps;

   NALU_HYPRE_Solver coarse_grid_solver;
   NALU_HYPRE_Int     (*coarse_grid_solver_setup)(void*, void*, void*, void*);
   NALU_HYPRE_Int     (*coarse_grid_solver_solve)(void*, void*, void*, void*);

   NALU_HYPRE_Int     use_default_cgrid_solver;
   // Mode to use an external AMG solver for F-relaxation
   // 0: use an external AMG solver that is already setup
   // 1: use an external AMG solver but do setup inside MGR
   // 2: use default internal AMG solver
   NALU_HYPRE_Int     fsolver_mode;
   //  NALU_HYPRE_Int     fsolver_type;
   NALU_HYPRE_Real    omega;

   /* temp vectors for solve phase */
   nalu_hypre_ParVector   *Vtemp;
   nalu_hypre_ParVector   *Ztemp;
   nalu_hypre_ParVector   *Utemp;
   nalu_hypre_ParVector   *Ftemp;

   NALU_HYPRE_Real          **level_diaginv;
   NALU_HYPRE_Real          **frelax_diaginv;
   NALU_HYPRE_Int           n_block;
   NALU_HYPRE_Int           left_size;
   NALU_HYPRE_Int           *blk_size;
   NALU_HYPRE_Int           *level_smooth_iters;
   NALU_HYPRE_Int           *level_smooth_type;
   NALU_HYPRE_Solver        *level_smoother;
   NALU_HYPRE_Int           global_smooth_cycle;

   /*
    Number of points that remain part of the coarse grid throughout the hierarchy.
    For example, number of well equations
    */
   NALU_HYPRE_Int reserved_coarse_size;
   NALU_HYPRE_BigInt *reserved_coarse_indexes;
   NALU_HYPRE_Int *reserved_Cpoint_local_indexes;

   NALU_HYPRE_Int set_non_Cpoints_to_F;
   NALU_HYPRE_BigInt *idx_array;

   /* F-relaxation type */
   NALU_HYPRE_Int *Frelax_method;
   NALU_HYPRE_Int *Frelax_type;

   NALU_HYPRE_Int *Frelax_num_functions;

   /* Non-Galerkin coarse grid */
   NALU_HYPRE_Int *mgr_coarse_grid_method;

   /* V-cycle F relaxation method */
   nalu_hypre_ParAMGData    **FrelaxVcycleData;
   nalu_hypre_ParVector   *VcycleRelaxVtemp;
   nalu_hypre_ParVector   *VcycleRelaxZtemp;

   NALU_HYPRE_Int   max_local_lvls;

   NALU_HYPRE_Int   print_coarse_system;
   NALU_HYPRE_Real  truncate_coarse_grid_threshold;

   /* how to set C points */
   NALU_HYPRE_Int   set_c_points_method;

   /* reduce reserved C-points before coarse grid solve? */
   /* this might be necessary for some applications, e.g. phase transitions */
   NALU_HYPRE_Int   lvl_to_keep_cpoints;

   /* block size for block Jacobi interpolation and relaxation */
   NALU_HYPRE_Int  block_jacobi_bsize;

   NALU_HYPRE_Real  cg_convergence_factor;

   /* Data for Gaussian elimination F-relaxation */
   nalu_hypre_ParAMGData    **GSElimData;

} nalu_hypre_ParMGRData;

// F-relaxation struct for future refactoring of F-relaxation in MGR
typedef struct
{
   NALU_HYPRE_Int relax_type;
   NALU_HYPRE_Int relax_nsweeps;

   nalu_hypre_ParCSRMatrix *A;
   nalu_hypre_ParVector    *b;

   // for hypre's smoother options
   NALU_HYPRE_Int *CF_marker;

   // for block Jacobi/GS option
   NALU_HYPRE_Complex *diaginv;

   // for ILU option
   NALU_HYPRE_Solver frelax_solver;

} nalu_hypre_MGRRelaxData;


#define FMRK  -1
#define CMRK  1
#define UMRK  0
#define S_CMRK  2

#define FPT(i, bsize) (((i) % (bsize)) == FMRK)
#define CPT(i, bsize) (((i) % (bsize)) == CMRK)

//#define SMALLREAL 1e-20
//#define DIVIDE_TOL 1e-32

#endif
