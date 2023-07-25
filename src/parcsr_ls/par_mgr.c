/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Two-grid system solver
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "par_mgr.h"
#include "_nalu_hypre_blas.h"
#include "_nalu_hypre_lapack.h"

//#ifdef NALU_HYPRE_USING_DSUPERLU
//#include "dsuperlu.h"
//#endif

#if defined(NALU_HYPRE_USING_GPU)
void nalu_hypre_NoGPUSupport(char *option)
{
   char msg[256];
   nalu_hypre_sprintf(msg, "Error: Chosen %s option is not currently supported on GPU\n\n", option);
   nalu_hypre_printf("%s ", msg);
   //  nalu_hypre_error_w_msg(1, msg);
   nalu_hypre_MPI_Abort(nalu_hypre_MPI_COMM_WORLD, -1);
}
#endif

/* Need to define these nalu_hypre_lapack protos here instead of including _nalu_hypre_lapack.h to avoid conflicts with
 * dsuperlu.h on some lapack functions. Alternative is to move superLU related functions to a separate file.
*/
/* dgetrf.c */
//NALU_HYPRE_Int nalu_hypre_dgetrf ( NALU_HYPRE_Int *m, NALU_HYPRE_Int *n, NALU_HYPRE_Real *a, NALU_HYPRE_Int *lda, NALU_HYPRE_Int *ipiv,
//                         NALU_HYPRE_Int *info );
/* dgetri.c */
//NALU_HYPRE_Int nalu_hypre_dgetri ( NALU_HYPRE_Int *n, NALU_HYPRE_Real *a, NALU_HYPRE_Int *lda, NALU_HYPRE_Int *ipiv,
//                         NALU_HYPRE_Real *work, NALU_HYPRE_Int *lwork, NALU_HYPRE_Int *info);

/* Create */
void *
nalu_hypre_MGRCreate(void)
{
   nalu_hypre_ParMGRData  *mgr_data;

   mgr_data = nalu_hypre_CTAlloc(nalu_hypre_ParMGRData,  1, NALU_HYPRE_MEMORY_HOST);

   /* block data */
   (mgr_data -> block_size) = 1;
   (mgr_data -> block_num_coarse_indexes) = NULL;
   (mgr_data -> point_marker_array) = NULL;
   (mgr_data -> block_cf_marker) = NULL;

   /* general data */
   (mgr_data -> max_num_coarse_levels) = 10;
   (mgr_data -> A_array) = NULL;
   (mgr_data -> B_array) = NULL;
   (mgr_data -> B_FF_array) = NULL;
#if defined(NALU_HYPRE_USING_GPU)
   (mgr_data -> P_FF_array) = NULL;
#endif
   (mgr_data -> P_array) = NULL;
   (mgr_data -> RT_array) = NULL;
   (mgr_data -> RAP) = NULL;
   (mgr_data -> CF_marker_array) = NULL;
   (mgr_data -> coarse_indices_lvls) = NULL;

   (mgr_data -> A_ff_array) = NULL;
   (mgr_data -> F_fine_array) = NULL;
   (mgr_data -> U_fine_array) = NULL;
   (mgr_data -> aff_solver) = NULL;
   (mgr_data -> fine_grid_solver_setup) = NULL;
   (mgr_data -> fine_grid_solver_solve) = NULL;

   (mgr_data -> F_array) = NULL;
   (mgr_data -> U_array) = NULL;
   (mgr_data -> residual) = NULL;
   (mgr_data -> rel_res_norms) = NULL;
   (mgr_data -> Vtemp) = NULL;
   (mgr_data -> Ztemp) = NULL;
   (mgr_data -> Utemp) = NULL;
   (mgr_data -> Ftemp) = NULL;

   (mgr_data -> num_iterations) = 0;
   (mgr_data -> num_interp_sweeps) = 1;
   (mgr_data -> num_restrict_sweeps) = 1;
   (mgr_data -> trunc_factor) = 0.0;
   (mgr_data -> max_row_sum) = 0.9;
   (mgr_data -> strong_threshold) = 0.25;
   (mgr_data -> P_max_elmts) = 0;

   (mgr_data -> coarse_grid_solver) = NULL;
   (mgr_data -> coarse_grid_solver_setup) = NULL;
   (mgr_data -> coarse_grid_solver_solve) = NULL;

   //(mgr_data -> global_smoother) = NULL;

   (mgr_data -> use_default_cgrid_solver) = 1;
   (mgr_data -> fsolver_mode) = -1; // user or hypre -prescribed F-solver
   (mgr_data -> omega) = 1.;
   (mgr_data -> max_iter) = 20;
   (mgr_data -> tol) = 1.0e-6;
   (mgr_data -> relax_type) = 0;
   (mgr_data -> Frelax_type) = NULL;
   (mgr_data -> relax_order) = 1; // not fully utilized. Only used to compute L1-norms.
   (mgr_data -> num_relax_sweeps) = NULL;
   (mgr_data -> relax_weight) = 1.0;

   (mgr_data -> interp_type) = NULL;
   (mgr_data -> restrict_type) = NULL;
   (mgr_data -> level_smooth_iters) = NULL;
   (mgr_data -> level_smooth_type) = NULL;
   (mgr_data -> level_smoother) = NULL;
   (mgr_data -> global_smooth_cycle) = 1; // Pre = 1 or Post  = 2 global smoothing

   (mgr_data -> logging) = 0;
   (mgr_data -> print_level) = 0;
   (mgr_data -> frelax_print_level) = 0;
   (mgr_data -> cg_print_level) = 0;

   (mgr_data -> l1_norms) = NULL;

   (mgr_data -> reserved_coarse_size) = 0;
   (mgr_data -> reserved_coarse_indexes) = NULL;
   (mgr_data -> reserved_Cpoint_local_indexes) = NULL;

   (mgr_data -> level_diaginv) = NULL;
   (mgr_data -> frelax_diaginv) = NULL;
   //(mgr_data -> global_smooth_iters) = 1;
   //(mgr_data -> global_smooth_type) = 0;

   (mgr_data -> set_non_Cpoints_to_F) = 0;
   (mgr_data -> idx_array) = NULL;

   (mgr_data -> Frelax_method) = NULL;
   (mgr_data -> VcycleRelaxVtemp) = NULL;
   (mgr_data -> VcycleRelaxZtemp) = NULL;
   (mgr_data -> FrelaxVcycleData) = NULL;
   (mgr_data -> Frelax_num_functions) = NULL;
   (mgr_data -> max_local_lvls) = 10;

   (mgr_data -> mgr_coarse_grid_method) = NULL;

   (mgr_data -> print_coarse_system) = 0;

   (mgr_data -> set_c_points_method) = 0;
   (mgr_data -> lvl_to_keep_cpoints) = 0;
   (mgr_data -> cg_convergence_factor) = 0.0;

   (mgr_data -> block_jacobi_bsize) = 0;
   (mgr_data -> blk_size) = NULL;

   (mgr_data -> truncate_coarse_grid_threshold) = 0.0;

   (mgr_data -> GSElimData) = NULL;

   return (void *) mgr_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/* Destroy */
NALU_HYPRE_Int
nalu_hypre_MGRDestroy( void *data )
{
   nalu_hypre_ParMGRData * mgr_data = (nalu_hypre_ParMGRData*) data;

   NALU_HYPRE_Int i;
   NALU_HYPRE_Int num_coarse_levels = (mgr_data -> num_coarse_levels);

   /* block info data */
   if ((mgr_data -> block_cf_marker))
   {
      for (i = 0; i < (mgr_data -> max_num_coarse_levels); i++)
      {
         nalu_hypre_TFree((mgr_data -> block_cf_marker)[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree((mgr_data -> block_cf_marker), NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_TFree(mgr_data -> block_num_coarse_indexes, NALU_HYPRE_MEMORY_HOST);

   /* final residual vector */
   if ((mgr_data -> residual))
   {
      nalu_hypre_ParVectorDestroy( (mgr_data -> residual) );
      (mgr_data -> residual) = NULL;
   }

   nalu_hypre_TFree( (mgr_data -> rel_res_norms), NALU_HYPRE_MEMORY_HOST);

   /* temp vectors for solve phase */
   if ((mgr_data -> Vtemp))
   {
      nalu_hypre_ParVectorDestroy( (mgr_data -> Vtemp) );
      (mgr_data -> Vtemp) = NULL;
   }
   if ((mgr_data -> Ztemp))
   {
      nalu_hypre_ParVectorDestroy( (mgr_data -> Ztemp) );
      (mgr_data -> Ztemp) = NULL;
   }
   if ((mgr_data -> Utemp))
   {
      nalu_hypre_ParVectorDestroy( (mgr_data -> Utemp) );
      (mgr_data -> Utemp) = NULL;
   }
   if ((mgr_data -> Ftemp))
   {
      nalu_hypre_ParVectorDestroy( (mgr_data -> Ftemp) );
      (mgr_data -> Ftemp) = NULL;
   }
   /* coarse grid solver */
   if ((mgr_data -> use_default_cgrid_solver))
   {
      if ((mgr_data -> coarse_grid_solver))
      {
         nalu_hypre_BoomerAMGDestroy( (mgr_data -> coarse_grid_solver) );
      }
      (mgr_data -> coarse_grid_solver) = NULL;
   }
   /* l1_norms */
   if ((mgr_data -> l1_norms))
   {
      for (i = 0; i < (num_coarse_levels); i++)
      {
         nalu_hypre_SeqVectorDestroy((mgr_data -> l1_norms)[i]);
      }
      nalu_hypre_TFree((mgr_data -> l1_norms), NALU_HYPRE_MEMORY_HOST);
   }

   /* coarse_indices_lvls */
   if ((mgr_data -> coarse_indices_lvls))
   {
      for (i = 0; i < (num_coarse_levels); i++)
      {
         nalu_hypre_TFree((mgr_data -> coarse_indices_lvls)[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree((mgr_data -> coarse_indices_lvls), NALU_HYPRE_MEMORY_HOST);
   }

   /* linear system and cf marker array */
   if (mgr_data -> A_array || mgr_data -> P_array || mgr_data -> RT_array ||
       mgr_data -> CF_marker_array)
   {
      for (i = 1; i < num_coarse_levels + 1; i++)
      {
         nalu_hypre_ParVectorDestroy((mgr_data -> F_array)[i]);
         nalu_hypre_ParVectorDestroy((mgr_data -> U_array)[i]);

         if ((mgr_data -> P_array)[i - 1])
         {
            nalu_hypre_ParCSRMatrixDestroy((mgr_data -> P_array)[i - 1]);
         }

         if ((mgr_data -> RT_array)[i - 1])
         {
            nalu_hypre_ParCSRMatrixDestroy((mgr_data -> RT_array)[i - 1]);
         }

         nalu_hypre_IntArrayDestroy(mgr_data -> CF_marker_array[i - 1]);
      }
      for (i = 1; i < (num_coarse_levels); i++)
      {
         if ((mgr_data -> A_array)[i])
         {
            nalu_hypre_ParCSRMatrixDestroy((mgr_data -> A_array)[i]);
         }
      }
   }

   /* Block relaxation/interpolation matrices */
   if (nalu_hypre_ParMGRDataBArray(mgr_data))
   {
      for (i = 0; i < num_coarse_levels; i++)
      {
         nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParMGRDataB(mgr_data, i));
      }
   }

   if (nalu_hypre_ParMGRDataBFFArray(mgr_data))
   {
      for (i = 0; i < num_coarse_levels; i++)
      {
         nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParMGRDataBFF(mgr_data, i));
      }
   }

#if defined(NALU_HYPRE_USING_GPU)
   if (mgr_data -> P_FF_array)
   {
      for (i = 0; i < num_coarse_levels; i++)
      {
         if ((mgr_data -> P_array)[i])
         {
            nalu_hypre_ParCSRMatrixDestroy((mgr_data -> P_FF_array)[i]);
         }
      }
      //nalu_hypre_TFree(P_FF_array, nalu_hypre_HandleMemoryLocation(nalu_hypre_handle()));
      nalu_hypre_TFree((mgr_data -> P_FF_array), NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> P_FF_array) = NULL;
   }
#endif

   /* AMG for Frelax */
   if (mgr_data -> A_ff_array || mgr_data -> F_fine_array || mgr_data -> U_fine_array)
   {
      for (i = 1; i < num_coarse_levels + 1; i++)
      {
         if (mgr_data -> F_fine_array[i])
         {
            nalu_hypre_ParVectorDestroy((mgr_data -> F_fine_array)[i]);
         }
         if (mgr_data -> U_fine_array[i])
         {
            nalu_hypre_ParVectorDestroy((mgr_data -> U_fine_array)[i]);
         }
      }
      for (i = 1; i < (num_coarse_levels); i++)
      {
         if ((mgr_data -> A_ff_array)[i])
         {
            nalu_hypre_ParCSRMatrixDestroy((mgr_data -> A_ff_array)[i]);
         }
      }
      if (mgr_data -> fsolver_mode != 0)
      {
         if ((mgr_data -> A_ff_array)[0])
         {
            nalu_hypre_ParCSRMatrixDestroy((mgr_data -> A_ff_array)[0]);
         }
      }
      nalu_hypre_TFree(mgr_data -> F_fine_array, NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> F_fine_array) = NULL;
      nalu_hypre_TFree(mgr_data -> U_fine_array, NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> U_fine_array) = NULL;
      nalu_hypre_TFree(mgr_data -> A_ff_array, NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> A_ff_array) = NULL;
   }

   if (mgr_data -> aff_solver)
   {
      for (i = 1; i < (num_coarse_levels); i++)
      {
         if ((mgr_data -> aff_solver)[i])
         {
            nalu_hypre_BoomerAMGDestroy((mgr_data -> aff_solver)[i]);
         }
      }
      if (mgr_data -> fsolver_mode == 2)
      {
         if ((mgr_data -> aff_solver)[0])
         {
            nalu_hypre_BoomerAMGDestroy((mgr_data -> aff_solver)[0]);
         }
      }
      nalu_hypre_TFree(mgr_data -> aff_solver, NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> aff_solver) = NULL;
   }

   if (mgr_data -> level_diaginv)
   {
      for (i = 0; i < (num_coarse_levels); i++)
      {
         nalu_hypre_TFree((mgr_data -> level_diaginv)[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(mgr_data -> level_diaginv, NALU_HYPRE_MEMORY_HOST);
   }

   if (mgr_data -> frelax_diaginv)
   {
      for (i = 0; i < (num_coarse_levels); i++)
      {
         nalu_hypre_TFree((mgr_data -> frelax_diaginv)[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(mgr_data -> frelax_diaginv, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree((mgr_data -> F_array), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree((mgr_data -> U_array), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree((mgr_data -> A_array), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree((mgr_data -> B_array), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree((mgr_data -> B_FF_array), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree((mgr_data -> P_array), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree((mgr_data -> RT_array), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree((mgr_data -> CF_marker_array), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree((mgr_data -> reserved_Cpoint_local_indexes), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mgr_data -> restrict_type, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mgr_data -> interp_type, NALU_HYPRE_MEMORY_HOST);
   /* Frelax_type */
   nalu_hypre_TFree(mgr_data -> Frelax_type, NALU_HYPRE_MEMORY_HOST);
   /* Frelax_method */
   nalu_hypre_TFree(mgr_data -> Frelax_method, NALU_HYPRE_MEMORY_HOST);
   /* Frelax_num_functions */
   nalu_hypre_TFree(mgr_data -> Frelax_num_functions, NALU_HYPRE_MEMORY_HOST);

   /* data for V-cycle F-relaxation */
   if ((mgr_data -> VcycleRelaxVtemp))
   {
      nalu_hypre_ParVectorDestroy( (mgr_data -> VcycleRelaxVtemp) );
      (mgr_data -> VcycleRelaxVtemp) = NULL;
   }
   if ((mgr_data -> VcycleRelaxZtemp))
   {
      nalu_hypre_ParVectorDestroy( (mgr_data -> VcycleRelaxZtemp) );
      (mgr_data -> VcycleRelaxZtemp) = NULL;
   }
   if (mgr_data -> FrelaxVcycleData)
   {
      for (i = 0; i < num_coarse_levels; i++)
      {
         nalu_hypre_MGRDestroyFrelaxVcycleData((mgr_data -> FrelaxVcycleData)[i]);
      }
      nalu_hypre_TFree(mgr_data -> FrelaxVcycleData, NALU_HYPRE_MEMORY_HOST);
   }
   /* data for reserved coarse nodes */
   nalu_hypre_TFree(mgr_data -> reserved_coarse_indexes, NALU_HYPRE_MEMORY_HOST);
   /* index array for setting Cpoints by global block */
   if ((mgr_data -> set_c_points_method) == 1)
   {
      nalu_hypre_TFree(mgr_data -> idx_array, NALU_HYPRE_MEMORY_HOST);
   }
   /* array for setting option to use non-Galerkin coarse grid */
   nalu_hypre_TFree(mgr_data -> mgr_coarse_grid_method, NALU_HYPRE_MEMORY_HOST);
   /* coarse level matrix - RAP */
   if ((mgr_data -> RAP))
   {
      nalu_hypre_ParCSRMatrixDestroy((mgr_data -> RAP));
   }

   if ((mgr_data -> level_smoother) != NULL)
   {
      for (i = 0; i < num_coarse_levels; i++)
      {
         if ((mgr_data -> level_smooth_iters)[i] > 0)
         {
            if ((mgr_data -> level_smooth_type)[i] == 8)
            {
               NALU_HYPRE_EuclidDestroy((mgr_data -> level_smoother)[i]);
            }
            else if ((mgr_data -> level_smooth_type)[i] == 16)
            {
               NALU_HYPRE_ILUDestroy((mgr_data -> level_smoother)[i]);
            }
         }
      }
      nalu_hypre_TFree(mgr_data -> level_smoother, NALU_HYPRE_MEMORY_HOST);
   }

   /* free level data */
   nalu_hypre_TFree(mgr_data -> blk_size, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mgr_data -> level_smooth_type, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mgr_data -> level_smooth_iters, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(mgr_data -> num_relax_sweeps, NALU_HYPRE_MEMORY_HOST);

   if (mgr_data -> GSElimData)
   {
      for (i = 0; i < num_coarse_levels; i++)
      {
         if ((mgr_data -> GSElimData)[i])
         {
            nalu_hypre_MGRDestroyGSElimData((mgr_data -> GSElimData)[i]);
            (mgr_data -> GSElimData)[i] = NULL;
         }
      }
      nalu_hypre_TFree(mgr_data -> GSElimData, NALU_HYPRE_MEMORY_HOST);
   }

   /* mgr data */
   nalu_hypre_TFree(mgr_data, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/* create data for Gaussian Elim. for F-relaxation */
void *
nalu_hypre_MGRCreateGSElimData( void )
{
   nalu_hypre_ParAMGData  *gsdata = nalu_hypre_CTAlloc(nalu_hypre_ParAMGData,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParAMGDataGSSetup(gsdata) = 0;
   nalu_hypre_ParAMGDataAMat(gsdata) = NULL;
   nalu_hypre_ParAMGDataAInv(gsdata) = NULL;
   nalu_hypre_ParAMGDataBVec(gsdata) = NULL;
   nalu_hypre_ParAMGDataCommInfo(gsdata) = NULL;
   nalu_hypre_ParAMGDataNewComm(gsdata) = nalu_hypre_MPI_COMM_NULL;

   return (void *) gsdata;
}

/* Destroy data for Gaussian Elim. for F-relaxation */
NALU_HYPRE_Int
nalu_hypre_MGRDestroyGSElimData( void *data )
{
   nalu_hypre_ParAMGData * gsdata = (nalu_hypre_ParAMGData*) data;
   MPI_Comm new_comm = nalu_hypre_ParAMGDataNewComm(gsdata);

   if (nalu_hypre_ParAMGDataAMat(gsdata)) { nalu_hypre_TFree(nalu_hypre_ParAMGDataAMat(gsdata), NALU_HYPRE_MEMORY_HOST); }
   if (nalu_hypre_ParAMGDataAInv(gsdata)) { nalu_hypre_TFree(nalu_hypre_ParAMGDataAInv(gsdata), NALU_HYPRE_MEMORY_HOST); }
   if (nalu_hypre_ParAMGDataBVec(gsdata)) { nalu_hypre_TFree(nalu_hypre_ParAMGDataBVec(gsdata), NALU_HYPRE_MEMORY_HOST); }
   if (nalu_hypre_ParAMGDataCommInfo(gsdata)) { nalu_hypre_TFree(nalu_hypre_ParAMGDataCommInfo(gsdata), NALU_HYPRE_MEMORY_HOST); }

   if (new_comm != nalu_hypre_MPI_COMM_NULL)
   {
      nalu_hypre_MPI_Comm_free (&new_comm);
   }

   nalu_hypre_TFree(gsdata, NALU_HYPRE_MEMORY_HOST);
   return nalu_hypre_error_flag;
}

/* Create data for V-cycle F-relaxtion */
void *
nalu_hypre_MGRCreateFrelaxVcycleData( void )
{
   nalu_hypre_ParAMGData  *vdata = nalu_hypre_CTAlloc(nalu_hypre_ParAMGData,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParAMGDataAArray(vdata) = NULL;
   nalu_hypre_ParAMGDataPArray(vdata) = NULL;
   nalu_hypre_ParAMGDataFArray(vdata) = NULL;
   nalu_hypre_ParAMGDataCFMarkerArray(vdata) = NULL;
   nalu_hypre_ParAMGDataVtemp(vdata)  = NULL;
   //   nalu_hypre_ParAMGDataAMat(vdata)  = NULL;
   //   nalu_hypre_ParAMGDataBVec(vdata)  = NULL;
   nalu_hypre_ParAMGDataZtemp(vdata)  = NULL;
   //   nalu_hypre_ParAMGDataCommInfo(vdata) = NULL;
   nalu_hypre_ParAMGDataUArray(vdata) = NULL;
   nalu_hypre_ParAMGDataNewComm(vdata) = nalu_hypre_MPI_COMM_NULL;
   nalu_hypre_ParAMGDataNumLevels(vdata) = 0;
   nalu_hypre_ParAMGDataMaxLevels(vdata) = 10;
   nalu_hypre_ParAMGDataNumFunctions(vdata) = 1;
   nalu_hypre_ParAMGDataSCommPkgSwitch(vdata) = 1.0;
   nalu_hypre_ParAMGDataRelaxOrder(vdata) = 1;
   nalu_hypre_ParAMGDataMaxCoarseSize(vdata) = 9;
   nalu_hypre_ParAMGDataMinCoarseSize(vdata) = 0;
   nalu_hypre_ParAMGDataUserCoarseRelaxType(vdata) = 9;

   /* Gaussian Elim data */
   nalu_hypre_ParAMGDataGSSetup(vdata) = 0;
   nalu_hypre_ParAMGDataAMat(vdata) = NULL;
   nalu_hypre_ParAMGDataAInv(vdata) = NULL;
   nalu_hypre_ParAMGDataBVec(vdata) = NULL;
   nalu_hypre_ParAMGDataCommInfo(vdata) = NULL;

   return (void *) vdata;
}

/* Destroy data for V-cycle F-relaxation */
NALU_HYPRE_Int
nalu_hypre_MGRDestroyFrelaxVcycleData( void *data )
{
   nalu_hypre_ParAMGData * vdata = (nalu_hypre_ParAMGData*) data;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int num_levels = nalu_hypre_ParAMGDataNumLevels(vdata);
   MPI_Comm new_comm = nalu_hypre_ParAMGDataNewComm(vdata);

   nalu_hypre_TFree(nalu_hypre_ParAMGDataDofFuncArray(vdata)[0], NALU_HYPRE_MEMORY_HOST);
   for (i = 1; i < num_levels + 1; i++)
   {
      if (nalu_hypre_ParAMGDataAArray(vdata)[i])
      {
         nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParAMGDataAArray(vdata)[i]);
      }

      if (nalu_hypre_ParAMGDataPArray(vdata)[i - 1])
      {
         nalu_hypre_ParCSRMatrixDestroy(nalu_hypre_ParAMGDataPArray(vdata)[i - 1]);
      }

      nalu_hypre_IntArrayDestroy(nalu_hypre_ParAMGDataCFMarkerArray(vdata)[i - 1]);
      nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataFArray(vdata)[i]);
      nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataUArray(vdata)[i]);
      nalu_hypre_TFree(nalu_hypre_ParAMGDataDofFuncArray(vdata)[i], NALU_HYPRE_MEMORY_HOST);
   }

   if (num_levels < 1)
   {
      nalu_hypre_IntArrayDestroy(nalu_hypre_ParAMGDataCFMarkerArray(vdata)[0]);
   }

   /* Points to VcycleRelaxVtemp of mgr_data, which is already destroyed */
   //nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataVtemp(vdata));
   nalu_hypre_TFree(nalu_hypre_ParAMGDataFArray(vdata), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_ParAMGDataUArray(vdata), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_ParAMGDataAArray(vdata), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_ParAMGDataPArray(vdata), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_ParAMGDataCFMarkerArray(vdata), NALU_HYPRE_MEMORY_HOST);
   //nalu_hypre_TFree(nalu_hypre_ParAMGDataGridRelaxType(vdata), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_ParAMGDataDofFuncArray(vdata), NALU_HYPRE_MEMORY_HOST);

   /* Points to VcycleRelaxZtemp of mgr_data, which is already destroyed */
   /*
     if (nalu_hypre_ParAMGDataZtemp(vdata))
         nalu_hypre_ParVectorDestroy(nalu_hypre_ParAMGDataZtemp(vdata));
   */

   if (nalu_hypre_ParAMGDataAMat(vdata)) { nalu_hypre_TFree(nalu_hypre_ParAMGDataAMat(vdata), NALU_HYPRE_MEMORY_HOST); }
   if (nalu_hypre_ParAMGDataAInv(vdata)) { nalu_hypre_TFree(nalu_hypre_ParAMGDataAInv(vdata), NALU_HYPRE_MEMORY_HOST); }
   if (nalu_hypre_ParAMGDataBVec(vdata)) { nalu_hypre_TFree(nalu_hypre_ParAMGDataBVec(vdata), NALU_HYPRE_MEMORY_HOST); }
   if (nalu_hypre_ParAMGDataCommInfo(vdata)) { nalu_hypre_TFree(nalu_hypre_ParAMGDataCommInfo(vdata), NALU_HYPRE_MEMORY_HOST); }

   if (new_comm != nalu_hypre_MPI_COMM_NULL)
   {
      nalu_hypre_MPI_Comm_free (&new_comm);
   }
   nalu_hypre_TFree(vdata, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/* Set C-point variables for each reduction level */
/* Currently not implemented */
NALU_HYPRE_Int
nalu_hypre_MGRSetReductionLevelCpoints( void      *mgr_vdata,
                                   NALU_HYPRE_Int  nlevels,
                                   NALU_HYPRE_Int *num_coarse_points,
                                   NALU_HYPRE_Int  **level_coarse_indexes)
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> num_coarse_levels) = nlevels;
   (mgr_data -> num_coarse_per_level) = num_coarse_points;
   (mgr_data -> level_coarse_indexes) = level_coarse_indexes;
   return nalu_hypre_error_flag;
}

/* Initialize some data */
/* Set whether non-coarse points on each level should be explicitly tagged as F-points */
NALU_HYPRE_Int
nalu_hypre_MGRSetNonCpointsToFpoints( void      *mgr_vdata, NALU_HYPRE_Int nonCptToFptFlag)
{
   nalu_hypre_ParMGRData *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> set_non_Cpoints_to_F) = nonCptToFptFlag;

   return nalu_hypre_error_flag;
}

/* Set whether the reserved C points are reduced before the coarse grid solve */
NALU_HYPRE_Int
nalu_hypre_MGRSetReservedCpointsLevelToKeep(void *mgr_vdata, NALU_HYPRE_Int level)
{
   nalu_hypre_ParMGRData *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> lvl_to_keep_cpoints) = level;

   return nalu_hypre_error_flag;
}

/* Set Cpoints by contiguous blocks, i.e. p1, p2, ..., pn, s1, s2, ..., sn, ... */
NALU_HYPRE_Int
nalu_hypre_MGRSetCpointsByContiguousBlock( void  *mgr_vdata,
                                      NALU_HYPRE_Int  block_size,
                                      NALU_HYPRE_Int  max_num_levels,
                                      NALU_HYPRE_BigInt  *begin_idx_array,
                                      NALU_HYPRE_Int  *block_num_coarse_points,
                                      NALU_HYPRE_Int  **block_coarse_indexes)
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i;
   if ((mgr_data -> idx_array) != NULL)
   {
      nalu_hypre_TFree(mgr_data -> idx_array, NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> idx_array) = NULL;
   }
   NALU_HYPRE_BigInt *index_array = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, block_size, NALU_HYPRE_MEMORY_HOST);
   if (begin_idx_array != NULL)
   {
      for (i = 0; i < block_size; i++)
      {
         index_array[i] = *(begin_idx_array + i);
      }
   }
   nalu_hypre_MGRSetCpointsByBlock(mgr_data, block_size, max_num_levels, block_num_coarse_points,
                              block_coarse_indexes);
   (mgr_data -> idx_array) = index_array;
   (mgr_data -> set_c_points_method) = 1;
   return nalu_hypre_error_flag;
}

/* Initialize/ set local block data information */
NALU_HYPRE_Int
nalu_hypre_MGRSetCpointsByBlock( void      *mgr_vdata,
                            NALU_HYPRE_Int  block_size,
                            NALU_HYPRE_Int  max_num_levels,
                            NALU_HYPRE_Int  *block_num_coarse_points,
                            NALU_HYPRE_Int  **block_coarse_indexes)
{
   NALU_HYPRE_Int  i, j;
   NALU_HYPRE_Int  **block_cf_marker = NULL;
   NALU_HYPRE_Int *block_num_coarse_indexes = NULL;

   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;

   /* free block cf_marker data if not previously destroyed */
   if ((mgr_data -> block_cf_marker) != NULL)
   {
      for (i = 0; i < (mgr_data -> max_num_coarse_levels); i++)
      {
         if ((mgr_data -> block_cf_marker)[i])
         {
            nalu_hypre_TFree((mgr_data -> block_cf_marker)[i], NALU_HYPRE_MEMORY_HOST);
            (mgr_data -> block_cf_marker)[i] = NULL;
         }
      }
      nalu_hypre_TFree(mgr_data -> block_cf_marker, NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> block_cf_marker) = NULL;
   }
   if ((mgr_data -> block_num_coarse_indexes))
   {
      nalu_hypre_TFree((mgr_data -> block_num_coarse_indexes), NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> block_num_coarse_indexes) = NULL;
   }

   /* store block cf_marker */
   block_cf_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int *, max_num_levels, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_levels; i++)
   {
      block_cf_marker[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, block_size, NALU_HYPRE_MEMORY_HOST);
      memset(block_cf_marker[i], FMRK, block_size * sizeof(NALU_HYPRE_Int));
   }
   for (i = 0; i < max_num_levels; i++)
   {
      for (j = 0; j < block_num_coarse_points[i]; j++)
      {
         (block_cf_marker[i])[block_coarse_indexes[i][j]] = CMRK;
      }
   }

   /* store block_num_coarse_points */
   if (max_num_levels > 0)
   {
      block_num_coarse_indexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_num_levels, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < max_num_levels; i++)
      {
         block_num_coarse_indexes[i] = block_num_coarse_points[i];
      }
   }
   /* set block data */
   (mgr_data -> max_num_coarse_levels) = max_num_levels;
   (mgr_data -> block_size) = block_size;
   (mgr_data -> block_num_coarse_indexes) = block_num_coarse_indexes;
   (mgr_data -> block_cf_marker) = block_cf_marker;
   (mgr_data -> set_c_points_method) = 0;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_MGRSetCpointsByPointMarkerArray( void      *mgr_vdata,
                                       NALU_HYPRE_Int  block_size,
                                       NALU_HYPRE_Int  max_num_levels,
                                       NALU_HYPRE_Int  *lvl_num_coarse_points,
                                       NALU_HYPRE_Int  **lvl_coarse_indexes,
                                       NALU_HYPRE_Int  *point_marker_array)
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int  i, j;
   NALU_HYPRE_Int  **block_cf_marker = NULL;
   NALU_HYPRE_Int *block_num_coarse_indexes = NULL;

   /* free block cf_marker data if not previously destroyed */
   if ((mgr_data -> block_cf_marker) != NULL)
   {
      for (i = 0; i < (mgr_data -> max_num_coarse_levels); i++)
      {
         if ((mgr_data -> block_cf_marker)[i])
         {
            nalu_hypre_TFree((mgr_data -> block_cf_marker)[i], NALU_HYPRE_MEMORY_HOST);
            (mgr_data -> block_cf_marker)[i] = NULL;
         }
      }
      nalu_hypre_TFree(mgr_data -> block_cf_marker, NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> block_cf_marker) = NULL;
   }
   if ((mgr_data -> block_num_coarse_indexes))
   {
      nalu_hypre_TFree((mgr_data -> block_num_coarse_indexes), NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> block_num_coarse_indexes) = NULL;
   }

   /* store block cf_marker */
   block_cf_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int *, max_num_levels, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_levels; i++)
   {
      block_cf_marker[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, block_size, NALU_HYPRE_MEMORY_HOST);
      memset(block_cf_marker[i], FMRK, block_size * sizeof(NALU_HYPRE_Int));
   }
   for (i = 0; i < max_num_levels; i++)
   {
      for (j = 0; j < lvl_num_coarse_points[i]; j++)
      {
         block_cf_marker[i][j] = lvl_coarse_indexes[i][j];
      }
   }

   /* store block_num_coarse_points */
   if (max_num_levels > 0)
   {
      block_num_coarse_indexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_num_levels, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < max_num_levels; i++)
      {
         block_num_coarse_indexes[i] = lvl_num_coarse_points[i];
      }
   }
   /* set block data */
   (mgr_data -> max_num_coarse_levels) = max_num_levels;
   (mgr_data -> block_size) = block_size;
   (mgr_data -> block_num_coarse_indexes) = block_num_coarse_indexes;
   (mgr_data -> block_cf_marker) = block_cf_marker;
   (mgr_data -> point_marker_array) = point_marker_array;
   (mgr_data -> set_c_points_method) = 2;

   return nalu_hypre_error_flag;
}

/*Set number of points that remain part of the coarse grid throughout the hierarchy */
NALU_HYPRE_Int
nalu_hypre_MGRSetReservedCoarseNodes(void      *mgr_vdata,
                                NALU_HYPRE_Int reserved_coarse_size,
                                NALU_HYPRE_BigInt *reserved_cpt_index)
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_BigInt *reserved_coarse_indexes = NULL;
   NALU_HYPRE_Int i;

   if (!mgr_data)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Warning! MGR object empty!\n");
      return nalu_hypre_error_flag;
   }

   if (reserved_coarse_size < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }
   /* free data not previously destroyed */
   if ((mgr_data -> reserved_coarse_indexes))
   {
      nalu_hypre_TFree((mgr_data -> reserved_coarse_indexes), NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> reserved_coarse_indexes) = NULL;
   }

   /* set reserved coarse nodes */
   if (reserved_coarse_size > 0)
   {
      reserved_coarse_indexes = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  reserved_coarse_size, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < reserved_coarse_size; i++)
      {
         reserved_coarse_indexes[i] = reserved_cpt_index[i];
      }
   }
   (mgr_data -> reserved_coarse_size) = reserved_coarse_size;
   (mgr_data -> reserved_coarse_indexes) = reserved_coarse_indexes;

   return nalu_hypre_error_flag;
}

/* Set CF marker array */
NALU_HYPRE_Int
nalu_hypre_MGRCoarsen(nalu_hypre_ParCSRMatrix *S,
                 nalu_hypre_ParCSRMatrix *A,
                 NALU_HYPRE_Int fixed_coarse_size,
                 NALU_HYPRE_Int *fixed_coarse_indexes,
                 NALU_HYPRE_Int debug_flag,
                 nalu_hypre_IntArray **CF_marker_ptr,
                 NALU_HYPRE_Int cflag)
{
   NALU_HYPRE_Int   *CF_marker = NULL;
   NALU_HYPRE_Int *cindexes = fixed_coarse_indexes;
   NALU_HYPRE_Int    i, row, nc;
   NALU_HYPRE_Int nloc =  nalu_hypre_ParCSRMatrixNumRows(A);
   NALU_HYPRE_MemoryLocation memory_location;

   /* If this is the last level, coarsen onto fixed coarse set */
   if (cflag)
   {
      if (*CF_marker_ptr != NULL)
      {
         nalu_hypre_IntArrayDestroy(*CF_marker_ptr);
      }
      *CF_marker_ptr = nalu_hypre_IntArrayCreate(nloc);
      nalu_hypre_IntArrayInitialize(*CF_marker_ptr);
      nalu_hypre_IntArraySetConstantValues(*CF_marker_ptr, FMRK);
      memory_location = nalu_hypre_IntArrayMemoryLocation(*CF_marker_ptr);

      if (nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_DEVICE)
      {
         nalu_hypre_IntArrayMigrate(*CF_marker_ptr, NALU_HYPRE_MEMORY_HOST);
      }
      CF_marker = nalu_hypre_IntArrayData(*CF_marker_ptr);

      /* first mark fixed coarse set */
      nc = fixed_coarse_size;
      for (i = 0; i < nc; i++)
      {
         CF_marker[cindexes[i]] = CMRK;
      }

      if (nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_DEVICE)
      {
         nalu_hypre_IntArrayMigrate(*CF_marker_ptr, NALU_HYPRE_MEMORY_DEVICE);
      }
   }
   else
   {
      /* First coarsen to get initial CF splitting.
       * This is then followed by updating the CF marker to pass
       * coarse information to the next levels. NOTE: It may be
       * convenient to implement this way (allows the use of multiple
       * coarsening strategies without changing too much code),
       * but not necessarily the best option, compared to initializing
       * CF_marker first and then coarsening on subgraph which excludes
       * the initialized coarse nodes.
      */
      nalu_hypre_BoomerAMGCoarsen(S, A, 0, debug_flag, CF_marker_ptr);
      CF_marker = nalu_hypre_IntArrayData(*CF_marker_ptr);

      /* Update CF_marker to correct Cpoints marked as Fpoints. */
      nc = fixed_coarse_size;
      for (i = 0; i < nc; i++)
      {
         CF_marker[cindexes[i]] = CMRK;
      }
      /* set F-points to FMRK. This is necessary since the different coarsening schemes differentiate
       * between type of F-points (example Ruge coarsening). We do not need that distinction here.
      */
      for (row = 0; row < nloc; row++)
      {
         if (CF_marker[row] == CMRK) { continue; }
         CF_marker[row] = FMRK;
      }
#if 0
      /* IMPORTANT: Update coarse_indexes array to define the positions of the fixed coarse points
       * in the next level.
       */
      nc = 0;
      index_i = 0;
      for (row = 0; row < nloc; row++)
      {
         /* loop through new c-points */
         if (CF_marker[row] == CMRK) { nc++; }
         else if (CF_marker[row] == S_CMRK)
         {
            /* previously marked c-point is part of fixed coarse set. Track its current local index */
            cindexes[index_i++] = nc;
            /* reset c-point from S_CMRK to CMRK */
            cf_marker[row] = CMRK;
            nc++;
         }
         /* set F-points to FMRK. This is necessary since the different coarsening schemes differentiate
          * between type of F-points (example Ruge coarsening). We do not need that distinction here.
          */
         else
         {
            CF_marker[row] = FMRK;
         }
      }
      /* check if this should be last level */
      if ( nc == fixed_coarse_size)
      {
         last_level = 1;
      }
      //printf(" nc = %d and fixed coarse size = %d \n", nc, fixed_coarse_size);
#endif
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRBuildPFromWp
 *
 * Build prolongation matrix from the Nf x Nc matrix
 *
 * TODOs:
 *   1) Remove debug_flag
 *   2) Move this function to par_interp.c ?
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRBuildPFromWp( nalu_hypre_ParCSRMatrix   *A,
                       nalu_hypre_ParCSRMatrix   *Wp,
                       NALU_HYPRE_Int            *CF_marker,
                       NALU_HYPRE_Int            debug_flag,
                       nalu_hypre_ParCSRMatrix   **P_ptr)
{
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_MGRBuildPFromWpDevice(A, Wp, CF_marker, P_ptr);
   }
   else
#endif
   {
      nalu_hypre_MGRBuildPFromWpHost(A, Wp, CF_marker, debug_flag, P_ptr);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRBuildPFromWpHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRBuildPFromWpHost( nalu_hypre_ParCSRMatrix   *A,
                           nalu_hypre_ParCSRMatrix   *Wp,
                           NALU_HYPRE_Int            *CF_marker,
                           NALU_HYPRE_Int            debug_flag,
                           nalu_hypre_ParCSRMatrix   **P_ptr)
{
   MPI_Comm          comm = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_MemoryLocation memory_location_P = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   nalu_hypre_ParCSRMatrix    *P;

   nalu_hypre_CSRMatrix *P_diag = NULL;
   nalu_hypre_CSRMatrix *P_offd = NULL;
   nalu_hypre_CSRMatrix *Wp_diag, *Wp_offd;

   NALU_HYPRE_Real      *P_diag_data, *Wp_diag_data;
   NALU_HYPRE_Int       *P_diag_i, *Wp_diag_i;
   NALU_HYPRE_Int       *P_diag_j, *Wp_diag_j;
   NALU_HYPRE_Real      *P_offd_data, *Wp_offd_data;
   NALU_HYPRE_Int       *P_offd_i, *Wp_offd_i;
   NALU_HYPRE_Int       *P_offd_j, *Wp_offd_j;

   NALU_HYPRE_Int        P_num_rows, P_diag_size, P_offd_size;

   NALU_HYPRE_Int        jj_counter, jj_counter_offd;

   NALU_HYPRE_Int        start_indexing = 0; /* start indexing for P_data at 0 */

   NALU_HYPRE_Int        i, jj;
   NALU_HYPRE_Int        row_Wp, coarse_counter;

   NALU_HYPRE_Real       one  = 1.0;

   NALU_HYPRE_Int        my_id;
   NALU_HYPRE_Int        num_procs;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   //num_threads = nalu_hypre_NumThreads();
   // Temporary fix, disable threading
   // TODO: enable threading
   P_num_rows = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));

   Wp_diag = nalu_hypre_ParCSRMatrixDiag(Wp);
   Wp_offd = nalu_hypre_ParCSRMatrixOffd(Wp);
   Wp_diag_i = nalu_hypre_CSRMatrixI(Wp_diag);
   Wp_diag_j = nalu_hypre_CSRMatrixJ(Wp_diag);
   Wp_diag_data = nalu_hypre_CSRMatrixData(Wp_diag);
   Wp_offd_i = nalu_hypre_CSRMatrixI(Wp_offd);
   Wp_offd_j = nalu_hypre_CSRMatrixJ(Wp_offd);
   Wp_offd_data = nalu_hypre_CSRMatrixData(Wp_offd);

   /*-----------------------------------------------------------------------
   *  Intialize counters and allocate mapping vector.
   *-----------------------------------------------------------------------*/
   P_diag_size = nalu_hypre_CSRMatrixNumNonzeros(Wp_diag) + nalu_hypre_CSRMatrixNumCols(Wp_diag);

   P_diag_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_num_rows + 1, memory_location_P);
   P_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_diag_size, memory_location_P);
   P_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  P_diag_size, memory_location_P);
   P_diag_i[P_num_rows] = P_diag_size;

   P_offd_size = nalu_hypre_CSRMatrixNumNonzeros(Wp_offd);

   P_offd_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_num_rows + 1, memory_location_P);
   P_offd_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_offd_size, memory_location_P);
   P_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  P_offd_size, memory_location_P);
   P_offd_i[P_num_rows] = P_offd_size;

   /*-----------------------------------------------------------------------
   *  Intialize some stuff.
   *-----------------------------------------------------------------------*/
   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   row_Wp = 0;
   coarse_counter = 0;
   for (i = 0; i < P_num_rows; i++)
   {
      /*--------------------------------------------------------------------
      *  If i is a c-point, interpolation is the identity.
      *--------------------------------------------------------------------*/
      if (CF_marker[i] >= 0)
      {
         P_diag_i[i] = jj_counter;
         P_diag_j[jj_counter]    = coarse_counter;
         P_diag_data[jj_counter] = one;
         coarse_counter++;
         jj_counter++;
      }
      /*--------------------------------------------------------------------
      *  If i is an F-point, build interpolation.
      *--------------------------------------------------------------------*/
      else
      {
         /* Diagonal part of P */
         P_diag_i[i] = jj_counter;
         for (jj = Wp_diag_i[row_Wp]; jj < Wp_diag_i[row_Wp + 1]; jj++)
         {
            P_diag_j[jj_counter]    = Wp_diag_j[jj];
            P_diag_data[jj_counter] = - Wp_diag_data[jj];
            jj_counter++;
         }

         /* Off-Diagonal part of P */
         P_offd_i[i] = jj_counter_offd;
         if (num_procs > 1)
         {
            for (jj = Wp_offd_i[row_Wp]; jj < Wp_offd_i[row_Wp + 1]; jj++)
            {
               P_offd_j[jj_counter_offd]    = Wp_offd_j[jj];
               P_offd_data[jj_counter_offd] = - Wp_offd_data[jj];
               jj_counter_offd++;
            }
         }
         row_Wp++;
      }
      P_offd_i[i + 1] = jj_counter_offd;
   }
   P = nalu_hypre_ParCSRMatrixCreate(comm,
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                nalu_hypre_ParCSRMatrixGlobalNumCols(Wp),
                                nalu_hypre_ParCSRMatrixColStarts(A),
                                nalu_hypre_ParCSRMatrixColStarts(Wp),
                                nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(Wp)),
                                P_diag_size,
                                P_offd_size);

   P_diag = nalu_hypre_ParCSRMatrixDiag(P);
   nalu_hypre_CSRMatrixData(P_diag) = P_diag_data;
   nalu_hypre_CSRMatrixI(P_diag) = P_diag_i;
   nalu_hypre_CSRMatrixJ(P_diag) = P_diag_j;

   P_offd = nalu_hypre_ParCSRMatrixOffd(P);
   nalu_hypre_CSRMatrixData(P_offd) = P_offd_data;
   nalu_hypre_CSRMatrixI(P_offd) = P_offd_i;
   nalu_hypre_CSRMatrixJ(P_offd) = P_offd_j;
   //nalu_hypre_ParCSRMatrixOwnsRowStarts(P) = 0;
   //nalu_hypre_ParCSRMatrixOwnsColStarts(Wp) = 0;
   //nalu_hypre_ParCSRMatrixOwnsColStarts(P) = 1;

   nalu_hypre_ParCSRMatrixDeviceColMapOffd(P) = nalu_hypre_ParCSRMatrixDeviceColMapOffd(Wp);
   nalu_hypre_ParCSRMatrixColMapOffd(P)       = nalu_hypre_ParCSRMatrixColMapOffd(Wp);
   //nalu_hypre_ParCSRMatrixDeviceColMapOffd(Wp) = NULL;
   //nalu_hypre_ParCSRMatrixColMapOffd(Wp)       = NULL;

   nalu_hypre_ParCSRMatrixNumNonzeros(P)  = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(P)) +
                                       nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(P));
   nalu_hypre_ParCSRMatrixDNumNonzeros(P) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(P);

   nalu_hypre_MatvecCommPkgCreate(P);
   *P_ptr = P;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRBuildBlockJacobiWp
 *
 * TODO: Move this to nalu_hypre_MGRBuildPBlockJacobi? (VPM)
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRBuildBlockJacobiWp( nalu_hypre_ParCSRMatrix   *A_FF,
                             nalu_hypre_ParCSRMatrix   *A_FC,
                             NALU_HYPRE_Int             blk_size,
                             NALU_HYPRE_Int            *CF_marker,
                             NALU_HYPRE_BigInt         *cpts_starts,
                             nalu_hypre_ParCSRMatrix  **Wp_ptr )
{
   nalu_hypre_ParCSRMatrix   *A_FF_inv;
   nalu_hypre_ParCSRMatrix   *Wp;

   /* Build A_FF_inv */
   nalu_hypre_ParCSRMatrixBlockDiagMatrix(A_FF, blk_size, -1, NULL, 1, &A_FF_inv);

   /* Compute Wp = A_FF_inv * A_FC */
   Wp = nalu_hypre_ParCSRMatMat(A_FF_inv, A_FC);

   /* Free memory */
   nalu_hypre_ParCSRMatrixDestroy(A_FF_inv);

   /* Set output pointer */
   *Wp_ptr = Wp;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRBuildPBlockJacobi
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRBuildPBlockJacobi( nalu_hypre_ParCSRMatrix   *A,
                            nalu_hypre_ParCSRMatrix   *A_FF,
                            nalu_hypre_ParCSRMatrix   *A_FC,
                            nalu_hypre_ParCSRMatrix   *Wp,
                            NALU_HYPRE_Int             blk_size,
                            NALU_HYPRE_Int            *CF_marker,
                            NALU_HYPRE_BigInt         *cpts_starts,
                            NALU_HYPRE_Int             debug_flag,
                            nalu_hypre_ParCSRMatrix  **P_ptr)
{
   nalu_hypre_ParCSRMatrix   *Wp_tmp;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   if (Wp == NULL)
   {
      nalu_hypre_MGRBuildBlockJacobiWp(A_FF, A_FC, blk_size, CF_marker, cpts_starts, &Wp_tmp);
      nalu_hypre_MGRBuildPFromWp(A, Wp_tmp, CF_marker, debug_flag, P_ptr);

      nalu_hypre_ParCSRMatrixDeviceColMapOffd(Wp_tmp) = NULL;
      nalu_hypre_ParCSRMatrixColMapOffd(Wp_tmp)       = NULL;

      nalu_hypre_ParCSRMatrixDestroy(Wp_tmp);
   }
   else
   {
      nalu_hypre_MGRBuildPFromWp(A, Wp, CF_marker, debug_flag, P_ptr);
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ExtendWtoPHost
 *
 * TODO: move this to par_interp.c?
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ExtendWtoPHost(NALU_HYPRE_Int      P_nr_of_rows,
                     NALU_HYPRE_Int     *CF_marker,
                     NALU_HYPRE_Int     *W_diag_i,
                     NALU_HYPRE_Int     *W_diag_j,
                     NALU_HYPRE_Complex *W_diag_data,
                     NALU_HYPRE_Int     *P_diag_i,
                     NALU_HYPRE_Int     *P_diag_j,
                     NALU_HYPRE_Complex *P_diag_data,
                     NALU_HYPRE_Int     *W_offd_i,
                     NALU_HYPRE_Int     *P_offd_i )
{
   NALU_HYPRE_Int              jj_counter, jj_counter_offd;

   NALU_HYPRE_Int              start_indexing = 0; /* start indexing for P_data at 0 */

   NALU_HYPRE_Int             *fine_to_coarse = NULL;
   NALU_HYPRE_Int              coarse_counter;

   NALU_HYPRE_Int              i, jj;

   NALU_HYPRE_Real       one  = 1.0;

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   fine_to_coarse = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_nr_of_rows, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < P_nr_of_rows; i++) { fine_to_coarse[i] = -1; }

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

   NALU_HYPRE_Int row_counter = 0;
   coarse_counter = 0;
   for (i = 0; i < P_nr_of_rows; i++)
   {
      /*--------------------------------------------------------------------
       *  If i is a C-point, interpolation is the identity. Also set up
       *  mapping vector.
       *--------------------------------------------------------------------*/

      if (CF_marker[i] > 0)
      {
         fine_to_coarse[i] = coarse_counter;
         coarse_counter++;
      }
   }

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   row_counter = 0;
   for (i = 0; i < P_nr_of_rows; i++)
   {
      /*--------------------------------------------------------------------
       *  If i is a c-point, interpolation is the identity.
       *--------------------------------------------------------------------*/
      if (CF_marker[i] >= 0)
      {
         P_diag_i[i] = jj_counter;
         P_diag_j[jj_counter]    = fine_to_coarse[i];
         P_diag_data[jj_counter] = one;
         jj_counter++;
      }
      /*--------------------------------------------------------------------
       *  If i is an F-point, build interpolation.
       *--------------------------------------------------------------------*/
      else
      {
         /* Diagonal part of P */
         P_diag_i[i] = jj_counter;
         for (jj = W_diag_i[row_counter]; jj < W_diag_i[row_counter + 1]; jj++)
         {
            //P_marker[row_counter] = jj_counter;
            P_diag_j[jj_counter]    = W_diag_j[jj];
            P_diag_data[jj_counter] = W_diag_data[jj];
            jj_counter++;
         }

         /* Off-Diagonal part of P */
         P_offd_i[i] = jj_counter_offd;
         jj_counter_offd += W_offd_i[row_counter + 1] - W_offd_i[row_counter];

         row_counter++;
      }
      /* update off-diagonal row pointer */
      P_offd_i[i + 1] = jj_counter_offd;
   }
   P_diag_i[P_nr_of_rows] = jj_counter;

   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   return 0;
}

/* Interpolation for MGR - Adapted from BoomerAMGBuildInterp */
NALU_HYPRE_Int
nalu_hypre_MGRBuildPHost( nalu_hypre_ParCSRMatrix   *A,
                     NALU_HYPRE_Int            *CF_marker,
                     NALU_HYPRE_BigInt         *num_cpts_global,
                     NALU_HYPRE_Int             method,
                     nalu_hypre_ParCSRMatrix  **P_ptr)
{
   MPI_Comm            comm = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int           num_procs, my_id;
   NALU_HYPRE_Int           A_nr_of_rows = nalu_hypre_ParCSRMatrixNumRows(A);

   nalu_hypre_ParCSRMatrix *A_FF = NULL, *A_FC = NULL, *P = NULL;
   nalu_hypre_CSRMatrix    *W_diag = NULL, *W_offd = NULL;
   NALU_HYPRE_Int           P_diag_nnz, nfpoints;
   NALU_HYPRE_Int          *P_diag_i = NULL, *P_diag_j = NULL, *P_offd_i = NULL;
   NALU_HYPRE_Complex      *P_diag_data = NULL, *diag = NULL, *diag1 = NULL;
   NALU_HYPRE_BigInt        nC_global;
   NALU_HYPRE_Int       i;

   NALU_HYPRE_MemoryLocation memory_location_P = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   nfpoints = 0;
   for (i = 0; i < A_nr_of_rows; i++)
   {
      if (CF_marker[i] == -1)
      {
         nfpoints++;
      }
   }

   if (method > 0)
   {
      nalu_hypre_ParCSRMatrixGenerateFFFCHost(A, CF_marker, num_cpts_global, NULL, &A_FC, &A_FF);
      diag = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nfpoints, memory_location_P);
      if (method == 1)
      {
         // extract diag inverse sqrt
         //        nalu_hypre_CSRMatrixExtractDiagonalHost(nalu_hypre_ParCSRMatrixDiag(A_FF), diag, 3);

         // L1-Jacobi-type interpolation
         NALU_HYPRE_Complex scal = 1.0;
         diag1 = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nfpoints, memory_location_P);
         nalu_hypre_CSRMatrixExtractDiagonalHost(nalu_hypre_ParCSRMatrixDiag(A_FF), diag, 0);
         nalu_hypre_CSRMatrixComputeRowSumHost(nalu_hypre_ParCSRMatrixDiag(A_FF), NULL, NULL, diag1, 1, 1.0, "set");
         nalu_hypre_CSRMatrixComputeRowSumHost(nalu_hypre_ParCSRMatrixDiag(A_FC), NULL, NULL, diag1, 1, 1.0, "add");
         nalu_hypre_CSRMatrixComputeRowSumHost(nalu_hypre_ParCSRMatrixOffd(A_FF), NULL, NULL, diag1, 1, 1.0, "add");
         nalu_hypre_CSRMatrixComputeRowSumHost(nalu_hypre_ParCSRMatrixOffd(A_FC), NULL, NULL, diag1, 1, 1.0, "add");

         for (i = 0; i < nfpoints; i++)
         {
            NALU_HYPRE_Complex dsum = diag[i] + scal * (diag1[i] - nalu_hypre_cabs(diag[i]));
            diag[i] = 1. / dsum;
         }
         nalu_hypre_TFree(diag1, memory_location_P);
      }
      else if (method == 2)
      {
         // extract diag inverse
         nalu_hypre_CSRMatrixExtractDiagonalHost(nalu_hypre_ParCSRMatrixDiag(A_FF), diag, 2);
      }

      for (i = 0; i < nfpoints; i++)
      {
         diag[i] = -diag[i];
      }

      nalu_hypre_Vector *D_FF_inv = nalu_hypre_SeqVectorCreate(nfpoints);
      nalu_hypre_VectorData(D_FF_inv) = diag;
      nalu_hypre_SeqVectorInitialize_v2(D_FF_inv, memory_location_P);
      nalu_hypre_CSRMatrixDiagScale(nalu_hypre_ParCSRMatrixDiag(A_FC), D_FF_inv, NULL);
      nalu_hypre_CSRMatrixDiagScale(nalu_hypre_ParCSRMatrixOffd(A_FC), D_FF_inv, NULL);
      nalu_hypre_SeqVectorDestroy(D_FF_inv);
      W_diag = nalu_hypre_ParCSRMatrixDiag(A_FC);
      W_offd = nalu_hypre_ParCSRMatrixOffd(A_FC);
      nC_global = nalu_hypre_ParCSRMatrixGlobalNumCols(A_FC);
   }
   else
   {
      W_diag = nalu_hypre_CSRMatrixCreate(nfpoints, A_nr_of_rows - nfpoints, 0);
      W_offd = nalu_hypre_CSRMatrixCreate(nfpoints, 0, 0);
      nalu_hypre_CSRMatrixInitialize_v2(W_diag, 0, memory_location_P);
      nalu_hypre_CSRMatrixInitialize_v2(W_offd, 0, memory_location_P);

      if (my_id == (num_procs - 1))
      {
         nC_global = num_cpts_global[1];
      }
      nalu_hypre_MPI_Bcast(&nC_global, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }

   /* Construct P from matrix product W_diag */
   P_diag_nnz  = nalu_hypre_CSRMatrixNumNonzeros(W_diag) + nalu_hypre_CSRMatrixNumCols(W_diag);
   P_diag_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,     A_nr_of_rows + 1, memory_location_P);
   P_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,     P_diag_nnz,     memory_location_P);
   P_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, P_diag_nnz,     memory_location_P);
   P_offd_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,     A_nr_of_rows + 1, memory_location_P);

   /* Extend W data to P data */
   nalu_hypre_ExtendWtoPHost( A_nr_of_rows,
                         CF_marker,
                         nalu_hypre_CSRMatrixI(W_diag),
                         nalu_hypre_CSRMatrixJ(W_diag),
                         nalu_hypre_CSRMatrixData(W_diag),
                         P_diag_i,
                         P_diag_j,
                         P_diag_data,
                         nalu_hypre_CSRMatrixI(W_offd),
                         P_offd_i );

   // finalize P
   P = nalu_hypre_ParCSRMatrixCreate(nalu_hypre_ParCSRMatrixComm(A),
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                nC_global,
                                nalu_hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                nalu_hypre_CSRMatrixNumCols(W_offd),
                                P_diag_nnz,
                                nalu_hypre_CSRMatrixNumNonzeros(W_offd) );

   nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixDiag(P)) = memory_location_P;
   nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixOffd(P)) = memory_location_P;

   nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(P))    = nalu_hypre_CSRMatrixJ(W_offd);
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(P)) = nalu_hypre_CSRMatrixData(W_offd);
   nalu_hypre_CSRMatrixJ(W_offd)    = NULL;
   nalu_hypre_CSRMatrixData(W_offd) = NULL;

   if (method > 0)
   {
      nalu_hypre_ParCSRMatrixColMapOffd(P)    = nalu_hypre_ParCSRMatrixColMapOffd(A_FC);
      nalu_hypre_ParCSRMatrixColMapOffd(P)          = nalu_hypre_ParCSRMatrixColMapOffd(A_FC);
      nalu_hypre_ParCSRMatrixColMapOffd(A_FC) = NULL;
      nalu_hypre_ParCSRMatrixColMapOffd(A_FC)       = NULL;
      nalu_hypre_ParCSRMatrixNumNonzeros(P)         = nalu_hypre_ParCSRMatrixNumNonzeros(
                                                    A_FC) + nalu_hypre_ParCSRMatrixGlobalNumCols(A_FC);
   }
   else
   {
      nalu_hypre_ParCSRMatrixNumNonzeros(P) = nC_global;
   }
   nalu_hypre_ParCSRMatrixDNumNonzeros(P) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(P);

   nalu_hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   if (A_FF)
   {
      nalu_hypre_ParCSRMatrixDestroy(A_FF);
   }
   if (A_FC)
   {
      nalu_hypre_ParCSRMatrixDestroy(A_FC);
   }

   if (method <= 0)
   {
      nalu_hypre_CSRMatrixDestroy(W_diag);
      nalu_hypre_CSRMatrixDestroy(W_offd);
   }

   return nalu_hypre_error_flag;
}
/* Interpolation for MGR - Adapted from BoomerAMGBuildInterp */
NALU_HYPRE_Int
nalu_hypre_MGRBuildP( nalu_hypre_ParCSRMatrix   *A,
                 NALU_HYPRE_Int            *CF_marker,
                 NALU_HYPRE_BigInt         *num_cpts_global,
                 NALU_HYPRE_Int             method,
                 NALU_HYPRE_Int             debug_flag,
                 nalu_hypre_ParCSRMatrix  **P_ptr)
{
   MPI_Comm          comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;
   NALU_HYPRE_MemoryLocation memory_location_P = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);

   nalu_hypre_CSRMatrix *A_offd         = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real      *A_offd_data    = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Int        num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_Real      *a_diag;

   nalu_hypre_ParCSRMatrix    *P;
   NALU_HYPRE_BigInt    *col_map_offd_P;
   NALU_HYPRE_Int       *tmp_map_offd = NULL;

   NALU_HYPRE_Int       *CF_marker_offd = NULL;

   nalu_hypre_CSRMatrix *P_diag;
   nalu_hypre_CSRMatrix *P_offd;

   NALU_HYPRE_Real      *P_diag_data;
   NALU_HYPRE_Int       *P_diag_i;
   NALU_HYPRE_Int       *P_diag_j;
   NALU_HYPRE_Real      *P_offd_data;
   NALU_HYPRE_Int       *P_offd_i;
   NALU_HYPRE_Int       *P_offd_j;

   NALU_HYPRE_Int        P_diag_size, P_offd_size;

   NALU_HYPRE_Int       *P_marker, *P_marker_offd;

   NALU_HYPRE_Int        jj_counter, jj_counter_offd;
   NALU_HYPRE_Int       *jj_count, *jj_count_offd;
   //   NALU_HYPRE_Int              jj_begin_row,jj_begin_row_offd;
   //   NALU_HYPRE_Int              jj_end_row,jj_end_row_offd;

   NALU_HYPRE_Int        start_indexing = 0; /* start indexing for P_data at 0 */

   NALU_HYPRE_Int        n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);

   NALU_HYPRE_Int       *fine_to_coarse;
   //NALU_HYPRE_BigInt    *fine_to_coarse_offd;
   NALU_HYPRE_Int       *coarse_counter;
   NALU_HYPRE_Int        coarse_shift;
   NALU_HYPRE_BigInt     total_global_cpts;
   //NALU_HYPRE_BigInt     my_first_cpt;
   NALU_HYPRE_Int        num_cols_P_offd;

   NALU_HYPRE_Int        i, i1;
   NALU_HYPRE_Int        j, jl, jj;
   NALU_HYPRE_Int        start;

   NALU_HYPRE_Real       one  = 1.0;

   NALU_HYPRE_Int        my_id;
   NALU_HYPRE_Int        num_procs;
   NALU_HYPRE_Int        num_threads;
   NALU_HYPRE_Int        num_sends;
   NALU_HYPRE_Int        index;
   NALU_HYPRE_Int        ns, ne, size, rest;

   NALU_HYPRE_Int       *int_buf_data;

   NALU_HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   //num_threads = nalu_hypre_NumThreads();
   // Temporary fix, disable threading
   // TODO: enable threading
   num_threads = 1;

   //my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
   * Get the CF_marker data for the off-processor columns
   *-------------------------------------------------------------------*/

   if (debug_flag < 0)
   {
      debug_flag = -debug_flag;
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);

   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                            num_sends), NALU_HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         int_buf_data[index++] = CF_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }

   comm_handle = nalu_hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, CF_marker_offd);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
   *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
   *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
   *  Intialize counters and allocate mapping vector.
   *-----------------------------------------------------------------------*/

   coarse_counter = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_threads, NALU_HYPRE_MEMORY_HOST);
   jj_count = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_threads, NALU_HYPRE_MEMORY_HOST);
   jj_count_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_threads, NALU_HYPRE_MEMORY_HOST);

   fine_to_coarse = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine, NALU_HYPRE_MEMORY_HOST);
#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
   *  Loop over fine grid.
   *-----------------------------------------------------------------------*/

   /* RDF: this looks a little tricky, but doable */
#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,i1,jj,ns,ne,size,rest) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (j = 0; j < num_threads; j++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;

      if (j < rest)
      {
         ns = j * size + j;
         ne = (j + 1) * size + j + 1;
      }
      else
      {
         ns = j * size + rest;
         ne = (j + 1) * size + rest;
      }
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
          *  If i is a C-point, interpolation is the identity. Also set up
          *  mapping vector.
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            jj_count[j]++;
            fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j]++;
         }
         /*--------------------------------------------------------------------
          *  If i is an F-point, interpolation is the approximation of A_{ff}^{-1}A_{fc}
          *--------------------------------------------------------------------*/
         else
         {
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];
               if ((CF_marker[i1] >= 0) && (method > 0))
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];
                  if ((CF_marker_offd[i1] >= 0) && (method > 0))
                  {
                     jj_count_offd[j]++;
                  }
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < num_threads - 1; i++)
   {
      coarse_counter[i + 1] += coarse_counter[i];
      jj_count[i + 1] += jj_count[i];
      jj_count_offd[i + 1] += jj_count_offd[i];
   }
   i = num_threads - 1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine + 1, memory_location_P);
   P_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_diag_size, memory_location_P);
   P_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  P_diag_size, memory_location_P);

   P_diag_i[n_fine] = jj_counter;

   P_offd_size = jj_counter_offd;

   P_offd_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine + 1, memory_location_P);
   P_offd_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_offd_size, memory_location_P);
   P_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  P_offd_size, memory_location_P);

   /*-----------------------------------------------------------------------
   *  Intialize some stuff.
   *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
   *  Send and receive fine_to_coarse info.
   *-----------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   //fine_to_coarse_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);

#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,ns,ne,size,rest,coarse_shift) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (j = 0; j < num_threads; j++)
   {
      coarse_shift = 0;
      if (j > 0) { coarse_shift = coarse_counter[j - 1]; }
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (j < rest)
      {
         ns = j * size + j;
         ne = (j + 1) * size + j + 1;
      }
      else
      {
         ns = j * size + rest;
         ne = (j + 1) * size + rest;
      }
      for (i = ns; i < ne; i++)
      {
         fine_to_coarse[i] += coarse_shift;
      }
   }

   /*   index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            big_buf_data[index++]
               = fine_to_coarse[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]+ my_first_cpt;
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate( 21, comm_pkg, big_buf_data,
                                       fine_to_coarse_offd);

   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   */
   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   //for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;

   /*-----------------------------------------------------------------------
   *  Loop over fine grid points.
   *-----------------------------------------------------------------------*/
   a_diag = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  n_fine, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < n_fine; i++)
   {
      if (CF_marker[i] < 0)
      {
         for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
         {
            i1 = A_diag_j[jj];
            if ( i == i1 ) /* diagonal of A only */
            {
               a_diag[i] = 1.0 / A_diag_data[jj];
            }
         }
      }
   }

#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,jl,i1,jj,ns,ne,size,rest,P_marker,P_marker_offd,jj_counter,jj_counter_offd,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (jl = 0; jl < num_threads; jl++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (jl < rest)
      {
         ns = jl * size + jl;
         ne = (jl + 1) * size + jl + 1;
      }
      else
      {
         ns = jl * size + rest;
         ne = (jl + 1) * size + rest;
      }
      jj_counter = 0;
      if (jl > 0) { jj_counter = jj_count[jl - 1]; }
      jj_counter_offd = 0;
      if (jl > 0) { jj_counter_offd = jj_count_offd[jl - 1]; }
      P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine, NALU_HYPRE_MEMORY_HOST);
      if (num_cols_A_offd)
      {
         P_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
      }
      else
      {
         P_marker_offd = NULL;
      }

      for (i = 0; i < n_fine; i++)
      {
         P_marker[i] = -1;
      }
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker_offd[i] = -1;
      }
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
         *  If i is a c-point, interpolation is the identity.
         *--------------------------------------------------------------------*/
         if (CF_marker[i] >= 0)
         {
            P_diag_i[i] = jj_counter;
            P_diag_j[jj_counter]    = fine_to_coarse[i];
            P_diag_data[jj_counter] = one;
            jj_counter++;
         }
         /*--------------------------------------------------------------------
         *  If i is an F-point, build interpolation.
         *--------------------------------------------------------------------*/
         else
         {
            /* Diagonal part of P */
            P_diag_i[i] = jj_counter;
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if ((CF_marker[i1] >= 0) && (method > 0))
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  /*
                  if(method == 0)
                  {
                    P_diag_data[jj_counter] = 0.0;
                  }
                  */
                  if (method == 1)
                  {
                     P_diag_data[jj_counter] = - A_diag_data[jj];
                  }
                  else if (method == 2)
                  {
                     P_diag_data[jj_counter] = - A_diag_data[jj] * a_diag[i];
                  }
                  jj_counter++;
               }
            }

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];

                  /*-----------------------------------------------------------
                  * If neighbor i1 is a C-point, set column number in P_offd_j
                  * and initialize interpolation weight to zero.
                  *-----------------------------------------------------------*/

                  if ((CF_marker_offd[i1] >= 0) && (method > 0))
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     /*P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];*/
                     P_offd_j[jj_counter_offd]  = i1;
                     /*
                     if(method == 0)
                     {
                       P_offd_data[jj_counter_offd] = 0.0;
                     }
                     */
                     if (method == 1)
                     {
                        P_offd_data[jj_counter_offd] = - A_offd_data[jj];
                     }
                     else if (method == 2)
                     {
                        P_offd_data[jj_counter_offd] = - A_offd_data[jj] * a_diag[i];
                     }
                     jj_counter_offd++;
                  }
               }
            }
         }
         P_offd_i[i + 1] = jj_counter_offd;
      }
      nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(P_marker_offd, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(a_diag, NALU_HYPRE_MEMORY_HOST);
   P = nalu_hypre_ParCSRMatrixCreate(comm,
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                nalu_hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                0,
                                P_diag_i[n_fine],
                                P_offd_i[n_fine]);

   P_diag = nalu_hypre_ParCSRMatrixDiag(P);
   nalu_hypre_CSRMatrixData(P_diag) = P_diag_data;
   nalu_hypre_CSRMatrixI(P_diag) = P_diag_i;
   nalu_hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = nalu_hypre_ParCSRMatrixOffd(P);
   nalu_hypre_CSRMatrixData(P_offd) = P_offd_data;
   nalu_hypre_CSRMatrixI(P_offd) = P_offd_i;
   nalu_hypre_CSRMatrixJ(P_offd) = P_offd_j;

   num_cols_P_offd = 0;

   if (P_offd_size)
   {
      P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
#if 0
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker[i] = 0;
      }
      num_cols_P_offd = 0;
      for (i = 0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker[index])
         {
            num_cols_P_offd++;
            P_marker[index] = 1;
         }
      }

      col_map_offd_P = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_P_offd, NALU_HYPRE_MEMORY_HOST);
      tmp_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_P_offd, NALU_HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index] == 0) { index++; }
         tmp_map_offd[i] = index++;
      }

#if 0
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i = 0; i < P_offd_size; i++)
         P_offd_j[i] = nalu_hypre_BinarySearch(tmp_map_offd,
                                          P_offd_j[i],
                                          num_cols_P_offd);
      nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < n_fine; i++)
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }
   if (num_cols_P_offd)
   {
      nalu_hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      nalu_hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   }
   nalu_hypre_GetCommPkgRTFromCommPkgA(P, A, fine_to_coarse, tmp_map_offd);

   *P_ptr = P;

   nalu_hypre_TFree(tmp_map_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   // nalu_hypre_TFree(fine_to_coarse_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(coarse_counter, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count_offd, NALU_HYPRE_MEMORY_HOST);

   return (0);
}


/* Interpolation for MGR - Dynamic Row Sum method */

NALU_HYPRE_Int
nalu_hypre_MGRBuildPDRS( nalu_hypre_ParCSRMatrix   *A,
                    NALU_HYPRE_Int            *CF_marker,
                    NALU_HYPRE_BigInt         *num_cpts_global,
                    NALU_HYPRE_Int             blk_size,
                    NALU_HYPRE_Int             reserved_coarse_size,
                    NALU_HYPRE_Int             debug_flag,
                    nalu_hypre_ParCSRMatrix  **P_ptr)
{
   MPI_Comm          comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);

   nalu_hypre_CSRMatrix *A_offd         = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real      *A_offd_data    = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Int        num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_Real      *a_diag;

   nalu_hypre_ParCSRMatrix    *P;
   NALU_HYPRE_BigInt    *col_map_offd_P;
   NALU_HYPRE_Int       *tmp_map_offd;

   NALU_HYPRE_Int       *CF_marker_offd = NULL;

   nalu_hypre_CSRMatrix *P_diag;
   nalu_hypre_CSRMatrix *P_offd;

   NALU_HYPRE_Real      *P_diag_data;
   NALU_HYPRE_Int       *P_diag_i;
   NALU_HYPRE_Int       *P_diag_j;
   NALU_HYPRE_Real      *P_offd_data;
   NALU_HYPRE_Int       *P_offd_i;
   NALU_HYPRE_Int       *P_offd_j;

   NALU_HYPRE_Int        P_diag_size, P_offd_size;

   NALU_HYPRE_Int       *P_marker, *P_marker_offd;

   NALU_HYPRE_Int        jj_counter, jj_counter_offd;
   NALU_HYPRE_Int       *jj_count, *jj_count_offd;
   //   NALU_HYPRE_Int              jj_begin_row,jj_begin_row_offd;
   //   NALU_HYPRE_Int              jj_end_row,jj_end_row_offd;

   NALU_HYPRE_Int        start_indexing = 0; /* start indexing for P_data at 0 */

   NALU_HYPRE_Int        n_fine  = nalu_hypre_CSRMatrixNumRows(A_diag);

   NALU_HYPRE_Int       *fine_to_coarse;
   //NALU_HYPRE_BigInt             *fine_to_coarse_offd;
   NALU_HYPRE_Int       *coarse_counter;
   NALU_HYPRE_Int        coarse_shift;
   NALU_HYPRE_BigInt     total_global_cpts;
   //NALU_HYPRE_BigInt     my_first_cpt;
   NALU_HYPRE_Int        num_cols_P_offd;

   NALU_HYPRE_Int        i, i1;
   NALU_HYPRE_Int        j, jl, jj;
   NALU_HYPRE_Int        start;

   NALU_HYPRE_Real       one  = 1.0;

   NALU_HYPRE_Int        my_id;
   NALU_HYPRE_Int        num_procs;
   NALU_HYPRE_Int        num_threads;
   NALU_HYPRE_Int        num_sends;
   NALU_HYPRE_Int        index;
   NALU_HYPRE_Int        ns, ne, size, rest;

   NALU_HYPRE_Int       *int_buf_data;

   NALU_HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   //num_threads = nalu_hypre_NumThreads();
   // Temporary fix, disable threading
   // TODO: enable threading
   num_threads = 1;

   //my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag < 0)
   {
      debug_flag = -debug_flag;
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   if (num_cols_A_offd) { CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST); }

   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                            num_sends), NALU_HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         int_buf_data[index++]
            = CF_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
   }

   comm_handle = nalu_hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                               CF_marker_offd);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_threads, NALU_HYPRE_MEMORY_HOST);
   jj_count = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_threads, NALU_HYPRE_MEMORY_HOST);
   jj_count_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_threads, NALU_HYPRE_MEMORY_HOST);

   fine_to_coarse = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine, NALU_HYPRE_MEMORY_HOST);
#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

   /* RDF: this looks a little tricky, but doable */
#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,i1,jj,ns,ne,size,rest) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (j = 0; j < num_threads; j++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;

      if (j < rest)
      {
         ns = j * size + j;
         ne = (j + 1) * size + j + 1;
      }
      else
      {
         ns = j * size + rest;
         ne = (j + 1) * size + rest;
      }
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
          *  If i is a C-point, interpolation is the identity. Also set up
          *  mapping vector.
          *--------------------------------------------------------------------*/

         if (CF_marker[i] >= 0)
         {
            jj_count[j]++;
            fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j]++;
         }
         /*--------------------------------------------------------------------
          *  If i is an F-point, interpolation is the approximation of A_{ff}^{-1}A_{fc}
          *--------------------------------------------------------------------*/
         else
         {
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];
               if (CF_marker[i1] >= 0)
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];
                  if (CF_marker_offd[i1] >= 0)
                  {
                     jj_count_offd[j]++;
                  }
               }
            }
         }
         /*--------------------------------------------------------------------
          *  Set up the indexes for the DRS method
          *--------------------------------------------------------------------*/

      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < num_threads - 1; i++)
   {
      coarse_counter[i + 1] += coarse_counter[i];
      jj_count[i + 1] += jj_count[i];
      jj_count_offd[i + 1] += jj_count_offd[i];
   }
   i = num_threads - 1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine + 1, NALU_HYPRE_MEMORY_HOST);
   P_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_diag_size, NALU_HYPRE_MEMORY_HOST);
   P_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  P_diag_size, NALU_HYPRE_MEMORY_HOST);

   P_diag_i[n_fine] = jj_counter;


   P_offd_size = jj_counter_offd;

   P_offd_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine + 1, NALU_HYPRE_MEMORY_HOST);
   P_offd_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_offd_size, NALU_HYPRE_MEMORY_HOST);
   P_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  P_offd_size, NALU_HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d     Interp: Internal work 1 =     %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   //fine_to_coarse_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);

#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,ns,ne,size,rest,coarse_shift) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (j = 0; j < num_threads; j++)
   {
      coarse_shift = 0;
      if (j > 0) { coarse_shift = coarse_counter[j - 1]; }
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (j < rest)
      {
         ns = j * size + j;
         ne = (j + 1) * size + j + 1;
      }
      else
      {
         ns = j * size + rest;
         ne = (j + 1) * size + rest;
      }
      for (i = ns; i < ne; i++)
      {
         fine_to_coarse[i] += coarse_shift;
      }
   }

   /*index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         int_buf_data[index++]
            = fine_to_coarse[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
   }

   comm_handle = nalu_hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                    fine_to_coarse_offd);

   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   */
   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif

   //for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/
   a_diag = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  n_fine, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < n_fine; i++)
   {
      for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
      {
         i1 = A_diag_j[jj];
         if ( i == i1 ) /* diagonal of A only */
         {
            a_diag[i] = 1.0 / A_diag_data[jj];
         }
      }
   }

#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,jl,i1,jj,ns,ne,size,rest,P_marker,P_marker_offd,jj_counter,jj_counter_offd,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (jl = 0; jl < num_threads; jl++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (jl < rest)
      {
         ns = jl * size + jl;
         ne = (jl + 1) * size + jl + 1;
      }
      else
      {
         ns = jl * size + rest;
         ne = (jl + 1) * size + rest;
      }
      jj_counter = 0;
      if (jl > 0) { jj_counter = jj_count[jl - 1]; }
      jj_counter_offd = 0;
      if (jl > 0) { jj_counter_offd = jj_count_offd[jl - 1]; }
      P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine, NALU_HYPRE_MEMORY_HOST);
      if (num_cols_A_offd)
      {
         P_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
      }
      else
      {
         P_marker_offd = NULL;
      }

      for (i = 0; i < n_fine; i++)
      {
         P_marker[i] = -1;
      }
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker_offd[i] = -1;
      }
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
          *  If i is a c-point, interpolation is the identity.
          *--------------------------------------------------------------------*/
         if (CF_marker[i] >= 0)
         {
            P_diag_i[i] = jj_counter;
            P_diag_j[jj_counter]    = fine_to_coarse[i];
            P_diag_data[jj_counter] = one;
            jj_counter++;
         }
         /*--------------------------------------------------------------------
          *  If i is an F-point, build interpolation.
          *--------------------------------------------------------------------*/
         else
         {
            /* Diagonal part of P */
            P_diag_i[i] = jj_counter;
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  P_diag_data[jj_counter] = - A_diag_data[jj] * a_diag[i];

                  jj_counter++;
               }
            }

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];

                  /*-----------------------------------------------------------
                   * If neighbor i1 is a C-point, set column number in P_offd_j
                   * and initialize interpolation weight to zero.
                   *-----------------------------------------------------------*/

                  if (CF_marker_offd[i1] >= 0)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     /*P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];*/
                     P_offd_j[jj_counter_offd]  = i1;
                     P_offd_data[jj_counter_offd] = - A_offd_data[jj] * a_diag[i];

                     jj_counter_offd++;
                  }
               }
            }
         }
         P_offd_i[i + 1] = jj_counter_offd;
      }
      nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(P_marker_offd, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(a_diag, NALU_HYPRE_MEMORY_HOST);
   P = nalu_hypre_ParCSRMatrixCreate(comm,
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                nalu_hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                0,
                                P_diag_i[n_fine],
                                P_offd_i[n_fine]);

   P_diag = nalu_hypre_ParCSRMatrixDiag(P);
   nalu_hypre_CSRMatrixData(P_diag) = P_diag_data;
   nalu_hypre_CSRMatrixI(P_diag) = P_diag_i;
   nalu_hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = nalu_hypre_ParCSRMatrixOffd(P);
   nalu_hypre_CSRMatrixData(P_offd) = P_offd_data;
   nalu_hypre_CSRMatrixI(P_offd) = P_offd_i;
   nalu_hypre_CSRMatrixJ(P_offd) = P_offd_j;

   num_cols_P_offd = 0;

   if (P_offd_size)
   {
      P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);

#if 0
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i = 0; i < num_cols_A_offd; i++)
      {
         P_marker[i] = 0;
      }
      num_cols_P_offd = 0;
      for (i = 0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker[index])
         {
            num_cols_P_offd++;
            P_marker[index] = 1;
         }
      }

      tmp_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_P_offd, NALU_HYPRE_MEMORY_HOST);
      col_map_offd_P = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_P_offd, NALU_HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_cols_P_offd; i++)
      {
         while (P_marker[index] == 0) { index++; }
         tmp_map_offd[i] = index++;
      }

#if 0
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i = 0; i < P_offd_size; i++)
         P_offd_j[i] = nalu_hypre_BinarySearch(tmp_map_offd,
                                          P_offd_j[i],
                                          num_cols_P_offd);
      nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < n_fine; i++)
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }
   if (num_cols_P_offd)
   {
      nalu_hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      nalu_hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   }
   nalu_hypre_GetCommPkgRTFromCommPkgA(P, A, fine_to_coarse, tmp_map_offd);

   *P_ptr = P;

   nalu_hypre_TFree(tmp_map_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   // nalu_hypre_TFree(fine_to_coarse_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(coarse_counter, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count_offd, NALU_HYPRE_MEMORY_HOST);

   return (0);
}

/* Scale ParCSR matrix A = scalar * A
 * A: the target CSR matrix
 * vector: array of real numbers
 */
NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixLeftScale(NALU_HYPRE_Real *vector,
                            nalu_hypre_ParCSRMatrix *A)
{
   NALU_HYPRE_Int i, j, n_local;
   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int             *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);

   nalu_hypre_CSRMatrix *A_offd         = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real      *A_offd_data    = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int             *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);

   n_local = nalu_hypre_CSRMatrixNumRows(A_diag);

   for (i = 0; i < n_local; i++)
   {
      NALU_HYPRE_Real factor = vector[i];
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         A_diag_data[j] *= factor;
      }
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         A_offd_data[j] *= factor;
      }
   }

   return (0);
}

NALU_HYPRE_Int
nalu_hypre_MGRGetAcfCPR(nalu_hypre_ParCSRMatrix     *A,
                   NALU_HYPRE_Int               blk_size,
                   NALU_HYPRE_Int              *c_marker,
                   NALU_HYPRE_Int              *f_marker,
                   nalu_hypre_ParCSRMatrix    **A_CF_ptr)
{
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int i, j, jj, jj1;
   NALU_HYPRE_Int jj_counter, cpts_cnt;
   nalu_hypre_ParCSRMatrix *A_CF = NULL;
   nalu_hypre_CSRMatrix *A_CF_diag = NULL;

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);
   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);

   NALU_HYPRE_Int *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Complex *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);

   NALU_HYPRE_Int total_fpts, n_fpoints;
   NALU_HYPRE_Int num_rows = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));
   NALU_HYPRE_Int nnz_diag_new = 0;
   NALU_HYPRE_Int num_procs, my_id;
   nalu_hypre_IntArray *wrap_cf = NULL;
   nalu_hypre_IntArray *coarse_dof_func_ptr = NULL;
   NALU_HYPRE_BigInt num_row_cpts_global[2], num_col_fpts_global[2];
   NALU_HYPRE_BigInt total_global_row_cpts, total_global_col_fpts;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   // Count total F-points
   // Also setup F to C column map
   total_fpts = 0;
   NALU_HYPRE_Int *f_to_c_col_map = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_rows; i++)
   {
      //      if (c_marker[i] == 1)
      //      {
      //         total_cpts++;
      //      }
      if (f_marker[i] == 1)
      {
         f_to_c_col_map[i] = total_fpts;
         total_fpts++;
      }
   }
   n_fpoints = blk_size;
   /* get the number of coarse rows */
   wrap_cf = nalu_hypre_IntArrayCreate(num_rows);
   nalu_hypre_IntArrayMemoryLocation(wrap_cf) = NALU_HYPRE_MEMORY_HOST;
   nalu_hypre_IntArrayData(wrap_cf) = c_marker;
   nalu_hypre_BoomerAMGCoarseParms(comm, num_rows, 1, NULL, wrap_cf, &coarse_dof_func_ptr,
                              num_row_cpts_global);
   nalu_hypre_IntArrayDestroy(coarse_dof_func_ptr);
   coarse_dof_func_ptr = NULL;

   //nalu_hypre_printf("my_id = %d, cpts_this = %d, cpts_next = %d\n", my_id, num_row_cpts_global[0], num_row_cpts_global[1]);

   if (my_id == (num_procs - 1)) { total_global_row_cpts = num_row_cpts_global[1]; }
   nalu_hypre_MPI_Bcast(&total_global_row_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /* get the number of coarse rows */
   nalu_hypre_IntArrayData(wrap_cf) = f_marker;
   nalu_hypre_BoomerAMGCoarseParms(comm, num_rows, 1, NULL, wrap_cf, &coarse_dof_func_ptr,
                              num_col_fpts_global);
   nalu_hypre_IntArrayDestroy(coarse_dof_func_ptr);
   coarse_dof_func_ptr = NULL;
   nalu_hypre_IntArrayData(wrap_cf) = NULL;
   nalu_hypre_IntArrayDestroy(wrap_cf);

   //nalu_hypre_printf("my_id = %d, cpts_this = %d, cpts_next = %d\n", my_id, num_col_fpts_global[0], num_col_fpts_global[1]);

   if (my_id == (num_procs - 1)) { total_global_col_fpts = num_col_fpts_global[1]; }
   nalu_hypre_MPI_Bcast(&total_global_col_fpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   // First pass: count the nnz of A_CF
   jj_counter = 0;
   cpts_cnt = 0;
   for (i = 0; i < num_rows; i++)
   {
      if (c_marker[i] == 1)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            jj = A_diag_j[j];
            if (f_marker[jj] == 1)
            {
               jj1 = f_to_c_col_map[jj];
               if (jj1 >= cpts_cnt * n_fpoints && jj1 < (cpts_cnt + 1)*n_fpoints)
               {
                  jj_counter++;
               }
            }
         }
         cpts_cnt++;
      }
   }
   nnz_diag_new = jj_counter;

   NALU_HYPRE_Int     *A_CF_diag_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int, cpts_cnt + 1, memory_location);
   NALU_HYPRE_Int     *A_CF_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nnz_diag_new, memory_location);
   NALU_HYPRE_Complex *A_CF_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nnz_diag_new, memory_location);
   A_CF_diag_i[cpts_cnt] = nnz_diag_new;

   jj_counter = 0;
   cpts_cnt = 0;
   for (i = 0; i < num_rows; i++)
   {
      if (c_marker[i] == 1)
      {
         A_CF_diag_i[cpts_cnt] = jj_counter;
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            jj = A_diag_j[j];
            if (f_marker[jj] == 1)
            {
               jj1 = f_to_c_col_map[jj];
               if (jj1 >= cpts_cnt * n_fpoints && jj1 < (cpts_cnt + 1)*n_fpoints)
               {
                  A_CF_diag_j[jj_counter] = jj1;
                  A_CF_diag_data[jj_counter] = A_diag_data[j];
                  jj_counter++;
               }
            }
         }
         cpts_cnt++;
      }
   }

   /* Create A_CF matrix */
   A_CF = nalu_hypre_ParCSRMatrixCreate(comm,
                                   total_global_row_cpts,
                                   total_global_col_fpts,
                                   num_row_cpts_global,
                                   num_col_fpts_global,
                                   0,
                                   nnz_diag_new,
                                   0);

   A_CF_diag = nalu_hypre_ParCSRMatrixDiag(A_CF);
   nalu_hypre_CSRMatrixData(A_CF_diag) = A_CF_diag_data;
   nalu_hypre_CSRMatrixI(A_CF_diag) = A_CF_diag_i;
   nalu_hypre_CSRMatrixJ(A_CF_diag) = A_CF_diag_j;

   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(A_CF)) = NULL;
   nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(A_CF)) = NULL;
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(A_CF)) = NULL;

   *A_CF_ptr = A_CF;

   nalu_hypre_TFree(f_to_c_col_map, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRTruncateAcfCPRDevice
 *
 * TODO (VPM): Port truncation to GPUs
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRTruncateAcfCPRDevice(nalu_hypre_ParCSRMatrix  *A_CF,
                              nalu_hypre_ParCSRMatrix **A_CF_new_ptr)
{
   nalu_hypre_ParCSRMatrix *hA_CF;
   nalu_hypre_ParCSRMatrix *A_CF_new;

   nalu_hypre_GpuProfilingPushRange("MGRTruncateAcfCPR");

   /* Clone matrix to host, truncate, and migrate result to device */
   hA_CF = nalu_hypre_ParCSRMatrixClone_v2(A_CF, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_MGRTruncateAcfCPR(hA_CF, &A_CF_new);
   nalu_hypre_ParCSRMatrixMigrate(A_CF_new, NALU_HYPRE_MEMORY_DEVICE);
   nalu_hypre_ParCSRMatrixDestroy(hA_CF);

   /* Set output pointer */
   *A_CF_new_ptr = A_CF_new;

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRTruncateAcfCPR
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRTruncateAcfCPR(nalu_hypre_ParCSRMatrix  *A_CF,
                        nalu_hypre_ParCSRMatrix **A_CF_new_ptr)
{
   /* Input matrix info */
   MPI_Comm             comm           = nalu_hypre_ParCSRMatrixComm(A_CF);
   NALU_HYPRE_BigInt         num_rows       = nalu_hypre_ParCSRMatrixGlobalNumRows(A_CF);
   NALU_HYPRE_BigInt         num_cols       = nalu_hypre_ParCSRMatrixGlobalNumCols(A_CF);

   nalu_hypre_CSRMatrix     *A_CF_diag      = nalu_hypre_ParCSRMatrixDiag(A_CF);
   NALU_HYPRE_Int           *A_CF_diag_i    = nalu_hypre_CSRMatrixI(A_CF_diag);
   NALU_HYPRE_Int           *A_CF_diag_j    = nalu_hypre_CSRMatrixJ(A_CF_diag);
   NALU_HYPRE_Complex       *A_CF_diag_data = nalu_hypre_CSRMatrixData(A_CF_diag);
   NALU_HYPRE_Int            num_rows_local = nalu_hypre_CSRMatrixNumRows(A_CF_diag);

   /* Output matrix info */
   nalu_hypre_ParCSRMatrix  *A_CF_new;
   nalu_hypre_CSRMatrix     *A_CF_diag_new;
   NALU_HYPRE_Int           *A_CF_diag_i_new;
   NALU_HYPRE_Int           *A_CF_diag_j_new;
   NALU_HYPRE_Complex       *A_CF_diag_data_new;
   NALU_HYPRE_Int            nnz_diag_new;

   /* Local variables */
   NALU_HYPRE_Int            i, j, jj;
   NALU_HYPRE_Int            jj_counter;
   NALU_HYPRE_Int            blk_size = num_cols / num_rows;

   /* Sanity check */
   nalu_hypre_assert(nalu_hypre_ParCSRMatrixMemoryLocation(A_CF) == NALU_HYPRE_MEMORY_HOST);

   /* First pass: count the nnz of truncated (new) A_CF */
   jj_counter = 0;
   for (i = 0; i < num_rows_local; i++)
   {
      for (j = A_CF_diag_i[i]; j < A_CF_diag_i[i + 1]; j++)
      {
         jj = A_CF_diag_j[j];
         if (jj >= i * blk_size && jj < (i + 1) * blk_size)
         {
            jj_counter++;
         }
      }
   }
   nnz_diag_new = jj_counter;

   /* Create truncated matrix */
   A_CF_new = nalu_hypre_ParCSRMatrixCreate(comm,
                                       num_rows,
                                       num_cols,
                                       nalu_hypre_ParCSRMatrixRowStarts(A_CF),
                                       nalu_hypre_ParCSRMatrixColStarts(A_CF),
                                       0,
                                       nnz_diag_new,
                                       0);

   nalu_hypre_ParCSRMatrixInitialize_v2(A_CF_new, NALU_HYPRE_MEMORY_HOST);
   A_CF_diag_new      = nalu_hypre_ParCSRMatrixDiag(A_CF_new);
   A_CF_diag_i_new    = nalu_hypre_CSRMatrixI(A_CF_diag_new);
   A_CF_diag_j_new    = nalu_hypre_CSRMatrixJ(A_CF_diag_new);
   A_CF_diag_data_new = nalu_hypre_CSRMatrixData(A_CF_diag_new);

   /* Second pass: fill entries of the truncated (new) A_CF */
   jj_counter = 0;
   for (i = 0; i < num_rows_local; i++)
   {
      A_CF_diag_i_new[i] = jj_counter;
      for (j = A_CF_diag_i[i]; j < A_CF_diag_i[i + 1]; j++)
      {
         jj = A_CF_diag_j[j];
         if (jj >= i * blk_size && jj < (i + 1) * blk_size)
         {
            A_CF_diag_j_new[jj_counter] = jj;
            A_CF_diag_data_new[jj_counter] = A_CF_diag_data[j];
            jj_counter++;
         }
      }
   }
   A_CF_diag_i_new[num_rows_local] = nnz_diag_new;

   /* Set output pointer */
   *A_CF_new_ptr = A_CF_new;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRComputeNonGalerkinCoarseGrid
 *
 * Computes the level (grid) operator A_H = RAP, assuming that restriction
 * equals to the injection operator: R = [0 I]
 *
 * Available methods:
 *   1: inv(A_FF) approximated by its (block) diagonal inverse
 *   2: CPR-like approx. with inv(A_FF) approx. by its diagonal inverse
 *   3: CPR-like approx. with inv(A_FF) approx. by its block diagonal inverse
 *   4: inv(A_FF) approximated by sparse approximate inverse
 *
 * TODO (VPM): Can we have a single function that works for host and device?
 *             RT is not being used.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRComputeNonGalerkinCoarseGrid(nalu_hypre_ParCSRMatrix    *A,
                                      nalu_hypre_ParCSRMatrix    *Wp,
                                      nalu_hypre_ParCSRMatrix    *RT,
                                      NALU_HYPRE_Int              bsize,
                                      NALU_HYPRE_Int              ordering,
                                      NALU_HYPRE_Int              method,
                                      NALU_HYPRE_Int              Pmax,
                                      NALU_HYPRE_Int             *CF_marker,
                                      nalu_hypre_ParCSRMatrix   **A_H_ptr)
{
   NALU_HYPRE_Int *c_marker, *f_marker;
   NALU_HYPRE_Int n_local_fine_grid, i, i1, jj;
   nalu_hypre_ParCSRMatrix *A_cc = NULL;
   nalu_hypre_ParCSRMatrix *A_ff = NULL;
   nalu_hypre_ParCSRMatrix *A_fc = NULL;
   nalu_hypre_ParCSRMatrix *A_cf = NULL;
   nalu_hypre_ParCSRMatrix *A_ff_inv = NULL;
   nalu_hypre_ParCSRMatrix *A_H = NULL;
   nalu_hypre_ParCSRMatrix *A_H_correction = NULL;
   NALU_HYPRE_Int  max_elmts = Pmax;
   NALU_HYPRE_Real alpha = -1.0;

   NALU_HYPRE_BigInt         coarse_pnts_global[2];
   NALU_HYPRE_BigInt         fine_pnts_global[2];
   nalu_hypre_IntArray *marker_array = NULL;
   //   NALU_HYPRE_Real wall_time = 0.;
   //   NALU_HYPRE_Real wall_time_1 = 0.;

   NALU_HYPRE_Int my_id;
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   //wall_time = time_getWallclockSeconds();
   n_local_fine_grid = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));
   c_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_local_fine_grid, memory_location);
   f_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_local_fine_grid, memory_location);

   for (i = 0; i < n_local_fine_grid; i++)
   {
      NALU_HYPRE_Int point_type = CF_marker[i];
      //nalu_hypre_assert(point_type == 1 || point_type == -1);
      c_marker[i] = point_type;
      f_marker[i] = -point_type;
   }
   // get local range for C and F points
   // Set IntArray pointers to obtain global row and col (start) ranges
   marker_array = nalu_hypre_IntArrayCreate(n_local_fine_grid);
   nalu_hypre_IntArrayMemoryLocation(marker_array) = memory_location;
   nalu_hypre_IntArrayData(marker_array) = c_marker;
   // get range for c_points
   nalu_hypre_BoomerAMGCoarseParms(comm, n_local_fine_grid, 1, NULL, marker_array, NULL,
                              coarse_pnts_global);
   // get range for f_points
   nalu_hypre_IntArrayData(marker_array) = f_marker;
   nalu_hypre_BoomerAMGCoarseParms(comm, n_local_fine_grid, 1, NULL, marker_array, NULL, fine_pnts_global);

   // Generate A_FF, A_FC, A_CC and A_CF submatrices.
   // Note: Not all submatrices are needed for each method below.
   // nalu_hypre_ParCSRMatrixGenerateFFFC computes A_FF and A_FC given the CF_marker and start locations for the global C-points.
   // To compute A_CC and A_CF, we need to pass in equivalent information for F-points. (i.e. CF_marker marking F-points
   // and start locations for the global F-points.
   nalu_hypre_ParCSRMatrixGenerateFFFC(A, c_marker, coarse_pnts_global, NULL, &A_fc, &A_ff);
   nalu_hypre_ParCSRMatrixGenerateFFFC(A, f_marker, fine_pnts_global, NULL, &A_cf, &A_cc);

   if (method == 1)
   {
      if (Wp != NULL)
      {
         A_H_correction = nalu_hypre_ParCSRMatMat(A_cf, Wp);
      }
      else
      {
         // Build block diagonal inverse for A_FF
         nalu_hypre_ParCSRMatrixBlockDiagMatrix(A_ff, 1, -1, NULL, 1, &A_ff_inv);
         // compute Wp = A_ff_inv * A_fc
         // NOTE: Use nalu_hypre_ParMatmul here instead of nalu_hypre_ParCSRMatMat to avoid padding
         // zero entries at diagonals for the latter routine. Use MatMat once this padding
         // issue is resolved since it is more efficient.
         //         nalu_hypre_ParCSRMatrix *Wp_tmp = nalu_hypre_ParCSRMatMat(A_ff_inv, A_fc);
         nalu_hypre_ParCSRMatrix *Wp_tmp = nalu_hypre_ParMatmul(A_ff_inv, A_fc);
         // compute correction A_H_correction = A_cf * (A_ff_inv * A_fc);
         //         A_H_correction = nalu_hypre_ParMatmul(A_cf, Wp_tmp);
         A_H_correction = nalu_hypre_ParCSRMatMat(A_cf, Wp_tmp);
         nalu_hypre_ParCSRMatrixDestroy(Wp_tmp);
         nalu_hypre_ParCSRMatrixDestroy(A_ff_inv);
      }
   }
   else if (method == 2 || method == 3)
   {
      // extract the diagonal of A_cf
      nalu_hypre_ParCSRMatrix *A_cf_truncated = NULL;
      nalu_hypre_MGRGetAcfCPR(A, bsize, c_marker, f_marker, &A_cf_truncated);
      if (Wp != NULL)
      {
         A_H_correction = nalu_hypre_ParCSRMatMat(A_cf_truncated, Wp);
      }
      else
      {
         /* TODO (VPM): Shouldn't blk_inv_size = bsize for method == 3? Check with DOK */
         NALU_HYPRE_Int blk_inv_size = method == 2 ? bsize : 1;
         nalu_hypre_ParCSRMatrixBlockDiagMatrix(A_ff, blk_inv_size, -1, NULL, 1, &A_ff_inv);
         nalu_hypre_ParCSRMatrix *Wr = NULL;
         Wr = nalu_hypre_ParCSRMatMat(A_cf_truncated, A_ff_inv);
         A_H_correction = nalu_hypre_ParCSRMatMat(Wr, A_fc);
         nalu_hypre_ParCSRMatrixDestroy(Wr);
         nalu_hypre_ParCSRMatrixDestroy(A_ff_inv);
      }
      nalu_hypre_ParCSRMatrixDestroy(A_cf_truncated);
   }
   else if (method == 4)
   {
      // Approximate inverse for ideal interploation
      nalu_hypre_ParCSRMatrix *A_ff_inv = NULL;
      nalu_hypre_ParCSRMatrix *minus_Wp = NULL;
      nalu_hypre_MGRApproximateInverse(A_ff, &A_ff_inv);
      minus_Wp = nalu_hypre_ParCSRMatMat(A_ff_inv, A_fc);
      A_H_correction = nalu_hypre_ParCSRMatMat(A_cf, minus_Wp);

      nalu_hypre_ParCSRMatrixDestroy(minus_Wp);
   }
   // Free data
   nalu_hypre_ParCSRMatrixDestroy(A_ff);
   nalu_hypre_ParCSRMatrixDestroy(A_fc);
   nalu_hypre_ParCSRMatrixDestroy(A_cf);

   // perform dropping for A_H_correction
   // specific to multiphase poromechanics
   // we only keep the diagonal of each block
   NALU_HYPRE_Int n_local_cpoints = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A_H_correction));

   nalu_hypre_CSRMatrix *A_H_correction_diag = nalu_hypre_ParCSRMatrixDiag(A_H_correction);
   NALU_HYPRE_Real      *A_H_correction_diag_data = nalu_hypre_CSRMatrixData(A_H_correction_diag);
   NALU_HYPRE_Int             *A_H_correction_diag_i = nalu_hypre_CSRMatrixI(A_H_correction_diag);
   NALU_HYPRE_Int             *A_H_correction_diag_j = nalu_hypre_CSRMatrixJ(A_H_correction_diag);
   NALU_HYPRE_Int             ncol_diag = nalu_hypre_CSRMatrixNumCols(A_H_correction_diag);

   nalu_hypre_CSRMatrix *A_H_correction_offd = nalu_hypre_ParCSRMatrixOffd(A_H_correction);
   NALU_HYPRE_Real      *A_H_correction_offd_data = nalu_hypre_CSRMatrixData(A_H_correction_offd);
   NALU_HYPRE_Int             *A_H_correction_offd_i = nalu_hypre_CSRMatrixI(A_H_correction_offd);
   NALU_HYPRE_Int             *A_H_correction_offd_j = nalu_hypre_CSRMatrixJ(A_H_correction_offd);

   // drop small entries in the correction
   if (Pmax > 0)
   {
      if (ordering == 0) // interleaved ordering
      {
         NALU_HYPRE_Int *A_H_correction_diag_i_new = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_local_cpoints + 1,
                                                              memory_location);
         NALU_HYPRE_Int *A_H_correction_diag_j_new = nalu_hypre_CTAlloc(NALU_HYPRE_Int,
                                                              (bsize + max_elmts) * n_local_cpoints, memory_location);
         NALU_HYPRE_Complex *A_H_correction_diag_data_new = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,
                                                                     (bsize + max_elmts) * n_local_cpoints, memory_location);
         NALU_HYPRE_Int num_nonzeros_diag_new = 0;

         NALU_HYPRE_Int *A_H_correction_offd_i_new = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_local_cpoints + 1,
                                                              memory_location);
         NALU_HYPRE_Int *A_H_correction_offd_j_new = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_elmts * n_local_cpoints,
                                                              memory_location);
         NALU_HYPRE_Complex *A_H_correction_offd_data_new = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,
                                                                     max_elmts * n_local_cpoints, memory_location);
         NALU_HYPRE_Int num_nonzeros_offd_new = 0;


         for (i = 0; i < n_local_cpoints; i++)
         {
            NALU_HYPRE_Int max_num_nonzeros = A_H_correction_diag_i[i + 1] - A_H_correction_diag_i[i] +
                                         A_H_correction_offd_i[i + 1] - A_H_correction_offd_i[i];
            NALU_HYPRE_Int *aux_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_nonzeros, memory_location);
            NALU_HYPRE_Real *aux_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, max_num_nonzeros, memory_location);
            NALU_HYPRE_Int row_start = i - (i % bsize);
            NALU_HYPRE_Int row_stop = row_start + bsize - 1;
            NALU_HYPRE_Int cnt = 0;
            for (jj = A_H_correction_offd_i[i]; jj < A_H_correction_offd_i[i + 1]; jj++)
            {
               aux_j[cnt] = A_H_correction_offd_j[jj] + ncol_diag;
               aux_data[cnt] = A_H_correction_offd_data[jj];
               cnt++;
            }
            for (jj = A_H_correction_diag_i[i]; jj < A_H_correction_diag_i[i + 1]; jj++)
            {
               aux_j[cnt] = A_H_correction_diag_j[jj];
               aux_data[cnt] = A_H_correction_diag_data[jj];
               cnt++;
            }
            nalu_hypre_qsort2_abs(aux_j, aux_data, 0, cnt - 1);

            for (jj = A_H_correction_diag_i[i]; jj < A_H_correction_diag_i[i + 1]; jj++)
            {
               i1 = A_H_correction_diag_j[jj];
               if (i1 >= row_start && i1 <= row_stop)
               {
                  // copy data to new arrays
                  A_H_correction_diag_j_new[num_nonzeros_diag_new] = i1;
                  A_H_correction_diag_data_new[num_nonzeros_diag_new] = A_H_correction_diag_data[jj];
                  ++num_nonzeros_diag_new;
               }
               else
               {
                  // Do nothing
               }
            }

            if (max_elmts > 0)
            {
               for (jj = 0; jj < nalu_hypre_min(max_elmts, cnt); jj++)
               {
                  NALU_HYPRE_Int col_idx = aux_j[jj];
                  NALU_HYPRE_Real col_value = aux_data[jj];
                  if (col_idx < ncol_diag && (col_idx < row_start || col_idx > row_stop))
                  {
                     A_H_correction_diag_j_new[num_nonzeros_diag_new] = col_idx;
                     A_H_correction_diag_data_new[num_nonzeros_diag_new] = col_value;
                     ++num_nonzeros_diag_new;
                  }
                  else if (col_idx >= ncol_diag)
                  {
                     A_H_correction_offd_j_new[num_nonzeros_offd_new] = col_idx - ncol_diag;
                     A_H_correction_offd_data_new[num_nonzeros_offd_new] = col_value;
                     ++num_nonzeros_offd_new;
                  }
               }
            }
            A_H_correction_diag_i_new[i + 1] = num_nonzeros_diag_new;
            A_H_correction_offd_i_new[i + 1] = num_nonzeros_offd_new;

            nalu_hypre_TFree(aux_j, memory_location);
            nalu_hypre_TFree(aux_data, memory_location);
         }

         nalu_hypre_TFree(A_H_correction_diag_i, memory_location);
         nalu_hypre_TFree(A_H_correction_diag_j, memory_location);
         nalu_hypre_TFree(A_H_correction_diag_data, memory_location);
         nalu_hypre_CSRMatrixI(A_H_correction_diag) = A_H_correction_diag_i_new;
         nalu_hypre_CSRMatrixJ(A_H_correction_diag) = A_H_correction_diag_j_new;
         nalu_hypre_CSRMatrixData(A_H_correction_diag) = A_H_correction_diag_data_new;
         nalu_hypre_CSRMatrixNumNonzeros(A_H_correction_diag) = num_nonzeros_diag_new;

         if (A_H_correction_offd_i) { nalu_hypre_TFree(A_H_correction_offd_i, memory_location); }
         if (A_H_correction_offd_j) { nalu_hypre_TFree(A_H_correction_offd_j, memory_location); }
         if (A_H_correction_offd_data) { nalu_hypre_TFree(A_H_correction_offd_data, memory_location); }
         nalu_hypre_CSRMatrixI(A_H_correction_offd) = A_H_correction_offd_i_new;
         nalu_hypre_CSRMatrixJ(A_H_correction_offd) = A_H_correction_offd_j_new;
         nalu_hypre_CSRMatrixData(A_H_correction_offd) = A_H_correction_offd_data_new;
         nalu_hypre_CSRMatrixNumNonzeros(A_H_correction_offd) = num_nonzeros_offd_new;
      }
      else
      {
         // do nothing. Dropping not yet implemented for non-interleaved variable ordering options
         //  nalu_hypre_printf("Error!! Block ordering for non-Galerkin coarse grid is not currently supported\n");
         //  exit(-1);
      }
   }

   /* Coarse grid / Schur complement */
   alpha = -1.0;
   nalu_hypre_ParCSRMatrixAdd(1.0, A_cc, alpha, A_H_correction, &A_H);

   /* Free memory */
   nalu_hypre_ParCSRMatrixDestroy(A_cc);
   nalu_hypre_ParCSRMatrixDestroy(A_H_correction);
   nalu_hypre_TFree(c_marker, memory_location);
   nalu_hypre_TFree(f_marker, memory_location);
   // free IntArray. Note: IntArrayData was not initialized so need not be freed here (pointer is already freed elsewhere).
   nalu_hypre_TFree(marker_array, memory_location);

   /* Set output pointer */
   *A_H_ptr = A_H;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_MGRComputeAlgebraicFixedStress(nalu_hypre_ParCSRMatrix  *A,
                                     NALU_HYPRE_BigInt        *mgr_idx_array,
                                     NALU_HYPRE_Solver         A_ff_solver)
{
   NALU_HYPRE_Int *U_marker, *S_marker, *P_marker;
   NALU_HYPRE_Int n_fine, i;
   NALU_HYPRE_BigInt ibegin;
   nalu_hypre_ParCSRMatrix *A_up;
   nalu_hypre_ParCSRMatrix *A_uu;
   nalu_hypre_ParCSRMatrix *A_su;
   nalu_hypre_ParCSRMatrix *A_pu;
   nalu_hypre_ParVector *e1_vector;
   nalu_hypre_ParVector *e2_vector;
   nalu_hypre_ParVector *e3_vector;
   nalu_hypre_ParVector *e4_vector;
   nalu_hypre_ParVector *e5_vector;

   n_fine = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));
   ibegin = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   nalu_hypre_assert(ibegin == mgr_idx_array[0]);
   U_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   S_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < n_fine; i++)
   {
      U_marker[i] = -1;
      S_marker[i] = -1;
      P_marker[i] = -1;
   }

   // create C and F markers
   for (i = 0; i < n_fine; i++)
   {
      if (i < mgr_idx_array[1] - ibegin)
      {
         U_marker[i] = 1;
      }
      else if (i >= (mgr_idx_array[1] - ibegin) && i < (mgr_idx_array[2] - ibegin))
      {
         S_marker[i] = 1;
      }
      else
      {
         P_marker[i] = 1;
      }
   }

   // Get A_up
   nalu_hypre_MGRGetSubBlock(A, U_marker, P_marker, 0, &A_up);
   // GetA_uu
   nalu_hypre_MGRGetSubBlock(A, U_marker, U_marker, 0, &A_uu);
   // Get A_su
   nalu_hypre_MGRGetSubBlock(A, S_marker, U_marker, 0, &A_su);
   // Get A_pu
   nalu_hypre_MGRGetSubBlock(A, P_marker, U_marker, 0, &A_pu);

   e1_vector = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A_up),
                                     nalu_hypre_ParCSRMatrixGlobalNumCols(A_up),
                                     nalu_hypre_ParCSRMatrixColStarts(A_up));
   nalu_hypre_ParVectorInitialize(e1_vector);
   nalu_hypre_ParVectorSetConstantValues(e1_vector, 1.0);

   e2_vector = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A_uu),
                                     nalu_hypre_ParCSRMatrixGlobalNumRows(A_uu),
                                     nalu_hypre_ParCSRMatrixRowStarts(A_uu));
   nalu_hypre_ParVectorInitialize(e2_vector);
   nalu_hypre_ParVectorSetConstantValues(e2_vector, 0.0);

   e3_vector = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A_uu),
                                     nalu_hypre_ParCSRMatrixGlobalNumRows(A_uu),
                                     nalu_hypre_ParCSRMatrixRowStarts(A_uu));
   nalu_hypre_ParVectorInitialize(e3_vector);
   nalu_hypre_ParVectorSetConstantValues(e3_vector, 0.0);

   e4_vector = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A_su),
                                     nalu_hypre_ParCSRMatrixGlobalNumRows(A_su),
                                     nalu_hypre_ParCSRMatrixRowStarts(A_su));
   nalu_hypre_ParVectorInitialize(e4_vector);
   nalu_hypre_ParVectorSetConstantValues(e4_vector, 0.0);

   e5_vector = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A_pu),
                                     nalu_hypre_ParCSRMatrixGlobalNumRows(A_pu),
                                     nalu_hypre_ParCSRMatrixRowStarts(A_pu));
   nalu_hypre_ParVectorInitialize(e5_vector);
   nalu_hypre_ParVectorSetConstantValues(e5_vector, 0.0);

   // compute e2 = A_up * e1
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(1.0, A_up, e1_vector, 0.0, e2_vector, e2_vector);

   // solve e3 = A_uu^-1 * e2
   nalu_hypre_BoomerAMGSolve(A_ff_solver, A_uu, e2_vector, e3_vector);

   // compute e4 = A_su * e3
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(1.0, A_su, e3_vector, 0.0, e4_vector, e4_vector);

   // compute e4 = A_su * e3
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(1.0, A_su, e3_vector, 0.0, e4_vector, e4_vector);

   // print e4
   nalu_hypre_ParVectorPrintIJ(e4_vector, 1, "Dsp");

   // compute e5 = A_pu * e3
   nalu_hypre_ParCSRMatrixMatvecOutOfPlace(1.0, A_pu, e3_vector, 0.0, e5_vector, e5_vector);

   nalu_hypre_ParVectorPrintIJ(e5_vector, 1, "Dpp");

   nalu_hypre_ParVectorDestroy(e1_vector);
   nalu_hypre_ParVectorDestroy(e2_vector);
   nalu_hypre_ParVectorDestroy(e3_vector);
   nalu_hypre_ParCSRMatrixDestroy(A_uu);
   nalu_hypre_ParCSRMatrixDestroy(A_up);
   nalu_hypre_ParCSRMatrixDestroy(A_pu);
   nalu_hypre_ParCSRMatrixDestroy(A_su);
   nalu_hypre_TFree(U_marker, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(S_marker, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}


NALU_HYPRE_Int
nalu_hypre_MGRApproximateInverse(nalu_hypre_ParCSRMatrix      *A,
                            nalu_hypre_ParCSRMatrix     **A_inv)
{
   NALU_HYPRE_Int print_level, mr_max_row_nnz, mr_max_iter, nsh_max_row_nnz, nsh_max_iter, mr_col_version;
   NALU_HYPRE_Real mr_tol, nsh_tol;
   NALU_HYPRE_Real *droptol = nalu_hypre_CTAlloc(NALU_HYPRE_Real, 2, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParCSRMatrix *approx_A_inv = NULL;

   print_level = 0;
   nsh_max_iter = 2;
   nsh_max_row_nnz = 2; // default 1000
   mr_max_iter = 1;
   mr_tol = 1.0e-3;
   mr_max_row_nnz = 2; // default 800
   mr_col_version = 0;
   nsh_tol = 1.0e-3;
   droptol[0] = 1.0e-2;
   droptol[1] = 1.0e-2;

   nalu_hypre_ILUParCSRInverseNSH(A, &approx_A_inv, droptol, mr_tol, nsh_tol, NALU_HYPRE_REAL_MIN,
                             mr_max_row_nnz,
                             nsh_max_row_nnz, mr_max_iter, nsh_max_iter, mr_col_version, print_level);
   *A_inv = approx_A_inv;

   if (droptol) { nalu_hypre_TFree(droptol, NALU_HYPRE_MEMORY_HOST); }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_MGRBuildInterpApproximateInverse(nalu_hypre_ParCSRMatrix   *A,
                                       NALU_HYPRE_Int            *CF_marker,
                                       NALU_HYPRE_BigInt            *num_cpts_global,
                                       NALU_HYPRE_Int            debug_flag,
                                       nalu_hypre_ParCSRMatrix   **P_ptr)
{
   NALU_HYPRE_Int            *C_marker;
   NALU_HYPRE_Int            *F_marker;
   nalu_hypre_ParCSRMatrix   *A_ff;
   nalu_hypre_ParCSRMatrix   *A_fc;
   nalu_hypre_ParCSRMatrix   *A_ff_inv;
   nalu_hypre_ParCSRMatrix   *W;
   MPI_Comm        comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRMatrix    *P;
   NALU_HYPRE_BigInt         *col_map_offd_P;
   NALU_HYPRE_Real      *P_diag_data;
   NALU_HYPRE_Int             *P_diag_i;
   NALU_HYPRE_Int             *P_diag_j;
   NALU_HYPRE_Int             *P_offd_i;
   NALU_HYPRE_Int              P_diag_nnz;
   NALU_HYPRE_Int              n_fine = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));
   NALU_HYPRE_BigInt              total_global_cpts;
   NALU_HYPRE_Int              num_cols_P_offd;

   NALU_HYPRE_Int              i;

   NALU_HYPRE_Real      m_one = -1.0;

   NALU_HYPRE_Int              my_id;
   NALU_HYPRE_Int              num_procs;

   NALU_HYPRE_MemoryLocation memory_location_P = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   C_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   F_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);

   // create C and F markers
   for (i = 0; i < n_fine; i++)
   {
      C_marker[i] = (CF_marker[i] == 1) ? 1 : -1;
      F_marker[i] = (CF_marker[i] == 1) ? -1 : 1;
   }

   // Get A_FF
   nalu_hypre_MGRGetSubBlock(A, F_marker, F_marker, 0, &A_ff);
   //  nalu_hypre_ParCSRMatrixPrintIJ(A_ff, 1, 1, "A_ff");
   // Get A_FC
   nalu_hypre_MGRGetSubBlock(A, F_marker, C_marker, 0, &A_fc);

   nalu_hypre_MGRApproximateInverse(A_ff, &A_ff_inv);
   //  nalu_hypre_ParCSRMatrixPrintIJ(A_ff_inv, 1, 1, "A_ff_inv");
   //  nalu_hypre_ParCSRMatrixPrintIJ(A_fc, 1, 1, "A_fc");
   W = nalu_hypre_ParMatmul(A_ff_inv, A_fc);
   nalu_hypre_ParCSRMatrixScale(W, m_one);
   //  nalu_hypre_ParCSRMatrixPrintIJ(W, 1, 1, "Wp");

   nalu_hypre_CSRMatrix *W_diag = nalu_hypre_ParCSRMatrixDiag(W);

   nalu_hypre_CSRMatrix *W_offd         = nalu_hypre_ParCSRMatrixOffd(W);

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/

   P_diag_nnz  = nalu_hypre_CSRMatrixNumNonzeros(W_diag) + nalu_hypre_CSRMatrixNumCols(W_diag);
   P_diag_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine + 1, memory_location_P);
   P_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_diag_nnz, memory_location_P);
   P_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  P_diag_nnz, memory_location_P);
   P_offd_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine + 1, memory_location_P);

   /* Extend W data to P data */
   nalu_hypre_ExtendWtoPHost( n_fine,
                         CF_marker,
                         nalu_hypre_CSRMatrixI(W_diag),
                         nalu_hypre_CSRMatrixJ(W_diag),
                         nalu_hypre_CSRMatrixData(W_diag),
                         P_diag_i,
                         P_diag_j,
                         P_diag_data,
                         nalu_hypre_CSRMatrixI(W_offd),
                         P_offd_i );
   // final P
   P = nalu_hypre_ParCSRMatrixCreate(comm,
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                nalu_hypre_ParCSRMatrixColStarts(A),
                                num_cpts_global,
                                nalu_hypre_CSRMatrixNumCols(W_offd),
                                P_diag_nnz,
                                nalu_hypre_CSRMatrixNumNonzeros(W_offd) );

   nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixDiag(P)) = memory_location_P;
   nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixOffd(P)) = memory_location_P;

   nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(P))    = P_diag_i;
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(P))    = P_diag_j;
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(P)) = P_diag_data;

   nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(P))    = P_offd_i;
   nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(P))    = nalu_hypre_CSRMatrixJ(W_offd);
   nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(P)) = nalu_hypre_CSRMatrixData(W_offd);
   nalu_hypre_CSRMatrixJ(W_offd)    = NULL;
   nalu_hypre_CSRMatrixData(W_offd) = NULL;

   num_cols_P_offd = nalu_hypre_CSRMatrixNumCols(W_offd);
   NALU_HYPRE_BigInt *col_map_offd_tmp = nalu_hypre_ParCSRMatrixColMapOffd(W);
   if (nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(P)))
   {
      col_map_offd_P = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_P_offd, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_cols_P_offd; i++)
      {
         col_map_offd_P[i] = col_map_offd_tmp[i];
      }
   }

   if (num_cols_P_offd)
   {
      nalu_hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(P)) = num_cols_P_offd;
   }
   nalu_hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   nalu_hypre_TFree(C_marker, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(F_marker, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParCSRMatrixDestroy(A_ff);
   nalu_hypre_ParCSRMatrixDestroy(A_fc);
   nalu_hypre_ParCSRMatrixDestroy(A_ff_inv);
   nalu_hypre_ParCSRMatrixDestroy(W);

   return 0;
}

/* Setup interpolation operator */
NALU_HYPRE_Int
nalu_hypre_MGRBuildInterp(nalu_hypre_ParCSRMatrix   *A,
                     nalu_hypre_ParCSRMatrix   *A_FF,
                     nalu_hypre_ParCSRMatrix   *A_FC,
                     NALU_HYPRE_Int            *CF_marker,
                     nalu_hypre_ParCSRMatrix   *aux_mat,
                     NALU_HYPRE_BigInt         *num_cpts_global,
                     NALU_HYPRE_Int             num_functions,
                     NALU_HYPRE_Int            *dof_func,
                     NALU_HYPRE_Int             debug_flag,
                     NALU_HYPRE_Real            trunc_factor,
                     NALU_HYPRE_Int             max_elmts,
                     NALU_HYPRE_Int             blk_size,
                     nalu_hypre_ParCSRMatrix   **P,
                     NALU_HYPRE_Int             interp_type,
                     NALU_HYPRE_Int             numsweeps)
{
   //  NALU_HYPRE_Int i;
   nalu_hypre_ParCSRMatrix    *P_ptr = NULL;
   //NALU_HYPRE_Real       jac_trunc_threshold = trunc_factor;
   //NALU_HYPRE_Real       jac_trunc_threshold_minus = 0.5*jac_trunc_threshold;
#if defined (NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );
#endif

   /* Interpolation for each level */
   if (interp_type < 3)
   {
#if defined (NALU_HYPRE_USING_GPU)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         nalu_hypre_MGRBuildPDevice(A, CF_marker, num_cpts_global, interp_type, &P_ptr);
         //nalu_hypre_ParCSRMatrixPrintIJ(P_ptr, 0, 0, "P_device");
      }
      else
#endif
      {
         //      nalu_hypre_MGRBuildP(A, CF_marker, num_cpts_global, interp_type, debug_flag, &P_ptr);
         nalu_hypre_MGRBuildPHost(A, CF_marker, num_cpts_global, interp_type, &P_ptr);
         //nalu_hypre_ParCSRMatrixPrintIJ(P_ptr, 0, 0, "P_host");
      }
      /* Could do a few sweeps of Jacobi to further improve Jacobi interpolation P */
      /*
          if(interp_type == 2)
          {
             for(i=0; i<numsweeps; i++)
             {
               nalu_hypre_BoomerAMGJacobiInterp(A, &P_ptr, S,1, NULL, CF_marker, 0, jac_trunc_threshold, jac_trunc_threshold_minus );
             }
             nalu_hypre_BoomerAMGInterpTruncation(P_ptr, trunc_factor, max_elmts);
          }
      */
   }
   else if (interp_type == 4)
   {
#if defined (NALU_HYPRE_USING_GPU)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         nalu_hypre_NoGPUSupport("interpolation");
      }
      else
#endif
      {
         nalu_hypre_MGRBuildInterpApproximateInverse(A, CF_marker, num_cpts_global,
                                                debug_flag, &P_ptr);
         nalu_hypre_BoomerAMGInterpTruncation(P_ptr, trunc_factor, max_elmts);
      }
   }
   else if (interp_type == 5)
   {
      nalu_hypre_BoomerAMGBuildModExtInterp(A, CF_marker, aux_mat, num_cpts_global,
                                       1, NULL, debug_flag, trunc_factor, max_elmts,
                                       &P_ptr);
   }
   else if (interp_type == 6)
   {
      nalu_hypre_BoomerAMGBuildModExtPIInterp(A, CF_marker, aux_mat, num_cpts_global,
                                         1, NULL, debug_flag, trunc_factor, max_elmts,
                                         &P_ptr);
   }
   else if (interp_type == 7)
   {
      nalu_hypre_BoomerAMGBuildModExtPEInterp(A, CF_marker, aux_mat, num_cpts_global,
                                         1, NULL, debug_flag, trunc_factor, max_elmts,
                                         &P_ptr);
   }
   else if (interp_type == 12)
   {
      nalu_hypre_MGRBuildPBlockJacobi(A, A_FF, A_FC, aux_mat, blk_size, CF_marker,
                                 num_cpts_global, debug_flag, &P_ptr);
   }
   else
   {
      /* Classical modified interpolation */
      nalu_hypre_BoomerAMGBuildInterp(A, CF_marker, aux_mat, num_cpts_global,
                                 1, NULL, debug_flag, trunc_factor, max_elmts,
                                 &P_ptr);
   }

   /* set pointer to P */
   *P = P_ptr;

   return nalu_hypre_error_flag;
}

/* Setup restriction operator. TODO: Change R -> RT (VPM) */
NALU_HYPRE_Int
nalu_hypre_MGRBuildRestrict( nalu_hypre_ParCSRMatrix    *A,
                        nalu_hypre_ParCSRMatrix    *A_FF,
                        nalu_hypre_ParCSRMatrix    *A_FC,
                        NALU_HYPRE_Int             *CF_marker,
                        NALU_HYPRE_BigInt          *num_cpts_global,
                        NALU_HYPRE_Int              num_functions,
                        NALU_HYPRE_Int             *dof_func,
                        NALU_HYPRE_Int              debug_flag,
                        NALU_HYPRE_Real             trunc_factor,
                        NALU_HYPRE_Int              max_elmts,
                        NALU_HYPRE_Real             strong_threshold,
                        NALU_HYPRE_Real             max_row_sum,
                        NALU_HYPRE_Int              blk_size,
                        nalu_hypre_ParCSRMatrix   **R_ptr,
                        NALU_HYPRE_Int              restrict_type,
                        NALU_HYPRE_Int              numsweeps )
{
   //   NALU_HYPRE_Int i;
   nalu_hypre_ParCSRMatrix    *R = NULL;
   nalu_hypre_ParCSRMatrix    *AT = NULL;
   nalu_hypre_ParCSRMatrix    *A_FFT = NULL;
   nalu_hypre_ParCSRMatrix    *A_FCT = NULL;
   nalu_hypre_ParCSRMatrix    *ST = NULL;
   //   NALU_HYPRE_Real       jac_trunc_threshold = trunc_factor;
   //   NALU_HYPRE_Real       jac_trunc_threshold_minus = 0.5*jac_trunc_threshold;
#if defined (NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );
#endif

   /* Build AT (transpose A) */
   if (restrict_type > 0)
   {
      nalu_hypre_ParCSRMatrixTranspose(A, &AT, 1);

      if (A_FF)
      {
         nalu_hypre_ParCSRMatrixTranspose(A_FF, &A_FFT, 1);
      }

      if (A_FC)
      {
         nalu_hypre_ParCSRMatrixTranspose(A_FC, &A_FCT, 1);
      }
   }

   /* Restriction for each level */
   if (restrict_type == 0)
   {
#if defined (NALU_HYPRE_USING_GPU)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         nalu_hypre_MGRBuildPDevice(A, CF_marker, num_cpts_global, restrict_type, &R);
         //nalu_hypre_ParCSRMatrixPrintIJ(R, 0, 0, "R_device");
      }
      else
#endif
      {
         nalu_hypre_MGRBuildP(A, CF_marker, num_cpts_global, restrict_type, debug_flag, &R);
         //nalu_hypre_ParCSRMatrixPrintIJ(R, 0, 0, "R_host");
      }
   }
   else if (restrict_type == 1 || restrict_type == 2)
   {
#if defined (NALU_HYPRE_USING_GPU)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         nalu_hypre_MGRBuildPDevice(AT, CF_marker, num_cpts_global, restrict_type, &R);
         //nalu_hypre_ParCSRMatrixPrintIJ(R, 0, 0, "R_device");
      }
      else
#endif
      {
         nalu_hypre_MGRBuildP(AT, CF_marker, num_cpts_global, restrict_type, debug_flag, &R);
         //nalu_hypre_ParCSRMatrixPrintIJ(R, 0, 0, "R_host");
      }
   }
   else if (restrict_type == 3)
   {
      /* move diagonal to first entry */
      nalu_hypre_CSRMatrixReorder(nalu_hypre_ParCSRMatrixDiag(AT));
      nalu_hypre_MGRBuildInterpApproximateInverse(AT, CF_marker, num_cpts_global, debug_flag, &R);
      nalu_hypre_BoomerAMGInterpTruncation(R, trunc_factor, max_elmts);
   }
   else if (restrict_type == 12)
   {
      nalu_hypre_MGRBuildPBlockJacobi(AT, A_FFT, A_FCT, NULL, blk_size, CF_marker,
                                 num_cpts_global, debug_flag, &R);
   }
   else if (restrict_type == 13) // CPR-like restriction operator
   {
      /* TODO: create a function with this block (VPM) */

      nalu_hypre_ParCSRMatrix *blk_A_cf = NULL;
      nalu_hypre_ParCSRMatrix *blk_A_cf_transpose = NULL;
      nalu_hypre_ParCSRMatrix *Wr_transpose = NULL;
      nalu_hypre_ParCSRMatrix *blk_A_ff_inv_transpose = NULL;
      NALU_HYPRE_Int *c_marker = NULL;
      NALU_HYPRE_Int *f_marker = NULL;
      NALU_HYPRE_Int i;
      NALU_HYPRE_Int nrows = nalu_hypre_ParCSRMatrixNumRows(A);

      NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

      /* TODO: Port this to GPU (VPM) */
      /* create C and F markers to extract A_CF */
      c_marker = CF_marker;
      f_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows, memory_location);
      for (i = 0; i < nrows; i++)
      {
         f_marker[i] = - CF_marker[i];
      }

#if defined (NALU_HYPRE_USING_GPU)
      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         nalu_hypre_NoGPUSupport("restriction");
      }
      else
#endif
      {
         /* get block A_cf */
         nalu_hypre_MGRGetAcfCPR(A, blk_size, c_marker, f_marker, &blk_A_cf);

         /* transpose block A_cf */
         nalu_hypre_ParCSRMatrixTranspose(blk_A_cf, &blk_A_cf_transpose, 1);

         /* compute block diagonal A_ff */
         nalu_hypre_ParCSRMatrixBlockDiagMatrix(AT, blk_size, -1, CF_marker, 1,
                                           &blk_A_ff_inv_transpose);

         /* compute  Wr = A^{-T} * A_cf^{T}  */
         Wr_transpose = nalu_hypre_ParCSRMatMat(blk_A_ff_inv_transpose, blk_A_cf_transpose);

         /* compute restriction operator R = [-Wr  I] (transposed for use with RAP) */
         nalu_hypre_MGRBuildPFromWp(AT, Wr_transpose, CF_marker, debug_flag, &R);
      }
      nalu_hypre_ParCSRMatrixDestroy(blk_A_cf);
      nalu_hypre_ParCSRMatrixDestroy(blk_A_cf_transpose);
      nalu_hypre_ParCSRMatrixDestroy(Wr_transpose);
      nalu_hypre_ParCSRMatrixDestroy(blk_A_ff_inv_transpose);
      nalu_hypre_TFree(f_marker, memory_location);
   }
   else
   {
      /* Build new strength matrix */
      nalu_hypre_BoomerAMGCreateS(AT, strong_threshold, max_row_sum, 1, NULL, &ST);

      /* Classical modified interpolation */
      nalu_hypre_BoomerAMGBuildInterp(AT, CF_marker, ST, num_cpts_global, 1, NULL, debug_flag,
                                 trunc_factor, max_elmts, &R);
   }

   /* Compute R^T so it can be used in the solve phase */
   if (!nalu_hypre_ParCSRMatrixDiagT(R))
   {
      nalu_hypre_CSRMatrixTranspose(nalu_hypre_ParCSRMatrixDiag(R), &nalu_hypre_ParCSRMatrixDiagT(R), 1);
   }
   if (!nalu_hypre_ParCSRMatrixOffdT(R))
   {
      nalu_hypre_CSRMatrixTranspose(nalu_hypre_ParCSRMatrixOffd(R), &nalu_hypre_ParCSRMatrixOffdT(R), 1);
   }

   /* Set pointer to R */
   *R_ptr = R;

   /* Free memory */
   if (restrict_type > 0)
   {
      nalu_hypre_ParCSRMatrixDestroy(AT);
      nalu_hypre_ParCSRMatrixDestroy(A_FFT);
      nalu_hypre_ParCSRMatrixDestroy(A_FCT);
   }
   if (restrict_type > 5)
   {
      nalu_hypre_ParCSRMatrixDestroy(ST);
   }

   return nalu_hypre_error_flag;
}

/* TODO: move matrix inversion functions outside parcsr_ls (VPM) */

void nalu_hypre_blas_smat_inv_n2 (NALU_HYPRE_Real *a)
{
   const NALU_HYPRE_Real a11 = a[0], a12 = a[1];
   const NALU_HYPRE_Real a21 = a[2], a22 = a[3];
   const NALU_HYPRE_Real det_inv = 1.0 / (a11 * a22 - a12 * a21);
   a[0] = a22 * det_inv; a[1] = -a12 * det_inv;
   a[2] = -a21 * det_inv; a[3] = a11 * det_inv;
}

void nalu_hypre_blas_smat_inv_n3 (NALU_HYPRE_Real *a)
{
   const NALU_HYPRE_Real a11 = a[0],  a12 = a[1],  a13 = a[2];
   const NALU_HYPRE_Real a21 = a[3],  a22 = a[4],  a23 = a[5];
   const NALU_HYPRE_Real a31 = a[6],  a32 = a[7],  a33 = a[8];

   const NALU_HYPRE_Real det = a11 * a22 * a33 - a11 * a23 * a32 - a12 * a21 * a33 + a12 * a23 * a31 + a13 *
                          a21 * a32 - a13 * a22 * a31;
   const NALU_HYPRE_Real det_inv = 1.0 / det;

   a[0] = (a22 * a33 - a23 * a32) * det_inv; a[1] = (a13 * a32 - a12 * a33) * det_inv;
   a[2] = (a12 * a23 - a13 * a22) * det_inv;
   a[3] = (a23 * a31 - a21 * a33) * det_inv; a[4] = (a11 * a33 - a13 * a31) * det_inv;
   a[5] = (a13 * a21 - a11 * a23) * det_inv;
   a[6] = (a21 * a32 - a22 * a31) * det_inv; a[7] = (a12 * a31 - a11 * a32) * det_inv;
   a[8] = (a11 * a22 - a12 * a21) * det_inv;
}

void nalu_hypre_blas_smat_inv_n4 (NALU_HYPRE_Real *a)
{
   const NALU_HYPRE_Real a11 = a[0],  a12 = a[1],  a13 = a[2],  a14 = a[3];
   const NALU_HYPRE_Real a21 = a[4],  a22 = a[5],  a23 = a[6],  a24 = a[7];
   const NALU_HYPRE_Real a31 = a[8],  a32 = a[9],  a33 = a[10], a34 = a[11];
   const NALU_HYPRE_Real a41 = a[12], a42 = a[13], a43 = a[14], a44 = a[15];

   const NALU_HYPRE_Real M11 = a22 * a33 * a44 + a23 * a34 * a42 + a24 * a32 * a43 - a22 * a34 * a43 - a23 *
                          a32 * a44 - a24 * a33 * a42;
   const NALU_HYPRE_Real M12 = a12 * a34 * a43 + a13 * a32 * a44 + a14 * a33 * a42 - a12 * a33 * a44 - a13 *
                          a34 * a42 - a14 * a32 * a43;
   const NALU_HYPRE_Real M13 = a12 * a23 * a44 + a13 * a24 * a42 + a14 * a22 * a43 - a12 * a24 * a43 - a13 *
                          a22 * a44 - a14 * a23 * a42;
   const NALU_HYPRE_Real M14 = a12 * a24 * a33 + a13 * a22 * a34 + a14 * a23 * a32 - a12 * a23 * a34 - a13 *
                          a24 * a32 - a14 * a22 * a33;
   const NALU_HYPRE_Real M21 = a21 * a34 * a43 + a23 * a31 * a44 + a24 * a33 * a41 - a21 * a33 * a44 - a23 *
                          a34 * a41 - a24 * a31 * a43;
   const NALU_HYPRE_Real M22 = a11 * a33 * a44 + a13 * a34 * a41 + a14 * a31 * a43 - a11 * a34 * a43 - a13 *
                          a31 * a44 - a14 * a33 * a41;
   const NALU_HYPRE_Real M23 = a11 * a24 * a43 + a13 * a21 * a44 + a14 * a23 * a41 - a11 * a23 * a44 - a13 *
                          a24 * a41 - a14 * a21 * a43;
   const NALU_HYPRE_Real M24 = a11 * a23 * a34 + a13 * a24 * a31 + a14 * a21 * a33 - a11 * a24 * a33 - a13 *
                          a21 * a34 - a14 * a23 * a31;
   const NALU_HYPRE_Real M31 = a21 * a32 * a44 + a22 * a34 * a41 + a24 * a31 * a42 - a21 * a34 * a42 - a22 *
                          a31 * a44 - a24 * a32 * a41;
   const NALU_HYPRE_Real M32 = a11 * a34 * a42 + a12 * a31 * a44 + a14 * a32 * a41 - a11 * a32 * a44 - a12 *
                          a34 * a41 - a14 * a31 * a42;
   const NALU_HYPRE_Real M33 = a11 * a22 * a44 + a12 * a24 * a41 + a14 * a21 * a42 - a11 * a24 * a42 - a12 *
                          a21 * a44 - a14 * a22 * a41;
   const NALU_HYPRE_Real M34 = a11 * a24 * a32 + a12 * a21 * a34 + a14 * a22 * a31 - a11 * a22 * a34 - a12 *
                          a24 * a31 - a14 * a21 * a32;
   const NALU_HYPRE_Real M41 = a21 * a33 * a42 + a22 * a31 * a43 + a23 * a32 * a41 - a21 * a32 * a43 - a22 *
                          a33 * a41 - a23 * a31 * a42;
   const NALU_HYPRE_Real M42 = a11 * a32 * a43 + a12 * a33 * a41 + a13 * a31 * a42 - a11 * a33 * a42 - a12 *
                          a31 * a43 - a13 * a32 * a41;
   const NALU_HYPRE_Real M43 = a11 * a23 * a42 + a12 * a21 * a43 + a13 * a22 * a41 - a11 * a22 * a43 - a12 *
                          a23 * a41 - a13 * a21 * a42;
   const NALU_HYPRE_Real M44 = a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a11 * a23 * a32 - a12 *
                          a21 * a33 - a13 * a22 * a31;

   const NALU_HYPRE_Real det = a11 * M11 + a12 * M21 + a13 * M31 + a14 * M41;
   NALU_HYPRE_Real det_inv;

   //if ( nalu_hypre_abs(det) < 1e-22 ) {
   //nalu_hypre_printf("### WARNING: Matrix is nearly singular! det = %e\n", det);
   /*
   printf("##----------------------------------------------\n");
   printf("## %12.5e %12.5e %12.5e \n", a0, a1, a2);
   printf("## %12.5e %12.5e %12.5e \n", a3, a4, a5);
   printf("## %12.5e %12.5e %12.5e \n", a5, a6, a7);
   printf("##----------------------------------------------\n");
   getchar();
   */
   //}

   det_inv = 1.0 / det;

   a[0] = M11 * det_inv;  a[1] = M12 * det_inv;  a[2] = M13 * det_inv;  a[3] = M14 * det_inv;
   a[4] = M21 * det_inv;  a[5] = M22 * det_inv;  a[6] = M23 * det_inv;  a[7] = M24 * det_inv;
   a[8] = M31 * det_inv;  a[9] = M32 * det_inv;  a[10] = M33 * det_inv; a[11] = M34 * det_inv;
   a[12] = M41 * det_inv; a[13] = M42 * det_inv; a[14] = M43 * det_inv; a[15] = M44 * det_inv;

}

void nalu_hypre_MGRSmallBlkInverse(NALU_HYPRE_Real *mat,
                              NALU_HYPRE_Int  blk_size)
{
   if (blk_size == 2)
   {
      nalu_hypre_blas_smat_inv_n2(mat);
   }
   else if (blk_size == 3)
   {
      nalu_hypre_blas_smat_inv_n3(mat);
   }
   else if (blk_size == 4)
   {
      nalu_hypre_blas_smat_inv_n4(mat);
   }
}

void nalu_hypre_blas_mat_inv(NALU_HYPRE_Real *a,
                        NALU_HYPRE_Int n)
{
   NALU_HYPRE_Int i, j, k, l, u, kn, in;
   NALU_HYPRE_Real alinv;
   if (n == 4)
   {
      nalu_hypre_blas_smat_inv_n4(a);
   }
   else
   {
      for (k = 0; k < n; ++k)
      {
         kn = k * n;
         l  = kn + k;

         //if (nalu_hypre_abs(a[l]) < NALU_HYPRE_REAL_MIN) {
         //   printf("### WARNING: Diagonal entry is close to zero!");
         //   printf("### WARNING: diag_%d=%e\n", k, a[l]);
         //   a[l] = NALU_HYPRE_REAL_MIN;
         //}
         alinv = 1.0 / a[l];
         a[l] = alinv;

         for (j = 0; j < k; ++j)
         {
            u = kn + j; a[u] *= alinv;
         }

         for (j = k + 1; j < n; ++j)
         {
            u = kn + j; a[u] *= alinv;
         }

         for (i = 0; i < k; ++i)
         {
            in = i * n;
            for (j = 0; j < n; ++j)
               if (j != k)
               {
                  u = in + j; a[u] -= a[in + k] * a[kn + j];
               } // end if (j!=k)
         }

         for (i = k + 1; i < n; ++i)
         {
            in = i * n;
            for (j = 0; j < n; ++j)
               if (j != k)
               {
                  u = in + j; a[u] -= a[in + k] * a[kn + j];
               } // end if (j!=k)
         }

         for (i = 0; i < k; ++i)
         {
            u = i * n + k; a[u] *= -alinv;
         }

         for (i = k + 1; i < n; ++i)
         {
            u = i * n + k; a[u] *= -alinv;
         }
      } // end for (k=0; k<n; ++k)
   }// end if
}

NALU_HYPRE_Int
nalu_hypre_block_jacobi_solve( nalu_hypre_ParCSRMatrix *A,
                          nalu_hypre_ParVector    *f,
                          nalu_hypre_ParVector    *u,
                          NALU_HYPRE_Int           blk_size,
                          NALU_HYPRE_Int           method,
                          NALU_HYPRE_Real         *diaginv,
                          nalu_hypre_ParVector    *Vtemp )
{
   MPI_Comm      comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data  = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i     = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j     = nalu_hypre_CSRMatrixJ(A_diag);
   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int       *A_offd_i     = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Real      *A_offd_data  = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_j     = nalu_hypre_CSRMatrixJ(A_offd);
   nalu_hypre_ParCSRCommPkg  *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle *comm_handle;

   NALU_HYPRE_Int        n       = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int        num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_Vector    *u_local = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Real      *u_data  = nalu_hypre_VectorData(u_local);

   nalu_hypre_Vector    *f_local = nalu_hypre_ParVectorLocalVector(f);
   NALU_HYPRE_Real      *f_data  = nalu_hypre_VectorData(f_local);

   nalu_hypre_Vector    *Vtemp_local = nalu_hypre_ParVectorLocalVector(Vtemp);
   NALU_HYPRE_Real      *Vtemp_data = nalu_hypre_VectorData(Vtemp_local);
   NALU_HYPRE_Real      *Vext_data = NULL;
   NALU_HYPRE_Real      *v_buf_data;

   NALU_HYPRE_Int        i, j, k;
   NALU_HYPRE_Int        ii, jj;
   NALU_HYPRE_Int        bidx, bidx1;
   NALU_HYPRE_Int        num_sends;
   NALU_HYPRE_Int        index, start;
   NALU_HYPRE_Int        num_procs, my_id;
   NALU_HYPRE_Real      *res;

   const NALU_HYPRE_Int  nb2 = blk_size * blk_size;
   const NALU_HYPRE_Int  n_block = n / blk_size;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   //   NALU_HYPRE_Int num_threads = nalu_hypre_NumThreads();

   res = nalu_hypre_CTAlloc(NALU_HYPRE_Real, blk_size, NALU_HYPRE_MEMORY_HOST);

   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   if (num_procs > 1)
   {
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);

      v_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,
                                 nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends),
                                 NALU_HYPRE_MEMORY_HOST);

      Vext_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      if (num_cols_offd)
      {
         A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
         A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
      }

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            v_buf_data[index++] = u_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg, v_buf_data, Vext_data);
   }

   /*-----------------------------------------------------------------
   * Copy current approximation into temporary vector.
   *-----------------------------------------------------------------*/

#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < n; i++)
   {
      Vtemp_data[i] = u_data[i];
      //printf("u_old[%d] = %e\n",i,Vtemp_data[i]);
   }
   if (num_procs > 1)
   {
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   /*-----------------------------------------------------------------
   * Relax points block by block
   *-----------------------------------------------------------------*/
   for (i = 0; i < n_block; i++)
   {
      for (j = 0; j < blk_size; j++)
      {
         bidx = i * blk_size + j;
         res[j] = f_data[bidx];
         for (jj = A_diag_i[bidx]; jj < A_diag_i[bidx + 1]; jj++)
         {
            ii = A_diag_j[jj];
            if (method == 0)
            {
               // Jacobi for diagonal part
               res[j] -= A_diag_data[jj] * Vtemp_data[ii];
            }
            else if (method == 1)
            {
               // Gauss-Seidel for diagonal part
               res[j] -= A_diag_data[jj] * u_data[ii];
            }
            else
            {
               // Default do Jacobi for diagonal part
               res[j] -= A_diag_data[jj] * Vtemp_data[ii];
            }
            //printf("%d: Au= %e * %e =%e\n",ii,A_diag_data[jj],Vtemp_data[ii], res[j]);
         }
         for (jj = A_offd_i[bidx]; jj < A_offd_i[bidx + 1]; jj++)
         {
            // always do Jacobi for off-diagonal part
            ii = A_offd_j[jj];
            res[j] -= A_offd_data[jj] * Vext_data[ii];
         }
         //printf("%d: res = %e\n",bidx,res[j]);
      }

      for (j = 0; j < blk_size; j++)
      {
         bidx1 = i * blk_size + j;
         for (k = 0; k < blk_size; k++)
         {
            bidx  = i * nb2 + j * blk_size + k;
            u_data[bidx1] += res[k] * diaginv[bidx];
            //printf("u[%d] = %e, diaginv[%d] = %e\n",bidx1,u_data[bidx1],bidx,diaginv[bidx]);
         }
         //printf("u[%d] = %e\n",bidx1,u_data[bidx1]);
      }
   }

   if (num_procs > 1)
   {
      nalu_hypre_TFree(Vext_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(v_buf_data, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(res, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRBlockRelaxSolveDevice
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRBlockRelaxSolveDevice( nalu_hypre_ParCSRMatrix  *B,
                                nalu_hypre_ParCSRMatrix  *A,
                                nalu_hypre_ParVector     *f,
                                nalu_hypre_ParVector     *u,
                                nalu_hypre_ParVector     *Vtemp,
                                NALU_HYPRE_Real           relax_weight )
{
   nalu_hypre_GpuProfilingPushRange("BlockRelaxSolve");

   /* Copy f into temporary vector */
   nalu_hypre_ParVectorCopy(f, Vtemp);

   /* Perform Matvec: Vtemp = w * (f - Au) */
   if (nalu_hypre_ParVectorAllZeros(u))
   {
#if defined(NALU_HYPRE_DEBUG)
      nalu_hypre_assert(nalu_hypre_ParVectorInnerProd(u, u) == 0.0);
#endif
      nalu_hypre_ParVectorScale(relax_weight, Vtemp);
   }
   else
   {
      nalu_hypre_ParCSRMatrixMatvec(-relax_weight, A, u, relax_weight, Vtemp);
   }

   /* Update solution: u += B * Vtemp */
   nalu_hypre_ParCSRMatrixMatvec(1.0, B, Vtemp, 1.0, u);
   nalu_hypre_ParVectorAllZeros(u) = 0;

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRBlockRelaxSolve
 *
 * Computes a block Jacobi relaxation of matrix A, given the inverse of the
 * diagonal blocks (of A) obtained by calling nalu_hypre_MGRBlockRelaxSetup.
 *
 * TODO: Adapt to relax on specific points based on CF_marker information
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRBlockRelaxSolve( nalu_hypre_ParCSRMatrix *A,
                          nalu_hypre_ParVector    *f,
                          nalu_hypre_ParVector    *u,
                          NALU_HYPRE_Int           blk_size,
                          NALU_HYPRE_Int           n_block,
                          NALU_HYPRE_Int           left_size,
                          NALU_HYPRE_Int           method,
                          NALU_HYPRE_Real         *diaginv,
                          nalu_hypre_ParVector    *Vtemp )
{
   MPI_Comm      comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data  = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i     = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j     = nalu_hypre_CSRMatrixJ(A_diag);
   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int       *A_offd_i     = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Real      *A_offd_data  = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_j     = nalu_hypre_CSRMatrixJ(A_offd);
   nalu_hypre_ParCSRCommPkg  *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle *comm_handle;

   NALU_HYPRE_Int        n       = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int        num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_Vector    *u_local = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Real      *u_data  = nalu_hypre_VectorData(u_local);

   nalu_hypre_Vector    *f_local = nalu_hypre_ParVectorLocalVector(f);
   NALU_HYPRE_Real      *f_data  = nalu_hypre_VectorData(f_local);

   nalu_hypre_Vector    *Vtemp_local = nalu_hypre_ParVectorLocalVector(Vtemp);
   NALU_HYPRE_Real      *Vtemp_data = nalu_hypre_VectorData(Vtemp_local);
   NALU_HYPRE_Real      *Vext_data = NULL;
   NALU_HYPRE_Real      *v_buf_data;

   NALU_HYPRE_Int        i, j, k;
   NALU_HYPRE_Int        ii, jj;
   NALU_HYPRE_Int        bidx, bidx1, bidxm1;
   NALU_HYPRE_Int        num_sends;
   NALU_HYPRE_Int        index, start;
   NALU_HYPRE_Int        num_procs, my_id;
   NALU_HYPRE_Real      *res;

   const NALU_HYPRE_Int  nb2 = blk_size * blk_size;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   //   NALU_HYPRE_Int num_threads = nalu_hypre_NumThreads();

   res = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  blk_size, NALU_HYPRE_MEMORY_HOST);

   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   if (num_procs > 1)
   {
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);

      v_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,
                                 nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends),
                                 NALU_HYPRE_MEMORY_HOST);

      Vext_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      if (num_cols_offd)
      {
         A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
         A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
      }

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            v_buf_data[index++]
               = u_data[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate( 1, comm_pkg, v_buf_data,
                                                  Vext_data);
   }

   /*-----------------------------------------------------------------
   * Copy current approximation into temporary vector.
   *-----------------------------------------------------------------*/

#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < n; i++)
   {
      Vtemp_data[i] = u_data[i];
      //printf("u_old[%d] = %e\n",i,Vtemp_data[i]);
   }
   if (num_procs > 1)
   {
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   /*-----------------------------------------------------------------
   * Relax points block by block
   *-----------------------------------------------------------------*/
   for (i = 0; i < n_block; i++)
   {
      bidxm1 = i * blk_size;
      for (j = 0; j < blk_size; j++)
      {
         bidx = bidxm1 + j;
         res[j] = f_data[bidx];
         for (jj = A_diag_i[bidx]; jj < A_diag_i[bidx + 1]; jj++)
         {
            ii = A_diag_j[jj];
            if (method == 0)
            {
               // Jacobi for diagonal part
               res[j] -= A_diag_data[jj] * Vtemp_data[ii];
            }
            else if (method == 1)
            {
               // Gauss-Seidel for diagonal part
               res[j] -= A_diag_data[jj] * u_data[ii];
            }
            else
            {
               // Default do Jacobi for diagonal part
               res[j] -= A_diag_data[jj] * Vtemp_data[ii];
            }
            //printf("%d: Au= %e * %e =%e\n",ii,A_diag_data[jj],Vtemp_data[ii], res[j]);
         }
         for (jj = A_offd_i[bidx]; jj < A_offd_i[bidx + 1]; jj++)
         {
            // always do Jacobi for off-diagonal part
            ii = A_offd_j[jj];
            res[j] -= A_offd_data[jj] * Vext_data[ii];
         }
         //printf("%d: res = %e\n",bidx,res[j]);
      }

      for (j = 0; j < blk_size; j++)
      {
         bidx1 = bidxm1 + j;
         for (k = 0; k < blk_size; k++)
         {
            bidx  = i * nb2 + j * blk_size + k;
            u_data[bidx1] += res[k] * diaginv[bidx];
            //printf("u[%d] = %e, diaginv[%d] = %e\n",bidx1,u_data[bidx1],bidx,diaginv[bidx]);
         }
         //printf("u[%d] = %e\n",bidx1,u_data[bidx1]);
      }
   }
   if (num_procs > 1)
   {
      nalu_hypre_TFree(Vext_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(v_buf_data, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(res, NALU_HYPRE_MEMORY_HOST);
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BlockDiagInvLapack(NALU_HYPRE_Real *diag, NALU_HYPRE_Int N, NALU_HYPRE_Int blk_size)
{
   NALU_HYPRE_Int nblock, left_size, i;
   //NALU_HYPRE_Int *IPIV = nalu_hypre_CTAlloc(NALU_HYPRE_Int, blk_size, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int LWORK = blk_size * blk_size;
   NALU_HYPRE_Real *WORK = nalu_hypre_CTAlloc(NALU_HYPRE_Real, LWORK, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int INFO;

   NALU_HYPRE_Real wall_time;
   NALU_HYPRE_Int my_id;
   MPI_Comm comm = nalu_hypre_MPI_COMM_WORLD;
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   nblock = N / blk_size;
   left_size = N - blk_size * nblock;
   NALU_HYPRE_Int *IPIV = nalu_hypre_CTAlloc(NALU_HYPRE_Int, blk_size, NALU_HYPRE_MEMORY_HOST);

   wall_time = time_getWallclockSeconds();
   if (blk_size >= 2 && blk_size <= 4)
   {
      for (i = 0; i < nblock; i++)
      {
         nalu_hypre_MGRSmallBlkInverse(diag + i * LWORK, blk_size);
         //nalu_hypre_blas_smat_inv_n2(diag+i*LWORK);
      }
   }
   else if (blk_size > 4)
   {
      for (i = 0; i < nblock; i++)
      {
         nalu_hypre_dgetrf(&blk_size, &blk_size, diag + i * LWORK, &blk_size, IPIV, &INFO);
         nalu_hypre_dgetri(&blk_size, diag + i * LWORK, &blk_size, IPIV, WORK, &LWORK, &INFO);
      }
   }

   // Left size
   if (left_size > 0)
   {
      nalu_hypre_dgetrf(&left_size, &left_size, diag + i * LWORK, &left_size, IPIV, &INFO);
      nalu_hypre_dgetri(&left_size, diag + i * LWORK, &left_size, IPIV, WORK, &LWORK, &INFO);
   }
   wall_time = time_getWallclockSeconds() - wall_time;
   //if (my_id == 0) nalu_hypre_printf("Proc = %d, Compute inverse time: %1.5f\n", my_id, wall_time);

   nalu_hypre_TFree(IPIV, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(WORK, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixExtractBlockDiagHost
 *
 * Extract the block diagonal part of a A or a principal submatrix of A
 * defined by a marker (point_type) in an associated CF_marker array.
 * The result is an array of (flattened) block diagonals.
 *
 * If CF marker array is NULL, it returns an array of the (flattened)
 * block diagonal of the entire matrix A.
 *
 * Options for diag_type are:
 *   diag_type = 1: return the inverse of the block diagonals
 *   otherwise    : return the block diagonals
 *
 * On return, blk_diag_size contains the size of the returned
 * (flattened) array. (i.e. nnz of extracted block diagonal)
 *
 * Input parameters are:
 *    A          - original ParCSR matrix
 *    blk_size   - Size of diagonal blocks to extract
 *    CF_marker  - Array prescribing submatrix from which to extract
 *                 block diagonals. Ignored if NULL.
 *    point_type - marker tag in CF_marker array to extract diagonal
 *    diag_type  - Type of block diagonal entries to return.
 *                 Currently supports block diagonal or inverse block
 *                 diagonal entries (diag_type = 1).
 *
 * Output parameters are:
 *      diag_ptr - Array of block diagonal entries
 * blk_diag_size - number of entries in extracted block diagonal
 *                 (size of diag_ptr).
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixExtractBlockDiagHost( nalu_hypre_ParCSRMatrix   *par_A,
                                        NALU_HYPRE_Int             blk_size,
                                        NALU_HYPRE_Int             num_points,
                                        NALU_HYPRE_Int             point_type,
                                        NALU_HYPRE_Int            *CF_marker,
                                        NALU_HYPRE_Int             diag_size,
                                        NALU_HYPRE_Int             diag_type,
                                        NALU_HYPRE_Real           *diag_data )
{
   nalu_hypre_CSRMatrix      *A_diag       = nalu_hypre_ParCSRMatrixDiag(par_A);
   NALU_HYPRE_Int             nrows        = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Complex        *A_diag_data  = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int            *A_diag_i     = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int            *A_diag_j     = nalu_hypre_CSRMatrixJ(A_diag);

   NALU_HYPRE_Int             i, j;
   NALU_HYPRE_Int             ii, jj;
   NALU_HYPRE_Int             bidx, bidxm1, bidxp1, ridx, didx;
   NALU_HYPRE_Int             row_offset;

   NALU_HYPRE_Int             whole_num_points, cnt, bstart;
   NALU_HYPRE_Int             bs2 = blk_size * blk_size;
   NALU_HYPRE_Int             num_blocks;
   NALU_HYPRE_Int             left_size = 0;

   // First count the number of points matching point_type in CF_marker
   num_blocks       = num_points / blk_size;
   whole_num_points = blk_size * num_blocks;
   left_size        = num_points - whole_num_points;
   bstart           = bs2 * num_blocks;

   /*-----------------------------------------------------------------
    * Get all the diagonal sub-blocks
    *-----------------------------------------------------------------*/

   NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "ExtractDiagSubBlocks");
   if (CF_marker == NULL)
   {
      // CF Marker is NULL. Consider all rows of matrix.
      for (i = 0; i < num_blocks; i++)
      {
         bidxm1 = i * blk_size;
         bidxp1 = (i + 1) * blk_size;

         for (j = 0; j < blk_size; j++)
         {
            for (ii = A_diag_i[bidxm1 + j]; ii < A_diag_i[bidxm1 + j + 1]; ii++)
            {
               jj = A_diag_j[ii];
               if ((jj >= bidxm1) &&
                   (jj < bidxp1)  &&
                   nalu_hypre_cabs(A_diag_data[ii]) > NALU_HYPRE_REAL_MIN)
               {
                  bidx = j * blk_size + jj - bidxm1;
                  diag_data[i * bs2 + bidx] = A_diag_data[ii];
               }
            }
         }
      }

      // deal with remaining points if any
      if (left_size)
      {
         bidxm1 = whole_num_points;
         bidxp1 = num_points;
         for (j = 0; j < left_size; j++)
         {
            for (ii = A_diag_i[bidxm1 + j]; ii < A_diag_i[bidxm1 + j + 1]; ii++)
            {
               jj = A_diag_j[ii];
               if ((jj >= bidxm1) &&
                   (jj < bidxp1)  &&
                   nalu_hypre_cabs(A_diag_data[ii]) > NALU_HYPRE_REAL_MIN)
               {
                  bidx = j * left_size + jj - bidxm1;
                  diag_data[bstart + bidx] = A_diag_data[ii];
               }
            }
         }
      }
   }
   else
   {
      // extract only block diagonal of submatrix defined by CF marker
      cnt = 0;
      row_offset = 0;
      for (i = 0; i < nrows; i++)
      {
         if (CF_marker[i] == point_type)
         {
            bidx = cnt / blk_size;
            ridx = cnt % blk_size;
            bidxm1 = bidx * blk_size;
            bidxp1 = (bidx + 1) * blk_size;
            for (ii = A_diag_i[i]; ii < A_diag_i[i + 1]; ii++)
            {
               jj = A_diag_j[ii];
               if (CF_marker[jj] == point_type)
               {
                  if ((jj - row_offset >= bidxm1) &&
                      (jj - row_offset < bidxp1)  &&
                      (nalu_hypre_cabs(A_diag_data[ii]) > NALU_HYPRE_REAL_MIN))
                  {
                     didx = bidx * bs2 + ridx * blk_size + jj - bidxm1 - row_offset;
                     diag_data[didx] = A_diag_data[ii];
                  }
               }
            }
            if (++cnt == whole_num_points)
            {
               break;
            }
         }
         else
         {
            row_offset++;
         }
      }

      // remaining points
      for (i = whole_num_points; i < num_points; i++)
      {
         if (CF_marker[i] == point_type)
         {
            bidx = num_blocks;
            ridx = cnt - whole_num_points;
            bidxm1 = whole_num_points;
            bidxp1 = num_points;
            for (ii = A_diag_i[i]; ii < A_diag_i[i + 1]; ii++)
            {
               jj = A_diag_j[ii];
               if (CF_marker[jj] == point_type)
               {
                  if ((jj - row_offset >= bidxm1) &&
                      (jj - row_offset < bidxp1)  &&
                      (nalu_hypre_cabs(A_diag_data[ii]) > NALU_HYPRE_REAL_MIN))
                  {
                     didx = bstart + ridx * left_size + jj - bidxm1 - row_offset;
                     diag_data[didx] = A_diag_data[ii];
                  }
               }
            }
            cnt++;
         }
         else
         {
            row_offset++;
         }
      }
   }
   NALU_HYPRE_ANNOTATE_REGION_END("%s", "ExtractDiagSubBlocks");

   /*-----------------------------------------------------------------
    * Compute the inverses of all the diagonal sub-blocks
    *-----------------------------------------------------------------*/

   if (diag_type == 1)
   {
      NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "InvertDiagSubBlocks");
      if (blk_size > 1)
      {
         nalu_hypre_BlockDiagInvLapack(diag_data, num_points, blk_size);
      }
      else
      {
         for (i = 0; i < num_points; i++)
         {
            if (nalu_hypre_cabs(diag_data[i]) < NALU_HYPRE_REAL_MIN)
            {
               diag_data[i] = 0.0;
            }
            else
            {
               diag_data[i] = 1.0 / diag_data[i];
            }
         }
      }
      NALU_HYPRE_ANNOTATE_REGION_END("%s", "InvertDiagSubBlocks");
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixBlockDiagMatrix
 *
 * Extract the block diagonal part of a A or a principal submatrix of A defined
 * by a marker (point_type) in an associated CF_marker array. The result is
 * a new block diagonal parCSR matrix.
 *
 * If CF marker array is NULL, it returns the block diagonal of the matrix A.
 *
 * Options for diag_type are:
 *    diag_type = 1: return the inverse of the block diagonals
 *    otherwise : return the block diagonals
 *
 * Input parameters are:
 *    par_A      - original ParCSR matrix
 *    blk_size   - Size of diagonal blocks to extract
 *    CF_marker  - Array prescribing submatrix from which to extract block
 *                 diagonals. Ignored if NULL.
 *    point_type - marker tag in CF_marker array to extract diagonal
 *    diag_type  - Type of block diagonal entries to return. Currently supports
 *                 block diagonal or inverse block diagonal entries.
 *
 * Output parameters are:
 *    B_ptr      - New block diagonal matrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixBlockDiagMatrix( nalu_hypre_ParCSRMatrix  *A,
                                   NALU_HYPRE_Int            blk_size,
                                   NALU_HYPRE_Int            point_type,
                                   NALU_HYPRE_Int           *CF_marker,
                                   NALU_HYPRE_Int            diag_type,
                                   nalu_hypre_ParCSRMatrix **B_ptr )
{
#if defined (NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_ParCSRMatrixBlockDiagMatrixDevice(A, blk_size, point_type,
                                              CF_marker, diag_type, B_ptr);
   }
   else
#endif
   {
      nalu_hypre_ParCSRMatrixBlockDiagMatrixHost(A, blk_size, point_type,
                                            CF_marker, diag_type, B_ptr);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixBlockDiagMatrixHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixBlockDiagMatrixHost( nalu_hypre_ParCSRMatrix  *A,
                                       NALU_HYPRE_Int            blk_size,
                                       NALU_HYPRE_Int            point_type,
                                       NALU_HYPRE_Int           *CF_marker,
                                       NALU_HYPRE_Int            diag_type,
                                       nalu_hypre_ParCSRMatrix **B_ptr )
{
   /* Input matrix info */
   MPI_Comm              comm            = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_BigInt         *row_starts_A    = nalu_hypre_ParCSRMatrixRowStarts(A);
   NALU_HYPRE_BigInt          num_rows_A      = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   nalu_hypre_CSRMatrix      *A_diag          = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int             A_diag_num_rows = nalu_hypre_CSRMatrixNumRows(A_diag);

   /* Global block matrix info */
   nalu_hypre_ParCSRMatrix   *par_B;
   NALU_HYPRE_BigInt          num_rows_B;
   NALU_HYPRE_BigInt          row_starts_B[2];

   /* Diagonal block matrix info */
   nalu_hypre_CSRMatrix      *B_diag;
   NALU_HYPRE_Int             B_diag_num_rows = 0;
   NALU_HYPRE_Int             B_diag_size;
   NALU_HYPRE_Int            *B_diag_i;
   NALU_HYPRE_Int            *B_diag_j;
   NALU_HYPRE_Complex        *B_diag_data;

   /* Local variables */
   NALU_HYPRE_BigInt          num_rows_big;
   NALU_HYPRE_BigInt          scan_recv;
   NALU_HYPRE_Int             num_procs, my_id;
   NALU_HYPRE_Int             nb2 = blk_size * blk_size;
   NALU_HYPRE_Int             num_blocks, num_left;
   NALU_HYPRE_Int             bidx, i, j, k;

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   /* Sanity check */
   if ((num_rows_A > 0) && (num_rows_A < blk_size))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error!!! Input matrix is smaller than block size.");
      return nalu_hypre_error_flag;
   }

   /*-----------------------------------------------------------------
    * Count the number of points matching point_type in CF_marker
    *-----------------------------------------------------------------*/

   if (CF_marker == NULL)
   {
      B_diag_num_rows = A_diag_num_rows;
   }
   else
   {
#if !defined(_MSC_VER) && defined(NALU_HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:B_diag_num_rows) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < A_diag_num_rows; i++)
      {
         B_diag_num_rows += (CF_marker[i] == point_type) ? 1 : 0;
      }
   }
   num_blocks  = B_diag_num_rows / blk_size;
   num_left    = B_diag_num_rows - num_blocks * blk_size;
   B_diag_size = blk_size * (blk_size * num_blocks) + num_left * num_left;

   /*-----------------------------------------------------------------
    * Compute global number of rows and partitionings
    *-----------------------------------------------------------------*/

   if (CF_marker)
   {
      num_rows_big = (NALU_HYPRE_BigInt) B_diag_num_rows;
      nalu_hypre_MPI_Scan(&num_rows_big, &scan_recv, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

      /* first point in my range */
      row_starts_B[0] = scan_recv - num_rows_big;

      /* first point in next proc's range */
      row_starts_B[1] = scan_recv;
      if (my_id == (num_procs - 1))
      {
         num_rows_B = row_starts_B[1];
      }
      nalu_hypre_MPI_Bcast(&num_rows_B, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      row_starts_B[0] = row_starts_A[0];
      row_starts_B[1] = row_starts_A[1];
      num_rows_B = num_rows_A;
   }

   /* Create matrix B */
   par_B = nalu_hypre_ParCSRMatrixCreate(comm,
                                    num_rows_B,
                                    num_rows_B,
                                    row_starts_B,
                                    row_starts_B,
                                    0,
                                    B_diag_size,
                                    0);
   nalu_hypre_ParCSRMatrixInitialize_v2(par_B, NALU_HYPRE_MEMORY_HOST);
   B_diag      = nalu_hypre_ParCSRMatrixDiag(par_B);
   B_diag_i    = nalu_hypre_CSRMatrixI(B_diag);
   B_diag_j    = nalu_hypre_CSRMatrixJ(B_diag);
   B_diag_data = nalu_hypre_CSRMatrixData(B_diag);

   /*-----------------------------------------------------------------------
    * Extract coefficients
    *-----------------------------------------------------------------------*/

   nalu_hypre_ParCSRMatrixExtractBlockDiagHost(A, blk_size, B_diag_num_rows,
                                          point_type, CF_marker,
                                          B_diag_size, diag_type,
                                          B_diag_data);

   /*-----------------------------------------------------------------
    * Set row/col indices of diagonal blocks
    *-----------------------------------------------------------------*/

   B_diag_i[B_diag_num_rows] = B_diag_size;
   for (i = 0; i < num_blocks; i++)
   {
      //diag_local = &diag[i * nb2];
      for (k = 0; k < blk_size; k++)
      {
         B_diag_i[i * blk_size + k] = i * nb2 + k * blk_size;

         for (j = 0; j < blk_size; j++)
         {
            bidx = i * nb2 + k * blk_size + j;
            B_diag_j[bidx] = i * blk_size + j;
            //B_diag_data[bidx] = diag_local[k * blk_size + j];
         }
      }
   }

   /*-----------------------------------------------------------------
    * Treat the remaining points
    *-----------------------------------------------------------------*/

   //diag_local = &diag[num_blocks * nb2];
   for (k = 0; k < num_left; k++)
   {
      B_diag_i[num_blocks * blk_size + k] = num_blocks * nb2 + k * num_left;

      for (j = 0; j < num_left; j++)
      {
         bidx = num_blocks * nb2 + k * num_left + j;
         B_diag_j[bidx] = num_blocks * blk_size + j;
         //B_diag_data[bidx] = diag_local[k * num_left + j];
      }
   }

   /* Set output pointer */
   *B_ptr = par_B;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRBlockRelaxSetup
 *
 * Setup block smoother. Computes the entries of the inverse of the block
 * diagonal matrix with blk_size diagonal blocks.
 *
 * Current implementation ignores reserved C-pts and acts on whole matrix.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRBlockRelaxSetup( nalu_hypre_ParCSRMatrix *A,
                          NALU_HYPRE_Int           blk_size,
                          NALU_HYPRE_Real        **diaginvptr )
{
   nalu_hypre_CSRMatrix      *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int             num_rows = nalu_hypre_CSRMatrixNumRows(A_diag);

   NALU_HYPRE_Int             num_blocks;
   NALU_HYPRE_Int             diag_size;
   NALU_HYPRE_Complex        *diaginv = *diaginvptr;

   num_blocks = 1 + (num_rows - 1) / blk_size;
   diag_size  = blk_size * (blk_size * num_blocks);

   nalu_hypre_TFree(diaginv, NALU_HYPRE_MEMORY_HOST);
   diaginv = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, diag_size, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParCSRMatrixExtractBlockDiagHost(A, blk_size, num_rows, 0, NULL,
                                          diag_size, 1, diaginv);

   *diaginvptr = diaginv;

#if 0
   MPI_Comm      comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real     *A_diag_data  = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int            *A_diag_i     = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int            *A_diag_j     = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Int             n       = nalu_hypre_CSRMatrixNumRows(A_diag);

   NALU_HYPRE_Int             i, j, k;
   NALU_HYPRE_Int             ii, jj;
   NALU_HYPRE_Int             bidx, bidxm1, bidxp1;
   NALU_HYPRE_Int         num_procs, my_id;

   const NALU_HYPRE_Int     nb2 = blk_size * blk_size;
   NALU_HYPRE_Int           n_block;
   NALU_HYPRE_Int           left_size, inv_size;
   NALU_HYPRE_Real        *diaginv = *diaginvptr;


   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   //NALU_HYPRE_Int num_threads = nalu_hypre_NumThreads();

   if (my_id == num_procs)
   {
      n_block   = (n - reserved_coarse_size) / blk_size;
      left_size = n - blk_size * n_block;
   }
   else
   {
      n_block = n / blk_size;
      left_size = n - blk_size * n_block;
   }

   n_block = n / blk_size;
   left_size = n - blk_size * n_block;

   inv_size  = nb2 * n_block + left_size * left_size;

   if (diaginv != NULL)
   {
      nalu_hypre_TFree(diaginv, NALU_HYPRE_MEMORY_HOST);
      diaginv = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  inv_size, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      diaginv = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  inv_size, NALU_HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------------
   * Get all the diagonal sub-blocks
   *-----------------------------------------------------------------*/
   for (i = 0; i < n_block; i++)
   {
      bidxm1 = i * blk_size;
      bidxp1 = (i + 1) * blk_size;
      //printf("bidxm1 = %d,bidxp1 = %d\n",bidxm1,bidxp1);

      for (k = 0; k < blk_size; k++)
      {
         for (j = 0; j < blk_size; j++)
         {
            bidx = i * nb2 + k * blk_size + j;
            diaginv[bidx] = 0.0;
         }

         for (ii = A_diag_i[bidxm1 + k]; ii < A_diag_i[bidxm1 + k + 1]; ii++)
         {
            jj = A_diag_j[ii];
            if (jj >= bidxm1 && jj < bidxp1 && nalu_hypre_cabs(A_diag_data[ii]) > NALU_HYPRE_REAL_MIN)
            {
               bidx = i * nb2 + k * blk_size + jj - bidxm1;
               //printf("jj = %d,val = %e, bidx = %d\n",jj,A_diag_data[ii],bidx);
               diaginv[bidx] = A_diag_data[ii];
            }
         }
      }
   }

   for (i = 0; i < left_size; i++)
   {
      bidxm1 = n_block * nb2 + i * blk_size;
      bidxp1 = n_block * nb2 + (i + 1) * blk_size;
      for (j = 0; j < left_size; j++)
      {
         bidx = n_block * nb2 + i * blk_size + j;
         diaginv[bidx] = 0.0;
      }

      for (ii = A_diag_i[n_block * blk_size + i]; ii < A_diag_i[n_block * blk_size + i + 1]; ii++)
      {
         jj = A_diag_j[ii];
         if (jj > n_block * blk_size)
         {
            bidx = n_block * nb2 + i * blk_size + jj - n_block * blk_size;
            diaginv[bidx] = A_diag_data[ii];
         }
      }
   }

   /*-----------------------------------------------------------------
   * compute the inverses of all the diagonal sub-blocks
   *-----------------------------------------------------------------*/
   if (blk_size > 1)
   {
      for (i = 0; i < n_block; i++)
      {
         nalu_hypre_blas_mat_inv(diaginv + i * nb2, blk_size);
      }
      nalu_hypre_blas_mat_inv(diaginv + (NALU_HYPRE_Int)(blk_size * nb2), left_size);
   }
   else
   {
      for (i = 0; i < n; i++)
      {
         /* TODO: zero-diagonal should be tested previously */
         if (nalu_hypre_cabs(diaginv[i]) < NALU_HYPRE_REAL_MIN)
         {
            diaginv[i] = 0.0;
         }
         else
         {
            diaginv[i] = 1.0 / diaginv[i];
         }
      }
   }

   *diaginvptr = diaginv;
#endif
   return nalu_hypre_error_flag;
}
#if 0
NALU_HYPRE_Int
nalu_hypre_blockRelax(nalu_hypre_ParCSRMatrix *A,
                 nalu_hypre_ParVector    *f,
                 nalu_hypre_ParVector    *u,
                 NALU_HYPRE_Int          blk_size,
                 NALU_HYPRE_Int          reserved_coarse_size,
                 NALU_HYPRE_Int          method,
                 nalu_hypre_ParVector    *Vtemp,
                 nalu_hypre_ParVector    *Ztemp)
{
   MPI_Comm      comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real     *A_diag_data  = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int            *A_diag_i     = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int            *A_diag_j     = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Int             n       = nalu_hypre_CSRMatrixNumRows(A_diag);

   NALU_HYPRE_Int             i, j, k;
   NALU_HYPRE_Int             ii, jj;

   NALU_HYPRE_Int             bidx, bidxm1, bidxp1;

   NALU_HYPRE_Int         num_procs, my_id;

   const NALU_HYPRE_Int     nb2 = blk_size * blk_size;
   NALU_HYPRE_Int           n_block;
   NALU_HYPRE_Int           left_size, inv_size;
   NALU_HYPRE_Real          *diaginv;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   //NALU_HYPRE_Int num_threads = nalu_hypre_NumThreads();

   if (my_id == num_procs)
   {
      n_block   = (n - reserved_coarse_size) / blk_size;
      left_size = n - blk_size * n_block;
   }
   else
   {
      n_block = n / blk_size;
      left_size = n - blk_size * n_block;
   }

   inv_size  = nb2 * n_block + left_size * left_size;

   diaginv = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  inv_size, NALU_HYPRE_MEMORY_HOST);
   /*-----------------------------------------------------------------
   * Get all the diagonal sub-blocks
   *-----------------------------------------------------------------*/
   for (i = 0; i < n_block; i++)
   {
      bidxm1 = i * blk_size;
      bidxp1 = (i + 1) * blk_size;
      //printf("bidxm1 = %d,bidxp1 = %d\n",bidxm1,bidxp1);

      for (k = 0; k < blk_size; k++)
      {
         for (j = 0; j < blk_size; j++)
         {
            bidx = i * nb2 + k * blk_size + j;
            diaginv[bidx] = 0.0;
         }

         for (ii = A_diag_i[bidxm1 + k]; ii < A_diag_i[bidxm1 + k + 1]; ii++)
         {
            jj = A_diag_j[ii];

            if (jj >= bidxm1 && jj < bidxp1 && nalu_hypre_abs(A_diag_data[ii]) > NALU_HYPRE_REAL_MIN)
            {
               bidx = i * nb2 + k * blk_size + jj - bidxm1;
               //printf("jj = %d,val = %e, bidx = %d\n",jj,A_diag_data[ii],bidx);
               diaginv[bidx] = A_diag_data[ii];
            }
         }
      }
   }

   for (i = 0; i < left_size; i++)
   {
      bidxm1 = n_block * nb2 + i * blk_size;
      bidxp1 = n_block * nb2 + (i + 1) * blk_size;
      for (j = 0; j < left_size; j++)
      {
         bidx = n_block * nb2 + i * blk_size + j;
         diaginv[bidx] = 0.0;
      }

      for (ii = A_diag_i[n_block * blk_size + i]; ii < A_diag_i[n_block * blk_size + i + 1]; ii++)
      {
         jj = A_diag_j[ii];
         if (jj > n_block * blk_size)
         {
            bidx = n_block * nb2 + i * blk_size + jj - n_block * blk_size;
            diaginv[bidx] = A_diag_data[ii];
         }
      }
   }
   /*
   for (i = 0;i < n_block; i++)
   {
     for (j = 0;j < blk_size; j++)
     {
       for (k = 0;k < blk_size; k ++)
       {
         bidx = i*nb2 + j*blk_size + k;
         printf("%e\t",diaginv[bidx]);
       }
       printf("\n");
     }
     printf("\n");
   }
   */
   /*-----------------------------------------------------------------
   * compute the inverses of all the diagonal sub-blocks
   *-----------------------------------------------------------------*/
   if (blk_size > 1)
   {
      for (i = 0; i < n_block; i++)
      {
         nalu_hypre_blas_mat_inv(diaginv + i * nb2, blk_size);
      }
      nalu_hypre_blas_mat_inv(diaginv + (NALU_HYPRE_Int)(blk_size * nb2), left_size);
      /*
      for (i = 0;i < n_block; i++)
      {
        for (j = 0;j < blk_size; j++)
        {
          for (k = 0;k < blk_size; k ++)
          {
            bidx = i*nb2 + j*blk_size + k;
            printf("%e\t",diaginv[bidx]);
          }
          printf("\n");
        }
        printf("\n");
      }
      */
   }
   else
   {
      for (i = 0; i < n; i++)
      {
         // FIX-ME: zero-diagonal should be tested previously
         if (nalu_hypre_abs(diaginv[i]) < NALU_HYPRE_REAL_MIN)
         {
            diaginv[i] = 0.0;
         }
         else
         {
            diaginv[i] = 1.0 / diaginv[i];
         }
      }
   }

   nalu_hypre_MGRBlockRelaxSolve(A, f, u, blk_size, n_block, left_size, method, diaginv, Vtemp);

   /*-----------------------------------------------------------------
   * Free temporary memory
   *-----------------------------------------------------------------*/
   nalu_hypre_TFree(diaginv, NALU_HYPRE_MEMORY_HOST);

   return (nalu_hypre_error_flag);
}
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_MGRSetFSolver
 *
 * set F-relaxation solver
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MGRSetFSolver( void  *mgr_vdata,
                     NALU_HYPRE_Int  (*fine_grid_solver_solve)(void*, void*, void*, void*),
                     NALU_HYPRE_Int  (*fine_grid_solver_setup)(void*, void*, void*, void*),
                     void       *fsolver )
{
   nalu_hypre_ParMGRData *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   NALU_HYPRE_Solver **aff_solver = (mgr_data -> aff_solver);

   if (aff_solver == NULL)
   {
      aff_solver = nalu_hypre_CTAlloc(NALU_HYPRE_Solver*, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   }

   /* only allow to set F-solver for the first level */
   aff_solver[0] = (NALU_HYPRE_Solver *) fsolver;

   (mgr_data -> fine_grid_solver_solve) = fine_grid_solver_solve;
   (mgr_data -> fine_grid_solver_setup) = fine_grid_solver_setup;
   (mgr_data -> aff_solver) = aff_solver;
   (mgr_data -> fsolver_mode) = 0;

   return nalu_hypre_error_flag;
}

/* set coarse grid solver */
NALU_HYPRE_Int
nalu_hypre_MGRSetCoarseSolver( void  *mgr_vdata,
                          NALU_HYPRE_Int  (*coarse_grid_solver_solve)(void*, void*, void*, void*),
                          NALU_HYPRE_Int  (*coarse_grid_solver_setup)(void*, void*, void*, void*),
                          void  *coarse_grid_solver )
{
   nalu_hypre_ParMGRData *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   (mgr_data -> coarse_grid_solver_solve) = coarse_grid_solver_solve;
   (mgr_data -> coarse_grid_solver_setup) = coarse_grid_solver_setup;
   (mgr_data -> coarse_grid_solver)       = (NALU_HYPRE_Solver) coarse_grid_solver;

   (mgr_data -> use_default_cgrid_solver) = 0;

   return nalu_hypre_error_flag;
}

/* Set the maximum number of coarse levels.
 * maxcoarselevs = 1 yields the default 2-grid scheme.
*/
NALU_HYPRE_Int
nalu_hypre_MGRSetMaxCoarseLevels( void *mgr_vdata, NALU_HYPRE_Int maxcoarselevs )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> max_num_coarse_levels) = maxcoarselevs;
   return nalu_hypre_error_flag;
}

/* Set the system block size */
NALU_HYPRE_Int
nalu_hypre_MGRSetBlockSize( void *mgr_vdata, NALU_HYPRE_Int bsize )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> block_size) = bsize;
   return nalu_hypre_error_flag;
}

/* Set the relaxation type for the fine levels of the reduction.
 * Currently supports the following flavors of relaxation types
 * as described in the documentation:
 * relax_types 0 - 8, 13, 14, 18, 19, 98.
 * See par_relax.c and par_relax_more.c for more details.
 * */
NALU_HYPRE_Int
nalu_hypre_MGRSetRelaxType( void *mgr_vdata, NALU_HYPRE_Int relax_type )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> relax_type) = relax_type;
   return nalu_hypre_error_flag;
}

/* Set the number of relaxation sweeps */
NALU_HYPRE_Int
nalu_hypre_MGRSetNumRelaxSweeps( void *mgr_vdata, NALU_HYPRE_Int nsweeps )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   nalu_hypre_TFree(mgr_data -> num_relax_sweeps, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int *num_relax_sweeps = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels,
                                               NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      num_relax_sweeps[i] = nsweeps;
   }
   (mgr_data -> num_relax_sweeps) = num_relax_sweeps;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_MGRSetLevelNumRelaxSweeps( void *mgr_vdata, NALU_HYPRE_Int *level_nsweeps )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   nalu_hypre_TFree(mgr_data -> num_relax_sweeps, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_Int *num_relax_sweeps = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels,
                                               NALU_HYPRE_MEMORY_HOST);
   if (level_nsweeps != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         num_relax_sweeps[i] = level_nsweeps[i];
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         num_relax_sweeps[i] = 0;
      }
   }
   (mgr_data -> num_relax_sweeps) = num_relax_sweeps;

   return nalu_hypre_error_flag;
}

/* Set the order of the global smoothing step at each level
 * 1=Down cycle/ Pre-smoothing (default)
 * 2=Up cycle/ Post-smoothing
 */
NALU_HYPRE_Int
nalu_hypre_MGRSetGlobalSmoothCycle( void *mgr_vdata, NALU_HYPRE_Int smooth_cycle )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> global_smooth_cycle) = smooth_cycle;
   return nalu_hypre_error_flag;
}

/* Set the F-relaxation strategy: 0=single level, 1=multi level */
NALU_HYPRE_Int
nalu_hypre_MGRSetFRelaxMethod( void *mgr_vdata, NALU_HYPRE_Int relax_method )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   nalu_hypre_TFree(mgr_data -> Frelax_method, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int *Frelax_method = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      Frelax_method[i] = relax_method;
   }
   (mgr_data -> Frelax_method) = Frelax_method;
   return nalu_hypre_error_flag;
}

/* Set the F-relaxation strategy: 0=single level, 1=multi level */
/* This will be removed later. Use SetLevelFrelaxType */
NALU_HYPRE_Int
nalu_hypre_MGRSetLevelFRelaxMethod( void *mgr_vdata, NALU_HYPRE_Int *relax_method )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   nalu_hypre_TFree(mgr_data -> Frelax_method, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_Int *Frelax_method = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   if (relax_method != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         Frelax_method[i] = relax_method[i];
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         Frelax_method[i] = 0;
      }
   }
   (mgr_data -> Frelax_method) = Frelax_method;
   return nalu_hypre_error_flag;
}

/* Set the F-relaxation type:
 * 0: Jacobi
 * 1: Vcycle smoother
 * 2: AMG
 * Otherwise: use standard BoomerAMGRelax options
*/
NALU_HYPRE_Int
nalu_hypre_MGRSetLevelFRelaxType( void *mgr_vdata, NALU_HYPRE_Int *relax_type )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   nalu_hypre_TFree(mgr_data -> Frelax_type, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_Int *Frelax_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   if (relax_type != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         Frelax_type[i] = relax_type[i];
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         Frelax_type[i] = 0;
      }
   }
   (mgr_data -> Frelax_type) = Frelax_type;
   return nalu_hypre_error_flag;
}

/* Coarse grid method: 0=Galerkin RAP, 1=non-Galerkin with dropping */
NALU_HYPRE_Int
nalu_hypre_MGRSetCoarseGridMethod( void *mgr_vdata, NALU_HYPRE_Int *cg_method )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);

   nalu_hypre_TFree(mgr_data -> mgr_coarse_grid_method, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int *mgr_coarse_grid_method = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels,
                                                     NALU_HYPRE_MEMORY_HOST);
   if (cg_method != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         mgr_coarse_grid_method[i] = cg_method[i];
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         mgr_coarse_grid_method[i] = 0;
      }
   }
   (mgr_data -> mgr_coarse_grid_method) = mgr_coarse_grid_method;
   return nalu_hypre_error_flag;
}

/* Set the F-relaxation number of functions for each level */
NALU_HYPRE_Int
nalu_hypre_MGRSetLevelFRelaxNumFunctions( void *mgr_vdata, NALU_HYPRE_Int *num_functions )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);

   nalu_hypre_TFree(mgr_data -> Frelax_num_functions, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_Int *Frelax_num_functions = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels,
                                                   NALU_HYPRE_MEMORY_HOST);
   if (num_functions != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         Frelax_num_functions[i] = num_functions[i];
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         Frelax_num_functions[i] = 1;
      }
   }
   (mgr_data -> Frelax_num_functions) = Frelax_num_functions;
   return nalu_hypre_error_flag;
}

/* Set the type of the restriction type
 * for computing restriction operator
*/
NALU_HYPRE_Int
nalu_hypre_MGRSetLevelRestrictType( void *mgr_vdata, NALU_HYPRE_Int *restrict_type)
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   nalu_hypre_TFree((mgr_data -> restrict_type), NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_Int *level_restrict_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   if (restrict_type != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_restrict_type[i] = *(restrict_type + i);
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_restrict_type[i] = 0;
      }
   }
   (mgr_data -> restrict_type) = level_restrict_type;
   return nalu_hypre_error_flag;
}

/* Set the type of the restriction type
 * for computing restriction operator
*/
NALU_HYPRE_Int
nalu_hypre_MGRSetRestrictType( void *mgr_vdata, NALU_HYPRE_Int restrict_type)
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   if ((mgr_data -> restrict_type) != NULL)
   {
      nalu_hypre_TFree((mgr_data -> restrict_type), NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> restrict_type) = NULL;
   }
   NALU_HYPRE_Int *level_restrict_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      level_restrict_type[i] = restrict_type;
   }
   (mgr_data -> restrict_type) = level_restrict_type;
   return nalu_hypre_error_flag;
}

/* Set the number of Jacobi interpolation iterations
 * for computing interpolation operator
*/
NALU_HYPRE_Int
nalu_hypre_MGRSetNumRestrictSweeps( void *mgr_vdata, NALU_HYPRE_Int nsweeps )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> num_restrict_sweeps) = nsweeps;
   return nalu_hypre_error_flag;
}

/* Set the type of the interpolation
 * for computing interpolation operator
*/
NALU_HYPRE_Int
nalu_hypre_MGRSetInterpType( void *mgr_vdata, NALU_HYPRE_Int interpType)
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   if ((mgr_data -> interp_type) != NULL)
   {
      nalu_hypre_TFree((mgr_data -> interp_type), NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> interp_type) = NULL;
   }
   NALU_HYPRE_Int *level_interp_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      level_interp_type[i] = interpType;
   }
   (mgr_data -> interp_type) = level_interp_type;
   return nalu_hypre_error_flag;
}

/* Set the type of the interpolation
 * for computing interpolation operator
*/
NALU_HYPRE_Int
nalu_hypre_MGRSetLevelInterpType( void *mgr_vdata, NALU_HYPRE_Int *interpType)
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   nalu_hypre_TFree((mgr_data -> interp_type), NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_Int *level_interp_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   if (interpType != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_interp_type[i] = *(interpType + i);
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_interp_type[i] = 2;
      }
   }
   (mgr_data -> interp_type) = level_interp_type;
   return nalu_hypre_error_flag;
}

/* Set the number of Jacobi interpolation iterations
 * for computing interpolation operator
*/
NALU_HYPRE_Int
nalu_hypre_MGRSetNumInterpSweeps( void *mgr_vdata, NALU_HYPRE_Int nsweeps )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> num_interp_sweeps) = nsweeps;
   return nalu_hypre_error_flag;
}

/* Set the threshold to truncate the coarse grid at each
 * level of reduction
*/
NALU_HYPRE_Int
nalu_hypre_MGRSetTruncateCoarseGridThreshold( void *mgr_vdata, NALU_HYPRE_Real threshold)
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> truncate_coarse_grid_threshold) = threshold;
   return nalu_hypre_error_flag;
}

/* Set block size for block Jacobi Interp/Relax */
NALU_HYPRE_Int
nalu_hypre_MGRSetBlockJacobiBlockSize( void *mgr_vdata, NALU_HYPRE_Int blk_size)
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> block_jacobi_bsize) = blk_size;
   return nalu_hypre_error_flag;
}

/* Set print level for F-relaxation solver */
NALU_HYPRE_Int
nalu_hypre_MGRSetFrelaxPrintLevel( void *mgr_vdata, NALU_HYPRE_Int print_level )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> frelax_print_level) = print_level;
   return nalu_hypre_error_flag;
}

/* Set print level for coarse grid solver */
NALU_HYPRE_Int
nalu_hypre_MGRSetCoarseGridPrintLevel( void *mgr_vdata, NALU_HYPRE_Int print_level )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> cg_print_level) = print_level;
   return nalu_hypre_error_flag;
}

/* Set print level for mgr solver */
NALU_HYPRE_Int
nalu_hypre_MGRSetPrintLevel( void *mgr_vdata, NALU_HYPRE_Int print_level )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> print_level) = print_level;
   return nalu_hypre_error_flag;
}

/* Set logging level for mgr solver */
NALU_HYPRE_Int
nalu_hypre_MGRSetLogging( void *mgr_vdata, NALU_HYPRE_Int logging )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> logging) = logging;
   return nalu_hypre_error_flag;
}

/* Set max number of iterations for mgr solver */
NALU_HYPRE_Int
nalu_hypre_MGRSetMaxIter( void *mgr_vdata, NALU_HYPRE_Int max_iter )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> max_iter) = max_iter;
   return nalu_hypre_error_flag;
}

/* Set convergence tolerance for mgr solver */
NALU_HYPRE_Int
nalu_hypre_MGRSetTol( void *mgr_vdata, NALU_HYPRE_Real tol )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> tol) = tol;
   return nalu_hypre_error_flag;
}

/* Set max number of iterations for mgr global smoother */
NALU_HYPRE_Int
nalu_hypre_MGRSetMaxGlobalSmoothIters( void *mgr_vdata, NALU_HYPRE_Int max_iter )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   if ((mgr_data -> level_smooth_iters) != NULL)
   {
      nalu_hypre_TFree((mgr_data -> level_smooth_iters), NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> level_smooth_iters) = NULL;
   }
   NALU_HYPRE_Int *level_smooth_iters = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   if (max_num_coarse_levels > 0)
   {
      level_smooth_iters[0] = max_iter;
   }
   (mgr_data -> level_smooth_iters) = level_smooth_iters;

   return nalu_hypre_error_flag;
}

/* Set global smoothing type for mgr solver */
NALU_HYPRE_Int
nalu_hypre_MGRSetGlobalSmoothType( void *mgr_vdata, NALU_HYPRE_Int gsmooth_type )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   if ((mgr_data -> level_smooth_type) != NULL)
   {
      nalu_hypre_TFree((mgr_data -> level_smooth_type), NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> level_smooth_type) = NULL;
   }
   NALU_HYPRE_Int *level_smooth_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   if (max_num_coarse_levels > 0)
   {
      level_smooth_type[0] = gsmooth_type;
   }
   (mgr_data -> level_smooth_type) = level_smooth_type;

   return nalu_hypre_error_flag;
}

/* Set global smoothing type for mgr solver */
NALU_HYPRE_Int
nalu_hypre_MGRSetLevelSmoothType( void *mgr_vdata, NALU_HYPRE_Int *gsmooth_type )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   nalu_hypre_TFree((mgr_data -> level_smooth_type), NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_Int *level_smooth_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   if (gsmooth_type != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_smooth_type[i] = gsmooth_type[i];
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_smooth_type[i] = 0;
      }
   }
   (mgr_data -> level_smooth_type) = level_smooth_type;
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_MGRSetLevelSmoothIters( void *mgr_vdata, NALU_HYPRE_Int *gsmooth_iters )
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   nalu_hypre_TFree((mgr_data -> level_smooth_iters), NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_Int *level_smooth_iters = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   if (gsmooth_iters != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_smooth_iters[i] = gsmooth_iters[i];
      }
   }
   else
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         level_smooth_iters[i] = 0;
      }
   }
   (mgr_data -> level_smooth_iters) = level_smooth_iters;
   return nalu_hypre_error_flag;
}

/* Set the maximum number of non-zero entries for restriction
   and interpolation operator if classical AMG interpolation is used */
NALU_HYPRE_Int
nalu_hypre_MGRSetPMaxElmts( void *mgr_vdata, NALU_HYPRE_Int P_max_elmts)
{
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   (mgr_data -> P_max_elmts) = P_max_elmts;
   return nalu_hypre_error_flag;
}

/* Get number of iterations for MGR solver */
NALU_HYPRE_Int
nalu_hypre_MGRGetNumIterations( void *mgr_vdata, NALU_HYPRE_Int *num_iterations )
{
   nalu_hypre_ParMGRData  *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *num_iterations = mgr_data->num_iterations;

   return nalu_hypre_error_flag;
}

/* Get residual norms for MGR solver */
NALU_HYPRE_Int
nalu_hypre_MGRGetFinalRelativeResidualNorm( void *mgr_vdata, NALU_HYPRE_Real *res_norm )
{
   nalu_hypre_ParMGRData  *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *res_norm = mgr_data->final_rel_residual_norm;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_MGRGetCoarseGridConvergenceFactor( void *mgr_vdata, NALU_HYPRE_Real *conv_factor )
{
   nalu_hypre_ParMGRData  *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *conv_factor = (mgr_data -> cg_convergence_factor);

   return nalu_hypre_error_flag;
}

/* Build A_FF matrix from A given a CF_marker array */
NALU_HYPRE_Int
nalu_hypre_MGRGetSubBlock( nalu_hypre_ParCSRMatrix   *A,
                      NALU_HYPRE_Int            *row_cf_marker,
                      NALU_HYPRE_Int            *col_cf_marker,
                      NALU_HYPRE_Int             debug_flag,
                      nalu_hypre_ParCSRMatrix  **A_block_ptr )
{
   MPI_Comm        comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int             *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int             *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);

   nalu_hypre_CSRMatrix *A_offd         = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real      *A_offd_data    = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int             *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int             *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Int              num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   //NALU_HYPRE_Int             *col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(A);

   nalu_hypre_IntArray          *coarse_dof_func_ptr = NULL;
   NALU_HYPRE_BigInt            num_row_cpts_global[2];
   NALU_HYPRE_BigInt            num_col_cpts_global[2];

   nalu_hypre_ParCSRMatrix    *Ablock;
   NALU_HYPRE_BigInt         *col_map_offd_Ablock;
   NALU_HYPRE_Int       *tmp_map_offd = NULL;

   NALU_HYPRE_Int             *CF_marker_offd = NULL;

   nalu_hypre_CSRMatrix    *Ablock_diag;
   nalu_hypre_CSRMatrix    *Ablock_offd;

   NALU_HYPRE_Real      *Ablock_diag_data;
   NALU_HYPRE_Int             *Ablock_diag_i;
   NALU_HYPRE_Int             *Ablock_diag_j;
   NALU_HYPRE_Real      *Ablock_offd_data;
   NALU_HYPRE_Int             *Ablock_offd_i;
   NALU_HYPRE_Int             *Ablock_offd_j;

   NALU_HYPRE_Int              Ablock_diag_size, Ablock_offd_size;

   NALU_HYPRE_Int             *Ablock_marker;

   NALU_HYPRE_Int              ii_counter;
   NALU_HYPRE_Int              jj_counter, jj_counter_offd;
   NALU_HYPRE_Int             *jj_count, *jj_count_offd;

   NALU_HYPRE_Int              start_indexing = 0; /* start indexing for Aff_data at 0 */

   NALU_HYPRE_Int              n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);

   NALU_HYPRE_Int             *fine_to_coarse;
   NALU_HYPRE_Int             *coarse_counter;
   NALU_HYPRE_Int             *col_coarse_counter;
   NALU_HYPRE_Int              coarse_shift;
   NALU_HYPRE_BigInt              total_global_row_cpts;
   NALU_HYPRE_BigInt              total_global_col_cpts;
   NALU_HYPRE_Int              num_cols_Ablock_offd;
   //  NALU_HYPRE_BigInt              my_first_row_cpt, my_first_col_cpt;

   NALU_HYPRE_Int              i, i1;
   NALU_HYPRE_Int              j, jl, jj;
   NALU_HYPRE_Int              start;

   NALU_HYPRE_Int              my_id;
   NALU_HYPRE_Int              num_procs;
   NALU_HYPRE_Int              num_threads;
   NALU_HYPRE_Int              num_sends;
   NALU_HYPRE_Int              index;
   NALU_HYPRE_Int              ns, ne, size, rest;
   NALU_HYPRE_Int             *int_buf_data;
   NALU_HYPRE_Int              local_numrows = nalu_hypre_CSRMatrixNumRows(A_diag);

   nalu_hypre_IntArray        *wrap_cf;

   //  NALU_HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   //num_threads = nalu_hypre_NumThreads();
   // Temporary fix, disable threading
   // TODO: enable threading
   num_threads = 1;

   /* get the number of coarse rows */
   wrap_cf = nalu_hypre_IntArrayCreate(local_numrows);
   nalu_hypre_IntArrayMemoryLocation(wrap_cf) = NALU_HYPRE_MEMORY_HOST;
   nalu_hypre_IntArrayData(wrap_cf) = row_cf_marker;
   nalu_hypre_BoomerAMGCoarseParms(comm, local_numrows, 1, NULL, wrap_cf, &coarse_dof_func_ptr,
                              num_row_cpts_global);
   nalu_hypre_IntArrayDestroy(coarse_dof_func_ptr);
   coarse_dof_func_ptr = NULL;

   //nalu_hypre_printf("my_id = %d, cpts_this = %d, cpts_next = %d\n", my_id, num_row_cpts_global[0], num_row_cpts_global[1]);

   //  my_first_row_cpt = num_row_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_row_cpts = num_row_cpts_global[1]; }
   nalu_hypre_MPI_Bcast(&total_global_row_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /* get the number of coarse rows */
   nalu_hypre_IntArrayData(wrap_cf) = col_cf_marker;
   nalu_hypre_BoomerAMGCoarseParms(comm, local_numrows, 1, NULL, wrap_cf, &coarse_dof_func_ptr,
                              num_col_cpts_global);
   nalu_hypre_IntArrayDestroy(coarse_dof_func_ptr);
   coarse_dof_func_ptr = NULL;

   //nalu_hypre_printf("my_id = %d, cpts_this = %d, cpts_next = %d\n", my_id, num_col_cpts_global[0], num_col_cpts_global[1]);

   //  my_first_col_cpt = num_col_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_col_cpts = num_col_cpts_global[1]; }
   nalu_hypre_MPI_Bcast(&total_global_col_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/
   if (debug_flag < 0)
   {
      debug_flag = -debug_flag;
   }

   //  if (debug_flag==4) wall_time = time_getWallclockSeconds();

   if (num_cols_A_offd) { CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST); }

   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                           num_sends), NALU_HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         int_buf_data[index++]
            = col_cf_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
   }

   comm_handle = nalu_hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                               CF_marker_offd);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of Ablock and fill in fine_to_coarse mapping.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   coarse_counter = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads, NALU_HYPRE_MEMORY_HOST);
   col_coarse_counter = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads, NALU_HYPRE_MEMORY_HOST);
   jj_count = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads, NALU_HYPRE_MEMORY_HOST);
   jj_count_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads, NALU_HYPRE_MEMORY_HOST);

   fine_to_coarse = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

   /* RDF: this looks a little tricky, but doable */
#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,i1,jj,ns,ne,size,rest) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (j = 0; j < num_threads; j++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;

      if (j < rest)
      {
         ns = j * size + j;
         ne = (j + 1) * size + j + 1;
      }
      else
      {
         ns = j * size + rest;
         ne = (j + 1) * size + rest;
      }
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
          *  If i is a F-point, we loop through the columns and select
          *  the F-columns. Also set up mapping vector.
          *--------------------------------------------------------------------*/

         if (col_cf_marker[i] > 0)
         {
            fine_to_coarse[i] = col_coarse_counter[j];
            col_coarse_counter[j]++;
         }

         if (row_cf_marker[i] > 0)
         {
            //fine_to_coarse[i] = coarse_counter[j];
            coarse_counter[j]++;
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];
               if (col_cf_marker[i1] > 0)
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];
                  if (CF_marker_offd[i1] > 0)
                  {
                     jj_count_offd[j]++;
                  }
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    *  Allocate  arrays.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < num_threads - 1; i++)
   {
      jj_count[i + 1] += jj_count[i];
      jj_count_offd[i + 1] += jj_count_offd[i];
      coarse_counter[i + 1] += coarse_counter[i];
      col_coarse_counter[i + 1] += col_coarse_counter[i];
   }
   i = num_threads - 1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];
   ii_counter = coarse_counter[i];

   Ablock_diag_size = jj_counter;

   Ablock_diag_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int, ii_counter + 1, memory_location);
   Ablock_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int, Ablock_diag_size, memory_location);
   Ablock_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, Ablock_diag_size, memory_location);

   Ablock_diag_i[ii_counter] = jj_counter;


   Ablock_offd_size = jj_counter_offd;

   Ablock_offd_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int, ii_counter + 1, memory_location);
   Ablock_offd_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int, Ablock_offd_size, memory_location);
   Ablock_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, Ablock_offd_size, memory_location);

   /*-----------------------------------------------------------------------
    *  Intialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   //-----------------------------------------------------------------------
   //  Send and receive fine_to_coarse info.
   //-----------------------------------------------------------------------

   //  if (debug_flag==4) wall_time = time_getWallclockSeconds();
#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,ns,ne,size,rest,coarse_shift) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (j = 0; j < num_threads; j++)
   {
      coarse_shift = 0;
      if (j > 0) { coarse_shift = col_coarse_counter[j - 1]; }
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (j < rest)
      {
         ns = j * size + j;
         ne = (j + 1) * size + j + 1;
      }
      else
      {
         ns = j * size + rest;
         ne = (j + 1) * size + rest;
      }
      for (i = ns; i < ne; i++)
      {
         fine_to_coarse[i] += coarse_shift;
      }
   }

   //  if (debug_flag==4) wall_time = time_getWallclockSeconds();
#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   //  for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_col_cpt;

#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,jl,i1,jj,ns,ne,size,rest,jj_counter,jj_counter_offd,ii_counter) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (jl = 0; jl < num_threads; jl++)
   {
      size = n_fine / num_threads;
      rest = n_fine - size * num_threads;
      if (jl < rest)
      {
         ns = jl * size + jl;
         ne = (jl + 1) * size + jl + 1;
      }
      else
      {
         ns = jl * size + rest;
         ne = (jl + 1) * size + rest;
      }
      jj_counter = 0;
      if (jl > 0) { jj_counter = jj_count[jl - 1]; }
      jj_counter_offd = 0;
      if (jl > 0) { jj_counter_offd = jj_count_offd[jl - 1]; }
      ii_counter = 0;
      for (i = ns; i < ne; i++)
      {
         /*--------------------------------------------------------------------
          *  If i is a F-point, we loop through the columns and select
          *  the F-columns. Also set up mapping vector.
          *--------------------------------------------------------------------*/
         if (row_cf_marker[i] > 0)
         {
            // Diagonal part of Ablock //
            Ablock_diag_i[ii_counter] = jj_counter;
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];
               if (col_cf_marker[i1] > 0)
               {
                  Ablock_diag_j[jj_counter]    = fine_to_coarse[i1];
                  Ablock_diag_data[jj_counter] = A_diag_data[jj];
                  jj_counter++;
               }
            }

            // Off-Diagonal part of Ablock //
            Ablock_offd_i[ii_counter] = jj_counter_offd;
            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];
                  if (CF_marker_offd[i1] > 0)
                  {
                     Ablock_offd_j[jj_counter_offd]  = i1;
                     Ablock_offd_data[jj_counter_offd] = A_offd_data[jj];
                     jj_counter_offd++;
                  }
               }
            }
            ii_counter++;
         }
      }
      Ablock_offd_i[ii_counter] = jj_counter_offd;
      Ablock_diag_i[ii_counter] = jj_counter;
   }
   Ablock = nalu_hypre_ParCSRMatrixCreate(comm,
                                     total_global_row_cpts,
                                     total_global_col_cpts,
                                     num_row_cpts_global,
                                     num_col_cpts_global,
                                     0,
                                     Ablock_diag_i[ii_counter],
                                     Ablock_offd_i[ii_counter]);

   Ablock_diag = nalu_hypre_ParCSRMatrixDiag(Ablock);
   nalu_hypre_CSRMatrixData(Ablock_diag) = Ablock_diag_data;
   nalu_hypre_CSRMatrixI(Ablock_diag) = Ablock_diag_i;
   nalu_hypre_CSRMatrixJ(Ablock_diag) = Ablock_diag_j;
   Ablock_offd = nalu_hypre_ParCSRMatrixOffd(Ablock);
   nalu_hypre_CSRMatrixData(Ablock_offd) = Ablock_offd_data;
   nalu_hypre_CSRMatrixI(Ablock_offd) = Ablock_offd_i;
   nalu_hypre_CSRMatrixJ(Ablock_offd) = Ablock_offd_j;

   num_cols_Ablock_offd = 0;

   if (Ablock_offd_size)
   {
      Ablock_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
#if 0
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i = 0; i < num_cols_A_offd; i++)
      {
         Ablock_marker[i] = 0;
      }
      num_cols_Ablock_offd = 0;
      for (i = 0; i < Ablock_offd_size; i++)
      {
         index = Ablock_offd_j[i];
         if (!Ablock_marker[index])
         {
            num_cols_Ablock_offd++;
            Ablock_marker[index] = 1;
         }
      }

      col_map_offd_Ablock = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_Ablock_offd, memory_location);
      tmp_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_Ablock_offd, NALU_HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_cols_Ablock_offd; i++)
      {
         while (Ablock_marker[index] == 0) { index++; }
         tmp_map_offd[i] = index++;
      }
#if 0
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
      for (i = 0; i < Ablock_offd_size; i++)
         Ablock_offd_j[i] = nalu_hypre_BinarySearch(tmp_map_offd,
                                               Ablock_offd_j[i],
                                               num_cols_Ablock_offd);
      nalu_hypre_TFree(Ablock_marker, NALU_HYPRE_MEMORY_HOST);
   }

   if (num_cols_Ablock_offd)
   {
      nalu_hypre_ParCSRMatrixColMapOffd(Ablock) = col_map_offd_Ablock;
      nalu_hypre_CSRMatrixNumCols(Ablock_offd) = num_cols_Ablock_offd;
   }

   nalu_hypre_GetCommPkgRTFromCommPkgA(Ablock, A, fine_to_coarse, tmp_map_offd);

   /* Create the assumed partition */
   if (nalu_hypre_ParCSRMatrixAssumedPartition(Ablock) == NULL)
   {
      nalu_hypre_ParCSRMatrixCreateAssumedPartition(Ablock);
   }

   *A_block_ptr = Ablock;

   nalu_hypre_TFree(tmp_map_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(coarse_counter, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(col_coarse_counter, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_IntArrayData(wrap_cf) = NULL;
   nalu_hypre_IntArrayDestroy(wrap_cf);

   return nalu_hypre_error_flag;
}

/* Build A_FF matrix from A given a CF_marker array */
NALU_HYPRE_Int
nalu_hypre_MGRBuildAff( nalu_hypre_ParCSRMatrix   *A,
                   NALU_HYPRE_Int            *CF_marker,
                   NALU_HYPRE_Int             debug_flag,
                   nalu_hypre_ParCSRMatrix  **A_ff_ptr )
{
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int local_numrows = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));
   /* create a copy of the CF_marker array and switch C-points to F-points */
   NALU_HYPRE_Int *CF_marker_copy = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_numrows, NALU_HYPRE_MEMORY_HOST);

#if 0
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
#endif
   for (i = 0; i < local_numrows; i++)
   {
      CF_marker_copy[i] = -CF_marker[i];
   }

   nalu_hypre_MGRGetSubBlock(A, CF_marker_copy, CF_marker_copy, debug_flag, A_ff_ptr);

   /* Free copy of CF marker */
   nalu_hypre_TFree(CF_marker_copy, NALU_HYPRE_MEMORY_HOST);
   return (0);
}

/*********************************************************************************
 * This routine assumes that the 'toVector' is larger than the 'fromVector' and
 * the CF_marker is of the same length as the toVector. There must be n 'point_type'
 * values in the CF_marker, where n is the length of the 'fromVector'.
 * It adds the values of the 'fromVector' to the 'toVector' where the marker is the
 * same as the 'point_type'
 *********************************************************************************/
NALU_HYPRE_Int
nalu_hypre_MGRAddVectorP ( nalu_hypre_IntArray  *CF_marker,
                      NALU_HYPRE_Int        point_type,
                      NALU_HYPRE_Real       a,
                      nalu_hypre_ParVector  *fromVector,
                      NALU_HYPRE_Real       b,
                      nalu_hypre_ParVector  **toVector )
{
   nalu_hypre_Vector    *fromVectorLocal = nalu_hypre_ParVectorLocalVector(fromVector);
   NALU_HYPRE_Real      *fromVectorData  = nalu_hypre_VectorData(fromVectorLocal);
   nalu_hypre_Vector    *toVectorLocal   = nalu_hypre_ParVectorLocalVector(*toVector);
   NALU_HYPRE_Real      *toVectorData    = nalu_hypre_VectorData(toVectorLocal);
   NALU_HYPRE_Int       *CF_marker_data = nalu_hypre_IntArrayData(CF_marker);

   //NALU_HYPRE_Int       n = nalu_hypre_ParVectorActualLocalSize(*toVector);
   NALU_HYPRE_Int       n = nalu_hypre_IntArraySize(CF_marker);
   NALU_HYPRE_Int       i, j;

   j = 0;
   for (i = 0; i < n; i++)
   {
      if (CF_marker_data[i] == point_type)
      {
         toVectorData[i] = b * toVectorData[i] + a * fromVectorData[j];
         j++;
      }
   }
   return 0;
}

/*************************************************************************************
 * This routine assumes that the 'fromVector' is larger than the 'toVector' and
 * the CF_marker is of the same length as the fromVector. There must be n 'point_type'
 * values in the CF_marker, where n is the length of the 'toVector'.
 * It adds the values of the 'fromVector' where the marker is the
 * same as the 'point_type' to the 'toVector'
 *************************************************************************************/
NALU_HYPRE_Int
nalu_hypre_MGRAddVectorR ( nalu_hypre_IntArray *CF_marker,
                      NALU_HYPRE_Int        point_type,
                      NALU_HYPRE_Real       a,
                      nalu_hypre_ParVector  *fromVector,
                      NALU_HYPRE_Real       b,
                      nalu_hypre_ParVector  **toVector )
{
   nalu_hypre_Vector    *fromVectorLocal = nalu_hypre_ParVectorLocalVector(fromVector);
   NALU_HYPRE_Real      *fromVectorData  = nalu_hypre_VectorData(fromVectorLocal);
   nalu_hypre_Vector    *toVectorLocal   = nalu_hypre_ParVectorLocalVector(*toVector);
   NALU_HYPRE_Real      *toVectorData    = nalu_hypre_VectorData(toVectorLocal);
   NALU_HYPRE_Int       *CF_marker_data = nalu_hypre_IntArrayData(CF_marker);

   //NALU_HYPRE_Int       n = nalu_hypre_ParVectorActualLocalSize(*toVector);
   NALU_HYPRE_Int       n = nalu_hypre_IntArraySize(CF_marker);
   NALU_HYPRE_Int       i, j;

   j = 0;
   for (i = 0; i < n; i++)
   {
      if (CF_marker_data[i] == point_type)
      {
         toVectorData[j] = b * toVectorData[j] + a * fromVectorData[i];
         j++;
      }
   }
   return 0;
}

/*
NALU_HYPRE_Int
nalu_hypre_MGRBuildAffRAP( MPI_Comm comm, NALU_HYPRE_Int local_num_variables, NALU_HYPRE_Int num_functions,
  NALU_HYPRE_Int *dof_func, NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int **coarse_dof_func_ptr, NALU_HYPRE_BigInt **coarse_pnts_global_ptr,
  nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int debug_flag, nalu_hypre_ParCSRMatrix **P_f_ptr, nalu_hypre_ParCSRMatrix **A_ff_ptr )
{
  NALU_HYPRE_Int *CF_marker_copy = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_num_variables, NALU_HYPRE_MEMORY_HOST);
  NALU_HYPRE_Int i;
  for (i = 0; i < local_num_variables; i++) {
    CF_marker_copy[i] = -CF_marker[i];
  }

  nalu_hypre_BoomerAMGCoarseParms(comm, local_num_variables, 1, NULL, CF_marker_copy, coarse_dof_func_ptr, coarse_pnts_global_ptr);
  nalu_hypre_MGRBuildP(A, CF_marker_copy, (*coarse_pnts_global_ptr), 0, debug_flag, P_f_ptr);
  nalu_hypre_BoomerAMGBuildCoarseOperator(*P_f_ptr, A, *P_f_ptr, A_ff_ptr);

  nalu_hypre_TFree(CF_marker_copy, NALU_HYPRE_MEMORY_HOST);
  return 0;
}
*/

/* Get pointer to coarse grid matrix for MGR solver */
NALU_HYPRE_Int
nalu_hypre_MGRGetCoarseGridMatrix( void *mgr_vdata, nalu_hypre_ParCSRMatrix **RAP )
{
   nalu_hypre_ParMGRData  *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (mgr_data -> RAP == NULL)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        " Coarse grid matrix is NULL. Please make sure MGRSetup() is called \n");
      return nalu_hypre_error_flag;
   }
   *RAP = mgr_data->RAP;

   return nalu_hypre_error_flag;
}

/* Get pointer to coarse grid solution for MGR solver */
NALU_HYPRE_Int
nalu_hypre_MGRGetCoarseGridSolution( void *mgr_vdata, nalu_hypre_ParVector **sol )
{
   nalu_hypre_ParMGRData  *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (mgr_data -> U_array == NULL)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        " MGR solution array is NULL. Please make sure MGRSetup() and MGRSolve() are called \n");
      return nalu_hypre_error_flag;
   }
   *sol = mgr_data->U_array[mgr_data->num_coarse_levels];

   return nalu_hypre_error_flag;
}

/* Get pointer to coarse grid solution for MGR solver */
NALU_HYPRE_Int
nalu_hypre_MGRGetCoarseGridRHS( void *mgr_vdata, nalu_hypre_ParVector **rhs )
{
   nalu_hypre_ParMGRData  *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;

   if (!mgr_data)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (mgr_data -> F_array == NULL)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        " MGR RHS array is NULL. Please make sure MGRSetup() and MGRSolve() are called \n");
      return nalu_hypre_error_flag;
   }
   *rhs = mgr_data->F_array[mgr_data->num_coarse_levels];

   return nalu_hypre_error_flag;
}

/* Print coarse grid linear system (for debugging)*/
NALU_HYPRE_Int
nalu_hypre_MGRPrintCoarseSystem( void *mgr_vdata, NALU_HYPRE_Int print_flag)
{
   nalu_hypre_ParMGRData  *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   mgr_data->print_coarse_system = print_flag;

   return nalu_hypre_error_flag;
}

/* Print solver params */
NALU_HYPRE_Int
nalu_hypre_MGRWriteSolverParams(void *mgr_vdata)
{
   nalu_hypre_ParMGRData  *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   NALU_HYPRE_Int i, j;
   NALU_HYPRE_Int max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   nalu_hypre_printf("MGR Setup parameters: \n");
   nalu_hypre_printf("Block size: %d\n", (mgr_data -> block_size));
   nalu_hypre_printf("Max number of coarse levels: %d\n", (mgr_data -> max_num_coarse_levels));
   //   nalu_hypre_printf("Relax type: %d\n", (mgr_data -> relax_type));
   nalu_hypre_printf("Set non-Cpoints to F-points: %d\n", (mgr_data -> set_non_Cpoints_to_F));
   nalu_hypre_printf("Set Cpoints method: %d\n", (mgr_data -> set_c_points_method));
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      nalu_hypre_printf("Lev = %d, Interpolation type: %d\n", i, (mgr_data -> interp_type)[i]);
      nalu_hypre_printf("Lev = %d, Restriction type: %d\n", i, (mgr_data -> restrict_type)[i]);
      nalu_hypre_printf("Lev = %d, F-relaxation type: %d\n", i, (mgr_data -> Frelax_type)[i]);
      nalu_hypre_printf("lev = %d, Number of relax sweeps: %d\n", i, (mgr_data -> num_relax_sweeps)[i]);
      nalu_hypre_printf("Lev = %d, Use non-Galerkin coarse grid: %d\n", i,
                   (mgr_data -> mgr_coarse_grid_method)[i]);
      NALU_HYPRE_Int lvl_num_coarse_points = (mgr_data -> block_num_coarse_indexes)[i];
      nalu_hypre_printf("Lev = %d, Number of Cpoints: %d\n", i, lvl_num_coarse_points);
      nalu_hypre_printf("Cpoints indices: ");
      for (j = 0; j < lvl_num_coarse_points; j++)
      {
         if ((mgr_data -> block_cf_marker)[i][j] == 1)
         {
            nalu_hypre_printf("%d ", j);
         }
      }
      nalu_hypre_printf("\n");
   }
   nalu_hypre_printf("Number of Reserved Cpoints: %d\n", (mgr_data -> reserved_coarse_size));
   nalu_hypre_printf("Keep reserved Cpoints to level: %d\n", (mgr_data -> lvl_to_keep_cpoints));

   nalu_hypre_printf("\n MGR Solver Parameters: \n");
   nalu_hypre_printf("Number of interpolation sweeps: %d\n", (mgr_data -> num_interp_sweeps));
   nalu_hypre_printf("Number of restriction sweeps: %d\n", (mgr_data -> num_restrict_sweeps));
   if (mgr_data -> level_smooth_type != NULL)
   {
      nalu_hypre_printf("Global smoother type: %d\n", (mgr_data -> level_smooth_type)[0]);
      nalu_hypre_printf("Number of global smoother sweeps: %d\n", (mgr_data -> level_smooth_iters)[0]);
   }
   nalu_hypre_printf("Max number of iterations: %d\n", (mgr_data -> max_iter));
   nalu_hypre_printf("Stopping tolerance: %e\n", (mgr_data -> tol));
   nalu_hypre_printf("Use default coarse grid solver: %d\n", (mgr_data -> use_default_cgrid_solver));
   /*
      if ((mgr_data -> fsolver_mode) >= 0)
      {
         nalu_hypre_printf("Use AMG solver for full AMG F-relaxation: %d\n", (mgr_data -> fsolver_mode));
      }
   */
   return nalu_hypre_error_flag;
}

#ifdef NALU_HYPRE_USING_DSUPERLU
void *
nalu_hypre_MGRDirectSolverCreate()
{
   //   nalu_hypre_DSLUData *dslu_data = nalu_hypre_CTAlloc(nalu_hypre_DSLUData, 1, NALU_HYPRE_MEMORY_HOST);
   //   return (void *) dslu_data;
   return NULL;
}

NALU_HYPRE_Int
nalu_hypre_MGRDirectSolverSetup( void                *solver,
                            nalu_hypre_ParCSRMatrix  *A,
                            nalu_hypre_ParVector     *f,
                            nalu_hypre_ParVector     *u )
{
   NALU_HYPRE_Int ierr;
   ierr = nalu_hypre_SLUDistSetup( solver, A, 0);

   return ierr;
}
NALU_HYPRE_Int
nalu_hypre_MGRDirectSolverSolve( void                *solver,
                            nalu_hypre_ParCSRMatrix  *A,
                            nalu_hypre_ParVector     *f,
                            nalu_hypre_ParVector     *u )
{
   nalu_hypre_SLUDistSolve(solver, f, u);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_MGRDirectSolverDestroy( void *solver )
{
   nalu_hypre_SLUDistDestroy(solver);

   return nalu_hypre_error_flag;
}
#endif
