/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "par_mgr.h"
#include "par_amg.h"

/* Setup MGR data */
NALU_HYPRE_Int
nalu_hypre_MGRSetup( void               *mgr_vdata,
                nalu_hypre_ParCSRMatrix *A,
                nalu_hypre_ParVector    *f,
                nalu_hypre_ParVector    *u )
{
   MPI_Comm           comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;

   NALU_HYPRE_Int       i, j, final_coarse_size, block_size, idx, **block_cf_marker;
   NALU_HYPRE_Int       *block_num_coarse_indexes, *point_marker_array;
   NALU_HYPRE_BigInt    row, end_idx;
   NALU_HYPRE_Int    lev, num_coarsening_levs, last_level;
   NALU_HYPRE_Int    num_c_levels, nc, index_i, cflag;
   NALU_HYPRE_Int      set_c_points_method;
   NALU_HYPRE_Int    debug_flag = 0;
   NALU_HYPRE_Int    block_jacobi_bsize;
   NALU_HYPRE_Int    *blk_size = mgr_data -> blk_size;
   //  NALU_HYPRE_Int  num_threads;

   nalu_hypre_ParCSRMatrix  *RT = NULL;
   nalu_hypre_ParCSRMatrix  *P = NULL;
   nalu_hypre_ParCSRMatrix  *S = NULL;
   nalu_hypre_ParCSRMatrix  *ST = NULL;
   nalu_hypre_ParCSRMatrix  *AT = NULL;
   nalu_hypre_ParCSRMatrix  *Wp = NULL;

   nalu_hypre_IntArray      *dof_func_buff = NULL;
   NALU_HYPRE_Int           *dof_func_buff_data = NULL;
   NALU_HYPRE_BigInt         coarse_pnts_global[2];
   nalu_hypre_Vector       **l1_norms = NULL;

   nalu_hypre_ParVector     *Ztemp;
   nalu_hypre_ParVector     *Vtemp;
   nalu_hypre_ParVector     *Utemp;
   nalu_hypre_ParVector     *Ftemp;

   /* pointers to mgr data */
   NALU_HYPRE_Int  use_default_cgrid_solver = (mgr_data -> use_default_cgrid_solver);
   NALU_HYPRE_Int  logging = (mgr_data -> logging);
   NALU_HYPRE_Int  print_level = (mgr_data -> print_level);
   NALU_HYPRE_Int  relax_type = (mgr_data -> relax_type);
   NALU_HYPRE_Int  relax_order = (mgr_data -> relax_order);

   NALU_HYPRE_Int  *interp_type = (mgr_data -> interp_type);
   NALU_HYPRE_Int  *restrict_type = (mgr_data -> restrict_type);
   NALU_HYPRE_Int  *num_relax_sweeps = (mgr_data -> num_relax_sweeps);
   NALU_HYPRE_Int num_interp_sweeps = (mgr_data -> num_interp_sweeps);
   NALU_HYPRE_Int num_restrict_sweeps = (mgr_data -> num_interp_sweeps);
   NALU_HYPRE_Int max_elmts = (mgr_data -> P_max_elmts);
   NALU_HYPRE_Real   max_row_sum = (mgr_data -> max_row_sum);
   NALU_HYPRE_Real   strong_threshold = (mgr_data -> strong_threshold);
   NALU_HYPRE_Real   trunc_factor = (mgr_data -> trunc_factor);
   NALU_HYPRE_Int  old_num_coarse_levels = (mgr_data -> num_coarse_levels);
   NALU_HYPRE_Int  max_num_coarse_levels = (mgr_data -> max_num_coarse_levels);
   NALU_HYPRE_Int * reserved_Cpoint_local_indexes = (mgr_data -> reserved_Cpoint_local_indexes);
   nalu_hypre_IntArray      **CF_marker_array = (mgr_data -> CF_marker_array);
   NALU_HYPRE_Int            *CF_marker;
   nalu_hypre_ParCSRMatrix  **A_array = (mgr_data -> A_array);
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_ParCSRMatrix  **P_FF_array = (mgr_data -> P_FF_array);
#endif
   nalu_hypre_ParCSRMatrix  **P_array = (mgr_data -> P_array);
   nalu_hypre_ParCSRMatrix  **RT_array = (mgr_data -> RT_array);
   nalu_hypre_ParCSRMatrix  *RAP_ptr = NULL;

   nalu_hypre_ParCSRMatrix  *A_ff_ptr = NULL;
   NALU_HYPRE_Solver **aff_solver = (mgr_data -> aff_solver);
   nalu_hypre_ParCSRMatrix  **A_ff_array = (mgr_data -> A_ff_array);
   nalu_hypre_ParVector    **F_fine_array = (mgr_data -> F_fine_array);
   nalu_hypre_ParVector    **U_fine_array = (mgr_data -> U_fine_array);

   NALU_HYPRE_Int (*fine_grid_solver_setup)(void*, void*, void*, void*);
   NALU_HYPRE_Int (*fine_grid_solver_solve)(void*, void*, void*, void*);

   nalu_hypre_ParVector    **F_array = (mgr_data -> F_array);
   nalu_hypre_ParVector    **U_array = (mgr_data -> U_array);
   nalu_hypre_ParVector    *residual = (mgr_data -> residual);
   NALU_HYPRE_Real    *rel_res_norms = (mgr_data -> rel_res_norms);
   NALU_HYPRE_Real    **frelax_diaginv = (mgr_data -> frelax_diaginv);
   NALU_HYPRE_Real    **level_diaginv = (mgr_data -> level_diaginv);

   NALU_HYPRE_Solver      default_cg_solver;
   NALU_HYPRE_Int (*coarse_grid_solver_setup)(void*, void*, void*, void*) = (NALU_HYPRE_Int (*)(void*, void*,
                                                                                      void*, void*)) (mgr_data -> coarse_grid_solver_setup);
   NALU_HYPRE_Int (*coarse_grid_solver_solve)(void*, void*, void*, void*) = (NALU_HYPRE_Int (*)(void*, void*,
                                                                                      void*, void*)) (mgr_data -> coarse_grid_solver_solve);

   NALU_HYPRE_Int    *level_smooth_type =  (mgr_data -> level_smooth_type);
   NALU_HYPRE_Int    *level_smooth_iters = (mgr_data -> level_smooth_iters);
   NALU_HYPRE_Solver *level_smoother = (mgr_data -> level_smoother);

   NALU_HYPRE_Int    reserved_coarse_size = (mgr_data -> reserved_coarse_size);

   NALU_HYPRE_Int      num_procs,  my_id;
   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int             n       = nalu_hypre_CSRMatrixNumRows(A_diag);

   NALU_HYPRE_Int use_VcycleSmoother = 0;
   //   NALU_HYPRE_Int use_GSElimSmoother = 0;
   //   NALU_HYPRE_Int use_ComplexSmoother = 0;
   nalu_hypre_ParVector     *VcycleRelaxZtemp;
   nalu_hypre_ParVector     *VcycleRelaxVtemp;
   nalu_hypre_ParAMGData    **FrelaxVcycleData;
   NALU_HYPRE_Int *Frelax_method = (mgr_data -> Frelax_method);
   NALU_HYPRE_Int *Frelax_num_functions = (mgr_data -> Frelax_num_functions);

   NALU_HYPRE_Int *Frelax_type = (mgr_data -> Frelax_type);

   NALU_HYPRE_Int *mgr_coarse_grid_method = (mgr_data -> mgr_coarse_grid_method);

   NALU_HYPRE_Int use_air = 0;
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( memory_location );
   NALU_HYPRE_Real truncate_cg_threshold = (mgr_data -> truncate_coarse_grid_threshold);
   NALU_HYPRE_Real wall_time;
   //   NALU_HYPRE_Real wall_time_lev;

   /* ----- begin -----*/
   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;
   //  num_threads = nalu_hypre_NumThreads();

   block_size = (mgr_data -> block_size);
   block_jacobi_bsize = (mgr_data -> block_jacobi_bsize);
   block_cf_marker = (mgr_data -> block_cf_marker);
   block_num_coarse_indexes = (mgr_data -> block_num_coarse_indexes);
   point_marker_array = (mgr_data -> point_marker_array);
   set_c_points_method = (mgr_data -> set_c_points_method);

   NALU_HYPRE_Int **level_coarse_indexes = NULL;
   NALU_HYPRE_Int *level_coarse_size = NULL;
   NALU_HYPRE_Int setNonCpointToF = (mgr_data -> set_non_Cpoints_to_F);
   NALU_HYPRE_BigInt *reserved_coarse_indexes = (mgr_data -> reserved_coarse_indexes);
   NALU_HYPRE_BigInt *idx_array = (mgr_data -> idx_array);
   NALU_HYPRE_Int lvl_to_keep_cpoints = (mgr_data -> lvl_to_keep_cpoints) > (mgr_data ->
                                                                        max_num_coarse_levels) ? (mgr_data -> max_num_coarse_levels) : (mgr_data -> lvl_to_keep_cpoints);

   NALU_HYPRE_Int nloc =  nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));
   NALU_HYPRE_BigInt ilower =  nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   NALU_HYPRE_BigInt iupper =  nalu_hypre_ParCSRMatrixLastRowIndex(A);

   //nalu_hypre_ParAMGData    **GSElimData;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* Trivial case: simply solve the coarse level problem */
   if ( block_size < 2 || (mgr_data -> max_num_coarse_levels) < 1)
   {
      if (my_id == 0 && print_level > 0)
      {
         nalu_hypre_printf("Warning: Block size is < 2 or number of coarse levels is < 1. \n");
         nalu_hypre_printf("Solving scalar problem on fine grid using coarse level solver \n");
      }

      if (use_default_cgrid_solver)
      {
         if (my_id == 0 && print_level > 0)
         {
            nalu_hypre_printf("No coarse grid solver provided. Using default AMG solver ... \n");
         }

         /* create and set default solver parameters here */
         /* create and initialize default_cg_solver */
         default_cg_solver = (NALU_HYPRE_Solver) nalu_hypre_BoomerAMGCreate();
         nalu_hypre_BoomerAMGSetMaxIter ( default_cg_solver, (mgr_data -> max_iter) );

         nalu_hypre_BoomerAMGSetRelaxOrder( default_cg_solver, 1);
         nalu_hypre_BoomerAMGSetPrintLevel(default_cg_solver, 3);
         /* set setup and solve functions */
         coarse_grid_solver_setup = (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) nalu_hypre_BoomerAMGSetup;
         coarse_grid_solver_solve = (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) nalu_hypre_BoomerAMGSolve;
         (mgr_data -> coarse_grid_solver_setup) = coarse_grid_solver_setup;
         (mgr_data -> coarse_grid_solver_solve) = coarse_grid_solver_solve;
         (mgr_data -> coarse_grid_solver) = default_cg_solver;
      }

      // keep reserved coarse indexes to coarsest grid

      if (reserved_coarse_size > 0)
      {
         NALU_HYPRE_BoomerAMGSetCPoints((mgr_data ->coarse_grid_solver), 25, reserved_coarse_size,
                                   reserved_coarse_indexes);
      }

      /* setup coarse grid solver */
      coarse_grid_solver_setup((mgr_data -> coarse_grid_solver), A, f, u);
      (mgr_data -> num_coarse_levels) = 0;
      NALU_HYPRE_ANNOTATE_FUNC_END;

      return nalu_hypre_error_flag;
   }

   /* If we reduce the reserved C-points, increase one level */
   if (lvl_to_keep_cpoints > 0) { max_num_coarse_levels++; }
   /* Initialize local indexes of coarse sets at different levels */
   level_coarse_indexes = nalu_hypre_CTAlloc(NALU_HYPRE_Int*,  max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      level_coarse_indexes[i] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nloc, NALU_HYPRE_MEMORY_HOST);
   }

   level_coarse_size = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int reserved_cpoints_eliminated = 0;
   // loop over levels
   for (i = 0; i < max_num_coarse_levels; i++)
   {
      // if we want to reduce the reserved Cpoints, set the current level
      // coarse indices the same as the previous level
      if (i == lvl_to_keep_cpoints && i > 0)
      {
         reserved_cpoints_eliminated++;
         for (j = 0; j < final_coarse_size; j++)
         {
            level_coarse_indexes[i][j] = level_coarse_indexes[i - 1][j];
         }
         level_coarse_size[i] = final_coarse_size;
         continue;
      }
      final_coarse_size = 0;
      if (set_c_points_method == 0) // interleaved ordering, i.e. s1,p1,s2,p2,...
      {
         // loop over rows
         for (row = ilower; row <= iupper; row++)
         {
            idx = row % block_size;
            if (block_cf_marker[i - reserved_cpoints_eliminated][idx] == CMRK)
            {
               level_coarse_indexes[i][final_coarse_size++] = (NALU_HYPRE_Int)(row - ilower);
            }
         }
      }
      else if (set_c_points_method == 1) // block ordering s1,s2,...,p1,p2,...
      {
         for (j = 0; j < block_size; j++)
         {
            if (block_cf_marker[i - reserved_cpoints_eliminated][j] == CMRK)
            {
               if (j == block_size - 1)
               {
                  end_idx = iupper + 1;
               }
               else
               {
                  end_idx = idx_array[j + 1];
               }
               for (row = idx_array[j]; row < end_idx; row++)
               {
                  level_coarse_indexes[i][final_coarse_size++] = (NALU_HYPRE_Int)(row - ilower);
               }
            }
         }
         //nalu_hypre_printf("Level %d, # of coarse points %d\n", i, final_coarse_size);
      }
      else if (set_c_points_method == 2)
      {
         NALU_HYPRE_Int isCpoint;
         // row start from 0 since point_marker_array is local
         for (row = 0; row < nloc; row++)
         {
            isCpoint = 0;
            for (j = 0; j < block_num_coarse_indexes[i]; j++)
            {
               if (point_marker_array[row] == block_cf_marker[i][j])
               {
                  isCpoint = 1;
                  break;
               }
            }
            if (isCpoint)
            {
               level_coarse_indexes[i][final_coarse_size++] = row;
               //printf("%d\n",row);
            }
         }
      }
      else
      {
         if (my_id == 0)
         {
            nalu_hypre_printf("ERROR! Unknown method for setting C points.");
         }
         exit(-1);
      }
      level_coarse_size[i] = final_coarse_size;
   }

   // Set reserved coarse indexes to be kept to the coarsest level of the MGR solver
   if ((mgr_data -> reserved_Cpoint_local_indexes) != NULL)
   {
      nalu_hypre_TFree((mgr_data -> reserved_Cpoint_local_indexes), NALU_HYPRE_MEMORY_HOST);
   }
   if (reserved_coarse_size > 0)
   {
      (mgr_data -> reserved_Cpoint_local_indexes) = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  reserved_coarse_size,
                                                                  NALU_HYPRE_MEMORY_HOST);
      reserved_Cpoint_local_indexes = (mgr_data -> reserved_Cpoint_local_indexes);
      for (i = 0; i < reserved_coarse_size; i++)
      {
         row = reserved_coarse_indexes[i];
         NALU_HYPRE_Int local_row = (NALU_HYPRE_Int)(row - ilower);
         reserved_Cpoint_local_indexes[i] = local_row;
         NALU_HYPRE_Int lvl = lvl_to_keep_cpoints == 0 ? max_num_coarse_levels : lvl_to_keep_cpoints;
         if (set_c_points_method < 2)
         {
            idx = row % block_size;
            for (j = 0; j < lvl; j++)
            {
               if (block_cf_marker[j][idx] != CMRK)
               {
                  level_coarse_indexes[j][level_coarse_size[j]++] = local_row;
               }
            }
         }
         else
         {
            NALU_HYPRE_Int isCpoint = 0;
            for (j = 0; j < lvl; j++)
            {
               NALU_HYPRE_Int k;
               for (k = 0; k < block_num_coarse_indexes[j]; k++)
               {
                  if (point_marker_array[local_row] == block_cf_marker[j][k])
                  {
                     isCpoint = 1;
                     break;
                  }
               }
               if (!isCpoint)
               {
                  level_coarse_indexes[j][level_coarse_size[j]++] = local_row;
               }
            }
         }
      }
   }

   (mgr_data -> level_coarse_indexes) = level_coarse_indexes;
   (mgr_data -> num_coarse_per_level) = level_coarse_size;

   /* Free Previously allocated data, if any not destroyed */
   if (A_array || P_array || RT_array || CF_marker_array)
   {
      for (j = 1; j < (old_num_coarse_levels); j++)
      {
         if (A_array[j])
         {
            nalu_hypre_ParCSRMatrixDestroy(A_array[j]);
            A_array[j] = NULL;
         }
      }

      for (j = 0; j < old_num_coarse_levels; j++)
      {
         if (P_array[j])
         {
            nalu_hypre_ParCSRMatrixDestroy(P_array[j]);
            P_array[j] = NULL;
         }

         if (RT_array[j])
         {
            nalu_hypre_ParCSRMatrixDestroy(RT_array[j]);
            RT_array[j] = NULL;
         }

         if (CF_marker_array[j])
         {
            nalu_hypre_IntArrayDestroy(CF_marker_array[j]);
            CF_marker_array[j] = NULL;
         }
      }
      nalu_hypre_TFree(P_array, NALU_HYPRE_MEMORY_HOST);
      P_array = NULL;
      nalu_hypre_TFree(RT_array, NALU_HYPRE_MEMORY_HOST);
      RT_array = NULL;
      nalu_hypre_TFree(CF_marker_array, NALU_HYPRE_MEMORY_HOST);
      CF_marker_array = NULL;
   }

#if defined(NALU_HYPRE_USING_GPU)
   if (P_FF_array)
   {
      for (j = 0; j < old_num_coarse_levels; j++)
      {
         if (P_FF_array[j])
         {
            nalu_hypre_ParCSRMatrixDestroy(P_FF_array[j]);
            P_FF_array[j] = NULL;
         }
      }
      nalu_hypre_TFree(P_FF_array, NALU_HYPRE_MEMORY_HOST);
   }
#endif

   /* Free previously allocated FrelaxVcycleData if not destroyed */
   if ((mgr_data -> VcycleRelaxZtemp))
   {
      nalu_hypre_ParVectorDestroy((mgr_data -> VcycleRelaxZtemp));
      (mgr_data -> VcycleRelaxZtemp) = NULL;
   }
   if ((mgr_data -> VcycleRelaxVtemp))
   {
      nalu_hypre_ParVectorDestroy((mgr_data -> VcycleRelaxVtemp));
      (mgr_data -> VcycleRelaxVtemp) = NULL;
   }
   if ((mgr_data -> FrelaxVcycleData))
   {
      for (j = 0; j < old_num_coarse_levels; j++)
      {
         if ((mgr_data -> FrelaxVcycleData)[j])
         {
            nalu_hypre_MGRDestroyFrelaxVcycleData((mgr_data -> FrelaxVcycleData)[j]);
            (mgr_data -> FrelaxVcycleData)[j] = NULL;
         }
      }
      nalu_hypre_TFree((mgr_data -> FrelaxVcycleData), NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> FrelaxVcycleData) = NULL;
   }

   /* destroy previously allocated Gaussian Elim. data */
   if ((mgr_data -> GSElimData))
   {
      for (j = 0; j < old_num_coarse_levels; j++)
      {
         if ((mgr_data -> GSElimData)[j])
         {
            nalu_hypre_MGRDestroyGSElimData((mgr_data -> GSElimData)[j]);
            (mgr_data -> GSElimData)[j] = NULL;
         }
      }
      nalu_hypre_TFree((mgr_data -> GSElimData), NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> GSElimData) = NULL;
   }

   /* destroy final coarse grid matrix, if not previously destroyed */
   if ((mgr_data -> RAP))
   {
      nalu_hypre_ParCSRMatrixDestroy((mgr_data -> RAP));
      (mgr_data -> RAP) = NULL;
   }

   /* Setup for global block smoothers*/
   if (set_c_points_method == 0)
   {
      if (my_id == num_procs)
      {
         mgr_data -> n_block   = (n - reserved_coarse_size) / block_size;
         mgr_data -> left_size = n - block_size * (mgr_data -> n_block);
      }
      else
      {
         mgr_data -> n_block = n / block_size;
         mgr_data -> left_size = n - block_size * (mgr_data -> n_block);
      }
   }
   else
   {
      mgr_data -> n_block = n;
      mgr_data -> left_size = 0;
   }

   /* clear old l1_norm data, if created */
   if ((mgr_data -> l1_norms))
   {
      for (j = 0; j < (old_num_coarse_levels); j++)
      {
         if ((mgr_data -> l1_norms)[j])
         {
            nalu_hypre_SeqVectorDestroy((mgr_data -> l1_norms)[i]);
            //nalu_hypre_TFree((mgr_data -> l1_norms)[j], NALU_HYPRE_MEMORY_HOST);
            (mgr_data -> l1_norms)[j] = NULL;
         }
      }
      nalu_hypre_TFree((mgr_data -> l1_norms), NALU_HYPRE_MEMORY_HOST);
   }

   if ((mgr_data -> frelax_diaginv))
   {
      for (j = 0; j < (old_num_coarse_levels); j++)
      {
         if ((mgr_data -> frelax_diaginv)[j])
         {
            nalu_hypre_TFree((mgr_data -> frelax_diaginv)[j], NALU_HYPRE_MEMORY_HOST);
            (mgr_data -> frelax_diaginv)[j] = NULL;
         }
      }
      nalu_hypre_TFree((mgr_data -> frelax_diaginv), NALU_HYPRE_MEMORY_HOST);
   }

   if ((mgr_data -> level_diaginv))
   {
      for (j = 0; j < (old_num_coarse_levels); j++)
      {
         if ((mgr_data -> level_diaginv)[j])
         {
            nalu_hypre_TFree((mgr_data -> level_diaginv)[j], NALU_HYPRE_MEMORY_HOST);
            (mgr_data -> level_diaginv)[j] = NULL;
         }
      }
      nalu_hypre_TFree((mgr_data -> level_diaginv), NALU_HYPRE_MEMORY_HOST);
   }

   /* setup temporary storage */
   if ((mgr_data -> Ztemp))
   {
      nalu_hypre_ParVectorDestroy((mgr_data -> Ztemp));
      (mgr_data -> Ztemp) = NULL;
   }
   if ((mgr_data -> Vtemp))
   {
      nalu_hypre_ParVectorDestroy((mgr_data -> Vtemp));
      (mgr_data -> Vtemp) = NULL;
   }
   if ((mgr_data -> Utemp))
   {
      nalu_hypre_ParVectorDestroy((mgr_data -> Utemp));
      (mgr_data -> Utemp) = NULL;
   }
   if ((mgr_data -> Ftemp))
   {
      nalu_hypre_ParVectorDestroy((mgr_data -> Ftemp));
      (mgr_data -> Ftemp) = NULL;
   }
   if ((mgr_data -> residual))
   {
      nalu_hypre_ParVectorDestroy((mgr_data -> residual));
      (mgr_data -> residual) = NULL;
   }
   nalu_hypre_TFree((mgr_data -> rel_res_norms), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree((mgr_data -> blk_size), NALU_HYPRE_MEMORY_HOST);

   Vtemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                 nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize(Vtemp);
   (mgr_data ->Vtemp) = Vtemp;

   Ztemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                 nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize(Ztemp);
   (mgr_data -> Ztemp) = Ztemp;

   Utemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                 nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize(Utemp);
   (mgr_data ->Utemp) = Utemp;

   Ftemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                 nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize(Ftemp);
   (mgr_data ->Ftemp) = Ftemp;

   /* Allocate memory for level structure */
   if (A_array == NULL)
   {
      A_array = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix*,  max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   }
   if (P_array == NULL && max_num_coarse_levels > 0)
   {
      P_array = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix*,  max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   }
#if defined(NALU_HYPRE_USING_GPU)
   if (P_FF_array == NULL && max_num_coarse_levels > 0)
   {
      P_FF_array = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix*, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   }
#endif
   if (RT_array == NULL && max_num_coarse_levels > 0)
   {
      RT_array = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix*,  max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   }
   if (CF_marker_array == NULL)
   {
      CF_marker_array = nalu_hypre_CTAlloc(nalu_hypre_IntArray*,  max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   }

   /* Set default for Frelax_method if not set already -- Supports deprecated function */
   /*
      if (Frelax_method == NULL)
      {
         Frelax_method = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < max_num_coarse_levels; i++)
         {
            Frelax_method[i] = 0;
         }
         (mgr_data -> Frelax_method) = Frelax_method;
      }
   */
   /* Set default for Frelax_type if not set already.
    * We also consolidate inputs from relax_type and Frelax_method here.
    * This should be simplified once the other options are removed.
   */
   if (Frelax_type == NULL)
   {
      Frelax_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);

      /* Use relax type/ frelax_method data or set default type to use */
      if (Frelax_method)
      {
         for (i = 0; i < max_num_coarse_levels; i++)
         {
            Frelax_type[i] = Frelax_method[i] > 0 ? Frelax_method[i] : relax_type;
         }
      }
      else if (relax_type)
      {
         for (i = 0; i < max_num_coarse_levels; i++)
         {
            Frelax_type[i] = relax_type;
         }
      }
      else /* set default here */
      {
         for (i = 0; i < max_num_coarse_levels; i++)
         {
            Frelax_type[i] = 0;
         }
      }
      (mgr_data -> Frelax_type) = Frelax_type;
   }
   /* Set default for using non-Galerkin coarse grid */
   if (mgr_coarse_grid_method == NULL)
   {
      mgr_coarse_grid_method = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         mgr_coarse_grid_method[i] = 0;
      }
      (mgr_data -> mgr_coarse_grid_method) = mgr_coarse_grid_method;
   }

   /*
   if (Frelax_num_functions== NULL)
   {
     Frelax_num_functions = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
     for (i = 0; i < max_num_coarse_levels; i++)
     {
       Frelax_num_functions[i] = 1;
     }
     (mgr_data -> Frelax_num_functions) = Frelax_num_functions;
   }
   */
   /* Set default for interp_type and restrict_type if not set already */
   if (interp_type == NULL)
   {
      interp_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         interp_type[i] = 2;
      }
      (mgr_data -> interp_type) = interp_type;
   }
   if (restrict_type == NULL)
   {
      restrict_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         restrict_type[i] = 0;
      }
      (mgr_data -> restrict_type) = restrict_type;
   }
   if (num_relax_sweeps == NULL)
   {
      num_relax_sweeps = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         num_relax_sweeps[i] = 1;
      }
      (mgr_data -> num_relax_sweeps) = num_relax_sweeps;
   }

   /* set interp_type, restrict_type, and Frelax_type if we reduce the reserved C-points */
   reserved_cpoints_eliminated = 0;
   if (lvl_to_keep_cpoints > 0 && reserved_coarse_size > 0)
   {
      NALU_HYPRE_Int *level_interp_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Int *level_restrict_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Int *level_frelax_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         if (i == lvl_to_keep_cpoints)
         {
            level_interp_type[i] = 2;
            level_restrict_type[i] = 0;
            level_frelax_type[i] = 2; //99;
            reserved_cpoints_eliminated++;
         }
         else
         {
            level_interp_type[i] = interp_type[i - reserved_cpoints_eliminated];
            level_restrict_type[i] = restrict_type[i - reserved_cpoints_eliminated];
            level_frelax_type[i] = Frelax_type[i - reserved_cpoints_eliminated];
         }
      }
      nalu_hypre_TFree(interp_type, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(restrict_type, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(Frelax_type, NALU_HYPRE_MEMORY_HOST);
      interp_type = level_interp_type;
      restrict_type = level_restrict_type;
      Frelax_type = level_frelax_type;
      (mgr_data -> interp_type) = level_interp_type;
      (mgr_data -> restrict_type) = level_restrict_type;
      (mgr_data -> Frelax_type) = level_frelax_type;
   }

   /* set pointers to mgr data */
   (mgr_data -> A_array) = A_array;
   (mgr_data -> P_array) = P_array;
   (mgr_data -> RT_array) = RT_array;
   (mgr_data -> CF_marker_array) = CF_marker_array;
#if defined(NALU_HYPRE_USING_GPU)
   (mgr_data -> P_FF_array) = P_FF_array;
#endif

   /* Set up solution and rhs arrays */
   if (F_array != NULL || U_array != NULL)
   {
      for (j = 1; j < old_num_coarse_levels + 1; j++)
      {
         if (F_array[j] != NULL)
         {
            nalu_hypre_ParVectorDestroy(F_array[j]);
            F_array[j] = NULL;
         }
         if (U_array[j] != NULL)
         {
            nalu_hypre_ParVectorDestroy(U_array[j]);
            U_array[j] = NULL;
         }
      }
   }

   if (F_array == NULL)
   {
      F_array = nalu_hypre_CTAlloc(nalu_hypre_ParVector*,  max_num_coarse_levels + 1, NALU_HYPRE_MEMORY_HOST);
   }
   if (U_array == NULL)
   {
      U_array = nalu_hypre_CTAlloc(nalu_hypre_ParVector*,  max_num_coarse_levels + 1, NALU_HYPRE_MEMORY_HOST);
   }

   if (A_ff_array)
   {
      for (j = 0; j < old_num_coarse_levels - 1; j++)
      {
         if (A_ff_array[j])
         {
            nalu_hypre_ParCSRMatrixDestroy(A_ff_array[j]);
            A_ff_array[j] = NULL;
         }
      }
      nalu_hypre_TFree(A_ff_array, NALU_HYPRE_MEMORY_HOST);
      A_ff_array = NULL;
   }
   if ((mgr_data -> fine_grid_solver_setup) != NULL)
   {
      fine_grid_solver_setup = (mgr_data -> fine_grid_solver_setup);
   }
   else
   {
      fine_grid_solver_setup = (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) nalu_hypre_BoomerAMGSetup;
      (mgr_data -> fine_grid_solver_setup) = fine_grid_solver_setup;
   }
   if ((mgr_data -> fine_grid_solver_solve) != NULL)
   {
      fine_grid_solver_solve = (mgr_data -> fine_grid_solver_solve);
   }
   else
   {
      fine_grid_solver_solve = (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) nalu_hypre_BoomerAMGSolve;
      (mgr_data -> fine_grid_solver_solve) = fine_grid_solver_solve;
   }


   /* Set up solution and rhs arrays for Frelax */
   if (F_fine_array != NULL || U_fine_array != NULL)
   {
      for (j = 1; j < old_num_coarse_levels + 1; j++)
      {
         if (F_fine_array[j] != NULL)
         {
            nalu_hypre_ParVectorDestroy(F_fine_array[j]);
            F_fine_array[j] = NULL;
         }
         if (U_fine_array[j] != NULL)
         {
            nalu_hypre_ParVectorDestroy(U_fine_array[j]);
            U_fine_array[j] = NULL;
         }
      }
   }

   if (F_fine_array == NULL)
   {
      F_fine_array = nalu_hypre_CTAlloc(nalu_hypre_ParVector*,  max_num_coarse_levels + 1, NALU_HYPRE_MEMORY_HOST);
   }
   if (U_fine_array == NULL)
   {
      U_fine_array = nalu_hypre_CTAlloc(nalu_hypre_ParVector*,  max_num_coarse_levels + 1, NALU_HYPRE_MEMORY_HOST);
   }
   if (aff_solver == NULL)
   {
      aff_solver = nalu_hypre_CTAlloc(NALU_HYPRE_Solver*, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   }
   if (A_ff_array == NULL)
   {
      A_ff_array = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix*, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   }
   if (frelax_diaginv == NULL)
   {
      frelax_diaginv = nalu_hypre_CTAlloc(NALU_HYPRE_Real*, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   }
   if (level_diaginv == NULL)
   {
      level_diaginv = nalu_hypre_CTAlloc(NALU_HYPRE_Real*, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   }
   if (blk_size == NULL)
   {
      blk_size = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   }

   if (level_smooth_type == NULL)
   {
      level_smooth_type = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   }
   if (level_smooth_iters == NULL)
   {
      level_smooth_iters = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   }
   if (level_smoother == NULL)
   {
      level_smoother = nalu_hypre_CTAlloc(NALU_HYPRE_Solver, max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
   }

   /* set solution and rhs pointers */
   F_array[0] = f;
   U_array[0] = u;

   (mgr_data -> F_array) = F_array;
   (mgr_data -> U_array) = U_array;

   (mgr_data -> F_fine_array) = F_fine_array;
   (mgr_data -> U_fine_array) = U_fine_array;
   (mgr_data -> aff_solver) = aff_solver;
   (mgr_data -> A_ff_array) = A_ff_array;
   (mgr_data -> frelax_diaginv) = frelax_diaginv;
   (mgr_data -> level_diaginv) = level_diaginv;
   (mgr_data -> blk_size) = blk_size;
   (mgr_data -> level_smooth_type) = level_smooth_type;
   (mgr_data -> level_smooth_iters) = level_smooth_iters;
   (mgr_data -> level_smoother) = level_smoother;

   /* begin coarsening loop */
   num_coarsening_levs = max_num_coarse_levels;
   /* initialize level data matrix here */
   RAP_ptr = A;
   /* loop over levels of coarsening */
   for (lev = 0; lev < num_coarsening_levs; lev++)
   {
      // wall_time_lev = time_getWallclockSeconds();
      /* check if this is the last level */
      last_level = ((lev == num_coarsening_levs - 1));

      /* initialize A_array */
      A_array[lev] = RAP_ptr;
      nloc = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A_array[lev]));

      wall_time = time_getWallclockSeconds();
      if (level_smooth_iters[lev] > 0)
      {
         NALU_HYPRE_Int level_blk_size = lev == 0 ? block_size : block_num_coarse_indexes[lev - 1];
         if (level_smooth_type[lev] == 0 || level_smooth_type[lev] == 1)
         {
            nalu_hypre_MGRBlockRelaxSetup(A_array[lev], level_blk_size,
                                     &(mgr_data -> level_diaginv)[lev]);
         }
         else if (level_smooth_type[lev] == 8)
         {
            NALU_HYPRE_EuclidCreate(comm, &(level_smoother[lev]));
            NALU_HYPRE_EuclidSetLevel(level_smoother[lev], 0);
            NALU_HYPRE_EuclidSetBJ(level_smoother[lev], 1);
            NALU_HYPRE_EuclidSetup(level_smoother[lev], A_array[lev], NULL, NULL);
         }
         else if (level_smooth_type[lev] == 16)
         {
            NALU_HYPRE_ILUCreate(&(level_smoother[lev]));
            NALU_HYPRE_ILUSetType(level_smoother[lev], 0);
            NALU_HYPRE_ILUSetLevelOfFill(level_smoother[lev], 0);
            NALU_HYPRE_ILUSetMaxIter(level_smoother[lev], level_smooth_iters[lev]);
            NALU_HYPRE_ILUSetTol(level_smoother[lev], 0.0);
            NALU_HYPRE_ILUSetup(level_smoother[lev], A_array[lev], NULL, NULL);
         }
      }
      wall_time = time_getWallclockSeconds() - wall_time;
      // if (my_id == 0) { nalu_hypre_printf("Lev = %d, proc = %d     Global smoother setup: %f\n", lev, my_id, wall_time); }

      /* Compute strength matrix for interpolation operator - use default parameters, to be modified later */
      cflag = last_level || setNonCpointToF;
      if (!cflag || interp_type[lev] == 3 || interp_type[lev] == 5 || interp_type[lev] == 6 ||
          interp_type[lev] == 7)
      {
         nalu_hypre_BoomerAMGCreateS(A_array[lev], strong_threshold, max_row_sum, 1, NULL, &S);
      }

      /* Coarsen: Build CF_marker array based on rows of A */
      nalu_hypre_MGRCoarsen(S, A_array[lev], level_coarse_size[lev], level_coarse_indexes[lev], debug_flag,
                       &CF_marker_array[lev], cflag);
      CF_marker = nalu_hypre_IntArrayData(CF_marker_array[lev]);

      /*
      char fname[256];
      sprintf(fname,"CF_marker_lvl_%d_new.%05d", lev, my_id);
      FILE* fout;
      fout = fopen(fname,"w");
      for (i=0; i < nloc; i++)
      {
        fprintf(fout, "%d %d\n", i, CF_marker[i]);
      }
      fclose(fout);
      */

      /* Get global coarse sizes. Note that we assume num_functions = 1
       * so dof_func arrays are NULL */
      nalu_hypre_BoomerAMGCoarseParms(comm, nloc, 1, NULL, CF_marker_array[lev], &dof_func_buff,
                                 coarse_pnts_global);
      if (dof_func_buff)
      {
         dof_func_buff_data = nalu_hypre_IntArrayData(dof_func_buff);
      }
      /* Compute Petrov-Galerkin operators */
      /* Interpolation operator */
      num_interp_sweeps = (mgr_data -> num_interp_sweeps);

      if (mgr_data -> block_jacobi_bsize == 0)
      {
         block_jacobi_bsize = (lev == 0 ? block_size : block_num_coarse_indexes[lev - 1]) -
                              block_num_coarse_indexes[lev];
      }
      if (block_jacobi_bsize == 1 && interp_type[lev] == 12)
      {
         interp_type[lev] = 2;
      }
      //nalu_hypre_printf("MyID = %d, Lev = %d, Block jacobi size = %d\n", my_id, lev, block_jacobi_bsize);

      if (interp_type[lev] == 12)
      {
         if (mgr_coarse_grid_method[lev] != 0)
         {
            wall_time = time_getWallclockSeconds();
            nalu_hypre_MGRBuildBlockJacobiWp(A_array[lev], block_jacobi_bsize, CF_marker, coarse_pnts_global, &Wp);
            wall_time = time_getWallclockSeconds() - wall_time;
            //   if (my_id == 0) { nalu_hypre_printf("Lev = %d, interp type = %d, proc = %d     Build Wp: %f\n", lev, interp_type[lev], my_id, wall_time); }
         }
         wall_time = time_getWallclockSeconds();
         nalu_hypre_MGRBuildInterp(A_array[lev], CF_marker, Wp, coarse_pnts_global, 1, dof_func_buff_data,
                              debug_flag, trunc_factor, max_elmts, block_jacobi_bsize, &P, interp_type[lev], num_interp_sweeps);
         wall_time = time_getWallclockSeconds() - wall_time;
         //  if (my_id == 0) { nalu_hypre_printf("Lev = %d, interp type = %d, proc = %d     BuildInterp: %f\n", lev, interp_type[lev], my_id, wall_time); }
      }
      else
      {
         wall_time = time_getWallclockSeconds();
         nalu_hypre_MGRBuildInterp(A_array[lev], CF_marker, S, coarse_pnts_global, 1, dof_func_buff_data,
                              debug_flag, trunc_factor, max_elmts, block_jacobi_bsize, &P, interp_type[lev], num_interp_sweeps);
         wall_time = time_getWallclockSeconds() - wall_time;
         //  if (my_id == 0) { nalu_hypre_printf("Lev = %d, interp type = %d, proc = %d     BuildInterp: %f\n", lev, interp_type[lev], my_id, wall_time); }
      }
      /*
      char fname[256];
      sprintf(fname, "P_lev_%d", lev);
      nalu_hypre_ParCSRMatrixPrintIJ(P, 0, 0, fname);
      */
      /* Use block Jacobi F-relaxation with block Jacobi interpolation */
      if (interp_type[lev] == 12 && (mgr_data -> num_relax_sweeps)[lev] > 0)
      {
         NALU_HYPRE_Real *diag_inv = NULL;
         NALU_HYPRE_Int inv_size;
         nalu_hypre_ParCSRMatrixExtractBlockDiag(A_array[lev], block_jacobi_bsize, -1, CF_marker, &inv_size,
                                            &diag_inv, 1);
         frelax_diaginv[lev] = diag_inv;
         blk_size[lev] = block_jacobi_bsize;
         nalu_hypre_MGRBuildAff(A_array[lev], CF_marker, debug_flag, &A_ff_ptr);

         F_fine_array[lev + 1] =
            nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A_ff_ptr),
                                  nalu_hypre_ParCSRMatrixGlobalNumRows(A_ff_ptr),
                                  nalu_hypre_ParCSRMatrixRowStarts(A_ff_ptr));
         nalu_hypre_ParVectorInitialize(F_fine_array[lev + 1]);

         U_fine_array[lev + 1] =
            nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A_ff_ptr),
                                  nalu_hypre_ParCSRMatrixGlobalNumRows(A_ff_ptr),
                                  nalu_hypre_ParCSRMatrixRowStarts(A_ff_ptr));
         nalu_hypre_ParVectorInitialize(U_fine_array[lev + 1]);

         A_ff_array[lev] = A_ff_ptr;
      }

      P_array[lev] = P;

      if (restrict_type[lev] == 4)
      {
         use_air = 1;
      }
      else if (restrict_type[lev] == 5)
      {
         use_air = 2;
      }
      else
      {
         use_air = 0;
      }

      if (use_air)
      {
         NALU_HYPRE_Real   filter_thresholdR = 0.0;
         NALU_HYPRE_Int     gmres_switch = 64;
         NALU_HYPRE_Int     is_triangular = 0;

         /* for AIR, need absolute value SOC */

         nalu_hypre_BoomerAMGCreateSabs(A_array[lev], strong_threshold, 1.0, 1, NULL, &ST);

         /* !!! Ensure that CF_marker contains -1 or 1 !!! */
         /*
         for (i = 0; i < nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A_array[level])); i++)
         {
           CF_marker[i] = CF_marker[i] > 0 ? 1 : -1;
         }
         */
         if (use_air == 1) /* distance-1 AIR */
         {
            nalu_hypre_BoomerAMGBuildRestrAIR(A_array[lev], CF_marker,
                                         ST, coarse_pnts_global, 1,
                                         dof_func_buff_data, filter_thresholdR,
                                         debug_flag, &RT,
                                         is_triangular, gmres_switch);
         }
         else /* distance-1.5 AIR - distance 2 locally and distance 1 across procs. */
         {
            nalu_hypre_BoomerAMGBuildRestrDist2AIR(A_array[lev], CF_marker,
                                              ST, coarse_pnts_global, 1,
                                              dof_func_buff_data, debug_flag, filter_thresholdR,
                                              &RT,
                                              1, is_triangular, gmres_switch );
         }

         RT_array[lev] = RT;

         /* Use two matrix products to generate A_H */
         nalu_hypre_ParCSRMatrix *AP = NULL;
         AP  = nalu_hypre_ParMatmul(A_array[lev], P_array[lev]);
         RAP_ptr = nalu_hypre_ParMatmul(RT, AP);
         if (num_procs > 1)
         {
            nalu_hypre_MatvecCommPkgCreate(RAP_ptr);
         }
         /* Delete AP */
         nalu_hypre_ParCSRMatrixDestroy(AP);
      }
      else
      {
         if (mgr_coarse_grid_method[lev] != 0)
         {
            NALU_HYPRE_Int block_num_f_points = (lev == 0 ? block_size : block_num_coarse_indexes[lev - 1]) -
                                           block_num_coarse_indexes[lev];
            if (block_num_f_points == 1 && restrict_type[lev] == 12)
            {
               restrict_type[lev] = 2;
            }
            //            if (restrict_type[lev] > 0)
            {
               wall_time = time_getWallclockSeconds();

               nalu_hypre_MGRBuildRestrict(A_array[lev], CF_marker, coarse_pnts_global, 1, dof_func_buff_data,
                                      debug_flag, trunc_factor, max_elmts, strong_threshold, max_row_sum, block_num_f_points, &RT,
                                      restrict_type[lev], num_restrict_sweeps);
               wall_time = time_getWallclockSeconds() - wall_time;

               RT_array[lev] = RT;
            }

            wall_time = time_getWallclockSeconds();
            nalu_hypre_MGRComputeNonGalerkinCoarseGrid(A_array[lev], Wp, RT, block_num_f_points,
                                                  /* ordering */set_c_points_method, /* method (approx. inverse or not) */
                                                  mgr_coarse_grid_method[lev], max_elmts, CF_marker, &RAP_ptr);

            if (interp_type[lev] == 12)
            {
               nalu_hypre_ParCSRMatrixDeviceColMapOffd(Wp) = NULL;
               nalu_hypre_ParCSRMatrixColMapOffd(Wp)       = NULL;
               nalu_hypre_ParCSRMatrixDestroy(Wp);
               Wp = NULL;
            }
            wall_time = time_getWallclockSeconds() - wall_time;
            //  if (my_id == 0) { nalu_hypre_printf("Lev = %d, proc = %d     BuildCoarseGrid: %1.8f\n", lev, my_id, wall_time); }
         }
         else
         {
            wall_time = time_getWallclockSeconds();
            if (block_jacobi_bsize == 1 && restrict_type[lev] == 12)
            {
               restrict_type[lev] = 2;
            }
            nalu_hypre_MGRBuildRestrict(A_array[lev], CF_marker, coarse_pnts_global, 1, dof_func_buff_data,
                                   debug_flag, trunc_factor, max_elmts, strong_threshold, max_row_sum, block_jacobi_bsize, &RT,
                                   restrict_type[lev], num_restrict_sweeps);
            RT_array[lev] = RT;

            wall_time = time_getWallclockSeconds() - wall_time;
            // if (my_id == 0) { nalu_hypre_printf("Lev = %d, restrict type = %d, proc = %d     BuildRestrict: %f\n", lev, restrict_type[lev], my_id, wall_time); }

            wall_time = time_getWallclockSeconds();
            //nalu_hypre_BoomerAMGBuildCoarseOperator(RT, A_array[lev], P, &RAP_ptr);
            RAP_ptr = nalu_hypre_ParCSRMatrixRAPKT(RT, A_array[lev], P, 1);
            //char fname[256];
            //sprintf(fname, "RAP_%d", lev);
            //nalu_hypre_ParCSRMatrixPrintIJ(RAP_ptr, 0, 0, fname);
            wall_time = time_getWallclockSeconds() - wall_time;
            //  if (my_id == 0) { nalu_hypre_printf("Lev = %d, proc = %d     BuildCoarseGrid: %f\n", lev, my_id, wall_time); }
         }
      }

      if (truncate_cg_threshold > 0.0)
      {
         // truncate the coarse grid
         if (exec == NALU_HYPRE_EXEC_HOST)
         {
            nalu_hypre_ParCSRMatrixTruncate(RAP_ptr, truncate_cg_threshold, 0, 0, 0);
         }
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
         else
         {
            nalu_hypre_ParCSRMatrixDropSmallEntriesDevice(RAP_ptr, truncate_cg_threshold, -1);
         }
#endif
      }

      if (Frelax_type[lev] == 2) // full AMG
      {
         //wall_time = time_getWallclockSeconds();
         // user provided AMG solver
         // only support AMG at the first level
         // TODO: input check to avoid crashing
         //*** This part needs some refactoring wrt use of fsolver_mode - DOK ***/
         if (lev == 0 && (mgr_data -> fsolver_mode) == 0)
         {
            if (((nalu_hypre_ParAMGData*)aff_solver[lev])->A_array != NULL)
            {
               if (((nalu_hypre_ParAMGData*)aff_solver[lev])->A_array[0] != NULL)
               {
                  // F-solver is already set up, only need to store A_ff_ptr
                  A_ff_ptr = ((nalu_hypre_ParAMGData*)aff_solver[lev])->A_array[0];
               }
               else
               {
                  if (my_id == 0)
                  {
                     printf("Error!!! F-relaxation solver has not been setup.\n");
                     nalu_hypre_error(1);
                     return nalu_hypre_error_flag;
                  }
               }
            }
            else
            {
               // Compute A_ff and setup F-solver
               if (exec == NALU_HYPRE_EXEC_HOST)
               {
                  nalu_hypre_MGRBuildAff(A_array[lev], CF_marker, debug_flag, &A_ff_ptr);
               }
#if defined (NALU_HYPRE_USING_CUDA) || defined (NALU_HYPRE_USING_HIP)
               else
               {
                  nalu_hypre_ParCSRMatrixGenerateFFFCDevice(A_array[lev], CF_marker, coarse_pnts_global, NULL, NULL,
                                                       &A_ff_ptr);
               }
#endif
               fine_grid_solver_setup(aff_solver[lev], A_ff_ptr, F_fine_array[lev + 1], U_fine_array[lev + 1]);

               A_ff_array[lev] = A_ff_ptr;
               (mgr_data -> fsolver_mode) = 1;
            }
         }
         else // construct default AMG solver
         {
            if (exec == NALU_HYPRE_EXEC_HOST)
            {
               nalu_hypre_MGRBuildAff(A_array[lev], CF_marker, debug_flag, &A_ff_ptr);
            }
#if defined (NALU_HYPRE_USING_CUDA) || defined (NALU_HYPRE_USING_HIP)
            else
            {
               nalu_hypre_ParCSRMatrixGenerateFFFCDevice(A_array[lev], CF_marker, coarse_pnts_global, NULL, NULL,
                                                    &A_ff_ptr);
            }
#endif
            aff_solver[lev] = (NALU_HYPRE_Solver*) nalu_hypre_BoomerAMGCreate();
            nalu_hypre_BoomerAMGSetMaxIter(aff_solver[lev], (mgr_data -> num_relax_sweeps)[lev]);
            nalu_hypre_BoomerAMGSetTol(aff_solver[lev], 0.0);
            //nalu_hypre_BoomerAMGSetStrongThreshold(aff_solver[lev], 0.6);
#if defined(NALU_HYPRE_USING_GPU)
            nalu_hypre_BoomerAMGSetRelaxType(aff_solver[lev], 18);
            nalu_hypre_BoomerAMGSetCoarsenType(aff_solver[lev], 8);
            nalu_hypre_BoomerAMGSetNumSweeps(aff_solver[lev], 3);
#else
            nalu_hypre_BoomerAMGSetRelaxOrder(aff_solver[lev], 1);
#endif
            nalu_hypre_BoomerAMGSetPrintLevel(aff_solver[lev], mgr_data -> frelax_print_level);

            fine_grid_solver_setup(aff_solver[lev], A_ff_ptr, F_fine_array[lev + 1], U_fine_array[lev + 1]);

            (mgr_data -> fsolver_mode) = 2;
         }
         //wall_time = time_getWallclockSeconds() - wall_time;
         //nalu_hypre_printf("Lev = %d, proc = %d     SetupAFF: %f\n", lev, my_id, wall_time);

#if defined (NALU_HYPRE_USING_CUDA) || defined (NALU_HYPRE_USING_HIP)
         nalu_hypre_IntArray *F_marker = nalu_hypre_IntArrayCreate(nloc);
         nalu_hypre_IntArrayInitialize(F_marker);
         nalu_hypre_IntArraySetConstantValues(F_marker, 0);
         NALU_HYPRE_Int *F_marker_data = nalu_hypre_IntArrayData(F_marker);
         for (j = 0; j < nloc; j++)
         {
            F_marker_data[j] = -CF_marker[j];
         }
         NALU_HYPRE_BigInt num_fpts_global[2];
         nalu_hypre_ParCSRMatrix *P_FF_ptr;
         nalu_hypre_BoomerAMGCoarseParms(comm, nloc, 1, NULL, F_marker, NULL, num_fpts_global);
         nalu_hypre_MGRBuildPDevice(A_array[lev], F_marker_data, num_fpts_global, 0, &P_FF_ptr);
         P_FF_array[lev] = P_FF_ptr;

         nalu_hypre_IntArrayDestroy(F_marker);
#endif
         F_fine_array[lev + 1] =
            nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A_ff_ptr),
                                  nalu_hypre_ParCSRMatrixGlobalNumRows(A_ff_ptr),
                                  nalu_hypre_ParCSRMatrixRowStarts(A_ff_ptr));
         nalu_hypre_ParVectorInitialize(F_fine_array[lev + 1]);

         U_fine_array[lev + 1] =
            nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A_ff_ptr),
                                  nalu_hypre_ParCSRMatrixGlobalNumRows(A_ff_ptr),
                                  nalu_hypre_ParCSRMatrixRowStarts(A_ff_ptr));
         nalu_hypre_ParVectorInitialize(U_fine_array[lev + 1]);
         A_ff_array[lev] = A_ff_ptr;
      }

      /* Update coarse level indexes for next levels */
      if (lev < num_coarsening_levs - 1)
      {
         for (i = lev + 1; i < max_num_coarse_levels; i++)
         {
            // first mark indexes to be updated
            for (j = 0; j < level_coarse_size[i]; j++)
            {
               CF_marker[level_coarse_indexes[i][j]] = S_CMRK;
            }

            // next: loop over levels to update indexes
            nc = 0;
            index_i = 0;
            for (j = 0; j < nloc; j++)
            {
               if (CF_marker[j] == CMRK) { nc++; }
               if (CF_marker[j] == S_CMRK)
               {
                  level_coarse_indexes[i][index_i++] = nc++;
               }
               //if(index_i == level_coarse_size[i]) break;
            }
            nalu_hypre_assert(index_i == level_coarse_size[i]);

            // then: reset previously marked indexes
            for (j = 0; j < level_coarse_size[lev]; j++)
            {
               CF_marker[level_coarse_indexes[lev][j]] = CMRK;
            }
         }
      }
      // update reserved coarse indexes to be kept to coarsest level
      // first mark indexes to be updated
      // skip if we reduce the reserved C-points before the coarse grid solve
      if (mgr_data -> lvl_to_keep_cpoints == 0)
      {
         for (i = 0; i < reserved_coarse_size; i++)
         {
            CF_marker[reserved_Cpoint_local_indexes[i]] = S_CMRK;
         }
         // loop to update reserved Cpoints
         nc = 0;
         index_i = 0;
         for (i = 0; i < nloc; i++)
         {
            if (CF_marker[i] == CMRK) { nc++; }
            if (CF_marker[i] == S_CMRK)
            {
               reserved_Cpoint_local_indexes[index_i++] = nc++;
               // reset modified CF marker array indexes
               CF_marker[i] = CMRK;
            }
         }
      }

      /* allocate space for solution and rhs arrays */
      F_array[lev + 1] =
         nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(RAP_ptr),
                               nalu_hypre_ParCSRMatrixGlobalNumRows(RAP_ptr),
                               nalu_hypre_ParCSRMatrixRowStarts(RAP_ptr));
      nalu_hypre_ParVectorInitialize(F_array[lev + 1]);

      U_array[lev + 1] =
         nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(RAP_ptr),
                               nalu_hypre_ParCSRMatrixGlobalNumRows(RAP_ptr),
                               nalu_hypre_ParCSRMatrixRowStarts(RAP_ptr));
      nalu_hypre_ParVectorInitialize(U_array[lev + 1]);

      /* free memory before starting next level */
      nalu_hypre_ParCSRMatrixDestroy(S);
      S = NULL;

      if (!use_air)
      {
         nalu_hypre_ParCSRMatrixDestroy(AT);
         AT = NULL;
      }
      nalu_hypre_ParCSRMatrixDestroy(ST);
      ST = NULL;

      /* check if Vcycle smoother setup required */
      if ((mgr_data -> max_local_lvls) > 1)
      {
         if (Frelax_type[lev] == 1)
         {
            use_VcycleSmoother = 1;
            //            use_ComplexSmoother = 1;
         }
      }
      else
      {
         /* Only check for vcycle smoother option.
         * Currently leaves Frelax_type[lev] = 2 (full amg) option as is
         */
         if (Frelax_type[lev] == 1)
         {
            Frelax_type[lev] = 0;
         }
      }

      if (Frelax_type[lev] == 9 ||
          Frelax_type[lev] == 9 ||
          Frelax_type[lev] == 199 )
      {
         //         use_GSElimSmoother = 1;
         //         use_ComplexSmoother = 1;
      }

      /* check if last level */
      //      wall_time_lev = time_getWallclockSeconds() - wall_time_lev;
      //      if (my_id == 0) { nalu_hypre_printf("Lev = %d, proc = %d     Setup time: %f\n", lev, my_id, wall_time_lev); }

      if (last_level) { break; }
   }

   /* set pointer to last level matrix */
   num_c_levels = lev + 1;
   (mgr_data->num_coarse_levels) = num_c_levels;
   (mgr_data->RAP) = RAP_ptr;

   /* setup default coarse grid solver */
   /* default is BoomerAMG */
   if (use_default_cgrid_solver)
   {
      if (my_id == 0 && print_level > 0)
      {
         nalu_hypre_printf("No coarse grid solver provided. Using default AMG solver ... \n");
      }

      /* create and set default solver parameters here */
      default_cg_solver = (NALU_HYPRE_Solver) nalu_hypre_BoomerAMGCreate();
      nalu_hypre_BoomerAMGSetMaxIter ( default_cg_solver, 1 );
      nalu_hypre_BoomerAMGSetTol ( default_cg_solver, 0.0 );
      nalu_hypre_BoomerAMGSetRelaxOrder( default_cg_solver, 1);
      nalu_hypre_BoomerAMGSetPrintLevel(default_cg_solver, mgr_data -> cg_print_level);
      /* set setup and solve functions */
      coarse_grid_solver_setup =  (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) nalu_hypre_BoomerAMGSetup;
      coarse_grid_solver_solve =  (NALU_HYPRE_Int (*)(void*, void*, void*, void*)) nalu_hypre_BoomerAMGSolve;
      (mgr_data -> coarse_grid_solver_setup) =   coarse_grid_solver_setup;
      (mgr_data -> coarse_grid_solver_solve) =   coarse_grid_solver_solve;
      (mgr_data -> coarse_grid_solver) = default_cg_solver;
   }
   // keep reserved coarse indexes to coarsest grid
   if (reserved_coarse_size > 0 && lvl_to_keep_cpoints == 0)
   {
      ilower = nalu_hypre_ParCSRMatrixFirstRowIndex(RAP_ptr);
      for (i = 0; i < reserved_coarse_size; i++)
      {
         reserved_coarse_indexes[i] = (NALU_HYPRE_BigInt) (reserved_Cpoint_local_indexes[i] + ilower);
      }
      NALU_HYPRE_BoomerAMGSetCPoints((mgr_data ->coarse_grid_solver), 25, reserved_coarse_size,
                                reserved_coarse_indexes);
   }

   /* setup coarse grid solver */
   //   wall_time = time_getWallclockSeconds();
   coarse_grid_solver_setup((mgr_data -> coarse_grid_solver), RAP_ptr, F_array[num_c_levels],
                            U_array[num_c_levels]);
   //   wall_time = time_getWallclockSeconds() - wall_time;
   //   if (my_id == 0) { nalu_hypre_printf("Proc = %d   Coarse grid setup: %f\n", my_id, wall_time); }

   /* Setup smoother for fine grid */
   /* Always allocate l1_norms data, for now. Avoids looping over frelax_type -- DOK */
   //   if ( relax_type == 8 || relax_type == 13 || relax_type == 14 || relax_type == 18 )
   //   {
   l1_norms = nalu_hypre_CTAlloc(nalu_hypre_Vector*, num_c_levels, NALU_HYPRE_MEMORY_HOST);
   (mgr_data -> l1_norms) = l1_norms;
   //   }

   for (j = 0; j < num_c_levels; j++)
   {
      NALU_HYPRE_Int frelax_type = Frelax_type[j];
      if ((mgr_data -> num_relax_sweeps)[j] > 0)
      {
         NALU_HYPRE_Real *l1_norm_data = NULL;
         CF_marker = nalu_hypre_IntArrayData(CF_marker_array[j]);

         if (frelax_type == 8 || frelax_type == 13 || frelax_type == 14)
         {
            if (relax_order)
            {
               nalu_hypre_ParCSRComputeL1Norms(A_array[j], 4, CF_marker, &l1_norm_data);
            }
            else
            {
               nalu_hypre_ParCSRComputeL1Norms(A_array[j], 4, NULL, &l1_norm_data);
            }
         }
         else if (frelax_type == 18)
         {
            if (relax_order)
            {
               nalu_hypre_ParCSRComputeL1Norms(A_array[j], 1, CF_marker, &l1_norm_data);
            }
            else
            {
               nalu_hypre_ParCSRComputeL1Norms(A_array[j], 1, NULL, &l1_norm_data);
            }
         }

         if (l1_norm_data)
         {
            l1_norms[j] = nalu_hypre_SeqVectorCreate(nalu_hypre_ParCSRMatrixNumRows(A_array[j]));
            nalu_hypre_VectorData(l1_norms[j]) = l1_norm_data;
            nalu_hypre_SeqVectorInitialize_v2(l1_norms[j], nalu_hypre_ParCSRMatrixMemoryLocation(A_array[j]));
         }
      }
   }

   /* Setup Vcycle data for Frelax_type == 1 */
   if (use_VcycleSmoother)
   {
      /* allocate memory and set pointer to (mgr_data -> FrelaxVcycleData) */
      FrelaxVcycleData = nalu_hypre_CTAlloc(nalu_hypre_ParAMGData*,  max_num_coarse_levels, NALU_HYPRE_MEMORY_HOST);
      (mgr_data -> FrelaxVcycleData) = FrelaxVcycleData;
      if (use_VcycleSmoother)
      {
         /* setup temporary storage */
         VcycleRelaxVtemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                                  nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                                  nalu_hypre_ParCSRMatrixRowStarts(A));
         nalu_hypre_ParVectorInitialize(VcycleRelaxVtemp);
         (mgr_data ->VcycleRelaxVtemp) = VcycleRelaxVtemp;

         VcycleRelaxZtemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                                  nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                                  nalu_hypre_ParCSRMatrixRowStarts(A));
         nalu_hypre_ParVectorInitialize(VcycleRelaxZtemp);
         (mgr_data -> VcycleRelaxZtemp) = VcycleRelaxZtemp;

         /* loop over levels */
         for (i = 0; i < (mgr_data->num_coarse_levels); i++)
         {
            if (Frelax_type[i] == 1)
            {
               FrelaxVcycleData[i] = (nalu_hypre_ParAMGData*) nalu_hypre_MGRCreateFrelaxVcycleData();
               if (Frelax_num_functions != NULL)
               {
                  nalu_hypre_ParAMGDataNumFunctions(FrelaxVcycleData[i]) = Frelax_num_functions[i];
               }
               (FrelaxVcycleData[i] -> Vtemp) = VcycleRelaxVtemp;
               (FrelaxVcycleData[i] -> Ztemp) = VcycleRelaxZtemp;

               // setup variables for the V-cycle in the F-relaxation step //
               nalu_hypre_MGRSetupFrelaxVcycleData(mgr_data, A_array[i], F_array[i], U_array[i], i);
            }
         }
      }
#if 0
      else if (use_GSElimSmoother)
      {
         /* loop over levels */
         for (i = 0; i < (mgr_data->num_coarse_levels); i++)
         {
            if (Frelax_type[i] == 9 || Frelax_type[i] == 99 || Frelax_type[i] == 199)
            {
               FrelaxVcycleData[i] = (nalu_hypre_ParAMGData*) nalu_hypre_MGRCreateFrelaxVcycleData();
               (FrelaxVcycleData[i] -> A_Array) = A_Array;
               (FrelaxVcycleData[i] -> F_Array) = F_Array;
               (FrelaxVcycleData[i] -> U_Array) = U_Array;

               // setup variables for the V-cycle in the F-relaxation step //
               //               nalu_hypre_MGRSetupFrelaxVcycleData(mgr_data, A_array[i], F_array[i], U_array[i], i);
            }
         }
      }
#endif
   }

   if ( logging > 1 )
   {
      residual = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A_array[0]),
                                       nalu_hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                       nalu_hypre_ParCSRMatrixRowStarts(A_array[0]) );
      nalu_hypre_ParVectorInitialize(residual);
      (mgr_data -> residual) = residual;
   }
   else
   {
      (mgr_data -> residual) = NULL;
   }
   rel_res_norms = nalu_hypre_CTAlloc(NALU_HYPRE_Real, (mgr_data -> max_iter), NALU_HYPRE_MEMORY_HOST);
   (mgr_data -> rel_res_norms) = rel_res_norms;

   /* free level_coarse_indexes data */
   if ( level_coarse_indexes != NULL)
   {
      for (i = 0; i < max_num_coarse_levels; i++)
      {
         nalu_hypre_TFree(level_coarse_indexes[i], NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree( level_coarse_indexes, NALU_HYPRE_MEMORY_HOST);
      level_coarse_indexes = NULL;
      nalu_hypre_TFree(level_coarse_size, NALU_HYPRE_MEMORY_HOST);
      level_coarse_size = NULL;
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/* Setup data for Frelax V-cycle */
NALU_HYPRE_Int
nalu_hypre_MGRSetupFrelaxVcycleData( void *mgr_vdata,
                                nalu_hypre_ParCSRMatrix *A,
                                nalu_hypre_ParVector    *f,
                                nalu_hypre_ParVector    *u,
                                NALU_HYPRE_Int      lev )
{
   MPI_Comm           comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;
   nalu_hypre_ParAMGData    **FrelaxVcycleData = mgr_data -> FrelaxVcycleData;

   NALU_HYPRE_Int i, j, num_procs, my_id;

   NALU_HYPRE_Int max_local_lvls = (mgr_data -> max_local_lvls);
   NALU_HYPRE_Int lev_local;
   NALU_HYPRE_Int not_finished;
   NALU_HYPRE_Int max_local_coarse_size = nalu_hypre_ParAMGDataMaxCoarseSize(FrelaxVcycleData[lev]);
   nalu_hypre_IntArray       **CF_marker_array = (mgr_data -> CF_marker_array);
   NALU_HYPRE_Int local_size;
   NALU_HYPRE_BigInt coarse_size;

   NALU_HYPRE_BigInt     coarse_pnts_global_lvl[2];
   nalu_hypre_IntArray  *coarse_dof_func_lvl    = NULL;
   nalu_hypre_IntArray  *dof_func               = NULL;
   NALU_HYPRE_Int       *dof_func_data          = NULL;

   nalu_hypre_ParCSRMatrix *RAP_local = NULL;
   nalu_hypre_ParCSRMatrix *P_local = NULL;
   nalu_hypre_ParCSRMatrix *S_local = NULL;

   NALU_HYPRE_Int     smrk_local = -1;
   NALU_HYPRE_Int       P_max_elmts = 4;
   NALU_HYPRE_Real      trunc_factor = 0.0;
   NALU_HYPRE_Int       debug_flag = 0;
   NALU_HYPRE_Int       measure_type = 0;
   NALU_HYPRE_Real      strong_threshold = 0.25;
   NALU_HYPRE_Real      max_row_sum = 0.9;

   NALU_HYPRE_Int       old_num_levels = nalu_hypre_ParAMGDataNumLevels(FrelaxVcycleData[lev]);
   nalu_hypre_IntArray       **CF_marker_array_local = (FrelaxVcycleData[lev] -> CF_marker_array);
   NALU_HYPRE_Int            *CF_marker_local = NULL;
   nalu_hypre_ParCSRMatrix   **A_array_local = (FrelaxVcycleData[lev] -> A_array);
   nalu_hypre_ParCSRMatrix   **P_array_local = (FrelaxVcycleData[lev] -> P_array);
   nalu_hypre_ParVector      **F_array_local = (FrelaxVcycleData[lev] -> F_array);
   nalu_hypre_ParVector      **U_array_local = (FrelaxVcycleData[lev] -> U_array);
   nalu_hypre_IntArray       **dof_func_array = (FrelaxVcycleData[lev] -> dof_func_array);
   NALU_HYPRE_Int            relax_type = 3;
   NALU_HYPRE_Int            indx, k, tms;
   NALU_HYPRE_Int            num_fine_points = 0;
   NALU_HYPRE_Int            num_functions = nalu_hypre_ParAMGDataNumFunctions(FrelaxVcycleData[lev]);
   NALU_HYPRE_Int            relax_order = nalu_hypre_ParAMGDataRelaxOrder(FrelaxVcycleData[lev]);

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);


   local_size = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));

   /* Free any local data not previously destroyed */
   if (A_array_local || P_array_local || CF_marker_array_local)
   {
      for (j = 1; j < old_num_levels; j++)
      {
         if (A_array_local[j])
         {
            nalu_hypre_ParCSRMatrixDestroy(A_array_local[j]);
            A_array_local[j] = NULL;
         }
      }

      for (j = 0; j < old_num_levels - 1; j++)
      {
         if (P_array_local[j])
         {
            nalu_hypre_ParCSRMatrixDestroy(P_array_local[j]);
            P_array_local[j] = NULL;
         }
      }

      for (j = 0; j < old_num_levels - 1; j++)
      {
         if (CF_marker_array_local[j])
         {
            nalu_hypre_IntArrayDestroy(CF_marker_array_local[j]);
            CF_marker_array_local[j] = NULL;
         }
      }
      nalu_hypre_TFree(A_array_local, NALU_HYPRE_MEMORY_HOST);
      A_array_local = NULL;
      nalu_hypre_TFree(P_array_local, NALU_HYPRE_MEMORY_HOST);
      P_array_local = NULL;
      nalu_hypre_TFree(CF_marker_array_local, NALU_HYPRE_MEMORY_HOST);
      CF_marker_array_local = NULL;
   }
   /* free solution arrays not previously destroyed */
   if (F_array_local != NULL || U_array_local != NULL)
   {
      for (j = 1; j < old_num_levels; j++)
      {
         if (F_array_local[j] != NULL)
         {
            nalu_hypre_ParVectorDestroy(F_array_local[j]);
            F_array_local[j] = NULL;
         }
         if (U_array_local[j] != NULL)
         {
            nalu_hypre_ParVectorDestroy(U_array_local[j]);
            U_array_local[j] = NULL;
         }
      }
      nalu_hypre_TFree(F_array_local, NALU_HYPRE_MEMORY_HOST);
      F_array_local = NULL;
      nalu_hypre_TFree(U_array_local, NALU_HYPRE_MEMORY_HOST);
      U_array_local = NULL;
   }

   /* Initialize some variables and allocate memory */
   not_finished = 1;
   lev_local = 0;
   if (A_array_local == NULL)
   {
      A_array_local = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix*,  max_local_lvls, NALU_HYPRE_MEMORY_HOST);
   }
   if (P_array_local == NULL && max_local_lvls > 1)
   {
      P_array_local = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix*,  max_local_lvls - 1, NALU_HYPRE_MEMORY_HOST);
   }
   if (F_array_local == NULL)
   {
      F_array_local = nalu_hypre_CTAlloc(nalu_hypre_ParVector*,  max_local_lvls, NALU_HYPRE_MEMORY_HOST);
   }
   if (U_array_local == NULL)
   {
      U_array_local = nalu_hypre_CTAlloc(nalu_hypre_ParVector*,  max_local_lvls, NALU_HYPRE_MEMORY_HOST);
   }
   if (CF_marker_array_local == NULL)
   {
      CF_marker_array_local = nalu_hypre_CTAlloc(nalu_hypre_IntArray*,  max_local_lvls, NALU_HYPRE_MEMORY_HOST);
   }
   if (dof_func_array == NULL)
   {
      dof_func_array = nalu_hypre_CTAlloc(nalu_hypre_IntArray*, max_local_lvls, NALU_HYPRE_MEMORY_HOST);
   }

   A_array_local[0] = A;
   F_array_local[0] = f;
   U_array_local[0] = u;

   for (i = 0; i < local_size; i++)
   {
      if (nalu_hypre_IntArrayData(CF_marker_array[lev])[i] == smrk_local)
      {
         num_fine_points++;
      }
   }
   //nalu_hypre_printf("My_ID = %d, Size of A_FF matrix: %d \n", my_id, num_fine_points);

   if (num_functions > 1 && dof_func == NULL)
   {
      dof_func = nalu_hypre_IntArrayCreate(num_fine_points);
      nalu_hypre_IntArrayInitialize(dof_func);
      indx = 0;
      tms = num_fine_points / num_functions;
      if (tms * num_functions + indx > num_fine_points) { tms--; }
      for (j = 0; j < tms; j++)
      {
         for (k = 0; k < num_functions; k++)
         {
            nalu_hypre_IntArrayData(dof_func)[indx++] = k;
         }
      }
      k = 0;
      while (indx < num_fine_points)
      {
         nalu_hypre_IntArrayData(dof_func)[indx++] = k++;
      }
      FrelaxVcycleData[lev] -> dof_func = dof_func;
   }
   dof_func_array[0] = dof_func;
   nalu_hypre_ParAMGDataDofFuncArray(FrelaxVcycleData[lev]) = dof_func_array;

   while (not_finished)
   {
      local_size = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A_array_local[lev_local]));
      dof_func_data = NULL;
      if (dof_func_array[lev_local])
      {
         dof_func_data = nalu_hypre_IntArrayData(dof_func_array[lev_local]);
      }

      if (lev_local == 0)
      {
         /* use the CF_marker from the outer MGR cycle to create the strength connection matrix */
         nalu_hypre_BoomerAMGCreateSFromCFMarker(A_array_local[lev_local], strong_threshold, max_row_sum,
                                            nalu_hypre_IntArrayData(CF_marker_array[lev]),
                                            num_functions, dof_func_data, smrk_local, &S_local);
         //nalu_hypre_ParCSRMatrixPrintIJ(S_local,0,0,"S_mat");
      }
      else if (lev_local > 0)
      {
         nalu_hypre_BoomerAMGCreateS(A_array_local[lev_local], strong_threshold, max_row_sum, num_functions,
                                dof_func_data, &S_local);
      }

      CF_marker_array_local[lev_local] = nalu_hypre_IntArrayCreate(local_size);
      nalu_hypre_IntArrayInitialize(CF_marker_array_local[lev_local]);
      CF_marker_local = nalu_hypre_IntArrayData(CF_marker_array_local[lev_local]);

      NALU_HYPRE_Int coarsen_cut_factor = 0;
      nalu_hypre_BoomerAMGCoarsenHMIS(S_local, A_array_local[lev_local], measure_type, coarsen_cut_factor,
                                 debug_flag, &(CF_marker_array_local[lev_local]));


      nalu_hypre_BoomerAMGCoarseParms(comm, local_size,
                                 num_functions, dof_func_array[lev_local], CF_marker_array_local[lev_local],
                                 &coarse_dof_func_lvl, coarse_pnts_global_lvl);

      if (my_id == (num_procs - 1))
      {
         coarse_size = coarse_pnts_global_lvl[1];
      }
      nalu_hypre_MPI_Bcast(&coarse_size, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
      //nalu_hypre_printf("Coarse size = %d \n", coarse_size);
      if (coarse_size == 0) // stop coarsening
      {
         if (S_local) { nalu_hypre_ParCSRMatrixDestroy(S_local); }
         nalu_hypre_IntArrayDestroy(coarse_dof_func_lvl);

         if (lev_local == 0)
         {
            // Save the cf_marker from outer MGR level (lev).
            if (relax_order == 1)
            {
               /* We need to mask out C-points from outer CF-marker for C/F relaxation at solve phase --DOK*/
               for (i = 0; i < local_size; i++)
               {
                  if (nalu_hypre_IntArrayData(CF_marker_array[lev])[i] == 1)
                  {
                     CF_marker_local[i] = 0;
                  }
               }
            }
            else
            {
               /* Do lexicographic relaxation on F-points from outer CF-marker --DOK*/
               for (i = 0; i < local_size; i++)
               {
                  CF_marker_local[i] = nalu_hypre_IntArrayData(CF_marker_array[lev])[i];
               }
            }
         }
         else
         {
            nalu_hypre_IntArrayDestroy(CF_marker_array_local[lev_local]);
            CF_marker_array_local[lev_local] = NULL;
         }
         break;
      }

      nalu_hypre_BoomerAMGBuildExtPIInterpHost(A_array_local[lev_local], CF_marker_local,
                                          S_local, coarse_pnts_global_lvl, num_functions, dof_func_data,
                                          debug_flag, trunc_factor, P_max_elmts, &P_local);

      //    nalu_hypre_BoomerAMGBuildInterp(A_array_local[lev_local], CF_marker_local,
      //                                   S_local, coarse_pnts_global_lvl, 1, NULL,
      //                                   0, 0.0, 0, NULL, &P_local);

      /* Save the CF_marker pointer. For lev_local = 0, save the cf_marker from outer MGR level (lev).
       * This is necessary to enable relaxations over the A_FF matrix during the solve phase. -- DOK
       */
      if (lev_local == 0)
      {
         if (relax_order == 1)
         {
            /* We need to mask out C-points from outer CF-marker for C/F relaxation at solve phase --DOK*/
            for (i = 0; i < local_size; i++)
            {
               if (nalu_hypre_IntArrayData(CF_marker_array[lev])[i] == 1)
               {
                  CF_marker_local[i] = 0;
               }
            }
         }
         else
         {
            /* Do lexicographic relaxation on F-points from outer CF-marker --DOK */
            for (i = 0; i < local_size; i++)
            {
               CF_marker_local[i] = nalu_hypre_IntArrayData(CF_marker_array[lev])[i];
            }
         }
      }
      /* Save interpolation matrix pointer */
      P_array_local[lev_local] = P_local;

      if (num_functions > 1)
      {
         dof_func_array[lev_local + 1] = coarse_dof_func_lvl;
      }

      /* build the coarse grid */
      nalu_hypre_BoomerAMGBuildCoarseOperatorKT(P_local, A_array_local[lev_local],
                                           P_local, 0, &RAP_local);
      /*
          if (my_id == (num_procs -1)) coarse_size = coarse_pnts_global_lvl[1];
          nalu_hypre_MPI_Bcast(&coarse_size, 1, NALU_HYPRE_MPI_BIG_INT, num_procs-1, comm);
      */
      lev_local++;

      if (S_local) { nalu_hypre_ParCSRMatrixDestroy(S_local); }
      S_local = NULL;
      if ( (lev_local == max_local_lvls - 1) || (coarse_size <= max_local_coarse_size) )
      {
         not_finished = 0;
      }

      A_array_local[lev_local] = RAP_local;
      F_array_local[lev_local] = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(RAP_local),
                                                       nalu_hypre_ParCSRMatrixGlobalNumRows(RAP_local),
                                                       nalu_hypre_ParCSRMatrixRowStarts(RAP_local));
      nalu_hypre_ParVectorInitialize(F_array_local[lev_local]);

      U_array_local[lev_local] = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(RAP_local),
                                                       nalu_hypre_ParCSRMatrixGlobalNumRows(RAP_local),
                                                       nalu_hypre_ParCSRMatrixRowStarts(RAP_local));
      nalu_hypre_ParVectorInitialize(U_array_local[lev_local]);
   } // end while loop

   // setup Vcycle data
   (FrelaxVcycleData[lev] -> A_array) = A_array_local;
   (FrelaxVcycleData[lev] -> P_array) = P_array_local;
   (FrelaxVcycleData[lev] -> F_array) = F_array_local;
   (FrelaxVcycleData[lev] -> U_array) = U_array_local;
   (FrelaxVcycleData[lev] -> CF_marker_array) = CF_marker_array_local;
   (FrelaxVcycleData[lev] -> num_levels) = lev_local;
   //if(lev == 1)
   //{
   //  for (i = 0; i < local_size; i++)
   //  {
   //    if(CF_marker_array_local[0][i] == 1)
   //    nalu_hypre_printf("cfmarker[%d] = %d\n",i, CF_marker_array_local[0][i]);
   //  }
   //}
   /* setup GE for coarsest level (if small enough) */
   if ((lev_local > 0) && (nalu_hypre_ParAMGDataUserCoarseRelaxType(FrelaxVcycleData[lev]) == 9))
   {
      if ((coarse_size <= max_local_coarse_size) && coarse_size > 0)
      {
         nalu_hypre_GaussElimSetup(FrelaxVcycleData[lev], lev_local, 9);
      }
      else
      {
         /* use relaxation */
         nalu_hypre_ParAMGDataUserCoarseRelaxType(FrelaxVcycleData[lev]) = relax_type;
      }
   }

   return nalu_hypre_error_flag;
}
