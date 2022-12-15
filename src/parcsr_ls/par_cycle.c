/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ParAMG cycling routine
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "par_amg.h"
#include "../parcsr_block_mv/par_csr_block_matrix.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGCycle
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGCycle( void              *amg_vdata,
                      nalu_hypre_ParVector  **F_array,
                      nalu_hypre_ParVector  **U_array   )
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) amg_vdata;

   NALU_HYPRE_Solver *smoother;

   /* Data Structure variables */
   nalu_hypre_ParCSRMatrix      **A_array;
   nalu_hypre_ParCSRMatrix      **P_array;
   nalu_hypre_ParCSRMatrix      **R_array;
   nalu_hypre_ParVector          *Utemp;
   nalu_hypre_ParVector          *Vtemp;
   nalu_hypre_ParVector          *Rtemp;
   nalu_hypre_ParVector          *Ptemp;
   nalu_hypre_ParVector          *Ztemp;
   nalu_hypre_ParVector          *Aux_U;
   nalu_hypre_ParVector          *Aux_F;
   nalu_hypre_ParCSRBlockMatrix **A_block_array;
   nalu_hypre_ParCSRBlockMatrix **P_block_array;
   nalu_hypre_ParCSRBlockMatrix **R_block_array;

   NALU_HYPRE_Real      *Ztemp_data;
   NALU_HYPRE_Real      *Ptemp_data;
   nalu_hypre_IntArray **CF_marker_array;
   NALU_HYPRE_Int       *CF_marker;
   /*
   NALU_HYPRE_Int     **unknown_map_array;
   NALU_HYPRE_Int     **point_map_array;
   NALU_HYPRE_Int     **v_at_point_array;
   */
   NALU_HYPRE_Real      cycle_op_count;
   NALU_HYPRE_Int       cycle_type;
   NALU_HYPRE_Int       fcycle, fcycle_lev;
   NALU_HYPRE_Int       num_levels;
   NALU_HYPRE_Int       max_levels;
   NALU_HYPRE_Real     *num_coeffs;
   NALU_HYPRE_Int      *num_grid_sweeps;
   NALU_HYPRE_Int      *grid_relax_type;
   NALU_HYPRE_Int     **grid_relax_points;
   NALU_HYPRE_Int       block_mode;
   NALU_HYPRE_Int       cheby_order;

   /* Local variables  */
   NALU_HYPRE_Int      *lev_counter;
   NALU_HYPRE_Int       Solve_err_flag;
   NALU_HYPRE_Int       k;
   NALU_HYPRE_Int       i, j, jj;
   NALU_HYPRE_Int       level;
   NALU_HYPRE_Int       cycle_param;
   NALU_HYPRE_Int       coarse_grid;
   NALU_HYPRE_Int       fine_grid;
   NALU_HYPRE_Int       Not_Finished;
   NALU_HYPRE_Int       num_sweep;
   NALU_HYPRE_Int       cg_num_sweep = 1;
   NALU_HYPRE_Int       relax_type;
   NALU_HYPRE_Int       relax_points;
   NALU_HYPRE_Int       relax_order;
   NALU_HYPRE_Int       relax_local;
   NALU_HYPRE_Int       old_version = 0;
   NALU_HYPRE_Real     *relax_weight;
   NALU_HYPRE_Real     *omega;
   NALU_HYPRE_Real      alfa, beta, gammaold;
   NALU_HYPRE_Real      gamma = 1.0;
   NALU_HYPRE_Int       local_size;
   /*   NALU_HYPRE_Int      *smooth_option; */
   NALU_HYPRE_Int       smooth_type;
   NALU_HYPRE_Int       smooth_num_levels;
   NALU_HYPRE_Int       my_id;
   NALU_HYPRE_Int       restri_type;
   NALU_HYPRE_Real      alpha;
   nalu_hypre_Vector  **l1_norms = NULL;
   nalu_hypre_Vector   *l1_norms_level;
   nalu_hypre_Vector  **ds = nalu_hypre_ParAMGDataChebyDS(amg_data);
   NALU_HYPRE_Real    **coefs = nalu_hypre_ParAMGDataChebyCoefs(amg_data);
   NALU_HYPRE_Int       seq_cg = 0;
   NALU_HYPRE_Int       partial_cycle_coarsest_level;
   NALU_HYPRE_Int       partial_cycle_control;
   MPI_Comm        comm;

#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
   char            nvtx_name[1024];
#endif

#if 0
   NALU_HYPRE_Real   *D_mat;
   NALU_HYPRE_Real   *S_vec;
#endif

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
   nalu_hypre_GpuProfilingPushRange("AMGCycle");
#endif

   /* Acquire data and allocate storage */
   A_array           = nalu_hypre_ParAMGDataAArray(amg_data);
   P_array           = nalu_hypre_ParAMGDataPArray(amg_data);
   R_array           = nalu_hypre_ParAMGDataRArray(amg_data);
   CF_marker_array   = nalu_hypre_ParAMGDataCFMarkerArray(amg_data);
   Vtemp             = nalu_hypre_ParAMGDataVtemp(amg_data);
   Rtemp             = nalu_hypre_ParAMGDataRtemp(amg_data);
   Ptemp             = nalu_hypre_ParAMGDataPtemp(amg_data);
   Ztemp             = nalu_hypre_ParAMGDataZtemp(amg_data);
   num_levels        = nalu_hypre_ParAMGDataNumLevels(amg_data);
   max_levels        = nalu_hypre_ParAMGDataMaxLevels(amg_data);
   cycle_type        = nalu_hypre_ParAMGDataCycleType(amg_data);
   fcycle            = nalu_hypre_ParAMGDataFCycle(amg_data);

   A_block_array     = nalu_hypre_ParAMGDataABlockArray(amg_data);
   P_block_array     = nalu_hypre_ParAMGDataPBlockArray(amg_data);
   R_block_array     = nalu_hypre_ParAMGDataRBlockArray(amg_data);
   block_mode        = nalu_hypre_ParAMGDataBlockMode(amg_data);

   num_grid_sweeps     = nalu_hypre_ParAMGDataNumGridSweeps(amg_data);
   grid_relax_type     = nalu_hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points   = nalu_hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_order         = nalu_hypre_ParAMGDataRelaxOrder(amg_data);
   relax_weight        = nalu_hypre_ParAMGDataRelaxWeight(amg_data);
   omega               = nalu_hypre_ParAMGDataOmega(amg_data);
   smooth_type         = nalu_hypre_ParAMGDataSmoothType(amg_data);
   smooth_num_levels   = nalu_hypre_ParAMGDataSmoothNumLevels(amg_data);
   l1_norms            = nalu_hypre_ParAMGDataL1Norms(amg_data);
   /* smooth_option       = nalu_hypre_ParAMGDataSmoothOption(amg_data); */
   /* RL */
   restri_type = nalu_hypre_ParAMGDataRestriction(amg_data);

   partial_cycle_coarsest_level = nalu_hypre_ParAMGDataPartialCycleCoarsestLevel(amg_data);
   partial_cycle_control = nalu_hypre_ParAMGDataPartialCycleControl(amg_data);

   /*max_eig_est = nalu_hypre_ParAMGDataMaxEigEst(amg_data);
   min_eig_est = nalu_hypre_ParAMGDataMinEigEst(amg_data);
   cheby_fraction = nalu_hypre_ParAMGDataChebyFraction(amg_data);*/
   cheby_order = nalu_hypre_ParAMGDataChebyOrder(amg_data);

   cycle_op_count = nalu_hypre_ParAMGDataCycleOpCount(amg_data);

   lev_counter = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_levels, NALU_HYPRE_MEMORY_HOST);

   if (nalu_hypre_ParAMGDataParticipate(amg_data))
   {
      seq_cg = 1;
   }

   /* Initialize */
   Solve_err_flag = 0;

   if (grid_relax_points)
   {
      old_version = 1;
   }

   num_coeffs = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_levels, NALU_HYPRE_MEMORY_HOST);
   num_coeffs[0]    = nalu_hypre_ParCSRMatrixDNumNonzeros(A_array[0]);
   comm = nalu_hypre_ParCSRMatrixComm(A_array[0]);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (block_mode)
   {
      for (j = 1; j < num_levels; j++)
      {
         num_coeffs[j] = nalu_hypre_ParCSRBlockMatrixNumNonzeros(A_block_array[j]);
      }
   }
   else
   {
      for (j = 1; j < num_levels; j++)
      {
         num_coeffs[j] = nalu_hypre_ParCSRMatrixDNumNonzeros(A_array[j]);
      }
   }

   /*---------------------------------------------------------------------
    *    Initialize cycling control counter
    *
    *     Cycling is controlled using a level counter: lev_counter[k]
    *
    *     Each time relaxation is performed on level k, the
    *     counter is decremented by 1. If the counter is then
    *     negative, we go to the next finer level. If non-
    *     negative, we go to the next coarser level. The
    *     following actions control cycling:
    *
    *     a. lev_counter[0] is initialized to 1.
    *     b. lev_counter[k] is initialized to cycle_type for k>0.
    *
    *     c. During cycling, when going down to level k, lev_counter[k]
    *        is set to the max of (lev_counter[k],cycle_type)
    *---------------------------------------------------------------------*/

   Not_Finished = 1;

   lev_counter[0] = 1;
   for (k = 1; k < num_levels; ++k)
   {
      if (fcycle)
      {
         lev_counter[k] = 1;
      }
      else
      {
         lev_counter[k] = cycle_type;
      }
   }
   fcycle_lev = num_levels - 2;

   level = 0;
   cycle_param = 1;

   smoother = nalu_hypre_ParAMGDataSmoother(amg_data);

   if (smooth_num_levels > 0)
   {
      if (smooth_type == 7 || smooth_type == 8
          || smooth_type == 17 || smooth_type == 18
          || smooth_type == 9 || smooth_type == 19)
      {
         NALU_HYPRE_Int actual_local_size = nalu_hypre_ParVectorActualLocalSize(Vtemp);
         Utemp = nalu_hypre_ParVectorCreate(comm, nalu_hypre_ParVectorGlobalSize(Vtemp),
                                       nalu_hypre_ParVectorPartitioning(Vtemp));
         local_size
            = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(Vtemp));
         if (local_size < actual_local_size)
         {
            nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Utemp)) =
               nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  actual_local_size, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_ParVectorActualLocalSize(Utemp) = actual_local_size;
         }
         else
         {
            nalu_hypre_ParVectorInitialize(Utemp);
         }
      }
   }

   /* Override level control and cycle param in the case of a partial cycle */
   if (partial_cycle_coarsest_level >= 0)
   {
      if (partial_cycle_control == 0)
      {
         level = 0;
         cycle_param = 1;
      }
      else
      {
         level = partial_cycle_coarsest_level;
         if (level == num_levels - 1)
         {
            cycle_param = 3;
         }
         else
         {
            cycle_param = 2;
         }
         for (k = 0; k < num_levels; ++k)
         {
            lev_counter[k] = 0;
         }
      }
   }

   /*---------------------------------------------------------------------
    * Main loop of cycling
    *--------------------------------------------------------------------*/

   NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
   nalu_hypre_sprintf(nvtx_name, "%s-%d", "AMG Level", level);
   nalu_hypre_GpuProfilingPushRange(nvtx_name);
#endif
   while (Not_Finished)
   {
      if (num_levels > 1)
      {
         local_size = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(F_array[level]));
         nalu_hypre_ParVectorSetLocalSize(Vtemp, local_size);

         if (smooth_num_levels <= level)
         {
            cg_num_sweep = 1;
            num_sweep = num_grid_sweeps[cycle_param];
            Aux_U = U_array[level];
            Aux_F = F_array[level];
         }
         else if (smooth_type > 9)
         {
            nalu_hypre_ParVectorSetLocalSize(Ztemp, local_size);
            nalu_hypre_ParVectorSetLocalSize(Rtemp, local_size);
            nalu_hypre_ParVectorSetLocalSize(Ptemp, local_size);

            Ztemp_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Ztemp));
            Ptemp_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Ptemp));
            nalu_hypre_ParVectorSetConstantValues(Ztemp, 0.0);
            alpha = -1.0;
            beta = 1.0;

            nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[level],
                                               U_array[level], beta, F_array[level], Rtemp);

            cg_num_sweep = nalu_hypre_ParAMGDataSmoothNumSweeps(amg_data);
            num_sweep = num_grid_sweeps[cycle_param];
            Aux_U = Ztemp;
            Aux_F = Rtemp;
         }
         else
         {
            cg_num_sweep = 1;
            num_sweep = nalu_hypre_ParAMGDataSmoothNumSweeps(amg_data);
            Aux_U = U_array[level];
            Aux_F = F_array[level];
         }
         relax_type = grid_relax_type[cycle_param];
      }
      else /* AB: 4/08: removed the max_levels > 1 check - should do this when max-levels = 1 also */
      {
         /* If no coarsening occurred, apply a simple smoother once */
         Aux_U = U_array[level];
         Aux_F = F_array[level];
         num_sweep = num_grid_sweeps[0];
         /* TK: Use the user relax type (instead of 0) to allow for setting a
           convergent smoother (e.g. in the solution of singular problems). */
         relax_type = nalu_hypre_ParAMGDataUserRelaxType(amg_data);
         if (relax_type == -1)
         {
            relax_type = 6;
         }
      }

      if (CF_marker_array[level] != NULL)
      {
         CF_marker = nalu_hypre_IntArrayData(CF_marker_array[level]);
      }
      else
      {
         CF_marker = NULL;
      }

      if (l1_norms != NULL)
      {
         l1_norms_level = l1_norms[level];
      }
      else
      {
         l1_norms_level = NULL;
      }

      if (cycle_param == 3 && seq_cg)
      {
         NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "Coarse solve");
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
         nalu_hypre_GpuProfilingPushRange("Coarse solve");
#endif
         nalu_hypre_seqAMGCycle(amg_data, level, F_array, U_array);
         NALU_HYPRE_ANNOTATE_REGION_END("%s", "Coarse solve");
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
         nalu_hypre_GpuProfilingPopRange();
#endif
      }
#ifdef NALU_HYPRE_USING_DSUPERLU
      else if (cycle_param == 3 && nalu_hypre_ParAMGDataDSLUSolver(amg_data) != NULL)
      {
         NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "Coarse solve");
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
         nalu_hypre_GpuProfilingPushRange("Coarse solve");
#endif
         nalu_hypre_SLUDistSolve(nalu_hypre_ParAMGDataDSLUSolver(amg_data), Aux_F, Aux_U);
         NALU_HYPRE_ANNOTATE_REGION_END("%s", "Coarse solve");
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
         nalu_hypre_GpuProfilingPopRange();
#endif
      }
#endif
      else
      {
         /*------------------------------------------------------------------
         * Do the relaxation num_sweep times
         *-----------------------------------------------------------------*/
         NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "Relaxation");
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
         nalu_hypre_GpuProfilingPushRange("Relaxation");
#endif

         for (jj = 0; jj < cg_num_sweep; jj++)
         {
            if (smooth_num_levels > level && smooth_type > 9)
            {
               nalu_hypre_ParVectorSetConstantValues(Aux_U, 0.0);
            }

            for (j = 0; j < num_sweep; j++)
            {
               if (num_levels == 1 && max_levels > 1)
               {
                  relax_points = 0;
                  relax_local = 0;
               }
               else
               {
                  if (old_version)
                  {
                     relax_points = grid_relax_points[cycle_param][j];
                  }
                  relax_local = relax_order;
               }

               /*-----------------------------------------------
                * VERY sloppy approximation to cycle complexity
                *-----------------------------------------------*/
               if (old_version && level < num_levels - 1)
               {
                  switch (relax_points)
                  {
                     case 1:
                        cycle_op_count += num_coeffs[level + 1];
                        break;

                     case -1:
                        cycle_op_count += (num_coeffs[level] - num_coeffs[level + 1]);
                        break;
                  }
               }
               else
               {
                  cycle_op_count += num_coeffs[level];
               }

               /*-----------------------------------------------
                  Choose Smoother
                -----------------------------------------------*/
               if ( smooth_num_levels > level &&
                    (smooth_type == 7 || smooth_type == 17 ||
                     smooth_type == 8 || smooth_type == 18 ||
                     smooth_type == 9 || smooth_type == 19) )
               {
                  nalu_hypre_ParVectorSetLocalSize(Utemp, local_size);

                  alpha = -1.0;
                  beta = 1.0;
                  nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[level],
                                                     U_array[level], beta, Aux_F, Vtemp);
                  if (smooth_type == 8 || smooth_type == 18)
                  {
                     NALU_HYPRE_ParCSRParaSailsSolve(smoother[level],
                                                (NALU_HYPRE_ParCSRMatrix) A_array[level],
                                                (NALU_HYPRE_ParVector) Vtemp,
                                                (NALU_HYPRE_ParVector) Utemp);
                  }
                  else if (smooth_type == 7 || smooth_type == 17)
                  {
                     NALU_HYPRE_ParCSRPilutSolve(smoother[level],
                                            (NALU_HYPRE_ParCSRMatrix) A_array[level],
                                            (NALU_HYPRE_ParVector) Vtemp,
                                            (NALU_HYPRE_ParVector) Utemp);
                  }
                  else if (smooth_type == 9 || smooth_type == 19)
                  {
                     NALU_HYPRE_EuclidSolve(smoother[level],
                                       (NALU_HYPRE_ParCSRMatrix) A_array[level],
                                       (NALU_HYPRE_ParVector) Vtemp,
                                       (NALU_HYPRE_ParVector) Utemp);
                  }
                  nalu_hypre_ParVectorAxpy(relax_weight[level], Utemp, Aux_U);
               }
               else if ( smooth_num_levels > level && (smooth_type == 4) )
               {
                  NALU_HYPRE_FSAISetZeroGuess(smoother[level], cycle_param - 2);
                  NALU_HYPRE_FSAISetMaxIterations(smoother[level], num_grid_sweeps[cycle_param]);
                  NALU_HYPRE_FSAISolve(smoother[level],
                                  (NALU_HYPRE_ParCSRMatrix) A_array[level],
                                  (NALU_HYPRE_ParVector) Aux_F,
                                  (NALU_HYPRE_ParVector) Aux_U);
               }
               else if ( smooth_num_levels > level && (smooth_type == 5 || smooth_type == 15) )
               {
                  NALU_HYPRE_ILUSolve(smoother[level],
                                 (NALU_HYPRE_ParCSRMatrix) A_array[level],
                                 (NALU_HYPRE_ParVector) Aux_F,
                                 (NALU_HYPRE_ParVector) Aux_U);
               }
               else if ( smooth_num_levels > level && (smooth_type == 6 || smooth_type == 16) )
               {
                  NALU_HYPRE_SchwarzSolve(smoother[level],
                                     (NALU_HYPRE_ParCSRMatrix) A_array[level],
                                     (NALU_HYPRE_ParVector) Aux_F,
                                     (NALU_HYPRE_ParVector) Aux_U);
               }
               else if (relax_type == 9 || relax_type == 99 || relax_type == 199)
               {
                  /* Gaussian elimination */
                  nalu_hypre_GaussElimSolve(amg_data, level, relax_type);
               }
               else if (relax_type == 18)
               {
                  /* L1 - Jacobi*/
                  Solve_err_flag = nalu_hypre_BoomerAMGRelaxIF(A_array[level],
                                                          Aux_F,
                                                          CF_marker,
                                                          relax_type,
                                                          relax_order,
                                                          cycle_param,
                                                          relax_weight[level],
                                                          omega[level],
                                                          l1_norms_level ? nalu_hypre_VectorData(l1_norms_level) : NULL,
                                                          Aux_U,
                                                          Vtemp,
                                                          Ztemp);
               }
               else if (relax_type == 15)
               {
                  /* CG */
                  if (j == 0) /* do num sweep iterations of CG */
                  {
                     nalu_hypre_ParCSRRelax_CG( smoother[level],
                                           A_array[level],
                                           Aux_F,
                                           Aux_U,
                                           num_sweep);
                  }
               }
               else if (relax_type == 16)
               {
                  /* scaled Chebyshev */
                  NALU_HYPRE_Int scale = nalu_hypre_ParAMGDataChebyScale(amg_data);
                  NALU_HYPRE_Int variant = nalu_hypre_ParAMGDataChebyVariant(amg_data);
                  nalu_hypre_ParCSRRelax_Cheby_Solve(A_array[level], Aux_F,
                                                nalu_hypre_VectorData(ds[level]), coefs[level],
                                                cheby_order, scale,
                                                variant, Aux_U, Vtemp, Ztemp, Ptemp, Rtemp );
               }
               else if (relax_type == 17)
               {
                  if (level == num_levels - 1)
                  {
                     /* if we are on the coarsest level, the cf_marker will be null
                        and we just do one sweep regular Jacobi */
                     nalu_hypre_assert(cycle_param == 3);
                     nalu_hypre_BoomerAMGRelax(A_array[level], Aux_F, CF_marker, 0, 0, relax_weight[level],
                                          0.0, NULL, Aux_U, Vtemp, NULL);
                  }
                  else
                  {
                     nalu_hypre_BoomerAMGRelax_FCFJacobi(A_array[level], Aux_F, CF_marker, relax_weight[level],
                                                    Aux_U, Vtemp);
                  }
               }
               else if (old_version)
               {
                  Solve_err_flag = nalu_hypre_BoomerAMGRelax(A_array[level],
                                                        Aux_F,
                                                        CF_marker,
                                                        relax_type,
                                                        relax_points,
                                                        relax_weight[level],
                                                        omega[level],
                                                        l1_norms_level ? nalu_hypre_VectorData(l1_norms_level) : NULL,
                                                        Aux_U,
                                                        Vtemp,
                                                        Ztemp);
               }
               else
               {
                  /* smoother than can have CF ordering */
                  if (block_mode)
                  {
                     Solve_err_flag = nalu_hypre_BoomerAMGBlockRelaxIF(A_block_array[level],
                                                                  Aux_F,
                                                                  CF_marker,
                                                                  relax_type,
                                                                  relax_local,
                                                                  cycle_param,
                                                                  relax_weight[level],
                                                                  omega[level],
                                                                  Aux_U,
                                                                  Vtemp);
                  }
                  else
                  {
                     Solve_err_flag = nalu_hypre_BoomerAMGRelaxIF(A_array[level],
                                                             Aux_F,
                                                             CF_marker,
                                                             relax_type,
                                                             relax_local,
                                                             cycle_param,
                                                             relax_weight[level],
                                                             omega[level],
                                                             l1_norms_level ? nalu_hypre_VectorData(l1_norms_level) : NULL,
                                                             Aux_U,
                                                             Vtemp,
                                                             Ztemp);
                  }
               }

               if (Solve_err_flag != 0)
               {
                  NALU_HYPRE_ANNOTATE_REGION_END("%s", "Relaxation");
                  NALU_HYPRE_ANNOTATE_MGLEVEL_END(level);
                  NALU_HYPRE_ANNOTATE_FUNC_END;
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
                  nalu_hypre_GpuProfilingPopRange();
                  nalu_hypre_GpuProfilingPopRange();
                  nalu_hypre_GpuProfilingPopRange();
#endif
                  return (Solve_err_flag);
               }
            } /* for (j = 0; j < num_sweep; j++) */

            if  (smooth_num_levels > level && smooth_type > 9)
            {
               gammaold = gamma;
               gamma = nalu_hypre_ParVectorInnerProd(Rtemp, Ztemp);
               if (jj == 0)
               {
                  nalu_hypre_ParVectorCopy(Ztemp, Ptemp);
               }
               else
               {
                  beta = gamma / gammaold;
                  for (i = 0; i < local_size; i++)
                  {
                     Ptemp_data[i] = Ztemp_data[i] + beta * Ptemp_data[i];
                  }
               }

               nalu_hypre_ParCSRMatrixMatvec(1.0, A_array[level], Ptemp, 0.0, Vtemp);
               alfa = gamma / nalu_hypre_ParVectorInnerProd(Ptemp, Vtemp);
               nalu_hypre_ParVectorAxpy(alfa, Ptemp, U_array[level]);
               nalu_hypre_ParVectorAxpy(-alfa, Vtemp, Rtemp);
            }
         } /* for (jj = 0; jj < cg_num_sweep; jj++) */

         NALU_HYPRE_ANNOTATE_REGION_END("%s", "Relaxation");
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
         nalu_hypre_GpuProfilingPopRange();
#endif
      }

      /*------------------------------------------------------------------
       * Decrement the control counter and determine which grid to visit next
       *-----------------------------------------------------------------*/

      --lev_counter[level];

      //if ( level != num_levels-1 && lev_counter[level] >= 0 )
      if (lev_counter[level] >= 0 && level != num_levels - 1)
      {
         /*---------------------------------------------------------------
          * Visit coarser level next.
          * Compute residual using nalu_hypre_ParCSRMatrixMatvec.
          * Perform restriction using nalu_hypre_ParCSRMatrixMatvecT.
          * Reset counters and cycling parameters for coarse level
          *--------------------------------------------------------------*/

         fine_grid = level;
         coarse_grid = level + 1;

         nalu_hypre_ParVectorSetZeros(U_array[coarse_grid]);

         alpha = -1.0;
         beta = 1.0;

         NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "Residual");
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
         nalu_hypre_GpuProfilingPushRange("Residual");
#endif
         if (block_mode)
         {
            nalu_hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
            nalu_hypre_ParCSRBlockMatrixMatvec(alpha, A_block_array[fine_grid], U_array[fine_grid],
                                          beta, Vtemp);
         }
         else
         {
            // JSP: avoid unnecessary copy using out-of-place version of SpMV
            nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[fine_grid], U_array[fine_grid],
                                               beta, F_array[fine_grid], Vtemp);
         }
         NALU_HYPRE_ANNOTATE_REGION_END("%s", "Residual");
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
         nalu_hypre_GpuProfilingPopRange();
#endif

         alpha = 1.0;
         beta = 0.0;

         NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "Restriction");
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
         nalu_hypre_GpuProfilingPushRange("Restriction");
#endif
         if (block_mode)
         {
            nalu_hypre_ParCSRBlockMatrixMatvecT(alpha, R_block_array[fine_grid], Vtemp,
                                           beta, F_array[coarse_grid]);
         }
         else
         {
            if (restri_type)
            {
               /* RL: no transpose for R */
               nalu_hypre_ParCSRMatrixMatvec(alpha, R_array[fine_grid], Vtemp,
                                        beta, F_array[coarse_grid]);
            }
            else
            {
               nalu_hypre_ParCSRMatrixMatvecT(alpha, R_array[fine_grid], Vtemp,
                                         beta, F_array[coarse_grid]);
            }
         }
         NALU_HYPRE_ANNOTATE_REGION_END("%s", "Restriction");
         NALU_HYPRE_ANNOTATE_MGLEVEL_END(level);
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
         nalu_hypre_GpuProfilingPopRange();
         nalu_hypre_GpuProfilingPopRange();
#endif

         ++level;
         lev_counter[level] = nalu_hypre_max(lev_counter[level], cycle_type);
         cycle_param = 1;
         if (level == num_levels - 1)
         {
            cycle_param = 3;
         }
         if (partial_cycle_coarsest_level >= 0 && level == partial_cycle_coarsest_level + 1)
         {
            Not_Finished = 0;
         }
         NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
         nalu_hypre_sprintf(nvtx_name, "%s-%d", "AMG Level", level);
         nalu_hypre_GpuProfilingPushRange(nvtx_name);
#endif
      }
      else if (level != 0)
      {
         /*---------------------------------------------------------------
          * Visit finer level next.
          * Interpolate and add correction using nalu_hypre_ParCSRMatrixMatvec.
          * Reset counters and cycling parameters for finer level.
          *--------------------------------------------------------------*/
         fine_grid = level - 1;
         coarse_grid = level;
         alpha = 1.0;
         beta = 1.0;

         NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "Interpolation");
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
         nalu_hypre_GpuProfilingPushRange("Interpolation");
#endif
         if (block_mode)
         {
            nalu_hypre_ParCSRBlockMatrixMatvec(alpha, P_block_array[fine_grid],
                                          U_array[coarse_grid],
                                          beta, U_array[fine_grid]);
         }
         else
         {
            /* printf("Proc %d: level %d, n %d, Interpolation\n", my_id, level, local_size); */
            nalu_hypre_ParCSRMatrixMatvec(alpha, P_array[fine_grid],
                                     U_array[coarse_grid],
                                     beta, U_array[fine_grid]);
            /* printf("Proc %d: level %d, n %d, Interpolation done\n", my_id, level, local_size); */
         }

         nalu_hypre_ParVectorAllZeros(U_array[fine_grid]) = 0;

         NALU_HYPRE_ANNOTATE_REGION_END("%s", "Interpolation");
         NALU_HYPRE_ANNOTATE_MGLEVEL_END(level);
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
         nalu_hypre_GpuProfilingPopRange();
         nalu_hypre_GpuProfilingPopRange();
#endif

         --level;
         cycle_param = 2;
         if (fcycle && fcycle_lev == level)
         {
            lev_counter[level] = nalu_hypre_max(lev_counter[level], 1);
            fcycle_lev --;
         }

         NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
         nalu_hypre_sprintf(nvtx_name, "%s-%d", "AMG Level", level);
         nalu_hypre_GpuProfilingPushRange(nvtx_name);
#endif
      }
      else
      {
         Not_Finished = 0;
      }
   } /* main loop: while (Not_Finished) */

   NALU_HYPRE_ANNOTATE_MGLEVEL_END(level);
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
   nalu_hypre_GpuProfilingPopRange();
#endif

   nalu_hypre_ParAMGDataCycleOpCount(amg_data) = cycle_op_count;

   nalu_hypre_TFree(lev_counter, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(num_coeffs, NALU_HYPRE_MEMORY_HOST);

   if (smooth_num_levels > 0)
   {
      if (smooth_type ==  7 || smooth_type ==  8 || smooth_type ==  9 ||
          smooth_type == 17 || smooth_type == 18 || smooth_type == 19 )
      {
         nalu_hypre_ParVectorDestroy(Utemp);
      }
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;
#if defined (NALU_HYPRE_USING_NVTX) || defined (NALU_HYPRE_USING_ROCTX)
   nalu_hypre_GpuProfilingPopRange();
#endif

   return (Solve_err_flag);
}
