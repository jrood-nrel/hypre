/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * MGR solve routine
 *
 *****************************************************************************/
#include "_nalu_hypre_parcsr_ls.h"
#include "par_mgr.h"
#include "par_amg.h"

/*--------------------------------------------------------------------
 * nalu_hypre_MGRSolve
 *--------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_MGRSolve( void               *mgr_vdata,
                nalu_hypre_ParCSRMatrix *A,
                nalu_hypre_ParVector    *f,
                nalu_hypre_ParVector    *u )
{

   MPI_Comm              comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;

   nalu_hypre_ParCSRMatrix  **A_array = (mgr_data -> A_array);
   nalu_hypre_ParVector    **F_array = (mgr_data -> F_array);
   nalu_hypre_ParVector    **U_array = (mgr_data -> U_array);

   NALU_HYPRE_Real           tol = (mgr_data -> tol);
   NALU_HYPRE_Int            logging = (mgr_data -> logging);
   NALU_HYPRE_Int            print_level = (mgr_data -> print_level);
   NALU_HYPRE_Int            max_iter = (mgr_data -> max_iter);
   NALU_HYPRE_Real           *norms = (mgr_data -> rel_res_norms);
   nalu_hypre_ParVector      *Vtemp = (mgr_data -> Vtemp);
   //   nalu_hypre_ParVector      *Utemp = (mgr_data -> Utemp);
   nalu_hypre_ParVector      *residual;

   NALU_HYPRE_Real           alpha = -1.0;
   NALU_HYPRE_Real           beta = 1.0;
   NALU_HYPRE_Real           conv_factor = 0.0;
   NALU_HYPRE_Real           resnorm = 1.0;
   NALU_HYPRE_Real           init_resnorm = 0.0;
   NALU_HYPRE_Real           rel_resnorm;
   NALU_HYPRE_Real           rhs_norm = 0.0;
   NALU_HYPRE_Real           old_resnorm;
   NALU_HYPRE_Real           ieee_check = 0.;

   NALU_HYPRE_Int            iter, num_procs, my_id;
   NALU_HYPRE_Int            Solve_err_flag;

   /*
      NALU_HYPRE_Real   total_coeffs;
      NALU_HYPRE_Real   total_variables;
      NALU_HYPRE_Real   operat_cmplxty;
      NALU_HYPRE_Real   grid_cmplxty;
      */
   NALU_HYPRE_Solver         cg_solver = (mgr_data -> coarse_grid_solver);
   NALU_HYPRE_Int            (*coarse_grid_solver_solve)(void*, void*, void*,
                                                    void*) = (mgr_data -> coarse_grid_solver_solve);
   //NALU_HYPRE_Real   wall_time = 0.0;

   //   NALU_HYPRE_Int    i;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;
   if (logging > 1)
   {
      residual = (mgr_data -> residual);
   }

   (mgr_data -> num_iterations) = 0;

   if ((mgr_data -> num_coarse_levels) == 0)
   {
      /* Do scalar AMG solve when only one level */
      coarse_grid_solver_solve(cg_solver, A, f, u);
      NALU_HYPRE_BoomerAMGGetNumIterations(cg_solver, &iter);
      NALU_HYPRE_BoomerAMGGetFinalRelativeResidualNorm(cg_solver, &rel_resnorm);
      (mgr_data -> num_iterations) = iter;
      (mgr_data -> final_rel_residual_norm) = rel_resnorm;
      NALU_HYPRE_ANNOTATE_FUNC_END;

      return nalu_hypre_error_flag;
   }

   U_array[0] = u;
   F_array[0] = f;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/
   if (my_id == 0 && print_level > 1)
   {
      nalu_hypre_MGRWriteSolverParams(mgr_data);
   }

   /*-----------------------------------------------------------------------
    *    Initialize the solver error flag and assorted bookkeeping variables
    *-----------------------------------------------------------------------*/

   Solve_err_flag = 0;
   /*
      total_coeffs = 0;
      total_variables = 0;
      operat_cmplxty = 0;
      grid_cmplxty = 0;
      */
   /*-----------------------------------------------------------------------
    *     write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && print_level > 1 && tol > 0.)
   {
      nalu_hypre_printf("\n\nTWO-GRID SOLVER SOLUTION INFO:\n");
   }


   /*-----------------------------------------------------------------------
    *    Compute initial fine-grid residual and print
    *-----------------------------------------------------------------------*/
   if (print_level > 1 || logging > 1 || tol > 0.)
   {
      if ( logging > 1 )
      {
         nalu_hypre_ParVectorCopy(F_array[0], residual );
         if (tol > 0.0)
         {
            nalu_hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, residual );
         }
         resnorm = sqrt(nalu_hypre_ParVectorInnerProd( residual, residual ));
      }
      else
      {
         nalu_hypre_ParVectorCopy(F_array[0], Vtemp);
         if (tol > 0.0)
         {
            nalu_hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
         }
         resnorm = sqrt(nalu_hypre_ParVectorInnerProd(Vtemp, Vtemp));
      }

      /* Since it is does not diminish performance, attempt to return an error flag
       * and notify users when they supply bad input. */
      if (resnorm != 0.) { ieee_check = resnorm / resnorm; } /* INF -> NaN conversion */
      if (ieee_check != ieee_check)
      {
         /* ...INFs or NaNs in input can make ieee_check a NaN.  This test
          * for ieee_check self-equality works on all IEEE-compliant compilers/
          * machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
          * by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
          * found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
         if (print_level > 0)
         {
            nalu_hypre_printf("\n\nERROR detected by Hypre ...  BEGIN\n");
            nalu_hypre_printf("ERROR -- nalu_hypre_MGRSolve: INFs and/or NaNs detected in input.\n");
            nalu_hypre_printf("User probably placed non-numerics in supplied A, x_0, or b.\n");
            nalu_hypre_printf("ERROR detected by Hypre ...  END\n\n\n");
         }
         nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
         NALU_HYPRE_ANNOTATE_FUNC_END;

         return nalu_hypre_error_flag;
      }

      init_resnorm = resnorm;
      rhs_norm = sqrt(nalu_hypre_ParVectorInnerProd(f, f));
      if (rhs_norm > NALU_HYPRE_REAL_EPSILON)
      {
         rel_resnorm = init_resnorm / rhs_norm;
      }
      else
      {
         /* rhs is zero, return a zero solution */
         nalu_hypre_ParVectorSetConstantValues(U_array[0], 0.0);
         if (logging > 0)
         {
            rel_resnorm = 0.0;
            (mgr_data -> final_rel_residual_norm) = rel_resnorm;
         }
         NALU_HYPRE_ANNOTATE_FUNC_END;

         return nalu_hypre_error_flag;
      }
   }
   else
   {
      rel_resnorm = 1.;
   }

   if (my_id == 0 && print_level > 1)
   {
      nalu_hypre_printf("                                            relative\n");
      nalu_hypre_printf("               residual        factor       residual\n");
      nalu_hypre_printf("               --------        ------       --------\n");
      nalu_hypre_printf("    Initial    %e                 %e\n", init_resnorm,
                   rel_resnorm);
   }
   /************** Main Solver Loop - always do 1 iteration ************/
   iter = 0;
   while ((rel_resnorm >= tol || iter < 1) && iter < max_iter)
   {
      //wall_time = time_getWallclockSeconds();
      /* Do one cycle of reduction solve on Ae=r */
      nalu_hypre_MGRCycle(mgr_data, F_array, U_array);
      //wall_time = time_getWallclockSeconds() - wall_time;
      //if (my_id == 0) nalu_hypre_printf("MGR Cycle time: %f\n", wall_time);

      /*---------------------------------------------------------------
       *    Compute  fine-grid residual and residual norm
       *----------------------------------------------------------------*/

      if (print_level > 1 || logging > 1 || tol > 0.)
      {
         old_resnorm = resnorm;

         if ( logging > 1 )
         {
            nalu_hypre_ParVectorCopy(F_array[0], residual);
            nalu_hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, residual );
            resnorm = sqrt(nalu_hypre_ParVectorInnerProd( residual, residual ));
         }
         else
         {
            nalu_hypre_ParVectorCopy(F_array[0], Vtemp);
            nalu_hypre_ParCSRMatrixMatvec(alpha, A_array[0], U_array[0], beta, Vtemp);
            resnorm = sqrt(nalu_hypre_ParVectorInnerProd(Vtemp, Vtemp));
         }

         if (old_resnorm) { conv_factor = resnorm / old_resnorm; }
         else { conv_factor = resnorm; }
         if (rhs_norm > NALU_HYPRE_REAL_EPSILON)
         {
            rel_resnorm = resnorm / rhs_norm;
         }
         else
         {
            rel_resnorm = resnorm;
         }

         norms[iter] = rel_resnorm;
      }

      ++iter;
      (mgr_data -> num_iterations) = iter;
      (mgr_data -> final_rel_residual_norm) = rel_resnorm;

      if (my_id == 0 && print_level > 1)
      {
         nalu_hypre_printf("    MGRCycle %2d   %e    %f     %e \n", iter,
                      resnorm, conv_factor, rel_resnorm);
      }
   }

   /* check convergence within max_iter */
   if (iter == max_iter && tol > 0.)
   {
      Solve_err_flag = 1;
      nalu_hypre_error(NALU_HYPRE_ERROR_CONV);
   }

   /*-----------------------------------------------------------------------
    *    Print closing statistics
    *    Add operator and grid complexity stats
    *-----------------------------------------------------------------------*/

   if (iter > 0 && init_resnorm)
   {
      conv_factor = pow((resnorm / init_resnorm), (1.0 / (NALU_HYPRE_Real) iter));
   }
   else
   {
      conv_factor = 1.;
   }

   if (print_level > 1)
   {
      /*** compute operator and grid complexities here ?? ***/
      if (my_id == 0)
      {
         if (Solve_err_flag == 1)
         {
            nalu_hypre_printf("\n\n==============================================");
            nalu_hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
            nalu_hypre_printf("      within the allowed %d iterations\n", max_iter);
            nalu_hypre_printf("==============================================");
         }
         nalu_hypre_printf("\n\n Average Convergence Factor = %f \n", conv_factor);
         nalu_hypre_printf(" Number of coarse levels = %d \n", (mgr_data -> num_coarse_levels));
         //         nalu_hypre_printf("\n\n     Complexity:    grid = %f\n",grid_cmplxty);
         //         nalu_hypre_printf("                operator = %f\n",operat_cmplxty);
         //         nalu_hypre_printf("                   cycle = %f\n\n\n\n",cycle_cmplxty);
      }
   }
   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_MGRFrelaxVcycle ( void   *Frelax_vdata, nalu_hypre_ParVector *f, nalu_hypre_ParVector *u )
{
   nalu_hypre_ParAMGData   *Frelax_data = (nalu_hypre_ParAMGData*) Frelax_vdata;

   NALU_HYPRE_Int Not_Finished = 0;
   NALU_HYPRE_Int level = 0;
   NALU_HYPRE_Int cycle_param = 1;
   NALU_HYPRE_Int j, Solve_err_flag, coarse_grid, fine_grid;
   NALU_HYPRE_Int local_size;
   NALU_HYPRE_Int num_sweeps = 1;
   NALU_HYPRE_Int relax_order = nalu_hypre_ParAMGDataRelaxOrder(Frelax_data);
   NALU_HYPRE_Int relax_type = 3;
   NALU_HYPRE_Int relax_weight = 1;
   NALU_HYPRE_Int omega = 1;
   //  NALU_HYPRE_Int max_coarse_size = nalu_hypre_ParAMGDataMaxCoarseSize(Frelax_data);

   nalu_hypre_ParVector    **F_array = (Frelax_data) -> F_array;
   nalu_hypre_ParVector    **U_array = (Frelax_data) -> U_array;

   nalu_hypre_ParCSRMatrix **A_array = ((Frelax_data) -> A_array);
   nalu_hypre_ParCSRMatrix **R_array = ((Frelax_data) -> P_array);
   nalu_hypre_ParCSRMatrix **P_array = ((Frelax_data) -> P_array);
   nalu_hypre_IntArray     **CF_marker_array = ((Frelax_data) -> CF_marker_array);
   NALU_HYPRE_Int           *CF_marker;

   nalu_hypre_ParVector *Vtemp = (Frelax_data) -> Vtemp;
   nalu_hypre_ParVector *Ztemp = (Frelax_data) -> Ztemp;

   NALU_HYPRE_Int num_c_levels = (Frelax_data) -> num_levels;

   nalu_hypre_ParVector *Aux_F = NULL;
   nalu_hypre_ParVector *Aux_U = NULL;

   NALU_HYPRE_Real alpha, beta;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   F_array[0] = f;
   U_array[0] = u;

   CF_marker = NULL;
   if (CF_marker_array[0])
   {
      CF_marker = nalu_hypre_IntArrayData(CF_marker_array[0]);
   }

   /* (Re)set local_size for Vtemp */
   local_size = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(F_array[0]));
   nalu_hypre_ParVectorSetLocalSize(Vtemp, local_size);

   /* smoother on finest level:
    * This is separated from subsequent levels since the finest level matrix
    * may be larger than what is needed for the vcycle solve
    */
   if (relax_order == 1) // C/F ordering for smoother
   {
      for (j = 0; j < num_sweeps; j++)
      {
         Solve_err_flag = nalu_hypre_BoomerAMGRelaxIF(A_array[0],
                                                 F_array[0],
                                                 CF_marker,
                                                 relax_type,
                                                 relax_order,
                                                 1,
                                                 relax_weight,
                                                 omega,
                                                 NULL,
                                                 U_array[0],
                                                 Vtemp,
                                                 Ztemp);
      }
   }
   else // lexicographic ordering for smoother (on F points in CF marker)
   {
      for (j = 0; j < num_sweeps; j++)
      {
         Solve_err_flag = nalu_hypre_BoomerAMGRelax(A_array[0],
                                               F_array[0],
                                               CF_marker,
                                               relax_type,
                                               -1,
                                               relax_weight,
                                               omega,
                                               NULL,
                                               U_array[0],
                                               Vtemp,
                                               Ztemp);
      }
   }

   /* coarse grids exist */
   if (num_c_levels > 0)
   {
      Not_Finished = 1;
   }

   while (Not_Finished)
   {
      if (cycle_param == 1)
      {
         //nalu_hypre_printf("Vcycle smoother (down cycle): vtemp size = %d, level = %d \n", nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(Vtemp)), level);
         /* compute coarse grid vectors */
         fine_grid = level;
         coarse_grid = level + 1;

         nalu_hypre_ParVectorSetConstantValues(U_array[coarse_grid], 0.0);

         alpha = -1.0;
         beta = 1.0;

         // JSP: avoid unnecessary copy using out-of-place version of SpMV
         nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[fine_grid], U_array[fine_grid],
                                            beta, F_array[fine_grid], Vtemp);

         alpha = 1.0;
         beta = 0.0;
         nalu_hypre_ParCSRMatrixMatvecT(alpha, R_array[fine_grid], Vtemp,
                                   beta, F_array[coarse_grid]);

         /* update level */
         ++level;

         /* Update scratch vector sizes */
         local_size = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(F_array[level]));
         nalu_hypre_ParVectorSetLocalSize(Vtemp, local_size);
         nalu_hypre_ParVectorSetLocalSize(Ztemp, local_size);

         CF_marker = NULL;
         if (CF_marker_array[level])
         {
            CF_marker = nalu_hypre_IntArrayData(CF_marker_array[level]);
         }

         /* next level is coarsest level */
         if (level == num_c_levels)
         {
            /* switch to coarsest level */
            cycle_param = 3;
         }
         else
         {
            Aux_F = F_array[level];
            Aux_U = U_array[level];
            /* relax and visit next coarse grid */
            for (j = 0; j < num_sweeps; j++)
            {
               Solve_err_flag = nalu_hypre_BoomerAMGRelaxIF(A_array[level],
                                                       Aux_F,
                                                       CF_marker,
                                                       relax_type,
                                                       relax_order,
                                                       cycle_param,
                                                       relax_weight,
                                                       omega,
                                                       NULL,
                                                       Aux_U,
                                                       Vtemp,
                                                       Ztemp);
            }
            cycle_param = 1;
         }
      }
      else if (cycle_param == 3)
      {
         if (nalu_hypre_ParAMGDataUserCoarseRelaxType(Frelax_data) == 9)
         {
            // solve the coarsest grid with Gaussian elimination
            nalu_hypre_GaussElimSolve(Frelax_data, level, 9);
         }
         else
         {
            // solve with relaxation
            Aux_F = F_array[level];
            Aux_U = U_array[level];
            for (j = 0; j < num_sweeps; j++)
            {
               Solve_err_flag = nalu_hypre_BoomerAMGRelaxIF(A_array[level],
                                                       Aux_F,
                                                       CF_marker,
                                                       relax_type,
                                                       relax_order,
                                                       cycle_param,
                                                       relax_weight,
                                                       omega,
                                                       NULL,
                                                       Aux_U,
                                                       Vtemp,
                                                       Ztemp);
            }
         }
         //nalu_hypre_printf("Vcycle smoother (coarse level): vtemp size = %d, level = %d \n", nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(Vtemp)), level);
         cycle_param = 2;
      }
      else if (cycle_param == 2)
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
         nalu_hypre_ParCSRMatrixMatvec(alpha, P_array[fine_grid],
                                  U_array[coarse_grid],
                                  beta, U_array[fine_grid]);

         --level;
         cycle_param = 2;
         if (level == 0) { cycle_param = 99; }

         /* Update scratch vector sizes */
         local_size = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(F_array[level]));
         nalu_hypre_ParVectorSetLocalSize(Vtemp, local_size);
         nalu_hypre_ParVectorSetLocalSize(Ztemp, local_size);
         //nalu_hypre_printf("Vcycle smoother (up cycle): vtemp size = %d, level = %d \n", nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(Vtemp)), level);
      }
      else
      {
         Not_Finished = 0;
      }
   }
   NALU_HYPRE_ANNOTATE_FUNC_END;

   return Solve_err_flag;
}

NALU_HYPRE_Int
nalu_hypre_MGRCycle( void               *mgr_vdata,
                nalu_hypre_ParVector    **F_array,
                nalu_hypre_ParVector    **U_array )
{
   MPI_Comm          comm;
   nalu_hypre_ParMGRData   *mgr_data = (nalu_hypre_ParMGRData*) mgr_vdata;

   NALU_HYPRE_Int       local_size;
   NALU_HYPRE_Int       Solve_err_flag;
   NALU_HYPRE_Int       level;
   NALU_HYPRE_Int       coarse_grid;
   NALU_HYPRE_Int       fine_grid;
   NALU_HYPRE_Int       Not_Finished;
   NALU_HYPRE_Int       cycle_type;
   NALU_HYPRE_Int            print_level = (mgr_data -> print_level);
   NALU_HYPRE_Int            frelax_print_level = (mgr_data -> frelax_print_level);

   nalu_hypre_ParCSRMatrix   **A_array = (mgr_data -> A_array);
   nalu_hypre_ParCSRMatrix   **RT_array  = (mgr_data -> RT_array);
   nalu_hypre_ParCSRMatrix   **P_array   = (mgr_data -> P_array);
#if defined(NALU_HYPRE_USING_GPU)
   nalu_hypre_ParCSRMatrix   **P_FF_array   = (mgr_data -> P_FF_array);
#endif
   nalu_hypre_ParCSRMatrix   *RAP = (mgr_data -> RAP);
   NALU_HYPRE_Int  use_default_cgrid_solver = (mgr_data -> use_default_cgrid_solver);
   NALU_HYPRE_Solver         cg_solver = (mgr_data -> coarse_grid_solver);
   NALU_HYPRE_Int            (*coarse_grid_solver_solve)(void*, void*, void*,
                                                    void*) = (mgr_data -> coarse_grid_solver_solve);

   nalu_hypre_IntArray     **CF_marker = (mgr_data -> CF_marker_array);
   NALU_HYPRE_Int          *nsweeps = (mgr_data -> num_relax_sweeps);
   NALU_HYPRE_Int            relax_type = (mgr_data -> relax_type);
   NALU_HYPRE_Real           relax_weight = (mgr_data -> relax_weight);
   NALU_HYPRE_Real           omega = (mgr_data -> omega);
   nalu_hypre_Vector       **relax_l1_norms = (mgr_data -> l1_norms);
   nalu_hypre_ParVector      *Vtemp = (mgr_data -> Vtemp);
   nalu_hypre_ParVector      *Ztemp = (mgr_data -> Ztemp);
   nalu_hypre_ParVector      *Utemp = (mgr_data -> Utemp);

   nalu_hypre_ParVector      **U_fine_array = (mgr_data -> U_fine_array);
   nalu_hypre_ParVector      **F_fine_array = (mgr_data -> F_fine_array);
   NALU_HYPRE_Int      (*fine_grid_solver_solve)(void*, void*, void*,
                                            void*) = (mgr_data -> fine_grid_solver_solve);
   nalu_hypre_ParCSRMatrix   **A_ff_array = (mgr_data -> A_ff_array);

   NALU_HYPRE_Int            i, relax_points;
   NALU_HYPRE_Int            num_coarse_levels = (mgr_data -> num_coarse_levels);

   NALU_HYPRE_Real    alpha;
   NALU_HYPRE_Real    beta;

   NALU_HYPRE_Int           *Frelax_type = (mgr_data -> Frelax_type);
   NALU_HYPRE_Int           *interp_type = (mgr_data -> interp_type);
   nalu_hypre_ParAMGData    **FrelaxVcycleData = (mgr_data -> FrelaxVcycleData);
   NALU_HYPRE_Real          **frelax_diaginv = (mgr_data -> frelax_diaginv);
   NALU_HYPRE_Int           *blk_size = (mgr_data -> blk_size);

   NALU_HYPRE_Int      *level_smooth_type = (mgr_data -> level_smooth_type);
   NALU_HYPRE_Int      *level_smooth_iters = (mgr_data -> level_smooth_iters);

   NALU_HYPRE_Int      *restrict_type = (mgr_data -> restrict_type);
   NALU_HYPRE_Int      pre_smoothing = (mgr_data -> global_smooth_cycle) == 1 ? 1 : 0;
   NALU_HYPRE_Int      post_smoothing = (mgr_data -> global_smooth_cycle) == 2 ? 1 : 0;
   NALU_HYPRE_Int      use_air = 0;
   NALU_HYPRE_Int      my_id;

   // NALU_HYPRE_Real     wall_time;
   NALU_HYPRE_Int    block_size = (mgr_data -> block_size);
   NALU_HYPRE_Int    *block_num_coarse_indexes = (mgr_data -> block_num_coarse_indexes);

   /* Initialize */
   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   comm = nalu_hypre_ParCSRMatrixComm(A_array[0]);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   Solve_err_flag = 0;
   Not_Finished = 1;
   cycle_type = 1;
   level = 0;

   /***** Main loop ******/
   while (Not_Finished)
   {
      /* Update scratch vector sizes */
      local_size = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(F_array[level]));
      nalu_hypre_ParVectorSetLocalSize(Vtemp, local_size);
      nalu_hypre_ParVectorSetLocalSize(Ztemp, local_size);
      nalu_hypre_ParVectorSetLocalSize(Utemp, local_size);

      /* Do coarse grid correction solve */
      if (cycle_type == 3)
      {
         /* call coarse grid solver here */
         /* default is BoomerAMG */
         //wall_time = time_getWallclockSeconds();
         coarse_grid_solver_solve(cg_solver, RAP, F_array[level], U_array[level]);
         if (use_default_cgrid_solver)
         {
            NALU_HYPRE_Real convergence_factor_cg;
            nalu_hypre_BoomerAMGGetRelResidualNorm(cg_solver, &convergence_factor_cg);
            (mgr_data -> cg_convergence_factor) = convergence_factor_cg;
            if ((print_level) > 1 && my_id == 0 && convergence_factor_cg > 1.0)
            {
               nalu_hypre_printf("Warning!!! Coarse grid solve diverges. Factor = %1.2e\n", convergence_factor_cg);
            }
         }

         // DEBUG: print the coarse system indicated by mgr_data ->print_coarse_system
         if (mgr_data -> print_coarse_system)
         {
            nalu_hypre_ParCSRMatrixPrintIJ(RAP, 1, 1, "RAP_mat");
            nalu_hypre_ParVectorPrintIJ(F_array[level], 1, "RAP_rhs");
            nalu_hypre_ParVectorPrintIJ(U_array[level], 1, "RAP_sol");
            mgr_data -> print_coarse_system--;
         }
         /**** cycle up ***/
         cycle_type = 2;
      }
      /* F-relaxation */
      else if (cycle_type == 1)
      {
         if (pre_smoothing)
         {
            if (level_smooth_iters[level] > 0)
            {
               //wall_time = time_getWallclockSeconds();
               //if (lvl_smth_type == 0 || lvl_smth_type == 1) //block Jacobi smoother
               if (level_smooth_type[level] == 0 || level_smooth_type[level] == 1) //block Jacobi smoother
               {
                  NALU_HYPRE_Real *level_diaginv = (mgr_data -> level_diaginv)[level];
                  NALU_HYPRE_Int level_blk_size = (level == 0 ? block_size : block_num_coarse_indexes[level - 1]);
                  NALU_HYPRE_Int nrows = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A_array[level]));
                  NALU_HYPRE_Int n_block = nrows / level_blk_size;
                  NALU_HYPRE_Int left_size = nrows - n_block * level_blk_size;
                  for (i = 0; i < level_smooth_iters[level]; i++)
                  {
                     nalu_hypre_MGRBlockRelaxSolve(A_array[level], F_array[level], U_array[level], level_blk_size, n_block,
                                              left_size, level_smooth_type[level], level_diaginv, Vtemp);
                  }
               }
               else if ((level_smooth_type[level] > 1) && (level_smooth_type[level] < 7))
               {
                  for (i = 0; i < level_smooth_iters[level]; i ++)
                  {
                     nalu_hypre_BoomerAMGRelax(A_array[level], F_array[level], NULL, level_smooth_type[level] - 1, 0, 1.0,
                                          0.0, NULL, U_array[level], Vtemp, NULL);
                  }
               }
               else if (level_smooth_type[level] == 8) //EUCLID ILU smoother
               {
                  for (i = 0; i < level_smooth_iters[level]; i++)
                  {
                     // compute residual
                     nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[level], U_array[level], beta, F_array[level],
                                                        Vtemp);
                     // solve
                     NALU_HYPRE_EuclidSolve( (mgr_data -> level_smoother)[level], A_array[level], Vtemp, Utemp);
                     // update solution
                     nalu_hypre_ParVectorAxpy(beta, Utemp, U_array[level]);
                  }
               }
               else if (level_smooth_type[level] == 16) // HYPRE ILU
               {
                  // solve
                  NALU_HYPRE_ILUSolve((mgr_data -> level_smoother)[level], A_array[level], F_array[level], U_array[level]);
               }
            }
         }// end pre-smoothing

         fine_grid = level;
         coarse_grid = level + 1;
         /* Relax solution - F-relaxation */
         relax_points = -1;

         //wall_time = time_getWallclockSeconds();
         if (Frelax_type[level] == 0)
         {
            /* (single level) relaxation for A_ff */
            if (interp_type[level] == 12)
            {
               NALU_HYPRE_Int nrows = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A_ff_array[fine_grid]));
               NALU_HYPRE_Int n_block = nrows / blk_size[fine_grid];
               NALU_HYPRE_Int left_size = nrows - n_block * blk_size[fine_grid];
               for (i = 0; i < nsweeps[level]; i++)
               {
                  // Block Jacobi
                  nalu_hypre_MGRAddVectorR(CF_marker[fine_grid], FMRK, 1.0, F_array[fine_grid], 0.0,
                                      &(F_fine_array[coarse_grid]));
                  nalu_hypre_ParVectorSetConstantValues(U_fine_array[coarse_grid], 0.0);

                  nalu_hypre_MGRBlockRelaxSolve(A_ff_array[fine_grid], F_fine_array[coarse_grid],
                                           U_fine_array[coarse_grid], blk_size[fine_grid], n_block, left_size, 0, frelax_diaginv[fine_grid],
                                           Vtemp);
                  nalu_hypre_MGRAddVectorP(CF_marker[fine_grid], FMRK, 1.0, U_fine_array[coarse_grid], 1.0,
                                      &(U_array[fine_grid]));
               }
            }
            else
            {
               if (relax_type == 18)
               {
#if defined (NALU_HYPRE_USING_CUDA) || defined (NALU_HYPRE_USING_HIP)
                  for (i = 0; i < nsweeps[level]; i++)
                  {
                     nalu_hypre_MGRRelaxL1JacobiDevice(A_array[fine_grid], F_array[fine_grid],
                                                  nalu_hypre_IntArrayData(CF_marker[fine_grid]),
                                                  relax_points, relax_weight,
                                                  relax_l1_norms[fine_grid] ? nalu_hypre_VectorData(relax_l1_norms[fine_grid]) : NULL,
                                                  U_array[fine_grid], Vtemp);
                  }
#else
                  for (i = 0; i < nsweeps[level]; i++)
                  {
                     nalu_hypre_ParCSRRelax_L1_Jacobi(A_array[fine_grid], F_array[fine_grid],
                                                 nalu_hypre_IntArrayData(CF_marker[fine_grid]),
                                                 relax_points, relax_weight,
                                                 relax_l1_norms[fine_grid] ? nalu_hypre_VectorData(relax_l1_norms[fine_grid]) : NULL,
                                                 U_array[fine_grid], Vtemp);
                  }
#endif
               }
               else if (relax_type == 8 || relax_type == 13 || relax_type == 14)
               {
                  for (i = 0; i < nsweeps[level]; i++)
                  {
                     nalu_hypre_BoomerAMGRelax(A_array[fine_grid], F_array[fine_grid],
                                          nalu_hypre_IntArrayData(CF_marker[fine_grid]),
                                          relax_type, relax_points, relax_weight,
                                          omega,
                                          relax_l1_norms[fine_grid] ? nalu_hypre_VectorData(relax_l1_norms[fine_grid]) : NULL,
                                          U_array[fine_grid], Vtemp, Ztemp);
                  }
               }
               else
               {
                  for (i = 0; i < nsweeps[level]; i++)
                  {
                     Solve_err_flag = nalu_hypre_BoomerAMGRelax(A_array[fine_grid], F_array[fine_grid],
                                                           nalu_hypre_IntArrayData(CF_marker[fine_grid]),
                                                           relax_type, relax_points, relax_weight, omega, NULL, U_array[fine_grid], Vtemp, Ztemp);
                  }
               }
            }
         }
         else if (Frelax_type[level] == 1)
         {
            /* v-cycle smoother for A_ff */
            //NALU_HYPRE_Real convergence_factor_frelax;
            // compute residual before solve
            // nalu_hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A_array[level],
            //                                    U_array[level], 1.0, F_array[level], Vtemp);
            //  convergence_factor_frelax = nalu_hypre_ParVectorInnerProd(Vtemp, Vtemp);

            NALU_HYPRE_Real resnorm, init_resnorm;
            NALU_HYPRE_Real rhs_norm, old_resnorm;
            NALU_HYPRE_Real rel_resnorm = 1.0;
            NALU_HYPRE_Real conv_factor = 1.0;
            if (frelax_print_level > 1)
            {
               nalu_hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A_array[level],
                                                  U_array[level], 1.0, F_array[level], Vtemp);

               resnorm = sqrt(nalu_hypre_ParVectorInnerProd(Vtemp, Vtemp));
               init_resnorm = resnorm;
               rhs_norm = sqrt(nalu_hypre_ParVectorInnerProd(F_array[level], F_array[level]));

               if (rhs_norm > NALU_HYPRE_REAL_EPSILON)
               {
                  rel_resnorm = init_resnorm / rhs_norm;
               }
               else
               {
                  /* rhs is zero, return a zero solution */
                  nalu_hypre_ParVectorSetConstantValues(U_array[0], 0.0);

                  NALU_HYPRE_ANNOTATE_FUNC_END;

                  return nalu_hypre_error_flag;
               }
               if (my_id == 0 && frelax_print_level > 1)
               {
                  nalu_hypre_printf("\nBegin F-relaxation: V-Cycle Smoother \n");
                  nalu_hypre_printf("                                            relative\n");
                  nalu_hypre_printf("               residual        factor       residual\n");
                  nalu_hypre_printf("               --------        ------       --------\n");
                  nalu_hypre_printf("    Initial    %e                 %e\n", init_resnorm,
                               rel_resnorm);
               }
            }

            for (i = 0; i < nsweeps[level]; i++)
            {
               nalu_hypre_MGRFrelaxVcycle(FrelaxVcycleData[level], F_array[level], U_array[level]);

               if (frelax_print_level > 1)
               {
                  old_resnorm = resnorm;
                  nalu_hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A_array[level],
                                                     U_array[level], 1.0, F_array[level], Vtemp);
                  resnorm = sqrt(nalu_hypre_ParVectorInnerProd(Vtemp, Vtemp));

                  if (old_resnorm) { conv_factor = resnorm / old_resnorm; }
                  else { conv_factor = resnorm; }

                  if (rhs_norm > NALU_HYPRE_REAL_EPSILON)
                  {
                     rel_resnorm = resnorm / rhs_norm;
                  }
                  else
                  {
                     rel_resnorm = resnorm;
                  }

                  if (my_id == 0 )
                  {
                     nalu_hypre_printf("\n    V-Cycle %2d   %e    %f     %e \n", i,
                                  resnorm, conv_factor, rel_resnorm);
                  }
               }
            }
            if (my_id == 0 && frelax_print_level > 1)
            {
               nalu_hypre_printf("End F-relaxation: V-Cycle Smoother \n\n");
            }
            // compute residual after solve
            //nalu_hypre_ParCSRMatrixMatvecOutOfPlace(-1.0, A_array[level],
            //                                  U_array[level], 1.0, F_array[level], Vtemp);
            //convergence_factor_frelax = nalu_hypre_ParVectorInnerProd(Vtemp, Vtemp)/convergence_factor_frelax;
            //nalu_hypre_printf("F-relaxation V-cycle convergence factor: %5f\n", convergence_factor_frelax);
         }
         else if (Frelax_type[level] == 2)
         {
            // We need to first compute residual to ensure that
            // F-relaxation is reducing the global residual
            alpha = -1.0;
            beta = 1.0;
            nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[fine_grid], U_array[fine_grid], beta,
                                               F_array[fine_grid], Vtemp);

            // restrict to F points
#if defined (NALU_HYPRE_USING_GPU)
            nalu_hypre_ParCSRMatrixMatvecT(1.0, P_FF_array[fine_grid], Vtemp, 0.0, F_fine_array[coarse_grid]);
#else
            nalu_hypre_MGRAddVectorR(CF_marker[fine_grid], FMRK, 1.0, Vtemp, 0.0, &(F_fine_array[coarse_grid]));
#endif
            nalu_hypre_ParVectorSetConstantValues(U_fine_array[coarse_grid], 0.0);
            // Do F-relaxation using AMG
            fine_grid_solver_solve((mgr_data -> aff_solver)[fine_grid], A_ff_array[fine_grid],
                                   F_fine_array[coarse_grid],
                                   U_fine_array[coarse_grid]);

            // Interpolate the solution back to the fine grid level
#if defined (NALU_HYPRE_USING_GPU)
            nalu_hypre_ParCSRMatrixMatvec(1.0, P_FF_array[fine_grid], U_fine_array[coarse_grid], 1.0,
                                     U_array[fine_grid]);
#else
            nalu_hypre_MGRAddVectorP(CF_marker[fine_grid], FMRK, 1.0, U_fine_array[coarse_grid], 1.0,
                                &(U_array[fine_grid]));
#endif
         }
         else
         {
            for (i = 0; i < nsweeps[level]; i++)
            {
               Solve_err_flag = nalu_hypre_BoomerAMGRelax(A_array[fine_grid], F_array[fine_grid],
                                                     nalu_hypre_IntArrayData(CF_marker[fine_grid]),
                                                     relax_type, relax_points, relax_weight, omega, NULL, U_array[fine_grid], Vtemp, Ztemp);
            }
         }
         //wall_time = time_getWallclockSeconds() - wall_time;
         //if (my_id == 0) nalu_hypre_printf("F-relaxation solve level %d: %f\n", coarse_grid, wall_time);

         // Update residual and compute coarse-grid rhs
         alpha = -1.0;
         beta = 1.0;

         nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[fine_grid], U_array[fine_grid],
                                            beta, F_array[fine_grid], Vtemp);

         alpha = 1.0;
         beta = 0.0;

         if (restrict_type[fine_grid] == 4 || restrict_type[fine_grid] == 5)
         {
            use_air = 1;
         }

         if (use_air)
         {
            /* no transpose necessary for R */
            nalu_hypre_ParCSRMatrixMatvec(alpha, RT_array[fine_grid], Vtemp,
                                     beta, F_array[coarse_grid]);
         }
         else
         {
            if (restrict_type[level] > 0)
            {
               nalu_hypre_ParCSRMatrixMatvecT(alpha, RT_array[fine_grid], Vtemp,
                                         beta, F_array[coarse_grid]);
            }
            else
            {
               nalu_hypre_MGRAddVectorR(CF_marker[fine_grid], CMRK, 1.0, Vtemp, 0.0, &(F_array[coarse_grid]));
            }
         }
         // initialize coarse grid solution array
         nalu_hypre_ParVectorSetConstantValues(U_array[coarse_grid], 0.0);

         ++level;

         if (level == num_coarse_levels) { cycle_type = 3; }
         //wall_time = time_getWallclockSeconds() - wall_time;
         //if (my_id == 0) nalu_hypre_printf("Solve restrict time: %f\n", wall_time);
      } // end cycle_type == 1
      else if (level != 0)
      {
         /* Interpolate */
         //wall_time = time_getWallclockSeconds();
         fine_grid = level - 1;
         coarse_grid = level;
         alpha = 1.0;
         beta = 1.0;

         if (interp_type[fine_grid] > 0)
         {
            nalu_hypre_ParCSRMatrixMatvec(alpha, P_array[fine_grid],
                                     U_array[coarse_grid],
                                     beta, U_array[fine_grid]);
         }
         else
         {
            nalu_hypre_MGRAddVectorP(CF_marker[fine_grid], CMRK, 1.0, U_array[coarse_grid], 1.0,
                                &(U_array[fine_grid]));
         }

         /* post smoothing sweeps */
         if (post_smoothing)
         {
            if (level_smooth_iters[fine_grid] > 0)
            {
               //wall_time = time_getWallclockSeconds();
               //if (lvl_smth_type == 0 || lvl_smth_type == 1) //block Jacobi smoother
               if (level_smooth_type[fine_grid] == 0 || level_smooth_type[fine_grid] == 1) //block Jacobi smoother
               {
                  NALU_HYPRE_Real *level_diaginv = (mgr_data -> level_diaginv)[fine_grid];
                  NALU_HYPRE_Int level_blk_size = (fine_grid == 0 ? block_size : block_num_coarse_indexes[fine_grid - 1]);
                  NALU_HYPRE_Int nrows = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A_array[fine_grid]));
                  NALU_HYPRE_Int n_block = nrows / level_blk_size;
                  NALU_HYPRE_Int left_size = nrows - n_block * level_blk_size;
                  for (i = 0; i < level_smooth_iters[fine_grid]; i++)
                  {
                     nalu_hypre_MGRBlockRelaxSolve(A_array[fine_grid], F_array[fine_grid], U_array[fine_grid], level_blk_size,
                                              n_block,
                                              left_size, level_smooth_type[fine_grid], level_diaginv, Vtemp);
                  }
               }
               else if ((level_smooth_type[fine_grid] > 1) && (level_smooth_type[fine_grid] < 7))
               {
                  for (i = 0; i < level_smooth_iters[level]; i ++)
                  {
                     nalu_hypre_BoomerAMGRelax(A_array[fine_grid], F_array[fine_grid], NULL, level_smooth_type[fine_grid] - 1,
                                          0, 1.0,
                                          0.0, NULL, U_array[fine_grid], Vtemp, NULL);
                  }
               }
               else if (level_smooth_type[fine_grid] == 8) //EUCLID ILU smoother
               {
                  for (i = 0; i < level_smooth_iters[fine_grid]; i++)
                  {
                     // compute residual
                     nalu_hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A_array[fine_grid], U_array[fine_grid], beta,
                                                        F_array[fine_grid],
                                                        Vtemp);
                     // solve
                     NALU_HYPRE_EuclidSolve( (mgr_data -> level_smoother)[fine_grid], A_array[fine_grid], Vtemp, Utemp);
                     // update solution
                     nalu_hypre_ParVectorAxpy(beta, Utemp, U_array[fine_grid]);
                  }
               }
               else if (level_smooth_type[fine_grid] == 16) // HYPRE ILU
               {
                  // solve
                  NALU_HYPRE_ILUSolve((mgr_data -> level_smoother)[fine_grid], A_array[fine_grid], F_array[fine_grid],
                                 U_array[fine_grid]);
               }
            }
         }// end post-smoothing

         if (Solve_err_flag != 0)
         {
            NALU_HYPRE_ANNOTATE_FUNC_END;
            return (Solve_err_flag);
         }

         --level;
         //wall_time = time_getWallclockSeconds() - wall_time;
         //if (my_id == 0) nalu_hypre_printf("Solve interp time: %f\n", wall_time);
      } // end interpolate
      else
      {
         Not_Finished = 0;
      }
   }
   NALU_HYPRE_ANNOTATE_FUNC_END;

   return Solve_err_flag;
}
