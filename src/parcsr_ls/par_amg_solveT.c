/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * AMG transpose solve routines
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "par_amg.h"

/*--------------------------------------------------------------------
 * nalu_hypre_BoomerAMGSolveT
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGSolveT( void               *amg_vdata,
                       nalu_hypre_ParCSRMatrix *A,
                       nalu_hypre_ParVector    *f,
                       nalu_hypre_ParVector    *u         )
{

   MPI_Comm          comm = nalu_hypre_ParCSRMatrixComm(A);

   nalu_hypre_ParAMGData   *amg_data = (nalu_hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */

   NALU_HYPRE_Int      amg_print_level;
   NALU_HYPRE_Int      amg_logging;
   NALU_HYPRE_Real  *num_coeffs;
   NALU_HYPRE_Int     *num_variables;
   NALU_HYPRE_Real   cycle_op_count;
   NALU_HYPRE_Int      num_levels;
   /* NALU_HYPRE_Int      num_unknowns; */
   NALU_HYPRE_Real   tol;
   nalu_hypre_ParCSRMatrix **A_array;
   nalu_hypre_ParVector    **F_array;
   nalu_hypre_ParVector    **U_array;

   /*  Local variables  */

   /*FILE    *fp;*/

   NALU_HYPRE_Int      j;
   NALU_HYPRE_Int      Solve_err_flag;
   NALU_HYPRE_Int      min_iter;
   NALU_HYPRE_Int      max_iter;
   NALU_HYPRE_Int      cycle_count;
   NALU_HYPRE_Real   total_coeffs;
   NALU_HYPRE_Int      total_variables;
   NALU_HYPRE_Int      num_procs, my_id;

   NALU_HYPRE_Real   alpha = 1.0;
   NALU_HYPRE_Real   beta = -1.0;
   NALU_HYPRE_Real   cycle_cmplxty = 0.0;
   NALU_HYPRE_Real   operat_cmplxty;
   NALU_HYPRE_Real   grid_cmplxty;
   NALU_HYPRE_Real   conv_factor;
   NALU_HYPRE_Real   resid_nrm;
   NALU_HYPRE_Real   resid_nrm_init;
   NALU_HYPRE_Real   relative_resid;
   NALU_HYPRE_Real   rhs_norm;
   NALU_HYPRE_Real   old_resid;

   nalu_hypre_ParVector  *Vtemp;
   nalu_hypre_ParVector  *Residual;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   amg_print_level = nalu_hypre_ParAMGDataPrintLevel(amg_data);
   amg_logging   = nalu_hypre_ParAMGDataLogging(amg_data);
   if ( amg_logging > 1 )
   {
      Residual = nalu_hypre_ParAMGDataResidual(amg_data);
   }
   /* num_unknowns  = nalu_hypre_ParAMGDataNumUnknowns(amg_data); */
   num_levels    = nalu_hypre_ParAMGDataNumLevels(amg_data);
   A_array       = nalu_hypre_ParAMGDataAArray(amg_data);
   F_array       = nalu_hypre_ParAMGDataFArray(amg_data);
   U_array       = nalu_hypre_ParAMGDataUArray(amg_data);

   tol           = nalu_hypre_ParAMGDataTol(amg_data);
   min_iter      = nalu_hypre_ParAMGDataMinIter(amg_data);
   max_iter      = nalu_hypre_ParAMGDataMaxIter(amg_data);

   num_coeffs = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_levels, NALU_HYPRE_MEMORY_HOST);
   num_variables = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_levels, NALU_HYPRE_MEMORY_HOST);
   num_coeffs[0]    = nalu_hypre_ParCSRMatrixDNumNonzeros(A_array[0]);
   num_variables[0] = nalu_hypre_ParCSRMatrixGlobalNumRows(A_array[0]);

   A_array[0] = A;
   F_array[0] = f;
   U_array[0] = u;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   /*   Vtemp = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A_array[0]),
                                    nalu_hypre_ParCSRMatrixGlobalNumRows(A_array[0]),
                                    nalu_hypre_ParCSRMatrixRowStarts(A_array[0]));
      nalu_hypre_ParVectorInitialize(Vtemp);
      nalu_hypre_ParAMGDataVtemp(amg_data) = Vtemp;
   */
   Vtemp = nalu_hypre_ParAMGDataVtemp(amg_data);
   for (j = 1; j < num_levels; j++)
   {
      num_coeffs[j]    = nalu_hypre_ParCSRMatrixDNumNonzeros(A_array[j]);
      num_variables[j] = nalu_hypre_ParCSRMatrixGlobalNumRows(A_array[j]);
   }

   /*-----------------------------------------------------------------------
    *    Write the solver parameters
    *-----------------------------------------------------------------------*/


   if (my_id == 0 && amg_print_level > 1)
   {
      nalu_hypre_BoomerAMGWriteSolverParams(amg_data);
   }



   /*-----------------------------------------------------------------------
    *    Initialize the solver error flag and assorted bookkeeping variables
    *-----------------------------------------------------------------------*/

   Solve_err_flag = 0;

   total_coeffs = 0;
   total_variables = 0;
   cycle_count = 0;
   operat_cmplxty = 0;
   grid_cmplxty = 0;

   /*-----------------------------------------------------------------------
    *     open the log file and write some initial info
    *-----------------------------------------------------------------------*/

   if (my_id == 0 && amg_print_level > 1)
   {
      /*fp = fopen(file_name, "a");*/

      nalu_hypre_printf("\n\nAMG SOLUTION INFO:\n");

   }

   /*-----------------------------------------------------------------------
    *    Compute initial fine-grid residual and print to logfile
    *-----------------------------------------------------------------------*/

   if ( amg_logging > 1 )
   {
      nalu_hypre_ParVectorCopy(F_array[0], Residual );
      nalu_hypre_ParCSRMatrixMatvecT(alpha, A_array[0], U_array[0], beta, Residual );
      resid_nrm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd( Residual, Residual ));
   }
   else
   {
      nalu_hypre_ParVectorCopy(F_array[0], Vtemp);
      nalu_hypre_ParCSRMatrixMatvecT(alpha, A_array[0], U_array[0], beta, Vtemp);
      resid_nrm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(Vtemp, Vtemp));
   }


   resid_nrm_init = resid_nrm;
   rhs_norm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(f, f));
   relative_resid = 9999;
   if (rhs_norm)
   {
      relative_resid = resid_nrm_init / rhs_norm;
   }

   if (my_id == 0 && (amg_print_level > 1))
   {
      nalu_hypre_printf("                                            relative\n");
      nalu_hypre_printf("               residual        factor       residual\n");
      nalu_hypre_printf("               --------        ------       --------\n");
      nalu_hypre_printf("    Initial    %e                 %e\n", resid_nrm_init,
                   relative_resid);
   }

   /*-----------------------------------------------------------------------
    *    Main V-cycle loop
    *-----------------------------------------------------------------------*/

   while ((relative_resid >= tol || cycle_count < min_iter)
          && cycle_count < max_iter
          && Solve_err_flag == 0)
   {
      nalu_hypre_ParAMGDataCycleOpCount(amg_data) = 0;
      /* Op count only needed for one cycle */

      Solve_err_flag = nalu_hypre_BoomerAMGCycleT(amg_data, F_array, U_array);

      old_resid = resid_nrm;

      /*---------------------------------------------------------------
       *    Compute  fine-grid residual and residual norm
       *----------------------------------------------------------------*/

      if ( amg_logging > 1 )
      {
         nalu_hypre_ParVectorCopy(F_array[0], Residual );
         nalu_hypre_ParCSRMatrixMatvecT(alpha, A_array[0], U_array[0], beta, Residual );
         resid_nrm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd( Residual, Residual ));
      }
      else
      {
         nalu_hypre_ParVectorCopy(F_array[0], Vtemp);
         nalu_hypre_ParCSRMatrixMatvecT(alpha, A_array[0], U_array[0], beta, Vtemp);
         resid_nrm = nalu_hypre_sqrt(nalu_hypre_ParVectorInnerProd(Vtemp, Vtemp));
      }

      conv_factor = resid_nrm / old_resid;
      relative_resid = 9999;
      if (rhs_norm)
      {
         relative_resid = resid_nrm / rhs_norm;
      }

      ++cycle_count;



      nalu_hypre_ParAMGDataRelativeResidualNorm(amg_data) = relative_resid;
      nalu_hypre_ParAMGDataNumIterations(amg_data) = cycle_count;

      if (my_id == 0 && (amg_print_level > 1))
      {
         nalu_hypre_printf("    Cycle %2d   %e    %f     %e \n", cycle_count,
                      resid_nrm, conv_factor, relative_resid);
      }
   }

   if (cycle_count == max_iter) { Solve_err_flag = 1; }

   /*-----------------------------------------------------------------------
    *    Compute closing statistics
    *-----------------------------------------------------------------------*/

   conv_factor = nalu_hypre_pow((resid_nrm / resid_nrm_init), (1.0 / ((NALU_HYPRE_Real) cycle_count)));


   for (j = 0; j < nalu_hypre_ParAMGDataNumLevels(amg_data); j++)
   {
      total_coeffs += num_coeffs[j];
      total_variables += num_variables[j];
   }

   cycle_op_count = nalu_hypre_ParAMGDataCycleOpCount(amg_data);

   if (num_variables[0])
   {
      grid_cmplxty = ((NALU_HYPRE_Real) total_variables) / ((NALU_HYPRE_Real) num_variables[0]);
   }
   if (num_coeffs[0])
   {
      operat_cmplxty = total_coeffs / num_coeffs[0];
      cycle_cmplxty = cycle_op_count / num_coeffs[0];
   }

   if (my_id == 0 && amg_print_level > 1)
   {
      if (Solve_err_flag == 1)
      {
         nalu_hypre_printf("\n\n==============================================");
         nalu_hypre_printf("\n NOTE: Convergence tolerance was not achieved\n");
         nalu_hypre_printf("      within the allowed %d V-cycles\n", max_iter);
         nalu_hypre_printf("==============================================");
      }
      nalu_hypre_printf("\n\n Average Convergence Factor = %f", conv_factor);
      nalu_hypre_printf("\n\n     Complexity:    grid = %f\n", grid_cmplxty);
      nalu_hypre_printf("                operator = %f\n", operat_cmplxty);
      nalu_hypre_printf("                   cycle = %f\n\n", cycle_cmplxty);
   }

   /*----------------------------------------------------------
    * Close the output file (if open)
    *----------------------------------------------------------*/

   /*if (my_id == 0 && amg_print_level >= 1)
   {
      fclose(fp);
   }*/

   nalu_hypre_TFree(num_coeffs, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(num_variables, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return (Solve_err_flag);
}

/******************************************************************************
 *
 * ParAMG cycling routine
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGCycleT
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGCycleT( void              *amg_vdata,
                       nalu_hypre_ParVector  **F_array,
                       nalu_hypre_ParVector  **U_array   )
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */

   nalu_hypre_ParCSRMatrix    **A_array;
   nalu_hypre_ParCSRMatrix    **P_array;
   nalu_hypre_ParCSRMatrix    **R_array;
   nalu_hypre_ParVector    *Vtemp;

   nalu_hypre_IntArray   **CF_marker_array;
   NALU_HYPRE_Int         *CF_marker;
   /* NALU_HYPRE_Int     **unknown_map_array; */
   /* NALU_HYPRE_Int     **point_map_array; */
   /* NALU_HYPRE_Int     **v_at_point_array; */

   NALU_HYPRE_Real    cycle_op_count;
   NALU_HYPRE_Int       cycle_type;
   NALU_HYPRE_Int       num_levels;
   NALU_HYPRE_Int       max_levels;

   NALU_HYPRE_Real   *num_coeffs;
   NALU_HYPRE_Int      *num_grid_sweeps;
   NALU_HYPRE_Int      *grid_relax_type;
   NALU_HYPRE_Int     **grid_relax_points;

   /* Local variables  */

   NALU_HYPRE_Int      *lev_counter;
   NALU_HYPRE_Int       Solve_err_flag;
   NALU_HYPRE_Int       k;
   NALU_HYPRE_Int       j;
   NALU_HYPRE_Int       level;
   NALU_HYPRE_Int       cycle_param;
   NALU_HYPRE_Int       coarse_grid;
   NALU_HYPRE_Int       fine_grid;
   NALU_HYPRE_Int       Not_Finished;
   NALU_HYPRE_Int       num_sweep;
   NALU_HYPRE_Int       relax_type;
   NALU_HYPRE_Int       relax_points;
   NALU_HYPRE_Real   *relax_weight;

   NALU_HYPRE_Int       old_version = 0;


   NALU_HYPRE_Real    alpha;
   NALU_HYPRE_Real    beta;
#if 0
   NALU_HYPRE_Real   *D_mat;
   NALU_HYPRE_Real   *S_vec;
#endif

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Acquire data and allocate storage */

   A_array           = nalu_hypre_ParAMGDataAArray(amg_data);
   P_array           = nalu_hypre_ParAMGDataPArray(amg_data);
   R_array           = nalu_hypre_ParAMGDataRArray(amg_data);
   CF_marker_array   = nalu_hypre_ParAMGDataCFMarkerArray(amg_data);
   /* unknown_map_array = nalu_hypre_ParAMGDataUnknownMapArray(amg_data); */
   /* point_map_array   = nalu_hypre_ParAMGDataPointMapArray(amg_data); */
   /* v_at_point_array  = nalu_hypre_ParAMGDataVatPointArray(amg_data); */
   Vtemp             = nalu_hypre_ParAMGDataVtemp(amg_data);
   num_levels        = nalu_hypre_ParAMGDataNumLevels(amg_data);
   max_levels        = nalu_hypre_ParAMGDataMaxLevels(amg_data);
   cycle_type        = nalu_hypre_ParAMGDataCycleType(amg_data);
   /* num_unknowns      =  nalu_hypre_ParCSRMatrixNumRows(A_array[0]); */

   num_grid_sweeps     = nalu_hypre_ParAMGDataNumGridSweeps(amg_data);
   grid_relax_type     = nalu_hypre_ParAMGDataGridRelaxType(amg_data);
   grid_relax_points   = nalu_hypre_ParAMGDataGridRelaxPoints(amg_data);
   relax_weight        = nalu_hypre_ParAMGDataRelaxWeight(amg_data);

   cycle_op_count = nalu_hypre_ParAMGDataCycleOpCount(amg_data);

   lev_counter = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_levels, NALU_HYPRE_MEMORY_HOST);

   /* Initialize */

   Solve_err_flag = 0;

   if (grid_relax_points) { old_version = 1; }

   num_coeffs = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_levels, NALU_HYPRE_MEMORY_HOST);
   num_coeffs[0]    = nalu_hypre_ParCSRMatrixDNumNonzeros(A_array[0]);

   for (j = 1; j < num_levels; j++)
   {
      num_coeffs[j] = nalu_hypre_ParCSRMatrixDNumNonzeros(A_array[j]);
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
      lev_counter[k] = cycle_type;
   }

   level = 0;
   cycle_param = 0;

   /*---------------------------------------------------------------------
    * Main loop of cycling
    *--------------------------------------------------------------------*/

   NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);
   while (Not_Finished)
   {
      num_sweep = num_grid_sweeps[cycle_param];
      relax_type = grid_relax_type[cycle_param];
      if (relax_type != 7 && relax_type != 9) { relax_type = 7; }
      /*------------------------------------------------------------------
       * Do the relaxation num_sweep times
       *-----------------------------------------------------------------*/

      for (j = 0; j < num_sweep; j++)
      {

         if (num_levels == 1 && max_levels > 1)
         {
            relax_points = 0;
         }
         else
         {
            if (old_version)
            {
               relax_points = grid_relax_points[cycle_param][j];
            }
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

         /* note: this does not use relax_points, so it doesn't matter if
            its the "old version" */

         if (CF_marker_array[level] == NULL)
         {
            CF_marker = NULL;
         }
         else
         {
            CF_marker = nalu_hypre_IntArrayData(CF_marker_array[level]);
         }
         Solve_err_flag = nalu_hypre_BoomerAMGRelaxT(A_array[level],
                                                F_array[level],
                                                CF_marker,
                                                relax_type,
                                                relax_points,
                                                relax_weight[level],
                                                U_array[level],
                                                Vtemp);


         if (Solve_err_flag != 0)
         {
            nalu_hypre_TFree(lev_counter, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(num_coeffs, NALU_HYPRE_MEMORY_HOST);
            NALU_HYPRE_ANNOTATE_MGLEVEL_END(level);
            NALU_HYPRE_ANNOTATE_FUNC_END;

            return (Solve_err_flag);
         }
      }


      /*------------------------------------------------------------------
       * Decrement the control counter and determine which grid to visit next
       *-----------------------------------------------------------------*/

      --lev_counter[level];

      if (lev_counter[level] >= 0 && level != num_levels - 1)
      {

         /*---------------------------------------------------------------
          * Visit coarser level next.  Compute residual using nalu_hypre_ParCSRMatrixMatvec.
          * Use interpolation (since transpose i.e. P^TATR instead of
          * RAP) using nalu_hypre_ParCSRMatrixMatvecT.
          * Reset counters and cycling parameters for coarse level
          *--------------------------------------------------------------*/

         fine_grid = level;
         coarse_grid = level + 1;

         nalu_hypre_ParVectorSetConstantValues(U_array[coarse_grid], 0.0);

         nalu_hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
         alpha = -1.0;
         beta = 1.0;
         nalu_hypre_ParCSRMatrixMatvecT(alpha, A_array[fine_grid], U_array[fine_grid],
                                   beta, Vtemp);

         alpha = 1.0;
         beta = 0.0;

         nalu_hypre_ParCSRMatrixMatvecT(alpha, P_array[fine_grid], Vtemp,
                                   beta, F_array[coarse_grid]);

         NALU_HYPRE_ANNOTATE_MGLEVEL_END(level);

         ++level;
         lev_counter[level] = nalu_hypre_max(lev_counter[level], cycle_type);
         cycle_param = 1;
         if (level == num_levels - 1) { cycle_param = 3; }

         NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);
      }

      else if (level != 0)
      {

         /*---------------------------------------------------------------
          * Visit finer level next.
          * Use restriction (since transpose i.e. P^TA^TR instead of RAP)
          * and add correction using nalu_hypre_ParCSRMatrixMatvec.
          * Reset counters and cycling parameters for finer level.
          *--------------------------------------------------------------*/

         fine_grid = level - 1;
         coarse_grid = level;
         alpha = 1.0;
         beta = 1.0;

         nalu_hypre_ParCSRMatrixMatvec(alpha, R_array[fine_grid], U_array[coarse_grid],
                                  beta, U_array[fine_grid]);

         NALU_HYPRE_ANNOTATE_MGLEVEL_END(level);

         --level;
         cycle_param = 2;
         if (level == 0) { cycle_param = 0; }

         NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);
      }
      else
      {
         Not_Finished = 0;
      }
   }

   NALU_HYPRE_ANNOTATE_MGLEVEL_END(level);

   nalu_hypre_ParAMGDataCycleOpCount(amg_data) = cycle_op_count;
   nalu_hypre_TFree(lev_counter, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(num_coeffs, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return (Solve_err_flag);
}

/******************************************************************************
 *
 * Relaxation scheme
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGRelaxT
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int  nalu_hypre_BoomerAMGRelaxT( nalu_hypre_ParCSRMatrix *A,
                                  nalu_hypre_ParVector    *f,
                                  NALU_HYPRE_Int                *cf_marker,
                                  NALU_HYPRE_Int                 relax_type,
                                  NALU_HYPRE_Int                 relax_points,
                                  NALU_HYPRE_Real          relax_weight,
                                  nalu_hypre_ParVector    *u,
                                  nalu_hypre_ParVector    *Vtemp )
{
   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data  = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i     = nalu_hypre_CSRMatrixI(A_diag);

   NALU_HYPRE_BigInt     global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   NALU_HYPRE_Int        n       = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_BigInt     first_index = nalu_hypre_ParVectorFirstIndex(u);

   nalu_hypre_Vector   *u_local = nalu_hypre_ParVectorLocalVector(u);
   NALU_HYPRE_Real     *u_data  = nalu_hypre_VectorData(u_local);

   nalu_hypre_Vector   *Vtemp_local = nalu_hypre_ParVectorLocalVector(Vtemp);
   NALU_HYPRE_Real     *Vtemp_data = nalu_hypre_VectorData(Vtemp_local);

   nalu_hypre_CSRMatrix *A_CSR;
   NALU_HYPRE_Int      *A_CSR_i;
   NALU_HYPRE_Int      *A_CSR_j;
   NALU_HYPRE_Real     *A_CSR_data;

   nalu_hypre_Vector    *f_vector;
   NALU_HYPRE_Real     *f_vector_data;

   NALU_HYPRE_Int        i;
   NALU_HYPRE_Int        jj;
   NALU_HYPRE_Int        column;
   NALU_HYPRE_Int        relax_error = 0;

   NALU_HYPRE_Real      *A_mat;
   NALU_HYPRE_Real      *b_vec;

   NALU_HYPRE_Real       zero = 0.0;

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   /*-----------------------------------------------------------------------
    * Switch statement to direct control based on relax_type:
    *     relax_type = 7 -> Jacobi (uses ParMatvec)
    *     relax_type = 9 -> Direct Solve
    *-----------------------------------------------------------------------*/

   switch (relax_type)
   {

      case 7: /* Jacobi (uses ParMatvec) */
      {

         /*-----------------------------------------------------------------
          * Copy f into temporary vector.
          *-----------------------------------------------------------------*/

         nalu_hypre_ParVectorCopy(f, Vtemp);

         /*-----------------------------------------------------------------
          * Perform MatvecT Vtemp=f-A^Tu
          *-----------------------------------------------------------------*/

         nalu_hypre_ParCSRMatrixMatvecT(-1.0, A, u, 1.0, Vtemp);
         for (i = 0; i < n; i++)
         {

            /*-----------------------------------------------------------
             * If diagonal is nonzero, relax point i; otherwise, skip it.
             *-----------------------------------------------------------*/

            if (A_diag_data[A_diag_i[i]] != zero)
            {
               u_data[i] += relax_weight * Vtemp_data[i]
                            / A_diag_data[A_diag_i[i]];
            }
         }
      }
      break;


      case 9: /* Direct solve: use gaussian elimination */
      {

         NALU_HYPRE_Int n_global = (NALU_HYPRE_Int) global_num_rows;
         /*-----------------------------------------------------------------
          *  Generate CSR matrix from ParCSRMatrix A
          *-----------------------------------------------------------------*/

         A_CSR = nalu_hypre_ParCSRMatrixToCSRMatrixAll(A);
         f_vector = nalu_hypre_ParVectorToVectorAll(f);
         if (n)
         {
            A_CSR_i = nalu_hypre_CSRMatrixI(A_CSR);
            A_CSR_j = nalu_hypre_CSRMatrixJ(A_CSR);
            A_CSR_data = nalu_hypre_CSRMatrixData(A_CSR);
            f_vector_data = nalu_hypre_VectorData(f_vector);

            A_mat = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  n_global * n_global, NALU_HYPRE_MEMORY_HOST);
            b_vec = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  n_global, NALU_HYPRE_MEMORY_HOST);

            /*---------------------------------------------------------------
             *  Load transpose of CSR matrix into A_mat.
             *---------------------------------------------------------------*/

            for (i = 0; i < n_global; i++)
            {
               for (jj = A_CSR_i[i]; jj < A_CSR_i[i + 1]; jj++)
               {
                  column = A_CSR_j[jj];
                  A_mat[column * n_global + i] = A_CSR_data[jj];
               }
               b_vec[i] = f_vector_data[i];
            }

            nalu_hypre_gselim(A_mat, b_vec, n_global, relax_error);

            for (i = 0; i < n; i++)
            {
               u_data[i] = b_vec[first_index + i];
            }

            nalu_hypre_TFree(A_mat, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(b_vec, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            nalu_hypre_SeqVectorDestroy(f_vector);
            f_vector = NULL;

         }
      }
      break;
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return (relax_error);
}
