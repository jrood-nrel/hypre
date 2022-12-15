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

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGCycle
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGAdditiveCycle( void              *amg_vdata)
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */

   nalu_hypre_ParCSRMatrix    **A_array;
   nalu_hypre_ParCSRMatrix    **P_array;
   nalu_hypre_ParCSRMatrix    **R_array;
   nalu_hypre_ParCSRMatrix    *Lambda;
   nalu_hypre_ParCSRMatrix    *Atilde;
   nalu_hypre_ParVector    **F_array;
   nalu_hypre_ParVector    **U_array;
   nalu_hypre_ParVector    *Vtemp;
   nalu_hypre_ParVector    *Ztemp;
   nalu_hypre_ParVector    *Xtilde, *Rtilde;
   nalu_hypre_IntArray    **CF_marker_array;
   NALU_HYPRE_Int          *CF_marker;

   NALU_HYPRE_Int       num_levels;
   NALU_HYPRE_Int       addlvl, add_end;
   NALU_HYPRE_Int       additive;
   NALU_HYPRE_Int       mult_additive;
   NALU_HYPRE_Int       simple;
   NALU_HYPRE_Int       add_last_lvl;
   NALU_HYPRE_Int       i, j, num_rows;
   NALU_HYPRE_Int       n_global;
   NALU_HYPRE_Int       rlx_order;

   /* Local variables  */
   NALU_HYPRE_Int       Solve_err_flag = 0;
   NALU_HYPRE_Int       level;
   NALU_HYPRE_Int       coarse_grid;
   NALU_HYPRE_Int       fine_grid;
   NALU_HYPRE_Int       rlx_down;
   NALU_HYPRE_Int       rlx_up;
   NALU_HYPRE_Int       rlx_coarse;
   NALU_HYPRE_Int      *grid_relax_type;
   NALU_HYPRE_Int      *num_grid_sweeps;
   nalu_hypre_Vector  **l1_norms;
   NALU_HYPRE_Real      alpha, beta;
   NALU_HYPRE_Real     *u_data;
   NALU_HYPRE_Real     *v_data;
   nalu_hypre_Vector   *l1_norms_lvl;
   NALU_HYPRE_Real     *D_inv;
   NALU_HYPRE_Real     *x_global;
   NALU_HYPRE_Real     *r_global;
   NALU_HYPRE_Real     *relax_weight;
   NALU_HYPRE_Real     *omega;

#if 0
   NALU_HYPRE_Real   *D_mat;
   NALU_HYPRE_Real   *S_vec;
#endif

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Acquire data and allocate storage */

   A_array           = nalu_hypre_ParAMGDataAArray(amg_data);
   F_array           = nalu_hypre_ParAMGDataFArray(amg_data);
   U_array           = nalu_hypre_ParAMGDataUArray(amg_data);
   P_array           = nalu_hypre_ParAMGDataPArray(amg_data);
   R_array           = nalu_hypre_ParAMGDataRArray(amg_data);
   CF_marker_array   = nalu_hypre_ParAMGDataCFMarkerArray(amg_data);
   Vtemp             = nalu_hypre_ParAMGDataVtemp(amg_data);
   Ztemp             = nalu_hypre_ParAMGDataZtemp(amg_data);
   num_levels        = nalu_hypre_ParAMGDataNumLevels(amg_data);
   additive          = nalu_hypre_ParAMGDataAdditive(amg_data);
   mult_additive     = nalu_hypre_ParAMGDataMultAdditive(amg_data);
   simple            = nalu_hypre_ParAMGDataSimple(amg_data);
   add_last_lvl      = nalu_hypre_ParAMGDataAddLastLvl(amg_data);
   grid_relax_type   = nalu_hypre_ParAMGDataGridRelaxType(amg_data);
   Lambda            = nalu_hypre_ParAMGDataLambda(amg_data);
   Atilde            = nalu_hypre_ParAMGDataAtilde(amg_data);
   Xtilde            = nalu_hypre_ParAMGDataXtilde(amg_data);
   Rtilde            = nalu_hypre_ParAMGDataRtilde(amg_data);
   l1_norms          = nalu_hypre_ParAMGDataL1Norms(amg_data);
   D_inv             = nalu_hypre_ParAMGDataDinv(amg_data);
   relax_weight      = nalu_hypre_ParAMGDataRelaxWeight(amg_data);
   omega             = nalu_hypre_ParAMGDataOmega(amg_data);
   rlx_order         = nalu_hypre_ParAMGDataRelaxOrder(amg_data);
   num_grid_sweeps   = nalu_hypre_ParAMGDataNumGridSweeps(amg_data);

   /* Initialize */

   addlvl = nalu_hypre_max(additive, mult_additive);
   addlvl = nalu_hypre_max(addlvl, simple);
   if (add_last_lvl == -1 ) { add_end = num_levels - 1; }
   else { add_end = add_last_lvl; }
   Solve_err_flag = 0;

   /*---------------------------------------------------------------------
    * Main loop of cycling --- multiplicative version --- V-cycle
    *--------------------------------------------------------------------*/

   /* down cycle */
   rlx_down = grid_relax_type[1];
   rlx_up = grid_relax_type[2];
   rlx_coarse = grid_relax_type[3];
   for (level = 0; level < num_levels - 1; level++)
   {
      NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);

      fine_grid = level;
      coarse_grid = level + 1;

      u_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(U_array[fine_grid]));
      v_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Vtemp));
      l1_norms_lvl = l1_norms[level];

      nalu_hypre_ParVectorSetConstantValues(U_array[coarse_grid], 0.0);

      if (level < addlvl || level > add_end) /* multiplicative version */
      {
         /* smoothing step */

         if (rlx_down == 0)
         {
            NALU_HYPRE_Real *A_data = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(A_array[fine_grid]));
            NALU_HYPRE_Int *A_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(A_array[fine_grid]));
            num_rows = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A_array[fine_grid]));
            for (j = 0; j < num_grid_sweeps[1]; j++)
            {
               nalu_hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
#ifdef NALU_HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  u_data[i] = relax_weight[level] * v_data[i] / A_data[A_i[i]];
               }
            }
         }

         else if (rlx_down != 18)
         {
            /*nalu_hypre_BoomerAMGRelax(A_array[fine_grid],F_array[fine_grid],NULL,rlx_down,0,*/
            CF_marker = nalu_hypre_IntArrayData(CF_marker_array[fine_grid]);
            for (j = 0; j < num_grid_sweeps[1]; j++)
            {
               nalu_hypre_BoomerAMGRelaxIF(A_array[fine_grid], F_array[fine_grid],
                                      CF_marker, rlx_down, rlx_order, 1,
                                      relax_weight[fine_grid], omega[fine_grid],
                                      l1_norms[level] ? nalu_hypre_VectorData(l1_norms[level]) : NULL,
                                      U_array[fine_grid], Vtemp, Ztemp);
               nalu_hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
            }
         }
         else
         {
            num_rows = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A_array[fine_grid]));
            for (j = 0; j < num_grid_sweeps[1]; j++)
            {
               nalu_hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
#ifdef NALU_HYPRE_USING_OPENMP
               #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  u_data[i] += v_data[i] / nalu_hypre_VectorData(l1_norms_lvl)[i];
               }
            }
         }

         alpha = -1.0;
         beta = 1.0;
         nalu_hypre_ParCSRMatrixMatvec(alpha, A_array[fine_grid], U_array[fine_grid],
                                  beta, Vtemp);

         alpha = 1.0;
         beta = 0.0;
         nalu_hypre_ParCSRMatrixMatvecT(alpha, R_array[fine_grid], Vtemp,
                                   beta, F_array[coarse_grid]);
      }
      else /* additive version */
      {
         nalu_hypre_ParVectorCopy(F_array[fine_grid], Vtemp);
         if (level == 0) /* compute residual */
         {
            nalu_hypre_ParVectorCopy(Vtemp, Rtilde);
            nalu_hypre_ParVectorCopy(U_array[fine_grid], Xtilde);
         }
         alpha = 1.0;
         beta = 0.0;
         nalu_hypre_ParCSRMatrixMatvecT(alpha, R_array[fine_grid], Vtemp,
                                   beta, F_array[coarse_grid]);
      }

      NALU_HYPRE_ANNOTATE_MGLEVEL_END(level);
   }

   /* additive smoothing and solve coarse grid */
   NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(num_levels - 1);
   if (addlvl < num_levels)
   {
      if (simple > -1)
      {
         x_global = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Xtilde));
         r_global = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Rtilde));
         n_global = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(Xtilde));
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < n_global; i++)
         {
            x_global[i] += D_inv[i] * r_global[i];
         }
      }
      else
      {
         if (num_grid_sweeps[1] > 1)
         {
            n_global = nalu_hypre_VectorSize(nalu_hypre_ParVectorLocalVector(Rtilde));
            nalu_hypre_ParVector *Tmptilde = nalu_hypre_CTAlloc(nalu_hypre_ParVector,  1, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_Vector *Tmptilde_local = nalu_hypre_SeqVectorCreate(n_global);
            nalu_hypre_SeqVectorInitialize(Tmptilde_local);
            nalu_hypre_ParVectorLocalVector(Tmptilde) = Tmptilde_local;
            nalu_hypre_ParVectorOwnsData(Tmptilde) = 1;
            nalu_hypre_ParCSRMatrixMatvec(1.0, Lambda, Rtilde, 0.0, Tmptilde);
            nalu_hypre_ParVectorScale(2.0, Rtilde);
            nalu_hypre_ParCSRMatrixMatvec(-1.0, Atilde, Tmptilde, 1.0, Rtilde);
            nalu_hypre_ParVectorDestroy(Tmptilde);
         }
         nalu_hypre_ParCSRMatrixMatvec(1.0, Lambda, Rtilde, 1.0, Xtilde);
      }
      if (addlvl == 0) { nalu_hypre_ParVectorCopy(Xtilde, U_array[0]); }
   }
   if (add_end < num_levels - 1)
   {
      fine_grid = num_levels - 1;
      for (j = 0; j < num_grid_sweeps[3]; j++)
         if (rlx_coarse == 18)
            nalu_hypre_ParCSRRelax(A_array[fine_grid], F_array[fine_grid],
                              1, 1,
                              l1_norms[fine_grid] ? nalu_hypre_VectorData(l1_norms[fine_grid]) : NULL,
                              1.0, 1.0, 0.0, 0.0, 0, 0.0,
                              U_array[fine_grid], Vtemp, Ztemp);
         else
            nalu_hypre_BoomerAMGRelaxIF(A_array[fine_grid], F_array[fine_grid],
                                   NULL, rlx_coarse, 0, 0,
                                   relax_weight[fine_grid], omega[fine_grid],
                                   l1_norms[fine_grid] ? nalu_hypre_VectorData(l1_norms[fine_grid]) : NULL,
                                   U_array[fine_grid], Vtemp, Ztemp);
   }
   NALU_HYPRE_ANNOTATE_MGLEVEL_END(num_levels - 1);

   /* up cycle */
   for (level = num_levels - 1; level > 0; level--)
   {
      NALU_HYPRE_ANNOTATE_MGLEVEL_BEGIN(level);

      fine_grid = level - 1;
      coarse_grid = level;

      if (level <= addlvl || level > add_end + 1) /* multiplicative version */
      {
         alpha = 1.0;
         beta = 1.0;
         nalu_hypre_ParCSRMatrixMatvec(alpha, P_array[fine_grid],
                                  U_array[coarse_grid],
                                  beta, U_array[fine_grid]);
         if (rlx_up != 18)
         {
            /*nalu_hypre_BoomerAMGRelax(A_array[fine_grid],F_array[fine_grid],NULL,rlx_up,0,*/
            CF_marker = nalu_hypre_IntArrayData(CF_marker_array[fine_grid]);
            for (j = 0; j < num_grid_sweeps[2]; j++)
            {
               nalu_hypre_BoomerAMGRelaxIF(A_array[fine_grid], F_array[fine_grid],
                                      CF_marker,
                                      rlx_up, rlx_order, 2,
                                      relax_weight[fine_grid], omega[fine_grid],
                                      l1_norms[fine_grid] ? nalu_hypre_VectorData(l1_norms[fine_grid]) : NULL,
                                      U_array[fine_grid], Vtemp, Ztemp);
            }
         }
         else if (rlx_order)
         {
            CF_marker = nalu_hypre_IntArrayData(CF_marker_array[fine_grid]);
            NALU_HYPRE_Int loc_relax_points[2];
            loc_relax_points[0] = -1;
            loc_relax_points[1] = 1;
            for (j = 0; j < num_grid_sweeps[2]; j++)
            {
               for (i = 0; i < 2; i++)
               {
                  nalu_hypre_ParCSRRelax_L1_Jacobi(A_array[fine_grid], F_array[fine_grid],
                                              CF_marker,
                                              loc_relax_points[i],
                                              1.0,
                                              l1_norms[fine_grid] ? nalu_hypre_VectorData(l1_norms[fine_grid]) : NULL,
                                              U_array[fine_grid], Vtemp);
               }
            }
         }
         else
            for (j = 0; j < num_grid_sweeps[2]; j++)
            {
               nalu_hypre_ParCSRRelax(A_array[fine_grid], F_array[fine_grid],
                                 1, 1,
                                 l1_norms[fine_grid] ? nalu_hypre_VectorData(l1_norms[fine_grid]) : NULL,
                                 1.0, 1.0, 0.0, 0.0, 0, 0.0,
                                 U_array[fine_grid], Vtemp, Ztemp);
            }
      }
      else /* additive version */
      {
         alpha = 1.0;
         beta = 1.0;
         nalu_hypre_ParCSRMatrixMatvec(alpha, P_array[fine_grid],
                                  U_array[coarse_grid],
                                  beta, U_array[fine_grid]);
      }

      NALU_HYPRE_ANNOTATE_MGLEVEL_END(level);
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return (Solve_err_flag);
}


NALU_HYPRE_Int nalu_hypre_CreateLambda(void *amg_vdata)
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */
   MPI_Comm comm;
   nalu_hypre_ParCSRMatrix **A_array;
   nalu_hypre_ParVector    **F_array;
   nalu_hypre_ParVector    **U_array;

   nalu_hypre_ParCSRMatrix *A_tmp;
   nalu_hypre_ParCSRMatrix *Lambda;
   nalu_hypre_CSRMatrix *L_diag;
   nalu_hypre_CSRMatrix *L_offd;
   nalu_hypre_ParCSRMatrix *Atilde;
   nalu_hypre_CSRMatrix *Atilde_diag;
   nalu_hypre_CSRMatrix *Atilde_offd;
   NALU_HYPRE_Real    *Atilde_diag_data;
   NALU_HYPRE_Real    *Atilde_offd_data;
   nalu_hypre_CSRMatrix *A_tmp_diag;
   nalu_hypre_CSRMatrix *A_tmp_offd;
   nalu_hypre_ParVector *Xtilde;
   nalu_hypre_ParVector *Rtilde;
   nalu_hypre_Vector *Xtilde_local;
   nalu_hypre_Vector *Rtilde_local;
   nalu_hypre_ParCSRCommPkg *comm_pkg;
   nalu_hypre_ParCSRCommPkg *L_comm_pkg = NULL;
   nalu_hypre_ParCSRCommHandle *comm_handle;
   NALU_HYPRE_Real    *L_diag_data;
   NALU_HYPRE_Real    *L_offd_data;
   NALU_HYPRE_Real    *buf_data = NULL;
   NALU_HYPRE_Real    *tmp_data;
   NALU_HYPRE_Real    *x_data;
   NALU_HYPRE_Real    *r_data;
   nalu_hypre_Vector  *l1_norms;
   NALU_HYPRE_Real    *A_tmp_diag_data;
   NALU_HYPRE_Real    *A_tmp_offd_data;
   NALU_HYPRE_Real    *D_data = NULL;
   NALU_HYPRE_Real    *D_data_offd = NULL;
   NALU_HYPRE_Int *L_diag_i;
   NALU_HYPRE_Int *L_diag_j;
   NALU_HYPRE_Int *L_offd_i;
   NALU_HYPRE_Int *L_offd_j;
   NALU_HYPRE_Int *Atilde_diag_i;
   NALU_HYPRE_Int *Atilde_diag_j;
   NALU_HYPRE_Int *Atilde_offd_i;
   NALU_HYPRE_Int *Atilde_offd_j;
   NALU_HYPRE_Int *A_tmp_diag_i;
   NALU_HYPRE_Int *A_tmp_offd_i;
   NALU_HYPRE_Int *A_tmp_diag_j;
   NALU_HYPRE_Int *A_tmp_offd_j;
   NALU_HYPRE_Int *L_recv_ptr = NULL;
   NALU_HYPRE_Int *L_send_ptr = NULL;
   NALU_HYPRE_Int *L_recv_procs = NULL;
   NALU_HYPRE_Int *L_send_procs = NULL;
   NALU_HYPRE_Int *L_send_map_elmts = NULL;
   NALU_HYPRE_Int *recv_procs;
   NALU_HYPRE_Int *send_procs;
   NALU_HYPRE_Int *send_map_elmts;
   NALU_HYPRE_Int *send_map_starts;
   NALU_HYPRE_Int *recv_vec_starts;
   NALU_HYPRE_Int *all_send_procs = NULL;
   NALU_HYPRE_Int *all_recv_procs = NULL;
   NALU_HYPRE_Int *remap = NULL;
   NALU_HYPRE_Int *level_start;

   NALU_HYPRE_Int       addlvl;
   NALU_HYPRE_Int       additive;
   NALU_HYPRE_Int       mult_additive;
   NALU_HYPRE_Int       num_levels;
   NALU_HYPRE_Int       num_add_lvls;
   NALU_HYPRE_Int       num_procs;
   NALU_HYPRE_Int       num_sends, num_recvs;
   NALU_HYPRE_Int       num_sends_L = 0;
   NALU_HYPRE_Int       num_recvs_L = 0;
   NALU_HYPRE_Int       send_data_L = 0;
   NALU_HYPRE_Int       num_rows_L = 0;
   NALU_HYPRE_Int       num_rows_tmp = 0;
   NALU_HYPRE_Int       num_cols_offd_L = 0;
   NALU_HYPRE_Int       num_cols_offd = 0;
   NALU_HYPRE_Int       level, i, j, k;
   NALU_HYPRE_Int       this_proc, cnt, cnt_diag, cnt_offd;
   NALU_HYPRE_Int       A_cnt_diag, A_cnt_offd;
   NALU_HYPRE_Int       cnt_recv, cnt_send, cnt_row, row_start;
   NALU_HYPRE_Int       start_diag, start_offd, indx, cnt_map;
   NALU_HYPRE_Int       start, j_indx, index, cnt_level;
   NALU_HYPRE_Int       max_sends, max_recvs;
   NALU_HYPRE_Int       ns;

   /* Local variables  */
   NALU_HYPRE_Int       Solve_err_flag = 0;
   NALU_HYPRE_Int       num_nonzeros_diag;
   NALU_HYPRE_Int       num_nonzeros_offd;

   nalu_hypre_Vector  **l1_norms_ptr = NULL;
   /*NALU_HYPRE_Real   *relax_weight = NULL;
   NALU_HYPRE_Int      relax_type; */
   NALU_HYPRE_Int       add_rlx;
   NALU_HYPRE_Int       add_last_lvl, add_end;
   NALU_HYPRE_Real  add_rlx_wt;

   /* Acquire data and allocate storage */

   A_array           = nalu_hypre_ParAMGDataAArray(amg_data);
   F_array           = nalu_hypre_ParAMGDataFArray(amg_data);
   U_array           = nalu_hypre_ParAMGDataUArray(amg_data);
   additive          = nalu_hypre_ParAMGDataAdditive(amg_data);
   mult_additive     = nalu_hypre_ParAMGDataMultAdditive(amg_data);
   add_last_lvl      = nalu_hypre_ParAMGDataAddLastLvl(amg_data);
   num_levels        = nalu_hypre_ParAMGDataNumLevels(amg_data);
   /*relax_weight      = nalu_hypre_ParAMGDataRelaxWeight(amg_data);
   relax_type        = nalu_hypre_ParAMGDataGridRelaxType(amg_data)[1];*/
   comm              = nalu_hypre_ParCSRMatrixComm(A_array[0]);
   add_rlx           = nalu_hypre_ParAMGDataAddRelaxType(amg_data);
   add_rlx_wt        = nalu_hypre_ParAMGDataAddRelaxWt(amg_data);
   ns                = nalu_hypre_ParAMGDataNumGridSweeps(amg_data)[1];

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   l1_norms_ptr      = nalu_hypre_ParAMGDataL1Norms(amg_data);

   addlvl = nalu_hypre_max(additive, mult_additive);
   if (add_last_lvl != -1) { add_end = add_last_lvl + 1; }
   else { add_end = num_levels; }
   num_add_lvls = add_end + 1 - addlvl;

   level_start = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_add_lvls + 1, NALU_HYPRE_MEMORY_HOST);
   send_data_L = 0;
   num_rows_L  = 0;
   num_cols_offd_L = 0;
   num_nonzeros_diag = 0;
   num_nonzeros_offd = 0;
   level_start[0] = 0;
   cnt = 1;
   max_sends = 0;
   max_recvs = 0;
   for (i = addlvl; i < add_end; i++)
   {
      A_tmp = A_array[i];
      A_tmp_diag = nalu_hypre_ParCSRMatrixDiag(A_tmp);
      A_tmp_offd = nalu_hypre_ParCSRMatrixOffd(A_tmp);
      A_tmp_diag_i = nalu_hypre_CSRMatrixI(A_tmp_diag);
      A_tmp_offd_i = nalu_hypre_CSRMatrixI(A_tmp_offd);
      num_rows_tmp = nalu_hypre_CSRMatrixNumRows(A_tmp_diag);
      num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_tmp_offd);
      num_rows_L += num_rows_tmp;
      level_start[cnt] = level_start[cnt - 1] + num_rows_tmp;
      cnt++;
      num_cols_offd_L += num_cols_offd;
      num_nonzeros_diag += A_tmp_diag_i[num_rows_tmp];
      num_nonzeros_offd += A_tmp_offd_i[num_rows_tmp];
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A_tmp);
      if (comm_pkg)
      {
         num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
         max_sends += num_sends;
         if (num_sends)
         {
            send_data_L += nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
         }
         max_recvs += nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      }
   }
   if (max_sends >= num_procs || max_recvs >= num_procs)
   {
      max_sends = num_procs;
      max_recvs = num_procs;
   }
   if (max_sends) { all_send_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_sends, NALU_HYPRE_MEMORY_HOST); }
   if (max_recvs) { all_recv_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_recvs, NALU_HYPRE_MEMORY_HOST); }

   cnt_send = 0;
   cnt_recv = 0;
   if (max_sends || max_recvs)
   {
      if (max_sends < num_procs && max_recvs < num_procs)
      {
         for (i = addlvl; i < add_end; i++)
         {
            A_tmp = A_array[i];
            comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A_tmp);
            if (comm_pkg)
            {
               num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
               num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
               send_procs = nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg);
               recv_procs = nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg);
               for (j = 0; j < num_sends; j++)
               {
                  all_send_procs[cnt_send++] = send_procs[j];
               }
               for (j = 0; j < num_recvs; j++)
               {
                  all_recv_procs[cnt_recv++] = recv_procs[j];
               }
            }
         }
         if (max_sends)
         {
            nalu_hypre_qsort0(all_send_procs, 0, max_sends - 1);
            num_sends_L = 1;
            this_proc = all_send_procs[0];
            for (i = 1; i < max_sends; i++)
            {
               if (all_send_procs[i] > this_proc)
               {
                  this_proc = all_send_procs[i];
                  all_send_procs[num_sends_L++] = this_proc;
               }
            }
            L_send_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends_L, NALU_HYPRE_MEMORY_HOST);
            for (j = 0; j < num_sends_L; j++)
            {
               L_send_procs[j] = all_send_procs[j];
            }
            nalu_hypre_TFree(all_send_procs, NALU_HYPRE_MEMORY_HOST);
         }
         if (max_recvs)
         {
            nalu_hypre_qsort0(all_recv_procs, 0, max_recvs - 1);
            num_recvs_L = 1;
            this_proc = all_recv_procs[0];
            for (i = 1; i < max_recvs; i++)
            {
               if (all_recv_procs[i] > this_proc)
               {
                  this_proc = all_recv_procs[i];
                  all_recv_procs[num_recvs_L++] = this_proc;
               }
            }
            L_recv_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs_L, NALU_HYPRE_MEMORY_HOST);
            for (j = 0; j < num_recvs_L; j++)
            {
               L_recv_procs[j] = all_recv_procs[j];
            }
            nalu_hypre_TFree(all_recv_procs, NALU_HYPRE_MEMORY_HOST);
         }

         L_recv_ptr = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs_L + 1, NALU_HYPRE_MEMORY_HOST);
         L_send_ptr = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends_L + 1, NALU_HYPRE_MEMORY_HOST);

         for (i = addlvl; i < add_end; i++)
         {
            A_tmp = A_array[i];
            comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A_tmp);
            if (comm_pkg)
            {
               num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
               num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
               send_procs = nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg);
               recv_procs = nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg);
               send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
               recv_vec_starts = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
            }
            else
            {
               num_sends = 0;
               num_recvs = 0;
            }
            for (k = 0; k < num_sends; k++)
            {
               this_proc = nalu_hypre_BinarySearch(L_send_procs, send_procs[k], num_sends_L);
               L_send_ptr[this_proc + 1] += send_map_starts[k + 1] - send_map_starts[k];
            }
            for (k = 0; k < num_recvs; k++)
            {
               this_proc = nalu_hypre_BinarySearch(L_recv_procs, recv_procs[k], num_recvs_L);
               L_recv_ptr[this_proc + 1] += recv_vec_starts[k + 1] - recv_vec_starts[k];
            }
         }

         L_recv_ptr[0] = 0;
         for (i = 1; i < num_recvs_L; i++)
         {
            L_recv_ptr[i + 1] += L_recv_ptr[i];
         }

         L_send_ptr[0] = 0;
         for (i = 1; i < num_sends_L; i++)
         {
            L_send_ptr[i + 1] += L_send_ptr[i];
         }
      }
      else
      {
         num_recvs_L = 0;
         num_sends_L = 0;
         for (i = addlvl; i < add_end; i++)
         {
            A_tmp = A_array[i];
            comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A_tmp);
            if (comm_pkg)
            {
               num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
               num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
               send_procs = nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg);
               recv_procs = nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg);
               send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
               recv_vec_starts = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
               for (j = 0; j < num_sends; j++)
               {
                  this_proc = send_procs[j];
                  if (all_send_procs[this_proc] == 0)
                  {
                     num_sends_L++;
                  }
                  all_send_procs[this_proc] += send_map_starts[j + 1] - send_map_starts[j];
               }
               for (j = 0; j < num_recvs; j++)
               {
                  this_proc = recv_procs[j];
                  if (all_recv_procs[this_proc] == 0)
                  {
                     num_recvs_L++;
                  }
                  all_recv_procs[this_proc] += recv_vec_starts[j + 1] - recv_vec_starts[j];
               }
            }
         }
         if (max_sends)
         {
            L_send_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends_L, NALU_HYPRE_MEMORY_HOST);
            L_send_ptr = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends_L + 1, NALU_HYPRE_MEMORY_HOST);
            num_sends_L = 0;
            for (j = 0; j < num_procs; j++)
            {
               this_proc = all_send_procs[j];
               if (this_proc)
               {
                  L_send_procs[num_sends_L++] = j;
                  L_send_ptr[num_sends_L] = this_proc + L_send_ptr[num_sends_L - 1];
               }
            }
         }
         if (max_recvs)
         {
            L_recv_procs = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs_L, NALU_HYPRE_MEMORY_HOST);
            L_recv_ptr = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs_L + 1, NALU_HYPRE_MEMORY_HOST);
            num_recvs_L = 0;
            for (j = 0; j < num_procs; j++)
            {
               this_proc = all_recv_procs[j];
               if (this_proc)
               {
                  L_recv_procs[num_recvs_L++] = j;
                  L_recv_ptr[num_recvs_L] = this_proc + L_recv_ptr[num_recvs_L - 1];
               }
            }
         }
      }
   }
   if (max_sends) { nalu_hypre_TFree(all_send_procs, NALU_HYPRE_MEMORY_HOST); }
   if (max_recvs) { nalu_hypre_TFree(all_recv_procs, NALU_HYPRE_MEMORY_HOST); }

   L_diag = nalu_hypre_CSRMatrixCreate(num_rows_L, num_rows_L, num_nonzeros_diag);
   L_offd = nalu_hypre_CSRMatrixCreate(num_rows_L, num_cols_offd_L, num_nonzeros_offd);
   nalu_hypre_CSRMatrixInitialize(L_diag);
   nalu_hypre_CSRMatrixInitialize(L_offd);

   if (num_nonzeros_diag)
   {
      L_diag_data = nalu_hypre_CSRMatrixData(L_diag);
      L_diag_j = nalu_hypre_CSRMatrixJ(L_diag);
   }
   L_diag_i = nalu_hypre_CSRMatrixI(L_diag);
   if (num_nonzeros_offd)
   {
      L_offd_data = nalu_hypre_CSRMatrixData(L_offd);
      L_offd_j = nalu_hypre_CSRMatrixJ(L_offd);
   }
   L_offd_i = nalu_hypre_CSRMatrixI(L_offd);

   if (ns > 1)
   {
      Atilde_diag = nalu_hypre_CSRMatrixCreate(num_rows_L, num_rows_L, num_nonzeros_diag);
      Atilde_offd = nalu_hypre_CSRMatrixCreate(num_rows_L, num_cols_offd_L, num_nonzeros_offd);
      nalu_hypre_CSRMatrixInitialize(Atilde_diag);
      nalu_hypre_CSRMatrixInitialize(Atilde_offd);
      if (num_nonzeros_diag)
      {
         Atilde_diag_data = nalu_hypre_CSRMatrixData(Atilde_diag);
         Atilde_diag_j = nalu_hypre_CSRMatrixJ(Atilde_diag);
      }
      Atilde_diag_i = nalu_hypre_CSRMatrixI(Atilde_diag);
      if (num_nonzeros_offd)
      {
         Atilde_offd_data = nalu_hypre_CSRMatrixData(Atilde_offd);
         Atilde_offd_j = nalu_hypre_CSRMatrixJ(Atilde_offd);
      }
      Atilde_offd_i = nalu_hypre_CSRMatrixI(Atilde_offd);
   }

   if (num_rows_L) { D_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_rows_L, NALU_HYPRE_MEMORY_HOST); }
   if (send_data_L)
   {
      L_send_map_elmts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  send_data_L, NALU_HYPRE_MEMORY_HOST);
      buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, send_data_L, NALU_HYPRE_MEMORY_HOST);
   }
   if (num_cols_offd_L)
   {
      D_data_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_cols_offd_L, NALU_HYPRE_MEMORY_HOST);
      /*L_col_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_L);*/
      remap = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd_L, NALU_HYPRE_MEMORY_HOST);
   }

   Rtilde = nalu_hypre_CTAlloc(nalu_hypre_ParVector,  1, NALU_HYPRE_MEMORY_HOST);
   Rtilde_local = nalu_hypre_SeqVectorCreate(num_rows_L);
   nalu_hypre_SeqVectorInitialize(Rtilde_local);
   nalu_hypre_ParVectorLocalVector(Rtilde) = Rtilde_local;
   nalu_hypre_ParVectorOwnsData(Rtilde) = 1;

   Xtilde = nalu_hypre_CTAlloc(nalu_hypre_ParVector,  1, NALU_HYPRE_MEMORY_HOST);
   Xtilde_local = nalu_hypre_SeqVectorCreate(num_rows_L);
   nalu_hypre_SeqVectorInitialize(Xtilde_local);
   nalu_hypre_ParVectorLocalVector(Xtilde) = Xtilde_local;
   nalu_hypre_ParVectorOwnsData(Xtilde) = 1;

   x_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Xtilde));
   r_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Rtilde));

   cnt = 0;
   cnt_level = 0;
   cnt_diag = 0;
   cnt_offd = 0;
   cnt_row = 1;
   L_diag_i[0] = 0;
   L_offd_i[0] = 0;
   if (ns > 1)
   {
      A_cnt_diag = 0;
      A_cnt_offd = 0;
      Atilde_diag_i[0] = 0;
      Atilde_offd_i[0] = 0;
   }
   for (level = addlvl; level < add_end; level++)
   {
      row_start = level_start[cnt_level];
      if (level != 0)
      {
         tmp_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(F_array[level]));
         if (tmp_data)
         {
            nalu_hypre_TFree(tmp_data, nalu_hypre_VectorMemoryLocation(nalu_hypre_ParVectorLocalVector(F_array[level])));
         }
         nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(F_array[level])) = &r_data[row_start];
         nalu_hypre_VectorOwnsData(nalu_hypre_ParVectorLocalVector(F_array[level])) = 0;

         tmp_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(U_array[level]));
         if (tmp_data)
         {
            nalu_hypre_TFree(tmp_data, nalu_hypre_VectorMemoryLocation(nalu_hypre_ParVectorLocalVector(U_array[level])));
         }
         nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(U_array[level])) = &x_data[row_start];
         nalu_hypre_VectorOwnsData(nalu_hypre_ParVectorLocalVector(U_array[level])) = 0;
      }
      cnt_level++;

      start_diag = L_diag_i[cnt_row - 1];
      start_offd = L_offd_i[cnt_row - 1];
      A_tmp = A_array[level];
      A_tmp_diag = nalu_hypre_ParCSRMatrixDiag(A_tmp);
      A_tmp_offd = nalu_hypre_ParCSRMatrixOffd(A_tmp);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A_tmp);
      A_tmp_diag_i = nalu_hypre_CSRMatrixI(A_tmp_diag);
      A_tmp_offd_i = nalu_hypre_CSRMatrixI(A_tmp_offd);
      A_tmp_diag_j = nalu_hypre_CSRMatrixJ(A_tmp_diag);
      A_tmp_offd_j = nalu_hypre_CSRMatrixJ(A_tmp_offd);
      A_tmp_diag_data = nalu_hypre_CSRMatrixData(A_tmp_diag);
      A_tmp_offd_data = nalu_hypre_CSRMatrixData(A_tmp_offd);
      num_rows_tmp = nalu_hypre_CSRMatrixNumRows(A_tmp_diag);
      if (comm_pkg)
      {
         num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
         num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
         send_procs = nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg);
         recv_procs = nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg);
         send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
         send_map_elmts = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
         recv_vec_starts = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      }
      else
      {
         num_sends = 0;
         num_recvs = 0;
      }

      /* Compute new combined communication package */
      for (i = 0; i < num_sends; i++)
      {
         this_proc = nalu_hypre_BinarySearch(L_send_procs, send_procs[i], num_sends_L);
         indx = L_send_ptr[this_proc];
         for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
         {
            L_send_map_elmts[indx++] = row_start + send_map_elmts[j];
         }
         L_send_ptr[this_proc] = indx;
      }

      cnt_map = 0;
      for (i = 0; i < num_recvs; i++)
      {
         this_proc = nalu_hypre_BinarySearch(L_recv_procs, recv_procs[i], num_recvs_L);
         indx = L_recv_ptr[this_proc];
         for (j = recv_vec_starts[i]; j < recv_vec_starts[i + 1]; j++)
         {
            remap[cnt_map++] = indx++;
         }
         L_recv_ptr[this_proc] = indx;
      }

      /* Compute Lambda */
      if (add_rlx == 0)
      {
         /*NALU_HYPRE_Real rlx_wt = relax_weight[level];*/
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows_tmp; i++)
         {
            D_data[i] = add_rlx_wt / A_tmp_diag_data[A_tmp_diag_i[i]];
            L_diag_i[cnt_row + i] = start_diag + A_tmp_diag_i[i + 1];
            L_offd_i[cnt_row + i] = start_offd + A_tmp_offd_i[i + 1];
         }
         if (ns > 1)
            for (i = 0; i < num_rows_tmp; i++)
            {
               Atilde_diag_i[cnt_row + i] = start_diag + A_tmp_diag_i[i + 1];
               Atilde_offd_i[cnt_row + i] = start_offd + A_tmp_offd_i[i + 1];
            }
      }
      else
      {
         l1_norms = l1_norms_ptr[level];
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows_tmp; i++)
         {
            D_data[i] = 1.0 / nalu_hypre_VectorData(l1_norms)[i];
            L_diag_i[cnt_row + i] = start_diag + A_tmp_diag_i[i + 1];
            L_offd_i[cnt_row + i] = start_offd + A_tmp_offd_i[i + 1];
         }
         if (ns > 1)
         {
            for (i = 0; i < num_rows_tmp; i++)
            {
               Atilde_diag_i[cnt_row + i] = start_diag + A_tmp_diag_i[i + 1];
               Atilde_offd_i[cnt_row + i] = start_offd + A_tmp_offd_i[i + 1];
            }
         }
      }

      if (num_procs > 1)
      {
         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = send_map_starts[i];
            for (j = start; j < send_map_starts[i + 1]; j++)
            {
               buf_data[index++] = D_data[send_map_elmts[j]];
            }
         }

         comm_handle = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg,
                                                    buf_data, D_data_offd);
         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      for (i = 0; i < num_rows_tmp; i++)
      {
         j_indx = A_tmp_diag_i[i];
         if (ns > 1)
         {
            Atilde_diag_data[A_cnt_diag] = A_tmp_diag_data[j_indx];
            Atilde_diag_j[A_cnt_diag++] = i + row_start;
         }
         L_diag_data[cnt_diag] = (2.0 - A_tmp_diag_data[j_indx] * D_data[i]) * D_data[i];
         L_diag_j[cnt_diag++] = i + row_start;
         for (j = A_tmp_diag_i[i] + 1; j < A_tmp_diag_i[i + 1]; j++)
         {
            j_indx = A_tmp_diag_j[j];
            L_diag_data[cnt_diag] = (- A_tmp_diag_data[j] * D_data[j_indx]) * D_data[i];
            L_diag_j[cnt_diag++] = j_indx + row_start;
         }
         for (j = A_tmp_offd_i[i]; j < A_tmp_offd_i[i + 1]; j++)
         {
            j_indx = A_tmp_offd_j[j];
            L_offd_data[cnt_offd] = (- A_tmp_offd_data[j] * D_data_offd[j_indx]) * D_data[i];
            L_offd_j[cnt_offd++] = remap[j_indx];
         }
         if (ns > 1)
         {
            for (j = A_tmp_diag_i[i] + 1; j < A_tmp_diag_i[i + 1]; j++)
            {
               j_indx = A_tmp_diag_j[j];
               Atilde_diag_data[A_cnt_diag] = A_tmp_diag_data[j];
               Atilde_diag_j[A_cnt_diag++] = j_indx + row_start;
            }
            for (j = A_tmp_offd_i[i]; j < A_tmp_offd_i[i + 1]; j++)
            {
               j_indx = A_tmp_offd_j[j];
               Atilde_offd_data[A_cnt_offd] = A_tmp_offd_data[j];
               Atilde_offd_j[A_cnt_offd++] = remap[j_indx];
            }
         }
      }
      cnt_row += num_rows_tmp;
   }

   if (L_send_ptr)
   {
      for (i = num_sends_L - 1; i > 0; i--)
      {
         L_send_ptr[i] = L_send_ptr[i - 1];
      }
      L_send_ptr[0] = 0;
   }
   else
   {
      L_send_ptr = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST);
   }

   if (L_recv_ptr)
   {
      for (i = num_recvs_L - 1; i > 0; i--)
      {
         L_recv_ptr[i] = L_recv_ptr[i - 1];
      }
      L_recv_ptr[0] = 0;
   }
   else
   {
      L_recv_ptr = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST);
   }

   /* Create and fill communication package */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs_L, L_recv_procs, L_recv_ptr,
                                    num_sends_L, L_send_procs, L_send_ptr,
                                    L_send_map_elmts,
                                    &L_comm_pkg);

   Lambda = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParCSRMatrixDiag(Lambda) = L_diag;
   nalu_hypre_ParCSRMatrixOffd(Lambda) = L_offd;
   nalu_hypre_ParCSRMatrixCommPkg(Lambda) = L_comm_pkg;
   nalu_hypre_ParCSRMatrixComm(Lambda) = comm;
   nalu_hypre_ParCSRMatrixOwnsData(Lambda) = 1;

   if (ns > 1)
   {
      Atilde = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix,  1, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParCSRMatrixDiag(Atilde) = Atilde_diag;
      nalu_hypre_ParCSRMatrixOffd(Atilde) = Atilde_offd;
      nalu_hypre_ParCSRMatrixCommPkg(Atilde) = L_comm_pkg;
      nalu_hypre_ParCSRMatrixComm(Atilde) = comm;
      nalu_hypre_ParCSRMatrixOwnsData(Atilde) = 1;
      nalu_hypre_ParAMGDataAtilde(amg_data) = Atilde;
   }

   nalu_hypre_ParAMGDataLambda(amg_data) = Lambda;
   nalu_hypre_ParAMGDataRtilde(amg_data) = Rtilde;
   nalu_hypre_ParAMGDataXtilde(amg_data) = Xtilde;

   nalu_hypre_TFree(D_data_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(D_data, NALU_HYPRE_MEMORY_HOST);
   if (num_procs > 1) { nalu_hypre_TFree(buf_data, NALU_HYPRE_MEMORY_HOST); }
   nalu_hypre_TFree(remap, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(level_start, NALU_HYPRE_MEMORY_HOST);

   return Solve_err_flag;
}

NALU_HYPRE_Int nalu_hypre_CreateDinv(void *amg_vdata)
{
   nalu_hypre_ParAMGData *amg_data = (nalu_hypre_ParAMGData*) amg_vdata;

   /* Data Structure variables */
   nalu_hypre_ParCSRMatrix **A_array;
   nalu_hypre_ParVector    **F_array;
   nalu_hypre_ParVector    **U_array;

   nalu_hypre_ParCSRMatrix *A_tmp;
   nalu_hypre_CSRMatrix *A_tmp_diag;
   nalu_hypre_ParVector *Xtilde;
   nalu_hypre_ParVector *Rtilde;
   nalu_hypre_Vector *Xtilde_local;
   nalu_hypre_Vector *Rtilde_local;
   NALU_HYPRE_Real    *x_data;
   NALU_HYPRE_Real    *r_data;
   NALU_HYPRE_Real    *tmp_data;
   NALU_HYPRE_Real    *D_inv = NULL;
   /*NALU_HYPRE_Real    *relax_weight = NULL;
   NALU_HYPRE_Real     relax_type;*/

   NALU_HYPRE_Int       addlvl;
   NALU_HYPRE_Int       num_levels;
   NALU_HYPRE_Int       num_rows_L;
   NALU_HYPRE_Int       num_rows_tmp;
   NALU_HYPRE_Int       level, i;
   NALU_HYPRE_Int       add_rlx;
   NALU_HYPRE_Real      add_rlx_wt;
   NALU_HYPRE_Int       add_last_lvl, add_end;

   /* Local variables  */
   NALU_HYPRE_Int       Solve_err_flag = 0;

   nalu_hypre_Vector  **l1_norms_ptr = NULL;
   nalu_hypre_Vector   *l1_norms;
   NALU_HYPRE_Int l1_start;

   /* Acquire data and allocate storage */

   A_array           = nalu_hypre_ParAMGDataAArray(amg_data);
   F_array           = nalu_hypre_ParAMGDataFArray(amg_data);
   U_array           = nalu_hypre_ParAMGDataUArray(amg_data);
   addlvl            = nalu_hypre_ParAMGDataSimple(amg_data);
   num_levels        = nalu_hypre_ParAMGDataNumLevels(amg_data);
   add_rlx_wt        = nalu_hypre_ParAMGDataAddRelaxWt(amg_data);
   add_rlx           = nalu_hypre_ParAMGDataAddRelaxType(amg_data);
   add_last_lvl      = nalu_hypre_ParAMGDataAddLastLvl(amg_data);
   /*relax_weight      = nalu_hypre_ParAMGDataRelaxWeight(amg_data);
   relax_type        = nalu_hypre_ParAMGDataGridRelaxType(amg_data)[1];*/

   l1_norms_ptr      = nalu_hypre_ParAMGDataL1Norms(amg_data);
   /* smooth_option       = nalu_hypre_ParAMGDataSmoothOption(amg_data); */
   if (add_last_lvl == -1 ) { add_end = num_levels; }
   else { add_end = add_last_lvl; }

   num_rows_L  = 0;
   for (i = addlvl; i < add_end; i++)
   {
      A_tmp = A_array[i];
      A_tmp_diag = nalu_hypre_ParCSRMatrixDiag(A_tmp);
      num_rows_tmp = nalu_hypre_CSRMatrixNumRows(A_tmp_diag);
      num_rows_L += num_rows_tmp;
   }

   Rtilde = nalu_hypre_CTAlloc(nalu_hypre_ParVector,  1, NALU_HYPRE_MEMORY_HOST);
   Rtilde_local = nalu_hypre_SeqVectorCreate(num_rows_L);
   nalu_hypre_SeqVectorInitialize(Rtilde_local);
   nalu_hypre_ParVectorLocalVector(Rtilde) = Rtilde_local;
   nalu_hypre_ParVectorOwnsData(Rtilde) = 1;

   Xtilde = nalu_hypre_CTAlloc(nalu_hypre_ParVector,  1, NALU_HYPRE_MEMORY_HOST);
   Xtilde_local = nalu_hypre_SeqVectorCreate(num_rows_L);
   nalu_hypre_SeqVectorInitialize(Xtilde_local);
   nalu_hypre_ParVectorLocalVector(Xtilde) = Xtilde_local;
   nalu_hypre_ParVectorOwnsData(Xtilde) = 1;

   x_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Xtilde));
   r_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(Rtilde));
   D_inv = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  num_rows_L, NALU_HYPRE_MEMORY_HOST);

   l1_start = 0;
   for (level = addlvl; level < add_end; level++)
   {
      if (level != 0)
      {
         tmp_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(F_array[level]));
         if (tmp_data)
         {
            nalu_hypre_TFree(tmp_data, nalu_hypre_VectorMemoryLocation(nalu_hypre_ParVectorLocalVector(F_array[level])));
         }
         nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(F_array[level])) = &r_data[l1_start];
         nalu_hypre_VectorOwnsData(nalu_hypre_ParVectorLocalVector(F_array[level])) = 0;

         tmp_data = nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(U_array[level]));
         if (tmp_data)
         {
            nalu_hypre_TFree(tmp_data, nalu_hypre_VectorMemoryLocation(nalu_hypre_ParVectorLocalVector(U_array[level])));
         }
         nalu_hypre_VectorData(nalu_hypre_ParVectorLocalVector(U_array[level])) = &x_data[l1_start];
         nalu_hypre_VectorOwnsData(nalu_hypre_ParVectorLocalVector(U_array[level])) = 0;
      }

      A_tmp = A_array[level];
      A_tmp_diag = nalu_hypre_ParCSRMatrixDiag(A_tmp);
      num_rows_tmp = nalu_hypre_CSRMatrixNumRows(A_tmp_diag);

      if (add_rlx == 0)
      {
         /*NALU_HYPRE_Real rlx_wt = relax_weight[level];*/
         NALU_HYPRE_Int *A_tmp_diag_i = nalu_hypre_CSRMatrixI(A_tmp_diag);
         NALU_HYPRE_Real *A_tmp_diag_data = nalu_hypre_CSRMatrixData(A_tmp_diag);
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows_tmp; i++)
         {
            D_inv[l1_start + i] = add_rlx_wt / A_tmp_diag_data[A_tmp_diag_i[i]];
         }
      }
      else
      {
         l1_norms = l1_norms_ptr[level];
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows_tmp; i++)
         {
            D_inv[l1_start + i] = 1.0 / nalu_hypre_VectorData(l1_norms)[i];
         }
      }
      l1_start += num_rows_tmp;
   }

   nalu_hypre_ParAMGDataDinv(amg_data) = D_inv;
   nalu_hypre_ParAMGDataRtilde(amg_data) = Rtilde;
   nalu_hypre_ParAMGDataXtilde(amg_data) = Xtilde;

   return Solve_err_flag;
}
