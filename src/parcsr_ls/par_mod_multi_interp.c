/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ParAMGBuildModMultipass
 * This routine implements Stuben's direct interpolation with multiple passes.
 * expressed with matrix matrix multiplications
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildModMultipassHost( nalu_hypre_ParCSRMatrix  *A,
                                      NALU_HYPRE_Int           *CF_marker,
                                      nalu_hypre_ParCSRMatrix  *S,
                                      NALU_HYPRE_BigInt        *num_cpts_global,
                                      NALU_HYPRE_Real           trunc_factor,
                                      NALU_HYPRE_Int            P_max_elmts,
                                      NALU_HYPRE_Int            interp_type,
                                      NALU_HYPRE_Int            num_functions,
                                      NALU_HYPRE_Int           *dof_func,
                                      nalu_hypre_ParCSRMatrix **P_ptr )
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_MULTIPASS_INTERP] -= nalu_hypre_MPI_Wtime();
#endif

   MPI_Comm                comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg    *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle *comm_handle;

   NALU_HYPRE_MemoryLocation memory_location_P = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int       *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int        n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int       *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Real      *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);

   NALU_HYPRE_Int        num_cols_offd_A = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_CSRMatrix *S_diag = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int       *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int       *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);

   nalu_hypre_CSRMatrix *S_offd = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int       *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int       *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);

   nalu_hypre_ParCSRMatrix **Pi;
   nalu_hypre_ParCSRMatrix *P;
   nalu_hypre_CSRMatrix *P_diag;
   NALU_HYPRE_Real      *P_diag_data;
   NALU_HYPRE_Int       *P_diag_i; /*at first counter of nonzero cols for each row,
                                      finally will be pointer to start of row */
   NALU_HYPRE_Int       *P_diag_j;

   nalu_hypre_CSRMatrix *P_offd;
   NALU_HYPRE_Real      *P_offd_data = NULL;
   NALU_HYPRE_Int       *P_offd_i; /*at first counter of nonzero cols for each row,
                                      finally will be pointer to start of row */
   NALU_HYPRE_Int       *P_offd_j = NULL;
   NALU_HYPRE_BigInt    *col_map_offd_P = NULL;
   NALU_HYPRE_Int        num_cols_offd_P = 0;

   NALU_HYPRE_Int        num_sends = 0;
   NALU_HYPRE_Int       *int_buf_data = NULL;

   NALU_HYPRE_Int       *fine_to_coarse;
   NALU_HYPRE_Int       *points_left;
   NALU_HYPRE_Int       *pass_marker;
   NALU_HYPRE_Int       *pass_marker_offd = NULL;
   NALU_HYPRE_Int       *pass_order;
   NALU_HYPRE_Int       *pass_starts;

   NALU_HYPRE_Int        i, j, i1, i2, j1;
   NALU_HYPRE_Int        num_passes, p;
   NALU_HYPRE_BigInt     global_remaining, remaining;
   NALU_HYPRE_Int        cnt, cnt_old, cnt_rem, current_pass;
   NALU_HYPRE_Int        startc, index;

   NALU_HYPRE_BigInt     total_global_cpts;
   NALU_HYPRE_Int        my_id, num_procs;
   NALU_HYPRE_Int        P_offd_size = 0;

   NALU_HYPRE_Int       *dof_func_offd = NULL;
   NALU_HYPRE_Real      *row_sums = NULL;

   /* MPI size and rank*/
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (num_procs > 1)
   {
      if (my_id == num_procs - 1)
      {
         total_global_cpts = num_cpts_global[1];
      }
      nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      total_global_cpts = num_cpts_global[1];
   }

   if (total_global_cpts == 0)
   {
      *P_ptr = NULL;
      return nalu_hypre_error_flag;
   }
   /* Generate pass marker array */

   pass_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   /* contains pass numbers for each variable according to original order */
   pass_order = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   /* contains row numbers according to new order, pass 1 followed by pass 2 etc */
   fine_to_coarse = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   /* reverse of pass_order, keeps track where original numbers go */
   points_left = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   /* contains row numbers of remaining points, auxiliary */
   pass_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int, 11, NALU_HYPRE_MEMORY_HOST);
   /* contains beginning for each pass in pass_order field, assume no more than 10 passes */

   P_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine + 1, memory_location_P);
   P_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine + 1, memory_location_P);

   cnt = 0;
   remaining = 0;
   for (i = 0; i < n_fine; i++)
   {
      if (CF_marker[i] == 1)
      {
         pass_marker[i] = 1;
         P_diag_i[i + 1] = 1;
         P_offd_i[i + 1] = 0;
         fine_to_coarse[i] = cnt;
         pass_order[cnt++] = i;
      }
      else
      {
         points_left[remaining++] = i;
      }
   }
   pass_starts[0] = 0;
   pass_starts[1] = cnt;

   if (num_functions > 1)
   {
      dof_func_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_A, NALU_HYPRE_MEMORY_HOST);
      index = 0;
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                   NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sends; i++)
      {
         startc = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = startc; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            int_buf_data[index++] = dof_func[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, dof_func_offd);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

      nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   }

   if (num_procs > 1)
   {
      pass_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_A, NALU_HYPRE_MEMORY_HOST);
      index = 0;
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                   NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sends; i++)
      {
         startc = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = startc; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            int_buf_data[index++] = pass_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, pass_marker_offd);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }
   current_pass = 1;
   num_passes = 1;
   /* color points according to pass number */
   nalu_hypre_MPI_Allreduce(&remaining, &global_remaining, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
   while (global_remaining > 0)
   {
      NALU_HYPRE_Int remaining_pts = (NALU_HYPRE_Int) remaining;
      NALU_HYPRE_BigInt old_global_remaining = global_remaining;
      cnt_rem = 0;
      for (i = 0; i < remaining_pts; i++)
      {
         i1 = points_left[i];
         cnt_old = cnt;
         for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
         {
            j1 = S_diag_j[j];
            if (pass_marker[j1] == current_pass)
            {
               pass_marker[i1] = current_pass + 1;
               pass_order[cnt++] = i1;
               remaining--;
               break;
            }
         }
         if (cnt == cnt_old)
         {
            for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
            {
               j1 = S_offd_j[j];
               if (pass_marker_offd[j1] == current_pass)
               {
                  pass_marker[i1] = current_pass + 1;
                  pass_order[cnt++] = i1;
                  remaining--;
                  break;
               }
            }
         }
         if (cnt == cnt_old)
         {
            points_left[cnt_rem++] = i1;
         }
      }
      remaining = (NALU_HYPRE_BigInt) cnt_rem;
      current_pass++;
      num_passes++;
      if (num_passes > 9)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, " Warning!!! too many passes! out of range!\n");
         break;
      }
      pass_starts[num_passes] = cnt;
      /* update pass_marker_offd */
      index = 0;
      if (num_procs > 1)
      {
         for (i = 0; i < num_sends; i++)
         {
            startc = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = startc; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               int_buf_data[index++] = pass_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
         }
         comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, pass_marker_offd);

         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      }
      old_global_remaining = global_remaining;
      nalu_hypre_MPI_Allreduce(&remaining, &global_remaining, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
      /* if the number of remaining points does not change, we have a situation of isolated areas of
       * fine points that are not connected to any C-points, and the pass generation process breaks
       * down. Those points can be ignored, i.e. the corresponding rows in P will just be 0
       * and can be ignored for the algorithm. */
      if (old_global_remaining == global_remaining) { break; }
   }
   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(points_left, NALU_HYPRE_MEMORY_HOST);

   /* generate row sum of weak points and C-points to be ignored */

   row_sums = nalu_hypre_CTAlloc(NALU_HYPRE_Real, n_fine, NALU_HYPRE_MEMORY_HOST);
   if (num_functions >  1)
   {
      for (i = 0; i < n_fine; i++)
      {
         if (CF_marker[i] < 0)
         {
            for (j = A_diag_i[i] + 1; j < A_diag_i[i + 1]; j++)
            {
               if (dof_func[i] == dof_func[A_diag_j[j]])
               {
                  row_sums[i] += A_diag_data[j];
               }
            }
            for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
            {
               if (dof_func[i] == dof_func_offd[A_offd_j[j]])
               {
                  row_sums[i] += A_offd_data[j];
               }
            }
         }
      }
   }
   else
   {
      for (i = 0; i < n_fine; i++)
      {
         if (CF_marker[i] < 0)
         {
            for (j = A_diag_i[i] + 1; j < A_diag_i[i + 1]; j++)
            {
               row_sums[i] += A_diag_data[j];
            }
            for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
            {
               row_sums[i] += A_offd_data[j];
            }
         }
      }
   }

   Pi = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix*, num_passes, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_GenerateMultipassPi(A, S, num_cpts_global, &pass_order[pass_starts[1]], pass_marker,
                             pass_marker_offd, pass_starts[2] - pass_starts[1], 1, row_sums, &Pi[0]);
   if (interp_type == 8)
   {
      for (i = 1; i < num_passes - 1; i++)
      {
         nalu_hypre_ParCSRMatrix *Q;
         NALU_HYPRE_BigInt *c_pts_starts = nalu_hypre_ParCSRMatrixRowStarts(Pi[i - 1]);
         nalu_hypre_GenerateMultipassPi(A, S, c_pts_starts, &pass_order[pass_starts[i + 1]], pass_marker,
                                   pass_marker_offd, pass_starts[i + 2] - pass_starts[i + 1], i + 1, row_sums, &Q);
         Pi[i] = nalu_hypre_ParMatmul(Q, Pi[i - 1]);
         nalu_hypre_ParCSRMatrixDestroy(Q);
      }
   }
   else if (interp_type == 9)
   {
      for (i = 1; i < num_passes - 1; i++)
      {
         NALU_HYPRE_BigInt *c_pts_starts = nalu_hypre_ParCSRMatrixRowStarts(Pi[i - 1]);
         nalu_hypre_GenerateMultiPi(A, S, Pi[i - 1], c_pts_starts, &pass_order[pass_starts[i + 1]], pass_marker,
                               pass_marker_offd, pass_starts[i + 2] - pass_starts[i + 1], i + 1,
                               num_functions, dof_func, dof_func_offd, &Pi[i]);
      }
   }

   /* p pulate P_diag_i[i+1] with nnz of i-th row */
   for (i = 0; i < num_passes - 1; i++)
   {
      NALU_HYPRE_Int *Pi_diag_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(Pi[i]));
      NALU_HYPRE_Int *Pi_offd_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(Pi[i]));
      j1 = 0;
      for (j = pass_starts[i + 1]; j < pass_starts[i + 2]; j++)
      {
         i1 = pass_order[j];
         P_diag_i[i1 + 1] = Pi_diag_i[j1 + 1] - Pi_diag_i[j1];
         P_offd_i[i1 + 1] = Pi_offd_i[j1 + 1] - Pi_offd_i[j1];
         j1++;
      }
   }

   for (i = 0; i < n_fine; i++)
   {
      P_diag_i[i + 1] += P_diag_i[i];
      P_offd_i[i + 1] += P_offd_i[i];
   }

   P_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, P_diag_i[n_fine], memory_location_P);
   P_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, P_diag_i[n_fine], memory_location_P);
   P_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, P_offd_i[n_fine], memory_location_P);
   P_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, P_offd_i[n_fine], memory_location_P);

   /* insert weights for coarse points */
   for (i = 0; i < pass_starts[1]; i++)
   {
      i1 = pass_order[i];
      j = P_diag_i[i1];
      P_diag_j[j] = fine_to_coarse[i1];
      P_diag_data[j] = 1.0;
   }

   /* generate col_map_offd_P by combining all col_map_offd_Pi
    * and reompute indices if needed */

   /* insert remaining weights */
   for (p = 0; p < num_passes - 1; p++)
   {
      NALU_HYPRE_Int *Pi_diag_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(Pi[p]));
      NALU_HYPRE_Int *Pi_offd_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(Pi[p]));
      NALU_HYPRE_Int *Pi_diag_j = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(Pi[p]));
      NALU_HYPRE_Int *Pi_offd_j = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(Pi[p]));
      NALU_HYPRE_Real *Pi_diag_data = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(Pi[p]));
      NALU_HYPRE_Real *Pi_offd_data = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(Pi[p]));
      j1 = 0;
      for (i = pass_starts[p + 1]; i < pass_starts[p + 2]; i++)
      {
         i1 = pass_order[i];
         i2 = Pi_diag_i[j1];
         for (j = P_diag_i[i1]; j < P_diag_i[i1 + 1]; j++)
         {
            P_diag_j[j] = Pi_diag_j[i2];
            P_diag_data[j] = Pi_diag_data[i2++];
         }
         i2 = Pi_offd_i[j1];
         for (j = P_offd_i[i1]; j < P_offd_i[i1 + 1]; j++)
         {
            P_offd_j[j] = Pi_offd_j[i2];
            P_offd_data[j] = Pi_offd_data[i2++];
         }
         j1++;
      }
   }
   /* Note that col indices in P_offd_j probably not consistent,
      this gets fixed after truncation */

   P = nalu_hypre_ParCSRMatrixCreate(comm,
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                total_global_cpts,
                                nalu_hypre_ParCSRMatrixRowStarts(A),
                                num_cpts_global,
                                num_cols_offd_P,
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

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0 || P_max_elmts > 0)
   {
      nalu_hypre_BoomerAMGInterpTruncation(P, trunc_factor, P_max_elmts);
      P_diag_data = nalu_hypre_CSRMatrixData(P_diag);
      P_diag_i = nalu_hypre_CSRMatrixI(P_diag);
      P_diag_j = nalu_hypre_CSRMatrixJ(P_diag);
      P_offd_data = nalu_hypre_CSRMatrixData(P_offd);
      P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
      P_offd_j = nalu_hypre_CSRMatrixJ(P_offd);
   }

   num_cols_offd_P = 0;
   P_offd_size = P_offd_i[n_fine];
   if (P_offd_size)
   {
      NALU_HYPRE_BigInt *tmp_P_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, P_offd_size, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_BigInt *big_P_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, P_offd_size, NALU_HYPRE_MEMORY_HOST);
      for (p = 0; p < num_passes - 1; p++)
      {
         NALU_HYPRE_BigInt *col_map_offd_Pi = nalu_hypre_ParCSRMatrixColMapOffd(Pi[p]);
         for (i = pass_starts[p + 1]; i < pass_starts[p + 2]; i++)
         {
            i1 = pass_order[i];
            for (j = P_offd_i[i1]; j < P_offd_i[i1 + 1]; j++)
            {
               big_P_offd_j[j] = col_map_offd_Pi[P_offd_j[j]];
            }
         }
      }

      for (i = 0; i < P_offd_size; i++)
      {
         tmp_P_offd_j[i] = big_P_offd_j[i];
      }

      nalu_hypre_BigQsort0(tmp_P_offd_j, 0, P_offd_size - 1);

      num_cols_offd_P = 1;
      for (i = 0; i < P_offd_size - 1; i++)
      {
         if (tmp_P_offd_j[i + 1] > tmp_P_offd_j[i])
         {
            tmp_P_offd_j[num_cols_offd_P++] = tmp_P_offd_j[i + 1];
         }
      }

      col_map_offd_P = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_P, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_cols_offd_P; i++)
      {
         col_map_offd_P[i] = tmp_P_offd_j[i];
      }

      for (i = 0; i < P_offd_size; i++)
      {
         P_offd_j[i] = nalu_hypre_BigBinarySearch(col_map_offd_P,
                                             big_P_offd_j[i],
                                             num_cols_offd_P);
      }
      nalu_hypre_TFree(tmp_P_offd_j, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(big_P_offd_j, NALU_HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < num_passes - 1; i++)
   {
      nalu_hypre_ParCSRMatrixDestroy(Pi[i]);
   }
   nalu_hypre_TFree (Pi, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree (pass_marker, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree (pass_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree (pass_order, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree (pass_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree (fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree (dof_func_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree (row_sums, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < n_fine; i++)
   {
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }
   }

   nalu_hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
   nalu_hypre_CSRMatrixNumCols(P_offd) = num_cols_offd_P;

   nalu_hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   return nalu_hypre_error_flag;

}


NALU_HYPRE_Int
nalu_hypre_GenerateMultipassPi( nalu_hypre_ParCSRMatrix  *A,
                           nalu_hypre_ParCSRMatrix  *S,
                           NALU_HYPRE_BigInt        *c_pts_starts,
                           NALU_HYPRE_Int
                           *pass_order, /* array containing row numbers of rows in A and S to be considered */
                           NALU_HYPRE_Int           *pass_marker,
                           NALU_HYPRE_Int           *pass_marker_offd,
                           NALU_HYPRE_Int            num_points,
                           NALU_HYPRE_Int            color,
                           NALU_HYPRE_Real          *row_sums,
                           nalu_hypre_ParCSRMatrix **P_ptr )
{
   MPI_Comm                comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg    *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle *comm_handle;

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Int        n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real      *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Int        num_cols_offd_A = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_CSRMatrix *S_diag = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int       *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int       *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);

   nalu_hypre_CSRMatrix *S_offd = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int       *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int       *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);
   NALU_HYPRE_BigInt    *col_map_offd_P = NULL;
   NALU_HYPRE_Int        num_cols_offd_P;
   NALU_HYPRE_Int        nnz_diag, nnz_offd;
   NALU_HYPRE_Int        n_cpts, i, j, i1, j1, j2;
   NALU_HYPRE_Int        startc, index;
   NALU_HYPRE_Int        cpt, cnt_diag, cnt_offd;

   nalu_hypre_ParCSRMatrix *P;
   nalu_hypre_CSRMatrix *P_diag;
   NALU_HYPRE_Real      *P_diag_data;
   NALU_HYPRE_Int       *P_diag_i; /*at first counter of nonzero cols for each row,
                                      finally will be pointer to start of row */
   NALU_HYPRE_Int       *P_diag_j;

   nalu_hypre_CSRMatrix *P_offd;
   NALU_HYPRE_Real      *P_offd_data = NULL;
   NALU_HYPRE_Int       *P_offd_i; /*at first counter of nonzero cols for each row,
                                      finally will be pointer to start of row */
   NALU_HYPRE_Int       *P_offd_j = NULL;
   NALU_HYPRE_Int       *fine_to_coarse;
   NALU_HYPRE_Int       *fine_to_coarse_offd = NULL;
   NALU_HYPRE_BigInt     f_pts_starts[2];
   NALU_HYPRE_Int        my_id, num_procs;
   NALU_HYPRE_BigInt     total_global_fpts;
   NALU_HYPRE_BigInt     total_global_cpts;
   NALU_HYPRE_BigInt    *big_convert;
   NALU_HYPRE_BigInt    *big_convert_offd = NULL;
   NALU_HYPRE_BigInt    *big_buf_data = NULL;
   NALU_HYPRE_Int        num_sends;
   NALU_HYPRE_Real      *row_sum_C;

   /* MPI size and rank*/
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* define P matrices */

   P_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_HOST);
   P_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_HOST);
   fine_to_coarse = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);

   /* fill P */

   n_cpts = 0;
   for (i = 0; i < n_fine; i++)
   {
      if (pass_marker[i] == color)
      {
         fine_to_coarse[i] = n_cpts++;
      }
      else
      {
         fine_to_coarse[i] = -1;
      }
   }

   if (num_procs > 1)
   {
      NALU_HYPRE_BigInt big_Fpts;
      big_Fpts = num_points;

      nalu_hypre_MPI_Scan(&big_Fpts, f_pts_starts + 1, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
      f_pts_starts[0] = f_pts_starts[1] - big_Fpts;
      if (my_id == num_procs - 1)
      {
         total_global_fpts = f_pts_starts[1];
         total_global_cpts = c_pts_starts[1];
      }
      nalu_hypre_MPI_Bcast(&total_global_fpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
      nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      f_pts_starts[0] = 0;
      f_pts_starts[1] = num_points;
      total_global_fpts = f_pts_starts[1];
      total_global_cpts = c_pts_starts[1];
   }

   {
      big_convert = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, n_fine, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < n_fine; i++)
      {
         if (pass_marker[i] == color)
         {
            big_convert[i] = (NALU_HYPRE_BigInt)fine_to_coarse[i] + c_pts_starts[0];
         }
      }

      num_cols_offd_P = 0;
      if (num_procs > 1)
      {
         big_convert_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_A, NALU_HYPRE_MEMORY_HOST);
         fine_to_coarse_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_A, NALU_HYPRE_MEMORY_HOST);
         index = 0;
         num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
         big_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                      NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sends; i++)
         {
            startc = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = startc; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               big_buf_data[index++] = big_convert[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
         }

         comm_handle = nalu_hypre_ParCSRCommHandleCreate( 21, comm_pkg, big_buf_data, big_convert_offd);

         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

         num_cols_offd_P = 0;
         for (i = 0; i < num_cols_offd_A; i++)
         {
            if (pass_marker_offd[i] == color)
            {
               fine_to_coarse_offd[i] = num_cols_offd_P++;
            }
         }

         col_map_offd_P = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_P, NALU_HYPRE_MEMORY_HOST);

         cpt = 0;
         for (i = 0; i < num_cols_offd_A; i++)
         {
            if (pass_marker_offd[i] == color)
            {
               col_map_offd_P[cpt++] = big_convert_offd[i];
            }
         }
      }
   }

   /* generate P_diag_i and P_offd_i */
   nnz_diag = 0;
   nnz_offd = 0;
   for (i = 0; i < num_points; i++)
   {
      i1 = pass_order[i];
      for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
      {
         j1 = S_diag_j[j];
         if (pass_marker[j1] == color)
         {
            P_diag_i[i + 1]++;
            nnz_diag++;
         }
      }
      for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
      {
         j1 = S_offd_j[j];
         if (pass_marker_offd[j1] == color)
         {
            P_offd_i[i + 1]++;
            nnz_offd++;
         }
      }
   }

   for (i = 1; i < num_points + 1; i++)
   {
      P_diag_i[i] += P_diag_i[i - 1];
      P_offd_i[i] += P_offd_i[i - 1];
   }

   P_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nnz_diag, NALU_HYPRE_MEMORY_HOST);
   P_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, nnz_diag, NALU_HYPRE_MEMORY_HOST);
   P_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nnz_offd, NALU_HYPRE_MEMORY_HOST);
   P_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, nnz_offd, NALU_HYPRE_MEMORY_HOST);

   cnt_diag = 0;
   cnt_offd = 0;
   for (i = 0; i < num_points; i++)
   {
      i1 = pass_order[i];
      j2 = A_diag_i[i1];
      for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
      {
         j1 = S_diag_j[j];
         while (A_diag_j[j2] != j1) { j2++; }
         if (pass_marker[j1] == color && A_diag_j[j2] == j1)
         {
            P_diag_j[cnt_diag] = fine_to_coarse[j1];
            P_diag_data[cnt_diag++] = A_diag_data[j2];
         }
      }
      j2 = A_offd_i[i1];
      for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
      {
         j1 = S_offd_j[j];
         while (A_offd_j[j2] != j1) { j2++; }
         if (pass_marker_offd[j1] == color && A_offd_j[j2] == j1)
         {
            P_offd_j[cnt_offd] = fine_to_coarse_offd[j1];
            P_offd_data[cnt_offd++] = A_offd_data[j2];
         }
      }
   }

   //row_sums = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_points, NALU_HYPRE_MEMORY_HOST);
   row_sum_C = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_points, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_points; i++)
   {
      NALU_HYPRE_Real diagonal, value;
      i1 = pass_order[i];
      diagonal = A_diag_data[A_diag_i[i1]];
      /*for (j=A_diag_i[i1]+1; j < A_diag_i[i1+1]; j++)
      {
         row_sums[i] += A_diag_data[j];
      }
      for (j=A_offd_i[i1]; j < A_offd_i[i1+1]; j++)
      {
         row_sums[i] += A_offd_data[j];
      }*/
      for (j = P_diag_i[i]; j < P_diag_i[i + 1]; j++)
      {
         row_sum_C[i] += P_diag_data[j];
      }
      for (j = P_offd_i[i]; j < P_offd_i[i + 1]; j++)
      {
         row_sum_C[i] += P_offd_data[j];
      }
      value = row_sum_C[i] * diagonal;
      if (value != 0)
      {
         row_sums[i1] /= value;
      }
      for (j = P_diag_i[i]; j < P_diag_i[i + 1]; j++)
      {
         P_diag_data[j] = -P_diag_data[j] * row_sums[i1];
      }
      for (j = P_offd_i[i]; j < P_offd_i[i + 1]; j++)
      {
         P_offd_data[j] = -P_offd_data[j] * row_sums[i1];
      }
   }


   P = nalu_hypre_ParCSRMatrixCreate(comm,
                                total_global_fpts,
                                total_global_cpts,
                                f_pts_starts,
                                c_pts_starts,
                                num_cols_offd_P,
                                P_diag_i[num_points],
                                P_offd_i[num_points]);

   P_diag = nalu_hypre_ParCSRMatrixDiag(P);
   nalu_hypre_CSRMatrixData(P_diag) = P_diag_data;
   nalu_hypre_CSRMatrixI(P_diag) = P_diag_i;
   nalu_hypre_CSRMatrixJ(P_diag) = P_diag_j;
   P_offd = nalu_hypre_ParCSRMatrixOffd(P);
   nalu_hypre_CSRMatrixData(P_offd) = P_offd_data;
   nalu_hypre_CSRMatrixI(P_offd) = P_offd_i;
   nalu_hypre_CSRMatrixJ(P_offd) = P_offd_j;
   nalu_hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;

   nalu_hypre_CSRMatrixMemoryLocation(P_diag) = NALU_HYPRE_MEMORY_HOST;
   nalu_hypre_CSRMatrixMemoryLocation(P_offd) = NALU_HYPRE_MEMORY_HOST;

   /* free stuff */
   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse_offd, NALU_HYPRE_MEMORY_HOST);
   //nalu_hypre_TFree(row_sums, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(row_sum_C, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_convert, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_convert_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_buf_data, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_MatvecCommPkgCreate(P);
   *P_ptr = P;

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_GenerateMultiPi( nalu_hypre_ParCSRMatrix  *A,
                       nalu_hypre_ParCSRMatrix  *S,
                       nalu_hypre_ParCSRMatrix  *P,
                       NALU_HYPRE_BigInt        *c_pts_starts,
                       NALU_HYPRE_Int
                       *pass_order, /* array containing row numbers of rows in A and S to be considered */
                       NALU_HYPRE_Int           *pass_marker,
                       NALU_HYPRE_Int           *pass_marker_offd,
                       NALU_HYPRE_Int            num_points,
                       NALU_HYPRE_Int            color,
                       NALU_HYPRE_Int            num_functions,
                       NALU_HYPRE_Int           *dof_func,
                       NALU_HYPRE_Int           *dof_func_offd,
                       nalu_hypre_ParCSRMatrix **Pi_ptr )
{
   MPI_Comm                comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg    *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle *comm_handle;

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Int        n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real      *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Int        num_cols_offd_A = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_CSRMatrix *S_diag = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int       *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int       *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);

   nalu_hypre_CSRMatrix *S_offd = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int       *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int       *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);
   NALU_HYPRE_BigInt    *col_map_offd_Q = NULL;
   NALU_HYPRE_Int        num_cols_offd_Q;

   nalu_hypre_ParCSRMatrix *Pi;
   nalu_hypre_CSRMatrix *Pi_diag;
   NALU_HYPRE_Int       *Pi_diag_i;
   NALU_HYPRE_Real      *Pi_diag_data;

   nalu_hypre_CSRMatrix *Pi_offd;
   NALU_HYPRE_Int       *Pi_offd_i;
   NALU_HYPRE_Real      *Pi_offd_data;

   NALU_HYPRE_Int        nnz_diag, nnz_offd;
   NALU_HYPRE_Int        n_cpts, i, j, i1, j1, j2;
   NALU_HYPRE_Int        startc, index;
   NALU_HYPRE_Int        cpt, cnt_diag, cnt_offd;

   nalu_hypre_ParCSRMatrix *Q;
   nalu_hypre_CSRMatrix *Q_diag;
   NALU_HYPRE_Real      *Q_diag_data;
   NALU_HYPRE_Int       *Q_diag_i; /*at first counter of nonzero cols for each row,
                                      finally will be pointer to start of row */
   NALU_HYPRE_Int       *Q_diag_j;

   nalu_hypre_CSRMatrix *Q_offd;
   NALU_HYPRE_Real      *Q_offd_data = NULL;
   NALU_HYPRE_Int       *Q_offd_i; /*at first counter of nonzero cols for each row,
                                      finally will be pointer to start of row */
   NALU_HYPRE_Int       *Q_offd_j = NULL;
   NALU_HYPRE_Int       *fine_to_coarse;
   NALU_HYPRE_Int       *fine_to_coarse_offd = NULL;
   NALU_HYPRE_BigInt     f_pts_starts[2];
   NALU_HYPRE_Int        my_id, num_procs;
   NALU_HYPRE_BigInt     total_global_fpts;
   NALU_HYPRE_BigInt     total_global_cpts;
   NALU_HYPRE_BigInt    *big_convert;
   NALU_HYPRE_BigInt    *big_convert_offd = NULL;
   NALU_HYPRE_BigInt    *big_buf_data = NULL;
   NALU_HYPRE_Int        num_sends;
   //NALU_HYPRE_Real      *row_sums;
   NALU_HYPRE_Real      *row_sums_C;
   NALU_HYPRE_Real      *w_row_sum;

   /* MPI size and rank*/
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* define P matrices */

   Q_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_HOST);
   Q_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_points + 1, NALU_HYPRE_MEMORY_HOST);
   fine_to_coarse = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);

   /* fill P */

   n_cpts = 0;
   for (i = 0; i < n_fine; i++)
   {
      if (pass_marker[i] == color)
      {
         fine_to_coarse[i] = n_cpts++;
      }
      else
      {
         fine_to_coarse[i] = -1;
      }
   }

   if (num_procs > 1)
   {
      NALU_HYPRE_BigInt big_Fpts;
      big_Fpts = num_points;

      nalu_hypre_MPI_Scan(&big_Fpts, f_pts_starts + 1, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
      f_pts_starts[0] = f_pts_starts[1] - big_Fpts;
      if (my_id == num_procs - 1)
      {
         total_global_fpts = f_pts_starts[1];
         total_global_cpts = c_pts_starts[1];
      }
      nalu_hypre_MPI_Bcast(&total_global_fpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
      nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }
   else
   {
      f_pts_starts[0] = 0;
      f_pts_starts[1] = num_points;
      total_global_fpts = f_pts_starts[1];
      total_global_cpts = c_pts_starts[1];
   }

   {
      big_convert = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, n_fine, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < n_fine; i++)
      {
         if (pass_marker[i] == color)
         {
            big_convert[i] = (NALU_HYPRE_BigInt)fine_to_coarse[i] + c_pts_starts[0];
         }
      }

      num_cols_offd_Q = 0;
      if (num_procs > 1)
      {
         big_convert_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_A, NALU_HYPRE_MEMORY_HOST);
         fine_to_coarse_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_A, NALU_HYPRE_MEMORY_HOST);
         index = 0;
         num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
         big_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                      NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sends; i++)
         {
            startc = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = startc; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               big_buf_data[index++] = big_convert[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
         }

         comm_handle = nalu_hypre_ParCSRCommHandleCreate( 21, comm_pkg, big_buf_data, big_convert_offd);

         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

         num_cols_offd_Q = 0;
         for (i = 0; i < num_cols_offd_A; i++)
         {
            if (pass_marker_offd[i] == color)
            {
               fine_to_coarse_offd[i] = num_cols_offd_Q++;
            }
         }

         col_map_offd_Q = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_Q, NALU_HYPRE_MEMORY_HOST);

         cpt = 0;
         for (i = 0; i < num_cols_offd_A; i++)
         {
            if (pass_marker_offd[i] == color)
            {
               col_map_offd_Q[cpt++] = big_convert_offd[i];
            }
         }
      }
   }

   /* generate Q_diag_i and Q_offd_i */
   nnz_diag = 0;
   nnz_offd = 0;
   for (i = 0; i < num_points; i++)
   {
      i1 = pass_order[i];
      for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
      {
         j1 = S_diag_j[j];
         if (pass_marker[j1] == color)
         {
            Q_diag_i[i + 1]++;
            nnz_diag++;
         }
      }
      for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
      {
         j1 = S_offd_j[j];
         if (pass_marker_offd[j1] == color)
         {
            Q_offd_i[i + 1]++;
            nnz_offd++;
         }
      }
   }

   for (i = 1; i < num_points + 1; i++)
   {
      Q_diag_i[i] += Q_diag_i[i - 1];
      Q_offd_i[i] += Q_offd_i[i - 1];
   }

   Q_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nnz_diag, NALU_HYPRE_MEMORY_HOST);
   Q_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, nnz_diag, NALU_HYPRE_MEMORY_HOST);
   Q_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nnz_offd, NALU_HYPRE_MEMORY_HOST);
   Q_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, nnz_offd, NALU_HYPRE_MEMORY_HOST);
   w_row_sum = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_points, NALU_HYPRE_MEMORY_HOST);

   cnt_diag = 0;
   cnt_offd = 0;
   if (num_functions > 1)
   {
      for (i = 0; i < num_points; i++)
      {
         i1 = pass_order[i];
         j2 = A_diag_i[i1] + 1;
         //if (w_row_minus) w_row_sum[i] = -w_row_minus[i1];
         for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
         {
            j1 = S_diag_j[j];
            while (A_diag_j[j2] != j1)
            {
               if (dof_func[i1] == dof_func[A_diag_j[j2]])
               {
                  w_row_sum[i] += A_diag_data[j2];
               }
               j2++;
            }
            if (pass_marker[j1] == color && A_diag_j[j2] == j1)
            {
               Q_diag_j[cnt_diag] = fine_to_coarse[j1];
               Q_diag_data[cnt_diag++] = A_diag_data[j2++];
            }
            else
            {
               if (dof_func[i1] == dof_func[A_diag_j[j2]])
               {
                  w_row_sum[i] += A_diag_data[j2];
               }
               j2++;
            }
         }
         while (j2 < A_diag_i[i1 + 1])
         {
            if (dof_func[i1] == dof_func[A_diag_j[j2]])
            {
               w_row_sum[i] += A_diag_data[j2];
            }
            j2++;
         }
         j2 = A_offd_i[i1];
         for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
         {
            j1 = S_offd_j[j];
            while (A_offd_j[j2] != j1)
            {
               if (dof_func[i1] == dof_func_offd[A_offd_j[j2]])
               {
                  w_row_sum[i] += A_offd_data[j2];
               }
               j2++;
            }
            if (pass_marker_offd[j1] == color && A_offd_j[j2] == j1)
            {
               Q_offd_j[cnt_offd] = fine_to_coarse_offd[j1];
               Q_offd_data[cnt_offd++] = A_offd_data[j2++];
            }
            else
            {
               if (dof_func[i1] == dof_func_offd[A_offd_j[j2]])
               {
                  w_row_sum[i] += A_offd_data[j2];
               }
               j2++;
            }
         }
         while (j2 < A_offd_i[i1 + 1])
         {
            if (dof_func[i1] == dof_func_offd[A_offd_j[j2]])
            {
               w_row_sum[i] += A_offd_data[j2];
            }
            j2++;
         }
      }
   }
   else
   {
      for (i = 0; i < num_points; i++)
      {
         i1 = pass_order[i];
         j2 = A_diag_i[i1] + 1;
         for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
         {
            j1 = S_diag_j[j];
            while (A_diag_j[j2] != j1)
            {
               w_row_sum[i] += A_diag_data[j2];
               j2++;
            }
            if (pass_marker[j1] == color && A_diag_j[j2] == j1)
            {
               Q_diag_j[cnt_diag] = fine_to_coarse[j1];
               Q_diag_data[cnt_diag++] = A_diag_data[j2++];
            }
            else
            {
               w_row_sum[i] += A_diag_data[j2];
               j2++;
            }
         }
         while (j2 < A_diag_i[i1 + 1])
         {
            w_row_sum[i] += A_diag_data[j2];
            j2++;
         }
         j2 = A_offd_i[i1];
         for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
         {
            j1 = S_offd_j[j];
            while (A_offd_j[j2] != j1)
            {
               w_row_sum[i] += A_offd_data[j2];
               j2++;
            }
            if (pass_marker_offd[j1] == color && A_offd_j[j2] == j1)
            {
               Q_offd_j[cnt_offd] = fine_to_coarse_offd[j1];
               Q_offd_data[cnt_offd++] = A_offd_data[j2++];
            }
            else
            {
               w_row_sum[i] += A_offd_data[j2];
               j2++;
            }
         }
         while (j2 < A_offd_i[i1 + 1])
         {
            w_row_sum[i] += A_offd_data[j2];
            j2++;
         }
      }
   }

   Q = nalu_hypre_ParCSRMatrixCreate(comm,
                                total_global_fpts,
                                total_global_cpts,
                                f_pts_starts,
                                c_pts_starts,
                                num_cols_offd_Q,
                                Q_diag_i[num_points],
                                Q_offd_i[num_points]);

   Q_diag = nalu_hypre_ParCSRMatrixDiag(Q);
   nalu_hypre_CSRMatrixData(Q_diag) = Q_diag_data;
   nalu_hypre_CSRMatrixI(Q_diag) = Q_diag_i;
   nalu_hypre_CSRMatrixJ(Q_diag) = Q_diag_j;
   Q_offd = nalu_hypre_ParCSRMatrixOffd(Q);
   nalu_hypre_CSRMatrixData(Q_offd) = Q_offd_data;
   nalu_hypre_CSRMatrixI(Q_offd) = Q_offd_i;
   nalu_hypre_CSRMatrixJ(Q_offd) = Q_offd_j;
   nalu_hypre_ParCSRMatrixColMapOffd(Q) = col_map_offd_Q;

   nalu_hypre_CSRMatrixMemoryLocation(Q_diag) = NALU_HYPRE_MEMORY_HOST;
   nalu_hypre_CSRMatrixMemoryLocation(Q_offd) = NALU_HYPRE_MEMORY_HOST;

   /* free stuff */
   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_convert, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_convert_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_buf_data, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_MatvecCommPkgCreate(Q);

   Pi = nalu_hypre_ParMatmul(Q, P);

   Pi_diag = nalu_hypre_ParCSRMatrixDiag(Pi);
   Pi_diag_data = nalu_hypre_CSRMatrixData(Pi_diag);
   Pi_diag_i = nalu_hypre_CSRMatrixI(Pi_diag);
   Pi_offd = nalu_hypre_ParCSRMatrixOffd(Pi);
   Pi_offd_data = nalu_hypre_CSRMatrixData(Pi_offd);
   Pi_offd_i = nalu_hypre_CSRMatrixI(Pi_offd);

   row_sums_C = nalu_hypre_CTAlloc(NALU_HYPRE_Real, num_points, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_points; i++)
   {
      NALU_HYPRE_Real diagonal, value;
      i1 = pass_order[i];
      diagonal = A_diag_data[A_diag_i[i1]];
      for (j = Pi_diag_i[i]; j < Pi_diag_i[i + 1]; j++)
      {
         row_sums_C[i] += Pi_diag_data[j];
      }
      for (j = Pi_offd_i[i]; j < Pi_offd_i[i + 1]; j++)
      {
         row_sums_C[i] += Pi_offd_data[j];
      }
      value = row_sums_C[i] * diagonal;
      row_sums_C[i] += w_row_sum[i];
      if (value != 0)
      {
         row_sums_C[i] /= value;
      }
      for (j = Pi_diag_i[i]; j < Pi_diag_i[i + 1]; j++)
      {
         Pi_diag_data[j] = -Pi_diag_data[j] * row_sums_C[i];
      }
      for (j = Pi_offd_i[i]; j < Pi_offd_i[i + 1]; j++)
      {
         Pi_offd_data[j] = -Pi_offd_data[j] * row_sums_C[i];
      }
   }

   nalu_hypre_ParCSRMatrixDestroy(Q);
   //nalu_hypre_TFree(row_sums, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(row_sums_C, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(w_row_sum, NALU_HYPRE_MEMORY_HOST);

   *Pi_ptr = Pi;

   return nalu_hypre_error_flag;
}


NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildModMultipass( nalu_hypre_ParCSRMatrix  *A,
                                  NALU_HYPRE_Int           *CF_marker,
                                  nalu_hypre_ParCSRMatrix  *S,
                                  NALU_HYPRE_BigInt        *num_cpts_global,
                                  NALU_HYPRE_Real           trunc_factor,
                                  NALU_HYPRE_Int            P_max_elmts,
                                  NALU_HYPRE_Int            interp_type,
                                  NALU_HYPRE_Int            num_functions,
                                  NALU_HYPRE_Int           *dof_func,
                                  nalu_hypre_ParCSRMatrix **P_ptr )
{
   nalu_hypre_GpuProfilingPushRange("ModMultipass");

   NALU_HYPRE_Int ierr = 0;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(A),
                                                      nalu_hypre_ParCSRMatrixMemoryLocation(S) );
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      ierr = nalu_hypre_BoomerAMGBuildModMultipassDevice( A, CF_marker, S, num_cpts_global,
                                                     trunc_factor, P_max_elmts,
                                                     interp_type, num_functions,
                                                     dof_func, P_ptr);
   }
   else
#endif
   {
      ierr = nalu_hypre_BoomerAMGBuildModMultipassHost( A, CF_marker, S, num_cpts_global,
                                                   trunc_factor, P_max_elmts,
                                                   interp_type, num_functions,
                                                   dof_func, P_ptr);
   }

   nalu_hypre_GpuProfilingPopRange();

   return ierr;
}

