/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

NALU_HYPRE_Int
nalu_hypre_GetCommPkgRTFromCommPkgA( nalu_hypre_ParCSRMatrix *RT,
                                nalu_hypre_ParCSRMatrix *A,
                                NALU_HYPRE_Int *fine_to_coarse,
                                NALU_HYPRE_Int *tmp_map_offd)
{
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(RT);
   nalu_hypre_ParCSRCommPkg *comm_pkg_A = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle *comm_handle;
   NALU_HYPRE_Int num_recvs_A = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg_A);
   NALU_HYPRE_Int *recv_procs_A = nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg_A);
   NALU_HYPRE_Int *recv_vec_starts_A = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
   NALU_HYPRE_Int num_sends_A = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg_A);
   NALU_HYPRE_Int *send_procs_A = nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg_A);
   NALU_HYPRE_Int *send_map_starts_A = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg_A);

   nalu_hypre_ParCSRCommPkg *comm_pkg = NULL;
   NALU_HYPRE_Int num_recvs_RT;
   NALU_HYPRE_Int *recv_procs_RT;
   NALU_HYPRE_Int *recv_vec_starts_RT;
   NALU_HYPRE_Int num_sends_RT;
   NALU_HYPRE_Int *send_procs_RT;
   NALU_HYPRE_Int *send_map_starts_RT;
   NALU_HYPRE_Int *send_map_elmts_RT;

   NALU_HYPRE_BigInt *col_map_offd_RT = nalu_hypre_ParCSRMatrixColMapOffd(RT);
   NALU_HYPRE_Int num_cols_offd_RT = nalu_hypre_CSRMatrixNumCols( nalu_hypre_ParCSRMatrixOffd(RT));
   NALU_HYPRE_BigInt first_col_diag = nalu_hypre_ParCSRMatrixFirstColDiag(RT);
   NALU_HYPRE_Int n_fine = nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(A));
   NALU_HYPRE_Int num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(A));
   NALU_HYPRE_BigInt *fine_to_coarse_offd = NULL;
   NALU_HYPRE_BigInt *big_buf_data = NULL;
   NALU_HYPRE_BigInt *send_big_elmts = NULL;
   NALU_HYPRE_BigInt my_first_cpt;

   NALU_HYPRE_Int i, j;
   NALU_HYPRE_Int vec_len, vec_start;
   NALU_HYPRE_Int num_procs, my_id;
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int num_requests;
   NALU_HYPRE_Int offd_col, proc_num;
   NALU_HYPRE_Int num_threads = nalu_hypre_NumThreads();
   NALU_HYPRE_Int size, rest, ns, ne, start;
   NALU_HYPRE_Int index;

   NALU_HYPRE_Int *proc_mark;
   NALU_HYPRE_Int *change_array;
   NALU_HYPRE_Int *coarse_counter;
   NALU_HYPRE_Int coarse_shift;

   nalu_hypre_MPI_Request *requests;
   nalu_hypre_MPI_Status *status;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /*--------------------------------------------------------------------------
    * determine num_recvs, recv_procs and recv_vec_starts for RT
    *--------------------------------------------------------------------------*/

   proc_mark = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs_A, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_recvs_A; i++)
   {
      proc_mark[i] = 0;
   }

   proc_num = 0;
   num_recvs_RT = 0;
   if (num_cols_offd_RT)
   {
      for (i = 0; i < num_recvs_A; i++)
      {
         for (j = recv_vec_starts_A[i]; j < recv_vec_starts_A[i + 1]; j++)
         {
            offd_col = tmp_map_offd[proc_num];
            if (offd_col == j)
            {
               proc_mark[i]++;
               proc_num++;
               if (proc_num == num_cols_offd_RT) { break; }
            }
         }
         if (proc_mark[i]) { num_recvs_RT++; }
         if (proc_num == num_cols_offd_RT) { break; }
      }
   }

   fine_to_coarse_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
   big_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, send_map_starts_A[num_sends_A], NALU_HYPRE_MEMORY_HOST);
   coarse_counter = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads, NALU_HYPRE_MEMORY_HOST);

   my_first_cpt = nalu_hypre_ParCSRMatrixColStarts(RT)[0];

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,ns,ne,size,rest,coarse_shift) NALU_HYPRE_SMP_SCHEDULE
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

   index = 0;
   for (i = 0; i < num_sends_A; i++)
   {
      start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i);
      for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg_A, i + 1); j++)
         big_buf_data[index++] = my_first_cpt +
                                 (NALU_HYPRE_BigInt)fine_to_coarse[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg_A, j)];
   }

   comm_handle = nalu_hypre_ParCSRCommHandleCreate( 21, comm_pkg_A, big_buf_data,
                                               fine_to_coarse_offd);

   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   for (i = 0; i < num_cols_offd_RT; i++)
   {
      col_map_offd_RT[i] = fine_to_coarse_offd[tmp_map_offd[i]];
   }

   nalu_hypre_TFree(big_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(coarse_counter, NALU_HYPRE_MEMORY_HOST);
   //nalu_hypre_TFree(tmp_map_offd, NALU_HYPRE_MEMORY_HOST);

   recv_procs_RT = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_recvs_RT, NALU_HYPRE_MEMORY_HOST);
   recv_vec_starts_RT = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs_RT + 1, NALU_HYPRE_MEMORY_HOST);

   j = 0;
   recv_vec_starts_RT[0] = 0;
   for (i = 0; i < num_recvs_A; i++)
   {
      if (proc_mark[i])
      {
         recv_procs_RT[j] = recv_procs_A[i];
         recv_vec_starts_RT[j + 1] = recv_vec_starts_RT[j] + proc_mark[i];
         j++;
      }
   }

   /*--------------------------------------------------------------------------
    * send num_changes to recv_procs_A and receive change_array from send_procs_A
    *--------------------------------------------------------------------------*/

   num_requests = num_recvs_A + num_sends_A;
   requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request,  num_requests, NALU_HYPRE_MEMORY_HOST);
   status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,  num_requests, NALU_HYPRE_MEMORY_HOST);

   change_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends_A, NALU_HYPRE_MEMORY_HOST);

   j = 0;
   for (i = 0; i < num_sends_A; i++)
      nalu_hypre_MPI_Irecv(&change_array[i], 1, NALU_HYPRE_MPI_INT, send_procs_A[i], 0, comm,
                      &requests[j++]);

   for (i = 0; i < num_recvs_A; i++)
      nalu_hypre_MPI_Isend(&proc_mark[i], 1, NALU_HYPRE_MPI_INT, recv_procs_A[i], 0, comm,
                      &requests[j++]);

   nalu_hypre_MPI_Waitall(num_requests, requests, status);

   nalu_hypre_TFree(proc_mark, NALU_HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------------
    * if change_array[i] is 0 , omit send_procs_A[i] in send_procs_RT
    *--------------------------------------------------------------------------*/

   num_sends_RT = 0;
   for (i = 0; i < num_sends_A; i++)
      if (change_array[i])
      {
         num_sends_RT++;
      }

   send_procs_RT = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends_RT, NALU_HYPRE_MEMORY_HOST);
   send_map_starts_RT = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends_RT + 1, NALU_HYPRE_MEMORY_HOST);

   j = 0;
   send_map_starts_RT[0] = 0;
   for (i = 0; i < num_sends_A; i++)
   {
      if (change_array[i])
      {
         send_procs_RT[j] = send_procs_A[i];
         send_map_starts_RT[j + 1] = send_map_starts_RT[j] + change_array[i];
         j++;
      }
   }

   /*--------------------------------------------------------------------------
    * generate send_map_elmts
    *--------------------------------------------------------------------------*/

   send_map_elmts_RT = nalu_hypre_CTAlloc(NALU_HYPRE_Int, send_map_starts_RT[num_sends_RT], NALU_HYPRE_MEMORY_HOST);
   send_big_elmts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, send_map_starts_RT[num_sends_RT], NALU_HYPRE_MEMORY_HOST);

   j = 0;
   for (i = 0; i < num_sends_RT; i++)
   {
      vec_start = send_map_starts_RT[i];
      vec_len = send_map_starts_RT[i + 1] - vec_start;
      nalu_hypre_MPI_Irecv(&send_big_elmts[vec_start], vec_len, NALU_HYPRE_MPI_BIG_INT,
                      send_procs_RT[i], 0, comm, &requests[j++]);
   }

   for (i = 0; i < num_recvs_RT; i++)
   {
      vec_start = recv_vec_starts_RT[i];
      vec_len = recv_vec_starts_RT[i + 1] - vec_start;
      nalu_hypre_MPI_Isend(&col_map_offd_RT[vec_start], vec_len, NALU_HYPRE_MPI_BIG_INT,
                      recv_procs_RT[i], 0, comm, &requests[j++]);
   }

   nalu_hypre_MPI_Waitall(j, requests, status);

   for (i = 0; i < send_map_starts_RT[num_sends_RT]; i++)
   {
      send_map_elmts_RT[i] = (NALU_HYPRE_Int)(send_big_elmts[i] - first_col_diag);
   }

   /* Create and fill communication package */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs_RT, recv_procs_RT, recv_vec_starts_RT,
                                    num_sends_RT, send_procs_RT, send_map_starts_RT,
                                    send_map_elmts_RT,
                                    &comm_pkg);

   nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_big_elmts, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParCSRMatrixCommPkg(RT) = comm_pkg;
   nalu_hypre_TFree(change_array, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_GenerateSendMapAndCommPkg(MPI_Comm comm, NALU_HYPRE_Int num_sends, NALU_HYPRE_Int num_recvs,
                                NALU_HYPRE_Int *recv_procs, NALU_HYPRE_Int *send_procs,
                                NALU_HYPRE_Int *recv_vec_starts, nalu_hypre_ParCSRMatrix *A)
{
   NALU_HYPRE_Int *send_map_starts;
   NALU_HYPRE_Int *send_map_elmts;
   NALU_HYPRE_Int i, j;
   NALU_HYPRE_Int num_requests = num_sends + num_recvs;
   nalu_hypre_MPI_Request *requests;
   nalu_hypre_MPI_Status *status;
   NALU_HYPRE_Int vec_len, vec_start;
   nalu_hypre_ParCSRCommPkg *comm_pkg = NULL;
   NALU_HYPRE_BigInt *col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(A);
   NALU_HYPRE_BigInt first_col_diag = nalu_hypre_ParCSRMatrixFirstColDiag(A);
   NALU_HYPRE_BigInt *send_big_elmts = NULL;

   /*--------------------------------------------------------------------------
    * generate send_map_starts and send_map_elmts
    *--------------------------------------------------------------------------*/

   requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request, num_requests, NALU_HYPRE_MEMORY_HOST);
   status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status, num_requests, NALU_HYPRE_MEMORY_HOST);
   send_map_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends + 1, NALU_HYPRE_MEMORY_HOST);
   j = 0;
   for (i = 0; i < num_sends; i++)
   {
      nalu_hypre_MPI_Irecv(&send_map_starts[i + 1], 1, NALU_HYPRE_MPI_INT, send_procs[i], 0, comm,
                      &requests[j++]);
   }

   for (i = 0; i < num_recvs; i++)
   {
      vec_len = recv_vec_starts[i + 1] - recv_vec_starts[i];
      nalu_hypre_MPI_Isend(&vec_len, 1, NALU_HYPRE_MPI_INT, recv_procs[i], 0, comm, &requests[j++]);
   }

   nalu_hypre_MPI_Waitall(j, requests, status);

   send_map_starts[0] = 0;
   for (i = 0; i < num_sends; i++)
   {
      send_map_starts[i + 1] += send_map_starts[i];
   }

   send_map_elmts = nalu_hypre_CTAlloc(NALU_HYPRE_Int, send_map_starts[num_sends], NALU_HYPRE_MEMORY_HOST);
   send_big_elmts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, send_map_starts[num_sends], NALU_HYPRE_MEMORY_HOST);

   j = 0;
   for (i = 0; i < num_sends; i++)
   {
      vec_start = send_map_starts[i];
      vec_len = send_map_starts[i + 1] - vec_start;
      nalu_hypre_MPI_Irecv(&send_big_elmts[vec_start], vec_len, NALU_HYPRE_MPI_BIG_INT,
                      send_procs[i], 0, comm, &requests[j++]);
   }

   for (i = 0; i < num_recvs; i++)
   {
      vec_start = recv_vec_starts[i];
      vec_len = recv_vec_starts[i + 1] - vec_start;
      nalu_hypre_MPI_Isend(&col_map_offd[vec_start], vec_len, NALU_HYPRE_MPI_BIG_INT,
                      recv_procs[i], 0, comm, &requests[j++]);
   }

   nalu_hypre_MPI_Waitall(j, requests, status);

   for (i = 0; i < send_map_starts[num_sends]; i++)
   {
      send_map_elmts[i] = (NALU_HYPRE_Int)(send_big_elmts[i] - first_col_diag);
   }

   /* Create and fill communication package */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs, recv_procs, recv_vec_starts,
                                    num_sends, send_procs, send_map_starts,
                                    send_map_elmts,
                                    &comm_pkg);

   nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_big_elmts, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParCSRMatrixCommPkg(A) = comm_pkg;

   return nalu_hypre_error_flag;
}
