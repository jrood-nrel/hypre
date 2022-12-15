/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_block_mv.h"


/* This function is the same as nalu_hypre_GetCommPkgRTFromCommPkgA, except that the
arguments are Block matrices.  We should change the code to take the commpkgs as input
(and a couple of other items) and then we would not need two functions. (Because
the commpkg is not different for a block matrix.) */



NALU_HYPRE_Int
nalu_hypre_GetCommPkgBlockRTFromCommPkgBlockA( nalu_hypre_ParCSRBlockMatrix *RT,
                                          nalu_hypre_ParCSRBlockMatrix *A,
                                          NALU_HYPRE_Int *tmp_map_offd,
                                          NALU_HYPRE_BigInt *fine_to_coarse_offd)
{
   MPI_Comm comm = nalu_hypre_ParCSRBlockMatrixComm(RT);
   nalu_hypre_ParCSRCommPkg *comm_pkg_A = nalu_hypre_ParCSRBlockMatrixCommPkg(A);
   NALU_HYPRE_Int num_recvs_A = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg_A);
   NALU_HYPRE_Int *recv_procs_A = nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg_A);
   NALU_HYPRE_Int *recv_vec_starts_A = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
   NALU_HYPRE_Int num_sends_A = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg_A);
   NALU_HYPRE_Int *send_procs_A = nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg_A);

   nalu_hypre_ParCSRCommPkg *comm_pkg = NULL;
   NALU_HYPRE_Int num_recvs_RT;
   NALU_HYPRE_Int *recv_procs_RT;
   NALU_HYPRE_Int *recv_vec_starts_RT;
   NALU_HYPRE_Int num_sends_RT;
   NALU_HYPRE_Int *send_procs_RT;
   NALU_HYPRE_Int *send_map_starts_RT;
   NALU_HYPRE_Int *send_map_elmts_RT;
   NALU_HYPRE_BigInt *send_big_elmts = NULL;

   NALU_HYPRE_BigInt *col_map_offd_RT = nalu_hypre_ParCSRBlockMatrixColMapOffd(RT);
   NALU_HYPRE_Int num_cols_offd_RT = nalu_hypre_CSRBlockMatrixNumCols( nalu_hypre_ParCSRMatrixOffd(RT));
   NALU_HYPRE_BigInt first_col_diag = nalu_hypre_ParCSRBlockMatrixFirstColDiag(RT);

   NALU_HYPRE_Int i, j;
   NALU_HYPRE_Int vec_len, vec_start;
   NALU_HYPRE_Int num_procs, my_id;
   NALU_HYPRE_Int num_requests;
   NALU_HYPRE_Int offd_col, proc_num;

   NALU_HYPRE_Int *proc_mark;
   NALU_HYPRE_Int *change_array;

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

   for (i = 0; i < num_cols_offd_RT; i++)
   {
      col_map_offd_RT[i] = fine_to_coarse_offd[tmp_map_offd[i]];
   }

   recv_procs_RT = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_recvs_RT, NALU_HYPRE_MEMORY_HOST);
   recv_vec_starts_RT = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs_RT + 1, NALU_HYPRE_MEMORY_HOST);

   j = 0;
   recv_vec_starts_RT[0] = 0;
   for (i = 0; i < num_recvs_A; i++)
      if (proc_mark[i])
      {
         recv_procs_RT[j] = recv_procs_A[i];
         recv_vec_starts_RT[j + 1] = recv_vec_starts_RT[j] + proc_mark[i];
         j++;
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
      if (change_array[i])
      {
         send_procs_RT[j] = send_procs_A[i];
         send_map_starts_RT[j + 1] = send_map_starts_RT[j] + change_array[i];
         j++;
      }

   /*--------------------------------------------------------------------------
    * generate send_map_elmts
    *--------------------------------------------------------------------------*/

   send_big_elmts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, send_map_starts_RT[num_sends_RT], NALU_HYPRE_MEMORY_HOST);
   send_map_elmts_RT = nalu_hypre_CTAlloc(NALU_HYPRE_Int, send_map_starts_RT[num_sends_RT], NALU_HYPRE_MEMORY_HOST);

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

   /* Create communication package */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs_RT,
                                    recv_procs_RT,
                                    recv_vec_starts_RT,
                                    num_sends_RT,
                                    send_procs_RT,
                                    send_map_starts_RT,
                                    send_map_elmts_RT,
                                    &comm_pkg);

   nalu_hypre_ParCSRBlockMatrixCommPkg(RT) = comm_pkg;

   /* Free memory */
   nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_big_elmts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(change_array, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

