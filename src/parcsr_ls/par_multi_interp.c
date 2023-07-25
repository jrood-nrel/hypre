/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ParAMGBuildMultipass
 * This routine implements Stuben's direct interpolation with multiple passes.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildMultipassHost( nalu_hypre_ParCSRMatrix  *A,
                                   NALU_HYPRE_Int           *CF_marker,
                                   nalu_hypre_ParCSRMatrix  *S,
                                   NALU_HYPRE_BigInt        *num_cpts_global,
                                   NALU_HYPRE_Int            num_functions,
                                   NALU_HYPRE_Int           *dof_func,
                                   NALU_HYPRE_Int            debug_flag,
                                   NALU_HYPRE_Real           trunc_factor,
                                   NALU_HYPRE_Int            P_max_elmts,
                                   NALU_HYPRE_Int            weight_option,
                                   nalu_hypre_ParCSRMatrix **P_ptr )
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_MULTIPASS_INTERP] -= nalu_hypre_MPI_Wtime();
#endif

   MPI_Comm                comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg    *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(S);
   nalu_hypre_ParCSRCommHandle *comm_handle;
   nalu_hypre_ParCSRCommPkg    *tmp_comm_pkg = NULL;

   NALU_HYPRE_MemoryLocation memory_location_P = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real      *A_offd_data = NULL;
   NALU_HYPRE_Int       *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j = NULL;
   //NALU_HYPRE_BigInt    *col_map_offd_A = nalu_hypre_ParCSRMatrixColMapOffd(A);
   NALU_HYPRE_Int        num_cols_offd_A = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_CSRMatrix *S_diag = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int       *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int       *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);

   nalu_hypre_CSRMatrix *S_offd = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int       *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int       *S_offd_j = NULL;
   /*NALU_HYPRE_BigInt    *col_map_offd_S = nalu_hypre_ParCSRMatrixColMapOffd(S);
   NALU_HYPRE_Int        num_cols_offd_S = nalu_hypre_CSRMatrixNumCols(S_offd);
   NALU_HYPRE_BigInt    *col_map_offd = NULL;*/
   NALU_HYPRE_Int        num_cols_offd;

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

   NALU_HYPRE_Int        num_sends = 0;
   NALU_HYPRE_Int       *int_buf_data = NULL;
   NALU_HYPRE_BigInt    *big_buf_data = NULL;
   NALU_HYPRE_Int       *send_map_start;
   NALU_HYPRE_Int       *send_map_elmt;
   NALU_HYPRE_Int       *send_procs;
   NALU_HYPRE_Int        num_recvs = 0;
   NALU_HYPRE_Int       *recv_vec_start;
   NALU_HYPRE_Int       *recv_procs;
   NALU_HYPRE_Int       *new_recv_vec_start = NULL;
   NALU_HYPRE_Int      **Pext_send_map_start = NULL;
   NALU_HYPRE_Int      **Pext_recv_vec_start = NULL;
   NALU_HYPRE_Int       *Pext_start = NULL;
   NALU_HYPRE_Int       *P_ncols = NULL;

   NALU_HYPRE_Int       *CF_marker_offd = NULL;
   NALU_HYPRE_Int       *dof_func_offd = NULL;
   NALU_HYPRE_Int       *P_marker;
   NALU_HYPRE_Int       *P_marker_offd = NULL;
   NALU_HYPRE_Int       *C_array;
   NALU_HYPRE_Int       *C_array_offd = NULL;
   NALU_HYPRE_Int       *pass_array = NULL; /* contains points ordered according to pass */
   NALU_HYPRE_Int       *pass_pointer = NULL; /* pass_pointer[j] contains pointer to first
                                                  point of pass j contained in pass_array */
   NALU_HYPRE_Int       *P_diag_start;
   NALU_HYPRE_Int       *P_offd_start = NULL;
   NALU_HYPRE_Int      **P_diag_pass;
   NALU_HYPRE_Int      **P_offd_pass = NULL;
   NALU_HYPRE_Int      **Pext_pass = NULL;
   NALU_HYPRE_BigInt    *big_temp_pass = NULL;
   NALU_HYPRE_BigInt   **new_elmts = NULL; /* new neighbors generated in each pass */
   NALU_HYPRE_Int       *new_counter = NULL; /* contains no. of new neighbors for
                                           each pass */
   NALU_HYPRE_Int       *loc = NULL; /* contains locations for new neighbor
                                   connections in int_o_buffer to avoid searching */
   NALU_HYPRE_Int       *Pext_i = NULL; /*contains P_diag_i and P_offd_i info for nonzero
                                     cols of off proc neighbors */
   NALU_HYPRE_BigInt    *Pext_send_buffer = NULL; /* used to collect global nonzero
                                                col ids in P_diag for send_map_elmts */

   NALU_HYPRE_Int       *map_S_to_new = NULL;
   NALU_HYPRE_BigInt    *new_col_map_offd = NULL;
   NALU_HYPRE_BigInt    *col_map_offd_P = NULL;
   NALU_HYPRE_Int       *permute = NULL;
   NALU_HYPRE_BigInt    *big_permute = NULL;

   NALU_HYPRE_Int        cnt;
   NALU_HYPRE_Int        cnt_nz;
   NALU_HYPRE_Int        total_nz;
   NALU_HYPRE_Int        pass;
   NALU_HYPRE_Int        num_passes;
   NALU_HYPRE_Int        max_num_passes = 10;

   NALU_HYPRE_Int        n_fine;
   NALU_HYPRE_Int        n_coarse = 0;
   NALU_HYPRE_Int        n_coarse_offd = 0;
   NALU_HYPRE_Int        n_SF = 0;

   NALU_HYPRE_Int       *fine_to_coarse = NULL;
   NALU_HYPRE_BigInt    *fine_to_coarse_offd = NULL;

   NALU_HYPRE_Int       *assigned = NULL;
   NALU_HYPRE_Int       *assigned_offd = NULL;

   NALU_HYPRE_Real      *Pext_send_data = NULL;
   NALU_HYPRE_Real      *Pext_data = NULL;

   NALU_HYPRE_Real       sum_C, sum_N;
   NALU_HYPRE_Real       sum_C_pos, sum_C_neg;
   NALU_HYPRE_Real       sum_N_pos, sum_N_neg;
   NALU_HYPRE_Real       diagonal;
   NALU_HYPRE_Real       alfa = 1.0;
   NALU_HYPRE_Real       beta = 1.0;
   NALU_HYPRE_Int        j_start;
   NALU_HYPRE_Int        j_end;

   NALU_HYPRE_Int        i, i1;
   NALU_HYPRE_Int        j, j1;
   NALU_HYPRE_Int        k, k1, k2, k3;
   NALU_HYPRE_BigInt     big_k1;
   NALU_HYPRE_Int        pass_array_size;
   NALU_HYPRE_BigInt     global_pass_array_size;
   NALU_HYPRE_BigInt     local_pass_array_size;
   NALU_HYPRE_Int        my_id, num_procs;
   NALU_HYPRE_Int        index, start;
   NALU_HYPRE_BigInt     my_first_cpt;
   NALU_HYPRE_BigInt     total_global_cpts;
   NALU_HYPRE_Int        p_cnt;
   NALU_HYPRE_Int        total_nz_offd;
   NALU_HYPRE_Int        cnt_nz_offd;
   NALU_HYPRE_Int        cnt_offd, cnt_new;
   NALU_HYPRE_Int        no_break;
   NALU_HYPRE_Int        not_found;
   NALU_HYPRE_Int        Pext_send_size;
   NALU_HYPRE_Int        Pext_recv_size;
   NALU_HYPRE_Int        old_Pext_send_size;
   NALU_HYPRE_Int        old_Pext_recv_size;
   NALU_HYPRE_Int        P_offd_size = 0;
   NALU_HYPRE_Int        local_index = -1;
   NALU_HYPRE_Int        new_num_cols_offd = 0;
   NALU_HYPRE_Int        num_cols_offd_P;

   /* Threading variables */
   NALU_HYPRE_Int my_thread_num, num_threads, thread_start, thread_stop;
   NALU_HYPRE_Int pass_length;
   NALU_HYPRE_Int *tmp_marker, *tmp_marker_offd;
   NALU_HYPRE_Int *tmp_array,  *tmp_array_offd;
   NALU_HYPRE_Int * max_num_threads = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int * cnt_nz_per_thread;
   NALU_HYPRE_Int * cnt_nz_offd_per_thread;

   /* NALU_HYPRE_Real wall_time;
      wall_time = nalu_hypre_MPI_Wtime(); */

   /* Initialize threading variables */
   max_num_threads[0] = nalu_hypre_NumThreads();
   cnt_nz_per_thread = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_num_threads[0], NALU_HYPRE_MEMORY_HOST);
   cnt_nz_offd_per_thread = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_num_threads[0], NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_threads[0]; i++)
   {
      cnt_nz_offd_per_thread[i] = 0;
      cnt_nz_per_thread[i] = 0;
   }


   /*-----------------------------------------------------------------------
    *  Access the CSR vectors for A and S. Also get size of fine grid.
    *-----------------------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   my_first_cpt = num_cpts_global[0];
   /*   total_global_cpts = 0; */
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   if (!comm_pkg)
   {
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
      if (!comm_pkg)
      {
         nalu_hypre_MatvecCommPkgCreate(A);

         comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
      }
   }

   //col_map_offd = col_map_offd_A;
   num_cols_offd = num_cols_offd_A;

   if (num_cols_offd_A)
   {
      A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
      A_offd_j    = nalu_hypre_CSRMatrixJ(A_offd);
   }

   if (num_cols_offd)
   {
      S_offd_j    = nalu_hypre_CSRMatrixJ(S_offd);
   }

   n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);

   /*-----------------------------------------------------------------------
    *  Intialize counters and allocate mapping vector.
    *-----------------------------------------------------------------------*/

   if (n_fine) { fine_to_coarse = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine, NALU_HYPRE_MEMORY_HOST); }

   n_coarse = 0;
   n_SF = 0;
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) reduction(+:n_coarse,n_SF ) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < n_fine; i++)
      if (CF_marker[i] == 1) { n_coarse++; }
      else if (CF_marker[i] == -3) { n_SF++; }

   pass_array_size = n_fine - n_coarse - n_SF;
   if (pass_array_size) { pass_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  pass_array_size, NALU_HYPRE_MEMORY_HOST); }
   pass_pointer = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_num_passes + 1, NALU_HYPRE_MEMORY_HOST);
   if (n_fine) { assigned = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine, NALU_HYPRE_MEMORY_HOST); }
   P_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine + 1, memory_location_P);
   P_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine + 1, memory_location_P);
   if (n_coarse) { C_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_coarse, NALU_HYPRE_MEMORY_HOST); }

   if (num_cols_offd)
   {
      CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      if (num_functions > 1) { dof_func_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST); }
   }

   if (num_procs > 1)
   {
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_procs = nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg);
      send_map_start = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmt = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
      num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_procs = nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      recv_vec_start = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      if (send_map_start[num_sends])
      {
         int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  send_map_start[num_sends], NALU_HYPRE_MEMORY_HOST);
         big_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  send_map_start[num_sends], NALU_HYPRE_MEMORY_HOST);
      }
   }


   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = send_map_start[i];
      for (j = start; j < send_map_start[i + 1]; j++)
      {
         int_buf_data[index++] = CF_marker[send_map_elmt[j]];
      }
   }
   if (num_procs > 1)
   {
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                 CF_marker_offd);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = send_map_start[i];
         for (j = start; j < send_map_start[i + 1]; j++)
         {
            int_buf_data[index++] = dof_func[send_map_elmt[j]];
         }
      }
      if (num_procs > 1)
      {
         comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                    dof_func_offd);
         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      }
   }

   n_coarse_offd = 0;
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) reduction(+:n_coarse_offd) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_cols_offd; i++)
      if (CF_marker_offd[i] == 1) { n_coarse_offd++; }

   if (num_cols_offd)
   {
      assigned_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      map_S_to_new = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      fine_to_coarse_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      new_col_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  n_coarse_offd, NALU_HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------------------
    *  First Pass: determine the maximal size of P, and elementsPerRow[i].
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Assigned points are points for which we know an interpolation
    *  formula already, and which are thus available to interpolate from.
    *  assigned[i]=0 for C points, and 1, 2, 3, ... for F points, depending
    *  in which pass their interpolation formula is determined.
    *
    *  pass_array contains the points ordered according to its pass, i.e.
    *  |  C-points   |  points of pass 1 | points of pass 2 | ....
    * C_points are points 0 through pass_pointer[1]-1,
    * points of pass k  (0 < k < num_passes) are contained in points
    * pass_pointer[k] through pass_pointer[k+1]-1 of pass_array .
    *
    * pass_array is also used to avoid going through all points for each pass,
    * i,e. at the bginning it contains all points in descending order starting
    * with n_fine-1. Then starting from the last point, we evaluate whether
    * it is a C_point (pass 0). If it is the point is brought to the front
    * and the length of the points to be searched is shortened.  This is
    * done until the parameter cnt (which determines the first point of
    * pass_array to be searched) becomes n_fine. Then all points have been
    * assigned a pass number.
    *-----------------------------------------------------------------------*/


   cnt = 0;
   p_cnt = pass_array_size - 1;
   P_diag_i[0] = 0;
   P_offd_i[0] = 0;
   for (i = 0; i < n_fine; i++)
   {
      if (CF_marker[i] == 1)
      {
         fine_to_coarse[i] = cnt; /* this C point is assigned index
                                     coarse_counter on coarse grid,
                                     and in column of P */
         C_array[cnt++] = i;
         assigned[i] = 0;
         P_diag_i[i + 1] = 1; /* one element in row i1 of P */
         P_offd_i[i + 1] = 0;
      }
      else if (CF_marker[i] == -1)
      {
         pass_array[p_cnt--] = i;
         P_diag_i[i + 1] = 0;
         P_offd_i[i + 1] = 0;
         assigned[i] = -1;
         fine_to_coarse[i] = -1;
      }
      else
      {
         P_diag_i[i + 1] = 0;
         P_offd_i[i + 1] = 0;
         assigned[i] = -1;
         fine_to_coarse[i] = -1;
      }
   }

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = send_map_start[i];
      for (j = start; j < send_map_start[i + 1]; j++)
      {
         big_buf_data[index] = (NALU_HYPRE_BigInt)fine_to_coarse[send_map_elmt[j]];
         if (big_buf_data[index] > -1)
         {
            big_buf_data[index] += my_first_cpt;
         }
         index++;
      }
   }
   if (num_procs > 1)
   {
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(21, comm_pkg, big_buf_data,
                                                 fine_to_coarse_offd);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   new_recv_vec_start = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_recvs + 1, NALU_HYPRE_MEMORY_HOST);

   if (n_coarse_offd)
   {
      C_array_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_coarse_offd, NALU_HYPRE_MEMORY_HOST);
   }

   cnt = 0;
   new_recv_vec_start[0] = 0;
   for (j = 0; j < num_recvs; j++)
   {
      for (i = recv_vec_start[j]; i < recv_vec_start[j + 1]; i++)
      {
         if (CF_marker_offd[i] == 1)
         {
            map_S_to_new[i] = cnt;
            C_array_offd[cnt] = i;
            new_col_map_offd[cnt++] = fine_to_coarse_offd[i];
            assigned_offd[i] = 0;
         }
         else
         {
            assigned_offd[i] = -1;
            map_S_to_new[i] = -1;
         }
      }
      new_recv_vec_start[j + 1] = cnt;
   }

   cnt = 0;
   nalu_hypre_TFree(fine_to_coarse_offd, NALU_HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Mark all local neighbors of C points as 'assigned'.
    *-----------------------------------------------------------------------*/

   pass_pointer[0] = 0;
   pass_pointer[1] = 0;
   total_nz = n_coarse;  /* accumulates total number of nonzeros in P_diag */
   total_nz_offd = 0; /* accumulates total number of nonzeros in P_offd */

   cnt = 0;
   cnt_offd = 0;
   cnt_nz = 0;
   cnt_nz_offd = 0;
   for (i = pass_array_size - 1; i > cnt - 1; i--)
   {
      i1 = pass_array[i];
      for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
      {
         j1 = S_diag_j[j];
         if (CF_marker[j1] == 1)
         {
            P_diag_i[i1 + 1]++;
            cnt_nz++;
            assigned[i1] = 1;
         }
      }
      for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
      {
         j1 = S_offd_j[j];
         if (CF_marker_offd[j1] == 1)
         {
            P_offd_i[i1 + 1]++;
            cnt_nz_offd++;
            assigned[i1] = 1;
         }
      }
      if (assigned[i1] == 1)
      {
         pass_array[i++] = pass_array[cnt];
         pass_array[cnt++] = i1;
      }
   }

   pass_pointer[2] = cnt;

   /*-----------------------------------------------------------------------
    *  All local neighbors are assigned, now need to exchange the boundary
    *  info for assigned strong neighbors.
    *-----------------------------------------------------------------------*/

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = send_map_start[i];
      for (j = start; j < send_map_start[i + 1]; j++)
      {    int_buf_data[index++] = assigned[send_map_elmt[j]]; }
   }
   if (num_procs > 1)
   {
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                 assigned_offd);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   /*-----------------------------------------------------------------------
    *  Now we need to determine strong neighbors of points of pass 1, etc.
    *  we need to update assigned_offd after each pass
    *-----------------------------------------------------------------------*/

   pass = 2;
   local_pass_array_size = (NALU_HYPRE_BigInt)(pass_array_size - cnt);
   nalu_hypre_MPI_Allreduce(&local_pass_array_size, &global_pass_array_size, 1, NALU_HYPRE_MPI_BIG_INT,
                       nalu_hypre_MPI_SUM, comm);
   while (global_pass_array_size && pass < max_num_passes)
   {
      for (i = pass_array_size - 1; i > cnt - 1; i--)
      {
         i1 = pass_array[i];
         no_break = 1;
         for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
         {
            j1 = S_diag_j[j];
            if (assigned[j1] == pass - 1)
            {
               pass_array[i++] = pass_array[cnt];
               pass_array[cnt++] = i1;
               assigned[i1] = pass;
               no_break = 0;
               break;
            }
         }
         if (no_break)
         {
            for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
            {
               j1 = S_offd_j[j];
               if (assigned_offd[j1] == pass - 1)
               {
                  pass_array[i++] = pass_array[cnt];
                  pass_array[cnt++] = i1;
                  assigned[i1] = pass;
                  break;
               }
            }
         }
      }
      /*nalu_hypre_printf("pass %d  remaining points %d \n", pass, local_pass_array_size);*/

      pass++;
      pass_pointer[pass] = cnt;

      local_pass_array_size = (NALU_HYPRE_BigInt)(pass_array_size - cnt);
      nalu_hypre_MPI_Allreduce(&local_pass_array_size, &global_pass_array_size, 1, NALU_HYPRE_MPI_BIG_INT,
                          nalu_hypre_MPI_SUM, comm);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = send_map_start[i];
         for (j = start; j < send_map_start[i + 1]; j++)
         {   int_buf_data[index++] = assigned[send_map_elmt[j]]; }
      }
      if (num_procs > 1)
      {
         comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                    assigned_offd);
         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      }
   }

   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_buf_data, NALU_HYPRE_MEMORY_HOST);

   num_passes = pass;

   P_diag_pass = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, num_passes,
                               NALU_HYPRE_MEMORY_HOST); /* P_diag_pass[i] will contain
                                                                              all column numbers for points of pass i */

   P_diag_pass[1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, cnt_nz, NALU_HYPRE_MEMORY_HOST);

   P_diag_start = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine, NALU_HYPRE_MEMORY_HOST); /* P_diag_start[i] contains
                                                                           pointer to begin of column numbers in P_pass for point i,
                                                                           P_diag_i[i+1] contains number of columns for point i */

   P_offd_start = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine, NALU_HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      P_offd_pass = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, num_passes, NALU_HYPRE_MEMORY_HOST);

      if (cnt_nz_offd)
      {
         P_offd_pass[1] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, cnt_nz_offd, NALU_HYPRE_MEMORY_HOST);
      }
      else
      {
         P_offd_pass[1] = NULL;
      }

      new_elmts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt*, num_passes, NALU_HYPRE_MEMORY_HOST);

      new_counter = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_passes + 1, NALU_HYPRE_MEMORY_HOST);

      new_counter[0] = 0;
      new_counter[1] = n_coarse_offd;
      new_num_cols_offd = n_coarse_offd;

      new_elmts[0] = new_col_map_offd;
   }

   /*-----------------------------------------------------------------------
    *  Pass 1: now we consider points of pass 1, with strong C_neighbors,
    *-----------------------------------------------------------------------*/

   cnt_nz = 0;
   cnt_nz_offd = 0;
   /* JBS: Possible candidate for threading */
   for (i = pass_pointer[1]; i < pass_pointer[2]; i++)
   {
      i1 = pass_array[i];
      P_diag_start[i1] = cnt_nz;
      P_offd_start[i1] = cnt_nz_offd;
      for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
      {
         j1 = S_diag_j[j];
         if (CF_marker[j1] == 1)
         {   P_diag_pass[1][cnt_nz++] = fine_to_coarse[j1]; }
      }
      for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
      {
         j1 = S_offd_j[j];
         if (CF_marker_offd[j1] == 1)
         {   P_offd_pass[1][cnt_nz_offd++] = map_S_to_new[j1]; }
      }
   }


   total_nz += cnt_nz;
   total_nz_offd += cnt_nz_offd;

   if (num_procs > 1)
   {
      Pext_send_map_start = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, num_passes, NALU_HYPRE_MEMORY_HOST);
      Pext_recv_vec_start = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, num_passes, NALU_HYPRE_MEMORY_HOST);
      Pext_pass = nalu_hypre_CTAlloc(NALU_HYPRE_Int*, num_passes, NALU_HYPRE_MEMORY_HOST);
      Pext_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd + 1, NALU_HYPRE_MEMORY_HOST);
      if (num_cols_offd) { Pext_start = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST); }
      if (send_map_start[num_sends])
      {
         P_ncols = nalu_hypre_CTAlloc(NALU_HYPRE_Int, send_map_start[num_sends], NALU_HYPRE_MEMORY_HOST);
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols_offd + 1; i++)
      {   Pext_i[i] = 0; }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < send_map_start[num_sends]; i++)
      {   P_ncols[i] = 0; }
   }

   old_Pext_send_size = 0;
   old_Pext_recv_size = 0;
   for (pass = 2; pass < num_passes; pass++)
   {

      if (num_procs > 1)
      {
         Pext_send_map_start[pass] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends + 1, NALU_HYPRE_MEMORY_HOST);
         Pext_recv_vec_start[pass] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs + 1, NALU_HYPRE_MEMORY_HOST);
         Pext_send_size = 0;
         Pext_send_map_start[pass][0] = 0;

         for (i = 0; i < num_sends; i++)
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(j,j1) reduction(+:Pext_send_size) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (j = send_map_start[i]; j < send_map_start[i + 1]; j++)
            {
               j1 = send_map_elmt[j];
               if (assigned[j1] == pass - 1)
               {
                  P_ncols[j] = P_diag_i[j1 + 1] + P_offd_i[j1 + 1];
                  Pext_send_size += P_ncols[j];
               }
            }
            Pext_send_map_start[pass][i + 1] = Pext_send_size;
         }

         comm_handle = nalu_hypre_ParCSRCommHandleCreate (11, comm_pkg,
                                                     P_ncols, &Pext_i[1]);
         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

         if (Pext_send_size > old_Pext_send_size)
         {
            nalu_hypre_TFree(Pext_send_buffer, NALU_HYPRE_MEMORY_HOST);
            Pext_send_buffer = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  Pext_send_size, NALU_HYPRE_MEMORY_HOST);
         }
         old_Pext_send_size = Pext_send_size;
      }

      cnt_offd = 0;
      for (i = 0; i < num_sends; i++)
      {
         for (j = send_map_start[i]; j < send_map_start[i + 1]; j++)
         {
            j1 = send_map_elmt[j];
            if (assigned[j1] == pass - 1)
            {
               j_start = P_diag_start[j1];
               j_end = j_start + P_diag_i[j1 + 1];
               for (k = j_start; k < j_end; k++)
               {
                  Pext_send_buffer[cnt_offd++] = my_first_cpt
                                                 + (NALU_HYPRE_BigInt) P_diag_pass[pass - 1][k];
               }
               j_start = P_offd_start[j1];
               j_end = j_start + P_offd_i[j1 + 1];
               for (k = j_start; k < j_end; k++)
               {
                  k1 = P_offd_pass[pass - 1][k];
                  k3 = 0;
                  while (k3 < pass - 1)
                  {
                     if (k1 < new_counter[k3 + 1])
                     {
                        k2 = k1 - new_counter[k3];
                        Pext_send_buffer[cnt_offd++] = new_elmts[k3][k2];
                        break;
                     }
                     k3++;
                  }
               }
            }
         }
      }

      if (num_procs > 1)
      {
         Pext_recv_size = 0;
         Pext_recv_vec_start[pass][0] = 0;
         cnt_offd = 0;
         for (i = 0; i < num_recvs; i++)
         {
            for (j = recv_vec_start[i]; j < recv_vec_start[i + 1]; j++)
            {
               if (assigned_offd[j] == pass - 1)
               {
                  Pext_start[j] = cnt_offd;
                  cnt_offd += Pext_i[j + 1];
               }
            }
            Pext_recv_size = cnt_offd;
            Pext_recv_vec_start[pass][i + 1] = Pext_recv_size;
         }

         /* Create temporary communication package */
         nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                          num_recvs, recv_procs, Pext_recv_vec_start[pass],
                                          num_sends, send_procs, Pext_send_map_start[pass],
                                          NULL,
                                          &tmp_comm_pkg);

         if (Pext_recv_size)
         {
            Pext_pass[pass] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  Pext_recv_size, NALU_HYPRE_MEMORY_HOST);
            new_elmts[pass - 1] = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, Pext_recv_size, NALU_HYPRE_MEMORY_HOST);
         }
         else
         {
            Pext_pass[pass] = NULL;
            new_elmts[pass - 1] = NULL;
         }

         if (Pext_recv_size > old_Pext_recv_size)
         {
            nalu_hypre_TFree(loc, NALU_HYPRE_MEMORY_HOST);
            loc = nalu_hypre_CTAlloc(NALU_HYPRE_Int, Pext_recv_size, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(big_temp_pass, NALU_HYPRE_MEMORY_HOST);
            big_temp_pass = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, Pext_recv_size, NALU_HYPRE_MEMORY_HOST);
         }
         old_Pext_recv_size = Pext_recv_size;

         comm_handle = nalu_hypre_ParCSRCommHandleCreate (21, tmp_comm_pkg,
                                                     Pext_send_buffer, big_temp_pass);
         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      cnt_new = 0;
      cnt_offd = 0;
      /* JBS: Possible candidate for threading */
      for (i = 0; i < num_recvs; i++)
      {
         for (j = recv_vec_start[i]; j < recv_vec_start[i + 1]; j++)
         {
            if (assigned_offd[j] == pass - 1)
            {
               for (j1 = cnt_offd; j1 < cnt_offd + Pext_i[j + 1]; j1++)
               {
                  big_k1 = big_temp_pass[j1];
                  k2 = (NALU_HYPRE_Int)(big_k1 - my_first_cpt);
                  if (k2 > -1 && k2 < n_coarse)
                  {  Pext_pass[pass][j1] = -k2 - 1; }
                  else
                  {
                     not_found = 1;
                     k3 = 0;
                     while (k3 < pass - 1 && not_found)
                     {
                        k2 = nalu_hypre_BigBinarySearch(new_elmts[k3], big_k1,
                                                   (new_counter[k3 + 1] - new_counter[k3]));
                        if (k2 > -1)
                        {
                           Pext_pass[pass][j1] = k2 + new_counter[k3];
                           not_found = 0;
                        }
                        else
                        {
                           k3++;
                        }
                     }
                     if (not_found)
                     {
                        new_elmts[pass - 1][cnt_new] = big_k1;
                        loc[cnt_new++] = j1;
                     }
                  }
               }
               cnt_offd += Pext_i[j + 1];
            }
         }
      }

      if (cnt_new)
      {
         nalu_hypre_BigQsortbi(new_elmts[pass - 1], loc, 0, cnt_new - 1);
         cnt = 0;
         local_index = new_counter[pass - 1];
         Pext_pass[pass][loc[0]] = local_index;

         for (i = 1; i < cnt_new; i++)
         {
            if (new_elmts[pass - 1][i] > new_elmts[pass - 1][cnt])
            {
               new_elmts[pass - 1][++cnt] = new_elmts[pass - 1][i];
               local_index++;
            }
            Pext_pass[pass][loc[i]] = local_index;
         }
         new_counter[pass] = local_index + 1;
      }
      else if (num_procs > 1)
      {
         new_counter[pass] = new_counter[pass - 1];
      }

      if (new_num_cols_offd < local_index + 1)
      {    new_num_cols_offd = local_index + 1; }

      pass_length = pass_pointer[pass + 1] - pass_pointer[pass];
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel private(i,my_thread_num,num_threads,thread_start,thread_stop,cnt_nz,cnt_nz_offd,i1,j,j1,j_start,j_end,k1,k,P_marker,P_marker_offd)
#endif
      {
         /* Thread by computing the sparsity structure for this pass only over
          * each thread's range of rows.  Rows are divided up evenly amongst
          * the threads.  The necessary thread-wise temporary arrays, like
          * P_marker, are initialized and de-allocated internally to the
          * parallel region. */

         my_thread_num = nalu_hypre_GetThreadNum();
         num_threads = nalu_hypre_NumActiveThreads();
         thread_start = (pass_length / num_threads) * my_thread_num;
         if (my_thread_num == num_threads - 1)
         {  thread_stop = pass_length; }
         else
         {  thread_stop = (pass_length / num_threads) * (my_thread_num + 1); }
         thread_start += pass_pointer[pass];
         thread_stop += pass_pointer[pass];

         /* Local initializations */
         cnt_nz = 0;
         cnt_nz_offd = 0;

         /* This block of code is to go to the top of the parallel region starting before
          * the loop over num_passes. */
         P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_coarse,
                                  NALU_HYPRE_MEMORY_HOST); /* marks points to see if they're counted */
         for (i = 0; i < n_coarse; i++)
         {   P_marker[i] = -1; }
         if (new_num_cols_offd == local_index + 1)
         {
            P_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, new_num_cols_offd, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < new_num_cols_offd; i++)
            {   P_marker_offd[i] = -1; }
         }
         else if (n_coarse_offd)
         {
            P_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_coarse_offd, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < n_coarse_offd; i++)
            {   P_marker_offd[i] = -1; }
         }


         /* Need some variables to store each threads cnt_nz and cnt_nz_offd, and
          * then stitch things together as in par_interp.c
          * This loop writes
          * P_diag_i, P_offd_i: data parallel here, and require no special treatment
          * P_diag_start, P_offd_start: are not data parallel, require special treatment
          */
         for (i = thread_start; i < thread_stop; i++)
         {
            i1 = pass_array[i];
            P_diag_start[i1] = cnt_nz;
            P_offd_start[i1] = cnt_nz_offd;
            for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
            {
               j1 = S_diag_j[j];
               if (assigned[j1] == pass - 1)
               {
                  j_start = P_diag_start[j1];
                  j_end = j_start + P_diag_i[j1 + 1];
                  for (k = j_start; k < j_end; k++)
                  {
                     k1 = P_diag_pass[pass - 1][k];
                     if (P_marker[k1] != i1)
                     {
                        cnt_nz++;
                        P_diag_i[i1 + 1]++;
                        P_marker[k1] = i1;
                     }
                  }
                  j_start = P_offd_start[j1];
                  j_end = j_start + P_offd_i[j1 + 1];
                  for (k = j_start; k < j_end; k++)
                  {
                     k1 = P_offd_pass[pass - 1][k];
                     if (P_marker_offd[k1] != i1)
                     {
                        cnt_nz_offd++;
                        P_offd_i[i1 + 1]++;
                        P_marker_offd[k1] = i1;
                     }
                  }
               }
            }
            j_start = 0;
            for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
            {
               j1 = S_offd_j[j];
               if (assigned_offd[j1] == pass - 1)
               {
                  j_start = Pext_start[j1];
                  j_end = j_start + Pext_i[j1 + 1];
                  for (k = j_start; k < j_end; k++)
                  {
                     k1 = Pext_pass[pass][k];
                     if (k1 < 0)
                     {
                        if (P_marker[-k1 - 1] != i1)
                        {
                           cnt_nz++;
                           P_diag_i[i1 + 1]++;
                           P_marker[-k1 - 1] = i1;
                        }
                     }
                     else if (P_marker_offd[k1] != i1)
                     {
                        cnt_nz_offd++;
                        P_offd_i[i1 + 1]++;
                        P_marker_offd[k1] = i1;
                     }
                  }
               }
            }
         }

         /* Update P_diag_start, P_offd_start with cumulative
          * nonzero counts over all threads */
         if (my_thread_num == 0)
         {   max_num_threads[0] = num_threads; }
         cnt_nz_offd_per_thread[my_thread_num] = cnt_nz_offd;
         cnt_nz_per_thread[my_thread_num] = cnt_nz;
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp barrier
#endif
         if (my_thread_num == 0)
         {
            for (i = 1; i < max_num_threads[0]; i++)
            {
               cnt_nz_offd_per_thread[i] += cnt_nz_offd_per_thread[i - 1];
               cnt_nz_per_thread[i] += cnt_nz_per_thread[i - 1];
            }
         }
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp barrier
#endif
         if (my_thread_num > 0)
         {
            /* update this thread's section of P_diag_start and P_offd_start
             * with the num of nz's counted by previous threads */
            for (i = thread_start; i < thread_stop; i++)
            {
               i1 = pass_array[i];
               P_diag_start[i1] += cnt_nz_per_thread[my_thread_num - 1];
               P_offd_start[i1] += cnt_nz_offd_per_thread[my_thread_num - 1];
            }
         }
         else /* if my_thread_num == 0 */
         {
            /* Grab the nz count for all threads */
            cnt_nz = cnt_nz_per_thread[max_num_threads[0] - 1];
            cnt_nz_offd = cnt_nz_offd_per_thread[max_num_threads[0] - 1];

            /* Updated total nz count */
            total_nz += cnt_nz;
            total_nz_offd += cnt_nz_offd;

            /* Allocate P_diag_pass and P_offd_pass for all threads */
            P_diag_pass[pass] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt_nz, NALU_HYPRE_MEMORY_HOST);
            if (cnt_nz_offd)
            {
               P_offd_pass[pass] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  cnt_nz_offd, NALU_HYPRE_MEMORY_HOST);
            }
            else if (num_procs > 1)
            {
               P_offd_pass[pass] = NULL;
            }
         }
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp barrier
#endif

         /* offset cnt_nz and cnt_nz_offd to point to the starting
          * point in P_diag_pass and P_offd_pass for each thread */
         if (my_thread_num > 0)
         {
            cnt_nz = cnt_nz_per_thread[my_thread_num - 1];
            cnt_nz_offd = cnt_nz_offd_per_thread[my_thread_num - 1];
         }
         else
         {
            cnt_nz = 0;
            cnt_nz_offd = 0;
         }

         /* Set P_diag_pass and P_offd_pass */
         for (i = thread_start; i < thread_stop; i++)
         {
            i1 = pass_array[i];
            for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
            {
               j1 = S_diag_j[j];
               if (assigned[j1] == pass - 1)
               {
                  j_start = P_diag_start[j1];
                  j_end = j_start + P_diag_i[j1 + 1];
                  for (k = j_start; k < j_end; k++)
                  {
                     k1 = P_diag_pass[pass - 1][k];
                     if (P_marker[k1] != -i1 - 1)
                     {
                        P_diag_pass[pass][cnt_nz++] = k1;
                        P_marker[k1] = -i1 - 1;
                     }
                  }
                  j_start = P_offd_start[j1];
                  j_end = j_start + P_offd_i[j1 + 1];
                  for (k = j_start; k < j_end; k++)
                  {
                     k1 = P_offd_pass[pass - 1][k];
                     if (P_marker_offd[k1] != -i1 - 1)
                     {
                        P_offd_pass[pass][cnt_nz_offd++] = k1;
                        P_marker_offd[k1] = -i1 - 1;
                     }
                  }
               }
            }
            for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
            {
               j1 = S_offd_j[j];
               if (assigned_offd[j1] == pass - 1)
               {
                  j_start = Pext_start[j1];
                  j_end = j_start + Pext_i[j1 + 1];
                  for (k = j_start; k < j_end; k++)
                  {
                     k1 = Pext_pass[pass][k];
                     if (k1 < 0)
                     {
                        if (P_marker[-k1 - 1] != -i1 - 1)
                        {
                           P_diag_pass[pass][cnt_nz++] = -k1 - 1;
                           P_marker[-k1 - 1] = -i1 - 1;
                        }
                     }
                     else if (P_marker_offd[k1] != -i1 - 1)
                     {
                        P_offd_pass[pass][cnt_nz_offd++] = k1;
                        P_marker_offd[k1] = -i1 - 1;
                     }
                  }
               }
            }
         }

         nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
         if ( (n_coarse_offd) || (new_num_cols_offd  == local_index + 1) )
         {    nalu_hypre_TFree(P_marker_offd, NALU_HYPRE_MEMORY_HOST); }

      } /* End parallel region */
   }


   nalu_hypre_TFree(loc, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(P_ncols, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Pext_send_buffer, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_temp_pass, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(new_recv_vec_start, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(cnt_nz_per_thread, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(cnt_nz_offd_per_thread, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(max_num_threads, NALU_HYPRE_MEMORY_HOST);

   P_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, total_nz, memory_location_P);
   P_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, total_nz, memory_location_P);


   if (total_nz_offd)
   {
      P_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, total_nz_offd, memory_location_P);
      P_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, total_nz_offd, memory_location_P);
   }

   for (i = 0; i < n_fine; i++)
   {
      P_diag_i[i + 1] += P_diag_i[i];
      P_offd_i[i + 1] += P_offd_i[i];
   }

   /* determine P for coarse points */

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,i1) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < n_coarse; i++)
   {
      i1 = C_array[i];
      P_diag_j[P_diag_i[i1]] = fine_to_coarse[i1];
      P_diag_data[P_diag_i[i1]] = 1.0;
   }


   if (weight_option) /*if this is set, weights are separated into
                        negative and positive offdiagonals and accumulated
                        accordingly */
   {

      pass_length = pass_pointer[2] - pass_pointer[1];
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel private(thread_start,thread_stop,my_thread_num,num_threads,P_marker,P_marker_offd,i,i1,sum_C_pos,sum_C_neg,sum_N_pos,sum_N_neg,j_start,j_end,j,k1,cnt,j1,cnt_offd,diagonal,alfa,beta)
#endif
      {
         /* Sparsity structure is now finished.  Next, calculate interpolation
          * weights for pass one.  Thread by computing the interpolation
          * weights only over each thread's range of rows.  Rows are divided
          * up evenly amongst the threads. */

         P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < n_fine; i++)
         {   P_marker[i] = -1; }
         if (num_cols_offd)
         {
            P_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < num_cols_offd; i++)
            {
               P_marker_offd[i] = -1;
            }
         }

         /* Compute this thread's range of pass_length */
         my_thread_num = nalu_hypre_GetThreadNum();
         num_threads = nalu_hypre_NumActiveThreads();
         thread_start = pass_pointer[1] + (pass_length / num_threads) * my_thread_num;
         if (my_thread_num == num_threads - 1)
         {  thread_stop = pass_pointer[1] + pass_length; }
         else
         {  thread_stop = pass_pointer[1] + (pass_length / num_threads) * (my_thread_num + 1); }

         /* determine P for points of pass 1, i.e. neighbors of coarse points */
         for (i = thread_start; i < thread_stop; i++)
         {
            i1 = pass_array[i];
            sum_C_pos = 0;
            sum_C_neg = 0;
            sum_N_pos = 0;
            sum_N_neg = 0;
            j_start = P_diag_start[i1];
            j_end = j_start + P_diag_i[i1 + 1] - P_diag_i[i1];
            for (j = j_start; j < j_end; j++)
            {
               k1 = P_diag_pass[1][j];
               P_marker[C_array[k1]] = i1;
            }
            cnt = P_diag_i[i1];
            for (j = A_diag_i[i1] + 1; j < A_diag_i[i1 + 1]; j++)
            {
               j1 = A_diag_j[j];
               if (CF_marker[j1] != -3 &&
                   (num_functions == 1 || dof_func[i1] == dof_func[j1]))
               {
                  if (A_diag_data[j] < 0)
                  {
                     sum_N_neg += A_diag_data[j];
                  }
                  else
                  {
                     sum_N_pos += A_diag_data[j];
                  }
               }
               if (j1 != -1 && P_marker[j1] == i1)
               {
                  P_diag_data[cnt] = A_diag_data[j];
                  P_diag_j[cnt++] = fine_to_coarse[j1];
                  if (A_diag_data[j] < 0)
                  {
                     sum_C_neg += A_diag_data[j];
                  }
                  else
                  {
                     sum_C_pos += A_diag_data[j];
                  }
               }
            }
            j_start = P_offd_start[i1];
            j_end = j_start + P_offd_i[i1 + 1] - P_offd_i[i1];
            for (j = j_start; j < j_end; j++)
            {
               k1 = P_offd_pass[1][j];
               P_marker_offd[C_array_offd[k1]] = i1;
            }
            cnt_offd = P_offd_i[i1];
            for (j = A_offd_i[i1]; j < A_offd_i[i1 + 1]; j++)
            {
               j1 = A_offd_j[j];
               if (CF_marker_offd[j1] != -3 &&
                   (num_functions == 1 || dof_func[i1] == dof_func_offd[j1]))
               {
                  if (A_offd_data[j] < 0)
                  {
                     sum_N_neg += A_offd_data[j];
                  }
                  else
                  {
                     sum_N_pos += A_offd_data[j];
                  }
               }
               if (j1 != -1 && P_marker_offd[j1] == i1)
               {
                  P_offd_data[cnt_offd] = A_offd_data[j];
                  P_offd_j[cnt_offd++] = map_S_to_new[j1];
                  if (A_offd_data[j] < 0)
                  {
                     sum_C_neg += A_offd_data[j];
                  }
                  else
                  {
                     sum_C_pos += A_offd_data[j];
                  }
               }
            }
            diagonal = A_diag_data[A_diag_i[i1]];
            if (sum_C_neg * diagonal != 0) { alfa = -sum_N_neg / (sum_C_neg * diagonal); }
            if (sum_C_pos * diagonal != 0) { beta = -sum_N_pos / (sum_C_pos * diagonal); }
            for (j = P_diag_i[i1]; j < cnt; j++)
               if (P_diag_data[j] < 0)
               {
                  P_diag_data[j] *= alfa;
               }
               else
               {
                  P_diag_data[j] *= beta;
               }
            for (j = P_offd_i[i1]; j < cnt_offd; j++)
               if (P_offd_data[j] < 0)
               {
                  P_offd_data[j] *= alfa;
               }
               else
               {
                  P_offd_data[j] *= beta;
               }
         }

         nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
         if (num_cols_offd)
         {    nalu_hypre_TFree(P_marker_offd, NALU_HYPRE_MEMORY_HOST); }
      } /* End Parallel Region */

      old_Pext_send_size = 0;
      old_Pext_recv_size = 0;

      if (n_coarse) { nalu_hypre_TFree(C_array, NALU_HYPRE_MEMORY_HOST); }
      nalu_hypre_TFree(C_array_offd, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(P_diag_pass[1], NALU_HYPRE_MEMORY_HOST);
      if (num_procs > 1) { nalu_hypre_TFree(P_offd_pass[1], NALU_HYPRE_MEMORY_HOST); }


      for (pass = 2; pass < num_passes; pass++)
      {

         if (num_procs > 1)
         {
            Pext_send_size = Pext_send_map_start[pass][num_sends];
            if (Pext_send_size > old_Pext_send_size)
            {
               nalu_hypre_TFree(Pext_send_data, NALU_HYPRE_MEMORY_HOST);
               Pext_send_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  Pext_send_size, NALU_HYPRE_MEMORY_HOST);
            }
            old_Pext_send_size = Pext_send_size;

            cnt_offd = 0;
            for (i = 0; i < num_sends; i++)
            {
               for (j = send_map_start[i]; j < send_map_start[i + 1]; j++)
               {
                  j1 = send_map_elmt[j];
                  if (assigned[j1] == pass - 1)
                  {
                     j_start = P_diag_i[j1];
                     j_end = P_diag_i[j1 + 1];
                     for (k = j_start; k < j_end; k++)
                     {   Pext_send_data[cnt_offd++] = P_diag_data[k]; }
                     j_start = P_offd_i[j1];
                     j_end = P_offd_i[j1 + 1];
                     for (k = j_start; k < j_end; k++)
                     {  Pext_send_data[cnt_offd++] = P_offd_data[k]; }
                  }
               }
            }

            nalu_hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
            nalu_hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) =
               Pext_send_map_start[pass];
            nalu_hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
            nalu_hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) =
               Pext_recv_vec_start[pass];

            Pext_recv_size = Pext_recv_vec_start[pass][num_recvs];

            if (Pext_recv_size > old_Pext_recv_size)
            {
               nalu_hypre_TFree(Pext_data, NALU_HYPRE_MEMORY_HOST);
               Pext_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  Pext_recv_size, NALU_HYPRE_MEMORY_HOST);
            }
            old_Pext_recv_size = Pext_recv_size;

            comm_handle = nalu_hypre_ParCSRCommHandleCreate (1, tmp_comm_pkg,
                                                        Pext_send_data, Pext_data);
            nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

            nalu_hypre_TFree(Pext_send_map_start[pass], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(Pext_recv_vec_start[pass], NALU_HYPRE_MEMORY_HOST);
         }

         pass_length = pass_pointer[pass + 1] - pass_pointer[pass];
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel private(thread_start,thread_stop,my_thread_num,num_threads,P_marker,P_marker_offd,i,i1,sum_C_neg,sum_C_pos,sum_N_neg,sum_N_pos,j_start,j_end,cnt,j,k1,cnt_offd,j1,k,alfa,beta,diagonal,C_array,C_array_offd)
#endif
         {
            /* Sparsity structure is now finished.  Next, calculate interpolation
             * weights for passes >= 2.  Thread by computing the interpolation
             * weights only over each thread's range of rows.  Rows are divided
             * up evenly amongst the threads. */

            P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < n_fine; i++)
            {   P_marker[i] = -1; }
            if (num_cols_offd)
            {
               P_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd, NALU_HYPRE_MEMORY_HOST);
               for (i = 0; i < num_cols_offd; i++)
               {
                  P_marker_offd[i] = -1;
               }
            }

            C_array = NULL;
            C_array_offd = NULL;
            if (n_coarse)
            {   C_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_coarse, NALU_HYPRE_MEMORY_HOST); }
            if (new_num_cols_offd > n_coarse_offd)
            {   C_array_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  new_num_cols_offd, NALU_HYPRE_MEMORY_HOST); }
            else if (n_coarse_offd)
            {   C_array_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_coarse_offd, NALU_HYPRE_MEMORY_HOST); }

            /* Compute this thread's range of pass_length */
            my_thread_num = nalu_hypre_GetThreadNum();
            num_threads = nalu_hypre_NumActiveThreads();
            thread_start = pass_pointer[pass] + (pass_length / num_threads) * my_thread_num;
            if (my_thread_num == num_threads - 1)
            {  thread_stop = pass_pointer[pass] + pass_length; }
            else
            {  thread_stop = pass_pointer[pass] + (pass_length / num_threads) * (my_thread_num + 1); }

            /* Loop over each thread's row-range */
            for (i = thread_start; i < thread_stop; i++)
            {
               i1 = pass_array[i];
               sum_C_neg = 0;
               sum_C_pos = 0;
               sum_N_neg = 0;
               sum_N_pos = 0;
               j_start = P_diag_start[i1];
               j_end = j_start + P_diag_i[i1 + 1] - P_diag_i[i1];
               cnt = P_diag_i[i1];
               for (j = j_start; j < j_end; j++)
               {
                  k1 = P_diag_pass[pass][j];
                  C_array[k1] = cnt;
                  P_diag_data[cnt] = 0;
                  P_diag_j[cnt++] = k1;
               }
               j_start = P_offd_start[i1];
               j_end = j_start + P_offd_i[i1 + 1] - P_offd_i[i1];
               cnt_offd = P_offd_i[i1];
               for (j = j_start; j < j_end; j++)
               {
                  k1 = P_offd_pass[pass][j];
                  C_array_offd[k1] = cnt_offd;
                  P_offd_data[cnt_offd] = 0;
                  P_offd_j[cnt_offd++] = k1;
               }
               for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
               {
                  j1 = S_diag_j[j];
                  if (assigned[j1] == pass - 1)
                  {
                     P_marker[j1] = i1;
                  }
               }
               for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
               {
                  j1 = S_offd_j[j];
                  if (assigned_offd[j1] == pass - 1)
                  {
                     P_marker_offd[j1] = i1;
                  }
               }
               for (j = A_diag_i[i1] + 1; j < A_diag_i[i1 + 1]; j++)
               {
                  j1 = A_diag_j[j];
                  if (P_marker[j1] == i1)
                  {
                     for (k = P_diag_i[j1]; k < P_diag_i[j1 + 1]; k++)
                     {
                        k1 = P_diag_j[k];
                        alfa = A_diag_data[j] * P_diag_data[k];
                        P_diag_data[C_array[k1]] += alfa;
                        if (alfa < 0)
                        {
                           sum_C_neg += alfa;
                           sum_N_neg += alfa;
                        }
                        else
                        {
                           sum_C_pos += alfa;
                           sum_N_pos += alfa;
                        }
                     }
                     for (k = P_offd_i[j1]; k < P_offd_i[j1 + 1]; k++)
                     {
                        k1 = P_offd_j[k];
                        alfa = A_diag_data[j] * P_offd_data[k];
                        P_offd_data[C_array_offd[k1]] += alfa;
                        if (alfa < 0)
                        {
                           sum_C_neg += alfa;
                           sum_N_neg += alfa;
                        }
                        else
                        {
                           sum_C_pos += alfa;
                           sum_N_pos += alfa;
                        }
                     }
                  }
                  else
                  {
                     if (CF_marker[j1] != -3 &&
                         (num_functions == 1 || dof_func[i1] == dof_func[j1]))
                     {
                        if (A_diag_data[j] < 0)
                        {
                           sum_N_neg += A_diag_data[j];
                        }
                        else
                        {
                           sum_N_pos += A_diag_data[j];
                        }
                     }
                  }
               }
               for (j = A_offd_i[i1]; j < A_offd_i[i1 + 1]; j++)
               {
                  j1 = A_offd_j[j];

                  if (j1 > -1 && P_marker_offd[j1] == i1)
                  {
                     j_start = Pext_start[j1];
                     j_end = j_start + Pext_i[j1 + 1];
                     for (k = j_start; k < j_end; k++)
                     {
                        k1 = Pext_pass[pass][k];
                        alfa = A_offd_data[j] * Pext_data[k];
                        if (k1 < 0)
                        {
                           P_diag_data[C_array[-k1 - 1]] += alfa;
                        }
                        else
                        {
                           P_offd_data[C_array_offd[k1]] += alfa;
                        }
                        if (alfa < 0)
                        {
                           sum_C_neg += alfa;
                           sum_N_neg += alfa;
                        }
                        else
                        {
                           sum_C_pos += alfa;
                           sum_N_pos += alfa;
                        }
                     }
                  }
                  else
                  {
                     if (CF_marker_offd[j1] != -3 &&
                         (num_functions == 1 || dof_func_offd[j1] == dof_func[i1]))
                     {
                        if ( A_offd_data[j] < 0)
                        {
                           sum_N_neg += A_offd_data[j];
                        }
                        else
                        {
                           sum_N_pos += A_offd_data[j];
                        }
                     }
                  }
               }
               diagonal = A_diag_data[A_diag_i[i1]];
               if (sum_C_neg * diagonal != 0) { alfa = -sum_N_neg / (sum_C_neg * diagonal); }
               if (sum_C_pos * diagonal != 0) { beta = -sum_N_pos / (sum_C_pos * diagonal); }

               for (j = P_diag_i[i1]; j < P_diag_i[i1 + 1]; j++)
                  if (P_diag_data[j] < 0)
                  {
                     P_diag_data[j] *= alfa;
                  }
                  else
                  {
                     P_diag_data[j] *= beta;
                  }
               for (j = P_offd_i[i1]; j < P_offd_i[i1 + 1]; j++)
                  if (P_offd_data[j] < 0)
                  {
                     P_offd_data[j] *= alfa;
                  }
                  else
                  {
                     P_offd_data[j] *= beta;
                  }
            }

            nalu_hypre_TFree(C_array, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(C_array_offd, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
            if (num_cols_offd)
            {   nalu_hypre_TFree(P_marker_offd, NALU_HYPRE_MEMORY_HOST); }

         } /* End OMP Parallel Section */

         nalu_hypre_TFree(P_diag_pass[pass], NALU_HYPRE_MEMORY_HOST);
         if (num_procs > 1)
         {
            nalu_hypre_TFree(P_offd_pass[pass], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(Pext_pass[pass], NALU_HYPRE_MEMORY_HOST);
         }
      } /* End num_passes for-loop */
   }
   else /* no distinction between positive and negative offdiagonal element */
   {

      pass_length = pass_pointer[2] - pass_pointer[1];
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel private(thread_start,thread_stop,my_thread_num,num_threads,k,k1,i,i1,j,j1,sum_C,sum_N,j_start,j_end,cnt,tmp_marker,tmp_marker_offd,cnt_offd,diagonal,alfa)
#endif
      {
         /* Sparsity structure is now finished.  Next, calculate interpolation
          * weights for pass one.  Thread by computing the interpolation
          * weights only over each thread's range of rows.  Rows are divided
          * up evenly amongst the threads. */

         /* Initialize thread-wise variables */
         tmp_marker = NULL;
         if (n_fine)
         {   tmp_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST); }
         tmp_marker_offd = NULL;
         if (num_cols_offd)
         {   tmp_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd, NALU_HYPRE_MEMORY_HOST); }
         for (i = 0; i < n_fine; i++)
         {   tmp_marker[i] = -1; }
         for (i = 0; i < num_cols_offd; i++)
         {   tmp_marker_offd[i] = -1; }

         /* Compute this thread's range of pass_length */
         my_thread_num = nalu_hypre_GetThreadNum();
         num_threads = nalu_hypre_NumActiveThreads();
         thread_start = pass_pointer[1] + (pass_length / num_threads) * my_thread_num;
         if (my_thread_num == num_threads - 1)
         {  thread_stop = pass_pointer[1] + pass_length; }
         else
         {  thread_stop = pass_pointer[1] + (pass_length / num_threads) * (my_thread_num + 1); }

         /* determine P for points of pass 1, i.e. neighbors of coarse points */
         for (i = thread_start; i < thread_stop; i++)
         {
            i1 = pass_array[i];
            sum_C = 0;
            sum_N = 0;
            j_start = P_diag_start[i1];
            j_end = j_start + P_diag_i[i1 + 1] - P_diag_i[i1];
            for (j = j_start; j < j_end; j++)
            {
               k1 = P_diag_pass[1][j];
               tmp_marker[C_array[k1]] = i1;
            }
            cnt = P_diag_i[i1];
            for (j = A_diag_i[i1] + 1; j < A_diag_i[i1 + 1]; j++)
            {
               j1 = A_diag_j[j];
               if (CF_marker[j1] != -3 &&
                   (num_functions == 1 || dof_func[i1] == dof_func[j1]))
               {
                  sum_N += A_diag_data[j];
               }
               if (j1 != -1 && tmp_marker[j1] == i1)
               {
                  P_diag_data[cnt] = A_diag_data[j];
                  P_diag_j[cnt++] = fine_to_coarse[j1];
                  sum_C += A_diag_data[j];
               }
            }
            j_start = P_offd_start[i1];
            j_end = j_start + P_offd_i[i1 + 1] - P_offd_i[i1];
            for (j = j_start; j < j_end; j++)
            {
               k1 = P_offd_pass[1][j];
               tmp_marker_offd[C_array_offd[k1]] = i1;
            }
            cnt_offd = P_offd_i[i1];
            for (j = A_offd_i[i1]; j < A_offd_i[i1 + 1]; j++)
            {
               j1 = A_offd_j[j];
               if (CF_marker_offd[j1] != -3 &&
                   (num_functions == 1 || dof_func[i1] == dof_func_offd[j1]))
               {
                  sum_N += A_offd_data[j];
               }
               if (j1 != -1 && tmp_marker_offd[j1] == i1)
               {
                  P_offd_data[cnt_offd] = A_offd_data[j];
                  P_offd_j[cnt_offd++] = map_S_to_new[j1];
                  sum_C += A_offd_data[j];
               }
            }
            diagonal = A_diag_data[A_diag_i[i1]];
            if (sum_C * diagonal != 0) { alfa = -sum_N / (sum_C * diagonal); }
            for (j = P_diag_i[i1]; j < cnt; j++)
            {
               P_diag_data[j] *= alfa;
            }
            for (j = P_offd_i[i1]; j < cnt_offd; j++)
            {
               P_offd_data[j] *= alfa;
            }
         }
         nalu_hypre_TFree(tmp_marker, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(tmp_marker_offd, NALU_HYPRE_MEMORY_HOST);
      } /* end OMP parallel region */

      old_Pext_send_size = 0;
      old_Pext_recv_size = 0;

      if (n_coarse) { nalu_hypre_TFree(C_array, NALU_HYPRE_MEMORY_HOST); }
      nalu_hypre_TFree(C_array_offd, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(P_diag_pass[1], NALU_HYPRE_MEMORY_HOST);
      if (num_procs > 1) { nalu_hypre_TFree(P_offd_pass[1], NALU_HYPRE_MEMORY_HOST); }

      for (pass = 2; pass < num_passes; pass++)
      {

         if (num_procs > 1)
         {
            Pext_send_size = Pext_send_map_start[pass][num_sends];
            if (Pext_send_size > old_Pext_send_size)
            {
               nalu_hypre_TFree(Pext_send_data, NALU_HYPRE_MEMORY_HOST);
               Pext_send_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  Pext_send_size, NALU_HYPRE_MEMORY_HOST);
            }
            old_Pext_send_size = Pext_send_size;

            cnt_offd = 0;
            for (i = 0; i < num_sends; i++)
            {
               for (j = send_map_start[i]; j < send_map_start[i + 1]; j++)
               {
                  j1 = send_map_elmt[j];
                  if (assigned[j1] == pass - 1)
                  {
                     j_start = P_diag_i[j1];
                     j_end = P_diag_i[j1 + 1];
                     for (k = j_start; k < j_end; k++)
                     {
                        Pext_send_data[cnt_offd++] = P_diag_data[k];
                     }
                     j_start = P_offd_i[j1];
                     j_end = P_offd_i[j1 + 1];
                     for (k = j_start; k < j_end; k++)
                     {
                        Pext_send_data[cnt_offd++] = P_offd_data[k];
                     }
                  }
               }
            }

            nalu_hypre_ParCSRCommPkgNumSends(tmp_comm_pkg) = num_sends;
            nalu_hypre_ParCSRCommPkgSendMapStarts(tmp_comm_pkg) =
               Pext_send_map_start[pass];
            nalu_hypre_ParCSRCommPkgNumRecvs(tmp_comm_pkg) = num_recvs;
            nalu_hypre_ParCSRCommPkgRecvVecStarts(tmp_comm_pkg) =
               Pext_recv_vec_start[pass];

            Pext_recv_size = Pext_recv_vec_start[pass][num_recvs];

            if (Pext_recv_size > old_Pext_recv_size)
            {
               nalu_hypre_TFree(Pext_data, NALU_HYPRE_MEMORY_HOST);
               Pext_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  Pext_recv_size, NALU_HYPRE_MEMORY_HOST);
            }
            old_Pext_recv_size = Pext_recv_size;

            comm_handle = nalu_hypre_ParCSRCommHandleCreate (1, tmp_comm_pkg,
                                                        Pext_send_data, Pext_data);
            nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

            nalu_hypre_TFree(Pext_send_map_start[pass], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(Pext_recv_vec_start[pass], NALU_HYPRE_MEMORY_HOST);
         }

         pass_length = pass_pointer[pass + 1] - pass_pointer[pass];
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel private(thread_start,thread_stop,my_thread_num,num_threads,k,k1,i,i1,j,j1,sum_C,sum_N,j_start,j_end,cnt,tmp_marker,tmp_marker_offd,cnt_offd,diagonal,alfa,tmp_array,tmp_array_offd)
#endif
         {
            /* Sparsity structure is now finished.  Next, calculate interpolation
             * weights for passes >= 2.  Thread by computing the interpolation
             * weights only over each thread's range of rows.  Rows are divided
             * up evenly amongst the threads. */

            /* Initialize thread-wise variables */
            tmp_marker = NULL;
            if (n_fine)
            {    tmp_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST); }
            tmp_marker_offd = NULL;
            if (num_cols_offd)
            {    tmp_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd, NALU_HYPRE_MEMORY_HOST); }
            tmp_array = NULL;
            if (n_coarse)
            {    tmp_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_coarse, NALU_HYPRE_MEMORY_HOST); }
            tmp_array_offd = NULL;
            if (new_num_cols_offd > n_coarse_offd)
            {    tmp_array_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, new_num_cols_offd, NALU_HYPRE_MEMORY_HOST); }
            else
            {    tmp_array_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_coarse_offd, NALU_HYPRE_MEMORY_HOST);}
            for (i = 0; i < n_fine; i++)
            {    tmp_marker[i] = -1; }
            for (i = 0; i < num_cols_offd; i++)
            {    tmp_marker_offd[i] = -1; }

            /* Compute this thread's range of pass_length */
            my_thread_num = nalu_hypre_GetThreadNum();
            num_threads = nalu_hypre_NumActiveThreads();
            thread_start = pass_pointer[pass] + (pass_length / num_threads) * my_thread_num;
            if (my_thread_num == num_threads - 1)
            {  thread_stop = pass_pointer[pass] + pass_length; }
            else
            {  thread_stop = pass_pointer[pass] + (pass_length / num_threads) * (my_thread_num + 1); }

            for (i = thread_start; i < thread_stop; i++)
            {
               i1 = pass_array[i];
               sum_C = 0;
               sum_N = 0;
               j_start = P_diag_start[i1];
               j_end = j_start + P_diag_i[i1 + 1] - P_diag_i[i1];
               cnt = P_diag_i[i1];
               for (j = j_start; j < j_end; j++)
               {
                  k1 = P_diag_pass[pass][j];
                  tmp_array[k1] = cnt;
                  P_diag_data[cnt] = 0;
                  P_diag_j[cnt++] = k1;
               }
               j_start = P_offd_start[i1];
               j_end = j_start + P_offd_i[i1 + 1] - P_offd_i[i1];
               cnt_offd = P_offd_i[i1];
               for (j = j_start; j < j_end; j++)
               {
                  k1 = P_offd_pass[pass][j];
                  tmp_array_offd[k1] = cnt_offd;
                  P_offd_data[cnt_offd] = 0;
                  P_offd_j[cnt_offd++] = k1;
               }
               for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
               {
                  j1 = S_diag_j[j];
                  if (assigned[j1] == pass - 1)
                  {
                     tmp_marker[j1] = i1;
                  }
               }
               for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
               {
                  j1 = S_offd_j[j];
                  if (assigned_offd[j1] == pass - 1)
                  {
                     tmp_marker_offd[j1] = i1;
                  }
               }
               for (j = A_diag_i[i1] + 1; j < A_diag_i[i1 + 1]; j++)
               {
                  j1 = A_diag_j[j];
                  if (tmp_marker[j1] == i1)
                  {
                     for (k = P_diag_i[j1]; k < P_diag_i[j1 + 1]; k++)
                     {
                        k1 = P_diag_j[k];
                        alfa = A_diag_data[j] * P_diag_data[k];
                        P_diag_data[tmp_array[k1]] += alfa;
                        sum_C += alfa;
                        sum_N += alfa;
                     }
                     for (k = P_offd_i[j1]; k < P_offd_i[j1 + 1]; k++)
                     {
                        k1 = P_offd_j[k];
                        alfa = A_diag_data[j] * P_offd_data[k];
                        P_offd_data[tmp_array_offd[k1]] += alfa;
                        sum_C += alfa;
                        sum_N += alfa;
                     }
                  }
                  else
                  {
                     if (CF_marker[j1] != -3 &&
                         (num_functions == 1 || dof_func[i1] == dof_func[j1]))
                     {
                        sum_N += A_diag_data[j];
                     }
                  }
               }
               for (j = A_offd_i[i1]; j < A_offd_i[i1 + 1]; j++)
               {
                  j1 = A_offd_j[j];

                  if (j1 > -1 && tmp_marker_offd[j1] == i1)
                  {
                     j_start = Pext_start[j1];
                     j_end = j_start + Pext_i[j1 + 1];
                     for (k = j_start; k < j_end; k++)
                     {
                        k1 = Pext_pass[pass][k];
                        alfa = A_offd_data[j] * Pext_data[k];
                        if (k1 < 0)
                        {
                           P_diag_data[tmp_array[-k1 - 1]] += alfa;
                        }
                        else
                        {
                           P_offd_data[tmp_array_offd[k1]] += alfa;
                        }
                        sum_C += alfa;
                        sum_N += alfa;
                     }
                  }
                  else
                  {
                     if (CF_marker_offd[j1] != -3 &&
                         (num_functions == 1 || dof_func_offd[j1] == dof_func[i1]))
                     {
                        sum_N += A_offd_data[j];
                     }
                  }
               }
               diagonal = A_diag_data[A_diag_i[i1]];
               if (sum_C * diagonal != 0.0) { alfa = -sum_N / (sum_C * diagonal); }

               for (j = P_diag_i[i1]; j < P_diag_i[i1 + 1]; j++)
               {
                  P_diag_data[j] *= alfa;
               }
               for (j = P_offd_i[i1]; j < P_offd_i[i1 + 1]; j++)
               {
                  P_offd_data[j] *= alfa;
               }
            }
            nalu_hypre_TFree(tmp_marker, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(tmp_marker_offd, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(tmp_array, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(tmp_array_offd, NALU_HYPRE_MEMORY_HOST);
         } /* End OMP Parallel Section */

         nalu_hypre_TFree(P_diag_pass[pass], NALU_HYPRE_MEMORY_HOST);
         if (num_procs > 1)
         {
            nalu_hypre_TFree(P_offd_pass[pass], NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(Pext_pass[pass], NALU_HYPRE_MEMORY_HOST);
         }
      }
   }

   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Pext_send_map_start, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Pext_recv_vec_start, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dof_func_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Pext_send_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Pext_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(P_diag_pass, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(P_offd_pass, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Pext_pass, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(P_diag_start, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(P_offd_start, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Pext_start, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Pext_i, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(assigned, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(assigned_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(pass_pointer, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(pass_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(map_S_to_new, NALU_HYPRE_MEMORY_HOST);
   if (num_procs > 1) { nalu_hypre_TFree(tmp_comm_pkg, NALU_HYPRE_MEMORY_HOST); }

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

   /* Compress P, removing coefficients smaller than trunc_factor * Max
      and/or keep yat most <P_max_elmts> per row absolutely maximal coefficients */

   if (trunc_factor != 0.0 || P_max_elmts != 0)
   {
      nalu_hypre_BoomerAMGInterpTruncation(P, trunc_factor, P_max_elmts);
      P_diag_data = nalu_hypre_CSRMatrixData(P_diag);
      P_diag_i = nalu_hypre_CSRMatrixI(P_diag);
      P_diag_j = nalu_hypre_CSRMatrixJ(P_diag);
      P_offd_data = nalu_hypre_CSRMatrixData(P_offd);
      P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
      P_offd_j = nalu_hypre_CSRMatrixJ(P_offd);
   }
   P_offd_size = P_offd_i[n_fine];

   num_cols_offd_P = 0;
   if (P_offd_size)
   {
      if (new_num_cols_offd > num_cols_offd)
      {   P_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, new_num_cols_offd, NALU_HYPRE_MEMORY_HOST); }
      else
      {   P_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd, NALU_HYPRE_MEMORY_HOST); }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < new_num_cols_offd; i++)
      {   P_marker_offd[i] = 0; }

      num_cols_offd_P = 0;
      for (i = 0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker_offd[index])
         {
            num_cols_offd_P++;
            P_marker_offd[index] = 1;
         }
      }

      col_map_offd_P = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_P, NALU_HYPRE_MEMORY_HOST);
      permute = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  new_counter[num_passes - 1], NALU_HYPRE_MEMORY_HOST);
      big_permute = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  new_counter[num_passes - 1], NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < new_counter[num_passes - 1]; i++)
      {
         big_permute[i] = -1;
      }

      cnt = 0;
      for (i = 0; i < num_passes - 1; i++)
      {
         for (j = new_counter[i]; j < new_counter[i + 1]; j++)
         {
            if (P_marker_offd[j])
            {
               col_map_offd_P[cnt] = new_elmts[i][j - (NALU_HYPRE_BigInt)new_counter[i]];
               big_permute[j] = col_map_offd_P[cnt++];
            }
         }
      }

      nalu_hypre_BigQsort0(col_map_offd_P, 0, num_cols_offd_P - 1);

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i,big_k1) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < new_counter[num_passes - 1]; i++)
      {
         big_k1 = big_permute[i];
         if (big_k1 != -1)
         {
            permute[i] = nalu_hypre_BigBinarySearch(col_map_offd_P, big_k1, num_cols_offd_P);
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < P_offd_size; i++)
      {   P_offd_j[i] = permute[P_offd_j[i]]; }

      nalu_hypre_TFree(P_marker_offd, NALU_HYPRE_MEMORY_HOST);
   }
   if (num_procs > 1)
   {
      for (i = 0; i < num_passes - 1; i++)
      {
         nalu_hypre_TFree(new_elmts[i], NALU_HYPRE_MEMORY_HOST);
      }
   }
   nalu_hypre_TFree(permute, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_permute, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(new_elmts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(new_counter, NALU_HYPRE_MEMORY_HOST);

   if (num_cols_offd_P)
   {
      nalu_hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      nalu_hypre_CSRMatrixNumCols(P_offd) = num_cols_offd_P;
   }

   if (n_SF)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < n_fine; i++)
         if (CF_marker[i] == -3) { CF_marker[i] = -1; }
   }

   if (num_procs > 1)
   {
      nalu_hypre_MatvecCommPkgCreate(P);
   }

   *P_ptr = P;

   /* wall_time = nalu_hypre_MPI_Wtime() - wall_time;
      nalu_hypre_printf("TOTAL TIME  %1.2e \n",wall_time); */

   /*-----------------------------------------------------------------------
    *  Build and return dof_func array for coarse grid.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Free mapping vector and marker array.
    *-----------------------------------------------------------------------*/

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_MULTIPASS_INTERP] += nalu_hypre_MPI_Wtime();
#endif

   return (0);
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildMultipass( nalu_hypre_ParCSRMatrix  *A,
                               NALU_HYPRE_Int           *CF_marker,
                               nalu_hypre_ParCSRMatrix  *S,
                               NALU_HYPRE_BigInt        *num_cpts_global,
                               NALU_HYPRE_Int            num_functions,
                               NALU_HYPRE_Int           *dof_func,
                               NALU_HYPRE_Int            debug_flag,
                               NALU_HYPRE_Real           trunc_factor,
                               NALU_HYPRE_Int            P_max_elmts,
                               NALU_HYPRE_Int            weight_option,
                               nalu_hypre_ParCSRMatrix **P_ptr )
{
   nalu_hypre_GpuProfilingPushRange("MultipassInterp");

   NALU_HYPRE_Int ierr = 0;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(A),
                                                      nalu_hypre_ParCSRMatrixMemoryLocation(S) );
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      /* Notice: call the mod version on GPUs */
      ierr = nalu_hypre_BoomerAMGBuildModMultipassDevice( A, CF_marker, S, num_cpts_global,
                                                     trunc_factor, P_max_elmts, 9,
                                                     num_functions, dof_func,
                                                     P_ptr );
   }
   else
#endif
   {
      ierr = nalu_hypre_BoomerAMGBuildMultipassHost( A, CF_marker, S, num_cpts_global,
                                                num_functions, dof_func, debug_flag,
                                                trunc_factor, P_max_elmts, weight_option,
                                                P_ptr );
   }

   nalu_hypre_GpuProfilingPopRange();

   return ierr;
}
