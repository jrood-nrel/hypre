/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"

/*---------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGBuildInterp
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildInterp( nalu_hypre_ParCSRMatrix      *A,
                            NALU_HYPRE_Int               *CF_marker,
                            nalu_hypre_ParCSRMatrix      *S,
                            NALU_HYPRE_BigInt            *num_cpts_global,
                            NALU_HYPRE_Int                num_functions,
                            NALU_HYPRE_Int               *dof_func,
                            NALU_HYPRE_Int                debug_flag,
                            NALU_HYPRE_Real               trunc_factor,
                            NALU_HYPRE_Int                max_elmts,
                            nalu_hypre_ParCSRMatrix     **P_ptr)
{
   MPI_Comm                 comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   NALU_HYPRE_MemoryLocation memory_location_P = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   nalu_hypre_CSRMatrix   *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real        *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int         *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int         *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);

   nalu_hypre_CSRMatrix   *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real        *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int         *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int         *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Int          num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt      *col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(A);

   nalu_hypre_CSRMatrix   *S_diag = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int         *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int         *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);

   nalu_hypre_CSRMatrix   *S_offd = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int         *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int         *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);

   nalu_hypre_ParCSRMatrix *P;
   NALU_HYPRE_BigInt       *col_map_offd_P;
   NALU_HYPRE_Int          *tmp_map_offd = NULL;

   NALU_HYPRE_Int         *CF_marker_offd = NULL;
   NALU_HYPRE_Int         *dof_func_offd = NULL;

   nalu_hypre_CSRMatrix   *A_ext;

   NALU_HYPRE_Real        *A_ext_data = NULL;
   NALU_HYPRE_Int         *A_ext_i = NULL;
   NALU_HYPRE_BigInt      *A_ext_j = NULL;

   nalu_hypre_CSRMatrix   *P_diag;
   nalu_hypre_CSRMatrix   *P_offd;

   NALU_HYPRE_Real        *P_diag_data;
   NALU_HYPRE_Int         *P_diag_i;
   NALU_HYPRE_Int         *P_diag_j;
   NALU_HYPRE_Real        *P_offd_data;
   NALU_HYPRE_Int         *P_offd_i;
   NALU_HYPRE_Int         *P_offd_j;

   NALU_HYPRE_Int          P_diag_size, P_offd_size;

   NALU_HYPRE_Int         *P_marker, *P_marker_offd;

   NALU_HYPRE_Int          jj_counter, jj_counter_offd;
   NALU_HYPRE_Int         *jj_count, *jj_count_offd;
   NALU_HYPRE_Int          jj_begin_row, jj_begin_row_offd;
   NALU_HYPRE_Int          jj_end_row, jj_end_row_offd;

   NALU_HYPRE_Int          start_indexing = 0; /* start indexing for P_data at 0 */

   NALU_HYPRE_Int          n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);

   NALU_HYPRE_Int          strong_f_marker;

   NALU_HYPRE_Int         *fine_to_coarse;
   //NALU_HYPRE_Int         *fine_to_coarse_offd;
   NALU_HYPRE_Int         *coarse_counter;
   NALU_HYPRE_Int          coarse_shift;
   NALU_HYPRE_BigInt       total_global_cpts;
   //NALU_HYPRE_BigInt       my_first_cpt;
   NALU_HYPRE_Int          num_cols_P_offd;

   NALU_HYPRE_Int          i, i1, i2;
   NALU_HYPRE_Int          j, jl, jj, jj1;
   NALU_HYPRE_Int          kc;
   NALU_HYPRE_BigInt       big_k;
   NALU_HYPRE_Int          start;
   NALU_HYPRE_Int          sgn;
   NALU_HYPRE_Int          c_num;

   NALU_HYPRE_Real         diagonal;
   NALU_HYPRE_Real         sum;
   NALU_HYPRE_Real         distribute;

   NALU_HYPRE_Real         zero = 0.0;
   NALU_HYPRE_Real         one  = 1.0;

   NALU_HYPRE_Int          my_id;
   NALU_HYPRE_Int          num_procs;
   NALU_HYPRE_Int          num_threads;
   NALU_HYPRE_Int          num_sends;
   NALU_HYPRE_Int          index;
   NALU_HYPRE_Int          ns, ne, size, rest;
   NALU_HYPRE_Int          print_level = 0;
   NALU_HYPRE_Int         *int_buf_data;

   NALU_HYPRE_BigInt col_1 = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   NALU_HYPRE_Int local_numrows = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_BigInt col_n = col_1 + (NALU_HYPRE_BigInt)local_numrows;

   NALU_HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   num_threads = nalu_hypre_NumThreads();

   //my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag < 0)
   {
      debug_flag = -debug_flag;
      print_level = 1;
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   if (num_cols_A_offd) { CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST); }
   if (num_functions > 1 && num_cols_A_offd)
   {
      dof_func_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
   }

   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                NALU_HYPRE_MEMORY_HOST);

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
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            int_buf_data[index++] = dof_func[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, dof_func_offd);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*----------------------------------------------------------------------
    * Get the ghost rows of A
    *---------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   if (num_procs > 1)
   {
      A_ext      = nalu_hypre_ParCSRMatrixExtractBExt(A, A, 1);
      A_ext_i    = nalu_hypre_CSRMatrixI(A_ext);
      A_ext_j    = nalu_hypre_CSRMatrixBigJ(A_ext);
      A_ext_data = nalu_hypre_CSRMatrixData(A_ext);
   }

   index = 0;
   for (i = 0; i < num_cols_A_offd; i++)
   {
      for (j = A_ext_i[i]; j < A_ext_i[i + 1]; j++)
      {
         big_k = A_ext_j[j];
         if (big_k >= col_1 && big_k < col_n)
         {
            A_ext_j[index] = big_k - col_1;
            A_ext_data[index++] = A_ext_data[j];
         }
         else
         {
            kc = nalu_hypre_BigBinarySearch(col_map_offd, big_k, num_cols_A_offd);
            if (kc > -1)
            {
               A_ext_j[index] = (NALU_HYPRE_BigInt)(-kc - 1);
               A_ext_data[index++] = A_ext_data[j];
            }
         }
      }
      A_ext_i[i] = index;
   }
   for (i = num_cols_A_offd; i > 0; i--)
   {
      A_ext_i[i] = A_ext_i[i - 1];
   }
   if (num_procs > 1) { A_ext_i[0] = 0; }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n",
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
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

   /* RDF: this looks a little tricky, but doable */
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,i1,jj,ns,ne,size,rest) NALU_HYPRE_SMP_SCHEDULE
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
          *  If i is an F-point, interpolation is from the C-points that
          *  strongly influence i.
          *--------------------------------------------------------------------*/

         else
         {
            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];
               if (CF_marker[i1] >= 0)
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];
                  if (CF_marker_offd[i1] >= 0)
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

   P_diag_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine + 1,  memory_location_P);
   P_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_diag_size, memory_location_P);
   P_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, P_diag_size, memory_location_P);

   P_diag_i[n_fine] = jj_counter;


   P_offd_size = jj_counter_offd;

   P_offd_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine + 1,  memory_location_P);
   P_offd_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_offd_size, memory_location_P);
   P_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, P_offd_size, memory_location_P);

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
      //fine_to_coarse[i] += my_first_cpt+coarse_shift;
   }
   /*index = 0;
     for (i = 0; i < num_sends; i++)
     {
     start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
     for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
     int_buf_data[index++]
     = fine_to_coarse[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
     }

     comm_handle = nalu_hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, fine_to_coarse_offd);

     nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

     if (debug_flag==4)
     {
     wall_time = time_getWallclockSeconds() - wall_time;
     nalu_hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
     my_id, wall_time);
     fflush(NULL);
     }*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   /*#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
   #endif
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt; */

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,jl,i1,i2,jj,jj1,ns,ne,size,rest,sum,diagonal,distribute,P_marker,P_marker_offd,strong_f_marker,jj_counter,jj_counter_offd,sgn,c_num,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd) NALU_HYPRE_SMP_SCHEDULE
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
      strong_f_marker = -2;

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
            jj_begin_row = jj_counter;

            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  P_diag_data[jj_counter] = zero;
                  jj_counter++;
               }

               /*--------------------------------------------------------------
                * If neighbor i1 is an F-point, mark it as a strong F-point
                * whose connection needs to be distributed.
                *--------------------------------------------------------------*/

               else if (CF_marker[i1] != -3)
               {
                  P_marker[i1] = strong_f_marker;
               }
            }
            jj_end_row = jj_counter;

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;
            jj_begin_row_offd = jj_counter_offd;


            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];

                  /*-----------------------------------------------------------
                   * If neighbor i1 is a C-point, set column number in P_offd_j
                   * and initialize interpolation weight to zero.
                   *-----------------------------------------------------------*/

                  if (CF_marker_offd[i1] >= 0)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     /*P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];*/
                     P_offd_j[jj_counter_offd]  = i1;
                     P_offd_data[jj_counter_offd] = zero;
                     jj_counter_offd++;
                  }

                  /*-----------------------------------------------------------
                   * If neighbor i1 is an F-point, mark it as a strong F-point
                   * whose connection needs to be distributed.
                   *-----------------------------------------------------------*/

                  else if (CF_marker_offd[i1] != -3)
                  {
                     P_marker_offd[i1] = strong_f_marker;
                  }
               }
            }

            jj_end_row_offd = jj_counter_offd;

            diagonal = A_diag_data[A_diag_i[i]];


            /* Loop over ith row of A.  First, the diagonal part of A */

            for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];

               /*--------------------------------------------------------------
                * Case 1: neighbor i1 is a C-point and strongly influences i,
                * accumulate a_{i,i1} into the interpolation weight.
                *--------------------------------------------------------------*/

               if (P_marker[i1] >= jj_begin_row)
               {
                  P_diag_data[P_marker[i1]] += A_diag_data[jj];
               }

               /*--------------------------------------------------------------
                * Case 2: neighbor i1 is an F-point and strongly influences i,
                * distribute a_{i,i1} to C-points that strongly infuence i.
                * Note: currently no distribution to the diagonal in this case.
                *--------------------------------------------------------------*/

               else if (P_marker[i1] == strong_f_marker)
               {
                  sum = zero;

                  /*-----------------------------------------------------------
                   * Loop over row of A for point i1 and calculate the sum
                   * of the connections to c-points that strongly influence i.
                   *-----------------------------------------------------------*/
                  sgn = 1;
                  if (A_diag_data[A_diag_i[i1]] < 0) { sgn = -1; }
                  /* Diagonal block part of row i1 */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row &&
                         (sgn * A_diag_data[jj1]) < 0)
                     {
                        sum += A_diag_data[jj1];
                     }
                  }

                  /* Off-Diagonal block part of row i1 */
                  if (num_procs > 1)
                  {
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                     {
                        i2 = A_offd_j[jj1];
                        if (P_marker_offd[i2] >= jj_begin_row_offd
                            && (sgn * A_offd_data[jj1]) < 0)
                        {
                           sum += A_offd_data[jj1];
                        }
                     }
                  }

                  if (sum != 0)
                  {
                     distribute = A_diag_data[jj] / sum;

                     /*-----------------------------------------------------------
                      * Loop over row of A for point i1 and do the distribution.
                      *-----------------------------------------------------------*/

                     /* Diagonal block part of row i1 */
                     for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                     {
                        i2 = A_diag_j[jj1];
                        if (P_marker[i2] >= jj_begin_row
                            && (sgn * A_diag_data[jj1]) < 0)
                        {
                           P_diag_data[P_marker[i2]]
                           += distribute * A_diag_data[jj1];
                        }
                     }

                     /* Off-Diagonal block part of row i1 */
                     if (num_procs > 1)
                     {
                        for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                        {
                           i2 = A_offd_j[jj1];
                           if (P_marker_offd[i2] >= jj_begin_row_offd
                               && (sgn * A_offd_data[jj1]) < 0)
                           {
                              P_offd_data[P_marker_offd[i2]]
                              += distribute * A_offd_data[jj1];
                           }
                        }
                     }
                  }
                  else
                  {
                     if (num_functions == 1 || dof_func[i] == dof_func[i1])
                     {
                        diagonal += A_diag_data[jj];
                     }
                  }
               }

               /*--------------------------------------------------------------
                * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                * into the diagonal.
                *--------------------------------------------------------------*/

               else if (CF_marker[i1] != -3)
               {
                  if (num_functions == 1 || dof_func[i] == dof_func[i1])
                  {
                     diagonal += A_diag_data[jj];
                  }
               }

            }


            /*----------------------------------------------------------------
             * Still looping over ith row of A. Next, loop over the
             * off-diagonal part of A
             *---------------------------------------------------------------*/

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];

                  /*--------------------------------------------------------------
                   * Case 1: neighbor i1 is a C-point and strongly influences i,
                   * accumulate a_{i,i1} into the interpolation weight.
                   *--------------------------------------------------------------*/

                  if (P_marker_offd[i1] >= jj_begin_row_offd)
                  {
                     P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
                  }

                  /*------------------------------------------------------------
                   * Case 2: neighbor i1 is an F-point and strongly influences i,
                   * distribute a_{i,i1} to C-points that strongly infuence i.
                   * Note: currently no distribution to the diagonal in this case.
                   *-----------------------------------------------------------*/

                  else if (P_marker_offd[i1] == strong_f_marker)
                  {
                     sum = zero;

                     /*---------------------------------------------------------
                      * Loop over row of A_ext for point i1 and calculate the sum
                      * of the connections to c-points that strongly influence i.
                      *---------------------------------------------------------*/

                     /* find row number */
                     c_num = A_offd_j[jj];

                     sgn = 1;
                     if (A_ext_data[A_ext_i[c_num]] < 0) { sgn = -1; }
                     for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num + 1]; jj1++)
                     {
                        i2 = (NALU_HYPRE_Int)A_ext_j[jj1];

                        if (i2 > -1)
                        {
                           /* in the diagonal block */
                           if (P_marker[i2] >= jj_begin_row
                               && (sgn * A_ext_data[jj1]) < 0)
                           {
                              sum += A_ext_data[jj1];
                           }
                        }
                        else
                        {
                           /* in the off_diagonal block  */
                           if (P_marker_offd[-i2 - 1] >= jj_begin_row_offd
                               && (sgn * A_ext_data[jj1]) < 0)
                           {
                              sum += A_ext_data[jj1];
                           }

                        }

                     }

                     if (sum != 0)
                     {
                        distribute = A_offd_data[jj] / sum;
                        /*---------------------------------------------------------
                         * Loop over row of A_ext for point i1 and do
                         * the distribution.
                         *--------------------------------------------------------*/

                        /* Diagonal block part of row i1 */

                        for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num + 1]; jj1++)
                        {
                           i2 = (NALU_HYPRE_Int)A_ext_j[jj1];

                           if (i2 > -1) /* in the diagonal block */
                           {
                              if (P_marker[i2] >= jj_begin_row
                                  && (sgn * A_ext_data[jj1]) < 0)
                              {
                                 P_diag_data[P_marker[i2]]
                                 += distribute * A_ext_data[jj1];
                              }
                           }
                           else
                           {
                              /* in the off_diagonal block  */
                              if (P_marker_offd[-i2 - 1] >= jj_begin_row_offd
                                  && (sgn * A_ext_data[jj1]) < 0)
                                 P_offd_data[P_marker_offd[-i2 - 1]]
                                 += distribute * A_ext_data[jj1];
                           }
                        }
                     }
                     else
                     {
                        if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                        {
                           diagonal += A_offd_data[jj];
                        }
                     }
                  }

                  /*-----------------------------------------------------------
                   * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                   * into the diagonal.
                   *-----------------------------------------------------------*/

                  else if (CF_marker_offd[i1] != -3)
                  {
                     if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                     {
                        diagonal += A_offd_data[jj];
                     }
                  }

               }
            }

            /*-----------------------------------------------------------------
             * Set interpolation weight by dividing by the diagonal.
             *-----------------------------------------------------------------*/

            if (diagonal == 0.0)
            {
               if (print_level)
               {
                  nalu_hypre_printf(" Warning! zero diagonal! Proc id %d row %d\n", my_id, i);
               }
               for (jj = jj_begin_row; jj < jj_end_row; jj++)
               {
                  P_diag_data[jj] = 0.0;
               }
               for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
               {
                  P_offd_data[jj] = 0.0;
               }
            }
            else
            {
               for (jj = jj_begin_row; jj < jj_end_row; jj++)
               {
                  P_diag_data[jj] /= -diagonal;
               }
               for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
               {
                  P_offd_data[jj] /= -diagonal;
               }
            }

         }

         strong_f_marker--;

         P_offd_i[i + 1] = jj_counter_offd;
      }
      nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(P_marker_offd, NALU_HYPRE_MEMORY_HOST);
   }

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

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      nalu_hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = nalu_hypre_CSRMatrixData(P_diag);
      P_diag_i = nalu_hypre_CSRMatrixI(P_diag);
      P_diag_j = nalu_hypre_CSRMatrixJ(P_diag);
      P_offd_data = nalu_hypre_CSRMatrixData(P_offd);
      P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
      P_offd_j = nalu_hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
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

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < P_offd_size; i++)
         P_offd_j[i] = nalu_hypre_BinarySearch(tmp_map_offd,
                                          P_offd_j[i],
                                          num_cols_P_offd);
      nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < n_fine; i++)
   {
      if (CF_marker[i] == -3) { CF_marker[i] = -1; }
   }

   if (num_cols_P_offd)
   {
      nalu_hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      nalu_hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;
   }

   nalu_hypre_GetCommPkgRTFromCommPkgA(P, A, fine_to_coarse, tmp_map_offd);


   *P_ptr = P;

   nalu_hypre_TFree(tmp_map_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dof_func_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   //nalu_hypre_TFree(fine_to_coarse_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(coarse_counter, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count_offd, NALU_HYPRE_MEMORY_HOST);

   if (num_procs > 1) { nalu_hypre_CSRMatrixDestroy(A_ext); }

   return nalu_hypre_error_flag;
}


/*---------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGBuildInterpHE
 * interpolation routine for hyperbolic PDEs
 * treats weak fine connections  like strong fine connections
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildInterpHE( nalu_hypre_ParCSRMatrix   *A,
                              NALU_HYPRE_Int            *CF_marker,
                              nalu_hypre_ParCSRMatrix   *S,
                              NALU_HYPRE_BigInt         *num_cpts_global,
                              NALU_HYPRE_Int             num_functions,
                              NALU_HYPRE_Int            *dof_func,
                              NALU_HYPRE_Int             debug_flag,
                              NALU_HYPRE_Real            trunc_factor,
                              NALU_HYPRE_Int             max_elmts,
                              nalu_hypre_ParCSRMatrix  **P_ptr)
{

   MPI_Comm      comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real      *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Int        num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt    *col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(A);

   nalu_hypre_CSRMatrix *S_diag = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int       *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int       *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);

   nalu_hypre_CSRMatrix *S_offd = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int       *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int       *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);

   nalu_hypre_ParCSRMatrix *P;
   NALU_HYPRE_BigInt      *col_map_offd_P;
   NALU_HYPRE_Int      *tmp_map_offd = NULL;

   NALU_HYPRE_Int          *CF_marker_offd = NULL;
   NALU_HYPRE_Int          *dof_func_offd = NULL;

   nalu_hypre_CSRMatrix *A_ext;

   NALU_HYPRE_Real      *A_ext_data = NULL;
   NALU_HYPRE_Int       *A_ext_i = NULL;
   NALU_HYPRE_BigInt    *A_ext_j = NULL;

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
   NALU_HYPRE_Int        jj_begin_row, jj_begin_row_offd;
   NALU_HYPRE_Int        jj_end_row, jj_end_row_offd;

   NALU_HYPRE_Int        start_indexing = 0; /* start indexing for P_data at 0 */

   NALU_HYPRE_Int        n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);

   NALU_HYPRE_Int       *fine_to_coarse;
   //NALU_HYPRE_Int       *fine_to_coarse_offd;
   NALU_HYPRE_Int       *coarse_counter;
   NALU_HYPRE_Int        coarse_shift;
   NALU_HYPRE_BigInt     total_global_cpts;
   //NALU_HYPRE_BigInt     my_first_cpt;
   NALU_HYPRE_Int        num_cols_P_offd;

   NALU_HYPRE_Int        i, i1, i2;
   NALU_HYPRE_Int        j, jl, jj, jj1;
   NALU_HYPRE_Int        kc;
   NALU_HYPRE_BigInt     big_k;
   NALU_HYPRE_Int        start;
   NALU_HYPRE_Int        sgn;
   NALU_HYPRE_Int        c_num;

   NALU_HYPRE_Real       diagonal;
   NALU_HYPRE_Real       sum;
   NALU_HYPRE_Real       distribute;

   NALU_HYPRE_Real       zero = 0.0;
   NALU_HYPRE_Real       one  = 1.0;

   NALU_HYPRE_Int        my_id;
   NALU_HYPRE_Int        num_procs;
   NALU_HYPRE_Int        num_threads;
   NALU_HYPRE_Int        num_sends;
   NALU_HYPRE_Int        index;
   NALU_HYPRE_Int        ns, ne, size, rest;
   NALU_HYPRE_Int       *int_buf_data;

   NALU_HYPRE_BigInt col_1 = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   NALU_HYPRE_Int local_numrows = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_BigInt col_n = col_1 + local_numrows;

   NALU_HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   num_threads = nalu_hypre_NumThreads();


   //my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   if (num_cols_A_offd) { CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST); }
   if (num_functions > 1 && num_cols_A_offd)
   {
      dof_func_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
   }

   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                NALU_HYPRE_MEMORY_HOST);

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         int_buf_data[index++]
            = CF_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
   }

   comm_handle = nalu_hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, CF_marker_offd);

   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            int_buf_data[index++]
               = dof_func[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, dof_func_offd);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*----------------------------------------------------------------------
    * Get the ghost rows of A
    *---------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   if (num_procs > 1)
   {
      A_ext      = nalu_hypre_ParCSRMatrixExtractBExt(A, A, 1);
      A_ext_i    = nalu_hypre_CSRMatrixI(A_ext);
      A_ext_j    = nalu_hypre_CSRMatrixBigJ(A_ext);
      A_ext_data = nalu_hypre_CSRMatrixData(A_ext);
   }

   index = 0;
   for (i = 0; i < num_cols_A_offd; i++)
   {
      for (j = A_ext_i[i]; j < A_ext_i[i + 1]; j++)
      {
         big_k = A_ext_j[j];
         if (big_k >= col_1 && big_k < col_n)
         {
            A_ext_j[index] = big_k - col_1;
            A_ext_data[index++] = A_ext_data[j];
         }
         else
         {
            kc = nalu_hypre_BigBinarySearch(col_map_offd, big_k, num_cols_A_offd);
            if (kc > -1)
            {
               A_ext_j[index] = (NALU_HYPRE_BigInt)(-kc - 1);
               A_ext_data[index++] = A_ext_data[j];
            }
         }
      }
      A_ext_i[i] = index;
   }
   for (i = num_cols_A_offd; i > 0; i--)
   {
      A_ext_i[i] = A_ext_i[i - 1];
   }
   if (num_procs > 1) { A_ext_i[0] = 0; }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n",
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
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

   /* RDF: this looks a little tricky, but doable */
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,i1,jj,ns,ne,size,rest) NALU_HYPRE_SMP_SCHEDULE
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
          *  If i is an F-point, interpolation is from the C-points that
          *  strongly influence i.
          *--------------------------------------------------------------------*/

         else
         {
            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];
               if (CF_marker[i1] >= 0)
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];
                  if (CF_marker_offd[i1] >= 0)
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

     if (debug_flag==4)
     {
     wall_time = time_getWallclockSeconds() - wall_time;
     nalu_hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
     my_id, wall_time);
     fflush(NULL);
     }*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   /*#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
   #endif
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;*/

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,jl,i1,i2,jj,jj1,ns,ne,size,rest,sum,diagonal,distribute,P_marker,P_marker_offd,jj_counter,jj_counter_offd,sgn,c_num,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd) NALU_HYPRE_SMP_SCHEDULE
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
            jj_begin_row = jj_counter;

            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  P_diag_data[jj_counter] = zero;
                  jj_counter++;
               }

            }
            jj_end_row = jj_counter;

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;
            jj_begin_row_offd = jj_counter_offd;


            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];

                  /*-----------------------------------------------------------
                   * If neighbor i1 is a C-point, set column number in P_offd_j
                   * and initialize interpolation weight to zero.
                   *-----------------------------------------------------------*/

                  if (CF_marker_offd[i1] >= 0)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd]  = i1;
                     P_offd_data[jj_counter_offd] = zero;
                     jj_counter_offd++;
                  }
               }
            }

            jj_end_row_offd = jj_counter_offd;

            diagonal = A_diag_data[A_diag_i[i]];


            /* Loop over ith row of A.  First, the diagonal part of A */

            for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];

               /*--------------------------------------------------------------
                * Case 1: neighbor i1 is a C-point and strongly influences i,
                * accumulate a_{i,i1} into the interpolation weight.
                *--------------------------------------------------------------*/

               if (P_marker[i1] >= jj_begin_row)
               {
                  P_diag_data[P_marker[i1]] += A_diag_data[jj];
               }

               /*--------------------------------------------------------------
                * Case 2: neighbor i1 is an F-point and influences i,
                * distribute a_{i,i1} to C-points that strongly influence i.
                * Note: currently no distribution to the diagonal in this case.
                *--------------------------------------------------------------*/

               else
               {
                  sum = zero;

                  /*-----------------------------------------------------------
                   * Loop over row of A for point i1 and calculate the sum
                   * of the connections to c-points that strongly influence i.
                   *-----------------------------------------------------------*/
                  sgn = 1;
                  if (A_diag_data[A_diag_i[i1]] < 0) { sgn = -1; }
                  /* Diagonal block part of row i1 */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if (P_marker[i2] >= jj_begin_row &&
                         (sgn * A_diag_data[jj1]) < 0)
                     {
                        sum += A_diag_data[jj1];
                     }
                  }

                  /* Off-Diagonal block part of row i1 */
                  if (num_procs > 1)
                  {
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                     {
                        i2 = A_offd_j[jj1];
                        if (P_marker_offd[i2] >= jj_begin_row_offd
                            && (sgn * A_offd_data[jj1]) < 0)
                        {
                           sum += A_offd_data[jj1];
                        }
                     }
                  }

                  if (sum != 0)
                  {
                     distribute = A_diag_data[jj] / sum;

                     /*-----------------------------------------------------------
                      * Loop over row of A for point i1 and do the distribution.
                      *-----------------------------------------------------------*/

                     /* Diagonal block part of row i1 */
                     for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                     {
                        i2 = A_diag_j[jj1];
                        if (P_marker[i2] >= jj_begin_row
                            && (sgn * A_diag_data[jj1]) < 0)
                        {
                           P_diag_data[P_marker[i2]]
                           += distribute * A_diag_data[jj1];
                        }
                     }

                     /* Off-Diagonal block part of row i1 */
                     if (num_procs > 1)
                     {
                        for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                        {
                           i2 = A_offd_j[jj1];
                           if (P_marker_offd[i2] >= jj_begin_row_offd
                               && (sgn * A_offd_data[jj1]) < 0)
                           {
                              P_offd_data[P_marker_offd[i2]]
                              += distribute * A_offd_data[jj1];
                           }
                        }
                     }
                  }
                  else
                  {
                     if (num_functions == 1 || dof_func[i] == dof_func[i1])
                     {
                        diagonal += A_diag_data[jj];
                     }
                  }
               }

            }


            /*----------------------------------------------------------------
             * Still looping over ith row of A. Next, loop over the
             * off-diagonal part of A
             *---------------------------------------------------------------*/

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];

                  /*--------------------------------------------------------------
                   * Case 1: neighbor i1 is a C-point and strongly influences i,
                   * accumulate a_{i,i1} into the interpolation weight.
                   *--------------------------------------------------------------*/

                  if (P_marker_offd[i1] >= jj_begin_row_offd)
                  {
                     P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
                  }

                  /*------------------------------------------------------------
                   * Case 2: neighbor i1 is an F-point and influences i,
                   * distribute a_{i,i1} to C-points that strongly infuence i.
                   * Note: currently no distribution to the diagonal in this case.
                   *-----------------------------------------------------------*/

                  else
                  {
                     sum = zero;

                     /*---------------------------------------------------------
                      * Loop over row of A_ext for point i1 and calculate the sum
                      * of the connections to c-points that strongly influence i.
                      *---------------------------------------------------------*/

                     /* find row number */
                     c_num = A_offd_j[jj];

                     sgn = 1;
                     if (A_ext_data[A_ext_i[c_num]] < 0) { sgn = -1; }
                     for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num + 1]; jj1++)
                     {
                        i2 = (NALU_HYPRE_Int)A_ext_j[jj1];

                        if (i2 > -1)
                        {
                           /* in the diagonal block */
                           if (P_marker[i2] >= jj_begin_row
                               && (sgn * A_ext_data[jj1]) < 0)
                           {
                              sum += A_ext_data[jj1];
                           }
                        }
                        else
                        {
                           /* in the off_diagonal block  */
                           if (P_marker_offd[-i2 - 1] >= jj_begin_row_offd
                               && (sgn * A_ext_data[jj1]) < 0)
                           {
                              sum += A_ext_data[jj1];
                           }

                        }

                     }

                     if (sum != 0)
                     {
                        distribute = A_offd_data[jj] / sum;
                        /*---------------------------------------------------------
                         * Loop over row of A_ext for point i1 and do
                         * the distribution.
                         *--------------------------------------------------------*/

                        /* Diagonal block part of row i1 */
                        for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num + 1]; jj1++)
                        {
                           i2 = (NALU_HYPRE_Int)A_ext_j[jj1];

                           if (i2 > -1) /* in the diagonal block */
                           {
                              if (P_marker[i2] >= jj_begin_row
                                  && (sgn * A_ext_data[jj1]) < 0)
                              {
                                 P_diag_data[P_marker[i2]]
                                 += distribute * A_ext_data[jj1];
                              }
                           }
                           else
                           {
                              /* in the off_diagonal block  */
                              if (P_marker_offd[-i2 - 1] >= jj_begin_row_offd
                                  && (sgn * A_ext_data[jj1]) < 0)
                                 P_offd_data[P_marker_offd[-i2 - 1]]
                                 += distribute * A_ext_data[jj1];
                           }
                        }
                     }
                     else
                     {
                        if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                        {
                           diagonal += A_offd_data[jj];
                        }
                     }
                  }
               }
            }

            /*-----------------------------------------------------------------
             * Set interpolation weight by dividing by the diagonal.
             *-----------------------------------------------------------------*/

            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               P_diag_data[jj] /= -diagonal;
            }

            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               P_offd_data[jj] /= -diagonal;
            }
         }

         P_offd_i[i + 1] = jj_counter_offd;
      }
      nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(P_marker_offd, NALU_HYPRE_MEMORY_HOST);
   }

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

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      nalu_hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = nalu_hypre_CSRMatrixData(P_diag);
      P_diag_i = nalu_hypre_CSRMatrixI(P_diag);
      P_diag_j = nalu_hypre_CSRMatrixJ(P_diag);
      P_offd_data = nalu_hypre_CSRMatrixData(P_offd);
      P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
      P_offd_j = nalu_hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
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

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
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

   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dof_func_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(tmp_map_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(coarse_counter, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count_offd, NALU_HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      nalu_hypre_CSRMatrixDestroy(A_ext);
   }

   return nalu_hypre_error_flag;
}


/*---------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGBuildDirInterp
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildDirInterpHost( nalu_hypre_ParCSRMatrix   *A,
                                   NALU_HYPRE_Int            *CF_marker,
                                   nalu_hypre_ParCSRMatrix   *S,
                                   NALU_HYPRE_BigInt         *num_cpts_global,
                                   NALU_HYPRE_Int             num_functions,
                                   NALU_HYPRE_Int            *dof_func,
                                   NALU_HYPRE_Int             debug_flag,
                                   NALU_HYPRE_Real            trunc_factor,
                                   NALU_HYPRE_Int             max_elmts,
                                   nalu_hypre_ParCSRMatrix  **P_ptr)
{
   MPI_Comm                 comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   NALU_HYPRE_MemoryLocation memory_location_P = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real      *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Int        num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_CSRMatrix *S_diag = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int       *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int       *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);

   nalu_hypre_CSRMatrix *S_offd = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int       *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int       *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);

   nalu_hypre_ParCSRMatrix *P;
   NALU_HYPRE_BigInt      *col_map_offd_P;
   NALU_HYPRE_Int         *tmp_map_offd = NULL;

   NALU_HYPRE_Int          *CF_marker_offd = NULL;
   NALU_HYPRE_Int          *dof_func_offd = NULL;

   nalu_hypre_CSRMatrix    *P_diag;
   nalu_hypre_CSRMatrix    *P_offd;

   NALU_HYPRE_Real      *P_diag_data;
   NALU_HYPRE_Int       *P_diag_i;
   NALU_HYPRE_Int       *P_diag_j;
   NALU_HYPRE_Real      *P_offd_data;
   NALU_HYPRE_Int       *P_offd_i;
   NALU_HYPRE_Int       *P_offd_j;

   NALU_HYPRE_Int        P_diag_size, P_offd_size;

   NALU_HYPRE_Int        jj_counter, jj_counter_offd;
   NALU_HYPRE_Int       *jj_count, *jj_count_offd;
   NALU_HYPRE_Int        jj_begin_row, jj_begin_row_offd;
   NALU_HYPRE_Int        jj_end_row, jj_end_row_offd;

   NALU_HYPRE_Int        start_indexing = 0; /* start indexing for P_data at 0 */

   NALU_HYPRE_Int        n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);

   NALU_HYPRE_Int       *fine_to_coarse;
   NALU_HYPRE_Int       *coarse_counter;
   NALU_HYPRE_Int        coarse_shift;
   NALU_HYPRE_BigInt     total_global_cpts;
   NALU_HYPRE_Int        num_cols_P_offd;
   //NALU_HYPRE_BigInt     my_first_cpt;

   NALU_HYPRE_Int        i, i1;
   NALU_HYPRE_Int        j, jl, jj;
   NALU_HYPRE_Int        start;

   NALU_HYPRE_Real       diagonal;
   NALU_HYPRE_Real       sum_N_pos, sum_P_pos;
   NALU_HYPRE_Real       sum_N_neg, sum_P_neg;
   NALU_HYPRE_Real       alfa = 1.0;
   NALU_HYPRE_Real       beta = 1.0;

   NALU_HYPRE_Real       zero = 0.0;
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
   num_threads = nalu_hypre_NumThreads();

   //my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   if (num_cols_A_offd) { CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST); }
   if (num_functions > 1 && num_cols_A_offd)
   {
      dof_func_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
   }

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
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            int_buf_data[index++]
               = dof_func[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                                  dof_func_offd);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }

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
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

   /* RDF: this looks a little tricky, but doable */
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,i1,jj,ns,ne,size,rest) NALU_HYPRE_SMP_SCHEDULE
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
          *  If i is an F-point, interpolation is from the C-points that
          *  strongly influence i.
          *--------------------------------------------------------------------*/

         else
         {
            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];
               if (CF_marker[i1] > 0)
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];
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
      coarse_counter[i + 1] += coarse_counter[i];
      jj_count[i + 1] += jj_count[i];
      jj_count_offd[i + 1] += jj_count_offd[i];
   }
   i = num_threads - 1;
   jj_counter = jj_count[i];
   jj_counter_offd = jj_count_offd[i];

   P_diag_size = jj_counter;

   P_diag_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine + 1,  memory_location_P);
   P_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_diag_size, memory_location_P);
   P_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, P_diag_size, memory_location_P);

   P_diag_i[n_fine] = jj_counter;


   P_offd_size = jj_counter_offd;

   P_offd_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine + 1,  memory_location_P);
   P_offd_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_offd_size, memory_location_P);
   P_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, P_offd_size, memory_location_P);

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

     if (debug_flag==4)
     {
     wall_time = time_getWallclockSeconds() - wall_time;
     nalu_hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
     my_id, wall_time);
     fflush(NULL);
     }*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   /*#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
   #endif
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;*/

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,jl,i1,jj,ns,ne,size,rest,diagonal,jj_counter,jj_counter_offd,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd,sum_P_pos,sum_P_neg,sum_N_pos,sum_N_neg,alfa,beta) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (jl = 0; jl < num_threads; jl++)
   {
      NALU_HYPRE_Int       *P_marker, *P_marker_offd;

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
            jj_begin_row = jj_counter;

            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  P_diag_data[jj_counter] = zero;
                  jj_counter++;
               }

            }
            jj_end_row = jj_counter;

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;
            jj_begin_row_offd = jj_counter_offd;


            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];

                  /*-----------------------------------------------------------
                   * If neighbor i1 is a C-point, set column number in P_offd_j
                   * and initialize interpolation weight to zero.
                   *-----------------------------------------------------------*/

                  if (CF_marker_offd[i1] >= 0)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     P_offd_j[jj_counter_offd]  = i1;
                     P_offd_data[jj_counter_offd] = zero;
                     jj_counter_offd++;
                  }
               }
            }

            jj_end_row_offd = jj_counter_offd;

            diagonal = A_diag_data[A_diag_i[i]];


            /* Loop over ith row of A.  First, the diagonal part of A */
            sum_N_pos = 0;
            sum_N_neg = 0;
            sum_P_pos = 0;
            sum_P_neg = 0;

            for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];
               if (num_functions == 1 || dof_func[i1] == dof_func[i])
               {
                  if (A_diag_data[jj] > 0)
                  {
                     sum_N_pos += A_diag_data[jj];
                  }
                  else
                  {
                     sum_N_neg += A_diag_data[jj];
                  }
               }
               /*--------------------------------------------------------------
                * Case 1: neighbor i1 is a C-point and strongly influences i,
                * accumulate a_{i,i1} into the interpolation weight.
                *--------------------------------------------------------------*/

               if (P_marker[i1] >= jj_begin_row)
               {
                  P_diag_data[P_marker[i1]] += A_diag_data[jj];
                  if (A_diag_data[jj] > 0)
                  {
                     sum_P_pos += A_diag_data[jj];
                  }
                  else
                  {
                     sum_P_neg += A_diag_data[jj];
                  }
               }
            }

            /*----------------------------------------------------------------
             * Still looping over ith row of A. Next, loop over the
             * off-diagonal part of A
             *---------------------------------------------------------------*/

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];
                  if (num_functions == 1 || dof_func_offd[i1] == dof_func[i])
                  {
                     if (A_offd_data[jj] > 0)
                     {
                        sum_N_pos += A_offd_data[jj];
                     }
                     else
                     {
                        sum_N_neg += A_offd_data[jj];
                     }
                  }

                  /*--------------------------------------------------------------
                   * Case 1: neighbor i1 is a C-point and strongly influences i,
                   * accumulate a_{i,i1} into the interpolation weight.
                   *--------------------------------------------------------------*/

                  if (P_marker_offd[i1] >= jj_begin_row_offd)
                  {
                     P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
                     if (A_offd_data[jj] > 0)
                     {
                        sum_P_pos += A_offd_data[jj];
                     }
                     else
                     {
                        sum_P_neg += A_offd_data[jj];
                     }
                  }

               }
            }
            if (sum_P_neg) { alfa = sum_N_neg / sum_P_neg / diagonal; }
            if (sum_P_pos) { beta = sum_N_pos / sum_P_pos / diagonal; }

            /*-----------------------------------------------------------------
             * Set interpolation weight by dividing by the diagonal.
             *-----------------------------------------------------------------*/

            for (jj = jj_begin_row; jj < jj_end_row; jj++)
            {
               if (P_diag_data[jj] > 0)
               {
                  P_diag_data[jj] *= -beta;
               }
               else
               {
                  P_diag_data[jj] *= -alfa;
               }
            }

            for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
            {
               if (P_offd_data[jj] > 0)
               {
                  P_offd_data[jj] *= -beta;
               }
               else
               {
                  P_offd_data[jj] *= -alfa;
               }
            }

         }

         P_offd_i[i + 1] = jj_counter_offd;
      }
      nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(P_marker_offd, NALU_HYPRE_MEMORY_HOST);
   }

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

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      nalu_hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = nalu_hypre_CSRMatrixData(P_diag);
      P_diag_i = nalu_hypre_CSRMatrixI(P_diag);
      P_diag_j = nalu_hypre_CSRMatrixJ(P_diag);
      P_offd_data = nalu_hypre_CSRMatrixData(P_offd);
      P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
      P_offd_j = nalu_hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      NALU_HYPRE_Int *P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
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

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < P_offd_size; i++)
      {
         P_offd_j[i] = nalu_hypre_BinarySearch(tmp_map_offd,
                                          P_offd_j[i],
                                          num_cols_P_offd);
      }
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

   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dof_func_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(tmp_map_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(coarse_counter, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count_offd, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildDirInterp( nalu_hypre_ParCSRMatrix   *A,
                               NALU_HYPRE_Int            *CF_marker,
                               nalu_hypre_ParCSRMatrix   *S,
                               NALU_HYPRE_BigInt         *num_cpts_global,
                               NALU_HYPRE_Int             num_functions,
                               NALU_HYPRE_Int            *dof_func,
                               NALU_HYPRE_Int             debug_flag,
                               NALU_HYPRE_Real            trunc_factor,
                               NALU_HYPRE_Int             max_elmts,
                               NALU_HYPRE_Int             interp_type,
                               nalu_hypre_ParCSRMatrix  **P_ptr)
{
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_GpuProfilingPushRange("DirInterp");
#endif

   NALU_HYPRE_Int ierr = 0;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      ierr = nalu_hypre_BoomerAMGBuildDirInterpDevice(A, CF_marker, S, num_cpts_global, num_functions,
                                                 dof_func,
                                                 debug_flag, trunc_factor, max_elmts,
                                                 interp_type, P_ptr);
   }
   else
#endif
   {
      ierr = nalu_hypre_BoomerAMGBuildDirInterpHost(A, CF_marker, S, num_cpts_global, num_functions, dof_func,
                                               debug_flag, trunc_factor, max_elmts, P_ptr);
   }

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_GpuProfilingPopRange();
#endif

   return ierr;
}

/*------------------------------------------------
 * Drop entries in interpolation matrix P
 * max_elmts == 0 means no limit on rownnz
 *------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_BoomerAMGInterpTruncation( nalu_hypre_ParCSRMatrix *P,
                                 NALU_HYPRE_Real          trunc_factor,
                                 NALU_HYPRE_Int           max_elmts)
{
   if (trunc_factor <= 0.0 && max_elmts == 0)
   {
      return 0;
   }

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(P) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      return nalu_hypre_BoomerAMGInterpTruncationDevice(P, trunc_factor, max_elmts);
   }
   else
#endif
   {
      NALU_HYPRE_Int rescale = 1; // rescale P
      NALU_HYPRE_Int nrm_type = 0; // Use infty-norm of row to perform treshold dropping
      return nalu_hypre_ParCSRMatrixTruncate(P, trunc_factor, max_elmts, rescale, nrm_type);
   }
}

/*---------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGBuildInterpModUnk - this is a modified interpolation for the unknown approach.
 * here we need to pass in a strength matrix built on the entire matrix.
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildInterpModUnk( nalu_hypre_ParCSRMatrix   *A,
                                  NALU_HYPRE_Int            *CF_marker,
                                  nalu_hypre_ParCSRMatrix   *S,
                                  NALU_HYPRE_BigInt         *num_cpts_global,
                                  NALU_HYPRE_Int             num_functions,
                                  NALU_HYPRE_Int            *dof_func,
                                  NALU_HYPRE_Int             debug_flag,
                                  NALU_HYPRE_Real            trunc_factor,
                                  NALU_HYPRE_Int             max_elmts,
                                  nalu_hypre_ParCSRMatrix  **P_ptr)
{

   MPI_Comm       comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real      *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Int        num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt    *col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(A);

   nalu_hypre_CSRMatrix *S_diag = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int       *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int       *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);

   nalu_hypre_CSRMatrix *S_offd = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int       *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int       *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);

   nalu_hypre_ParCSRMatrix *P;
   NALU_HYPRE_BigInt       *col_map_offd_P;
   NALU_HYPRE_Int          *tmp_map_offd;

   NALU_HYPRE_Int          *CF_marker_offd = NULL;
   NALU_HYPRE_Int          *dof_func_offd = NULL;

   nalu_hypre_CSRMatrix *A_ext;

   NALU_HYPRE_Real      *A_ext_data = NULL;
   NALU_HYPRE_Int       *A_ext_i = NULL;
   NALU_HYPRE_BigInt    *A_ext_j = NULL;

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
   NALU_HYPRE_Int        jj_begin_row, jj_begin_row_offd;
   NALU_HYPRE_Int        jj_end_row, jj_end_row_offd;

   NALU_HYPRE_Int        start_indexing = 0; /* start indexing for P_data at 0 */

   NALU_HYPRE_Int        n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);

   NALU_HYPRE_Int        strong_f_marker;

   NALU_HYPRE_Int       *fine_to_coarse;
   //NALU_HYPRE_Int       *fine_to_coarse_offd;
   NALU_HYPRE_Int       *coarse_counter;
   NALU_HYPRE_Int        coarse_shift;
   NALU_HYPRE_BigInt     total_global_cpts;
   NALU_HYPRE_Int        num_cols_P_offd;
   //NALU_HYPRE_BigInt     my_first_cpt;

   NALU_HYPRE_Int        i, i1, i2;
   NALU_HYPRE_Int        j, jl, jj, jj1;
   NALU_HYPRE_Int        kc;
   NALU_HYPRE_BigInt     big_k;
   NALU_HYPRE_Int        start;
   NALU_HYPRE_Int        sgn;
   NALU_HYPRE_Int        c_num;

   NALU_HYPRE_Real       diagonal;
   NALU_HYPRE_Real       sum;
   NALU_HYPRE_Real       distribute;

   NALU_HYPRE_Real       zero = 0.0;
   NALU_HYPRE_Real       one  = 1.0;

   NALU_HYPRE_Int        my_id;
   NALU_HYPRE_Int        num_procs;
   NALU_HYPRE_Int        num_threads;
   NALU_HYPRE_Int        num_sends;
   NALU_HYPRE_Int        index;
   NALU_HYPRE_Int        ns, ne, size, rest;
   NALU_HYPRE_Int        print_level = 0;
   NALU_HYPRE_Int       *int_buf_data;

   NALU_HYPRE_BigInt col_1 = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   NALU_HYPRE_Int local_numrows = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_BigInt col_n = col_1 + local_numrows;

   NALU_HYPRE_Real       wall_time;  /* for debugging instrumentation  */

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   num_threads = nalu_hypre_NumThreads();


   //my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/

   if (debug_flag < 0)
   {
      debug_flag = -debug_flag;
      print_level = 1;
   }

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   if (num_cols_A_offd) { CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST); }
   if (num_functions > 1 && num_cols_A_offd)
   {
      dof_func_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
   }

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
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            int_buf_data[index++]
               = dof_func[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                                  dof_func_offd);

      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d     Interp: Comm 1 CF_marker =    %f\n",
                   my_id, wall_time);
      fflush(NULL);
   }

   /*----------------------------------------------------------------------
    * Get the ghost rows of A
    *---------------------------------------------------------------------*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   if (num_procs > 1)
   {
      A_ext      = nalu_hypre_ParCSRMatrixExtractBExt(A, A, 1);
      A_ext_i    = nalu_hypre_CSRMatrixI(A_ext);
      A_ext_j    = nalu_hypre_CSRMatrixBigJ(A_ext);
      A_ext_data = nalu_hypre_CSRMatrixData(A_ext);
   }

   index = 0;
   for (i = 0; i < num_cols_A_offd; i++)
   {
      for (j = A_ext_i[i]; j < A_ext_i[i + 1]; j++)
      {
         big_k = A_ext_j[j];
         if (big_k >= col_1 && big_k < col_n)
         {
            A_ext_j[index] = big_k - col_1;
            A_ext_data[index++] = A_ext_data[j];
         }
         else
         {
            kc = nalu_hypre_BigBinarySearch(col_map_offd, big_k, num_cols_A_offd);
            if (kc > -1)
            {
               A_ext_j[index] = (NALU_HYPRE_BigInt)(-kc - 1);
               A_ext_data[index++] = A_ext_data[j];
            }
         }
      }
      A_ext_i[i] = index;
   }
   for (i = num_cols_A_offd; i > 0; i--)
   {
      A_ext_i[i] = A_ext_i[i - 1];
   }
   if (num_procs > 1) { A_ext_i[0] = 0; }

   if (debug_flag == 4)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      nalu_hypre_printf("Proc = %d  Interp: Comm 2   Get A_ext =  %f\n",
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
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < n_fine; i++) { fine_to_coarse[i] = -1; }

   jj_counter = start_indexing;
   jj_counter_offd = start_indexing;

   /*-----------------------------------------------------------------------
    *  Loop over fine grid.
    *-----------------------------------------------------------------------*/

   /* RDF: this looks a little tricky, but doable */
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,i1,jj,ns,ne,size,rest) NALU_HYPRE_SMP_SCHEDULE
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
          *  If i is an F-point, interpolation is from the C-points that
          *  strongly influence i.
          *--------------------------------------------------------------------*/

         else
         {
            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];
               if (CF_marker[i1] >= 0)
               {
                  jj_count[j]++;
               }
            }

            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];
                  if (CF_marker_offd[i1] >= 0)
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

     if (debug_flag==4)
     {
     wall_time = time_getWallclockSeconds() - wall_time;
     nalu_hypre_printf("Proc = %d     Interp: Comm 4 FineToCoarse = %f\n",
     my_id, wall_time);
     fflush(NULL);
     }*/

   if (debug_flag == 4) { wall_time = time_getWallclockSeconds(); }

   /*#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
   #endif
   for (i = 0; i < n_fine; i++) fine_to_coarse[i] -= my_first_cpt;*/

   /*-----------------------------------------------------------------------
    *  Loop over fine grid points.
    *-----------------------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,jl,i1,i2,jj,jj1,ns,ne,size,rest,sum,diagonal,distribute,P_marker,P_marker_offd,strong_f_marker,jj_counter,jj_counter_offd,sgn,c_num,jj_begin_row,jj_end_row,jj_begin_row_offd,jj_end_row_offd) NALU_HYPRE_SMP_SCHEDULE
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
      strong_f_marker = -2;

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
            jj_begin_row = jj_counter;

            for (jj = S_diag_i[i]; jj < S_diag_i[i + 1]; jj++)
            {
               i1 = S_diag_j[jj];

               /*--------------------------------------------------------------
                * If neighbor i1 is a C-point, set column number in P_diag_j
                * and initialize interpolation weight to zero.
                *--------------------------------------------------------------*/

               if (CF_marker[i1] >= 0)
               {
                  P_marker[i1] = jj_counter;
                  P_diag_j[jj_counter]    = fine_to_coarse[i1];
                  P_diag_data[jj_counter] = zero;
                  jj_counter++;
               }

               /*--------------------------------------------------------------
                * If neighbor i1 is an F-point, mark it as a strong F-point
                * whose connection needs to be distributed.
                *--------------------------------------------------------------*/

               else if (CF_marker[i1] != -3)
               {
                  P_marker[i1] = strong_f_marker;
               }
            }
            jj_end_row = jj_counter;

            /* Off-Diagonal part of P */
            P_offd_i[i] = jj_counter_offd;
            jj_begin_row_offd = jj_counter_offd;


            if (num_procs > 1)
            {
               for (jj = S_offd_i[i]; jj < S_offd_i[i + 1]; jj++)
               {
                  i1 = S_offd_j[jj];

                  /*-----------------------------------------------------------
                   * If neighbor i1 is a C-point, set column number in P_offd_j
                   * and initialize interpolation weight to zero.
                   *-----------------------------------------------------------*/

                  if (CF_marker_offd[i1] >= 0)
                  {
                     P_marker_offd[i1] = jj_counter_offd;
                     /*P_offd_j[jj_counter_offd]  = fine_to_coarse_offd[i1];*/
                     P_offd_j[jj_counter_offd]  = i1;
                     P_offd_data[jj_counter_offd] = zero;
                     jj_counter_offd++;
                  }

                  /*-----------------------------------------------------------
                   * If neighbor i1 is an F-point, mark it as a strong F-point
                   * whose connection needs to be distributed.
                   *-----------------------------------------------------------*/

                  else if (CF_marker_offd[i1] != -3)
                  {
                     P_marker_offd[i1] = strong_f_marker;
                  }
               }
            }

            jj_end_row_offd = jj_counter_offd;

            diagonal = A_diag_data[A_diag_i[i]];


            /* Loop over ith row of A.  First, the diagonal part of A */

            for (jj = A_diag_i[i] + 1; jj < A_diag_i[i + 1]; jj++)
            {
               i1 = A_diag_j[jj];

               /*--------------------------------------------------------------
                * Case 1: neighbor i1 is a C-point and strongly influences i,
                * accumulate a_{i,i1} into the interpolation weight.
                *--------------------------------------------------------------*/

               if (P_marker[i1] >= jj_begin_row)
               {
                  P_diag_data[P_marker[i1]] += A_diag_data[jj];
               }

               /*--------------------------------------------------------------
                * Case 2: neighbor i1 is an F-point and strongly influences i,
                * distribute a_{i,i1} to C-points that strongly infuence i.
                * Note: currently no distribution to the diagonal in this case.

                HERE, we only want to distribut to points of the SAME function type

                *--------------------------------------------------------------*/

               else if (P_marker[i1] == strong_f_marker)
               {
                  sum = zero;

                  /*-----------------------------------------------------------
                   * Loop over row of A for point i1 and calculate the sum
                   * of the connections to c-points that strongly influence i.
                   *-----------------------------------------------------------*/
                  sgn = 1;
                  if (A_diag_data[A_diag_i[i1]] < 0) { sgn = -1; }
                  /* Diagonal block part of row i1 */
                  for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                  {
                     i2 = A_diag_j[jj1];
                     if (num_functions == 1 || dof_func[i1] == dof_func[i2])
                     {

                        if (P_marker[i2] >= jj_begin_row &&
                            (sgn * A_diag_data[jj1]) < 0 )
                        {
                           sum += A_diag_data[jj1];
                        }
                     }

                  }

                  /* Off-Diagonal block part of row i1 */
                  if (num_procs > 1)
                  {
                     for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                     {
                        i2 = A_offd_j[jj1];
                        if (num_functions == 1 || dof_func[i1] == dof_func[i2])
                        {
                           if (P_marker_offd[i2] >= jj_begin_row_offd
                               && (sgn * A_offd_data[jj1]) < 0)
                           {
                              sum += A_offd_data[jj1];
                           }
                        }
                     }
                  }

                  if (sum != 0)
                  {
                     distribute = A_diag_data[jj] / sum;

                     /*-----------------------------------------------------------
                      * Loop over row of A for point i1 and do the distribution.
                      *-----------------------------------------------------------*/

                     /* Diagonal block part of row i1 */
                     for (jj1 = A_diag_i[i1]; jj1 < A_diag_i[i1 + 1]; jj1++)
                     {
                        i2 = A_diag_j[jj1];
                        if (num_functions == 1 || dof_func[i1] == dof_func[i2])
                        {
                           if (P_marker[i2] >= jj_begin_row
                               && (sgn * A_diag_data[jj1]) < 0)
                           {
                              P_diag_data[P_marker[i2]]
                              += distribute * A_diag_data[jj1];
                           }
                        }

                     }

                     /* Off-Diagonal block part of row i1 */
                     if (num_procs > 1)
                     {
                        for (jj1 = A_offd_i[i1]; jj1 < A_offd_i[i1 + 1]; jj1++)
                        {
                           i2 = A_offd_j[jj1];
                           if (num_functions == 1 || dof_func[i1] == dof_func[i2])
                           {
                              if (P_marker_offd[i2] >= jj_begin_row_offd
                                  && (sgn * A_offd_data[jj1]) < 0)
                              {
                                 P_offd_data[P_marker_offd[i2]]
                                 += distribute * A_offd_data[jj1];
                              }
                           }
                        }

                     }
                  }
                  else /* sum = 0 - only add to diag if the same function type */
                  {
                     if (num_functions == 1 || dof_func[i] == dof_func[i1])
                     {
                        diagonal += A_diag_data[jj];
                     }
                  }
               }

               /*--------------------------------------------------------------
                * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                * into the diagonal. (only if the same function type)
                *--------------------------------------------------------------*/

               else if (CF_marker[i1] != -3)
               {
                  if (num_functions == 1 || dof_func[i] == dof_func[i1])
                  {
                     diagonal += A_diag_data[jj];
                  }
               }

            }


            /*----------------------------------------------------------------
             * Still looping over ith row of A. Next, loop over the
             * off-diagonal part of A
             *---------------------------------------------------------------*/

            if (num_procs > 1)
            {
               for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
               {
                  i1 = A_offd_j[jj];

                  /*--------------------------------------------------------------
                   * Case 1: neighbor i1 is a C-point and strongly influences i,
                   * accumulate a_{i,i1} into the interpolation weight.
                   *--------------------------------------------------------------*/

                  if (P_marker_offd[i1] >= jj_begin_row_offd)
                  {
                     P_offd_data[P_marker_offd[i1]] += A_offd_data[jj];
                  }

                  /*------------------------------------------------------------
                   * Case 2: neighbor i1 is an F-point and strongly influences i,
                   * distribute a_{i,i1} to C-points that strongly infuence i.
                   * Note: currently no distribution to the diagonal in this case.

                   AGAIN, we only want to distribut to points of the SAME function type

                   *-----------------------------------------------------------*/

                  else if (P_marker_offd[i1] == strong_f_marker)
                  {
                     sum = zero;

                     /*---------------------------------------------------------
                      * Loop over row of A_ext for point i1 and calculate the sum
                      * of the connections to c-points that strongly influence i.
                      *---------------------------------------------------------*/

                     /* find row number */
                     c_num = A_offd_j[jj];

                     sgn = 1;
                     if (A_ext_data[A_ext_i[c_num]] < 0) { sgn = -1; }
                     for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num + 1]; jj1++)
                     {
                        i2 = (NALU_HYPRE_Int)A_ext_j[jj1];
                        if (num_functions == 1 || dof_func[i1] == dof_func[i2])
                        {
                           if (i2 > -1)
                           {
                              /* in the diagonal block */
                              if (P_marker[i2] >= jj_begin_row
                                  && (sgn * A_ext_data[jj1]) < 0)
                              {
                                 sum += A_ext_data[jj1];
                              }
                           }
                           else
                           {
                              /* in the off_diagonal block  */
                              if (P_marker_offd[-i2 - 1] >= jj_begin_row_offd
                                  && (sgn * A_ext_data[jj1]) < 0)
                              {
                                 sum += A_ext_data[jj1];
                              }
                           }

                        }
                     }
                     if (sum != 0)
                     {
                        distribute = A_offd_data[jj] / sum;
                        /*---------------------------------------------------------
                         * Loop over row of A_ext for point i1 and do
                         * the distribution.
                         *--------------------------------------------------------*/

                        /* Diagonal block part of row i1 */

                        for (jj1 = A_ext_i[c_num]; jj1 < A_ext_i[c_num + 1]; jj1++)
                        {
                           i2 = (NALU_HYPRE_Int)A_ext_j[jj1];
                           if (num_functions == 1 || dof_func[i1] == dof_func[i2])
                           {
                              if (i2 > -1) /* in the diagonal block */
                              {
                                 if (P_marker[i2] >= jj_begin_row
                                     && (sgn * A_ext_data[jj1]) < 0)
                                 {
                                    P_diag_data[P_marker[i2]]
                                    += distribute * A_ext_data[jj1];
                                 }
                              }
                              else
                              {
                                 /* in the off_diagonal block  */
                                 if (P_marker_offd[-i2 - 1] >= jj_begin_row_offd
                                     && (sgn * A_ext_data[jj1]) < 0)
                                    P_offd_data[P_marker_offd[-i2 - 1]]
                                    += distribute * A_ext_data[jj1];
                              }
                           }
                        }
                     }
                     else /* sum = 0 */
                     {
                        if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                        {
                           diagonal += A_offd_data[jj];
                        }
                     }
                  }

                  /*-----------------------------------------------------------
                   * Case 3: neighbor i1 weakly influences i, accumulate a_{i,i1}
                   * into the diagonal.
                   *-----------------------------------------------------------*/

                  else if (CF_marker_offd[i1] != -3)
                  {
                     if (num_functions == 1 || dof_func[i] == dof_func_offd[i1])
                     {
                        diagonal += A_offd_data[jj];
                     }
                  }

               }
            }

            /*-----------------------------------------------------------------
             * Set interpolation weight by dividing by the diagonal.
             *-----------------------------------------------------------------*/

            if (diagonal == 0.0)
            {
               if (print_level)
               {
                  nalu_hypre_printf(" Warning! zero diagonal! Proc id %d row %d\n", my_id, i);
               }
               for (jj = jj_begin_row; jj < jj_end_row; jj++)
               {
                  P_diag_data[jj] = 0.0;
               }
               for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
               {
                  P_offd_data[jj] = 0.0;
               }
            }
            else
            {
               for (jj = jj_begin_row; jj < jj_end_row; jj++)
               {
                  P_diag_data[jj] /= -diagonal;
               }
               for (jj = jj_begin_row_offd; jj < jj_end_row_offd; jj++)
               {
                  P_offd_data[jj] /= -diagonal;
               }
            }
         }

         strong_f_marker--;

         P_offd_i[i + 1] = jj_counter_offd;
      }
      nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(P_marker_offd, NALU_HYPRE_MEMORY_HOST);
   }

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

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      nalu_hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_diag_data = nalu_hypre_CSRMatrixData(P_diag);
      P_diag_i = nalu_hypre_CSRMatrixI(P_diag);
      P_diag_j = nalu_hypre_CSRMatrixJ(P_diag);
      P_offd_data = nalu_hypre_CSRMatrixData(P_offd);
      P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
      P_offd_j = nalu_hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[n_fine];
      P_offd_size = P_offd_i[n_fine];
   }

   num_cols_P_offd = 0;
   if (P_offd_size)
   {
      P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
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

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
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

   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dof_func_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(tmp_map_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(coarse_counter, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count_offd, NALU_HYPRE_MEMORY_HOST);

   if (num_procs > 1) { nalu_hypre_CSRMatrixDestroy(A_ext); }

   return nalu_hypre_error_flag;

}

/*---------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGTruncandBuild
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGTruncandBuild( nalu_hypre_ParCSRMatrix   *P,
                              NALU_HYPRE_Real                trunc_factor,
                              NALU_HYPRE_Int                 max_elmts)
{
   nalu_hypre_CSRMatrix *P_offd = nalu_hypre_ParCSRMatrixOffd(P);
   nalu_hypre_ParCSRCommPkg   *commpkg_P = nalu_hypre_ParCSRMatrixCommPkg(P);
   NALU_HYPRE_BigInt          *col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(P);
   NALU_HYPRE_Int             *P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
   NALU_HYPRE_Int             *P_offd_j = nalu_hypre_CSRMatrixJ(P_offd);
   NALU_HYPRE_Int              num_cols_offd = nalu_hypre_CSRMatrixNumCols(P_offd);
   NALU_HYPRE_Int              n_fine = nalu_hypre_CSRMatrixNumRows(P_offd);

   NALU_HYPRE_BigInt          *new_col_map_offd;
   NALU_HYPRE_Int             *tmp_map_offd = NULL;

   NALU_HYPRE_Int              P_offd_size = 0, new_num_cols_offd;

   NALU_HYPRE_Int             *P_marker;

   NALU_HYPRE_Int              i;

   NALU_HYPRE_Int              index;

   /* Compress P, removing coefficients smaller than trunc_factor * Max */

   if (trunc_factor != 0.0 || max_elmts > 0)
   {
      nalu_hypre_BoomerAMGInterpTruncation(P, trunc_factor, max_elmts);
      P_offd_j = nalu_hypre_CSRMatrixJ(P_offd);
      P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
      P_offd_size = P_offd_i[n_fine];
   }

   new_num_cols_offd = 0;
   if (P_offd_size)
   {
      P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      /*#define NALU_HYPRE_SMP_PRIVATE i
      #include "../utilities/nalu_hypre_smp_forloop.h"*/
      for (i = 0; i < num_cols_offd; i++)
      {
         P_marker[i] = 0;
      }

      for (i = 0; i < P_offd_size; i++)
      {
         index = P_offd_j[i];
         if (!P_marker[index])
         {
            new_num_cols_offd++;
            P_marker[index] = 1;
         }
      }

      tmp_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, new_num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      new_col_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, new_num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      index = 0;
      for (i = 0; i < new_num_cols_offd; i++)
      {
         while (P_marker[index] == 0) { index++; }
         tmp_map_offd[i] = index++;
      }

      /*#define NALU_HYPRE_SMP_PRIVATE i
      #include "../utilities/nalu_hypre_smp_forloop.h"*/
      for (i = 0; i < P_offd_size; i++)
         P_offd_j[i] = nalu_hypre_BinarySearch(tmp_map_offd,
                                          P_offd_j[i],
                                          new_num_cols_offd);
   }

   index = 0;
   for (i = 0; i < new_num_cols_offd; i++)
   {
      while (P_marker[index] == 0) { index++; }

      new_col_map_offd[i] = col_map_offd[index];
      index++;
   }

   if (P_offd_size) { nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST); }

   if (new_num_cols_offd)
   {
      nalu_hypre_TFree(tmp_map_offd, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(col_map_offd, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParCSRMatrixColMapOffd(P) = new_col_map_offd;
      nalu_hypre_CSRMatrixNumCols(P_offd) = new_num_cols_offd;
   }

   if (commpkg_P != NULL) { nalu_hypre_MatvecCommPkgDestroy(commpkg_P); }
   nalu_hypre_MatvecCommPkgCreate(P);

   return nalu_hypre_error_flag;

}

nalu_hypre_ParCSRMatrix *nalu_hypre_CreateC( nalu_hypre_ParCSRMatrix  *A,
                                   NALU_HYPRE_Real w)
{
   MPI_Comm    comm = nalu_hypre_ParCSRMatrixComm(A);

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);

   NALU_HYPRE_Real *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int  *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int  *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);

   NALU_HYPRE_Real *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int  *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int  *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);

   NALU_HYPRE_BigInt *row_starts = nalu_hypre_ParCSRMatrixRowStarts(A);
   NALU_HYPRE_BigInt *col_map_offd_A = nalu_hypre_ParCSRMatrixColMapOffd(A);
   NALU_HYPRE_Int    num_rows = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int    num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(A);

   nalu_hypre_ParCSRMatrix *C;
   nalu_hypre_CSRMatrix *C_diag;
   nalu_hypre_CSRMatrix *C_offd;

   NALU_HYPRE_Real      *C_diag_data;
   NALU_HYPRE_Int       *C_diag_i;
   NALU_HYPRE_Int       *C_diag_j;

   NALU_HYPRE_Real      *C_offd_data;
   NALU_HYPRE_Int       *C_offd_i;
   NALU_HYPRE_Int       *C_offd_j;
   NALU_HYPRE_BigInt    *col_map_offd_C;

   NALU_HYPRE_Int i, j, index;
   NALU_HYPRE_Real  invdiag;
   NALU_HYPRE_Real  w_local = w;

   C = nalu_hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_rows, row_starts,
                                row_starts, num_cols_offd, A_diag_i[num_rows], A_offd_i[num_rows]);

   nalu_hypre_ParCSRMatrixInitialize(C);

   C_diag = nalu_hypre_ParCSRMatrixDiag(C);
   C_offd = nalu_hypre_ParCSRMatrixOffd(C);

   C_diag_i = nalu_hypre_CSRMatrixI(C_diag);
   C_diag_j = nalu_hypre_CSRMatrixJ(C_diag);
   C_diag_data = nalu_hypre_CSRMatrixData(C_diag);

   C_offd_i = nalu_hypre_CSRMatrixI(C_offd);
   C_offd_j = nalu_hypre_CSRMatrixJ(C_offd);
   C_offd_data = nalu_hypre_CSRMatrixData(C_offd);

   col_map_offd_C = nalu_hypre_ParCSRMatrixColMapOffd(C);

   for (i = 0; i < num_cols_offd; i++)
   {
      col_map_offd_C[i] = col_map_offd_A[i];
   }

   for (i = 0; i < num_rows; i++)
   {
      index = A_diag_i[i];
      invdiag = -w / A_diag_data[index];
      C_diag_data[index] = 1.0 - w;
      C_diag_j[index] = A_diag_j[index];
      if (w == 0)
      {
         w_local = fabs(A_diag_data[index]);
         for (j = index + 1; j < A_diag_i[i + 1]; j++)
         {
            w_local += fabs(A_diag_data[j]);
         }
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            w_local += fabs(A_offd_data[j]);
         }
         invdiag = -1 / w_local;
         C_diag_data[index] = 1.0 - A_diag_data[index] / w_local;
      }
      C_diag_i[i] = index;
      C_offd_i[i] = A_offd_i[i];
      for (j = index + 1; j < A_diag_i[i + 1]; j++)
      {
         C_diag_data[j] = A_diag_data[j] * invdiag;
         C_diag_j[j] = A_diag_j[j];
      }
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         C_offd_data[j] = A_offd_data[j] * invdiag;
         C_offd_j[j] = A_offd_j[j];
      }
   }
   C_diag_i[num_rows] = A_diag_i[num_rows];
   C_offd_i[num_rows] = A_offd_i[num_rows];

   return C;
}

/* RL */
NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildInterpOnePntHost( nalu_hypre_ParCSRMatrix  *A,
                                      NALU_HYPRE_Int           *CF_marker,
                                      nalu_hypre_ParCSRMatrix  *S,
                                      NALU_HYPRE_BigInt        *num_cpts_global,
                                      NALU_HYPRE_Int            num_functions,
                                      NALU_HYPRE_Int           *dof_func,
                                      NALU_HYPRE_Int            debug_flag,
                                      nalu_hypre_ParCSRMatrix **P_ptr)
{
   MPI_Comm                 comm     = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   NALU_HYPRE_MemoryLocation memory_location_P = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   nalu_hypre_CSRMatrix         *A_diag      = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real              *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int               *A_diag_i    = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int               *A_diag_j    = nalu_hypre_CSRMatrixJ(A_diag);

   nalu_hypre_CSRMatrix         *A_offd      = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real              *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int               *A_offd_i    = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int               *A_offd_j    = nalu_hypre_CSRMatrixJ(A_offd);

   NALU_HYPRE_Int                num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   //NALU_HYPRE_Int               *col_map_offd_A    = nalu_hypre_ParCSRMatrixColMapOffd(A);

   nalu_hypre_CSRMatrix         *S_diag   = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int               *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int               *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);

   nalu_hypre_CSRMatrix         *S_offd   = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int               *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int               *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);

   /* Interpolation matrix P */
   nalu_hypre_ParCSRMatrix      *P;
   /* csr's */
   nalu_hypre_CSRMatrix    *P_diag;
   nalu_hypre_CSRMatrix    *P_offd;
   /* arrays */
   NALU_HYPRE_Real         *P_diag_data;
   NALU_HYPRE_Int          *P_diag_i;
   NALU_HYPRE_Int          *P_diag_j;
   NALU_HYPRE_Real         *P_offd_data;
   NALU_HYPRE_Int          *P_offd_i;
   NALU_HYPRE_Int          *P_offd_j;
   NALU_HYPRE_Int           num_cols_offd_P;
   NALU_HYPRE_Int          *tmp_map_offd = NULL;
   NALU_HYPRE_BigInt       *col_map_offd_P = NULL;
   /* CF marker off-diag part */
   NALU_HYPRE_Int          *CF_marker_offd = NULL;
   /* func type off-diag part */
   NALU_HYPRE_Int          *dof_func_offd  = NULL;
   /* nnz */
   NALU_HYPRE_Int           nnz_diag, nnz_offd, cnt_diag, cnt_offd;
   NALU_HYPRE_Int          *marker_diag, *marker_offd = NULL;
   /* local size */
   NALU_HYPRE_Int           n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);
   /* number of C-pts */
   NALU_HYPRE_Int           n_cpts = 0;
   /* fine to coarse mapping: diag part and offd part */
   NALU_HYPRE_Int          *fine_to_coarse;
   NALU_HYPRE_BigInt       *fine_to_coarse_offd = NULL;
   NALU_HYPRE_BigInt        total_global_cpts, my_first_cpt;
   NALU_HYPRE_Int           my_id, num_procs;
   NALU_HYPRE_Int           num_sends;
   NALU_HYPRE_Int          *int_buf_data = NULL;
   NALU_HYPRE_BigInt       *big_int_buf_data = NULL;
   //NALU_HYPRE_Int col_start = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   //NALU_HYPRE_Int col_end   = col_start + n_fine;

   NALU_HYPRE_Int           i, j, i1, j1, k1, index, start;
   NALU_HYPRE_Int          *max_abs_cij;
   char               *max_abs_diag_offd;
   NALU_HYPRE_Real          max_abs_aij, vv;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   my_first_cpt = num_cpts_global[0];
   if (my_id == (num_procs - 1)) { total_global_cpts = num_cpts_global[1]; }
   nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/
   /* CF marker for the off-diag columns */
   if (num_cols_A_offd)
   {
      CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
   }
   /* function type indicator for the off-diag columns */
   if (num_functions > 1 && num_cols_A_offd)
   {
      dof_func_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
   }
   /* if CommPkg of A is not present, create it */
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }
   /* number of sends to do (number of procs) */
   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   /* send buffer, of size send_map_starts[num_sends]),
    * i.e., number of entries to send */
   int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                NALU_HYPRE_MEMORY_HOST);

   /* copy CF markers of elements to send to buffer
    * RL: why copy them with two for loops? Why not just loop through all in one */
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      /* start pos of elements sent to send_proc[i] */
      start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      /* loop through all elems to send_proc[i] */
      for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         /* CF marker of send_map_elemts[j] */
         int_buf_data[index++] = CF_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }
   /* create a handle to start communication. 11: for integer */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);
   /* destroy the handle to finish communication */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /* do a similar communication for dof_func */
   if (num_functions > 1)
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            int_buf_data[index++] = dof_func[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, dof_func_offd);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of P and fill in fine_to_coarse mapping,
    *  and find the most strongly influencing C-pt for each F-pt
    *-----------------------------------------------------------------------*/
   /* nnz in diag and offd parts */
   cnt_diag = 0;
   cnt_offd = 0;
   max_abs_cij       = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   max_abs_diag_offd = nalu_hypre_CTAlloc(char, n_fine, NALU_HYPRE_MEMORY_HOST);
   fine_to_coarse    = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);

   /* markers initialized as zeros */
   marker_diag = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < n_fine; i++)
   {
      /*--------------------------------------------------------------------
       *  If i is a C-point, interpolation is the identity. Also set up
       *  mapping vector.
       *--------------------------------------------------------------------*/
      if (CF_marker[i] >= 0)
      {
         //fine_to_coarse[i] = my_first_cpt + n_cpts;
         fine_to_coarse[i] = n_cpts;
         n_cpts++;
         continue;
      }

      /* mark all the strong connections: in S */
      NALU_HYPRE_Int MARK = i + 1;
      /* loop through row i of S, diag part  */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         marker_diag[S_diag_j[j]] = MARK;
      }
      /* loop through row i of S, offd part  */
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            j1 = S_offd_j[j];
            marker_offd[j1] = MARK;
         }
      }

      fine_to_coarse[i] = -1;
      /*---------------------------------------------------------------------------
       *  If i is an F-pt, interpolation is from the most strongly influencing C-pt
       *  Find this C-pt and save it
       *--------------------------------------------------------------------------*/
      /* if we failed to find any strong C-pt, mark this point as an 'n' */
      char marker = 'n';
      /* max abs val */
      max_abs_aij = -1.0;
      /* loop through row i of A, diag part  */
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         i1 = A_diag_j[j];
         vv = fabs(A_diag_data[j]);
#if 0
         /* !!! this is a hack just for code verification purpose !!!
            it basically says:
            1. if we see |a_ij| < 1e-14, force it to be 1e-14
            2. if we see |a_ij| == the max(|a_ij|) so far exactly,
               replace it if the j idx is smaller
            Reasons:
            1. numerical round-off for eps-level values
            2. entries in CSR rows may be listed in different orders
         */
         vv = vv < 1e-14 ? 1e-14 : vv;
         if (CF_marker[i1] >= 0 && marker_diag[i1] == MARK &&
             vv == max_abs_aij && i1 < max_abs_cij[i])
         {
            /* mark it as a 'd' */
            marker         = 'd';
            max_abs_cij[i] = i1;
            max_abs_aij    = vv;
            continue;
         }
#endif
         /* it is a strong C-pt and has abs val larger than what have seen */
         if (CF_marker[i1] >= 0 && marker_diag[i1] == MARK && vv > max_abs_aij)
         {
            /* mark it as a 'd' */
            marker         = 'd';
            max_abs_cij[i] = i1;
            max_abs_aij    = vv;
         }
      }
      /* offd part */
      if (num_procs > 1)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            i1 = A_offd_j[j];
            vv = fabs(A_offd_data[j]);
            if (CF_marker_offd[i1] >= 0 && marker_offd[i1] == MARK && vv > max_abs_aij)
            {
               /* mark it as an 'o' */
               marker         = 'o';
               max_abs_cij[i] = i1;
               max_abs_aij    = vv;
            }
         }
      }

      max_abs_diag_offd[i] = marker;

      if (marker == 'd')
      {
         cnt_diag ++;
      }
      else if (marker == 'o')
      {
         cnt_offd ++;
      }
   }

   nnz_diag = cnt_diag + n_cpts;
   nnz_offd = cnt_offd;

   /*------------- allocate arrays */
   P_diag_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine + 1, memory_location_P);
   P_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_diag, memory_location_P);
   P_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, nnz_diag, memory_location_P);

   /* not in ``if num_procs > 1'',
    * allocation needed even for empty CSR */
   P_offd_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_fine + 1, memory_location_P);
   P_offd_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_offd, memory_location_P);
   P_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, nnz_offd, memory_location_P);

   /* redundant */
   P_diag_i[0] = 0;
   P_offd_i[0] = 0;

   /* reset counters */
   cnt_diag = 0;
   cnt_offd = 0;

   /*-----------------------------------------------------------------------
    *  Send and receive fine_to_coarse info.
    *-----------------------------------------------------------------------*/
   fine_to_coarse_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
   big_int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                    NALU_HYPRE_MEMORY_HOST);
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         big_int_buf_data[index++] = my_first_cpt
                                     + (NALU_HYPRE_BigInt)fine_to_coarse[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(21, comm_pkg, big_int_buf_data, fine_to_coarse_offd);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /*-----------------------------------------------------------------------
    *  Second Pass: Populate P
    *-----------------------------------------------------------------------*/
   for (i = 0; i < n_fine; i++)
   {
      if (CF_marker[i] >= 0)
      {
         /*--------------------------------------------------------------------
          *  If i is a C-point, interpolation is the identity.
          *--------------------------------------------------------------------*/
         //P_diag_j[cnt_diag] = fine_to_coarse[i] - my_first_cpt;
         P_diag_j[cnt_diag] = fine_to_coarse[i];
         P_diag_data[cnt_diag++] = 1.0;
      }
      else
      {
         /*---------------------------------------------------------------------------
          *  If i is an F-pt, interpolation is from the most strongly influencing C-pt
          *--------------------------------------------------------------------------*/
         if (max_abs_diag_offd[i] == 'd')
         {
            /* on diag part of P */
            j = max_abs_cij[i];
            //P_diag_j[cnt_diag] = fine_to_coarse[j] - my_first_cpt;
            P_diag_j[cnt_diag] = fine_to_coarse[j];
            P_diag_data[cnt_diag++] = 1.0;
         }
         else if (max_abs_diag_offd[i] == 'o')
         {
            /* on offd part of P */
            j = max_abs_cij[i];
            P_offd_j[cnt_offd] = j;
            P_offd_data[cnt_offd++] = 1.0;
         }
      }

      P_diag_i[i + 1] = cnt_diag;
      P_offd_i[i + 1] = cnt_offd;
   }

   nalu_hypre_assert(cnt_diag == nnz_diag);
   nalu_hypre_assert(cnt_offd == nnz_offd);

   /* num of cols in the offd part of P */
   num_cols_offd_P = 0;

   /* marker_offd: all -1 */
   for (i = 0; i < num_cols_A_offd; i++)
   {
      marker_offd[i] = -1;
   }
   for (i = 0; i < nnz_offd; i++)
   {
      i1 = P_offd_j[i];
      if (marker_offd[i1] == -1)
      {
         num_cols_offd_P++;
         marker_offd[i1] = 1;
      }
   }

   /* col_map_offd_P: the col indices of the offd of P
    * we first keep them be the offd-idx of A */
   col_map_offd_P = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_P, NALU_HYPRE_MEMORY_HOST);
   tmp_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_P, NALU_HYPRE_MEMORY_HOST);
   for (i = 0, i1 = 0; i < num_cols_A_offd; i++)
   {
      if (marker_offd[i] == 1)
      {
         tmp_map_offd[i1++] = i;
      }
   }
   nalu_hypre_assert(i1 == num_cols_offd_P);

   /* now, adjust P_offd_j to local idx w.r.t col_map_offd_R
    * by searching */
   for (i = 0; i < nnz_offd; i++)
   {
      i1 = P_offd_j[i];
      k1 = nalu_hypre_BinarySearch(tmp_map_offd, i1, num_cols_offd_P);
      /* search must succeed */
      nalu_hypre_assert(k1 >= 0 && k1 < num_cols_offd_P);
      P_offd_j[i] = k1;
   }

   /* change col_map_offd_P to global coarse ids */
   for (i = 0; i < num_cols_offd_P; i++)
   {
      col_map_offd_P[i] = fine_to_coarse_offd[tmp_map_offd[i]];
   }

   /* Now, we should have everything of Parcsr matrix P */
   P = nalu_hypre_ParCSRMatrixCreate(comm,
                                nalu_hypre_ParCSRMatrixGlobalNumCols(A), /* global num of rows */
                                total_global_cpts, /* global num of cols */
                                nalu_hypre_ParCSRMatrixColStarts(A), /* row_starts */
                                num_cpts_global, /* col_starts */
                                num_cols_offd_P, /* num cols offd */
                                nnz_diag,
                                nnz_offd);

   P_diag = nalu_hypre_ParCSRMatrixDiag(P);
   nalu_hypre_CSRMatrixData(P_diag) = P_diag_data;
   nalu_hypre_CSRMatrixI(P_diag)    = P_diag_i;
   nalu_hypre_CSRMatrixJ(P_diag)    = P_diag_j;

   P_offd = nalu_hypre_ParCSRMatrixOffd(P);
   nalu_hypre_CSRMatrixData(P_offd) = P_offd_data;
   nalu_hypre_CSRMatrixI(P_offd)    = P_offd_i;
   nalu_hypre_CSRMatrixJ(P_offd)    = P_offd_j;

   nalu_hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;

   /* create CommPkg of P */
   nalu_hypre_MatvecCommPkgCreate(P);

   *P_ptr = P;

   /* free workspace */
   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dof_func_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(tmp_map_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_int_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(marker_diag, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(max_abs_cij, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(max_abs_diag_offd, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildInterpOnePnt( nalu_hypre_ParCSRMatrix  *A,
                                  NALU_HYPRE_Int           *CF_marker,
                                  nalu_hypre_ParCSRMatrix  *S,
                                  NALU_HYPRE_BigInt        *num_cpts_global,
                                  NALU_HYPRE_Int            num_functions,
                                  NALU_HYPRE_Int           *dof_func,
                                  NALU_HYPRE_Int            debug_flag,
                                  nalu_hypre_ParCSRMatrix **P_ptr)
{
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_GpuProfilingPushRange("OnePntInterp");
#endif

   NALU_HYPRE_Int ierr = 0;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      ierr = nalu_hypre_BoomerAMGBuildInterpOnePntDevice(A, CF_marker, S, num_cpts_global, num_functions,
                                                    dof_func, debug_flag, P_ptr);
   }
   else
#endif
   {
      ierr = nalu_hypre_BoomerAMGBuildInterpOnePntHost(A, CF_marker, S, num_cpts_global, num_functions,
                                                  dof_func, debug_flag, P_ptr);
   }

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_GpuProfilingPopRange();
#endif

   return ierr;
}
