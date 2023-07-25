/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_utilities.h"
#include "_nalu_hypre_parcsr_mv.h"
#include "_nalu_hypre_lapack.h"
#include "_nalu_hypre_blas.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGenerateFFFCHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGenerateFFFCHost( nalu_hypre_ParCSRMatrix  *A,
                                    NALU_HYPRE_Int           *CF_marker,
                                    NALU_HYPRE_BigInt        *cpts_starts,
                                    nalu_hypre_ParCSRMatrix  *S,
                                    nalu_hypre_ParCSRMatrix **A_FC_ptr,
                                    nalu_hypre_ParCSRMatrix **A_FF_ptr)
{
   MPI_Comm                 comm     = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_MemoryLocation memory_location_P = nalu_hypre_ParCSRMatrixMemoryLocation(A);
   if (!nalu_hypre_ParCSRMatrixCommPkg(A))
   {
      nalu_hypre_MatvecCommPkgCreate(A);
   }
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   /* diag part of A */
   nalu_hypre_CSRMatrix    *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int          *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int          *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   nalu_hypre_CSRMatrix    *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int          *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int          *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);

   NALU_HYPRE_Int           n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int           num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   /* diag part of S */
   nalu_hypre_CSRMatrix    *S_diag   = S ? nalu_hypre_ParCSRMatrixDiag(S) : A_diag;
   NALU_HYPRE_Int          *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int          *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);
   NALU_HYPRE_Int           skip_diag = S ? 0 : 1;
   /* off-diag part of S */
   nalu_hypre_CSRMatrix    *S_offd   = S ? nalu_hypre_ParCSRMatrixOffd(S) : A_offd;
   NALU_HYPRE_Int          *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int          *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);

   nalu_hypre_ParCSRMatrix *A_FC;
   nalu_hypre_CSRMatrix    *A_FC_diag, *A_FC_offd;
   NALU_HYPRE_Int          *A_FC_diag_i, *A_FC_diag_j, *A_FC_offd_i, *A_FC_offd_j = NULL;
   NALU_HYPRE_Complex      *A_FC_diag_data, *A_FC_offd_data = NULL;
   NALU_HYPRE_Int           num_cols_offd_A_FC;
   NALU_HYPRE_BigInt       *col_map_offd_A_FC = NULL;

   nalu_hypre_ParCSRMatrix *A_FF;
   nalu_hypre_CSRMatrix    *A_FF_diag, *A_FF_offd;
   NALU_HYPRE_Int          *A_FF_diag_i, *A_FF_diag_j, *A_FF_offd_i, *A_FF_offd_j;
   NALU_HYPRE_Complex      *A_FF_diag_data, *A_FF_offd_data;
   NALU_HYPRE_Int           num_cols_offd_A_FF;
   NALU_HYPRE_BigInt       *col_map_offd_A_FF = NULL;

   NALU_HYPRE_Int          *fine_to_coarse;
   NALU_HYPRE_Int          *fine_to_fine;
   NALU_HYPRE_Int          *fine_to_coarse_offd = NULL;
   NALU_HYPRE_Int          *fine_to_fine_offd = NULL;

   NALU_HYPRE_Int           i, j, jj;
   NALU_HYPRE_Int           startc, index;
   NALU_HYPRE_Int           cpt, fpt, row;
   NALU_HYPRE_Int          *CF_marker_offd = NULL, *marker_offd = NULL;
   NALU_HYPRE_Int          *int_buf_data = NULL;
   NALU_HYPRE_BigInt       *big_convert;
   NALU_HYPRE_BigInt       *big_convert_offd = NULL;
   NALU_HYPRE_BigInt       *big_buf_data = NULL;

   NALU_HYPRE_BigInt        total_global_fpts, total_global_cpts, fpts_starts[2];
   NALU_HYPRE_Int           my_id, num_procs, num_sends;
   NALU_HYPRE_Int           d_count_FF, d_count_FC, o_count_FF, o_count_FC;
   NALU_HYPRE_Int           n_Fpts;
   NALU_HYPRE_Int          *cpt_array, *fpt_array;
   NALU_HYPRE_Int           start, stop;
   NALU_HYPRE_Int           num_threads;

   num_threads = nalu_hypre_NumThreads();

   /* MPI size and rank*/
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   fine_to_coarse = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   fine_to_fine = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   big_convert = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, n_fine, NALU_HYPRE_MEMORY_HOST);

   cpt_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads + 1, NALU_HYPRE_MEMORY_HOST);
   fpt_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads + 1, NALU_HYPRE_MEMORY_HOST);
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel private(i,j,jj,start,stop,row,cpt,fpt,d_count_FC,d_count_FF,o_count_FC,o_count_FF)
#endif
   {
      NALU_HYPRE_Int my_thread_num = nalu_hypre_GetThreadNum();

      start = (n_fine / num_threads) * my_thread_num;
      if (my_thread_num == num_threads - 1)
      {
         stop = n_fine;
      }
      else
      {
         stop = (n_fine / num_threads) * (my_thread_num + 1);
      }
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] > 0)
         {
            cpt_array[my_thread_num + 1]++;
         }
         else
         {
            fpt_array[my_thread_num + 1]++;
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         for (i = 1; i < num_threads; i++)
         {
            cpt_array[i + 1] += cpt_array[i];
            fpt_array[i + 1] += fpt_array[i];
         }
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      cpt = cpt_array[my_thread_num];
      fpt = fpt_array[my_thread_num];
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] > 0)
         {
            fine_to_coarse[i] = cpt++;
            fine_to_fine[i] = -1;
         }
         else
         {
            fine_to_fine[i] = fpt++;
            fine_to_coarse[i] = -1;
         }
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      if (my_thread_num == 0)
      {
         NALU_HYPRE_BigInt big_Fpts;
         n_Fpts = fpt_array[num_threads];
         big_Fpts = n_Fpts;

         nalu_hypre_MPI_Scan(&big_Fpts, fpts_starts + 1, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
         fpts_starts[0] = fpts_starts[1] - big_Fpts;
         if (my_id == num_procs - 1)
         {
            total_global_fpts = fpts_starts[1];
            total_global_cpts = cpts_starts[1];
         }
         nalu_hypre_MPI_Bcast(&total_global_fpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
         nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] > 0)
         {
            big_convert[i] = (NALU_HYPRE_BigInt)fine_to_coarse[i] + cpts_starts[0];
         }
         else
         {
            big_convert[i] = (NALU_HYPRE_BigInt)fine_to_fine[i] + fpts_starts[0];
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         if (num_cols_A_offd)
         {
            CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
            big_convert_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
            fine_to_coarse_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
            fine_to_fine_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
         }
         index = 0;
         num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
         int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                      NALU_HYPRE_MEMORY_HOST);
         big_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                      NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sends; i++)
         {
            startc = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = startc; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               int_buf_data[index] = CF_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
               big_buf_data[index++] = big_convert[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
         }

         comm_handle = nalu_hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, CF_marker_offd);

         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

         comm_handle = nalu_hypre_ParCSRCommHandleCreate( 21, comm_pkg, big_buf_data, big_convert_offd);

         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

         marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < n_fine; i++)
         {
            if (CF_marker[i] < 0)
            {
               for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
               {
                  marker_offd[S_offd_j[j]] = 1;
               }
            }
         }

         num_cols_offd_A_FC = 0;
         num_cols_offd_A_FF = 0;
         if (num_cols_A_offd)
         {
            for (i = 0; i < num_cols_A_offd; i++)
            {
               if (CF_marker_offd[i] > 0 && marker_offd[i] > 0)
               {
                  fine_to_coarse_offd[i] = num_cols_offd_A_FC++;
                  fine_to_fine_offd[i] = -1;
               }
               else if (CF_marker_offd[i] < 0 && marker_offd[i] > 0)
               {
                  fine_to_fine_offd[i] = num_cols_offd_A_FF++;
                  fine_to_coarse_offd[i] = -1;
               }
            }

            col_map_offd_A_FF = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_A_FF, NALU_HYPRE_MEMORY_HOST);
            col_map_offd_A_FC = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_A_FC, NALU_HYPRE_MEMORY_HOST);

            cpt = 0;
            fpt = 0;
            for (i = 0; i < num_cols_A_offd; i++)
            {
               if (CF_marker_offd[i] > 0 && marker_offd[i] > 0)
               {
                  col_map_offd_A_FC[cpt++] = big_convert_offd[i];
               }
               else if (CF_marker_offd[i] < 0 && marker_offd[i] > 0)
               {
                  col_map_offd_A_FF[fpt++] = big_convert_offd[i];
               }
            }
         }

         A_FF_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_Fpts + 1, memory_location_P);
         A_FC_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_Fpts + 1, memory_location_P);
         A_FF_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_Fpts + 1, memory_location_P);
         A_FC_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_Fpts + 1, memory_location_P);
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      d_count_FC = 0;
      d_count_FF = 0;
      o_count_FC = 0;
      o_count_FF = 0;
      row = fpt_array[my_thread_num];
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] < 0)
         {
            row++;
            d_count_FF++; /* account for diagonal element */
            for (j = S_diag_i[i] + skip_diag; j < S_diag_i[i + 1]; j++)
            {
               jj = S_diag_j[j];
               if (CF_marker[jj] > 0)
               {
                  d_count_FC++;
               }
               else
               {
                  d_count_FF++;
               }
            }
            A_FF_diag_i[row] = d_count_FF;
            A_FC_diag_i[row] = d_count_FC;
            for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
            {
               jj = S_offd_j[j];
               if (CF_marker_offd[jj] > 0)
               {
                  o_count_FC++;
               }
               else
               {
                  o_count_FF++;
               }
            }
            A_FF_offd_i[row] = o_count_FF;
            A_FC_offd_i[row] = o_count_FC;
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         NALU_HYPRE_Int fpt2;
         for (i = 1; i < num_threads + 1; i++)
         {
            fpt = fpt_array[i];
            fpt2 = fpt_array[i - 1];

            if (fpt == fpt2)
            {
               continue;
            }

            A_FC_diag_i[fpt] += A_FC_diag_i[fpt2];
            A_FF_diag_i[fpt] += A_FF_diag_i[fpt2];
            A_FC_offd_i[fpt] += A_FC_offd_i[fpt2];
            A_FF_offd_i[fpt] += A_FF_offd_i[fpt2];
         }
         row = fpt_array[num_threads];
         d_count_FC = A_FC_diag_i[row];
         d_count_FF = A_FF_diag_i[row];
         o_count_FC = A_FC_offd_i[row];
         o_count_FF = A_FF_offd_i[row];
         A_FF_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, d_count_FF, memory_location_P);
         A_FC_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, d_count_FC, memory_location_P);
         A_FF_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, o_count_FF, memory_location_P);
         A_FC_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, o_count_FC, memory_location_P);
         A_FF_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, d_count_FF, memory_location_P);
         A_FC_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, d_count_FC, memory_location_P);
         A_FF_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, o_count_FF, memory_location_P);
         A_FC_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, o_count_FC, memory_location_P);
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      row = fpt_array[my_thread_num];
      d_count_FC = A_FC_diag_i[row];
      d_count_FF = A_FF_diag_i[row];
      o_count_FC = A_FC_offd_i[row];
      o_count_FF = A_FF_offd_i[row];
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] < 0)
         {
            NALU_HYPRE_Int jS, jA;
            row++;
            jA = A_diag_i[i];
            A_FF_diag_j[d_count_FF] = fine_to_fine[A_diag_j[jA]];
            A_FF_diag_data[d_count_FF++] = A_diag_data[jA++];
            for (j = S_diag_i[i] + skip_diag; j < S_diag_i[i + 1]; j++)
            {
               jA = A_diag_i[i] + 1;
               jS = S_diag_j[j];
               while (A_diag_j[jA] != jS) { jA++; }
               if (CF_marker[S_diag_j[j]] > 0)
               {
                  A_FC_diag_j[d_count_FC] = fine_to_coarse[A_diag_j[jA]];
                  A_FC_diag_data[d_count_FC++] = A_diag_data[jA++];
               }
               else
               {
                  A_FF_diag_j[d_count_FF] = fine_to_fine[A_diag_j[jA]];
                  A_FF_diag_data[d_count_FF++] = A_diag_data[jA++];
               }
            }
            A_FF_diag_i[row] = d_count_FF;
            A_FC_diag_i[row] = d_count_FC;
            for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
            {
               jA = A_offd_i[i];
               jS = S_offd_j[j];
               while (jS != A_offd_j[jA]) { jA++; }
               if (CF_marker_offd[S_offd_j[j]] > 0)
               {
                  A_FC_offd_j[o_count_FC] = fine_to_coarse_offd[A_offd_j[jA]];
                  A_FC_offd_data[o_count_FC++] = A_offd_data[jA++];
               }
               else
               {
                  A_FF_offd_j[o_count_FF] = fine_to_fine_offd[A_offd_j[jA]];
                  A_FF_offd_data[o_count_FF++] = A_offd_data[jA++];
               }
            }
            A_FF_offd_i[row] = o_count_FF;
            A_FC_offd_i[row] = o_count_FC;
         }
      }
   } /*end parallel region */

   A_FC = nalu_hypre_ParCSRMatrixCreate(comm,
                                   total_global_fpts,
                                   total_global_cpts,
                                   fpts_starts,
                                   cpts_starts,
                                   num_cols_offd_A_FC,
                                   A_FC_diag_i[n_Fpts],
                                   A_FC_offd_i[n_Fpts]);

   A_FF = nalu_hypre_ParCSRMatrixCreate(comm,
                                   total_global_fpts,
                                   total_global_fpts,
                                   fpts_starts,
                                   fpts_starts,
                                   num_cols_offd_A_FF,
                                   A_FF_diag_i[n_Fpts],
                                   A_FF_offd_i[n_Fpts]);

   A_FC_diag = nalu_hypre_ParCSRMatrixDiag(A_FC);
   nalu_hypre_CSRMatrixData(A_FC_diag) = A_FC_diag_data;
   nalu_hypre_CSRMatrixI(A_FC_diag) = A_FC_diag_i;
   nalu_hypre_CSRMatrixJ(A_FC_diag) = A_FC_diag_j;
   A_FC_offd = nalu_hypre_ParCSRMatrixOffd(A_FC);
   nalu_hypre_CSRMatrixData(A_FC_offd) = A_FC_offd_data;
   nalu_hypre_CSRMatrixI(A_FC_offd) = A_FC_offd_i;
   nalu_hypre_CSRMatrixJ(A_FC_offd) = A_FC_offd_j;
   nalu_hypre_ParCSRMatrixColMapOffd(A_FC) = col_map_offd_A_FC;

   nalu_hypre_CSRMatrixMemoryLocation(A_FC_diag) = memory_location_P;
   nalu_hypre_CSRMatrixMemoryLocation(A_FC_offd) = memory_location_P;

   A_FF_diag = nalu_hypre_ParCSRMatrixDiag(A_FF);
   nalu_hypre_CSRMatrixData(A_FF_diag) = A_FF_diag_data;
   nalu_hypre_CSRMatrixI(A_FF_diag) = A_FF_diag_i;
   nalu_hypre_CSRMatrixJ(A_FF_diag) = A_FF_diag_j;
   A_FF_offd = nalu_hypre_ParCSRMatrixOffd(A_FF);
   nalu_hypre_CSRMatrixData(A_FF_offd) = A_FF_offd_data;
   nalu_hypre_CSRMatrixI(A_FF_offd) = A_FF_offd_i;
   nalu_hypre_CSRMatrixJ(A_FF_offd) = A_FF_offd_j;
   nalu_hypre_ParCSRMatrixColMapOffd(A_FF) = col_map_offd_A_FF;

   nalu_hypre_CSRMatrixMemoryLocation(A_FF_diag) = memory_location_P;
   nalu_hypre_CSRMatrixMemoryLocation(A_FF_offd) = memory_location_P;

   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_fine, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_convert, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_fine_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_convert_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(cpt_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fpt_array, NALU_HYPRE_MEMORY_HOST);

   *A_FC_ptr = A_FC;
   *A_FF_ptr = A_FF;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGenerateFFFC
 *
 * Generate AFF or AFC
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGenerateFFFC( nalu_hypre_ParCSRMatrix  *A,
                                NALU_HYPRE_Int           *CF_marker,
                                NALU_HYPRE_BigInt        *cpts_starts,
                                nalu_hypre_ParCSRMatrix  *S,
                                nalu_hypre_ParCSRMatrix **A_FC_ptr,
                                nalu_hypre_ParCSRMatrix **A_FF_ptr)
{
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_ParCSRMatrixGenerateFFFCDevice(A, CF_marker, cpts_starts, S, A_FC_ptr, A_FF_ptr);
   }
   else
#endif
   {
      nalu_hypre_ParCSRMatrixGenerateFFFCHost(A, CF_marker, cpts_starts, S, A_FC_ptr, A_FF_ptr);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGenerateFFFC3
 *
 * generate AFF, AFC, for 2 stage extended interpolation
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGenerateFFFC3( nalu_hypre_ParCSRMatrix  *A,
                                 NALU_HYPRE_Int           *CF_marker,
                                 NALU_HYPRE_BigInt        *cpts_starts,
                                 nalu_hypre_ParCSRMatrix  *S,
                                 nalu_hypre_ParCSRMatrix **A_FC_ptr,
                                 nalu_hypre_ParCSRMatrix **A_FF_ptr)
{
   MPI_Comm                 comm     = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_MemoryLocation memory_location_P = nalu_hypre_ParCSRMatrixMemoryLocation(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   /* diag part of A */
   nalu_hypre_CSRMatrix    *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int          *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int          *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);

   /* off-diag part of A */
   nalu_hypre_CSRMatrix    *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int          *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int          *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);

   NALU_HYPRE_Int           n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int           num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   /* diag part of S */
   nalu_hypre_CSRMatrix    *S_diag   = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int          *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int          *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);

   /* off-diag part of S */
   nalu_hypre_CSRMatrix    *S_offd   = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int          *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int          *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);

   nalu_hypre_ParCSRMatrix *A_FC;
   nalu_hypre_CSRMatrix    *A_FC_diag, *A_FC_offd;
   NALU_HYPRE_Int          *A_FC_diag_i, *A_FC_diag_j, *A_FC_offd_i, *A_FC_offd_j = NULL;
   NALU_HYPRE_Complex      *A_FC_diag_data, *A_FC_offd_data = NULL;
   NALU_HYPRE_Int           num_cols_offd_A_FC;
   NALU_HYPRE_BigInt       *col_map_offd_A_FC = NULL;

   nalu_hypre_ParCSRMatrix *A_FF;
   nalu_hypre_CSRMatrix    *A_FF_diag, *A_FF_offd;
   NALU_HYPRE_Int          *A_FF_diag_i, *A_FF_diag_j, *A_FF_offd_i, *A_FF_offd_j;
   NALU_HYPRE_Complex      *A_FF_diag_data, *A_FF_offd_data;
   NALU_HYPRE_Int           num_cols_offd_A_FF;
   NALU_HYPRE_BigInt       *col_map_offd_A_FF = NULL;

   NALU_HYPRE_Int          *fine_to_coarse;
   NALU_HYPRE_Int          *fine_to_fine;
   NALU_HYPRE_Int          *fine_to_coarse_offd = NULL;
   NALU_HYPRE_Int          *fine_to_fine_offd = NULL;

   NALU_HYPRE_Int           i, j, jj;
   NALU_HYPRE_Int           startc, index;
   NALU_HYPRE_Int           cpt, fpt, new_fpt, row, rowc;
   NALU_HYPRE_Int          *CF_marker_offd = NULL;
   NALU_HYPRE_Int          *int_buf_data = NULL;
   NALU_HYPRE_BigInt       *big_convert;
   NALU_HYPRE_BigInt       *big_convert_offd = NULL;
   NALU_HYPRE_BigInt       *big_buf_data = NULL;

   NALU_HYPRE_BigInt        total_global_fpts, total_global_cpts, total_global_new_fpts;
   NALU_HYPRE_BigInt        fpts_starts[2], new_fpts_starts[2];
   NALU_HYPRE_Int           my_id, num_procs, num_sends;
   NALU_HYPRE_Int           d_count_FF, d_count_FC, o_count_FF, o_count_FC;
   NALU_HYPRE_Int           n_Fpts;
   NALU_HYPRE_Int           n_new_Fpts;
   NALU_HYPRE_Int          *cpt_array, *fpt_array, *new_fpt_array;
   NALU_HYPRE_Int           start, stop;
   NALU_HYPRE_Int           num_threads;

   num_threads = nalu_hypre_NumThreads();

   /* MPI size and rank*/
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   fine_to_coarse = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   fine_to_fine = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   big_convert = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, n_fine, NALU_HYPRE_MEMORY_HOST);

   cpt_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads + 1, NALU_HYPRE_MEMORY_HOST);
   fpt_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads + 1, NALU_HYPRE_MEMORY_HOST);
   new_fpt_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads + 1, NALU_HYPRE_MEMORY_HOST);
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel private(i,j,jj,start,stop,row,rowc,cpt,new_fpt,fpt,d_count_FC,d_count_FF,o_count_FC,o_count_FF)
#endif
   {
      NALU_HYPRE_Int my_thread_num = nalu_hypre_GetThreadNum();

      start = (n_fine / num_threads) * my_thread_num;
      if (my_thread_num == num_threads - 1)
      {
         stop = n_fine;
      }
      else
      {
         stop = (n_fine / num_threads) * (my_thread_num + 1);
      }
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] > 0)
         {
            cpt_array[my_thread_num + 1]++;
         }
         else if (CF_marker[i] == -2)
         {
            new_fpt_array[my_thread_num + 1]++;
            fpt_array[my_thread_num + 1]++;
         }
         else
         {
            fpt_array[my_thread_num + 1]++;
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         for (i = 1; i < num_threads; i++)
         {
            cpt_array[i + 1] += cpt_array[i];
            fpt_array[i + 1] += fpt_array[i];
            new_fpt_array[i + 1] += new_fpt_array[i];
         }
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      cpt = cpt_array[my_thread_num];
      fpt = fpt_array[my_thread_num];
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] > 0)
         {
            fine_to_coarse[i] = cpt++;
            fine_to_fine[i] = -1;
         }
         else
         {
            fine_to_fine[i] = fpt++;
            fine_to_coarse[i] = -1;
         }
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      if (my_thread_num == 0)
      {
         NALU_HYPRE_BigInt big_Fpts, big_new_Fpts;
         n_Fpts = fpt_array[num_threads];
         n_new_Fpts = new_fpt_array[num_threads];
         big_Fpts = n_Fpts;
         big_new_Fpts = n_new_Fpts;

         nalu_hypre_MPI_Scan(&big_Fpts, fpts_starts + 1, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
         nalu_hypre_MPI_Scan(&big_new_Fpts, new_fpts_starts + 1, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
         fpts_starts[0] = fpts_starts[1] - big_Fpts;
         new_fpts_starts[0] = new_fpts_starts[1] - big_new_Fpts;
         if (my_id == num_procs - 1)
         {
            total_global_new_fpts = new_fpts_starts[1];
            total_global_fpts = fpts_starts[1];
            total_global_cpts = cpts_starts[1];
         }
         nalu_hypre_MPI_Bcast(&total_global_new_fpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
         nalu_hypre_MPI_Bcast(&total_global_fpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
         nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] > 0)
         {
            big_convert[i] = (NALU_HYPRE_BigInt)fine_to_coarse[i] + cpts_starts[0];
         }
         else
         {
            big_convert[i] = (NALU_HYPRE_BigInt)fine_to_fine[i] + fpts_starts[0];
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         if (num_cols_A_offd)
         {
            CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
            big_convert_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
            fine_to_coarse_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
            fine_to_fine_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
         }
         index = 0;
         num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
         int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int,
                                      nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                      NALU_HYPRE_MEMORY_HOST);
         big_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,
                                      nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                      NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sends; i++)
         {
            startc = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = startc; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               int_buf_data[index] = CF_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
               big_buf_data[index++] = big_convert[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
         }

         comm_handle = nalu_hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data, CF_marker_offd);

         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

         comm_handle = nalu_hypre_ParCSRCommHandleCreate(21, comm_pkg, big_buf_data, big_convert_offd);

         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

         num_cols_offd_A_FC = 0;
         num_cols_offd_A_FF = 0;
         if (num_cols_A_offd)
         {
            for (i = 0; i < num_cols_A_offd; i++)
            {
               if (CF_marker_offd[i] > 0)
               {
                  fine_to_coarse_offd[i] = num_cols_offd_A_FC++;
                  fine_to_fine_offd[i] = -1;
               }
               else
               {
                  fine_to_fine_offd[i] = num_cols_offd_A_FF++;
                  fine_to_coarse_offd[i] = -1;
               }
            }

            col_map_offd_A_FF = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_A_FF, NALU_HYPRE_MEMORY_HOST);
            col_map_offd_A_FC = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_A_FC, NALU_HYPRE_MEMORY_HOST);

            cpt = 0;
            fpt = 0;
            for (i = 0; i < num_cols_A_offd; i++)
            {
               if (CF_marker_offd[i] > 0)
               {
                  col_map_offd_A_FC[cpt++] = big_convert_offd[i];
               }
               else
               {
                  col_map_offd_A_FF[fpt++] = big_convert_offd[i];
               }
            }
         }

         A_FF_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_new_Fpts + 1, memory_location_P);
         A_FC_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_Fpts + 1, memory_location_P);
         A_FF_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_new_Fpts + 1, memory_location_P);
         A_FC_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_Fpts + 1, memory_location_P);
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      d_count_FC = 0;
      d_count_FF = 0;
      o_count_FC = 0;
      o_count_FF = 0;
      row = new_fpt_array[my_thread_num];
      rowc = fpt_array[my_thread_num];
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] == -2)
         {
            row++;
            rowc++;
            d_count_FF++; /* account for diagonal element */
            for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
            {
               jj = S_diag_j[j];
               if (CF_marker[jj] > 0)
               {
                  d_count_FC++;
               }
               else
               {
                  d_count_FF++;
               }
            }
            A_FF_diag_i[row] = d_count_FF;
            A_FC_diag_i[rowc] = d_count_FC;
            for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
            {
               jj = S_offd_j[j];
               if (CF_marker_offd[jj] > 0)
               {
                  o_count_FC++;
               }
               else
               {
                  o_count_FF++;
               }
            }
            A_FF_offd_i[row] = o_count_FF;
            A_FC_offd_i[rowc] = o_count_FC;
         }
         else if (CF_marker[i] < 0)
         {
            rowc++;
            for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
            {
               jj = S_diag_j[j];
               if (CF_marker[jj] > 0)
               {
                  d_count_FC++;
               }
            }
            A_FC_diag_i[rowc] = d_count_FC;
            for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
            {
               jj = S_offd_j[j];
               if (CF_marker_offd[jj] > 0)
               {
                  o_count_FC++;
               }
            }
            A_FC_offd_i[rowc] = o_count_FC;
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         NALU_HYPRE_Int fpt2, new_fpt2;
         for (i = 1; i < num_threads + 1; i++)
         {
            fpt = fpt_array[i];
            new_fpt = new_fpt_array[i];
            fpt2 = fpt_array[i - 1];
            new_fpt2 = new_fpt_array[i - 1];
            if (new_fpt != new_fpt2)
            {
               A_FF_diag_i[new_fpt] += A_FF_diag_i[new_fpt2];
               A_FF_offd_i[new_fpt] += A_FF_offd_i[new_fpt2];
            }
            if (fpt != fpt2)
            {
               A_FC_diag_i[fpt] += A_FC_diag_i[fpt2];
               A_FC_offd_i[fpt] += A_FC_offd_i[fpt2];
            }
         }
         row = new_fpt_array[num_threads];
         rowc = fpt_array[num_threads];
         d_count_FC = A_FC_diag_i[rowc];
         d_count_FF = A_FF_diag_i[row];
         o_count_FC = A_FC_offd_i[rowc];
         o_count_FF = A_FF_offd_i[row];
         A_FF_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, d_count_FF, memory_location_P);
         A_FC_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, d_count_FC, memory_location_P);
         A_FF_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, o_count_FF, memory_location_P);
         A_FC_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, o_count_FC, memory_location_P);
         A_FF_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, d_count_FF, memory_location_P);
         A_FC_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, d_count_FC, memory_location_P);
         A_FF_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, o_count_FF, memory_location_P);
         A_FC_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, o_count_FC, memory_location_P);
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      row = new_fpt_array[my_thread_num];
      rowc = fpt_array[my_thread_num];
      d_count_FC = A_FC_diag_i[rowc];
      d_count_FF = A_FF_diag_i[row];
      o_count_FC = A_FC_offd_i[rowc];
      o_count_FF = A_FF_offd_i[row];
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] == -2)
         {
            NALU_HYPRE_Int jS, jA;
            row++;
            rowc++;
            jA = A_diag_i[i];
            A_FF_diag_j[d_count_FF] = fine_to_fine[A_diag_j[jA]];
            A_FF_diag_data[d_count_FF++] = A_diag_data[jA++];
            for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
            {
               jA = A_diag_i[i] + 1;
               jS = S_diag_j[j];
               while (A_diag_j[jA] != jS) { jA++; }
               if (CF_marker[S_diag_j[j]] > 0)
               {
                  A_FC_diag_j[d_count_FC] = fine_to_coarse[A_diag_j[jA]];
                  A_FC_diag_data[d_count_FC++] = A_diag_data[jA++];
               }
               else
               {
                  A_FF_diag_j[d_count_FF] = fine_to_fine[A_diag_j[jA]];
                  A_FF_diag_data[d_count_FF++] = A_diag_data[jA++];
               }
            }
            A_FF_diag_i[row] = d_count_FF;
            A_FC_diag_i[rowc] = d_count_FC;
            for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
            {
               jA = A_offd_i[i];
               jS = S_offd_j[j];
               while (jS != A_offd_j[jA]) { jA++; }
               if (CF_marker_offd[S_offd_j[j]] > 0)
               {
                  A_FC_offd_j[o_count_FC] = fine_to_coarse_offd[A_offd_j[jA]];
                  A_FC_offd_data[o_count_FC++] = A_offd_data[jA++];
               }
               else
               {
                  A_FF_offd_j[o_count_FF] = fine_to_fine_offd[A_offd_j[jA]];
                  A_FF_offd_data[o_count_FF++] = A_offd_data[jA++];
               }
            }
            A_FF_offd_i[row] = o_count_FF;
            A_FC_offd_i[rowc] = o_count_FC;
         }
         else if (CF_marker[i] < 0)
         {
            NALU_HYPRE_Int jS, jA;
            rowc++;
            for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
            {
               jA = A_diag_i[i] + 1;
               jS = S_diag_j[j];
               while (A_diag_j[jA] != jS) { jA++; }
               if (CF_marker[S_diag_j[j]] > 0)
               {
                  A_FC_diag_j[d_count_FC] = fine_to_coarse[A_diag_j[jA]];
                  A_FC_diag_data[d_count_FC++] = A_diag_data[jA++];
               }
            }
            A_FC_diag_i[rowc] = d_count_FC;
            for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
            {
               jA = A_offd_i[i];
               jS = S_offd_j[j];
               while (jS != A_offd_j[jA]) { jA++; }
               if (CF_marker_offd[S_offd_j[j]] > 0)
               {
                  A_FC_offd_j[o_count_FC] = fine_to_coarse_offd[A_offd_j[jA]];
                  A_FC_offd_data[o_count_FC++] = A_offd_data[jA++];
               }
            }
            A_FC_offd_i[rowc] = o_count_FC;
         }
      }
   } /*end parallel region */

   A_FC = nalu_hypre_ParCSRMatrixCreate(comm,
                                   total_global_fpts,
                                   total_global_cpts,
                                   fpts_starts,
                                   cpts_starts,
                                   num_cols_offd_A_FC,
                                   A_FC_diag_i[n_Fpts],
                                   A_FC_offd_i[n_Fpts]);

   A_FF = nalu_hypre_ParCSRMatrixCreate(comm,
                                   total_global_new_fpts,
                                   total_global_fpts,
                                   new_fpts_starts,
                                   fpts_starts,
                                   num_cols_offd_A_FF,
                                   A_FF_diag_i[n_new_Fpts],
                                   A_FF_offd_i[n_new_Fpts]);

   A_FC_diag = nalu_hypre_ParCSRMatrixDiag(A_FC);
   nalu_hypre_CSRMatrixData(A_FC_diag) = A_FC_diag_data;
   nalu_hypre_CSRMatrixI(A_FC_diag) = A_FC_diag_i;
   nalu_hypre_CSRMatrixJ(A_FC_diag) = A_FC_diag_j;
   A_FC_offd = nalu_hypre_ParCSRMatrixOffd(A_FC);
   nalu_hypre_CSRMatrixData(A_FC_offd) = A_FC_offd_data;
   nalu_hypre_CSRMatrixI(A_FC_offd) = A_FC_offd_i;
   nalu_hypre_CSRMatrixJ(A_FC_offd) = A_FC_offd_j;
   nalu_hypre_ParCSRMatrixColMapOffd(A_FC) = col_map_offd_A_FC;

   nalu_hypre_CSRMatrixMemoryLocation(A_FC_diag) = memory_location_P;
   nalu_hypre_CSRMatrixMemoryLocation(A_FC_offd) = memory_location_P;

   A_FF_diag = nalu_hypre_ParCSRMatrixDiag(A_FF);
   nalu_hypre_CSRMatrixData(A_FF_diag) = A_FF_diag_data;
   nalu_hypre_CSRMatrixI(A_FF_diag) = A_FF_diag_i;
   nalu_hypre_CSRMatrixJ(A_FF_diag) = A_FF_diag_j;
   A_FF_offd = nalu_hypre_ParCSRMatrixOffd(A_FF);
   nalu_hypre_CSRMatrixData(A_FF_offd) = A_FF_offd_data;
   nalu_hypre_CSRMatrixI(A_FF_offd) = A_FF_offd_i;
   nalu_hypre_CSRMatrixJ(A_FF_offd) = A_FF_offd_j;
   nalu_hypre_ParCSRMatrixColMapOffd(A_FF) = col_map_offd_A_FF;

   nalu_hypre_CSRMatrixMemoryLocation(A_FF_diag) = memory_location_P;
   nalu_hypre_CSRMatrixMemoryLocation(A_FF_offd) = memory_location_P;

   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_fine, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_convert, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_fine_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_convert_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_buf_data, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(cpt_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fpt_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(new_fpt_array, NALU_HYPRE_MEMORY_HOST);

   *A_FC_ptr = A_FC;
   *A_FF_ptr = A_FF;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGenerateFFFCD3
 *
 * Generate AFF, AFC, AFFC for 2 stage extended+i(e)interpolation
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGenerateFFFCD3( nalu_hypre_ParCSRMatrix *A,
                                  NALU_HYPRE_Int           *CF_marker,
                                  NALU_HYPRE_BigInt        *cpts_starts,
                                  nalu_hypre_ParCSRMatrix  *S,
                                  nalu_hypre_ParCSRMatrix **A_FC_ptr,
                                  nalu_hypre_ParCSRMatrix **A_FF_ptr,
                                  NALU_HYPRE_Real         **D_lambda_ptr)
{
   MPI_Comm                 comm     = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_MemoryLocation memory_location_P = nalu_hypre_ParCSRMatrixMemoryLocation(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   /* diag part of A */
   nalu_hypre_CSRMatrix    *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int          *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int          *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   nalu_hypre_CSRMatrix    *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int          *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int          *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);

   NALU_HYPRE_Int           n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int           num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   /* diag part of S */
   nalu_hypre_CSRMatrix    *S_diag   = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int          *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int          *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);
   /* off-diag part of S */
   nalu_hypre_CSRMatrix    *S_offd   = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int          *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int          *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);

   NALU_HYPRE_Real         *D_lambda;
   nalu_hypre_ParCSRMatrix *A_FC;
   nalu_hypre_CSRMatrix    *A_FC_diag, *A_FC_offd;
   NALU_HYPRE_Int          *A_FC_diag_i, *A_FC_diag_j, *A_FC_offd_i, *A_FC_offd_j = NULL;
   NALU_HYPRE_Complex      *A_FC_diag_data, *A_FC_offd_data = NULL;
   NALU_HYPRE_Int           num_cols_offd_A_FC;
   NALU_HYPRE_BigInt       *col_map_offd_A_FC = NULL;

   nalu_hypre_ParCSRMatrix *A_FF;
   nalu_hypre_CSRMatrix    *A_FF_diag, *A_FF_offd;
   NALU_HYPRE_Int          *A_FF_diag_i, *A_FF_diag_j, *A_FF_offd_i, *A_FF_offd_j;
   NALU_HYPRE_Complex      *A_FF_diag_data, *A_FF_offd_data;
   NALU_HYPRE_Int           num_cols_offd_A_FF;
   NALU_HYPRE_BigInt       *col_map_offd_A_FF = NULL;

   NALU_HYPRE_Int          *fine_to_coarse;
   NALU_HYPRE_Int          *fine_to_fine;
   NALU_HYPRE_Int          *fine_to_coarse_offd = NULL;
   NALU_HYPRE_Int          *fine_to_fine_offd = NULL;

   NALU_HYPRE_Int           i, j, jj;
   NALU_HYPRE_Int           startc, index;
   NALU_HYPRE_Int           cpt, fpt, new_fpt, row, rowc;
   NALU_HYPRE_Int          *CF_marker_offd = NULL;
   NALU_HYPRE_Int          *int_buf_data = NULL;
   NALU_HYPRE_BigInt       *big_convert;
   NALU_HYPRE_BigInt       *big_convert_offd = NULL;
   NALU_HYPRE_BigInt       *big_buf_data = NULL;

   NALU_HYPRE_BigInt        total_global_fpts, total_global_cpts, total_global_new_fpts;
   NALU_HYPRE_BigInt        fpts_starts[2], new_fpts_starts[2];
   NALU_HYPRE_Int           my_id, num_procs, num_sends;
   NALU_HYPRE_Int           d_count_FF, d_count_FC, o_count_FF, o_count_FC;
   NALU_HYPRE_Int           n_Fpts;
   NALU_HYPRE_Int           n_new_Fpts;
   NALU_HYPRE_Int          *cpt_array, *fpt_array, *new_fpt_array;
   NALU_HYPRE_Int           start, stop;
   NALU_HYPRE_Int           num_threads;

   num_threads = nalu_hypre_NumThreads();

   /* MPI size and rank*/
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   fine_to_coarse = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   fine_to_fine = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   big_convert = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, n_fine, NALU_HYPRE_MEMORY_HOST);

   cpt_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads + 1, NALU_HYPRE_MEMORY_HOST);
   fpt_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads + 1, NALU_HYPRE_MEMORY_HOST);
   new_fpt_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads + 1, NALU_HYPRE_MEMORY_HOST);
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel private(i,j,jj,start,stop,row,rowc,cpt,new_fpt,fpt,d_count_FC,d_count_FF,o_count_FC,o_count_FF)
#endif
   {
      NALU_HYPRE_Int my_thread_num = nalu_hypre_GetThreadNum();

      start = (n_fine / num_threads) * my_thread_num;
      if (my_thread_num == num_threads - 1)
      {
         stop = n_fine;
      }
      else
      {
         stop = (n_fine / num_threads) * (my_thread_num + 1);
      }
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] > 0)
         {
            cpt_array[my_thread_num + 1]++;
         }
         else if (CF_marker[i] == -2)
         {
            new_fpt_array[my_thread_num + 1]++;
            fpt_array[my_thread_num + 1]++;
         }
         else
         {
            fpt_array[my_thread_num + 1]++;
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         for (i = 1; i < num_threads; i++)
         {
            cpt_array[i + 1] += cpt_array[i];
            fpt_array[i + 1] += fpt_array[i];
            new_fpt_array[i + 1] += new_fpt_array[i];
         }
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      cpt = cpt_array[my_thread_num];
      fpt = fpt_array[my_thread_num];
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] > 0)
         {
            fine_to_coarse[i] = cpt++;
            fine_to_fine[i] = -1;
         }
         else
         {
            fine_to_fine[i] = fpt++;
            fine_to_coarse[i] = -1;
         }
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      if (my_thread_num == 0)
      {
         NALU_HYPRE_BigInt big_Fpts, big_new_Fpts;
         n_Fpts = fpt_array[num_threads];
         n_new_Fpts = new_fpt_array[num_threads];
         big_Fpts = n_Fpts;
         big_new_Fpts = n_new_Fpts;

         nalu_hypre_MPI_Scan(&big_Fpts, fpts_starts + 1, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
         nalu_hypre_MPI_Scan(&big_new_Fpts, new_fpts_starts + 1, 1, NALU_HYPRE_MPI_BIG_INT,
                        nalu_hypre_MPI_SUM, comm);
         fpts_starts[0] = fpts_starts[1] - big_Fpts;
         new_fpts_starts[0] = new_fpts_starts[1] - big_new_Fpts;
         if (my_id == num_procs - 1)
         {
            total_global_new_fpts = new_fpts_starts[1];
            total_global_fpts = fpts_starts[1];
            total_global_cpts = cpts_starts[1];
         }
         nalu_hypre_MPI_Bcast(&total_global_new_fpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
         nalu_hypre_MPI_Bcast(&total_global_fpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
         nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] > 0)
         {
            big_convert[i] = (NALU_HYPRE_BigInt)fine_to_coarse[i] + cpts_starts[0];
         }
         else
         {
            big_convert[i] = (NALU_HYPRE_BigInt)fine_to_fine[i] + fpts_starts[0];
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         if (num_cols_A_offd)
         {
            CF_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
            big_convert_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
            fine_to_coarse_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
            fine_to_fine_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
         }
         index = 0;
         num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
         int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int,
                                      nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                      NALU_HYPRE_MEMORY_HOST);
         big_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,
                                      nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                      NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_sends; i++)
         {
            startc = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = startc; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               int_buf_data[index] = CF_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
               big_buf_data[index++] = big_convert[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
         }

         comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, CF_marker_offd);

         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

         comm_handle = nalu_hypre_ParCSRCommHandleCreate(21, comm_pkg, big_buf_data, big_convert_offd);

         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

         num_cols_offd_A_FC = 0;
         num_cols_offd_A_FF = 0;
         if (num_cols_A_offd)
         {
            for (i = 0; i < num_cols_A_offd; i++)
            {
               if (CF_marker_offd[i] > 0)
               {
                  fine_to_coarse_offd[i] = num_cols_offd_A_FC++;
                  fine_to_fine_offd[i] = -1;
               }
               else
               {
                  fine_to_fine_offd[i] = num_cols_offd_A_FF++;
                  fine_to_coarse_offd[i] = -1;
               }
            }

            col_map_offd_A_FF = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_A_FF, NALU_HYPRE_MEMORY_HOST);
            col_map_offd_A_FC = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_A_FC, NALU_HYPRE_MEMORY_HOST);

            cpt = 0;
            fpt = 0;
            for (i = 0; i < num_cols_A_offd; i++)
            {
               if (CF_marker_offd[i] > 0)
               {
                  col_map_offd_A_FC[cpt++] = big_convert_offd[i];
               }
               else
               {
                  col_map_offd_A_FF[fpt++] = big_convert_offd[i];
               }
            }
         }

         A_FF_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_new_Fpts + 1, memory_location_P);
         A_FC_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_Fpts + 1, memory_location_P);
         A_FF_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_new_Fpts + 1, memory_location_P);
         A_FC_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_Fpts + 1, memory_location_P);
         D_lambda = nalu_hypre_CTAlloc(NALU_HYPRE_Real, n_Fpts, memory_location_P);
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      d_count_FC = 0;
      d_count_FF = 0;
      o_count_FC = 0;
      o_count_FF = 0;
      row = new_fpt_array[my_thread_num];
      rowc = fpt_array[my_thread_num];
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] == -2)
         {
            row++;
            rowc++;
            d_count_FF++; /* account for diagonal element */
            for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
            {
               jj = S_diag_j[j];
               if (CF_marker[jj] > 0)
               {
                  d_count_FC++;
               }
               else
               {
                  d_count_FF++;
               }
            }
            A_FF_diag_i[row] = d_count_FF;
            A_FC_diag_i[rowc] = d_count_FC;
            for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
            {
               jj = S_offd_j[j];
               if (CF_marker_offd[jj] > 0)
               {
                  o_count_FC++;
               }
               else
               {
                  o_count_FF++;
               }
            }
            A_FF_offd_i[row] = o_count_FF;
            A_FC_offd_i[rowc] = o_count_FC;
         }
         else if (CF_marker[i] < 0)
         {
            rowc++;
            for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
            {
               jj = S_diag_j[j];
               if (CF_marker[jj] > 0)
               {
                  d_count_FC++;
               }
            }
            A_FC_diag_i[rowc] = d_count_FC;
            for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
            {
               jj = S_offd_j[j];
               if (CF_marker_offd[jj] > 0)
               {
                  o_count_FC++;
               }
            }
            A_FC_offd_i[rowc] = o_count_FC;
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         NALU_HYPRE_Int fpt2, new_fpt2;
         for (i = 1; i < num_threads + 1; i++)
         {
            fpt = fpt_array[i];
            new_fpt = new_fpt_array[i];
            fpt2 = fpt_array[i - 1];
            new_fpt2 = new_fpt_array[i - 1];
            if (fpt != fpt2)
            {
               A_FC_diag_i[fpt] += A_FC_diag_i[fpt2];
               A_FC_offd_i[fpt] += A_FC_offd_i[fpt2];
            }
            if (new_fpt != new_fpt2)
            {
               A_FF_diag_i[new_fpt] += A_FF_diag_i[new_fpt2];
               A_FF_offd_i[new_fpt] += A_FF_offd_i[new_fpt2];
            }
         }
         row = new_fpt_array[num_threads];
         rowc = fpt_array[num_threads];
         d_count_FC = A_FC_diag_i[rowc];
         d_count_FF = A_FF_diag_i[row];
         o_count_FC = A_FC_offd_i[rowc];
         o_count_FF = A_FF_offd_i[row];
         A_FF_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, d_count_FF, memory_location_P);
         A_FC_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, d_count_FC, memory_location_P);
         A_FF_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, o_count_FF, memory_location_P);
         A_FC_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, o_count_FC, memory_location_P);
         A_FF_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, d_count_FF, memory_location_P);
         A_FC_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, d_count_FC, memory_location_P);
         A_FF_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, o_count_FF, memory_location_P);
         A_FC_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, o_count_FC, memory_location_P);
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      row = new_fpt_array[my_thread_num];
      rowc = fpt_array[my_thread_num];
      d_count_FC = A_FC_diag_i[rowc];
      d_count_FF = A_FF_diag_i[row];
      o_count_FC = A_FC_offd_i[rowc];
      o_count_FF = A_FF_offd_i[row];
      for (i = start; i < stop; i++)
      {
         if (CF_marker[i] == -2)
         {
            NALU_HYPRE_Int jS, jA;
            NALU_HYPRE_Real sum = 0;
            row++;
            jA = A_diag_i[i];
            A_FF_diag_j[d_count_FF] = fine_to_fine[A_diag_j[jA]];
            A_FF_diag_data[d_count_FF++] = A_diag_data[jA++];
            for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
            {
               jA = A_diag_i[i] + 1;
               jS = S_diag_j[j];
               while (A_diag_j[jA] != jS) { jA++; }
               if (CF_marker[S_diag_j[j]] > 0)
               {
                  A_FC_diag_j[d_count_FC] = fine_to_coarse[A_diag_j[jA]];
                  A_FC_diag_data[d_count_FC++] = A_diag_data[jA++];
               }
               else
               {
                  sum += 1;
                  D_lambda[rowc] += A_diag_data[jA];
                  A_FF_diag_j[d_count_FF] = fine_to_fine[A_diag_j[jA]];
                  A_FF_diag_data[d_count_FF++] = A_diag_data[jA++];
               }
            }
            for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
            {
               jA = A_offd_i[i];
               jS = S_offd_j[j];
               while (jS != A_offd_j[jA]) { jA++; }
               if (CF_marker_offd[S_offd_j[j]] > 0)
               {
                  A_FC_offd_j[o_count_FC] = fine_to_coarse_offd[A_offd_j[jA]];
                  A_FC_offd_data[o_count_FC++] = A_offd_data[jA++];
               }
               else
               {
                  sum += 1;
                  D_lambda[rowc] += A_offd_data[jA];
                  A_FF_offd_j[o_count_FF] = fine_to_fine_offd[A_offd_j[jA]];
                  A_FF_offd_data[o_count_FF++] = A_offd_data[jA++];
               }
            }
            if (sum) { D_lambda[rowc] = D_lambda[rowc] / sum; }
            rowc++;
            A_FF_diag_i[row] = d_count_FF;
            A_FC_diag_i[rowc] = d_count_FC;
            A_FF_offd_i[row] = o_count_FF;
            A_FC_offd_i[rowc] = o_count_FC;
         }
         else if (CF_marker[i] < 0)
         {
            NALU_HYPRE_Int jS, jA;
            NALU_HYPRE_Real sum = 0;
            for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
            {
               jA = A_diag_i[i] + 1;
               jS = S_diag_j[j];
               while (A_diag_j[jA] != jS) { jA++; }
               if (CF_marker[S_diag_j[j]] > 0)
               {
                  A_FC_diag_j[d_count_FC] = fine_to_coarse[A_diag_j[jA]];
                  A_FC_diag_data[d_count_FC++] = A_diag_data[jA++];
               }
               else
               {
                  sum += 1;
                  D_lambda[rowc] += A_diag_data[jA];
               }
            }
            for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
            {
               jA = A_offd_i[i];
               jS = S_offd_j[j];
               while (jS != A_offd_j[jA]) { jA++; }
               if (CF_marker_offd[S_offd_j[j]] > 0)
               {
                  A_FC_offd_j[o_count_FC] = fine_to_coarse_offd[A_offd_j[jA]];
                  A_FC_offd_data[o_count_FC++] = A_offd_data[jA++];
               }
               else
               {
                  sum += 1;
                  D_lambda[rowc] += A_offd_data[jA];
               }
            }
            if (sum) { D_lambda[rowc] = D_lambda[rowc] / sum; }
            rowc++;
            A_FC_diag_i[rowc] = d_count_FC;
            A_FC_offd_i[rowc] = o_count_FC;
         }
      }
   } /*end parallel region */

   A_FC = nalu_hypre_ParCSRMatrixCreate(comm,
                                   total_global_fpts,
                                   total_global_cpts,
                                   fpts_starts,
                                   cpts_starts,
                                   num_cols_offd_A_FC,
                                   A_FC_diag_i[n_Fpts],
                                   A_FC_offd_i[n_Fpts]);

   A_FF = nalu_hypre_ParCSRMatrixCreate(comm,
                                   total_global_new_fpts,
                                   total_global_fpts,
                                   new_fpts_starts,
                                   fpts_starts,
                                   num_cols_offd_A_FF,
                                   A_FF_diag_i[n_new_Fpts],
                                   A_FF_offd_i[n_new_Fpts]);

   A_FC_diag = nalu_hypre_ParCSRMatrixDiag(A_FC);
   nalu_hypre_CSRMatrixData(A_FC_diag) = A_FC_diag_data;
   nalu_hypre_CSRMatrixI(A_FC_diag) = A_FC_diag_i;
   nalu_hypre_CSRMatrixJ(A_FC_diag) = A_FC_diag_j;
   A_FC_offd = nalu_hypre_ParCSRMatrixOffd(A_FC);
   nalu_hypre_CSRMatrixData(A_FC_offd) = A_FC_offd_data;
   nalu_hypre_CSRMatrixI(A_FC_offd) = A_FC_offd_i;
   nalu_hypre_CSRMatrixJ(A_FC_offd) = A_FC_offd_j;
   nalu_hypre_ParCSRMatrixColMapOffd(A_FC) = col_map_offd_A_FC;

   nalu_hypre_CSRMatrixMemoryLocation(A_FC_diag) = memory_location_P;
   nalu_hypre_CSRMatrixMemoryLocation(A_FC_offd) = memory_location_P;

   A_FF_diag = nalu_hypre_ParCSRMatrixDiag(A_FF);
   nalu_hypre_CSRMatrixData(A_FF_diag) = A_FF_diag_data;
   nalu_hypre_CSRMatrixI(A_FF_diag) = A_FF_diag_i;
   nalu_hypre_CSRMatrixJ(A_FF_diag) = A_FF_diag_j;
   A_FF_offd = nalu_hypre_ParCSRMatrixOffd(A_FF);
   nalu_hypre_CSRMatrixData(A_FF_offd) = A_FF_offd_data;
   nalu_hypre_CSRMatrixI(A_FF_offd) = A_FF_offd_i;
   nalu_hypre_CSRMatrixJ(A_FF_offd) = A_FF_offd_j;
   nalu_hypre_ParCSRMatrixColMapOffd(A_FF) = col_map_offd_A_FF;

   nalu_hypre_CSRMatrixMemoryLocation(A_FF_diag) = memory_location_P;
   nalu_hypre_CSRMatrixMemoryLocation(A_FF_offd) = memory_location_P;

   nalu_hypre_TFree(fine_to_coarse, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_fine, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_convert, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_coarse_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fine_to_fine_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_convert_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(big_buf_data, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(cpt_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(fpt_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(new_fpt_array, NALU_HYPRE_MEMORY_HOST);

   *A_FC_ptr = A_FC;
   *A_FF_ptr = A_FF;
   *D_lambda_ptr = D_lambda;

   return nalu_hypre_error_flag;
}
