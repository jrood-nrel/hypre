/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGBuildCoarseOperator
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildCoarseOperator( nalu_hypre_ParCSRMatrix  *RT,
                                    nalu_hypre_ParCSRMatrix  *A,
                                    nalu_hypre_ParCSRMatrix  *P,
                                    nalu_hypre_ParCSRMatrix **RAP_ptr )
{
   nalu_hypre_BoomerAMGBuildCoarseOperatorKT( RT, A, P, 0, RAP_ptr);
   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildCoarseOperatorKT( nalu_hypre_ParCSRMatrix  *RT,
                                      nalu_hypre_ParCSRMatrix  *A,
                                      nalu_hypre_ParCSRMatrix  *P,
                                      NALU_HYPRE_Int keepTranspose,
                                      nalu_hypre_ParCSRMatrix **RAP_ptr )

{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_RAP] -= nalu_hypre_MPI_Wtime();
#endif

   MPI_Comm        comm = nalu_hypre_ParCSRMatrixComm(A);

   NALU_HYPRE_MemoryLocation memory_location_RAP = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   nalu_hypre_CSRMatrix *RT_diag = nalu_hypre_ParCSRMatrixDiag(RT);
   nalu_hypre_CSRMatrix *RT_offd = nalu_hypre_ParCSRMatrixOffd(RT);
   NALU_HYPRE_Int             num_cols_diag_RT = nalu_hypre_CSRMatrixNumCols(RT_diag);
   NALU_HYPRE_Int             num_cols_offd_RT = nalu_hypre_CSRMatrixNumCols(RT_offd);
   NALU_HYPRE_Int             num_rows_offd_RT = nalu_hypre_CSRMatrixNumRows(RT_offd);
   nalu_hypre_ParCSRCommPkg   *comm_pkg_RT = nalu_hypre_ParCSRMatrixCommPkg(RT);
   NALU_HYPRE_Int             num_recvs_RT = 0;
   NALU_HYPRE_Int             num_sends_RT = 0;
   NALU_HYPRE_Int             *send_map_starts_RT;
   NALU_HYPRE_Int             *send_map_elmts_RT;

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);

   NALU_HYPRE_Real      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int             *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int             *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);

   NALU_HYPRE_Real      *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int             *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int             *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);

   NALU_HYPRE_Int  num_cols_diag_A = nalu_hypre_CSRMatrixNumCols(A_diag);
   NALU_HYPRE_Int  num_cols_offd_A = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_CSRMatrix *P_diag = nalu_hypre_ParCSRMatrixDiag(P);

   NALU_HYPRE_Real      *P_diag_data = nalu_hypre_CSRMatrixData(P_diag);
   NALU_HYPRE_Int             *P_diag_i = nalu_hypre_CSRMatrixI(P_diag);
   NALU_HYPRE_Int             *P_diag_j = nalu_hypre_CSRMatrixJ(P_diag);

   nalu_hypre_CSRMatrix *P_offd = nalu_hypre_ParCSRMatrixOffd(P);
   NALU_HYPRE_BigInt    *col_map_offd_P = nalu_hypre_ParCSRMatrixColMapOffd(P);

   NALU_HYPRE_Real      *P_offd_data = nalu_hypre_CSRMatrixData(P_offd);
   NALU_HYPRE_Int             *P_offd_i = nalu_hypre_CSRMatrixI(P_offd);
   NALU_HYPRE_Int             *P_offd_j = nalu_hypre_CSRMatrixJ(P_offd);

   NALU_HYPRE_BigInt  first_col_diag_P = nalu_hypre_ParCSRMatrixFirstColDiag(P);
   NALU_HYPRE_BigInt  last_col_diag_P;
   NALU_HYPRE_Int  num_cols_diag_P = nalu_hypre_CSRMatrixNumCols(P_diag);
   NALU_HYPRE_Int  num_cols_offd_P = nalu_hypre_CSRMatrixNumCols(P_offd);
   NALU_HYPRE_BigInt *coarse_partitioning = nalu_hypre_ParCSRMatrixColStarts(P);
   NALU_HYPRE_BigInt *RT_partitioning = nalu_hypre_ParCSRMatrixColStarts(RT);

   nalu_hypre_ParCSRMatrix *RAP;
   NALU_HYPRE_BigInt       *col_map_offd_RAP = NULL;
   NALU_HYPRE_BigInt       *new_col_map_offd_RAP = NULL;

   nalu_hypre_CSRMatrix *RAP_int = NULL;

   NALU_HYPRE_Real      *RAP_int_data;
   NALU_HYPRE_Int             *RAP_int_i;
   NALU_HYPRE_BigInt    *RAP_int_j;

   nalu_hypre_CSRMatrix *RAP_ext;

   NALU_HYPRE_Real      *RAP_ext_data = NULL;
   NALU_HYPRE_Int             *RAP_ext_i = NULL;
   NALU_HYPRE_BigInt    *RAP_ext_j = NULL;

   nalu_hypre_CSRMatrix *RAP_diag;

   NALU_HYPRE_Real      *RAP_diag_data;
   NALU_HYPRE_Int             *RAP_diag_i;
   NALU_HYPRE_Int             *RAP_diag_j;

   nalu_hypre_CSRMatrix *RAP_offd;

   NALU_HYPRE_Real      *RAP_offd_data = NULL;
   NALU_HYPRE_Int             *RAP_offd_i = NULL;
   NALU_HYPRE_Int             *RAP_offd_j = NULL;

   NALU_HYPRE_Int              RAP_size;
   NALU_HYPRE_Int              RAP_ext_size;
   NALU_HYPRE_Int              RAP_diag_size;
   NALU_HYPRE_Int              RAP_offd_size;
   NALU_HYPRE_Int              P_ext_diag_size;
   NALU_HYPRE_Int              P_ext_offd_size;
   NALU_HYPRE_BigInt     first_col_diag_RAP;
   NALU_HYPRE_BigInt     last_col_diag_RAP;
   NALU_HYPRE_Int              num_cols_offd_RAP = 0;

   nalu_hypre_CSRMatrix *R_diag;

   NALU_HYPRE_Real      *R_diag_data;
   NALU_HYPRE_Int             *R_diag_i;
   NALU_HYPRE_Int             *R_diag_j;

   nalu_hypre_CSRMatrix *R_offd;

   NALU_HYPRE_Real      *R_offd_data;
   NALU_HYPRE_Int             *R_offd_i;
   NALU_HYPRE_Int             *R_offd_j;

   NALU_HYPRE_Real *RA_diag_data_array = NULL;
   NALU_HYPRE_Int *RA_diag_j_array = NULL;
   NALU_HYPRE_Real *RA_offd_data_array = NULL;
   NALU_HYPRE_Int *RA_offd_j_array = NULL;

   nalu_hypre_CSRMatrix *Ps_ext;

   NALU_HYPRE_Real      *Ps_ext_data;
   NALU_HYPRE_Int             *Ps_ext_i;
   NALU_HYPRE_BigInt    *Ps_ext_j;

   NALU_HYPRE_Real      *P_ext_diag_data = NULL;
   NALU_HYPRE_Int             *P_ext_diag_i = NULL;
   NALU_HYPRE_Int             *P_ext_diag_j = NULL;

   NALU_HYPRE_Real      *P_ext_offd_data = NULL;
   NALU_HYPRE_Int             *P_ext_offd_i = NULL;
   NALU_HYPRE_Int             *P_ext_offd_j = NULL;
   NALU_HYPRE_BigInt    *P_big_offd_j = NULL;

   NALU_HYPRE_BigInt    *col_map_offd_Pext;
   NALU_HYPRE_Int             *map_P_to_Pext = NULL;
   NALU_HYPRE_Int             *map_P_to_RAP = NULL;
   NALU_HYPRE_Int             *map_Pext_to_RAP = NULL;

   NALU_HYPRE_Int             *P_marker;
   NALU_HYPRE_Int            **P_mark_array;
   NALU_HYPRE_Int            **A_mark_array;
   NALU_HYPRE_Int             *A_marker;
   NALU_HYPRE_BigInt    *temp;

   NALU_HYPRE_BigInt     n_coarse, n_coarse_RT;
   NALU_HYPRE_Int              square = 1;
   NALU_HYPRE_Int              num_cols_offd_Pext = 0;

   NALU_HYPRE_Int              ic, i, j, k;
   NALU_HYPRE_Int              i1, i2, i3, ii, ns, ne, size, rest;
   NALU_HYPRE_Int              cnt = 0; /*value; */
   NALU_HYPRE_Int              jj1, jj2, jj3, jcol;

   NALU_HYPRE_Int             *jj_count, *jj_cnt_diag, *jj_cnt_offd;
   NALU_HYPRE_Int              jj_counter, jj_count_diag, jj_count_offd;
   NALU_HYPRE_Int              jj_row_begining, jj_row_begin_diag, jj_row_begin_offd;
   NALU_HYPRE_Int              start_indexing = 0; /* start indexing for RAP_data at 0 */
   NALU_HYPRE_Int              num_nz_cols_A;
   NALU_HYPRE_Int              num_procs;
   NALU_HYPRE_Int              num_threads;

   NALU_HYPRE_Real       r_entry;
   NALU_HYPRE_Real       r_a_product;
   NALU_HYPRE_Real       r_a_p_product;

   NALU_HYPRE_Real       zero = 0.0;
   NALU_HYPRE_Int      *prefix_sum_workspace;

   /*-----------------------------------------------------------------------
    *  Copy ParCSRMatrix RT into CSRMatrix R so that we have row-wise access
    *  to restriction .
    *-----------------------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   num_threads = nalu_hypre_NumThreads();

   if (comm_pkg_RT)
   {
      num_recvs_RT = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg_RT);
      num_sends_RT = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg_RT);
      send_map_starts_RT = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg_RT);
      send_map_elmts_RT = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg_RT);
   }
   else if (num_procs > 1)
   {
      nalu_hypre_MatvecCommPkgCreate(RT);
      comm_pkg_RT = nalu_hypre_ParCSRMatrixCommPkg(RT);
      num_recvs_RT = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg_RT);
      num_sends_RT = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg_RT);
      send_map_starts_RT = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg_RT);
      send_map_elmts_RT = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg_RT);
   }

   nalu_hypre_CSRMatrixTranspose(RT_diag, &R_diag, 1);
   if (num_cols_offd_RT)
   {
      nalu_hypre_CSRMatrixTranspose(RT_offd, &R_offd, 1);
      R_offd_data = nalu_hypre_CSRMatrixData(R_offd);
      R_offd_i    = nalu_hypre_CSRMatrixI(R_offd);
      R_offd_j    = nalu_hypre_CSRMatrixJ(R_offd);
   }

   /*-----------------------------------------------------------------------
    *  Access the CSR vectors for R. Also get sizes of fine and
    *  coarse grids.
    *-----------------------------------------------------------------------*/

   R_diag_data = nalu_hypre_CSRMatrixData(R_diag);
   R_diag_i    = nalu_hypre_CSRMatrixI(R_diag);
   R_diag_j    = nalu_hypre_CSRMatrixJ(R_diag);

   n_coarse = nalu_hypre_ParCSRMatrixGlobalNumCols(P);
   num_nz_cols_A = num_cols_diag_A + num_cols_offd_A;

   n_coarse_RT = nalu_hypre_ParCSRMatrixGlobalNumCols(RT);

   if (n_coarse != n_coarse_RT || num_cols_diag_RT != num_cols_diag_P)
   {
      square = 0;
   }

   /*-----------------------------------------------------------------------
    *  Generate Ps_ext, i.e. portion of P that is stored on neighbor procs
    *  and needed locally for triple matrix product
    *-----------------------------------------------------------------------*/

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_UnorderedIntMap send_map_elmts_RT_inverse_map;
   NALU_HYPRE_Int *send_map_elmts_starts_RT_aggregated = NULL;
   NALU_HYPRE_Int *send_map_elmts_RT_aggregated = NULL;

   NALU_HYPRE_Int send_map_elmts_RT_inverse_map_initialized =
      num_sends_RT > 0 && send_map_starts_RT[num_sends_RT] - send_map_starts_RT[0] > 0;
   if (send_map_elmts_RT_inverse_map_initialized)
   {
      nalu_hypre_UnorderedIntSet send_map_elmts_set;
      nalu_hypre_UnorderedIntSetCreate(&send_map_elmts_set,
                                  2 * (send_map_starts_RT[num_sends_RT] - send_map_starts_RT[0]), 16 * nalu_hypre_NumThreads());

      #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
      for (i = send_map_starts_RT[0]; i < send_map_starts_RT[num_sends_RT]; i++)
      {
         NALU_HYPRE_Int key = send_map_elmts_RT[i];
         nalu_hypre_UnorderedIntSetPut(&send_map_elmts_set, key);
      }

      NALU_HYPRE_Int send_map_elmts_unique_size;
      NALU_HYPRE_Int *send_map_elmts_unique = nalu_hypre_UnorderedIntSetCopyToArray(&send_map_elmts_set,
                                                                          &send_map_elmts_unique_size);
      nalu_hypre_UnorderedIntSetDestroy(&send_map_elmts_set);

      nalu_hypre_UnorderedIntMapCreate(&send_map_elmts_RT_inverse_map, 2 * send_map_elmts_unique_size,
                                  16 * nalu_hypre_NumThreads());
      #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
      for (i = 0; i < send_map_elmts_unique_size; i++)
      {
         nalu_hypre_UnorderedIntMapPutIfAbsent(&send_map_elmts_RT_inverse_map, send_map_elmts_unique[i], i);
      }
      nalu_hypre_TFree(send_map_elmts_unique, NALU_HYPRE_MEMORY_HOST);

      send_map_elmts_starts_RT_aggregated = nalu_hypre_TAlloc(NALU_HYPRE_Int,  send_map_elmts_unique_size + 1,
                                                         NALU_HYPRE_MEMORY_HOST);
      send_map_elmts_RT_aggregated = nalu_hypre_TAlloc(NALU_HYPRE_Int,  send_map_starts_RT[num_sends_RT],
                                                  NALU_HYPRE_MEMORY_HOST);

      #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
      for (i = 0; i < send_map_elmts_unique_size; i++)
      {
         send_map_elmts_starts_RT_aggregated[i] = 0;
      }

      #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
      for (i = send_map_starts_RT[0]; i < send_map_starts_RT[num_sends_RT]; i++)
      {
         NALU_HYPRE_Int idx = nalu_hypre_UnorderedIntMapGet(&send_map_elmts_RT_inverse_map, send_map_elmts_RT[i]);
         #pragma omp atomic
         send_map_elmts_starts_RT_aggregated[idx]++;
      }

      for (i = 0; i < send_map_elmts_unique_size - 1; i++)
      {
         send_map_elmts_starts_RT_aggregated[i + 1] += send_map_elmts_starts_RT_aggregated[i];
      }
      send_map_elmts_starts_RT_aggregated[send_map_elmts_unique_size] = send_map_starts_RT[num_sends_RT];

      #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
      for (i = send_map_starts_RT[num_sends_RT] - 1; i >= send_map_starts_RT[0]; i--)
      {
         NALU_HYPRE_Int idx = nalu_hypre_UnorderedIntMapGet(&send_map_elmts_RT_inverse_map, send_map_elmts_RT[i]);
         NALU_HYPRE_Int offset = nalu_hypre_fetch_and_add(send_map_elmts_starts_RT_aggregated + idx, -1) - 1;
         send_map_elmts_RT_aggregated[offset] = i;
      }
   }
#endif /* NALU_HYPRE_CONCURRENT_HOPSCOTCH */

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX] -= nalu_hypre_MPI_Wtime();
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX_RAP] -= nalu_hypre_MPI_Wtime();
#endif

   if (num_procs > 1)
   {
      Ps_ext = nalu_hypre_ParCSRMatrixExtractBExt(P, A, 1);
      Ps_ext_data = nalu_hypre_CSRMatrixData(Ps_ext);
      Ps_ext_i    = nalu_hypre_CSRMatrixI(Ps_ext);
      Ps_ext_j    = nalu_hypre_CSRMatrixBigJ(Ps_ext);
   }

   P_ext_diag_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_offd_A + 1, NALU_HYPRE_MEMORY_HOST);
   P_ext_offd_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_offd_A + 1, NALU_HYPRE_MEMORY_HOST);
   P_ext_diag_i[0] = 0;
   P_ext_offd_i[0] = 0;
   P_ext_diag_size = 0;
   P_ext_offd_size = 0;
   last_col_diag_P = first_col_diag_P + (NALU_HYPRE_BigInt) num_cols_diag_P - 1;

   /*NALU_HYPRE_Int prefix_sum_workspace[2*(num_threads + 1)];*/
   prefix_sum_workspace = nalu_hypre_TAlloc(NALU_HYPRE_Int,  2 * (num_threads + 1), NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel private(i,j)
#endif /* This threading causes problem, maybe the prefix_sum in combination with BigInt? */
   {
      NALU_HYPRE_Int i_begin, i_end;
      nalu_hypre_GetSimpleThreadPartition(&i_begin, &i_end, num_cols_offd_A);

      NALU_HYPRE_Int P_ext_diag_size_private = 0;
      NALU_HYPRE_Int P_ext_offd_size_private = 0;

      for (i = i_begin; i < i_end; i++)
      {
         for (j = Ps_ext_i[i]; j < Ps_ext_i[i + 1]; j++)
            if (Ps_ext_j[j] < first_col_diag_P || Ps_ext_j[j] > last_col_diag_P)
            {
               P_ext_offd_size_private++;
            }
            else
            {
               P_ext_diag_size_private++;
            }
      }

      nalu_hypre_prefix_sum_pair(&P_ext_diag_size_private, &P_ext_diag_size, &P_ext_offd_size_private,
                            &P_ext_offd_size, prefix_sum_workspace);

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp master
#endif
      {
         if (P_ext_diag_size)
         {
            P_ext_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_ext_diag_size, NALU_HYPRE_MEMORY_HOST);
            P_ext_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  P_ext_diag_size, NALU_HYPRE_MEMORY_HOST);
         }
         if (P_ext_offd_size)
         {
            P_ext_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
            P_big_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  P_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
            P_ext_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  P_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
            //temp = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  P_ext_offd_size+num_cols_offd_P, NALU_HYPRE_MEMORY_HOST);
         }
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      for (i = i_begin; i < i_end; i++)
      {
         for (j = Ps_ext_i[i]; j < Ps_ext_i[i + 1]; j++)
         {
            NALU_HYPRE_BigInt value = Ps_ext_j[j];
            if (value < first_col_diag_P || value > last_col_diag_P)
            {
               //Ps_ext_j[P_ext_offd_size_private] = value;
               //temp[P_ext_offd_size_private] = value;
               P_big_offd_j[P_ext_offd_size_private] = value;
               P_ext_offd_data[P_ext_offd_size_private++] = Ps_ext_data[j];
            }
            else
            {
               P_ext_diag_j[P_ext_diag_size_private] = (NALU_HYPRE_Int)(Ps_ext_j[j] - first_col_diag_P);
               P_ext_diag_data[P_ext_diag_size_private++] = Ps_ext_data[j];
            }
         }
         P_ext_diag_i[i + 1] = P_ext_diag_size_private;
         P_ext_offd_i[i + 1] = P_ext_offd_size_private;
      }
   } /* omp parallel */
   nalu_hypre_TFree(prefix_sum_workspace, NALU_HYPRE_MEMORY_HOST);

   if (num_procs > 1)
   {
      nalu_hypre_CSRMatrixDestroy(Ps_ext);
      Ps_ext = NULL;
   }

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   if (P_ext_offd_size || num_cols_offd_P)
   {
      nalu_hypre_UnorderedBigIntSet found_set;
      nalu_hypre_UnorderedBigIntSetCreate(&found_set, P_ext_offd_size + num_cols_offd_P,
                                     16 * nalu_hypre_NumThreads());

      #pragma omp parallel private(i)
      {
         #pragma omp for NALU_HYPRE_SMP_SCHEDULE
         for (i = 0; i < P_ext_offd_size; i++)
         {
            //nalu_hypre_UnorderedBigIntSetPut(&found_set, Ps_ext_j[i]);
            nalu_hypre_UnorderedBigIntSetPut(&found_set, P_big_offd_j[i]);
         }

         #pragma omp for NALU_HYPRE_SMP_SCHEDULE
         for (i = 0; i < num_cols_offd_P; i++)
         {
            nalu_hypre_UnorderedBigIntSetPut(&found_set, col_map_offd_P[i]);
         }
      } /* omp parallel */

      /* Warning on getting temp right !!!!! */

      temp = nalu_hypre_UnorderedBigIntSetCopyToArray(&found_set, &num_cols_offd_Pext);
      nalu_hypre_UnorderedBigIntSetDestroy(&found_set);

      nalu_hypre_UnorderedBigIntMap col_map_offd_Pext_inverse;
      nalu_hypre_big_sort_and_create_inverse_map(temp, num_cols_offd_Pext, &col_map_offd_Pext,
                                            &col_map_offd_Pext_inverse);

      #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
      for (i = 0 ; i < P_ext_offd_size; i++)
         //Ps_ext_j[i] = nalu_hypre_UnorderedBigIntMapGet(&col_map_offd_Pext_inverse, Ps_ext_j[i]);
      {
         P_ext_offd_j[i] = nalu_hypre_UnorderedBigIntMapGet(&col_map_offd_Pext_inverse, P_big_offd_j[i]);
      }
      if (num_cols_offd_Pext) { nalu_hypre_UnorderedBigIntMapDestroy(&col_map_offd_Pext_inverse); }
   }
#else /* !NALU_HYPRE_CONCURRENT_HOPSCOTCH */
   if (P_ext_offd_size || num_cols_offd_P)
   {
      temp = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  P_ext_offd_size + num_cols_offd_P, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < P_ext_offd_size; i++)
         //Ps_ext_j[i] = temp[i];
         //temp[i] = Ps_ext_j[i];
      {
         temp[i] = P_big_offd_j[i];
      }
      cnt = P_ext_offd_size;
      for (i = 0; i < num_cols_offd_P; i++)
      {
         temp[cnt++] = col_map_offd_P[i];
      }
   }
   if (cnt)
   {
      nalu_hypre_BigQsort0(temp, 0, cnt - 1);

      num_cols_offd_Pext = 1;
      NALU_HYPRE_BigInt value = temp[0];
      for (i = 1; i < cnt; i++)
      {
         if (temp[i] > value)
         {
            value = temp[i];
            temp[num_cols_offd_Pext++] = value;
         }
      }
   }

   if (num_cols_offd_Pext)
   {
      col_map_offd_Pext = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_Pext, NALU_HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < num_cols_offd_Pext; i++)
   {
      col_map_offd_Pext[i] = temp[i];
   }

   if (P_ext_offd_size || num_cols_offd_P)
   {
      nalu_hypre_TFree(temp, NALU_HYPRE_MEMORY_HOST);
   }

   /*if (P_ext_offd_size)
     P_ext_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_ext_offd_size, NALU_HYPRE_MEMORY_HOST);*/
   for (i = 0 ; i < P_ext_offd_size; i++)
      P_ext_offd_j[i] = nalu_hypre_BigBinarySearch(col_map_offd_Pext,
                                              //Ps_ext_j[i],
                                              P_big_offd_j[i],
                                              num_cols_offd_Pext);
#endif /* !NALU_HYPRE_CONCURRENT_HOPSCOTCH */

   if (P_ext_offd_size)
   {
      nalu_hypre_TFree(P_big_offd_j, NALU_HYPRE_MEMORY_HOST);
   }
   /*if (num_procs > 1)
     {
     nalu_hypre_CSRMatrixDestroy(Ps_ext);
     Ps_ext = NULL;
     }*/

   if (num_cols_offd_P)
   {
      map_P_to_Pext = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_P, NALU_HYPRE_MEMORY_HOST);

      cnt = 0;
      for (i = 0; i < num_cols_offd_Pext; i++)
         if (col_map_offd_Pext[i] == col_map_offd_P[cnt])
         {
            map_P_to_Pext[cnt++] = i;
            if (cnt == num_cols_offd_P) { break; }
         }
   }
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX] += nalu_hypre_MPI_Wtime();
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX_RAP] += nalu_hypre_MPI_Wtime();
#endif

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of RAP_int and set up RAP_int_i if there
    *  are more than one processor and nonzero elements in R_offd
    *-----------------------------------------------------------------------*/

   P_mark_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  num_threads, NALU_HYPRE_MEMORY_HOST);
   A_mark_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  num_threads, NALU_HYPRE_MEMORY_HOST);

   if (num_cols_offd_RT)
   {
      jj_count = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_threads, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i,ii,ic,i1,i2,i3,jj1,jj2,jj3,ns,ne,size,rest,jj_counter,jj_row_begining,A_marker,P_marker) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (ii = 0; ii < num_threads; ii++)
      {
         size = num_cols_offd_RT / num_threads;
         rest = num_cols_offd_RT - size * num_threads;
         if (ii < rest)
         {
            ns = ii * size + ii;
            ne = (ii + 1) * size + ii + 1;
         }
         else
         {
            ns = ii * size + rest;
            ne = (ii + 1) * size + rest;
         }

         /*-----------------------------------------------------------------------
          *  Allocate marker arrays.
          *-----------------------------------------------------------------------*/

         if (num_cols_offd_Pext || num_cols_diag_P)
         {
            P_mark_array[ii] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_diag_P + num_cols_offd_Pext,
                                             NALU_HYPRE_MEMORY_HOST);
            P_marker = P_mark_array[ii];
         }
         A_mark_array[ii] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_nz_cols_A, NALU_HYPRE_MEMORY_HOST);
         A_marker = A_mark_array[ii];
         /*-----------------------------------------------------------------------
          *  Initialize some stuff.
          *-----------------------------------------------------------------------*/

         jj_counter = start_indexing;
         for (ic = 0; ic < num_cols_diag_P + num_cols_offd_Pext; ic++)
         {
            P_marker[ic] = -1;
         }
         for (i = 0; i < num_nz_cols_A; i++)
         {
            A_marker[i] = -1;
         }

         /*-----------------------------------------------------------------------
          *  Loop over exterior c-points
          *-----------------------------------------------------------------------*/

         for (ic = ns; ic < ne; ic++)
         {

            jj_row_begining = jj_counter;

            /*--------------------------------------------------------------------
             *  Loop over entries in row ic of R_offd.
             *--------------------------------------------------------------------*/

            for (jj1 = R_offd_i[ic]; jj1 < R_offd_i[ic + 1]; jj1++)
            {
               i1  = R_offd_j[jj1];

               /*-----------------------------------------------------------------
                *  Loop over entries in row i1 of A_offd.
                *-----------------------------------------------------------------*/

               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
               {
                  i2 = A_offd_j[jj2];

                  /*--------------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited. New entries in RAP only occur from unmarked points.
                   *--------------------------------------------------------------*/

                  if (A_marker[i2] != ic)
                  {

                     /*-----------------------------------------------------------
                      *  Mark i2 as visited.
                      *-----------------------------------------------------------*/

                     A_marker[i2] = ic;

                     /*-----------------------------------------------------------
                      *  Loop over entries in row i2 of P_ext.
                      *-----------------------------------------------------------*/

                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];

                        /*--------------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not already
                         *  been accounted for. If it has not, mark it and increment
                         *  counter.
                         *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter++;
                        }
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_offd_j[jj3] + num_cols_diag_P;

                        /*--------------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not already
                         *  been accounted for. If it has not, mark it and increment
                         *  counter.
                         *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter++;
                        }
                     }
                  }
               }
               /*-----------------------------------------------------------------
                *  Loop over entries in row i1 of A_diag.
                *-----------------------------------------------------------------*/

               for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
               {
                  i2 = A_diag_j[jj2];

                  /*--------------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited. New entries in RAP only occur from unmarked points.
                   *--------------------------------------------------------------*/

                  if (A_marker[i2 + num_cols_offd_A] != ic)
                  {

                     /*-----------------------------------------------------------
                      *  Mark i2 as visited.
                      *-----------------------------------------------------------*/

                     A_marker[i2 + num_cols_offd_A] = ic;

                     /*-----------------------------------------------------------
                      *  Loop over entries in row i2 of P_diag.
                      *-----------------------------------------------------------*/

                     for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_diag_j[jj3];

                        /*--------------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not already
                         *  been accounted for. If it has not, mark it and increment
                         *  counter.
                         *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter++;
                        }
                     }
                     /*-----------------------------------------------------------
                      *  Loop over entries in row i2 of P_offd.
                      *-----------------------------------------------------------*/

                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;

                        /*--------------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not already
                         *  been accounted for. If it has not, mark it and increment
                         *  counter.
                         *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter++;
                        }
                     }
                  }
               }
            }
         }

         jj_count[ii] = jj_counter;

      }

      /*-----------------------------------------------------------------------
       *  Allocate RAP_int_data and RAP_int_j arrays.
       *-----------------------------------------------------------------------*/
      for (i = 0; i < num_threads - 1; i++)
      {
         jj_count[i + 1] += jj_count[i];
      }

      RAP_size = jj_count[num_threads - 1];
      RAP_int_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd_RT + 1, NALU_HYPRE_MEMORY_HOST);
      RAP_int_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  RAP_size, NALU_HYPRE_MEMORY_HOST);
      RAP_int_j    = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  RAP_size, NALU_HYPRE_MEMORY_HOST);

      RAP_int_i[num_cols_offd_RT] = RAP_size;

      /*-----------------------------------------------------------------------
       *  Second Pass: Fill in RAP_int_data and RAP_int_j.
       *-----------------------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i,ii,ic,i1,i2,i3,jj1,jj2,jj3,ns,ne,size,rest,jj_counter,jj_row_begining,A_marker,P_marker,r_entry,r_a_product,r_a_p_product) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (ii = 0; ii < num_threads; ii++)
      {
         size = num_cols_offd_RT / num_threads;
         rest = num_cols_offd_RT - size * num_threads;
         if (ii < rest)
         {
            ns = ii * size + ii;
            ne = (ii + 1) * size + ii + 1;
         }
         else
         {
            ns = ii * size + rest;
            ne = (ii + 1) * size + rest;
         }

         /*-----------------------------------------------------------------------
          *  Initialize some stuff.
          *-----------------------------------------------------------------------*/
         if (num_cols_offd_Pext || num_cols_diag_P)
         {
            P_marker = P_mark_array[ii];
         }
         A_marker = A_mark_array[ii];

         jj_counter = start_indexing;
         if (ii > 0) { jj_counter = jj_count[ii - 1]; }

         for (ic = 0; ic < num_cols_diag_P + num_cols_offd_Pext; ic++)
         {
            P_marker[ic] = -1;
         }
         for (i = 0; i < num_nz_cols_A; i++)
         {
            A_marker[i] = -1;
         }

         /*-----------------------------------------------------------------------
          *  Loop over exterior c-points.
          *-----------------------------------------------------------------------*/

         for (ic = ns; ic < ne; ic++)
         {

            jj_row_begining = jj_counter;
            RAP_int_i[ic] = jj_counter;

            /*--------------------------------------------------------------------
             *  Loop over entries in row ic of R_offd.
             *--------------------------------------------------------------------*/

            for (jj1 = R_offd_i[ic]; jj1 < R_offd_i[ic + 1]; jj1++)
            {
               i1  = R_offd_j[jj1];
               r_entry = R_offd_data[jj1];

               /*-----------------------------------------------------------------
                *  Loop over entries in row i1 of A_offd.
                *-----------------------------------------------------------------*/

               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
               {
                  i2 = A_offd_j[jj2];
                  r_a_product = r_entry * A_offd_data[jj2];

                  /*--------------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited. New entries in RAP only occur from unmarked points.
                   *--------------------------------------------------------------*/

                  if (A_marker[i2] != ic)
                  {

                     /*-----------------------------------------------------------
                      *  Mark i2 as visited.
                      *-----------------------------------------------------------*/

                     A_marker[i2] = ic;

                     /*-----------------------------------------------------------
                      *  Loop over entries in row i2 of P_ext.
                      *-----------------------------------------------------------*/

                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];
                        r_a_p_product = r_a_product * P_ext_diag_data[jj3];

                        /*--------------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not already
                         *  been accounted for. If it has not, create a new entry.
                         *  If it has, add new contribution.
                         *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           RAP_int_data[jj_counter] = r_a_p_product;
                           RAP_int_j[jj_counter] = (NALU_HYPRE_BigInt)i3 + first_col_diag_P;
                           jj_counter++;
                        }
                        else
                        {
                           RAP_int_data[P_marker[i3]] += r_a_p_product;
                        }
                     }

                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_offd_j[jj3] + num_cols_diag_P;
                        r_a_p_product = r_a_product * P_ext_offd_data[jj3];

                        /*--------------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not already
                         *  been accounted for. If it has not, create a new entry.
                         *  If it has, add new contribution.
                         *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           RAP_int_data[jj_counter] = r_a_p_product;
                           RAP_int_j[jj_counter]
                              = col_map_offd_Pext[i3 - num_cols_diag_P];
                           jj_counter++;
                        }
                        else
                        {
                           RAP_int_data[P_marker[i3]] += r_a_p_product;
                        }
                     }
                  }

                  /*--------------------------------------------------------------
                   *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                   *  no new entries in RAP and can just add new contributions.
                   *--------------------------------------------------------------*/

                  else
                  {
                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];
                        r_a_p_product = r_a_product * P_ext_diag_data[jj3];
                        RAP_int_data[P_marker[i3]] += r_a_p_product;
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_offd_j[jj3] + num_cols_diag_P;
                        r_a_p_product = r_a_product * P_ext_offd_data[jj3];
                        RAP_int_data[P_marker[i3]] += r_a_p_product;
                     }
                  }
               }

               /*-----------------------------------------------------------------
                *  Loop over entries in row i1 of A_diag.
                *-----------------------------------------------------------------*/

               for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
               {
                  i2 = A_diag_j[jj2];
                  r_a_product = r_entry * A_diag_data[jj2];

                  /*--------------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited. New entries in RAP only occur from unmarked points.
                   *--------------------------------------------------------------*/

                  if (A_marker[i2 + num_cols_offd_A] != ic)
                  {

                     /*-----------------------------------------------------------
                      *  Mark i2 as visited.
                      *-----------------------------------------------------------*/

                     A_marker[i2 + num_cols_offd_A] = ic;

                     /*-----------------------------------------------------------
                      *  Loop over entries in row i2 of P_diag.
                      *-----------------------------------------------------------*/

                     for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_diag_j[jj3];
                        r_a_p_product = r_a_product * P_diag_data[jj3];

                        /*--------------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not already
                         *  been accounted for. If it has not, create a new entry.
                         *  If it has, add new contribution.
                         *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           RAP_int_data[jj_counter] = r_a_p_product;
                           RAP_int_j[jj_counter] = (NALU_HYPRE_BigInt)i3 + first_col_diag_P;
                           jj_counter++;
                        }
                        else
                        {
                           RAP_int_data[P_marker[i3]] += r_a_p_product;
                        }
                     }
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;
                        r_a_p_product = r_a_product * P_offd_data[jj3];

                        /*--------------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not already
                         *  been accounted for. If it has not, create a new entry.
                         *  If it has, add new contribution.
                         *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           RAP_int_data[jj_counter] = r_a_p_product;
                           RAP_int_j[jj_counter] =
                              col_map_offd_Pext[i3 - num_cols_diag_P];
                           jj_counter++;
                        }
                        else
                        {
                           RAP_int_data[P_marker[i3]] += r_a_p_product;
                        }
                     }
                  }

                  /*--------------------------------------------------------------
                   *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                   *  no new entries in RAP and can just add new contributions.
                   *--------------------------------------------------------------*/

                  else
                  {
                     for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_diag_j[jj3];
                        r_a_p_product = r_a_product * P_diag_data[jj3];
                        RAP_int_data[P_marker[i3]] += r_a_p_product;
                     }
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;
                        r_a_p_product = r_a_product * P_offd_data[jj3];
                        RAP_int_data[P_marker[i3]] += r_a_p_product;
                     }
                  }
               }
            }
         }
         if (num_cols_offd_Pext || num_cols_diag_P)
         {
            nalu_hypre_TFree(P_mark_array[ii], NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_TFree(A_mark_array[ii], NALU_HYPRE_MEMORY_HOST);
      }

      RAP_int = nalu_hypre_CSRMatrixCreate(num_cols_offd_RT, num_rows_offd_RT, RAP_size);

      nalu_hypre_CSRMatrixMemoryLocation(RAP_int) = NALU_HYPRE_MEMORY_HOST;

      nalu_hypre_CSRMatrixI(RAP_int) = RAP_int_i;
      nalu_hypre_CSRMatrixBigJ(RAP_int) = RAP_int_j;
      nalu_hypre_CSRMatrixData(RAP_int) = RAP_int_data;
      nalu_hypre_TFree(jj_count, NALU_HYPRE_MEMORY_HOST);
   }

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX] -= nalu_hypre_MPI_Wtime();
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX_RAP] -= nalu_hypre_MPI_Wtime();
#endif

   RAP_ext_size = 0;
   if (num_sends_RT || num_recvs_RT)
   {
      void *request;
      nalu_hypre_ExchangeExternalRowsInit(RAP_int, comm_pkg_RT, &request);
      RAP_ext = nalu_hypre_ExchangeExternalRowsWait(request);
      RAP_ext_i = nalu_hypre_CSRMatrixI(RAP_ext);
      RAP_ext_j = nalu_hypre_CSRMatrixBigJ(RAP_ext);
      RAP_ext_data = nalu_hypre_CSRMatrixData(RAP_ext);
      RAP_ext_size = RAP_ext_i[nalu_hypre_CSRMatrixNumRows(RAP_ext)];
   }
   if (num_cols_offd_RT)
   {
      nalu_hypre_CSRMatrixDestroy(RAP_int);
      RAP_int = NULL;
   }

   RAP_diag_i = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_cols_diag_RT + 1, memory_location_RAP);
   RAP_offd_i = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_cols_diag_RT + 1, memory_location_RAP);

   first_col_diag_RAP = first_col_diag_P;
   last_col_diag_RAP = first_col_diag_P + num_cols_diag_P - 1;

   /*-----------------------------------------------------------------------
    *  check for new nonzero columns in RAP_offd generated through RAP_ext
    *-----------------------------------------------------------------------*/

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_UnorderedBigIntMap col_map_offd_RAP_inverse;
   if (RAP_ext_size || num_cols_offd_Pext)
   {
      nalu_hypre_UnorderedBigIntSet found_set;
      nalu_hypre_UnorderedBigIntSetCreate(&found_set, 2 * (RAP_ext_size + num_cols_offd_Pext),
                                     16 * nalu_hypre_NumThreads());
      cnt = 0;

      #pragma omp parallel private(i)
      {
         #pragma omp for NALU_HYPRE_SMP_SCHEDULE
         for (i = 0; i < RAP_ext_size; i++)
         {
            if (RAP_ext_j[i] < first_col_diag_RAP
                || RAP_ext_j[i] > last_col_diag_RAP)
            {
               nalu_hypre_UnorderedBigIntSetPut(&found_set, RAP_ext_j[i]);
            }
         }

         #pragma omp for NALU_HYPRE_SMP_SCHEDULE
         for (i = 0; i < num_cols_offd_Pext; i++)
         {
            nalu_hypre_UnorderedBigIntSetPut(&found_set, col_map_offd_Pext[i]);
         }
      } /* omp parallel */

      temp = nalu_hypre_UnorderedBigIntSetCopyToArray(&found_set, &num_cols_offd_RAP);
      nalu_hypre_UnorderedBigIntSetDestroy(&found_set);
      nalu_hypre_big_sort_and_create_inverse_map(temp, num_cols_offd_RAP, &col_map_offd_RAP,
                                            &col_map_offd_RAP_inverse);
   }
#else /* !NALU_HYPRE_CONCURRENT_HOPSCOTCH */
   if (RAP_ext_size || num_cols_offd_Pext)
   {
      temp = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, RAP_ext_size + num_cols_offd_Pext, NALU_HYPRE_MEMORY_HOST);
      cnt = 0;
      for (i = 0; i < RAP_ext_size; i++)
         if (RAP_ext_j[i] < first_col_diag_RAP
             || RAP_ext_j[i] > last_col_diag_RAP)
         {
            temp[cnt++] = RAP_ext_j[i];
         }
      for (i = 0; i < num_cols_offd_Pext; i++)
      {
         temp[cnt++] = col_map_offd_Pext[i];
      }


      if (cnt)
      {
         nalu_hypre_BigQsort0(temp, 0, cnt - 1);
         NALU_HYPRE_BigInt value = temp[0];
         num_cols_offd_RAP = 1;
         for (i = 1; i < cnt; i++)
         {
            if (temp[i] > value)
            {
               value = temp[i];
               temp[num_cols_offd_RAP++] = value;
            }
         }
      }

      /* now evaluate col_map_offd_RAP */
      if (num_cols_offd_RAP)
      {
         col_map_offd_RAP = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_cols_offd_RAP, NALU_HYPRE_MEMORY_HOST);
      }

      for (i = 0 ; i < num_cols_offd_RAP; i++)
      {
         col_map_offd_RAP[i] = temp[i];
      }

      nalu_hypre_TFree(temp, NALU_HYPRE_MEMORY_HOST);
   }
#endif /* !NALU_HYPRE_CONCURRENT_HOPSCOTCH */

   if (num_cols_offd_P)
   {
      map_P_to_RAP = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_offd_P, NALU_HYPRE_MEMORY_HOST);

      cnt = 0;
      for (i = 0; i < num_cols_offd_RAP; i++)
         if (col_map_offd_RAP[i] == col_map_offd_P[cnt])
         {
            map_P_to_RAP[cnt++] = i;
            if (cnt == num_cols_offd_P) { break; }
         }
   }

   if (num_cols_offd_Pext)
   {
      map_Pext_to_RAP = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_offd_Pext, NALU_HYPRE_MEMORY_HOST);

      cnt = 0;
      for (i = 0; i < num_cols_offd_RAP; i++)
         if (col_map_offd_RAP[i] == col_map_offd_Pext[cnt])
         {
            map_Pext_to_RAP[cnt++] = i;
            if (cnt == num_cols_offd_Pext) { break; }
         }
   }

   /*-----------------------------------------------------------------------
    *  Convert RAP_ext column indices
    *-----------------------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < RAP_ext_size; i++)
      if (RAP_ext_j[i] < first_col_diag_RAP
          || RAP_ext_j[i] > last_col_diag_RAP)
         RAP_ext_j[i] = (NALU_HYPRE_BigInt)num_cols_diag_P
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
                        + (NALU_HYPRE_BigInt)nalu_hypre_UnorderedBigIntMapGet(&col_map_offd_RAP_inverse, RAP_ext_j[i]);
#else
                        +(NALU_HYPRE_BigInt)nalu_hypre_BigBinarySearch(col_map_offd_RAP, RAP_ext_j[i], num_cols_offd_RAP);
#endif
      else
      {
         RAP_ext_j[i] -= first_col_diag_RAP;
      }

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   if (num_cols_offd_RAP)
   {
      nalu_hypre_UnorderedBigIntMapDestroy(&col_map_offd_RAP_inverse);
   }
#endif

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX] += nalu_hypre_MPI_Wtime();
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX_RAP] += nalu_hypre_MPI_Wtime();
#endif

   /*   need to allocate new P_marker etc. and make further changes */
   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/
   jj_cnt_diag = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_threads, NALU_HYPRE_MEMORY_HOST);
   jj_cnt_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_threads, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,k,jcol,ii,ic,i1,i2,i3,jj1,jj2,jj3,ns,ne,size,rest,jj_count_diag,jj_count_offd,jj_row_begin_diag,jj_row_begin_offd,A_marker,P_marker) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (ii = 0; ii < num_threads; ii++)
   {
      size = num_cols_diag_RT / num_threads;
      rest = num_cols_diag_RT - size * num_threads;
      if (ii < rest)
      {
         ns = ii * size + ii;
         ne = (ii + 1) * size + ii + 1;
      }
      else
      {
         ns = ii * size + rest;
         ne = (ii + 1) * size + rest;
      }

      P_mark_array[ii] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_diag_P + num_cols_offd_RAP,
                                       NALU_HYPRE_MEMORY_HOST);
      A_mark_array[ii] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_nz_cols_A, NALU_HYPRE_MEMORY_HOST);
      P_marker = P_mark_array[ii];
      A_marker = A_mark_array[ii];
      jj_count_diag = start_indexing;
      jj_count_offd = start_indexing;

      for (ic = 0; ic < num_cols_diag_P + num_cols_offd_RAP; ic++)
      {
         P_marker[ic] = -1;
      }
      for (i = 0; i < num_nz_cols_A; i++)
      {
         A_marker[i] = -1;
      }

      /*-----------------------------------------------------------------------
       *  Loop over interior c-points.
       *-----------------------------------------------------------------------*/

      for (ic = ns; ic < ne; ic++)
      {

         /*--------------------------------------------------------------------
          *  Set marker for diagonal entry, RAP_{ic,ic}. and for all points
          *  being added to row ic of RAP_diag and RAP_offd through RAP_ext
          *--------------------------------------------------------------------*/

         jj_row_begin_diag = jj_count_diag;
         jj_row_begin_offd = jj_count_offd;

         if (square)
         {
            P_marker[ic] = jj_count_diag++;
         }

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
         if (send_map_elmts_RT_inverse_map_initialized)
         {
            NALU_HYPRE_Int i = nalu_hypre_UnorderedIntMapGet(&send_map_elmts_RT_inverse_map, ic);
            if (i != -1)
            {
               for (j = send_map_elmts_starts_RT_aggregated[i]; j < send_map_elmts_starts_RT_aggregated[i + 1];
                    j++)
               {
                  NALU_HYPRE_Int jj = send_map_elmts_RT_aggregated[j];
                  for (k = RAP_ext_i[jj]; k < RAP_ext_i[jj + 1]; k++)
                  {
                     jcol = (NALU_HYPRE_Int)RAP_ext_j[k];
                     if (jcol < num_cols_diag_P)
                     {
                        if (P_marker[jcol] < jj_row_begin_diag)
                        {
                           P_marker[jcol] = jj_count_diag;
                           jj_count_diag++;
                        }
                     }
                     else
                     {
                        if (P_marker[jcol] < jj_row_begin_offd)
                        {
                           P_marker[jcol] = jj_count_offd;
                           jj_count_offd++;
                        }
                     }
                  }
               }
            } // if (set)
         }
#else /* !NALU_HYPRE_CONCURRENT_HOPSCOTCH */
         for (i = 0; i < num_sends_RT; i++)
            for (j = send_map_starts_RT[i]; j < send_map_starts_RT[i + 1]; j++)
               if (send_map_elmts_RT[j] == ic)
               {
                  for (k = RAP_ext_i[j]; k < RAP_ext_i[j + 1]; k++)
                  {
                     jcol = (NALU_HYPRE_Int) RAP_ext_j[k];
                     if (jcol < num_cols_diag_P)
                     {
                        if (P_marker[jcol] < jj_row_begin_diag)
                        {
                           P_marker[jcol] = jj_count_diag;
                           jj_count_diag++;
                        }
                     }
                     else
                     {
                        if (P_marker[jcol] < jj_row_begin_offd)
                        {
                           P_marker[jcol] = jj_count_offd;
                           jj_count_offd++;
                        }
                     }
                  }
                  break;
               }
#endif /* !NALU_HYPRE_CONCURRENT_HOPSCOTCH */

         /*--------------------------------------------------------------------
          *  Loop over entries in row ic of R_diag.
          *--------------------------------------------------------------------*/

         for (jj1 = R_diag_i[ic]; jj1 < R_diag_i[ic + 1]; jj1++)
         {
            i1  = R_diag_j[jj1];

            /*-----------------------------------------------------------------
             *  Loop over entries in row i1 of A_offd.
             *-----------------------------------------------------------------*/

            if (num_cols_offd_A)
            {
               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
               {
                  i2 = A_offd_j[jj2];

                  /*--------------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited. New entries in RAP only occur from unmarked points.
                   *--------------------------------------------------------------*/

                  if (A_marker[i2] != ic)
                  {

                     /*-----------------------------------------------------------
                      *  Mark i2 as visited.
                      *-----------------------------------------------------------*/

                     A_marker[i2] = ic;

                     /*-----------------------------------------------------------
                      *  Loop over entries in row i2 of P_ext.
                      *-----------------------------------------------------------*/

                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];

                        /*--------------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not already
                         *  been accounted for. If it has not, mark it and increment
                         *  counter.
                         *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begin_diag)
                        {
                           P_marker[i3] = jj_count_diag;
                           jj_count_diag++;
                        }
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]] + num_cols_diag_P;

                        /*--------------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not already
                         *  been accounted for. If it has not, mark it and increment
                         *  counter.
                         *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begin_offd)
                        {
                           P_marker[i3] = jj_count_offd;
                           jj_count_offd++;
                        }
                     }
                  }
               }
            }
            /*-----------------------------------------------------------------
             *  Loop over entries in row i1 of A_diag.
             *-----------------------------------------------------------------*/

            for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
            {
               i2 = A_diag_j[jj2];

               /*--------------------------------------------------------------
                *  Check A_marker to see if point i2 has been previously
                *  visited. New entries in RAP only occur from unmarked points.
                *--------------------------------------------------------------*/

               if (A_marker[i2 + num_cols_offd_A] != ic)
               {

                  /*-----------------------------------------------------------
                   *  Mark i2 as visited.
                   *-----------------------------------------------------------*/

                  A_marker[i2 + num_cols_offd_A] = ic;

                  /*-----------------------------------------------------------
                   *  Loop over entries in row i2 of P_diag.
                   *-----------------------------------------------------------*/

                  for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++)
                  {
                     i3 = P_diag_j[jj3];

                     /*--------------------------------------------------------
                      *  Check P_marker to see that RAP_{ic,i3} has not already
                      *  been accounted for. If it has not, mark it and increment
                      *  counter.
                      *--------------------------------------------------------*/

                     if (P_marker[i3] < jj_row_begin_diag)
                     {
                        P_marker[i3] = jj_count_diag;
                        jj_count_diag++;
                     }
                  }
                  /*-----------------------------------------------------------
                   *  Loop over entries in row i2 of P_offd.
                   *-----------------------------------------------------------*/

                  if (num_cols_offd_P)
                  {
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;

                        /*--------------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not already
                         *  been accounted for. If it has not, mark it and increment
                         *  counter.
                         *--------------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begin_offd)
                        {
                           P_marker[i3] = jj_count_offd;
                           jj_count_offd++;
                        }
                     }
                  }
               }
            }
         }

         /*--------------------------------------------------------------------
          * Set RAP_diag_i and RAP_offd_i for this row.
          *--------------------------------------------------------------------*/
         /*
            RAP_diag_i[ic] = jj_row_begin_diag;
            RAP_offd_i[ic] = jj_row_begin_offd;
            */
      }
      jj_cnt_diag[ii] = jj_count_diag;
      jj_cnt_offd[ii] = jj_count_offd;
   }

   for (i = 0; i < num_threads - 1; i++)
   {
      jj_cnt_diag[i + 1] += jj_cnt_diag[i];
      jj_cnt_offd[i + 1] += jj_cnt_offd[i];
   }

   jj_count_diag = jj_cnt_diag[num_threads - 1];
   jj_count_offd = jj_cnt_offd[num_threads - 1];

   RAP_diag_i[num_cols_diag_RT] = jj_count_diag;
   RAP_offd_i[num_cols_diag_RT] = jj_count_offd;

   /*-----------------------------------------------------------------------
    *  Allocate RAP_diag_data and RAP_diag_j arrays.
    *  Allocate RAP_offd_data and RAP_offd_j arrays.
    *-----------------------------------------------------------------------*/

   RAP_diag_size = jj_count_diag;
   if (RAP_diag_size)
   {
      RAP_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, RAP_diag_size, memory_location_RAP);
      RAP_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  RAP_diag_size, memory_location_RAP);
   }

   RAP_offd_size = jj_count_offd;
   if (RAP_offd_size)
   {
      RAP_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real, RAP_offd_size, memory_location_RAP);
      RAP_offd_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  RAP_offd_size, memory_location_RAP);
   }

   if (RAP_offd_size == 0 && num_cols_offd_RAP != 0)
   {
      num_cols_offd_RAP = 0;
      nalu_hypre_TFree(col_map_offd_RAP, NALU_HYPRE_MEMORY_HOST);
   }

   RA_diag_data_array = nalu_hypre_TAlloc(NALU_HYPRE_Real,  num_cols_diag_A * num_threads, NALU_HYPRE_MEMORY_HOST);
   RA_diag_j_array = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_cols_diag_A * num_threads, NALU_HYPRE_MEMORY_HOST);
   if (num_cols_offd_A)
   {
      RA_offd_data_array = nalu_hypre_TAlloc(NALU_HYPRE_Real,  num_cols_offd_A * num_threads, NALU_HYPRE_MEMORY_HOST);
      RA_offd_j_array = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_cols_offd_A * num_threads, NALU_HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in RAP_diag_data and RAP_diag_j.
    *  Second Pass: Fill in RAP_offd_data and RAP_offd_j.
    *-----------------------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,k,jcol,ii,ic,i1,i2,i3,jj1,jj2,jj3,ns,ne,size,rest,jj_count_diag,jj_count_offd,jj_row_begin_diag,jj_row_begin_offd,A_marker,P_marker,r_entry,r_a_product,r_a_p_product) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (ii = 0; ii < num_threads; ii++)
   {
      size = num_cols_diag_RT / num_threads;
      rest = num_cols_diag_RT - size * num_threads;
      if (ii < rest)
      {
         ns = ii * size + ii;
         ne = (ii + 1) * size + ii + 1;
      }
      else
      {
         ns = ii * size + rest;
         ne = (ii + 1) * size + rest;
      }

      /*-----------------------------------------------------------------------
       *  Initialize some stuff.
       *-----------------------------------------------------------------------*/

      P_marker = P_mark_array[ii];
      A_marker = A_mark_array[ii];
      for (ic = 0; ic < num_cols_diag_P + num_cols_offd_RAP; ic++)
      {
         P_marker[ic] = -1;
      }
      for (i = 0; i < num_nz_cols_A ; i++)
      {
         A_marker[i] = -1;
      }

      jj_count_diag = start_indexing;
      jj_count_offd = start_indexing;
      if (ii > 0)
      {
         jj_count_diag = jj_cnt_diag[ii - 1];
         jj_count_offd = jj_cnt_offd[ii - 1];
      }

      // temporal matrix RA = R*A
      // only need to store one row per thread because R*A and (R*A)*P are fused
      // into one loop.
      nalu_hypre_CSRMatrix RA_diag, RA_offd;
      RA_diag.data = RA_diag_data_array + num_cols_diag_A * ii;
      RA_diag.j = RA_diag_j_array + num_cols_diag_A * ii;
      RA_diag.num_nonzeros = 0;
      RA_offd.num_nonzeros = 0;

      if (num_cols_offd_A)
      {
         RA_offd.data = RA_offd_data_array + num_cols_offd_A * ii;
         RA_offd.j = RA_offd_j_array + num_cols_offd_A * ii;
      }

      /*-----------------------------------------------------------------------
       *  Loop over interior c-points.
       *-----------------------------------------------------------------------*/

      for (ic = ns; ic < ne; ic++)
      {

         /*--------------------------------------------------------------------
          *  Create diagonal entry, RAP_{ic,ic} and add entries of RAP_ext
          *--------------------------------------------------------------------*/

         jj_row_begin_diag = jj_count_diag;
         jj_row_begin_offd = jj_count_offd;
         RAP_diag_i[ic] = jj_row_begin_diag;
         RAP_offd_i[ic] = jj_row_begin_offd;

         NALU_HYPRE_Int ra_row_begin_diag = RA_diag.num_nonzeros;
         NALU_HYPRE_Int ra_row_begin_offd = RA_offd.num_nonzeros;

         if (square)
         {
            P_marker[ic] = jj_count_diag;
            RAP_diag_data[jj_count_diag] = zero;
            RAP_diag_j[jj_count_diag] = ic;
            jj_count_diag++;
         }

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
         if (send_map_elmts_RT_inverse_map_initialized)
         {
            NALU_HYPRE_Int i = nalu_hypre_UnorderedIntMapGet(&send_map_elmts_RT_inverse_map, ic);
            if (i != -1)
            {
               for (j = send_map_elmts_starts_RT_aggregated[i]; j < send_map_elmts_starts_RT_aggregated[i + 1];
                    j++)
               {
                  NALU_HYPRE_Int jj = send_map_elmts_RT_aggregated[j];
                  for (k = RAP_ext_i[jj]; k < RAP_ext_i[jj + 1]; k++)
                  {
                     jcol = (NALU_HYPRE_Int)RAP_ext_j[k];
                     if (jcol < num_cols_diag_P)
                     {
                        if (P_marker[jcol] < jj_row_begin_diag)
                        {
                           P_marker[jcol] = jj_count_diag;
                           RAP_diag_data[jj_count_diag]
                              = RAP_ext_data[k];
                           RAP_diag_j[jj_count_diag] = jcol;
                           jj_count_diag++;
                        }
                        else
                           RAP_diag_data[P_marker[jcol]]
                           += RAP_ext_data[k];
                     }
                     else
                     {
                        if (P_marker[jcol] < jj_row_begin_offd)
                        {
                           P_marker[jcol] = jj_count_offd;
                           RAP_offd_data[jj_count_offd]
                              = RAP_ext_data[k];
                           RAP_offd_j[jj_count_offd]
                              = jcol - num_cols_diag_P;
                           jj_count_offd++;
                        }
                        else
                           RAP_offd_data[P_marker[jcol]]
                           += RAP_ext_data[k];
                     }
                  }
               }
            } // if (set)
         }
#else /* !NALU_HYPRE_CONCURRENT_HOPSCOTCH */
         for (i = 0; i < num_sends_RT; i++)
            for (j = send_map_starts_RT[i]; j < send_map_starts_RT[i + 1]; j++)
               if (send_map_elmts_RT[j] == ic)
               {
                  for (k = RAP_ext_i[j]; k < RAP_ext_i[j + 1]; k++)
                  {
                     jcol = (NALU_HYPRE_Int)RAP_ext_j[k];
                     if (jcol < num_cols_diag_P)
                     {
                        if (P_marker[jcol] < jj_row_begin_diag)
                        {
                           P_marker[jcol] = jj_count_diag;
                           RAP_diag_data[jj_count_diag]
                              = RAP_ext_data[k];
                           RAP_diag_j[jj_count_diag] = jcol;
                           jj_count_diag++;
                        }
                        else
                           RAP_diag_data[P_marker[jcol]]
                           += RAP_ext_data[k];
                     }
                     else
                     {
                        if (P_marker[jcol] < jj_row_begin_offd)
                        {
                           P_marker[jcol] = jj_count_offd;
                           RAP_offd_data[jj_count_offd]
                              = RAP_ext_data[k];
                           RAP_offd_j[jj_count_offd]
                              = jcol - num_cols_diag_P;
                           jj_count_offd++;
                        }
                        else
                           RAP_offd_data[P_marker[jcol]]
                           += RAP_ext_data[k];
                     }
                  }
                  break;
               }
#endif /* !NALU_HYPRE_CONCURRENT_HOPSCOTCH */

         /*--------------------------------------------------------------------
          *  Loop over entries in row ic of R_diag and compute row ic of RA.
          *--------------------------------------------------------------------*/

         for (jj1 = R_diag_i[ic]; jj1 < R_diag_i[ic + 1]; jj1++)
         {
            i1  = R_diag_j[jj1];
            r_entry = R_diag_data[jj1];

            /*-----------------------------------------------------------------
             *  Loop over entries in row i1 of A_offd.
             *-----------------------------------------------------------------*/

            if (num_cols_offd_A)
            {
               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
               {
                  i2 = A_offd_j[jj2];
                  NALU_HYPRE_Real a_entry = A_offd_data[jj2];
                  NALU_HYPRE_Int marker = A_marker[i2];

                  /*--------------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited. New entries in RAP only occur from unmarked points.
                   *--------------------------------------------------------------*/

                  if (marker < ra_row_begin_offd)
                  {
                     /*-----------------------------------------------------------
                      *  Mark i2 as visited.
                      *-----------------------------------------------------------*/

                     A_marker[i2] = RA_offd.num_nonzeros;
                     RA_offd.data[RA_offd.num_nonzeros - ra_row_begin_offd] = r_entry * a_entry;
                     RA_offd.j[RA_offd.num_nonzeros - ra_row_begin_offd] = i2;
                     RA_offd.num_nonzeros++;
                  }
                  /*--------------------------------------------------------------
                   *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                   *  no new entries in RA and can just add new contributions.
                   *--------------------------------------------------------------*/
                  else
                  {
                     RA_offd.data[marker - ra_row_begin_offd] += r_entry * a_entry;
                     // JSP: compiler will more likely to generate FMA instructions
                     // when we don't eliminate common subexpressions of
                     // r_entry * A_offd_data[jj2] manually.
                  }
               } // loop over entries in row i1 of A_offd
            } // num_cols_offd_A

            /*-----------------------------------------------------------------
             *  Loop over entries in row i1 of A_diag.
             *-----------------------------------------------------------------*/

            for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
            {
               i2 = A_diag_j[jj2];
               NALU_HYPRE_Real a_entry = A_diag_data[jj2];
               NALU_HYPRE_Int marker = A_marker[i2 + num_cols_offd_A];

               /*--------------------------------------------------------------
                *  Check A_marker to see if point i2 has been previously
                *  visited. New entries in RAP only occur from unmarked points.
                *--------------------------------------------------------------*/

               if (marker < ra_row_begin_diag)
               {
                  /*-----------------------------------------------------------
                   *  Mark i2 as visited.
                   *-----------------------------------------------------------*/
                  A_marker[i2 + num_cols_offd_A] = RA_diag.num_nonzeros;
                  RA_diag.data[RA_diag.num_nonzeros - ra_row_begin_diag] = r_entry * a_entry;
                  RA_diag.j[RA_diag.num_nonzeros - ra_row_begin_diag] = i2;
                  RA_diag.num_nonzeros++;
               }
               /*--------------------------------------------------------------
                *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                *  no new entries in RA and can just add new contributions.
                *--------------------------------------------------------------*/
               else
               {
                  RA_diag.data[marker - ra_row_begin_diag] += r_entry * a_entry;
               }
            } // loop over entries in row i1 of A_diag
         } // loop over entries in row ic of R_diag

         /*--------------------------------------------------------------------
          * Loop over entries in row ic of RA_offd.
          *--------------------------------------------------------------------*/

         for (jj1 = ra_row_begin_offd; jj1 < RA_offd.num_nonzeros; jj1++)
         {
            i1 = RA_offd.j[jj1 - ra_row_begin_offd];
            r_a_product = RA_offd.data[jj1 - ra_row_begin_offd];

            /*-----------------------------------------------------------
             *  Loop over entries in row i1 of P_ext.
             *-----------------------------------------------------------*/
            for (jj2 = P_ext_diag_i[i1]; jj2 < P_ext_diag_i[i1 + 1]; jj2++)
            {
               i2 = P_ext_diag_j[jj2];
               NALU_HYPRE_Real p_entry = P_ext_diag_data[jj2];
               NALU_HYPRE_Int marker = P_marker[i2];

               /*--------------------------------------------------------
                *  Check P_marker to see that RAP_{ic,i2} has not already
                *  been accounted for. If it has not, create a new entry.
                *  If it has, add new contribution.
                *--------------------------------------------------------*/
               if (marker < jj_row_begin_diag)
               {
                  P_marker[i2] = jj_count_diag;
                  RAP_diag_data[jj_count_diag] = r_a_product * p_entry;
                  RAP_diag_j[jj_count_diag] = i2;
                  jj_count_diag++;
               }
               else
               {
                  RAP_diag_data[marker] += r_a_product * p_entry;
               }
            }
            for (jj2 = P_ext_offd_i[i1]; jj2 < P_ext_offd_i[i1 + 1]; jj2++)
            {
               i2 = map_Pext_to_RAP[P_ext_offd_j[jj2]] + num_cols_diag_P;
               NALU_HYPRE_Real p_entry = P_ext_offd_data[jj2];
               NALU_HYPRE_Int marker = P_marker[i2];

               /*--------------------------------------------------------
                *  Check P_marker to see that RAP_{ic,i2} has not already
                *  been accounted for. If it has not, create a new entry.
                *  If it has, add new contribution.
                *--------------------------------------------------------*/
               if (marker < jj_row_begin_offd)
               {
                  P_marker[i2] = jj_count_offd;
                  RAP_offd_data[jj_count_offd] = r_a_product * p_entry;
                  RAP_offd_j[jj_count_offd] = i2 - num_cols_diag_P;
                  jj_count_offd++;
               }
               else
               {
                  RAP_offd_data[marker] += r_a_product * p_entry;
               }
            }
         } // loop over entries in row ic of RA_offd

         /*--------------------------------------------------------------------
          * Loop over entries in row ic of RA_diag.
          *--------------------------------------------------------------------*/

         for (jj1 = ra_row_begin_diag; jj1 < RA_diag.num_nonzeros; jj1++)
         {
            NALU_HYPRE_Int i1 = RA_diag.j[jj1 - ra_row_begin_diag];
            NALU_HYPRE_Real r_a_product = RA_diag.data[jj1 - ra_row_begin_diag];

            /*-----------------------------------------------------------------
             *  Loop over entries in row i1 of P_diag.
             *-----------------------------------------------------------------*/
            for (jj2 = P_diag_i[i1]; jj2 < P_diag_i[i1 + 1]; jj2++)
            {
               i2 = P_diag_j[jj2];
               NALU_HYPRE_Real p_entry = P_diag_data[jj2];
               NALU_HYPRE_Int marker = P_marker[i2];

               /*--------------------------------------------------------
                *  Check P_marker to see that RAP_{ic,i2} has not already
                *  been accounted for. If it has not, create a new entry.
                *  If it has, add new contribution.
                *--------------------------------------------------------*/

               if (marker < jj_row_begin_diag)
               {
                  P_marker[i2] = jj_count_diag;
                  RAP_diag_data[jj_count_diag] = r_a_product * p_entry;
                  RAP_diag_j[jj_count_diag] = i2;
                  jj_count_diag++;
               }
               else
               {
                  RAP_diag_data[marker] += r_a_product * p_entry;
               }
            }
            if (num_cols_offd_P)
            {
               for (jj2 = P_offd_i[i1]; jj2 < P_offd_i[i1 + 1]; jj2++)
               {
                  i2 = map_P_to_RAP[P_offd_j[jj2]] + num_cols_diag_P;
                  NALU_HYPRE_Real p_entry = P_offd_data[jj2];
                  NALU_HYPRE_Int marker = P_marker[i2];

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i2} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (marker < jj_row_begin_offd)
                  {
                     P_marker[i2] = jj_count_offd;
                     RAP_offd_data[jj_count_offd] = r_a_product * p_entry;
                     RAP_offd_j[jj_count_offd] = i2 - num_cols_diag_P;
                     jj_count_offd++;
                  }
                  else
                  {
                     RAP_offd_data[marker] += r_a_product * p_entry;
                  }
               }
            } // num_cols_offd_P
         } // loop over entries in row ic of RA_diag.
      } // Loop over interior c-points.
      nalu_hypre_TFree(P_mark_array[ii], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(A_mark_array[ii], NALU_HYPRE_MEMORY_HOST);
   } // omp parallel for

   /* check if really all off-diagonal entries occurring in col_map_offd_RAP
      are represented and eliminate if necessary */

   P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_RAP, NALU_HYPRE_MEMORY_HOST);
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_cols_offd_RAP; i++)
   {
      P_marker[i] = -1;
   }

   jj_count_offd = 0;
#ifdef NALU_HYPRE_USING_ATOMIC
   #pragma omp parallel for private(i3) reduction(+:jj_count_offd) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < RAP_offd_size; i++)
   {
      i3 = RAP_offd_j[i];
#ifdef NALU_HYPRE_USING_ATOMIC
      if (nalu_hypre_compare_and_swap(P_marker + i3, -1, 0) == -1)
      {
         jj_count_offd++;
      }
#else
      if (P_marker[i3])
      {
         P_marker[i3] = 0;
         jj_count_offd++;
      }
#endif
   }

   if (jj_count_offd < num_cols_offd_RAP)
   {
      new_col_map_offd_RAP = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, jj_count_offd, NALU_HYPRE_MEMORY_HOST);
      jj_counter = 0;
      for (i = 0; i < num_cols_offd_RAP; i++)
         if (!P_marker[i])
         {
            P_marker[i] = jj_counter;
            new_col_map_offd_RAP[jj_counter++] = col_map_offd_RAP[i];
         }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i3) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < RAP_offd_size; i++)
      {
         i3 = RAP_offd_j[i];
         RAP_offd_j[i] = P_marker[i3];
      }

      num_cols_offd_RAP = jj_count_offd;
      nalu_hypre_TFree(col_map_offd_RAP, NALU_HYPRE_MEMORY_HOST);
      col_map_offd_RAP = new_col_map_offd_RAP;
   }
   nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);

   RAP = nalu_hypre_ParCSRMatrixCreate(comm, n_coarse_RT, n_coarse,
                                  RT_partitioning, coarse_partitioning,
                                  num_cols_offd_RAP, RAP_diag_size,
                                  RAP_offd_size);

   RAP_diag = nalu_hypre_ParCSRMatrixDiag(RAP);
   nalu_hypre_CSRMatrixI(RAP_diag) = RAP_diag_i;
   if (RAP_diag_size)
   {
      nalu_hypre_CSRMatrixData(RAP_diag) = RAP_diag_data;
      nalu_hypre_CSRMatrixJ(RAP_diag) = RAP_diag_j;
   }

   RAP_offd = nalu_hypre_ParCSRMatrixOffd(RAP);
   nalu_hypre_CSRMatrixI(RAP_offd) = RAP_offd_i;
   if (num_cols_offd_RAP)
   {
      nalu_hypre_CSRMatrixData(RAP_offd) = RAP_offd_data;
      nalu_hypre_CSRMatrixJ(RAP_offd) = RAP_offd_j;
      nalu_hypre_ParCSRMatrixColMapOffd(RAP) = col_map_offd_RAP;
   }
   if (num_procs > 1)
   {
      /* nalu_hypre_GenerateRAPCommPkg(RAP, A); */
      nalu_hypre_MatvecCommPkgCreate(RAP);
   }

   *RAP_ptr = RAP;

   /*-----------------------------------------------------------------------
    *  Free R, P_ext and marker arrays.
    *-----------------------------------------------------------------------*/


   if (keepTranspose)
   {
      nalu_hypre_ParCSRMatrixDiagT(RT) = R_diag;
   }
   else
   {
      nalu_hypre_CSRMatrixDestroy(R_diag);
   }
   R_diag = NULL;

   if (num_cols_offd_RT)
   {
      if (keepTranspose)
      {
         nalu_hypre_ParCSRMatrixOffdT(RT) = R_offd;
      }
      else
      {
         nalu_hypre_CSRMatrixDestroy(R_offd);
      }
      R_offd = NULL;
   }

   if (num_sends_RT || num_recvs_RT)
   {
      nalu_hypre_CSRMatrixDestroy(RAP_ext);
      RAP_ext = NULL;
   }
   nalu_hypre_TFree(P_mark_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(A_mark_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(P_ext_diag_i, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(P_ext_offd_i, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_cnt_diag, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_cnt_offd, NALU_HYPRE_MEMORY_HOST);
   if (num_cols_offd_P)
   {
      nalu_hypre_TFree(map_P_to_Pext, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(map_P_to_RAP, NALU_HYPRE_MEMORY_HOST);
   }
   if (num_cols_offd_Pext)
   {
      nalu_hypre_TFree(col_map_offd_Pext, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(map_Pext_to_RAP, NALU_HYPRE_MEMORY_HOST);
   }
   if (P_ext_diag_size)
   {
      nalu_hypre_TFree(P_ext_diag_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(P_ext_diag_j, NALU_HYPRE_MEMORY_HOST);
   }
   if (P_ext_offd_size)
   {
      nalu_hypre_TFree(P_ext_offd_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(P_ext_offd_j, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(RA_diag_data_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(RA_diag_j_array, NALU_HYPRE_MEMORY_HOST);
   if (num_cols_offd_A)
   {
      nalu_hypre_TFree(RA_offd_data_array, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(RA_offd_j_array, NALU_HYPRE_MEMORY_HOST);
   }
#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   if (send_map_elmts_RT_inverse_map_initialized)
   {
      nalu_hypre_UnorderedIntMapDestroy(&send_map_elmts_RT_inverse_map);
   }
   nalu_hypre_TFree(send_map_elmts_starts_RT_aggregated, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_map_elmts_RT_aggregated, NALU_HYPRE_MEMORY_HOST);
#endif

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_RAP] += nalu_hypre_MPI_Wtime();
#endif

   return (0);
}
