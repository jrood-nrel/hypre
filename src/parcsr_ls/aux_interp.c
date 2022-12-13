/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "aux_interp.h"

/*---------------------------------------------------------------------------
 * Auxilary routines for the long range interpolation methods.
 *  Implemented: "standard", "extended", "multipass", "FF"
 *--------------------------------------------------------------------------*/
/* AHB 11/06: Modification of the above original - takes two
   communication packages and inserts nodes to position expected for
   OUT_marker

   offd nodes from comm_pkg take up first chunk of CF_marker_offd, offd
   nodes from extend_comm_pkg take up the second chunk of CF_marker_offd. */



NALU_HYPRE_Int hypre_alt_insert_new_nodes(hypre_ParCSRCommPkg *comm_pkg,
                                     hypre_ParCSRCommPkg *extend_comm_pkg,
                                     NALU_HYPRE_Int *IN_marker,
                                     NALU_HYPRE_Int full_off_procNodes,
                                     NALU_HYPRE_Int *OUT_marker)
{
   hypre_ParCSRCommHandle  *comm_handle;

   NALU_HYPRE_Int i, index, shift;

   NALU_HYPRE_Int num_sends, num_recvs;

   NALU_HYPRE_Int *recv_vec_starts;

   NALU_HYPRE_Int e_num_sends;

   NALU_HYPRE_Int *int_buf_data;
   NALU_HYPRE_Int *e_out_marker;


   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   num_recvs =  hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);

   e_num_sends = hypre_ParCSRCommPkgNumSends(extend_comm_pkg);


   index = hypre_max(hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                     hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, e_num_sends));

   int_buf_data = hypre_CTAlloc(NALU_HYPRE_Int,  index, NALU_HYPRE_MEMORY_HOST);

   /* orig commpkg data*/
   index = 0;

   NALU_HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
   NALU_HYPRE_Int end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = begin; i < end; ++i)
   {
      int_buf_data[i - begin] =
         IN_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                               OUT_marker);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   /* now do the extend commpkg */

   /* first we need to shift our position in the OUT_marker */
   shift = recv_vec_starts[num_recvs];
   e_out_marker = OUT_marker + shift;

   index = 0;

   begin = hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, 0);
   end = hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, e_num_sends);
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = begin; i < end; ++i)
   {
      int_buf_data[i - begin] =
         IN_marker[hypre_ParCSRCommPkgSendMapElmt(extend_comm_pkg, i)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, extend_comm_pkg, int_buf_data,
                                               e_out_marker);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

NALU_HYPRE_Int hypre_big_insert_new_nodes(hypre_ParCSRCommPkg *comm_pkg,
                                     hypre_ParCSRCommPkg *extend_comm_pkg,
                                     NALU_HYPRE_Int *IN_marker,
                                     NALU_HYPRE_Int full_off_procNodes,
                                     NALU_HYPRE_BigInt offset,
                                     NALU_HYPRE_BigInt *OUT_marker)
{
   hypre_ParCSRCommHandle  *comm_handle;

   NALU_HYPRE_Int i, index, shift;

   NALU_HYPRE_Int num_sends, num_recvs;

   NALU_HYPRE_Int *recv_vec_starts;

   NALU_HYPRE_Int e_num_sends;

   NALU_HYPRE_BigInt *int_buf_data;
   NALU_HYPRE_BigInt *e_out_marker;


   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   num_recvs =  hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);

   e_num_sends = hypre_ParCSRCommPkgNumSends(extend_comm_pkg);


   index = hypre_max(hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                     hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, e_num_sends));

   int_buf_data = hypre_CTAlloc(NALU_HYPRE_BigInt,  index, NALU_HYPRE_MEMORY_HOST);

   /* orig commpkg data*/
   index = 0;

   NALU_HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
   NALU_HYPRE_Int end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = begin; i < end; ++i)
   {
      int_buf_data[i - begin] = offset +
                                (NALU_HYPRE_BigInt) IN_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 21, comm_pkg, int_buf_data,
                                               OUT_marker);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   /* now do the extend commpkg */

   /* first we need to shift our position in the OUT_marker */
   shift = recv_vec_starts[num_recvs];
   e_out_marker = OUT_marker + shift;

   index = 0;

   begin = hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, 0);
   end = hypre_ParCSRCommPkgSendMapStart(extend_comm_pkg, e_num_sends);
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = begin; i < end; ++i)
   {
      int_buf_data[i - begin] = offset +
                                (NALU_HYPRE_BigInt) IN_marker[hypre_ParCSRCommPkgSendMapElmt(extend_comm_pkg, i)];
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 21, extend_comm_pkg, int_buf_data,
                                               e_out_marker);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/* sort for non-ordered arrays */
NALU_HYPRE_Int hypre_ssort(NALU_HYPRE_BigInt *data, NALU_HYPRE_Int n)
{
   NALU_HYPRE_Int i, si;
   NALU_HYPRE_Int change = 0;

   if (n > 0)
      for (i = n - 1; i > 0; i--)
      {
         si = hypre_index_of_minimum(data, i + 1);
         if (i != si)
         {
            hypre_swap_int(data, i, si);
            change = 1;
         }
      }
   return change;
}

/* Auxilary function for hypre_ssort */
NALU_HYPRE_Int hypre_index_of_minimum(NALU_HYPRE_BigInt *data, NALU_HYPRE_Int n)
{
   NALU_HYPRE_Int answer;
   NALU_HYPRE_Int i;

   answer = 0;
   for (i = 1; i < n; i++)
      if (data[answer] < data[i])
      {
         answer = i;
      }

   return answer;
}

void hypre_swap_int(NALU_HYPRE_BigInt *data, NALU_HYPRE_Int a, NALU_HYPRE_Int b)
{
   NALU_HYPRE_BigInt temp;

   temp = data[a];
   data[a] = data[b];
   data[b] = temp;

   return;
}

/* Initialize CF_marker_offd, CF_marker, P_marker, P_marker_offd, tmp */
void hypre_initialize_vecs(NALU_HYPRE_Int diag_n, NALU_HYPRE_Int offd_n, NALU_HYPRE_Int *diag_ftc,
                           NALU_HYPRE_BigInt *offd_ftc,
                           NALU_HYPRE_Int *diag_pm, NALU_HYPRE_Int *offd_pm, NALU_HYPRE_Int *tmp_CF)
{
   NALU_HYPRE_Int i;

   /* Quicker initialization */
   if (offd_n < diag_n)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < offd_n; i++)
      {
         diag_ftc[i] = -1;
         offd_ftc[i] = -1;
         tmp_CF[i] = -1;
         if (diag_pm != NULL)
         {  diag_pm[i] = -1; }
         if (offd_pm != NULL)
         {  offd_pm[i] = -1;}
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = offd_n; i < diag_n; i++)
      {
         diag_ftc[i] = -1;
         if (diag_pm != NULL)
         {  diag_pm[i] = -1; }
      }
   }
   else
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < diag_n; i++)
      {
         diag_ftc[i] = -1;
         offd_ftc[i] = -1;
         tmp_CF[i] = -1;
         if (diag_pm != NULL)
         {  diag_pm[i] = -1;}
         if (offd_pm != NULL)
         {  offd_pm[i] = -1;}
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = diag_n; i < offd_n; i++)
      {
         offd_ftc[i] = -1;
         tmp_CF[i] = -1;
         if (offd_pm != NULL)
         {  offd_pm[i] = -1;}
      }
   }
   return;
}

/* Find nodes that are offd and are not contained in original offd
 * (neighbors of neighbors) */
static NALU_HYPRE_Int hypre_new_offd_nodes(NALU_HYPRE_BigInt **found, NALU_HYPRE_Int num_cols_A_offd,
                                      NALU_HYPRE_Int *A_ext_i, NALU_HYPRE_BigInt *A_ext_j,
                                      NALU_HYPRE_Int num_cols_S_offd, NALU_HYPRE_BigInt *col_map_offd, NALU_HYPRE_BigInt col_1,
                                      NALU_HYPRE_BigInt col_n, NALU_HYPRE_Int *Sop_i, NALU_HYPRE_BigInt *Sop_j,
                                      NALU_HYPRE_Int *CF_marker_offd)
{
#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX] -= hypre_MPI_Wtime();
#endif

   NALU_HYPRE_BigInt big_i1, big_k1;
   NALU_HYPRE_Int i, j, kk;
   NALU_HYPRE_Int got_loc, loc_col;

   /*NALU_HYPRE_Int min;*/
   NALU_HYPRE_Int newoff = 0;

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   hypre_UnorderedBigIntMap col_map_offd_inverse;
   hypre_UnorderedBigIntMapCreate(&col_map_offd_inverse, 2 * num_cols_A_offd, 16 * hypre_NumThreads());

   #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
   for (i = 0; i < num_cols_A_offd; i++)
   {
      hypre_UnorderedBigIntMapPutIfAbsent(&col_map_offd_inverse, col_map_offd[i], i);
   }

   /* Find nodes that will be added to the off diag list */
   NALU_HYPRE_Int size_offP = A_ext_i[num_cols_A_offd];
   hypre_UnorderedBigIntSet set;
   hypre_UnorderedBigIntSetCreate(&set, size_offP, 16 * hypre_NumThreads());

   #pragma omp parallel private(i,j,big_i1)
   {
      #pragma omp for NALU_HYPRE_SMP_SCHEDULE
      for (i = 0; i < num_cols_A_offd; i++)
      {
         if (CF_marker_offd[i] < 0)
         {
            for (j = A_ext_i[i]; j < A_ext_i[i + 1]; j++)
            {
               big_i1 = A_ext_j[j];
               if (big_i1 < col_1 || big_i1 >= col_n)
               {
                  if (!hypre_UnorderedBigIntSetContains(&set, big_i1))
                  {
                     NALU_HYPRE_Int k = hypre_UnorderedBigIntMapGet(&col_map_offd_inverse, big_i1);
                     if (-1 == k)
                     {
                        hypre_UnorderedBigIntSetPut(&set, big_i1);
                     }
                     else
                     {
                        A_ext_j[j] = -k - 1;
                     }
                  }
               }
            }
            for (j = Sop_i[i]; j < Sop_i[i + 1]; j++)
            {
               big_i1 = Sop_j[j];
               if (big_i1 < col_1 || big_i1 >= col_n)
               {
                  if (!hypre_UnorderedBigIntSetContains(&set, big_i1))
                  {
                     NALU_HYPRE_Int k = hypre_UnorderedBigIntMapGet(&col_map_offd_inverse, big_i1);
                     if (-1 == k)
                     {
                        hypre_UnorderedBigIntSetPut(&set, big_i1);
                     }
                     else
                     {
                        Sop_j[j] = -k - 1;
                     }
                  }
               }
            }
         } /* CF_marker_offd[i] < 0 */
      } /* for each row */
   } /* omp parallel */

   hypre_UnorderedBigIntMapDestroy(&col_map_offd_inverse);
   NALU_HYPRE_BigInt *tmp_found = hypre_UnorderedBigIntSetCopyToArray(&set, &newoff);
   hypre_UnorderedBigIntSetDestroy(&set);

   /* Put found in monotone increasing order */
#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_MERGE] -= hypre_MPI_Wtime();
#endif

   hypre_UnorderedBigIntMap tmp_found_inverse;
   if (newoff > 0)
   {
      hypre_big_sort_and_create_inverse_map(tmp_found, newoff, &tmp_found, &tmp_found_inverse);
   }

#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_MERGE] += hypre_MPI_Wtime();
#endif

   /* Set column indices for Sop and A_ext such that offd nodes are
    * negatively indexed */
   #pragma omp parallel for private(kk,big_k1,got_loc,loc_col) NALU_HYPRE_SMP_SCHEDULE
   for (i = 0; i < num_cols_A_offd; i++)
   {
      if (CF_marker_offd[i] < 0)
      {
         for (kk = Sop_i[i]; kk < Sop_i[i + 1]; kk++)
         {
            big_k1 = Sop_j[kk];
            if (big_k1 > -1 && (big_k1 < col_1 || big_k1 >= col_n))
            {
               got_loc = hypre_UnorderedBigIntMapGet(&tmp_found_inverse, big_k1);
               loc_col = got_loc + num_cols_A_offd;
               Sop_j[kk] = (NALU_HYPRE_BigInt)(-loc_col - 1);
            }
         }
         for (kk = A_ext_i[i]; kk < A_ext_i[i + 1]; kk++)
         {
            big_k1 = A_ext_j[kk];
            if (big_k1 > -1 && (big_k1 < col_1 || big_k1 >= col_n))
            {
               got_loc = hypre_UnorderedBigIntMapGet(&tmp_found_inverse, big_k1);
               loc_col = got_loc + num_cols_A_offd;
               A_ext_j[kk] = (NALU_HYPRE_BigInt)(-loc_col - 1);
            }
         }
      }
   }
   if (newoff)
   {
      hypre_UnorderedBigIntMapDestroy(&tmp_found_inverse);
   }
#else /* !NALU_HYPRE_CONCURRENT_HOPSCOTCH */
   NALU_HYPRE_Int size_offP;

   NALU_HYPRE_BigInt *tmp_found;
   NALU_HYPRE_Int min;
   NALU_HYPRE_Int ifound;

   size_offP = A_ext_i[num_cols_A_offd] + Sop_i[num_cols_A_offd];
   tmp_found = hypre_CTAlloc(NALU_HYPRE_BigInt, size_offP, NALU_HYPRE_MEMORY_HOST);

   /* Find nodes that will be added to the off diag list */
   for (i = 0; i < num_cols_A_offd; i++)
   {
      if (CF_marker_offd[i] < 0)
      {
         for (j = A_ext_i[i]; j < A_ext_i[i + 1]; j++)
         {
            big_i1 = A_ext_j[j];
            if (big_i1 < col_1 || big_i1 >= col_n)
            {
               ifound = hypre_BigBinarySearch(col_map_offd, big_i1, num_cols_A_offd);
               if (ifound == -1)
               {
                  tmp_found[newoff] = big_i1;
                  newoff++;
               }
               else
               {
                  A_ext_j[j] = (NALU_HYPRE_BigInt)(-ifound - 1);
               }
            }
         }
         for (j = Sop_i[i]; j < Sop_i[i + 1]; j++)
         {
            big_i1 = Sop_j[j];
            if (big_i1 < col_1 || big_i1 >= col_n)
            {
               ifound = hypre_BigBinarySearch(col_map_offd, big_i1, num_cols_A_offd);
               if (ifound == -1)
               {
                  tmp_found[newoff] = big_i1;
                  newoff++;
               }
               else
               {
                  Sop_j[j] = (NALU_HYPRE_BigInt)(-ifound - 1);
               }
            }
         }
      }
   }
   /* Put found in monotone increasing order */
   if (newoff > 0)
   {
      hypre_BigQsort0(tmp_found, 0, newoff - 1);
      ifound = tmp_found[0];
      min = 1;
      for (i = 1; i < newoff; i++)
      {
         if (tmp_found[i] > ifound)
         {
            ifound = tmp_found[i];
            tmp_found[min++] = ifound;
         }
      }
      newoff = min;
   }

   /* Set column indices for Sop and A_ext such that offd nodes are
    * negatively indexed */
   for (i = 0; i < num_cols_A_offd; i++)
   {
      if (CF_marker_offd[i] < 0)
      {
         for (kk = Sop_i[i]; kk < Sop_i[i + 1]; kk++)
         {
            big_k1 = Sop_j[kk];
            if (big_k1 > -1 && (big_k1 < col_1 || big_k1 >= col_n))
            {
               got_loc = hypre_BigBinarySearch(tmp_found, big_k1, newoff);
               if (got_loc > -1)
               {
                  loc_col = got_loc + num_cols_A_offd;
               }
               Sop_j[kk] = (NALU_HYPRE_BigInt)(-loc_col - 1);
            }
         }
         for (kk = A_ext_i[i]; kk < A_ext_i[i + 1]; kk++)
         {
            big_k1 = A_ext_j[kk];
            if (big_k1 > -1 && (big_k1 < col_1 || big_k1 >= col_n))
            {
               got_loc = hypre_BigBinarySearch(tmp_found, big_k1, newoff);
               loc_col = got_loc + num_cols_A_offd;
               A_ext_j[kk] = (NALU_HYPRE_BigInt)(-loc_col - 1);
            }
         }
      }
   }
#endif /* !NALU_HYPRE_CONCURRENT_HOPSCOTCH */

   *found = tmp_found;

#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX] += hypre_MPI_Wtime();
#endif

   return newoff;
}

NALU_HYPRE_Int hypre_exchange_marker(hypre_ParCSRCommPkg *comm_pkg,
                                NALU_HYPRE_Int *IN_marker,
                                NALU_HYPRE_Int *OUT_marker)
{
   NALU_HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
   NALU_HYPRE_Int end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   NALU_HYPRE_Int *int_buf_data = hypre_CTAlloc(NALU_HYPRE_Int, end, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_Int i;
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = begin; i < end; ++i)
   {
      int_buf_data[i - begin] =
         IN_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
   }

   hypre_ParCSRCommHandle *comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, int_buf_data,
                                                                       OUT_marker);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

NALU_HYPRE_Int hypre_exchange_interp_data(
   NALU_HYPRE_Int **CF_marker_offd,
   NALU_HYPRE_Int **dof_func_offd,
   hypre_CSRMatrix **A_ext,
   NALU_HYPRE_Int *full_off_procNodes,
   hypre_CSRMatrix **Sop,
   hypre_ParCSRCommPkg **extend_comm_pkg,
   hypre_ParCSRMatrix *A,
   NALU_HYPRE_Int *CF_marker,
   hypre_ParCSRMatrix *S,
   NALU_HYPRE_Int num_functions,
   NALU_HYPRE_Int *dof_func,
   NALU_HYPRE_Int skip_fine_or_same_sign) // skip_fine_or_same_sign if we want to skip fine points in S and nnz with the same sign as diagonal in A
{
#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_EXCHANGE_INTERP_DATA] -= hypre_MPI_Wtime();
#endif

   hypre_ParCSRCommPkg   *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int              num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt          *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
   NALU_HYPRE_BigInt           col_1 = hypre_ParCSRMatrixFirstRowIndex(A);
   NALU_HYPRE_Int              local_numrows = hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_BigInt           col_n = col_1 + (NALU_HYPRE_BigInt)local_numrows;
   NALU_HYPRE_BigInt          *found = NULL;

   /*----------------------------------------------------------------------
    * Get the off processors rows for A and S, associated with columns in
    * A_offd and S_offd.
    *---------------------------------------------------------------------*/
   *CF_marker_offd = hypre_TAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
   hypre_exchange_marker(comm_pkg, CF_marker, *CF_marker_offd);

   hypre_ParCSRCommHandle *comm_handle_a_idx, *comm_handle_a_data;
   *A_ext         = hypre_ParCSRMatrixExtractBExt_Overlap(A, A, 1, &comm_handle_a_idx,
                                                          &comm_handle_a_data,
                                                          CF_marker, *CF_marker_offd, skip_fine_or_same_sign, skip_fine_or_same_sign);
   NALU_HYPRE_Int *A_ext_i        = hypre_CSRMatrixI(*A_ext);
   NALU_HYPRE_BigInt *A_ext_j        = hypre_CSRMatrixBigJ(*A_ext);
   NALU_HYPRE_Int  A_ext_rows     = hypre_CSRMatrixNumRows(*A_ext);

   hypre_ParCSRCommHandle *comm_handle_s_idx;
   *Sop           = hypre_ParCSRMatrixExtractBExt_Overlap(S, A, 0, &comm_handle_s_idx, NULL, CF_marker,
                                                          *CF_marker_offd, skip_fine_or_same_sign, 0);
   NALU_HYPRE_Int *Sop_i          = hypre_CSRMatrixI(*Sop);
   NALU_HYPRE_BigInt *Sop_j       = hypre_CSRMatrixBigJ(*Sop);
   NALU_HYPRE_Int  Soprows        = hypre_CSRMatrixNumRows(*Sop);

   NALU_HYPRE_Int *send_idx = (NALU_HYPRE_Int *)comm_handle_s_idx->send_data;
   hypre_ParCSRCommHandleDestroy(comm_handle_s_idx);
   hypre_TFree(send_idx, NALU_HYPRE_MEMORY_HOST);

   send_idx = (NALU_HYPRE_Int *)comm_handle_a_idx->send_data;
   hypre_ParCSRCommHandleDestroy(comm_handle_a_idx);
   hypre_TFree(send_idx, NALU_HYPRE_MEMORY_HOST);

   /* Find nodes that are neighbors of neighbors, not found in offd */
#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_EXCHANGE_INTERP_DATA] += hypre_MPI_Wtime();
#endif
   NALU_HYPRE_Int newoff = hypre_new_offd_nodes(&found, A_ext_rows, A_ext_i, A_ext_j,
                                           Soprows, col_map_offd, col_1, col_n,
                                           Sop_i, Sop_j, *CF_marker_offd);
#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_EXCHANGE_INTERP_DATA] -= hypre_MPI_Wtime();
#endif
   if (newoff >= 0)
   {
      *full_off_procNodes = newoff + num_cols_A_offd;
   }
   else
   {
      return hypre_error_flag;
   }

   /* Possibly add new points and new processors to the comm_pkg, all
    * processors need new_comm_pkg */

   /* AHB - create a new comm package just for extended info -
      this will work better with the assumed partition*/
   hypre_ParCSRFindExtendCommPkg(hypre_ParCSRMatrixComm(A),
                                 hypre_ParCSRMatrixGlobalNumCols(A),
                                 hypre_ParCSRMatrixFirstColDiag(A),
                                 hypre_CSRMatrixNumCols(A_diag),
                                 hypre_ParCSRMatrixColStarts(A),
                                 hypre_ParCSRMatrixAssumedPartition(A),
                                 newoff,
                                 found,
                                 extend_comm_pkg);

   *CF_marker_offd = hypre_TReAlloc(*CF_marker_offd, NALU_HYPRE_Int, *full_off_procNodes,
                                    NALU_HYPRE_MEMORY_HOST);
   hypre_exchange_marker(*extend_comm_pkg, CF_marker, *CF_marker_offd + A_ext_rows);

   if (num_functions > 1)
   {
      if (*full_off_procNodes > 0)
      {
         *dof_func_offd = hypre_CTAlloc(NALU_HYPRE_Int, *full_off_procNodes, NALU_HYPRE_MEMORY_HOST);
      }

      hypre_alt_insert_new_nodes(comm_pkg, *extend_comm_pkg, dof_func,
                                 *full_off_procNodes, *dof_func_offd);
   }

   hypre_TFree(found, NALU_HYPRE_MEMORY_HOST);

   NALU_HYPRE_Real *send_data = (NALU_HYPRE_Real *)comm_handle_a_data->send_data;
   hypre_ParCSRCommHandleDestroy(comm_handle_a_data);
   hypre_TFree(send_data, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_EXCHANGE_INTERP_DATA] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

void hypre_build_interp_colmap(hypre_ParCSRMatrix *P, NALU_HYPRE_Int full_off_procNodes,
                               NALU_HYPRE_Int *tmp_CF_marker_offd, NALU_HYPRE_BigInt *fine_to_coarse_offd)
{
#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX] -= hypre_MPI_Wtime();
#endif
   NALU_HYPRE_Int     n_fine = hypre_CSRMatrixNumRows(P->diag);

   NALU_HYPRE_Int     P_offd_size = P->offd->i[n_fine];
   NALU_HYPRE_Int    *P_offd_j = P->offd->j;
   NALU_HYPRE_BigInt *col_map_offd_P = NULL;
   NALU_HYPRE_Int    *P_marker = NULL;
   NALU_HYPRE_Int    *prefix_sum_workspace;
   NALU_HYPRE_Int     num_cols_P_offd = 0;
   NALU_HYPRE_Int     i, index;

   if (full_off_procNodes)
   {
      P_marker = hypre_TAlloc(NALU_HYPRE_Int, full_off_procNodes, NALU_HYPRE_MEMORY_HOST);
   }
   prefix_sum_workspace = hypre_TAlloc(NALU_HYPRE_Int, hypre_NumThreads() + 1, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < full_off_procNodes; i++)
   {
      P_marker[i] = 0;
   }

   /* These two loops set P_marker[i] to 1 if it appears in P_offd_j and if
    * tmp_CF_marker_offd has i marked. num_cols_P_offd is then set to the
    * total number of times P_marker is set */
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,index) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < P_offd_size; i++)
   {
      index = P_offd_j[i];
      if (tmp_CF_marker_offd[index] >= 0)
      {
         P_marker[index] = 1;
      }
   }

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel private(i)
#endif
   {
      NALU_HYPRE_Int i_begin, i_end;
      hypre_GetSimpleThreadPartition(&i_begin, &i_end, full_off_procNodes);

      NALU_HYPRE_Int local_num_cols_P_offd = 0;
      for (i = i_begin; i < i_end; i++)
      {
         if (P_marker[i] == 1) { local_num_cols_P_offd++; }
      }

      hypre_prefix_sum(&local_num_cols_P_offd, &num_cols_P_offd, prefix_sum_workspace);

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp master
#endif
      {
         if (num_cols_P_offd)
         {
            col_map_offd_P = hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_P_offd, NALU_HYPRE_MEMORY_HOST);
         }
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      for (i = i_begin; i < i_end; i++)
      {
         if (P_marker[i] == 1)
         {
            col_map_offd_P[local_num_cols_P_offd++] = fine_to_coarse_offd[i];
         }
      }
   }

   hypre_UnorderedBigIntMap col_map_offd_P_inverse;
   hypre_big_sort_and_create_inverse_map(col_map_offd_P, num_cols_P_offd, &col_map_offd_P,
                                         &col_map_offd_P_inverse);

   // find old idx -> new idx map
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for
#endif
   for (i = 0; i < full_off_procNodes; i++)
   {
      P_marker[i] = hypre_UnorderedBigIntMapGet(&col_map_offd_P_inverse, fine_to_coarse_offd[i]);
   }

   if (num_cols_P_offd)
   {
      hypre_UnorderedBigIntMapDestroy(&col_map_offd_P_inverse);
   }

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for
#endif
   for (i = 0; i < P_offd_size; i++)
   {
      P_offd_j[i] = P_marker[P_offd_j[i]];
   }

   hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(prefix_sum_workspace, NALU_HYPRE_MEMORY_HOST);

   if (num_cols_P_offd)
   {
      hypre_ParCSRMatrixColMapOffd(P) = col_map_offd_P;
      hypre_CSRMatrixNumCols(P->offd) = num_cols_P_offd;
   }

#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX] += hypre_MPI_Wtime();
#endif
}
