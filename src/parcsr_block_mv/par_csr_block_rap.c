/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <NALU_HYPRE_config.h>
#include "_nalu_hypre_utilities.h"
#include "par_csr_block_matrix.h"
#include "../parcsr_mv/_nalu_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * used in RAP function - block size must be an argument because RAP_int may
 * by NULL
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRBlockMatrix *
nalu_hypre_ExchangeRAPBlockData(nalu_hypre_CSRBlockMatrix *RAP_int,
                           nalu_hypre_ParCSRCommPkg *comm_pkg_RT, NALU_HYPRE_Int block_size)
{
   NALU_HYPRE_Int     *RAP_int_i;
   NALU_HYPRE_BigInt  *RAP_int_j = NULL;
   NALU_HYPRE_Complex *RAP_int_data = NULL;
   NALU_HYPRE_Int     num_cols = 0;

   MPI_Comm comm = nalu_hypre_ParCSRCommPkgComm(comm_pkg_RT);
   NALU_HYPRE_Int num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg_RT);
   NALU_HYPRE_Int *recv_procs = nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg_RT);
   NALU_HYPRE_Int *recv_vec_starts = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_RT);
   NALU_HYPRE_Int num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg_RT);
   NALU_HYPRE_Int *send_procs = nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg_RT);
   NALU_HYPRE_Int *send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg_RT);

   /*   NALU_HYPRE_Int block_size = nalu_hypre_CSRBlockMatrixBlockSize(RAP_int); */

   nalu_hypre_CSRBlockMatrix *RAP_ext;

   NALU_HYPRE_Int     *RAP_ext_i;
   NALU_HYPRE_BigInt  *RAP_ext_j = NULL;
   NALU_HYPRE_Complex *RAP_ext_data = NULL;

   nalu_hypre_ParCSRCommHandle *comm_handle = NULL;
   nalu_hypre_ParCSRCommPkg *tmp_comm_pkg = NULL;

   NALU_HYPRE_Int *jdata_recv_vec_starts;
   NALU_HYPRE_Int *jdata_send_map_starts;

   NALU_HYPRE_Int num_rows;
   NALU_HYPRE_Int num_nonzeros;
   NALU_HYPRE_Int i, j, bnnz;
   NALU_HYPRE_Int num_procs, my_id;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   bnnz = block_size * block_size;

   RAP_ext_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  send_map_starts[num_sends] + 1, NALU_HYPRE_MEMORY_HOST);
   jdata_recv_vec_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs + 1, NALU_HYPRE_MEMORY_HOST);
   jdata_send_map_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends + 1, NALU_HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------------
    * recompute RAP_int_i so that RAP_int_i[j+1] contains the number of
    * elements of row j (to be determined through send_map_elmnts on the
    * receiving end)
    *--------------------------------------------------------------------------*/

   if (num_recvs)
   {
      RAP_int_i = nalu_hypre_CSRBlockMatrixI(RAP_int);
      RAP_int_j = nalu_hypre_CSRBlockMatrixBigJ(RAP_int);
      RAP_int_data = nalu_hypre_CSRBlockMatrixData(RAP_int);
      num_cols = nalu_hypre_CSRBlockMatrixNumCols(RAP_int);
   }
   jdata_recv_vec_starts[0] = 0;
   for (i = 0; i < num_recvs; i++)
   {
      jdata_recv_vec_starts[i + 1] = RAP_int_i[recv_vec_starts[i + 1]];
   }

   for (i = num_recvs; i > 0; i--)
      for (j = recv_vec_starts[i]; j > recv_vec_starts[i - 1]; j--)
      {
         RAP_int_i[j] -= RAP_int_i[j - 1];
      }

   /*--------------------------------------------------------------------------
    * initialize communication
    *--------------------------------------------------------------------------*/

   if (num_recvs && num_sends)
   {
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(12, comm_pkg_RT,
                                                 &RAP_int_i[1], &RAP_ext_i[1]);
   }
   else if (num_recvs)
   {
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(12, comm_pkg_RT,
                                                 &RAP_int_i[1], NULL);
   }
   else if (num_sends)
   {
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(12, comm_pkg_RT,
                                                 NULL, &RAP_ext_i[1]);
   }

   /* Create temporary communication package - note: send and recv are reversed */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_sends, send_procs, jdata_send_map_starts,
                                    num_recvs, recv_procs, jdata_recv_vec_starts,
                                    NULL, &tmp_comm_pkg);

   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   /*--------------------------------------------------------------------------
    * compute num_nonzeros for RAP_ext
    *--------------------------------------------------------------------------*/

   for (i = 0; i < num_sends; i++)
   {
      for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
      {
         RAP_ext_i[j + 1] += RAP_ext_i[j];
      }
   }

   num_rows = send_map_starts[num_sends];
   num_nonzeros = RAP_ext_i[num_rows];
   if (num_nonzeros)
   {
      RAP_ext_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_nonzeros, NALU_HYPRE_MEMORY_HOST);
      RAP_ext_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  num_nonzeros * bnnz, NALU_HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < num_sends + 1; i++)
   {
      jdata_send_map_starts[i] = RAP_ext_i[send_map_starts[i]];
   }

   comm_handle = nalu_hypre_ParCSRBlockCommHandleCreate(1, bnnz, tmp_comm_pkg,
                                                   (void *) RAP_int_data, (void *) RAP_ext_data);
   nalu_hypre_ParCSRBlockCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   comm_handle = nalu_hypre_ParCSRCommHandleCreate(21, tmp_comm_pkg, RAP_int_j,
                                              RAP_ext_j);
   RAP_ext = nalu_hypre_CSRBlockMatrixCreate(block_size, num_rows, num_cols,
                                        num_nonzeros);

   nalu_hypre_CSRBlockMatrixI(RAP_ext) = RAP_ext_i;
   if (num_nonzeros)
   {
      nalu_hypre_CSRBlockMatrixBigJ(RAP_ext) = RAP_ext_j;
      nalu_hypre_CSRBlockMatrixData(RAP_ext) = RAP_ext_data;
   }

   /* Free memory */
   nalu_hypre_TFree(jdata_recv_vec_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jdata_send_map_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(tmp_comm_pkg, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   return RAP_ext;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_BoomerAMGBuildCoarseOperator
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixRAP(nalu_hypre_ParCSRBlockMatrix  *RT,
                           nalu_hypre_ParCSRBlockMatrix  *A,
                           nalu_hypre_ParCSRBlockMatrix  *P,
                           nalu_hypre_ParCSRBlockMatrix **RAP_ptr )

{
   MPI_Comm        comm = nalu_hypre_ParCSRBlockMatrixComm(A);

   nalu_hypre_CSRBlockMatrix *RT_diag = nalu_hypre_ParCSRBlockMatrixDiag(RT);
   nalu_hypre_CSRBlockMatrix *RT_offd = nalu_hypre_ParCSRBlockMatrixOffd(RT);
   NALU_HYPRE_Int             num_cols_offd_RT = nalu_hypre_CSRBlockMatrixNumCols(RT_offd);
   NALU_HYPRE_Int             num_rows_offd_RT = nalu_hypre_CSRBlockMatrixNumRows(RT_offd);
   nalu_hypre_ParCSRCommPkg   *comm_pkg_RT = nalu_hypre_ParCSRBlockMatrixCommPkg(RT);
   NALU_HYPRE_Int             num_recvs_RT = 0;
   NALU_HYPRE_Int             num_sends_RT = 0;
   NALU_HYPRE_Int             *send_map_starts_RT;
   NALU_HYPRE_Int             *send_map_elmts_RT;

   nalu_hypre_CSRBlockMatrix *A_diag = nalu_hypre_ParCSRBlockMatrixDiag(A);

   NALU_HYPRE_Complex         *A_diag_data = nalu_hypre_CSRBlockMatrixData(A_diag);
   NALU_HYPRE_Int             *A_diag_i = nalu_hypre_CSRBlockMatrixI(A_diag);
   NALU_HYPRE_Int             *A_diag_j = nalu_hypre_CSRBlockMatrixJ(A_diag);
   NALU_HYPRE_Int             block_size = nalu_hypre_CSRBlockMatrixBlockSize(A_diag);

   nalu_hypre_CSRBlockMatrix *A_offd = nalu_hypre_ParCSRBlockMatrixOffd(A);

   NALU_HYPRE_Complex         *A_offd_data = nalu_hypre_CSRBlockMatrixData(A_offd);
   NALU_HYPRE_Int             *A_offd_i = nalu_hypre_CSRBlockMatrixI(A_offd);
   NALU_HYPRE_Int             *A_offd_j = nalu_hypre_CSRBlockMatrixJ(A_offd);

   NALU_HYPRE_Int  num_cols_diag_A = nalu_hypre_CSRBlockMatrixNumCols(A_diag);
   NALU_HYPRE_Int  num_cols_offd_A = nalu_hypre_CSRBlockMatrixNumCols(A_offd);

   nalu_hypre_CSRBlockMatrix *P_diag = nalu_hypre_ParCSRBlockMatrixDiag(P);

   NALU_HYPRE_Complex         *P_diag_data = nalu_hypre_CSRBlockMatrixData(P_diag);
   NALU_HYPRE_Int             *P_diag_i = nalu_hypre_CSRBlockMatrixI(P_diag);
   NALU_HYPRE_Int             *P_diag_j = nalu_hypre_CSRBlockMatrixJ(P_diag);

   nalu_hypre_CSRBlockMatrix *P_offd = nalu_hypre_ParCSRBlockMatrixOffd(P);
   NALU_HYPRE_BigInt          *col_map_offd_P = nalu_hypre_ParCSRBlockMatrixColMapOffd(P);

   NALU_HYPRE_Complex         *P_offd_data = nalu_hypre_CSRBlockMatrixData(P_offd);
   NALU_HYPRE_Int             *P_offd_i = nalu_hypre_CSRBlockMatrixI(P_offd);
   NALU_HYPRE_Int             *P_offd_j = nalu_hypre_CSRBlockMatrixJ(P_offd);

   NALU_HYPRE_BigInt  first_col_diag_P = nalu_hypre_ParCSRBlockMatrixFirstColDiag(P);
   NALU_HYPRE_BigInt  last_col_diag_P;
   NALU_HYPRE_Int  num_cols_diag_P = nalu_hypre_CSRBlockMatrixNumCols(P_diag);
   NALU_HYPRE_Int  num_cols_offd_P = nalu_hypre_CSRBlockMatrixNumCols(P_offd);
   NALU_HYPRE_BigInt *coarse_partitioning = nalu_hypre_ParCSRBlockMatrixColStarts(P);
   NALU_HYPRE_BigInt row_starts[2], col_starts[2];

   nalu_hypre_ParCSRBlockMatrix *RAP;
   NALU_HYPRE_BigInt            *col_map_offd_RAP;

   nalu_hypre_CSRBlockMatrix *RAP_int = NULL;

   NALU_HYPRE_Complex         *RAP_int_data;
   NALU_HYPRE_Int             *RAP_int_i;
   NALU_HYPRE_BigInt          *RAP_int_j;

   nalu_hypre_CSRBlockMatrix *RAP_ext;

   NALU_HYPRE_Complex         *RAP_ext_data;
   NALU_HYPRE_Int             *RAP_ext_i;
   NALU_HYPRE_BigInt          *RAP_ext_j;

   nalu_hypre_CSRBlockMatrix *RAP_diag;

   NALU_HYPRE_Complex         *RAP_diag_data;
   NALU_HYPRE_Int             *RAP_diag_i;
   NALU_HYPRE_Int             *RAP_diag_j;

   nalu_hypre_CSRBlockMatrix *RAP_offd;

   NALU_HYPRE_Complex         *RAP_offd_data;
   NALU_HYPRE_Int             *RAP_offd_i;
   NALU_HYPRE_Int             *RAP_offd_j;

   NALU_HYPRE_Int              RAP_size;
   NALU_HYPRE_Int              RAP_ext_size;
   NALU_HYPRE_Int              RAP_diag_size;
   NALU_HYPRE_Int              RAP_offd_size;
   NALU_HYPRE_Int              P_ext_diag_size;
   NALU_HYPRE_Int              P_ext_offd_size;
   NALU_HYPRE_Int              first_col_diag_RAP;
   NALU_HYPRE_Int              last_col_diag_RAP;
   NALU_HYPRE_Int              num_cols_offd_RAP = 0;

   nalu_hypre_CSRBlockMatrix *R_diag;

   NALU_HYPRE_Complex         *R_diag_data;
   NALU_HYPRE_Int             *R_diag_i;
   NALU_HYPRE_Int             *R_diag_j;

   nalu_hypre_CSRBlockMatrix *R_offd;

   NALU_HYPRE_Complex         *R_offd_data;
   NALU_HYPRE_Int             *R_offd_i;
   NALU_HYPRE_Int             *R_offd_j;

   nalu_hypre_CSRBlockMatrix *Ps_ext;

   NALU_HYPRE_Complex         *Ps_ext_data;
   NALU_HYPRE_Int             *Ps_ext_i;
   NALU_HYPRE_BigInt          *Ps_ext_j;

   NALU_HYPRE_Complex         *P_ext_diag_data;
   NALU_HYPRE_Int             *P_ext_diag_i;
   NALU_HYPRE_Int             *P_ext_diag_j;

   NALU_HYPRE_Complex         *P_ext_offd_data;
   NALU_HYPRE_Int             *P_ext_offd_i;
   NALU_HYPRE_Int             *P_ext_offd_j;

   NALU_HYPRE_BigInt          *col_map_offd_Pext;
   NALU_HYPRE_Int             *map_P_to_Pext;
   NALU_HYPRE_Int             *map_P_to_RAP;
   NALU_HYPRE_Int             *map_Pext_to_RAP;

   NALU_HYPRE_Int             *P_marker;
   NALU_HYPRE_Int            **P_mark_array;
   NALU_HYPRE_Int            **A_mark_array;
   NALU_HYPRE_Int             *A_marker;
   NALU_HYPRE_BigInt          *temp = NULL;

   NALU_HYPRE_Int              n_coarse;
   NALU_HYPRE_Int              num_cols_offd_Pext = 0;

   NALU_HYPRE_Int              ic, i, j, k, bnnz, kk;
   NALU_HYPRE_Int              i1, i2, i3, ii, ns, ne, size, rest;
   NALU_HYPRE_Int              cnt, cnt_offd, cnt_diag;
   NALU_HYPRE_Int              jj1, jj2, jj3, jcol;
   NALU_HYPRE_BigInt           value;

   NALU_HYPRE_Int             *jj_count, *jj_cnt_diag, *jj_cnt_offd;
   NALU_HYPRE_Int              jj_counter, jj_count_diag, jj_count_offd;
   NALU_HYPRE_Int              jj_row_begining, jj_row_begin_diag, jj_row_begin_offd;
   NALU_HYPRE_Int              start_indexing = 0; /* start indexing for RAP_data at 0 */
   NALU_HYPRE_Int              num_nz_cols_A;
   NALU_HYPRE_Int              num_procs;
   NALU_HYPRE_Int              num_threads, ind;

   NALU_HYPRE_Complex          *r_entries;
   NALU_HYPRE_Complex          *r_a_products;
   NALU_HYPRE_Complex          *r_a_p_products;

   NALU_HYPRE_Complex          zero = 0.0;

   /*-----------------------------------------------------------------------
    *  Copy ParCSRBlockMatrix RT into CSRBlockMatrix R so that we have
    *  row-wise access to restriction .
    *-----------------------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   /* num_threads = nalu_hypre_NumThreads(); */
   num_threads = 1;

   bnnz = block_size * block_size;
   r_a_products = nalu_hypre_TAlloc(NALU_HYPRE_Complex, bnnz, NALU_HYPRE_MEMORY_HOST);
   r_a_p_products = nalu_hypre_TAlloc(NALU_HYPRE_Complex, bnnz, NALU_HYPRE_MEMORY_HOST);

   if (comm_pkg_RT)
   {
      num_recvs_RT = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg_RT);
      num_sends_RT = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg_RT);
      send_map_starts_RT = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg_RT);
      send_map_elmts_RT = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg_RT);
   }

   nalu_hypre_CSRBlockMatrixTranspose(RT_diag, &R_diag, 1);
   if (num_cols_offd_RT)
   {
      nalu_hypre_CSRBlockMatrixTranspose(RT_offd, &R_offd, 1);
      R_offd_data = nalu_hypre_CSRBlockMatrixData(R_offd);
      R_offd_i    = nalu_hypre_CSRBlockMatrixI(R_offd);
      R_offd_j    = nalu_hypre_CSRBlockMatrixJ(R_offd);
   }

   /*-----------------------------------------------------------------------
    *  Access the CSR vectors for R. Also get sizes of fine and
    *  coarse grids.
    *-----------------------------------------------------------------------*/

   R_diag_data = nalu_hypre_CSRBlockMatrixData(R_diag);
   R_diag_i    = nalu_hypre_CSRBlockMatrixI(R_diag);
   R_diag_j    = nalu_hypre_CSRBlockMatrixJ(R_diag);

   n_coarse = nalu_hypre_ParCSRBlockMatrixGlobalNumCols(P);
   num_nz_cols_A = num_cols_diag_A + num_cols_offd_A;

   /*-----------------------------------------------------------------------
    *  Generate Ps_ext, i.e. portion of P that is stored on neighbor procs
    *  and needed locally for triple matrix product
    *-----------------------------------------------------------------------*/

   if (num_procs > 1)
   {
      Ps_ext = nalu_hypre_ParCSRBlockMatrixExtractBExt(P, A, 1);
      Ps_ext_data = nalu_hypre_CSRBlockMatrixData(Ps_ext);
      Ps_ext_i    = nalu_hypre_CSRBlockMatrixI(Ps_ext);
      Ps_ext_j    = nalu_hypre_CSRBlockMatrixBigJ(Ps_ext);
   }

   P_ext_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_A + 1, NALU_HYPRE_MEMORY_HOST);
   P_ext_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_A + 1, NALU_HYPRE_MEMORY_HOST);
   P_ext_diag_size = 0;
   P_ext_offd_size = 0;
   last_col_diag_P = first_col_diag_P + (NALU_HYPRE_BigInt)num_cols_diag_P - 1;

   for (i = 0; i < num_cols_offd_A; i++)
   {
      for (j = Ps_ext_i[i]; j < Ps_ext_i[i + 1]; j++)
         if (Ps_ext_j[j] < first_col_diag_P || Ps_ext_j[j] > last_col_diag_P)
         {
            P_ext_offd_size++;
         }
         else
         {
            P_ext_diag_size++;
         }
      P_ext_diag_i[i + 1] = P_ext_diag_size;
      P_ext_offd_i[i + 1] = P_ext_offd_size;
   }

   if (P_ext_diag_size)
   {
      P_ext_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_ext_diag_size, NALU_HYPRE_MEMORY_HOST);
      P_ext_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  P_ext_diag_size * bnnz, NALU_HYPRE_MEMORY_HOST);
   }
   if (P_ext_offd_size)
   {
      P_ext_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  P_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
      P_ext_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  P_ext_offd_size * bnnz, NALU_HYPRE_MEMORY_HOST);
   }

   cnt_offd = 0;
   cnt_diag = 0;
   cnt = 0;
   for (i = 0; i < num_cols_offd_A; i++)
   {
      for (j = Ps_ext_i[i]; j < Ps_ext_i[i + 1]; j++)
         if (Ps_ext_j[j] < first_col_diag_P || Ps_ext_j[j] > last_col_diag_P)
         {
            Ps_ext_j[cnt_offd] = Ps_ext_j[j];
            for (kk = 0; kk < bnnz; kk++)
            {
               P_ext_offd_data[cnt_offd * bnnz + kk] = Ps_ext_data[j * bnnz + kk];
            }
            cnt_offd++;
         }
         else
         {
            P_ext_diag_j[cnt_diag] = (NALU_HYPRE_Int)(Ps_ext_j[j] - first_col_diag_P);
            for (kk = 0; kk < bnnz; kk++)
            {
               P_ext_diag_data[cnt_diag * bnnz + kk] = Ps_ext_data[j * bnnz + kk];
            }
            cnt_diag++;
         }
   }
   if (P_ext_offd_size || num_cols_offd_P)
   {
      temp = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  P_ext_offd_size + num_cols_offd_P, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < P_ext_offd_size; i++)
      {
         temp[i] = Ps_ext_j[i];
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
      value = temp[0];
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

   for (i = 0 ; i < P_ext_offd_size; i++)
      P_ext_offd_j[i] = nalu_hypre_BigBinarySearch(col_map_offd_Pext,
                                              Ps_ext_j[i],
                                              num_cols_offd_Pext);
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

   if (num_procs > 1)
   {
      nalu_hypre_CSRBlockMatrixDestroy(Ps_ext);
      Ps_ext = NULL;
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of RAP_int and set up RAP_int_i if there
    *  are more than one processor and nonzero elements in R_offd
    *-----------------------------------------------------------------------*/

   P_mark_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  num_threads, NALU_HYPRE_MEMORY_HOST);
   A_mark_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int *,  num_threads, NALU_HYPRE_MEMORY_HOST);

   if (num_cols_offd_RT)
   {
      jj_count = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_threads, NALU_HYPRE_MEMORY_HOST);

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

         /*--------------------------------------------------------------------
          *  Allocate marker arrays.
          *--------------------------------------------------------------------*/

         if (num_cols_offd_Pext || num_cols_diag_P)
         {
            P_mark_array[ii] = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_diag_P + num_cols_offd_Pext,
                                             NALU_HYPRE_MEMORY_HOST);
            P_marker = P_mark_array[ii];
         }
         A_mark_array[ii] = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_nz_cols_A, NALU_HYPRE_MEMORY_HOST);
         A_marker = A_mark_array[ii];

         /*--------------------------------------------------------------------
          *  Initialize some stuff.
          *--------------------------------------------------------------------*/

         jj_counter = start_indexing;
         for (ic = 0; ic < num_cols_diag_P + num_cols_offd_Pext; ic++)
         {
            P_marker[ic] = -1;
         }
         for (i = 0; i < num_nz_cols_A; i++)
         {
            A_marker[i] = -1;
         }

         /*--------------------------------------------------------------------
          *  Loop over exterior c-points
          *--------------------------------------------------------------------*/

         for (ic = ns; ic < ne; ic++)
         {

            jj_row_begining = jj_counter;

            /*-----------------------------------------------------------------
             *  Loop over entries in row ic of R_offd.
             *-----------------------------------------------------------------*/

            for (jj1 = R_offd_i[ic]; jj1 < R_offd_i[ic + 1]; jj1++)
            {
               i1  = R_offd_j[jj1];

               /*--------------------------------------------------------------
                *  Loop over entries in row i1 of A_offd.
                *--------------------------------------------------------------*/

               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
               {
                  i2 = A_offd_j[jj2];

                  /*-----------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited. New entries in RAP only occur from unmarked points.
                   *-----------------------------------------------------------*/

                  if (A_marker[i2] != ic)
                  {

                     /*--------------------------------------------------------
                      *  Mark i2 as visited.
                      *--------------------------------------------------------*/

                     A_marker[i2] = ic;

                     /*--------------------------------------------------------
                      *  Loop over entries in row i2 of P_ext.
                      *--------------------------------------------------------*/

                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, mark it and
                         *  increment counter.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter++;
                        }
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_offd_j[jj3] + num_cols_diag_P;

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, mark it and
                         *  increment counter.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter++;
                        }
                     }
                  }
               }
               /*--------------------------------------------------------------
                *  Loop over entries in row i1 of A_diag.
                *--------------------------------------------------------------*/

               for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
               {
                  i2 = A_diag_j[jj2];

                  /*-----------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited. New entries in RAP only occur from unmarked
                   * points.
                   *-----------------------------------------------------------*/

                  if (A_marker[i2 + num_cols_offd_A] != ic)
                  {

                     /*--------------------------------------------------------
                      *  Mark i2 as visited.
                      *--------------------------------------------------------*/

                     A_marker[i2 + num_cols_offd_A] = ic;

                     /*--------------------------------------------------------
                      *  Loop over entries in row i2 of P_diag.
                      *--------------------------------------------------------*/

                     for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_diag_j[jj3];

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, mark it and
                         *  increment counter.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           jj_counter++;
                        }
                     }

                     /*--------------------------------------------------------
                      *  Loop over entries in row i2 of P_offd.
                      *--------------------------------------------------------*/

                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, mark it and
                         *  increment counter.
                         *-----------------------------------------------------*/

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

      for (i = 0; i < num_threads - 1; i++) { jj_count[i + 1] += jj_count[i]; }

      RAP_size = jj_count[num_threads - 1];
      RAP_int_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd_RT + 1, NALU_HYPRE_MEMORY_HOST);
      RAP_int_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  RAP_size * bnnz, NALU_HYPRE_MEMORY_HOST);
      RAP_int_j    = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  RAP_size, NALU_HYPRE_MEMORY_HOST);
      RAP_int_i[num_cols_offd_RT] = RAP_size;

      /*-----------------------------------------------------------------------
       *  Second Pass: Fill in RAP_int_data and RAP_int_j.
       *-----------------------------------------------------------------------*/

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

         /*--------------------------------------------------------------------
          *  Initialize some stuff.
          *--------------------------------------------------------------------*/

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

         /*--------------------------------------------------------------------
          *  Loop over exterior c-points.
          *--------------------------------------------------------------------*/

         for (ic = ns; ic < ne; ic++)
         {
            jj_row_begining = jj_counter;
            RAP_int_i[ic] = jj_counter;

            /*-----------------------------------------------------------------
             *  Loop over entries in row ic of R_offd.
             *-----------------------------------------------------------------*/

            for (jj1 = R_offd_i[ic]; jj1 < R_offd_i[ic + 1]; jj1++)
            {
               i1  = R_offd_j[jj1];
               r_entries = &(R_offd_data[jj1 * bnnz]);

               /*--------------------------------------------------------------
                *  Loop over entries in row i1 of A_offd.
                *--------------------------------------------------------------*/

               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
               {
                  i2 = A_offd_j[jj2];
                  nalu_hypre_CSRBlockMatrixBlockMultAdd(r_entries,
                                                   &(A_offd_data[jj2 * bnnz]), zero,
                                                   r_a_products, block_size);

                  /*-----------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited.New entries in RAP only occur from unmarked points.
                   *-----------------------------------------------------------*/

                  if (A_marker[i2] != ic)
                  {
                     /*--------------------------------------------------------
                      *  Mark i2 as visited.
                      *--------------------------------------------------------*/

                     A_marker[i2] = ic;

                     /*--------------------------------------------------------
                      *  Loop over entries in row i2 of P_ext.
                      *--------------------------------------------------------*/

                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];
                        nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_diag_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, create a new
                         *  entry. If it has, add new contribution.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           for (kk = 0; kk < bnnz; kk++)
                              RAP_int_data[jj_counter * bnnz + kk] =
                                 r_a_p_products[kk];
                           RAP_int_j[jj_counter] = i3 + first_col_diag_P;
                           jj_counter++;
                        }
                        else
                        {
                           for (kk = 0; kk < bnnz; kk++)
                              RAP_int_data[P_marker[i3]*bnnz + kk] +=
                                 r_a_p_products[kk];
                        }
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_offd_j[jj3] + num_cols_diag_P;
                        nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_offd_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);

                        /*--------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, create a new
                         *  entry. If it has, add new contribution.
                         *--------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           for (kk = 0; kk < bnnz; kk++)
                              RAP_int_data[jj_counter * bnnz + kk] =
                                 r_a_p_products[kk];
                           RAP_int_j[jj_counter]
                              = col_map_offd_Pext[i3 - num_cols_diag_P];
                           jj_counter++;
                        }
                        else
                        {
                           for (kk = 0; kk < bnnz; kk++)
                              RAP_int_data[P_marker[i3]*bnnz + kk] +=
                                 r_a_p_products[kk];
                        }
                     }
                  }

                  /*-----------------------------------------------------------
                   *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                   *  no new entries in RAP and can just add new contributions.
                   *-----------------------------------------------------------*/

                  else
                  {
                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];
                        nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_diag_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);
                        for (kk = 0; kk < bnnz; kk++)
                           RAP_int_data[P_marker[i3]*bnnz + kk] +=
                              r_a_p_products[kk];
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_offd_j[jj3] + num_cols_diag_P;
                        nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_offd_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);
                        ind = P_marker[i3] * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_int_data[ind++] += r_a_p_products[kk];
                        }
                     }
                  }
               }

               /*--------------------------------------------------------------
                *  Loop over entries in row i1 of A_diag.
                *--------------------------------------------------------------*/

               for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
               {
                  i2 = A_diag_j[jj2];
                  nalu_hypre_CSRBlockMatrixBlockMultAdd(r_entries,
                                                   &(A_diag_data[jj2 * bnnz]), zero, r_a_products,
                                                   block_size);

                  /*-----------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited. New entries in RAP only occur from unmarked points.
                   *-----------------------------------------------------------*/

                  if (A_marker[i2 + num_cols_offd_A] != ic)
                  {

                     /*--------------------------------------------------------
                      *  Mark i2 as visited.
                      *--------------------------------------------------------*/

                     A_marker[i2 + num_cols_offd_A] = ic;

                     /*--------------------------------------------------------
                      *  Loop over entries in row i2 of P_diag.
                      *--------------------------------------------------------*/

                     for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_diag_j[jj3];
                        nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_diag_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, create a new
                         *  entry. If it has, add new contribution.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           ind = jj_counter * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_int_data[ind++] = r_a_p_products[kk];
                           }
                           RAP_int_j[jj_counter] = (NALU_HYPRE_BigInt)i3 + first_col_diag_P;
                           jj_counter++;
                        }
                        else
                        {
                           ind = P_marker[i3] * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_int_data[ind++] += r_a_p_products[kk];
                           }
                        }
                     }
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;
                        nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_offd_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, create a new
                         *  entry. If it has, add new contribution.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begining)
                        {
                           P_marker[i3] = jj_counter;
                           ind = jj_counter * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_int_data[ind++] = r_a_p_products[kk];
                           }
                           RAP_int_j[jj_counter] =
                              col_map_offd_Pext[i3 - num_cols_diag_P];
                           jj_counter++;
                        }
                        else
                        {
                           ind = P_marker[i3] * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_int_data[ind++] += r_a_p_products[kk];
                           }
                        }
                     }
                  }

                  /*-----------------------------------------------------------
                   *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                   *  no new entries in RAP and can just add new contributions.
                   *-----------------------------------------------------------*/

                  else
                  {
                     for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_diag_j[jj3];
                        nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_diag_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);
                        ind = P_marker[i3] * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_int_data[ind++] += r_a_p_products[kk];
                        }
                     }
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;
                        nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_offd_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);
                        ind = P_marker[i3] * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_int_data[ind++] += r_a_p_products[kk];
                        }
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

      RAP_int = nalu_hypre_CSRBlockMatrixCreate(block_size, num_cols_offd_RT,
                                           num_rows_offd_RT, RAP_size);
      nalu_hypre_CSRBlockMatrixI(RAP_int) = RAP_int_i;
      nalu_hypre_CSRBlockMatrixBigJ(RAP_int) = RAP_int_j;
      nalu_hypre_CSRBlockMatrixData(RAP_int) = RAP_int_data;
      nalu_hypre_TFree(jj_count, NALU_HYPRE_MEMORY_HOST);
   }

   RAP_ext_size = 0;
   if (num_sends_RT || num_recvs_RT)
   {
      RAP_ext = nalu_hypre_ExchangeRAPBlockData(RAP_int, comm_pkg_RT, block_size);
      RAP_ext_i = nalu_hypre_CSRBlockMatrixI(RAP_ext);
      RAP_ext_j = nalu_hypre_CSRBlockMatrixBigJ(RAP_ext);
      RAP_ext_data = nalu_hypre_CSRBlockMatrixData(RAP_ext);
      RAP_ext_size = RAP_ext_i[nalu_hypre_CSRBlockMatrixNumRows(RAP_ext)];
   }
   if (num_cols_offd_RT)
   {
      nalu_hypre_CSRBlockMatrixDestroy(RAP_int);
      RAP_int = NULL;
   }

   RAP_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_diag_P + 1, NALU_HYPRE_MEMORY_HOST);
   RAP_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_diag_P + 1, NALU_HYPRE_MEMORY_HOST);

   first_col_diag_RAP = first_col_diag_P;
   last_col_diag_RAP = first_col_diag_P + (NALU_HYPRE_BigInt)num_cols_diag_P - 1;

   /*-----------------------------------------------------------------------
    *  check for new nonzero columns in RAP_offd generated through RAP_ext
    *-----------------------------------------------------------------------*/

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
         value = temp[0];
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

   if (num_cols_offd_P)
   {
      map_P_to_RAP = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_P, NALU_HYPRE_MEMORY_HOST);

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
      map_Pext_to_RAP = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_Pext, NALU_HYPRE_MEMORY_HOST);

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

   for (i = 0; i < RAP_ext_size; i++)
      if (RAP_ext_j[i] < first_col_diag_RAP
          || RAP_ext_j[i] > last_col_diag_RAP)
         RAP_ext_j[i] = (NALU_HYPRE_BigInt)(num_cols_diag_P)
                        + nalu_hypre_BigBinarySearch(col_map_offd_RAP,
                                                RAP_ext_j[i], num_cols_offd_RAP);
      else
      {
         RAP_ext_j[i] -= first_col_diag_RAP;
      }

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_cnt_diag = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_threads, NALU_HYPRE_MEMORY_HOST);
   jj_cnt_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_threads, NALU_HYPRE_MEMORY_HOST);

   for (ii = 0; ii < num_threads; ii++)
   {
      size = num_cols_diag_P / num_threads;
      rest = num_cols_diag_P - size * num_threads;
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

         P_marker[ic] = jj_count_diag;
         jj_row_begin_diag = jj_count_diag;
         jj_row_begin_offd = jj_count_offd;
         jj_count_diag++;

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

         /*-----------------------------------------------------------------
          *  Loop over entries in row ic of R_diag.
          *-----------------------------------------------------------------*/

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

                  /*-----------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited.New entries in RAP only occur from unmarked points.
                   *-----------------------------------------------------------*/

                  if (A_marker[i2] != ic)
                  {
                     /*--------------------------------------------------------
                      *  Mark i2 as visited.
                      *--------------------------------------------------------*/

                     A_marker[i2] = ic;

                     /*--------------------------------------------------------
                      *  Loop over entries in row i2 of P_ext.
                      *--------------------------------------------------------*/

                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, mark it and
                         *  increment counter.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begin_diag)
                        {
                           P_marker[i3] = jj_count_diag;
                           jj_count_diag++;
                        }
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]] + num_cols_diag_P;

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, mark it and
                         *  increment counter.
                         *-----------------------------------------------------*/

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

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, mark it and
                         *  increment counter.
                         *-----------------------------------------------------*/

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

   RAP_diag_i[num_cols_diag_P] = jj_count_diag;
   RAP_offd_i[num_cols_diag_P] = jj_count_offd;

   /*-----------------------------------------------------------------------
    *  Allocate RAP_diag_data and RAP_diag_j arrays.
    *  Allocate RAP_offd_data and RAP_offd_j arrays.
    *-----------------------------------------------------------------------*/

   RAP_diag_size = jj_count_diag;
   if (RAP_diag_size)
   {
      RAP_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  RAP_diag_size * bnnz, NALU_HYPRE_MEMORY_HOST);
      RAP_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  RAP_diag_size, NALU_HYPRE_MEMORY_HOST);
   }

   RAP_offd_size = jj_count_offd;
   if (RAP_offd_size)
   {
      RAP_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  RAP_offd_size * bnnz, NALU_HYPRE_MEMORY_HOST);
      RAP_offd_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  RAP_offd_size, NALU_HYPRE_MEMORY_HOST);
   }

   if (RAP_offd_size == 0 && num_cols_offd_RAP != 0)
   {
      num_cols_offd_RAP = 0;
      nalu_hypre_TFree(col_map_offd_RAP, NALU_HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in RAP_diag_data and RAP_diag_j.
    *  Second Pass: Fill in RAP_offd_data and RAP_offd_j.
    *-----------------------------------------------------------------------*/

   for (ii = 0; ii < num_threads; ii++)
   {
      size = num_cols_diag_P / num_threads;
      rest = num_cols_diag_P - size * num_threads;
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

      /*-----------------------------------------------------------------------
       *  Loop over interior c-points.
       *-----------------------------------------------------------------------*/

      for (ic = ns; ic < ne; ic++)
      {

         /*--------------------------------------------------------------------
          *  Create diagonal entry, RAP_{ic,ic} and add entries of RAP_ext
          *--------------------------------------------------------------------*/

         P_marker[ic] = jj_count_diag;
         jj_row_begin_diag = jj_count_diag;
         jj_row_begin_offd = jj_count_offd;
         RAP_diag_i[ic] = jj_row_begin_diag;
         RAP_offd_i[ic] = jj_row_begin_offd;
         ind = jj_count_diag * bnnz;
         for (kk = 0; kk < bnnz; kk++)
         {
            RAP_diag_data[ind++] = zero;
         }
         RAP_diag_j[jj_count_diag] = ic;
         jj_count_diag++;

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
                           ind = jj_count_diag * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_diag_data[ind++] = RAP_ext_data[k * bnnz + kk];
                           }
                           RAP_diag_j[jj_count_diag] = jcol;
                           jj_count_diag++;
                        }
                        else
                        {
                           ind = P_marker[jcol] * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_diag_data[ind++] += RAP_ext_data[k * bnnz + kk];
                           }
                        }
                     }
                     else
                     {
                        if (P_marker[jcol] < jj_row_begin_offd)
                        {
                           P_marker[jcol] = jj_count_offd;
                           ind = jj_count_offd * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_offd_data[ind++] = RAP_ext_data[k * bnnz + kk];
                           }
                           RAP_offd_j[jj_count_offd]
                              = jcol - num_cols_diag_P;
                           jj_count_offd++;
                        }
                        else
                        {
                           ind = P_marker[jcol] * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_offd_data[ind++] += RAP_ext_data[k * bnnz + kk];
                           }
                        }
                     }
                  }
                  break;
               }

         /*--------------------------------------------------------------------
          *  Loop over entries in row ic of R_diag.
          *--------------------------------------------------------------------*/

         for (jj1 = R_diag_i[ic]; jj1 < R_diag_i[ic + 1]; jj1++)
         {
            i1  = R_diag_j[jj1];
            r_entries = &(R_diag_data[jj1 * bnnz]);

            /*-----------------------------------------------------------------
             *  Loop over entries in row i1 of A_offd.
             *-----------------------------------------------------------------*/

            if (num_cols_offd_A)
            {
               for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
               {
                  i2 = A_offd_j[jj2];
                  nalu_hypre_CSRBlockMatrixBlockMultAdd(r_entries,
                                                   &(A_offd_data[jj2 * bnnz]), zero, r_a_products,
                                                   block_size);

                  /*-----------------------------------------------------------
                   *  Check A_marker to see if point i2 has been previously
                   *  visited.New entries in RAP only occur from unmarked points.
                   *-----------------------------------------------------------*/

                  if (A_marker[i2] != ic)
                  {
                     /*--------------------------------------------------------
                      *  Mark i2 as visited.
                      *--------------------------------------------------------*/

                     A_marker[i2] = ic;

                     /*--------------------------------------------------------
                      *  Loop over entries in row i2 of P_ext.
                      *--------------------------------------------------------*/

                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];
                        nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_diag_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, create a new
                         *  entry. If it has, add new contribution.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begin_diag)
                        {
                           P_marker[i3] = jj_count_diag;
                           ind = jj_count_diag * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_diag_data[ind++] = r_a_p_products[kk];
                           }
                           RAP_diag_j[jj_count_diag] = i3;
                           jj_count_diag++;
                        }
                        else
                        {
                           ind = P_marker[i3] * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_diag_data[ind++] += r_a_p_products[kk];
                           }
                        }
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]] + num_cols_diag_P;
                        nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_offd_data[jj3 * bnnz]),
                                                         zero, r_a_p_products, block_size);

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not
                         *  been accounted for. If it has not, create a new
                         *  entry. If it has, add new contribution.
                         *-----------------------------------------------------*/
                        if (P_marker[i3] < jj_row_begin_offd)
                        {
                           P_marker[i3] = jj_count_offd;
                           ind = jj_count_offd * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_offd_data[ind++] = r_a_p_products[kk];
                           }
                           RAP_offd_j[jj_count_offd] = i3 - num_cols_diag_P;
                           jj_count_offd++;
                        }
                        else
                        {
                           ind = P_marker[i3] * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_offd_data[ind++] += r_a_p_products[kk];
                           }
                        }
                     }
                  }

                  /*-----------------------------------------------------------
                   *  If i2 is previously visited ( A_marker[12]=ic ) it yields
                   *  no new entries in RAP and can just add new contributions.
                   *-----------------------------------------------------------*/
                  else
                  {
                     for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2 + 1]; jj3++)
                     {
                        i3 = P_ext_diag_j[jj3];
                        nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_diag_data[jj3 * bnnz]), zero,
                                                         r_a_p_products, block_size);
                        ind = P_marker[i3] * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_diag_data[ind++] += r_a_p_products[kk];
                        }
                     }
                     for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]] + num_cols_diag_P;
                        nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_ext_offd_data[jj3 * bnnz]),
                                                         zero, r_a_p_products, block_size);
                        ind = P_marker[i3] * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_offd_data[ind++] += r_a_p_products[kk];
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
               nalu_hypre_CSRBlockMatrixBlockMultAdd(r_entries,
                                                &(A_diag_data[jj2 * bnnz]),
                                                zero, r_a_products, block_size);

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
                     nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                      &(P_diag_data[jj3 * bnnz]),
                                                      zero, r_a_p_products, block_size);

                     /*--------------------------------------------------------
                      *  Check P_marker to see that RAP_{ic,i3} has not already
                      *  been accounted for. If it has not, create a new entry.
                      *  If it has, add new contribution.
                      *--------------------------------------------------------*/

                     if (P_marker[i3] < jj_row_begin_diag)
                     {
                        P_marker[i3] = jj_count_diag;
                        ind = jj_count_diag * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_diag_data[ind++] = r_a_p_products[kk];
                        }
                        RAP_diag_j[jj_count_diag] = P_diag_j[jj3];
                        jj_count_diag++;
                     }
                     else
                     {
                        ind = P_marker[i3] * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_diag_data[ind++] += r_a_p_products[kk];
                        }
                     }
                  }
                  if (num_cols_offd_P)
                  {
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;
                        nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_offd_data[jj3 * bnnz]),
                                                         zero, r_a_p_products, block_size);

                        /*-----------------------------------------------------
                         *  Check P_marker to see that RAP_{ic,i3} has not already
                         *  been accounted for. If it has not, create a new entry.
                         *  If it has, add new contribution.
                         *-----------------------------------------------------*/

                        if (P_marker[i3] < jj_row_begin_offd)
                        {
                           P_marker[i3] = jj_count_offd;
                           ind = jj_count_offd * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_offd_data[ind++] = r_a_p_products[kk];
                           }
                           RAP_offd_j[jj_count_offd] = i3 - num_cols_diag_P;
                           jj_count_offd++;
                        }
                        else
                        {
                           ind = P_marker[i3] * bnnz;
                           for (kk = 0; kk < bnnz; kk++)
                           {
                              RAP_offd_data[ind++] += r_a_p_products[kk];
                           }
                        }
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
                     nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                      &(P_diag_data[jj3 * bnnz]),
                                                      zero, r_a_p_products, block_size);
                     ind = P_marker[i3] * bnnz;
                     for (kk = 0; kk < bnnz; kk++)
                     {
                        RAP_diag_data[ind++] += r_a_p_products[kk];
                     }
                  }
                  if (num_cols_offd_P)
                  {
                     for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2 + 1]; jj3++)
                     {
                        i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;
                        nalu_hypre_CSRBlockMatrixBlockMultAdd(r_a_products,
                                                         &(P_offd_data[jj3 * bnnz]),
                                                         zero, r_a_p_products, block_size);
                        ind = P_marker[i3] * bnnz;
                        for (kk = 0; kk < bnnz; kk++)
                        {
                           RAP_offd_data[ind++] += r_a_p_products[kk];
                        }
                     }
                  }
               }
            }
         }
      }
      nalu_hypre_TFree(P_mark_array[ii], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(A_mark_array[ii], NALU_HYPRE_MEMORY_HOST);
   }


   for (i = 0; i < 2; i++)
   {
      row_starts[i] = col_starts[i] = coarse_partitioning[i];
   }

   RAP = nalu_hypre_ParCSRBlockMatrixCreate(comm, block_size, n_coarse, n_coarse,
                                       row_starts, col_starts,
                                       num_cols_offd_RAP, RAP_diag_size, RAP_offd_size);

   RAP_diag = nalu_hypre_ParCSRBlockMatrixDiag(RAP);
   nalu_hypre_CSRBlockMatrixI(RAP_diag) = RAP_diag_i;
   if (RAP_diag_size)
   {
      nalu_hypre_CSRBlockMatrixData(RAP_diag) = RAP_diag_data;
      nalu_hypre_CSRBlockMatrixJ(RAP_diag) = RAP_diag_j;
   }

   RAP_offd = nalu_hypre_ParCSRBlockMatrixOffd(RAP);
   nalu_hypre_CSRBlockMatrixI(RAP_offd) = RAP_offd_i;
   if (num_cols_offd_RAP)
   {
      nalu_hypre_CSRBlockMatrixData(RAP_offd) = RAP_offd_data;
      nalu_hypre_CSRBlockMatrixJ(RAP_offd) = RAP_offd_j;
      nalu_hypre_ParCSRBlockMatrixColMapOffd(RAP) = col_map_offd_RAP;
   }
   if (num_procs > 1)
   {
      nalu_hypre_BlockMatvecCommPkgCreate(RAP);
   }

   *RAP_ptr = RAP;

   /*-----------------------------------------------------------------------
    *  Free R, P_ext and marker arrays.
    *-----------------------------------------------------------------------*/

   nalu_hypre_CSRBlockMatrixDestroy(R_diag);
   R_diag = NULL;

   if (num_cols_offd_RT)
   {
      nalu_hypre_CSRBlockMatrixDestroy(R_offd);
      R_offd = NULL;
   }

   if (num_sends_RT || num_recvs_RT)
   {
      nalu_hypre_CSRBlockMatrixDestroy(RAP_ext);
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

   nalu_hypre_TFree(r_a_products, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(r_a_p_products, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}
