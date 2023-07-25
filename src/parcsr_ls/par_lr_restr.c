/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_lapack.h"
#include "_nalu_hypre_blas.h"


// TODO : delete csrAi, csrAi_i, csrAi_j,
//             csrAi_a, csrAiT_i, csrAiT_j, csrAiT_a
//    Use
//       nalu_hypre_dense_topo_sort(NALU_HYPRE_Real *L, NALU_HYPRE_Int *ordering, NALU_HYPRE_Int n)
//    to get ordering for triangular solve. Can provide


NALU_HYPRE_Int AIR_TOT_SOL_SIZE = 0;
NALU_HYPRE_Int AIR_MAX_SOL_SIZE = 0;

#define AIR_DEBUG 0
#define EPSILON 1e-18
#define EPSIMAC 1e-16

void nalu_hypre_fgmresT(NALU_HYPRE_Int n, NALU_HYPRE_Complex *A, NALU_HYPRE_Complex *b, NALU_HYPRE_Real tol, NALU_HYPRE_Int kdim,
                   NALU_HYPRE_Complex *x, NALU_HYPRE_Real *relres, NALU_HYPRE_Int *iter, NALU_HYPRE_Int job);
void nalu_hypre_ordered_GS(const NALU_HYPRE_Complex L[], const NALU_HYPRE_Complex rhs[], NALU_HYPRE_Complex x[],
                      const NALU_HYPRE_Int n);

/*
NALU_HYPRE_Real air_time0 = 0.0;
NALU_HYPRE_Real air_time_comm = 0.0;
NALU_HYPRE_Real air_time1 = 0.0;
NALU_HYPRE_Real air_time2 = 0.0;
NALU_HYPRE_Real air_time3 = 0.0;
NALU_HYPRE_Real air_time4 = 0.0;
*/

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildRestrDist2AIR( nalu_hypre_ParCSRMatrix   *A,
                                   NALU_HYPRE_Int            *CF_marker,
                                   nalu_hypre_ParCSRMatrix   *S,
                                   NALU_HYPRE_BigInt         *num_cpts_global,
                                   NALU_HYPRE_Int             num_functions,
                                   NALU_HYPRE_Int            *dof_func,
                                   NALU_HYPRE_Real            filter_thresholdR,
                                   NALU_HYPRE_Int             debug_flag,
                                   nalu_hypre_ParCSRMatrix  **R_ptr,
                                   NALU_HYPRE_Int             AIR1_5,
                                   NALU_HYPRE_Int             is_triangular,
                                   NALU_HYPRE_Int             gmres_switch)
{
   /* NALU_HYPRE_Real t0 = nalu_hypre_MPI_Wtime(); */

   MPI_Comm                 comm     = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   nalu_hypre_ParCSRCommPkg     *comm_pkg_SF = NULL;

   /* diag part of A */
   nalu_hypre_CSRMatrix *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_a = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   nalu_hypre_CSRMatrix *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_a = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);

   NALU_HYPRE_Int        num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt    *col_map_offd_A  = nalu_hypre_ParCSRMatrixColMapOffd(A);
   /* Strength matrix S */
   /* diag part of S */
   nalu_hypre_CSRMatrix *S_diag   = nalu_hypre_ParCSRMatrixDiag(S);
   NALU_HYPRE_Int       *S_diag_i = nalu_hypre_CSRMatrixI(S_diag);
   NALU_HYPRE_Int       *S_diag_j = nalu_hypre_CSRMatrixJ(S_diag);
   /* off-diag part of S */
   nalu_hypre_CSRMatrix *S_offd   = nalu_hypre_ParCSRMatrixOffd(S);
   NALU_HYPRE_Int       *S_offd_i = nalu_hypre_CSRMatrixI(S_offd);
   NALU_HYPRE_Int       *S_offd_j = nalu_hypre_CSRMatrixJ(S_offd);
   /* Restriction matrix R */
   nalu_hypre_ParCSRMatrix *R;
   /* csr's */
   nalu_hypre_CSRMatrix *R_diag;
   nalu_hypre_CSRMatrix *R_offd;
   /* arrays */
   NALU_HYPRE_Complex   *R_diag_data;
   NALU_HYPRE_Int       *R_diag_i;
   NALU_HYPRE_Int       *R_diag_j;
   NALU_HYPRE_Complex   *R_offd_data;
   NALU_HYPRE_Int       *R_offd_i;
   NALU_HYPRE_Int       *R_offd_j;
   NALU_HYPRE_BigInt    *col_map_offd_R;
   NALU_HYPRE_Int       *tmp_map_offd = NULL;
   /* CF marker off-diag part */
   NALU_HYPRE_Int       *CF_marker_offd = NULL;
   /* func type off-diag part */
   NALU_HYPRE_Int       *dof_func_offd  = NULL;

   NALU_HYPRE_BigInt     big_i1, big_j1, big_k1;
   NALU_HYPRE_Int        i, j, j1, j2, k, i1, i2, k1, k2, k3, rr, cc, ic, index, start, end,
                    local_max_size, local_size, num_cols_offd_R;
   /*NALU_HYPRE_Int        i6;*/
   NALU_HYPRE_BigInt     *FF2_offd;
   NALU_HYPRE_Int        FF2_offd_len;

   /* LAPACK */
   NALU_HYPRE_Complex *DAi, *Dbi, *Dxi;
#if AIR_DEBUG
   NALU_HYPRE_Complex *TMPA, *TMPb, *TMPd;
   nalu_hypre_Vector *tmpv;
#endif
   NALU_HYPRE_Int *Ipi, lapack_info, ione = 1, *RRi, *KKi;
   char charT = 'T';

   /* if the size of local system is larger than gmres_switch, use GMRES */
   char Aisol_method;
   NALU_HYPRE_Int gmresAi_maxit = 50;
   NALU_HYPRE_Real gmresAi_tol = 1e-3;

   NALU_HYPRE_Int my_id, num_procs;
   NALU_HYPRE_BigInt total_global_cpts/*, my_first_cpt*/;
   NALU_HYPRE_Int nnz_diag, nnz_offd, cnt_diag, cnt_offd;
   NALU_HYPRE_Int *Marker_diag, *Marker_offd;
   NALU_HYPRE_Int *Marker_diag_j, Marker_diag_count;
   NALU_HYPRE_Int num_sends, num_recvs, num_elems_send;
   /* local size, local num of C points */
   NALU_HYPRE_Int n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int n_cpts = 0;
   /* my column range */
   NALU_HYPRE_BigInt col_start = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   NALU_HYPRE_BigInt col_end   = col_start + (NALU_HYPRE_BigInt)n_fine;

   NALU_HYPRE_Int  *send_buf_i;

   /* recv_SF means the Strong F-neighbors of offd elements in col_map_offd */
   NALU_HYPRE_Int *send_SF_i, send_SF_jlen;
   NALU_HYPRE_BigInt *send_SF_j;
   NALU_HYPRE_BigInt *recv_SF_j;
   NALU_HYPRE_Int *recv_SF_i, *recv_SF_j2, recv_SF_jlen;
   NALU_HYPRE_Int *send_SF_jstarts, *recv_SF_jstarts;
   NALU_HYPRE_BigInt *recv_SF_offd_list;
   NALU_HYPRE_Int recv_SF_offd_list_len;
   NALU_HYPRE_Int *Mapper_recv_SF_offd_list, *Mapper_offd_A, *Marker_recv_SF_offd_list;
   NALU_HYPRE_Int *Marker_FF2_offd;
   NALU_HYPRE_Int *Marker_FF2_offd_j, Marker_FF2_offd_count;

   /* for communication of offd F and F^2 rows of A */
   nalu_hypre_ParCSRCommPkg *comm_pkg_FF2_i, *comm_pkg_FF2_j = NULL;
   NALU_HYPRE_BigInt *send_FF2_j, *recv_FF2_j;
   NALU_HYPRE_Int num_sends_FF2, *send_FF2_i, send_FF2_ilen, send_FF2_jlen,
             num_recvs_FF2, *recv_FF2_i, recv_FF2_ilen, recv_FF2_jlen,
             *send_FF2_jstarts, *recv_FF2_jstarts;
   NALU_HYPRE_Complex *send_FF2_a, *recv_FF2_a;

   /* ghost rows: offd F and F2-pts */
   nalu_hypre_CSRMatrix *A_offd_FF2   = NULL;

   /*
   NALU_HYPRE_Real tcomm = nalu_hypre_MPI_Wtime();
   */

   /* MPI size and rank*/
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /*-------------- global number of C points and my start position */
   /*my_first_cpt = num_cpts_global[0];*/
   if (my_id == (num_procs - 1))
   {
      total_global_cpts = num_cpts_global[1];
   }
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

   /* init markers to zeros */
   Marker_diag = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   Marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);

   /* number of sends (number of procs) */
   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);

   /* number of recvs (number of procs) */
   num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);

   /* number of elements to send */
   num_elems_send = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

   /* send buffer, of size send_map_starts[num_sends]),
    * i.e., number of entries to send */
   send_buf_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_elems_send, NALU_HYPRE_MEMORY_HOST);

   /* copy CF markers of elements to send to buffer
    * RL: why copy them with two for loops? Why not just loop through all in one */
   for (i = 0, index = 0; i < num_sends; i++)
   {
      /* start pos of elements sent to send_proc[i] */
      start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      /* loop through all elems to send_proc[i] */
      for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
      {
         /* CF marker of send_map_elemts[j] */
         send_buf_i[index++] = CF_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
      }
   }
   /* create a handle to start communication. 11: for integer */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, send_buf_i, CF_marker_offd);
   /* destroy the handle to finish communication */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /* do a similar communication for dof_func */
   if (num_functions > 1)
   {
      for (i = 0, index = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            send_buf_i[index++] = dof_func[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, send_buf_i, dof_func_offd);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *        Send/Recv Offd F-neighbors' strong F-neighbors
    *        F^2: OffdF - F
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
   send_SF_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_elems_send, NALU_HYPRE_MEMORY_HOST);
   recv_SF_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd + 1, NALU_HYPRE_MEMORY_HOST);

   /* for each F-elem to send, find the number of strong F-neighbors */
   for (i = 0, send_SF_jlen = 0; i < num_elems_send; i++)
   {
      /* number of strong F-pts */
      send_SF_i[i] = 0;
      /* elem i1 */
      i1 = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i);
      /* ignore C-pts */
      if (CF_marker[i1] >= 0)
      {
         continue;
      }
      /* diag part of row i1 */
      for (j = S_diag_i[i1]; j < S_diag_i[i1 + 1]; j++)
      {
         if (CF_marker[S_diag_j[j]] < 0)
         {
            send_SF_i[i] ++;
         }
      }
      /* offd part of row i1 */
      for (j = S_offd_i[i1]; j < S_offd_i[i1 + 1]; j++)
      {
         j1 = S_offd_j[j];
         if (CF_marker_offd[j1] < 0)
         {
            send_SF_i[i] ++;
         }
      }

      /* add to the num of elems going to be sent */
      send_SF_jlen += send_SF_i[i];
   }

   /* do communication */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, send_SF_i, recv_SF_i + 1);
   /* ... */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   send_SF_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, send_SF_jlen, NALU_HYPRE_MEMORY_HOST);
   send_SF_jstarts = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_sends + 1, NALU_HYPRE_MEMORY_HOST);

   for (i = 0, i1 = 0; i < num_sends; i++)
   {
      /* start pos of elements sent to send_proc[i] */
      start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      /* 1-past-the-end pos */
      end   = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);

      for (j = start; j < end; j++)
      {
         /* strong F-pt, j1 */
         j1 = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         /* ignore C-pts */
         if (CF_marker[j1] >= 0)
         {
            continue;
         }
         /* diag part of row j1 */
         for (k = S_diag_i[j1]; k < S_diag_i[j1 + 1]; k++)
         {
            k1 = S_diag_j[k];
            if (CF_marker[k1] < 0)
            {
               send_SF_j[i1++] = col_start + (NALU_HYPRE_BigInt)k1;
            }
         }
         /* offd part of row j1 */
         for (k = S_offd_i[j1]; k < S_offd_i[j1 + 1]; k++)
         {
            k1 = S_offd_j[k];
            if (CF_marker_offd[k1] < 0)
            {
               send_SF_j[i1++] = col_map_offd_A[k1];
            }
         }
      }
      send_SF_jstarts[i + 1] = i1;
   }

   nalu_hypre_assert(i1 == send_SF_jlen);

   /* adjust recv_SF_i to ptrs */
   for (i = 1; i <= num_cols_A_offd; i++)
   {
      recv_SF_i[i] += recv_SF_i[i - 1];
   }

   recv_SF_jlen = recv_SF_i[num_cols_A_offd];
   recv_SF_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, recv_SF_jlen, NALU_HYPRE_MEMORY_HOST);
   recv_SF_jstarts = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_recvs + 1, NALU_HYPRE_MEMORY_HOST);

   for (i = 1; i <= num_recvs; i++)
   {
      start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
      recv_SF_jstarts[i] = recv_SF_i[start];
   }

   /* create a communication package for SF_j */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs,
                                    nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                    recv_SF_jstarts,
                                    num_sends,
                                    nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                    send_SF_jstarts,
                                    NULL,
                                    &comm_pkg_SF);

   /* do communication */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(21, comm_pkg_SF, send_SF_j, recv_SF_j);
   /* ... */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    * recv_SF_offd_list: a sorted list of offd elems in recv_SF_j
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   recv_SF_offd_list = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, recv_SF_jlen, NALU_HYPRE_MEMORY_HOST);
   for (i = 0, j = 0; i < recv_SF_jlen; i++)
   {
      NALU_HYPRE_Int flag = 1;
      big_i1 = recv_SF_j[i];
      /* offd */
      if (big_i1 < col_start || big_i1 >= col_end)
      {
         if (AIR1_5)
         {
            flag = nalu_hypre_BigBinarySearch(col_map_offd_A, big_i1, num_cols_A_offd) != -1;
         }
         if (flag)
         {
            recv_SF_offd_list[j++] = big_i1;
         }
      }
   }

   /* remove redundancy after sorting */
   nalu_hypre_BigQsort0(recv_SF_offd_list, 0, j - 1);

   for (i = 0, recv_SF_offd_list_len = 0; i < j; i++)
   {
      if (i == 0 || recv_SF_offd_list[i] != recv_SF_offd_list[i - 1])
      {
         recv_SF_offd_list[recv_SF_offd_list_len++] = recv_SF_offd_list[i];
      }
   }

   /* make a copy of recv_SF_j in which
    * adjust the offd indices corresponding to recv_SF_offd_list */
   recv_SF_j2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int, recv_SF_jlen, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < recv_SF_jlen; i++)
   {
      big_i1 = recv_SF_j[i];
      if (big_i1 < col_start || big_i1 >= col_end)
      {
         j = nalu_hypre_BigBinarySearch(recv_SF_offd_list, big_i1, recv_SF_offd_list_len);
         if (!AIR1_5)
         {
            nalu_hypre_assert(j >= 0 && j < recv_SF_offd_list_len);
         }
         recv_SF_j2[i] = j;
      }
      else
      {
         recv_SF_j2[i] = -1;
      }
   }

   /* mapping to col_map_offd_A */
   Mapper_recv_SF_offd_list = nalu_hypre_CTAlloc(NALU_HYPRE_Int, recv_SF_offd_list_len, NALU_HYPRE_MEMORY_HOST);
   Marker_recv_SF_offd_list = nalu_hypre_CTAlloc(NALU_HYPRE_Int, recv_SF_offd_list_len, NALU_HYPRE_MEMORY_HOST);

   /* create a mapping from recv_SF_offd_list to col_map_offd_A for their intersections */
   for (i = 0; i < recv_SF_offd_list_len; i++)
   {
      big_i1 = recv_SF_offd_list[i];
      nalu_hypre_assert(big_i1 < col_start || big_i1 >= col_end);
      j = nalu_hypre_BigBinarySearch(col_map_offd_A, big_i1, num_cols_A_offd);
      /* mapping to col_map_offd_A, if not found equal to -1 */
      Mapper_recv_SF_offd_list[i] = j;
   }

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *       Find offd F and F-F (F^2) neighboring points for C-pts
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
   for (i = 0, FF2_offd_len = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      /* diag(F)-offd(F) */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         j1 = S_diag_j[j];
         /* if it is F */
         if (CF_marker[j1] < 0)
         {
            /* go through its offd part */
            for (k = S_offd_i[j1]; k < S_offd_i[j1 + 1]; k++)
            {
               k1 = S_offd_j[k];
               if (CF_marker_offd[k1] < 0)
               {
                  /* mark F pts */
                  if (!Marker_offd[k1])
                  {
                     FF2_offd_len ++;
                     Marker_offd[k1] = 1;
                  }
               }
            }
         }
      }

      /* offd(F) and offd(F)-offd(F)
       * NOTE: we are working with two marker arrays here: Marker_offd and Marker_recv_SF_offd_list
       * which may have overlap.
       * So, we always check the first marker array */
      for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
      {
         j1 = S_offd_j[j];
         /* offd F pts */
         if (CF_marker_offd[j1] < 0)
         {
            if (!Marker_offd[j1])
            {
               FF2_offd_len ++;
               Marker_offd[j1] = 1;
            }
            /* offd(F)-offd(F), need to open recv_SF */
            for (k = recv_SF_i[j1]; k < recv_SF_i[j1 + 1]; k++)
            {
               /* k1: global index */
               big_k1 = recv_SF_j[k];
               /* if k1 is not in my range */
               if (big_k1 < col_start || big_k1 >= col_end)
               {
                  /* index in recv_SF_offd_list */
                  k2 = recv_SF_j2[k];

                  if (AIR1_5 && k2 == -1)
                  {
                     continue;
                  }

                  nalu_hypre_assert(recv_SF_offd_list[k2] == big_k1);

                  /* map to offd_A */
                  k3 = Mapper_recv_SF_offd_list[k2];
                  if (k3 >= 0)
                  {
                     if (!Marker_offd[k3])
                     {
                        FF2_offd_len ++;
                        Marker_offd[k3] = 1;
                     }
                  }
                  else
                  {
                     if (!Marker_recv_SF_offd_list[k2])
                     {
                        FF2_offd_len ++;
                        Marker_recv_SF_offd_list[k2] = 1;
                     }
                  }
               }
            }
         }
      }
   }

   /* create a list of offd F, F2 points
    * and RESET the markers to ZEROs*/
   FF2_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, FF2_offd_len, NALU_HYPRE_MEMORY_HOST);
   for (i = 0, k = 0; i < num_cols_A_offd; i++)
   {
      if (Marker_offd[i])
      {
         FF2_offd[k++] = col_map_offd_A[i];
         Marker_offd[i] = 0;
      }
   }

   for (i = 0; i < recv_SF_offd_list_len; i++)
   {
      /* debug: if mapping exists, this marker should not be set */
      if (Mapper_recv_SF_offd_list[i] >= 0)
      {
         nalu_hypre_assert(Marker_recv_SF_offd_list[i] == 0);
      }

      if (Marker_recv_SF_offd_list[i])
      {
         big_i1 = recv_SF_offd_list[i];
         nalu_hypre_assert(big_i1 < col_start || big_i1 >= col_end);
         FF2_offd[k++] = big_i1;
         Marker_recv_SF_offd_list[i] = 0;
      }
   }
   nalu_hypre_assert(k == FF2_offd_len);

   /* sort the list */
   nalu_hypre_BigQsort0(FF2_offd, 0, FF2_offd_len - 1);

   /* there must be no repetition in FF2_offd */
   for (i = 1; i < FF2_offd_len; i++)
   {
      nalu_hypre_assert(FF2_offd[i] != FF2_offd[i - 1]);
   }

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *    Create CommPkgs for exchanging offd F and F2 rows of A
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   /* we will create TWO commPkg: one for row lengths and one for row data,
    * similar to what we have done above for SF_i, SF_j */
   nalu_hypre_ParCSRFindExtendCommPkg(comm,
                                 nalu_hypre_ParCSRMatrixGlobalNumCols(A),
                                 nalu_hypre_ParCSRMatrixFirstColDiag(A),
                                 nalu_hypre_CSRMatrixNumCols(A_diag),
                                 nalu_hypre_ParCSRMatrixColStarts(A),
                                 nalu_hypre_ParCSRMatrixAssumedPartition(A),
                                 FF2_offd_len,
                                 FF2_offd,
                                 &comm_pkg_FF2_i);
   /* number of sends (#procs) */
   num_sends_FF2 = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg_FF2_i);
   /* number of rows to send */
   send_FF2_ilen = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg_FF2_i, num_sends_FF2);
   /* number of recvs (#procs) */
   num_recvs_FF2 = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg_FF2_i);
   /* number of rows to recv */
   recv_FF2_ilen = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg_FF2_i, num_recvs_FF2);

   nalu_hypre_assert(FF2_offd_len == recv_FF2_ilen);

   send_FF2_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, send_FF2_ilen, NALU_HYPRE_MEMORY_HOST);
   recv_FF2_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, recv_FF2_ilen + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0, send_FF2_jlen = 0; i < send_FF2_ilen; i++)
   {
      j = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg_FF2_i, i);
      for (k = A_diag_i[j]; k < A_diag_i[j + 1]; k++)
      {
         if (CF_marker[A_diag_j[k]] < 0)
         {
            send_FF2_i[i]++;
         }
      }
      if (num_procs > 1)
      {
         for (k = A_offd_i[j]; k < A_offd_i[j + 1]; k++)
         {
            if (CF_marker_offd[A_offd_j[k]] < 0)
            {
               send_FF2_i[i]++;
            }
         }
      }
      //send_FF2_i[i] = A_diag_i[j+1] - A_diag_i[j] + A_offd_i[j+1] - A_offd_i[j];
      send_FF2_jlen += send_FF2_i[i];
   }

   /* do communication */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg_FF2_i, send_FF2_i, recv_FF2_i + 1);
   /* ... */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   send_FF2_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, send_FF2_jlen, NALU_HYPRE_MEMORY_HOST);
   send_FF2_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, send_FF2_jlen, NALU_HYPRE_MEMORY_HOST);
   send_FF2_jstarts = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_sends_FF2 + 1, NALU_HYPRE_MEMORY_HOST);

   for (i = 0, i1 = 0; i < num_sends_FF2; i++)
   {
      start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg_FF2_i, i);
      end   = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg_FF2_i, i + 1);
      for (j = start; j < end; j++)
      {
         /* will send row j1 to send_proc[i] */
         j1 = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg_FF2_i, j);
         /* open row j1 and fill ja and a */
         for (k = A_diag_i[j1]; k < A_diag_i[j1 + 1]; k++)
         {
            NALU_HYPRE_Int k1 = A_diag_j[k];
            if (CF_marker[k1] < 0)
            {
               send_FF2_j[i1] = col_start + k1;
               send_FF2_a[i1] = A_diag_a[k];
               i1++;
            }
         }
         if (num_procs > 1)
         {
            for (k = A_offd_i[j1]; k < A_offd_i[j1 + 1]; k++)
            {
               NALU_HYPRE_Int k1 = A_offd_j[k];
               if (CF_marker_offd[k1] < 0)
               {
                  send_FF2_j[i1] = col_map_offd_A[k1];
                  send_FF2_a[i1] = A_offd_a[k];
                  i1++;
               }
            }
         }
      }
      send_FF2_jstarts[i + 1] = i1;
   }
   nalu_hypre_assert(i1 == send_FF2_jlen);

   /* adjust recv_FF2_i to ptrs */
   for (i = 1; i <= recv_FF2_ilen; i++)
   {
      recv_FF2_i[i] += recv_FF2_i[i - 1];
   }

   recv_FF2_jlen = recv_FF2_i[recv_FF2_ilen];
   recv_FF2_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, recv_FF2_jlen, NALU_HYPRE_MEMORY_HOST);
   recv_FF2_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, recv_FF2_jlen, NALU_HYPRE_MEMORY_HOST);
   recv_FF2_jstarts = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_recvs_FF2 + 1, NALU_HYPRE_MEMORY_HOST);

   for (i = 1; i <= num_recvs_FF2; i++)
   {
      start = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg_FF2_i, i);
      recv_FF2_jstarts[i] = recv_FF2_i[start];
   }

   /* create a communication package for FF2_j */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs_FF2,
                                    nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg_FF2_i),
                                    recv_FF2_jstarts,
                                    num_sends_FF2,
                                    nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg_FF2_i),
                                    send_FF2_jstarts,
                                    NULL,
                                    &comm_pkg_FF2_j);

   /* do communication */
   /* ja */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(21, comm_pkg_FF2_j, send_FF2_j, recv_FF2_j);
   /* ... */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /* a */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate( 1, comm_pkg_FF2_j, send_FF2_a, recv_FF2_a);
   /* ... */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /* A_offd_FF2 is ready ! */
   /* Careful! Wrong data type for number of columns ! */
   //A_offd_FF2 = nalu_hypre_CSRMatrixCreate(recv_FF2_ilen, nalu_hypre_ParCSRMatrixGlobalNumCols(A),
   /* Careful! Wrong column size! Hopefully won't matter! */
   A_offd_FF2 = nalu_hypre_CSRMatrixCreate(recv_FF2_ilen, recv_FF2_ilen,
                                      recv_FF2_jlen);

   nalu_hypre_CSRMatrixI   (A_offd_FF2) = recv_FF2_i;
   nalu_hypre_CSRMatrixBigJ (A_offd_FF2) = recv_FF2_j;
   nalu_hypre_CSRMatrixData(A_offd_FF2) = recv_FF2_a;

   /*
   for (i6 = 0; i6 < num_procs; i6 ++)
   {
      if (i6 == my_id)
      {
         nalu_hypre_assert(nalu_hypre_CSRMatrixNumNonzeros(A_offd_FF2) == \
                      nalu_hypre_CSRMatrixI(A_offd_FF2)[nalu_hypre_CSRMatrixNumRows(A_offd_FF2)]);

         for (i = 0; i < nalu_hypre_CSRMatrixNumRows(A_offd_FF2); i++)
         {
            for (j = nalu_hypre_CSRMatrixI(A_offd_FF2)[i]; j < nalu_hypre_CSRMatrixI(A_offd_FF2)[i+1]; j++)
            {
               NALU_HYPRE_Int r = FF2_offd[i];
               NALU_HYPRE_Int c = nalu_hypre_CSRMatrixJ(A_offd_FF2)[j];
               nalu_hypre_assert(c >= 0 && c < nalu_hypre_CSRMatrixNumCols(A_offd_FF2));
               NALU_HYPRE_Complex v = nalu_hypre_CSRMatrixData(A_offd_FF2)[j];
               nalu_hypre_printf("%8d %8d     % e\n", r, c, v);
            }
         }
         nalu_hypre_printf("\n\n");
      }
      nalu_hypre_MPI_Barrier(nalu_hypre_MPI_COMM_WORLD);
   }
   */

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    * FF2_offd contains all the offd indices and corresponds to matrix A_offd_FF2
    * So, we are able to use indices in terms of FF2_offd to bookkeeping all offd
    * information.
    * [ FF2_offd is a subset of col_map_offd_A UNION recv_SF_offd_list ]
    * Mappings from col_map_offd_A and recv_SF_offd_list will be created
    * markers for FF2_offd will also be created
    * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

   /* Mapping from col_map_offd_A */
   Mapper_offd_A = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_cols_A_offd; i++)
   {
      Mapper_offd_A[i] = nalu_hypre_BigBinarySearch(FF2_offd, col_map_offd_A[i], FF2_offd_len);
   }

   /* Mapping from recv_SF_offd_list, overwrite the old one*/
   for (i = 0; i < recv_SF_offd_list_len; i++)
   {
      Mapper_recv_SF_offd_list[i] = nalu_hypre_BigBinarySearch(FF2_offd, recv_SF_offd_list[i], FF2_offd_len);
   }

   /* marker */
   Marker_FF2_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, FF2_offd_len, NALU_HYPRE_MEMORY_HOST);

   /*
   tcomm = nalu_hypre_MPI_Wtime() - tcomm;
   air_time_comm += tcomm;

   NALU_HYPRE_Real t1 = nalu_hypre_MPI_Wtime();
   */

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *  First Pass: Determine the nnz of R and the max local size
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   /* nnz in diag and offd parts */
   cnt_diag = 0;
   cnt_offd = 0;
   /* maximum size of local system: will allocate space of this size */
   local_max_size = 0;

   for (i = 0; i < n_fine; i++)
   {
      NALU_HYPRE_Int MARK = i + 1;

      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      /* size of the local dense problem */
      local_size = 0;

      /* i is a C-pt, increase the number of C-pts */
      n_cpts ++;

      /* diag part of row i */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         j1 = S_diag_j[j];
         if (CF_marker[j1] >= 0)
         {
            continue;
         }
         /* j1, F: D1 */
         if (Marker_diag[j1] != MARK)
         {
            Marker_diag[j1] = MARK;
            local_size ++;
            cnt_diag ++;
         }
         /* F^2: D1-D2. Open row j1 */
         for (k = S_diag_i[j1]; k < S_diag_i[j1 + 1]; k++)
         {
            k1 = S_diag_j[k];
            /* F-pt and never seen before */
            if (CF_marker[k1] < 0 && Marker_diag[k1] != MARK)
            {
               Marker_diag[k1] = MARK;
               local_size ++;
               cnt_diag ++;
            }
         }
         /* F^2: D1-O2. Open row j1 */
         for (k = S_offd_i[j1]; k < S_offd_i[j1 + 1]; k++)
         {
            k1 = S_offd_j[k];

            if (CF_marker_offd[k1] < 0)
            {
               /* map to FF2_offd */
               k2 = Mapper_offd_A[k1];

               /* this mapping must be successful */
               nalu_hypre_assert(k2 >= 0 && k2 < FF2_offd_len);

               /* an F-pt and never seen before */
               if (Marker_FF2_offd[k2] != MARK)
               {
                  Marker_FF2_offd[k2] = MARK;
                  local_size ++;
                  cnt_offd ++;
               }
            }
         }
      }

      /* offd part of row i */
      for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
      {
         j1 = S_offd_j[j];

         if (CF_marker_offd[j1] >= 0)
         {
            continue;
         }

         /* map to FF2_offd */
         j2 = Mapper_offd_A[j1];

         /* this mapping must be successful */
         nalu_hypre_assert(j2 >= 0 && j2 < FF2_offd_len);

         /* j1, F: O1 */
         if (Marker_FF2_offd[j2] != MARK)
         {
            Marker_FF2_offd[j2] = MARK;
            local_size ++;
            cnt_offd ++;
         }

         /* F^2: O1-D2, O1-O2 */
         /* row j1 is an external row. check recv_SF for strong F-neighbors  */
         for (k = recv_SF_i[j1]; k < recv_SF_i[j1 + 1]; k++)
         {
            /* k1: global index */
            big_k1 = recv_SF_j[k];
            /* if big_k1 is in the diag part */
            if (big_k1 >= col_start && big_k1 < col_end)
            {
               k3 = (NALU_HYPRE_Int)(big_k1 - col_start);
               nalu_hypre_assert(CF_marker[k3] < 0);
               if (Marker_diag[k3] != MARK)
               {
                  Marker_diag[k3] = MARK;
                  local_size ++;
                  cnt_diag ++;
               }
            }
            else /* k1 is in the offd part */
            {
               /* index in recv_SF_offd_list */
               k2 = recv_SF_j2[k];

               if (AIR1_5 && k2 == -1)
               {
                  continue;
               }

               nalu_hypre_assert(recv_SF_offd_list[k2] == big_k1);

               /* map to FF2_offd */
               k3 = Mapper_recv_SF_offd_list[k2];

               /* this mapping must be successful */
               nalu_hypre_assert(k3 >= 0 && k3 < FF2_offd_len);

               if (Marker_FF2_offd[k3] != MARK)
               {
                  Marker_FF2_offd[k3] = MARK;
                  local_size ++;
                  cnt_offd ++;
               }
            }
         }
      }

      /* keep ths max size */
      local_max_size = nalu_hypre_max(local_max_size, local_size);
   } /* for (i=0,...) */

   /*
   t1 = nalu_hypre_MPI_Wtime() - t1;
   air_time1 += t1;
   */

   /* this is because of the indentity matrix in C part
    * each C-pt has an entry 1.0 */
   cnt_diag += n_cpts;

   nnz_diag = cnt_diag;
   nnz_offd = cnt_offd;

   /*------------- allocate arrays */
   R_diag_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_cpts + 1, NALU_HYPRE_MEMORY_HOST);
   R_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_diag, NALU_HYPRE_MEMORY_HOST);
   R_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nnz_diag, NALU_HYPRE_MEMORY_HOST);

   /* not in ``if num_procs > 1'',
    * allocation needed even for empty CSR */
   R_offd_i    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_cpts + 1, NALU_HYPRE_MEMORY_HOST);
   R_offd_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_offd, NALU_HYPRE_MEMORY_HOST);
   R_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nnz_offd, NALU_HYPRE_MEMORY_HOST);

   /* redundant */
   R_diag_i[0] = 0;
   R_offd_i[0] = 0;

   /* reset counters */
   cnt_diag = 0;
   cnt_offd = 0;

   /* RESET marker arrays */
   for (i = 0; i < n_fine; i++)
   {
      Marker_diag[i] = -1;
   }
   Marker_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < FF2_offd_len; i++)
   {
      Marker_FF2_offd[i] = -1;
   }
   Marker_FF2_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, FF2_offd_len, NALU_HYPRE_MEMORY_HOST);

   // TODO bs : what is this for? Should we remove?
   //for (i = 0; i < num_cols_A_offd; i++)
   //{
   //   Marker_offd[i] = -1;
   //}
   //for (i = 0; i < recv_SF_offd_list_len; i++)
   //{
   //   Marker_recv_SF_list[i] = -1;
   //}
   //printf("AIR: max local dense solve size %d\n", local_max_size);

   // Allocate the rhs and dense local matrix in column-major form (for LAPACK)
   DAi = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, local_max_size * local_max_size, NALU_HYPRE_MEMORY_HOST);
   Dbi = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, local_max_size, NALU_HYPRE_MEMORY_HOST);
   Dxi = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, local_max_size, NALU_HYPRE_MEMORY_HOST);
   Ipi = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_max_size, NALU_HYPRE_MEMORY_HOST); // pivot matrix

   // Allocate memory for GMRES if it will be used
   NALU_HYPRE_Int kdim_max = nalu_hypre_min(gmresAi_maxit, local_max_size);
   if (gmres_switch < local_max_size)
   {
      nalu_hypre_fgmresT(local_max_size, NULL, NULL, 0.0, kdim_max, NULL, NULL, NULL, -1);
   }

#if AIR_DEBUG
   /* FOR DEBUG */
   TMPA = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, local_max_size * local_max_size, NALU_HYPRE_MEMORY_HOST);
   TMPb = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, local_max_size, NALU_HYPRE_MEMORY_HOST);
   TMPd = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, local_max_size, NALU_HYPRE_MEMORY_HOST);
#endif

   /*- - - - - - - - - - - - - - - - - - - - - - - - -
    * space to save row indices of the local problem,
    * if diag, save the local indices,
    * if offd, save the indices in FF2_offd,
    *          since we will use it to access A_offd_FF2
    *- - - - - - - - - - - - - - - - - - - - - - - - - */
   RRi = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_max_size, NALU_HYPRE_MEMORY_HOST);
   /* indicators for RRi of being local (0) or offd (1) */
   KKi = nalu_hypre_CTAlloc(NALU_HYPRE_Int, local_max_size, NALU_HYPRE_MEMORY_HOST);

   /*
   NALU_HYPRE_Real t2 = nalu_hypre_MPI_Wtime();
   */

   /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    *                        Second Pass: Populate R
    *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   for (i = 0, ic = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      Marker_diag_count = 0;
      Marker_FF2_offd_count = 0;

      /* size of Ai, bi */
      local_size = 0;

      /* Access matrices for the First time, mark the points we want */
      /* diag part of row i */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         j1 = S_diag_j[j];
         if (CF_marker[j1] >= 0)
         {
            continue;
         }
         /* j1, F: D1 */
         if (Marker_diag[j1] == -1)
         {
            RRi[local_size] = j1;
            KKi[local_size] = 0;
            Marker_diag_j[Marker_diag_count++] = j1;
            Marker_diag[j1] = local_size ++;
         }
         /* F^2: D1-D2. Open row j1 */
         for (k = S_diag_i[j1]; k < S_diag_i[j1 + 1]; k++)
         {
            k1 = S_diag_j[k];
            /* F-pt and never seen before */
            if (CF_marker[k1] < 0 && Marker_diag[k1] == -1)
            {
               RRi[local_size] = k1;
               KKi[local_size] = 0;
               Marker_diag_j[Marker_diag_count++] = k1;
               Marker_diag[k1] = local_size ++;
            }
         }
         /* F^2: D1-O2. Open row j1 */
         for (k = S_offd_i[j1]; k < S_offd_i[j1 + 1]; k++)
         {
            k1 = S_offd_j[k];

            if (CF_marker_offd[k1] < 0)
            {
               /* map to FF2_offd */
               k2 = Mapper_offd_A[k1];

               /* this mapping must be successful */
               nalu_hypre_assert(k2 >= 0 && k2 < FF2_offd_len);

               /* an F-pt and never seen before */
               if (Marker_FF2_offd[k2] == -1)
               {
                  /* NOTE: we save this mapped index */
                  RRi[local_size] = k2;
                  KKi[local_size] = 1;
                  Marker_FF2_offd_j[Marker_FF2_offd_count++] = k2;
                  Marker_FF2_offd[k2] = local_size ++;
               }
            }
         }
      }

      /* offd part of row i */
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            j1 = S_offd_j[j];

            if (CF_marker_offd[j1] >= 0)
            {
               continue;
            }

            /* map to FF2_offd */
            j2 = Mapper_offd_A[j1];

            /* this mapping must be successful */
            nalu_hypre_assert(j2 >= 0 && j2 < FF2_offd_len);

            /* j1, F: O1 */
            if (Marker_FF2_offd[j2] == -1)
            {
               /* NOTE: we save this mapped index */
               RRi[local_size] = j2;
               KKi[local_size] = 1;
               Marker_FF2_offd_j[Marker_FF2_offd_count++] = j2;
               Marker_FF2_offd[j2] = local_size ++;
            }

            /* F^2: O1-D2, O1-O2 */
            /* row j1 is an external row. check recv_SF for strong F-neighbors  */
            for (k = recv_SF_i[j1]; k < recv_SF_i[j1 + 1]; k++)
            {
               /* k1: global index */
               big_k1 = recv_SF_j[k];
               /* if big_k1 is in the diag part */
               if (big_k1 >= col_start && big_k1 < col_end)
               {
                  k3 = (NALU_HYPRE_Int)(big_k1 - col_start);

                  nalu_hypre_assert(CF_marker[k3] < 0);

                  if (Marker_diag[k3] == -1)
                  {
                     RRi[local_size] = k3;
                     KKi[local_size] = 0;
                     Marker_diag_j[Marker_diag_count++] = k3;
                     Marker_diag[k3] = local_size ++;
                  }
               }
               else /* k1 is in the offd part */
               {
                  /* index in recv_SF_offd_list */
                  k2 = recv_SF_j2[k];

                  if (AIR1_5 && k2 == -1)
                  {
                     continue;
                  }

                  nalu_hypre_assert(recv_SF_offd_list[k2] == big_k1);

                  /* map to FF2_offd */
                  k3 = Mapper_recv_SF_offd_list[k2];

                  /* this mapping must be successful */
                  nalu_hypre_assert(k3 >= 0 && k3 < FF2_offd_len);

                  if (Marker_FF2_offd[k3] == -1)
                  {
                     /* NOTE: we save this mapped index */
                     RRi[local_size] = k3;
                     KKi[local_size] = 1;
                     Marker_FF2_offd_j[Marker_FF2_offd_count++] = k3;
                     Marker_FF2_offd[k3] = local_size ++;
                  }
               }
            }
         }
      }

      nalu_hypre_assert(local_size <= local_max_size);

      /* Second, copy values to local system: Ai and bi from A */
      /* now we have marked all rows/cols we want. next we extract the entries
       * we need from these rows and put them in Ai and bi*/

      /* clear DAi and bi */
      memset(DAi, 0, local_size * local_size * sizeof(NALU_HYPRE_Complex));
      memset(Dxi, 0, local_size * sizeof(NALU_HYPRE_Complex));
      memset(Dbi, 0, local_size * sizeof(NALU_HYPRE_Complex));


      /* we will populate Ai row-by-row */
      for (rr = 0; rr < local_size; rr++)
      {
         /* row index */
         i1 = RRi[rr];
         /* diag-offd indicator */
         i2 = KKi[rr];

         if (i2)  /* i2 == 1, i1 is an offd row */
         {
            /* open row i1, a remote row */
            for (j = nalu_hypre_CSRMatrixI(A_offd_FF2)[i1]; j < nalu_hypre_CSRMatrixI(A_offd_FF2)[i1 + 1]; j++)
            {
               /* big_j1 is a global index */
               big_j1 = nalu_hypre_CSRMatrixBigJ(A_offd_FF2)[j];

               /* if big_j1 is in the diag part */
               if (big_j1 >= col_start && big_j1 < col_end)
               {
                  j2 = (NALU_HYPRE_Int)(big_j1 - col_start);
                  /* if this col is marked with its local dense id */
                  if ((cc = Marker_diag[j2]) >= 0)
                  {
                     nalu_hypre_assert(CF_marker[j2] < 0);
                     /* copy the value */
                     /* rr and cc: local dense ids */
                     NALU_HYPRE_Complex vv = nalu_hypre_CSRMatrixData(A_offd_FF2)[j];
                     DAi[rr + cc * local_size] = vv;

                  }
               }
               else
               {
                  /* big_j1 is in offd part, search it in FF2_offd */
                  j2 =  nalu_hypre_BigBinarySearch(FF2_offd, big_j1, FF2_offd_len);
                  /* if found */
                  if (j2 > -1)
                  {
                     /* if this col is marked with its local dense id */
                     if ((cc = Marker_FF2_offd[j2]) >= 0)
                     {
                        /* copy the value */
                        /* rr and cc: local dense ids */
                        NALU_HYPRE_Complex vv = nalu_hypre_CSRMatrixData(A_offd_FF2)[j];
                        DAi[rr + cc * local_size] = vv;
                     }
                  }
               }
            }
         }
         else /* i2 == 0, i1 is a local row */
         {
            /* open row i1, a local row */
            for (j = A_diag_i[i1]; j < A_diag_i[i1 + 1]; j++)
            {
               /* j1 is a local index */
               j1 = A_diag_j[j];
               /* if this col is marked with its local dense id */
               if ((cc = Marker_diag[j1]) >= 0)
               {
                  nalu_hypre_assert(CF_marker[j1] < 0);

                  /* copy the value */
                  /* rr and cc: local dense ids */
                  NALU_HYPRE_Complex vv = A_diag_a[j];
                  DAi[rr + cc * local_size] = vv;

               }
            }

            if (num_procs > 1)
            {
               for (j = A_offd_i[i1]; j < A_offd_i[i1 + 1]; j++)
               {
                  j1 = A_offd_j[j];
                  /* map to FF2_offd */
                  j2 = Mapper_offd_A[j1];
                  /* if found */
                  if (j2 > -1)
                  {
                     /* if this col is marked with its local dense id */
                     if ((cc = Marker_FF2_offd[j2]) >= 0)
                     {
                        nalu_hypre_assert(CF_marker_offd[j1] < 0);
                        /* copy the value */
                        /* rr and cc: local dense ids */
                        NALU_HYPRE_Complex vv = A_offd_a[j];
                        DAi[rr + cc * local_size] = vv;

                     }
                  }
               }
            }
         }
         /* done with row rr */
      }

      /* TODO bs: remove?
      {
         char Buf[4096];
         char Buf2[4096];
         nalu_hypre_MPI_Status stat;
         nalu_hypre_sprintf(Buf, "size %d\n", local_size);
         NALU_HYPRE_Int ii, jj;
         for (ii = 0; ii < local_size; ii++)
         {
            for (jj = 0; jj < local_size; jj++)
            {
               nalu_hypre_sprintf(Buf+strlen(Buf), "% .1f ", DAi[ii + jj * local_size]);
            }
            nalu_hypre_sprintf(Buf+strlen(Buf), "\n");
         }
         nalu_hypre_sprintf(Buf+strlen(Buf), "\n");

         if (my_id)
         {
            nalu_hypre_MPI_Send(Buf, 4096, nalu_hypre_MPI_CHAR, 0, 0, nalu_hypre_MPI_COMM_WORLD);
         }

         if (my_id == 0)
         {
            nalu_hypre_fprintf(stdout, "%s\n", Buf);

            for (i6 = 1; i6 < num_procs; i6++)
            {
               nalu_hypre_MPI_Recv(Buf2, 4096, nalu_hypre_MPI_CHAR, i6, 0, nalu_hypre_MPI_COMM_WORLD, &stat);
               nalu_hypre_fprintf(stdout, "%s\n", Buf2);
            }
         }
      }
      */

      /* rhs bi: entries from row i of A */
      rr = 0;
      /* diag part */
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         i1 = A_diag_j[j];
         if ((cc = Marker_diag[i1]) >= 0)
         {
            nalu_hypre_assert(i1 == RRi[cc] && KKi[cc] == 0);
            /* Note the sign change */
            Dbi[cc] = -A_diag_a[j];
            rr++;
         }
      }

      /* if parallel, offd part */
      if (num_procs > 1)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            i1 = A_offd_j[j];
            i2 = Mapper_offd_A[i1];
            if (i2 > -1)
            {
               if ((cc = Marker_FF2_offd[i2]) >= 0)
               {
                  nalu_hypre_assert(i2 == RRi[cc] && KKi[cc] == 1);
                  /* Note the sign change */
                  Dbi[cc] = -A_offd_a[j];
                  rr++;
               }
            }
         }
      }

      nalu_hypre_assert(rr <= local_size);

      /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       * We have Ai and bi built. Solve the linear system by:
       *    - forward solve for triangular matrix
       *    - LU factorization (LAPACK) for local_size <= gmres_switch
       *    - Dense GMRES for local_size > gmres_switch
       *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
      Aisol_method = local_size <= gmres_switch ? 'L' : 'G';
      if (local_size > 0)
      {
         if (is_triangular)
         {
            nalu_hypre_ordered_GS(DAi, Dbi, Dxi, local_size);
#if AIR_DEBUG
            NALU_HYPRE_Real alp = -1.0, err;
            colmaj_mvT(DAi, Dxi, TMPd, local_size);
            nalu_hypre_daxpy(&local_size, &alp, Dbi, &ione, TMPd, &ione);
            err = nalu_hypre_dnrm2(&local_size, TMPd, &ione);
            if (err > 1e-8)
            {
               nalu_hypre_printf("triangular solve res: %e\n", err);
               exit(0);
            }
#endif
         }
         // Solve using LAPACK and LU factorization
         else if (Aisol_method == 'L')
         {
#if AIR_DEBUG
            memcpy(TMPA, DAi, local_size * local_size * sizeof(NALU_HYPRE_Complex));
            memcpy(TMPb, Dbi, local_size * sizeof(NALU_HYPRE_Complex));
#endif
            nalu_hypre_dgetrf(&local_size, &local_size, DAi, &local_size, Ipi,
                         &lapack_info);

            nalu_hypre_assert(lapack_info == 0);

            if (lapack_info == 0)
            {
               /* solve A_i^T x_i = b_i,
                * solution is saved in b_i on return */
               nalu_hypre_dgetrs(&charT, &local_size, &ione, DAi, &local_size,
                            Ipi, Dbi, &local_size, &lapack_info);
               nalu_hypre_assert(lapack_info == 0);
            }
#if AIR_DEBUG
            NALU_HYPRE_Real alp = 1.0, bet = 0.0, err;
            nalu_hypre_dgemv(&charT, &local_size, &local_size, &alp, TMPA, &local_size, Dbi,
                        &ione, &bet, TMPd, &ione);
            alp = -1.0;
            nalu_hypre_daxpy(&local_size, &alp, TMPb, &ione, TMPd, &ione);
            err = nalu_hypre_dnrm2(&local_size, TMPd, &ione);
            if (err > 1e-8)
            {
               nalu_hypre_printf("dense: local res norm %e\n", err);
               exit(0);
            }
#endif
         }
         // Solve by GMRES
         else
         {
            NALU_HYPRE_Real gmresAi_res;
            NALU_HYPRE_Int  gmresAi_niter;
            NALU_HYPRE_Int kdim = nalu_hypre_min(gmresAi_maxit, local_size);

            nalu_hypre_fgmresT(local_size, DAi, Dbi, gmresAi_tol, kdim, Dxi,
                          &gmresAi_res, &gmresAi_niter, 0);

            if (gmresAi_res > gmresAi_tol)
            {
               nalu_hypre_printf("gmres/jacobi not converge to %e: final_res %e\n", gmresAi_tol, gmresAi_res);
            }

#if AIR_DEBUG
            NALU_HYPRE_Real err, nrmb;
            colmaj_mvT(DAi, Dxi, TMPd, local_size);
            NALU_HYPRE_Real alp = -1.0;
            nrmb = nalu_hypre_dnrm2(&local_size, Dbi, &ione);
            nalu_hypre_daxpy(&local_size, &alp, Dbi, &ione, TMPd, &ione);
            err = nalu_hypre_dnrm2(&local_size, TMPd, &ione);
            if (err / nrmb > gmresAi_tol)
            {
               nalu_hypre_printf("GMRES/Jacobi: res norm %e, nrmb %e, relative %e\n", err, nrmb, err / nrmb);
               nalu_hypre_printf("GMRES/Jacobi: relative %e\n", gmresAi_res);
               exit(0);
            }
#endif
         }
      }

      NALU_HYPRE_Complex *Soli = (is_triangular || (Aisol_method == 'G')) ? Dxi : Dbi;

      /*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       * Now we are ready to fill this row of R
       *- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
      for (rr = 0; rr < local_size; rr++)
      {
         /* row index */
         i1 = RRi[rr];
         /* diag-offd indicator */
         i2 = KKi[rr];

         if (i2) /* offd */
         {
            nalu_hypre_assert(Marker_FF2_offd[i1] == rr);

            /* col idx: use the index in FF2_offd,
             * and you will see why later (very soon!) */
            R_offd_j[cnt_offd] = i1;
            /* copy the value */
            R_offd_data[cnt_offd++] = Soli[rr];
         }
         else /* diag */
         {
            nalu_hypre_assert(Marker_diag[i1] == rr);

            /* col idx: use local index i1 */
            R_diag_j[cnt_diag] = i1;
            /* copy the value */
            R_diag_data[cnt_diag++] = Soli[rr];
         }
      }

      /* don't forget the identity to this row */
      /* global col idx of this entry is ``col_start + i'' */
      R_diag_j[cnt_diag] = i;
      R_diag_data[cnt_diag++] = 1.0;

      /* row ptr of the next row */
      R_diag_i[ic + 1] = cnt_diag;

      R_offd_i[ic + 1] = cnt_offd;

      /* RESET marker arrays */
      for (j = 0; j < Marker_diag_count; j++)
      {
         Marker_diag[Marker_diag_j[j]] = -1;
      }

      for (j = 0; j < Marker_FF2_offd_count; j++)
      {
         Marker_FF2_offd[Marker_FF2_offd_j[j]] = -1;
      }

      /* next C-pt */
      ic++;
   } /* outermost loop, for (i=0,...), for each C-pt find restriction */

   /*
   nalu_hypre_MPI_Barrier(comm);
   t2 = nalu_hypre_MPI_Wtime() - t2;
   air_time2 += t2;
   */

   nalu_hypre_assert(ic == n_cpts);
   nalu_hypre_assert(cnt_diag == nnz_diag);
   nalu_hypre_assert(cnt_offd == nnz_offd);

   /*
   NALU_HYPRE_Real t3 = nalu_hypre_MPI_Wtime();
   */

   /* num of cols in the offd part of R */
   num_cols_offd_R = 0;
   /* to this point, Marker_FF2_offd should be all -1 */
   /*
   for (i = 0; i < FF2_offd_len; i++)
   {
      nalu_hypre_assert(Marker_FF2_offd[i] == - 1);
   }
   */

   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      if (Marker_FF2_offd[i1] == -1)
      {
         num_cols_offd_R++;
         Marker_FF2_offd[i1] = 1;
      }
   }

   tmp_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_R, NALU_HYPRE_MEMORY_HOST);
   col_map_offd_R = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_R, NALU_HYPRE_MEMORY_HOST);
   /* col_map_offd_R: the col indices of the offd of R
    * we first keep them be the local indices in FF2_offd [will be changed] */
   for (i = 0, i1 = 0; i < FF2_offd_len; i++)
   {
      if (Marker_FF2_offd[i] == 1)
      {
         tmp_map_offd[i1++] = i;
      }
   }

   nalu_hypre_assert(i1 == num_cols_offd_R);
   //printf("FF2_offd_len %d, num_cols_offd_R %d\n", FF2_offd_len, num_cols_offd_R);

   /* now, adjust R_offd_j to local idx w.r.t FF2_offd
    * by searching */
   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      k1 = nalu_hypre_BinarySearch(tmp_map_offd, i1, num_cols_offd_R);
      /* searching must succeed */
      nalu_hypre_assert(k1 >= 0 && k1 < num_cols_offd_R);
      /* change index */
      R_offd_j[i] = k1;
   }

   /* change col_map_offd_R to global ids [guaranteed to be sorted] */
   for (i = 0; i < num_cols_offd_R; i++)
   {
      col_map_offd_R[i] = FF2_offd[tmp_map_offd[i]];
   }

   /* Now, we should have everything of Parcsr matrix R */
   R = nalu_hypre_ParCSRMatrixCreate(comm,
                                total_global_cpts, /* global num of rows */
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A), /* global num of cols */
                                num_cpts_global, /* row_starts */
                                nalu_hypre_ParCSRMatrixRowStarts(A), /* col_starts */
                                num_cols_offd_R, /* num cols offd */
                                nnz_diag,
                                nnz_offd);

   R_diag = nalu_hypre_ParCSRMatrixDiag(R);
   nalu_hypre_CSRMatrixData(R_diag) = R_diag_data;
   nalu_hypre_CSRMatrixI(R_diag)    = R_diag_i;
   nalu_hypre_CSRMatrixJ(R_diag)    = R_diag_j;

   R_offd = nalu_hypre_ParCSRMatrixOffd(R);
   nalu_hypre_CSRMatrixData(R_offd) = R_offd_data;
   nalu_hypre_CSRMatrixI(R_offd)    = R_offd_i;
   nalu_hypre_CSRMatrixJ(R_offd)    = R_offd_j;

   nalu_hypre_ParCSRMatrixColMapOffd(R) = col_map_offd_R;

   /*
   t3 = nalu_hypre_MPI_Wtime() - t3;
   air_time3 += t3;

   NALU_HYPRE_Real t4 = nalu_hypre_MPI_Wtime();
   */

   /* create CommPkg of R */
   nalu_hypre_ParCSRMatrixAssumedPartition(R) = nalu_hypre_ParCSRMatrixAssumedPartition(A);
   nalu_hypre_ParCSRMatrixOwnsAssumedPartition(R) = 0;
   nalu_hypre_MatvecCommPkgCreate(R);

   /*
   t4 = nalu_hypre_MPI_Wtime() - t4;
   air_time4 += t4;
   */

   /* Filter small entries from R */
   if (filter_thresholdR > 0)
   {
      nalu_hypre_ParCSRMatrixDropSmallEntries(R, filter_thresholdR, -1);
   }

   *R_ptr = R;

   nalu_hypre_TFree(tmp_map_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dof_func_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Marker_diag, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_buf_i, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_SF_i, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_SF_i, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_SF_j, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_SF_jstarts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_SF_j, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_SF_jstarts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(comm_pkg_SF, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_SF_offd_list, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_SF_j2, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Mapper_recv_SF_offd_list, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Marker_recv_SF_offd_list, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(FF2_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_FF2_i, NALU_HYPRE_MEMORY_HOST);
   /* nalu_hypre_TFree(recv_FF2_i); */
   nalu_hypre_TFree(send_FF2_j, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_FF2_a, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_FF2_jstarts, NALU_HYPRE_MEMORY_HOST);
   /* nalu_hypre_TFree(recv_FF2_j); */
   /* nalu_hypre_TFree(recv_FF2_a); */
   nalu_hypre_CSRMatrixDestroy(A_offd_FF2);
   nalu_hypre_TFree(recv_FF2_jstarts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_MatvecCommPkgDestroy(comm_pkg_FF2_i);
   nalu_hypre_TFree(comm_pkg_FF2_j, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Mapper_offd_A, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Marker_FF2_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Marker_diag_j, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Marker_FF2_offd_j, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(DAi, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Dbi, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Dxi, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Ipi, NALU_HYPRE_MEMORY_HOST);
#if AIR_DEBUG
   nalu_hypre_TFree(TMPA, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(TMPb, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(TMPd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_SeqVectorDestroy(tmpv);
#endif
   nalu_hypre_TFree(RRi, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(KKi, NALU_HYPRE_MEMORY_HOST);

   if (gmres_switch < local_max_size)
   {
      nalu_hypre_fgmresT(0, NULL, NULL, 0.0, 0, NULL, NULL, NULL, -2);
   }

   /*
   t0 = nalu_hypre_MPI_Wtime() - t0;
   air_time0 += t0;
   */

   return 0;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildRestrNeumannAIRHost( nalu_hypre_ParCSRMatrix   *A,
                                         NALU_HYPRE_Int            *CF_marker,
                                         NALU_HYPRE_BigInt         *num_cpts_global,
                                         NALU_HYPRE_Int             num_functions,
                                         NALU_HYPRE_Int            *dof_func,
                                         NALU_HYPRE_Int             NeumannDeg,
                                         NALU_HYPRE_Real            strong_thresholdR,
                                         NALU_HYPRE_Real            filter_thresholdR,
                                         NALU_HYPRE_Int             debug_flag,
                                         nalu_hypre_ParCSRMatrix  **R_ptr)
{
   /* NALU_HYPRE_Real t0 = nalu_hypre_MPI_Wtime(); */
   MPI_Comm                 comm     = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   /* diag part of A */
   nalu_hypre_CSRMatrix *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);

   /* Restriction matrix R and CSR's */
   nalu_hypre_ParCSRMatrix *R;
   nalu_hypre_CSRMatrix *R_diag;
   nalu_hypre_CSRMatrix *R_offd;

   /* arrays */
   NALU_HYPRE_Complex   *R_diag_a;
   NALU_HYPRE_Int       *R_diag_i;
   NALU_HYPRE_Int       *R_diag_j;
   NALU_HYPRE_Complex   *R_offd_a;
   NALU_HYPRE_Int       *R_offd_i;
   NALU_HYPRE_Int       *R_offd_j;
   NALU_HYPRE_BigInt    *col_map_offd_R;

   NALU_HYPRE_Int        i, j, j1, ic,
                    num_cols_offd_R;
   NALU_HYPRE_Int        my_id, num_procs;
   NALU_HYPRE_BigInt     total_global_cpts/*, my_first_cpt*/;
   NALU_HYPRE_Int        nnz_diag, nnz_offd, cnt_diag, cnt_offd;
   NALU_HYPRE_BigInt    *send_buf_i;

   /* local size */
   NALU_HYPRE_Int n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_BigInt col_start = nalu_hypre_ParCSRMatrixFirstRowIndex(A);

   /* MPI size and rank*/
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   NALU_HYPRE_MemoryLocation memory_location_R = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   /*-------------- global number of C points and my start position */
   /*my_first_cpt = num_cpts_global[0];*/
   if (my_id == (num_procs - 1))
   {
      total_global_cpts = num_cpts_global[1];
   }
   nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   /*-------------------------------------------------------------------
    * Get the CF_marker data for the off-processor columns
    *-------------------------------------------------------------------*/
   /* CF marker for the off-diag columns */
#if 0
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

   /* number of sends (number of procs) */
   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);

   /* number of recvs (number of procs) */
   num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);

   /* number of elements to send */
   num_elems_send = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);

   /* send buffer, of size send_map_starts[num_sends]),
    * i.e., number of entries to send */
   send_buf_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_elems_send, NALU_HYPRE_MEMORY_HOST);

   /* copy CF markers of elements to send to buffer */
   for (i = 0;  i < num_elems_send; i++)
   {
      send_buf_i[i] = CF_marker[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
   }
   /* create a handle to start communication. 11: for integer */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, send_buf_i, CF_marker_offd);
   /* destroy the handle to finish communication */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /* do a similar communication for dof_func */
   if (num_functions > 1)
   {
      for (i = 0; i < num_elems_send; i++)
      {
         send_buf_i[i] = dof_func[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)];
      }
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, send_buf_i, dof_func_offd);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   }

   /* init markers to zeros */
   Marker_diag = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   Marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
#endif

   nalu_hypre_ParCSRMatrix *AFF, *ACF, *X, *X2, *Z, *Z2;
   nalu_hypre_ParCSRMatrixExtractSubmatrixFC(A, CF_marker, num_cpts_global, "FF", &AFF, strong_thresholdR);
   nalu_hypre_ParCSRMatrixExtractSubmatrixFC(A, CF_marker, num_cpts_global, "CF", &ACF, strong_thresholdR);

   /* A_FF := I - D^{-1}*A_FF */
   nalu_hypre_CSRMatrix *AFF_diag = nalu_hypre_ParCSRMatrixDiag(AFF);
   nalu_hypre_CSRMatrix *AFF_offd = nalu_hypre_ParCSRMatrixOffd(AFF);
   NALU_HYPRE_Complex   *AFF_diag_a = nalu_hypre_CSRMatrixData(AFF_diag);
   NALU_HYPRE_Int       *AFF_diag_i = nalu_hypre_CSRMatrixI(AFF_diag);
   NALU_HYPRE_Int       *AFF_diag_j = nalu_hypre_CSRMatrixJ(AFF_diag);
   NALU_HYPRE_Complex   *AFF_offd_a = nalu_hypre_CSRMatrixData(AFF_offd);
   NALU_HYPRE_Int       *AFF_offd_i = nalu_hypre_CSRMatrixI(AFF_offd);
   NALU_HYPRE_Int       *AFF_offd_j = nalu_hypre_CSRMatrixJ(AFF_offd);
   NALU_HYPRE_Int        n_fpts = nalu_hypre_CSRMatrixNumRows(AFF_diag);
   NALU_HYPRE_Int        n_cpts = n_fine - n_fpts;
   nalu_hypre_assert(n_cpts == nalu_hypre_CSRMatrixNumRows(nalu_hypre_ParCSRMatrixDiag(ACF)));

   NALU_HYPRE_Int       *Fmap = nalu_hypre_TAlloc(NALU_HYPRE_Int, n_fpts, NALU_HYPRE_MEMORY_HOST);

   /* map from F-pts to all points */
   for (i = 0, j = 0; i < n_fine; i++)
   {
      if (CF_marker[i] < 0)
      {
         Fmap[j++] = i;
      }
   }

   nalu_hypre_assert(j == n_fpts);

   NALU_HYPRE_Complex *diag_entries = nalu_hypre_TAlloc(NALU_HYPRE_Complex, n_fpts, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < n_fpts; i++)
   {
      /* find the diagonal element and store inverse */
      for (j = AFF_diag_i[i]; j < AFF_diag_i[i + 1]; j++)
      {
         if (AFF_diag_j[j] == i)
         {
            diag_entries[i] = 1.0 / AFF_diag_a[j];
            AFF_diag_a[j] = 0.0;
            break;
         }
      }

      for (j = AFF_diag_i[i]; j < AFF_diag_i[i + 1]; j++)
      {
         AFF_diag_a[j] *= -diag_entries[i];
      }
      if (num_procs > 1)
      {
         for (j = AFF_offd_i[i]; j < AFF_offd_i[i + 1]; j++)
         {
            nalu_hypre_assert( nalu_hypre_ParCSRMatrixColMapOffd(AFF)[AFF_offd_j[j]] != \
                          i + nalu_hypre_ParCSRMatrixFirstRowIndex(AFF) );

            AFF_offd_a[j] *= -diag_entries[i];
         }
      }
   }

   /* Z = Acf * (I + N + N^2 + ... + N^k] * D^{-1}
    * N = I - D^{-1} * A_FF (computed above)
    * the last D^{-1} will not be done here (but later)
    */
   if (NeumannDeg < 1)
   {
      Z = ACF;
   }
   else if (NeumannDeg == 1)
   {
      X = nalu_hypre_ParMatmul(ACF, AFF);
      nalu_hypre_ParCSRMatrixAdd(1.0, ACF, 1.0, X, &Z);
      nalu_hypre_ParCSRMatrixDestroy(X);
   }
   else
   {
      X = nalu_hypre_ParMatmul(AFF, AFF);
      nalu_hypre_ParCSRMatrixAdd(1.0, AFF, 1.0, X, &Z);
      for (i = 2; i < NeumannDeg; i++)
      {
         X2 = nalu_hypre_ParMatmul(X, AFF);
         nalu_hypre_ParCSRMatrixAdd(1.0, Z, 1.0, X2, &Z2);
         nalu_hypre_ParCSRMatrixDestroy(X);
         nalu_hypre_ParCSRMatrixDestroy(Z);
         Z = Z2;
         X = X2;
      }
      nalu_hypre_ParCSRMatrixDestroy(X);
      X = nalu_hypre_ParMatmul(ACF, Z);
      nalu_hypre_ParCSRMatrixDestroy(Z);
      nalu_hypre_ParCSRMatrixAdd(1.0, ACF, 1.0, X, &Z);
      nalu_hypre_ParCSRMatrixDestroy(X);
   }

   nalu_hypre_ParCSRMatrixDestroy(AFF);
   if (NeumannDeg >= 1)
   {
      nalu_hypre_ParCSRMatrixDestroy(ACF);
   }

   nalu_hypre_CSRMatrix *Z_diag = nalu_hypre_ParCSRMatrixDiag(Z);
   nalu_hypre_CSRMatrix *Z_offd = nalu_hypre_ParCSRMatrixOffd(Z);
   NALU_HYPRE_Complex   *Z_diag_a = nalu_hypre_CSRMatrixData(Z_diag);
   NALU_HYPRE_Int       *Z_diag_i = nalu_hypre_CSRMatrixI(Z_diag);
   NALU_HYPRE_Int       *Z_diag_j = nalu_hypre_CSRMatrixJ(Z_diag);
   NALU_HYPRE_Complex   *Z_offd_a = nalu_hypre_CSRMatrixData(Z_offd);
   NALU_HYPRE_Int       *Z_offd_i = nalu_hypre_CSRMatrixI(Z_offd);
   NALU_HYPRE_Int       *Z_offd_j = nalu_hypre_CSRMatrixJ(Z_offd);
   NALU_HYPRE_Int        num_cols_offd_Z = nalu_hypre_CSRMatrixNumCols(Z_offd);
   /*
   NALU_HYPRE_BigInt       *col_map_offd_Z  = nalu_hypre_ParCSRMatrixColMapOffd(Z);
   */
   /* send and recv diagonal entries (wrt Z) */
   NALU_HYPRE_Complex *diag_entries_offd = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_cols_offd_Z, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParCSRCommPkg *comm_pkg_Z = nalu_hypre_ParCSRMatrixCommPkg(Z);
   NALU_HYPRE_Int num_sends_Z = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg_Z);
   NALU_HYPRE_Int num_elems_send_Z = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg_Z, num_sends_Z);
   NALU_HYPRE_Complex *send_buf_Z = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_elems_send_Z, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_elems_send_Z; i++)
   {
      send_buf_Z[i] = diag_entries[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg_Z, i)];
   }
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg_Z, send_buf_Z, diag_entries_offd);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /* send and recv Fmap (wrt Z): global */
   NALU_HYPRE_BigInt *Fmap_offd_global = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_Z, NALU_HYPRE_MEMORY_HOST);
   send_buf_i = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_elems_send_Z, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_elems_send_Z; i++)
   {
      send_buf_i[i] = Fmap[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg_Z, i)] + col_start;
   }
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(21, comm_pkg_Z, send_buf_i, Fmap_offd_global);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   nnz_diag = nalu_hypre_CSRMatrixNumNonzeros(Z_diag) + n_cpts;
   nnz_offd = nalu_hypre_CSRMatrixNumNonzeros(Z_offd);

   /*------------- allocate arrays */
   R_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_cpts + 1, memory_location_R);
   R_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_diag, memory_location_R);
   R_diag_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nnz_diag, memory_location_R);

   /* not in ``if num_procs > 1'',
    * allocation needed even for empty CSR */
   R_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  n_cpts + 1, memory_location_R);
   R_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_offd, memory_location_R);
   R_offd_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nnz_offd, memory_location_R);

   /* redundant */
   R_diag_i[0] = 0;
   R_offd_i[0] = 0;

   /* reset counters */
   cnt_diag = 0;
   cnt_offd = 0;

   for (i = 0, ic = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      for (j = Z_diag_i[ic]; j < Z_diag_i[ic + 1]; j++)
      {
         j1 = Z_diag_j[j];
         R_diag_j[cnt_diag] = Fmap[j1];
         R_diag_a[cnt_diag++] = -Z_diag_a[j] * diag_entries[j1];
      }

      /* identity */
      R_diag_j[cnt_diag] = i;
      R_diag_a[cnt_diag++] = 1.0;

      for (j = Z_offd_i[ic]; j < Z_offd_i[ic + 1]; j++)
      {
         j1 = Z_offd_j[j];
         R_offd_j[cnt_offd] = j1;
         R_offd_a[cnt_offd++] = -Z_offd_a[j] * diag_entries_offd[j1];
      }

      R_diag_i[ic + 1] = cnt_diag;
      R_offd_i[ic + 1] = cnt_offd;

      ic++;
   }

   nalu_hypre_assert(ic == n_cpts);
   nalu_hypre_assert(cnt_diag == nnz_diag);
   nalu_hypre_assert(cnt_offd == nnz_offd);

   num_cols_offd_R = num_cols_offd_Z;
   col_map_offd_R = Fmap_offd_global;

   /* Now, we should have everything of Parcsr matrix R */
   R = nalu_hypre_ParCSRMatrixCreate(comm,
                                total_global_cpts, /* global num of rows */
                                nalu_hypre_ParCSRMatrixGlobalNumRows(A), /* global num of cols */
                                num_cpts_global, /* row_starts */
                                nalu_hypre_ParCSRMatrixRowStarts(A), /* col_starts */
                                num_cols_offd_R, /* num cols offd */
                                nnz_diag,
                                nnz_offd);

   R_diag = nalu_hypre_ParCSRMatrixDiag(R);
   nalu_hypre_CSRMatrixData(R_diag) = R_diag_a;
   nalu_hypre_CSRMatrixI(R_diag)    = R_diag_i;
   nalu_hypre_CSRMatrixJ(R_diag)    = R_diag_j;

   R_offd = nalu_hypre_ParCSRMatrixOffd(R);
   nalu_hypre_CSRMatrixData(R_offd) = R_offd_a;
   nalu_hypre_CSRMatrixI(R_offd)    = R_offd_i;
   nalu_hypre_CSRMatrixJ(R_offd)    = R_offd_j;

   nalu_hypre_ParCSRMatrixColMapOffd(R) = col_map_offd_R;

   /* create CommPkg of R */
   nalu_hypre_ParCSRMatrixAssumedPartition(R) = nalu_hypre_ParCSRMatrixAssumedPartition(A);
   nalu_hypre_ParCSRMatrixOwnsAssumedPartition(R) = 0;
   nalu_hypre_MatvecCommPkgCreate(R);

   /* Filter small entries from R */
   if (filter_thresholdR > 0)
   {
      nalu_hypre_ParCSRMatrixDropSmallEntries(R, filter_thresholdR, -1);
   }

   *R_ptr = R;

   nalu_hypre_ParCSRMatrixDestroy(Z);
   nalu_hypre_TFree(Fmap, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(diag_entries, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(diag_entries_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_buf_i, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_buf_Z, NALU_HYPRE_MEMORY_HOST);

   return 0;
}

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildRestrNeumannAIR( nalu_hypre_ParCSRMatrix   *A,
                                     NALU_HYPRE_Int            *CF_marker,
                                     NALU_HYPRE_BigInt         *num_cpts_global,
                                     NALU_HYPRE_Int             num_functions,
                                     NALU_HYPRE_Int            *dof_func,
                                     NALU_HYPRE_Int             NeumannDeg,
                                     NALU_HYPRE_Real            strong_thresholdR,
                                     NALU_HYPRE_Real            filter_thresholdR,
                                     NALU_HYPRE_Int             debug_flag,
                                     nalu_hypre_ParCSRMatrix  **R_ptr)
{
   nalu_hypre_GpuProfilingPushRange("RestrNeumannAIR");

   NALU_HYPRE_Int ierr = 0;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      ierr = nalu_hypre_BoomerAMGBuildRestrNeumannAIRDevice(A, CF_marker, num_cpts_global, num_functions,
                                                       dof_func,
                                                       NeumannDeg, strong_thresholdR, filter_thresholdR,
                                                       debug_flag, R_ptr);
   }
   else
#endif
   {
      ierr = nalu_hypre_BoomerAMGBuildRestrNeumannAIRHost(A, CF_marker, num_cpts_global, num_functions,
                                                     dof_func,
                                                     NeumannDeg, strong_thresholdR, filter_thresholdR,
                                                     debug_flag, R_ptr);
   }

   nalu_hypre_GpuProfilingPopRange();

   return ierr;
}
