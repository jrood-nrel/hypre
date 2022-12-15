/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_blas.h"
#include "_nalu_hypre_lapack.h"

#define AIR_DEBUG 0
#define EPSILON 1e-18
#define EPSIMAC 1e-16

void nalu_hypre_fgmresT(NALU_HYPRE_Int n, NALU_HYPRE_Complex *A, NALU_HYPRE_Complex *b, NALU_HYPRE_Real tol, NALU_HYPRE_Int kdim,
                   NALU_HYPRE_Complex *x, NALU_HYPRE_Real *relres, NALU_HYPRE_Int *iter, NALU_HYPRE_Int job);
void nalu_hypre_ordered_GS(const NALU_HYPRE_Complex L[], const NALU_HYPRE_Complex rhs[], NALU_HYPRE_Complex x[],
                      const NALU_HYPRE_Int n);

NALU_HYPRE_Int
nalu_hypre_BoomerAMGBuildRestrAIR( nalu_hypre_ParCSRMatrix   *A,
                              NALU_HYPRE_Int            *CF_marker,
                              nalu_hypre_ParCSRMatrix   *S,
                              NALU_HYPRE_BigInt         *num_cpts_global,
                              NALU_HYPRE_Int             num_functions,
                              NALU_HYPRE_Int            *dof_func,
                              NALU_HYPRE_Real            filter_thresholdR,
                              NALU_HYPRE_Int             debug_flag,
                              nalu_hypre_ParCSRMatrix  **R_ptr,
                              NALU_HYPRE_Int             is_triangular,
                              NALU_HYPRE_Int             gmres_switch)
{

   MPI_Comm                 comm     = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;
   /* diag part of A */
   nalu_hypre_CSRMatrix *A_diag      = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i    = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j    = nalu_hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   nalu_hypre_CSRMatrix *A_offd      = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i    = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j    = nalu_hypre_CSRMatrixJ(A_offd);

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
   NALU_HYPRE_Complex      *R_diag_data;
   NALU_HYPRE_Int       *R_diag_i;
   NALU_HYPRE_Int       *R_diag_j;
   NALU_HYPRE_Complex      *R_offd_data;
   NALU_HYPRE_Int       *R_offd_i;
   NALU_HYPRE_Int       *R_offd_j;
   NALU_HYPRE_BigInt    *col_map_offd_R = NULL;
   NALU_HYPRE_Int       *tmp_map_offd = NULL;
   /* CF marker off-diag part */
   NALU_HYPRE_Int       *CF_marker_offd = NULL;
   /* func type off-diag part */
   NALU_HYPRE_Int       *dof_func_offd  = NULL;
   /* ghost rows */
   nalu_hypre_CSRMatrix *A_ext      = NULL;
   NALU_HYPRE_Complex      *A_ext_data = NULL;
   NALU_HYPRE_Int       *A_ext_i    = NULL;
   NALU_HYPRE_BigInt    *A_ext_j    = NULL;

   NALU_HYPRE_Int        i, j, k, i1, k1, k2, rr, cc, ic, index, start,
                    local_max_size, local_size, num_cols_offd_R;

   /* LAPACK */
   NALU_HYPRE_Complex *DAi, *Dbi, *Dxi;
#if AIR_DEBUG
   NALU_HYPRE_Complex *TMPA, *TMPb, *TMPd;
#endif
   NALU_HYPRE_Int *Ipi, lapack_info, ione = 1;
   char charT = 'T';
   char Aisol_method;

   /* if the size of local system is larger than gmres_switch, use GMRES */
   NALU_HYPRE_Int gmresAi_maxit = 50;
   NALU_HYPRE_Real gmresAi_tol = 1e-3;

   NALU_HYPRE_Int my_id, num_procs;
   NALU_HYPRE_BigInt total_global_cpts/*, my_first_cpt*/;
   NALU_HYPRE_Int nnz_diag, nnz_offd, cnt_diag, cnt_offd;
   NALU_HYPRE_Int *marker_diag, *marker_offd;
   NALU_HYPRE_Int num_sends, *int_buf_data;
   /* local size, local num of C points */
   NALU_HYPRE_Int n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int n_cpts = 0;
   /* my first column range */
   NALU_HYPRE_BigInt col_start = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   NALU_HYPRE_BigInt col_end   = col_start + (NALU_HYPRE_BigInt)n_fine;

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

   /*-----------------------------------------------------------------------
    *  First Pass: Determine the nnz of R and the max local size
    *-----------------------------------------------------------------------*/
   /* nnz in diag and offd parts */
   cnt_diag = 0;
   cnt_offd = 0;
   /* maximum size of local system: will allocate space of this size */
   local_max_size = 0;
   for (i = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }
      /* local number of C-pts */
      n_cpts ++;
      /* If i is a C-point, the restriction is from the F-points that
       * strongly influence i */
      local_size = 0;
      /* loop through the diag part of S */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            cnt_diag ++;
            local_size ++;
         }
      }
      /* if parallel, loop through the offd part */
      if (num_procs > 1)
      {
         /* use this mapping to have offd indices of A */
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            i1 = S_offd_j[j];
            if (CF_marker_offd[i1] < 0)
            {
               cnt_offd ++;
               local_size ++;
            }
         }
      }
      /* keep ths max size */
      local_max_size = nalu_hypre_max(local_max_size, local_size);
   }

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

   /*----------------------------------------       .-.
    * Get the GHOST rows of A,                     (o o) boo!
    * i.e., adjacent rows to this proc             | O \
    * whose row indices are in A->col_map_offd      \   \
    *-----------------------------------------       `~~~'  */
   /* external rows of A that are needed for perform A multiplication,
    * the last arg means need data
    * the number of rows is num_cols_A_offd */
   if (num_procs > 1)
   {
      A_ext      = nalu_hypre_ParCSRMatrixExtractBExt(A, A, 1);
      A_ext_i    = nalu_hypre_CSRMatrixI(A_ext);
      A_ext_j    = nalu_hypre_CSRMatrixBigJ(A_ext);
      A_ext_data = nalu_hypre_CSRMatrixData(A_ext);
   }

   /* marker array: if this point is i's strong F neighbors
    *             >=  0: yes, and is the local dense id
    *             == -1: no */
   marker_diag = nalu_hypre_CTAlloc(NALU_HYPRE_Int, n_fine, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < n_fine; i++)
   {
      marker_diag[i] = -1;
   }
   marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_cols_A_offd; i++)
   {
      marker_offd[i] = -1;
   }

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

   /*-----------------------------------------------------------------------
    *  Second Pass: Populate R
    *-----------------------------------------------------------------------*/
   for (i = 0, ic = 0; i < n_fine; i++)
   {
      /* ignore F-points */
      if (CF_marker[i] < 0)
      {
         continue;
      }

      /* size of Ai, bi */
      local_size = 0;

      /* If i is a C-point, build the restriction, from the F-points that
       * strongly influence i
       * Access S for the first time, mark the points we want */
      /* 1: loop through the diag part of S */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            nalu_hypre_assert(marker_diag[i1] == -1);
            /* mark this point */
            marker_diag[i1] = local_size ++;
         }
      }
      /* 2: if parallel, loop through the offd part */
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            /* use this mapping to have offd indices of A */
            i1 = S_offd_j[j];
            /* F-point */
            if (CF_marker_offd[i1] < 0)
            {
               nalu_hypre_assert(marker_offd[i1] == -1);
               /* mark this point */
               marker_offd[i1] = local_size ++;
            }
         }
      }

      /* DEBUG FOR local_size == 0 */
      /*
      if (local_size == 0)
      {
         printf("my_id %d:  ", my_id);
         for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
         {
            i1 = S_diag_j[j];
            printf("%d[d, %d] ", i1, CF_marker[i1]);
         }
         printf("\n");
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         {
            i1 = S_offd_j[j];
            printf("%d[o, %d] ", i1, CF_marker_offd[i1]);
         }

         printf("\n");

         exit(0);
      }
      */

      /* Second, copy values to local system: Ai and bi from A */
      /* now we have marked all rows/cols we want. next we extract the entries
       * we need from these rows and put them in Ai and bi*/

      /* clear DAi and bi */
      memset(DAi, 0, local_size * local_size * sizeof(NALU_HYPRE_Complex));
      memset(Dxi, 0, local_size * sizeof(NALU_HYPRE_Complex));
      memset(Dbi, 0, local_size * sizeof(NALU_HYPRE_Complex));

      /* we will populate Ai, bi row-by-row
       * rr is the local dense matrix row counter */
      rr = 0;
      /* 1. diag part of row i */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         /* row i1 */
         i1 = S_diag_j[j];
         /* i1 is an F point */
         if (CF_marker[i1] < 0)
         {
            /* go through row i1 of A: a local row */
            /* diag part of row i1 */
            for (k = A_diag_i[i1]; k < A_diag_i[i1 + 1]; k++)
            {
               k1 = A_diag_j[k];
               /* if this col is marked with its local dense id */
               if ((cc = marker_diag[k1]) >= 0)
               {
                  nalu_hypre_assert(CF_marker[k1] < 0);
                  /* copy the value */
                  /* rr and cc: local dense ids */
                  DAi[rr + cc * local_size] = A_diag_data[k];
               }
            }
            /* if parallel, offd part of row i1 */
            if (num_procs > 1)
            {
               for (k = A_offd_i[i1]; k < A_offd_i[i1 + 1]; k++)
               {
                  k1 = A_offd_j[k];
                  /* if this col is marked with its local dense id */
                  if ((cc = marker_offd[k1]) >= 0)
                  {
                     nalu_hypre_assert(CF_marker_offd[k1] < 0);
                     /* copy the value */
                     /* rr and cc: local dense ids */
                     DAi[rr + cc * local_size] = A_offd_data[k];
                  }
               }
            }
            /* done with row i1 */
            rr++;
         }
      } /* for (j=...), diag part of row i done */

      /* 2. if parallel, offd part of row i. The corresponding rows are
       *    in matrix A_ext */
      if (num_procs > 1)
      {
         NALU_HYPRE_BigInt big_k1;
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            /* row i1: use this mapping to have offd indices of A */
            i1 = S_offd_j[j];
            /* if this is an F point */
            if (CF_marker_offd[i1] < 0)
            {
               /* loop through row i1 of A_ext, a global CSR matrix */
               for (k = A_ext_i[i1]; k < A_ext_i[i1 + 1]; k++)
               {
                  /* k1 is a global index! */
                  big_k1 = A_ext_j[k];
                  if (big_k1 >= col_start && big_k1 < col_end)
                  {
                     /* big_k1 is in the diag part, adjust to local index */
                     k1 = (NALU_HYPRE_Int)(big_k1 - col_start);
                     /* if this col is marked with its local dense id*/
                     if ((cc = marker_diag[k1]) >= 0)
                     {
                        nalu_hypre_assert(CF_marker[k1] < 0);
                        /* copy the value */
                        /* rr and cc: local dense ids */
                        DAi[rr + cc * local_size] = A_ext_data[k];
                     }
                  }
                  else
                  {
                     /* k1 is in the offd part
                      * search k1 in A->col_map_offd */
                     k2 = nalu_hypre_BigBinarySearch(col_map_offd_A, big_k1, num_cols_A_offd);
                     /* if found, k2 is the position of column id k1 in col_map_offd */
                     if (k2 > -1)
                     {
                        /* if this col is marked with its local dense id */
                        if ((cc = marker_offd[k2]) >= 0)
                        {
                           nalu_hypre_assert(CF_marker_offd[k2] < 0);
                           /* copy the value */
                           /* rr and cc: local dense ids */
                           DAi[rr + cc * local_size] = A_ext_data[k];
                        }
                     }
                  }
               }
               /* done with row i1 */
               rr++;
            }
         }
      }

      nalu_hypre_assert(rr == local_size);

      /* assemble rhs bi: entries from row i of A */
      rr = 0;
      /* diag part */
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         i1 = A_diag_j[j];
         if ((cc = marker_diag[i1]) >= 0)
         {
            /* this should be true but not very important
             * what does it say is that eqn order == unknown order
             * this is true, since order in A is preserved in S */
            nalu_hypre_assert(rr == cc);
            /* Note the sign change */
            Dbi[cc] = -A_diag_data[j];
            rr++;
         }
      }
      /* if parallel, offd part */
      if (num_procs > 1)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            i1 = A_offd_j[j];
            if ((cc = marker_offd[i1]) >= 0)
            {
               /* this should be true but not very important
                * what does it say is that eqn order == unknown order
                * this is true, since order in A is preserved in S */
               nalu_hypre_assert(rr == cc);
               /* Note the sign change */
               Dbi[cc] = -A_offd_data[j];
               rr++;
            }
         }
      }
      nalu_hypre_assert(rr == local_size);

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

      /* now we are ready to fill this row of R */
      /* diag part */
      rr = 0;
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            nalu_hypre_assert(marker_diag[i1] == rr);
            /* col idx: use i1, local idx  */
            R_diag_j[cnt_diag] = i1;
            /* copy the value */
            R_diag_data[cnt_diag++] = Soli[rr++];
         }
      }

      /* don't forget the identity to this row */
      /* global col idx of this entry is ``col_start + i''; */
      R_diag_j[cnt_diag] = i;
      R_diag_data[cnt_diag++] = 1.0;

      /* row ptr of the next row */
      R_diag_i[ic + 1] = cnt_diag;

      /* offd part */
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            /* use this mapping to have offd indices of A */
            i1 = S_offd_j[j];
            /* F-point */
            if (CF_marker_offd[i1] < 0)
            {
               nalu_hypre_assert(marker_offd[i1] == rr);
               /* col idx: use the local col id of A_offd,
                * and you will see why later (very soon!) */
               R_offd_j[cnt_offd] = i1;
               /* copy the value */
               R_offd_data[cnt_offd++] = Soli[rr++];
            }
         }
      }
      /* row ptr of the next row */
      R_offd_i[ic + 1] = cnt_offd;

      /* we must have copied all entries */
      nalu_hypre_assert(rr == local_size);

      /* reset markers */
      for (j = S_diag_i[i]; j < S_diag_i[i + 1]; j++)
      {
         i1 = S_diag_j[j];
         /* F point */
         if (CF_marker[i1] < 0)
         {
            nalu_hypre_assert(marker_diag[i1] >= 0);
            marker_diag[i1] = -1;
         }
      }
      if (num_procs > 1)
      {
         for (j = S_offd_i[i]; j < S_offd_i[i + 1]; j++)
         {
            /* use this mapping to have offd indices of A */
            i1 = S_offd_j[j];
            /* F-point */
            if (CF_marker_offd[i1] < 0)
            {
               nalu_hypre_assert(marker_offd[i1] >= 0);
               marker_offd[i1] = -1;
            }
         }
      }

      /* next C-pt */
      ic++;
   } /* outermost loop, for (i=0,...), for each C-pt find restriction */

   nalu_hypre_assert(ic == n_cpts);
   nalu_hypre_assert(cnt_diag == nnz_diag);
   nalu_hypre_assert(cnt_offd == nnz_offd);

   /* num of cols in the offd part of R */
   num_cols_offd_R = 0;
   /* to this point, marker_offd should be all -1 */
   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      if (marker_offd[i1] == -1)
      {
         num_cols_offd_R++;
         marker_offd[i1] = 1;
      }
   }

   /* col_map_offd_R: the col indices of the offd of R
    * we first keep them be the offd-idx of A */
   if (num_cols_offd_R)
   {
      col_map_offd_R = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_R, NALU_HYPRE_MEMORY_HOST);
      tmp_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_R, NALU_HYPRE_MEMORY_HOST);
   }
   for (i = 0, i1 = 0; i < num_cols_A_offd; i++)
   {
      if (marker_offd[i] == 1)
      {
         tmp_map_offd[i1++] = i;
      }
   }
   nalu_hypre_assert(i1 == num_cols_offd_R);

   /* now, adjust R_offd_j to local idx w.r.t col_map_offd_R
    * by searching */
   for (i = 0; i < nnz_offd; i++)
   {
      i1 = R_offd_j[i];
      k1 = nalu_hypre_BinarySearch(tmp_map_offd, i1, num_cols_offd_R);
      /* search must succeed */
      nalu_hypre_assert(k1 >= 0 && k1 < num_cols_offd_R);
      R_offd_j[i] = k1;
   }

   /* change col_map_offd_R to global ids */
   for (i = 0; i < num_cols_offd_R; i++)
   {
      col_map_offd_R[i] = col_map_offd_A[tmp_map_offd[i]];
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

   /* free workspace */
   nalu_hypre_TFree(tmp_map_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(CF_marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dof_func_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(marker_diag, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(marker_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(DAi, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Dbi, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(Dxi, NALU_HYPRE_MEMORY_HOST);
#if AIR_DEBUG
   nalu_hypre_TFree(TMPA, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(TMPb, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(TMPd, NALU_HYPRE_MEMORY_HOST);
#endif
   nalu_hypre_TFree(Ipi, NALU_HYPRE_MEMORY_HOST);
   if (num_procs > 1)
   {
      nalu_hypre_CSRMatrixDestroy(A_ext);
   }

   if (gmres_switch < local_max_size)
   {
      nalu_hypre_fgmresT(0, NULL, NULL, 0.0, 0, NULL, NULL, NULL, -2);
   }

   return 0;
}


/* Compute matvec A^Tx = y, where A is stored in column major form. */
// This can also probably be accomplished with BLAS
static inline void colmaj_mvT(NALU_HYPRE_Complex *A, NALU_HYPRE_Complex *x, NALU_HYPRE_Complex *y, NALU_HYPRE_Int n)
{
   memset(y, 0, n * sizeof(NALU_HYPRE_Complex));
   NALU_HYPRE_Int i, j;
   for (i = 0; i < n; i++)
   {
      NALU_HYPRE_Int row0 = i * n;
      for (j = 0; j < n; j++)
      {
         y[i] += x[j] * A[row0 + j];
      }
   }
}

// TODO : need to initialize and de-initialize GMRES
void nalu_hypre_fgmresT(NALU_HYPRE_Int n,
                   NALU_HYPRE_Complex *A,
                   NALU_HYPRE_Complex *b,
                   NALU_HYPRE_Real tol,
                   NALU_HYPRE_Int kdim,
                   NALU_HYPRE_Complex *x,
                   NALU_HYPRE_Real *relres,
                   NALU_HYPRE_Int *iter,
                   NALU_HYPRE_Int job)
{

   NALU_HYPRE_Int one = 1, i, j, k;
   static NALU_HYPRE_Complex *V = NULL, *Z = NULL, *H = NULL, *c = NULL, *s = NULL, *rs = NULL;
   NALU_HYPRE_Complex *v, *z, *w;
   NALU_HYPRE_Real t, normr, normr0, tolr;

   if (job == -1)
   {
      V  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, n * (kdim + 1),    NALU_HYPRE_MEMORY_HOST);
      /* Z  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, n*kdim,        NALU_HYPRE_MEMORY_HOST); */
      /* XXX NO PRECOND */
      Z = V;
      H  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, (kdim + 1) * kdim, NALU_HYPRE_MEMORY_HOST);
      c  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, kdim,          NALU_HYPRE_MEMORY_HOST);
      s  = nalu_hypre_TAlloc(NALU_HYPRE_Complex, kdim,          NALU_HYPRE_MEMORY_HOST);
      rs = nalu_hypre_TAlloc(NALU_HYPRE_Complex, kdim + 1,        NALU_HYPRE_MEMORY_HOST);
      return;
   }
   else if (job == -2)
   {
      nalu_hypre_TFree(V,  NALU_HYPRE_MEMORY_HOST);
      /* nalu_hypre_TFree(Z,  NALU_HYPRE_MEMORY_HOST); */
      Z = NULL;
      nalu_hypre_TFree(H,  NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(c,  NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(s,  NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(rs, NALU_HYPRE_MEMORY_HOST);
      return;
   }

   /* XXX: x_0 is all ZERO !!! so r0 = b */
   v = V;
   nalu_hypre_TMemcpy(v, b, NALU_HYPRE_Complex, n, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
   normr0 = sqrt(nalu_hypre_ddot(&n, v, &one, v, &one));

   if (normr0 < EPSIMAC)
   {
      return;
   }

   tolr = tol * normr0;

   rs[0] = normr0;
   t = 1.0 / normr0;
   nalu_hypre_dscal(&n, &t, v, &one);
   i = 0;
   while (i < kdim)
   {
      i++;
      // zi = M^{-1} * vi;
      v = V + (i - 1) * n;
      z = Z + (i - 1) * n;
      /* XXX NO PRECOND */
      /* memcpy(z, v, n*sizeof(NALU_HYPRE_Complex)); */
      // w = v_{i+1} = A * zi
      w = V + i * n;
      colmaj_mvT(A, z, w, n);
      // modified Gram-schmidt
      for (j = 0; j < i; j++)
      {
         v = V + j * n;
         H[j + (i - 1)*kdim] = t = nalu_hypre_ddot(&n, v, &one, w, &one);
         t = -t;
         nalu_hypre_daxpy(&n, &t, v, &one, w, &one);
      }
      H[i + (i - 1)*kdim] = t = sqrt(nalu_hypre_ddot(&n, w, &one, w, &one));
      if (fabs(t) > EPSILON)
      {
         t = 1.0 / t;
         nalu_hypre_dscal(&n, &t, w, &one);
      }
      // Least square problem of H
      for (j = 1; j < i; j++)
      {
         t = H[j - 1 + (i - 1) * kdim];
         H[j - 1 + (i - 1)*kdim] =  c[j - 1] * t + s[j - 1] * H[j + (i - 1) * kdim];
         H[j + (i - 1)*kdim]   = -s[j - 1] * t + c[j - 1] * H[j + (i - 1) * kdim];
      }
      NALU_HYPRE_Complex hii  = H[i - 1 + (i - 1) * kdim];
      NALU_HYPRE_Complex hii1 = H[i + (i - 1) * kdim];
      NALU_HYPRE_Complex gam = sqrt(hii * hii + hii1 * hii1);

      if (fabs(gam) < EPSILON)
      {
         gam = EPSIMAC;
      }
      c[i - 1] = hii / gam;
      s[i - 1] = hii1 / gam;
      rs[i]   = -s[i - 1] * rs[i - 1];
      rs[i - 1] =  c[i - 1] * rs[i - 1];
      // residue norm
      H[i - 1 + (i - 1)*kdim] = c[i - 1] * hii + s[i - 1] * hii1;
      normr = fabs(rs[i]);
      if (normr <= tolr)
      {
         break;
      }
   }

   // solve the upper triangular system
   rs[i - 1] /= H[i - 1 + (i - 1) * kdim];
   for (k = i - 2; k >= 0; k--)
   {
      for (j = k + 1; j < i; j++)
      {
         rs[k] -= H[k + j * kdim] * rs[j];
      }
      rs[k] /= H[k + k * kdim];
   }
   // get solution
   for (j = 0; j < i; j++)
   {
      z = Z + j * n;
      nalu_hypre_daxpy(&n, rs + j, z, &one, x, &one);
   }

   *relres = normr / normr0;
   *iter = i;
}


/* Ordered Gauss Seidel on A^T in column major format. Since we are
 * solving A^T, equivalent to solving A in row major format. */
void nalu_hypre_ordered_GS(const NALU_HYPRE_Complex L[],
                      const NALU_HYPRE_Complex rhs[],
                      NALU_HYPRE_Complex x[],
                      const NALU_HYPRE_Int n)
{
   // Get triangular ordering of L^T in col major as ordering of L in row major
   NALU_HYPRE_Int *ordering = nalu_hypre_TAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_dense_topo_sort(L, ordering, n, 0);

   // Ordered Gauss-Seidel iteration
   NALU_HYPRE_Int i, col;
   for (i = 0; i < n; i++)
   {
      NALU_HYPRE_Int row = ordering[i];
      NALU_HYPRE_Complex temp = rhs[row];
      for (col = 0; col < n; col++)
      {
         if (col != row)
         {
            temp -= L[row * n + col] * x[col]; // row-major
         }
      }
      NALU_HYPRE_Complex diag = L[row * n + row];
      if (fabs(diag) < 1e-12)
      {
         x[row] = 0.0;
      }
      else
      {
         x[row] = temp / diag;
      }
   }

   nalu_hypre_TFree(ordering, NALU_HYPRE_MEMORY_HOST);
}
