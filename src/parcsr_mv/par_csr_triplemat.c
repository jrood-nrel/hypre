/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

#include "_nalu_hypre_utilities.h"
#include "../parcsr_mv/_nalu_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatMat : multiplies two ParCSRMatrices A and B and returns
 * the product in ParCSRMatrix C
 * Note that C does not own the partitionings since its row_starts
 * is owned by A and col_starts by B.
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRMatrix*
nalu_hypre_ParCSRMatMatHost( nalu_hypre_ParCSRMatrix  *A,
                        nalu_hypre_ParCSRMatrix  *B )
{
   MPI_Comm         comm = nalu_hypre_ParCSRMatrixComm(A);

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);

   NALU_HYPRE_BigInt    *row_starts_A = nalu_hypre_ParCSRMatrixRowStarts(A);
   NALU_HYPRE_Int        num_cols_diag_A = nalu_hypre_CSRMatrixNumCols(A_diag);
   NALU_HYPRE_Int        num_rows_diag_A = nalu_hypre_CSRMatrixNumRows(A_diag);

   nalu_hypre_CSRMatrix *B_diag = nalu_hypre_ParCSRMatrixDiag(B);

   nalu_hypre_CSRMatrix *B_offd = nalu_hypre_ParCSRMatrixOffd(B);
   NALU_HYPRE_BigInt    *col_map_offd_B = nalu_hypre_ParCSRMatrixColMapOffd(B);

   NALU_HYPRE_BigInt     first_col_diag_B = nalu_hypre_ParCSRMatrixFirstColDiag(B);
   NALU_HYPRE_BigInt     last_col_diag_B;
   NALU_HYPRE_BigInt    *col_starts_B = nalu_hypre_ParCSRMatrixColStarts(B);
   NALU_HYPRE_Int        num_rows_diag_B = nalu_hypre_CSRMatrixNumRows(B_diag);
   NALU_HYPRE_Int        num_cols_diag_B = nalu_hypre_CSRMatrixNumCols(B_diag);
   NALU_HYPRE_Int        num_cols_offd_B = nalu_hypre_CSRMatrixNumCols(B_offd);

   nalu_hypre_ParCSRMatrix *C;
   NALU_HYPRE_BigInt    *col_map_offd_C = NULL;
   NALU_HYPRE_Int       *map_B_to_C = NULL;

   nalu_hypre_CSRMatrix *C_diag = NULL;

   nalu_hypre_CSRMatrix *C_offd = NULL;

   NALU_HYPRE_Int        num_cols_offd_C = 0;

   nalu_hypre_CSRMatrix *Bs_ext;

   nalu_hypre_CSRMatrix *Bext_diag;

   nalu_hypre_CSRMatrix *Bext_offd;

   nalu_hypre_CSRMatrix *AB_diag;
   nalu_hypre_CSRMatrix *AB_offd;
   NALU_HYPRE_Int        AB_offd_num_nonzeros;
   NALU_HYPRE_Int       *AB_offd_j;
   nalu_hypre_CSRMatrix *ABext_diag;
   nalu_hypre_CSRMatrix *ABext_offd;

   NALU_HYPRE_BigInt     n_rows_A, n_cols_A;
   NALU_HYPRE_BigInt     n_rows_B, n_cols_B;
   NALU_HYPRE_Int        cnt, i;
   NALU_HYPRE_Int        num_procs;
   NALU_HYPRE_Int        my_id;

   n_rows_A = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
   n_rows_B = nalu_hypre_ParCSRMatrixGlobalNumRows(B);
   n_cols_B = nalu_hypre_ParCSRMatrixGlobalNumCols(B);

   if (n_cols_A != n_rows_B || num_cols_diag_A != num_rows_diag_B)
   {
      nalu_hypre_error_in_arg(1);
      nalu_hypre_printf(" Error! Incompatible matrix dimensions!\n");
      return NULL;
   }

   /*-----------------------------------------------------------------------
    *  Extract B_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product
    *-----------------------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   last_col_diag_B = first_col_diag_B + num_cols_diag_B - 1;

   if (num_procs > 1)
   {
      /*---------------------------------------------------------------------
       * If there exists no CommPkg for A, a CommPkg is generated using
       * equally load balanced partitionings within
       * nalu_hypre_ParCSRMatrixExtractBExt
       *--------------------------------------------------------------------*/
      Bs_ext = nalu_hypre_ParCSRMatrixExtractBExt(B, A, 1); /* contains communication
                                                          which should be explicitly included to allow for overlap */


      nalu_hypre_CSRMatrixSplit(Bs_ext, first_col_diag_B, last_col_diag_B, num_cols_offd_B, col_map_offd_B,
                           &num_cols_offd_C, &col_map_offd_C, &Bext_diag, &Bext_offd);

      nalu_hypre_CSRMatrixDestroy(Bs_ext);

      /* These are local and could be overlapped with communication */
      AB_diag = nalu_hypre_CSRMatrixMultiplyHost(A_diag, B_diag);
      AB_offd = nalu_hypre_CSRMatrixMultiplyHost(A_diag, B_offd);

      /* These require data from other processes */
      ABext_diag = nalu_hypre_CSRMatrixMultiplyHost(A_offd, Bext_diag);
      ABext_offd = nalu_hypre_CSRMatrixMultiplyHost(A_offd, Bext_offd);

      nalu_hypre_CSRMatrixDestroy(Bext_diag);
      nalu_hypre_CSRMatrixDestroy(Bext_offd);

      if (num_cols_offd_B)
      {
         NALU_HYPRE_Int i;
         map_B_to_C = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_B, NALU_HYPRE_MEMORY_HOST);

         cnt = 0;
         for (i = 0; i < num_cols_offd_C; i++)
         {
            if (col_map_offd_C[i] == col_map_offd_B[cnt])
            {
               map_B_to_C[cnt++] = i;
               if (cnt == num_cols_offd_B)
               {
                  break;
               }
            }
         }
      }
      AB_offd_num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(AB_offd);
      AB_offd_j = nalu_hypre_CSRMatrixJ(AB_offd);
      for (i = 0; i < AB_offd_num_nonzeros; i++)
      {
         AB_offd_j[i] = map_B_to_C[AB_offd_j[i]];
      }

      if (num_cols_offd_B)
      {
         nalu_hypre_TFree(map_B_to_C, NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_CSRMatrixNumCols(AB_diag) = num_cols_diag_B;
      nalu_hypre_CSRMatrixNumCols(ABext_diag) = num_cols_diag_B;
      nalu_hypre_CSRMatrixNumCols(AB_offd) = num_cols_offd_C;
      nalu_hypre_CSRMatrixNumCols(ABext_offd) = num_cols_offd_C;
      C_diag = nalu_hypre_CSRMatrixAdd(1.0, AB_diag, 1.0, ABext_diag);
      C_offd = nalu_hypre_CSRMatrixAdd(1.0, AB_offd, 1.0, ABext_offd);

      nalu_hypre_CSRMatrixDestroy(AB_diag);
      nalu_hypre_CSRMatrixDestroy(ABext_diag);
      nalu_hypre_CSRMatrixDestroy(AB_offd);
      nalu_hypre_CSRMatrixDestroy(ABext_offd);
   }
   else
   {
      C_diag = nalu_hypre_CSRMatrixMultiplyHost(A_diag, B_diag);
      C_offd = nalu_hypre_CSRMatrixCreate(num_rows_diag_A, 0, 0);
      nalu_hypre_CSRMatrixInitialize_v2(C_offd, 0, nalu_hypre_CSRMatrixMemoryLocation(C_diag));
   }

   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/

   C = nalu_hypre_ParCSRMatrixCreate(comm, n_rows_A, n_cols_B, row_starts_A,
                                col_starts_B, num_cols_offd_C,
                                C_diag->num_nonzeros, C_offd->num_nonzeros);

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(C));
   nalu_hypre_ParCSRMatrixDiag(C) = C_diag;

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(C));
   nalu_hypre_ParCSRMatrixOffd(C) = C_offd;

   if (num_cols_offd_C)
   {
      nalu_hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;
   }

   return C;
}

nalu_hypre_ParCSRMatrix*
nalu_hypre_ParCSRMatMat( nalu_hypre_ParCSRMatrix  *A,
                    nalu_hypre_ParCSRMatrix  *B )
{
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_GpuProfilingPushRange("Mat-Mat");
#endif

   nalu_hypre_ParCSRMatrix *C = NULL;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(A),
                                                      nalu_hypre_ParCSRMatrixMemoryLocation(B) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      C = nalu_hypre_ParCSRMatMatDevice(A, B);
   }
   else
#endif
   {
      C = nalu_hypre_ParCSRMatMatHost(A, B);
   }

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_GpuProfilingPopRange();
#endif

   return C;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRTMatMatKT : multiplies two ParCSRMatrices transpose(A) and B and returns
 * the product in ParCSRMatrix C
 * Note that C does not own the partitionings since its row_starts
 * is owned by A and col_starts by B.
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRMatrix*
nalu_hypre_ParCSRTMatMatKTHost( nalu_hypre_ParCSRMatrix  *A,
                           nalu_hypre_ParCSRMatrix  *B,
                           NALU_HYPRE_Int            keep_transpose)
{
   MPI_Comm             comm       = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg *comm_pkg_A = NULL;

   nalu_hypre_CSRMatrix *A_diag  = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix *A_offd  = nalu_hypre_ParCSRMatrixOffd(A);
   nalu_hypre_CSRMatrix *B_diag  = nalu_hypre_ParCSRMatrixDiag(B);
   nalu_hypre_CSRMatrix *B_offd  = nalu_hypre_ParCSRMatrixOffd(B);
   nalu_hypre_CSRMatrix *AT_diag = NULL;

   NALU_HYPRE_Int num_rows_diag_A  = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int num_cols_diag_A  = nalu_hypre_CSRMatrixNumCols(A_diag);
   NALU_HYPRE_Int num_rows_diag_B  = nalu_hypre_CSRMatrixNumRows(B_diag);
   NALU_HYPRE_Int num_cols_diag_B  = nalu_hypre_CSRMatrixNumCols(B_diag);
   NALU_HYPRE_Int num_cols_offd_B  = nalu_hypre_CSRMatrixNumCols(B_offd);
   NALU_HYPRE_BigInt first_col_diag_B = nalu_hypre_ParCSRMatrixFirstColDiag(B);

   NALU_HYPRE_BigInt *col_map_offd_B = nalu_hypre_ParCSRMatrixColMapOffd(B);

   NALU_HYPRE_BigInt *col_starts_A = nalu_hypre_ParCSRMatrixColStarts(A);
   NALU_HYPRE_BigInt *col_starts_B = nalu_hypre_ParCSRMatrixColStarts(B);

   nalu_hypre_ParCSRMatrix *C;
   nalu_hypre_CSRMatrix *C_diag = NULL;
   nalu_hypre_CSRMatrix *C_offd = NULL;

   NALU_HYPRE_BigInt *col_map_offd_C = NULL;
   NALU_HYPRE_Int *map_B_to_C;
   NALU_HYPRE_BigInt  first_col_diag_C;
   NALU_HYPRE_BigInt  last_col_diag_C;
   NALU_HYPRE_Int  num_cols_offd_C = 0;

   NALU_HYPRE_BigInt n_rows_A, n_cols_A;
   NALU_HYPRE_BigInt n_rows_B, n_cols_B;
   NALU_HYPRE_Int j_indx, cnt;
   NALU_HYPRE_Int num_procs, my_id;

   n_rows_A = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
   n_rows_B = nalu_hypre_ParCSRMatrixGlobalNumRows(B);
   n_cols_B = nalu_hypre_ParCSRMatrixGlobalNumCols(B);

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if (n_rows_A != n_rows_B || num_rows_diag_A != num_rows_diag_B)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, " Error! Incompatible matrix dimensions!\n");
      return NULL;
   }

   /*if (num_cols_diag_A == num_cols_diag_B) allsquare = 1;*/

   nalu_hypre_CSRMatrixTranspose(A_diag, &AT_diag, 1);

   if (num_procs == 1)
   {
      C_diag = nalu_hypre_CSRMatrixMultiplyHost(AT_diag, B_diag);
      C_offd = nalu_hypre_CSRMatrixCreate(num_cols_diag_A, 0, 0);
      nalu_hypre_CSRMatrixInitialize_v2(C_offd, 0, nalu_hypre_CSRMatrixMemoryLocation(C_diag));
      nalu_hypre_CSRMatrixNumRownnz(C_offd) = 0;
      if (keep_transpose)
      {
         A->diagT = AT_diag;
      }
      else
      {
         nalu_hypre_CSRMatrixDestroy(AT_diag);
      }
   }
   else
   {
      nalu_hypre_CSRMatrix *AT_offd    = NULL;
      nalu_hypre_CSRMatrix *C_tmp_diag = NULL;
      nalu_hypre_CSRMatrix *C_tmp_offd = NULL;
      nalu_hypre_CSRMatrix *C_int      = NULL;
      nalu_hypre_CSRMatrix *C_ext      = NULL;
      nalu_hypre_CSRMatrix *C_ext_diag = NULL;
      nalu_hypre_CSRMatrix *C_ext_offd = NULL;
      nalu_hypre_CSRMatrix *C_int_diag = NULL;
      nalu_hypre_CSRMatrix *C_int_offd = NULL;

      NALU_HYPRE_Int  i;
      NALU_HYPRE_Int *C_tmp_offd_i;
      NALU_HYPRE_Int *C_tmp_offd_j;
      NALU_HYPRE_Int *send_map_elmts_A;
      void      *request;

      nalu_hypre_CSRMatrixTranspose(A_offd, &AT_offd, 1);

      C_int_diag = nalu_hypre_CSRMatrixMultiplyHost(AT_offd, B_diag);
      C_int_offd = nalu_hypre_CSRMatrixMultiplyHost(AT_offd, B_offd);

      nalu_hypre_ParCSRMatrixDiag(B) = C_int_diag;
      nalu_hypre_ParCSRMatrixOffd(B) = C_int_offd;

      C_int = nalu_hypre_MergeDiagAndOffd(B);

      nalu_hypre_ParCSRMatrixDiag(B) = B_diag;
      nalu_hypre_ParCSRMatrixOffd(B) = B_offd;

      if (!nalu_hypre_ParCSRMatrixCommPkg(A))
      {
         nalu_hypre_MatvecCommPkgCreate(A);
      }
      comm_pkg_A = nalu_hypre_ParCSRMatrixCommPkg(A);

      /* contains communication; should be explicitly included to allow for overlap */
      nalu_hypre_ExchangeExternalRowsInit(C_int, comm_pkg_A, &request);
      C_ext = nalu_hypre_ExchangeExternalRowsWait(request);

      nalu_hypre_CSRMatrixDestroy(C_int);
      nalu_hypre_CSRMatrixDestroy(C_int_diag);
      nalu_hypre_CSRMatrixDestroy(C_int_offd);

      C_tmp_diag = nalu_hypre_CSRMatrixMultiplyHost(AT_diag, B_diag);
      C_tmp_offd = nalu_hypre_CSRMatrixMultiplyHost(AT_diag, B_offd);

      if (keep_transpose)
      {
         A->diagT = AT_diag;
      }
      else
      {
         nalu_hypre_CSRMatrixDestroy(AT_diag);
      }

      if (keep_transpose)
      {
         A->offdT = AT_offd;
      }
      else
      {
         nalu_hypre_CSRMatrixDestroy(AT_offd);
      }

      /*-----------------------------------------------------------------------
       *  Add contents of C_ext to C_tmp_diag and C_tmp_offd
       *  to obtain C_diag and C_offd
       *-----------------------------------------------------------------------*/

      /* split C_ext in local C_ext_diag and nonlocal part C_ext_offd,
         also generate new col_map_offd and adjust column indices accordingly */
      first_col_diag_C = first_col_diag_B;
      last_col_diag_C = first_col_diag_B + num_cols_diag_B - 1;

      if (C_ext)
      {
         nalu_hypre_CSRMatrixSplit(C_ext, first_col_diag_C, last_col_diag_C,
                              num_cols_offd_B, col_map_offd_B, &num_cols_offd_C, &col_map_offd_C,
                              &C_ext_diag, &C_ext_offd);

         nalu_hypre_CSRMatrixDestroy(C_ext);
         C_ext = NULL;
      }

      C_tmp_offd_i = nalu_hypre_CSRMatrixI(C_tmp_offd);
      C_tmp_offd_j = nalu_hypre_CSRMatrixJ(C_tmp_offd);

      if (num_cols_offd_B)
      {
         map_B_to_C = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_B, NALU_HYPRE_MEMORY_HOST);

         cnt = 0;
         for (i = 0; i < num_cols_offd_C; i++)
         {
            if (col_map_offd_C[i] == col_map_offd_B[cnt])
            {
               map_B_to_C[cnt++] = i;
               if (cnt == num_cols_offd_B)
               {
                  break;
               }
            }
         }
         for (i = 0; i < C_tmp_offd_i[nalu_hypre_CSRMatrixNumRows(C_tmp_offd)]; i++)
         {
            j_indx = C_tmp_offd_j[i];
            C_tmp_offd_j[i] = map_B_to_C[j_indx];
         }
         nalu_hypre_TFree(map_B_to_C, NALU_HYPRE_MEMORY_HOST);
      }

      /*-----------------------------------------------------------------------
       *  Need to compute C_diag = C_tmp_diag + C_ext_diag
       *  and  C_offd = C_tmp_offd + C_ext_offd   !!!!
       *-----------------------------------------------------------------------*/
      send_map_elmts_A = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg_A);
      C_diag = nalu_hypre_CSRMatrixAddPartial(C_tmp_diag, C_ext_diag, send_map_elmts_A);
      nalu_hypre_CSRMatrixNumCols(C_tmp_offd) = num_cols_offd_C;
      C_offd = nalu_hypre_CSRMatrixAddPartial(C_tmp_offd, C_ext_offd, send_map_elmts_A);

      nalu_hypre_CSRMatrixDestroy(C_tmp_diag);
      nalu_hypre_CSRMatrixDestroy(C_tmp_offd);
      nalu_hypre_CSRMatrixDestroy(C_ext_diag);
      nalu_hypre_CSRMatrixDestroy(C_ext_offd);
   }

   C = nalu_hypre_ParCSRMatrixCreate(comm, n_cols_A, n_cols_B, col_starts_A, col_starts_B,
                                num_cols_offd_C, C_diag->num_nonzeros, C_offd->num_nonzeros);

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(C));
   nalu_hypre_ParCSRMatrixDiag(C) = C_diag;

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(C));
   nalu_hypre_ParCSRMatrixOffd(C) = C_offd;

   nalu_hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;

   return C;
}

nalu_hypre_ParCSRMatrix*
nalu_hypre_ParCSRTMatMatKT( nalu_hypre_ParCSRMatrix  *A,
                       nalu_hypre_ParCSRMatrix  *B,
                       NALU_HYPRE_Int            keep_transpose)
{
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_GpuProfilingPushRange("Mat-T-Mat");
#endif

   nalu_hypre_ParCSRMatrix *C = NULL;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(A),
                                                      nalu_hypre_ParCSRMatrixMemoryLocation(B) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      C = nalu_hypre_ParCSRTMatMatKTDevice(A, B, keep_transpose);
   }
   else
#endif
   {
      C = nalu_hypre_ParCSRTMatMatKTHost(A, B, keep_transpose);
   }

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_GpuProfilingPopRange();
#endif

   return C;
}

nalu_hypre_ParCSRMatrix*
nalu_hypre_ParCSRTMatMat( nalu_hypre_ParCSRMatrix  *A,
                     nalu_hypre_ParCSRMatrix  *B)
{
   return nalu_hypre_ParCSRTMatMatKT( A, B, 0);
}

nalu_hypre_ParCSRMatrix*
nalu_hypre_ParCSRMatrixRAPKTHost( nalu_hypre_ParCSRMatrix *R,
                             nalu_hypre_ParCSRMatrix *A,
                             nalu_hypre_ParCSRMatrix *P,
                             NALU_HYPRE_Int           keep_transpose )
{
   MPI_Comm         comm = nalu_hypre_ParCSRMatrixComm(A);

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);

   NALU_HYPRE_BigInt    *row_starts_A = nalu_hypre_ParCSRMatrixRowStarts(A);
   NALU_HYPRE_Int        num_rows_diag_A = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int        num_cols_diag_A = nalu_hypre_CSRMatrixNumCols(A_diag);
   NALU_HYPRE_Int        num_cols_offd_A = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_CSRMatrix *P_diag = nalu_hypre_ParCSRMatrixDiag(P);

   nalu_hypre_CSRMatrix *P_offd = nalu_hypre_ParCSRMatrixOffd(P);
   NALU_HYPRE_BigInt    *col_map_offd_P = nalu_hypre_ParCSRMatrixColMapOffd(P);

   NALU_HYPRE_BigInt     first_col_diag_P = nalu_hypre_ParCSRMatrixFirstColDiag(P);
   NALU_HYPRE_BigInt    *col_starts_P = nalu_hypre_ParCSRMatrixColStarts(P);
   NALU_HYPRE_Int        num_rows_diag_P = nalu_hypre_CSRMatrixNumRows(P_diag);
   NALU_HYPRE_Int        num_cols_diag_P = nalu_hypre_CSRMatrixNumCols(P_diag);
   NALU_HYPRE_Int        num_cols_offd_P = nalu_hypre_CSRMatrixNumCols(P_offd);

   nalu_hypre_ParCSRMatrix *Q;
   NALU_HYPRE_BigInt       *col_map_offd_Q = NULL;
   NALU_HYPRE_Int          *map_P_to_Q = NULL;

   nalu_hypre_CSRMatrix *Q_diag = NULL;

   nalu_hypre_CSRMatrix *Q_offd = NULL;

   NALU_HYPRE_Int        num_cols_offd_Q = 0;

   nalu_hypre_CSRMatrix *Ps_ext;

   nalu_hypre_CSRMatrix *Pext_diag;

   nalu_hypre_CSRMatrix *Pext_offd;

   nalu_hypre_CSRMatrix *AP_diag;
   nalu_hypre_CSRMatrix *AP_offd;
   NALU_HYPRE_Int        AP_offd_num_nonzeros;
   NALU_HYPRE_Int       *AP_offd_j;
   nalu_hypre_CSRMatrix *APext_diag;
   nalu_hypre_CSRMatrix *APext_offd;

   nalu_hypre_ParCSRCommPkg *comm_pkg_R = nalu_hypre_ParCSRMatrixCommPkg(R);

   nalu_hypre_CSRMatrix *R_diag = nalu_hypre_ParCSRMatrixDiag(R);
   nalu_hypre_CSRMatrix *RT_diag = NULL;

   nalu_hypre_CSRMatrix *R_offd = nalu_hypre_ParCSRMatrixOffd(R);

   NALU_HYPRE_Int    num_rows_diag_R = nalu_hypre_CSRMatrixNumRows(R_diag);
   NALU_HYPRE_Int    num_cols_diag_R = nalu_hypre_CSRMatrixNumCols(R_diag);
   NALU_HYPRE_Int    num_cols_offd_R = nalu_hypre_CSRMatrixNumCols(R_offd);

   NALU_HYPRE_BigInt *col_starts_R = nalu_hypre_ParCSRMatrixColStarts(R);

   nalu_hypre_ParCSRMatrix *C;
   NALU_HYPRE_BigInt       *col_map_offd_C = NULL;
   NALU_HYPRE_Int          *map_Q_to_C;

   nalu_hypre_CSRMatrix *C_diag = NULL;

   NALU_HYPRE_BigInt    first_col_diag_C;
   NALU_HYPRE_BigInt    last_col_diag_C;

   nalu_hypre_CSRMatrix *C_offd = NULL;

   NALU_HYPRE_Int        num_cols_offd_C = 0;

   NALU_HYPRE_Int        j_indx;

   NALU_HYPRE_BigInt     n_rows_R, n_cols_R;
   NALU_HYPRE_Int        num_procs, my_id;
   NALU_HYPRE_BigInt     n_rows_A, n_cols_A;
   NALU_HYPRE_BigInt     n_rows_P, n_cols_P;
   NALU_HYPRE_Int        cnt, i;

   n_rows_R = nalu_hypre_ParCSRMatrixGlobalNumRows(R);
   n_cols_R = nalu_hypre_ParCSRMatrixGlobalNumCols(R);
   n_rows_A = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
   n_rows_P = nalu_hypre_ParCSRMatrixGlobalNumRows(P);
   n_cols_P = nalu_hypre_ParCSRMatrixGlobalNumCols(P);

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   if ( n_rows_R != n_rows_A || num_rows_diag_R != num_rows_diag_A ||
        n_cols_A != n_rows_P || num_cols_diag_A != num_rows_diag_P )
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, " Error! Incompatible matrix dimensions!\n");
      return NULL;
   }


   /*nalu_hypre_CSRMatrixTranspose(R_diag, &RT_diag, 1);*/

   if (num_procs > 1)
   {
      NALU_HYPRE_BigInt     last_col_diag_P;
      nalu_hypre_CSRMatrix *RT_offd = NULL;
      nalu_hypre_CSRMatrix *C_tmp_diag = NULL;
      nalu_hypre_CSRMatrix *C_tmp_offd = NULL;
      nalu_hypre_CSRMatrix *C_int = NULL;
      nalu_hypre_CSRMatrix *C_ext = NULL;
      nalu_hypre_CSRMatrix *C_ext_diag = NULL;
      nalu_hypre_CSRMatrix *C_ext_offd = NULL;
      nalu_hypre_CSRMatrix *C_int_diag = NULL;
      nalu_hypre_CSRMatrix *C_int_offd = NULL;

      NALU_HYPRE_Int   *C_tmp_offd_i;
      NALU_HYPRE_Int   *C_tmp_offd_j;

      NALU_HYPRE_Int   *send_map_elmts_R;
      void        *request;
      /*---------------------------------------------------------------------
       * If there exists no CommPkg for A, a CommPkg is generated using
       * equally load balanced partitionings within
       * nalu_hypre_ParCSRMatrixExtractBExt
       *--------------------------------------------------------------------*/
      Ps_ext = nalu_hypre_ParCSRMatrixExtractBExt(P, A, 1); /* contains communication
                                                          which should be explicitly included to allow for overlap */
      if (num_cols_offd_A)
      {
         last_col_diag_P = first_col_diag_P + num_cols_diag_P - 1;
         nalu_hypre_CSRMatrixSplit(Ps_ext, first_col_diag_P, last_col_diag_P, num_cols_offd_P, col_map_offd_P,
                              &num_cols_offd_Q, &col_map_offd_Q, &Pext_diag, &Pext_offd);
         /* These require data from other processes */
         APext_diag = nalu_hypre_CSRMatrixMultiplyHost(A_offd, Pext_diag);
         APext_offd = nalu_hypre_CSRMatrixMultiplyHost(A_offd, Pext_offd);

         nalu_hypre_CSRMatrixDestroy(Pext_diag);
         nalu_hypre_CSRMatrixDestroy(Pext_offd);
      }
      else
      {
         num_cols_offd_Q = num_cols_offd_P;
         col_map_offd_Q = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_Q, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < num_cols_offd_P; i++)
         {
            col_map_offd_Q[i] = col_map_offd_P[i];
         }
      }
      nalu_hypre_CSRMatrixDestroy(Ps_ext);
      /* These are local and could be overlapped with communication */
      AP_diag = nalu_hypre_CSRMatrixMultiplyHost(A_diag, P_diag);

      if (num_cols_offd_P)
      {
         NALU_HYPRE_Int i;
         AP_offd = nalu_hypre_CSRMatrixMultiplyHost(A_diag, P_offd);
         if (num_cols_offd_Q > num_cols_offd_P)
         {
            map_P_to_Q = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_P, NALU_HYPRE_MEMORY_HOST);

            cnt = 0;
            for (i = 0; i < num_cols_offd_Q; i++)
            {
               if (col_map_offd_Q[i] == col_map_offd_P[cnt])
               {
                  map_P_to_Q[cnt++] = i;
                  if (cnt == num_cols_offd_P)
                  {
                     break;
                  }
               }
            }
            AP_offd_num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(AP_offd);
            AP_offd_j = nalu_hypre_CSRMatrixJ(AP_offd);
            for (i = 0; i < AP_offd_num_nonzeros; i++)
            {
               AP_offd_j[i] = map_P_to_Q[AP_offd_j[i]];
            }

            nalu_hypre_TFree(map_P_to_Q, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_CSRMatrixNumCols(AP_offd) = num_cols_offd_Q;
         }
      }

      if (num_cols_offd_A) /* number of rows for Pext_diag */
      {
         Q_diag = nalu_hypre_CSRMatrixAdd(1.0, AP_diag, 1.0, APext_diag);
         nalu_hypre_CSRMatrixDestroy(AP_diag);
         nalu_hypre_CSRMatrixDestroy(APext_diag);
      }
      else
      {
         Q_diag = AP_diag;
      }

      if (num_cols_offd_P && num_cols_offd_A)
      {
         Q_offd = nalu_hypre_CSRMatrixAdd(1.0, AP_offd, 1.0, APext_offd);
         nalu_hypre_CSRMatrixDestroy(APext_offd);
         nalu_hypre_CSRMatrixDestroy(AP_offd);
      }
      else if (num_cols_offd_A)
      {
         Q_offd = APext_offd;
      }
      else if (num_cols_offd_P)
      {
         Q_offd = AP_offd;
      }
      else
      {
         Q_offd = nalu_hypre_CSRMatrixClone(A_offd, 1);
      }

      Q = nalu_hypre_ParCSRMatrixCreate(comm, n_rows_A, n_cols_P, row_starts_A,
                                   col_starts_P, num_cols_offd_Q,
                                   Q_diag->num_nonzeros, Q_offd->num_nonzeros);

      nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(Q));
      nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(Q));
      nalu_hypre_ParCSRMatrixDiag(Q) = Q_diag;
      nalu_hypre_ParCSRMatrixOffd(Q) = Q_offd;
      nalu_hypre_ParCSRMatrixColMapOffd(Q) = col_map_offd_Q;

      nalu_hypre_CSRMatrixTranspose(R_diag, &RT_diag, 1);
      C_tmp_diag = nalu_hypre_CSRMatrixMultiplyHost(RT_diag, Q_diag);
      if (num_cols_offd_Q)
      {
         C_tmp_offd = nalu_hypre_CSRMatrixMultiplyHost(RT_diag, Q_offd);
      }
      else
      {
         C_tmp_offd = nalu_hypre_CSRMatrixClone(Q_offd, 1);
         nalu_hypre_CSRMatrixNumRows(C_tmp_offd) = num_cols_diag_R;
      }

      if (keep_transpose)
      {
         R->diagT = RT_diag;
      }
      else
      {
         nalu_hypre_CSRMatrixDestroy(RT_diag);
      }

      if (num_cols_offd_R)
      {
         nalu_hypre_CSRMatrixTranspose(R_offd, &RT_offd, 1);
         C_int_diag = nalu_hypre_CSRMatrixMultiplyHost(RT_offd, Q_diag);
         C_int_offd = nalu_hypre_CSRMatrixMultiplyHost(RT_offd, Q_offd);

         nalu_hypre_ParCSRMatrixDiag(Q) = C_int_diag;
         nalu_hypre_ParCSRMatrixOffd(Q) = C_int_offd;
         C_int = nalu_hypre_MergeDiagAndOffd(Q);
         nalu_hypre_ParCSRMatrixDiag(Q) = Q_diag;
         nalu_hypre_ParCSRMatrixOffd(Q) = Q_offd;
      }
      else
      {
         C_int = nalu_hypre_CSRMatrixCreate(0, 0, 0);
         nalu_hypre_CSRMatrixInitialize(C_int);
      }

      /* contains communication; should be explicitly included to allow for overlap */
      nalu_hypre_ExchangeExternalRowsInit(C_int, comm_pkg_R, &request);
      C_ext = nalu_hypre_ExchangeExternalRowsWait(request);

      nalu_hypre_CSRMatrixDestroy(C_int);
      if (num_cols_offd_R)
      {
         nalu_hypre_CSRMatrixDestroy(C_int_diag);
         nalu_hypre_CSRMatrixDestroy(C_int_offd);
         if (keep_transpose)
         {
            R->offdT = RT_offd;
         }
         else
         {
            nalu_hypre_CSRMatrixDestroy(RT_offd);
         }
      }

      /*-----------------------------------------------------------------------
       *  Add contents of C_ext to C_tmp_diag and C_tmp_offd
       *  to obtain C_diag and C_offd
       *-----------------------------------------------------------------------*/

      /* split C_ext in local C_ext_diag and nonlocal part C_ext_offd,
         also generate new col_map_offd and adjust column indices accordingly */

      if (C_ext)
      {
         first_col_diag_C = first_col_diag_P;
         last_col_diag_C = first_col_diag_P + num_cols_diag_P - 1;

         nalu_hypre_CSRMatrixSplit(C_ext, first_col_diag_C, last_col_diag_C,
                              num_cols_offd_Q, col_map_offd_Q, &num_cols_offd_C, &col_map_offd_C,
                              &C_ext_diag, &C_ext_offd);

         nalu_hypre_CSRMatrixDestroy(C_ext);
         C_ext = NULL;
         /*if (C_ext_offd->num_nonzeros == 0) C_ext_offd->num_cols = 0;*/
      }

      if (num_cols_offd_Q && C_tmp_offd->num_cols)
      {
         C_tmp_offd_i = nalu_hypre_CSRMatrixI(C_tmp_offd);
         C_tmp_offd_j = nalu_hypre_CSRMatrixJ(C_tmp_offd);

         map_Q_to_C = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_Q, NALU_HYPRE_MEMORY_HOST);

         cnt = 0;
         for (i = 0; i < num_cols_offd_C; i++)
         {
            if (col_map_offd_C[i] == col_map_offd_Q[cnt])
            {
               map_Q_to_C[cnt++] = i;
               if (cnt == num_cols_offd_Q)
               {
                  break;
               }
            }
         }
         for (i = 0; i < C_tmp_offd_i[nalu_hypre_CSRMatrixNumRows(C_tmp_offd)]; i++)
         {
            j_indx = C_tmp_offd_j[i];
            C_tmp_offd_j[i] = map_Q_to_C[j_indx];
         }
         nalu_hypre_TFree(map_Q_to_C, NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_CSRMatrixNumCols(C_tmp_offd) = num_cols_offd_C;
      nalu_hypre_ParCSRMatrixDestroy(Q);

      /*-----------------------------------------------------------------------
       *  Need to compute C_diag = C_tmp_diag + C_ext_diag
       *  and  C_offd = C_tmp_offd + C_ext_offd   !!!!
       *-----------------------------------------------------------------------*/
      send_map_elmts_R = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg_R);
      if (C_ext_diag)
      {
         C_diag = nalu_hypre_CSRMatrixAddPartial(C_tmp_diag, C_ext_diag, send_map_elmts_R);
         nalu_hypre_CSRMatrixDestroy(C_tmp_diag);
         nalu_hypre_CSRMatrixDestroy(C_ext_diag);
      }
      else
      {
         C_diag = C_tmp_diag;
      }
      if (C_ext_offd)
      {
         C_offd = nalu_hypre_CSRMatrixAddPartial(C_tmp_offd, C_ext_offd, send_map_elmts_R);
         nalu_hypre_CSRMatrixDestroy(C_tmp_offd);
         nalu_hypre_CSRMatrixDestroy(C_ext_offd);
      }
      else
      {
         C_offd = C_tmp_offd;
      }
   }
   else
   {
      Q_diag = nalu_hypre_CSRMatrixMultiplyHost(A_diag, P_diag);
      nalu_hypre_CSRMatrixTranspose(R_diag, &RT_diag, 1);
      C_diag = nalu_hypre_CSRMatrixMultiplyHost(RT_diag, Q_diag);
      C_offd = nalu_hypre_CSRMatrixCreate(num_cols_diag_R, 0, 0);
      nalu_hypre_CSRMatrixInitialize_v2(C_offd, 0, nalu_hypre_CSRMatrixMemoryLocation(C_diag));
      if (keep_transpose)
      {
         R->diagT = RT_diag;
      }
      else
      {
         nalu_hypre_CSRMatrixDestroy(RT_diag);
      }
      nalu_hypre_CSRMatrixDestroy(Q_diag);
   }

   C = nalu_hypre_ParCSRMatrixCreate(comm, n_cols_R, n_cols_P, col_starts_R,
                                col_starts_P, num_cols_offd_C, 0, 0);

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(C));
   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(C));
   nalu_hypre_ParCSRMatrixDiag(C) = C_diag;

   if (C_offd)
   {
      nalu_hypre_ParCSRMatrixOffd(C) = C_offd;
   }
   else
   {
      C_offd = nalu_hypre_CSRMatrixCreate(num_cols_diag_R, 0, 0);
      nalu_hypre_CSRMatrixInitialize(C_offd);
      nalu_hypre_ParCSRMatrixOffd(C) = C_offd;
   }

   nalu_hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;

   if (num_procs > 1)
   {
      /* nalu_hypre_GenerateRAPCommPkg(RAP, A); */
      nalu_hypre_MatvecCommPkgCreate(C);
   }

   return C;
}

nalu_hypre_ParCSRMatrix*
nalu_hypre_ParCSRMatrixRAPKT( nalu_hypre_ParCSRMatrix  *R,
                         nalu_hypre_ParCSRMatrix  *A,
                         nalu_hypre_ParCSRMatrix  *P,
                         NALU_HYPRE_Int            keep_transpose)
{
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_GpuProfilingPushRange("TripleMat-RAP");
#endif

   nalu_hypre_ParCSRMatrix *C = NULL;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(R),
                                                      nalu_hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      C = nalu_hypre_ParCSRMatrixRAPKTDevice(R, A, P, keep_transpose);
   }
   else
#endif
   {
      C = nalu_hypre_ParCSRMatrixRAPKTHost(R, A, P, keep_transpose);
   }

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_GpuProfilingPopRange();
#endif

   return C;
}

nalu_hypre_ParCSRMatrix*
nalu_hypre_ParCSRMatrixRAP( nalu_hypre_ParCSRMatrix *R,
                       nalu_hypre_ParCSRMatrix *A,
                       nalu_hypre_ParCSRMatrix *P )
{
   return nalu_hypre_ParCSRMatrixRAPKT(R, A, P, 0);
}

/*--------------------------------------------------------------------------
 * OLD NOTES:
 * Sketch of John's code to build RAP
 *
 * Uses two integer arrays icg and ifg as marker arrays
 *
 *  icg needs to be of size n_fine; size of ia.
 *     A negative value of icg(i) indicates i is a f-point, otherwise
 *     icg(i) is the converts from fine to coarse grid orderings.
 *     Note that I belive the code assumes that if i<j and both are
 *     c-points, then icg(i) < icg(j).
 *  ifg needs to be of size n_coarse; size of irap
 *     I don't think it has meaning as either input or output.
 *
 * In the code, both the interpolation and restriction operator
 * are stored row-wise in the array b. If i is a f-point,
 * ib(i) points the row of the interpolation operator for point
 * i. If i is a c-point, ib(i) points the row of the restriction
 * operator for point i.
 *
 * In the CSR storage for rap, its guaranteed that the rows will
 * be ordered ( i.e. ic<jc -> irap(ic) < irap(jc)) but I don't
 * think there is a guarantee that the entries within a row will
 * be ordered in any way except that the diagonal entry comes first.
 *
 * As structured now, the code requires that the size of rap be
 * predicted up front. To avoid this, one could execute the code
 * twice, the first time would only keep track of icg ,ifg and ka.
 * Then you would know how much memory to allocate for rap and jrap.
 * The second time would fill in these arrays. Actually you might
 * be able to include the filling in of jrap into the first pass;
 * just overestimate its size (its an integer array) and cut it
 * back before the second time through. This would avoid some if tests
 * in the second pass.
 *
 * Questions
 *            1) parallel (PetSc) version?
 *            2) what if we don't store R row-wise and don't
 *               even want to store a copy of it in this form
 *               temporarily?
 *--------------------------------------------------------------------------*/
