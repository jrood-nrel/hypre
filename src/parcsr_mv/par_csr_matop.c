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
 * nalu_hypre_ParMatmul_RowSizes:
 *
 * Computes sizes of C rows. Formerly part of nalu_hypre_ParMatmul but removed
 * so it can also be used for multiplication of Boolean matrices.
 *
 * Arrays computed: C_diag_i, C_offd_i.
 *
 * Arrays needed: (17, all NALU_HYPRE_Int*)
 *   rownnz_A,
 *   A_diag_i, A_diag_j,
 *   A_offd_i, A_offd_j,
 *   B_diag_i, B_diag_j,
 *   B_offd_i, B_offd_j,
 *   B_ext_i, B_ext_j,
 *   col_map_offd_B, col_map_offd_B,
 *   B_offd_i, B_offd_j,
 *   B_ext_i, B_ext_j.
 *
 * Scalars computed: C_diag_size, C_offd_size.
 *
 * Scalars needed:
 *   num_rownnz_A, num_rows_diag_A, num_cols_offd_A, allsquare,
 *   first_col_diag_B, num_cols_diag_B, num_cols_offd_B, num_cols_offd_C
 *--------------------------------------------------------------------------*/

void
nalu_hypre_ParMatmul_RowSizes( NALU_HYPRE_MemoryLocation memory_location,
                          NALU_HYPRE_Int **C_diag_i,
                          NALU_HYPRE_Int **C_offd_i,
                          NALU_HYPRE_Int  *rownnz_A,
                          NALU_HYPRE_Int  *A_diag_i,
                          NALU_HYPRE_Int  *A_diag_j,
                          NALU_HYPRE_Int  *A_offd_i,
                          NALU_HYPRE_Int  *A_offd_j,
                          NALU_HYPRE_Int  *B_diag_i,
                          NALU_HYPRE_Int  *B_diag_j,
                          NALU_HYPRE_Int  *B_offd_i,
                          NALU_HYPRE_Int  *B_offd_j,
                          NALU_HYPRE_Int  *B_ext_diag_i,
                          NALU_HYPRE_Int  *B_ext_diag_j,
                          NALU_HYPRE_Int  *B_ext_offd_i,
                          NALU_HYPRE_Int  *B_ext_offd_j,
                          NALU_HYPRE_Int  *map_B_to_C,
                          NALU_HYPRE_Int  *C_diag_size,
                          NALU_HYPRE_Int  *C_offd_size,
                          NALU_HYPRE_Int   num_rownnz_A,
                          NALU_HYPRE_Int   num_rows_diag_A,
                          NALU_HYPRE_Int   num_cols_offd_A,
                          NALU_HYPRE_Int   allsquare,
                          NALU_HYPRE_Int   num_cols_diag_B,
                          NALU_HYPRE_Int   num_cols_offd_B,
                          NALU_HYPRE_Int   num_cols_offd_C )
{
   NALU_HYPRE_Int *jj_count_diag_array;
   NALU_HYPRE_Int *jj_count_offd_array;

   NALU_HYPRE_Int  start_indexing = 0; /* start indexing for C_data at 0 */
   NALU_HYPRE_Int  num_threads = nalu_hypre_NumThreads();

   *C_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows_diag_A + 1, memory_location);
   *C_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows_diag_A + 1, memory_location);

   jj_count_diag_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads, NALU_HYPRE_MEMORY_HOST);
   jj_count_offd_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_threads, NALU_HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Loop over rows of A
    *-----------------------------------------------------------------------*/
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      NALU_HYPRE_Int  *B_marker = NULL;
      NALU_HYPRE_Int   jj_row_begin_diag, jj_count_diag;
      NALU_HYPRE_Int   jj_row_begin_offd, jj_count_offd;
      NALU_HYPRE_Int   i1, ii1, i2, i3, jj2, jj3;
      NALU_HYPRE_Int   size, rest, num_threads;
      NALU_HYPRE_Int   ii, ns, ne;

      num_threads = nalu_hypre_NumActiveThreads();
      size = num_rownnz_A / num_threads;
      rest = num_rownnz_A - size * num_threads;

      ii = nalu_hypre_GetThreadNum();
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
      jj_count_diag = start_indexing;
      jj_count_offd = start_indexing;

      if (num_cols_diag_B || num_cols_offd_C)
      {
         B_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_diag_B + num_cols_offd_C, NALU_HYPRE_MEMORY_HOST);
      }

      for (i1 = 0; i1 < num_cols_diag_B + num_cols_offd_C; i1++)
      {
         B_marker[i1] = -1;
      }

      for (i1 = ns; i1 < ne; i1++)
      {
         jj_row_begin_diag = jj_count_diag;
         jj_row_begin_offd = jj_count_offd;
         if (rownnz_A)
         {
            ii1 = rownnz_A[i1];
         }
         else
         {
            ii1 = i1;

            /*--------------------------------------------------------------------
             *  Set marker for diagonal entry, C_{i1,i1} (for square matrices).
             *--------------------------------------------------------------------*/

            if (allsquare)
            {
               B_marker[i1] = jj_count_diag;
               jj_count_diag++;
            }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row ii1 of A_offd.
          *-----------------------------------------------------------------*/

         if (num_cols_offd_A)
         {
            for (jj2 = A_offd_i[ii1]; jj2 < A_offd_i[ii1 + 1]; jj2++)
            {
               i2 = A_offd_j[jj2];

               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_ext.
                *-----------------------------------------------------------*/

               for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2 + 1]; jj3++)
               {
                  i3 = num_cols_diag_B + B_ext_offd_j[jj3];

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{ii1,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_offd)
                  {
                     B_marker[i3] = jj_count_offd;
                     jj_count_offd++;
                  }
               }

               for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2 + 1]; jj3++)
               {
                  i3 = B_ext_diag_j[jj3];

                  if (B_marker[i3] < jj_row_begin_diag)
                  {
                     B_marker[i3] = jj_count_diag;
                     jj_count_diag++;
                  }
               }
            }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row ii1 of A_diag.
          *-----------------------------------------------------------------*/

         for (jj2 = A_diag_i[ii1]; jj2 < A_diag_i[ii1 + 1]; jj2++)
         {
            i2 = A_diag_j[jj2];

            /*-----------------------------------------------------------
             *  Loop over entries in row i2 of B_diag.
             *-----------------------------------------------------------*/

            for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2 + 1]; jj3++)
            {
               i3 = B_diag_j[jj3];

               /*--------------------------------------------------------
                *  Check B_marker to see that C_{ii1,i3} has not already
                *  been accounted for. If it has not, mark it and increment
                *  counter.
                *--------------------------------------------------------*/

               if (B_marker[i3] < jj_row_begin_diag)
               {
                  B_marker[i3] = jj_count_diag;
                  jj_count_diag++;
               }
            }

            /*-----------------------------------------------------------
             *  Loop over entries in row i2 of B_offd.
             *-----------------------------------------------------------*/

            if (num_cols_offd_B)
            {
               for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2 + 1]; jj3++)
               {
                  i3 = num_cols_diag_B + map_B_to_C[B_offd_j[jj3]];

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{ii1,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_offd)
                  {
                     B_marker[i3] = jj_count_offd;
                     jj_count_offd++;
                  }
               }
            }
         }

         /*--------------------------------------------------------------------
          * Set C_diag_i and C_offd_i for this row.
          *--------------------------------------------------------------------*/

         (*C_diag_i)[ii1] = jj_row_begin_diag;
         (*C_offd_i)[ii1] = jj_row_begin_offd;
      }

      jj_count_diag_array[ii] = jj_count_diag;
      jj_count_offd_array[ii] = jj_count_offd;

      nalu_hypre_TFree(B_marker, NALU_HYPRE_MEMORY_HOST);
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      /* Correct diag_i and offd_i - phase 1 */
      if (ii)
      {
         jj_count_diag = jj_count_diag_array[0];
         jj_count_offd = jj_count_offd_array[0];
         for (i1 = 1; i1 < ii; i1++)
         {
            jj_count_diag += jj_count_diag_array[i1];
            jj_count_offd += jj_count_offd_array[i1];
         }

         for (i1 = ns; i1 < ne; i1++)
         {
            ii1 = rownnz_A ? rownnz_A[i1] : i1;
            (*C_diag_i)[ii1] += jj_count_diag;
            (*C_offd_i)[ii1] += jj_count_offd;
         }
      }
      else
      {
         (*C_diag_i)[num_rows_diag_A] = 0;
         (*C_offd_i)[num_rows_diag_A] = 0;
         for (i1 = 0; i1 < num_threads; i1++)
         {
            (*C_diag_i)[num_rows_diag_A] += jj_count_diag_array[i1];
            (*C_offd_i)[num_rows_diag_A] += jj_count_offd_array[i1];
         }
      }

      /* Correct diag_i and offd_i - phase 2 */
      if (rownnz_A != NULL)
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp barrier
#endif
         for (i1 = ns; i1 < (ne - 1); i1++)
         {
            for (ii1 = rownnz_A[i1] + 1; ii1 < rownnz_A[i1 + 1]; ii1++)
            {
               (*C_diag_i)[ii1] = (*C_diag_i)[rownnz_A[i1 + 1]];
               (*C_offd_i)[ii1] = (*C_offd_i)[rownnz_A[i1 + 1]];
            }
         }

         if (ii < (num_threads - 1))
         {
            for (ii1 = rownnz_A[ne - 1] + 1; ii1 < rownnz_A[ne]; ii1++)
            {
               (*C_diag_i)[ii1] = (*C_diag_i)[rownnz_A[ne]];
               (*C_offd_i)[ii1] = (*C_offd_i)[rownnz_A[ne]];
            }
         }
         else
         {
            for (ii1 = rownnz_A[ne - 1] + 1; ii1 < num_rows_diag_A; ii1++)
            {
               (*C_diag_i)[ii1] = (*C_diag_i)[num_rows_diag_A];
               (*C_offd_i)[ii1] = (*C_offd_i)[num_rows_diag_A];
            }
         }
      }
   } /* end parallel loop */

   *C_diag_size = (*C_diag_i)[num_rows_diag_A];
   *C_offd_size = (*C_offd_i)[num_rows_diag_A];

#ifdef NALU_HYPRE_DEBUG
   NALU_HYPRE_Int i;

   for (i = 0; i < num_rows_diag_A; i++)
   {
      nalu_hypre_assert((*C_diag_i)[i] <= (*C_diag_i)[i + 1]);
      nalu_hypre_assert((*C_offd_i)[i] <= (*C_offd_i)[i + 1]);
   }
#endif

   nalu_hypre_TFree(jj_count_diag_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jj_count_offd_array, NALU_HYPRE_MEMORY_HOST);

   /* End of First Pass */
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParMatmul:
 *
 * Multiplies two ParCSRMatrices A and B and returns the product in
 * ParCSRMatrix C.
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRMatrix*
nalu_hypre_ParMatmul( nalu_hypre_ParCSRMatrix  *A,
                 nalu_hypre_ParCSRMatrix  *B )
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_MATMUL] -= nalu_hypre_MPI_Wtime();
#endif

   /* ParCSRMatrix A */
   MPI_Comm            comm              = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_BigInt        nrows_A           = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   NALU_HYPRE_BigInt        ncols_A           = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
   NALU_HYPRE_BigInt       *row_starts_A      = nalu_hypre_ParCSRMatrixRowStarts(A);
   NALU_HYPRE_Int           num_rownnz_A;
   NALU_HYPRE_Int          *rownnz_A = NULL;

   /* ParCSRMatrix B */
   NALU_HYPRE_BigInt        nrows_B           = nalu_hypre_ParCSRMatrixGlobalNumRows(B);
   NALU_HYPRE_BigInt        ncols_B           = nalu_hypre_ParCSRMatrixGlobalNumCols(B);
   NALU_HYPRE_BigInt        first_col_diag_B  = nalu_hypre_ParCSRMatrixFirstColDiag(B);
   NALU_HYPRE_BigInt       *col_starts_B      = nalu_hypre_ParCSRMatrixColStarts(B);
   NALU_HYPRE_BigInt        last_col_diag_B;

   /* A_diag */
   nalu_hypre_CSRMatrix    *A_diag            = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_data       = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int          *A_diag_i          = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int          *A_diag_j          = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Int          *A_diag_ir         = nalu_hypre_CSRMatrixRownnz(A_diag);
   NALU_HYPRE_Int           num_rownnz_diag_A = nalu_hypre_CSRMatrixNumRownnz(A_diag);
   NALU_HYPRE_Int           num_rows_diag_A   = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int           num_cols_diag_A   = nalu_hypre_CSRMatrixNumCols(A_diag);

   /* A_offd */
   nalu_hypre_CSRMatrix    *A_offd            = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_data       = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int          *A_offd_i          = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int          *A_offd_j          = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Int          *A_offd_ir         = nalu_hypre_CSRMatrixRownnz(A_offd);
   NALU_HYPRE_Int           num_rownnz_offd_A = nalu_hypre_CSRMatrixNumRownnz(A_offd);
   NALU_HYPRE_Int           num_rows_offd_A   = nalu_hypre_CSRMatrixNumRows(A_offd);
   NALU_HYPRE_Int           num_cols_offd_A   = nalu_hypre_CSRMatrixNumCols(A_offd);

   /* B_diag */
   nalu_hypre_CSRMatrix    *B_diag            = nalu_hypre_ParCSRMatrixDiag(B);
   NALU_HYPRE_Complex      *B_diag_data       = nalu_hypre_CSRMatrixData(B_diag);
   NALU_HYPRE_Int          *B_diag_i          = nalu_hypre_CSRMatrixI(B_diag);
   NALU_HYPRE_Int          *B_diag_j          = nalu_hypre_CSRMatrixJ(B_diag);
   NALU_HYPRE_Int           num_rows_diag_B   = nalu_hypre_CSRMatrixNumRows(B_diag);
   NALU_HYPRE_Int           num_cols_diag_B   = nalu_hypre_CSRMatrixNumCols(B_diag);

   /* B_offd */
   nalu_hypre_CSRMatrix    *B_offd            = nalu_hypre_ParCSRMatrixOffd(B);
   NALU_HYPRE_BigInt       *col_map_offd_B    = nalu_hypre_ParCSRMatrixColMapOffd(B);
   NALU_HYPRE_Complex      *B_offd_data       = nalu_hypre_CSRMatrixData(B_offd);
   NALU_HYPRE_Int          *B_offd_i          = nalu_hypre_CSRMatrixI(B_offd);
   NALU_HYPRE_Int          *B_offd_j          = nalu_hypre_CSRMatrixJ(B_offd);
   NALU_HYPRE_Int           num_cols_offd_B   = nalu_hypre_CSRMatrixNumCols(B_offd);

   /* ParCSRMatrix C */
   nalu_hypre_ParCSRMatrix *C;
   NALU_HYPRE_BigInt       *col_map_offd_C;
   NALU_HYPRE_Int          *map_B_to_C = NULL;

   /* C_diag */
   nalu_hypre_CSRMatrix    *C_diag;
   NALU_HYPRE_Complex      *C_diag_data;
   NALU_HYPRE_Int          *C_diag_i;
   NALU_HYPRE_Int          *C_diag_j;
   NALU_HYPRE_Int           C_offd_size;
   NALU_HYPRE_Int           num_cols_offd_C = 0;

   /* C_offd */
   nalu_hypre_CSRMatrix    *C_offd;
   NALU_HYPRE_Complex      *C_offd_data = NULL;
   NALU_HYPRE_Int          *C_offd_i = NULL;
   NALU_HYPRE_Int          *C_offd_j = NULL;
   NALU_HYPRE_Int           C_diag_size;

   /* Bs_ext */
   nalu_hypre_CSRMatrix    *Bs_ext;
   NALU_HYPRE_Complex      *Bs_ext_data;
   NALU_HYPRE_Int          *Bs_ext_i;
   NALU_HYPRE_BigInt       *Bs_ext_j;
   NALU_HYPRE_Complex      *B_ext_diag_data;
   NALU_HYPRE_Int          *B_ext_diag_i;
   NALU_HYPRE_Int          *B_ext_diag_j;
   NALU_HYPRE_Int           B_ext_diag_size;
   NALU_HYPRE_Complex      *B_ext_offd_data;
   NALU_HYPRE_Int          *B_ext_offd_i;
   NALU_HYPRE_Int          *B_ext_offd_j;
   NALU_HYPRE_BigInt       *B_big_offd_j = NULL;
   NALU_HYPRE_Int           B_ext_offd_size;

   NALU_HYPRE_Int           allsquare = 0;
   NALU_HYPRE_Int           num_procs;
   NALU_HYPRE_Int          *my_diag_array;
   NALU_HYPRE_Int          *my_offd_array;
   NALU_HYPRE_Int           max_num_threads;

   NALU_HYPRE_Complex       zero = 0.0;

   NALU_HYPRE_MemoryLocation memory_location_A = nalu_hypre_ParCSRMatrixMemoryLocation(A);
   NALU_HYPRE_MemoryLocation memory_location_B = nalu_hypre_ParCSRMatrixMemoryLocation(B);

   /* RL: TODO cannot guarantee, maybe should never assert
   nalu_hypre_assert(memory_location_A == memory_location_B);
   */

   /* RL: in the case of A=H, B=D, or A=D, B=H, let C = D,
    * not sure if this is the right thing to do.
    * Also, need something like this in other places
    * TODO */
   NALU_HYPRE_MemoryLocation memory_location_C = nalu_hypre_max(memory_location_A, memory_location_B);

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   max_num_threads = nalu_hypre_NumThreads();
   my_diag_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_threads, NALU_HYPRE_MEMORY_HOST);
   my_offd_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_threads, NALU_HYPRE_MEMORY_HOST);

   if (ncols_A != nrows_B || num_cols_diag_A != num_rows_diag_B)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, " Error! Incompatible matrix dimensions!\n");

      NALU_HYPRE_ANNOTATE_FUNC_END;
      return NULL;
   }

   /* if C=A*B is square globally and locally, then C_diag should be square also */
   if ( num_rows_diag_A == num_cols_diag_B && nrows_A == ncols_B )
   {
      allsquare = 1;
   }

   /* Set rownnz of A */
   if (num_rownnz_diag_A != num_rows_diag_A &&
       num_rownnz_offd_A != num_rows_offd_A )
   {
      nalu_hypre_IntArray arr_diag;
      nalu_hypre_IntArray arr_offd;
      nalu_hypre_IntArray arr_rownnz;

      nalu_hypre_IntArrayData(&arr_diag) = A_diag_ir;
      nalu_hypre_IntArrayData(&arr_offd) = A_offd_ir;
      nalu_hypre_IntArraySize(&arr_diag) = num_rownnz_diag_A;
      nalu_hypre_IntArraySize(&arr_offd) = num_rownnz_offd_A;
      nalu_hypre_IntArrayMemoryLocation(&arr_rownnz) = memory_location_A;

      nalu_hypre_IntArrayMergeOrdered(&arr_diag, &arr_offd, &arr_rownnz);

      num_rownnz_A = nalu_hypre_IntArraySize(&arr_rownnz);
      rownnz_A     = nalu_hypre_IntArrayData(&arr_rownnz);
   }
   else
   {
      num_rownnz_A = nalu_hypre_max(num_rows_diag_A, num_rows_offd_A);
   }

   /*-----------------------------------------------------------------------
    *  Extract B_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product
    *-----------------------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX] -= nalu_hypre_MPI_Wtime();
#endif

   if (num_procs > 1)
   {
      /*---------------------------------------------------------------------
       * If there exists no CommPkg for A, a CommPkg is generated using
       * equally load balanced partitionings within
       * nalu_hypre_ParCSRMatrixExtractBExt
       *--------------------------------------------------------------------*/
      Bs_ext      = nalu_hypre_ParCSRMatrixExtractBExt(B, A, 1);
      Bs_ext_data = nalu_hypre_CSRMatrixData(Bs_ext);
      Bs_ext_i    = nalu_hypre_CSRMatrixI(Bs_ext);
      Bs_ext_j    = nalu_hypre_CSRMatrixBigJ(Bs_ext);
   }
   B_ext_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_A + 1, NALU_HYPRE_MEMORY_HOST);
   B_ext_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_A + 1, NALU_HYPRE_MEMORY_HOST);
   B_ext_diag_size = 0;
   B_ext_offd_size = 0;
   last_col_diag_B = first_col_diag_B + (NALU_HYPRE_BigInt) num_cols_diag_B - 1;

#ifdef NALU_HYPRE_CONCURRENT_HOPSCOTCH
   nalu_hypre_UnorderedBigIntSet set;

   #pragma omp parallel
   {
      NALU_HYPRE_Int size, rest, ii;
      NALU_HYPRE_Int ns, ne;
      NALU_HYPRE_Int i1, i, j;
      NALU_HYPRE_Int my_offd_size, my_diag_size;
      NALU_HYPRE_Int cnt_offd, cnt_diag;
      NALU_HYPRE_Int num_threads = nalu_hypre_NumActiveThreads();

      size = num_cols_offd_A / num_threads;
      rest = num_cols_offd_A - size * num_threads;
      ii = nalu_hypre_GetThreadNum();
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

      my_diag_size = 0;
      my_offd_size = 0;
      for (i = ns; i < ne; i++)
      {
         B_ext_diag_i[i] = my_diag_size;
         B_ext_offd_i[i] = my_offd_size;
         for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         {
            if (Bs_ext_j[j] < first_col_diag_B ||
                Bs_ext_j[j] > last_col_diag_B)
            {
               my_offd_size++;
            }
            else
            {
               my_diag_size++;
            }
         }
      }
      my_diag_array[ii] = my_diag_size;
      my_offd_array[ii] = my_offd_size;

      #pragma omp barrier

      if (ii)
      {
         my_diag_size = my_diag_array[0];
         my_offd_size = my_offd_array[0];
         for (i1 = 1; i1 < ii; i1++)
         {
            my_diag_size += my_diag_array[i1];
            my_offd_size += my_offd_array[i1];
         }

         for (i1 = ns; i1 < ne; i1++)
         {
            B_ext_diag_i[i1] += my_diag_size;
            B_ext_offd_i[i1] += my_offd_size;
         }
      }
      else
      {
         B_ext_diag_size = 0;
         B_ext_offd_size = 0;
         for (i1 = 0; i1 < num_threads; i1++)
         {
            B_ext_diag_size += my_diag_array[i1];
            B_ext_offd_size += my_offd_array[i1];
         }
         B_ext_diag_i[num_cols_offd_A] = B_ext_diag_size;
         B_ext_offd_i[num_cols_offd_A] = B_ext_offd_size;

         if (B_ext_diag_size)
         {
            B_ext_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  B_ext_diag_size, NALU_HYPRE_MEMORY_HOST);
            B_ext_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, B_ext_diag_size, NALU_HYPRE_MEMORY_HOST);
         }
         if (B_ext_offd_size)
         {
            B_ext_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, B_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
            B_big_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, B_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
            B_ext_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, B_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
         }
         nalu_hypre_UnorderedBigIntSetCreate(&set, B_ext_offd_size + num_cols_offd_B, 16 * nalu_hypre_NumThreads());
      }


      #pragma omp barrier

      cnt_offd = B_ext_offd_i[ns];
      cnt_diag = B_ext_diag_i[ns];
      for (i = ns; i < ne; i++)
      {
         for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         {
            if (Bs_ext_j[j] < first_col_diag_B ||
                Bs_ext_j[j] > last_col_diag_B)
            {
               nalu_hypre_UnorderedBigIntSetPut(&set, Bs_ext_j[j]);
               B_big_offd_j[cnt_offd] = Bs_ext_j[j];
               //Bs_ext_j[cnt_offd] = Bs_ext_j[j];
               B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
            }
            else
            {
               B_ext_diag_j[cnt_diag] = (NALU_HYPRE_Int)(Bs_ext_j[j] - first_col_diag_B);
               B_ext_diag_data[cnt_diag++] = Bs_ext_data[j];
            }
         }
      }

      NALU_HYPRE_Int i_begin, i_end;
      nalu_hypre_GetSimpleThreadPartition(&i_begin, &i_end, num_cols_offd_B);
      for (i = i_begin; i < i_end; i++)
      {
         nalu_hypre_UnorderedBigIntSetPut(&set, col_map_offd_B[i]);
      }
   } /* omp parallel */

   col_map_offd_C = nalu_hypre_UnorderedBigIntSetCopyToArray(&set, &num_cols_offd_C);
   nalu_hypre_UnorderedBigIntSetDestroy(&set);
   nalu_hypre_UnorderedBigIntMap col_map_offd_C_inverse;
   nalu_hypre_big_sort_and_create_inverse_map(col_map_offd_C,
                                         num_cols_offd_C,
                                         &col_map_offd_C,
                                         &col_map_offd_C_inverse);

   NALU_HYPRE_Int i, j;
   #pragma omp parallel for private(j) NALU_HYPRE_SMP_SCHEDULE
   for (i = 0; i < num_cols_offd_A; i++)
   {
      for (j = B_ext_offd_i[i]; j < B_ext_offd_i[i + 1]; j++)
      {
         //B_ext_offd_j[j] = nalu_hypre_UnorderedIntMapGet(&col_map_offd_C_inverse, B_ext_offd_j[j]);
         B_ext_offd_j[j] = nalu_hypre_UnorderedBigIntMapGet(&col_map_offd_C_inverse, B_big_offd_j[j]);
      }
   }

   if (num_cols_offd_C)
   {
      nalu_hypre_UnorderedBigIntMapDestroy(&col_map_offd_C_inverse);
   }

   nalu_hypre_TFree(my_diag_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(my_offd_array, NALU_HYPRE_MEMORY_HOST);

   if (num_cols_offd_B)
   {
      NALU_HYPRE_Int i;
      map_B_to_C = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_B, NALU_HYPRE_MEMORY_HOST);

      #pragma omp parallel private(i)
      {
         NALU_HYPRE_Int i_begin, i_end;
         nalu_hypre_GetSimpleThreadPartition(&i_begin, &i_end, num_cols_offd_C);

         NALU_HYPRE_Int cnt;
         if (i_end > i_begin)
         {
            cnt = nalu_hypre_BigLowerBound(col_map_offd_B,
                                      col_map_offd_B + (NALU_HYPRE_BigInt)num_cols_offd_B,
                                      col_map_offd_C[i_begin]) - col_map_offd_B;
         }

         for (i = i_begin; i < i_end && cnt < num_cols_offd_B; i++)
         {
            if (col_map_offd_C[i] == col_map_offd_B[cnt])
            {
               map_B_to_C[cnt++] = i;
            }
         }
      }
   }
   if (num_procs > 1)
   {
      nalu_hypre_CSRMatrixDestroy(Bs_ext);
      Bs_ext = NULL;
   }

#else /* !NALU_HYPRE_CONCURRENT_HOPSCOTCH */

   NALU_HYPRE_BigInt *temp;
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      NALU_HYPRE_Int size, rest, ii;
      NALU_HYPRE_Int ns, ne;
      NALU_HYPRE_Int i1, i, j;
      NALU_HYPRE_Int my_offd_size, my_diag_size;
      NALU_HYPRE_Int cnt_offd, cnt_diag;

      NALU_HYPRE_Int num_threads = nalu_hypre_NumActiveThreads();

      size = num_cols_offd_A / num_threads;
      rest = num_cols_offd_A - size * num_threads;
      ii = nalu_hypre_GetThreadNum();
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

      my_diag_size = 0;
      my_offd_size = 0;
      for (i = ns; i < ne; i++)
      {
         B_ext_diag_i[i] = my_diag_size;
         B_ext_offd_i[i] = my_offd_size;
         for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         {
            if (Bs_ext_j[j] < first_col_diag_B ||
                Bs_ext_j[j] > last_col_diag_B)
            {
               my_offd_size++;
            }
            else
            {
               my_diag_size++;
            }
         }
      }
      my_diag_array[ii] = my_diag_size;
      my_offd_array[ii] = my_offd_size;

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      if (ii)
      {
         my_diag_size = my_diag_array[0];
         my_offd_size = my_offd_array[0];
         for (i1 = 1; i1 < ii; i1++)
         {
            my_diag_size += my_diag_array[i1];
            my_offd_size += my_offd_array[i1];
         }

         for (i1 = ns; i1 < ne; i1++)
         {
            B_ext_diag_i[i1] += my_diag_size;
            B_ext_offd_i[i1] += my_offd_size;
         }
      }
      else
      {
         B_ext_diag_size = 0;
         B_ext_offd_size = 0;
         for (i1 = 0; i1 < num_threads; i1++)
         {
            B_ext_diag_size += my_diag_array[i1];
            B_ext_offd_size += my_offd_array[i1];
         }
         B_ext_diag_i[num_cols_offd_A] = B_ext_diag_size;
         B_ext_offd_i[num_cols_offd_A] = B_ext_offd_size;

         if (B_ext_diag_size)
         {
            B_ext_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, B_ext_diag_size, NALU_HYPRE_MEMORY_HOST);
            B_ext_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, B_ext_diag_size, NALU_HYPRE_MEMORY_HOST);
         }

         if (B_ext_offd_size)
         {
            B_ext_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, B_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
            B_big_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, B_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
            B_ext_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, B_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
         }

         if (B_ext_offd_size || num_cols_offd_B)
         {
            temp = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, B_ext_offd_size + num_cols_offd_B, NALU_HYPRE_MEMORY_HOST);
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      cnt_offd = B_ext_offd_i[ns];
      cnt_diag = B_ext_diag_i[ns];
      for (i = ns; i < ne; i++)
      {
         for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         {
            if (Bs_ext_j[j] < first_col_diag_B ||
                Bs_ext_j[j] > last_col_diag_B)
            {
               temp[cnt_offd] = Bs_ext_j[j];
               B_big_offd_j[cnt_offd] = Bs_ext_j[j];
               //Bs_ext_j[cnt_offd] = Bs_ext_j[j];
               B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
            }
            else
            {
               B_ext_diag_j[cnt_diag] = (NALU_HYPRE_Int)(Bs_ext_j[j] - first_col_diag_B);
               B_ext_diag_data[cnt_diag++] = Bs_ext_data[j];
            }
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      if (ii == 0)
      {
         NALU_HYPRE_Int cnt;

         if (num_procs > 1)
         {
            nalu_hypre_CSRMatrixDestroy(Bs_ext);
            Bs_ext = NULL;
         }

         cnt = 0;
         if (B_ext_offd_size || num_cols_offd_B)
         {
            cnt = B_ext_offd_size;
            for (i = 0; i < num_cols_offd_B; i++)
            {
               temp[cnt++] = col_map_offd_B[i];
            }

            if (cnt)
            {
               NALU_HYPRE_BigInt value;

               nalu_hypre_BigQsort0(temp, 0, cnt - 1);
               num_cols_offd_C = 1;
               value = temp[0];
               for (i = 1; i < cnt; i++)
               {
                  if (temp[i] > value)
                  {
                     value = temp[i];
                     temp[num_cols_offd_C++] = value;
                  }
               }
            }

            if (num_cols_offd_C)
            {
               col_map_offd_C = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_C, NALU_HYPRE_MEMORY_HOST);
            }

            for (i = 0; i < num_cols_offd_C; i++)
            {
               col_map_offd_C[i] = temp[i];
            }

            nalu_hypre_TFree(temp, NALU_HYPRE_MEMORY_HOST);
         }
      }


#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      for (i = ns; i < ne; i++)
      {
         for (j = B_ext_offd_i[i]; j < B_ext_offd_i[i + 1]; j++)
         {
            B_ext_offd_j[j] = nalu_hypre_BigBinarySearch(col_map_offd_C, B_big_offd_j[j],
                                                    //B_ext_offd_j[j] = nalu_hypre_BigBinarySearch(col_map_offd_C, Bs_ext_j[j],
                                                    num_cols_offd_C);
         }
      }

   } /* end parallel region */
   nalu_hypre_TFree(B_big_offd_j, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(my_diag_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(my_offd_array, NALU_HYPRE_MEMORY_HOST);

   if (num_cols_offd_B)
   {
      NALU_HYPRE_Int i, cnt;
      map_B_to_C = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_B, NALU_HYPRE_MEMORY_HOST);

      cnt = 0;
      for (i = 0; i < num_cols_offd_C; i++)
      {
         if (col_map_offd_C[i] == col_map_offd_B[cnt])
         {
            map_B_to_C[cnt++] = i;
            if (cnt == num_cols_offd_B) { break; }
         }
      }
   }

#endif /* !NALU_HYPRE_CONCURRENT_HOPSCOTCH */

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_RENUMBER_COLIDX] += nalu_hypre_MPI_Wtime();
#endif

   NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "First pass");
   nalu_hypre_ParMatmul_RowSizes(memory_location_C, &C_diag_i, &C_offd_i,
                            rownnz_A, A_diag_i, A_diag_j,
                            A_offd_i, A_offd_j,
                            B_diag_i, B_diag_j,
                            B_offd_i, B_offd_j,
                            B_ext_diag_i, B_ext_diag_j,
                            B_ext_offd_i, B_ext_offd_j, map_B_to_C,
                            &C_diag_size, &C_offd_size,
                            num_rownnz_A, num_rows_diag_A, num_cols_offd_A,
                            allsquare, num_cols_diag_B, num_cols_offd_B,
                            num_cols_offd_C);
   NALU_HYPRE_ANNOTATE_REGION_END("%s", "First pass");

   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/

   last_col_diag_B = first_col_diag_B + (NALU_HYPRE_BigInt)num_cols_diag_B - 1;
   C_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, C_diag_size, memory_location_C);
   C_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int, C_diag_size, memory_location_C);
   if (C_offd_size)
   {
      C_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, C_offd_size, memory_location_C);
      C_offd_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int, C_offd_size, memory_location_C);
   }

   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in C_diag_data and C_diag_j.
    *  Second Pass: Fill in C_offd_data and C_offd_j.
    *-----------------------------------------------------------------------*/

   NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "Second pass");

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      NALU_HYPRE_Int     *B_marker = NULL;
      NALU_HYPRE_Int      ns, ne, size, rest, ii;
      NALU_HYPRE_Int      i1, ii1, i2, i3, jj2, jj3;
      NALU_HYPRE_Int      jj_row_begin_diag, jj_count_diag;
      NALU_HYPRE_Int      jj_row_begin_offd, jj_count_offd;
      NALU_HYPRE_Int      num_threads;
      NALU_HYPRE_Complex  a_entry; /*, a_b_product;*/

      num_threads = nalu_hypre_NumActiveThreads();
      size = num_rownnz_A / num_threads;
      rest = num_rownnz_A - size * num_threads;

      ii = nalu_hypre_GetThreadNum();
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
      jj_count_diag = C_diag_i[rownnz_A ? rownnz_A[ns] : ns];
      jj_count_offd = C_offd_i[rownnz_A ? rownnz_A[ns] : ns];

      if (num_cols_diag_B || num_cols_offd_C)
      {
         B_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_diag_B + num_cols_offd_C,
                                  NALU_HYPRE_MEMORY_HOST);
         for (i1 = 0; i1 < num_cols_diag_B + num_cols_offd_C; i1++)
         {
            B_marker[i1] = -1;
         }
      }

      /*-----------------------------------------------------------------------
       *  Loop over interior c-points.
       *-----------------------------------------------------------------------*/
      for (i1 = ns; i1 < ne; i1++)
      {
         jj_row_begin_diag = jj_count_diag;
         jj_row_begin_offd = jj_count_offd;
         if (rownnz_A)
         {
            ii1 = rownnz_A[i1];
         }
         else
         {
            ii1 = i1;

            /*--------------------------------------------------------------------
             *  Create diagonal entry, C_{i1,i1}
             *--------------------------------------------------------------------*/

            if (allsquare)
            {
               B_marker[i1] = jj_count_diag;
               C_diag_data[jj_count_diag] = zero;
               C_diag_j[jj_count_diag] = i1;
               jj_count_diag++;
            }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/

         if (num_cols_offd_A)
         {
            for (jj2 = A_offd_i[ii1]; jj2 < A_offd_i[ii1 + 1]; jj2++)
            {
               i2 = A_offd_j[jj2];
               a_entry = A_offd_data[jj2];

               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of B_ext.
                *-----------------------------------------------------------*/

               for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2 + 1]; jj3++)
               {
                  i3 = num_cols_diag_B + B_ext_offd_j[jj3];

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{ii1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_offd)
                  {
                     B_marker[i3] = jj_count_offd;
                     C_offd_data[jj_count_offd] = a_entry * B_ext_offd_data[jj3];
                     C_offd_j[jj_count_offd] = i3 - num_cols_diag_B;
                     jj_count_offd++;
                  }
                  else
                  {
                     C_offd_data[B_marker[i3]] += a_entry * B_ext_offd_data[jj3];
                  }
               }
               for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2 + 1]; jj3++)
               {
                  i3 = B_ext_diag_j[jj3];
                  if (B_marker[i3] < jj_row_begin_diag)
                  {
                     B_marker[i3] = jj_count_diag;
                     C_diag_data[jj_count_diag] = a_entry * B_ext_diag_data[jj3];
                     C_diag_j[jj_count_diag] = i3;
                     jj_count_diag++;
                  }
                  else
                  {
                     C_diag_data[B_marker[i3]] += a_entry * B_ext_diag_data[jj3];
                  }
               }
            }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row ii1 of A_diag.
          *-----------------------------------------------------------------*/

         for (jj2 = A_diag_i[ii1]; jj2 < A_diag_i[ii1 + 1]; jj2++)
         {
            i2 = A_diag_j[jj2];
            a_entry = A_diag_data[jj2];

            /*-----------------------------------------------------------
             *  Loop over entries in row i2 of B_diag.
             *-----------------------------------------------------------*/

            for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2 + 1]; jj3++)
            {
               i3 = B_diag_j[jj3];

               /*--------------------------------------------------------
                *  Check B_marker to see that C_{ii1,i3} has not already
                *  been accounted for. If it has not, create a new entry.
                *  If it has, add new contribution.
                *--------------------------------------------------------*/

               if (B_marker[i3] < jj_row_begin_diag)
               {
                  B_marker[i3] = jj_count_diag;
                  C_diag_data[jj_count_diag] = a_entry * B_diag_data[jj3];
                  C_diag_j[jj_count_diag] = i3;
                  jj_count_diag++;
               }
               else
               {
                  C_diag_data[B_marker[i3]] += a_entry * B_diag_data[jj3];
               }
            }
            if (num_cols_offd_B)
            {
               for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2 + 1]; jj3++)
               {
                  i3 = num_cols_diag_B + map_B_to_C[B_offd_j[jj3]];

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{ii1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (B_marker[i3] < jj_row_begin_offd)
                  {
                     B_marker[i3] = jj_count_offd;
                     C_offd_data[jj_count_offd] = a_entry * B_offd_data[jj3];
                     C_offd_j[jj_count_offd] = i3 - num_cols_diag_B;
                     jj_count_offd++;
                  }
                  else
                  {
                     C_offd_data[B_marker[i3]] += a_entry * B_offd_data[jj3];
                  }
               }
            }
         }
      }

      nalu_hypre_TFree(B_marker, NALU_HYPRE_MEMORY_HOST);
   } /*end parallel region */
   NALU_HYPRE_ANNOTATE_REGION_END("%s", "Second pass");

   C = nalu_hypre_ParCSRMatrixCreate(comm, nrows_A, ncols_B, row_starts_A,
                                col_starts_B, num_cols_offd_C,
                                C_diag_size, C_offd_size);

   C_diag = nalu_hypre_ParCSRMatrixDiag(C);
   nalu_hypre_CSRMatrixData(C_diag) = C_diag_data;
   nalu_hypre_CSRMatrixI(C_diag)    = C_diag_i;
   nalu_hypre_CSRMatrixJ(C_diag)    = C_diag_j;
   nalu_hypre_CSRMatrixMemoryLocation(C_diag) = memory_location_C;
   nalu_hypre_CSRMatrixSetRownnz(C_diag);

   C_offd = nalu_hypre_ParCSRMatrixOffd(C);
   nalu_hypre_CSRMatrixI(C_offd)  = C_offd_i;
   nalu_hypre_ParCSRMatrixOffd(C) = C_offd;
   if (num_cols_offd_C)
   {
      nalu_hypre_CSRMatrixData(C_offd)     = C_offd_data;
      nalu_hypre_CSRMatrixJ(C_offd)        = C_offd_j;
      nalu_hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;
   }
   nalu_hypre_CSRMatrixMemoryLocation(C_offd) = memory_location_C;
   nalu_hypre_CSRMatrixSetRownnz(C_offd);


   /*-----------------------------------------------------------------------
    *  Free various arrays
    *-----------------------------------------------------------------------*/
   nalu_hypre_TFree(B_ext_diag_i, NALU_HYPRE_MEMORY_HOST);
   if (B_ext_diag_size)
   {
      nalu_hypre_TFree(B_ext_diag_j, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(B_ext_diag_data, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(B_ext_offd_i, NALU_HYPRE_MEMORY_HOST);
   if (B_ext_offd_size)
   {
      nalu_hypre_TFree(B_ext_offd_j, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(B_ext_offd_data, NALU_HYPRE_MEMORY_HOST);
   }
   if (num_cols_offd_B)
   {
      nalu_hypre_TFree(map_B_to_C, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(rownnz_A, memory_location_A);

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_MATMUL] += nalu_hypre_MPI_Wtime();
#endif

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return C;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixExtractBExt_Arrays_Overlap
 *
 * The following function was formerly part of nalu_hypre_ParCSRMatrixExtractBExt
 * but the code was removed so it can be used for a corresponding function
 * for Boolean matrices
 *
 * JSP: to allow communication overlapping, it returns comm_handle_idx and
 * comm_handle_data. Before accessing B, they should be destroyed (including
 * send_data contained in the comm_handle).
 *--------------------------------------------------------------------------*/

void nalu_hypre_ParCSRMatrixExtractBExt_Arrays_Overlap(
   NALU_HYPRE_Int ** pB_ext_i,
   NALU_HYPRE_BigInt ** pB_ext_j,
   NALU_HYPRE_Complex ** pB_ext_data,
   NALU_HYPRE_BigInt ** pB_ext_row_map,
   NALU_HYPRE_Int * num_nonzeros,
   NALU_HYPRE_Int data,
   NALU_HYPRE_Int find_row_map,
   MPI_Comm comm,
   nalu_hypre_ParCSRCommPkg * comm_pkg,
   NALU_HYPRE_Int num_cols_B,
   NALU_HYPRE_Int num_recvs,
   NALU_HYPRE_Int num_sends,
   NALU_HYPRE_BigInt first_col_diag,
   NALU_HYPRE_BigInt * row_starts,
   NALU_HYPRE_Int * recv_vec_starts,
   NALU_HYPRE_Int * send_map_starts,
   NALU_HYPRE_Int * send_map_elmts,
   NALU_HYPRE_Int * diag_i,
   NALU_HYPRE_Int * diag_j,
   NALU_HYPRE_Int * offd_i,
   NALU_HYPRE_Int * offd_j,
   NALU_HYPRE_BigInt * col_map_offd,
   NALU_HYPRE_Real * diag_data,
   NALU_HYPRE_Real * offd_data,
   nalu_hypre_ParCSRCommHandle **comm_handle_idx,
   nalu_hypre_ParCSRCommHandle **comm_handle_data,
   NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *CF_marker_offd,
   NALU_HYPRE_Int skip_fine, /* 1 if only coarse points are needed */
   NALU_HYPRE_Int skip_same_sign /* 1 if only points that have the same sign are needed */
   // extended based long range interpolation: skip_fine = 1, skip_same_sign = 0 for S matrix, skip_fine = 1, skip_same_sign = 1 for A matrix
   // other interpolation: skip_fine = 0, skip_same_sign = 0
)
{
   nalu_hypre_ParCSRCommHandle *comm_handle, *row_map_comm_handle = NULL;
   nalu_hypre_ParCSRCommPkg *tmp_comm_pkg = NULL;
   NALU_HYPRE_Int *B_int_i;
   NALU_HYPRE_BigInt *B_int_j;
   NALU_HYPRE_Int *B_ext_i;
   NALU_HYPRE_BigInt * B_ext_j;
   NALU_HYPRE_Complex * B_ext_data;
   NALU_HYPRE_Complex * B_int_data;
   NALU_HYPRE_BigInt * B_int_row_map;
   NALU_HYPRE_BigInt * B_ext_row_map;
   NALU_HYPRE_Int num_procs, my_id;
   NALU_HYPRE_Int *jdata_recv_vec_starts;
   NALU_HYPRE_Int *jdata_send_map_starts;

   NALU_HYPRE_Int i, j, k;
   NALU_HYPRE_Int start_index;
   /*NALU_HYPRE_Int jrow;*/
   NALU_HYPRE_Int num_rows_B_ext;
   NALU_HYPRE_Int *prefix_sum_workspace;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   NALU_HYPRE_BigInt first_row_index = row_starts[0];

   num_rows_B_ext = recv_vec_starts[num_recvs];
   if ( num_rows_B_ext < 0 )    /* no B_ext, no communication */
   {
      *pB_ext_i = NULL;
      *pB_ext_j = NULL;
      if ( data ) { *pB_ext_data = NULL; }
      if ( find_row_map ) { *pB_ext_row_map = NULL; }
      *num_nonzeros = 0;
      return;
   };
   B_int_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  send_map_starts[num_sends] + 1, NALU_HYPRE_MEMORY_HOST);
   B_ext_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_rows_B_ext + 1, NALU_HYPRE_MEMORY_HOST);
   *pB_ext_i = B_ext_i;
   if ( find_row_map )
   {
      B_int_row_map = nalu_hypre_CTAlloc( NALU_HYPRE_BigInt,  send_map_starts[num_sends] + 1, NALU_HYPRE_MEMORY_HOST);
      B_ext_row_map = nalu_hypre_CTAlloc( NALU_HYPRE_BigInt,  num_rows_B_ext + 1, NALU_HYPRE_MEMORY_HOST);
      *pB_ext_row_map = B_ext_row_map;
   };

   /*--------------------------------------------------------------------------
    * generate B_int_i through adding number of row-elements of offd and diag
    * for corresponding rows. B_int_i[j+1] contains the number of elements of
    * a row j (which is determined through send_map_elmts)
    *--------------------------------------------------------------------------*/

   jdata_send_map_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends + 1, NALU_HYPRE_MEMORY_HOST);
   jdata_recv_vec_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs + 1, NALU_HYPRE_MEMORY_HOST);
   jdata_send_map_starts[0] = B_int_i[0] = 0;

   /*NALU_HYPRE_Int prefix_sum_workspace[(nalu_hypre_NumThreads() + 1)*num_sends];*/
   prefix_sum_workspace = nalu_hypre_TAlloc(NALU_HYPRE_Int,  (nalu_hypre_NumThreads() + 1) * num_sends,
                                       NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel private(i,j,k)
#endif
   {
      /*NALU_HYPRE_Int counts[num_sends];*/
      NALU_HYPRE_Int *counts;
      counts = nalu_hypre_TAlloc(NALU_HYPRE_Int,  num_sends, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_sends; i++)
      {
         NALU_HYPRE_Int j_begin, j_end;
         nalu_hypre_GetSimpleThreadPartition(&j_begin, &j_end, send_map_starts[i + 1] - send_map_starts[i]);
         j_begin += send_map_starts[i];
         j_end += send_map_starts[i];

         NALU_HYPRE_Int count = 0;
         if (skip_fine && skip_same_sign)
         {
            for (j = j_begin; j < j_end; j++)
            {
               NALU_HYPRE_Int jrow = send_map_elmts[j];
               NALU_HYPRE_Int len = 0;

               if (diag_data[diag_i[jrow]] >= 0)
               {
                  for (k = diag_i[jrow] + 1; k < diag_i[jrow + 1]; k++)
                  {
                     if (diag_data[k] < 0 && CF_marker[diag_j[k]] >= 0) { len++; }
                  }
                  for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                  {
                     if (offd_data[k] < 0) { len++; }
                  }
               }
               else
               {
                  for (k = diag_i[jrow] + 1; k < diag_i[jrow + 1]; k++)
                  {
                     if (diag_data[k] > 0 && CF_marker[diag_j[k]] >= 0) { len++; }
                  }
                  for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                  {
                     if (offd_data[k] > 0) { len++; }
                  }
               }

               B_int_i[j + 1] = len;
               count += len;
            }
         }
         else if (skip_fine)
         {
            for (j = j_begin; j < j_end; j++)
            {
               NALU_HYPRE_Int jrow = send_map_elmts[j];
               NALU_HYPRE_Int len = 0;

               for (k = diag_i[jrow]; k < diag_i[jrow + 1]; k++)
               {
                  if (CF_marker[diag_j[k]] >= 0) { len++; }
               }
               for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
               {
                  if (CF_marker_offd[offd_j[k]] >= 0) { len++; }
               }

               B_int_i[j + 1] = len;
               count += len;
            }
         }
         else
         {
            for (j = j_begin; j < j_end; j++)
            {
               NALU_HYPRE_Int jrow = send_map_elmts[j];
               NALU_HYPRE_Int len = diag_i[jrow + 1] - diag_i[jrow];
               len += offd_i[jrow + 1] - offd_i[jrow];
               B_int_i[j + 1] = len;
               count += len;
            }
         }

         if (find_row_map)
         {
            for (j = j_begin; j < j_end; j++)
            {
               NALU_HYPRE_Int jrow = send_map_elmts[j];
               B_int_row_map[j] = (NALU_HYPRE_BigInt)jrow + first_row_index;
            }
         }

         counts[i] = count;
      }

      nalu_hypre_prefix_sum_multiple(counts, jdata_send_map_starts + 1, num_sends, prefix_sum_workspace);

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp master
#endif
      {
         for (i = 1; i < num_sends; i++)
         {
            jdata_send_map_starts[i + 1] += jdata_send_map_starts[i];
         }

         /*--------------------------------------------------------------------------
          * initialize communication
          *--------------------------------------------------------------------------*/

         comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg,
                                                    &B_int_i[1], &(B_ext_i[1]) );
         if ( find_row_map )
         {
            /* scatter/gather B_int row numbers to form array of B_ext row numbers */
            row_map_comm_handle = nalu_hypre_ParCSRCommHandleCreate
                                  (21, comm_pkg, B_int_row_map, B_ext_row_map );
         }

         B_int_j = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  jdata_send_map_starts[num_sends], NALU_HYPRE_MEMORY_HOST);
         if (data) { B_int_data = nalu_hypre_TAlloc(NALU_HYPRE_Complex,  jdata_send_map_starts[num_sends], NALU_HYPRE_MEMORY_HOST); }
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      for (i = 0; i < num_sends; i++)
      {
         NALU_HYPRE_Int j_begin, j_end;
         nalu_hypre_GetSimpleThreadPartition(&j_begin, &j_end, send_map_starts[i + 1] - send_map_starts[i]);
         j_begin += send_map_starts[i];
         j_end += send_map_starts[i];

         NALU_HYPRE_Int count = counts[i] + jdata_send_map_starts[i];

         if (data)
         {
            if (skip_same_sign && skip_fine)
            {
               for (j = j_begin; j < j_end; j++)
               {
                  NALU_HYPRE_Int jrow = send_map_elmts[j];
                  /*NALU_HYPRE_Int count_begin = count;*/

                  if (diag_data[diag_i[jrow]] >= 0)
                  {
                     for (k = diag_i[jrow] + 1; k < diag_i[jrow + 1]; k++)
                     {
                        if (diag_data[k] < 0 && CF_marker[diag_j[k]] >= 0)
                        {
                           B_int_j[count] = (NALU_HYPRE_BigInt)diag_j[k] + first_col_diag;
                           B_int_data[count] = diag_data[k];
                           count++;
                        }
                     }
                     for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                     {
                        NALU_HYPRE_Int c = offd_j[k];
                        NALU_HYPRE_BigInt c_global = col_map_offd[c];
                        if (offd_data[k] < 0)
                        {
                           B_int_j[count] = c_global;
                           B_int_data[count] = offd_data[k];
                           count++;
                        }
                     }
                  }
                  else
                  {
                     for (k = diag_i[jrow] + 1; k < diag_i[jrow + 1]; k++)
                     {
                        if (diag_data[k] > 0 && CF_marker[diag_j[k]] >= 0)
                        {
                           B_int_j[count] = (NALU_HYPRE_BigInt)diag_j[k] + first_col_diag;
                           B_int_data[count] = diag_data[k];
                           count++;
                        }
                     }
                     for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                     {
                        NALU_HYPRE_Int c = offd_j[k];
                        NALU_HYPRE_BigInt c_global = col_map_offd[c];
                        if (offd_data[k] > 0)
                        {
                           B_int_j[count] = c_global;
                           B_int_data[count] = offd_data[k];
                           count++;
                        }
                     }
                  }
               }
            }
            else
            {
               for (j = j_begin; j < j_end; ++j)
               {
                  NALU_HYPRE_Int jrow = send_map_elmts[j];
                  for (k = diag_i[jrow]; k < diag_i[jrow + 1]; k++)
                  {
                     B_int_j[count] = (NALU_HYPRE_BigInt)diag_j[k] + first_col_diag;
                     B_int_data[count] = diag_data[k];
                     count++;
                  }
                  for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                  {
                     B_int_j[count] = col_map_offd[offd_j[k]];
                     B_int_data[count] = offd_data[k];
                     count++;
                  }
               }
            }
         } // data
         else
         {
            if (skip_fine)
            {
               for (j = j_begin; j < j_end; j++)
               {
                  NALU_HYPRE_Int jrow = send_map_elmts[j];
                  for (k = diag_i[jrow]; k < diag_i[jrow + 1]; k++)
                  {
                     if (CF_marker[diag_j[k]] >= 0)
                     {
                        B_int_j[count] = (NALU_HYPRE_BigInt)diag_j[k] + first_col_diag;
                        count++;
                     }
                  }
                  for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                  {
                     if (CF_marker_offd[offd_j[k]] >= 0)
                     {
                        B_int_j[count] = col_map_offd[offd_j[k]];
                        count++;
                     }
                  }
               }
            }
            else
            {
               for (j = j_begin; j < j_end; ++j)
               {
                  NALU_HYPRE_Int jrow = send_map_elmts[j];
                  for (k = diag_i[jrow]; k < diag_i[jrow + 1]; k++)
                  {
                     B_int_j[count] = (NALU_HYPRE_BigInt)diag_j[k] + first_col_diag;
                     count++;
                  }
                  for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
                  {
                     B_int_j[count] = col_map_offd[offd_j[k]];
                     count++;
                  }
               }
            }
         } // !data
      } /* for each send target */
      nalu_hypre_TFree(counts, NALU_HYPRE_MEMORY_HOST);
   } /* omp parallel. JSP: this takes most of time in this function */
   nalu_hypre_TFree(prefix_sum_workspace, NALU_HYPRE_MEMORY_HOST);

   /* Create temporary communication package */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs,
                                    nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                    jdata_recv_vec_starts,
                                    num_sends,
                                    nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                    jdata_send_map_starts,
                                    NULL,
                                    &tmp_comm_pkg);

   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   /*--------------------------------------------------------------------------
    * after communication exchange B_ext_i[j+1] contains the number of elements
    * of a row j !
    * evaluate B_ext_i and compute *num_nonzeros for B_ext
    *--------------------------------------------------------------------------*/

   for (i = 0; i < num_recvs; i++)
   {
      for (j = recv_vec_starts[i]; j < recv_vec_starts[i + 1]; j++)
      {
         B_ext_i[j + 1] += B_ext_i[j];
      }
   }

   *num_nonzeros = B_ext_i[num_rows_B_ext];

   *pB_ext_j = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  *num_nonzeros, NALU_HYPRE_MEMORY_HOST);
   B_ext_j = *pB_ext_j;
   if (data)
   {
      *pB_ext_data = nalu_hypre_TAlloc(NALU_HYPRE_Complex,  *num_nonzeros, NALU_HYPRE_MEMORY_HOST);
      B_ext_data = *pB_ext_data;
   }

   for (i = 0; i < num_recvs; i++)
   {
      start_index = B_ext_i[recv_vec_starts[i]];
      *num_nonzeros = B_ext_i[recv_vec_starts[i + 1]] - start_index;
      jdata_recv_vec_starts[i + 1] = B_ext_i[recv_vec_starts[i + 1]];
   }

   *comm_handle_idx = nalu_hypre_ParCSRCommHandleCreate(21, tmp_comm_pkg, B_int_j, B_ext_j);
   if (data)
   {
      *comm_handle_data = nalu_hypre_ParCSRCommHandleCreate(1, tmp_comm_pkg, B_int_data,
                                                       B_ext_data);
   }

   /* Free memory */
   nalu_hypre_TFree(jdata_send_map_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jdata_recv_vec_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(tmp_comm_pkg, NALU_HYPRE_MEMORY_HOST);
   if (row_map_comm_handle)
   {
      nalu_hypre_ParCSRCommHandleDestroy(row_map_comm_handle);
      row_map_comm_handle = NULL;
   }
   if (find_row_map)
   {
      nalu_hypre_TFree(B_int_row_map, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(B_int_i, NALU_HYPRE_MEMORY_HOST);

   /* end generic part */
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixExtractBExt_Arrays
 *--------------------------------------------------------------------------*/

void nalu_hypre_ParCSRMatrixExtractBExt_Arrays(
   NALU_HYPRE_Int ** pB_ext_i,
   NALU_HYPRE_BigInt ** pB_ext_j,
   NALU_HYPRE_Complex ** pB_ext_data,
   NALU_HYPRE_BigInt ** pB_ext_row_map,
   NALU_HYPRE_Int * num_nonzeros,
   NALU_HYPRE_Int data,
   NALU_HYPRE_Int find_row_map,
   MPI_Comm comm,
   nalu_hypre_ParCSRCommPkg * comm_pkg,
   NALU_HYPRE_Int num_cols_B,
   NALU_HYPRE_Int num_recvs,
   NALU_HYPRE_Int num_sends,
   NALU_HYPRE_BigInt first_col_diag,
   NALU_HYPRE_BigInt * row_starts,
   NALU_HYPRE_Int * recv_vec_starts,
   NALU_HYPRE_Int * send_map_starts,
   NALU_HYPRE_Int * send_map_elmts,
   NALU_HYPRE_Int * diag_i,
   NALU_HYPRE_Int * diag_j,
   NALU_HYPRE_Int * offd_i,
   NALU_HYPRE_Int * offd_j,
   NALU_HYPRE_BigInt * col_map_offd,
   NALU_HYPRE_Real * diag_data,
   NALU_HYPRE_Real * offd_data
)
{
   nalu_hypre_ParCSRCommHandle *comm_handle_idx, *comm_handle_data;

   nalu_hypre_ParCSRMatrixExtractBExt_Arrays_Overlap(
      pB_ext_i, pB_ext_j, pB_ext_data, pB_ext_row_map, num_nonzeros,
      data, find_row_map, comm, comm_pkg, num_cols_B, num_recvs, num_sends,
      first_col_diag, row_starts, recv_vec_starts, send_map_starts, send_map_elmts,
      diag_i, diag_j, offd_i, offd_j, col_map_offd, diag_data, offd_data,
      &comm_handle_idx, &comm_handle_data,
      NULL, NULL,
      0, 0);

   NALU_HYPRE_Int *send_idx = (NALU_HYPRE_Int *)comm_handle_idx->send_data;
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle_idx);
   nalu_hypre_TFree(send_idx, NALU_HYPRE_MEMORY_HOST);

   if (data)
   {
      NALU_HYPRE_Real *send_data = (NALU_HYPRE_Real *)comm_handle_data->send_data;
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle_data);
      nalu_hypre_TFree(send_data, NALU_HYPRE_MEMORY_HOST);
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixExtractBExt : extracts rows from B which are located on
 * other processors and needed for multiplication with A locally. The rows
 * are returned as CSRMatrix.
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix *
nalu_hypre_ParCSRMatrixExtractBExt_Overlap( nalu_hypre_ParCSRMatrix *B,
                                       nalu_hypre_ParCSRMatrix *A,
                                       NALU_HYPRE_Int data,
                                       nalu_hypre_ParCSRCommHandle **comm_handle_idx,
                                       nalu_hypre_ParCSRCommHandle **comm_handle_data,
                                       NALU_HYPRE_Int *CF_marker, NALU_HYPRE_Int *CF_marker_offd,
                                       NALU_HYPRE_Int skip_fine, NALU_HYPRE_Int skip_same_sign )
{
   MPI_Comm  comm = nalu_hypre_ParCSRMatrixComm(B);
   NALU_HYPRE_BigInt first_col_diag = nalu_hypre_ParCSRMatrixFirstColDiag(B);
   /*NALU_HYPRE_Int first_row_index = nalu_hypre_ParCSRMatrixFirstRowIndex(B);*/
   NALU_HYPRE_BigInt *col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(B);

   nalu_hypre_ParCSRCommPkg *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   NALU_HYPRE_Int num_recvs;
   NALU_HYPRE_Int *recv_vec_starts;
   NALU_HYPRE_Int num_sends;
   NALU_HYPRE_Int *send_map_starts;
   NALU_HYPRE_Int *send_map_elmts;

   nalu_hypre_CSRMatrix *diag = nalu_hypre_ParCSRMatrixDiag(B);

   NALU_HYPRE_Int *diag_i = nalu_hypre_CSRMatrixI(diag);
   NALU_HYPRE_Int *diag_j = nalu_hypre_CSRMatrixJ(diag);
   NALU_HYPRE_Real *diag_data = nalu_hypre_CSRMatrixData(diag);

   nalu_hypre_CSRMatrix *offd = nalu_hypre_ParCSRMatrixOffd(B);

   NALU_HYPRE_Int *offd_i = nalu_hypre_CSRMatrixI(offd);
   NALU_HYPRE_Int *offd_j = nalu_hypre_CSRMatrixJ(offd);
   NALU_HYPRE_Real *offd_data = nalu_hypre_CSRMatrixData(offd);

   NALU_HYPRE_Int num_cols_B, num_nonzeros;
   NALU_HYPRE_Int num_rows_B_ext;

   nalu_hypre_CSRMatrix *B_ext;

   NALU_HYPRE_Int *B_ext_i;
   NALU_HYPRE_BigInt *B_ext_j;
   NALU_HYPRE_Complex *B_ext_data;
   NALU_HYPRE_BigInt *idummy;

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/

   if (!nalu_hypre_ParCSRMatrixCommPkg(A))
   {
      nalu_hypre_MatvecCommPkgCreate(A);
   }

   comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_vec_starts = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   send_map_elmts = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

   num_cols_B = nalu_hypre_ParCSRMatrixGlobalNumCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];

   nalu_hypre_ParCSRMatrixExtractBExt_Arrays_Overlap
   ( &B_ext_i, &B_ext_j, &B_ext_data, &idummy,
     &num_nonzeros,
     data, 0, comm, comm_pkg,
     num_cols_B, num_recvs, num_sends,
     first_col_diag, B->row_starts,
     recv_vec_starts, send_map_starts, send_map_elmts,
     diag_i, diag_j, offd_i, offd_j, col_map_offd,
     diag_data, offd_data,
     comm_handle_idx, comm_handle_data,
     CF_marker, CF_marker_offd,
     skip_fine, skip_same_sign
   );

   B_ext = nalu_hypre_CSRMatrixCreate(num_rows_B_ext, num_cols_B, num_nonzeros);
   nalu_hypre_CSRMatrixMemoryLocation(B_ext) = NALU_HYPRE_MEMORY_HOST;
   nalu_hypre_CSRMatrixI(B_ext) = B_ext_i;
   nalu_hypre_CSRMatrixBigJ(B_ext) = B_ext_j;
   if (data) { nalu_hypre_CSRMatrixData(B_ext) = B_ext_data; }

   return B_ext;
}

nalu_hypre_CSRMatrix *
nalu_hypre_ParCSRMatrixExtractBExt( nalu_hypre_ParCSRMatrix *B,
                               nalu_hypre_ParCSRMatrix *A,
                               NALU_HYPRE_Int want_data )
{
#if 0
   nalu_hypre_ParCSRCommHandle *comm_handle_idx, *comm_handle_data;

   nalu_hypre_CSRMatrix *B_ext = nalu_hypre_ParCSRMatrixExtractBExt_Overlap(B, A, want_data, &comm_handle_idx,
                                                                  &comm_handle_data, NULL, NULL, 0, 0);

   NALU_HYPRE_Int *send_idx = (NALU_HYPRE_Int *)comm_handle_idx->send_data;
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle_idx);
   nalu_hypre_TFree(send_idx, NALU_HYPRE_MEMORY_HOST);

   if (want_data)
   {
      NALU_HYPRE_Real *send_data = (NALU_HYPRE_Real *)comm_handle_data->send_data;
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle_data);
      nalu_hypre_TFree(send_data, NALU_HYPRE_MEMORY_HOST);
   }
#else
   nalu_hypre_assert( nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixDiag(B)) ==
                 nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixOffd(B)) );

   nalu_hypre_CSRMatrix *B_ext;
   void            *request;

   if (!nalu_hypre_ParCSRMatrixCommPkg(A))
   {
      nalu_hypre_MatvecCommPkgCreate(A);
   }

   nalu_hypre_ParcsrGetExternalRowsInit(B,
                                   nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(A)),
                                   nalu_hypre_ParCSRMatrixColMapOffd(A),
                                   nalu_hypre_ParCSRMatrixCommPkg(A),
                                   want_data,
                                   &request);

   B_ext = nalu_hypre_ParcsrGetExternalRowsWait(request);
#endif

   return B_ext;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixTransposeHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixTransposeHost( nalu_hypre_ParCSRMatrix  *A,
                                 nalu_hypre_ParCSRMatrix **AT_ptr,
                                 NALU_HYPRE_Int            data )
{
   MPI_Comm                 comm     = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_CSRMatrix         *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix         *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int                num_cols = nalu_hypre_ParCSRMatrixNumCols(A);
   NALU_HYPRE_BigInt             first_row_index = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   NALU_HYPRE_BigInt            *row_starts = nalu_hypre_ParCSRMatrixRowStarts(A);
   NALU_HYPRE_BigInt            *col_starts = nalu_hypre_ParCSRMatrixColStarts(A);

   NALU_HYPRE_Int                num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_Int                num_sends, num_recvs, num_cols_offd_AT;
   NALU_HYPRE_Int                i, j, k, index, counter, j_row;
   NALU_HYPRE_BigInt             value;

   nalu_hypre_ParCSRMatrix      *AT;
   nalu_hypre_CSRMatrix         *AT_diag;
   nalu_hypre_CSRMatrix         *AT_offd;
   nalu_hypre_CSRMatrix         *AT_tmp;

   NALU_HYPRE_BigInt             first_row_index_AT, first_col_diag_AT;
   NALU_HYPRE_Int                local_num_rows_AT, local_num_cols_AT;

   NALU_HYPRE_Int               *AT_tmp_i;
   NALU_HYPRE_Int               *AT_tmp_j;
   NALU_HYPRE_BigInt            *AT_big_j = NULL;
   NALU_HYPRE_Complex           *AT_tmp_data;

   NALU_HYPRE_Int               *AT_buf_i;
   NALU_HYPRE_BigInt            *AT_buf_j;
   NALU_HYPRE_Complex           *AT_buf_data;

   NALU_HYPRE_Int               *AT_offd_i;
   NALU_HYPRE_Int               *AT_offd_j;
   NALU_HYPRE_Complex           *AT_offd_data;
   NALU_HYPRE_BigInt            *col_map_offd_AT;
   NALU_HYPRE_BigInt             row_starts_AT[2];
   NALU_HYPRE_BigInt             col_starts_AT[2];

   NALU_HYPRE_Int                num_procs, my_id;

   NALU_HYPRE_Int               *recv_procs, *send_procs;
   NALU_HYPRE_Int               *recv_vec_starts;
   NALU_HYPRE_Int               *send_map_starts;
   NALU_HYPRE_Int               *send_map_elmts;
   NALU_HYPRE_Int               *tmp_recv_vec_starts;
   NALU_HYPRE_Int               *tmp_send_map_starts;
   nalu_hypre_ParCSRCommPkg     *tmp_comm_pkg = NULL;
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   num_cols_offd_AT = 0;
   counter = 0;
   AT_offd_j = NULL;
   AT_offd_data = NULL;
   col_map_offd_AT = NULL;

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/

   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   if (num_procs > 1)
   {
      nalu_hypre_CSRMatrixTranspose (A_offd, &AT_tmp, data);

      AT_tmp_i = nalu_hypre_CSRMatrixI(AT_tmp);
      AT_tmp_j = nalu_hypre_CSRMatrixJ(AT_tmp);
      if (data)
      {
         AT_tmp_data = nalu_hypre_CSRMatrixData(AT_tmp);
      }

      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_procs = nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      send_procs = nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg);
      recv_vec_starts = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
      send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
      send_map_elmts = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

      AT_buf_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, send_map_starts[num_sends], NALU_HYPRE_MEMORY_HOST);
      if (AT_tmp_i[num_cols_offd])
      {
         AT_big_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, AT_tmp_i[num_cols_offd], NALU_HYPRE_MEMORY_HOST);
      }

      for (i = 0; i < AT_tmp_i[num_cols_offd]; i++)
      {
         //AT_tmp_j[i] += first_row_index;
         AT_big_j[i] = (NALU_HYPRE_BigInt)AT_tmp_j[i] + first_row_index;
      }

      for (i = 0; i < num_cols_offd; i++)
      {
         AT_tmp_i[i] = AT_tmp_i[i + 1] - AT_tmp_i[i];
      }

      comm_handle = nalu_hypre_ParCSRCommHandleCreate(12, comm_pkg, AT_tmp_i, AT_buf_i);
   }

   nalu_hypre_CSRMatrixTranspose(A_diag, &AT_diag, data);

   AT_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols + 1, memory_location);

   if (num_procs > 1)
   {
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;

      tmp_send_map_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_sends + 1, NALU_HYPRE_MEMORY_HOST);
      tmp_recv_vec_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_recvs + 1, NALU_HYPRE_MEMORY_HOST);

      tmp_send_map_starts[0] = send_map_starts[0];
      for (i = 0; i < num_sends; i++)
      {
         tmp_send_map_starts[i + 1] = tmp_send_map_starts[i];
         for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
         {
            tmp_send_map_starts[i + 1] += AT_buf_i[j];
            AT_offd_i[send_map_elmts[j] + 1] += AT_buf_i[j];
         }
      }
      for (i = 0; i < num_cols; i++)
      {
         AT_offd_i[i + 1] += AT_offd_i[i];
      }

      tmp_recv_vec_starts[0] = recv_vec_starts[0];
      for (i = 0; i < num_recvs; i++)
      {
         tmp_recv_vec_starts[i + 1] = tmp_recv_vec_starts[i];
         for (j = recv_vec_starts[i]; j < recv_vec_starts[i + 1]; j++)
         {
            tmp_recv_vec_starts[i + 1] +=  AT_tmp_i[j];
         }
      }

      /* Create temporary communication package */
      nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                       num_recvs, recv_procs, tmp_recv_vec_starts,
                                       num_sends, send_procs, tmp_send_map_starts,
                                       NULL,
                                       &tmp_comm_pkg);

      AT_buf_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, tmp_send_map_starts[num_sends], NALU_HYPRE_MEMORY_HOST);
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(22, tmp_comm_pkg, AT_big_j,
                                                 AT_buf_j);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
      comm_handle = NULL;
      nalu_hypre_TFree(AT_big_j, NALU_HYPRE_MEMORY_HOST);

      if (data)
      {
         AT_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, tmp_send_map_starts[num_sends], NALU_HYPRE_MEMORY_HOST);
         comm_handle = nalu_hypre_ParCSRCommHandleCreate(2, tmp_comm_pkg, AT_tmp_data,
                                                    AT_buf_data);
         nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
         comm_handle = NULL;
      }

      nalu_hypre_TFree(tmp_recv_vec_starts, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(tmp_send_map_starts, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(tmp_comm_pkg, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CSRMatrixDestroy(AT_tmp);

      if (AT_offd_i[num_cols])
      {
         AT_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, AT_offd_i[num_cols], memory_location);
         AT_big_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, AT_offd_i[num_cols], NALU_HYPRE_MEMORY_HOST);
         if (data)
         {
            AT_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  AT_offd_i[num_cols], memory_location);
         }
      }
      else
      {
         AT_offd_j = NULL;
         AT_offd_data = NULL;
      }

      counter = 0;
      for (i = 0; i < num_sends; i++)
      {
         for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
         {
            j_row = send_map_elmts[j];
            index = AT_offd_i[j_row];
            for (k = 0; k < AT_buf_i[j]; k++)
            {
               if (data)
               {
                  AT_offd_data[index] = AT_buf_data[counter];
               }
               AT_big_j[index++] = AT_buf_j[counter++];
            }
            AT_offd_i[j_row] = index;
         }
      }
      for (i = num_cols; i > 0; i--)
      {
         AT_offd_i[i] = AT_offd_i[i - 1];
      }
      AT_offd_i[0] = 0;

      if (counter)
      {
         nalu_hypre_BigQsort0(AT_buf_j, 0, counter - 1);
         num_cols_offd_AT = 1;
         value = AT_buf_j[0];
         for (i = 1; i < counter; i++)
         {
            if (value < AT_buf_j[i])
            {
               AT_buf_j[num_cols_offd_AT++] = AT_buf_j[i];
               value = AT_buf_j[i];
            }
         }
      }

      if (num_cols_offd_AT)
      {
         col_map_offd_AT = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_AT, NALU_HYPRE_MEMORY_HOST);
      }
      else
      {
         col_map_offd_AT = NULL;
      }

      for (i = 0; i < num_cols_offd_AT; i++)
      {
         col_map_offd_AT[i] = AT_buf_j[i];
      }
      nalu_hypre_TFree(AT_buf_i, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(AT_buf_j, NALU_HYPRE_MEMORY_HOST);
      if (data)
      {
         nalu_hypre_TFree(AT_buf_data, NALU_HYPRE_MEMORY_HOST);
      }

      for (i = 0; i < counter; i++)
      {
         AT_offd_j[i] = nalu_hypre_BigBinarySearch(col_map_offd_AT, AT_big_j[i],
                                              num_cols_offd_AT);
      }
      nalu_hypre_TFree(AT_big_j, NALU_HYPRE_MEMORY_HOST);
   }

   AT_offd = nalu_hypre_CSRMatrixCreate(num_cols, num_cols_offd_AT, counter);
   nalu_hypre_CSRMatrixMemoryLocation(AT_offd) = memory_location;
   nalu_hypre_CSRMatrixI(AT_offd) = AT_offd_i;
   nalu_hypre_CSRMatrixJ(AT_offd) = AT_offd_j;
   nalu_hypre_CSRMatrixData(AT_offd) = AT_offd_data;

   for (i = 0; i < 2; i++)
   {
      row_starts_AT[i] = col_starts[i];
      col_starts_AT[i] = row_starts[i];
   }

   first_row_index_AT = row_starts_AT[0];
   first_col_diag_AT  = col_starts_AT[0];

   local_num_rows_AT = (NALU_HYPRE_Int)(row_starts_AT[1] - first_row_index_AT );
   local_num_cols_AT = (NALU_HYPRE_Int)(col_starts_AT[1] - first_col_diag_AT);

   AT = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParCSRMatrixComm(AT) = comm;
   nalu_hypre_ParCSRMatrixDiag(AT) = AT_diag;
   nalu_hypre_ParCSRMatrixOffd(AT) = AT_offd;
   nalu_hypre_ParCSRMatrixGlobalNumRows(AT) = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
   nalu_hypre_ParCSRMatrixGlobalNumCols(AT) = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   nalu_hypre_ParCSRMatrixRowStarts(AT)[0]  = row_starts_AT[0];
   nalu_hypre_ParCSRMatrixRowStarts(AT)[1]  = row_starts_AT[1];
   nalu_hypre_ParCSRMatrixColStarts(AT)[0]  = col_starts_AT[0];
   nalu_hypre_ParCSRMatrixColStarts(AT)[1]  = col_starts_AT[1];
   nalu_hypre_ParCSRMatrixColMapOffd(AT)    = col_map_offd_AT;

   nalu_hypre_ParCSRMatrixFirstRowIndex(AT) = first_row_index_AT;
   nalu_hypre_ParCSRMatrixFirstColDiag(AT)  = first_col_diag_AT;

   nalu_hypre_ParCSRMatrixLastRowIndex(AT) = first_row_index_AT + local_num_rows_AT - 1;
   nalu_hypre_ParCSRMatrixLastColDiag(AT)  = first_col_diag_AT + local_num_cols_AT - 1;

   nalu_hypre_ParCSRMatrixOwnsData(AT) = 1;
   nalu_hypre_ParCSRMatrixCommPkg(AT)  = NULL;
   nalu_hypre_ParCSRMatrixCommPkgT(AT) = NULL;

   nalu_hypre_ParCSRMatrixRowindices(AT) = NULL;
   nalu_hypre_ParCSRMatrixRowvalues(AT)  = NULL;
   nalu_hypre_ParCSRMatrixGetrowactive(AT) = 0;

   nalu_hypre_ParCSRMatrixOwnsAssumedPartition(AT) = 1;

   *AT_ptr = AT;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixTranspose
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixTranspose( nalu_hypre_ParCSRMatrix  *A,
                             nalu_hypre_ParCSRMatrix **AT_ptr,
                             NALU_HYPRE_Int            data )
{
   nalu_hypre_GpuProfilingPushRange("ParCSRMatrixTranspose");

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_ParCSRMatrixTransposeDevice(A, AT_ptr, data);
   }
   else
#endif
   {
      nalu_hypre_ParCSRMatrixTransposeHost(A, AT_ptr, data);
   }

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixLocalTranspose
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixLocalTranspose( nalu_hypre_ParCSRMatrix  *A )
{
   if (!nalu_hypre_ParCSRMatrixDiagT(A))
   {
      nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
      if (A_diag)
      {
         nalu_hypre_CSRMatrix *AT_diag = NULL;
         nalu_hypre_CSRMatrixTranspose(A_diag, &AT_diag, 1);
         nalu_hypre_ParCSRMatrixDiagT(A) = AT_diag;
      }
   }

   if (!nalu_hypre_ParCSRMatrixOffdT(A))
   {
      nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
      if (A_offd)
      {
         nalu_hypre_CSRMatrix *AT_offd = NULL;
         nalu_hypre_CSRMatrixTranspose(A_offd, &AT_offd, 1);
         nalu_hypre_ParCSRMatrixOffdT(A) = AT_offd;
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGenSpanningTree
 *
 * generate a parallel spanning tree (for Maxwell Equation)
 * G_csr is the node to edge connectivity matrix
 *--------------------------------------------------------------------------*/

void
nalu_hypre_ParCSRMatrixGenSpanningTree( nalu_hypre_ParCSRMatrix *G_csr,
                                   NALU_HYPRE_Int         **indices,
                                   NALU_HYPRE_Int           G_type )
{
   NALU_HYPRE_BigInt nrows_G, ncols_G;
   NALU_HYPRE_Int *G_diag_i, *G_diag_j, *GT_diag_mat, i, j, k, edge;
   NALU_HYPRE_Int *nodes_marked, *edges_marked, *queue, queue_tail, queue_head, node;
   NALU_HYPRE_Int mypid, nprocs, n_children, *children, nsends, *send_procs, *recv_cnts;
   NALU_HYPRE_Int nrecvs, *recv_procs, n_proc_array, *proc_array, *pgraph_i, *pgraph_j;
   NALU_HYPRE_Int parent, proc, proc2, node2, found, *t_indices, tree_size, *T_diag_i;
   NALU_HYPRE_Int *T_diag_j, *counts, offset;
   MPI_Comm            comm;
   nalu_hypre_ParCSRCommPkg *comm_pkg;
   nalu_hypre_CSRMatrix     *G_diag;

   /* fetch G matrix (G_type = 0 ==> node to edge) */

   if (G_type == 0)
   {
      nrows_G = nalu_hypre_ParCSRMatrixGlobalNumRows(G_csr);
      ncols_G = nalu_hypre_ParCSRMatrixGlobalNumCols(G_csr);
      G_diag = nalu_hypre_ParCSRMatrixDiag(G_csr);
      G_diag_i = nalu_hypre_CSRMatrixI(G_diag);
      G_diag_j = nalu_hypre_CSRMatrixJ(G_diag);
   }
   else
   {
      nrows_G = nalu_hypre_ParCSRMatrixGlobalNumCols(G_csr);
      ncols_G = nalu_hypre_ParCSRMatrixGlobalNumRows(G_csr);
      G_diag = nalu_hypre_ParCSRMatrixDiag(G_csr);
      T_diag_i = nalu_hypre_CSRMatrixI(G_diag);
      T_diag_j = nalu_hypre_CSRMatrixJ(G_diag);
      counts = nalu_hypre_TAlloc(NALU_HYPRE_Int, nrows_G, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < nrows_G; i++) { counts[i] = 0; }
      for (i = 0; i < T_diag_i[ncols_G]; i++) { counts[T_diag_j[i]]++; }
      G_diag_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, (nrows_G + 1), NALU_HYPRE_MEMORY_HOST);
      G_diag_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, T_diag_i[ncols_G], NALU_HYPRE_MEMORY_HOST);
      G_diag_i[0] = 0;
      for (i = 1; i <= nrows_G; i++) { G_diag_i[i] = G_diag_i[i - 1] + counts[i - 1]; }
      for (i = 0; i < ncols_G; i++)
      {
         for (j = T_diag_i[i]; j < T_diag_i[i + 1]; j++)
         {
            k = T_diag_j[j];
            offset = G_diag_i[k]++;
            G_diag_j[offset] = i;
         }
      }
      G_diag_i[0] = 0;
      for (i = 1; i <= nrows_G; i++)
      {
         G_diag_i[i] = G_diag_i[i - 1] + counts[i - 1];
      }
      nalu_hypre_TFree(counts, NALU_HYPRE_MEMORY_HOST);
   }

   /* form G transpose in special form (2 nodes per edge max) */

   GT_diag_mat = nalu_hypre_TAlloc(NALU_HYPRE_Int, 2 * ncols_G, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < 2 * ncols_G; i++) { GT_diag_mat[i] = -1; }
   for (i = 0; i < nrows_G; i++)
   {
      for (j = G_diag_i[i]; j < G_diag_i[i + 1]; j++)
      {
         edge = G_diag_j[j];
         if (GT_diag_mat[edge * 2] == -1) { GT_diag_mat[edge * 2] = i; }
         else { GT_diag_mat[edge * 2 + 1] = i; }
      }
   }

   /* BFS on the local matrix graph to find tree */

   nodes_marked = nalu_hypre_TAlloc(NALU_HYPRE_Int, nrows_G, NALU_HYPRE_MEMORY_HOST);
   edges_marked = nalu_hypre_TAlloc(NALU_HYPRE_Int, ncols_G, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nrows_G; i++) { nodes_marked[i] = 0; }
   for (i = 0; i < ncols_G; i++) { edges_marked[i] = 0; }
   queue = nalu_hypre_TAlloc(NALU_HYPRE_Int, nrows_G, NALU_HYPRE_MEMORY_HOST);
   queue_head = 0;
   queue_tail = 1;
   queue[0] = 0;
   nodes_marked[0] = 1;
   while ((queue_tail - queue_head) > 0)
   {
      node = queue[queue_tail - 1];
      queue_tail--;
      for (i = G_diag_i[node]; i < G_diag_i[node + 1]; i++)
      {
         edge = G_diag_j[i];
         if (edges_marked[edge] == 0)
         {
            if (GT_diag_mat[2 * edge + 1] != -1)
            {
               node2 = GT_diag_mat[2 * edge];
               if (node2 == node) { node2 = GT_diag_mat[2 * edge + 1]; }
               if (nodes_marked[node2] == 0)
               {
                  nodes_marked[node2] = 1;
                  edges_marked[edge] = 1;
                  queue[queue_tail] = node2;
                  queue_tail++;
               }
            }
         }
      }
   }
   nalu_hypre_TFree(nodes_marked, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(queue, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(GT_diag_mat, NALU_HYPRE_MEMORY_HOST);

   /* fetch the communication information from */

   comm = nalu_hypre_ParCSRMatrixComm(G_csr);
   nalu_hypre_MPI_Comm_rank(comm, &mypid);
   nalu_hypre_MPI_Comm_size(comm, &nprocs);
   comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(G_csr);
   if (nprocs == 1 && comm_pkg == NULL)
   {

      nalu_hypre_MatvecCommPkgCreate((nalu_hypre_ParCSRMatrix *) G_csr);

      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(G_csr);
   }

   /* construct processor graph based on node-edge connection */
   /* (local edges connected to neighbor processor nodes)     */

   n_children = 0;
   nrecvs = nsends = 0;
   if (nprocs > 1)
   {
      nsends     = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      send_procs = nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg);
      nrecvs     = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
      recv_procs = nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg);
      proc_array = NULL;
      if ((nsends + nrecvs) > 0)
      {
         n_proc_array = 0;
         proc_array = nalu_hypre_TAlloc(NALU_HYPRE_Int, (nsends + nrecvs), NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < nsends; i++) { proc_array[i] = send_procs[i]; }
         for (i = 0; i < nrecvs; i++) { proc_array[nsends + i] = recv_procs[i]; }
         nalu_hypre_qsort0(proc_array, 0, nsends + nrecvs - 1);
         n_proc_array = 1;
         for (i = 1; i < nrecvs + nsends; i++)
            if (proc_array[i] != proc_array[n_proc_array])
            {
               proc_array[n_proc_array++] = proc_array[i];
            }
      }
      pgraph_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, (nprocs + 1), NALU_HYPRE_MEMORY_HOST);
      recv_cnts = nalu_hypre_TAlloc(NALU_HYPRE_Int, nprocs, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_MPI_Allgather(&n_proc_array, 1, NALU_HYPRE_MPI_INT, recv_cnts, 1,
                          NALU_HYPRE_MPI_INT, comm);
      pgraph_i[0] = 0;
      for (i = 1; i <= nprocs; i++)
      {
         pgraph_i[i] = pgraph_i[i - 1] + recv_cnts[i - 1];
      }
      pgraph_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, pgraph_i[nprocs], NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_MPI_Allgatherv(proc_array, n_proc_array, NALU_HYPRE_MPI_INT, pgraph_j,
                           recv_cnts, pgraph_i, NALU_HYPRE_MPI_INT, comm);
      nalu_hypre_TFree(recv_cnts, NALU_HYPRE_MEMORY_HOST);

      /* BFS on the processor graph to determine parent and children */

      nodes_marked = nalu_hypre_TAlloc(NALU_HYPRE_Int, nprocs, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < nprocs; i++) { nodes_marked[i] = -1; }
      queue = nalu_hypre_TAlloc(NALU_HYPRE_Int, nprocs, NALU_HYPRE_MEMORY_HOST);
      queue_head = 0;
      queue_tail = 1;
      node = 0;
      queue[0] = node;
      while ((queue_tail - queue_head) > 0)
      {
         proc = queue[queue_tail - 1];
         queue_tail--;
         for (i = pgraph_i[proc]; i < pgraph_i[proc + 1]; i++)
         {
            proc2 = pgraph_j[i];
            if (nodes_marked[proc2] < 0)
            {
               nodes_marked[proc2] = proc;
               queue[queue_tail] = proc2;
               queue_tail++;
            }
         }
      }
      parent = nodes_marked[mypid];
      n_children = 0;
      for (i = 0; i < nprocs; i++) if (nodes_marked[i] == mypid) { n_children++; }
      if (n_children == 0) {n_children = 0; children = NULL;}
      else
      {
         children = nalu_hypre_TAlloc(NALU_HYPRE_Int, n_children, NALU_HYPRE_MEMORY_HOST);
         n_children = 0;
         for (i = 0; i < nprocs; i++)
            if (nodes_marked[i] == mypid) { children[n_children++] = i; }
      }
      nalu_hypre_TFree(nodes_marked, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(queue, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(pgraph_i, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(pgraph_j, NALU_HYPRE_MEMORY_HOST);
   }

   /* first, connection with my parent : if the edge in my parent *
    * is incident to one of my nodes, then my parent will mark it */

   found = 0;
   for (i = 0; i < nrecvs; i++)
   {
      proc = nalu_hypre_ParCSRCommPkgRecvProc(comm_pkg, i);
      if (proc == parent)
      {
         found = 1;
         break;
      }
   }

   /* but if all the edges connected to my parent are on my side, *
    * then I will just pick one of them as tree edge              */

   if (found == 0)
   {
      for (i = 0; i < nsends; i++)
      {
         proc = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
         if (proc == parent)
         {
            k = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            edge = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, k);
            edges_marked[edge] = 1;
            break;
         }
      }
   }

   /* next, if my processor has an edge incident on one node in my *
    * child, put this edge on the tree. But if there is no such    *
    * edge, then I will assume my child will pick up an edge       */

   for (j = 0; j < n_children; j++)
   {
      proc = children[j];
      for (i = 0; i < nsends; i++)
      {
         proc2 = nalu_hypre_ParCSRCommPkgSendProc(comm_pkg, i);
         if (proc == proc2)
         {
            k = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            edge = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, k);
            edges_marked[edge] = 1;
            break;
         }
      }
   }
   if (n_children > 0)
   {
      nalu_hypre_TFree(children, NALU_HYPRE_MEMORY_HOST);
   }

   /* count the size of the tree */

   tree_size = 0;
   for (i = 0; i < ncols_G; i++)
      if (edges_marked[i] == 1) { tree_size++; }
   t_indices = nalu_hypre_TAlloc(NALU_HYPRE_Int, (tree_size + 1), NALU_HYPRE_MEMORY_HOST);
   t_indices[0] = tree_size;
   tree_size = 1;
   for (i = 0; i < ncols_G; i++)
      if (edges_marked[i] == 1) { t_indices[tree_size++] = i; }
   (*indices) = t_indices;
   nalu_hypre_TFree(edges_marked, NALU_HYPRE_MEMORY_HOST);
   if (G_type != 0)
   {
      nalu_hypre_TFree(G_diag_i, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(G_diag_j, NALU_HYPRE_MEMORY_HOST);
   }
}

/* -----------------------------------------------------------------------------
 * extract submatrices based on given indices
 * ----------------------------------------------------------------------------- */

void nalu_hypre_ParCSRMatrixExtractSubmatrices( nalu_hypre_ParCSRMatrix *A_csr,
                                           NALU_HYPRE_Int *indices2,
                                           nalu_hypre_ParCSRMatrix ***submatrices )
{
   NALU_HYPRE_Int    nrows_A, nindices, *indices, *A_diag_i, *A_diag_j, mypid, nprocs;
   NALU_HYPRE_Int    i, j, k, *proc_offsets1, *proc_offsets2, *exp_indices;
   NALU_HYPRE_BigInt *itmp_array;
   NALU_HYPRE_Int    nnz11, nnz12, nnz21, nnz22, col, ncols_offd, nnz_offd, nnz_diag;
   NALU_HYPRE_Int    nrows, nnz;
   NALU_HYPRE_BigInt global_nrows, global_ncols, *row_starts, *col_starts;
   NALU_HYPRE_Int    *diag_i, *diag_j, row, *offd_i;
   NALU_HYPRE_Complex *A_diag_a, *diag_a;
   nalu_hypre_ParCSRMatrix *A11_csr, *A12_csr, *A21_csr, *A22_csr;
   nalu_hypre_CSRMatrix    *A_diag, *diag, *offd;
   MPI_Comm           comm;

   /* -----------------------------------------------------
    * first make sure the incoming indices are in order
    * ----------------------------------------------------- */

   nindices = indices2[0];
   indices  = &(indices2[1]);
   nalu_hypre_qsort0(indices, 0, nindices - 1);

   /* -----------------------------------------------------
    * fetch matrix information
    * ----------------------------------------------------- */

   nrows_A = (NALU_HYPRE_Int) nalu_hypre_ParCSRMatrixGlobalNumRows(A_csr);
   A_diag = nalu_hypre_ParCSRMatrixDiag(A_csr);
   A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   A_diag_a = nalu_hypre_CSRMatrixData(A_diag);
   comm = nalu_hypre_ParCSRMatrixComm(A_csr);
   nalu_hypre_MPI_Comm_rank(comm, &mypid);
   nalu_hypre_MPI_Comm_size(comm, &nprocs);
   if (nprocs > 1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ExtractSubmatrices: cannot handle nprocs > 1 yet.\n");
      exit(1);
   }

   /* -----------------------------------------------------
    * compute new matrix dimensions
    * ----------------------------------------------------- */

   proc_offsets1 = nalu_hypre_TAlloc(NALU_HYPRE_Int, (nprocs + 1), NALU_HYPRE_MEMORY_HOST);
   proc_offsets2 = nalu_hypre_TAlloc(NALU_HYPRE_Int, (nprocs + 1), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_MPI_Allgather(&nindices, 1, NALU_HYPRE_MPI_INT, proc_offsets1, 1,
                       NALU_HYPRE_MPI_INT, comm);
   k = 0;
   for (i = 0; i < nprocs; i++)
   {
      j = proc_offsets1[i];
      proc_offsets1[i] = k;
      k += j;
   }
   proc_offsets1[nprocs] = k;
   itmp_array = nalu_hypre_ParCSRMatrixRowStarts(A_csr);
   for (i = 0; i <= nprocs; i++)
   {
      proc_offsets2[i] = itmp_array[i] - proc_offsets1[i];
   }

   /* -----------------------------------------------------
    * assign id's to row and col for later processing
    * ----------------------------------------------------- */

   exp_indices = nalu_hypre_TAlloc(NALU_HYPRE_Int, nrows_A, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nrows_A; i++) { exp_indices[i] = -1; }
   for (i = 0; i < nindices; i++)
   {
      if (exp_indices[indices[i]] == -1) { exp_indices[indices[i]] = i; }
      else
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ExtractSubmatrices: wrong index %d %d\n");
         exit(1);
      }
   }
   k = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         exp_indices[i] = - k - 1;
         k++;
      }
   }

   /* -----------------------------------------------------
    * compute number of nonzeros for each block
    * ----------------------------------------------------- */

   nnz11 = nnz12 = nnz21 = nnz22 = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0) { nnz11++; }
            else { nnz12++; }
         }
      }
      else
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0) { nnz21++; }
            else { nnz22++; }
         }
      }
   }

   /* -----------------------------------------------------
    * create A11 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz11;
   /* This case is not yet implemented! */
   global_nrows = 0;
   global_ncols = 0;
   row_starts = NULL;
   col_starts = NULL;
   A11_csr = nalu_hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd);
   nrows = nindices;
   diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows + 1, NALU_HYPRE_MEMORY_HOST);
   diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_diag, NALU_HYPRE_MEMORY_HOST);
   diag_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  nnz_diag, NALU_HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0)
            {
               diag_j[nnz] = exp_indices[col];
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = nalu_hypre_ParCSRMatrixDiag(A11_csr);
   nalu_hypre_CSRMatrixI(diag) = diag_i;
   nalu_hypre_CSRMatrixJ(diag) = diag_j;
   nalu_hypre_CSRMatrixData(diag) = diag_a;

   offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= nrows; i++) { offd_i[i] = 0; }
   offd = nalu_hypre_ParCSRMatrixOffd(A11_csr);
   nalu_hypre_CSRMatrixI(offd) = offd_i;
   nalu_hypre_CSRMatrixJ(offd) = NULL;
   nalu_hypre_CSRMatrixData(offd) = NULL;

   /* -----------------------------------------------------
    * create A12 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz12;
   global_nrows = (NALU_HYPRE_BigInt)proc_offsets1[nprocs];
   global_ncols = (NALU_HYPRE_BigInt)proc_offsets2[nprocs];
   row_starts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   col_starts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = (NALU_HYPRE_BigInt)proc_offsets1[i];
      col_starts[i] = (NALU_HYPRE_BigInt)proc_offsets2[i];
   }
   A12_csr = nalu_hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd);
   nrows = nindices;
   diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows + 1, NALU_HYPRE_MEMORY_HOST);
   diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_diag, NALU_HYPRE_MEMORY_HOST);
   diag_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  nnz_diag, NALU_HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] < 0)
            {
               diag_j[nnz] = - exp_indices[col] - 1;
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }

   if (nnz > nnz_diag)
   {
      nalu_hypre_assert(0);
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
   }

   diag = nalu_hypre_ParCSRMatrixDiag(A12_csr);
   nalu_hypre_CSRMatrixI(diag) = diag_i;
   nalu_hypre_CSRMatrixJ(diag) = diag_j;
   nalu_hypre_CSRMatrixData(diag) = diag_a;

   offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= nrows; i++) { offd_i[i] = 0; }
   offd = nalu_hypre_ParCSRMatrixOffd(A12_csr);
   nalu_hypre_CSRMatrixI(offd) = offd_i;
   nalu_hypre_CSRMatrixJ(offd) = NULL;
   nalu_hypre_CSRMatrixData(offd) = NULL;
   nalu_hypre_TFree(row_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(col_starts, NALU_HYPRE_MEMORY_HOST);

   /* -----------------------------------------------------
    * create A21 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz21;
   global_nrows = (NALU_HYPRE_BigInt)proc_offsets2[nprocs];
   global_ncols = (NALU_HYPRE_BigInt)proc_offsets1[nprocs];
   row_starts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   col_starts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = (NALU_HYPRE_BigInt)proc_offsets2[i];
      col_starts[i] = (NALU_HYPRE_BigInt)proc_offsets1[i];
   }
   A21_csr = nalu_hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd);
   nrows = nrows_A - nindices;
   diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows + 1, NALU_HYPRE_MEMORY_HOST);
   diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_diag, NALU_HYPRE_MEMORY_HOST);
   diag_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  nnz_diag, NALU_HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0)
            {
               diag_j[nnz] = exp_indices[col];
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = nalu_hypre_ParCSRMatrixDiag(A21_csr);
   nalu_hypre_CSRMatrixI(diag) = diag_i;
   nalu_hypre_CSRMatrixJ(diag) = diag_j;
   nalu_hypre_CSRMatrixData(diag) = diag_a;

   offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= nrows; i++) { offd_i[i] = 0; }
   offd = nalu_hypre_ParCSRMatrixOffd(A21_csr);
   nalu_hypre_CSRMatrixI(offd) = offd_i;
   nalu_hypre_CSRMatrixJ(offd) = NULL;
   nalu_hypre_CSRMatrixData(offd) = NULL;
   nalu_hypre_TFree(row_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(col_starts, NALU_HYPRE_MEMORY_HOST);

   /* -----------------------------------------------------
    * create A22 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = 0;
   nnz_offd   = 0;
   nnz_diag   = nnz22;
   global_nrows = (NALU_HYPRE_BigInt)proc_offsets2[nprocs];
   global_ncols = (NALU_HYPRE_BigInt)proc_offsets2[nprocs];
   row_starts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   col_starts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = (NALU_HYPRE_BigInt)proc_offsets2[i];
      col_starts[i] = (NALU_HYPRE_BigInt)proc_offsets2[i];
   }
   A22_csr = nalu_hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd);
   nrows = nrows_A - nindices;
   diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows + 1, NALU_HYPRE_MEMORY_HOST);
   diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_diag, NALU_HYPRE_MEMORY_HOST);
   diag_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  nnz_diag, NALU_HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] < 0)
            {
               diag_j[nnz] = - exp_indices[col] - 1;
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = nalu_hypre_ParCSRMatrixDiag(A22_csr);
   nalu_hypre_CSRMatrixI(diag) = diag_i;
   nalu_hypre_CSRMatrixJ(diag) = diag_j;
   nalu_hypre_CSRMatrixData(diag) = diag_a;

   offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= nrows; i++) { offd_i[i] = 0; }
   offd = nalu_hypre_ParCSRMatrixOffd(A22_csr);
   nalu_hypre_CSRMatrixI(offd) = offd_i;
   nalu_hypre_CSRMatrixJ(offd) = NULL;
   nalu_hypre_CSRMatrixData(offd) = NULL;
   nalu_hypre_TFree(row_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(col_starts, NALU_HYPRE_MEMORY_HOST);

   /* -----------------------------------------------------
    * hand the matrices back to the caller and clean up
    * ----------------------------------------------------- */

   (*submatrices)[0] = A11_csr;
   (*submatrices)[1] = A12_csr;
   (*submatrices)[2] = A21_csr;
   (*submatrices)[3] = A22_csr;
   nalu_hypre_TFree(proc_offsets1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(proc_offsets2, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(exp_indices, NALU_HYPRE_MEMORY_HOST);
}

/* -----------------------------------------------------------------------------
 * extract submatrices of a rectangular matrix
 * ----------------------------------------------------------------------------- */

void nalu_hypre_ParCSRMatrixExtractRowSubmatrices( nalu_hypre_ParCSRMatrix *A_csr,
                                              NALU_HYPRE_Int *indices2,
                                              nalu_hypre_ParCSRMatrix ***submatrices )
{
   NALU_HYPRE_Int    nrows_A, nindices, *indices, *A_diag_i, *A_diag_j, mypid, nprocs;
   NALU_HYPRE_Int    i, j, k, *proc_offsets1, *proc_offsets2, *exp_indices;
   NALU_HYPRE_Int    nnz11, nnz21, col, ncols_offd, nnz_offd, nnz_diag;
   NALU_HYPRE_Int    *A_offd_i, *A_offd_j;
   NALU_HYPRE_Int    nrows, nnz;
   NALU_HYPRE_BigInt global_nrows, global_ncols, *row_starts, *col_starts, *itmp_array;
   NALU_HYPRE_Int    *diag_i, *diag_j, row, *offd_i, *offd_j, nnz11_offd, nnz21_offd;
   NALU_HYPRE_Complex *A_diag_a, *diag_a, *offd_a;
   nalu_hypre_ParCSRMatrix *A11_csr, *A21_csr;
   nalu_hypre_CSRMatrix    *A_diag, *diag, *A_offd, *offd;
   MPI_Comm           comm;

   /* -----------------------------------------------------
    * first make sure the incoming indices are in order
    * ----------------------------------------------------- */

   nindices = indices2[0];
   indices  = &(indices2[1]);
   nalu_hypre_qsort0(indices, 0, nindices - 1);

   /* -----------------------------------------------------
    * fetch matrix information
    * ----------------------------------------------------- */

   nrows_A = (NALU_HYPRE_Int)nalu_hypre_ParCSRMatrixGlobalNumRows(A_csr);
   A_diag = nalu_hypre_ParCSRMatrixDiag(A_csr);
   A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   A_diag_a = nalu_hypre_CSRMatrixData(A_diag);
   A_offd = nalu_hypre_ParCSRMatrixOffd(A_csr);
   A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   comm = nalu_hypre_ParCSRMatrixComm(A_csr);
   nalu_hypre_MPI_Comm_rank(comm, &mypid);
   nalu_hypre_MPI_Comm_size(comm, &nprocs);

   /* -----------------------------------------------------
    * compute new matrix dimensions
    * ----------------------------------------------------- */

   proc_offsets1 = nalu_hypre_TAlloc(NALU_HYPRE_Int, (nprocs + 1), NALU_HYPRE_MEMORY_HOST);
   proc_offsets2 = nalu_hypre_TAlloc(NALU_HYPRE_Int, (nprocs + 1), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_MPI_Allgather(&nindices, 1, NALU_HYPRE_MPI_INT, proc_offsets1, 1,
                       NALU_HYPRE_MPI_INT, comm);
   k = 0;
   for (i = 0; i < nprocs; i++)
   {
      j = proc_offsets1[i];
      proc_offsets1[i] = k;
      k += j;
   }
   proc_offsets1[nprocs] = k;
   itmp_array = nalu_hypre_ParCSRMatrixRowStarts(A_csr);
   for (i = 0; i <= nprocs; i++)
   {
      proc_offsets2[i] = (NALU_HYPRE_Int)(itmp_array[i] - proc_offsets1[i]);
   }

   /* -----------------------------------------------------
    * assign id's to row and col for later processing
    * ----------------------------------------------------- */

   exp_indices = nalu_hypre_TAlloc(NALU_HYPRE_Int, nrows_A, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nrows_A; i++) { exp_indices[i] = -1; }
   for (i = 0; i < nindices; i++)
   {
      if (exp_indices[indices[i]] == -1) { exp_indices[indices[i]] = i; }
      else
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ExtractRowSubmatrices: wrong index %d %d\n");
         exit(1);
      }
   }
   k = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         exp_indices[i] = - k - 1;
         k++;
      }
   }

   /* -----------------------------------------------------
    * compute number of nonzeros for each block
    * ----------------------------------------------------- */

   nnz11 = nnz21 = nnz11_offd = nnz21_offd = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0) { nnz11++; }
         }
         nnz11_offd += A_offd_i[i + 1] - A_offd_i[i];
      }
      else
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] < 0) { nnz21++; }
         }
         nnz21_offd += A_offd_i[i + 1] - A_offd_i[i];
      }
   }

   /* -----------------------------------------------------
    * create A11 matrix (assume sequential for the moment)
    * ----------------------------------------------------- */

   ncols_offd = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixDiag(A_csr));
   nnz_diag   = nnz11;
   nnz_offd   = nnz11_offd;

   global_nrows = (NALU_HYPRE_BigInt)proc_offsets1[nprocs];
   itmp_array   = nalu_hypre_ParCSRMatrixColStarts(A_csr);
   global_ncols = itmp_array[nprocs];
   row_starts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   col_starts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = (NALU_HYPRE_BigInt)proc_offsets1[i];
      col_starts[i] = itmp_array[i];
   }
   A11_csr = nalu_hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd);
   nrows = nindices;
   diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows + 1, NALU_HYPRE_MEMORY_HOST);
   diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_diag, NALU_HYPRE_MEMORY_HOST);
   diag_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  nnz_diag, NALU_HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            if (exp_indices[col] >= 0)
            {
               diag_j[nnz] = exp_indices[col];
               diag_a[nnz++] = A_diag_a[j];
            }
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = nalu_hypre_ParCSRMatrixDiag(A11_csr);
   nalu_hypre_CSRMatrixI(diag) = diag_i;
   nalu_hypre_CSRMatrixJ(diag) = diag_j;
   nalu_hypre_CSRMatrixData(diag) = diag_a;

   offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows + 1, NALU_HYPRE_MEMORY_HOST);
   offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_offd, NALU_HYPRE_MEMORY_HOST);
   offd_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  nnz_offd, NALU_HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   offd_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] >= 0)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            offd_j[nnz] = A_offd_j[j];
            offd_a[nnz++] = A_diag_a[j];
         }
         row++;
         offd_i[row] = nnz;
      }
   }
   offd = nalu_hypre_ParCSRMatrixOffd(A11_csr);
   nalu_hypre_CSRMatrixI(offd) = offd_i;
   nalu_hypre_CSRMatrixJ(offd) = offd_j;
   nalu_hypre_CSRMatrixData(offd) = offd_a;
   nalu_hypre_TFree(row_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(col_starts, NALU_HYPRE_MEMORY_HOST);

   /* -----------------------------------------------------
    * create A21 matrix
    * ----------------------------------------------------- */

   ncols_offd = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixDiag(A_csr));
   nnz_offd   = nnz21_offd;
   nnz_diag   = nnz21;
   global_nrows = (NALU_HYPRE_BigInt)proc_offsets2[nprocs];
   itmp_array   = nalu_hypre_ParCSRMatrixColStarts(A_csr);
   global_ncols = itmp_array[nprocs];
   row_starts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   col_starts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i <= nprocs; i++)
   {
      row_starts[i] = (NALU_HYPRE_BigInt)proc_offsets2[i];
      col_starts[i] = itmp_array[i];
   }
   A21_csr = nalu_hypre_ParCSRMatrixCreate(comm, global_nrows, global_ncols,
                                      row_starts, col_starts, ncols_offd,
                                      nnz_diag, nnz_offd);
   nrows = nrows_A - nindices;
   diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows + 1, NALU_HYPRE_MEMORY_HOST);
   diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_diag, NALU_HYPRE_MEMORY_HOST);
   diag_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  nnz_diag, NALU_HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   diag_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            diag_j[nnz] = A_diag_j[j];
            diag_a[nnz++] = A_diag_a[j];
         }
         row++;
         diag_i[row] = nnz;
      }
   }
   diag = nalu_hypre_ParCSRMatrixDiag(A21_csr);
   nalu_hypre_CSRMatrixI(diag) = diag_i;
   nalu_hypre_CSRMatrixJ(diag) = diag_j;
   nalu_hypre_CSRMatrixData(diag) = diag_a;

   offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows + 1, NALU_HYPRE_MEMORY_HOST);
   offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nnz_offd, NALU_HYPRE_MEMORY_HOST);
   offd_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  nnz_offd, NALU_HYPRE_MEMORY_HOST);
   nnz = 0;
   row = 0;
   offd_i[0] = 0;
   for (i = 0; i < nrows_A; i++)
   {
      if (exp_indices[i] < 0)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            offd_j[nnz] = A_offd_j[j];
            offd_a[nnz++] = A_diag_a[j];
         }
         row++;
         offd_i[row] = nnz;
      }
   }
   offd = nalu_hypre_ParCSRMatrixOffd(A21_csr);
   nalu_hypre_CSRMatrixI(offd) = offd_i;
   nalu_hypre_CSRMatrixJ(offd) = offd_j;
   nalu_hypre_CSRMatrixData(offd) = offd_a;
   nalu_hypre_TFree(row_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(col_starts, NALU_HYPRE_MEMORY_HOST);

   /* -----------------------------------------------------
    * hand the matrices back to the caller and clean up
    * ----------------------------------------------------- */

   (*submatrices)[0] = A11_csr;
   (*submatrices)[1] = A21_csr;
   nalu_hypre_TFree(proc_offsets1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(proc_offsets2, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(exp_indices, NALU_HYPRE_MEMORY_HOST);
}

/* -----------------------------------------------------------------------------
 * return the sum of all local elements of the matrix
 * ----------------------------------------------------------------------------- */

NALU_HYPRE_Complex nalu_hypre_ParCSRMatrixLocalSumElts( nalu_hypre_ParCSRMatrix * A )
{
   nalu_hypre_CSRMatrix * A_diag = nalu_hypre_ParCSRMatrixDiag( A );
   nalu_hypre_CSRMatrix * A_offd = nalu_hypre_ParCSRMatrixOffd( A );

   return nalu_hypre_CSRMatrixSumElts(A_diag) + nalu_hypre_CSRMatrixSumElts(A_offd);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixMatAminvDB
 * computes C = (A - inv(D)B) where D is a diagonal matrix
 * Note: Data structure of A is expected to be a subset of data structure of B!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixAminvDB( nalu_hypre_ParCSRMatrix  *A,
                           nalu_hypre_ParCSRMatrix  *B,
                           NALU_HYPRE_Complex       *d,
                           nalu_hypre_ParCSRMatrix **C_ptr)
{
   MPI_Comm              comm            = nalu_hypre_ParCSRMatrixComm(B);
   nalu_hypre_CSRMatrix      *A_diag          = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix      *A_offd          = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int             num_cols_offd_A = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_ParCSRCommPkg  *comm_pkg_B      = nalu_hypre_ParCSRMatrixCommPkg(B);
   nalu_hypre_CSRMatrix      *B_diag          = nalu_hypre_ParCSRMatrixDiag(B);
   nalu_hypre_CSRMatrix      *B_offd          = nalu_hypre_ParCSRMatrixOffd(B);
   NALU_HYPRE_Int             num_cols_offd_B = nalu_hypre_CSRMatrixNumCols(B_offd);
   NALU_HYPRE_Int             num_sends_B;
   NALU_HYPRE_Int             num_recvs_B;
   NALU_HYPRE_Int             i, j, cnt;

   NALU_HYPRE_Int            *A_diag_i       = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int            *A_diag_j       = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Complex        *A_diag_data    = nalu_hypre_CSRMatrixData(A_diag);

   NALU_HYPRE_Int            *A_offd_i       = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int            *A_offd_j       = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Complex        *A_offd_data    = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_BigInt         *col_map_offd_A = nalu_hypre_ParCSRMatrixColMapOffd(A);

   NALU_HYPRE_Int             num_rows       = nalu_hypre_CSRMatrixNumRows(B_diag);
   NALU_HYPRE_Int            *B_diag_i       = nalu_hypre_CSRMatrixI(B_diag);
   NALU_HYPRE_Int            *B_diag_j       = nalu_hypre_CSRMatrixJ(B_diag);
   NALU_HYPRE_Complex        *B_diag_data    = nalu_hypre_CSRMatrixData(B_diag);

   NALU_HYPRE_Int            *B_offd_i       = nalu_hypre_CSRMatrixI(B_offd);
   NALU_HYPRE_Int            *B_offd_j       = nalu_hypre_CSRMatrixJ(B_offd);
   NALU_HYPRE_Complex        *B_offd_data    = nalu_hypre_CSRMatrixData(B_offd);
   NALU_HYPRE_BigInt         *col_map_offd_B = nalu_hypre_ParCSRMatrixColMapOffd(B);

   nalu_hypre_ParCSRMatrix   *C           = NULL;
   nalu_hypre_CSRMatrix      *C_diag      = NULL;
   nalu_hypre_CSRMatrix      *C_offd      = NULL;
   NALU_HYPRE_Int            *C_diag_i    = NULL;
   NALU_HYPRE_Int            *C_diag_j    = NULL;
   NALU_HYPRE_Complex        *C_diag_data = NULL;
   NALU_HYPRE_Int            *C_offd_i    = NULL;
   NALU_HYPRE_Int            *C_offd_j    = NULL;
   NALU_HYPRE_Complex        *C_offd_data = NULL;

   NALU_HYPRE_Int             num_procs, my_id;
   NALU_HYPRE_Int            *recv_procs_B;
   NALU_HYPRE_Int            *send_procs_B;
   NALU_HYPRE_Int            *recv_vec_starts_B;
   NALU_HYPRE_Int            *send_map_starts_B;
   NALU_HYPRE_Int            *send_map_elmts_B;
   nalu_hypre_ParCSRCommPkg  *comm_pkg_C = NULL;
   NALU_HYPRE_Int            *recv_procs_C;
   NALU_HYPRE_Int            *send_procs_C;
   NALU_HYPRE_Int            *recv_vec_starts_C;
   NALU_HYPRE_Int            *send_map_starts_C;
   NALU_HYPRE_Int            *send_map_elmts_C;
   NALU_HYPRE_Int            *map_to_B;
   NALU_HYPRE_Complex        *D_tmp;
   NALU_HYPRE_Int             size, rest, num_threads, ii;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   num_threads = nalu_hypre_NumThreads();

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for B, a CommPkg is generated
    *--------------------------------------------------------------------*/

   if (!comm_pkg_B)
   {
      nalu_hypre_MatvecCommPkgCreate(B);
      comm_pkg_B = nalu_hypre_ParCSRMatrixCommPkg(B);
   }

   C = nalu_hypre_ParCSRMatrixClone(B, 0);
   /*nalu_hypre_ParCSRMatrixInitialize(C);*/

   C_diag = nalu_hypre_ParCSRMatrixDiag(C);
   C_diag_i = nalu_hypre_CSRMatrixI(C_diag);
   C_diag_j = nalu_hypre_CSRMatrixJ(C_diag);
   C_diag_data = nalu_hypre_CSRMatrixData(C_diag);
   C_offd = nalu_hypre_ParCSRMatrixOffd(C);
   C_offd_i = nalu_hypre_CSRMatrixI(C_offd);
   C_offd_j = nalu_hypre_CSRMatrixJ(C_offd);
   C_offd_data = nalu_hypre_CSRMatrixData(C_offd);

   size = num_rows / num_threads;
   rest = num_rows - size * num_threads;

   D_tmp = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, num_rows, NALU_HYPRE_MEMORY_HOST);

   if (num_cols_offd_A)
   {
      map_to_B = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_A, NALU_HYPRE_MEMORY_HOST);
      cnt = 0;
      for (i = 0; i < num_cols_offd_A; i++)
      {
         while (col_map_offd_B[cnt] < col_map_offd_A[i])
         {
            cnt++;
         }
         map_to_B[i] = cnt;
         cnt++;
      }
   }

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(ii, i, j)
#endif
   for (ii = 0; ii < num_threads; ii++)
   {
      NALU_HYPRE_Int *A_marker = NULL;
      NALU_HYPRE_Int ns, ne, A_col, num_cols, nmax;
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
      nmax = nalu_hypre_max(num_rows, num_cols_offd_B);
      A_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nmax, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_rows; i++)
      {
         A_marker[i] = -1;
      }

      for (i = ns; i < ne; i++)
      {
         D_tmp[i] = 1.0 / d[i];
      }

      num_cols = C_diag_i[ns];
      for (i = ns; i < ne; i++)
      {
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            A_col = A_diag_j[j];
            if (A_marker[A_col] < C_diag_i[i])
            {
               A_marker[A_col] = num_cols;
               C_diag_j[num_cols] = A_col;
               C_diag_data[num_cols] = A_diag_data[j];
               num_cols++;
            }
            else
            {
               C_diag_data[A_marker[A_col]] += A_diag_data[j];
            }
         }
         for (j = B_diag_i[i]; j < B_diag_i[i + 1]; j++)
         {
            A_col = B_diag_j[j];
            if (A_marker[A_col] < C_diag_i[i])
            {
               A_marker[A_col] = num_cols;
               C_diag_j[num_cols] = A_col;
               C_diag_data[num_cols] = -D_tmp[i] * B_diag_data[j];
               num_cols++;
            }
            else
            {
               C_diag_data[A_marker[A_col]] -= D_tmp[i] * B_diag_data[j];
            }
         }
      }

      for (i = 0; i < num_cols_offd_B; i++)
      {
         A_marker[i] = -1;
      }

      num_cols = C_offd_i[ns];
      for (i = ns; i < ne; i++)
      {
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            A_col = map_to_B[A_offd_j[j]];
            if (A_marker[A_col] < B_offd_i[i])
            {
               A_marker[A_col] = num_cols;
               C_offd_j[num_cols] = A_col;
               C_offd_data[num_cols] = A_offd_data[j];
               num_cols++;
            }
            else
            {
               C_offd_data[A_marker[A_col]] += A_offd_data[j];
            }
         }
         for (j = B_offd_i[i]; j < B_offd_i[i + 1]; j++)
         {
            A_col = B_offd_j[j];
            if (A_marker[A_col] < B_offd_i[i])
            {
               A_marker[A_col] = num_cols;
               C_offd_j[num_cols] = A_col;
               C_offd_data[num_cols] = -D_tmp[i] * B_offd_data[j];
               num_cols++;
            }
            else
            {
               C_offd_data[A_marker[A_col]] -= D_tmp[i] * B_offd_data[j];
            }
         }
      }
      nalu_hypre_TFree(A_marker, NALU_HYPRE_MEMORY_HOST);

   } /* end parallel region */

   /*for (i=0; i < num_cols_offd_B; i++)
     col_map_offd_C[i] = col_map_offd_B[i]; */

   num_sends_B       = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg_B);
   num_recvs_B       = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg_B);
   recv_procs_B      = nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg_B);
   recv_vec_starts_B = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_B);
   send_procs_B      = nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg_B);
   send_map_starts_B = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg_B);
   send_map_elmts_B  = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg_B);

   recv_procs_C      = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_recvs_B, NALU_HYPRE_MEMORY_HOST);
   recv_vec_starts_C = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_recvs_B + 1, NALU_HYPRE_MEMORY_HOST);
   send_procs_C      = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_sends_B, NALU_HYPRE_MEMORY_HOST);
   send_map_starts_C = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_sends_B + 1, NALU_HYPRE_MEMORY_HOST);
   send_map_elmts_C  = nalu_hypre_CTAlloc(NALU_HYPRE_Int, send_map_starts_B[num_sends_B], NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_recvs_B; i++)
   {
      recv_procs_C[i] = recv_procs_B[i];
   }
   for (i = 0; i < num_recvs_B + 1; i++)
   {
      recv_vec_starts_C[i] = recv_vec_starts_B[i];
   }
   for (i = 0; i < num_sends_B; i++)
   {
      send_procs_C[i] = send_procs_B[i];
   }
   for (i = 0; i < num_sends_B + 1; i++)
   {
      send_map_starts_C[i] = send_map_starts_B[i];
   }
   for (i = 0; i < send_map_starts_B[num_sends_B]; i++)
   {
      send_map_elmts_C[i] = send_map_elmts_B[i];
   }

   /* Create communication package */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs_B, recv_procs_C, recv_vec_starts_C,
                                    num_sends_B, send_procs_C, send_map_starts_C,
                                    send_map_elmts_C,
                                    &comm_pkg_C);

   nalu_hypre_ParCSRMatrixCommPkg(C) = comm_pkg_C;

   nalu_hypre_TFree(D_tmp, NALU_HYPRE_MEMORY_HOST);
   if (num_cols_offd_A)
   {
      nalu_hypre_TFree(map_to_B, NALU_HYPRE_MEMORY_HOST);
   }

   *C_ptr = C;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParTMatmul:
 *
 * Multiplies two ParCSRMatrices transpose(A) and B and returns
 * the product in ParCSRMatrix C
 *
 * Note that C does not own the partitionings since its row_starts
 * is owned by A and col_starts by B.
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRMatrix*
nalu_hypre_ParTMatmul( nalu_hypre_ParCSRMatrix  *A,
                  nalu_hypre_ParCSRMatrix  *B)
{
   MPI_Comm        comm = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg *comm_pkg_A = nalu_hypre_ParCSRMatrixCommPkg(A);

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix *AT_diag = NULL;

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   nalu_hypre_CSRMatrix *AT_offd = NULL;

   NALU_HYPRE_Int    num_rows_diag_A = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int    num_cols_diag_A = nalu_hypre_CSRMatrixNumCols(A_diag);

   nalu_hypre_CSRMatrix *B_diag = nalu_hypre_ParCSRMatrixDiag(B);

   nalu_hypre_CSRMatrix *B_offd = nalu_hypre_ParCSRMatrixOffd(B);
   NALU_HYPRE_BigInt    *col_map_offd_B = nalu_hypre_ParCSRMatrixColMapOffd(B);

   NALU_HYPRE_BigInt    first_col_diag_B = nalu_hypre_ParCSRMatrixFirstColDiag(B);
   NALU_HYPRE_BigInt *col_starts_A = nalu_hypre_ParCSRMatrixColStarts(A);
   NALU_HYPRE_BigInt *col_starts_B = nalu_hypre_ParCSRMatrixColStarts(B);
   NALU_HYPRE_Int    num_rows_diag_B = nalu_hypre_CSRMatrixNumRows(B_diag);
   NALU_HYPRE_Int    num_cols_diag_B = nalu_hypre_CSRMatrixNumCols(B_diag);
   NALU_HYPRE_Int    num_cols_offd_B = nalu_hypre_CSRMatrixNumCols(B_offd);

   nalu_hypre_ParCSRMatrix *C;
   NALU_HYPRE_BigInt       *col_map_offd_C = NULL;
   NALU_HYPRE_Int          *map_B_to_C;

   nalu_hypre_CSRMatrix *C_diag = NULL;
   nalu_hypre_CSRMatrix *C_tmp_diag = NULL;

   NALU_HYPRE_Complex   *C_diag_data = NULL;
   NALU_HYPRE_Int       *C_diag_i = NULL;
   NALU_HYPRE_Int       *C_diag_j = NULL;
   NALU_HYPRE_BigInt    first_col_diag_C;
   NALU_HYPRE_BigInt    last_col_diag_C;

   nalu_hypre_CSRMatrix *C_offd = NULL;
   nalu_hypre_CSRMatrix *C_tmp_offd = NULL;
   nalu_hypre_CSRMatrix *C_int = NULL;
   nalu_hypre_CSRMatrix *C_ext = NULL;
   NALU_HYPRE_Int   *C_ext_i;
   NALU_HYPRE_BigInt   *C_ext_j;
   NALU_HYPRE_Complex   *C_ext_data;
   NALU_HYPRE_Int   *C_ext_diag_i;
   NALU_HYPRE_Int   *C_ext_diag_j;
   NALU_HYPRE_Complex   *C_ext_diag_data;
   NALU_HYPRE_Int   *C_ext_offd_i;
   NALU_HYPRE_Int   *C_ext_offd_j;
   NALU_HYPRE_Complex   *C_ext_offd_data;
   NALU_HYPRE_Int    C_ext_size = 0;
   NALU_HYPRE_Int    C_ext_diag_size = 0;
   NALU_HYPRE_Int    C_ext_offd_size = 0;

   NALU_HYPRE_Int   *C_tmp_diag_i;
   NALU_HYPRE_Int   *C_tmp_diag_j;
   NALU_HYPRE_Complex   *C_tmp_diag_data;
   NALU_HYPRE_Int   *C_tmp_offd_i;
   NALU_HYPRE_Int   *C_tmp_offd_j;
   NALU_HYPRE_Complex   *C_tmp_offd_data;

   NALU_HYPRE_Complex   *C_offd_data = NULL;
   NALU_HYPRE_Int       *C_offd_i = NULL;
   NALU_HYPRE_Int       *C_offd_j = NULL;

   NALU_HYPRE_BigInt    *temp;
   NALU_HYPRE_Int       *send_map_starts_A;
   NALU_HYPRE_Int       *send_map_elmts_A;
   NALU_HYPRE_Int        num_sends_A;

   NALU_HYPRE_Int        num_cols_offd_C = 0;

   NALU_HYPRE_Int       *P_marker;

   NALU_HYPRE_Int        i, j;
   NALU_HYPRE_Int        i1, j_indx;

   NALU_HYPRE_BigInt     nrows_A, ncols_A;
   NALU_HYPRE_BigInt     nrows_B, ncols_B;
   /*NALU_HYPRE_Int              allsquare = 0;*/
   NALU_HYPRE_Int        cnt, cnt_offd, cnt_diag;
   NALU_HYPRE_BigInt     value;
   NALU_HYPRE_Int        num_procs, my_id;
   NALU_HYPRE_Int        max_num_threads;
   NALU_HYPRE_Int       *C_diag_array = NULL;
   NALU_HYPRE_Int       *C_offd_array = NULL;

   NALU_HYPRE_BigInt first_row_index, first_col_diag;
   NALU_HYPRE_Int local_num_rows, local_num_cols;

   nrows_A = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   ncols_A = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
   nrows_B = nalu_hypre_ParCSRMatrixGlobalNumRows(B);
   ncols_B = nalu_hypre_ParCSRMatrixGlobalNumCols(B);

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   max_num_threads = nalu_hypre_NumThreads();

   if (nrows_A != nrows_B || num_rows_diag_A != num_rows_diag_B)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, " Error! Incompatible matrix dimensions!\n");
      return NULL;
   }

   NALU_HYPRE_MemoryLocation memory_location_A = nalu_hypre_ParCSRMatrixMemoryLocation(A);
   NALU_HYPRE_MemoryLocation memory_location_B = nalu_hypre_ParCSRMatrixMemoryLocation(B);

   /* RL: TODO cannot guarantee, maybe should never assert
   nalu_hypre_assert(memory_location_A == memory_location_B);
   */

   /* RL: in the case of A=H, B=D, or A=D, B=H, let C = D,
    * not sure if this is the right thing to do.
    * Also, need something like this in other places
    * TODO */
   NALU_HYPRE_MemoryLocation memory_location_C = nalu_hypre_max(memory_location_A, memory_location_B);

   /*if (num_cols_diag_A == num_cols_diag_B) allsquare = 1;*/

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   if (!comm_pkg_A)
   {
      nalu_hypre_MatvecCommPkgCreate(A);
      comm_pkg_A = nalu_hypre_ParCSRMatrixCommPkg(A);
   }

   nalu_hypre_CSRMatrixTranspose(A_diag, &AT_diag, 1);
   nalu_hypre_CSRMatrixTranspose(A_offd, &AT_offd, 1);

   C_tmp_diag = nalu_hypre_CSRMatrixMultiply(AT_diag, B_diag);
   C_ext_size = 0;
   if (num_procs > 1)
   {
      nalu_hypre_CSRMatrix *C_int_diag;
      nalu_hypre_CSRMatrix *C_int_offd;
      void            *request;

      C_tmp_offd = nalu_hypre_CSRMatrixMultiply(AT_diag, B_offd);
      C_int_diag = nalu_hypre_CSRMatrixMultiply(AT_offd, B_diag);
      C_int_offd = nalu_hypre_CSRMatrixMultiply(AT_offd, B_offd);
      nalu_hypre_ParCSRMatrixDiag(B) = C_int_diag;
      nalu_hypre_ParCSRMatrixOffd(B) = C_int_offd;
      C_int = nalu_hypre_MergeDiagAndOffd(B);
      nalu_hypre_ParCSRMatrixDiag(B) = B_diag;
      nalu_hypre_ParCSRMatrixOffd(B) = B_offd;
      nalu_hypre_ExchangeExternalRowsInit(C_int, comm_pkg_A, &request);
      C_ext = nalu_hypre_ExchangeExternalRowsWait(request);
      C_ext_i = nalu_hypre_CSRMatrixI(C_ext);
      C_ext_j = nalu_hypre_CSRMatrixBigJ(C_ext);
      C_ext_data = nalu_hypre_CSRMatrixData(C_ext);
      C_ext_size = C_ext_i[nalu_hypre_CSRMatrixNumRows(C_ext)];

      nalu_hypre_CSRMatrixDestroy(C_int);
      nalu_hypre_CSRMatrixDestroy(C_int_diag);
      nalu_hypre_CSRMatrixDestroy(C_int_offd);
   }
   else
   {
      C_tmp_offd = nalu_hypre_CSRMatrixCreate(num_cols_diag_A, 0, 0);
      nalu_hypre_CSRMatrixInitialize(C_tmp_offd);
      nalu_hypre_CSRMatrixNumRownnz(C_tmp_offd) = 0;
   }
   nalu_hypre_CSRMatrixDestroy(AT_diag);
   nalu_hypre_CSRMatrixDestroy(AT_offd);

   /*-----------------------------------------------------------------------
    *  Add contents of C_ext to C_tmp_diag and C_tmp_offd
    *  to obtain C_diag and C_offd
    *-----------------------------------------------------------------------*/

   /* check for new nonzero columns in C_offd generated through C_ext */

   first_col_diag_C = first_col_diag_B;
   last_col_diag_C = first_col_diag_B + (NALU_HYPRE_BigInt)num_cols_diag_B - 1;

   C_tmp_diag_i = nalu_hypre_CSRMatrixI(C_tmp_diag);
   if (C_ext_size || num_cols_offd_B)
   {
      NALU_HYPRE_Int C_ext_num_rows;

      num_sends_A = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg_A);
      send_map_starts_A = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg_A);
      send_map_elmts_A = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg_A);
      C_ext_num_rows =  send_map_starts_A[num_sends_A];

      C_ext_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  C_ext_num_rows + 1, NALU_HYPRE_MEMORY_HOST);
      C_ext_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  C_ext_num_rows + 1, NALU_HYPRE_MEMORY_HOST);
      temp = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  C_ext_size + num_cols_offd_B, NALU_HYPRE_MEMORY_HOST);
      C_ext_diag_size = 0;
      C_ext_offd_size = 0;
      for (i = 0; i < C_ext_num_rows; i++)
      {
         for (j = C_ext_i[i]; j < C_ext_i[i + 1]; j++)
         {
            if (C_ext_j[j] < first_col_diag_C ||
                C_ext_j[j] > last_col_diag_C)
            {
               temp[C_ext_offd_size++] = C_ext_j[j];
            }
            else
            {
               C_ext_diag_size++;
            }
         }
         C_ext_diag_i[i + 1] = C_ext_diag_size;
         C_ext_offd_i[i + 1] = C_ext_offd_size;
      }
      cnt = C_ext_offd_size;
      for (i = 0; i < num_cols_offd_B; i++)
      {
         temp[cnt++] = col_map_offd_B[i];
      }

      if (cnt)
      {
         nalu_hypre_BigQsort0(temp, 0, cnt - 1);
         value = temp[0];
         num_cols_offd_C = 1;
         for (i = 1; i < cnt; i++)
         {
            if (temp[i] > value)
            {
               value = temp[i];
               temp[num_cols_offd_C++] = value;
            }
         }
      }

      if (num_cols_offd_C)
      {
         col_map_offd_C = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_C, NALU_HYPRE_MEMORY_HOST);
      }
      for (i = 0; i < num_cols_offd_C; i++)
      {
         col_map_offd_C[i] = temp[i];
      }

      nalu_hypre_TFree(temp, NALU_HYPRE_MEMORY_HOST);

      if (C_ext_diag_size)
      {
         C_ext_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  C_ext_diag_size, NALU_HYPRE_MEMORY_HOST);
         C_ext_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  C_ext_diag_size, NALU_HYPRE_MEMORY_HOST);
      }
      if (C_ext_offd_size)
      {
         C_ext_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  C_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
         C_ext_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  C_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
      }

      C_tmp_diag_j = nalu_hypre_CSRMatrixJ(C_tmp_diag);
      C_tmp_diag_data = nalu_hypre_CSRMatrixData(C_tmp_diag);

      C_tmp_offd_i = nalu_hypre_CSRMatrixI(C_tmp_offd);
      C_tmp_offd_j = nalu_hypre_CSRMatrixJ(C_tmp_offd);
      C_tmp_offd_data = nalu_hypre_CSRMatrixData(C_tmp_offd);

      cnt_offd = 0;
      cnt_diag = 0;
      for (i = 0; i < C_ext_num_rows; i++)
      {
         for (j = C_ext_i[i]; j < C_ext_i[i + 1]; j++)
         {
            if (C_ext_j[j] < first_col_diag_C ||
                C_ext_j[j] > last_col_diag_C)
            {
               C_ext_offd_j[cnt_offd] = nalu_hypre_BigBinarySearch(col_map_offd_C,
                                                              C_ext_j[j],
                                                              num_cols_offd_C);
               C_ext_offd_data[cnt_offd++] = C_ext_data[j];
            }
            else
            {
               C_ext_diag_j[cnt_diag] = (NALU_HYPRE_Int)(C_ext_j[j] - first_col_diag_C);
               C_ext_diag_data[cnt_diag++] = C_ext_data[j];
            }
         }
      }
   }

   if (C_ext)
   {
      nalu_hypre_CSRMatrixDestroy(C_ext);
      C_ext = NULL;
   }

   if (num_cols_offd_B)
   {
      map_B_to_C = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_B, NALU_HYPRE_MEMORY_HOST);

      cnt = 0;
      for (i = 0; i < num_cols_offd_C; i++)
      {
         if (col_map_offd_C[i] == col_map_offd_B[cnt])
         {
            map_B_to_C[cnt++] = i;
            if (cnt == num_cols_offd_B) { break; }
         }
      }
      for (i = 0; i < nalu_hypre_CSRMatrixI(C_tmp_offd)[nalu_hypre_CSRMatrixNumRows(C_tmp_offd)]; i++)
      {
         j_indx = C_tmp_offd_j[i];
         C_tmp_offd_j[i] = map_B_to_C[j_indx];
      }
   }

   /*-----------------------------------------------------------------------
    *  Need to compute:
    *    C_diag = C_tmp_diag + C_ext_diag
    *    C_offd = C_tmp_offd + C_ext_offd
    *
    *  First generate structure
    *-----------------------------------------------------------------------*/

   if (C_ext_size || num_cols_offd_B)
   {
      C_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_diag_A + 1, memory_location_C);
      C_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_diag_A + 1, memory_location_C);

      C_diag_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_num_threads, NALU_HYPRE_MEMORY_HOST);
      C_offd_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_num_threads, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel
#endif
      {
         NALU_HYPRE_Int *B_marker = NULL;
         NALU_HYPRE_Int *B_marker_offd = NULL;
         NALU_HYPRE_Int ik, jk, j1, j2, jcol;
         NALU_HYPRE_Int ns, ne, ii, nnz_d, nnz_o;
         NALU_HYPRE_Int rest, size;
         NALU_HYPRE_Int num_threads = nalu_hypre_NumActiveThreads();

         size = num_cols_diag_A / num_threads;
         rest = num_cols_diag_A - size * num_threads;
         ii = nalu_hypre_GetThreadNum();
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

         B_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_diag_B, NALU_HYPRE_MEMORY_HOST);
         B_marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd_C, NALU_HYPRE_MEMORY_HOST);

         for (ik = 0; ik < num_cols_diag_B; ik++)
         {
            B_marker[ik] = -1;
         }

         for (ik = 0; ik < num_cols_offd_C; ik++)
         {
            B_marker_offd[ik] = -1;
         }

         nnz_d = 0;
         nnz_o = 0;
         for (ik = ns; ik < ne; ik++)
         {
            for (jk = C_tmp_diag_i[ik]; jk < C_tmp_diag_i[ik + 1]; jk++)
            {
               jcol = C_tmp_diag_j[jk];
               B_marker[jcol] = ik;
               nnz_d++;
            }

            for (jk = C_tmp_offd_i[ik]; jk < C_tmp_offd_i[ik + 1]; jk++)
            {
               jcol = C_tmp_offd_j[jk];
               B_marker_offd[jcol] = ik;
               nnz_o++;
            }

            for (jk = 0; jk < num_sends_A; jk++)
            {
               for (j1 = send_map_starts_A[jk]; j1 < send_map_starts_A[jk + 1]; j1++)
               {
                  if (send_map_elmts_A[j1] == ik)
                  {
                     for (j2 = C_ext_diag_i[j1]; j2 < C_ext_diag_i[j1 + 1]; j2++)
                     {
                        jcol = C_ext_diag_j[j2];
                        if (B_marker[jcol] < ik)
                        {
                           B_marker[jcol] = ik;
                           nnz_d++;
                        }
                     }
                     for (j2 = C_ext_offd_i[j1]; j2 < C_ext_offd_i[j1 + 1]; j2++)
                     {
                        jcol = C_ext_offd_j[j2];
                        if (B_marker_offd[jcol] < ik)
                        {
                           B_marker_offd[jcol] = ik;
                           nnz_o++;
                        }
                     }
                     break;
                  }
               }
            }
            C_diag_array[ii] = nnz_d;
            C_offd_array[ii] = nnz_o;
         }
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp barrier
#endif

         if (ii == 0)
         {
            nnz_d = 0;
            nnz_o = 0;
            for (ik = 0; ik < num_threads - 1; ik++)
            {
               C_diag_array[ik + 1] += C_diag_array[ik];
               C_offd_array[ik + 1] += C_offd_array[ik];
            }
            nnz_d = C_diag_array[num_threads - 1];
            nnz_o = C_offd_array[num_threads - 1];
            C_diag_i[num_cols_diag_A] = nnz_d;
            C_offd_i[num_cols_diag_A] = nnz_o;

            C_diag = nalu_hypre_CSRMatrixCreate(num_cols_diag_A, num_cols_diag_A, nnz_d);
            C_offd = nalu_hypre_CSRMatrixCreate(num_cols_diag_A, num_cols_offd_C, nnz_o);
            nalu_hypre_CSRMatrixI(C_diag) = C_diag_i;
            nalu_hypre_CSRMatrixInitialize_v2(C_diag, 0, memory_location_C);
            C_diag_j = nalu_hypre_CSRMatrixJ(C_diag);
            C_diag_data = nalu_hypre_CSRMatrixData(C_diag);
            nalu_hypre_CSRMatrixI(C_offd) = C_offd_i;
            nalu_hypre_CSRMatrixInitialize_v2(C_offd, 0, memory_location_C);
            C_offd_j = nalu_hypre_CSRMatrixJ(C_offd);
            C_offd_data = nalu_hypre_CSRMatrixData(C_offd);
         }
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp barrier
#endif

         /*-----------------------------------------------------------------------
          *  Need to compute C_diag = C_tmp_diag + C_ext_diag
          *  and  C_offd = C_tmp_offd + C_ext_offd   !!!!
          *  Now fill in values
          *-----------------------------------------------------------------------*/

         for (ik = 0; ik < num_cols_diag_B; ik++)
         {
            B_marker[ik] = -1;
         }

         for (ik = 0; ik < num_cols_offd_C; ik++)
         {
            B_marker_offd[ik] = -1;
         }

         /*-----------------------------------------------------------------------
          *  Populate matrices
          *-----------------------------------------------------------------------*/

         nnz_d = 0;
         nnz_o = 0;
         if (ii)
         {
            nnz_d = C_diag_array[ii - 1];
            nnz_o = C_offd_array[ii - 1];
         }
         for (ik = ns; ik < ne; ik++)
         {
            C_diag_i[ik] = nnz_d;
            C_offd_i[ik] = nnz_o;
            for (jk = C_tmp_diag_i[ik]; jk < C_tmp_diag_i[ik + 1]; jk++)
            {
               jcol = C_tmp_diag_j[jk];
               C_diag_j[nnz_d] = jcol;
               C_diag_data[nnz_d] = C_tmp_diag_data[jk];
               B_marker[jcol] = nnz_d;
               nnz_d++;
            }

            for (jk = C_tmp_offd_i[ik]; jk < C_tmp_offd_i[ik + 1]; jk++)
            {
               jcol = C_tmp_offd_j[jk];
               C_offd_j[nnz_o] = jcol;
               C_offd_data[nnz_o] = C_tmp_offd_data[jk];
               B_marker_offd[jcol] = nnz_o;
               nnz_o++;
            }

            for (jk = 0; jk < num_sends_A; jk++)
            {
               for (j1 = send_map_starts_A[jk]; j1 < send_map_starts_A[jk + 1]; j1++)
               {
                  if (send_map_elmts_A[j1] == ik)
                  {
                     for (j2 = C_ext_diag_i[j1]; j2 < C_ext_diag_i[j1 + 1]; j2++)
                     {
                        jcol = C_ext_diag_j[j2];
                        if (B_marker[jcol] < C_diag_i[ik])
                        {
                           C_diag_j[nnz_d] = jcol;
                           C_diag_data[nnz_d] = C_ext_diag_data[j2];
                           B_marker[jcol] = nnz_d;
                           nnz_d++;
                        }
                        else
                        {
                           C_diag_data[B_marker[jcol]] += C_ext_diag_data[j2];
                        }
                     }
                     for (j2 = C_ext_offd_i[j1]; j2 < C_ext_offd_i[j1 + 1]; j2++)
                     {
                        jcol = C_ext_offd_j[j2];
                        if (B_marker_offd[jcol] < C_offd_i[ik])
                        {
                           C_offd_j[nnz_o] = jcol;
                           C_offd_data[nnz_o] = C_ext_offd_data[j2];
                           B_marker_offd[jcol] = nnz_o;
                           nnz_o++;
                        }
                        else
                        {
                           C_offd_data[B_marker_offd[jcol]] += C_ext_offd_data[j2];
                        }
                     }
                     break;
                  }
               }
            }
         }
         nalu_hypre_TFree(B_marker, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(B_marker_offd, NALU_HYPRE_MEMORY_HOST);

      } /*end parallel region */

      nalu_hypre_TFree(C_diag_array, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(C_offd_array, NALU_HYPRE_MEMORY_HOST);
   }

   /*C = nalu_hypre_ParCSRMatrixCreate(comm, ncols_A, ncols_B, col_starts_A,
     col_starts_B, num_cols_offd_C, nnz_diag, nnz_offd);

     nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(C));
     nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(C)); */
   /* row_starts[0] is start of local rows.  row_starts[1] is start of next
      processor's rows */
   first_row_index = col_starts_A[0];
   local_num_rows = (NALU_HYPRE_Int)(col_starts_A[1] - first_row_index );
   first_col_diag = col_starts_B[0];
   local_num_cols = (NALU_HYPRE_Int)(col_starts_B[1] - first_col_diag);

   C = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParCSRMatrixComm(C) = comm;
   nalu_hypre_ParCSRMatrixGlobalNumRows(C) = ncols_A;
   nalu_hypre_ParCSRMatrixGlobalNumCols(C) = ncols_B;
   nalu_hypre_ParCSRMatrixFirstRowIndex(C) = first_row_index;
   nalu_hypre_ParCSRMatrixFirstColDiag(C) = first_col_diag;
   nalu_hypre_ParCSRMatrixLastRowIndex(C) = first_row_index + (NALU_HYPRE_BigInt)local_num_rows - 1;
   nalu_hypre_ParCSRMatrixLastColDiag(C) = first_col_diag + (NALU_HYPRE_BigInt)local_num_cols - 1;
   nalu_hypre_ParCSRMatrixColMapOffd(C) = NULL;
   nalu_hypre_ParCSRMatrixAssumedPartition(C) = NULL;
   nalu_hypre_ParCSRMatrixCommPkg(C) = NULL;
   nalu_hypre_ParCSRMatrixCommPkgT(C) = NULL;

   /* C row/col starts*/
   nalu_hypre_ParCSRMatrixRowStarts(C)[0] = col_starts_A[0];
   nalu_hypre_ParCSRMatrixRowStarts(C)[1] = col_starts_A[1];
   nalu_hypre_ParCSRMatrixColStarts(C)[0] = col_starts_B[0];
   nalu_hypre_ParCSRMatrixColStarts(C)[1] = col_starts_B[1];

   /* set defaults */
   nalu_hypre_ParCSRMatrixOwnsData(C) = 1;
   nalu_hypre_ParCSRMatrixRowindices(C) = NULL;
   nalu_hypre_ParCSRMatrixRowvalues(C) = NULL;
   nalu_hypre_ParCSRMatrixGetrowactive(C) = 0;

   if (C_diag)
   {
      nalu_hypre_CSRMatrixSetRownnz(C_diag);
      nalu_hypre_ParCSRMatrixDiag(C) = C_diag;
   }
   else
   {
      nalu_hypre_ParCSRMatrixDiag(C) = C_tmp_diag;
   }

   if (C_offd)
   {
      nalu_hypre_CSRMatrixSetRownnz(C_offd);
      nalu_hypre_ParCSRMatrixOffd(C) = C_offd;
   }
   else
   {
      nalu_hypre_ParCSRMatrixOffd(C) = C_tmp_offd;
   }

   nalu_hypre_assert(nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixDiag(C)) == memory_location_C);
   nalu_hypre_assert(nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixOffd(C)) == memory_location_C);

   if (num_cols_offd_C)
   {
      NALU_HYPRE_Int jj_count_offd, nnz_offd;
      NALU_HYPRE_BigInt *new_col_map_offd_C = NULL;

      P_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_C, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_cols_offd_C; i++)
      {
         P_marker[i] = -1;
      }

      jj_count_offd = 0;
      nnz_offd = C_offd_i[num_cols_diag_A];
      for (i = 0; i < nnz_offd; i++)
      {
         i1 = C_offd_j[i];
         if (P_marker[i1])
         {
            P_marker[i1] = 0;
            jj_count_offd++;
         }
      }

      if (jj_count_offd < num_cols_offd_C)
      {
         new_col_map_offd_C = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, jj_count_offd, NALU_HYPRE_MEMORY_HOST);
         jj_count_offd = 0;
         for (i = 0; i < num_cols_offd_C; i++)
         {
            if (!P_marker[i])
            {
               P_marker[i] = jj_count_offd;
               new_col_map_offd_C[jj_count_offd++] = col_map_offd_C[i];
            }
         }

         for (i = 0; i < nnz_offd; i++)
         {
            i1 = C_offd_j[i];
            C_offd_j[i] = P_marker[i1];
         }

         num_cols_offd_C = jj_count_offd;
         nalu_hypre_TFree(col_map_offd_C, NALU_HYPRE_MEMORY_HOST);
         col_map_offd_C = new_col_map_offd_C;
         nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(C)) = num_cols_offd_C;
      }
      nalu_hypre_TFree(P_marker, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;

   /*-----------------------------------------------------------------------
    *  Free various arrays
    *-----------------------------------------------------------------------*/
   if (C_ext_size || num_cols_offd_B)
   {
      nalu_hypre_TFree(C_ext_diag_i, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(C_ext_offd_i, NALU_HYPRE_MEMORY_HOST);
   }

   if (C_ext_diag_size)
   {
      nalu_hypre_TFree(C_ext_diag_j, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(C_ext_diag_data, NALU_HYPRE_MEMORY_HOST);
   }

   if (C_ext_offd_size)
   {
      nalu_hypre_TFree(C_ext_offd_j, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(C_ext_offd_data, NALU_HYPRE_MEMORY_HOST);
   }

   if (num_cols_offd_B)
   {
      nalu_hypre_TFree(map_B_to_C, NALU_HYPRE_MEMORY_HOST);
   }

   if (C_diag)
   {
      nalu_hypre_CSRMatrixDestroy(C_tmp_diag);
   }

   if (C_offd)
   {
      nalu_hypre_CSRMatrixDestroy(C_tmp_offd);
   }

#if defined(NALU_HYPRE_USING_GPU)
   if ( nalu_hypre_GetExecPolicy2(memory_location_A, memory_location_B) == NALU_HYPRE_EXEC_DEVICE )
   {
      nalu_hypre_CSRMatrixMoveDiagFirstDevice(nalu_hypre_ParCSRMatrixDiag(C));
      nalu_hypre_SyncComputeStream(nalu_hypre_handle());
   }
#endif

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return C;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParvecBdiagInvScal
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParvecBdiagInvScal( nalu_hypre_ParVector     *b,
                          NALU_HYPRE_Int            blockSize,
                          nalu_hypre_ParVector    **bs,
                          nalu_hypre_ParCSRMatrix  *A)
{
   MPI_Comm         comm     = nalu_hypre_ParCSRMatrixComm(b);
   NALU_HYPRE_Int        num_procs, my_id;
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   NALU_HYPRE_Int i, j, s, block_start, block_end;
   NALU_HYPRE_BigInt nrow_global = nalu_hypre_ParVectorGlobalSize(b);
   NALU_HYPRE_BigInt first_row   = nalu_hypre_ParVectorFirstIndex(b);
   NALU_HYPRE_BigInt last_row    = nalu_hypre_ParVectorLastIndex(b);
   NALU_HYPRE_BigInt end_row     = last_row + 1; /* one past-the-last */
   NALU_HYPRE_BigInt first_row_block = first_row / (NALU_HYPRE_BigInt)(blockSize) * (NALU_HYPRE_BigInt)blockSize;
   NALU_HYPRE_BigInt end_row_block   = nalu_hypre_min( (last_row / (NALU_HYPRE_BigInt)blockSize + 1) *
                                             (NALU_HYPRE_BigInt)blockSize, nrow_global );

   nalu_hypre_assert(blockSize == A->bdiag_size);
   NALU_HYPRE_Complex *bdiaginv = A->bdiaginv;
   nalu_hypre_ParCSRCommPkg *comm_pkg = A->bdiaginv_comm_pkg;

   NALU_HYPRE_Complex *dense = bdiaginv;

   //for (i=first_row_block; i < end_row; i+=blockSize) ;
   //printf("===[%d %d), [ %d %d ) %d === \n", first_row, end_row, first_row_block, end_row_block, i);

   /* local vector of b */
   nalu_hypre_Vector    *b_local      = nalu_hypre_ParVectorLocalVector(b);
   NALU_HYPRE_Complex   *b_local_data = nalu_hypre_VectorData(b_local);
   /* number of sends (#procs) */
   NALU_HYPRE_Int num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   /* number of rows to send */
   NALU_HYPRE_Int num_rows_send = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   /* number of recvs (#procs) */
   NALU_HYPRE_Int num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   /* number of rows to recv */
   NALU_HYPRE_Int num_rows_recv = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   nalu_hypre_ParVector *bnew = nalu_hypre_ParVectorCreate( nalu_hypre_ParVectorComm(b),
                                                  nalu_hypre_ParVectorGlobalSize(b), nalu_hypre_ParVectorPartitioning(b) );
   nalu_hypre_ParVectorInitialize(bnew);
   nalu_hypre_Vector    *bnew_local      = nalu_hypre_ParVectorLocalVector(bnew);
   NALU_HYPRE_Complex   *bnew_local_data = nalu_hypre_VectorData(bnew_local);

   /* send and recv b */
   NALU_HYPRE_Complex *send_b = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_rows_send, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Complex *recv_b = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_rows_recv, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_rows_send; i++)
   {
      j = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i);
      send_b[i] = b_local_data[j];
   }
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg, send_b, recv_b);
   /* ... */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   for (block_start = first_row_block; block_start < end_row_block; block_start += blockSize)
   {
      NALU_HYPRE_BigInt big_i;
      block_end = nalu_hypre_min(block_start + (NALU_HYPRE_BigInt)blockSize, nrow_global);
      s = (NALU_HYPRE_Int)(block_end - block_start);
      for (big_i = block_start; big_i < block_end; big_i++)
      {
         if (big_i < first_row || big_i >= end_row)
         {
            continue;
         }

         NALU_HYPRE_Int local_i = (NALU_HYPRE_Int)(big_i - first_row);
         NALU_HYPRE_Int block_i = (NALU_HYPRE_Int)(big_i - block_start);

         bnew_local_data[local_i] = 0.0;

         for (j = 0; j < s; j++)
         {
            NALU_HYPRE_BigInt global_rid = block_start + (NALU_HYPRE_BigInt)j;
            NALU_HYPRE_Complex val = dense[block_i + j * blockSize];
            if (val == 0.0)
            {
               continue;
            }
            if (global_rid >= first_row && global_rid < end_row)
            {
               NALU_HYPRE_Int rid = (NALU_HYPRE_Int)(global_rid - first_row);
               bnew_local_data[local_i] += val * b_local_data[rid];
            }
            else
            {
               NALU_HYPRE_Int rid;

               if (global_rid < first_row)
               {
                  rid = (NALU_HYPRE_Int)(global_rid - first_row_block);
               }
               else
               {
                  rid = (NALU_HYPRE_Int)(first_row - first_row_block + global_rid - end_row);
               }
               bnew_local_data[local_i] += val * recv_b[rid];
            }
         }
      }
      dense += blockSize * blockSize;
   }

   nalu_hypre_TFree(send_b, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(recv_b, NALU_HYPRE_MEMORY_HOST);
   *bs = bnew;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParcsrBdiagInvScal
 *
 * Compute As = B^{-1}*A, where B is the block diagonal of A.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParcsrBdiagInvScal( nalu_hypre_ParCSRMatrix   *A,
                          NALU_HYPRE_Int             blockSize,
                          nalu_hypre_ParCSRMatrix  **As)
{
   MPI_Comm         comm     = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int        num_procs, my_id;
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   NALU_HYPRE_Int i, j, k, s;
   NALU_HYPRE_BigInt block_start, block_end;
   /* diag part of A */
   nalu_hypre_CSRMatrix *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_a = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   nalu_hypre_CSRMatrix *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real      *A_offd_a = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);

   NALU_HYPRE_Int        num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt    *col_map_offd_A  = nalu_hypre_ParCSRMatrixColMapOffd(A);


   NALU_HYPRE_Int nrow_local = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_BigInt first_row  = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   NALU_HYPRE_BigInt last_row   = nalu_hypre_ParCSRMatrixLastRowIndex(A);
   NALU_HYPRE_BigInt end_row    = first_row + (NALU_HYPRE_BigInt)nrow_local; /* one past-the-last */

   NALU_HYPRE_Int ncol_local = nalu_hypre_CSRMatrixNumCols(A_diag);
   NALU_HYPRE_BigInt first_col  = nalu_hypre_ParCSRMatrixFirstColDiag(A);
   /* NALU_HYPRE_Int last_col   = nalu_hypre_ParCSRMatrixLastColDiag(A); */
   NALU_HYPRE_BigInt end_col    = first_col + (NALU_HYPRE_BigInt)ncol_local;

   NALU_HYPRE_BigInt nrow_global = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   NALU_HYPRE_BigInt ncol_global = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
   NALU_HYPRE_BigInt *row_starts = nalu_hypre_ParCSRMatrixRowStarts(A);
   void *request;

   /* if square globally and locally */
   NALU_HYPRE_Int square2 = (nrow_global == ncol_global) && (nrow_local == ncol_local) &&
                       (first_row == first_col);

   if (nrow_global != ncol_global)
   {
      nalu_hypre_printf("nalu_hypre_ParcsrBdiagInvScal: only support N_ROW == N_COL\n");
      return nalu_hypre_error_flag;
   }

   /* in block diagonals, row range of the blocks this proc span */
   NALU_HYPRE_BigInt first_row_block = first_row / (NALU_HYPRE_BigInt)blockSize * (NALU_HYPRE_BigInt)blockSize;
   NALU_HYPRE_BigInt end_row_block   = nalu_hypre_min( (last_row / (NALU_HYPRE_BigInt)blockSize + 1) *
                                             (NALU_HYPRE_BigInt)blockSize, nrow_global );
   NALU_HYPRE_Int num_blocks = (NALU_HYPRE_Int)(last_row / (NALU_HYPRE_BigInt)blockSize + 1 - first_row /
                                      (NALU_HYPRE_BigInt)blockSize);

   //for (i=first_row_block; i < end_row; i+=blockSize) ;
   //printf("===[%d %d), [ %d %d ) %d === \n", first_row, end_row, first_row_block, end_row_block, i);
   //return 0;

   /* number of external rows */
   NALU_HYPRE_Int num_ext_rows = (NALU_HYPRE_Int)(end_row_block - first_row_block - (end_row - first_row));
   NALU_HYPRE_BigInt *ext_indices;
   NALU_HYPRE_Int A_ext_nnz;

   nalu_hypre_CSRMatrix *A_ext   = NULL;
   NALU_HYPRE_Complex   *A_ext_a = NULL;
   NALU_HYPRE_Int       *A_ext_i = NULL;
   NALU_HYPRE_BigInt    *A_ext_j = NULL;

   NALU_HYPRE_Real *dense_all = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, num_blocks * blockSize * blockSize,
                                         NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Real *dense = dense_all;
   NALU_HYPRE_Int *IPIV  = nalu_hypre_TAlloc(NALU_HYPRE_Int, blockSize, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Complex *dgetri_work = NULL;
   NALU_HYPRE_Int      dgetri_lwork = -1, lapack_info;

   NALU_HYPRE_Int  num_cols_A_offd_new;
   NALU_HYPRE_BigInt *col_map_offd_A_new;
   NALU_HYPRE_BigInt big_i;
   NALU_HYPRE_Int *offd2new = NULL;
   NALU_HYPRE_Int *marker_diag, *marker_newoffd;

   NALU_HYPRE_Int nnz_diag = A_diag_i[nrow_local];
   NALU_HYPRE_Int nnz_offd = A_offd_i[nrow_local];
   NALU_HYPRE_Int nnz_diag_new = 0, nnz_offd_new = 0;
   NALU_HYPRE_Int *A_diag_i_new, *A_diag_j_new, *A_offd_i_new, *A_offd_j_new;
   NALU_HYPRE_Complex *A_diag_a_new, *A_offd_a_new;
   /* heuristic */
   NALU_HYPRE_Int nnz_diag_alloc = 2 * nnz_diag;
   NALU_HYPRE_Int nnz_offd_alloc = 2 * nnz_offd;

   A_diag_i_new = nalu_hypre_CTAlloc(NALU_HYPRE_Int,     nrow_local + 1, NALU_HYPRE_MEMORY_HOST);
   A_diag_j_new = nalu_hypre_CTAlloc(NALU_HYPRE_Int,     nnz_diag_alloc, NALU_HYPRE_MEMORY_HOST);
   A_diag_a_new = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nnz_diag_alloc, NALU_HYPRE_MEMORY_HOST);
   A_offd_i_new = nalu_hypre_CTAlloc(NALU_HYPRE_Int,     nrow_local + 1, NALU_HYPRE_MEMORY_HOST);
   A_offd_j_new = nalu_hypre_CTAlloc(NALU_HYPRE_Int,     nnz_offd_alloc, NALU_HYPRE_MEMORY_HOST);
   A_offd_a_new = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nnz_offd_alloc, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParCSRMatrix *Anew;
   nalu_hypre_CSRMatrix    *Anew_diag;
   nalu_hypre_CSRMatrix    *Anew_offd;

   NALU_HYPRE_Real eps = 2.2e-16;

   /* Start with extracting the external rows */
   NALU_HYPRE_BigInt *ext_offd;
   ext_indices = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_ext_rows, NALU_HYPRE_MEMORY_HOST);
   j = 0;
   for (big_i = first_row_block; big_i < first_row; big_i++)
   {
      ext_indices[j++] = big_i;
   }
   for (big_i = end_row; big_i < end_row_block; big_i++)
   {
      ext_indices[j++] = big_i;
   }

   nalu_hypre_assert(j == num_ext_rows);

   /* create CommPkg for external rows */
   nalu_hypre_ParCSRFindExtendCommPkg(comm, nrow_global, first_row, nrow_local, row_starts,
                                 nalu_hypre_ParCSRMatrixAssumedPartition(A),
                                 num_ext_rows, ext_indices, &A->bdiaginv_comm_pkg);

   nalu_hypre_ParcsrGetExternalRowsInit(A, num_ext_rows, ext_indices, A->bdiaginv_comm_pkg, 1, &request);
   A_ext = nalu_hypre_ParcsrGetExternalRowsWait(request);

   nalu_hypre_TFree(ext_indices, NALU_HYPRE_MEMORY_HOST);

   A_ext_i = nalu_hypre_CSRMatrixI(A_ext);
   A_ext_j = nalu_hypre_CSRMatrixBigJ(A_ext);
   A_ext_a = nalu_hypre_CSRMatrixData(A_ext);
   A_ext_nnz = A_ext_i[num_ext_rows];
   ext_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, A_ext_nnz, NALU_HYPRE_MEMORY_HOST);

   /* fint the offd incides in A_ext */
   for (i = 0, j = 0; i < A_ext_nnz; i++)
   {
      /* global index */
      NALU_HYPRE_BigInt cid = A_ext_j[i];
      /* keep the offd indices */
      if (cid < first_col || cid >= end_col)
      {
         ext_offd[j++] = cid;
      }
   }
   /* remove duplicates after sorting (TODO better ways?) */
   nalu_hypre_BigQsort0(ext_offd, 0, j - 1);
   for (i = 0, k = 0; i < j; i++)
   {
      if (i == 0 || ext_offd[i] != ext_offd[i - 1])
      {
         ext_offd[k++] = ext_offd[i];
      }
   }
   /* uniion these `k' new indices into col_map_offd_A */
   col_map_offd_A_new = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_A_offd + k, NALU_HYPRE_MEMORY_HOST);
   if (k)
   {
      /* map offd to offd_new */
      offd2new = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_union2(num_cols_A_offd, col_map_offd_A, k, ext_offd,
                &num_cols_A_offd_new, col_map_offd_A_new, offd2new, NULL);
   nalu_hypre_TFree(ext_offd, NALU_HYPRE_MEMORY_HOST);
   /*
    *   adjust column indices in A_ext
    */
   for (i = 0; i < A_ext_nnz; i++)
   {
      NALU_HYPRE_BigInt cid = A_ext_j[i];
      if (cid < first_col || cid >= end_col)
      {
         j = nalu_hypre_BigBinarySearch(col_map_offd_A_new, cid, num_cols_A_offd_new);
         /* searching must succeed */
         nalu_hypre_assert(j >= 0 && j < num_cols_A_offd_new);
         /* trick: save ncol_local + j back */
         A_ext_j[i] = ncol_local + j;
      }
      else
      {
         /* save local index: [0, ncol_local-1] */
         A_ext_j[i] = cid - first_col;
      }
   }

   /* marker for diag */
   marker_diag = nalu_hypre_TAlloc(NALU_HYPRE_Int, ncol_local, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < ncol_local; i++)
   {
      marker_diag[i] = -1;
   }
   /* marker for newoffd */
   marker_newoffd = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_A_offd_new, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_cols_A_offd_new; i++)
   {
      marker_newoffd[i] = -1;
   }

   /* outer most loop for blocks */
   for (block_start = first_row_block; block_start < end_row_block;
        block_start += (NALU_HYPRE_BigInt)blockSize)
   {
      NALU_HYPRE_BigInt big_i;
      block_end = nalu_hypre_min(block_start + (NALU_HYPRE_BigInt)blockSize, nrow_global);
      s = (NALU_HYPRE_Int)(block_end - block_start);

      /* 1. fill the dense block diag matrix */
      for (big_i = block_start; big_i < block_end; big_i++)
      {
         /* row index in this block */
         NALU_HYPRE_Int block_i = (NALU_HYPRE_Int)(big_i - block_start);

         /* row index i: it can be local or external */
         if (big_i >= first_row && big_i < end_row)
         {
            /* is a local row */
            j = (NALU_HYPRE_Int)(big_i - first_row);
            for (k = A_diag_i[j]; k < A_diag_i[j + 1]; k++)
            {
               NALU_HYPRE_BigInt cid = (NALU_HYPRE_BigInt)A_diag_j[k] + first_col;
               if (cid >= block_start && cid < block_end)
               {
                  dense[block_i + (NALU_HYPRE_Int)(cid - block_start)*blockSize] = A_diag_a[k];
               }
            }
            if (num_cols_A_offd)
            {
               for (k = A_offd_i[j]; k < A_offd_i[j + 1]; k++)
               {
                  NALU_HYPRE_BigInt cid = col_map_offd_A[A_offd_j[k]];
                  if (cid >= block_start && cid < block_end)
                  {
                     dense[block_i + (NALU_HYPRE_Int)(cid - block_start)*blockSize] = A_offd_a[k];
                  }
               }
            }
         }
         else
         {
            /* is an external row */
            if (big_i < first_row)
            {
               j = (NALU_HYPRE_Int)(big_i - first_row_block);
            }
            else
            {
               j = (NALU_HYPRE_Int)(first_row - first_row_block + big_i - end_row);
            }
            for (k = A_ext_i[j]; k < A_ext_i[j + 1]; k++)
            {
               NALU_HYPRE_BigInt cid = A_ext_j[k];
               /* recover the global index */
               cid = cid < (NALU_HYPRE_BigInt)ncol_local ? cid + first_col : col_map_offd_A_new[cid - ncol_local];
               if (cid >= block_start && cid < block_end)
               {
                  dense[block_i + (NALU_HYPRE_Int)(cid - block_start)*blockSize] = A_ext_a[k];
               }
            }
         }
      }

      /* 2. invert the dense matrix */
      nalu_hypre_dgetrf(&s, &s, dense, &blockSize, IPIV, &lapack_info);

      nalu_hypre_assert(lapack_info == 0);

      if (lapack_info == 0)
      {
         NALU_HYPRE_Int query = -1;
         NALU_HYPRE_Real lwork_opt;
         /* query the optimal size of work */
         nalu_hypre_dgetri(&s, dense, &blockSize, IPIV, &lwork_opt, &query, &lapack_info);

         nalu_hypre_assert(lapack_info == 0);

         if (lwork_opt > dgetri_lwork)
         {
            dgetri_lwork = (NALU_HYPRE_Int)lwork_opt;
            dgetri_work = nalu_hypre_TReAlloc(dgetri_work, NALU_HYPRE_Complex, dgetri_lwork, NALU_HYPRE_MEMORY_HOST);
         }

         nalu_hypre_dgetri(&s, dense, &blockSize, IPIV, dgetri_work, &dgetri_lwork, &lapack_info);

         nalu_hypre_assert(lapack_info == 0);
      }

      /* filter out *zeros* */
      NALU_HYPRE_Real Fnorm = 0.0;
      for (i = 0; i < s; i++)
      {
         for (j = 0; j < s; j++)
         {
            NALU_HYPRE_Complex t = dense[j + i * blockSize];
            Fnorm += t * t;
         }
      }

      Fnorm = nalu_hypre_sqrt(Fnorm);

      for (i = 0; i < s; i++)
      {
         for (j = 0; j < s; j++)
         {
            if ( nalu_hypre_abs(dense[j + i * blockSize]) < eps * Fnorm )
            {
               dense[j + i * blockSize] = 0.0;
            }
         }
      }

      /* 3. premultiplication: one-pass dynamic allocation */
      for (big_i = block_start; big_i < block_end; big_i++)
      {
         /* starting points of this row in j */
         NALU_HYPRE_Int diag_i_start = nnz_diag_new;
         NALU_HYPRE_Int offd_i_start = nnz_offd_new;

         /* compute a new row with global index 'i' and local index 'local_i' */
         NALU_HYPRE_Int local_i = (NALU_HYPRE_Int)(big_i - first_row);
         /* row index in this block */
         NALU_HYPRE_Int block_i = (NALU_HYPRE_Int)(big_i - block_start);

         if (big_i < first_row || big_i >= end_row)
         {
            continue;
         }

         /* if square^2: reserve the first space in diag part to the diag entry */
         if (square2)
         {
            marker_diag[local_i] = nnz_diag_new;
            if (nnz_diag_new == nnz_diag_alloc)
            {
               nnz_diag_alloc = nnz_diag_alloc * 2 + 1;
               A_diag_j_new = nalu_hypre_TReAlloc(A_diag_j_new, NALU_HYPRE_Int,     nnz_diag_alloc, NALU_HYPRE_MEMORY_HOST);
               A_diag_a_new = nalu_hypre_TReAlloc(A_diag_a_new, NALU_HYPRE_Complex, nnz_diag_alloc, NALU_HYPRE_MEMORY_HOST);
            }
            A_diag_j_new[nnz_diag_new] = local_i;
            A_diag_a_new[nnz_diag_new] = 0.0;
            nnz_diag_new ++;
         }

         /* combine s rows */
         for (j = 0; j < s; j++)
         {
            /* row to combine: global row id */
            NALU_HYPRE_BigInt global_rid = block_start + (NALU_HYPRE_BigInt)j;
            /* the multipiler */
            NALU_HYPRE_Complex val = dense[block_i + j * blockSize];

            if (val == 0.0)
            {
               continue;
            }

            if (global_rid >= first_row && global_rid < end_row)
            {
               /* this row is local */
               NALU_HYPRE_Int rid = (NALU_HYPRE_Int)(global_rid - first_row);
               NALU_HYPRE_Int ii;

               for (ii = A_diag_i[rid]; ii < A_diag_i[rid + 1]; ii++)
               {
                  NALU_HYPRE_Int col = A_diag_j[ii];
                  NALU_HYPRE_Complex vv = A_diag_a[ii];

                  if (marker_diag[col] < diag_i_start)
                  {
                     /* this col has not been seen before, create new entry */
                     marker_diag[col] = nnz_diag_new;
                     if (nnz_diag_new == nnz_diag_alloc)
                     {
                        nnz_diag_alloc = nnz_diag_alloc * 2 + 1;
                        A_diag_j_new = nalu_hypre_TReAlloc(A_diag_j_new, NALU_HYPRE_Int,     nnz_diag_alloc, NALU_HYPRE_MEMORY_HOST);
                        A_diag_a_new = nalu_hypre_TReAlloc(A_diag_a_new, NALU_HYPRE_Complex, nnz_diag_alloc, NALU_HYPRE_MEMORY_HOST);
                     }
                     A_diag_j_new[nnz_diag_new] = col;
                     A_diag_a_new[nnz_diag_new] = val * vv;
                     nnz_diag_new ++;
                  }
                  else
                  {
                     /* existing entry, update */
                     NALU_HYPRE_Int p = marker_diag[col];

                     nalu_hypre_assert(A_diag_j_new[p] == col);

                     A_diag_a_new[p] += val * vv;
                  }
               }

               for (ii = A_offd_i[rid]; ii < A_offd_i[rid + 1]; ii++)
               {
                  NALU_HYPRE_Int col = A_offd_j[ii];
                  /* use the mapper to map to new offd */
                  NALU_HYPRE_Int col_new = offd2new ? offd2new[col] : col;
                  NALU_HYPRE_Complex vv = A_offd_a[ii];

                  if (marker_newoffd[col_new] < offd_i_start)
                  {
                     /* this col has not been seen before, create new entry */
                     marker_newoffd[col_new] = nnz_offd_new;
                     if (nnz_offd_new == nnz_offd_alloc)
                     {
                        nnz_offd_alloc = nnz_offd_alloc * 2 + 1;
                        A_offd_j_new = nalu_hypre_TReAlloc(A_offd_j_new, NALU_HYPRE_Int,     nnz_offd_alloc, NALU_HYPRE_MEMORY_HOST);
                        A_offd_a_new = nalu_hypre_TReAlloc(A_offd_a_new, NALU_HYPRE_Complex, nnz_offd_alloc, NALU_HYPRE_MEMORY_HOST);
                     }
                     A_offd_j_new[nnz_offd_new] = col_new;
                     A_offd_a_new[nnz_offd_new] = val * vv;
                     nnz_offd_new ++;
                  }
                  else
                  {
                     /* existing entry, update */
                     NALU_HYPRE_Int p = marker_newoffd[col_new];

                     nalu_hypre_assert(A_offd_j_new[p] == col_new);

                     A_offd_a_new[p] += val * vv;
                  }
               }
            }
            else
            {
               /* this is an external row: go to A_ext */
               NALU_HYPRE_Int rid, ii;

               if (global_rid < first_row)
               {
                  rid = (NALU_HYPRE_Int)(global_rid - first_row_block);
               }
               else
               {
                  rid = (NALU_HYPRE_Int)(first_row - first_row_block + global_rid - end_row);
               }

               for (ii = A_ext_i[rid]; ii < A_ext_i[rid + 1]; ii++)
               {
                  NALU_HYPRE_Int col = (NALU_HYPRE_Int)A_ext_j[ii];
                  NALU_HYPRE_Complex vv = A_ext_a[ii];

                  if (col < ncol_local)
                  {
                     /* in diag part */
                     if (marker_diag[col] < diag_i_start)
                     {
                        /* this col has not been seen before, create new entry */
                        marker_diag[col] = nnz_diag_new;
                        if (nnz_diag_new == nnz_diag_alloc)
                        {
                           nnz_diag_alloc = nnz_diag_alloc * 2 + 1;
                           A_diag_j_new = nalu_hypre_TReAlloc(A_diag_j_new, NALU_HYPRE_Int,     nnz_diag_alloc, NALU_HYPRE_MEMORY_HOST);
                           A_diag_a_new = nalu_hypre_TReAlloc(A_diag_a_new, NALU_HYPRE_Complex, nnz_diag_alloc, NALU_HYPRE_MEMORY_HOST);
                        }
                        A_diag_j_new[nnz_diag_new] = col;
                        A_diag_a_new[nnz_diag_new] = val * vv;
                        nnz_diag_new ++;
                     }
                     else
                     {
                        /* existing entry, update */
                        NALU_HYPRE_Int p = marker_diag[col];

                        nalu_hypre_assert(A_diag_j_new[p] == col);

                        A_diag_a_new[p] += val * vv;
                     }
                  }
                  else
                  {
                     /* in offd part */
                     col -= ncol_local;

                     if (marker_newoffd[col] < offd_i_start)
                     {
                        /* this col has not been seen before, create new entry */
                        marker_newoffd[col] = nnz_offd_new;
                        if (nnz_offd_new == nnz_offd_alloc)
                        {
                           nnz_offd_alloc = nnz_offd_alloc * 2 + 1;
                           A_offd_j_new = nalu_hypre_TReAlloc(A_offd_j_new, NALU_HYPRE_Int,     nnz_offd_alloc, NALU_HYPRE_MEMORY_HOST);
                           A_offd_a_new = nalu_hypre_TReAlloc(A_offd_a_new, NALU_HYPRE_Complex, nnz_offd_alloc, NALU_HYPRE_MEMORY_HOST);
                        }
                        A_offd_j_new[nnz_offd_new] = col;
                        A_offd_a_new[nnz_offd_new] = val * vv;
                        nnz_offd_new ++;
                     }
                     else
                     {
                        /* existing entry, update */
                        NALU_HYPRE_Int p = marker_newoffd[col];

                        nalu_hypre_assert(A_offd_j_new[p] == col);

                        A_offd_a_new[p] += val * vv;
                     }
                  }
               }
            }
         }

         /* done for row local_i */
         A_diag_i_new[local_i + 1] = nnz_diag_new;
         A_offd_i_new[local_i + 1] = nnz_offd_new;
      } /* for i, each row */

      dense += blockSize * blockSize;
   } /* for each block */

   /* done with all rows */
   /* resize properly */
   A_diag_j_new = nalu_hypre_TReAlloc(A_diag_j_new, NALU_HYPRE_Int,     nnz_diag_new, NALU_HYPRE_MEMORY_HOST);
   A_diag_a_new = nalu_hypre_TReAlloc(A_diag_a_new, NALU_HYPRE_Complex, nnz_diag_new, NALU_HYPRE_MEMORY_HOST);
   A_offd_j_new = nalu_hypre_TReAlloc(A_offd_j_new, NALU_HYPRE_Int,     nnz_offd_new, NALU_HYPRE_MEMORY_HOST);
   A_offd_a_new = nalu_hypre_TReAlloc(A_offd_a_new, NALU_HYPRE_Complex, nnz_offd_new, NALU_HYPRE_MEMORY_HOST);

   /* readjust col_map_offd_new */
   for (i = 0; i < num_cols_A_offd_new; i++)
   {
      marker_newoffd[i] = -1;
   }
   for (i = 0; i < nnz_offd_new; i++)
   {
      j = A_offd_j_new[i];
      if (marker_newoffd[j] == -1)
      {
         marker_newoffd[j] = 1;
      }
   }
   for (i = 0, j = 0; i < num_cols_A_offd_new; i++)
   {
      if (marker_newoffd[i] == 1)
      {
         col_map_offd_A_new[j] = col_map_offd_A_new[i];
         marker_newoffd[i] = j++;
      }
   }
   num_cols_A_offd_new = j;

   for (i = 0; i < nnz_offd_new; i++)
   {
      j = marker_newoffd[A_offd_j_new[i]];
      nalu_hypre_assert(j >= 0 && j < num_cols_A_offd_new);
      A_offd_j_new[i] = j;
   }

   /* Now, we should have everything of Parcsr matrix As */
   Anew = nalu_hypre_ParCSRMatrixCreate(comm,
                                   nrow_global,
                                   ncol_global,
                                   nalu_hypre_ParCSRMatrixRowStarts(A),
                                   nalu_hypre_ParCSRMatrixColStarts(A),
                                   num_cols_A_offd_new,
                                   nnz_diag_new,
                                   nnz_offd_new);

   Anew_diag = nalu_hypre_ParCSRMatrixDiag(Anew);
   nalu_hypre_CSRMatrixData(Anew_diag) = A_diag_a_new;
   nalu_hypre_CSRMatrixI(Anew_diag)    = A_diag_i_new;
   nalu_hypre_CSRMatrixJ(Anew_diag)    = A_diag_j_new;

   Anew_offd = nalu_hypre_ParCSRMatrixOffd(Anew);
   nalu_hypre_CSRMatrixData(Anew_offd) = A_offd_a_new;
   nalu_hypre_CSRMatrixI(Anew_offd)    = A_offd_i_new;
   nalu_hypre_CSRMatrixJ(Anew_offd)    = A_offd_j_new;

   nalu_hypre_ParCSRMatrixColMapOffd(Anew) = col_map_offd_A_new;

   nalu_hypre_ParCSRMatrixSetNumNonzeros(Anew);
   nalu_hypre_ParCSRMatrixDNumNonzeros(Anew) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(Anew);
   //printf("nnz_diag %d --> %d, nnz_offd %d --> %d\n", nnz_diag, nnz_diag_new, nnz_offd, nnz_offd_new);

   /* create CommPkg of Anew */
   nalu_hypre_MatvecCommPkgCreate(Anew);

   *As = Anew;

   /*
   if (bdiaginv)
   {
      *bdiaginv = dense_all;
   }
   else
   {
      nalu_hypre_TFree(dense_all, NALU_HYPRE_MEMORY_HOST);
   }
   */
   /* save diagonal blocks in A */
   A->bdiag_size = blockSize;
   A->bdiaginv = dense_all;

   /* free workspace */
   nalu_hypre_TFree(IPIV, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(dgetri_work, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(marker_diag, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(marker_newoffd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(offd2new, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_CSRMatrixDestroy(A_ext);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParcsrGetExternalRowsInit
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParcsrGetExternalRowsInit( nalu_hypre_ParCSRMatrix   *A,
                                 NALU_HYPRE_Int             indices_len,
                                 NALU_HYPRE_BigInt         *indices,
                                 nalu_hypre_ParCSRCommPkg  *comm_pkg,
                                 NALU_HYPRE_Int             want_data,
                                 void                **request_ptr)
{
   MPI_Comm                 comm           = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_BigInt             first_col      = nalu_hypre_ParCSRMatrixFirstColDiag(A);
   NALU_HYPRE_BigInt            *col_map_offd_A = nalu_hypre_ParCSRMatrixColMapOffd(A);

   /* diag part of A */
   nalu_hypre_CSRMatrix         *A_diag    = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real              *A_diag_a  = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int               *A_diag_i  = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int               *A_diag_j  = nalu_hypre_CSRMatrixJ(A_diag);

   /* off-diag part of A */
   nalu_hypre_CSRMatrix         *A_offd    = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real              *A_offd_a  = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int               *A_offd_i  = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int               *A_offd_j  = nalu_hypre_CSRMatrixJ(A_offd);

   nalu_hypre_CSRMatrix         *A_ext;
   NALU_HYPRE_Int                num_procs, my_id;
   void                   **vrequest;

   NALU_HYPRE_Int                i, j, k;
   NALU_HYPRE_Int                num_sends, num_rows_send, num_nnz_send, *send_i;
   NALU_HYPRE_Int                num_recvs, num_rows_recv, num_nnz_recv, *recv_i;
   NALU_HYPRE_Int               *send_jstarts, *recv_jstarts, *send_i_offset;
   NALU_HYPRE_BigInt            *send_j, *recv_j;
   NALU_HYPRE_Complex           *send_a = NULL, *recv_a = NULL;
   nalu_hypre_ParCSRCommPkg     *comm_pkg_j = NULL;
   nalu_hypre_ParCSRCommHandle  *comm_handle, *comm_handle_j, *comm_handle_a;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   /* number of sends (#procs) */
   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   /* number of rows to send */
   num_rows_send = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
   /* number of recvs (#procs) */
   num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   /* number of rows to recv */
   num_rows_recv = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs);

   /* must be true if indices contains proper offd indices */
   nalu_hypre_assert(indices_len == num_rows_recv);

   /* send_i/recv_i:
    * the arrays to send and recv: we first send and recv the row lengths */
   send_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows_send, NALU_HYPRE_MEMORY_HOST);
   recv_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows_recv + 1, NALU_HYPRE_MEMORY_HOST);
   /* fill the send array with row lengths */
   for (i = 0, num_nnz_send = 0; i < num_rows_send; i++)
   {
      /* j: row index to send */
      j = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i);
      send_i[i] = A_diag_i[j + 1] - A_diag_i[j] + A_offd_i[j + 1] - A_offd_i[j];
      num_nnz_send += send_i[i];
   }

   /* send this array out: note the shift in recv_i by one (async) */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg, send_i, recv_i + 1);

   /* prepare data to send out. overlap with the above commmunication */
   send_j = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_nnz_send, NALU_HYPRE_MEMORY_HOST);
   if (want_data)
   {
      send_a = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_nnz_send, NALU_HYPRE_MEMORY_HOST);
   }

   send_i_offset = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows_send + 1, NALU_HYPRE_MEMORY_HOST);
   send_i_offset[0] = 0;
   nalu_hypre_TMemcpy(send_i_offset + 1, send_i, NALU_HYPRE_Int, num_rows_send,
                 NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
   /* prefix sum. TODO: OMP parallelization */
   for (i = 1; i <= num_rows_send; i++)
   {
      send_i_offset[i] += send_i_offset[i - 1];
   }
   nalu_hypre_assert(send_i_offset[num_rows_send] == num_nnz_send);

   /* pointers to each proc in send_j */
   send_jstarts = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_sends + 1, NALU_HYPRE_MEMORY_HOST);
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i <= num_sends; i++)
   {
      send_jstarts[i] = send_i_offset[nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i)];
   }
   nalu_hypre_assert(send_jstarts[num_sends] == num_nnz_send);

   /* fill the CSR matrix: j and a */
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for NALU_HYPRE_SMP_SCHEDULE private(i,j,k)
#endif
   for (i = 0; i < num_rows_send; i++)
   {
      NALU_HYPRE_Int i1 = send_i_offset[i];
      j = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i);
      /* open row j and fill ja and a to send */
      for (k = A_diag_i[j]; k < A_diag_i[j + 1]; k++)
      {
         send_j[i1] = first_col + A_diag_j[k];
         if (want_data)
         {
            send_a[i1] = A_diag_a[k];
         }
         i1++;
      }
      if (num_procs > 1)
      {
         for (k = A_offd_i[j]; k < A_offd_i[j + 1]; k++)
         {
            send_j[i1] = col_map_offd_A[A_offd_j[k]];
            if (want_data)
            {
               send_a[i1] = A_offd_a[k];
            }
            i1++;
         }
      }
      nalu_hypre_assert(send_i_offset[i + 1] == i1);
   }

   /* finish the above communication: send_i/recv_i */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /* adjust recv_i to ptrs */
   for (i = 1; i <= num_rows_recv; i++)
   {
      recv_i[i] += recv_i[i - 1];
   }
   num_nnz_recv = recv_i[num_rows_recv];
   recv_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_nnz_recv, NALU_HYPRE_MEMORY_HOST);
   if (want_data)
   {
      recv_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, num_nnz_recv, NALU_HYPRE_MEMORY_HOST);
   }
   recv_jstarts = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_recvs + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 1; i <= num_recvs; i++)
   {
      j = nalu_hypre_ParCSRCommPkgRecvVecStart(comm_pkg, i);
      recv_jstarts[i] = recv_i[j];
   }

   /* Create communication package */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs,
                                    nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg),
                                    recv_jstarts,
                                    num_sends,
                                    nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg),
                                    send_jstarts,
                                    NULL,
                                    &comm_pkg_j);

   /* init communication */
   /* ja */
   comm_handle_j = nalu_hypre_ParCSRCommHandleCreate(21, comm_pkg_j, send_j, recv_j);
   if (want_data)
   {
      /* a */
      comm_handle_a = nalu_hypre_ParCSRCommHandleCreate(1, comm_pkg_j, send_a, recv_a);
   }
   else
   {
      comm_handle_a = NULL;
   }

   /* create A_ext */
   A_ext = nalu_hypre_CSRMatrixCreate(num_rows_recv, nalu_hypre_ParCSRMatrixGlobalNumCols(A), num_nnz_recv);
   nalu_hypre_CSRMatrixMemoryLocation(A_ext) = NALU_HYPRE_MEMORY_HOST;
   nalu_hypre_CSRMatrixI   (A_ext) = recv_i;
   nalu_hypre_CSRMatrixBigJ(A_ext) = recv_j;
   nalu_hypre_CSRMatrixData(A_ext) = recv_a;

   /* output */
   vrequest = nalu_hypre_TAlloc(void *, 4, NALU_HYPRE_MEMORY_HOST);
   vrequest[0] = (void *) comm_handle_j;
   vrequest[1] = (void *) comm_handle_a;
   vrequest[2] = (void *) A_ext;
   vrequest[3] = (void *) comm_pkg_j;

   *request_ptr = (void *) vrequest;

   /* free */
   nalu_hypre_TFree(send_i, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_i_offset, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParcsrGetExternalRowsWait
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix*
nalu_hypre_ParcsrGetExternalRowsWait(void *vrequest)
{
   void **request = (void **) vrequest;

   nalu_hypre_ParCSRCommHandle *comm_handle_j = (nalu_hypre_ParCSRCommHandle *) request[0];
   nalu_hypre_ParCSRCommHandle *comm_handle_a = (nalu_hypre_ParCSRCommHandle *) request[1];
   nalu_hypre_CSRMatrix        *A_ext         = (nalu_hypre_CSRMatrix *)        request[2];
   nalu_hypre_ParCSRCommPkg    *comm_pkg_j    = (nalu_hypre_ParCSRCommPkg *)    request[3];
   NALU_HYPRE_BigInt           *send_j        = (NALU_HYPRE_BigInt *) nalu_hypre_ParCSRCommHandleSendData(
                                              comm_handle_j);

   if (comm_handle_a)
   {
      NALU_HYPRE_Complex *send_a = (NALU_HYPRE_Complex *) nalu_hypre_ParCSRCommHandleSendData(comm_handle_a);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle_a);
      nalu_hypre_TFree(send_a, NALU_HYPRE_MEMORY_HOST);
   }

   nalu_hypre_ParCSRCommHandleDestroy(comm_handle_j);
   nalu_hypre_TFree(send_j, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(comm_pkg_j, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(request, NALU_HYPRE_MEMORY_HOST);

   return A_ext;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixAddHost
 *
 * Host (CPU) version of nalu_hypre_ParCSRMatrixAdd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixAddHost( NALU_HYPRE_Complex        alpha,
                           nalu_hypre_ParCSRMatrix  *A,
                           NALU_HYPRE_Complex        beta,
                           nalu_hypre_ParCSRMatrix  *B,
                           nalu_hypre_ParCSRMatrix **C_ptr )
{
   /* ParCSRMatrix data */
   MPI_Comm          comm       = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_BigInt      num_rows_A = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   NALU_HYPRE_BigInt      num_cols_A = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
   /* NALU_HYPRE_BigInt      num_rows_B = nalu_hypre_ParCSRMatrixGlobalNumRows(B); */
   /* NALU_HYPRE_BigInt      num_cols_B = nalu_hypre_ParCSRMatrixGlobalNumCols(B); */

   /* diag part of A */
   nalu_hypre_CSRMatrix    *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int     *rownnz_diag_A = nalu_hypre_CSRMatrixRownnz(A_diag);
   NALU_HYPRE_Int  num_rownnz_diag_A = nalu_hypre_CSRMatrixNumRownnz(A_diag);
   NALU_HYPRE_Int    num_rows_diag_A = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int    num_cols_diag_A = nalu_hypre_CSRMatrixNumCols(A_diag);

   /* off-diag part of A */
   nalu_hypre_CSRMatrix    *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int     *rownnz_offd_A = nalu_hypre_CSRMatrixRownnz(A_offd);
   NALU_HYPRE_Int  num_rownnz_offd_A = nalu_hypre_CSRMatrixNumRownnz(A_offd);
   NALU_HYPRE_Int    num_rows_offd_A = nalu_hypre_CSRMatrixNumRows(A_offd);
   NALU_HYPRE_Int    num_cols_offd_A = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt *col_map_offd_A = nalu_hypre_ParCSRMatrixColMapOffd(A);
   NALU_HYPRE_Int          *A2C_offd;

   /* diag part of B */
   nalu_hypre_CSRMatrix    *B_diag   = nalu_hypre_ParCSRMatrixDiag(B);
   NALU_HYPRE_Int     *rownnz_diag_B = nalu_hypre_CSRMatrixRownnz(B_diag);
   NALU_HYPRE_Int  num_rownnz_diag_B = nalu_hypre_CSRMatrixNumRownnz(B_diag);
   NALU_HYPRE_Int    num_rows_diag_B = nalu_hypre_CSRMatrixNumRows(B_diag);
   /* NALU_HYPRE_Int    num_cols_diag_B = nalu_hypre_CSRMatrixNumCols(B_diag); */

   /* off-diag part of B */
   nalu_hypre_CSRMatrix    *B_offd   = nalu_hypre_ParCSRMatrixOffd(B);
   NALU_HYPRE_Int     *rownnz_offd_B = nalu_hypre_CSRMatrixRownnz(B_offd);
   NALU_HYPRE_Int  num_rownnz_offd_B = nalu_hypre_CSRMatrixNumRownnz(B_offd);
   NALU_HYPRE_Int    num_rows_offd_B = nalu_hypre_CSRMatrixNumRows(B_offd);
   NALU_HYPRE_Int    num_cols_offd_B = nalu_hypre_CSRMatrixNumCols(B_offd);
   NALU_HYPRE_BigInt *col_map_offd_B = nalu_hypre_ParCSRMatrixColMapOffd(B);
   NALU_HYPRE_Int          *B2C_offd;

   /* C data */
   nalu_hypre_ParCSRMatrix   *C;
   nalu_hypre_CSRMatrix      *C_diag;
   nalu_hypre_CSRMatrix      *C_offd;
   NALU_HYPRE_BigInt         *col_map_offd_C;
   NALU_HYPRE_Int            *C_diag_i, *C_offd_i;
   NALU_HYPRE_Int            *rownnz_diag_C = NULL;
   NALU_HYPRE_Int            *rownnz_offd_C = NULL;
   NALU_HYPRE_Int             num_rownnz_diag_C;
   NALU_HYPRE_Int             num_rownnz_offd_C;
   NALU_HYPRE_Int             num_rows_diag_C = num_rows_diag_A;
   NALU_HYPRE_Int             num_cols_diag_C = num_cols_diag_A;
   NALU_HYPRE_Int             num_rows_offd_C = num_rows_offd_A;
   NALU_HYPRE_Int             num_cols_offd_C = num_cols_offd_A + num_cols_offd_B;
   NALU_HYPRE_Int            *twspace;

   NALU_HYPRE_MemoryLocation  memory_location_A = nalu_hypre_ParCSRMatrixMemoryLocation(A);
   NALU_HYPRE_MemoryLocation  memory_location_B = nalu_hypre_ParCSRMatrixMemoryLocation(B);

   /* RL: TODO cannot guarantee, maybe should never assert
   nalu_hypre_assert(memory_location_A == memory_location_B);
   */

   /* RL: in the case of A=H, B=D, or A=D, B=H, let C = D,
    * not sure if this is the right thing to do.
    * Also, need something like this in other places
    * TODO */
   NALU_HYPRE_MemoryLocation  memory_location_C = nalu_hypre_max(memory_location_A, memory_location_B);

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Allocate memory */
   twspace  = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_NumThreads(), NALU_HYPRE_MEMORY_HOST);
   C_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows_diag_A + 1, memory_location_C);
   C_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows_offd_A + 1, memory_location_C);
   col_map_offd_C = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_C, NALU_HYPRE_MEMORY_HOST);

   /* Compute num_cols_offd_C, A2C_offd, and B2C_offd*/
   A2C_offd = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_offd_A, NALU_HYPRE_MEMORY_HOST);
   B2C_offd = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_offd_B, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_union2(num_cols_offd_A, col_map_offd_A,
                num_cols_offd_B, col_map_offd_B,
                &num_cols_offd_C, col_map_offd_C,
                A2C_offd, B2C_offd);

   /* Set nonzero rows data of diag_C */
   num_rownnz_diag_C = num_rows_diag_A;
   if ((num_rownnz_diag_A < num_rows_diag_A) &&
       (num_rownnz_diag_B < num_rows_diag_B))
   {
      nalu_hypre_IntArray arr_diagA;
      nalu_hypre_IntArray arr_diagB;
      nalu_hypre_IntArray arr_diagC;

      nalu_hypre_IntArrayData(&arr_diagA) = rownnz_diag_A;
      nalu_hypre_IntArrayData(&arr_diagB) = rownnz_diag_B;
      nalu_hypre_IntArraySize(&arr_diagA) = num_rownnz_diag_A;
      nalu_hypre_IntArraySize(&arr_diagB) = num_rownnz_diag_B;
      nalu_hypre_IntArrayMemoryLocation(&arr_diagC) = memory_location_C;

      nalu_hypre_IntArrayMergeOrdered(&arr_diagA, &arr_diagB, &arr_diagC);

      num_rownnz_diag_C = nalu_hypre_IntArraySize(&arr_diagC);
      rownnz_diag_C     = nalu_hypre_IntArrayData(&arr_diagC);
   }

   /* Set nonzero rows data of offd_C */
   num_rownnz_offd_C = num_rows_offd_A;
   if ((num_rownnz_offd_A < num_rows_offd_A) &&
       (num_rownnz_offd_B < num_rows_offd_B))
   {
      nalu_hypre_IntArray arr_offdA;
      nalu_hypre_IntArray arr_offdB;
      nalu_hypre_IntArray arr_offdC;

      nalu_hypre_IntArrayData(&arr_offdA) = rownnz_offd_A;
      nalu_hypre_IntArrayData(&arr_offdB) = rownnz_offd_B;
      nalu_hypre_IntArraySize(&arr_offdA) = num_rownnz_offd_A;
      nalu_hypre_IntArraySize(&arr_offdB) = num_rownnz_offd_B;
      nalu_hypre_IntArrayMemoryLocation(&arr_offdC) = memory_location_C;

      nalu_hypre_IntArrayMergeOrdered(&arr_offdA, &arr_offdB, &arr_offdC);

      num_rownnz_offd_C = nalu_hypre_IntArraySize(&arr_offdC);
      rownnz_offd_C     = nalu_hypre_IntArrayData(&arr_offdC);
   }

   /* Set diag_C */
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      NALU_HYPRE_Int   ii, num_threads;
      NALU_HYPRE_Int   size, rest, ns, ne;
      NALU_HYPRE_Int  *marker_diag;
      NALU_HYPRE_Int  *marker_offd;

      ii = nalu_hypre_GetThreadNum();
      num_threads = nalu_hypre_NumActiveThreads();

      /*-----------------------------------------------------------------------
       *  Compute C_diag = alpha*A_diag + beta*B_diag
       *-----------------------------------------------------------------------*/

      size = num_rownnz_diag_C / num_threads;
      rest = num_rownnz_diag_C - size * num_threads;
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

      marker_diag = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_diag_A, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CSRMatrixAddFirstPass(ns, ne, twspace, marker_diag,
                                  NULL, NULL, A_diag, B_diag,
                                  num_rows_diag_C, num_rownnz_diag_C,
                                  num_cols_diag_C, rownnz_diag_C,
                                  memory_location_C, C_diag_i, &C_diag);
      nalu_hypre_CSRMatrixAddSecondPass(ns, ne, twspace, marker_diag,
                                   NULL, NULL, rownnz_diag_C,
                                   alpha, beta, A_diag, B_diag, C_diag);
      nalu_hypre_TFree(marker_diag, NALU_HYPRE_MEMORY_HOST);

      /*-----------------------------------------------------------------------
       *  Compute C_offd = alpha*A_offd + beta*B_offd
       *-----------------------------------------------------------------------*/

      size = num_rownnz_offd_C / num_threads;
      rest = num_rownnz_offd_C - size * num_threads;
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

      marker_offd = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_offd_C, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CSRMatrixAddFirstPass(ns, ne, twspace, marker_offd,
                                  A2C_offd, B2C_offd, A_offd, B_offd,
                                  num_rows_offd_C, num_rownnz_offd_C,
                                  num_cols_offd_C, rownnz_offd_C,
                                  memory_location_C, C_offd_i, &C_offd);
      nalu_hypre_CSRMatrixAddSecondPass(ns, ne, twspace, marker_offd,
                                   A2C_offd, B2C_offd, rownnz_offd_C,
                                   alpha, beta, A_offd, B_offd, C_offd);
      nalu_hypre_TFree(marker_offd, NALU_HYPRE_MEMORY_HOST);
   } /* end of omp parallel region */

   /* Free memory */
   nalu_hypre_TFree(twspace, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(A2C_offd, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(B2C_offd, NALU_HYPRE_MEMORY_HOST);

   /* Create ParCSRMatrix C */
   C = nalu_hypre_ParCSRMatrixCreate(comm,
                                num_rows_A,
                                num_cols_A,
                                nalu_hypre_ParCSRMatrixRowStarts(A),
                                nalu_hypre_ParCSRMatrixColStarts(A),
                                num_cols_offd_C,
                                nalu_hypre_CSRMatrixNumNonzeros(C_diag),
                                nalu_hypre_CSRMatrixNumNonzeros(C_offd));

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(C));
   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(C));
   nalu_hypre_ParCSRMatrixDiag(C) = C_diag;
   nalu_hypre_ParCSRMatrixOffd(C) = C_offd;
   nalu_hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;
   nalu_hypre_ParCSRMatrixSetNumNonzeros(C);
   nalu_hypre_ParCSRMatrixDNumNonzeros(C) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(C);

   /* create CommPkg of C */
   nalu_hypre_MatvecCommPkgCreate(C);

   *C_ptr = C;

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixAdd
 *
 * Interface for Host/Device functions for computing C = alpha*A + beta*B
 *
 * A and B are assumed to have the same row and column partitionings
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixAdd( NALU_HYPRE_Complex        alpha,
                       nalu_hypre_ParCSRMatrix  *A,
                       NALU_HYPRE_Complex        beta,
                       nalu_hypre_ParCSRMatrix  *B,
                       nalu_hypre_ParCSRMatrix **C_ptr )
{
   nalu_hypre_assert(nalu_hypre_ParCSRMatrixGlobalNumRows(A) == nalu_hypre_ParCSRMatrixGlobalNumRows(B));
   nalu_hypre_assert(nalu_hypre_ParCSRMatrixGlobalNumCols(A) == nalu_hypre_ParCSRMatrixGlobalNumCols(B));
   nalu_hypre_assert(nalu_hypre_ParCSRMatrixNumRows(A) == nalu_hypre_ParCSRMatrixNumRows(B));
   nalu_hypre_assert(nalu_hypre_ParCSRMatrixNumCols(A) == nalu_hypre_ParCSRMatrixNumCols(B));

#if defined(NALU_HYPRE_USING_GPU)
   if ( nalu_hypre_GetExecPolicy2( nalu_hypre_ParCSRMatrixMemoryLocation(A),
                              nalu_hypre_ParCSRMatrixMemoryLocation(B) ) == NALU_HYPRE_EXEC_DEVICE )
   {
      nalu_hypre_ParCSRMatrixAddDevice(alpha, A, beta, B, C_ptr);
   }
   else
#endif
   {
      nalu_hypre_ParCSRMatrixAddHost(alpha, A, beta, B, C_ptr);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixFnorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real
nalu_hypre_ParCSRMatrixFnorm( nalu_hypre_ParCSRMatrix *A )
{
   MPI_Comm   comm = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Real f_diag, f_offd, local_result, result;

   f_diag = nalu_hypre_CSRMatrixFnorm(nalu_hypre_ParCSRMatrixDiag(A));
   f_offd = nalu_hypre_CSRMatrixFnorm(nalu_hypre_ParCSRMatrixOffd(A));
   local_result = f_diag * f_diag + f_offd * f_offd;

   nalu_hypre_MPI_Allreduce(&local_result, &result, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);

   return nalu_hypre_sqrt(result);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixInfNorm
 *
 * Computes the infinity norm of A:
 *
 *       norm = max_{i} sum_{j} |A_{ij}|
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixInfNorm( nalu_hypre_ParCSRMatrix  *A,
                           NALU_HYPRE_Real          *norm )
{
   MPI_Comm            comm     = nalu_hypre_ParCSRMatrixComm(A);

   /* diag part of A */
   nalu_hypre_CSRMatrix    *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_a = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int          *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int    num_rows_diag_A = nalu_hypre_CSRMatrixNumRows(A_diag);

   /* off-diag part of A */
   nalu_hypre_CSRMatrix    *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_a = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int          *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);

   /* Local variables */
   NALU_HYPRE_Int           i, j;
   NALU_HYPRE_Real          maxsum = 0.0;
   NALU_HYPRE_Real          rowsum;

#ifdef _MSC_VER
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel private(i,j,rowsum)
#endif
   {
      NALU_HYPRE_Real maxsum_local;

      maxsum_local = 0.0;
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp for NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows_diag_A; i++)
      {
         rowsum = 0.0;
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
         {
            rowsum += nalu_hypre_cabs(A_diag_a[j]);
         }
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            rowsum += nalu_hypre_cabs(A_offd_a[j]);
         }

         maxsum_local = nalu_hypre_max(maxsum_local, rowsum);
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp critical
#endif
      {
         maxsum = nalu_hypre_max(maxsum, maxsum_local);
      }
   }
#else
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,j,rowsum) reduction(max:maxsum) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_rows_diag_A; i++)
   {
      rowsum = 0.0;
      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         rowsum += nalu_hypre_cabs(A_diag_a[j]);
      }
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         rowsum += nalu_hypre_cabs(A_offd_a[j]);
      }

      maxsum = nalu_hypre_max(maxsum, rowsum);
   }
#endif

   nalu_hypre_MPI_Allreduce(&maxsum, norm, 1, NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_MAX, comm);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ExchangeExternalRowsInit
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ExchangeExternalRowsInit( nalu_hypre_CSRMatrix      *B_ext,
                                nalu_hypre_ParCSRCommPkg  *comm_pkg_A,
                                void                **request_ptr)
{
   MPI_Comm   comm             = nalu_hypre_ParCSRCommPkgComm(comm_pkg_A);
   NALU_HYPRE_Int  num_recvs        = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg_A);
   NALU_HYPRE_Int *recv_procs       = nalu_hypre_ParCSRCommPkgRecvProcs(comm_pkg_A);
   NALU_HYPRE_Int *recv_vec_starts  = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_A);
   NALU_HYPRE_Int  num_sends        = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg_A);
   NALU_HYPRE_Int *send_procs       = nalu_hypre_ParCSRCommPkgSendProcs(comm_pkg_A);
   NALU_HYPRE_Int *send_map_starts  = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg_A);

   NALU_HYPRE_Int  num_elmts_send   = send_map_starts[num_sends];
   NALU_HYPRE_Int  num_elmts_recv   = recv_vec_starts[num_recvs];

   NALU_HYPRE_Int     *B_ext_i      = B_ext ? nalu_hypre_CSRMatrixI(B_ext) : NULL;
   NALU_HYPRE_BigInt  *B_ext_j      = B_ext ? nalu_hypre_CSRMatrixBigJ(B_ext) : NULL;
   NALU_HYPRE_Complex *B_ext_data   = B_ext ? nalu_hypre_CSRMatrixData(B_ext) : NULL;
   NALU_HYPRE_Int      B_ext_ncols  = B_ext ? nalu_hypre_CSRMatrixNumCols(B_ext) : 0;
   NALU_HYPRE_Int      B_ext_nrows  = B_ext ? nalu_hypre_CSRMatrixNumRows(B_ext) : 0;
   NALU_HYPRE_Int     *B_ext_rownnz = nalu_hypre_CTAlloc(NALU_HYPRE_Int, B_ext_nrows, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_assert(num_elmts_recv == B_ext_nrows);

   /* output matrix */
   nalu_hypre_CSRMatrix *B_int;
   NALU_HYPRE_Int        B_int_nrows = num_elmts_send;
   NALU_HYPRE_Int        B_int_ncols = B_ext_ncols;
   NALU_HYPRE_Int       *B_int_i     = nalu_hypre_TAlloc(NALU_HYPRE_Int, B_int_nrows + 1, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_BigInt    *B_int_j     = NULL;
   NALU_HYPRE_Complex   *B_int_data  = NULL;
   NALU_HYPRE_Int        B_int_nnz;

   nalu_hypre_ParCSRCommHandle *comm_handle, *comm_handle_j, *comm_handle_a;
   nalu_hypre_ParCSRCommPkg    *comm_pkg_j = NULL;

   NALU_HYPRE_Int *jdata_recv_vec_starts;
   NALU_HYPRE_Int *jdata_send_map_starts;

   NALU_HYPRE_Int i;
   NALU_HYPRE_Int num_procs;
   void **vrequest;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   jdata_send_map_starts = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_sends + 1, NALU_HYPRE_MEMORY_HOST);

   /*--------------------------------------------------------------------------
    * B_ext_rownnz contains the number of elements of row j
    * (to be determined through send_map_elmnts on the receiving end)
    *--------------------------------------------------------------------------*/
   for (i = 0; i < B_ext_nrows; i++)
   {
      B_ext_rownnz[i] = B_ext_i[i + 1] - B_ext_i[i];
   }

   /*--------------------------------------------------------------------------
    * initialize communication: send/recv the row nnz
    * (note the use of comm_pkg_A, mode 12, as in transpose matvec
    *--------------------------------------------------------------------------*/
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(12, comm_pkg_A, B_ext_rownnz, B_int_i + 1);

   jdata_recv_vec_starts = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_recvs + 1, NALU_HYPRE_MEMORY_HOST);
   jdata_recv_vec_starts[0] = 0;
   for (i = 1; i <= num_recvs; i++)
   {
      jdata_recv_vec_starts[i] = B_ext_i[recv_vec_starts[i]];
   }

   /* Create communication package -  note the order of send/recv is reversed */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_sends, send_procs, jdata_send_map_starts,
                                    num_recvs, recv_procs, jdata_recv_vec_starts,
                                    NULL,
                                    &comm_pkg_j);

   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   /*--------------------------------------------------------------------------
    * compute B_int: row nnz to row ptrs
    *--------------------------------------------------------------------------*/
   B_int_i[0] = 0;
   for (i = 1; i <= B_int_nrows; i++)
   {
      B_int_i[i] += B_int_i[i - 1];
   }

   B_int_nnz = B_int_i[B_int_nrows];

   B_int_j    = nalu_hypre_TAlloc(NALU_HYPRE_BigInt,  B_int_nnz, NALU_HYPRE_MEMORY_HOST);
   B_int_data = nalu_hypre_TAlloc(NALU_HYPRE_Complex, B_int_nnz, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i <= num_sends; i++)
   {
      jdata_send_map_starts[i] = B_int_i[send_map_starts[i]];
   }

   /* send/recv CSR rows */
   comm_handle_a = nalu_hypre_ParCSRCommHandleCreate( 1, comm_pkg_j, B_ext_data, B_int_data);
   comm_handle_j = nalu_hypre_ParCSRCommHandleCreate(21, comm_pkg_j, B_ext_j, B_int_j);

   /* create CSR */
   B_int = nalu_hypre_CSRMatrixCreate(B_int_nrows, B_int_ncols, B_int_nnz);
   nalu_hypre_CSRMatrixMemoryLocation(B_int) = NALU_HYPRE_MEMORY_HOST;
   nalu_hypre_CSRMatrixI(B_int)    = B_int_i;
   nalu_hypre_CSRMatrixBigJ(B_int) = B_int_j;
   nalu_hypre_CSRMatrixData(B_int) = B_int_data;

   /* output */
   vrequest = nalu_hypre_TAlloc(void *, 4, NALU_HYPRE_MEMORY_HOST);
   vrequest[0] = (void *) comm_handle_j;
   vrequest[1] = (void *) comm_handle_a;
   vrequest[2] = (void *) B_int;
   vrequest[3] = (void *) comm_pkg_j;

   *request_ptr = (void *) vrequest;

   nalu_hypre_TFree(B_ext_rownnz, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ExchangeExternalRowsWait
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix*
nalu_hypre_ExchangeExternalRowsWait(void *vrequest)
{
   void **request = (void **) vrequest;

   nalu_hypre_ParCSRCommHandle *comm_handle_j = (nalu_hypre_ParCSRCommHandle *) request[0];
   nalu_hypre_ParCSRCommHandle *comm_handle_a = (nalu_hypre_ParCSRCommHandle *) request[1];
   nalu_hypre_CSRMatrix        *B_int         = (nalu_hypre_CSRMatrix *)        request[2];
   nalu_hypre_ParCSRCommPkg    *comm_pkg_j    = (nalu_hypre_ParCSRCommPkg *)    request[3];

   /* communication done */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle_a);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle_j);

   nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg_j), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_j), NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(comm_pkg_j, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_TFree(request, NALU_HYPRE_MEMORY_HOST);

   return B_int;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixExtractSubmatrixFC
 *
 * extract submatrix A_{FF}, A_{FC}, A_{CF} or A_{CC}
 * char job[2] = "FF", "FC", "CF" or "CC"
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixExtractSubmatrixFC( nalu_hypre_ParCSRMatrix  *A,
                                      NALU_HYPRE_Int           *CF_marker,
                                      NALU_HYPRE_BigInt        *cpts_starts,
                                      const char          *job,
                                      nalu_hypre_ParCSRMatrix **B_ptr,
                                      NALU_HYPRE_Real           strength_thresh)
{
   MPI_Comm                 comm     = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_ParCSRCommPkg     *comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
   nalu_hypre_ParCSRCommHandle  *comm_handle;

   /* diag part of A */
   nalu_hypre_CSRMatrix    *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Complex      *A_diag_a = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int          *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int          *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   nalu_hypre_CSRMatrix    *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Complex      *A_offd_a = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int          *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int          *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);

   NALU_HYPRE_Int           num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   //NALU_HYPRE_Int          *col_map_offd_A  = nalu_hypre_ParCSRMatrixColMapOffd(A);

   nalu_hypre_ParCSRMatrix *B;
   nalu_hypre_CSRMatrix    *B_diag, *B_offd;
   NALU_HYPRE_Real         *B_maxel_row;
   NALU_HYPRE_Int          *B_diag_i, *B_diag_j, *B_offd_i, *B_offd_j;
   NALU_HYPRE_Complex      *B_diag_a, *B_offd_a;
   NALU_HYPRE_Int           num_cols_B_offd;
   NALU_HYPRE_BigInt       *col_map_offd_B;

   NALU_HYPRE_Int           i, j, k, k1, k2;
   NALU_HYPRE_BigInt        B_nrow_global, B_ncol_global;
   NALU_HYPRE_Int           A_nlocal, B_nrow_local, B_ncol_local,
                       B_nnz_diag, B_nnz_offd;
   NALU_HYPRE_BigInt        total_global_fpts, total_global_cpts, fpts_starts[2];
   NALU_HYPRE_Int           nf_local, nc_local;
   NALU_HYPRE_BigInt        big_nf_local;
   NALU_HYPRE_Int           row_set, col_set;
   NALU_HYPRE_BigInt       *B_row_starts, *B_col_starts, B_first_col;
   NALU_HYPRE_Int           my_id, num_procs;
   NALU_HYPRE_Int          *sub_idx_diag;
   NALU_HYPRE_BigInt       *sub_idx_offd;
   NALU_HYPRE_Int           num_sends;
   NALU_HYPRE_BigInt       *send_buf_data;

   /* MPI size and rank*/
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   row_set = job[0] == 'F' ? -1 : 1;
   col_set = job[1] == 'F' ? -1 : 1;

   A_nlocal = nalu_hypre_CSRMatrixNumRows(A_diag);

   /*-------------- global number of C points and local C points
    *               assuming cpts_starts is given */
   if (row_set == 1 || col_set == 1)
   {
      if (my_id == (num_procs - 1))
      {
         total_global_cpts = cpts_starts[1];
      }
      nalu_hypre_MPI_Bcast(&total_global_cpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
      nc_local = (NALU_HYPRE_Int)(cpts_starts[1] - cpts_starts[0]);
   }

   /*-------------- global number of F points, local F points, and F starts */
   if (row_set == -1 || col_set == -1)
   {
      nf_local = 0;
      for (i = 0; i < A_nlocal; i++)
      {
         if (CF_marker[i] < 0)
         {
            nf_local++;
         }
      }
      big_nf_local = (NALU_HYPRE_BigInt) nf_local;
      nalu_hypre_MPI_Scan(&big_nf_local, fpts_starts + 1, 1, NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);
      fpts_starts[0] = fpts_starts[1] - nf_local;
      if (my_id == num_procs - 1)
      {
         total_global_fpts = fpts_starts[1];
      }
      nalu_hypre_MPI_Bcast(&total_global_fpts, 1, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);
   }

   if (row_set == -1 && col_set == -1)
   {
      /* FF */
      B_nrow_local = nf_local;
      B_ncol_local = nf_local;
      B_nrow_global = total_global_fpts;
      B_ncol_global = total_global_fpts;

      B_row_starts = B_col_starts = fpts_starts;
   }
   else if (row_set == -1 && col_set == 1)
   {
      /* FC */
      B_nrow_local = nf_local;
      B_ncol_local = nc_local;
      B_nrow_global = total_global_fpts;
      B_ncol_global = total_global_cpts;

      B_row_starts = fpts_starts;
      B_col_starts = cpts_starts;
   }
   else if (row_set == 1 && col_set == -1)
   {
      /* CF */
      B_nrow_local = nc_local;
      B_ncol_local = nf_local;
      B_nrow_global = total_global_cpts;
      B_ncol_global = total_global_fpts;

      B_row_starts = cpts_starts;
      B_col_starts = fpts_starts;
   }
   else
   {
      /* CC */
      B_nrow_local = nc_local;
      B_ncol_local = nc_local;
      B_nrow_global = total_global_cpts;
      B_ncol_global = total_global_cpts;

      B_row_starts = B_col_starts = cpts_starts;
   }

   /* global index of my first col */
   B_first_col = B_col_starts[0];

   /* sub_idx_diag: [local] mapping from F+C to F/C, if not selected, be -1 */
   sub_idx_diag = nalu_hypre_TAlloc(NALU_HYPRE_Int, A_nlocal, NALU_HYPRE_MEMORY_HOST);
   for (i = 0, k = 0; i < A_nlocal; i++)
   {
      NALU_HYPRE_Int CF_i = CF_marker[i] > 0 ? 1 : -1;
      if (CF_i == col_set)
      {
         sub_idx_diag[i] = k++;
      }
      else
      {
         sub_idx_diag[i] = -1;
      }
   }

   nalu_hypre_assert(k == B_ncol_local);

   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_buf_data = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                NALU_HYPRE_MEMORY_HOST);
   k = 0;
   for (i = 0; i < num_sends; i++)
   {
      /* start pos of elements sent to send_proc[i] */
      NALU_HYPRE_Int si = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      NALU_HYPRE_Int ei = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
      /* loop through all elems to send_proc[i] */
      for (j = si; j < ei; j++)
      {
         /* j1: local idx */
         NALU_HYPRE_BigInt j1 = sub_idx_diag[nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         if (j1 != -1)
         {
            /* adjust j1 to B global idx */
            j1 += B_first_col;
         }
         send_buf_data[k++] = j1;
      }
   }

   nalu_hypre_assert(k == nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

   /* recv buffer */
   sub_idx_offd = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);
   /* create a handle to start communication. 11: for integer */
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf_data, sub_idx_offd);
   /* destroy the handle to finish communication */
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

   for (i = 0, num_cols_B_offd = 0; i < num_cols_A_offd; i++)
   {
      if (sub_idx_offd[i] != -1)
      {
         num_cols_B_offd ++;
      }
   }
   col_map_offd_B = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_B_offd, NALU_HYPRE_MEMORY_HOST);
   for (i = 0, k = 0; i < num_cols_A_offd; i++)
   {
      if (sub_idx_offd[i] != -1)
      {
         col_map_offd_B[k] = sub_idx_offd[i];
         sub_idx_offd[i] = k++;
      }
   }

   nalu_hypre_assert(k == num_cols_B_offd);

   /* count nnz and set ia */
   B_nnz_diag = B_nnz_offd = 0;
   B_maxel_row = nalu_hypre_TAlloc(NALU_HYPRE_Real, B_nrow_local, NALU_HYPRE_MEMORY_HOST);
   B_diag_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, B_nrow_local + 1, NALU_HYPRE_MEMORY_HOST);
   B_offd_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, B_nrow_local + 1, NALU_HYPRE_MEMORY_HOST);
   B_diag_i[0] = B_offd_i[0] = 0;

   for (i = 0, k = 0; i < A_nlocal; i++)
   {
      NALU_HYPRE_Int CF_i = CF_marker[i] > 0 ? 1 : -1;
      if (CF_i != row_set)
      {
         continue;
      }
      k++;

      // Get max abs-value element of this row
      NALU_HYPRE_Real temp_max = 0;
      if (strength_thresh > 0)
      {
         for (j = A_diag_i[i] + 1; j < A_diag_i[i + 1]; j++)
         {
            if (nalu_hypre_cabs(A_diag_a[j]) > temp_max)
            {
               temp_max = nalu_hypre_cabs(A_diag_a[j]);
            }
         }
         for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
         {
            if (nalu_hypre_cabs(A_offd_a[j]) > temp_max)
            {
               temp_max = nalu_hypre_cabs(A_offd_a[j]);
            }
         }
      }
      B_maxel_row[k - 1] = temp_max;

      // add one for diagonal element
      j = A_diag_i[i];
      if (sub_idx_diag[A_diag_j[j]] != -1)
      {
         B_nnz_diag++;
      }

      // Count nnzs larger than tolerance times max row element
      for (j = A_diag_i[i] + 1; j < A_diag_i[i + 1]; j++)
      {
         if ( (sub_idx_diag[A_diag_j[j]] != -1) &&
              (nalu_hypre_cabs(A_diag_a[j]) > (strength_thresh * temp_max)) )
         {
            B_nnz_diag++;
         }
      }
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         if ( (sub_idx_offd[A_offd_j[j]] != -1) &&
              (nalu_hypre_cabs(A_offd_a[j]) > (strength_thresh * temp_max)) )
         {
            B_nnz_offd++;
         }
      }
      B_diag_i[k] = B_nnz_diag;
      B_offd_i[k] = B_nnz_offd;
   }

   nalu_hypre_assert(k == B_nrow_local);

   B_diag_j = nalu_hypre_TAlloc(NALU_HYPRE_Int,     B_nnz_diag, NALU_HYPRE_MEMORY_HOST);
   B_diag_a = nalu_hypre_TAlloc(NALU_HYPRE_Complex, B_nnz_diag, NALU_HYPRE_MEMORY_HOST);
   B_offd_j = nalu_hypre_TAlloc(NALU_HYPRE_Int,     B_nnz_offd, NALU_HYPRE_MEMORY_HOST);
   B_offd_a = nalu_hypre_TAlloc(NALU_HYPRE_Complex, B_nnz_offd, NALU_HYPRE_MEMORY_HOST);

   for (i = 0, k = 0, k1 = 0, k2 = 0; i < A_nlocal; i++)
   {
      NALU_HYPRE_Int CF_i = CF_marker[i] > 0 ? 1 : -1;
      if (CF_i != row_set)
      {
         continue;
      }
      NALU_HYPRE_Real maxel = B_maxel_row[k];
      k++;

      for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
      {
         NALU_HYPRE_Int j1 = sub_idx_diag[A_diag_j[j]];
         if ( (j1 != -1) && ( (nalu_hypre_cabs(A_diag_a[j]) > (strength_thresh * maxel)) || j == A_diag_i[i] ) )
         {
            B_diag_j[k1] = j1;
            B_diag_a[k1] = A_diag_a[j];
            k1++;
         }
      }
      for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
      {
         NALU_HYPRE_Int j1 = sub_idx_offd[A_offd_j[j]];
         if ((j1 != -1) && (nalu_hypre_cabs(A_offd_a[j]) > (strength_thresh * maxel)))
         {
            nalu_hypre_assert(j1 >= 0 && j1 < num_cols_B_offd);
            B_offd_j[k2] = j1;
            B_offd_a[k2] = A_offd_a[j];
            k2++;
         }
      }
   }

   nalu_hypre_assert(k1 == B_nnz_diag && k2 == B_nnz_offd);

   /* ready to create B = A(rowset, colset) */
   B = nalu_hypre_ParCSRMatrixCreate(comm,
                                B_nrow_global,
                                B_ncol_global,
                                B_row_starts,
                                B_col_starts,
                                num_cols_B_offd,
                                B_nnz_diag,
                                B_nnz_offd);

   B_diag = nalu_hypre_ParCSRMatrixDiag(B);
   nalu_hypre_CSRMatrixMemoryLocation(B_diag) = NALU_HYPRE_MEMORY_HOST;
   nalu_hypre_CSRMatrixData(B_diag) = B_diag_a;
   nalu_hypre_CSRMatrixI(B_diag)    = B_diag_i;
   nalu_hypre_CSRMatrixJ(B_diag)    = B_diag_j;

   B_offd = nalu_hypre_ParCSRMatrixOffd(B);
   nalu_hypre_CSRMatrixMemoryLocation(B_offd) = NALU_HYPRE_MEMORY_HOST;
   nalu_hypre_CSRMatrixData(B_offd) = B_offd_a;
   nalu_hypre_CSRMatrixI(B_offd)    = B_offd_i;
   nalu_hypre_CSRMatrixJ(B_offd)    = B_offd_j;

   nalu_hypre_ParCSRMatrixColMapOffd(B) = col_map_offd_B;

   nalu_hypre_ParCSRMatrixSetNumNonzeros(B);
   nalu_hypre_ParCSRMatrixDNumNonzeros(B) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(B);

   nalu_hypre_MatvecCommPkgCreate(B);

   *B_ptr = B;

   nalu_hypre_TFree(B_maxel_row, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_buf_data, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(sub_idx_diag, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(sub_idx_offd, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixDropSmallEntriesHost
 *
 * drop the entries that are not on the diagonal and smaller than:
 *    type 0: tol (TODO)
 *    type 1: tol*(1-norm of row)
 *    type 2: tol*(2-norm of row)
 *    type -1: tol*(infinity norm of row)
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixDropSmallEntriesHost( nalu_hypre_ParCSRMatrix *A,
                                        NALU_HYPRE_Real          tol,
                                        NALU_HYPRE_Int           type)
{
   NALU_HYPRE_Int i, j, k, nnz_diag, nnz_offd, A_diag_i_i, A_offd_i_i;

   MPI_Comm         comm     = nalu_hypre_ParCSRMatrixComm(A);
   /* diag part of A */
   nalu_hypre_CSRMatrix *A_diag   = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Real      *A_diag_a = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int       *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int       *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   /* off-diag part of A */
   nalu_hypre_CSRMatrix *A_offd   = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Real      *A_offd_a = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int       *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int       *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);

   NALU_HYPRE_Int  num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(A_offd);
   NALU_HYPRE_BigInt *col_map_offd_A  = nalu_hypre_ParCSRMatrixColMapOffd(A);
   NALU_HYPRE_Int *marker_offd = NULL;

   NALU_HYPRE_BigInt first_row  = nalu_hypre_ParCSRMatrixFirstRowIndex(A);
   NALU_HYPRE_Int nrow_local = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int my_id, num_procs;
   /* MPI size and rank*/
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   marker_offd = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_A_offd, NALU_HYPRE_MEMORY_HOST);

   nnz_diag = nnz_offd = A_diag_i_i = A_offd_i_i = 0;
   for (i = 0; i < nrow_local; i++)
   {
      /* compute row norm */
      NALU_HYPRE_Real row_nrm = 0.0;
      for (j = A_diag_i_i; j < A_diag_i[i + 1]; j++)
      {
         NALU_HYPRE_Complex v = A_diag_a[j];
         if (type == 1)
         {
            row_nrm += nalu_hypre_cabs(v);
         }
         else if (type == 2)
         {
            row_nrm += v * v;
         }
         else
         {
            row_nrm = nalu_hypre_max(row_nrm, nalu_hypre_cabs(v));
         }
      }
      if (num_procs > 1)
      {
         for (j = A_offd_i_i; j < A_offd_i[i + 1]; j++)
         {
            NALU_HYPRE_Complex v = A_offd_a[j];
            if (type == 1)
            {
               row_nrm += nalu_hypre_cabs(v);
            }
            else if (type == 2)
            {
               row_nrm += v * v;
            }
            else
            {
               row_nrm = nalu_hypre_max(row_nrm, nalu_hypre_cabs(v));
            }
         }
      }

      if (type == 2)
      {
         row_nrm = nalu_hypre_sqrt(row_nrm);
      }

      /* drop small entries based on tol and row norm */
      for (j = A_diag_i_i; j < A_diag_i[i + 1]; j++)
      {
         NALU_HYPRE_Int     col = A_diag_j[j];
         NALU_HYPRE_Complex val = A_diag_a[j];
         if (i == col || nalu_hypre_cabs(val) >= tol * row_nrm)
         {
            A_diag_j[nnz_diag] = col;
            A_diag_a[nnz_diag] = val;
            nnz_diag ++;
         }
      }
      if (num_procs > 1)
      {
         for (j = A_offd_i_i; j < A_offd_i[i + 1]; j++)
         {
            NALU_HYPRE_Int     col = A_offd_j[j];
            NALU_HYPRE_Complex val = A_offd_a[j];
            /* in normal cases: diagonal entry should not
             * appear in A_offd (but this can still be possible) */
            if (i + first_row == col_map_offd_A[col] || nalu_hypre_cabs(val) >= tol * row_nrm)
            {
               if (0 == marker_offd[col])
               {
                  marker_offd[col] = 1;
               }
               A_offd_j[nnz_offd] = col;
               A_offd_a[nnz_offd] = val;
               nnz_offd ++;
            }
         }
      }
      A_diag_i_i = A_diag_i[i + 1];
      A_offd_i_i = A_offd_i[i + 1];
      A_diag_i[i + 1] = nnz_diag;
      A_offd_i[i + 1] = nnz_offd;
   }

   nalu_hypre_CSRMatrixNumNonzeros(A_diag) = nnz_diag;
   nalu_hypre_CSRMatrixNumNonzeros(A_offd) = nnz_offd;
   nalu_hypre_ParCSRMatrixSetNumNonzeros(A);
   nalu_hypre_ParCSRMatrixDNumNonzeros(A) = (NALU_HYPRE_Real) nalu_hypre_ParCSRMatrixNumNonzeros(A);

   for (i = 0, k = 0; i < num_cols_A_offd; i++)
   {
      if (marker_offd[i])
      {
         col_map_offd_A[k] = col_map_offd_A[i];
         marker_offd[i] = k++;
      }
   }
   /* num_cols_A_offd = k; */
   nalu_hypre_CSRMatrixNumCols(A_offd) = k;
   for (i = 0; i < nnz_offd; i++)
   {
      A_offd_j[i] = marker_offd[A_offd_j[i]];
   }

   if ( nalu_hypre_ParCSRMatrixCommPkg(A) )
   {
      nalu_hypre_MatvecCommPkgDestroy( nalu_hypre_ParCSRMatrixCommPkg(A) );
   }
   nalu_hypre_MatvecCommPkgCreate(A);

   nalu_hypre_TFree(marker_offd, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixDropSmallEntries
 *
 * drop the entries that are not on the diagonal and smaller than
 *    type 0: tol
 *    type 1: tol*(1-norm of row)
 *    type 2: tol*(2-norm of row)
 *    type -1: tol*(infinity norm of row)
 *    NOTE: some type options above unavailable on either host or device
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixDropSmallEntries( nalu_hypre_ParCSRMatrix *A,
                                    NALU_HYPRE_Real          tol,
                                    NALU_HYPRE_Int           type)
{
   if (tol <= 0.0)
   {
      return nalu_hypre_error_flag;
   }

   nalu_hypre_GpuProfilingPushRange("ParCSRMatrixDropSmallEntries");

   NALU_HYPRE_Int ierr = 0;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      ierr = nalu_hypre_ParCSRMatrixDropSmallEntriesDevice(A, tol, type);
   }
   else
#endif
   {
      ierr = nalu_hypre_ParCSRMatrixDropSmallEntriesHost(A, tol, type);
   }

   nalu_hypre_GpuProfilingPopRange();

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixScale
 *
 * Computes A = scalar * A
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixScale(nalu_hypre_ParCSRMatrix *A,
                        NALU_HYPRE_Complex       scalar)
{
   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);

   nalu_hypre_CSRMatrixScale(A_diag, scalar);
   nalu_hypre_CSRMatrixScale(A_offd, scalar);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixDiagScaleHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixDiagScaleHost( nalu_hypre_ParCSRMatrix *par_A,
                                 nalu_hypre_ParVector    *par_ld,
                                 nalu_hypre_ParVector    *par_rd )
{
   /* Input variables */
   nalu_hypre_ParCSRCommPkg   *comm_pkg  = nalu_hypre_ParCSRMatrixCommPkg(par_A);
   NALU_HYPRE_Int              num_sends;
   NALU_HYPRE_Int             *send_map_elmts;
   NALU_HYPRE_Int             *send_map_starts;

   nalu_hypre_CSRMatrix       *A_diag        = nalu_hypre_ParCSRMatrixDiag(par_A);
   nalu_hypre_CSRMatrix       *A_offd        = nalu_hypre_ParCSRMatrixOffd(par_A);
   NALU_HYPRE_Int              num_cols_offd = nalu_hypre_CSRMatrixNumCols(A_offd);

   nalu_hypre_Vector          *ld            = (par_ld) ? nalu_hypre_ParVectorLocalVector(par_ld) : NULL;
   nalu_hypre_Vector          *rd            = nalu_hypre_ParVectorLocalVector(par_rd);
   NALU_HYPRE_Complex         *rd_data       = nalu_hypre_VectorData(rd);

   /* Local variables */
   NALU_HYPRE_Int              i;
   nalu_hypre_Vector          *rdbuf;
   NALU_HYPRE_Complex         *recv_rdbuf_data;
   NALU_HYPRE_Complex         *send_rdbuf_data;

   /*---------------------------------------------------------------------
    * Communication phase
    *--------------------------------------------------------------------*/

   /* Create buffer vectors */
   rdbuf = nalu_hypre_SeqVectorCreate(num_cols_offd);

   /* If there exists no CommPkg for A, create it. */
   if (!comm_pkg)
   {
      nalu_hypre_MatvecCommPkgCreate(par_A);
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(par_A);
   }
   num_sends       = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_map_elmts  = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
   send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);

#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   nalu_hypre_ParCSRPersistentCommHandle *comm_handle =
      nalu_hypre_ParCSRCommPkgGetPersistentCommHandle(1, comm_pkg);

   nalu_hypre_VectorData(rdbuf) = (NALU_HYPRE_Complex *)
                             nalu_hypre_ParCSRCommHandleRecvDataBuffer(comm_handle);
   nalu_hypre_SeqVectorSetDataOwner(rdbuf, 0);

#else
   nalu_hypre_ParCSRCommHandle *comm_handle;
#endif

   /* Initialize rdbuf */
   nalu_hypre_SeqVectorInitialize_v2(rdbuf, NALU_HYPRE_MEMORY_HOST);
   recv_rdbuf_data = nalu_hypre_VectorData(rdbuf);

   /* Allocate send buffer for rdbuf */
#if defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   send_rdbuf_data = (NALU_HYPRE_Complex *) nalu_hypre_ParCSRCommHandleSendDataBuffer(comm_handle);
#else
   send_rdbuf_data = nalu_hypre_TAlloc(NALU_HYPRE_Complex, send_map_starts[num_sends], NALU_HYPRE_MEMORY_HOST);
#endif

   /* Pack send data */
#if defined(NALU_HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = send_map_starts[0]; i < send_map_starts[num_sends]; i++)
   {
      send_rdbuf_data[i] = rd_data[send_map_elmts[i]];
   }

   /* Non-blocking communication starts */
#ifdef NALU_HYPRE_USING_PERSISTENT_COMM
   nalu_hypre_ParCSRPersistentCommHandleStart(comm_handle, NALU_HYPRE_MEMORY_HOST, send_rdbuf_data);

#else
   comm_handle = nalu_hypre_ParCSRCommHandleCreate_v2(1, comm_pkg,
                                                 NALU_HYPRE_MEMORY_HOST, send_rdbuf_data,
                                                 NALU_HYPRE_MEMORY_HOST, recv_rdbuf_data);
#endif

   /*---------------------------------------------------------------------
    * Computation phase
    *--------------------------------------------------------------------*/

   /* A_diag = diag(ld) * A_diag * diag(rd) */
   nalu_hypre_CSRMatrixDiagScale(A_diag, ld, rd);

   /* Non-blocking communication ends */
#ifdef NALU_HYPRE_USING_PERSISTENT_COMM
   nalu_hypre_ParCSRPersistentCommHandleWait(comm_handle, NALU_HYPRE_MEMORY_HOST, recv_rdbuf_data);
#else
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
#endif

   /* A_offd = diag(ld) * A_offd * diag(rd) */
   nalu_hypre_CSRMatrixDiagScale(A_offd, ld, rdbuf);

   /* Free memory */
   nalu_hypre_SeqVectorDestroy(rdbuf);
#if !defined(NALU_HYPRE_USING_PERSISTENT_COMM)
   nalu_hypre_TFree(send_rdbuf_data, NALU_HYPRE_MEMORY_HOST);
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixDiagScale
 *
 * Computes A = diag(ld) * A * diag(rd), where the diagonal matrices
 * "diag(ld)" and "diag(rd)" are stored as distributed vectors.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixDiagScale( nalu_hypre_ParCSRMatrix *par_A,
                             nalu_hypre_ParVector    *par_ld,
                             nalu_hypre_ParVector    *par_rd )
{
   /* Input variables */
   nalu_hypre_CSRMatrix    *A_diag = nalu_hypre_ParCSRMatrixDiag(par_A);
   nalu_hypre_CSRMatrix    *A_offd = nalu_hypre_ParCSRMatrixOffd(par_A);
   nalu_hypre_Vector       *ld;

   /* Sanity check */
   if (!par_rd && !par_ld)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Scaling matrices are not set!\n");
      return nalu_hypre_error_flag;
   }

   /* Perform row scaling only (no communication) */
   if (!par_rd && par_ld)
   {
      ld = nalu_hypre_ParVectorLocalVector(par_ld);

      nalu_hypre_CSRMatrixDiagScale(A_diag, ld, NULL);
      nalu_hypre_CSRMatrixDiagScale(A_offd, ld, NULL);

      return nalu_hypre_error_flag;
   }

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(par_A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_ParCSRMatrixDiagScaleDevice(par_A, par_ld, par_rd);
   }
   else
#endif
   {
      nalu_hypre_ParCSRMatrixDiagScaleHost(par_A, par_ld, par_rd);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixReorder:
 *
 * Reorders the column and data arrays of a the diagonal component of a square
 * ParCSR matrix, such that the first entry in each row is the diagonal one.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixReorder(nalu_hypre_ParCSRMatrix *A)
{
   NALU_HYPRE_BigInt      nrows_A = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   NALU_HYPRE_BigInt      ncols_A = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
   nalu_hypre_CSRMatrix  *A_diag  = nalu_hypre_ParCSRMatrixDiag(A);

   if (nrows_A != ncols_A)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, " Error! Matrix should be square!\n");
      return nalu_hypre_error_flag;
   }

   nalu_hypre_CSRMatrixReorder(A_diag);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixCompressOffdMap
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixCompressOffdMap(nalu_hypre_ParCSRMatrix *A)
{
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   nalu_hypre_GpuProfilingPushRange("nalu_hypre_ParCSRMatrixCompressOffdMap");
#endif

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_ParCSRMatrixCompressOffdMapDevice(A);
   }
#endif
   // RL: I guess it's not needed for the host code [?]

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRDiagScaleVectorHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRDiagScaleVectorHost( nalu_hypre_ParCSRMatrix *par_A,
                                 nalu_hypre_ParVector    *par_y,
                                 nalu_hypre_ParVector    *par_x )
{
   /* Local Matrix and Vectors */
   nalu_hypre_CSRMatrix    *A_diag        = nalu_hypre_ParCSRMatrixDiag(par_A);
   nalu_hypre_Vector       *x             = nalu_hypre_ParVectorLocalVector(par_x);
   nalu_hypre_Vector       *y             = nalu_hypre_ParVectorLocalVector(par_y);

   /* Local vector x info */
   NALU_HYPRE_Complex      *x_data        = nalu_hypre_VectorData(x);
   NALU_HYPRE_Int           x_num_vectors = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int           x_vecstride   = nalu_hypre_VectorVectorStride(x);

   /* Local vector y info */
   NALU_HYPRE_Complex      *y_data        = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int           y_vecstride   = nalu_hypre_VectorVectorStride(y);

   /* Local matrix A info */
   NALU_HYPRE_Complex      *A_data        = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int          *A_i           = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int           num_rows      = nalu_hypre_CSRMatrixNumRows(A_diag);

   /* Local variables */
   NALU_HYPRE_Int           i, k;
   NALU_HYPRE_Complex       coef;

   switch (x_num_vectors)
   {
      case 1:
#if defined(NALU_HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
         {
            x_data[i] = y_data[i] / A_data[A_i[i]];
         }
         break;

      case 2:
#if defined(NALU_HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i, coef) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
         {
            coef = 1.0 / A_data[A_i[i]];

            x_data[i] = y_data[i] * coef;
            x_data[i + x_vecstride] = y_data[i + y_vecstride] * coef;
         }
         break;

      case 3:
#if defined(NALU_HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i, coef) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
         {
            coef = 1.0 / A_data[A_i[i]];

            x_data[i] = y_data[i] * coef;
            x_data[i +     x_vecstride] = y_data[i +     y_vecstride] * coef;
            x_data[i + 2 * x_vecstride] = y_data[i + 2 * y_vecstride] * coef;
         }
         break;

      case 4:
#if defined(NALU_HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i, coef) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
         {
            coef = 1.0 / A_data[A_i[i]];

            x_data[i] = y_data[i] * coef;
            x_data[i +     x_vecstride] = y_data[i +     y_vecstride] * coef;
            x_data[i + 2 * x_vecstride] = y_data[i + 2 * y_vecstride] * coef;
            x_data[i + 3 * x_vecstride] = y_data[i + 3 * y_vecstride] * coef;
         }
         break;

      default:
#if defined(NALU_HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i, k, coef) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
         {
            coef = 1.0 / A_data[A_i[i]];

            for (k = 0; k < x_num_vectors; k++)
            {
               x_data[i + k * x_vecstride] = y_data[i + k * y_vecstride] * coef;
            }
         }
         break;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRDiagScaleVector
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRDiagScaleVector( nalu_hypre_ParCSRMatrix *par_A,
                             nalu_hypre_ParVector    *par_y,
                             nalu_hypre_ParVector    *par_x )
{
   nalu_hypre_GpuProfilingPushRange("nalu_hypre_ParCSRDiagScaleVector");

   /* Local Matrix and Vectors */
   nalu_hypre_CSRMatrix    *A_diag        = nalu_hypre_ParCSRMatrixDiag(par_A);
   nalu_hypre_Vector       *x             = nalu_hypre_ParVectorLocalVector(par_x);
   nalu_hypre_Vector       *y             = nalu_hypre_ParVectorLocalVector(par_y);

   /* Local vector x info */
   NALU_HYPRE_Int           x_size        = nalu_hypre_VectorSize(x);
   NALU_HYPRE_Int           x_num_vectors = nalu_hypre_VectorNumVectors(x);
   NALU_HYPRE_Int           x_vecstride   = nalu_hypre_VectorVectorStride(x);

   /* Local vector y info */
   NALU_HYPRE_Int           y_size        = nalu_hypre_VectorSize(y);
   NALU_HYPRE_Int           y_num_vectors = nalu_hypre_VectorNumVectors(y);
   NALU_HYPRE_Int           y_vecstride   = nalu_hypre_VectorVectorStride(y);

   /* Local matrix A info */
   NALU_HYPRE_Int           num_rows      = nalu_hypre_CSRMatrixNumRows(A_diag);

   /*---------------------------------------------
    * Sanity checks
    *---------------------------------------------*/

   if (x_num_vectors != y_num_vectors)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error! incompatible number of vectors!\n");
      return nalu_hypre_error_flag;
   }

   if (num_rows != x_size)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error! incompatible x size!\n");
      return nalu_hypre_error_flag;
   }

   if (x_size > 0 && x_vecstride <= 0)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error! non-positive x vector stride!\n");
      return nalu_hypre_error_flag;
   }

   if (y_size > 0 && y_vecstride <= 0)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error! non-positive y vector stride!\n");
      return nalu_hypre_error_flag;
   }

   if (num_rows != y_size)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error! incompatible y size!\n");
      return nalu_hypre_error_flag;
   }

   /*---------------------------------------------
    * Computation
    *---------------------------------------------*/

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(par_A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_ParCSRDiagScaleVectorDevice(par_A, par_y, par_x);
   }
   else
#endif
   {
      nalu_hypre_ParCSRDiagScaleVectorHost(par_A, par_y, par_x);
   }

   nalu_hypre_GpuProfilingPopRange();

   return nalu_hypre_error_flag;
}
