/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

nalu_hypre_ParCSRBooleanMatrix*
nalu_hypre_ParBooleanMatmul( nalu_hypre_ParCSRBooleanMatrix *A,
                        nalu_hypre_ParCSRBooleanMatrix *B )
{
   MPI_Comm       comm = nalu_hypre_ParCSRBooleanMatrix_Get_Comm(A);

   nalu_hypre_CSRBooleanMatrix *A_diag = nalu_hypre_ParCSRBooleanMatrix_Get_Diag(A);
   NALU_HYPRE_Int              *A_diag_i = nalu_hypre_CSRBooleanMatrix_Get_I(A_diag);
   NALU_HYPRE_Int              *A_diag_j = nalu_hypre_CSRBooleanMatrix_Get_J(A_diag);

   nalu_hypre_CSRBooleanMatrix *A_offd = nalu_hypre_ParCSRBooleanMatrix_Get_Offd(A);
   NALU_HYPRE_Int              *A_offd_i = nalu_hypre_CSRBooleanMatrix_Get_I(A_offd);
   NALU_HYPRE_Int              *A_offd_j = nalu_hypre_CSRBooleanMatrix_Get_J(A_offd);

   NALU_HYPRE_BigInt *row_starts_A = nalu_hypre_ParCSRBooleanMatrix_Get_RowStarts(A);
   NALU_HYPRE_Int   num_rows_diag_A = nalu_hypre_CSRBooleanMatrix_Get_NRows(A_diag);
   NALU_HYPRE_Int   num_cols_diag_A = nalu_hypre_CSRBooleanMatrix_Get_NCols(A_diag);
   NALU_HYPRE_Int   num_cols_offd_A = nalu_hypre_CSRBooleanMatrix_Get_NCols(A_offd);

   nalu_hypre_CSRBooleanMatrix *B_diag = nalu_hypre_ParCSRBooleanMatrix_Get_Diag(B);
   NALU_HYPRE_Int              *B_diag_i = nalu_hypre_CSRBooleanMatrix_Get_I(B_diag);
   NALU_HYPRE_Int              *B_diag_j = nalu_hypre_CSRBooleanMatrix_Get_J(B_diag);

   nalu_hypre_CSRBooleanMatrix *B_offd = nalu_hypre_ParCSRBooleanMatrix_Get_Offd(B);
   NALU_HYPRE_BigInt        *col_map_offd_B = nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(B);
   NALU_HYPRE_Int              *B_offd_i = nalu_hypre_CSRBooleanMatrix_Get_I(B_offd);
   NALU_HYPRE_Int              *B_offd_j = nalu_hypre_CSRBooleanMatrix_Get_J(B_offd);

   NALU_HYPRE_BigInt   first_col_diag_B = nalu_hypre_ParCSRBooleanMatrix_Get_FirstColDiag(B);
   NALU_HYPRE_BigInt   last_col_diag_B;
   NALU_HYPRE_BigInt *col_starts_B = nalu_hypre_ParCSRBooleanMatrix_Get_ColStarts(B);
   NALU_HYPRE_Int   num_rows_diag_B = nalu_hypre_CSRBooleanMatrix_Get_NRows(B_diag);
   NALU_HYPRE_Int   num_cols_diag_B = nalu_hypre_CSRBooleanMatrix_Get_NCols(B_diag);
   NALU_HYPRE_Int   num_cols_offd_B = nalu_hypre_CSRBooleanMatrix_Get_NCols(B_offd);

   nalu_hypre_ParCSRBooleanMatrix *C;
   NALU_HYPRE_BigInt            *col_map_offd_C;
   NALU_HYPRE_Int            *map_B_to_C = NULL;

   nalu_hypre_CSRBooleanMatrix *C_diag;
   NALU_HYPRE_Int             *C_diag_i;
   NALU_HYPRE_Int             *C_diag_j;

   nalu_hypre_CSRBooleanMatrix *C_offd;
   NALU_HYPRE_Int             *C_offd_i = NULL;
   NALU_HYPRE_Int             *C_offd_j = NULL;

   NALU_HYPRE_Int              C_diag_size;
   NALU_HYPRE_Int              C_offd_size;
   NALU_HYPRE_Int          num_cols_offd_C = 0;

   nalu_hypre_CSRBooleanMatrix *Bs_ext;
   NALU_HYPRE_Int             *Bs_ext_i;
   NALU_HYPRE_BigInt          *Bs_ext_j;

   NALU_HYPRE_Int             *B_ext_diag_i;
   NALU_HYPRE_Int             *B_ext_diag_j;
   NALU_HYPRE_Int        B_ext_diag_size;

   NALU_HYPRE_Int             *B_ext_offd_i;
   NALU_HYPRE_Int             *B_ext_offd_j;
   NALU_HYPRE_BigInt          *B_tmp_offd_j;
   NALU_HYPRE_Int        B_ext_offd_size;

   NALU_HYPRE_Int       *B_marker;
   NALU_HYPRE_BigInt       *temp;

   NALU_HYPRE_Int              i, j;
   NALU_HYPRE_Int              i1, i2, i3;
   NALU_HYPRE_Int              jj2, jj3;

   NALU_HYPRE_Int              jj_count_diag, jj_count_offd;
   NALU_HYPRE_Int              jj_row_begin_diag, jj_row_begin_offd;
   NALU_HYPRE_Int              start_indexing = 0; /* start indexing for C_data at 0 */
   NALU_HYPRE_BigInt        n_rows_A, n_cols_A;
   NALU_HYPRE_BigInt        n_rows_B, n_cols_B;
   NALU_HYPRE_Int              allsquare = 0;
   NALU_HYPRE_Int              cnt, cnt_offd, cnt_diag;
   NALU_HYPRE_Int              num_procs;
   NALU_HYPRE_Int              value;

   n_rows_A = nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNRows(A);
   n_cols_A = nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNCols(A);
   n_rows_B = nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNRows(B);
   n_cols_B = nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNCols(B);

   if (n_cols_A != n_rows_B || num_cols_diag_A != num_rows_diag_B)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, " Error! Incompatible matrix dimensions!\n");
      return NULL;
   }
   if ( num_rows_diag_A == num_cols_diag_B ) { allsquare = 1; }

   /*-----------------------------------------------------------------------
    *  Extract B_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product
    *-----------------------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   if (num_procs > 1)
   {
      /*---------------------------------------------------------------------
      * If there exists no CommPkg for A, a CommPkg is generated using
      * equally load balanced partitionings
      *--------------------------------------------------------------------*/
      if (!nalu_hypre_ParCSRBooleanMatrix_Get_CommPkg(A))
      {
         nalu_hypre_BooleanMatvecCommPkgCreate(A);
      }

      Bs_ext = nalu_hypre_ParCSRBooleanMatrixExtractBExt(B, A);
      Bs_ext_i    = nalu_hypre_CSRBooleanMatrix_Get_I(Bs_ext);
      Bs_ext_j    = nalu_hypre_CSRBooleanMatrix_Get_BigJ(Bs_ext);
   }

   B_ext_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd_A + 1, NALU_HYPRE_MEMORY_HOST);
   B_ext_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd_A + 1, NALU_HYPRE_MEMORY_HOST);
   B_ext_diag_size = 0;
   B_ext_offd_size = 0;
   last_col_diag_B = first_col_diag_B + num_cols_diag_B - 1;

   for (i = 0; i < num_cols_offd_A; i++)
   {
      for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
         {
            B_ext_offd_size++;
         }
         else
         {
            B_ext_diag_size++;
         }
      B_ext_diag_i[i + 1] = B_ext_diag_size;
      B_ext_offd_i[i + 1] = B_ext_offd_size;
   }

   if (B_ext_diag_size)
   {
      B_ext_diag_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  B_ext_diag_size, NALU_HYPRE_MEMORY_HOST);
   }

   if (B_ext_offd_size)
   {
      B_ext_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  B_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
      B_tmp_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  B_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
   }

   cnt_offd = 0;
   cnt_diag = 0;
   for (i = 0; i < num_cols_offd_A; i++)
   {
      for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
         {
            B_tmp_offd_j[cnt_offd++] = Bs_ext_j[j];
            //temp[cnt_offd++] = Bs_ext_j[j];
         }
         else
         {
            B_ext_diag_j[cnt_diag++] = (NALU_HYPRE_Int)(Bs_ext_j[j] - first_col_diag_B);
         }
   }

   if (num_procs > 1)
   {
      nalu_hypre_CSRBooleanMatrixDestroy(Bs_ext);
      Bs_ext = NULL;
   }

   cnt = 0;
   if (B_ext_offd_size || num_cols_offd_B)
   {
      temp = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  B_ext_offd_size + num_cols_offd_B, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < B_ext_offd_size; i++)
      {
         temp[i] = B_tmp_offd_j[i];
      }
      cnt = B_ext_offd_size;
      for (i = 0; i < num_cols_offd_B; i++)
      {
         temp[cnt++] = col_map_offd_B[i];
      }
   }
   if (cnt)
   {
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

   if (B_ext_offd_size || num_cols_offd_B)
   {
      nalu_hypre_TFree(temp, NALU_HYPRE_MEMORY_HOST);
   }

   for (i = 0 ; i < B_ext_offd_size; i++)
      B_ext_offd_j[i] = nalu_hypre_BigBinarySearch(col_map_offd_C,
                                              B_tmp_offd_j[i],
                                              num_cols_offd_C);
   if (B_ext_offd_size)
   {
      nalu_hypre_TFree(B_tmp_offd_j, NALU_HYPRE_MEMORY_HOST);
   }

   if (num_cols_offd_B)
   {
      map_B_to_C = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_offd_B, NALU_HYPRE_MEMORY_HOST);

      cnt = 0;
      for (i = 0; i < num_cols_offd_C; i++)
         if (col_map_offd_C[i] == col_map_offd_B[cnt])
         {
            map_B_to_C[cnt++] = i;
            if (cnt == num_cols_offd_B) { break; }
         }
   }

   nalu_hypre_ParMatmul_RowSizes(
      /*&C_diag_i, &C_offd_i, &B_marker,*/
      /* BooleanMatrix only uses HOST memory for now */
      NALU_HYPRE_MEMORY_HOST,
      &C_diag_i, &C_offd_i, NULL,
      A_diag_i, A_diag_j, A_offd_i, A_offd_j,
      B_diag_i, B_diag_j, B_offd_i, B_offd_j,
      B_ext_diag_i, B_ext_diag_j,
      B_ext_offd_i, B_ext_offd_j, map_B_to_C,
      &C_diag_size, &C_offd_size,
      num_rows_diag_A, num_rows_diag_A,
      num_cols_offd_A, allsquare,
      num_cols_diag_B, num_cols_offd_B,
      num_cols_offd_C
   );


   /*-----------------------------------------------------------------------
    *  Allocate C_diag_j arrays.
    *  Allocate C_offd_j arrays.
    *-----------------------------------------------------------------------*/

   last_col_diag_B = first_col_diag_B + (NALU_HYPRE_BigInt)num_cols_diag_B - 1;
   C_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  C_diag_size, NALU_HYPRE_MEMORY_HOST);
   if (C_offd_size)
   {
      C_offd_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  C_offd_size, NALU_HYPRE_MEMORY_HOST);
   }


   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in C_diag_j.
    *  Second Pass: Fill in C_offd_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
   *  Allocate marker array.
    *-----------------------------------------------------------------------*/

   B_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_diag_B + num_cols_offd_C, NALU_HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;
   for (i1 = 0; i1 < num_cols_diag_B + num_cols_offd_C; i1++)
   {
      B_marker[i1] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over interior c-points.
    *-----------------------------------------------------------------------*/

   for (i1 = 0; i1 < num_rows_diag_A; i1++)
   {

      /*--------------------------------------------------------------------
       *  Create diagonal entry, C_{i1,i1}
       *--------------------------------------------------------------------*/

      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      if ( allsquare )
      {
         B_marker[i1] = jj_count_diag;
         C_diag_j[jj_count_diag] = i1;
         jj_count_diag++;
      }

      /*-----------------------------------------------------------------
       *  Loop over entries in row i1 of A_offd.
       *-----------------------------------------------------------------*/

      if (num_cols_offd_A)
      {
         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
         {
            i2 = A_offd_j[jj2];

            /*-----------------------------------------------------------
             *  Loop over entries in row i2 of B_ext.
             *-----------------------------------------------------------*/

            for (jj3 = B_ext_offd_i[i2]; jj3 < B_ext_offd_i[i2 + 1]; jj3++)
            {
               i3 = num_cols_diag_B + B_ext_offd_j[jj3];

               /*--------------------------------------------------------
                *  Check B_marker to see that C_{i1,i3} has not already
                *  been accounted for. If it has not, create a new entry.
                *  If it has, add new contribution.
                *--------------------------------------------------------*/
               if (B_marker[i3] < jj_row_begin_offd)
               {
                  B_marker[i3] = jj_count_offd;
                  C_offd_j[jj_count_offd] = i3 - num_cols_diag_B;
                  jj_count_offd++;
               }
            }
            for (jj3 = B_ext_diag_i[i2]; jj3 < B_ext_diag_i[i2 + 1]; jj3++)
            {
               i3 = B_ext_diag_j[jj3];

               if (B_marker[i3] < jj_row_begin_diag)
               {
                  B_marker[i3] = jj_count_diag;
                  C_diag_j[jj_count_diag] = i3;
                  jj_count_diag++;
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

         /*-----------------------------------------------------------
          *  Loop over entries in row i2 of B_diag.
          *-----------------------------------------------------------*/

         for (jj3 = B_diag_i[i2]; jj3 < B_diag_i[i2 + 1]; jj3++)
         {
            i3 = B_diag_j[jj3];

            /*--------------------------------------------------------
             *  Check B_marker to see that C_{i1,i3} has not already
             *  been accounted for. If it has not, create a new entry.
             *  If it has, add new contribution.
             *--------------------------------------------------------*/

            if (B_marker[i3] < jj_row_begin_diag)
            {
               B_marker[i3] = jj_count_diag;
               C_diag_j[jj_count_diag] = i3;
               jj_count_diag++;
            }
         }
         if (num_cols_offd_B)
         {
            for (jj3 = B_offd_i[i2]; jj3 < B_offd_i[i2 + 1]; jj3++)
            {
               i3 = num_cols_diag_B + map_B_to_C[B_offd_j[jj3]];

               /*--------------------------------------------------------
                *  Check B_marker to see that C_{i1,i3} has not already
                *  been accounted for. If it has not, create a new entry.
                *  If it has, add new contribution.
                *--------------------------------------------------------*/

               if (B_marker[i3] < jj_row_begin_offd)
               {
                  B_marker[i3] = jj_count_offd;
                  C_offd_j[jj_count_offd] = i3 - num_cols_diag_B;
                  jj_count_offd++;
               }
            }
         }
      }
   }

   C = nalu_hypre_ParCSRBooleanMatrixCreate(comm, n_rows_A, n_cols_B, row_starts_A,
                                       col_starts_B, num_cols_offd_C, C_diag_size, C_offd_size);

   /* Note that C does not own the partitionings */
   nalu_hypre_ParCSRBooleanMatrixSetRowStartsOwner(C, 0);
   nalu_hypre_ParCSRBooleanMatrixSetColStartsOwner(C, 0);

   C_diag = nalu_hypre_ParCSRBooleanMatrix_Get_Diag(C);
   nalu_hypre_CSRBooleanMatrix_Get_I(C_diag) = C_diag_i;
   nalu_hypre_CSRBooleanMatrix_Get_J(C_diag) = C_diag_j;
   C_offd = nalu_hypre_ParCSRBooleanMatrix_Get_Offd(C);
   nalu_hypre_CSRBooleanMatrix_Get_I(C_offd) = C_offd_i;
   nalu_hypre_ParCSRBooleanMatrix_Get_Offd(C) = C_offd;

   if (num_cols_offd_C)
   {
      nalu_hypre_CSRBooleanMatrix_Get_J(C_offd) = C_offd_j;
      nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(C) = col_map_offd_C;

   }

   /*-----------------------------------------------------------------------
    *  Free B_ext and marker array.
    *-----------------------------------------------------------------------*/

   nalu_hypre_TFree(B_marker, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(B_ext_diag_i, NALU_HYPRE_MEMORY_HOST);
   if (B_ext_diag_size)
   {
      nalu_hypre_TFree(B_ext_diag_j, NALU_HYPRE_MEMORY_HOST);
   }
   nalu_hypre_TFree(B_ext_offd_i, NALU_HYPRE_MEMORY_HOST);
   if (B_ext_offd_size)
   {
      nalu_hypre_TFree(B_ext_offd_j, NALU_HYPRE_MEMORY_HOST);
   }
   if (num_cols_offd_B) { nalu_hypre_TFree(map_B_to_C, NALU_HYPRE_MEMORY_HOST); }

   return C;

}



/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixExtractBExt :
 * extracts rows from B which are located on other
 * processors and needed for multiplication with A locally. The rows
 * are returned as CSRBooleanMatrix.
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRBooleanMatrix *
nalu_hypre_ParCSRBooleanMatrixExtractBExt
( nalu_hypre_ParCSRBooleanMatrix *B, nalu_hypre_ParCSRBooleanMatrix *A )
{
   MPI_Comm comm = nalu_hypre_ParCSRBooleanMatrix_Get_Comm(B);
   NALU_HYPRE_BigInt first_col_diag = nalu_hypre_ParCSRBooleanMatrix_Get_FirstColDiag(B);
   /*NALU_HYPRE_Int first_row_index = nalu_hypre_ParCSRBooleanMatrix_Get_FirstRowIndex(B);*/
   NALU_HYPRE_BigInt *col_map_offd = nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(B);

   nalu_hypre_ParCSRCommPkg *comm_pkg = nalu_hypre_ParCSRBooleanMatrix_Get_CommPkg(A);
   NALU_HYPRE_Int num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   NALU_HYPRE_Int *recv_vec_starts = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   NALU_HYPRE_Int num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int *send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   NALU_HYPRE_Int *send_map_elmts = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

   nalu_hypre_CSRBooleanMatrix *diag = nalu_hypre_ParCSRBooleanMatrix_Get_Diag(B);
   NALU_HYPRE_Int *diag_i = nalu_hypre_CSRBooleanMatrix_Get_I(diag);
   NALU_HYPRE_Int *diag_j = nalu_hypre_CSRBooleanMatrix_Get_J(diag);

   nalu_hypre_CSRBooleanMatrix *offd = nalu_hypre_ParCSRBooleanMatrix_Get_Offd(B);
   NALU_HYPRE_Int *offd_i = nalu_hypre_CSRBooleanMatrix_Get_I(offd);
   NALU_HYPRE_Int *offd_j = nalu_hypre_CSRBooleanMatrix_Get_J(offd);

   NALU_HYPRE_Int num_cols_B, num_nonzeros;
   NALU_HYPRE_Int num_rows_B_ext;

   nalu_hypre_CSRBooleanMatrix *B_ext;
   NALU_HYPRE_Int *B_ext_i;
   NALU_HYPRE_BigInt *B_ext_j;

   NALU_HYPRE_Complex *B_ext_data = NULL, *diag_data = NULL, *offd_data = NULL;
   NALU_HYPRE_BigInt *B_ext_row_map = NULL;
   /* ... not referenced, but needed for function call */

   num_cols_B = nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];

   nalu_hypre_ParCSRMatrixExtractBExt_Arrays
   ( &B_ext_i, &B_ext_j, &B_ext_data, &B_ext_row_map,
     &num_nonzeros,
     0, 0, comm, comm_pkg,
     num_cols_B, num_recvs, num_sends,
     first_col_diag, B->row_starts,
     recv_vec_starts, send_map_starts, send_map_elmts,
     diag_i, diag_j, offd_i, offd_j, col_map_offd,
     diag_data, offd_data
   );

   B_ext = nalu_hypre_CSRBooleanMatrixCreate(num_rows_B_ext, num_cols_B, num_nonzeros);
   nalu_hypre_CSRBooleanMatrix_Get_I(B_ext) = B_ext_i;
   nalu_hypre_CSRBooleanMatrix_Get_BigJ(B_ext) = B_ext_j;

   return B_ext;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixExtractAExt : extracts rows from A which are located on other
 * processors and needed for multiplying A^T with the local part of A. The rows
 * are returned as CSRBooleanMatrix.  A row map for A_ext (like the ParCSRColMap) is
 * returned through the third argument.
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRBooleanMatrix *
nalu_hypre_ParCSRBooleanMatrixExtractAExt( nalu_hypre_ParCSRBooleanMatrix *A,
                                      NALU_HYPRE_BigInt ** pA_ext_row_map )
{
   /* Note that A's role as the first factor in A*A^T is used only
      through ...CommPkgT(A), which basically says which rows of A
      (columns of A^T) are needed.  In all the other places where A
      serves as an input, it is through its role as A^T, the matrix
      whose data needs to be passed between processors. */
   MPI_Comm comm = nalu_hypre_ParCSRBooleanMatrix_Get_Comm(A);
   NALU_HYPRE_BigInt first_col_diag = nalu_hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A);
   /*NALU_HYPRE_Int first_row_index = nalu_hypre_ParCSRBooleanMatrix_Get_FirstRowIndex(A);*/
   NALU_HYPRE_BigInt *col_map_offd = nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A);

   nalu_hypre_ParCSRCommPkg *comm_pkg = nalu_hypre_ParCSRBooleanMatrix_Get_CommPkgT(A);
   /* ... CommPkgT(A) should identify all rows of A^T needed for A*A^T (that is
    * generally a bigger set than ...CommPkg(A), the rows of B needed for A*B) */
   NALU_HYPRE_Int num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   NALU_HYPRE_Int *recv_vec_starts = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   NALU_HYPRE_Int num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int *send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   NALU_HYPRE_Int *send_map_elmts = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

   nalu_hypre_CSRBooleanMatrix *diag = nalu_hypre_ParCSRBooleanMatrix_Get_Diag(A);

   NALU_HYPRE_Int *diag_i = nalu_hypre_CSRMatrixI(diag);
   NALU_HYPRE_Int *diag_j = nalu_hypre_CSRMatrixJ(diag);

   nalu_hypre_CSRBooleanMatrix *offd = nalu_hypre_ParCSRBooleanMatrix_Get_Offd(A);

   NALU_HYPRE_Int *offd_i = nalu_hypre_CSRMatrixI(offd);
   NALU_HYPRE_Int *offd_j = nalu_hypre_CSRMatrixJ(offd);

   NALU_HYPRE_BigInt num_cols_A;
   NALU_HYPRE_Int num_nonzeros;
   NALU_HYPRE_Int num_rows_A_ext;

   nalu_hypre_CSRBooleanMatrix *A_ext;

   NALU_HYPRE_Int *A_ext_i;
   NALU_HYPRE_BigInt *A_ext_j;

   NALU_HYPRE_Int data = 0;
   NALU_HYPRE_Complex *A_ext_data = NULL, *diag_data = NULL, *offd_data = NULL;
   /* ... not referenced, but needed for function call */

   num_cols_A = nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNCols(A);
   num_rows_A_ext = recv_vec_starts[num_recvs];

   nalu_hypre_ParCSRMatrixExtractBExt_Arrays
   ( &A_ext_i, &A_ext_j, &A_ext_data, pA_ext_row_map,
     &num_nonzeros,
     data, 1, comm, comm_pkg,
     num_cols_A, num_recvs, num_sends,
     first_col_diag, A->row_starts,
     recv_vec_starts, send_map_starts, send_map_elmts,
     diag_i, diag_j, offd_i, offd_j, col_map_offd,
     diag_data, offd_data
   );

   A_ext = nalu_hypre_CSRBooleanMatrixCreate(num_rows_A_ext, num_cols_A, num_nonzeros);
   nalu_hypre_CSRBooleanMatrix_Get_I(A_ext) = A_ext_i;
   nalu_hypre_CSRBooleanMatrix_Get_BigJ(A_ext) = A_ext_j;

   return A_ext;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParBooleanAAT : multiplies nalu_hypre_ParCSRBooleanMatrix A by its transpose,
 * A*A^T, and returns the product in nalu_hypre_ParCSRBooleanMatrix C
 * Note that C does not own the partitionings
 * This is based on nalu_hypre_ParCSRAAt.
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRBooleanMatrix * nalu_hypre_ParBooleanAAt( nalu_hypre_ParCSRBooleanMatrix  * A )
{
   MPI_Comm       comm = nalu_hypre_ParCSRBooleanMatrix_Get_Comm(A);

   nalu_hypre_CSRBooleanMatrix *A_diag = nalu_hypre_ParCSRBooleanMatrix_Get_Diag(A);

   NALU_HYPRE_Int             *A_diag_i = nalu_hypre_CSRBooleanMatrix_Get_I(A_diag);
   NALU_HYPRE_Int             *A_diag_j = nalu_hypre_CSRBooleanMatrix_Get_J(A_diag);

   nalu_hypre_CSRBooleanMatrix *A_offd = nalu_hypre_ParCSRBooleanMatrix_Get_Offd(A);
   NALU_HYPRE_Int             *A_offd_i = nalu_hypre_CSRBooleanMatrix_Get_I(A_offd);
   NALU_HYPRE_Int             *A_offd_j = nalu_hypre_CSRBooleanMatrix_Get_J(A_offd);

   NALU_HYPRE_BigInt          *A_col_map_offd = nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A);
   NALU_HYPRE_BigInt          *A_ext_row_map;

   NALU_HYPRE_BigInt *row_starts_A = nalu_hypre_ParCSRBooleanMatrix_Get_RowStarts(A);
   NALU_HYPRE_Int   num_rows_diag_A = nalu_hypre_CSRBooleanMatrix_Get_NRows(A_diag);
   NALU_HYPRE_Int   num_cols_offd_A = nalu_hypre_CSRBooleanMatrix_Get_NCols(A_offd);

   nalu_hypre_ParCSRBooleanMatrix *C;
   NALU_HYPRE_BigInt            *col_map_offd_C;

   nalu_hypre_CSRBooleanMatrix *C_diag;

   NALU_HYPRE_Int             *C_diag_i;
   NALU_HYPRE_Int             *C_diag_j;

   nalu_hypre_CSRBooleanMatrix *C_offd;

   NALU_HYPRE_Int             *C_offd_i = NULL;
   NALU_HYPRE_Int             *C_offd_j = NULL;
   NALU_HYPRE_Int             *new_C_offd_j;

   NALU_HYPRE_Int              C_diag_size;
   NALU_HYPRE_Int              C_offd_size;
   NALU_HYPRE_Int          last_col_diag_C;
   NALU_HYPRE_Int          num_cols_offd_C;

   nalu_hypre_CSRBooleanMatrix *A_ext;

   NALU_HYPRE_Int             *A_ext_i;
   NALU_HYPRE_BigInt          *A_ext_j;
   NALU_HYPRE_Int             num_rows_A_ext = 0;

   NALU_HYPRE_BigInt   first_row_index_A = nalu_hypre_ParCSRBooleanMatrix_Get_FirstRowIndex(A);
   NALU_HYPRE_BigInt   first_col_diag_A = nalu_hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A);
   NALU_HYPRE_Int         *B_marker;

   NALU_HYPRE_Int              i;
   NALU_HYPRE_Int              i1, i2, i3;
   NALU_HYPRE_Int              jj2, jj3;

   NALU_HYPRE_Int              jj_count_diag, jj_count_offd;
   NALU_HYPRE_Int              jj_row_begin_diag, jj_row_begin_offd;
   NALU_HYPRE_Int              start_indexing = 0; /* start indexing for C_data at 0 */
   NALU_HYPRE_Int          count;
   NALU_HYPRE_BigInt          n_rows_A, n_cols_A;

   n_rows_A = nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNRows(A);
   n_cols_A = nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNCols(A);

   if (n_cols_A != n_rows_A)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, " Error! Incompatible matrix dimensions!\n");
      return NULL;
   }
   /*-----------------------------------------------------------------------
    *  Extract A_ext, i.e. portion of A that is stored on neighbor procs
    *  and needed locally for A^T in the matrix matrix product A*A^T
    *-----------------------------------------------------------------------*/

   if ((NALU_HYPRE_BigInt)num_rows_diag_A != n_rows_A)
   {
      /*---------------------------------------------------------------------
      * If there exists no CommPkg for A, a CommPkg is generated using
      * equally load balanced partitionings
      *--------------------------------------------------------------------*/
      if (!nalu_hypre_ParCSRBooleanMatrix_Get_CommPkg(A))
      {
         nalu_hypre_BooleanMatTCommPkgCreate(A);
      }

      A_ext = nalu_hypre_ParCSRBooleanMatrixExtractAExt( A, &A_ext_row_map );
      A_ext_i    = nalu_hypre_CSRBooleanMatrix_Get_I(A_ext);
      A_ext_j    = nalu_hypre_CSRBooleanMatrix_Get_BigJ(A_ext);
      num_rows_A_ext = nalu_hypre_CSRBooleanMatrix_Get_NRows(A_ext);
   }
   /*-----------------------------------------------------------------------
   *  Allocate marker array.
    *-----------------------------------------------------------------------*/

   B_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_rows_diag_A + num_rows_A_ext, NALU_HYPRE_MEMORY_HOST);

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   for ( i1 = 0; i1 < num_rows_diag_A + num_rows_A_ext; ++i1 )
   {
      B_marker[i1] = -1;
   }


   nalu_hypre_ParAat_RowSizes(
      &C_diag_i, &C_offd_i, B_marker,
      A_diag_i, A_diag_j,
      A_offd_i, A_offd_j, A_col_map_offd,
      A_ext_i, A_ext_j, A_ext_row_map,
      &C_diag_size, &C_offd_size,
      num_rows_diag_A, num_cols_offd_A,
      num_rows_A_ext,
      first_col_diag_A, first_row_index_A
   );

#if 0
   /* debugging output: */
   nalu_hypre_printf("A_ext_row_map (%i):", num_rows_A_ext);
   for ( i1 = 0; i1 < num_rows_A_ext; ++i1 ) { nalu_hypre_printf(" %i", A_ext_row_map[i1] ); }
   nalu_hypre_printf("\nC_diag_i (%i):", C_diag_size);
   for ( i1 = 0; i1 <= num_rows_diag_A; ++i1 ) { nalu_hypre_printf(" %i", C_diag_i[i1] ); }
   nalu_hypre_printf("\nC_offd_i (%i):", C_offd_size);
   for ( i1 = 0; i1 <= num_rows_diag_A; ++i1 ) { nalu_hypre_printf(" %i", C_offd_i[i1] ); }
   nalu_hypre_printf("\n");
#endif

   /*-----------------------------------------------------------------------
    *  Allocate C_diag_j arrays.
    *  Allocate C_offd_j arrays.
    *-----------------------------------------------------------------------*/

   last_col_diag_C = first_row_index_A + num_rows_diag_A - 1;
   C_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  C_diag_size, NALU_HYPRE_MEMORY_HOST);
   if (C_offd_size)
   {
      C_offd_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  C_offd_size, NALU_HYPRE_MEMORY_HOST);
   }


   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in C_diag_j.
    *  Second Pass: Fill in C_offd_j.
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;
   for ( i1 = 0; i1 < num_rows_diag_A + num_rows_A_ext; ++i1 )
   {
      B_marker[i1] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over interior c-points.
    *-----------------------------------------------------------------------*/

   for (i1 = 0; i1 < num_rows_diag_A; i1++)
   {

      /*--------------------------------------------------------------------
       *  Create diagonal entry, C_{i1,i1}
       *--------------------------------------------------------------------*/

      B_marker[i1] = jj_count_diag;
      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      C_diag_j[jj_count_diag] = i1;
      jj_count_diag++;

      /*-----------------------------------------------------------------
       *  Loop over entries in row i1 of A_offd.
       *-----------------------------------------------------------------*/

      /* There are 3 CSRMatrix or CSRBooleanMatrix objects here:
         ext*ext, ext*diag, and ext*offd belong to another processor.
         diag*offd and offd*diag don't count - never share a column by definition.
         So we have to do 4 cases:
         diag*ext, offd*ext, diag*diag, and offd*offd.
      */

      for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
      {
         i2 = A_diag_j[jj2];

         /* diag*ext */
         /*-----------------------------------------------------------
          *  Loop over entries (columns) i3 in row i2 of (A_ext)^T
          *  That is, rows i3 having a column i2 of A_ext.
          *  For now, for each row i3 of A_ext we crudely check _all_
          *  columns to see whether one matches i2.
          *  For each entry (i2,i3) of (A_ext)^T, A(i1,i2)*A(i3,i2) defines
          *  C(i1,i3) .  This contributes to both the diag and offd
          *  blocks of C.
          *-----------------------------------------------------------*/

         for ( i3 = 0; i3 < num_rows_A_ext; i3++ )
         {
            for ( jj3 = A_ext_i[i3]; jj3 < A_ext_i[i3 + 1]; jj3++ )
            {
               if ( A_ext_j[jj3] == (NALU_HYPRE_BigInt)i2 + first_col_diag_A )
               {
                  /* row i3, column i2 of A_ext; or,
                     row i2, column i3 of (A_ext)^T */

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *--------------------------------------------------------*/

                  if ( A_ext_row_map[i3] < first_row_index_A ||
                       A_ext_row_map[i3] > last_col_diag_C )   /* offd */
                  {
                     if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_offd)
                     {
                        B_marker[i3 + num_rows_diag_A] = jj_count_offd;
                        C_offd_j[jj_count_offd] = i3;
                        jj_count_offd++;
                     }
                  }
                  else                                                /* diag */
                  {
                     if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_diag)
                     {
                        B_marker[i3 + num_rows_diag_A] = jj_count_diag;
                        C_diag_j[jj_count_diag] = i3 - (NALU_HYPRE_Int)first_col_diag_A;
                        jj_count_diag++;
                     }
                  }
               }
            }
         }
      }

      if (num_cols_offd_A)
      {
         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
         {
            i2 = A_offd_j[jj2];

            /* offd * ext */
            /*-----------------------------------------------------------
             *  Loop over entries (columns) i3 in row i2 of (A_ext)^T
             *  That is, rows i3 having a column i2 of A_ext.
             *  For now, for each row i3 of A_ext we crudely check _all_
             *  columns to see whether one matches i2.
             *  For each entry (i2,i3) of (A_ext)^T, A(i1,i2)*A(i3,i2) defines
             *  C(i1,i3) .  This contributes to both the diag and offd
             *  blocks of C.
             *-----------------------------------------------------------*/

            for ( i3 = 0; i3 < num_rows_A_ext; i3++ )
            {
               for ( jj3 = A_ext_i[i3]; jj3 < A_ext_i[i3 + 1]; jj3++ )
               {
                  if ( A_ext_j[jj3] == A_col_map_offd[i2] )
                  {
                     /* row i3, column i2 of A_ext; or,
                        row i2, column i3 of (A_ext)^T */

                     /*--------------------------------------------------------
                      *  Check B_marker to see that C_{i1,i3} has not already
                      *  been accounted for. If it has not, create a new entry.
                      *  If it has, add new contribution.
                      *--------------------------------------------------------*/

                     if ( A_ext_row_map[i3] < first_row_index_A ||
                          A_ext_row_map[i3] > last_col_diag_C )   /* offd */
                     {
                        if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_offd)
                        {
                           B_marker[i3 + num_rows_diag_A] = jj_count_offd;
                           C_offd_j[jj_count_offd] = i3;
                           jj_count_offd++;
                        }
                     }
                     else                                                /* diag */
                     {
                        if (B_marker[i3 + num_rows_diag_A] < jj_row_begin_diag)
                        {
                           B_marker[i3 + num_rows_diag_A] = jj_count_diag;
                           C_diag_j[jj_count_diag] = i3 - (NALU_HYPRE_Int)first_row_index_A;
                           jj_count_diag++;
                        }
                     }
                  }
               }
            }
         }
      }

      /* diag * diag */
      /*-----------------------------------------------------------------
       *  Loop over entries (columns) i2 in row i1 of A_diag.
       *  For each such column we will find the contributions of the
       *  corresponding rows i2 of A^T to C=A*A^T .  Now we only look
       *  at the local part of A^T - with columns (rows of A) living
       *  on this processor.
       *-----------------------------------------------------------------*/

      for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1 + 1]; jj2++)
      {
         i2 = A_diag_j[jj2];

         /*-----------------------------------------------------------
          *  Loop over entries (columns) i3 in row i2 of A^T
          *  That is, rows i3 having a column i2 of A (local part).
          *  For now, for each row i3 of A we crudely check _all_
          *  columns to see whether one matches i2.
          *  This i3-loop is for the diagonal block of A.
          *  It contributes to the diagonal block of C.
          *  For each entry (i2,i3) of A^T, A(i1,i2)*A(i3,i2) defines
          *  to C(i1,i3)
          *-----------------------------------------------------------*/
         for ( i3 = 0; i3 < num_rows_diag_A; i3++ )
         {
            for ( jj3 = A_diag_i[i3]; jj3 < A_diag_i[i3 + 1]; jj3++ )
            {
               if ( A_diag_j[jj3] == i2 )
               {
                  /* row i3, column i2 of A; or,
                     row i2, column i3 of A^T */

                  /*--------------------------------------------------------
                   *  Check B_marker to see that C_{i1,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/
                  if (B_marker[i3] < jj_row_begin_diag)
                  {
                     B_marker[i3] = jj_count_diag;
                     C_diag_j[jj_count_diag] = i3;
                     jj_count_diag++;
                  }
               }
            }
         } /* end of i3 loop */
      } /* end of third i2 loop */


      /* offd * offd */
      /*-----------------------------------------------------------
       *  Loop over offd columns i2 of A in A*A^T.  Then
       *  loop over offd entries (columns) i3 in row i2 of A^T
       *  That is, rows i3 having a column i2 of A (local part).
       *  For now, for each row i3 of A we crudely check _all_
       *  columns to see whether one matches i2.
       *  This i3-loop is for the off-diagonal block of A.
       *  It contributes to the diag block of C.
       *  For each entry (i2,i3) of A^T, A*A^T defines C
       *-----------------------------------------------------------*/
      if (num_cols_offd_A)
      {

         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1 + 1]; jj2++)
         {
            i2 = A_offd_j[jj2];

            for ( i3 = 0; i3 < num_rows_diag_A; i3++ )
            {
               /* ... note that num_rows_diag_A == num_rows_offd_A */
               for ( jj3 = A_offd_i[i3]; jj3 < A_offd_i[i3 + 1]; jj3++ )
               {
                  if ( A_offd_j[jj3] == i2 )
                  {
                     /* row i3, column i2 of A; or,
                        row i2, column i3 of A^T */

                     /*--------------------------------------------------------
                      *  Check B_marker to see that C_{i1,i3} has not already
                      *  been accounted for. If it has not, create a new entry.
                      *  If it has, add new contribution
                      *--------------------------------------------------------*/

                     if (B_marker[i3] < jj_row_begin_diag)
                     {
                        B_marker[i3] = jj_count_diag;
                        C_diag_j[jj_count_diag] = i3;
                        jj_count_diag++;
                     }
                  }
               }
            }  /* end of last i3 loop */
         }     /* end of if (num_cols_offd_A) */

      }        /* end of fourth and last i2 loop */
#if 0          /* debugging printout */
      nalu_hypre_printf("end of i1 loop: i1=%i jj_count_diag=%i\n", i1, jj_count_diag );
      nalu_hypre_printf("  C_diag_j=");
      for ( jj3 = 0; jj3 < jj_count_diag; ++jj3) { nalu_hypre_printf("%i ", C_diag_j[jj3]); }
      nalu_hypre_printf("\n");
      nalu_hypre_printf("  C_offd_j=");
      for ( jj3 = 0; jj3 < jj_count_offd; ++jj3) { nalu_hypre_printf("%i ", C_offd_j[jj3]); }
      nalu_hypre_printf("\n");
      nalu_hypre_printf( "  B_marker =" );
      for ( it = 0; it < num_rows_diag_A + num_rows_A_ext; ++it )
      {
         nalu_hypre_printf(" %i", B_marker[it] );
      }
      nalu_hypre_printf( "\n" );
#endif
   }           /* end of i1 loop */

   /*-----------------------------------------------------------------------
    *  Delete 0-columns in C_offd, i.e. generate col_map_offd and reset
    *  C_offd_j.  Note that (with the indexing we have coming into this
    *  block) col_map_offd_C[i3]==A_ext_row_map[i3].
    *-----------------------------------------------------------------------*/

   for ( i = 0; i < num_rows_diag_A + num_rows_A_ext; ++i )
   {
      B_marker[i] = -1;
   }
   for ( i = 0; i < C_offd_size; i++ )
   {
      B_marker[ C_offd_j[i] ] = -2;
   }

   count = 0;
   for (i = 0; i < num_rows_diag_A + num_rows_A_ext; i++)
   {
      if (B_marker[i] == -2)
      {
         B_marker[i] = count;
         count++;
      }
   }
   num_cols_offd_C = count;

   if (num_cols_offd_C)
   {
      col_map_offd_C = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_C, NALU_HYPRE_MEMORY_HOST);
      new_C_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, C_offd_size, NALU_HYPRE_MEMORY_HOST);
      /* ... a bit big, but num_cols_offd_C is too small.  It might be worth
         computing the correct size, which is sum( no. columns in row i, over all rows i )
      */

      for (i = 0; i < C_offd_size; i++)
      {
         new_C_offd_j[i] = B_marker[C_offd_j[i]];
         col_map_offd_C[ new_C_offd_j[i] ] = A_ext_row_map[ C_offd_j[i] ];
      }

      nalu_hypre_TFree(C_offd_j, NALU_HYPRE_MEMORY_HOST);
      C_offd_j = new_C_offd_j;

   }

   /*----------------------------------------------------------------
    * Create C
    *----------------------------------------------------------------*/

   C = nalu_hypre_ParCSRBooleanMatrixCreate(comm, n_rows_A, n_rows_A, row_starts_A,
                                       row_starts_A, num_cols_offd_C, C_diag_size, C_offd_size);

   /* Note that C does not own the partitionings */
   nalu_hypre_ParCSRBooleanMatrixSetRowStartsOwner(C, 0);
   nalu_hypre_ParCSRBooleanMatrixSetColStartsOwner(C, 0);

   C_diag = nalu_hypre_ParCSRBooleanMatrix_Get_Diag(C);
   nalu_hypre_CSRBooleanMatrix_Get_I(C_diag) = C_diag_i;
   nalu_hypre_CSRBooleanMatrix_Get_J(C_diag) = C_diag_j;

   if (num_cols_offd_C)
   {
      C_offd = nalu_hypre_ParCSRBooleanMatrix_Get_Offd(C);
      nalu_hypre_CSRBooleanMatrix_Get_I(C_offd) = C_offd_i;
      nalu_hypre_CSRBooleanMatrix_Get_J(C_offd) = C_offd_j;
      nalu_hypre_ParCSRBooleanMatrix_Get_Offd(C) = C_offd;
      nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(C) = col_map_offd_C;

   }
   else
   {
      nalu_hypre_TFree(C_offd_i, NALU_HYPRE_MEMORY_HOST);
   }

   /*-----------------------------------------------------------------------
    *  Free B_ext and marker array.
    *-----------------------------------------------------------------------*/

   if (num_cols_offd_A)
   {
      nalu_hypre_CSRBooleanMatrixDestroy(A_ext);
      A_ext = NULL;
   }
   nalu_hypre_TFree(B_marker, NALU_HYPRE_MEMORY_HOST);
   if ( num_rows_diag_A != n_rows_A )
   {
      nalu_hypre_TFree(A_ext_row_map, NALU_HYPRE_MEMORY_HOST);
   }

   return C;

}


/* ----------------------------------------------------------------------
 * nalu_hypre_BooleanMatTCommPkgCreate
 * generates a special comm_pkg for a Boolean matrix A - for use in multiplying
 * by its transpose, A * A^T
 * if no row and/or column partitioning is given, the routine determines
 * them with MPE_Decomp1d
 * ---------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BooleanMatTCommPkgCreate ( nalu_hypre_ParCSRBooleanMatrix *A)
{
   MPI_Comm       comm = nalu_hypre_ParCSRBooleanMatrix_Get_Comm(A);
   NALU_HYPRE_BigInt  *col_map_offd = nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A);
   NALU_HYPRE_BigInt   first_col_diag = nalu_hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A);
   NALU_HYPRE_BigInt  *col_starts = nalu_hypre_ParCSRBooleanMatrix_Get_ColStarts(A);
   NALU_HYPRE_Int      num_rows_diag = nalu_hypre_CSRBooleanMatrix_Get_NRows(nalu_hypre_ParCSRBooleanMatrix_Get_Diag(
                                                                      A));
   NALU_HYPRE_Int      num_cols_diag = nalu_hypre_CSRBooleanMatrix_Get_NCols(nalu_hypre_ParCSRBooleanMatrix_Get_Diag(
                                                                      A));
   NALU_HYPRE_Int      num_cols_offd = nalu_hypre_CSRBooleanMatrix_Get_NCols(nalu_hypre_ParCSRBooleanMatrix_Get_Offd(
                                                                      A));
   NALU_HYPRE_BigInt  *row_starts = nalu_hypre_ParCSRBooleanMatrix_Get_RowStarts(A);

   NALU_HYPRE_Int      num_sends;
   NALU_HYPRE_Int     *send_procs;
   NALU_HYPRE_Int     *send_map_starts;
   NALU_HYPRE_Int     *send_map_elmts;
   NALU_HYPRE_Int      num_recvs;
   NALU_HYPRE_Int     *recv_procs;
   NALU_HYPRE_Int     *recv_vec_starts;

   nalu_hypre_ParCSRCommPkg  *comm_pkg = NULL;

   nalu_hypre_MatTCommPkgCreate_core (
      comm, col_map_offd, first_col_diag, col_starts,
      num_rows_diag, num_cols_diag, num_cols_offd, row_starts,
      nalu_hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A),
      nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A),
      nalu_hypre_CSRBooleanMatrix_Get_I( nalu_hypre_ParCSRBooleanMatrix_Get_Diag(A) ),
      nalu_hypre_CSRBooleanMatrix_Get_J( nalu_hypre_ParCSRBooleanMatrix_Get_Diag(A) ),
      nalu_hypre_CSRBooleanMatrix_Get_I( nalu_hypre_ParCSRBooleanMatrix_Get_Offd(A) ),
      nalu_hypre_CSRBooleanMatrix_Get_J( nalu_hypre_ParCSRBooleanMatrix_Get_Offd(A) ),
      0,
      &num_recvs, &recv_procs, &recv_vec_starts,
      &num_sends, &send_procs, &send_map_starts,
      &send_map_elmts
   );

   /* Create communication package */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs, recv_procs, recv_vec_starts,
                                    num_sends, send_procs, send_map_starts,
                                    send_map_elmts,
                                    &comm_pkg);

   nalu_hypre_ParCSRBooleanMatrix_Get_CommPkgT(A) = comm_pkg;

   return nalu_hypre_error_flag;
}

/* ----------------------------------------------------------------------
 * nalu_hypre_BooleanMatvecCommPkgCreate
 * generates the comm_pkg for a Boolean matrix A , to be used for A*B.
 * if no row and/or column partitioning is given, the routine determines
 * them with MPE_Decomp1d
 * ---------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BooleanMatvecCommPkgCreate ( nalu_hypre_ParCSRBooleanMatrix *A)
{
   MPI_Comm        comm = nalu_hypre_ParCSRBooleanMatrix_Get_Comm(A);
   NALU_HYPRE_BigInt   *col_map_offd = nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(A);
   NALU_HYPRE_BigInt    first_col_diag = nalu_hypre_ParCSRBooleanMatrix_Get_FirstColDiag(A);
   NALU_HYPRE_BigInt   *col_starts = nalu_hypre_ParCSRBooleanMatrix_Get_ColStarts(A);
   NALU_HYPRE_Int       num_cols_diag = nalu_hypre_CSRBooleanMatrix_Get_NCols(nalu_hypre_ParCSRBooleanMatrix_Get_Diag(
                                                                       A));
   NALU_HYPRE_Int       num_cols_offd = nalu_hypre_CSRBooleanMatrix_Get_NCols(nalu_hypre_ParCSRBooleanMatrix_Get_Offd(
                                                                       A));

   NALU_HYPRE_Int       num_sends;
   NALU_HYPRE_Int      *send_procs;
   NALU_HYPRE_Int      *send_map_starts;
   NALU_HYPRE_Int      *send_map_elmts;
   NALU_HYPRE_Int       num_recvs;
   NALU_HYPRE_Int      *recv_procs;
   NALU_HYPRE_Int      *recv_vec_starts;

   nalu_hypre_ParCSRCommPkg  *comm_pkg = NULL;

   nalu_hypre_ParCSRCommPkgCreate_core
   (
      comm, col_map_offd, first_col_diag, col_starts,
      num_cols_diag, num_cols_offd,
      &num_recvs, &recv_procs, &recv_vec_starts,
      &num_sends, &send_procs, &send_map_starts,
      &send_map_elmts
   );

   /* Create communication package */
   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs, recv_procs, recv_vec_starts,
                                    num_sends, send_procs, send_map_starts,
                                    send_map_elmts,
                                    &comm_pkg);

   nalu_hypre_ParCSRBooleanMatrix_Get_CommPkg(A) = comm_pkg;

   return nalu_hypre_error_flag;
}
