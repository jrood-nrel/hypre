/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * IJMatrix_ISIS interface
 *
 *****************************************************************************/

#include "_hypre_IJ_mv.h"

/******************************************************************************
 *
 * hypre_IJMatrixSetLocalSizeISIS
 *
 * sets local number of rows and number of columns of diagonal matrix on
 * current processor.
 *
 *****************************************************************************/

NALU_HYPRE_Int
hypre_IJMatrixSetLocalSizeISIS(hypre_IJMatrix *matrix,
                               NALU_HYPRE_Int       local_m,
                               NALU_HYPRE_Int       local_n)
{
   NALU_HYPRE_Int ierr = 0;
   hypre_AuxParCSRMatrix *aux_data;
   aux_data = hypre_IJMatrixTranslator(matrix);
   if (aux_data)
   {
      hypre_AuxParCSRMatrixLocalNumRows(aux_data) = local_m;
      hypre_AuxParCSRMatrixLocalNumCols(aux_data) = local_n;
   }
   else
   {
      hypre_IJMatrixTranslator(matrix) =
         hypre_AuxParCSRMatrixCreate(local_m, local_n, NULL);
   }
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixCreateISIS
 *
 * creates AuxParCSRMatrix and ParCSRMatrix if necessary,
 * generates arrays row_starts and col_starts using either previously
 * set data local_m and local_n (user defined) or generates them evenly
 * distributed if not previously defined by user.
 *
 *****************************************************************************/
NALU_HYPRE_Int
hypre_IJMatrixCreateISIS(hypre_IJMatrix *matrix)
{
   MPI_Comm comm = hypre_IJMatrixContext(matrix);
   NALU_HYPRE_BigInt global_m = hypre_IJMatrixM(matrix);
   NALU_HYPRE_BigInt global_n = hypre_IJMatrixN(matrix);
   hypre_AuxParCSRMatrix *aux_matrix = hypre_IJMatrixTranslator(matrix);
   NALU_HYPRE_Int local_m;
   NALU_HYPRE_Int local_n;
   NALU_HYPRE_Int ierr = 0;


   NALU_HYPRE_BigInt *row_starts;
   NALU_HYPRE_BigInt *col_starts;
   NALU_HYPRE_Int num_cols_offd = 0;
   NALU_HYPRE_Int num_nonzeros_diag = 0;
   NALU_HYPRE_Int num_nonzeros_offd = 0;
   NALU_HYPRE_Int num_procs, my_id;
   NALU_HYPRE_Int equal;
   NALU_HYPRE_Int i;
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (aux_matrix)
   {
      local_m = hypre_AuxParCSRMatrixLocalNumRows(aux_matrix);
      local_n = hypre_AuxParCSRMatrixLocalNumCols(aux_matrix);
   }
   else
   {
      aux_matrix = hypre_AuxParCSRMatrixCreate(-1, -1, NULL);
      local_m = -1;
      local_n = -1;
      hypre_IJMatrixTranslator(matrix) = aux_matrix;
   }

   if (local_m < 0)
   {
      row_starts = NULL;
   }
   else
   {
      row_starts = hypre_CTAlloc(NALU_HYPRE_Int, num_procs + 1, NALU_HYPRE_MEMORY_HOST);

      if (my_id == 0 && (NALU_HYPRE_BigInt)local_m == global_m)
      {
         row_starts[1] = (NALU_HYPRE_BigInt)local_m;
      }
      else
      {
         NALU_HYPRE_BigInt big_local_m = (NALU_HYPRE_BigInt) local_m;
         hypre_MPI_Allgather(&big_local_m, 1, NALU_HYPRE_MPI_BIG_INT, &row_starts[1], 1, NALU_HYPRE_MPI_BIG_INT, comm);
      }

   }
   if (local_n < 0)
   {
      col_starts = NULL;
   }
   else
   {
      col_starts = hypre_CTAlloc(NALU_HYPRE_Int, num_procs + 1, NALU_HYPRE_MEMORY_HOST);

      if (my_id == 0 && local_n == global_n)
      {
         col_starts[1] = (NALU_HYPRE_BigInt)local_n;
      }
      else
      {
         NALU_HYPRE_BigInt big_local_n = (NALU_HYPRE_BigInt) local_n;
         hypre_MPI_Allgather(&big_local_n, 1, NALU_HYPRE_MPI_BIG_INT, &col_starts[1], 1, NALU_HYPRE_MPI_BIG_INT, comm);
      }
   }

   if (row_starts && col_starts)
   {
      equal = 1;
      for (i = 0; i < num_procs; i++)
      {
         row_starts[i + 1] += row_starts[i];
         col_starts[i + 1] += col_starts[i];
         if (row_starts[i + 1] != col_starts[i + 1])
         {
            equal = 0;
         }
      }
      if (equal)
      {
         hypre_TFree(col_starts, NALU_HYPRE_MEMORY_HOST);
         col_starts = row_starts;
      }
   }

   hypre_IJMatrixLocalStorage(matrix) = hypre_ParCSRMatrixCreate(comm, global_m,
                                                                 global_n, row_starts, col_starts, num_cols_offd,
                                                                 num_nonzeros_diag, num_nonzeros_offd);
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixSetRowSizesISIS
 *
 *****************************************************************************/
NALU_HYPRE_Int
hypre_IJMatrixSetRowSizesISIS(hypre_IJMatrix *matrix,
                              NALU_HYPRE_Int          *sizes)
{
   NALU_HYPRE_Int *row_space;
   NALU_HYPRE_Int local_num_rows;
   NALU_HYPRE_Int i;
   hypre_AuxParCSRMatrix *aux_matrix;
   aux_matrix = hypre_IJMatrixTranslator(matrix);
   if (aux_matrix)
   {
      local_num_rows = hypre_AuxParCSRMatrixLocalNumRows(aux_matrix);
   }
   else
   {
      return -1;
   }

   row_space =  hypre_AuxParCSRMatrixRowSpace(aux_matrix);
   if (!row_space)
   {
      row_space = hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows, NALU_HYPRE_MEMORY_HOST);
   }
   for (i = 0; i < local_num_rows; i++)
   {
      row_space[i] = sizes[i];
   }
   hypre_AuxParCSRMatrixRowSpace(aux_matrix) = row_space;
   return 0;
}

/******************************************************************************
 *
 * hypre_IJMatrixSetDiagRowSizesISIS
 * sets diag_i inside the diag part of the ParCSRMatrix,
 * requires exact sizes for diag
 *
 *****************************************************************************/
NALU_HYPRE_Int
hypre_IJMatrixSetDiagRowSizesISIS(hypre_IJMatrix *matrix,
                                  NALU_HYPRE_Int             *sizes)
{
   NALU_HYPRE_Int local_num_rows;
   NALU_HYPRE_Int i;
   hypre_ParCSRMatrix *par_matrix;
   hypre_CSRMatrix *diag;
   NALU_HYPRE_Int *diag_i;
   par_matrix = hypre_IJMatrixLocalStorage(matrix);
   if (!par_matrix)
   {
      return -1;
   }

   diag =  hypre_ParCSRMatrixDiag(par_matrix);
   diag_i =  hypre_CSRMatrixI(diag);
   local_num_rows = hypre_CSRMatrixNumRows(diag);
   if (!diag_i)
   {
      diag_i = hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows + 1, NALU_HYPRE_MEMORY_HOST);
   }
   for (i = 0; i < local_num_rows + 1; i++)
   {
      diag_i[i] = sizes[i];
   }
   hypre_CSRMatrixI(diag) = diag_i;
   hypre_CSRMatrixNumNonzeros(diag) = diag_i[local_num_rows];
   return 0;
}

/******************************************************************************
 *
 * hypre_IJMatrixSetOffDiagRowSizesISIS
 * sets offd_i inside the offd part of the ParCSRMatrix,
 * requires exact sizes for offd
 *
 *****************************************************************************/
NALU_HYPRE_Int
hypre_IJMatrixSetOffDiagRowSizesISIS(hypre_IJMatrix *matrix,
                                     NALU_HYPRE_Int       *sizes)
{
   NALU_HYPRE_Int local_num_rows;
   NALU_HYPRE_Int i;
   hypre_ParCSRMatrix *par_matrix;
   hypre_CSRMatrix *offd;
   NALU_HYPRE_Int *offd_i;
   par_matrix = hypre_IJMatrixLocalStorage(matrix);
   if (!par_matrix)
   {
      return -1;
   }

   offd =  hypre_ParCSRMatrixOffd(par_matrix);
   offd_i =  hypre_CSRMatrixI(offd);
   local_num_rows = hypre_CSRMatrixNumRows(offd);
   if (!offd_i)
   {
      offd_i = hypre_CTAlloc(NALU_HYPRE_Int,  local_num_rows + 1, NALU_HYPRE_MEMORY_HOST);
   }
   for (i = 0; i < local_num_rows + 1; i++)
   {
      offd_i[i] = sizes[i];
   }
   hypre_CSRMatrixI(offd) = offd_i;
   hypre_CSRMatrixNumNonzeros(offd) = offd_i[local_num_rows];
   return 0;
}

/******************************************************************************
 *
 * hypre_IJMatrixInitializeISIS
 *
 * initializes AuxParCSRMatrix and ParCSRMatrix as necessary
 *
 *****************************************************************************/

NALU_HYPRE_Int
hypre_IJMatrixInitializeISIS(hypre_IJMatrix *matrix)
{
   NALU_HYPRE_Int ierr = 0;
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixLocalStorage(matrix);
   hypre_AuxParCSRMatrix *aux_matrix = hypre_IJMatrixTranslator(matrix);
   NALU_HYPRE_Int local_num_rows = hypre_AuxParCSRMatrixLocalNumRows(aux_matrix);
   NALU_HYPRE_Int local_num_cols = hypre_AuxParCSRMatrixLocalNumCols(aux_matrix);
   NALU_HYPRE_Int *row_space = hypre_AuxParCSRMatrixRowSpace(aux_matrix);
   NALU_HYPRE_Int num_nonzeros = hypre_ParCSRMatrixNumNonzeros(par_matrix);
   NALU_HYPRE_Int local_nnz;
   NALU_HYPRE_Int num_procs, my_id;
   MPI_Comm  comm = hypre_IJMatrixContext(matrix);
   NALU_HYPRE_BigInt global_num_rows = hypre_IJMatrixM(matrix);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   local_nnz = (num_nonzeros / global_num_rows + 1) * local_num_rows;
   if (local_num_rows < 0)
      hypre_AuxParCSRMatrixLocalNumRows(aux_matrix) =
         hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(par_matrix));
   if (local_num_cols < 0)
      hypre_AuxParCSRMatrixLocalNumCols(aux_matrix) =
         hypre_CSRMatrixNumCols(hypre_ParCSRMatrixDiag(par_matrix));
   ierr = hypre_AuxParCSRMatrixInitialize(aux_matrix);
   ierr += hypre_ParCSRMatrixBigInitialize(par_matrix);
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixInsertBlockISIS
 *
 * inserts a block of values into an IJMatrix, currently it just uses
 * InsertIJMatrixRowISIS
 *
 *****************************************************************************/
NALU_HYPRE_Int
hypre_IJMatrixInsertBlockISIS(hypre_IJMatrix *matrix,
                              NALU_HYPRE_Int           m,
                              NALU_HYPRE_Int         n,
                              NALU_HYPRE_BigInt *rows,
                              NALU_HYPRE_BigInt *cols,
                              NALU_HYPRE_Complex   *coeffs)
{
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int i, in;
   for (i = 0; i < m; i++)
   {
      in = i * n;
      hypre_IJMatrixInsertRowISIS(matrix, n, rows[i], &cols[in], &coeffs[in]);
   }
   return ierr;
}
/******************************************************************************
 *
 * hypre_IJMatrixAddToBlockISIS
 *
 * adds a block of values to an IJMatrix, currently it just uses
 * AddIJMatrixRowISIS
 *
 *****************************************************************************/

NALU_HYPRE_Int
hypre_IJMatrixAddToBlockISIS(hypre_IJMatrix *matrix,
                             NALU_HYPRE_Int         m,
                             NALU_HYPRE_Int          n,
                             NALU_HYPRE_BigInt         *rows,
                             NALU_HYPRE_BigInt         *cols,
                             NALU_HYPRE_Complex        *coeffs)
{
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int i, in;
   for (i = 0; i < m; i++)
   {
      in = i * n;
      hypre_IJMatrixAddToRowISIS(matrix, n, rows[i], &cols[in], &coeffs[in]);
   }
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixInsertRowISIS
 *
 * inserts a row into an IJMatrix,
 * if diag_i and offd_i are known, those values are inserted directly
 * into the ParCSRMatrix,
 * if they are not known, an auxiliary structure, AuxParCSRMatrix is used
 *
 *****************************************************************************/
NALU_HYPRE_Int
hypre_IJMatrixInsertRowISIS(hypre_IJMatrix *matrix,
                            NALU_HYPRE_Int       n,
                            NALU_HYPRE_BigInt    row,
                            NALU_HYPRE_BigInt   *indices,
                            NALU_HYPRE_Complex  *coeffs)
{
   NALU_HYPRE_Int ierr = 0;
   hypre_ParCSRMatrix *par_matrix;
   hypre_AuxParCSRMatrix *aux_matrix;
   NALU_HYPRE_BigInt *row_starts;
   NALU_HYPRE_BigInt *col_starts;
   MPI_Comm comm = hypre_IJMatrixContext(matrix);
   NALU_HYPRE_Int num_procs, my_id;
   NALU_HYPRE_Int row_local;
   NALU_HYPRE_BigInt col_0, col_n;
   NALU_HYPRE_Int i, temp;
   NALU_HYPRE_Int *indx_diag, *indx_offd;
   NALU_HYPRE_BigInt **aux_j;
   NALU_HYPRE_BigInt *local_j;
   NALU_HYPRE_Complex **aux_data;
   NALU_HYPRE_Complex *local_data;
   NALU_HYPRE_Int diag_space, offd_space;
   NALU_HYPRE_Int *row_length, *row_space;
   NALU_HYPRE_Int need_aux;
   NALU_HYPRE_Int indx_0;
   NALU_HYPRE_Int diag_indx, offd_indx;

   hypre_CSRMatrix *diag;
   NALU_HYPRE_Int *diag_i;
   NALU_HYPRE_Int *diag_j;
   NALU_HYPRE_Complex *diag_data;

   hypre_CSRMatrix *offd;
   NALU_HYPRE_Int *offd_i;
   NALU_HYPRE_BigInt *big_offd_j;
   NALU_HYPRE_Complex *offd_data;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   par_matrix = hypre_IJMatrixLocalStorage( matrix );
   aux_matrix = hypre_IJMatrixTranslator(matrix);
   row_space = hypre_AuxParCSRMatrixRowSpace(aux_matrix);
   row_length = hypre_AuxParCSRMatrixRowLength(aux_matrix);
   col_n = hypre_ParCSRMatrixFirstColDiag(par_matrix);
   row_starts = hypre_ParCSRMatrixRowStarts(par_matrix);
   col_starts = hypre_ParCSRMatrixColStarts(par_matrix);
   col_0 = col_starts[my_id];
   col_n = col_starts[my_id + 1] - 1;
   need_aux = hypre_AuxParCSRMatrixNeedAux(aux_matrix);

   if (row >= row_starts[my_id] && row < row_starts[my_id + 1])
   {
      if (need_aux)
      {
         row_local = (NALU_HYPRE_Int)(row - row_starts[my_id]); /* compute local row number */
         aux_j = hypre_AuxParCSRMatrixAuxJ(aux_matrix);
         aux_data = hypre_AuxParCSRMatrixAuxData(aux_matrix);
         local_j = aux_j[row_local];
         local_data = aux_data[row_local];

         row_length[row_local] = n;

         if ( row_space[row_local] < n)
         {
            hypre_TFree(local_j, NALU_HYPRE_MEMORY_HOST);
            hypre_TFree(local_data, NALU_HYPRE_MEMORY_HOST);
            local_j = hypre_CTAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
            local_data = hypre_CTAlloc(NALU_HYPRE_Complex, n, NALU_HYPRE_MEMORY_HOST);
            row_space[row_local] = n;
         }

         for (i = 0; i < n; i++)
         {
            local_j[i] = indices[i];
            local_data[i] = coeffs[i];
         }

         /* make sure first element is diagonal element, if not, find it and
            exchange it with first element */
         if (local_j[0] != row_local)
         {
            for (i = 1; i < n; i++)
            {
               if (local_j[i] == row_local)
               {
                  local_j[i] = local_j[0];
                  local_j[0] = (NALU_HYPRE_BigInt)row_local;
                  temp = local_data[0];
                  local_data[0] = local_data[i];
                  local_data[i] = temp;
                  break;
               }
            }
         }
         /* sort data according to column indices, except for first element */

         BigQsort1(local_j, local_data, 1, n - 1);

      }
      else /* insert immediately into data into ParCSRMatrix structure */
      {
         diag = hypre_ParCSRMatrixDiag(par_matrix);
         offd = hypre_ParCSRMatrixOffd(par_matrix);
         diag_i = hypre_CSRMatrixI(diag);
         diag_j = hypre_CSRMatrixJ(diag);
         diag_data = hypre_CSRMatrixData(diag);
         offd_i = hypre_CSRMatrixI(offd);
         big_offd_j = hypre_CSRMatrixBigJ(offd);
         offd_data = hypre_CSRMatrixData(offd);
         offd_indx = offd_i[row_local];
         indx_0 = diag_i[row_local];
         diag_indx = indx_0 + 1;

         for (i = 0; i < n; i++)
         {
            if (indices[i] < col_0 || indices[i] > col_n)/* insert into offd */
            {
               big_offd_j[offd_indx] = indices[i];
               offd_data[offd_indx++] = coeffs[i];
            }
            else if (indices[i] == row) /* diagonal element */
            {
               diag_j[indx_0] = (NALU_HYPRE_Int)(indices[i] - col_0);
               diag_data[indx_0] = coeffs[i];
            }
            else  /* insert into diag */
            {
               diag_j[diag_indx] = (NALU_HYPRE_Int)(indices[i] - col_0);
               diag_data[diag_indx++] = coeffs[i];
            }
         }
         BigQsort1(big_offd_j, offd_data, 0, offd_indx - 1);
         qsort1(diag_j, diag_data, 1, diag_indx - 1);

         hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local] = diag_indx;
         hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = offd_indx;
      }
   }
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixAddToRowISIS
 *
 * adds a row to an IJMatrix before assembly,
 *
 *****************************************************************************/
NALU_HYPRE_Int
hypre_IJMatrixAddToRowISIS(hypre_IJMatrix *matrix,
                           NALU_HYPRE_Int         n,
                           NALU_HYPRE_BigInt      row,
                           NALU_HYPRE_BigInt     *indices,
                           NALU_HYPRE_Complex  *coeffs)
{
   NALU_HYPRE_Int ierr = 0;
   hypre_ParCSRMatrix *par_matrix;
   hypre_CSRMatrix *diag, *offd;
   hypre_AuxParCSRMatrix *aux_matrix;
   NALU_HYPRE_BigInt *row_starts;
   NALU_HYPRE_BigInt *col_starts;
   MPI_Comm comm = hypre_IJMatrixContext(matrix);
   NALU_HYPRE_Int num_procs, my_id;
   NALU_HYPRE_Int row_local;
   NALU_HYPRE_BigInt col_0, col_n;
   NALU_HYPRE_Int i, temp;
   NALU_HYPRE_Int *indx_diag, *indx_offd;
   NALU_HYPRE_BigInt **aux_j;
   NALU_HYPRE_BigInt *local_j;
   NALU_HYPRE_BigInt *tmp_j, *tmp2_j;
   NALU_HYPRE_Complex **aux_data;
   NALU_HYPRE_Complex *local_data;
   NALU_HYPRE_Complex *tmp_data, *tmp2_data;
   NALU_HYPRE_Int diag_space, offd_space;
   NALU_HYPRE_Int *row_length, *row_space;
   NALU_HYPRE_Int need_aux;
   NALU_HYPRE_Int tmp_indx, indx;
   NALU_HYPRE_Int size, old_size;
   NALU_HYPRE_Int cnt, cnt_diag, cnt_offd, indx_0;
   NALU_HYPRE_Int offd_indx, diag_indx;
   NALU_HYPRE_Int *diag_i;
   NALU_HYPRE_Int *diag_j;
   NALU_HYPRE_Complex *diag_data;
   NALU_HYPRE_Int *offd_i;
   NALU_HYPRE_BigInt *big_offd_j;
   NALU_HYPRE_Complex *offd_data;
   NALU_HYPRE_Int *tmp_diag_i;
   NALU_HYPRE_Int *tmp_diag_j;
   NALU_HYPRE_Complex *tmp_diag_data;
   NALU_HYPRE_Int *tmp_offd_i;
   NALU_HYPRE_BigInt *tmp_offd_j;
   NALU_HYPRE_Complex *tmp_offd_data;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   par_matrix = hypre_IJMatrixLocalStorage( matrix );
   aux_matrix = hypre_IJMatrixTranslator(matrix);
   row_space = hypre_AuxParCSRMatrixRowSpace(aux_matrix);
   row_length = hypre_AuxParCSRMatrixRowLength(aux_matrix);
   row_starts = hypre_ParCSRMatrixRowStarts(par_matrix);
   col_starts = hypre_ParCSRMatrixColStarts(par_matrix);
   col_0 = col_starts[my_id];
   col_n = col_starts[my_id + 1] - 1;
   need_aux = hypre_AuxParCSRMatrixNeedAux(aux_matrix);

   if (row >= row_starts[my_id] && row < row_starts[my_id + 1])
   {
      if (need_aux)
      {
         row_local = (NALU_HYPRE_Int)(row - row_starts[my_id]); /* compute local row number */
         aux_j = hypre_AuxParCSRMatrixAuxJ(aux_matrix);
         aux_data = hypre_AuxParCSRMatrixAuxData(aux_matrix);
         local_j = aux_j[row_local];
         local_data = aux_data[row_local];
         tmp_j = hypre_CTAlloc(NALU_HYPRE_BigInt, n, NALU_HYPRE_MEMORY_HOST);
         tmp_data = hypre_CTAlloc(NALU_HYPRE_Complex, n, NALU_HYPRE_MEMORY_HOST);
         tmp_indx = 0;
         for (i = 0; i < n; i++)
         {
            if (indices[i] == row)
            {
               local_data[0] += coeffs[i];
            }
            else
            {
               tmp_j[tmp_indx] = indices[i];
               tmp_data[tmp_indx++] = coeffs[i];
            }
         }
         BigQsort1(tmp_j, tmp_data, 0, tmp_indx - 1);
         indx = 0;
         size = 0;
         for (i = 1; i < row_length[row_local]; i++)
         {
            while (local_j[i] > tmp_j[indx])
            {
               size++;
               indx++;
            }
            if (local_j[i] == tmp_j[indx])
            {
               size++;
               indx++;
            }
         }
         size += tmp_indx - indx;

         old_size = row_length[row_local];
         row_length[row_local] = size;

         if ( row_space[row_local] < size)
         {
            tmp2_j = hypre_CTAlloc(NALU_HYPRE_BigInt, size, NALU_HYPRE_MEMORY_HOST);
            tmp2_data = hypre_CTAlloc(NALU_HYPRE_Complex, size, NALU_HYPRE_MEMORY_HOST);
            for (i = 0; i < old_size; i++)
            {
               tmp2_j[i] = local_j[i];
               tmp2_data[i] = local_data[i];
            }
            hypre_TFree(local_j, NALU_HYPRE_MEMORY_HOST);
            hypre_TFree(local_data, NALU_HYPRE_MEMORY_HOST);
            local_j = tmp2_j;
            local_data = tmp2_data;
            row_space[row_local] = n;
         }
         /* merge local and tmp into local */

         indx = 0;
         cnt = row_length[row_local];

         for (i = 1; i < old_size; i++)
         {
            while (local_j[i] > tmp_j[indx])
            {
               local_j[cnt] = tmp_j[indx];
               local_data[cnt++] = tmp_data[indx++];
            }
            if (local_j[i] == tmp_j[indx])
            {
               local_j[i] += tmp_j[indx];
               local_data[i] += tmp_data[indx++];
            }
         }
         for (i = indx; i < tmp_indx; i++)
         {
            local_j[cnt] = tmp_j[i];
            local_data[cnt++] = tmp_data[i];
         }

         /* sort data according to column indices, except for first element */

         BigQsort1(local_j, local_data, 1, n - 1);
         hypre_TFree(tmp_j, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(tmp_data, NALU_HYPRE_MEMORY_HOST);
      }
      else /* insert immediately into data into ParCSRMatrix structure */
      {
         offd_indx = hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
         diag_indx = hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
         diag = hypre_ParCSRMatrixDiag(par_matrix);
         diag_i = hypre_CSRMatrixI(diag);
         diag_j = hypre_CSRMatrixJ(diag);
         diag_data = hypre_CSRMatrixData(diag);
         offd = hypre_ParCSRMatrixOffd(par_matrix);
         offd_i = hypre_CSRMatrixI(offd);
         big_offd_j = hypre_CSRMatrixBigJ(offd);
         offd_data = hypre_CSRMatrixData(offd);

         indx_0 = diag_i[row_local];
         diag_indx = indx_0 + 1;

         tmp_diag_j = hypre_CTAlloc(NALU_HYPRE_Int, n, NALU_HYPRE_MEMORY_HOST);
         tmp_diag_data = hypre_CTAlloc(NALU_HYPRE_Complex, n, NALU_HYPRE_MEMORY_HOST);
         cnt_diag = 0;
         tmp_offd_j = hypre_CTAlloc(NALU_HYPRE_BigInt, n, NALU_HYPRE_MEMORY_HOST);
         tmp_offd_data = hypre_CTAlloc(NALU_HYPRE_Complex, n, NALU_HYPRE_MEMORY_HOST);
         cnt_offd = 0;
         for (i = 0; i < n; i++)
         {
            if (indices[i] < col_0 || indices[i] > col_n)/* insert into offd */
            {
               tmp_offd_j[cnt_offd] = indices[i];
               tmp_offd_data[cnt_offd++] = coeffs[i];
            }
            else if (indices[i] == row) /* diagonal element */
            {
               diag_j[indx_0] = (NALU_HYPRE_Int)(indices[i] - col_0);
               diag_data[indx_0] += coeffs[i];
            }
            else  /* insert into diag */
            {
               tmp_diag_j[cnt_diag] = (NALU_HYPRE_Int)(indices[i] - col_0);
               tmp_diag_data[cnt_diag++] = coeffs[i];
            }
         }
         qsort1(tmp_diag_j, tmp_diag_data, 0, cnt_diag - 1);
         BigQsort1(tmp_offd_j, tmp_offd_data, 0, cnt_offd - 1);

         diag_indx = hypre_AuxParCSRMatrixIndxDiag(aux_matrix)[row_local];
         cnt = diag_indx;
         indx = 0;
         for (i = diag_i[row_local] + 1; i < diag_indx; i++)
         {
            while (diag_j[i] > tmp_diag_j[indx])
            {
               diag_j[cnt] = tmp_diag_j[indx];
               diag_data[cnt++] = tmp_diag_data[indx++];
            }
            if (diag_j[i] == tmp_diag_j[indx])
            {
               diag_j[i] += tmp_diag_j[indx];
               diag_data[i] += tmp_diag_data[indx++];
            }
         }
         for (i = indx; i < cnt_diag; i++)
         {
            diag_j[cnt] = tmp_diag_j[i];
            diag_data[cnt++] = tmp_diag_data[i];
         }

         /* sort data according to column indices, except for first element */

         qsort1(diag_j, diag_data, 1, cnt - 1);
         hypre_TFree(tmp_diag_j, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(tmp_diag_data, NALU_HYPRE_MEMORY_HOST);

         hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt;

         offd_indx = hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local];
         cnt = offd_indx;
         indx = 0;
         for (i = offd_i[row_local] + 1; i < offd_indx; i++)
         {
            while (big_offd_j[i] > tmp_offd_j[indx])
            {
               big_offd_j[cnt] = tmp_offd_j[indx];
               offd_data[cnt++] = tmp_offd_data[indx++];
            }
            if (big_offd_j[i] == tmp_offd_j[indx])
            {
               big_offd_j[i] += tmp_offd_j[indx];
               offd_data[i] += tmp_offd_data[indx++];
            }
         }
         for (i = indx; i < cnt_offd; i++)
         {
            big_offd_j[cnt] = tmp_offd_j[i];
            offd_data[cnt++] = tmp_offd_data[i];
         }

         /* sort data according to column indices, except for first element */

         BigQsort1(big_offd_j, offd_data, 1, cnt - 1);
         hypre_TFree(tmp_offd_j, NALU_HYPRE_MEMORY_HOST);
         hypre_TFree(tmp_offd_data, NALU_HYPRE_MEMORY_HOST);

         hypre_AuxParCSRMatrixIndxOffd(aux_matrix)[row_local] = cnt;
      }
   }
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixAssembleISIS
 *
 * assembles IJMAtrix from AuxParCSRMatrix auxiliary structure
 *****************************************************************************/
NALU_HYPRE_Int
hypre_IJMatrixAssembleISIS(hypre_IJMatrix *matrix)
{
   NALU_HYPRE_Int ierr = 0;
   MPI_Comm comm = hypre_IJMatrixContext(matrix);
   hypre_ParCSRMatrix *par_matrix = hypre_IJMatrixLocalStorage(matrix);
   hypre_AuxParCSRMatrix *aux_matrix = hypre_IJMatrixTranslator(matrix);
   hypre_CSRMatrix *diag;
   hypre_CSRMatrix *offd;
   NALU_HYPRE_Int *diag_i;
   NALU_HYPRE_Int *offd_i;
   NALU_HYPRE_Int *diag_j;
   NALU_HYPRE_Int *offd_j;
   NALU_HYPRE_BigInt *big_offd_j;
   NALU_HYPRE_Complex *diag_data;
   NALU_HYPRE_Complex *offd_data;
   NALU_HYPRE_BigInt *row_starts = hypre_ParCSRMatrixRowStarts(par_matrix);
   NALU_HYPRE_BigInt *col_starts = hypre_ParCSRMatrixColStarts(par_matrix);
   NALU_HYPRE_Int j_indx, cnt, i, j;
   NALU_HYPRE_Int num_cols_offd;
   NALU_HYPRE_BigInt *col_map_offd;
   NALU_HYPRE_Int *row_length;
   NALU_HYPRE_Int *row_space;
   NALU_HYPRE_BigInt **aux_j;
   NALU_HYPRE_Complex **aux_data;
   NALU_HYPRE_Int *indx_diag;
   NALU_HYPRE_Int *indx_offd;
   NALU_HYPRE_Int need_aux = hypre_AuxParCSRMatrixNeedAux(aux_matrix);
   NALU_HYPRE_Int my_id, num_procs;
   NALU_HYPRE_Int num_rows;
   NALU_HYPRE_Int i_diag, i_offd;
   NALU_HYPRE_BigInt *local_j;
   NALU_HYPRE_Complex *local_data;
   NALU_HYPRE_BigInt col_0, col_n;
   NALU_HYPRE_Int nnz_offd;
   NALU_HYPRE_BigInt *aux_offd_j;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   num_rows = row_starts[my_id + 1] - row_starts[my_id];
   /* move data into ParCSRMatrix if not there already */
   if (need_aux)
   {
      col_0 = col_starts[my_id];
      col_n = col_starts[my_id + 1] - 1;
      i_diag = 0;
      i_offd = 0;
      for (i = 0; i < num_rows; i++)
      {
         local_j = aux_j[i];
         local_data = aux_data[i];
         for (j = 0; j < row_length[i]; j++)
         {
            if (local_j[j] < col_0 || local_j[j] > col_n)
            {
               i_offd++;
            }
            else
            {
               i_diag++;
            }
         }
         diag_i[i] = i_diag;
         offd_i[i] = i_offd;
      }
      diag_j = hypre_CTAlloc(NALU_HYPRE_Int, i_diag, NALU_HYPRE_MEMORY_HOST);
      diag_data = hypre_CTAlloc(NALU_HYPRE_Complex, i_diag, NALU_HYPRE_MEMORY_HOST);
      offd_j = hypre_CTAlloc(NALU_HYPRE_BigInt, i_offd, NALU_HYPRE_MEMORY_HOST);
      big_offd_j = hypre_CTAlloc(NALU_HYPRE_BigInt, i_offd, NALU_HYPRE_MEMORY_HOST);
      offd_data = hypre_CTAlloc(NALU_HYPRE_Complex, i_offd, NALU_HYPRE_MEMORY_HOST);
      i_diag = 0;
      i_offd = 0;
      for (i = 0; i < num_rows; i++)
      {
         local_j = aux_j[i];
         local_data = aux_data[i];
         for (j = 0; j < row_length[i]; j++)
         {
            if (local_j[j] < col_0 || local_j[j] > col_n)
            {
               big_offd_j[i_offd] = local_j[j];
               offd_data[i_offd++] = local_data[j];
            }
            else
            {
               diag_j[i_diag] = local_j[j];
               diag_data[i_diag++] = local_data[j];
            }
         }
      }
      hypre_CSRMatrixJ(diag) = diag_j;
      hypre_CSRMatrixData(diag) = diag_data;
      hypre_CSRMatrixNumNonzeros(diag) = diag_i[num_rows];
      hypre_CSRMatrixBigJ(offd) = big_offd_j;
      hypre_CSRMatrixJ(offd) = offd_j;
      hypre_CSRMatrixData(offd) = offd_data;
      hypre_CSRMatrixNumNonzeros(offd) = offd_i[num_rows];
   }

   /*  generate col_map_offd */
   nnz_offd = offd_i[num_rows];
   aux_offd_j = hypre_CTAlloc(NALU_HYPRE_BigInt,  nnz_offd, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nnz_offd; i++)
   {
      aux_offd_j[i] = big_offd_j[i];
   }
   BigQsort0(aux_offd_j, 0, nnz_offd - 1);
   num_cols_offd = 1;
   cnt = 0;
   for (i = 0; i < nnz_offd - 1; i++)
   {
      if (aux_offd_j[i + 1] > aux_offd_j[i])
      {
         cnt++;
         aux_offd_j[cnt] = aux_offd_j[i + 1];
         num_cols_offd++;
      }
   }
   col_map_offd = hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_cols_offd; i++)
   {
      col_map_offd[i] = aux_offd_j[i];
   }

   for (i = 0; i < nnz_offd; i++)
   {
      offd_j[i] = hypre_BigBinarySearch(col_map_offd, big_offd_j[i], num_cols_offd);
   }
   hypre_ParCSRMatrixColMapOffd(par_matrix) = col_map_offd;
   hypre_CSRMatrixNumCols(offd) = num_cols_offd;

   hypre_AuxParCSRMatrixDestroy(aux_matrix);
   hypre_TFree(aux_offd_j, NALU_HYPRE_MEMORY_HOST);
   hypre_TFree(big_offd_j, NALU_HYPRE_MEMORY_HOST);
   hypre_CSRMatrixBigJ(offd) = NULL;

   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixDistributeISIS
 *
 * takes an IJMatrix generated for one processor and distributes it
 * across many processors according to row_starts and col_starts,
 * if row_starts and/or col_starts NULL, it distributes them evenly.
 *
 *****************************************************************************/
NALU_HYPRE_Int
hypre_IJMatrixDistributeISIS(hypre_IJMatrix *matrix,
                             NALU_HYPRE_BigInt   *row_starts,
                             NALU_HYPRE_BigInt   *col_starts)
{
   NALU_HYPRE_Int ierr = 0;
   hypre_ParCSRMatrix *old_matrix = hypre_IJMatrixLocalStorage(matrix);
   hypre_ParCSRMatrix *par_matrix;
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(old_matrix);
   par_matrix = hypre_CSRMatrixToParCSRMatrix(hypre_ParCSRMatrixComm(old_matrix)
                                              , diag, row_starts, col_starts);
   ierr = hypre_ParCSRMatrixDestroy(old_matrix);
   hypre_IJMatrixLocalStorage(matrix) = par_matrix;
   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixApplyISIS
 *
 * NOT IMPLEMENTED YET
 *
 *****************************************************************************/
NALU_HYPRE_Int
hypre_IJMatrixApplyISIS(hypre_IJMatrix  *matrix,
                        hypre_ParVector *x,
                        hypre_ParVector *b)
{
   NALU_HYPRE_Int ierr = 0;

   return ierr;
}

/******************************************************************************
 *
 * hypre_IJMatrixDestroyISIS
 *
 * frees an IJMatrix
 *
 *****************************************************************************/
NALU_HYPRE_Int
hypre_IJMatrixDestroyISIS(hypre_IJMatrix *matrix)
{
   return hypre_ParCSRMatrixDestroy(hypre_IJMatrixLocalStorage(matrix));
}

/******************************************************************************
 *
 * hypre_IJMatrixSetTotalSizeISIS
 *
 * sets the total number of nonzeros of matrix, can be somewhat useful
 * for storage estimates
 *
 *****************************************************************************/
NALU_HYPRE_Int
hypre_IJMatrixSetTotalSizeISIS(hypre_IJMatrix *matrix,
                               NALU_HYPRE_Int         size)
{
   NALU_HYPRE_Int ierr = 0;
   hypre_ParCSRMatrix *par_matrix;
   par_matrix = hypre_IJMatrixLocalStorage(matrix);
   if (!par_matrix)
   {
      ierr = hypre_IJMatrixCreateISIS(matrix);
      par_matrix = hypre_IJMatrixLocalStorage(matrix);
   }
   hypre_ParCSRMatrixNumNonzeros(par_matrix) = size;
   return ierr;
}

