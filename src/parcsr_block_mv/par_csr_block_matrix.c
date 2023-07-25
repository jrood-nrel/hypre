/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_ParCSRBlockMatrix class.
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_block_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBlockMatrixCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRBlockMatrix *
nalu_hypre_ParCSRBlockMatrixCreate( MPI_Comm      comm,
                               NALU_HYPRE_Int     block_size,
                               NALU_HYPRE_BigInt  global_num_rows,
                               NALU_HYPRE_BigInt  global_num_cols,
                               NALU_HYPRE_BigInt *row_starts_in,
                               NALU_HYPRE_BigInt *col_starts_in,
                               NALU_HYPRE_Int     num_cols_offd,
                               NALU_HYPRE_Int     num_nonzeros_diag,
                               NALU_HYPRE_Int     num_nonzeros_offd )
{
   nalu_hypre_ParCSRBlockMatrix  *matrix;
   NALU_HYPRE_Int       num_procs, my_id;
   NALU_HYPRE_Int       local_num_rows;
   NALU_HYPRE_Int       local_num_cols;
   NALU_HYPRE_BigInt    first_row_index, first_col_diag;
   NALU_HYPRE_BigInt    row_starts[2];
   NALU_HYPRE_BigInt    col_starts[2];

   matrix = nalu_hypre_CTAlloc(nalu_hypre_ParCSRBlockMatrix, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   if (!row_starts_in)
   {
      nalu_hypre_GenerateLocalPartitioning(global_num_rows, num_procs, my_id,
                                      row_starts);
   }
   else
   {
      row_starts[0] = row_starts_in[0];
      row_starts[1] = row_starts_in[1];
   }

   if (!col_starts_in)
   {
      nalu_hypre_GenerateLocalPartitioning(global_num_cols, num_procs, my_id,
                                      col_starts);
   }
   else
   {
      col_starts[0] = col_starts_in[0];
      col_starts[1] = col_starts_in[1];
   }

   /* row_starts[0] is start of local rows.
      row_starts[1] is start of next processor's rows */
   first_row_index = row_starts[0];
   local_num_rows = (NALU_HYPRE_Int)(row_starts[1] - first_row_index) ;
   first_col_diag = col_starts[0];
   local_num_cols = (NALU_HYPRE_Int)(col_starts[1] - first_col_diag);
   nalu_hypre_ParCSRBlockMatrixComm(matrix) = comm;
   nalu_hypre_ParCSRBlockMatrixDiag(matrix) =
      nalu_hypre_CSRBlockMatrixCreate(block_size, local_num_rows,
                                 local_num_cols, num_nonzeros_diag);
   nalu_hypre_ParCSRBlockMatrixOffd(matrix) =
      nalu_hypre_CSRBlockMatrixCreate(block_size, local_num_rows,
                                 num_cols_offd, num_nonzeros_offd);

   nalu_hypre_ParCSRBlockMatrixBlockSize(matrix)     = block_size;
   nalu_hypre_ParCSRBlockMatrixGlobalNumRows(matrix) = global_num_rows;
   nalu_hypre_ParCSRBlockMatrixGlobalNumCols(matrix) = global_num_cols;
   nalu_hypre_ParCSRBlockMatrixFirstRowIndex(matrix) = first_row_index;
   nalu_hypre_ParCSRBlockMatrixFirstColDiag(matrix)  = first_col_diag;
   nalu_hypre_ParCSRBlockMatrixLastRowIndex(matrix)  = first_row_index + (NALU_HYPRE_BigInt)local_num_rows - 1;
   nalu_hypre_ParCSRBlockMatrixLastColDiag(matrix)   = first_col_diag  + (NALU_HYPRE_BigInt)local_num_cols - 1;
   nalu_hypre_ParCSRBlockMatrixRowStarts(matrix)[0]  = row_starts[0];
   nalu_hypre_ParCSRBlockMatrixRowStarts(matrix)[1]  = row_starts[1];
   nalu_hypre_ParCSRBlockMatrixColStarts(matrix)[0]  = col_starts[0];
   nalu_hypre_ParCSRBlockMatrixColStarts(matrix)[1]  = col_starts[1];
   nalu_hypre_ParCSRBlockMatrixColMapOffd(matrix)    = NULL;
   nalu_hypre_ParCSRBlockMatrixCommPkg(matrix)       = NULL;
   nalu_hypre_ParCSRBlockMatrixCommPkgT(matrix)      = NULL;
   nalu_hypre_ParCSRBlockMatrixAssumedPartition(matrix) = NULL;

   /* set defaults */
   nalu_hypre_ParCSRBlockMatrixOwnsData(matrix) = 1;

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBlockMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixDestroy( nalu_hypre_ParCSRBlockMatrix *matrix )
{

   if (matrix)
   {
      if ( nalu_hypre_ParCSRBlockMatrixOwnsData(matrix) )
      {
         nalu_hypre_CSRBlockMatrixDestroy(nalu_hypre_ParCSRBlockMatrixDiag(matrix));
         nalu_hypre_CSRBlockMatrixDestroy(nalu_hypre_ParCSRBlockMatrixOffd(matrix));
         if (nalu_hypre_ParCSRBlockMatrixColMapOffd(matrix))
         {
            nalu_hypre_TFree(nalu_hypre_ParCSRBlockMatrixColMapOffd(matrix), NALU_HYPRE_MEMORY_HOST);
         }
         if (nalu_hypre_ParCSRBlockMatrixCommPkg(matrix))
         {
            nalu_hypre_MatvecCommPkgDestroy(nalu_hypre_ParCSRBlockMatrixCommPkg(matrix));
         }
         if (nalu_hypre_ParCSRBlockMatrixCommPkgT(matrix))
         {
            nalu_hypre_MatvecCommPkgDestroy(nalu_hypre_ParCSRBlockMatrixCommPkgT(matrix));
         }
      }

      if (nalu_hypre_ParCSRBlockMatrixAssumedPartition(matrix))
      {
         nalu_hypre_ParCSRBlockMatrixDestroyAssumedPartition(matrix);
      }

      nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBlockMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixInitialize( nalu_hypre_ParCSRBlockMatrix *matrix )
{
   NALU_HYPRE_Int  ierr = 0;

   nalu_hypre_CSRBlockMatrixInitialize(nalu_hypre_ParCSRBlockMatrixDiag(matrix));
   nalu_hypre_CSRBlockMatrixInitialize(nalu_hypre_ParCSRBlockMatrixOffd(matrix));
   nalu_hypre_ParCSRBlockMatrixColMapOffd(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,
                                                             nalu_hypre_CSRBlockMatrixNumCols(nalu_hypre_ParCSRBlockMatrixOffd(matrix)), NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBlockMatrixSetNumNonzeros
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixSetNumNonzeros( nalu_hypre_ParCSRBlockMatrix *matrix)
{
   MPI_Comm comm = nalu_hypre_ParCSRBlockMatrixComm(matrix);
   nalu_hypre_CSRBlockMatrix *diag = nalu_hypre_ParCSRBlockMatrixDiag(matrix);
   NALU_HYPRE_Int *diag_i = nalu_hypre_CSRBlockMatrixI(diag);
   nalu_hypre_CSRBlockMatrix *offd = nalu_hypre_ParCSRBlockMatrixOffd(matrix);
   NALU_HYPRE_Int *offd_i = nalu_hypre_CSRBlockMatrixI(offd);
   NALU_HYPRE_Int local_num_rows = nalu_hypre_CSRBlockMatrixNumRows(diag);
   NALU_HYPRE_BigInt total_num_nonzeros;
   NALU_HYPRE_BigInt local_num_nonzeros;
   NALU_HYPRE_Int ierr = 0;

   local_num_nonzeros = (NALU_HYPRE_BigInt)(diag_i[local_num_rows] + offd_i[local_num_rows]);
   nalu_hypre_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, NALU_HYPRE_MPI_BIG_INT,
                       nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRBlockMatrixNumNonzeros(matrix) = total_num_nonzeros;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBlockMatrixSetDNumNonzeros
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixSetDNumNonzeros( nalu_hypre_ParCSRBlockMatrix *matrix)
{
   MPI_Comm comm = nalu_hypre_ParCSRBlockMatrixComm(matrix);
   nalu_hypre_CSRBlockMatrix *diag = nalu_hypre_ParCSRBlockMatrixDiag(matrix);
   NALU_HYPRE_Int *diag_i = nalu_hypre_CSRBlockMatrixI(diag);
   nalu_hypre_CSRBlockMatrix *offd = nalu_hypre_ParCSRBlockMatrixOffd(matrix);
   NALU_HYPRE_Int *offd_i = nalu_hypre_CSRBlockMatrixI(offd);
   NALU_HYPRE_Int local_num_rows = nalu_hypre_CSRBlockMatrixNumRows(diag);
   NALU_HYPRE_Real total_num_nonzeros;
   NALU_HYPRE_Real local_num_nonzeros;
   NALU_HYPRE_Int ierr = 0;

   local_num_nonzeros = (NALU_HYPRE_Real) diag_i[local_num_rows] + (NALU_HYPRE_Real) offd_i[local_num_rows];
   nalu_hypre_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1,
                       NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRBlockMatrixDNumNonzeros(matrix) = total_num_nonzeros;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBlockMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixSetDataOwner( nalu_hypre_ParCSRBlockMatrix *matrix,
                                     NALU_HYPRE_Int              owns_data )
{
   NALU_HYPRE_Int    ierr = 0;

   nalu_hypre_ParCSRBlockMatrixOwnsData(matrix) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBlockMatrixCompress
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRMatrix *
nalu_hypre_ParCSRBlockMatrixCompress( nalu_hypre_ParCSRBlockMatrix *matrix )
{
   MPI_Comm comm = nalu_hypre_ParCSRBlockMatrixComm(matrix);
   nalu_hypre_CSRBlockMatrix *diag = nalu_hypre_ParCSRBlockMatrixDiag(matrix);
   nalu_hypre_CSRBlockMatrix *offd = nalu_hypre_ParCSRBlockMatrixOffd(matrix);
   NALU_HYPRE_BigInt global_num_rows = nalu_hypre_ParCSRBlockMatrixGlobalNumRows(matrix);
   NALU_HYPRE_BigInt global_num_cols = nalu_hypre_ParCSRBlockMatrixGlobalNumCols(matrix);
   NALU_HYPRE_BigInt *row_starts = nalu_hypre_ParCSRBlockMatrixRowStarts(matrix);
   NALU_HYPRE_BigInt *col_starts = nalu_hypre_ParCSRBlockMatrixColStarts(matrix);
   NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRBlockMatrixNumCols(offd);
   NALU_HYPRE_Int num_nonzeros_diag = nalu_hypre_CSRBlockMatrixNumNonzeros(diag);
   NALU_HYPRE_Int num_nonzeros_offd = nalu_hypre_CSRBlockMatrixNumNonzeros(offd);

   nalu_hypre_ParCSRMatrix *matrix_C;

   NALU_HYPRE_Int i;

   matrix_C = nalu_hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                       row_starts, col_starts, num_cols_offd, num_nonzeros_diag, num_nonzeros_offd);
   nalu_hypre_ParCSRMatrixInitialize(matrix_C);

   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(matrix_C));
   nalu_hypre_ParCSRMatrixDiag(matrix_C) = nalu_hypre_CSRBlockMatrixCompress(diag);
   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(matrix_C));
   nalu_hypre_ParCSRMatrixOffd(matrix_C) = nalu_hypre_CSRBlockMatrixCompress(offd);

   for (i = 0; i < num_cols_offd; i++)
      nalu_hypre_ParCSRMatrixColMapOffd(matrix_C)[i] =
         nalu_hypre_ParCSRBlockMatrixColMapOffd(matrix)[i];
   return matrix_C;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBlockMatrixConvertToParCSRMatrix
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRMatrix *
nalu_hypre_ParCSRBlockMatrixConvertToParCSRMatrix(nalu_hypre_ParCSRBlockMatrix *matrix)
{
   MPI_Comm comm = nalu_hypre_ParCSRBlockMatrixComm(matrix);
   nalu_hypre_CSRBlockMatrix *diag = nalu_hypre_ParCSRBlockMatrixDiag(matrix);
   nalu_hypre_CSRBlockMatrix *offd = nalu_hypre_ParCSRBlockMatrixOffd(matrix);
   NALU_HYPRE_Int block_size = nalu_hypre_ParCSRBlockMatrixBlockSize(matrix);
   NALU_HYPRE_BigInt global_num_rows = nalu_hypre_ParCSRBlockMatrixGlobalNumRows(matrix);
   NALU_HYPRE_BigInt global_num_cols = nalu_hypre_ParCSRBlockMatrixGlobalNumCols(matrix);
   NALU_HYPRE_BigInt *row_starts = nalu_hypre_ParCSRBlockMatrixRowStarts(matrix);
   NALU_HYPRE_BigInt *col_starts = nalu_hypre_ParCSRBlockMatrixColStarts(matrix);
   NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRBlockMatrixNumCols(offd);
   NALU_HYPRE_Int num_nonzeros_diag = nalu_hypre_CSRBlockMatrixNumNonzeros(diag);
   NALU_HYPRE_Int num_nonzeros_offd = nalu_hypre_CSRBlockMatrixNumNonzeros(offd);

   nalu_hypre_ParCSRMatrix *matrix_C;
   NALU_HYPRE_BigInt matrix_C_row_starts[2];
   NALU_HYPRE_BigInt matrix_C_col_starts[2];

   NALU_HYPRE_Int *counter, *new_j_map;
   NALU_HYPRE_Int size_j, size_map, index, new_num_cols, removed = 0;
   NALU_HYPRE_Int *offd_j;
   NALU_HYPRE_BigInt *col_map_offd, *new_col_map_offd;


   NALU_HYPRE_Int num_procs, i, j;

   nalu_hypre_CSRMatrix *diag_nozeros, *offd_nozeros;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   for (i = 0; i < 2; i++)
   {
      matrix_C_row_starts[i] = row_starts[i] * (NALU_HYPRE_BigInt)block_size;
      matrix_C_col_starts[i] = col_starts[i] * (NALU_HYPRE_BigInt)block_size;
   }

   matrix_C = nalu_hypre_ParCSRMatrixCreate(comm, global_num_rows * (NALU_HYPRE_BigInt)block_size,
                                       global_num_cols * (NALU_HYPRE_BigInt)block_size,
                                       matrix_C_row_starts,
                                       matrix_C_col_starts,
                                       num_cols_offd * block_size,
                                       num_nonzeros_diag * block_size * block_size,
                                       num_nonzeros_offd * block_size * block_size);
   nalu_hypre_ParCSRMatrixInitialize(matrix_C);

   /* DIAG */
   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(matrix_C));
   nalu_hypre_ParCSRMatrixDiag(matrix_C) =
      nalu_hypre_CSRBlockMatrixConvertToCSRMatrix(diag);

   /* AB - added to delete zeros */
   diag_nozeros = nalu_hypre_CSRMatrixDeleteZeros(
                     nalu_hypre_ParCSRMatrixDiag(matrix_C), 1e-14);
   if (diag_nozeros)
   {
      nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(matrix_C));
      nalu_hypre_ParCSRMatrixDiag(matrix_C) = diag_nozeros;
   }

   /* OFF-DIAG */
   nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(matrix_C));
   nalu_hypre_ParCSRMatrixOffd(matrix_C) =
      nalu_hypre_CSRBlockMatrixConvertToCSRMatrix(offd);

   /* AB - added to delete zeros - this just deletes from data and j arrays */
   offd_nozeros = nalu_hypre_CSRMatrixDeleteZeros(
                     nalu_hypre_ParCSRMatrixOffd(matrix_C), 1e-14);
   if (offd_nozeros)
   {
      nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(matrix_C));
      nalu_hypre_ParCSRMatrixOffd(matrix_C) = offd_nozeros;
      removed = 1;

   }

   /* now convert the col_map_offd */
   for (i = 0; i < num_cols_offd; i++)
      for (j = 0; j < block_size; j++)
         nalu_hypre_ParCSRMatrixColMapOffd(matrix_C)[i * block_size + j] =
            nalu_hypre_ParCSRBlockMatrixColMapOffd(matrix)[i] * (NALU_HYPRE_BigInt)block_size + (NALU_HYPRE_BigInt)j;

   /* if we deleted zeros, then it is possible that col_map_offd can be
      compressed as well - this requires some amount of work that could be skipped... */

   if (removed)
   {
      size_map =   num_cols_offd * block_size;
      counter = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size_map, NALU_HYPRE_MEMORY_HOST);
      new_j_map = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  size_map, NALU_HYPRE_MEMORY_HOST);

      offd_j = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(matrix_C));
      col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(matrix_C);

      size_j = nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(matrix_C));
      /* mark which off_d entries are found in j */
      for (i = 0; i < size_j; i++)
      {
         counter[offd_j[i]] = 1;
      }
      /*now find new numbering for columns (we will delete the
        cols where counter = 0*/
      index = 0;
      for (i = 0; i < size_map; i++)
      {
         if (counter[i]) { new_j_map[i] = index++; }
      }
      new_num_cols = index;
      /* if there are some col entries to remove: */
      if (!(index == size_map))
      {
         /* go thru j and adjust entries */
         for (i = 0; i < size_j; i++)
         {
            offd_j[i] = new_j_map[offd_j[i]];
         }
         /*now go thru col map and get rid of non-needed entries */
         new_col_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  new_num_cols, NALU_HYPRE_MEMORY_HOST);
         index = 0;
         for (i = 0; i < size_map; i++)
         {
            if (counter[i])
            {
               new_col_map_offd[index++] = col_map_offd[i];
            }
         }
         /* set the new col map */
         nalu_hypre_TFree(col_map_offd, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_ParCSRMatrixColMapOffd(matrix_C) = new_col_map_offd;
         /* modify the number of cols */
         nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(matrix_C)) = new_num_cols;
      }
      nalu_hypre_TFree(new_j_map, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(counter, NALU_HYPRE_MEMORY_HOST);

   }

   nalu_hypre_ParCSRMatrixSetNumNonzeros( matrix_C );
   nalu_hypre_ParCSRMatrixSetDNumNonzeros( matrix_C );

   /* we will not copy the comm package */
   nalu_hypre_ParCSRMatrixCommPkg(matrix_C) = NULL;

   return matrix_C;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBlockMatrixConvertFromParCSRMatrix
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRBlockMatrix *
nalu_hypre_ParCSRBlockMatrixConvertFromParCSRMatrix(nalu_hypre_ParCSRMatrix *matrix,
                                               NALU_HYPRE_Int matrix_C_block_size )
{
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(matrix);
   nalu_hypre_CSRMatrix *diag = nalu_hypre_ParCSRMatrixDiag(matrix);
   nalu_hypre_CSRMatrix *offd = nalu_hypre_ParCSRMatrixOffd(matrix);
   NALU_HYPRE_BigInt global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(matrix);
   NALU_HYPRE_BigInt global_num_cols = nalu_hypre_ParCSRMatrixGlobalNumCols(matrix);
   NALU_HYPRE_BigInt *row_starts = nalu_hypre_ParCSRMatrixRowStarts(matrix);
   NALU_HYPRE_BigInt *col_starts = nalu_hypre_ParCSRMatrixColStarts(matrix);
   NALU_HYPRE_Int num_cols_offd = nalu_hypre_CSRMatrixNumCols(offd);
   NALU_HYPRE_BigInt *col_map_offd = nalu_hypre_ParCSRBlockMatrixColMapOffd(matrix);
   NALU_HYPRE_BigInt *map_to_node = NULL;
   NALU_HYPRE_Int *counter = NULL, *col_in_j_map = NULL;
   NALU_HYPRE_BigInt *matrix_C_col_map_offd = NULL;

   NALU_HYPRE_Int matrix_C_num_cols_offd;
   NALU_HYPRE_Int matrix_C_num_nonzeros_offd;
   NALU_HYPRE_Int num_rows, num_nodes;

   NALU_HYPRE_Int *offd_i        = nalu_hypre_CSRMatrixI(offd);
   NALU_HYPRE_Int *offd_j        = nalu_hypre_CSRMatrixJ(offd);
   NALU_HYPRE_Complex * offd_data = nalu_hypre_CSRMatrixData(offd);

   nalu_hypre_ParCSRBlockMatrix *matrix_C;
   NALU_HYPRE_BigInt matrix_C_row_starts[2];
   NALU_HYPRE_BigInt matrix_C_col_starts[2];
   nalu_hypre_CSRBlockMatrix *matrix_C_diag;
   nalu_hypre_CSRBlockMatrix *matrix_C_offd;

   NALU_HYPRE_Int *matrix_C_offd_i = NULL, *matrix_C_offd_j = NULL;
   NALU_HYPRE_Complex *matrix_C_offd_data = NULL;

   NALU_HYPRE_Int num_procs, i, j, k, k_map, count, index, start_index, pos, row;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   for (i = 0; i < 2; i++)
   {
      matrix_C_row_starts[i] = row_starts[i] / (NALU_HYPRE_BigInt)matrix_C_block_size;
      matrix_C_col_starts[i] = col_starts[i] / (NALU_HYPRE_BigInt)matrix_C_block_size;
   }

   /************* create the diagonal part ************/
   matrix_C_diag = nalu_hypre_CSRBlockMatrixConvertFromCSRMatrix(diag,
                                                            matrix_C_block_size);

   /*******  the offd part *******************/

   /* can't use the same function for the offd part - because this isn't square
      and the offd j entries aren't global numbering (have to consider the offd
      map) - need to look at col_map_offd first */

   /* figure out the new number of offd columns (num rows is same as diag) */
   num_cols_offd = nalu_hypre_CSRMatrixNumCols(offd);
   num_rows = nalu_hypre_CSRMatrixNumRows(diag);
   num_nodes =  num_rows / matrix_C_block_size;

   matrix_C_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_nodes + 1, NALU_HYPRE_MEMORY_HOST);

   matrix_C_num_cols_offd = 0;
   matrix_C_offd_i[0] = 0;
   matrix_C_num_nonzeros_offd = 0;

   if (num_cols_offd)
   {
      map_to_node = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      matrix_C_num_cols_offd = 1;
      map_to_node[0] = col_map_offd[0] / (NALU_HYPRE_BigInt)matrix_C_block_size;
      for (i = 1; i < num_cols_offd; i++)
      {
         map_to_node[i] = col_map_offd[i] / (NALU_HYPRE_BigInt)matrix_C_block_size;
         if (map_to_node[i] > map_to_node[i - 1]) { matrix_C_num_cols_offd++; }
      }

      matrix_C_col_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  matrix_C_num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      col_in_j_map = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);

      matrix_C_col_map_offd[0] = map_to_node[0];
      col_in_j_map[0] = 0;
      count = 1;
      j = 1;

      /* fill in the col_map_off_d - these are global numbers.  Then we need to
         map these to j entries (these have local numbers) */
      for (i = 1; i < num_cols_offd; i++)
      {
         if (map_to_node[i] > map_to_node[i - 1])
         {
            matrix_C_col_map_offd[count++] = map_to_node[i];
         }
         col_in_j_map[j++] = count - 1;
      }

      /* now figure the nonzeros */
      matrix_C_num_nonzeros_offd = 0;
      counter = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  matrix_C_num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < matrix_C_num_cols_offd; i++)
      {
         counter[i] = -1;
      }

      for (i = 0; i < num_nodes; i++) /* for each block row */
      {
         matrix_C_offd_i[i] = matrix_C_num_nonzeros_offd;
         for (j = 0; j < matrix_C_block_size; j++)
         {
            row = i * matrix_C_block_size + j;
            for (k = offd_i[row]; k < offd_i[row + 1]; k++) /* go through single row */
            {
               k_map = col_in_j_map[offd_j[k]]; /*nodal col - see if this has
                                                  been in this block row (i)
                                                  already*/

               if (counter[k_map] < i) /* not yet counted for this nodal row */
               {
                  counter[k_map] = i;
                  matrix_C_num_nonzeros_offd++;
               }
            }
         }
      }
      /* fill in final i entry */
      matrix_C_offd_i[num_nodes] = matrix_C_num_nonzeros_offd;
   }

   /* create offd matrix */
   matrix_C_offd = nalu_hypre_CSRBlockMatrixCreate(matrix_C_block_size, num_nodes,
                                              matrix_C_num_cols_offd,
                                              matrix_C_num_nonzeros_offd);

   /* assign i */
   nalu_hypre_CSRBlockMatrixI(matrix_C_offd) = matrix_C_offd_i;


   /* create (and allocate j and data) */
   if (matrix_C_num_nonzeros_offd)
   {
      matrix_C_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  matrix_C_num_nonzeros_offd, NALU_HYPRE_MEMORY_HOST);
      matrix_C_offd_data =
         nalu_hypre_CTAlloc(NALU_HYPRE_Complex,
                       matrix_C_num_nonzeros_offd * matrix_C_block_size *
                       matrix_C_block_size, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_CSRBlockMatrixJ(matrix_C_offd) = matrix_C_offd_j;
      nalu_hypre_CSRMatrixData(matrix_C_offd) = matrix_C_offd_data;

      for (i = 0; i < matrix_C_num_cols_offd; i++)
      {
         counter[i] = -1;
      }

      index = 0; /*keep track of entry in matrix_C_offd_j*/
      start_index = 0;
      for (i = 0; i < num_nodes; i++) /* for each block row */
      {

         for (j = 0; j < matrix_C_block_size; j++) /* for each row in block */
         {
            row = i * matrix_C_block_size + j;
            for (k = offd_i[row]; k < offd_i[row + 1]; k++) /* go through single row's cols */
            {
               k_map = col_in_j_map[offd_j[k]]; /*nodal col  for off_d */
               if (counter[k_map] < start_index) /* not yet counted for this nodal row */
               {
                  counter[k_map] = index;
                  matrix_C_offd_j[index] = k_map;
                  /*copy the data: which position (corresponds to j array) + which row + which col */
                  pos =  (index * matrix_C_block_size * matrix_C_block_size) + (j * matrix_C_block_size) +
                         (NALU_HYPRE_Int)(col_map_offd[offd_j[k]] % (NALU_HYPRE_BigInt)matrix_C_block_size);
                  matrix_C_offd_data[pos] = offd_data[k];
                  index ++;
               }
               else  /* this col has already been listed for this row */
               {

                  /*copy the data: which position (corresponds to j array) + which row + which col */
                  pos =  (counter[k_map] * matrix_C_block_size * matrix_C_block_size) + (j * matrix_C_block_size) +
                         (NALU_HYPRE_Int)(col_map_offd[offd_j[k]] % (NALU_HYPRE_BigInt)(matrix_C_block_size));
                  matrix_C_offd_data[pos] = offd_data[k];
               }
            }
         }
         start_index = index; /* first index for current nodal row */
      }
   }

   /* *********create the new matrix  *************/
   matrix_C = nalu_hypre_ParCSRBlockMatrixCreate(comm, matrix_C_block_size,
                                            global_num_rows / (NALU_HYPRE_BigInt)matrix_C_block_size,
                                            global_num_cols / (NALU_HYPRE_BigInt)matrix_C_block_size,
                                            matrix_C_row_starts,
                                            matrix_C_col_starts,
                                            matrix_C_num_cols_offd,
                                            nalu_hypre_CSRBlockMatrixNumNonzeros(matrix_C_diag),
                                            matrix_C_num_nonzeros_offd);

   /* use the diag and off diag matrices we have already created */
   nalu_hypre_CSRBlockMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(matrix_C));
   nalu_hypre_ParCSRBlockMatrixDiag(matrix_C) = matrix_C_diag;
   nalu_hypre_CSRBlockMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(matrix_C));
   nalu_hypre_ParCSRBlockMatrixOffd(matrix_C) = matrix_C_offd;

   nalu_hypre_ParCSRMatrixColMapOffd(matrix_C) = matrix_C_col_map_offd;

   /* *********don't bother to copy the comm_pkg *************/

   nalu_hypre_ParCSRBlockMatrixCommPkg(matrix_C) = NULL;

   /* CLEAN UP !!!! */
   nalu_hypre_TFree(map_to_node, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(col_in_j_map, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(counter, NALU_HYPRE_MEMORY_HOST);

   return matrix_C;
}

/* ----------------------------------------------------------------------
 * nalu_hypre_BlockMatvecCommPkgCreate
 * ---------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BlockMatvecCommPkgCreate(nalu_hypre_ParCSRBlockMatrix *A)
{
   NALU_HYPRE_Int        num_recvs, *recv_procs, *recv_vec_starts;
   NALU_HYPRE_Int        num_sends, *send_procs, *send_map_starts;
   NALU_HYPRE_Int       *send_map_elmts;

   NALU_HYPRE_Int        num_cols_off_d;
   NALU_HYPRE_BigInt    *col_map_off_d;

   NALU_HYPRE_BigInt     first_col_diag;
   NALU_HYPRE_BigInt     global_num_cols;

   MPI_Comm         comm;

   nalu_hypre_ParCSRCommPkg   *comm_pkg = NULL;
   nalu_hypre_IJAssumedPart   *apart;

   /*-----------------------------------------------------------
    * get parcsr_A information
    *----------------------------------------------------------*/
   col_map_off_d =  nalu_hypre_ParCSRBlockMatrixColMapOffd(A);
   num_cols_off_d = nalu_hypre_CSRBlockMatrixNumCols(nalu_hypre_ParCSRBlockMatrixOffd(A));

   global_num_cols = nalu_hypre_ParCSRBlockMatrixGlobalNumCols(A);

   comm = nalu_hypre_ParCSRBlockMatrixComm(A);

   first_col_diag = nalu_hypre_ParCSRBlockMatrixFirstColDiag(A);

   /* Create the assumed partition */
   if (nalu_hypre_ParCSRBlockMatrixAssumedPartition(A) == NULL)
   {
      nalu_hypre_ParCSRBlockMatrixCreateAssumedPartition(A);
   }

   apart = nalu_hypre_ParCSRBlockMatrixAssumedPartition(A);

   /*-----------------------------------------------------------
    * get commpkg info information
    *----------------------------------------------------------*/

   nalu_hypre_ParCSRCommPkgCreateApart_core( comm, col_map_off_d, first_col_diag,
                                        num_cols_off_d, global_num_cols,
                                        &num_recvs, &recv_procs, &recv_vec_starts,
                                        &num_sends, &send_procs, &send_map_starts,
                                        &send_map_elmts, apart);

   if (!num_recvs)
   {
      nalu_hypre_TFree(recv_procs, NALU_HYPRE_MEMORY_HOST);
      recv_procs = NULL;
   }
   if (!num_sends)
   {
      nalu_hypre_TFree(send_procs, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(send_map_elmts, NALU_HYPRE_MEMORY_HOST);
      send_procs = NULL;
      send_map_elmts = NULL;
   }

   /*-----------------------------------------------------------
    * setup commpkg
    *----------------------------------------------------------*/

   nalu_hypre_ParCSRCommPkgCreateAndFill(comm,
                                    num_recvs, recv_procs, recv_vec_starts,
                                    num_sends, send_procs, send_map_starts,
                                    send_map_elmts,
                                    &comm_pkg);

   nalu_hypre_ParCSRBlockMatrixCommPkg(A) = comm_pkg;

   return nalu_hypre_error_flag;
}

/* ----------------------------------------------------------------------
 * nalu_hypre_ParCSRBlockMatrixExtractBExt: extracts rows from B which are located on
 * other processors and needed for multiplication with A locally. The rows
 * are returned as CSRBlockMatrix.
 * ---------------------------------------------------------------------*/

nalu_hypre_CSRBlockMatrix *
nalu_hypre_ParCSRBlockMatrixExtractBExt(nalu_hypre_ParCSRBlockMatrix *B,
                                   nalu_hypre_ParCSRBlockMatrix *A, NALU_HYPRE_Int data)
{
   MPI_Comm comm = nalu_hypre_ParCSRBlockMatrixComm(B);
   NALU_HYPRE_BigInt first_col_diag = nalu_hypre_ParCSRBlockMatrixFirstColDiag(B);
   NALU_HYPRE_BigInt *col_map_offd = nalu_hypre_ParCSRBlockMatrixColMapOffd(B);
   NALU_HYPRE_Int block_size = nalu_hypre_ParCSRBlockMatrixBlockSize(B);

   nalu_hypre_ParCSRCommPkg *comm_pkg = nalu_hypre_ParCSRBlockMatrixCommPkg(A);
   NALU_HYPRE_Int num_recvs = nalu_hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   NALU_HYPRE_Int *recv_vec_starts = nalu_hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
   NALU_HYPRE_Int num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   NALU_HYPRE_Int *send_map_starts = nalu_hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
   NALU_HYPRE_Int *send_map_elmts = nalu_hypre_ParCSRCommPkgSendMapElmts(comm_pkg);

   nalu_hypre_ParCSRCommHandle *comm_handle;
   nalu_hypre_ParCSRCommPkg *tmp_comm_pkg = NULL;

   nalu_hypre_CSRBlockMatrix *diag = nalu_hypre_ParCSRBlockMatrixDiag(B);

   NALU_HYPRE_Int *diag_i = nalu_hypre_CSRBlockMatrixI(diag);
   NALU_HYPRE_Int *diag_j = nalu_hypre_CSRBlockMatrixJ(diag);
   NALU_HYPRE_Complex *diag_data = nalu_hypre_CSRBlockMatrixData(diag);

   nalu_hypre_CSRBlockMatrix *offd = nalu_hypre_ParCSRBlockMatrixOffd(B);

   NALU_HYPRE_Int *offd_i = nalu_hypre_CSRBlockMatrixI(offd);
   NALU_HYPRE_Int *offd_j = nalu_hypre_CSRBlockMatrixJ(offd);
   NALU_HYPRE_Complex *offd_data = nalu_hypre_CSRBlockMatrixData(offd);

   NALU_HYPRE_Int *B_int_i;
   NALU_HYPRE_BigInt *B_int_j;
   NALU_HYPRE_Complex *B_int_data;

   NALU_HYPRE_Int num_cols_B, num_nonzeros;
   NALU_HYPRE_Int num_rows_B_ext;
   NALU_HYPRE_Int num_procs, my_id;

   nalu_hypre_CSRBlockMatrix *B_ext;

   NALU_HYPRE_Int *B_ext_i;
   NALU_HYPRE_BigInt *B_ext_j;
   NALU_HYPRE_Complex *B_ext_data;

   NALU_HYPRE_Int *jdata_recv_vec_starts;
   NALU_HYPRE_Int *jdata_send_map_starts;

   NALU_HYPRE_Int i, j, k, l, counter, bnnz;
   NALU_HYPRE_Int start_index;
   NALU_HYPRE_Int j_cnt, jrow;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   bnnz = block_size * block_size;
   num_cols_B = nalu_hypre_ParCSRMatrixGlobalNumCols(B);
   num_rows_B_ext = recv_vec_starts[num_recvs];
   B_int_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  send_map_starts[num_sends] + 1, NALU_HYPRE_MEMORY_HOST);
   B_ext_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_rows_B_ext + 1, NALU_HYPRE_MEMORY_HOST);
   /*--------------------------------------------------------------------------
    * generate B_int_i through adding number of row-elements of offd and diag
    * for corresponding rows. B_int_i[j+1] contains the number of elements of
    * a row j (which is determined through send_map_elmts)
    *--------------------------------------------------------------------------*/
   B_int_i[0] = 0;
   j_cnt = 0;
   num_nonzeros = 0;
   for (i = 0; i < num_sends; i++)
   {
      for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
      {
         jrow = send_map_elmts[j];
         B_int_i[++j_cnt] = offd_i[jrow + 1] - offd_i[jrow]
                            + diag_i[jrow + 1] - diag_i[jrow];
         num_nonzeros += B_int_i[j_cnt];
      }
   }

   /*--------------------------------------------------------------------------
    * initialize communication
    *--------------------------------------------------------------------------*/
   comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg,
                                              &B_int_i[1], &B_ext_i[1]);

   B_int_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_nonzeros, NALU_HYPRE_MEMORY_HOST);
   if (data) { B_int_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  num_nonzeros * bnnz, NALU_HYPRE_MEMORY_HOST); }

   jdata_send_map_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_sends + 1, NALU_HYPRE_MEMORY_HOST);
   jdata_recv_vec_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_recvs + 1, NALU_HYPRE_MEMORY_HOST);
   start_index = B_int_i[0];
   jdata_send_map_starts[0] = start_index;
   counter = 0;
   for (i = 0; i < num_sends; i++)
   {
      num_nonzeros = counter;
      for (j = send_map_starts[i]; j < send_map_starts[i + 1]; j++)
      {
         jrow = send_map_elmts[j];
         for (k = diag_i[jrow]; k < diag_i[jrow + 1]; k++)
         {
            B_int_j[counter] = (NALU_HYPRE_BigInt)diag_j[k] + first_col_diag;
            if (data)
            {
               for (l = 0; l < bnnz; l++)
               {
                  B_int_data[counter * bnnz + l] = diag_data[k * bnnz + l];
               }
            }
            counter++;
         }
         for (k = offd_i[jrow]; k < offd_i[jrow + 1]; k++)
         {
            B_int_j[counter] = col_map_offd[offd_j[k]];
            if (data)
            {
               for (l = 0; l < bnnz; l++)
                  B_int_data[counter * bnnz + l] =
                     offd_data[k * bnnz + l];
            }
            counter++;
         }
      }
      num_nonzeros = counter - num_nonzeros;
      start_index += num_nonzeros;
      jdata_send_map_starts[i + 1] = start_index;
   }

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
    * evaluate B_ext_i and compute num_nonzeros for B_ext
    *--------------------------------------------------------------------------*/

   for (i = 0; i < num_recvs; i++)
   {
      for (j = recv_vec_starts[i]; j < recv_vec_starts[i + 1]; j++)
      {
         B_ext_i[j + 1] += B_ext_i[j];
      }
   }

   num_nonzeros = B_ext_i[num_rows_B_ext];

   B_ext = nalu_hypre_CSRBlockMatrixCreate(block_size, num_rows_B_ext, num_cols_B,
                                      num_nonzeros);
   B_ext_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_nonzeros, NALU_HYPRE_MEMORY_HOST);
   if (data)
   {
      B_ext_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  num_nonzeros * bnnz, NALU_HYPRE_MEMORY_HOST);
   }

   for (i = 0; i < num_recvs; i++)
   {
      start_index = B_ext_i[recv_vec_starts[i]];
      num_nonzeros = B_ext_i[recv_vec_starts[i + 1]] - start_index;
      jdata_recv_vec_starts[i + 1] = B_ext_i[recv_vec_starts[i + 1]];
   }

   comm_handle = nalu_hypre_ParCSRCommHandleCreate(21, tmp_comm_pkg, B_int_j, B_ext_j);
   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   if (data)
   {
      comm_handle = nalu_hypre_ParCSRBlockCommHandleCreate(1, bnnz, tmp_comm_pkg,
                                                      B_int_data, B_ext_data);
      nalu_hypre_ParCSRBlockCommHandleDestroy(comm_handle);
      comm_handle = NULL;
   }

   nalu_hypre_CSRBlockMatrixI(B_ext) = B_ext_i;
   nalu_hypre_CSRBlockMatrixBigJ(B_ext) = B_ext_j;
   if (data)
   {
      nalu_hypre_CSRBlockMatrixData(B_ext) = B_ext_data;
   }

   /* Free memory */
   nalu_hypre_TFree(jdata_send_map_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(jdata_recv_vec_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(tmp_comm_pkg, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(B_int_i, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(B_int_j, NALU_HYPRE_MEMORY_HOST);
   if (data)
   {
      nalu_hypre_TFree(B_int_data, NALU_HYPRE_MEMORY_HOST);
   }

   return B_ext;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParVectorCreateFromBlock
 *--------------------------------------------------------------------------*/

nalu_hypre_ParVector *
nalu_hypre_ParVectorCreateFromBlock(  MPI_Comm comm,
                                 NALU_HYPRE_BigInt p_global_size,
                                 NALU_HYPRE_BigInt *p_partitioning, NALU_HYPRE_Int block_size)
{
   nalu_hypre_ParVector  *vector;
   NALU_HYPRE_Int num_procs, my_id;
   NALU_HYPRE_BigInt global_size;
   NALU_HYPRE_BigInt new_partitioning[2]; /* need to create a new partitioning - son't want to write over
                                     what is passed in */

   global_size = p_global_size * (NALU_HYPRE_BigInt)block_size;

   vector = nalu_hypre_CTAlloc(nalu_hypre_ParVector, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   if (!p_partitioning)
   {
      nalu_hypre_GenerateLocalPartitioning(global_size, num_procs, my_id, new_partitioning);
   }
   else /* adjust for block_size */
   {
      new_partitioning[0] = p_partitioning[0] * (NALU_HYPRE_BigInt)block_size;
      new_partitioning[1] = p_partitioning[1] * (NALU_HYPRE_BigInt)block_size;
   }

   nalu_hypre_ParVectorComm(vector) = comm;
   nalu_hypre_ParVectorGlobalSize(vector) = global_size;
   nalu_hypre_ParVectorFirstIndex(vector) = new_partitioning[0];
   nalu_hypre_ParVectorLastIndex(vector)  = new_partitioning[1] - 1;
   nalu_hypre_ParVectorPartitioning(vector)[0] = new_partitioning[0];
   nalu_hypre_ParVectorPartitioning(vector)[1] = new_partitioning[1];
   nalu_hypre_ParVectorLocalVector(vector) =
      nalu_hypre_SeqVectorCreate(new_partitioning[1] - new_partitioning[0]);

   /* set defaults */
   nalu_hypre_ParVectorOwnsData(vector) = 1;

   return vector;
}
