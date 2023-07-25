/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_CSRBooleanMatrix and nalu_hypre_ParCSRBooleanMatrix class.
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRBooleanMatrixCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRBooleanMatrix *nalu_hypre_CSRBooleanMatrixCreate(NALU_HYPRE_Int num_rows, NALU_HYPRE_Int num_cols,
                                                     NALU_HYPRE_Int num_nonzeros )
{
   nalu_hypre_CSRBooleanMatrix *matrix;

   matrix = nalu_hypre_CTAlloc(nalu_hypre_CSRBooleanMatrix,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_CSRBooleanMatrix_Get_I(matrix)     = NULL;
   nalu_hypre_CSRBooleanMatrix_Get_J(matrix)     = NULL;
   nalu_hypre_CSRBooleanMatrix_Get_BigJ(matrix)  = NULL;
   nalu_hypre_CSRBooleanMatrix_Get_NRows(matrix) = num_rows;
   nalu_hypre_CSRBooleanMatrix_Get_NCols(matrix) = num_cols;
   nalu_hypre_CSRBooleanMatrix_Get_NNZ(matrix)   = num_nonzeros;
   nalu_hypre_CSRBooleanMatrix_Get_OwnsData(matrix) = 1;

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRBooleanMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_CSRBooleanMatrixDestroy( nalu_hypre_CSRBooleanMatrix *matrix )
{
   if (matrix)
   {
      nalu_hypre_TFree(nalu_hypre_CSRBooleanMatrix_Get_I(matrix), NALU_HYPRE_MEMORY_HOST);
      if ( nalu_hypre_CSRBooleanMatrix_Get_OwnsData(matrix) )
      {
         nalu_hypre_TFree(nalu_hypre_CSRBooleanMatrix_Get_J(matrix), NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(nalu_hypre_CSRBooleanMatrix_Get_BigJ(matrix), NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRBooleanMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_CSRBooleanMatrixInitialize( nalu_hypre_CSRBooleanMatrix *matrix )
{
   NALU_HYPRE_Int  num_rows     = nalu_hypre_CSRBooleanMatrix_Get_NRows(matrix);
   NALU_HYPRE_Int  num_nonzeros = nalu_hypre_CSRBooleanMatrix_Get_NNZ(matrix);

   if ( ! nalu_hypre_CSRBooleanMatrix_Get_I(matrix) )
   {
      nalu_hypre_CSRBooleanMatrix_Get_I(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_rows + 1, NALU_HYPRE_MEMORY_HOST);
   }
   if ( ! nalu_hypre_CSRBooleanMatrix_Get_J(matrix) )
   {
      nalu_hypre_CSRBooleanMatrix_Get_J(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_nonzeros, NALU_HYPRE_MEMORY_HOST);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRBooleanMatrixBigInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_CSRBooleanMatrixBigInitialize( nalu_hypre_CSRBooleanMatrix *matrix )
{
   NALU_HYPRE_Int  num_rows     = nalu_hypre_CSRBooleanMatrix_Get_NRows(matrix);
   NALU_HYPRE_Int  num_nonzeros = nalu_hypre_CSRBooleanMatrix_Get_NNZ(matrix);

   if ( ! nalu_hypre_CSRBooleanMatrix_Get_I(matrix) )
   {
      nalu_hypre_CSRBooleanMatrix_Get_I(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_rows + 1, NALU_HYPRE_MEMORY_HOST);
   }
   if ( ! nalu_hypre_CSRBooleanMatrix_Get_BigJ(matrix) )
   {
      nalu_hypre_CSRBooleanMatrix_Get_BigJ(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_nonzeros,
                                                              NALU_HYPRE_MEMORY_HOST);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRBooleanMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_CSRBooleanMatrixSetDataOwner( nalu_hypre_CSRBooleanMatrix *matrix,
                                              NALU_HYPRE_Int owns_data )
{
   nalu_hypre_CSRBooleanMatrix_Get_OwnsData(matrix) = owns_data;
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRBooleanMatrixRead
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRBooleanMatrix *
nalu_hypre_CSRBooleanMatrixRead( const char *file_name )
{
   nalu_hypre_CSRBooleanMatrix  *matrix;

   FILE    *fp;

   NALU_HYPRE_Int     *matrix_i;
   NALU_HYPRE_Int     *matrix_j;
   NALU_HYPRE_Int      num_rows;
   NALU_HYPRE_Int      num_nonzeros;
   NALU_HYPRE_Int      max_col = 0;

   NALU_HYPRE_Int      file_base = 1;

   NALU_HYPRE_Int      j;

   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/

   fp = fopen(file_name, "r");

   nalu_hypre_fscanf(fp, "%d", &num_rows);

   matrix_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_rows + 1, NALU_HYPRE_MEMORY_HOST);
   for (j = 0; j < num_rows + 1; j++)
   {
      nalu_hypre_fscanf(fp, "%d", &matrix_i[j]);
      matrix_i[j] -= file_base;
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = nalu_hypre_CSRBooleanMatrixCreate(num_rows, num_rows, matrix_i[num_rows]);
   nalu_hypre_CSRBooleanMatrix_Get_I(matrix) = matrix_i;
   nalu_hypre_CSRBooleanMatrixInitialize(matrix);

   matrix_j = nalu_hypre_CSRBooleanMatrix_Get_J(matrix);
   for (j = 0; j < num_nonzeros; j++)
   {
      nalu_hypre_fscanf(fp, "%d", &matrix_j[j]);
      matrix_j[j] -= file_base;

      if (matrix_j[j] > max_col)
      {
         max_col = matrix_j[j];
      }
   }

   fclose(fp);

   nalu_hypre_CSRBooleanMatrix_Get_NNZ(matrix) = num_nonzeros;
   nalu_hypre_CSRBooleanMatrix_Get_NCols(matrix) = ++max_col;

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRBooleanMatrixPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRBooleanMatrixPrint( nalu_hypre_CSRBooleanMatrix *matrix,
                             const char             *file_name )
{
   FILE    *fp;

   NALU_HYPRE_Int     *matrix_i;
   NALU_HYPRE_Int     *matrix_j;
   NALU_HYPRE_Int      num_rows;

   NALU_HYPRE_Int      file_base = 1;

   NALU_HYPRE_Int      j;

   NALU_HYPRE_Int      ierr = 0;

   /*----------------------------------------------------------
    * Print the matrix data
    *----------------------------------------------------------*/

   matrix_i    = nalu_hypre_CSRBooleanMatrix_Get_I(matrix);
   matrix_j    = nalu_hypre_CSRBooleanMatrix_Get_J(matrix);
   num_rows    = nalu_hypre_CSRBooleanMatrix_Get_NRows(matrix);

   fp = fopen(file_name, "w");

   nalu_hypre_fprintf(fp, "%d\n", num_rows);

   for (j = 0; j <= num_rows; j++)
   {
      nalu_hypre_fprintf(fp, "%d\n", matrix_i[j] + file_base);
   }

   for (j = 0; j < matrix_i[num_rows]; j++)
   {
      nalu_hypre_fprintf(fp, "%d\n", matrix_j[j] + file_base);
   }

   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRBooleanMatrix *nalu_hypre_ParCSRBooleanMatrixCreate( MPI_Comm comm,
                                                            NALU_HYPRE_BigInt global_num_rows, NALU_HYPRE_BigInt global_num_cols,
                                                            NALU_HYPRE_BigInt *row_starts, NALU_HYPRE_BigInt *col_starts,
                                                            NALU_HYPRE_Int num_cols_offd, NALU_HYPRE_Int num_nonzeros_diag,
                                                            NALU_HYPRE_Int num_nonzeros_offd)
{
   nalu_hypre_ParCSRBooleanMatrix *matrix;
   NALU_HYPRE_Int                     num_procs, my_id;
   NALU_HYPRE_Int                     local_num_rows, local_num_cols;
   NALU_HYPRE_BigInt                  first_row_index, first_col_diag;

   matrix = nalu_hypre_CTAlloc(nalu_hypre_ParCSRBooleanMatrix,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   if (!row_starts)
   {
      nalu_hypre_GeneratePartitioning(global_num_rows, num_procs, &row_starts);
   }

   if (!col_starts)
   {
      if (global_num_rows == global_num_cols)
      {
         col_starts = row_starts;
      }
      else
      {
         nalu_hypre_GeneratePartitioning(global_num_cols, num_procs, &col_starts);
      }
   }

   first_row_index = row_starts[my_id];
   local_num_rows = row_starts[my_id + 1] - first_row_index;
   first_col_diag = col_starts[my_id];
   local_num_cols = col_starts[my_id + 1] - first_col_diag;
   nalu_hypre_ParCSRBooleanMatrix_Get_Comm(matrix) = comm;
   nalu_hypre_ParCSRBooleanMatrix_Get_Diag(matrix) =
      nalu_hypre_CSRBooleanMatrixCreate(local_num_rows, local_num_cols,
                                   num_nonzeros_diag);
   nalu_hypre_ParCSRBooleanMatrix_Get_Offd(matrix) =
      nalu_hypre_CSRBooleanMatrixCreate(local_num_rows, num_cols_offd,
                                   num_nonzeros_offd);
   nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNRows(matrix) = global_num_rows;
   nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNCols(matrix) = global_num_cols;
   nalu_hypre_ParCSRBooleanMatrix_Get_StartRow(matrix) = first_row_index;
   nalu_hypre_ParCSRBooleanMatrix_Get_FirstColDiag(matrix) = first_col_diag;
   nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) = NULL;
   nalu_hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix) = row_starts;
   nalu_hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix) = col_starts;
   nalu_hypre_ParCSRBooleanMatrix_Get_CommPkg(matrix) = NULL;

   nalu_hypre_ParCSRBooleanMatrix_Get_OwnsData(matrix)      = 1;
   nalu_hypre_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) = 1;
   nalu_hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
   {
      nalu_hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = 0;
   }

   nalu_hypre_ParCSRBooleanMatrix_Get_Rowindices(matrix)   = NULL;
   nalu_hypre_ParCSRBooleanMatrix_Get_Getrowactive(matrix) = 0;

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixDestroy( nalu_hypre_ParCSRBooleanMatrix *matrix )
{
   NALU_HYPRE_Int  ierr = 0;

   if (matrix)
   {
      if ( nalu_hypre_ParCSRBooleanMatrix_Get_OwnsData(matrix) )
      {
         nalu_hypre_CSRBooleanMatrixDestroy(nalu_hypre_ParCSRBooleanMatrix_Get_Diag(matrix));
         nalu_hypre_CSRBooleanMatrixDestroy(nalu_hypre_ParCSRBooleanMatrix_Get_Offd(matrix));
         if (nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix))
         {
            nalu_hypre_TFree(nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix), NALU_HYPRE_MEMORY_HOST);
         }
         if (nalu_hypre_ParCSRBooleanMatrix_Get_CommPkg(matrix))
         {
            nalu_hypre_MatvecCommPkgDestroy(nalu_hypre_ParCSRBooleanMatrix_Get_CommPkg(matrix));
         }
      }
      if ( nalu_hypre_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) )
      {
         nalu_hypre_TFree(nalu_hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix), NALU_HYPRE_MEMORY_HOST);
      }
      if ( nalu_hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) )
      {
         nalu_hypre_TFree(nalu_hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix), NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_TFree(nalu_hypre_ParCSRBooleanMatrix_Get_Rowindices(matrix), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixInitialize( nalu_hypre_ParCSRBooleanMatrix *matrix )
{
   NALU_HYPRE_Int  ierr = 0;

   nalu_hypre_CSRBooleanMatrixInitialize(nalu_hypre_ParCSRBooleanMatrix_Get_Diag(matrix));
   nalu_hypre_CSRBooleanMatrixInitialize(nalu_hypre_ParCSRBooleanMatrix_Get_Offd(matrix));
   nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) =
      nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nalu_hypre_CSRBooleanMatrix_Get_NCols(
                       nalu_hypre_ParCSRBooleanMatrix_Get_Offd(matrix)), NALU_HYPRE_MEMORY_HOST);
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixSetNNZ
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixSetNNZ( nalu_hypre_ParCSRBooleanMatrix *matrix)
{
   MPI_Comm comm = nalu_hypre_ParCSRBooleanMatrix_Get_Comm(matrix);
   nalu_hypre_CSRBooleanMatrix *diag = nalu_hypre_ParCSRBooleanMatrix_Get_Diag(matrix);
   NALU_HYPRE_Int *diag_i = nalu_hypre_CSRBooleanMatrix_Get_I(diag);
   nalu_hypre_CSRBooleanMatrix *offd = nalu_hypre_ParCSRBooleanMatrix_Get_Offd(matrix);
   NALU_HYPRE_Int *offd_i = nalu_hypre_CSRBooleanMatrix_Get_I(offd);
   NALU_HYPRE_Int local_num_rows = nalu_hypre_CSRBooleanMatrix_Get_NRows(diag);
   NALU_HYPRE_Int total_num_nonzeros;
   NALU_HYPRE_Int local_num_nonzeros;
   NALU_HYPRE_Int ierr = 0;

   local_num_nonzeros = diag_i[local_num_rows] + offd_i[local_num_rows];
   nalu_hypre_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, NALU_HYPRE_MPI_INT,
                       nalu_hypre_MPI_SUM, comm);
   nalu_hypre_ParCSRBooleanMatrix_Get_NNZ(matrix) = total_num_nonzeros;
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixSetDataOwner(nalu_hypre_ParCSRBooleanMatrix *matrix,
                                                NALU_HYPRE_Int owns_data )
{
   nalu_hypre_ParCSRBooleanMatrix_Get_OwnsData(matrix) = owns_data;
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixSetRowStartsOwner
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixSetRowStartsOwner(nalu_hypre_ParCSRBooleanMatrix *matrix,
                                                     NALU_HYPRE_Int owns_row_starts )
{
   nalu_hypre_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) = owns_row_starts;
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixSetColStartsOwner
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixSetColStartsOwner(nalu_hypre_ParCSRBooleanMatrix *matrix,
                                                     NALU_HYPRE_Int owns_col_starts )
{
   nalu_hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = owns_col_starts;
   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixRead
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRBooleanMatrix *
nalu_hypre_ParCSRBooleanMatrixRead( MPI_Comm comm, const char *file_name )
{
   nalu_hypre_ParCSRBooleanMatrix  *matrix;
   nalu_hypre_CSRBooleanMatrix  *diag;
   nalu_hypre_CSRBooleanMatrix  *offd;
   NALU_HYPRE_Int  my_id, i, num_procs;
   char new_file_d[80], new_file_o[80], new_file_info[80];
   NALU_HYPRE_BigInt  global_num_rows, global_num_cols;
   NALU_HYPRE_Int  num_cols_offd;
   NALU_HYPRE_Int  local_num_rows;
   NALU_HYPRE_BigInt  *row_starts;
   NALU_HYPRE_BigInt  *col_starts;
   NALU_HYPRE_BigInt  *col_map_offd;
   FILE *fp;
   NALU_HYPRE_Int equal = 1;

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   row_starts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_procs + 1, NALU_HYPRE_MEMORY_HOST);
   col_starts = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_procs + 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_sprintf(new_file_d, "%s.D.%d", file_name, my_id);
   nalu_hypre_sprintf(new_file_o, "%s.O.%d", file_name, my_id);
   nalu_hypre_sprintf(new_file_info, "%s.INFO.%d", file_name, my_id);
   fp = fopen(new_file_info, "r");
   nalu_hypre_fscanf(fp, "%b", &global_num_rows);
   nalu_hypre_fscanf(fp, "%b", &global_num_cols);
   nalu_hypre_fscanf(fp, "%d", &num_cols_offd);
   for (i = 0; i < num_procs; i++)
   {
      nalu_hypre_fscanf(fp, "%b %b", &row_starts[i], &col_starts[i]);
   }
   row_starts[num_procs] = global_num_rows;
   col_starts[num_procs] = global_num_cols;
   col_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_cols_offd; i++)
   {
      nalu_hypre_fscanf(fp, "%b", &col_map_offd[i]);
   }

   fclose(fp);

   for (i = num_procs; i >= 0; i--)
      if (row_starts[i] != col_starts[i])
      {
         equal = 0;
         break;
      }

   if (equal)
   {
      nalu_hypre_TFree(col_starts, NALU_HYPRE_MEMORY_HOST);
      col_starts = row_starts;
   }

   diag = nalu_hypre_CSRBooleanMatrixRead(new_file_d);
   local_num_rows = nalu_hypre_CSRBooleanMatrix_Get_NRows(diag);

   if (num_cols_offd)
   {
      offd = nalu_hypre_CSRBooleanMatrixRead(new_file_o);
   }
   else
   {
      offd = nalu_hypre_CSRBooleanMatrixCreate(local_num_rows, 0, 0);
   }


   matrix = nalu_hypre_CTAlloc(nalu_hypre_ParCSRBooleanMatrix,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParCSRBooleanMatrix_Get_Comm(matrix) = comm;
   nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNRows(matrix) = global_num_rows;
   nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNCols(matrix) = global_num_cols;
   nalu_hypre_ParCSRBooleanMatrix_Get_StartRow(matrix) = row_starts[my_id];
   nalu_hypre_ParCSRBooleanMatrix_Get_FirstColDiag(matrix) = col_starts[my_id];
   nalu_hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix) = row_starts;
   nalu_hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix) = col_starts;
   nalu_hypre_ParCSRBooleanMatrix_Get_CommPkg(matrix) = NULL;

   /* set defaults */
   nalu_hypre_ParCSRBooleanMatrix_Get_OwnsData(matrix) = 1;
   nalu_hypre_ParCSRBooleanMatrix_Get_OwnsRowStarts(matrix) = 1;
   nalu_hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = 1;
   if (row_starts == col_starts)
   {
      nalu_hypre_ParCSRBooleanMatrix_Get_OwnsColStarts(matrix) = 0;
   }

   nalu_hypre_ParCSRBooleanMatrix_Get_Diag(matrix) = diag;
   nalu_hypre_ParCSRBooleanMatrix_Get_Offd(matrix) = offd;
   if (num_cols_offd)
   {
      nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) = col_map_offd;
   }
   else
   {
      nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) = NULL;
   }

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixPrint( nalu_hypre_ParCSRBooleanMatrix *matrix,
                                          const char                *file_name )
{
   MPI_Comm comm = nalu_hypre_ParCSRBooleanMatrix_Get_Comm(matrix);
   NALU_HYPRE_BigInt global_num_rows = nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNRows(matrix);
   NALU_HYPRE_BigInt global_num_cols = nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNCols(matrix);
   NALU_HYPRE_BigInt *col_map_offd = nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix);
   NALU_HYPRE_BigInt *row_starts = nalu_hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix);
   NALU_HYPRE_BigInt *col_starts = nalu_hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix);
   NALU_HYPRE_Int  my_id, i, num_procs;
   char new_file_d[80], new_file_o[80], new_file_info[80];
   NALU_HYPRE_Int  ierr = 0;
   FILE *fp;
   NALU_HYPRE_Int  num_cols_offd = 0;

   if (nalu_hypre_ParCSRBooleanMatrix_Get_Offd(matrix)) num_cols_offd =
         nalu_hypre_CSRBooleanMatrix_Get_NCols(nalu_hypre_ParCSRBooleanMatrix_Get_Offd(matrix));

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   nalu_hypre_sprintf(new_file_d, "%s.D.%d", file_name, my_id);
   nalu_hypre_sprintf(new_file_o, "%s.O.%d", file_name, my_id);
   nalu_hypre_sprintf(new_file_info, "%s.INFO.%d", file_name, my_id);
   nalu_hypre_CSRBooleanMatrixPrint(nalu_hypre_ParCSRBooleanMatrix_Get_Diag(matrix), new_file_d);
   if (num_cols_offd != 0)
      nalu_hypre_CSRBooleanMatrixPrint(nalu_hypre_ParCSRBooleanMatrix_Get_Offd(matrix),
                                  new_file_o);

   fp = fopen(new_file_info, "w");
   nalu_hypre_fprintf(fp, "%b\n", global_num_rows);
   nalu_hypre_fprintf(fp, "%b\n", global_num_cols);
   nalu_hypre_fprintf(fp, "%d\n", num_cols_offd);
   for (i = 0; i < num_procs; i++)
   {
      nalu_hypre_fprintf(fp, "%b %b\n", row_starts[i], col_starts[i]);
   }
   for (i = 0; i < num_cols_offd; i++)
   {
      nalu_hypre_fprintf(fp, "%b\n", col_map_offd[i]);
   }
   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixPrintIJ
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixPrintIJ( nalu_hypre_ParCSRBooleanMatrix *matrix,
                                            const char                *filename )
{
   MPI_Comm comm = nalu_hypre_ParCSRBooleanMatrix_Get_Comm(matrix);
   NALU_HYPRE_BigInt      global_num_rows = nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNRows(matrix);
   NALU_HYPRE_BigInt      global_num_cols = nalu_hypre_ParCSRBooleanMatrix_Get_GlobalNCols(matrix);
   NALU_HYPRE_BigInt      first_row_index = nalu_hypre_ParCSRBooleanMatrix_Get_StartRow(matrix);
   NALU_HYPRE_BigInt      first_col_diag  = nalu_hypre_ParCSRBooleanMatrix_Get_FirstColDiag(matrix);
   NALU_HYPRE_BigInt     *col_map_offd    = nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix);
   NALU_HYPRE_Int      num_rows        = nalu_hypre_ParCSRBooleanMatrix_Get_NRows(matrix);
   NALU_HYPRE_Int     *diag_i;
   NALU_HYPRE_Int     *diag_j;
   NALU_HYPRE_Int     *offd_i;
   NALU_HYPRE_Int     *offd_j;
   NALU_HYPRE_Int      myid, i, j;
   NALU_HYPRE_BigInt   I, J;
   NALU_HYPRE_Int      ierr = 0;
   char     new_filename[255];
   FILE    *file;
   nalu_hypre_CSRBooleanMatrix *diag = nalu_hypre_ParCSRBooleanMatrix_Get_Diag(matrix);
   nalu_hypre_CSRBooleanMatrix *offd = nalu_hypre_ParCSRBooleanMatrix_Get_Offd(matrix);
   NALU_HYPRE_Int  num_cols_offd = 0;

   if (offd) num_cols_offd =
         nalu_hypre_CSRBooleanMatrix_Get_NCols(nalu_hypre_ParCSRBooleanMatrix_Get_Offd(matrix));

   nalu_hypre_MPI_Comm_rank(comm, &myid);

   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      nalu_hypre_printf("Error: can't open output file %s\n", new_filename);
      exit(1);
   }

   nalu_hypre_fprintf(file, "%b, %b\n", global_num_rows, global_num_cols);
   nalu_hypre_fprintf(file, "%d\n", num_rows);

   diag_i    = nalu_hypre_CSRBooleanMatrix_Get_I(diag);
   diag_j    = nalu_hypre_CSRBooleanMatrix_Get_J(diag);
   if (num_cols_offd)
   {
      offd_i    = nalu_hypre_CSRBooleanMatrix_Get_I(offd);
      offd_j    = nalu_hypre_CSRBooleanMatrix_Get_J(offd);
   }
   for (i = 0; i < num_rows; i++)
   {
      I = first_row_index + i;

      /* print diag columns */
      for (j = diag_i[i]; j < diag_i[i + 1]; j++)
      {
         J = first_col_diag + diag_j[j];
         nalu_hypre_fprintf(file, "%b, %b\n", I, J );
      }

      /* print offd columns */
      if (num_cols_offd)
      {
         for (j = offd_i[i]; j < offd_i[i + 1]; j++)
         {
            J = col_map_offd[offd_j[j]];
            nalu_hypre_fprintf(file, "%b, %b \n", I, J);
         }
      }
   }

   fclose(file);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixGetLocalRange
 * returns the row numbers of the rows stored on this processor.
 * "End" is actually the row number of the last row on this processor.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixGetLocalRange(nalu_hypre_ParCSRBooleanMatrix *matrix,
                                                 NALU_HYPRE_BigInt *row_start, NALU_HYPRE_BigInt *row_end,
                                                 NALU_HYPRE_BigInt *col_start, NALU_HYPRE_BigInt *col_end )
{
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int my_id;

   nalu_hypre_MPI_Comm_rank( nalu_hypre_ParCSRBooleanMatrix_Get_Comm(matrix), &my_id );

   *row_start = nalu_hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix)[ my_id ];
   *row_end   = nalu_hypre_ParCSRBooleanMatrix_Get_RowStarts(matrix)[ my_id + 1 ] - 1;
   *col_start = nalu_hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix)[ my_id ];
   *col_end   = nalu_hypre_ParCSRBooleanMatrix_Get_ColStarts(matrix)[ my_id + 1 ] - 1;

   return ( ierr );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixGetRow
 * Returns global column indices for a given row in the global matrix.
 * Global row number is used, but the row must be stored locally or
 * an error is returned. This implementation copies from the two matrices that
 * store the local data, storing them in the nalu_hypre_ParCSRBooleanMatrix structure.
 * Only a single row can be accessed via this function at any one time; the
 * corresponding RestoreRow function must be called, to avoid bleeding memory,
 * and to be able to look at another row.  All indices are returned in 0-based
 * indexing, no matter what is used under the hood.
 * EXCEPTION: currently this only works if the local CSR matrices
 * use 0-based indexing.
 * This code, semantics, implementation, etc., are all based on PETSc's nalu_hypre_MPI_AIJ
 * matrix code, adjusted for our data and software structures.
 * AJC 4/99.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixGetRow(nalu_hypre_ParCSRBooleanMatrix  *mat,
                                          NALU_HYPRE_BigInt row, NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind)
{
   NALU_HYPRE_Int    i, m, ierr = 0, max = 1, tmp, my_id;
   NALU_HYPRE_BigInt row_start, row_end, cstart;
   NALU_HYPRE_Int    *cworkA, *cworkB;
   NALU_HYPRE_Int    nztot, nzA, nzB, lrow;
   NALU_HYPRE_BigInt    *cmap, *idx_p;
   nalu_hypre_CSRBooleanMatrix *Aa, *Ba;

   Aa = (nalu_hypre_CSRBooleanMatrix *) nalu_hypre_ParCSRBooleanMatrix_Get_Diag(mat);
   Ba = (nalu_hypre_CSRBooleanMatrix *) nalu_hypre_ParCSRBooleanMatrix_Get_Offd(mat);

   if (nalu_hypre_ParCSRBooleanMatrix_Get_Getrowactive(mat)) { return (-1); }

   nalu_hypre_MPI_Comm_rank( nalu_hypre_ParCSRBooleanMatrix_Get_Comm(mat), &my_id );

   nalu_hypre_ParCSRBooleanMatrix_Get_Getrowactive(mat) = 1;

   row_end   = nalu_hypre_ParCSRBooleanMatrix_Get_RowStarts(mat)[ my_id + 1 ];
   row_start = nalu_hypre_ParCSRBooleanMatrix_Get_RowStarts(mat)[ my_id ];
   lrow      = (NALU_HYPRE_Int)(row - row_start);

   if (row < row_start || row >= row_end) { return (-1); }

   if ( col_ind )
   {
      m = (NALU_HYPRE_Int)(row_end - row_start);
      for ( i = 0; i < m; i++ )
      {
         tmp = nalu_hypre_CSRBooleanMatrix_Get_I(Aa)[i + 1] -
               nalu_hypre_CSRBooleanMatrix_Get_I(Aa)[i] +
               nalu_hypre_CSRBooleanMatrix_Get_I(Ba)[i + 1] -
               nalu_hypre_CSRBooleanMatrix_Get_I(Ba)[i];
         if (max < tmp) { max = tmp; }
      }
      nalu_hypre_ParCSRBooleanMatrix_Get_Rowindices(mat) = (NALU_HYPRE_BigInt *) nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, max,
                                                                                     NALU_HYPRE_MEMORY_HOST);
   }

   cstart = nalu_hypre_ParCSRBooleanMatrix_Get_FirstColDiag(mat);

   nzA = nalu_hypre_CSRBooleanMatrix_Get_I(Aa)[lrow + 1] -
         nalu_hypre_CSRBooleanMatrix_Get_I(Aa)[lrow];
   cworkA = &(nalu_hypre_CSRBooleanMatrix_Get_J(Aa)[nalu_hypre_CSRBooleanMatrix_Get_I(Aa)[lrow]]);

   nzB = nalu_hypre_CSRBooleanMatrix_Get_I(Ba)[lrow + 1] -
         nalu_hypre_CSRBooleanMatrix_Get_I(Ba)[lrow];
   cworkB = &(nalu_hypre_CSRBooleanMatrix_Get_J(Ba)[nalu_hypre_CSRBooleanMatrix_Get_I(Ba)[lrow]]);

   nztot = nzA + nzB;

   cmap  = nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(mat);

   if (col_ind)
   {
      if (nztot)
      {
         NALU_HYPRE_Int imark = -1;
         if (col_ind)
         {
            *col_ind = idx_p = nalu_hypre_ParCSRBooleanMatrix_Get_Rowindices(mat);
            if (imark > -1)
            {
               for ( i = 0; i < imark; i++ ) { idx_p[i] = cmap[cworkB[i]]; }
            }
            else
            {
               for ( i = 0; i < nzB; i++ )
               {
                  if (cmap[cworkB[i]] < cstart) { idx_p[i] = cmap[cworkB[i]]; }
                  else { break; }
               }
               imark = i;
            }
            for ( i = 0; i < nzA; i++ ) { idx_p[imark + i] = cstart + (NALU_HYPRE_BigInt)cworkA[i]; }
            for ( i = imark; i < nzB; i++ ) { idx_p[nzA + i]   = cmap[cworkB[i]]; }
         }
      }
      else
      {
         if (col_ind) { *col_ind = 0; }
      }
   }
   *size = nztot;
   return ( ierr );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBooleanMatrixRestoreRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int nalu_hypre_ParCSRBooleanMatrixRestoreRow( nalu_hypre_ParCSRBooleanMatrix *matrix,
                                               NALU_HYPRE_BigInt row, NALU_HYPRE_Int *size, NALU_HYPRE_BigInt **col_ind)
{

   if (!nalu_hypre_ParCSRBooleanMatrix_Get_Getrowactive(matrix)) { return ( -1 ); }

   nalu_hypre_ParCSRBooleanMatrix_Get_Getrowactive(matrix) = 0;

   return ( 0 );
}


/*--------------------------------------------------------------------------
 * nalu_hypre_BuildCSRBooleanMatrixMPIDataType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_BuildCSRBooleanMatrixMPIDataType(
   NALU_HYPRE_Int num_nonzeros, NALU_HYPRE_Int num_rows, NALU_HYPRE_Int *a_i, NALU_HYPRE_Int *a_j,
   nalu_hypre_MPI_Datatype *csr_matrix_datatype )
{
   NALU_HYPRE_Int      block_lens[2];
   nalu_hypre_MPI_Aint displ[2];
   nalu_hypre_MPI_Datatype   types[2];
   NALU_HYPRE_Int      ierr = 0;

   block_lens[0] = num_rows + 1;
   block_lens[1] = num_nonzeros;

   types[0] = NALU_HYPRE_MPI_INT;
   types[1] = NALU_HYPRE_MPI_INT;

   nalu_hypre_MPI_Address(a_i, &displ[0]);
   nalu_hypre_MPI_Address(a_j, &displ[1]);
   nalu_hypre_MPI_Type_struct(2, block_lens, displ, types, csr_matrix_datatype);
   nalu_hypre_MPI_Type_commit(csr_matrix_datatype);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRBooleanMatrixToParCSRBooleanMatrix:
 * generates a ParCSRBooleanMatrix distributed across the processors in comm
 * from a CSRBooleanMatrix on proc 0 .
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRBooleanMatrix *
nalu_hypre_CSRBooleanMatrixToParCSRBooleanMatrix
( MPI_Comm comm, nalu_hypre_CSRBooleanMatrix *A,
  NALU_HYPRE_BigInt *row_starts, NALU_HYPRE_BigInt *col_starts )
{
   NALU_HYPRE_BigInt       global_data[2];
   NALU_HYPRE_BigInt       global_num_rows;
   NALU_HYPRE_BigInt       global_num_cols;
   NALU_HYPRE_Int          *local_num_rows;

   NALU_HYPRE_Int          num_procs, my_id;
   NALU_HYPRE_Int          *local_num_nonzeros = NULL;
   NALU_HYPRE_Int          num_nonzeros;

   NALU_HYPRE_Int          *a_i;
   NALU_HYPRE_Int          *a_j;

   nalu_hypre_CSRBooleanMatrix *local_A;

   nalu_hypre_MPI_Request  *requests;
   nalu_hypre_MPI_Status   *status, status0;
   nalu_hypre_MPI_Datatype *csr_matrix_datatypes;

   nalu_hypre_ParCSRBooleanMatrix *par_matrix;

   NALU_HYPRE_BigInt       first_col_diag;
   NALU_HYPRE_BigInt       last_col_diag;

   NALU_HYPRE_Int i, j, ind;

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   if (my_id == 0)
   {
      global_data[0] = (NALU_HYPRE_BigInt)nalu_hypre_CSRBooleanMatrix_Get_NRows(A);
      global_data[1] = (NALU_HYPRE_BigInt)nalu_hypre_CSRBooleanMatrix_Get_NCols(A);
      a_i = nalu_hypre_CSRBooleanMatrix_Get_I(A);
      a_j = nalu_hypre_CSRBooleanMatrix_Get_J(A);
   }
   nalu_hypre_MPI_Bcast(global_data, 2, NALU_HYPRE_MPI_BIG_INT, 0, comm);
   global_num_rows = global_data[0];
   global_num_cols = global_data[1];

   local_num_rows = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_procs, NALU_HYPRE_MEMORY_HOST);
   csr_matrix_datatypes = nalu_hypre_CTAlloc(nalu_hypre_MPI_Datatype,  num_procs, NALU_HYPRE_MEMORY_HOST);

   par_matrix = nalu_hypre_ParCSRBooleanMatrixCreate (comm, global_num_rows,
                                                 global_num_cols, row_starts, col_starts, 0, 0, 0);

   row_starts = nalu_hypre_ParCSRBooleanMatrix_Get_RowStarts(par_matrix);
   col_starts = nalu_hypre_ParCSRBooleanMatrix_Get_ColStarts(par_matrix);

   for (i = 0; i < num_procs; i++)
   {
      local_num_rows[i] = (NALU_HYPRE_Int)(row_starts[i + 1] - row_starts[i]);
   }

   if (my_id == 0)
   {
      local_num_nonzeros = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_procs, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_procs - 1; i++)
         local_num_nonzeros[i] = a_i[(NALU_HYPRE_Int)row_starts[i + 1]]
                                 - a_i[(NALU_HYPRE_Int)row_starts[i]];
      local_num_nonzeros[num_procs - 1] = a_i[(NALU_HYPRE_Int)global_num_rows]
                                          - a_i[(NALU_HYPRE_Int)row_starts[num_procs - 1]];
   }
   nalu_hypre_MPI_Scatter(local_num_nonzeros, 1, NALU_HYPRE_MPI_INT, &num_nonzeros, 1, NALU_HYPRE_MPI_INT, 0, comm);

   if (my_id == 0) { num_nonzeros = local_num_nonzeros[0]; }

   local_A = nalu_hypre_CSRBooleanMatrixCreate(local_num_rows[my_id], (NALU_HYPRE_Int)global_num_cols,
                                          num_nonzeros);
   if (my_id == 0)
   {
      requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request,  num_procs - 1, NALU_HYPRE_MEMORY_HOST);
      status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,  num_procs - 1, NALU_HYPRE_MEMORY_HOST);
      j = 0;
      for (i = 1; i < num_procs; i++)
      {
         ind = a_i[(NALU_HYPRE_Int)row_starts[i]];
         nalu_hypre_BuildCSRBooleanMatrixMPIDataType(local_num_nonzeros[i],
                                                local_num_rows[i],
                                                &a_i[(NALU_HYPRE_Int)row_starts[i]],
                                                &a_j[ind],
                                                &csr_matrix_datatypes[i]);
         nalu_hypre_MPI_Isend(nalu_hypre_MPI_BOTTOM, 1, csr_matrix_datatypes[i], i, 0, comm,
                         &requests[j++]);
         nalu_hypre_MPI_Type_free(&csr_matrix_datatypes[i]);
      }
      nalu_hypre_CSRBooleanMatrix_Get_I(local_A) = a_i;
      nalu_hypre_CSRBooleanMatrix_Get_J(local_A) = a_j;
      nalu_hypre_MPI_Waitall(num_procs - 1, requests, status);
      nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(local_num_nonzeros, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      nalu_hypre_CSRBooleanMatrixInitialize(local_A);
      nalu_hypre_BuildCSRBooleanMatrixMPIDataType(num_nonzeros,
                                             local_num_rows[my_id],
                                             nalu_hypre_CSRBooleanMatrix_Get_I(local_A),
                                             nalu_hypre_CSRBooleanMatrix_Get_J(local_A),
                                             csr_matrix_datatypes);
      nalu_hypre_MPI_Recv(nalu_hypre_MPI_BOTTOM, 1, csr_matrix_datatypes[0], 0, 0, comm, &status0);
      nalu_hypre_MPI_Type_free(csr_matrix_datatypes);
   }

   first_col_diag = col_starts[my_id];
   last_col_diag = col_starts[my_id + 1] - 1;

   nalu_hypre_BooleanGenerateDiagAndOffd(local_A, par_matrix, first_col_diag, last_col_diag);

   /* set pointers back to NULL before destroying */
   if (my_id == 0)
   {
      nalu_hypre_CSRBooleanMatrix_Get_I(local_A) = NULL;
      nalu_hypre_CSRBooleanMatrix_Get_J(local_A) = NULL;
   }
   nalu_hypre_CSRBooleanMatrixDestroy(local_A);
   nalu_hypre_TFree(local_num_rows, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(csr_matrix_datatypes, NALU_HYPRE_MEMORY_HOST);

   return par_matrix;
}

NALU_HYPRE_Int
nalu_hypre_BooleanGenerateDiagAndOffd(nalu_hypre_CSRBooleanMatrix *A,
                                 nalu_hypre_ParCSRBooleanMatrix *matrix,
                                 NALU_HYPRE_BigInt first_col_diag,
                                 NALU_HYPRE_BigInt last_col_diag)
{
   NALU_HYPRE_Int  i, j;
   NALU_HYPRE_Int  jo, jd;
   NALU_HYPRE_Int  ierr = 0;
   NALU_HYPRE_Int  num_rows = nalu_hypre_CSRBooleanMatrix_Get_NRows(A);
   NALU_HYPRE_Int  num_cols = nalu_hypre_CSRBooleanMatrix_Get_NCols(A);
   NALU_HYPRE_Int *a_i = nalu_hypre_CSRBooleanMatrix_Get_I(A);
   NALU_HYPRE_Int *a_j = nalu_hypre_CSRBooleanMatrix_Get_J(A);

   nalu_hypre_CSRBooleanMatrix *diag = nalu_hypre_ParCSRBooleanMatrix_Get_Diag(matrix);
   nalu_hypre_CSRBooleanMatrix *offd = nalu_hypre_ParCSRBooleanMatrix_Get_Offd(matrix);

   NALU_HYPRE_BigInt  *col_map_offd;

   NALU_HYPRE_Int  *diag_i, *offd_i;
   NALU_HYPRE_Int  *diag_j, *offd_j;
   NALU_HYPRE_Int  *marker;
   NALU_HYPRE_Int num_cols_diag, num_cols_offd;
   NALU_HYPRE_Int first_elmt = a_i[0];
   NALU_HYPRE_Int num_nonzeros = a_i[num_rows] - first_elmt;
   NALU_HYPRE_Int counter;

   num_cols_diag = (NALU_HYPRE_Int)(last_col_diag - first_col_diag + 1);
   num_cols_offd = 0;

   if (num_cols - num_cols_diag)
   {
      nalu_hypre_CSRBooleanMatrixInitialize(diag);
      diag_i = nalu_hypre_CSRBooleanMatrix_Get_I(diag);

      nalu_hypre_CSRBooleanMatrixInitialize(offd);
      offd_i = nalu_hypre_CSRBooleanMatrix_Get_I(offd);
      marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_cols; i++)
      {
         marker[i] = 0;
      }

      jo = 0;
      jd = 0;
      for (i = 0; i < num_rows; i++)
      {
         offd_i[i] = jo;
         diag_i[i] = jd;

         for (j = a_i[i] - first_elmt; j < a_i[i + 1] - first_elmt; j++)
            if (a_j[j] < (NALU_HYPRE_Int)first_col_diag || a_j[j] > (NALU_HYPRE_Int)last_col_diag)
            {
               if (!marker[a_j[j]])
               {
                  marker[a_j[j]] = 1;
                  num_cols_offd++;
               }
               jo++;
            }
            else
            {
               jd++;
            }
      }
      offd_i[num_rows] = jo;
      diag_i[num_rows] = jd;

      nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix) =
         nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd, NALU_HYPRE_MEMORY_HOST);
      col_map_offd = nalu_hypre_ParCSRBooleanMatrix_Get_ColMapOffd(matrix);

      counter = 0;
      for (i = 0; i < num_cols; i++)
         if (marker[i])
         {
            col_map_offd[counter] = (NALU_HYPRE_BigInt)i;
            marker[i] = counter;
            counter++;
         }

      nalu_hypre_CSRBooleanMatrix_Get_NNZ(diag) = jd;
      nalu_hypre_CSRBooleanMatrixInitialize(diag);
      diag_j = nalu_hypre_CSRBooleanMatrix_Get_J(diag);

      nalu_hypre_CSRBooleanMatrix_Get_NNZ(offd) = jo;
      nalu_hypre_CSRBooleanMatrix_Get_NCols(offd) = num_cols_offd;
      nalu_hypre_CSRBooleanMatrixInitialize(offd);
      offd_j = nalu_hypre_CSRBooleanMatrix_Get_J(offd);

      jo = 0;
      jd = 0;
      for (i = 0; i < num_rows; i++)
      {
         for (j = a_i[i] - first_elmt; j < a_i[i + 1] - first_elmt; j++)
            if (a_j[j] < first_col_diag || a_j[j] > last_col_diag)
            {
               offd_j[jo++] = marker[a_j[j]];
            }
            else
            {
               diag_j[jd++] = a_j[j] - (NALU_HYPRE_Int)first_col_diag;
            }
      }
      nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      nalu_hypre_CSRBooleanMatrix_Get_NNZ(diag) = num_nonzeros;
      nalu_hypre_CSRBooleanMatrixInitialize(diag);
      diag_i = nalu_hypre_CSRBooleanMatrix_Get_I(diag);
      diag_j = nalu_hypre_CSRBooleanMatrix_Get_J(diag);

      for (i = 0; i < num_nonzeros; i++)
      {
         diag_j[i] = a_j[i];
      }
      offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_rows + 1, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_rows + 1; i++)
      {
         diag_i[i] = a_i[i];
         offd_i[i] = 0;
      }

      nalu_hypre_CSRBooleanMatrix_Get_NCols(offd) = 0;
      nalu_hypre_CSRBooleanMatrix_Get_I(offd) = offd_i;
   }

   return ierr;
}


