/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_ParCSRMatrix class.
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

#include "../seq_mv/NALU_HYPRE_seq_mv.h"
#include "../seq_mv/csr_matrix.h"

/* In addition to publically accessible interface in NALU_HYPRE_mv.h, the
   implementation in this file uses accessor macros into the sequential matrix
   structure, and so includes the .h that defines that structure. Should those
   accessor functions become proper functions at some later date, this will not
   be necessary. AJC 4/99 */

NALU_HYPRE_Int nalu_hypre_FillResponseParToCSRMatrix(void*, NALU_HYPRE_Int, NALU_HYPRE_Int, void*, MPI_Comm, void**,
                                           NALU_HYPRE_Int*);

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixCreate
 *--------------------------------------------------------------------------*/

/* If create is called and row_starts and col_starts are NOT null, then it is
   assumed that they are of length 2 containing the start row of the calling
   processor followed by the start row of the next processor - AHB 6/05 */

nalu_hypre_ParCSRMatrix*
nalu_hypre_ParCSRMatrixCreate( MPI_Comm      comm,
                          NALU_HYPRE_BigInt  global_num_rows,
                          NALU_HYPRE_BigInt  global_num_cols,
                          NALU_HYPRE_BigInt *row_starts_in,
                          NALU_HYPRE_BigInt *col_starts_in,
                          NALU_HYPRE_Int     num_cols_offd,
                          NALU_HYPRE_Int     num_nonzeros_diag,
                          NALU_HYPRE_Int     num_nonzeros_offd )
{
   nalu_hypre_ParCSRMatrix  *matrix;
   NALU_HYPRE_Int            num_procs, my_id;
   NALU_HYPRE_Int            local_num_rows;
   NALU_HYPRE_Int            local_num_cols;
   NALU_HYPRE_BigInt         row_starts[2];
   NALU_HYPRE_BigInt         col_starts[2];
   NALU_HYPRE_BigInt         first_row_index, first_col_diag;

   matrix = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix, 1, NALU_HYPRE_MEMORY_HOST);

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
   local_num_rows  = row_starts[1] - first_row_index;
   first_col_diag  = col_starts[0];
   local_num_cols  = col_starts[1] - first_col_diag;

   nalu_hypre_ParCSRMatrixComm(matrix) = comm;
   nalu_hypre_ParCSRMatrixDiag(matrix) =
      nalu_hypre_CSRMatrixCreate(local_num_rows, local_num_cols, num_nonzeros_diag);
   nalu_hypre_ParCSRMatrixOffd(matrix) =
      nalu_hypre_CSRMatrixCreate(local_num_rows, num_cols_offd, num_nonzeros_offd);
   nalu_hypre_ParCSRMatrixDiagT(matrix) = NULL;
   nalu_hypre_ParCSRMatrixOffdT(matrix) = NULL; // JSP: transposed matrices are optional
   nalu_hypre_ParCSRMatrixGlobalNumRows(matrix)   = global_num_rows;
   nalu_hypre_ParCSRMatrixGlobalNumCols(matrix)   = global_num_cols;
   nalu_hypre_ParCSRMatrixGlobalNumRownnz(matrix) = global_num_rows;
   nalu_hypre_ParCSRMatrixFirstRowIndex(matrix)   = first_row_index;
   nalu_hypre_ParCSRMatrixFirstColDiag(matrix)    = first_col_diag;
   nalu_hypre_ParCSRMatrixLastRowIndex(matrix) = first_row_index + local_num_rows - 1;
   nalu_hypre_ParCSRMatrixLastColDiag(matrix)  = first_col_diag + local_num_cols - 1;

   nalu_hypre_ParCSRMatrixRowStarts(matrix)[0] = row_starts[0];
   nalu_hypre_ParCSRMatrixRowStarts(matrix)[1] = row_starts[1];
   nalu_hypre_ParCSRMatrixColStarts(matrix)[0] = col_starts[0];
   nalu_hypre_ParCSRMatrixColStarts(matrix)[1] = col_starts[1];

   nalu_hypre_ParCSRMatrixColMapOffd(matrix)       = NULL;
   nalu_hypre_ParCSRMatrixDeviceColMapOffd(matrix) = NULL;
   nalu_hypre_ParCSRMatrixProcOrdering(matrix)     = NULL;

   nalu_hypre_ParCSRMatrixAssumedPartition(matrix) = NULL;
   nalu_hypre_ParCSRMatrixOwnsAssumedPartition(matrix) = 1;

   nalu_hypre_ParCSRMatrixCommPkg(matrix)  = NULL;
   nalu_hypre_ParCSRMatrixCommPkgT(matrix) = NULL;

   /* set defaults */
   nalu_hypre_ParCSRMatrixOwnsData(matrix)     = 1;
   nalu_hypre_ParCSRMatrixRowindices(matrix)   = NULL;
   nalu_hypre_ParCSRMatrixRowvalues(matrix)    = NULL;
   nalu_hypre_ParCSRMatrixGetrowactive(matrix) = 0;

   matrix->bdiaginv = NULL;
   matrix->bdiaginv_comm_pkg = NULL;
   matrix->bdiag_size = -1;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   nalu_hypre_ParCSRMatrixSocDiagJ(matrix) = NULL;
   nalu_hypre_ParCSRMatrixSocOffdJ(matrix) = NULL;
#endif

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixDestroy( nalu_hypre_ParCSRMatrix *matrix )
{
   if (matrix)
   {
      NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(matrix);

      if ( nalu_hypre_ParCSRMatrixOwnsData(matrix) )
      {
         nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(matrix));
         nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(matrix));

         if ( nalu_hypre_ParCSRMatrixDiagT(matrix) )
         {
            nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiagT(matrix));
         }

         if ( nalu_hypre_ParCSRMatrixOffdT(matrix) )
         {
            nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffdT(matrix));
         }

         if (nalu_hypre_ParCSRMatrixColMapOffd(matrix))
         {
            nalu_hypre_TFree(nalu_hypre_ParCSRMatrixColMapOffd(matrix), NALU_HYPRE_MEMORY_HOST);
         }

         if (nalu_hypre_ParCSRMatrixDeviceColMapOffd(matrix))
         {
            nalu_hypre_TFree(nalu_hypre_ParCSRMatrixDeviceColMapOffd(matrix), NALU_HYPRE_MEMORY_DEVICE);
         }

         if (nalu_hypre_ParCSRMatrixCommPkg(matrix))
         {
            nalu_hypre_MatvecCommPkgDestroy(nalu_hypre_ParCSRMatrixCommPkg(matrix));
         }

         if (nalu_hypre_ParCSRMatrixCommPkgT(matrix))
         {
            nalu_hypre_MatvecCommPkgDestroy(nalu_hypre_ParCSRMatrixCommPkgT(matrix));
         }
      }

      /* RL: this is actually not correct since the memory_location may have been changed after allocation
       * put them in containers TODO */
      nalu_hypre_TFree(nalu_hypre_ParCSRMatrixRowindices(matrix), memory_location);
      nalu_hypre_TFree(nalu_hypre_ParCSRMatrixRowvalues(matrix), memory_location);

      if ( nalu_hypre_ParCSRMatrixAssumedPartition(matrix) && nalu_hypre_ParCSRMatrixOwnsAssumedPartition(matrix) )
      {
         nalu_hypre_AssumedPartitionDestroy(nalu_hypre_ParCSRMatrixAssumedPartition(matrix));
      }

      if ( nalu_hypre_ParCSRMatrixProcOrdering(matrix) )
      {
         nalu_hypre_TFree(nalu_hypre_ParCSRMatrixProcOrdering(matrix), NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_TFree(matrix->bdiaginv, NALU_HYPRE_MEMORY_HOST);
      if (matrix->bdiaginv_comm_pkg)
      {
         nalu_hypre_MatvecCommPkgDestroy(matrix->bdiaginv_comm_pkg);
      }

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
      nalu_hypre_TFree(nalu_hypre_ParCSRMatrixSocDiagJ(matrix), NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TFree(nalu_hypre_ParCSRMatrixSocOffdJ(matrix), NALU_HYPRE_MEMORY_DEVICE);
#endif

      nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixInitialize_v2( nalu_hypre_ParCSRMatrix *matrix, NALU_HYPRE_MemoryLocation memory_location )
{
   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_CSRMatrixInitialize_v2(nalu_hypre_ParCSRMatrixDiag(matrix), 0, memory_location);
   nalu_hypre_CSRMatrixInitialize_v2(nalu_hypre_ParCSRMatrixOffd(matrix), 0, memory_location);

   nalu_hypre_ParCSRMatrixColMapOffd(matrix) =
      nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(matrix)),
                    NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixInitialize( nalu_hypre_ParCSRMatrix *matrix )
{
   return nalu_hypre_ParCSRMatrixInitialize_v2(matrix, nalu_hypre_ParCSRMatrixMemoryLocation(matrix));
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixClone
 * Creates and returns a new copy S of the argument A
 * The following variables are not copied because they will be constructed
 * later if needed: CommPkg, CommPkgT, rowindices, rowvalues
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRMatrix*
nalu_hypre_ParCSRMatrixClone_v2(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int copy_data,
                           NALU_HYPRE_MemoryLocation memory_location)
{
   nalu_hypre_ParCSRMatrix *S;

   S = nalu_hypre_ParCSRMatrixCreate( nalu_hypre_ParCSRMatrixComm(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                 nalu_hypre_ParCSRMatrixGlobalNumCols(A),
                                 nalu_hypre_ParCSRMatrixRowStarts(A),
                                 nalu_hypre_ParCSRMatrixColStarts(A),
                                 nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(A)),
                                 nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixDiag(A)),
                                 nalu_hypre_CSRMatrixNumNonzeros(nalu_hypre_ParCSRMatrixOffd(A)) );

   nalu_hypre_ParCSRMatrixNumNonzeros(S)  = nalu_hypre_ParCSRMatrixNumNonzeros(A);
   nalu_hypre_ParCSRMatrixDNumNonzeros(S) = nalu_hypre_ParCSRMatrixNumNonzeros(A);

   nalu_hypre_ParCSRMatrixInitialize_v2(S, memory_location);

   nalu_hypre_ParCSRMatrixCopy(A, S, copy_data);

   return S;
}

nalu_hypre_ParCSRMatrix*
nalu_hypre_ParCSRMatrixClone(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_Int copy_data)
{
   return nalu_hypre_ParCSRMatrixClone_v2(A, copy_data, nalu_hypre_ParCSRMatrixMemoryLocation(A));
}

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixMigrate(nalu_hypre_ParCSRMatrix *A, NALU_HYPRE_MemoryLocation memory_location)
{
   if (!A)
   {
      return nalu_hypre_error_flag;
   }

   NALU_HYPRE_MemoryLocation old_memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   if ( nalu_hypre_GetActualMemLocation(memory_location) != nalu_hypre_GetActualMemLocation(
           old_memory_location) )
   {
      nalu_hypre_CSRMatrix *A_diag = nalu_hypre_CSRMatrixClone_v2(nalu_hypre_ParCSRMatrixDiag(A), 1, memory_location);
      nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixDiag(A));
      nalu_hypre_ParCSRMatrixDiag(A) = A_diag;

      nalu_hypre_CSRMatrix *A_offd = nalu_hypre_CSRMatrixClone_v2(nalu_hypre_ParCSRMatrixOffd(A), 1, memory_location);
      nalu_hypre_CSRMatrixDestroy(nalu_hypre_ParCSRMatrixOffd(A));
      nalu_hypre_ParCSRMatrixOffd(A) = A_offd;

      nalu_hypre_TFree(nalu_hypre_ParCSRMatrixRowindices(A), old_memory_location);
      nalu_hypre_TFree(nalu_hypre_ParCSRMatrixRowvalues(A), old_memory_location);
   }
   else
   {
      nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixDiag(A)) = memory_location;
      nalu_hypre_CSRMatrixMemoryLocation(nalu_hypre_ParCSRMatrixOffd(A)) = memory_location;
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixSetNumNonzeros_core( nalu_hypre_ParCSRMatrix *matrix, const char* format )
{
   MPI_Comm comm;
   nalu_hypre_CSRMatrix *diag;
   nalu_hypre_CSRMatrix *offd;

   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   comm = nalu_hypre_ParCSRMatrixComm(matrix);
   diag = nalu_hypre_ParCSRMatrixDiag(matrix);
   offd = nalu_hypre_ParCSRMatrixOffd(matrix);

#if defined(NALU_HYPRE_DEBUG)
   nalu_hypre_CSRMatrixCheckSetNumNonzeros(diag);
   nalu_hypre_CSRMatrixCheckSetNumNonzeros(offd);
#endif

   if (format[0] == 'I')
   {
      NALU_HYPRE_BigInt total_num_nonzeros;
      NALU_HYPRE_BigInt local_num_nonzeros;
      local_num_nonzeros = (NALU_HYPRE_BigInt) ( nalu_hypre_CSRMatrixNumNonzeros(diag) +
                                            nalu_hypre_CSRMatrixNumNonzeros(offd) );

      nalu_hypre_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1, NALU_HYPRE_MPI_BIG_INT,
                          nalu_hypre_MPI_SUM, comm);

      nalu_hypre_ParCSRMatrixNumNonzeros(matrix) = total_num_nonzeros;
   }
   else if (format[0] == 'D')
   {
      NALU_HYPRE_Real total_num_nonzeros;
      NALU_HYPRE_Real local_num_nonzeros;
      local_num_nonzeros = (NALU_HYPRE_Real) ( nalu_hypre_CSRMatrixNumNonzeros(diag) +
                                          nalu_hypre_CSRMatrixNumNonzeros(offd) );

      nalu_hypre_MPI_Allreduce(&local_num_nonzeros, &total_num_nonzeros, 1,
                          NALU_HYPRE_MPI_REAL, nalu_hypre_MPI_SUM, comm);

      nalu_hypre_ParCSRMatrixDNumNonzeros(matrix) = total_num_nonzeros;
   }
   else
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixSetNumNonzeros
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixSetNumNonzeros( nalu_hypre_ParCSRMatrix *matrix )
{
   return nalu_hypre_ParCSRMatrixSetNumNonzeros_core(matrix, "Int");
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixSetDNumNonzeros
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixSetDNumNonzeros( nalu_hypre_ParCSRMatrix *matrix )
{
   return nalu_hypre_ParCSRMatrixSetNumNonzeros_core(matrix, "Double");
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixSetNumRownnz
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixSetNumRownnz( nalu_hypre_ParCSRMatrix *matrix )
{
   MPI_Comm          comm = nalu_hypre_ParCSRMatrixComm(matrix);
   nalu_hypre_CSRMatrix  *diag = nalu_hypre_ParCSRMatrixDiag(matrix);
   nalu_hypre_CSRMatrix  *offd = nalu_hypre_ParCSRMatrixOffd(matrix);
   NALU_HYPRE_Int        *rownnz_diag = nalu_hypre_CSRMatrixRownnz(diag);
   NALU_HYPRE_Int        *rownnz_offd = nalu_hypre_CSRMatrixRownnz(offd);
   NALU_HYPRE_Int         num_rownnz_diag = nalu_hypre_CSRMatrixNumRownnz(diag);
   NALU_HYPRE_Int         num_rownnz_offd = nalu_hypre_CSRMatrixNumRownnz(offd);

   NALU_HYPRE_BigInt      local_num_rownnz;
   NALU_HYPRE_BigInt      global_num_rownnz;
   NALU_HYPRE_Int         i, j;

   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   local_num_rownnz = i = j = 0;
   while (i < num_rownnz_diag && j < num_rownnz_offd)
   {
      local_num_rownnz++;
      if (rownnz_diag[i] < rownnz_offd[j])
      {
         i++;
      }
      else
      {
         j++;
      }
   }

   local_num_rownnz += (NALU_HYPRE_BigInt) ((num_rownnz_diag - i) + (num_rownnz_offd - j));

   nalu_hypre_MPI_Allreduce(&local_num_rownnz, &global_num_rownnz, 1,
                       NALU_HYPRE_MPI_BIG_INT, nalu_hypre_MPI_SUM, comm);

   nalu_hypre_ParCSRMatrixGlobalNumRownnz(matrix) = global_num_rownnz;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixSetDataOwner( nalu_hypre_ParCSRMatrix *matrix,
                                NALU_HYPRE_Int           owns_data )
{
   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParCSRMatrixOwnsData(matrix) = owns_data;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixSetPatternOnly
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixSetPatternOnly( nalu_hypre_ParCSRMatrix *matrix,
                                  NALU_HYPRE_Int           pattern_only)
{
   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_CSRMatrix *diag = nalu_hypre_ParCSRMatrixDiag(matrix);
   if (diag) { nalu_hypre_CSRMatrixSetPatternOnly(diag, pattern_only); }

   nalu_hypre_CSRMatrix *offd = nalu_hypre_ParCSRMatrixOffd(matrix);
   if (offd) { nalu_hypre_CSRMatrixSetPatternOnly(offd, pattern_only); }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixRead
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRMatrix *
nalu_hypre_ParCSRMatrixRead( MPI_Comm    comm,
                        const char *file_name )
{
   nalu_hypre_ParCSRMatrix  *matrix;
   nalu_hypre_CSRMatrix     *diag;
   nalu_hypre_CSRMatrix     *offd;

   NALU_HYPRE_Int            my_id, num_procs;
   NALU_HYPRE_Int            num_cols_offd;
   NALU_HYPRE_Int            i, local_num_rows;

   NALU_HYPRE_BigInt         row_starts[2];
   NALU_HYPRE_BigInt         col_starts[2];
   NALU_HYPRE_BigInt        *col_map_offd;
   NALU_HYPRE_BigInt         row_s, row_e, col_s, col_e;
   NALU_HYPRE_BigInt         global_num_rows, global_num_cols;

   FILE                *fp;
   char                 new_file_d[256], new_file_o[256], new_file_info[256];

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   nalu_hypre_sprintf(new_file_d, "%s.D.%d", file_name, my_id);
   nalu_hypre_sprintf(new_file_o, "%s.O.%d", file_name, my_id);
   nalu_hypre_sprintf(new_file_info, "%s.INFO.%d", file_name, my_id);
   fp = fopen(new_file_info, "r");
   nalu_hypre_fscanf(fp, "%b", &global_num_rows);
   nalu_hypre_fscanf(fp, "%b", &global_num_cols);
   nalu_hypre_fscanf(fp, "%d", &num_cols_offd);
   /* the bgl input file should only contain the EXACT range for local processor */
   nalu_hypre_fscanf(fp, "%b %b %b %b", &row_s, &row_e, &col_s, &col_e);
   row_starts[0] = row_s;
   row_starts[1] = row_e;
   col_starts[0] = col_s;
   col_starts[1] = col_e;

   col_map_offd = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd, NALU_HYPRE_MEMORY_HOST);

   for (i = 0; i < num_cols_offd; i++)
   {
      nalu_hypre_fscanf(fp, "%b", &col_map_offd[i]);
   }

   fclose(fp);

   diag = nalu_hypre_CSRMatrixRead(new_file_d);
   local_num_rows = nalu_hypre_CSRMatrixNumRows(diag);

   if (num_cols_offd)
   {
      offd = nalu_hypre_CSRMatrixRead(new_file_o);
   }
   else
   {
      offd = nalu_hypre_CSRMatrixCreate(local_num_rows, 0, 0);
      nalu_hypre_CSRMatrixInitialize_v2(offd, 0, NALU_HYPRE_MEMORY_HOST);
   }

   matrix = nalu_hypre_CTAlloc(nalu_hypre_ParCSRMatrix, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_ParCSRMatrixComm(matrix) = comm;
   nalu_hypre_ParCSRMatrixGlobalNumRows(matrix) = global_num_rows;
   nalu_hypre_ParCSRMatrixGlobalNumCols(matrix) = global_num_cols;
   nalu_hypre_ParCSRMatrixFirstRowIndex(matrix) = row_s;
   nalu_hypre_ParCSRMatrixFirstColDiag(matrix) = col_s;
   nalu_hypre_ParCSRMatrixLastRowIndex(matrix) = row_e - 1;
   nalu_hypre_ParCSRMatrixLastColDiag(matrix) = col_e - 1;

   nalu_hypre_ParCSRMatrixRowStarts(matrix)[0] = row_starts[0];
   nalu_hypre_ParCSRMatrixRowStarts(matrix)[1] = row_starts[1];
   nalu_hypre_ParCSRMatrixColStarts(matrix)[0] = col_starts[0];
   nalu_hypre_ParCSRMatrixColStarts(matrix)[1] = col_starts[1];

   nalu_hypre_ParCSRMatrixCommPkg(matrix) = NULL;

   /* set defaults */
   nalu_hypre_ParCSRMatrixOwnsData(matrix) = 1;
   nalu_hypre_ParCSRMatrixDiag(matrix) = diag;
   nalu_hypre_ParCSRMatrixOffd(matrix) = offd;
   if (num_cols_offd)
   {
      nalu_hypre_ParCSRMatrixColMapOffd(matrix) = col_map_offd;
   }
   else
   {
      nalu_hypre_ParCSRMatrixColMapOffd(matrix) = NULL;
   }

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixPrint( nalu_hypre_ParCSRMatrix *matrix,
                         const char         *file_name )
{
   MPI_Comm comm;
   NALU_HYPRE_BigInt global_num_rows;
   NALU_HYPRE_BigInt global_num_cols;
   NALU_HYPRE_BigInt *col_map_offd;
   NALU_HYPRE_Int  my_id, i, num_procs;
   char   new_file_d[256], new_file_o[256], new_file_info[256];
   FILE *fp;
   NALU_HYPRE_Int num_cols_offd = 0;
   NALU_HYPRE_BigInt row_s, row_e, col_s, col_e;
   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   comm = nalu_hypre_ParCSRMatrixComm(matrix);
   global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(matrix);
   global_num_cols = nalu_hypre_ParCSRMatrixGlobalNumCols(matrix);
   col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(matrix);
   if (nalu_hypre_ParCSRMatrixOffd(matrix))
   {
      num_cols_offd = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(matrix));
   }

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   nalu_hypre_sprintf(new_file_d, "%s.D.%d", file_name, my_id);
   nalu_hypre_sprintf(new_file_o, "%s.O.%d", file_name, my_id);
   nalu_hypre_sprintf(new_file_info, "%s.INFO.%d", file_name, my_id);
   nalu_hypre_CSRMatrixPrint(nalu_hypre_ParCSRMatrixDiag(matrix), new_file_d);
   if (num_cols_offd != 0)
   {
      nalu_hypre_CSRMatrixPrint(nalu_hypre_ParCSRMatrixOffd(matrix), new_file_o);
   }

   fp = fopen(new_file_info, "w");
   nalu_hypre_fprintf(fp, "%b\n", global_num_rows);
   nalu_hypre_fprintf(fp, "%b\n", global_num_cols);
   nalu_hypre_fprintf(fp, "%d\n", num_cols_offd);
   row_s = nalu_hypre_ParCSRMatrixFirstRowIndex(matrix);
   row_e = nalu_hypre_ParCSRMatrixLastRowIndex(matrix);
   col_s =  nalu_hypre_ParCSRMatrixFirstColDiag(matrix);
   col_e =  nalu_hypre_ParCSRMatrixLastColDiag(matrix);
   /* add 1 to the ends because this is a starts partition */
   nalu_hypre_fprintf(fp, "%b %b %b %b\n", row_s, row_e + 1, col_s, col_e + 1);
   for (i = 0; i < num_cols_offd; i++)
   {
      nalu_hypre_fprintf(fp, "%b\n", col_map_offd[i]);
   }
   fclose(fp);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixPrintIJ
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixPrintIJ( const nalu_hypre_ParCSRMatrix *matrix,
                           const NALU_HYPRE_Int           base_i,
                           const NALU_HYPRE_Int           base_j,
                           const char               *filename )
{
   nalu_hypre_ParCSRMatrix  *h_matrix;

   MPI_Comm             comm;
   NALU_HYPRE_BigInt         first_row_index;
   NALU_HYPRE_BigInt         first_col_diag;
   nalu_hypre_CSRMatrix     *diag;
   nalu_hypre_CSRMatrix     *offd;
   NALU_HYPRE_BigInt        *col_map_offd;
   NALU_HYPRE_Int            num_rows;
   const NALU_HYPRE_BigInt  *row_starts;
   const NALU_HYPRE_BigInt  *col_starts;
   NALU_HYPRE_Complex       *diag_data;
   NALU_HYPRE_Int           *diag_i;
   NALU_HYPRE_Int           *diag_j;
   NALU_HYPRE_Complex       *offd_data;
   NALU_HYPRE_Int           *offd_i;
   NALU_HYPRE_Int           *offd_j;
   NALU_HYPRE_Int            myid, num_procs, i, j;
   NALU_HYPRE_BigInt         I, J;
   char                 new_filename[255];
   FILE                *file;
   NALU_HYPRE_Int            num_nonzeros_offd;
   NALU_HYPRE_BigInt         ilower, iupper, jlower, jupper;

   NALU_HYPRE_MemoryLocation memory_location =
      nalu_hypre_ParCSRMatrixMemoryLocation((nalu_hypre_ParCSRMatrix*) matrix);

   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /* Create temporary matrix on host memory if needed */
   if (nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_HOST)
   {
      h_matrix = (nalu_hypre_ParCSRMatrix *) matrix;
   }
   else
   {
      h_matrix = nalu_hypre_ParCSRMatrixClone_v2((nalu_hypre_ParCSRMatrix *) matrix, 1, NALU_HYPRE_MEMORY_HOST);
   }

   comm            = nalu_hypre_ParCSRMatrixComm(h_matrix);
   first_row_index = nalu_hypre_ParCSRMatrixFirstRowIndex(h_matrix);
   first_col_diag  = nalu_hypre_ParCSRMatrixFirstColDiag(h_matrix);
   diag            = nalu_hypre_ParCSRMatrixDiag(h_matrix);
   offd            = nalu_hypre_ParCSRMatrixOffd(h_matrix);
   col_map_offd    = nalu_hypre_ParCSRMatrixColMapOffd(h_matrix);
   num_rows        = nalu_hypre_ParCSRMatrixNumRows(h_matrix);
   row_starts      = nalu_hypre_ParCSRMatrixRowStarts(h_matrix);
   col_starts      = nalu_hypre_ParCSRMatrixColStarts(h_matrix);
   nalu_hypre_MPI_Comm_rank(comm, &myid);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "w")) == NULL)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error: can't open output file %s\n");
      return nalu_hypre_error_flag;
   }

   diag_data = nalu_hypre_CSRMatrixData(diag);
   diag_i    = nalu_hypre_CSRMatrixI(diag);
   diag_j    = nalu_hypre_CSRMatrixJ(diag);

   num_nonzeros_offd = nalu_hypre_CSRMatrixNumNonzeros(offd);
   if (num_nonzeros_offd)
   {
      offd_data = nalu_hypre_CSRMatrixData(offd);
      offd_i    = nalu_hypre_CSRMatrixI(offd);
      offd_j    = nalu_hypre_CSRMatrixJ(offd);
   }

   ilower = row_starts[0] + (NALU_HYPRE_BigInt) base_i;
   iupper = row_starts[1] + (NALU_HYPRE_BigInt) base_i - 1;
   jlower = col_starts[0] + (NALU_HYPRE_BigInt) base_j;
   jupper = col_starts[1] + (NALU_HYPRE_BigInt) base_j - 1;

   nalu_hypre_fprintf(file, "%b %b %b %b\n", ilower, iupper, jlower, jupper);

   for (i = 0; i < num_rows; i++)
   {
      I = first_row_index + (NALU_HYPRE_BigInt)(i + base_i);

      /* print diag columns */
      for (j = diag_i[i]; j < diag_i[i + 1]; j++)
      {
         J = first_col_diag + (NALU_HYPRE_BigInt)(diag_j[j] + base_j);
         if (diag_data)
         {
#ifdef NALU_HYPRE_COMPLEX
            nalu_hypre_fprintf(file, "%b %b %.14e , %.14e\n", I, J,
                          nalu_hypre_creal(diag_data[j]), nalu_hypre_cimag(diag_data[j]));
#else
            nalu_hypre_fprintf(file, "%b %b %.14e\n", I, J, diag_data[j]);
#endif
         }
         else
         {
            nalu_hypre_fprintf(file, "%b %b\n", I, J);
         }
      }

      /* print offd columns */
      if (num_nonzeros_offd)
      {
         for (j = offd_i[i]; j < offd_i[i + 1]; j++)
         {
            J = col_map_offd[offd_j[j]] + (NALU_HYPRE_BigInt) base_j;
            if (offd_data)
            {
#ifdef NALU_HYPRE_COMPLEX
               nalu_hypre_fprintf(file, "%b %b %.14e , %.14e\n", I, J,
                             nalu_hypre_creal(offd_data[j]), nalu_hypre_cimag(offd_data[j]));
#else
               nalu_hypre_fprintf(file, "%b %b %.14e\n", I, J, offd_data[j]);
#endif
            }
            else
            {
               nalu_hypre_fprintf(file, "%b %b\n", I, J);
            }
         }
      }
   }

   fclose(file);

   /* Free temporary matrix */
   if (nalu_hypre_GetActualMemLocation(memory_location) != nalu_hypre_MEMORY_HOST)
   {
      nalu_hypre_ParCSRMatrixDestroy(h_matrix);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixReadIJ
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixReadIJ( MPI_Comm             comm,
                          const char          *filename,
                          NALU_HYPRE_Int           *base_i_ptr,
                          NALU_HYPRE_Int           *base_j_ptr,
                          nalu_hypre_ParCSRMatrix **matrix_ptr)
{
   NALU_HYPRE_BigInt        global_num_rows;
   NALU_HYPRE_BigInt        global_num_cols;
   NALU_HYPRE_BigInt        first_row_index;
   NALU_HYPRE_BigInt        first_col_diag;
   NALU_HYPRE_BigInt        last_col_diag;
   nalu_hypre_ParCSRMatrix *matrix;
   nalu_hypre_CSRMatrix    *diag;
   nalu_hypre_CSRMatrix    *offd;
   NALU_HYPRE_BigInt       *col_map_offd;
   NALU_HYPRE_BigInt        row_starts[2];
   NALU_HYPRE_BigInt        col_starts[2];
   NALU_HYPRE_Int           num_rows;
   NALU_HYPRE_BigInt        big_base_i, big_base_j;
   NALU_HYPRE_Int           base_i, base_j;
   NALU_HYPRE_Complex      *diag_data;
   NALU_HYPRE_Int          *diag_i;
   NALU_HYPRE_Int          *diag_j;
   NALU_HYPRE_Complex      *offd_data;
   NALU_HYPRE_Int          *offd_i;
   NALU_HYPRE_Int          *offd_j;
   NALU_HYPRE_BigInt       *tmp_j;
   NALU_HYPRE_BigInt       *aux_offd_j;
   NALU_HYPRE_BigInt        I, J;
   NALU_HYPRE_Int           myid, num_procs, i, i2, j;
   char                new_filename[255];
   FILE               *file;
   NALU_HYPRE_Int           num_cols_offd, num_nonzeros_diag, num_nonzeros_offd;
   NALU_HYPRE_Int           i_col, num_cols;
   NALU_HYPRE_Int           diag_cnt, offd_cnt, row_cnt;
   NALU_HYPRE_Complex       data;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &myid);

   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);

   if ((file = fopen(new_filename, "r")) == NULL)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error: can't open output file %s\n");
      return nalu_hypre_error_flag;
   }

   nalu_hypre_fscanf(file, "%b %b", &global_num_rows, &global_num_cols);
   nalu_hypre_fscanf(file, "%d %d %d", &num_rows, &num_cols, &num_cols_offd);
   nalu_hypre_fscanf(file, "%d %d", &num_nonzeros_diag, &num_nonzeros_offd);
   nalu_hypre_fscanf(file, "%b %b %b %b", &row_starts[0], &col_starts[0], &row_starts[1], &col_starts[1]);

   big_base_i = row_starts[0];
   big_base_j = col_starts[0];
   base_i = (NALU_HYPRE_Int) row_starts[0];
   base_j = (NALU_HYPRE_Int) col_starts[0];

   matrix = nalu_hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                     row_starts, col_starts, num_cols_offd,
                                     num_nonzeros_diag, num_nonzeros_offd);
   nalu_hypre_ParCSRMatrixInitialize(matrix);

   diag = nalu_hypre_ParCSRMatrixDiag(matrix);
   offd = nalu_hypre_ParCSRMatrixOffd(matrix);

   diag_data = nalu_hypre_CSRMatrixData(diag);
   diag_i    = nalu_hypre_CSRMatrixI(diag);
   diag_j    = nalu_hypre_CSRMatrixJ(diag);

   offd_i    = nalu_hypre_CSRMatrixI(offd);
   if (num_nonzeros_offd)
   {
      offd_data = nalu_hypre_CSRMatrixData(offd);
      offd_j    = nalu_hypre_CSRMatrixJ(offd);
      tmp_j     = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_nonzeros_offd, NALU_HYPRE_MEMORY_HOST);
   }

   first_row_index = nalu_hypre_ParCSRMatrixFirstRowIndex(matrix);
   first_col_diag = nalu_hypre_ParCSRMatrixFirstColDiag(matrix);
   last_col_diag = first_col_diag + (NALU_HYPRE_BigInt)num_cols - 1;

   diag_cnt = 0;
   offd_cnt = 0;
   row_cnt = 0;
   for (i = 0; i < num_nonzeros_diag + num_nonzeros_offd; i++)
   {
      /* read values */
      nalu_hypre_fscanf(file, "%b %b %le", &I, &J, &data);
      i2 = (NALU_HYPRE_Int)(I - big_base_i - first_row_index);
      J -= big_base_j;
      if (i2 > row_cnt)
      {
         diag_i[i2] = diag_cnt;
         offd_i[i2] = offd_cnt;
         row_cnt++;
      }
      if (J < first_col_diag || J > last_col_diag)
      {
         tmp_j[offd_cnt] = J;
         offd_data[offd_cnt++] = data;
      }
      else
      {
         diag_j[diag_cnt] = (NALU_HYPRE_Int)(J - first_col_diag);
         diag_data[diag_cnt++] = data;
      }
   }
   diag_i[num_rows] = diag_cnt;
   offd_i[num_rows] = offd_cnt;

   fclose(file);

   /*  generate col_map_offd */
   if (num_nonzeros_offd)
   {
      aux_offd_j = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_nonzeros_offd, NALU_HYPRE_MEMORY_HOST);
      for (i = 0; i < num_nonzeros_offd; i++)
      {
         aux_offd_j[i] = (NALU_HYPRE_BigInt)offd_j[i];
      }
      nalu_hypre_BigQsort0(aux_offd_j, 0, num_nonzeros_offd - 1);
      col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(matrix);
      col_map_offd[0] = aux_offd_j[0];
      offd_cnt = 0;
      for (i = 1; i < num_nonzeros_offd; i++)
      {
         if (aux_offd_j[i] > col_map_offd[offd_cnt])
         {
            col_map_offd[++offd_cnt] = aux_offd_j[i];
         }
      }
      for (i = 0; i < num_nonzeros_offd; i++)
      {
         offd_j[i] = nalu_hypre_BigBinarySearch(col_map_offd, tmp_j[i], num_cols_offd);
      }
      nalu_hypre_TFree(aux_offd_j, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(tmp_j, NALU_HYPRE_MEMORY_HOST);
   }

   /* move diagonal element in first position in each row */
   for (i = 0; i < num_rows; i++)
   {
      i_col = diag_i[i];
      for (j = i_col; j < diag_i[i + 1]; j++)
      {
         if (diag_j[j] == i)
         {
            diag_j[j] = diag_j[i_col];
            data = diag_data[j];
            diag_data[j] = diag_data[i_col];
            diag_data[i_col] = data;
            diag_j[i_col] = i;
            break;
         }
      }
   }

   *base_i_ptr = base_i;
   *base_j_ptr = base_j;
   *matrix_ptr = matrix;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGetLocalRange
 * returns the row numbers of the rows stored on this processor.
 * "End" is actually the row number of the last row on this processor.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGetLocalRange( nalu_hypre_ParCSRMatrix *matrix,
                                 NALU_HYPRE_BigInt       *row_start,
                                 NALU_HYPRE_BigInt       *row_end,
                                 NALU_HYPRE_BigInt       *col_start,
                                 NALU_HYPRE_BigInt       *col_end )
{
   NALU_HYPRE_Int my_id;

   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_MPI_Comm_rank( nalu_hypre_ParCSRMatrixComm(matrix), &my_id );

   *row_start = nalu_hypre_ParCSRMatrixFirstRowIndex(matrix);
   *row_end = nalu_hypre_ParCSRMatrixLastRowIndex(matrix);
   *col_start =  nalu_hypre_ParCSRMatrixFirstColDiag(matrix);
   *col_end =  nalu_hypre_ParCSRMatrixLastColDiag(matrix);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixGetRow
 * Returns global column indices and/or values for a given row in the global
 * matrix. Global row number is used, but the row must be stored locally or
 * an error is returned. This implementation copies from the two matrices that
 * store the local data, storing them in the nalu_hypre_ParCSRMatrix structure.
 * Only a single row can be accessed via this function at any one time; the
 * corresponding RestoreRow function must be called, to avoid bleeding memory,
 * and to be able to look at another row.
 * Either one of col_ind and values can be left null, and those values will
 * not be returned.
 * All indices are returned in 0-based indexing, no matter what is used under
 * the hood. EXCEPTION: currently this only works if the local CSR matrices
 * use 0-based indexing.
 * This code, semantics, implementation, etc., are all based on PETSc's nalu_hypre_MPI_AIJ
 * matrix code, adjusted for our data and software structures.
 * AJC 4/99.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGetRowHost( nalu_hypre_ParCSRMatrix  *mat,
                              NALU_HYPRE_BigInt         row,
                              NALU_HYPRE_Int           *size,
                              NALU_HYPRE_BigInt       **col_ind,
                              NALU_HYPRE_Complex      **values )
{
   NALU_HYPRE_Int my_id;
   NALU_HYPRE_BigInt row_start, row_end;
   nalu_hypre_CSRMatrix *Aa;
   nalu_hypre_CSRMatrix *Ba;

   if (!mat)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   Aa = (nalu_hypre_CSRMatrix *) nalu_hypre_ParCSRMatrixDiag(mat);
   Ba = (nalu_hypre_CSRMatrix *) nalu_hypre_ParCSRMatrixOffd(mat);

   if (nalu_hypre_ParCSRMatrixGetrowactive(mat))
   {
      return (-1);
   }

   nalu_hypre_MPI_Comm_rank( nalu_hypre_ParCSRMatrixComm(mat), &my_id );

   nalu_hypre_ParCSRMatrixGetrowactive(mat) = 1;
   row_start = nalu_hypre_ParCSRMatrixFirstRowIndex(mat);
   row_end = nalu_hypre_ParCSRMatrixLastRowIndex(mat) + 1;
   if (row < row_start || row >= row_end)
   {
      return (-1);
   }

   /* if buffer is not allocated and some information is requested,
      allocate buffer */
   if (!nalu_hypre_ParCSRMatrixRowvalues(mat) && ( col_ind || values ))
   {
      /*
        allocate enough space to hold information from the longest row.
      */
      NALU_HYPRE_Int max = 1, tmp;
      NALU_HYPRE_Int i;
      NALU_HYPRE_Int m = row_end - row_start;

      for ( i = 0; i < m; i++ )
      {
         tmp = nalu_hypre_CSRMatrixI(Aa)[i + 1] - nalu_hypre_CSRMatrixI(Aa)[i] +
               nalu_hypre_CSRMatrixI(Ba)[i + 1] - nalu_hypre_CSRMatrixI(Ba)[i];
         if (max < tmp)
         {
            max = tmp;
         }
      }

      nalu_hypre_ParCSRMatrixRowvalues(mat)  =
         (NALU_HYPRE_Complex *) nalu_hypre_CTAlloc(NALU_HYPRE_Complex, max, nalu_hypre_ParCSRMatrixMemoryLocation(mat));
      nalu_hypre_ParCSRMatrixRowindices(mat) =
         (NALU_HYPRE_BigInt *)  nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  max, nalu_hypre_ParCSRMatrixMemoryLocation(mat));
   }

   /* Copy from dual sequential matrices into buffer */
   {
      NALU_HYPRE_Complex    *vworkA, *vworkB, *v_p;
      NALU_HYPRE_Int        i, *cworkA, *cworkB;
      NALU_HYPRE_BigInt     cstart = nalu_hypre_ParCSRMatrixFirstColDiag(mat);
      NALU_HYPRE_Int        nztot, nzA, nzB, lrow = (NALU_HYPRE_Int)(row - row_start);
      NALU_HYPRE_BigInt     *cmap, *idx_p;

      nzA = nalu_hypre_CSRMatrixI(Aa)[lrow + 1] - nalu_hypre_CSRMatrixI(Aa)[lrow];
      cworkA = &( nalu_hypre_CSRMatrixJ(Aa)[ nalu_hypre_CSRMatrixI(Aa)[lrow] ] );
      vworkA = &( nalu_hypre_CSRMatrixData(Aa)[ nalu_hypre_CSRMatrixI(Aa)[lrow] ] );

      nzB = nalu_hypre_CSRMatrixI(Ba)[lrow + 1] - nalu_hypre_CSRMatrixI(Ba)[lrow];
      cworkB = &( nalu_hypre_CSRMatrixJ(Ba)[ nalu_hypre_CSRMatrixI(Ba)[lrow] ] );
      vworkB = &( nalu_hypre_CSRMatrixData(Ba)[ nalu_hypre_CSRMatrixI(Ba)[lrow] ] );

      nztot = nzA + nzB;

      cmap = nalu_hypre_ParCSRMatrixColMapOffd(mat);

      if (values || col_ind)
      {
         if (nztot)
         {
            /* Sort by increasing column numbers, assuming A and B already sorted */
            NALU_HYPRE_Int imark = -1;

            if (values)
            {
               *values = v_p = nalu_hypre_ParCSRMatrixRowvalues(mat);
               for ( i = 0; i < nzB; i++ )
               {
                  if (cmap[cworkB[i]] < cstart)
                  {
                     v_p[i] = vworkB[i];
                  }
                  else
                  {
                     break;
                  }
               }
               imark = i;
               for ( i = 0; i < nzA; i++ )
               {
                  v_p[imark + i] = vworkA[i];
               }
               for ( i = imark; i < nzB; i++ )
               {
                  v_p[nzA + i] = vworkB[i];
               }
            }

            if (col_ind)
            {
               *col_ind = idx_p = nalu_hypre_ParCSRMatrixRowindices(mat);
               if (imark > -1)
               {
                  for ( i = 0; i < imark; i++ )
                  {
                     idx_p[i] = cmap[cworkB[i]];
                  }
               }
               else
               {
                  for ( i = 0; i < nzB; i++ )
                  {
                     if (cmap[cworkB[i]] < cstart)
                     {
                        idx_p[i] = cmap[cworkB[i]];
                     }
                     else
                     {
                        break;
                     }
                  }
                  imark = i;
               }
               for ( i = 0; i < nzA; i++ )
               {
                  idx_p[imark + i] = cstart + cworkA[i];
               }
               for ( i = imark; i < nzB; i++ )
               {
                  idx_p[nzA + i] = cmap[cworkB[i]];
               }
            }
         }
         else
         {
            if (col_ind)
            {
               *col_ind = 0;
            }
            if (values)
            {
               *values = 0;
            }
         }
      }

      *size = nztot;
   } /* End of copy */

   return nalu_hypre_error_flag;
}


NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixGetRow( nalu_hypre_ParCSRMatrix  *mat,
                          NALU_HYPRE_BigInt         row,
                          NALU_HYPRE_Int           *size,
                          NALU_HYPRE_BigInt       **col_ind,
                          NALU_HYPRE_Complex      **values )
{
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_ParCSRMatrixMemoryLocation(mat) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      return nalu_hypre_ParCSRMatrixGetRowDevice(mat, row, size, col_ind, values);
   }
   else
#endif
   {
      return nalu_hypre_ParCSRMatrixGetRowHost(mat, row, size, col_ind, values);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixRestoreRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixRestoreRow( nalu_hypre_ParCSRMatrix *matrix,
                              NALU_HYPRE_BigInt        row,
                              NALU_HYPRE_Int          *size,
                              NALU_HYPRE_BigInt      **col_ind,
                              NALU_HYPRE_Complex     **values )
{
   if (!nalu_hypre_ParCSRMatrixGetrowactive(matrix))
   {
      nalu_hypre_error(NALU_HYPRE_ERROR_GENERIC);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParCSRMatrixGetrowactive(matrix) = 0;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixToParCSRMatrix:
 *
 * Generates a ParCSRMatrix distributed across the processors in comm
 * from a CSRMatrix on proc 0 .
 *
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRMatrix *
nalu_hypre_CSRMatrixToParCSRMatrix( MPI_Comm         comm,
                               nalu_hypre_CSRMatrix *A,
                               NALU_HYPRE_BigInt    *global_row_starts,
                               NALU_HYPRE_BigInt    *global_col_starts )
{
   nalu_hypre_ParCSRMatrix *parcsr_A;

   NALU_HYPRE_BigInt       *global_data;
   NALU_HYPRE_BigInt        global_size;
   NALU_HYPRE_BigInt        global_num_rows;
   NALU_HYPRE_BigInt        global_num_cols;

   NALU_HYPRE_Int           num_procs, my_id;
   NALU_HYPRE_Int          *num_rows_proc;
   NALU_HYPRE_Int          *num_nonzeros_proc;
   NALU_HYPRE_BigInt        row_starts[2];
   NALU_HYPRE_BigInt        col_starts[2];

   nalu_hypre_CSRMatrix    *local_A;
   NALU_HYPRE_Complex      *A_data;
   NALU_HYPRE_Int          *A_i;
   NALU_HYPRE_Int          *A_j;

   nalu_hypre_MPI_Request  *requests;
   nalu_hypre_MPI_Status   *status, status0;
   nalu_hypre_MPI_Datatype *csr_matrix_datatypes;

   NALU_HYPRE_Int           free_global_row_starts = 0;
   NALU_HYPRE_Int           free_global_col_starts = 0;

   NALU_HYPRE_Int           total_size;
   NALU_HYPRE_BigInt        first_col_diag;
   NALU_HYPRE_BigInt        last_col_diag;
   NALU_HYPRE_Int           num_rows;
   NALU_HYPRE_Int           num_nonzeros;
   NALU_HYPRE_Int           i, ind;

   nalu_hypre_MPI_Comm_rank(comm, &my_id);
   nalu_hypre_MPI_Comm_size(comm, &num_procs);

   total_size = 4;
   if (my_id == 0)
   {
      total_size += 2 * (num_procs + 1);
   }

   global_data = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, total_size, NALU_HYPRE_MEMORY_HOST);
   if (my_id == 0)
   {
      global_size = 3;
      if (global_row_starts)
      {
         if (global_col_starts)
         {
            if (global_col_starts != global_row_starts)
            {
               /* contains code for what to expect,
                  if 0: global_row_starts = global_col_starts, only global_row_starts given
                  if 1: only global_row_starts given, global_col_starts = NULL
                  if 2: both global_row_starts and global_col_starts given
                  if 3: only global_col_starts given, global_row_starts = NULL */
               global_data[3] = 2;
               global_size += (NALU_HYPRE_BigInt) (2 * (num_procs + 1) + 1);
               for (i = 0; i < (num_procs + 1); i++)
               {
                  global_data[i + 4] = global_row_starts[i];
               }
               for (i = 0; i < (num_procs + 1); i++)
               {
                  global_data[i + num_procs + 5] = global_col_starts[i];
               }
            }
            else
            {
               global_data[3] = 0;
               global_size += (NALU_HYPRE_BigInt) ((num_procs + 1) + 1);
               for (i = 0; i < (num_procs + 1); i++)
               {
                  global_data[i + 4] = global_row_starts[i];
               }
            }
         }
         else
         {
            global_data[3] = 1;
            global_size += (NALU_HYPRE_BigInt) ((num_procs + 1) + 1);
            for (i = 0; i < (num_procs + 1); i++)
            {
               global_data[i + 4] = global_row_starts[i];
            }
         }
      }
      else
      {
         if (global_col_starts)
         {
            global_data[3] = 3;
            global_size += (NALU_HYPRE_BigInt) ((num_procs + 1) + 1);
            for (i = 0; i < (num_procs + 1); i++)
            {
               global_data[i + 4] = global_col_starts[i];
            }
         }
      }

      global_data[0] = (NALU_HYPRE_BigInt) nalu_hypre_CSRMatrixNumRows(A);
      global_data[1] = (NALU_HYPRE_BigInt) nalu_hypre_CSRMatrixNumCols(A);
      global_data[2] = global_size;
      A_data = nalu_hypre_CSRMatrixData(A);
      A_i = nalu_hypre_CSRMatrixI(A);
      A_j = nalu_hypre_CSRMatrixJ(A);
   }
   nalu_hypre_MPI_Bcast(global_data, 3, NALU_HYPRE_MPI_BIG_INT, 0, comm);
   global_num_rows = global_data[0];
   global_num_cols = global_data[1];
   global_size     = global_data[2];

   if (global_size > 3)
   {
      NALU_HYPRE_Int  send_start;

      if (global_data[3] == 2)
      {
         send_start = 4;
         nalu_hypre_MPI_Scatter(&global_data[send_start], 1, NALU_HYPRE_MPI_BIG_INT,
                           &row_starts[0], 1, NALU_HYPRE_MPI_BIG_INT, 0, comm);

         send_start = 5;
         nalu_hypre_MPI_Scatter(&global_data[send_start], 1, NALU_HYPRE_MPI_BIG_INT,
                           &row_starts[1], 1, NALU_HYPRE_MPI_BIG_INT, 0, comm);

         send_start = 4 + (num_procs + 1);
         nalu_hypre_MPI_Scatter(&global_data[send_start], 1, NALU_HYPRE_MPI_BIG_INT,
                           &col_starts[0], 1, NALU_HYPRE_MPI_BIG_INT, 0, comm);

         send_start = 5 + (num_procs + 1);
         nalu_hypre_MPI_Scatter(&global_data[send_start], 1, NALU_HYPRE_MPI_BIG_INT,
                           &col_starts[1], 1, NALU_HYPRE_MPI_BIG_INT, 0, comm);
      }
      else if ((global_data[3] == 0) || (global_data[3] == 1))
      {
         send_start = 4;
         nalu_hypre_MPI_Scatter(&global_data[send_start], 1, NALU_HYPRE_MPI_BIG_INT,
                           &row_starts[0], 1, NALU_HYPRE_MPI_BIG_INT, 0, comm);

         send_start = 5;
         nalu_hypre_MPI_Scatter(&global_data[send_start], 1, NALU_HYPRE_MPI_BIG_INT,
                           &row_starts[1], 1, NALU_HYPRE_MPI_BIG_INT, 0, comm);

         if (global_data[3] == 0)
         {
            col_starts[0] = row_starts[0];
            col_starts[1] = row_starts[1];
         }
      }
      else
      {
         send_start = 4;
         nalu_hypre_MPI_Scatter(&global_data[send_start], 1, NALU_HYPRE_MPI_BIG_INT,
                           &col_starts[0], 1, NALU_HYPRE_MPI_BIG_INT, 0, comm);

         send_start = 5;
         nalu_hypre_MPI_Scatter(&global_data[send_start], 1, NALU_HYPRE_MPI_BIG_INT,
                           &col_starts[1], 1, NALU_HYPRE_MPI_BIG_INT, 0, comm);
      }
   }
   nalu_hypre_TFree(global_data, NALU_HYPRE_MEMORY_HOST);

   // Create ParCSR matrix
   parcsr_A = nalu_hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                       row_starts, col_starts, 0, 0, 0);

   // Allocate memory for building ParCSR matrix
   num_rows_proc     = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_procs, NALU_HYPRE_MEMORY_HOST);
   num_nonzeros_proc = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_procs, NALU_HYPRE_MEMORY_HOST);

   if (my_id == 0)
   {
      if (!global_row_starts)
      {
         nalu_hypre_GeneratePartitioning(global_num_rows, num_procs, &global_row_starts);
         free_global_row_starts = 1;
      }
      if (!global_col_starts)
      {
         nalu_hypre_GeneratePartitioning(global_num_rows, num_procs, &global_col_starts);
         free_global_col_starts = 1;
      }

      for (i = 0; i < num_procs; i++)
      {
         num_rows_proc[i] = (NALU_HYPRE_Int) (global_row_starts[i + 1] - global_row_starts[i]);
         num_nonzeros_proc[i] = A_i[(NALU_HYPRE_Int)global_row_starts[i + 1]] -
                                A_i[(NALU_HYPRE_Int)global_row_starts[i]];
      }
      //num_nonzeros_proc[num_procs-1] = A_i[(NALU_HYPRE_Int)global_num_rows] - A_i[(NALU_HYPRE_Int)row_starts[num_procs-1]];
   }
   nalu_hypre_MPI_Scatter(num_rows_proc, 1, NALU_HYPRE_MPI_INT, &num_rows, 1, NALU_HYPRE_MPI_INT, 0, comm);
   nalu_hypre_MPI_Scatter(num_nonzeros_proc, 1, NALU_HYPRE_MPI_INT, &num_nonzeros, 1, NALU_HYPRE_MPI_INT, 0, comm);

   /* RL: this is not correct: (NALU_HYPRE_Int) global_num_cols */
   local_A = nalu_hypre_CSRMatrixCreate(num_rows, (NALU_HYPRE_Int) global_num_cols, num_nonzeros);

   csr_matrix_datatypes = nalu_hypre_CTAlloc(nalu_hypre_MPI_Datatype,  num_procs, NALU_HYPRE_MEMORY_HOST);
   if (my_id == 0)
   {
      requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request, num_procs - 1, NALU_HYPRE_MEMORY_HOST);
      status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status, num_procs - 1, NALU_HYPRE_MEMORY_HOST);
      for (i = 1; i < num_procs; i++)
      {
         ind = A_i[(NALU_HYPRE_Int) global_row_starts[i]];

         nalu_hypre_BuildCSRMatrixMPIDataType(num_nonzeros_proc[i],
                                         num_rows_proc[i],
                                         &A_data[ind],
                                         &A_i[(NALU_HYPRE_Int) global_row_starts[i]],
                                         &A_j[ind],
                                         &csr_matrix_datatypes[i]);
         nalu_hypre_MPI_Isend(nalu_hypre_MPI_BOTTOM, 1, csr_matrix_datatypes[i], i, 0, comm,
                         &requests[i - 1]);
         nalu_hypre_MPI_Type_free(&csr_matrix_datatypes[i]);
      }
      nalu_hypre_CSRMatrixData(local_A) = A_data;
      nalu_hypre_CSRMatrixI(local_A) = A_i;
      nalu_hypre_CSRMatrixJ(local_A) = A_j;
      nalu_hypre_CSRMatrixOwnsData(local_A) = 0;

      nalu_hypre_MPI_Waitall(num_procs - 1, requests, status);

      nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(num_rows_proc, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(num_nonzeros_proc, NALU_HYPRE_MEMORY_HOST);

      if (free_global_row_starts)
      {
         nalu_hypre_TFree(global_row_starts, NALU_HYPRE_MEMORY_HOST);
      }
      if (free_global_col_starts)
      {
         nalu_hypre_TFree(global_col_starts, NALU_HYPRE_MEMORY_HOST);
      }
   }
   else
   {
      nalu_hypre_CSRMatrixInitialize(local_A);
      nalu_hypre_BuildCSRMatrixMPIDataType(num_nonzeros,
                                      num_rows,
                                      nalu_hypre_CSRMatrixData(local_A),
                                      nalu_hypre_CSRMatrixI(local_A),
                                      nalu_hypre_CSRMatrixJ(local_A),
                                      &csr_matrix_datatypes[0]);
      nalu_hypre_MPI_Recv(nalu_hypre_MPI_BOTTOM, 1, csr_matrix_datatypes[0], 0, 0, comm, &status0);
      nalu_hypre_MPI_Type_free(csr_matrix_datatypes);
   }

   first_col_diag = nalu_hypre_ParCSRMatrixFirstColDiag(parcsr_A);
   last_col_diag  = nalu_hypre_ParCSRMatrixLastColDiag(parcsr_A);

   GenerateDiagAndOffd(local_A, parcsr_A, first_col_diag, last_col_diag);

   /* set pointers back to NULL before destroying */
   if (my_id == 0)
   {
      nalu_hypre_CSRMatrixData(local_A) = NULL;
      nalu_hypre_CSRMatrixI(local_A) = NULL;
      nalu_hypre_CSRMatrixJ(local_A) = NULL;
   }
   nalu_hypre_CSRMatrixDestroy(local_A);
   nalu_hypre_TFree(csr_matrix_datatypes, NALU_HYPRE_MEMORY_HOST);

   return parcsr_A;
}

/* RL: XXX this is not a scalable routine, see `marker' therein */
NALU_HYPRE_Int
GenerateDiagAndOffd(nalu_hypre_CSRMatrix    *A,
                    nalu_hypre_ParCSRMatrix *matrix,
                    NALU_HYPRE_BigInt        first_col_diag,
                    NALU_HYPRE_BigInt        last_col_diag)
{
   NALU_HYPRE_Int  i, j;
   NALU_HYPRE_Int  jo, jd;
   NALU_HYPRE_Int  num_rows = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int  num_cols = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Complex *a_data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int *a_i = nalu_hypre_CSRMatrixI(A);
   /*RL: XXX FIXME if A spans global column space, the following a_j should be bigJ */
   NALU_HYPRE_Int *a_j = nalu_hypre_CSRMatrixJ(A);

   nalu_hypre_CSRMatrix *diag = nalu_hypre_ParCSRMatrixDiag(matrix);
   nalu_hypre_CSRMatrix *offd = nalu_hypre_ParCSRMatrixOffd(matrix);

   NALU_HYPRE_BigInt  *col_map_offd;

   NALU_HYPRE_Complex *diag_data, *offd_data;
   NALU_HYPRE_Int  *diag_i, *offd_i;
   NALU_HYPRE_Int  *diag_j, *offd_j;
   NALU_HYPRE_Int  *marker;
   NALU_HYPRE_Int num_cols_diag, num_cols_offd;
   NALU_HYPRE_Int first_elmt = a_i[0];
   NALU_HYPRE_Int num_nonzeros = a_i[num_rows] - first_elmt;
   NALU_HYPRE_Int counter;

   num_cols_diag = (NALU_HYPRE_Int)(last_col_diag - first_col_diag + 1);
   num_cols_offd = 0;

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_CSRMatrixMemoryLocation(A);

   if (num_cols - num_cols_diag)
   {
      nalu_hypre_CSRMatrixInitialize_v2(diag, 0, memory_location);
      diag_i = nalu_hypre_CSRMatrixI(diag);

      nalu_hypre_CSRMatrixInitialize_v2(offd, 0, memory_location);
      offd_i = nalu_hypre_CSRMatrixI(offd);
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
         {
            if (a_j[j] < first_col_diag || a_j[j] > last_col_diag)
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
      }
      offd_i[num_rows] = jo;
      diag_i[num_rows] = jd;

      nalu_hypre_ParCSRMatrixColMapOffd(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd,
                                                           NALU_HYPRE_MEMORY_HOST);
      col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(matrix);

      counter = 0;
      for (i = 0; i < num_cols; i++)
      {
         if (marker[i])
         {
            col_map_offd[counter] = (NALU_HYPRE_BigInt) i;
            marker[i] = counter;
            counter++;
         }
      }

      nalu_hypre_CSRMatrixNumNonzeros(diag) = jd;
      nalu_hypre_CSRMatrixInitialize(diag);
      diag_data = nalu_hypre_CSRMatrixData(diag);
      diag_j = nalu_hypre_CSRMatrixJ(diag);

      nalu_hypre_CSRMatrixNumNonzeros(offd) = jo;
      nalu_hypre_CSRMatrixNumCols(offd) = num_cols_offd;
      nalu_hypre_CSRMatrixInitialize(offd);
      offd_data = nalu_hypre_CSRMatrixData(offd);
      offd_j = nalu_hypre_CSRMatrixJ(offd);

      jo = 0;
      jd = 0;
      for (i = 0; i < num_rows; i++)
      {
         for (j = a_i[i] - first_elmt; j < a_i[i + 1] - first_elmt; j++)
         {
            if (a_j[j] < (NALU_HYPRE_Int)first_col_diag || a_j[j] > (NALU_HYPRE_Int)last_col_diag)
            {
               offd_data[jo] = a_data[j];
               offd_j[jo++] = marker[a_j[j]];
            }
            else
            {
               diag_data[jd] = a_data[j];
               diag_j[jd++] = (NALU_HYPRE_Int)(a_j[j] - first_col_diag);
            }
         }
      }
      nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);
   }
   else
   {
      nalu_hypre_CSRMatrixNumNonzeros(diag) = num_nonzeros;
      nalu_hypre_CSRMatrixInitialize(diag);
      diag_data = nalu_hypre_CSRMatrixData(diag);
      diag_i = nalu_hypre_CSRMatrixI(diag);
      diag_j = nalu_hypre_CSRMatrixJ(diag);

      for (i = 0; i < num_nonzeros; i++)
      {
         diag_data[i] = a_data[i];
         diag_j[i] = a_j[i];
      }
      offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_rows + 1, NALU_HYPRE_MEMORY_HOST);

      for (i = 0; i < num_rows + 1; i++)
      {
         diag_i[i] = a_i[i];
         offd_i[i] = 0;
      }

      nalu_hypre_CSRMatrixNumCols(offd) = 0;
      nalu_hypre_CSRMatrixI(offd) = offd_i;
   }

   return nalu_hypre_error_flag;
}

nalu_hypre_CSRMatrix *
nalu_hypre_MergeDiagAndOffd(nalu_hypre_ParCSRMatrix *par_matrix)
{
   nalu_hypre_CSRMatrix  *diag = nalu_hypre_ParCSRMatrixDiag(par_matrix);
   nalu_hypre_CSRMatrix  *offd = nalu_hypre_ParCSRMatrixOffd(par_matrix);
   nalu_hypre_CSRMatrix  *matrix;

   NALU_HYPRE_BigInt       num_cols = nalu_hypre_ParCSRMatrixGlobalNumCols(par_matrix);
   NALU_HYPRE_BigInt       first_col_diag = nalu_hypre_ParCSRMatrixFirstColDiag(par_matrix);
   NALU_HYPRE_BigInt      *col_map_offd = nalu_hypre_ParCSRMatrixColMapOffd(par_matrix);
   NALU_HYPRE_Int          num_rows = nalu_hypre_CSRMatrixNumRows(diag);

   NALU_HYPRE_Int          *diag_i = nalu_hypre_CSRMatrixI(diag);
   NALU_HYPRE_Int          *diag_j = nalu_hypre_CSRMatrixJ(diag);
   NALU_HYPRE_Complex      *diag_data = nalu_hypre_CSRMatrixData(diag);
   NALU_HYPRE_Int          *offd_i = nalu_hypre_CSRMatrixI(offd);
   NALU_HYPRE_Int          *offd_j = nalu_hypre_CSRMatrixJ(offd);
   NALU_HYPRE_Complex      *offd_data = nalu_hypre_CSRMatrixData(offd);

   NALU_HYPRE_Int          *matrix_i;
   NALU_HYPRE_BigInt       *matrix_j;
   NALU_HYPRE_Complex      *matrix_data;

   NALU_HYPRE_Int          num_nonzeros, i, j;
   NALU_HYPRE_Int          count;
   NALU_HYPRE_Int          size, rest, num_threads, ii;

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(par_matrix);

   num_nonzeros = diag_i[num_rows] + offd_i[num_rows];

   matrix = nalu_hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   nalu_hypre_CSRMatrixMemoryLocation(matrix) = memory_location;
   nalu_hypre_CSRMatrixBigInitialize(matrix);

   matrix_i = nalu_hypre_CSRMatrixI(matrix);
   matrix_j = nalu_hypre_CSRMatrixBigJ(matrix);
   matrix_data = nalu_hypre_CSRMatrixData(matrix);
   num_threads = nalu_hypre_NumThreads();
   size = num_rows / num_threads;
   rest = num_rows - size * num_threads;

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(ii, i, j, count) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (ii = 0; ii < num_threads; ii++)
   {
      NALU_HYPRE_Int ns, ne;
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
      count = diag_i[ns] + offd_i[ns];;
      for (i = ns; i < ne; i++)
      {
         matrix_i[i] = count;
         for (j = diag_i[i]; j < diag_i[i + 1]; j++)
         {
            matrix_data[count] = diag_data[j];
            matrix_j[count++] = (NALU_HYPRE_BigInt)diag_j[j] + first_col_diag;
         }
         for (j = offd_i[i]; j < offd_i[i + 1]; j++)
         {
            matrix_data[count] = offd_data[j];
            matrix_j[count++] = col_map_offd[offd_j[j]];
         }
      }
   } /* end parallel region */

   matrix_i[num_rows] = num_nonzeros;

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixToCSRMatrixAll:
 * generates a CSRMatrix from a ParCSRMatrix on all processors that have
 * parts of the ParCSRMatrix
 * Warning: this only works for a ParCSRMatrix that is smaller than 2^31-1
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix *
nalu_hypre_ParCSRMatrixToCSRMatrixAll(nalu_hypre_ParCSRMatrix *par_matrix)
{
   MPI_Comm comm = nalu_hypre_ParCSRMatrixComm(par_matrix);
   nalu_hypre_CSRMatrix *matrix;
   nalu_hypre_CSRMatrix *local_matrix;
   NALU_HYPRE_Int num_rows = (NALU_HYPRE_Int)nalu_hypre_ParCSRMatrixGlobalNumRows(par_matrix);
   NALU_HYPRE_Int num_cols = (NALU_HYPRE_Int)nalu_hypre_ParCSRMatrixGlobalNumCols(par_matrix);
   NALU_HYPRE_Int *matrix_i;
   NALU_HYPRE_Int *matrix_j;
   NALU_HYPRE_Complex *matrix_data;

   NALU_HYPRE_Int *local_matrix_i;
   NALU_HYPRE_Int *local_matrix_j;
   NALU_HYPRE_Complex *local_matrix_data;

   NALU_HYPRE_Int i, j;
   NALU_HYPRE_Int local_num_rows;
   NALU_HYPRE_Int local_num_nonzeros;
   NALU_HYPRE_Int num_nonzeros;
   NALU_HYPRE_Int num_data;
   NALU_HYPRE_Int num_requests;
   NALU_HYPRE_Int vec_len, offset;
   NALU_HYPRE_Int start_index;
   NALU_HYPRE_Int proc_id;
   NALU_HYPRE_Int num_procs, my_id;
   NALU_HYPRE_Int num_types;
   NALU_HYPRE_Int *used_procs;

   nalu_hypre_MPI_Request *requests;
   nalu_hypre_MPI_Status *status;

   NALU_HYPRE_Int *new_vec_starts;

   NALU_HYPRE_Int num_contacts;
   NALU_HYPRE_Int contact_proc_list[1];
   NALU_HYPRE_Int contact_send_buf[1];
   NALU_HYPRE_Int contact_send_buf_starts[2];
   NALU_HYPRE_Int max_response_size;
   NALU_HYPRE_Int *response_recv_buf = NULL;
   NALU_HYPRE_Int *response_recv_buf_starts = NULL;
   nalu_hypre_DataExchangeResponse response_obj;
   nalu_hypre_ProcListElements send_proc_obj;

   NALU_HYPRE_Int *send_info = NULL;
   nalu_hypre_MPI_Status  status1;
   NALU_HYPRE_Int count, tag1 = 11112, tag2 = 22223, tag3 = 33334;
   NALU_HYPRE_Int start;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   local_num_rows = (NALU_HYPRE_Int)(nalu_hypre_ParCSRMatrixLastRowIndex(par_matrix)  -
                                nalu_hypre_ParCSRMatrixFirstRowIndex(par_matrix) + 1);


   local_matrix = nalu_hypre_MergeDiagAndOffd(par_matrix); /* creates matrix */
   nalu_hypre_CSRMatrixBigJtoJ(local_matrix); /* copies big_j to j */
   local_matrix_i = nalu_hypre_CSRMatrixI(local_matrix);
   local_matrix_j = nalu_hypre_CSRMatrixJ(local_matrix);
   local_matrix_data = nalu_hypre_CSRMatrixData(local_matrix);


   /* determine procs that have vector data and store their ids in used_procs */
   /* we need to do an exchange data for this.  If I own row then I will contact
      processor 0 with the endpoint of my local range */

   if (local_num_rows > 0)
   {
      num_contacts = 1;
      contact_proc_list[0] = 0;
      contact_send_buf[0] =  (NALU_HYPRE_Int)nalu_hypre_ParCSRMatrixLastRowIndex(par_matrix);
      contact_send_buf_starts[0] = 0;
      contact_send_buf_starts[1] = 1;

   }
   else
   {
      num_contacts = 0;
      contact_send_buf_starts[0] = 0;
      contact_send_buf_starts[1] = 0;
   }
   /*build the response object*/
   /*send_proc_obj will  be for saving info from contacts */
   send_proc_obj.length = 0;
   send_proc_obj.storage_length = 10;
   send_proc_obj.id = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  send_proc_obj.storage_length, NALU_HYPRE_MEMORY_HOST);
   send_proc_obj.vec_starts =
      nalu_hypre_CTAlloc(NALU_HYPRE_Int,  send_proc_obj.storage_length + 1, NALU_HYPRE_MEMORY_HOST);
   send_proc_obj.vec_starts[0] = 0;
   send_proc_obj.element_storage_length = 10;
   send_proc_obj.elements =
      nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  send_proc_obj.element_storage_length, NALU_HYPRE_MEMORY_HOST);

   max_response_size = 0; /* each response is null */
   response_obj.fill_response = nalu_hypre_FillResponseParToCSRMatrix;
   response_obj.data1 = NULL;
   response_obj.data2 = &send_proc_obj; /*this is where we keep info from contacts*/


   nalu_hypre_DataExchangeList(num_contacts,
                          contact_proc_list, contact_send_buf,
                          contact_send_buf_starts, sizeof(NALU_HYPRE_Int),
                          sizeof(NALU_HYPRE_Int), &response_obj,
                          max_response_size, 1,
                          comm, (void**) &response_recv_buf,
                          &response_recv_buf_starts);

   /* now processor 0 should have a list of ranges for processors that have rows -
      these are in send_proc_obj - it needs to create the new list of processors
      and also an array of vec starts - and send to those who own row*/
   if (my_id)
   {
      if (local_num_rows)
      {
         /* look for a message from processor 0 */
         nalu_hypre_MPI_Probe(0, tag1, comm, &status1);
         nalu_hypre_MPI_Get_count(&status1, NALU_HYPRE_MPI_INT, &count);

         send_info = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  count, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_MPI_Recv(send_info, count, NALU_HYPRE_MPI_INT, 0, tag1, comm, &status1);

         /* now unpack */
         num_types = send_info[0];
         used_procs =  nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_types, NALU_HYPRE_MEMORY_HOST);
         new_vec_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_types + 1, NALU_HYPRE_MEMORY_HOST);

         for (i = 1; i <= num_types; i++)
         {
            used_procs[i - 1] = send_info[i];
         }
         for (i = num_types + 1; i < count; i++)
         {
            new_vec_starts[i - num_types - 1] = send_info[i] ;
         }
      }
      else /* clean up and exit */
      {
         nalu_hypre_TFree(send_proc_obj.vec_starts, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(send_proc_obj.id, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(send_proc_obj.elements, NALU_HYPRE_MEMORY_HOST);
         if (response_recv_buf) { nalu_hypre_TFree(response_recv_buf, NALU_HYPRE_MEMORY_HOST); }
         if (response_recv_buf_starts) { nalu_hypre_TFree(response_recv_buf_starts, NALU_HYPRE_MEMORY_HOST); }


         if (nalu_hypre_CSRMatrixOwnsData(local_matrix))
         {
            nalu_hypre_CSRMatrixDestroy(local_matrix);
         }
         else
         {
            nalu_hypre_TFree(local_matrix, NALU_HYPRE_MEMORY_HOST);
         }

         return NULL;
      }
   }
   else /* my_id ==0 */
   {
      num_types = send_proc_obj.length;
      used_procs =  nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_types, NALU_HYPRE_MEMORY_HOST);
      new_vec_starts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_types + 1, NALU_HYPRE_MEMORY_HOST);

      new_vec_starts[0] = 0;
      for (i = 0; i < num_types; i++)
      {
         used_procs[i] = send_proc_obj.id[i];
         new_vec_starts[i + 1] = send_proc_obj.elements[i] + 1;
      }
      nalu_hypre_qsort0(used_procs, 0, num_types - 1);
      nalu_hypre_qsort0(new_vec_starts, 0, num_types);
      /*now we need to put into an array to send */
      count =  2 * num_types + 2;
      send_info = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  count, NALU_HYPRE_MEMORY_HOST);
      send_info[0] = num_types;
      for (i = 1; i <= num_types; i++)
      {
         send_info[i] = (NALU_HYPRE_BigInt)used_procs[i - 1];
      }
      for (i = num_types + 1; i < count; i++)
      {
         send_info[i] = new_vec_starts[i - num_types - 1];
      }
      requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request,  num_types, NALU_HYPRE_MEMORY_HOST);
      status =  nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,  num_types, NALU_HYPRE_MEMORY_HOST);

      /* don't send to myself  - these are sorted so my id would be first*/
      start = 0;
      if (num_types && used_procs[0] == 0)
      {
         start = 1;
      }

      for (i = start; i < num_types; i++)
      {
         nalu_hypre_MPI_Isend(send_info, count, NALU_HYPRE_MPI_INT, used_procs[i], tag1,
                         comm, &requests[i - start]);
      }
      nalu_hypre_MPI_Waitall(num_types - start, requests, status);

      nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
   }
   /* clean up */
   nalu_hypre_TFree(send_proc_obj.vec_starts, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_proc_obj.id, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_proc_obj.elements, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(send_info, NALU_HYPRE_MEMORY_HOST);
   if (response_recv_buf) { nalu_hypre_TFree(response_recv_buf, NALU_HYPRE_MEMORY_HOST); }
   if (response_recv_buf_starts) { nalu_hypre_TFree(response_recv_buf_starts, NALU_HYPRE_MEMORY_HOST); }

   /* now proc 0 can exit if it has no rows */
   if (!local_num_rows)
   {
      if (nalu_hypre_CSRMatrixOwnsData(local_matrix))
      {
         nalu_hypre_CSRMatrixDestroy(local_matrix);
      }
      else
      {
         nalu_hypre_TFree(local_matrix, NALU_HYPRE_MEMORY_HOST);
      }

      nalu_hypre_TFree(new_vec_starts, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(used_procs, NALU_HYPRE_MEMORY_HOST);

      return NULL;
   }

   /* everyone left has rows and knows: new_vec_starts, num_types, and used_procs */

   /* this matrix should be rather small */
   matrix_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_rows + 1, NALU_HYPRE_MEMORY_HOST);

   num_requests = 4 * num_types;
   requests = nalu_hypre_CTAlloc(nalu_hypre_MPI_Request,  num_requests, NALU_HYPRE_MEMORY_HOST);
   status = nalu_hypre_CTAlloc(nalu_hypre_MPI_Status,  num_requests, NALU_HYPRE_MEMORY_HOST);

   /* exchange contents of local_matrix_i - here we are sending to ourself also*/

   j = 0;
   for (i = 0; i < num_types; i++)
   {
      proc_id = used_procs[i];
      vec_len = (NALU_HYPRE_Int)(new_vec_starts[i + 1] - new_vec_starts[i]);
      nalu_hypre_MPI_Irecv(&matrix_i[new_vec_starts[i] + 1], vec_len, NALU_HYPRE_MPI_INT,
                      proc_id, tag2, comm, &requests[j++]);
   }
   for (i = 0; i < num_types; i++)
   {
      proc_id = used_procs[i];
      nalu_hypre_MPI_Isend(&local_matrix_i[1], local_num_rows, NALU_HYPRE_MPI_INT,
                      proc_id, tag2, comm, &requests[j++]);
   }

   nalu_hypre_MPI_Waitall(j, requests, status);

   /* generate matrix_i from received data */
   /* global numbering?*/
   offset = matrix_i[new_vec_starts[1]];
   for (i = 1; i < num_types; i++)
   {
      for (j = new_vec_starts[i]; j < new_vec_starts[i + 1]; j++)
      {
         matrix_i[j + 1] += offset;
      }
      offset = matrix_i[new_vec_starts[i + 1]];
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = nalu_hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);

   nalu_hypre_CSRMatrixMemoryLocation(matrix) = NALU_HYPRE_MEMORY_HOST;

   nalu_hypre_CSRMatrixI(matrix) = matrix_i;
   nalu_hypre_CSRMatrixInitialize(matrix);
   matrix_j = nalu_hypre_CSRMatrixJ(matrix);
   matrix_data = nalu_hypre_CSRMatrixData(matrix);

   /* generate datatypes for further data exchange and exchange remaining
      data, i.e. column info and actual data */

   j = 0;
   for (i = 0; i < num_types; i++)
   {
      proc_id = used_procs[i];
      start_index = matrix_i[(NALU_HYPRE_Int)new_vec_starts[i]];
      num_data = matrix_i[(NALU_HYPRE_Int)new_vec_starts[i + 1]] - start_index;
      nalu_hypre_MPI_Irecv(&matrix_data[start_index], num_data, NALU_HYPRE_MPI_COMPLEX,
                      used_procs[i], tag1, comm, &requests[j++]);
      nalu_hypre_MPI_Irecv(&matrix_j[start_index], num_data, NALU_HYPRE_MPI_INT,
                      used_procs[i], tag3, comm, &requests[j++]);
   }
   local_num_nonzeros = local_matrix_i[local_num_rows];
   for (i = 0; i < num_types; i++)
   {
      nalu_hypre_MPI_Isend(local_matrix_data, local_num_nonzeros, NALU_HYPRE_MPI_COMPLEX,
                      used_procs[i], tag1, comm, &requests[j++]);
      nalu_hypre_MPI_Isend(local_matrix_j, local_num_nonzeros, NALU_HYPRE_MPI_INT,
                      used_procs[i], tag3, comm, &requests[j++]);
   }


   nalu_hypre_MPI_Waitall(num_requests, requests, status);

   nalu_hypre_TFree(new_vec_starts, NALU_HYPRE_MEMORY_HOST);

   if (nalu_hypre_CSRMatrixOwnsData(local_matrix))
   {
      nalu_hypre_CSRMatrixDestroy(local_matrix);
   }
   else
   {
      nalu_hypre_TFree(local_matrix, NALU_HYPRE_MEMORY_HOST);
   }

   if (num_requests)
   {
      nalu_hypre_TFree(requests, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(status, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(used_procs, NALU_HYPRE_MEMORY_HOST);
   }

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixCopy,
 * copies B to A,
 * if copy_data = 0, only the structure of A is copied to B
 * the routine does not check whether the dimensions of A and B are compatible
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixCopy( nalu_hypre_ParCSRMatrix *A,
                        nalu_hypre_ParCSRMatrix *B,
                        NALU_HYPRE_Int copy_data )
{
   nalu_hypre_CSRMatrix *A_diag;
   nalu_hypre_CSRMatrix *A_offd;
   NALU_HYPRE_BigInt *col_map_offd_A;
   nalu_hypre_CSRMatrix *B_diag;
   nalu_hypre_CSRMatrix *B_offd;
   NALU_HYPRE_BigInt *col_map_offd_B;
   NALU_HYPRE_Int num_cols_offd_A;
   NALU_HYPRE_Int num_cols_offd_B;

   if (!A)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   if (!B)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   B_diag = nalu_hypre_ParCSRMatrixDiag(B);
   B_offd = nalu_hypre_ParCSRMatrixOffd(B);

   num_cols_offd_A = nalu_hypre_CSRMatrixNumCols(A_offd);
   num_cols_offd_B = nalu_hypre_CSRMatrixNumCols(B_offd);

   nalu_hypre_assert(num_cols_offd_A == num_cols_offd_B);

   col_map_offd_A = nalu_hypre_ParCSRMatrixColMapOffd(A);
   col_map_offd_B = nalu_hypre_ParCSRMatrixColMapOffd(B);

   nalu_hypre_CSRMatrixCopy(A_diag, B_diag, copy_data);
   nalu_hypre_CSRMatrixCopy(A_offd, B_offd, copy_data);

   /* should not happen if B has been initialized */
   if (num_cols_offd_B && col_map_offd_B == NULL)
   {
      col_map_offd_B = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_offd_B, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParCSRMatrixColMapOffd(B) = col_map_offd_B;
   }

   nalu_hypre_TMemcpy(col_map_offd_B, col_map_offd_A, NALU_HYPRE_BigInt, num_cols_offd_B,
                 NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------
 * nalu_hypre_FillResponseParToCSRMatrix
 * Fill response function for determining the send processors
 * data exchange
 *--------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FillResponseParToCSRMatrix( void       *p_recv_contact_buf,
                                  NALU_HYPRE_Int   contact_size,
                                  NALU_HYPRE_Int   contact_proc,
                                  void       *ro,
                                  MPI_Comm    comm,
                                  void      **p_send_response_buf,
                                  NALU_HYPRE_Int *response_message_size )
{
   NALU_HYPRE_Int    myid;
   NALU_HYPRE_Int    i, index, count, elength;

   NALU_HYPRE_BigInt    *recv_contact_buf = (NALU_HYPRE_BigInt * ) p_recv_contact_buf;

   nalu_hypre_DataExchangeResponse  *response_obj = (nalu_hypre_DataExchangeResponse*)ro;

   nalu_hypre_ProcListElements      *send_proc_obj = (nalu_hypre_ProcListElements*)response_obj->data2;

   nalu_hypre_MPI_Comm_rank(comm, &myid );

   /*check to see if we need to allocate more space in send_proc_obj for ids*/
   if (send_proc_obj->length == send_proc_obj->storage_length)
   {
      send_proc_obj->storage_length += 10; /*add space for 10 more processors*/
      send_proc_obj->id = nalu_hypre_TReAlloc(send_proc_obj->id, NALU_HYPRE_Int,
                                         send_proc_obj->storage_length, NALU_HYPRE_MEMORY_HOST);
      send_proc_obj->vec_starts =
         nalu_hypre_TReAlloc(send_proc_obj->vec_starts, NALU_HYPRE_Int,
                        send_proc_obj->storage_length + 1, NALU_HYPRE_MEMORY_HOST);
   }

   /*initialize*/
   count = send_proc_obj->length;
   index = send_proc_obj->vec_starts[count]; /*this is the number of elements*/

   /*send proc*/
   send_proc_obj->id[count] = contact_proc;

   /*do we need more storage for the elements?*/
   if (send_proc_obj->element_storage_length < index + contact_size)
   {
      elength = nalu_hypre_max(contact_size, 10);
      elength += index;
      send_proc_obj->elements = nalu_hypre_TReAlloc(send_proc_obj->elements,
                                               NALU_HYPRE_BigInt,  elength, NALU_HYPRE_MEMORY_HOST);
      send_proc_obj->element_storage_length = elength;
   }
   /*populate send_proc_obj*/
   for (i = 0; i < contact_size; i++)
   {
      send_proc_obj->elements[index++] = recv_contact_buf[i];
   }
   send_proc_obj->vec_starts[count + 1] = index;
   send_proc_obj->length++;

   /*output - no message to return (confirmation) */
   *response_message_size = 0;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixUnion
 * Creates and returns a new matrix whose elements are the union of A and B.
 * Data is not copied, only structural information is created.
 * A and B must have the same communicator, numbers and distributions of rows
 * and columns (they can differ in which row-column pairs are nonzero, thus
 * in which columns are in a offd block)
 *--------------------------------------------------------------------------*/

nalu_hypre_ParCSRMatrix * nalu_hypre_ParCSRMatrixUnion( nalu_hypre_ParCSRMatrix * A,
                                              nalu_hypre_ParCSRMatrix * B )
{
   nalu_hypre_ParCSRMatrix *C;
   NALU_HYPRE_BigInt       *col_map_offd_C = NULL;
   NALU_HYPRE_Int           my_id, p;
   MPI_Comm            comm = nalu_hypre_ParCSRMatrixComm( A );

   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   C = nalu_hypre_CTAlloc( nalu_hypre_ParCSRMatrix,  1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_ParCSRMatrixComm( C ) = nalu_hypre_ParCSRMatrixComm( A );
   nalu_hypre_ParCSRMatrixGlobalNumRows( C ) = nalu_hypre_ParCSRMatrixGlobalNumRows( A );
   nalu_hypre_ParCSRMatrixGlobalNumCols( C ) = nalu_hypre_ParCSRMatrixGlobalNumCols( A );
   nalu_hypre_ParCSRMatrixFirstRowIndex( C ) = nalu_hypre_ParCSRMatrixFirstRowIndex( A );
   nalu_hypre_assert( nalu_hypre_ParCSRMatrixFirstRowIndex( B )
                 == nalu_hypre_ParCSRMatrixFirstRowIndex( A ) );
   nalu_hypre_TMemcpy(nalu_hypre_ParCSRMatrixRowStarts(C), nalu_hypre_ParCSRMatrixRowStarts(A),
                 NALU_HYPRE_BigInt, 2, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TMemcpy(nalu_hypre_ParCSRMatrixColStarts(C), nalu_hypre_ParCSRMatrixColStarts(A),
                 NALU_HYPRE_BigInt, 2, NALU_HYPRE_MEMORY_HOST, NALU_HYPRE_MEMORY_HOST);
   for (p = 0; p < 2; ++p)
      nalu_hypre_assert( nalu_hypre_ParCSRMatrixColStarts(A)[p]
                    == nalu_hypre_ParCSRMatrixColStarts(B)[p] );
   nalu_hypre_ParCSRMatrixFirstColDiag( C ) = nalu_hypre_ParCSRMatrixFirstColDiag( A );
   nalu_hypre_ParCSRMatrixLastRowIndex( C ) = nalu_hypre_ParCSRMatrixLastRowIndex( A );
   nalu_hypre_ParCSRMatrixLastColDiag( C ) = nalu_hypre_ParCSRMatrixLastColDiag( A );

   nalu_hypre_ParCSRMatrixDiag( C ) =
      nalu_hypre_CSRMatrixUnion( nalu_hypre_ParCSRMatrixDiag(A), nalu_hypre_ParCSRMatrixDiag(B),
                            0, 0, 0 );
   nalu_hypre_ParCSRMatrixOffd( C ) =
      nalu_hypre_CSRMatrixUnion( nalu_hypre_ParCSRMatrixOffd(A), nalu_hypre_ParCSRMatrixOffd(B),
                            nalu_hypre_ParCSRMatrixColMapOffd(A),
                            nalu_hypre_ParCSRMatrixColMapOffd(B), &col_map_offd_C );
   nalu_hypre_ParCSRMatrixColMapOffd( C ) = col_map_offd_C;
   nalu_hypre_ParCSRMatrixCommPkg( C ) = NULL;
   nalu_hypre_ParCSRMatrixCommPkgT( C ) = NULL;
   nalu_hypre_ParCSRMatrixOwnsData( C ) = 1;
   /*  SetNumNonzeros, SetDNumNonzeros are global, need nalu_hypre_MPI_Allreduce.
       I suspect, but don't know, that other parts of hypre do not assume that
       the correct values have been set.
       nalu_hypre_ParCSRMatrixSetNumNonzeros( C );
       nalu_hypre_ParCSRMatrixSetDNumNonzeros( C );*/
   nalu_hypre_ParCSRMatrixNumNonzeros( C ) = 0;
   nalu_hypre_ParCSRMatrixDNumNonzeros( C ) = 0.0;
   nalu_hypre_ParCSRMatrixRowindices( C ) = NULL;
   nalu_hypre_ParCSRMatrixRowvalues( C ) = NULL;
   nalu_hypre_ParCSRMatrixGetrowactive( C ) = 0;

   return C;
}

/* Perform dual truncation of ParCSR matrix.
 * This code is adapted from original BoomerAMGInterpTruncate()
 * A: parCSR matrix to be modified
 * tol: relative tolerance or truncation factor for dropping small terms
 * max_row_elmts: maximum number of (largest) nonzero elements to keep.
 * rescale: Boolean on whether or not to scale resulting matrix. Scaling for
 * each row satisfies: sum(nonzero values before dropping)/ sum(nonzero values after dropping),
 * this way, the application of the truncated matrix on a constant vector is the same as that of
 * the original matrix.
 * nrm_type: type of norm used for dropping with tol.
 * -- 0 = infinity-norm
 * -- 1 = 1-norm
 * -- 2 = 2-norm
*/
NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixTruncate(nalu_hypre_ParCSRMatrix *A,
                           NALU_HYPRE_Real          tol,
                           NALU_HYPRE_Int           max_row_elmts,
                           NALU_HYPRE_Int           rescale,
                           NALU_HYPRE_Int           nrm_type)
{
#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_INTERP_TRUNC] -= nalu_hypre_MPI_Wtime();
#endif

   nalu_hypre_CSRMatrix *A_diag = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int *A_diag_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int *A_diag_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Real *A_diag_data = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int *A_diag_j_new;
   NALU_HYPRE_Real *A_diag_data_new;

   nalu_hypre_CSRMatrix *A_offd = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int *A_offd_i = nalu_hypre_CSRMatrixI(A_offd);
   NALU_HYPRE_Int *A_offd_j = nalu_hypre_CSRMatrixJ(A_offd);
   NALU_HYPRE_Real *A_offd_data = nalu_hypre_CSRMatrixData(A_offd);
   NALU_HYPRE_Int *A_offd_j_new;
   NALU_HYPRE_Real *A_offd_data_new;

   NALU_HYPRE_Int n_fine = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int num_cols = nalu_hypre_CSRMatrixNumCols(A_diag);
   NALU_HYPRE_Int i, j, start_j;
   NALU_HYPRE_Int ierr = 0;
   NALU_HYPRE_Int next_open;
   NALU_HYPRE_Int now_checking;
   NALU_HYPRE_Int num_lost;
   NALU_HYPRE_Int num_lost_global = 0;
   NALU_HYPRE_Int next_open_offd;
   NALU_HYPRE_Int now_checking_offd;
   NALU_HYPRE_Int num_lost_offd;
   NALU_HYPRE_Int num_lost_global_offd;
   NALU_HYPRE_Int A_diag_size;
   NALU_HYPRE_Int A_offd_size;
   NALU_HYPRE_Int num_elmts;
   NALU_HYPRE_Int cnt, cnt_diag, cnt_offd;
   NALU_HYPRE_Real row_nrm;
   NALU_HYPRE_Real drop_coeff;
   NALU_HYPRE_Real row_sum;
   NALU_HYPRE_Real scale;

   NALU_HYPRE_MemoryLocation memory_location_diag = nalu_hypre_CSRMatrixMemoryLocation(A_diag);
   NALU_HYPRE_MemoryLocation memory_location_offd = nalu_hypre_CSRMatrixMemoryLocation(A_offd);

   /* Threading variables.  Entry i of num_lost_(offd_)per_thread  holds the
    * number of dropped entries over thread i's row range. Cum_lost_per_thread
    * will temporarily store the cumulative number of dropped entries up to
    * each thread. */
   NALU_HYPRE_Int my_thread_num, num_threads, start, stop;
   NALU_HYPRE_Int * max_num_threads = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  1, NALU_HYPRE_MEMORY_HOST);
   NALU_HYPRE_Int * cum_lost_per_thread;
   NALU_HYPRE_Int * num_lost_per_thread;
   NALU_HYPRE_Int * num_lost_offd_per_thread;

   /* Initialize threading variables */
   max_num_threads[0] = nalu_hypre_NumThreads();
   cum_lost_per_thread = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_num_threads[0], NALU_HYPRE_MEMORY_HOST);
   num_lost_per_thread = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_num_threads[0], NALU_HYPRE_MEMORY_HOST);
   num_lost_offd_per_thread = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  max_num_threads[0], NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < max_num_threads[0]; i++)
   {
      num_lost_per_thread[i] = 0;
      num_lost_offd_per_thread[i] = 0;
   }

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel private(i,my_thread_num,num_threads,row_nrm, drop_coeff,j,start_j,row_sum,scale,num_lost,now_checking,next_open,num_lost_offd,now_checking_offd,next_open_offd,start,stop,cnt_diag,cnt_offd,num_elmts,cnt)
#endif
   {
      my_thread_num = nalu_hypre_GetThreadNum();
      num_threads = nalu_hypre_NumActiveThreads();

      /* Compute each thread's range of rows to truncate and compress.  Note,
       * that i, j and data are all compressed as entries are dropped, but
       * that the compression only occurs locally over each thread's row
       * range.  A_diag_i is only made globally consistent at the end of this
       * routine.  During the dropping phases, A_diag_i[stop] will point to
       * the start of the next thread's row range.  */

      /* my row range */
      start = (n_fine / num_threads) * my_thread_num;
      if (my_thread_num == num_threads - 1)
      {
         stop = n_fine;
      }
      else
      {
         stop = (n_fine / num_threads) * (my_thread_num + 1);
      }

      /*
       * Truncate based on truncation tolerance
       */
      if (tol > 0)
      {
         num_lost = 0;
         num_lost_offd = 0;

         next_open = A_diag_i[start];
         now_checking = A_diag_i[start];
         next_open_offd = A_offd_i[start];;
         now_checking_offd = A_offd_i[start];;

         for (i = start; i < stop; i++)
         {
            row_nrm = 0;
            /* compute norm for dropping small terms */
            if (nrm_type == 0)
            {
               /* infty-norm */
               for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
               {
                  row_nrm = (row_nrm < nalu_hypre_cabs(A_diag_data[j])) ?
                            nalu_hypre_cabs(A_diag_data[j]) : row_nrm;
               }
               for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
               {
                  row_nrm = (row_nrm < nalu_hypre_cabs(A_offd_data[j])) ?
                            nalu_hypre_cabs(A_offd_data[j]) : row_nrm;
               }
            }
            if (nrm_type == 1)
            {
               /* 1-norm */
               for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
               {
                  row_nrm += nalu_hypre_cabs(A_diag_data[j]);
               }
               for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
               {
                  row_nrm += nalu_hypre_cabs(A_offd_data[j]);
               }
            }
            if (nrm_type == 2)
            {
               /* 2-norm */
               for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
               {
                  NALU_HYPRE_Complex v = A_diag_data[j];
                  row_nrm += v * v;
               }
               for (j = A_offd_i[i]; j < A_offd_i[i + 1]; j++)
               {
                  NALU_HYPRE_Complex v = A_offd_data[j];
                  row_nrm += v * v;
               }
               row_nrm  = sqrt(row_nrm);
            }
            drop_coeff = tol * row_nrm;

            start_j = A_diag_i[i];
            if (num_lost)
            {
               A_diag_i[i] -= num_lost;
            }
            row_sum = 0;
            scale = 0;
            for (j = start_j; j < A_diag_i[i + 1]; j++)
            {
               row_sum += A_diag_data[now_checking];
               if (nalu_hypre_cabs(A_diag_data[now_checking]) < drop_coeff)
               {
                  num_lost++;
                  now_checking++;
               }
               else
               {
                  scale += A_diag_data[now_checking];
                  A_diag_data[next_open] = A_diag_data[now_checking];
                  A_diag_j[next_open] = A_diag_j[now_checking];
                  now_checking++;
                  next_open++;
               }
            }

            start_j = A_offd_i[i];
            if (num_lost_offd)
            {
               A_offd_i[i] -= num_lost_offd;
            }

            for (j = start_j; j < A_offd_i[i + 1]; j++)
            {
               row_sum += A_offd_data[now_checking_offd];
               if (nalu_hypre_cabs(A_offd_data[now_checking_offd]) < drop_coeff)
               {
                  num_lost_offd++;
                  now_checking_offd++;
               }
               else
               {
                  scale += A_offd_data[now_checking_offd];
                  A_offd_data[next_open_offd] = A_offd_data[now_checking_offd];
                  A_offd_j[next_open_offd] = A_offd_j[now_checking_offd];
                  now_checking_offd++;
                  next_open_offd++;
               }
            }

            /* scale row of A */
            if (rescale && scale != 0.)
            {
               if (scale != row_sum)
               {
                  scale = row_sum / scale;
                  for (j = A_diag_i[i]; j < (A_diag_i[i + 1] - num_lost); j++)
                  {
                     A_diag_data[j] *= scale;
                  }
                  for (j = A_offd_i[i]; j < (A_offd_i[i + 1] - num_lost_offd); j++)
                  {
                     A_offd_data[j] *= scale;
                  }
               }
            }
         } /* end loop for (i = 0; i < n_fine; i++) */

         /* store number of dropped elements and number of threads */
         if (my_thread_num == 0)
         {
            max_num_threads[0] = num_threads;
         }
         num_lost_per_thread[my_thread_num] = num_lost;
         num_lost_offd_per_thread[my_thread_num] = num_lost_offd;

      } /* end if (trunc_factor > 0) */

      /*
       * Truncate based on capping the nnz per row
       *
       */
      if (max_row_elmts > 0)
      {
         NALU_HYPRE_Int A_mxnum, cnt1, last_index, last_index_offd;
         NALU_HYPRE_Int *A_aux_j;
         NALU_HYPRE_Real *A_aux_data;

         /* find maximum row length locally over this row range */
         A_mxnum = 0;
         for (i = start; i < stop; i++)
         {
            /* Note A_diag_i[stop] is the starting point for the next thread
             * in j and data, not the stop point for this thread */
            last_index = A_diag_i[i + 1];
            last_index_offd = A_offd_i[i + 1];
            if (i == stop - 1)
            {
               last_index -= num_lost_per_thread[my_thread_num];
               last_index_offd -= num_lost_offd_per_thread[my_thread_num];
            }
            cnt1 = last_index - A_diag_i[i] + last_index_offd - A_offd_i[i];
            if (cnt1 > A_mxnum)
            {
               A_mxnum = cnt1;
            }
         }

         /* Some rows exceed max_row_elmts, and require truncation.  Essentially,
          * each thread truncates and compresses its range of rows locally. */
         if (A_mxnum > max_row_elmts)
         {
            num_lost = 0;
            num_lost_offd = 0;

            /* two temporary arrays to hold row i for temporary operations */
            A_aux_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  A_mxnum, NALU_HYPRE_MEMORY_HOST);
            A_aux_data = nalu_hypre_CTAlloc(NALU_HYPRE_Real,  A_mxnum, NALU_HYPRE_MEMORY_HOST);
            cnt_diag = A_diag_i[start];
            cnt_offd = A_offd_i[start];

            for (i = start; i < stop; i++)
            {
               /* Note A_diag_i[stop] is the starting point for the next thread
                * in j and data, not the stop point for this thread */
               last_index = A_diag_i[i + 1];
               last_index_offd = A_offd_i[i + 1];
               if (i == stop - 1)
               {
                  last_index -= num_lost_per_thread[my_thread_num];
                  last_index_offd -= num_lost_offd_per_thread[my_thread_num];
               }

               row_sum = 0;
               num_elmts = last_index - A_diag_i[i] + last_index_offd - A_offd_i[i];
               if (max_row_elmts < num_elmts)
               {
                  /* copy both diagonal and off-diag parts of row i to _aux_ arrays */
                  cnt = 0;
                  for (j = A_diag_i[i]; j < last_index; j++)
                  {
                     A_aux_j[cnt] = A_diag_j[j];
                     A_aux_data[cnt++] = A_diag_data[j];
                     row_sum += A_diag_data[j];
                  }
                  num_lost += cnt;
                  cnt1 = cnt;
                  for (j = A_offd_i[i]; j < last_index_offd; j++)
                  {
                     A_aux_j[cnt] = A_offd_j[j] + num_cols;
                     A_aux_data[cnt++] = A_offd_data[j];
                     row_sum += A_offd_data[j];
                  }
                  num_lost_offd += cnt - cnt1;

                  /* sort data */
                  nalu_hypre_qsort2_abs(A_aux_j, A_aux_data, 0, cnt - 1);
                  scale = 0;
                  if (i > start)
                  {
                     A_diag_i[i] = cnt_diag;
                     A_offd_i[i] = cnt_offd;
                  }
                  for (j = 0; j < max_row_elmts; j++)
                  {
                     scale += A_aux_data[j];
                     if (A_aux_j[j] < num_cols)
                     {
                        A_diag_j[cnt_diag] = A_aux_j[j];
                        A_diag_data[cnt_diag++] = A_aux_data[j];
                     }
                     else
                     {
                        A_offd_j[cnt_offd] = A_aux_j[j] - num_cols;
                        A_offd_data[cnt_offd++] = A_aux_data[j];
                     }
                  }
                  num_lost -= cnt_diag - A_diag_i[i];
                  num_lost_offd -= cnt_offd - A_offd_i[i];

                  /* scale row of A */
                  if (rescale && (scale != 0.))
                  {
                     if (scale != row_sum)
                     {
                        scale = row_sum / scale;
                        for (j = A_diag_i[i]; j < cnt_diag; j++)
                        {
                           A_diag_data[j] *= scale;
                        }
                        for (j = A_offd_i[i]; j < cnt_offd; j++)
                        {
                           A_offd_data[j] *= scale;
                        }
                     }
                  }
               }  /* end if (max_row_elmts < num_elmts) */
               else
               {
                  /* nothing dropped from this row, but still have to shift entries back
                   * by the number dropped so far */
                  if (A_diag_i[i] != cnt_diag)
                  {
                     start_j = A_diag_i[i];
                     A_diag_i[i] = cnt_diag;
                     for (j = start_j; j < last_index; j++)
                     {
                        A_diag_j[cnt_diag] = A_diag_j[j];
                        A_diag_data[cnt_diag++] = A_diag_data[j];
                     }
                  }
                  else
                  {
                     cnt_diag += last_index - A_diag_i[i];
                  }

                  if (A_offd_i[i] != cnt_offd)
                  {
                     start_j = A_offd_i[i];
                     A_offd_i[i] = cnt_offd;
                     for (j = start_j; j < last_index_offd; j++)
                     {
                        A_offd_j[cnt_offd] = A_offd_j[j];
                        A_offd_data[cnt_offd++] = A_offd_data[j];
                     }
                  }
                  else
                  {
                     cnt_offd += last_index_offd - A_offd_i[i];
                  }
               }
            } /* end for (i = 0; i < n_fine; i++) */

            num_lost_per_thread[my_thread_num] += num_lost;
            num_lost_offd_per_thread[my_thread_num] += num_lost_offd;
            nalu_hypre_TFree(A_aux_j, NALU_HYPRE_MEMORY_HOST);
            nalu_hypre_TFree(A_aux_data, NALU_HYPRE_MEMORY_HOST);

         } /* end if (A_mxnum > max_row_elmts) */
      } /* end if (max_row_elmts > 0) */


      /* Sum up num_lost_global */
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (my_thread_num == 0)
      {
         num_lost_global = 0;
         num_lost_global_offd = 0;
         for (i = 0; i < max_num_threads[0]; i++)
         {
            num_lost_global += num_lost_per_thread[i];
            num_lost_global_offd += num_lost_offd_per_thread[i];
         }
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      /*
       * Synchronize and create new diag data structures
       */
      if (num_lost_global)
      {
         /* Each thread has it's own locally compressed CSR matrix from rows start
          * to stop.  Now, we have to copy each thread's chunk into the new
          * process-wide CSR data structures
          *
          * First, we compute the new process-wide number of nonzeros (i.e.,
          * A_diag_size), and compute cum_lost_per_thread[k] so that this
          * entry holds the cumulative sum of entries dropped up to and
          * including thread k. */
         if (my_thread_num == 0)
         {
            A_diag_size = A_diag_i[n_fine];

            for (i = 0; i < max_num_threads[0]; i++)
            {
               A_diag_size -= num_lost_per_thread[i];
               if (i > 0)
               {
                  cum_lost_per_thread[i] = num_lost_per_thread[i] + cum_lost_per_thread[i - 1];
               }
               else
               {
                  cum_lost_per_thread[i] = num_lost_per_thread[i];
               }
            }

            A_diag_j_new = nalu_hypre_CTAlloc(NALU_HYPRE_Int, A_diag_size, memory_location_diag);
            A_diag_data_new = nalu_hypre_CTAlloc(NALU_HYPRE_Real, A_diag_size, memory_location_diag);
         }
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp barrier
#endif

         /* points to next open spot in new data structures for this thread */
         if (my_thread_num == 0)
         {
            next_open = 0;
         }
         else
         {
            /* remember, cum_lost_per_thread[k] stores the num dropped up to and
             * including thread k */
            next_open = A_diag_i[start] - cum_lost_per_thread[my_thread_num - 1];
         }

         /* copy the j and data arrays over */
         for (i = A_diag_i[start]; i < A_diag_i[stop] - num_lost_per_thread[my_thread_num]; i++)
         {
            A_diag_j_new[next_open] = A_diag_j[i];
            A_diag_data_new[next_open] = A_diag_data[i];
            next_open += 1;
         }

#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp barrier
#endif
         /* update A_diag_i with number of dropped entries by all lower ranked
          * threads */
         if (my_thread_num > 0)
         {
            for (i = start; i < stop; i++)
            {
               A_diag_i[i] -= cum_lost_per_thread[my_thread_num - 1];
            }
         }

         if (my_thread_num == 0)
         {
            /* Set last entry */
            A_diag_i[n_fine] = A_diag_size ;

            nalu_hypre_TFree(A_diag_j, memory_location_diag);
            nalu_hypre_TFree(A_diag_data, memory_location_diag);
            nalu_hypre_CSRMatrixJ(A_diag) = A_diag_j_new;
            nalu_hypre_CSRMatrixData(A_diag) = A_diag_data_new;
            nalu_hypre_CSRMatrixNumNonzeros(A_diag) = A_diag_size;
         }
      }

      /*
       * Synchronize and create new offd data structures
       */
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      if (num_lost_global_offd)
      {
         /* Repeat process for off-diagonal */
         if (my_thread_num == 0)
         {
            A_offd_size = A_offd_i[n_fine];
            for (i = 0; i < max_num_threads[0]; i++)
            {
               A_offd_size -= num_lost_offd_per_thread[i];
               if (i > 0)
               {
                  cum_lost_per_thread[i] = num_lost_offd_per_thread[i] + cum_lost_per_thread[i - 1];
               }
               else
               {
                  cum_lost_per_thread[i] = num_lost_offd_per_thread[i];
               }
            }

            A_offd_j_new = nalu_hypre_CTAlloc(NALU_HYPRE_Int, A_offd_size, memory_location_offd);
            A_offd_data_new = nalu_hypre_CTAlloc(NALU_HYPRE_Real, A_offd_size, memory_location_offd);
         }
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp barrier
#endif

         /* points to next open spot in new data structures for this thread */
         if (my_thread_num == 0)
         {
            next_open = 0;
         }
         else
         {
            /* remember, cum_lost_per_thread[k] stores the num dropped up to and
             * including thread k */
            next_open = A_offd_i[start] - cum_lost_per_thread[my_thread_num - 1];
         }

         /* copy the j and data arrays over */
         for (i = A_offd_i[start]; i < A_offd_i[stop] - num_lost_offd_per_thread[my_thread_num]; i++)
         {
            A_offd_j_new[next_open] = A_offd_j[i];
            A_offd_data_new[next_open] = A_offd_data[i];
            next_open += 1;
         }

#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp barrier
#endif
         /* update A_offd_i with number of dropped entries by all lower ranked
          * threads */
         if (my_thread_num > 0)
         {
            for (i = start; i < stop; i++)
            {
               A_offd_i[i] -= cum_lost_per_thread[my_thread_num - 1];
            }
         }

         if (my_thread_num == 0)
         {
            /* Set last entry */
            A_offd_i[n_fine] = A_offd_size ;

            nalu_hypre_TFree(A_offd_j, memory_location_offd);
            nalu_hypre_TFree(A_offd_data, memory_location_offd);
            nalu_hypre_CSRMatrixJ(A_offd) = A_offd_j_new;
            nalu_hypre_CSRMatrixData(A_offd) = A_offd_data_new;
            nalu_hypre_CSRMatrixNumNonzeros(A_offd) = A_offd_size;
         }
      }

   } /* end parallel region */

   nalu_hypre_TFree(max_num_threads, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(cum_lost_per_thread, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(num_lost_per_thread, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(num_lost_offd_per_thread, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_PROFILE
   nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_INTERP_TRUNC] += nalu_hypre_MPI_Wtime();
#endif

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_ParCSRMatrixSetConstantValues( nalu_hypre_ParCSRMatrix *A,
                                     NALU_HYPRE_Complex       value )
{
   nalu_hypre_CSRMatrixSetConstantValues(nalu_hypre_ParCSRMatrixDiag(A), value);
   nalu_hypre_CSRMatrixSetConstantValues(nalu_hypre_ParCSRMatrixOffd(A), value);

   return nalu_hypre_error_flag;
}

void
nalu_hypre_ParCSRMatrixCopyColMapOffdToDevice(nalu_hypre_ParCSRMatrix *A)
{
#if defined(NALU_HYPRE_USING_GPU)
   if (nalu_hypre_ParCSRMatrixDeviceColMapOffd(A) == NULL)
   {
      const NALU_HYPRE_Int num_cols_A_offd = nalu_hypre_CSRMatrixNumCols(nalu_hypre_ParCSRMatrixOffd(A));
      nalu_hypre_ParCSRMatrixDeviceColMapOffd(A) = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_cols_A_offd,
                                                           NALU_HYPRE_MEMORY_DEVICE);
      nalu_hypre_TMemcpy(nalu_hypre_ParCSRMatrixDeviceColMapOffd(A), nalu_hypre_ParCSRMatrixColMapOffd(A), NALU_HYPRE_BigInt,
                    num_cols_A_offd,
                    NALU_HYPRE_MEMORY_DEVICE, NALU_HYPRE_MEMORY_HOST);
   }
#endif
}
