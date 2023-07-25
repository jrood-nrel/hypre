/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_ParCSRMatrix interface
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixCreate( MPI_Comm            comm,
                          NALU_HYPRE_BigInt        global_num_rows,
                          NALU_HYPRE_BigInt        global_num_cols,
                          NALU_HYPRE_BigInt       *row_starts,
                          NALU_HYPRE_BigInt       *col_starts,
                          NALU_HYPRE_Int           num_cols_offd,
                          NALU_HYPRE_Int           num_nonzeros_diag,
                          NALU_HYPRE_Int           num_nonzeros_offd,
                          NALU_HYPRE_ParCSRMatrix *matrix )
{
   if (!matrix)
   {
      nalu_hypre_error_in_arg(9);
      return nalu_hypre_error_flag;
   }

   *matrix = (NALU_HYPRE_ParCSRMatrix)
             nalu_hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                      row_starts, col_starts, num_cols_offd,
                                      num_nonzeros_diag, num_nonzeros_offd);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixDestroy( NALU_HYPRE_ParCSRMatrix matrix )
{
   return ( nalu_hypre_ParCSRMatrixDestroy( (nalu_hypre_ParCSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixInitialize( NALU_HYPRE_ParCSRMatrix matrix )
{
   return ( nalu_hypre_ParCSRMatrixInitialize( (nalu_hypre_ParCSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixRead
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixRead( MPI_Comm            comm,
                        const char         *file_name,
                        NALU_HYPRE_ParCSRMatrix *matrix)
{
   if (!matrix)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   *matrix = (NALU_HYPRE_ParCSRMatrix) nalu_hypre_ParCSRMatrixRead( comm, file_name );
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixPrint( NALU_HYPRE_ParCSRMatrix  matrix,
                         const char         *file_name )
{
   nalu_hypre_ParCSRMatrixPrint( (nalu_hypre_ParCSRMatrix *) matrix,
                            file_name );
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetComm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixGetComm( NALU_HYPRE_ParCSRMatrix  matrix,
                           MPI_Comm           *comm )
{
   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *comm = nalu_hypre_ParCSRMatrixComm((nalu_hypre_ParCSRMatrix *) matrix);

   return nalu_hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetDims
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixGetDims( NALU_HYPRE_ParCSRMatrix  matrix,
                           NALU_HYPRE_BigInt       *M,
                           NALU_HYPRE_BigInt       *N )
{
   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }
   *M = nalu_hypre_ParCSRMatrixGlobalNumRows((nalu_hypre_ParCSRMatrix *) matrix);
   *N = nalu_hypre_ParCSRMatrixGlobalNumCols((nalu_hypre_ParCSRMatrix *) matrix);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetRowPartitioning
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixGetRowPartitioning( NALU_HYPRE_ParCSRMatrix   matrix,
                                      NALU_HYPRE_BigInt       **row_partitioning_ptr )
{
   NALU_HYPRE_BigInt *row_partitioning, *row_starts;
   NALU_HYPRE_Int num_procs, i;

   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_MPI_Comm_size(nalu_hypre_ParCSRMatrixComm((nalu_hypre_ParCSRMatrix *) matrix),
                       &num_procs);
   row_starts = nalu_hypre_ParCSRMatrixRowStarts((nalu_hypre_ParCSRMatrix *) matrix);
   if (!row_starts) { return -1; }
   row_partitioning = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_procs + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_procs + 1; i++)
   {
      row_partitioning[i] = row_starts[i];
   }

   *row_partitioning_ptr = row_partitioning;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetGlobalRowPartitioning
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixGetGlobalRowPartitioning( NALU_HYPRE_ParCSRMatrix   matrix,
                                            NALU_HYPRE_Int            all_procs,
                                            NALU_HYPRE_BigInt       **row_partitioning_ptr )
{
   MPI_Comm        comm;
   NALU_HYPRE_Int       my_id;
   NALU_HYPRE_BigInt   *row_partitioning = NULL;

   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   comm = nalu_hypre_ParCSRMatrixComm((nalu_hypre_ParCSRMatrix *) matrix);
   nalu_hypre_MPI_Comm_rank(comm, &my_id);

   NALU_HYPRE_Int       num_procs;
   NALU_HYPRE_BigInt    row_start;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   if (my_id == 0 || all_procs)
   {
      row_partitioning = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_procs + 1, NALU_HYPRE_MEMORY_HOST);
   }

   row_start = nalu_hypre_ParCSRMatrixFirstRowIndex((nalu_hypre_ParCSRMatrix *) matrix);
   if (all_procs)
   {
      nalu_hypre_MPI_Allgather(&row_start, 1, NALU_HYPRE_MPI_BIG_INT, row_partitioning,
                          1, NALU_HYPRE_MPI_BIG_INT, comm);
   }
   else
   {
      nalu_hypre_MPI_Gather(&row_start, 1, NALU_HYPRE_MPI_BIG_INT, row_partitioning,
                       1, NALU_HYPRE_MPI_BIG_INT, 0, comm);
   }

   if (my_id == 0 || all_procs)
   {
      row_partitioning[num_procs] = nalu_hypre_ParCSRMatrixGlobalNumRows((nalu_hypre_ParCSRMatrix *) matrix);
   }

   *row_partitioning_ptr = row_partitioning;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetColPartitioning
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixGetColPartitioning( NALU_HYPRE_ParCSRMatrix   matrix,
                                      NALU_HYPRE_BigInt       **col_partitioning_ptr )
{
   NALU_HYPRE_BigInt *col_partitioning, *col_starts;
   NALU_HYPRE_Int num_procs, i;

   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_MPI_Comm_size(nalu_hypre_ParCSRMatrixComm((nalu_hypre_ParCSRMatrix *) matrix),
                       &num_procs);
   col_starts = nalu_hypre_ParCSRMatrixColStarts((nalu_hypre_ParCSRMatrix *) matrix);
   if (!col_starts) { return -1; }
   col_partitioning = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt,  num_procs + 1, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < num_procs + 1; i++)
   {
      col_partitioning[i] = col_starts[i];
   }

   *col_partitioning_ptr = col_partitioning;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetLocalRange
 *--------------------------------------------------------------------------*/
/**
   Returns range of rows and columns owned by this processor.
   Not collective.

   @return integer error code
   @param NALU_HYPRE_ParCSRMatrix matrix [IN]
   the matrix to be operated on.
   @param NALU_HYPRE_Int *row_start [OUT]
   the global number of the first row stored on this processor
   @param NALU_HYPRE_Int *row_end [OUT]
   the global number of the first row stored on this processor
   @param NALU_HYPRE_Int *col_start [OUT]
   the global number of the first column stored on this processor
   @param NALU_HYPRE_Int *col_end [OUT]
   the global number of the first column stored on this processor
*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixGetLocalRange( NALU_HYPRE_ParCSRMatrix  matrix,
                                 NALU_HYPRE_BigInt       *row_start,
                                 NALU_HYPRE_BigInt       *row_end,
                                 NALU_HYPRE_BigInt       *col_start,
                                 NALU_HYPRE_BigInt       *col_end )
{
   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParCSRMatrixGetLocalRange( (nalu_hypre_ParCSRMatrix *) matrix,
                                    row_start, row_end, col_start, col_end );
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixGetRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixGetRow( NALU_HYPRE_ParCSRMatrix  matrix,
                          NALU_HYPRE_BigInt        row,
                          NALU_HYPRE_Int          *size,
                          NALU_HYPRE_BigInt      **col_ind,
                          NALU_HYPRE_Complex     **values )
{
   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParCSRMatrixGetRow( (nalu_hypre_ParCSRMatrix *) matrix,
                             row, size, col_ind, values );
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixRestoreRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixRestoreRow( NALU_HYPRE_ParCSRMatrix  matrix,
                              NALU_HYPRE_BigInt        row,
                              NALU_HYPRE_Int          *size,
                              NALU_HYPRE_BigInt      **col_ind,
                              NALU_HYPRE_Complex     **values )
{
   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_ParCSRMatrixRestoreRow( (nalu_hypre_ParCSRMatrix *) matrix,
                                 row, size, col_ind, values );
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CSRMatrixToParCSRMatrix
 * Output argument (fifth argument): a new ParCSRmatrix.
 * Input arguments: MPI communicator, CSR matrix, and optional partitionings.
 * If you don't have partitionings, just pass a null pointer for the third
 * and fourth arguments and they will be computed.
 * Note that it is not possible to provide a null pointer if this is called
 * from Fortran code; so you must provide the paritionings from Fortran.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CSRMatrixToParCSRMatrix( MPI_Comm            comm,
                               NALU_HYPRE_CSRMatrix     A_CSR,
                               NALU_HYPRE_BigInt       *row_partitioning,
                               NALU_HYPRE_BigInt       *col_partitioning,
                               NALU_HYPRE_ParCSRMatrix *matrix)
{
   if (!matrix)
   {
      nalu_hypre_error_in_arg(5);
      return nalu_hypre_error_flag;
   }
   *matrix = (NALU_HYPRE_ParCSRMatrix)
             nalu_hypre_CSRMatrixToParCSRMatrix( comm, (nalu_hypre_CSRMatrix *) A_CSR,
                                            row_partitioning, col_partitioning) ;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning
 * Output argument (third argument): a new ParCSRmatrix.
 * Input arguments: MPI communicator, CSR matrix.
 * Row and column partitionings are computed for the output matrix.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CSRMatrixToParCSRMatrix_WithNewPartitioning(
   MPI_Comm            comm,
   NALU_HYPRE_CSRMatrix     A_CSR,
   NALU_HYPRE_ParCSRMatrix *matrix )
{
   if (!matrix)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   *matrix = (NALU_HYPRE_ParCSRMatrix)
             nalu_hypre_CSRMatrixToParCSRMatrix( comm, (nalu_hypre_CSRMatrix *) A_CSR, NULL, NULL ) ;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixMatvec( NALU_HYPRE_Complex      alpha,
                          NALU_HYPRE_ParCSRMatrix A,
                          NALU_HYPRE_ParVector    x,
                          NALU_HYPRE_Complex      beta,
                          NALU_HYPRE_ParVector    y )
{
   return ( nalu_hypre_ParCSRMatrixMatvec(
               alpha, (nalu_hypre_ParCSRMatrix *) A,
               (nalu_hypre_ParVector *) x, beta, (nalu_hypre_ParVector *) y) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_ParCSRMatrixMatvecT
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMatrixMatvecT( NALU_HYPRE_Complex      alpha,
                           NALU_HYPRE_ParCSRMatrix A,
                           NALU_HYPRE_ParVector    x,
                           NALU_HYPRE_Complex      beta,
                           NALU_HYPRE_ParVector    y )
{
   return ( nalu_hypre_ParCSRMatrixMatvecT(
               alpha, (nalu_hypre_ParCSRMatrix *) A,
               (nalu_hypre_ParVector *) x, beta, (nalu_hypre_ParVector *) y) );
}
