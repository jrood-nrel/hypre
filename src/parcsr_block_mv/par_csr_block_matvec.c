/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for nalu_hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "_nalu_hypre_parcsr_block_mv.h"

#include "NALU_HYPRE.h"
#include "parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "seq_mv/seq_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixMatvec(NALU_HYPRE_Complex alpha,
                              nalu_hypre_ParCSRBlockMatrix *A,
                              nalu_hypre_ParVector *x,
                              NALU_HYPRE_Complex beta,
                              nalu_hypre_ParVector *y)
{
   nalu_hypre_ParCSRCommHandle *comm_handle;
   nalu_hypre_ParCSRCommPkg    *comm_pkg;
   nalu_hypre_CSRBlockMatrix   *diag, *offd;
   nalu_hypre_Vector           *x_local, *y_local, *x_tmp;
   NALU_HYPRE_BigInt            num_rows, num_cols;
   NALU_HYPRE_Int               i, j, k, index;
   NALU_HYPRE_Int               blk_size, size;
   NALU_HYPRE_BigInt            x_size, y_size;
   NALU_HYPRE_Int               num_cols_offd, start, finish, elem;
   NALU_HYPRE_Int               ierr = 0, nprocs, num_sends, mypid;
   NALU_HYPRE_Complex          *x_tmp_data, *x_buf_data, *x_local_data;

   nalu_hypre_MPI_Comm_size(nalu_hypre_ParCSRBlockMatrixComm(A), &nprocs);
   nalu_hypre_MPI_Comm_rank(nalu_hypre_ParCSRBlockMatrixComm(A), &mypid);
   comm_pkg = nalu_hypre_ParCSRBlockMatrixCommPkg(A);
   num_rows = nalu_hypre_ParCSRBlockMatrixGlobalNumRows(A);
   num_cols = nalu_hypre_ParCSRBlockMatrixGlobalNumCols(A);
   blk_size = nalu_hypre_ParCSRBlockMatrixBlockSize(A);
   diag   = nalu_hypre_ParCSRBlockMatrixDiag(A);
   offd   = nalu_hypre_ParCSRBlockMatrixOffd(A);
   num_cols_offd = nalu_hypre_CSRBlockMatrixNumCols(offd);
   x_local  = nalu_hypre_ParVectorLocalVector(x);
   y_local  = nalu_hypre_ParVectorLocalVector(y);
   x_size = nalu_hypre_ParVectorGlobalSize(x);
   y_size = nalu_hypre_ParVectorGlobalSize(y);
   x_local_data = nalu_hypre_VectorData(x_local);

   /*---------------------------------------------------------------------
    *  Check for size compatibility.
    *--------------------------------------------------------------------*/

   if (num_cols * (NALU_HYPRE_BigInt)blk_size != x_size) { ierr = 11; }
   if (num_rows * (NALU_HYPRE_BigInt)blk_size != y_size) { ierr = 12; }
   if (num_cols * (NALU_HYPRE_BigInt)blk_size != x_size && num_rows * (NALU_HYPRE_BigInt)blk_size != y_size) { ierr = 13; }

   if (nprocs > 1)
   {
      x_tmp = nalu_hypre_SeqVectorCreate(num_cols_offd * blk_size);
      nalu_hypre_SeqVectorInitialize(x_tmp);
      x_tmp_data = nalu_hypre_VectorData(x_tmp);

      if (!comm_pkg)
      {
         nalu_hypre_BlockMatvecCommPkgCreate(A);
         comm_pkg = nalu_hypre_ParCSRBlockMatrixCommPkg(A);
      }
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      size = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * blk_size;
      x_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  size, NALU_HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         finish = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
         for (j = start; j < finish; j++)
         {
            elem = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j) * blk_size;
            for (k = 0; k < blk_size; k++)
            {
               x_buf_data[index++] = x_local_data[elem++];
            }
         }
      }
      comm_handle = nalu_hypre_ParCSRBlockCommHandleCreate(1, blk_size, comm_pkg,
                                                      x_buf_data, x_tmp_data);
   }
   nalu_hypre_CSRBlockMatrixMatvec(alpha, diag, x_local, beta, y_local);
   if (nprocs > 1)
   {
      nalu_hypre_ParCSRBlockCommHandleDestroy(comm_handle);
      comm_handle = NULL;
      if (num_cols_offd)
      {
         nalu_hypre_CSRBlockMatrixMatvec(alpha, offd, x_tmp, 1.0, y_local);
      }
      nalu_hypre_SeqVectorDestroy(x_tmp);
      x_tmp = NULL;
      nalu_hypre_TFree(x_buf_data, NALU_HYPRE_MEMORY_HOST);
   }
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_ParCSRBlockMatrixMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_ParCSRBlockMatrixMatvecT( NALU_HYPRE_Complex    alpha,
                                nalu_hypre_ParCSRBlockMatrix *A,
                                nalu_hypre_ParVector    *x,
                                NALU_HYPRE_Complex    beta,
                                nalu_hypre_ParVector    *y     )
{
   nalu_hypre_ParCSRCommHandle       *comm_handle;
   nalu_hypre_ParCSRCommPkg  *comm_pkg = nalu_hypre_ParCSRBlockMatrixCommPkg(A);
   nalu_hypre_CSRBlockMatrix *diag = nalu_hypre_ParCSRBlockMatrixDiag(A);
   nalu_hypre_CSRBlockMatrix *offd = nalu_hypre_ParCSRBlockMatrixOffd(A);
   nalu_hypre_Vector *x_local = nalu_hypre_ParVectorLocalVector(x);
   nalu_hypre_Vector *y_local = nalu_hypre_ParVectorLocalVector(y);
   nalu_hypre_Vector *y_tmp;

   NALU_HYPRE_Complex    *y_local_data;
   NALU_HYPRE_Int         blk_size = nalu_hypre_ParCSRBlockMatrixBlockSize(A);
   NALU_HYPRE_BigInt      x_size = nalu_hypre_ParVectorGlobalSize(x);
   NALU_HYPRE_BigInt      y_size = nalu_hypre_ParVectorGlobalSize(y);
   NALU_HYPRE_Complex    *y_tmp_data, *y_buf_data;


   NALU_HYPRE_BigInt      num_rows  = nalu_hypre_ParCSRBlockMatrixGlobalNumRows(A);
   NALU_HYPRE_BigInt      num_cols  = nalu_hypre_ParCSRBlockMatrixGlobalNumCols(A);
   NALU_HYPRE_Int         num_cols_offd = nalu_hypre_CSRBlockMatrixNumCols(offd);


   NALU_HYPRE_Int         i, j, index, start, finish, elem, num_sends;
   NALU_HYPRE_Int         size, k;


   NALU_HYPRE_Int         ierr  = 0;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/

   if (num_rows * (NALU_HYPRE_BigInt)blk_size != x_size)
   {
      ierr = 1;
   }

   if (num_cols * (NALU_HYPRE_BigInt)blk_size != y_size)
   {
      ierr = 2;
   }

   if (num_rows * (NALU_HYPRE_BigInt)blk_size != x_size && num_cols * (NALU_HYPRE_BigInt)blk_size != y_size)
   {
      ierr = 3;
   }
   /*-----------------------------------------------------------------------
    *-----------------------------------------------------------------------*/


   y_tmp = nalu_hypre_SeqVectorCreate(num_cols_offd * blk_size);
   nalu_hypre_SeqVectorInitialize(y_tmp);

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      nalu_hypre_BlockMatvecCommPkgCreate(A);
      comm_pkg = nalu_hypre_ParCSRBlockMatrixCommPkg(A);
   }

   num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
   size = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) * blk_size;
   y_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  size, NALU_HYPRE_MEMORY_HOST);

   y_tmp_data = nalu_hypre_VectorData(y_tmp);
   y_local_data = nalu_hypre_VectorData(y_local);

   if (num_cols_offd) { nalu_hypre_CSRBlockMatrixMatvecT(alpha, offd, x_local, 0.0, y_tmp); }

   comm_handle = nalu_hypre_ParCSRBlockCommHandleCreate
                 ( 2, blk_size, comm_pkg, y_tmp_data, y_buf_data);


   nalu_hypre_CSRBlockMatrixMatvecT(alpha, diag, x_local, beta, y_local);


   nalu_hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      finish = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);

      for (j = start; j < finish; j++)
      {
         elem =  nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j) * blk_size;
         for (k = 0; k < blk_size; k++)
         {
            y_local_data[elem++]
            += y_buf_data[index++];
         }
      }
   }

   nalu_hypre_TFree(y_buf_data, NALU_HYPRE_MEMORY_HOST);


   nalu_hypre_SeqVectorDestroy(y_tmp);
   y_tmp = NULL;

   return ierr;
}
