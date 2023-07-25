/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_mv.h"
#include "eliminate_rowscols.h"

NALU_HYPRE_Int nalu_hypre_ParCSRMatrixEliminateRowsCols (nalu_hypre_ParCSRMatrix *A,
                                               NALU_HYPRE_Int nrows_to_eliminate,
                                               NALU_HYPRE_Int *rows_to_eliminate)
{
   NALU_HYPRE_Int ierr = 0;

   MPI_Comm         comm      = nalu_hypre_ParCSRMatrixComm(A);

   nalu_hypre_CSRMatrix *diag      = nalu_hypre_ParCSRMatrixDiag(A);
   nalu_hypre_CSRMatrix *offd      = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int diag_nrows       = nalu_hypre_CSRMatrixNumRows(diag);
   NALU_HYPRE_Int offd_ncols       = nalu_hypre_CSRMatrixNumCols(offd);

   NALU_HYPRE_Int ncols_to_eliminate;
   NALU_HYPRE_Int *cols_to_eliminate;

   NALU_HYPRE_Int       myproc;
   NALU_HYPRE_Int       ibeg;

   nalu_hypre_MPI_Comm_rank(comm, &myproc);
   ibeg = 0;


   /* take care of the diagonal part (sequential elimination) */
   nalu_hypre_CSRMatrixEliminateRowsColsDiag (A, nrows_to_eliminate,
                                         rows_to_eliminate);

   /* eliminate the off-diagonal rows */
   nalu_hypre_CSRMatrixEliminateRowsOffd (A, nrows_to_eliminate,
                                     rows_to_eliminate);

   /* figure out which offd cols should be eliminated */
   {
      nalu_hypre_ParCSRCommHandle *comm_handle;
      nalu_hypre_ParCSRCommPkg *comm_pkg;
      NALU_HYPRE_Int num_sends, *int_buf_data;
      NALU_HYPRE_Int index, start;
      NALU_HYPRE_Int i, j, k;

      NALU_HYPRE_Int *eliminate_row = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  diag_nrows, NALU_HYPRE_MEMORY_HOST);
      NALU_HYPRE_Int *eliminate_col = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  offd_ncols, NALU_HYPRE_MEMORY_HOST);

      /* make sure A has a communication package */
      comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
      if (!comm_pkg)
      {
         nalu_hypre_MatvecCommPkgCreate(A);
         comm_pkg = nalu_hypre_ParCSRMatrixCommPkg(A);
      }

      /* which of the local rows are to be eliminated */
      for (i = 0; i < diag_nrows; i++)
      {
         eliminate_row[i] = 0;
      }
      for (i = 0; i < nrows_to_eliminate; i++)
      {
         eliminate_row[rows_to_eliminate[i] - ibeg] = 1;
      }

      /* use a Matvec communication pattern to find (in eliminate_col)
         which of the local offd columns are to be eliminated */
      num_sends = nalu_hypre_ParCSRCommPkgNumSends(comm_pkg);
      int_buf_data = nalu_hypre_CTAlloc(NALU_HYPRE_Int,
                                   nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                   num_sends), NALU_HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < nalu_hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            k = nalu_hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
            int_buf_data[index++] = eliminate_row[k];
         }
      }
      comm_handle = nalu_hypre_ParCSRCommHandleCreate(11, comm_pkg,
                                                 int_buf_data, eliminate_col);
      nalu_hypre_ParCSRCommHandleDestroy(comm_handle);

      /* set the array cols_to_eliminate */
      ncols_to_eliminate = 0;
      for (i = 0; i < offd_ncols; i++)
         if (eliminate_col[i])
         {
            ncols_to_eliminate++;
         }

      cols_to_eliminate = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  ncols_to_eliminate, NALU_HYPRE_MEMORY_HOST);

      ncols_to_eliminate = 0;
      for (i = 0; i < offd_ncols; i++)
         if (eliminate_col[i])
         {
            cols_to_eliminate[ncols_to_eliminate++] = i;
         }

      nalu_hypre_TFree(int_buf_data, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(eliminate_row, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(eliminate_col, NALU_HYPRE_MEMORY_HOST);
   }

   /* eliminate the off-diagonal columns */
   nalu_hypre_CSRMatrixEliminateColsOffd (offd, ncols_to_eliminate,
                                     cols_to_eliminate);

   nalu_hypre_TFree(cols_to_eliminate, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}


NALU_HYPRE_Int nalu_hypre_CSRMatrixEliminateRowsColsDiag (nalu_hypre_ParCSRMatrix *A,
                                                NALU_HYPRE_Int nrows_to_eliminate,
                                                NALU_HYPRE_Int *rows_to_eliminate)
{
   NALU_HYPRE_Int ierr = 0;

   MPI_Comm          comm      = nalu_hypre_ParCSRMatrixComm(A);
   nalu_hypre_CSRMatrix  *Adiag     = nalu_hypre_ParCSRMatrixDiag(A);

   NALU_HYPRE_Int         i, j;
   NALU_HYPRE_Int         irow, ibeg, iend;

   NALU_HYPRE_Int         nnz       = nalu_hypre_CSRMatrixNumNonzeros(Adiag);
   NALU_HYPRE_Int        *Ai        = nalu_hypre_CSRMatrixI(Adiag);
   NALU_HYPRE_Int        *Aj        = nalu_hypre_CSRMatrixJ(Adiag);
   NALU_HYPRE_Real       *Adata     = nalu_hypre_CSRMatrixData(Adiag);

   NALU_HYPRE_Int        *local_rows;

   NALU_HYPRE_Int         myproc;

   nalu_hypre_MPI_Comm_rank(comm, &myproc);
   ibeg = 0;

   /* grab local rows to eliminate */
   local_rows = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nrows_to_eliminate, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nrows_to_eliminate; i++)
   {
      local_rows[i] = rows_to_eliminate[i] - ibeg;
   }

   /* remove the columns */
   for (i = 0; i < nnz; i++)
   {
      irow = nalu_hypre_BinarySearch(local_rows, Aj[i],
                                nrows_to_eliminate);
      if (irow != -1)
      {
         Adata[i] = 0.0;
      }
   }

   /* remove the rows and set the diagonal equal to 1 */
   for (i = 0; i < nrows_to_eliminate; i++)
   {
      irow = local_rows[i];
      ibeg = Ai[irow];
      iend = Ai[irow + 1];
      for (j = ibeg; j < iend; j++)
         if (Aj[j] == irow)
         {
            Adata[j] = 1.0;
         }
         else
         {
            Adata[j] = 0.0;
         }
   }

   nalu_hypre_TFree(local_rows, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

NALU_HYPRE_Int nalu_hypre_CSRMatrixEliminateRowsOffd (nalu_hypre_ParCSRMatrix *A,
                                            NALU_HYPRE_Int  nrows_to_eliminate,
                                            NALU_HYPRE_Int *rows_to_eliminate)
{
   NALU_HYPRE_Int ierr = 0;

   MPI_Comm         comm      = nalu_hypre_ParCSRMatrixComm(A);

   nalu_hypre_CSRMatrix *Aoffd     = nalu_hypre_ParCSRMatrixOffd(A);
   NALU_HYPRE_Int       *Ai        = nalu_hypre_CSRMatrixI(Aoffd);

   NALU_HYPRE_Real      *Adata     = nalu_hypre_CSRMatrixData(Aoffd);

   NALU_HYPRE_Int i, j;
   NALU_HYPRE_Int ibeg, iend;

   NALU_HYPRE_Int *local_rows;
   NALU_HYPRE_Int myproc;

   nalu_hypre_MPI_Comm_rank(comm, &myproc);
   ibeg = 0;

   /* grab local rows to eliminate */
   local_rows = nalu_hypre_TAlloc(NALU_HYPRE_Int,  nrows_to_eliminate, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nrows_to_eliminate; i++)
   {
      local_rows[i] = rows_to_eliminate[i] - ibeg;
   }

   for (i = 0; i < nrows_to_eliminate; i++)
   {
      ibeg = Ai[local_rows[i]];
      iend = Ai[local_rows[i] + 1];
      for (j = ibeg; j < iend; j++)
      {
         Adata[j] = 0.0;
      }
   }

   nalu_hypre_TFree(local_rows, NALU_HYPRE_MEMORY_HOST);

   return ierr;
}

NALU_HYPRE_Int nalu_hypre_CSRMatrixEliminateColsOffd (nalu_hypre_CSRMatrix *Aoffd,
                                            NALU_HYPRE_Int ncols_to_eliminate,
                                            NALU_HYPRE_Int *cols_to_eliminate)
{
   NALU_HYPRE_Int ierr = 0;

   NALU_HYPRE_Int i;
   NALU_HYPRE_Int icol;

   NALU_HYPRE_Int nnz = nalu_hypre_CSRMatrixNumNonzeros(Aoffd);
   NALU_HYPRE_Int *Aj = nalu_hypre_CSRMatrixJ(Aoffd);
   NALU_HYPRE_Real *Adata = nalu_hypre_CSRMatrixData(Aoffd);

   for (i = 0; i < nnz; i++)
   {
      icol = nalu_hypre_BinarySearch(cols_to_eliminate, Aj[i],
                                ncols_to_eliminate);
      if (icol != -1)
      {
         Adata[i] = 0.0;
      }
   }

   return ierr;
}
