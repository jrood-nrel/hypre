/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "../utilities/_nalu_hypre_utilities.h"
#include "../seq_mv/seq_mv.h"
#include "../parcsr_mv/_nalu_hypre_parcsr_mv.h"
#include "../parcsr_ls/_nalu_hypre_parcsr_ls.h"
#include "../krylov/krylov.h"
#include "par_csr_block_matrix.h"

extern NALU_HYPRE_Int MyBuildParLaplacian9pt(NALU_HYPRE_ParCSRMatrix  *A_ptr);

/*--------------------------------------------------------------------------
 * Test driver for unstructured matrix interface
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int main( NALU_HYPRE_Int   argc, char *argv[] )
{
   nalu_hypre_ParCSRMatrix      *par_matrix, *g_matrix, **submatrices;
   nalu_hypre_CSRMatrix         *A_diag, *A_offd;
   nalu_hypre_CSRBlockMatrix    *diag;
   nalu_hypre_CSRBlockMatrix    *offd;
   nalu_hypre_ParCSRBlockMatrix *par_blk_matrix, *par_blk_matrixT, *rap_matrix;
   nalu_hypre_Vector        *x_local;
   nalu_hypre_Vector        *y_local;
   nalu_hypre_ParVector     *x;
   nalu_hypre_ParVector     *y;
   NALU_HYPRE_Solver        gmres_solver, precon;
   NALU_HYPRE_Int                 *diag_i, *diag_j, *offd_i, *offd_j;
   NALU_HYPRE_Int                 *diag_i2, *diag_j2, *offd_i2, *offd_j2;
   NALU_HYPRE_Complex       *diag_d, *diag_d2, *offd_d, *offd_d2;
   NALU_HYPRE_Int                   mypid, local_size, nprocs;
   NALU_HYPRE_Int                   global_num_rows, global_num_cols, num_cols_offd;
   NALU_HYPRE_Int                   num_nonzeros_diag, num_nonzeros_offd, *colMap;
   NALU_HYPRE_Int                   ii, jj, kk, row, col, nnz, *indices, *colMap2;
   NALU_HYPRE_Complex               *data, ddata, *y_data;
   NALU_HYPRE_Int                   *row_starts, *col_starts, *rstarts, *cstarts;
   NALU_HYPRE_Int                   *row_starts2, *col_starts2;
   NALU_HYPRE_Int                 block_size = 2, bnnz = 4, *index_set;
   FILE                *fp;

   /* --------------------------------------------- */
   /* Initialize MPI                                */
   /* --------------------------------------------- */

   nalu_hypre_MPI_Init(&argc, &argv);
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &mypid);
   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &nprocs);

   /* build and fetch matrix */
   MyBuildParLaplacian9pt((NALU_HYPRE_ParCSRMatrix *) &par_matrix);
   global_num_rows = nalu_hypre_ParCSRMatrixGlobalNumRows(par_matrix);
   global_num_cols = nalu_hypre_ParCSRMatrixGlobalNumCols(par_matrix);
   row_starts = nalu_hypre_ParCSRMatrixRowStarts(par_matrix);
   col_starts = nalu_hypre_ParCSRMatrixColStarts(par_matrix);
   A_diag = nalu_hypre_ParCSRMatrixDiag(par_matrix);
   A_offd = nalu_hypre_ParCSRMatrixOffd(par_matrix);
   num_cols_offd     = nalu_hypre_CSRMatrixNumCols(A_offd);
   num_nonzeros_diag = nalu_hypre_CSRMatrixNumNonzeros(A_diag);
   num_nonzeros_offd = nalu_hypre_CSRMatrixNumNonzeros(A_offd);

   /* --------------------------------------------- */
   /* build vector and apply matvec                 */
   /* --------------------------------------------- */

   x = nalu_hypre_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_num_cols, col_starts);
   nalu_hypre_ParVectorInitialize(x);
   x_local = nalu_hypre_ParVectorLocalVector(x);
   data    = nalu_hypre_VectorData(x_local);
   local_size = col_starts[mypid + 1] - col_starts[mypid];
   for (ii = 0; ii < local_size; ii++) { data[ii] = 1.0; }
   y = nalu_hypre_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_num_rows, row_starts);
   nalu_hypre_ParVectorInitialize(y);
   nalu_hypre_ParCSRMatrixMatvec (1.0, par_matrix, x, 0.0, y);
   ddata = nalu_hypre_ParVectorInnerProd(y, y);
   if (mypid == 0) { nalu_hypre_printf("y inner product = %e\n", ddata); }
   nalu_hypre_ParVectorDestroy(x);
   nalu_hypre_ParVectorDestroy(y);

   /* --------------------------------------------- */
   /* build block matrix                            */
   /* --------------------------------------------- */

   rstarts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= nprocs; ii++) { rstarts[ii] = row_starts[ii]; }
   cstarts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= nprocs; ii++) { cstarts[ii] = col_starts[ii]; }

   par_blk_matrix = nalu_hypre_ParCSRBlockMatrixCreate(nalu_hypre_MPI_COMM_WORLD, block_size,
                                                  global_num_rows, global_num_cols, rstarts,
                                                  cstarts, num_cols_offd, num_nonzeros_diag,
                                                  num_nonzeros_offd);
   colMap  = nalu_hypre_ParCSRMatrixColMapOffd(par_matrix);
   if (num_cols_offd > 0) { colMap2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST); }
   else { colMap2 = NULL; }
   for (ii = 0; ii < num_cols_offd; ii++) { colMap2[ii] = colMap[ii]; }
   nalu_hypre_ParCSRBlockMatrixColMapOffd(par_blk_matrix) = colMap2;
   diag_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(par_matrix));
   diag_j = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(par_matrix));
   diag_d = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(par_matrix));
   diag = nalu_hypre_ParCSRBlockMatrixDiag(par_blk_matrix);
   diag_i2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_size + 1, NALU_HYPRE_MEMORY_HOST);
   diag_j2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_nonzeros_diag, NALU_HYPRE_MEMORY_HOST);
   diag_d2 = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  num_nonzeros_diag * bnnz, NALU_HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= local_size; ii++) { diag_i2[ii] = diag_i[ii]; }
   for (ii = 0; ii < num_nonzeros_diag; ii++) { diag_j2[ii] = diag_j[ii]; }
   nalu_hypre_CSRBlockMatrixI(diag) = diag_i2;
   nalu_hypre_CSRBlockMatrixJ(diag) = diag_j2;
   for (ii = 0; ii < num_nonzeros_diag; ii++)
   {
      for (jj = 0; jj < block_size; jj++)
         for (kk = 0; kk < block_size; kk++)
         {
            if (jj <= kk)
            {
               diag_d2[ii * bnnz + jj * block_size + kk] = diag_d[ii];
            }
            else
            {
               diag_d2[ii * bnnz + jj * block_size + kk] = 0.0;
            }
         }
   }
   nalu_hypre_CSRBlockMatrixData(diag) = diag_d2;

   offd_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(par_matrix));
   offd_j = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(par_matrix));
   offd_d = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(par_matrix));
   offd   = nalu_hypre_ParCSRBlockMatrixOffd(par_blk_matrix);
   offd_i2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_size + 1, NALU_HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= local_size; ii++) { offd_i2[ii] = offd_i[ii]; }
   nalu_hypre_CSRBlockMatrixI(offd) = offd_i2;
   if (num_cols_offd)
   {
      offd_j2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_nonzeros_offd, NALU_HYPRE_MEMORY_HOST);
      for (ii = 0; ii < num_nonzeros_offd; ii++) { offd_j2[ii] = offd_j[ii]; }
      nalu_hypre_CSRBlockMatrixJ(offd) = offd_j2;
      offd_d2 = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  num_nonzeros_offd * bnnz, NALU_HYPRE_MEMORY_HOST);
      for (ii = 0; ii < num_nonzeros_offd; ii++)
      {
         for (jj = 0; jj < block_size; jj++)
            for (kk = 0; kk < block_size; kk++)
            {
               if (jj <= kk)
               {
                  offd_d2[ii * bnnz + jj * block_size + kk] = offd_d[ii];
               }
               else
               {
                  offd_d2[ii * bnnz + jj * block_size + kk] = 0.0;
               }
            }
      }
      nalu_hypre_CSRBlockMatrixData(offd) = offd_d2;
   }
   else
   {
      nalu_hypre_CSRBlockMatrixJ(offd) = NULL;
      nalu_hypre_CSRBlockMatrixData(offd) = NULL;
   }

   /* --------------------------------------------- */
   /* build block matrix transpose                  */
   /* --------------------------------------------- */

   rstarts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= nprocs; ii++) { rstarts[ii] = row_starts[ii]; }
   cstarts = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= nprocs; ii++) { cstarts[ii] = col_starts[ii]; }

   par_blk_matrixT = nalu_hypre_ParCSRBlockMatrixCreate(nalu_hypre_MPI_COMM_WORLD, block_size,
                                                   global_num_rows, global_num_cols, rstarts,
                                                   cstarts, num_cols_offd, num_nonzeros_diag,
                                                   num_nonzeros_offd);
   colMap  = nalu_hypre_ParCSRMatrixColMapOffd(par_matrix);
   colMap2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_cols_offd, NALU_HYPRE_MEMORY_HOST);
   for (ii = 0; ii < num_cols_offd; ii++) { colMap2[ii] = colMap[ii]; }
   nalu_hypre_ParCSRBlockMatrixColMapOffd(par_blk_matrixT) = colMap2;
   diag_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixDiag(par_matrix));
   diag_j = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixDiag(par_matrix));
   diag_d = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixDiag(par_matrix));
   diag = nalu_hypre_ParCSRBlockMatrixDiag(par_blk_matrixT);
   diag_i2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_size + 1, NALU_HYPRE_MEMORY_HOST);
   diag_j2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_nonzeros_diag, NALU_HYPRE_MEMORY_HOST);
   diag_d2 = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  num_nonzeros_diag * bnnz, NALU_HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= local_size; ii++) { diag_i2[ii] = diag_i[ii]; }
   for (ii = 0; ii < num_nonzeros_diag; ii++) { diag_j2[ii] = diag_j[ii]; }
   nalu_hypre_CSRBlockMatrixI(diag) = diag_i2;
   nalu_hypre_CSRBlockMatrixJ(diag) = diag_j2;
   for (ii = 0; ii < num_nonzeros_diag; ii++)
   {
      for (jj = 0; jj < block_size; jj++)
         for (kk = 0; kk < block_size; kk++)
         {
            if (jj >= kk)
            {
               diag_d2[ii * bnnz + jj * block_size + kk] = diag_d[ii];
            }
            else
            {
               diag_d2[ii * bnnz + jj * block_size + kk] = 0.0;
            }
         }
   }
   nalu_hypre_CSRBlockMatrixData(diag) = diag_d2;

   offd_i = nalu_hypre_CSRMatrixI(nalu_hypre_ParCSRMatrixOffd(par_matrix));
   offd_j = nalu_hypre_CSRMatrixJ(nalu_hypre_ParCSRMatrixOffd(par_matrix));
   offd_d = nalu_hypre_CSRMatrixData(nalu_hypre_ParCSRMatrixOffd(par_matrix));
   offd   = nalu_hypre_ParCSRBlockMatrixOffd(par_blk_matrixT);
   offd_i2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_size + 1, NALU_HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= local_size; ii++) { offd_i2[ii] = offd_i[ii]; }
   nalu_hypre_CSRBlockMatrixI(offd) = offd_i2;
   if (num_cols_offd)
   {
      offd_j2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_nonzeros_offd, NALU_HYPRE_MEMORY_HOST);
      for (ii = 0; ii < num_nonzeros_offd; ii++) { offd_j2[ii] = offd_j[ii]; }
      nalu_hypre_CSRBlockMatrixJ(offd) = offd_j2;
      offd_d2 = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  num_nonzeros_offd * bnnz, NALU_HYPRE_MEMORY_HOST);
      for (ii = 0; ii < num_nonzeros_offd; ii++)
      {
         for (jj = 0; jj < block_size; jj++)
            for (kk = 0; kk < block_size; kk++)
            {
               if (jj >= kk)
               {
                  offd_d2[ii * bnnz + jj * block_size + kk] = offd_d[ii];
               }
               else
               {
                  offd_d2[ii * bnnz + jj * block_size + kk] = 0.0;
               }
            }
      }
      nalu_hypre_CSRBlockMatrixData(offd) = offd_d2;
   }
   else
   {
      nalu_hypre_CSRBlockMatrixJ(offd) = NULL;
      nalu_hypre_CSRBlockMatrixData(offd) = NULL;
   }

   /* --------------------------------------------- */
   /* block matvec                                  */
   /* --------------------------------------------- */

   col_starts2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= nprocs; ii++)
   {
      col_starts2[ii] = col_starts[ii] * block_size;
   }
   x = nalu_hypre_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_num_cols * block_size,
                             col_starts2);
   nalu_hypre_ParVectorInitialize(x);
   x_local = nalu_hypre_ParVectorLocalVector(x);
   data = nalu_hypre_VectorData(x_local);
   local_size = col_starts2[mypid + 1] - col_starts2[mypid];
   for (ii = 0; ii < local_size; ii++) { data[ii] = 1.0; }
   row_starts2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
   for (ii = 0; ii <= nprocs; ii++)
   {
      row_starts2[ii] = row_starts[ii] * block_size;
   }
   y = nalu_hypre_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_num_rows * block_size,
                             row_starts2);
   nalu_hypre_ParVectorInitialize(y);
   y_local = nalu_hypre_ParVectorLocalVector(y);
   y_data  = nalu_hypre_VectorData(y_local);

   nalu_hypre_BlockMatvecCommPkgCreate(par_blk_matrix);
   ddata = nalu_hypre_ParVectorInnerProd(x, x);
   if (mypid == 0) { nalu_hypre_printf("block x inner product = %e\n", ddata); }
   nalu_hypre_ParCSRBlockMatrixMatvec (1.0, par_blk_matrix, x, 0.0, y);
   ddata = nalu_hypre_ParVectorInnerProd(y, y);
   if (mypid == 0) { nalu_hypre_printf("block y inner product = %e\n", ddata); }

   /* --------------------------------------------- */
   /* RAP                                           */
   /* --------------------------------------------- */

   nalu_hypre_printf("Verifying RAP\n");
   nalu_hypre_ParCSRBlockMatrixRAP(par_blk_matrix, par_blk_matrix,
                              par_blk_matrix, &rap_matrix);
   for (ii = 0; ii < local_size; ii++) { data[ii] = 1.0; }
   nalu_hypre_ParCSRBlockMatrixMatvec (1.0, par_blk_matrix, x, 0.0, y);
   nalu_hypre_ParCSRBlockMatrixMatvec (1.0, par_blk_matrix, y, 0.0, x);
   nalu_hypre_ParCSRBlockMatrixMatvec (1.0, par_blk_matrixT, x, 0.0, y);
   ddata = nalu_hypre_ParVectorInnerProd(y, y);
   if (mypid == 0) { nalu_hypre_printf("(1) A^2 block inner product = %e\n", ddata); }
   for (ii = 0; ii < local_size; ii++) { data[ii] = 1.0; }
   nalu_hypre_ParCSRBlockMatrixMatvec (1.0, rap_matrix, x, 0.0, y);
   ddata = nalu_hypre_ParVectorInnerProd(y, y);
   if (mypid == 0) { nalu_hypre_printf("(2) A^2 block inner product = %e\n", ddata); }
   if (mypid == 0) { nalu_hypre_printf("(1) and (2) should be equal.\n"); }

#if 0
   /* --------------------------------------------- */
   /* diagnostics: print out the matrix             */
   /* --------------------------------------------- */

   diag_i = nalu_hypre_CSRBlockMatrixI(A_diag);
   diag_j = nalu_hypre_CSRBlockMatrixJ(A_diag);
   diag_d = nalu_hypre_CSRBlockMatrixData(A_diag);
   for (ii = 0; ii < nalu_hypre_ParCSRMatrixNumRows(par_matrix); ii++)
      for (jj = diag_i[ii]; jj < diag_i[ii + 1]; jj++)
      {
         nalu_hypre_printf("A %4d %4d = %e\n", ii, diag_j[jj], diag_d[jj]);
      }

   diag = nalu_hypre_ParCSRBlockMatrixDiag(rap_matrix);
   diag_i = nalu_hypre_CSRBlockMatrixI(diag);
   diag_j = nalu_hypre_CSRBlockMatrixJ(diag);
   diag_d = nalu_hypre_CSRBlockMatrixData(diag);
   nalu_hypre_printf("RAP block size = %d\n", nalu_hypre_ParCSRBlockMatrixBlockSize(rap_matrix));
   nalu_hypre_printf("RAP num rows   = %d\n", nalu_hypre_ParCSRBlockMatrixNumRows(rap_matrix));
   for (ii = 0; ii < nalu_hypre_ParCSRBlockMatrixNumRows(rap_matrix); ii++)
      for (row = 0; row < block_size; row++)
         for (jj = diag_i[ii]; jj < diag_i[ii + 1]; jj++)
            for (col = 0; col < block_size; col++)
               nalu_hypre_printf("RAP %4d %4d = %e\n", ii * block_size + row,
                            diag_j[jj]*block_size + col, diag_d[(jj + row)*block_size + col]);
   offd = nalu_hypre_ParCSRBlockMatrixOffd(rap_matrix);
   offd_i = nalu_hypre_CSRBlockMatrixI(offd);
   offd_j = nalu_hypre_CSRBlockMatrixJ(offd);
   offd_d = nalu_hypre_CSRBlockMatrixData(offd);
   if (num_cols_offd)
   {
      for (ii = 0; ii < nalu_hypre_ParCSRBlockMatrixNumRows(rap_matrix); ii++)
         for (row = 0; row < block_size; row++)
            for (jj = offd_i[ii]; jj < offd_i[ii + 1]; jj++)
               for (col = 0; col < block_size; col++)
                  nalu_hypre_printf("RAPOFFD %4d %4d = %e\n", ii * block_size + row,
                               offd_j[jj]*block_size + col, offd_d[(jj + row)*block_size + col]);
   }
#endif
   nalu_hypre_ParVectorDestroy(x);
   nalu_hypre_ParVectorDestroy(y);
   nalu_hypre_ParCSRMatrixDestroy(par_matrix);
   nalu_hypre_ParCSRBlockMatrixDestroy(par_blk_matrix);
   nalu_hypre_ParCSRBlockMatrixDestroy(par_blk_matrixT);
   nalu_hypre_ParCSRBlockMatrixDestroy(rap_matrix);

#if 0
   /* --------------------------------------------- */
   /* read in A_ee and create a NALU_HYPRE_ParCSRMatrix  */
   /* --------------------------------------------- */

   if (nprocs == 1)
   {
      fp = fopen("Amat_ee", "r");
      nalu_hypre_fscanf(fp, "%d %d", &global_num_rows, &num_nonzeros_diag);
      diag_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, (global_num_rows + 1), NALU_HYPRE_MEMORY_HOST);
      diag_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_nonzeros_diag, NALU_HYPRE_MEMORY_HOST);
      diag_d = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_nonzeros_diag, NALU_HYPRE_MEMORY_HOST);
      row = 0;
      nnz = 0;
      diag_i[0] = 0;
      for (ii = 0; ii < num_nonzeros_diag; ii++)
      {
         nalu_hypre_fscanf(fp, "%d %d %lg", &jj, &col, &ddata);
         if ((jj - 1) != row)
         {
            row++;
            diag_i[row] = nnz;
         }
         diag_j[nnz] = col - 1;
         diag_d[nnz++] = ddata;
      }
      diag_i[global_num_rows] = nnz;
      fclose(fp);
      nalu_hypre_printf("nrows = %d, nnz = %d\n", row + 1, nnz);

      row_starts = nalu_hypre_TAlloc(NALU_HYPRE_Int, 2, NALU_HYPRE_MEMORY_HOST);
      col_starts = nalu_hypre_TAlloc(NALU_HYPRE_Int, 2, NALU_HYPRE_MEMORY_HOST);
      row_starts[0] = col_starts[0] = 0;
      row_starts[1] = col_starts[1] = global_num_rows;
      num_cols_offd = 0;
      num_nonzeros_offd = 0;
      par_matrix = nalu_hypre_ParCSRMatrixCreate(nalu_hypre_MPI_COMM_WORLD, global_num_rows,
                                            global_num_rows, row_starts, col_starts, num_cols_offd,
                                            num_nonzeros_diag, num_nonzeros_offd);
      A_diag = nalu_hypre_ParCSRMatrixDiag(par_matrix);
      nalu_hypre_CSRMatrixI(A_diag) = diag_i;
      nalu_hypre_CSRMatrixJ(A_diag) = diag_j;
      nalu_hypre_CSRMatrixData(A_diag) = diag_d;

      /* --------------------------------------------- */
      /* read in discrete gradient matrix              */
      /* --------------------------------------------- */

      fp = fopen("Gmat", "r");
      nalu_hypre_fscanf(fp, "%d %d %d", &global_num_rows, &global_num_cols,
                   &num_nonzeros_diag);
      diag_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, (global_num_rows + 1), NALU_HYPRE_MEMORY_HOST);
      diag_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_nonzeros_diag, NALU_HYPRE_MEMORY_HOST);
      diag_d = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_nonzeros_diag, NALU_HYPRE_MEMORY_HOST);
      row = 0;
      nnz = 0;
      diag_i[0] = 0;
      for (ii = 0; ii < num_nonzeros_diag; ii++)
      {
         nalu_hypre_fscanf(fp, "%d %d %lg", &jj, &col, &ddata);
         if ((jj - 1) != row)
         {
            row++;
            diag_i[row] = nnz;
         }
         diag_j[nnz] = col - 1;
         diag_d[nnz++] = ddata;
      }
      diag_i[global_num_rows] = nnz;
      fclose(fp);

      row_starts = nalu_hypre_TAlloc(NALU_HYPRE_Int, 2, NALU_HYPRE_MEMORY_HOST);
      col_starts = nalu_hypre_TAlloc(NALU_HYPRE_Int, 2, NALU_HYPRE_MEMORY_HOST);
      row_starts[0] = col_starts[0] = 0;
      row_starts[1] = global_num_rows;
      col_starts[1] = global_num_cols;
      num_cols_offd = 0;
      num_nonzeros_offd = 0;
      g_matrix = nalu_hypre_ParCSRMatrixCreate(nalu_hypre_MPI_COMM_WORLD, global_num_rows,
                                          global_num_cols, row_starts, col_starts, num_cols_offd,
                                          num_nonzeros_diag, num_nonzeros_offd);
      A_diag = nalu_hypre_ParCSRMatrixDiag(g_matrix);
      nalu_hypre_CSRMatrixI(A_diag) = diag_i;
      nalu_hypre_CSRMatrixJ(A_diag) = diag_j;
      nalu_hypre_CSRMatrixData(A_diag) = diag_d;

      /* --------------------------------------------- */
      /* Check spanning tree and matrix extraction     */
      /* --------------------------------------------- */

      nalu_hypre_ParCSRMatrixGenSpanningTree(g_matrix, &indices, 0);
      submatrices = (nalu_hypre_ParCSRMatrix **)
                    nalu_hypre_TAlloc(nalu_hypre_ParCSRMatrix*, 4, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_ParCSRMatrixExtractSubmatrices(par_matrix, indices, &submatrices);
   }
#endif

   /* test block tridiagonal solver */

   if (nprocs == 1)
   {
      MyBuildParLaplacian9pt((NALU_HYPRE_ParCSRMatrix *) &par_matrix);
      row_starts = nalu_hypre_ParCSRMatrixRowStarts(par_matrix);
      col_starts = nalu_hypre_ParCSRMatrixColStarts(par_matrix);
      NALU_HYPRE_ParCSRGMRESCreate(nalu_hypre_MPI_COMM_WORLD, &gmres_solver);
      NALU_HYPRE_GMRESSetKDim(gmres_solver, 10);
      NALU_HYPRE_GMRESSetMaxIter(gmres_solver, 1000);
      NALU_HYPRE_GMRESSetTol(gmres_solver, 1.0e-6);
      NALU_HYPRE_GMRESSetLogging(gmres_solver, 1);
      NALU_HYPRE_GMRESSetPrintLevel(gmres_solver, 2);
      NALU_HYPRE_BlockTridiagCreate(&precon);
      NALU_HYPRE_BlockTridiagSetPrintLevel(precon, 0);
      NALU_HYPRE_BlockTridiagSetAMGNumSweeps(precon, 1);
      local_size = col_starts[mypid + 1] - col_starts[mypid];
      index_set = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  local_size + 1, NALU_HYPRE_MEMORY_HOST);
      jj = 0;
      /* for (ii = 0; ii < local_size/2; ii++) index_set[jj++] = ii * 2; */
      for (ii = 0; ii < local_size / 2; ii++) { index_set[jj++] = ii; }
      NALU_HYPRE_BlockTridiagSetIndexSet(precon, jj, index_set);
      NALU_HYPRE_GMRESSetPrecond(gmres_solver,
                            (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BlockTridiagSolve,
                            (NALU_HYPRE_PtrToSolverFcn) NALU_HYPRE_BlockTridiagSetup,
                            precon);
      col_starts2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
      for (ii = 0; ii <= nprocs; ii++) { col_starts2[ii] = col_starts[ii]; }
      x = nalu_hypre_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_num_cols, col_starts2);
      nalu_hypre_ParVectorInitialize(x);
      x_local = nalu_hypre_ParVectorLocalVector(x);
      local_size = col_starts2[mypid + 1] - col_starts2[mypid];
      data = nalu_hypre_VectorData(x_local);
      for (ii = 0; ii < local_size; ii++) { data[ii] = 0.0; }
      row_starts2 = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nprocs + 1, NALU_HYPRE_MEMORY_HOST);
      for (ii = 0; ii <= nprocs; ii++) { row_starts2[ii] = row_starts[ii]; }
      y = nalu_hypre_ParVectorCreate(nalu_hypre_MPI_COMM_WORLD, global_num_rows, row_starts2);
      nalu_hypre_ParVectorInitialize(y);
      y_local = nalu_hypre_ParVectorLocalVector(y);
      data = nalu_hypre_VectorData(y_local);
      for (ii = 0; ii < local_size; ii++) { data[ii] = 1.0; }

      NALU_HYPRE_GMRESSetup(gmres_solver, (NALU_HYPRE_Matrix) par_matrix,
                       (NALU_HYPRE_Vector) y, (NALU_HYPRE_Vector) x);
      NALU_HYPRE_GMRESSolve(gmres_solver, (NALU_HYPRE_Matrix) par_matrix,
                       (NALU_HYPRE_Vector) y, (NALU_HYPRE_Vector) x);

      nalu_hypre_ParVectorDestroy(x);
      nalu_hypre_ParVectorDestroy(y);
      nalu_hypre_ParCSRMatrixDestroy(par_matrix);
   }

   /* Finalize MPI */
   nalu_hypre_MPI_Finalize();
   return 0;
}

/*----------------------------------------------------------------------
 * Build standard 9-point laplacian in 2D with grid and anisotropy.
 * Parameters given in command line.
 *----------------------------------------------------------------------*/

NALU_HYPRE_Int MyBuildParLaplacian9pt(NALU_HYPRE_ParCSRMatrix  *A_ptr)
{
   NALU_HYPRE_Int                 nx, ny;
   NALU_HYPRE_Int                 P, Q;
   NALU_HYPRE_ParCSRMatrix  A;
   NALU_HYPRE_Int                 num_procs, myid;
   NALU_HYPRE_Int                 p, q;
   NALU_HYPRE_Complex      *values;

   /*-----------------------------------------------------------
    * Initialize some stuff
    *-----------------------------------------------------------*/

   nalu_hypre_MPI_Comm_size(nalu_hypre_MPI_COMM_WORLD, &num_procs );
   nalu_hypre_MPI_Comm_rank(nalu_hypre_MPI_COMM_WORLD, &myid );

   /*-----------------------------------------------------------
    * Set defaults
    *-----------------------------------------------------------*/

   nx = 200;
   ny = 200;
   P  = 2;
   if (num_procs == 1) { P = 1; }
   Q  = num_procs / P;

   /*-----------------------------------------------------------
    * Print driver parameters
    *-----------------------------------------------------------*/

   if (myid == 0)
   {
      nalu_hypre_printf("  Laplacian 9pt:\n");
      nalu_hypre_printf("    (nx, ny) = (%d, %d)\n", nx, ny);
      nalu_hypre_printf("    (Px, Py) = (%d, %d)\n\n", P,  Q);
   }

   /*-----------------------------------------------------------
    * Set up the grid structure
    *-----------------------------------------------------------*/

   p = myid % P;
   q = ( myid - p) / P;

   /*-----------------------------------------------------------
    * Generate the matrix
    *-----------------------------------------------------------*/

   values = nalu_hypre_CTAlloc(NALU_HYPRE_Complex,  2, NALU_HYPRE_MEMORY_HOST);
   values[1] = -1.;
   values[0] = 0.;
   if (nx > 1) { values[0] += 2.0; }
   if (ny > 1) { values[0] += 2.0; }
   if (nx > 1 && ny > 1) { values[0] += 4.0; }
   A = (NALU_HYPRE_ParCSRMatrix) GenerateLaplacian9pt(nalu_hypre_MPI_COMM_WORLD,
                                                 nx, ny, P, Q, p, q, values);
   nalu_hypre_TFree(values, NALU_HYPRE_MEMORY_HOST);
   *A_ptr = A;
   return (0);
}
