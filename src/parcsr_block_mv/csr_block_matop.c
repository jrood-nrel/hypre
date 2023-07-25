/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matrix operation functions for nalu_hypre_CSRMatrix class.
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixAdd:
 * adds two CSR Matrices A and B and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
         in A and B. To remove those, use nalu_hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRBlockMatrix *
nalu_hypre_CSRBlockMatrixAdd(nalu_hypre_CSRBlockMatrix *A, nalu_hypre_CSRBlockMatrix *B)
{
   NALU_HYPRE_Complex    *A_data   = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int        *A_i      = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int        *A_j      = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int         nrows_A  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int         ncols_A  = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Complex    *B_data   = nalu_hypre_CSRMatrixData(B);
   NALU_HYPRE_Int        *B_i      = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int        *B_j      = nalu_hypre_CSRMatrixJ(B);
   NALU_HYPRE_Int         nrows_B  = nalu_hypre_CSRMatrixNumRows(B);
   NALU_HYPRE_Int         ncols_B  = nalu_hypre_CSRMatrixNumCols(B);
   nalu_hypre_CSRMatrix  *C;
   NALU_HYPRE_Complex    *C_data;
   NALU_HYPRE_Int        *C_i;
   NALU_HYPRE_Int        *C_j;

   NALU_HYPRE_Int         block_size  = nalu_hypre_CSRBlockMatrixBlockSize(A);
   NALU_HYPRE_Int         block_sizeB = nalu_hypre_CSRBlockMatrixBlockSize(B);
   NALU_HYPRE_Int         ia, ib, ic, ii, jcol, num_nonzeros, bnnz;
   NALU_HYPRE_Int           pos;
   NALU_HYPRE_Int         *marker;

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
      nalu_hypre_printf("Warning! incompatible matrix dimensions!\n");
      return NULL;
   }
   if (block_size != block_sizeB)
   {
      nalu_hypre_printf("Warning! incompatible matrix block size!\n");
      return NULL;
   }

   bnnz = block_size * block_size;
   marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  ncols_A, NALU_HYPRE_MEMORY_HOST);
   C_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows_A + 1, NALU_HYPRE_MEMORY_HOST);

   for (ia = 0; ia < ncols_A; ia++) { marker[ia] = -1; }

   num_nonzeros = 0;
   C_i[0] = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
      {
         jcol = A_j[ia];
         marker[jcol] = ic;
         num_nonzeros++;
      }
      for (ib = B_i[ic]; ib < B_i[ic + 1]; ib++)
      {
         jcol = B_j[ib];
         if (marker[jcol] != ic)
         {
            marker[jcol] = ic;
            num_nonzeros++;
         }
      }
      C_i[ic + 1] = num_nonzeros;
   }

   C = nalu_hypre_CSRBlockMatrixCreate(block_size, nrows_A, ncols_A, num_nonzeros);
   nalu_hypre_CSRMatrixI(C) = C_i;
   nalu_hypre_CSRMatrixInitialize(C);
   C_j = nalu_hypre_CSRMatrixJ(C);
   C_data = nalu_hypre_CSRMatrixData(C);

   for (ia = 0; ia < ncols_A; ia++) { marker[ia] = -1; }

   pos = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
      {
         jcol = A_j[ia];
         C_j[pos] = jcol;
         for (ii = 0; ii < bnnz; ii++)
         {
            C_data[pos * bnnz + ii] = A_data[ia * bnnz + ii];
         }
         marker[jcol] = pos;
         pos++;
      }
      for (ib = B_i[ic]; ib < B_i[ic + 1]; ib++)
      {
         jcol = B_j[ib];
         if (marker[jcol] < C_i[ic])
         {
            C_j[pos] = jcol;
            for (ii = 0; ii < bnnz; ii++)
            {
               C_data[pos * bnnz + ii] = B_data[ib * bnnz + ii];
            }
            marker[jcol] = pos;
            pos++;
         }
         else
         {
            for (ii = 0; ii < bnnz; ii++)
            {
               C_data[marker[jcol]*bnnz + ii] = B_data[ib * bnnz + ii];
            }
         }
      }
   }
   nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);
   return C;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixMultiply
 * multiplies two CSR Matrices A and B and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
         in A and B. To remove those, use nalu_hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRBlockMatrix *
nalu_hypre_CSRBlockMatrixMultiply(nalu_hypre_CSRBlockMatrix *A, nalu_hypre_CSRBlockMatrix *B)
{
   NALU_HYPRE_Complex    *A_data   = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int        *A_i      = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int        *A_j      = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int         nrows_A  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int         ncols_A  = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int         block_size  = nalu_hypre_CSRBlockMatrixBlockSize(A);
   NALU_HYPRE_Complex    *B_data   = nalu_hypre_CSRMatrixData(B);
   NALU_HYPRE_Int        *B_i      = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int        *B_j      = nalu_hypre_CSRMatrixJ(B);
   NALU_HYPRE_Int         nrows_B  = nalu_hypre_CSRMatrixNumRows(B);
   NALU_HYPRE_Int         ncols_B  = nalu_hypre_CSRMatrixNumCols(B);
   NALU_HYPRE_Int         block_sizeB = nalu_hypre_CSRBlockMatrixBlockSize(B);
   nalu_hypre_CSRMatrix  *C;
   NALU_HYPRE_Complex    *C_data;
   NALU_HYPRE_Int        *C_i;
   NALU_HYPRE_Int        *C_j;

   NALU_HYPRE_Int         ia, ib, ic, ja, jb, num_nonzeros = 0, bnnz;
   NALU_HYPRE_Int         row_start, counter;
   NALU_HYPRE_Complex    *a_entries, *b_entries, *c_entries, dzero = 0.0, done = 1.0;
   NALU_HYPRE_Int        *B_marker;

   if (ncols_A != nrows_B)
   {
      nalu_hypre_printf("Warning! incompatible matrix dimensions!\n");
      return NULL;
   }
   if (block_size != block_sizeB)
   {
      nalu_hypre_printf("Warning! incompatible matrix block size!\n");
      return NULL;
   }

   bnnz = block_size * block_size;
   B_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  ncols_B, NALU_HYPRE_MEMORY_HOST);
   C_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  nrows_A + 1, NALU_HYPRE_MEMORY_HOST);

   for (ib = 0; ib < ncols_B; ib++) { B_marker[ib] = -1; }

   for (ic = 0; ic < nrows_A; ic++)
   {
      for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
      {
         ja = A_j[ia];
         for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++)
         {
            jb = B_j[ib];
            if (B_marker[jb] != ic)
            {
               B_marker[jb] = ic;
               num_nonzeros++;
            }
         }
      }
      C_i[ic + 1] = num_nonzeros;
   }

   C = nalu_hypre_CSRBlockMatrixCreate(block_size, nrows_A, ncols_B, num_nonzeros);
   nalu_hypre_CSRMatrixI(C) = C_i;
   nalu_hypre_CSRMatrixInitialize(C);
   C_j = nalu_hypre_CSRMatrixJ(C);
   C_data = nalu_hypre_CSRMatrixData(C);

   for (ib = 0; ib < ncols_B; ib++) { B_marker[ib] = -1; }

   counter = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      row_start = C_i[ic];
      for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
      {
         ja = A_j[ia];
         a_entries = &(A_data[ia * bnnz]);
         for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++)
         {
            jb = B_j[ib];
            b_entries = &(B_data[ib * bnnz]);
            if (B_marker[jb] < row_start)
            {
               B_marker[jb] = counter;
               C_j[B_marker[jb]] = jb;
               c_entries = &(C_data[B_marker[jb] * bnnz]);
               nalu_hypre_CSRBlockMatrixBlockMultAdd(a_entries, b_entries, dzero,
                                                c_entries, block_size);
               counter++;
            }
            else
            {
               c_entries = &(C_data[B_marker[jb] * bnnz]);
               nalu_hypre_CSRBlockMatrixBlockMultAdd(a_entries, b_entries, done,
                                                c_entries, block_size);
            }
         }
      }
   }
   nalu_hypre_TFree(B_marker, NALU_HYPRE_MEMORY_HOST);
   return C;
}

