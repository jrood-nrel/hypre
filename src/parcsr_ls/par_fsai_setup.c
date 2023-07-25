/******************************************************************************
 *  Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 *  NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 *  SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "_nalu_hypre_blas.h"
#include "_nalu_hypre_lapack.h"

/*****************************************************************************
 *
 * Routine for driving the setup phase of FSAI
 *
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixExtractDenseMat
 *
 * Extract A[P, P] into a dense matrix.
 *
 * Parameters:
 * - A:       The nalu_hypre_CSRMatrix whose submatrix will be extracted.
 * - A_sub:   A patt_size^2 - sized array to hold the lower triangular of
 *            the symmetric submatrix A[P, P].
 * - pattern: A patt_size - sized array to hold the wanted rows/cols.
 * - marker:  A work array of length equal to the number of columns in A.
 *            All values should be -1.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixExtractDenseMat( nalu_hypre_CSRMatrix *A,
                                nalu_hypre_Vector    *A_sub,
                                NALU_HYPRE_Int       *pattern,
                                NALU_HYPRE_Int        patt_size,
                                NALU_HYPRE_Int       *marker )
{
   NALU_HYPRE_Int     *A_i = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *A_j = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Complex *A_a = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Complex *A_sub_data = nalu_hypre_VectorData(A_sub);

   /* Local variables */
   NALU_HYPRE_Int      cc, i, ii, j;

   // TODO: Do we need to reinitialize all entries?
   for (i = 0; i < nalu_hypre_VectorSize(A_sub); i++)
   {
      A_sub_data[i] = 0.0;
   }

   for (i = 0; i < patt_size; i++)
   {
      ii = pattern[i];
      for (j = A_i[ii]; j < A_i[ii + 1]; j++)
      {
         if ((A_j[j] <= ii) &&
             (cc = marker[A_j[j]]) >= 0)
         {
            A_sub_data[cc * patt_size + i] = A_a[j];
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixExtractDenseRow
 *
 * Extract the dense subrow from a matrix (A[i, P])
 *
 * Parameters:
 * - A:         The nalu_hypre_CSRMatrix whose subrow will be extracted.
 * - A_subrow:  The extracted subrow of A[i, P].
 * - marker:    A work array of length equal to the number of row in A.
 *              Assumed to be set to all -1.
 * - row_num:   which row index of A we want to extract data from.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixExtractDenseRow( nalu_hypre_CSRMatrix *A,
                                nalu_hypre_Vector    *A_subrow,
                                NALU_HYPRE_Int       *marker,
                                NALU_HYPRE_Int        row_num )
{
   NALU_HYPRE_Int      *A_i = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int      *A_j = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Complex  *A_a = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Complex  *sub_row_data = nalu_hypre_VectorData(A_subrow);

   /* Local variables */
   NALU_HYPRE_Int       j, cc;

   for (j = 0; j < nalu_hypre_VectorSize(A_subrow); j++)
   {
      sub_row_data[j] = 0.0;
   }

   for (j = A_i[row_num]; j < A_i[row_num + 1]; j++)
   {
      if ((cc = marker[A_j[j]]) >= 0)
      {
         sub_row_data[cc] = A_a[j];
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FindKapGrad
 *
 * Finding the Kaporin Gradient contribution (psi) of a given row.
 *
 * Parameters:
 *  - A:            CSR matrix diagonal of A.
 *  - kap_grad:     Array holding the kaporin gradient.
 *                  This will we modified.
 *  - kg_pos:       Array of the nonzero column indices of kap_grad.
 *                  To be modified.
 *  - G_temp:       Work array of G for row i.
 *  - pattern:      Array of column indices of the nonzeros of G_temp.
 *  - patt_size:    Number of column indices of the nonzeros of G_temp.
 *  - max_row_size: To ensure we don't overfill kap_grad.
 *  - row_num:      Which row of G we are working on.
 *  - marker:       Array of length equal to the number of rows in A.
 *                  Assumed to all be set to -1.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FindKapGrad( nalu_hypre_CSRMatrix  *A_diag,
                   nalu_hypre_Vector     *kap_grad,
                   NALU_HYPRE_Int        *kg_pos,
                   nalu_hypre_Vector     *G_temp,
                   NALU_HYPRE_Int        *pattern,
                   NALU_HYPRE_Int         patt_size,
                   NALU_HYPRE_Int         max_row_size,
                   NALU_HYPRE_Int         row_num,
                   NALU_HYPRE_Int        *kg_marker )
{

   NALU_HYPRE_Int      *A_i = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int      *A_j = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Complex  *A_a = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Complex  *G_temp_data   = nalu_hypre_VectorData(G_temp);
   NALU_HYPRE_Complex  *kap_grad_data = nalu_hypre_VectorData(kap_grad);

   /* Local Variables */
   NALU_HYPRE_Int       i, ii, j, k, count, col;

   count = 0;

   /* Compute A[row_num, 0:(row_num-1)]*G_temp[i,i] */
   for (j = A_i[row_num]; j < A_i[row_num + 1]; j++)
   {
      col = A_j[j];
      if (col < row_num)
      {
         if (kg_marker[col] > -1)
         {
            /* Add A[row_num, col] to the tentative pattern */
            kg_marker[col] = count + 1;
            kg_pos[count] = col;
            kap_grad_data[count] = A_a[j];
            count++;
         }
      }
   }

   /* Compute A[0:(row_num-1), P]*G_temp[P, i] */
   for (i = 0; i < patt_size; i++)
   {
      ii = pattern[i];
      for (j = A_i[ii]; j < A_i[ii + 1]; j++)
      {
         col = A_j[j];
         if (col < row_num)
         {
            k = kg_marker[col];
            if (k == 0)
            {
               /* New entry in the tentative pattern */
               kg_marker[col] = count + 1;
               kg_pos[count] = col;
               kap_grad_data[count] = G_temp_data[i] * A_a[j];
               count++;
            }
            else if (k > 0)
            {
               /* Already existing entry in the tentative pattern */
               kap_grad_data[k - 1] += G_temp_data[i] * A_a[j];
            }
         }
      }
   }

   /* Update number of nonzero coefficients held in kap_grad */
   nalu_hypre_VectorSize(kap_grad) = count;

   /* Update to absolute values */
   for (i = 0; i < count; i++)
   {
      kap_grad_data[i] = nalu_hypre_cabs(kap_grad_data[i]);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_swap2_ci
 *--------------------------------------------------------------------------*/

void
nalu_hypre_swap2_ci( NALU_HYPRE_Complex  *v,
                NALU_HYPRE_Int      *w,
                NALU_HYPRE_Int       i,
                NALU_HYPRE_Int       j )
{
   NALU_HYPRE_Complex  temp;
   NALU_HYPRE_Int      temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_qsort2_ci
 *
 * Quick Sort (largest to smallest) for complex arrays.
 * Sort on real portion of v (NALU_HYPRE_Complex), move w.
 *--------------------------------------------------------------------------*/

void
nalu_hypre_qsort2_ci( NALU_HYPRE_Complex  *v,
                 NALU_HYPRE_Int      *w,
                 NALU_HYPRE_Int      left,
                 NALU_HYPRE_Int      right )
{
   NALU_HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }

   nalu_hypre_swap2_ci(v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (nalu_hypre_creal(v[i]) > nalu_hypre_creal(v[left]))
      {
         nalu_hypre_swap2_ci(v, w, ++last, i);
      }
   }

   nalu_hypre_swap2_ci(v, w, left, last);
   nalu_hypre_qsort2_ci(v, w, left, last - 1);
   nalu_hypre_qsort2_ci(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_PartialSelectSortCI
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_PartialSelectSortCI( NALU_HYPRE_Complex  *v,
                           NALU_HYPRE_Int      *w,
                           NALU_HYPRE_Int       size,
                           NALU_HYPRE_Int       nentries )
{
   NALU_HYPRE_Int  i, k, pos;

   for (k = 0; k < nentries; k++)
   {
      /* Find largest kth entry */
      pos = k;
      for (i = k + 1; i < size; i++)
      {
         if (nalu_hypre_creal(v[i]) > nalu_hypre_creal(v[pos]))
         {
            pos = i;
         }
      }

      /* Move entry to beggining of the array */
      nalu_hypre_swap2_ci(v, w, k, pos);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_AddToPattern
 *
 * Take the largest elements from the kaporin gradient and add their
 * locations to pattern.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_AddToPattern( nalu_hypre_Vector *kap_grad,
                    NALU_HYPRE_Int    *kg_pos,
                    NALU_HYPRE_Int    *pattern,
                    NALU_HYPRE_Int    *patt_size,
                    NALU_HYPRE_Int    *kg_marker,
                    NALU_HYPRE_Int     max_step_size )
{
   NALU_HYPRE_Int       kap_grad_size = nalu_hypre_VectorSize(kap_grad);
   NALU_HYPRE_Complex  *kap_grad_data = nalu_hypre_VectorData(kap_grad);

   NALU_HYPRE_Int       i, nentries;

   /* Number of entries that can be added */
   nentries = nalu_hypre_min(kap_grad_size, max_step_size);

   /* Reorder candidates according to larger weights */
   //nalu_hypre_qsort2_ci(kap_grad_data, &kg_pos, 0, kap_grad_size-1);
   nalu_hypre_PartialSelectSortCI(kap_grad_data, kg_pos, kap_grad_size, nentries);

   /* Update pattern with new entries */
   for (i = 0; i < nentries; i++)
   {
      pattern[*patt_size + i] = kg_pos[i];
   }
   *patt_size += nentries;

   /* Put pattern in ascending order */
   nalu_hypre_qsort0(pattern, 0, (*patt_size) - 1);

   /* Reset marked entries that are added to pattern */
   for (i = 0; i < nentries; i++)
   {
      kg_marker[kg_pos[i]] = -1;
   }
   for (i = nentries; i < kap_grad_size; i++)
   {
      kg_marker[kg_pos[i]] = 0;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DenseSPDSystemSolve
 *
 * Solve the dense SPD linear system with LAPACK:
 *
 *    mat*lhs = -rhs
 *
 * Note: the contents of A change to its Cholesky factor.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_DenseSPDSystemSolve( nalu_hypre_Vector *mat,
                           nalu_hypre_Vector *rhs,
                           nalu_hypre_Vector *lhs )
{
   NALU_HYPRE_Int      size = nalu_hypre_VectorSize(rhs);
   NALU_HYPRE_Complex *mat_data = nalu_hypre_VectorData(mat);
   NALU_HYPRE_Complex *rhs_data = nalu_hypre_VectorData(rhs);
   NALU_HYPRE_Complex *lhs_data = nalu_hypre_VectorData(lhs);

   /* Local variables */
   NALU_HYPRE_Int      num_rhs = 1;
   char           uplo = 'L';
   char           msg[512];
   NALU_HYPRE_Int      i, info;

   /* Copy RHS into LHS */
   for (i = 0; i < size; i++)
   {
      lhs_data[i] = -rhs_data[i];
   }

   /* Compute Cholesky factor */
   nalu_hypre_dpotrf(&uplo, &size, mat_data, &size, &info);
   if (info)
   {
      nalu_hypre_sprintf(msg, "Error: dpotrf failed with code %d\n", info);
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, msg);
      return nalu_hypre_error_flag;
   }

   /* Solve dense linear system */
   nalu_hypre_dpotrs(&uplo, &size, &num_rhs, mat_data, &size, lhs_data, &size, &info);
   if (info)
   {
      nalu_hypre_sprintf(msg, "Error: dpotrs failed with code %d\n", info);
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, msg);
      return nalu_hypre_error_flag;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FSAISetupNative
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FSAISetupNative( void               *fsai_vdata,
                       nalu_hypre_ParCSRMatrix *A,
                       nalu_hypre_ParVector    *f,
                       nalu_hypre_ParVector    *u )
{
   /* Data structure variables */
   nalu_hypre_ParFSAIData      *fsai_data        = (nalu_hypre_ParFSAIData*) fsai_vdata;
   NALU_HYPRE_Real              kap_tolerance    = nalu_hypre_ParFSAIDataKapTolerance(fsai_data);
   NALU_HYPRE_Int               max_steps        = nalu_hypre_ParFSAIDataMaxSteps(fsai_data);
   NALU_HYPRE_Int               max_step_size    = nalu_hypre_ParFSAIDataMaxStepSize(fsai_data);

   /* CSRMatrix A_diag variables */
   nalu_hypre_CSRMatrix        *A_diag           = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int              *A_i              = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Complex          *A_a              = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int               num_rows_diag_A  = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int               num_nnzs_diag_A  = nalu_hypre_CSRMatrixNumNonzeros(A_diag);
   NALU_HYPRE_Int               avg_nnzrow_diag_A;

   /* Matrix G variables */
   nalu_hypre_ParCSRMatrix     *G = nalu_hypre_ParFSAIDataGmat(fsai_data);
   nalu_hypre_CSRMatrix        *G_diag;
   NALU_HYPRE_Int              *G_i;
   NALU_HYPRE_Int              *G_j;
   NALU_HYPRE_Complex          *G_a;
   NALU_HYPRE_Int               max_nnzrow_diag_G;   /* Max. number of nonzeros per row in G_diag */
   NALU_HYPRE_Int               max_cand_size;       /* Max size of kg_pos */

   /* Local variables */
   char                     msg[512];    /* Warning message */
   NALU_HYPRE_Int           *twspace;     /* shared work space for omp threads */

   /* Initalize some variables */
   avg_nnzrow_diag_A = (num_rows_diag_A > 0) ? num_nnzs_diag_A / num_rows_diag_A : 0;
   max_nnzrow_diag_G = max_steps * max_step_size + 1;
   max_cand_size     = avg_nnzrow_diag_A * max_nnzrow_diag_G;

   G_diag = nalu_hypre_ParCSRMatrixDiag(G);
   G_a = nalu_hypre_CSRMatrixData(G_diag);
   G_i = nalu_hypre_CSRMatrixI(G_diag);
   G_j = nalu_hypre_CSRMatrixJ(G_diag);

   /* Allocate shared work space array for OpenMP threads */
   twspace = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nalu_hypre_NumThreads() + 1, NALU_HYPRE_MEMORY_HOST);

   /**********************************************************************
   * Start of Adaptive FSAI algorithm
   ***********************************************************************/

   /* Cycle through each of the local rows */
   NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "MainLoop");
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      nalu_hypre_Vector   *G_temp;        /* Vector holding the values of G[i,:] */
      nalu_hypre_Vector   *A_sub;         /* Vector holding the dense submatrix A[P, P] */
      nalu_hypre_Vector   *A_subrow;      /* Vector holding A[i, P] */
      nalu_hypre_Vector   *kap_grad;      /* Vector holding the Kaporin gradient values */
      NALU_HYPRE_Int      *kg_pos;        /* Indices of nonzero entries of kap_grad */
      NALU_HYPRE_Int      *kg_marker;     /* Marker array with nonzeros pointing to kg_pos */
      NALU_HYPRE_Int      *marker;        /* Marker array with nonzeros pointing to P */
      NALU_HYPRE_Int      *pattern;       /* Array holding column indices of G[i,:] */
      NALU_HYPRE_Int       patt_size;     /* Number of entries in current pattern */
      NALU_HYPRE_Int       patt_size_old; /* Number of entries in previous pattern */
      NALU_HYPRE_Int       ii;            /* Thread identifier */
      NALU_HYPRE_Int       num_threads;   /* Number of active threads */
      NALU_HYPRE_Int       ns, ne;        /* Initial and last row indices */
      NALU_HYPRE_Int       i, j, k, iloc; /* Loop variables */
      NALU_HYPRE_Complex   old_psi;       /* GAG' before k-th interation of aFSAI */
      NALU_HYPRE_Complex   new_psi;       /* GAG' after k-th interation of aFSAI */
      NALU_HYPRE_Complex   row_scale;     /* Scaling factor for G_temp */
      NALU_HYPRE_Complex  *G_temp_data;
      NALU_HYPRE_Complex  *A_subrow_data;

      NALU_HYPRE_Int       num_rows_Gloc;
      NALU_HYPRE_Int       num_nnzs_Gloc;
      NALU_HYPRE_Int      *Gloc_i;
      NALU_HYPRE_Int      *Gloc_j;
      NALU_HYPRE_Complex  *Gloc_a;

      /* Allocate and initialize local vector variables */
      G_temp    = nalu_hypre_SeqVectorCreate(max_nnzrow_diag_G);
      A_subrow  = nalu_hypre_SeqVectorCreate(max_nnzrow_diag_G);
      kap_grad  = nalu_hypre_SeqVectorCreate(max_cand_size);
      A_sub     = nalu_hypre_SeqVectorCreate(max_nnzrow_diag_G * max_nnzrow_diag_G);
      pattern   = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_nnzrow_diag_G, NALU_HYPRE_MEMORY_HOST);
      kg_pos    = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_cand_size, NALU_HYPRE_MEMORY_HOST);
      kg_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows_diag_A, NALU_HYPRE_MEMORY_HOST);
      marker    = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows_diag_A, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_SeqVectorInitialize_v2(G_temp, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_SeqVectorInitialize_v2(A_subrow, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_SeqVectorInitialize_v2(kap_grad, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_SeqVectorInitialize_v2(A_sub, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_Memset(marker, -1, num_rows_diag_A * sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_HOST);

      /* Setting data variables for vectors */
      G_temp_data   = nalu_hypre_VectorData(G_temp);
      A_subrow_data = nalu_hypre_VectorData(A_subrow);

      ii = nalu_hypre_GetThreadNum();
      num_threads = nalu_hypre_NumActiveThreads();
      nalu_hypre_partition1D(num_rows_diag_A, num_threads, ii, &ns, &ne);

      num_rows_Gloc = ne - ns;
      if (num_threads == 1)
      {
         Gloc_i = G_i;
         Gloc_j = G_j;
         Gloc_a = G_a;
      }
      else
      {
         num_nnzs_Gloc = num_rows_Gloc * max_nnzrow_diag_G;

         Gloc_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows_Gloc + 1, NALU_HYPRE_MEMORY_HOST);
         Gloc_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_nnzs_Gloc, NALU_HYPRE_MEMORY_HOST);
         Gloc_a = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, num_nnzs_Gloc, NALU_HYPRE_MEMORY_HOST);
      }

      for (i = ns; i < ne; i++)
      {
         patt_size = 0;

         /* Set old_psi up front so we don't have to compute GAG' twice in the inner for-loop */
         new_psi = old_psi = A_a[A_i[i]];

         /* Cycle through each iteration for that row */
         for (k = 0; k < max_steps; k++)
         {
            /* Compute Kaporin Gradient */
            nalu_hypre_FindKapGrad(A_diag, kap_grad, kg_pos, G_temp, pattern,
                              patt_size, max_nnzrow_diag_G, i, kg_marker);

            /* Find max_step_size largest values of the kaporin gradient,
               find their column indices, and add it to pattern */
            patt_size_old = patt_size;
            nalu_hypre_AddToPattern(kap_grad, kg_pos, pattern, &patt_size,
                               kg_marker, max_step_size);

            /* Update sizes */
            nalu_hypre_VectorSize(A_sub)    = patt_size * patt_size;
            nalu_hypre_VectorSize(A_subrow) = patt_size;
            nalu_hypre_VectorSize(G_temp)   = patt_size;

            if (patt_size == patt_size_old)
            {
               new_psi = old_psi;
               break;
            }
            else
            {
               /* Gather A[P, P] and -A[i, P] */
               for (j = 0; j < patt_size; j++)
               {
                  marker[pattern[j]] = j;
               }
               nalu_hypre_CSRMatrixExtractDenseMat(A_diag, A_sub, pattern, patt_size, marker);
               nalu_hypre_CSRMatrixExtractDenseRow(A_diag, A_subrow, marker, i);

               /* Solve A[P, P] G[i, P]' = -A[i, P] */
               nalu_hypre_DenseSPDSystemSolve(A_sub, A_subrow, G_temp);

               /* Determine psi_{k+1} = G_temp[i] * A[P, P] * G_temp[i]' */
               new_psi = A_a[A_i[i]];
               for (j = 0; j < patt_size; j++)
               {
                  new_psi += G_temp_data[j] * A_subrow_data[j];
               }

               /* Check psi reduction */
               if (nalu_hypre_cabs(new_psi - old_psi) < nalu_hypre_creal(kap_tolerance * old_psi))
               {
                  break;
               }
               else
               {
                  old_psi = new_psi;
               }
            }
         }

         /* Reset marker for building dense linear system */
         for (j = 0; j < patt_size; j++)
         {
            marker[pattern[j]] = -1;
         }

         /* Compute scaling factor */
         if (nalu_hypre_creal(new_psi) > 0 && nalu_hypre_cimag(new_psi) == 0)
         {
            row_scale = 1.0 / nalu_hypre_csqrt(new_psi);
         }
         else
         {
            nalu_hypre_sprintf(msg, "Warning: complex scaling factor found in row %d\n", i);
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, msg);

            row_scale = 1.0 / nalu_hypre_cabs(A_a[A_i[i]]);
            nalu_hypre_VectorSize(G_temp) = patt_size = 0;
         }

         /* Pass values of G_temp into G */
         iloc = i - ns;
         Gloc_j[Gloc_i[iloc]] = i;
         Gloc_a[Gloc_i[iloc]] = row_scale;
         for (k = 0; k < patt_size; k++)
         {
            j = Gloc_i[iloc] + k + 1;
            Gloc_j[j] = pattern[k];
            Gloc_a[j] = row_scale * G_temp_data[k];
            kg_marker[pattern[k]] = 0;
         }
         Gloc_i[iloc + 1] = Gloc_i[iloc] + k + 1;
      }

      /* Copy data to shared memory */
      twspace[ii + 1] = Gloc_i[num_rows_Gloc] - Gloc_i[0];
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
      #pragma omp single
#endif
      {
         for (i = 0; i < num_threads; i++)
         {
            twspace[i + 1] += twspace[i];
         }
      }

      if (num_threads > 1)
      {
         /* Correct row pointer G_i */
         G_i[ns] = twspace[ii];
         for (i = ns; i < ne; i++)
         {
            iloc = i - ns;
            G_i[i + 1] = G_i[i] + Gloc_i[iloc + 1] - Gloc_i[iloc];
         }

         /* Move G_j and G_a */
         for (i = ns; i < ne; i++)
         {
            for (j = G_i[i]; j < G_i[i + 1]; j++)
            {
               G_j[j] = Gloc_j[j - G_i[ns]];
               G_a[j] = Gloc_a[j - G_i[ns]];
            }
         }

         nalu_hypre_TFree(Gloc_i, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(Gloc_j, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_TFree(Gloc_a, NALU_HYPRE_MEMORY_HOST);
      }

      /* Free memory */
      nalu_hypre_SeqVectorDestroy(G_temp);
      nalu_hypre_SeqVectorDestroy(A_subrow);
      nalu_hypre_SeqVectorDestroy(kap_grad);
      nalu_hypre_SeqVectorDestroy(A_sub);
      nalu_hypre_TFree(kg_pos, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(pattern, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(kg_marker, NALU_HYPRE_MEMORY_HOST);
   } /* end openmp region */
   NALU_HYPRE_ANNOTATE_REGION_END("%s", "MainLoop");

   /* Free memory */
   nalu_hypre_TFree(twspace, NALU_HYPRE_MEMORY_HOST);

   /* Update local number of nonzeros of G */
   nalu_hypre_CSRMatrixNumNonzeros(G_diag) = G_i[num_rows_diag_A];

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FSAISetupOMPDyn
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FSAISetupOMPDyn( void               *fsai_vdata,
                       nalu_hypre_ParCSRMatrix *A,
                       nalu_hypre_ParVector    *f,
                       nalu_hypre_ParVector    *u )
{
   /* Data structure variables */
   nalu_hypre_ParFSAIData      *fsai_data        = (nalu_hypre_ParFSAIData*) fsai_vdata;
   NALU_HYPRE_Real              kap_tolerance    = nalu_hypre_ParFSAIDataKapTolerance(fsai_data);
   NALU_HYPRE_Int               max_steps        = nalu_hypre_ParFSAIDataMaxSteps(fsai_data);
   NALU_HYPRE_Int               max_step_size    = nalu_hypre_ParFSAIDataMaxStepSize(fsai_data);

   /* CSRMatrix A_diag variables */
   nalu_hypre_CSRMatrix        *A_diag           = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int              *A_i              = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Complex          *A_a              = nalu_hypre_CSRMatrixData(A_diag);
   NALU_HYPRE_Int               num_rows_diag_A  = nalu_hypre_CSRMatrixNumRows(A_diag);
   NALU_HYPRE_Int               num_nnzs_diag_A  = nalu_hypre_CSRMatrixNumNonzeros(A_diag);
   NALU_HYPRE_Int               avg_nnzrow_diag_A;

   /* Matrix G variables */
   nalu_hypre_ParCSRMatrix     *G = nalu_hypre_ParFSAIDataGmat(fsai_data);
   nalu_hypre_CSRMatrix        *G_diag;
   NALU_HYPRE_Int              *G_i;
   NALU_HYPRE_Int              *G_j;
   NALU_HYPRE_Complex          *G_a;
   NALU_HYPRE_Int              *G_nnzcnt;          /* Array holding number of nonzeros of row G[i,:] */
   NALU_HYPRE_Int               max_nnzrow_diag_G; /* Max. number of nonzeros per row in G_diag */
   NALU_HYPRE_Int               max_cand_size;     /* Max size of kg_pos */

   /* Local variables */
   NALU_HYPRE_Int                i, j, jj;
   char                     msg[512];    /* Warning message */
   NALU_HYPRE_Complex           *twspace;     /* shared work space for omp threads */

   /* Initalize some variables */
   avg_nnzrow_diag_A = num_nnzs_diag_A / num_rows_diag_A;
   max_nnzrow_diag_G = max_steps * max_step_size + 1;
   max_cand_size     = avg_nnzrow_diag_A * max_nnzrow_diag_G;

   G_diag = nalu_hypre_ParCSRMatrixDiag(G);
   G_a = nalu_hypre_CSRMatrixData(G_diag);
   G_i = nalu_hypre_CSRMatrixI(G_diag);
   G_j = nalu_hypre_CSRMatrixJ(G_diag);
   G_nnzcnt = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows_diag_A, NALU_HYPRE_MEMORY_HOST);

   /* Allocate shared work space array for OpenMP threads */
   twspace = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, nalu_hypre_NumThreads() + 1, NALU_HYPRE_MEMORY_HOST);

   /**********************************************************************
   * Start of Adaptive FSAI algorithm
   ***********************************************************************/

   /* Cycle through each of the local rows */
   NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "MainLoop");
#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      nalu_hypre_Vector   *G_temp;        /* Vector holding the values of G[i,:] */
      nalu_hypre_Vector   *A_sub;         /* Vector holding the dense submatrix A[P, P] */
      nalu_hypre_Vector   *A_subrow;      /* Vector holding A[i, P] */
      nalu_hypre_Vector   *kap_grad;      /* Vector holding the Kaporin gradient values */
      NALU_HYPRE_Int      *kg_pos;        /* Indices of nonzero entries of kap_grad */
      NALU_HYPRE_Int      *kg_marker;     /* Marker array with nonzeros pointing to kg_pos */
      NALU_HYPRE_Int      *marker;        /* Marker array with nonzeros pointing to P */
      NALU_HYPRE_Int      *pattern;       /* Array holding column indices of G[i,:] */
      NALU_HYPRE_Int       patt_size;     /* Number of entries in current pattern */
      NALU_HYPRE_Int       patt_size_old; /* Number of entries in previous pattern */
      NALU_HYPRE_Int       i, j, k;       /* Loop variables */
      NALU_HYPRE_Complex   old_psi;       /* GAG' before k-th interation of aFSAI */
      NALU_HYPRE_Complex   new_psi;       /* GAG' after k-th interation of aFSAI */
      NALU_HYPRE_Complex   row_scale;     /* Scaling factor for G_temp */
      NALU_HYPRE_Complex  *G_temp_data;
      NALU_HYPRE_Complex  *A_subrow_data;


      /* Allocate and initialize local vector variables */
      G_temp    = nalu_hypre_SeqVectorCreate(max_nnzrow_diag_G);
      A_subrow  = nalu_hypre_SeqVectorCreate(max_nnzrow_diag_G);
      kap_grad  = nalu_hypre_SeqVectorCreate(max_cand_size);
      A_sub     = nalu_hypre_SeqVectorCreate(max_nnzrow_diag_G * max_nnzrow_diag_G);
      pattern   = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_nnzrow_diag_G, NALU_HYPRE_MEMORY_HOST);
      kg_pos    = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_cand_size, NALU_HYPRE_MEMORY_HOST);
      kg_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows_diag_A, NALU_HYPRE_MEMORY_HOST);
      marker    = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows_diag_A, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_SeqVectorInitialize_v2(G_temp, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_SeqVectorInitialize_v2(A_subrow, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_SeqVectorInitialize_v2(kap_grad, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_SeqVectorInitialize_v2(A_sub, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_Memset(marker, -1, num_rows_diag_A * sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_HOST);

      /* Setting data variables for vectors */
      G_temp_data   = nalu_hypre_VectorData(G_temp);
      A_subrow_data = nalu_hypre_VectorData(A_subrow);

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp for schedule(dynamic)
#endif
      for (i = 0; i < num_rows_diag_A; i++)
      {
         patt_size = 0;

         /* Set old_psi up front so we don't have to compute GAG' twice in the inner for-loop */
         new_psi = old_psi = A_a[A_i[i]];

         /* Cycle through each iteration for that row */
         for (k = 0; k < max_steps; k++)
         {
            /* Compute Kaporin Gradient */
            nalu_hypre_FindKapGrad(A_diag, kap_grad, kg_pos, G_temp, pattern,
                              patt_size, max_nnzrow_diag_G, i, kg_marker);

            /* Find max_step_size largest values of the kaporin gradient,
               find their column indices, and add it to pattern */
            patt_size_old = patt_size;
            nalu_hypre_AddToPattern(kap_grad, kg_pos, pattern, &patt_size,
                               kg_marker, max_step_size);

            /* Update sizes */
            nalu_hypre_VectorSize(A_sub)    = patt_size * patt_size;
            nalu_hypre_VectorSize(A_subrow) = patt_size;
            nalu_hypre_VectorSize(G_temp)   = patt_size;

            if (patt_size == patt_size_old)
            {
               new_psi = old_psi;
               break;
            }
            else
            {
               /* Gather A[P, P] and -A[i, P] */
               for (j = 0; j < patt_size; j++)
               {
                  marker[pattern[j]] = j;
               }
               nalu_hypre_CSRMatrixExtractDenseMat(A_diag, A_sub, pattern, patt_size, marker);
               nalu_hypre_CSRMatrixExtractDenseRow(A_diag, A_subrow, marker, i);

               /* Solve A[P, P] G[i, P]' = -A[i, P] */
               nalu_hypre_DenseSPDSystemSolve(A_sub, A_subrow, G_temp);

               /* Determine psi_{k+1} = G_temp[i] * A[P, P] * G_temp[i]' */
               new_psi = A_a[A_i[i]];
               for (j = 0; j < patt_size; j++)
               {
                  new_psi += G_temp_data[j] * A_subrow_data[j];
               }

               /* Check psi reduction */
               if (nalu_hypre_cabs(new_psi - old_psi) < nalu_hypre_creal(kap_tolerance * old_psi))
               {
                  break;
               }
               else
               {
                  old_psi = new_psi;
               }
            }
         }

         /* Reset marker for building dense linear system */
         for (j = 0; j < patt_size; j++)
         {
            marker[pattern[j]] = -1;
         }

         /* Compute scaling factor */
         if (nalu_hypre_creal(new_psi) > 0 && nalu_hypre_cimag(new_psi) == 0)
         {
            row_scale = 1.0 / nalu_hypre_csqrt(new_psi);
         }
         else
         {
            nalu_hypre_sprintf(msg, "Warning: complex scaling factor found in row %d\n", i);
            nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, msg);

            row_scale = 1.0 / nalu_hypre_cabs(A_a[A_i[i]]);
            nalu_hypre_VectorSize(G_temp) = patt_size = 0;
         }

         /* Pass values of G_temp into G */
         j = i * max_nnzrow_diag_G;
         G_j[j] = i;
         G_a[j] = row_scale;
         j++;
         for (k = 0; k < patt_size; k++)
         {
            G_j[j] = pattern[k];
            G_a[j++] = row_scale * G_temp_data[k];
            kg_marker[pattern[k]] = 0;
         }
         G_nnzcnt[i] = patt_size + 1;
      } /* omp for schedule(dynamic) */

      /* Free memory */
      nalu_hypre_SeqVectorDestroy(G_temp);
      nalu_hypre_SeqVectorDestroy(A_subrow);
      nalu_hypre_SeqVectorDestroy(kap_grad);
      nalu_hypre_SeqVectorDestroy(A_sub);
      nalu_hypre_TFree(kg_pos, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(pattern, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(kg_marker, NALU_HYPRE_MEMORY_HOST);
   } /* end openmp region */
   NALU_HYPRE_ANNOTATE_REGION_END("%s", "MainLoop");

   /* Reorder array */
   G_i[0] = 0;
   for (i = 0; i < num_rows_diag_A; i++)
   {
      G_i[i + 1] = G_i[i] + G_nnzcnt[i];
      jj = i * max_nnzrow_diag_G;
      for (j = G_i[i]; j < G_i[i + 1]; j++)
      {
         G_j[j] = G_j[jj];
         G_a[j] = G_a[jj++];
      }
   }

   /* Free memory */
   nalu_hypre_TFree(twspace, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(G_nnzcnt, NALU_HYPRE_MEMORY_HOST);

   /* Update local number of nonzeros of G */
   nalu_hypre_CSRMatrixNumNonzeros(G_diag) = G_i[num_rows_diag_A];

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FSAISetup
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FSAISetup( void               *fsai_vdata,
                 nalu_hypre_ParCSRMatrix *A,
                 nalu_hypre_ParVector    *f,
                 nalu_hypre_ParVector    *u )
{
   nalu_hypre_ParFSAIData       *fsai_data     = (nalu_hypre_ParFSAIData*) fsai_vdata;
   NALU_HYPRE_Int                max_steps     = nalu_hypre_ParFSAIDataMaxSteps(fsai_data);
   NALU_HYPRE_Int                max_step_size = nalu_hypre_ParFSAIDataMaxStepSize(fsai_data);
   NALU_HYPRE_Int                max_nnz_row   = nalu_hypre_ParFSAIDataMaxNnzRow(fsai_data);
   NALU_HYPRE_Int                algo_type     = nalu_hypre_ParFSAIDataAlgoType(fsai_data);
   NALU_HYPRE_Int                print_level   = nalu_hypre_ParFSAIDataPrintLevel(fsai_data);
   NALU_HYPRE_Int                eig_max_iters = nalu_hypre_ParFSAIDataEigMaxIters(fsai_data);

   /* ParCSRMatrix A variables */
   MPI_Comm                 comm          = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_BigInt             num_rows_A    = nalu_hypre_ParCSRMatrixGlobalNumRows(A);
   NALU_HYPRE_BigInt             num_cols_A    = nalu_hypre_ParCSRMatrixGlobalNumCols(A);
   NALU_HYPRE_BigInt            *row_starts_A  = nalu_hypre_ParCSRMatrixRowStarts(A);
   NALU_HYPRE_BigInt            *col_starts_A  = nalu_hypre_ParCSRMatrixColStarts(A);

   /* CSRMatrix A_diag variables */
   nalu_hypre_CSRMatrix         *A_diag           = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int                num_rows_diag_A  = nalu_hypre_CSRMatrixNumRows(A_diag);

   /* Work vectors */
   nalu_hypre_ParVector         *r_work;
   nalu_hypre_ParVector         *z_work;

   /* G variables */
   nalu_hypre_ParCSRMatrix      *G;
   NALU_HYPRE_Int                max_nnzrow_diag_G;   /* Max. number of nonzeros per row in G_diag */
   NALU_HYPRE_Int                max_nonzeros_diag_G; /* Max. number of nonzeros in G_diag */

   /* Sanity check */
   if (f && nalu_hypre_ParVectorNumVectors(f) > 1)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "FSAI doesn't support multicomponent vectors");
      return nalu_hypre_error_flag;
   }

   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Create and initialize work vectors used in the solve phase */
   r_work = nalu_hypre_ParVectorCreate(comm, num_rows_A, row_starts_A);
   z_work = nalu_hypre_ParVectorCreate(comm, num_rows_A, row_starts_A);

   nalu_hypre_ParVectorInitialize(r_work);
   nalu_hypre_ParVectorInitialize(z_work);

   nalu_hypre_ParFSAIDataRWork(fsai_data) = r_work;
   nalu_hypre_ParFSAIDataZWork(fsai_data) = z_work;

   /* Create the matrix G */
   if (algo_type == 1 || algo_type == 2)
   {
      max_nnzrow_diag_G = max_steps * max_step_size + 1;
   }
   else
   {
      max_nnzrow_diag_G = max_nnz_row + 1;
   }
   max_nonzeros_diag_G = num_rows_diag_A * max_nnzrow_diag_G;
   G = nalu_hypre_ParCSRMatrixCreate(comm, num_rows_A, num_cols_A,
                                row_starts_A, col_starts_A,
                                0, max_nonzeros_diag_G, 0);
   nalu_hypre_ParFSAIDataGmat(fsai_data) = G;

   /* Initialize and compute lower triangular factor G */
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_MemoryLocation  memloc_A = nalu_hypre_ParCSRMatrixMemoryLocation(A);
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1(memloc_A);

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_FSAISetupDevice(fsai_vdata, A, f, u);
   }
   else
#endif
   {
      /* Initialize matrix */
      nalu_hypre_ParCSRMatrixInitialize(G);

      switch (algo_type)
      {
         case 1:
            // TODO: Change name to nalu_hypre_FSAISetupAdaptive
            nalu_hypre_FSAISetupNative(fsai_vdata, A, f, u);
            break;

         case 2:
            // TODO: Change name to nalu_hypre_FSAISetupAdaptiveOMPDynamic
            nalu_hypre_FSAISetupOMPDyn(fsai_vdata, A, f, u);
            break;

         default:
            nalu_hypre_FSAISetupNative(fsai_vdata, A, f, u);
            break;
      }
   }

   /* Compute G^T */
   G  = nalu_hypre_ParFSAIDataGmat(fsai_data);
   nalu_hypre_ParCSRMatrixTranspose(G, &nalu_hypre_ParFSAIDataGTmat(fsai_data), 1);

   /* Update omega if requested */
   if (eig_max_iters)
   {
      nalu_hypre_FSAIComputeOmega(fsai_vdata, A);
   }

   /* Print setup info */
   if (print_level == 1)
   {
      nalu_hypre_FSAIPrintStats(fsai_data, A);
   }
   else if (print_level > 2)
   {
      char filename[] = "FSAI.out.G.ij";
      nalu_hypre_ParCSRMatrixPrintIJ(G, 0, 0, filename);
   }

#if defined (DEBUG_FSAI)
#if !defined (NALU_HYPRE_USING_GPU) ||
   (defined (NALU_HYPRE_USING_GPU) && defined (NALU_HYPRE_USING_UNIFIED_MEMORY))
   nalu_hypre_FSAIDumpLocalLSDense(fsai_vdata, "fsai_dense_ls.out", A);
#endif
#endif

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FSAIPrintStats
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FSAIPrintStats( void *fsai_vdata,
                      nalu_hypre_ParCSRMatrix *A )
{
   /* Data structure variables */
   nalu_hypre_ParFSAIData      *fsai_data        = (nalu_hypre_ParFSAIData*) fsai_vdata;
   NALU_HYPRE_Int               algo_type        = nalu_hypre_ParFSAIDataAlgoType(fsai_data);
   NALU_HYPRE_Int               local_solve_type = nalu_hypre_ParFSAIDataLocalSolveType(fsai_data);
   NALU_HYPRE_Real              kap_tolerance    = nalu_hypre_ParFSAIDataKapTolerance(fsai_data);
   NALU_HYPRE_Int               max_steps        = nalu_hypre_ParFSAIDataMaxSteps(fsai_data);
   NALU_HYPRE_Int               max_step_size    = nalu_hypre_ParFSAIDataMaxStepSize(fsai_data);
   NALU_HYPRE_Int               max_nnz_row      = nalu_hypre_ParFSAIDataMaxNnzRow(fsai_data);
   NALU_HYPRE_Int               num_levels       = nalu_hypre_ParFSAIDataNumLevels(fsai_data);
   NALU_HYPRE_Real              threshold        = nalu_hypre_ParFSAIDataThreshold(fsai_data);
   NALU_HYPRE_Int               eig_max_iters    = nalu_hypre_ParFSAIDataEigMaxIters(fsai_data);
   NALU_HYPRE_Real              density;

   nalu_hypre_ParCSRMatrix     *G = nalu_hypre_ParFSAIDataGmat(fsai_data);

   /* Local variables */
   NALU_HYPRE_Int               nprocs;
   NALU_HYPRE_Int               my_id;

   nalu_hypre_MPI_Comm_size(nalu_hypre_ParCSRMatrixComm(A), &nprocs);
   nalu_hypre_MPI_Comm_rank(nalu_hypre_ParCSRMatrixComm(A), &my_id);

   /* Compute density */
   nalu_hypre_ParCSRMatrixSetDNumNonzeros(G);
   nalu_hypre_ParCSRMatrixSetDNumNonzeros(A);
   density = nalu_hypre_ParCSRMatrixDNumNonzeros(G) /
             nalu_hypre_ParCSRMatrixDNumNonzeros(A);
   nalu_hypre_ParFSAIDataDensity(fsai_data) = density;

   if (!my_id)
   {
      nalu_hypre_printf("*************************\n");
      nalu_hypre_printf("* NALU_HYPRE FSAI Setup Info *\n");
      nalu_hypre_printf("*************************\n\n");

      nalu_hypre_printf("+---------------------------+\n");
      nalu_hypre_printf("| No. MPI tasks:     %6d |\n", nprocs);
      nalu_hypre_printf("| No. threads:       %6d |\n", nalu_hypre_NumThreads());
      nalu_hypre_printf("| Algorithm type:    %6d |\n", algo_type);
      nalu_hypre_printf("| Local solve type:  %6d |\n", local_solve_type);
      if (algo_type == 1 || algo_type == 2)
      {
         nalu_hypre_printf("| Max no. steps:     %6d |\n", max_steps);
         nalu_hypre_printf("| Max step size:     %6d |\n", max_step_size);
         nalu_hypre_printf("| Kap grad tol:    %8.1e |\n", kap_tolerance);
      }
      else
      {
         nalu_hypre_printf("| Max nnz. row:      %6d |\n", max_nnz_row);
         nalu_hypre_printf("| Number of levels:  %6d |\n", num_levels);
         nalu_hypre_printf("| Threshold:       %8.1e |\n", threshold);
      }
      nalu_hypre_printf("| Prec. density:   %8.3f |\n", density);
      nalu_hypre_printf("| Eig max iters:     %6d |\n", eig_max_iters);
      nalu_hypre_printf("| Omega factor:    %8.3f |\n", nalu_hypre_ParFSAIDataOmega(fsai_data));
      nalu_hypre_printf("+---------------------------+\n");

      nalu_hypre_printf("\n\n");
   }

   return nalu_hypre_error_flag;
}

/*****************************************************************************
 * nalu_hypre_FSAIComputeOmega
 *
 * Approximates the relaxation factor omega with 1/eigmax(G^T*G*A), where the
 * maximum eigenvalue is computed with a fixed number of iterations via the
 * power method.
 ******************************************************************************/

NALU_HYPRE_Int
nalu_hypre_FSAIComputeOmega( void               *fsai_vdata,
                        nalu_hypre_ParCSRMatrix *A )
{
   nalu_hypre_ParFSAIData    *fsai_data       = (nalu_hypre_ParFSAIData*) fsai_vdata;
   nalu_hypre_ParCSRMatrix   *G               = nalu_hypre_ParFSAIDataGmat(fsai_data);
   nalu_hypre_ParCSRMatrix   *GT              = nalu_hypre_ParFSAIDataGTmat(fsai_data);
   nalu_hypre_ParVector      *r_work          = nalu_hypre_ParFSAIDataRWork(fsai_data);
   nalu_hypre_ParVector      *z_work          = nalu_hypre_ParFSAIDataZWork(fsai_data);
   NALU_HYPRE_Int             eig_max_iters   = nalu_hypre_ParFSAIDataEigMaxIters(fsai_data);
   NALU_HYPRE_MemoryLocation  memory_location = nalu_hypre_ParCSRMatrixMemoryLocation(A);

   nalu_hypre_ParVector      *eigvec;
   nalu_hypre_ParVector      *eigvec_old;

   NALU_HYPRE_Int             i;
   NALU_HYPRE_Real            norm, invnorm, lambda, omega;

   eigvec_old = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                      nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                      nalu_hypre_ParCSRMatrixRowStarts(A));
   eigvec = nalu_hypre_ParVectorCreate(nalu_hypre_ParCSRMatrixComm(A),
                                  nalu_hypre_ParCSRMatrixGlobalNumRows(A),
                                  nalu_hypre_ParCSRMatrixRowStarts(A));
   nalu_hypre_ParVectorInitialize_v2(eigvec, memory_location);
   nalu_hypre_ParVectorInitialize_v2(eigvec_old, memory_location);

#if defined(NALU_HYPRE_USING_GPU)
   /* Make random number generation faster on GPUs */
   if (nalu_hypre_GetExecPolicy1(memory_location) == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_Vector  *eigvec_local = nalu_hypre_ParVectorLocalVector(eigvec);
      NALU_HYPRE_Complex *eigvec_data  = nalu_hypre_VectorData(eigvec_local);
      NALU_HYPRE_Int      eigvec_size  = nalu_hypre_VectorSize(eigvec_local);

      nalu_hypre_CurandUniform(eigvec_size, eigvec_data, 0, 0, 0, 0);
   }
   else
#endif
   {
      nalu_hypre_ParVectorSetRandomValues(eigvec, 256);
   }

   /* Power method iteration */
   for (i = 0; i < eig_max_iters; i++)
   {
      norm = nalu_hypre_ParVectorInnerProd(eigvec, eigvec);
      invnorm = 1.0 / nalu_hypre_sqrt(norm);
      nalu_hypre_ParVectorScale(invnorm, eigvec);

      if (i == (eig_max_iters - 1))
      {
         nalu_hypre_ParVectorCopy(eigvec, eigvec_old);
      }

      /* eigvec = GT * G * A * eigvec */
      nalu_hypre_ParCSRMatrixMatvec(1.0, A,  eigvec, 0.0, r_work);
      nalu_hypre_ParCSRMatrixMatvec(1.0, G,  r_work, 0.0, z_work);
      nalu_hypre_ParCSRMatrixMatvec(1.0, GT, z_work, 0.0, eigvec);
   }
   norm = nalu_hypre_ParVectorInnerProd(eigvec, eigvec_old);
   lambda = nalu_hypre_sqrt(norm);

   /* Check lambda */
   if (lambda < NALU_HYPRE_REAL_EPSILON)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Found small lambda. Reseting it to one!");
      lambda = 1.0;
   }

   /* Free memory */
   nalu_hypre_ParVectorDestroy(eigvec_old);
   nalu_hypre_ParVectorDestroy(eigvec);

   /* Update omega */
   omega = 1.0 / lambda;
   nalu_hypre_FSAISetOmega(fsai_vdata, omega);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_FSAIDumpLocalLSDense
 *
 * Dump local linear systems to file. Matrices are written in dense format.
 * This functions serves for debugging.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_FSAIDumpLocalLSDense( void               *fsai_vdata,
                            const char         *filename,
                            nalu_hypre_ParCSRMatrix *A )
{
   nalu_hypre_ParFSAIData      *fsai_data = (nalu_hypre_ParFSAIData*) fsai_vdata;

   /* Data structure variables */
   MPI_Comm                comm = nalu_hypre_ParCSRMatrixComm(A);
   NALU_HYPRE_Int               max_steps = nalu_hypre_ParFSAIDataMaxSteps(fsai_data);
   NALU_HYPRE_Int               max_step_size = nalu_hypre_ParFSAIDataMaxStepSize(fsai_data);
   nalu_hypre_ParCSRMatrix     *G = nalu_hypre_ParFSAIDataGmat(fsai_data);
   nalu_hypre_CSRMatrix        *G_diag = nalu_hypre_ParCSRMatrixDiag(G);
   NALU_HYPRE_Int              *G_i = nalu_hypre_CSRMatrixI(G_diag);
   NALU_HYPRE_Int              *G_j = nalu_hypre_CSRMatrixJ(G_diag);
   NALU_HYPRE_Int               num_rows_diag_G = nalu_hypre_CSRMatrixNumRows(G_diag);

   /* CSRMatrix A_diag variables */
   nalu_hypre_CSRMatrix        *A_diag           = nalu_hypre_ParCSRMatrixDiag(A);
   NALU_HYPRE_Int              *A_i              = nalu_hypre_CSRMatrixI(A_diag);
   NALU_HYPRE_Int              *A_j              = nalu_hypre_CSRMatrixJ(A_diag);
   NALU_HYPRE_Complex          *A_a              = nalu_hypre_CSRMatrixData(A_diag);

   FILE                   *fp;
   char                    new_filename[1024];
   NALU_HYPRE_Int               myid;
   NALU_HYPRE_Int               i, j, k, m, n;
   NALU_HYPRE_Int               ii, jj;
   NALU_HYPRE_Int               nnz, col, index;
   NALU_HYPRE_Int              *indices;
   NALU_HYPRE_Int              *marker;
   NALU_HYPRE_Real             *data;
   NALU_HYPRE_Int               data_size;
   NALU_HYPRE_Real              density;
   NALU_HYPRE_Int               width = 20; //6
   NALU_HYPRE_Int               prec  = 16; //2

   nalu_hypre_MPI_Comm_rank(comm, &myid);
   nalu_hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((fp = fopen(new_filename, "w")) == NULL)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error: can't open output file %s\n");
      return nalu_hypre_error_flag;
   }

   /* Allocate memory */
   data_size = (max_steps * max_step_size) *
               (max_steps * max_step_size + 1);
   indices = nalu_hypre_CTAlloc(NALU_HYPRE_Int, data_size, NALU_HYPRE_MEMORY_HOST);
   data    = nalu_hypre_CTAlloc(NALU_HYPRE_Real, data_size, NALU_HYPRE_MEMORY_HOST);
   marker  = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows_diag_G, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_Memset(marker, -1, num_rows_diag_G * sizeof(NALU_HYPRE_Int), NALU_HYPRE_MEMORY_HOST);

   /* Write header info */
   nalu_hypre_fprintf(fp, "num_linear_sys = %d\n", num_rows_diag_G);
   nalu_hypre_fprintf(fp, "max_data_size = %d\n", data_size);
   nalu_hypre_fprintf(fp, "max_num_steps = %d\n", nalu_hypre_ParFSAIDataMaxSteps(fsai_data));
   nalu_hypre_fprintf(fp, "max_step_size = %d\n", nalu_hypre_ParFSAIDataMaxStepSize(fsai_data));
   nalu_hypre_fprintf(fp, "max_step_size = %g\n", nalu_hypre_ParFSAIDataKapTolerance(fsai_data));
   nalu_hypre_fprintf(fp, "algo_type = %d\n\n", nalu_hypre_ParFSAIDataAlgoType(fsai_data));

   /* Write local full linear systems */
   for (i = 0; i < num_rows_diag_G; i++)
   {
      /* Build marker array */
      n = G_i[i + 1] - G_i[i] - 1;
      m = n + 1;
      for (j = (G_i[i] + 1); j < G_i[i + 1]; j++)
      {
         marker[G_j[j]] = j - G_i[i] - 1;
      }

      /* Gather matrix coefficients */
      nnz = 0;
      for (j = (G_i[i] + 1); j < G_i[i + 1]; j++)
      {
         for (k = A_i[G_j[j]]; k < A_i[G_j[j] + 1]; k++)
         {
            if ((col = marker[A_j[k]]) >= 0)
            {
               /* Add A(i,j) entry */
               index = (j - G_i[i] - 1) * n + col;
               data[index] = A_a[k];
               indices[nnz] = index;
               nnz++;
            }
         }
      }
      density = (n > 0) ? (NALU_HYPRE_Real) nnz / (n * n) : 0.0;

      /* Gather RHS coefficients */
      for (j = A_i[i]; j < A_i[i + 1]; j++)
      {
         if ((col = marker[A_j[j]]) >= 0)
         {
            index = (m - 1) * n + col;
            data[index] = A_a[j];
            indices[nnz] = index;
            nnz++;
         }
      }

      /* Write coefficients to file */
      nalu_hypre_fprintf(fp, "id = %d, (m, n) = (%d, %d), rho = %.3f\n", i, m, n, density);
      for (ii = 0; ii < n; ii++)
      {
         for (jj = 0; jj < n; jj++)
         {
            nalu_hypre_fprintf(fp, "%*.*f ", width, prec, data[ii * n + jj]);
         }
         nalu_hypre_fprintf(fp, "\n");
      }
      for (jj = 0; jj < n; jj++)
      {
         nalu_hypre_fprintf(fp, "%*.*f ", width, prec, data[ii * n + jj]);
      }
      nalu_hypre_fprintf(fp, "\n");


      /* Reset work arrays */
      for (j = (G_i[i] + 1); j < G_i[i + 1]; j++)
      {
         marker[G_j[j]] = -1;
      }

      for (k = 0; k < nnz; k++)
      {
         data[indices[k]] = 0.0;
      }
   }

   /* Close stream */
   fclose(fp);

   /* Free memory */
   nalu_hypre_TFree(indices, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(data, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}
