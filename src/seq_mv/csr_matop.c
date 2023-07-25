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

#include "seq_mv.h"
#include "csr_matrix.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixAddFirstPass:
 *
 * Performs the first pass needed for Matrix/Matrix addition (C = A + B).
 * This function:
 *    1) Computes the row pointer of the resulting matrix C_i
 *    2) Allocates memory for the matrix C and returns it to the user
 *
 * Notes: 1) It can be used safely inside OpenMP parallel regions.
 *        2) firstrow, lastrow and marker are private variables.
 *        3) The remaining arguments are shared variables.
 *        4) twspace (thread workspace) must be allocated outside the
 *           parallel region.
 *        5) The mapping arrays map_A2C and map_B2C are used when adding
 *           off-diagonal matrices. They can be set to NULL pointer when
 *           adding diagonal matrices.
 *        6) Assumes that the elements of C_i are initialized to zero.
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_CSRMatrixAddFirstPass( NALU_HYPRE_Int              firstrow,
                             NALU_HYPRE_Int              lastrow,
                             NALU_HYPRE_Int             *twspace,
                             NALU_HYPRE_Int             *marker,
                             NALU_HYPRE_Int             *map_A2C,
                             NALU_HYPRE_Int             *map_B2C,
                             nalu_hypre_CSRMatrix       *A,
                             nalu_hypre_CSRMatrix       *B,
                             NALU_HYPRE_Int              nrows_C,
                             NALU_HYPRE_Int              nnzrows_C,
                             NALU_HYPRE_Int              ncols_C,
                             NALU_HYPRE_Int             *rownnz_C,
                             NALU_HYPRE_MemoryLocation   memory_location_C,
                             NALU_HYPRE_Int             *C_i,
                             nalu_hypre_CSRMatrix      **C_ptr )
{
   NALU_HYPRE_Int   *A_i = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int   *A_j = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int   *B_i = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int   *B_j = nalu_hypre_CSRMatrixJ(B);

   NALU_HYPRE_Int    i, ia, ib, ic, iic, ii, i1;
   NALU_HYPRE_Int    jcol, jj;
   NALU_HYPRE_Int    num_threads = nalu_hypre_NumActiveThreads();
   NALU_HYPRE_Int    num_nonzeros;

   /* Initialize marker array */
   for (i = 0; i < ncols_C; i++)
   {
      marker[i] = -1;
   }

   ii = nalu_hypre_GetThreadNum();
   num_nonzeros = 0;
   for (ic = firstrow; ic < lastrow; ic++)
   {
      iic = rownnz_C ? rownnz_C[ic] : ic;

      if (map_A2C)
      {
         for (ia = A_i[iic]; ia < A_i[iic + 1]; ia++)
         {
            jcol = map_A2C[A_j[ia]];
            marker[jcol] = iic;
            num_nonzeros++;
         }
      }
      else
      {
         for (ia = A_i[iic]; ia < A_i[iic + 1]; ia++)
         {
            jcol = A_j[ia];
            marker[jcol] = iic;
            num_nonzeros++;
         }
      }

      if (map_B2C)
      {
         for (ib = B_i[iic]; ib < B_i[iic + 1]; ib++)
         {
            jcol = map_B2C[B_j[ib]];
            if (marker[jcol] != iic)
            {
               marker[jcol] = iic;
               num_nonzeros++;
            }
         }
      }
      else
      {
         for (ib = B_i[iic]; ib < B_i[iic + 1]; ib++)
         {
            jcol = B_j[ib];
            if (marker[jcol] != iic)
            {
               marker[jcol] = iic;
               num_nonzeros++;
            }
         }
      }
      C_i[iic + 1] = num_nonzeros;
   }
   twspace[ii] = num_nonzeros;

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp barrier
#endif

   /* Correct C_i - phase 1 */
   if (ii)
   {
      jj = twspace[0];
      for (i1 = 1; i1 < ii; i1++)
      {
         jj += twspace[i1];
      }

      for (ic = firstrow; ic < lastrow; ic++)
      {
         iic = rownnz_C ? rownnz_C[ic] : ic;
         C_i[iic + 1] += jj;
      }
   }
   else
   {
      num_nonzeros = 0;
      for (i1 = 0; i1 < num_threads; i1++)
      {
         num_nonzeros += twspace[i1];
      }

      *C_ptr = nalu_hypre_CSRMatrixCreate(nrows_C, ncols_C, num_nonzeros);
      nalu_hypre_CSRMatrixI(*C_ptr) = C_i;
      nalu_hypre_CSRMatrixRownnz(*C_ptr) = rownnz_C;
      nalu_hypre_CSRMatrixNumRownnz(*C_ptr) = nnzrows_C;
      nalu_hypre_CSRMatrixInitialize_v2(*C_ptr, 0, memory_location_C);
   }

   /* Correct C_i - phase 2 */
   if (rownnz_C != NULL)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif
      for (ic = firstrow; ic < (lastrow - 1); ic++)
      {
         for (iic = rownnz_C[ic] + 1; iic < rownnz_C[ic + 1]; iic++)
         {
            nalu_hypre_assert(C_i[iic + 1] == 0);
            C_i[iic + 1] = C_i[rownnz_C[ic] + 1];
         }
      }

      if (ii < (num_threads - 1))
      {
         for (iic = rownnz_C[lastrow - 1] + 1; iic < rownnz_C[lastrow]; iic++)
         {
            nalu_hypre_assert(C_i[iic + 1] == 0);
            C_i[iic + 1] = C_i[rownnz_C[lastrow - 1] + 1];
         }
      }
      else
      {
         for (iic = rownnz_C[lastrow - 1] + 1; iic < nrows_C; iic++)
         {
            nalu_hypre_assert(C_i[iic + 1] == 0);
            C_i[iic + 1] = C_i[rownnz_C[lastrow - 1] + 1];
         }
      }
   }

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp barrier
#endif

#ifdef NALU_HYPRE_DEBUG
   if (!ii)
   {
      for (i = 0; i < nrows_C; i++)
      {
         nalu_hypre_assert(C_i[i] <= C_i[i + 1]);
         nalu_hypre_assert(((A_i[i + 1] - A_i[i]) +
                       (B_i[i + 1] - B_i[i])) >=
                      (C_i[i + 1] - C_i[i]));
         nalu_hypre_assert((C_i[i + 1] - C_i[i]) >= (A_i[i + 1] - A_i[i]));
         nalu_hypre_assert((C_i[i + 1] - C_i[i]) >= (B_i[i + 1] - B_i[i]));
      }
      nalu_hypre_assert((C_i[nrows_C] - C_i[0]) == num_nonzeros);
   }
#endif

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixAddSecondPass:
 *
 * Performs the second pass needed for Matrix/Matrix addition (C = A + B).
 * This function computes C_j and C_data.
 *
 * Notes: see notes for nalu_hypre_CSRMatrixAddFirstPass
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_CSRMatrixAddSecondPass( NALU_HYPRE_Int          firstrow,
                              NALU_HYPRE_Int          lastrow,
                              NALU_HYPRE_Int         *twspace,
                              NALU_HYPRE_Int         *marker,
                              NALU_HYPRE_Int         *map_A2C,
                              NALU_HYPRE_Int         *map_B2C,
                              NALU_HYPRE_Int         *rownnz_C,
                              NALU_HYPRE_Complex      alpha,
                              NALU_HYPRE_Complex      beta,
                              nalu_hypre_CSRMatrix   *A,
                              nalu_hypre_CSRMatrix   *B,
                              nalu_hypre_CSRMatrix   *C )
{
   NALU_HYPRE_Int        *A_i      = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int        *A_j      = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Complex    *A_data   = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int         nnzs_A   = nalu_hypre_CSRMatrixNumNonzeros(A);

   NALU_HYPRE_Int        *B_i      = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int        *B_j      = nalu_hypre_CSRMatrixJ(B);
   NALU_HYPRE_Complex    *B_data   = nalu_hypre_CSRMatrixData(B);
   NALU_HYPRE_Int         nnzs_B   = nalu_hypre_CSRMatrixNumNonzeros(B);

   NALU_HYPRE_Int        *C_i      = nalu_hypre_CSRMatrixI(C);
   NALU_HYPRE_Int        *C_j      = nalu_hypre_CSRMatrixJ(C);
   NALU_HYPRE_Complex    *C_data   = nalu_hypre_CSRMatrixData(C);
   NALU_HYPRE_Int         ncols_C  = nalu_hypre_CSRMatrixNumCols(C);

   NALU_HYPRE_Int         ia, ib, ic, iic;
   NALU_HYPRE_Int         jcol, pos;

   nalu_hypre_assert(( map_A2C &&  map_B2C) ||
                (!map_A2C && !map_B2C) ||
                ( map_A2C && (nnzs_B == 0)) ||
                ( map_B2C && (nnzs_A == 0)));

   /* Initialize marker vector */
   for (ia = 0; ia < ncols_C; ia++)
   {
      marker[ia] = -1;
   }

   pos = C_i[rownnz_C ? rownnz_C[firstrow] : firstrow];
   if ((map_A2C && map_B2C) || ( map_A2C && (nnzs_B == 0)) || ( map_B2C && (nnzs_A == 0)))
   {
      for (ic = firstrow; ic < lastrow; ic++)
      {
         iic = rownnz_C ? rownnz_C[ic] : ic;

         for (ia = A_i[iic]; ia < A_i[iic + 1]; ia++)
         {
            jcol = map_A2C[A_j[ia]];
            C_j[pos] = jcol;
            C_data[pos] = alpha * A_data[ia];
            marker[jcol] = pos;
            pos++;
         }

         for (ib = B_i[iic]; ib < B_i[iic + 1]; ib++)
         {
            jcol = map_B2C[B_j[ib]];
            if (marker[jcol] < C_i[iic])
            {
               C_j[pos] = jcol;
               C_data[pos] = beta * B_data[ib];
               marker[jcol] = pos;
               pos++;
            }
            else
            {
               nalu_hypre_assert(C_j[marker[jcol]] == jcol);
               C_data[marker[jcol]] += beta * B_data[ib];
            }
         }
         nalu_hypre_assert(pos == C_i[iic + 1]);
      } /* end for loop */
   }
   else
   {
      for (ic = firstrow; ic < lastrow; ic++)
      {
         iic = rownnz_C ? rownnz_C[ic] : ic;

         for (ia = A_i[iic]; ia < A_i[iic + 1]; ia++)
         {
            jcol = A_j[ia];
            C_j[pos] = jcol;
            C_data[pos] = alpha * A_data[ia];
            marker[jcol] = pos;
            pos++;
         }

         for (ib = B_i[iic]; ib < B_i[iic + 1]; ib++)
         {
            jcol = B_j[ib];
            if (marker[jcol] < C_i[iic])
            {
               C_j[pos] = jcol;
               C_data[pos] = beta * B_data[ib];
               marker[jcol] = pos;
               pos++;
            }
            else
            {
               nalu_hypre_assert(C_j[marker[jcol]] == jcol);
               C_data[marker[jcol]] += beta * B_data[ib];
            }
         }
         nalu_hypre_assert(pos == C_i[iic + 1]);
      } /* end for loop */
   }
   nalu_hypre_assert(pos == C_i[rownnz_C ? rownnz_C[lastrow - 1] + 1 : lastrow]);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixAdd:
 *
 * Adds two CSR Matrices A and B and returns a CSR Matrix C = alpha*A + beta*B;
 *
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 *       in A and B. To remove those, use nalu_hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix*
nalu_hypre_CSRMatrixAddHost ( NALU_HYPRE_Complex    alpha,
                         nalu_hypre_CSRMatrix *A,
                         NALU_HYPRE_Complex    beta,
                         nalu_hypre_CSRMatrix *B )
{
   /* CSRMatrix A */
   NALU_HYPRE_Int        *rownnz_A  = nalu_hypre_CSRMatrixRownnz(A);
   NALU_HYPRE_Int         nrows_A   = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int         nnzrows_A = nalu_hypre_CSRMatrixNumRownnz(A);
   NALU_HYPRE_Int         ncols_A   = nalu_hypre_CSRMatrixNumCols(A);

   /* CSRMatrix B */
   NALU_HYPRE_Int        *rownnz_B  = nalu_hypre_CSRMatrixRownnz(B);
   NALU_HYPRE_Int         nrows_B   = nalu_hypre_CSRMatrixNumRows(B);
   NALU_HYPRE_Int         nnzrows_B = nalu_hypre_CSRMatrixNumRownnz(B);
   NALU_HYPRE_Int         ncols_B   = nalu_hypre_CSRMatrixNumCols(B);

   /* CSRMatrix C */
   nalu_hypre_CSRMatrix  *C;
   NALU_HYPRE_Int        *C_i;
   NALU_HYPRE_Int        *rownnz_C;
   NALU_HYPRE_Int         nnzrows_C;

   NALU_HYPRE_Int        *twspace;

   NALU_HYPRE_MemoryLocation memory_location_A = nalu_hypre_CSRMatrixMemoryLocation(A);
   NALU_HYPRE_MemoryLocation memory_location_B = nalu_hypre_CSRMatrixMemoryLocation(B);

   /* RL: TODO cannot guarantee, maybe should never assert
   nalu_hypre_assert(memory_location_A == memory_location_B);
   */

   /* RL: in the case of A=H, B=D, or A=D, B=H, let C = D,
    * not sure if this is the right thing to do.
    * Also, need something like this in other places
    * TODO */
   NALU_HYPRE_MemoryLocation memory_location_C = nalu_hypre_max(memory_location_A, memory_location_B);

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   /* Allocate memory */
   twspace = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_NumThreads(), NALU_HYPRE_MEMORY_HOST);
   C_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows_A + 1, memory_location_C);

   /* Set nonzero rows data of diag_C */
   nnzrows_C = nrows_A;
   if ((nnzrows_A < nrows_A) && (nnzrows_B < nrows_B))
   {
      nalu_hypre_IntArray arr_A;
      nalu_hypre_IntArray arr_B;
      nalu_hypre_IntArray arr_C;

      nalu_hypre_IntArrayData(&arr_A) = rownnz_A;
      nalu_hypre_IntArrayData(&arr_B) = rownnz_B;
      nalu_hypre_IntArraySize(&arr_A) = nnzrows_A;
      nalu_hypre_IntArraySize(&arr_B) = nnzrows_B;
      nalu_hypre_IntArrayMemoryLocation(&arr_C) = memory_location_C;

      nalu_hypre_IntArrayMergeOrdered(&arr_A, &arr_B, &arr_C);

      nnzrows_C = nalu_hypre_IntArraySize(&arr_C);
      rownnz_C  = nalu_hypre_IntArrayData(&arr_C);
   }
   else
   {
      rownnz_C = NULL;
   }

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      NALU_HYPRE_Int   ns, ne;
      NALU_HYPRE_Int  *marker = NULL;

      nalu_hypre_partition1D(nnzrows_C, nalu_hypre_NumActiveThreads(), nalu_hypre_GetThreadNum(), &ns, &ne);

      marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, ncols_A, NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_CSRMatrixAddFirstPass(ns, ne, twspace, marker, NULL, NULL,
                                  A, B, nrows_A, nnzrows_C, ncols_A, rownnz_C,
                                  memory_location_C, C_i, &C);

      nalu_hypre_CSRMatrixAddSecondPass(ns, ne, twspace, marker, NULL, NULL,
                                   rownnz_C, alpha, beta, A, B, C);

      nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);
   } /* end of parallel region */

   /* Free memory */
   nalu_hypre_TFree(twspace, NALU_HYPRE_MEMORY_HOST);

   return C;
}

nalu_hypre_CSRMatrix*
nalu_hypre_CSRMatrixAdd( NALU_HYPRE_Complex    alpha,
                    nalu_hypre_CSRMatrix *A,
                    NALU_HYPRE_Complex    beta,
                    nalu_hypre_CSRMatrix *B)
{
   nalu_hypre_CSRMatrix *C = NULL;

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_CSRMatrixMemoryLocation(A),
                                                      nalu_hypre_CSRMatrixMemoryLocation(B) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      C = nalu_hypre_CSRMatrixAddDevice(alpha, A, beta, B);
   }
   else
#endif
   {
      C = nalu_hypre_CSRMatrixAddHost(alpha, A, beta, B);
   }

   return C;
}

#if 0
/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixBigAdd:
 *
 * RL: comment it out which was used in ams.c. Should be combined with
 *     above nalu_hypre_CSRMatrixAddHost whenever it is needed again
 *
 * Adds two CSR Matrices A and B with column indices stored as NALU_HYPRE_BigInt
 * and returns a CSR Matrix C;
 *
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 *       in A and B. To remove those, use nalu_hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix *
nalu_hypre_CSRMatrixBigAdd( nalu_hypre_CSRMatrix *A,
                       nalu_hypre_CSRMatrix *B )
{
   NALU_HYPRE_Complex    *A_data   = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int        *A_i      = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_BigInt     *A_j      = nalu_hypre_CSRMatrixBigJ(A);
   NALU_HYPRE_Int         nrows_A  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int         ncols_A  = nalu_hypre_CSRMatrixNumCols(A);

   NALU_HYPRE_Complex    *B_data   = nalu_hypre_CSRMatrixData(B);
   NALU_HYPRE_Int        *B_i      = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_BigInt     *B_j      = nalu_hypre_CSRMatrixBigJ(B);
   NALU_HYPRE_Int         nrows_B  = nalu_hypre_CSRMatrixNumRows(B);
   NALU_HYPRE_Int         ncols_B  = nalu_hypre_CSRMatrixNumCols(B);

   nalu_hypre_CSRMatrix  *C;
   NALU_HYPRE_Complex    *C_data;
   NALU_HYPRE_Int        *C_i;
   NALU_HYPRE_BigInt     *C_j;
   NALU_HYPRE_Int        *twspace;

   NALU_HYPRE_MemoryLocation memory_location_A = nalu_hypre_CSRMatrixMemoryLocation(A);
   NALU_HYPRE_MemoryLocation memory_location_B = nalu_hypre_CSRMatrixMemoryLocation(B);

   /* RL: TODO cannot guarantee, maybe should never assert
   nalu_hypre_assert(memory_location_A == memory_location_B);
   */

   /* RL: in the case of A=H, B=D, or A=D, B=H, let C = D,
    * not sure if this is the right thing to do.
    * Also, need something like this in other places
    * TODO */
   NALU_HYPRE_MemoryLocation memory_location_C = nalu_hypre_max(memory_location_A, memory_location_B);

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   /* Allocate memory */
   twspace = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_NumThreads(), NALU_HYPRE_MEMORY_HOST);
   C_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows_A + 1, memory_location_C);

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      NALU_HYPRE_Int     ia, ib, ic, num_nonzeros;
      NALU_HYPRE_Int     ns, ne, pos;
      NALU_HYPRE_BigInt  jcol;
      NALU_HYPRE_Int     ii, num_threads;
      NALU_HYPRE_Int     jj;
      NALU_HYPRE_Int    *marker = NULL;

      ii = nalu_hypre_GetThreadNum();
      num_threads = nalu_hypre_NumActiveThreads();
      nalu_hypre_partition1D(nrows_A, num_threads, ii, &ns, &ne);

      marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, ncols_A, NALU_HYPRE_MEMORY_HOST);
      for (ia = 0; ia < ncols_A; ia++)
      {
         marker[ia] = -1;
      }

      /* First pass */
      num_nonzeros = 0;
      for (ic = ns; ic < ne; ic++)
      {
         C_i[ic] = num_nonzeros;
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
      twspace[ii] = num_nonzeros;

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      /* Correct row pointer */
      if (ii)
      {
         jj = twspace[0];
         for (ic = 1; ic < ii; ic++)
         {
            jj += twspace[ia];
         }

         for (ic = ns; ic < ne; ic++)
         {
            C_i[ic] += jj;
         }
      }
      else
      {
         C_i[nrows_A] = 0;
         for (ic = 0; ic < num_threads; ic++)
         {
            C_i[nrows_A] += twspace[ic];
         }

         C = nalu_hypre_CSRMatrixCreate(nrows_A, ncols_A, C_i[nrows_A]);
         nalu_hypre_CSRMatrixI(C) = C_i;
         nalu_hypre_CSRMatrixInitialize_v2(C, 1, memory_location_C);
         C_j = nalu_hypre_CSRMatrixBigJ(C);
         C_data = nalu_hypre_CSRMatrixData(C);
      }

      /* Second pass */
      for (ia = 0; ia < ncols_A; ia++)
      {
         marker[ia] = -1;
      }

      pos = C_i[ns];
      for (ic = ns; ic < ne; ic++)
      {
         for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
         {
            jcol = A_j[ia];
            C_j[pos] = jcol;
            C_data[pos] = A_data[ia];
            marker[jcol] = pos;
            pos++;
         }

         for (ib = B_i[ic]; ib < B_i[ic + 1]; ib++)
         {
            jcol = B_j[ib];
            if (marker[jcol] < C_i[ic])
            {
               C_j[pos] = jcol;
               C_data[pos] = B_data[ib];
               marker[jcol] = pos;
               pos++;
            }
            else
            {
               C_data[marker[jcol]] += B_data[ib];
            }
         }
      }
      nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);
   } /* end of parallel region */

   /* Free memory */
   nalu_hypre_TFree(twspace, NALU_HYPRE_MEMORY_HOST);

   return C;
}

#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixMultiplyHost
 *
 * Multiplies two CSR Matrices A and B and returns a CSR Matrix C;
 *
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 *       in A and B. To remove those, use nalu_hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix*
nalu_hypre_CSRMatrixMultiplyHost( nalu_hypre_CSRMatrix *A,
                             nalu_hypre_CSRMatrix *B )
{
   NALU_HYPRE_Complex        *A_data    = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int            *A_i       = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int            *A_j       = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int            *rownnz_A  = nalu_hypre_CSRMatrixRownnz(A);
   NALU_HYPRE_Int             nrows_A   = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int             ncols_A   = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int             nnzrows_A = nalu_hypre_CSRMatrixNumRownnz(A);
   NALU_HYPRE_Int             num_nnz_A = nalu_hypre_CSRMatrixNumNonzeros(A);

   NALU_HYPRE_Complex        *B_data    = nalu_hypre_CSRMatrixData(B);
   NALU_HYPRE_Int            *B_i       = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int            *B_j       = nalu_hypre_CSRMatrixJ(B);
   NALU_HYPRE_Int             nrows_B   = nalu_hypre_CSRMatrixNumRows(B);
   NALU_HYPRE_Int             ncols_B   = nalu_hypre_CSRMatrixNumCols(B);
   NALU_HYPRE_Int             num_nnz_B = nalu_hypre_CSRMatrixNumNonzeros(B);

   NALU_HYPRE_MemoryLocation  memory_location_A = nalu_hypre_CSRMatrixMemoryLocation(A);
   NALU_HYPRE_MemoryLocation  memory_location_B = nalu_hypre_CSRMatrixMemoryLocation(B);

   nalu_hypre_CSRMatrix      *C;
   NALU_HYPRE_Complex        *C_data;
   NALU_HYPRE_Int            *C_i;
   NALU_HYPRE_Int            *C_j;

   NALU_HYPRE_Int             ia, ib, ic, ja, jb, num_nonzeros;
   NALU_HYPRE_Int             counter;
   NALU_HYPRE_Complex         a_entry, b_entry;
   NALU_HYPRE_Int             allsquare = 0;
   NALU_HYPRE_Int            *twspace;

   /* RL: TODO cannot guarantee, maybe should never assert
   nalu_hypre_assert(memory_location_A == memory_location_B);
   */

   /* RL: in the case of A=H, B=D, or A=D, B=H, let C = D,
    * not sure if this is the right thing to do.
    * Also, need something like this in other places
    * TODO */
   NALU_HYPRE_MemoryLocation memory_location_C = nalu_hypre_max(memory_location_A, memory_location_B);

   if (ncols_A != nrows_B)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   if (nrows_A == ncols_B)
   {
      allsquare = 1;
   }

   if ((num_nnz_A == 0) || (num_nnz_B == 0))
   {
      C = nalu_hypre_CSRMatrixCreate(nrows_A, ncols_B, 0);
      nalu_hypre_CSRMatrixNumRownnz(C) = 0;
      nalu_hypre_CSRMatrixInitialize_v2(C, 0, memory_location_C);

      return C;
   }

   /* Allocate memory */
   twspace = nalu_hypre_TAlloc(NALU_HYPRE_Int, nalu_hypre_NumThreads(), NALU_HYPRE_MEMORY_HOST);
   C_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows_A + 1, memory_location_C);

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel private(ia, ib, ic, ja, jb, num_nonzeros, counter, a_entry, b_entry)
#endif
   {
      NALU_HYPRE_Int  *B_marker = NULL;
      NALU_HYPRE_Int   ns, ne, ii, jj;
      NALU_HYPRE_Int   num_threads;
      NALU_HYPRE_Int   i1, iic;

      ii = nalu_hypre_GetThreadNum();
      num_threads = nalu_hypre_NumActiveThreads();
      nalu_hypre_partition1D(nnzrows_A, num_threads, ii, &ns, &ne);

      B_marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, ncols_B, NALU_HYPRE_MEMORY_HOST);
      for (ib = 0; ib < ncols_B; ib++)
      {
         B_marker[ib] = -1;
      }

      NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "First pass");

      /* First pass: compute sizes of C rows. */
      num_nonzeros = 0;
      for (ic = ns; ic < ne; ic++)
      {
         if (rownnz_A)
         {
            iic = rownnz_A[ic];
            C_i[iic] = num_nonzeros;
         }
         else
         {
            iic = ic;
            C_i[iic] = num_nonzeros;
            if (allsquare)
            {
               B_marker[iic] = iic;
               num_nonzeros++;
            }
         }

         for (ia = A_i[iic]; ia < A_i[iic + 1]; ia++)
         {
            ja = A_j[ia];
            for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++)
            {
               jb = B_j[ib];
               if (B_marker[jb] != iic)
               {
                  B_marker[jb] = iic;
                  num_nonzeros++;
               }
            }
         }
      }
      twspace[ii] = num_nonzeros;

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      /* Correct C_i - phase 1 */
      if (ii)
      {
         jj = twspace[0];
         for (i1 = 1; i1 < ii; i1++)
         {
            jj += twspace[i1];
         }

         for (i1 = ns; i1 < ne; i1++)
         {
            iic = rownnz_A ? rownnz_A[i1] : i1;
            C_i[iic] += jj;
         }
      }
      else
      {
         C_i[nrows_A] = 0;
         for (i1 = 0; i1 < num_threads; i1++)
         {
            C_i[nrows_A] += twspace[i1];
         }

         C = nalu_hypre_CSRMatrixCreate(nrows_A, ncols_B, C_i[nrows_A]);
         nalu_hypre_CSRMatrixI(C) = C_i;
         nalu_hypre_CSRMatrixInitialize_v2(C, 0, memory_location_C);
         C_j = nalu_hypre_CSRMatrixJ(C);
         C_data = nalu_hypre_CSRMatrixData(C);
      }

      /* Correct C_i - phase 2 */
      if (rownnz_A != NULL)
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp barrier
#endif
         for (ic = ns; ic < (ne - 1); ic++)
         {
            for (iic = rownnz_A[ic] + 1; iic < rownnz_A[ic + 1]; iic++)
            {
               C_i[iic] = C_i[rownnz_A[ic + 1]];
            }
         }

         if (ii < (num_threads - 1))
         {
            for (iic = rownnz_A[ne - 1] + 1; iic < rownnz_A[ne]; iic++)
            {
               C_i[iic] = C_i[rownnz_A[ne]];
            }
         }
         else
         {
            for (iic = rownnz_A[ne - 1] + 1; iic < nrows_A; iic++)
            {
               C_i[iic] = C_i[nrows_A];
            }
         }
      }
      /* End of First Pass */
      NALU_HYPRE_ANNOTATE_REGION_END("%s", "First pass");

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      /* Second pass: Fill in C_data and C_j. */
      NALU_HYPRE_ANNOTATE_REGION_BEGIN("%s", "Second pass");
      for (ib = 0; ib < ncols_B; ib++)
      {
         B_marker[ib] = -1;
      }

      counter = rownnz_A ? C_i[rownnz_A[ns]] : C_i[ns];
      for (ic = ns; ic < ne; ic++)
      {
         if (rownnz_A)
         {
            iic = rownnz_A[ic];
         }
         else
         {
            iic = ic;
            if (allsquare)
            {
               B_marker[ic] = counter;
               C_data[counter] = 0;
               C_j[counter] = ic;
               counter++;
            }
         }

         for (ia = A_i[iic]; ia < A_i[iic + 1]; ia++)
         {
            ja = A_j[ia];
            a_entry = A_data[ia];
            for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++)
            {
               jb = B_j[ib];
               b_entry = B_data[ib];
               if (B_marker[jb] < C_i[iic])
               {
                  B_marker[jb] = counter;
                  C_j[B_marker[jb]] = jb;
                  C_data[B_marker[jb]] = a_entry * b_entry;
                  counter++;
               }
               else
               {
                  C_data[B_marker[jb]] += a_entry * b_entry;
               }
            }
         }
      }
      NALU_HYPRE_ANNOTATE_REGION_END("%s", "Second pass");

      /* End of Second Pass */
      nalu_hypre_TFree(B_marker, NALU_HYPRE_MEMORY_HOST);
   } /*end parallel region */

#ifdef NALU_HYPRE_DEBUG
   for (ic = 0; ic < nrows_A; ic++)
   {
      nalu_hypre_assert(C_i[ic] <= C_i[ic + 1]);
   }
#endif

   // Set rownnz and num_rownnz
   nalu_hypre_CSRMatrixSetRownnz(C);

   /* Free memory */
   nalu_hypre_TFree(twspace, NALU_HYPRE_MEMORY_HOST);

   return C;
}

nalu_hypre_CSRMatrix*
nalu_hypre_CSRMatrixMultiply( nalu_hypre_CSRMatrix *A,
                         nalu_hypre_CSRMatrix *B)
{
   nalu_hypre_CSRMatrix *C = NULL;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy2( nalu_hypre_CSRMatrixMemoryLocation(A),
                                                      nalu_hypre_CSRMatrixMemoryLocation(B) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      C = nalu_hypre_CSRMatrixMultiplyDevice(A, B);
   }
   else
#endif
   {
      C = nalu_hypre_CSRMatrixMultiplyHost(A, B);
   }

   return C;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix *
nalu_hypre_CSRMatrixDeleteZeros( nalu_hypre_CSRMatrix *A,
                            NALU_HYPRE_Real       tol )
{
   NALU_HYPRE_Complex    *A_data   = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int        *A_i      = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int        *A_j      = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int         nrows_A  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int         ncols_A  = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int         num_nonzeros  = nalu_hypre_CSRMatrixNumNonzeros(A);

   nalu_hypre_CSRMatrix  *B;
   NALU_HYPRE_Complex    *B_data;
   NALU_HYPRE_Int        *B_i;
   NALU_HYPRE_Int        *B_j;

   NALU_HYPRE_Int         zeros;
   NALU_HYPRE_Int         i, j;
   NALU_HYPRE_Int         pos_A, pos_B;

   zeros = 0;
   for (i = 0; i < num_nonzeros; i++)
   {
      if (nalu_hypre_cabs(A_data[i]) <= tol)
      {
         zeros++;
      }
   }

   if (zeros)
   {
      B = nalu_hypre_CSRMatrixCreate(nrows_A, ncols_A, num_nonzeros - zeros);
      nalu_hypre_CSRMatrixInitialize(B);
      B_i = nalu_hypre_CSRMatrixI(B);
      B_j = nalu_hypre_CSRMatrixJ(B);
      B_data = nalu_hypre_CSRMatrixData(B);
      B_i[0] = 0;
      pos_A = pos_B = 0;
      for (i = 0; i < nrows_A; i++)
      {
         for (j = A_i[i]; j < A_i[i + 1]; j++)
         {
            if (nalu_hypre_cabs(A_data[j]) <= tol)
            {
               pos_A++;
            }
            else
            {
               B_data[pos_B] = A_data[pos_A];
               B_j[pos_B] = A_j[pos_A];
               pos_B++;
               pos_A++;
            }
         }
         B_i[i + 1] = pos_B;
      }

      return B;
   }
   else
   {
      return NULL;
   }
}

/******************************************************************************
 *
 * Finds transpose of a nalu_hypre_CSRMatrix
 *
 *****************************************************************************/

/**
 * idx = idx2*dim1 + idx1
 * -> ret = idx1*dim2 + idx2
 *        = (idx%dim1)*dim2 + idx/dim1
 */
static inline NALU_HYPRE_Int
transpose_idx (NALU_HYPRE_Int idx, NALU_HYPRE_Int dim1, NALU_HYPRE_Int dim2)
{
   return idx % dim1 * dim2 + idx / dim1;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixTransposeHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixTransposeHost(nalu_hypre_CSRMatrix  *A,
                             nalu_hypre_CSRMatrix **AT,
                             NALU_HYPRE_Int         data)

{
   NALU_HYPRE_Complex        *A_data     = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int            *A_i        = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int            *A_j        = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int            *rownnz_A   = nalu_hypre_CSRMatrixRownnz(A);
   NALU_HYPRE_Int             nnzrows_A  = nalu_hypre_CSRMatrixNumRownnz(A);
   NALU_HYPRE_Int             num_rows_A = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int             num_cols_A = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int             num_nnzs_A = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_MemoryLocation  memory_location = nalu_hypre_CSRMatrixMemoryLocation(A);

   NALU_HYPRE_Complex        *AT_data;
   NALU_HYPRE_Int            *AT_j;
   NALU_HYPRE_Int             num_rows_AT;
   NALU_HYPRE_Int             num_cols_AT;
   NALU_HYPRE_Int             num_nnzs_AT;

   NALU_HYPRE_Int             max_col;
   NALU_HYPRE_Int             i, j;

   /*--------------------------------------------------------------
    * First, ascertain that num_cols and num_nonzeros has been set.
    * If not, set them.
    *--------------------------------------------------------------*/
   NALU_HYPRE_ANNOTATE_FUNC_BEGIN;

   if (!num_nnzs_A && A_i)
   {
      num_nnzs_A = A_i[num_rows_A];
   }

   if (num_rows_A && num_nnzs_A && ! num_cols_A)
   {
      max_col = -1;
      for (i = 0; i < num_rows_A; ++i)
      {
         for (j = A_i[i]; j < A_i[i + 1]; j++)
         {
            if (A_j[j] > max_col)
            {
               max_col = A_j[j];
            }
         }
      }
      num_cols_A = max_col + 1;
   }

   num_rows_AT = num_cols_A;
   num_cols_AT = num_rows_A;
   num_nnzs_AT = num_nnzs_A;

   *AT = nalu_hypre_CSRMatrixCreate(num_rows_AT, num_cols_AT, num_nnzs_AT);
   nalu_hypre_CSRMatrixMemoryLocation(*AT) = memory_location;

   if (num_cols_A == 0)
   {
      // JSP: parallel counting sorting breaks down
      // when A has no columns
      nalu_hypre_CSRMatrixInitialize(*AT);
      NALU_HYPRE_ANNOTATE_FUNC_END;

      return nalu_hypre_error_flag;
   }

   AT_j = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_nnzs_AT, memory_location);
   nalu_hypre_CSRMatrixJ(*AT) = AT_j;
   if (data)
   {
      AT_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, num_nnzs_AT, memory_location);
      nalu_hypre_CSRMatrixData(*AT) = AT_data;
   }

   /*-----------------------------------------------------------------
    * Parallel count sort
    *-----------------------------------------------------------------*/
   NALU_HYPRE_Int *bucket = nalu_hypre_CTAlloc(NALU_HYPRE_Int, (num_cols_A + 1) * nalu_hypre_NumThreads(),
                                     NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      NALU_HYPRE_Int   ii, num_threads, ns, ne;
      NALU_HYPRE_Int   i, j, j0, j1, ir;
      NALU_HYPRE_Int   idx, offset;
      NALU_HYPRE_Int   transpose_i;
      NALU_HYPRE_Int   transpose_i_minus_1;
      NALU_HYPRE_Int   transpose_i0;
      NALU_HYPRE_Int   transpose_j0;
      NALU_HYPRE_Int   transpose_j1;

      ii = nalu_hypre_GetThreadNum();
      num_threads = nalu_hypre_NumActiveThreads();
      nalu_hypre_partition1D(nnzrows_A, num_threads, ii, &ns, &ne);

      /*-----------------------------------------------------------------
       * Count the number of entries that will go into each bucket
       * bucket is used as NALU_HYPRE_Int[num_threads][num_colsA] 2D array
       *-----------------------------------------------------------------*/
      if (rownnz_A == NULL)
      {
         for (j = A_i[ns]; j < A_i[ne]; ++j)
         {
            bucket[ii * num_cols_A + A_j[j]]++;
         }
      }
      else
      {
         for (i = ns; i < ne; i++)
         {
            ir = rownnz_A[i];
            for (j = A_i[ir]; j < A_i[ir + 1]; ++j)
            {
               bucket[ii * num_cols_A + A_j[j]]++;
            }
         }
      }

      /*-----------------------------------------------------------------
       * Parallel prefix sum of bucket with length num_colsA * num_threads
       * accessed as if it is transposed as NALU_HYPRE_Int[num_colsA][num_threads]
       *-----------------------------------------------------------------*/
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      for (i = ii * num_cols_A + 1; i < (ii + 1)*num_cols_A; ++i)
      {
         transpose_i = transpose_idx(i, num_threads, num_cols_A);
         transpose_i_minus_1 = transpose_idx(i - 1, num_threads, num_cols_A);

         bucket[transpose_i] += bucket[transpose_i_minus_1];
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
      #pragma omp master
#endif
      {
         for (i = 1; i < num_threads; ++i)
         {
            j0 = num_cols_A * i - 1;
            j1 = num_cols_A * (i + 1) - 1;
            transpose_j0 = transpose_idx(j0, num_threads, num_cols_A);
            transpose_j1 = transpose_idx(j1, num_threads, num_cols_A);

            bucket[transpose_j1] += bucket[transpose_j0];
         }
      }
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      if (ii > 0)
      {
         transpose_i0 = transpose_idx(num_cols_A * ii - 1, num_threads, num_cols_A);
         offset = bucket[transpose_i0];

         for (i = ii * num_cols_A; i < (ii + 1)*num_cols_A - 1; ++i)
         {
            transpose_i = transpose_idx(i, num_threads, num_cols_A);

            bucket[transpose_i] += offset;
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      /*----------------------------------------------------------------
       * Load the data and column numbers of AT
       *----------------------------------------------------------------*/

      if (data)
      {
         for (i = ne - 1; i >= ns; --i)
         {
            ir = rownnz_A ? rownnz_A[i] : i;
            for (j = A_i[ir + 1] - 1; j >= A_i[ir]; --j)
            {
               idx = A_j[j];
               --bucket[ii * num_cols_A + idx];

               offset = bucket[ii * num_cols_A + idx];
               AT_data[offset] = A_data[j];
               AT_j[offset] = ir;
            }
         }
      }
      else
      {
         for (i = ne - 1; i >= ns; --i)
         {
            ir = rownnz_A ? rownnz_A[i] : i;
            for (j = A_i[ir + 1] - 1; j >= A_i[ir]; --j)
            {
               idx = A_j[j];
               --bucket[ii * num_cols_A + idx];

               offset = bucket[ii * num_cols_A + idx];
               AT_j[offset] = ir;
            }
         }
      }
   } /* end parallel region */

   nalu_hypre_CSRMatrixI(*AT) = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_cols_A + 1, memory_location);
   nalu_hypre_TMemcpy(nalu_hypre_CSRMatrixI(*AT), bucket, NALU_HYPRE_Int, num_cols_A + 1, memory_location,
                 NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_CSRMatrixI(*AT)[num_cols_A] = num_nnzs_A;
   nalu_hypre_TFree(bucket, NALU_HYPRE_MEMORY_HOST);

   // Set rownnz and num_rownnz
   if (nalu_hypre_CSRMatrixNumRownnz(A) < num_rows_A)
   {
      nalu_hypre_CSRMatrixSetRownnz(*AT);
   }

   NALU_HYPRE_ANNOTATE_FUNC_END;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixTranspose
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixTranspose(nalu_hypre_CSRMatrix  *A,
                         nalu_hypre_CSRMatrix **AT,
                         NALU_HYPRE_Int         data)
{
   NALU_HYPRE_Int ierr = 0;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_CSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      ierr = nalu_hypre_CSRMatrixTransposeDevice(A, AT, data);
   }
   else
#endif
   {
      ierr = nalu_hypre_CSRMatrixTransposeHost(A, AT, data);
   }

   nalu_hypre_CSRMatrixSetPatternOnly(*AT, nalu_hypre_CSRMatrixPatternOnly(A));

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixSplit
 *--------------------------------------------------------------------------*/

/* RL: TODO add memory locations */
NALU_HYPRE_Int nalu_hypre_CSRMatrixSplit(nalu_hypre_CSRMatrix  *Bs_ext,
                               NALU_HYPRE_BigInt      first_col_diag_B,
                               NALU_HYPRE_BigInt      last_col_diag_B,
                               NALU_HYPRE_Int         num_cols_offd_B,
                               NALU_HYPRE_BigInt     *col_map_offd_B,
                               NALU_HYPRE_Int        *num_cols_offd_C_ptr,
                               NALU_HYPRE_BigInt    **col_map_offd_C_ptr,
                               nalu_hypre_CSRMatrix **Bext_diag_ptr,
                               nalu_hypre_CSRMatrix **Bext_offd_ptr)
{
   NALU_HYPRE_Complex   *Bs_ext_data = nalu_hypre_CSRMatrixData(Bs_ext);
   NALU_HYPRE_Int       *Bs_ext_i    = nalu_hypre_CSRMatrixI(Bs_ext);
   NALU_HYPRE_BigInt    *Bs_ext_j    = nalu_hypre_CSRMatrixBigJ(Bs_ext);
   NALU_HYPRE_Int        num_rows_Bext = nalu_hypre_CSRMatrixNumRows(Bs_ext);
   NALU_HYPRE_Int        B_ext_diag_size = 0;
   NALU_HYPRE_Int        B_ext_offd_size = 0;
   NALU_HYPRE_Int       *B_ext_diag_i = NULL;
   NALU_HYPRE_Int       *B_ext_diag_j = NULL;
   NALU_HYPRE_Complex   *B_ext_diag_data = NULL;
   NALU_HYPRE_Int       *B_ext_offd_i = NULL;
   NALU_HYPRE_Int       *B_ext_offd_j = NULL;
   NALU_HYPRE_Complex   *B_ext_offd_data = NULL;
   NALU_HYPRE_Int       *my_diag_array;
   NALU_HYPRE_Int       *my_offd_array;
   NALU_HYPRE_BigInt    *temp;
   NALU_HYPRE_Int        max_num_threads;
   NALU_HYPRE_Int        cnt = 0;
   nalu_hypre_CSRMatrix *Bext_diag = NULL;
   nalu_hypre_CSRMatrix *Bext_offd = NULL;
   NALU_HYPRE_BigInt    *col_map_offd_C = NULL;
   NALU_HYPRE_Int        num_cols_offd_C = 0;

   B_ext_diag_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows_Bext + 1, NALU_HYPRE_MEMORY_HOST);
   B_ext_offd_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows_Bext + 1, NALU_HYPRE_MEMORY_HOST);

   max_num_threads = nalu_hypre_NumThreads();
   my_diag_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_threads, NALU_HYPRE_MEMORY_HOST);
   my_offd_array = nalu_hypre_CTAlloc(NALU_HYPRE_Int, max_num_threads, NALU_HYPRE_MEMORY_HOST);

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      NALU_HYPRE_Int ns, ne, ii, num_threads;
      NALU_HYPRE_Int i1, i, j;
      NALU_HYPRE_Int my_offd_size, my_diag_size;
      NALU_HYPRE_Int cnt_offd, cnt_diag;

      ii = nalu_hypre_GetThreadNum();
      num_threads = nalu_hypre_NumActiveThreads();
      nalu_hypre_partition1D(num_rows_Bext, num_threads, ii, &ns, &ne);

      my_diag_size = 0;
      my_offd_size = 0;
      for (i = ns; i < ne; i++)
      {
         B_ext_diag_i[i] = my_diag_size;
         B_ext_offd_i[i] = my_offd_size;
         for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         {
            if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
            {
               my_offd_size++;
            }
            else
            {
               my_diag_size++;
            }
         }
      }
      my_diag_array[ii] = my_diag_size;
      my_offd_array[ii] = my_offd_size;

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      if (ii)
      {
         my_diag_size = my_diag_array[0];
         my_offd_size = my_offd_array[0];
         for (i1 = 1; i1 < ii; i1++)
         {
            my_diag_size += my_diag_array[i1];
            my_offd_size += my_offd_array[i1];
         }

         for (i1 = ns; i1 < ne; i1++)
         {
            B_ext_diag_i[i1] += my_diag_size;
            B_ext_offd_i[i1] += my_offd_size;
         }
      }
      else
      {
         B_ext_diag_size = 0;
         B_ext_offd_size = 0;
         for (i1 = 0; i1 < num_threads; i1++)
         {
            B_ext_diag_size += my_diag_array[i1];
            B_ext_offd_size += my_offd_array[i1];
         }
         B_ext_diag_i[num_rows_Bext] = B_ext_diag_size;
         B_ext_offd_i[num_rows_Bext] = B_ext_offd_size;

         if (B_ext_diag_size)
         {
            B_ext_diag_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,     B_ext_diag_size, NALU_HYPRE_MEMORY_HOST);
            B_ext_diag_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, B_ext_diag_size, NALU_HYPRE_MEMORY_HOST);
         }
         if (B_ext_offd_size)
         {
            B_ext_offd_j    = nalu_hypre_CTAlloc(NALU_HYPRE_Int,     B_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
            B_ext_offd_data = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, B_ext_offd_size, NALU_HYPRE_MEMORY_HOST);
         }
         if (B_ext_offd_size || num_cols_offd_B)
         {
            temp = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, B_ext_offd_size + num_cols_offd_B, NALU_HYPRE_MEMORY_HOST);
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      cnt_offd = B_ext_offd_i[ns];
      cnt_diag = B_ext_diag_i[ns];
      for (i = ns; i < ne; i++)
      {
         for (j = Bs_ext_i[i]; j < Bs_ext_i[i + 1]; j++)
         {
            if (Bs_ext_j[j] < first_col_diag_B || Bs_ext_j[j] > last_col_diag_B)
            {
               temp[cnt_offd] = Bs_ext_j[j];
               B_ext_offd_j[cnt_offd] = Bs_ext_j[j];
               B_ext_offd_data[cnt_offd++] = Bs_ext_data[j];
            }
            else
            {
               B_ext_diag_j[cnt_diag] = Bs_ext_j[j] - first_col_diag_B;
               B_ext_diag_data[cnt_diag++] = Bs_ext_data[j];
            }
         }
      }

      /* This computes the mappings */
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      if (ii == 0)
      {
         cnt = 0;
         if (B_ext_offd_size || num_cols_offd_B)
         {
            cnt = B_ext_offd_size;
            for (i = 0; i < num_cols_offd_B; i++)
            {
               temp[cnt++] = col_map_offd_B[i];
            }
            if (cnt)
            {
               nalu_hypre_BigQsort0(temp, 0, cnt - 1);
               num_cols_offd_C = 1;
               NALU_HYPRE_BigInt value = temp[0];
               for (i = 1; i < cnt; i++)
               {
                  if (temp[i] > value)
                  {
                     value = temp[i];
                     temp[num_cols_offd_C++] = value;
                  }
               }
            }

            if (num_cols_offd_C)
            {
               col_map_offd_C = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_cols_offd_C, NALU_HYPRE_MEMORY_HOST);
            }

            for (i = 0; i < num_cols_offd_C; i++)
            {
               col_map_offd_C[i] = temp[i];
            }

            nalu_hypre_TFree(temp, NALU_HYPRE_MEMORY_HOST);
         }
      }

#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp barrier
#endif

      for (i = ns; i < ne; i++)
      {
         for (j = B_ext_offd_i[i]; j < B_ext_offd_i[i + 1]; j++)
         {
            B_ext_offd_j[j] = nalu_hypre_BigBinarySearch(col_map_offd_C, B_ext_offd_j[j], num_cols_offd_C);
         }
      }
   } /* end parallel region */

   nalu_hypre_TFree(my_diag_array, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(my_offd_array, NALU_HYPRE_MEMORY_HOST);

   Bext_diag = nalu_hypre_CSRMatrixCreate(num_rows_Bext, last_col_diag_B - first_col_diag_B + 1,
                                     B_ext_diag_size);
   nalu_hypre_CSRMatrixMemoryLocation(Bext_diag) = NALU_HYPRE_MEMORY_HOST;
   Bext_offd = nalu_hypre_CSRMatrixCreate(num_rows_Bext, num_cols_offd_C, B_ext_offd_size);
   nalu_hypre_CSRMatrixMemoryLocation(Bext_offd) = NALU_HYPRE_MEMORY_HOST;
   nalu_hypre_CSRMatrixI(Bext_diag)    = B_ext_diag_i;
   nalu_hypre_CSRMatrixJ(Bext_diag)    = B_ext_diag_j;
   nalu_hypre_CSRMatrixData(Bext_diag) = B_ext_diag_data;
   nalu_hypre_CSRMatrixI(Bext_offd)    = B_ext_offd_i;
   nalu_hypre_CSRMatrixJ(Bext_offd)    = B_ext_offd_j;
   nalu_hypre_CSRMatrixData(Bext_offd) = B_ext_offd_data;

   *col_map_offd_C_ptr = col_map_offd_C;
   *Bext_diag_ptr = Bext_diag;
   *Bext_offd_ptr = Bext_offd;
   *num_cols_offd_C_ptr = num_cols_offd_C;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixReorderHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixReorderHost(nalu_hypre_CSRMatrix *A)
{
   NALU_HYPRE_Complex *A_data     = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int     *A_i        = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *A_j        = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int     *rownnz_A   = nalu_hypre_CSRMatrixRownnz(A);
   NALU_HYPRE_Int      nnzrows_A  = nalu_hypre_CSRMatrixNumRownnz(A);
   NALU_HYPRE_Int      num_rows_A = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int      num_cols_A = nalu_hypre_CSRMatrixNumCols(A);

   NALU_HYPRE_Int      i, ii, j;

   /* the matrix should be square */
   if (num_rows_A != num_cols_A)
   {
      return -1;
   }

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i, ii, j) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < nnzrows_A; i++)
   {
      ii = rownnz_A ? rownnz_A[i] : i;
      for (j = A_i[ii]; j < A_i[ii + 1]; j++)
      {
         if (A_j[j] == ii)
         {
            if (j != A_i[ii])
            {
               nalu_hypre_swap(A_j, A_i[ii], j);
               nalu_hypre_swap_c(A_data, A_i[ii], j);
            }
            break;
         }
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixReorder:
 *
 * Reorders the column and data arrays of a square CSR matrix, such that the
 * first entry in each row is the diagonal one.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixReorder(nalu_hypre_CSRMatrix *A)
{
   NALU_HYPRE_Int ierr = 0;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_CSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      ierr = nalu_hypre_CSRMatrixMoveDiagFirstDevice(A);
   }
   else
#endif
   {
      ierr = nalu_hypre_CSRMatrixReorderHost(A);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixAddPartial:
 * adds matrix rows in the CSR matrix B to the CSR Matrix A, where row_nums[i]
 * defines to which row of A the i-th row of B is added, and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 *       in A and B. To remove those, use nalu_hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/
nalu_hypre_CSRMatrix *
nalu_hypre_CSRMatrixAddPartial( nalu_hypre_CSRMatrix *A,
                           nalu_hypre_CSRMatrix *B,
                           NALU_HYPRE_Int *row_nums)
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

   NALU_HYPRE_Int         ia, ib, ic, jcol, num_nonzeros;
   NALU_HYPRE_Int         pos, i, i2, j, cnt;
   NALU_HYPRE_Int         *marker;
   NALU_HYPRE_Int         *map;
   NALU_HYPRE_Int         *temp;

   NALU_HYPRE_MemoryLocation memory_location_A = nalu_hypre_CSRMatrixMemoryLocation(A);
   NALU_HYPRE_MemoryLocation memory_location_B = nalu_hypre_CSRMatrixMemoryLocation(B);

   /* RL: TODO cannot guarantee, maybe should never assert
   nalu_hypre_assert(memory_location_A == memory_location_B);
   */

   /* RL: in the case of A=H, B=D, or A=D, B=H, let C = D,
    * not sure if this is the right thing to do.
    * Also, need something like this in other places
    * TODO */
   NALU_HYPRE_MemoryLocation memory_location_C = nalu_hypre_max(memory_location_A, memory_location_B);

   if (ncols_A != ncols_B)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Warning! incompatible matrix dimensions!\n");
      return NULL;
   }

   map = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows_B, NALU_HYPRE_MEMORY_HOST);
   temp = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows_B, NALU_HYPRE_MEMORY_HOST);
   for (i = 0; i < nrows_B; i++)
   {
      map[i] = i;
      temp[i] = row_nums[i];
   }

   nalu_hypre_qsort2i(temp, map, 0, nrows_B - 1);

   marker = nalu_hypre_CTAlloc(NALU_HYPRE_Int, ncols_A, NALU_HYPRE_MEMORY_HOST);
   C_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows_A + 1, memory_location_C);

   for (ia = 0; ia < ncols_A; ia++)
   {
      marker[ia] = -1;
   }

   num_nonzeros = 0;
   C_i[0] = 0;
   cnt = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
      {
         jcol = A_j[ia];
         marker[jcol] = ic;
         num_nonzeros++;
      }
      if (cnt < nrows_B && temp[cnt] == ic)
      {
         for (j = cnt; j < nrows_B; j++)
         {
            if (temp[j] == ic)
            {
               i2 = map[cnt++];
               for (ib = B_i[i2]; ib < B_i[i2 + 1]; ib++)
               {
                  jcol = B_j[ib];
                  if (marker[jcol] != ic)
                  {
                     marker[jcol] = ic;
                     num_nonzeros++;
                  }
               }
            }
            else
            {
               break;
            }
         }
      }
      C_i[ic + 1] = num_nonzeros;
   }

   C = nalu_hypre_CSRMatrixCreate(nrows_A, ncols_A, num_nonzeros);
   nalu_hypre_CSRMatrixI(C) = C_i;
   nalu_hypre_CSRMatrixInitialize_v2(C, 0, memory_location_C);
   C_j = nalu_hypre_CSRMatrixJ(C);
   C_data = nalu_hypre_CSRMatrixData(C);

   for (ia = 0; ia < ncols_A; ia++)
   {
      marker[ia] = -1;
   }

   cnt = 0;
   pos = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
      {
         jcol = A_j[ia];
         C_j[pos] = jcol;
         C_data[pos] = A_data[ia];
         marker[jcol] = pos;
         pos++;
      }
      if (cnt < nrows_B && temp[cnt] == ic)
      {
         for (j = cnt; j < nrows_B; j++)
         {
            if (temp[j] == ic)
            {
               i2 = map[cnt++];
               for (ib = B_i[i2]; ib < B_i[i2 + 1]; ib++)
               {
                  jcol = B_j[ib];
                  if (marker[jcol] < C_i[ic])
                  {
                     C_j[pos] = jcol;
                     C_data[pos] = B_data[ib];
                     marker[jcol] = pos;
                     pos++;
                  }
                  else
                  {
                     C_data[marker[jcol]] += B_data[ib];
                  }
               }
            }
            else
            {
               break;
            }
         }
      }
   }

   nalu_hypre_TFree(marker, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(map, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_TFree(temp, NALU_HYPRE_MEMORY_HOST);

   return C;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixSumElts:
 * Returns the sum of all matrix elements.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Complex
nalu_hypre_CSRMatrixSumElts( nalu_hypre_CSRMatrix *A )
{
   NALU_HYPRE_Complex  sum = 0;
   NALU_HYPRE_Complex *data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int      num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int      i;

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) reduction(+:sum) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_nonzeros; i++)
   {
      sum += data[i];
   }

   return sum;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixFnorm
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Real
nalu_hypre_CSRMatrixFnorm( nalu_hypre_CSRMatrix *A )
{
   NALU_HYPRE_Int       nrows        = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int       num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int      *A_i          = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Complex  *A_data       = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int       i;
   NALU_HYPRE_Complex   sum = 0;

   nalu_hypre_assert(num_nonzeros == A_i[nrows]);

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i) reduction(+:sum) NALU_HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < num_nonzeros; ++i)
   {
      NALU_HYPRE_Complex v = A_data[i];
      sum += v * v;
   }

   return nalu_hypre_sqrt(sum);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixComputeRowSumHost
 *
 * type == 0, sum,
 *         1, abs sum
 *         2, square sum
 *--------------------------------------------------------------------------*/

void
nalu_hypre_CSRMatrixComputeRowSumHost( nalu_hypre_CSRMatrix *A,
                                  NALU_HYPRE_Int       *CF_i,
                                  NALU_HYPRE_Int       *CF_j,
                                  NALU_HYPRE_Complex   *row_sum,
                                  NALU_HYPRE_Int        type,
                                  NALU_HYPRE_Complex    scal,
                                  const char      *set_or_add)
{
   NALU_HYPRE_Int      nrows  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Complex *A_data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int     *A_i    = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *A_j    = nalu_hypre_CSRMatrixJ(A);

   NALU_HYPRE_Int i, j;

   for (i = 0; i < nrows; i++)
   {
      NALU_HYPRE_Complex row_sum_i = set_or_add[0] == 's' ? 0.0 : row_sum[i];

      for (j = A_i[i]; j < A_i[i + 1]; j++)
      {
         if (CF_i && CF_j && CF_i[i] != CF_j[A_j[j]])
         {
            continue;
         }

         if (type == 0)
         {
            row_sum_i += scal * A_data[j];
         }
         else if (type == 1)
         {
            row_sum_i += scal * nalu_hypre_cabs(A_data[j]);
         }
         else if (type == 2)
         {
            row_sum_i += scal * A_data[j] * A_data[j];
         }
      }

      row_sum[i] = row_sum_i;
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixComputeRowSum
 *--------------------------------------------------------------------------*/

void
nalu_hypre_CSRMatrixComputeRowSum( nalu_hypre_CSRMatrix *A,
                              NALU_HYPRE_Int       *CF_i,
                              NALU_HYPRE_Int       *CF_j,
                              NALU_HYPRE_Complex   *row_sum,
                              NALU_HYPRE_Int        type,
                              NALU_HYPRE_Complex    scal,
                              const char      *set_or_add)
{
   nalu_hypre_assert( (CF_i && CF_j) || (!CF_i && !CF_j) );

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_CSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_CSRMatrixComputeRowSumDevice(A, CF_i, CF_j, row_sum, type, scal, set_or_add);
   }
   else
#endif
   {
      nalu_hypre_CSRMatrixComputeRowSumHost(A, CF_i, CF_j, row_sum, type, scal, set_or_add);
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixExtractDiagonalHost
 * type 0: diag
 *      1: abs diag
 *      2: diag inverse
 *      3: diag inverse sqrt
 *      4: abs diag inverse sqrt
 *--------------------------------------------------------------------------*/

void
nalu_hypre_CSRMatrixExtractDiagonalHost( nalu_hypre_CSRMatrix *A,
                                    NALU_HYPRE_Complex   *d,
                                    NALU_HYPRE_Int        type)
{
   NALU_HYPRE_Int      nrows  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Complex *A_data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int     *A_i    = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *A_j    = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int      i, j;
   NALU_HYPRE_Complex  d_i;

   for (i = 0; i < nrows; i++)
   {
      d_i = 0.0;
      for (j = A_i[i]; j < A_i[i + 1]; j++)
      {
         if (A_j[j] == i)
         {
            if (type == 0)
            {
               d_i = A_data[j];
            }
            else if (type == 1)
            {
               d_i = nalu_hypre_cabs(A_data[j]);
            }
            else if (type == 2)
            {
               d_i = 1.0 / (A_data[j]);
            }
            else if (type == 3)
            {
               d_i = 1.0 / (nalu_hypre_sqrt(A_data[j]));
            }
            else if (type == 4)
            {
               d_i = 1.0 / (nalu_hypre_sqrt(nalu_hypre_cabs(A_data[j])));
            }
            break;
         }
      }
      d[i] = d_i;
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixExtractDiagonal
 *
 * type 0: diag
 *      1: abs diag
 *      2: diag inverse
 *      3: diag inverse sqrt
 *--------------------------------------------------------------------------*/

void
nalu_hypre_CSRMatrixExtractDiagonal( nalu_hypre_CSRMatrix *A,
                                NALU_HYPRE_Complex   *d,
                                NALU_HYPRE_Int        type)
{
#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_CSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_CSRMatrixExtractDiagonalDevice(A, d, type);
   }
   else
#endif
   {
      nalu_hypre_CSRMatrixExtractDiagonalHost(A, d, type);
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixScale
 *
 * Scales CSR matrix: A = scalar * A.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixScale( nalu_hypre_CSRMatrix *A,
                      NALU_HYPRE_Complex    scalar)
{
   NALU_HYPRE_Complex *data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int      i;
   NALU_HYPRE_Int      k = nalu_hypre_CSRMatrixNumNonzeros(A);

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_CSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      hypreDevice_ComplexScalen(data, k, data, scalar);
   }
   else
#endif
   {
      for (i = 0; i < k; i++)
      {
         data[i] *= scalar;
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixDiagScaleHost
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixDiagScaleHost( nalu_hypre_CSRMatrix *A,
                              nalu_hypre_Vector    *ld,
                              nalu_hypre_Vector    *rd)
{

   NALU_HYPRE_Int      nrows  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Complex *A_data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int     *A_i    = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *A_j    = nalu_hypre_CSRMatrixJ(A);

   NALU_HYPRE_Complex *ldata  = ld ? nalu_hypre_VectorData(ld) : NULL;
   NALU_HYPRE_Complex *rdata  = rd ? nalu_hypre_VectorData(rd) : NULL;
   NALU_HYPRE_Int      lsize  = ld ? nalu_hypre_VectorSize(ld) : 0;
   NALU_HYPRE_Int      rsize  = rd ? nalu_hypre_VectorSize(rd) : 0;

   NALU_HYPRE_Int      i, j;
   NALU_HYPRE_Complex  sl;
   NALU_HYPRE_Complex  sr;

   if (ldata && rdata)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i, j, sl, sr) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < nrows; i++)
      {
         sl = ldata[i];
         for (j = A_i[i]; j < A_i[i + 1]; j++)
         {
            sr = rdata[A_j[j]];
            A_data[j] = sl * A_data[j] * sr;
         }
      }
   }
   else if (ldata && !rdata)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i, j, sl) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < nrows; i++)
      {
         sl = ldata[i];
         for (j = A_i[i]; j < A_i[i + 1]; j++)
         {
            A_data[j] = sl * A_data[j];
         }
      }
   }
   else if (!ldata && rdata)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i, j, sr) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < nrows; i++)
      {
         for (j = A_i[i]; j < A_i[i + 1]; j++)
         {
            sr = rdata[A_j[j]];
            A_data[j] = A_data[j] * sr;
         }
      }
   }
   else
   {
      /* Throw an error if the scaling factors should have a size different than zero */
      if (lsize || rsize)
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Scaling matrices are not set!\n");
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixDiagScale
 *
 * Computes A = diag(ld) * A * diag(rd), where the diagonal matrices
 * "diag(ld)" and "diag(rd)" are stored as local vectors.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixDiagScale( nalu_hypre_CSRMatrix *A,
                          nalu_hypre_Vector    *ld,
                          nalu_hypre_Vector    *rd)
{
   /* Sanity checks */
   if (ld && nalu_hypre_VectorSize(ld) && !nalu_hypre_VectorData(ld))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "ld scaling coefficients are not set\n");
      return nalu_hypre_error_flag;
   }

   if (rd && nalu_hypre_VectorSize(rd) && !nalu_hypre_VectorData(rd))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "rd scaling coefficients are not set\n");
      return nalu_hypre_error_flag;
   }

   if (!rd && !ld)
   {
      return nalu_hypre_error_flag;
   }

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec;

   if (ld && rd)
   {
      /* TODO (VPM): replace with GetExecPolicy3 */
      exec = nalu_hypre_GetExecPolicy2(nalu_hypre_CSRMatrixMemoryLocation(A),
                                  nalu_hypre_VectorMemoryLocation(ld));
   }
   else if (ld)
   {
      exec = nalu_hypre_GetExecPolicy2(nalu_hypre_CSRMatrixMemoryLocation(A),
                                  nalu_hypre_VectorMemoryLocation(ld));
   }
   else
   {
      exec = nalu_hypre_GetExecPolicy2(nalu_hypre_CSRMatrixMemoryLocation(A),
                                  nalu_hypre_VectorMemoryLocation(rd));
   }

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_CSRMatrixDiagScaleDevice(A, ld, rd);
   }
   else
#endif
   {
      nalu_hypre_CSRMatrixDiagScaleHost(A, ld, rd);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixSetConstantValues
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixSetConstantValues( nalu_hypre_CSRMatrix *A,
                                  NALU_HYPRE_Complex    value)
{
   NALU_HYPRE_Int i;
   NALU_HYPRE_Int nnz = nalu_hypre_CSRMatrixNumNonzeros(A);

   if (!nalu_hypre_CSRMatrixData(A))
   {
      nalu_hypre_CSRMatrixData(A) = nalu_hypre_TAlloc(NALU_HYPRE_Complex, nnz, nalu_hypre_CSRMatrixMemoryLocation(A));
   }

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_CSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      hypreDevice_ComplexFilln(nalu_hypre_CSRMatrixData(A), nnz, value);
   }
   else
#endif
   {
      for (i = 0; i < nnz; i++)
      {
         nalu_hypre_CSRMatrixData(A)[i] = value;
      }
   }

   return nalu_hypre_error_flag;
}
