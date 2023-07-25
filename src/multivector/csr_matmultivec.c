/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for nalu_hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "csr_multimatvec.h"
#include "seq_mv.h"
#include "seq_multivector.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixMultiMatvec
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixMatMultivec(NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A,
                           nalu_hypre_Multivector *x, NALU_HYPRE_Complex beta,
                           nalu_hypre_Multivector *y)
{
   NALU_HYPRE_Complex *A_data   = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int    *A_i      = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int    *A_j      = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int    num_rows = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int    num_cols = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Complex *x_data = nalu_hypre_MultivectorData(x);
   NALU_HYPRE_Complex *y_data = nalu_hypre_MultivectorData(y);
   NALU_HYPRE_Int    x_size = nalu_hypre_MultivectorSize(x);
   NALU_HYPRE_Int    y_size = nalu_hypre_MultivectorSize(y);
   NALU_HYPRE_Int    num_vectors = nalu_hypre_MultivectorNumVectors(x);
   NALU_HYPRE_Int    *x_active_ind = x->active_indices;
   NALU_HYPRE_Int    *y_active_ind = y->active_indices;
   NALU_HYPRE_Int    num_active_vectors = x->num_active_vectors;
   NALU_HYPRE_Int    i, j, jj, m, ierr = 0, optimize;
   NALU_HYPRE_Complex temp, tempx, xpar = 0.7, *xptr, *yptr;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/

   nalu_hypre_assert(num_active_vectors == y->num_active_vectors);
   if (num_cols != x_size) { ierr = 1; }
   if (num_rows != y_size) { ierr = 2; }
   if (num_cols != x_size && num_rows != y_size) { ierr = 3; }
   optimize = 0;
   if (num_active_vectors == num_vectors && num_vectors == y->num_vectors)
   {
      optimize = 1;
   }

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows * num_vectors; i++) { y_data[i] *= beta; }

      return ierr;
   }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/

   temp = beta / alpha;

   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows * num_vectors; i++) { y_data[i] = 0.0; }
      }
      else
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows * num_vectors; i++) { y_data[i] *= temp; }
      }
   }

   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/

   if ( num_vectors == 1 )
   {
      for (i = 0; i < num_rows; i++)
      {
         temp = y_data[i];
         for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
         {
            temp += A_data[jj] * x_data[A_j[jj]];
         }
         y_data[i] = temp;
      }
   }
   else
   {
      if (optimize == 0)
      {
         for (i = 0; i < num_rows; i++)
         {
            for (j = 0; j < num_active_vectors; ++j)
            {
               xptr = x_data[x_active_ind[j] * x_size];
               temp = y_data[y_active_ind[j] * y_size + i];
               for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
               {
                  temp += A_data[jj] * xptr[A_j[jj]];
               }
               y_data[y_active_ind[j]*y_size + i] = temp;
            }
         }
      }
      else
      {
         for (i = 0; i < num_rows; i++)
         {
            for (j = 0; j < num_vectors; ++j)
            {
               xptr = x_data[j * x_size];
               temp = y_data[j * y_size + i];
               for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
               {
                  temp += A_data[jj] * xptr[A_j[jj]];
               }
               y_data[j * y_size + i] = temp;
            }
         }
         /* different version
         for (j=0; j<num_vectors; ++j)
         {
            xptr = x_data[j*x_size];
            for (i = 0; i < num_rows; i++)
            {
               temp = y_data[j*y_size+i];
               for (jj = A_i[i]; jj < A_i[i+1]; jj++)
                  temp += A_data[jj] * xptr[A_j[jj]];
               y_data[j*y_size+i] = temp;
            }
         }
         */
      }
   }

   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows * num_vectors; i++)
      {
         y_data[i] *= alpha;
      }
   }
   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixMultiMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of nalu_hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixMatMultivecT(NALU_HYPRE_Complex alpha, nalu_hypre_CSRMatrix *A,
                            nalu_hypre_Multivector *x, NALU_HYPRE_Complex beta,
                            nalu_hypre_Multivector *y)
{
   NALU_HYPRE_Complex *A_data    = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int    *A_i       = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int    *A_j       = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int    num_rows  = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int    num_cols  = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Complex *x_data = nalu_hypre_MultivectorData(x);
   NALU_HYPRE_Complex *y_data = nalu_hypre_MultivectorData(y);
   NALU_HYPRE_Int    x_size = nalu_hypre_MultivectorSize(x);
   NALU_HYPRE_Int    y_size = nalu_hypre_MultivectorSize(y);
   NALU_HYPRE_Int    num_vectors = nalu_hypre_MultivectorNumVectors(x);
   NALU_HYPRE_Int    *x_active_ind = x->active_indices;
   NALU_HYPRE_Int    *y_active_ind = y->active_indices;
   NALU_HYPRE_Int    num_active_vectors = x->num_active_vectors;
   NALU_HYPRE_Complex temp;
   NALU_HYPRE_Int    i, jv, jj, size, ierr = 0;

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

   nalu_hypre_assert(num_active_vectors == y->num_active_vectors);
   if (num_rows != x_size) { ierr = 1; }
   if (num_cols != y_size) { ierr = 2; }
   if (num_rows != x_size && num_cols != y_size) { ierr = 3; }

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols * num_vectors; i++) { y_data[i] *= beta; }
      return ierr;
   }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/

   temp = beta / alpha;

   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_cols * num_vectors; i++) { y_data[i] = 0.0; }
      }
      else
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_cols * num_vectors; i++) { y_data[i] *= temp; }
      }
   }

   /*-----------------------------------------------------------------
    * y += A^T*x
    *-----------------------------------------------------------------*/

   if ( num_vectors == 1 )
   {
      for (i = 0; i < num_rows; i++)
      {
         for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
         {
            y_data[A_j[jj]] += A_data[jj] * x_data[i];
         }
      }
   }
   else
   {
      for ( jv = 0; jv < num_vectors; ++jv )
      {
         for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
         {
            y_data[A_j[jj] + jv * y_size] += A_data[jj] * x_data[i + jv * x_size];
         }
      }
   }

   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols * num_vectors; i++)
      {
         y_data[i] *= alpha;
      }
   }

   return ierr;
}

