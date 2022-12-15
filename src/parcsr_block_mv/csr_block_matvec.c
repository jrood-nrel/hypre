/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for nalu_hypre_CSRBlockMatrix class.
 *
 *****************************************************************************/

#include "csr_block_matrix.h"
#include "../seq_mv/seq_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRBlockMatrixMatvec
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRBlockMatrixMatvec(NALU_HYPRE_Complex alpha, nalu_hypre_CSRBlockMatrix *A,
                           nalu_hypre_Vector *x, NALU_HYPRE_Complex beta, nalu_hypre_Vector *y)
{
   NALU_HYPRE_Complex    *A_data   = nalu_hypre_CSRBlockMatrixData(A);
   NALU_HYPRE_Int        *A_i      = nalu_hypre_CSRBlockMatrixI(A);
   NALU_HYPRE_Int        *A_j      = nalu_hypre_CSRBlockMatrixJ(A);
   NALU_HYPRE_Int         num_rows = nalu_hypre_CSRBlockMatrixNumRows(A);
   NALU_HYPRE_Int         num_cols = nalu_hypre_CSRBlockMatrixNumCols(A);
   NALU_HYPRE_Int         blk_size = nalu_hypre_CSRBlockMatrixBlockSize(A);

   NALU_HYPRE_Complex    *x_data = nalu_hypre_VectorData(x);
   NALU_HYPRE_Complex    *y_data = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int         x_size = nalu_hypre_VectorSize(x);
   NALU_HYPRE_Int         y_size = nalu_hypre_VectorSize(y);

   NALU_HYPRE_Int         i, b1, b2, jj, bnnz = blk_size * blk_size;
   NALU_HYPRE_Int         ierr = 0;
   NALU_HYPRE_Complex     temp;

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

   if (num_cols * blk_size != x_size) { ierr = 1; }
   if (num_rows * blk_size != y_size) { ierr = 2; }
   if (num_cols * blk_size != x_size && num_rows * blk_size != y_size) { ierr = 3; }

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows * blk_size; i++) { y_data[i] *= beta; }

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
         for (i = 0; i < num_rows * blk_size; i++)
         {
            y_data[i] = 0.0;
         }
      }
      else
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows * blk_size; i++)
         {
            y_data[i] *= temp;
         }
      }
   }

   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,jj,b1,b2,temp) NALU_HYPRE_SMP_SCHEDULE
#endif

   for (i = 0; i < num_rows; i++)
   {
      for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
      {
         for (b1 = 0; b1 < blk_size; b1++)
         {
            temp = y_data[i * blk_size + b1];
            for (b2 = 0; b2 < blk_size; b2++)
            {
               temp += A_data[jj * bnnz + b1 * blk_size + b2] * x_data[A_j[jj] * blk_size + b2];
            }
            y_data[i * blk_size + b1] = temp;
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
      for (i = 0; i < num_rows * blk_size; i++)
      {
         y_data[i] *= alpha;
      }
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_CSRBlockMatrixMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of nalu_hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRBlockMatrixMatvecT( NALU_HYPRE_Complex         alpha,
                             nalu_hypre_CSRBlockMatrix *A,
                             nalu_hypre_Vector         *x,
                             NALU_HYPRE_Complex         beta,
                             nalu_hypre_Vector          *y     )
{
   NALU_HYPRE_Complex    *A_data    = nalu_hypre_CSRBlockMatrixData(A);
   NALU_HYPRE_Int        *A_i       = nalu_hypre_CSRBlockMatrixI(A);
   NALU_HYPRE_Int        *A_j       = nalu_hypre_CSRBlockMatrixJ(A);
   NALU_HYPRE_Int         num_rows  = nalu_hypre_CSRBlockMatrixNumRows(A);
   NALU_HYPRE_Int         num_cols  = nalu_hypre_CSRBlockMatrixNumCols(A);

   NALU_HYPRE_Complex    *x_data = nalu_hypre_VectorData(x);
   NALU_HYPRE_Complex    *y_data = nalu_hypre_VectorData(y);
   NALU_HYPRE_Int         x_size = nalu_hypre_VectorSize(x);
   NALU_HYPRE_Int         y_size = nalu_hypre_VectorSize(y);

   NALU_HYPRE_Complex     temp;

   NALU_HYPRE_Int         i, j, jj;
   NALU_HYPRE_Int         ierr  = 0;
   NALU_HYPRE_Int         b1, b2;

   NALU_HYPRE_Int         blk_size = nalu_hypre_CSRBlockMatrixBlockSize(A);
   NALU_HYPRE_Int         bnnz = blk_size * blk_size;

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

   if (num_rows * blk_size != x_size)
   {
      ierr = 1;
   }

   if (num_cols * blk_size != y_size)
   {
      ierr = 2;
   }

   if (num_rows * blk_size != x_size && num_cols * blk_size != y_size)
   {
      ierr = 3;
   }
   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

   if (alpha == 0.0)
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_cols * blk_size; i++)
      {
         y_data[i] *= beta;
      }

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
         for (i = 0; i < num_cols * blk_size; i++)
         {
            y_data[i] = 0.0;
         }
      }
      else
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_cols * blk_size; i++)
         {
            y_data[i] *= temp;
         }
      }
   }

   /*-----------------------------------------------------------------
    * y += A^T*x
    *-----------------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i, jj,j, b1, b2) NALU_HYPRE_SMP_SCHEDULE
#endif

   for (i = 0; i < num_rows; i++)
   {
      for (jj = A_i[i]; jj < A_i[i + 1]; jj++) /*each nonzero in that row*/
      {
         for (b1 = 0; b1 < blk_size; b1++) /*row */
         {
            for (b2 = 0; b2 < blk_size; b2++) /*col*/
            {
               j = A_j[jj]; /*col */
               y_data[j * blk_size + b2] +=
                  A_data[jj * bnnz + b1 * blk_size + b2] * x_data[i * blk_size + b1];
            }
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
      for (i = 0; i < num_cols * blk_size; i++)
      {
         y_data[i] *= alpha;
      }
   }

   return ierr;
}

