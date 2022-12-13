/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec
 *--------------------------------------------------------------------------*/

/* y[offset:end] = alpha*A[offset:end,:]*x + beta*b[offset:end] */
NALU_HYPRE_Int
hypre_CSRMatrixMatvecOutOfPlaceHost( NALU_HYPRE_Complex    alpha,
                                     hypre_CSRMatrix *A,
                                     hypre_Vector    *x,
                                     NALU_HYPRE_Complex    beta,
                                     hypre_Vector    *b,
                                     hypre_Vector    *y,
                                     NALU_HYPRE_Int        offset )
{
   NALU_HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   NALU_HYPRE_Int        *A_i      = hypre_CSRMatrixI(A) + offset;
   NALU_HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A) - offset;
   NALU_HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);

   NALU_HYPRE_Int        *A_rownnz = hypre_CSRMatrixRownnz(A);
   NALU_HYPRE_Int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

   NALU_HYPRE_Complex    *x_data = hypre_VectorData(x);
   NALU_HYPRE_Complex    *b_data = hypre_VectorData(b) + offset;
   NALU_HYPRE_Complex    *y_data = hypre_VectorData(y) + offset;
   NALU_HYPRE_Int         x_size = hypre_VectorSize(x);
   NALU_HYPRE_Int         b_size = hypre_VectorSize(b) - offset;
   NALU_HYPRE_Int         y_size = hypre_VectorSize(y) - offset;
   NALU_HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   NALU_HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   NALU_HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   NALU_HYPRE_Int         idxstride_b = hypre_VectorIndexStride(b);
   NALU_HYPRE_Int         vecstride_b = hypre_VectorVectorStride(b);
   NALU_HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   NALU_HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);
   NALU_HYPRE_Complex     temp, tempx;
   NALU_HYPRE_Int         i, j, jj, m, ierr = 0;
   NALU_HYPRE_Real        xpar = 0.7;
   hypre_Vector     *x_tmp = NULL;

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

   hypre_assert(num_vectors == hypre_VectorNumVectors(y));
   hypre_assert(num_vectors == hypre_VectorNumVectors(b));
   hypre_assert(idxstride_b == idxstride_y);
   hypre_assert(vecstride_b == vecstride_y);

   if (num_cols != x_size)
   {
      ierr = 1;
   }

   if (num_rows != y_size || num_rows != b_size)
   {
      ierr = 2;
   }

   if (num_cols != x_size && (num_rows != y_size || num_rows != b_size))
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
      for (i = 0; i < num_rows * num_vectors; i++)
      {
         y_data[i] = beta * b_data[i];
      }

#ifdef NALU_HYPRE_PROFILE
      hypre_profile_times[NALU_HYPRE_TIMER_ID_MATVEC] += hypre_MPI_Wtime() - time_begin;
#endif

      return ierr;
   }

   if (x == y)
   {
      x_tmp = hypre_SeqVectorCloneDeep(x);
      x_data = hypre_VectorData(x_tmp);
   }

   temp = beta / alpha;

   if (num_vectors > 1)
   {
      /*-----------------------------------------------------------------------
       * y = (beta/alpha)*b
       *-----------------------------------------------------------------------*/

      if (temp == 0.0)
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows * num_vectors; i++)
         {
            y_data[i] = 0.0;
         }
      }
      else if (temp == 1.0)
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows * num_vectors; i++)
         {
            y_data[i] = b_data[i];
         }
      }
      else if (temp == -1.0)
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows * num_vectors; i++)
         {
            y_data[i] = -b_data[i];
         }
      }
      else
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows * num_vectors; i++)
         {
            y_data[i] = temp * b_data[i];
         }
      }

      /*-----------------------------------------------------------------
       * y += A*x
       *-----------------------------------------------------------------*/

      if (num_rownnz < xpar * num_rows)
      {
         switch (num_vectors)
         {
            case 2:
#ifdef NALU_HYPRE_USING_OPENMP
               #pragma omp parallel for private(i,j,jj,m) NALU_HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  m = A_rownnz[i];

                  NALU_HYPRE_Complex tmp[2] = {0.0, 0.0};
                  for (jj = A_i[m]; jj < A_i[m + 1]; jj++)
                  {
                     NALU_HYPRE_Int     xidx = A_j[jj] * idxstride_x;
                     NALU_HYPRE_Complex coef = A_data[jj];

                     tmp[0] += coef * x_data[xidx];
                     tmp[1] += coef * x_data[xidx + vecstride_x];
                  }
                  NALU_HYPRE_Int yidx = m * idxstride_y;

                  y_data[yidx] += tmp[0];
                  y_data[yidx + vecstride_y] += tmp[1];
               }
               break;

            case 3:
#ifdef NALU_HYPRE_USING_OPENMP
               #pragma omp parallel for private(i,j,jj,m) NALU_HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  m = A_rownnz[i];

                  NALU_HYPRE_Complex tmp[3] = {0.0, 0.0, 0.0};
                  for (jj = A_i[m]; jj < A_i[m + 1]; jj++)
                  {
                     NALU_HYPRE_Int     xidx = A_j[jj] * idxstride_x;
                     NALU_HYPRE_Complex coef = A_data[jj];

                     tmp[0] += coef * x_data[xidx];
                     tmp[1] += coef * x_data[xidx +   vecstride_x];
                     tmp[2] += coef * x_data[xidx + 2 * vecstride_x];
                  }
                  NALU_HYPRE_Int yidx = m * idxstride_y;

                  y_data[yidx] += tmp[0];
                  y_data[yidx +   vecstride_y] += tmp[1];
                  y_data[yidx + 2 * vecstride_y] += tmp[2];
               }
               break;

            case 4:
#ifdef NALU_HYPRE_USING_OPENMP
               #pragma omp parallel for private(i,j,jj,m) NALU_HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  m = A_rownnz[i];

                  NALU_HYPRE_Complex tmp[4] = {0.0, 0.0, 0.0, 0.0};
                  for (jj = A_i[m]; jj < A_i[m + 1]; jj++)
                  {
                     NALU_HYPRE_Int     xidx = A_j[jj] * idxstride_x;
                     NALU_HYPRE_Complex coef = A_data[jj];

                     tmp[0] += coef * x_data[xidx];
                     tmp[1] += coef * x_data[xidx +   vecstride_x];
                     tmp[2] += coef * x_data[xidx + 2 * vecstride_x];
                     tmp[3] += coef * x_data[xidx + 3 * vecstride_x];
                  }
                  NALU_HYPRE_Int yidx = m * idxstride_y;

                  y_data[yidx] += tmp[0];
                  y_data[yidx +   vecstride_y] += tmp[1];
                  y_data[yidx + 2 * vecstride_y] += tmp[2];
                  y_data[yidx + 3 * vecstride_y] += tmp[3];
               }
               break;

            default:
#ifdef NALU_HYPRE_USING_OPENMP
               #pragma omp parallel for private(i,j,jj,m,tempx) NALU_HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rownnz; i++)
               {
                  m = A_rownnz[i];
                  for (j = 0; j < num_vectors; j++)
                  {
                     tempx = 0.0;
                     for (jj = A_i[m]; jj < A_i[m + 1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j * vecstride_x + A_j[jj] * idxstride_x];
                     }
                     y_data[j * vecstride_y + m * idxstride_y] += tempx;
                  }
               }
               break;
         } /* switch (num_vectors) */
      }
      else
      {
         switch (num_vectors)
         {
            case 2:
#ifdef NALU_HYPRE_USING_OPENMP
               #pragma omp parallel for private(i,j,jj) NALU_HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  NALU_HYPRE_Complex tmp[2] = {0.0, 0.0};
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     NALU_HYPRE_Int     xidx = A_j[jj] * idxstride_x;
                     NALU_HYPRE_Complex coef = A_data[jj];

                     tmp[0] += coef * x_data[xidx];
                     tmp[1] += coef * x_data[xidx + vecstride_x];
                  }
                  NALU_HYPRE_Int yidx = i * idxstride_y;

                  y_data[yidx] += tmp[0];
                  y_data[yidx + vecstride_y] += tmp[1];
               }
               break;

            case 3:
#ifdef NALU_HYPRE_USING_OPENMP
               #pragma omp parallel for private(i,j,jj) NALU_HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  NALU_HYPRE_Complex tmp[3] = {0.0, 0.0, 0.0};
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     NALU_HYPRE_Int     xidx = A_j[jj] * idxstride_x;
                     NALU_HYPRE_Complex coef = A_data[jj];

                     tmp[0] += coef * x_data[xidx];
                     tmp[1] += coef * x_data[xidx +   vecstride_x];
                     tmp[2] += coef * x_data[xidx + 2 * vecstride_x];
                  }
                  NALU_HYPRE_Int yidx = i * idxstride_y;

                  y_data[yidx] += tmp[0];
                  y_data[yidx +   vecstride_y] += tmp[1];
                  y_data[yidx + 2 * vecstride_y] += tmp[2];
               }
               break;

            case 4:
#ifdef NALU_HYPRE_USING_OPENMP
               #pragma omp parallel for private(i,j,jj) NALU_HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  NALU_HYPRE_Complex tmp[4] = {0.0, 0.0, 0.0, 0.0};
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     NALU_HYPRE_Int     xidx = A_j[jj] * idxstride_x;
                     NALU_HYPRE_Complex coef = A_data[jj];

                     tmp[0] += coef * x_data[xidx];
                     tmp[1] += coef * x_data[xidx +   vecstride_x];
                     tmp[2] += coef * x_data[xidx + 2 * vecstride_x];
                     tmp[3] += coef * x_data[xidx + 3 * vecstride_x];
                  }
                  NALU_HYPRE_Int yidx = i * idxstride_y;

                  y_data[yidx] += tmp[0];
                  y_data[yidx +   vecstride_y] += tmp[1];
                  y_data[yidx + 2 * vecstride_y] += tmp[2];
                  y_data[yidx + 3 * vecstride_y] += tmp[3];
               }
               break;

            default:
#ifdef NALU_HYPRE_USING_OPENMP
               #pragma omp parallel for private(i,j,jj,tempx) NALU_HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < num_rows; i++)
               {
                  for (j = 0; j < num_vectors; ++j)
                  {
                     tempx = 0.0;
                     for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                     {
                        tempx += A_data[jj] * x_data[j * vecstride_x + A_j[jj] * idxstride_x];
                     }
                     y_data[j * vecstride_y + i * idxstride_y] += tempx;
                  }
               }
               break;
         } /* switch (num_vectors) */
      } /* if (num_rownnz < xpar * num_rows) */

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
   }
   else if (num_rownnz < xpar * num_rows)
   {
      /* use rownnz pointer to do the A*x multiplication when
         num_rownnz is smaller than xpar*num_rows */

      if (temp == 0.0)
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
         {
            y_data[i] = 0.0;
         }

         if (alpha == 1.0)
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i,j,m,tempx) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rownnz; i++)
            {
               m = A_rownnz[i];
               tempx = 0.0;
               for (j = A_i[m]; j < A_i[m + 1]; j++)
               {
                  tempx += A_data[j] * x_data[A_j[j]];
               }
               y_data[m] = tempx;
            }
         } // y = A*x
         else if (alpha == -1.0)
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i,j,m,tempx) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rownnz; i++)
            {
               m = A_rownnz[i];
               tempx = 0.0;
               for (j = A_i[m]; j < A_i[m + 1]; j++)
               {
                  tempx -= A_data[j] * x_data[A_j[j]];
               }
               y_data[m] = tempx;
            }
         } // y = -A*x
         else
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i,j,m,tempx) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rownnz; i++)
            {
               m = A_rownnz[i];
               tempx = 0.0;
               for (j = A_i[m]; j < A_i[m + 1]; j++)
               {
                  tempx += A_data[j] * x_data[A_j[j]];
               }
               y_data[m] = alpha * tempx;
            }
         } // y = alpha*A*x
      } // temp == 0
      else if (temp == -1.0) // beta == -alpha
      {
         if (alpha == 1.0)
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               y_data[i] = -b_data[i];
            }

#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i,j,m,tempx) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rownnz; i++)
            {
               m = A_rownnz[i];
               tempx = 0.0;
               for (j = A_i[m]; j < A_i[m + 1]; j++)
               {
                  tempx += A_data[j] * x_data[A_j[j]];
               }
               y_data[m] += tempx;
            }
         } // y = A*x - b
         else if (alpha == -1.0)
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               y_data[i] = b_data[i];
            }

#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i,j,m,tempx) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rownnz; i++)
            {
               m = A_rownnz[i];
               tempx = 0.0;
               for (j = A_i[m]; j < A_i[m + 1]; j++)
               {
                  tempx += A_data[j] * x_data[A_j[j]];
               }
               y_data[m] -= tempx;
            }
         } // y = -A*x + b
         else
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               y_data[i] = -alpha * b_data[i];
            }

#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i,j,m,tempx) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rownnz; i++)
            {
               m = A_rownnz[i];
               tempx = 0.0;
               for (j = A_i[m]; j < A_i[m + 1]; j++)
               {
                  tempx += A_data[j] * x_data[A_j[j]];
               }
               y_data[m] += alpha * tempx;
            }
         } // y = alpha*(A*x - b)
      } // temp == -1
      else if (temp == 1.0) // beta == alpha
      {
         if (alpha == 1.0)
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               y_data[i] = b_data[i];
            }

#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i,j,m,tempx) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rownnz; i++)
            {
               m = A_rownnz[i];
               tempx = 0.0;
               for (j = A_i[m]; j < A_i[m + 1]; j++)
               {
                  tempx += A_data[j] * x_data[A_j[j]];
               }
               y_data[m] += tempx;
            }
         } // y = A*x + b
         else if (alpha == -1.0)
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               y_data[i] = -b_data[i];
            }

#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i,j,m,tempx) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rownnz; i++)
            {
               m = A_rownnz[i];
               tempx = 0.0;
               for (j = A_i[m]; j < A_i[m + 1]; j++)
               {
                  tempx -= A_data[j] * x_data[A_j[j]];
               }
               y_data[m] += tempx;
            }
         } // y = -A*x - b
         else
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               y_data[i] = alpha * b_data[i];
            }

#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i,j,m,tempx) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rownnz; i++)
            {
               m = A_rownnz[i];
               tempx = 0.0;
               for (j = A_i[m]; j < A_i[m + 1]; j++)
               {
                  tempx += A_data[j] * x_data[A_j[j]];
               }
               y_data[m] += alpha * tempx;
            }
         } // y = alpha*(A*x + b)
      }
      else
      {
         if (alpha == 1.0)
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               y_data[i] = beta * b_data[i];
            }

#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i,j,m,tempx) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rownnz; i++)
            {
               m = A_rownnz[i];
               tempx = 0.0;
               for (j = A_i[m]; j < A_i[m + 1]; j++)
               {
                  tempx += A_data[j] * x_data[A_j[j]];
               }
               y_data[m] += tempx;
            }
         } // y = A*x + beta*b
         else if (-1 == alpha)
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               y_data[i] = -temp * b_data[i];
            }

#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i,j,m,tempx) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rownnz; i++)
            {
               m = A_rownnz[i];
               tempx = 0.0;
               for (j = A_i[m]; j < A_i[m + 1]; j++)
               {
                  tempx -= A_data[j] * x_data[A_j[j]];
               }
               y_data[m] += tempx;
            }
         } // y = -A*x - temp*b
         else
         {
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               y_data[i] = beta * b_data[i];
            }

#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp parallel for private(i,j,m,tempx) NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rownnz; i++)
            {
               m = A_rownnz[i];
               tempx = 0.0;
               for (j = A_i[m]; j < A_i[m + 1]; j++)
               {
                  tempx += A_data[j] * x_data[A_j[j]];
               }
               y_data[m] += alpha * tempx;
            }
         } // y = alpha*(A*x + temp*b)
      } // temp != 0 && temp != -1 && temp != 1
   }
   else
   {
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel private(i,jj,tempx)
#endif
      {
         NALU_HYPRE_Int iBegin = hypre_CSRMatrixGetLoadBalancedPartitionBegin(A);
         NALU_HYPRE_Int iEnd = hypre_CSRMatrixGetLoadBalancedPartitionEnd(A);
         hypre_assert(iBegin <= iEnd);
         hypre_assert(iBegin >= 0 && iBegin <= num_rows);
         hypre_assert(iEnd >= 0 && iEnd <= num_rows);

         if (temp == 0.0)
         {
            if (alpha == 1.0) // JSP: a common path
            {
               for (i = iBegin; i < iEnd; i++)
               {
                  tempx = 0.0;
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = tempx;
               }
            } // y = A*x
            else if (alpha == -1.0)
            {
               for (i = iBegin; i < iEnd; i++)
               {
                  tempx = 0.0;
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     tempx -= A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = tempx;
               }
            } // y = -A*x
            else
            {
               for (i = iBegin; i < iEnd; i++)
               {
                  tempx = 0.0;
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] = alpha * tempx;
               }
            } // y = alpha*A*x
         } // temp == 0
         else if (temp == -1.0) // beta == -alpha
         {
            if (alpha == 1.0) // JSP: a common path
            {
               for (i = iBegin; i < iEnd; i++)
               {
                  y_data[i] = -b_data[i];
                  tempx = 0.0;
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] += tempx;
               }
            } // y = A*x - y
            else if (alpha == -1.0) // JSP: a common path
            {
               for (i = iBegin; i < iEnd; i++)
               {
                  y_data[i] = b_data[i];
                  tempx = 0.0;
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     tempx -= A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] += tempx;
               }
            } // y = -A*x + y
            else
            {
               for (i = iBegin; i < iEnd; i++)
               {
                  y_data[i] = -alpha * b_data[i];
                  tempx = 0.0;
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] += alpha * tempx;
               }
            } // y = alpha*(A*x - y)
         } // temp == -1
         else if (temp == 1.0)
         {
            if (alpha == 1.0) // JSP: a common path
            {
               for (i = iBegin; i < iEnd; i++)
               {
                  y_data[i] = b_data[i];
                  tempx = 0.0;
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] += tempx;
               }
            } // y = A*x + y
            else if (alpha == -1.0)
            {
               for (i = iBegin; i < iEnd; i++)
               {
                  y_data[i] = -b_data[i];
                  tempx = 0.0;
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     tempx -= A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] += tempx;
               }
            } // y = -A*x - y
            else
            {
               for (i = iBegin; i < iEnd; i++)
               {
                  y_data[i] = alpha * b_data[i];
                  tempx = 0.0;
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] += alpha * tempx;
               }
            } // y = alpha*(A*x + y)
         }
         else
         {
            if (alpha == 1.0) // JSP: a common path
            {
               for (i = iBegin; i < iEnd; i++)
               {
                  y_data[i] = b_data[i] * temp;
                  tempx = 0.0;
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] += tempx;
               }
            } // y = A*x + temp*y
            else if (alpha == -1.0)
            {
               for (i = iBegin; i < iEnd; i++)
               {
                  y_data[i] = -b_data[i] * temp;
                  tempx = 0.0;
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     tempx -= A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] += tempx;
               }
            } // y = -A*x - temp*y
            else
            {
               for (i = iBegin; i < iEnd; i++)
               {
                  y_data[i] = b_data[i] * beta;
                  tempx = 0.0;
                  for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
                  {
                     tempx += A_data[jj] * x_data[A_j[jj]];
                  }
                  y_data[i] += alpha * tempx;
               }
            } // y = alpha*(A*x + temp*y)
         } // temp != 0 && temp != -1 && temp != 1
      } // omp parallel
   }

   if (x == y)
   {
      hypre_SeqVectorDestroy(x_tmp);
   }

   return ierr;
}

NALU_HYPRE_Int
hypre_CSRMatrixMatvecOutOfPlace( NALU_HYPRE_Complex    alpha,
                                 hypre_CSRMatrix *A,
                                 hypre_Vector    *x,
                                 NALU_HYPRE_Complex    beta,
                                 hypre_Vector    *b,
                                 hypre_Vector    *y,
                                 NALU_HYPRE_Int        offset )
{
#ifdef NALU_HYPRE_PROFILE
   NALU_HYPRE_Real time_begin = hypre_MPI_Wtime();
#endif

   NALU_HYPRE_Int ierr = 0;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_CSRMatrixMemoryLocation(A) );
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_CSRMatrixMatvecDevice(0, alpha, A, x, beta, b, y, offset);
   }
   else
#endif
   {
      ierr = hypre_CSRMatrixMatvecOutOfPlaceHost(alpha, A, x, beta, b, y, offset);
   }

#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_MATVEC] += hypre_MPI_Wtime() - time_begin;
#endif

   return ierr;
}

NALU_HYPRE_Int
hypre_CSRMatrixMatvec( NALU_HYPRE_Complex    alpha,
                       hypre_CSRMatrix *A,
                       hypre_Vector    *x,
                       NALU_HYPRE_Complex    beta,
                       hypre_Vector    *y     )
{
   return hypre_CSRMatrixMatvecOutOfPlace(alpha, A, x, beta, y, y, 0);
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvecT
 *
 *  This version is using a different (more efficient) threading scheme

 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_CSRMatrixMatvecTHost( NALU_HYPRE_Complex    alpha,
                            hypre_CSRMatrix *A,
                            hypre_Vector    *x,
                            NALU_HYPRE_Complex    beta,
                            hypre_Vector    *y     )
{
   NALU_HYPRE_Complex    *A_data    = hypre_CSRMatrixData(A);
   NALU_HYPRE_Int        *A_i       = hypre_CSRMatrixI(A);
   NALU_HYPRE_Int        *A_j       = hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int         num_rows  = hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int         num_cols  = hypre_CSRMatrixNumCols(A);

   NALU_HYPRE_Complex    *x_data = hypre_VectorData(x);
   NALU_HYPRE_Complex    *y_data = hypre_VectorData(y);
   NALU_HYPRE_Int         x_size = hypre_VectorSize(x);
   NALU_HYPRE_Int         y_size = hypre_VectorSize(y);
   NALU_HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   NALU_HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
   NALU_HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
   NALU_HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
   NALU_HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);

   NALU_HYPRE_Complex     temp;

   NALU_HYPRE_Complex    *y_data_expand;
   NALU_HYPRE_Int         my_thread_num = 0, offset = 0;

   NALU_HYPRE_Int         i, j, jv, jj;
   NALU_HYPRE_Int         num_threads;

   NALU_HYPRE_Int         ierr  = 0;

   hypre_Vector     *x_tmp = NULL;

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

   hypre_assert( num_vectors == hypre_VectorNumVectors(y) );

   if (num_rows != x_size)
   {
      ierr = 1;
   }

   if (num_cols != y_size)
   {
      ierr = 2;
   }

   if (num_rows != x_size && num_cols != y_size)
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
      for (i = 0; i < num_cols * num_vectors; i++)
      {
         y_data[i] *= beta;
      }

      return ierr;
   }

   if (x == y)
   {
      x_tmp = hypre_SeqVectorCloneDeep(x);
      x_data = hypre_VectorData(x_tmp);
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
         for (i = 0; i < num_cols * num_vectors; i++)
         {
            y_data[i] = 0.0;
         }
      }
      else
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_cols * num_vectors; i++)
         {
            y_data[i] *= temp;
         }
      }
   }

   /*-----------------------------------------------------------------
    * y += A^T*x
    *-----------------------------------------------------------------*/
   num_threads = hypre_NumThreads();
   if (num_threads > 1)
   {
      y_data_expand = hypre_CTAlloc(NALU_HYPRE_Complex,  num_threads * y_size, NALU_HYPRE_MEMORY_HOST);

      if ( num_vectors == 1 )
      {

#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel private(i,jj,j,my_thread_num,offset)
#endif
         {
            my_thread_num = hypre_GetThreadNum();
            offset =  y_size * my_thread_num;
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp for NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < num_rows; i++)
            {
               for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
               {
                  j = A_j[jj];
                  y_data_expand[offset + j] += A_data[jj] * x_data[i];
               }
            }

            /* implied barrier (for threads)*/
#ifdef NALU_HYPRE_USING_OPENMP
            #pragma omp for NALU_HYPRE_SMP_SCHEDULE
#endif
            for (i = 0; i < y_size; i++)
            {
               for (j = 0; j < num_threads; j++)
               {
                  y_data[i] += y_data_expand[j * y_size + i];

               }
            }

         } /* end parallel threaded region */
      }
      else
      {
         /* multiple vector case is not threaded */
         for (i = 0; i < num_rows; i++)
         {
            for ( jv = 0; jv < num_vectors; ++jv )
            {
               for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
               {
                  j = A_j[jj];
                  y_data[ j * idxstride_y + jv * vecstride_y ] +=
                     A_data[jj] * x_data[ i * idxstride_x + jv * vecstride_x];
               }
            }
         }
      }

      hypre_TFree(y_data_expand, NALU_HYPRE_MEMORY_HOST);

   }
   else
   {
      for (i = 0; i < num_rows; i++)
      {
         if ( num_vectors == 1 )
         {
            for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
            {
               j = A_j[jj];
               y_data[j] += A_data[jj] * x_data[i];
            }
         }
         else
         {
            for ( jv = 0; jv < num_vectors; ++jv )
            {
               for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
               {
                  j = A_j[jj];
                  y_data[ j * idxstride_y + jv * vecstride_y ] +=
                     A_data[jj] * x_data[ i * idxstride_x + jv * vecstride_x ];
               }
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
      for (i = 0; i < num_cols * num_vectors; i++)
      {
         y_data[i] *= alpha;
      }
   }

   if (x == y)
   {
      hypre_SeqVectorDestroy(x_tmp);
   }

   return ierr;
}

NALU_HYPRE_Int
hypre_CSRMatrixMatvecT( NALU_HYPRE_Complex    alpha,
                        hypre_CSRMatrix *A,
                        hypre_Vector    *x,
                        NALU_HYPRE_Complex    beta,
                        hypre_Vector    *y )
{
#ifdef NALU_HYPRE_PROFILE
   NALU_HYPRE_Real time_begin = hypre_MPI_Wtime();
#endif

   NALU_HYPRE_Int ierr = 0;

#if defined(NALU_HYPRE_USING_GPU)
   NALU_HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_CSRMatrixMemoryLocation(A) );
   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      ierr = hypre_CSRMatrixMatvecDevice(1, alpha, A, x, beta, y, y, 0 );
   }
   else
#endif
   {
      ierr = hypre_CSRMatrixMatvecTHost(alpha, A, x, beta, y);
   }

#ifdef NALU_HYPRE_PROFILE
   hypre_profile_times[NALU_HYPRE_TIMER_ID_MATVEC] += hypre_MPI_Wtime() - time_begin;
#endif

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec_FF
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_CSRMatrixMatvec_FF( NALU_HYPRE_Complex    alpha,
                          hypre_CSRMatrix *A,
                          hypre_Vector    *x,
                          NALU_HYPRE_Complex    beta,
                          hypre_Vector    *y,
                          NALU_HYPRE_Int       *CF_marker_x,
                          NALU_HYPRE_Int       *CF_marker_y,
                          NALU_HYPRE_Int        fpt )
{
   NALU_HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   NALU_HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   NALU_HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);

   NALU_HYPRE_Complex    *x_data = hypre_VectorData(x);
   NALU_HYPRE_Complex    *y_data = hypre_VectorData(y);
   NALU_HYPRE_Int         x_size = hypre_VectorSize(x);
   NALU_HYPRE_Int         y_size = hypre_VectorSize(y);

   NALU_HYPRE_Complex      temp;

   NALU_HYPRE_Int         i, jj;

   NALU_HYPRE_Int         ierr = 0;

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

   if (num_cols != x_size)
   {
      ierr = 1;
   }

   if (num_rows != y_size)
   {
      ierr = 2;
   }

   if (num_cols != x_size && num_rows != y_size)
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
      for (i = 0; i < num_rows; i++)
         if (CF_marker_x[i] == fpt) { y_data[i] *= beta; }

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
         for (i = 0; i < num_rows; i++)
            if (CF_marker_x[i] == fpt) { y_data[i] = 0.0; }
      }
      else
      {
#ifdef NALU_HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) NALU_HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < num_rows; i++)
            if (CF_marker_x[i] == fpt) { y_data[i] *= temp; }
      }
   }

   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/

#ifdef NALU_HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,jj) NALU_HYPRE_SMP_SCHEDULE
#endif

   for (i = 0; i < num_rows; i++)
   {
      if (CF_marker_x[i] == fpt)
      {
         temp = y_data[i];
         for (jj = A_i[i]; jj < A_i[i + 1]; jj++)
            if (CF_marker_y[A_j[jj]] == fpt) { temp += A_data[jj] * x_data[A_j[jj]]; }
         y_data[i] = temp;
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
      for (i = 0; i < num_rows; i++)
         if (CF_marker_x[i] == fpt) { y_data[i] *= alpha; }
   }

   return ierr;
}
