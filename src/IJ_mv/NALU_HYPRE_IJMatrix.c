/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_IJMatrix interface
 *
 *****************************************************************************/

#include "./_nalu_hypre_IJ_mv.h"

#include "../NALU_HYPRE.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixCreate( MPI_Comm        comm,
                      NALU_HYPRE_BigInt    ilower,
                      NALU_HYPRE_BigInt    iupper,
                      NALU_HYPRE_BigInt    jlower,
                      NALU_HYPRE_BigInt    jupper,
                      NALU_HYPRE_IJMatrix *matrix )
{
   NALU_HYPRE_BigInt info[2];
   NALU_HYPRE_Int num_procs;
   NALU_HYPRE_Int myid;

   nalu_hypre_IJMatrix *ijmatrix;

   NALU_HYPRE_BigInt  row0, col0, rowN, colN;

   ijmatrix = nalu_hypre_CTAlloc(nalu_hypre_IJMatrix, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_IJMatrixComm(ijmatrix)           = comm;
   nalu_hypre_IJMatrixObject(ijmatrix)         = NULL;
   nalu_hypre_IJMatrixTranslator(ijmatrix)     = NULL;
   nalu_hypre_IJMatrixAssumedPart(ijmatrix)    = NULL;
   nalu_hypre_IJMatrixObjectType(ijmatrix)     = NALU_HYPRE_UNITIALIZED;
   nalu_hypre_IJMatrixAssembleFlag(ijmatrix)   = 0;
   nalu_hypre_IJMatrixPrintLevel(ijmatrix)     = 0;
   nalu_hypre_IJMatrixOMPFlag(ijmatrix)        = 0;

   nalu_hypre_MPI_Comm_size(comm, &num_procs);
   nalu_hypre_MPI_Comm_rank(comm, &myid);


   if (ilower > iupper + 1 || ilower < 0)
   {
      nalu_hypre_error_in_arg(2);
      nalu_hypre_TFree(ijmatrix, NALU_HYPRE_MEMORY_HOST);
      return nalu_hypre_error_flag;
   }

   if (iupper < -1)
   {
      nalu_hypre_error_in_arg(3);
      nalu_hypre_TFree(ijmatrix, NALU_HYPRE_MEMORY_HOST);
      return nalu_hypre_error_flag;
   }

   if (jlower > jupper + 1 || jlower < 0)
   {
      nalu_hypre_error_in_arg(4);
      nalu_hypre_TFree(ijmatrix, NALU_HYPRE_MEMORY_HOST);
      return nalu_hypre_error_flag;
   }

   if (jupper < -1)
   {
      nalu_hypre_error_in_arg(5);
      nalu_hypre_TFree(ijmatrix, NALU_HYPRE_MEMORY_HOST);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_IJMatrixRowPartitioning(ijmatrix)[0] = ilower;
   nalu_hypre_IJMatrixRowPartitioning(ijmatrix)[1] = iupper + 1;
   nalu_hypre_IJMatrixColPartitioning(ijmatrix)[0] = jlower;
   nalu_hypre_IJMatrixColPartitioning(ijmatrix)[1] = jupper + 1;

   /* now we need the global number of rows and columns as well
      as the global first row and column index */

   /* proc 0 has the first row and col */
   if (myid == 0)
   {
      info[0] = ilower;
      info[1] = jlower;
   }
   nalu_hypre_MPI_Bcast(info, 2, NALU_HYPRE_MPI_BIG_INT, 0, comm);
   row0 = info[0];
   col0 = info[1];

   /* proc (num_procs-1) has the last row and col */
   if (myid == (num_procs - 1))
   {
      info[0] = iupper;
      info[1] = jupper;
   }
   nalu_hypre_MPI_Bcast(info, 2, NALU_HYPRE_MPI_BIG_INT, num_procs - 1, comm);

   rowN = info[0];
   colN = info[1];

   nalu_hypre_IJMatrixGlobalFirstRow(ijmatrix) = row0;
   nalu_hypre_IJMatrixGlobalFirstCol(ijmatrix) = col0;
   nalu_hypre_IJMatrixGlobalNumRows(ijmatrix) = rowN - row0 + 1;
   nalu_hypre_IJMatrixGlobalNumCols(ijmatrix) = colN - col0 + 1;

   *matrix = (NALU_HYPRE_IJMatrix) ijmatrix;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixDestroy( NALU_HYPRE_IJMatrix matrix )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (ijmatrix)
   {
      if nalu_hypre_IJMatrixAssumedPart(ijmatrix)
      {
         nalu_hypre_AssumedPartitionDestroy((nalu_hypre_IJAssumedPart*)nalu_hypre_IJMatrixAssumedPart(ijmatrix));
      }
      if ( nalu_hypre_IJMatrixObjectType(ijmatrix) == NALU_HYPRE_PARCSR )
      {
         nalu_hypre_IJMatrixDestroyParCSR( ijmatrix );
      }
      else if ( nalu_hypre_IJMatrixObjectType(ijmatrix) != -1 )
      {
         nalu_hypre_error_in_arg(1);
         return nalu_hypre_error_flag;
      }
   }

   nalu_hypre_TFree(ijmatrix, NALU_HYPRE_MEMORY_HOST);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixInitialize( NALU_HYPRE_IJMatrix matrix )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixObjectType(ijmatrix) == NALU_HYPRE_PARCSR )
   {
      nalu_hypre_IJMatrixInitializeParCSR( ijmatrix ) ;
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;

}

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixInitialize_v2( NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_MemoryLocation memory_location )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixObjectType(ijmatrix) == NALU_HYPRE_PARCSR )
   {
      nalu_hypre_IJMatrixInitializeParCSR_v2( ijmatrix, memory_location ) ;
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;

}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixSetPrintLevel( NALU_HYPRE_IJMatrix matrix,
                             NALU_HYPRE_Int print_level )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_IJMatrixPrintLevel(ijmatrix) = 1;
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * This is a helper routine to compute a prefix sum of integer values.
 *
 * The current implementation is okay for modest numbers of threads.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_PrefixSumInt(NALU_HYPRE_Int   nvals,
                   NALU_HYPRE_Int  *vals,
                   NALU_HYPRE_Int  *sums)
{
   NALU_HYPRE_Int  j, nthreads, bsize;

   nthreads = nalu_hypre_NumThreads();
   bsize = (nvals + nthreads - 1) / nthreads; /* This distributes the remainder */

   if (nvals < nthreads || bsize == 1)
   {
      sums[0] = 0;
      for (j = 1; j < nvals; j++)
      {
         sums[j] += sums[j - 1] + vals[j - 1];
      }
   }
   else
   {

      /* Compute preliminary partial sums (in parallel) within each interval */
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (j = 0; j < nvals; j += bsize)
      {
         NALU_HYPRE_Int  i, n = nalu_hypre_min((j + bsize), nvals);

         sums[j] = 0;
         for (i = j + 1; i < n; i++)
         {
            sums[i] = sums[i - 1] + vals[i - 1];
         }
      }

      /* Compute final partial sums (in serial) for the first entry of every interval */
      for (j = bsize; j < nvals; j += bsize)
      {
         sums[j] = sums[j - bsize] + sums[j - 1] + vals[j - 1];
      }

      /* Compute final partial sums (in parallel) for the remaining entries */
#ifdef NALU_HYPRE_USING_OPENMP
      #pragma omp parallel for private(j) NALU_HYPRE_SMP_SCHEDULE
#endif
      for (j = bsize; j < nvals; j += bsize)
      {
         NALU_HYPRE_Int  i, n = nalu_hypre_min((j + bsize), nvals);

         for (i = j + 1; i < n; i++)
         {
            sums[i] += sums[j];
         }
      }
   }

   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixSetValues( NALU_HYPRE_IJMatrix       matrix,
                         NALU_HYPRE_Int            nrows,
                         NALU_HYPRE_Int           *ncols,
                         const NALU_HYPRE_BigInt  *rows,
                         const NALU_HYPRE_BigInt  *cols,
                         const NALU_HYPRE_Complex *values )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (nrows == 0)
   {
      return nalu_hypre_error_flag;
   }

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /*
   if (!ncols)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   */

   if (!rows)
   {
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }

   if (!cols)
   {
      nalu_hypre_error_in_arg(5);
      return nalu_hypre_error_flag;
   }

   if (!values)
   {
      nalu_hypre_error_in_arg(6);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixObjectType(ijmatrix) != NALU_HYPRE_PARCSR )
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   NALU_HYPRE_IJMatrixSetValues2(matrix, nrows, ncols, rows, NULL, cols, values);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_IJMatrixSetValues2( NALU_HYPRE_IJMatrix       matrix,
                          NALU_HYPRE_Int            nrows,
                          NALU_HYPRE_Int           *ncols,
                          const NALU_HYPRE_BigInt  *rows,
                          const NALU_HYPRE_Int     *row_indexes,
                          const NALU_HYPRE_BigInt  *cols,
                          const NALU_HYPRE_Complex *values )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (nrows == 0)
   {
      return nalu_hypre_error_flag;
   }

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (nrows < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   /*
   if (!ncols)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   */

   if (!rows)
   {
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }

   if (!cols)
   {
      nalu_hypre_error_in_arg(6);
      return nalu_hypre_error_flag;
   }

   if (!values)
   {
      nalu_hypre_error_in_arg(7);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixObjectType(ijmatrix) != NALU_HYPRE_PARCSR )
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_IJMatrixMemoryLocation(matrix) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_IJMatrixSetAddValuesParCSRDevice(ijmatrix, nrows, ncols, rows, row_indexes, cols, values,
                                             "set");
   }
   else
#endif
   {
      NALU_HYPRE_Int *row_indexes_tmp = (NALU_HYPRE_Int *) row_indexes;
      NALU_HYPRE_Int *ncols_tmp = ncols;

      if (!ncols_tmp)
      {
         NALU_HYPRE_Int i;
         ncols_tmp = nalu_hypre_TAlloc(NALU_HYPRE_Int, nrows, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < nrows; i++)
         {
            ncols_tmp[i] = 1;
         }
      }

      if (!row_indexes)
      {
         row_indexes_tmp = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_PrefixSumInt(nrows, ncols_tmp, row_indexes_tmp);
      }

      if (nalu_hypre_IJMatrixOMPFlag(ijmatrix))
      {
         nalu_hypre_IJMatrixSetValuesOMPParCSR(ijmatrix, nrows, ncols_tmp, rows, row_indexes_tmp, cols, values);
      }
      else
      {
         nalu_hypre_IJMatrixSetValuesParCSR(ijmatrix, nrows, ncols_tmp, rows, row_indexes_tmp, cols, values);
      }

      if (!ncols)
      {
         nalu_hypre_TFree(ncols_tmp, NALU_HYPRE_MEMORY_HOST);
      }

      if (!row_indexes)
      {
         nalu_hypre_TFree(row_indexes_tmp, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
NALU_HYPRE_IJMatrixSetConstantValues( NALU_HYPRE_IJMatrix matrix, NALU_HYPRE_Complex value)
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixObjectType(ijmatrix) == NALU_HYPRE_PARCSR )
   {
      return ( nalu_hypre_IJMatrixSetConstantValuesParCSR( ijmatrix, value));
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixAddToValues( NALU_HYPRE_IJMatrix       matrix,
                           NALU_HYPRE_Int            nrows,
                           NALU_HYPRE_Int           *ncols,
                           const NALU_HYPRE_BigInt  *rows,
                           const NALU_HYPRE_BigInt  *cols,
                           const NALU_HYPRE_Complex *values )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (nrows == 0)
   {
      return nalu_hypre_error_flag;
   }

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (nrows < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   /*
   if (!ncols)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   */

   if (!rows)
   {
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }

   if (!cols)
   {
      nalu_hypre_error_in_arg(5);
      return nalu_hypre_error_flag;
   }

   if (!values)
   {
      nalu_hypre_error_in_arg(6);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixObjectType(ijmatrix) != NALU_HYPRE_PARCSR )
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   NALU_HYPRE_IJMatrixAddToValues2(matrix, nrows, ncols, rows, NULL, cols, values);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixAddToValues2( NALU_HYPRE_IJMatrix       matrix,
                            NALU_HYPRE_Int            nrows,
                            NALU_HYPRE_Int           *ncols,
                            const NALU_HYPRE_BigInt  *rows,
                            const NALU_HYPRE_Int     *row_indexes,
                            const NALU_HYPRE_BigInt  *cols,
                            const NALU_HYPRE_Complex *values )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (nrows == 0)
   {
      return nalu_hypre_error_flag;
   }

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (nrows < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   /*
   if (!ncols)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }
   */

   if (!rows)
   {
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }

   if (!cols)
   {
      nalu_hypre_error_in_arg(6);
      return nalu_hypre_error_flag;
   }

   if (!values)
   {
      nalu_hypre_error_in_arg(7);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixObjectType(ijmatrix) != NALU_HYPRE_PARCSR )
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_IJMatrixMemoryLocation(matrix) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_IJMatrixSetAddValuesParCSRDevice(ijmatrix, nrows, ncols, rows, row_indexes, cols, values,
                                             "add");
   }
   else
#endif
   {
      NALU_HYPRE_Int *row_indexes_tmp = (NALU_HYPRE_Int *) row_indexes;
      NALU_HYPRE_Int *ncols_tmp = ncols;

      if (!ncols_tmp)
      {
         NALU_HYPRE_Int i;
         ncols_tmp = nalu_hypre_TAlloc(NALU_HYPRE_Int, nrows, NALU_HYPRE_MEMORY_HOST);
         for (i = 0; i < nrows; i++)
         {
            ncols_tmp[i] = 1;
         }
      }

      if (!row_indexes)
      {
         row_indexes_tmp = nalu_hypre_CTAlloc(NALU_HYPRE_Int, nrows, NALU_HYPRE_MEMORY_HOST);
         nalu_hypre_PrefixSumInt(nrows, ncols_tmp, row_indexes_tmp);
      }

      if (nalu_hypre_IJMatrixOMPFlag(ijmatrix))
      {
         nalu_hypre_IJMatrixAddToValuesOMPParCSR(ijmatrix, nrows, ncols_tmp, rows, row_indexes_tmp, cols, values);
      }
      else
      {
         nalu_hypre_IJMatrixAddToValuesParCSR(ijmatrix, nrows, ncols_tmp, rows, row_indexes_tmp, cols, values);
      }

      if (!ncols)
      {
         nalu_hypre_TFree(ncols_tmp, NALU_HYPRE_MEMORY_HOST);
      }

      if (!row_indexes)
      {
         nalu_hypre_TFree(row_indexes_tmp, NALU_HYPRE_MEMORY_HOST);
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixAssemble( NALU_HYPRE_IJMatrix matrix )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixObjectType(ijmatrix) == NALU_HYPRE_PARCSR )
   {
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP) || defined(NALU_HYPRE_USING_SYCL)
      NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_IJMatrixMemoryLocation(matrix) );

      if (exec == NALU_HYPRE_EXEC_DEVICE)
      {
         return ( nalu_hypre_IJMatrixAssembleParCSRDevice( ijmatrix ) );
      }
      else
#endif
      {
         return ( nalu_hypre_IJMatrixAssembleParCSR( ijmatrix ) );
      }
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixGetRowCounts( NALU_HYPRE_IJMatrix matrix,
                            NALU_HYPRE_Int      nrows,
                            NALU_HYPRE_BigInt  *rows,
                            NALU_HYPRE_Int     *ncols )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (nrows == 0)
   {
      return nalu_hypre_error_flag;
   }

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (nrows < 0)
   {
      nalu_hypre_error_in_arg(2);
      return nalu_hypre_error_flag;
   }

   if (!rows)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }

   if (!ncols)
   {
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixObjectType(ijmatrix) == NALU_HYPRE_PARCSR )
   {
      nalu_hypre_IJMatrixGetRowCountsParCSR( ijmatrix, nrows, rows, ncols );
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixGetValues( NALU_HYPRE_IJMatrix matrix,
                         NALU_HYPRE_Int      nrows,
                         NALU_HYPRE_Int     *ncols,
                         NALU_HYPRE_BigInt  *rows,
                         NALU_HYPRE_BigInt  *cols,
                         NALU_HYPRE_Complex *values )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (nrows == 0)
   {
      return nalu_hypre_error_flag;
   }

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (!ncols)
   {
      nalu_hypre_error_in_arg(3);
      return nalu_hypre_error_flag;
   }

   if (!rows)
   {
      nalu_hypre_error_in_arg(4);
      return nalu_hypre_error_flag;
   }

   if (!cols)
   {
      nalu_hypre_error_in_arg(5);
      return nalu_hypre_error_flag;
   }

   if (!values)
   {
      nalu_hypre_error_in_arg(6);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixObjectType(ijmatrix) == NALU_HYPRE_PARCSR )
   {
      nalu_hypre_IJMatrixGetValuesParCSR( ijmatrix, nrows, ncols,
                                     rows, cols, values );
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;

}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixSetObjectType( NALU_HYPRE_IJMatrix matrix,
                             NALU_HYPRE_Int      type )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_IJMatrixObjectType(ijmatrix) = type;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixGetObjectType( NALU_HYPRE_IJMatrix  matrix,
                             NALU_HYPRE_Int      *type )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *type = nalu_hypre_IJMatrixObjectType(ijmatrix);
   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixGetLocalRange( NALU_HYPRE_IJMatrix  matrix,
                             NALU_HYPRE_BigInt   *ilower,
                             NALU_HYPRE_BigInt   *iupper,
                             NALU_HYPRE_BigInt   *jlower,
                             NALU_HYPRE_BigInt   *jupper )
{
   nalu_hypre_IJMatrix  *ijmatrix = (nalu_hypre_IJMatrix *) matrix;
   NALU_HYPRE_BigInt    *row_partitioning;
   NALU_HYPRE_BigInt    *col_partitioning;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   row_partitioning = nalu_hypre_IJMatrixRowPartitioning(ijmatrix);
   col_partitioning = nalu_hypre_IJMatrixColPartitioning(ijmatrix);

   *ilower = row_partitioning[0];
   *iupper = row_partitioning[1] - 1;
   *jlower = col_partitioning[0];
   *jupper = col_partitioning[1] - 1;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
   Returns a pointer to an underlying ijmatrix type used to implement IJMatrix.
   Assumes that the implementation has an underlying matrix, so it would not
   work with a direct implementation of IJMatrix.

   @return integer error code
   @param IJMatrix [IN]
   The ijmatrix to be pointed to.
*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixGetObject( NALU_HYPRE_IJMatrix   matrix,
                         void           **object )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   *object = nalu_hypre_IJMatrixObject( ijmatrix );

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixSetRowSizes( NALU_HYPRE_IJMatrix   matrix,
                           const NALU_HYPRE_Int *sizes )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixObjectType(ijmatrix) == NALU_HYPRE_PARCSR )
   {
      return ( nalu_hypre_IJMatrixSetRowSizesParCSR( ijmatrix, sizes ) );
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixSetDiagOffdSizes( NALU_HYPRE_IJMatrix   matrix,
                                const NALU_HYPRE_Int *diag_sizes,
                                const NALU_HYPRE_Int *offdiag_sizes )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixObjectType(ijmatrix) == NALU_HYPRE_PARCSR )
   {
      nalu_hypre_IJMatrixSetDiagOffdSizesParCSR( ijmatrix, diag_sizes, offdiag_sizes );
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }
   return nalu_hypre_error_flag;

}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixSetMaxOffProcElmts( NALU_HYPRE_IJMatrix matrix,
                                  NALU_HYPRE_Int      max_off_proc_elmts)
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if ( nalu_hypre_IJMatrixObjectType(ijmatrix) == NALU_HYPRE_PARCSR )
   {
      return ( nalu_hypre_IJMatrixSetMaxOffProcElmtsParCSR(ijmatrix,
                                                      max_off_proc_elmts) );
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixRead
 * create IJMatrix on host memory
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixRead( const char     *filename,
                    MPI_Comm        comm,
                    NALU_HYPRE_Int       type,
                    NALU_HYPRE_IJMatrix *matrix_ptr )
{
   nalu_hypre_IJMatrixRead(filename, comm, type, matrix_ptr, 0);

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixReadMM( const char     *filename,
                      MPI_Comm        comm,
                      NALU_HYPRE_Int       type,
                      NALU_HYPRE_IJMatrix *matrix_ptr )
{
   nalu_hypre_IJMatrixRead(filename, comm, type, matrix_ptr, 1);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixPrint( NALU_HYPRE_IJMatrix  matrix,
                     const char     *filename )
{
   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if ( (nalu_hypre_IJMatrixObjectType(matrix) != NALU_HYPRE_PARCSR) )
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   void *object;
   NALU_HYPRE_IJMatrixGetObject(matrix, &object);
   nalu_hypre_ParCSRMatrix *par_csr = (nalu_hypre_ParCSRMatrix*) object;

   nalu_hypre_ParCSRMatrixPrintIJ(par_csr, 0, 0, filename);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixSetOMPFlag
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixSetOMPFlag( NALU_HYPRE_IJMatrix matrix,
                          NALU_HYPRE_Int      omp_flag )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   nalu_hypre_IJMatrixOMPFlag(ijmatrix) = omp_flag;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixTranspose
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixTranspose( NALU_HYPRE_IJMatrix  matrix_A,
                         NALU_HYPRE_IJMatrix *matrix_AT )
{
   nalu_hypre_IJMatrix   *ij_A = (nalu_hypre_IJMatrix *) matrix_A;
   nalu_hypre_IJMatrix   *ij_AT;
   NALU_HYPRE_Int         i;

   if (!ij_A)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   ij_AT = nalu_hypre_CTAlloc(nalu_hypre_IJMatrix, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_IJMatrixComm(ij_AT)           = nalu_hypre_IJMatrixComm(ij_A);
   nalu_hypre_IJMatrixObject(ij_AT)         = NULL;
   nalu_hypre_IJMatrixTranslator(ij_AT)     = NULL;
   nalu_hypre_IJMatrixAssumedPart(ij_AT)    = NULL;
   nalu_hypre_IJMatrixObjectType(ij_AT)     = nalu_hypre_IJMatrixObjectType(ij_A);
   nalu_hypre_IJMatrixAssembleFlag(ij_AT)   = 1;
   nalu_hypre_IJMatrixPrintLevel(ij_AT)     = nalu_hypre_IJMatrixPrintLevel(ij_A);
   nalu_hypre_IJMatrixGlobalFirstRow(ij_AT) = nalu_hypre_IJMatrixGlobalFirstCol(ij_A);
   nalu_hypre_IJMatrixGlobalFirstCol(ij_AT) = nalu_hypre_IJMatrixGlobalFirstRow(ij_A);
   nalu_hypre_IJMatrixGlobalNumRows(ij_AT)  = nalu_hypre_IJMatrixGlobalNumCols(ij_A);
   nalu_hypre_IJMatrixGlobalNumCols(ij_AT)  = nalu_hypre_IJMatrixGlobalNumRows(ij_A);

   for (i = 0; i < 2; i++)
   {
      nalu_hypre_IJMatrixRowPartitioning(ij_AT)[i] = nalu_hypre_IJMatrixColPartitioning(ij_A)[i];
      nalu_hypre_IJMatrixColPartitioning(ij_AT)[i] = nalu_hypre_IJMatrixRowPartitioning(ij_A)[i];
   }

   if (nalu_hypre_IJMatrixObjectType(ij_A) == NALU_HYPRE_PARCSR)
   {
      nalu_hypre_IJMatrixTransposeParCSR(ij_A, ij_AT);
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   *matrix_AT = (NALU_HYPRE_IJMatrix) ij_AT;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixNorm
 *
 *  TODO: Add other norms
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixNorm( NALU_HYPRE_IJMatrix  matrix,
                    NALU_HYPRE_Real     *norm )
{
   nalu_hypre_IJMatrix *ijmatrix = (nalu_hypre_IJMatrix *) matrix;

   if (!ijmatrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   if (nalu_hypre_IJMatrixObjectType(ijmatrix) == NALU_HYPRE_PARCSR)
   {
      nalu_hypre_IJMatrixNormParCSR(ijmatrix, norm);
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_IJMatrixAdd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_IJMatrixAdd( NALU_HYPRE_Complex    alpha,
                   NALU_HYPRE_IJMatrix   matrix_A,
                   NALU_HYPRE_Complex    beta,
                   NALU_HYPRE_IJMatrix   matrix_B,
                   NALU_HYPRE_IJMatrix  *matrix_C )
{
   nalu_hypre_IJMatrix   *ij_A = (nalu_hypre_IJMatrix *) matrix_A;
   nalu_hypre_IJMatrix   *ij_B = (nalu_hypre_IJMatrix *) matrix_B;
   nalu_hypre_IJMatrix   *ij_C;

   NALU_HYPRE_BigInt     *row_partitioning_A;
   NALU_HYPRE_BigInt     *col_partitioning_A;
   NALU_HYPRE_BigInt     *row_partitioning_B;
   NALU_HYPRE_BigInt     *col_partitioning_B;
   NALU_HYPRE_Int         i;

   if (!ij_A)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /* Check if A and B have the same row/col partitionings */
   row_partitioning_A = nalu_hypre_IJMatrixRowPartitioning(ij_A);
   row_partitioning_B = nalu_hypre_IJMatrixRowPartitioning(ij_B);
   col_partitioning_A = nalu_hypre_IJMatrixColPartitioning(ij_A);
   col_partitioning_B = nalu_hypre_IJMatrixColPartitioning(ij_B);
   for (i = 0; i < 2; i++)
   {
      if (row_partitioning_A[i] != row_partitioning_B[i])
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                           "Input matrices must have same row partitioning!");
         return nalu_hypre_error_flag;
      }

      if (col_partitioning_A[i] != col_partitioning_B[i])
      {
         nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                           "Input matrices must have same col partitioning!");
         return nalu_hypre_error_flag;
      }
   }

   ij_C = nalu_hypre_CTAlloc(nalu_hypre_IJMatrix, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_IJMatrixComm(ij_C)            = nalu_hypre_IJMatrixComm(ij_A);
   nalu_hypre_IJMatrixObject(ij_C)          = NULL;
   nalu_hypre_IJMatrixTranslator(ij_C)      = NULL;
   nalu_hypre_IJMatrixAssumedPart(ij_C)     = NULL;
   nalu_hypre_IJMatrixObjectType(ij_C)      = nalu_hypre_IJMatrixObjectType(ij_A);
   nalu_hypre_IJMatrixAssembleFlag(ij_C)    = 1;
   nalu_hypre_IJMatrixPrintLevel(ij_C)      = nalu_hypre_IJMatrixPrintLevel(ij_A);

   /* Copy row/col partitioning of A to C */
   for (i = 0; i < 2; i++)
   {
      nalu_hypre_IJMatrixRowPartitioning(ij_C)[i] = row_partitioning_A[i];
      nalu_hypre_IJMatrixColPartitioning(ij_C)[i] = col_partitioning_A[i];
   }

   if (nalu_hypre_IJMatrixObjectType(ij_A) == NALU_HYPRE_PARCSR)
   {
      nalu_hypre_IJMatrixAddParCSR(alpha, ij_A, beta, ij_B, ij_C);
   }
   else
   {
      nalu_hypre_error_in_arg(1);
   }

   *matrix_C = (NALU_HYPRE_IJMatrix) ij_C;

   return nalu_hypre_error_flag;
}
