/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"

#ifdef NALU_HYPRE_PROFILE
NALU_HYPRE_Real hypre_profile_times[NALU_HYPRE_TIMER_ID_COUNT] = { 0 };
#endif

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixCreate( NALU_HYPRE_Int num_rows,
                       NALU_HYPRE_Int num_cols,
                       NALU_HYPRE_Int num_nonzeros )
{
   hypre_CSRMatrix  *matrix;

   matrix = hypre_CTAlloc(hypre_CSRMatrix, 1, NALU_HYPRE_MEMORY_HOST);

   hypre_CSRMatrixData(matrix)           = NULL;
   hypre_CSRMatrixI(matrix)              = NULL;
   hypre_CSRMatrixJ(matrix)              = NULL;
   hypre_CSRMatrixBigJ(matrix)           = NULL;
   hypre_CSRMatrixRownnz(matrix)         = NULL;
   hypre_CSRMatrixNumRows(matrix)        = num_rows;
   hypre_CSRMatrixNumRownnz(matrix)      = num_rows;
   hypre_CSRMatrixNumCols(matrix)        = num_cols;
   hypre_CSRMatrixNumNonzeros(matrix)    = num_nonzeros;
   hypre_CSRMatrixMemoryLocation(matrix) = hypre_HandleMemoryLocation(hypre_handle());

   /* set defaults */
   hypre_CSRMatrixOwnsData(matrix)       = 1;

#if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) || defined(NALU_HYPRE_USING_ONEMKLSPARSE)
   hypre_CSRMatrixSortedJ(matrix)        = NULL;
   hypre_CSRMatrixSortedData(matrix)     = NULL;
   hypre_CSRMatrixCsrsvData(matrix)      = NULL;
#endif

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_CSRMatrixDestroy( hypre_CSRMatrix *matrix )
{
   if (matrix)
   {
      NALU_HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(matrix);

      hypre_TFree(hypre_CSRMatrixI(matrix),      memory_location);
      hypre_TFree(hypre_CSRMatrixRownnz(matrix), memory_location);

      if ( hypre_CSRMatrixOwnsData(matrix) )
      {
         hypre_TFree(hypre_CSRMatrixData(matrix), memory_location);
         hypre_TFree(hypre_CSRMatrixJ(matrix),    memory_location);
         /* RL: TODO There might be cases BigJ cannot be freed FIXME
          * Not so clear how to do it */
         hypre_TFree(hypre_CSRMatrixBigJ(matrix), memory_location);
      }

#if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) || defined(NALU_HYPRE_USING_ONEMKLSPARSE)
      hypre_TFree(hypre_CSRMatrixSortedData(matrix), memory_location);
      hypre_TFree(hypre_CSRMatrixSortedJ(matrix), memory_location);
      hypre_CsrsvDataDestroy(hypre_CSRMatrixCsrsvData(matrix));
      hypre_GpuMatDataDestroy(hypre_CSRMatrixGPUMatData(matrix));
#endif

      hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_CSRMatrixInitialize_v2( hypre_CSRMatrix      *matrix,
                              NALU_HYPRE_Int             bigInit,
                              NALU_HYPRE_MemoryLocation  memory_location )
{
   NALU_HYPRE_Int  num_rows     = hypre_CSRMatrixNumRows(matrix);
   NALU_HYPRE_Int  num_nonzeros = hypre_CSRMatrixNumNonzeros(matrix);
   /* NALU_HYPRE_Int  num_rownnz = hypre_CSRMatrixNumRownnz(matrix); */

   NALU_HYPRE_Int ierr = 0;

   hypre_CSRMatrixMemoryLocation(matrix) = memory_location;

   /* Caveat: for pre-existing i, j, data, their memory location must be guaranteed to be consistent with `memory_location'
    * Otherwise, mismatches will exist and problems will be encountered when being used, and freed */

   if ( !hypre_CSRMatrixData(matrix) && num_nonzeros )
   {
      hypre_CSRMatrixData(matrix) = hypre_CTAlloc(NALU_HYPRE_Complex, num_nonzeros, memory_location);
   }
   /*
   else
   {
     //if (PointerAttributes(hypre_CSRMatrixData(matrix))==NALU_HYPRE_HOST_POINTER) printf("MATREIX INITIAL WITH JHOST DATA\n");
   }
   */

   if ( !hypre_CSRMatrixI(matrix) )
   {
      hypre_CSRMatrixI(matrix) = hypre_CTAlloc(NALU_HYPRE_Int, num_rows + 1, memory_location);
   }

   /*
   if (!hypre_CSRMatrixRownnz(matrix))
   {
      hypre_CSRMatrixRownnz(matrix) = hypre_CTAlloc(NALU_HYPRE_Int,  num_rownnz, memory_location);
   }
   */

   if (bigInit)
   {
      if ( !hypre_CSRMatrixBigJ(matrix) && num_nonzeros )
      {
         hypre_CSRMatrixBigJ(matrix) = hypre_CTAlloc(NALU_HYPRE_BigInt, num_nonzeros, memory_location);
      }
   }
   else
   {
      if ( !hypre_CSRMatrixJ(matrix) && num_nonzeros )
      {
         hypre_CSRMatrixJ(matrix) = hypre_CTAlloc(NALU_HYPRE_Int, num_nonzeros, memory_location);
      }
   }

   return ierr;
}

NALU_HYPRE_Int
hypre_CSRMatrixInitialize( hypre_CSRMatrix *matrix )
{
   NALU_HYPRE_Int ierr;

   ierr = hypre_CSRMatrixInitialize_v2( matrix, 0, hypre_CSRMatrixMemoryLocation(matrix) );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixResize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_CSRMatrixResize( hypre_CSRMatrix *matrix, NALU_HYPRE_Int new_num_rows, NALU_HYPRE_Int new_num_cols,
                       NALU_HYPRE_Int new_num_nonzeros )
{
   NALU_HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(matrix);
   NALU_HYPRE_Int old_num_nonzeros = hypre_CSRMatrixNumNonzeros(matrix);
   NALU_HYPRE_Int old_num_rows = hypre_CSRMatrixNumRows(matrix);

   if (!hypre_CSRMatrixOwnsData(matrix))
   {
      hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Error: called hypre_CSRMatrixResize on a matrix that doesn't own the data\n");
      return 1;
   }

   hypre_CSRMatrixNumCols(matrix) = new_num_cols;

   if (new_num_nonzeros != hypre_CSRMatrixNumNonzeros(matrix))
   {
      hypre_CSRMatrixNumNonzeros(matrix) = new_num_nonzeros;

      if (!hypre_CSRMatrixData(matrix))
      {
         hypre_CSRMatrixData(matrix) = hypre_CTAlloc(NALU_HYPRE_Complex, new_num_nonzeros, memory_location);
      }
      else
      {
         hypre_CSRMatrixData(matrix) = hypre_TReAlloc_v2(hypre_CSRMatrixData(matrix), NALU_HYPRE_Complex,
                                                         old_num_nonzeros, NALU_HYPRE_Complex, new_num_nonzeros, memory_location);
      }

      if (!hypre_CSRMatrixJ(matrix))
      {
         hypre_CSRMatrixJ(matrix) = hypre_CTAlloc(NALU_HYPRE_Int, new_num_nonzeros, memory_location);
      }
      else
      {
         hypre_CSRMatrixJ(matrix) = hypre_TReAlloc_v2(hypre_CSRMatrixJ(matrix), NALU_HYPRE_Int, old_num_nonzeros,
                                                      NALU_HYPRE_Int, new_num_nonzeros, memory_location);
      }
   }

   if (new_num_rows != hypre_CSRMatrixNumRows(matrix))
   {
      hypre_CSRMatrixNumRows(matrix) = new_num_rows;

      if (!hypre_CSRMatrixI(matrix))
      {
         hypre_CSRMatrixI(matrix) = hypre_CTAlloc(NALU_HYPRE_Int, new_num_rows + 1, memory_location);
      }
      else
      {
         hypre_CSRMatrixI(matrix) = hypre_TReAlloc_v2(hypre_CSRMatrixI(matrix), NALU_HYPRE_Int, old_num_rows + 1,
                                                      NALU_HYPRE_Int, new_num_rows + 1, memory_location);
      }
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixBigInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_CSRMatrixBigInitialize( hypre_CSRMatrix *matrix )
{
   NALU_HYPRE_Int ierr;

   ierr = hypre_CSRMatrixInitialize_v2( matrix, 1, hypre_CSRMatrixMemoryLocation(matrix) );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixBigJtoJ
 * RL: TODO GPU impl.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_CSRMatrixBigJtoJ( hypre_CSRMatrix *matrix )
{
   NALU_HYPRE_Int     num_nonzeros = hypre_CSRMatrixNumNonzeros(matrix);
   NALU_HYPRE_BigInt *matrix_big_j = hypre_CSRMatrixBigJ(matrix);
   NALU_HYPRE_Int    *matrix_j = NULL;

   if (num_nonzeros && matrix_big_j)
   {
#if defined(NALU_HYPRE_MIXEDINT) || defined(NALU_HYPRE_BIGINT)
      NALU_HYPRE_Int i;
      matrix_j = hypre_TAlloc(NALU_HYPRE_Int, num_nonzeros, hypre_CSRMatrixMemoryLocation(matrix));
      for (i = 0; i < num_nonzeros; i++)
      {
         matrix_j[i] = (NALU_HYPRE_Int) matrix_big_j[i];
      }
      hypre_TFree(matrix_big_j, hypre_CSRMatrixMemoryLocation(matrix));
#else
      hypre_assert(sizeof(NALU_HYPRE_Int) == sizeof(NALU_HYPRE_BigInt));
      matrix_j = (NALU_HYPRE_Int *) matrix_big_j;
#endif
      hypre_CSRMatrixJ(matrix) = matrix_j;
      hypre_CSRMatrixBigJ(matrix) = NULL;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixJtoBigJ
 * RL: TODO GPU impl.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_CSRMatrixJtoBigJ( hypre_CSRMatrix *matrix )
{
   NALU_HYPRE_Int     num_nonzeros = hypre_CSRMatrixNumNonzeros(matrix);
   NALU_HYPRE_Int    *matrix_j = hypre_CSRMatrixJ(matrix);
   NALU_HYPRE_BigInt *matrix_big_j = NULL;

   if (num_nonzeros && matrix_j)
   {
#if defined(NALU_HYPRE_MIXEDINT) || defined(NALU_HYPRE_BIGINT)
      NALU_HYPRE_Int i;
      matrix_big_j = hypre_TAlloc(NALU_HYPRE_BigInt, num_nonzeros, hypre_CSRMatrixMemoryLocation(matrix));
      for (i = 0; i < num_nonzeros; i++)
      {
         matrix_big_j[i] = (NALU_HYPRE_BigInt) matrix_j[i];
      }
      hypre_TFree(matrix_j, hypre_CSRMatrixMemoryLocation(matrix));
#else
      hypre_assert(sizeof(NALU_HYPRE_Int) == sizeof(NALU_HYPRE_BigInt));
      matrix_big_j = (NALU_HYPRE_BigInt *) matrix_j;
#endif
      hypre_CSRMatrixBigJ(matrix) = matrix_big_j;
      hypre_CSRMatrixJ(matrix) = NULL;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_CSRMatrixSetDataOwner( hypre_CSRMatrix *matrix,
                             NALU_HYPRE_Int        owns_data )
{
   NALU_HYPRE_Int ierr = 0;

   hypre_CSRMatrixOwnsData(matrix) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSetPatternOnly
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
hypre_CSRMatrixSetPatternOnly( hypre_CSRMatrix *matrix,
                               NALU_HYPRE_Int        pattern_only )
{
   NALU_HYPRE_Int ierr = 0;

   hypre_CSRMatrixPatternOnly(matrix) = pattern_only;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSetRownnz
 *
 * function to set the substructure rownnz and num_rowsnnz inside the CSRMatrix
 * it needs the A_i substructure of CSRMatrix to find the nonzero rows.
 * It runs after the create CSR and when A_i is known..It does not check for
 * the existence of A_i or of the CSR matrix.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_CSRMatrixSetRownnzHost( hypre_CSRMatrix *matrix )
{
   NALU_HYPRE_Int   num_rows = hypre_CSRMatrixNumRows(matrix);
   NALU_HYPRE_Int  *A_i = hypre_CSRMatrixI(matrix);
   NALU_HYPRE_Int  *Arownnz = hypre_CSRMatrixRownnz(matrix);
   NALU_HYPRE_Int   i;
   NALU_HYPRE_Int   irownnz = 0;

   for (i = 0; i < num_rows; i++)
   {
      if ((A_i[i + 1] - A_i[i]) > 0)
      {
         irownnz++;
      }
   }

   hypre_CSRMatrixNumRownnz(matrix) = irownnz;

   NALU_HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(matrix);

   /* Free old rownnz pointer */
   hypre_TFree(Arownnz, memory_location);

   /* Set new rownnz pointer */
   if (irownnz == 0 || irownnz == num_rows)
   {
      hypre_CSRMatrixRownnz(matrix) = NULL;
   }
   else
   {
      Arownnz = hypre_CTAlloc(NALU_HYPRE_Int, irownnz, memory_location);
      irownnz = 0;
      for (i = 0; i < num_rows; i++)
      {
         if ((A_i[i + 1] - A_i[i]) > 0)
         {
            Arownnz[irownnz++] = i;
         }
      }
      hypre_CSRMatrixRownnz(matrix) = Arownnz;
   }

   return hypre_error_flag;
}

NALU_HYPRE_Int
hypre_CSRMatrixSetRownnz( hypre_CSRMatrix *matrix )
{
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_CSRMatrixMemoryLocation(matrix) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      // TODO RL: there's no need currently for having rownnz on GPUs
   }
   else
#endif
   {
      hypre_CSRMatrixSetRownnzHost(matrix);
   }

   return hypre_error_flag;
}

/* check if numnonzeros was properly set to be ia[nrow] */
NALU_HYPRE_Int
hypre_CSRMatrixCheckSetNumNonzeros( hypre_CSRMatrix *matrix )
{
   if (!matrix)
   {
      return 0;
   }

   NALU_HYPRE_Int nnz, ierr = 0;

   hypre_TMemcpy(&nnz, hypre_CSRMatrixI(matrix) + hypre_CSRMatrixNumRows(matrix),
                 NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, hypre_CSRMatrixMemoryLocation(matrix));

   if (hypre_CSRMatrixNumNonzeros(matrix) != nnz)
   {
      ierr = 1;
      hypre_printf("warning: CSR matrix nnz was not set properly (!= ia[nrow], %d %d)\n",
                   hypre_CSRMatrixNumNonzeros(matrix), nnz );
      hypre_assert(0);
      hypre_CSRMatrixNumNonzeros(matrix) = nnz;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixRead
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixRead( char *file_name )
{
   hypre_CSRMatrix  *matrix;

   FILE    *fp;

   NALU_HYPRE_Complex *matrix_data;
   NALU_HYPRE_Int     *matrix_i;
   NALU_HYPRE_Int     *matrix_j;
   NALU_HYPRE_Int      num_rows;
   NALU_HYPRE_Int      num_nonzeros;
   NALU_HYPRE_Int      max_col = 0;

   NALU_HYPRE_Int      file_base = 1;

   NALU_HYPRE_Int      j;

   /*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/
   fp = fopen(file_name, "r");

   hypre_fscanf(fp, "%d", &num_rows);

   matrix_i = hypre_CTAlloc(NALU_HYPRE_Int, num_rows + 1, NALU_HYPRE_MEMORY_HOST);
   for (j = 0; j < num_rows + 1; j++)
   {
      hypre_fscanf(fp, "%d", &matrix_i[j]);
      matrix_i[j] -= file_base;
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = hypre_CSRMatrixCreate(num_rows, num_rows, matrix_i[num_rows]);
   hypre_CSRMatrixI(matrix) = matrix_i;
   hypre_CSRMatrixInitialize_v2(matrix, 0, NALU_HYPRE_MEMORY_HOST);
   matrix_j = hypre_CSRMatrixJ(matrix);

   for (j = 0; j < num_nonzeros; j++)
   {
      hypre_fscanf(fp, "%d", &matrix_j[j]);
      matrix_j[j] -= file_base;

      if (matrix_j[j] > max_col)
      {
         max_col = matrix_j[j];
      }
   }

   matrix_data = hypre_CSRMatrixData(matrix);
   for (j = 0; j < matrix_i[num_rows]; j++)
   {
      hypre_fscanf(fp, "%le", &matrix_data[j]);
   }

   fclose(fp);

   hypre_CSRMatrixNumNonzeros(matrix) = num_nonzeros;
   hypre_CSRMatrixNumCols(matrix) = ++max_col;
   hypre_CSRMatrixSetRownnz(matrix);

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_CSRMatrixPrint( hypre_CSRMatrix *matrix,
                      const char      *file_name )
{
   FILE    *fp;

   NALU_HYPRE_Complex *matrix_data;
   NALU_HYPRE_Int     *matrix_i;
   NALU_HYPRE_Int     *matrix_j;
   NALU_HYPRE_BigInt  *matrix_bigj;
   NALU_HYPRE_Int      num_rows;

   NALU_HYPRE_Int      file_base = 1;

   NALU_HYPRE_Int      j;

   NALU_HYPRE_Int      ierr = 0;

   /*----------------------------------------------------------
    * Print the matrix data
    *----------------------------------------------------------*/

   matrix_data = hypre_CSRMatrixData(matrix);
   matrix_i    = hypre_CSRMatrixI(matrix);
   matrix_j    = hypre_CSRMatrixJ(matrix);
   matrix_bigj = hypre_CSRMatrixBigJ(matrix);
   num_rows    = hypre_CSRMatrixNumRows(matrix);

   fp = fopen(file_name, "w");

   hypre_fprintf(fp, "%d\n", num_rows);

   for (j = 0; j <= num_rows; j++)
   {
      hypre_fprintf(fp, "%d\n", matrix_i[j] + file_base);
   }

   if (matrix_j)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
         hypre_fprintf(fp, "%d\n", matrix_j[j] + file_base);
      }
   }

   if (matrix_bigj)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
         hypre_fprintf(fp, "%d\n", matrix_bigj[j] + file_base);
      }
   }

   if (matrix_data)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
#ifdef NALU_HYPRE_COMPLEX
         hypre_fprintf(fp, "%.14e , %.14e\n",
                       hypre_creal(matrix_data[j]), hypre_cimag(matrix_data[j]));
#else
         hypre_fprintf(fp, "%.14e\n", matrix_data[j]);
#endif
      }
   }
   else
   {
      hypre_fprintf(fp, "Warning: No matrix data!\n");
   }

   fclose(fp);

   return ierr;
}

NALU_HYPRE_Int
hypre_CSRMatrixPrintMM( hypre_CSRMatrix *matrix,
                        NALU_HYPRE_Int        basei,
                        NALU_HYPRE_Int        basej,
                        NALU_HYPRE_Int        trans,
                        const char      *file_name )
{
   hypre_assert(hypre_CSRMatrixI(matrix)[hypre_CSRMatrixNumRows(matrix)] == hypre_CSRMatrixNumNonzeros(
                   matrix));

   FILE *fp = file_name ? fopen(file_name, "w") : stdout;

   if (!fp)
   {
      hypre_error_w_msg(1, "Cannot open output file");
      return hypre_error_flag;
   }

   const NALU_HYPRE_Complex *matrix_data = hypre_CSRMatrixData(matrix);
   const NALU_HYPRE_Int     *matrix_i    = hypre_CSRMatrixI(matrix);
   const NALU_HYPRE_Int     *matrix_j    = hypre_CSRMatrixJ(matrix);

   if (matrix_data)
   {
      hypre_fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
   }
   else
   {
      hypre_fprintf(fp, "%%%%MatrixMarket matrix coordinate pattern general\n");
   }

   hypre_fprintf(fp, "%d %d %d\n",
                 trans ? hypre_CSRMatrixNumCols(matrix) : hypre_CSRMatrixNumRows(matrix),
                 trans ? hypre_CSRMatrixNumRows(matrix) : hypre_CSRMatrixNumCols(matrix),
                 hypre_CSRMatrixNumNonzeros(matrix));

   NALU_HYPRE_Int i, j;

   for (i = 0; i < hypre_CSRMatrixNumRows(matrix); i++)
   {
      for (j = matrix_i[i]; j < matrix_i[i + 1]; j++)
      {
         const NALU_HYPRE_Int row = (trans ? matrix_j[j] : i) + basei;
         const NALU_HYPRE_Int col = (trans ? i : matrix_j[j]) + basej;
         if (matrix_data)
         {
            hypre_fprintf(fp, "%d %d %.15e\n", row, col, matrix_data[j]);
         }
         else
         {
            hypre_fprintf(fp, "%d %d\n", row, col);
         }
      }
   }

   if (file_name)
   {
      fclose(fp);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixPrintHB: print a CSRMatrix in Harwell-Boeing format
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_CSRMatrixPrintHB( hypre_CSRMatrix *matrix_input,
                        char            *file_name )
{
   FILE            *fp;
   hypre_CSRMatrix *matrix;
   NALU_HYPRE_Complex   *matrix_data;
   NALU_HYPRE_Int       *matrix_i;
   NALU_HYPRE_Int       *matrix_j;
   NALU_HYPRE_Int        num_rows;
   NALU_HYPRE_Int        file_base = 1;
   NALU_HYPRE_Int        j, totcrd, ptrcrd, indcrd, valcrd, rhscrd;
   NALU_HYPRE_Int        ierr = 0;

   /*----------------------------------------------------------
    * Print the matrix data
    *----------------------------------------------------------*/

   /* First transpose the input matrix, since HB is in CSC format */
   hypre_CSRMatrixTranspose(matrix_input, &matrix, 1);

   matrix_data = hypre_CSRMatrixData(matrix);
   matrix_i    = hypre_CSRMatrixI(matrix);
   matrix_j    = hypre_CSRMatrixJ(matrix);
   num_rows    = hypre_CSRMatrixNumRows(matrix);

   fp = fopen(file_name, "w");

   hypre_fprintf(fp, "%-70s  Key     \n", "Title");
   ptrcrd = num_rows;
   indcrd = matrix_i[num_rows];
   valcrd = matrix_i[num_rows];
   rhscrd = 0;
   totcrd = ptrcrd + indcrd + valcrd + rhscrd;
   hypre_fprintf (fp, "%14d%14d%14d%14d%14d\n",
                  totcrd, ptrcrd, indcrd, valcrd, rhscrd);
   hypre_fprintf (fp, "%-14s%14i%14i%14i%14i\n", "RUA",
                  num_rows, num_rows, valcrd, 0);
   hypre_fprintf (fp, "%-16s%-16s%-16s%26s\n", "(1I8)", "(1I8)", "(1E16.8)", "");

   for (j = 0; j <= num_rows; j++)
   {
      hypre_fprintf(fp, "%8d\n", matrix_i[j] + file_base);
   }

   for (j = 0; j < matrix_i[num_rows]; j++)
   {
      hypre_fprintf(fp, "%8d\n", matrix_j[j] + file_base);
   }

   if (matrix_data)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
#ifdef NALU_HYPRE_COMPLEX
         hypre_fprintf(fp, "%16.8e , %16.8e\n",
                       hypre_creal(matrix_data[j]), hypre_cimag(matrix_data[j]));
#else
         hypre_fprintf(fp, "%16.8e\n", matrix_data[j]);
#endif
      }
   }
   else
   {
      hypre_fprintf(fp, "Warning: No matrix data!\n");
   }

   fclose(fp);

   hypre_CSRMatrixDestroy(matrix);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixCopy: copy A to B,
 *
 * if copy_data = 0 only the structure of A is copied to B.
 * the routine does not check if the dimensions/sparsity of A and B match !!!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_CSRMatrixCopy( hypre_CSRMatrix *A, hypre_CSRMatrix *B, NALU_HYPRE_Int copy_data )
{
   NALU_HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int num_nonzeros = hypre_CSRMatrixNumNonzeros(A);

   NALU_HYPRE_Int     *A_i      = hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *A_j      = hypre_CSRMatrixJ(A);
   NALU_HYPRE_BigInt  *A_bigj   = hypre_CSRMatrixBigJ(A);
   NALU_HYPRE_Int     *A_rownnz = hypre_CSRMatrixRownnz(A);
   NALU_HYPRE_Complex *A_data;

   NALU_HYPRE_Int     *B_i      = hypre_CSRMatrixI(B);
   NALU_HYPRE_Int     *B_j      = hypre_CSRMatrixJ(B);
   NALU_HYPRE_BigInt  *B_bigj   = hypre_CSRMatrixBigJ(B);
   NALU_HYPRE_Int     *B_rownnz = hypre_CSRMatrixRownnz(B);
   NALU_HYPRE_Complex *B_data;

   NALU_HYPRE_MemoryLocation memory_location_A = hypre_CSRMatrixMemoryLocation(A);
   NALU_HYPRE_MemoryLocation memory_location_B = hypre_CSRMatrixMemoryLocation(B);

   hypre_TMemcpy(B_i, A_i, NALU_HYPRE_Int, num_rows + 1, memory_location_B, memory_location_A);

   if (A_rownnz)
   {
      if (!B_rownnz)
      {
         B_rownnz = hypre_TAlloc(NALU_HYPRE_Int,
                                 hypre_CSRMatrixNumRownnz(A),
                                 memory_location_B);
         hypre_CSRMatrixRownnz(B) = B_rownnz;
      }
      hypre_TMemcpy(B_rownnz, A_rownnz,
                    NALU_HYPRE_Int, hypre_CSRMatrixNumRownnz(A),
                    memory_location_B, memory_location_A);
   }
   hypre_CSRMatrixNumRownnz(B) = hypre_CSRMatrixNumRownnz(A);

   if (A_j && B_j)
   {
      hypre_TMemcpy(B_j, A_j, NALU_HYPRE_Int, num_nonzeros, memory_location_B, memory_location_A);
   }

   if (A_bigj && B_bigj)
   {
      hypre_TMemcpy(B_bigj, A_bigj, NALU_HYPRE_BigInt, num_nonzeros, memory_location_B, memory_location_A);
   }

   if (copy_data)
   {
      A_data = hypre_CSRMatrixData(A);
      B_data = hypre_CSRMatrixData(B);
      hypre_TMemcpy(B_data, A_data, NALU_HYPRE_Complex, num_nonzeros, memory_location_B, memory_location_A);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixClone_v2
 *
 * This function does the same job as hypre_CSRMatrixClone; however, here
 * the user can specify the memory location of the resulting matrix.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixClone_v2( hypre_CSRMatrix *A, NALU_HYPRE_Int copy_data,
                         NALU_HYPRE_MemoryLocation memory_location )
{
   NALU_HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int num_cols = hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int num_nonzeros = hypre_CSRMatrixNumNonzeros(A);

   hypre_CSRMatrix *B = hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);

   NALU_HYPRE_Int bigInit = hypre_CSRMatrixBigJ(A) != NULL;

   hypre_CSRMatrixInitialize_v2(B, bigInit, memory_location);

   hypre_CSRMatrixCopy(A, B, copy_data);

   return B;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixClone
 *
 * Creates and returns a new copy of the argument, A.
 * Performs a deep copy of information (no pointers are copied);
 * New arrays are created where necessary.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixClone( hypre_CSRMatrix *A, NALU_HYPRE_Int copy_data )
{
   return hypre_CSRMatrixClone_v2(A, copy_data, hypre_CSRMatrixMemoryLocation(A));
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixUnion
 * Creates and returns a matrix whose elements are the union of those of A and B.
 * Data is not computed, only structural information is created.
 * A and B must have the same numbers of rows.
 * Nothing is done about Rownnz.
 *
 * If col_map_offd_A and col_map_offd_B are zero, A and B are expected to have
 * the same column indexing.  Otherwise, col_map_offd_A, col_map_offd_B should
 * be the arrays of that name from two ParCSRMatrices of which A and B are the
 * offd blocks.
 *
 * The algorithm can be expected to have reasonable efficiency only for very
 * sparse matrices (many rows, few nonzeros per row).
 * The nonzeros of a computed row are NOT necessarily in any particular order.
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixUnion( hypre_CSRMatrix *A,
                      hypre_CSRMatrix *B,
                      NALU_HYPRE_BigInt *col_map_offd_A,
                      NALU_HYPRE_BigInt *col_map_offd_B,
                      NALU_HYPRE_BigInt **col_map_offd_C )
{
   NALU_HYPRE_Int num_rows = hypre_CSRMatrixNumRows( A );
   NALU_HYPRE_Int num_cols_A = hypre_CSRMatrixNumCols( A );
   NALU_HYPRE_Int num_cols_B = hypre_CSRMatrixNumCols( B );
   NALU_HYPRE_Int num_cols;
   NALU_HYPRE_Int num_nonzeros;
   NALU_HYPRE_Int *A_i = hypre_CSRMatrixI(A);
   NALU_HYPRE_Int *A_j = hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int *B_i = hypre_CSRMatrixI(B);
   NALU_HYPRE_Int *B_j = hypre_CSRMatrixJ(B);
   NALU_HYPRE_Int *C_i;
   NALU_HYPRE_Int *C_j;
   NALU_HYPRE_Int *jC = NULL;
   NALU_HYPRE_BigInt jBg, big_jA, big_jB;
   NALU_HYPRE_Int i, jA, jB;
   NALU_HYPRE_Int ma, mb, mc, ma_min, ma_max, match;
   hypre_CSRMatrix* C;

   NALU_HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(A);

   hypre_assert( num_rows == hypre_CSRMatrixNumRows(B) );

   if ( col_map_offd_B )
   {
      hypre_assert( col_map_offd_A );
   }

   if ( col_map_offd_A )
   {
      hypre_assert( col_map_offd_B );
   }

   /* ==== First, go through the columns of A and B to count the columns of C. */
   if ( col_map_offd_A == 0 )
   {
      /* The matrices are diagonal blocks.
         Normally num_cols_A==num_cols_B, col_starts is the same, etc.
      */
      num_cols = hypre_max( num_cols_A, num_cols_B );
   }
   else
   {
      /* The matrices are offdiagonal blocks. */
      jC = hypre_CTAlloc(NALU_HYPRE_Int, num_cols_B, NALU_HYPRE_MEMORY_HOST);
      num_cols = num_cols_A;  /* initialization; we'll compute the actual value */
      for ( jB = 0; jB < num_cols_B; ++jB )
      {
         match = 0;
         jBg = col_map_offd_B[jB];
         for ( ma = 0; ma < num_cols_A; ++ma )
         {
            if ( col_map_offd_A[ma] == jBg )
            {
               match = 1;
            }
         }
         if ( match == 0 )
         {
            jC[jB] = num_cols;
            ++num_cols;
         }
      }
   }

   /* ==== If we're working on a ParCSRMatrix's offd block,
      make and load col_map_offd_C */
   if ( col_map_offd_A )
   {
      *col_map_offd_C = hypre_CTAlloc( NALU_HYPRE_BigInt, num_cols, NALU_HYPRE_MEMORY_HOST);
      for ( jA = 0; jA < num_cols_A; ++jA )
      {
         (*col_map_offd_C)[jA] = col_map_offd_A[jA];
      }
      for ( jB = 0; jB < num_cols_B; ++jB )
      {
         match = 0;
         jBg = col_map_offd_B[jB];
         for ( ma = 0; ma < num_cols_A; ++ma )
         {
            if ( col_map_offd_A[ma] == jBg )
            {
               match = 1;
            }
         }
         if ( match == 0 )
         {
            (*col_map_offd_C)[ jC[jB] ] = jBg;
         }
      }
   }


   /* ==== The first run through A and B is to count the number of nonzero elements,
      without NALU_HYPRE_Complex-counting duplicates.  Then we can create C. */
   num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   for ( i = 0; i < num_rows; ++i )
   {
      ma_min = A_i[i];  ma_max = A_i[i + 1];
      for ( mb = B_i[i]; mb < B_i[i + 1]; ++mb )
      {
         jB = B_j[mb];
         if ( col_map_offd_B )
         {
            big_jB = col_map_offd_B[jB];
         }
         match = 0;
         for ( ma = ma_min; ma < ma_max; ++ma )
         {
            jA = A_j[ma];
            if ( col_map_offd_A )
            {
               big_jA = col_map_offd_A[jA];
            }
            if ( big_jB == big_jA )
            {
               match = 1;
               if ( ma == ma_min )
               {
                  ++ma_min;
               }
               break;
            }
         }
         if ( match == 0 )
         {
            ++num_nonzeros;
         }
      }
   }

   C = hypre_CSRMatrixCreate( num_rows, num_cols, num_nonzeros );
   hypre_CSRMatrixInitialize_v2( C, 0, memory_location );

   /* ==== The second run through A and B is to pick out the column numbers
      for each row, and put them in C. */
   C_i = hypre_CSRMatrixI(C);
   C_i[0] = 0;
   C_j = hypre_CSRMatrixJ(C);
   mc = 0;
   for ( i = 0; i < num_rows; ++i )
   {
      ma_min = A_i[i];
      ma_max = A_i[i + 1];
      for ( ma = ma_min; ma < ma_max; ++ma )
      {
         C_j[mc] = A_j[ma];
         ++mc;
      }
      for ( mb = B_i[i]; mb < B_i[i + 1]; ++mb )
      {
         jB = B_j[mb];
         if ( col_map_offd_B )
         {
            big_jB = col_map_offd_B[jB];
         }
         match = 0;
         for ( ma = ma_min; ma < ma_max; ++ma )
         {
            jA = A_j[ma];
            if ( col_map_offd_A )
            {
               big_jA = col_map_offd_A[jA];
            }
            if ( big_jB == big_jA )
            {
               match = 1;
               if ( ma == ma_min )
               {
                  ++ma_min;
               }
               break;
            }
         }
         if ( match == 0 )
         {
            if ( col_map_offd_A )
            {
               C_j[mc] = jC[ B_j[mb] ];
            }
            else
            {
               C_j[mc] = B_j[mb];
            }
            /* ... I don't know whether column indices are required to be in any
               particular order.  If so, we'll need to sort. */
            ++mc;
         }
      }
      C_i[i + 1] = mc;
   }

   hypre_assert( mc == num_nonzeros );

   hypre_TFree(jC, NALU_HYPRE_MEMORY_HOST);

   return C;
}

static NALU_HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionBoundary(hypre_CSRMatrix *A, NALU_HYPRE_Int idx)
{
   NALU_HYPRE_Int num_nonzerosA = hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int num_rowsA = hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int *A_i = hypre_CSRMatrixI(A);

   NALU_HYPRE_Int num_threads = hypre_NumActiveThreads();

   NALU_HYPRE_Int nonzeros_per_thread = (num_nonzerosA + num_threads - 1) / num_threads;

   if (idx <= 0)
   {
      return 0;
   }
   else if (idx >= num_threads)
   {
      return num_rowsA;
   }
   else
   {
      return (NALU_HYPRE_Int)(hypre_LowerBound(A_i, A_i + num_rowsA, nonzeros_per_thread * idx) - A_i);
   }
}

NALU_HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionBegin(hypre_CSRMatrix *A)
{
   return hypre_CSRMatrixGetLoadBalancedPartitionBoundary(A, hypre_GetThreadNum());
}

NALU_HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionEnd(hypre_CSRMatrix *A)
{
   return hypre_CSRMatrixGetLoadBalancedPartitionBoundary(A, hypre_GetThreadNum() + 1);
}

NALU_HYPRE_Int
hypre_CSRMatrixPrefetch( hypre_CSRMatrix *A, NALU_HYPRE_MemoryLocation memory_location )
{
   NALU_HYPRE_Int      ierr = 0;

#ifdef NALU_HYPRE_USING_UNIFIED_MEMORY
   if (hypre_CSRMatrixMemoryLocation(A) != NALU_HYPRE_MEMORY_DEVICE)
   {
      return 1;
   }

   NALU_HYPRE_Complex *data = hypre_CSRMatrixData(A);
   NALU_HYPRE_Int     *ia   = hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *ja   = hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int      nrow = hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int      nnzA = hypre_CSRMatrixNumNonzeros(A);

   hypre_MemPrefetch(data, sizeof(NALU_HYPRE_Complex)*nnzA, memory_location);
   hypre_MemPrefetch(ia,   sizeof(NALU_HYPRE_Int) * (nrow + 1), memory_location);
   hypre_MemPrefetch(ja,   sizeof(NALU_HYPRE_Int)*nnzA,     memory_location);
#endif

   return ierr;
}

#if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) || defined(NALU_HYPRE_USING_ONEMKLSPARSE)
hypre_GpuMatData *
hypre_CSRMatrixGetGPUMatData(hypre_CSRMatrix *matrix)
{
   if (!matrix)
   {
      return NULL;
   }

   if (!hypre_CSRMatrixGPUMatData(matrix))
   {
      hypre_CSRMatrixGPUMatData(matrix) = hypre_GpuMatDataCreate();
      hypre_GPUMatDataSetCSRData(hypre_CSRMatrixGPUMatData(matrix), matrix);
   }

   return hypre_CSRMatrixGPUMatData(matrix);
}
#endif
