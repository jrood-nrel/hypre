/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"

#ifdef NALU_HYPRE_PROFILE
NALU_HYPRE_Real nalu_hypre_profile_times[NALU_HYPRE_TIMER_ID_COUNT] = { 0 };
#endif

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix *
nalu_hypre_CSRMatrixCreate( NALU_HYPRE_Int num_rows,
                       NALU_HYPRE_Int num_cols,
                       NALU_HYPRE_Int num_nonzeros )
{
   nalu_hypre_CSRMatrix  *matrix;

   matrix = nalu_hypre_CTAlloc(nalu_hypre_CSRMatrix, 1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_CSRMatrixData(matrix)           = NULL;
   nalu_hypre_CSRMatrixI(matrix)              = NULL;
   nalu_hypre_CSRMatrixJ(matrix)              = NULL;
   nalu_hypre_CSRMatrixBigJ(matrix)           = NULL;
   nalu_hypre_CSRMatrixRownnz(matrix)         = NULL;
   nalu_hypre_CSRMatrixNumRows(matrix)        = num_rows;
   nalu_hypre_CSRMatrixNumRownnz(matrix)      = num_rows;
   nalu_hypre_CSRMatrixNumCols(matrix)        = num_cols;
   nalu_hypre_CSRMatrixNumNonzeros(matrix)    = num_nonzeros;
   nalu_hypre_CSRMatrixMemoryLocation(matrix) = nalu_hypre_HandleMemoryLocation(nalu_hypre_handle());

   /* set defaults */
   nalu_hypre_CSRMatrixOwnsData(matrix)       = 1;

#if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) || defined(NALU_HYPRE_USING_ONEMKLSPARSE)
   nalu_hypre_CSRMatrixSortedJ(matrix)        = NULL;
   nalu_hypre_CSRMatrixSortedData(matrix)     = NULL;
   nalu_hypre_CSRMatrixCsrsvData(matrix)      = NULL;
#endif

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixDestroy( nalu_hypre_CSRMatrix *matrix )
{
   if (matrix)
   {
      NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_CSRMatrixMemoryLocation(matrix);

      nalu_hypre_TFree(nalu_hypre_CSRMatrixI(matrix),      memory_location);
      nalu_hypre_TFree(nalu_hypre_CSRMatrixRownnz(matrix), memory_location);

      if ( nalu_hypre_CSRMatrixOwnsData(matrix) )
      {
         nalu_hypre_TFree(nalu_hypre_CSRMatrixData(matrix), memory_location);
         nalu_hypre_TFree(nalu_hypre_CSRMatrixJ(matrix),    memory_location);
         /* RL: TODO There might be cases BigJ cannot be freed FIXME
          * Not so clear how to do it */
         nalu_hypre_TFree(nalu_hypre_CSRMatrixBigJ(matrix), memory_location);
      }

#if defined(NALU_HYPRE_USING_CUSPARSE) || defined(NALU_HYPRE_USING_ROCSPARSE) || defined(NALU_HYPRE_USING_ONEMKLSPARSE)
      nalu_hypre_TFree(nalu_hypre_CSRMatrixSortedData(matrix), memory_location);
      nalu_hypre_TFree(nalu_hypre_CSRMatrixSortedJ(matrix), memory_location);
      nalu_hypre_CsrsvDataDestroy(nalu_hypre_CSRMatrixCsrsvData(matrix));
      nalu_hypre_GpuMatDataDestroy(nalu_hypre_CSRMatrixGPUMatData(matrix));
#endif

      nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixInitialize_v2( nalu_hypre_CSRMatrix      *matrix,
                              NALU_HYPRE_Int             bigInit,
                              NALU_HYPRE_MemoryLocation  memory_location )
{
   NALU_HYPRE_Int  num_rows     = nalu_hypre_CSRMatrixNumRows(matrix);
   NALU_HYPRE_Int  num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(matrix);
   /* NALU_HYPRE_Int  num_rownnz = nalu_hypre_CSRMatrixNumRownnz(matrix); */

   NALU_HYPRE_Int ierr = 0;

   nalu_hypre_CSRMatrixMemoryLocation(matrix) = memory_location;

   /* Caveat: for pre-existing i, j, data, their memory location must be guaranteed to be consistent with `memory_location'
    * Otherwise, mismatches will exist and problems will be encountered when being used, and freed */

   if ( !nalu_hypre_CSRMatrixData(matrix) && num_nonzeros )
   {
      nalu_hypre_CSRMatrixData(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, num_nonzeros, memory_location);
   }
   /*
   else
   {
     //if (PointerAttributes(nalu_hypre_CSRMatrixData(matrix))==NALU_HYPRE_HOST_POINTER) printf("MATREIX INITIAL WITH JHOST DATA\n");
   }
   */

   if ( !nalu_hypre_CSRMatrixI(matrix) )
   {
      nalu_hypre_CSRMatrixI(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows + 1, memory_location);
   }

   /*
   if (!nalu_hypre_CSRMatrixRownnz(matrix))
   {
      nalu_hypre_CSRMatrixRownnz(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_Int,  num_rownnz, memory_location);
   }
   */

   if (bigInit)
   {
      if ( !nalu_hypre_CSRMatrixBigJ(matrix) && num_nonzeros )
      {
         nalu_hypre_CSRMatrixBigJ(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_BigInt, num_nonzeros, memory_location);
      }
   }
   else
   {
      if ( !nalu_hypre_CSRMatrixJ(matrix) && num_nonzeros )
      {
         nalu_hypre_CSRMatrixJ(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_nonzeros, memory_location);
      }
   }

   return ierr;
}

NALU_HYPRE_Int
nalu_hypre_CSRMatrixInitialize( nalu_hypre_CSRMatrix *matrix )
{
   NALU_HYPRE_Int ierr;

   ierr = nalu_hypre_CSRMatrixInitialize_v2( matrix, 0, nalu_hypre_CSRMatrixMemoryLocation(matrix) );

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixResize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixResize( nalu_hypre_CSRMatrix *matrix, NALU_HYPRE_Int new_num_rows, NALU_HYPRE_Int new_num_cols,
                       NALU_HYPRE_Int new_num_nonzeros )
{
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_CSRMatrixMemoryLocation(matrix);
   NALU_HYPRE_Int old_num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(matrix);
   NALU_HYPRE_Int old_num_rows = nalu_hypre_CSRMatrixNumRows(matrix);

   if (!nalu_hypre_CSRMatrixOwnsData(matrix))
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC,
                        "Error: called nalu_hypre_CSRMatrixResize on a matrix that doesn't own the data\n");
      return 1;
   }

   nalu_hypre_CSRMatrixNumCols(matrix) = new_num_cols;

   if (new_num_nonzeros != nalu_hypre_CSRMatrixNumNonzeros(matrix))
   {
      nalu_hypre_CSRMatrixNumNonzeros(matrix) = new_num_nonzeros;

      if (!nalu_hypre_CSRMatrixData(matrix))
      {
         nalu_hypre_CSRMatrixData(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_Complex, new_num_nonzeros, memory_location);
      }
      else
      {
         nalu_hypre_CSRMatrixData(matrix) = nalu_hypre_TReAlloc_v2(nalu_hypre_CSRMatrixData(matrix), NALU_HYPRE_Complex,
                                                         old_num_nonzeros, NALU_HYPRE_Complex, new_num_nonzeros, memory_location);
      }

      if (!nalu_hypre_CSRMatrixJ(matrix))
      {
         nalu_hypre_CSRMatrixJ(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, new_num_nonzeros, memory_location);
      }
      else
      {
         nalu_hypre_CSRMatrixJ(matrix) = nalu_hypre_TReAlloc_v2(nalu_hypre_CSRMatrixJ(matrix), NALU_HYPRE_Int, old_num_nonzeros,
                                                      NALU_HYPRE_Int, new_num_nonzeros, memory_location);
      }
   }

   if (new_num_rows != nalu_hypre_CSRMatrixNumRows(matrix))
   {
      nalu_hypre_CSRMatrixNumRows(matrix) = new_num_rows;

      if (!nalu_hypre_CSRMatrixI(matrix))
      {
         nalu_hypre_CSRMatrixI(matrix) = nalu_hypre_CTAlloc(NALU_HYPRE_Int, new_num_rows + 1, memory_location);
      }
      else
      {
         nalu_hypre_CSRMatrixI(matrix) = nalu_hypre_TReAlloc_v2(nalu_hypre_CSRMatrixI(matrix), NALU_HYPRE_Int, old_num_rows + 1,
                                                      NALU_HYPRE_Int, new_num_rows + 1, memory_location);
      }
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixBigInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixBigInitialize( nalu_hypre_CSRMatrix *matrix )
{
   NALU_HYPRE_Int ierr;

   ierr = nalu_hypre_CSRMatrixInitialize_v2( matrix, 1, nalu_hypre_CSRMatrixMemoryLocation(matrix) );

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixBigJtoJ
 * RL: TODO GPU impl.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixBigJtoJ( nalu_hypre_CSRMatrix *matrix )
{
   NALU_HYPRE_Int     num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(matrix);
   NALU_HYPRE_BigInt *matrix_big_j = nalu_hypre_CSRMatrixBigJ(matrix);
   NALU_HYPRE_Int    *matrix_j = NULL;

   if (num_nonzeros && matrix_big_j)
   {
#if defined(NALU_HYPRE_MIXEDINT) || defined(NALU_HYPRE_BIGINT)
      NALU_HYPRE_Int i;
      matrix_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_nonzeros, nalu_hypre_CSRMatrixMemoryLocation(matrix));
      for (i = 0; i < num_nonzeros; i++)
      {
         matrix_j[i] = (NALU_HYPRE_Int) matrix_big_j[i];
      }
      nalu_hypre_TFree(matrix_big_j, nalu_hypre_CSRMatrixMemoryLocation(matrix));
#else
      nalu_hypre_assert(sizeof(NALU_HYPRE_Int) == sizeof(NALU_HYPRE_BigInt));
      matrix_j = (NALU_HYPRE_Int *) matrix_big_j;
#endif
      nalu_hypre_CSRMatrixJ(matrix) = matrix_j;
      nalu_hypre_CSRMatrixBigJ(matrix) = NULL;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixJtoBigJ
 * RL: TODO GPU impl.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixJtoBigJ( nalu_hypre_CSRMatrix *matrix )
{
   NALU_HYPRE_Int     num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(matrix);
   NALU_HYPRE_Int    *matrix_j = nalu_hypre_CSRMatrixJ(matrix);
   NALU_HYPRE_BigInt *matrix_big_j = NULL;

   if (num_nonzeros && matrix_j)
   {
#if defined(NALU_HYPRE_MIXEDINT) || defined(NALU_HYPRE_BIGINT)
      NALU_HYPRE_Int i;
      matrix_big_j = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_nonzeros, nalu_hypre_CSRMatrixMemoryLocation(matrix));
      for (i = 0; i < num_nonzeros; i++)
      {
         matrix_big_j[i] = (NALU_HYPRE_BigInt) matrix_j[i];
      }
      nalu_hypre_TFree(matrix_j, nalu_hypre_CSRMatrixMemoryLocation(matrix));
#else
      nalu_hypre_assert(sizeof(NALU_HYPRE_Int) == sizeof(NALU_HYPRE_BigInt));
      matrix_big_j = (NALU_HYPRE_BigInt *) matrix_j;
#endif
      nalu_hypre_CSRMatrixBigJ(matrix) = matrix_big_j;
      nalu_hypre_CSRMatrixJ(matrix) = NULL;
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixSetDataOwner( nalu_hypre_CSRMatrix *matrix,
                             NALU_HYPRE_Int        owns_data )
{
   NALU_HYPRE_Int ierr = 0;

   nalu_hypre_CSRMatrixOwnsData(matrix) = owns_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixSetPatternOnly
 *--------------------------------------------------------------------------*/
NALU_HYPRE_Int
nalu_hypre_CSRMatrixSetPatternOnly( nalu_hypre_CSRMatrix *matrix,
                               NALU_HYPRE_Int        pattern_only )
{
   NALU_HYPRE_Int ierr = 0;

   nalu_hypre_CSRMatrixPatternOnly(matrix) = pattern_only;

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixSetRownnz
 *
 * function to set the substructure rownnz and num_rowsnnz inside the CSRMatrix
 * it needs the A_i substructure of CSRMatrix to find the nonzero rows.
 * It runs after the create CSR and when A_i is known..It does not check for
 * the existence of A_i or of the CSR matrix.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixSetRownnzHost( nalu_hypre_CSRMatrix *matrix )
{
   NALU_HYPRE_Int   num_rows = nalu_hypre_CSRMatrixNumRows(matrix);
   NALU_HYPRE_Int  *A_i = nalu_hypre_CSRMatrixI(matrix);
   NALU_HYPRE_Int  *Arownnz = nalu_hypre_CSRMatrixRownnz(matrix);
   NALU_HYPRE_Int   i;
   NALU_HYPRE_Int   irownnz = 0;

   for (i = 0; i < num_rows; i++)
   {
      if ((A_i[i + 1] - A_i[i]) > 0)
      {
         irownnz++;
      }
   }

   nalu_hypre_CSRMatrixNumRownnz(matrix) = irownnz;

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_CSRMatrixMemoryLocation(matrix);

   /* Free old rownnz pointer */
   nalu_hypre_TFree(Arownnz, memory_location);

   /* Set new rownnz pointer */
   if (irownnz == 0 || irownnz == num_rows)
   {
      nalu_hypre_CSRMatrixRownnz(matrix) = NULL;
   }
   else
   {
      Arownnz = nalu_hypre_CTAlloc(NALU_HYPRE_Int, irownnz, memory_location);
      irownnz = 0;
      for (i = 0; i < num_rows; i++)
      {
         if ((A_i[i + 1] - A_i[i]) > 0)
         {
            Arownnz[irownnz++] = i;
         }
      }
      nalu_hypre_CSRMatrixRownnz(matrix) = Arownnz;
   }

   return nalu_hypre_error_flag;
}

NALU_HYPRE_Int
nalu_hypre_CSRMatrixSetRownnz( nalu_hypre_CSRMatrix *matrix )
{
#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_CSRMatrixMemoryLocation(matrix) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      // TODO RL: there's no need currently for having rownnz on GPUs
   }
   else
#endif
   {
      nalu_hypre_CSRMatrixSetRownnzHost(matrix);
   }

   return nalu_hypre_error_flag;
}

/* check if numnonzeros was properly set to be ia[nrow] */
NALU_HYPRE_Int
nalu_hypre_CSRMatrixCheckSetNumNonzeros( nalu_hypre_CSRMatrix *matrix )
{
   if (!matrix)
   {
      return 0;
   }

   NALU_HYPRE_Int nnz, ierr = 0;

   nalu_hypre_TMemcpy(&nnz, nalu_hypre_CSRMatrixI(matrix) + nalu_hypre_CSRMatrixNumRows(matrix),
                 NALU_HYPRE_Int, 1, NALU_HYPRE_MEMORY_HOST, nalu_hypre_CSRMatrixMemoryLocation(matrix));

   if (nalu_hypre_CSRMatrixNumNonzeros(matrix) != nnz)
   {
      ierr = 1;
      nalu_hypre_printf("warning: CSR matrix nnz was not set properly (!= ia[nrow], %d %d)\n",
                   nalu_hypre_CSRMatrixNumNonzeros(matrix), nnz );
      nalu_hypre_assert(0);
      nalu_hypre_CSRMatrixNumNonzeros(matrix) = nnz;
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixRead
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix *
nalu_hypre_CSRMatrixRead( char *file_name )
{
   nalu_hypre_CSRMatrix  *matrix;

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

   nalu_hypre_fscanf(fp, "%d", &num_rows);

   matrix_i = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_rows + 1, NALU_HYPRE_MEMORY_HOST);
   for (j = 0; j < num_rows + 1; j++)
   {
      nalu_hypre_fscanf(fp, "%d", &matrix_i[j]);
      matrix_i[j] -= file_base;
   }

   num_nonzeros = matrix_i[num_rows];

   matrix = nalu_hypre_CSRMatrixCreate(num_rows, num_rows, matrix_i[num_rows]);
   nalu_hypre_CSRMatrixI(matrix) = matrix_i;
   nalu_hypre_CSRMatrixInitialize_v2(matrix, 0, NALU_HYPRE_MEMORY_HOST);
   matrix_j = nalu_hypre_CSRMatrixJ(matrix);

   for (j = 0; j < num_nonzeros; j++)
   {
      nalu_hypre_fscanf(fp, "%d", &matrix_j[j]);
      matrix_j[j] -= file_base;

      if (matrix_j[j] > max_col)
      {
         max_col = matrix_j[j];
      }
   }

   matrix_data = nalu_hypre_CSRMatrixData(matrix);
   for (j = 0; j < matrix_i[num_rows]; j++)
   {
      nalu_hypre_fscanf(fp, "%le", &matrix_data[j]);
   }

   fclose(fp);

   nalu_hypre_CSRMatrixNumNonzeros(matrix) = num_nonzeros;
   nalu_hypre_CSRMatrixNumCols(matrix) = ++max_col;
   nalu_hypre_CSRMatrixSetRownnz(matrix);

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixPrint( nalu_hypre_CSRMatrix *matrix,
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

   matrix_data = nalu_hypre_CSRMatrixData(matrix);
   matrix_i    = nalu_hypre_CSRMatrixI(matrix);
   matrix_j    = nalu_hypre_CSRMatrixJ(matrix);
   matrix_bigj = nalu_hypre_CSRMatrixBigJ(matrix);
   num_rows    = nalu_hypre_CSRMatrixNumRows(matrix);

   fp = fopen(file_name, "w");

   nalu_hypre_fprintf(fp, "%d\n", num_rows);

   for (j = 0; j <= num_rows; j++)
   {
      nalu_hypre_fprintf(fp, "%d\n", matrix_i[j] + file_base);
   }

   if (matrix_j)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
         nalu_hypre_fprintf(fp, "%d\n", matrix_j[j] + file_base);
      }
   }

   if (matrix_bigj)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
         nalu_hypre_fprintf(fp, "%d\n", matrix_bigj[j] + file_base);
      }
   }

   if (matrix_data)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
#ifdef NALU_HYPRE_COMPLEX
         nalu_hypre_fprintf(fp, "%.14e , %.14e\n",
                       nalu_hypre_creal(matrix_data[j]), nalu_hypre_cimag(matrix_data[j]));
#else
         nalu_hypre_fprintf(fp, "%.14e\n", matrix_data[j]);
#endif
      }
   }
   else
   {
      nalu_hypre_fprintf(fp, "Warning: No matrix data!\n");
   }

   fclose(fp);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixPrintIJ
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixPrintIJ( nalu_hypre_CSRMatrix  *matrix,
                        NALU_HYPRE_Int         base_i,
                        NALU_HYPRE_Int         base_j,
                        char             *filename )
{
   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_CSRMatrixMemoryLocation(matrix);
   nalu_hypre_CSRMatrix     *h_matrix;

   NALU_HYPRE_Int            patt_only;
   NALU_HYPRE_Int            num_rows;
   NALU_HYPRE_Int            num_cols;
   NALU_HYPRE_Int           *matrix_i;
   NALU_HYPRE_Int           *matrix_j;
   NALU_HYPRE_BigInt        *matrix_bj;
   NALU_HYPRE_Complex       *matrix_a;

   NALU_HYPRE_Int            i, j, ii, jj;
   NALU_HYPRE_Int            ilower, iupper, jlower, jupper;
   FILE                *file;

   if (!matrix)
   {
      nalu_hypre_error_in_arg(1);
      return nalu_hypre_error_flag;
   }

   /* Create temporary matrix on host memory if needed */
   h_matrix = (nalu_hypre_GetActualMemLocation(memory_location) == nalu_hypre_MEMORY_DEVICE) ?
              nalu_hypre_CSRMatrixClone_v2(matrix, 1, NALU_HYPRE_MEMORY_HOST) : matrix;

   /* Set matrix info */
   patt_only = nalu_hypre_CSRMatrixPatternOnly(h_matrix);
   num_rows  = nalu_hypre_CSRMatrixNumRows(h_matrix);
   num_cols  = nalu_hypre_CSRMatrixNumCols(h_matrix);
   matrix_i  = nalu_hypre_CSRMatrixI(h_matrix);
   matrix_j  = nalu_hypre_CSRMatrixJ(h_matrix);
   matrix_bj = nalu_hypre_CSRMatrixBigJ(h_matrix);
   matrix_a  = nalu_hypre_CSRMatrixData(h_matrix);

   if ((file = fopen(filename, "w")) == NULL)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "Error: can't open output file %s\n");
      return nalu_hypre_error_flag;
   }

   /* Print matrix bounds */
   ilower = base_i;
   iupper = num_rows + base_i - 1;
   jlower = base_j;
   jupper = num_cols + base_j - 1;
   nalu_hypre_fprintf(file, "%b %b %b %b\n", ilower, iupper, jlower, jupper);

   for (i = 0; i < num_rows; i++)
   {
      ii = i + base_i;

      /* print diag columns */
      for (j = matrix_i[i]; j < matrix_i[i + 1]; j++)
      {
         jj = (matrix_bj) ? (matrix_bj[j] + base_j) : (matrix_j[j] + base_j);

         if (!patt_only)
         {
#ifdef NALU_HYPRE_COMPLEX
            nalu_hypre_fprintf(file, "%b %b %.14e , %.14e\n", ii, jj,
                          nalu_hypre_creal(matrix_a[j]), nalu_hypre_cimag(matrix_a[j]));
#else
            nalu_hypre_fprintf(file, "%b %b %.14e\n", ii, jj, matrix_a[j]);
#endif
         }
         else
         {
            nalu_hypre_fprintf(file, "%b %b\n", ii, jj);
         }
      }
   }

   fclose(file);

   /* Free temporary matrix */
   if (h_matrix != matrix)
   {
      nalu_hypre_CSRMatrixDestroy(h_matrix);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixPrintMM
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixPrintMM( nalu_hypre_CSRMatrix *matrix,
                        NALU_HYPRE_Int        basei,
                        NALU_HYPRE_Int        basej,
                        NALU_HYPRE_Int        trans,
                        const char      *file_name )
{
   FILE *fp = file_name ? fopen(file_name, "w") : stdout;

   if (!fp)
   {
      nalu_hypre_error_w_msg(1, "Cannot open output file");
      return nalu_hypre_error_flag;
   }

   const NALU_HYPRE_Complex *matrix_data = nalu_hypre_CSRMatrixData(matrix);
   const NALU_HYPRE_Int     *matrix_i    = nalu_hypre_CSRMatrixI(matrix);
   const NALU_HYPRE_Int     *matrix_j    = nalu_hypre_CSRMatrixJ(matrix);

   nalu_hypre_assert(nalu_hypre_CSRMatrixI(matrix)[nalu_hypre_CSRMatrixNumRows(matrix)] ==
                nalu_hypre_CSRMatrixNumNonzeros(matrix));

   if (matrix_data)
   {
      nalu_hypre_fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
   }
   else
   {
      nalu_hypre_fprintf(fp, "%%%%MatrixMarket matrix coordinate pattern general\n");
   }

   nalu_hypre_fprintf(fp, "%d %d %d\n",
                 trans ? nalu_hypre_CSRMatrixNumCols(matrix) : nalu_hypre_CSRMatrixNumRows(matrix),
                 trans ? nalu_hypre_CSRMatrixNumRows(matrix) : nalu_hypre_CSRMatrixNumCols(matrix),
                 nalu_hypre_CSRMatrixNumNonzeros(matrix));

   NALU_HYPRE_Int i, j;

   for (i = 0; i < nalu_hypre_CSRMatrixNumRows(matrix); i++)
   {
      for (j = matrix_i[i]; j < matrix_i[i + 1]; j++)
      {
         const NALU_HYPRE_Int row = (trans ? matrix_j[j] : i) + basei;
         const NALU_HYPRE_Int col = (trans ? i : matrix_j[j]) + basej;
         if (matrix_data)
         {
            nalu_hypre_fprintf(fp, "%d %d %.15e\n", row, col, matrix_data[j]);
         }
         else
         {
            nalu_hypre_fprintf(fp, "%d %d\n", row, col);
         }
      }
   }

   if (file_name)
   {
      fclose(fp);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixPrintHB:
 *
 * Print a CSRMatrix in Harwell-Boeing format
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixPrintHB( nalu_hypre_CSRMatrix *matrix_input,
                        char            *file_name )
{
   FILE            *fp;
   nalu_hypre_CSRMatrix *matrix;
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
   nalu_hypre_CSRMatrixTranspose(matrix_input, &matrix, 1);

   matrix_data = nalu_hypre_CSRMatrixData(matrix);
   matrix_i    = nalu_hypre_CSRMatrixI(matrix);
   matrix_j    = nalu_hypre_CSRMatrixJ(matrix);
   num_rows    = nalu_hypre_CSRMatrixNumRows(matrix);

   fp = fopen(file_name, "w");

   nalu_hypre_fprintf(fp, "%-70s  Key     \n", "Title");
   ptrcrd = num_rows;
   indcrd = matrix_i[num_rows];
   valcrd = matrix_i[num_rows];
   rhscrd = 0;
   totcrd = ptrcrd + indcrd + valcrd + rhscrd;
   nalu_hypre_fprintf (fp, "%14d%14d%14d%14d%14d\n",
                  totcrd, ptrcrd, indcrd, valcrd, rhscrd);
   nalu_hypre_fprintf (fp, "%-14s%14i%14i%14i%14i\n", "RUA",
                  num_rows, num_rows, valcrd, 0);
   nalu_hypre_fprintf (fp, "%-16s%-16s%-16s%26s\n", "(1I8)", "(1I8)", "(1E16.8)", "");

   for (j = 0; j <= num_rows; j++)
   {
      nalu_hypre_fprintf(fp, "%8d\n", matrix_i[j] + file_base);
   }

   for (j = 0; j < matrix_i[num_rows]; j++)
   {
      nalu_hypre_fprintf(fp, "%8d\n", matrix_j[j] + file_base);
   }

   if (matrix_data)
   {
      for (j = 0; j < matrix_i[num_rows]; j++)
      {
#ifdef NALU_HYPRE_COMPLEX
         nalu_hypre_fprintf(fp, "%16.8e , %16.8e\n",
                       nalu_hypre_creal(matrix_data[j]), nalu_hypre_cimag(matrix_data[j]));
#else
         nalu_hypre_fprintf(fp, "%16.8e\n", matrix_data[j]);
#endif
      }
   }
   else
   {
      nalu_hypre_fprintf(fp, "Warning: No matrix data!\n");
   }

   fclose(fp);

   nalu_hypre_CSRMatrixDestroy(matrix);

   return ierr;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixCopy: copy A to B,
 *
 * if copy_data = 0 only the structure of A is copied to B.
 * the routine does not check if the dimensions/sparsity of A and B match !!!
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixCopy( nalu_hypre_CSRMatrix *A, nalu_hypre_CSRMatrix *B, NALU_HYPRE_Int copy_data )
{
   NALU_HYPRE_Int num_rows = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(A);

   NALU_HYPRE_Int     *A_i      = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *A_j      = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_BigInt  *A_bigj   = nalu_hypre_CSRMatrixBigJ(A);
   NALU_HYPRE_Int     *A_rownnz = nalu_hypre_CSRMatrixRownnz(A);
   NALU_HYPRE_Complex *A_data;

   NALU_HYPRE_Int     *B_i      = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int     *B_j      = nalu_hypre_CSRMatrixJ(B);
   NALU_HYPRE_BigInt  *B_bigj   = nalu_hypre_CSRMatrixBigJ(B);
   NALU_HYPRE_Int     *B_rownnz = nalu_hypre_CSRMatrixRownnz(B);
   NALU_HYPRE_Complex *B_data;

   NALU_HYPRE_MemoryLocation memory_location_A = nalu_hypre_CSRMatrixMemoryLocation(A);
   NALU_HYPRE_MemoryLocation memory_location_B = nalu_hypre_CSRMatrixMemoryLocation(B);

   nalu_hypre_TMemcpy(B_i, A_i, NALU_HYPRE_Int, num_rows + 1, memory_location_B, memory_location_A);

   if (A_rownnz)
   {
      if (!B_rownnz)
      {
         B_rownnz = nalu_hypre_TAlloc(NALU_HYPRE_Int,
                                 nalu_hypre_CSRMatrixNumRownnz(A),
                                 memory_location_B);
         nalu_hypre_CSRMatrixRownnz(B) = B_rownnz;
      }
      nalu_hypre_TMemcpy(B_rownnz, A_rownnz,
                    NALU_HYPRE_Int, nalu_hypre_CSRMatrixNumRownnz(A),
                    memory_location_B, memory_location_A);
   }
   nalu_hypre_CSRMatrixNumRownnz(B) = nalu_hypre_CSRMatrixNumRownnz(A);

   if (A_j && B_j)
   {
      nalu_hypre_TMemcpy(B_j, A_j, NALU_HYPRE_Int, num_nonzeros, memory_location_B, memory_location_A);
   }

   if (A_bigj && B_bigj)
   {
      nalu_hypre_TMemcpy(B_bigj, A_bigj, NALU_HYPRE_BigInt, num_nonzeros,
                    memory_location_B, memory_location_A);
   }

   if (copy_data)
   {
      A_data = nalu_hypre_CSRMatrixData(A);
      B_data = nalu_hypre_CSRMatrixData(B);
      nalu_hypre_TMemcpy(B_data, A_data, NALU_HYPRE_Complex, num_nonzeros,
                    memory_location_B, memory_location_A);
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixMigrate
 *
 * Migrates matrix row pointer, column indices and data to memory_location
 * if it is different to the current one.
 *
 * Note: Does not move rownnz array.
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixMigrate( nalu_hypre_CSRMatrix     *A,
                        NALU_HYPRE_MemoryLocation memory_location )
{
   /* Input matrix info */
   NALU_HYPRE_Int       num_rows     = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int       num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int      *A_ri         = nalu_hypre_CSRMatrixRownnz(A);
   NALU_HYPRE_Int      *A_i          = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int      *A_j          = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_BigInt   *A_big_j      = nalu_hypre_CSRMatrixBigJ(A);
   NALU_HYPRE_Complex  *A_data       = nalu_hypre_CSRMatrixData(A);

   NALU_HYPRE_MemoryLocation old_memory_location = nalu_hypre_CSRMatrixMemoryLocation(A);

   /* Output matrix info */
   NALU_HYPRE_Int      *B_i;
   NALU_HYPRE_Int      *B_j;
   NALU_HYPRE_BigInt   *B_big_j;
   NALU_HYPRE_Complex  *B_data;
   NALU_HYPRE_Int      *B_ri;

   /* Check pointer locations in debug mode */
#if defined(NALU_HYPRE_DEBUG)
   nalu_hypre_CheckMemoryLocation((void*) A_ri,    nalu_hypre_GetActualMemLocation(old_memory_location));
   nalu_hypre_CheckMemoryLocation((void*) A_i,     nalu_hypre_GetActualMemLocation(old_memory_location));
   nalu_hypre_CheckMemoryLocation((void*) A_j,     nalu_hypre_GetActualMemLocation(old_memory_location));
   nalu_hypre_CheckMemoryLocation((void*) A_big_j, nalu_hypre_GetActualMemLocation(old_memory_location));
   nalu_hypre_CheckMemoryLocation((void*) A_data,  nalu_hypre_GetActualMemLocation(old_memory_location));
#endif

   /* Update A's memory location */
   nalu_hypre_CSRMatrixMemoryLocation(A) = memory_location;

   if ( nalu_hypre_GetActualMemLocation(memory_location) !=
        nalu_hypre_GetActualMemLocation(old_memory_location) )
   {
      if (A_ri)
      {
         B_ri = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows, memory_location);
         nalu_hypre_TMemcpy(B_ri, A_ri, NALU_HYPRE_Int, num_rows,
                       memory_location, old_memory_location);
         nalu_hypre_TFree(A_ri, old_memory_location);
         nalu_hypre_CSRMatrixRownnz(A) = B_ri;
      }

      if (A_i)
      {
         B_i = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_rows + 1, memory_location);
         nalu_hypre_TMemcpy(B_i, A_i, NALU_HYPRE_Int, num_rows + 1,
                       memory_location, old_memory_location);
         nalu_hypre_TFree(A_i, old_memory_location);
         nalu_hypre_CSRMatrixI(A) = B_i;
      }

      if (A_j)
      {
         B_j = nalu_hypre_TAlloc(NALU_HYPRE_Int, num_nonzeros, memory_location);
         nalu_hypre_TMemcpy(B_j, A_j, NALU_HYPRE_Int, num_nonzeros,
                       memory_location, old_memory_location);
         nalu_hypre_TFree(A_j, old_memory_location);
         nalu_hypre_CSRMatrixJ(A) = B_j;
      }

      if (A_big_j)
      {
         B_big_j = nalu_hypre_TAlloc(NALU_HYPRE_BigInt, num_nonzeros, memory_location);
         nalu_hypre_TMemcpy(B_big_j, A_big_j, NALU_HYPRE_BigInt, num_nonzeros,
                       memory_location, old_memory_location);
         nalu_hypre_TFree(A_big_j, old_memory_location);
         nalu_hypre_CSRMatrixBigJ(A) = B_big_j;
      }

      if (A_data)
      {
         B_data = nalu_hypre_TAlloc(NALU_HYPRE_Complex, num_nonzeros, memory_location);
         nalu_hypre_TMemcpy(B_data, A_data, NALU_HYPRE_Complex, num_nonzeros,
                       memory_location, old_memory_location);
         nalu_hypre_TFree(A_data, old_memory_location);
         nalu_hypre_CSRMatrixData(A) = B_data;
      }
   }

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixClone_v2
 *
 * This function does the same job as nalu_hypre_CSRMatrixClone; however, here
 * the user can specify the memory location of the resulting matrix.
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix*
nalu_hypre_CSRMatrixClone_v2( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int copy_data,
                         NALU_HYPRE_MemoryLocation memory_location )
{
   NALU_HYPRE_Int num_rows = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int num_cols = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(A);

   nalu_hypre_CSRMatrix *B = nalu_hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);

   NALU_HYPRE_Int bigInit = nalu_hypre_CSRMatrixBigJ(A) != NULL;

   nalu_hypre_CSRMatrixInitialize_v2(B, bigInit, memory_location);

   nalu_hypre_CSRMatrixCopy(A, B, copy_data);

   return B;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixClone
 *
 * Creates and returns a new copy of the argument, A.
 * Performs a deep copy of information (no pointers are copied);
 * New arrays are created where necessary.
 *--------------------------------------------------------------------------*/

nalu_hypre_CSRMatrix*
nalu_hypre_CSRMatrixClone( nalu_hypre_CSRMatrix *A, NALU_HYPRE_Int copy_data )
{
   return nalu_hypre_CSRMatrixClone_v2(A, copy_data, nalu_hypre_CSRMatrixMemoryLocation(A));
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixPermuteHost
 *
 * See nalu_hypre_CSRMatrixPermute. TODO (VPM): OpenMP implementation
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixPermuteHost( nalu_hypre_CSRMatrix  *A,
                            NALU_HYPRE_Int        *perm,
                            NALU_HYPRE_Int        *rqperm,
                            nalu_hypre_CSRMatrix  *B )
{
   /* Input variables */
   NALU_HYPRE_Int         num_rows     = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int         num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int        *A_i          = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int        *A_j          = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Complex    *A_a          = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int        *B_i          = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int        *B_j          = nalu_hypre_CSRMatrixJ(B);
   NALU_HYPRE_Complex    *B_a          = nalu_hypre_CSRMatrixData(B);

   /* Local variables */
   NALU_HYPRE_Int         i, j, k;

   /* Build B = A(perm, qperm) */
   k = 0;
   for (i = 0; i < num_rows; i++)
   {
      B_i[i] = k;
      for (j = A_i[perm[i]]; j < A_i[perm[i] + 1]; j++)
      {
         B_j[k] = rqperm[A_j[j]];
         B_a[k++] = A_a[j];
      }
   }
   B_i[num_rows] = k;
   nalu_hypre_assert(k == num_nonzeros);

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixPermute
 *
 * Reorder a CSRMatrix according to a row-permutation array (perm) and
 * reverse column-permutation array (rqperm).
 *
 * Notes:
 *  1) This function does not move the diagonal to the first entry of a row
 *  2) When perm == rqperm == NULL, B is a deep copy of A.
 *
 * TODO (VPM): add check for permutation arrays under NALU_HYPRE_DEBUG
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixPermute( nalu_hypre_CSRMatrix  *A,
                        NALU_HYPRE_Int        *perm,
                        NALU_HYPRE_Int        *rqperm,
                        nalu_hypre_CSRMatrix **B_ptr )
{
   NALU_HYPRE_Int          num_rows     = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int          num_cols     = nalu_hypre_CSRMatrixNumCols(A);
   NALU_HYPRE_Int          num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(A);
   nalu_hypre_CSRMatrix   *B;

   nalu_hypre_GpuProfilingPushRange("CSRMatrixPermute");

   /* Special case: one of the permutation vectors are not provided, then B = A */
   if (!perm || !rqperm)
   {
      *B_ptr = nalu_hypre_CSRMatrixClone(A, 1);
      nalu_hypre_GpuProfilingPopRange();

      return nalu_hypre_error_flag;
   }

   /* Create output matrix B */
   B = nalu_hypre_CSRMatrixCreate(num_rows, num_cols, num_nonzeros);
   nalu_hypre_CSRMatrixInitialize_v2(B, 0, nalu_hypre_CSRMatrixMemoryLocation(A));

#if defined(NALU_HYPRE_USING_CUDA) || defined(NALU_HYPRE_USING_HIP)
   NALU_HYPRE_ExecutionPolicy exec = nalu_hypre_GetExecPolicy1( nalu_hypre_CSRMatrixMemoryLocation(A) );

   if (exec == NALU_HYPRE_EXEC_DEVICE)
   {
      nalu_hypre_CSRMatrixPermuteDevice(A, perm, rqperm, B);
   }
   else
#endif
   {
      nalu_hypre_CSRMatrixPermuteHost(A, perm, rqperm, B);
   }

   nalu_hypre_GpuProfilingPopRange();

   /* Set output pointer */
   *B_ptr = B;

   return nalu_hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixUnion
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

nalu_hypre_CSRMatrix*
nalu_hypre_CSRMatrixUnion( nalu_hypre_CSRMatrix *A,
                      nalu_hypre_CSRMatrix *B,
                      NALU_HYPRE_BigInt *col_map_offd_A,
                      NALU_HYPRE_BigInt *col_map_offd_B,
                      NALU_HYPRE_BigInt **col_map_offd_C )
{
   NALU_HYPRE_Int num_rows = nalu_hypre_CSRMatrixNumRows( A );
   NALU_HYPRE_Int num_cols_A = nalu_hypre_CSRMatrixNumCols( A );
   NALU_HYPRE_Int num_cols_B = nalu_hypre_CSRMatrixNumCols( B );
   NALU_HYPRE_Int num_cols;
   NALU_HYPRE_Int num_nonzeros;
   NALU_HYPRE_Int *A_i = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int *A_j = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int *B_i = nalu_hypre_CSRMatrixI(B);
   NALU_HYPRE_Int *B_j = nalu_hypre_CSRMatrixJ(B);
   NALU_HYPRE_Int *C_i;
   NALU_HYPRE_Int *C_j;
   NALU_HYPRE_Int *jC = NULL;
   NALU_HYPRE_BigInt jBg, big_jA, big_jB;
   NALU_HYPRE_Int i, jA, jB;
   NALU_HYPRE_Int ma, mb, mc, ma_min, ma_max, match;
   nalu_hypre_CSRMatrix* C;

   NALU_HYPRE_MemoryLocation memory_location = nalu_hypre_CSRMatrixMemoryLocation(A);

   nalu_hypre_assert( num_rows == nalu_hypre_CSRMatrixNumRows(B) );

   if ( col_map_offd_B )
   {
      nalu_hypre_assert( col_map_offd_A );
   }

   if ( col_map_offd_A )
   {
      nalu_hypre_assert( col_map_offd_B );
   }

   /* ==== First, go through the columns of A and B to count the columns of C. */
   if ( col_map_offd_A == 0 )
   {
      /* The matrices are diagonal blocks.
         Normally num_cols_A==num_cols_B, col_starts is the same, etc.
      */
      num_cols = nalu_hypre_max( num_cols_A, num_cols_B );
   }
   else
   {
      /* The matrices are offdiagonal blocks. */
      jC = nalu_hypre_CTAlloc(NALU_HYPRE_Int, num_cols_B, NALU_HYPRE_MEMORY_HOST);
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
      *col_map_offd_C = nalu_hypre_CTAlloc( NALU_HYPRE_BigInt, num_cols, NALU_HYPRE_MEMORY_HOST);
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
   num_nonzeros = nalu_hypre_CSRMatrixNumNonzeros(A);
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

   C = nalu_hypre_CSRMatrixCreate( num_rows, num_cols, num_nonzeros );
   nalu_hypre_CSRMatrixInitialize_v2( C, 0, memory_location );

   /* ==== The second run through A and B is to pick out the column numbers
      for each row, and put them in C. */
   C_i = nalu_hypre_CSRMatrixI(C);
   C_i[0] = 0;
   C_j = nalu_hypre_CSRMatrixJ(C);
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

   nalu_hypre_assert( mc == num_nonzeros );

   nalu_hypre_TFree(jC, NALU_HYPRE_MEMORY_HOST);

   return C;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixGetLoadBalancedPartitionBoundary
 *--------------------------------------------------------------------------*/

static NALU_HYPRE_Int
nalu_hypre_CSRMatrixGetLoadBalancedPartitionBoundary(nalu_hypre_CSRMatrix *A,
                                                NALU_HYPRE_Int        idx)
{
   NALU_HYPRE_Int num_nonzerosA = nalu_hypre_CSRMatrixNumNonzeros(A);
   NALU_HYPRE_Int num_rowsA = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int *A_i = nalu_hypre_CSRMatrixI(A);

   NALU_HYPRE_Int num_threads = nalu_hypre_NumActiveThreads();

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
      return (NALU_HYPRE_Int)(nalu_hypre_LowerBound(A_i, A_i + num_rowsA, nonzeros_per_thread * idx) - A_i);
   }
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixGetLoadBalancedPartitionBegin
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixGetLoadBalancedPartitionBegin(nalu_hypre_CSRMatrix *A)
{
   return nalu_hypre_CSRMatrixGetLoadBalancedPartitionBoundary(A, nalu_hypre_GetThreadNum());
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixGetLoadBalancedPartitionEnd
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixGetLoadBalancedPartitionEnd(nalu_hypre_CSRMatrix *A)
{
   return nalu_hypre_CSRMatrixGetLoadBalancedPartitionBoundary(A, nalu_hypre_GetThreadNum() + 1);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixPrefetch
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_CSRMatrixPrefetch( nalu_hypre_CSRMatrix      *A,
                         NALU_HYPRE_MemoryLocation  memory_location )
{
#ifdef NALU_HYPRE_USING_UNIFIED_MEMORY
   if (nalu_hypre_CSRMatrixMemoryLocation(A) != NALU_HYPRE_MEMORY_DEVICE)
   {
      nalu_hypre_error_w_msg(NALU_HYPRE_ERROR_GENERIC, "A is not at NALU_HYPRE_MEMORY_DEVICE");
      return nalu_hypre_error_flag;
   }

   NALU_HYPRE_Complex *data = nalu_hypre_CSRMatrixData(A);
   NALU_HYPRE_Int     *ia   = nalu_hypre_CSRMatrixI(A);
   NALU_HYPRE_Int     *ja   = nalu_hypre_CSRMatrixJ(A);
   NALU_HYPRE_Int      nrow = nalu_hypre_CSRMatrixNumRows(A);
   NALU_HYPRE_Int      nnzA = nalu_hypre_CSRMatrixNumNonzeros(A);

   nalu_hypre_MemPrefetch(data, sizeof(NALU_HYPRE_Complex)*nnzA, memory_location);
   nalu_hypre_MemPrefetch(ia,   sizeof(NALU_HYPRE_Int) * (nrow + 1), memory_location);
   nalu_hypre_MemPrefetch(ja,   sizeof(NALU_HYPRE_Int)*nnzA,     memory_location);
#endif

   return nalu_hypre_error_flag;
}

#if defined(NALU_HYPRE_USING_CUSPARSE)  ||\
    defined(NALU_HYPRE_USING_ROCSPARSE) ||\
    defined(NALU_HYPRE_USING_ONEMKLSPARSE)
/*--------------------------------------------------------------------------
 * nalu_hypre_CSRMatrixGetGPUMatData
 *--------------------------------------------------------------------------*/

nalu_hypre_GpuMatData*
nalu_hypre_CSRMatrixGetGPUMatData(nalu_hypre_CSRMatrix *matrix)
{
   if (!matrix)
   {
      return NULL;
   }

   if (!nalu_hypre_CSRMatrixGPUMatData(matrix))
   {
      nalu_hypre_CSRMatrixGPUMatData(matrix) = nalu_hypre_GpuMatDataCreate();
      nalu_hypre_GPUMatDataSetCSRData(matrix);
   }

   return nalu_hypre_CSRMatrixGPUMatData(matrix);
}
#endif
