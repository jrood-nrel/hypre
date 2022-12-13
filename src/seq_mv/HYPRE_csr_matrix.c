/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_CSRMatrix interface
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CSRMatrixCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_CSRMatrix
NALU_HYPRE_CSRMatrixCreate( NALU_HYPRE_Int  num_rows,
                       NALU_HYPRE_Int  num_cols,
                       NALU_HYPRE_Int *row_sizes )
{
   hypre_CSRMatrix *matrix;
   NALU_HYPRE_Int             *matrix_i;
   NALU_HYPRE_Int              i;

   matrix_i = hypre_CTAlloc(NALU_HYPRE_Int,  num_rows + 1, NALU_HYPRE_MEMORY_HOST);
   matrix_i[0] = 0;
   for (i = 0; i < num_rows; i++)
   {
      matrix_i[i + 1] = matrix_i[i] + row_sizes[i];
   }

   matrix = hypre_CSRMatrixCreate(num_rows, num_cols, matrix_i[num_rows]);
   hypre_CSRMatrixI(matrix) = matrix_i;

   return ( (NALU_HYPRE_CSRMatrix) matrix );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CSRMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CSRMatrixDestroy( NALU_HYPRE_CSRMatrix matrix )
{
   return ( hypre_CSRMatrixDestroy( (hypre_CSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CSRMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CSRMatrixInitialize( NALU_HYPRE_CSRMatrix matrix )
{
   return ( hypre_CSRMatrixInitialize( (hypre_CSRMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CSRMatrixRead
 *--------------------------------------------------------------------------*/

NALU_HYPRE_CSRMatrix
NALU_HYPRE_CSRMatrixRead( char            *file_name )
{
   return ( (NALU_HYPRE_CSRMatrix) hypre_CSRMatrixRead( file_name ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CSRMatrixPrint
 *--------------------------------------------------------------------------*/

void
NALU_HYPRE_CSRMatrixPrint( NALU_HYPRE_CSRMatrix  matrix,
                      char            *file_name )
{
   hypre_CSRMatrixPrint( (hypre_CSRMatrix *) matrix,
                         file_name );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_CSRMatrixGetNumRows
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_CSRMatrixGetNumRows( NALU_HYPRE_CSRMatrix matrix, NALU_HYPRE_Int *num_rows )
{
   hypre_CSRMatrix *csr_matrix = (hypre_CSRMatrix *) matrix;

   *num_rows =  hypre_CSRMatrixNumRows( csr_matrix );

   return 0;
}

