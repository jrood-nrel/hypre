/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_MultiblockMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_MultiblockMatrix *
hypre_MultiblockMatrixCreate( )
{
   hypre_MultiblockMatrix  *matrix;

   matrix = hypre_CTAlloc(hypre_MultiblockMatrix,  1, NALU_HYPRE_MEMORY_HOST);

   return ( matrix );
}

/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MultiblockMatrixDestroy( hypre_MultiblockMatrix *matrix )
{
   NALU_HYPRE_Int  ierr = 0, i;

   if (matrix)
   {
      for (i = 0; i < hypre_MultiblockMatrixNumSubmatrices(matrix); i++)
      {
         hypre_TFree(hypre_MultiblockMatrixSubmatrix(matrix, i), NALU_HYPRE_MEMORY_HOST);
      }
      hypre_TFree(hypre_MultiblockMatrixSubmatrices(matrix), NALU_HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_MultiblockMatrixSubmatrixTypes(matrix), NALU_HYPRE_MEMORY_HOST);

      hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MultiblockMatrixLimitedDestroy( hypre_MultiblockMatrix *matrix )
{
   NALU_HYPRE_Int  ierr = 0;

   if (matrix)
   {
      hypre_TFree(hypre_MultiblockMatrixSubmatrices(matrix), NALU_HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_MultiblockMatrixSubmatrixTypes(matrix), NALU_HYPRE_MEMORY_HOST);

      hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MultiblockMatrixInitialize( hypre_MultiblockMatrix *matrix )
{
   NALU_HYPRE_Int    ierr = 0;

   if ( hypre_MultiblockMatrixNumSubmatrices(matrix) <= 0 )
   {
      return (-1);
   }

   hypre_MultiblockMatrixSubmatrixTypes(matrix) =
      hypre_CTAlloc( NALU_HYPRE_Int,  hypre_MultiblockMatrixNumSubmatrices(matrix), NALU_HYPRE_MEMORY_HOST);

   hypre_MultiblockMatrixSubmatrices(matrix) =
      hypre_CTAlloc( void *,  hypre_MultiblockMatrixNumSubmatrices(matrix), NALU_HYPRE_MEMORY_HOST);

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixAssemble
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MultiblockMatrixAssemble( hypre_MultiblockMatrix *matrix )
{
   NALU_HYPRE_Int    ierr = 0;

   return (ierr);
}


/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_MultiblockMatrixPrint(hypre_MultiblockMatrix *matrix  )
{
   hypre_printf("Stub for hypre_MultiblockMatrix\n");
}

/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixSetNumSubmatrices
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MultiblockMatrixSetNumSubmatrices(hypre_MultiblockMatrix *matrix, NALU_HYPRE_Int n  )
{
   NALU_HYPRE_Int ierr = 0;

   hypre_MultiblockMatrixNumSubmatrices(matrix) = n;
   return ( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixSetSubmatrixType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MultiblockMatrixSetSubmatrixType(hypre_MultiblockMatrix *matrix,
                                       NALU_HYPRE_Int j,
                                       NALU_HYPRE_Int type  )
{
   NALU_HYPRE_Int ierr = 0;

   if ( (j < 0) ||
        (j >= hypre_MultiblockMatrixNumSubmatrices(matrix)) )
   {
      return (-1);
   }

   hypre_MultiblockMatrixSubmatrixType(matrix, j) = type;

   return ( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_MultiblockMatrixSetSubmatrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MultiblockMatrixSetSubmatrix(hypre_MultiblockMatrix *matrix,
                                   NALU_HYPRE_Int j,
                                   void *submatrix  )
{
   NALU_HYPRE_Int ierr = 0;

   if ( (j < 0) ||
        (j >= hypre_MultiblockMatrixNumSubmatrices(matrix)) )
   {
      return (-1);
   }

   hypre_MultiblockMatrixSubmatrix(matrix, j) = submatrix;

   return ( ierr );
}


