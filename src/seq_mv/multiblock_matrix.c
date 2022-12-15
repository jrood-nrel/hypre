/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_MultiblockMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_MultiblockMatrixCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_MultiblockMatrix *
nalu_hypre_MultiblockMatrixCreate( )
{
   nalu_hypre_MultiblockMatrix  *matrix;

   matrix = nalu_hypre_CTAlloc(nalu_hypre_MultiblockMatrix,  1, NALU_HYPRE_MEMORY_HOST);

   return ( matrix );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MultiblockMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MultiblockMatrixDestroy( nalu_hypre_MultiblockMatrix *matrix )
{
   NALU_HYPRE_Int  ierr = 0, i;

   if (matrix)
   {
      for (i = 0; i < nalu_hypre_MultiblockMatrixNumSubmatrices(matrix); i++)
      {
         nalu_hypre_TFree(nalu_hypre_MultiblockMatrixSubmatrix(matrix, i), NALU_HYPRE_MEMORY_HOST);
      }
      nalu_hypre_TFree(nalu_hypre_MultiblockMatrixSubmatrices(matrix), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_MultiblockMatrixSubmatrixTypes(matrix), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_MultiblockMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MultiblockMatrixLimitedDestroy( nalu_hypre_MultiblockMatrix *matrix )
{
   NALU_HYPRE_Int  ierr = 0;

   if (matrix)
   {
      nalu_hypre_TFree(nalu_hypre_MultiblockMatrixSubmatrices(matrix), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_MultiblockMatrixSubmatrixTypes(matrix), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_MultiblockMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MultiblockMatrixInitialize( nalu_hypre_MultiblockMatrix *matrix )
{
   NALU_HYPRE_Int    ierr = 0;

   if ( nalu_hypre_MultiblockMatrixNumSubmatrices(matrix) <= 0 )
   {
      return (-1);
   }

   nalu_hypre_MultiblockMatrixSubmatrixTypes(matrix) =
      nalu_hypre_CTAlloc( NALU_HYPRE_Int,  nalu_hypre_MultiblockMatrixNumSubmatrices(matrix), NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_MultiblockMatrixSubmatrices(matrix) =
      nalu_hypre_CTAlloc( void *,  nalu_hypre_MultiblockMatrixNumSubmatrices(matrix), NALU_HYPRE_MEMORY_HOST);

   return ierr;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_MultiblockMatrixAssemble
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MultiblockMatrixAssemble( nalu_hypre_MultiblockMatrix *matrix )
{
   NALU_HYPRE_Int    ierr = 0;

   return (ierr);
}


/*--------------------------------------------------------------------------
 * nalu_hypre_MultiblockMatrixPrint
 *--------------------------------------------------------------------------*/

void
nalu_hypre_MultiblockMatrixPrint(nalu_hypre_MultiblockMatrix *matrix  )
{
   nalu_hypre_printf("Stub for nalu_hypre_MultiblockMatrix\n");
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MultiblockMatrixSetNumSubmatrices
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MultiblockMatrixSetNumSubmatrices(nalu_hypre_MultiblockMatrix *matrix, NALU_HYPRE_Int n  )
{
   NALU_HYPRE_Int ierr = 0;

   nalu_hypre_MultiblockMatrixNumSubmatrices(matrix) = n;
   return ( ierr );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MultiblockMatrixSetSubmatrixType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MultiblockMatrixSetSubmatrixType(nalu_hypre_MultiblockMatrix *matrix,
                                       NALU_HYPRE_Int j,
                                       NALU_HYPRE_Int type  )
{
   NALU_HYPRE_Int ierr = 0;

   if ( (j < 0) ||
        (j >= nalu_hypre_MultiblockMatrixNumSubmatrices(matrix)) )
   {
      return (-1);
   }

   nalu_hypre_MultiblockMatrixSubmatrixType(matrix, j) = type;

   return ( ierr );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MultiblockMatrixSetSubmatrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MultiblockMatrixSetSubmatrix(nalu_hypre_MultiblockMatrix *matrix,
                                   NALU_HYPRE_Int j,
                                   void *submatrix  )
{
   NALU_HYPRE_Int ierr = 0;

   if ( (j < 0) ||
        (j >= nalu_hypre_MultiblockMatrixNumSubmatrices(matrix)) )
   {
      return (-1);
   }

   nalu_hypre_MultiblockMatrixSubmatrix(matrix, j) = submatrix;

   return ( ierr );
}


