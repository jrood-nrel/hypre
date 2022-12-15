/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_MappedMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * nalu_hypre_MappedMatrixCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_MappedMatrix *
nalu_hypre_MappedMatrixCreate(  )
{
   nalu_hypre_MappedMatrix  *matrix;


   matrix = nalu_hypre_CTAlloc(nalu_hypre_MappedMatrix,  1, NALU_HYPRE_MEMORY_HOST);

   return ( matrix );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MappedMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MappedMatrixDestroy( nalu_hypre_MappedMatrix *matrix )
{
   NALU_HYPRE_Int  ierr = 0;

   if (matrix)
   {
      nalu_hypre_TFree(nalu_hypre_MappedMatrixMatrix(matrix), NALU_HYPRE_MEMORY_HOST);
      nalu_hypre_TFree(nalu_hypre_MappedMatrixMapData(matrix), NALU_HYPRE_MEMORY_HOST);

      nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_MappedMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MappedMatrixLimitedDestroy( nalu_hypre_MappedMatrix *matrix )
{
   NALU_HYPRE_Int  ierr = 0;

   if (matrix)
   {
      nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_MappedMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MappedMatrixInitialize( nalu_hypre_MappedMatrix *matrix )
{
   NALU_HYPRE_Int    ierr = 0;

   return ierr;
}


/*--------------------------------------------------------------------------
 * nalu_hypre_MappedMatrixAssemble
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MappedMatrixAssemble( nalu_hypre_MappedMatrix *matrix )
{
   NALU_HYPRE_Int    ierr = 0;

   if ( matrix == NULL )
   {
      return ( -1 ) ;
   }

   if ( nalu_hypre_MappedMatrixMatrix(matrix) == NULL )
   {
      return ( -1 ) ;
   }

   if ( nalu_hypre_MappedMatrixColMap(matrix) == NULL )
   {
      return ( -1 ) ;
   }

   if ( nalu_hypre_MappedMatrixMapData(matrix) == NULL )
   {
      return ( -1 ) ;
   }

   return (ierr);
}


/*--------------------------------------------------------------------------
 * nalu_hypre_MappedMatrixPrint
 *--------------------------------------------------------------------------*/

void
nalu_hypre_MappedMatrixPrint(nalu_hypre_MappedMatrix *matrix  )
{
   nalu_hypre_printf("Stub for nalu_hypre_MappedMatrix\n");
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MappedMatrixGetColIndex
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MappedMatrixGetColIndex(nalu_hypre_MappedMatrix *matrix, NALU_HYPRE_Int j  )
{
   return ( nalu_hypre_MappedMatrixColIndex(matrix, j) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MappedMatrixGetMatrix
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_MappedMatrixGetMatrix(nalu_hypre_MappedMatrix *matrix )
{
   return ( nalu_hypre_MappedMatrixMatrix(matrix) );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MappedMatrixSetMatrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MappedMatrixSetMatrix(nalu_hypre_MappedMatrix *matrix, void *matrix_data  )
{
   NALU_HYPRE_Int ierr = 0;

   nalu_hypre_MappedMatrixMatrix(matrix) = matrix_data;

   return (ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MappedMatrixSetColMap
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MappedMatrixSetColMap(nalu_hypre_MappedMatrix *matrix,
                            NALU_HYPRE_Int (*ColMap)(NALU_HYPRE_Int, void *)  )
{
   NALU_HYPRE_Int ierr = 0;

   nalu_hypre_MappedMatrixColMap(matrix) = ColMap;

   return (ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_MappedMatrixSetMapData
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
nalu_hypre_MappedMatrixSetMapData(nalu_hypre_MappedMatrix *matrix,
                             void *map_data )
{
   NALU_HYPRE_Int ierr = 0;

   nalu_hypre_MappedMatrixMapData(matrix) = map_data;

   return (ierr);
}

