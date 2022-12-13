/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_MappedMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_MappedMatrix *
hypre_MappedMatrixCreate(  )
{
   hypre_MappedMatrix  *matrix;


   matrix = hypre_CTAlloc(hypre_MappedMatrix,  1, NALU_HYPRE_MEMORY_HOST);

   return ( matrix );
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MappedMatrixDestroy( hypre_MappedMatrix *matrix )
{
   NALU_HYPRE_Int  ierr = 0;

   if (matrix)
   {
      hypre_TFree(hypre_MappedMatrixMatrix(matrix), NALU_HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_MappedMatrixMapData(matrix), NALU_HYPRE_MEMORY_HOST);

      hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MappedMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MappedMatrixLimitedDestroy( hypre_MappedMatrix *matrix )
{
   NALU_HYPRE_Int  ierr = 0;

   if (matrix)
   {
      hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MappedMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MappedMatrixInitialize( hypre_MappedMatrix *matrix )
{
   NALU_HYPRE_Int    ierr = 0;

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_MappedMatrixAssemble
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MappedMatrixAssemble( hypre_MappedMatrix *matrix )
{
   NALU_HYPRE_Int    ierr = 0;

   if ( matrix == NULL )
   {
      return ( -1 ) ;
   }

   if ( hypre_MappedMatrixMatrix(matrix) == NULL )
   {
      return ( -1 ) ;
   }

   if ( hypre_MappedMatrixColMap(matrix) == NULL )
   {
      return ( -1 ) ;
   }

   if ( hypre_MappedMatrixMapData(matrix) == NULL )
   {
      return ( -1 ) ;
   }

   return (ierr);
}


/*--------------------------------------------------------------------------
 * hypre_MappedMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_MappedMatrixPrint(hypre_MappedMatrix *matrix  )
{
   hypre_printf("Stub for hypre_MappedMatrix\n");
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixGetColIndex
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MappedMatrixGetColIndex(hypre_MappedMatrix *matrix, NALU_HYPRE_Int j  )
{
   return ( hypre_MappedMatrixColIndex(matrix, j) );
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixGetMatrix
 *--------------------------------------------------------------------------*/

void *
hypre_MappedMatrixGetMatrix(hypre_MappedMatrix *matrix )
{
   return ( hypre_MappedMatrixMatrix(matrix) );
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixSetMatrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MappedMatrixSetMatrix(hypre_MappedMatrix *matrix, void *matrix_data  )
{
   NALU_HYPRE_Int ierr = 0;

   hypre_MappedMatrixMatrix(matrix) = matrix_data;

   return (ierr);
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixSetColMap
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MappedMatrixSetColMap(hypre_MappedMatrix *matrix,
                            NALU_HYPRE_Int (*ColMap)(NALU_HYPRE_Int, void *)  )
{
   NALU_HYPRE_Int ierr = 0;

   hypre_MappedMatrixColMap(matrix) = ColMap;

   return (ierr);
}

/*--------------------------------------------------------------------------
 * hypre_MappedMatrixSetMapData
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
hypre_MappedMatrixSetMapData(hypre_MappedMatrix *matrix,
                             void *map_data )
{
   NALU_HYPRE_Int ierr = 0;

   hypre_MappedMatrixMapData(matrix) = map_data;

   return (ierr);
}

