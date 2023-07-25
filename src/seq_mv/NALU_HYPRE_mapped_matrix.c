/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_MappedMatrix interface
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MappedMatrixCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_MappedMatrix
NALU_HYPRE_MappedMatrixCreate( void )
{
   return ( (NALU_HYPRE_MappedMatrix)
            nalu_hypre_MappedMatrixCreate(  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MappedMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MappedMatrixDestroy( NALU_HYPRE_MappedMatrix matrix )
{
   return ( nalu_hypre_MappedMatrixDestroy( (nalu_hypre_MappedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MappedMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MappedMatrixLimitedDestroy( NALU_HYPRE_MappedMatrix matrix )
{
   return ( nalu_hypre_MappedMatrixLimitedDestroy( (nalu_hypre_MappedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MappedMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MappedMatrixInitialize( NALU_HYPRE_MappedMatrix matrix )
{
   return ( nalu_hypre_MappedMatrixInitialize( (nalu_hypre_MappedMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_MappedMatrixAssemble
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MappedMatrixAssemble( NALU_HYPRE_MappedMatrix matrix )
{
   return ( nalu_hypre_MappedMatrixAssemble( (nalu_hypre_MappedMatrix *) matrix ) );
}



/*--------------------------------------------------------------------------
 * NALU_HYPRE_MappedMatrixPrint
 *--------------------------------------------------------------------------*/

void
NALU_HYPRE_MappedMatrixPrint( NALU_HYPRE_MappedMatrix matrix )
{
   nalu_hypre_MappedMatrixPrint( (nalu_hypre_MappedMatrix *) matrix );
}

/****************************************************************************
 END OF ROUTINES THAT ARE ESSENTIALLY JUST CALLS THROUGH TO OTHER ROUTINES
 AND THAT ARE INDEPENDENT OF THE PARTICULAR MATRIX TYPE (except for names)
 ***************************************************************************/

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MappedMatrixGetColIndex
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MappedMatrixGetColIndex( NALU_HYPRE_MappedMatrix matrix, NALU_HYPRE_Int j )
{
   return ( nalu_hypre_MappedMatrixGetColIndex( (nalu_hypre_MappedMatrix *) matrix, j ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MappedMatrixGetMatrix
 *--------------------------------------------------------------------------*/

void *
NALU_HYPRE_MappedMatrixGetMatrix( NALU_HYPRE_MappedMatrix matrix )
{
   return ( nalu_hypre_MappedMatrixGetMatrix( (nalu_hypre_MappedMatrix *) matrix ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MappedMatrixSetMatrix
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MappedMatrixSetMatrix( NALU_HYPRE_MappedMatrix matrix, void *matrix_data )
{
   return ( nalu_hypre_MappedMatrixSetMatrix( (nalu_hypre_MappedMatrix *) matrix, matrix_data ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MappedMatrixSetColMap
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MappedMatrixSetColMap( NALU_HYPRE_MappedMatrix matrix, NALU_HYPRE_Int (*ColMap)(NALU_HYPRE_Int, void *) )
{
   return ( nalu_hypre_MappedMatrixSetColMap( (nalu_hypre_MappedMatrix *) matrix, ColMap ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MappedMatrixSetMapData
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MappedMatrixSetMapData( NALU_HYPRE_MappedMatrix matrix, void *MapData )
{
   return ( nalu_hypre_MappedMatrixSetMapData( (nalu_hypre_MappedMatrix *) matrix, MapData ) );
}
