/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_MultiblockMatrix interface
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MultiblockMatrixCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_MultiblockMatrix
NALU_HYPRE_MultiblockMatrixCreate( void )
{
   return ( (NALU_HYPRE_MultiblockMatrix)
            nalu_hypre_MultiblockMatrixCreate(  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MultiblockMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MultiblockMatrixDestroy( NALU_HYPRE_MultiblockMatrix matrix )
{
   return ( nalu_hypre_MultiblockMatrixDestroy( (nalu_hypre_MultiblockMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MultiblockMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MultiblockMatrixLimitedDestroy( NALU_HYPRE_MultiblockMatrix matrix )
{
   return ( nalu_hypre_MultiblockMatrixLimitedDestroy( (nalu_hypre_MultiblockMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MultiblockMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MultiblockMatrixInitialize( NALU_HYPRE_MultiblockMatrix matrix )
{
   return ( nalu_hypre_MultiblockMatrixInitialize( (nalu_hypre_MultiblockMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_MultiblockMatrixAssemble
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MultiblockMatrixAssemble( NALU_HYPRE_MultiblockMatrix matrix )
{
   return ( nalu_hypre_MultiblockMatrixAssemble( (nalu_hypre_MultiblockMatrix *) matrix ) );
}



/*--------------------------------------------------------------------------
 * NALU_HYPRE_MultiblockMatrixPrint
 *--------------------------------------------------------------------------*/

void
NALU_HYPRE_MultiblockMatrixPrint( NALU_HYPRE_MultiblockMatrix matrix )
{
   nalu_hypre_MultiblockMatrixPrint( (nalu_hypre_MultiblockMatrix *) matrix );
}

/****************************************************************************
 END OF ROUTINES THAT ARE ESSENTIALLY JUST CALLS THROUGH TO OTHER ROUTINES
 AND THAT ARE INDEPENDENT OF THE PARTICULAR MATRIX TYPE (except for names)
 ***************************************************************************/

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MultiblockMatrixSetNumSubmatrices
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MultiblockMatrixSetNumSubmatrices( NALU_HYPRE_MultiblockMatrix matrix, NALU_HYPRE_Int n )
{
   return ( nalu_hypre_MultiblockMatrixSetNumSubmatrices(
               (nalu_hypre_MultiblockMatrix *) matrix, n ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MultiblockMatrixSetSubmatrixType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MultiblockMatrixSetSubmatrixType( NALU_HYPRE_MultiblockMatrix matrix,
                                        NALU_HYPRE_Int j,
                                        NALU_HYPRE_Int type )
{
   return ( nalu_hypre_MultiblockMatrixSetSubmatrixType(
               (nalu_hypre_MultiblockMatrix *) matrix, j, type ) );
}
