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
NALU_HYPRE_MultiblockMatrixCreate( )
{
   return ( (NALU_HYPRE_MultiblockMatrix)
            hypre_MultiblockMatrixCreate(  ));
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MultiblockMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MultiblockMatrixDestroy( NALU_HYPRE_MultiblockMatrix matrix )
{
   return ( hypre_MultiblockMatrixDestroy( (hypre_MultiblockMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MultiblockMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MultiblockMatrixLimitedDestroy( NALU_HYPRE_MultiblockMatrix matrix )
{
   return ( hypre_MultiblockMatrixLimitedDestroy( (hypre_MultiblockMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MultiblockMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MultiblockMatrixInitialize( NALU_HYPRE_MultiblockMatrix matrix )
{
   return ( hypre_MultiblockMatrixInitialize( (hypre_MultiblockMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_MultiblockMatrixAssemble
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MultiblockMatrixAssemble( NALU_HYPRE_MultiblockMatrix matrix )
{
   return ( hypre_MultiblockMatrixAssemble( (hypre_MultiblockMatrix *) matrix ) );
}



/*--------------------------------------------------------------------------
 * NALU_HYPRE_MultiblockMatrixPrint
 *--------------------------------------------------------------------------*/

void
NALU_HYPRE_MultiblockMatrixPrint( NALU_HYPRE_MultiblockMatrix matrix )
{
   hypre_MultiblockMatrixPrint( (hypre_MultiblockMatrix *) matrix );
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
   return ( hypre_MultiblockMatrixSetNumSubmatrices(
               (hypre_MultiblockMatrix *) matrix, n ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_MultiblockMatrixSetSubmatrixType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_MultiblockMatrixSetSubmatrixType( NALU_HYPRE_MultiblockMatrix matrix,
                                        NALU_HYPRE_Int j,
                                        NALU_HYPRE_Int type )
{
   return ( hypre_MultiblockMatrixSetSubmatrixType(
               (hypre_MultiblockMatrix *) matrix, j, type ) );
}
