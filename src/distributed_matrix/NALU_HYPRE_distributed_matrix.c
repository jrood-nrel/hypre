/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_DistributedMatrix interface
 *
 *****************************************************************************/

#include "./distributed_matrix.h"


/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
NALU_HYPRE_DistributedMatrixCreate( MPI_Comm context, NALU_HYPRE_DistributedMatrix *matrix )
{
   NALU_HYPRE_Int ierr = 0;

   *matrix = (NALU_HYPRE_DistributedMatrix)
	    hypre_DistributedMatrixCreate( context );

   return ( ierr );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
NALU_HYPRE_DistributedMatrixDestroy( NALU_HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixDestroy( (hypre_DistributedMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
NALU_HYPRE_DistributedMatrixLimitedDestroy( NALU_HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixLimitedDestroy( (hypre_DistributedMatrix *) matrix ) );
}


/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
NALU_HYPRE_DistributedMatrixInitialize( NALU_HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixInitialize( (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixAssemble
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
NALU_HYPRE_DistributedMatrixAssemble( NALU_HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixAssemble( (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixSetLocalStorageType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_DistributedMatrixSetLocalStorageType( NALU_HYPRE_DistributedMatrix matrix,
				 NALU_HYPRE_Int               type           )
{
   return( hypre_DistributedMatrixSetLocalStorageType(
      (hypre_DistributedMatrix *) matrix, type ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixGetLocalStorageType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_DistributedMatrixGetLocalStorageType( NALU_HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixGetLocalStorageType(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixSetLocalStorage
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_DistributedMatrixSetLocalStorage( NALU_HYPRE_DistributedMatrix matrix,
				      void                 *LocalStorage )
{
   return( hypre_DistributedMatrixSetLocalStorage(
      (hypre_DistributedMatrix *) matrix, LocalStorage ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixGetLocalStorage
 *--------------------------------------------------------------------------*/

void *
NALU_HYPRE_DistributedMatrixGetLocalStorage( NALU_HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixGetLocalStorage(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixSetTranslator
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_DistributedMatrixSetTranslator( NALU_HYPRE_DistributedMatrix matrix,
				      void                 *Translator )
{
   return( hypre_DistributedMatrixSetTranslator(
      (hypre_DistributedMatrix *) matrix, Translator ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixGetTranslator
 *--------------------------------------------------------------------------*/

void *
NALU_HYPRE_DistributedMatrixGetTranslator( NALU_HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixGetTranslator(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixSetAuxiliaryData
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_DistributedMatrixSetAuxiliaryData( NALU_HYPRE_DistributedMatrix matrix,
				      void                 *AuxiliaryData )
{
   return( hypre_DistributedMatrixSetAuxiliaryData(
      (hypre_DistributedMatrix *) matrix, AuxiliaryData ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixGetAuxiliaryData
 *--------------------------------------------------------------------------*/

void *
NALU_HYPRE_DistributedMatrixGetAuxiliaryData( NALU_HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixAuxiliaryData(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixGetContext
 *--------------------------------------------------------------------------*/

MPI_Comm
NALU_HYPRE_DistributedMatrixGetContext( NALU_HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixContext(
      (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixGetDims
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_DistributedMatrixGetDims( NALU_HYPRE_DistributedMatrix matrix, 
                               NALU_HYPRE_BigInt *M, NALU_HYPRE_BigInt *N )
{
   NALU_HYPRE_Int ierr=0;

   *M = hypre_DistributedMatrixM( (hypre_DistributedMatrix *) matrix );
   *N = hypre_DistributedMatrixN( (hypre_DistributedMatrix *) matrix );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixSetDims
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_DistributedMatrixSetDims( NALU_HYPRE_DistributedMatrix matrix, 
                               NALU_HYPRE_BigInt M, NALU_HYPRE_BigInt N )
{
   NALU_HYPRE_Int ierr=0;

   hypre_DistributedMatrixM( (hypre_DistributedMatrix *) matrix ) = M;
   hypre_DistributedMatrixN( (hypre_DistributedMatrix *) matrix ) = N;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * Optional routines that depend on underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
NALU_HYPRE_DistributedMatrixPrint( NALU_HYPRE_DistributedMatrix matrix )
{
   return( hypre_DistributedMatrixPrint( (hypre_DistributedMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_DistributedMatrixGetLocalRange( NALU_HYPRE_DistributedMatrix matrix, 
                               NALU_HYPRE_BigInt *row_start, NALU_HYPRE_BigInt *row_end ,
                               NALU_HYPRE_BigInt *col_start, NALU_HYPRE_BigInt *col_end )
{
   return( hypre_DistributedMatrixGetLocalRange( (hypre_DistributedMatrix *) matrix,
                             row_start, row_end, col_start, col_end ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixGetRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
NALU_HYPRE_DistributedMatrixGetRow( NALU_HYPRE_DistributedMatrix matrix,
                             NALU_HYPRE_BigInt row,
                             NALU_HYPRE_Int *size,
                             NALU_HYPRE_BigInt **col_ind,
                             NALU_HYPRE_Real **values )
{
   return( hypre_DistributedMatrixGetRow( (hypre_DistributedMatrix *) matrix,
                             row,
                             size,
                             col_ind,
                             values ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_DistributedMatrixRestoreRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
NALU_HYPRE_DistributedMatrixRestoreRow( NALU_HYPRE_DistributedMatrix matrix,
                             NALU_HYPRE_BigInt row,
                             NALU_HYPRE_Int *size,
                             NALU_HYPRE_BigInt **col_ind,
                             NALU_HYPRE_Real **values )
{
   return( hypre_DistributedMatrixRestoreRow( (hypre_DistributedMatrix *) matrix,
                             row,
                             size,
                             col_ind,
                             values ) );
}
