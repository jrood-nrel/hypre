/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for nalu_hypre_DistributedMatrix class.
 *
 *****************************************************************************/

#include "distributed_matrix.h"
#include "NALU_HYPRE.h"

/*--------------------------------------------------------------------------
 *     BASIC CONSTRUCTION/DESTRUCTION SEQUENCE
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixCreate
 *--------------------------------------------------------------------------*/

nalu_hypre_DistributedMatrix *
nalu_hypre_DistributedMatrixCreate( MPI_Comm     context  )
{
   nalu_hypre_DistributedMatrix    *matrix;

   matrix = nalu_hypre_CTAlloc(nalu_hypre_DistributedMatrix,  1, NALU_HYPRE_MEMORY_HOST);

   nalu_hypre_DistributedMatrixContext(matrix) = context;
   nalu_hypre_DistributedMatrixM(matrix)    = -1;
   nalu_hypre_DistributedMatrixN(matrix)    = -1;
   nalu_hypre_DistributedMatrixAuxiliaryData(matrix)    = NULL;
   nalu_hypre_DistributedMatrixLocalStorage(matrix) = NULL;
   nalu_hypre_DistributedMatrixTranslator(matrix) = NULL;
   nalu_hypre_DistributedMatrixLocalStorageType(matrix) = NALU_HYPRE_UNITIALIZED;

#ifdef NALU_HYPRE_TIMING
   matrix->GetRow_timer = nalu_hypre_InitializeTiming( "GetRow" );
#endif

   return matrix;
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixDestroy( nalu_hypre_DistributedMatrix *matrix )
{

   if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PETSC )
      nalu_hypre_DistributedMatrixDestroyPETSc( matrix );
   else if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_ISIS )
      nalu_hypre_FreeDistributedMatrixISIS( matrix );
   else if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PARCSR )
      nalu_hypre_DistributedMatrixDestroyParCSR( matrix );
   else
      return(-1);

#ifdef NALU_HYPRE_TIMING
   nalu_hypre_FinalizeTiming ( matrix->GetRow_timer );
#endif
   nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);

   return(0);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixLimitedDestroy( nalu_hypre_DistributedMatrix *matrix )
{

   nalu_hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);

   return(0);
}


/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixInitialize( nalu_hypre_DistributedMatrix *matrix )
{
   NALU_HYPRE_Int ierr = 0;

   if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PETSC )
      return( 0 );
   else if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_ISIS )
      ierr = nalu_hypre_InitializeDistributedMatrixISIS(matrix);
   else if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PARCSR )
      ierr = nalu_hypre_DistributedMatrixInitializeParCSR(matrix);
   else
      ierr = -1;

   return( ierr );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixAssemble
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixAssemble( nalu_hypre_DistributedMatrix *matrix )
{

   if( 
       (nalu_hypre_DistributedMatrixLocalStorageType(matrix) != NALU_HYPRE_PETSC )
    && (nalu_hypre_DistributedMatrixLocalStorageType(matrix) != NALU_HYPRE_ISIS )
    && (nalu_hypre_DistributedMatrixLocalStorageType(matrix) != NALU_HYPRE_PARCSR )
     )
     return(-1);


   if( nalu_hypre_DistributedMatrixLocalStorage(matrix) == NULL )
     return(-1);

   if( (nalu_hypre_DistributedMatrixM(matrix) < 0 ) ||
       (nalu_hypre_DistributedMatrixN(matrix) < 0 ) )
     return(-1);

   return(0);
}

/*--------------------------------------------------------------------------
 *     Get/Sets that are independent of underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixSetLocalStorageType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixSetLocalStorageType( nalu_hypre_DistributedMatrix *matrix,
				 NALU_HYPRE_Int                type   )
{
   NALU_HYPRE_Int ierr=0;

   nalu_hypre_DistributedMatrixLocalStorageType(matrix) = type;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixGetLocalStorageType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixGetLocalStorageType( nalu_hypre_DistributedMatrix *matrix  )
{
   NALU_HYPRE_Int ierr=0;

   ierr = nalu_hypre_DistributedMatrixLocalStorageType(matrix);

   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixSetLocalStorage
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixSetLocalStorage( nalu_hypre_DistributedMatrix *matrix,
				 void                  *local_storage  )
{
   NALU_HYPRE_Int ierr=0;

   nalu_hypre_DistributedMatrixLocalStorage(matrix) = local_storage;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixGetLocalStorage
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_DistributedMatrixGetLocalStorage( nalu_hypre_DistributedMatrix *matrix  )
{
   return( nalu_hypre_DistributedMatrixLocalStorage(matrix) );

}


/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixSetTranslator
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixSetTranslator( nalu_hypre_DistributedMatrix *matrix,
				 void                  *translator  )
{
   nalu_hypre_DistributedMatrixTranslator(matrix) = translator;

   return(0);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixGetTranslator
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_DistributedMatrixGetTranslator( nalu_hypre_DistributedMatrix *matrix  )
{
   return( nalu_hypre_DistributedMatrixTranslator(matrix) );

}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixSetAuxiliaryData
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixSetAuxiliaryData( nalu_hypre_DistributedMatrix *matrix,
				 void                  *auxiliary_data  )
{
   nalu_hypre_DistributedMatrixAuxiliaryData(matrix) = auxiliary_data;

   return(0);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixGetAuxiliaryData
 *--------------------------------------------------------------------------*/

void *
nalu_hypre_DistributedMatrixGetAuxiliaryData( nalu_hypre_DistributedMatrix *matrix  )
{
   return( nalu_hypre_DistributedMatrixAuxiliaryData(matrix) );

}

/*--------------------------------------------------------------------------
 * Optional routines that depend on underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixPrint( nalu_hypre_DistributedMatrix *matrix )
{
   if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PETSC )
      return( nalu_hypre_DistributedMatrixPrintPETSc( matrix ) );
   else if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_ISIS )
      return( nalu_hypre_PrintDistributedMatrixISIS( matrix ) );
   else if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PARCSR )
      return( nalu_hypre_DistributedMatrixPrintParCSR( matrix ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixGetLocalRange( nalu_hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt *row_start,
                             NALU_HYPRE_BigInt *row_end,
                             NALU_HYPRE_BigInt *col_start,
                             NALU_HYPRE_BigInt *col_end )
{
   if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PETSC )
      return( nalu_hypre_DistributedMatrixGetLocalRangePETSc( matrix, row_start, row_end ) );
   else if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_ISIS )
      return( nalu_hypre_GetDistributedMatrixLocalRangeISIS( matrix, row_start, row_end ) );
   else if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PARCSR )
      return( nalu_hypre_DistributedMatrixGetLocalRangeParCSR( matrix, row_start, row_end, col_start, col_end ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixGetRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixGetRow( nalu_hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt row,
                             NALU_HYPRE_Int *size,
                             NALU_HYPRE_BigInt **col_ind,
                             NALU_HYPRE_Real **values )
{
   NALU_HYPRE_Int ierr = 0;

#ifdef NALU_HYPRE_TIMING
   nalu_hypre_BeginTiming( matrix->GetRow_timer );
#endif

   if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PETSC ) {
      ierr = nalu_hypre_DistributedMatrixGetRowPETSc( matrix, row, size, col_ind, values );
   }
   else if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_ISIS ) {
      ierr = nalu_hypre_GetDistributedMatrixRowISIS( matrix, row, size, col_ind, values );
   }
   else if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PARCSR ) {
      ierr = nalu_hypre_DistributedMatrixGetRowParCSR( matrix, row, size, col_ind, values );
   }
   else
      ierr = -1;

#ifdef NALU_HYPRE_TIMING
   nalu_hypre_EndTiming( matrix->GetRow_timer );
#endif

   return( ierr );
}

/*--------------------------------------------------------------------------
 * nalu_hypre_DistributedMatrixRestoreRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
nalu_hypre_DistributedMatrixRestoreRow( nalu_hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt row,
                             NALU_HYPRE_Int *size,
                             NALU_HYPRE_BigInt **col_ind,
                             NALU_HYPRE_Real **values )
{
   NALU_HYPRE_Int ierr = 0;

#ifdef NALU_HYPRE_TIMING
   nalu_hypre_BeginTiming( matrix->GetRow_timer );
#endif

   if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PETSC )
      ierr = nalu_hypre_DistributedMatrixRestoreRowPETSc( matrix, row, size, col_ind, values );
   else if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_ISIS )
      ierr = nalu_hypre_RestoreDistributedMatrixRowISIS( matrix, row, size, col_ind, values );
   else if ( nalu_hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PARCSR )
      ierr = nalu_hypre_DistributedMatrixRestoreRowParCSR( matrix, row, size, col_ind, values );
   else
      ierr = -1;

#ifdef NALU_HYPRE_TIMING
   nalu_hypre_EndTiming( matrix->GetRow_timer );
#endif

   return( ierr );
}
