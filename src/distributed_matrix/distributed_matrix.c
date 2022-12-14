/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_DistributedMatrix class.
 *
 *****************************************************************************/

#include "distributed_matrix.h"
#include "NALU_HYPRE.h"

/*--------------------------------------------------------------------------
 *     BASIC CONSTRUCTION/DESTRUCTION SEQUENCE
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixCreate
 *--------------------------------------------------------------------------*/

hypre_DistributedMatrix *
hypre_DistributedMatrixCreate( MPI_Comm     context  )
{
   hypre_DistributedMatrix    *matrix;

   matrix = hypre_CTAlloc(hypre_DistributedMatrix,  1, NALU_HYPRE_MEMORY_HOST);

   hypre_DistributedMatrixContext(matrix) = context;
   hypre_DistributedMatrixM(matrix)    = -1;
   hypre_DistributedMatrixN(matrix)    = -1;
   hypre_DistributedMatrixAuxiliaryData(matrix)    = NULL;
   hypre_DistributedMatrixLocalStorage(matrix) = NULL;
   hypre_DistributedMatrixTranslator(matrix) = NULL;
   hypre_DistributedMatrixLocalStorageType(matrix) = NALU_HYPRE_UNITIALIZED;

#ifdef NALU_HYPRE_TIMING
   matrix->GetRow_timer = hypre_InitializeTiming( "GetRow" );
#endif

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixDestroy( hypre_DistributedMatrix *matrix )
{

   if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PETSC )
      hypre_DistributedMatrixDestroyPETSc( matrix );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_ISIS )
      hypre_FreeDistributedMatrixISIS( matrix );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PARCSR )
      hypre_DistributedMatrixDestroyParCSR( matrix );
   else
      return(-1);

#ifdef NALU_HYPRE_TIMING
   hypre_FinalizeTiming ( matrix->GetRow_timer );
#endif
   hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixLimitedDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixLimitedDestroy( hypre_DistributedMatrix *matrix )
{

   hypre_TFree(matrix, NALU_HYPRE_MEMORY_HOST);

   return(0);
}


/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixInitialize( hypre_DistributedMatrix *matrix )
{
   NALU_HYPRE_Int ierr = 0;

   if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PETSC )
      return( 0 );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_ISIS )
      ierr = hypre_InitializeDistributedMatrixISIS(matrix);
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PARCSR )
      ierr = hypre_DistributedMatrixInitializeParCSR(matrix);
   else
      ierr = -1;

   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixAssemble
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixAssemble( hypre_DistributedMatrix *matrix )
{

   if( 
       (hypre_DistributedMatrixLocalStorageType(matrix) != NALU_HYPRE_PETSC )
    && (hypre_DistributedMatrixLocalStorageType(matrix) != NALU_HYPRE_ISIS )
    && (hypre_DistributedMatrixLocalStorageType(matrix) != NALU_HYPRE_PARCSR )
     )
     return(-1);


   if( hypre_DistributedMatrixLocalStorage(matrix) == NULL )
     return(-1);

   if( (hypre_DistributedMatrixM(matrix) < 0 ) ||
       (hypre_DistributedMatrixN(matrix) < 0 ) )
     return(-1);

   return(0);
}

/*--------------------------------------------------------------------------
 *     Get/Sets that are independent of underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixSetLocalStorageType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixSetLocalStorageType( hypre_DistributedMatrix *matrix,
				 NALU_HYPRE_Int                type   )
{
   NALU_HYPRE_Int ierr=0;

   hypre_DistributedMatrixLocalStorageType(matrix) = type;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetLocalStorageType
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixGetLocalStorageType( hypre_DistributedMatrix *matrix  )
{
   NALU_HYPRE_Int ierr=0;

   ierr = hypre_DistributedMatrixLocalStorageType(matrix);

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixSetLocalStorage
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixSetLocalStorage( hypre_DistributedMatrix *matrix,
				 void                  *local_storage  )
{
   NALU_HYPRE_Int ierr=0;

   hypre_DistributedMatrixLocalStorage(matrix) = local_storage;

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetLocalStorage
 *--------------------------------------------------------------------------*/

void *
hypre_DistributedMatrixGetLocalStorage( hypre_DistributedMatrix *matrix  )
{
   return( hypre_DistributedMatrixLocalStorage(matrix) );

}


/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixSetTranslator
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixSetTranslator( hypre_DistributedMatrix *matrix,
				 void                  *translator  )
{
   hypre_DistributedMatrixTranslator(matrix) = translator;

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetTranslator
 *--------------------------------------------------------------------------*/

void *
hypre_DistributedMatrixGetTranslator( hypre_DistributedMatrix *matrix  )
{
   return( hypre_DistributedMatrixTranslator(matrix) );

}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixSetAuxiliaryData
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixSetAuxiliaryData( hypre_DistributedMatrix *matrix,
				 void                  *auxiliary_data  )
{
   hypre_DistributedMatrixAuxiliaryData(matrix) = auxiliary_data;

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetAuxiliaryData
 *--------------------------------------------------------------------------*/

void *
hypre_DistributedMatrixGetAuxiliaryData( hypre_DistributedMatrix *matrix  )
{
   return( hypre_DistributedMatrixAuxiliaryData(matrix) );

}

/*--------------------------------------------------------------------------
 * Optional routines that depend on underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixPrint( hypre_DistributedMatrix *matrix )
{
   if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PETSC )
      return( hypre_DistributedMatrixPrintPETSc( matrix ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_ISIS )
      return( hypre_PrintDistributedMatrixISIS( matrix ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PARCSR )
      return( hypre_DistributedMatrixPrintParCSR( matrix ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetLocalRange
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixGetLocalRange( hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt *row_start,
                             NALU_HYPRE_BigInt *row_end,
                             NALU_HYPRE_BigInt *col_start,
                             NALU_HYPRE_BigInt *col_end )
{
   if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PETSC )
      return( hypre_DistributedMatrixGetLocalRangePETSc( matrix, row_start, row_end ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_ISIS )
      return( hypre_GetDistributedMatrixLocalRangeISIS( matrix, row_start, row_end ) );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PARCSR )
      return( hypre_DistributedMatrixGetLocalRangeParCSR( matrix, row_start, row_end, col_start, col_end ) );
   else
      return(-1);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixGetRow( hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt row,
                             NALU_HYPRE_Int *size,
                             NALU_HYPRE_BigInt **col_ind,
                             NALU_HYPRE_Real **values )
{
   NALU_HYPRE_Int ierr = 0;

#ifdef NALU_HYPRE_TIMING
   hypre_BeginTiming( matrix->GetRow_timer );
#endif

   if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PETSC ) {
      ierr = hypre_DistributedMatrixGetRowPETSc( matrix, row, size, col_ind, values );
   }
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_ISIS ) {
      ierr = hypre_GetDistributedMatrixRowISIS( matrix, row, size, col_ind, values );
   }
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PARCSR ) {
      ierr = hypre_DistributedMatrixGetRowParCSR( matrix, row, size, col_ind, values );
   }
   else
      ierr = -1;

#ifdef NALU_HYPRE_TIMING
   hypre_EndTiming( matrix->GetRow_timer );
#endif

   return( ierr );
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixRestoreRow
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int 
hypre_DistributedMatrixRestoreRow( hypre_DistributedMatrix *matrix,
                             NALU_HYPRE_BigInt row,
                             NALU_HYPRE_Int *size,
                             NALU_HYPRE_BigInt **col_ind,
                             NALU_HYPRE_Real **values )
{
   NALU_HYPRE_Int ierr = 0;

#ifdef NALU_HYPRE_TIMING
   hypre_BeginTiming( matrix->GetRow_timer );
#endif

   if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PETSC )
      ierr = hypre_DistributedMatrixRestoreRowPETSc( matrix, row, size, col_ind, values );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_ISIS )
      ierr = hypre_RestoreDistributedMatrixRowISIS( matrix, row, size, col_ind, values );
   else if ( hypre_DistributedMatrixLocalStorageType(matrix) == NALU_HYPRE_PARCSR )
      ierr = hypre_DistributedMatrixRestoreRowParCSR( matrix, row, size, col_ind, values );
   else
      ierr = -1;

#ifdef NALU_HYPRE_TIMING
   hypre_EndTiming( matrix->GetRow_timer );
#endif

   return( ierr );
}
