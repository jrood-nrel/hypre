/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * NALU_HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_nalu_hypre_parcsr_ls.h"
#include "interpreter.h"
#include "NALU_HYPRE_MatvecFunctions.h"
#include "temp_multivector.h"

NALU_HYPRE_Int
nalu_hypre_ParSetRandomValues( void* v, NALU_HYPRE_Int seed )
{

   NALU_HYPRE_ParVectorSetRandomValues( (NALU_HYPRE_ParVector)v, seed );
   return 0;
}

NALU_HYPRE_Int
nalu_hypre_ParPrintVector( void* v, const char* file )
{

   return nalu_hypre_ParVectorPrint( (nalu_hypre_ParVector*)v, file );
}

void*
nalu_hypre_ParReadVector( MPI_Comm comm, const char* file )
{

   return (void*)nalu_hypre_ParVectorRead( comm, file );
}

NALU_HYPRE_Int nalu_hypre_ParVectorSize(void * x)
{
   return 0;
}

NALU_HYPRE_Int
NALU_HYPRE_ParCSRMultiVectorPrint( void* x_, const char* fileName )
{

   NALU_HYPRE_Int i, ierr;
   mv_TempMultiVector* x;
   char fullName[128];

   x = (mv_TempMultiVector*)x_;
   nalu_hypre_assert( x != NULL );

   ierr = 0;
   for ( i = 0; i < x->numVectors; i++ )
   {
      nalu_hypre_sprintf( fullName, "%s.%d", fileName, i );
      ierr = ierr ||
             nalu_hypre_ParPrintVector( x->vector[i], fullName );
   }
   return ierr;
}

void*
NALU_HYPRE_ParCSRMultiVectorRead( MPI_Comm comm, void* ii_, const char* fileName )
{

   NALU_HYPRE_Int i, n, id;
   FILE* fp;
   char fullName[128];
   mv_TempMultiVector* x;
   mv_InterfaceInterpreter* ii = (mv_InterfaceInterpreter*)ii_;

   nalu_hypre_MPI_Comm_rank( comm, &id );

   n = 0;
   do
   {
      nalu_hypre_sprintf( fullName, "%s.%d.%d", fileName, n, id );
      if ( (fp = fopen(fullName, "r")) )
      {
         n++;
         fclose( fp );
      }
   }
   while ( fp );

   if ( n == 0 )
   {
      return NULL;
   }

   x = nalu_hypre_TAlloc(mv_TempMultiVector, 1, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( x != NULL );

   x->interpreter = ii;

   x->numVectors = n;

   x->vector = nalu_hypre_CTAlloc(void*,  n, NALU_HYPRE_MEMORY_HOST);
   nalu_hypre_assert( x->vector != NULL );

   x->ownsVectors = 1;

   for ( i = 0; i < n; i++ )
   {
      nalu_hypre_sprintf( fullName, "%s.%d", fileName, i );
      x->vector[i] = nalu_hypre_ParReadVector( comm, fullName );
   }

   x->mask = NULL;
   x->ownsMask = 0;

   return x;
}

NALU_HYPRE_Int
aux_maskCount( NALU_HYPRE_Int n, NALU_HYPRE_Int* mask )
{

   NALU_HYPRE_Int i, m;

   if ( mask == NULL )
   {
      return n;
   }

   for ( i = m = 0; i < n; i++ )
      if ( mask[i] )
      {
         m++;
      }

   return m;
}

void
aux_indexFromMask( NALU_HYPRE_Int n, NALU_HYPRE_Int* mask, NALU_HYPRE_Int* index )
{

   NALU_HYPRE_Int i, j;

   if ( mask != NULL )
   {
      for ( i = 0, j = 0; i < n; i++ )
         if ( mask[i] )
         {
            index[j++] = i + 1;
         }
   }
   else
      for ( i = 0; i < n; i++ )
      {
         index[i] = i + 1;
      }

}


/* The function below is a temporary one that fills the multivector
   part of the NALU_HYPRE_InterfaceInterpreter structure with pointers
   that come from the temporary implementation of the multivector
   (cf. temp_multivector.h).
   It must be eventually replaced with a function that
   provides the respective pointers to properly implemented
   parcsr multivector functions */

NALU_HYPRE_Int
NALU_HYPRE_TempParCSRSetupInterpreter( mv_InterfaceInterpreter *i )
{
   /* Vector part */

   i->CreateVector = nalu_hypre_ParKrylovCreateVector;
   i->DestroyVector = nalu_hypre_ParKrylovDestroyVector;
   i->InnerProd = nalu_hypre_ParKrylovInnerProd;
   i->CopyVector = nalu_hypre_ParKrylovCopyVector;
   i->ClearVector = nalu_hypre_ParKrylovClearVector;
   i->SetRandomValues = nalu_hypre_ParSetRandomValues;
   i->ScaleVector = nalu_hypre_ParKrylovScaleVector;
   i->Axpy = nalu_hypre_ParKrylovAxpy;

   /* Multivector part */

   i->CreateMultiVector = mv_TempMultiVectorCreateFromSampleVector;
   i->CopyCreateMultiVector = mv_TempMultiVectorCreateCopy;
   i->DestroyMultiVector = mv_TempMultiVectorDestroy;

   i->Width = mv_TempMultiVectorWidth;
   i->Height = mv_TempMultiVectorHeight;
   i->SetMask = mv_TempMultiVectorSetMask;
   i->CopyMultiVector = mv_TempMultiVectorCopy;
   i->ClearMultiVector = mv_TempMultiVectorClear;
   i->SetRandomVectors = mv_TempMultiVectorSetRandom;
   i->MultiInnerProd = mv_TempMultiVectorByMultiVector;
   i->MultiInnerProdDiag = mv_TempMultiVectorByMultiVectorDiag;
   i->MultiVecMat = mv_TempMultiVectorByMatrix;
   i->MultiVecMatDiag = mv_TempMultiVectorByDiagonal;
   i->MultiAxpy = mv_TempMultiVectorAxpy;
   i->MultiXapy = mv_TempMultiVectorXapy;
   i->Eval = mv_TempMultiVectorEval;

   return 0;
}

NALU_HYPRE_Int
NALU_HYPRE_ParCSRSetupInterpreter( mv_InterfaceInterpreter *i )
{
   return NALU_HYPRE_TempParCSRSetupInterpreter( i );
}

NALU_HYPRE_Int
NALU_HYPRE_ParCSRSetupMatvec(NALU_HYPRE_MatvecFunctions * mv)
{
   mv->MatvecCreate = nalu_hypre_ParKrylovMatvecCreate;
   mv->Matvec = nalu_hypre_ParKrylovMatvec;
   mv->MatvecDestroy = nalu_hypre_ParKrylovMatvecDestroy;

   mv->MatMultiVecCreate = NULL;
   mv->MatMultiVec = NULL;
   mv->MatMultiVecDestroy = NULL;

   return 0;
}
