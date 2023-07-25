/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef TEMPORARY_MULTIVECTOR_FUNCTION_PROTOTYPES
#define TEMPORARY_MULTIVECTOR_FUNCTION_PROTOTYPES

#include "interpreter.h"

typedef struct
{
   NALU_HYPRE_Int    numVectors;
   NALU_HYPRE_Int*   mask;
   void**       vector;
   NALU_HYPRE_Int    ownsVectors;
   NALU_HYPRE_Int    ownsMask;

   mv_InterfaceInterpreter* interpreter;

} mv_TempMultiVector;

/*typedef struct mv_TempMultiVector* mv_TempMultiVectorPtr;  */
typedef  mv_TempMultiVector* mv_TempMultiVectorPtr;

/*******************************************************************/
/*
The above is a temporary implementation of the nalu_hypre_MultiVector
data type, just to get things going with LOBPCG eigensolver.

A more proper implementation would be to define nalu_hypre_MultiParVector,
nalu_hypre_MultiStructVector and nalu_hypre_MultiSStructVector by adding a new
record

NALU_HYPRE_Int numVectors;

in nalu_hypre_ParVector, nalu_hypre_StructVector and nalu_hypre_SStructVector,
and increasing the size of data numVectors times. Respective
modifications of most vector operations are straightforward
(it is strongly suggested that BLAS routines are used wherever
possible), efficient implementation of matrix-by-multivector
multiplication may be more difficult.

With the above implementation of hypre vectors, the definition
of nalu_hypre_MultiVector becomes simply (cf. multivector.h)

typedef struct
{
  void* multiVector;
  NALU_HYPRE_InterfaceInterpreter* interpreter;
} nalu_hypre_MultiVector;

with pointers to abstract multivector functions added to the structure
NALU_HYPRE_InterfaceInterpreter (cf. NALU_HYPRE_interpreter.h; particular values
are assigned to these pointers by functions
NALU_HYPRE_ParCSRSetupInterpreter, NALU_HYPRE_StructSetupInterpreter and
NALU_HYPRE_Int NALU_HYPRE_SStructSetupInterpreter),
and the abstract multivector functions become simply interfaces
to the actual multivector functions of the form (cf. multivector.c):

void
nalu_hypre_MultiVectorCopy( nalu_hypre_MultiVectorPtr src_, nalu_hypre_MultiVectorPtr dest_ ) {

  nalu_hypre_MultiVector* src = (nalu_hypre_MultiVector*)src_;
  nalu_hypre_MultiVector* dest = (nalu_hypre_MultiVector*)dest_;
  nalu_hypre_assert( src != NULL && dest != NULL );
  (src->interpreter->CopyMultiVector)( src->data, dest->data );
}


*/
/*********************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

void*
mv_TempMultiVectorCreateFromSampleVector( void*, NALU_HYPRE_Int n, void* sample );

void*
mv_TempMultiVectorCreateCopy( void*, NALU_HYPRE_Int copyValues );

void
mv_TempMultiVectorDestroy( void* );

NALU_HYPRE_Int
mv_TempMultiVectorWidth( void* v );

NALU_HYPRE_Int
mv_TempMultiVectorHeight( void* v );

void
mv_TempMultiVectorSetMask( void* v, NALU_HYPRE_Int* mask );

void
mv_TempMultiVectorClear( void* );

void
mv_TempMultiVectorSetRandom( void* v, NALU_HYPRE_Int seed );

void
mv_TempMultiVectorCopy( void* src, void* dest );

void
mv_TempMultiVectorAxpy( NALU_HYPRE_Complex, void*, void* );

void
mv_TempMultiVectorByMultiVector( void*, void*,
                                 NALU_HYPRE_Int gh, NALU_HYPRE_Int h, NALU_HYPRE_Int w, NALU_HYPRE_Complex* v );

void
mv_TempMultiVectorByMultiVectorDiag( void* x, void* y,
                                     NALU_HYPRE_Int* mask, NALU_HYPRE_Int n, NALU_HYPRE_Complex* diag );

void
mv_TempMultiVectorByMatrix( void*,
                            NALU_HYPRE_Int gh, NALU_HYPRE_Int h, NALU_HYPRE_Int w, NALU_HYPRE_Complex* v,
                            void* );

void
mv_TempMultiVectorXapy( void* x,
                        NALU_HYPRE_Int gh, NALU_HYPRE_Int h, NALU_HYPRE_Int w, NALU_HYPRE_Complex* v,
                        void* y );

void mv_TempMultiVectorByDiagonal( void* x,
                                   NALU_HYPRE_Int* mask, NALU_HYPRE_Int n, NALU_HYPRE_Complex* diag,
                                   void* y );

void
mv_TempMultiVectorEval( void (*f)( void*, void*, void* ), void* par,
                        void* x, void* y );

#ifdef __cplusplus
}
#endif

#endif /* MULTIVECTOR_FUNCTION_PROTOTYPES */

