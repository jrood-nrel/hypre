/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef LOBPCG_INTERFACE_INTERPRETER
#define LOBPCG_INTERFACE_INTERPRETER

#include "NALU_HYPRE_utilities.h"

typedef struct
{
   /* vector operations */
   void*  (*CreateVector)  ( void *vector );
   NALU_HYPRE_Int    (*DestroyVector) ( void *vector );

   NALU_HYPRE_Real   (*InnerProd)     ( void *x, void *y );
   NALU_HYPRE_Int    (*CopyVector)    ( void *x, void *y );
   NALU_HYPRE_Int    (*ClearVector)   ( void *x );
   NALU_HYPRE_Int    (*SetRandomValues)   ( void *x, NALU_HYPRE_Int seed );
   NALU_HYPRE_Int    (*ScaleVector)   ( NALU_HYPRE_Complex alpha, void *x );
   NALU_HYPRE_Int    (*Axpy)          ( NALU_HYPRE_Complex alpha, void *x, void *y );
   NALU_HYPRE_Int    (*VectorSize)    (void * vector);

   /* multivector operations */
   /* do we need the following entry? */
   void*  (*CreateMultiVector)  ( void*, NALU_HYPRE_Int n, void *vector );
   void*  (*CopyCreateMultiVector)  ( void *x, NALU_HYPRE_Int );
   void    (*DestroyMultiVector) ( void *x );

   NALU_HYPRE_Int    (*Width)  ( void *x );
   NALU_HYPRE_Int    (*Height) ( void *x );

   void   (*SetMask) ( void *x, NALU_HYPRE_Int *mask );

   void   (*CopyMultiVector)    ( void *x, void *y );
   void   (*ClearMultiVector)   ( void *x );
   void   (*SetRandomVectors)   ( void *x, NALU_HYPRE_Int seed );
   void   (*MultiInnerProd)     ( void *x, void *y, NALU_HYPRE_Int, NALU_HYPRE_Int, NALU_HYPRE_Int, NALU_HYPRE_Real* );
   void   (*MultiInnerProdDiag) ( void *x, void *y, NALU_HYPRE_Int*, NALU_HYPRE_Int, NALU_HYPRE_Real* );
   void   (*MultiVecMat)        ( void *x, NALU_HYPRE_Int, NALU_HYPRE_Int, NALU_HYPRE_Int, NALU_HYPRE_Complex*, void *y );
   void   (*MultiVecMatDiag)    ( void *x, NALU_HYPRE_Int*, NALU_HYPRE_Int, NALU_HYPRE_Complex*, void *y );
   void   (*MultiAxpy)          ( NALU_HYPRE_Complex alpha, void *x, void *y );

   /* do we need the following 2 entries? */
   void   (*MultiXapy)          ( void *x, NALU_HYPRE_Int, NALU_HYPRE_Int, NALU_HYPRE_Int, NALU_HYPRE_Complex*, void *y );
   void   (*Eval)               ( void (*f)( void*, void*, void* ), void*, void *x, void *y );

} mv_InterfaceInterpreter;

#endif
