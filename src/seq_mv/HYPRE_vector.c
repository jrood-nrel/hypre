/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * NALU_HYPRE_Vector interface
 *
 *****************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * NALU_HYPRE_VectorCreate
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Vector
NALU_HYPRE_VectorCreate( NALU_HYPRE_Int size )
{
   return ( (NALU_HYPRE_Vector) nalu_hypre_SeqVectorCreate(size) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_VectorDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_VectorDestroy( NALU_HYPRE_Vector vector )
{
   return ( nalu_hypre_SeqVectorDestroy( (nalu_hypre_Vector *) vector ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_VectorInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_VectorInitialize( NALU_HYPRE_Vector vector )
{
   return ( nalu_hypre_SeqVectorInitialize( (nalu_hypre_Vector *) vector ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_VectorPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_VectorPrint( NALU_HYPRE_Vector  vector,
                   char         *file_name )
{
   return ( nalu_hypre_SeqVectorPrint( (nalu_hypre_Vector *) vector,
                                  file_name ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_VectorRead
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Vector
NALU_HYPRE_VectorRead( char         *file_name )
{
   return ( (NALU_HYPRE_Vector) nalu_hypre_SeqVectorRead( file_name ) );
}
