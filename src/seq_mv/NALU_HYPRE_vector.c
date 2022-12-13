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
   return ( (NALU_HYPRE_Vector) hypre_SeqVectorCreate(size) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_VectorDestroy
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_VectorDestroy( NALU_HYPRE_Vector vector )
{
   return ( hypre_SeqVectorDestroy( (hypre_Vector *) vector ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_VectorInitialize
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_VectorInitialize( NALU_HYPRE_Vector vector )
{
   return ( hypre_SeqVectorInitialize( (hypre_Vector *) vector ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_VectorPrint
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Int
NALU_HYPRE_VectorPrint( NALU_HYPRE_Vector  vector,
                   char         *file_name )
{
   return ( hypre_SeqVectorPrint( (hypre_Vector *) vector,
                                  file_name ) );
}

/*--------------------------------------------------------------------------
 * NALU_HYPRE_VectorRead
 *--------------------------------------------------------------------------*/

NALU_HYPRE_Vector
NALU_HYPRE_VectorRead( char         *file_name )
{
   return ( (NALU_HYPRE_Vector) hypre_SeqVectorRead( file_name ) );
}
